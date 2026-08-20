import cv2
import numpy as np
import pandas as pd

from attention_alignment.behavior import (
    EyeLandmarkTracker,
    PolygonROI,
    ROI,
    apply_pupil_quality_control,
    detect_pupil,
    extract_behavior,
    interpolate_short_nan_gaps,
    load_roi_config,
    pupil_audit_frame_indices,
    pupil_prior_from_polygon,
    pupil_result_from_polygon,
    preview_saved_pupil_detections,
    representative_invalid_frame_indices,
    representative_pupil_review_frame_indices,
    region_from_mapping,
    region_to_mapping,
    save_camera_annotation,
    save_pupil_manual_anchor,
)
from attention_alignment.models import TimeTransform


def test_detect_dark_pupil_ellipse():
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    cv2.ellipse(frame, (80, 60), (18, 12), 0, 0, 360, (10, 10, 10), -1)
    result = detect_pupil(frame, ROI(40, 30, 80, 60), threshold=80)
    assert result["pupil_valid"]
    assert abs(result["pupil_center_x"] - 80) < 2
    assert abs(result["pupil_center_y"] - 60) < 2
    assert result["pupil_major_axis"] > result["pupil_minor_axis"]


def test_polygon_detection_preserves_boundary_and_area():
    frame = np.full((140, 180, 3), 180, dtype=np.uint8)
    pupil = np.asarray([[70, 50], [95, 45], [110, 62], [100, 83], [73, 87], [58, 67]])
    cv2.fillPoly(frame, [pupil.astype(np.int32)], (12, 12, 12))
    eye = PolygonROI(((45, 35), (125, 35), (130, 100), (40, 100)))

    result = detect_pupil(frame, eye, threshold=80, boundary_points=16)

    assert result["pupil_valid"]
    assert abs(result["pupil_area"] - cv2.contourArea(pupil.astype(np.float32))) < 40
    assert len(result["pupil_boundary_x"]) == 16
    assert len(result["pupil_boundary_y"]) == 16
    assert result["pupil_mask_area_px"] > result["pupil_area"]
    assert result["pupil_hull_correction_fraction"] < 0.03


def test_reflection_notch_is_closed_by_convex_measurement_boundary():
    frame = np.full((140, 180, 3), 180, dtype=np.uint8)
    pupil_with_notch = np.asarray(
        [
            [55, 48],
            [105, 48],
            [105, 82],
            [88, 82],
            [88, 63],
            [73, 63],
            [73, 82],
            [55, 82],
        ]
    )
    cv2.fillPoly(frame, [pupil_with_notch.astype(np.int32)], (12, 12, 12))

    result = detect_pupil(frame, ROI(40, 35, 80, 65), threshold=80, boundary_points=24)

    boundary = np.column_stack([result["pupil_boundary_x"], result["pupil_boundary_y"]])
    edges = np.roll(boundary, -1, axis=0) - boundary
    next_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    non_collinear_turns = turns[np.abs(turns) > 1e-5]
    assert result["pupil_valid"]
    assert np.all(non_collinear_turns >= 0) or np.all(non_collinear_turns <= 0)
    assert result["pupil_area"] > result["pupil_raw_contour_area"]
    assert result["pupil_hull_correction_fraction"] > 0.1
    assert result["pupil_perimeter_px"] < result["pupil_raw_perimeter_px"]


def test_bright_reflection_split_is_rejoined_by_multiscale_candidate():
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    cv2.ellipse(frame, (80, 60), (20, 14), 0, 0, 360, (15, 15, 15), -1)
    cv2.rectangle(frame, (77, 43), (83, 77), (220, 220, 220), -1)
    reference = {
        "pupil_valid": True,
        "pupil_center_x": 80.0,
        "pupil_center_y": 60.0,
        "pupil_area": float(np.pi * 20 * 14),
    }

    result = detect_pupil(
        frame,
        ROI(40, 30, 80, 60),
        threshold=80,
        previous_result=reference,
        reference_result=reference,
    )

    assert result["pupil_valid"]
    assert abs(result["pupil_center_x"] - 80) < 2
    assert result["pupil_area"] > 0.8 * reference["pupil_area"]
    assert result["pupil_glare_close_size"] > 5


def test_temporal_prior_selects_the_matching_dark_candidate():
    frame = np.full((120, 180, 3), 180, dtype=np.uint8)
    cv2.circle(frame, (50, 60), 11, (10, 10, 10), -1)
    cv2.circle(frame, (130, 60), 17, (10, 10, 10), -1)
    previous = {
        "pupil_valid": True,
        "pupil_center_x": 50.0,
        "pupil_center_y": 60.0,
        "pupil_area": float(np.pi * 11**2),
    }

    result = detect_pupil(
        frame,
        ROI(20, 25, 140, 70),
        threshold=80,
        previous_result=previous,
    )

    assert result["pupil_center_x"] < 80


def test_reference_prior_can_recover_from_wrong_temporal_candidate():
    frame = np.full((120, 180, 3), 180, dtype=np.uint8)
    cv2.circle(frame, (50, 60), 12, (10, 10, 10), -1)
    cv2.circle(frame, (130, 60), 16, (10, 10, 10), -1)
    previous = {
        "pupil_valid": True,
        "pupil_center_x": 130.0,
        "pupil_center_y": 60.0,
        "pupil_area": float(np.pi * 16**2),
    }
    reference = {
        "pupil_valid": True,
        "pupil_center_x": 50.0,
        "pupil_center_y": 60.0,
        "pupil_area": float(np.pi * 12**2),
    }

    result = detect_pupil(
        frame,
        ROI(20, 25, 140, 70),
        threshold=80,
        previous_result=previous,
        reference_result=reference,
    )

    assert result["pupil_valid"]
    assert result["pupil_center_x"] < 80


def test_adaptive_threshold_tracks_pupil_across_illumination_shift():
    dark = np.full((120, 160, 3), 180, dtype=np.uint8)
    bright = np.full((120, 160, 3), 120, dtype=np.uint8)
    cv2.ellipse(dark, (80, 60), (18, 12), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(bright, (80, 60), (18, 12), 0, 0, 360, (65, 65, 65), -1)
    eye = ROI(40, 30, 80, 60)

    first = detect_pupil(dark, eye, threshold=None)
    second = detect_pupil(bright, eye, threshold=None, previous_result=first)
    fixed = detect_pupil(bright, eye, threshold=55, previous_result=first)

    assert first["pupil_valid"]
    assert second["pupil_valid"]
    assert not fixed["pupil_valid"]
    assert abs(second["pupil_area"] / first["pupil_area"] - 1.0) < 0.15
    assert second["pupil_threshold_mode"] == "adaptive"


def test_polygon_config_round_trip_and_seed_prior(tmp_path):
    config_path = tmp_path / "rois.yaml"
    config_path.write_text("schema_version: 1\nsessions: {}\n", encoding="utf-8")
    eye = PolygonROI(((10, 10), (40, 10), (35, 35), (10, 30)))
    movement = ROI(0, 0, 50, 50)
    seed = PolygonROI(((18, 17), (28, 16), (31, 24), (20, 27)))

    save_camera_annotation(
        config_path,
        "session",
        "01",
        eye,
        movement,
        pupil_seed=seed,
        eye_landmarks={"anchor_a": (12, 12), "anchor_b": (38, 12)},
        reference_frame_index=4,
    )
    saved = load_roi_config(config_path)["session"]["01"]

    assert region_from_mapping(saved["eye"]) == eye
    assert region_from_mapping(saved["movement"]) == movement
    assert region_from_mapping(region_to_mapping(seed)) == seed
    prior = pupil_prior_from_polygon(seed)
    assert prior["pupil_valid"]
    assert prior["pupil_area"] > 0


def test_manual_pupil_anchor_is_saved_without_overwriting_base_roi(tmp_path):
    config_path = tmp_path / "rois.yaml"
    config_path.write_text("schema_version: 1\nsessions: {}\n", encoding="utf-8")
    eye = PolygonROI(((10, 10), (40, 10), (35, 35), (10, 30)))
    movement = ROI(0, 0, 50, 50)
    pupil = PolygonROI(((18, 17), (25, 15), (31, 20), (30, 27), (22, 29), (17, 24)))
    save_camera_annotation(config_path, "session", "01", eye, movement)

    anchors = save_pupil_manual_anchor(
        config_path, "session", "01", 42, pupil
    )
    saved = load_roi_config(config_path)["session"]["01"]

    assert "42" in anchors
    assert region_from_mapping(saved["eye"]) == eye
    assert region_from_mapping(saved["pupil_manual_anchors"]["42"]) == pupil


def test_landmark_tracker_moves_eye_polygon_with_image_translation():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    translation = np.float32([[1, 0, 3], [0, 1, 2]])
    shifted = cv2.warpAffine(frame, translation, (160, 120))
    landmarks = {
        "a": (40, 35),
        "b": (100, 35),
        "c": (100, 80),
        "d": (40, 80),
    }
    eye = PolygonROI(((55, 45), (85, 45), (85, 70), (55, 70)))
    tracker = EyeLandmarkTracker(frame, landmarks, eye)

    tracker.update(shifted)

    measurements = tracker.measurements()
    moved = np.asarray(tracker.eye_region.vertices)
    assert measurements["eye_stabilization_valid_fraction"] >= 0.75
    assert np.allclose(moved.mean(axis=0), np.asarray(eye.vertices).mean(axis=0) + [3, 2], atol=1)


def test_landmark_tracker_rejects_implausible_jump_without_moving_eye_roi():
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    jumped = cv2.warpAffine(frame, np.float32([[1, 0, 55], [0, 1, 35]]), (160, 120))
    landmarks = {"a": (40, 35), "b": (100, 35), "c": (100, 80), "d": (40, 80)}
    eye = PolygonROI(((55, 45), (85, 45), (85, 70), (55, 70)))
    tracker = EyeLandmarkTracker(frame, landmarks, eye)

    tracker.update(jumped)

    measurements = tracker.measurements()
    assert not measurements["eye_stabilization_valid"]
    assert measurements["eye_stabilization_status"] != "accepted"
    assert np.allclose(np.asarray(tracker.eye_region.vertices), np.asarray(eye.vertices))


def test_landmark_tracker_periodically_reanchors_to_reference_templates():
    rng = np.random.default_rng(19)
    frame = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    translation = np.float32([[1, 0, 3], [0, 1, 2]])
    shifted = cv2.warpAffine(frame, translation, (160, 120))
    landmarks = {"a": (40, 35), "b": (100, 35), "c": (100, 80), "d": (40, 80)}
    eye = PolygonROI(((55, 45), (85, 45), (85, 70), (55, 70)))
    tracker = EyeLandmarkTracker(
        frame, landmarks, eye, reanchor_interval_frames=1
    )
    tracker.points += np.asarray([10.0, 0.0], dtype=np.float32)

    tracker.update(shifted)

    measurements = tracker.measurements()
    assert measurements["eye_stabilization_reanchor_count"] == 1
    assert measurements["eye_stabilization_status"] == "accepted_reanchored"
    assert np.allclose(
        np.asarray(tracker.eye_region.vertices).mean(axis=0),
        np.asarray(eye.vertices).mean(axis=0) + [3, 2],
        atol=1,
    )


def test_extract_behavior_writes_contour_lists_to_parquet(tmp_path):
    video_path = tmp_path / "short.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120)
    )
    assert writer.isOpened()
    for index in range(12):
        frame = np.full((120, 160, 3), 180, dtype=np.uint8)
        cv2.ellipse(frame, (75, 55), (14 + index // 4, 10), 0, 0, 360, (10, 10, 10), -1)
        cv2.rectangle(frame, (115 + index % 2, 70), (145, 100), (60, 60, 60), -1)
        writer.write(frame)
    writer.release()
    transform = TimeTransform(
        stream="01",
        anchor_device_s=0.0,
        anchor_session_s=0.0,
        segment_start_device_s=0.0,
        segment_start_session_s=0.0,
    )
    eye = PolygonROI(((50, 35), (100, 35), (105, 78), (45, 78)))
    seed = PolygonROI(((61, 46), (75, 43), (89, 47), (90, 63), (75, 67), (61, 63)))

    behavior = extract_behavior(
        "session",
        "01",
        video_path,
        transform,
        eye,
        ROI(110, 65, 40, 40),
        train_first_onset_s=0.5,
        pupil_threshold=80,
        pupil_seed=seed,
        pupil_boundary_points=12,
        manual_pupil_anchors={
            str(index): {
                "type": "polygon",
                "vertices": [[61, 46], [75, 43], [89, 47], [90, 63], [75, 67], [61, 63]],
            }
            for index in range(26)
        },
        max_manual_pupil_anchors=100,
    )
    parquet_path = tmp_path / "behavior.parquet"
    behavior.to_parquet(parquet_path, index=False)
    restored = pd.read_parquet(parquet_path)

    assert len(restored) == 12
    assert restored["pupil_valid"].all()
    assert restored["pupil_boundary_x"].map(len).eq(12).all()
    assert restored["pupil_raw_contour_area"].notna().all()
    assert restored["pupil_hull_correction_fraction"].between(0, 1).all()
    assert restored["pupil_area_normalized"].notna().all()
    assert restored["pupil_analysis_valid"].all()
    assert restored.loc[5, "pupil_manual_anchor"]
    assert restored.loc[5, "pupil_detection_status"] == "manual_anchor"


def test_audit_indices_cover_video_every_two_minutes_and_include_last_frame():
    indices = pupil_audit_frame_indices(
        144_282, 30.0, start_frame=100, interval_s=120.0, include_last=True
    )

    assert indices[:2] == [100, 3700]
    assert indices[-1] == 144_281
    assert len(indices) == 42


def test_representative_invalid_frames_are_bounded_and_span_recording():
    behavior = pd.DataFrame(
        {"frame_index": np.arange(20), "pupil_valid": [True] * 2 + [False] * 18}
    )

    selected = representative_invalid_frame_indices(behavior, max_frames=4)

    assert len(selected) == 4
    assert selected[0] == 2
    assert selected[-1] == 19


def test_review_frames_include_valid_but_flagged_observations():
    behavior = pd.DataFrame(
        {
            "frame_index": np.arange(6),
            "pupil_valid": [True, True, False, True, True, True],
            "pupil_review_required": [False, True, True, False, False, True],
        }
    )

    selected = representative_pupil_review_frame_indices(behavior, max_frames=6)

    assert selected == [1, 2, 5]


def test_bidirectional_qc_rejects_isolated_area_outlier_but_keeps_contour():
    behavior = pd.DataFrame(
        {
            "frame_index": np.arange(9),
            "pupil_valid": True,
            "pupil_detection_status": "observed",
            "pupil_area": [100.0] * 4 + [20.0] + [100.0] * 4,
            "pupil_center_x": 50.0,
            "pupil_center_y": 50.0,
            "pupil_hull_correction_fraction": 0.05,
            "pupil_review_required": False,
            "pupil_review_reason": "",
            "pupil_reference_area_ratio": [1.0] * 4 + [5.0] + [1.0] * 4,
            "pupil_reference_center_distance_px": 0.0,
            "pupil_eye_region_scale_px": 60.0,
            "pupil_touches_eye_boundary": False,
            "pupil_boundary_x": [[45.0, 55.0, 55.0, 45.0]] * 9,
            "pupil_boundary_y": [[45.0, 45.0, 55.0, 55.0]] * 9,
        }
    )

    checked = apply_pupil_quality_control(behavior, 10.0)

    assert checked.loc[4, "pupil_detector_valid"]
    assert not checked.loc[4, "pupil_quality_valid"]
    assert checked.loc[4, "pupil_qc_rejected"]
    assert "bidirectional_area_outlier" in checked.loc[4, "pupil_review_reason"]
    assert len(checked.loc[4, "pupil_boundary_x"]) == 4

    anchored = behavior.copy()
    anchored["pupil_manual_anchor"] = False
    anchored.loc[4, "pupil_manual_anchor"] = True
    trusted = apply_pupil_quality_control(anchored, 10.0)
    assert trusted.loc[4, "pupil_quality_valid"]
    assert not trusted.loc[4, "pupil_qc_rejected"]


def test_only_short_nan_gaps_are_interpolated():
    values = [1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0]
    result = interpolate_short_nan_gaps(values, max_gap_frames=1)
    assert result[1] == 2.0
    assert np.isnan(result[3:6]).all()
