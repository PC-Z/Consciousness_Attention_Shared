from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .behavior import extract_behavior, load_roi_config, region_from_mapping
from .calcium import build_calcium_frame_table, inspect_calcium_source
from .config import ProjectConfig, SessionConfig, load_config
from .ephys import build_ephys_manifest
from .errors import AlignmentError, StimulusMatchError
from .models import RecordingSegment, StimulusBlock, TimeTransform, TriggerCluster
from .paths import PathPolicy, inventory_signature
from .stimulus import build_stimulus_event_table, match_stimulus_blocks, parse_order_file
from .timestamps import (
    alignment_qc,
    choose_main_segment,
    cluster_channel2,
    make_time_transforms,
    parse_marker_file,
)
from .video import build_video_frame_table, probe_video


@dataclass
class SessionAlignment:
    session_id: str
    manifest: dict[str, Any]
    qc: dict[str, Any]
    stimulus_events: pd.DataFrame
    calcium_frames: pd.DataFrame
    marker_segments: dict[str, list[RecordingSegment]]
    main_segments: dict[str, RecordingSegment]
    trigger_clusters: list[TriggerCluster]
    stimulus_blocks: list[StimulusBlock]
    transforms: dict[str, TimeTransform]

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.qc["status"],
            "stimulus_events": len(self.stimulus_events),
            "calcium_frames": len(self.calcium_frames),
            "channel2_clusters": len(self.trigger_clusters),
            "unpaired_tail_pulses": self.qc["calcium"]["unpaired_tail_pulses"],
            "video_01": self.manifest["videos"]["01"]["status"],
            "video_02": self.manifest["videos"]["02"]["status"],
        }


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (RecordingSegment, StimulusBlock, TimeTransform, TriggerCluster)):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _segment_with_blocks(
    segments: list[RecordingSegment],
    condition,
    cluster_gap_s: float,
    groups: int,
    items: int,
) -> RecordingSegment:
    def block_count(segment: RecordingSegment) -> int:
        try:
            clusters = cluster_channel2(segment, cluster_gap_s)
            blocks, _ = match_stimulus_blocks(clusters, condition, groups, items)
            return len(blocks)
        except StimulusMatchError:
            return 0

    main = choose_main_segment(segments, block_count)
    if block_count(main) != 2:
        raise StimulusMatchError("Selected main segment does not contain unique Train/Test blocks")
    return main


def build_session(config: ProjectConfig, session: SessionConfig) -> SessionAlignment:
    condition = config.condition(session.condition_label_ms)
    cluster_gap_s = float(config.stimulus["cluster_gap_s"])
    groups = int(config.stimulus["groups_per_phase"])
    items = int(config.stimulus["items_per_group"])

    segments = {
        stream: parse_marker_file(config.marker_path(session.id, stream))
        for stream in ("01", "02")
    }
    main_segments = {
        stream: _segment_with_blocks(
            segments[stream], condition, cluster_gap_s, groups, items
        )
        for stream in ("01", "02")
    }
    transforms = make_time_transforms(main_segments["01"], main_segments["02"])
    clusters = cluster_channel2(main_segments["02"], cluster_gap_s)
    blocks, trigger_qc = match_stimulus_blocks(clusters, condition, groups, items)
    test_sequences = parse_order_file(config.order_path(condition.label_ms))
    angles = {str(key): int(value) for key, value in config.stimulus["angle_degrees"].items()}
    stimulus_events = build_stimulus_event_table(
        session.id,
        condition,
        test_sequences,
        angles,
        blocks,
        clusters,
        transforms["02"],
    )

    calcium_info = inspect_calcium_source(
        config.calcium_path(session.id), config.calcium["datasets"]
    )
    calcium_frames, calcium_qc = build_calcium_frame_table(
        session.id, main_segments["02"], transforms["02"], calcium_info
    )
    ephys_manifest = build_ephys_manifest(
        config.eeg_dir(session.id) / str(config.ephys["dat_file"]),
        config.eeg_dir(session.id) / str(config.ephys["metadata_file"]),
        main_segments["02"].segment_index,
        transforms["02"],
        int(config.ephys["tick_hz"]),
    )
    videos = {
        stream: probe_video(config.video_path(session.id, stream)) for stream in ("01", "02")
    }
    marker_qc = alignment_qc(main_segments["01"], main_segments["02"], cluster_gap_s)
    timing_outliers = int((stimulus_events["timing_qc"] != "ok").sum())
    pending_videos = [stream for stream, item in videos.items() if item["status"] != "ok"]
    status = "ok_with_pending_video" if pending_videos else "ok"
    qc: dict[str, Any] = {
        "status": status,
        "marker_alignment": marker_qc,
        "triggers": trigger_qc,
        "calcium": calcium_qc,
        "stimulus_timing_outliers": timing_outliers,
        "pending_video_streams": pending_videos,
        "warnings": [
            "VIDEO_UNREADABLE_PENDING_REPAIR" if pending_videos else None,
            "STIMULUS_TIMING_OUTLIER" if timing_outliers else None,
        ],
    }
    qc["warnings"] = [item for item in qc["warnings"] if item is not None]

    manifest = {
        "schema_version": 1,
        "session_id": session.id,
        "condition_label_ms": condition.label_ms,
        "expected_stripe_duration_s": condition.expected_stripe_s,
        "expected_within_group_gray_s": condition.expected_gray_s,
        "sources": {
            "marker_01": str(config.marker_path(session.id, "01").resolve()),
            "marker_02": str(config.marker_path(session.id, "02").resolve()),
            "calcium": calcium_info,
            "ephys_02": ephys_manifest,
        },
        "videos": videos,
        "main_segment_index": {
            stream: main_segments[stream].segment_index for stream in ("01", "02")
        },
        "time_transforms": {stream: asdict(value) for stream, value in transforms.items()},
        "stimulus_blocks": [asdict(value) for value in blocks],
    }
    return SessionAlignment(
        session_id=session.id,
        manifest=manifest,
        qc=qc,
        stimulus_events=stimulus_events,
        calcium_frames=calcium_frames,
        marker_segments=segments,
        main_segments=main_segments,
        trigger_clusters=clusters,
        stimulus_blocks=blocks,
        transforms=transforms,
    )


def export_session(
    config: ProjectConfig,
    result: SessionAlignment,
    include_video_indexes: bool = True,
    include_behavior: bool = True,
) -> Path:
    """Write one reviewed alignment result under outputs/<session_id>."""

    policy = PathPolicy(config.output_root)
    output_dir = policy.ensure_output_dir(result.session_id)
    result.stimulus_events.to_parquet(output_dir / "stimulus_events.parquet", index=False)
    result.calcium_frames.to_parquet(output_dir / "calcium_frames.parquet", index=False)
    _write_json(output_dir / "manifest.json", result.manifest)
    _write_json(output_dir / "qc.json", result.qc)
    _write_json(output_dir / "ephys_02_manifest.json", result.manifest["sources"]["ephys_02"])

    if include_video_indexes:
        for camera in ("01", "02"):
            if result.manifest["videos"][camera]["status"] != "ok":
                continue
            table = build_video_frame_table(
                result.session_id,
                camera,
                config.video_path(result.session_id, camera),
                result.transforms[camera],
            )
            table.to_parquet(output_dir / f"video_frames_{camera}.parquet", index=False)

    roi_sessions = load_roi_config(
        config.project_root / "configs" / config.behavior["roi_config_file"]
    )
    session_rois = roi_sessions.get(result.session_id, {})
    if include_behavior and session_rois:
        first_onset = float(result.stimulus_events.iloc[0]["measured_onset_s"])
        for camera in ("01", "02"):
            camera_rois = session_rois.get(camera)
            if not camera_rois or result.manifest["videos"][camera]["status"] != "ok":
                continue
            behavior = extract_behavior(
                result.session_id,
                camera,
                config.video_path(result.session_id, camera),
                result.transforms[camera],
                region_from_mapping(camera_rois["eye"]),
                region_from_mapping(camera_rois["movement"]),
                first_onset,
                float(config.behavior["max_interpolation_gap_s"]),
                camera_rois.get("pupil_threshold"),
                pupil_seed=(
                    region_from_mapping(camera_rois["pupil_seed"])
                    if camera_rois.get("pupil_seed")
                    else None
                ),
                eye_landmarks=camera_rois.get("eye_landmarks"),
                landmark_reference_frame=int(camera_rois.get("reference_frame_index", 0)),
                pupil_boundary_points=int(config.behavior.get("pupil_boundary_points", 32)),
                landmark_reanchor_interval_frames=int(
                    config.behavior.get("landmark_reanchor_interval_frames", 30)
                ),
                pupil_qc_window_s=float(
                    config.behavior.get("pupil_qc_window_s", 0.5)
                ),
                exclude_review_from_analysis=bool(
                    config.behavior.get("exclude_review_from_analysis", True)
                ),
                use_stabilized_eye_roi=bool(
                    config.behavior.get("use_stabilized_eye_roi", False)
                ),
                manual_pupil_anchors=camera_rois.get("pupil_manual_anchors"),
                max_manual_pupil_anchors=int(
                    config.behavior.get("max_manual_pupil_anchors", 20)
                ),
                manual_anchor_influence_frames=int(
                    config.behavior.get("manual_anchor_influence_frames", 60)
                ),
            )
            behavior.to_parquet(output_dir / f"behavior_{camera}.parquet", index=False)
    return output_dir


def build_alignment(
    config: str | Path | ProjectConfig | None = None,
    dry_run: bool = True,
    session_ids: Iterable[str] | None = None,
) -> dict[str, SessionAlignment]:
    """Build all compact alignments; dry-run performs no filesystem writes."""

    project = config if isinstance(config, ProjectConfig) else load_config(config)
    requested = set(session_ids) if session_ids is not None else None
    source_roots = [project.data_root, project.stimuli_root]
    before = inventory_signature(source_roots)
    results: dict[str, SessionAlignment] = {}
    for session in project.sessions:
        if requested is not None and session.id not in requested:
            continue
        results[session.id] = build_session(project, session)
    if not dry_run:
        for result in results.values():
            export_session(project, result)
    after = inventory_signature(source_roots)
    if before != after:
        raise AlignmentError("A read-only source file changed while the pipeline was running")
    return results
