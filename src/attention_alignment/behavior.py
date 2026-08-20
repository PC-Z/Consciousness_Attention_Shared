from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import fill
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Ellipse, Polygon as PolygonPatch
from matplotlib.widgets import PolygonSelector, RectangleSelector

from .models import TimeTransform
from .video import iter_video_frames, read_video_frame


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    def validate(self, image: np.ndarray) -> None:
        image_height, image_width = image.shape[:2]
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")
        if (
            self.x < 0
            or self.y < 0
            or self.x + self.width > image_width
            or self.y + self.height > image_height
        ):
            raise ValueError(f"ROI {self} is outside image bounds {image_width}x{image_height}")

    def bounds(self, image: np.ndarray) -> tuple[int, int, int, int]:
        self.validate(image)
        return self.x, self.y, self.x + self.width, self.y + self.height

    def crop(self, image: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds(image)
        return image[y0:y1, x0:x1]


@dataclass(frozen=True)
class PolygonROI:
    """A polygonal image region in full-frame pixel coordinates."""

    vertices: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        normalized = tuple((float(x), float(y)) for x, y in self.vertices)
        if len(normalized) < 3:
            raise ValueError("PolygonROI requires at least three vertices")
        if not np.isfinite(np.asarray(normalized, dtype=float)).all():
            raise ValueError("PolygonROI vertices must be finite")
        object.__setattr__(self, "vertices", normalized)

    def validate(self, image: np.ndarray) -> None:
        image_height, image_width = image.shape[:2]
        points = np.asarray(self.vertices, dtype=float)
        if (
            (points[:, 0] < 0).any()
            or (points[:, 1] < 0).any()
            or (points[:, 0] >= image_width).any()
            or (points[:, 1] >= image_height).any()
        ):
            raise ValueError(
                f"PolygonROI is outside image bounds {image_width}x{image_height}"
            )

    def bounds(self, image: np.ndarray) -> tuple[int, int, int, int]:
        self.validate(image)
        points = np.asarray(self.vertices, dtype=float)
        image_height, image_width = image.shape[:2]
        x0 = max(0, int(np.floor(points[:, 0].min())))
        y0 = max(0, int(np.floor(points[:, 1].min())))
        x1 = min(image_width, int(np.ceil(points[:, 0].max())) + 1)
        y1 = min(image_height, int(np.ceil(points[:, 1].max())) + 1)
        return x0, y0, x1, y1

    def crop(self, image: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds(image)
        return image[y0:y1, x0:x1]


Region = ROI | PolygonROI


def region_to_mapping(region: Region) -> dict[str, object]:
    if isinstance(region, PolygonROI):
        return {
            "type": "polygon",
            "vertices": [[float(x), float(y)] for x, y in region.vertices],
        }
    return {"type": "rectangle", **asdict(region)}


def region_from_mapping(value: Mapping[str, Any]) -> Region:
    region_type = value.get("type")
    if region_type == "polygon" or "vertices" in value:
        return PolygonROI(tuple(tuple(point) for point in value["vertices"]))
    if region_type not in (None, "rectangle"):
        raise ValueError(f"Unsupported ROI type: {region_type}")
    return ROI(
        x=int(value["x"]),
        y=int(value["y"]),
        width=int(value["width"]),
        height=int(value["height"]),
    )


def _region_crop_and_mask(
    frame: np.ndarray, region: Region
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = region.bounds(frame)
    crop = frame[y0:y1, x0:x1]
    mask = np.ones(crop.shape[:2], dtype=np.uint8) * 255
    if isinstance(region, PolygonROI):
        mask.fill(0)
        points = np.rint(
            np.asarray(region.vertices, dtype=float) - np.asarray([x0, y0], dtype=float)
        ).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    return crop, mask, (x0, y0, x1, y1)


class NotebookROISelector:
    """A visibly styled, draggable rectangle selector for notebook use."""

    def __init__(self, video_path: str | Path, frame_index: int = 0, title: str = "Select ROI"):
        frame = read_video_frame(video_path, frame_index)
        self._roi: ROI | None = None
        self.figure, self.axis = plt.subplots(figsize=(9, 6))
        self.axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.axis.set_title(f"{title}\nDrag to draw; drag handles to adjust")
        self.axis.set_axis_off()
        self._status = self.figure.text(0.5, 0.02, "No ROI selected", ha="center")

        def selected(click, release):
            if None in (click.xdata, click.ydata, release.xdata, release.ydata):
                return
            x0, x1 = sorted((int(round(click.xdata)), int(round(release.xdata))))
            y0, y1 = sorted((int(round(click.ydata)), int(round(release.ydata))))
            self._roi = ROI(x0, y0, x1 - x0, y1 - y0)
            self._roi.validate(frame)
            value = asdict(self._roi)
            self._status.set_text(f"ROI: {value}")
            self.figure.canvas.draw_idle()
            print("ROI:", value)

        self.selector = RectangleSelector(
            self.axis,
            selected,
            useblit=False,
            interactive=True,
            drag_from_anywhere=True,
            props={
                "facecolor": "#00b7d6",
                "edgecolor": "#ffe600",
                "alpha": 0.28,
                "fill": True,
                "linewidth": 2,
            },
            handle_props={"markerfacecolor": "#ffe600", "markersize": 7},
        )
        self.figure.tight_layout(rect=(0, 0.05, 1, 1))

    @property
    def roi(self) -> ROI:
        if self._roi is None:
            raise RuntimeError("Draw a rectangle before reading .roi")
        return self._roi


class NotebookPolygonSelector:
    """Click-to-label polygon with a live line and persistent filled overlay."""

    def __init__(
        self,
        video_path: str | Path,
        frame_index: int = 0,
        title: str = "Select polygon ROI",
    ):
        frame = read_video_frame(video_path, frame_index)
        self._roi: PolygonROI | None = None
        self._patch: PolygonPatch | None = None
        self.figure, self.axis = plt.subplots(figsize=(9, 6))
        self.axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.axis.set_title(f"{title}\nClick vertices; click the first vertex to close")
        self.axis.set_axis_off()
        self._status = self.figure.text(0.5, 0.02, "Polygon is not closed", ha="center")

        def selected(vertices):
            self._roi = PolygonROI(tuple((float(x), float(y)) for x, y in vertices))
            self._roi.validate(frame)
            if self._patch is not None:
                self._patch.remove()
            self._patch = PolygonPatch(
                self._roi.vertices,
                closed=True,
                facecolor="#00b7d6",
                edgecolor="#ffe600",
                linewidth=2,
                alpha=0.28,
            )
            self.axis.add_patch(self._patch)
            self._status.set_text(f"Polygon confirmed: {len(self._roi.vertices)} vertices")
            self.figure.canvas.draw_idle()
            print("Polygon ROI:", region_to_mapping(self._roi))

        self.selector = PolygonSelector(
            self.axis,
            selected,
            useblit=False,
            props={"color": "#ffe600", "linewidth": 2, "alpha": 0.95},
            handle_props={
                "marker": "o",
                "markerfacecolor": "#00b7d6",
                "markeredgecolor": "#ffe600",
                "markersize": 7,
            },
            draw_bounding_box=True,
        )
        self.figure.tight_layout(rect=(0, 0.05, 1, 1))

    @property
    def roi(self) -> PolygonROI:
        if self._roi is None:
            raise RuntimeError("Close the polygon before reading .roi")
        return self._roi


class NotebookLandmarkSelector:
    """Sequential landmark annotator with immediate point labels and connections."""

    def __init__(
        self,
        video_path: str | Path,
        frame_index: int = 0,
        names: Sequence[str] = ("anchor_1", "anchor_2", "anchor_3", "anchor_4"),
        title: str = "Select eye stabilization landmarks",
    ):
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("Provide at least two unique landmark names")
        self.names = tuple(str(name) for name in names)
        self._points: list[tuple[float, float]] = []
        self._labels: list[Any] = []
        frame = read_video_frame(video_path, frame_index)
        self.figure, self.axis = plt.subplots(figsize=(9, 6))
        self.axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.axis.set_title(f"{title}\nLeft click in order; Backspace=undo, R=reset")
        self.axis.set_axis_off()
        (self._line,) = self.axis.plot([], [], "-", color="#ffe600", linewidth=1.5)
        self._scatter = self.axis.scatter([], [], s=55, c="#00e5ff", edgecolors="black")
        self._status = self.figure.text(0.5, 0.02, "", ha="center")
        self._render()
        self._click_id = self.figure.canvas.mpl_connect("button_press_event", self._clicked)
        self._key_id = self.figure.canvas.mpl_connect("key_press_event", self._key_pressed)
        self.figure.tight_layout(rect=(0, 0.05, 1, 1))

    def _clicked(self, event) -> None:
        if event.button != 1 or event.inaxes is not self.axis or event.xdata is None:
            return
        if len(self._points) >= len(self.names):
            return
        point = (float(event.xdata), float(event.ydata))
        self._points.append(point)
        print(f"Landmark {self.names[len(self._points) - 1]}: {point}")
        self._render()

    def _key_pressed(self, event) -> None:
        if event.key in ("backspace", "delete"):
            self.undo()
        elif event.key in ("r", "R"):
            self.reset()

    def _render(self) -> None:
        for label in self._labels:
            label.remove()
        self._labels.clear()
        if self._points:
            points = np.asarray(self._points, dtype=float)
            self._scatter.set_offsets(points)
            line_points = points
            if len(points) == len(self.names) and len(points) > 2:
                line_points = np.vstack([points, points[0]])
            self._line.set_data(line_points[:, 0], line_points[:, 1])
            for name, (x, y) in zip(self.names, self._points, strict=False):
                self._labels.append(
                    self.axis.text(
                        x + 4,
                        y - 4,
                        name,
                        color="white",
                        fontsize=9,
                        bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
                    )
                )
        else:
            self._scatter.set_offsets(np.empty((0, 2)))
            self._line.set_data([], [])
        if len(self._points) < len(self.names):
            self._status.set_text(
                f"Next: {self.names[len(self._points)]} ({len(self._points)}/{len(self.names)})"
            )
        else:
            self._status.set_text(f"All {len(self.names)} landmarks confirmed")
        self.figure.canvas.draw_idle()

    def undo(self) -> None:
        if self._points:
            self._points.pop()
            self._render()

    def reset(self) -> None:
        self._points.clear()
        self._render()

    @property
    def landmarks(self) -> dict[str, tuple[float, float]]:
        if len(self._points) != len(self.names):
            raise RuntimeError("Select every landmark before reading .landmarks")
        return dict(zip(self.names, self._points, strict=True))


def save_roi_config(path: str | Path, sessions: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {"schema_version": 1, "sessions": sessions}
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)
    temporary.replace(target)


def load_roi_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if payload.get("schema_version") != 1:
        raise ValueError("ROI config schema_version must be 1")
    return payload.get("sessions", {})


def ensure_session_roi_config(
    session_path: str | Path,
    session_id: str,
    aggregate_path: str | Path | None = None,
) -> Path:
    """Create a per-session ROI file, migrating its entry from an aggregate file.

    Existing per-session files are never rewritten. When a session file does not
    exist, only the requested session entry is copied from ``aggregate_path``;
    this keeps future annotation edits isolated between collaborators.
    """

    target = Path(session_path)
    if target.is_file():
        return target
    sessions: dict[str, object] = {}
    fallback = Path(aggregate_path) if aggregate_path is not None else None
    if fallback is not None and fallback.is_file():
        sessions = load_roi_config(fallback)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_roi_config(target, {session_id: sessions.get(session_id, {})})
    return target


def save_camera_annotation(
    path: str | Path,
    session_id: str,
    camera: str,
    eye: Region,
    movement: Region,
    *,
    pupil_seed: PolygonROI | None = None,
    eye_landmarks: Mapping[str, Sequence[float]] | None = None,
    reference_frame_index: int = 0,
    pupil_threshold: float | None = None,
) -> dict[str, object]:
    """Atomically persist one reviewed camera annotation."""

    sessions = load_roi_config(path)
    camera_config: dict[str, object] = {
        "eye": region_to_mapping(eye),
        "movement": region_to_mapping(movement),
        "reference_frame_index": int(reference_frame_index),
    }
    if eye_landmarks:
        camera_config["eye_landmarks"] = {
            str(name): [float(point[0]), float(point[1])]
            for name, point in eye_landmarks.items()
        }
    if pupil_seed is not None:
        camera_config["pupil_seed"] = region_to_mapping(pupil_seed)
    if pupil_threshold is not None:
        camera_config["pupil_threshold"] = float(pupil_threshold)
    existing_camera = sessions.get(session_id, {}).get(camera, {})
    if existing_camera.get("pupil_manual_anchors"):
        camera_config["pupil_manual_anchors"] = existing_camera[
            "pupil_manual_anchors"
        ]
    sessions.setdefault(session_id, {})[camera] = camera_config
    save_roi_config(path, sessions)
    return camera_config


def save_pupil_manual_anchor(
    path: str | Path,
    session_id: str,
    camera: str,
    frame_index: int,
    pupil: PolygonROI,
    *,
    max_anchors: int = 20,
) -> dict[str, object]:
    """Atomically add or replace one manually reviewed pupil frame."""

    if frame_index < 0:
        raise ValueError("Manual anchor frame_index must be non-negative")
    if max_anchors <= 0:
        raise ValueError("max_anchors must be positive")
    sessions = load_roi_config(path)
    try:
        camera_config = sessions[session_id][camera]
    except KeyError as error:
        raise KeyError(
            f"Save the base ROI annotation before adding a pupil anchor: "
            f"{session_id}/{camera}"
        ) from error
    anchors = camera_config.setdefault("pupil_manual_anchors", {})
    key = str(int(frame_index))
    if key not in anchors and len(anchors) >= max_anchors:
        raise ValueError(f"At most {max_anchors} manual pupil anchors are allowed")
    anchors[key] = region_to_mapping(pupil)
    save_roi_config(path, sessions)
    return anchors


def _resample_contour(contour: np.ndarray, count: int) -> np.ndarray:
    points = contour.reshape(-1, 2).astype(float)
    if len(points) == 0 or count <= 0:
        return np.empty((0, 2), dtype=float)
    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if cumulative[-1] <= 0:
        return np.repeat(points[:1], count, axis=0)
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    return np.column_stack(
        [np.interp(samples, cumulative, closed[:, axis]) for axis in (0, 1)]
    )


def _invalid_pupil(
    threshold_value: float,
    *,
    threshold_mode: str,
    reason: str = "no_plausible_dark_component",
) -> dict[str, Any]:
    return {
        "pupil_valid": False,
        "blink_or_invalid": True,
        "pupil_detection_status": reason,
        "pupil_manual_anchor": False,
        "pupil_review_required": True,
        "pupil_review_reason": reason,
        "pupil_threshold_mode": threshold_mode,
        "pupil_candidate_count": 0,
        "pupil_center_x": np.nan,
        "pupil_center_y": np.nan,
        "pupil_major_axis": np.nan,
        "pupil_minor_axis": np.nan,
        "pupil_angle_degrees": np.nan,
        "pupil_area": np.nan,
        "pupil_raw_contour_area": np.nan,
        "pupil_hull_correction_fraction": np.nan,
        "pupil_mask_area_px": np.nan,
        "pupil_perimeter_px": np.nan,
        "pupil_raw_perimeter_px": np.nan,
        "pupil_equivalent_radius": np.nan,
        "pupil_confidence": 0.0,
        "pupil_threshold": threshold_value,
        "pupil_glare_close_size": np.nan,
        "pupil_touches_eye_boundary": False,
        "pupil_eye_boundary_contact_fraction": np.nan,
        "pupil_reference_center_distance_px": np.nan,
        "pupil_reference_area_ratio": np.nan,
        "pupil_reference_score": np.nan,
        "pupil_eye_region_scale_px": np.nan,
        "pupil_boundary_x": [],
        "pupil_boundary_y": [],
    }


def pupil_prior_from_polygon(polygon: PolygonROI) -> dict[str, Any]:
    """Convert a manually reviewed pupil boundary into a temporal prior."""

    points = np.asarray(polygon.vertices, dtype=np.float32)
    hull = cv2.convexHull(points)
    area = float(abs(cv2.contourArea(hull)))
    moments = cv2.moments(hull)
    if area <= 0 or moments["m00"] == 0:
        raise ValueError("Pupil seed polygon must have positive area")
    return {
        "pupil_valid": True,
        "pupil_center_x": float(moments["m10"] / moments["m00"]),
        "pupil_center_y": float(moments["m01"] / moments["m00"]),
        "pupil_area": area,
    }


def pupil_result_from_polygon(
    frame: np.ndarray,
    eye_roi: Region,
    polygon: PolygonROI,
    *,
    boundary_points: int = 32,
) -> dict[str, Any]:
    """Measure one manually reviewed pupil polygon without re-segmenting it."""

    polygon.validate(frame)
    points = np.asarray(polygon.vertices, dtype=np.float32)
    hull = cv2.convexHull(points)
    area = float(abs(cv2.contourArea(hull)))
    if area <= 0 or len(hull) < 5:
        raise ValueError("Manual pupil polygon must contain at least five usable points")
    moments = cv2.moments(hull)
    center_x = float(moments["m10"] / moments["m00"])
    center_y = float(moments["m01"] / moments["m00"])
    (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(hull)
    perimeter = float(cv2.arcLength(hull, True))
    boundary = _resample_contour(hull, boundary_points)
    crop, region_mask, (x0, y0, _, _) = _region_crop_and_mask(frame, eye_roi)
    local_hull = hull.copy()
    local_hull[:, 0, 0] -= x0
    local_hull[:, 0, 1] -= y0
    local_hull = np.rint(local_hull).astype(np.int32)
    candidate_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(candidate_mask, [local_hull], -1, 255, cv2.FILLED)
    candidate_edge = cv2.subtract(
        candidate_mask,
        cv2.erode(candidate_mask, np.ones((3, 3), dtype=np.uint8)),
    )
    boundary_band = cv2.subtract(
        region_mask,
        cv2.erode(region_mask, np.ones((3, 3), dtype=np.uint8)),
    )
    edge_pixels = max(float(cv2.countNonZero(candidate_edge)), 1.0)
    contact_fraction = float(
        np.count_nonzero((candidate_edge > 0) & (boundary_band > 0))
        / edge_pixels
    )
    return {
        "pupil_valid": True,
        "blink_or_invalid": False,
        "pupil_detection_status": "manual_anchor",
        "pupil_manual_anchor": True,
        "pupil_review_required": False,
        "pupil_review_reason": "",
        "pupil_threshold_mode": "manual",
        "pupil_candidate_count": 0,
        "pupil_center_x": center_x,
        "pupil_center_y": center_y,
        "pupil_major_axis": max(float(axis_a), float(axis_b)),
        "pupil_minor_axis": min(float(axis_a), float(axis_b)),
        "pupil_angle_degrees": float(angle),
        "pupil_area": area,
        "pupil_raw_contour_area": area,
        "pupil_hull_correction_fraction": 0.0,
        "pupil_mask_area_px": float(cv2.countNonZero(candidate_mask)),
        "pupil_perimeter_px": perimeter,
        "pupil_raw_perimeter_px": perimeter,
        "pupil_equivalent_radius": float(np.sqrt(area / np.pi)),
        "pupil_confidence": 1.0,
        "pupil_threshold": np.nan,
        "pupil_glare_close_size": np.nan,
        "pupil_touches_eye_boundary": bool(contact_fraction > 0),
        "pupil_eye_boundary_contact_fraction": contact_fraction,
        "pupil_reference_center_distance_px": 0.0,
        "pupil_reference_area_ratio": 1.0,
        "pupil_reference_score": 1.0,
        "pupil_eye_region_scale_px": float(max(crop.shape[:2])),
        "pupil_boundary_x": boundary[:, 0].tolist(),
        "pupil_boundary_y": boundary[:, 1].tolist(),
    }


def detect_pupil(
    frame: np.ndarray,
    roi: Region,
    threshold: float | None = None,
    min_area_fraction: float = 0.005,
    max_area_fraction: float = 0.75,
    *,
    previous_result: Mapping[str, Any] | None = None,
    reference_result: Mapping[str, Any] | None = None,
    boundary_points: int = 32,
) -> dict[str, Any]:
    """Segment the pupil inside a reviewed region and measure its convex hull.

    The dark-pixel contour selects the anatomical candidate. Its convex hull is
    the measurement boundary so small specular highlights cannot create inward
    notches in the saved pupil polygon. Raw geometry is retained for audit.
    """

    crop, region_mask, (x0, y0, _, _) = _region_crop_and_mask(frame, roi)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    region_values = blurred[region_mask > 0]
    if region_values.size == 0:
        raise ValueError("Eye ROI contains no pixels")
    threshold_mode = "adaptive" if threshold is None else "fixed"
    if threshold is None:
        # Rank-based thresholds are stable under global illumination changes. Multiple
        # candidates let the temporal prior choose the anatomical pupil boundary.
        threshold_values = sorted(
            {
                float(value)
                for value in np.percentile(region_values, (8, 14, 20, 26, 32, 38))
            }
        )
    else:
        threshold_values = [float(threshold)]
    region_area = float(cv2.countNonZero(region_mask))
    region_scale = max(crop.shape[:2])
    region_median = float(np.median(region_values))
    region_std = float(np.std(region_values)) + 1.0
    boundary_band = cv2.subtract(
        region_mask,
        cv2.erode(region_mask, np.ones((3, 3), dtype=np.uint8)),
    )
    candidates: list[tuple[float, dict[str, Any]]] = []
    open_kernel = np.ones((3, 3), dtype=np.uint8)
    for threshold_value in threshold_values:
        threshold_mask = np.zeros_like(gray, dtype=np.uint8)
        threshold_mask[(blurred <= threshold_value) & (region_mask > 0)] = 255
        # A bright corneal reflection can split one dark pupil into several components.
        # Keep several closing scales as candidates instead of forcing every frame to
        # use an aggressively smoothed boundary.
        for close_size in (5, 9):
            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_size, close_size)
            )
            binary = cv2.morphologyEx(
                threshold_mask, cv2.MORPH_CLOSE, close_kernel
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
            binary[region_mask == 0] = 0
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for contour in contours:
                raw_area = float(cv2.contourArea(contour))
                hull = cv2.convexHull(contour)
                area = float(cv2.contourArea(hull))
                if len(contour) < 5 or area <= 0 or not (
                    min_area_fraction * region_area
                    <= area
                    <= max_area_fraction * region_area
                ):
                    continue
                raw_perimeter = float(cv2.arcLength(contour, True))
                perimeter = float(cv2.arcLength(hull, True))
                solidity = raw_area / area
                circularity = (
                    4.0 * np.pi * raw_area / (raw_perimeter * raw_perimeter)
                    if raw_perimeter
                    else 0.0
                )
                ellipse_points = hull if len(hull) >= 5 else contour
                (center_x, center_y), (axis_a, axis_b), angle = cv2.fitEllipse(
                    ellipse_points
                )
                hull_moments = cv2.moments(hull)
                if hull_moments["m00"]:
                    center_x = float(hull_moments["m10"] / hull_moments["m00"])
                    center_y = float(hull_moments["m01"] / hull_moments["m00"])
                candidate_mask = np.zeros_like(binary)
                cv2.drawContours(
                    candidate_mask, [contour], -1, 255, thickness=cv2.FILLED
                )
                mask_area = float(cv2.countNonZero(candidate_mask))
                candidate_values = gray[candidate_mask > 0]
                contrast = float(
                    np.clip(
                        (region_median - float(candidate_values.mean()))
                        / (3.0 * region_std),
                        0,
                        1,
                    )
                )
                touches_boundary = bool(
                    np.any((candidate_mask > 0) & (boundary_band > 0))
                )
                candidate_edge = cv2.subtract(
                    candidate_mask,
                    cv2.erode(
                        candidate_mask, np.ones((3, 3), dtype=np.uint8)
                    ),
                )
                edge_pixels = max(float(cv2.countNonZero(candidate_edge)), 1.0)
                boundary_contact_fraction = float(
                    np.count_nonzero((candidate_edge > 0) & (boundary_band > 0))
                    / edge_pixels
                )
                major_axis = max(float(axis_a), float(axis_b), 1.0)
                minor_axis = min(float(axis_a), float(axis_b))
                axis_ratio = minor_axis / major_axis
                area_fraction = area / region_area
                area_plausibility = float(
                    np.exp(-abs(np.log(max(area_fraction, 1e-4) / 0.22)))
                )
                smoothing_penalty = 0.015 * ((close_size - 5) / 4)
                boundary_score = 1.0 - float(
                    np.clip(boundary_contact_fraction / 0.40, 0.0, 1.0)
                )
                base_score = (
                    0.28 * np.clip(solidity, 0, 1)
                    + 0.20 * np.clip(circularity, 0, 1)
                    + 0.22 * contrast
                    + 0.10 * boundary_score
                    + 0.12 * area_plausibility
                    + 0.08 * np.clip(axis_ratio / 0.55, 0, 1)
                    - smoothing_penalty
                )
                temporal_score = 0.5
                has_temporal = bool(
                    previous_result and previous_result.get("pupil_valid")
                )
                if has_temporal:
                    previous_center = np.asarray(
                        [
                            float(previous_result["pupil_center_x"]) - x0,
                            float(previous_result["pupil_center_y"]) - y0,
                        ]
                    )
                    distance = float(
                        np.linalg.norm(
                            np.asarray([center_x, center_y]) - previous_center
                        )
                    )
                    center_score = float(
                        np.exp(-distance / max(0.15 * region_scale, 1.0))
                    )
                    previous_area = float(previous_result["pupil_area"])
                    previous_area_score = float(
                        np.exp(
                            -abs(
                                np.log(
                                    max(area, 1.0) / max(previous_area, 1.0)
                                )
                            )
                        )
                    )
                    temporal_score = (
                        0.65 * center_score + 0.35 * previous_area_score
                    )

                reference_score = 0.5
                has_reference = bool(
                    reference_result and reference_result.get("pupil_valid")
                )
                if has_reference:
                    reference_center = np.asarray(
                        [
                            float(reference_result["pupil_center_x"]) - x0,
                            float(reference_result["pupil_center_y"]) - y0,
                        ]
                    )
                    reference_distance = float(
                        np.linalg.norm(
                            np.asarray([center_x, center_y]) - reference_center
                        )
                    )
                    reference_center_score = float(
                        np.exp(-reference_distance / max(0.22 * region_scale, 1.0))
                    )
                    reference_area = float(reference_result["pupil_area"])
                    reference_area_score = float(
                        np.exp(
                            -abs(
                                np.log(max(area, 1.0) / max(reference_area, 1.0))
                            )
                            / 1.1
                        )
                    )
                    reference_score = (
                        0.70 * reference_center_score + 0.30 * reference_area_score
                    )

                if has_temporal and has_reference:
                    score = (
                        0.35 * base_score
                        + 0.25 * temporal_score
                        + 0.40 * reference_score
                    )
                elif has_temporal:
                    score = 0.50 * base_score + 0.50 * temporal_score
                elif has_reference:
                    score = 0.55 * base_score + 0.45 * reference_score
                else:
                    score = base_score
                candidates.append(
                    (
                        float(score),
                        {
                            "contour": hull,
                            "area": area,
                            "raw_area": raw_area,
                            "mask_area": mask_area,
                            "perimeter": perimeter,
                            "raw_perimeter": raw_perimeter,
                            "center_x": center_x,
                            "center_y": center_y,
                            "axis_a": axis_a,
                            "axis_b": axis_b,
                            "angle": angle,
                            "base_score": base_score,
                            "temporal_score": temporal_score,
                            "reference_score": reference_score,
                            "touches_boundary": touches_boundary,
                            "boundary_contact_fraction": boundary_contact_fraction,
                            "close_size": close_size,
                            "threshold": threshold_value,
                        },
                    )
                )
    if not candidates:
        return _invalid_pupil(
            float(np.median(threshold_values)), threshold_mode=threshold_mode
        )

    score, selected = max(candidates, key=lambda item: item[0])
    if previous_result and bool(previous_result.get("pupil_valid")):
        center_distance = float(
            np.linalg.norm(
                np.asarray([selected["center_x"] + x0, selected["center_y"] + y0])
                - np.asarray(
                    [
                        float(previous_result["pupil_center_x"]),
                        float(previous_result["pupil_center_y"]),
                    ]
                )
            )
        )
        previous_area = float(previous_result["pupil_area"])
        area_ratio = max(
            float(selected["area"]) / max(previous_area, 1.0),
            previous_area / max(float(selected["area"]), 1.0),
        )
        reference_recovery = False
        if reference_result and bool(reference_result.get("pupil_valid")):
            reference_center = np.asarray(
                [
                    float(reference_result["pupil_center_x"]),
                    float(reference_result["pupil_center_y"]),
                ]
            )
            selected_reference_distance = float(
                np.linalg.norm(
                    np.asarray(
                        [selected["center_x"] + x0, selected["center_y"] + y0]
                    )
                    - reference_center
                )
            )
            previous_reference_distance = float(
                np.linalg.norm(
                    np.asarray(
                        [
                            float(previous_result["pupil_center_x"]),
                            float(previous_result["pupil_center_y"]),
                        ]
                    )
                    - reference_center
                )
            )
            reference_recovery = bool(
                selected_reference_distance <= 0.20 * region_scale
                and selected_reference_distance + 2.0 < previous_reference_distance
            )
        if (
            center_distance > 0.25 * region_scale or area_ratio > 2.5
        ) and not reference_recovery:
            invalid = _invalid_pupil(
                float(selected["threshold"]),
                threshold_mode=threshold_mode,
                reason="implausible_temporal_jump",
            )
            invalid["pupil_candidate_count"] = len(candidates)
            return invalid
    contour = selected["contour"].copy()
    contour[:, 0, 0] += x0
    contour[:, 0, 1] += y0
    boundary = _resample_contour(contour, boundary_points)
    major = max(float(selected["axis_a"]), float(selected["axis_b"]))
    minor = min(float(selected["axis_a"]), float(selected["axis_b"]))
    confidence = float(np.clip(score, 0.0, 1.0))
    hull_correction = float(
        1.0 - float(selected["raw_area"]) / float(selected["area"])
    )
    reference_center_distance = np.nan
    reference_area_ratio = np.nan
    if reference_result and bool(reference_result.get("pupil_valid")):
        reference_center_distance = float(
            np.linalg.norm(
                np.asarray([selected["center_x"] + x0, selected["center_y"] + y0])
                - np.asarray(
                    [
                        float(reference_result["pupil_center_x"]),
                        float(reference_result["pupil_center_y"]),
                    ]
                )
            )
        )
        reference_area = float(reference_result["pupil_area"])
        reference_area_ratio = max(
            float(selected["area"]) / max(reference_area, 1.0),
            reference_area / max(float(selected["area"]), 1.0),
        )
    review_reasons: list[str] = []
    if confidence < 0.60:
        review_reasons.append("low_confidence")
    if hull_correction > 0.30:
        review_reasons.append("large_hull_correction")
    if selected["boundary_contact_fraction"] > 0.30:
        review_reasons.append("touches_eye_boundary")
    if (
        np.isfinite(reference_center_distance)
        and reference_center_distance > 0.28 * region_scale
    ):
        review_reasons.append("reference_center_deviation")
    return {
        "pupil_valid": True,
        "blink_or_invalid": False,
        "pupil_detection_status": "observed",
        "pupil_manual_anchor": False,
        "pupil_review_required": bool(review_reasons),
        "pupil_review_reason": ";".join(review_reasons),
        "pupil_threshold_mode": threshold_mode,
        "pupil_candidate_count": len(candidates),
        "pupil_center_x": x0 + float(selected["center_x"]),
        "pupil_center_y": y0 + float(selected["center_y"]),
        "pupil_major_axis": major,
        "pupil_minor_axis": minor,
        "pupil_angle_degrees": float(selected["angle"]),
        "pupil_area": float(selected["area"]),
        "pupil_raw_contour_area": float(selected["raw_area"]),
        "pupil_hull_correction_fraction": hull_correction,
        "pupil_mask_area_px": float(selected["mask_area"]),
        "pupil_perimeter_px": float(selected["perimeter"]),
        "pupil_raw_perimeter_px": float(selected["raw_perimeter"]),
        "pupil_equivalent_radius": float(np.sqrt(float(selected["area"]) / np.pi)),
        "pupil_confidence": confidence,
        "pupil_threshold": float(selected["threshold"]),
        "pupil_glare_close_size": int(selected["close_size"]),
        "pupil_touches_eye_boundary": bool(selected["touches_boundary"]),
        "pupil_eye_boundary_contact_fraction": float(
            selected["boundary_contact_fraction"]
        ),
        "pupil_reference_center_distance_px": reference_center_distance,
        "pupil_reference_area_ratio": reference_area_ratio,
        "pupil_reference_score": float(selected["reference_score"]),
        "pupil_eye_region_scale_px": float(region_scale),
        "pupil_boundary_x": boundary[:, 0].tolist(),
        "pupil_boundary_y": boundary[:, 1].tolist(),
    }


def _read_requested_frames(
    video_path: str | Path, frame_indices: Iterable[int]
) -> dict[int, np.ndarray]:
    requested = sorted(set(int(index) for index in frame_indices))
    if not requested or requested[0] < 0:
        raise ValueError("At least one non-negative frame index is required")
    remaining = set(requested)
    frames: dict[int, np.ndarray] = {}
    for frame_index, _, frame in iter_video_frames(video_path):
        if frame_index in remaining:
            frames[frame_index] = frame
            remaining.remove(frame_index)
        if not remaining or frame_index >= requested[-1]:
            break
    if remaining:
        raise IndexError(f"Video does not contain frames: {sorted(remaining)}")
    return frames


def _read_frame_contexts(
    video_path: str | Path,
    frame_indices: Iterable[int],
    context_frames: int,
    minimum_frame: int = 0,
) -> dict[int, list[np.ndarray]]:
    """Random-access short temporal contexts without decoding the complete video."""

    if context_frames < 0:
        raise ValueError("context_frames must be non-negative")
    requested = sorted(set(int(index) for index in frame_indices))
    if not requested or requested[0] < 0:
        raise ValueError("At least one non-negative frame index is required")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Unable to open video: {video_path}")
    contexts: dict[int, list[np.ndarray]] = {}
    try:
        for target in requested:
            start = max(int(minimum_frame), target - context_frames)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames: list[np.ndarray] = []
            for _ in range(start, target + 1):
                ok, frame = capture.read()
                if not ok:
                    raise IndexError(f"Video does not contain frame {target}")
                frames.append(frame)
            contexts[target] = frames
    finally:
        capture.release()
    return contexts


def pupil_audit_frame_indices(
    frame_count: int,
    fps: float,
    *,
    start_frame: int = 0,
    interval_s: float = 120.0,
    include_last: bool = False,
) -> list[int]:
    """Return regular whole-video audit frames at a requested time interval."""

    if frame_count <= 0 or fps <= 0 or interval_s <= 0:
        raise ValueError("frame_count, fps, and interval_s must be positive")
    if not 0 <= start_frame < frame_count:
        raise ValueError("start_frame must be inside the video")
    step = max(1, int(round(fps * interval_s)))
    result = list(range(int(start_frame), int(frame_count), step))
    if include_last and result[-1] != frame_count - 1:
        result.append(frame_count - 1)
    return result


def representative_invalid_frame_indices(
    behavior: pd.DataFrame, max_frames: int = 12
) -> list[int]:
    """Sample unresolved observations across the full recording for visual audit."""

    if max_frames <= 0:
        return []
    required = {"frame_index", "pupil_valid"}
    missing = required.difference(behavior.columns)
    if missing:
        raise ValueError(f"Behavior table is missing columns: {sorted(missing)}")
    invalid = behavior.loc[
        ~behavior["pupil_valid"].fillna(False), "frame_index"
    ].to_numpy(dtype=int)
    if not len(invalid):
        return []
    positions = np.linspace(0, len(invalid) - 1, min(max_frames, len(invalid)))
    return sorted(set(invalid[np.rint(positions).astype(int)].tolist()))


def representative_pupil_review_frame_indices(
    behavior: pd.DataFrame, max_frames: int = 12
) -> list[int]:
    """Sample unresolved or automatically flagged observations for visual review."""

    if max_frames <= 0:
        return []
    required = {"frame_index", "pupil_valid"}
    missing = required.difference(behavior.columns)
    if missing:
        raise ValueError(f"Behavior table is missing columns: {sorted(missing)}")
    needs_review = ~behavior["pupil_valid"].fillna(False)
    if "pupil_review_required" in behavior:
        needs_review |= behavior["pupil_review_required"].fillna(False)
    candidates = behavior.loc[needs_review, "frame_index"].to_numpy(dtype=int)
    if not len(candidates):
        return []
    positions = np.linspace(0, len(candidates) - 1, min(max_frames, len(candidates)))
    return sorted(set(candidates[np.rint(positions).astype(int)].tolist()))


def summarize_pupil_tracking(behavior: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed, recovered, and stabilization outcomes separately."""

    if "pupil_valid" not in behavior:
        raise ValueError("Behavior table has no pupil_valid column")
    total = len(behavior)
    rows: list[dict[str, Any]] = []
    for label, column in (
        ("detector pupil", "pupil_detector_valid"),
        ("quality pupil", "pupil_valid"),
        ("analysis observed pupil", "pupil_analysis_observed_valid"),
        ("analysis pupil", "pupil_analysis_valid"),
        ("eye stabilization", "eye_stabilization_valid"),
    ):
        if column in behavior:
            count = int(behavior[column].fillna(False).sum())
            rows.append(
                {
                    "category": "validity",
                    "status": label,
                    "frames": count,
                    "fraction": count / total if total else np.nan,
                }
            )
    if "pupil_detection_status" in behavior:
        for status, count in behavior["pupil_detection_status"].fillna("missing").value_counts().items():
            rows.append(
                {
                    "category": "detector",
                    "status": str(status),
                    "frames": int(count),
                    "fraction": int(count) / total if total else np.nan,
                }
            )
    if "pupil_review_required" in behavior:
        count = int(behavior["pupil_review_required"].fillna(False).sum())
        rows.append(
            {
                "category": "review",
                "status": "automatic review required",
                "frames": count,
                "fraction": count / total if total else np.nan,
            }
        )
    if "eye_stabilization_status" in behavior:
        for status, count in behavior["eye_stabilization_status"].fillna("not_started").value_counts().items():
            rows.append(
                {
                    "category": "stabilization",
                    "status": str(status),
                    "frames": int(count),
                    "fraction": int(count) / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def preview_pupil_thresholds(
    video_path: str | Path,
    eye_roi: Region,
    frame_index: int,
    thresholds: Iterable[float] | None = None,
    *,
    pupil_seed: PolygonROI | None = None,
):
    """Compare fixed segmentation thresholds on one reviewed reference frame."""

    frame = read_video_frame(video_path, frame_index)
    crop, region_mask, (x0, y0, _, _) = _region_crop_and_mask(frame, eye_roi)
    if thresholds is None:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        center = float(np.percentile(gray[region_mask > 0], 30))
        thresholds = [max(0.0, center + offset) for offset in (-12, -6, 0, 6, 12)]
    values = [float(value) for value in thresholds]
    if not values:
        raise ValueError("At least one threshold is required")
    prior = pupil_prior_from_polygon(pupil_seed) if pupil_seed is not None else None
    figure, axes = plt.subplots(1, len(values), figsize=(4 * len(values), 4), squeeze=False)
    for axis, threshold_value in zip(axes[0], values, strict=True):
        result = detect_pupil(
            frame,
            eye_roi,
            threshold=threshold_value,
            previous_result=prior,
            reference_result=prior,
        )
        axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        title = (
            f"threshold={threshold_value:.1f}\n"
            f"area={result['pupil_area']:.1f} | "
            f"confidence={result['pupil_confidence']:.2f}\n"
            f"hull correction={result['pupil_hull_correction_fraction']:.1%}"
        )
        axis.set_title(title)
        axis.set_axis_off()
        if result["pupil_valid"]:
            boundary = np.column_stack(
                [result["pupil_boundary_x"], result["pupil_boundary_y"]]
            ) - np.asarray([x0, y0])
            axis.add_patch(
                PolygonPatch(
                    boundary,
                    closed=True,
                    fill=False,
                    edgecolor="#00ff66",
                    linewidth=2,
                )
            )
    figure.tight_layout()
    return figure


def preview_pupil_detection(
    video_path: str | Path,
    eye_roi: Region,
    frame_indices: Iterable[int],
    threshold: float | None = None,
    *,
    pupil_seed: PolygonROI | None = None,
    reference_frame_index: int = 0,
    max_columns: int = 5,
    temporal_context_frames: int = 30,
):
    """Plot the segmented boundary and secondary ellipse on audited eye crops."""

    requested = list(frame_indices)
    if not requested:
        raise ValueError("At least one preview frame is required")
    if max_columns <= 0:
        raise ValueError("max_columns must be positive")
    contexts = _read_frame_contexts(
        video_path,
        requested,
        temporal_context_frames,
        minimum_frame=reference_frame_index if pupil_seed is not None else 0,
    )
    columns = min(max_columns, len(requested))
    rows = int(np.ceil(len(requested) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.8 * rows),
        squeeze=False,
    )
    for axis, frame_index in zip(axes.flat, requested, strict=False):
        reference: Mapping[str, Any] | None = (
            pupil_prior_from_polygon(pupil_seed)
            if pupil_seed is not None and frame_index >= reference_frame_index
            else None
        )
        previous = reference
        result: dict[str, Any] | None = None
        for frame in contexts[frame_index]:
            result = detect_pupil(
                frame,
                eye_roi,
                threshold=threshold,
                previous_result=previous,
                reference_result=reference,
            )
            if result["pupil_valid"]:
                previous = result
        if result is None:
            raise RuntimeError(f"No preview context was decoded for frame {frame_index}")
        crop, _, (x0, y0, _, _) = _region_crop_and_mask(frame, eye_roi)
        axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        status_text = result["pupil_detection_status"]
        if result["pupil_valid"] and result["pupil_review_required"]:
            status_text += f" | review: {result['pupil_review_reason']}"
        title = (
            f"frame {frame_index} | {status_text}\n"
            f"area={result['pupil_area']:.1f} | confidence={result['pupil_confidence']:.2f}\n"
            f"threshold={result['pupil_threshold']:.1f} ({result['pupil_threshold_mode']}) | "
            f"hull={result['pupil_hull_correction_fraction']:.1%}"
        )
        title_color = "#b42318"
        if result["pupil_valid"]:
            title_color = "#b54708" if result["pupil_review_required"] else "#176b3a"
        axis.set_title(title, color=title_color)
        axis.set_axis_off()
        if isinstance(eye_roi, PolygonROI):
            region_points = np.asarray(eye_roi.vertices) - np.asarray([x0, y0])
            axis.add_patch(
                PolygonPatch(
                    region_points,
                    closed=True,
                    fill=False,
                    edgecolor="#ffe600",
                    linewidth=1.2,
                    linestyle=":",
                )
            )
        if result["pupil_valid"]:
            boundary = np.column_stack(
                [result["pupil_boundary_x"], result["pupil_boundary_y"]]
            ) - np.asarray([x0, y0])
            axis.add_patch(
                PolygonPatch(
                    boundary,
                    closed=True,
                    fill=False,
                    edgecolor="#00ff66",
                    linewidth=2,
                )
            )
            axis.add_patch(
                Ellipse(
                    (
                        float(result["pupil_center_x"]) - x0,
                        float(result["pupil_center_y"]) - y0,
                    ),
                    width=float(result["pupil_major_axis"]),
                    height=float(result["pupil_minor_axis"]),
                    angle=float(result["pupil_angle_degrees"]),
                    fill=False,
                    edgecolor="red",
                    linewidth=1,
                    linestyle="--",
                )
            )
    for axis in axes.flat[len(requested) :]:
        axis.set_visible(False)
    figure.tight_layout()
    return figure


def preview_saved_pupil_detections(
    video_path: str | Path,
    behavior: pd.DataFrame,
    frame_indices: Iterable[int],
    *,
    eye_roi: Region | None = None,
    max_columns: int = 4,
):
    """Plot the exact contours already stored in a behavior table.

    Unlike ``preview_pupil_detection``, this function never runs the detector again.
    It is therefore the authoritative audit view for downstream analysis inputs.
    """

    requested = list(dict.fromkeys(int(index) for index in frame_indices))
    if not requested:
        raise ValueError("At least one saved preview frame is required")
    required = {"frame_index", "pupil_boundary_x", "pupil_boundary_y"}
    missing = required.difference(behavior.columns)
    if missing:
        raise ValueError(f"Behavior table is missing columns: {sorted(missing)}")
    if behavior["frame_index"].duplicated().any():
        raise ValueError("Behavior table contains duplicate frame_index values")
    indexed = behavior.set_index("frame_index", drop=False)
    missing_frames = sorted(set(requested).difference(indexed.index))
    if missing_frames:
        raise IndexError(f"Behavior table does not contain frames: {missing_frames}")
    contexts = _read_frame_contexts(video_path, requested, context_frames=0)
    columns = min(max_columns, len(requested))
    rows = int(np.ceil(len(requested) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.8 * columns, 4.3 * rows), squeeze=False
    )
    for axis, frame_index in zip(axes.flat, requested, strict=False):
        row = indexed.loc[frame_index]
        frame = contexts[frame_index][0]
        if eye_roi is None:
            crop = frame
            x0 = y0 = 0
        else:
            crop, _, (x0, y0, _, _) = _region_crop_and_mask(frame, eye_roi)
        axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axis.set_axis_off()
        detector_valid = bool(row.get("pupil_detector_valid", row.get("pupil_valid", False)))
        quality_valid = bool(row.get("pupil_quality_valid", row.get("pupil_valid", False)))
        analysis_observed = bool(
            row.get("pupil_analysis_observed_valid", quality_valid)
        )
        review_required = bool(row.get("pupil_review_required", False))
        rejected = bool(row.get("pupil_qc_rejected", False))
        boundary_x = np.asarray(row["pupil_boundary_x"], dtype=float)
        boundary_y = np.asarray(row["pupil_boundary_y"], dtype=float)
        edge_color = "#00b85a"
        if rejected or not quality_valid:
            edge_color = "#d92d20"
        elif review_required:
            edge_color = "#f79009"
        if len(boundary_x) >= 3 and len(boundary_x) == len(boundary_y):
            boundary = np.column_stack([boundary_x - x0, boundary_y - y0])
            axis.add_patch(
                PolygonPatch(
                    boundary,
                    closed=True,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=2,
                )
            )
        if eye_roi is not None and isinstance(eye_roi, PolygonROI):
            region_points = np.asarray(eye_roi.vertices) - np.asarray([x0, y0])
            axis.add_patch(
                PolygonPatch(
                    region_points,
                    closed=True,
                    fill=False,
                    edgecolor="#ffe600",
                    linewidth=1.0,
                    linestyle=":",
                )
            )
        area = float(row.get("pupil_area", np.nan))
        confidence = float(row.get("pupil_confidence", np.nan))
        status = str(row.get("pupil_detection_status", "unknown"))
        reason = str(row.get("pupil_review_reason", ""))
        reason_text = reason if reason else "none"
        reason_text = fill(reason_text.replace(";", "; "), width=34)
        axis.set_title(
            f"frame {frame_index} | {status}\n"
            f"area={area:.1f} | confidence={confidence:.2f}\n"
            f"detector={detector_valid} | analysis={analysis_observed} | "
            f"review={review_required}\n{reason_text}",
            color=edge_color,
            fontsize=8.5,
            pad=8,
        )
    for axis in axes.flat[len(requested) :]:
        axis.set_visible(False)
    figure.tight_layout(pad=1.4, h_pad=2.0, w_pad=1.5)
    return figure


class EyeLandmarkTracker:
    """Track eye-adjacent anchors while rejecting geometrically impossible drift."""

    def __init__(
        self,
        frame: np.ndarray,
        landmarks: Mapping[str, Sequence[float]],
        eye_region: Region,
        *,
        reanchor_interval_frames: int = 30,
        template_radius: int = 7,
        search_radius: int = 20,
        minimum_template_score: float = 0.40,
    ):
        if len(landmarks) < 2:
            raise ValueError("At least two landmarks are required for stabilization")
        self.names = tuple(landmarks)
        self.reference_points = np.asarray(
            list(landmarks.values()), dtype=np.float32
        ).reshape(-1, 2)
        self.points = self.reference_points.reshape(-1, 1, 2).copy()
        if reanchor_interval_frames < 0:
            raise ValueError("reanchor_interval_frames must be non-negative")
        self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.reference_gray = self.previous_gray.copy()
        self.reference_vertices = _region_vertices(eye_region)
        self.vertices = self.reference_vertices.copy()
        self.valid = np.ones(len(self.names), dtype=bool)
        self.frame_shape = frame.shape[:2]
        self.transform = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.status = "reference"
        self.scale = 1.0
        self.rotation_degrees = 0.0
        self.translation = np.zeros(2, dtype=float)
        self.forward_backward_error = 0.0
        self.rejected_updates = 0
        self.region_scale = max(np.ptp(self.reference_vertices, axis=0).max(), 1.0)
        self.reanchor_interval_frames = int(reanchor_interval_frames)
        self.template_radius = int(template_radius)
        self.search_radius = int(search_radius)
        self.minimum_template_score = float(minimum_template_score)
        self.update_count = 0
        self.reanchor_count = 0
        self.reanchor_score = np.nan
        self.reference_templates: list[np.ndarray | None] = []
        height, width = self.reference_gray.shape
        for x, y in self.reference_points:
            x0 = int(round(x)) - self.template_radius
            y0 = int(round(y)) - self.template_radius
            x1 = x0 + 2 * self.template_radius + 1
            y1 = y0 + 2 * self.template_radius + 1
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                self.reference_templates.append(None)
            else:
                self.reference_templates.append(
                    self.reference_gray[y0:y1, x0:x1].copy()
                )

    def _reject(self, gray: np.ndarray, reason: str, fb_error: float = np.nan) -> None:
        self.valid[:] = False
        self.status = reason
        self.forward_backward_error = float(fb_error)
        self.rejected_updates += 1
        self.previous_gray = gray
        homogeneous = np.column_stack(
            [self.reference_points, np.ones(len(self.reference_points))]
        )
        self.points = (homogeneous @ self.transform.T).astype(np.float32).reshape(-1, 1, 2)

    def _evaluate_transform(
        self,
        transform: np.ndarray,
        valid: np.ndarray,
        candidate: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[bool, float, float, float, np.ndarray, np.ndarray]:
        linear = transform[:, :2]
        determinant = float(np.linalg.det(linear))
        scale = float(np.sqrt(abs(determinant)))
        rotation_degrees = float(
            np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))
        )
        reference_homogeneous = np.column_stack(
            [self.reference_points, np.ones(len(self.reference_points))]
        )
        fitted_points = reference_homogeneous @ transform.T
        residual = float(
            np.median(np.linalg.norm(fitted_points[valid] - candidate[valid], axis=1))
        )
        reference_center = self.reference_vertices.mean(axis=0)
        transformed_center = np.append(reference_center, 1.0) @ transform.T
        translation = transformed_center - reference_center
        vertices_homogeneous = np.column_stack(
            [self.reference_vertices, np.ones(len(self.reference_vertices))]
        )
        fitted_vertices = vertices_homogeneous @ transform.T
        inside_frame = bool(
            (fitted_vertices[:, 0] >= 0).all()
            and (fitted_vertices[:, 0] < width).all()
            and (fitted_vertices[:, 1] >= 0).all()
            and (fitted_vertices[:, 1] < height).all()
        )
        plausible = bool(
            determinant > 0
            and 0.90 <= scale <= 1.10
            and abs(rotation_degrees) <= 10.0
            and np.linalg.norm(translation) <= 0.25 * self.region_scale
            and residual <= 3.0
            and inside_frame
        )
        return (
            plausible,
            scale,
            rotation_degrees,
            residual,
            translation,
            fitted_vertices,
        )

    def _template_reanchor(
        self, gray: np.ndarray, predicted_transform: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        homogeneous = np.column_stack(
            [self.reference_points, np.ones(len(self.reference_points))]
        )
        predicted = homogeneous @ predicted_transform.T
        matched = np.full_like(self.reference_points, np.nan, dtype=np.float32)
        valid = np.zeros(len(self.reference_points), dtype=bool)
        scores = np.full(len(self.reference_points), np.nan, dtype=float)
        height, width = gray.shape
        for index, (template, point) in enumerate(
            zip(self.reference_templates, predicted, strict=True)
        ):
            if template is None:
                continue
            template_height, template_width = template.shape
            center_x, center_y = np.rint(point).astype(int)
            search_x0 = max(
                0, center_x - self.search_radius - template_width // 2
            )
            search_y0 = max(
                0, center_y - self.search_radius - template_height // 2
            )
            search_x1 = min(
                width, center_x + self.search_radius + template_width // 2 + 1
            )
            search_y1 = min(
                height, center_y + self.search_radius + template_height // 2 + 1
            )
            search = gray[search_y0:search_y1, search_x0:search_x1]
            if (
                search.shape[0] < template_height
                or search.shape[1] < template_width
            ):
                continue
            response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(response)
            if not np.isfinite(score) or score < self.minimum_template_score:
                continue
            matched[index] = (
                search_x0 + location[0] + (template_width - 1) / 2,
                search_y0 + location[1] + (template_height - 1) / 2,
            )
            valid[index] = True
            scores[index] = score
        if valid.sum() < 2:
            return None
        transform, _ = cv2.estimateAffinePartial2D(
            self.reference_points[valid], matched[valid], method=cv2.LMEDS
        )
        if transform is None:
            return None
        return transform, valid, matched, float(np.nanmedian(scores[valid]))

    def update(self, frame: np.ndarray) -> None:
        self.update_count += 1
        used_reanchor = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            self._reject(gray, "lk_forward_failed")
            return
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.previous_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if back_points is None or back_status is None:
            self._reject(gray, "lk_backward_failed")
            return
        height, width = gray.shape
        candidate = next_points.reshape(-1, 2)
        backward = back_points.reshape(-1, 2)
        previous = self.points.reshape(-1, 2)
        forward_backward_error = np.linalg.norm(backward - previous, axis=1)
        error_values = (
            errors.reshape(-1) if errors is not None else np.zeros(len(candidate), dtype=float)
        )
        forward_valid = (
            status.reshape(-1).astype(bool)
            & np.isfinite(candidate).all(axis=1)
            & (candidate[:, 0] >= 0)
            & (candidate[:, 0] < width)
            & (candidate[:, 1] >= 0)
            & (candidate[:, 1] < height)
            & (error_values < 40)
        )
        backward_consistent = (
            forward_valid
            & back_status.reshape(-1).astype(bool)
            & np.isfinite(forward_backward_error)
            & (forward_backward_error < 1.5)
        )
        # Some codecs/textures make the reverse LK status conservative even when all
        # forward landmarks agree on one plausible transform. Geometry checks below
        # remain mandatory in that fallback case.
        valid = backward_consistent if backward_consistent.sum() >= 2 else forward_valid
        median_fb = (
            float(np.median(forward_backward_error[backward_consistent]))
            if backward_consistent.any()
            else np.nan
        )
        if valid.sum() < 2:
            self._reject(gray, "insufficient_consistent_landmarks", median_fb)
            return

        transform, _ = cv2.estimateAffinePartial2D(
            self.reference_points[valid], candidate[valid], method=cv2.LMEDS
        )
        if transform is None:
            self._reject(gray, "affine_fit_failed", median_fb)
            return
        evaluation = self._evaluate_transform(
            transform, valid, candidate, width, height
        )
        plausible, scale, rotation_degrees, _, translation, fitted_vertices = (
            evaluation
        )
        should_reanchor = bool(
            self.reanchor_interval_frames
            and self.update_count % self.reanchor_interval_frames == 0
        )
        reanchored = None
        if should_reanchor or not plausible:
            predicted_transform = transform if plausible else self.transform
            reanchored = self._template_reanchor(gray, predicted_transform)
        if reanchored is not None:
            candidate_transform, template_valid, template_points, template_score = (
                reanchored
            )
            template_evaluation = self._evaluate_transform(
                candidate_transform,
                template_valid,
                template_points,
                width,
                height,
            )
            if template_evaluation[0]:
                transform = candidate_transform
                valid = template_valid
                candidate = template_points
                (
                    plausible,
                    scale,
                    rotation_degrees,
                    _,
                    translation,
                    fitted_vertices,
                ) = template_evaluation
                self.reanchor_count += 1
                self.reanchor_score = template_score
                used_reanchor = True
        if not plausible:
            self._reject(gray, "implausible_landmark_geometry", median_fb)
            return

        reference_homogeneous = np.column_stack(
            [self.reference_points, np.ones(len(self.reference_points))]
        )
        fitted_points = reference_homogeneous @ transform.T

        self.transform = transform
        self.vertices = fitted_vertices
        self.points = fitted_points.astype(np.float32).reshape(-1, 1, 2)
        self.valid = valid
        self.status = "accepted_reanchored" if used_reanchor else "accepted"
        self.scale = scale
        self.rotation_degrees = rotation_degrees
        self.translation = translation
        self.forward_backward_error = median_fb
        self.previous_gray = gray

    @property
    def eye_region(self) -> PolygonROI:
        return PolygonROI(tuple(map(tuple, self.vertices)))

    def transform_pupil_prior(
        self, prior: Mapping[str, Any]
    ) -> dict[str, float | bool]:
        """Map the reference pupil center and area into the current eye frame."""

        center = np.asarray(
            [float(prior["pupil_center_x"]), float(prior["pupil_center_y"]), 1.0]
        )
        transformed_center = center @ self.transform.T
        area_scale = abs(float(np.linalg.det(self.transform[:, :2])))
        return {
            "pupil_valid": bool(prior.get("pupil_valid")),
            "pupil_center_x": float(transformed_center[0]),
            "pupil_center_y": float(transformed_center[1]),
            "pupil_area": float(prior["pupil_area"]) * area_scale,
        }

    def measurements(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "eye_stabilization_valid_fraction": float(self.valid.mean()),
            "eye_stabilization_valid": bool(self.valid.sum() >= 2),
            "eye_stabilization_status": self.status,
            "eye_stabilization_scale": self.scale,
            "eye_stabilization_rotation_degrees": self.rotation_degrees,
            "eye_stabilization_translation_x": float(self.translation[0]),
            "eye_stabilization_translation_y": float(self.translation[1]),
            "eye_stabilization_fb_error": self.forward_backward_error,
            "eye_stabilization_rejected_updates": self.rejected_updates,
            "eye_stabilization_reanchor_count": self.reanchor_count,
            "eye_stabilization_reanchor_score": self.reanchor_score,
        }
        points = self.points.reshape(-1, 2)
        for name, point, valid in zip(self.names, points, self.valid, strict=True):
            safe_name = "".join(character if character.isalnum() else "_" for character in name)
            result[f"eye_landmark_{safe_name}_x"] = float(point[0]) if valid else np.nan
            result[f"eye_landmark_{safe_name}_y"] = float(point[1]) if valid else np.nan
            result[f"eye_landmark_{safe_name}_valid"] = bool(valid)
        return result


def _region_vertices(region: Region) -> np.ndarray:
    if isinstance(region, PolygonROI):
        return np.asarray(region.vertices, dtype=float)
    return np.asarray(
        [
            [region.x, region.y],
            [region.x + region.width - 1, region.y],
            [region.x + region.width - 1, region.y + region.height - 1],
            [region.x, region.y + region.height - 1],
        ],
        dtype=float,
    )


def interpolate_short_nan_gaps(values: Iterable[float], max_gap_frames: int) -> np.ndarray:
    values_array = np.asarray(list(values), dtype=float)
    result = values_array.copy()
    missing = np.isnan(values_array)
    index = 0
    while index < len(result):
        if not missing[index]:
            index += 1
            continue
        start = index
        while index < len(result) and missing[index]:
            index += 1
        stop = index
        gap = stop - start
        if gap <= max_gap_frames and start > 0 and stop < len(result):
            result[start:stop] = np.linspace(result[start - 1], result[stop], gap + 2)[1:-1]
    return result


def apply_pupil_quality_control(
    behavior: pd.DataFrame,
    frame_rate_hz: float,
    *,
    bidirectional_window_s: float = 0.5,
) -> pd.DataFrame:
    """Separate detector output from conservative analysis eligibility.

    The centered window uses both earlier and later observations. Detector geometry
    is retained even when a frame is rejected so every decision remains auditable.
    """

    if frame_rate_hz <= 0 or bidirectional_window_s <= 0:
        raise ValueError("QC frame rate and window must be positive")
    required = {
        "pupil_valid",
        "pupil_area",
        "pupil_center_x",
        "pupil_center_y",
        "pupil_hull_correction_fraction",
        "pupil_review_required",
        "pupil_review_reason",
    }
    missing = required.difference(behavior.columns)
    if missing:
        raise ValueError(f"Behavior table is missing QC columns: {sorted(missing)}")
    result = behavior.copy()
    detector_valid = result["pupil_valid"].fillna(False).to_numpy(dtype=bool)
    result["pupil_detector_valid"] = detector_valid
    result["pupil_detector_status"] = result["pupil_detection_status"].astype(str)
    manual_anchor = result.get(
        "pupil_manual_anchor", pd.Series(False, index=result.index)
    ).fillna(False).to_numpy(dtype=bool)

    window = max(5, int(round(frame_rate_hz * bidirectional_window_s)))
    if window % 2 == 0:
        window += 1
    minimum = max(3, window // 3)
    area = result["pupil_area"].where(detector_valid)
    center_x = result["pupil_center_x"].where(detector_valid)
    center_y = result["pupil_center_y"].where(detector_valid)
    local_area = area.rolling(window, center=True, min_periods=minimum).median()
    local_center_x = center_x.rolling(
        window, center=True, min_periods=minimum
    ).median()
    local_center_y = center_y.rolling(
        window, center=True, min_periods=minimum
    ).median()
    safe_area = area.clip(lower=1.0)
    safe_local_area = local_area.clip(lower=1.0)
    area_ratio = np.maximum(
        safe_area / safe_local_area, safe_local_area / safe_area
    )
    center_deviation = np.hypot(
        center_x - local_center_x, center_y - local_center_y
    )
    result["pupil_bidirectional_area_ratio"] = area_ratio
    result["pupil_bidirectional_center_deviation_px"] = center_deviation

    hull = result["pupil_hull_correction_fraction"].to_numpy(dtype=float)
    reference_ratio = result.get(
        "pupil_reference_area_ratio", pd.Series(np.nan, index=result.index)
    ).to_numpy(dtype=float)
    reference_distance = result.get(
        "pupil_reference_center_distance_px", pd.Series(np.nan, index=result.index)
    ).to_numpy(dtype=float)
    region_scale = result.get(
        "pupil_eye_region_scale_px", pd.Series(np.nan, index=result.index)
    ).to_numpy(dtype=float)
    boundary_contact = result.get(
        "pupil_eye_boundary_contact_fraction",
        result.get(
            "pupil_touches_eye_boundary", pd.Series(False, index=result.index)
        ).astype(float),
    ).fillna(0.0).to_numpy(dtype=float)
    area_ratio_values = area_ratio.to_numpy(dtype=float)
    center_values = np.asarray(center_deviation, dtype=float)

    review_masks = {
        "bidirectional_area_deviation": area_ratio_values > 1.5,
        "bidirectional_center_deviation": center_values > 0.14 * region_scale,
        "large_hull_correction": hull > 0.30,
        "reference_center_deviation": reference_distance > 0.28 * region_scale,
        "touches_eye_boundary": boundary_contact > 0.30,
    }
    reject_masks = {
        "bidirectional_area_outlier": area_ratio_values > 2.0,
        "bidirectional_center_outlier": center_values > 0.24 * region_scale,
        "extreme_hull_correction": hull > 0.45,
        "reference_area_and_shape_deviation": (reference_ratio > 3.0)
        & (hull > 0.25),
        "temporal_and_shape_deviation": (area_ratio_values > 1.75)
        & (hull > 0.20),
        "touches_eye_boundary": boundary_contact > 0.40,
    }
    review_required = (
        result["pupil_review_required"].fillna(False).to_numpy(bool).copy()
    )
    qc_rejected = np.zeros(len(result), dtype=bool)
    reasons = result["pupil_review_reason"].fillna("").astype(str).tolist()

    def append_reason(index: int, reason: str) -> None:
        values = [value for value in reasons[index].split(";") if value]
        if reason not in values:
            values.append(reason)
        reasons[index] = ";".join(values)

    for reason, mask in review_masks.items():
        selected = detector_valid & ~manual_anchor & np.nan_to_num(mask, nan=False)
        review_required[selected] = True
        for index in np.flatnonzero(selected):
            append_reason(int(index), reason)
    for reason, mask in reject_masks.items():
        selected = detector_valid & ~manual_anchor & np.nan_to_num(mask, nan=False)
        qc_rejected[selected] = True
        review_required[selected] = True
        for index in np.flatnonzero(selected):
            append_reason(int(index), reason)

    quality_valid = detector_valid & ~qc_rejected
    result["pupil_qc_rejected"] = qc_rejected
    result["pupil_quality_valid"] = quality_valid
    result["pupil_valid"] = quality_valid
    result["pupil_review_required"] = review_required
    result["pupil_review_reason"] = reasons
    result.loc[qc_rejected, "pupil_detection_status"] = "quality_rejected"
    return result


def extract_behavior(
    session_id: str,
    camera: str,
    video_path: str | Path,
    transform: TimeTransform,
    eye_roi: Region,
    movement_roi: Region,
    train_first_onset_s: float,
    max_interpolation_gap_s: float = 0.5,
    pupil_threshold: float | None = None,
    *,
    pupil_seed: PolygonROI | None = None,
    eye_landmarks: Mapping[str, Sequence[float]] | None = None,
    landmark_reference_frame: int = 0,
    pupil_boundary_points: int = 32,
    landmark_reanchor_interval_frames: int = 30,
    pupil_qc_window_s: float = 0.5,
    exclude_review_from_analysis: bool = True,
    use_stabilized_eye_roi: bool = False,
    manual_pupil_anchors: Mapping[str | int, Mapping[str, Any] | PolygonROI]
    | None = None,
    max_manual_pupil_anchors: int = 20,
    manual_anchor_influence_frames: int = 60,
) -> pd.DataFrame:
    """Extract PTS-aligned pupil contours, stabilization landmarks, and movement."""

    rows: list[dict[str, object]] = []
    previous_movement: np.ndarray | None = None
    previous_pupil: Mapping[str, Any] | None = None
    missed_pupil_frames = 0
    landmark_tracker: EyeLandmarkTracker | None = None
    landmark_names = tuple(eye_landmarks or {})
    reference_pupil = (
        pupil_prior_from_polygon(pupil_seed) if pupil_seed is not None else None
    )
    manual_polygons: dict[int, PolygonROI] = {}
    for frame_key, value in (manual_pupil_anchors or {}).items():
        polygon = value if isinstance(value, PolygonROI) else region_from_mapping(value)
        if not isinstance(polygon, PolygonROI):
            raise ValueError("Manual pupil anchors must be polygon regions")
        manual_polygons[int(frame_key)] = polygon
    if max_manual_pupil_anchors <= 0:
        raise ValueError("max_manual_pupil_anchors must be positive")
    if len(manual_polygons) > max_manual_pupil_anchors:
        raise ValueError(
            f"At most {max_manual_pupil_anchors} manual pupil anchors are allowed"
        )
    if manual_anchor_influence_frames < 0:
        raise ValueError("manual_anchor_influence_frames must be non-negative")
    manual_priors = {
        frame_index: pupil_prior_from_polygon(polygon)
        for frame_index, polygon in manual_polygons.items()
    }
    for frame_index, pts_s, frame in iter_video_frames(video_path):
        if pupil_seed is not None and frame_index == landmark_reference_frame:
            previous_pupil = reference_pupil
            missed_pupil_frames = 0
        if eye_landmarks and frame_index == landmark_reference_frame:
            landmark_tracker = EyeLandmarkTracker(
                frame,
                eye_landmarks,
                eye_roi,
                reanchor_interval_frames=landmark_reanchor_interval_frames,
            )
        elif landmark_tracker is not None:
            landmark_tracker.update(frame)

        current_eye_roi = (
            landmark_tracker.eye_region
            if landmark_tracker is not None and use_stabilized_eye_roi
            else eye_roi
        )
        stabilization: dict[str, Any] = {}
        if landmark_tracker:
            stabilization = landmark_tracker.measurements()
        elif landmark_names:
            stabilization = {
                "eye_stabilization_valid_fraction": 0.0,
                "eye_stabilization_valid": False,
                "eye_stabilization_status": "not_started",
                "eye_stabilization_scale": np.nan,
                "eye_stabilization_rotation_degrees": np.nan,
                "eye_stabilization_translation_x": np.nan,
                "eye_stabilization_translation_y": np.nan,
                "eye_stabilization_fb_error": np.nan,
                "eye_stabilization_rejected_updates": 0,
                "eye_stabilization_reanchor_count": 0,
                "eye_stabilization_reanchor_score": np.nan,
            }
            for name in landmark_names:
                safe_name = "".join(
                    character if character.isalnum() else "_" for character in name
                )
                stabilization[f"eye_landmark_{safe_name}_x"] = np.nan
                stabilization[f"eye_landmark_{safe_name}_y"] = np.nan
                stabilization[f"eye_landmark_{safe_name}_valid"] = False

        current_reference_pupil = reference_pupil
        if (
            landmark_tracker is not None
            and reference_pupil is not None
            and use_stabilized_eye_roi
        ):
            current_reference_pupil = landmark_tracker.transform_pupil_prior(
                reference_pupil
            )
        nearby_manual_frame: int | None = None
        if manual_priors:
            candidate_frame = min(
                manual_priors, key=lambda anchor: abs(anchor - frame_index)
            )
            if abs(candidate_frame - frame_index) <= manual_anchor_influence_frames:
                nearby_manual_frame = candidate_frame
                current_reference_pupil = manual_priors[candidate_frame]
        if frame_index in manual_polygons:
            pupil = pupil_result_from_polygon(
                frame,
                current_eye_roi,
                manual_polygons[frame_index],
                boundary_points=pupil_boundary_points,
            )
            pupil["pupil_manual_anchor_nearby_frame"] = frame_index
        else:
            pupil = detect_pupil(
                frame,
                current_eye_roi,
                threshold=pupil_threshold,
                previous_result=previous_pupil,
                reference_result=current_reference_pupil,
                boundary_points=pupil_boundary_points,
            )
            pupil["pupil_manual_anchor_nearby_frame"] = nearby_manual_frame
        if pupil["pupil_valid"]:
            previous_pupil = pupil
            missed_pupil_frames = 0
        else:
            missed_pupil_frames += 1
            if missed_pupil_frames > 15:
                previous_pupil = None

        movement_crop, movement_mask, _ = _region_crop_and_mask(frame, movement_roi)
        movement = cv2.cvtColor(movement_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        valid_movement = movement_mask > 0
        mean_intensity = float(movement[valid_movement].mean())
        if previous_movement is None:
            signed_difference = 0.0
            absolute_difference = 0.0
        else:
            delta = movement - previous_movement
            signed_difference = float(delta[valid_movement].mean())
            absolute_difference = float(np.abs(delta[valid_movement]).mean())
        previous_movement = movement
        rows.append(
            {
                "session_id": session_id,
                "camera": camera,
                "frame_index": frame_index,
                "pts_s": pts_s,
                "t_session_s": transform.video_to_session(pts_s),
                **pupil,
                **stabilization,
                "movement_mean_intensity": mean_intensity,
                "movement_signed_difference": signed_difference,
                "movement_abs_difference": absolute_difference,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    valid_times = np.diff(result["t_session_s"].to_numpy(dtype=float))
    frame_rate = 1.0 / float(np.median(valid_times[valid_times > 0]))
    result = apply_pupil_quality_control(
        result,
        frame_rate,
        bidirectional_window_s=pupil_qc_window_s,
    )
    max_gap_frames = max(1, int(round(max_interpolation_gap_s * frame_rate)))
    quality_valid = result["pupil_valid"].fillna(False).to_numpy(dtype=bool)
    review_required = (
        result["pupil_review_required"].fillna(False).to_numpy(dtype=bool)
    )
    observed_valid = quality_valid & (
        ~review_required if exclude_review_from_analysis else True
    )
    result["pupil_analysis_observed_valid"] = observed_valid
    radius = result["pupil_equivalent_radius"].where(observed_valid).to_numpy(dtype=float)
    area = result["pupil_area"].where(observed_valid).to_numpy(dtype=float)
    result["pupil_equivalent_radius_interpolated"] = interpolate_short_nan_gaps(
        radius, max_gap_frames
    )
    result["pupil_area_interpolated"] = interpolate_short_nan_gaps(area, max_gap_frames)
    recovered = (
        ~observed_valid
        & result["pupil_equivalent_radius_interpolated"].notna().to_numpy()
        & result["pupil_area_interpolated"].notna().to_numpy()
    )
    result["pupil_recovered"] = recovered
    result["pupil_analysis_valid"] = observed_valid | recovered
    result["pupil_analysis_source"] = np.select(
        [
            observed_valid,
            recovered,
            quality_valid & review_required & exclude_review_from_analysis,
        ],
        ["observed", "short_gap_interpolation", "excluded_review"],
        default="unresolved",
    )
    baseline_mask = (
        (result["t_session_s"] >= transform.segment_start_session_s)
        & (result["t_session_s"] < train_first_onset_s)
        & result["pupil_analysis_valid"]
    )
    baseline_radius = result.loc[baseline_mask, "pupil_equivalent_radius_interpolated"]
    baseline_area = result.loc[baseline_mask, "pupil_area_interpolated"]
    baseline_radius_value = float(baseline_radius.median()) if len(baseline_radius) else np.nan
    baseline_area_value = float(baseline_area.median()) if len(baseline_area) else np.nan
    result["pupil_baseline_radius"] = baseline_radius_value
    result["pupil_baseline_area"] = baseline_area_value
    result["pupil_radius_normalized"] = (
        result["pupil_equivalent_radius_interpolated"] / baseline_radius_value
        if np.isfinite(baseline_radius_value) and baseline_radius_value > 0
        else np.nan
    )
    result["pupil_area_normalized"] = (
        result["pupil_area_interpolated"] / baseline_area_value
        if np.isfinite(baseline_area_value) and baseline_area_value > 0
        else np.nan
    )
    return result
