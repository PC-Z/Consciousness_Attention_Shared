from __future__ import annotations

from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np
import pandas as pd

from .models import TimeTransform


def probe_video(path: str | Path) -> dict[str, object]:
    """Probe a video and report decode failures without modifying the source."""

    source = Path(path).resolve()
    try:
        with av.open(str(source), mode="r") as container:
            stream = container.streams.video[0]
            rate = float(stream.average_rate) if stream.average_rate else None
            duration_s = (
                float(stream.duration * stream.time_base) if stream.duration is not None else None
            )
            return {
                "path": str(source),
                "status": "ok",
                "codec": stream.codec_context.name,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "average_rate_hz": rate,
                "frame_count_header": int(stream.frames) if stream.frames else None,
                "duration_s": duration_s,
                "time_base": str(stream.time_base),
            }
    except Exception as exc:  # PyAV raises format-specific exception subclasses.
        return {"path": str(source), "status": "unreadable", "error": str(exc)}


def iter_video_frames(path: str | Path) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield frame index, exact PTS seconds, and BGR image."""

    with av.open(str(Path(path)), mode="r") as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"Frame {index} in {path} has no PTS")
            pts_s = float(frame.pts * stream.time_base)
            yield index, pts_s, frame.to_ndarray(format="bgr24")


def build_video_frame_table(
    session_id: str,
    camera: str,
    path: str | Path,
    transform: TimeTransform,
) -> pd.DataFrame:
    rows = [
        {
            "session_id": session_id,
            "camera": camera,
            "frame_index": frame_index,
            "pts_s": pts_s,
            "t_session_s": transform.video_to_session(pts_s),
        }
        for frame_index, pts_s, _ in iter_video_frames(path)
    ]
    return pd.DataFrame(rows)


def reference_video_runs(
    path: str | Path,
    scan_stop_s: float = 90.0,
    stripe_std_threshold: float = 10.0,
) -> pd.DataFrame:
    """Classify early reference-video frames as uniform gray or stripe."""

    runs: list[dict[str, object]] = []
    current_state: str | None = None
    start_index = 0
    start_pts = 0.0
    previous_pts = 0.0
    frame_step = None
    for index, pts_s, frame in iter_video_frames(path):
        if pts_s > scan_stop_s:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        roi = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
        state = "stripe" if float(np.std(roi)) > stripe_std_threshold else "gray"
        if index:
            frame_step = pts_s - previous_pts
        if current_state is None:
            current_state, start_index, start_pts = state, index, pts_s
        elif state != current_state:
            runs.append(
                {
                    "state": current_state,
                    "start_frame": start_index,
                    "stop_frame": index,
                    "start_s": start_pts,
                    "stop_s": pts_s,
                    "duration_s": pts_s - start_pts,
                }
            )
            current_state, start_index, start_pts = state, index, pts_s
        previous_pts = pts_s
        if len(runs) >= 16:
            break
    if current_state is not None:
        stop_s = previous_pts + (frame_step or 0.0)
        runs.append(
            {
                "state": current_state,
                "start_frame": start_index,
                "stop_frame": index + 1,
                "start_s": start_pts,
                "stop_s": stop_s,
                "duration_s": stop_s - start_pts,
            }
        )
    return pd.DataFrame(runs)


def read_video_frame(path: str | Path, frame_index: int = 0) -> np.ndarray:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    for index, _, frame in iter_video_frames(path):
        if index == frame_index:
            return frame
    raise IndexError(f"Video has no frame {frame_index}: {path}")
