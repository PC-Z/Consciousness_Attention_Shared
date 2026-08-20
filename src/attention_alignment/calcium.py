from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

from .errors import SourceFormatError
from .models import RecordingSegment, TimeTransform


def inspect_calcium_source(
    path: str | Path,
    required_datasets: Iterable[str] = ("whole_trace_dff",),
) -> dict[str, object]:
    """Read HDF5 metadata without loading any calcium signal matrix."""

    source = Path(path).resolve()
    datasets: dict[str, dict[str, object]] = {}
    with h5py.File(source, "r") as handle:
        for name in handle.keys():
            item = handle[name]
            if isinstance(item, h5py.Dataset):
                datasets[name] = {"shape": list(item.shape), "dtype": str(item.dtype)}
        missing = [name for name in required_datasets if name not in handle]
        if missing:
            raise SourceFormatError(f"Missing calcium datasets in {source}: {missing}")
        frame_count = int(handle["whole_trace_dff"].shape[0])
        cell_count = int(handle["whole_trace_dff"].shape[1])
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "frame_count": frame_count,
        "cell_count": cell_count,
        "datasets": datasets,
    }


def build_calcium_frame_table(
    session_id: str,
    segment: RecordingSegment,
    transform: TimeTransform,
    source_info: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Bind HDF5 frame N to channel-1 pulse N in the selected 02 segment."""

    pulses = segment.channel("1")
    frame_count = int(source_info["frame_count"])
    if len(pulses) < frame_count:
        raise SourceFormatError(
            f"Calcium matrix has {frame_count} frames but only {len(pulses)} channel-1 pulses"
        )
    rows = [
        {
            "session_id": session_id,
            "frame_index": index,
            "hdf5_frame_index": index,
            "source_marker_row": marker.row_index,
            "t_device_s": marker.relative_time_s,
            "t_calcium_anchor_s": transform.device_to_calcium_anchor(marker.relative_time_s),
            "t_session_s": transform.device_to_session(marker.relative_time_s),
            "abs_tick_100ns": marker.abs_tick_100ns,
        }
        for index, marker in enumerate(pulses[:frame_count])
    ]
    qc = {
        "channel1_pulse_count": len(pulses),
        "calcium_frame_count": frame_count,
        "unpaired_tail_pulses": len(pulses) - frame_count,
    }
    return pd.DataFrame(rows), qc


def read_calcium_window(
    source_path: str | Path,
    frame_table: pd.DataFrame,
    start_s: float,
    stop_s: float,
    cells: Iterable[int] | None = None,
    kind: str = "whole_trace_dff",
) -> dict[str, np.ndarray]:
    """Lazily read a time window from a MATLAB v7.3 HDF5 calcium dataset."""

    selected = frame_table.loc[
        (frame_table["t_session_s"] >= start_s) & (frame_table["t_session_s"] < stop_s)
    ]
    if selected.empty:
        return {
            "t_session_s": np.array([], dtype=float),
            "frame_index": np.array([], dtype=np.int64),
            "cell_index": np.array([], dtype=np.int64),
            "values": np.empty((0, 0), dtype=np.float32),
        }
    frame_indices = selected["hdf5_frame_index"].to_numpy(dtype=np.int64)
    first, last = int(frame_indices[0]), int(frame_indices[-1]) + 1
    with h5py.File(Path(source_path), "r") as handle:
        if kind not in handle:
            raise SourceFormatError(f"Dataset {kind!r} is absent from {source_path}")
        dataset = handle[kind]
        if dataset.ndim != 2:
            raise SourceFormatError(f"Calcium dataset {kind!r} must be 2-D, got {dataset.shape}")
        if cells is None:
            cell_indices = np.arange(dataset.shape[1], dtype=np.int64)
            values = dataset[first:last, :]
        else:
            requested = np.asarray(list(cells), dtype=np.int64)
            if np.any(requested < 0) or np.any(requested >= dataset.shape[1]):
                raise IndexError("Requested calcium cell index is out of range")
            order = np.argsort(requested)
            sorted_cells = requested[order]
            sorted_values = dataset[first:last, sorted_cells]
            inverse = np.argsort(order)
            values = sorted_values[:, inverse]
            cell_indices = requested
    return {
        "t_session_s": selected["t_session_s"].to_numpy(dtype=float),
        "frame_index": frame_indices,
        "cell_index": cell_indices,
        "values": np.asarray(values),
    }
