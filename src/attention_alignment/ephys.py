from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import SourceFormatError
from .models import TimeTransform


EPHYS_DTYPE = np.dtype(
    [
        ("tick", "<i8"),
        ("EEG1", "<f4"),
        ("EEG2", "<f4"),
        ("EMG", "<f4"),
        ("Activ", "<f4"),
    ]
)


def read_ephys_metadata(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    labels = [item["Label"] for item in raw.get("Signals", [])]
    rates = [int(item["Sample rate"]) for item in raw.get("Signals", [])]
    if labels != ["EEG1", "EEG2", "EMG", "Activ"] or len(set(rates)) != 1:
        raise SourceFormatError(f"Unexpected ephys metadata in {path}: {labels}, {rates}")
    return {"labels": labels, "sample_rate_hz": rates[0], "raw": raw}


def discover_dat_segments(path: str | Path) -> list[dict[str, int]]:
    """Find DAT segments at non-increasing tick boundaries without copying samples."""

    source = Path(path).resolve()
    if source.stat().st_size % EPHYS_DTYPE.itemsize:
        raise SourceFormatError(
            f"DAT size is not divisible by {EPHYS_DTYPE.itemsize} bytes: {source}"
        )
    records = np.memmap(source, mode="r", dtype=EPHYS_DTYPE)
    reset_after = np.flatnonzero(np.diff(records["tick"]) <= 0) + 1
    starts = np.concatenate(([0], reset_after))
    stops = np.concatenate((reset_after, [len(records)]))
    return [
        {
            "segment_index": index,
            "start_record": int(start),
            "stop_record": int(stop),
            "record_count": int(stop - start),
            "first_tick": int(records["tick"][start]),
            "last_tick": int(records["tick"][stop - 1]),
        }
        for index, (start, stop) in enumerate(zip(starts, stops, strict=True))
    ]


def build_ephys_manifest(
    dat_path: str | Path,
    metadata_path: str | Path,
    selected_segment_index: int,
    transform: TimeTransform,
    tick_hz: int = 10_000_000,
) -> dict[str, object]:
    metadata = read_ephys_metadata(metadata_path)
    segments = discover_dat_segments(dat_path)
    if selected_segment_index >= len(segments):
        selected_segment_index = max(
            range(len(segments)), key=lambda index: segments[index]["record_count"]
        )
    return {
        "path": str(Path(dat_path).resolve()),
        "metadata_path": str(Path(metadata_path).resolve()),
        "dtype": EPHYS_DTYPE.descr,
        "record_size_bytes": EPHYS_DTYPE.itemsize,
        "tick_hz": tick_hz,
        "sample_rate_hz": metadata["sample_rate_hz"],
        "channels": metadata["labels"],
        "segments": segments,
        "selected_segment_index": selected_segment_index,
        "time_offset_session_s": transform.anchor_session_s - transform.anchor_device_s,
    }


def read_ephys_window(
    manifest: dict[str, object],
    start_s: float,
    stop_s: float,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    """Read only the requested 02 DAT time window using a structured memmap."""

    allowed = list(manifest["channels"])
    selected_channels = allowed if channels is None else channels
    unknown = set(selected_channels) - set(allowed)
    if unknown:
        raise KeyError(f"Unknown ephys channels: {sorted(unknown)}")
    records = np.memmap(str(manifest["path"]), mode="r", dtype=EPHYS_DTYPE)
    segment = manifest["segments"][int(manifest["selected_segment_index"])]
    segment_records = records[int(segment["start_record"]) : int(segment["stop_record"])]
    tick_hz = float(manifest["tick_hz"])
    offset = float(manifest["time_offset_session_s"])
    first_tick = int(np.ceil((start_s - offset) * tick_hz))
    stop_tick = int(np.ceil((stop_s - offset) * tick_hz))
    ticks = segment_records["tick"]
    first = int(np.searchsorted(ticks, first_tick, side="left"))
    last = int(np.searchsorted(ticks, stop_tick, side="left"))
    window = segment_records[first:last]
    result = {"t_session_s": window["tick"].astype(np.float64) / tick_hz + offset}
    for channel in selected_channels:
        result[channel] = np.asarray(window[channel])
    return pd.DataFrame(result)
