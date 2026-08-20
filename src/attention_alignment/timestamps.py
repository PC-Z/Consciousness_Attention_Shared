from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .errors import MarkerParseError
from .models import MarkerRecord, RecordingSegment, TimeTransform, TriggerCluster


VALID_MARKERS = {"R-Start", "R-End", "1", "2"}


def parse_marker_file(path: str | Path) -> list[RecordingSegment]:
    """Parse an EEG marker text file into explicit recording segments."""

    source = Path(path).resolve()
    segments: list[RecordingSegment] = []
    current: RecordingSegment | None = None
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for row_index, raw in enumerate(handle):
            fields = raw.split()
            if not fields:
                continue
            if len(fields) != 3 or fields[1] not in VALID_MARKERS:
                raise MarkerParseError(f"Malformed marker row {row_index + 1} in {source}: {raw!r}")
            try:
                record = MarkerRecord(
                    row_index=row_index,
                    relative_time_s=float(fields[0]),
                    kind=fields[1],  # type: ignore[arg-type]
                    abs_tick_100ns=int(fields[2]),
                )
            except ValueError as exc:
                raise MarkerParseError(
                    f"Invalid numeric marker row {row_index + 1} in {source}: {raw!r}"
                ) from exc

            if record.kind == "R-Start":
                if current is not None:
                    current.close_reason = "next_start"
                current = RecordingSegment(
                    source=source,
                    segment_index=len(segments),
                    start=record,
                )
                segments.append(current)
            elif current is None:
                raise MarkerParseError(f"Marker before R-Start at row {row_index + 1} in {source}")
            elif record.kind == "R-End":
                current.end = record
                current.close_reason = "explicit_end"
                current = None
            else:
                current.records.append(record)
    if not segments:
        raise MarkerParseError(f"No R-Start marker found in {source}")
    return segments


def cluster_channel2(
    segment: RecordingSegment,
    max_within_cluster_gap_s: float = 0.010,
) -> list[TriggerCluster]:
    """Collapse duplicate channel-2 rows into physical boundary clusters."""

    records = segment.channel("2")
    grouped: list[list[MarkerRecord]] = []
    for record in records:
        if (
            not grouped
            or record.relative_time_s < grouped[-1][-1].relative_time_s
            or record.relative_time_s - grouped[-1][-1].relative_time_s > max_within_cluster_gap_s
        ):
            grouped.append([record])
        else:
            grouped[-1].append(record)
    return [
        TriggerCluster(
            cluster_index=index,
            start_time_s=group[0].relative_time_s,
            end_time_s=group[-1].relative_time_s,
            start_abs_tick_100ns=group[0].abs_tick_100ns,
            end_abs_tick_100ns=group[-1].abs_tick_100ns,
            first_row_index=group[0].row_index,
            last_row_index=group[-1].row_index,
            record_count=len(group),
        )
        for index, group in enumerate(grouped)
    ]


def choose_main_segment(
    segments: Iterable[RecordingSegment],
    block_counter: Callable[[RecordingSegment], int] | None = None,
) -> RecordingSegment:
    """Choose the segment with formal blocks, then calcium count and duration."""

    candidates = list(segments)
    if not candidates:
        raise MarkerParseError("No recording segments are available")

    def rank(segment: RecordingSegment) -> tuple[int, int, float]:
        blocks = int(block_counter(segment)) if block_counter is not None else 0
        return blocks, len(segment.channel("1")), segment.duration_s

    return max(candidates, key=rank)


def make_time_transforms(
    segment_01: RecordingSegment,
    segment_02: RecordingSegment,
    tick_hz: int = 10_000_000,
) -> dict[str, TimeTransform]:
    """Create per-stream transforms anchored at each first calcium pulse."""

    calcium_01 = segment_01.channel("1")
    calcium_02 = segment_02.channel("1")
    if not calcium_01 or not calcium_02:
        raise MarkerParseError("Both 01 and 02 main segments need channel-1 markers")
    session_start_tick = min(segment_01.start.abs_tick_100ns, segment_02.start.abs_tick_100ns)

    def transform(stream: str, segment: RecordingSegment, anchor: MarkerRecord) -> TimeTransform:
        return TimeTransform(
            stream=stream,
            anchor_device_s=anchor.relative_time_s,
            anchor_session_s=(anchor.abs_tick_100ns - session_start_tick) / tick_hz,
            segment_start_device_s=segment.start.relative_time_s,
            segment_start_session_s=(segment.start.abs_tick_100ns - session_start_tick) / tick_hz,
        )

    return {
        "01": transform("01", segment_01, calcium_01[0]),
        "02": transform("02", segment_02, calcium_02[0]),
    }


def _nearest_residual(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    if not len(reference) or not len(query):
        return np.array([], dtype=float)
    positions = np.searchsorted(reference, query)
    left = np.clip(positions - 1, 0, len(reference) - 1)
    right = np.clip(positions, 0, len(reference) - 1)
    left_delta = query - reference[left]
    right_delta = query - reference[right]
    choose_right = np.abs(right_delta) < np.abs(left_delta)
    return np.where(choose_right, right_delta, left_delta)


def alignment_qc(
    segment_01: RecordingSegment,
    segment_02: RecordingSegment,
    cluster_gap_s: float = 0.010,
) -> dict[str, float | int | bool]:
    """Quantify 01/02 agreement after independent first-calcium anchoring."""

    channel_01 = np.array([r.relative_time_s for r in segment_01.channel("1")], dtype=float)
    channel_02 = np.array([r.relative_time_s for r in segment_02.channel("1")], dtype=float)
    channel_01 -= channel_01[0]
    channel_02 -= channel_02[0]
    calcium_n = min(len(channel_01), len(channel_02))
    calcium_residual = channel_01[:calcium_n] - channel_02[:calcium_n]

    clusters_01 = np.array(
        [item.start_time_s for item in cluster_channel2(segment_01, cluster_gap_s)], dtype=float
    )
    clusters_02 = np.array(
        [item.start_time_s for item in cluster_channel2(segment_02, cluster_gap_s)], dtype=float
    )
    clusters_01 -= channel_01[0] + segment_01.channel("1")[0].relative_time_s
    clusters_02 -= channel_02[0] + segment_02.channel("1")[0].relative_time_s
    trigger_residual = _nearest_residual(clusters_01, clusters_02)
    return {
        "calcium_count_01": len(channel_01),
        "calcium_count_02": len(channel_02),
        "calcium_compared": calcium_n,
        "calcium_max_abs_residual_s": float(np.max(np.abs(calcium_residual)))
        if calcium_n
        else float("nan"),
        "trigger_clusters_01": len(clusters_01),
        "trigger_clusters_02": len(clusters_02),
        "trigger_max_abs_nearest_residual_s": float(np.max(np.abs(trigger_residual)))
        if len(trigger_residual)
        else float("nan"),
        "passes_1ms": bool(
            calcium_n
            and np.max(np.abs(calcium_residual)) <= 0.001
            and len(trigger_residual)
            and np.max(np.abs(trigger_residual)) <= 0.001
        ),
    }
