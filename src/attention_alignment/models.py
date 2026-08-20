from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


MarkerKind = Literal["R-Start", "R-End", "1", "2"]


@dataclass(frozen=True)
class MarkerRecord:
    row_index: int
    relative_time_s: float
    kind: MarkerKind
    abs_tick_100ns: int


@dataclass
class RecordingSegment:
    source: Path
    segment_index: int
    start: MarkerRecord
    end: MarkerRecord | None = None
    records: list[MarkerRecord] = field(default_factory=list)
    close_reason: str = "eof"

    def channel(self, kind: Literal["1", "2"]) -> list[MarkerRecord]:
        return [record for record in self.records if record.kind == kind]

    @property
    def duration_s(self) -> float:
        if self.end is not None:
            return self.end.relative_time_s - self.start.relative_time_s
        if self.records:
            return self.records[-1].relative_time_s - self.start.relative_time_s
        return 0.0


@dataclass(frozen=True)
class TriggerCluster:
    cluster_index: int
    start_time_s: float
    end_time_s: float
    start_abs_tick_100ns: int
    end_abs_tick_100ns: int
    first_row_index: int
    last_row_index: int
    record_count: int

    @property
    def width_s(self) -> float:
        return self.end_time_s - self.start_time_s


@dataclass(frozen=True)
class TimeTransform:
    stream: str
    anchor_device_s: float
    anchor_session_s: float
    segment_start_device_s: float
    segment_start_session_s: float

    def device_to_calcium_anchor(self, value: float) -> float:
        return value - self.anchor_device_s

    def device_to_session(self, value: float) -> float:
        return self.anchor_session_s + value - self.anchor_device_s

    def video_to_session(self, pts_s: float) -> float:
        return self.segment_start_session_s + pts_s


@dataclass(frozen=True)
class StimulusBlock:
    phase: Literal["train", "test"]
    first_cluster_index: int
    cluster_indices: tuple[int | None, ...]
    mismatch_count: int
    score: float
