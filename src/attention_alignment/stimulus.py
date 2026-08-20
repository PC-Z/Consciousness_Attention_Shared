from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ConditionConfig
from .errors import StimulusMatchError
from .models import StimulusBlock, TimeTransform, TriggerCluster


ORDER_PATTERN = re.compile(r"^(\d+):\s*([ABC]{5})\s*\(([ABC])\)\s*$")


def parse_order_file(path: str | Path) -> list[str]:
    """Parse `1: AAAAB (B)` order files with strict structural validation."""

    source = Path(path)
    sequences: list[str] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            match = ORDER_PATTERN.fullmatch(raw.strip())
            if match is None:
                raise StimulusMatchError(f"Malformed order line {line_number} in {source}: {raw!r}")
            index, sequence, label = match.groups()
            expected_index = len(sequences) + 1
            if int(index) != expected_index:
                raise StimulusMatchError(
                    f"Order index {index} is not consecutive; expected {expected_index} in {source}"
                )
            if sequence[-1] != label:
                raise StimulusMatchError(
                    f"Parenthesized label {label} does not match {sequence} in {source}"
                )
            sequences.append(sequence)
    if len(sequences) != 100:
        raise StimulusMatchError(f"Expected 100 sequences in {source}, found {len(sequences)}")
    return sequences


def experiment_sequences(test_sequences: Iterable[str]) -> dict[str, list[str]]:
    test = list(test_sequences)
    if len(test) != 100:
        raise StimulusMatchError(f"Expected 100 Test sequences, found {len(test)}")
    return {"train": ["AAAAB"] * 100, "test": test}


def sequence_table(test_sequences: Iterable[str], angles: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    global_sequence = 0
    global_item = 0
    for phase, sequences in experiment_sequences(test_sequences).items():
        for sequence_index, sequence in enumerate(sequences, start=1):
            global_sequence += 1
            for item_position, symbol in enumerate(sequence, start=1):
                global_item += 1
                rows.append(
                    {
                        "phase": phase,
                        "sequence_index": sequence_index,
                        "global_sequence_index": global_sequence,
                        "item_position": item_position,
                        "global_item_index": global_item,
                        "sequence_pattern": sequence,
                        "symbol": symbol,
                        "angle_deg": angles[symbol],
                    }
                )
    return pd.DataFrame(rows)


def _expected_interval_types(groups: int = 100, items: int = 5) -> list[str]:
    result: list[str] = []
    for group in range(groups):
        for item in range(items):
            result.append("stripe")
            if group == groups - 1 and item == items - 1:
                continue
            result.append("group_gap" if item == items - 1 else "gray")
    return result


def _score_window(
    starts: np.ndarray,
    start: int,
    condition: ConditionConfig,
    groups: int,
    items: int,
) -> tuple[int, float]:
    edge_count = groups * items * 2
    if start + edge_count > len(starts):
        return edge_count, float("inf")
    intervals = np.diff(starts[start : start + edge_count])
    interval_types = _expected_interval_types(groups, items)
    mismatch = 0
    score = 0.0
    for value, interval_type in zip(intervals, interval_types, strict=True):
        if interval_type == "stripe":
            expected = condition.expected_stripe_s
            tolerance = condition.onset_tolerance_s
            valid = abs(value - expected) <= tolerance
        elif interval_type == "gray":
            expected = condition.expected_gray_s
            tolerance = condition.gray_tolerance_s
            valid = abs(value - expected) <= tolerance
        else:
            expected = condition.expected_group_gap_s
            tolerance = 1.5
            valid = 4.0 <= value <= 12.0
        mismatch += int(not valid)
        score += min(abs(value - expected) / max(tolerance, 1e-6), 10.0)
    return mismatch, score / len(intervals)


def _expected_edge_times(
    condition: ConditionConfig,
    groups: int,
    items: int,
) -> np.ndarray:
    values = [0.0]
    time_s = 0.0
    for group in range(groups):
        for item in range(items):
            time_s += condition.expected_stripe_s
            values.append(time_s)
            if group == groups - 1 and item == items - 1:
                continue
            time_s += (
                condition.expected_group_gap_s
                if item == items - 1
                else condition.expected_gray_s
            )
            values.append(time_s)
    return np.asarray(values, dtype=float)


def _align_imperfect_block(
    starts: np.ndarray,
    candidate_start: int,
    condition: ConditionConfig,
    groups: int,
    items: int,
    max_missing: int = 2,
    max_extra: int = 2,
) -> tuple[tuple[int | None, ...], int, float] | None:
    """Align a nearly complete block using expected cumulative edge times.

    The exact matcher remains the normal path. This bounded fallback preserves
    one or two missing/extra boundaries instead of shifting every later event.
    """

    if candidate_start < 0 or candidate_start >= len(starts):
        return None
    expected = _expected_edge_times(condition, groups, items)
    block_stop_s = starts[candidate_start] + expected[-1] + 2.0
    stop = int(np.searchsorted(starts, block_stop_s, side="right"))
    observed_indices = np.arange(candidate_start, stop, dtype=np.int64)
    if len(observed_indices) < len(expected) - max_missing:
        return None
    observed = starts[observed_indices] - starts[candidate_start]

    # Force the candidate to be the first onset, then align the remaining edges.
    expected_tail = expected[1:]
    observed_tail = observed[1:]
    n, m = len(expected_tail), len(observed_tail)
    gap_penalty = 3.0
    tolerance = 0.15
    costs = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    actions = np.zeros((n + 1, m + 1), dtype=np.uint8)
    costs[:, 0] = np.arange(n + 1, dtype=np.float32) * gap_penalty
    costs[0, :] = np.arange(m + 1, dtype=np.float32) * gap_penalty
    actions[1:, 0] = 2  # missing expected boundary
    actions[0, 1:] = 3  # extra observed boundary
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_cost = costs[i - 1, j - 1] + min(
                abs(expected_tail[i - 1] - observed_tail[j - 1]) / tolerance,
                20.0,
            )
            missing_cost = costs[i - 1, j] + gap_penalty
            extra_cost = costs[i, j - 1] + gap_penalty
            choices = (match_cost, missing_cost, extra_cost)
            choice = int(np.argmin(choices))
            costs[i, j] = choices[choice]
            actions[i, j] = choice + 1

    mapping: list[int | None] = [candidate_start]
    reversed_mapping: list[int | None] = []
    extra_indices: list[int] = []
    i, j = n, m
    while i or j:
        action = int(actions[i, j])
        if action == 1:
            reversed_mapping.append(int(observed_indices[j]))
            i -= 1
            j -= 1
        elif action == 2:
            reversed_mapping.append(None)
            i -= 1
        elif action == 3:
            extra_indices.append(int(observed_indices[j]))
            j -= 1
        else:
            return None
    mapping.extend(reversed(reversed_mapping))
    missing_count = sum(value is None for value in mapping)
    extra_count = len(extra_indices)
    if len(mapping) != len(expected) or missing_count > max_missing or extra_count > max_extra:
        return None
    return tuple(mapping), missing_count + extra_count, float(costs[n, m] / len(expected))


def match_stimulus_blocks(
    clusters: list[TriggerCluster],
    condition: ConditionConfig,
    groups_per_phase: int = 100,
    items_per_group: int = 5,
    max_interval_mismatches: int = 0,
) -> tuple[list[StimulusBlock], dict[str, list[int] | int]]:
    """Find unique Train/Test blocks while leaving unrelated TTL clusters unused."""

    edge_count = groups_per_phase * items_per_group * 2
    starts = np.array([cluster.start_time_s for cluster in clusters], dtype=float)
    candidates: list[tuple[int, int, float]] = []
    for start in range(max(0, len(starts) - edge_count + 1)):
        mismatch, score = _score_window(
            starts, start, condition, groups_per_phase, items_per_group
        )
        if mismatch <= max_interval_mismatches:
            candidates.append((start, mismatch, score))

    pairs: list[tuple[tuple[int, int, float], tuple[int, int, float]]] = []
    for first in candidates:
        for second in candidates:
            if second[0] >= first[0] + edge_count:
                pairs.append((first, second))
    blocks: list[StimulusBlock] = []
    if pairs:
        pair = min(
            pairs,
            key=lambda item: (item[0][1] + item[1][1], item[0][2] + item[1][2]),
        )
        best_key = (pair[0][1] + pair[1][1], pair[0][2] + pair[1][2])
        if sum(
            1
            for item in pairs
            if (item[0][1] + item[1][1], item[0][2] + item[1][2]) == best_key
        ) > 1:
            raise StimulusMatchError("Train/Test trigger block match is not unique")
        for phase, candidate in zip(("train", "test"), pair, strict=True):
            start, mismatch, score = candidate
            blocks.append(
                StimulusBlock(
                    phase=phase,  # type: ignore[arg-type]
                    first_cluster_index=start,
                    cluster_indices=tuple(range(start, start + edge_count)),
                    mismatch_count=mismatch,
                    score=score,
                )
            )
    else:
        likely_starts = [0] + [
            index
            for index in range(1, len(starts))
            if starts[index] - starts[index - 1] >= 20.0
        ]
        imperfect = []
        for start in likely_starts:
            aligned = _align_imperfect_block(
                starts, start, condition, groups_per_phase, items_per_group
            )
            if aligned is not None:
                mapping, mismatch, score = aligned
                imperfect.append((start, mapping, mismatch, score))
        imperfect_pairs = [
            (first, second)
            for first in imperfect
            for second in imperfect
            if second[0] > max(index for index in first[1] if index is not None)
        ]
        if not imperfect_pairs:
            best = sorted(
                (
                    (
                        start,
                        *_score_window(
                            starts, start, condition, groups_per_phase, items_per_group
                        ),
                    )
                    for start in range(max(0, len(starts) - edge_count + 1))
                ),
                key=lambda item: (item[1], item[2]),
            )[:4]
            raise StimulusMatchError(
                f"Could not find two non-overlapping 100-group blocks; best candidates={best}"
            )
        first, second = min(
            imperfect_pairs,
            key=lambda item: (item[0][2] + item[1][2], item[0][3] + item[1][3]),
        )
        for phase, candidate in zip(("train", "test"), (first, second), strict=True):
            start, mapping, mismatch, score = candidate
            blocks.append(
                StimulusBlock(
                    phase=phase,  # type: ignore[arg-type]
                    first_cluster_index=start,
                    cluster_indices=mapping,
                    mismatch_count=mismatch,
                    score=score,
                )
            )

    used = {
        index
        for block in blocks
        for index in block.cluster_indices
        if index is not None
    }
    first_formal = blocks[0].first_cluster_index
    train_last = max(index for index in blocks[0].cluster_indices if index is not None)
    test_last = max(index for index in blocks[1].cluster_indices if index is not None)
    qc: dict[str, list[int] | int] = {
        "prelude_trigger_indices": [index for index in range(first_formal) if index not in used],
        "between_phase_extra_indices": [
            index
            for index in range(
                train_last + 1, blocks[1].first_cluster_index
            )
            if index not in used
        ],
        "postlude_trigger_indices": [
            index for index in range(test_last + 1, len(clusters))
        ],
        "extra_within_block_indices": [
            index
            for block in blocks
            for index in range(
                min(item for item in block.cluster_indices if item is not None),
                max(item for item in block.cluster_indices if item is not None) + 1,
            )
            if index not in used
        ],
        "missing_formal_boundary_count": sum(
            index is None for block in blocks for index in block.cluster_indices
        ),
        "formal_boundary_count": sum(len(block.cluster_indices) for block in blocks),
        "observed_formal_boundary_count": len(used),
        "all_cluster_count": len(clusters),
    }
    return blocks, qc


def build_stimulus_event_table(
    session_id: str,
    condition: ConditionConfig,
    test_sequences: Iterable[str],
    angles: dict[str, int],
    blocks: list[StimulusBlock],
    clusters: list[TriggerCluster],
    transform: TimeTransform,
) -> pd.DataFrame:
    """Pair formal boundaries and attach the intended A/B/C sequence identity."""

    design = sequence_table(test_sequences, angles)
    rows: list[dict[str, object]] = []
    for phase_index, phase in enumerate(("train", "test")):
        phase_design = design.loc[design["phase"] == phase].reset_index(drop=True)
        block = blocks[phase_index]
        for item_index, design_row in phase_design.iterrows():
            onset_index = block.cluster_indices[item_index * 2]
            offset_index = block.cluster_indices[item_index * 2 + 1]
            onset = clusters[onset_index] if onset_index is not None else None
            offset = clusters[offset_index] if offset_index is not None else None
            duration = (
                offset.start_time_s - onset.start_time_s
                if onset is not None and offset is not None
                else np.nan
            )
            next_onset_index = (
                block.cluster_indices[item_index * 2 + 2]
                if item_index < len(phase_design) - 1
                else None
            )
            post_gray = (
                clusters[next_onset_index].start_time_s - offset.start_time_s
                if next_onset_index is not None and offset is not None
                else np.nan
            )
            item_position = int(design_row["item_position"])
            expected_gray = (
                condition.expected_group_gap_s
                if item_position == 5
                else condition.expected_gray_s
            )
            duration_ok = (
                np.isfinite(duration)
                and abs(duration - condition.expected_stripe_s) <= condition.onset_tolerance_s
            )
            gray_ok = (
                True
                if np.isnan(post_gray)
                else (
                    4.0 <= post_gray <= 12.0
                    if item_position == 5
                    else abs(post_gray - expected_gray) <= condition.gray_tolerance_s
                )
            )
            rows.append(
                {
                    "session_id": session_id,
                    "condition_label_ms": condition.label_ms,
                    **design_row.to_dict(),
                    "expected_stripe_duration_s": condition.expected_stripe_s,
                    "expected_gray_duration_s": expected_gray,
                    "measured_onset_s": transform.device_to_session(onset.start_time_s)
                    if onset is not None
                    else np.nan,
                    "measured_offset_s": transform.device_to_session(offset.start_time_s)
                    if offset is not None
                    else np.nan,
                    "measured_duration_s": duration,
                    "measured_post_stimulus_gray_s": post_gray,
                    "trigger_index_on": onset_index,
                    "trigger_index_off": offset_index,
                    "timing_qc": (
                        "MISSING_TRIGGER"
                        if onset is None or offset is None
                        else "ok"
                        if duration_ok and gray_ok
                        else "STIMULUS_TIMING_OUTLIER"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    observed_onsets = result["measured_onset_s"].dropna()
    if len(result) != 1000 or not observed_onsets.is_monotonic_increasing:
        raise StimulusMatchError("Stimulus event table failed count or monotonicity validation")
    return result
