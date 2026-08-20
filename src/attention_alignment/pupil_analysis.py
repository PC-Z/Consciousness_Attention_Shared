from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StimulusWindowSpec:
    """Condition-specific windows inferred from the aligned event table."""

    condition_label_ms: int
    pre_stimulus_s: float
    stimulus_s: float
    sequence_s: float


def infer_stimulus_window(events: pd.DataFrame) -> StimulusWindowSpec:
    """Infer the single-item and five-item windows from exported event metadata."""

    required = {
        "condition_label_ms",
        "item_position",
        "expected_stripe_duration_s",
        "expected_gray_duration_s",
    }
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"Stimulus event table is missing columns: {sorted(missing)}")
    labels = events["condition_label_ms"].dropna().unique()
    stripes = events["expected_stripe_duration_s"].dropna().unique()
    within_gray = events.loc[
        events["item_position"].astype(int) < 5, "expected_gray_duration_s"
    ].dropna().unique()
    if len(labels) != 1 or len(stripes) != 1 or len(within_gray) != 1:
        raise ValueError("One event table must contain exactly one stimulus condition")
    stimulus_s = float(stripes[0])
    gray_s = float(within_gray[0])
    return StimulusWindowSpec(
        condition_label_ms=int(labels[0]),
        pre_stimulus_s=gray_s,
        stimulus_s=stimulus_s,
        sequence_s=5.0 * stimulus_s + 4.0 * gray_s,
    )


def _analysis_valid_column(table: pd.DataFrame) -> str:
    return "pupil_analysis_valid" if "pupil_analysis_valid" in table else "pupil_valid"


def _longest_invalid_duration(table: pd.DataFrame, valid_column: str) -> float:
    if table.empty or valid_column not in table:
        return float("nan")
    invalid = ~table[valid_column].fillna(False).to_numpy(dtype=bool)
    times = table["t_session_s"].to_numpy(dtype=float)
    longest = 0.0
    start: int | None = None
    for index, is_invalid in enumerate(invalid):
        if is_invalid and start is None:
            start = index
        if start is not None and (not is_invalid or index == len(invalid) - 1):
            stop = index if is_invalid else index - 1
            longest = max(longest, float(times[stop] - times[start]))
            start = None
    return longest


def pupil_qc_table(behavior: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Summarize pupil coverage separately for the full video, train, and test."""

    required = {"t_session_s", "pupil_valid"}
    missing = required.difference(behavior.columns)
    if missing:
        raise KeyError(f"Behavior table is missing columns: {sorted(missing)}")
    spec = infer_stimulus_window(events)
    valid_column = _analysis_valid_column(behavior)
    rows: list[dict[str, object]] = []
    for phase in ("all", "train", "test"):
        if phase == "all":
            selected = behavior
        else:
            phase_events = events.loc[events["phase"].eq(phase)]
            start = float(phase_events["measured_onset_s"].min()) - spec.pre_stimulus_s
            stop = float(phase_events["measured_offset_s"].max())
            selected = behavior.loc[behavior["t_session_s"].between(start, stop)]
        observed_valid = selected["pupil_valid"].fillna(False).astype(bool)
        detector_valid = selected.get(
            "pupil_detector_valid", selected["pupil_valid"]
        ).fillna(False).astype(bool)
        strict_observed_valid = selected.get(
            "pupil_analysis_observed_valid", selected["pupil_valid"]
        ).fillna(False).astype(bool)
        valid = selected[valid_column].fillna(False).astype(bool)
        correction = (
            selected.loc[valid, "pupil_hull_correction_fraction"].dropna()
            if "pupil_hull_correction_fraction" in selected
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "phase": phase,
                "frames": int(len(selected)),
                "pupil_valid_fraction": float(valid.mean()) if len(valid) else np.nan,
                "pupil_observed_valid_fraction": float(observed_valid.mean())
                if len(observed_valid)
                else np.nan,
                "pupil_detector_valid_fraction": float(detector_valid.mean())
                if len(detector_valid)
                else np.nan,
                "pupil_analysis_observed_valid_fraction": float(
                    strict_observed_valid.mean()
                )
                if len(strict_observed_valid)
                else np.nan,
                "pupil_review_fraction": float(
                    selected.get(
                        "pupil_review_required",
                        pd.Series(False, index=selected.index),
                    )
                    .fillna(False)
                    .mean()
                )
                if len(selected)
                else np.nan,
                "pupil_qc_rejected_fraction": float(
                    selected.get(
                        "pupil_qc_rejected",
                        pd.Series(False, index=selected.index),
                    )
                    .fillna(False)
                    .mean()
                )
                if len(selected)
                else np.nan,
                "longest_invalid_gap_s": _longest_invalid_duration(
                    selected, valid_column
                ),
                "hull_correction_median": float(correction.median())
                if len(correction)
                else np.nan,
                "hull_correction_p95": float(correction.quantile(0.95))
                if len(correction)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _time_grid(start_s: float, stop_s: float, sample_rate_hz: float) -> np.ndarray:
    if sample_rate_hz <= 0 or stop_s <= start_s:
        raise ValueError("Window and sample rate must be positive")
    count = int(round((stop_s - start_s) * sample_rate_hz))
    return start_s + np.arange(count + 1, dtype=float) / sample_rate_hz


def _interpolate_preserving_gaps(
    times: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
    max_gap_s: float,
) -> np.ndarray:
    finite = np.isfinite(times) & np.isfinite(values)
    source_times = times[finite]
    source_values = values[finite]
    result = np.full(len(target_times), np.nan, dtype=float)
    if len(source_times) < 2:
        return result
    order = np.argsort(source_times)
    source_times = source_times[order]
    source_values = source_values[order]
    unique = np.concatenate([[True], np.diff(source_times) > 0])
    source_times = source_times[unique]
    source_values = source_values[unique]
    if len(source_times) < 2:
        return result
    right = np.searchsorted(source_times, target_times, side="left")
    left = np.searchsorted(source_times, target_times, side="right") - 1
    inside = (left >= 0) & (right < len(source_times))
    safe_left = np.clip(left, 0, len(source_times) - 1)
    safe_right = np.clip(right, 0, len(source_times) - 1)
    supported = inside & (
        source_times[safe_right] - source_times[safe_left] <= max_gap_s + 1e-9
    )
    interpolated = np.interp(target_times, source_times, source_values)
    result[supported] = interpolated[supported]
    return result


def _extract_anchors(
    anchors: pd.DataFrame,
    behavior: pd.DataFrame,
    *,
    start_s: float,
    stop_s: float,
    response_stop_s: float,
    sample_rate_hz: float,
    max_interpolation_gap_s: float,
    min_valid_fraction: float,
) -> pd.DataFrame:
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    radius_column = "pupil_equivalent_radius_interpolated"
    if radius_column not in behavior:
        raise KeyError(f"Behavior table is missing {radius_column!r}")
    session_valid = behavior[_analysis_valid_column(behavior)].fillna(False).astype(bool)
    session_diameter = 2.0 * behavior.loc[session_valid, radius_column].dropna()
    session_mean = float(session_diameter.mean())
    session_std = float(session_diameter.std(ddof=0))
    movement = behavior.get("movement_abs_difference", pd.Series(np.nan, index=behavior.index))
    movement_mean = float(movement.mean())
    movement_std = float(movement.std(ddof=0))
    relative_grid = _time_grid(start_s, stop_s, sample_rate_hz)
    rows: list[pd.DataFrame] = []
    for anchor in anchors.itertuples(index=False):
        onset = float(anchor.anchor_onset_s)
        absolute_grid = onset + relative_grid
        source = behavior.loc[
            behavior["t_session_s"].between(
                absolute_grid[0] - max_interpolation_gap_s,
                absolute_grid[-1] + max_interpolation_gap_s,
            )
        ]
        source_time = source["t_session_s"].to_numpy(dtype=float)
        diameter = 2.0 * _interpolate_preserving_gaps(
            source_time,
            source[radius_column].to_numpy(dtype=float),
            absolute_grid,
            max_interpolation_gap_s,
        )
        area_column = "pupil_area_interpolated"
        area = (
            _interpolate_preserving_gaps(
                source_time,
                source[area_column].to_numpy(dtype=float),
                absolute_grid,
                max_interpolation_gap_s,
            )
            if area_column in source
            else np.full(len(relative_grid), np.nan)
        )
        movement_trace = (
            _interpolate_preserving_gaps(
                source_time,
                source["movement_abs_difference"].to_numpy(dtype=float),
                absolute_grid,
                max_interpolation_gap_s,
            )
            if "movement_abs_difference" in source
            else np.full(len(relative_grid), np.nan)
        )
        correction_trace = (
            _interpolate_preserving_gaps(
                source_time,
                source["pupil_hull_correction_fraction"].to_numpy(dtype=float),
                absolute_grid,
                max_interpolation_gap_s,
            )
            if "pupil_hull_correction_fraction" in source
            else np.full(len(relative_grid), np.nan)
        )
        baseline_mask = relative_grid < 0
        response_mask = (relative_grid >= 0) & (relative_grid <= response_stop_s)
        baseline_valid_fraction = float(np.isfinite(diameter[baseline_mask]).mean())
        response_valid_fraction = float(np.isfinite(diameter[response_mask]).mean())
        baseline = float(np.nanmedian(diameter[baseline_mask]))
        trial_valid = bool(
            np.isfinite(baseline)
            and baseline > 0
            and baseline_valid_fraction >= min_valid_fraction
            and response_valid_fraction >= min_valid_fraction
        )
        pupil_delta = diameter / baseline - 1.0 if trial_valid else np.full_like(diameter, np.nan)
        pupil_z = (
            (diameter - session_mean) / session_std
            if np.isfinite(session_std) and session_std > 0
            else np.full_like(diameter, np.nan)
        )
        movement_z = (
            (movement_trace - movement_mean) / movement_std
            if np.isfinite(movement_std) and movement_std > 0
            else np.full_like(movement_trace, np.nan)
        )
        metadata = anchor._asdict()
        metadata.pop("anchor_onset_s")
        frame = pd.DataFrame(
            {
                **metadata,
                "trial_time_s": relative_grid,
                "t_session_s": absolute_grid,
                "pupil_diameter_px": diameter,
                "pupil_area_px2": area,
                "pupil_delta_fraction": pupil_delta,
                "pupil_session_z": pupil_z,
                "movement_abs_difference": movement_trace,
                "movement_z": movement_z,
                "pupil_hull_correction_fraction": correction_trace,
                "pupil_baseline_diameter_px": baseline,
                "baseline_valid_fraction": baseline_valid_fraction,
                "response_valid_fraction": response_valid_fraction,
                "trial_valid": trial_valid,
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def extract_item_trials(
    events: pd.DataFrame,
    behavior: pd.DataFrame,
    *,
    phase: str,
    item_positions: int | Iterable[int],
    sequence_patterns: Iterable[str] | None = None,
    symbols: Iterable[str] | None = None,
    sample_rate_hz: float = 30.0,
    max_interpolation_gap_s: float = 0.5,
    min_valid_fraction: float = 0.8,
) -> pd.DataFrame:
    """Extract baseline-plus-stimulus windows aligned to selected item onsets."""

    spec = infer_stimulus_window(events)
    positions = [item_positions] if isinstance(item_positions, int) else list(item_positions)
    selected = events.loc[
        events["phase"].eq(phase) & events["item_position"].isin(positions)
    ].copy()
    if sequence_patterns is not None:
        selected = selected.loc[selected["sequence_pattern"].isin(sequence_patterns)]
    if symbols is not None:
        selected = selected.loc[selected["symbol"].isin(symbols)]
    selected = selected.sort_values(["global_sequence_index", "item_position"])
    selected["anchor_onset_s"] = selected["measured_onset_s"].astype(float)
    selected["alignment"] = selected["item_position"].map(lambda value: f"P{int(value)}")
    selected["trial_id"] = selected.apply(
        lambda row: f"{row.phase}:{int(row.sequence_index)}:P{int(row.item_position)}",
        axis=1,
    )
    metadata = [
        "session_id",
        "condition_label_ms",
        "phase",
        "sequence_index",
        "global_sequence_index",
        "sequence_pattern",
        "item_position",
        "symbol",
        "alignment",
        "trial_id",
        "anchor_onset_s",
    ]
    return _extract_anchors(
        selected[metadata],
        behavior,
        start_s=-spec.pre_stimulus_s,
        stop_s=spec.stimulus_s,
        response_stop_s=spec.stimulus_s,
        sample_rate_hz=sample_rate_hz,
        max_interpolation_gap_s=max_interpolation_gap_s,
        min_valid_fraction=min_valid_fraction,
    )


def extract_sequence_trials(
    events: pd.DataFrame,
    behavior: pd.DataFrame,
    *,
    phase: str,
    sequence_patterns: Iterable[str] | None = None,
    sample_rate_hz: float = 30.0,
    max_interpolation_gap_s: float = 0.5,
    min_valid_fraction: float = 0.8,
) -> pd.DataFrame:
    """Extract full AAAAX windows aligned to the onset of P1."""

    spec = infer_stimulus_window(events)
    selected = events.loc[
        events["phase"].eq(phase) & events["item_position"].eq(1)
    ].copy()
    if sequence_patterns is not None:
        selected = selected.loc[selected["sequence_pattern"].isin(sequence_patterns)]
    selected = selected.sort_values("global_sequence_index")
    selected["anchor_onset_s"] = selected["measured_onset_s"].astype(float)
    selected["alignment"] = "sequence_P1"
    selected["symbol"] = selected["sequence_pattern"]
    selected["item_position"] = 0
    selected["trial_id"] = selected.apply(
        lambda row: f"{row.phase}:{int(row.sequence_index)}:sequence", axis=1
    )
    metadata = [
        "session_id",
        "condition_label_ms",
        "phase",
        "sequence_index",
        "global_sequence_index",
        "sequence_pattern",
        "item_position",
        "symbol",
        "alignment",
        "trial_id",
        "anchor_onset_s",
    ]
    return _extract_anchors(
        selected[metadata],
        behavior,
        start_s=-spec.pre_stimulus_s,
        stop_s=spec.sequence_s,
        response_stop_s=spec.sequence_s,
        sample_rate_hz=sample_rate_hz,
        max_interpolation_gap_s=max_interpolation_gap_s,
        min_valid_fraction=min_valid_fraction,
    )


def p5_minus_p4(item_trials: pd.DataFrame) -> pd.DataFrame:
    """Subtract P4-aligned from P5-aligned pupil traces within each sequence."""

    selected = item_trials.loc[item_trials["item_position"].isin([4, 5])].copy()
    keys = [
        "session_id",
        "condition_label_ms",
        "phase",
        "sequence_index",
        "global_sequence_index",
        "sequence_pattern",
        "trial_time_s",
    ]
    required_values = [
        "symbol",
        "pupil_delta_fraction",
        "pupil_session_z",
        "movement_z",
        "trial_valid",
    ]
    optional_values = [
        "pupil_baseline_diameter_px",
        "pupil_hull_correction_fraction",
        "baseline_valid_fraction",
        "response_valid_fraction",
    ]
    values = required_values + [
        column for column in optional_values if column in selected.columns
    ]
    p4 = selected.loc[selected["item_position"].eq(4), keys + values].rename(
        columns={column: f"{column}_p4" for column in values}
    )
    p5 = selected.loc[selected["item_position"].eq(5), keys + values].rename(
        columns={column: f"{column}_p5" for column in values}
    )
    result = p5.merge(p4, on=keys, how="inner", validate="one_to_one")
    result["trial_id"] = result.apply(
        lambda row: f"{row.phase}:{int(row.sequence_index)}:P5-P4", axis=1
    )
    result["p5_minus_p4_pupil_z"] = (
        result["pupil_session_z_p5"] - result["pupil_session_z_p4"]
    )
    result["p5_minus_p4_delta_fraction"] = (
        result["pupil_delta_fraction_p5"] - result["pupil_delta_fraction_p4"]
    )
    result["p5_minus_p4_movement_z"] = result["movement_z_p5"] - result["movement_z_p4"]
    result["trial_valid"] = result["trial_valid_p5"] & result["trial_valid_p4"]
    return result


def trial_metrics(
    traces: pd.DataFrame,
    *,
    value_column: str = "pupil_delta_fraction",
    response_window_s: tuple[float, float] | None = None,
    movement_column: str | None = "movement_z",
) -> pd.DataFrame:
    """Reduce aligned traces to one response row per trial."""

    if traces.empty:
        return pd.DataFrame()
    if value_column not in traces:
        raise KeyError(value_column)
    if response_window_s is None:
        response_window_s = (0.0, float(traces["trial_time_s"].max()))
    metadata_columns = [
        column
        for column in (
            "session_id",
            "condition_label_ms",
            "phase",
            "sequence_index",
            "global_sequence_index",
            "sequence_pattern",
            "item_position",
            "symbol",
            "alignment",
            "trial_id",
        )
        if column in traces
    ]
    rows: list[dict[str, object]] = []
    for _, trial in traces.groupby("trial_id", sort=False):
        first = trial.iloc[0]
        response_columns = ["trial_time_s", value_column]
        if movement_column is not None and movement_column in trial:
            response_columns.append(movement_column)
        response = trial.loc[
            trial["trial_time_s"].between(*response_window_s),
            response_columns,
        ].dropna(subset=[value_column])
        valid = bool(first["trial_valid"]) and not response.empty
        x = response["trial_time_s"].to_numpy(dtype=float)
        y = response[value_column].to_numpy(dtype=float)
        row = {
            **{column: first[column] for column in metadata_columns},
            "trial_valid": valid,
            "baseline_valid_fraction": float(
                first.get("baseline_valid_fraction", np.nan)
            ),
            "response_valid_fraction": float(
                first.get("response_valid_fraction", np.nan)
            ),
            "response_mean": float(np.mean(y)) if valid else np.nan,
            "response_auc": float(np.trapezoid(y, x))
            if valid and len(y) > 1
            else np.nan,
            "peak_dilation": float(np.max(y)) if valid else np.nan,
            "peak_constriction": float(np.min(y)) if valid else np.nan,
            "movement_mean_z": float(response[movement_column].mean())
            if valid and movement_column is not None and movement_column in response
            else np.nan,
        }
        for column in (
            "pupil_baseline_diameter_px_p4",
            "pupil_baseline_diameter_px_p5",
            "pupil_hull_correction_fraction_p4",
            "pupil_hull_correction_fraction_p5",
        ):
            if column in trial:
                row[column] = float(trial.loc[trial["trial_time_s"].between(*response_window_s), column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def add_trial_groups(
    metrics: pd.DataFrame,
    group_size: int,
    *,
    group_columns: Sequence[str] = ("phase", "sequence_pattern", "alignment"),
) -> pd.DataFrame:
    """Assign chronological, one-based groups within each requested condition."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    result = metrics.copy()
    columns = [column for column in group_columns if column in result]
    result = result.sort_values(columns + ["global_sequence_index", "sequence_index"])
    occurrence = result.groupby(columns, dropna=False).cumcount() if columns else np.arange(len(result))
    result["condition_trial_number"] = occurrence + 1
    result["trial_group"] = occurrence // group_size + 1
    return result


def add_original_trial_groups(
    metrics: pd.DataFrame,
    group_size: int,
    *,
    trial_number_column: str = "sequence_index",
) -> pd.DataFrame:
    """Assign fixed bins from original one-based trial numbers without regrouping."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if trial_number_column not in metrics:
        raise KeyError(trial_number_column)
    result = metrics.copy()
    numbers = pd.to_numeric(result[trial_number_column], errors="coerce")
    if numbers.isna().any() or (numbers < 1).any() or not np.allclose(numbers, np.round(numbers)):
        raise ValueError(f"{trial_number_column} must contain positive integers")
    result["condition_trial_number"] = numbers.astype(int)
    result["trial_group"] = ((numbers.astype(int) - 1) // group_size) + 1
    return result.sort_values("condition_trial_number").reset_index(drop=True)


def pair_immediate_preceding_b(
    metrics: pd.DataFrame,
    *,
    catch_pattern: str,
    value_column: str = "response_mean",
    reference_pattern: str = "AAAAB",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair each valid catch only with the valid B at original trial index minus one."""

    required = {
        "trial_id",
        "sequence_index",
        "sequence_pattern",
        "trial_valid",
        value_column,
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise KeyError(f"Trial metrics are missing columns: {sorted(missing)}")
    if catch_pattern == reference_pattern:
        raise ValueError("catch_pattern and reference_pattern must differ")
    if metrics["sequence_index"].duplicated().any():
        raise ValueError("Trial metrics must contain one row per sequence_index")

    indexed = metrics.set_index("sequence_index", drop=False)
    catch_rows = metrics.loc[metrics["sequence_pattern"].eq(catch_pattern)].sort_values(
        "sequence_index"
    )
    pair_rows: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []
    reference_label = "B_A" if catch_pattern == "AAAAA" else "B_C" if catch_pattern == "AAAAC" else "B"
    comparison = f"{catch_pattern}_vs_{reference_label}"

    for catch in catch_rows.itertuples(index=False):
        catch_index = int(catch.sequence_index)
        reason: str | None = None
        reference = None
        if not bool(catch.trial_valid):
            reason = "catch_invalid"
        elif not np.isfinite(float(getattr(catch, value_column))):
            reason = "catch_response_missing"
        elif catch_index - 1 not in indexed.index:
            reason = "immediate_predecessor_missing"
        else:
            reference = indexed.loc[catch_index - 1]
            if str(reference["sequence_pattern"]) != reference_pattern:
                reason = "immediate_predecessor_not_B"
            elif not bool(reference["trial_valid"]):
                reason = "immediate_predecessor_invalid"
            elif not np.isfinite(float(reference[value_column])):
                reason = "immediate_predecessor_response_missing"

        if reason is not None:
            unmatched_rows.append(
                {
                    "comparison": comparison,
                    "catch_pattern": catch_pattern,
                    "catch_trial_id": catch.trial_id,
                    "catch_sequence_index": catch_index,
                    "expected_reference_sequence_index": catch_index - 1,
                    "reason": reason,
                }
            )
            continue

        assert reference is not None
        row: dict[str, object] = {
            "comparison": comparison,
            "catch_pattern": catch_pattern,
            "reference_label": reference_label,
            "pair_id": f"{catch.trial_id}|{reference['trial_id']}",
            "catch_trial_id": catch.trial_id,
            "reference_trial_id": reference["trial_id"],
            "catch_sequence_index": catch_index,
            "reference_sequence_index": int(reference["sequence_index"]),
            "catch_response": float(getattr(catch, value_column)),
            "reference_response": float(reference[value_column]),
        }
        row["response_difference"] = row["catch_response"] - row["reference_response"]
        for column in (
            "movement_mean_z",
            "pupil_baseline_diameter_px_p4",
            "pupil_baseline_diameter_px_p5",
            "pupil_hull_correction_fraction_p4",
            "pupil_hull_correction_fraction_p5",
        ):
            if column in metrics:
                row[f"catch_{column}"] = float(getattr(catch, column))
                row[f"reference_{column}"] = float(reference[column])
        pair_rows.append(row)

    return pd.DataFrame(pair_rows), pd.DataFrame(unmatched_rows)


def stratified_b_resampling(
    metrics: pd.DataFrame,
    *,
    catch_pattern: str,
    value_column: str = "response_mean",
    reference_pattern: str = "AAAAB",
    block_size: int = 10,
    repetitions: int = 500,
    seed: int = 20260817,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Equal-sample B within fixed chronology blocks and summarize selection sensitivity."""

    if block_size <= 0 or repetitions <= 0:
        raise ValueError("block_size and repetitions must be positive")
    required = {
        "trial_id",
        "sequence_index",
        "sequence_pattern",
        "trial_valid",
        value_column,
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise KeyError(f"Trial metrics are missing columns: {sorted(missing)}")

    valid = metrics.loc[
        metrics["trial_valid"].fillna(False) & metrics[value_column].notna()
    ].copy()
    valid["chronology_block"] = (
        (valid["sequence_index"].astype(int) - 1) // block_size + 1
    )
    catches = valid.loc[valid["sequence_pattern"].eq(catch_pattern)]
    references = valid.loc[valid["sequence_pattern"].eq(reference_pattern)]
    if catches.empty:
        raise ValueError(f"No valid {catch_pattern} catch trials are available")

    block_counts = catches.groupby("chronology_block").size()
    for block, catch_count in block_counts.items():
        reference_count = int(references["chronology_block"].eq(block).sum())
        if reference_count < int(catch_count):
            raise ValueError(
                f"Chronology block {int(block)} has {reference_count} valid B trials "
                f"for {int(catch_count)} valid {catch_pattern} catches"
            )

    rng = np.random.default_rng(seed)
    selected_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    catch_mean = float(catches[value_column].mean())
    comparison = f"{catch_pattern}_vs_resampled_B"
    for repeat in range(1, repetitions + 1):
        selected_indices: list[int] = []
        for block, catch_count in block_counts.items():
            candidates = references.index[
                references["chronology_block"].eq(block)
            ].to_numpy()
            chosen = rng.choice(candidates, size=int(catch_count), replace=False)
            selected_indices.extend(int(index) for index in chosen)
        selected = references.loc[selected_indices]
        reference_mean = float(selected[value_column].mean())
        effect_rows.append(
            {
                "comparison": comparison,
                "repeat": repeat,
                "catch_count": int(len(catches)),
                "reference_count": int(len(selected)),
                "catch_mean": catch_mean,
                "reference_mean": reference_mean,
                "response_difference": catch_mean - reference_mean,
            }
        )
        for reference in selected.itertuples(index=False):
            selected_rows.append(
                {
                    "comparison": comparison,
                    "repeat": repeat,
                    "catch_pattern": catch_pattern,
                    "chronology_block": int(reference.chronology_block),
                    "reference_trial_id": reference.trial_id,
                    "reference_sequence_index": int(reference.sequence_index),
                }
            )

    selections = pd.DataFrame(selected_rows)
    effects = pd.DataFrame(effect_rows)
    mean_effect = float(effects["response_difference"].mean())
    if np.isclose(mean_effect, 0.0):
        sign_consistency = np.nan
    else:
        sign_consistency = float(
            (np.sign(effects["response_difference"]) == np.sign(mean_effect)).mean()
        )
    summary = pd.DataFrame(
        [
            {
                "comparison": comparison,
                "repetitions": repetitions,
                "block_size": block_size,
                "seed": seed,
                "catch_count": int(len(catches)),
                "mean_effect": mean_effect,
                "b_selection_interval_low": float(
                    effects["response_difference"].quantile(0.025)
                ),
                "b_selection_interval_high": float(
                    effects["response_difference"].quantile(0.975)
                ),
                "sign_consistency": sign_consistency,
            }
        ]
    )
    return selections, effects, summary


def resampled_trace_effects(
    traces: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    catch_pattern: str,
    value_column: str = "p5_minus_p4_pupil_z",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate catch-minus-selected-B traces for every resampling repeat."""

    required_trace = {"trial_id", "trial_time_s", "sequence_pattern", "trial_valid", value_column}
    missing_trace = required_trace.difference(traces.columns)
    if missing_trace:
        raise KeyError(f"Trace table is missing columns: {sorted(missing_trace)}")
    required_selection = {"repeat", "reference_trial_id"}
    missing_selection = required_selection.difference(selections.columns)
    if missing_selection:
        raise KeyError(f"Selection table is missing columns: {sorted(missing_selection)}")

    valid = traces.loc[traces["trial_valid"] & traces[value_column].notna()].copy()
    catch_trace = (
        valid.loc[valid["sequence_pattern"].eq(catch_pattern)]
        .groupby("trial_time_s")[value_column]
        .mean()
        .rename("catch_mean")
    )
    reference_trace = selections.merge(
        valid[["trial_id", "trial_time_s", value_column]],
        left_on="reference_trial_id",
        right_on="trial_id",
        how="left",
        validate="many_to_many",
    )
    if reference_trace[value_column].isna().all():
        raise ValueError("Selected reference trial IDs were not found in traces")
    reference_trace = (
        reference_trace.groupby(["repeat", "trial_time_s"])[value_column]
        .mean()
        .rename("reference_mean")
        .reset_index()
    )
    repeat_traces = reference_trace.merge(
        catch_trace.reset_index(), on="trial_time_s", how="left", validate="many_to_one"
    )
    repeat_traces["response_difference"] = (
        repeat_traces["catch_mean"] - repeat_traces["reference_mean"]
    )
    summary = (
        repeat_traces.groupby("trial_time_s")["response_difference"]
        .agg(
            mean="mean",
            b_selection_interval_low=lambda values: values.quantile(0.025),
            b_selection_interval_high=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    return repeat_traces, summary


def _mean_and_error(
    traces: pd.DataFrame,
    value_column: str,
    error_band: str,
) -> pd.DataFrame:
    summary = traces.groupby("trial_time_s")[value_column].agg(["mean", "std", "count"])
    sem = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    if error_band == "sem":
        width = sem
    elif error_band == "ci95":
        width = 1.96 * sem
    else:
        raise ValueError("error_band must be 'sem' or 'ci95'")
    summary["lower"] = summary["mean"] - width.fillna(0.0)
    summary["upper"] = summary["mean"] + width.fillna(0.0)
    return summary.reset_index()


def sequence_stimulus_intervals(
    pattern: str, spec: StimulusWindowSpec
) -> list[tuple[float, float, str]]:
    """Return relative stimulus intervals for shading a five-item sequence plot."""

    if len(pattern) != 5:
        raise ValueError("Sequence pattern must contain five symbols")
    step = spec.stimulus_s + spec.pre_stimulus_s
    return [
        (position * step, position * step + spec.stimulus_s, symbol)
        for position, symbol in enumerate(pattern)
    ]


def _shade_intervals(axis, intervals: Sequence[tuple[float, float, str]]) -> None:
    colors = {"A": "#D9E8F5", "B": "#F4C7C3", "C": "#CDE9D7", "stimulus": "#E5E5E5"}
    for start, stop, label in intervals:
        axis.axvspan(start, stop, color=colors.get(label, "#E5E5E5"), alpha=0.7, zorder=0)
        axis.text((start + stop) / 2, 0.98, label, transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=8)


def plot_grouped_trial_traces(
    traces: pd.DataFrame,
    *,
    group_size: int = 5,
    value_column: str = "pupil_delta_fraction",
    stimulus_intervals: Sequence[tuple[float, float, str]] = (),
    columns: int = 4,
    fixed_original_groups: bool = False,
):
    """Plot each chronological trial group with gray trials and a black mean."""

    valid = traces.loc[traces["trial_valid"] & traces[value_column].notna()].copy()
    order_source = traces if fixed_original_groups else valid
    trial_order = (
        order_source[["trial_id", "global_sequence_index", "sequence_index"]]
        .drop_duplicates()
        .sort_values(["global_sequence_index", "sequence_index"])
    )
    if trial_order.empty or valid.empty:
        raise ValueError("No valid trials are available for grouped plotting")
    if fixed_original_groups:
        trial_order["fixed_group"] = (
            (trial_order["sequence_index"].astype(int) - 1) // group_size + 1
        )
        group_numbers = range(
            int(trial_order["fixed_group"].min()),
            int(trial_order["fixed_group"].max()) + 1,
        )
        groups = [
            trial_order.loc[trial_order["fixed_group"].eq(group_number)]
            for group_number in group_numbers
        ]
    else:
        groups = [
            trial_order.iloc[start : start + group_size]
            for start in range(0, len(trial_order), group_size)
        ]
        group_numbers = range(1, len(groups) + 1)
    columns = max(1, min(columns, len(groups)))
    rows = ceil(len(groups) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 2.8 * rows), sharex=True, sharey=True, squeeze=False)
    for axis, group_number, group in zip(axes.flat, group_numbers, groups, strict=False):
        group_traces = valid.loc[valid["trial_id"].isin(group["trial_id"])]
        for _, trial in group_traces.groupby("trial_id", sort=False):
            axis.plot(trial["trial_time_s"], trial[value_column], color="#B8B8B8", linewidth=0.8, alpha=0.75)
        if not group_traces.empty:
            mean = group_traces.groupby("trial_time_s")[value_column].mean()
            axis.plot(mean.index, mean.values, color="black", linewidth=2.0)
        _shade_intervals(axis, stimulus_intervals)
        if fixed_original_groups:
            first = (int(group_number) - 1) * group_size + 1
            last = int(group_number) * group_size
        else:
            first = int(group["sequence_index"].iloc[0])
            last = int(group["sequence_index"].iloc[-1])
        valid_count = int(group_traces["trial_id"].nunique())
        axis.set_title(
            f"Group {group_number}: trials {first}-{last} (n={valid_count})",
            fontsize=10,
        )
        axis.axhline(0, color="#777777", linewidth=0.6)
    for axis in axes.flat[len(groups) :]:
        axis.set_visible(False)
    figure.supxlabel("Time from aligned onset (s)")
    figure.supylabel(value_column)
    figure.tight_layout()
    return figure


def plot_first_last_group_traces(
    traces: pd.DataFrame,
    *,
    group_size: int = 5,
    value_column: str = "pupil_delta_fraction",
    error_band: str = "sem",
    stimulus_intervals: Sequence[tuple[float, float, str]] = (),
    fixed_original_ranges: bool = False,
):
    """Plot first and last group means with parameterized SEM or 95% CI bands."""

    valid = traces.loc[traces["trial_valid"] & traces[value_column].notna()].copy()
    order_source = traces if fixed_original_ranges else valid
    order = (
        order_source[["trial_id", "global_sequence_index", "sequence_index"]]
        .drop_duplicates()
        .sort_values(["global_sequence_index", "sequence_index"])
    )
    if fixed_original_ranges:
        first_number = int(order["sequence_index"].min())
        last_number = int(order["sequence_index"].max())
        if last_number - first_number + 1 < 2 * group_size:
            raise ValueError(f"At least {2 * group_size} original trials are required")
        selections = [
            (
                "First",
                order.loc[
                    order["sequence_index"].between(
                        first_number, first_number + group_size - 1
                    )
                ],
                first_number,
                first_number + group_size - 1,
            ),
            (
                "Last",
                order.loc[
                    order["sequence_index"].between(
                        last_number - group_size + 1, last_number
                    )
                ],
                last_number - group_size + 1,
                last_number,
            ),
        ]
    else:
        if len(order) < 2 * group_size:
            raise ValueError(f"At least {2 * group_size} valid trials are required")
        selections = [
            ("First", order.head(group_size), int(order["sequence_index"].iloc[0]), int(order["sequence_index"].iloc[group_size - 1])),
            ("Last", order.tail(group_size), int(order["sequence_index"].iloc[-group_size]), int(order["sequence_index"].iloc[-1])),
        ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for axis, (label, selected, first, last) in zip(axes, selections, strict=True):
        subset = valid.loc[valid["trial_id"].isin(selected["trial_id"])]
        if subset.empty:
            raise ValueError(f"No valid trials are available in {label.lower()} range")
        summary = _mean_and_error(subset, value_column, error_band)
        axis.fill_between(summary["trial_time_s"], summary["lower"], summary["upper"], color="#777777", alpha=0.25, linewidth=0)
        axis.plot(summary["trial_time_s"], summary["mean"], color="black", linewidth=2.2)
        _shade_intervals(axis, stimulus_intervals)
        valid_count = int(subset["trial_id"].nunique())
        axis.set_title(
            f"{label} group: trials {first}-{last} (n={valid_count}, {error_band})"
        )
        axis.axhline(0, color="#777777", linewidth=0.6)
        axis.set_xlabel("Time from aligned onset (s)")
    axes[0].set_ylabel(value_column)
    figure.tight_layout()
    return figure


def plot_trial_heatmap(
    traces: pd.DataFrame,
    *,
    value_column: str = "pupil_delta_fraction",
):
    """Plot chronological trial-by-time pupil responses."""

    valid = traces.loc[traces["trial_valid"]].copy()
    order = (
        valid[["trial_id", "global_sequence_index", "sequence_index"]]
        .drop_duplicates()
        .sort_values(["global_sequence_index", "sequence_index"])
    )
    pivot = valid.pivot(index="trial_id", columns="trial_time_s", values=value_column).reindex(order["trial_id"])
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        interpolation="none",
        origin="lower",
        extent=[float(pivot.columns.min()), float(pivot.columns.max()), 1, len(pivot)],
        cmap="RdBu_r",
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Time from aligned onset (s)")
    axis.set_ylabel("Valid trial order")
    figure.colorbar(image, ax=axis, label=value_column)
    figure.tight_layout()
    return figure


def plot_metric_trend(
    grouped_metrics: pd.DataFrame,
    *,
    metric: str = "response_auc",
):
    """Plot individual trial metrics and parameterized group means."""

    valid = grouped_metrics.loc[grouped_metrics["trial_valid"] & grouped_metrics[metric].notna()]
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.scatter(valid["condition_trial_number"], valid[metric], s=18, color="#B8B8B8", alpha=0.8)
    summary = valid.groupby("trial_group").agg(
        x=("condition_trial_number", "mean"),
        mean=(metric, "mean"),
        sem=(metric, "sem"),
    )
    axis.errorbar(summary["x"], summary["mean"], yerr=summary["sem"].fillna(0), color="black", marker="o", linewidth=1.8, capsize=3)
    axis.axhline(0, color="#777777", linewidth=0.6)
    axis.set_xlabel("Trial occurrence")
    axis.set_ylabel(metric)
    figure.tight_layout()
    return figure


def plot_condition_traces(
    traces: pd.DataFrame,
    *,
    value_column: str,
    condition_column: str = "sequence_pattern",
    error_band: str = "sem",
):
    """Overlay condition means and error bands while retaining condition identity."""

    colors = {"AAAAB": "#D55E00", "AAAAA": "#0072B2", "AAAAC": "#009E73"}
    figure, axis = plt.subplots(figsize=(10, 4.5))
    valid = traces.loc[traces["trial_valid"] & traces[value_column].notna()]
    for condition, subset in valid.groupby(condition_column, sort=True):
        summary = _mean_and_error(subset, value_column, error_band)
        color = colors.get(str(condition), "#333333")
        axis.fill_between(summary["trial_time_s"], summary["lower"], summary["upper"], color=color, alpha=0.18, linewidth=0)
        axis.plot(summary["trial_time_s"], summary["mean"], color=color, linewidth=2, label=str(condition))
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axhline(0, color="#777777", linewidth=0.6)
    axis.set_xlabel("Time from aligned onset (s)")
    axis.set_ylabel(value_column)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def save_figure_bundle(figure, output_stem: str | Path, dpi: int = 200) -> list[Path]:
    """Save one analysis figure as a reviewable PNG and editable SVG."""

    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    figure.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    return paths
