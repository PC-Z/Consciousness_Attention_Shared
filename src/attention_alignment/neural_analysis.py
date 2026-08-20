from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from .errors import SourceFormatError


def open_neural_activity(
    path: str | Path,
    *,
    frame_count: int,
    cell_count: int,
) -> np.ndarray:
    """Memory-map a 2-D NPY trace and orient it as cells by aligned frames."""

    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    values = np.load(source, mmap_mode="r", allow_pickle=False)
    if values.ndim != 2:
        raise SourceFormatError(f"Neural activity must be 2-D, got {values.shape} in {source}")

    expected = (int(cell_count), int(frame_count))
    transposed = (int(frame_count), int(cell_count))
    if values.shape == expected:
        return values
    if values.shape == transposed:
        return values.T
    raise SourceFormatError(
        f"Neural activity shape {values.shape} does not match cells x frames {expected} "
        f"or frames x cells {transposed}: {source}"
    )


def _decode_matlab_reference_strings(handle: h5py.File, key: str) -> np.ndarray:
    references = np.asarray(handle[key]).reshape(-1, order="F")
    decoded: list[str] = []
    for reference in references:
        if not reference:
            decoded.append("")
            continue
        codes = np.asarray(handle[reference]).reshape(-1, order="F")
        decoded.append("".join(chr(int(code)) for code in codes if int(code) != 0))
    return np.asarray(decoded, dtype=object)


def load_neuron_atlas(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load neuron coordinates, Allen atlas labels, and the CCF outline."""

    source = Path(path).resolve()
    required = {"whole_center_2d_T", "ids", "acs", "names", "CCF"}
    with h5py.File(source, "r") as handle:
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise SourceFormatError(f"Missing atlas datasets in {source}: {missing}")
        centers = np.asarray(handle["whole_center_2d_T"], dtype=float).T
        atlas_ids = np.asarray(handle["ids"], dtype=float).reshape(-1, order="F")
        acronyms = _decode_matlab_reference_strings(handle, "acs")
        region_names = _decode_matlab_reference_strings(handle, "names")
        ccf = np.asarray(handle["CCF"], dtype=np.uint8).T.astype(bool)

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise SourceFormatError(f"whole_center_2d_T must resolve to N x 2, got {centers.shape}")
    lengths = {len(centers), len(atlas_ids), len(acronyms), len(region_names)}
    if len(lengths) != 1:
        raise SourceFormatError(
            "Atlas neuron fields have inconsistent lengths: "
            f"centers={len(centers)}, ids={len(atlas_ids)}, "
            f"acs={len(acronyms)}, names={len(region_names)}"
        )

    table = pd.DataFrame(
        {
            "cell_index": np.arange(len(centers), dtype=np.int64),
            "x": centers[:, 0],
            "y": centers[:, 1],
            "atlas_id": atlas_ids,
            "atlas_acronym": acronyms,
            "atlas_name": region_names,
        }
    )
    return table, ccf


def _validate_frame_times(frame_times_s: np.ndarray, frame_count: int) -> np.ndarray:
    times = np.asarray(frame_times_s, dtype=float)
    if times.ndim != 1 or len(times) != frame_count:
        raise ValueError(f"Expected {frame_count} frame times, got shape {times.shape}")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Calcium frame times must be finite and strictly increasing")
    return times


def downsample_neural_activity(
    activity: np.ndarray,
    frame_times_s: np.ndarray,
    *,
    max_time_bins: int = 2_000,
    cell_indices: Iterable[int] | None = None,
    chunk_size: int = 128,
    method: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce the complete time axis by bin means or evenly spaced samples."""

    if activity.ndim != 2:
        raise ValueError(f"activity must be cells x frames, got {activity.shape}")
    if max_time_bins < 1 or chunk_size < 1:
        raise ValueError("max_time_bins and chunk_size must be positive")
    if method not in {"mean", "sample"}:
        raise ValueError("method must be 'mean' or 'sample'")
    cell_count, frame_count = activity.shape
    times = _validate_frame_times(frame_times_s, frame_count)

    if cell_indices is None:
        indices = np.arange(cell_count, dtype=np.int64)
    else:
        indices = np.asarray(list(cell_indices), dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= cell_count):
        raise IndexError("cell_indices contains an out-of-range cell")

    if method == "sample":
        sample_count = min(max_time_bins, frame_count)
        time_indices = np.unique(
            np.linspace(0, frame_count - 1, sample_count, dtype=np.int64)
        )
        binned_times = times[time_indices]
        binned = np.empty((len(indices), len(time_indices)), dtype=np.float32)
        for output_start in range(0, len(indices), chunk_size):
            output_stop = min(output_start + chunk_size, len(indices))
            batch_indices = indices[output_start:output_stop]
            binned[output_start:output_stop] = np.asarray(
                activity[np.ix_(batch_indices, time_indices)], dtype=np.float32
            )
        return binned_times, binned, indices

    bin_width = max(1, int(np.ceil(frame_count / max_time_bins)))
    starts = np.arange(0, frame_count, bin_width, dtype=np.int64)
    widths = np.diff(np.append(starts, frame_count))
    binned_times = np.add.reduceat(times, starts) / widths
    binned = np.full((len(indices), len(starts)), np.nan, dtype=np.float32)

    for output_start in range(0, len(indices), chunk_size):
        output_stop = min(output_start + chunk_size, len(indices))
        batch_indices = indices[output_start:output_stop]
        block = np.asarray(activity[batch_indices, :], dtype=np.float32)
        finite = np.isfinite(block)
        sums = np.add.reduceat(np.where(finite, block, 0.0), starts, axis=1)
        counts = np.add.reduceat(finite.astype(np.int32), starts, axis=1)
        np.divide(
            sums,
            counts,
            out=binned[output_start:output_stop],
            where=counts > 0,
        )
    return binned_times, binned, indices


def plot_activity_heatmap(
    activity: np.ndarray,
    bin_times_s: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    title: str = "Whole-brain denoised activity",
) -> Figure:
    """Plot a cells-by-time overview produced by downsample_neural_activity."""

    values = np.asarray(activity)
    times = np.asarray(bin_times_s, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(times):
        raise ValueError("Heatmap values must be cells x time bins")
    x0 = float(times[0] / 60.0)
    x1 = float(times[-1] / 60.0)
    if x0 == x1:
        x1 = x0 + 1e-6

    figure, axis = plt.subplots(figsize=(13, 5.5))
    image = axis.imshow(
        values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(x0, x1, 0, values.shape[0]),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set(
        xlabel="Session time (min)",
        ylabel="Neuron index",
        title=f"{title} ({values.shape[0]:,} neurons)",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Denoised fluorescence")
    figure.tight_layout()
    return figure


def plot_neurons_on_atlas(
    atlas: pd.DataFrame,
    ccf: np.ndarray,
    *,
    selected: np.ndarray | Iterable[int] | None = None,
    selection_values: np.ndarray | None = None,
    title: str = "Neuron distribution on atlas",
) -> Figure:
    """Plot all neurons and optionally emphasize a selected subset."""

    required = {"cell_index", "x", "y"}
    if not required.issubset(atlas.columns):
        raise ValueError(f"atlas is missing columns: {sorted(required - set(atlas.columns))}")
    background = np.asarray(ccf, dtype=bool)
    if background.ndim != 2:
        raise ValueError("ccf must be a 2-D mask")

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.imshow(
        ~background,
        cmap="gray",
        origin="lower",
        interpolation="nearest",
        extent=(0, background.shape[1], 0, background.shape[0]),
    )
    axis.scatter(atlas["x"], atlas["y"], s=2, color="#B8BDC5", alpha=0.45, linewidths=0)

    if selected is not None:
        raw_selected = np.asarray(list(selected) if not isinstance(selected, np.ndarray) else selected)
        if raw_selected.dtype == bool:
            if len(raw_selected) != len(atlas):
                raise ValueError("Boolean selected mask must match atlas length")
            selected_rows = atlas.loc[raw_selected]
            selected_indices = np.flatnonzero(raw_selected)
        else:
            selected_indices = raw_selected.astype(np.int64)
            selected_rows = atlas.iloc[selected_indices]

        if selection_values is None:
            axis.scatter(
                selected_rows["x"], selected_rows["y"],
                s=8, color="#D55E00", alpha=0.85, linewidths=0,
            )
        else:
            values = np.asarray(selection_values, dtype=float)
            if len(values) == len(atlas):
                values = values[selected_indices]
            if len(values) != len(selected_rows):
                raise ValueError("selection_values must match all or selected neurons")
            points = axis.scatter(
                selected_rows["x"], selected_rows["y"], s=9,
                c=values, cmap="magma", vmin=0.0, vmax=1.0,
                alpha=0.9, linewidths=0,
            )
            colorbar = figure.colorbar(points, ax=axis, fraction=0.04, pad=0.02)
            colorbar.set_label("Responsive trial fraction")

    axis.set(
        xlim=(0, background.shape[1]),
        ylim=(0, background.shape[0]),
        aspect="equal",
        title=title,
    )
    axis.set_axis_off()
    figure.tight_layout()
    return figure


def build_stimulus_windows(
    events: pd.DataFrame,
    frame_times_s: np.ndarray,
    *,
    phase: str,
    symbols: Iterable[str] | None,
    item_positions: Iterable[int] | None,
    baseline_window_s: tuple[float, float],
    response_window_s: tuple[float, float | None],
    min_window_frames: int = 3,
    require_timing_qc: bool = True,
) -> pd.DataFrame:
    """Convert measured stimulus times into auditable calcium-frame intervals."""

    required = {
        "phase", "symbol", "item_position", "measured_onset_s", "measured_offset_s",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events is missing columns: {missing}")
    baseline_start, baseline_stop = map(float, baseline_window_s)
    response_start = float(response_window_s[0])
    response_stop = response_window_s[1]
    if not baseline_start < baseline_stop <= 0:
        raise ValueError("baseline_window_s must satisfy start < stop <= 0")
    if response_start < 0 or (response_stop is not None and response_start >= response_stop):
        raise ValueError("response_window_s must satisfy 0 <= start < stop")
    if min_window_frames < 1:
        raise ValueError("min_window_frames must be positive")

    times = np.asarray(frame_times_s, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("frame_times_s must be finite and strictly increasing")
    source_events = events.copy()
    if "source_event_index" not in source_events.columns:
        source_events["source_event_index"] = source_events.index.to_numpy(dtype=np.int64)
    selected = source_events.loc[source_events["phase"].eq(phase)].copy()
    if symbols is not None:
        selected = selected.loc[selected["symbol"].isin(list(symbols))]
    if item_positions is not None:
        selected = selected.loc[selected["item_position"].isin(list(item_positions))]
    if require_timing_qc and "timing_qc" in selected.columns:
        selected = selected.loc[selected["timing_qc"].eq("ok")]
    selected = selected.sort_values("measured_onset_s").reset_index(drop=True)
    if selected.empty:
        raise ValueError("No stimulus events match the requested phase/symbol/position filters")

    onset = selected["measured_onset_s"].to_numpy(dtype=float)
    baseline_start_s = onset + baseline_start
    baseline_stop_s = onset + baseline_stop
    response_start_s = onset + response_start
    if response_stop is None:
        response_stop_s = selected["measured_offset_s"].to_numpy(dtype=float)
    else:
        response_stop_s = onset + float(response_stop)

    selected["baseline_start_s"] = baseline_start_s
    selected["baseline_stop_s"] = baseline_stop_s
    selected["response_start_s"] = response_start_s
    selected["response_stop_s"] = response_stop_s
    for label, values in (
        ("baseline_start_index", baseline_start_s),
        ("baseline_stop_index", baseline_stop_s),
        ("response_start_index", response_start_s),
        ("response_stop_index", response_stop_s),
    ):
        selected[label] = np.searchsorted(times, values, side="left").astype(np.int64)

    selected["baseline_frame_count"] = (
        selected["baseline_stop_index"] - selected["baseline_start_index"]
    )
    selected["response_frame_count"] = (
        selected["response_stop_index"] - selected["response_start_index"]
    )
    selected["window_valid"] = (
        (selected["baseline_start_s"] >= times[0])
        & (selected["response_stop_s"] <= times[-1])
        & (selected["baseline_frame_count"] >= min_window_frames)
        & (selected["response_frame_count"] >= min_window_frames)
    )
    return selected


def _interval_values(
    prefix: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    return prefix[:, stops] - prefix[:, starts]


def compute_stimulus_response_statistics(
    activity: np.ndarray,
    windows: pd.DataFrame,
    *,
    cell_indices: Iterable[int] | None = None,
    min_valid_frames: int = 3,
    chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    """Compute reusable per-neuron, per-trial baseline and response moments."""

    if activity.ndim != 2:
        raise ValueError("activity must be cells x frames")
    if min_valid_frames < 1:
        raise ValueError("min_valid_frames must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    required = {
        "baseline_start_index", "baseline_stop_index",
        "response_start_index", "response_stop_index", "window_valid",
    }
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"windows is missing columns: {missing}")
    usable = windows.loc[windows["window_valid"]].copy()
    if usable.empty:
        raise ValueError("No valid stimulus windows are available for neural selection")

    baseline_starts = usable["baseline_start_index"].to_numpy(dtype=np.int64)
    baseline_stops = usable["baseline_stop_index"].to_numpy(dtype=np.int64)
    response_starts = usable["response_start_index"].to_numpy(dtype=np.int64)
    response_stops = usable["response_stop_index"].to_numpy(dtype=np.int64)
    source_frame_count = activity.shape[1]
    all_indices = np.concatenate(
        [baseline_starts, baseline_stops, response_starts, response_stops]
    )
    if np.any(all_indices < 0) or np.any(all_indices > source_frame_count):
        raise IndexError("Stimulus window indices are outside the activity matrix")

    required_frames = np.unique(
        np.concatenate(
            [
                np.arange(start, stop, dtype=np.int64)
                for starts, stops in (
                    (baseline_starts, baseline_stops),
                    (response_starts, response_stops),
                )
                for start, stop in zip(starts, stops, strict=True)
            ]
        )
    )
    baseline_starts = np.searchsorted(required_frames, baseline_starts)
    baseline_stops = np.searchsorted(required_frames, baseline_stops)
    response_starts = np.searchsorted(required_frames, response_starts)
    response_stops = np.searchsorted(required_frames, response_stops)
    compact_frame_count = len(required_frames)

    if cell_indices is None:
        source_cell_indices = np.arange(activity.shape[0], dtype=np.int64)
    else:
        source_cell_indices = np.asarray(list(cell_indices), dtype=np.int64)
    if source_cell_indices.ndim != 1 or len(source_cell_indices) == 0:
        raise ValueError("At least one cell index is required")
    if (
        np.any(source_cell_indices < 0)
        or np.any(source_cell_indices >= activity.shape[0])
        or len(np.unique(source_cell_indices)) != len(source_cell_indices)
    ):
        raise IndexError("cell_indices must be unique, in-range activity rows")

    cell_count = len(source_cell_indices)
    trial_count = len(usable)
    statistics = {
        "baseline_mean": np.full((cell_count, trial_count), np.nan, dtype=np.float32),
        "baseline_std": np.full((cell_count, trial_count), np.nan, dtype=np.float32),
        "response_mean": np.full((cell_count, trial_count), np.nan, dtype=np.float32),
        "response_std": np.full((cell_count, trial_count), np.nan, dtype=np.float32),
        "eligible": np.zeros((cell_count, trial_count), dtype=bool),
        "cell_index": source_cell_indices,
        "window_index": usable.index.to_numpy(dtype=np.int64),
    }

    for cell_start in range(0, cell_count, chunk_size):
        cell_stop = min(cell_start + chunk_size, cell_count)
        batch_cell_indices = source_cell_indices[cell_start:cell_stop]
        block = np.asarray(
            activity[np.ix_(batch_cell_indices, required_frames)], dtype=np.float32
        )
        finite = np.isfinite(block)
        safe = np.where(finite, block, 0.0)

        count_prefix = np.empty((len(block), compact_frame_count + 1), dtype=np.int32)
        count_prefix[:, 0] = 0
        np.cumsum(finite, axis=1, dtype=np.int32, out=count_prefix[:, 1:])
        baseline_counts = _interval_values(count_prefix, baseline_starts, baseline_stops)
        response_counts = _interval_values(count_prefix, response_starts, response_stops)
        del count_prefix, finite

        sum_prefix = np.empty((len(block), compact_frame_count + 1), dtype=np.float64)
        sum_prefix[:, 0] = 0.0
        np.cumsum(safe, axis=1, dtype=np.float64, out=sum_prefix[:, 1:])
        baseline_sums = _interval_values(sum_prefix, baseline_starts, baseline_stops)
        response_sums = _interval_values(sum_prefix, response_starts, response_stops)
        del sum_prefix

        baseline_means = np.full_like(baseline_sums, np.nan, dtype=float)
        response_means = np.full_like(response_sums, np.nan, dtype=float)
        np.divide(baseline_sums, baseline_counts, out=baseline_means, where=baseline_counts > 0)
        np.divide(response_sums, response_counts, out=response_means, where=response_counts > 0)

        square_prefix = np.empty((len(block), compact_frame_count + 1), dtype=np.float64)
        square_prefix[:, 0] = 0.0
        np.cumsum(safe * safe, axis=1, dtype=np.float64, out=square_prefix[:, 1:])
        baseline_square_sums = _interval_values(
            square_prefix, baseline_starts, baseline_stops
        )
        response_square_sums = _interval_values(
            square_prefix, response_starts, response_stops
        )
        del square_prefix, safe, block

        baseline_variances = np.full_like(baseline_square_sums, np.nan, dtype=float)
        response_variances = np.full_like(response_square_sums, np.nan, dtype=float)
        np.divide(
            baseline_square_sums,
            baseline_counts,
            out=baseline_variances,
            where=baseline_counts > 0,
        )
        np.divide(
            response_square_sums,
            response_counts,
            out=response_variances,
            where=response_counts > 0,
        )
        baseline_variances -= baseline_means**2
        response_variances -= response_means**2
        baseline_stds = np.sqrt(np.maximum(baseline_variances, 0.0))
        response_stds = np.sqrt(np.maximum(response_variances, 0.0))

        eligible = (
            (baseline_counts >= min_valid_frames)
            & (response_counts >= min_valid_frames)
            & np.isfinite(baseline_means)
            & np.isfinite(response_means)
        )
        row_slice = slice(cell_start, cell_stop)
        statistics["baseline_mean"][row_slice] = baseline_means
        statistics["baseline_std"][row_slice] = baseline_stds
        statistics["response_mean"][row_slice] = response_means
        statistics["response_std"][row_slice] = response_stds
        statistics["eligible"][row_slice] = eligible
    return statistics


def summarize_stimulus_responses(
    statistics: dict[str, np.ndarray],
    *,
    std_multiplier: float = 3.0,
    require_response_std_gt_baseline: bool = True,
    min_responsive_trials: int = 4,
    min_responsive_fraction: float = 0.5,
) -> pd.DataFrame:
    """Apply adjustable thresholds to cached baseline/response statistics."""

    if std_multiplier < 0 or min_responsive_trials < 1:
        raise ValueError("Selection thresholds must be non-negative and non-zero where required")
    if not 0 <= min_responsive_fraction <= 1:
        raise ValueError("min_responsive_fraction must be between 0 and 1")
    required = {"baseline_mean", "baseline_std", "response_mean", "response_std", "eligible"}
    missing = sorted(required - set(statistics))
    if missing:
        raise ValueError(f"statistics is missing arrays: {missing}")

    baseline_means = np.asarray(statistics["baseline_mean"])
    baseline_stds = np.asarray(statistics["baseline_std"])
    response_means = np.asarray(statistics["response_mean"])
    response_stds = np.asarray(statistics["response_std"])
    eligible = np.asarray(statistics["eligible"], dtype=bool).copy()
    shapes = {
        baseline_means.shape,
        baseline_stds.shape,
        response_means.shape,
        response_stds.shape,
        eligible.shape,
    }
    if len(shapes) != 1 or baseline_means.ndim != 2:
        raise ValueError("All response statistic arrays must share a cells x trials shape")

    responsive = eligible & (
        response_means > baseline_means + std_multiplier * baseline_stds
    )
    if require_response_std_gt_baseline:
        responsive &= response_stds > baseline_stds

    responsive_counts = responsive.sum(axis=1, dtype=np.int64)
    eligible_counts = eligible.sum(axis=1, dtype=np.int64)
    deltas = np.where(eligible, response_means - baseline_means, 0.0)
    mean_deltas = np.full(len(eligible), np.nan, dtype=float)
    np.divide(
        deltas.sum(axis=1),
        eligible_counts,
        out=mean_deltas,
        where=eligible_counts > 0,
    )

    cell_count = len(eligible)
    cell_indices = np.asarray(
        statistics.get("cell_index", np.arange(cell_count)), dtype=np.int64
    )
    if cell_indices.shape != (cell_count,):
        raise ValueError("statistics['cell_index'] must match the statistic row count")
    responsive_fractions = np.full(cell_count, np.nan, dtype=float)
    np.divide(
        responsive_counts,
        eligible_counts,
        out=responsive_fractions,
        where=eligible_counts > 0,
    )
    selected = (
        (responsive_counts >= min_responsive_trials)
        & (responsive_fractions >= min_responsive_fraction)
    )
    return pd.DataFrame(
        {
            "cell_index": cell_indices,
            "eligible_trial_count": eligible_counts,
            "responsive_trial_count": responsive_counts,
            "responsive_trial_fraction": responsive_fractions,
            "mean_response_minus_baseline": mean_deltas,
            "selected": selected,
        }
    )


def summarize_mean_stimulus_responses(
    statistics: dict[str, np.ndarray],
    *,
    std_multiplier: float = 1.97,
    min_eligible_trials: int = 4,
) -> pd.DataFrame:
    """Screen neurons by their across-trial mean response above baseline.

    The baseline scale pools within-trial variance and between-trial baseline
    shifts. Unlike :func:`summarize_stimulus_responses`, this screen does not
    require a neuron to cross a threshold on any fixed fraction of trials.
    """

    if std_multiplier < 0 or min_eligible_trials < 1:
        raise ValueError("Selection thresholds must be non-negative and non-zero")
    required = {"baseline_mean", "baseline_std", "response_mean", "eligible"}
    missing = sorted(required - set(statistics))
    if missing:
        raise ValueError(f"statistics is missing arrays: {missing}")

    baseline_means = np.asarray(statistics["baseline_mean"], dtype=float)
    baseline_stds = np.asarray(statistics["baseline_std"], dtype=float)
    response_means = np.asarray(statistics["response_mean"], dtype=float)
    eligible = np.asarray(statistics["eligible"], dtype=bool).copy()
    shapes = {
        baseline_means.shape,
        baseline_stds.shape,
        response_means.shape,
        eligible.shape,
    }
    if len(shapes) != 1 or baseline_means.ndim != 2:
        raise ValueError("All response statistic arrays must share a cells x trials shape")

    eligible &= (
        np.isfinite(baseline_means)
        & np.isfinite(baseline_stds)
        & np.isfinite(response_means)
    )
    eligible_counts = eligible.sum(axis=1, dtype=np.int64)

    def eligible_mean(values: np.ndarray) -> np.ndarray:
        result = np.full(values.shape[0], np.nan, dtype=float)
        np.divide(
            np.where(eligible, values, 0.0).sum(axis=1),
            eligible_counts,
            out=result,
            where=eligible_counts > 0,
        )
        return result

    mean_baseline = eligible_mean(baseline_means)
    mean_response = eligible_mean(response_means)
    baseline_second_moment = eligible_mean(baseline_stds**2 + baseline_means**2)
    pooled_baseline_sd = np.sqrt(
        np.maximum(baseline_second_moment - mean_baseline**2, 0.0)
    )
    mean_delta = mean_response - mean_baseline
    response_z = np.full_like(mean_delta, np.nan)
    np.divide(
        mean_delta,
        pooled_baseline_sd,
        out=response_z,
        where=np.isfinite(pooled_baseline_sd) & (pooled_baseline_sd > 0),
    )

    cell_count = len(eligible)
    cell_indices = np.asarray(
        statistics.get("cell_index", np.arange(cell_count)), dtype=np.int64
    )
    if cell_indices.shape != (cell_count,):
        raise ValueError("statistics['cell_index'] must match the statistic row count")
    selected = (
        (eligible_counts >= min_eligible_trials)
        & np.isfinite(response_z)
        & (response_z > std_multiplier)
    )
    return pd.DataFrame(
        {
            "cell_index": cell_indices,
            "eligible_trial_count": eligible_counts,
            "mean_baseline": mean_baseline,
            "mean_response": mean_response,
            "mean_response_minus_baseline": mean_delta,
            "pooled_baseline_sd": pooled_baseline_sd,
            "mean_response_z": response_z,
            "selected": selected,
        }
    )


def select_stimulus_responsive_neurons(
    activity: np.ndarray,
    windows: pd.DataFrame,
    *,
    cell_indices: Iterable[int] | None = None,
    std_multiplier: float = 3.0,
    require_response_std_gt_baseline: bool = True,
    min_responsive_trials: int = 4,
    min_responsive_fraction: float = 0.5,
    min_valid_frames: int = 3,
    chunk_size: int = 128,
) -> pd.DataFrame:
    """Compute response statistics and apply the reference screening rule."""

    statistics = compute_stimulus_response_statistics(
        activity,
        windows,
        cell_indices=cell_indices,
        min_valid_frames=min_valid_frames,
        chunk_size=chunk_size,
    )
    return summarize_stimulus_responses(
        statistics,
        std_multiplier=std_multiplier,
        require_response_std_gt_baseline=require_response_std_gt_baseline,
        min_responsive_trials=min_responsive_trials,
        min_responsive_fraction=min_responsive_fraction,
    )


def extract_event_aligned_neural_traces(
    activity: np.ndarray,
    frame_times_s: np.ndarray,
    windows: pd.DataFrame,
    cell_indices: Iterable[int],
    *,
    trace_window_s: tuple[float, float],
    baseline_window_s: tuple[float, float],
    sample_rate_hz: float | None = None,
) -> dict[str, np.ndarray]:
    """Interpolate selected cells onto a shared event-relative time axis."""

    if activity.ndim != 2:
        raise ValueError("activity must be cells x frames")
    times = _validate_frame_times(frame_times_s, activity.shape[1])
    cells = np.asarray(list(cell_indices), dtype=np.int64)
    if cells.ndim != 1 or len(cells) == 0:
        raise ValueError("At least one selected cell is required for trace extraction")
    if np.any(cells < 0) or np.any(cells >= activity.shape[0]):
        raise IndexError("cell_indices contains an out-of-range cell")

    trace_start, trace_stop = map(float, trace_window_s)
    baseline_start, baseline_stop = map(float, baseline_window_s)
    if not trace_start < trace_stop:
        raise ValueError("trace_window_s must satisfy start < stop")
    if not trace_start <= baseline_start < baseline_stop <= trace_stop:
        raise ValueError("baseline_window_s must lie inside trace_window_s")
    if "measured_onset_s" not in windows.columns:
        raise ValueError("windows must contain measured_onset_s")
    if "window_valid" in windows.columns:
        usable = windows.loc[windows["window_valid"]].copy()
    else:
        usable = windows.copy()
    if usable.empty:
        raise ValueError("No valid stimulus windows are available for trace extraction")

    if sample_rate_hz is None:
        sample_rate_hz = 1.0 / float(np.median(np.diff(times)))
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    sample_count = int(round((trace_stop - trace_start) * sample_rate_hz)) + 1
    relative_times = np.linspace(trace_start, trace_stop, sample_count, dtype=float)
    baseline_mask = (
        (relative_times >= baseline_start) & (relative_times < baseline_stop)
    )
    if baseline_mask.sum() < 2:
        raise ValueError("The aligned grid must contain at least two baseline samples")

    aligned = np.full(
        (len(cells), len(usable), len(relative_times)), np.nan, dtype=np.float32
    )
    onsets = usable["measured_onset_s"].to_numpy(dtype=float)
    for trial_index, onset in enumerate(onsets):
        target_times = onset + relative_times
        source_start = max(0, int(np.searchsorted(times, target_times[0], side="left")) - 1)
        source_stop = min(
            len(times), int(np.searchsorted(times, target_times[-1], side="right")) + 1
        )
        source_indices = np.arange(source_start, source_stop, dtype=np.int64)
        source_times = times[source_indices]
        block = np.asarray(activity[np.ix_(cells, source_indices)], dtype=np.float32)
        for local_cell_index, values in enumerate(block):
            finite = np.isfinite(values)
            if finite.sum() >= 2:
                aligned[local_cell_index, trial_index] = np.interp(
                    target_times,
                    source_times[finite],
                    values[finite],
                    left=np.nan,
                    right=np.nan,
                )

    baseline_values = aligned[:, :, baseline_mask]
    baseline_counts = np.isfinite(baseline_values).sum(axis=2)
    baseline_sums = np.nansum(baseline_values, axis=2)
    baseline_means = np.full(baseline_sums.shape, np.nan, dtype=np.float32)
    np.divide(
        baseline_sums,
        baseline_counts,
        out=baseline_means,
        where=baseline_counts > 0,
    )
    delta = aligned - baseline_means[:, :, np.newaxis]
    return {
        "relative_time_s": relative_times,
        "cell_index": cells,
        "window_index": usable.index.to_numpy(dtype=np.int64),
        "values": aligned,
        "baseline_mean": baseline_means,
        "delta": delta.astype(np.float32, copy=False),
    }


def build_sequence_trace_windows(
    stimulus_windows: pd.DataFrame,
    *,
    sequence_indices: Iterable[int],
    item_positions: Iterable[int] = (1, 2, 3, 4, 5),
) -> dict[str, pd.DataFrame]:
    """Build P1-anchored sequence windows and measured item timing."""

    required = {
        "sequence_index", "item_position", "symbol",
        "measured_onset_s", "measured_offset_s",
    }
    missing = sorted(required - set(stimulus_windows.columns))
    if missing:
        raise ValueError(f"stimulus_windows is missing columns: {missing}")
    sequences = tuple(dict.fromkeys(int(value) for value in sequence_indices))
    positions = tuple(dict.fromkeys(int(value) for value in item_positions))
    if not sequences or not positions:
        raise ValueError("sequence_indices and item_positions cannot be empty")

    selected = stimulus_windows.loc[
        stimulus_windows["sequence_index"].isin(sequences)
        & stimulus_windows["item_position"].isin(positions)
    ].copy()
    if "window_valid" in selected.columns:
        selected = selected.loc[selected["window_valid"]]
    sequence_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    for sequence_index in sequences:
        group = selected.loc[selected["sequence_index"].eq(sequence_index)].copy()
        group = group.sort_values("item_position")
        observed = tuple(group["item_position"].astype(int))
        if observed != positions:
            raise ValueError(
                f"Sequence {sequence_index} has positions {observed}, expected {positions}"
            )
        sequence_onset = float(group.iloc[0]["measured_onset_s"])
        sequence_offset = float(group.iloc[-1]["measured_offset_s"])
        if not np.isfinite(sequence_onset) or not sequence_onset < sequence_offset:
            raise ValueError(f"Sequence {sequence_index} has invalid measured timing")
        sequence_rows.append({
            "sequence_index": sequence_index,
            "sequence_pattern": str(group.iloc[0].get("sequence_pattern", "")),
            "measured_onset_s": sequence_onset,
            "measured_offset_s": sequence_offset,
            "sequence_duration_s": sequence_offset - sequence_onset,
            "window_valid": True,
        })
        for row in group.itertuples(index=False):
            timing_rows.append({
                "sequence_index": sequence_index,
                "item_position": int(row.item_position),
                "symbol": str(row.symbol),
                "relative_onset_s": float(row.measured_onset_s) - sequence_onset,
                "relative_offset_s": float(row.measured_offset_s) - sequence_onset,
            })

    timing = pd.DataFrame(timing_rows)
    stimulus_timing = (
        timing.groupby(["item_position", "symbol"], as_index=False, sort=True)
        .agg(
            relative_onset_s=("relative_onset_s", "median"),
            relative_offset_s=("relative_offset_s", "median"),
            sequence_count=("sequence_index", "nunique"),
        )
        .sort_values("item_position")
        .reset_index(drop=True)
    )
    return {
        "sequence_windows": pd.DataFrame(sequence_rows),
        "stimulus_timing": stimulus_timing,
    }


def prepare_selected_neuron_heatmap(
    aligned_delta: np.ndarray,
    relative_time_s: np.ndarray,
    *,
    response_window_s: tuple[float, float],
    normalization: str = "minmax",
) -> dict[str, np.ndarray]:
    """Average trials, normalize rows, and sort neurons by response peak time."""

    delta = np.asarray(aligned_delta, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    if delta.ndim != 3 or delta.shape[2] != len(times):
        raise ValueError("aligned_delta must be cells x trials x relative time")
    if normalization not in {"minmax", "none"}:
        raise ValueError("normalization must be 'minmax' or 'none'")
    response_start, response_stop = map(float, response_window_s)
    response_mask = (times >= response_start) & (times <= response_stop)
    if not response_mask.any():
        raise ValueError("response_window_s does not overlap relative_time_s")

    valid_counts = np.isfinite(delta).sum(axis=1)
    sums = np.nansum(delta, axis=1)
    mean_delta = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, valid_counts, out=mean_delta, where=valid_counts > 0)

    heatmap = mean_delta.copy()
    if normalization == "minmax":
        finite_rows = np.isfinite(heatmap).any(axis=1)
        row_min = np.zeros(len(heatmap), dtype=float)
        row_max = np.ones(len(heatmap), dtype=float)
        row_min[finite_rows] = np.nanmin(heatmap[finite_rows], axis=1)
        row_max[finite_rows] = np.nanmax(heatmap[finite_rows], axis=1)
        row_range = row_max - row_min
        heatmap = np.divide(
            heatmap - row_min[:, np.newaxis],
            row_range[:, np.newaxis],
            out=np.zeros_like(heatmap),
            where=row_range[:, np.newaxis] > 0,
        )

    response_values = mean_delta[:, response_mask]
    response_times = times[response_mask]
    peak_times = np.full(len(mean_delta), np.inf, dtype=float)
    finite_rows = np.isfinite(response_values).any(axis=1)
    peak_indices = np.argmax(
        np.where(np.isfinite(response_values[finite_rows]), response_values[finite_rows], -np.inf),
        axis=1,
    )
    peak_times[finite_rows] = response_times[peak_indices]
    order = np.argsort(peak_times, kind="stable")
    return {
        "mean_delta": mean_delta.astype(np.float32),
        "heatmap": heatmap[order].astype(np.float32),
        "order": order.astype(np.int64),
        "peak_time_s": peak_times.astype(np.float32),
    }


def screen_stimulus_conditions(
    activity: np.ndarray,
    frame_times_s: np.ndarray,
    events: pd.DataFrame,
    conditions: Iterable[dict[str, object]],
    *,
    baseline_window_s: tuple[float, float],
    response_window_s: tuple[float, float | None],
    trace_window_s: tuple[float, float],
    std_multiplier: float = 3.0,
    require_response_std_gt_baseline: bool = True,
    min_responsive_trials: int = 4,
    min_responsive_fraction: float = 0.5,
    min_valid_frames: int = 3,
    chunk_size: int = 128,
    sample_rate_hz: float | None = None,
    heatmap_normalization: str = "minmax",
) -> dict[str, dict[str, object]]:
    """Independently screen several event conditions after one shared source scan.

    Every condition receives its own response thresholding, selected-cell set, and
    peak-time order. The shared statistics scan is only a performance optimization;
    trials and selection decisions are never pooled across conditions.
    """

    required_fields = {"key", "label", "phase", "symbols", "item_positions"}
    specifications: list[dict[str, object]] = []
    keys: set[str] = set()
    for raw_condition in conditions:
        condition = dict(raw_condition)
        missing = sorted(required_fields - set(condition))
        if missing:
            raise ValueError(f"Condition is missing fields: {missing}")
        key = str(condition["key"])
        if not key or key in keys:
            raise ValueError("Condition keys must be non-empty and unique")
        keys.add(key)
        condition["key"] = key
        condition["label"] = str(condition["label"])
        condition["phase"] = str(condition["phase"])
        condition["symbols"] = tuple(str(value) for value in condition["symbols"])
        condition["item_positions"] = tuple(
            int(value) for value in condition["item_positions"]
        )
        specifications.append(condition)
    if not specifications:
        raise ValueError("At least one condition is required")

    windows_by_key: dict[str, pd.DataFrame] = {}
    combined_windows: list[pd.DataFrame] = []
    for condition in specifications:
        key = str(condition["key"])
        windows = build_stimulus_windows(
            events,
            frame_times_s,
            phase=str(condition["phase"]),
            symbols=condition["symbols"],
            item_positions=condition["item_positions"],
            baseline_window_s=baseline_window_s,
            response_window_s=response_window_s,
            min_window_frames=min_valid_frames,
        )
        windows = windows.reset_index(drop=True)
        windows_by_key[key] = windows
        tagged = windows.copy()
        tagged["condition_key"] = key
        combined_windows.append(tagged)

    all_windows = pd.concat(combined_windows, ignore_index=True)
    shared_statistics = compute_stimulus_response_statistics(
        activity,
        all_windows,
        min_valid_frames=min_valid_frames,
        chunk_size=chunk_size,
    )
    valid_condition_keys = all_windows.loc[
        all_windows["window_valid"], "condition_key"
    ].to_numpy(dtype=object)
    if shared_statistics["eligible"].shape[1] != len(valid_condition_keys):
        raise RuntimeError("Condition labels do not match the shared trial statistics")

    results: dict[str, dict[str, object]] = {}
    statistic_names = (
        "baseline_mean", "baseline_std", "response_mean", "response_std", "eligible"
    )
    for condition in specifications:
        key = str(condition["key"])
        condition_trial_mask = valid_condition_keys == key
        condition_statistics = {
            name: shared_statistics[name][:, condition_trial_mask]
            for name in statistic_names
        }
        selection = summarize_stimulus_responses(
            condition_statistics,
            std_multiplier=std_multiplier,
            require_response_std_gt_baseline=require_response_std_gt_baseline,
            min_responsive_trials=min_responsive_trials,
            min_responsive_fraction=min_responsive_fraction,
        )
        selected_cell_indices = selection.loc[
            selection["selected"], "cell_index"
        ].to_numpy(dtype=np.int64)
        windows = windows_by_key[key]
        valid_windows = windows.loc[windows["window_valid"]]
        stimulus_duration_s = float(np.median(
            valid_windows["measured_offset_s"] - valid_windows["measured_onset_s"]
        ))

        aligned: dict[str, np.ndarray] | None = None
        heatmap: dict[str, np.ndarray] | None = None
        if len(selected_cell_indices) and len(valid_windows):
            aligned = extract_event_aligned_neural_traces(
                activity,
                frame_times_s,
                windows,
                selected_cell_indices,
                trace_window_s=trace_window_s,
                baseline_window_s=baseline_window_s,
                sample_rate_hz=sample_rate_hz,
            )
            response_stop_s = (
                stimulus_duration_s
                if response_window_s[1] is None
                else float(response_window_s[1])
            )
            heatmap = prepare_selected_neuron_heatmap(
                aligned["delta"],
                aligned["relative_time_s"],
                response_window_s=(float(response_window_s[0]), response_stop_s),
                normalization=heatmap_normalization,
            )

        results[key] = {
            "condition": condition,
            "windows": windows,
            "selection": selection,
            "selected_cell_index": selected_cell_indices,
            "valid_trial_count": int(condition_trial_mask.sum()),
            "stimulus_duration_s": stimulus_duration_s,
            "aligned": aligned,
            "heatmap": heatmap,
        }
    return results


def summarize_condition_screens(
    results: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Create one auditable row per independently screened condition."""

    rows: list[dict[str, object]] = []
    for key, result in results.items():
        condition = result["condition"]
        selection = result["selection"]
        selected = selection["selected"].to_numpy(dtype=bool)
        rows.append(
            {
                "condition_key": key,
                "condition_label": condition["label"],
                "phase": condition["phase"],
                "symbols": ",".join(condition["symbols"]),
                "item_positions": ",".join(
                    str(value) for value in condition["item_positions"]
                ),
                "valid_trials": result["valid_trial_count"],
                "total_neurons": len(selection),
                "selected_neurons": int(selected.sum()),
                "selected_fraction": float(selected.mean()),
            }
        )
    return pd.DataFrame(rows)


def _condition_axes(
    condition_count: int,
    *,
    panel_size: tuple[float, float],
) -> tuple[Figure, np.ndarray]:
    columns = min(2, condition_count)
    rows = int(np.ceil(condition_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel_size[0] * columns, panel_size[1] * rows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)
    for axis in flat_axes[condition_count:]:
        axis.set_visible(False)
    return figure, flat_axes


def plot_condition_neuron_traces(
    results: dict[str, dict[str, object]],
    *,
    baseline_window_s: tuple[float, float],
    max_individual_traces: int = 300,
) -> Figure:
    """Plot each independently selected population in a separate trace panel."""

    if max_individual_traces < 1:
        raise ValueError("max_individual_traces must be positive")
    figure, axes = _condition_axes(len(results), panel_size=(6.0, 4.0))
    for axis, result in zip(axes, results.values(), strict=False):
        condition = result["condition"]
        aligned = result["aligned"]
        heatmap = result["heatmap"]
        title = (
            f'{condition["label"]}\n'
            f'n={len(result["selected_cell_index"]):,}, trials={result["valid_trial_count"]:,}'
        )
        if aligned is None or heatmap is None:
            axis.text(0.5, 0.5, "No neurons passed", ha="center", va="center")
            axis.set_title(title)
            axis.set_axis_off()
            continue
        times = aligned["relative_time_s"]
        traces = heatmap["mean_delta"]
        if len(traces) > max_individual_traces:
            shown = np.linspace(
                0, len(traces) - 1, max_individual_traces, dtype=np.int64
            )
            traces_to_plot = traces[shown]
        else:
            traces_to_plot = traces
        axis.plot(times, traces_to_plot.T, color="#B8BDC5", alpha=0.22, linewidth=0.55)
        mean, sem = _mean_and_sem(traces, axis=0)
        axis.fill_between(times, mean - sem, mean + sem, color="#111111", alpha=0.18)
        axis.plot(times, mean, color="#111111", linewidth=1.8)
        axis.axvspan(*baseline_window_s, color="#D9DEE3", alpha=0.35, linewidth=0)
        axis.axvspan(
            0,
            float(result["stimulus_duration_s"]),
            color="#E69F00",
            alpha=0.12,
            linewidth=0,
        )
        axis.axhline(0, color="#777777", linewidth=0.6)
        axis.axvline(0, color="#111111", linewidth=0.8)
        axis.set(
            title=title,
            xlabel="Time from P5 onset (s)",
            ylabel="Baseline-subtracted fluorescence",
        )
    figure.suptitle("Independent P5 response screens: selected-neuron traces", y=1.01)
    return figure


def plot_condition_neuron_heatmaps(
    results: dict[str, dict[str, object]],
    *,
    normalization: str = "minmax",
) -> Figure:
    """Plot condition-specific neuron heatmaps with independent row sorting."""

    if normalization not in {"minmax", "none"}:
        raise ValueError("normalization must be 'minmax' or 'none'")
    figure, axes = _condition_axes(len(results), panel_size=(6.0, 4.5))
    images = []
    for axis, result in zip(axes, results.values(), strict=False):
        condition = result["condition"]
        aligned = result["aligned"]
        heatmap = result["heatmap"]
        title = (
            f'{condition["label"]}\n'
            f'n={len(result["selected_cell_index"]):,}, trials={result["valid_trial_count"]:,}'
        )
        if aligned is None or heatmap is None:
            axis.text(0.5, 0.5, "No neurons passed", ha="center", va="center")
            axis.set_title(title)
            axis.set_axis_off()
            continue
        values = heatmap["heatmap"]
        times = aligned["relative_time_s"]
        if normalization == "minmax":
            vmin, vmax, cmap = 0.0, 1.0, "coolwarm"
        else:
            limit = float(np.nanpercentile(np.abs(values), 99))
            limit = limit if np.isfinite(limit) and limit > 0 else 1.0
            vmin, vmax, cmap = -limit, limit, "RdBu_r"
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(times[0], times[-1], 0, len(values)),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        axis.axvline(0, color="#111111", linewidth=0.8)
        axis.axvline(
            float(result["stimulus_duration_s"]),
            color="#111111",
            linewidth=0.7,
            linestyle="--",
        )
        axis.set(
            title=title,
            xlabel="Time from P5 onset (s)",
            ylabel="Selected neuron (independent peak order)",
        )
    if images:
        colorbar = figure.colorbar(
            images[0],
            ax=[axis for axis in axes if axis.get_visible()],
            pad=0.02,
            shrink=0.82,
        )
        colorbar.set_label(
            "Within-neuron min-max" if normalization == "minmax" else "Fluorescence"
        )
    figure.suptitle("Independent P5 response screens: neuron heatmaps", y=1.01)
    return figure


def plot_condition_neurons_on_atlas(
    atlas: pd.DataFrame,
    ccf: np.ndarray,
    results: dict[str, dict[str, object]],
) -> Figure:
    """Map every independently selected population onto the same atlas geometry."""

    required = {"cell_index", "x", "y"}
    if not required.issubset(atlas.columns):
        raise ValueError(f"atlas is missing columns: {sorted(required - set(atlas.columns))}")
    background = np.asarray(ccf, dtype=bool)
    if background.ndim != 2:
        raise ValueError("ccf must be a 2-D mask")
    figure, axes = _condition_axes(len(results), panel_size=(5.0, 5.0))
    points = []
    for axis, result in zip(axes, results.values(), strict=False):
        condition = result["condition"]
        selection = result["selection"]
        selected = selection["selected"].to_numpy(dtype=bool)
        selected_rows = atlas.loc[selected]
        axis.imshow(
            ~background,
            cmap="gray",
            origin="lower",
            interpolation="nearest",
            extent=(0, background.shape[1], 0, background.shape[0]),
        )
        axis.scatter(
            atlas["x"], atlas["y"], s=1.5, color="#B8BDC5", alpha=0.32, linewidths=0
        )
        if selected.any():
            point = axis.scatter(
                selected_rows["x"],
                selected_rows["y"],
                s=8,
                c=selection.loc[selected, "responsive_trial_fraction"],
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                alpha=0.9,
                linewidths=0,
            )
            points.append(point)
        axis.set(
            xlim=(0, background.shape[1]),
            ylim=(0, background.shape[0]),
            aspect="equal",
            title=(
                f'{condition["label"]}\n'
                f'n={int(selected.sum()):,}, trials={result["valid_trial_count"]:,}'
            ),
        )
        axis.set_axis_off()
    if points:
        colorbar = figure.colorbar(
            points[0],
            ax=[axis for axis in axes if axis.get_visible()],
            pad=0.02,
            shrink=0.82,
        )
        colorbar.set_label("Responsive trial fraction")
    figure.suptitle("Independent P5 response screens: atlas distributions", y=1.01)
    return figure


def _mean_and_sem(values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=float)
    counts = np.isfinite(data).sum(axis=axis)
    sums = np.nansum(data, axis=axis)
    mean = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=mean, where=counts > 0)
    centered = np.where(np.isfinite(data), data - np.expand_dims(mean, axis), 0.0)
    squared = np.sum(centered**2, axis=axis)
    standard_deviation = np.full(sums.shape, np.nan, dtype=float)
    np.divide(
        squared,
        counts - 1,
        out=standard_deviation,
        where=counts > 1,
    )
    standard_deviation = np.sqrt(standard_deviation)
    sem = np.full(sums.shape, np.nan, dtype=float)
    np.divide(
        standard_deviation,
        np.sqrt(counts),
        out=sem,
        where=counts > 0,
    )
    return mean, sem


def _median_and_iqr(
    values: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=float)
    return (
        np.nanmedian(data, axis=axis),
        np.nanpercentile(data, 25, axis=axis),
        np.nanpercentile(data, 75, axis=axis),
    )


def select_atlas_cell_indices(
    atlas: pd.DataFrame,
    region_acronym: str | None = "VISp",
) -> np.ndarray:
    """Return atlas cell indices for one Allen region without matching child names."""

    required = {"cell_index", "atlas_acronym"}
    if not required.issubset(atlas.columns):
        raise ValueError(f"atlas is missing columns: {sorted(required - set(atlas.columns))}")
    if region_acronym is None or str(region_acronym).lower() == "all":
        return atlas["cell_index"].to_numpy(dtype=np.int64)
    region = str(region_acronym).strip()
    if not region:
        raise ValueError("region_acronym cannot be empty")
    # Allen layer suffixes start with a digit (VISp1, VISp2/3, ...). This keeps
    # VISp separate from the neighboring acronym VISpm.
    pattern = re.compile(rf"^{re.escape(region)}(?:$|[0-9].*)")
    mask = atlas["atlas_acronym"].fillna("").astype(str).str.match(pattern)
    indices = atlas.loc[mask, "cell_index"].to_numpy(dtype=np.int64)
    if len(indices) == 0:
        raise ValueError(f"No atlas neurons match region acronym {region!r}")
    return indices


def subset_stimulus_response_statistics(
    statistics: dict[str, np.ndarray],
    trial_mask: Iterable[bool],
) -> dict[str, np.ndarray]:
    """Subset cached response statistics along the trial axis."""

    mask = np.asarray(list(trial_mask), dtype=bool)
    required = {
        "baseline_mean", "baseline_std", "response_mean", "response_std",
        "eligible", "window_index",
    }
    missing = sorted(required - set(statistics))
    if missing:
        raise ValueError(f"statistics is missing arrays: {missing}")
    shape = np.asarray(statistics["eligible"]).shape
    if len(shape) != 2 or mask.shape != (shape[1],):
        raise ValueError("trial_mask must match the statistics trial axis")
    if not mask.any():
        raise ValueError("trial_mask selects no trials")
    result: dict[str, np.ndarray] = {}
    for key in ("baseline_mean", "baseline_std", "response_mean", "response_std", "eligible"):
        result[key] = np.asarray(statistics[key])[:, mask]
    result["window_index"] = np.asarray(statistics["window_index"])[mask]
    result["cell_index"] = np.asarray(
        statistics.get("cell_index", np.arange(shape[0])), dtype=np.int64
    )
    return result


def response_delta_from_statistics(
    statistics: dict[str, np.ndarray],
    *,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """Return eligible response-minus-baseline scores, optionally standardized."""

    response = np.asarray(statistics["response_mean"], dtype=float)
    baseline = np.asarray(statistics["baseline_mean"], dtype=float)
    eligible = np.asarray(statistics["eligible"], dtype=bool)
    if response.shape != baseline.shape or response.shape != eligible.shape:
        raise ValueError("Response statistic arrays must share a cells x trials shape")
    delta = np.where(eligible, response - baseline, np.nan)
    if scale is not None:
        divisor = np.asarray(scale, dtype=float)
        if divisor.shape != (delta.shape[0],):
            raise ValueError("scale must have one value per statistic row")
        delta = np.divide(
            delta,
            divisor[:, np.newaxis],
            out=np.full_like(delta, np.nan),
            where=np.isfinite(divisor[:, np.newaxis]) & (divisor[:, np.newaxis] > 0),
        )
    return delta


def estimate_baseline_noise_scale(
    statistics: dict[str, np.ndarray],
    *,
    floor_percentile: float = 5.0,
) -> np.ndarray:
    """Estimate one stable baseline-SD scale per neuron for cross-condition plots."""

    if not 0 <= floor_percentile < 100:
        raise ValueError("floor_percentile must be in [0, 100)")
    baseline_std = np.asarray(statistics["baseline_std"], dtype=float)
    eligible = np.asarray(statistics["eligible"], dtype=bool)
    if baseline_std.shape != eligible.shape or baseline_std.ndim != 2:
        raise ValueError("baseline_std and eligible must share a cells x trials shape")
    values = np.where(eligible & (baseline_std > 0), baseline_std, np.nan)
    scale = np.nanmedian(values, axis=1)
    positive = scale[np.isfinite(scale) & (scale > 0)]
    if len(positive) == 0:
        raise ValueError("No positive baseline variability is available for standardization")
    floor = float(np.percentile(positive, floor_percentile))
    scale = np.where(np.isfinite(scale) & (scale > 0), np.maximum(scale, floor), np.nan)
    return scale.astype(np.float32)


def robust_normalize_neuron_trial_traces(
    trace_groups: Iterable[np.ndarray],
    *,
    robust_percentile: float = 99.0,
    floor_percentile: float = 5.0,
    clip_limit: float = 1.0,
) -> dict[str, object]:
    """Scale several neuron-by-trial trace groups once per neuron for display."""

    if not 0 < robust_percentile <= 100:
        raise ValueError("robust_percentile must be in (0, 100]")
    if not 0 <= floor_percentile < 100:
        raise ValueError("floor_percentile must be in [0, 100)")
    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("clip_limit must be positive")

    groups = tuple(np.asarray(group, dtype=float) for group in trace_groups)
    if not groups:
        raise ValueError("At least one trace group is required")
    neuron_count = groups[0].shape[0] if groups[0].ndim == 3 else -1
    if neuron_count < 1 or any(
        group.ndim != 3 or group.shape[0] != neuron_count for group in groups
    ):
        raise ValueError("Every trace group must be neurons x trials x time")

    scale = np.full(neuron_count, np.nan, dtype=float)
    for neuron_index in range(neuron_count):
        finite = np.concatenate([
            np.abs(group[neuron_index][np.isfinite(group[neuron_index])])
            for group in groups
        ])
        if len(finite):
            scale[neuron_index] = np.percentile(finite, robust_percentile)
    positive = scale[np.isfinite(scale) & (scale > 0)]
    if len(positive) == 0:
        raise ValueError("No positive finite trace magnitude is available for scaling")
    floor = float(np.percentile(positive, floor_percentile))
    scale = np.where(np.isfinite(scale) & (scale > 0), np.maximum(scale, floor), np.nan)

    normalized = tuple(
        np.clip(
            np.divide(
                group,
                scale[:, np.newaxis, np.newaxis],
                out=np.full_like(group, np.nan),
                where=np.isfinite(scale[:, np.newaxis, np.newaxis]),
            ),
            -clip_limit,
            clip_limit,
        ).astype(np.float32)
        for group in groups
    )
    return {
        "traces": normalized,
        "scale": scale.astype(np.float32),
        "scale_floor": floor,
        "clip_limit": float(clip_limit),
    }


def minmax_normalize_neuron_trial_traces(
    trace_groups: Iterable[np.ndarray],
) -> dict[str, object]:
    """Min-max normalize trial traces once per neuron across all supplied groups."""

    groups = tuple(np.asarray(group, dtype=float) for group in trace_groups)
    if not groups:
        raise ValueError("At least one trace group is required")
    neuron_count = groups[0].shape[0] if groups[0].ndim == 3 else -1
    if neuron_count < 1 or any(
        group.ndim != 3 or group.shape[0] != neuron_count for group in groups
    ):
        raise ValueError("Every trace group must be neurons x trials x time")

    minimum = np.full(neuron_count, np.nan, dtype=float)
    maximum = np.full(neuron_count, np.nan, dtype=float)
    normalized = [np.full_like(group, np.nan) for group in groups]
    for neuron_index in range(neuron_count):
        finite = np.concatenate([
            group[neuron_index][np.isfinite(group[neuron_index])]
            for group in groups
        ])
        if not len(finite):
            continue
        minimum[neuron_index] = np.min(finite)
        maximum[neuron_index] = np.max(finite)
        span = maximum[neuron_index] - minimum[neuron_index]
        for output, group in zip(normalized, groups, strict=True):
            valid = np.isfinite(group[neuron_index])
            output[neuron_index, valid] = (
                (group[neuron_index, valid] - minimum[neuron_index]) / span
                if span > 0 else 0.0
            )

    return {
        "traces": tuple(group.astype(np.float32) for group in normalized),
        "minimum": minimum.astype(np.float32),
        "maximum": maximum.astype(np.float32),
    }


def compute_peak_time_row_order(
    mean_traces: np.ndarray,
    relative_time_s: np.ndarray,
    *,
    response_window_s: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Order neurons by response peak time with amplitude as a stable tie-breaker."""

    traces = np.asarray(mean_traces, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    if traces.ndim != 2 or traces.shape[1] != len(times):
        raise ValueError("mean_traces must be neurons x relative time")
    start_s, stop_s = map(float, response_window_s)
    if stop_s < start_s:
        raise ValueError("response_window_s must be increasing")
    response_mask = (times >= start_s) & (times <= stop_s)
    if not response_mask.any():
        raise ValueError("response_window_s contains no aligned samples")

    response = traces[:, response_mask]
    finite = np.isfinite(response).any(axis=1)
    safe = np.where(np.isfinite(response), response, -np.inf)
    peak_index = np.argmax(safe, axis=1)
    response_times = times[response_mask]
    peak_time_s = np.full(len(traces), np.nan, dtype=float)
    peak_amplitude = np.full(len(traces), np.nan, dtype=float)
    rows = np.flatnonzero(finite)
    peak_time_s[rows] = response_times[peak_index[rows]]
    peak_amplitude[rows] = response[rows, peak_index[rows]]

    sort_time = np.where(finite, peak_time_s, np.inf)
    sort_amplitude = np.where(finite, peak_amplitude, -np.inf)
    row_order = np.lexsort((np.arange(len(traces)), -sort_amplitude, sort_time))
    return {
        "row_order": row_order.astype(np.int64),
        "peak_time_s": peak_time_s.astype(np.float32),
        "peak_amplitude": peak_amplitude.astype(np.float32),
    }


def select_balanced_test_p5_events(
    events: pd.DataFrame,
    *,
    symbols: Iterable[str] = ("A", "B", "C"),
    trial_count: int | None = None,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Balance test P5 conditions, prioritizing B trials preceding catch sequences."""

    required = {
        "phase", "item_position", "symbol", "measured_onset_s", "sequence_index",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events is missing columns: {missing}")
    requested = tuple(str(symbol) for symbol in symbols)
    selected = events.copy()
    if "source_event_index" not in selected.columns:
        selected["source_event_index"] = selected.index.to_numpy(dtype=np.int64)
    selected = selected.loc[
        selected["phase"].eq("test")
        & selected["item_position"].eq(5)
        & selected["symbol"].isin(requested)
    ].copy()
    if "timing_qc" in selected.columns:
        selected = selected.loc[selected["timing_qc"].eq("ok")]
    selected = selected.sort_values("measured_onset_s")
    counts = selected.groupby("symbol").size()
    if any(symbol not in counts for symbol in requested):
        missing_symbols = [symbol for symbol in requested if symbol not in counts]
        raise ValueError(f"Missing test P5 conditions: {missing_symbols}")

    preceding_reference_indices: list[int] = []
    p5_rows = selected.reset_index(drop=True)
    for target_index in p5_rows.index[p5_rows["symbol"].ne("B")]:
        if target_index == 0:
            continue
        previous = p5_rows.iloc[target_index - 1]
        if previous["symbol"] == "B":
            preceding_reference_indices.append(int(previous["source_event_index"]))
    preceding_reference_indices = list(dict.fromkeys(preceding_reference_indices))

    maximum = min(int(counts[symbol]) for symbol in requested)
    if trial_count is not None:
        if trial_count < 1:
            raise ValueError("trial_count must be positive")
        maximum = min(maximum, int(trial_count))
    rng = np.random.default_rng(random_seed)
    balanced: list[pd.DataFrame] = []
    for symbol in requested:
        pool = selected.loc[selected["symbol"].eq(symbol)].copy()
        source = "all_available"
        if symbol == "B" and len(preceding_reference_indices) >= maximum:
            pool = pool.loc[pool["source_event_index"].isin(preceding_reference_indices)]
            source = "immediately_preceding_catch"
        chosen = np.sort(rng.choice(len(pool), size=maximum, replace=False))
        part = pool.iloc[chosen].copy()
        part["balance_source"] = source
        balanced.append(part)
    result = pd.concat(balanced, ignore_index=True)
    result = result.sort_values("measured_onset_s").reset_index(drop=True)
    result["balanced_trial_index"] = result.groupby("symbol").cumcount() + 1
    return result


def select_paired_catch_reference_events(
    events: pd.DataFrame,
    *,
    catch_symbols: Iterable[str] = ("A", "C"),
    reference_symbol: str = "B",
    require_all_catches: bool = True,
) -> pd.DataFrame:
    """Pair each test P5 catch with its nearest preceding P5 reference trial.

    The returned long table contains two rows per matched pair. Reference rows are
    labeled ``B_A`` or ``B_C`` so the two comparison-specific B sets remain
    auditable even when the same source event appears in both comparisons. Within
    one catch condition, reference events are used at most once.
    """

    required = {
        "phase", "item_position", "symbol", "measured_onset_s", "sequence_index",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"events is missing columns: {missing}")
    catches = tuple(str(symbol) for symbol in catch_symbols)
    if not catches or reference_symbol in catches:
        raise ValueError("catch_symbols must be non-empty and exclude reference_symbol")

    selected = events.copy()
    if "source_event_index" not in selected.columns:
        selected["source_event_index"] = selected.index.to_numpy(dtype=np.int64)
    selected = selected.loc[
        selected["phase"].eq("test")
        & selected["item_position"].eq(5)
        & selected["symbol"].isin((*catches, reference_symbol))
    ].copy()
    if "timing_qc" in selected.columns:
        selected = selected.loc[selected["timing_qc"].eq("ok")]
    selected = selected.sort_values("measured_onset_s").reset_index(drop=True)

    records: list[dict[str, object]] = []
    unmatched: dict[str, list[int]] = {symbol: [] for symbol in catches}
    pair_counts = {symbol: 0 for symbol in catches}
    used_references: dict[str, set[int]] = {symbol: set() for symbol in catches}
    for row_position, catch_row in selected.iterrows():
        catch_symbol = str(catch_row["symbol"])
        if catch_symbol not in catches:
            continue
        preceding = selected.iloc[:row_position]
        preceding = preceding.loc[preceding["symbol"].astype(str).eq(reference_symbol)]
        preceding = preceding.loc[
            ~preceding["source_event_index"].astype(int).isin(used_references[catch_symbol])
        ]
        if preceding.empty:
            unmatched[catch_symbol].append(int(catch_row["sequence_index"]))
            continue
        reference_row = preceding.iloc[-1]
        used_references[catch_symbol].add(int(reference_row["source_event_index"]))
        pair_counts[catch_symbol] += 1
        pair_index = pair_counts[catch_symbol]
        comparison = f"{catch_symbol}_vs_{reference_symbol}_{catch_symbol}"
        pair_gap_s = float(catch_row["measured_onset_s"] - reference_row["measured_onset_s"])
        common = {
            "comparison": comparison,
            "catch_symbol": catch_symbol,
            "pair_index": pair_index,
            "pair_gap_s": pair_gap_s,
            "catch_sequence_index": int(catch_row["sequence_index"]),
            "reference_sequence_index": int(reference_row["sequence_index"]),
            "catch_source_event_index": int(catch_row["source_event_index"]),
            "reference_source_event_index": int(reference_row["source_event_index"]),
        }
        for row, role, label in (
            (reference_row, "reference", f"{reference_symbol}_{catch_symbol}"),
            (catch_row, "catch", catch_symbol),
        ):
            record = row.to_dict()
            record.update(common)
            record["pair_role"] = role
            record["condition_label"] = label
            records.append(record)

    unmatched = {symbol: indices for symbol, indices in unmatched.items() if indices}
    if require_all_catches and unmatched:
        raise ValueError(
            "Some catch trials have no unused preceding reference trial: "
            f"{unmatched}"
        )
    if not records:
        raise ValueError("No catch/reference P5 pairs were found")
    return pd.DataFrame.from_records(records).sort_values(
        ["catch_symbol", "pair_index", "pair_role"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def compute_paired_p5_neuron_effects(
    response_scores: np.ndarray,
    metadata: pd.DataFrame,
    paired_events: pd.DataFrame,
    *,
    cell_indices: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Return one auditable row per neuron for paired P5 catch-minus-B effects."""

    scores = np.asarray(response_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(metadata):
        raise ValueError("response_scores must be cells x trials and match metadata")
    if "source_event_index" not in metadata or metadata["source_event_index"].duplicated().any():
        raise ValueError("metadata must contain unique source_event_index values")
    required = {"catch_symbol", "pair_index", "pair_role", "condition_label", "source_event_index"}
    missing = sorted(required - set(paired_events.columns))
    if missing:
        raise ValueError(f"paired_events is missing columns: {missing}")

    cells = (
        np.arange(scores.shape[0], dtype=np.int64)
        if cell_indices is None
        else np.asarray(list(cell_indices), dtype=np.int64)
    )
    if cells.shape != (scores.shape[0],):
        raise ValueError("cell_indices must contain one value per response row")
    column_by_source = pd.Series(
        np.arange(len(metadata), dtype=np.int64),
        index=metadata["source_event_index"].astype(np.int64),
    )
    result = pd.DataFrame({"cell_index": cells})
    for catch_symbol in paired_events["catch_symbol"].drop_duplicates().astype(str):
        group = paired_events.loc[paired_events["catch_symbol"].astype(str).eq(catch_symbol)]
        catch = group.loc[group["pair_role"].eq("catch")].sort_values("pair_index")
        reference = group.loc[group["pair_role"].eq("reference")].sort_values("pair_index")
        if not np.array_equal(catch["pair_index"].to_numpy(), reference["pair_index"].to_numpy()):
            raise ValueError(f"Pair indices do not match for {catch_symbol}")
        columns = []
        for frame in (catch, reference):
            mapped = frame["source_event_index"].astype(np.int64).map(column_by_source)
            if mapped.isna().any():
                raise ValueError("A paired source event is absent from response metadata")
            columns.append(mapped.to_numpy(dtype=np.int64))
        catch_scores = scores[:, columns[0]]
        reference_scores = scores[:, columns[1]]
        reference_label = str(reference["condition_label"].iloc[0])
        result[f"{catch_symbol}_mean_response"] = np.nanmean(catch_scores, axis=1)
        result[f"{reference_label}_mean_response"] = np.nanmean(reference_scores, axis=1)
        result[f"{catch_symbol}_minus_{reference_label}"] = np.nanmean(
            catch_scores - reference_scores, axis=1
        )
        result[f"{catch_symbol}_valid_trial_count"] = np.sum(
            np.isfinite(catch_scores), axis=1
        )
        result[f"{reference_label}_valid_trial_count"] = np.sum(
            np.isfinite(reference_scores), axis=1
        )
    return result


def select_representative_p5_neurons(
    effects: pd.DataFrame,
    *,
    per_group: int = 2,
    minimum_valid_fraction: float = 0.8,
) -> pd.DataFrame:
    """Select deterministic quantile examples without using absolute extrema."""

    if per_group < 1:
        raise ValueError("per_group must be positive")
    if not 0 <= minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be in [0, 1]")
    required = {
        "cell_index", "A_minus_B_A", "C_minus_B_C",
        "A_valid_trial_count", "B_A_valid_trial_count",
        "C_valid_trial_count", "B_C_valid_trial_count",
    }
    missing = sorted(required - set(effects.columns))
    if missing:
        raise ValueError(f"effects is missing columns: {missing}")

    table = effects.copy()
    count_columns = [column for column in required if column.endswith("valid_trial_count")]
    maxima = table[count_columns].max(axis=0).replace(0, np.nan)
    table["minimum_valid_fraction"] = table[count_columns].div(maxima).min(axis=1)
    x = table["A_minus_B_A"].to_numpy(dtype=float)
    y = table["C_minus_B_C"].to_numpy(dtype=float)
    eligible = np.isfinite(x) & np.isfinite(y) & (
        table["minimum_valid_fraction"].to_numpy(dtype=float) >= minimum_valid_fraction
    )
    if eligible.sum() < 3 * per_group:
        raise ValueError("Too few reliable neurons for representative selection")
    table = table.loc[eligible].copy()
    table["b_dominance_score"] = -0.5 * (
        table["A_minus_B_A"] + table["C_minus_B_C"]
    )
    table["stable_distance"] = np.hypot(
        table["A_minus_B_A"], table["C_minus_B_C"]
    )

    selected_parts: list[pd.DataFrame] = []
    used: set[int] = set()

    def choose(
        candidates: pd.DataFrame,
        score_column: str,
        quantiles: np.ndarray,
        group: str,
    ) -> None:
        pool = candidates.loc[~candidates["cell_index"].astype(int).isin(used)].copy()
        if len(pool) < per_group:
            pool = table.loc[~table["cell_index"].astype(int).isin(used)].copy()
        scores = pool[score_column].to_numpy(dtype=float)
        picks = []
        for quantile in quantiles:
            target = float(np.nanquantile(scores, quantile))
            available = pool.loc[~pool["cell_index"].astype(int).isin(used)]
            ranked = available.assign(
                _distance=(available[score_column] - target).abs()
            ).sort_values(["_distance", "cell_index"], kind="stable")
            chosen = ranked.iloc[[0]].drop(columns="_distance")
            cell = int(chosen["cell_index"].iloc[0])
            used.add(cell)
            chosen["representative_group"] = group
            chosen["selection_score"] = float(chosen[score_column].iloc[0])
            chosen["selection_target_quantile"] = float(quantile)
            picks.append(chosen)
        selected_parts.append(pd.concat(picks, ignore_index=True))

    quantiles = np.linspace(0.75, 0.9, per_group)
    choose(table.loc[table["C_minus_B_C"] > 0], "C_minus_B_C", quantiles, "C-enhanced")
    choose(
        table.loc[(table["A_minus_B_A"] < 0) & (table["C_minus_B_C"] < 0)],
        "b_dominance_score", quantiles, "B-dominant",
    )
    choose(table, "stable_distance", np.linspace(0.05, 0.2, per_group), "stable")
    return pd.concat(selected_parts, ignore_index=True)


def repeated_p5_reference_subsampling(
    response_scores: np.ndarray,
    metadata: pd.DataFrame,
    *,
    cell_indices: Iterable[int] | None = None,
    catch_symbols: Iterable[str] = ("A", "C"),
    reference_symbol: str = "B",
    repeats: int = 500,
    random_seed: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Compare each catch with repeated equal-size samples from the P5 B pool.

    The 2.5th and 97.5th percentiles describe the distribution induced by B-trial
    subsampling. They are sensitivity intervals, not biological confidence
    intervals across animals.
    """

    scores = np.asarray(response_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(metadata):
        raise ValueError("response_scores must be cells x trials and match metadata")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    required = {"symbol", "source_event_index"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"metadata is missing columns: {missing}")
    if metadata["source_event_index"].duplicated().any():
        raise ValueError("metadata source_event_index values must be unique")
    catches = tuple(str(symbol) for symbol in catch_symbols)
    if not catches or reference_symbol in catches:
        raise ValueError("catch_symbols must be non-empty and exclude reference_symbol")

    if cell_indices is None:
        cells = np.arange(scores.shape[0], dtype=np.int64)
    else:
        cells = np.asarray(list(cell_indices), dtype=np.int64)
        if cells.shape != (scores.shape[0],):
            raise ValueError("cell_indices must contain one value per response row")

    symbols = metadata["symbol"].astype(str).to_numpy()
    reference_columns = np.flatnonzero(symbols == reference_symbol)
    if len(reference_columns) == 0:
        raise ValueError(f"No {reference_symbol} reference trials are available")

    seed_sequences = np.random.SeedSequence(random_seed).spawn(len(catches))
    repeat_tables: list[pd.DataFrame] = []
    neuron_tables: list[pd.DataFrame] = []
    inclusion_tables: list[pd.DataFrame] = []
    draw_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for catch_symbol, seed_sequence in zip(catches, seed_sequences, strict=True):
        catch_columns = np.flatnonzero(symbols == catch_symbol)
        sample_size = len(catch_columns)
        if sample_size == 0:
            raise ValueError(f"No {catch_symbol} catch trials are available")
        if sample_size > len(reference_columns):
            raise ValueError(
                f"Cannot draw {sample_size} {reference_symbol} trials from a pool of "
                f"{len(reference_columns)}"
            )

        comparison = f"{catch_symbol}_vs_resampled_{reference_symbol}"
        rng = np.random.default_rng(seed_sequence)
        catch_mean = np.nanmean(scores[:, catch_columns], axis=1)
        effects = np.full((scores.shape[0], repeats), np.nan, dtype=np.float32)
        inclusion_count = np.zeros(len(reference_columns), dtype=np.int64)
        population_mean = np.full(repeats, np.nan, dtype=float)
        population_median = np.full(repeats, np.nan, dtype=float)
        positive_fraction = np.full(repeats, np.nan, dtype=float)
        draw_rows: list[dict[str, object]] = []

        for repeat_index in range(repeats):
            sampled_pool_positions = np.sort(
                rng.choice(len(reference_columns), size=sample_size, replace=False)
            )
            sampled_columns = reference_columns[sampled_pool_positions]
            inclusion_count[sampled_pool_positions] += 1
            reference_mean = np.nanmean(scores[:, sampled_columns], axis=1)
            effect = catch_mean - reference_mean
            effects[:, repeat_index] = effect.astype(np.float32)
            finite = effect[np.isfinite(effect)]
            if len(finite):
                population_mean[repeat_index] = float(np.mean(finite))
                population_median[repeat_index] = float(np.median(finite))
                positive_fraction[repeat_index] = float(np.mean(finite > 0))
            for column in sampled_columns:
                draw_rows.append({
                    "comparison": comparison,
                    "repeat": repeat_index + 1,
                    "source_event_index": int(metadata.iloc[column]["source_event_index"]),
                })

        repeat_table = pd.DataFrame({
            "comparison": comparison,
            "repeat": np.arange(1, repeats + 1, dtype=np.int64),
            "catch_trial_count": sample_size,
            "reference_pool_count": len(reference_columns),
            "reference_sample_count": sample_size,
            "population_mean_effect": population_mean,
            "population_median_effect": population_median,
            "positive_neuron_fraction": positive_fraction,
        })
        repeat_tables.append(repeat_table)
        draw_tables.append(pd.DataFrame.from_records(draw_rows))

        valid_effect_counts = np.sum(np.isfinite(effects), axis=1)
        positive_counts = np.sum(np.isfinite(effects) & (effects > 0), axis=1)
        neuron_tables.append(pd.DataFrame({
            "comparison": comparison,
            "cell_index": cells,
            "effect_mean": np.nanmean(effects, axis=1),
            "effect_resampling_95_low": np.nanpercentile(effects, 2.5, axis=1),
            "effect_resampling_95_high": np.nanpercentile(effects, 97.5, axis=1),
            "probability_effect_positive": np.divide(
                positive_counts,
                valid_effect_counts,
                out=np.full(scores.shape[0], np.nan, dtype=float),
                where=valid_effect_counts > 0,
            ),
        }))

        reference_metadata = metadata.iloc[reference_columns].reset_index(drop=True)
        inclusion = pd.DataFrame({
            "comparison": comparison,
            "source_event_index": reference_metadata["source_event_index"].astype(int),
            "inclusion_count": inclusion_count,
            "inclusion_probability": inclusion_count / repeats,
            "expected_inclusion_probability": sample_size / len(reference_columns),
        })
        if "sequence_index" in reference_metadata.columns:
            inclusion["sequence_index"] = reference_metadata["sequence_index"].to_numpy()
        inclusion_tables.append(inclusion)

        finite_population = population_mean[np.isfinite(population_mean)]
        summary_rows.append({
            "comparison": comparison,
            "catch_trial_count": sample_size,
            "reference_pool_count": len(reference_columns),
            "reference_sample_count": sample_size,
            "repeats": repeats,
            "population_mean_effect": float(np.mean(finite_population)),
            "population_effect_resampling_95_low": float(np.percentile(finite_population, 2.5)),
            "population_effect_resampling_95_high": float(np.percentile(finite_population, 97.5)),
            "mean_positive_neuron_fraction": float(np.nanmean(positive_fraction)),
        })

    return {
        "summary": pd.DataFrame.from_records(summary_rows),
        "repeat_summary": pd.concat(repeat_tables, ignore_index=True),
        "neuron_summary": pd.concat(neuron_tables, ignore_index=True),
        "reference_inclusion": pd.concat(inclusion_tables, ignore_index=True),
        "sampled_reference_trials": pd.concat(draw_tables, ignore_index=True),
    }


def adjust_pvalues_holm(p_values: Iterable[float]) -> np.ndarray:
    """Holm-adjust a one-dimensional collection of p-values."""

    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(values[index]))
        adjusted[index] = min(running, 1.0)
    return adjusted


def _paired_response_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    alternative: str,
) -> dict[str, float | int]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    left_values = left_values[finite]
    right_values = right_values[finite]
    if len(left_values) < 2:
        return {
            "n_neurons": len(left_values), "left_mean": np.nan, "right_mean": np.nan,
            "mean_difference": np.nan, "median_difference": np.nan,
            "cohen_dz": np.nan, "statistic": np.nan, "p_value": np.nan,
        }
    difference = left_values - right_values
    if np.allclose(difference, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        test = stats.wilcoxon(
            left_values, right_values, alternative=alternative, zero_method="wilcox"
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)
    difference_sd = float(np.std(difference, ddof=1))
    cohen_dz = float(np.mean(difference) / difference_sd) if difference_sd > 0 else np.nan
    return {
        "n_neurons": len(left_values),
        "left_mean": float(np.mean(left_values)),
        "right_mean": float(np.mean(right_values)),
        "mean_difference": float(np.mean(difference)),
        "median_difference": float(np.median(difference)),
        "cohen_dz": cohen_dz,
        "statistic": statistic,
        "p_value": p_value,
    }


def compare_neuron_response_means(
    left_trials: np.ndarray,
    right_trials: np.ndarray,
    *,
    comparison: str,
    alternative: str = "two-sided",
) -> dict[str, object]:
    """Compare condition means across a fixed set of neurons."""

    left = np.asarray(left_trials, dtype=float)
    right = np.asarray(right_trials, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("left_trials and right_trials must share a neurons axis")
    left_mean = np.nanmean(left, axis=1)
    right_mean = np.nanmean(right, axis=1)
    result: dict[str, object] = {
        "comparison": comparison,
        "alternative": alternative,
        "left_trials": left.shape[1],
        "right_trials": right.shape[1],
    }
    result.update(_paired_response_test(left_mean, right_mean, alternative=alternative))
    return result


def paired_equivalence_test(
    left_trials: np.ndarray,
    right_trials: np.ndarray,
    *,
    margin: float,
) -> dict[str, float | int | bool]:
    """Run a paired two-one-sided equivalence test on neuron mean responses."""

    if margin <= 0:
        raise ValueError("margin must be positive")
    left = np.nanmean(np.asarray(left_trials, dtype=float), axis=1)
    right = np.nanmean(np.asarray(right_trials, dtype=float), axis=1)
    difference = left - right
    difference = difference[np.isfinite(difference)]
    if len(difference) < 2 or np.std(difference, ddof=1) == 0:
        return {
            "n_neurons": len(difference), "margin": margin,
            "mean_difference": float(np.mean(difference)) if len(difference) else np.nan,
            "p_lower": np.nan, "p_upper": np.nan, "p_tost": np.nan,
            "equivalent_at_0_05": False,
        }
    lower = stats.ttest_1samp(difference, -margin, alternative="greater")
    upper = stats.ttest_1samp(difference, margin, alternative="less")
    p_tost = max(float(lower.pvalue), float(upper.pvalue))
    return {
        "n_neurons": len(difference),
        "margin": float(margin),
        "mean_difference": float(np.mean(difference)),
        "p_lower": float(lower.pvalue),
        "p_upper": float(upper.pvalue),
        "p_tost": p_tost,
        "equivalent_at_0_05": bool(p_tost < 0.05),
    }


def analyze_train_sequence_adaptation(
    statistics: dict[str, np.ndarray],
    windows: pd.DataFrame,
    *,
    scale: np.ndarray | None = None,
    early_trial_count: int = 7,
    late_trial_count: int = 7,
    bin_size: int = 7,
) -> dict[str, object]:
    """Summarize first/last sequence responses and the P5 trajectory."""

    if early_trial_count < 1 or late_trial_count < 1 or bin_size < 1:
        raise ValueError("early_trial_count, late_trial_count, and bin_size must be positive")
    metadata = windows.loc[np.asarray(statistics["window_index"], dtype=np.int64)].reset_index()
    delta = response_delta_from_statistics(statistics, scale=scale)
    if len(metadata) != delta.shape[1]:
        raise ValueError("windows and statistics trial axes do not match")
    required = {"sequence_index", "item_position", "symbol"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"windows is missing columns: {sorted(required - set(metadata.columns))}")
    sequences = np.sort(metadata["sequence_index"].unique())
    if len(sequences) < early_trial_count + late_trial_count:
        raise ValueError("Not enough training sequences for disjoint early and late groups")
    early_sequences = sequences[:early_trial_count]
    late_sequences = sequences[-late_trial_count:]
    cell_indices = np.asarray(
        statistics.get("cell_index", np.arange(delta.shape[0])), dtype=np.int64
    )

    population_rows: list[dict[str, object]] = []
    neuron_rows: list[pd.DataFrame] = []
    test_rows: list[dict[str, object]] = []
    for position in sorted(metadata["item_position"].unique()):
        position_mask = metadata["item_position"].eq(position).to_numpy()
        symbol = str(metadata.loc[position_mask, "symbol"].mode().iloc[0])
        epoch_values: dict[str, np.ndarray] = {}
        for epoch, epoch_sequences in (("first", early_sequences), ("last", late_sequences)):
            mask = position_mask & metadata["sequence_index"].isin(epoch_sequences).to_numpy()
            values = np.nanmean(delta[:, mask], axis=1)
            epoch_values[epoch] = values
            mean, sem = _mean_and_sem(values[:, np.newaxis], axis=0)
            median, q25, q75 = _median_and_iqr(values[:, np.newaxis], axis=0)
            population_rows.append({
                "epoch": epoch,
                "item_position": int(position),
                "symbol": symbol,
                "sequence_count": int(mask.sum()),
                "neuron_count": int(np.isfinite(values).sum()),
                "mean_response": float(mean[0]),
                "sem_response": float(sem[0]),
                "median_response": float(median[0]),
                "q25_response": float(q25[0]),
                "q75_response": float(q75[0]),
            })
        neuron_rows.append(pd.DataFrame({
            "cell_index": cell_indices,
            "item_position": int(position),
            "symbol": symbol,
            "first_mean": epoch_values["first"],
            "last_mean": epoch_values["last"],
            "first_minus_last": epoch_values["first"] - epoch_values["last"],
        }))
        test_row: dict[str, object] = {"item_position": int(position), "symbol": symbol}
        test_row.update(
            _paired_response_test(
                epoch_values["first"], epoch_values["last"], alternative="two-sided"
            )
        )
        test_rows.append(test_row)

    position_tests = pd.DataFrame(test_rows)
    position_tests["p_holm_5_positions"] = adjust_pvalues_holm(position_tests["p_value"])
    p5_row = position_tests.loc[position_tests["item_position"].eq(5)].iloc[0]
    p5_neurons = pd.concat(neuron_rows, ignore_index=True)
    p5_neurons = p5_neurons.loc[p5_neurons["item_position"].eq(5)]
    p5_directional = _paired_response_test(
        p5_neurons["first_mean"].to_numpy(),
        p5_neurons["last_mean"].to_numpy(),
        alternative="greater",
    )

    p5_columns = np.flatnonzero(metadata["item_position"].eq(5).to_numpy())
    p5_columns = p5_columns[np.argsort(metadata.iloc[p5_columns]["sequence_index"].to_numpy())]
    trial_rows: list[dict[str, object]] = []
    for ordinal, column in enumerate(p5_columns, start=1):
        mean, sem = _mean_and_sem(delta[:, column][:, np.newaxis], axis=0)
        median, q25, q75 = _median_and_iqr(delta[:, column][:, np.newaxis], axis=0)
        trial_rows.append({
            "sequence_ordinal": ordinal,
            "sequence_index": int(metadata.iloc[column]["sequence_index"]),
            "mean_response": float(mean[0]),
            "sem_response": float(sem[0]),
            "median_response": float(median[0]),
            "q25_response": float(q25[0]),
            "q75_response": float(q75[0]),
        })
    trajectory = pd.DataFrame(trial_rows)
    bin_rows: list[dict[str, object]] = []
    for bin_index, start in enumerate(range(0, len(p5_columns), bin_size), start=1):
        columns = p5_columns[start:start + bin_size]
        neuron_means = np.nanmean(delta[:, columns], axis=1)
        mean, sem = _mean_and_sem(neuron_means[:, np.newaxis], axis=0)
        median, q25, q75 = _median_and_iqr(neuron_means[:, np.newaxis], axis=0)
        bin_rows.append({
            "bin_index": bin_index,
            "first_sequence_ordinal": start + 1,
            "last_sequence_ordinal": start + len(columns),
            "sequence_count": len(columns),
            "mean_response": float(mean[0]),
            "sem_response": float(sem[0]),
            "median_response": float(median[0]),
            "q25_response": float(q25[0]),
            "q75_response": float(q75[0]),
        })
    binned_trajectory = pd.DataFrame(bin_rows)
    finite = np.isfinite(trajectory["median_response"])
    correlation = stats.spearmanr(
        trajectory.loc[finite, "sequence_ordinal"],
        trajectory.loc[finite, "median_response"],
        alternative="less",
    )
    return {
        "early_sequences": early_sequences,
        "late_sequences": late_sequences,
        "population_by_position": pd.DataFrame(population_rows),
        "neuron_by_position": pd.concat(neuron_rows, ignore_index=True),
        "position_tests": position_tests,
        "p5_directional_test": pd.DataFrame([{
            "comparison": f"first {early_trial_count} > last {late_trial_count}",
            **p5_directional,
        }]),
        "p5_trajectory": trajectory,
        "p5_binned_trajectory": binned_trajectory,
        "p5_spearman_test": pd.DataFrame([{
            "comparison": "Median P5 response decreases across train",
            "rho": float(correlation.statistic),
            "p_value_one_sided": float(correlation.pvalue),
            "sequence_count": int(finite.sum()),
        }]),
        "p5_two_sided_position_row": p5_row.to_dict(),
    }


def analyze_balanced_p5_responses(
    statistics: dict[str, np.ndarray],
    windows: pd.DataFrame,
    *,
    scale: np.ndarray | None = None,
    condition_order: Iterable[str] = ("A", "B", "C"),
) -> dict[str, object]:
    """Compare balanced P5 conditions using one fixed neuron population."""

    order = tuple(str(value) for value in condition_order)
    metadata = windows.loc[np.asarray(statistics["window_index"], dtype=np.int64)].reset_index()
    delta = response_delta_from_statistics(statistics, scale=scale)
    condition_means: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []
    for condition in order:
        mask = metadata["symbol"].eq(condition).to_numpy()
        if not mask.any():
            raise ValueError(f"No balanced trials are available for condition {condition}")
        neuron_means = np.nanmean(delta[:, mask], axis=1)
        condition_means[condition] = neuron_means
        mean, sem = _mean_and_sem(neuron_means[:, np.newaxis], axis=0)
        median, q25, q75 = _median_and_iqr(neuron_means[:, np.newaxis], axis=0)
        summary_rows.append({
            "condition": condition,
            "trial_count": int(mask.sum()),
            "neuron_count": int(np.isfinite(neuron_means).sum()),
            "mean_response": float(mean[0]),
            "sem_response": float(sem[0]),
            "median_response": float(median[0]),
            "q25_response": float(q25[0]),
            "q75_response": float(q75[0]),
            "mean_ci95_low": float(mean[0] - 1.96 * sem[0]),
            "mean_ci95_high": float(mean[0] + 1.96 * sem[0]),
        })
    complete = np.column_stack([condition_means[key] for key in order])
    complete = complete[np.isfinite(complete).all(axis=1)]
    if len(complete) >= 2:
        omnibus = stats.friedmanchisquare(*[complete[:, i] for i in range(len(order))])
        omnibus_table = pd.DataFrame([{
            "test": "Friedman repeated-neuron test",
            "n_neurons": len(complete),
            "statistic": float(omnibus.statistic),
            "p_value": float(omnibus.pvalue),
        }])
    else:
        omnibus_table = pd.DataFrame([{
            "test": "Friedman repeated-neuron test", "n_neurons": len(complete),
            "statistic": np.nan, "p_value": np.nan,
        }])
    pair_rows: list[dict[str, object]] = []
    for first_index, first in enumerate(order):
        for second in order[first_index + 1:]:
            row: dict[str, object] = {
                "comparison": f"{first} vs {second}", "alternative": "two-sided",
            }
            row.update(
                _paired_response_test(
                    condition_means[first], condition_means[second], alternative="two-sided"
                )
            )
            pair_rows.append(row)
    pairwise = pd.DataFrame(pair_rows)
    pairwise["p_holm_3_pairs"] = adjust_pvalues_holm(pairwise["p_value"])
    neuron_table = pd.DataFrame({
        "cell_index": np.asarray(
            statistics.get("cell_index", np.arange(delta.shape[0])), dtype=np.int64
        ),
        **{condition: condition_means[condition] for condition in order},
    })
    return {
        "condition_summary": pd.DataFrame(summary_rows),
        "neuron_condition_response": neuron_table,
        "omnibus_test": omnibus_table,
        "pairwise_tests": pairwise,
    }


def plot_train_adaptation_summary(
    analysis: dict[str, object],
    *,
    title: str,
) -> Figure:
    """Plot a Figure-S1-inspired train summary without a random control."""

    trajectory = analysis["p5_trajectory"]
    binned = analysis["p5_binned_trajectory"]
    positions = analysis["population_by_position"]
    neurons = analysis["neuron_by_position"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axis = axes[0]
    x = trajectory["sequence_ordinal"].to_numpy(dtype=float)
    y = trajectory["median_response"].to_numpy(dtype=float)
    lower = trajectory["q25_response"].to_numpy(dtype=float)
    upper = trajectory["q75_response"].to_numpy(dtype=float)
    axis.plot(x, y, color="#A6A6A6", linewidth=0.9, alpha=0.8, label="single sequence")
    axis.fill_between(x, lower, upper, color="#D8D8D8", alpha=0.35, linewidth=0)
    bx = (
        binned["first_sequence_ordinal"].to_numpy(dtype=float)
        + binned["last_sequence_ordinal"].to_numpy(dtype=float)
    ) / 2
    by = binned["median_response"].to_numpy(dtype=float)
    blow = binned["q25_response"].to_numpy(dtype=float)
    bhigh = binned["q75_response"].to_numpy(dtype=float)
    axis.plot(bx, by, color="#111111", marker="o", markersize=4, linewidth=1.8, label="binned")
    axis.fill_between(bx, blow, bhigh, color="#111111", alpha=0.14, linewidth=0)
    early_count = len(analysis["early_sequences"])
    late_count = len(analysis["late_sequences"])
    axis.axvspan(0.5, early_count + 0.5, color="#D55E00", alpha=0.08)
    axis.axvspan(len(x) - late_count + 0.5, len(x) + 0.5, color="#0072B2", alpha=0.08)
    axis.set(xlabel="AAAAB sequence ordinal", ylabel="Median standardized P5-B response", title="P5-B across train")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    colors = {"first": "#D55E00", "last": "#0072B2"}
    for epoch, offset in (("first", -0.08), ("last", 0.08)):
        table = positions.loc[positions["epoch"].eq(epoch)].sort_values("item_position")
        pos = table["item_position"].to_numpy(dtype=float) + offset
        median = table["median_response"].to_numpy(dtype=float)
        errors = np.vstack([
            median - table["q25_response"].to_numpy(dtype=float),
            table["q75_response"].to_numpy(dtype=float) - median,
        ])
        axis.errorbar(
            pos, median, yerr=errors,
            color=colors[epoch], marker="o", linewidth=1.5, capsize=2,
            label=f"{epoch} {early_count if epoch == 'first' else late_count}",
        )
    axis.axhline(0, color="#777777", linewidth=0.7)
    axis.set(xticks=np.arange(1, 6), xlabel="Sequence position", ylabel="Median standardized response", title="First vs last sequences")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[2]
    p5 = neurons.loc[neurons["item_position"].eq(5), ["first_mean", "last_mean"]].dropna()
    differences = p5["first_mean"] - p5["last_mean"]
    axis.boxplot(
        differences.to_numpy(dtype=float), orientation="vertical", widths=0.45,
        whis=(5, 95), showfliers=False,
        medianprops={"color": "#111111", "linewidth": 2},
        boxprops={"color": "#555555"}, whiskerprops={"color": "#777777"},
        capprops={"color": "#777777"},
    )
    axis.axhline(0, color="#888888", linewidth=0.8)
    axis.set(xticks=[1], xticklabels=[f"First {early_count} - last {late_count}"], ylabel="Standardized response difference", title="Matched-neuron difference")
    figure.suptitle(title, y=1.02, fontsize=13)
    figure.tight_layout()
    return figure


def plot_train_sequence_mean_sem(
    aligned_delta: np.ndarray,
    relative_time_s: np.ndarray,
    sequence_metadata: pd.DataFrame,
    stimulus_timing: pd.DataFrame,
    *,
    early_sequences: Iterable[int],
    late_sequences: Iterable[int],
    population_groups: Mapping[str, Iterable[bool]] | None = None,
    ylabel: str = "Mean +/- SEM baseline-subtracted activity (a.u.)",
    title: str = "Train full-sequence first/last responses",
) -> Figure:
    """Plot P1-anchored full-sequence mean traces for fixed populations."""

    delta = np.asarray(aligned_delta, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    if (
        delta.ndim != 3
        or delta.shape[1] != len(sequence_metadata)
        or delta.shape[2] != len(times)
    ):
        raise ValueError("aligned_delta must be neurons x sequences x relative time")
    required_metadata = {"sequence_index"}
    required_timing = {
        "item_position", "symbol", "relative_onset_s", "relative_offset_s",
    }
    if not required_metadata.issubset(sequence_metadata.columns):
        raise ValueError("sequence_metadata must contain sequence_index")
    if not required_timing.issubset(stimulus_timing.columns):
        missing = sorted(required_timing - set(stimulus_timing.columns))
        raise ValueError(f"stimulus_timing is missing columns: {missing}")

    if population_groups is None:
        groups = [("Population", np.ones(delta.shape[0], dtype=bool))]
    else:
        groups = []
        for label, values in population_groups.items():
            group_mask = np.asarray(values, dtype=bool)
            if group_mask.shape != (delta.shape[0],):
                raise ValueError(
                    f"Population group {label!r} must have one mask value per neuron"
                )
            groups.append((str(label), group_mask))
        if not groups:
            raise ValueError("population_groups must contain at least one group")

    epochs = (
        ("First", set(map(int, early_sequences)), "#D55E00"),
        ("Last", set(map(int, late_sequences)), "#0072B2"),
    )
    stimulus_colors = {"A": "#56B4E9", "B": "#D55E00", "C": "#009E73"}
    figure, axes = plt.subplots(
        len(groups), 2, figsize=(15, 3.7 * len(groups)),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, (group_label, group_mask) in enumerate(groups):
        for column, (epoch_label, sequences, trace_color) in enumerate(epochs):
            axis = axes[row, column]
            sequence_mask = sequence_metadata["sequence_index"].isin(sequences).to_numpy()
            if not sequence_mask.any():
                raise ValueError(
                    f"No aligned sequences belong to the {epoch_label} group"
                )
            for timing in stimulus_timing.itertuples(index=False):
                symbol = str(timing.symbol)
                axis.axvspan(
                    float(timing.relative_onset_s),
                    float(timing.relative_offset_s),
                    color=stimulus_colors.get(symbol, "#BDBDBD"),
                    alpha=0.12,
                    linewidth=0,
                )
                axis.text(
                    (
                        float(timing.relative_onset_s)
                        + float(timing.relative_offset_s)
                    ) / 2,
                    0.98,
                    f"P{int(timing.item_position)}-{symbol}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                )
            axis.axhline(0, color="#888888", linewidth=0.7)
            if group_mask.any():
                per_neuron = np.nanmean(
                    delta[group_mask][:, sequence_mask, :], axis=1
                )
                mean, sem = _mean_and_sem(per_neuron, axis=0)
                axis.plot(times, mean, color=trace_color, linewidth=2.0)
                axis.fill_between(
                    times, mean - sem, mean + sem,
                    color=trace_color, alpha=0.2, linewidth=0,
                )
            else:
                axis.text(
                    0.5, 0.5, "No neurons in this group",
                    transform=axis.transAxes, ha="center", va="center",
                )
            axis.set(
                xlim=(float(times[0]), float(times[-1])),
                xlabel="Time from P1 onset (s)",
                title=(
                    f"{group_label} (n={int(group_mask.sum())}) | "
                    f"{epoch_label} {int(sequence_mask.sum())}"
                ),
            )
            if column == 0:
                axis.set_ylabel(ylabel)
    figure.suptitle(title, y=1.01)
    figure.tight_layout()
    return figure


def plot_balanced_p5_traces(
    aligned_delta: np.ndarray,
    relative_time_s: np.ndarray,
    metadata: pd.DataFrame,
    *,
    stimulus_duration_s: float,
    condition_order: Iterable[str] = ("A", "B", "C"),
    display_window_s: tuple[float, float] | None = None,
) -> Figure:
    """Plot balanced P5 traces for one fixed neuron population."""

    delta = np.asarray(aligned_delta, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    if delta.ndim != 3 or delta.shape[1] != len(metadata) or delta.shape[2] != len(times):
        raise ValueError("aligned_delta must be neurons x metadata trials x time")
    order = tuple(condition_order)
    colors = {"A": "#0072B2", "B": "#444444", "C": "#D55E00"}
    figure, axes = plt.subplots(1, len(order), figsize=(4.4 * len(order), 3.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, condition in zip(axes, order, strict=True):
        mask = metadata["symbol"].eq(condition).to_numpy()
        per_neuron = np.nanmean(delta[:, mask, :], axis=1)
        median, q25, q75 = _median_and_iqr(per_neuron, axis=0)
        color = colors.get(condition, "#333333")
        axis.axvspan(0, stimulus_duration_s, color="#D9D9D9", alpha=0.45, linewidth=0)
        axis.axhline(0, color="#888888", linewidth=0.7)
        axis.plot(times, median, color=color, linewidth=2)
        axis.fill_between(times, q25, q75, color=color, alpha=0.2, linewidth=0)
        axis.set(title=f"P5-{condition} | {int(mask.sum())} trials", xlabel="Time from P5 onset (s)")
        if display_window_s is not None:
            axis.set_xlim(*map(float, display_window_s))
    axes[0].set_ylabel("Median baseline-SD response")
    figure.suptitle("Balanced P5 responses: fixed neurons, no random-sequence control", y=1.03)
    figure.tight_layout()
    return figure


def _validated_row_order(row_order: Iterable[int], neuron_count: int) -> np.ndarray:
    order = np.asarray(list(row_order), dtype=np.int64)
    if order.shape != (neuron_count,) or not np.array_equal(
        np.sort(order), np.arange(neuron_count, dtype=np.int64)
    ):
        raise ValueError("row_order must be a permutation of all neuron rows")
    return order


def _robust_symmetric_limit(values: Iterable[np.ndarray], percentile: float) -> float:
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    finite = np.concatenate([
        np.abs(np.asarray(value, dtype=float)[np.isfinite(value)]) for value in values
    ])
    limit = float(np.percentile(finite, percentile)) if len(finite) else 1.0
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def plot_balanced_p5_heatmaps(
    aligned_delta: np.ndarray,
    relative_time_s: np.ndarray,
    metadata: pd.DataFrame,
    *,
    stimulus_duration_s: float,
    condition_order: Iterable[str] = ("A", "B", "C"),
    sort_condition: str = "C",
    row_order: Iterable[int] | None = None,
    normalization: str = "within_neuron_z",
    z_limit: float | None = 2.5,
    robust_percentile: float = 99.0,
    row_order_label: str | None = None,
    display_window_s: tuple[float, float] | None = None,
) -> tuple[Figure, dict[str, object]]:
    """Plot matched A/B/C heatmaps with one supplied or condition-derived row order."""

    delta = np.asarray(aligned_delta, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    order = tuple(condition_order)
    means = {
        condition: np.nanmean(
            delta[:, metadata["symbol"].eq(condition).to_numpy(), :], axis=1
        )
        for condition in order
    }
    if normalization == "within_neuron_z":
        concatenated = np.concatenate([means[condition] for condition in order], axis=1)
        row_mean = np.nanmean(concatenated, axis=1)
        row_std = np.nanstd(concatenated, axis=1)
        displayed = {
            condition: np.divide(
                means[condition] - row_mean[:, np.newaxis],
                row_std[:, np.newaxis],
                out=np.zeros_like(means[condition]),
                where=row_std[:, np.newaxis] > 0,
            )
            for condition in order
        }
        colorbar_label = "Shared within-neuron z-score"
    elif normalization == "baseline_sd":
        displayed = means
        colorbar_label = "Baseline-SD response"
    else:
        raise ValueError("normalization must be 'within_neuron_z' or 'baseline_sd'")

    response_mask = (times >= 0) & (times <= stimulus_duration_s)
    if row_order is None:
        if sort_condition not in displayed:
            raise ValueError(f"Unknown sort condition: {sort_condition}")
        sort_score = np.nanmean(displayed[sort_condition][:, response_mask], axis=1)
        resolved_order = np.argsort(
            np.nan_to_num(sort_score, nan=-np.inf), kind="stable"
        )
        resolved_label = row_order_label or f"ordered by P5-{sort_condition}"
    else:
        resolved_order = _validated_row_order(row_order, delta.shape[0])
        sort_score = np.full(delta.shape[0], np.nan, dtype=float)
        resolved_label = row_order_label or "fixed external row order"
    limit = (
        _robust_symmetric_limit(displayed.values(), robust_percentile)
        if z_limit is None else float(z_limit)
    )
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("z_limit must be positive and finite")
    figure, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 6.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    image = None
    for axis, condition in zip(axes, order, strict=True):
        image = axis.imshow(
            displayed[condition][resolved_order], aspect="auto", origin="lower",
            interpolation="nearest", extent=(times[0], times[-1], 0, len(resolved_order)),
            cmap="RdBu_r", vmin=-limit, vmax=limit,
        )
        axis.axvline(0, color="#111111", linewidth=0.7)
        axis.axvline(stimulus_duration_s, color="#111111", linewidth=0.7, linestyle="--")
        axis.set(title=f"P5-{condition}", xlabel="Time from onset (s)")
        if display_window_s is not None:
            axis.set_xlim(*map(float, display_window_s))
    axes[0].set_ylabel(f"Same neurons, {resolved_label}")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes.tolist(), pad=0.02, shrink=0.8)
        colorbar.set_label(colorbar_label)
    figure.suptitle("Matched-cell P5 heatmaps", y=0.98)
    return figure, {
        "row_order": resolved_order.astype(np.int64),
        "sort_score": sort_score.astype(np.float32),
        "color_limit": limit,
        "normalization": normalization,
        **{f"mean_{condition}": means[condition].astype(np.float32) for condition in order},
        **{f"display_{condition}": displayed[condition].astype(np.float32) for condition in order},
        **{f"z_{condition}": displayed[condition].astype(np.float32) for condition in order},
    }


def plot_p5_neuron_effect_scatter(
    effects: pd.DataFrame,
    *,
    representatives: pd.DataFrame | None = None,
    axis_percentile: float = 99.5,
) -> tuple[Figure, pd.DataFrame]:
    """Plot paired A-B_A and C-B_C effects and return descriptive quadrants."""

    required = {"cell_index", "A_minus_B_A", "C_minus_B_C"}
    missing = sorted(required - set(effects.columns))
    if missing:
        raise ValueError(f"effects is missing columns: {missing}")
    finite = effects.loc[
        np.isfinite(effects["A_minus_B_A"]) & np.isfinite(effects["C_minus_B_C"])
    ].copy()
    x = finite["A_minus_B_A"].to_numpy(dtype=float)
    y = finite["C_minus_B_C"].to_numpy(dtype=float)
    if not 0 < axis_percentile <= 100:
        raise ValueError("axis_percentile must be in (0, 100]")
    quadrant = np.select(
        [(x >= 0) & (y >= 0), (x < 0) & (y >= 0), (x < 0) & (y < 0)],
        ["A+ / C+", "A- / C+", "A- / C-"],
        default="A+ / C-",
    )
    quadrant_summary = (
        pd.Series(quadrant, name="quadrant").value_counts()
        .reindex(["A+ / C+", "A- / C+", "A- / C-", "A+ / C-"], fill_value=0)
        .rename_axis("quadrant").reset_index(name="neuron_count")
    )
    quadrant_summary["neuron_fraction"] = quadrant_summary["neuron_count"] / len(finite)

    figure, axis = plt.subplots(figsize=(6.3, 5.7))
    axis.scatter(x, y, s=8, color="#6F7782", alpha=0.35, linewidths=0)
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.axvline(0, color="#222222", linewidth=0.8)
    if representatives is not None and len(representatives):
        colors = {"C-enhanced": "#D55E00", "B-dominant": "#0072B2", "stable": "#009E73"}
        label_offsets = {
            "C-enhanced": ((4, 5), (4, 5)),
            "B-dominant": ((4, 5), (4, 5)),
            "stable": ((7, 10), (7, -14)),
        }
        group_counts = {group: 0 for group in label_offsets}
        for _, row in representatives.iterrows():
            group = str(row["representative_group"])
            color = colors.get(group, "#111111")
            axis.scatter(
                row["A_minus_B_A"], row["C_minus_B_C"], s=42,
                facecolor=color, edgecolor="white", linewidth=0.7, zorder=3,
            )
            offsets = label_offsets.get(group, ((4, 5),))
            offset = offsets[min(group_counts.get(group, 0), len(offsets) - 1)]
            group_counts[group] = group_counts.get(group, 0) + 1
            axis.annotate(
                str(int(row["cell_index"])),
                (row["A_minus_B_A"], row["C_minus_B_C"]),
                xytext=offset, textcoords="offset points", fontsize=7,
            )
    axis.set(
        xlabel="A - B_A response effect (baseline-SD units)",
        ylabel="C - B_C response effect (baseline-SD units)",
        title="Fixed-neuron paired P5 response effects",
    )
    x_limit = _robust_symmetric_limit((x,), axis_percentile)
    y_limit = _robust_symmetric_limit((y,), axis_percentile)
    outside = int(np.sum((np.abs(x) > x_limit) | (np.abs(y) > y_limit)))
    axis.set_xlim(-x_limit, x_limit)
    axis.set_ylim(-y_limit, y_limit)
    axis.text(
        0.02, 0.98,
        f"Central {axis_percentile:g}% axes; {outside} neurons outside view",
        transform=axis.transAxes, va="top", fontsize=8, color="#555555",
    )
    figure.tight_layout()
    return figure, quadrant_summary


def plot_representative_p5_traces(
    aligned_delta: np.ndarray,
    relative_time_s: np.ndarray,
    metadata: pd.DataFrame,
    cell_indices: Iterable[int],
    representatives: pd.DataFrame,
    *,
    stimulus_duration_s: float,
    display_window_s: tuple[float, float] | None = None,
) -> Figure:
    """Plot balanced A/B/C mean and SEM for deterministic representative neurons."""

    delta = np.asarray(aligned_delta, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    cells = np.asarray(list(cell_indices), dtype=np.int64)
    if delta.ndim != 3 or delta.shape[0] != len(cells) or delta.shape[2] != len(times):
        raise ValueError("aligned_delta must be cells x trials x time")
    row_by_cell = {int(cell): row for row, cell in enumerate(cells)}
    missing = set(representatives["cell_index"].astype(int)) - set(row_by_cell)
    if missing:
        raise ValueError(f"Representative cells are absent from aligned data: {sorted(missing)}")

    colors = {"A": "#0072B2", "B": "#3C3C3C", "C": "#D55E00"}
    columns = 3
    rows = int(np.ceil(len(representatives) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12.0, 3.5 * rows), squeeze=False)
    for axis, (_, representative) in zip(axes.flat, representatives.iterrows(), strict=False):
        neuron = delta[row_by_cell[int(representative["cell_index"])]]
        axis.axvspan(0, stimulus_duration_s, color="#D9D9D9", alpha=0.4, linewidth=0)
        axis.axhline(0, color="#888888", linewidth=0.6)
        for condition in ("A", "B", "C"):
            values = neuron[metadata["symbol"].eq(condition).to_numpy()]
            mean, sem = _mean_and_sem(values, axis=0)
            axis.plot(times, mean, color=colors[condition], linewidth=1.7, label=condition)
            axis.fill_between(times, mean - sem, mean + sem, color=colors[condition], alpha=0.18)
        atlas_label = representative.get("atlas_acronym", "")
        axis.set_title(
            f"{representative['representative_group']} | cell {int(representative['cell_index'])} | {atlas_label}"
        )
        axis.set_xlabel("Time from P5 onset (s)")
        if display_window_s is not None:
            axis.set_xlim(*map(float, display_window_s))
    for axis in axes[:, 0]:
        axis.set_ylabel("Baseline-SD response")
    for axis in axes.flat[len(representatives):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, ncol=3)
    figure.suptitle("Deterministically selected representative neurons", y=1.01)
    figure.tight_layout()
    return figure


def plot_fixed_population_condition_atlas(
    atlas: pd.DataFrame,
    ccf: np.ndarray,
    selections: dict[str, pd.DataFrame],
    *,
    title: str,
) -> Figure:
    """Plot independently responsive subsets using one fixed atlas denominator."""

    figure, axes = _condition_axes(len(selections), panel_size=(4.6, 4.6))
    background = np.asarray(ccf, dtype=bool)
    colors = {"A": "#0072B2", "B": "#444444", "C": "#D55E00"}
    for axis, (condition, selection) in zip(axes, selections.items(), strict=False):
        merged = atlas.merge(selection[["cell_index", "selected"]], on="cell_index", how="inner")
        selected = merged["selected"].to_numpy(dtype=bool)
        axis.imshow(
            ~background, cmap="gray", origin="lower", interpolation="nearest",
            extent=(0, background.shape[1], 0, background.shape[0]),
        )
        axis.scatter(merged["x"], merged["y"], s=3, color="#B8BDC5", alpha=0.4, linewidths=0)
        axis.scatter(
            merged.loc[selected, "x"], merged.loc[selected, "y"], s=10,
            color=colors.get(condition, "#D55E00"), alpha=0.9, linewidths=0,
        )
        axis.set(title=f"P5-{condition}: {int(selected.sum()):,}/{len(selected):,}")
        axis.set_axis_off()
    figure.suptitle(title, fontsize=12)
    return figure


def plot_selected_neuron_traces(
    mean_delta: np.ndarray,
    relative_time_s: np.ndarray,
    *,
    stimulus_duration_s: float,
    baseline_window_s: tuple[float, float],
    response_stop_s: float,
    max_individual_traces: int = 300,
) -> Figure:
    """Plot individual selected-neuron means and the population mean with SEM."""

    traces = np.asarray(mean_delta, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    if traces.ndim != 2 or traces.shape[1] != len(times):
        raise ValueError("mean_delta must be neurons x relative time")
    if len(traces) == 0 or max_individual_traces < 1:
        raise ValueError("At least one trace and a positive display limit are required")

    if len(traces) > max_individual_traces:
        display_indices = np.linspace(
            0, len(traces) - 1, max_individual_traces, dtype=np.int64
        )
    else:
        display_indices = np.arange(len(traces), dtype=np.int64)
    population_mean, population_sem = _mean_and_sem(traces, axis=0)

    figure, axis = plt.subplots(figsize=(10.5, 4.5))
    axis.axvspan(*baseline_window_s, color="#A8ADB4", alpha=0.14, linewidth=0)
    axis.axvspan(0, stimulus_duration_s, color="#D55E00", alpha=0.12, linewidth=0)
    axis.plot(
        times,
        traces[display_indices].T,
        color="#9AA0A6",
        linewidth=0.45,
        alpha=0.18,
    )
    axis.fill_between(
        times,
        population_mean - population_sem,
        population_mean + population_sem,
        color="#222222",
        alpha=0.18,
        linewidth=0,
    )
    axis.plot(times, population_mean, color="#111111", linewidth=2.0)
    axis.axvline(0, color="#D55E00", linewidth=0.9)
    axis.axvline(response_stop_s, color="#777777", linewidth=0.8, linestyle="--")
    axis.axhline(0, color="#777777", linewidth=0.6)
    axis.set(
        xlabel="Time from stimulus onset (s)",
        ylabel="Baseline-subtracted fluorescence",
        title=f"Selected neurons: individual means and population mean +/- SEM (n={len(traces):,})",
    )
    figure.tight_layout()
    return figure


def plot_selected_neuron_heatmap(
    heatmap: np.ndarray,
    relative_time_s: np.ndarray,
    *,
    stimulus_duration_s: float,
    normalization: str,
) -> Figure:
    """Plot trial-averaged selected neurons sorted by response peak time."""

    values = np.asarray(heatmap, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(times):
        raise ValueError("heatmap must be neurons x relative time")
    if normalization == "minmax":
        vmin, vmax = 0.0, 1.0
        colorbar_label = "Row min-max normalized response"
    elif normalization == "none":
        limit = float(np.nanpercentile(np.abs(values), 99))
        limit = limit if np.isfinite(limit) and limit > 0 else 1.0
        vmin, vmax = -limit, limit
        colorbar_label = "Baseline-subtracted fluorescence"
    else:
        raise ValueError("normalization must be 'minmax' or 'none'")

    figure, axis = plt.subplots(figsize=(9.5, 6.0))
    image = axis.imshow(
        values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(times[0], times[-1], 0, len(values)),
        cmap="coolwarm" if normalization == "minmax" else "RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    axis.axvline(0, color="#111111", linewidth=0.9)
    axis.axvline(stimulus_duration_s, color="#111111", linewidth=0.8, linestyle="--")
    axis.set(
        xlabel="Time from stimulus onset (s)",
        ylabel="Selected neuron (sorted by peak time)",
        title="Trial-averaged selected-neuron responses",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()
    return figure


def plot_selected_trial_heatmap(
    trial_population_delta: np.ndarray,
    relative_time_s: np.ndarray,
    *,
    stimulus_duration_s: float,
) -> Figure:
    """Plot chronological trials after averaging the selected-neuron population."""

    values = np.asarray(trial_population_delta, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(times):
        raise ValueError("trial_population_delta must be trials x relative time")
    limit = float(np.nanpercentile(np.abs(values), 99))
    limit = limit if np.isfinite(limit) and limit > 0 else 1.0

    figure, axis = plt.subplots(figsize=(9.5, 6.0))
    image = axis.imshow(
        values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(times[0], times[-1], 1, len(values) + 1),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axis.axvline(0, color="#111111", linewidth=0.9)
    axis.axvline(stimulus_duration_s, color="#111111", linewidth=0.8, linestyle="--")
    axis.set(
        xlabel="Time from stimulus onset (s)",
        ylabel="Trial (chronological)",
        title="Selected-population response across trials",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Population baseline-subtracted fluorescence")
    figure.tight_layout()
    return figure


def plot_fixed_order_trial_heatmaps(
    trial_traces: np.ndarray,
    relative_time_s: np.ndarray,
    row_order: Iterable[int],
    *,
    trial_labels: Iterable[str],
    stimulus_duration_s: float,
    columns: int = 5,
    robust_percentile: float = 99.0,
    display_window_s: tuple[float, float] | None = None,
    color_limit: float | None = None,
    color_range: tuple[float, float] | None = None,
    cmap: str = "RdBu_r",
    colorbar_label: str = "Baseline-SD response",
    figure_title: str = "First consecutive P5-B trials: fixed Trial-1 neuron order",
) -> tuple[Figure, dict[str, object]]:
    """Plot neuron-by-time trials with one row order and one shared color scale."""

    values = np.asarray(trial_traces, dtype=float)
    times = np.asarray(relative_time_s, dtype=float)
    if values.ndim != 3 or values.shape[2] != len(times):
        raise ValueError("trial_traces must be neurons x trials x relative time")
    order = _validated_row_order(row_order, values.shape[0])
    labels = tuple(str(label) for label in trial_labels)
    if len(labels) != values.shape[1]:
        raise ValueError("trial_labels must contain one label per trial")
    if columns < 1:
        raise ValueError("columns must be positive")
    if color_range is not None:
        vmin, vmax = map(float, color_range)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            raise ValueError("color_range must contain finite increasing limits")
        limit = max(abs(vmin), abs(vmax))
    elif color_limit is None:
        limit = _robust_symmetric_limit((values,), robust_percentile)
        vmin, vmax = -limit, limit
    else:
        limit = float(color_limit)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("color_limit must be positive")
        vmin, vmax = -limit, limit
    rows = int(np.ceil(values.shape[1] / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.0 * columns, 3.0 * rows),
        sharex=True, sharey=True, squeeze=False,
    )
    image = None
    for trial_index, axis in enumerate(axes.flat[:values.shape[1]]):
        image = axis.imshow(
            values[order, trial_index], aspect="auto", origin="lower",
            interpolation="nearest", extent=(times[0], times[-1], 0, len(order)),
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
        axis.axvline(0, color="#111111", linewidth=0.7)
        axis.axvline(stimulus_duration_s, color="#111111", linewidth=0.7, linestyle="--")
        axis.set_title(labels[trial_index], fontsize=9)
        if display_window_s is not None:
            axis.set_xlim(*map(float, display_window_s))
    for axis in axes.flat[values.shape[1]:]:
        axis.set_visible(False)
    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel("Time from P5 onset (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Fixed neuron order")
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), pad=0.015, shrink=0.85)
        colorbar.set_label(colorbar_label)
    figure.suptitle(figure_title, y=0.99)
    return figure, {
        "row_order": order,
        "color_limit": limit,
        "color_range": np.asarray([vmin, vmax], dtype=float),
        "ordered_traces": values[order].astype(np.float32),
    }


def plot_response_selection_summary(
    summary: pd.DataFrame,
    *,
    fraction_threshold: float,
) -> Figure:
    """Show response prevalence and effect size with selected cells emphasized."""

    required = {"responsive_trial_fraction", "mean_response_minus_baseline", "selected"}
    if not required.issubset(summary.columns):
        raise ValueError(f"summary is missing columns: {sorted(required - set(summary.columns))}")
    finite = summary["responsive_trial_fraction"].notna()
    selected = summary["selected"].astype(bool)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(
        summary.loc[finite, "responsive_trial_fraction"],
        bins=np.linspace(0, 1, 41), color="#7A8793", edgecolor="white",
    )
    axes[0].axvline(fraction_threshold, color="#D55E00", linewidth=1.5)
    axes[0].set(
        xlabel="Responsive trial fraction",
        ylabel="Neuron count",
        title="Trial-level response prevalence",
    )
    axes[1].scatter(
        summary.loc[~selected, "responsive_trial_fraction"],
        summary.loc[~selected, "mean_response_minus_baseline"],
        s=5, color="#B8BDC5", alpha=0.45, linewidths=0,
    )
    axes[1].scatter(
        summary.loc[selected, "responsive_trial_fraction"],
        summary.loc[selected, "mean_response_minus_baseline"],
        s=8, color="#D55E00", alpha=0.75, linewidths=0,
    )
    axes[1].axvline(fraction_threshold, color="#D55E00", linewidth=1.0)
    axes[1].axhline(0, color="#777777", linewidth=0.6)
    axes[1].set(
        xlabel="Responsive trial fraction",
        ylabel="Mean response - baseline",
        title=f"Selected neurons: {int(selected.sum()):,} / {len(summary):,}",
    )
    figure.tight_layout()
    return figure
