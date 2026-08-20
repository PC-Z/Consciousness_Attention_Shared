import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from attention_alignment.errors import SourceFormatError
from attention_alignment.neural_analysis import (
    adjust_pvalues_holm,
    analyze_balanced_p5_responses,
    analyze_train_sequence_adaptation,
    build_sequence_trace_windows,
    build_stimulus_windows,
    compare_neuron_response_means,
    compute_paired_p5_neuron_effects,
    compute_peak_time_row_order,
    compute_stimulus_response_statistics,
    downsample_neural_activity,
    estimate_baseline_noise_scale,
    extract_event_aligned_neural_traces,
    load_neuron_atlas,
    minmax_normalize_neuron_trial_traces,
    open_neural_activity,
    plot_activity_heatmap,
    plot_condition_neuron_heatmaps,
    plot_condition_neuron_traces,
    plot_condition_neurons_on_atlas,
    plot_balanced_p5_heatmaps,
    plot_balanced_p5_traces,
    plot_fixed_population_condition_atlas,
    plot_fixed_order_trial_heatmaps,
    plot_neurons_on_atlas,
    plot_p5_neuron_effect_scatter,
    plot_representative_p5_traces,
    plot_response_selection_summary,
    plot_selected_neuron_heatmap,
    plot_selected_neuron_traces,
    plot_selected_trial_heatmap,
    plot_train_adaptation_summary,
    plot_train_sequence_mean_sem,
    prepare_selected_neuron_heatmap,
    screen_stimulus_conditions,
    select_atlas_cell_indices,
    select_balanced_test_p5_events,
    select_paired_catch_reference_events,
    select_representative_p5_neurons,
    select_stimulus_responsive_neurons,
    subset_stimulus_response_statistics,
    summarize_condition_screens,
    summarize_mean_stimulus_responses,
    summarize_stimulus_responses,
    repeated_p5_reference_subsampling,
    robust_normalize_neuron_trial_traces,
)


def test_open_neural_activity_orients_cells_by_frames(tmp_path):
    path = tmp_path / "trace.npy"
    np.save(path, np.arange(15, dtype=np.float32).reshape(5, 3))

    activity = open_neural_activity(path, frame_count=5, cell_count=3)

    assert activity.shape == (3, 5)
    assert np.array_equal(activity[0], [0, 3, 6, 9, 12])


def test_open_neural_activity_rejects_unmatched_axes(tmp_path):
    path = tmp_path / "trace.npy"
    np.save(path, np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(SourceFormatError, match="does not match"):
        open_neural_activity(path, frame_count=5, cell_count=3)


def test_load_neuron_atlas_decodes_matlab_reference_strings(tmp_path):
    path = tmp_path / "atlas.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("whole_center_2d_T", data=[[1.0, 3.0], [2.0, 4.0]])
        handle.create_dataset("ids", data=[[10.0, 20.0]])
        handle.create_dataset("CCF", data=np.eye(5, dtype=np.uint8))
        acs = handle.create_dataset("acs", shape=(2, 1), dtype=h5py.ref_dtype)
        names = handle.create_dataset("names", shape=(2, 1), dtype=h5py.ref_dtype)
        for index, (acronym, name) in enumerate((("VISp", "Visual"), ("MOs", "Motor"))):
            acs_data = handle.create_dataset(
                f"acs_{index}", data=np.asarray([[ord(char)] for char in acronym], dtype=np.uint16)
            )
            name_data = handle.create_dataset(
                f"name_{index}", data=np.asarray([[ord(char)] for char in name], dtype=np.uint16)
            )
            acs[index, 0] = acs_data.ref
            names[index, 0] = name_data.ref

    atlas, ccf = load_neuron_atlas(path)

    assert atlas["atlas_acronym"].tolist() == ["VISp", "MOs"]
    assert atlas["atlas_name"].tolist() == ["Visual", "Motor"]
    assert atlas[["x", "y"]].to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert ccf.shape == (5, 5)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "phase": ["train", "train"],
            "symbol": ["B", "B"],
            "item_position": [5, 5],
            "measured_onset_s": [10.0, 20.0],
            "measured_offset_s": [13.0, 23.0],
            "timing_qc": ["ok", "ok"],
        }
    )


def test_reference_response_rule_and_trial_fraction_select_stable_cells():
    times = np.arange(30, dtype=float)
    activity = np.zeros((3, len(times)), dtype=np.float32)
    for onset in (10, 20):
        activity[0, onset : onset + 3] = [4.0, 6.0, 5.0]
        activity[1, onset : onset + 3] = 5.0
        activity[2, onset - 3 : onset - 1] = [-2.0, 2.0]
        activity[2, onset : onset + 3] = [1.0, 2.0, 1.0]

    windows = build_stimulus_windows(
        _events(), times,
        phase="train", symbols=["B"], item_positions=[5],
        baseline_window_s=(-3.0, -1.0), response_window_s=(0.0, 3.0),
        min_window_frames=2,
    )
    summary = select_stimulus_responsive_neurons(
        activity, windows,
        std_multiplier=3.0,
        require_response_std_gt_baseline=True,
        min_responsive_trials=2,
        min_responsive_fraction=1.0,
        min_valid_frames=2,
        chunk_size=2,
    )

    assert summary["responsive_trial_count"].tolist() == [2, 0, 0]
    assert summary["selected"].tolist() == [True, False, False]

    statistics = compute_stimulus_response_statistics(
        activity, windows, min_valid_frames=2, chunk_size=2
    )
    relaxed = summarize_stimulus_responses(
        statistics,
        std_multiplier=3.0,
        require_response_std_gt_baseline=False,
        min_responsive_trials=2,
        min_responsive_fraction=1.0,
    )
    assert relaxed["selected"].tolist() == [True, True, False]


def test_mean_response_screen_does_not_require_persistent_single_trial_responses():
    statistics = {
        "baseline_mean": np.zeros((2, 4)),
        "baseline_std": np.ones((2, 4)),
        "response_mean": np.asarray([[5.0, 0.0, 0.0, 0.0], [0.5] * 4]),
        "eligible": np.ones((2, 4), dtype=bool),
        "cell_index": np.asarray([11, 12]),
    }

    summary = summarize_mean_stimulus_responses(
        statistics, std_multiplier=1.0, min_eligible_trials=4
    )

    assert summary["cell_index"].tolist() == [11, 12]
    assert summary["mean_response_z"].tolist() == pytest.approx([1.25, 0.5])
    assert summary["selected"].tolist() == [True, False]
    assert "responsive_trial_fraction" not in summary.columns


def test_full_sequence_windows_and_plot_use_measured_item_timing():
    rows = []
    for sequence_index, sequence_onset in ((1, 10.0), (10, 40.0)):
        for position in range(1, 6):
            onset = sequence_onset + 2.0 * (position - 1)
            rows.append({
                "sequence_index": sequence_index,
                "sequence_pattern": "AAAAB",
                "item_position": position,
                "symbol": "B" if position == 5 else "A",
                "measured_onset_s": onset,
                "measured_offset_s": onset + 1.0,
                "window_valid": True,
            })
    sequence_data = build_sequence_trace_windows(
        pd.DataFrame(rows), sequence_indices=[1, 10]
    )

    sequence_windows = sequence_data["sequence_windows"]
    stimulus_timing = sequence_data["stimulus_timing"]
    assert sequence_windows["sequence_duration_s"].tolist() == [9.0, 9.0]
    assert stimulus_timing["relative_onset_s"].tolist() == pytest.approx(
        [0.0, 2.0, 4.0, 6.0, 8.0]
    )

    relative_time = np.linspace(-1.0, 10.0, 23)
    aligned = np.zeros((3, 2, len(relative_time)), dtype=float)
    for onset in stimulus_timing["relative_onset_s"]:
        aligned[:, :, np.argmin(np.abs(relative_time - onset))] = 2.0
    figure = plot_train_sequence_mean_sem(
        aligned,
        relative_time,
        sequence_windows,
        stimulus_timing,
        early_sequences=[1],
        late_sequences=[10],
    )

    assert len(figure.axes) == 2
    assert all(len(axis.patches) == 5 for axis in figure.axes)
    assert [text.get_text() for text in figure.axes[0].texts] == [
        "P1-A", "P2-A", "P3-A", "P4-A", "P5-B",
    ]
    plt.close(figure)

    grouped_figure = plot_train_sequence_mean_sem(
        aligned,
        relative_time,
        sequence_windows,
        stimulus_timing,
        early_sequences=[1],
        late_sequences=[10],
        population_groups={
            "A-only": [True, False, False],
            "B-only": [False, True, False],
            "A-or-B union": [True, True, True],
        },
    )

    assert len(grouped_figure.axes) == 6
    assert all(len(axis.patches) == 5 for axis in grouped_figure.axes)
    assert grouped_figure.axes[0].get_title().startswith("A-only (n=1)")
    assert grouped_figure.axes[2].get_title().startswith("B-only (n=1)")
    assert grouped_figure.axes[4].get_title().startswith("A-or-B union (n=3)")
    plt.close(grouped_figure)


def test_downsample_and_plots_use_all_time_bins():
    activity = np.arange(24, dtype=np.float32).reshape(3, 8)
    times = np.arange(8, dtype=float)

    binned_times, binned, indices = downsample_neural_activity(
        activity, times, max_time_bins=3, chunk_size=2
    )

    assert indices.tolist() == [0, 1, 2]
    assert binned.shape == (3, 3)
    assert np.allclose(binned_times, [1.0, 4.0, 6.5])
    assert np.allclose(binned[0], [1.0, 4.0, 6.5])

    sampled_times, sampled, _ = downsample_neural_activity(
        activity, times, max_time_bins=3, method="sample"
    )
    assert np.array_equal(sampled_times, [0.0, 3.0, 7.0])
    assert np.array_equal(sampled[0], [0.0, 3.0, 7.0])

    atlas = pd.DataFrame(
        {"cell_index": [0, 1, 2], "x": [1, 2, 3], "y": [2, 3, 4]}
    )
    summary = pd.DataFrame(
        {
            "responsive_trial_fraction": [0.1, 0.6, 0.8],
            "mean_response_minus_baseline": [-0.1, 0.2, 0.4],
            "selected": [False, True, True],
        }
    )
    figures = [
        plot_activity_heatmap(binned, binned_times),
        plot_neurons_on_atlas(
            atlas, np.ones((6, 6), dtype=bool),
            selected=summary["selected"].to_numpy(),
            selection_values=summary["responsive_trial_fraction"].to_numpy(),
        ),
        plot_response_selection_summary(summary, fraction_threshold=0.5),
    ]
    assert all(figure.axes for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_selected_neuron_traces_and_heatmaps_are_event_aligned():
    times = np.arange(0.0, 30.0, 0.5)
    activity = np.zeros((2, len(times)), dtype=np.float32)
    for onset in (10.0, 20.0):
        response = (times >= onset) & (times <= onset + 1.0)
        activity[0, response] = [1.0, 3.0, 2.0]
        activity[1, response] = [2.0, 1.0, 0.5]
    windows = build_stimulus_windows(
        _events(), times,
        phase="train", symbols=["B"], item_positions=[5],
        baseline_window_s=(-1.0, 0.0), response_window_s=(0.0, 2.0),
        min_window_frames=2,
    )

    aligned = extract_event_aligned_neural_traces(
        activity,
        times,
        windows,
        [0, 1],
        trace_window_s=(-1.0, 2.0),
        baseline_window_s=(-1.0, 0.0),
        sample_rate_hz=2.0,
    )
    prepared = prepare_selected_neuron_heatmap(
        aligned["delta"],
        aligned["relative_time_s"],
        response_window_s=(0.0, 2.0),
        normalization="minmax",
    )
    trial_population = np.nanmean(aligned["delta"], axis=0)

    assert aligned["delta"].shape == (2, 2, 7)
    assert np.allclose(aligned["baseline_mean"], 0.0)
    assert prepared["heatmap"].shape == (2, 7)
    assert np.nanmin(prepared["heatmap"]) == 0.0
    assert np.nanmax(prepared["heatmap"]) == 1.0

    figures = [
        plot_selected_neuron_traces(
            prepared["mean_delta"],
            aligned["relative_time_s"],
            stimulus_duration_s=1.0,
            baseline_window_s=(-1.0, 0.0),
            response_stop_s=2.0,
        ),
        plot_selected_neuron_heatmap(
            prepared["heatmap"],
            aligned["relative_time_s"],
            stimulus_duration_s=1.0,
            normalization="minmax",
        ),
        plot_selected_trial_heatmap(
            trial_population,
            aligned["relative_time_s"],
            stimulus_duration_s=1.0,
        ),
    ]
    assert all(figure.axes for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_condition_screens_select_and_plot_each_population_independently():
    times = np.arange(0.0, 50.0, 0.5)
    events = pd.DataFrame(
        {
            "phase": ["train", "train", "test", "test", "test", "test", "test", "test"],
            "symbol": ["B", "B", "A", "A", "B", "B", "C", "C"],
            "item_position": [5] * 8,
            "measured_onset_s": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
            "measured_offset_s": [6.0, 11.0, 16.0, 21.0, 26.0, 31.0, 36.0, 41.0],
            "timing_qc": ["ok"] * 8,
        }
    )
    activity = np.zeros((4, len(times)), dtype=np.float32)
    condition_onsets = (
        (5.0, 10.0),
        (15.0, 20.0),
        (25.0, 30.0),
        (35.0, 40.0),
    )
    for cell_index, onsets in enumerate(condition_onsets):
        for onset in onsets:
            response = (times >= onset) & (times <= onset + 1.0)
            activity[cell_index, response] = [1.0, 3.0, 2.0]
    conditions = (
        {"key": "train_b", "label": "Train B", "phase": "train", "symbols": ("B",),
         "item_positions": (5,)},
        {"key": "test_a", "label": "Test A", "phase": "test", "symbols": ("A",),
         "item_positions": (5,)},
        {"key": "test_b", "label": "Test B", "phase": "test", "symbols": ("B",),
         "item_positions": (5,)},
        {"key": "test_c", "label": "Test C", "phase": "test", "symbols": ("C",),
         "item_positions": (5,)},
    )

    results = screen_stimulus_conditions(
        activity,
        times,
        events,
        conditions,
        baseline_window_s=(-1.0, 0.0),
        response_window_s=(0.0, 1.0),
        trace_window_s=(-1.0, 1.0),
        std_multiplier=1.0,
        require_response_std_gt_baseline=False,
        min_responsive_trials=2,
        min_responsive_fraction=1.0,
        min_valid_frames=2,
        chunk_size=2,
        sample_rate_hz=2.0,
    )

    assert [results[key]["selected_cell_index"].tolist() for key in results] == [
        [0], [1], [2], [3]
    ]
    condition_summary = summarize_condition_screens(results)
    assert condition_summary["valid_trials"].tolist() == [2, 2, 2, 2]
    assert condition_summary["selected_neurons"].tolist() == [1, 1, 1, 1]

    atlas = pd.DataFrame(
        {
            "cell_index": np.arange(4),
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [4.0, 3.0, 2.0, 1.0],
        }
    )
    figures = [
        plot_condition_neuron_traces(
            results, baseline_window_s=(-1.0, 0.0), max_individual_traces=10
        ),
        plot_condition_neuron_heatmaps(results, normalization="minmax"),
        plot_condition_neurons_on_atlas(
            atlas, np.ones((6, 6), dtype=bool), results
        ),
    ]
    assert all(figure.axes for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_fixed_cell_statistics_preserve_source_indices_and_region_boundary():
    times = np.arange(30, dtype=float)
    activity = np.zeros((3, len(times)), dtype=np.float32)
    activity[2, 10:13] = [2.0, 4.0, 3.0]
    activity[2, 20:23] = [2.0, 4.0, 3.0]
    windows = build_stimulus_windows(
        _events(), times,
        phase="train", symbols=["B"], item_positions=[5],
        baseline_window_s=(-3.0, -1.0), response_window_s=(0.0, 3.0),
        min_window_frames=2,
    )
    statistics = compute_stimulus_response_statistics(
        activity, windows, cell_indices=[2, 0], min_valid_frames=2
    )
    summary = summarize_stimulus_responses(
        statistics, require_response_std_gt_baseline=False,
        min_responsive_trials=1, min_responsive_fraction=0.0,
    )
    subset = subset_stimulus_response_statistics(statistics, [True, False])

    assert statistics["cell_index"].tolist() == [2, 0]
    assert summary["cell_index"].tolist() == [2, 0]
    assert subset["response_mean"].shape == (2, 1)

    atlas = pd.DataFrame({
        "cell_index": [0, 1, 2, 3],
        "atlas_acronym": ["VISp1", "VISp2/3", "VISpm1", "MOs1"],
    })
    assert select_atlas_cell_indices(atlas, "VISp").tolist() == [0, 1]
    assert select_atlas_cell_indices(atlas, "all").tolist() == [0, 1, 2, 3]


def test_balanced_p5_and_paper_inspired_analyses_keep_matched_neurons():
    p5_rows = []
    onset = 10.0
    patterns = ["AAAAB", "AAAAA", "AAAAB", "AAAAC"] * 3
    for sequence_index, pattern in enumerate(patterns, start=1):
        symbol = pattern[-1]
        p5_rows.append({
            "phase": "test", "sequence_index": sequence_index,
            "item_position": 5, "sequence_pattern": pattern, "symbol": symbol,
            "measured_onset_s": onset, "measured_offset_s": onset + 1,
            "timing_qc": "ok", "source_event_index": sequence_index - 1,
        })
        onset += 5
    balanced = select_balanced_test_p5_events(
        pd.DataFrame(p5_rows), trial_count=3, random_seed=3
    )
    assert balanced.groupby("symbol").size().to_dict() == {"A": 3, "B": 3, "C": 3}
    assert set(balanced.loc[balanced["symbol"].eq("B"), "balance_source"]) == {
        "immediately_preceding_catch"
    }

    train_windows = pd.DataFrame([
        {
            "sequence_index": sequence, "item_position": position,
            "symbol": "B" if position == 5 else "A",
        }
        for sequence in range(1, 11)
        for position in range(1, 6)
    ])
    train_response = np.ones((4, len(train_windows)), dtype=np.float32)
    p5 = train_windows["item_position"].eq(5).to_numpy()
    early = train_windows["sequence_index"].le(2).to_numpy()
    late = train_windows["sequence_index"].ge(9).to_numpy()
    train_response[:, p5 & early] = 3.0
    train_response[:, p5 & late] = 0.5
    train_statistics = {
        "baseline_mean": np.zeros_like(train_response),
        "baseline_std": np.ones_like(train_response),
        "response_mean": train_response,
        "response_std": np.ones_like(train_response),
        "eligible": np.ones_like(train_response, dtype=bool),
        "cell_index": np.arange(4),
        "window_index": np.arange(len(train_windows)),
    }
    train_analysis = analyze_train_sequence_adaptation(
        train_statistics, train_windows,
        early_trial_count=2, late_trial_count=2, bin_size=2,
    )
    assert len(train_analysis["p5_trajectory"]) == 10
    assert train_analysis["p5_directional_test"].iloc[0]["mean_difference"] > 0

    test_windows = balanced.reset_index(drop=True)
    test_response = np.zeros((4, len(test_windows)), dtype=np.float32)
    for column, symbol in enumerate(test_windows["symbol"]):
        test_response[:, column] = {"A": -1.0, "B": 0.0, "C": 2.0}[symbol]
    test_statistics = {
        "baseline_mean": np.zeros_like(test_response),
        "baseline_std": np.ones_like(test_response),
        "response_mean": test_response,
        "response_std": np.ones_like(test_response),
        "eligible": np.ones_like(test_response, dtype=bool),
        "cell_index": np.arange(4),
        "window_index": np.arange(len(test_windows)),
    }
    scale = estimate_baseline_noise_scale(test_statistics)
    test_analysis = analyze_balanced_p5_responses(
        test_statistics, test_windows, scale=scale
    )
    assert test_analysis["neuron_condition_response"].columns.tolist() == [
        "cell_index", "A", "B", "C"
    ]
    assert adjust_pvalues_holm([0.01, 0.04, 0.03]).tolist() == pytest.approx(
        [0.03, 0.06, 0.06]
    )
    comparison = compare_neuron_response_means(
        test_response[:, test_windows["symbol"].eq("C")],
        test_response[:, test_windows["symbol"].eq("B")],
        comparison="C > B", alternative="greater",
    )
    assert comparison["mean_difference"] == pytest.approx(2.0)

    relative_times = np.linspace(-1, 2, 7)
    aligned = np.repeat(test_response[:, :, np.newaxis], len(relative_times), axis=2)
    figures = [
        plot_train_adaptation_summary(train_analysis, title="Train"),
        plot_balanced_p5_traces(
            aligned, relative_times, test_windows, stimulus_duration_s=1.0
        ),
    ]
    heatmap_figure, heatmap_data = plot_balanced_p5_heatmaps(
        aligned, relative_times, test_windows, stimulus_duration_s=1.0
    )
    figures.append(heatmap_figure)
    selections = {
        symbol: pd.DataFrame({"cell_index": np.arange(4), "selected": [True] * 4})
        for symbol in ("A", "B", "C")
    }
    atlas = pd.DataFrame({
        "cell_index": np.arange(4), "x": [1, 2, 3, 4], "y": [1, 2, 3, 4]
    })
    figures.append(plot_fixed_population_condition_atlas(
        atlas, np.ones((6, 6), dtype=bool), selections, title="Atlas"
    ))
    assert len(heatmap_data["row_order"]) == 4
    assert all(figure.axes for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_paired_catch_references_and_repeated_subsampling_are_auditable():
    rows = []
    patterns = ["AAAAB", "AAAAA", "AAAAC", "AAAAB"] * 3
    for sequence_index, pattern in enumerate(patterns, start=1):
        rows.append({
            "phase": "test",
            "sequence_index": sequence_index,
            "item_position": 5,
            "sequence_pattern": pattern,
            "symbol": pattern[-1],
            "measured_onset_s": sequence_index * 5.0,
            "measured_offset_s": sequence_index * 5.0 + 1.0,
            "timing_qc": "ok",
            "source_event_index": sequence_index - 1,
        })
    events = pd.DataFrame(rows)

    pairs = select_paired_catch_reference_events(events)
    assert pairs.groupby(["catch_symbol", "condition_label"]).size().to_dict() == {
        ("A", "A"): 3,
        ("A", "B_A"): 3,
        ("C", "B_C"): 3,
        ("C", "C"): 3,
    }
    assert pairs.groupby("catch_symbol")["pair_index"].nunique().to_dict() == {
        "A": 3, "C": 3,
    }
    assert pairs.groupby("catch_symbol")["pair_gap_s"].first().to_dict() == {
        "A": 5.0, "C": 10.0,
    }

    metadata = events.reset_index(drop=True)
    scores = np.zeros((2, len(metadata)), dtype=np.float32)
    scores[:, metadata["symbol"].eq("A")] = 1.0
    scores[:, metadata["symbol"].eq("C")] = 2.0
    sensitivity = repeated_p5_reference_subsampling(
        scores,
        metadata,
        cell_indices=[11, 12],
        repeats=25,
        random_seed=7,
    )
    assert sensitivity["summary"]["population_mean_effect"].tolist() == pytest.approx(
        [1.0, 2.0]
    )
    assert set(sensitivity["neuron_summary"]["cell_index"]) == {11, 12}
    for _, group in sensitivity["reference_inclusion"].groupby("comparison"):
        assert group["inclusion_count"].sum() == 25 * 3
        assert group["expected_inclusion_probability"].iloc[0] == pytest.approx(0.5)
    assert len(sensitivity["sampled_reference_trials"]) == 2 * 25 * 3


def test_peak_time_order_and_matched_heatmaps_preserve_external_rows():
    times = np.array([-0.5, 0.0, 0.5, 1.0])
    sort_traces = np.array([
        [0.0, 0.0, 2.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 1.0],
        [np.nan, np.nan, np.nan, np.nan],
    ])
    ordering = compute_peak_time_row_order(
        sort_traces, times, response_window_s=(0.0, 1.0)
    )
    assert ordering["row_order"].tolist() == [1, 2, 0, 3]

    metadata = pd.DataFrame({"symbol": ["A", "A", "B", "B", "C", "C"]})
    aligned = np.arange(4 * 6 * 4, dtype=float).reshape(4, 6, 4)
    figure, plotted = plot_balanced_p5_heatmaps(
        aligned, times, metadata,
        stimulus_duration_s=1.0,
        row_order=ordering["row_order"],
        normalization="baseline_sd",
        z_limit=None,
    )
    assert plotted["row_order"].tolist() == [1, 2, 0, 3]
    assert plotted["normalization"] == "baseline_sd"
    expected_a = np.nanmean(aligned[:, :2], axis=1)[ordering["row_order"]]
    assert np.array_equal(figure.axes[0].images[0].get_array(), expected_a)
    assert all(
        np.array_equal(axis.images[0].get_array().shape, expected_a.shape)
        for axis in figure.axes[:3]
    )
    plt.close(figure)


def test_first_trial_order_is_reused_for_all_single_trial_heatmaps():
    times = np.array([-0.5, 0.0, 0.5, 1.0])
    traces = np.arange(3 * 10 * 4, dtype=float).reshape(3, 10, 4)
    order = np.array([2, 0, 1])
    figure, plotted = plot_fixed_order_trial_heatmaps(
        traces, times, order,
        trial_labels=[f"Trial {index}" for index in range(1, 11)],
        stimulus_duration_s=1.0,
    )
    assert plotted["row_order"].tolist() == order.tolist()
    for trial, axis in enumerate(figure.axes[:10]):
        assert np.array_equal(axis.images[0].get_array(), traces[order, trial])
        assert axis.images[0].get_clim() == (
            -plotted["color_limit"], plotted["color_limit"]
        )
    plt.close(figure)


def test_robust_trial_trace_normalization_uses_one_scale_across_groups():
    train = np.array([
        [[-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        [[-20.0, 0.0, 20.0], [-10.0, 0.0, 10.0]],
    ])
    test = train * 0.5

    result = robust_normalize_neuron_trial_traces(
        (train, test), robust_percentile=100.0, floor_percentile=0.0
    )
    normalized_train, normalized_test = result["traces"]

    assert result["scale"].tolist() == pytest.approx([2.0, 20.0])
    assert np.nanmax(np.abs(normalized_train)) == pytest.approx(1.0)
    assert np.nanmax(np.abs(normalized_test)) == pytest.approx(0.5)
    assert np.allclose(normalized_test, normalized_train * 0.5)


def test_minmax_trial_trace_normalization_shares_neuron_bounds_across_groups():
    train = np.array([
        [[-2.0, 0.0, 2.0]],
        [[5.0, 10.0, 15.0]],
        [[3.0, 3.0, 3.0]],
    ])
    test = np.array([
        [[-1.0, 1.0]],
        [[7.5, 12.5]],
        [[3.0, np.nan]],
    ])

    result = minmax_normalize_neuron_trial_traces((train, test))
    normalized_train, normalized_test = result["traces"]

    assert result["minimum"].tolist() == pytest.approx([-2.0, 5.0, 3.0])
    assert result["maximum"].tolist() == pytest.approx([2.0, 15.0, 3.0])
    assert normalized_train[0, 0].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert normalized_test[0, 0].tolist() == pytest.approx([0.25, 0.75])
    assert normalized_test[1, 0].tolist() == pytest.approx([0.25, 0.75])
    assert np.array_equal(normalized_train[2, 0], [0.0, 0.0, 0.0])
    assert normalized_test[2, 0, 0] == pytest.approx(0.0)
    assert np.isnan(normalized_test[2, 0, 1])


def test_fixed_order_trial_heatmaps_accepts_fixed_display_scale():
    times = np.array([-0.5, 0.0, 0.5])
    traces = np.array([
        [[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]],
        [[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]],
    ])
    figure, plotted = plot_fixed_order_trial_heatmaps(
        traces, times, [1, 0], trial_labels=["Trial 1", "Trial 2"],
        stimulus_duration_s=0.5, color_limit=1.0,
        colorbar_label="Shared robust normalized response", figure_title="Test title",
    )

    assert plotted["color_limit"] == pytest.approx(1.0)
    assert figure.axes[0].images[0].get_clim() == (-1.0, 1.0)
    assert figure._suptitle.get_text() == "Test title"
    plt.close(figure)


def test_fixed_order_trial_heatmaps_accepts_activation_only_color_range():
    times = np.array([-0.5, 0.0, 0.5])
    traces = np.array([[[0.0, 0.5, 1.0]], [[-1.0, 0.0, 0.5]]])
    figure, plotted = plot_fixed_order_trial_heatmaps(
        traces, times, [0, 1], trial_labels=["Trial 1"],
        stimulus_duration_s=0.5, color_range=(0.0, 1.0), cmap="Reds",
    )

    assert figure.axes[0].images[0].get_clim() == (0.0, 1.0)
    assert plotted["color_range"].tolist() == [0.0, 1.0]
    plt.close(figure)


def test_paired_effects_and_representative_selection_are_deterministic():
    metadata = pd.DataFrame({"source_event_index": [10, 11, 12, 13]})
    scores = np.vstack([
        np.array([row, row + 1, row + 3, row + 2], dtype=float)
        for row in np.linspace(-2, 2, 12)
    ])
    paired = pd.DataFrame([
        {"catch_symbol": "A", "pair_index": 1, "pair_role": "catch", "condition_label": "A", "source_event_index": 10},
        {"catch_symbol": "A", "pair_index": 1, "pair_role": "reference", "condition_label": "B_A", "source_event_index": 11},
        {"catch_symbol": "C", "pair_index": 1, "pair_role": "catch", "condition_label": "C", "source_event_index": 12},
        {"catch_symbol": "C", "pair_index": 1, "pair_role": "reference", "condition_label": "B_C", "source_event_index": 13},
    ])
    effects = compute_paired_p5_neuron_effects(
        scores, metadata, paired, cell_indices=np.arange(100, 112)
    )
    assert np.allclose(effects["A_minus_B_A"], -1.0)
    assert np.allclose(effects["C_minus_B_C"], 1.0)

    effects["A_minus_B_A"] = np.linspace(-2.0, 2.0, len(effects))
    effects["C_minus_B_C"] = np.linspace(2.5, -1.5, len(effects))
    first = select_representative_p5_neurons(effects)
    second = select_representative_p5_neurons(effects)
    pd.testing.assert_frame_equal(first, second)
    assert first["representative_group"].value_counts().to_dict() == {
        "C-enhanced": 2, "B-dominant": 2, "stable": 2,
    }
    assert first["cell_index"].is_unique

    figure, quadrants = plot_p5_neuron_effect_scatter(effects, representatives=first)
    assert quadrants["neuron_count"].sum() == len(effects)
    plt.close(figure)

    aligned = np.ones((len(effects), 6, 4), dtype=float)
    condition_metadata = pd.DataFrame({"symbol": ["A", "A", "B", "B", "C", "C"]})
    representative_figure = plot_representative_p5_traces(
        aligned, np.array([-0.5, 0.0, 0.5, 1.0]), condition_metadata,
        effects["cell_index"], first,
        stimulus_duration_s=1.0,
    )
    assert len(representative_figure.axes) == 6
    plt.close(representative_figure)
