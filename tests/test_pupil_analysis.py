import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attention_alignment.pupil_analysis import (
    add_original_trial_groups,
    add_trial_groups,
    extract_item_trials,
    extract_sequence_trials,
    infer_stimulus_window,
    pair_immediate_preceding_b,
    p5_minus_p4,
    plot_first_last_group_traces,
    resampled_trace_effects,
    sequence_stimulus_intervals,
    stratified_b_resampling,
    trial_metrics,
)


def _events(sequence_count: int = 4) -> pd.DataFrame:
    rows = []
    global_item = 0
    for sequence_index in range(1, sequence_count + 1):
        sequence_start = 10.0 + (sequence_index - 1) * 12.0
        for item_position, symbol in enumerate("AAAAB", start=1):
            global_item += 1
            onset = sequence_start + (item_position - 1) * 1.5
            rows.append(
                {
                    "session_id": "mouse_500ms",
                    "condition_label_ms": 500,
                    "phase": "train",
                    "sequence_index": sequence_index,
                    "global_sequence_index": sequence_index,
                    "global_item_index": global_item,
                    "sequence_pattern": "AAAAB",
                    "item_position": item_position,
                    "symbol": symbol,
                    "expected_stripe_duration_s": 1.0,
                    "expected_gray_duration_s": 0.5 if item_position < 5 else 6.5,
                    "measured_onset_s": onset,
                    "measured_offset_s": onset + 1.0,
                }
            )
    return pd.DataFrame(rows)


def _behavior() -> pd.DataFrame:
    times = np.arange(0.0, 100.0, 0.05)
    radius = 10.0 + 0.1 * np.sin(times / 2.0)
    return pd.DataFrame(
        {
            "t_session_s": times,
            "pupil_valid": True,
            "pupil_equivalent_radius_interpolated": radius,
            "pupil_area_interpolated": np.pi * radius**2,
            "movement_abs_difference": 1.0 + 0.1 * np.cos(times),
            "pupil_hull_correction_fraction": 0.02,
        }
    )


def test_infer_confirmed_500ms_condition_windows():
    spec = infer_stimulus_window(_events())
    assert spec.pre_stimulus_s == 0.5
    assert spec.stimulus_s == 1.0
    assert spec.sequence_s == 7.0


def test_item_and_sequence_extraction_use_parameterized_windows():
    events = _events()
    behavior = _behavior()
    item = extract_item_trials(
        events,
        behavior,
        phase="train",
        item_positions=5,
        sample_rate_hz=20,
    )
    sequence = extract_sequence_trials(
        events,
        behavior,
        phase="train",
        sample_rate_hz=20,
    )

    assert item["trial_id"].nunique() == 4
    assert np.isclose(item["trial_time_s"].min(), -0.5)
    assert np.isclose(item["trial_time_s"].max(), 1.0)
    assert item.groupby("trial_id").size().eq(31).all()
    assert item.groupby("trial_id")["trial_valid"].first().all()
    assert sequence["trial_id"].nunique() == 4
    assert np.isclose(sequence["trial_time_s"].max(), 7.0)


def test_long_missing_pupil_gap_is_not_interpolated_across_trial():
    events = _events(sequence_count=1)
    behavior = _behavior()
    p5_onset = float(events.loc[events["item_position"].eq(5), "measured_onset_s"].iloc[0])
    gap = behavior["t_session_s"].between(p5_onset - 0.4, p5_onset + 0.8)
    behavior.loc[gap, "pupil_equivalent_radius_interpolated"] = np.nan

    item = extract_item_trials(
        events,
        behavior,
        phase="train",
        item_positions=5,
        sample_rate_hz=20,
        max_interpolation_gap_s=0.5,
    )

    assert not bool(item["trial_valid"].iloc[0])
    assert item["pupil_delta_fraction"].isna().all()


def test_p5_minus_p4_is_computed_at_matched_relative_times():
    rows = []
    for position, value in ((4, 1.0), (5, 3.0)):
        for trial_time in (-0.5, 0.0, 0.5):
            rows.append(
                {
                    "session_id": "mouse",
                    "condition_label_ms": 500,
                    "phase": "test",
                    "sequence_index": 1,
                    "global_sequence_index": 101,
                    "sequence_pattern": "AAAAA",
                    "item_position": position,
                    "symbol": "A",
                    "trial_time_s": trial_time,
                    "pupil_delta_fraction": value / 10,
                    "pupil_session_z": value,
                    "movement_z": value / 2,
                    "trial_valid": True,
                }
            )

    difference = p5_minus_p4(pd.DataFrame(rows))

    assert len(difference) == 3
    assert np.allclose(difference["p5_minus_p4_pupil_z"], 2.0)
    assert np.allclose(difference["p5_minus_p4_delta_fraction"], 0.2)
    assert difference["trial_valid"].all()


def test_group_size_controls_metrics_and_first_last_trace_plot():
    events = _events(sequence_count=6)
    behavior = _behavior()
    traces = extract_sequence_trials(events, behavior, phase="train", sample_rate_hz=10)
    metrics = trial_metrics(traces, response_window_s=(0.0, 7.0))
    grouped = add_trial_groups(metrics, group_size=2)
    spec = infer_stimulus_window(events)

    assert grouped["trial_group"].tolist() == [1, 1, 2, 2, 3, 3]
    figure = plot_first_last_group_traces(
        traces,
        group_size=2,
        error_band="sem",
        stimulus_intervals=sequence_stimulus_intervals("AAAAB", spec),
    )
    assert len(figure.axes) == 2
    assert "trials 1-2" in figure.axes[0].get_title()
    assert "trials 5-6" in figure.axes[1].get_title()
    plt.close(figure)


def test_original_trial_groups_do_not_shift_after_invalid_trials():
    metrics = pd.DataFrame(
        {
            "sequence_index": [1, 2, 4, 5, 8],
            "trial_valid": [True, False, True, True, True],
        }
    )

    grouped = add_original_trial_groups(metrics, group_size=3)

    assert grouped["condition_trial_number"].tolist() == [1, 2, 4, 5, 8]
    assert grouped["trial_group"].tolist() == [1, 1, 2, 2, 3]


def _test_metrics() -> pd.DataFrame:
    patterns = [
        "AAAAB",
        "AAAAB",
        "AAAAA",
        "AAAAB",
        "AAAAC",
        "AAAAB",
        "AAAAB",
        "AAAAA",
        "AAAAB",
        "AAAAC",
    ]
    return pd.DataFrame(
        {
            "trial_id": [f"test:{index}:P5-P4" for index in range(1, 11)],
            "sequence_index": np.arange(1, 11),
            "sequence_pattern": patterns,
            "trial_valid": True,
            "response_mean": np.arange(1, 11, dtype=float),
            "movement_mean_z": np.linspace(-0.5, 0.5, 10),
        }
    )


def test_catches_use_only_the_immediately_preceding_valid_b():
    metrics = _test_metrics()
    metrics.loc[metrics["sequence_index"].eq(4), "sequence_pattern"] = "AAAAA"

    pairs, unmatched = pair_immediate_preceding_b(metrics, catch_pattern="AAAAC")

    assert pairs["catch_sequence_index"].tolist() == [10]
    assert pairs["reference_sequence_index"].tolist() == [9]
    assert unmatched["catch_sequence_index"].tolist() == [5]
    assert unmatched["reason"].tolist() == ["immediate_predecessor_not_B"]


def test_stratified_b_resampling_matches_catch_counts_within_blocks():
    metrics = _test_metrics()

    selections, effects, summary = stratified_b_resampling(
        metrics,
        catch_pattern="AAAAA",
        block_size=5,
        repetitions=20,
        seed=11,
    )

    counts = selections.groupby(["repeat", "chronology_block"]).size()
    assert counts.eq(1).all()
    assert len(effects) == 20
    assert summary.loc[0, "repetitions"] == 20
    assert 0 <= summary.loc[0, "sign_consistency"] <= 1

    traces = pd.DataFrame(
        [
            {
                "trial_id": row.trial_id,
                "sequence_pattern": row.sequence_pattern,
                "trial_valid": True,
                "trial_time_s": trial_time,
                "p5_minus_p4_pupil_z": row.response_mean + trial_time,
            }
            for row in metrics.itertuples(index=False)
            for trial_time in (0.0, 1.0)
        ]
    )
    repeat_traces, trace_summary = resampled_trace_effects(
        traces,
        selections,
        catch_pattern="AAAAA",
    )
    assert repeat_traces["repeat"].nunique() == 20
    assert trace_summary["trial_time_s"].tolist() == [0.0, 1.0]
