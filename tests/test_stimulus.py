from pathlib import Path

import pytest

from attention_alignment.config import ConditionConfig
from attention_alignment.errors import StimulusMatchError
from attention_alignment.models import TriggerCluster
from attention_alignment.stimulus import match_stimulus_blocks, parse_order_file


def condition() -> ConditionConfig:
    return ConditionConfig(
        label_ms=500,
        expected_stripe_s=1.0,
        expected_gray_s=0.5,
        expected_group_gap_s=6.5,
        onset_tolerance_s=0.1,
        gray_tolerance_s=0.1,
        order_file="order.txt",
        reference_video="stim.mp4",
    )


def cluster(index: int, time_s: float) -> TriggerCluster:
    return TriggerCluster(index, time_s, time_s, int(time_s * 1e7), int(time_s * 1e7), index, index, 1)


def block_times(start: float, groups: int = 100) -> list[float]:
    values = []
    time_s = start
    for group in range(groups):
        for item in range(5):
            values.append(time_s)
            time_s += 1.0
            values.append(time_s)
            if group == groups - 1 and item == 4:
                continue
            time_s += 6.5 if item == 4 else 0.5
    return values


def test_matcher_skips_prelude_and_between_phase_extras():
    times = [3.0] + block_times(60.0)
    times += [1500.0 + index for index in range(6)]
    times += block_times(1800.0)
    clusters = [cluster(index, value) for index, value in enumerate(times)]
    blocks, qc = match_stimulus_blocks(clusters, condition())
    assert [item.first_cluster_index for item in blocks] == [1, 1007]
    assert qc["prelude_trigger_indices"] == [0]
    assert qc["between_phase_extra_indices"] == list(range(1001, 1007))
    assert qc["formal_boundary_count"] == 2000


def test_matcher_preserves_missing_boundary():
    times = [3.0] + block_times(60.0) + block_times(1800.0)
    del times[250]
    clusters = [cluster(index, value) for index, value in enumerate(times)]
    blocks, qc = match_stimulus_blocks(clusters, condition())
    assert len(blocks) == 2
    assert qc["missing_formal_boundary_count"] == 1
    assert sum(index is None for block in blocks for index in block.cluster_indices) == 1


def test_matcher_skips_extra_boundary_inside_block():
    times = [3.0] + block_times(60.0) + block_times(1800.0)
    times.insert(250, (times[249] + times[250]) / 2)
    clusters = [cluster(index, value) for index, value in enumerate(times)]
    blocks, qc = match_stimulus_blocks(clusters, condition())
    assert len(blocks) == 2
    assert qc["missing_formal_boundary_count"] == 0
    assert len(qc["extra_within_block_indices"]) == 1


def test_order_parser_is_strict(tmp_path: Path):
    valid = tmp_path / "valid.txt"
    valid.write_text("\n".join(f"{index}: AAAAB (B)" for index in range(1, 101)), encoding="utf-8")
    assert len(parse_order_file(valid)) == 100
    invalid = tmp_path / "invalid.txt"
    invalid.write_text("1: AAAAC (B)\n", encoding="utf-8")
    with pytest.raises(StimulusMatchError):
        parse_order_file(invalid)
