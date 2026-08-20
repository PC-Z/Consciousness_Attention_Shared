from pathlib import Path

import numpy as np

from attention_alignment.timestamps import (
    alignment_qc,
    choose_main_segment,
    cluster_channel2,
    parse_marker_file,
)


def test_parse_restarted_segments_and_cluster_duplicates(tmp_path: Path):
    source = tmp_path / "02.txt"
    source.write_text(
        "0.1 R-Start 1000000\n"
        "0.2 1 2000000\n"
        "0.3 R-Start 3000000\n"
        "0.4 1 4000000\n"
        "0.5 2 5000000\n"
        "0.5 2 5000000\n"
        "0.5002 2 5002000\n"
        "1.0 2 10000000\n",
        encoding="utf-8",
    )
    segments = parse_marker_file(source)
    assert len(segments) == 2
    assert segments[0].close_reason == "next_start"
    assert choose_main_segment(segments).segment_index == 1
    clusters = cluster_channel2(segments[1])
    assert len(clusters) == 2
    assert clusters[0].record_count == 3


def test_first_calcium_alignment_qc(tmp_path: Path):
    contents = (
        "0.0 R-Start 100000000\n"
        "2.0 1 120000000\n"
        "2.2 1 122000000\n"
        "3.0 2 130000000\n"
        "3.0 2 130000000\n"
        "3.5 2 135000000\n"
        "4.0 R-End 140000000\n"
    )
    paths = []
    for stream, shift in (("01", 0.0), ("02", 7.0)):
        path = tmp_path / f"{stream}.txt"
        shifted = []
        for line in contents.splitlines():
            fields = line.split()
            fields[0] = str(float(fields[0]) + shift)
            shifted.append(" ".join(fields))
        path.write_text("\n".join(shifted) + "\n", encoding="utf-8")
        paths.append(path)
    first = parse_marker_file(paths[0])[0]
    second = parse_marker_file(paths[1])[0]
    qc = alignment_qc(first, second)
    assert np.isclose(qc["calcium_max_abs_residual_s"], 0.0)
    assert np.isclose(qc["trigger_max_abs_nearest_residual_s"], 0.0)
    assert qc["passes_1ms"]
