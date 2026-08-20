import json
from pathlib import Path

import numpy as np

from attention_alignment.ephys import (
    EPHYS_DTYPE,
    build_ephys_manifest,
    discover_dat_segments,
    read_ephys_window,
)
from attention_alignment.models import TimeTransform


def test_dat_restart_and_lazy_window(tmp_path: Path):
    data = np.zeros(10, dtype=EPHYS_DTYPE)
    data["tick"][:4] = np.arange(4) * 10_000
    data["tick"][4:] = np.arange(6) * 10_000
    data["EEG1"] = np.arange(10)
    dat = tmp_path / "02-0.dat"
    data.tofile(dat)
    metadata = tmp_path / "02.json"
    metadata.write_text(
        json.dumps(
            {
                "Signals": [
                    {"Label": label, "Sample rate": 1000}
                    for label in ("EEG1", "EEG2", "EMG", "Activ")
                ]
            }
        ),
        encoding="utf-8",
    )
    segments = discover_dat_segments(dat)
    assert [item["record_count"] for item in segments] == [4, 6]
    transform = TimeTransform("02", 0.0, 10.0, 0.0, 10.0)
    manifest = build_ephys_manifest(dat, metadata, 1, transform)
    window = read_ephys_window(manifest, 10.001, 10.004, ["EEG1"])
    assert window["EEG1"].tolist() == [5.0, 6.0, 7.0]
