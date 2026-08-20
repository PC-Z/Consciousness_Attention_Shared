from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .calcium import read_calcium_window as _read_calcium_window
from .config import ProjectConfig, load_config
from .ephys import read_ephys_window as _read_ephys_window


class AlignedSession:
    """Read-only access to one exported compact alignment."""

    def __init__(self, session_id: str, config: ProjectConfig):
        self.session_id = session_id
        self.config = config
        self.output_dir = (config.output_root / session_id).resolve()
        with (self.output_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        with (self.output_dir / "ephys_02_manifest.json").open("r", encoding="utf-8") as handle:
            self.ephys_manifest = json.load(handle)
        self.stimulus_events = pd.read_parquet(self.output_dir / "stimulus_events.parquet")
        self.calcium_frames = pd.read_parquet(self.output_dir / "calcium_frames.parquet")

    def read_calcium_window(
        self,
        start: float,
        stop: float,
        cells: list[int] | None = None,
        kind: str = "whole_trace_dff",
    ) -> dict[str, Any]:
        return _read_calcium_window(
            self.manifest["sources"]["calcium"]["path"],
            self.calcium_frames,
            start,
            stop,
            cells,
            kind,
        )

    def read_ephys_window(
        self,
        start: float,
        stop: float,
        channels: list[str] | None = None,
    ) -> pd.DataFrame:
        return _read_ephys_window(self.ephys_manifest, start, stop, channels)

    def extract_trials(
        self,
        filters: dict[str, object] | None,
        window: tuple[float, float],
    ) -> dict[str, pd.DataFrame]:
        """Return native-timestamp indexes around filtered measured onsets."""

        events = self.stimulus_events
        if filters:
            for column, value in filters.items():
                if column not in events:
                    raise KeyError(f"Unknown stimulus filter column: {column}")
                values = value if isinstance(value, (list, tuple, set)) else [value]
                events = events.loc[events[column].isin(values)]
        events = events.reset_index(drop=True).copy()
        events["trial_id"] = range(len(events))
        event_windows = events[["trial_id", "measured_onset_s"]].copy()
        event_windows["window_start_s"] = event_windows["measured_onset_s"] + window[0]
        event_windows["window_stop_s"] = event_windows["measured_onset_s"] + window[1]

        calcium_parts = []
        for trial in event_windows.itertuples(index=False):
            selected = self.calcium_frames.loc[
                (self.calcium_frames["t_session_s"] >= trial.window_start_s)
                & (self.calcium_frames["t_session_s"] < trial.window_stop_s)
            ].copy()
            selected["trial_id"] = trial.trial_id
            selected["trial_time_s"] = selected["t_session_s"] - trial.measured_onset_s
            calcium_parts.append(selected)
        result: dict[str, pd.DataFrame] = {
            "events": events,
            "windows": event_windows,
            "calcium_frames": pd.concat(calcium_parts, ignore_index=True)
            if calcium_parts
            else self.calcium_frames.iloc[0:0].copy(),
        }
        ephys_parts = []
        for trial in event_windows.itertuples(index=False):
            selected = self.read_ephys_window(trial.window_start_s, trial.window_stop_s)
            selected["trial_id"] = trial.trial_id
            selected["trial_time_s"] = selected["t_session_s"] - trial.measured_onset_s
            ephys_parts.append(selected)
        result["ephys_02"] = (
            pd.concat(ephys_parts, ignore_index=True) if ephys_parts else pd.DataFrame()
        )
        for camera in ("01", "02"):
            for prefix in ("video_frames", "behavior"):
                path = self.output_dir / f"{prefix}_{camera}.parquet"
                if not path.exists():
                    continue
                table = pd.read_parquet(path)
                parts = []
                for trial in event_windows.itertuples(index=False):
                    selected = table.loc[
                        (table["t_session_s"] >= trial.window_start_s)
                        & (table["t_session_s"] < trial.window_stop_s)
                    ].copy()
                    selected["trial_id"] = trial.trial_id
                    selected["trial_time_s"] = selected["t_session_s"] - trial.measured_onset_s
                    parts.append(selected)
                result[f"{prefix}_{camera}"] = pd.concat(parts, ignore_index=True) if parts else table.iloc[0:0]
        return result


def open_session(
    session_id: str,
    config: str | Path | ProjectConfig | None = None,
) -> AlignedSession:
    project = config if isinstance(config, ProjectConfig) else load_config(config)
    return AlignedSession(session_id, project)
