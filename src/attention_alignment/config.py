from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError


@dataclass(frozen=True)
class ConditionConfig:
    label_ms: int
    expected_stripe_s: float
    expected_gray_s: float
    expected_group_gap_s: float
    onset_tolerance_s: float
    gray_tolerance_s: float
    order_file: str
    reference_video: str


@dataclass(frozen=True)
class SessionConfig:
    id: str
    condition_label_ms: int
    video_status: str = "available"


@dataclass(frozen=True)
class ProjectConfig:
    config_path: Path
    project_root: Path
    workspace_root: Path
    data_root: Path
    stimuli_root: Path
    output_root: Path
    stimulus: dict[str, Any]
    calcium: dict[str, Any]
    ephys: dict[str, Any]
    behavior: dict[str, Any]
    sessions: tuple[SessionConfig, ...]
    conditions: dict[int, ConditionConfig]

    def session(self, session_id: str) -> SessionConfig:
        for session in self.sessions:
            if session.id == session_id:
                return session
        raise ConfigurationError(f"Unknown session: {session_id}")

    def condition(self, label_ms: int) -> ConditionConfig:
        try:
            return self.conditions[int(label_ms)]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown condition label: {label_ms}") from exc

    def session_dir(self, session_id: str) -> Path:
        return self.data_root / session_id

    def eeg_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "eeg"

    def marker_path(self, session_id: str, stream: str) -> Path:
        return self.eeg_dir(session_id) / f"{stream}.txt"

    def video_path(self, session_id: str, stream: str) -> Path:
        return self.eeg_dir(session_id) / f"{stream}.mp4"

    def calcium_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / str(self.calcium["file"])

    def order_path(self, label_ms: int) -> Path:
        return self.stimuli_root / self.condition(label_ms).order_file

    def reference_video_path(self, label_ms: int) -> Path:
        return self.stimuli_root / self.condition(label_ms).reference_video


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load and validate the project configuration.

    Relative paths are resolved from the directory containing the YAML file.
    This makes notebook and CLI behavior independent of the current directory.
    """

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "sessions.yaml"
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw.get("schema_version") != 1:
        raise ConfigurationError("Only sessions.yaml schema_version 1 is supported")

    base = config_path.parent
    project_root = config_path.parents[1]
    conditions: dict[int, ConditionConfig] = {}
    for raw_label, item in raw["stimulus"]["conditions"].items():
        label = int(raw_label)
        conditions[label] = ConditionConfig(label_ms=label, **item)
    sessions = tuple(SessionConfig(**item) for item in raw["sessions"])
    configured_labels = set(conditions)
    unknown = {session.condition_label_ms for session in sessions} - configured_labels
    if unknown:
        raise ConfigurationError(f"Sessions use undefined condition labels: {sorted(unknown)}")

    result = ProjectConfig(
        config_path=config_path,
        project_root=project_root,
        workspace_root=_resolve(base, raw["workspace_root"]),
        data_root=_resolve(base, raw["data_root"]),
        stimuli_root=_resolve(base, raw["stimuli_root"]),
        output_root=_resolve(base, raw["output_root"]),
        stimulus=raw["stimulus"],
        calcium=raw["calcium"],
        ephys=raw["ephys"],
        behavior=raw["behavior"],
        sessions=sessions,
        conditions=conditions,
    )
    if result.project_root not in result.output_root.parents:
        raise ConfigurationError("output_root must be inside alignment_pipeline")
    if result.data_root == result.output_root or result.stimuli_root == result.output_root:
        raise ConfigurationError("Source and output roots must be different")
    return result
