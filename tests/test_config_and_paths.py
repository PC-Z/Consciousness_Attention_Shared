from pathlib import Path

import pytest

from attention_alignment.config import load_config
from attention_alignment.errors import ConfigurationError
from attention_alignment.paths import PathPolicy


def test_workspace_config_has_all_sessions():
    config = load_config(Path("configs/sessions.template.yaml"))
    assert len(config.sessions) == 1
    assert set(config.conditions) == {500, 1000}
    assert config.condition(500).expected_stripe_s == 1.0
    assert config.condition(1000).expected_stripe_s == 2.0
    assert config.output_root.parent == config.project_root


def test_path_policy_rejects_parent_escape(tmp_path: Path):
    policy = PathPolicy(tmp_path / "outputs")
    assert policy.resolve_output("session", "qc.json").parent.name == "session"
    with pytest.raises(ConfigurationError):
        policy.resolve_output("..", "outside.json")
