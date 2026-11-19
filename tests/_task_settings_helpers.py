"""A sealed settings install and its owner writer for route and snapshot tests."""

import pytest

from ouroboros import config
from tests._socket_seal import block_network_except_socketpair


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    # Env alone does not rebind already imported supervisor writers.
    from supervisor import git_ops, queue, state, workers

    root = tmp_path / "app"
    data, repo = root / "data", root / "repo"
    data.mkdir(parents=True)
    repo.mkdir()
    for name, value in (("APP_ROOT", root), ("DATA_DIR", data),
                        ("REPO_DIR", repo), ("SETTINGS_PATH", data / "settings.json"),
                        ("HOME", root / "home")):
        monkeypatch.setattr(config, name, value)
    for key in set(config.SETTINGS_DEFAULTS) | set(config.RETIRED_COMMA_LIST_SETTING_KEYS):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv(config.SETTINGS_INTEGRITY_ENV, raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    monkeypatch.setenv("OUROBOROS_APP_ROOT", str(root))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(config.SETTINGS_PATH))
    monkeypatch.setenv("OUROBOROS_REPO_DIR", str(repo))
    state.init(data, 200.0)
    queue.init(data)
    for module in (git_ops, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", data)
        monkeypatch.setattr(module, "REPO_DIR", repo)

    def no_network(*args, **kwargs):
        raise AssertionError("Snapshot regressions must never contact a provider")

    block_network_except_socketpair(monkeypatch, no_network)
    yield data


def save_owner(values):
    from ouroboros.gateway.owner_settings import _owner_write_settings

    _owner_write_settings(values, authored_keys=tuple(values),
                          allow_context_lowering=True, allow_safety_lowering=True)
