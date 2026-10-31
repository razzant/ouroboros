"""Test child HOME isolation without moving the parent live-data guard."""
import pathlib
import sys

import pytest

from tests.test_tool_api_v2_public_surface import _registry_under_fake_home


@pytest.mark.serial
@pytest.mark.parametrize("tool", ["run_command", "run_script"])
def test_fake_home_fixture_binds_the_actual_child_environment(tmp_path, monkeypatch, tool):
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    registry, _repo, data, _desktop = _registry_under_fake_home(tmp_path, monkeypatch)
    home = pathlib.Path.home()
    expected = {"HOME": str(home), "USERPROFILE": str(home),
                "OUROBOROS_DATA_DIR": str(data), "OUROBOROS_SETTINGS_PATH": str(data / "settings.json")}
    body = "import os\nfor key in " + repr(list(expected)) + ": print(key + '=' + os.environ[key])\n"
    args = {"cmd": [sys.executable, "-c", body]} if tool == "run_command" else {"script": body}
    result = registry.execute(tool, {**args, "cwd": "task_drive"})
    assert "exit_code=0" in result, result
    for key, value in expected.items():
        assert key + "=" + value in result, result


def test_fake_home_fixture_preserves_the_parent_live_data_guard(tmp_path, monkeypatch):
    import os
    from ouroboros.utils import assert_test_data_path

    parent_home = pathlib.Path(os.path.expanduser("~"))
    _registry_under_fake_home(tmp_path, monkeypatch)
    assert pathlib.Path(os.path.expanduser("~")) == parent_home
    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        assert_test_data_path(parent_home / "Ouroboros/data/settings.json")
