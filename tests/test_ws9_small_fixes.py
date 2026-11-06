"""WS9 small confirmed fixes (v6.33.0): C2, C4, C6, S8, S9."""

from __future__ import annotations

import pathlib


from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.search import _is_timeout_error


def _registry(tmp_path: pathlib.Path) -> ToolRegistry:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    reg = ToolRegistry(repo_dir=repo, drive_root=data)
    reg.set_context(ToolContext(repo_dir=repo, drive_root=data, task_metadata={"work_kind": "owner"}))  # an owner task
    return reg


def test_c6_is_timeout_error_classifies_only_timeouts():
    assert _is_timeout_error(TimeoutError("x")) is True

    class APITimeoutError(Exception):
        pass

    assert _is_timeout_error(APITimeoutError("slow")) is True
    assert _is_timeout_error(ValueError("nope")) is False


def test_s9_brace_group_points_at_sh_c(tmp_path):
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": "{ echo hi; }"})
    assert "SHELL_CMD_ERROR" in result
    assert "sh" in result and "-c" in result
    # Not the misleading "malformed JSON list" error.
    assert "JSON/Python list literal" not in result
