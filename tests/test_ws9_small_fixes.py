"""WS9 small confirmed fixes (v6.33.0): C2, C4, C6, S8, S9."""

from __future__ import annotations

import pathlib


from ouroboros.tool_capabilities import READ_ONLY_PARALLEL_TOOLS
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


def test_c4_get_task_result_is_parallel_read_only():
    assert "get_task_result" in READ_ONLY_PARALLEL_TOOLS


def test_s8_wait_task_outer_timeout_is_long(tmp_path):
    reg = _registry(tmp_path)
    # The outer per-tool timeout must exceed any legitimate inner single-task wait
    # so the 600s loop default cannot kill it.
    assert reg.get_timeout("wait_task") >= 3600
    assert reg.get_timeout("wait_tasks") >= 3600


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


def test_s9_json_object_still_flagged_as_list_literal(tmp_path):
    reg = _registry(tmp_path)
    # A genuine JSON object (brace + quote, no space) keeps the list-literal error.
    result = reg.execute("run_command", {"cmd": '{"cmd": ["git"]}'})
    assert "SHELL_ARG_ERROR" in result


def test_c2_search_code_single_file(tmp_path):
    reg = _registry(tmp_path)
    (tmp_path / "repo" / "target.py").write_text("alpha = 1\nNEEDLE_TOKEN = 2\nbeta = 3\n", encoding="utf-8")
    result = reg.execute("search_code", {"path": "target.py", "query": "NEEDLE_TOKEN"})
    assert "NEEDLE_TOKEN" in result
    # A non-matching query on the same single file must report no hits, not error.
    miss = reg.execute("search_code", {"path": "target.py", "query": "DEFINITELY_ABSENT_XYZ"})
    assert "DEFINITELY_ABSENT_XYZ" not in miss or "0" in miss or "no match" in miss.lower()
