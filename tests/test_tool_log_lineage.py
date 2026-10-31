"""M2: every ``tools.jsonl`` row of a task carries ``root_task_id`` and
``delegation_role`` through the ONE lineage resolver
(``task_results.resolve_task_lineage``; a direct root is its own root), and the
``proactive_message`` event row carries the ``task_id`` that makes it visible in
the task's own log stream (``gateway/logs.py`` filters rows by task/parent/root
id). A child scheduled from a direct turn is linked by its own recorded
parent/root ids; the retired ``schedule_task_from_direct_chat`` duplicate-work
warning is not written (one-bubble D16). Rows that belong to no task stay
exactly as they were.
"""
from __future__ import annotations

import json
import pathlib
import queue
import types

import pytest

from ouroboros.loop_tool_execution import _append_tool_log


def _rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tools(meta: dict | None, **attrs) -> types.SimpleNamespace:
    ctx = types.SimpleNamespace(task_metadata=meta if meta is not None else {}, **attrs)
    return types.SimpleNamespace(_ctx=ctx)


@pytest.mark.parametrize("shape,meta,expected", [
    ("direct_root", {}, {"root_task_id": "t-self", "delegation_role": "root"}),
    ("pooled_root", {"delegation_role": "root", "root_task_id": "t-self"},
     {"root_task_id": "t-self", "delegation_role": "root"}),
    ("child", {"parent_task_id": "t-parent", "root_task_id": "t-root", "delegation_role": "subagent"},
     {"root_task_id": "t-root", "delegation_role": "subagent", "parent_task_id": "t-parent"}),
    ("headless_child", {"parent_task_id": "t-parent", "root_task_id": "t-root", "delegation_role": "subagent",
                        "headless_child_drive_root": "/elsewhere", "task_depth": 2},
     {"root_task_id": "t-root", "delegation_role": "subagent", "parent_task_id": "t-parent", "task_depth": 2}),
])
def test_every_task_row_carries_lineage_from_the_one_resolver(tmp_path, shape, meta, expected):
    logs = tmp_path / "logs"
    logs.mkdir()
    _append_tool_log(_tools(meta), logs, {"type": "tool_call", "tool": "read_file", "task_id": "t-self"})
    [row] = _rows(logs / "tools.jsonl")
    for key, value in expected.items():
        assert row[key] == value, (shape, key, row)
    if "parent_task_id" not in expected:
        assert "parent_task_id" not in row, (shape, row)  # a root invents no parent


def test_budget_root_mirror_row_carries_the_same_lineage(tmp_path):
    logs = tmp_path / "child" / "logs"
    logs.mkdir(parents=True)
    budget = tmp_path / "budget"
    meta = {"parent_task_id": "t-parent", "root_task_id": "t-root", "delegation_role": "subagent",
            "budget_drive_root": str(budget)}
    _append_tool_log(_tools(meta), logs, {"type": "tool_call", "tool": "run_command", "task_id": "t-child"})
    for path in (logs / "tools.jsonl", budget / "logs" / "tools.jsonl"):
        [row] = _rows(path)
        assert row["root_task_id"] == "t-root" and row["delegation_role"] == "subagent", path


def test_rows_without_a_task_stay_as_today(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    _append_tool_log(_tools({}), logs, {"type": "tool_call", "tool": "read_file", "task_id": ""})
    [row] = _rows(logs / "tools.jsonl")
    assert "root_task_id" not in row and "delegation_role" not in row, row


class _Queue:
    def __init__(self):
        self.items: list = []

    def put_nowait(self, item):
        self.items.append(item)

    put = put_nowait


def test_proactive_message_row_carries_the_task_id(tmp_path):
    from ouroboros.tools.control import _send_user_message

    ctx = types.SimpleNamespace(
        current_chat_id=7, pending_events=[], drive_root=None, task_id="t-live",
        task_metadata={"root_task_id": "t-live"}, event_queue=_Queue(),
        drive_logs=lambda: tmp_path,
    )
    assert "OK" in _send_user_message(ctx, "a word while I work", reason="progress")
    [row] = [r for r in _rows(tmp_path / "events.jsonl") if r["type"] == "proactive_message"]
    assert row["task_id"] == "t-live", row
    assert row["reason"] == "progress" and row["text_preview"] == "a word while I work"


def test_schedule_from_direct_chat_links_the_child_without_the_retired_warning(tmp_path, monkeypatch):
    import re

    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    subagent_id = configure_test_subagent(monkeypatch)
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    (drive / "logs").mkdir(parents=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="direct-1", is_direct_chat=True,
                      current_chat_id=5, event_queue=queue.Queue())
    out = _schedule_task(ctx, subagent_id=subagent_id, objective="scout X", expected_output="Y")
    [child_id] = re.findall(r"Subagent request queued (\w+):", out)
    child = load_task_result(drive, child_id)
    # The direct turn's own log stream reaches the child through its recorded lineage.
    assert (child["parent_task_id"], child["root_task_id"]) == ("direct-1", "direct-1"), child
    events = drive / "logs" / "events.jsonl"
    assert not [r for r in (_rows(events) if events.exists() else []) if r["type"] == "schedule_task_from_direct_chat"]
