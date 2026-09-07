"""An ordinary parent names concrete task targets; descendants only narrow them."""
from __future__ import annotations

import json
import queue

import pytest

from ouroboros.contracts.task_contract import build_task_contract, normalize_allowed_origins, normalize_browser_origin
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.control import _schedule_task
from ouroboros.tools.control_subagent_spec import _validated_schedule_fields
from ouroboros.tools.registry import ToolContext
from supervisor.task_dispatch import build_scheduled_task_payload
from tests._shared import configure_test_subagent


@pytest.mark.parametrize("source,expected", [
    ("HTTP://Example.test./", "http://example.test:80"),
    ("https://bücher.test", "https://xn--bcher-kva.test:443"),
    ("http://[::1]:5173", "http://[::1]:5173"),
    ("http://192.168.1.3:5173", "http://192.168.1.3:5173"),
    ("http://*.test", ""), ("http://user:pass@host", ""),
    ("file:///tmp", ""), ("http://host:0", ""), ("http://host:99999", ""),
    ("http://host/path", ""), ("http://host?secret=value", ""),
])
def test_only_concrete_origins_normalize(source, expected):
    assert normalize_browser_origin(source, origin_only=True) == expected


def test_origin_contract_normalization_keeps_other_resource_policy():
    policy = {"allowed_origins": ["http://dev.test", "http://dev.test:80/", "*"], "unrelated": {"x": 1}}
    result = build_task_contract({"resource_policy": policy})
    assert result["resource_policy"] == {"allowed_origins": ["http://dev.test:80"], "unrelated": {"x": 1}}
    assert policy["allowed_origins"][-1] == "*"
    assert normalize_allowed_origins("http://dev.test") == []


def test_main_schedule_persists_target_and_child_can_only_inherit_or_narrow(tmp_path, monkeypatch):
    subagent = configure_test_subagent(monkeypatch)
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")
    canonical = tmp_path / "data"
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=canonical, is_direct_chat=True)
    ctx.task_id, ctx.task_depth, ctx.current_chat_id = "main", 0, 1
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {"root_task_id": "main"}
    origins = ["http://192.168.1.20:5173", "http://dev.test:5173"]
    answer = _schedule_task(ctx, subagent_id=subagent, objective="Verify the assigned UI",
                            expected_output="Screenshot and findings", allowed_origins=origins)
    assert "error" not in answer.lower(), answer
    event = ctx.event_queue.get_nowait()
    saved = json.loads((canonical / "task_results" / f"{event['task_id']}.json").read_text())
    contract = event["task_contract"]
    assert saved["task_contract"]["resource_policy"]["allowed_origins"] == origins
    assert contract["source"] == "parent_delegation"
    task = build_scheduled_task_payload({**event, "tid": event["task_id"], "parent_id": "main"})
    child = ToolContext(repo_dir=ctx.repo_dir, drive_root=canonical, task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    child.task_metadata, child.task_contract = task["metadata"], task["task_contract"]
    request = {"objective": "Inspect", "expected_output": "Findings"}
    inherited, error = _validated_schedule_fields(request, ctx=child)
    assert not error and inherited["resource_policy"]["allowed_origins"] == origins
    for selected in ([], origins[:1]):
        narrowed, error = _validated_schedule_fields({**request, "allowed_origins": selected}, ctx=child)
        assert not error and narrowed["resource_policy"]["allowed_origins"] == selected
    _, error = _validated_schedule_fields({**request, "allowed_origins": ["http://192.168.1.21:5173"]}, ctx=child)
    assert "BROWSER_ORIGIN_NOT_GRANTED" in error
    assert child.task_contract["resource_policy"]["allowed_origins"] == origins
    child.task_id, child.task_depth, child.current_chat_id = event["task_id"], 1, 1
    child.event_queue = queue.Queue()
    answer = _schedule_task(child, subagent_id=subagent, **request, allowed_origins=origins[:1])
    assert "error" not in answer.lower(), answer
    grandchild = child.event_queue.get_nowait()
    saved = json.loads((canonical / "task_results" / f"{grandchild['task_id']}.json").read_text())
    assert saved["task_contract"]["resource_policy"]["allowed_origins"] == origins[:1]
    assert grandchild["task_contract"]["resource_policy"]["allowed_origins"] == origins[:1]
    assert child.task_contract["resource_policy"]["allowed_origins"] == origins


def test_external_presence_cannot_mint_origin_from_an_event(tmp_path):
    from ouroboros.presence_authority import PresenceCapabilityCeiling, presence_ceiling_payload

    ceiling = PresenceCapabilityCeiling("helper", "a" * 64, "b" * 64, "c" * 64,
                                        "d" * 64, "main", 10, (), (), "0" * 64)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, is_direct_chat=True)
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    _, error = _validated_schedule_fields({"objective": "Open a page", "expected_output": "Screenshot",
                                           "allowed_origins": ["http://10.1.2.3:5173"]}, ctx=ctx)
    assert "BROWSER_ORIGIN_NOT_GRANTED" in error
