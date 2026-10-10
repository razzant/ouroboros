"""#1574: a tool argument named like a Pause forwarder's own parameter reaches the tool.

``run_tool_handler``/``run_operation``/``submit_async_operation`` take ``source`` and
``function`` positionally, so a public ``source`` (``promote_chat_to_task``, the
review tools) or ``function`` keyword belongs to the callee. Each case enters at
the registry (or the loop's sticky handoff), never at the handler, and the Pause
fence, real argument errors and handler exceptions keep their meaning.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros import owner_pause
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolEntry, ToolRegistry

GIT_URL = "https://github.com/razzant/ouroboros.git"


@pytest.fixture
def probe(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    calls = []

    def handler(ctx, source="", function="", note=""):
        calls.append({"ctx": ctx, "source": source, "function": function, "note": note,
                      "thread": threading.get_ident()})
        if note == "raise":
            raise TypeError("the body's own TypeError")
        return f"probe source={source} function={function} note={note}"

    params = {key: {"type": "string", "default": ""} for key in ("source", "function", "note")}
    registry.register(ToolEntry("forwarding_probe", {
        "name": "forwarding_probe", "description": "fixture",
        "parameters": {"type": "object", "properties": params, "required": []},
    }, handler))
    return registry, calls


def _claims(tmp_path):
    return (load_task_result(tmp_path, "root") or {}).get("launch_handoffs") or {}


@pytest.mark.parametrize("args", [
    {"source": GIT_URL}, {"function": "f"}, {"source": GIT_URL, "function": "f", "note": "n"}, {"note": "n"}, {}])
def test_registry_dispatch_forwards_every_public_keyword_to_the_tool(tmp_path, probe, args):
    registry, calls = probe

    result = registry.execute_result("forwarding_probe", dict(args))

    assert (result.status, result.code) == ("ok", "OK"), result.text
    expected = {"source": "", "function": "", "note": "", **args}
    assert result.text == "probe source={source} function={function} note={note}".format(**expected)
    (call,) = calls
    assert call["ctx"] is registry._ctx and {k: call[k] for k in expected} == expected
    assert call["thread"] != threading.get_ident()  # the not-handed branch: run_operation's worker
    assert _claims(tmp_path) == {}  # the invocation settled through the same forwarder


def test_sticky_handoff_branch_forwards_source_and_keeps_executor_affinity(tmp_path, probe):
    registry, calls = probe
    with ThreadPoolExecutor(max_workers=1) as executor:
        identity = executor.submit(threading.get_ident).result()
        future = owner_pause.submit_tool(registry._ctx, "forwarding_probe", executor.submit,
                                         registry.execute_result, "forwarding_probe",
                                         {"source": GIT_URL, "function": "f"})
        result = future.result()
    assert (result.status, result.code) == ("ok", "OK"), result.text
    assert [(c["source"], c["function"], c["thread"]) for c in calls] == [(GIT_URL, "f", identity)]
    assert _claims(tmp_path) == {}


def test_pause_fence_still_refuses_before_the_body(tmp_path, probe):
    registry, calls = probe
    owner_pause.install_fence(tmp_path, "root", request_id="accepted")

    result = registry.execute_result("forwarding_probe", {"source": GIT_URL})

    assert result.meta["owner_pause_not_started"], result
    assert calls == []


def test_argument_and_body_errors_keep_their_meaning(probe):
    registry, calls = probe

    unknown = registry.execute_result("forwarding_probe", {"source": GIT_URL, "unexpected": 1})
    assert unknown.status == "error" and calls == [], unknown
    raised = registry.execute_result("forwarding_probe", {"source": GIT_URL, "note": "raise"})
    assert raised.status == "error" and "the body's own TypeError" in raised.text, raised
    assert len(calls) == 1  # the body ran once; nothing was retried


def test_promote_chat_to_task_receives_its_public_source(tmp_path, probe, monkeypatch):
    from ouroboros.tools import control_routing

    registry, _calls = probe
    events = []
    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing",
                        lambda _ctx, evt: events.append(dict(evt)) or ("live", {"status": "scheduled"}))

    result = registry.execute_result("promote_chat_to_task", {"objective": "Fix the clone", "source": GIT_URL,
                                                             "predecessor_task_id": ""})

    assert "got multiple values" not in result.text, result.text
    assert "accepted and durably scheduled" in result.text, result.text
    assert [evt["source"] for evt in events] == [GIT_URL]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_extension_call_args_named_source_and_function_reach_the_extension(tmp_path, monkeypatch, asynchronous):
    from tests.test_batch4_extension_completion import _local

    received = []

    def sync(ctx, source="", function=""):
        received.append((source, function))
        return '{"ok": true}'

    async def async_handler(ctx, source="", function=""):
        received.append((source, function))
        return '{"ok": true}'

    registry, _queue, _workers, name = _local(tmp_path, monkeypatch, async_handler if asynchronous else sync)

    result = registry.execute_result(name, {"source": GIT_URL, "function": "f"})

    assert result.status == "ok", result.text
    assert received == [(GIT_URL, "f")]


def test_async_operation_forwards_keywords_and_still_refuses_after_pause(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    source = SimpleNamespace(drive_root=tmp_path, task_id="root", root_task_id="root")
    entered = []

    async def body(*, source, function):
        entered.append((source, function))
        return "done"

    async def call():
        return await owner_pause.submit_async_operation(source, body, source=GIT_URL, function="f")

    assert asyncio.run(call()) == "done" and entered == [(GIT_URL, "f")]
    owner_pause.install_fence(tmp_path, "root", request_id="accepted")
    with pytest.raises(owner_pause.OwnerPauseRefused):
        asyncio.run(call())
    assert len(entered) == 1
