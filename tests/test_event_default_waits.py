"""Event-default waits through registry, native park, mailbox and tool-result consumers.

No provider call or private trace is involved. Full sources and checkpoint ACK
are checked independently of the compact observation shown to the model.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import model_sleep
from ouroboros.owner_mailbox import (
    TASK_ATTENTION_KINDS,
    acknowledged_task_message_ids,
    wait_message_requires_attention,
    write_task_message,
)
from ouroboros.task_results import write_task_result
from ouroboros.tools import control_task_results as waits
from tests.test_native_owner_wait import native_context


def seed(root, tid, status="running", **kw):
    write_task_result(root, tid, status, parent_task_id="t-wait", root_task_id="t-wait", delegation_role="subagent", **kw)


def test_registry_defaults_are_event_owned_and_explicit_snapshots_stay_available(tmp_path):
    from ouroboros.tools.registry import ToolRegistry
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    schema = {r["function"]["name"]: r["function"] for r in registry.schemas()}
    for name in ("wait_task", "wait_tasks", "await_messages"):
        assert "default" not in schema[name]["parameters"]["properties"]["timeout_sec"]
    assert schema["await_messages"]["parameters"]["properties"]["mode"]["default"] == "warm"
    assert registry.get_timeout("wait_tasks") > waits._event_wait_window(registry._ctx)
    assert "wait_tasks([id])" in schema["wait_task"]["description"]


@pytest.mark.parametrize("kind", TASK_ATTENTION_KINDS)
@pytest.mark.parametrize("provenance", ["ancestor_task", "peer_task", "descendant_task", "independent_task"])
def test_typed_attention_never_borrows_owner_authority(tmp_path, kind, provenance):
    ctx = SimpleNamespace(task_id="t-wait", drive_root=tmp_path, task_attempt=1,
                          _loop_mailbox_seen_ids=set(), task_metadata={})
    seed(tmp_path, "child")
    assert write_task_message(tmp_path, "a changed contract", "t-wait", source_task_id="child",
                              provenance=provenance, attention_kind=kind, msg_id="attention")
    chosen = model_sleep.selectors(ctx, tasks=["child"])
    assert model_sleep.wake_reason(ctx, chosen) == "mail:child"
    assert acknowledged_task_message_ids(tmp_path, "t-wait") == set()
    assert ctx._loop_mailbox_seen_ids == set()


def test_all_task_provenances_are_quiet_unless_selected_or_typed():
    for provenance in ("ancestor_task", "peer_task", "descendant_task", "independent_task", "system"):
        entry = {"kind": "task_message", "provenance": provenance, "source_task_id": "peer"}
        assert not wait_message_requires_attention(entry)
        assert wait_message_requires_attention(entry, ["peer"])
    assert wait_message_requires_attention({"kind": "owner_text"})
    assert wait_message_requires_attention({"kind": "finalize_now"})
    assert wait_message_requires_attention({"kind": "task_message", "provenance": "system",
                                           "review_feedback": {"panel": "p"}})


def test_live_child_snapshot_preserves_simultaneous_terminals_and_exact_escape(tmp_path):
    ctx = SimpleNamespace(task_id="t-wait", drive_root=tmp_path, task_attempt=1,
                          _loop_mailbox_seen_ids=set(), task_metadata={})
    seed(tmp_path, "named")
    seed(tmp_path, "sibling")
    seed(tmp_path, "old", "completed", result="already read")
    snapshot = json.loads(waits._wait_for_task(ctx, "named", timeout_sec=0))
    assert set(snapshot["tasks"]) == {"named", "sibling"}
    exact = json.loads(waits._wait_for_tasks(ctx, ["named"], timeout_sec=0))
    assert set(exact["tasks"]) == {"named"}
    # Both completing before the next observation must remain in that batch.
    seed(tmp_path, "named", "completed", result="first")
    seed(tmp_path, "sibling", "completed", result="second")
    observed = json.loads(waits._wait_for_tasks(ctx, list(snapshot["tasks"]), timeout_sec=0))
    assert observed["all_terminal"]
    assert {r["result"] for r in observed["tasks"].values()} == {"first", "second"}


def test_quiet_mail_survives_real_native_park_and_enters_same_resumed_request(tmp_path, monkeypatch):
    import queue

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_wait import wait_after_tools
    from ouroboros.working_checkpoint import flush_content_acks

    ctx = native_context(tmp_path)
    # native_context uses root-1, so bind the actual child to that parent.
    write_task_result(tmp_path, "one", "running", parent_task_id=ctx.task_id, root_task_id=ctx.task_id)
    armed = json.loads(waits._wait_for_tasks(ctx, ["one"]))
    assert armed["reason"] == "sleep_armed"
    assert ctx._model_sleep["wake_at"] == ""
    polls = []
    def events(_seconds):
        polls.append(1)
        if len(polls) == 1:
            assert write_task_message(tmp_path, "retain this full original", ctx.task_id,
                source_task_id="peer", provenance="peer_task", msg_id="quiet")
        elif len(polls) == 2:
            assert model_sleep.wake_reason(ctx, ctx._model_sleep) == ""
            write_task_result(tmp_path, "one", "completed", result="done", parent_task_id=ctx.task_id)
        else:
            pytest.fail("terminal should wake without another renewal")
    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", events)
    messages = [{"role": "tool", "tool_call_id": "wait", "content": json.dumps(armed)}]
    wait_after_tools(ctx, messages, {"tool_calls": []}, {}, 1, [], set())
    assert len(polls) == 2
    assert "quiet" in messages[-1]["content"]
    assert "retain this full original" not in messages[-1]["content"]
    assert acknowledged_task_message_ids(tmp_path, ctx.task_id) == set()
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, ctx.task_id, queue.Queue(),
        ctx._loop_mailbox_seen_ids, owner_ctx=ctx, defer_content_ack=True)
    assert "retain this full original" in json.dumps(messages)
    assert acknowledged_task_message_ids(tmp_path, ctx.task_id) == set()
    flush_content_acks(ctx)
    assert "quiet" in acknowledged_task_message_ids(tmp_path, ctx.task_id)


@pytest.mark.parametrize("status", ["running", "completed", "failed", "cancelled"])
def test_large_result_bounds_whole_wait_and_retains_exact_source(tmp_path, status):
    import hashlib
    ctx = SimpleNamespace(task_id="t-wait", drive_root=tmp_path, task_attempt=1,
                          _loop_mailbox_seen_ids=set(), task_metadata={})
    text = "🔎 result with a late important fact\n" * 2200 + "LAST_FACT"
    seed(tmp_path, "large", status, result=text, trace_summary="step\n" * 4000)
    response = waits._wait_for_tasks(ctx, ["large"], timeout_sec=0)
    assert len(response) <= 15_000
    view = json.loads(response)
    assert view["preview_only"] and view["complete_source"]
    ref = view["complete_source"]
    from ouroboros.artifacts import read_actor_source_bytes, task_artifact_dir_path
    path = task_artifact_dir_path(tmp_path, ctx.task_id) / ref["path"]
    data = read_actor_source_bytes(tmp_path, ctx.task_id, ref)
    assert hashlib.sha256(data).hexdigest() == ref["sha256"]
    full = json.loads(data)
    assert full["tasks"]["large"]["result"] == text
    known = view["tasks"]["large"]["child_result_sha256"]
    unchanged = json.loads(waits._wait_for_tasks(ctx, ["large"], timeout_sec=0,
        known_result_sha256_by_task={"large": known}))
    assert unchanged["tasks"]["large"]["result_unchanged"]
    assert "result" not in unchanged["tasks"]["large"]
    assert path.read_bytes() == data


def test_standalone_wait_envelope_is_strictly_inside_operation_lease(tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.runtime_limits import NESTED_SETTLEMENT_MARGIN_SEC, OPERATION_WINDOW_FALLBACK_SEC
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: None)
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="t-wait")
    assert waits._event_wait_window(ctx) == OPERATION_WINDOW_FALLBACK_SEC - NESTED_SETTLEMENT_MARGIN_SEC


def test_childless_sleep_requires_a_reply_source_and_missing_continuation_refuses(tmp_path):
    ctx = SimpleNamespace(task_id="t-wait", drive_root=tmp_path, task_attempt=1,
                          _loop_mailbox_seen_ids=set(), task_metadata={}, owner_wait_callback=lambda *_: None)
    assert "no live children or selected source" in waits._await_messages(ctx)
    seed(tmp_path, "peer")
    assert json.loads(waits._await_messages(ctx, senders=["peer"]))["reason"] == "sleep_armed"
    ctx.owner_wait_callback = None
    assert "no continuation owner" in waits._await_messages(ctx, senders=["peer"])


def test_delegate_information_does_not_wake_or_get_acknowledged(tmp_path):
    from ouroboros.delegate_supervision import supervised_wait
    ctx = SimpleNamespace(task_id="nanny", drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_metadata={"configured_subagent": {"config_fingerprint": "fp"}})
    assert write_task_message(tmp_path, "retain peer information", "nanny", source_task_id="peer",
                              provenance="peer_task", msg_id="quiet-delegate")
    polls = []
    def observed(_ctx, run, _seconds, _seq):
        polls.append(1)
        return json.dumps({"status": "completed" if len(polls) == 3 else "no_progress",
                           "run_id": run, "last_seq": len(polls)})
    out = json.loads(supervised_wait(ctx, "run-fixture", wait_once=observed).text)
    assert len(polls) == 3 and out["status"] == "completed"
    assert not [r for r in out.get("wake_events", []) if r.get("msg_id") == "quiet-delegate"]
    assert acknowledged_task_message_ids(tmp_path, "nanny") == set()


def test_delegate_content_ack_requires_successful_working_checkpoint(tmp_path, monkeypatch):
    from ouroboros.loop_tool_execution import process_tool_results
    from ouroboros.working_checkpoint import checkpoint_path, save_ready
    ctx = native_context(tmp_path)
    delivered = []
    monkeypatch.setattr("ouroboros.delegate_supervision.acknowledge_pending_wake",
                        lambda _ctx, body: delivered.append(body))
    messages = []
    trace = {"tool_calls": []}
    tools = SimpleNamespace(_ctx=ctx)
    process_tool_results([{"fn_name": "delegate_wait", "tool_call_id": "wake1",
        "result": json.dumps({"supervision_wake_id": "wake", "wake_events": [{"text": "full original"}]}),
        "is_error": False, "args_for_log": {}, "tool_args": {},
        "result_meta": {"tool_result_meta": {"supervision_wake_id": "wake"}}}],
        messages, trace, lambda *_a, **_kw: None, tools=tools)
    assert delivered == []
    limit = SimpleNamespace(tools=tools, messages=messages, llm_trace=trace,
                            accumulated_usage={}, round_idx=1, tool_schemas=[], owner_msg_seen=set())
    from ouroboros import utils
    real = utils.write_bytes_atomic
    monkeypatch.setattr(utils, "write_bytes_atomic", lambda *_a: (_ for _ in ()).throw(OSError("fixture disk failure")))
    save_ready(limit)
    assert delivered == [] and ctx._pending_content_acks
    monkeypatch.setattr(utils, "write_bytes_atomic", real)
    save_ready(limit)
    assert len(delivered) == 1 and "full original" in delivered[0]
    checkpoint = json.loads(checkpoint_path(tmp_path, ctx.task_id, 1).read_bytes())
    assert "full original" in json.dumps(checkpoint["messages"])
    save_ready(limit)
    assert len(delivered) == 1


def test_queue_only_child_can_arm_a_default_wait_without_result_file(tmp_path):
    from ouroboros.utils import atomic_write_json
    ctx = SimpleNamespace(task_id="t-wait", drive_root=tmp_path, task_attempt=1,
        _loop_mailbox_seen_ids=set(), task_metadata={}, owner_wait_callback=lambda *_: None)
    atomic_write_json(tmp_path / "state" / "queue_snapshot.json", {"ts": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "pending": [{"id": "queued", "task": {"id": "queued", "parent_task_id": "t-wait", "root_task_id": "t-wait", "delegation_role": "subagent"}}],
        "running": []})
    observed = json.loads(waits._wait_for_tasks(ctx, ["queued"]))
    assert observed["reason"] == "sleep_armed" and ctx._model_sleep["tasks"] == ["queued"]


def test_beacon_writer_and_wait_reader_share_canonical_root(tmp_path):
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.task_tree import _tree_note
    root, fork = tmp_path / "canonical", tmp_path / "fork"
    ctx = ToolContext(repo_dir=tmp_path, drive_root=fork)
    ctx.task_id, ctx.budget_drive_root = "child", root
    ctx.task_metadata = {"root_task_id": "root", "budget_drive_root": str(root)}
    assert _tree_note(ctx, "partial_finding", "quiet", needs_parent_attention=True).startswith("OK")
    parent = SimpleNamespace(task_id="root", drive_root=fork, budget_drive_root=root,
        task_metadata={"root_task_id": "root"}, task_attempt=1, _loop_mailbox_seen_ids=set())
    assert waits._wait_attention_poll(parent, "", ["child"])({}, {}) is None
    assert _tree_note(ctx, "question", "need a decision").startswith("OK")
    wake = waits._wait_attention_poll(parent, "", ["child"])({}, {})
    assert wake["reason"] == "child_attention_beacon"
    assert wake["beacons"][0]["text"] == "need a decision"


def test_retained_tail_is_readable_by_a_new_registry_after_producer_lifetime(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tools.registry import ToolRegistry
    producer = SimpleNamespace(task_id="parent1", drive_root=tmp_path, task_attempt=1,
        task_metadata={}, _loop_mailbox_seen_ids=set())
    seed(tmp_path, "tail-child", "completed", result="payload\n" * 10000 + "LAST_MEANINGFUL_FACT")
    view = json.loads(waits._wait_for_tasks(producer, ["tail-child"], timeout_sec=0))
    ref = view["complete_source"]
    full_text = read_actor_source_bytes(tmp_path, producer.task_id, ref).decode("utf-8")
    del producer
    reader = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    reader._ctx.task_id = "parent1"
    response = reader.execute("read_file", {"root": "artifact_store", "path": ref["path"],
        "start_char": full_text.index("LAST_MEANINGFUL_FACT") - 30, "max_lines": 1})
    assert "LAST_MEANINGFUL_FACT" in response


def test_explicit_zero_snapshot_and_invalid_mode_do_not_arm_sleep(tmp_path):
    ctx = SimpleNamespace(task_id="parent1", drive_root=tmp_path)
    for mode in ("warm", "cold", "in_slot"):
        assert json.loads(waits._await_messages(ctx, timeout_sec=0, mode=mode))["slept"] is False
    assert "TOOL_ARG_ERROR" in waits._await_messages(ctx, timeout_sec=0, mode="invalid")
    assert not getattr(ctx, "_model_sleep", None)
