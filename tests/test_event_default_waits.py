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


@pytest.mark.parametrize("ids", [["one", "phantom"], ["phantom"]])
def test_registry_default_unknown_set_returns_repair_without_holding_slot(tmp_path, monkeypatch, ids):
    from ouroboros.tools.registry import ToolRegistry
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = "root-1", 1
    ctx.owner_wait_callback = lambda *_: None
    write_task_result(tmp_path, "one", "running", parent_task_id=ctx.task_id, root_task_id=ctx.task_id)
    from ouroboros.utils import atomic_write_json, utc_now_iso
    atomic_write_json(tmp_path / "state" / "queue_snapshot.json",
                      {"ts": utc_now_iso(), "pending": [], "running": []})
    real_wait = waits.wait_for_effective_tasks
    windows = []
    def snapshot(*args, **kwargs):
        windows.append(kwargs["timeout_sec"])
        assert kwargs["timeout_sec"] == 0, "unknown IDs must not divert a loop wait into a bounded slot hold"
        return real_wait(*args, **kwargs)
    monkeypatch.setattr(waits, "wait_for_effective_tasks", snapshot)
    raw = registry.execute("wait_tasks", {"task_ids": ids})
    assert raw.startswith("{"), raw
    view = json.loads(raw)
    assert windows == [0]
    assert view["unknown_task_ids"] == ["phantom"]
    assert view["tasks"]["phantom"]["unknown_task_id"] is True
    assert view["wait_short_circuited"]["reason"] == "unknown_task_ids_require_repair"
    assert not getattr(ctx, "_model_sleep", None)
    # Repairing the set takes the surviving positive path: genuine warm sleep,
    # with no timer, until the real child or an actionable event is ready.
    repaired = json.loads(registry.execute("wait_tasks", {"task_ids": ["one"]}))
    assert repaired["reason"] == "sleep_armed"
    assert ctx._model_sleep["tasks"] == ["one"] and ctx._model_sleep["wake_at"] == ""


def test_default_wait_sleep_validation_is_a_typed_registry_error(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = "root-1", 1
    ctx.owner_wait_callback = lambda *_: None
    write_task_result(tmp_path, "one", "running", parent_task_id=ctx.task_id)
    def refuse(*_a, **_kw):
        raise ValueError("selected task vanished before sleep")
    monkeypatch.setattr(model_sleep, "selectors", refuse)
    assert "TOOL_ARG_ERROR (wait_tasks)" in registry.execute("wait_tasks", {"task_ids": ["one"]})


@pytest.mark.parametrize("mode", ["warm", "cold", "in_slot"])
def test_zero_snapshot_observes_reply_and_owner_without_sleep_or_ack(tmp_path, monkeypatch, mode):
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.tools.registry import ToolRegistry
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id = "t-wait"  # standalone observation has no continuation owner/attempt
    seed(tmp_path, "peer")
    monkeypatch.setattr(model_sleep, "request_sleep", lambda *_a, **_kw: pytest.fail("snapshot must not arm"))
    args = {"timeout_sec": 0, "mode": mode, "senders": ["peer"]}
    before = json.loads(registry.execute("await_messages", args))
    assert before["reason"] == "snapshot" and not before["ready"]
    assert write_task_message(tmp_path, "complete peer reply", ctx.task_id,
                              source_task_id="peer", provenance="peer_task", msg_id="zero-peer")
    ready = json.loads(registry.execute("await_messages", args))
    assert ready["ready"] and ready["woke_by"] == "mail:peer"
    assert not json.loads(registry.execute("await_messages", {"timeout_sec": 0, "mode": mode}))["ready"]
    assert write_owner_message(tmp_path, "owner decision", ctx.task_id, msg_id="zero-owner")
    owner = json.loads(registry.execute("await_messages", {"timeout_sec": 0, "mode": mode}))
    assert owner["ready"] and owner["woke_by"] == "owner_text"
    assert acknowledged_task_message_ids(tmp_path, ctx.task_id) == set()
    assert not getattr(ctx, "_loop_mailbox_seen_ids", set())
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_owner_wait_requested", None)


def test_zero_snapshot_retains_terminal_digest_and_does_not_consume_beacon(tmp_path):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.task_tree import _tree_note
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id = "t-wait"
    ctx.task_metadata = {"root_task_id": "t-wait"}
    seed(tmp_path, "terminal", "completed", result="terminal full result")
    terminal = json.loads(registry.execute("await_messages", {"timeout_sec": 0, "tasks": ["terminal"]}))
    assert terminal["woke_by"] == "task:terminal:completed"
    assert terminal["tasks"]["terminal"]["result"] == "terminal full result"
    assert len(terminal["tasks"]["terminal"]["child_result_sha256"]) == 64
    child = SimpleNamespace(task_id="active", drive_root=tmp_path, budget_drive_root=tmp_path,
                            task_metadata={"root_task_id": "t-wait"})
    seed(tmp_path, "active")
    assert _tree_note(child, "question", "decision required").startswith("OK")
    args = {"timeout_sec": 0, "tasks": ["active"]}
    first = json.loads(registry.execute("await_messages", args))
    second = json.loads(registry.execute("await_messages", args))
    assert first["woke_by"] == second["woke_by"] == "child_attention_beacon"
    assert first["wake_beacons"] == second["wake_beacons"]
    assert not getattr(ctx, "_wait_attention_cursors", None)
    assert waits._wait_attention_poll(ctx, "", ["active"])({}, {})["beacons"][0]["text"] == "decision required"
    assert waits._wait_attention_poll(ctx, "", ["active"])({}, {}) is None
    assert "TOOL_ARG_ERROR" in registry.execute("await_messages", {"timeout_sec": 0, "senders": ["missing"]})


@pytest.mark.parametrize("selector", ["tasks", "senders"])
@pytest.mark.parametrize("same_project", [True, False])
def test_queue_only_peer_uses_the_admitted_project_lease(tmp_path, monkeypatch, selector, same_project):
    from ouroboros.project_lease import candidate_is_leasable, running_project_ids
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolRegistry
    from tests._budget_pause_exact_helpers import _install_queue

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    peer = queue.enqueue_task({"id": "queued-peer", "type": "task", "chat_id": 0,
        "root_task_id": "queued-peer", "project_id": "lane" if same_project else "other-lane"})
    assert not peer.get("_admission_blocked"), peer
    workers.RUNNING["sleeper"] = {"task": {"id": "sleeper", "project_id": "lane"}}
    assert queue.persist_queue_snapshot(reason="queue-only-lease-regression")
    assert load_task_result(tmp_path, "queued-peer", strict=True) is None
    assert candidate_is_leasable(peer, running_project_ids(workers.RUNNING.values())) is (not same_project)
    write_task_result(tmp_path, "sleeper", "running", project_id="lane", root_task_id="sleeper")
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id = ctx.root_task_id = "sleeper"
    ctx.task_attempt = 1
    ctx.project_id = "lane"
    ctx.owner_wait_callback = lambda *_a, **_kw: "unknown"
    out = registry.execute("await_messages", {selector: ["queued-peer"]})
    if same_project:
        assert "project lease" in out
        assert not getattr(ctx, "_model_sleep", None)
        assert not getattr(ctx, "_owner_wait_requested", None)
    else:
        assert json.loads(out)["reason"] == "sleep_armed"


@pytest.mark.parametrize("selector", ["tasks", "senders"])
def test_exact_paused_peer_cannot_hide_its_project_lease_dependency(tmp_path, monkeypatch, selector):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolRegistry
    from tests._budget_pause_exact_helpers import _install_queue, _parked

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _parked(tmp_path, monkeypatch, task_id="paused-peer", extra={"project_id": "lane"})
    workers.RUNNING["sleeper"] = {"task": {"id": "sleeper", "project_id": "lane"}}
    assert queue.persist_queue_snapshot(reason="paused-peer-lease-regression")
    assert load_task_result(tmp_path, "paused-peer", strict=True)["status"] == "running"
    write_task_result(tmp_path, "sleeper", "running", project_id="lane", root_task_id="sleeper")
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id = ctx.root_task_id = "sleeper"
    ctx.task_attempt = 1
    ctx.project_id = "lane"
    ctx.owner_wait_callback = lambda *_a, **_kw: "unknown"
    assert "project lease" in registry.execute("await_messages", {selector: ["paused-peer"]})
    assert not getattr(ctx, "_model_sleep", None)


@pytest.mark.parametrize("selector", ["tasks", "senders"])
@pytest.mark.parametrize("same_tree", [True, False])
def test_cold_queue_only_dependency_respects_the_tree_launch_fence(tmp_path, monkeypatch, selector, same_tree):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolRegistry
    from tests._budget_pause_exact_helpers import _install_queue

    queue, _, _workers = _install_queue(tmp_path, monkeypatch)
    peer = queue.enqueue_task({"id": "queued-dependency", "type": "task", "chat_id": 0,
        "root_task_id": "sleeper" if same_tree else "independent",
        "parent_task_id": "sleeper" if same_tree else "independent", "delegation_role": "subagent"})
    assert not peer.get("_admission_blocked"), peer
    assert queue.persist_queue_snapshot(reason="cold-queue-dependency-regression")
    assert load_task_result(tmp_path, "queued-dependency", strict=True) is None
    write_task_result(tmp_path, "sleeper", "running", root_task_id="sleeper")
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id = ctx.root_task_id = "sleeper"
    ctx.task_attempt = 1
    ctx.owner_wait_callback = lambda *_a, **_kw: "unknown"
    out = registry.execute("await_messages", {"mode": "cold", selector: ["queued-dependency"]})
    if same_tree:
        assert "queued_member queued-dependency" in out
        assert not getattr(ctx, "_model_sleep", None)
    else:
        assert json.loads(out)["reason"] == "sleep_armed"


def test_queue_only_named_child_expands_and_wakes_for_its_live_sibling(tmp_path, monkeypatch):
    from ouroboros.owner_wait import wait_after_tools
    from ouroboros.tools.registry import ToolRegistry
    from tests._budget_pause_exact_helpers import _install_queue

    queue, _, _workers = _install_queue(tmp_path, monkeypatch)
    ctx = native_context(tmp_path)
    named = queue.enqueue_task({"id": "queued-named", "type": "task", "chat_id": 0,
        "parent_task_id": ctx.task_id, "root_task_id": ctx.task_id, "delegation_role": "subagent"})
    foreign = queue.enqueue_task({"id": "queued-foreign", "type": "task", "chat_id": 0,
        "parent_task_id": "foreign-root", "root_task_id": "foreign-root", "delegation_role": "subagent"})
    assert not named.get("_admission_blocked") and not foreign.get("_admission_blocked")
    assert queue.persist_queue_snapshot(reason="queue-only-sibling-regression")
    write_task_result(tmp_path, "live-sibling", "running", parent_task_id=ctx.task_id,
        root_task_id=ctx.task_id, delegation_role="subagent")
    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    registry._ctx.__dict__.update(vars(ctx))
    ctx = registry._ctx
    ctx.model_wait_context.tool_context = ctx
    foreign_view = json.loads(registry.execute("wait_task", {"task_id": "queued-foreign", "timeout_sec": 0}))
    assert set(foreign_view["tasks"]) == {"queued-foreign"}
    armed = json.loads(registry.execute("wait_task", {"task_id": "queued-named"}))
    assert armed["reason"] == "sleep_armed"
    assert set(ctx._model_sleep["tasks"]) == {"queued-named", "live-sibling"}
    calls = []
    def terminal(_seconds):
        calls.append(1)
        assert len(calls) == 1
        write_task_result(tmp_path, "live-sibling", "completed", result="sibling first",
            parent_task_id=ctx.task_id, root_task_id=ctx.task_id, delegation_role="subagent")
    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", terminal)
    messages = [{"role": "tool", "tool_call_id": "single", "content": json.dumps(armed)}]
    wait_after_tools(ctx, messages, {"tool_calls": []}, {}, 1, [], set())
    assert len(calls) == 1
    assert "task live-sibling reaching completed" in messages[-1]["content"]


def test_ledger_only_default_wait_returns_full_set_repair_not_selector_error(tmp_path, monkeypatch):
    from ouroboros.task_tree_ledger import tree_ledger_append
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.utils import atomic_write_json, utc_now_iso

    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = "t-wait", 1
    ctx.task_metadata = {"root_task_id": ctx.task_id}
    ctx.owner_wait_callback = lambda *_a: None
    seed(tmp_path, "real")
    atomic_write_json(tmp_path / "state" / "queue_snapshot.json",
                      {"ts": utc_now_iso(), "pending": [], "running": []})
    assert tree_ledger_append("t-wait", "note", "historical mention", task_id="ledger-only",
                             data_root=tmp_path).startswith("OK")
    assert waits._unminted_wait_ids(ctx, tmp_path, ["ledger-only"]) == [], "minted is not wakeable"
    raw = registry.execute("wait_tasks", {"task_ids": ["real", "ledger-only"]})
    view = json.loads(raw)
    assert set(view["tasks"]) == {"real", "ledger-only"}
    assert view["unknown_task_ids"] == ["ledger-only"]
    assert "no waitable result/queue" in view["tasks"]["ledger-only"]["note"]
    assert view["wait_short_circuited"]["waited_sec"] < 0.1
    assert not getattr(ctx, "_model_sleep", None)
    assert json.loads(registry.execute("wait_tasks", {"task_ids": ["real"]}))["reason"] == "sleep_armed"


@pytest.mark.parametrize("selected", [{"tasks": ["child"]}, {"senders": ["child"]}])
def test_sleep_note_describes_task_only_vs_selected_mail(tmp_path, selected):
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt = "t-wait", 1
    ctx.owner_wait_callback = lambda *_a: None
    seed(tmp_path, "child")
    view = json.loads(registry.execute("await_messages", selected))
    assert view["reason"] == "sleep_armed"
    assert ("selected senders' mail wakes" in view["note"]) == bool(selected.get("senders"))
    assert "owner's messages and controls always" in view["note"]
    if selected.get("tasks"):
        assert "unselected mail stays unread" in view["note"]


def test_beacon_fifo_survives_durable_cold_continuation_without_snapshot_consumption(tmp_path, monkeypatch):
    from ouroboros import task_tree_ledger as ledger
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.owner_wait import continuation_state, restore_continuation_state, store_continuation_source

    monkeypatch.setattr(ledger, "utc_now_iso", lambda: "2026-10-10T00:00:00Z")
    ctx = native_context(tmp_path)
    ctx.task_metadata = {"root_task_id": ctx.task_id}
    for i in range(7):
        assert ledger.tree_ledger_append(ctx.task_id, "question", f"q-{i}", task_id="child",
                                        data_root=tmp_path).startswith("OK")
    first = waits._wait_attention_poll(ctx, "", ["child"])({}, {})
    assert [r["text"] for r in first["beacons"]] == [f"q-{i}" for i in range(5)]
    state = continuation_state(ctx, [{"role": "user", "content": "retained"}], {}, {}, 3, [], set())
    ref = store_continuation_source(ctx, state, "beacon-cold-test")
    restored_state = json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, ref))
    resumed = native_context(tmp_path)
    resumed.task_metadata = ctx.task_metadata
    restore_continuation_state(SimpleNamespace(_ctx=resumed), restored_state, [], {}, {}, set())
    snapshot = waits._wait_attention_poll(resumed, "", ["child"], consume=False)({}, {})
    assert [r["text"] for r in snapshot["beacons"]] == ["q-5", "q-6"]
    second = waits._wait_attention_poll(resumed, "", ["child"])({}, {})
    assert second["beacons"] == snapshot["beacons"]
    assert waits._wait_attention_poll(resumed, "", ["child"])({}, {}) is None
    assert ledger.tree_ledger_append(ctx.task_id, "question", "fresh", task_id="child",
                                    data_root=tmp_path).startswith("OK")
    fresh = waits._wait_attention_poll(resumed, "", ["child"])({}, {})
    assert [r["text"] for r in fresh["beacons"]] == ["fresh"]
