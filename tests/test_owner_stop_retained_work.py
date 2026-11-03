"""#1564 A: retained child material reaches the existing final input/receipt, and the
owner reads every saved text whole through the task detail and its file route.

Offline fixtures only. The author does not execute this suite; the parent uses
safe_test in its external test root, with the existing test admission overlay.
"""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_collection as collection
from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_terminal as terminal
from ouroboros import owner_stop_receipt as osr
from ouroboros.cancel_intents import SCOPE_CASCADE, STOP_POLICY_FINALIZE, request_cancel
from ouroboros.delegate_output import delegated_output_source_projection
from ouroboros.task_finalization import prepare_terminal_send_event, terminal_host_notice_text
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import owner_stop
from tests.test_delegate_body_collection import started
from tests.test_delegate_body_collection import terminal as detail

CHILD_FIELD = "owner_stop_children"
# The terminal event's receipt field, spelled out so the suite also runs against an
# older owner_stop_receipt as a negative control (each test then fails on its own).
EVENT_FIELD = "terminal_retained_work"
MODEL_CALLS = ("get_task_result", "include_authority", "owner_stop_children", "Material facts")
RAW_CODES = ("ready_with_changes", "not_recorded", "observation_only", "retained_complete",
             "terminal_observation", "receipt_publication_unconfirmed")
DOOR = "Download “Work saved at Wrap up” from this task's Files for the full list and every saved text."


def _capture(root, task_id, stop=()):
    owner_stop._child_result_projection(SimpleNamespace(DRIVE_ROOT=root), task_id, stop=stop)
    return (load_task_result(root, task_id) or {}).get("delegate_terminal_reconciliation") or {}


def _child(root, tid="kid", **kwargs):
    write_task_result(root, tid, "cancelled", root_task_id="parent1", parent_task_id="parent1", **kwargs)


def _wrap_up(root, task_id="parent1"):
    intent = request_cancel(root, task_id, scope=SCOPE_CASCADE, requested_stop_policy=STOP_POLICY_FINALIZE)
    return intent, (task_id, intent["request_id"])


def _collect(root, monkeypatch, *, tid="kid", rid="run-one", state="cancelled", money=False, text="saved child answer"):
    from ouroboros import usage_accounting

    def record(*_args, **_kwargs):
        if not money:
            raise OSError("test accounting unavailable")

    monkeypatch.setattr(usage_accounting, "record_subscription_session", record)
    started(root, tid=tid, rid=rid, root_task_id="parent1")
    data = detail(rid, text=text)
    data["summary"].update(state=state, resumable={"carriers": ["session", "packet"]})
    collection.collect_terminal(root, rid, detail=data)
    return current.run(root, rid)


def _retained_long(root, *, tid="kid", rid="run-long", whole="The long answer begins, and it ends here.",
                   report="The long answer begins"):
    """A stopped run whose engine report is a bounded preview while its whole output was saved."""
    from ouroboros.delegate_output import retain_terminal_bytes

    entry = started(root, tid=tid, rid=rid, root_task_id="parent1")
    data = detail(rid, text=report)
    data["summary"]["state"] = "cancelled"
    data["primaryOutput"].update(bytes=len(whole.encode("utf-8")), truncated=True)
    collection.observe_run(root, entry, data)
    payload = json.dumps({"primary_output": {"text": whole}})
    artifact = retain_terminal_bytes(root, tid, rid, payload, payload.encode("utf-8"))
    custody.emit(root, custody.OUTPUT_SPILLED, {"run_id": rid, "task_id": tid,
        "artifact": artifact["source"]["path"], "source": artifact["source"], "staged": True,
        "full_content": True, "sha256": artifact["sha256"],
        "terminal_sha": current.run(root, rid).terminal_source["sha256"]})
    return current.run(root, rid)


def _client(root):
    """The real detail and file routes the Files row uses (``tests/test_task_file_serving._client``)."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.tasks import api_task_artifact, api_task_get

    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    app = Starlette(routes=[
        Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"]),
        Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET", "HEAD"]),
    ])
    app.state.drive_root = root
    return TestClient(app)


def _download(client, task_id="parent1", **kwargs):
    """The owner's door: the task detail's ``wrap_up_document``, addressed as ``taskWrapUpDocumentUrl`` builds it."""
    saved = client.get(f"/api/tasks/{task_id}").json()["wrap_up_document"]
    return saved, client.get(f"/api/tasks/{task_id}/artifacts/{saved['name']}",
                             params={"wrap_up": saved["request_id"]}, **kwargs)


def _document(root, task_id="parent1"):
    sink = io.BytesIO()
    osr.write_wrap_up_document(root, load_task_result(root, task_id), sink)
    return sink.getvalue().decode("utf-8")


def _captured_patch(root, tid="kid", rid="run-patch", status="ready_with_changes", state="cancelled", text="engine words"):
    entry = started(root, tid=tid, rid=rid, execution_root=str(root / "snapshot"), snapshot_id="snapshot")
    data = detail(rid, text=text)
    data["summary"]["state"] = state
    collection.observe_run(root, entry, data)
    cap = custody.delegated_capture_dir(root, tid, custody.capture_key(entry))
    cap.mkdir(parents=True, exist_ok=True)
    (cap / "workspace_patch.json").write_text(json.dumps({"status": status}))
    if status == "ready_with_changes":
        (cap / "workspace.patch").write_text("diff --git a/a b/a\n")
    custody.emit(root, custody.PATCH_CAPTURED, {"run_id": entry.run_id, "task_id": tid})
    return entry


def test_terminal_observed_before_money_reaches_prompt_and_published_sources(tmp_path, monkeypatch):
    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="child task answer")
    entry = _collect(tmp_path, monkeypatch)
    prompt = owner_stop._child_result_projection(SimpleNamespace(DRIVE_ROOT=tmp_path), "parent1")
    assert "saved child answer" in prompt and "run-one" in prompt
    assert "Output excerpt (the run's own words): saved child answer" in prompt
    assert "accounting not settled" in prompt and "cost unknown" in prompt
    assert "engine_capability_only" in prompt
    audit = load_task_result(tmp_path, "parent1")["delegate_terminal_reconciliation"]
    child = audit[CHILD_FIELD]["children"][0]
    run = child["custody"]["terminal_runs"][0]
    assert run["state"] == "cancelled" and run["staged_output"]["complete"] is True
    assert run["accounting"] == {"settled": False, "ledger_recorded": False,
                                  "cost_usd": None, "cost_final": False}
    assert not audit["open_run_ids"], "child accounting must not become the root's own execution debt"
    for name in ("staged_output", "terminal_observation"):
        source = run[name]["source_read"]["arguments"]
        assert source["task_id"] == "kid"
        stored = load_task_result(tmp_path, "kid")
        size = delegated_output_source_projection(tmp_path, stored, source["delegated_output_sha256"])
        page = delegated_output_source_projection(tmp_path, stored, source["delegated_output_sha256"],
                                                  0, size["complete_chars"])
        assert page["status"] == "available"
        assert "saved child answer" in page["text"]
        if name == "terminal_observation":
            # A complete observation is the exact engine read-back, never proof of the whole output.
            assert page["content_kind"] == "terminal_observation"
    assert not current.run(tmp_path, entry.run_id).output_consumed


def test_ordinary_terminal_audit_publishes_locators_without_reading_any_payload(tmp_path, monkeypatch):
    """F1: no Wrap up, a large retained output and a captured patch: every ordinary terminal
    audit stays read-free. Wrap up's one capture reads each observation once, stats the
    output instead of rereading it and starts, collects or pays for nothing."""
    from ouroboros import artifacts

    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="child task answer")
    _collect(tmp_path, monkeypatch, rid="run-out", text="x" * 200_000)
    _captured_patch(tmp_path)
    reads, captures = [], []
    real_read, real_capture = artifacts.read_actor_source_bytes, collection.capture_projection
    monkeypatch.setattr(artifacts, "read_actor_source_bytes",
                        lambda *a, **k: reads.append(a[2].get("path")) or real_read(*a, **k))
    monkeypatch.setattr(collection, "capture_projection",
                        lambda *a, **k: captures.append(a[1].run_id) or real_capture(*a, **k))
    audit = terminal.terminal_custody_disclosure(tmp_path, "kid")
    assert terminal.refresh_terminal_reconciliation(tmp_path, "kid")
    event = prepare_terminal_send_event(tmp_path, {"id": "kid", "chat_id": 1}, "answer", {"terminal_origin": "model_final"},
                                        {"type": "send_message", "task_id": "kid", "chat_id": 1, "text": "answer"},
                                        presence=False)
    assert reads == [] and captures == [], "an ordinary terminal audit never reads or hashes payloads"
    assert osr.EVENT_FIELD == EVENT_FIELD
    assert EVENT_FIELD not in event and CHILD_FIELD not in audit
    rows = {row["run_id"]: row for row in load_task_result(tmp_path, "kid")["delegate_terminal_reconciliation"]["terminal_runs"]}
    assert set(rows) == {"run-out", "run-patch"}
    for row in rows.values():
        observation = row["terminal_observation"]
        assert observation["source_read"]["arguments"]["task_id"] == "kid"
        assert observation["complete"] is True and observation["content_kind"] == "terminal_observation"
        assert "read" not in observation, "the stored receipt keeps its original immutable shape"
        assert not {"accounting", "patch", "answer_preview", "engine_continuity", "gaps"} & set(row)
    assert rows["run-out"]["staged_output"]["complete"] is True

    forbidden = lambda *_a, **_k: pytest.fail("Wrap up's capture must not collect, cancel or start work")  # noqa: E731
    monkeypatch.setattr(collection, "collect_terminal", forbidden)
    monkeypatch.setattr(custody, "cancel_and_verify", forbidden)
    _capture(tmp_path, "parent1")
    observations = sorted(current.run(tmp_path, rid).terminal_source["path"] for rid in ("run-out", "run-patch"))
    assert sorted(reads) == observations, "one read per observation; the retained output is only stat-checked"
    assert current.run(tmp_path, "run-out").output_source["path"] not in reads
    assert sorted(captures) == ["run-out", "run-patch"]
    runs = {row["run_id"]: row for row in
            load_task_result(tmp_path, "parent1")["delegate_terminal_reconciliation"][CHILD_FIELD]["children"][0]
            ["custody"]["terminal_runs"]}
    assert runs["run-out"]["output_state"] == "retained_complete" and "gaps" in runs["run-out"]
    assert not runs["run-out"]["gaps"]
    assert runs["run-patch"]["patch"]["capture"]["status"] == "ready_with_changes"


def test_root_terminal_accounting_is_not_presented_as_live_execution(tmp_path, monkeypatch):
    write_task_result(tmp_path, "parent1", "running")
    _collect(tmp_path, monkeypatch, tid="parent1")
    audit = _capture(tmp_path, "parent1")
    assert audit["open_run_ids"] == ["run-one"]  # custody policy stays unchanged
    notice = terminal.terminal_custody_notice({"delegate_terminal_reconciliation": audit}, owner_row=True)
    assert notice == "Confirmed delegated terminal receipts: run-one: cancelled on test (accounting not settled)."
    receipt = osr.retained_work_fields({"delegate_terminal_reconciliation": audit})[EVENT_FIELD]["text"]
    assert ("This task's run run-one stopped: full output saved; cost unknown; accounting not settled"
            in receipt)


def test_terminal_run_with_unsettled_money_reads_terminal_not_live_outside_wrap_up(tmp_path, monkeypatch):
    """F5: without any Wrap up the common renderer keeps both facts and calls the run terminal."""
    write_task_result(tmp_path, "solo", "failed")
    _collect(tmp_path, monkeypatch, tid="solo")
    audit = terminal.terminal_custody_disclosure(tmp_path, "solo")
    assert CHILD_FIELD not in audit
    assert audit["open_run_ids"] == audit["unreconciled"] == ["run-one"], "custody and its debt are unchanged"
    assert [row["run_id"] for row in audit["terminal_runs"]] == ["run-one"]
    envelope = {"delegate_terminal_reconciliation": audit}
    owner = terminal.terminal_custody_notice(envelope, owner_row=True)
    assert owner == "Confirmed delegated terminal receipts: run-one: cancelled on test (accounting not settled)."
    shared = terminal.terminal_custody_notice(envelope)
    assert "Open delegated execution: none recorded by the current audit." in shared
    assert "run-one: cancelled on test (accounting not settled)" in shared


def test_cancel_readback_survives_even_when_money_cannot_settle(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    from ouroboros.artifacts import read_actor_source_bytes

    entry = started(tmp_path, tid="kid")
    data = detail(text="last saved output")
    data["summary"].update(state="cancelled", resumable={"carriers": ["packet"]})
    gateway = SimpleNamespace(cancel_run=lambda *_a, **_k: {"accepted": True}, get_run=lambda _rid: data)
    monkeypatch.setattr(usage_accounting, "record_subscription_session",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("money unavailable")))
    answer = custody.cancel_and_verify(tmp_path, gateway, entry, "owner_task_gone")
    assert answer["state"] == "cancelled"
    held = current.run(tmp_path, entry.run_id)
    assert not held.settled and held.terminal_state == "cancelled"
    retained = json.loads(read_actor_source_bytes(tmp_path, "kid", held.terminal_source))
    assert retained == data
    # Ordinary collection needs no second GET or generation to recover these bytes.
    collection.collect_terminal(tmp_path, entry.run_id)
    assert collection.retained_payload(tmp_path, current.run(tmp_path, entry.run_id))["primary_output"]["text"] == "last saved output"


def test_wrap_up_mailbox_carries_cancelled_child_answer_and_existing_capture_once(tmp_path, monkeypatch):
    import time

    from ouroboros import usage_accounting
    from supervisor import workers
    from tests.test_owner_stop_s3 import _fake_queue, _finalize_rows

    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path)
    entry = started(tmp_path, tid="kid", execution_root=str(tmp_path / "snapshot"), snapshot_id="snapshot")
    data = detail(text="cancelled child answer")
    data["summary"]["state"] = "cancelled"
    gateway = SimpleNamespace(cancel_run=lambda *_a, **_k: {"accepted": True}, get_run=lambda _rid: data)
    monkeypatch.setattr(usage_accounting, "record_subscription_session", lambda *_a, **_k: None)
    assert custody.cancel_and_verify(tmp_path, gateway, entry, "owner_task_gone")["state"] == "cancelled"
    cap = custody.delegated_capture_dir(tmp_path, "kid", custody.capture_key(entry))
    cap.mkdir(parents=True, exist_ok=True)
    (cap / "workspace_patch.json").write_text(json.dumps({"status": "ready_with_changes"}))
    (cap / "workspace.patch").write_text("diff --git a/a b/a\n")
    custody.emit(tmp_path, custody.PATCH_CAPTURED, {"run_id": entry.run_id, "task_id": "kid"})
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None))
    intent = request_cancel(tmp_path, "parent1", requested_stop_policy=STOP_POLICY_FINALIZE)
    meta = {"task": {"id": "parent1", "chat_id": 0}, "started_at": time.time()}
    queue = _fake_queue(tmp_path, {"parent1": meta})
    for _ in range(2):
        assert owner_stop._arm_owner_stop_episode(queue, "parent1", intent, meta, now=time.time(), cascade=True)
    controls = _finalize_rows(tmp_path, "parent1")
    assert len(controls) == 1
    assert "cancelled child answer" in controls[0]["text"]
    assert "ready_with_changes" in controls[0]["text"]
    audit = load_task_result(tmp_path, "parent1")["delegate_terminal_reconciliation"]
    assert audit[CHILD_FIELD]["stop"] == {"task_id": "parent1", "request_id": intent["request_id"]}
    held = audit[CHILD_FIELD]["children"][0]["custody"]["terminal_runs"][0]
    assert held["patch"]["capture"]["patch_read"]["path"].endswith("workspace.patch")
    assert held["terminal_observation"]["source_read"]["arguments"]["task_id"] == "kid"
    assert not audit["open_run_ids"]


@pytest.mark.parametrize("exit_kind", ["candidate", "expired", "error", "one_call"])
def test_existing_terminal_event_keeps_receipt_on_every_wrap_up_exit(tmp_path, monkeypatch, exit_kind):
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    _child(tmp_path, result="retained child answer")
    _collect(tmp_path, monkeypatch)
    intent, stop = _wrap_up(tmp_path)
    control = REASON_OWNER_REQUESTED_FINALIZATION + "\n" + owner_stop._child_result_projection(
        SimpleNamespace(DRIVE_ROOT=tmp_path), "parent1", stop=stop)
    calls = []

    def final(*args, **_kwargs):
        calls.append(args[1][-1]["content"])
        if exit_kind == "error":
            raise RuntimeError("test final call failed")
        return {"role": "assistant", "content": "Final model bytes."}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", final)
    if exit_kind == "candidate":
        loop._replace_delivery_candidate(registry, ctx, trace, "Existing model bytes.", control="candidate")
    if exit_kind == "expired":
        monkeypatch.setattr("ouroboros.loop_round_limits._owner_stop_window_elapsed", lambda _ctx: True)
    text, usage, _ = loop._handle_forced_finalization(ctx, control)
    assert len(calls) == (0 if exit_kind in {"candidate", "expired"} else 1)
    if calls:
        assert "saved child answer" in calls[0] and "engine_capability_only" in calls[0]
    if exit_kind == "candidate":
        assert text == "Existing model bytes."
    event = prepare_terminal_send_event(tmp_path, {"id": "parent1"}, text, usage,
                                        {"text": text}, presence=False)
    assert event["text"] == text
    receipt = event[EVENT_FIELD]
    assert receipt["id"] == f"cascade:parent1:{intent['request_id']}:retained_work"
    assert "Subtask kid (Stopped): a saved text with no recorded author" in receipt["text"]
    assert "Run run-one stopped: full output saved; cost unknown; accounting not settled" in receipt["text"]
    assert DOOR in receipt["text"]
    assert "document" not in usage["delegate_terminal_reconciliation"][CHILD_FIELD], "the Stop path writes no file"
    assert "retained child answer" not in receipt["text"], "child model words stay in the child's own card"
    assert "session transfer" in terminal_host_notice_text(usage)
    assert usage["delegate_terminal_reconciliation"][CHILD_FIELD]["children"][0]["task_id"] == "kid"


@pytest.mark.parametrize("exit_kind", ["expired", "error"])
def test_failed_wrap_up_final_then_cascade_then_replay_send_one_receipt(tmp_path, monkeypatch, exit_kind):
    """F2/F3 through the real seams: a Wrap up whose root settles failed (the zero-call
    expiry, or a failed final call) sends its final frame AND the cascade fallback, and
    both carry the receipt under one Stop identity. With every replay the web gets one
    receipt row and Telegram, progress mirroring off, one receipt message; the final
    text, the cancel receipt and the root's own custody row arrive once each, unchanged."""
    import asyncio
    from collections import deque

    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from ouroboros.task_finalization import register_final_answer_owed
    from ouroboros.utils import append_jsonl
    from supervisor import events_chat_delivery as delivery
    from supervisor import message_bus, terminal_delivery
    from tests.test_delivery_forced_finalization import _forced_test_context
    from tests.test_telegram_subagent_cards import _Rec, _setup

    plugin, api = _setup(tmp_path, monkeypatch)  # TELEGRAM_MIRROR_PROGRESS unset: progress stays off
    handle = plugin._make_outbound(api)
    frames = []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda _topic, event: asyncio.run(handle(dict(event))))
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    host = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=append_jsonl,
                           send_with_budget=message_bus.send_with_budget)

    loop, _registry, ctx, _trace = _forced_test_context(tmp_path)
    _child(tmp_path, result="kid model words")
    _collect(tmp_path, monkeypatch)
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-root", task_id="parent1", route_id="test", model="test"))
    intent, stop = _wrap_up(tmp_path)
    receipt_id = f"cascade:parent1:{intent['request_id']}:retained_work"
    control = REASON_OWNER_REQUESTED_FINALIZATION + "\n" + owner_stop._child_result_projection(
        SimpleNamespace(DRIVE_ROOT=tmp_path), "parent1", receipt_task_id="parent1", stop=stop)
    calls = []

    def final_call(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("test final call failed")

    monkeypatch.setattr(loop, "call_llm_with_retry", final_call)
    if exit_kind == "expired":
        monkeypatch.setattr("ouroboros.loop_round_limits._owner_stop_window_elapsed", lambda _ctx: True)
    answer, usage, _ = loop._handle_forced_finalization(ctx, control)
    assert len(calls) == (0 if exit_kind == "expired" else 1)
    assert answer and usage["execution_status"] == "failed"
    assert not usage.get("_best_effort_extracted"), "no extracted answer: the root settles failed, the cascade sends"

    task = {"id": "parent1", "type": "task", "chat_id": 1}
    final = prepare_terminal_send_event(tmp_path, task, answer, usage, {
        "type": "send_message", "task_id": "parent1", "chat_id": 1,
        "text": answer, "log_text": answer, "format": "markdown"}, presence=False)
    assert final[EVENT_FIELD]["id"] == receipt_id
    assert "Open delegated execution: run-root." in final["terminal_custody_notice"]
    register_final_answer_owed(task, final, env_drive_root=tmp_path)
    delivery._handle_send_message(dict(final), host)
    write_task_result(tmp_path, "parent1", "failed", result=answer, chat_id=1)

    # The cascade fallback owes its own receipt; its delivery is the replay below.
    monkeypatch.setattr(terminal_delivery, "deliver_unreviewed_salvage", lambda *_a, **_k: False)
    for _ in range(2):
        assert terminal_delivery.deliver_cascade_summary(tmp_path, "parent1", {"chat_id": 1}, {"kid": "cancelled"})
        for row in terminal_delivery.pending_deliveries(tmp_path):
            delivery._handle_send_message({**row, "_delivery_replay": True}, host)
    delivery._handle_send_message(dict(final), host)  # a buffered duplicate of the final frame
    assert terminal_delivery.pending_deliveries(tmp_path) == []

    def by_row(frame_id):
        return [frame for frame in frames if frame.get("card_row_id") == frame_id]

    assert len(by_row(receipt_id)) == 1
    (row,) = by_row(receipt_id)
    assert row["role"] == "system" and row["system_type"] == "custody_notice"
    assert row["terminal_receipt"] == "retained_work" and row["content"].startswith("Work saved at Wrap up:")
    assert usage["terminal_origin"] == "host_notice"
    answer_frames = [frame for frame in frames if frame.get("content") == answer]
    assert len(answer_frames) == 1
    assert answer_frames[0]["role"] == "system"
    assert not answer_frames[0].get("system_type"), "the final host answer is not a custody or cancel receipt"
    assert not [frame for frame in frames if frame.get("role") == "assistant"]
    custody_rows = [frame for frame in frames if str(frame.get("card_row_id") or "").startswith("final:parent1:")]
    assert [frame["content"] for frame in custody_rows] == [final["terminal_custody_notice"]]
    assert len([frame for frame in frames if frame.get("system_type") == "cancel_receipt"]) == 1
    sent = [text for _chat, text, _mode, _id in _Rec.sent]
    assert sent.count(row["content"]) == 1, "Telegram receives the terminal receipt once, progress off"
    assert sent.count(answer) == 1
    assert final["terminal_custody_notice"] not in sent, "other card rows still follow the progress toggle"


def test_mixed_api_failure_and_session_material_keep_both_host_receipts(tmp_path, monkeypatch):
    write_task_result(tmp_path, "parent1", "running")
    api_notice = "The API response timed out; its completion is unknown."
    write_task_result(tmp_path, "api-child", "failed", root_task_id="parent1", parent_task_id="parent1",
                      result="API partial model bytes", terminal_host_notice=api_notice)
    _child(tmp_path, result="Session child model bytes")
    _collect(tmp_path, monkeypatch)
    prompt = owner_stop._child_result_projection(SimpleNamespace(DRIVE_ROOT=tmp_path), "parent1")
    assert api_notice in prompt and "run-one" in prompt
    usage = {"terminal_origin": "model_final", "terminal_host_notice": "Existing parent host notice."}
    event = prepare_terminal_send_event(tmp_path, {"id": "parent1"}, "Root model bytes", usage,
                                        {"text": "Root model bytes"}, presence=False)
    assert event["text"] == "Root model bytes"
    assert usage["terminal_host_notice"] == "Existing parent host notice."
    owner = event[EVENT_FIELD]["text"]
    assert "Subtask api-child (Failed): a saved text with no recorded author\n- Subtask api-child: Host note: " + api_notice in owner
    assert "Run run-one stopped: full output saved; cost unknown; accounting not settled" in owner
    joined = terminal_host_notice_text(usage)
    assert joined.startswith("Existing parent host notice.") and joined.count(api_notice) == 1
    children = usage["delegate_terminal_reconciliation"][CHILD_FIELD]["children"]
    assert children[0]["terminal_host_notice"] == api_notice
    assert children[0]["result_preview"] == "API partial model bytes"
    assert load_task_result(tmp_path, "api-child")["terminal_host_notice"] == api_notice


def test_owner_row_names_saved_work_and_where_it_is_never_model_calls(tmp_path, monkeypatch):
    """F4: the owner's row states what was saved in owner words and points to the existing
    child card and its Files, which really list the capture; the model input keeps exact readers."""
    from ouroboros.artifacts import collect_task_artifact_records

    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="Session child model words", terminal_origin="model_final")
    _captured_patch(tmp_path, rid="run-one", text="engine words only")
    intent, stop = _wrap_up(tmp_path)
    prompt = owner_stop._child_result_projection(SimpleNamespace(DRIVE_ROOT=tmp_path), "parent1", stop=stop)
    stored = load_task_result(tmp_path, "parent1")
    row = osr.retained_work_fields(stored)[EVENT_FIELD]
    assert row["id"] == f"cascade:parent1:{intent['request_id']}:retained_work"
    assert row["text"] == (
        "Work saved at Wrap up:\n"
        "- Subtask kid (Stopped): its answer is saved\n"
        "- Subtask kid: Run run-one stopped: only the engine's last report saved; file changes captured, not applied; "
        "cost unknown; accounting not settled\n"
        f"- {DOOR}\n"
        "- Each subtask's card shows what it saved, and its Files list captured file changes.")
    # The door the row names: the task detail's typed entry, never a link in the row's Markdown.
    assert osr.wrap_up_document_view(stored) == {"name": "work-saved-at-wrap-up-parent1.md",
                                                 "request_id": intent["request_id"]}
    assert "](" not in row["text"] and "/api/" not in row["text"]
    for token in (*MODEL_CALLS, *RAW_CODES, "Session child model words", "engine words only"):
        assert token not in row["text"]
    # The pointer names doors that exist: the child's own saved text and its Files listing.
    assert load_task_result(tmp_path, "kid")["result"] == "Session child model words"
    relpaths = [record.get("relpath") or "" for record in collect_task_artifact_records(tmp_path, "kid")]
    assert any(path.startswith("delegated_runs/") and path.endswith("/workspace.patch") for path in relpaths)
    # The model's one input keeps the exact facts and readers, its excerpts labelled as the run's words.
    assert "get_task_result(task_id='parent1', include_authority=True)" in prompt
    assert "The owner downloads all rows and saved texts as “Work saved at Wrap up” from this task's Files." in prompt
    assert "kid (cancelled): saved answer excerpt (its own words): Session child model words" in prompt
    assert "ready_with_changes" in prompt and "Output excerpt (the run's own words): engine words only" in prompt
    run = stored["delegate_terminal_reconciliation"][CHILD_FIELD]["children"][0]["custody"]["terminal_runs"][0]
    assert run["output_state"] == "observation_only" and run["terminal_observation"]["complete"] is True


def test_cancelled_leaf_output_without_child_result_reaches_the_owners_download(tmp_path, monkeypatch):
    """Check 3: a stopped leaf that wrote no task result keeps its session output. Wrap up reads
    only each run's engine report; the owner's download reads the saved outputs, quoted whole as
    the runs' own words, and repeats no report a complete output was collected from."""
    from ouroboros import artifacts

    write_task_result(tmp_path, "parent1", "running")
    write_task_result(tmp_path, "kid", "cancelled", root_task_id="parent1", parent_task_id="parent1")
    _collect(tmp_path, monkeypatch, rid="run-short", text="The short saved answer.")
    whole = "The long answer begins, and it ends here."
    _retained_long(tmp_path, whole=whole)
    reads = []
    real = artifacts.read_actor_source_bytes
    monkeypatch.setattr(artifacts, "read_actor_source_bytes",
                        lambda *a, **k: reads.append(a[2].get("path")) or real(*a, **k))
    _intent, stop = _wrap_up(tmp_path)
    _capture(tmp_path, "parent1", stop)
    runs = {rid: current.run(tmp_path, rid) for rid in ("run-short", "run-long")}
    assert sorted(reads) == sorted(run.terminal_source["path"] for run in runs.values()), "Wrap up reads no output"
    owner = osr.retained_work_fields(load_task_result(tmp_path, "parent1"))[EVENT_FIELD]["text"]
    assert "- Subtask kid (Stopped): no text saved" in owner
    assert "Run run-long stopped: full output saved; cost unknown; accounting not settled" in owner
    assert DOOR in owner
    write_task_result(tmp_path, "parent1", "failed")
    reads.clear()
    _saved, response = _download(_client(tmp_path))
    assert response.status_code == 200, response.text
    document = response.content.decode("utf-8")
    assert ("### Subtask kid, run run-short\n\nFull output, in the run's own words (from its saved output):"
            "\n\n> The short saved answer.") in document
    assert ("### Subtask kid, run run-long\n\nFull output, in the run's own words (from its saved output):"
            "\n\n> " + whole) in document
    assert "> The long answer begins\n" not in document, "the partial report is not quoted beside the whole output"
    assert {run.output_source["path"] for run in runs.values()} <= set(reads)
    for foreign in ('"primaryOutput"', "runDir", '"summary"', "Material facts", "get_task_result", "source_handles"):
        assert foreign not in document


def test_wrap_up_stop_path_reads_and_writes_no_saved_output(tmp_path, monkeypatch):
    """Check 4: through the arm, the final frame and the cascade fallback, the only payload reads
    are each run's engine report (the existing observation read, once per capture). No saved output
    is read, the owner's copy is not written, no task file is stored, and nothing is collected,
    cancelled or started; a child's kept salvage copy waits for the download."""
    import time

    from ouroboros import artifacts, observability
    from supervisor import terminal_delivery, workers
    from tests.test_owner_stop_s3 import _fake_queue, _finalize_rows

    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path, result="Running task cancelled and worker terminated.")
    _collect(tmp_path, monkeypatch)
    _retained_long(tmp_path, whole="x" * 300_000)
    observability.preserve_salvaged_output(tmp_path, "kid", "kept salvage copy")
    reports = sorted(current.run(tmp_path, rid).terminal_source["path"] for rid in ("run-one", "run-long"))
    reads = []
    real = artifacts.read_actor_source_bytes
    monkeypatch.setattr(artifacts, "read_actor_source_bytes",
                        lambda *a, **k: reads.append(a[2].get("path")) or real(*a, **k))
    forbidden = lambda *_a, **_k: pytest.fail("the Stop path reads no saved work and starts no work")  # noqa: E731
    for owner, name in ((artifacts, "store_task_artifact_bytes"), (artifacts, "copy_file_to_task_artifacts"),
                        (osr, "write_wrap_up_document"), (collection, "collect_terminal"),
                        (custody, "cancel_and_verify")):
        monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None))
    intent = request_cancel(tmp_path, "parent1", requested_stop_policy=STOP_POLICY_FINALIZE)
    meta = {"task": {"id": "parent1", "chat_id": 0}, "started_at": time.time()}
    assert owner_stop._arm_owner_stop_episode(_fake_queue(tmp_path, {"parent1": meta}), "parent1", intent, meta,
                                              now=time.time(), cascade=True)
    assert len(_finalize_rows(tmp_path, "parent1")) == 1
    assert sorted(reads) == reports, "the arm reads each engine report once and no saved output"
    event = prepare_terminal_send_event(tmp_path, {"id": "parent1", "chat_id": 17}, "Final words",
                                        {"terminal_origin": "model_final"},
                                        {"type": "send_message", "task_id": "parent1", "chat_id": 17,
                                         "text": "Final words"}, presence=False)
    assert event["text"] == "Final words" and DOOR in event[EVENT_FIELD]["text"]
    assert sorted(reads) == reports, "the final frame reads nothing"
    write_task_result(tmp_path, "parent1", "failed", chat_id=17)
    monkeypatch.setattr(terminal_delivery, "deliver_unreviewed_salvage", lambda *_a, **_k: None)
    monkeypatch.setattr(terminal_delivery, "register_pending_delivery", lambda *_a, **_k: True)
    assert terminal_delivery.deliver_cascade_summary(tmp_path, "parent1", {"chat_id": 17}, {"kid": "cancelled"})
    assert sorted(reads) == sorted(reports * 2), "the fallback capture reads the same reports, nothing else"


def test_unreadable_receipt_claims_wrap_up_only_with_the_stops_own_provenance(tmp_path, monkeypatch):
    """F7: a transient read failure on an ordinary Stop adds no Wrap up row; only an open
    finalize-then-cancel cascade proves the receipt applies and is merely unknown."""
    from supervisor.terminal_delivery import build_unreviewed_salvage_event

    monkeypatch.setattr(osr, "retained_work_fields",
                        lambda _result: (_ for _ in ()).throw(OSError("transient read")))
    write_task_result(tmp_path, "plain", "running")
    assert osr.terminal_event_fields(tmp_path, "plain") == {}
    request_cancel(tmp_path, "plain")  # an ordinary immediate Stop
    assert osr.terminal_event_fields(tmp_path, "plain") == {}
    write_task_result(tmp_path, "plain", "cancelled", chat_id=17)
    event = build_unreviewed_salvage_event(tmp_path, {"chat_id": 17}, "plain", outcome="stopped by the owner")
    assert event is not None and EVENT_FIELD not in event
    write_task_result(tmp_path, "wrap", "running")
    intent, _stop = _wrap_up(tmp_path, "wrap")
    unknown = osr.terminal_event_fields(tmp_path, "wrap")[EVENT_FIELD]
    assert unknown == {"id": f"cascade:wrap:{intent['request_id']}:retained_work",
                       "text": "Work saved at Wrap up could not be read; what was kept is unknown."}


def test_all_children_and_runs_survive_bounded_prompt_without_review_work(tmp_path, monkeypatch):
    write_task_result(tmp_path, "parent1", "running")
    for index in range(23):
        _child(tmp_path, tid=f"kid-{index:02}", result="answer " * 100)
    for index in range(4):
        _collect(tmp_path, monkeypatch, tid="kid-00", rid=f"run-{index}")
    review = started(tmp_path, tid="kid-00", rid="review", source="review_substrate:change")
    assert review.review_owned
    collection.observe_run(tmp_path, review, detail("review"))
    prompt = owner_stop._child_result_projection(SimpleNamespace(DRIVE_ROOT=tmp_path), "parent1")
    assert "7 more receipt rows omitted" in prompt  # 23 task rows + four run rows
    assert "include_authority=True" in prompt
    audit = load_task_result(tmp_path, "parent1")["delegate_terminal_reconciliation"]
    children = audit[CHILD_FIELD]["children"]
    assert len(children) == 23
    assert len(children[0]["custody"]["terminal_runs"]) == 4
    assert "review" not in [r["run_id"] for r in children[0]["custody"]["terminal_runs"]]
    assert children[-1]["result_preview_omitted_chars"] == 460
    assert "kid-22" not in prompt
    owner = osr.retained_work_fields(load_task_result(tmp_path, "parent1"))[EVENT_FIELD]["text"]
    assert len(owner.splitlines()) == 14 and (
        "\n- 17 more lines are not shown here; “Work saved at Wrap up” in this task's Files has them all.\n") in owner
    assert "kid-22" not in owner
    document = _document(tmp_path)
    listed = [line for line in document.split("## Saved texts")[0].splitlines() if line.lstrip().startswith("- ")]
    assert len(listed) == 27, "the download lists every row the card omits"
    assert "- Subtask kid-22 (Stopped): a saved text with no recorded author" in document
    assert "### Subtask kid-22\n\nIts saved task text; the author is not recorded:\n\n> " + "answer " * 99 + "answer" in document


@pytest.mark.parametrize("status", ["ready_with_changes", "ready_no_changes", "failed", "missing", "unconfirmed"])
def test_patch_capture_states_and_disposition_are_facts_not_hash_inferences(tmp_path, monkeypatch, status):
    from ouroboros.artifacts import task_artifact_dir_path

    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path)
    entry = started(tmp_path, tid="kid", execution_root=str(tmp_path / "snapshot"), snapshot_id="snapshot")
    collection.observe_run(tmp_path, entry, detail())
    cap = custody.delegated_capture_dir(tmp_path, "kid", custody.capture_key(entry))
    if status != "unconfirmed":
        cap.mkdir(parents=True, exist_ok=True)
        (cap / "workspace_patch.json").write_text(json.dumps({"status": status, "sha256": "a" * 64}))
    if status == "ready_with_changes":
        (cap / "workspace.patch").write_text("diff --git a/a b/a\n")
    if status != "unconfirmed":
        custody.emit(tmp_path, custody.PATCH_CAPTURED, {"run_id": entry.run_id, "task_id": "kid"})
    audit = _capture(tmp_path, "parent1")
    patch = audit[CHILD_FIELD]["children"][0]["custody"]["terminal_runs"][0]["patch"]
    assert patch["capture"]["status"] == ("pending" if status == "unconfirmed" else status)
    assert bool(patch["capture"].get("patch_read")) is (status == "ready_with_changes")
    assert patch["disposition"] == "not_recorded" and not patch["apply_pending"]
    if status == "unconfirmed":
        assert patch["captured"] is False
        assert "manifest_read" not in patch["capture"]
    else:
        assert str(task_artifact_dir_path(tmp_path, "kid")) in patch["capture"]["manifest_read"]["path"]
    owner = osr.retained_work_fields({"delegate_terminal_reconciliation": audit})[EVENT_FIELD]["text"]
    words = {"ready_with_changes": "file changes captured, not applied", "ready_no_changes": "no file changes",
             "failed": "capturing file changes failed", "missing": "captured file changes are missing",
             "unconfirmed": "file changes not captured yet"}[status]
    assert words in owner and not any(code in owner for code in ("ready_with_changes", "ready_no_changes", "not_recorded"))


def test_no_output_legacy_unknown_and_partial_or_lost_sources_stay_distinct(tmp_path, monkeypatch):
    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path)
    none = started(tmp_path, tid="kid", rid="none")
    collection.observe_run(tmp_path, none, {"summary": {"state": "cancelled"}, "lastSeq": 1,
                                           "primaryOutput": None})
    legacy = started(tmp_path, tid="kid", rid="legacy")
    custody.emit(tmp_path, custody.SETTLED, {"run_id": legacy.run_id, "task_id": "kid", "state": "cancelled"})
    partial = _collect(tmp_path, monkeypatch, rid="partial")
    custody.emit(tmp_path, custody.OUTPUT_SPILLED, {"run_id": partial.run_id, "task_id": "kid",
        "artifact": partial.output_artifact, "source": partial.output_source, "staged": True,
        "full_content": False, "sha256": partial.output_sha, "terminal_sha": partial.output_terminal_sha})
    audit = _capture(tmp_path, "parent1")
    runs = {r["run_id"]: r for r in audit[CHILD_FIELD]["children"][0]["custody"]["terminal_runs"]}
    assert runs["none"]["output_state"] == "none_reported"
    assert runs["partial"]["output_state"] == "retained_partial"
    # A legacy run recorded no observation: its output is unknown, its recorded money still known.
    assert runs["legacy"]["output_state"] == "unknown"
    assert runs["legacy"]["gaps"] == ["terminal_observation_not_recorded"]
    assert runs["legacy"]["accounting"]["settled"] is True
    assert "output unknown" in terminal.terminal_custody_notice({"delegate_terminal_reconciliation": audit})
    from ouroboros.delegate_output import staged_source_path
    staged = staged_source_path(tmp_path, partial.output_source)
    # A size that no longer matches, then no file at all: the Stop path only stats, so a lost
    # canonical copy is unverified (a retained placement could hold it); the download reads.
    for damage, gap, words in (("corrupt", "retained_output_unavailable", "its bytes no longer match what was recorded"),
                               ("lost", "retained_output_unverified", "it was not found where it is kept")):
        if damage == "corrupt":
            staged.write_bytes(b"{}")
        else:
            staged.unlink()
        page = delegated_output_source_projection(tmp_path, load_task_result(tmp_path, "kid"), partial.output_sha)
        assert page["status"] == "unavailable", damage
        audit = _capture(tmp_path, "parent1")
        runs = {r["run_id"]: r for r in audit[CHILD_FIELD]["children"][0]["custody"]["terminal_runs"]}
        assert gap in runs["partial"]["gaps"], damage
        assert ("### Subtask kid, run partial\n\nThe text from its saved output could not be read: "
                f"{words}.") in _document(tmp_path), damage
    owner = osr.retained_work_fields({"delegate_terminal_reconciliation": audit})[EVENT_FIELD]["text"]
    assert "Run partial stopped: partial output saved; cost unknown; accounting not settled; " \
           "whether the saved output is still readable is unknown" in owner
    assert "Run none stopped: no output reported" in owner and "Run legacy stopped: output unknown" in owner


def test_physical_retry_receipt_and_stale_terminal_writer_keep_full_facts(tmp_path):
    write_task_result(tmp_path, "physical", "running", root_task_id="logical")
    write_task_result(tmp_path, "kid", "cancelled", root_task_id="logical", parent_task_id="logical", result="kept")
    prompt = owner_stop._child_result_projection(SimpleNamespace(DRIVE_ROOT=tmp_path), "logical",
                                                receipt_task_id="physical", stop=("logical", "ci_retry"))
    assert "task_id='physical'" in prompt
    before = load_task_result(tmp_path, "physical")["delegate_terminal_reconciliation"][CHILD_FIELD]
    # The receipt's row identity is the cascade intent's, whichever physical task stores it.
    assert osr.retained_work_fields(load_task_result(tmp_path, "physical"))[EVENT_FIELD]["id"] == \
        "cascade:logical:ci_retry:retained_work"
    write_task_result(tmp_path, "physical", "failed", delegate_terminal_reconciliation={
        "audit_status": "ok", "terminal_runs": [], "unreconciled": []})
    assert load_task_result(tmp_path, "physical")["delegate_terminal_reconciliation"][CHILD_FIELD] == before


def test_later_observation_preserves_literal_old_source_without_calling_it_current_output(tmp_path, monkeypatch):
    write_task_result(tmp_path, "parent1", "running")
    _child(tmp_path)
    entry = _collect(tmp_path, monkeypatch)
    first = _capture(tmp_path, "parent1")[CHILD_FIELD]["children"][0]["custody"]["terminal_runs"][0]
    original = first["terminal_observation"]["source_read"]["arguments"]
    revision = detail(text="later saved output", seq=5)
    revision["summary"]["state"] = "cancelled"
    collection.observe_run(tmp_path, entry, revision)
    latest = _capture(tmp_path, "parent1")[CHILD_FIELD]["children"][0]["custody"]["terminal_runs"][0]
    assert latest["output_state"] == "retained_earlier_observation"
    assert latest["other_terminal_observations"][0]["sha256"] == original["delegated_output_sha256"]
    assert not latest.get("other_staged_outputs"), "observations are not additional full outputs"
    read = delegated_output_source_projection(tmp_path, load_task_result(tmp_path, "kid"),
                                               original["delegated_output_sha256"])
    assert read["status"] == "available" and read["content_kind"] == "terminal_observation"


@pytest.mark.parametrize("membership_unavailable", [False, True])
def test_expired_unarmed_cascade_carries_one_existing_receipt(tmp_path, monkeypatch, membership_unavailable):
    from supervisor import terminal_cascade
    from supervisor import terminal_delivery as delivery

    intent, _stop = _wrap_up(tmp_path)
    write_task_result(tmp_path, "parent1", "cancelled", chat_id=17)
    _child(tmp_path, result="kept through fallback")
    _collect(tmp_path, monkeypatch)
    queued = []
    monkeypatch.setattr(delivery, "deliver_unreviewed_salvage", lambda *_a, **_k: None)
    monkeypatch.setattr(delivery, "register_pending_delivery", lambda _root, event: queued.append(event) or True)
    if membership_unavailable:
        monkeypatch.setattr(terminal_cascade, "_cascade_descendant_rows",
                            lambda *_a, **_k: (_ for _ in ()).throw(OSError("tree unavailable")))
    for _ in range(2):
        assert delivery.deliver_cascade_summary(tmp_path, "parent1", {"chat_id": 17}, {"kid": "cancelled"})
    assert queued[0]["delivery_id"] == queued[1]["delivery_id"]
    assert queued[0]["role"] == "system" and queued[0]["system_type"] == "cancel_receipt"
    assert queued[0][EVENT_FIELD]["id"] == queued[1][EVENT_FIELD]["id"] == \
        f"cascade:parent1:{intent['request_id']}:retained_work"
    owner = queued[0][EVENT_FIELD]["text"]
    assert "Subtask kid (Stopped): a saved text with no recorded author" in owner and "kept through fallback" not in owner
    assert "Run run-one stopped" in owner
    assert ("could not be read; this list may be incomplete" in owner) is membership_unavailable
    assert load_task_result(tmp_path, "parent1")["cancel_receipt"]["delivery_id"] == queued[0]["delivery_id"]


def test_unknown_membership_is_a_durable_gap(tmp_path, monkeypatch):
    from supervisor import terminal_delivery
    write_task_result(tmp_path, "parent1", "running")
    monkeypatch.setattr(terminal_delivery, "_cascade_descendant_rows",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("tree unavailable")))
    text = owner_stop._child_result_projection(SimpleNamespace(DRIVE_ROOT=tmp_path), "parent1")
    assert "membership unknown" in text
    assert load_task_result(tmp_path, "parent1")["delegate_terminal_reconciliation"][CHILD_FIELD]["gap"]


def test_literal_published_reader_survives_child_drive_retention(tmp_path, monkeypatch):
    from ouroboros import headless, task_custody
    from ouroboros.tools.registry import ToolContext
    from tests.test_delegate_retry_consumers import call

    parent = tmp_path / "canonical"
    child = headless.prepare_task_drive(parent, "kid", "empty")
    _child(child, result="saved in fork")
    # The body collector owns the canonical custody root, even while the
    # nanny's result is written on its execution drive (emit_task_results).
    child_ctx = ToolContext(repo_dir=tmp_path, drive_root=child, task_id="kid")
    child_ctx.budget_drive_root = parent
    collection_root = custody.custody_root(child_ctx)
    assert collection_root == parent
    _collect(collection_root, monkeypatch, money=True)
    audit = terminal.terminal_custody_disclosure(collection_root, "kid")
    terminal.record_terminal_reconciliation(child, "kid", audit)
    literal = [audit["terminal_runs"][0][key]["source_read"]
               for key in ("staged_output", "terminal_observation")]
    write_task_result(parent, "kid", "running", child_drive_root=str(child))
    headless.copy_child_task_result(parent, {"id": "kid", "drive_root": str(child)})
    promoted = headless.retry_child_task_refs(parent, child, "kid")
    assert promoted["child_ref_promotion"]["status"] == "complete"
    assert task_custody.settle_child_drive(parent, "kid", child, live=lambda _: False)["status"] == "removed"
    drive = tmp_path / "reader"
    drive.mkdir()
    ctx = ToolContext(repo_dir=tmp_path, drive_root=drive, task_id="parent1")
    ctx.budget_drive_root = parent
    write_task_result(parent, "parent1", "running")
    _capture(parent, "parent1")
    for source in literal:
        metadata = call(ctx, source["tool"], **source["arguments"])
        assert metadata.status == "ok", metadata.text
        source_view = json.loads(metadata.text)["delegated_output_source"]
        assert source_view.get("status") == "available", metadata.text
        size = source_view["complete_chars"]
        read = call(ctx, source["tool"], **source["arguments"], source_start_char=0, source_end_char=size)
        assert read.status == "ok", read.text
        data = json.loads(read.text)["delegated_output_source"]
        assert data["status"] == "available"
        assert "saved child answer" in data["text"]
        assert data["sha256"] == source["arguments"]["delegated_output_sha256"]
    full = call(ctx, "get_task_result", task_id="parent1", include_authority=True)
    assert full.status == "ok", full.text
    authority = json.loads(full.text)["authority"]
    assert authority["delegate_terminal_reconciliation"][CHILD_FIELD]["children"][0]["task_id"] == "kid"


def _write_reciprocal_retry_root(root, logical_id="root-logical", physical_id="root-physical",
                                 *, physical_status="running", chat_id=17):
    write_task_result(root, logical_id, "failed", chat_id=chat_id, root_task_id=logical_id,
                      superseded_by=physical_id, retry_task_id=physical_id, reason_code="TASK_HARD_TIMEOUT")
    write_task_result(root, physical_id, physical_status, chat_id=chat_id, root_task_id=logical_id,
                      delegation_role="root", parent_task_id="", supersedes_task_id=logical_id, original_task_id=logical_id, timeout_retry_from=logical_id,
                      metadata={"is_retry_root_attempt": True, "original_task_id": logical_id,
                                "timeout_retry_from": logical_id})


@pytest.mark.parametrize("mode", ["unarmed_fallback", "armed_then_hardened", "grace_kill"])
def test_retried_root_wrap_up_keeps_physical_receipt_and_predecessor_runs_across_fallbacks(tmp_path, monkeypatch, mode):
    """Defect 1: with reciprocal validated retry lineage L -> R, unarmed pending/expired fallback,
    armed-then-hardened Stop now, and grace-deadline kill all publish on R under Stop identity
    (L, request_id), never list R as a Subtask of L, include L's predecessor root runs, and serve
    task detail, Files download, and get_task_result(include_authority=True) through L and R."""
    import time
    from collections import deque
    from ouroboros.tools.registry import ToolContext
    from supervisor import events_chat_delivery as chat_delivery, message_bus, terminal_delivery, workers
    from tests.test_delegate_retry_consumers import call
    from tests.test_owner_stop_s3 import _fake_queue

    _write_reciprocal_retry_root(tmp_path, "L", "R", physical_status="running", chat_id=17)
    write_task_result(tmp_path, "kid", "cancelled", root_task_id="L", parent_task_id="L",
                      result="kid saved answer", terminal_origin="model_final")
    _collect(tmp_path, monkeypatch, tid="L", rid="run-pred", text="predecessor root run output")
    _collect(tmp_path, monkeypatch, tid="R", rid="run-curr", text="physical retry run output")
    _collect(tmp_path, monkeypatch, tid="kid", rid="run-kid", text="child run output")
    frames, bridge = [], message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 7})
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    monkeypatch.setattr(chat_delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=256))
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=lambda _event: None))
    host = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, append_jsonl=lambda *_a, **_k: None,
                           send_with_budget=message_bus.send_with_budget)
    # The cascade ingress allows a settled logical root whose retry/tree still runs.
    intent = request_cancel(tmp_path, "L", scope=SCOPE_CASCADE,
                            requested_stop_policy=STOP_POLICY_FINALIZE, allow_settled_target=True)
    req_id, receipt_row_id = intent["request_id"], f"cascade:L:{intent['request_id']}:retained_work"
    if mode in ("armed_then_hardened", "grace_kill"):
        meta = {"task": {"id": "R", "root_task_id": "L", "chat_id": 17}, "started_at": time.time()}
        assert owner_stop._arm_owner_stop_episode(_fake_queue(tmp_path, {"R": meta}), "R", intent, meta,
                                                  now=time.time(), cascade=True, logical_task_id="L")
        if mode == "armed_then_hardened":
            hardened = request_cancel(tmp_path, "L", scope=SCOPE_CASCADE,
                                      requested_stop_policy="immediate")
            assert hardened["request_id"] == req_id and hardened.get("hardened_at")
    _write_reciprocal_retry_root(tmp_path, "L", "R", physical_status="cancelled", chat_id=17)
    if mode == "grace_kill":
        final = prepare_terminal_send_event(
            tmp_path, {"id": "R", "root_task_id": "L", "chat_id": 17}, "Timeout notice",
            {"terminal_origin": "host_notice"},
            {"type": "send_message", "task_id": "R", "chat_id": 17, "text": "Timeout notice",
             "terminal_projection_token": "tok_r"}, presence=False)
        chat_delivery._handle_send_message(dict(final), host)
    monkeypatch.setattr(terminal_delivery, "deliver_unreviewed_salvage", lambda *_a, **_k: False)
    assert terminal_delivery.deliver_cascade_summary(tmp_path, "L", {"chat_id": 17},
                                                     {"R": "cancelled", "kid": "cancelled"})
    for row in terminal_delivery.pending_deliveries(tmp_path):
        chat_delivery._handle_send_message({**row, "_delivery_replay": True}, host)
    retained_frames = [f for f in frames if f.get("card_row_id") == receipt_row_id]
    assert len(retained_frames) == 1, mode
    owner_text = retained_frames[0]["content"]
    assert "Subtask R" not in owner_text, "physical retry R must never be described as a child of L"
    for expected in ("- This task's run run-pred stopped: full output saved",
                     "- This task's run run-curr stopped: full output saved",
                     "- Subtask kid (Stopped): its answer is saved",
                     "- Subtask kid: Run run-kid stopped: full output saved"):
        assert expected in owner_text
    client = _client(tmp_path)
    for query_tid in ("L", "R"):
        detail_json = client.get(f"/api/tasks/{query_tid}").json()
        assert detail_json["wrap_up_document"] == {"name": "work-saved-at-wrap-up-R.md", "request_id": req_id}
        _, resp = _download(client, task_id=query_tid)
        assert resp.status_code == 200 and "Subtask R" not in resp.text
        for snippet in ("### This task's run run-pred", "> predecessor root run output",
                        "### This task's run run-curr", "> physical retry run output",
                        "### Subtask kid, run run-kid", "> child run output"):
            assert snippet in resp.text
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="parent-reader")
    ctx.budget_drive_root = tmp_path
    res = call(ctx, "get_task_result", task_id="L", include_authority=True)
    assert res.status == "ok", res.text
    payload = json.loads(res.text)
    receipt = payload["authority"]["delegate_terminal_reconciliation"][CHILD_FIELD]
    assert receipt["task_id"] == "R" and receipt["stop"] == {"task_id": "L", "request_id": req_id}
    assert [c["task_id"] for c in receipt["children"]] == ["kid"]
    assert [r["run_id"] for r in receipt["runs"]] == ["run-pred", "run-curr"]
    summary = call(ctx, "get_task_result", task_id="L")
    assert summary.status == "ok", summary.text
    public = json.loads(summary.text.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0])
    assert public["delegated_custody"]["predecessor_terminal_runs"][0]["run_id"] == "run-pred"


def test_degraded_recapture_preserves_same_stop_membership_and_refs_without_false_no_text(tmp_path, monkeypatch):
    """Defect 2: after a successful capture, degraded recaptures keep previously verified children
    and run refs for the same Stop while adding explicit gap disclosures, never union cleared root
    custody debt or mix different Stop requests, and never emit 'kept no saved text' when unknown."""
    from ouroboros import task_results

    write_task_result(tmp_path, "parent1", "running")
    for tid, words in (("kid-a", "kid-a saved words"), ("kid-b", "kid-b saved words")):
        write_task_result(tmp_path, tid, "cancelled", root_task_id="parent1", parent_task_id="parent1",
                          result=words, terminal_origin="model_final")
    _collect(tmp_path, monkeypatch, tid="kid-a", rid="run-a", text="kid-a full run output")
    _collect(tmp_path, monkeypatch, tid="parent1", rid="run-root", money=False, text="root full run output")
    _intent, stop = _wrap_up(tmp_path)
    first_audit = _capture(tmp_path, "parent1", stop)
    assert first_audit["open_run_ids"] == ["run-root"]
    assert [c["task_id"] for c in first_audit[CHILD_FIELD]["children"]] == ["kid-a", "kid-b"]
    custody.emit(tmp_path, custody.SETTLED, {"run_id": "run-root", "task_id": "parent1",
                                             "state": "cancelled", "cost_usd": 0.0, "spend_disclosed": True})
    orig_load = task_results.load_task_result
    monkeypatch.setattr(task_results, "load_task_result",
                        lambda root, tid, *a, **k: (_ for _ in ()).throw(OSError("child read failed"))
                        if tid == "kid-a" else orig_load(root, tid, *a, **k))
    second_audit = _capture(tmp_path, "parent1", stop)
    assert second_audit["open_run_ids"] == [] and second_audit["unreconciled"] == [], "cleared root debt is not unioned"
    children_by_id = {c["task_id"]: c for c in second_audit[CHILD_FIELD]["children"]}
    assert set(children_by_id) == {"kid-a", "kid-b"} and children_by_id["kid-a"]["gap"] == "child_receipt_unavailable"
    assert children_by_id["kid-a"]["result_preview"] == "kid-a saved words"
    assert children_by_id["kid-a"]["custody"]["terminal_runs"][0]["run_id"] == "run-a"
    monkeypatch.setattr(task_results, "load_task_result", orig_load)

    from supervisor import terminal_delivery
    monkeypatch.setattr(terminal_delivery, "_cascade_descendant_rows",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("tree enumeration failed")))
    receipt = _capture(tmp_path, "parent1", stop)[CHILD_FIELD]
    assert receipt["gap"] == "child_receipt_unavailable"
    assert [c["task_id"] for c in receipt["children"]] == ["kid-a", "kid-b"] and all(c.get("gap") for c in receipt["children"])
    write_task_result(tmp_path, "parent1", "failed")
    doc = _download(_client(tmp_path))[1].text
    assert "Which subtasks belonged to this work could not be read; this list may be incomplete." in doc
    assert "- Subtask kid-a (Stopped): its answer is saved; its saved facts could not all be read" in doc
    for expected in ("> kid-a saved words", "> kid-b saved words", "> kid-a full run output", "> root full run output"):
        assert expected in doc
    fresh = osr.capture_receipt(tmp_path, "parent1", {"kid-b": "cancelled"}, stop=("parent1", "ci_new_stop"))
    assert [c["task_id"] for c in fresh[CHILD_FIELD]["children"]] == ["kid-b"] and "gap" not in fresh[CHILD_FIELD]
    write_task_result(tmp_path, "empty-unknown", "failed")
    osr.capture_receipt(tmp_path, "empty-unknown", {}, gap="descendant_enumeration_unavailable",
                        stop=("empty-unknown", "ci_u"))
    unknown_doc = _document(tmp_path, "empty-unknown")
    assert "This work kept no saved text." not in unknown_doc
    assert "Whether this work kept saved text could not be verified." in unknown_doc
