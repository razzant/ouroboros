"""Which stop the owner pressed rides the terminal of the turn it ended.

Stop now on an in-process turn and Wrap up both end the loop under one reason
(``owner_requested_finalization``); the durable stop intent, still open while the
turn publishes its terminal, is where the command is known. The pipeline carries it
as the existing ``cancel_origin`` with its effective ``stop_policy`` on the final
frame and the stored result, and every reader folds it one way: Stop now reads
Stopped whatever it kept, a Wrap up reads Done over its answer and Stopped without
one, failed file preservation beside Wrap up still reads Failed, and an untyped record
keeps its recorded word. Stored status and axes are never rewritten.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import ouroboros.agent_task_pipeline as pipeline
from ouroboros import cancel_intents
from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION as OWNER_STOP
from ouroboros.project_dialogue import _completion_verdict, completion_status_label, outcome_phase
from ouroboros.task_results import load_task_result, write_task_result
from supervisor.owner_stop import owner_stop_ending, owner_stop_origin

STOPPED_TEXT = "⏹ The owner stopped this chat turn; no further work was done."
WRAP_FALLBACK = ("⚠️ The owner requested finalize-then-stop; no final answer could be "
                 "produced inside the grace window.")
ANSWER = "Found one cause: the cadence test waits on a timer, not on the end of the pass."


def _press(root, task_id, button, *, hardened=False):
    """The intents the app's two buttons record (gateway/tasks.py), before the turn ends."""
    if button == "stop_now":
        return cancel_intents.request_cancel(
            root, task_id, source="http_single", requested_stop_policy="immediate",
            observation={"request_origin": {"kind": "http_client", "source": "http_single"}})
    intent = cancel_intents.request_cancel(
        root, task_id, source="http_graceful", requested_by="owner", reason="owner requested finalize-then-stop",
        requested_stop_policy="finalize_then_cancel")
    if hardened:  # Stop now pressed while the Wrap up still ran: the same request, hardened
        intent = cancel_intents.request_cancel(root, task_id, source="http_single", requested_stop_policy="immediate")
    return intent


def _end_turn(root, monkeypatch, task_id, *, button=None, answer=True, hardened=False, reason=OWNER_STOP,
              forced_text=None):
    """One direct turn that took the loop's owner-stop rail, finished through the real pipeline."""
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)
    write_task_result(root, task_id, "running", chat_id=1, _is_direct_chat=True)
    intent = _press(root, task_id, button, hardened=hardened) if button else {}
    if reason == OWNER_STOP:
        usage = {"rounds": 3, "execution_status": "failed", "reason_code": OWNER_STOP,
                 **({"_best_effort_extracted": True} if answer else {})}
        text = ANSWER if answer else (STOPPED_TEXT if button == "stop_now" else WRAP_FALLBACK)
    else:  # completion won the race: the loop ended on its own final message
        usage, text = {"rounds": 3}, ANSWER
    trace = {"tool_calls": [], "reasoning_notes": []}
    if forced_text is not None:
        from tests.test_delivery_forced_finalization import _forced_test_context
        loop, registry, limit_ctx, trace = _forced_test_context(root)
        assert task_id == "parent1"  # the real forced-loop fixture's admitted task
        monkeypatch.setattr(loop, "_call_forced_model_once", lambda *_a, **_k: forced_text)
        text, usage, trace = loop._handle_forced_finalization(limit_ctx, OWNER_STOP)
        assert usage["terminal_origin"] == ("model_final" if forced_text else "host_notice")
        assert bool(usage.get("_best_effort_extracted")) == bool(forced_text)
    task = {"id": task_id, "type": "task", "chat_id": 1, "_is_direct_chat": True,
            "text": "Check why the cadence test keeps failing on Linux."}
    events = []
    # Stop now's rail marks the turn so no paid post-task synthesis runs (loop_round_limits).
    ctx = SimpleNamespace(_skip_post_task_synthesis=button == "stop_now" and reason == OWNER_STOP and not hardened)
    pipeline.emit_task_results(SimpleNamespace(drive_root=root, repo_dir=root), None, None, events, task, text,
                               usage, trace, 0.0, root / "logs", ctx=ctx)
    if (load_task_result(root, task_id) or {}).get("root_phase_checkpoint"):
        # Every other root's post-task synthesis settles after the answer (stubbed out above).
        write_task_result(root, task_id, (load_task_result(root, task_id) or {})["status"],
                          root_phase_checkpoint={"post_task_synthesis": "completed"})
    stored = load_task_result(root, task_id) or {}
    final = next(event for event in events if event["type"] == "send_message")
    done = next(event for event in events if event["type"] == "task_done")
    return intent, stored, final, done


@pytest.fixture
def root(tmp_path):
    (tmp_path / "logs").mkdir()
    return tmp_path


@pytest.mark.parametrize("projected_partial", [False, True])
def test_terminal_cost_carrier_is_merged_once_before_store(root, monkeypatch, projected_partial):
    # The authoritative carrier may already include the root openness key.
    # This initial own-cost mirror remains partial until the subtree refresh.
    original = pipeline._task_cost_fields
    def fields(env, task):
        return {**original(env, task), "cost_with_children_partial": projected_partial}
    monkeypatch.setattr(pipeline, "_task_cost_fields", fields)
    _, stored, _, _ = _end_turn(root, monkeypatch, "cost-carrier", button="wrap_up")
    assert stored["status"] == "completed" and stored["result"] == ANSWER
    assert stored["cost_with_children_partial"] is True
    assert stored["accounted_upper_bound_usd_with_children"] == stored["accounted_upper_bound_usd"]


@pytest.mark.parametrize("answer, status", [(True, "completed"), (False, "failed")])
def test_stop_now_reads_stopped_whatever_the_turn_kept(root, monkeypatch, answer, status):
    intent, stored, final, done = _end_turn(root, monkeypatch, f"stop-{answer}", button="stop_now", answer=answer)
    # The stored record keeps every raw fact: its status, its execution axis, its result bytes.
    assert stored["status"] == status
    assert stored["outcome_axes"]["execution"]["status"] == ("best_effort" if answer else "failed")
    assert stored["result"] == (ANSWER if answer else STOPPED_TEXT)
    assert stored["reason_code"] == OWNER_STOP
    # ... and the typed stop beside them, the same cancel_origin a settled cancel records.
    origin = stored["cancel_origin"]
    assert origin["source"] == "http_single" and origin["stop_policy"] == "immediate"
    assert origin["request_id"] == intent["request_id"]
    assert origin["request_origin"] == {"kind": "http_client", "source": "http_single"}
    # The live final carries it too (answer first), so its card never shows another word.
    assert final["progress_meta"]["cancel_origin"] == origin
    assert final["progress_meta"]["task_terminal_status"] == status
    assert outcome_phase(stored, done) == outcome_phase(stored, {}) == outcome_phase({}, final["progress_meta"]) == "cancelled"
    assert completion_status_label(stored, done) == "Stopped"
    assert _completion_verdict(stored, done) == "Stopped from the app (Stop now)"


def test_wrap_up_reads_done_over_its_answer_and_stopped_without_one(root, monkeypatch):
    _intent, answered, final, done = _end_turn(root, monkeypatch, "wrap-answer", button="wrap_up", answer=True)
    assert answered["status"] == "completed"
    assert answered["cancel_origin"]["stop_policy"] == "finalize_then_cancel"
    assert final["progress_meta"]["cancel_origin"]["source"] == "http_graceful"
    assert completion_status_label(answered, done) == "Done"
    assert _completion_verdict(answered, done) == ""  # the details' owner marker says it, not a cause
    _intent, empty, _final, done = _end_turn(root, monkeypatch, "wrap-empty", button="wrap_up", answer=False)
    assert empty["status"] == "failed" and empty["result"] == WRAP_FALLBACK
    assert completion_status_label(empty, done) == "Stopped"
    assert _completion_verdict(empty, done) == "Stopped from the app (Wrap up)"


@pytest.mark.parametrize("button,answer", [("wrap_up", True), ("wrap_up", False), ("stop_now", True), ("stop_now", False)])
def test_result_reader_names_the_control_without_claiming_cancellation(root, monkeypatch, button, answer):
    from ouroboros.tools.control_task_results import _get_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256
    from tests._store_shared import result_signature

    _, stored, _, _ = _end_turn(root, monkeypatch, "reader", button=button, answer=answer)
    before = json.dumps(stored, sort_keys=True)
    signature = result_signature(root, "reader")
    ctx = SimpleNamespace(drive_root=root, task_metadata={})
    full = _get_task_result(ctx, "reader")
    conditional = _get_task_result(ctx, "reader", known_result_sha256=_child_result_sha256(stored))
    marker = f"[OWNER_CONTROL] command={button} {json.dumps(stored['cancel_origin'], ensure_ascii=False)}"
    assert stored["result"] in full
    for read in (full, conditional):
        assert marker in read and "[CANCELLED_BY]" not in read
    assert '"result_unchanged": true' in conditional
    assert json.dumps(load_task_result(root, "reader"), sort_keys=True) == before
    assert result_signature(root, "reader") == signature


def test_an_independent_failure_beside_a_wrap_up_still_reads_failed(root, monkeypatch):
    _intent, stored, _final, done = _end_turn(root, monkeypatch, "wrap-files", button="wrap_up", answer=True)
    # A failed capture as the workspace finalizer records it: bundle, status and axis together.
    broken = {**stored, "artifact_status": "failed", "artifact_bundle": {**stored["artifact_bundle"], "status": "failed"},
              "outcome_axes": {**stored["outcome_axes"], "artifacts": {"status": "failed"}}}
    assert completion_status_label(broken, done) == "Failed"
    assert _completion_verdict(broken, done) == "Some result files are missing or could not be preserved."
    unanswered = {**broken, "status": "failed", "outcome_axes": {**broken["outcome_axes"], "execution": {
        "status": "failed", "reason_code": OWNER_STOP}}}
    assert completion_status_label(unanswered, {}) == "Failed"
    assert _completion_verdict(unanswered, {}) == (
        "Stopped from the app (Wrap up) · Some result files are missing or could not be preserved.")


def test_stop_now_pressed_during_a_wrap_up_ends_as_the_stop_now_it_became(root, monkeypatch):
    _intent, stored, _final, done = _end_turn(root, monkeypatch, "hardened", button="wrap_up", answer=True, hardened=True)
    assert stored["cancel_origin"]["stop_policy"] == "immediate"
    assert owner_stop_ending(stored) == ("stop_now", "stop")
    assert completion_status_label(stored, done) == "Stopped"
    assert stored["cancel_origin"]["source"] == "http_graceful", "the original request remains recorded"
    assert _completion_verdict(stored, done) == "Stopped from the app (Stop now)"


@pytest.mark.parametrize("text", [ANSWER, "⚠️ " + ANSWER, "❌ " + ANSWER, ""])
@pytest.mark.parametrize("hardened", [False, True])
def test_forced_producer_saved_answer_uses_provenance_not_its_prefix(root, monkeypatch, text, hardened):
    _intent, stored, final, done = _end_turn(
        root, monkeypatch, "parent1", button="wrap_up", hardened=hardened, forced_text=text)
    assert stored["result"] == (text or WRAP_FALLBACK)
    assert stored["terminal_origin"] == ("model_final" if text else "host_notice")
    expected = "Stopped" if hardened or not text else "Done"
    for row in (stored, final["progress_meta"]):
        assert completion_status_label(row, {}) == expected
    # The worker event is internal; supervisor publishes it with the stored origin.
    assert completion_status_label(stored, done) == expected
    assert stored["outcome_axes"]["execution"]["status"] == ("best_effort" if text else "failed")
    if hardened:
        assert _completion_verdict(stored, {}) == "Stopped from the app (Stop now)"


@pytest.mark.parametrize("prefix", ["", "⚠️ ", "❌ "])
@pytest.mark.parametrize("status,reason", [("infra_failed", OWNER_STOP), ("infra_failed", "provider_failure"),
                                          ("failed", "provider_unavailable")])
def test_answer_provenance_never_overrides_an_independent_execution_failure(prefix, status, reason):
    from ouroboros.outcomes import derive_loop_outcome

    outcome = derive_loop_outcome(prefix + ANSWER, {"execution_status": status, "reason_code": reason,
        "terminal_origin": "model_final", "_best_effort_extracted": True}, {})
    assert outcome["outcome_axes"]["execution"]["status"] == status
    record = {"status": "failed", "cancel_origin": {"source": "http_graceful", "stop_policy": "finalize_then_cancel"},
              **outcome}
    assert completion_status_label(record, {}) == "Failed"


@pytest.mark.parametrize("answer, word", [(True, "Done"), (False, "Failed")])
def test_without_a_recorded_stop_the_turn_keeps_its_earlier_reading(root, monkeypatch, answer, word):
    # A headless finalize_now or an unreadable intent: no button is invented.
    _intent, stored, final, done = _end_turn(root, monkeypatch, f"untyped-{answer}", button=None, answer=answer)
    assert "cancel_origin" not in stored and "cancel_origin" not in final["progress_meta"]
    assert owner_stop_ending(stored) == ("", "summary")
    assert completion_status_label(stored, done) == word


def test_a_turn_that_finished_on_its_own_carries_no_stop(root, monkeypatch):
    # Completion won: the loop ended on its final message while a Stop was still open.
    _intent, stored, final, done = _end_turn(root, monkeypatch, "won", button="stop_now", reason="final_message")
    assert "cancel_origin" not in stored and "cancel_origin" not in final.get("progress_meta", {})
    assert completion_status_label(stored, done) == "Done"


def test_owner_stop_origin_answers_nothing_it_cannot_read(root, tmp_path):
    assert owner_stop_origin(root, "no-intent") == {}
    (root / "state").mkdir(exist_ok=True)
    (root / "state" / "cancel_intents.json").write_text("{not json", encoding="utf-8")
    assert owner_stop_origin(root, "any") == {}


def test_history_task_detail_and_terminal_row_say_the_same_word(root, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.tasks import api_task_get
    from ouroboros.terminal_projection import _project_row

    _intent, stored, final, _done = _end_turn(root, monkeypatch, "consumers", button="stop_now", answer=True)
    # The durable terminal row the host writes into the room (and Main's replay authority).
    row = _project_row("consumers", {"id": "consumers", **stored}, {}, {"chat_id": 1})
    assert (row["outcome"], row["outcome_phase"]) == ("Stopped", "cancelled")
    assert row["text"].startswith("Stopped. Root task consumers. Stopped from the app (Stop now)")
    # Task detail (the card's replay source) returns the stored fact unchanged.
    app = Starlette(routes=[Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"])])
    app.state.drive_root = root
    detail = TestClient(app).get("/api/tasks/consumers").json()
    assert detail["status"] == "completed" and detail["cancel_origin"]["stop_policy"] == "immediate"
    # History attaches the persisted result's truth to the final row it replays.
    (root / "logs" / "chat.jsonl").write_text(json.dumps({
        "ts": "2026-10-25T11:00:10Z", "direction": "out", "chat_id": 1, "task_id": "consumers", "text": ANSWER,
        "format": "markdown", "task_terminal_status": final["progress_meta"]["task_terminal_status"]}) + "\n",
        encoding="utf-8")
    (root / "logs" / "progress.jsonl").write_text("", encoding="utf-8")
    endpoint = make_chat_history_endpoint(root)
    payload = json.loads(asyncio.run(endpoint(SimpleNamespace(query_params={"chat_id": "1"}))).body)
    replayed = next(message for message in payload["messages"] if message.get("task_id") == "consumers")
    assert replayed["outcome_phase"] == "cancelled"
    assert replayed["cancel_origin"]["stop_policy"] == "immediate"


@pytest.mark.parametrize("answer", [False, True])
def test_compact_terminal_keeps_stop_when_result_read_is_unavailable(root, monkeypatch, answer):
    from pathlib import Path
    import subprocess
    from ouroboros import task_store
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.preflight_node import resolve_node
    from ouroboros.terminal_projection import _project_row

    _, stored, _, _ = _end_turn(root, monkeypatch, "unavailable-stop", button="stop_now", answer=answer)
    row = _project_row("unavailable-stop", stored, {}, {"chat_id": 7})
    from ouroboros import dialogue_store
    dialogue_store.append(root, "chat", row)
    def unavailable(*_a, **_kw):
        raise OSError("test: result store temporarily unavailable")
    monkeypatch.setattr(task_store, "display_documents", unavailable)
    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"chat_id": "7"})))
    replayed = next(r for r in json.loads(response.body)["messages"] if r.get("task_id") == "unavailable-stop")
    assert replayed["cancel_origin"] == stored["cancel_origin"]
    node = resolve_node()
    assert node
    script = """import {taskOutcomeSeverity} from './web/modules/log_events.js';
let text=''; for await (const chunk of process.stdin) text += chunk;
console.log(taskOutcomeSeverity(JSON.parse(text)));"""
    checked = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(replayed),
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    assert checked.stdout.strip() == "cancelled"
    assert load_task_result(root, "unavailable-stop") == stored


def test_the_census_and_the_work_row_read_the_same_stop(root, monkeypatch):
    from ouroboros.gateway import work_status

    _intent, empty, _final, _done = _end_turn(root, monkeypatch, "census", button="wrap_up", answer=False)
    # A Wrap up without an answer whose post-task work still runs: the row says Stopped, not Failed.
    assert work_status.shown_outcome(empty) == "cancelled"
    assert work_status.shown_outcome({**empty, "cancel_origin": None}) == "failed"
    assert work_status.shown_outcome({"status": "failed", "outcome_axes": {"lifecycle": {"status": "failed"}}}) == "failed"
    # The run-liveness reader keeps reading the stored lifecycle.
    assert work_status._outcome(empty) == "failed"
