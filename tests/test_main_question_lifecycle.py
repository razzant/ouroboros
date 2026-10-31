"""Main question authority survives terminality, delivery failure and stale routing."""
from types import SimpleNamespace

import pytest

from ouroboros.owner_quiz import (
    current_question, mark_discharged, quiz_needs_retention, quiz_states,
    reconcile_terminal, record_answered, record_asked, record_closed, record_publication,
)
from tests.test_quiz_answer import _decision_app, _post, _tool_ctx
from tests import _dialogue as dialogue_test


def ask(root, qid="q", task="t"):
    return record_asked(root, task, quiz_id=qid, question="Which approach?",
                        options=["First", "Second"], assumption="Research meanwhile", chat_id=1)


def test_all_accepted_expired_and_undischarged_closed_survive_capacity(tmp_path):
    for index in range(20):
        ask(tmp_path, f"expired-{index}")
        reconcile_terminal(tmp_path, "t")
        ask(tmp_path, f"answer-{index}")
        record_answered(tmp_path, "t", quiz_id=f"answer-{index}", option_index=0, request_id=str(index))
        ask(tmp_path, f"closed-{index}")
        record_closed(tmp_path, "t", quiz_id=f"closed-{index}", reason="Superseded by measured result", actor_task_id="t")
    blocks = quiz_states(tmp_path, "t")
    assert len(blocks) == 60
    assert all(quiz_needs_retention(block) for block in blocks.values())
    assert sum(current_question(block) for block in blocks.values()) == 20
    late = record_answered(tmp_path, "t", quiz_id="expired-0", option_index=1,
                           request_id="late", allow_expired=True)
    assert late["ok"] and late["block"]["answered_after_terminal"]
    for index in range(20):
        mark_discharged(tmp_path, "t", f"answer-{index}")
        mark_discharged(tmp_path, "t", f"closed-{index}")
    assert all(f"expired-{index}" in quiz_states(tmp_path, "t") for index in range(20))


def test_model_close_is_not_answer_or_approval_and_requires_own_lineage(tmp_path):
    ask(tmp_path)
    assert record_closed(tmp_path, "t", quiz_id="q", reason="", actor_task_id="t")["error"] == "close_reason_required"
    assert record_closed(tmp_path, "t", quiz_id="q", reason="Retired", actor_task_id="other")["error"] == "question_not_owned"
    assert current_question(quiz_states(tmp_path, "t")["q"])
    closed = record_closed(tmp_path, "t", quiz_id="q", reason="The plan was withdrawn", actor_task_id="t")["block"]
    assert closed["state"] == "closed" and closed["closed_reason"] == "The plan was withdrawn"
    assert "answered_index" not in closed and "answered_at" not in closed
    assert quiz_needs_retention(closed) and not current_question(closed)
    assert record_answered(tmp_path, "t", quiz_id="q", option_index=0, request_id="owner", allow_expired=True)["error"] == "quiz_closed"
    retry = record_closed(tmp_path, "t", quiz_id="q", reason="Changed text", actor_task_id="t")
    assert retry["duplicate"] and retry["block"]["closed_reason"] == "The plan was withdrawn"
    ask(tmp_path, "already-answered")
    record_answered(tmp_path, "t", quiz_id="already-answered", option_index=1, request_id="owner-answer")
    assert record_closed(tmp_path, "t", quiz_id="already-answered", reason="Too late", actor_task_id="t")["error"] == "quiz_already_answered"


def test_recorded_continue_can_close_predecessor_question(tmp_path):
    from ouroboros.owner_continue import binding_sha
    from ouroboros.task_results import write_task_result

    ask(tmp_path)
    binding = {"successor_task_id": "continued"}
    write_task_result(tmp_path, "t", "completed", continued_by={
        "state": "admitted", "successor_task_id": "continued", "binding": binding,
        "binding_sha256": binding_sha(binding),
    })
    ctx = _tool_ctx(tmp_path, task_id="continued")
    from ouroboros.tools.core_artifacts import _escalate

    receipt = _escalate(ctx, action="close", quiz_id="q", question_task_id="t", reason="The owner changed the goal")
    assert receipt.startswith("OK:")
    assert ctx.pending_events[0]["type"] == "send_quiz_state"
    assert ctx.pending_events[0]["task_id"] == "t"
    assert quiz_states(tmp_path, "t")["q"]["closed_by_task_id"] == "continued"


@pytest.mark.parametrize("retry_from,role,allowed", [("t", "root", True), ("other", "root", False), ("t", "subagent", False)])
def test_close_uses_validated_canonical_retry_lineage_not_root_equality(tmp_path, retry_from, role, allowed):
    from ouroboros.task_results import write_task_result

    ask(tmp_path)
    write_task_result(tmp_path, "retry", "running", root_task_id="t", parent_task_id="",
                      delegation_role=role, original_task_id="t", timeout_retry_from=retry_from)
    result = record_closed(tmp_path, "t", quiz_id="q", reason="The replacement run retired this branch", actor_task_id="retry")
    assert result["ok"] is allowed
    block = quiz_states(tmp_path, "t")["q"]
    assert (block["state"] == "closed") is allowed
    if not allowed:
        assert result["error"] == "question_not_owned"


@pytest.mark.parametrize("words", ["", "   "])
def test_the_owner_closes_in_one_tap_while_a_model_still_explains(tmp_path, words):
    from ouroboros.owner_quiz import OWNER_CLOSE_REASON

    ask(tmp_path, "tapped")
    ask(tmp_path, "typed")
    assert record_closed(tmp_path, "t", quiz_id="tapped", reason=words, actor_task_id="t")["error"] == "close_reason_required"
    assert current_question(quiz_states(tmp_path, "t")["tapped"])
    tapped = record_closed(tmp_path, "t", quiz_id="tapped", reason=words, actor="owner", request_id="tap")["block"]
    assert tapped["closed_reason"] == OWNER_CLOSE_REASON == "Closed by the owner without an answer."
    assert tapped["closed_actor"] == "owner" and "closed_by_task_id" not in tapped
    assert "answered_index" not in tapped and not current_question(tapped)
    typed = record_closed(tmp_path, "t", quiz_id="typed", reason=" No longer needed ", actor="owner", request_id="typed")
    assert typed["block"]["closed_reason"] == " No longer needed "  # the owner's own words stay verbatim
    retry = record_closed(tmp_path, "t", quiz_id="tapped", reason="Different words", actor="owner", request_id="tap")
    assert retry["duplicate"] and retry["block"]["closed_reason"] == OWNER_CLOSE_REASON


@pytest.mark.parametrize("finished", [True, False])
def test_a_model_closes_another_tasks_question_only_once_that_task_finished(tmp_path, finished):
    ask(tmp_path, task="author")
    if finished:
        reconcile_terminal(tmp_path, "author")
    outcome = record_closed(tmp_path, "author", quiz_id="q", reason="The shipped release settled it", actor_task_id="later")
    block = quiz_states(tmp_path, "author")["q"]
    if not finished:  # a live task's question stays its own line's or the owner's
        assert outcome == {"ok": False, "error": "question_not_owned", "state": "open", "block": block}
        assert current_question(block) and "closed_reason" not in block
        return
    assert outcome["ok"] and block["state"] == "closed" and block["closed_actor"] == "model"
    assert block["closed_reason"] == "The shipped release settled it" and block["closed_by_task_id"] == "later"
    retry = record_closed(tmp_path, "author", quiz_id="q", reason="Other words", actor_task_id="later")
    assert retry["duplicate"] and retry["block"]["closed_reason"] == "The shipped release settled it"
    assert record_closed(tmp_path, "author", quiz_id="q", reason="Mine too", actor_task_id="third")["error"] == "quiz_closed"


@pytest.mark.parametrize("caller", ["root", "child"])
def test_only_an_active_root_closes_a_finished_tasks_question_and_gains_no_material_authority(tmp_path, caller):
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.core_artifacts import _escalate

    write_task_result(tmp_path, "finished", "completed", chat_id=1)
    store_task_artifact_bytes(tmp_path, "finished", name="evidence.txt", data=b"Finished evidence")
    ask(tmp_path, task="finished")
    reconcile_terminal(tmp_path, "finished")
    ctx = _tool_ctx(tmp_path, task_id="later", parent="" if caller == "root" else "root-1")
    receipt = _escalate(ctx, action="close", quiz_id="q", question_task_id="finished", reason="The report answered it")
    block = quiz_states(tmp_path, "finished")["q"]
    if caller == "child":
        assert "QUIZ_CLOSE_UNAVAILABLE" in receipt and block["state"] == "expired_terminal"
        assert not ctx.pending_events
        return
    assert receipt.startswith("OK:") and block["closed_by_task_id"] == "later"
    assert ctx.pending_events[0]["type"] == "send_quiz_state" and ctx.pending_events[0]["task_id"] == "finished"
    # Closing that question is not a claim on the finished task's line: its files stay outside materials.
    refused = _escalate(ctx, question="Proceed?", assumption="Research",
                        materials=[{"task_id": "finished", "name": "evidence.txt"}])
    assert "QUIZ_MATERIAL_INVALID" in refused and not quiz_states(tmp_path, "later")


def test_explicit_close_in_same_tool_batch_cancels_only_its_pending_wait(tmp_path):
    from ouroboros.tools.core_artifacts import _escalate

    ctx = _tool_ctx(tmp_path)
    ctx.owner_wait_callback = lambda *args: None
    assert _escalate(ctx, question="Proceed?", wait_for_answer=True).startswith("OK:")
    quiz_id = ctx._owner_wait_requested
    assert quiz_id
    assert _escalate(ctx, action="close", quiz_id=quiz_id, reason="The planned action was withdrawn").startswith("OK:")
    assert not ctx._owner_wait_requested
    assert quiz_states(tmp_path, "root-1")[quiz_id]["state"] == "closed"


def test_publication_address_precedes_exposure_and_wins_over_stale_retries(tmp_path, monkeypatch):
    from supervisor import message_bus

    ask(tmp_path)
    bridge = message_bus.LocalChatBridge()
    observed = []
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "publish_event", lambda *args: None)
    def observe(frame):
        block = quiz_states(tmp_path, "t")["q"]
        assert block["publication_chat_id"] == frame["chat_id"] == 77
        row = dialogue_test.chat_rows(tmp_path)[-1]
        assert row["chat_id"] == 77 and row["quiz"]["quiz_id"] == "q"
        observed.append(frame)
    bridge._broadcast_fn = observe
    for requested in (77, 1):
        assert bridge.send_quiz(requested, "q", "Which approach?", ["First", "Second"],
                                assumption="Research meanwhile", task_id="t")[0]
    assert len(observed) == 2
    from ouroboros.gateway.task_decision import _late_answer_destination
    assert _late_answer_destination(tmp_path, "t", quiz_states(tmp_path, "t")["q"]) == (77, "")


def test_quiz_append_failure_exposes_nothing_and_keeps_question(tmp_path, monkeypatch):
    from supervisor import message_bus

    ask(tmp_path)
    frames = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "publish_event", lambda *args: frames.append(args))
    def fail(*args, **kwargs):
        assert kwargs["require_write"] is True
        raise OSError("disk unavailable")
    monkeypatch.setattr(message_bus, "log_chat", fail)
    with pytest.raises(OSError):
        bridge.send_quiz(1, "q", "Which approach?", ["First"], assumption="Research", task_id="t")
    assert not frames and current_question(quiz_states(tmp_path, "t")["q"])


@pytest.mark.parametrize("closed", [False, True])
def test_stale_quiz_delivery_replays_canonical_terminal_choice_and_explanation(tmp_path, monkeypatch, closed):
    from supervisor import message_bus

    ask(tmp_path)
    if closed:
        record_closed(tmp_path, "t", quiz_id="q", reason="The plan was withdrawn", actor_task_id="t")
    else:
        record_answered(tmp_path, "t", quiz_id="q", option_index=1, request_id="owner", comment="Use the measured route")
    frames = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "publish_event", lambda *args: None)
    assert bridge.send_quiz(1, "q", "Stale text", ["Stale option"], assumption="Stale assumption", task_id="t", wait_for_answer=True)[0]
    [frame] = frames
    assert frame["question"] == "Which approach?"
    assert frame["options"] == [{"label": "First"}, {"label": "Second"}]
    if closed:
        assert frame["state"] == "closed" and frame["closed_reason"] == "The plan was withdrawn"
        assert "answered_index" not in frame
    else:
        assert frame["state"] == "answered" and frame["answered_index"] == 1
        assert frame["comment"] == "Use the measured route"


def test_legacy_destination_cannot_infer_publication_from_current_task_binding(tmp_path):
    from ouroboros.gateway.task_decision import _late_answer_destination
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result

    ask(tmp_path)
    project = create_project(tmp_path, "late-route", name="Late route")
    bind_task_to_project(tmp_path, "t", project["id"], origin={"absent": "system"})
    write_task_result(tmp_path, "t", "completed", chat_id=project["chat_id"], project_id=project["id"])
    block = quiz_states(tmp_path, "t")["q"]
    assert block["chat_id"] == 1 and "publication_chat_id" not in block
    assert _late_answer_destination(tmp_path, "t", block) == (None, "publication_address_unknown")
    # A proven publication remains the answer destination after conversion.
    block = record_publication(tmp_path, "t", "q", 1)
    assert _late_answer_destination(tmp_path, "t", block) == (1, "")


def test_closure_append_failure_keeps_debt_until_durable_reason(tmp_path, monkeypatch):
    from supervisor import events_chat_delivery, message_bus

    ask(tmp_path)
    record_publication(tmp_path, "t", "q", 1)
    record_closed(tmp_path, "t", quiz_id="q", reason="No longer applicable", actor_task_id="t")
    frames = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(send_quiz_state=lambda **kw: frames.append(kw)))
    event = {"task_id": "t", "quiz_id": "q", "type": "send_quiz_state"}
    real_log = message_bus.log_chat
    monkeypatch.setattr(message_bus, "log_chat", lambda *a, **kw: (_ for _ in ()).throw(OSError("full disk")))
    with pytest.raises(OSError):
        events_chat_delivery._handle_send_quiz_state(event, ctx)
    assert not frames and quiz_needs_retention(quiz_states(tmp_path, "t")["q"])
    monkeypatch.setattr(message_bus, "log_chat", real_log)
    from ouroboros.owner_quiz import _mutate_projection

    _mutate_projection(tmp_path, "t", lambda quizzes: quizzes["q"]["delivery"].update(next_retry_at=""))
    events_chat_delivery._handle_send_quiz_state(event, ctx)
    assert not quiz_needs_retention(quiz_states(tmp_path, "t")["q"])
    assert frames[0]["closed_reason"] == "No longer applicable"


def test_answer_history_failure_retains_canonical_answer_across_more_questions(tmp_path, monkeypatch):
    from ouroboros.gateway import task_decision

    ask(tmp_path)
    app = _decision_app(tmp_path, monkeypatch, live_task={"id": "t"})
    monkeypatch.setattr(task_decision, "_record_quiz_answer_history", lambda *a, **k: (_ for _ in ()).throw(OSError("full disk")))
    reply = _post(app, {"request_id": "first", "decision_id": "quiz:t:q", "option_index": 1})
    assert reply.status_code == 503
    for index in range(20):
        ask(tmp_path, f"more-{index}")
        record_answered(tmp_path, "t", quiz_id=f"more-{index}", option_index=0, request_id=f"more-{index}")
    assert quiz_states(tmp_path, "t")["q"]["answered_index"] == 1
    assert quiz_needs_retention(quiz_states(tmp_path, "t")["q"])


def test_late_answer_accepted_metadata_rebuilds_full_frame_after_canonical_gc(tmp_path, monkeypatch):
    from ouroboros.owner_quiz import late_answer_model_text, late_answer_ref
    from tests.test_quiz_late_answer import _inbox, _late_bridge

    for index in range(20):
        ask(tmp_path, f"q{index}")
        reconcile_terminal(tmp_path, "t")
    bridge, _frames = _late_bridge(tmp_path, monkeypatch)
    app = _decision_app(tmp_path, monkeypatch, live_task=None)
    reply = _post(app, {"request_id": "late", "decision_id": "quiz:t:q0", "option_index": 1})
    assert reply.status_code == 200 and reply.json()["forwarded"]
    assert "q0" not in quiz_states(tmp_path, "t"), "Discharged history may leave the canonical hot set"
    [queued] = _inbox(bridge)
    ref = late_answer_ref(queued["task_metadata"]["late_answer"])
    frame = late_answer_model_text(tmp_path, ref, queued["text"])
    assert "Question was: Which approach?" in frame
    assert "The owner chose option 2: Second" in frame
    assert "Research meanwhile" in frame and "could not be read" not in frame
    # Snapshot identity cannot silently point at another question or task.
    mismatched = {**ref, "task_id": "other"}
    assert "answer_snapshot" not in late_answer_ref(mismatched)


def test_materials_capture_explicit_registered_bytes_before_ask(tmp_path):
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.tools.core_artifacts import _escalate

    ctx = _tool_ctx(tmp_path)
    stored = store_task_artifact_bytes(tmp_path, "root-1", name="choice.txt", data=b"Exact decision evidence")
    receipt = _escalate(ctx, question="Choose?", options=["A"], assumption="Research", materials=[{"name": "choice.txt"}])
    assert receipt.startswith("OK:")
    [block] = quiz_states(tmp_path, "root-1").values()
    [material] = block["materials"]
    assert material["sha256"] == stored["sha256"] and material["label"] == "choice.txt"
    assert material["download_url"].startswith("/api/tasks/root-1/artifacts/")
    assert ctx.pending_events[0]["materials"] == block["materials"]


def test_material_reference_uses_descendant_inventory_and_rejects_forged_path(tmp_path):
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.core_artifacts import _escalate

    ctx = _tool_ctx(tmp_path)
    write_task_result(tmp_path, "child", "completed", parent_task_id="root-1")
    store_task_artifact_bytes(tmp_path, "child", name="chart.txt", data=b"Measured evidence")
    assert _escalate(ctx, question="Proceed?", assumption="Research", materials=[{"task_id": "child", "name": "chart.txt"}]).startswith("OK:")
    [block] = quiz_states(tmp_path, "root-1").values()
    assert block["materials"][0]["task_id"] == "root-1", "Selected bytes move into question custody"
    outside = tmp_path / "unregistered.txt"
    outside.write_text("not in the task's artifact store")
    write_task_result(tmp_path, "child", "completed", artifacts=[{"name": "unregistered.txt", "path": str(outside), "status": "ready"}])
    refused = _escalate(ctx, question="Proceed?", assumption="Research", materials=[{"task_id": "child", "name": "unregistered.txt"}])
    assert "QUIZ_MATERIAL_INVALID" in refused
    assert len(quiz_states(tmp_path, "root-1")) == 1


def test_material_reference_refuses_symlinked_store_authority(tmp_path):
    from ouroboros.artifacts import store_task_artifact_bytes, task_artifact_dir_path
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.core_artifacts import _escalate

    ctx = _tool_ctx(tmp_path)
    write_task_result(tmp_path, "child", "completed", parent_task_id="root-1")
    store_task_artifact_bytes(tmp_path, "child", name="chart.txt", data=b"Evidence")
    store = task_artifact_dir_path(tmp_path, "child")
    moved = tmp_path / "outside-artifact-authority"
    store.rename(moved)
    store.symlink_to(moved, target_is_directory=True)
    refused = _escalate(ctx, question="Proceed?", assumption="Research", materials=[{"task_id": "child", "name": "chart.txt"}])
    assert "QUIZ_MATERIAL_INVALID" in refused
    assert not ctx.pending_events and not quiz_states(tmp_path, "root-1")


@pytest.mark.parametrize("reference", [{"name": "../secret"}, {"name": "missing.txt"},
                                      {"name": "choice.txt", "path": "/secret"},
                                      {"task_id": "unrelated", "name": "choice.txt"}])
def test_materials_refuse_unknown_paths_and_unrelated_authority(tmp_path, reference):
    from ouroboros.tools.core_artifacts import _escalate

    ctx = _tool_ctx(tmp_path)
    outcome = _escalate(ctx, question="Choose?", options=[], assumption="Research", materials=[reference])
    assert "QUIZ_MATERIAL_INVALID" in outcome
    assert not ctx.pending_events and not quiz_states(tmp_path, "root-1")
