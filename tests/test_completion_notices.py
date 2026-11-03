"""Completion causes reach the real summary/notifier, without Telegram network I/O.

The records are the ones older tasks stored (an acceptance decision on the review axis,
written before the result review replaced task acceptance); the renderers still read them."""
import asyncio
import copy

import pytest

from ouroboros import outcomes, project_dialogue as dialogue, terminal_projection
from tests.test_telegram_notifications import _api, _load, _Rec, _write_rows


def _reviewed_trace():
    return {
        "review_runs": [{"authority": "host_root", "panel_id": "p1", "binding_hash": "b1",
                         "aggregate_signal": "PASS", "superseded_by_revision": "r2",
                         "actors": [{"signal": "PASS", "parsed": {"verdict": "PASS", "outcome_tier": "solved"}}]}],
        "review_decision": {"panel_id": "p1", "binding_hash": "b1"},
        "acceptance_decision": {"status": "accepted", "reason": "previous_revision_accepted",
                                "reviewer_signal": "PASS", "reviewed_panel_id": "p1"},
        "delivery_candidate": {"acceptance_binding": {"authoritative": False}},
    }


def _stored_review_axes(trace):
    """The review and objective axes an older task stored for this trace.

    Its host acceptance decision applied the earlier revision's PASS panel by name
    (``previous_revision_accepted``). That selection left with task acceptance, so the
    current producer reads the superseded run as a gap; old task cards and notices still
    render the stored record, so the applied panel is restored here."""
    applied = copy.deepcopy(trace)
    for run in applied["review_runs"]:
        run.pop("superseded_by_revision", None)
    review = outcomes._review_axis(applied)
    return review, outcomes._objective_axis(review)


def _record(case):
    """An older task's stored record: the real producer over its trace, with the review axes it stored."""
    trace = _reviewed_trace()
    if case == "delivery_control":
        trace["delivery_candidate"].update(degraded=True, degraded_reason="invalid_delivery_control_after_repair")
    elif case == "verification_output":
        trace["verification_events"] = [{"kind": "test", "services": [{"name": "verification", "artifact_output_failed": True}]}]
    elif case == "partial":
        trace["review_runs"][0]["actors"][0]["parsed"]["outcome_tier"] = "best_effort"
    outcome = outcomes.derive_loop_outcome("The answer is retained.", {}, trace)
    review, objective = _stored_review_axes(trace)
    outcome["outcome_axes"].update(review=review, objective=objective)
    outcome.update(review_eligibility=review["eligibility"], review_trigger=review["trigger"])
    record = {**outcome, "task_id": case, "status": "completed", "chat_id": 1}
    if case in {"receipt", "mixed", "unknown_warning"}:
        objective = record["outcome_axes"]["objective"]
        outcomes._merge_objective_warning(objective, "receipt_absent")
        if case != "receipt":
            outcomes._merge_objective_warning(objective, "expected_output_ungrounded" if case == "mixed" else "future_warning")
    if case == "artifact_missing":
        record["outcome_axes"]["artifacts"] = {"status": "missing"}
    return record


@pytest.mark.parametrize("case,fragment", [
    ("clean", "finished"), ("receipt", "finished"),
    ("delivery_control", "final delivery instruction could not be read"),
    ("verification_output", "A tool this task used failed"),
    ("partial", "The result is partial"),
    ("mixed", "declared output has no recorded tool work"),
    ("unknown_warning", "future_warning"),
    ("artifact_missing", "result files are missing"),
    ("unknown_delivery", "Delivery of the final answer could not be confirmed"),
])
def test_actual_outcome_summary_and_phone(case, fragment, tmp_path, monkeypatch):
    nt = _load()
    api, data = _api(tmp_path)
    monkeypatch.setattr(nt, "TelegramClient", _Rec)
    record = _record(case)
    before = copy.deepcopy(record)
    summary = terminal_projection._project_row(case, record, {}, {"chat_id": 1},
        delivery={"state": "unknown"} if case == "unknown_delivery" else None)
    assert record == before  # presentation never changes health or independent evidence
    if case in {"clean", "receipt"}:
        assert summary["completion_notice"]["neutral"] is True
        assert "earlier version" in summary["reason_detail"]
    else:
        assert summary["completion_notice"]["neutral"] is False
        assert fragment in summary["reason_detail"]
    if case == "receipt":
        assert summary["outcome_phase"] == "warn"  # notification calm is not a green verdict
        assert "No host verification receipt" in summary["reason_detail"]
    if case == "mixed":
        assert "receipt_absent" in summary["outcome_axes"]["objective"]["warnings"]
    _write_rows(data, [summary])
    _Rec.sent = []
    disabled = {"notified_task_ids": []}
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "off"}, 42, disabled, "en"))
    assert not _Rec.sent and disabled["notified_task_ids"] == [case]
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, disabled, "en"))
    assert not _Rec.sent
    asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "en"))
    assert len(_Rec.sent) == 1
    message = _Rec.sent[0][1]
    assert fragment in message
    assert "earlier version" not in message
    assert "No host verification receipt" not in message
    if case in {"clean", "receipt"}:
        assert "⚠" not in message and "✅" not in message and "done" not in message


@pytest.mark.parametrize("trigger,enforcement,obligations,expected_neutral", [
    ("auto_effect", "advisory", [], True),
    ("required_effect", "advisory", [], False),
    ("auto_effect", "blocking", [], False),
    ("auto_effect", "advisory", ["unmet-review"], False),
    ("", "advisory", [], False),
    ("auto_effect", "", [], False),
])
def test_recorded_author_completion_and_review_obligations(trigger, enforcement, obligations, expected_neutral):
    record = _record("review")
    axes = record["outcome_axes"]
    axes["objective"] = {"status": "pass", "source": "author_acceptance", "reason": "author_finish"}
    axes["review"] = {"status": "degraded", "trigger": trigger, "acceptance_decision": {
        "status": "finalized_unaccepted", "reason": "author_finish", "enforcement": enforcement,
        "open_obligations": obligations}}
    before = copy.deepcopy(record)
    notice = terminal_projection.completion_notice(record)
    neutral, detail = notice["neutral"], notice["detail"]
    assert neutral is expected_neutral
    assert bool(detail) is not expected_neutral
    assert record == before
    assert "had not signed it off" in dialogue._completion_verdict(record, {})


def test_unavailable_review_without_an_author_completion_is_not_guessed_optional():
    record = _record("review")
    record["outcome_axes"]["review"] = {"status": "degraded", "trigger": "auto_effect", "acceptance_decision": {
        "status": "finalized_unaccepted", "reason": "infra_failure", "enforcement": "advisory"}}
    record["outcome_axes"]["objective"] = {"status": "degraded", "source": "task_acceptance_review"}
    notice = terminal_projection.completion_notice(record)
    neutral, detail = notice["neutral"], notice["detail"]
    assert not neutral and "infrastructure failure" in detail


def test_cancellation_and_custody_are_projected_before_the_summary_drops_them():
    record = _record("cancel")
    record.update(status="cancelled", cancel_origin={"source": "owner_restart", "reason": "Owner restart"})
    record["outcome_axes"]["lifecycle"]["status"] = "cancelled"
    summary = terminal_projection._project_row("cancel", record, {}, {"chat_id": 1})
    assert summary["completion_notice"] == {"neutral": False, "detail": "Stopped by the Restart command.", "phase": "cancelled"}
    record = _record("custody")
    record.update(reason_code="delegated_custody_unreconciled", delegated_runs_unreconciled=["run-one"])
    summary = terminal_projection._project_row("custody", record, {}, {"chat_id": 1})
    assert "delegated work was never reconciled" in summary["completion_notice"]["detail"]
    assert summary["completion_notice"]["neutral"] is False


def test_bypass_and_rail_are_one_cause_beside_independent_limitations():
    record = _record("limits")
    record.update(reason_code="round_limit", terminal_plan_review_open=True)
    axes = record["outcome_axes"]
    axes["execution"].update(status="best_effort", reason_code="round_limit")
    axes["objective"].update(status="best_effort", deferred_count=1)
    axes["review"]["acceptance_decision"] = {"status": "finalized_unaccepted", "reason": "acceptance_bypassed_round_limit"}
    detail = dialogue._completion_verdict(record, {})
    assert detail.count("round limit") == 1
    assert detail.count("sub-task results were deferred") == 1
    assert detail.count("plan review was still open") == 1


def test_earlier_approval_does_not_hide_current_open_obligations():
    record = _record("open-debt")
    decision = record["outcome_axes"]["review"]["acceptance_decision"]
    decision["open_obligations"] = ["current-review-request"]
    summary = terminal_projection._project_row("open-debt", record, {}, {"chat_id": 1})
    assert not summary["completion_notice"]["neutral"]
    assert "reviewer requests still open" in summary["completion_notice"]["detail"]
    assert "earlier version" in summary["reason_detail"]


@pytest.mark.parametrize("notify", ["off", "on"])
@pytest.mark.parametrize("mirror", ["all", "telegram_only"])
def test_files_questions_and_project_replies_are_independent(tmp_path, monkeypatch, notify, mirror):
    import base64
    import json
    from tests.test_telegram_format_parity import _load_skill, _Client, _Api
    from tests.test_telegram_quiz_answers import _EVENT, Client as QuizClient
    from tests.test_telegram_subagent_cards import _setup, _Rec as ReplyClient

    plugin, api = _setup(tmp_path, monkeypatch, TELEGRAM_NOTIFY_TASKS=notify, TELEGRAM_MIRROR_MODE=mirror)
    event = {"text": "The project's complete answer", "is_progress": False, "chat_id": 42,
             "project_id": "project", "transport": {"kind": "telegram"}}
    asyncio.run(plugin._make_outbound(api)(event))
    assert ReplyClient.sent and "complete answer" in ReplyClient.sent[0][1]
    media_plugin, _ = _load_skill()
    settings = {"TELEGRAM_CHAT_ID": "42", "TELEGRAM_NOTIFY_TASKS": notify, "TELEGRAM_MIRROR_MODE": mirror}
    (tmp_path / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    client = _Client()
    monkeypatch.setattr(media_plugin, "TelegramClient", lambda *_a, **_kw: client)
    asyncio.run(media_plugin._make_document(_Api(tmp_path))({"chat_id": 42,
        "transport": {"kind": "telegram"}, "file_base64": base64.b64encode(b"report").decode(),
        "filename": "report.txt", "mime": "text/plain"}))
    assert len(client.documents) == 1 and client.documents[0][1:3] == (b"report", "report.txt")
    quiz_client = QuizClient("token")
    monkeypatch.setattr(media_plugin, "TelegramClient", lambda *_a, **_kw: quiz_client)
    asyncio.run(media_plugin._make_quiz(_Api(tmp_path))({**_EVENT, "transport": {"kind": "telegram"}, "wait_for_answer": True}))
    assert len(quiz_client.panels) == 1 and "Which db?" in quiz_client.panels[0][1]


@pytest.mark.parametrize("lane", ["periodic", "mirror"])
def test_degraded_review_history_never_leaves_a_bare_warning(lane, tmp_path, monkeypatch):
    # Independent Claude reproduction: a retained DEGRADED wave precedes the
    # accepted earlier PASS. Filtering provenance must not erase the only reason
    # when the derived objective still prevents a positively neutral phone view.
    trace = _reviewed_trace()
    trace["review_runs"].insert(0, {"authority": "host_root", "panel_id": "p0", "binding_hash": "b1",
        "aggregate_signal": "DEGRADED", "degraded": True,
        "actors": [{"signal": "DEGRADED", "parsed": {"verdict": "DEGRADED"}}]})
    record = {**outcomes.derive_loop_outcome("The answer is retained.", {}, trace),
              "task_id": "degraded-review", "status": "completed", "chat_id": 1}
    before = copy.deepcopy(record)
    if lane == "periodic":
        nt = _load()
        api, data = _api(tmp_path)
        monkeypatch.setattr(nt, "TelegramClient", _Rec)
        summary = terminal_projection._project_row("degraded-review", record, {}, {"chat_id": 1})
        notice = summary["completion_notice"]
        _write_rows(data, [summary])
        _Rec.sent = []
        asyncio.run(nt._check_tasks_notify(api, {"TELEGRAM_NOTIFY_TASKS": "on"}, 42, {"notified_task_ids": []}, "en"))
        sent = _Rec.sent[0][1]
    else:
        from tests.test_project_completion_mirror import _enqueue_row
        from tests.test_completion_legacy_replay import _legacy_sender
        from tests.test_telegram_subagent_cards import _Rec as MirrorRec

        event = _enqueue_row(tmp_path / "host", monkeypatch, {k: v for k, v in record.items() if k != "task_id"})
        notice = event["progress_meta"]["completion_notice"]
        send, _frames = _legacy_sender(tmp_path / "host", monkeypatch)
        send(event)
        sent = MirrorRec.sent[0][1]
    assert record == before
    assert not notice["neutral"]  # do not invent finished/approved from degraded facts
    assert notice["detail"] and "earlier version" in notice["detail"], sent
    assert "earlier version" in sent
