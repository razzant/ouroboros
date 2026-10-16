"""Control-authority refusals retain the review's physical facts and paid draft."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.review_custody import _ReviewAttemptHistory, _review_exception_projection
from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
from ouroboros.review_substrate import run_review_request
from tests.test_native_tool_round_executor import _assignment, _ScriptedLLM, _tool_call, subject_repo as _subject_repo

subject_repo = _subject_repo  # Register the shared pytest fixture.


@pytest.mark.parametrize("cause", ["dispatch_publication_unconfirmed", "dispatch_document_unreadable"])
def test_control_refusal_reaches_the_real_slot_as_not_dispatched(subject_repo, tmp_path, monkeypatch, cause):
    calls = []

    class Refuses(_ScriptedLLM):
        def chat(self, **kwargs):
            calls.append("admission")
            raise ua.ControlAuthorityUnavailable("dispatch authority could not be read", cause=cause)

    llm = Refuses([])
    assignment = _assignment(subject_repo, llm)
    monkeypatch.setattr("ouroboros.review_substrate._review_route_executor",
                        lambda assignment, **kw: NativeToolRoundReviewExecutor(assignment, llm=llm))
    result = run_review_request(assignment.request, slots=[assignment.slot], drive_root=tmp_path,
                               usage_ctx=SimpleNamespace())
    actor = result.actors[0]
    assert calls == ["admission"]
    assert actor["status"] == "error"
    assert actor["operation_state"] == "not_dispatched"
    assert actor["failure_code"] == cause
    assert actor["usage"]["native_rounds"] == 0
    assert actor["usage"]["native_end_reason"] == "control_authority_unavailable"
    assert actor["usage"]["review_failure_phase"] == "admission"
    assert not actor.get("late_result_pending")


@pytest.mark.parametrize("known_outcome", [False, True])
def test_control_refusal_keeps_paid_custody_and_unknown_is_never_resent(subject_repo, tmp_path, monkeypatch, known_outcome):
    admissions, sends = [], []

    class Provider503(RuntimeError):
        status_code = 503

    class FailsThenRefuses(_ScriptedLLM):
        def chat(self, **kwargs):
            admissions.append("admission")
            if len(admissions) > 1:
                raise ua.ControlAuthorityUnavailable("unreadable fence", cause="dispatch_document_unreadable")

            def send():
                sends.append("physical send")
                raise Provider503("known terminal response") if known_outcome else TimeoutError("unknown outcome")

            return ua.execute_physical_attempt(ua.AttemptRequest(
                model="test/model", provider="openrouter", reservation_usd=0.01,
                drive_root=tmp_path, task_id="t-native", root_task_id="t-native"), send)

    llm = FailsThenRefuses([])
    assignment = _assignment(subject_repo, llm)
    assignment.request.retry_key = "control-authority:paid:1"
    # A plain API executor exposes the outer retry rail (native episodes retain
    # their own round custody). Only the transport is scripted.
    assignment = replace(assignment, slot=replace(assignment.slot, native_retrieval_override=False))
    from ouroboros.review_execution import ApiChatReviewExecutor
    monkeypatch.setattr("ouroboros.review_substrate._review_route_executor",
                        lambda assignment, **kw: ApiChatReviewExecutor(assignment, llm=llm))
    ctx = SimpleNamespace()
    args = dict(slots=[assignment.slot], drive_root=tmp_path, usage_ctx=ctx)
    first = run_review_request(assignment.request, **args).actors[0]
    if not known_outcome:
        second = run_review_request(assignment.request, **args).actors[0]
        assert second["operation_state"] == "custody_lost"
    assert sends == ["physical send"]
    assert len(admissions) == (2 if known_outcome else 1)
    assert first["operation_state"] == ("settled" if known_outcome else "custody_lost")
    assert first["http_status"] == (503 if known_outcome else None)
    assert first["usage"]["physical_attempt_state"] == "unresolved"
    assert first["failure_code"] == ("dispatch_document_unreadable" if known_outcome else "provider_outcome_unknown")


@pytest.mark.parametrize("prior", ["settled", "unresolved"])
@pytest.mark.parametrize("current", ["", "released"])
def test_control_refusal_cannot_erase_earlier_paid_or_unknown_history(prior, current):
    history = _ReviewAttemptHistory()
    paid = RuntimeError("earlier physical send")
    paid.physical_attempt_capture = SimpleNamespace(state=prior, provider_status_code=None)
    history.observe(paid)
    refused = ua.ControlAuthorityUnavailable("unreadable fence", cause="dispatch_document_unreadable")
    if current:
        refused.physical_attempt_capture = SimpleNamespace(state=current, provider_status_code=None)
    custody, _, status, operation, code = _review_exception_projection(refused, {}, history, {})
    assert custody["physical_attempt_state"] == prior
    assert status is None
    assert operation == ("custody_lost" if prior == "unresolved" else "settled")
    assert code == ("provider_outcome_unknown" if prior == "unresolved" else refused.cause)


@pytest.mark.parametrize("shape", ["report", "verdict"])
@pytest.mark.parametrize("paid_round", [False, True])
def test_native_control_refusal_preserves_only_an_available_report_draft(subject_repo, tmp_path, shape, paid_round):
    draft = "# Paid draft\n\nThe greeting file says hello.\n"
    admissions = []
    error = ua.ControlAuthorityUnavailable("dispatch publication unconfirmed", cause="dispatch_publication_unconfirmed")

    class RefusesAfterDraft(_ScriptedLLM):
        def chat(self, **kwargs):
            admissions.append("admission")
            if not self.script:
                raise error
            message, usage = super().chat(**kwargs)
            return message, {**usage, "cost": 0.125, "ledger_attempt_ids": ["paid-first"]}

    llm = RefusesAfterDraft([{"content": draft, "tool_calls": [
        _tool_call("read_file", {"path": "greeting.txt"})]}] if paid_round else [])
    assignment = replace(_assignment(subject_repo, llm), custody_root=tmp_path / "custody")
    if shape == "report":
        assignment.request.surface = "deep_self_review"
    executor = NativeToolRoundReviewExecutor(assignment, llm=llm)
    if shape == "report" and paid_round:
        result = executor.execute()
        assert result.raw_text == result.message["content"] == draft
        assert result.usage["native_incomplete"] == "control_authority_unavailable"
        assert executor.execute().raw_text == draft  # local reuse, no second episode
    else:
        with pytest.raises(ua.ControlAuthorityUnavailable) as raised:
            executor.execute()
        assert raised.value is error and raised.value.cause == "dispatch_publication_unconfirmed"
    custody = executor.failure_custody()
    assert custody["native_end_reason"] == "control_authority_unavailable"
    assert custody["native_rounds"] == int(paid_round)
    assert len(admissions) == 1 + int(paid_round)
    assert len(llm.calls) == int(paid_round)
    if paid_round:
        assert custody["cost"] == 0.125
        assert custody["ledger_attempt_ids"] == ["paid-first"]
        assert draft.strip() in custody["native_terminal_round"].replace("\\n", "\n")
        assert custody["native_tool_receipts"][0]["path"] == "greeting.txt"
        assert custody["native_transcript_chars"] > 0
