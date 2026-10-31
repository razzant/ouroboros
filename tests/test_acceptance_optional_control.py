"""Explicit answer selection under armed delivery control, and the pending-review choice.

While a retained answer waits for Main's explicit completion (delivery control
armed), prose and historical control objects never select bytes; ``finish_task``
selects them, records the ``pending_review`` choice and acknowledges the owner
source. The choice is offered only while a result review of the answer is still
running and enforcement does not block (``review_result.pending_review_choice_offered``).
The task-acceptance wait these tests once entered through is gone; a result
review still running is the state that now offers the choice.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from ouroboros import loop
from ouroboros.loop_delivery import completion_observation, consume_completion_request
from ouroboros.tools.control_runtime import _finish_task
from tests.test_acceptance_async_loop import ANSWER
from tests.test_delivery_forced_finalization import _forced_test_context

RUNNING_REVIEW = "rl-running"


def _select(registry, ctx, trace, **arguments):
    registry._ctx._completion_observation = completion_observation(registry._ctx, trace)
    result = _finish_task(registry._ctx, "finish", **arguments)
    assert json.loads(result)["status"] == "completion_requested"
    return consume_completion_request(registry, ctx, trace)


def _held(tmp_path):
    """A retained answer under armed delivery control while its result review still runs."""
    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, ANSWER, control="candidate")
    registry._ctx._result_review_pending = RUNNING_REVIEW
    loop._arm_delivery_control(registry, ctx, trace)
    assert candidate.finalization_control == "awaiting_control"
    return registry, ctx, trace, candidate


@pytest.mark.parametrize("enforcement,mode", [
    ("advisory", "advanced"), ("blocking", "advanced"), ("blocking", "cyber_pro"),
])
@pytest.mark.parametrize("answer", ["prose", "retained", "replace", "finish"])
def test_answer_choice_preserves_the_pending_review_wait_policy(tmp_path, monkeypatch, enforcement, mode, answer):
    from ouroboros.tools.review_result import pending_review_choice_offered

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    registry, ctx, trace, candidate = _held(tmp_path)
    revised = ANSWER + " Budget: $12.\nThe complete timeline is two weeks."
    if answer == "prose":
        assert loop._resolve_delivery_control(revised, registry, ctx, trace) == ("retry", ANSWER)
        assert candidate.full_text == ANSWER
        assert getattr(registry._ctx, "_completion_selected", None) is None
    else:
        selected = ({"answer": revised} if answer == "replace"
                    else {"answer_sha256": candidate.content_sha256})
        if answer == "finish":
            selected["pending_review"] = "finish"
        assert _select(registry, ctx, trace, **selected)
        assert registry._ctx._delivery_candidate.full_text == (revised if answer == "replace" else ANSWER)
    choice = getattr(registry._ctx, "_acceptance_pending_review_choice", "") or "wait"
    assert choice == ("finish" if answer == "finish" else "wait")
    # The host waits for the running review unless the choice is offered AND Main chose finish.
    waits = not (pending_review_choice_offered(registry._ctx) and choice == "finish")
    assert waits is ((enforcement == "blocking" and mode != "cyber_pro") or answer != "finish")
    assert registry._ctx._result_review_pending == RUNNING_REVIEW
    assert candidate.control_episode_seen is True
    assert "Interim prose is activity, not a new completion selection" in str(ctx.messages)
    assert ("Pending critics: choose pending_review=wait" in str(ctx.messages)) is (
        enforcement == "advisory" or mode == "cyber_pro"
    )


@pytest.mark.parametrize("raw", [
    "", "   ", [{"type": "thinking", "thinking": "reasoning only"}],
    '{"delivery_control":"unknown"}', '{"delivery_control":"replace","full_answer":""}',
    '{"delivery_control":"replace","full_answer":', '{"full_answer":"missing verb"}',
    '{"delivery_control":"keep","pending_review":"unknown"}',
    '{"delivery_control":"keep","delivery_control":"replace","full_answer":"bad"}',
    'A notice.\n{"delivery_control":"keep"}', '```json\n{"delivery_control":\n```',
])
def test_unselected_or_historical_control_never_forces_completion(tmp_path, monkeypatch, raw):
    registry, ctx, trace, candidate = _held(tmp_path)
    for _ in range(3):
        assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", ANSWER)
        assert candidate.full_text == ANSWER and not candidate.degraded
        assert ctx.messages[-1]["role"] == "user"
        loop._arm_delivery_control(registry, ctx, trace)
    assert registry._ctx._result_review_pending == RUNNING_REVIEW


def test_explicit_selection_after_a_malformed_old_control(tmp_path, monkeypatch):
    registry, ctx, trace, candidate = _held(tmp_path)
    assert loop._resolve_delivery_control("", registry, ctx, trace) == ("retry", ANSWER)
    revised = ANSWER + " Budget: $12."
    assert _select(registry, ctx, trace, answer=revised)
    assert registry._ctx._delivery_candidate.full_text == revised
    assert not candidate.degraded
    assert registry._ctx._acceptance_pending_review_choice == "wait"


@pytest.mark.parametrize("arguments", [
    {"action": "unknown", "answer": ANSWER},
    {"action": "finish", "answer": ""},
    {"action": "finish", "answer_sha256": ""},
    {"action": "finish", "answer": ANSWER, "answer_sha256": "both"},
    {"action": "stop", "answer": ANSWER},
    {"action": "finish", "answer": ANSWER, "pending_review": "unknown"},
])
def test_invalid_completion_retains_answer_then_allows_explicit_reselection(tmp_path, monkeypatch, arguments):
    registry, ctx, trace, candidate = _held(tmp_path)
    result = _finish_task(registry._ctx, **arguments)
    assert "ERROR: COMPLETION_ARGUMENT:" in result
    assert getattr(registry._ctx, "_completion_request", None) is None
    assert candidate.full_text == ANSWER
    assert registry._ctx._result_review_pending == RUNNING_REVIEW
    assert _select(registry, ctx, trace, answer_sha256=candidate.content_sha256)
    assert registry._ctx._delivery_candidate.full_text == ANSWER


def test_a_strict_schema_empty_twin_field_names_nothing(tmp_path, monkeypatch):
    """Strict tool schemas make a model fill the field it did not choose with "": an answer
    beside an empty selector is that answer, and an empty answer beside a real selector is that
    selection. Both empty, or an empty selector alone, still name nothing."""
    registry, ctx, trace, candidate = _held(tmp_path)
    assert _select(registry, ctx, trace, answer=ANSWER, answer_sha256="")
    assert registry._ctx._delivery_candidate.full_text == ANSWER
    registry, ctx, trace, candidate = _held(tmp_path)
    assert _select(registry, ctx, trace, answer="", answer_sha256=candidate.content_sha256)
    assert registry._ctx._delivery_candidate.full_text == ANSWER
    for arguments in ({"action": "finish", "answer": "", "answer_sha256": ""},
                      {"action": "finish", "answer_sha256": "  "}):
        registry, ctx, trace, candidate = _held(tmp_path)
        assert "ERROR: COMPLETION_ARGUMENT:" in _finish_task(registry._ctx, **arguments)
        assert getattr(registry._ctx, "_completion_request", None) is None


def test_changed_owner_source_still_requires_an_exact_current_acknowledgement(tmp_path, monkeypatch):
    registry, ctx, trace, old = _held(tmp_path)
    stale_source = registry._ctx._acceptance_observation["owner_source_sha256"]
    registry._ctx._owner_directives = [{"content": "Also give the delivery date."}]
    loop._arm_delivery_control(registry, ctx, trace)  # the re-armed control re-observes the source
    revised = ANSWER + " Delivery date: Friday."
    assert loop._resolve_delivery_control(revised, registry, ctx, trace) == ("retry", ANSWER)
    assert old.full_text == ANSWER
    assert not _select(registry, ctx, trace, answer=revised,
                       acceptance_subject={"owner_source_sha256": stale_source})
    assert old.full_text == ANSWER
    observed = registry._ctx._acceptance_observation
    assert _select(registry, ctx, trace, answer=revised, acceptance_subject={
        "owner_source_sha256": observed["owner_source_sha256"],
    })
    assert registry._ctx._delivery_candidate.owner_source_sha256 == observed["owner_source_sha256"]


def test_explicit_replacement_after_material_change_binds_the_new_evidence(tmp_path, monkeypatch):
    registry, ctx, trace, old = _held(tmp_path)
    trace["tool_calls"].append({"tool": "write_file", "status": "ok", "is_error": False,
                                "result": "new material evidence"})
    revised = ANSWER + " Budget: $12."
    assert _select(registry, ctx, trace, answer=revised)
    fresh = registry._ctx._delivery_candidate
    assert fresh.evidence_revision > old.evidence_revision
    assert fresh.evidence_fingerprint != old.evidence_fingerprint
    assert fresh.acceptance_binding["authoritative"] is False
    assert fresh.control_episode_seen is True


@pytest.mark.parametrize("raw,expected", [
    ('{"action":"finish","answer":', (ANSWER, True, True, True, False)),
    (json.dumps({"action": "finish", "answer_sha256": hashlib.sha256(ANSWER.encode()).hexdigest()}),
     (ANSWER, True, False, True, False)),
    ('{"action":"finish","answer":"Complete replacement."}',
     ("Complete replacement.", False, False, True, True)),
    ("Forced complete answer.", ("Forced complete answer.", False, False, True, False)),
])
def test_forced_transport_uses_explicit_completion_or_plain_final_prose(tmp_path, monkeypatch, raw, expected):
    from ouroboros.loop_delivery import _resolve_forced_delivery_control_body

    registry, _ctx, _trace, candidate = _held(tmp_path)
    assert _resolve_forced_delivery_control_body(
        raw, candidate, armed=registry._ctx._delivery_control_required,
    ) == expected
