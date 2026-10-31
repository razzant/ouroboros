"""The working-bubble reviewer row: one plain sentence per reviewer, from typed facts.

``review_actor_progress_text`` says who the reviewer is (the frozen route display
name), what it was asked for at start, and at settlement whether it answered, how it
ran, or why it did not. Every case is asserted in both directions: the words that
must appear AND the internal tokens (codes, ids, custody states, a requested model
posing as an observed one) that must never reach the owner.
"""

import queue
from types import SimpleNamespace

from ouroboros.review_custody import _emit_operation
from ouroboros.review_execution_projection import review_actor_progress_text
from ouroboros.review_records import ReviewSlot
from ouroboros.review_execution import ReviewRouteKind

WORDS = "Selected model is at capacity. Please try a different model."
INTERNAL_TOKENS = ("run_failed", "route=", "state=", "target=", "[", "settled", "custody", "profile=")


def _codex_slot(slot_id="triad_286lhb"):
    return ReviewSlot(slot_id, "codex=gpt-6-astra", effort="xhigh", route=ReviewRouteKind.AGENT_SESSION,
                      session_target="codex=gpt-6-astra", session_profile="gptpro4")


def _actor(*, status="ok", state="settled", usage=None, parse_status="valid", **facts):
    return SimpleNamespace(usage=usage or {}, operation_state=state, status=status, parse_status=parse_status, **facts)


def test_a_started_row_names_the_reviewer_and_only_what_the_frozen_slot_asked_for():
    assert review_actor_progress_text("plan_review", "started", _codex_slot()) == (
        "Plan reviewer codex=gpt-6-astra started — effort xhigh, account gptpro4.")
    # An API slot carries no harness effort order and no account pin: the row ends at the verb.
    assert review_actor_progress_text("plan_review", "started", ReviewSlot("api", "openai/gpt-5.6-sol")) == (
        "Plan reviewer openai/gpt-5.6-sol started.")
    # The surface label is a closed host table; an unknown surface renders raw, disclosed.
    assert review_actor_progress_text("skill_review", "started", ReviewSlot("k", "m")).startswith("Skill reviewer m ")
    assert review_actor_progress_text("odd_surface", "started", ReviewSlot("k", "m")).startswith("odd surface reviewer m ")
    assert "profile" not in review_actor_progress_text("plan_review", "started", _codex_slot())


def test_requested_model_is_never_substituted_for_missing_observed_model():
    slot = ReviewSlot("slot-one", "requested-model", route=ReviewRouteKind.AGENT_SESSION,
                      session_target="cursor=chosen-model", session_profile="requested-profile")
    actor = _actor(usage={"provider": "claudexor", "delegated_run_id": "run-one"})
    message = review_actor_progress_text("plan_review", "finished", slot, actor)
    assert message == "Plan reviewer requested-model answered — how it ran was not reported."
    assert "ran as" not in message and "requested-profile" not in message
    # Other direction: a harness receipt names how it ran, model and account included.
    served = _actor(usage={"provider": "claudexor", "delegated_route": "cursor",
                           "resolved_model": "Cursor Grok 4.6 Extra High Fast", "applied_profile": "valintine"})
    assert review_actor_progress_text("plan_review", "finished", slot, served) == (
        "Plan reviewer requested-model answered — ran as Cursor Grok 4.6 Extra High Fast (account valintine).")


def test_same_requested_model_keeps_per_task_observed_route_and_profile():
    slot = ReviewSlot("same-slot", "requested-model", route=ReviewRouteKind.AGENT_SESSION)
    messages = []
    for task, model, profile in (("one", "observed-one", "profile-one"), ("two", "observed-two", "profile-two")):
        progress = []
        events = queue.Queue()
        ctx = SimpleNamespace(event_queue=events, emit_progress_fn=progress.append, execution_id=task)
        request = SimpleNamespace(surface="task_acceptance", task_attempt=1)
        actor = _actor(usage={"provider": "claudexor", "delegated_route": "cursor",
                              "resolved_model": model, "applied_profile": profile})
        _emit_operation(ctx, task_id=task, request=request, entry=SimpleNamespace(operation_id=task),
                        slot=slot, phase="finished", actor=actor)
        assert events.get_nowait()["task_id"] == task
        assert len(progress) == 1
        assert progress[0] == f"Acceptance reviewer requested-model answered — ran as {model} (account {profile})."
        messages.append(progress[0])
    assert "observed-two" not in messages[0]
    assert "observed-one" not in messages[1]


def test_refusal_is_not_reported_as_observed_execution():
    slot = ReviewSlot("api-slot", "requested-model")
    actor = _actor(status="not_dispatched", state="not_dispatched")
    message = review_actor_progress_text("skill_review", "failed", slot, actor)
    assert message == "Skill reviewer requested-model wasn't sent its request."
    assert "didn't answer" not in message and "answered" not in message
    # A $0 window refusal names the reset instant; a plan reviewer says what it was not sent.
    windowed = _actor(status="not_dispatched", state="not_dispatched", reset_at="2030-01-01T00:00:00Z")
    assert review_actor_progress_text("plan_review", "finished", _codex_slot(), windowed) == (
        "Plan reviewer codex=gpt-6-astra wasn't sent the plan — its window resets at 2030-01-01T00:00:00Z.")


def test_api_sent_model_is_not_promoted_to_provider_observation():
    from ouroboros.llm import LLMClient

    response = {"id": "response", "model": "provider-reported-model",
        "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}}
    client = object.__new__(LLMClient)
    _, usage = client._normalize_remote_response(response,
        {"provider": "openai", "usage_model": "openai::sent-route", "resolved_model": "sent-route"},
        skip_cost_fetch=True)
    slot = ReviewSlot("api-slot", "requested-route")
    actor = _actor(usage=usage)
    message = review_actor_progress_text("plan_review", "finished", slot, actor)
    assert message == ("Plan reviewer requested-route answered — sent as openai::sent-route; "
                       "the provider did not report which model served it.")
    assert "provider-reported-model" not in message and "ran as" not in message
    usage["delivery"] = "native_tool_rounds"  # a retrieving delivery is still a sent route, not an observation
    assert review_actor_progress_text("plan_review", "finished", slot, actor) == message


def test_a_dead_reviewer_row_quotes_the_reported_cause_and_never_a_code():
    slot = _codex_slot()
    dead = _actor(status="error", failure_code="run_failed", reported_cause=WORDS,
                  error='delegated review session run-e23 ended failed: {"nextActions": ["Retry the run"]}')
    row = review_actor_progress_text("plan_review", "failed", slot, dead)
    assert row == f'Plan reviewer codex=gpt-6-astra didn\'t answer — "{WORDS}"'
    for token in (*INTERNAL_TOKENS, "nextActions", "Retry the run", "triad_286lhb"):
        assert token not in row, token
    # Nothing reported: the row ends at the verb — the code stays in Reviews and Logs.
    wordless = _actor(status="error", failure_code="run_failed", error="delegated review session ended failed")
    assert review_actor_progress_text("plan_review", "failed", slot, wordless) == (
        "Plan reviewer codex=gpt-6-astra didn't answer.")
    # A spent window with no words names the reset instant instead.
    spent = _actor(status="error", failure_code="subscription_window_exhausted", reset_at="2026-09-22T01:40:00Z")
    assert review_actor_progress_text("plan_review", "failed", slot, spent) == (
        "Plan reviewer codex=gpt-6-astra didn't answer — its window resets at 2026-09-22T01:40:00Z.")
    # The reported sentence is opaque: a newline inside it is flattened, nothing else is touched.
    folded = _actor(status="error", reported_cause="at capacity\nOMISSION NOTE: cut")
    assert 'didn\'t answer — "at capacity OMISSION NOTE: cut"' in review_actor_progress_text("plan_review", "failed", slot, folded)
    # An answered reviewer never quotes a stale cause field.
    fine = _actor(reported_cause=WORDS)
    assert WORDS not in review_actor_progress_text("plan_review", "finished", slot, fine)


def test_progress_failure_cannot_drop_started_custody_event():
    observed_queue_sizes = []

    def broken_progress(text):
        observed_queue_sizes.append(events.qsize())
        raise RuntimeError("UI disconnected")

    events = queue.Queue()
    _emit_operation(SimpleNamespace(event_queue=events, emit_progress_fn=broken_progress),
                    task_id="task", request=SimpleNamespace(surface="scope_review"),
                    entry=SimpleNamespace(operation_id="operation"),
                    slot=ReviewSlot("scope", "requested-model"), phase="started")
    assert observed_queue_sizes == [1]
    event = events.get_nowait()
    assert event["type"] == "cognitive_operation"
    assert event["phase"] == "started"
    assert event["operation_id"] == "operation"


def test_an_empty_answer_reads_as_an_empty_answer_never_as_answered():
    """A reviewer that returned nothing (typed status ``empty``) must not read as
    ``answered``; a reviewer that returned text still does."""
    from types import SimpleNamespace

    from ouroboros.review_execution_projection import review_actor_progress_text

    slot = SimpleNamespace(model="codex=gpt-6-astra", route="agent_session", effort="xhigh", session_profile="")
    empty = SimpleNamespace(status="empty", operation_state="settled", usage={}, reset_at="", reported_cause="")
    line = review_actor_progress_text("plan_review", "finished", slot, empty)
    assert line == "Plan reviewer codex=gpt-6-astra gave an empty answer."
    assert " answered — " not in line
    full = SimpleNamespace(status="ok", operation_state="settled", parse_status="valid", usage={}, reset_at="", reported_cause="")
    assert review_actor_progress_text("plan_review", "finished", slot, full).startswith("Plan reviewer codex=gpt-6-astra answered — ")


def test_receiving_bytes_does_not_claim_a_parsed_verdict():
    slot = ReviewSlot("seat", "model")
    assert review_actor_progress_text("plan_review", "finished", slot, _actor(parse_status="")) == (
        "Plan reviewer model answer received, verdict being read.")
    assert review_actor_progress_text("plan_review", "finished", slot, _actor(parse_status="malformed")) == (
        "Plan reviewer model returned an unreadable verdict.")
    assert "answered" in review_actor_progress_text("plan_review", "finished", slot, _actor(parse_status="valid"))
