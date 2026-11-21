"""#869 — what the owner is told when a dispatched attempt has NO outcome.

The rail already refuses to resend (``provider_no_call_source``), and it already
records ``infra_failed`` / ``provider_unavailable``. Two things were still wrong
in what the OWNER read:

* the durable source asked only for the round's repeat RECORD, while the owner
  sentence asked the wider question (record OR the sticky
  ``provider_outcome_unknown`` kind) — so an attempt interrupted in flight was
  described as unknown and stamped ``transport_unavailable_no_resend``;
* the terminal never said what the unresolved attempt COST, and on the no-call
  rails it never said how long the turn had actually waited.

This is presentation and provenance only. The differential guard below is the
contract: no additional model call, unchanged metadata, unchanged salvage.
"""
from __future__ import annotations

import pytest

import ouroboros.loop as loop_mod
import ouroboros.loop_transport as loop_transport
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY


class _NoCallLLM:
    """Any touch is a model call the unknown-outcome terminal must not make."""

    def __getattr__(self, name):  # pragma: no cover - the failure IS the point
        pytest.fail(f"the unknown-outcome terminal reached the provider: .{name}")


def _ctx(tmp_path, accumulated, messages=None):
    return loop_mod._RoundLimitContext(
        messages=messages or [{"role": "user", "content": "go"}],
        llm=_NoCallLLM(), active_model="test-model", active_effort="low", max_retries=3,
        drive_logs=tmp_path, task_id="t-unknown", round_idx=1, event_queue=None,
        accumulated_usage=accumulated, task_type="task", active_use_local=False,
        max_rounds=200, drive_root=tmp_path,
    )


def _text(accumulated, **over):
    kwargs = {
        "is_context_overflow": False, "is_transport_wait": True, "waited_sec": 0.0,
        "interactive": False, "is_deadline_exhausted": False,
    }
    kwargs.update(over)
    return loop_transport.provider_terminal_fallback_text(accumulated, **kwargs)


class TestTheSourcePredicate:
    def test_an_unknown_outcome_without_a_repeat_record_keeps_the_unknown_source(self, tmp_path):
        accumulated = {"_last_llm_error_kind": "provider_outcome_unknown",
                       "execution_status": "infra_failed", "reason_code": "provider_unavailable"}
        text, usage, trace = loop_mod._handle_provider_unavailable(
            _ctx(tmp_path, accumulated), error_kind="provider_outcome_unknown",
            wait_cause="transport_unavailable", waited_sec=300.0,
        )
        assert TRANSPORT_DEATHS_KEY not in usage
        assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
        # Metadata is the rail's, unchanged by the wording fix.
        assert usage["execution_status"] == "infra_failed"
        assert usage["reason_code"] == "provider_unavailable"
        # The owner reads the no-resend fence AND the money fact that follows it.
        assert "no retry or paid fallback was sent" in text
        assert text.endswith(loop_transport.UNKNOWN_ATTEMPT_COST_NOTE)

    def test_a_plain_connection_failure_still_takes_the_transport_source(self, tmp_path):
        accumulated = {"_last_llm_error_kind": "transport_unavailable",
                       "execution_status": "infra_failed", "reason_code": "provider_unavailable"}
        text, usage, trace = loop_mod._handle_provider_unavailable(
            _ctx(tmp_path, accumulated), error_kind="provider_unavailable",
            wait_cause="transport_unavailable", waited_sec=300.0,
        )
        assert trace["forced_finalization"]["source"] == "transport_unavailable_no_resend"
        assert loop_transport.UNKNOWN_ATTEMPT_COST_NOTE not in text
        assert text.startswith("⚠️ The request did not reach the provider; the task waited")
        assert "Retry when connectivity" not in text and "consciousness" not in text

    def test_the_repeat_record_path_is_unchanged(self, tmp_path):
        accumulated = {
            TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 1, "backoff_sec": 4.0,
                                   "error_kind": "transport_unavailable"},
            "execution_status": "infra_failed", "reason_code": "llm_api_error",
        }
        text, usage, trace = loop_mod._handle_provider_unavailable(
            _ctx(tmp_path, accumulated), error_kind="context_overflow",
            wait_cause="transport_unavailable", waited_sec=610.0,
        )
        assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
        assert usage["reason_code"] == "provider_unavailable"
        assert "1 earlier physical attempt(s) of the last dispatched round" in text
        assert "context exceeded" not in text


class TestWhatTheOwnerReads:
    def test_the_unknown_wait_states_the_fence_the_outcome_and_the_cost(self):
        unknown = {"_last_llm_error_kind": "provider_outcome_unknown"}
        text = _text(unknown, waited_sec=600.0)
        assert "provider wait" in text and "10.0 min" in text
        assert "no terminal provider outcome" in text
        assert "no retry or paid fallback was sent" in text
        assert "does not establish the attempt's cost" in text and "not a settled receipt" in text
        assert "redialed" not in text

    def test_the_no_call_rails_say_the_real_wait_instead_of_implying_none(self):
        unknown = {"_last_llm_error_kind": "provider_outcome_unknown"}
        # The generic terminal (the `provider_outcome_unknown_no_resend` no-call
        # rail) used to name neither the wait nor the price.
        generic = _text(unknown, is_transport_wait=False, waited_sec=600.0)
        assert "The task spent 10.0 min in the provider wait" in generic
        assert generic.endswith("Any files written so far are preserved in the workspace.")
        assert loop_transport.UNKNOWN_ATTEMPT_COST_NOTE.strip() in generic
        # An interactive turn speaks for the turn, not the task.
        assert "This turn spent 10.0 min in the provider wait" in _text(
            unknown, is_transport_wait=False, waited_sec=600.0, interactive=True)
        # A zero wait claims none.
        assert "waited" not in _text(unknown, is_transport_wait=False, waited_sec=0.0)

    def test_a_known_outcome_gains_no_cost_or_wait_clause(self):
        known = {"_last_llm_error_kind": "provider_incomplete_response"}
        text = _text(known, is_transport_wait=False, waited_sec=600.0)
        assert "waited" not in text
        assert loop_transport.UNKNOWN_ATTEMPT_COST_NOTE not in text

    def test_the_deadline_terminal_carries_the_same_two_facts_when_unknown(self):
        unknown = {"_last_llm_error_kind": "provider_outcome_unknown"}
        text = _text(unknown, is_transport_wait=False, is_deadline_exhausted=True)
        assert "owner deadline ended primary model work" in text
        assert text.endswith(loop_transport.UNKNOWN_ATTEMPT_COST_NOTE)


class TestSalvageAndBoundaries:
    def test_a_salvaged_answer_still_wins_over_the_host_sentence(self, tmp_path):
        accumulated = {"_last_llm_error_kind": "provider_outcome_unknown",
                       "execution_status": "infra_failed", "reason_code": "provider_unavailable"}
        messages = [{"role": "user", "content": "go"},
                    {"role": "assistant", "content": "Partial work is on disk."}]
        text, _usage, trace = loop_mod._handle_provider_unavailable(
            _ctx(tmp_path, accumulated, messages), error_kind="provider_outcome_unknown",
            wait_cause="transport_unavailable", waited_sec=300.0,
        )
        assert "Partial work is on disk." in text
        assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"

    def test_the_no_call_and_wait_predicates_themselves_are_untouched(self):
        from ouroboros.loop_llm_call import provider_no_call_source

        assert provider_no_call_source(
            {"_last_llm_error_kind": "provider_outcome_unknown"}, False,
        ) == ("provider_outcome_unknown_no_resend", False)
        assert provider_no_call_source({}, False) == ("", False)
        # The rail's durable source now asks the SAME question this predicate and
        # the owner sentence ask: a record OR the sticky unknown kind. One
        # question, one answer, on all three surfaces.
        for usage in ({"_last_llm_error_kind": "provider_outcome_unknown"},
                      {TRANSPORT_DEATHS_KEY: {"count": 1}},
                      {TRANSPORT_DEATHS_KEY: {"count": 1},
                       "_last_llm_error_kind": "context_overflow"}):
            assert provider_no_call_source(usage, False)[0] == "provider_outcome_unknown_no_resend"


@pytest.mark.parametrize('control,action', [
    ('owner_requested_finalization', 'Wrap up'), ('owner_stopped_direct_turn', 'Stop'),
])
def test_stop_and_wrapup_text_are_byte_preserved(control, action):
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    reason = REASON_OWNER_REQUESTED_FINALIZATION if action == 'Wrap up' else REASON_OWNER_STOPPED_DIRECT_TURN
    usage = {'_last_llm_error_kind': 'provider_outcome_unknown'}
    expected = (
        f'⚠️ The owner requested {action} while the provider had not returned a usable answer.'
        ' The wait ended after 5.0 min; No new summary request was sent. Any files written so far are preserved.'
        ' The dispatched request has no terminal provider outcome, so no '
        'retry or paid fallback was sent; either could duplicate live work.'
    )
    assert _text(usage, control_reason=reason, waited_sec=300).encode() == expected.encode()


@pytest.mark.parametrize(("salvage", "wait"), [('', False), ('Saved **exact** bytes.\n\n```text\nα < β\n```', True)])
def test_terminal_wording_is_differentially_call_free_and_preserves_salvage(tmp_path, monkeypatch, wait, salvage):
    import copy

    original = loop_mod._provider_terminal_fallback_text
    accumulated = {'_last_llm_error_kind': 'provider_outcome_unknown',
                   'execution_status': 'infra_failed', 'reason_code': 'provider_unavailable'}
    messages = [{'role': 'user', 'content': 'go'}]
    if salvage:
        messages.append({'role': 'assistant', 'content': salvage})

    def run():
        return loop_mod._handle_provider_unavailable(
            _ctx(tmp_path, copy.deepcopy(accumulated), copy.deepcopy(messages)),
            error_kind='provider_outcome_unknown',
            wait_cause='transport_unavailable' if wait else '', waited_sec=300,
        )

    # Replace ONLY the presentation seam for the reference run. Both exercise
    # the actual rail and the fail-on-any-provider-access sentinel above.
    monkeypatch.setattr(loop_mod, '_provider_terminal_fallback_text', lambda *_a, **_kw: 'reference terminal notice')
    before_text, before_usage, before_trace = run()
    monkeypatch.setattr(loop_mod, '_provider_terminal_fallback_text', original)
    after_text, after_usage, after_trace = run()
    before_usage.pop('terminal_provider_notice', None)
    after_usage.pop('terminal_provider_notice', None)
    assert before_usage == after_usage
    assert before_trace['forced_finalization']['source'] == after_trace['forced_finalization']['source']
    if salvage:
        assert before_text.encode() == after_text.encode() == salvage.encode()
    else:
        assert before_text == 'reference terminal notice'
        assert after_text != before_text and 'redialed' not in after_text


@pytest.mark.parametrize("terminal", ["outage", "unknown_wait", "unknown_deadline",
                                       "latched_outage", "latched_unknown_wait", "latched_unknown_deadline"])
@pytest.mark.parametrize("refusal", ["stated", "prediction", "missing_facts", "old_round", "none"])
def test_wait_terminal_keeps_only_its_rounds_earlier_size_refusal(tmp_path, terminal, refusal):
    """Exercise the no-call production terminal, including the review's outage witness.

    The terminal keeps its classification and costs; the earlier request's own
    numbers are separate, with no handbook advice even for a known prediction.
    """
    from copy import deepcopy
    from types import SimpleNamespace

    from ouroboros.context_view_facts import SIZE_REFUSAL_KEY, remember_size_refusal
    from ouroboros.physical_attempt_context import PhysicalAttemptContext

    usage = {"execution_id": "wait-execution"}
    if refusal != "none":
        measured = refusal != "missing_facts"
        if measured:
            usage.update(_context_route_fp="primary-route", _context_window={
                "text": "window 100,000 total: server metadata; usable input 90,000"})
        if refusal in {"stated", "old_round"}:
            usage["_last_llm_provider_message"] = (
                "This model's maximum context length is 100000 tokens. However, you requested 120000 tokens "
                "(110000 in the messages, 10000 in the completion).")
        candidate = None
        if refusal == "prediction":
            candidate = SimpleNamespace(
                estimated_input_tokens=110_000, predicted_free_tokens=-20_000,
                capacity_total_tokens=100_000, response_reserve_tokens=10_000,
                physical_context=PhysicalAttemptContext(
                    profile="owner_low", rendered_mode="low", measurement_basis="cold_estimate",
                    route_fp="primary-route", round_id="wait-execution:round:1", target_total_tokens=None,
                    capacity_total_tokens=100_000, context_target_miss=False, automatic_pass_used=False,
                    measurement_density=1.0, response_reserve_tokens=10_000, required_book_tokens=30_000))
        remember_size_refusal(SimpleNamespace(accumulated_usage=usage, context_fit_plan=None,
            round_idx=0 if refusal == "old_round" else 1, active_model="primary-model"), candidate, measured=measured)
    original = deepcopy(usage.get(SIZE_REFUSAL_KEY))
    latched, terminal = terminal.startswith("latched_"), terminal.removeprefix("latched_")
    # The later fallback's numbers must not be borrowed by the earlier refusal.
    usage.update(_last_llm_error_kind="transport_unavailable" if terminal == "outage" else "provider_outcome_unknown",
        _last_llm_provider_message="", execution_status="infra_failed", reason_code="provider_unavailable",
        _context_prompt_estimate=777_777, _context_window={"text": "window 999,999 total"},
        _context_required_book_tokens=33_333, _context_capacity_total_tokens=999_999,
        _context_response_reserve_tokens=65_536)
    if latched:  # the wait/unknown rail outranks a later mutable size error
        usage["_last_llm_error_kind"] = "context_overflow"
        if terminal != "outage":
            usage["_pending_transport_outcome"] = True
    options = dict(error_kind="deadline_exhausted" if terminal == "unknown_deadline" else "provider_unavailable",
                   wait_cause="" if terminal == "unknown_deadline" else "transport_unavailable", waited_sec=300.0)
    without = {key: value for key, value in deepcopy(usage).items() if key != SIZE_REFUSAL_KEY}
    baseline, baseline_usage, baseline_trace = loop_mod._handle_provider_unavailable(_ctx(tmp_path, without), **options)
    text, after, trace = loop_mod._handle_provider_unavailable(_ctx(tmp_path, usage), **options)
    assert after.get(SIZE_REFUSAL_KEY) == original
    expected_source = "transport_unavailable_no_resend" if terminal == "outage" else "provider_outcome_unknown_no_resend"
    assert trace["forced_finalization"]["source"] == baseline_trace["forced_finalization"]["source"] == expected_source
    # Presentation is the only difference in the record (the slot already existed).
    for record in (after, baseline_usage):
        record.pop("terminal_provider_notice", None)
    assert {key: value for key, value in after.items() if key != SIZE_REFUSAL_KEY} == baseline_usage
    prefix = "\nEarlier in this round, a size refusal: "
    if refusal in {"old_round", "none"}:
        assert text == baseline and prefix not in text
    else:
        assert text.startswith(baseline + prefix) and text.count(prefix) == 1
        assert "on primary-model" in text
        if refusal == "missing_facts":
            assert "request size unknown on primary-model; its window is not known here." in text
        else:
            assert "110,000 tokens" in text and "window 100,000 total" in text
            assert ("host prediction before sending, no provider response" in text) is (refusal == "prediction")
    assert all(word not in text for word in ("777,777", "999,999", "33,333", "handbook", "without it", "The choices", "Context-size recovery"))
    if terminal == "outage":
        assert "The request did not reach the provider" in text
        assert loop_transport.UNKNOWN_ATTEMPT_COST_NOTE not in text
    else:
        assert loop_transport.UNKNOWN_ATTEMPT_COST_NOTE in text
        assert "no retry or paid fallback was sent" in text
    if terminal == "unknown_deadline":
        assert text.startswith("⚠️ The owner deadline ended primary model work")


def test_deadline_words_keep_earlier_size_facts_without_changing_the_deadline():
    from ouroboros.context_view_facts import SIZE_REFUSAL_KEY

    usage = {"_last_llm_error_kind": "deadline_exhausted", SIZE_REFUSAL_KEY: {
        "round": 1, "model": "m", "tokens": 0, "basis": "", "kind": "input", "reserve": 0}}
    baseline = _text({}, is_transport_wait=False, is_deadline_exhausted=True)
    text = _text(usage, round_idx=1, is_transport_wait=False, is_deadline_exhausted=True)
    assert text == baseline + "\nEarlier in this round, a size refusal: request size unknown on m; its window is not known here."
    assert loop_transport.UNKNOWN_ATTEMPT_COST_NOTE not in text
