"""Cross-phase contracts of the two transport rails.

The free wait episode (a released pre-dispatch failure, $0, ``transport_unavailable``)
and the paid repeat rail (a dispatched typed transport death, ``provider_outcome_unknown``)
never overlap, and while the round holds a repeat record that record outranks the
wait terminal:

- a dispatched death on an interactive turn takes the repeat rail and never opens a
  wait episode (unknown is not the episode's kind);
- inside an episode the chain's local-only pass exists for ``transport_unavailable``
  alone, and a round record blocks even that pass;
- an episode exhausted on a round that still holds a repeat record ends on the
  record's source, worded as both the wait and the unresolved attempt — and that
  wording names the class the repeat was RELEASED with (it rides the round record,
  so the generic no-call terminal names it too), never the later refusal that
  closed the window or the later free redial's own class;
- the terminal precedence is decided once, at the provider-death rail: a round
  record outranks the latched wait cause, which outranks the overflow salvage —
  so the ``context_overflow`` a failed local-only pass leaves in the mutable kind
  never turns a waited-out outage into the overflow terminal.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop_mod
import ouroboros.loop_llm_call as call_mod
import ouroboros.loop_transport as loop_transport
from ouroboros.loop import run_llm_loop
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
from ouroboros.outcomes import derive_loop_outcome
from tests.test_loop_transport_wait import _FakeClock
from tests.test_transport_death_retry import (
    _ScriptedLLM,
    _death,
    _events,
    _loop_kwargs,
    _no_chain,
    _overflow_failure,
    _released_connect,
    _status_failure,
)


@pytest.fixture
def no_sleep(monkeypatch):
    """The repeat rail's backoff sleeps, recorded instead of slept (as in the
    transport-death suite; a fixture is defined where it is used)."""
    sleeps = []
    monkeypatch.setattr(call_mod, "_sleep_within_deadline", lambda sec, _dl: (sleeps.append(sec), True)[1])
    return sleeps


@pytest.mark.parametrize("turn_flag", ["is_direct_chat", "is_ephemeral_turn"])
@pytest.mark.parametrize("deaths", [2, 3])
def test_interactive_turn_death_takes_the_repeat_rail_and_never_enters_a_wait_episode(
    tmp_path, monkeypatch, no_sleep, turn_flag, deaths,
):
    """A direct-chat or ephemeral turn whose DISPATCHED request died with a typed
    transport death is on the paid repeat rail (its round dispatch is primary),
    never in the free wait episode: `provider_outcome_unknown` is not the
    episode's `transport_unavailable`, so no `network_wait` event exists, the
    interactive idle bound never starts, and an exhausted round ends on the
    unknown no-resend terminal, not on the wait window's."""
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setattr(
        loop_transport, "interruptible_wait_sleep",
        lambda _sec, _wake: pytest.fail("a dispatched death must never open a wait episode"),
    )
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(*([_death] * deaths))
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    setattr(kwargs["tools"]._ctx, turn_flag, True)
    result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 3  # the primary send plus two paid repeats, whatever the turn kind
    assert no_sleep == [4.0, 8.0]  # the repeat rail's backoffs, never the episode's wait
    assert _events(tmp_path, "network_wait") == []
    assert not any("provider connection" in note.lower() for note in notes)
    if deaths == 2:
        assert result == "done"
        assert usage.get("reason_code") is None
        assert TRANSPORT_DEATHS_KEY not in usage
    else:
        assert usage.get("execution_status") == "infra_failed"
        assert trace.get("forced_finalization", {}).get("source") == "provider_outcome_unknown_no_resend"
        assert usage[TRANSPORT_DEATHS_KEY]["count"] == 2
        assert "2 earlier physical attempt(s) of the last dispatched round" in result
        assert "waited and redialed" not in result and "no wait window" not in result


@pytest.mark.parametrize("turn", ["managed", "is_direct_chat", "is_ephemeral_turn"])
@pytest.mark.parametrize("with_record", [True, False])
def test_wait_episode_exhausted_on_a_round_holding_a_repeat_record_takes_the_unknown_source(
    tmp_path, monkeypatch, no_sleep, turn, with_record,
):
    """death → granted repeat RELEASED (`transport_unavailable`) → wait episode →
    window exhausted (a managed task's deadline, an interactive turn's idle
    bound): the round still holds an unresolved paid attempt, so the record
    fence outranks the wait terminal — durable source
    `provider_outcome_unknown_no_resend`, owner text saying both that it waited
    and that an earlier attempt stays unresolved; execution status and reason
    code stay the wait terminal's. Control: the same episode without a record
    keeps `transport_unavailable_no_resend` and the byte-identical base wording."""
    from ouroboros.config import get_finalization_grace_sec

    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    script = ([_death] if with_record else []) + [_released_connect] * 12
    llm = _ScriptedLLM(*script)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    if turn == "managed":
        deadline = datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 8)
        kwargs["tools"]._ctx.task_metadata = {"deadline_at": deadline.isoformat()}
    else:
        setattr(kwargs["tools"]._ctx, turn, True)
    result, usage, trace = run_llm_loop(**kwargs)

    assert (3 if with_record else 2) <= llm.calls <= len(script)  # the episode ended; the turn never recovered
    assert no_sleep == ([4.0] if with_record else [])  # one repeat backoff; released redials never re-arm it
    ended = [row["detail"] for row in _events(tmp_path, "network_wait") if row["phase"] == "ended"]
    assert ended == ["deadline_after_final_redial" if turn == "managed" else "interactive_wait_window_exhausted"]
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "provider_unavailable"
    base = loop_transport.provider_terminal_fallback_text(
        {}, is_context_overflow=False, is_transport_wait=True, waited_sec=sum(clock.sleeps),
        interactive=turn != "managed", is_deadline_exhausted=False,
    )
    assert "waited and redialed" in base
    if with_record:
        assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
        assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
        assert "1 earlier physical attempt(s) of the last dispatched round" in result
        assert "unresolved at their upper bound" in result
        assert result == base + loop_transport.provider_recovery_hint(usage)
    else:
        assert trace["forced_finalization"]["source"] == "transport_unavailable_no_resend"
        assert TRANSPORT_DEATHS_KEY not in usage
        assert result == base  # byte-identical base wording


def test_deadline_refused_redial_still_names_the_class_the_repeat_was_released_with(
    tmp_path, monkeypatch, no_sleep,
):
    """death → granted repeat RELEASED (`transport_unavailable`) → wait episode →
    the owner window closes while the episode sleeps, so the next FREE redial is
    refused before dispatch (`deadline_refused_dispatch`) and the sticky kind
    becomes `deadline_exhausted`. The terminal is still the record's, and its
    repeat sentence names the class the repeat itself was released with — the
    sticky kind by then belongs to the refusal of a later, free, $0 redial and
    would misname the paid attempt the owner is being told about."""
    from ouroboros.config import get_finalization_grace_sec

    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(_death, _released_connect, _released_connect)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    # Room for the grant (backoff 4 s + the admission reserve) and for one wait.
    metadata = {"deadline_at": (
        datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 8)
    ).isoformat()}
    kwargs["tools"]._ctx.task_metadata = metadata
    waits = []

    def _window_closes_while_waiting(sec, _wake):
        waits.append(sec)
        metadata["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        return False

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", _window_closes_while_waiting)
    result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 2  # the primary send and its granted repeat; the redial never dispatched
    assert no_sleep == [4.0] and len(waits) == 1
    assert [row["reason_code"] for row in _events(tmp_path, "llm_not_dispatched")] == ["deadline_exhausted"]
    ended = [row["detail"] for row in _events(tmp_path, "network_wait") if row["phase"] == "ended"]
    assert ended == ["deadline_refused_dispatch"]
    assert usage["_last_llm_error_kind"] == "deadline_exhausted"  # the refusal's kind, not the repeat's
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == "provider_unavailable"
    assert "waited and redialed" in result  # the wait wording is still there
    assert "the repeat failed as transport_unavailable" in result
    assert "deadline_exhausted" not in result


def test_generic_terminal_names_the_class_the_repeat_was_released_with(tmp_path, monkeypatch, no_sleep):
    """death → granted repeat RELEASED (`transport_unavailable`) → wait episode →
    the free redial gets past the connect phase and fails as a 400, which ends
    the episode (the transport is passable) and leaves the round fenced on its
    record. The generic no-call terminal names the class the repeat itself
    failed with, exactly as the wait terminal does: the sticky `bad_request`
    belongs to the free redial, not to the paid attempt the owner is told about."""
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(_death, _released_connect, lambda: _status_failure(400))
    notes = []
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, notes))

    assert llm.calls == 3  # the primary send, its released repeat, one free redial
    assert no_sleep == [4.0]
    ended = [row["detail"] for row in _events(tmp_path, "network_wait") if row["phase"] == "ended"]
    assert ended == ["error_kind_changed:bad_request"]
    assert usage["_last_llm_error_kind"] == "bad_request"  # the free redial's class stays the sticky kind
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    assert usage[TRANSPORT_DEATHS_KEY]["error_kind"] == "transport_unavailable"
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == "provider_unavailable"
    assert "waited and redialed" not in result  # the generic arm, not the wait terminal
    assert "1 earlier physical attempt(s) of the last dispatched round" in result
    assert "the repeat failed as transport_unavailable" in result
    assert "the repeat failed as bad_request" not in result


def test_fallback_chain_fence_holds_inside_a_wait_episode_too(monkeypatch):
    """`fallback_chain_allowed` on the combined tree: inside a wait episode the
    one local-only chain pass exists for `transport_unavailable` alone (never
    for the unknown kind), and a round record — an unresolved attempt of this
    round — blocks even that pass whatever the kind says, because no candidate
    may dial over a request that may still be live."""
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "1")
    routable = SimpleNamespace(exact_model_route=False)
    record = {"round_id": "r", "count": 1, "backoff_sec": 4.0}
    episode = loop_transport.TransportWaitEpisode(started_monotonic=1.0, interactive=True, wait_bound_sec=900.0)

    assert loop_transport.fallback_chain_allowed(routable, "provider_outcome_unknown", episode) is False
    assert loop_transport.fallback_chain_allowed(
        routable, "transport_unavailable", episode, {TRANSPORT_DEATHS_KEY: record},
    ) is False
    assert episode.local_pass_used is False  # neither refusal spent the episode's single local pass
    assert loop_transport.fallback_chain_allowed(routable, "transport_unavailable", episode) is True
    assert episode.local_pass_used is True
    assert loop_transport.fallback_chain_allowed(routable, "transport_unavailable", episode) is False  # once per episode
    assert loop_transport.fallback_chain_allowed(routable, "provider_outcome_unknown", None) is False


def _overflowing_local_pass(spend_window):
    """The episode's one local-only chain pass, failing with a context overflow
    exactly as ``call_llm_with_retry`` stamps it, and slow enough that the wait
    window is already spent when the round gate reads it (``spend_window``)."""
    chain_calls = {"n": 0}

    def failing_chain(**kwargs):
        chain_calls["n"] += 1
        assert kwargs["active_use_local"] is False  # the primary route that failed pre-dispatch
        assert kwargs["accumulated_usage"]["_last_llm_error_kind"] == "transport_unavailable"
        kwargs["accumulated_usage"].update(
            _last_llm_error_kind="context_overflow", execution_status="infra_failed", reason_code="llm_api_error",
        )
        spend_window()
        return (
            None, kwargs["active_model"], kwargs["active_use_local"],
            kwargs["context_fit_plan"], kwargs["active_context_mode"],
        )

    return failing_chain, chain_calls


@pytest.mark.parametrize("turn", ["managed", "is_direct_chat", "is_ephemeral_turn"])
def test_latched_wait_cause_outranks_the_overflow_a_failed_local_pass_left(tmp_path, monkeypatch, turn):
    """outage latched → the episode's one local-only pass fails with a context
    overflow → the binding window (a managed task's deadline, an interactive
    turn's idle bound) is spent before any redial: the round now holds BOTH
    facts (``wait_cause == "transport_unavailable"`` and the pass's
    ``context_overflow`` in the mutable kind), and the latched wait cause wins —
    durable source ``transport_unavailable_no_resend``, the wait terminal's
    ``provider_unavailable``/``infra_failed`` stamps, the outage wording and no
    forced-final dial — never the overflow salvage's source, ``llm_api_error``
    or window wording."""
    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "local/candidate")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "1")
    llm = _ScriptedLLM(_released_connect)  # a redial would recover, so none may happen
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    metadata = {}
    if turn == "managed":
        from ouroboros.config import get_finalization_grace_sec

        deadline = datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 8)
        metadata["deadline_at"] = deadline.isoformat()
        kwargs["tools"]._ctx.task_metadata = metadata
    else:
        setattr(kwargs["tools"]._ctx, turn, True)

    def _slow_local_inference():
        clock.now += 61.0  # past the interactive idle bound
        if metadata:
            metadata["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()

    failing_chain, chain_calls = _overflowing_local_pass(_slow_local_inference)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", failing_chain)
    result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 1 and chain_calls["n"] == 1  # the primary send and the local pass; no redial, no forced dial
    assert clock.sleeps == []  # the window was spent inside the pass, so the episode never slept
    assert usage["_last_llm_error_kind"] == "context_overflow"  # the pass's overwrite is still the sticky kind
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "provider_unavailable"
    assert trace["forced_finalization"]["source"] == "transport_unavailable_no_resend"
    assert TRANSPORT_DEATHS_KEY not in usage
    ended = [row["detail"] for row in _events(tmp_path, "network_wait") if row["phase"] == "ended"]
    assert ended == ["deadline_exhausted" if turn == "managed" else "interactive_wait_window_exhausted"]
    base = loop_transport.provider_terminal_fallback_text(
        {}, is_context_overflow=False, is_transport_wait=True, waited_sec=0.0,
        interactive=turn != "managed", is_deadline_exhausted=False,
    )
    assert result == base  # byte-identical zero-wait outage wording
    assert "Could not establish a provider connection" in result
    assert "context exceeded" not in result
    # The published projection says what the terminal said: no overflow kind under the wait terminal's reason code.
    assert derive_loop_outcome(result, usage, trace)["failure"] == {"kind": "provider", "reason_code": "provider_unavailable"}


def test_context_overflow_with_no_episode_keeps_the_overflow_salvage(tmp_path, monkeypatch):
    """Control for the precedence: a primary dispatch rejected as a context
    overflow with NO wait episode never walks the chain (a local fallback being
    configured changes nothing) and keeps the overflow terminal unchanged —
    source ``context_overflow_local_salvage``, ``llm_api_error``, the window
    wording."""
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "local/candidate")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "1")
    llm = _ScriptedLLM(_overflow_failure)
    notes = []
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, notes))

    assert llm.calls == 1
    assert _events(tmp_path, "network_wait") == []
    assert usage["_last_llm_error_kind"] == "context_overflow"
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "llm_api_error"
    assert trace["forced_finalization"]["source"] == "context_overflow_local_salvage"
    assert result == loop_transport.provider_terminal_fallback_text(
        {}, is_context_overflow=True, is_transport_wait=False, waited_sec=0.0, is_deadline_exhausted=False,
    )
    assert "context exceeded the selected model window" in result


def test_round_record_outranks_a_wait_cause_that_holds_an_overflow_kind(tmp_path):
    """The rail with all three facts at once: a round record, a latched wait
    cause and an overflow kind. The round loop cannot produce this triple today
    — a record blocks the episode's local-only pass (pinned above), and an
    overflow on a redial or a repeat ends the episode as proof the transport is
    passable — so the fence on top of the precedence is pinned at the rail
    itself: the record's source, the wait terminal's stamps, and text saying
    both the wait and the unresolved attempt, never the overflow salvage."""
    accumulated = {
        # A record an episode can hold names the class its repeat was released with.
        TRANSPORT_DEATHS_KEY: {"round_id": "r", "count": 1, "backoff_sec": 4.0, "error_kind": "transport_unavailable"},
        "_last_llm_error_kind": "context_overflow",
        "execution_status": "infra_failed", "reason_code": "llm_api_error",
    }
    ctx = loop_mod._RoundLimitContext(
        messages=[{"role": "user", "content": "go"}], llm=SimpleNamespace(), active_model="test-model",
        active_effort="low", max_retries=3, drive_logs=tmp_path, task_id="t-death", round_idx=1,
        event_queue=None, accumulated_usage=accumulated, task_type="task", active_use_local=False,
        max_rounds=200, drive_root=tmp_path,
    )
    text, usage, trace = loop_mod._handle_provider_unavailable(
        ctx, error_kind="context_overflow", wait_cause="transport_unavailable", waited_sec=610.0,
    )

    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "provider_unavailable"
    assert "the task waited and redialed" in text
    assert "1 earlier physical attempt(s) of the last dispatched round" in text
    assert "the repeat failed as transport_unavailable" in text
    assert "context exceeded" not in text


def test_published_failure_projection_never_contradicts_the_terminal():
    """`derive_loop_outcome` stamps `failure.error_kind = context_overflow` only for
    the overflow salvage's own reason code: a waited-out outage (or the unknown
    no-resend fence) may leave that sticky kind behind under `provider_unavailable`,
    and the published projection must say what the terminal said."""
    overflow = {"execution_status": "infra_failed", "reason_code": "llm_api_error", "_last_llm_error_kind": "context_overflow"}
    assert derive_loop_outcome("⚠️ The context exceeded the selected model window", overflow, {})["failure"] == {
        "kind": "provider", "reason_code": "llm_api_error", "error_kind": "context_overflow",
    }
    waited_out = {**overflow, "reason_code": "provider_unavailable"}
    assert derive_loop_outcome("⚠️ Could not establish a provider connection", waited_out, {})["failure"] == {
        "kind": "provider", "reason_code": "provider_unavailable",
    }
