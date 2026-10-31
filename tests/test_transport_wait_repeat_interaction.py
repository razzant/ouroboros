"""Paced transport recovery and retained legacy repeat-custody precedence.

Production Presence uses the ordinary round recovery. A historical repeat record
still outranks wait and overflow terminal causes and never authorizes a new paid
fallback; fixtures name that retained record explicitly instead of recreating the
retired Presence-specific repeat policy.
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
    _presence_turn,
    _released_connect,
    _status_failure,
)
from tests._shared import saved_settings


def _held_repeat(kwargs):
    """A retained round with one unresolved explicit repeat, before its next free redial."""
    registry, model = kwargs["tools"], kwargs["llm"]
    original = model.chat
    seeded = False
    def send(**options):
        nonlocal seeded
        if not seeded:
            usage = registry._ctx._accumulated_usage
            usage[TRANSPORT_DEATHS_KEY] = {"round_id": f"{usage['execution_id']}:round:1",
                "count": 1, "backoff_sec": 4.0, "error_kind": "transport_unavailable"}
            seeded = True
        return original(**options)
    model.chat = send


def _failing_walk(walks):
    """A configured-route walk that finds no answer and records the kind it was entered with."""
    def chain(**kwargs):
        walks.append(kwargs["accumulated_usage"].get("_last_llm_error_kind"))
        return (None, kwargs["active_model"], kwargs["active_use_local"],
                kwargs["context_fit_plan"], kwargs["active_context_mode"])
    return chain


@pytest.fixture
def no_sleep(monkeypatch):
    """The repeat rail's backoff sleeps, recorded instead of slept (as in the
    transport-death suite; a fixture is defined where it is used)."""
    sleeps = []
    monkeypatch.setattr(call_mod, "_sleep_within_deadline", lambda sec, _dl, **_kw: (sleeps.append(sec), True)[1])
    return sleeps


@pytest.mark.parametrize("deaths", [1, 2])
def test_presence_unknown_uses_paced_recovery_without_paid_repeat_counter(
    tmp_path, monkeypatch, no_sleep, deaths,
):
    """Raw HTTP losses use the same paced continuation as ordinary turns."""
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.setattr(loop_transport, "upstream_transport_reachable",
                        lambda *_a, **_kw: {"kind": "upstream_http", "status_code": 200})
    clock = _FakeClock(monkeypatch)
    llm = _ScriptedLLM(*([_death] * deaths))
    result, usage, _trace = run_llm_loop(**_presence_turn(_loop_kwargs(tmp_path, llm, [])))
    assert result == "done" and llm.calls == deaths + 1
    assert no_sleep == [] and len(clock.sleeps) == deaths
    assert clock.sleeps == [4.0, 8.0][:deaths]
    assert TRANSPORT_DEATHS_KEY not in usage
    assert usage["transport_recovery"]["old_outcome"] == "unknown"
    events = _events(tmp_path, "network_wait")
    assert len([row for row in events if row["phase"] == "entered"]) == 1
    assert events[-1]["phase"] == "recovered"



@pytest.mark.parametrize("turn,with_record", [
    ("managed", False), ("presence", True), ("presence", False),
])
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
    walks = []
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _failing_walk(walks))
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    script = [_released_connect] * 12
    llm = _ScriptedLLM(*script)
    notes = []
    kwargs = _loop_kwargs(tmp_path, llm, notes)
    if turn == "managed":
        deadline = datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 8)
        kwargs["tools"]._ctx.task_metadata = {"deadline_at": deadline.isoformat()}
    else:
        _presence_turn(kwargs)
    if with_record:
        _held_repeat(kwargs)
    result, usage, trace = run_llm_loop(**kwargs)
    # Each failed redial may revisit alternatives; a retained repeat record never may.
    if with_record:
        assert walks == []
    else:
        assert len(walks) > 1 and set(walks) == {"transport_unavailable"}

    assert 2 <= llm.calls <= len(script)  # the episode ended; the turn never recovered
    assert no_sleep == []  # only free redials follow the retained record
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
        assert "any recorded bound is retained" in result
        assert "Retry when connectivity returns" not in result
        assert "Inspect the preserved facts before starting another run." in result
        # #869: the unknown terminal ends with the no-resend fence AND the money
        # uncertainty: no bound is invented when the attempt has none.
        assert result.endswith(
            loop_transport._unknown_terminal_recovery_hint(usage) + loop_transport.UNKNOWN_ATTEMPT_COST_NOTE)
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
    llm = _ScriptedLLM(_released_connect, _released_connect)
    notes = []
    kwargs = _presence_turn(_loop_kwargs(tmp_path, llm, notes))
    _held_repeat(kwargs)
    # Room for the grant (backoff 4 s + the admission reserve) and for one wait.
    metadata = {"deadline_at": (
        datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 8)
    ).isoformat()}
    kwargs["tools"]._ctx.task_metadata = metadata
    clock = _FakeClock(monkeypatch)

    def _window_closes_while_waiting(sec, _wake):
        clock.sleep(sec, _wake)
        metadata["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        return False

    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", _window_closes_while_waiting)
    result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 1  # the free attempt released; the next redial never dispatched
    assert no_sleep == [] and len(clock.sleeps) == 1
    assert [row["reason_code"] for row in _events(tmp_path, "llm_not_dispatched")] == ["deadline_exhausted"]
    ended = [row["detail"] for row in _events(tmp_path, "network_wait") if row["phase"] == "ended"]
    assert ended == ["deadline_refused_dispatch"]
    assert usage["_last_llm_error_kind"] == "deadline_exhausted"  # the refusal's kind, not the repeat's
    assert usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"
    assert usage["execution_status"] == "infra_failed" and usage["reason_code"] == "provider_unavailable"
    assert "provider wait" in result and "redialed" not in result
    assert "the repeat failed as transport_unavailable" in result
    assert "deadline_exhausted" not in result


def test_generic_terminal_names_the_class_the_repeat_was_released_with(tmp_path, monkeypatch, no_sleep):
    """death → granted repeat RELEASED (`transport_unavailable`) → wait episode →
    the free redial gets past the connect phase and fails as a 400, which ends
    the episode (the transport is passable) and leaves the round fenced on its
    record. The generic no-call terminal names the class the repeat itself
    failed with, exactly as the wait terminal does: the sticky `bad_request`
    belongs to the free redial, not to the paid attempt the owner is told about."""
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _no_chain)
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _sec, _wake: False)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "other/model")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    llm = _ScriptedLLM(_released_connect, lambda: _status_failure(400))
    notes = []
    kwargs = _presence_turn(_loop_kwargs(tmp_path, llm, notes))
    _held_repeat(kwargs)
    result, usage, trace = run_llm_loop(**kwargs)

    assert llm.calls == 2  # released attempt, then one free redial
    assert no_sleep == []
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
    """`fallback_chain_allowed` on the combined tree: the configured-route walk runs
    before a wait episode opens and after paced redials (remote routes included); a round
    record — an unresolved paid repeat of this round — blocks it whatever the kind says;
    an unknown outcome walks only when it is eligible for a new generation."""
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    routable = SimpleNamespace(exact_model_route=False, task_id="t")
    record = {"round_id": "r", "count": 1, "backoff_sec": 4.0}
    episode = loop_transport.TransportWaitEpisode(started_monotonic=1.0, interactive=True, wait_bound_sec=900.0)

    assert loop_transport.fallback_chain_allowed(routable, "transport_unavailable", None, {}) is True
    assert loop_transport.fallback_chain_allowed(routable, "provider_outcome_unknown", None, {}) is True
    for kind in ("transport_unavailable", "provider_outcome_unknown", "bad_request"):
        assert loop_transport.fallback_chain_allowed(routable, kind, episode, {}) is True
        assert loop_transport.fallback_chain_allowed(routable, kind, None, {TRANSPORT_DEATHS_KEY: record}) is False
    rejoinable = {"_pending_transport_outcome": {"same_operation_recoverable": True}}
    assert loop_transport.fallback_chain_allowed(routable, "provider_outcome_unknown", None, rejoinable) is False
    presence = SimpleNamespace(exact_model_route=False, task_id="t", current_task_type="presence")
    assert loop_transport.fallback_chain_allowed(presence, "provider_outcome_unknown", None, {}) is True
    assert not loop_transport.fallback_chain_allowed(presence, "provider_outcome_unknown", None, rejoinable)
    assert loop_transport.fallback_chain_allowed(presence, "transport_unavailable", None, {}) is True
    assert not hasattr(episode, "local_pass_used")


def _overflowing_local_pass(spend_window):
    """The one configured-route walk before the wait, failing with a context overflow
    exactly as ``call_llm_with_retry`` stamps it, and slow enough that the owner
    deadline is already spent when the round gate reads it (``spend_window``)."""
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


@pytest.mark.parametrize("turn", ["managed"])
def test_latched_wait_cause_outranks_the_overflow_a_failed_local_pass_left(tmp_path, monkeypatch, turn):
    """outage → the one route walk before the wait fails with a context overflow →
    the owner deadline is spent before any redial: the round's own outage owns its
    wait (the walk's overflow never re-aims it), and the latched wait cause wins —
    (An interactive turn's idle bound measures waiting from episode entry, which now
    follows the walk, so a slow walk cannot spend it; the rail-level precedence is
    pinned below.)
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

    assert llm.calls == 1 and chain_calls["n"] == 1  # the primary send and the walk; no redial, no forced dial
    assert clock.sleeps == []  # the window was spent inside the walk, so the episode never slept
    assert usage["_last_llm_error_kind"] == "transport_unavailable"  # the round's own outage, not the walk's overflow
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
    assert "The request did not reach the provider" in result
    assert "context exceeded" not in result
    # The published projection says what the terminal said: no overflow kind under the wait terminal's reason code.
    assert derive_loop_outcome(result, usage, trace)["failure"] == {"kind": "provider", "reason_code": "provider_unavailable"}


def test_context_overflow_with_no_episode_keeps_the_overflow_salvage(tmp_path, monkeypatch):
    """Control for the precedence: a primary dispatch rejected as a context
    overflow with NO wait episode walks the configured routes exactly once after
    the primary's own recovery ladder (owner decision 7A, rung f: the candidates
    inherit the view the ladder left) and, when that walk finds no answer, keeps
    the overflow terminal unchanged — source ``context_overflow_local_salvage``,
    ``llm_api_error``, the window wording."""
    walks = []
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", _failing_walk(walks))
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "local/candidate")
    monkeypatch.setenv("USE_LOCAL_FALLBACK", "1")
    llm = _ScriptedLLM(_overflow_failure)
    notes = []
    result, usage, trace = run_llm_loop(**_loop_kwargs(tmp_path, llm, notes))

    assert llm.calls == 1 and walks == ["context_overflow"]  # one walk, entered with the refusal's own kind
    assert _events(tmp_path, "network_wait") == []
    assert usage["_last_llm_error_kind"] == "context_overflow"
    assert usage["execution_status"] == "infra_failed"
    assert usage["reason_code"] == "llm_api_error"
    assert trace["forced_finalization"]["source"] == "context_overflow_local_salvage"
    assert result == loop_transport.provider_terminal_fallback_text(
        usage, is_context_overflow=True, is_transport_wait=False, waited_sec=0.0, is_deadline_exhausted=False,
    )
    assert "Context-size recovery" in result and "exact transcript" in result
    # Nothing substantive could leave the view, so no renewal was sent and the refusal names the frame.
    assert "a renewed view could not make it fit (largest parts: " in result


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
    assert "The task spent" in text and "provider wait" in text
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
