"""TZ-2 C1: every typed reason the runtime can stamp on a task row has one owner
sentence in BOTH cause twins, and the rails that end a task say it on every
owner surface — the card line, the reaper's grace toast, kill notice and salvage
line, and the loop's own fallback text — while the typed code stays on the
record and on the ``task_incident`` key.

Coverage is IMPORT-based: the sets below are the modules' own typed
vocabularies (``outcomes.BEST_EFFORT_REASON_CODES``, both sides of
``outcomes.ACCEPTANCE_BYPASS_REASON_BY_RAIL``, the ``REASON_*`` constants and
``queue_timeouts.TIMEOUT_TERMINAL_REASONS``), never a regex over source text,
so a reason minted outside a registry is a registry bug rather than a silent
raw code on the card. A code with no sentence still stays raw by design
(docs/DESIGN.md §4); the exemptions below are the codes that render nothing on
purpose.
"""

from __future__ import annotations

import pathlib
import time
import types
from types import SimpleNamespace

import pytest

from ouroboros import outcomes
from ouroboros.project_dialogue import (
    OUTCOME_PHASE_HEADLINE,
    TASK_CAUSE_PHRASES,
    _completion_verdict,
    outcome_phase,
)
from supervisor.queue_timeouts import TIMEOUT_TERMINAL_REASONS
from tests._cancel_intents_shared import qenv  # noqa: F401 - shared reaper fixture

REPO = pathlib.Path(__file__).resolve().parents[1]
TWIN = REPO / "web" / "modules" / "log_events.js"

# The codes that state nothing by design: a clean delivery has no cause, and the
# owner's own stop carries its marker instead of a sentence, on both surfaces.
SILENT_BY_DESIGN = frozenset({
    outcomes.REASON_FINAL_MESSAGE,
    outcomes.REASON_OWNER_REQUESTED_FINALIZATION,
    outcomes.ACCEPTANCE_BYPASS_REASON_BY_RAIL[outcomes.REASON_OWNER_REQUESTED_FINALIZATION],
})


def _imported_reason_codes() -> set:
    named = {value for name, value in vars(outcomes).items()
             if name.startswith("REASON_") and isinstance(value, str)}
    return (
        named
        | set(outcomes.BEST_EFFORT_REASON_CODES)
        | set(outcomes.ACCEPTANCE_BYPASS_REASON_BY_RAIL)
        | set(outcomes.ACCEPTANCE_BYPASS_REASONS)
        | set(TIMEOUT_TERMINAL_REASONS)
    ) - SILENT_BY_DESIGN


# ---------------------------------------------------------------- the vocabulary

def test_every_imported_reason_code_has_one_sentence_in_both_twins():
    codes = _imported_reason_codes()
    assert {"round_limit", "finalization_grace", "deadline_local", "children_unabsorbed",
            "context_overflow", "absolute_ceiling", "deadline", "idle_timeout",
            "provider_failure", "empty_final_text"} <= codes
    twin = TWIN.read_text(encoding="utf-8")
    for code in sorted(codes):
        assert code in TASK_CAUSE_PHRASES, f"no owner sentence for typed reason {code!r}"
        assert f'{code}: "{TASK_CAUSE_PHRASES[code]}",' in twin, f"the browser twin lacks {code!r}"


def test_the_twins_carry_byte_identical_sentences_for_every_key():
    twin = TWIN.read_text(encoding="utf-8")
    for code, sentence in TASK_CAUSE_PHRASES.items():
        assert f'{code}: "{sentence}",' in twin, code


def test_the_timeout_rails_are_one_typed_tuple_in_priority_order():
    from supervisor import queue_timeouts

    assert TIMEOUT_TERMINAL_REASONS == ("absolute_ceiling", "deadline", "idle_timeout")
    assert (queue_timeouts.REASON_ABSOLUTE_CEILING, queue_timeouts.REASON_DEADLINE,
            queue_timeouts.REASON_IDLE_TIMEOUT) == TIMEOUT_TERMINAL_REASONS


def test_c1_adds_no_status_word():
    assert list(OUTCOME_PHASE_HEADLINE.values()) == ["Working", "Done", "Done with warnings", "Failed", "Cancelled"]


# ---------------------------------------------------------------- the card line

def _reaped(reason: str) -> dict:
    return {"status": "failed", "reason_code": reason,
            "outcome_axes": outcomes.terminal_outcome_axes(
                lifecycle="failed", execution=outcomes.EXECUTION_INFRA_FAILED,
                reason_code=reason, review_trigger="supervisor_terminal")}


@pytest.mark.parametrize("reason", TIMEOUT_TERMINAL_REASONS)
def test_a_reaped_row_states_the_rail_in_owner_words(reason):
    record = _reaped(reason)
    assert outcome_phase(record, {}) == "error"
    expected = f"{TASK_CAUSE_PHRASES[reason]}."
    assert _completion_verdict(record, {}) == _completion_verdict({}, record) == expected


def test_a_railed_best_effort_row_states_the_rail_in_owner_words():
    railed = {"status": "completed", "reason_code": "round_limit",
              "outcome_axes": {"execution": {"status": "best_effort", "reason_code": "round_limit"}}}
    assert outcome_phase(railed, {}) == "warn"
    assert _completion_verdict(railed, {}) == "The task hit its round limit before it could finish cleanly."


def test_an_unknown_rail_code_still_stays_raw_on_the_row():
    assert _completion_verdict(_reaped("some_future_rail"), {}) == "some_future_rail."


# ---------------------------------------------------------------- the reaper's surfaces

def test_the_grace_toast_speaks_the_sentence_and_keeps_the_typed_incident(tmp_path, monkeypatch):
    from supervisor import task_reaper, workers

    put: list = []
    monkeypatch.setattr(workers, "get_event_q", lambda: types.SimpleNamespace(put=put.append), raising=False)
    assert task_reaper.request_finalization_grace(tmp_path, "t1", "idle_timeout", chat_id=7, stamp=1000)
    (toast,) = put
    assert toast["text"].startswith(
        "⏳ Task t1: The task made no progress for too long. Finalize artifacts/results now; ")
    assert "idle_timeout" not in toast["text"]
    assert toast["progress_meta"] == {"task_incident": "idle_timeout", "toast_once": "t1:idle_timeout:1000",
                                      "toast_tone": "warning"}
    # An unknown rail keeps its raw code on the toast rather than borrowing a sentence.
    put.clear()
    task_reaper.request_finalization_grace(tmp_path, "t2", "some_future_rail", chat_id=7, stamp=1000)
    assert put[0]["text"].startswith("⏳ Task t2: some_future_rail. Finalize")
    assert put[0]["progress_meta"]["task_incident"] == "some_future_rail"
    assert put[0]["progress_meta"]["toast_tone"] == "warning"


def test_the_salvage_line_names_the_cause_after_the_supervisor_stop(tmp_path, monkeypatch):
    from ouroboros import observability
    from supervisor import task_reaper, terminal_delivery

    delivered: list = []
    monkeypatch.setattr(observability, "latest_llm_response_text", lambda *_a, **_k: "partial work")
    monkeypatch.setattr(observability, "preserved_salvage_path", lambda *_a, **_k: "")
    monkeypatch.setattr(terminal_delivery, "deliver_unreviewed_salvage",
                        lambda drive, task, tid, **kw: delivered.append({"task_id": tid, **kw}))
    q = types.SimpleNamespace(DRIVE_ROOT=tmp_path, _task_drive_for_task=lambda task, tid: tmp_path)
    task = {"id": "r1", "chat_id": 4}
    task_reaper._deliver_reap_salvage(q, task, "r1", "absolute_ceiling")
    (call,) = delivered
    # TZ-2 C1: the reaper hands over the TYPED rail, not a prose string.
    assert call["reason_code"] == "absolute_ceiling" and call["outcome"] == ""
    # The real builder frames it as one owner line, the code nowhere in the chat text.
    event = terminal_delivery.build_unreviewed_salvage_event(
        tmp_path, task, "r1", outcome=call["outcome"], reason_code=call["reason_code"],
        salvaged_text=call["salvaged_text"])
    assert event["text"].startswith(
        "⚠️ Task r1 was stopped by the supervisor. The task reached its maximum running time. "
        "Below is the last persisted intermediate model message")
    assert "absolute_ceiling" not in event["text"]
    assert event["reason_code"] == "absolute_ceiling"  # the code rides the event, not the prose
    # An unknown rail stays raw rather than borrowing a sentence.
    raw = terminal_delivery.build_unreviewed_salvage_event(
        tmp_path, task, "r1", outcome="", reason_code="some_future_rail", salvaged_text="")
    assert raw["text"].startswith("⚠️ Task r1 was stopped by the supervisor. some_future_rail.")


def test_the_kill_notice_leads_with_the_cause_and_keeps_the_typed_task_done(qenv, monkeypatch):  # noqa: F811
    import ouroboros.tools.services as services_mod
    from supervisor import task_reaper as tr
    from supervisor import workers as workers_mod

    events: list = []
    sent: list = []
    monkeypatch.setattr(tr, "_kill_and_confirm_worker_dead", lambda *_a, **_kw: True)
    monkeypatch.setattr(tr, "_deliver_reap_salvage", lambda *_a, **_kw: None)
    monkeypatch.setattr(tr, "send_with_budget", lambda cid, text, **kw: sent.append((cid, text, kw)))
    monkeypatch.setattr(services_mod, "archive_task_service_logs", lambda *a, **k: None)
    monkeypatch.setattr(workers_mod, "get_event_q", lambda: types.SimpleNamespace(put=events.append))
    monkeypatch.setattr(qenv.q, "reconstruct_task_cost",
                        lambda tid, fields=True, **_kw: {"cost_accounting_status": "available",
                                                         "cost_final": True, "cost_usd": 0.0})
    tr.reap_timed_out_task({
        "worker_id": 0, "proc": None, "task_id": "reap1", "task": {"id": "reap1", "chat_id": 4},
        "task_type": "chat", "terminal_reason": "absolute_ceiling", "attempt": 3, "owner_chat_id": 0,
        "runtime_sec": 10.0, "will_retry": False, "ceiling_reached": True,
    })
    (chat_id, text, kw), = sent
    assert chat_id == 4
    assert text.startswith("🛑 The task reached its maximum running time: task reap1 killed after 10s.\n")
    assert "Absolute ceiling reached; task stopped." in text
    assert "absolute_ceiling" not in text
    assert kw["progress_meta"]["task_incident"] == "task_reaper_stopped"
    (done,) = [e for e in events if e.get("type") == "task_done"]
    assert done["reason_code"] == "absolute_ceiling"
    assert done["outcome_axes"]["execution"]["reason_code"] == "absolute_ceiling"


# ---------------------------------------------------------------- the loop's fallback

def _limit_ctx(tmp_path):
    import ouroboros.loop as loop_mod

    return loop_mod._RoundLimitContext(
        messages=[], llm=object(), active_model="model", active_effort="high",
        max_retries=1, drive_logs=tmp_path / "logs", task_id="task", round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="task",
        active_use_local=False, max_rounds=1, deadline_ts=time.time() - 1,
        tools=SimpleNamespace(_ctx=SimpleNamespace()), llm_trace={},
    )


def test_the_loop_fallback_speaks_the_rail_and_keeps_an_unknown_rail_raw(tmp_path, monkeypatch):
    import ouroboros.loop as loop_mod

    monkeypatch.setattr(loop_mod, "_finalize_forced_services", lambda *_args: None)
    monkeypatch.setattr(loop_mod, "_call_forced_model_once",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("paid final call")))
    monkeypatch.setattr(loop_mod, "_forced_fallback_result",
                        lambda _ctx, _trace, text, reason, **_kw: (text, _ctx.accumulated_usage, _trace))
    for rail, expected in (
        ("absolute_ceiling", "⚠️ The task reached its maximum running time; finalization grace produced no answer."),
        ("deadline", "⚠️ The task reached its deadline; finalization grace produced no answer."),
        ("", "⚠️ The task reached its deadline; finalization grace produced no answer."),
        ("some_future_rail", "⚠️ Task reached some_future_rail; finalization grace produced no answer."),
    ):
        ctx = _limit_ctx(tmp_path)
        text, usage, _trace = loop_mod._handle_forced_finalization(ctx, rail)
        assert text == expected, rail
        assert usage["reason_code"] == "finalization_grace"
