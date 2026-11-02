"""A full local concurrency limit is narrated as local capacity, never as a provider outage.

The real limiter's refusal (its one slot held by a live holder, the waiter's bound spent)
is classified ``transport_unavailable`` with the typed ``model_capacity_unavailable`` code,
and no request is sent. The transport episode keeps its own wait, pacing and retries; only
its words follow that code, per attempt: a later attempt that fails another way, or an
earlier one whose outcome is unknown, keeps the wording it had.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ouroboros.loop_transport as loop_transport
from ouroboros import model_concurrency as mc
from tests.test_loop_transport_wait import _FakeClock
from tests.test_loop_transport_wait_interactive import _ctx, _enter, _NoteRecorder, _run_until_terminal

# Real file locks, as in test_model_concurrency.
pytestmark = pytest.mark.serial

ROUTE = {"provider": "openrouter", "base_url": "https://provider.invalid/v1", "api_key": "fake-secret"}
TERMINAL = dict(is_context_overflow=False, is_transport_wait=True, is_deadline_exhausted=False,
                wait_cause="transport_unavailable")


@pytest.fixture
def refused_usage(monkeypatch, tmp_path):
    """The usage facts ``loop_llm_call`` stamps from a real slot refusal and the real classifier.

    The refusal goes through the real physical-attempt accounting, whose released capture is
    what makes it a never-sent ``transport_unavailable``; the bare limiter exception alone
    classifies as a transient provider failure."""
    from ouroboros import config
    from ouroboros import usage_accounting as ua
    from ouroboros.loop_llm_call import classify_llm_exception
    from ouroboros.task_results import write_task_result

    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    mc.reset_for_tests()
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    scope = ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root")
    sends = []
    wrapped = mc.limit_physical_send(ROUTE, lambda: sends.append(True) or {"answer": "ok"})
    try:
        with ua.usage_scope(scope), mc.model_call_slot("", route=ROUTE):  # another live holder fills the one slot
            with mc.model_call_scope("waiting", deadline_ts=0):
                with pytest.raises(mc.ModelSlotTimeout) as refused:
                    ua.execute_physical_attempt(
                        ua.AttemptRequest(model="m", provider="openrouter", reservation_usd=.2), wrapped,
                        extractor=lambda _r: ({"prompt_tokens": 3, "completion_tokens": 2}, .1, True))
    finally:
        mc.reset_for_tests()
    assert sends == [] and getattr(refused.value.physical_attempt_capture, "state", None) == "released"
    classification = classify_llm_exception(refused.value)
    assert (classification.kind, classification.provider_code) == ("transport_unavailable", mc.ModelSlotUnavailable.code)
    return {"_last_llm_error_kind": classification.kind, "_last_llm_provider_code": classification.provider_code,
            "_last_llm_attempt_custody": "released"}


@pytest.mark.parametrize("interactive,waited", [(True, 180.0), (False, 180.0), (True, 0.0)])
def test_the_terminal_names_local_capacity_and_the_unsent_last_request(refused_usage, interactive, waited):
    text = loop_transport.provider_terminal_fallback_text(refused_usage, waited_sec=waited, interactive=interactive,
                                                          **TERMINAL)

    assert text.startswith("⚠️ Local model capacity stayed unavailable: ")
    assert "Concurrent Requests per Model" in text and "so it was not sent" in text
    assert "outage" not in text and "did not reach the provider" not in text
    assert text.endswith("Last reported code: model_capacity_unavailable.")
    assert ("this turn" in text) == interactive and ("3.0 min" in text) == bool(waited)


def test_another_released_failure_and_an_unknown_outcome_keep_their_own_words(refused_usage):
    from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY

    connect = {**refused_usage, "_last_llm_provider_code": "connect_error"}
    assert "ended as an outage" in loop_transport.provider_terminal_fallback_text(
        connect, waited_sec=60.0, interactive=True, **TERMINAL)
    # An earlier attempt with no known outcome: nothing claims that no request was sent.
    unknown = {**refused_usage, TRANSPORT_DEATHS_KEY: {"attempt-1": {}}}
    text = loop_transport.provider_terminal_fallback_text(unknown, waited_sec=60.0, interactive=True, **TERMINAL)
    assert text.startswith("⚠️ The dispatched attempt has no confirmed provider outcome.")
    assert "not sent" not in text and "Local model capacity" not in text


def test_the_wait_notes_follow_the_latest_attempt(refused_usage, tmp_path, monkeypatch):
    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    usage = dict(refused_usage)
    ctx = _ctx(is_direct_chat=True, _accumulated_usage=usage)
    notes = _NoteRecorder()

    episode = _enter(tmp_path, ctx, notes)
    episode.last_note_monotonic = clock.now - 10_000.0  # force a periodic note
    _run_until_terminal(episode, tmp_path, SimpleNamespace(_ctx=ctx), notes)

    assert notes.texts[0].startswith("🌐 No local model slot was free, so this request was not sent")
    assert any(text.startswith("🌐 Still waiting for a free local model slot") and "latest request was not sent"
               in text for text in notes.texts[1:-1])
    assert notes.texts[-1].startswith("🌐 Stopped waiting for a free local model slot")
    assert not any("outage" in text or "provider connection" in text for text in notes.texts)

    # A later attempt that got a slot and then could not connect is a connection wait again.
    usage["_last_llm_provider_code"] = "connect_error"
    again = _NoteRecorder()
    episode = _enter(tmp_path, ctx, again, task_id="t-connect")
    _run_until_terminal(episode, tmp_path, SimpleNamespace(_ctx=ctx), again, task_id="t-connect")
    assert "waiting and redialing automatically" in again.texts[0]
    assert "this turn ends as an outage" in again.texts[-1]
