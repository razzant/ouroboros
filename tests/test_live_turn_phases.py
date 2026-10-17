"""What the owner's surface is told during a call: phases, closing words, activity, and session windows.

Uses the driver harness of ``tests/test_live_turn.py`` (fake hub, engine, LLM and ledger).
"""

from __future__ import annotations

import threading

import pytest

from ouroboros import live_turn
from ouroboros.call_protocol import STATE_CLOSED, STATE_CONNECTING, STATE_LIVE, PeerAudio
from ouroboros.gateways.realtime_engine import ENGINE_GOOGLE, AudioIn, EngineError, ToolCall
from tests import test_live_turn as base
from tests.test_live_turn import _spec, _wait_for

harness = base.harness  # the same fixture, registered in this module


def test_an_owner_who_ends_the_call_while_it_prepares_never_opens_an_engine(harness):
    h = harness()
    h.hub.ended_by_owner = True  # the owner's surface said bye while the turn was reading memory
    h.thread = threading.Thread(target=lambda: setattr(h, "result", live_turn.run_live_turn(
        messages=h.messages, tools=h.tools, llm=h.llm, drive_logs=h.root / "logs", emit_progress=h.emit_progress,
        incoming_messages=h.incoming, task_id="t-call", task_type="task", budget_remaining_usd=10.0,
        event_queue=h.events, drive_root=h.root, spec=h.spec, task=h.task, context_captured_at=h.context_captured_at)),
        daemon=True)
    h.thread.start()
    text, usage, _trace = h.finish()
    assert h.llm.engines == [] and h.llm.routes == []  # nothing opened, nothing paid for
    assert usage["live_call"]["ended"] == "owner_bye" and "reason_code" not in usage
    assert h.hub.states[-1] == (STATE_CLOSED, "owner_bye") and h.hub.closed == ["owner_bye"]
    assert all(state != STATE_CONNECTING for state, _reason in h.hub.states)


def test_a_missing_key_ends_the_call_with_its_own_words_and_the_settings_step(harness):
    from ouroboros.gateways.realtime_transport import LiveEngineRefusal

    h = harness()

    def refuse(route):
        raise LiveEngineRefusal("missing_credential", "Configure GOOGLE_API_KEY for calls.")

    h.llm.open_live = refuse
    h.thread = threading.Thread(target=lambda: setattr(h, "result", live_turn.run_live_turn(
        messages=h.messages, tools=h.tools, llm=h.llm, drive_logs=h.root / "logs", emit_progress=h.emit_progress,
        incoming_messages=h.incoming, task_id="t-call", task_type="task", budget_remaining_usd=10.0,
        event_queue=h.events, drive_root=h.root, spec=h.spec, task=h.task, context_captured_at=h.context_captured_at)),
        daemon=True)
    h.thread.start()
    text, usage, _trace = h.finish()
    # A typed refusal is not "the voice service could not be reached".
    assert usage["live_call"]["ended"] == "missing_credential" and usage["reason_code"] == "live_engine_refused"
    assert ("closed", "Configure GOOGLE_API_KEY for calls.", "settings") in h.hub.words
    assert [state for state, _ in h.hub.states][:1] == [STATE_CONNECTING]


def test_a_tool_batch_tells_the_surface_it_is_working_and_then_idle(harness):
    h = harness()
    engine = h.start()
    engine.push(ToolCall(call_id="c1", name="web_search", arguments="{}"))
    assert _wait_for(lambda: ("idle", "") in h.hub.activity)
    assert h.hub.activity[:2] == [("working", "web_search"), ("idle", "")]
    h.stop()
    h.finish()


def test_a_call_compresses_from_its_watermarks(harness):
    h = harness()
    config = h.start().config
    assert config.compress_at_tokens == 131_072 - 8_192
    assert config.compress_at_tokens > config.compress_to_tokens > 0  # the view plus half of the room to talk
    h.stop()
    h.finish()


def test_microphone_audio_is_labelled_with_the_rate_the_session_declares(harness):
    h = harness()
    engine = h.start()
    assert _wait_for(lambda: (STATE_LIVE, "") in h.hub.states)
    h.hub.feed(PeerAudio(peer_id="owner-1", role="owner", pcm=b"\x01\x00" * 320))
    assert _wait_for(lambda: engine.sent_of(AudioIn))
    assert engine.sent_of(AudioIn)[0].rate == 16000  # Gemini Live refuses audio at any other rate
    h.stop()
    h.finish()


def test_a_provider_refusal_ends_the_call_with_its_own_words(harness):
    h = harness()
    engine = h.start()
    words = "user system instruction has 140000 tokens and the sum is above the 131072 limit"
    engine.push(EngineError(kind="invalid_request", message=words, retryable=False, provider_words=words))
    text, usage, _trace = h.finish()
    assert usage["live_call"]["ended"] == "invalid_request" and words in text
    assert ("closed", words, "") in h.hub.words  # no Try again: the same setup would be refused the same way


def test_a_fresh_session_replays_the_newest_rows_that_fit_and_names_the_rest(harness):
    from dataclasses import replace as _replace

    h = harness(spec=_spec(engine=ENGINE_GOOGLE, model="gemini-3.8-live"))
    call = live_turn._LiveCall(messages=h.messages, tools=h.tools, llm=h.llm, drive_logs=h.root / "logs",
                               emit_progress=h.emit_progress, incoming_messages=h.incoming, task_id="t-call",
                               task_type="task", event_queue=h.events, drive_root=h.root, spec=h.spec)
    small = _replace(call.route.fact, window_tokens=8_192 + 2_000)  # one reply plus a small room
    call.route = _replace(call.route, fact=small)
    for index in range(60):
        h.messages.append({"role": "user" if index % 2 else "assistant", "content": f"utterance {index} " + "word " * 40})
    view = call._history_view()
    assert view[0]["role"] == "user" and view[0]["content"].startswith("[Earlier in this call:")
    assert view[-1]["content"].startswith("utterance 59 ")
    assert all(not str(item["content"]).startswith("utterance 0 ") for item in view)
    unbounded = _replace(call.route, fact=None)
    call.route = unbounded
    assert len(call._history_view()) == 61  # the task row and every utterance when no route fact bounds it


def test_the_routes_density_scales_the_replay_and_the_kept_view():
    from types import SimpleNamespace

    from ouroboros.live_session_view import compression_watermarks, history_view

    fact = SimpleNamespace(window_tokens=8_192 + 4_000, reply_allowance_tokens=8_192)
    view = "view " * 400
    messages = [{"role": "system", "content": view}] + [
        {"role": "user", "content": f"utterance {index} " + "word " * 40} for index in range(60)]
    plain, dense = history_view(fact, messages, [], 1.0), history_view(fact, messages, [], 1.5)
    assert 1 < len(dense) < len(plain)  # the same room holds fewer rows when the route counts them larger
    (start, kept), (dense_start, dense_kept) = (compression_watermarks(fact, view, [], ratio) for ratio in (1.0, 1.5))
    assert start == dense_start == 4_000 and dense_kept > kept  # the room is fixed; the view it keeps weighs more


def test_a_replayed_row_weighs_its_tool_call_arguments_too():
    from types import SimpleNamespace

    from ouroboros.live_session_view import history_view

    fact = SimpleNamespace(window_tokens=8_192 + 4_000, reply_allowance_tokens=8_192)
    call = {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": "x" * 40_000}}
    messages = [{"role": "system", "content": "view"},
                {"role": "user", "content": "please write it"},
                {"role": "assistant", "content": "", "tool_calls": [call]},  # empty words, 10k tokens of arguments
                {"role": "tool", "tool_call_id": "c1", "content": "written"}]
    view = history_view(fact, messages, [])
    assert not any(item.get("tool_calls") for item in view)  # the arguments do not ride in for free
    assert view[0]["content"].startswith("[Earlier in this call:") and view[-1]["content"] == "written"
    small = dict(call, function={"name": "write_file", "arguments": "{}"})
    messages[2] = {"role": "assistant", "content": "", "tool_calls": [small]}
    assert any(item.get("tool_calls") for item in history_view(fact, messages, []))  # a small call still replays


def _replayed_text(view):
    from ouroboros.gateways.realtime_gemini import history_turn

    return "".join(part.get("text", "") for row in view for part in history_turn(row)["parts"])


def test_a_replayed_tool_call_weighs_what_gemini_replays_in_any_script():
    import json
    import math
    from types import SimpleNamespace

    from ouroboros.live_session_view import _view_tokens, history_view
    from ouroboros.utils import estimate_tokens

    fact, ratio = SimpleNamespace(window_tokens=131_072, reply_allowance_tokens=8_192), 1.2
    view = "x" * 360_000  # a large call view, as on a mature install
    arguments = json.dumps({"path": "notes.md", "content": "я" * 15_000}, ensure_ascii=False)
    messages = [{"role": "system", "content": view},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "w1", "type": "function", "function": {"name": "write_file", "arguments": arguments}}]},
                {"role": "tool", "tool_call_id": "w1", "content": "Written"}]
    kept = history_view(fact, messages, [], ratio)
    budget = (131_072 - 8_192 - _view_tokens(view, [], ratio)) // 2
    assert math.ceil(estimate_tokens(_replayed_text(kept)) * ratio) <= budget  # Cyrillic replays as itself
    assert any(item.get("tool_calls") for item in kept)  # and a call that fits at its real weight still replays


def test_the_replay_notice_is_budgeted_with_the_rows_it_names():
    from types import SimpleNamespace

    from ouroboros.live_session_view import _view_tokens, history_view
    from ouroboros.utils import estimate_tokens

    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "old " * 1_000},
                {"role": "user", "content": "x" * 100}]
    for room in (100, 200):
        fact = SimpleNamespace(window_tokens=8_192 + room, reply_allowance_tokens=8_192)
        kept = history_view(fact, messages, [])
        assert estimate_tokens(_replayed_text(kept)) <= (room - _view_tokens("x", [])) // 2
        assert kept[0]["content"].startswith("[Earlier in this call:")
    assert kept[-1]["content"] == "x" * 100  # with room for both, the newest row replays beside its notice


def test_sessions_that_die_before_answering_end_the_call_instead_of_reopening_forever(harness, monkeypatch):
    from ouroboros.gateways.realtime_engine import ResponseDone

    monkeypatch.setattr(live_turn, "RECONNECT_ATTEMPTS", 2)
    h = harness()
    engine = h.start()
    engine.push(ResponseDone(item_id="a1", status="completed", usage={}))  # the first session answered
    engine.end_stream()
    assert _wait_for(lambda: len(h.llm.engines) == 2 and h.llm.engines[1].config is not None)
    second = h.llm.engines[1]
    second.push(ResponseDone(item_id="a2", status="completed", usage={}))  # an answer proves the reopened one too
    second.end_stream()
    assert _wait_for(lambda: len(h.llm.engines) == 3 and h.llm.engines[2].config is not None)
    h.llm.engines[2].end_stream()  # from here every reopened session dies before answering anything
    assert _wait_for(lambda: len(h.llm.engines) == 4 and h.llm.engines[3].config is not None)
    h.llm.engines[3].end_stream()
    text, usage, _trace = h.finish()
    assert len(h.llm.engines) == 4  # two sessions in a row ended unproven: no more paid sessions
    assert usage["live_call"]["ended"] == "reconnect_failed"
    assert any(word[0] == "closed" and "kept ending" in word[1] and word[2] == "retry" for word in h.hub.words)


def test_planned_handovers_never_end_a_quiet_call(harness, monkeypatch):
    from ouroboros.gateways.realtime_engine import GoAway

    monkeypatch.setattr(live_turn, "RECONNECT_ATTEMPTS", 2)
    h = harness()
    h.start()
    for handed_over in range(3):  # three of the provider's own handovers while nobody speaks
        h.llm.engines[-1].push(GoAway(time_left_s=5))
        assert _wait_for(lambda: len(h.llm.engines) == handed_over + 2 and h.llm.engines[-1].config is not None)
    h.stop()
    text, usage, _trace = h.finish()
    assert usage["live_call"]["ended"] == "owner_stop" and usage["live_call"]["reconnects"] == 3


@pytest.mark.parametrize("proven_after_s, drops, engines, ended",
                         [(0.0, 4, 5, "owner_stop"), (3600.0, 2, 2, "reconnect_failed")])
def test_a_quiet_session_that_stayed_up_is_resumed_on_every_drop(harness, monkeypatch, proven_after_s, drops,
                                                                engines, ended):
    # Nobody speaks; each session's setup is answered and its connection then drops. A session that
    # stayed up past SESSION_PROVEN_S was a working connection; one that ends sooner proves nothing.
    monkeypatch.setattr(live_turn, "RECONNECT_ATTEMPTS", 2)
    monkeypatch.setattr(live_turn, "SESSION_PROVEN_S", proven_after_s)
    h = harness()
    h.start()
    for dropped in range(drops):
        assert _wait_for(lambda: sum(state == STATE_LIVE for state, _ in h.hub.states) == dropped + 1)
        h.llm.engines[-1].end_stream()
        assert _wait_for(lambda: len(h.llm.engines) == dropped + 2 or h.hub.closed)
    if ended == "owner_stop":
        assert _wait_for(lambda: sum(state == STATE_LIVE for state, _ in h.hub.states) == engines)
        h.stop()
    text, usage, _trace = h.finish()
    assert len(h.llm.engines) == engines and usage["live_call"]["ended"] == ended


def test_sessions_whose_connect_fails_silently_are_retried_with_a_wait_and_then_end_the_call(harness, monkeypatch):
    # The real adapter reports a failed connect as an event, never by raising from open().
    from ouroboros.gateways.realtime_engine import EngineError

    class Unreachable(base.FakeEngine):
        auto_ready = False

        async def open(self, config):
            await super().open(config)
            self._queue.put_nowait(EngineError("transport", "connect failed", retryable=True))

    monkeypatch.setattr(live_turn, "RECONNECT_ATTEMPTS", 3)
    monkeypatch.setattr(live_turn, "RECONNECT_BACKOFF_S", 0.0371)  # distinct from the driver's other waits
    waits = []
    real_sleep = live_turn.asyncio.sleep

    async def sleep(delay, *args, **kwargs):
        if delay in (0.0371, 0.0742, 0.1484):
            waits.append(delay)
        return await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(live_turn.asyncio, "sleep", sleep)
    h = harness()

    def open_live(route):
        h.llm.routes.append(route)
        h.llm.engines.append(Unreachable())
        return h.llm.engines[-1]

    h.llm.open_live = open_live
    h.start()
    text, usage, _trace = h.finish()
    assert len(h.llm.engines) == 3 and usage["live_call"]["ended"] == "reconnect_failed"
    assert waits == [0.0371, 0.0742]  # each reopen after a session that proved nothing waits longer
    assert not any(state == STATE_LIVE for state, _ in h.hub.states)


def test_a_working_session_clears_the_count_whatever_ends_it(harness, monkeypatch):
    from ouroboros.gateways.realtime_engine import Caption, GoAway

    monkeypatch.setattr(live_turn, "RECONNECT_ATTEMPTS", 3)
    monkeypatch.setattr(live_turn, "RECONNECT_BACKOFF_S", 0.01)
    h = harness()
    h.start()
    for lost in range(2):  # two sessions in a row end before proving themselves
        h.llm.engines[-1].end_stream()
        assert _wait_for(lambda: len(h.llm.engines) == lost + 2 and h.llm.engines[-1].config is not None)
    h.llm.engines[-1].push(Caption(speaker="owner", text="Hello", final=True, item_id="o1"))  # this one works
    h.llm.engines[-1].push(GoAway(time_left_s=5))
    assert _wait_for(lambda: len(h.llm.engines) == 4 and h.llm.engines[-1].config is not None)
    h.llm.engines[-1].end_stream()  # the first unproven loss after a working session is resumed
    assert _wait_for(lambda: len(h.llm.engines) == 5 and h.llm.engines[-1].config is not None)
    h.stop()
    text, usage, _trace = h.finish()
    assert usage["live_call"]["ended"] == "owner_stop"


def test_a_planned_handover_neither_spends_nor_clears_the_unplanned_count(harness, monkeypatch):
    from ouroboros.gateways.realtime_engine import GoAway

    monkeypatch.setattr(live_turn, "RECONNECT_ATTEMPTS", 3)
    h = harness()
    h.start()
    for lost in range(2):  # two sessions in a row end before proving themselves
        h.llm.engines[-1].end_stream()
        assert _wait_for(lambda: len(h.llm.engines) == lost + 2 and h.llm.engines[-1].config is not None)
    h.llm.engines[-1].push(GoAway(time_left_s=5))  # the provider's own handover still opens a session
    assert _wait_for(lambda: len(h.llm.engines) == 4 and h.llm.engines[-1].config is not None)
    h.llm.engines[-1].end_stream()  # and the next unproven loss is the third in a row
    text, usage, _trace = h.finish()
    assert len(h.llm.engines) == 4 and usage["live_call"]["ended"] == "reconnect_failed"


def _density(root):
    from ouroboros.capability_evidence import resolve_main_token_density, route_fingerprint

    return resolve_main_token_density(root, route_fingerprint(provider=ENGINE_GOOGLE, model="gemini-3.8-live"),
                                      "gemini-3.8-live")


def test_the_first_response_teaches_the_next_fit_how_the_route_counts_text(harness, monkeypatch):
    from ouroboros import capability_evidence
    from ouroboros.gateways.realtime_engine import ResponseDone
    from ouroboros.live_session_view import text_estimate

    h = harness()
    monkeypatch.setattr(capability_evidence, "canonical_evidence_root", lambda: h.root)
    monkeypatch.setattr(capability_evidence, "_DENSITY_MEMO", {})  # the process-wide write throttle
    h.messages[0]["content"] = "Память Уробороса. " * 3000  # a view large enough to measure
    engine = h.start()
    assert _wait_for(lambda: (STATE_LIVE, "") in h.hub.states)
    estimate = text_estimate(engine.config.instructions, engine.config.history, engine.config.tools)
    engine.push(ResponseDone(item_id="a1", status="completed", usage={"text_input_tokens": round(estimate * 1.2)}))
    # The witness sits under the identity the call's fit reads (``known_route``), on the fit's own basis.
    assert _wait_for(lambda: _density(h.root)[1] == "fresh_route_usage")
    assert abs(_density(h.root)[0] - 1.2) < 0.01
    h.stop()
    h.finish()


def test_text_sent_before_the_first_response_leaves_no_witness(harness, monkeypatch):
    from ouroboros import capability_evidence
    from ouroboros.call_protocol import PeerControl
    from ouroboros.gateways.realtime_engine import ContextItem, ResponseDone
    from ouroboros.live_session_view import text_estimate

    h = harness()
    monkeypatch.setattr(capability_evidence, "canonical_evidence_root", lambda: h.root)
    monkeypatch.setattr(capability_evidence, "_DENSITY_MEMO", {})  # the process-wide write throttle
    h.messages[0]["content"] = "Память Уробороса. " * 3000
    engine = h.start()
    assert _wait_for(lambda: (STATE_LIVE, "") in h.hub.states)
    h.hub.feed(PeerControl(peer_id="owner-1", role="owner", message={"type": "context", "text": "a typed line"}))
    assert _wait_for(lambda: engine.sent_of(ContextItem))
    estimate = text_estimate(engine.config.instructions, engine.config.history, engine.config.tools)
    engine.push(ResponseDone(item_id="a1", status="completed", usage={"text_input_tokens": round(estimate * 1.2)}))
    assert _wait_for(lambda: h.ledger.settled)
    h.stop()
    h.finish()
    assert _density(h.root) == (1.0, "cold_estimate")  # the session held more text than it was opened with
