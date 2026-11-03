"""#1697 in the real SPA: web_search progress through the actual host-progress chain, live and replayed.

``tools/search._web_search`` runs with controlled offline transports (no provider is
called). Its ``emit_progress_fn`` is a task's real ``Agent._bind_task_progress`` binding;
each event it queues goes through ``message_bus.send_with_budget``, the one send seam,
which writes the durable progress row into the running server's dialogue store and hands
the bridge its WebSocket payload. The browser receives those payloads live, then the
same rows again from history after a reload.
"""
import json
import queue
import sys
import types

import pytest

from tests.test_emoji_free_browser import _browser, _open, _server
from tests.ui_chat_viewport_smoke import _emit_ws_frame

direct_server_with_data = _server
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def _dispatch(monkeypatch, scenario, emit):
    """The real dispatcher over controlled transports; returns its tool result."""
    from ouroboros import llm, search_routes
    from ouroboros.tools import search
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "auto")
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_MODEL", "")
    monkeypatch.setattr(search_routes, "_package_discoverable", lambda name: True)
    monkeypatch.setitem(sys.modules, "anthropic", None)  # discoverable, yet its import fails at dispatch
    if scenario == "openai":
        # A controlled SDK stream: the attempt row precedes dispatch, the provider's own
        # web_search_call event is what says a search ran.
        from unittest.mock import MagicMock
        from tests.test_web_search import _FakeStream, _make_completed_event, _make_event

        monkeypatch.setenv("OPENAI_API_KEY", "fixture")
        monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", "openai")
        sdk = MagicMock()
        sdk.OpenAI.return_value.responses.create.return_value = _FakeStream([
            _make_event("response.web_search_call.searching", item_id="ws1", output_index=0, sequence_number=1),
            _make_event("response.output_text.delta", delta="Answer", content_index=0, item_id="m1",
                        output_index=1, sequence_number=2, logprobs=[]),
            _make_completed_event(10, 5)])
        monkeypatch.setitem(sys.modules, "openai", sdk)

    def openrouter(**kwargs):
        if scenario == "fallback":
            error = RuntimeError("HTTP 403 terminal refusal")
            error.status_code = 403
        else:
            error = RuntimeError("connection reset after send")
            error.physical_attempt_capture = PhysicalAttemptCapture("lost", "openai/gpt-5.2", "openrouter", "dispatched", "opaque")
        raise error

    monkeypatch.setattr(llm, "openrouter_web_search_server_tool", openrouter)
    if scenario == "unknown":
        monkeypatch.setattr("ouroboros.transport_custody.outcome_unknown_on_chain", lambda exc: True)
    monkeypatch.setattr(search, "_web_search_ddgs", lambda query: json.dumps(
        {"answer": "a", "answer_type": "summary", "sources": [{"url": "https://example.invalid/a"}], "backend": "ddgs"}))
    ctx = types.SimpleNamespace(pending_events=[], task_metadata={}, emit_progress_fn=emit)
    return json.loads(search._web_search(ctx, "neutral query"))


def _host_chain(monkeypatch, data_dir, task_id):
    """A task's real progress binding, and the real send seam into ``data_dir``."""
    from ouroboros.agent import OuroborosAgent
    from supervisor import message_bus

    events = queue.Queue()
    worker = types.SimpleNamespace(_event_queue=events, _current_chat_id=1, _current_task_id=task_id,
                                   _current_task_metadata={}, _last_progress_ts=0.0, tools=None)
    worker._emit_progress = lambda text, **kw: OuroborosAgent._emit_progress(worker, text, **kw)
    worker._subagent_progress_meta = lambda event: OuroborosAgent._subagent_progress_meta(worker, event)
    emit = OuroborosAgent._bind_task_progress(worker, task_id, 1, {}, None)
    bridge, frames = message_bus.LocalChatBridge(), []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "DATA_DIR", data_dir)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})

    def deliver():
        sent = []
        while not events.empty():
            evt = events.get_nowait()
            sent.append(evt)
            message_bus.send_with_budget(int(evt["chat_id"]), evt["text"], fmt=evt["format"], is_progress=True,
                                         task_id=evt["task_id"], progress_meta=evt["progress_meta"], ts=evt["ts"],
                                         role=evt["role"], system_type=evt["system_type"])
        return sent

    return emit, deliver, frames


@pytest.mark.parametrize("scenario", ["fallback", "unknown", "openai"])
def test_search_progress_reads_the_same_live_and_after_replay(direct_server_with_data, request, monkeypatch, scenario):
    from ouroboros.agent import OuroborosAgent  # noqa: F401 — the producer the chain binds

    server = direct_server_with_data
    task_id = f"search-{scenario}"
    with _browser(server, request, "chromium") as (page, evidence, errors):
        _open(page, server["url"])
        emit, deliver, frames = _host_chain(monkeypatch, server["data_dir"], task_id)
        result = _dispatch(monkeypatch, scenario, emit)
        events = deliver()
        assert events and all(evt["system_type"] == "host_progress" and evt["role"] == "system" for evt in events)
        assert len(frames) == len(events) and all(frame["type"] == "chat" for frame in frames)
        produced = [evt["text"] for evt in events]  # "💬 <line>" exactly as the worker queued it
        for frame in frames:
            _emit_ws_frame(page, frame)
        card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
        card.locator("[data-live-disclosure]").click()
        live = card.inner_text()
        card.scroll_into_view_if_needed()
        page.screenshot(path=str(evidence / f"search-progress-{scenario}-live.png"))
        _open(page, server["url"], reload=True)
        card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
        card.locator("[data-live-disclosure]").click()
        replay = card.inner_text()
        card.scroll_into_view_if_needed()
        page.screenshot(path=str(evidence / f"search-progress-{scenario}-replay.png"))
        facts = {"result": result, "events": events, "frames": frames, "live": live, "replay": replay}
        (evidence / "chain.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        for text in produced:
            words = text.removeprefix("💬 ").split(" ", 1)[1]  # the listed opening's mark is display chrome
            assert words in live and words in replay, facts
        assert not any(mark in live + replay for mark in ("🔍", "↪", "⚠️", "💬")), facts
        assert "Searching via" not in live + replay
        # Exception text stays in the tool result and Logs, never in the owner's rows.
        assert not any(raw in live + replay for raw in ("HTTP 403", "ModuleNotFoundError", "connection reset")), facts
        if scenario == "openai":
            assert result["answer"] == "Answer" and [leg["outcome"] for leg in result["legs_tried"]] == ["succeeded"]
            assert [text.removeprefix("💬 ") for text in produced] == [
                "🔍 Trying OpenAI web search: neutral query", "🔍 Searching: neutral query"]
        elif scenario == "fallback":
            assert result["backend"] == "ddgs"
            assert [leg["outcome"] for leg in result["legs_tried"]] == ["failed", "failed", "succeeded"]
            assert "HTTP 403" in result["legs_tried"][0]["error"]
            assert "Trying DDGS after Anthropic web search failed; details in Logs" in replay
            assert "Answer from DDGS after 2 failed sources" in replay
        else:
            assert result["reason_code"] == "provider_outcome_unknown"
            assert [leg["outcome"] for leg in result["legs_tried"]] == ["provider_outcome_unknown"]
            assert "Trying DDGS" not in replay and "Web search outcome unknown (OpenRouter)" in replay
        assert not errors, errors
