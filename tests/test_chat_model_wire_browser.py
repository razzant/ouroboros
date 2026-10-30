"""Room-model writes through real SPA Send and direct owner admission to the HTTP model wire.

The existing isolated candidate/server fixture supplies only a loopback provider. No paid call or account login is
possible here. Model selection, per-Project isolation and the ordinary fallback walk run in the real server.
"""
from __future__ import annotations

import json
import re
import threading

import pytest

from tests import fixtures_mock_llm
from tests.test_effort_range_browser import _launch
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 -- fixture

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_new_room_work_reaches_the_chosen_model_and_keeps_automatic_fallback(direct_server_with_data, monkeypatch):  # noqa: F811
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    server["stop_server"]()
    settings_path = server["data_dir"] / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(OUROBOROS_MODEL="openai-compatible::mock-main",
                    OUROBOROS_MODEL_LIGHT="openai-compatible::mock-light",
                    OUROBOROS_MODEL_FALLBACKS="openai-compatible::mock-fallback",
                    OUROBOROS_TASK_REVIEW_MODE="off", OUROBOROS_SAFETY_MODE="off",
                    OUROBOROS_SUBAGENTS=json.dumps({"enabled": False, "items": []}))
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    server["start_server"]()
    received = []
    lock = threading.Lock()

    def provider(handler):
        body = json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
        with lock:
            received.append(body)
        broken = body["model"] == "mock-broken"
        # The production wire may carry text blocks and host framing, not a bare prompt string.
        markers = re.findall(r"\bROOM_WIRE_[A-Z_]+\b", json.dumps(body.get("messages", [])))
        assert markers, "The request must carry the current unique test marker"
        answer = f"WIRE_OK {markers[-1]}"
        payload = {"error": {"message": "model unavailable", "type": "invalid_request_error", "code": "model_not_found"}} if broken else {
            "id": "mock-chat", "object": "chat.completion", "model": body["model"],
            "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        streaming = body.get('stream') and not broken
        if streaming:
            chunks = [
                {'id': 'mock-chat', 'object': 'chat.completion.chunk', 'model': body['model'],
                 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': answer}, 'finish_reason': None}]},
                {'id': 'mock-chat', 'object': 'chat.completion.chunk', 'model': body['model'],
                 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': payload['usage']},
            ]
            data = (''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks) + 'data: [DONE]\n\n').encode()
        else:
            data = json.dumps(payload).encode()
        handler.send_response(404 if broken else 200)
        handler.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", provider)
    with sync_playwright() as pw:
        browser, page = _launch(pw, "webkit", viewport={"width": 1512, "height": 949})
        try:
            # Only suggestions are intercepted. The actual model requests reach the loopback HTTP server above.
            page.route("**/api/model-catalog", lambda route: route.fulfill(content_type="application/json", body=json.dumps({
                "items": [{"value": "openai-compatible::mock-picked", "name": "Picked", "provider_id": "openai-compatible"},
                          {"value": "openai-compatible::mock-broken", "name": "Broken", "provider_id": "openai-compatible"}],
                "errors": [], "model_sources": []})))
            page.goto(server["url"], wait_until="domcontentloaded")
            page.wait_for_function("() => document.querySelector('#page-chat .chat-model-picker')?.dataset.known === 'true'")

            def post(path, data):
                result = page.evaluate("async ([path, data]) => { const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)}); return {status:r.status, body:await r.json()}; }", [path, data])
                assert result["status"] == 200, result
                return result["body"]

            def send(marker, prefix="chat"):
                before = len(received)
                page.fill(f"#{prefix}-input", marker)
                page.click(f"#{prefix}-send")
                # Reloaded history may arrive after Send: only this request's unique reply proves completion.
                page.wait_for_function("([prefix, marker]) => document.querySelector('#' + prefix + '-messages')?.innerText.includes('WIRE_OK ' + marker) && document.querySelector('#' + prefix + '-input').value === ''", arg=[prefix, marker], timeout=60_000)
                # Wait for the room's own completion, not merely HTTP acceptance or an earlier answer.
                page.wait_for_function("async () => { const s = await (await fetch('/api/state')).json(); return s.active_chat_activities_complete && s.active_chat_activities.length === 0; }", timeout=60_000)
                with lock:
                    own = [body for body in received[before:] if body['model'] != 'mock-light'
                           and any(marker in str(m.get('content', '')) for m in body.get('messages', []))]
                assert own, (marker, [body["model"] for body in received[before:]])
                return [body["model"] for body in own]

            # Choose with the actual menu, then Send. No global role slot is rewritten.
            page.click("#page-chat .chat-model-trigger")
            page.locator(".chat-model-option", has_text="Picked").click()
            page.wait_for_function("() => document.querySelector('#page-chat .chat-model-picker').dataset.saving === 'false'")
            assert send("ROOM_WIRE_PICKED") == ["mock-picked"]
            page.reload(wait_until="domcontentloaded")
            page.wait_for_function("() => document.querySelector('#page-chat .chat-model-picker')?.dataset.moved === 'true'")
            assert send("ROOM_WIRE_RELOADED") == ["mock-picked"]

            project = post("/api/projects", {"name": "Wire room"})["project"]
            page.evaluate("project => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail:{project}}))", project)
            prefix = f"pchat-{project['id']}"
            page.wait_for_selector(f"#{prefix}-input")
            assert send("ROOM_WIRE_PROJECT_DEFAULT", prefix) == ["mock-main"]
            post("/api/owner/chat-model", {"project_id": project["id"], "kind": "fallback"})
            assert send("ROOM_WIRE_PROJECT_FALLBACK", prefix) == ["mock-fallback"]

            # A permanent model refusal uses the existing automatic chain, not a room-specific alternate algorithm.
            post("/api/owner/chat-model", {"project_id": project["id"], "kind": "model", "model": "openai-compatible::mock-broken"})
            models = send("ROOM_WIRE_AUTOMATIC", prefix)
            assert models[0] == "mock-broken" and models[-1] == "mock-fallback", models
            post("/api/owner/chat-model", {"project_id": project["id"], "kind": "main"})
            assert send("ROOM_WIRE_RESET", prefix) == ["mock-main"]
            saved = json.loads(settings_path.read_text(encoding="utf-8"))
            for key in ("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"):
                assert saved[key] == settings[key]
        finally:
            import os
            import pathlib
            from tests.test_effort_range_browser import _evidence_dir
            target = _evidence_dir(pathlib.Path(os.environ.get('OUROBOROS_TEST_TEMP_ROOT', '/tmp')))
            (target / 'wire-models.json').write_text(json.dumps([{'model': b.get('model'), 'stream': b.get('stream')} for b in received]))
            (target / 'wire-page.txt').write_text(page.locator('body').inner_text())
            events = server['data_dir'] / 'logs/events.jsonl'
            if events.exists():
                rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
                failures = [row for row in rows if 'error' in str(row.get('type', '')) or 'failed' in str(row.get('type', ''))]
                (target / 'wire-failures.json').write_text(json.dumps(failures, indent=2))
                print('WIRE_DIAGNOSTICS', [b.get('model') for b in received], failures[-3:])
            browser.close()
