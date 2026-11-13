"""#2107 in the real SPA over a real isolated core: the heads a late quiz answer and a press on a finished
task's card start are named by what they answer on their cards and in Main's work list, while the receipt under
each later answer that joins one in Main says only that it was saved, never that name (owner 12.11), live and
after a reload, at the owner's desktop and phone widths in both themes.

A real ``server.py`` (disposable candidate checkout, isolated data, the mock provider) receives the owner's
decisions over its own HTTP API and runs each head in its own conversation-lane carrier process. The mock
answers a head's first call with one read-only tool call (real work, so the head is a card) and holds its next
call, so both heads are live when the later answers arrive; Light's naming call gets no title, as when Light is
unavailable, so each head keeps its admission name. Frames, the engine and its version are written to
``OUROBOROS_UI_EVIDENCE_DIR``.
"""
from __future__ import annotations

import itertools
import json
import os
import threading
import time
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data  # noqa: F401
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, wait_drawer_closed

pytestmark = [pytest.mark.ui_browser]

QUESTION = ("Which database should the new billing service use for its write-heavy event log and the nightly "
            "reports?\nIt writes a lot and reads little.")
NAME = "Which database should the new billing service use for its write-heavy event log…"  # 80 characters
CARD = "Flights to Lisbon for the team offsite"
ACTION = {"type": "action", "label": "Book it", "submit": {"id": "book"}}
# `fits`: the whole line fits its bubble at that bubble's widest (its max-width share of the feed, within its margins),
# in whatever font this platform sets it.
LINE = """cmid => {
    const note = document.querySelector(`.chat-bubble.user[data-client-message-id="${cmid}"] .msg-routing-annotation`);
    if (!note) return null;
    const bubble = note.closest('.chat-bubble'), feed = bubble.parentElement;
    const [fs, bs, ns] = [feed, bubble, note].map((node) => getComputedStyle(node));
    const px = (style, ...sides) => sides.reduce((sum, side) => sum + (parseFloat(style[side]) || 0), 0);
    const column = feed.clientWidth - px(fs, 'paddingLeft', 'paddingRight');
    const max = bs.maxWidth.endsWith('%') ? column * parseFloat(bs.maxWidth) / 100 : parseFloat(bs.maxWidth) || column;
    const room = Math.min(max, column - px(bs, 'marginLeft', 'marginRight'))
        - px(bs, 'paddingLeft', 'paddingRight', 'borderLeftWidth', 'borderRightWidth') - px(ns, 'marginLeft', 'marginRight');
    return {text: note.textContent, cut: note.scrollWidth > note.clientWidth + 1, fits: note.scrollWidth <= room + 1};
}"""
TITLE = """id => {
    const title = document.querySelector(`.chat-live-card[data-task-id="${id}"] [data-live-title]`);
    if (!title) return null;
    const box = title.getBoundingClientRect();
    return {text: title.textContent, cut: title.scrollWidth > title.clientWidth + 1, visible: box.width > 0 && box.height > 0};
}"""
# Main's work list (the status control's Shade): each row's visible name and the whole name it says to assistive tech.
ROWS = """() => {
    const trigger = document.querySelector('#page-chat .rw-trigger');
    const surface = trigger ? document.getElementById(trigger.getAttribute('aria-controls')) : null;
    return Object.fromEntries([...(surface?.querySelectorAll('.rw-row') || [])].map((row) => {
        const name = row.querySelector('.rw-name'), box = name?.getBoundingClientRect();
        return [row.dataset.id, {text: (name?.textContent || '').trim(),
                                 label: row.querySelector('.rw-main')?.getAttribute('aria-label') || '',
                                 visible: Boolean(box && box.width > 0 && box.height > 0)}];
    }));
}"""


def _scripted_provider(monkeypatch):
    """The mock provider's script; ``release`` lets every held call answer."""
    from tests import fixtures_mock_llm

    release, calls, ids = threading.Event(), [], itertools.count(1)

    def respond(handler):
        payload = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0))))
        messages = payload.get("messages") or [{}]
        first = str(messages[0].get("content") or "")
        if first.startswith("Name this project from the owner's request below"):
            calls.append("light")
            message = {"role": "assistant", "content": ""}  # no title: the admission name stands
        elif payload.get("tools") and not any(m.get("role") == "tool" for m in messages):
            calls.append("work")
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{next(ids)}", "type": "function",
                "function": {"name": "knowledge_list", "arguments": "{}"}}]}
        elif payload.get("tools"):
            calls.append("held")
            release.wait(180)
            message = {"role": "assistant", "content": "Done."}
        else:
            message = {"role": "assistant", "content": "OK"}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if payload.get("stream") is True:
            delta = {**message, **({"tool_calls": [{**message["tool_calls"][0], "index": 0}]} if message.get("tool_calls") else {})}
            chunks = [{"id": "mock", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                      {"id": "mock", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage}]
            body = ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
            kind = "text/event-stream"
        else:
            body = json.dumps({"id": "mock", "object": "chat.completion", "usage": usage,
                               "choices": [{"index": 0, "finish_reason": finish, "message": message}]}).encode()
            kind = "application/json"
        handler.send_response(200)
        handler.send_header("Content-Type", kind)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", respond)
    return release, calls


def _seed(data_dir, monkeypatch):
    """While the core is stopped: task-1 asked two questions in Main and finished; task-2 showed a card and finished."""
    import supervisor.message_bus as mb
    from ouroboros.card_store import write_card
    from ouroboros.owner_quiz import reconcile_terminal, record_asked
    from ouroboros.task_results import write_task_result
    from supervisor.card_delivery import send_card

    monkeypatch.setattr(mb, "DATA_DIR", data_dir)
    bridge = mb.LocalChatBridge()
    bridge._broadcast_fn = lambda _frame: None
    for quiz_id, question in (("q1", QUESTION), ("q2", "Should the migrations run on deploy?")):
        record_asked(data_dir, "task-1", quiz_id=quiz_id, question=question, options=["sqlite", "postgres"],
                     assumption="sqlite meanwhile", chat_id=1)
        ok, detail = bridge.send_quiz(1, quiz_id, question, [{"label": "sqlite"}, {"label": "postgres"}],
                                      assumption="sqlite meanwhile", task_id="task-1")
        assert ok, detail
    reconcile_terminal(data_dir, "task-1")
    card = write_card(data_dir, chat_id=1, task_id="task-2", title=CARD, fallback_text="Book it or skip",
                      kind="components", body={"components": [ACTION], "data": {}})
    assert send_card(bridge, card, 1, task_id="task-2", interactive=True) == (True, "ok")
    for task_id in ("task-1", "task-2"):
        write_task_result(data_dir, task_id, "completed", chat_id=1, result="Asked and finished.")
    return card


def _decide(page, url, body):
    response = page.request.post(f"{url}/api/decisions", data=json.dumps(body),
                                 headers={"Content-Type": "application/json"})
    assert response.status in (200, 202), response.text()
    return response.json()


def _wait(predicate, what, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.25)
    raise AssertionError(f"timed out waiting for {what}")


def _heads(page, url):
    state = page.request.get(f"{url}/api/state").json()
    return [row["activity_id"] for row in state.get("active_chat_activities") or [] if row.get("kind") == "direct_chat"]


def _open(browser, url, width, height, theme):
    context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=2,
                                  color_scheme=theme)
    page = context.new_page()
    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)", timeout=60_000)
    if width < 700 and page.evaluate("() => document.body.classList.contains('nav-drawer-open')"):
        page.locator('#primary-sidebar [data-nav-page="chat"]').click()  # the phone opened its drawer: back to Main
        wait_drawer_closed(page)
    return context, page


def _shown(page, receipts, heads):
    return {"receipts": {cmid: page.evaluate(f"({LINE})({json.dumps(cmid)})") for cmid, _ in receipts},
            "titles": {task_id: page.evaluate(f"({TITLE})({json.dumps(task_id)})") for task_id, _ in heads}}


def _look(page, out, label, receipts, heads):
    """The shown receipts and card titles, then the frame and 2x crops of each card and receipt. When a name never
    appears, what the page shows instead is kept beside the frames (``<label>-unmet``) before the failure surfaces."""
    try:
        for cmid, name in receipts:
            page.wait_for_function(f"() => ({LINE})({json.dumps(cmid)})?.text === {json.dumps(name)}",
                                   timeout=60_000)
        for task_id, name in heads:
            page.wait_for_function(f"() => ({TITLE})({json.dumps(task_id)})?.text === {json.dumps(name)}",
                                   timeout=60_000)
    except Exception:
        page.screenshot(path=str(out / f"{label}-unmet.png"))
        (out / f"{label}-unmet.json").write_text(json.dumps(_shown(page, receipts, heads), indent=1,
                                                            ensure_ascii=False), encoding="utf-8")
        raise
    page.wait_for_timeout(400)
    facts = _shown(page, receipts, heads)
    page.screenshot(path=str(out / f"{label}.png"))
    crops = [(f'.chat-live-card[data-task-id="{task_id}"]', f"card{index + 1}") for index, (task_id, _) in enumerate(heads)]
    crops += [(f'.chat-bubble.user[data-client-message-id="{cmid}"]', f"receipt{index + 1}")
              for index, (cmid, _) in enumerate(receipts)]
    for selector, name in crops:  # centred, clear of the shell's floating top controls
        element = page.locator(selector)
        element.evaluate("el => el.scrollIntoView({block: 'center'})")
        page.wait_for_timeout(200)
        element.screenshot(path=str(out / f"{label}-{name}@2x.png"))
    return facts


def _list(page, out, label, heads):
    """Main's work list, opened from its status control while the heads run: each head's row names it."""
    trigger = page.locator("#page-chat .rw-trigger")
    trigger.click()
    for task_id, name in heads:
        page.wait_for_function(f"() => ({ROWS})()[{json.dumps(task_id)}]?.text === {json.dumps(name)}", timeout=60_000)
    page.wait_for_timeout(400)
    rows = page.evaluate(f"({ROWS})()")
    page.screenshot(path=str(out / f"{label}.png"))
    page.locator(f"#{trigger.get_attribute('aria-controls')}").screenshot(path=str(out / f"{label}-rows@2x.png"))
    page.keyboard.press("Escape")
    return {"rows": {task_id: rows.get(task_id) for task_id, _ in heads}}


# A short name: its temporary directory holds the server's TMPDIR, and the multiprocessing manager's socket
# below it must stay within macOS's 104-byte AF_UNIX path limit (#2517).
def test_frames(direct_server_with_data, monkeypatch):
    """Both heads and their rows in Main's work list name what the heads answer while both receipts say only saved,
    live and after a reload, 1148x808 and 390 wide, light and dark."""
    from playwright.sync_api import sync_playwright

    server, url = direct_server_with_data, direct_server_with_data["url"]
    out = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or server["data_dir"].parent) / "late-input-heads"
    out.mkdir(parents=True, exist_ok=True)
    release, calls = _scripted_provider(monkeypatch)
    server["stop_server"]()
    card = _seed(server["data_dir"], monkeypatch)
    server["start_server"]()
    engine = os.environ.get("OUROBOROS_UI_BROWSER_ENGINE", "chromium")
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch()
            facts = {"engine": browser.browser_type.name, "version": browser.version, "url": url}
            try:
                context, page = _open(browser, url, 1148, 808, "light")
                _decide(page, url, {"request_id": "r1", "decision_id": "quiz:task-1:q1", "option_index": 1})
                [quiz_head] = _wait(lambda: _heads(page, url), "the late answer's head")
                _wait(lambda: "held" in calls, "the late answer's head at work")
                _decide(page, url, {"request_id": "r2", "decision_id": "quiz:task-1:q2", "option_index": 0})
                press = f"card:{card['card_id']}:book"
                _decide(page, url, {"request_id": "p1", "decision_id": press, "revision": 1})
                card_head = _wait(lambda: [h for h in _heads(page, url) if h != quiz_head], "the press's head")[0]
                _wait(lambda: calls.count("held") >= 2, "the press's head at work")
                _decide(page, url, {"request_id": "p2", "decision_id": press, "revision": 1})
                receipts = [("quiz_late_answer:task-1:q2", "Message saved for task"),
                            (f"card_input:{card['card_id']}:p2", "Message saved for task")]
                heads = [(quiz_head, NAME), (card_head, CARD)]
                facts["live-1148x808-light"] = _look(page, out, "live-1148x808-light", receipts, heads)
                facts["list-1148x808-light"] = _list(page, out, "list-1148x808-light", heads)
                narrow_context, narrow = _open(browser, url, 390, 844, "dark")  # the phone, while both still run
                facts["live-390x844-dark"] = _look(narrow, out, "live-390x844-dark", receipts, heads)
                facts["list-390x844-dark"] = _list(narrow, out, "list-390x844-dark", heads)
                narrow_context.close()
                release.set()
                _wait(lambda: not _heads(page, url), "both heads to finish", timeout=120)
                facts["ended-1148x808-light"] = _look(page, out, "ended-1148x808-light", receipts, heads)
                context.close()
                for width, height, theme in ((390, 844, "light"), (1148, 808, "dark"), (390, 844, "dark")):
                    label = f"reload-{width}x{height}-{theme}"
                    context, page = _open(browser, url, width, height, theme)
                    facts[label] = _look(page, out, label, receipts, heads)
                    context.close()
                facts["provider_calls"] = calls
            finally:
                (out / "facts.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False), encoding="utf-8")
                browser.close()
    finally:
        release.set()
    for label, shown in facts.items():
        if isinstance(shown, dict) and "receipts" in shown:
            # The receipt is one line (its words stay in it): at the desktop width whole when the line fits its bubble
            # at that bubble's widest and ellipsized only when it does not.
            lines = shown["receipts"].values()
            assert "-390x" in label or all(row["cut"] != row["fits"] for row in lines), (label, shown)
            assert all(row["visible"] and not row["cut"] for row in shown["titles"].values()), (label, shown)
        if isinstance(shown, dict) and "rows" in shown:
            assert all(row and row["visible"] for row in shown["rows"].values()), (label, shown)
