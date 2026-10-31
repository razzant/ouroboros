"""Emoji-free host chrome through the real SPA, WS decoder, and durable history.

Only synthetic host frames and stored rows are supplied at their public boundary.
The candidate fixture binds the server's Python and static bytes to this checkout.
Copy is observed at the clipboard boundary; the actual button/listener runs.
"""
from contextlib import contextmanager
import hashlib
import json
import shutil

import pytest

from tests.ci_evidence import output_dir
from tests.test_ui_smoke_playwright import direct_server_with_data as _server
from tests.ui_console_errors import PageErrors
from tests.ui_chat_viewport_smoke import (
    _CAPTURE_TEST_SOCKET, _OBSERVE_STATE_READS, _emit_ws_frame, _wait_socket_open_quiescent, wait_drawer_closed,
)

direct_server_with_data = _server
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

_CLIPBOARD = """() => {
    window.__copied = [];
    Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {
        writeText: async text => { window.__copied.push(text); }
    }});
}"""


@contextmanager
def _browser(server, request, engine):
    from playwright.sync_api import sync_playwright

    evidence = (output_dir(request.config) or server["data_dir"].parent / "emoji-evidence") / request.node.name
    evidence.mkdir(parents=True, exist_ok=True)
    facts = {"candidate_identity": server["candidate_identity"], "repo": str(server["repo_dir"]),
             "url": server["url"], "runtime_version": server["candidate_version"], "engine": engine}
    (evidence / "subject.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        facts["browser_version"] = browser.version
        (evidence / "subject.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
        context = browser.new_context(viewport={"width": 1280, "height": 900}, color_scheme="dark",
                                      record_har_path=str(evidence / "network.har"))
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        sentinel = page.request.get(server["url"] + "/static/candidate-sentinel.txt").body()
        assert sentinel == server["candidate_sentinel"]
        facts["sentinel_sha256"] = hashlib.sha256(sentinel).hexdigest()
        facts["health"] = page.request.get(server["url"] + "/api/health").json()
        assert facts["health"]["runtime_version"] == server["candidate_version"]
        (evidence / "subject.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
        errors, events = PageErrors(page, engine), []
        page.on("console", lambda msg: events.append({"type": msg.type, "text": msg.text}))
        page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
        page.add_init_script(f"({_OBSERVE_STATE_READS})()")
        page.add_init_script(f"({_CLIPBOARD})()")
        try:
            yield page, evidence, errors
        finally:
            page.screenshot(path=str(evidence / "final.png"), full_page=True)
            (evidence / "dom.html").write_text(page.content(), encoding="utf-8")
            (evidence / "console.json").write_text(json.dumps({"errors": errors, "events": events}, indent=2),
                                                   encoding="utf-8")
            context.tracing.stop(path=str(evidence / "trace.zip"))
            context.close()
            browser.close()
            server["stop_server"]()
            if (server["data_dir"] / "logs").exists():
                shutil.copytree(server["data_dir"] / "logs", evidence / "server-logs", dirs_exist_ok=True)


def _open(page, url, *, reload=False):
    if reload:
        page.reload(wait_until="domcontentloaded")
    else:
        page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector('#chat-messages[data-history-hydrated="true"]', state="attached")
    _wait_socket_open_quiescent(page)


def _persist(root, rows):
    from supervisor.message_bus import log_chat

    for row in rows:
        direction = {"user": "in", "assistant": "out", "system": "system"}[row["role"]]
        log_chat(direction, row["chat_id"], 0, row["content"], drive_root=root,
                 ts=row["ts"], record_type=row.get("system_type", ""),
                 message_meta={key: row[key] for key in ("narration", "skill", "job_id") if key in row})


def _rows(chat_id):
    cases = [
        ("system", "startup_notice", "♻️ Restored pending queue from snapshot: old startup", None),
        ("system", "startup_notice", "Restored pending queue from snapshot: new startup", None),
        ("system", "retired_settings_notice", "⚙️ Settings: old setting", None),
        ("system", "unknown_notice", "♻️ Restored pending queue from snapshot: unknown type", None),
        ("system", "startup_notice", "🦊 An unknown opening stays", None),
        ("user", "", "♻️ Restored pending queue from snapshot: owner quote", None),
        ("assistant", "startup_notice", "♻️ Restored pending queue from snapshot: model quote", None),
        ("system", "reminder", "🦊 Remember the authored reminder", None),
        ("system", "skill_notice", "🦊 Keep the authored skill notice", None),
    ]
    return [{"type": "chat", "role": role, "system_type": kind, "content": text, "chat_id": chat_id,
             "ts": f"2026-10-10T12:00:{index:02d}+00:00", **({"narration": narration} if narration is not None else {})}
            for index, (role, kind, text, narration) in enumerate(cases)]


def _check_rows(page, feed, rows):
    for index, row in enumerate(rows):
        text = row["content"]
        expected = text.split(" ", 1)[1] if index in (0, 2) else text
        bubble = page.locator(f"{feed} .chat-bubble.{row['role']}").filter(has_text=expected)
        assert bubble.count() == 1, (row, bubble.count())
        assert bubble.locator(":scope > .message").inner_text() == expected
        assert text in bubble.get_attribute("data-message-key"), "display cleanup must retain the raw message identity"
        bubble.hover()
        bubble.get_by_role("button", name="Copy message", exact=True).click()
        page.wait_for_function("text => window.__copied.at(-1) === text", arg=text)


def _screenshots(page, evidence, feed):
    first = page.locator(f'{feed} .chat-bubble[data-system-type="startup_notice"]').first
    for theme in ("dark", "light"):
        page.emulate_media(color_scheme=theme)
        page.wait_for_function("theme => document.documentElement.dataset.theme === theme", arg=theme)
        for width in (1280, 390):
            page.set_viewport_size({"width": width, "height": 900 if width > 400 else 844})
            if width < 980:
                wait_drawer_closed(page)
            page.evaluate("""() => Promise.all(document.getAnimations()
                .filter(a => typeof CSSTransition === 'function' && a instanceof CSSTransition)
                .map(a => a.finished.catch(() => null)))""")
            page.locator(feed).evaluate("node => { node.scrollTop = 0; }")
            page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
            facts = first.evaluate("""node => {
                const s = getComputedStyle(node), r = node.getBoundingClientRect();
                const sample = document.createElement('span'); sample.style.color = 'var(--system-bubble-bg)';
                node.append(sample); const amber = getComputedStyle(sample).color; sample.remove();
                return {background: s.backgroundColor, amber, border: s.borderRadius,
                    x: r.x, right: r.right, padding: s.padding, sender: node.querySelector('.sender').textContent};
            }""")
            assert facts["background"] == facts["amber"] and facts["sender"].strip() == "System", facts
            assert facts["border"] != "0px" and facts["padding"] != "0px", facts
            assert facts["x"] >= -1 and facts["right"] <= width + 1, facts
            (evidence / f"geometry-{theme}-{width}.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
            page.screenshot(path=str(evidence / f"{theme}-{width}.png"))
    page.set_viewport_size({"width": 1280, "height": 900})


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("surface", ["main", "project"])
def test_host_display_copy_authorship_reload_reconnect(direct_server_with_data, request, engine, surface):
    server = direct_server_with_data
    with _browser(server, request, engine) as (page, evidence, errors):
        project = None
        if surface == "project":
            response = page.request.post(server["url"] + "/api/projects", data={"name": "Emoji qualification"})
            assert response.ok, response.text()
            project = response.json()["project"]
        chat_id = project["chat_id"] if project else 1
        feed = f'[id="pchat-{project["id"]}-messages"]' if project else "#chat-messages"
        rows = _rows(chat_id)
        (evidence / "frames.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        def open_room(*, reload=False):
            _open(page, server["url"], reload=reload)
            if project:
                page.locator(f'.nav-project-row[data-project-id="{project["id"]}"]').click()
                page.locator(feed).wait_for(state="visible")
                page.wait_for_function("feed => document.querySelector(feed)?.dataset.historyHydrated === 'true'", arg=feed)
                _wait_socket_open_quiescent(page)

        open_room()
        for row in rows:
            _emit_ws_frame(page, row)
        _check_rows(page, feed, rows)
        keys = page.locator(f"{feed} .chat-bubble[data-message-key]:not([data-ephemeral])").evaluate_all("nodes => nodes.map(n => n.dataset.messageKey)")
        _persist(server["data_dir"], rows)
        history = page.request.get(server["url"] + f"/api/chat/history?chat_id={chat_id}").json()
        (evidence / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        for row in rows:
            assert any(stored.get("text") == row["content"] for stored in history["messages"]), row
        # A real socket close re-enters the production reconnect/history path.
        previous = page.evaluate("window.__testSockets.length")
        page.evaluate("window.__testSockets.find(s => s.readyState === WebSocket.OPEN).close()")
        page.wait_for_function("n => window.__testSockets.length > n", arg=previous)
        _wait_socket_open_quiescent(page)
        _check_rows(page, feed, rows)
        assert page.locator(f"{feed} .chat-bubble[data-message-key]:not([data-ephemeral])").evaluate_all("nodes => nodes.map(n => n.dataset.messageKey)") == keys
        open_room(reload=True)
        _check_rows(page, feed, rows)
        _screenshots(page, evidence, feed)
        assert not errors.unexpected(), errors


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("kind,text", [
    ("startup_notice", "♻️ Restored pending queue from snapshot: quoted by the narrator"),
    ("skill_review", "✅ Skill review round 1 — quoted by the narrator"),
], ids=["startup", "skill-review"])  # IDs name artifact folders; quoted payloads may contain ':'.
def test_authored_narration_with_stale_host_type(direct_server_with_data, request, engine, kind, text):
    server = direct_server_with_data
    with _browser(server, request, engine) as (page, evidence, errors):
        _open(page, server["url"])
        row = {"type": "chat", "role": "system", "narration": True, "is_progress": False,
               "system_type": kind, "content": text, "chat_id": 1,
               "ts": "2026-10-10T13:00:00+00:00"}
        _emit_ws_frame(page, row)
        live = page.locator(f'.chat-bubble[data-system-type="{kind}"] .message').inner_text()
        _persist(server["data_dir"], [row])
        _open(page, server["url"], reload=True)
        replay = page.locator(f'.chat-bubble[data-system-type="{kind}"] .message').inner_text()
        inset = page.locator(f'.chat-bubble[data-system-type="{kind}"]').evaluate(
            "node => parseFloat(getComputedStyle(node).paddingLeft)")
        assert inset > 0, "an authored row keeps ordinary bubble geometry, not disclosure padding"
        facts = {"row": row, "live": live, "history": replay, "inset": inset}
        (evidence / "narration.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
        assert live == replay == text, facts
        assert not errors.unexpected(), errors


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_supported_host_table_reaches_real_bubbles(direct_server_with_data, request, engine):
    from datetime import datetime, timedelta, timezone

    server = direct_server_with_data
    with _browser(server, request, engine) as (page, evidence, errors):
        _open(page, server["url"])
        prefixes = page.evaluate("""async () => {
            const {HOST_NOTICE_PREFIXES} = await import('/static/modules/host_notice_text.js');
            return [...HOST_NOTICE_PREFIXES].flatMap(([kind, rows]) => rows.map(prefix => [kind, prefix]));
        }""")
        facts = []
        for index, (kind, prefix) in enumerate(prefixes):
            text = prefix + f"sample-{index}"
            expected = text.split(" ", 1)[1]
            if prefix == "⚠️ PR #":
                expected = "Warning: " + expected
            row = {"type": "chat", "role": "system", "system_type": kind, "content": text, "chat_id": 1,
                   "ts": (datetime(2026, 10, 10, 17, tzinfo=timezone.utc) + timedelta(seconds=index)).isoformat()}
            _emit_ws_frame(page, row)
            bubble = page.locator(f'.chat-bubble[data-system-type="{kind}"]').filter(has_text=f"sample-{index}")
            target = bubble.locator('.skill-review-summary-main' if kind == "skill_review" else ':scope > .message')
            actual = target.inner_text()
            facts.append({"row": row, "expected": expected, "actual": actual})
        (evidence / "supported-table.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
        assert all(f["expected"] == f["actual"] for f in facts), [f for f in facts if f["expected"] != f["actual"]]
        assert not errors, errors
