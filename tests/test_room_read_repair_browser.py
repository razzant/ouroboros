"""What the owner actually saw, and a Reply whose target heals, end to end in the real SPA.

A message shown in two rooms is read once (DESIGN "Unread"), and only once all of it was in view in
one room: a code block wider than its box shows only its viewport, so it is read by parts while it
scrolls sideways, and sideways scrolling never shows the rows outside the read band; rows a notice
lies over are not in view until it is gone. A Reply an older body saved to a Project message whose
stored bytes could not be read stays Main's and is a gap of the Project; once the target is stored
again, the same server shows it in its Project and the gap closes. A quiet scheduled run keeps its
sentence under its schedule's name in its own room across reloads (the history seam both share).
"""

import json
import uuid

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.ci_evidence import output_dir
from tests.system_e2e.harness import (
    ArtifactOracle,
    ModelGate,
    ScriptedStubModel,
    body_text,
    keyless_settings,
    start_server,
    wait_durable_result,
    wait_until,
)
from tests.test_main_notice_browser import (
    CARD_NAME, _NAV_FACTS, _around, _history, _nav, _open_drawer, _open_project, _prefs, _scroll_to, _shot,
)
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET
from tests.ui_console_errors import PageErrors
from tests.ui_failure_evidence import FailureEvidence

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.ui_browser]

ENGINES = pytest.mark.parametrize("engine,width,height,theme", [
    ("webkit", 1165, 851, "dark"), ("chromium", 1440, 900, "light"),
], ids=["webkit-desktop-dark", "chromium-desktop-light"])

# One shared notice's geometry in its feed: the bubble, its code block and the band the reader sees.
_NOTICE_FACTS = """({feed, mark}) => {
    const root = document.querySelector(feed);
    const node = [...root.querySelectorAll('.chat-bubble[data-system-type="main_notice"]')]
        .find((bubble) => bubble.textContent.includes(mark));  // a sleeping row keeps its text, not its innerText
    if (!node) return null;
    const box = node.getBoundingClientRect(), band = root.getBoundingClientRect();
    const header = root.closest('.page')?.querySelector('.chat-page-header');
    const composer = document.querySelector('#chat-input-area');
    const seen = [header?.getClientRects().length ? header.getBoundingClientRect().bottom - band.top : 0,
                  composer?.getClientRects().length ? composer.getBoundingClientRect().top - band.top : root.clientHeight];
    const pre = node.querySelector('.message pre');
    const code = pre && pre.getBoundingClientRect();
    const toasts = [...document.querySelectorAll('#page-chat .toast-stack--room .toast')].map((toast) => {
        const rect = toast.getBoundingClientRect();
        return {top: rect.top - band.top, bottom: rect.bottom - band.top, left: rect.left, right: rect.right,
                text: toast.textContent || ''};
    });
    return {top: box.top - band.top + root.scrollTop, height: box.height, bottom_on_screen: box.bottom - band.top,
            top_on_screen: box.top - band.top, left: box.left, right: box.right, seen, scroll_top: root.scrollTop,
            revisions: JSON.parse(node.dataset.roomRevisions || '{}'), toasts,
            pre: pre ? {top: code.top - band.top + root.scrollTop, height: pre.clientHeight, width: pre.clientWidth,
                        scroll_width: pre.scrollWidth, scroll_left: pre.scrollLeft} : null};
}"""

_SIDEWAYS = """async ({feed, mark, left}) => {
    const node = [...document.querySelector(feed).querySelectorAll('.chat-bubble[data-system-type="main_notice"]')]
        .find((bubble) => bubble.textContent.includes(mark));  // a sleeping row keeps its text, not its innerText
    const pre = node.querySelector('.message pre');
    pre.scrollLeft = left;  // a real scroll of the block: the browser fires its scroll event
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    return pre.scrollLeft;
}"""

_HOLD_TOAST = """async (text) => {
    const { showToast } = await import('/static/modules/toast.js');
    window.__heldToast = showToast(text, 'info', { ttl: 0, haptic: false, dismissLabel: 'Dismiss held notice',
        host: document.querySelector('#page-chat') });
    return Boolean(window.__heldToast?.isConnected);
}"""


def _read_ahead(page, chat):
    held = (_prefs(page).get("read_ahead_revisions") or {}).get(str(chat)) or []
    return {value for lo, hi in held for value in range(lo, hi + 1)}


def _count(page, project):
    return page.evaluate(_NAV_FACTS, project["id"])["projects_count"]


def _sideways(page, feed, mark, facts):
    """The block scrolled from its left edge to its right one by half its width, as a reader pans it."""
    left, end = 0, facts["pre"]["scroll_width"] - facts["pre"]["width"]
    while True:
        page.evaluate(_SIDEWAYS, {"feed": feed, "mark": mark, "left": min(left, end)})
        if left >= end:
            break
        left += facts["pre"]["width"] // 2


def _isolated_home(tmp_path, monkeypatch):
    from tests.system_e2e.harness import KeylessIsolatedServer

    home = tmp_path / "home"
    home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(home), "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config")})


@ENGINES
def test_a_wide_notice_is_read_once_all_of_it_was_in_view_and_rows_under_a_notice_are_not(
    wait_clone, tmp_path, monkeypatch, request, engine, width, height, theme,
):
    """Four Main notices of a Project turn, read in Main only. The short one is brought under a notice the
    room holds over its last lines: nothing is recorded until that notice is gone and the reader is at it
    again. The wide one's code block is panned across while only its top rows are in the band: nothing;
    panned across again with all of it in the band: recorded. Two long ones around them are never read whole
    here, and the Project's own answer stays unread; no cursor moves, and a reload shows both rooms the same rows."""
    from playwright.sync_api import sync_playwright

    tag = uuid.uuid4().hex[:12]
    marker = f"PROJECT_WIDE_{tag}"
    wide_mark, short_mark, answer_mark = f"WIDE_NOTICE_{tag}", f"SHORT_NOTICE_{tag}", f"WIDE_ANSWER_{tag}"
    long_mark, head_mark = f"LONG_NOTICE_{tag}", f"HEAD_NOTICE_{tag}"
    wide_notice = (f"Run these once in a terminal, in order. [{wide_mark}]\n\n```\n" + "\n".join(
        f"step {index:02d}: " + "titan --resume --segment=north --verify " * 9 for index in range(14)) + "\n```")
    short_notice = (f"Owner action: confirm the billing change in Settings. [{short_mark}]\n\n"
                    "It waits for you; the Project keeps working meanwhile.")
    def log(mark):
        return "\n\n".join([f"The series log, for later. [{mark}]"] + [
            f"Entry {index:02d}: the Titan series step finished and its checks passed; nothing here needs you now."
            for index in range(40)])
    started = ModelGate(lambda body: bool(body.get("tools")) and marker in body_text(body) and not any(
        str(m.get("role") or "") == "tool" for m in body.get("messages", []) if isinstance(m, dict)), timeout=300)

    def respond(body):
        if marker not in body_text(body):
            return {"final": "No scripted work for this actor."}
        done = sum(str(m.get("role") or "") == "tool" for m in body.get("messages", []) if isinstance(m, dict))
        if done < 4:
            return {"tool": "send_user_message", "arguments": {
                "text": (log(head_mark), short_notice, wide_notice, log(long_mark))[done], "reason": "owner-only remedy",
                "destination": "main"}}
        return {"final": f"Audit complete. [{answer_mark}]"}

    _isolated_home(tmp_path, monkeypatch)
    evidence = output_dir(request.config)
    with ScriptedStubModel([respond] * 12, final_answer=CARD_NAME, gate=started) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            project = _api(server.base_url, "POST", "/api/projects", {"name": "Wide notice room"})["project"]
            chat = int(project["chat_id"])
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=theme)
                errors = PageErrors(page, engine)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
                feed = "#chat-messages"
                with FailureEvidence(page, browser, evidence, request.node.nodeid, engine) as capture:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    _open_project(page, project)
                    page.locator(f'[id="pchat-{project["id"]}-input"]').fill(f"{marker} Audit and tell me in Main.")
                    page.locator(f'[id="pchat-{project["id"]}-send"]').click()
                    assert started.arrived.wait(120), "the Project turn never reached its first round"
                    _nav(page, "dashboard")  # away before anything is said
                    started.release.set()
                    task = wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
                        if marker in str(row.get("task", {}).get("text") or "")), None), 60)
                    assert wait_durable_result(oracle, task["id"], timeout=120)["status"] == "completed"
                    page.wait_for_function(
                        "() => document.querySelector('#nav-projects-count')?.textContent.trim() === '5'", timeout=45_000)
                    seen_before = (_prefs(page).get("project_seen_revision") or {}).get(project["id"], 0)
                    capture.checkpoint("1-away-five-unread")

                    # ---- the short notice lands under a notice the room holds over its last lines ----
                    assert page.evaluate(_HOLD_TOAST, "A notice held over the conversation's last lines for this "
                                                      "check. It stays until it is dismissed.")
                    _nav(page, "chat")  # the feed opens at its end: the long notice's last part
                    page.locator(f'{feed} .chat-bubble[data-system-type="main_notice"]').filter(
                        has_text=long_mark).wait_for(timeout=30_000)
                    page.wait_for_timeout(800)
                    short = page.evaluate(_NOTICE_FACTS, {"feed": feed, "mark": short_mark})
                    (held,) = [toast for toast in short["toasts"] if "held over" in toast["text"]]
                    # One move puts the short notice's end under the held notice, its start in the band.
                    _scroll_to(page, feed, short["scroll_top"] + short["bottom_on_screen"] - (held["top"] + held["bottom"]) / 2)
                    short = page.evaluate(_NOTICE_FACTS, {"feed": feed, "mark": short_mark})
                    (held,) = [toast for toast in short["toasts"] if "held over" in toast["text"]]
                    assert held["top"] < short["bottom_on_screen"] < held["bottom"], short
                    assert short["seen"][0] <= short["top_on_screen"] and held["left"] < short["right"], short
                    page.wait_for_timeout(1500)
                    revision = short["revisions"][str(chat)]
                    assert revision not in _read_ahead(page, chat) and _count(page, project) == "5"
                    _shot(page, request, "T1-short-notice-under-a-held-notice")
                    capture.checkpoint("2-under-a-notice")
                    page.evaluate("() => { for (const toast of document.querySelectorAll("
                                  "'#page-chat .toast-stack--room .toast')) toast.dispose?.(); }")
                    page.wait_for_function("() => !document.querySelector('#page-chat .toast-stack--room .toast')")
                    # Its end is a reading moment: no scroll, the reader is already at the short notice.
                    page.wait_for_function(
                        "() => document.querySelector('#nav-projects-count')?.textContent.trim() === '4'", timeout=30_000)
                    assert _read_ahead(page, chat) == {revision}
                    _shot(page, request, "T2-short-notice-read-once-the-notice-is-gone")

                    # ---- the wide notice: panned with only its top rows in the band, then with all of it ----
                    wide = page.evaluate(_NOTICE_FACTS, {"feed": feed, "mark": wide_mark})
                    above, below = wide["seen"]
                    code = wide["pre"]
                    assert code and code["scroll_width"] > 1.5 * code["width"], wide  # the shipped block clips its lines
                    assert wide["height"] < below - above - 20, wide  # all of it fits the band at once
                    _scroll_to(page, feed, code["top"] - below + int(0.4 * code["height"]))
                    partial = page.evaluate(_NOTICE_FACTS, {"feed": feed, "mark": wide_mark})
                    assert partial["pre"]["top"] - partial["scroll_top"] < below < partial["pre"]["top"] - partial["scroll_top"] + code["height"]
                    _sideways(page, feed, wide_mark, wide)
                    page.wait_for_timeout(1500)
                    wide_revision = wide["revisions"][str(chat)]
                    assert wide_revision not in _read_ahead(page, chat) and _count(page, project) == "4"
                    _shot(page, request, "W1-wide-panned-with-its-lower-rows-out-of-view")
                    capture.checkpoint("3-panned-partly")
                    _sideways(page, feed, wide_mark, {**wide, "pre": {**code, "scroll_width": code["width"]}})  # back to 0
                    _scroll_to(page, feed, wide["top"] - above - 12)
                    page.wait_for_timeout(800)
                    assert wide_revision not in _read_ahead(page, chat), "its box whole is not its content whole"
                    _sideways(page, feed, wide_mark, wide)
                    page.wait_for_function(
                        "() => document.querySelector('#nav-projects-count')?.textContent.trim() === '3'", timeout=30_000)
                    assert _read_ahead(page, chat) == {revision, wide_revision}
                    prefs = _prefs(page)
                    assert (prefs.get("project_seen_revision") or {}).get(project["id"], 0) == seen_before  # no cursor moved
                    _shot(page, request, "W2-wide-read-once-all-of-it-was-in-view")
                    capture.checkpoint("4-wide-read")

                    # ---- reload: the same rows in both rooms, the Project's own answer still unread ----
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    page.locator(f'{feed} .chat-bubble[data-system-type="main_notice"]').filter(
                        has_text=wide_mark).wait_for(timeout=30_000)
                    page.wait_for_function(
                        "() => document.querySelector('#nav-projects-count')?.textContent.trim() === '3'", timeout=30_000)

                    def address(rows, mark):
                        return [row.get("row_address") for row in rows if mark in str(row.get("text") or "")]
                    main_history, project_history = _history(page, 1), _history(page, chat)
                    for mark in (head_mark, wide_mark, short_mark, long_mark):
                        assert address(main_history, mark) == address(project_history, mark) != [], mark
                        (one,) = address(main_history, mark)
                        assert all(_around(page, room, one)["anchor"]["state"] == "found" for room in (1, chat))
                    assert address(project_history, answer_mark) and not address(main_history, answer_mark)
                    _shot(page, request, "R1-main-after-reload")
                    _open_project(page, project)
                    page.locator(f'[id="pchat-{project["id"]}-messages"]').get_by_text(answer_mark, exact=False).first.wait_for(
                        timeout=30_000)
                    _shot(page, request, "R2-project-after-reload")
                    assert not errors.unexpected(), errors
        finally:
            started.release.set()
            server.stop()
            assert server.proc.poll() is not None


@ENGINES
def test_a_repaired_reply_target_joins_its_project_and_a_quiet_run_keeps_its_words_across_reloads(
    wait_clone, tmp_path, monkeypatch, request, engine, width, height, theme,
):
    """An older body's Main Reply to a Project's completion summary whose stored line is torn: Main shows it,
    the Project does not and its history says why (``chat_reply_target_unreadable``). The summary's exact row
    is stored again while the same server runs: after a reload the Project shows the Reply at its one Main
    address and the gap is gone. The Project's quiet scheduled run shows its sentence under its schedule's name
    there, before and after, and never in Main."""
    from playwright.sync_api import sync_playwright

    from ouroboros import dialogue_store
    from ouroboros.chat_references import reference_from_row
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import write_task_result
    from ouroboros.terminal_projection import _project_row

    tag = uuid.uuid4().hex[:12]
    target_mark, reply_mark = f"REPAIRED_TARGET_{tag}", f"LEGACY_REPLY_{tag}"
    sentence = f"Checked the disk and the gateway; nothing needed you. [QUIET_WORDS_{tag}]"
    data = tmp_path / "instance" / "data"
    data.mkdir(parents=True)
    project = create_project(data, "repair-room", name="Repair room")
    chat = int(project["chat_id"])
    write_task_result(data, "repair-root", "completed", chat_id=chat, title="Repair root", result="Series finished.")
    target = {"ts": "2026-10-01T05:00:00+00:00", "direction": "system", "chat_id": 1, "task_id": "repair-root",
              "type": "project_completion_summary", "text": f"Repair room finished its series. [{target_mark}]"}
    legacy = {key: value for key, value in reference_from_row(target).items() if key != "target_home"}
    reply = {"ts": "2026-10-01T06:00:00+00:00", "direction": "in", "chat_id": 1, "client_message_id": f"legacy-{tag}",
             "text": f"Why is this not in the Repair room? [{reply_mark}]", "reply_to": legacy}
    quiet = write_task_result(data, "quiet-run", "completed", chat_id=chat, root_task_id="quiet-run",
                              delegation_role="root", title="Daily check", suggested_name="Daily check",
                              schedule_name="Daily check", wake_summary=sentence, wake_silent=True,
                              metadata={"schedule_occurrence": {"schedule_id": "daily", "token": "t1"}},
                              result="PRIVATE NOTE", terminal_origin="model_final", _terminal_observed=True)
    terminal = {**_project_row("quiet-run", quiet, {}, {"chat_id": chat}), "ts": "2026-10-01T07:00:00+00:00"}
    archive = data / "archive" / "chat_20261001T000000.jsonl"
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps(target)[:48] + "\n" + json.dumps(reply) + "\n" + json.dumps(terminal) + "\n",
                       encoding="utf-8")
    narration = {"task_id": "quiet-run", "chat_id": chat, "ts": "2026-10-01T06:59:00+00:00", "role": "assistant",
                 "content": "Checking the disk and the gateway."}  # the run's own activity: what its card is built from
    (data / "archive" / "progress_20261001T000000.jsonl").write_text(json.dumps(narration) + "\n", encoding="utf-8")

    _isolated_home(tmp_path, monkeypatch)
    evidence = output_dir(request.config)
    with ScriptedStubModel([lambda body: {"final": "No scripted work for this actor."}] * 12,
                           final_answer=CARD_NAME) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        pid = server.proc.pid
        try:
            def window(room):
                return _api(server.base_url, "GET", f"/api/chat/history?chat_id={room}")

            def texts(page_):
                return " ".join(str(row.get("text") or "") for row in page_.get("messages") or [])

            # The boot planned the archive; the maintenance child imports and derives it.
            wait_until(lambda: reply_mark in texts(window(1)), 120)
            settled = wait_until(lambda: (lambda page_: page_ if not {"chat_appearances_deriving", "chat_importing"} & set(
                page_["window"].get("truncated_by") or ()) else None)(window(chat)), 120)
            assert settled, window(chat)["window"]
            assert reply_mark not in texts(settled)
            assert "chat_reply_target_unreadable" in settled["window"]["truncated_by"], settled["window"]
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=theme)
                errors = PageErrors(page, engine)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
                project_feed = f'[id="pchat-{project["id"]}-messages"]'
                with FailureEvidence(page, browser, evidence, request.node.nodeid, engine) as capture:
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    page.locator("#chat-messages .chat-bubble.user").filter(has_text=reply_mark).wait_for(timeout=30_000)
                    _open_project(page, project)
                    card = page.locator(f'{project_feed} .chat-live-card[data-task-id="quiet-run"]')
                    card.wait_for(timeout=30_000)
                    assert "Daily check" in card.inner_text() and sentence in card.inner_text(), card.inner_text()
                    assert reply_mark not in page.locator(project_feed).inner_text()
                    _shot(page, request, "P1-project-before-the-target-is-stored-again")
                    capture.checkpoint("1-before-repair")

                    # ---- the summary's exact row is stored again; the server keeps running ----
                    dialogue_store.append(data, "chat", dict(target))
                    healed = window(chat)
                    assert reply_mark in texts(healed), healed["window"]
                    assert "chat_reply_target_unreadable" not in healed["window"].get("truncated_by", []), healed["window"]
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    _open_project(page, project)
                    page.locator(f"{project_feed} .chat-bubble.user").filter(has_text=reply_mark).wait_for(timeout=30_000)
                    card = page.locator(f'{project_feed} .chat-live-card[data-task-id="quiet-run"]')
                    card.wait_for(timeout=30_000)
                    assert "Daily check" in card.inner_text() and sentence in card.inner_text(), card.inner_text()
                    _shot(page, request, "P2-project-after-the-target-is-stored-again-and-a-reload")
                    capture.checkpoint("2-after-repair")

                    def address(rows, mark):
                        return [row.get("row_address") for row in rows if mark in str(row.get("text") or "")]
                    main_history, project_history = _history(page, 1), _history(page, chat)
                    assert address(main_history, reply_mark) == address(project_history, reply_mark) != []
                    (one,) = address(main_history, reply_mark)
                    assert all(_around(page, room, one)["anchor"]["state"] == "found" for room in (1, chat))
                    _nav(page, "chat")
                    page.locator("#chat-messages .chat-bubble.user").filter(has_text=reply_mark).wait_for(timeout=30_000)
                    assert sentence not in page.locator("#chat-messages").inner_text()  # the run stays in its room
                    assert not [row for row in main_history if row.get("task_id") == "quiet-run"]
                    _open_drawer(page)
                    _shot(page, request, "P3-main-after-the-repair")
                    assert server.proc.pid == pid and server.proc.poll() is None  # the same process, never restarted
                    assert not errors.unexpected(), errors
        finally:
            server.stop()
            assert server.proc.poll() is not None
