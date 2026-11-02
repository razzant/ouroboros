"""A Project root's Main notice, end to end in the real SPA: one message in two rooms.

A turn bound to a Project room may judge that the owner needs a notice in Main
(``send_user_message(destination="main")``). The owner chose (28.10, A/A) that
a Project keeps its whole conversation: the notice is written to Main and also
stands in its Project, one message read once; his direct Reply to it in Main
stands in the Project too, telling the work nothing. What no unit test certifies
is the consumer: the shipped SPA must show the notice once in each room, count it
unread in both, clear it in both when he reads it in Main without opening the
Project, never read it as the turn's answer while the turn still runs, show his
Reply in the Project, and replay the same rows after a reload. On a phone the
Project room is a full-screen sheet over Main, so its surface must be opaque.

The turn is a real keyless server turn. ``ModelGate`` holds it twice at the
HTTP boundary: before its first round, so the owner can leave the chat page
before the notice exists, and inside its second round, after the notice tool
returned and before the final answer, so "still running" is a fact, not a race.
"""

import re
import uuid

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.system_e2e.harness import (
    ArtifactOracle,
    ModelGate,
    ScriptedStubModel,
    body_text,
    keyless_settings,
    message_text,
    start_server,
    wait_durable_result,
    wait_until,
)
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET
from tests.ci_evidence import output_dir
from tests.ui_console_errors import PageErrors
from tests.ui_failure_evidence import FailureEvidence

wait_clone = clone_fixture
pytestmark = [pytest.mark.serial, pytest.mark.ui_browser]

# Tool-less namer calls answer with this, so it never matches a feed assertion.
CARD_NAME = "Project audit turn"

_FEED_FACTS = """({feed, taskId, notice, answer}) => {
    const root = document.querySelector(feed);
    if (!root) return null;
    const bubbles = [...root.querySelectorAll('.chat-bubble')].map(node => ({
        classes: node.className,
        type: node.dataset.systemType || '',
        task: node.dataset.taskId || '',
        sender: node.querySelector(':scope > .sender')?.textContent.trim() || '',
        text: node.querySelector(':scope > .message')?.innerText || '',
    }));
    const cards = [...root.querySelectorAll(`.chat-live-card[data-task-id="${taskId}"]`)];
    const text = root.innerText || '';
    return {
        visible: root.getClientRects().length > 0,
        notice_bubbles: bubbles.filter(b => b.text.includes(notice)),
        answer_bubbles: bubbles.filter(b => b.text.includes(answer)),
        notice_anywhere: text.includes(notice),
        answer_anywhere: text.includes(answer),
        cards: cards.length,
        card_finished: cards[0]?.dataset.finished ?? null,
        card_phase: cards[0]?.querySelector('[data-live-phase]')?.textContent.trim() || '',
    };
}"""

_NAV_FACTS = """projectId => {
    const row = document.querySelector(`.nav-project-row[data-project-id="${projectId}"]`);
    return {
        main_badge: document.querySelector('[data-nav-page="chat"] .unread-badge')?.textContent.trim() || '',
        projects_count: document.querySelector('#nav-projects-count')?.textContent.trim() || '',
        project_label: row?.getAttribute('aria-label') || '',
        active_page: document.querySelector('.page.active')?.id || '',
    };
}"""

_PANEL_COVER = """() => {
    const panel = document.querySelector('#project-panel');
    const notice = document.querySelector('#chat-messages .chat-bubble[data-system-type="main_notice"]');
    return {
        open: panel.classList.contains('open') && !panel.hidden,
        background: getComputedStyle(panel).backgroundColor,
        behind: document.querySelector('.page.active')?.id || '',
        notice_behind: Boolean(notice?.getClientRects().length),
    };
}"""


def _mobile(page):
    return page.viewport_size["width"] <= 640


def _open_drawer(page):
    """At phone width the sidebar is a drawer behind the page header toggle."""
    if not _mobile(page):
        return
    close = page.locator("#project-panel-close")
    if close.is_visible():
        close.click()
    toggle = page.locator(".page.active [data-mobile-nav-toggle]")
    # The handover keeps the drawer painted after its logical close.
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()
    page.wait_for_function("() => !document.querySelector('#primary-sidebar').inert")


def _nav(page, name):
    _open_drawer(page)
    page.locator(f'#primary-sidebar [data-nav-page="{name}"]').click()
    page.wait_for_selector(f"#page-{name}.active")


def _open_project(page, project):
    _open_drawer(page)
    page.locator(f'.nav-project-row[data-project-id="{project["id"]}"]').click()
    feed = f'[id="pchat-{project["id"]}-messages"]'
    page.locator(feed).wait_for(state="visible", timeout=30_000)
    # The drawer handover paints the room before it releases inert. WebKit's fill can
    # otherwise return with an empty field, since the room cannot take focus yet.
    page.wait_for_function("""selector => {
        const room = document.querySelector(selector);
        return room && !room.closest('[inert]');
    }""", arg=feed)
    return feed


def _alpha(color):
    """Alpha of a computed CSS color: rgb()/rgba() in comma or slash syntax, or color(<space> ...)."""
    fn, _, args = color.strip().partition("(")
    parts = re.split(r"[\s,/]+", args.rstrip(")").strip())
    if fn == "color":
        parts = parts[1:]  # the color space name
    assert fn in {"rgb", "rgba", "color"} and len(parts) in (3, 4), color
    alpha = parts[3] if len(parts) == 4 else "1"
    return float(alpha[:-1]) / 100 if alpha.endswith("%") else float(alpha)


def _panel_cover(page):
    """On a phone the open Project room covers Main; a see-through surface would expose the notice."""
    if not _mobile(page):
        return None  # the desktop side sheet is translucent glass by design
    cover = page.evaluate(_PANEL_COVER)
    assert cover["open"] and cover["behind"] == "page-chat" and cover["notice_behind"], cover
    assert _alpha(cover["background"]) == 1, cover
    return cover


def _feed(page, feed, task_id, notice, answer):
    return page.evaluate(_FEED_FACTS, {"feed": feed, "taskId": task_id, "notice": notice, "answer": answer})


def _chat_rows(oracle, needle):
    return [{"direction": row.get("direction"), "chat_id": row.get("chat_id"), "type": row.get("type", ""),
             "task_id": row.get("task_id", ""), "text": str(row.get("text") or "")}
            for row in oracle._jsonl("logs/chat.jsonl") if needle in str(row.get("text") or "")]


def _identities(rows):
    return [(row["direction"], row["chat_id"], row["type"], row["task_id"]) for row in rows]


def _revision(server, project_id):
    rows = _api(server.base_url, "GET", "/api/projects")["projects"]
    return int(next(row for row in rows if row["id"] == project_id).get("visible_revision") or 0)


def _history(page, chat_id):
    return page.evaluate("async id => (await (await fetch(`/api/chat/history?chat_id=${id}`)).json()).messages || []",
                         chat_id)


_PANEL_AT_REST = """() => {
    const rect = JSON.stringify(document.querySelector('#project-panel')?.getBoundingClientRect() || null);
    const same = window.__panelRect === rect;
    window.__panelRect = rect;
    return same;
}"""


def _shot(page, request, name):
    """A screenshot of the state just asserted, kept with the run's evidence (never asserted on), taken once the
    Project panel's motion has come to rest."""
    root = output_dir(request.config)
    if root is not None:
        page.evaluate("() => { window.__panelRect = undefined; }")
        try:
            page.wait_for_function(_PANEL_AT_REST, polling=100, timeout=5_000)
        except Exception:  # evidence only: a panel still moving is photographed as it is
            pass
        path = root / "screens" / f"{request.node.callspec.id}-{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path))


def _prefs(page):
    return page.evaluate("async () => (await (await fetch('/api/ui/preferences', {cache: 'no-store'})).json())")


def _around(page, chat_id, address):
    """The page the history pager reads to land on one exact row (``around=``), as the shipped client asks it."""
    return page.evaluate("async ([id, address]) => (await (await fetch("
                         "`/api/chat/history?chat_id=${id}&around=${encodeURIComponent(address)}`)).json())",
                         [chat_id, address])


def _mailbox(server, task_id):
    """The owner words waiting for one work (``owner_mailbox``): a Reply that told it something would be here."""
    path = server.data_root / "memory" / "owner_mailbox" / f"{task_id}.jsonl"
    return path.read_bytes() if path.exists() else b""


@pytest.mark.parametrize("engine,width,height,theme", [
    ("webkit", 1165, 851, "dark"), ("webkit", 1165, 851, "light"),
    ("chromium", 1440, 900, "dark"), ("webkit", 390, 900, "light"),
], ids=["webkit-desktop-dark", "webkit-desktop-light", "chromium-desktop-dark", "webkit-mobile-light"])
def test_project_root_main_notice_is_one_message_in_main_and_its_project(
    wait_clone, tmp_path, monkeypatch, request, engine, width, height, theme,
):
    from playwright.sync_api import sync_playwright

    from tests.system_e2e.harness import KeylessIsolatedServer

    tag = uuid.uuid4().hex[:12]
    marker = f"PROJECT_AUDIT_{tag}"
    notice = (f"Owner action: the executor account needs its billing re-enabled in Settings; "
              f"I am continuing this Project audit meanwhile. [MAIN_NOTICE_{tag}]")
    answer = f"The Project audit is complete and nothing else needs your action. [PROJECT_ANSWER_{tag}]"
    reply = f"Why only here? Keep the Project informed. [OWNER_REPLY_{tag}]"
    main_answer = f"Noted here in Main. [MAIN_ANSWER_{tag}]"
    notice_mark, answer_mark, reply_mark = f"MAIN_NOTICE_{tag}", f"PROJECT_ANSWER_{tag}", f"OWNER_REPLY_{tag}"
    main_answer_mark = f"MAIN_ANSWER_{tag}"
    tool_results, strangers = [], []

    def turn_round(body, *, after_tool):
        results = [message_text(m) for m in body.get("messages", [])
                   if isinstance(m, dict) and str(m.get("role") or "") == "tool"]
        matched = (bool(body.get("tools")) and marker in body_text(body) and reply_mark not in body_text(body)
                   and bool(results) is after_tool)
        if matched and after_tool:
            # Read at the gate: the held round is the model's view of the tool's receipt.
            tool_results.extend(results)
        return matched

    before_notice = ModelGate(lambda body: turn_round(body, after_tool=False), timeout=300)
    before_answer = ModelGate(lambda body: turn_round(body, after_tool=True), timeout=300)

    def respond(body):
        if reply_mark in body_text(body):
            return {"final": main_answer}  # the owner's Reply is an ordinary Main turn
        if marker not in body_text(body):
            strangers.append(body_text(body)[:300])
            return {"final": "No scripted work for this actor."}
        if not any(str(m.get("role") or "") == "tool" for m in body.get("messages", []) if isinstance(m, dict)):
            # The model's own judgment: one notice for Main.
            return {"tool": "send_user_message", "arguments": {
                "text": notice, "reason": "owner-only remedy outside this Project", "destination": "main"}}
        return {"final": answer}

    home = tmp_path / "home"
    home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(home), "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
    })
    evidence = output_dir(request.config)
    gates = lambda body: (before_notice(body), before_answer(body))  # noqa: E731
    # The stub narrates its tool round ("still working"): that durable progress row is
    # what a re-mounted Project panel rebuilds the running card from.
    with ScriptedStubModel([respond] * 12, final_answer=CARD_NAME, gate=gates) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            project = _api(server.base_url, "POST", "/api/projects", {"name": "Main notice room"})["project"]
            project_chat = int(project["chat_id"])
            assert project_chat != 1, project
            unrelated = int(_api(server.base_url, "POST", "/api/projects", {"name": "Unrelated room"})["project"]["chat_id"])
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": width, "height": height}, has_touch=width < 980,
                                        color_scheme=theme)
                errors = PageErrors(page, engine)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
                main_feed = "#chat-messages"
                with FailureEvidence(page, browser, evidence, request.node.nodeid, engine) as capture:
                    capture.details.update({"theme": theme, "width": width, "height": height})
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")

                    # ---- the owner speaks in the Project room ----
                    project_feed = _open_project(page, project)
                    composer = page.locator(f'[id="pchat-{project["id"]}-input"]')
                    prompt = f"{marker} Audit this Project, and tell me in Main if something needs my action."
                    composer.fill(prompt)
                    assert composer.input_value() == prompt
                    page.locator(f'[id="pchat-{project["id"]}-send"]').click()
                    task = wait_until(lambda: next((row["task"] for row in oracle.events("task_received")
                        if marker in str(row.get("task", {}).get("text") or "")), None), 90)
                    assert task, "the Project composer send never reached a turn"
                    task_id = task["id"]
                    assert int(task.get("chat_id") or 0) == project_chat, task
                    assert before_notice.arrived.wait(120), "the Project turn never reached its first round"

                    # ---- the owner is away from chat before the notice exists ----
                    _nav(page, "dashboard")
                    revision_before = _revision(server, project["id"])
                    away_before = page.evaluate(_NAV_FACTS, project["id"])
                    assert away_before["main_badge"] == "" and away_before["projects_count"] == "", away_before

                    before_notice.release.set()
                    assert before_answer.arrived.wait(120), "the turn never returned from its notice tool"
                    # One durable row, written to Main: the address the Project shows too.
                    main_row = ("out", 1, "main_notice", task_id)
                    wait_until(lambda: main_row in _identities(_chat_rows(oracle, notice_mark)), 60)
                    notice_rows = _chat_rows(oracle, notice_mark)
                    assert _identities(notice_rows) == [main_row], notice_rows
                    assert any("notice sent to the main chat" in text for text in tool_results), tool_results
                    proactive = [row for row in oracle.events("proactive_message") if row.get("task_id") == task_id]
                    assert [row.get("destination") for row in proactive] == ["main"], proactive
                    # Unread where it is shown, never silent before he reads it: Main and the Project.
                    page.wait_for_function(
                        "() => document.querySelector('[data-nav-page=\"chat\"] .unread-badge')?.textContent.trim() === '1'",
                        timeout=30_000)
                    page.wait_for_function(
                        "() => document.querySelector('#nav-projects-count')?.textContent.trim() === '1'", timeout=45_000)
                    assert _revision(server, project["id"]) == revision_before + 1
                    assert str(oracle.task_result(task_id).get("status") or "") == "running"
                    seen_before = (_prefs(page).get("project_seen_revision") or {}).get(project["id"])
                    _open_drawer(page)
                    capture.checkpoint("1-away-after-notice")

                    # ---- he reads it in Main: read once, in both rooms, the Project never opened ----
                    _nav(page, "chat")
                    page.locator(f'{main_feed} .chat-bubble.assistant[data-system-type="main_notice"]').filter(
                        has_text=notice_mark).wait_for(timeout=30_000)
                    main_live = _feed(page, main_feed, task_id, notice_mark, answer_mark)
                    (bubble,) = main_live["notice_bubbles"]
                    assert "assistant" in bubble["classes"].split(), bubble
                    assert bubble["type"] == "main_notice" and bubble["task"] == task_id, bubble
                    assert bubble["sender"] == "Ouroboros", bubble
                    assert main_live["cards"] == 0 and not main_live["answer_anywhere"], main_live
                    page.wait_for_function(
                        "() => (document.querySelector('#nav-projects-count')?.textContent.trim() || '') === ''",
                        timeout=45_000)
                    prefs = _prefs(page)
                    assert (prefs.get("project_seen_revision") or {}).get(project["id"]) == seen_before, prefs
                    assert prefs["read_ahead_revisions"].get(str(project_chat)), prefs
                    assert page.evaluate(_NAV_FACTS, project["id"])["main_badge"] == ""
                    _shot(page, request, "2-main-notice-read-once")
                    capture.checkpoint("2-main-notice-live")

                    # ---- the Project shows the same message; the work still runs ----
                    _open_project(page, project)
                    card = page.locator(f'{project_feed} .chat-live-card[data-task-id="{task_id}"]')
                    card.wait_for(timeout=30_000)
                    page.locator(f'{project_feed} .chat-bubble[data-system-type="main_notice"]').filter(
                        has_text=notice_mark).wait_for(timeout=30_000)
                    project_held = _feed(page, project_feed, task_id, notice_mark, answer_mark)
                    assert project_held["card_finished"] == "0", project_held
                    assert project_held["cards"] == 1 and len(project_held["notice_bubbles"]) == 1, project_held
                    assert not project_held["answer_anywhere"], project_held
                    capture.checkpoint("project_held:panel_cover")
                    _panel_cover(page)
                    _shot(page, request, "3-project-shows-notice-while-running")
                    capture.checkpoint("3-project-held-running")

                    # ---- the answer lands in the Project while the owner is away ----
                    _nav(page, "dashboard")
                    before_answer.release.set()
                    stored = wait_durable_result(oracle, task_id, timeout=120)
                    assert stored["status"] == "completed", stored
                    assert answer_mark in str(stored.get("result") or "")
                    assert notice_mark not in str(stored.get("result") or ""), stored
                    answer_rows = wait_until(lambda: _chat_rows(oracle, answer_mark), 60) or []
                    assert answer_rows and {row["chat_id"] for row in answer_rows} == {project_chat}, answer_rows
                    assert _chat_rows(oracle, notice_mark) == notice_rows, "the final re-delivered the notice"
                    # Only the answer is unread in the Project now: the notice was read in Main.
                    page.wait_for_function(
                        "() => document.querySelector('#nav-projects-count')?.textContent.trim() === '1'",
                        timeout=45_000)
                    assert page.evaluate(_NAV_FACTS, project["id"])["main_badge"] == ""
                    _open_drawer(page)
                    capture.checkpoint("4-away-after-answer")

                    # ---- he Replies to the notice in Main ----
                    held_words, settled = _mailbox(server, task_id), oracle.task_result(task_id)
                    received = len(oracle.events("task_received"))
                    _nav(page, "chat")
                    notice_bubble = page.locator(f'{main_feed} .chat-bubble[data-system-type="main_notice"]').filter(
                        has_text=notice_mark)
                    notice_bubble.hover()
                    notice_bubble.locator(".chat-message-reply").click()
                    page.locator("#chat-input").fill(reply)
                    page.locator("#chat-send").click()
                    page.locator(f"{main_feed} .chat-bubble.user").filter(has_text=reply_mark).wait_for(timeout=30_000)
                    page.locator(main_feed).get_by_text(main_answer_mark, exact=False).first.wait_for(timeout=90_000)
                    reply_rows = _chat_rows(oracle, reply_mark)
                    assert [(row["direction"], row["chat_id"]) for row in reply_rows] == [("in", 1)], reply_rows
                    assert str(oracle.task_result(task_id).get("status") or "") == "completed"  # told, woken: nothing
                    # Preservation only: Main answers its own turn; nothing starts in the Project, the work is told
                    # nothing and its settled result stays as it was.
                    started = [row["task"] for row in oracle.events("task_received")[received:]]
                    assert {int(row.get("chat_id") or 0) for row in started} <= {1}, started
                    assert task_id not in {row.get("id") for row in started}, started
                    assert _mailbox(server, task_id) == held_words
                    assert oracle.task_result(task_id).get("result") == settled.get("result")
                    _shot(page, request, "4-main-reply-to-notice")

                    # ---- the Project shows his Reply, not Main's answer to it ----
                    _open_project(page, project)
                    page.locator(project_feed).get_by_text(answer_mark, exact=False).first.wait_for(timeout=30_000)
                    page.locator(f"{project_feed} .chat-bubble.user").filter(has_text=reply_mark).wait_for(timeout=30_000)
                    project_final = _feed(page, project_feed, task_id, notice_mark, answer_mark)
                    assert len(project_final["notice_bubbles"]) == 1 and project_final["answer_anywhere"], project_final
                    assert main_answer_mark not in page.locator(project_feed).inner_text()
                    # His Reply's quote lands on the notice in the Project itself, the one row both rooms show.
                    page.locator(f"{project_feed} .chat-bubble.user").filter(has_text=reply_mark).locator(
                        ".chat-quote-main").click()
                    page.wait_for_function("""([feed, mark]) => {
                        const node = document.activeElement;
                        return Boolean(node?.matches?.('.chat-bubble[data-system-type="main_notice"]')
                            && node.closest(feed) && node.innerText.includes(mark));
                    }""", arg=[project_feed, notice_mark], timeout=15_000)
                    page.wait_for_function(
                        "() => (document.querySelector('#nav-projects-count')?.textContent.trim() || '') === ''",
                        timeout=45_000)
                    _shot(page, request, "5-project-whole-conversation")
                    capture.checkpoint("5-project-final")

                    # ---- reload: one row per message, the same addresses in both rooms ----
                    page.reload(wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    page.locator(f'{main_feed} .chat-bubble.assistant[data-system-type="main_notice"]').filter(
                        has_text=notice_mark).wait_for(timeout=30_000)
                    main_reloaded = _feed(page, main_feed, task_id, notice_mark, answer_mark)
                    assert len(main_reloaded["notice_bubbles"]) == 1 and not main_reloaded["answer_anywhere"], main_reloaded
                    main_history = _history(page, 1)
                    project_history = _history(page, project_chat)

                    def address(rows, mark):
                        return [row.get("row_address") for row in rows if mark in str(row.get("text") or "")]
                    assert address(main_history, notice_mark) == address(project_history, notice_mark) != []
                    assert address(main_history, reply_mark) == address(project_history, reply_mark) != []
                    assert address(project_history, main_answer_mark) == [] != address(main_history, main_answer_mark)
                    (notice_address,) = address(main_history, notice_mark)
                    for chat in (1, project_chat):
                        landed = _around(page, chat, notice_address)
                        assert landed["anchor"]["state"] == "found", (chat, landed.get("anchor"))
                        assert address(landed["messages"], notice_mark) == [notice_address], chat
                    assert _around(page, unrelated, notice_address)["anchor"]["state"] == "moved"
                    nav_reloaded = page.evaluate(_NAV_FACTS, project["id"])
                    assert nav_reloaded["main_badge"] == "" and nav_reloaded["projects_count"] == "", nav_reloaded
                    capture.checkpoint("6-reloaded-main")
                    _open_project(page, project)
                    page.locator(project_feed).get_by_text(answer_mark, exact=False).first.wait_for(timeout=30_000)
                    page.wait_for_selector(
                        f'{project_feed} .chat-live-card[data-task-id="{task_id}"][data-finished="1"]', timeout=30_000)
                    project_reloaded = _feed(page, project_feed, task_id, notice_mark, answer_mark)
                    assert len(project_reloaded["notice_bubbles"]) == 1, project_reloaded
                    capture.checkpoint("project_reloaded:panel_cover")
                    _panel_cover(page)
                    _shot(page, request, "6-project-after-reload")
                    capture.checkpoint("7-reloaded-project")
                    assert not strangers, strangers
                    assert not errors.unexpected(), errors
        finally:
            before_notice.release.set()
            before_answer.release.set()
            server.stop()
            assert server.proc.poll() is not None
            assert not before_notice.timed_out and not before_answer.timed_out


_PART_FACTS = """({feed, mark}) => {
    const root = document.querySelector(feed);
    const node = [...root.querySelectorAll('.chat-bubble[data-system-type="main_notice"]')]
        .find((bubble) => bubble.innerText.includes(mark));
    if (!node) return null;
    const box = node.getBoundingClientRect(), band = root.getBoundingClientRect();
    const pre = node.querySelector('.message pre');
    // What the reader sees of the feed: below the page header drawn over it, above the composer.
    const header = root.closest('.page')?.querySelector('.chat-page-header');
    const composer = document.querySelector('#chat-input-area');
    const seen = [header?.getClientRects().length ? header.getBoundingClientRect().bottom - band.top : 0,
                  composer?.getClientRects().length ? composer.getBoundingClientRect().top - band.top : root.clientHeight];
    return {top: box.top - band.top + root.scrollTop, height: box.height, viewport: root.clientHeight, seen,
            scrollTop: root.scrollTop, clipped: pre ? pre.scrollWidth > pre.clientWidth : null,
            revisions: JSON.parse(node.dataset.roomRevisions || '{}')};
}"""


def _scroll_to(page, feed, top):
    page.evaluate("([feed, top]) => { document.querySelector(feed).scrollTop = top; }", [feed, top])
    page.evaluate("() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))")


@pytest.mark.parametrize("engine,width,height,theme", [("webkit", 1165, 851, "dark"), ("chromium", 1440, 900, "light")],
                         ids=["webkit-desktop-dark", "chromium-desktop-light"])
def test_a_long_or_clipped_shared_notice_is_read_only_when_every_part_was_shown(
    wait_clone, tmp_path, monkeypatch, request, engine, width, height, theme,
):
    """A Main notice taller than the viewport is read in the Project too only once every part of it was on
    screen in Main: its bottom alone records nothing, reading it by parts does; a notice whose code line
    is wider than its box (a clipped part no position shows) is never recorded. Cursors never move."""
    from playwright.sync_api import sync_playwright

    from tests.system_e2e.harness import KeylessIsolatedServer

    tag = uuid.uuid4().hex[:12]
    marker = f"PROJECT_LONG_{tag}"
    long_mark, clipped_mark, answer_mark = f"LONG_NOTICE_{tag}", f"CLIPPED_NOTICE_{tag}", f"LONG_ANSWER_{tag}"
    long_notice = "\n\n".join([f"Owner action needed in Settings. [{long_mark}]"] + [
        f"Paragraph {index:02d}: the Titan series ended and the experiment branch needs your review before "
        f"anything else is started; this paragraph exists so the notice is taller than the window." for index in range(46)])
    clipped_notice = f"Run this once in a terminal. [{clipped_mark}]\n\n```\n{'titan --resume ' * 90}\n```"
    started = ModelGate(lambda body: bool(body.get("tools")) and marker in body_text(body) and not any(
        str(m.get("role") or "") == "tool" for m in body.get("messages", []) if isinstance(m, dict)), timeout=300)

    def respond(body):
        if marker not in body_text(body):
            return {"final": "No scripted work for this actor."}
        done = sum(str(m.get("role") or "") == "tool" for m in body.get("messages", []) if isinstance(m, dict))
        if done < 2:
            return {"tool": "send_user_message", "arguments": {
                "text": (long_notice, clipped_notice)[done], "reason": "owner-only remedy", "destination": "main"}}
        return {"final": f"Audit complete. [{answer_mark}]"}

    home = tmp_path / "home"
    home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(home), "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config")})
    evidence = output_dir(request.config)
    with ScriptedStubModel([respond] * 12, final_answer=CARD_NAME, gate=started) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            project = _api(server.base_url, "POST", "/api/projects", {"name": "Long notice room"})["project"]
            project_chat = str(project["chat_id"])
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=theme)
                errors = PageErrors(page, engine)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
                main_feed = "#chat-messages"
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
                    # Two notices and the answer: unread in the Project, the notices in Main too.
                    page.wait_for_function(
                        "() => document.querySelector('#nav-projects-count')?.textContent.trim() === '3'", timeout=45_000)
                    capture.checkpoint("1-away-three-unread")

                    _nav(page, "chat")
                    page.locator(f'{main_feed} .chat-bubble[data-system-type="main_notice"]').filter(
                        has_text=clipped_mark).wait_for(timeout=30_000)
                    long_facts = page.evaluate(_PART_FACTS, {"feed": main_feed, "mark": long_mark})
                    clipped = page.evaluate(_PART_FACTS, {"feed": main_feed, "mark": clipped_mark})
                    assert long_facts["height"] > 2 * long_facts["viewport"], long_facts  # really taller than the window
                    assert clipped["clipped"] is True, clipped  # the shipped code block clips its long line
                    # Only the long notice's bottom part on screen (the clipped notice whole, below it).
                    _scroll_to(page, main_feed, long_facts["top"] + long_facts["height"] - long_facts["viewport"] // 2)
                    page.wait_for_timeout(1500)
                    prefs = _prefs(page)
                    assert not prefs["read_ahead_revisions"].get(project_chat), prefs  # nothing recorded yet
                    assert page.evaluate(_NAV_FACTS, project["id"])["projects_count"] == "3"
                    _shot(page, request, "L1-long-bottom-only-clipped-whole-box")
                    capture.checkpoint("2-bottom-only")

                    # Reading it by parts, top to bottom, records exactly the long notice: its first line just
                    # below the header drawn over the feed, then half of what the reader sees per step.
                    above, below = long_facts["seen"]
                    assert 0 < above < below <= long_facts["viewport"], long_facts
                    top = long_facts["top"] - above
                    while top < long_facts["top"] + long_facts["height"]:
                        _scroll_to(page, main_feed, top)
                        top += (below - above) // 2
                    page.wait_for_function(
                        "() => document.querySelector('#nav-projects-count')?.textContent.trim() === '2'", timeout=30_000)
                    prefs = _prefs(page)
                    held = prefs["read_ahead_revisions"].get(project_chat) or []
                    revision = long_facts["revisions"][project_chat]
                    assert held == [[revision, revision]], (held, long_facts, clipped)
                    assert clipped["revisions"][project_chat] not in {value for span in held for value in range(span[0], span[1] + 1)}
                    assert (prefs.get("project_seen_revision") or {}).get(project["id"], 0) < revision  # no cursor moved
                    _shot(page, request, "L2-long-read-by-parts")
                    _scroll_to(page, main_feed, clipped["top"] - 40)
                    page.wait_for_timeout(1500)
                    assert page.evaluate(_NAV_FACTS, project["id"])["projects_count"] == "2"  # the clipped one stays
                    _shot(page, request, "L3-clipped-never-read")
                    capture.checkpoint("3-read-by-parts")
                    assert not errors.unexpected(), errors
        finally:
            started.release.set()
            server.stop()
            assert server.proc.poll() is not None


_LANDING_FACTS = """({feed, area, first, shared}) => {
    const root = document.querySelector(feed);
    const find = (mark) => root && [...root.querySelectorAll('.chat-bubble')].find((node) => node.innerText.includes(mark));
    const one = find(first), two = find(shared);
    if (!one || !two) return null;
    // The band a reader sees of the room: below the panel's bar and the page header, above its composer.
    const box = root.getBoundingClientRect();
    const bars = [root.closest('.project-panel')?.querySelector('.project-panel-bar'),
                  root.closest('.page, .chat-instance-panel')?.querySelector('.chat-page-header')]
        .filter((bar) => bar?.getClientRects().length).map((bar) => bar.getBoundingClientRect().bottom);
    const composer = document.querySelector(area);
    const top = Math.max(box.top, ...bars);
    const bottom = composer?.getClientRects().length ? Math.min(box.bottom, composer.getBoundingClientRect().top) : box.bottom;
    const a = one.getBoundingClientRect(), b = two.getBoundingClientRect();
    return {first_top: a.top - top, first_bottom: a.bottom - top, shared_top: b.top - top, band: bottom - top,
            scroll_top: root.scrollTop, bottom_gap: root.scrollHeight - root.clientHeight - root.scrollTop};
}"""


def _scroll_top(page, feed):
    return page.evaluate("feed => document.querySelector(feed).scrollTop", feed)


@pytest.mark.parametrize("engine,width,height,theme", [
    ("webkit", 1165, 851, "dark"), ("chromium", 1440, 900, "light"),
], ids=["webkit-desktop-dark", "chromium-desktop-light"])
def test_a_shared_notice_read_in_main_leaves_its_projects_own_unread_landing_and_scroll(
    wait_clone, tmp_path, monkeypatch, request, engine, width, height, theme,
):
    """Reading in Main the one message both rooms show reads it in its Project too, and nothing else: the
    Project's own earlier message stays unread, opening the Project lands on that message (never again on the
    notice read in Main), and neither room's reading moves the other room's scroll position (owner A, 28.10)."""
    from playwright.sync_api import sync_playwright

    from tests.system_e2e.harness import KeylessIsolatedServer

    tag = uuid.uuid4().hex[:12]
    marker = f"PROJECT_LAND_{tag}"
    first_mark, shared_mark = f"PROJECT_ONLY_{tag}", f"SHARED_NOTICE_{tag}"
    first = f"First finding, for this Project only. [{first_mark}]"
    shared = "\n\n".join([f"Owner action needed in Settings. [{shared_mark}]"] + [
        f"Paragraph {index:02d}: the series ended and the branch needs your review before anything else is "
        f"started; this paragraph exists so the notice is taller than the window." for index in range(40)])

    def tools(body):
        return sum(str(m.get("role") or "") == "tool" for m in body.get("messages", []) if isinstance(m, dict))

    started = ModelGate(lambda body: bool(body.get("tools")) and marker in body_text(body) and tools(body) == 0, timeout=300)
    held = ModelGate(lambda body: bool(body.get("tools")) and marker in body_text(body) and tools(body) == 2, timeout=300)

    def respond(body):
        if marker not in body_text(body):
            return {"final": "No scripted work for this actor."}
        if tools(body) == 0:  # its own room first, then one notice for Main
            return {"tool": "send_user_message", "arguments": {"text": first, "reason": "a finding for this Project"}}
        if tools(body) == 1:
            return {"tool": "send_user_message", "arguments": {
                "text": shared, "reason": "owner-only remedy", "destination": "main"}}
        return {"final": f"Audit complete. [LANDING_ANSWER_{tag}]"}

    home = tmp_path / "home"
    home.mkdir()
    original_env = KeylessIsolatedServer._env
    monkeypatch.setattr(KeylessIsolatedServer, "_env", lambda server: {
        **original_env(server), "HOME": str(home), "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home / ".config")})
    evidence = output_dir(request.config)
    gates = lambda body: (started(body), held(body))  # noqa: E731
    with ScriptedStubModel([respond] * 12, final_answer=CARD_NAME, gate=gates) as stub:
        server = start_server(wait_clone, tmp_path / "instance", keyless_settings(stub, OUROBOROS_MAX_WORKERS=1))
        oracle = ArtifactOracle(server.data_root)
        try:
            project = _api(server.base_url, "POST", "/api/projects", {"name": "Landing room"})["project"]
            project_chat = str(project["chat_id"])
            with sync_playwright() as pw:
                browser = getattr(pw, engine).launch()
                page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=theme)
                errors = PageErrors(page, engine)
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
                main_feed = "#chat-messages"
                with FailureEvidence(page, browser, evidence, request.node.nodeid, engine) as capture:
                    capture.details.update({"theme": theme, "width": width, "height": height})
                    page.goto(server.base_url, wait_until="domcontentloaded")
                    page.wait_for_function("() => window.__testSockets?.[0]?.readyState === WebSocket.OPEN")
                    project_feed = _open_project(page, project)
                    project_area = f'[id="pchat-{project["id"]}-input-area"]'
                    page.locator(f'[id="pchat-{project["id"]}-input"]').fill(f"{marker} Audit and tell me in Main.")
                    page.locator(f'[id="pchat-{project["id"]}-send"]').click()
                    assert started.arrived.wait(120), "the Project turn never reached its first round"
                    _nav(page, "dashboard")  # away before anything is said
                    started.release.set()
                    assert held.arrived.wait(120), "the turn never returned from its two messages"
                    # One durable row each: its own message in its room, the notice written to Main.
                    own = wait_until(lambda: _chat_rows(oracle, first_mark), 60) or []
                    assert [(row["direction"], row["chat_id"]) for row in own] == [("out", int(project_chat))], own
                    notice_rows = wait_until(lambda: _chat_rows(oracle, shared_mark), 60) or []
                    assert [(row["direction"], row["chat_id"], row["type"]) for row in notice_rows] == [
                        ("out", 1, "main_notice")], notice_rows
                    # Its own message and the notice are unread in the Project; the notice is unread in Main.
                    page.wait_for_function(
                        "() => document.querySelector('#nav-projects-count')?.textContent.trim() === '2'", timeout=45_000)
                    page.wait_for_function(
                        "() => document.querySelector('[data-nav-page=\"chat\"] .unread-badge')?.textContent.trim() === '1'",
                        timeout=30_000)
                    seen_before = (_prefs(page).get("project_seen_revision") or {}).get(project["id"], 0)
                    capture.checkpoint("1-away-two-unread")

                    # ---- he reads the notice in full in Main, by parts: it is read in the Project too ----
                    _nav(page, "chat")
                    page.locator(f'{main_feed} .chat-bubble[data-system-type="main_notice"]').filter(
                        has_text=shared_mark).wait_for(timeout=30_000)
                    facts = page.evaluate(_PART_FACTS, {"feed": main_feed, "mark": shared_mark})
                    assert facts["height"] > facts["viewport"], facts  # taller than the window: read by parts
                    above, below = facts["seen"]
                    top = facts["top"] - above
                    while top < facts["top"] + facts["height"]:
                        _scroll_to(page, main_feed, top)
                        top += (below - above) // 2
                    # The Project's own message stays unread there: one read, of one message.
                    page.wait_for_function(
                        "() => document.querySelector('#nav-projects-count')?.textContent.trim() === '1'", timeout=30_000)
                    page.wait_for_function(
                        "() => !document.querySelector('[data-nav-page=\"chat\"] .unread-badge')?.textContent.trim()",
                        timeout=30_000)
                    prefs = _prefs(page)
                    revision = facts["revisions"][project_chat]
                    assert prefs["read_ahead_revisions"].get(project_chat) == [[revision, revision]], prefs
                    assert (prefs.get("project_seen_revision") or {}).get(project["id"], 0) == seen_before < revision
                    main_scroll = _scroll_top(page, main_feed)
                    _shot(page, request, "R1-main-read-the-shared-notice")
                    capture.checkpoint("2-read-in-main")

                    # ---- the Project lands on its own unread message, never again on the notice ----
                    _open_project(page, project)
                    landed = page.wait_for_function("""(args) => {
                        const facts = (%s)(args);
                        return facts && Math.abs(facts.first_top) <= 4 && facts.shared_top > facts.first_bottom - 1
                            && facts.bottom_gap > 40 ? facts : null;
                    }""" % _LANDING_FACTS, arg={"feed": project_feed, "area": project_area, "first": first_mark,
                                                "shared": shared_mark}, timeout=30_000).json_value()
                    assert _scroll_top(page, main_feed) == main_scroll  # opening and landing the Project left Main still
                    project_scroll = landed["scroll_top"]
                    _shot(page, request, "R2-project-landed-on-its-own-unread")
                    capture.checkpoint("3-project-landed")

                    # ---- each room keeps its own position while the other is read ----
                    _scroll_to(page, main_feed, 0)
                    page.wait_for_timeout(800)
                    assert _scroll_top(page, project_feed) == project_scroll, "reading Main moved the Project"
                    main_scroll = _scroll_top(page, main_feed)
                    _scroll_to(page, project_feed, page.evaluate("feed => document.querySelector(feed).scrollHeight",
                                                                  project_feed))
                    page.wait_for_function(
                        "() => (document.querySelector('#nav-projects-count')?.textContent.trim() || '') === ''",
                        timeout=30_000)
                    assert _scroll_top(page, main_feed) == main_scroll, "reading the Project moved Main"
                    assert not (_prefs(page).get("read_ahead_revisions") or {}).get("1")  # Main read the notice itself
                    _shot(page, request, "R3-each-room-kept-its-own-position")
                    capture.checkpoint("4-positions-independent")
                    assert not errors.unexpected(), errors
        finally:
            started.release.set()
            held.release.set()
            server.stop()
            assert server.proc.poll() is not None
