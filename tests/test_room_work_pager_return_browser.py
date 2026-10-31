"""View question and Back to reading through the PRODUCTION history pager (DESIGN "Room work status",
"History edges").

The real server, history endpoint, census, owner wait and SPA of ``direct_server_with_data``; only
model judgment is scripted (``test_room_work_backend_browser.ScriptedModel``). A room holds several
real history pages (150 conversation rows each); every page is read by the room's own
``Load more history`` and every release is the pager's own (``chat_history.createChatHistoryPager``
keeps three pages and lets the farthest go; ``↓`` re-anchors on the newest page and lets the old
chain go). Nothing here removes a node or filters a response.

The room's newest window stays mounted while the reader walks older (``chat.js``
``retainedHistoryIds``), so the waited question is made older than it: the turn waits, then 300
newer saved lines follow it in the room's log, and the question lives in an older page.

``test_view_question_and_back_through_real_history_pages`` (Main and a Project, Chromium and WebKit),
by keyboard, and View question never posts a decision:
  loaded      the question's older page and a newer one are loaded, the reader in the newer one:
              Back returns exactly to the passage left.
  released    reading on releases the question's page: View question reads the task detail and
              shows the question; the passage left is still loaded, and Back returns exactly.
  origin gone after View question, ``↓`` re-anchors on the present and the pager releases the
              passage left: Back says ``Earlier place is no longer loaded`` and reads nothing; the
              current Back loads no lost place (no bookmark, pager or reload of its own).
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess

import pytest

from ouroboros import dialogue_store
from tests import test_ui_smoke_playwright as smoke
from tests.test_chat_history_paging_browser import _OBSERVE_HISTORY, _idle, _step
from tests.test_room_work_backend_browser import (
    ASK,
    Evidence,
    ScriptedModel,
    _decision_posts,
    _facts,
    _http,
    _launch,
    _open_chat,
    _send_from_composer,
    _wait,
    _wait_facts,
    _waiting_turn,
)

direct_server_with_data = smoke.direct_server_with_data

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

FILLER = 700  # older saved lines; pages hold 150 conversation rows (gateway/history.py _DEFAULT_N_HUMAN)
LATER = 300   # newer saved lines written after the question, so it lives two pages back
_FRAMES = "() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))"

# The reader's place as the feed's anchor owner sees it: the first visible top-level node.
_PLACE = """feed => {
    const root = document.querySelector(feed), box = root.getBoundingClientRect();
    const node = [...root.children].find(n => !n.classList.contains('typing-bubble') && !n.classList.contains('chat-load-older')
        && n.getBoundingClientRect().bottom > box.top && n.getBoundingClientRect().top < box.bottom);
    return {scrollTop: root.scrollTop, history: node?.dataset.historyId || '', text: (node?.textContent || '').slice(0, 40),
            offset: node ? Math.round(node.getBoundingClientRect().top - box.top) : null,
            mounted: root.querySelectorAll('[data-history-id]').length};
}"""
# Put one row the reader chose at a reading position below the floating header.
_READ_AT = """([feed, id]) => {
    const root = document.querySelector(feed), node = root.querySelector(`[data-history-id="${CSS.escape(id)}"]`);
    if (!node) return false;
    root.scrollTop += node.getBoundingClientRect().top - root.getBoundingClientRect().top - 140;
    root.dispatchEvent(new Event('scroll'));
    return true;
}"""
_ACTIVE = """() => { const a = document.activeElement;
    return !a || a === document.body ? 'body' : `${a.tagName}#${a.id}.${String(a.className || '')}`; }"""
_LIVE = """root => document.querySelector(`${root} .rw-sr[role=status]`)?.textContent || ''"""


def _seed_filler(data_dir, chat_id, tag, count, later=False):
    """Saved exchanges in the room's own log: older than the run, or (``later``) just after now."""
    now = _dt.datetime.now(_dt.timezone.utc)
    rows = []
    for index in range(count):
        ts = (now + _dt.timedelta(milliseconds=index + 1) if later
              else now - _dt.timedelta(minutes=2 * count - index)).isoformat()
        name = f"{tag}{'L' if later else ''}-{index:04d}"
        row = {"ts": ts, "chat_id": chat_id, "direction": "in" if index % 2 == 0 else "out",
               "text": f"PG-{name} saved reading line."}
        if index % 2 == 0:
            row["client_message_id"] = f"pg-{name}"
        rows.append(row)
    for row in rows:
        dialogue_store.append(data_dir, "chat", row)


def _reads(page, chat_id):
    return page.evaluate("id => window.__historyReads.filter(read => read.chatId === id)", chat_id)


def _ids(read):
    return [row["history_id"] for row in (read.get("body") or {}).get("messages") or [] if row.get("history_id")]


def _filler_id(read, tag):
    """A saved line from the middle of one real page: far from either edge of the reading window."""
    rows = [row for row in read["body"]["messages"] if str(row.get("text", "")).startswith(f"PG-{tag}")]
    assert len(rows) > 40, ("the page carries the room's saved lines", len(rows))
    return rows[len(rows) // 2]["history_id"]


def _mounted(page, feed, ids):
    return page.evaluate("([feed, ids]) => { const root = document.querySelector(feed);"
                         " return ids.filter(id => root.querySelector(`[data-history-id=\"${CSS.escape(id)}\"]`)).length; }",
                         [feed, list(ids)])


def _read_at(page, feed, history_id):
    assert page.evaluate(_READ_AT, [feed, history_id]), ("the chosen passage is mounted", history_id)
    page.evaluate(_FRAMES)
    page.wait_for_timeout(350)
    return page.evaluate(_PLACE, feed)


_ON_PILL = """id => document.activeElement?.closest('.rw-row')?.dataset.id === id
    && document.activeElement.classList.contains('rw-pill-question')"""


def _view_question_by_keyboard(page, room, task_id, engine):
    """The status control by keyboard (ArrowDown opens into the list), then the row's View question
    by Enter. Chromium reaches the button by Tab. macOS WebKit's Tab visits only text fields: it would
    leave the list for the question's answer field and scroll the feed there, so on WebKit the button
    is focused directly and still pressed by its key."""
    page.locator(f"{room['root']} .rw-trigger").focus()
    page.keyboard.press("ArrowDown")
    _wait_facts(page, room["scope"], lambda f: (f["active"] or {}).get("inSurface"), "the keyboard in the list")
    if engine == "chromium":
        for _ in range(4):
            if page.evaluate(_ON_PILL, task_id):
                break
            page.keyboard.press("Tab")
    else:
        page.locator(f"{room['root']} .rw-row[data-id='{task_id}'] .rw-pill-question").focus()
    assert page.evaluate(_ON_PILL, task_id), page.evaluate(_ACTIVE)
    page.keyboard.press("Enter")


def _landed(page, room, card):
    card.wait_for(state="visible", timeout=30_000)
    page.wait_for_function("() => document.activeElement?.classList.contains('chat-quiz-question')", timeout=30_000)
    facts = card.evaluate("""card => { const feed = card.closest('#chat-messages, .chat-messages');
        const box = feed.getBoundingClientRect(), r = card.getBoundingClientRect();
        const top = [...feed.children].find(n => n.contains(card));
        return { focus: document.activeElement.className, onScreen: r.bottom > box.top && r.top < box.bottom,
                 comment: card.querySelector('.chat-quiz-comment')?.value ?? null,
                 enabled: [...card.querySelectorAll('.chat-quiz-option')].filter(b => !b.disabled).length,
                 position: [...feed.children].indexOf(top), of: feed.children.length,
                 history: card.closest('[data-history-id]')?.dataset.historyId || '' }; }""")
    assert facts["onScreen"] and facts["comment"] in ("", None) and facts["enabled"] == 2, facts
    assert _facts(page, room["scope"])["back"] == "Back to reading"
    return facts


def _back_by_keyboard(page, room):
    page.locator(f"{room['root']} .rw-back").focus()
    page.keyboard.press("Enter")
    page.evaluate(_FRAMES)
    page.wait_for_timeout(400)
    return {"place": page.evaluate(_PLACE, room["feed"]), "back": _facts(page, room["scope"])["back"],
            "live": page.evaluate(_LIVE, room["root"]), "focus": page.evaluate(_ACTIVE)}


def _assert_exact_return(before, after, label):
    assert after["live"] == "Returned to where you were reading.", (label, after)
    assert after["back"] == "", (label, "Back leaves once it has returned", after)
    assert after["place"]["history"] == before["history"] and abs(after["place"]["offset"] - before["offset"]) <= 2, \
        (label, before, after)
    assert abs(after["place"]["scrollTop"] - before["scrollTop"]) <= 2, (label, before, after)
    assert after["focus"] != "body", (label, "Back by keyboard dropped focus to the document", after)


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("scope", ["main", "project"])
def test_view_question_and_back_through_real_history_pages(direct_server_with_data, monkeypatch, tmp_path, scope, engine):
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    evidence = Evidence(data_dir, f"pager-{scope}-{engine}")
    model = ScriptedModel(monkeypatch, ask=[ASK])
    project = None
    if scope == "project":
        folder = tmp_path / "pager-room"
        folder.mkdir()
        subprocess.run(["git", "init", "-q", str(folder)], check=True, capture_output=True)
        project = _http(url, "POST", "/api/projects", {"name": "Pager reading room", "path": str(folder)})["project"]
    chat_id = int(project["chat_id"]) if project else 1
    room = ({"scope": "main", "root": "#page-chat", "feed": "#chat-messages"} if not project else
            {"scope": "project", "root": "#project-panel", "feed": f"#pchat-{project['id']}-messages"})
    tag = "P" if project else "M"
    server["stop_server"]()
    _seed_filler(data_dir, chat_id, tag, FILLER)
    server["start_server"]()

    def open_room(page):
        _open_chat(page, url)
        if project:
            page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))", project)
            page.wait_for_selector(f"{room['root']} .project-panel-bar .rw-trigger", state="attached", timeout=30_000)
        _idle(page, room["feed"])

    try:
        with sync_playwright() as pw:
            browser, page, log = _launch(pw, engine)
            page.add_init_script(f"({_OBSERVE_HISTORY})()")
            try:
                open_room(page)
                _send_from_composer(page, f"{ASK}: check which evidence the reading notes should cite, and ask me first.",
                                    room["scope"], project["id"] if project else "")
                turn = _wait(lambda: _waiting_turn(url, chat_id), 90, "the room's turn waiting on its own question")
                task_id, quiz_id = turn["activity_id"], turn["owner_wait"]["quiz_id"]
                evidence.record("turn", {"task_id": task_id, "quiz_id": quiz_id, "chat_id": chat_id})
                card = page.locator(f'{room["feed"]} .chat-quiz-card[data-task-id="{task_id}"][data-quiz-id="{quiz_id}"]')
                card.wait_for(state="attached", timeout=30_000)
                # The turn waits for its answer while the room's log goes on: the question is now older
                # than the newest window, which stays mounted while the reader walks older.
                _seed_filler(data_dir, chat_id, tag, LATER, later=True)

                def holds_question(read):
                    return quiz_id in json.dumps((read.get("body") or {}).get("messages") or [])

                # A fresh session: every row on screen comes from the room's real history pages.
                open_room(page)
                reads = _reads(page, chat_id)
                newest = reads[-1]
                assert not newest["cursor"] and newest["body"]["has_more"], ("one newest page, older ones exist", newest["cursor"])
                assert not holds_question(newest) and card.count() == 0, "the question is older than the newest page"
                newest_ids = _ids(newest)
                base = len(reads)

                # ---- loaded: read older until the question's page lands; the reader in the newer page before it.
                for _ in range(4):
                    _step(page, room["feed"])
                    if holds_question(_reads(page, chat_id)[-1]):
                        break
                walk = _reads(page, chat_id)[base:]
                assert all(read["cursor"] for read in walk) and holds_question(walk[-1]), [read["cursor"] for read in walk]
                assert len(walk) >= 2, ("the question is at least two pages back", len(walk))
                question_ids = _ids(walk[-1])
                card.wait_for(state="attached", timeout=30_000)
                evidence.record("question_page", {"older_reads_to_it": len(walk), "rows": len(question_ids)})
                origin = _read_at(page, room["feed"], _filler_id(walk[-2], tag))
                assert _mounted(page, room["feed"], newest_ids) == len(newest_ids), "every page read so far is loaded"
                evidence.record("loaded_origin", origin)
                _view_question_by_keyboard(page, room, task_id, engine)
                evidence.record("loaded_landed", _landed(page, room, card))
                evidence.shot(page, "01-loaded-question")
                reads_before = len(_reads(page, chat_id))
                after = _back_by_keyboard(page, room)
                evidence.record("loaded_back", after)
                _assert_exact_return(origin, after, "loaded")
                assert len(_reads(page, chat_id)) == reads_before, "Back reads no history"
                evidence.shot(page, "02-loaded-back")

                # ---- released: reading on lets the question's page go (the pager keeps three pages).
                for _ in range(6):
                    if card.count() == 0 and _mounted(page, room["feed"], question_ids) == 0:
                        break
                    _step(page, room["feed"])
                pages = _reads(page, chat_id)[base:]
                evidence.record("walk", {"older_reads": len(pages), "rows": [len(_ids(read)) for read in pages],
                                         "question_page_mounted": _mounted(page, room["feed"], question_ids),
                                         "newest_page_mounted": _mounted(page, room["feed"], newest_ids),
                                         "newest_page_rows": len(newest_ids), "place": page.evaluate(_PLACE, room["feed"])})
                assert all(read["cursor"] for read in pages), "each older page read by its cursor"
                assert card.count() == 0, "reading on released the question's page"
                assert _mounted(page, room["feed"], question_ids) == 0, "the question's page left the feed"
                assert page.locator(f"{room['feed']} .chat-load-older button").evaluate("n => !n.hidden"), \
                    "the room's beginning is still ahead"
                origin = _read_at(page, room["feed"], _filler_id(pages[-1], tag))
                evidence.record("released_origin", origin)
                detail_reads = sum(1 for method, u in log["requests"] if method == "GET" and u.endswith(f"/api/tasks/{task_id}"))
                _view_question_by_keyboard(page, room, task_id, engine)
                landed = _landed(page, room, card)
                evidence.record("released_landed", landed)
                assert sum(1 for method, u in log["requests"] if method == "GET"
                           and u.endswith(f"/api/tasks/{task_id}")) > detail_reads, "the reveal read the real task detail"
                evidence.shot(page, "03-released-question")
                after = _back_by_keyboard(page, room)
                evidence.record("released_back", after)
                _assert_exact_return(origin, after, "released question, loaded passage")
                evidence.shot(page, "04-released-back")

                # ---- origin gone: from the question, ↓ re-anchors on the present and the pager lets the passage go.
                origin_id = origin["history"]
                _view_question_by_keyboard(page, room, task_id, engine)
                evidence.record("gone_landed", _landed(page, room, card))
                cursor_reads = len([read for read in _reads(page, chat_id) if read["cursor"]])
                latest_reads = len(_reads(page, chat_id))
                page.locator(f"{room['root']} .chat-scroll-bottom-btn").focus()
                page.keyboard.press("Enter")
                page.wait_for_function("([n, id]) => window.__historyReads.filter(r => r.chatId === id).length > n"
                                       " && window.__historyReads.every(r => r.done)", arg=[latest_reads, chat_id],
                                       timeout=30_000)
                page.evaluate(_FRAMES)
                page.wait_for_timeout(600)
                present = _reads(page, chat_id)[latest_reads:]
                evidence.record("present", {"cursors": [read["cursor"] for read in present], "place": page.evaluate(_PLACE, room["feed"]),
                                            "origin_mounted": _mounted(page, room["feed"], [origin_id])})
                assert present and not present[0]["cursor"], "↓ read the room's newest page"
                assert _mounted(page, room["feed"], [origin_id]) == 0, "the pager released the passage left"
                assert _facts(page, room["scope"])["back"] == "Back to reading", "Back stays offered"
                evidence.shot(page, "05-present-origin-released")
                place_before = page.evaluate(_PLACE, room["feed"])
                after = _back_by_keyboard(page, room)
                evidence.record("gone_back", after)
                assert after["back"] == "Earlier place is no longer loaded", after
                assert after["live"] == "The earlier reading place is no longer loaded.", after
                assert len([read for read in _reads(page, chat_id) if read["cursor"]]) == cursor_reads, \
                    "Back loads no lost place: no older page is read again"
                assert _mounted(page, room["feed"], [origin_id]) == 0, "the lost passage is not rebuilt"
                assert abs(after["place"]["scrollTop"] - place_before["scrollTop"]) <= 2, ("the reader stays", place_before, after)
                assert after["focus"] != "body", after
                evidence.shot(page, "06-gone-back")
                page.wait_for_timeout(4300)
                later = {"back": _facts(page, room["scope"])["back"], "focus": page.evaluate(_ACTIVE)}
                evidence.record("gone_after_4s", later)
                assert later["back"] == "" and later["focus"] != "body", later

                assert _decision_posts(log) == [], "View question answers nothing"
                assert (_waiting_turn(url, chat_id) or {}).get("activity_id") == task_id, "the turn still waits for the owner"
                assert log["pageerrors"].unexpected() == [], log["pageerrors"]
            except Exception:
                evidence.shot(page, "failure")
                raise
            finally:
                evidence.record("history_reads", [{k: read.get(k) for k in ("cursor", "chatId", "status", "error")}
                                                  | {"rows": len((read.get("body") or {}).get("messages") or []),
                                                     "has_more": (read.get("body") or {}).get("has_more")}
                                                  for read in page.evaluate("() => window.__historyReads || []")])
                evidence.record("log", log)
                browser.close()
    finally:
        model.finish()
        evidence.record("model_calls", list(model.calls))
