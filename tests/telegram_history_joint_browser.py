"""Browser consumers of the Hub's Telegram import over this checkout's real Host.

The Place sheet's conversation reader: the browser runs this checkout's Skills page, Place sheet and reader
modules (``tests._skills_page_harness``), and its places, rooms and history requests are answered by the
real core routes over the data root that the Hub's importer and live intake wrote through the real Host
Service (``tests/telegram_history_joint.py``). That reader pages only older history (``Load older``): it
reads no page around a row and no arrivals, so neither is claimed here.

The Hub widget's History tab: the Hub's own widget harness (its module bridge, registered routes and
plugin), with the companion's importer writing through the real Host instead of the Hub's wire fake.

The core repair (task b520a762) asserts what the reader now does: it reads a room afresh after its view
changes, names the year of every row time outside the current one, calls its start the record saved when
it was read and reads newer saves through Refresh history. Rows sharing one Telegram second read in send
order: the Hub (r3) sends ``message.source_order``. Screenshots go to OUROBOROS_UI_EVIDENCE_DIR. CI never collects this module. Run it explicitly with
OUROBOROS_RUN_UI_SMOKE=1, ``-m ui_browser`` and OUROBOROS_TELEGRAM_SKILL_ROOT passed through ``env`` inside
the safe_test command.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from ouroboros import dialogue_store as ds
from ouroboros import places as p
from tests._skills_page_harness import URL, _document, fixture, open_skills
from tests._skills_page_harness import skills_browser as skills_browser
from tests.telegram_history_joint import ACCOUNT, CONNECTOR, DAY, GROUP, MINUTE, MIRA, SKILL, TOM, World, hub, room
from tests.telegram_history_joint import _mutation as _mutation  # the red controls (OUROBOROS_JOINT_MUTATION)

REPO = Path(__file__).resolve().parents[1]
HUB_WIDGET_TESTS = SKILL.parents[1] / "tests" / "test_telegram_participant_widget_browser.py"
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial,
              pytest.mark.skipif(not (SKILL / "userbot" / "history_import.py").is_file() or not HUB_WIDGET_TESTS.is_file(),
                                 reason="set OUROBOROS_TELEGRAM_SKILL_ROOT to a telegram-user-session checkout")]
BOARD = -1001  # the Hub widget harness's synthetic group
_ROWS = """() => [...document.querySelectorAll('[data-history-messages] .place-convo-row')].map(row => ({
    who: row.querySelector('.place-convo-who')?.textContent || '', voice: row.className.replace('place-convo-row', '').trim(),
    shown: row.querySelector('time')?.textContent || '', ts: row.querySelector('time')?.getAttribute('datetime') || '',
    title: row.querySelector('time')?.getAttribute('title') || '',
    text: (row.querySelector('.place-convo-text')?.textContent || '').trim()}))"""


def _hub_widget_harness():
    """The Hub's own widget browser harness (READ ONLY), served over this checkout's module bridge."""
    os.environ["OUROBOROS_THEME_CORE_ROOT"] = str(REPO)
    name = "hub_telegram_widget_browser"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HUB_WIDGET_TESTS)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


if HUB_WIDGET_TESTS.is_file():
    widget_server = _hub_widget_harness().widget_server  # the Hub's fixture, used as it is


class Producers:
    """A World and every write into it on one thread: Playwright's sync API owns this thread's event loop,
    and the companion's journal is a SQLite handle of the thread that opened it."""

    def __init__(self, setup, *args, **kwargs):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="joint-producers")
        try:
            self.world = self.run(setup, *args, **kwargs)
        except BaseException:
            # pytest.skip also escapes setup; an unconstructed fixture has no finalizer.
            self.pool.shutdown(wait=True)
            raise

    def run(self, call, *args, **kwargs):
        return self.pool.submit(call, *args, **kwargs).result()

    def close(self):
        try:
            self.run(self.world.close)
        finally:
            self.pool.shutdown(wait=True)


@pytest.mark.parametrize("failure", [ValueError("setup failed"), pytest.skip.Exception("setup skipped")])
def test_producers_join_their_thread_when_setup_does_not_return(failure):
    import threading

    workers = []
    def fail_setup():
        workers.append(threading.current_thread())
        raise failure

    with pytest.raises(type(failure), match=str(failure)):
        Producers(fail_setup)
    assert len(workers) == 1
    assert not workers[0].is_alive()


def _place_world(base):
    """Two years of Board games on Fridays, an older year outside the first plan, and Mira's live chat."""
    w = World(base, consciousness=False)
    T, now, say = w.hub.T, w.now, w.say
    photo = T.MessageMediaPhoto(photo=T.Photo(id=78, access_hash=1, file_reference=b"r", date=None, sizes=[], dc_id=1))
    mid = 100
    for n in range(40):  # three years ago: outside the first plan's dates
        say(GROUP, mid, f"From the first season, note {n:02d}", now - (1100 - n * 7) * DAY, (MIRA, TOM)[n % 2])
        mid += 1
    marks = {}
    for n in range(300):  # the last two years, one message every 2.4 days
        if n == 150:
            marks["year_ago"] = say(GROUP, mid, "A year ago today: our first game night at the cafe.",
                                    now - 365 * DAY, MIRA)
            mid += 1
        if n == 200:
            marks["forwarded"] = [say(GROUP, mid + k, f"Forwarded rule {k + 1} of 4", now - 200 * DAY, TOM)
                                  for k in range(4)]  # one Telegram second
            mid += 4
        if n == 250:
            marks["own"] = say(GROUP, mid, "I booked the big table for Friday.", now - 90 * DAY, own=True)
            marks["photo"] = say(GROUP, mid + 1, "", now - 90 * DAY + 3 * MINUTE, TOM, media=photo)
            mid += 2
        say(GROUP, mid, f"Game night chat {n:03d}", now - timedelta(days=720 - n * 2.4), (MIRA, TOM)[n % 2])
        mid += 1
    marks["today"] = say(GROUP, mid, "Same day this year: who is in for Friday?", now - 3 * 60 * MINUTE, MIRA)
    marks["sent_here"] = say(GROUP, mid + 1, "I will bring the snacks.", now - 9 * MINUTE, own=True)
    marks["live"] = say(GROUP, mid + 2, "See you all on Friday.", now - 8 * MINUTE, TOM)
    # Even confirmed as the owner's own account, its imported messages are never shown as the owner's.
    p.edit_place(w.data, w.place.place_id, "confirm_owner_account", author="owner", platform_actor_id=str(ACCOUNT),
                 account_id=str(ACCOUNT))
    w.companion.report_delivered(GROUP, marks["sent_here"])
    w.companion.deliver_live(marks["live"], GROUP, TOM)
    first = w.plan([GROUP], now - 800 * DAY)
    w.act(first, "start")
    assert w.drive(first, 200) == "done", w.view(first)
    # Mira's chat: 155 live messages first (a room read in stored order), five earlier ones in Telegram.
    for n in range(5):
        say(MIRA, 10 + n, f"Before the companion: Mira {n}", now - (30 - n) * DAY, MIRA)
    for n in range(155):
        w.companion.deliver_live(say(MIRA, 100 + n, f"Live with Mira {n:03d}", now - (60 - n * 0.3) * MINUTE, MIRA),
                                 MIRA, MIRA)
    w.marks = marks
    return w


def _expected(pages):
    """The rows the reader shows for pages read newest first: older pages above, each row once."""
    seen, shown = set(), []
    for page in reversed(pages):
        for row in page["messages"]:
            if not row.get("place_history") or row.get("is_progress") or row.get("role") not in {"user", "assistant", "system"}:
                continue
            key = row.get("row_address") or row.get("history_id")
            if key and key in seen:
                continue
            seen.add(key)
            shown.append(row["text"])
    return shown


@pytest.mark.parametrize("engine,width,theme", [("chromium", 1280, "light"), ("webkit", 390, "dark")])
def test_imported_telegram_history_reads_in_the_place_reader(skills_browser, tmp_path, monkeypatch, engine, width, theme):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import places_api
    from ouroboros.gateway.history import make_chat_history_endpoint
    producers = Producers(_place_world, tmp_path)
    w = producers.world
    marks = w.marks
    monkeypatch.setattr(places_api, "request_drive_root", lambda _request: w.data)
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [])
    client = TestClient(Starlette(routes=[*places_api.places_routes(),
                                          Route("/api/chat/history", make_chat_history_endpoint(w.data))]))
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "evidence") / "telegram-history-joint"
    evidence.mkdir(parents=True, exist_ok=True)
    prefix = f"{engine}-{width}-{theme}"
    observations = {"synthetic": True, "engine": engine, "width": width, "theme": theme, "reads": []}
    skill = fixture("extensions_fix_r1.json")["skills"][0]
    skill.update(name=CONNECTOR, description="Synthetic Telegram connector for the joint import check.",
                 connects=["telegram"], install=None)
    browser = getattr(skills_browser, engine).launch(headless=True)
    page = browser.new_page(viewport={"width": width, "height": 844 if width == 390 else 900}, device_scale_factor=1,
                            locale="en-US", timezone_id="UTC")
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    skills = open_skills(page, extensions=[skill])
    page.route(URL, lambda route: route.fulfill(content_type="text/html; charset=utf-8", body=_document().replace(
        "<head>", '<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">')))

    def api(request):
        assert request.method == "GET", request.url
        url = urlsplit(request.url)
        response = client.get(url.path + ("?" + url.query if url.query else ""))
        if url.path == "/api/chat/history":
            body = response.json()
            observations["reads"].append({"status": response.status_code, "count": len(body.get("messages") or []),
                                          "has_more": body.get("has_more"), "error": body.get("error")})
        return response.status_code, response.json()
    for path in ("/api/places", f"/api/places/{w.place.place_id}/rooms", "/api/chat/history"):
        skills.answers[path] = api
    sheet = page.locator(".place-sheet")
    reader = sheet.locator(".place-convo")
    status = reader.locator("[data-history-status]")

    def shot(name):
        page.screenshot(path=str(evidence / f"{prefix}-{name}.png"))

    def rows():
        return page.evaluate(_ROWS)

    def settled():
        page.wait_for_function("() => { const s = document.querySelector('.place-convo [data-history-status]');"
                               " return s && !/Reading captured history/.test(s.textContent); }")

    def open_room(chat_id):
        sheet.locator(f'[data-place-room="{chat_id}"]').click()
        reader.locator(".place-convo-row").first.wait_for()
        settled()

    def back():
        reader.locator("[data-history-back]").click()
        sheet.locator("[data-place-room]").first.wait_for()

    def older_to_start():
        while reader.locator("[data-history-older]").is_visible():
            reader.locator("[data-history-older]").click()
            settled()

    def notes():
        return reader.locator("[data-history-coverage]").inner_text()

    try:
        page.goto(URL, wait_until="networkidle")
        page.add_script_tag(url="/static/purify.min.js")
        page.add_script_tag(url="/static/marked.min.js")
        page.add_style_tag(url="/static/settings.css")
        page.add_style_tag(url="/static/consciousness.css")
        page.evaluate("({theme,width}) => { document.documentElement.classList.add('ouro-ui'); document.documentElement"
                      ".dataset.theme=theme; document.documentElement.dataset.layout=width < 720 ? 'stack' : 'docked'; }",
                      {"theme": theme, "width": width})
        page.locator(f'.skills-card[data-skill="{CONNECTOR}"] .skills-card-name').click()
        page.locator(f'.skills-detail-sheet [data-place-open="{w.place.place_id}"]').click()
        sheet.locator(f'[data-place-room="{room(GROUP)}"]').wait_for()
        shot("rooms")

        # The newest page, then every older one: the room in time order, each row once, to its saved start,
        # with no note of a missing range.
        open_room(room(GROUP))
        api_pages = w.pages(GROUP, 150)
        assert [row["text"] for row in rows()] == _expected(api_pages[:1])
        assert "Earlier saved records are available." in notes()
        shot("newest")
        older_to_start()
        everything = rows()
        assert [row["text"] for row in everything] == _expected(api_pages)
        assert len(everything) == len({(row["ts"], row["text"]) for row in everything}) == len(w.reading_order(GROUP))
        assert [row["ts"] for row in everything] == sorted(row["ts"] for row in everything)
        from ouroboros.place_arrivals import reading_key
        group = {entry.row["text"]: entry.row for entry in ds.iter_rows(w.data, "chat") if entry.row.get("chat_id") == room(GROUP)}
        # Each row shows its reading time, the Telegram send time its arrival recorded or else when it was saved;
        # its title keeps the actual saved time (an import's receipt), never a send time a receipt row lacks.
        assert all(row["ts"] == reading_key(group[row["text"]], 0)[0] for row in everything)
        for row in everything:
            stored_row = group[row["text"]]
            saved = (stored_row.get("historical_import") or {}).get("received_at") or stored_row["ts"]
            sent = (stored_row.get("source_order") or {}).get("time")
            assert row["title"] == (f"Sent {sent} · saved {saved}" if sent else stored_row["ts"]), (row, stored_row)
        assert "Start of the record saved when this was read. Refresh history to include anything saved since." in notes()
        assert not any(words in notes() for words in ("unavailable", "not fully reported", "limited", "still being imported"))
        # Authors as Telegram knows them: participants, the account's own messages (never "(you)", although it is
        # confirmed as the owner's), and the one row Ouroboros sent here, its delivery receipt.
        by_text = {row["text"]: row for row in everything}
        own, sent_here = by_text[marks["own"].message], by_text[marks["sent_here"].message]
        assert (own["who"], own["voice"]) == ("Ouroboros", "is-participant") and sent_here["voice"] == "is-assistant"
        assert "(you)" not in reader.inner_text()
        observations["forwarded_order"] = [row["text"] for row in everything if row["text"].startswith("Forwarded rule")]
        assert observations["forwarded_order"] == [f"Forwarded rule {k} of 4" for k in range(1, 5)]  # one Telegram second
        observations["same_day_labels"] = [{key: by_text[marks[name].message][key] for key in ("shown", "ts")}
                                           for name in ("year_ago", "today")]
        this_year = datetime.now(timezone.utc).year  # the page runs in UTC: a time outside this year names its year
        assert all((row["ts"][:4] in row["shown"]) == (int(row["ts"][:4]) != this_year) for row in everything), [
            (row["ts"], row["shown"]) for row in everything if (row["ts"][:4] in row["shown"]) != (int(row["ts"][:4]) != this_year)][:3]
        observations["oldest_rows"] = [{key: row[key] for key in ("shown", "ts", "who", "text")} for row in everything[:2]]
        reader.locator(".place-convo-row", has_text="Forwarded rule 1 of 4").scroll_into_view_if_needed()
        shot("same-second")
        reader.locator(".place-convo-row", has_text=marks["year_ago"].message).scroll_into_view_if_needed()
        shot("year-ago")
        reader.locator(".place-convo-row").first.scroll_into_view_if_needed()
        shot("start")

        # An import lands while the reader is open: what its pages and its start note then say.
        back()
        open_room(room(GROUP))
        second = producers.run(w.plan, [GROUP], w.now - 1200 * DAY, w.now - 800 * DAY)
        producers.run(w.act, second, "start")
        assert producers.run(w.drive, second) == "done"
        older_to_start()
        stale = rows()
        observations["import_while_open"] = {
            "shown": len(stale), "stored": len(w.reading_order(GROUP)), "note": notes(),
            "new_rows_shown": sum(row["text"].startswith("From the first season") for row in stale)}
        # The start is the record as it was read; Refresh history reads what was saved since.
        assert not observations["import_while_open"]["new_rows_shown"]
        assert notes().startswith("Start of the record saved when this was read. Refresh history to include anything saved since.")
        reader.locator(".place-convo-row").first.scroll_into_view_if_needed()
        shot("import-while-open-start")
        reader.locator("[data-history-refresh]").click()
        settled()
        older_to_start()
        refreshed = rows()
        assert [row["text"] for row in refreshed] == _expected(w.pages(GROUP, 150)) and len(refreshed) == len(w.reading_order(GROUP))
        assert refreshed[0]["text"] == "From the first season, note 00" and str(int(refreshed[0]["ts"][:4])) in refreshed[0]["shown"]
        back()
        open_room(room(GROUP))
        older_to_start()
        fresh = rows()
        assert [row["text"] for row in fresh] == _expected(w.pages(GROUP, 150)) and len(fresh) == len(w.reading_order(GROUP))
        assert fresh[0]["text"] == "From the first season, note 00"
        observations["oldest_rows_after_reopen"] = [{key: row[key] for key in ("shown", "ts", "text")} for row in fresh[:2]]
        reader.locator(".place-convo-row").first.scroll_into_view_if_needed()
        shot("import-while-open-reopened")

        # The room's first import lands while its stored-order reader is open: what the next older read says.
        back()
        open_room(room(MIRA))
        assert len(rows()) == 150 and reader.locator("[data-history-older]").is_visible()
        third = producers.run(w.plan, [MIRA], w.now - 60 * DAY, w.now - 2 * DAY)
        producers.run(w.act, third, "start")
        assert producers.run(w.drive, third) == "done"
        reader.locator("[data-history-older]").click()
        settled()
        observations["first_import_while_open"] = {"status": status.inner_text(), "notes": notes(),
                                                   "retry": reader.locator("[data-history-retry]").is_visible()}
        # The refused stored-order cursor is never resent: the room is read again from its newest page.
        assert observations["first_import_while_open"]["status"] == "Saved history changed, so it was read again from the newest messages."
        assert not observations["first_import_while_open"]["retry"]
        status.scroll_into_view_if_needed()
        shot("first-import-while-open")
        older_to_start()
        assert [row["text"] for row in rows()][0] == "Before the companion: Mira 0" and len(rows()) == 160
        back()
        open_room(room(MIRA))
        older_to_start()
        mira = rows()
        assert [row["text"] for row in mira] == _expected(w.pages(MIRA, 150)) and len(mira) == 160
        assert mira[0]["text"] == "Before the companion: Mira 0"
        assert errors == [], errors
        assert skills.posts == []
    finally:
        observations["pageerrors"] = errors
        observations["browser_version"] = browser.version
        (evidence / f"{prefix}.json").write_text(json.dumps(observations, indent=2, default=str))
        skills.close(browser)
        client.close()
        producers.close()


def _view_change_world(base):
    """Mira's chat: 155 live messages (a room read in stored order) and five earlier ones only Telegram holds."""
    w = World(base, consciousness=False)
    for n in range(5):
        w.say(MIRA, 10 + n, f"Before the companion: Mira {n}", w.now - (30 - n) * DAY, MIRA)
    for n in range(155):
        w.companion.deliver_live(w.say(MIRA, 100 + n, f"Live with Mira {n:03d}", w.now - (60 - n * 0.3) * MINUTE, MIRA),
                                 MIRA, MIRA)
    return w


def test_an_open_place_reader_recovers_after_the_rooms_first_import(skills_browser, tmp_path, monkeypatch):
    """The owner reads Mira's live chat while the first import into it lands. The next older read is refused
    (409 history_view_changed: the room now reads in time order); the reader reads the room afresh, says so and
    reaches its saved start, never resending the refused cursor or showing the bare reason code. (QA 2f335d42 held
    this as a strict-xfail RED; that original and its RED run are kept in .review-evidence/joint-original.)"""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import places_api
    from ouroboros.gateway.history import make_chat_history_endpoint
    producers = Producers(_view_change_world, tmp_path)
    w = producers.world
    monkeypatch.setattr(places_api, "request_drive_root", lambda _request: w.data)
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [])
    client = TestClient(Starlette(routes=[*places_api.places_routes(),
                                          Route("/api/chat/history", make_chat_history_endpoint(w.data))]))
    skill = fixture("extensions_fix_r1.json")["skills"][0]
    skill.update(name=CONNECTOR, description="Synthetic Telegram connector for the joint import check.",
                 connects=["telegram"], install=None)
    browser = skills_browser.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900}, locale="en-US", timezone_id="UTC")
    skills = open_skills(page, extensions=[skill])

    def api(request):
        url = urlsplit(request.url)
        response = client.get(url.path + ("?" + url.query if url.query else ""))
        return response.status_code, response.json()
    for path in ("/api/places", f"/api/places/{w.place.place_id}/rooms", "/api/chat/history"):
        skills.answers[path] = api
    reader = page.locator(".place-sheet .place-convo")
    settled = ("() => { const s = document.querySelector('.place-convo [data-history-status]');"
               " return s && !/Reading captured history/.test(s.textContent); }")
    try:
        page.goto(URL, wait_until="networkidle")
        page.add_script_tag(url="/static/purify.min.js")
        page.add_script_tag(url="/static/marked.min.js")
        page.locator(f'.skills-card[data-skill="{CONNECTOR}"] .skills-card-name').click()
        page.locator(f'.skills-detail-sheet [data-place-open="{w.place.place_id}"]').click()
        page.locator(f'.place-sheet [data-place-room="{room(MIRA)}"]').click()
        reader.locator(".place-convo-row").first.wait_for()
        page.wait_for_function(settled)
        plan = producers.run(w.plan, [MIRA], w.now - 60 * DAY, w.now - 2 * DAY)
        producers.run(w.act, plan, "start")
        assert producers.run(w.drive, plan) == "done"
        reader.locator("[data-history-older]").click()
        page.wait_for_function(settled)
        assert reader.locator("[data-history-status]").inner_text() == (
            "Saved history changed, so it was read again from the newest messages.")
        if reader.locator("[data-history-retry]").is_visible():
            reader.locator("[data-history-retry]").click()
            page.wait_for_function(settled)
        for _ in range(5):  # a refused cursor keeps Load older offered: bounded, never a wait for the start
            if not reader.locator("[data-history-older]").is_visible():
                break
            reader.locator("[data-history-older]").click()
            page.wait_for_function(settled)
        status = reader.locator("[data-history-status]").inner_text()
        texts = [row["text"] for row in page.evaluate(_ROWS)]
        assert "history_view_changed" not in status, status
        assert texts[0] == "Before the companion: Mira 0" and len(texts) == 160, (status, texts[:2], len(texts))
    finally:
        skills.close(browser)
        client.close()
        producers.close()


def _widget_world(base, hub_root):
    """The Hub widget test's import scenario with the Host's earlier copies made by the real Host: an earlier
    companion install's live capture of message 10 under other text, live intake's admitted capture of 29,
    this companion's receipt for 32 recorded and its receipt for 31 not sent yet."""
    h = hub()
    board = h.T.Chat(id=1001, title="Board games on Fridays", photo=h.T.ChatPhotoEmpty(), participants_count=24,
                     date=None, version=1)
    w = World(base, consciousness=False, entities={BOARD: board, MIRA: h.fakes.user(MIRA, "Mira", "mira")},
              companion_state=hub_root)
    now = datetime.now(timezone.utc)
    w.days = lambda n: now - timedelta(days=n)
    photo = h.T.MessageMediaPhoto(photo=h.T.Photo(id=78, access_hash=1, file_reference=b"r", date=None, sizes=[], dc_id=1))
    say, days = w.say, w.days
    say(BOARD, 9, "Planning from last season", days(40), TOM)
    say(BOARD, 10, "Has anyone tried this with six players?", days(12), TOM)
    say(BOARD, 11, "", days(11), MIRA, media=photo)
    say(BOARD, 12, "Six works, but setup takes longer.", days(10), TOM)
    asked = say(MIRA, 29, "Are we still on for Friday?", days(9), MIRA)
    say(MIRA, 30, "Can you bring the rules?", days(8), MIRA)
    sure = say(MIRA, 31, "Sure, I will bring them.", days(8) + 2 * MINUTE, own=True)
    summary = say(MIRA, 32, "Here is the rules summary.", days(7), own=True)
    earlier = w.open_companion(base / "earlier")
    earlier.deliver_live(h.fakes.message(10, BOARD, "Has anyone tried this with five players?", days(12), sender=TOM),
                         BOARD, TOM)
    w.companion.deliver_live(asked, MIRA, MIRA)
    w.companion.report_delivered(MIRA, summary)
    w.companion.journal.report("auto:31", {"schema_version": 1, "delivery_id": "auto:31", "part_id": "0", "state": "delivered",
                                           "provider": "telegram", "account_id": str(ACCOUNT), "conversation_id": str(MIRA),
                                           "thread_id": "", "text": sure.message, "format": "text",
                                           "message": {"message_id": "31"}, "origin": {"kind": "automatic"}})
    status, identity = w.companion.engine.host.call("GET", "/identity")  # the companion's start-up identity check
    assert status == 200 and identity["presence_history_import_version"] == 1
    h.participant.record_host_facts(hub_root, identity)
    return w


def _finish(w, limit=60):
    for _ in range(limit):
        if not w.companion.journal.db.execute("SELECT 1 FROM imports WHERE state='running'").fetchone():
            return
        w.step()
    raise AssertionError("the import did not finish")


@pytest.mark.parametrize("engine,width", [("chromium", 1100), ("webkit", 390)])
def test_the_widget_import_runs_against_the_real_host(widget_server, tmp_path, monkeypatch, engine, width):
    """The Hub test's Start, partial confirmation, Pause, Resume, held own message and Done, with every history
    request answered by the real Host and the widget's words checked against what the Host then holds."""
    from playwright.sync_api import expect, sync_playwright
    harness = _hub_widget_harness()
    root, state, url = widget_server
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "evidence") / "telegram-history-joint-widget"
    evidence.mkdir(parents=True, exist_ok=True)
    producers = Producers(_widget_world, tmp_path, Path(root))
    w = producers.world
    real, appended = ds.append, []

    def unconfirmed_second(root_, stream, row, **kwargs):
        if row.get("historical_import"):
            appended.append(row["client_message_id"])
            if len(appended) == 2:  # the second record of the request: unconfirmed, every later one unattempted
                raise ds.StoreBusy("dialogue store busy", reason="contention")
        return real(root_, stream, row, **kwargs)
    messages = []

    def shot(page, frame, name):
        harness.settle(page, frame)
        page.screenshot(path=str(evidence / f"telegram-{engine}-{width}-{name}.png"), full_page=True)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            page, frame = harness.open_widget(browser, url, width, messages, clock=True)
            frame.get_by_role("button", name="4 · History").click()
            expect(frame.locator("#import-availability")).to_contain_text("can record imported history")
            frame.get_by_role("button", name="Chosen chats").click()
            frame.locator(".check", has_text="Board games on Fridays").locator("input").check()
            frame.locator(".check", has_text="Mira").locator("input").check()
            frame.locator('input[data-draft="importFrom"]').fill(w.days(20).astimezone().strftime("%Y-%m-%d"))
            frame.locator('input[data-draft="importTo"]').fill(w.days(1).astimezone().strftime("%Y-%m-%d"))
            frame.get_by_role("button", name="Save import plan").click()
            expect(frame.locator('[data-result="import"]')).to_contain_text("until you press Start")
            plan = frame.locator(".plan")
            expect(plan).to_contain_text("Saved · not started")
            producers.run(w.step, 3)
            assert w.companion.wire.history == [] and w.telegram.reads() == []  # a saved plan never starts by itself
            plan.get_by_role("button", name="Start").click()
            expect(plan).to_contain_text("Importing · chat 1 of 2")
            # The companion reads one page; the Host confirms its first record and not the second.
            monkeypatch.setattr(ds, "append", unconfirmed_second)
            producers.run(w.step, 2)
            monkeypatch.setattr(ds, "append", real)
            assert [status for status, _body in w.companion.wire.answers] == [503]  # the history request
            page.clock.run_for(4100)
            expect(plan).to_contain_text("Importing · waiting: Ouroboros did not confirm some records; they go again unchanged")
            expect(plan.locator(".plan-counts")).to_have_text("1 recorded")
            shot(page, frame, "history-import-waiting-dark")
            plan.get_by_role("button", name="Pause").click()
            expect(plan).to_contain_text("Paused · by you")
            calls = len(w.companion.wire.calls)
            producers.run(w.step, 3)
            assert len(w.companion.wire.calls) == calls  # paused: nothing read, nothing sent
            plan.get_by_role("button", name="Resume").click()
            expect(plan).to_contain_text("Importing")
            producers.run(w.step)
            page.clock.run_for(4100)
            expect(plan.locator(".plan-counts")).to_have_text("2 recorded · 1 not imported")
            producers.run(_finish, w)
            page.clock.run_for(4100)
            expect(plan.locator(".plan-state")).to_have_text("Partial · 1 message held back")
            expect(plan.locator(".plan-counts")).to_have_text(
                "3 recorded · 2 already in the conversation · 1 held back · 1 not imported")
            plan.locator("summary").click()
            expect(plan.locator(".plan-lines li", has_text="Board games on Fridays:")).to_contain_text(
                "3 read · 2 recorded · 1 not imported (already recorded with other content) · every message in these dates read")
            expect(plan.locator(".plan-lines li", has_text="message 31")).to_have_text(
                "Mira, message 31: own message held back: its delivery receipt is not confirmed yet; Resume checks again")
            shot(page, frame, "history-import-partial-dark")
            posts = len(w.companion.wire.history)
            plan.get_by_role("button", name="Resume").click()  # before the receipt is recorded: still held
            expect(plan.locator(".plan-state")).to_contain_text("Importing")
            producers.run(_finish, w)
            page.clock.run_for(4100)
            expect(plan.locator(".plan-state")).to_have_text("Partial · 1 message held back")
            assert len(w.companion.wire.history) == posts
            # The companion reports message 31's receipt and the real Host records it; Resume then skips 31.
            producers.run(lambda: asyncio.run(w.companion.engine.reports_once()))
            plan.get_by_role("button", name="Resume").click()
            expect(plan.locator(".plan-state")).to_contain_text("Importing")
            producers.run(_finish, w)
            page.clock.run_for(4100)
            expect(plan.locator(".plan-state")).to_have_text("Done · 1 not imported")
            expect(plan.locator(".plan-state")).to_have_class(re.compile(r"\btone-warn\b"))
            expect(plan.locator(".plan-counts")).to_have_text("3 recorded · 3 already in the conversation · 1 not imported")
            expect(plan.locator("button")).to_have_count(0)
            expect(plan.locator(".plan-lines li", has_text="message 10")).to_contain_text("already recorded with other content")
            assert len(w.companion.wire.history) == posts
            shot(page, frame, "history-import-done-dark")
            page.evaluate("ouroTheme.set('light')")
            expect(frame.locator("html")).to_have_attribute("data-theme", "light")
            shot(page, frame, "history-import-done-light")
        finally:
            browser.close()
            producers.close()
    assert messages == [], messages
    assert not state["faults"], state["faults"]
    assert [body["action"] for name, body in state["posts"]] == ["plan", "start", "pause", "resume", "resume", "resume"]
    assert w.sent() == [["12", "11", "10"], ["11", "10"], ["30"]]
    assert [status for status, _body in w.companion.wire.answers] == [503, 409, 200]
    # The Host holds each Telegram message once, as the widget says.
    board, mira = (w.by_message(peer) for peer in (BOARD, MIRA))
    assert {mid: "historical_import" in row for mid, row in board.items()} == {"10": False, "11": True, "12": True}
    assert board["10"]["text"] == "Has anyone tried this with five players?"
    assert {mid: row.get("type") for mid, row in mira.items()} == {"29": "place_arrival", "30": "place_arrival",
                                                                  "31": "presence_delivery", "32": "presence_delivery"}
    assert "historical_import" in mira["30"] and "historical_import" not in mira["29"]
