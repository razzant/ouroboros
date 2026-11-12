"""The sidebar's Projects list beyond its recent feed (web/modules/projects_nav.js) through the actual SPA, the real
gateway and registry (the `pins_ui` fixture): reads that land in either order, a list ordered by the newest message
whose browse keeps sorting and the place being read, the rooms' real unread beyond the feed, and a reference that
reads its room."""
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from ouroboros import dialogue_store, projects_registry as registry, room_messages
from tests.test_project_pin_browser import (
    LONG, choose, click_reference, coarse, expect, no_read_acknowledged, open_drawer, order,
    reads_of, refresh, row, seed_archive, settle, shown_ids, until,
)
from tests._dialogue import advance_room_revision
from tests._shared import producer_evidence_dir
pytest_plugins = ("tests.test_project_pin_browser", "tests.test_subscription_setup_browser")

pytestmark = pytest.mark.ui_browser


LATER = lambda minute: (datetime(2099, 6, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)).isoformat()  # noqa: E731
SPREAD = lambda minute: (datetime(2099, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)).isoformat()  # noqa: E731


def date_history(root):
    """The history pass the first maintenance look after an update runs (``room_messages``)."""
    dialogue_store.ensure(root)
    room_messages.run(root, lambda: False)


def room_of(root, pid):
    return registry.get_project(root, pid)["chat_id"]


def say(root, chat, text, *, ts, **fields):
    """A message in a room through the one writer seam (``dialogue_store.append``)."""
    dialogue_store.append(root, "chat", {"ts": ts, "chat_id": chat, "direction": "out", "role": "assistant",
                                         "text": text, **fields})


def spread(root, count=498):
    """``seed_archive``'s rooms each with a message, a minute apart in their order, after every room's creation."""
    date_history(root)
    for index in range(count):
        say(root, 40000 + index, f"Message {index}", ts=SPREAD(index))


def is_state(request):
    return urlparse(request.url).path == '/api/state'


def is_list(request):
    return parse_qs(urlparse(request.url).query).get('all') == ['1']


def is_cursors(response):
    return urlparse(response.url).path == '/api/ui/preferences' and response.request.method == 'GET'


def changed(control, pid):
    """The registry's `projects_changed` on the page's socket: a forced state read, and a complete list read while a
    find or a browse is open."""
    control["sockets"][-1].send(json.dumps({"type": "projects_changed", "project_id": pid}))


def reopen(page, control):
    """The page's socket closes and ws.js opens another a second later, as across a network change."""
    opened = len(control["sockets"])
    control["sockets"][-1].close()
    until(page, lambda: page.clock.run_for(500) or len(control["sockets"]) > opened, rounds=60)


def test_reads_meet_in_either_order_by_request_time_and_a_deleted_room_stays_gone(pins_ui):
    """Reads are ordered by their requests, whichever answer lands first. A state read requested before a rename and a
    deletion lands after the complete list read that saw them: the list's facts stand (the old feed renames nothing
    back and brings no deleted room back). A complete list requested before another rename lands after the state read
    that saw it: the feed's name stands, while the list still speaks for a room only it renamed."""
    ui, page, root, control = pins_ui, pins_ui["page"], pins_ui["root"], pins_ui["control"]
    seed_archive(root)
    refresh(page)
    find = page.locator('#nav-projects .nav-find-input')
    status = page.locator('.nav-find-status')
    find.fill("Recent project")
    expect()(status).to_have_text("1 project found")  # the first complete read settled
    control["unread"].add("newer")  # the old feed's own mark: the header's count says when it has landed
    control["hold"] = True
    with page.expect_request(is_state):
        refresh(page)  # requested now, answered only after the changes below
    ui["client"].post('/api/projects/newer/update', json={"name": "Renamed freshly"}).raise_for_status()
    registry.begin_project_deletion(root, "older")
    registry.complete_project_deletion(root, "older")
    with page.expect_response(lambda response: is_list(response.request)):
        changed(control, "newer")
    expect()(status).to_have_text('No project names contain “Recent project”.')  # that list is applied
    stale_route, stale_body = control["held"].pop(0)
    with page.expect_request(is_state):  # the forced read queued behind it starts once it settles, and is held
        stale_route.fulfill(content_type="application/json", body=stale_body)
    expect()(page.locator('#nav-projects-count')).to_have_text("2")  # the old feed has landed
    find.fill("Renamed freshly")
    expect()(row(page, "newer").locator('.nav-row-label')).to_have_text("Renamed freshly")
    find.fill("Old project")
    expect()(status).to_have_text('No project names contain “Old project”.')
    expect()(row(page, "older")).to_have_count(0)
    # The other order: a complete list requested before a rename lands after the state read that saw the rename.
    follow_route, follow_body = control["held"].pop(0)
    control["hold"] = False
    with page.expect_response(lambda response: is_state(response.request)):
        follow_route.fulfill(content_type="application/json", body=follow_body)
    ui["client"].post('/api/projects/archive-200/update', json={"name": "Archive renamed for the list"}).raise_for_status()
    find.fill("renamed")
    expect()(row(page, "newer").locator('.nav-row-label')).to_have_text("Renamed freshly")
    control["hold_lists"] = True
    with page.expect_request(is_list):
        changed(control, "archive-200")
    ui["client"].post('/api/projects/newer/update', json={"name": "Renamed again"}).raise_for_status()
    with page.expect_response(lambda response: is_state(response.request)):
        refresh(page)
    expect()(row(page, "newer").locator('.nav-row-label')).to_have_text("Renamed again")
    list_route, list_body = control["held_lists"].pop()
    control["hold_lists"] = False
    list_route.fulfill(content_type="application/json", body=list_body)
    expect()(row(page, "archive-200").locator('.nav-row-label')).to_have_text("Archive renamed for the list")  # applied
    expect()(row(page, "newer").locator('.nav-row-label')).to_have_text("Renamed again")
    find.fill("")
    expect()(row(page, "newer")).to_have_count(1)
    expect()(row(page, "older")).to_have_count(0)


def test_reads_in_one_millisecond_keep_their_request_order(pins_ui):
    """With Date.now() standing still, the reads and a confirmed pin are told apart by their request order alone: a
    state read after a list read speaks for the names it carries, that older list cannot take them back, and an
    older list cannot undo a confirmed pin."""
    ui, page, root, control = pins_ui, pins_ui["page"], pins_ui["root"], pins_ui["control"]
    label = lambda pid: row(page, pid).locator('.nav-row-label')  # noqa: E731
    seed_archive(root)
    refresh(page)
    page.locator('.nav-find-input').fill("no such name")
    page.locator('.nav-projects-more', has_text="Show all projects").click()
    expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(499)
    page.clock.set_fixed_time(page.evaluate("Date.now()"))  # every read from here starts in the same millisecond
    ui["client"].post('/api/projects/newer/update', json={"name": "Name one"}).raise_for_status()
    with page.expect_response(lambda response: is_list(response.request)):
        changed(control, "newer")
    expect()(label("newer")).to_have_text("Name one")
    # A list read waits; a state read requested after it carries a newer name; the list then lands, older.
    control["hold_lists"] = True
    ui["client"].post('/api/projects/archive-300/update', json={"name": "Archive renamed for the list"}).raise_for_status()
    with page.expect_request(is_list):
        changed(control, "archive-300")
    ui["client"].post('/api/projects/newer/update', json={"name": "Name two"}).raise_for_status()
    with page.expect_response(lambda response: is_state(response.request)):
        refresh(page)
    expect()(label("newer")).to_have_text("Name two")
    list_route, list_body = control["held_lists"].pop()
    list_route.fulfill(content_type="application/json", body=list_body)
    expect()(label("archive-300")).to_have_text("Archive renamed for the list")  # that list is applied
    expect()(label("newer")).to_have_text("Name two")
    # A list read requested before a confirmed pin lands after it: the pin stands.
    ui["client"].post('/api/projects/archive-301/update', json={"name": "Archive renamed before the pin"}).raise_for_status()
    with page.expect_request(is_list):
        changed(control, "archive-301")
    # Use the real keyboard menu: a frozen wall clock deliberately cannot expire a touch lift's click guard.
    target = row(page, "archive-000").locator('.nav-project-row')
    target.focus()
    target.press("Shift+F10")
    pin = page.locator('body > .project-row-menu:not([data-leaving]) [data-prm="pin"]')
    expect()(pin).to_have_text("Pin")
    pin.press("Enter")
    expect()(row(page, "archive-000")).to_have_attribute("data-pinned", "")
    list_route, list_body = control["held_lists"].pop()
    control["hold_lists"] = False
    list_route.fulfill(content_type="application/json", body=list_body)
    expect()(label("archive-301")).to_have_text("Archive renamed before the pin")  # that list is applied
    expect()(row(page, "archive-000")).to_have_attribute("data-pinned", "")
    ui["client"].post('/api/projects/newer/update', json={"name": "Name three"}).raise_for_status()
    with page.expect_response(lambda response: is_list(response.request)):
        changed(control, "newer")
    expect()(label("newer")).to_have_text("Name three")


def test_browse_keeps_the_row_in_view_when_rooms_above_it_come_or_go_and_says_a_failed_read(pins_ui):
    page, root, control = pins_ui["page"], pins_ui["root"], pins_ui["control"]
    seed_archive(root)
    refresh(page)
    page.locator('#nav-projects .nav-find-input').fill("no such name")
    page.locator('#nav-projects-list > .nav-projects-more', has_text="Show all projects").click()
    items = page.locator('#nav-projects-list > .nav-project-item')
    expect()(items).to_have_count(499)
    page.locator('#nav-projects-list').evaluate("list => { list.scrollTop = list.scrollHeight / 2; }")
    first_visible = """() => {
        const list = document.getElementById('nav-projects-list'), top = list.getBoundingClientRect().top;
        const item = [...list.children].find((el) => el.dataset.navProjectId && el.getBoundingClientRect().bottom > top);
        return [item.dataset.navProjectId, item.getBoundingClientRect().top];
    }"""
    pid, y = page.evaluate(first_visible)
    # A room above the view is deleted; a new room arrives with the next state read and joins after the leading pin.
    above = shown_ids(page)[5]
    registry.begin_project_deletion(root, above)
    registry.complete_project_deletion(root, above)
    with page.expect_response(lambda response: 'all=1' in response.url):
        control["sockets"][-1].send(json.dumps({"type": "projects_changed", "project_id": above}))
    expect()(row(page, above)).to_have_count(0)
    registry.create_project(root, "brand-new-room", name="Brand new room", origin="owner_ui")
    refresh(page)
    expect()(row(page, "brand-new-room")).to_have_count(1)
    assert shown_ids(page).index("brand-new-room") == 1
    current, top = page.evaluate(first_visible)
    assert current == pid and top == pytest.approx(y, abs=1)
    # A failed read again keeps the rows and says so, with the time of the list shown and a way to try again.
    control["fail_list"] = True
    with page.expect_response(lambda response: 'all=1' in response.url):
        control["sockets"][-1].send(json.dumps({"type": "projects_changed", "project_id": "newer"}))
    note = page.locator('#nav-projects-list > .nav-projects-note')
    expect()(note).to_contain_text("Could not read all projects again: Synthetic list outage. Older rows are as read at")
    expect()(items).to_have_count(499)
    control["fail_list"] = False
    page.locator('#nav-projects-list > .nav-projects-more', has_text="Try again").click()
    expect()(note).to_have_count(0)
    expect()(items).to_have_count(499)


def test_slow_complete_list_fails_explicitly_and_retry_keeps_the_feed(pins_ui):
    page, root, control = pins_ui["page"], pins_ui["root"], pins_ui["control"]
    seed_archive(root)
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(51)
    control["hold_lists"] = True
    page.locator('.nav-find-input').fill("archive 000")
    until(page, lambda: len(control["held_lists"]) == 1)
    page.clock.fast_forward(30_001)
    note = page.locator('#nav-projects-list > .nav-projects-note')
    expect()(note).to_contain_text("Project list did not arrive within 30 seconds")
    expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(0)
    control["hold_lists"] = False
    page.locator('#nav-projects-list > .nav-projects-more', has_text="Try again").click()
    expect()(row(page, "archive-000")).to_be_visible()
    page.locator('.nav-find-input').fill("")
    expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(51)


def test_unknown_listable_count_uses_reserved_room_fallback(pins_ui):
    page, root, control = pins_ui["page"], pins_ui["root"], pins_ui["control"]
    seed_archive(root)
    control["null_project_count"] = True
    refresh(page)
    expect()(page.locator('#nav-projects-list > .nav-projects-more')).to_have_text("Show more")


def test_open_room_closes_when_registry_deletes_it_on_projects_changed(pins_ui):
    page, root, control = pins_ui["page"], pins_ui["root"], pins_ui["control"]
    seed_archive(root)
    refresh(page)
    row(page, "newer").locator('.nav-project-row').click()
    expect()(page.locator('#project-panel-title')).to_have_text("Recent project")
    reads = []
    page.on("request", lambda request: reads.append(request.url))
    registry.begin_project_deletion(root, "newer")
    registry.complete_project_deletion(root, "newer")
    control["sockets"][-1].send(json.dumps({"type": "projects_changed", "project_id": "newer"}))
    expect()(page.locator('#project-panel')).to_be_hidden()
    assert reads_of(reads, "/api/projects", id="newer")


def test_show_more_reaches_every_room_once_read_and_keeps_its_rows(pins_ui, tmp_path):
    ui, page, root, control = pins_ui, pins_ui["page"], pins_ui["root"], pins_ui["control"]
    evidence = producer_evidence_dir(tmp_path, "project-find", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    eligible = seed_archive(root)
    assert len(eligible) == 499
    reads = []
    page.on("request", lambda request: reads.append(request.url))
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(51)  # the pin and the 50 most recent others
    assert reads_of(reads, "/api/projects", all="1") == []  # the complete list is read only when asked for
    more = page.locator('#nav-projects-list > .nav-projects-more')
    expect()(more).to_have_text("Show more")
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"resting-{ui['tag']}.png"), animations="disabled")
    shown = 51
    while more.count():
        more.click()
        shown = min(shown + 50, len(eligible))
        expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(shown)
    ids = shown_ids(page)
    assert sorted(ids) == sorted(eligible) and len(set(ids)) == len(ids)  # every id, once, the oldest included
    assert "archive-000" in ids and "archive-004" not in ids  # a tombstone is never listed
    assert row(page, "archive-003").locator('.nav-project-row').is_disabled()  # deleting: shown, not openable
    assert len(reads_of(reads, "/api/projects", all="1")) == 1
    # State polls go on; the list is not read again on a timer, and nothing is read or acknowledged by browsing.
    for _ in range(2):
        with page.expect_response(lambda response: urlparse(response.url).path == '/api/state'):
            refresh(page)
    assert len(reads_of(reads, "/api/projects", all="1")) == 1
    assert not reads_of(reads, "/api/chat/history", chat_id="40200") and no_read_acknowledged(ui)
    expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(499)  # a snapshot keeps the browse
    # A registry change announced on the socket re-reads the list; a row takes its new name in its own node.
    page.evaluate("window.savedRow = document.querySelector('[data-nav-project-id=archive-200]')")
    client = ui["client"]
    assert client.post("/api/projects/archive-200/update", json={"name": "Archive 200 · renamed elsewhere"}).status_code == 200
    control["sockets"][-1].send(json.dumps({"type": "projects_changed", "project_id": "archive-200", "chat_id": 40200}))
    expect()(row(page, "archive-200").locator('.nav-row-label')).to_have_text("Archive 200 · renamed elsewhere")
    assert page.evaluate("savedRow.isConnected && savedRow === document.querySelector('[data-nav-project-id=archive-200]')")
    assert len(reads_of(reads, "/api/projects", all="1")) == 2
    # The browse keeps its scroll through a snapshot that repaints the rows (a feed row turns unread).
    page.locator('#nav-projects-list').evaluate("list => { list.scrollTop = 2400; }")
    before = page.locator('#nav-projects-list').evaluate("list => list.scrollTop")
    control["unread"].add("older")
    refresh(page)
    expect()(row(page, "older").locator('.nav-project-row')).to_have_attribute("data-unread", "")
    assert abs(page.locator('#nav-projects-list').evaluate("list => list.scrollTop") - before) <= 1
    expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(499)
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"browse-{ui['tag']}.png"), animations="disabled")
    # The owner's own collapse ends the browse: the list rests on the pin and the fifty newest again.
    page.locator('#nav-projects-toggle').click()
    page.locator('#nav-projects-toggle').click()
    expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(51)


def test_find_matches_every_name_in_any_case_or_accent_and_never_claims_a_room_is_missing(pins_ui, tmp_path):
    ui, page, root, control = pins_ui, pins_ui["page"], pins_ui["root"], pins_ui["control"]
    evidence = producer_evidence_dir(tmp_path, "project-find", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    seed_archive(root)
    reads = []
    page.on("request", lambda request: reads.append(request.url))
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(51)
    find = page.locator('#nav-projects .nav-find-input')
    note = page.locator('#nav-projects-list > .nav-projects-note')
    # An unreadable list is said as such: never "no match", with a way to try again.
    control["fail_list"] = True
    find.fill("archive 000")
    expect()(note).to_contain_text("Could not search all projects")
    expect()(page.locator('#nav-projects-list > .nav-projects-more', has_text="Try again")).to_be_visible()
    assert "No project names contain" not in page.locator('#nav-projects-list').inner_text()
    control["fail_list"] = False
    page.locator('#nav-projects-list > .nav-projects-more', has_text="Try again").click()
    # The oldest room, beyond the feed and the plain list's 200, by a fragment of its name.
    shown = page.locator('#nav-projects-list > .nav-project-item')
    expect()(shown).to_have_count(1)
    expect()(row(page, "archive-000")).to_be_visible()
    # Any case or accent, the words in any order; closed rooms are found, deleting ones are shown and not openable.
    for query, expected in (("café плановые", "archive-001"), ("РАБОТЫ unicode cafe", "archive-001"),
                            ("ARCHIVE 010", "archive-010"), ("archive 003", "archive-003")):
        find.fill(query)
        expect()(shown).to_have_count(1)
        expect()(row(page, expected)).to_be_visible()
    assert row(page, "archive-003").locator('.nav-project-row').is_disabled()
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"found-{ui['tag']}.png"), animations="disabled")
    # Many matches: the pin leads, then a page of the rest and Show N more.
    find.fill("archive")
    expect()(page.locator('#nav-projects-list > .nav-projects-more')).to_have_text("Show 50 more")
    expect()(shown).to_have_count(51)
    assert shown_ids(page)[0] == "archive-100"
    # A snapshot while finding keeps the query, the results and the caret where the owner left them.
    find.focus()
    find.evaluate("input => input.setSelectionRange(2, 2)")
    control["unread"].add("newer")
    refresh(page)
    expect()(page.locator('#nav-projects-count')).to_have_text("2")  # the snapshot has landed (a feed fact)
    assert find.evaluate("input => [document.activeElement === input, input.value, input.selectionStart]") == [True, "archive", 2]
    expect()(shown).to_have_count(51)
    # No match: said of the names, with Clear search and Show all projects; a tombstone is never found.
    find.fill("archive 004")
    expect()(note).to_have_text("No project names contain “archive 004”.")
    expect()(shown).to_have_count(0)
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"no-match-{ui['tag']}.png"), animations="disabled")
    page.locator('#nav-projects-list > .nav-projects-more', has_text="Show all projects").click()
    expect()(find).to_have_value("")
    expect()(shown).to_have_count(499)
    # Keys: ↓ from the field to the first result, ↑ back, Escape clears and keeps the drawer or the room.
    find.fill("archive 000")
    expect()(shown).to_have_count(1)
    find.press("ArrowDown")
    expect()(row(page, "archive-000").locator('.nav-project-row')).to_be_focused()
    page.keyboard.press("ArrowUp")
    expect()(find).to_be_focused()
    find.press("Escape")
    expect()(find).to_have_value("")
    if ui["width"] < 720:
        expect()(page.locator('#primary-sidebar.open')).to_have_count(1)
    # Finding read no room and acknowledged nothing; Enter opens the first result by its id, in its own room.
    assert no_read_acknowledged(ui) and not [u for u in reads_of(reads, "/api/chat/history")
                                             if parse_qs(urlparse(u).query).get("chat_id", ["1"])[0].startswith("40")]
    find.fill("archive 000")
    expect()(shown).to_have_count(1)
    find.press("Enter")
    expect()(page.locator('#project-panel-title')).to_have_text("Archive 000 · garden irrigation")
    assert len(reads_of(reads, "/api/projects", all="1")) == 2 and reads_of(reads, "/api/projects", id="archive-000")
    until(page, lambda: bool(reads_of(reads, "/api/chat/history", chat_id="40000")))


def test_reference_beyond_both_caps_opens_its_own_room_and_a_superseded_or_failed_one_keeps_the_view(pins_ui):
    ui, page, root, control = pins_ui, pins_ui["page"], pins_ui["root"], pins_ui["control"]
    seed_archive(root)
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(51)
    reads = []
    page.on("request", lambda request: reads.append(request.url))
    deep = registry.get_project(root, "archive-000")
    assert row(page, deep["id"]).count() == 0  # beyond the feed (50) and the plain list (200)
    title = page.locator('#project-panel-title')
    click_reference(page, deep)
    expect()(title).to_have_text(deep["name"])
    settle(page)
    expect()(page.locator(f'[id="panel-pchat-{deep["id"]}"]')).to_have_count(1)
    until(page, lambda: bool(reads_of(reads, "/api/chat/history", chat_id=str(deep["chat_id"]))))
    assert reads_of(reads, "/api/projects", id=deep["id"])
    # The composer of that room sends to it: its own chat id and Project, never Main.
    page.locator(f'[id="pchat-{deep["id"]}-input"]').fill("Neutral check for the old room")
    page.locator(f'[id="pchat-{deep["id"]}-send"]').dispatch_event("click")
    def frames():
        parsed = []
        for message in control["sent"]:
            try:
                frame = json.loads(message)
            except (TypeError, ValueError):
                continue
            if isinstance(frame, dict) and frame.get("type") == "chat":
                parsed.append(frame)
        return parsed
    until(page, lambda: bool(frames()))
    assert [(f.get("chat_id"), f.get("project_id")) for f in frames()] == [(deep["chat_id"], deep["id"])]
    # A slow lookup is superseded by the next press: A's late answer changes nothing, B opens.
    a, b = registry.get_project(root, "archive-001"), registry.get_project(root, "archive-002")
    control["hold_lookups"] = True
    click_reference(page, a)
    until(page, lambda: len(control["held_lookups"]) == 1)
    click_reference(page, b)
    until(page, lambda: len(control["held_lookups"]) == 2)
    expect()(title).to_have_text(deep["name"])  # the room on screen stays while both are read
    ui["forward_get"](control["held_lookups"].pop(0))
    page.evaluate("0")
    expect()(title).to_have_text(deep["name"])
    ui["forward_get"](control["held_lookups"].pop(0))
    expect()(title).to_have_text(b["name"])
    settle(page)
    assert not reads_of(reads, "/api/chat/history", chat_id=str(a["chat_id"]))
    # A close or another page while a lookup waits: the late answer opens nothing.
    click_reference(page, a)
    until(page, lambda: len(control["held_lookups"]) == 1)
    page.locator('#project-panel-close').click()
    expect()(page.locator('#project-panel')).to_be_hidden()
    ui["forward_get"](control["held_lookups"].pop(0))
    page.evaluate("0")
    settle(page)
    expect()(page.locator('#project-panel')).to_be_hidden()
    assert not reads_of(reads, "/api/chat/history", chat_id=str(a["chat_id"]))
    control["hold_lookups"] = False
    # Deleting, tombstoned and unreadable answers keep the room on screen and say why.
    open_drawer(page, ui["width"])
    row(page, "newer").locator('.nav-project-row').click()
    expect()(title).to_have_text("Recent project")
    assert not reads_of(reads, "/api/projects", id="newer")  # a row the feed carries opens as it shows, at once
    for pid, words in (("archive-003", "is being deleted"), ("archive-004", "is no longer available")):
        click_reference(page, registry.get_reserved_project(root, pid))
        expect()(page.locator('#toast-stack')).to_contain_text(words)
        expect()(title).to_have_text("Recent project")
    control["fail_lookup"] = True
    click_reference(page, a)
    expect()(page.locator('#toast-stack')).to_contain_text("Synthetic lookup outage")
    expect()(title).to_have_text("Recent project")
    control["fail_lookup"] = False
    assert not reads_of(reads, "/api/chat/history", chat_id="40003") and not reads_of(reads, "/api/chat/history", chat_id="40004")


@pytest.mark.parametrize("fate,words", [("deleted", "is no longer available"), ("deleting", "is being deleted")])
def test_a_reference_keeps_the_room_when_a_newer_list_took_its_room_away_from_an_old_feed(pins_ui, fate, words):
    """The compact feed the page holds still calls the room active; a complete list requested after the change no
    longer does. A reference (its id and name only) reads the room by its id, so the room on screen, its draft and its
    history reads stay."""
    page, root, control = pins_ui["page"], pins_ui["root"], pins_ui["control"]
    title = page.locator('#project-panel-title')
    seed_archive(root)
    refresh(page)
    page.locator('.nav-find-input').fill("project")
    expect()(page.locator('.nav-find-status')).to_have_text("2 projects found")
    row(page, "newer").locator('.nav-project-row').click()
    expect()(title).to_have_text("Recent project")
    settle(page)
    composer = page.locator('[id="pchat-newer-input"]')
    composer.fill("A draft that stays")
    reads = []
    page.on("request", lambda request: reads.append(request.url))
    control["hold"] = True
    with page.expect_request(is_state):
        refresh(page)  # requested before the change and never answered: the feed stays the old one
    registry.begin_project_deletion(root, "older")
    if fate == "deleted":
        registry.complete_project_deletion(root, "older")
    with page.expect_response(lambda response: is_list(response.request)):
        changed(control, "older")
    if fate == "deleted":
        expect()(row(page, "older")).to_have_count(0)  # that list is applied
    else:
        expect()(row(page, "older").locator('.nav-project-row')).to_be_disabled()
    room_chat, gone_chat = (str(registry.get_reserved_project(root, pid)["chat_id"]) for pid in ("newer", "older"))
    room_reads = len(reads_of(reads, "/api/chat/history", chat_id=room_chat))
    click_reference(page, {"id": "older", "name": LONG})
    expect()(page.locator('#toast-stack')).to_contain_text(words)
    assert reads_of(reads, "/api/projects", id="older")
    expect()(title).to_have_text("Recent project")
    expect()(composer).to_have_value("A draft that stays")
    expect()(page.locator('[id="panel-pchat-older"]')).to_have_count(0)
    assert not reads_of(reads, "/api/chat/history", chat_id=gone_chat)
    assert len(reads_of(reads, "/api/chat/history", chat_id=room_chat)) == room_reads


def test_a_reference_opens_what_its_read_answers_over_an_older_feed_row_and_a_closed_room_too(pins_ui):
    ui, page, root, control = pins_ui, pins_ui["page"], pins_ui["root"], pins_ui["control"]
    title = page.locator('#project-panel-title')
    seed_archive(root)
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(51)
    reads = []
    page.on("request", lambda request: reads.append(request.url))
    control["hold"] = True
    with page.expect_request(is_state):
        refresh(page)  # the feed stays the one that still has the old name
    ui["client"].post('/api/projects/older/update', json={"name": "Renamed after that read"}).raise_for_status()
    click_reference(page, {"id": "older", "name": LONG})
    expect()(title).to_have_text("Renamed after that read")
    assert reads_of(reads, "/api/projects", id="older")
    settle(page)
    until(page, lambda: bool(reads_of(reads, "/api/chat/history", chat_id=str(registry.get_project(root, "older")["chat_id"]))))
    closed = registry.get_project(root, "archive-010")  # closed by the owner (seed_archive), beyond the feed
    click_reference(page, closed)
    expect()(title).to_have_text(closed["name"])
    expect()(page.locator('[id="panel-pchat-archive-010"]')).to_have_count(1)
    until(page, lambda: bool(reads_of(reads, "/api/chat/history", chat_id=str(closed["chat_id"]))))


def test_an_unread_room_beyond_the_feed_counts_while_a_find_shows_it_and_another_clients_read_clears_it(pins_ui):
    """Main and every recent room are read and one old room holds a stored message past the shared cursor: nothing in
    the compact feed says so. While a find shows the room, the Projects number and ☰ count it. Another client reads the
    room: the next ordinary state read asks after the shared cursors, since a row shown here is unread, and clears its
    mark, the number and ☰ while the find still shows it. No read here was acknowledged."""
    ui, page, root, control = pins_ui, pins_ui["page"], pins_ui["root"], pins_ui["control"]
    seed_archive(root)
    # The stored row is older than the room's own activity, so the room stays far beyond the feed.
    dialogue_store.append(root, "chat", {"ts": "2025-01-02T00:00:00Z", "direction": "out", "chat_id": 40200,
                                         "text": "An old reply nobody here has read"})
    revision = advance_room_revision(root, project_id="archive-200")["visible_revision"]
    assert revision > 0
    control["shared_cursors"] = True
    posts = []
    page.on("request", lambda request: posts.append(request.post_data or "")
            if request.method == "POST" and urlparse(request.url).path == "/api/ui/preferences" else None)
    count, nav = page.locator('#nav-projects-count'), page.locator('#page-chat [data-mobile-nav-toggle]')
    find, mark = page.locator('.nav-find-input'), row(page, "archive-200").locator('.nav-project-row')
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(51)
    assert "archive-200" not in shown_ids(page)
    expect()(count).to_have_text("")
    expect()(nav).to_have_attribute("aria-label", "Open navigation")
    with page.expect_response(is_cursors):  # a list that shows an unread room asks after its cursor at once
        find.fill("archive 200")
    expect()(mark).to_have_attribute("data-unread", "")
    expect()(count).to_have_text(str(revision))
    expect()(nav).to_have_attribute("aria-label", "Open navigation · new messages")
    assert "in Projects" in page.locator('#nav-projects-toggle').get_attribute("title")
    assert page.locator('#nav-projects-list [data-unread]').count() == 1  # the old room alone
    ui["client"].post('/api/ui/preferences', json={"project_seen_revision": {"archive-200": revision}}).raise_for_status()
    with page.expect_response(is_cursors):
        refresh(page)  # an ordinary state read: Main and the feed show nothing unread, the old room shown here does
    expect()(mark).not_to_have_attribute("data-unread", "")
    expect()(count).to_have_text("")
    expect()(nav).to_have_attribute("aria-label", "Open navigation")
    assert "in Projects" not in (page.locator('#nav-projects-toggle').get_attribute("title") or "")
    find.fill("")
    expect()(row(page, "archive-200")).to_have_count(0)
    expect()(count).to_have_text("")
    assert not [body for body in posts if "project_seen_revision" in body]


def test_an_old_unread_room_no_find_shows_counts_nowhere_and_no_complete_list_is_read_for_it(pins_ui):
    """The number counts what this sidebar shows: the feed's rooms and those an open find or browse lists, never one a
    complete list only holds. A find for another old room reads a list that holds an unread room no row shows: the
    number and ☰ stay as the feed says. While the find is open a registry change reads the list again. A find that
    shows the room ends by a collapse or by clearing it: the number drops at once, and then neither a registry change
    nor a reopened socket reads the list for the unread it holds. The next find or browse reads it again, as does a
    socket reopened while that browse is open. Nothing is acknowledged."""
    page, root, control = pins_ui["page"], pins_ui["root"], pins_ui["control"]
    seed_archive(root)
    dialogue_store.append(root, "chat", {"ts": "2025-01-03T00:00:00Z", "direction": "out", "chat_id": 40200,
                                         "text": "An old reply this sidebar does not show"})
    revision = advance_room_revision(root, project_id="archive-200")["visible_revision"]
    control["shared_cursors"] = True
    reads, posts = [], []
    page.on("request", lambda request: reads.append(request.url))
    page.on("request", lambda request: posts.append(request.post_data or "")
            if request.method == "POST" and urlparse(request.url).path == "/api/ui/preferences" else None)
    lists = lambda: len(reads_of(reads, "/api/projects", all="1"))  # noqa: E731
    count, nav = page.locator('#nav-projects-count'), page.locator('#page-chat [data-mobile-nav-toggle]')
    find, status = page.locator('.nav-find-input'), page.locator('.nav-find-status')
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(51)
    # A find that does not match the unread room: its list holds that room, which no row here shows.
    find.fill("archive 201")
    expect()(status).to_have_text("1 project found")  # that list is applied, and the number said after it
    expect()(row(page, "archive-201")).to_be_visible()
    expect()(count).to_have_text("")
    expect()(nav).to_have_attribute("aria-label", "Open navigation")
    with page.expect_response(is_list):  # while a find is open, a registry change reads its list again
        changed(control, "archive-201")
    find.fill("archive 200")
    expect()(row(page, "archive-200").locator('.nav-project-row')).to_have_attribute("data-unread", "")
    expect()(count).to_have_text(str(revision))
    expect()(nav).to_have_attribute("aria-label", "Open navigation · new messages")
    toggle = page.locator('#nav-projects-toggle')
    toggle.click()  # a collapse ends the find: the number and ☰ say the feed's rooms again, at once
    expect()(count).to_have_text("")
    expect()(nav).to_have_attribute("aria-label", "Open navigation")
    toggle.click()
    expect()(find).to_have_value("")
    with page.expect_response(is_list):  # the next find reads the list as it starts
        find.fill("archive 200")
    expect()(count).to_have_text(str(revision))
    find.fill("")  # clearing it ends it the same way
    expect()(row(page, "archive-200")).to_have_count(0)
    expect()(count).to_have_text("")
    expect()(nav).to_have_attribute("aria-label", "Open navigation")
    before = lists()
    with page.expect_response(lambda response: is_state(response.request)):
        changed(control, "archive-200")
    with page.expect_response(lambda response: is_state(response.request)):
        reopen(page, control)
    settle(page)
    assert lists() == before, "a list was read again for unread no row shows"
    expect()(count).to_have_text("")
    more = page.locator('#nav-projects-list > .nav-projects-more')
    expect()(more).to_have_text("Show 50 more")
    with page.expect_request(is_list):  # the next browse reads the list as it starts
        more.click()
    expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(101)
    with page.expect_request(is_list):  # and a socket reopened while it is open reads it again
        reopen(page, control)
    expect()(row(page, "archive-200")).to_have_count(0)
    expect()(count).to_have_text("")
    assert not [body for body in posts if "project_seen_revision" in body]


def test_a_no_match_is_not_repeated_while_its_list_is_read_again(pins_ui):
    """A registry change restarts the complete read under a shown no-match: the answer that may be out of date is no
    longer said, both ways on stay with the keyboard, a failed read says so, and a failure stands with its Try again
    until the next read answers."""
    ui, page, root, control = pins_ui, pins_ui["page"], pins_ui["root"], pins_ui["control"]
    seed_archive(root)
    refresh(page)
    status, note = page.locator('.nav-find-status'), page.locator('#nav-projects-list > .nav-projects-note')
    page.locator('.nav-find-input').fill("Later name")
    expect()(note).to_have_text('No project names contain “Later name”.')
    command = page.locator('.nav-projects-more', has_text="Show all projects")
    command.focus()
    page.evaluate("window.savedCommand = document.activeElement")
    ui["client"].post('/api/projects/archive-200/update', json={"name": "Later name"}).raise_for_status()
    control["hold_lists"] = True
    with page.expect_request(is_list):
        changed(control, "archive-200")
    expect()(note).to_have_text("Searching all projects…")
    expect()(status).to_have_text("Searching all projects…")
    expect()(command).to_be_focused()
    assert page.evaluate("savedCommand === document.activeElement && savedCommand.isConnected")
    until(page, lambda: len(control["held_lists"]) == 1)
    route, _ = control["held_lists"].pop()
    route.fulfill(status=503, content_type="application/json", body='{"error":"Synthetic list outage"}')
    expect()(note).to_have_text("Could not search all projects again: Synthetic list outage.")
    retry = page.locator('.nav-projects-more', has_text="Try again")
    retry.focus()
    with page.expect_request(is_list):
        changed(control, "archive-200")
    expect()(note).to_have_text("Could not search all projects again: Synthetic list outage.")
    expect()(retry).to_be_focused()
    until(page, lambda: len(control["held_lists"]) == 1)
    route, body = control["held_lists"].pop()
    control["hold_lists"] = False
    route.fulfill(content_type="application/json", body=body)
    expect()(row(page, "archive-200")).to_be_visible()
    expect()(status).to_have_text("1 project found")
    expect()(note).to_have_count(0)


@pytest.mark.parametrize("press", ["#nav-projects-count", "#nav-projects-toggle"], ids=["header-facts", "toggle"])
def test_a_pointer_collapse_of_a_browse_leaves_the_keyboard_on_the_toggle(pins_ui, press):
    """A deep row has the keyboard when a pointer presses the header. A press need not focus the toggle (WebKit), the
    number beside it is no control at all, and the row goes with the browse: the toggle takes the keyboard."""
    page, root, control = pins_ui["page"], pins_ui["root"], pins_ui["control"]
    seed_archive(root)
    control["unread"].add("newer")  # the header's number has a body to press
    refresh(page)
    expect()(page.locator('#nav-projects-count')).to_have_text("2")
    page.locator('.nav-find-input').fill("no such name")
    page.locator('.nav-projects-more', has_text="Show all projects").click()
    expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(499)
    row(page, "archive-000").locator('.nav-project-row').focus()
    page.locator(press).click()
    expect()(page.locator('#nav-projects-list > .nav-project-item:not([hidden])')).to_have_count(1)  # the pin
    expect()(page.locator('#nav-projects-toggle')).to_be_focused()


def test_a_collapse_leaves_the_keyboard_on_a_pinned_row_and_takes_it_from_any_other(pins_ui):
    """A press that moves no focus, as an assistive technology's does: a pinned row stays where it is, with the
    keyboard; a row or a command that goes with the browse hands it to the toggle first."""
    page, root = pins_ui["page"], pins_ui["root"]
    toggle = page.locator('#nav-projects-toggle')
    press = "button => button.click()"
    seed_archive(root)
    refresh(page)

    def browse():
        page.locator('.nav-find-input').fill("no such name")
        page.locator('.nav-projects-more', has_text="Show all projects").click()
        expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(499)

    browse()
    pinned = row(page, "archive-100").locator('.nav-project-row')
    pinned.focus()
    toggle.evaluate(press)
    expect()(row(page, "archive-000")).to_have_count(0)
    expect()(pinned).to_be_visible()
    expect()(pinned).to_be_focused()
    toggle.evaluate(press)
    browse()
    row(page, "archive-000").locator('.nav-project-row').focus()
    toggle.evaluate(press)
    expect()(row(page, "archive-000")).to_have_count(0)
    expect()(toggle).to_be_focused()
    assert page.evaluate("document.activeElement.id") == "nav-projects-toggle"


def test_find_field_is_a_row_high_capsule_on_the_rows_columns(pins_ui, tmp_path):
    ui, page = pins_ui, pins_ui["page"]
    evidence = producer_evidence_dir(tmp_path, "project-find", env="OUROBOROS_BROWSER_EVIDENCE_OUT")
    geometry = page.evaluate("""() => {
        const box = (el) => el.getBoundingClientRect();
        const find = document.querySelector('#nav-projects .nav-projects-find');
        const input = find.querySelector('.nav-find-input');
        const row = document.querySelector('#nav-projects-list .nav-project-row');
        const style = getComputedStyle(find);
        return {
            height: box(find).height, rowHeight: box(row).height, radius: parseFloat(style.borderTopLeftRadius),
            glyphX: box(find.querySelector('.nav-find-glyph')).left,
            iconX: box(document.querySelector('#nav-projects-toggle .nav-row-icon')).left,
            textX: box(input).left + parseFloat(getComputedStyle(input).paddingLeft),
            labelX: box(row.querySelector('.nav-row-label')).left,
            above: box(find).top - box(document.querySelector('.nav-projects-header')).bottom,
            below: box(row).top - box(find).bottom,
            shadow: style.boxShadow, inputBorder: getComputedStyle(input).borderTopWidth,
            fontSize: parseFloat(getComputedStyle(input).fontSize), placeholder: input.placeholder,
            clearHidden: find.querySelector('.nav-find-clear').hidden,
        };
    }""")
    touch = coarse(page)
    assert geometry["height"] == pytest.approx(44 if touch else 32, abs=0.5), geometry
    assert geometry["height"] == pytest.approx(geometry["rowHeight"], abs=0.5), geometry
    assert geometry["radius"] >= geometry["height"] / 2, geometry  # a capsule, as every one-line control
    assert geometry["glyphX"] == pytest.approx(geometry["iconX"], abs=1), geometry
    assert geometry["textX"] == pytest.approx(geometry["labelX"], abs=1), geometry
    assert geometry["shadow"] != "none" and geometry["inputBorder"] == "0px", geometry
    assert geometry["fontSize"] >= (16 if touch else 14), geometry
    assert 0 < geometry["above"] < geometry["below"], geometry
    assert geometry["placeholder"] == "Find a project" and geometry["clearHidden"] is True
    find = page.locator('#nav-projects .nav-find-input')
    find.fill("Recent")
    expect()(page.locator('#nav-projects .nav-find-clear')).to_be_visible()
    find.focus()
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f"field-focused-{ui['tag']}.png"), animations="disabled")
    page.locator('#nav-projects .nav-find-clear').click()
    expect()(find).to_have_value("")
    expect()(find).to_be_focused()
    expect()(page.locator('#nav-projects .nav-find-clear')).to_be_hidden()
    # Collapsed, the field hides with the rest and a query ends; reopened, the list is as it rested.
    find.fill("Recent")
    page.locator('#nav-projects-toggle').click()
    expect()(page.locator('#nav-projects .nav-projects-find')).to_be_hidden()
    page.locator('#nav-projects-toggle').click()
    expect()(find).to_have_value("")
    assert order(page) == ["newer", "older"]


@pytest.mark.parametrize("case", ["fresh-name", "failed-refresh"])
def test_complete_list_refresh_is_not_hidden_by_a_stale_feed(pins_ui, case, tmp_path):
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    seed_archive(ui["root"])
    refresh(page)
    find = page.locator('.nav-find-input')
    find.fill("Recent project")
    expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(1)
    expect()(page.locator('.nav-find-status')).to_have_text("1 project found")  # The first full read settled.
    control["hold"] = True  # Keep the compact feed older than the following full-list read.
    if case == "fresh-name":
        ui["client"].post('/api/projects/newer/update', json={"name": "Renamed freshly"}).raise_for_status()
    else:
        control["fail_list"] = True
    with page.expect_response(lambda response: 'all=1' in response.url):
        control["sockets"][-1].send(json.dumps({"type": "projects_changed", "project_id": "newer"}))
    find.fill("Renamed freshly" if case == "fresh-name" else "not in the old list")
    page.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    evidence = producer_evidence_dir(tmp_path, 'project-freshness', env='OUROBOROS_BROWSER_EVIDENCE_OUT')
    page.locator('#primary-sidebar').screenshot(path=str(evidence / f'{case}-{ui["tag"]}.png'))
    if case == "fresh-name":
        expect()(row(page, "newer").locator('.nav-row-label')).to_have_text("Renamed freshly")
    else:
        text = page.locator('#nav-projects-list').inner_text()
        assert "Could not" in text and "No project names contain" not in text, text
        expect()(page.locator('.nav-projects-more', has_text="Try again")).to_be_visible()


def test_the_list_follows_the_newest_message_and_a_closure_or_unread_never_lifts_a_room(pins_ui):
    """The screenshot's case (owner 2026-11-07, option 1): a cleanup's closure notices leave their rooms where their
    last message put them, unread only as a mark; a pinned room with only such a notice stays under a pin with a newer
    message, read or not."""
    ui, page, root, control = pins_ui, pins_ui["page"], pins_ui["root"], pins_ui["control"]
    for pid in ("jeff", "audit", "redline", "live", "issues"):
        registry.create_project(root, pid, name=pid.title(), origin="owner_ui")
    date_history(root)
    say(root, room_of(root, "issues"), "Fifty issues triaged", ts=LATER(1))
    say(root, room_of(root, "live"), "A fresh result", ts=LATER(2))
    for pid in ("jeff", "audit", "redline", "issues"):  # today's memory cleanup closes them, after every message
        say(root, room_of(root, pid), "Ouroboros closed this project.", ts=LATER(3), direction="system", role="system",
            type="project_closure", system_type="project_closure")
    for pid in ("issues", "live"):
        assert ui["client"].post(f"/api/projects/{pid}/update", json={"pinned": True}).status_code == 200
    control["unread"].update({"jeff", "audit", "redline", "issues"})
    refresh(page)
    expect()(row(page, "jeff").locator('.nav-project-row')).to_have_attribute("data-unread", "")
    ids = shown_ids(page)
    assert ids[:2] == ["live", "issues"]  # pins: the newer message first, the unread closure no lift
    assert ids[2:] == ["redline", "audit", "jeff", "newer", "older"]  # no message since: their creation
    expect()(row(page, "issues").locator('.nav-project-row')).to_have_attribute("data-unread", "")


def test_a_focused_row_a_message_moves_keeps_the_keyboard_and_the_place_being_read_stays(pins_ui):
    ui, page, root = pins_ui, pins_ui["page"], pins_ui["root"]
    seed_archive(root)
    spread(root)
    refresh(page)
    page.locator('.nav-find-input').fill("no such name")
    page.locator('.nav-projects-more', has_text="Show all projects").click()
    expect()(page.locator('#nav-projects-list > .nav-project-item')).to_have_count(499)
    page.mouse.move(ui["width"] - 1, 1)  # No hover hold: the keyboard owns this place.
    row(page, "archive-250").evaluate("el => el.scrollIntoView({block: 'start'})")
    first, y = page.evaluate(FIRST_IN_VIEW)
    target = row(page, "archive-248").locator('.nav-project-row')
    target.focus()
    say(root, 40248, "News for the focused room", ts=LATER(1))
    refresh(page)
    until(page, lambda: shown_ids(page)[1] == "archive-248", rounds=3000)  # after the pin: the newest message
    expect()(target).to_be_focused()
    assert page.evaluate(FIRST_IN_VIEW) == [first, pytest.approx(y, abs=1)]


def test_show_more_keeps_sorting_and_holds_the_place_and_reads_again_when_an_older_room_moves(pins_ui):
    """Owner 2026-11-07, Show more option 1: the browse keeps every row it showed and keeps sorting; the first row in
    view whose place holds keeps the place being read, also when the row in view is the one a message moves; and a
    loaded room beyond the feed that a late message moves among the older rows is placed by the complete list read
    again on the moved revision (Astra), never on a timer."""
    ui, page, root = pins_ui, pins_ui["page"], pins_ui["root"]
    seed_archive(root)
    spread(root)
    reads = []
    page.on("request", lambda request: reads.append(request.url))
    refresh(page)
    items = page.locator('#nav-projects-list > .nav-project-item')
    expect()(items).to_have_count(51)
    page.locator('#nav-projects-list > .nav-projects-more').click()
    expect()(items).to_have_count(101)
    assert shown_ids(page)[:3] == ["archive-100", "archive-497", "archive-496"]
    page.mouse.move(ui["width"] - 1, 1)
    row(page, "archive-430").evaluate("el => el.scrollIntoView({block: 'start'})")
    first, y = page.evaluate(FIRST_IN_VIEW)
    assert first.startswith("archive-4") and first != "archive-420", (first, shown_ids(page)[:4])
    # A room further down gets a message and goes to the top; the place being read stays.
    say(root, 40420, "News further down", ts=LATER(1))
    refresh(page)
    until(page, lambda: shown_ids(page)[1] == "archive-420", rounds=3000)  # behind a complete list being read again
    assert page.evaluate(FIRST_IN_VIEW) == [first, pytest.approx(y, abs=1)]
    # The row holding the place moves itself: the next one keeps the place, the view does not follow it up.
    after = shown_ids(page)[shown_ids(page).index(first) + 1]
    after_y = row(page, after).bounding_box()["y"]
    say(root, 40000 + int(first.split("-")[1]), "News in view", ts=LATER(2))
    refresh(page)
    until(page, lambda: shown_ids(page)[1] == first, rounds=3000)
    assert row(page, after).bounding_box()["y"] == pytest.approx(after_y, abs=1)
    assert len(shown_ids(page)) == 101  # every row the browse showed stays
    # A loaded room's late message, older than the feed's fifty: the browse reads the complete list again for it
    # (as it did for each message above: a moved revision, never progress or a timer).
    lists = len(reads_of(reads, "/api/projects", all="1"))
    say(root, 40410, "A late answer", ts=SPREAD(415.5))
    refresh(page)
    until(page, lambda: len(reads_of(reads, "/api/projects", all="1")) > lists)
    until(page, lambda: shown_ids(page).index("archive-416") < shown_ids(page).index("archive-410")
          < shown_ids(page).index("archive-415"), rounds=5000)  # the complete list's answer, however slow
    lists, sockets = len(reads_of(reads, "/api/projects", all="1")), len(ui["control"]["sockets"])
    for _ in range(2):  # an unchanged revision reads nothing more (a reopened socket reads once, as it always did)
        with page.expect_response(lambda response: urlparse(response.url).path == '/api/state'):
            refresh(page)
    assert len(reads_of(reads, "/api/projects", all="1")) == lists + len(ui["control"]["sockets"]) - sockets
    page.locator('#nav-projects-toggle').click()
    page.locator('#nav-projects-toggle').click()
    expect()(items).to_have_count(51)
    assert shown_ids(page)[:3] == ["archive-100", first, "archive-420"]


@pytest.mark.parametrize("action", ["show-all", "failed-more"])
def test_keyboard_browse_handoff_finishes_and_never_steals_later_composer_focus(pins_ui, action):
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    find = page.locator('.nav-find-input')
    if action == "show-all":
        find.fill("no such name")
        expect()(page.locator('.nav-find-status')).to_have_text('No project names contain “no such name”.')
        command = page.locator('.nav-projects-more', has_text="Show all projects")
        command.focus()
        page.keyboard.press("Enter")
        expect()(find).to_be_focused()  # No extra rows exist: the hand-off still ends.
    else:
        seed_archive(ui["root"])
        refresh(page)
        expect()(page.locator('.nav-project-item')).to_have_count(51)
        control["fail_list"] = True
        page.locator('.nav-projects-more', has_text="Show more").focus()
        page.keyboard.press("Enter")
        expect()(page.locator('.nav-projects-more', has_text="Try again")).to_be_focused()
    if ui["width"] < 720:
        page.locator('#nav-drawer-close').click()
        settle(page)
    composer = page.locator('#page-chat textarea').first
    composer.fill("Keep writing here")
    control["fail_list"] = False
    registry.create_project(ui["root"], "later-room", name="Later room", origin="owner_ui")
    control["sockets"][-1].send(json.dumps({"type": "projects_changed", "project_id": "later-room"}))
    expect()(row(page, "later-room")).to_have_count(1)
    expect()(composer).to_be_focused()
    expect()(composer).to_have_value("Keep writing here")


def test_a_delayed_browse_read_does_not_reclaim_keyboard_after_owner_moves_it(pins_ui):
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    seed_archive(ui["root"])
    refresh(page)
    expect()(page.locator('.nav-project-item')).to_have_count(51)
    control["hold_lists"] = True
    page.locator('.nav-projects-more', has_text="Show more").focus()
    page.keyboard.press("Enter")
    until(page, lambda: len(control["held_lists"]) == 1)
    if ui["width"] < 720:
        page.locator('#nav-drawer-close').click()
        settle(page)
    composer = page.locator('#page-chat textarea').first
    composer.fill("Do not move my keyboard")
    route, body = control["held_lists"].pop()
    control["hold_lists"] = False
    route.fulfill(content_type="application/json", body=body)
    expect()(page.locator('.nav-project-item')).to_have_count(101)
    expect()(composer).to_be_focused()


def test_newer_feed_metadata_survives_leaving_the_recent_subset(pins_ui):
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    seed_archive(ui["root"])
    refresh(page)
    page.locator('.nav-find-input').fill("no such name")
    page.locator('.nav-projects-more', has_text="Show all projects").click()
    expect()(page.locator('.nav-project-item')).to_have_count(499)
    registry.update_project(ui["root"], "newer", name="New name from the feed")
    refresh(page)
    expect()(row(page, "newer").locator('.nav-row-label')).to_have_text("New name from the feed")
    registry.update_project(ui["root"], "newer", last_active_at="1900-01-01T00:00:00Z")
    control["unread"].add("archive-497")
    with page.expect_response(lambda response: urlparse(response.url).path == '/api/state'
            and any(p['id'] == 'archive-497' and p.get('visible_revision') == 2
                    for p in response.json().get('projects', []))):
        refresh(page)
    expect()(row(page, "archive-497").locator('.nav-project-row')).to_have_attribute("data-unread", "")
    assert "newer" not in {p["id"] for p in registry.projects_summary(ui["root"], include_pinned=True)}
    expect()(row(page, "newer").locator('.nav-row-label')).to_have_text("New name from the feed")


def test_pin_outside_recent_feed_survives_an_older_full_read_and_unpins_in_place(pins_ui):
    ui, page, control = pins_ui, pins_ui["page"], pins_ui["control"]
    seed_archive(ui["root"])
    refresh(page)
    page.locator('.nav-find-input').fill("no such name")
    page.locator('.nav-projects-more', has_text="Show all projects").click()
    expect()(page.locator('.nav-project-item')).to_have_count(499)
    control["hold_lists"] = True
    control["sockets"][-1].send(json.dumps({"type": "projects_changed", "project_id": "archive-000"}))
    until(page, lambda: len(control["held_lists"]) == 1)
    choose(page, "archive-000", "Pin")
    expect()(row(page, "archive-000")).to_have_attribute("data-pinned", "")
    assert "archive-000" in shown_ids(page)[:2]
    route, body = control["held_lists"].pop()
    control["hold_lists"] = False
    route.fulfill(content_type="application/json", body=body)
    expect()(page.locator('.nav-projects-note')).to_have_count(0)
    expect()(row(page, "archive-000")).to_have_attribute("data-pinned", "")
    choose(page, "archive-000", "Unpin")
    expect()(row(page, "archive-000")).not_to_have_attribute("data-pinned", "")
    assert shown_ids(page)[0] == "archive-100"


FIRST_IN_VIEW = """() => {
    const list = document.getElementById('nav-projects-list'), top = Math.max(list.getBoundingClientRect().top, 0);
    const item = [...list.children].find((el) => el.dataset.navProjectId && el.getBoundingClientRect().bottom > top);
    return [item.dataset.navProjectId, item.getBoundingClientRect().top];
}"""
SHOWS = """(el) => {
    const r = el.getBoundingClientRect(), hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
    return el === hit || el.contains(hit);
}"""


@pytest.mark.parametrize("how", ["escape", "clear-button", "clear-command"])
def test_short_window_clear_and_escape_return_to_the_browse_row_with_visible_focus(pins_ui, how):
    """Real presses in a short window, where the whole column scrolls. The find begins with the field in view (a
    click brings it there), so Escape, the field's × and Clear search put the row that was first in view back at its
    place, and the keyboard stays in the field, which shows; nothing scrolls after that. Escape is pressed after the
    owner scrolled the column on through the matches, the field gone from view."""
    ui, page = pins_ui, pins_ui["page"]
    page.set_viewport_size({"width": ui["width"], "height": 500})
    seed_archive(ui["root"])
    refresh(page)
    find = page.locator('#nav-projects .nav-find-input')
    find.fill("no such name")
    page.locator('#nav-projects-list > .nav-projects-more', has_text="Show all projects").click()
    items = page.locator('#nav-projects-list > .nav-project-item')
    expect()(items).to_have_count(499)
    sidebar = page.locator('#primary-sidebar')
    assert sidebar.evaluate("el => el.scrollHeight > el.clientHeight")  # the whole column scrolls here
    row(page, "archive-250").scroll_into_view_if_needed()  # the owner reads deep in the browse
    find.click()
    saved = page.evaluate(FIRST_IN_VIEW)
    page.keyboard.type("zzz no such" if how == "clear-command" else "archive")
    expect()(items).to_have_count(0 if how == "clear-command" else 51)
    if how == "escape":
        sidebar.evaluate("el => { el.scrollTop += 600; }")
        assert not find.evaluate(SHOWS)
        page.keyboard.press("Escape")
    elif how == "clear-button":
        page.locator('#nav-projects .nav-find-clear').click()
    else:
        page.locator('#nav-projects-list > .nav-projects-more', has_text="Clear search").click()
    expect()(find).to_have_value("")
    expect()(items).to_have_count(499)
    current, top = page.evaluate(FIRST_IN_VIEW)
    assert current == saved[0] and top == pytest.approx(saved[1], abs=1), (saved, current, top)
    expect()(find).to_be_focused()
    assert find.evaluate(SHOWS)
    page.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    assert page.evaluate(FIRST_IN_VIEW)[1] == pytest.approx(saved[1], abs=1)  # nothing moved it afterwards
