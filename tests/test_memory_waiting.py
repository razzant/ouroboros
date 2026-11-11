"""Integrating starts expose measurable waiting classes and the existing acts close each one."""
import dataclasses
import json
import re

import pytest

from ouroboros import memory_view as mv, projects_registry as pr
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.context_view_facts import memory_facts
from ouroboros.late_review_marks import publish_late_review
from ouroboros.memory_waiting import WAITING_CLASSES
from ouroboros.store import StoreLost
from ouroboros.tools.chronicle import _chronicle_write, _memory_mark, _memory_read
from tests import _memory_inventory_shared as shared
from tests.test_late_review_marks import _ctx, _record
from tests.test_memory_view_lever import plan


def view(root, role="integrator"):
    return mv.capture_memory_view(root, {"id": "root0001", "chat_id": 1},
                                  dataclasses.replace(mv.ROLE_DEFAULTS[role], room_id="1"))


def line(snapshot):
    p, _ = plan()
    return memory_facts(dataclasses.replace(p, core=dataclasses.replace(p.core, memory_view_json=mv.snapshot_json(snapshot))), "max")


def test_counts_include_foreign_reviews_notes_and_only_open_projects_and_fall_after_their_acts(tmp_path):
    rooms = shared.world(tmp_path)
    ctx = _ctx(tmp_path, "root0001")
    review = publish_late_review(tmp_path, _record(tmp_path, "bound", 1, "late words")).mark_id
    note = json.loads(_chronicle_write(ctx, kind="note", room_id=str(rooms["beta"]), text="A thought without open work."))["node_id"]
    ordinary = json.loads(_memory_mark(ctx, task_id="failed001", text="An ordinary mark"))["mark_id"]
    for role in ("integrator", "consciousness", "presence"):
        snapshot = view(tmp_path, role)
        assert snapshot.active
        facts, shown = snapshot.waiting, line(snapshot)
        assert set(facts) == set(WAITING_CLASSES)
        assert facts["late_reviews"]["count"] == facts["notes"]["count"] == facts["late_reviews"]["in_rooms"] == 1
        projects = pr.list_projects(tmp_path, closed=False, oldest_first=True)
        assert facts["projects"] == {"count": len(projects), "oldest": projects[0]["last_active_at"]}
        assert facts["notes"]["rooms"] == [(str(rooms["beta"]), 1)]
        assert "waiting for me at the start: late reviews 1 in 1 room, oldest " in shown
        assert "release_ids=[...]" in shown and "chronicle_write(kind='page'" in shown and "close_project(" in shown
        assert "failed001" not in shown and "late words" not in shown
    for role in ("child", "nanny"):
        snapshot = view(tmp_path, role)
        assert snapshot.waiting == {} and line(snapshot) == ""
    listed = _memory_read(ctx, room_id="late_reviews")
    assert review in listed and ordinary not in listed and "result review of task bound" in listed
    assert note in _memory_read(ctx, room_id="notes")
    assert json.loads(_memory_mark(ctx, release_ids=[review], reason="considered"))["ok"]
    page = json.loads(_chronicle_write(ctx, kind="page", room_id=str(rooms["beta"]), text="I resolved the thought.", covers={"note_ids": [note]}))
    assert page["ok"] and page["rows"] == 0 and page["notes"] == 1
    assert "covers (notes): 0 rows" in _memory_read(ctx, node_id=page["node_id"])
    for project in pr.list_projects(tmp_path):
        pr.update_project(tmp_path, project["id"], closed={"at": "2026-10-29T00:00:00Z", "by": "mind", "note": "completed"})
    from ouroboros.memory_view_conversations import owed
    debts = owed(view(tmp_path).conversations)
    assert view(tmp_path).waiting["conversations"]["count"] == len(debts) > 0  # a closed state seals no words
    for item in debts:  # the class's own act: a page over each room's open words
        for room in item["rooms"]:
            assert json.loads(_chronicle_write(ctx, kind="page", room_id=room["room_id"], text="Read and closed.",
                                               covers={"to": room["newest"]}))["ok"]
    assert all(f["count"] == 0 for f in view(tmp_path).waiting.values())
    assert review not in _memory_read(ctx, room_id="late_reviews") and note not in _memory_read(ctx, room_id="notes")
    assert {m["id"] for m in ChronicleStore(tmp_path).active_marks(None)} == {ordinary}


def test_note_only_page_closes_a_room_without_chat_rows_but_not_marks_and_refuses_bad_coverage(tmp_path):
    ctx = _ctx(tmp_path, "root0001")
    note = json.loads(_chronicle_write(ctx, kind="note", text="Only a note."))["node_id"]
    foreign = json.loads(_chronicle_write(ctx, kind="note", room_id="555", text="Elsewhere."))["node_id"]
    mark = json.loads(_memory_mark(ctx, node_id=note, text="Keep this."))["mark_id"]
    for ids in ([], [note, note], ["absent"], [foreign], [mark], "not-an-array"):
        assert "TOOL_ARG_ERROR" in _chronicle_write(ctx, kind="page", text="x", covers={"note_ids": ids})
    assert view(tmp_path).waiting["notes"]["count"] == 2
    page = json.loads(_chronicle_write(ctx, kind="page", text="Settled.", covers={"note_ids": [note]}))
    assert page["ok"] and page["notes"] == 1
    again = json.loads(_chronicle_write(ctx, kind="page", text="Repeated.", covers={"note_ids": [note]}))
    assert again["reason"] == "already_sealed"
    assert [m["id"] for m in ChronicleStore(tmp_path).active_marks("1")] == [mark]
    assert note not in _memory_read(ctx, room_id="notes") and foreign in _memory_read(ctx, room_id="notes")
    assert "Only a note." in _memory_read(ctx, node_id=note)


def test_waiting_readers_page_all_ids_and_render_from_a_frozen_snapshot(tmp_path, monkeypatch):
    from ouroboros.tools import chronicle

    ctx = _ctx(tmp_path, "root0001")
    ids = [json.loads(_chronicle_write(ctx, kind="note", text=f"Thought {i}"))["node_id"] for i in range(12)]
    snapshot = view(tmp_path)
    monkeypatch.setattr(chronicle, "tool_result_limit", lambda name: 500)
    chunks, start = [], 0
    for _ in range(30):
        chunk = _memory_read(ctx, room_id="notes", start=start)
        assert len(chunk) <= 500
        chunks.append(chunk)
        continuation = re.search(r"next_start=(\d+)", chunk.splitlines()[1])
        if not continuation:
            break
        start = int(continuation.group(1))
    else:
        pytest.fail("reader did not finish")
    assert all(ident in "".join(chunks) for ident in ids)
    before = line(snapshot)
    monkeypatch.setattr(pr, "list_projects", lambda *a, **kw: pytest.fail("render must use the snapshot"))
    monkeypatch.setattr(ChronicleStore, "active_marks", lambda *a, **kw: pytest.fail("render must use the snapshot"))
    assert line(snapshot) == before


@pytest.mark.parametrize("error", [ValueError, OSError, StoreLost])
def test_unavailable_project_census_keeps_readable_memory_and_other_waiting_facts(tmp_path, monkeypatch, error):
    shared.world(tmp_path)
    ctx = _ctx(tmp_path, "root0001")
    _chronicle_write(ctx, kind="note", text="A readable thought.")
    before = view(tmp_path)
    assert before.active and before.waiting["projects"]["count"] > 0

    def unavailable(*args, **kwargs):
        raise error("project census unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(pr, "list_projects", unavailable)
        snapshot = view(tmp_path)
        assert snapshot.active and snapshot.room == before.room and snapshot.story == before.story
        assert snapshot.conversations == before.conversations and snapshot.marks == before.marks
        assert snapshot.waiting["notes"] == before.waiting["notes"]
        assert snapshot.waiting["late_reviews"] == before.waiting["late_reviews"]
        assert snapshot.waiting["projects"] == {"count": "unknown", "oldest": "unknown"}
        shown = line(snapshot)
        assert "open projects unknown, oldest unknown" in shown
        assert "unsealed mind notes 1" in shown and "list_projects(closed=false, oldest_first=true)" in shown
    assert view(tmp_path).waiting == before.waiting


def _github_row(conversation, thread, minute, text, **extra):
    """A row of one issue thread of a GitHub repository (a place conversation without its thread)."""
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    chat = presence_chat_id(conversation_key("github", "me", conversation, thread))
    transport = {"provider": "github", "account_id": "me", "conversation_id": conversation, "thread_id": thread,
                 "actor": {"platform_actor_id": "u7"}}
    return {"chat_id": chat, "direction": "in", "ts": f"2026-09-06T01:{minute:02d}:00+00:00", "text": text,
            "source": "presence:github", "client_message_id": f"gh-{conversation}-{thread}-{minute}",
            "transport": transport, **extra}


def test_a_conversation_waits_while_words_where_i_spoke_are_open_and_never_for_ambient_rooms(tmp_path):
    """S4: a real conversation waits for its page; a connector-marked bot's event does not."""
    from ouroboros import chat_chain, dialogue_store as ds
    from ouroboros.memory_view_conversations import owed

    shared.world(tmp_path)
    ctx = _ctx(tmp_path, "root0001")
    asked = _github_row("repo.1", "i1", 1, "can you look?")
    answer = {**_github_row("repo.1", "i1", 2, "Looking."), "direction": "out", "type": "presence_delivery",
              "task_id": "p1", "client_message_id": "", "transport": {**asked["transport"], "delivery": {"state": "delivered"}}}
    ambient = _github_row("repo.2", "i9", 3, "a dependabot bump")
    ambient["transport"]["actor"]["is_bot"] = True
    for row in (asked, answer, ambient):
        ds.append(tmp_path, "chat", row)

    def owing():
        return {item["group"]["key"] for item in owed(view(tmp_path).conversations)}

    assert "github:me:repo.1" in owing() and "github:me:repo.2" not in owing()
    snapshot = view(tmp_path)
    facts, shown = snapshot.waiting["conversations"], line(snapshot)
    assert facts["count"] == len(owed(snapshot.conversations)) and "conversations whose words no page holds" in shown
    assert "memory_read(room_id='conversations', start=0)" in shown and "'conversation': true" in shown
    listing = _memory_read(ctx, room_id="conversations")
    assert listing.startswith("Conversations whose words no page holds: ") and "repo.2" not in listing
    newest = chat_chain.format_address(chat_chain.row_address(answer))
    assert f"covers={{'to': '{newest}'}}" in listing
    assert json.loads(_chronicle_write(ctx, kind="page", room_id=str(asked["chat_id"]), text="Answered; closed.",
                                       covers={"to": newest}))["ok"]
    assert "github:me:repo.1" not in owing()


def _event_row(provider, thread, minute, author="own", text=None, **extra):
    """The same connector contract under unrelated provider names."""
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id
    key = conversation_key(provider, "account", "garden", thread)
    actor = {} if author == "unknown" else {"platform_actor_id": "self" if author == "own" else author}
    if author == "bot":
        actor["is_bot"] = True
    return {"chat_id": presence_chat_id(key), "direction": "in", "source": "presence:" + provider,
            "ts": f"2026-09-09T00:{minute:02d}:00+00:00", "text": text or f"{thread} title\n{author} body {minute}",
            "client_message_id": f"{provider}-{thread}-{minute}",
            "transport": {"provider": provider, "account_id": "account", "conversation_id": "garden",
                          "thread_id": thread, "actor": actor, "identity": {"sends_as": {"id": "self"}}}, **extra}


def _event_group(snapshot, provider="harbor"):
    return next(g for g in snapshot.conversations if g["key"] == f"{provider}:account:garden")


@pytest.mark.parametrize("provider", ["harbor", "kiln", "fern", "kite"])
@pytest.mark.parametrize("with_bot", [False, True])
def test_account_arrivals_are_ordinary_events_independent_of_provider(tmp_path, provider, with_bot):
    from ouroboros import dialogue_store as ds, memory_floor as mf
    from ouroboros.memory_view_conversations import owed
    shared.world(tmp_path)
    ds.append(tmp_path, "chat", _event_row(provider, "one", 1))
    if with_bot:
        ds.append(tmp_path, "chat", _event_row(provider, "one", 2, "bot"))
    snapshot = view(tmp_path)
    group = _event_group(snapshot, provider)
    assert not group["rooms"][0]["engaged"] and not owed([group])
    isolated = dataclasses.replace(snapshot, room=None, story=[], conversations=[group])
    counts = mf.form_counts(isolated, mv.FULL_VIEW)
    assert not counts["emergency"] and counts["host"]["events"] == {"lines": 1, "sources": 1 + with_bot}
    shown = mv.render_room(isolated)
    assert "account not confirmed as my human's" in shown and "one title" in shown
    assert "Event:" in shown and "own body" not in shown


@pytest.mark.parametrize("prior", ["human", "reply"])
@pytest.mark.parametrize("sealed", [False, True])
def test_a_recorded_person_or_reply_engages_a_room_even_when_sealed(tmp_path, prior, sealed):
    from ouroboros import chat_chain, dialogue_store as ds
    shared.world(tmp_path)
    row = _event_row("harbor", "one", 1, "Ann", text="A person's real question.")
    if prior == "reply":
        row.update(direction="out", type="presence_delivery", task_id="answer01", text="My outward reply.")
    ds.append(tmp_path, "chat", row)
    if sealed:
        reply = json.loads(_chronicle_write(_ctx(tmp_path, "root0001"), kind="page", room_id=str(row["chat_id"]),
                          text="I kept that exchange.", covers={"to": chat_chain.format_address(chat_chain.row_address(row))}))
        assert reply["ok"], reply
    ds.append(tmp_path, "chat", _event_row("harbor", "one", 2, "bot", text="Bot words stay whole here.\nSECOND_LINE"))
    snapshot = view(tmp_path)
    group = _event_group(snapshot)
    assert group["rooms"][0]["engaged"] and not group["rooms"][0].get("events")
    bot = group["rooms"][0]["words"][-1]
    assert bot["text"] == "Bot words stay whole here.\nSECOND_LINE"
    assert bot["line"] in mv.render_room(snapshot)


def test_unauthenticated_same_id_is_a_person_and_a_marked_mail_bot_is_an_event(tmp_path):
    from ouroboros import dialogue_store as ds
    shared.world(tmp_path)
    row = _event_row("harbor", "one", 1)
    row["transport"]["actor"]["authenticated"] = False
    ds.append(tmp_path, "chat", row)
    ds.append(tmp_path, "chat", _event_row("mailbox", "inbox", 2, "bot"))
    snapshot = view(tmp_path)
    assert _event_group(snapshot)["rooms"][0]["engaged"]
    assert not _event_group(snapshot, "mailbox")["rooms"][0]["engaged"]


@pytest.mark.parametrize("provider", ["harbor", "mailbox"])
@pytest.mark.parametrize("count_unknown", [True, False])
def test_unattributed_arrivals_have_one_owner_choice_and_keep_their_reader(tmp_path, monkeypatch, provider, count_unknown):
    from ouroboros import dialogue_store as ds, memory_view_events as events
    from ouroboros import memory_view_conversations as conv
    shared.world(tmp_path)
    monkeypatch.setattr(events, "count_unattributed", lambda: count_unknown)
    text = "Unknown opening. " + "unconfirmed words " * 30
    ds.append(tmp_path, "chat", _event_row(provider, "one", 1, "unknown", text=text))
    snapshot = view(tmp_path)
    group = _event_group(snapshot, provider)
    room = group["rooms"][0]
    shown = conv.render_group(group, set(), set(), 0)
    assert not room["engaged"] and not conv.owed([group])
    assert f"author not recorded: 1 arrivals, {len(text)} chars" in shown
    assert "last 2026-09-09T00:01:00+00:00" in shown
    assert (text in shown) is not count_unknown
    assert text in _memory_read(_ctx(tmp_path, "root0001"), room_id=room["room_id"], rows=True)
    if not count_unknown:
        word = room["words"][0]
        opened = conv.render_group(group, {word["id"]}, set(), 0)
        assert text not in opened and "Unknown opening." in opened and "opening, no retelling" in opened
    ds.append(tmp_path, "chat", _event_row(provider, "one", 2, "Ann"))
    live = _event_group(view(tmp_path), provider)["rooms"][0]
    assert live["engaged"] and live["words"][0]["text"] == text


def test_event_counts_use_the_full_catalogue_offset_and_remain_ordinary(tmp_path):
    from ouroboros import dialogue_store as ds, memory_floor as mf
    from ouroboros.memory_view_conversations import pieces
    from ouroboros.tools.chronicle import _groups, _read_rooms
    shared.world(tmp_path)
    # The focus's own group sorts before the tested conversation, so omitting it gives a wrong offset.
    rows = [_event_row("aaa", "focus", 0, "Ann"),
            *[_event_row("harbor", f"topic{i}", i + 1) for i in range(3)],
            _event_row("harbor", "topic0", 4, "bot", text="Latest recorded title\nBOT_BODY"),
            _event_row("harbor", "topic2", 5, "unknown", text="Unknown tail\nUNATTRIBUTED_BODY")]
    for row in rows:
        ds.append(tmp_path, "chat", row)
    task = {"id": "root0001", "chat_id": rows[0]["chat_id"]}
    snapshot = mv.capture_memory_view(tmp_path, task, dataclasses.replace(mv.ROLE_DEFAULTS["integrator"], room_id=str(task["chat_id"])))
    group = _event_group(snapshot)
    selected = {"F6g": {"events:" + group["key"]}}
    def draw(gone):
        return "\n".join(pieces([group], gone, lambda _key, render: render()))
    whole, counted = draw({}), draw(selected)
    assert all(f"topic{i} title" in whole and f"topic{i} title" not in counted for i in range(3))
    assert "Events: 3 rooms, 5 arrivals" in counted and "Latest recorded title" in counted
    assert "author not recorded: 1 arrivals" in counted and "Unknown tail" in counted
    assert "a bot" in counted and "00:04" in counted
    offset = int(re.search(r"memory_read\(room_id='\*', start=(\d+)\)", counted)[1])
    assert offset > 0
    store = ChronicleStore(tmp_path)
    from ouroboros.memory_view_events import catalogue_lines
    groups = _groups(tmp_path, store)
    catalogue = "\n".join(line for g in groups for line in catalogue_lines(g))
    from ouroboros.memory_view_conversations import identity_line
    expected = identity_line(next(g for g in groups if g["key"] == group["key"]))
    assert catalogue[offset:].startswith(expected)
    assert expected in _read_rooms(tmp_path, store, offset, 10_000)
    for prefix in ("all:", "last:"):
        assert "Events: 3 rooms, 5 arrivals" in draw({"F6g": {prefix + group["key"]}})
    physical = draw({"F6a": {group["key"]}})
    assert "3 event rooms, 5 arrivals" in physical and "Latest recorded title" not in physical
    isolated = dataclasses.replace(snapshot, room=None, story=[], conversations=[group])
    level = mv.FloorLevel((("F6g", tuple(selected["F6g"])),))
    from ouroboros.memory_forms import stage_of
    first = mf.floor_elements(isolated)[0]
    assert stage_of(isolated, *first[:2]) == "remote_events"
    frozen = mv.snapshot_json(isolated)
    for gone in ({}, selected, {"F6g": {"all:" + group["key"]}},
                 {"F6g": {"last:" + group["key"]}}, {"F6a": {group["key"]}}):
        state = mv.FloorLevel(tuple((step, tuple(ids)) for step, ids in gone.items()))
        assert mv.room_chars(isolated, state) == mv.json_chars(mv.render_room(isolated, state))
    assert mv.snapshot_json(isolated) == frozen and '"_samples"' not in frozen
    counted_level = mv.FloorLevel((("F6a", (group["key"],)),))
    assert "conversation count" in mf.form_counts(isolated, counted_level)["emergency"]  # R2
    assert mf.form_counts(isolated, counted_level)["host"]["event counts"]["sources"] == 5
    ordinary = mf.ordinary_level(isolated)
    assert not mf.form_counts(isolated, level)["emergency"]
    budget = mf.tokens_of(mv.room_chars(isolated, ordinary)) + mf.tokens_of(mv.story_chars(isolated, ordinary)) + 30
    fitted = mf.fit_memory_view(isolated, {"margin": budget, "physical": 10**9, "budget": None})
    assert not mf.form_counts(isolated, fitted)["emergency"]
    from ouroboros.memory_view_conversations import folded_rooms
    assert not folded_rooms([group], dict(fitted.addressed))
