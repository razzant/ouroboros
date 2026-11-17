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


def test_the_line_counts_clean_passes_from_published_facts_and_the_list_reads_an_older_marks_record(tmp_path, monkeypatch):
    """Late reviews in the waiting line: how many are a PASS with nothing to read and how many carry no outcome
    facts, from the captured marks alone; the list prints every mark's outcome and reads an older mark's record."""
    from ouroboros import memory_waiting, review_ledger
    from ouroboros.late_review_marks import outcome_facts
    from tests.test_late_review_marks import _settled

    rooms = shared.world(tmp_path)
    ctx = _ctx(tmp_path, "root0001")
    record_id = "rl-20261017T000000-bbbbbbbbbbbb"
    for task, chat, extra in (("main1", 1, {"outcome": outcome_facts(_settled("PASS"))}),
                              ("bound", 1, {"outcome": outcome_facts(_settled("PASS", advisory=1))}),
                              ("beta1", rooms["beta"], {"subject": {"record_id": record_id}})):
        assert publish_late_review(tmp_path, {**_record(tmp_path, task, chat, "late words"), **extra}).status == "saved"
    review_ledger.ledger_dir(tmp_path, create=True)
    review_ledger.record_path(tmp_path, record_id).write_text(json.dumps(_settled("PASS")), encoding="utf-8")

    with monkeypatch.context() as patch:  # the line is built from the captured marks: no record is read for it
        patch.setattr(review_ledger, "load_record", lambda *a, **kw: pytest.fail("the line reads no record"))
        snapshot = view(tmp_path)
        facts, shown = snapshot.waiting["late_reviews"], line(snapshot)
    assert (facts["count"], facts["in_rooms"], facts["clean_pass"], facts["unrecorded"]) == (3, 3, 1, 1)
    assert "late reviews 3 in 3 rooms, oldest " in shown
    assert (", 1 PASS with no findings, 1 with no outcome kept on the mark (the list reads the record where one exists); "
            "memory_read(room_id='late_reviews'") in shown
    assert ", 1 PASS with no findings; memory_read(room_id='late_reviews'" in memory_waiting.text({
        **snapshot.waiting, "late_reviews": {**facts, "unrecorded": 0}}), "every mark keeps its outcome: one fact"
    older = {**snapshot.waiting, "late_reviews": {k: v for k, v in facts.items() if k not in ("clean_pass", "unrecorded")}}
    assert "PASS with no findings" not in memory_waiting.text(older), "a view captured before the counts keeps its words"
    assert "PASS with no findings" not in memory_waiting.text({**snapshot.waiting, "late_reviews": {
        **facts, "count": 0, "clean_pass": 0, "unrecorded": 0}}), "nothing waits: nothing to count"

    listed = _memory_read(ctx, room_id="late_reviews")
    assert listed.splitlines()[0] == ("late reviews: 3 in 3 rooms; 2 PASS with no findings; 0 PASS commit reviews with "
                                      "no commit under that review and no merged receipt of their task")
    for task, words in (("main1", "PASS, no findings"), ("bound", "PASS, 1 advisory"), ("beta1", "PASS, no findings")):
        assert f"result review of task {task}; {words}; memory_read(node_id='late-review-" in listed, task
    review_ledger.record_path(tmp_path, record_id).unlink()
    assert "result review of task beta1; outcome not recorded; memory_read(" in _memory_read(ctx, room_id="late_reviews")


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
def test_unattributed_arrivals_stand_in_events_and_stay_words_in_a_live_conversation(tmp_path, provider):
    from ouroboros import dialogue_store as ds
    from ouroboros import memory_view_conversations as conv
    shared.world(tmp_path)
    text = "Unknown opening. " + "unconfirmed words " * 30
    ds.append(tmp_path, "chat", _event_row(provider, "one", 1, "unknown", text=text))
    snapshot = view(tmp_path)
    group = _event_group(snapshot, provider)
    room = group["rooms"][0]
    shown = conv.render_group(group, set(), set(), 0)
    assert not room["engaged"] and not conv.owed([group])
    assert f"author not recorded: 1 arrivals, {len(text)} chars" in shown
    assert "last 2026-09-09T00:01:00+00:00" in shown
    assert text not in shown
    assert text in _memory_read(_ctx(tmp_path, "root0001"), room_id=room["room_id"], rows=True)
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


def _overheard_view(root, provider="harbor"):
    from ouroboros import chat_chain, dialogue_store as ds
    shared.world(root)
    focus = _event_row("aaa", "focus", 0, "Ann")
    ds.append(root, "chat", focus)
    reply = _event_row(provider, "mine", 1, direction="out", type="presence_delivery", task_id="answer01")
    ds.append(root, "chat", reply)
    result = json.loads(_chronicle_write(_ctx(root, "root0001"), kind="page", room_id=str(reply["chat_id"]),
        text="I answered here.", covers={"to": chat_chain.format_address(chat_chain.row_address(reply))}))
    assert result["ok"], result
    for i in range(6):
        ds.append(root, "chat", _event_row(provider, f"topic{i // 2}", 10 + i, "Ann",
            text=f"Topic {i} opening.\n" + f"The details of question {i} stay readable. " * 30))
    ds.append(root, "chat", _event_row(provider, "mine", 16, "Ben", text="A new question for me.\n" + "Detail. " * 200))
    task = {"id": "root0001", "chat_id": focus["chat_id"]}
    return mv.capture_memory_view(root, task, dataclasses.replace(mv.ROLE_DEFAULTS["integrator"], room_id=str(focus["chat_id"])))


@pytest.mark.parametrize("provider", ["harbor", "kiln", "fern", "kite"])
def test_overheard_unread_is_per_conversation_only_in_line_forms(tmp_path, provider):
    from ouroboros import memory_floor as mf, memory_view_conversation_forms as forms
    snapshot = _overheard_view(tmp_path, provider)
    group = _event_group(snapshot, provider)
    snapshot = dataclasses.replace(snapshot, room=None, story=[], conversations=[group])
    unread = [w for r in group["rooms"] for w in r["words"]]
    latest = max((w for w in unread if "Topic" in w["text"]), key=lambda w: w["pos"])
    def draw(gone):
        return "\n".join(forms.pieces([group], gone, lambda _key, render: render()))
    whole = draw({})
    assert all(w["line"] in whole for w in unread) and "No recorded consideration:" not in whole
    room_lines = draw({"F6w": {w["id"] for w in unread}})
    assert room_lines.count("No recorded consideration:") == 4 and "with no recorded word of my human or reply of mine" not in room_lines
    for gone in ({"F6g": {prefix + group["key"]}} for prefix in ("", "all:", "last:")):
        shown = draw(gone)
        assert "No recorded consideration: 6 words from 1 person, 2026-09-09 → 2026-09-09, in 3 rooms with no recorded word of my human or reply of mine;" in shown
        assert "No recorded consideration: 1 words;" in shown and shown.count("No recorded consideration:") == 2
        counter = next(line for line in shown.splitlines() if "with no recorded word of my human or reply of mine" in line)
        assert latest["address"] in counter and latest["author"] in counter and "Topic 5 opening." in counter
        assert all(w["line"] not in shown for w in unread)
    for gone in ({"F6g": {group["key"]}}, {"F6a": {group["key"]}}):
        assert "in 3 rooms with no recorded word of my human or reply of mine" in draw(gone)
        level = mv.FloorLevel(tuple((step, tuple(ids)) for step, ids in gone.items()))
        host = mf.form_counts(snapshot, level)["host"]
        assert host["unread in rooms without my human's word or my reply"] == {"lines": 1, "sources": 6}
        assert host["unread"] == {"lines": 1, "sources": 1}
    aggregate = {"F6a": {group["key"], "unread"}}
    assert "7 words without recorded consideration" in draw(aggregate) and "with no recorded word of my human or reply of mine" not in draw(aggregate)
    counts = mf.form_counts(snapshot, mv.FloorLevel((("F6a", tuple(aggregate["F6a"])),)))
    assert counts["window"]["unread count"] == {"lines": 1, "sources": 7}
    assert "unread in rooms without my human's word or my reply" not in counts["host"]


@pytest.mark.parametrize("prior", ["owner", "forward", "reply", "stranger", "own_account"])
@pytest.mark.parametrize("sealed", [False, True])
def test_overheard_excludes_my_humans_or_my_participation_in_open_and_sealed_rows(tmp_path, prior, sealed):
    from ouroboros import chat_chain, dialogue_store as ds, places
    from ouroboros.memory_view_conversation_forms import render_group
    shared.world(tmp_path)
    place = places.connect_place(tmp_path, connector="bridge", provider="harbor", account_id="account", label="Garden")
    places.edit_place(tmp_path, place.place_id, "confirm_owner_account", author="owner", platform_actor_id="anton",
                      account_id="account")
    author = "anton" if prior in ("owner", "forward") else "own" if prior == "own_account" else "Ann"
    row = _event_row("harbor", "one", 1, author, presence_provenance={"place_id": place.place_id})
    if prior == "forward":
        row["transport"]["message"] = {"forward_origin": {"sender_user_name": "Other"}}
    if prior == "reply":
        row.update(direction="out", type="presence_delivery", task_id="answer01")
    ds.append(tmp_path, "chat", row)
    if sealed:
        result = json.loads(_chronicle_write(_ctx(tmp_path, "root0001"), kind="page", room_id=str(row["chat_id"]),
            text="The earlier exchange.", covers={"to": chat_chain.format_address(chat_chain.row_address(row))}))
        assert result["ok"], result
    ds.append(tmp_path, "chat", _event_row("harbor", "one", 2, "Ann", text="Still waiting for consideration."))
    group = _event_group(view(tmp_path))
    room = group["rooms"][0]
    assert room["engaged"] and room["overheard"] == (prior in ("stranger", "own_account"))
    if prior == "owner" and not sealed:
        owner = next(w for w in room["words"] if w["owner_side"])
        assert all(owner["line"] in render_group(group, set(), set(), form) for form in range(4))


def test_overheard_reader_guard_and_local_prices_share_the_conversation_state(tmp_path):
    from ouroboros import memory_floor as mf, memory_view_conversation_forms as forms
    from ouroboros.memory_view_events import catalogue_lines
    from ouroboros.tools.chronicle import _groups, _read_rooms
    snapshot = _overheard_view(tmp_path)
    group = _event_group(snapshot)
    snapshot = dataclasses.replace(snapshot, room=None, story=[], conversations=[group])
    ordinary = mf.ordinary_level(snapshot)
    text = mv.render_room(snapshot, ordinary)
    assert "in 3 rooms with no recorded word of my human or reply of mine" in text
    offset = int(re.search(r"memory_read\(room_id='\*', start=(\d+)\)", text)[1])
    groups = _groups(tmp_path, ChronicleStore(tmp_path))
    catalogue = "\n".join(line for g in groups for line in catalogue_lines(g))
    identity = forms.identity_line(next(g for g in groups if g["key"] == group["key"]))
    assert offset > 0 and catalogue[offset:].startswith(identity)
    assert identity in _read_rooms(tmp_path, ChronicleStore(tmp_path), offset, 10_000)
    frozen = mv.snapshot_json(snapshot)
    index = forms.source_index(snapshot)[group["key"]]
    selected = {"F6g": {group["key"]}}
    assert forms.local_key(group, selected, index) == forms.local_key(group, {"F6g": {group["key"], "unrelated"}}, index)
    assert forms.local_key(group, selected, index) != forms.local_key(group, {}, index)
    for gone in ({}, selected, {"F6g": {"all:" + group["key"]}}, {"F6g": {"last:" + group["key"]}},
                 {"F6w": {w["id"] for r in group["rooms"] for w in r["words"]}},
                 {"F6a": {group["key"]}}, {"F6a": {group["key"], "unread"}}):
        level = mv.FloorLevel(tuple((step, tuple(sorted(ids))) for step, ids in gone.items()))
        assert mv.room_chars(snapshot, level) == mv.json_chars(mv.render_room(snapshot, level))
        counts = mf.form_counts(snapshot, level)["host"]
        assert counts.get("unread in rooms without my human's word or my reply", {}).get("lines", 0) == mv.render_room(snapshot, level).count("with no recorded word of my human or reply of mine")
    assert mv.snapshot_json(snapshot) == frozen
    budget = mf.tokens_of(mv.room_chars(snapshot, ordinary)) + mf.tokens_of(mv.story_chars(snapshot, ordinary))
    fitted = mf.fit_memory_view(snapshot, {"margin": budget, "physical": budget, "budget": None})
    counts = mf.form_counts(snapshot, fitted)
    assert not counts["emergency"] and not counts["window"]
    assert counts["host"]["unread in rooms without my human's word or my reply"] == {"lines": 1, "sources": 6}
    # This host count alone gives no reason to commission a retelling.
    from ouroboros.memory_inventory import shortage_from_trace, VIEW_TRACE_KEY
    overheard = {**group, "rooms": [r for r in group["rooms"] if r["overheard"]]}
    alone = dataclasses.replace(snapshot, conversations=[overheard])
    lined = mv.FloorLevel((("F6g", (group["key"],)),))
    fact = mf.view_facts(alone, lined, window_tokens=budget, mode="max", allowances={"margin": budget})
    assert not fact["floor"]["folded_rooms"] and shortage_from_trace({VIEW_TRACE_KEY: fact}) is None
    # Unread words that yield alone are host facts too (rev_grok REV1 R2): no room owes a telling for them.
    assert forms.folded_rooms([overheard], {"F6w": {w["id"] for r in overheard["rooms"] for w in r["words"]}}) == []
    # Under pressure the new line arrives before this room's final reply yields to a work line.
    from tests.test_memory_distance_forms import completed_room, size
    completed, _gist = completed_room()
    room = completed.room
    with_work = dataclasses.replace(snapshot, room=room)
    bound = size(with_work, lined)
    fitted = mf.fit_memory_view(with_work, {"margin": bound, "physical": bound, "budget": None})
    assert "in 3 rooms with no recorded word of my human or reply of mine" in mv.render_room(with_work, fitted)
    assert room["lane1"][-1]["line"] in mv.render_room(with_work, fitted)


def test_unread_counter_names_the_predicate_people_days_and_its_conversation(tmp_path):
    from ouroboros import memory_view_conversation_forms as forms
    from ouroboros.memory_view_events import catalogue_lines
    group = _event_group(_overheard_view(tmp_path))
    expected = ("No recorded consideration: 6 words from 1 person, 2026-09-09 → 2026-09-09, "
                "in 3 rooms with no recorded word of my human or reply of mine;")
    assert expected in forms.render_group(group, set(), set(), 1)
    counted = forms._counted_text([group], set(), False)
    assert group["label"] + ": " + expected in counted
    assert expected not in forms.render_group(group, set(), set(), 0)
    catalogue = catalogue_lines(group)
    for room, line in zip(group["rooms"], catalogue[1:]):
        predicate = "a person spoke here; no word of my human or reply of mine"
        assert (predicate in line) is room["overheard"]
        assert ("a person or I spoke here" in line) is not room["overheard"]


@pytest.mark.parametrize("last_person, phrase", [("Ann", "1 person"), ("Ben", "2 people")])
def test_unread_people_ignore_threads_but_keep_connection_and_speaker(tmp_path, last_person, phrase):
    from ouroboros import dialogue_store as ds, memory_view_conversation_forms as forms
    from ouroboros.memory_view_work import speech_facts
    shared.world(tmp_path)
    for i, person in enumerate(("Ann", "Ann", last_person)):
        ds.append(tmp_path, "chat", _event_row("harbor", f"thread{i}", i + 1, person))
    group = _event_group(view(tmp_path))
    assert f"3 words from {phrase}," in forms.render_group(group, set(), set(), 1)
    counted = forms._counted_text([group], set(), True)
    assert f"consideration, {phrase}, 3 rooms" in counted
    assert " authors," not in counted
    # A restored old snapshot keeps its historical label-based count.
    old = json.loads(json.dumps(group))
    for room in old["rooms"]:
        for word in room["words"]:
            word.pop("speaker", None)
    assert "3 words from 3 people," in forms.render_group(old, set(), set(), 1)
    row = _event_row("harbor", "first", 1, "Ann")
    key = speech_facts(row, {})["speaker"]
    row["transport"].update(conversation_id="elsewhere", thread_id="second")
    assert speech_facts(row, {})["speaker"] == key
    row["transport"]["account_id"] = "another-connection"
    assert speech_facts(row, {})["speaker"] != key


def test_unread_host_counts_do_not_create_page_debt_or_restore_it_at_the_window_floor(tmp_path):
    from ouroboros import memory_view_conversation_forms as forms
    group = _event_group(_overheard_view(tmp_path))
    ours = next(r for r in group["rooms"] if not r["overheard"])
    others = [r for r in group["rooms"] if r["overheard"]]
    assert forms.owed([group])[0]["rooms"] == [ours]
    assert not forms.owed([{**group, "rooms": others}])
    for step, ids in (("F6g", [group["key"]]), ("F6a", [group["key"]]), ("F6a", [group["key"], "unread"])):
        expected = [] if "unread" in ids else [ours["room_id"]]
        assert forms.folded_rooms([group], {step: ids}) == expected
    # A room of other people alone never reaches the writer, even with a considered word: a page there would seal
    # the unread words beside it. Where I spoke, a considered word is still an untold source under that count.
    others[0]["words"][0]["rank"] = 1
    assert forms.folded_rooms([group], {"F6a": [group["key"], "unread"]}) == []
    ours["words"][0]["rank"] = 1
    assert forms.folded_rooms([group], {"F6a": [group["key"], "unread"]}) == [ours["room_id"]]


@pytest.mark.parametrize("overheard", [True, False])
def test_an_unread_word_yielding_alone_through_a_real_fit_owes_no_telling(overheard):
    """rev_grok REV1 R2: across a one-token gap the fit took one unread word (F6w) and printed its host line, yet
    ``folded_rooms`` listed the room, so a writer with consciousness off would draft a page sealing words nobody
    considered. Unread words are host facts wherever they yield, in an overheard room and in one where my human or I
    spoke; a considered word that yields the same way still owes a telling where my human or I spoke, and a room of
    other people alone never reaches the writer (rev_opus and rev_fable REV2)."""
    from ouroboros import memory_floor as mf
    from ouroboros.memory_inventory import VIEW_TRACE_KEY, shortage_from_trace
    from tests import _memory_view_synthetic as syn

    def fitted(rank):
        group = syn.conversation(5, words=4, word_chars=2_000)
        group.update(place=overheard, catalogue_start=0)
        room = group["rooms"][0]
        room["overheard"] = overheard
        for word in room["words"]:
            word["rank"] = rank
        snap = syn.snapshot(room_facts=syn.room("1"), conversations=[group])
        whole = mf.tokens_of(mv.story_chars(snap)) + mf.tokens_of(mv.room_chars(snap))
        level = mf.fit_memory_view(snap, {"margin": whole - 1, "physical": whole - 1, "budget": None})
        fact = mf.view_facts(snap, level, window_tokens=whole, mode="max", allowances={"margin": whole - 1})
        return snap, level, fact, room["room_id"]

    snap, level, fact, _room = fitted(2)
    assert dict(level.steps) == {"F6w": 1}, level
    assert "No recorded consideration: 1 words;" in mv.render_room(snap, level)
    assert fact["floor"]["folded_rooms"] == [] and shortage_from_trace({VIEW_TRACE_KEY: fact}) is None
    # The other direction: a considered word yields as its opening, an untold source that owes a telling where my
    # human or I spoke; in a room of other people alone the same form gives the writer nothing.
    snap, level, fact, room_id = fitted(1)
    assert dict(level.steps) == {"F6w": 1} and "opening, no retelling" in mv.render_room(snap, level), level
    shortage = shortage_from_trace({VIEW_TRACE_KEY: fact})
    if overheard:
        assert fact["floor"]["folded_rooms"] == [] and shortage is None
    else:
        assert fact["floor"]["folded_rooms"] == [room_id] and shortage.kind == "other_room"


def test_a_reaction_beside_unread_words_owes_no_telling_through_a_real_capture_and_fit(tmp_path):
    """rev_opus REV2 finding 1: a thread of other people alone held three unread words and one reaction. The
    reaction is a place fact with no words, yet ``folded_rooms`` listed the room once its conversation took its
    ordinary line, so a writer with consciousness off would draft a page sealing words nobody considered. Such a
    room never reaches the writer, whatever else it holds (rev_fable REV2: an arrival with no recorded author did
    the same); the same lined conversation still owes a telling for a room where my human or I spoke."""
    from ouroboros import dialogue_store as ds, memory_floor as mf, memory_view_conversation_forms as forms
    from ouroboros.memory_inventory import VIEW_TRACE_KEY, shortage_from_trace
    shared.world(tmp_path)
    for i in range(3):
        ds.append(tmp_path, "chat", _event_row("harbor", "t1", 1 + i, "Ann",
                                               text=f"Ann question {i}.\n" + "Detail of an unread word. " * 60))
    reaction = _event_row("harbor", "t1", 9, "Cid")
    reaction["text"] = ""
    reaction["transport"]["message"] = {"reaction": {"schema_version": 1, "action": "added", "values": [{"name": "eyes"}]}}
    ds.append(tmp_path, "chat", reaction)
    snapshot = view(tmp_path)
    group = _event_group(snapshot)
    room = group["rooms"][0]
    assert room["engaged"] and room["overheard"]
    alone = dataclasses.replace(snapshot, room=None, story=[], conversations=[group])
    assert sorted(forms.source_index(alone)[group["key"]]["kinds"].values()) == ["fixed", "unread", "unread", "unread"]
    ordinary = mf.ordinary_level(alone)
    budget = mf.tokens_of(mv.room_chars(alone, ordinary)) + mf.tokens_of(mv.story_chars(alone, ordinary))
    fitted = mf.fit_memory_view(alone, {"margin": budget, "physical": 10**9, "budget": None})
    assert dict(fitted.steps).get("F6g") == 1 and not mf.form_counts(alone, fitted)["emergency"], fitted
    fact = mf.view_facts(alone, fitted, window_tokens=budget, mode="max", allowances={"margin": budget})
    assert fact["floor"]["folded_rooms"] == [] and shortage_from_trace({VIEW_TRACE_KEY: fact}) is None
    # Neither a considered word nor an arrival with no recorded author changes that for such a room ...
    lined = {"F6g": [group["key"]]}
    next(w for w in room["words"] if w.get("kind") == "human")["rank"] = 1
    assert forms.folded_rooms([group], lined) == []
    # ... and the other direction: the same lined conversation owes a telling once my human or I spoke in the room.
    room["overheard"] = False
    assert forms.folded_rooms([group], lined) == [room["room_id"]]


@pytest.mark.parametrize("own_text", ["", "Their own first line.\nTheir later text."])
def test_unread_openings_quote_only_the_rows_own_text(own_text):
    from ouroboros import memory_view_conversation_forms as forms
    from tests import _memory_view_synthetic as syn
    group = syn.conversation(1, words=1)
    group["catalogue_start"] = 12
    room = group["rooms"][0]
    room["overheard"] = True
    word = room["words"][0]
    host = "What the connector reports about this message (host facts, not their words):"
    word.update(text=own_text, line=word["head"] + " " + (own_text or host), author="Ann")
    expected = "«Their own first line." if own_text else "no text of their own"
    for text in (forms._unread_line([word]), forms.render_group(group, set(), set(), 1),
                 forms._counted_text([group], set(), True)):
        assert expected in text and host not in text
        assert "«no text of their own»" not in text
    assert word["line"] in forms.render_group(group, set(), set(), 0)


@pytest.mark.parametrize("author", ["own", "unknown"])
@pytest.mark.parametrize("length", [40, 500])
def test_event_openings_are_bounded_and_an_empty_room_names_its_last_fact(tmp_path, author, length):
    from ouroboros import dialogue_store as ds, memory_view_events as events
    shared.world(tmp_path)
    row = _event_row("harbor", "one", 1, direction="system", type="task_summary", task_id="done")
    ds.append(tmp_path, "chat", row)
    room = _event_group(view(tmp_path))["rooms"][0]
    empty = events.room_line(room)
    assert "; no arrivals; last 2026-09-09 00:01; memory_read(" in empty
    assert "author not recorded" not in empty
    first = "A" * length
    ds.append(tmp_path, "chat", _event_row("harbor", "one", 2, author, text=first + "\nLater paragraph."))
    room = _event_group(view(tmp_path))["rooms"][0]
    event = room["events"]
    assert event["first"]["opening"] == first[:160]
    assert "Event: 1 arrivals" in events.room_line(room) and "no arrivals" not in events.room_line(room)


def test_a_conversation_of_several_rooms_reads_its_header_through_its_catalogue_place(tmp_path):
    """rev_grok REV1 R3: a collapsed conversation's header counted all its rooms but read only the last one. Several
    rooms read through the conversation's place in the room catalogue, which a real memory_read opens at that header
    and lists every room's reader; one room keeps its direct reader; tasks without a gist read in their own room."""
    import copy
    from ouroboros import dialogue_store as ds, memory_view_conversation_forms as forms
    shared.world(tmp_path)
    focus = _event_row("aaa", "focus", 0, "Ann")
    rows = [focus]
    for n, thread in enumerate(("one", "two")):  # my reply and Ann's later word in each thread: no events, none overheard
        rows += [_event_row("harbor", thread, 1 + 2 * n, direction="out", type="presence_delivery", task_id=f"answer0{n}"),
                 _event_row("harbor", thread, 2 + 2 * n, "Ann", text=f"Ann's word in thread {thread}.")]
    for row in rows:
        ds.append(tmp_path, "chat", row)
    group = _event_group(mv.capture_memory_view(tmp_path, {"id": "root0001", "chat_id": focus["chat_id"]},
        dataclasses.replace(mv.ROLE_DEFAULTS["integrator"], room_id=str(focus["chat_id"]))))
    assert len(group["rooms"]) == 2 and not any(r["overheard"] or r.get("events") for r in group["rooms"])
    header = forms.identity_line(group)
    assert header in forms.render_group(group, set(), set(), 1) and "rows=true" not in header
    start = int(re.search(r"memory_read\(room_id='\*', start=(\d+)\)$", header)[1])
    ctx = _ctx(tmp_path, "root0001", chat=focus["chat_id"])
    catalogue = _memory_read(ctx, room_id="*", start=start)
    assert catalogue.splitlines()[2] == header
    for thread, room in zip(("one", "two"), group["rooms"]):
        assert f"memory_read(room_id='{room['room_id']}', rows=true)" in catalogue
        assert f"Ann's word in thread {thread}." in _memory_read(ctx, room_id=room["room_id"], rows=True)
    single = {**group, "rooms": group["rooms"][:1]}
    assert forms.identity_line(single).endswith(f"; memory_read(room_id='{group['rooms'][0]['room_id']}', rows=true)")
    first, last = (room["room_id"] for room in group["rooms"])
    tasked = copy.deepcopy(group)
    tasked["rooms"][0]["tasks"] = [{"task": "silent01", "line": "[t; host; task silent01] Done", "pos": 1}]
    line = forms.render_group(tasked, set(), set(), 1)
    assert f"1 tasks without a gist; memory_read(room_id='{first}', rows=true)" in line
    assert f"memory_read(room_id='{last}', rows=true)" not in line
