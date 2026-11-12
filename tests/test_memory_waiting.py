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
    assert ", 1 PASS with no findings, 1 without a recorded outcome; memory_read(room_id='late_reviews'" in shown
    older = {**snapshot.waiting, "late_reviews": {k: v for k, v in facts.items() if k not in ("clean_pass", "unrecorded")}}
    assert "PASS with no findings" not in memory_waiting.text(older), "a view captured before the counts keeps its words"
    assert "PASS with no findings" not in memory_waiting.text({**snapshot.waiting, "late_reviews": {
        **facts, "count": 0, "clean_pass": 0, "unrecorded": 0}}), "nothing waits: nothing to count"

    listed = _memory_read(ctx, room_id="late_reviews")
    assert listed.splitlines()[0] == "late reviews: 3 in 3 rooms; 2 PASS with no findings"
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
    """S4: the fourth class counts a conversation whose room where my human or I spoke holds words no page holds; a
    conversation where neither spoke does not; a page to the room's head takes it out; the listing names the act."""
    from ouroboros import chat_chain, dialogue_store as ds
    from ouroboros.memory_view_conversations import owed

    shared.world(tmp_path)
    ctx = _ctx(tmp_path, "root0001")
    asked = _github_row("repo.1", "i1", 1, "can you look?")
    answer = {**_github_row("repo.1", "i1", 2, "Looking."), "direction": "out", "type": "presence_delivery",
              "task_id": "p1", "client_message_id": "", "transport": {**asked["transport"], "delivery": {"state": "delivered"}}}
    for row in (asked, answer, _github_row("repo.2", "i9", 3, "a dependabot bump")):
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
