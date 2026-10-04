"""The memory view by role, its current room and the helper's role line (``ouroboros.memory_view``).

The role is read in ``focus_signature``'s order with the one nanny fact; the current room
is ``own_room_chat`` with Main outside the Projects (a Presence turn keeps its own
conversation); a helper's ``## Working sources`` lists exactly what its spec loads. Every
rule is pinned in both directions on a fixture with three rooms besides Main.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from types import SimpleNamespace

import pytest

from ouroboros import memory_view as mv
from tests import _memory_inventory_shared as shared

NANNY_ROUTE = {"configured_subagent": {"id": "leaf", "route": {"kind": "agent_session", "harness": "claude"}}}
API_ROUTE = {"configured_subagent": {"id": "researcher", "route": {"kind": "api_model", "model": "m"}}}
PRESENCE = {"presence": {"binding_id": "b1", "event": {"provider": "telegram", "conversation_id": "c1"}}}


# --- roles ----------------------------------------------------------------------------------------

def test_role_defaults_are_the_table_of_the_spec():
    table = {  # story, room_page, room_lanes, origin_words, live_rooms, marks, knowledge, owner_words
        "integrator": (True, True, True, True, "lines", "all", True, False),
        "consciousness": (True, False, False, False, "lines", "all", True, False),
        "presence": (True, True, True, False, "none", "room_and_global", True, False),
        "child": (True, True, False, True, "none", "room_and_global", False, True),
        "nanny": (False, True, False, True, "none", "room_and_global", False, True),
    }
    assert set(mv.ROLE_DEFAULTS) == set(table) == set(mv.ROLES)
    for role, row in table.items():
        spec = mv.ROLE_DEFAULTS[role]
        assert spec.role == role and spec.room_id is None
        assert (spec.story, spec.room_page, spec.room_lanes, spec.origin_words, spec.live_rooms, spec.marks,
                spec.knowledge, spec.owner_words) == row, role


@pytest.mark.parametrize("meta,role", [
    ({}, "integrator"),
    ({"usage_category": "consciousness"}, "consciousness"),
    ({"usage_category": "consciousness_task"}, "integrator"),  # work a wake starts is a root
    (PRESENCE, "presence"),
    ({"delegation_role": "subagent"}, "child"),
    ({"delegation_role": "subagent", **API_ROUTE}, "child"),
    ({"delegation_role": "subagent", **NANNY_ROUTE}, "nanny"),
    ({**NANNY_ROUTE, "usage_category": "consciousness"}, "nanny"),  # the nanny fact comes first
    ({"delegation_role": "subagent", **PRESENCE}, "child"),
])
def test_the_view_role_follows_the_focus_signature_order(meta, role):
    from ouroboros.knowledge import focus_signature

    assert mv.view_role({"id": "t1", "metadata": meta}) == role
    if role != "presence":  # a Presence turn is named under metadata (or by the host's _presence_* flags)
        assert mv.view_role({"id": "t1", **meta}) == role  # top-level facts count the same
    focus = focus_signature(SimpleNamespace(task_metadata=meta, task_id="t1", is_direct_chat=False))["focus"]["role"]
    assert {"root": "integrator", "main": "integrator"}.get(focus, focus) == role


def test_a_nanny_sees_no_story_and_an_api_child_does(tmp_path):
    shared.projects(tmp_path)
    nanny = mv.view_spec_for_task({"id": "n1", "chat_id": 1, "delegation_role": "subagent", **NANNY_ROUTE}, tmp_path)
    child = mv.view_spec_for_task({"id": "c1", "chat_id": 1, "delegation_role": "subagent", **API_ROUTE}, tmp_path)
    assert (nanny.role, nanny.story, nanny.knowledge, nanny.owner_words) == ("nanny", False, False, True)
    assert (child.role, child.story, child.knowledge, child.owner_words) == ("child", True, False, True)


# --- the current room -----------------------------------------------------------------------------

def test_the_view_room_is_the_task_own_room_and_main_outside_the_projects(tmp_path):
    rooms = shared.projects(tmp_path)  # binds task "bound" to alpha
    alpha, beta = str(rooms["alpha"]), str(rooms["beta"])
    cases = {
        "bound": ({"id": "bound", "chat_id": 1}, alpha),  # the binding wins over chat_id 1
        "beta": ({"id": "tb", "chat_id": rooms["beta"]}, beta),
        "main": ({"id": "tm", "chat_id": 1}, "1"),
        "hidden": ({"id": "th", "chat_id": 0}, "1"),
        "transport": ({"id": "tt", "chat_id": 777}, "1"),
        "letter": ({"id": "update-letter"}, "1"),  # no chat at all
        "child": ({"id": "kid", "chat_id": 1, "delegation_role": "subagent", "parent_task_id": "bound",
                   "root_task_id": "bound"}, alpha),
    }
    for name, (task, room) in cases.items():
        assert mv.view_spec_for_task(task, tmp_path).room_id == room, name
    presence = mv.view_spec_for_task({"id": "tp", "chat_id": 555, "metadata": PRESENCE}, tmp_path)
    assert (presence.role, presence.room_id) == ("presence", "555")  # Presence keeps its own conversation
    wake = mv.view_spec_for_task({"id": "w", "chat_id": 1, "metadata": {"usage_category": "consciousness"}}, tmp_path)
    assert wake.room_id is None


def test_the_view_room_matches_own_room_chat_inside_projects(tmp_path):
    from ouroboros.dialogue_evidence import own_room_chat

    rooms = shared.projects(tmp_path)
    for task in ({"id": "bound", "chat_id": 1}, {"id": "tb", "chat_id": rooms["beta"]},
                 {"id": "kid", "chat_id": 1, "parent_task_id": "bound", "root_task_id": "bound"}):
        ctx = SimpleNamespace(task_id=task["id"], current_chat_id=task["chat_id"],
                              task_metadata={k: v for k, v in task.items() if k.endswith("task_id")})
        assert mv.view_spec_for_task(task, tmp_path).room_id == str(own_room_chat(ctx, tmp_path))


def test_an_explicit_spec_is_taken_after_its_fields_check_and_a_bad_one_falls_back(tmp_path, caplog):
    shared.projects(tmp_path)
    task = {"id": "c1", "chat_id": 1, "delegation_role": "subagent"}
    chosen = mv.view_spec_for_task({**task, "memory_view": {"story": False, "knowledge": True}}, tmp_path)
    assert (chosen.role, chosen.story, chosen.knowledge, chosen.room_id) == ("child", False, True, "1")
    with caplog.at_level(logging.WARNING, logger="ouroboros.memory_view"):
        for bad in ({"stories": False}, {"story": "no"}, {"live_rooms": "all"}, ["story"]):
            assert mv.view_spec_for_task({**task, "memory_view": bad}, tmp_path) == dataclasses.replace(
                mv.ROLE_DEFAULTS["child"], room_id="1"), bad
    assert caplog.text.count("memory_view of task c1 refused") == 4
    events = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    refused = [event for event in events if event.get("type") == "context_memory_view_spec_refused"]
    assert [(event["task_id"], event["role"]) for event in refused] == [("c1", "child")] * 4
    assert refused[0]["reason"] == "unknown ViewSpec field(s): stories"  # an accepted spec writes no event


# --- the helper's role line -----------------------------------------------------------------------

def test_the_working_sources_line_lists_exactly_what_the_spec_loads():
    child = dataclasses.replace(mv.ROLE_DEFAULTS["child"], room_id="1")
    line = mv.working_sources_line(child)
    assert line.startswith("## Working sources\n\nLoaded above: ") and line.endswith(mv.CHILD_ROLE_TEXT)
    loaded, missing = line.split("Loaded above: ", 1)[1].split(" Not loaded: ", 1)
    assert "the top level of your story" in loaded and "the page of your parent's room" in loaded
    assert "the words of my human that caused this work" in loaded and "raw conversations" in missing
    assert "knowledge (overview, index, patterns)" in missing and "knowledge" not in loaded
    assert "shared biography" not in line
    # Each field moves its item: the line changes with the spec, never independently of it.
    for field, value, gone, now_missing in (
        ("story", False, "the top level of your story", "the top level of your story"),
        ("owner_words", False, "the words of my human that caused this work", None),
        ("knowledge", True, None, None),
    ):
        changed = mv.working_sources_line(dataclasses.replace(child, **{field: value}))
        assert changed != line, field
        new_loaded, new_missing = changed.split("Loaded above: ", 1)[1].split(" Not loaded: ", 1)
        if gone:
            assert gone not in new_loaded, field
        if now_missing:
            assert now_missing in new_missing, field
        if field == "knowledge":
            assert "knowledge (overview, index, patterns)" in new_loaded
            assert "knowledge (overview, index, patterns)" not in new_missing
    nanny = mv.working_sources_line(dataclasses.replace(mv.ROLE_DEFAULTS["nanny"], room_id="1"))
    assert "the top level of your story" in nanny.split(" Not loaded: ", 1)[1]
    # With a captured view the owner's words are named only when their block was drawn.
    words = "the words of my human that caused this work"
    drawn = mv.MemoryViewSnapshot(spec=child, store_status={"state": "active"}, frontier={}, owner_words="W")
    absent = dataclasses.replace(drawn, owner_words="")
    assert words in mv.working_sources_line(child, drawn).split(" Not loaded: ", 1)[0]
    assert words not in mv.working_sources_line(child, absent)



def test_before_activation_the_role_line_names_no_memory_item_as_loaded():
    """The line cannot disagree with the view; until the import the view holds none of my memory."""
    child = dataclasses.replace(mv.ROLE_DEFAULTS["child"], room_id="1")
    items = ("the top level of your story", "the page of your parent's room Project Alpha",
             "the words that started that work", "the memory marks of that room and global ones")
    pending = mv.MemoryViewSnapshot(spec=child, store_status={"state": "import_pending", "reason": "another importer"},
                                    frontier={}, room={"room_id": "1", "label": "Project Alpha"}, owner_words="W")
    loaded, missing = mv.working_sources_line(child, pending).split("Loaded above: ", 1)[1].split(" Not loaded: ", 1)
    assert not any(item in loaded for item in items)
    assert all(item in missing for item in items) and "(my memory is not activated yet: another importer)" in missing
    assert "the words of my human that caused this work" in loaded  # drawn from the task's rows, not the chronicle
    view = mv.render_story(pending) + mv.render_room(pending)
    assert "## My story — unavailable now (another importer)" in view and "## Marks I keep in view" not in view
    # Once active, the same items are loaded and nothing is said about activation.
    active = dataclasses.replace(pending, store_status={"state": "active"})
    loaded, missing = mv.working_sources_line(child, active).split("Loaded above: ", 1)[1].split(" Not loaded: ", 1)
    assert all(item in loaded for item in items) and "not activated" not in missing

# --- the live part on a richer installation ------------------------------------------------------
#
# Main, two Projects (alpha, beta) and a transport chat; two legacy rows before the frontier and
# twenty-three open rows after it. The first row carrying a child's lineage (pos 4) is the epoch,
# so the outgoing row at pos 3 without lineage precedes it and the one at pos 5 follows it.

from ouroboros import chat_chain  # noqa: E402
from ouroboros.chronicle_store import ChronicleStore  # noqa: E402

TRANSPORT = {"provider": "telegram", "conversation_id": "c7", "actor": {"display_name": "Ann"}}
LATE = {"settled_after_terminal": True, "reviewed_revision": "delivered", "reviewer_outputs": []}


def _ts(minute: int) -> str:
    return f"2026-09-05T10:{minute:02d}:00+00:00"


def _rows(rooms):
    a, b, m = rooms["alpha"], rooms["beta"], shared.msg
    kid = {"subagent_task_id": "kid1", "parent_task_id": "root1", "root_task_id": "root1", "delegation_role": "subagent"}
    return [
        m(_ts(2), "please look at X", client_message_id="m1"),
        m(_ts(3), "pre-epoch words", direction="out", task_id="pre1"),
        m(_ts(4), "CHILD REPORT TEXT", direction="out", task_id="kid1", **kid),
        m(_ts(5), "my answer to you", direction="out", task_id="root1"),
        m(_ts(6), "LIGHT RETELLING TEXT", direction="system", type="task_summary", task_id="root1",
          summary_kind="authored_root_summary"),
        m(_ts(7), "Failed. child", direction="system", type="task_summary", task_id="kid1", parent_task_id="root1",
          root_task_id="root1", summary_kind="terminal_result_projection", status="failed"),
        m(_ts(8), "Completed. Root task root1.", direction="system", type="task_summary", task_id="root1",
          root_task_id="root1", summary_kind="terminal_root_projection", status="completed"),
        m(_ts(9), "late critique", direction="system", type="acceptance_late_settlement", task_id="root1",
          late_evidence=LATE),
        m(_ts(10), "", direction="out", type="quiz", task_id="root4",
          quiz={"quiz_id": "q1", "question": "Which way?", "options": [{"label": "Left"}, {"label": "Right"}]}),
        m(_ts(11), "", direction="system", type="quiz_answer", client_message_id="quiz_answer:q1",
          quiz={"quiz_id": "q1", "question": "Which way?", "options": [{"label": "Left"}, {"label": "Right"}],
                "answered_index": 1}),
        m(_ts(12), "A word on my own initiative.", direction="out", type="proactive_message", task_id="root4"),
        m(_ts(13), "alpha words", chat_id=a, client_message_id="a1"),
        m(_ts(14), "alpha reply", chat_id=a, direction="out", task_id="ta1"),
        m(_ts(15), "beta words", chat_id=b, client_message_id="b1"),
        m(_ts(16), "beta reply", chat_id=b, direction="out", task_id="tb1"),
        m(_ts(17), "sent?", chat_id=777, direction="system", type="presence_delivery", task_id="pres1",
          transport={**TRANSPORT, "delivery": {"state": "failed"}, "message": {"id": "x"}}),
        m(_ts(18), "", direction="system", type="task_summary", task_id="root2", summary_kind="host_task_facts",
          status="running", result_ref={"reader": "get_task_result", "task_id": "root2"}),
        m(_ts(19), "line one\n## a heading in my reply", direction="out", task_id="root2"),
        m(_ts(20), "transport words", chat_id=777, transport=TRANSPORT),
        m(_ts(21), "Task started", direction="system", type="task_started", task_id="root3"),
        m(_ts(22), "KID5 REPORT", direction="out", task_id="kid5", subagent_task_id="kid5", parent_task_id="root5",
          root_task_id="root5", delegation_role="subagent"),
        m(_ts(23), "ROOT5 RETELLING", direction="system", type="task_summary", task_id="root5",
          summary_kind="authored_root_summary"),
        m(_ts(24), "and finally", client_message_id="m9"),
        m(_ts(25), "Task root6 was cancelled. Below is the last persisted model message: UNREVIEWED DRAFT",
          direction="system", type="cancel_receipt", task_id="root6"),
        m(_ts(26), "custody text", direction="system", type="custody_notice", task_id="root6"),
        m(_ts(27), "Project X › root6 · Cancelled", direction="system", type="project_completion_summary",
          task_id="root6", status="cancelled"),
    ]


def _install(root):
    """The installation above, activated: the legacy memory covers the two archive rows."""
    import json

    from ouroboros.utils import jsonl_generation_signature

    rooms = shared.projects(root)
    archive = root / "archive" / "chat_20260905T090000.jsonl"
    shared.append(archive, shared.msg(_ts(0), "old hello", client_message_id="m0"),
                  shared.msg(_ts(1), "old reply", direction="out", task_id="old1"))
    rows = _rows(rooms)
    shared.append(root / "logs" / "chat.jsonl", *rows)
    memory = root / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    (memory / "dialogue_blocks.json").write_text(json.dumps([{
        "ts": _ts(1), "type": "summary", "range": "10:00 - 10:01", "message_count": 2, "content": "Old Main.",
        "rooms": [{"room_id": "1", "label": "Main", "message_count": 2, "content": "Old Main talk."}]}]),
        encoding="utf-8")
    (memory / "dialogue_meta.json").write_text(json.dumps({
        "chat_log_signature": jsonl_generation_signature(archive), "last_consolidated_offset": 2}), encoding="utf-8")
    assert ChronicleStore(root).ensure_activated()["kind"] == "activation"
    return rooms, rows


def _view(root, task, **spec_fields):
    spec = mv.view_spec_for_task(task, root)
    spec = dataclasses.replace(spec, **spec_fields) if spec_fields else spec
    snapshot = mv.capture_memory_view(root, task, spec)
    return snapshot, mv.render_room(snapshot)


def _addr(row) -> str:
    return chat_chain.format_address(chat_chain.row_address(row))


def _section(text: str, title: str) -> str:
    """One ``### …`` subsection (or ``## …`` section) of a rendered block, up to the next heading of its level."""
    level = title.split(" ", 1)[0] + " "
    start = text.index(title)
    rest = text[start + len(title):]
    ends = [i for i in (rest.find("\n" + level), rest.find("\n## ") if level == "### " else -1) if i >= 0]
    return text[start:start + len(title) + (min(ends) if ends else len(rest))]


MAIN_TASK = {"id": "turn0001", "chat_id": 1}


def test_lane_one_is_people_and_my_words_verbatim_and_nothing_else_is_signed_ouroboros(tmp_path):
    rooms, rows = _install(tmp_path)
    assert ChronicleStore(tmp_path).activation()["metadata"]["lineage_epoch"]["pos"] == 4
    snapshot, text = _view(tmp_path, MAIN_TASK)
    lane1 = _section(text, "### Open conversation since")
    assert lane1.startswith("### Open conversation since 2026-09-05 10:02 (verbatim: people and my replies)")
    mine = {_addr(rows[i]) for i in (3, 8, 10, 17)}  # after the epoch: my answer, quiz, proactive word, reply
    for line in lane1.split("\n")[1:]:
        if line.startswith("["):
            label, address = line[1:].split("] ", 1)[0].split("; ")[1:3]
            assert label != "Ouroboros" or address in mine, line
    for i in (1, 2, 4, 5, 6, 7, 15, 16, 19, 20, 21, 23, 24, 25):  # unattributed, child, helper and host rows
        assert _addr(rows[i]) not in lane1, i
    assert f"[{_ts(5)}; Ouroboros; {_addr(rows[3])}] my answer to you" in lane1
    assert f"[{_ts(10)}; Ouroboros; {_addr(rows[8])}] [question q1] Which way? — options: (1) Left (2) Right" in lane1
    assert f"; Owner; {_addr(rows[9])}]" in lane1 and "Right" in lane1.split(_addr(rows[9]))[1].split("\n")[0]
    assert f"Ouroboros; {_addr(rows[10])}] A word on my own initiative." in lane1
    assert f"{_addr(rows[17])}] line one\n  ## a heading in my reply" in lane1  # never a section of its own
    assert f"Ann [provider=telegram; conversation=c7]; {_addr(rows[18])}] transport words" in lane1
    assert "pre-epoch words" not in text and "CHILD REPORT TEXT" not in text and "LIGHT RETELLING TEXT" not in text
    assert "KID5 REPORT" not in text and "ROOT5 RETELLING" not in text and "UNREVIEWED DRAFT" not in text


def test_lane_two_is_one_host_line_per_root_task_with_typed_facts_by_address(tmp_path):
    rooms, rows = _install(tmp_path)
    _snapshot, text = _view(tmp_path, MAIN_TASK)
    lane2 = _section(text, "### Task facts of this conversation").split("\n")[1:]
    by_task = {line.split("; task ", 1)[1].split("]", 1)[0]: line for line in lane2 if "; host; task " in line}
    assert set(by_task) == {"pre1", "root1", "pres1", "root2", "root3", "root5", "root6"} and len(lane2) == 7
    root1 = by_task["root1"]
    assert root1 == (f"[{_ts(4)}; host; task root1] Completed. Root task root1.; children 1 (failed 1); rows 5; "
                     f"get_task_result(task_id='root1'); {_addr(rows[2])}..{_addr(rows[7])}; "
                     f"late review evidence: {_addr(rows[7])}")
    assert by_task["root2"].startswith(f"[{_ts(18)}; host; task root2] host facts for root2: status=running; "
                                       "result: get_task_result(task_id=root2); rows 1;")
    assert f"delivery failed: {_addr(rows[15])}" in by_task["pres1"]
    assert by_task["root3"].startswith(f"[{_ts(21)}; host; task root3] running or unreported; last row task_started "
                                       f"by host, 12 chars, {_addr(rows[19])}")
    # A running root with a child: its status comes from fields, never from the last row's text.
    assert by_task["root5"].startswith(f"[{_ts(22)}; host; task root5] running or unreported; last row task_summary "
                                       f"by Light (legacy retelling), 15 chars, {_addr(rows[21])}; children 1;")
    assert "outgoing, author not recorded" in by_task["pre1"]  # before the epoch: not mine
    assert by_task["root6"] == (f"[{_ts(25)}; host; task root6] Project X › root6 · Cancelled; rows 3; "
                                f"get_task_result(task_id='root6'); {_addr(rows[23])}..{_addr(rows[25])}; "
                                f"cancel receipt: {_addr(rows[23])}; custody notice: {_addr(rows[24])}")
    joined = "\n".join(lane2)
    assert '{"' not in joined and "Delivery details" not in joined and "Late review evidence:" not in joined


def test_an_old_task_summary_without_a_kind_is_model_prose_and_never_a_status(tmp_path):
    """The old writer's task_summary rows carry no summary kind and were written by a model: the lane-2 line
    takes its status from fields; a Project's completion row, written by the host without a kind, stays one."""
    rooms, rows = _install(tmp_path)
    shared.append(tmp_path / "logs" / "chat.jsonl", shared.msg(_ts(28), "OLD MODEL PROSE OF ROOT7", direction="system",
                                                             type="task_summary", task_id="root7", status="completed"))
    _snapshot, text = _view(tmp_path, MAIN_TASK)
    lane2 = _section(text, "### Task facts of this conversation").split("\n")[1:]
    root7 = next(line for line in lane2 if "; host; task root7]" in line)
    assert root7.startswith(f"[{_ts(28)}; host; task root7] running or unreported; last row task_summary by ")
    assert "OLD MODEL PROSE" not in text
    root6 = next(line for line in lane2 if "; host; task root6]" in line)
    assert root6.startswith(f"[{_ts(25)}; host; task root6] Project X › root6 · Cancelled; ")

def test_a_page_seals_rows_only_in_its_room_and_a_row_in_two_rooms_stays_open_in_the_other(tmp_path):
    from ouroboros.tools.chronicle import page_covers

    rooms, rows = _install(tmp_path)
    store = ChronicleStore(tmp_path)

    def seal(room, row):
        covers = page_covers(tmp_path, room, from_addr=_addr(row), to_addr=_addr(row))["covers"]
        assert store.publish_page(room_id=room, text="sealed", covers=covers, author=shared.MIND).ok

    seal("1", rows[0])
    seal("777", rows[18])
    _snapshot, text = _view(tmp_path, MAIN_TASK)
    lane1 = _section(text, "### Open conversation since")
    assert _addr(rows[0]) not in lane1 and _addr(rows[3]) in lane1  # inside the page / its neighbour
    assert lane1.startswith("### Open conversation since 2026-09-05 10:03")
    assert _addr(rows[18]) in lane1  # sealed in the transport room, still open in Main
    wake, wake_text = _view(tmp_path, {"id": "w1", "chat_id": 1, "metadata": {"usage_category": "consciousness"}})
    transport = next(room for room in wake.live_rooms if room["room_id"] == "777")
    assert (transport["rows"], transport["people"]) == (1, 0)


def test_live_rooms_are_rooms_with_open_rows_or_open_notes_and_carry_no_words_by_default(tmp_path):
    from ouroboros.tools.chronicle import page_covers

    rooms, rows = _install(tmp_path)
    beta = str(rooms["beta"])
    store = ChronicleStore(tmp_path)
    _snapshot, text = _view(tmp_path, MAIN_TASK)
    live = _section(text, "## Live rooms")
    header = f"### Project Beta [chat_id={beta}] — open 2026-09-05 10:15 → 2026-09-05 10:16; people 1, mine 1, task facts 0"
    assert header in live and f"memory_read(room_id='{beta}', rows=true)" in live
    assert "beta words" not in live and "alpha words" not in live  # other rooms' people only by count
    _snap, worded = _view(tmp_path, MAIN_TASK, live_rooms="lines_with_words")
    assert "beta words" in _section(worded, "## Live rooms") and "beta reply" not in worded
    words_only = page_covers(tmp_path, beta, from_addr=_addr(rows[13]), to_addr=_addr(rows[13]))["covers"]
    assert store.publish_page(room_id=beta, text="beta asked", covers=words_only, author=shared.MIND).ok
    note = store.write_note(room_id=beta, task_id="tb1", text="Beta waits for the owner.", author=shared.MIND)
    _snap, noted = _view(tmp_path, MAIN_TASK)
    assert f"### Project Beta [chat_id={beta}] — open 2026-09-05 10:16 → 2026-09-05 10:16; people 0, mine 1" in noted
    assert f"note {note.record['id']} by root on " in noted and "  Beta waits for the owner." in noted
    by_task = page_covers(tmp_path, beta, task_ids=["tb1"])["covers"]
    assert f"note:{note.record['id']}" in by_task["rows"]  # the task's page takes its note too
    assert store.publish_page(room_id=beta, text="beta answered", covers=by_task, author=shared.MIND).ok
    sealed = _view(tmp_path, MAIN_TASK)[1]
    assert f"chat_id={beta}]" not in sealed and "Beta waits for the owner." not in sealed  # nothing open: not live
    store.write_note(room_id=beta, task_id="tb1", text="One more thought on beta.", author=shared.MIND)
    _snap, again = _view(tmp_path, MAIN_TASK)
    assert f"### Project Beta [chat_id={beta}] — no open rows; my notes not yet sealed: 1" in again


def test_this_room_has_its_head_retold_records_origin_words_and_notes(tmp_path):
    rooms, rows = _install(tmp_path)
    alpha = str(rooms["alpha"])
    store = ChronicleStore(tmp_path)
    snapshot, text = _view(tmp_path, {"id": "ra1", "chat_id": rooms["alpha"]})
    head = store.room_head(alpha)
    assert f"## This room (Project Alpha [chat_id={alpha}]) — head {head}" in text
    origin = _section(text, "### Words that started this work (retention-proof)")
    assert f"[2026-09-01T00:05:00+00:00; owner; chat 1 / origin-a] {shared.ORIGIN}" in origin
    # Once the origin row itself is open in the room, it is read there, not twice.
    shared.append(tmp_path / "logs" / "chat.jsonl", shared.msg("2026-09-01T00:05:00+00:00", shared.ORIGIN,
                                                               client_message_id="origin-a"))
    _snap, again = _view(tmp_path, {"id": "ra1", "chat_id": rooms["alpha"]})
    assert "### Words that started this work" not in again and shared.ORIGIN in _section(again, "### Open conversation")
    main_snapshot, main_text = _view(tmp_path, MAIN_TASK)
    whole = "#### legacy-b00-r1 — 2026-09-05 10:00 → 2026-09-05 10:01"
    # Main's integrator reads its first-block retelling whole in the story, so the room page does not repeat it;
    # a view without the story reads it on the room page.
    assert whole + " — Main — 2 rows retold in 14 chars\n  Old Main talk." in mv.render_story(main_snapshot)
    assert "### Retold before the update" not in main_text and "Old Main talk." not in main_text
    retold = _section(_view(tmp_path, MAIN_TASK, story=False)[1],
                      "### Retold before the update (helper retelling, not lived)")
    assert whole + "\n  Old Main talk." in retold
    assert "### Words that started this work" not in main_text  # Main is no Project
    note = store.write_note(room_id="1", task_id="root1", text="Keep root1 in mind.", author=shared.MIND)
    _snap, noted = _view(tmp_path, MAIN_TASK)
    assert f"note {note.record['id']} by root on " in _section(noted, "### My notes not yet sealed")


def test_main_room_page_shows_the_room_less_retellings_whole_to_its_integrator_and_a_pointer_elsewhere(tmp_path):
    """The flat summary and a room-less era predate rooms and were Main's memory: whole in
    Main's room page, under their own label, for Main's integrator and for a child that starts with the top
    level of the life account; a pointer in the story, and nothing more, for another room or consciousness;
    nothing at all for a nanny in Main, which carries no story."""
    import json

    rooms = shared.world(tmp_path, flat="The retired flat summary of everything.", activate=False)
    path = tmp_path / "memory" / "dialogue_blocks.json"
    blocks = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps([*blocks, {"type": "era", "message_count": 0, "content": "A room-less era."}]),
                    encoding="utf-8")
    assert ChronicleStore(tmp_path).ensure_activated()["kind"] == "activation"
    main = mv.capture_memory_view(tmp_path, MAIN_TASK, mv.view_spec_for_task(MAIN_TASK, tmp_path))
    retold = _section(mv.render_room(main), "### Retold before the update (helper retelling, not lived)")
    for words in ("  The retired flat summary of everything.", "  A room-less era."):
        assert retold.index(words) < retold.index("  Main was quiet.")  # older than the rooms: first
    era = "#### legacy-b02-rlegacy — period known from the retelling text only — Unknown provenance [legacy mixed record]"
    assert era + "\n  A room-less era." in retold
    assert "#### legacy-b01-r1 — 2026-09-02 00:00 → 2026-09-02 00:03 (block period)\n" in retold  # Main's own: no label
    assert "Main talk." not in retold  # the first block is whole in the story, not repeated here
    # By address (F4) the line says what it holds.
    short = mv.render_room(main, mv.FloorLevel(addressed=(("F4", ("legacy-b02-rlegacy",)),)))
    assert era + " — 16 chars — memory_read(node_id='legacy-b02-rlegacy')" in short and "  A room-less era." not in short
    story = mv.render_story(main)
    assert "The retired flat summary" not in story and "memory_read(node_id='legacy-flat-" in story
    assert "memory_read(node_id='legacy-b02-rlegacy')" in story and "A room-less era." not in story
    others = {"bound": {"id": "bound", "chat_id": 1},  # bound to alpha
              "nanny": {"id": "n1", "chat_id": 1, "delegation_role": "subagent", "root_task_id": "root1", **NANNY_ROUTE},
              "wake": {"id": "w1", "chat_id": 1, "metadata": {"usage_category": "consciousness"}}}
    for name, task in others.items():
        snapshot = mv.capture_memory_view(tmp_path, task, mv.view_spec_for_task(task, tmp_path))
        room = mv.render_room(snapshot)
        assert "A room-less era." not in room and "The retired flat summary" not in room, name
        story = mv.render_story(snapshot)
        if name == "nanny":  # a nanny carries no story at all, so not even the pointer
            assert story == "" and "legacy-b02-rlegacy" not in room, name
        else:
            assert "memory_read(node_id='legacy-b02-rlegacy')" in story, name
        assert snapshot.spec.room_id == {"bound": str(rooms["alpha"]), "nanny": "1", "wake": None}[name]
    # A child starts with the top level of the life account, so Main's room-less retellings stand whole on its page.
    kid = {"id": "kid1", "chat_id": 1, "delegation_role": "subagent", "root_task_id": "root1"}
    child = mv.capture_memory_view(tmp_path, kid, mv.view_spec_for_task(kid, tmp_path))
    room = mv.render_room(child)
    assert "## This room (Main)" in room and "  Main talk." in room
    assert "  A room-less era." in room and "  The retired flat summary of everything." in room


def test_each_role_sees_its_parts_of_the_live_view(tmp_path):
    rooms, rows = _install(tmp_path)
    store = ChronicleStore(tmp_path)
    store.mark({"kind": "task", "task_id": "root1"}, "Watch root1", shared.MIND, room_id="1")
    store.mark({"kind": "task", "task_id": "ta1"}, "Watch alpha", shared.MIND, room_id=str(rooms["alpha"]))
    store.mark({"kind": "task", "task_id": "x"}, "For everyone", shared.MIND, room_id="1", scope="global")
    words = {"governing_owner_words": [{"text": "Find the cause", "source": "initial_user", "task_id": "root1"}]}
    main, main_text = _view(tmp_path, MAIN_TASK)
    assert [mark["text"] for mark in main.marks] == ["Watch root1", "Watch alpha", "For everyone"]
    wake, wake_text = _view(tmp_path, {"id": "w1", "chat_id": 1, "metadata": {"usage_category": "consciousness"}})
    assert "## This room" not in wake_text and "- [global; Main] For everyone" in wake_text
    assert "### Main — open 2026-09-05 10:02 → 2026-09-05 10:27; people " in wake_text  # Main is one line
    assert "please look at X" not in wake_text and "transport words" not in wake_text  # no people's words
    child_task = {"id": "kid9", "chat_id": 1, "delegation_role": "subagent", "root_task_id": "root1",
                  "metadata": words}
    child, child_text = _view(tmp_path, child_task)
    assert mv.render_story(child).startswith("## My story\n") and "## This room (Main) — head " in child_text
    assert "### Retold before the update" in child_text and "### Open conversation" not in child_text
    assert "## Live rooms" not in child_text and "Watch alpha" not in child_text and "Watch root1" in child_text
    assert child_text.startswith("## Words of my human that caused this work (verbatim)\n")
    assert "Find the cause" in child_text and "Find the cause" not in main_text
    nanny_task = {**child_task, "id": "nan9", "configured_subagent": {"route": {"kind": "agent_session"}}}
    nanny, nanny_text = _view(tmp_path, nanny_task)
    assert mv.render_story(nanny) == "" and "Find the cause" in nanny_text and "### Retold before the update" in nanny_text
    assert "### Pages of this room" not in nanny_text and "### Pages of this room" not in child_text  # nothing sealed yet
    line = mv.working_sources_line(child.spec, child)
    assert "the words of my human that caused this work" in line and "the page of your parent's room Main" in line
    assert "the words that started that work" not in line  # Main holds no Project origin to show
    alpha_child = mv.working_sources_line(*(lambda s: (s.spec, s))(_view(tmp_path, {**child_task, "id": "kid8",
                                                                                     "chat_id": rooms["alpha"]})[0]))
    assert "the words that started that work" in alpha_child


def test_a_helpers_owner_words_are_indented_and_survive_local_compaction(tmp_path):
    """A line of the owner's own that starts with '## ' stays inside the words; the local model's compaction
    keeps the section, as it keeps this room and the marks (the owner's words never degrade)."""
    from ouroboros.llm_local import _compact_local_text, _split_markdown_sections
    from ouroboros.owner_words import render_owner_words

    _install(tmp_path)
    said = "Find the cause\n## not a heading of the request"
    child_task = {"id": "kid9", "chat_id": 1, "delegation_role": "subagent", "root_task_id": "root1", "metadata": {
        "governing_owner_words": [{"text": said, "source": "initial_user", "task_id": "root1"}]}}
    _child, child_text = _view(tmp_path, child_task)
    assert "\n  Find the cause\n  ## not a heading of the request" in child_text
    titles = [title for title, _body in _split_markdown_sections(child_text)[1]]
    assert "not a heading of the request" not in titles
    assert titles[0] == "Words of my human that caused this work (verbatim)"
    compacted = _compact_local_text(child_text, "dynamic")
    assert "## not a heading of the request" in compacted and "Compacted for local-model context" not in \
        compacted.split("## Marks I keep in view")[0]
    # The indent is the view's: every other audience prints the words as recorded.
    rows = [{"text": said, "source": "initial_user", "task_id": "root1"}]
    assert "\nFind the cause\n## not a heading" in render_owner_words(rows, audience="reviewer")

def test_before_activation_the_room_is_one_line_with_a_reader_that_works_without_the_chronicle(tmp_path, monkeypatch):
    shared.projects(tmp_path)
    monkeypatch.setattr(ChronicleStore, "ensure_activated",
                        lambda self, **kw: {"kind": "import_pending", "reason": "legacy_memory_lock_busy"})
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert text == ("## This room (Main)\n\nOpen conversation unavailable until my memory is activated "
                    "(legacy_memory_lock_busy); read it: chat_history(count=100) — every room, newest first; this "
                    "room is chat_id 1.")
    assert snapshot.live_rooms == () and snapshot.marks == ()


def test_the_view_and_memory_read_print_a_row_with_one_grammar(tmp_path):
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.registry import ToolContext

    rooms, rows = _install(tmp_path)
    snapshot, _text = _view(tmp_path, MAIN_TASK)
    read = _memory_read(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="turn0001", current_chat_id=1),
                        rows=True)
    assert snapshot.room["lane1"]
    for item in snapshot.room["lane1"]:
        assert item["line"].replace("\n" + mv.INDENT, "\n") in read, item["line"]


def test_a_nanny_keeps_its_rooms_sealed_pages_on_its_room_page_and_a_child_reads_them_in_its_story(tmp_path):
    """Regression: a nanny starts without the story but with its room's page, the
    conversation it works in; once the room is sealed into pages, those pages are the room page — never
    \"Nothing open, retold or noted\". A child keeps them in its story, not twice."""
    from ouroboros.tools.chronicle import page_covers

    _rooms, rows = _install(tmp_path)
    covers = page_covers(tmp_path, "1", from_addr=_addr(rows[0]), to_addr=_addr(rows[1]))["covers"]
    sealed = ChronicleStore(tmp_path).publish_page(room_id="1", text="I sealed the opening of Main.", covers=covers,
                                                   author=shared.MIND)
    assert sealed.ok, sealed
    base = {"id": "kid9", "chat_id": 1, "delegation_role": "subagent", "root_task_id": "root1"}
    nanny, nanny_text = _view(tmp_path, {**base, "id": "nan9", "configured_subagent": {"route": {"kind": "agent_session"}}})
    assert mv.render_story(nanny) == ""
    assert "I sealed the opening of Main." in _section(nanny_text, "### Pages of this room")
    child, child_text = _view(tmp_path, base)
    assert "I sealed the opening of Main." in mv.render_story(child) and "### Pages of this room" not in child_text
    # A helper's draft reaches the nanny with its evidence, as the story prints it: who drafted it,
    # the host stamp of its tasks and the verified words it quotes — never as unattributed prose.
    from ouroboros.chronicle_import import row_lineage
    from ouroboros.tools.chronicle import _quote_resolver

    covers = page_covers(tmp_path, "1", from_addr=_addr(rows[2]), to_addr=_addr(rows[2]))["covers"]
    quote = {"address": _addr(rows[0]), "text": "please look", "speaker": "human"}
    draft = ChronicleStore(tmp_path).publish_page(
        room_id="1", text="A helper says the work finished.", covers=covers, author={"kind": "helper", "route": "light"},
        quotes=[quote], host_stamp={"tasks": [{"task_id": "t9", "status": "failed", "outcome_phase": "failed"}]},
        quote_resolver=_quote_resolver(tmp_path, row_lineage(tmp_path)))
    assert draft.ok, draft
    _nanny, nanny_text = _view(tmp_path, {**base, "id": "nan9", "configured_subagent": {"route": {"kind": "agent_session"}}})
    page = _section(nanny_text, "### Pages of this room")
    assert "A helper says the work finished." in page and f"- quote (human, {quote['address']}): {quote['text']}" in page
    assert "(draft by a helper (Light), not yet accepted or rejected by me)" in page and "failed 1" in page
    # Order stays the room's record order (F4 takes the room page oldest first): a page folded later into a
    # part stands after the older top-level pages, not before them.
    store = ChronicleStore(tmp_path)
    covers = page_covers(tmp_path, "1", from_addr=_addr(rows[3]), to_addr=_addr(rows[3]))["covers"]
    newer = store.publish_page(room_id="1", text="A newer page.", covers=covers, author=shared.MIND)
    part = store.publish_part(room_id="1", text="A part over the newer page.", member_ids=[newer.record["id"]],
                              author=shared.MIND, expected_sequence=store.room_head("1"))
    assert newer.ok and part.ok, (newer, part)
    nanny, _text = _view(tmp_path, {**base, "id": "nan9", "configured_subagent": {"route": {"kind": "agent_session"}}})
    assert [item["id"] for item in nanny.room["under_parts"]] == [
        sealed.record["id"], draft.record["id"], newer.record["id"], part.record["id"]]


def test_a_storyless_room_page_keeps_story_order_when_an_older_period_is_sealed_later(tmp_path):
    """Regression: a nanny's room page is in story (stream) order, not publication order,
    so F4 (oldest first) addresses the older period first even when its page was published after a newer one."""
    from ouroboros.tools.chronicle import page_covers

    _rooms, rows = _install(tmp_path)
    store = ChronicleStore(tmp_path)

    def seal(first, last, text):
        covers = page_covers(tmp_path, "1", from_addr=_addr(rows[first]), to_addr=_addr(rows[last]))["covers"]
        result = store.publish_page(room_id="1", text=text, covers=covers, author=shared.MIND)
        assert result.ok, result
        return result.record["id"]

    newer = seal(3, 3, "The newer period.")
    older = seal(0, 1, "The older period, sealed later.")
    nanny, text = _view(tmp_path, {"id": "nan9", "chat_id": 1, "delegation_role": "subagent", "root_task_id": "root1",
                                   "configured_subagent": {"route": {"kind": "agent_session"}}})
    assert [item["id"] for item in nanny.room["under_parts"]] == [older, newer]
    page = _section(text, "### Pages of this room")
    assert page.index("The older period, sealed later.") < page.index("The newer period.")
