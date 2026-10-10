"""Block B of the memory view, ``## My story`` (``ouroboros.memory_view.render_story``).

The retold old memory comes first, every unfolded record whole whatever its block (no block
is privileged; the physical floor, F3, is what turns a room's records into one line), then
my pages and parts of every room in stream order (never by publication time); a record
folded into a part shows only through the part, and the mind's corrections and rejections
of a part's members stand under it. What is folded is ``memory_inventory``'s one rule. The
block is byte-identical for every focus that carries a story, a child included, and depends
on no capture time, task, room or chat row. Each rule is pinned in both directions on a
fixture with three rooms besides Main and twenty rows.
"""
from __future__ import annotations

import re

from ouroboros import chat_chain
from ouroboros import memory_view as mv
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared

MAIN = {"id": "turn0001", "chat_id": 1}
KID = {"id": "kid00001", "chat_id": 1, "delegation_role": "subagent"}
HELPER = {"kind": "helper", "route": "configured-light"}
OLD = "### Old memory retold by a helper before the update (not lived; "
WHOLE = OLD + "whole where the window holds it, else read by id)"


def _snapshot(root, task=MAIN):
    return mv.capture_memory_view(root, task, mv.view_spec_for_task(task, root))


def _story(root, task=MAIN) -> str:
    return mv.render_story(_snapshot(root, task))


def _page(root, room, first, last, *, author=shared.MIND, text=None):
    from ouroboros.tools.chronicle import page_covers

    addresses = {pos: address for address, _row, pos in chat_chain.iter_rows(root)}
    covers = page_covers(root, room, from_addr=addresses[first], to_addr=addresses[last])["covers"]
    result = ChronicleStore(root).publish_page(room_id=room, text=text or f"Page {room} {first}-{last}.",
                                               covers=covers, author=author)
    assert result.ok, result
    return result.record["id"]


def _part(root, room, members, text="A part of my story."):
    store = ChronicleStore(root)
    result = store.publish_part(room_id=room, text=text, member_ids=members, author=shared.MIND,
                                expected_sequence=store.room_head(room))
    assert result.ok, result
    return result.record["id"]


def _headers(text):
    return [line for line in text.split("\n") if line.startswith("### ")]


def _fold_block_zero(root, rooms):
    _page(root, str(rooms["alpha"]), 2, 5)
    _page(root, "777", 4, 4)
    _page(root, "1", 0, 5)


def test_pointers_come_first_then_pages_of_all_rooms_by_stream_position_not_publication(tmp_path):
    rooms = shared.world(tmp_path)
    alpha, beta = str(rooms["alpha"]), str(rooms["beta"])
    # Published newest stream first: sequence and record time run against the stream.
    late = _page(tmp_path, beta, 14, 14)
    early = _page(tmp_path, "1", 10, 12)
    middle = _page(tmp_path, alpha, 13, 13)
    text = _story(tmp_path)
    headers = _headers(text)
    assert headers[0] == WHOLE
    assert [header.rsplit(" ", 1)[1] for header in headers[1:]] == [early, middle, late]
    assert headers[1].startswith("### Main · 2026-09-03 00:00 → 2026-09-03 00:02 · page ")
    assert headers[3].startswith(f"### Project Beta [chat_id={beta}] · ")
    whole = "#### legacy-b00-r1 — 2026-09-01 00:00 → 2026-09-01 00:05 — Main — 4 rows retold in 10 chars\n  Main talk."
    later = ("#### legacy-b01-r1 — 2026-09-02 00:00 → 2026-09-02 00:03 (block period) — Main — no row of this room in "
             "that period; retold in 15 chars\n  Main was quiet.")
    assert whole in text and text.index(whole) < text.index(later) < text.index(headers[1])
    assert "memory_read(node_id='legacy-" not in text  # whole: no pointer line repeats a record
    # A child reads the same story, byte for byte: no block is privileged for anyone.
    assert _story(tmp_path, KID) == text


def test_a_pointers_period_is_its_rooms_own_rows_and_without_rows_the_marked_block_period(tmp_path):
    """A retold record names the period of its room's rows, not its block's; a room with no row in
    its block names the block's period, marked as such. The story (whole records, every focus) and
    the room page of a storyless focus (a nanny) say the same; under the floor (F3) the pointer line
    of a room names the same periods and sizes."""
    rooms = shared.world(tmp_path)
    alpha = rooms["alpha"]
    text = _story(tmp_path, KID)
    assert text == _story(tmp_path)
    # The transport's one row of block zero is at 00:04, Alpha's rows of block one end at 00:01.
    assert ("#### legacy-b00-r777 — 2026-09-01 00:04 → 2026-09-01 00:04 — Transport — 1 row retold in 17 chars\n"
            "  A transport line.") in text
    assert (f"#### legacy-b01-r{alpha} — 2026-09-02 00:00 → 2026-09-02 00:01 — Alpha — 2 rows retold in 13 chars\n"
            "  Alpha worked.") in text
    # Main's rows span the whole first block: the same minutes, and no mark.
    assert "#### legacy-b00-r1 — 2026-09-01 00:00 → 2026-09-01 00:05 — Main — 4 rows retold in 10 chars\n  Main talk." in text
    # Main has no row in the second block: that block's period, marked, and what its address holds.
    quiet = "2026-09-02 00:00 → 2026-09-02 00:03 (block period)"
    assert f"#### legacy-b01-r1 — {quiet} — Main — no row of this room in that period; retold in 15 chars\n  Main was quiet." in text
    assert "0 rows retold" not in text and text.count("(block period)") == 1
    # The floor's one line per room names the same facts: the pointer form of each record.
    snapshot = _snapshot(tmp_path)
    lines = mv.render_story(snapshot, mv.FloorLevel((("F3", ("1", "777", str(alpha))),))).split("\n")
    assert ("- Main; 2026-09-01 00:00 → 2026-09-01 00:05; 2026-09-02 00:00 → 2026-09-02 00:03 (block period); 2 retold "
            "records in 25 chars: legacy-b00-r1, legacy-b01-r1; memory_read(node_id=<id>) reads each") in lines
    assert "- Transport; 2026-09-01 00:04 → 2026-09-01 00:04; 1 retold record in 17 chars: legacy-b00-r777; memory_read(node_id=<id>) reads each" in lines
    # A nanny carries no story: its room page shows its room's retold records whole, under the same periods.
    nanny = {"id": "nan00001", "chat_id": 1, "delegation_role": "subagent",
             "configured_subagent": {"route": {"kind": "agent_session"}}}
    room = mv.render_room(_snapshot(tmp_path, nanny))
    assert f"#### legacy-b01-r1 — {quiet}\n  Main was quiet." in room
    assert "#### legacy-b00-r1 — 2026-09-01 00:00 → 2026-09-01 00:05\n  Main talk." in room
    # A focus with a story never repeats on its room page what the story shows whole.
    for task in (MAIN, KID):
        assert "legacy-b0" not in mv.render_room(_snapshot(tmp_path, task)), task["id"]


def test_a_room_less_pointer_names_the_old_writers_count_and_length_never_zero_rows(tmp_path):
    """A record without rows of its room (a mixed era) says what its address holds — the old writer's
    message count and the retelling's length — instead of «0 rows retold»; with rows it names the rows."""
    import json

    rooms = shared.world(tmp_path, activate=False)
    path = tmp_path / "memory" / "dialogue_blocks.json"
    blocks = json.loads(path.read_text(encoding="utf-8"))
    blocks[1] = {"type": "era", "range": "2026-09-02 00:00 - 00:03", "message_count": 4,
                 "content": "A mixed era of four messages."}  # the same count: the ranges stay exact
    path.write_text(json.dumps(blocks), encoding="utf-8")
    assert ChronicleStore(tmp_path).ensure_activated()["kind"] == "activation"
    snapshot = _snapshot(tmp_path)
    lines = mv.render_story(snapshot).split("\n")
    era = [line for line in lines if "legacy-b01-rlegacy" in line]
    assert era == ["#### legacy-b01-rlegacy — 2026-09-02 00:00 → 2026-09-02 00:03 (block period) — Unknown provenance "
                   "[legacy mixed record] — 4 messages retold (the old writer's count) in 29 chars"]
    assert lines[lines.index(era[0]) + 1] == "  A mixed era of four messages."
    short = mv.render_story(snapshot, mv.FloorLevel((("F3", ("legacy",)),))).split("\n")
    assert ("- Unknown provenance [legacy mixed record]; 2026-09-02 00:00 → 2026-09-02 00:03 (block period); 1 retold "
            "record in 29 chars: legacy-b01-rlegacy; memory_read(node_id=<id>) reads each") in short
    assert ("#### legacy-b00-r1 — 2026-09-01 00:00 → 2026-09-01 00:05 — Main — 4 rows retold in 10 chars"
            ) in lines  # rows of its room: the rows, not the old count
    assert f"legacy-b01-r{rooms['alpha']}" not in "\n".join(lines)


def test_a_part_shows_its_members_do_not_and_corrections_of_members_stand_under_it(tmp_path):
    shared.world(tmp_path)
    first, second = _page(tmp_path, "1", 10, 11), _page(tmp_path, "1", 12, 12)
    plain = _part(tmp_path, "1", [first, second])
    text = _story(tmp_path)
    assert f"part {plain}" in text and first not in text and second not in text
    assert "correction by me of" not in text  # no correction, no line
    store = ChronicleStore(tmp_path)
    assert store.correct(first, "The first page, said right.", shared.MIND,
                         expected_sequence=store.room_head("1")).ok
    corrected = _story(tmp_path)
    assert f"- correction by me of {first}:\n  The first page, said right." in corrected
    assert corrected.count("correction by me of") == 1 and second not in corrected  # no cascade


def test_a_correction_of_a_page_folded_twice_stands_under_the_surviving_part(tmp_path):
    """Regression: a correction reaches through nested parts. page -> part -> part, then the
    page is corrected: the outer part shows it; the inner part (folded) does not appear."""
    shared.world(tmp_path)
    page = _page(tmp_path, "1", 10, 11)
    inner = _part(tmp_path, "1", [page])
    outer = _part(tmp_path, "1", [inner], text="An older part of my story.")
    assert "correction by me of" not in _story(tmp_path)
    store = ChronicleStore(tmp_path)
    assert store.correct(page, "The page, said right.", shared.MIND, expected_sequence=store.room_head("1")).ok
    text = _story(tmp_path)
    assert f"part {outer}" in text and f"part {inner}" not in text
    assert f"- correction by me of {page}:\n  The page, said right." in text and text.count("correction by me of") == 1


def test_the_one_folded_rule_removes_a_pointer_only_when_every_row_is_sealed_and_counts_the_block(tmp_path):
    rooms = shared.world(tmp_path)
    before = _story(tmp_path)
    assert "folded 0 of 2 blocks" in before and "#### legacy-b00-r777 — " in before
    _page(tmp_path, str(rooms["alpha"]), 2, 5)
    _page(tmp_path, "1", 0, 1)  # Main's rows of block zero are 0, 1, 4 and 5: a partial page folds nothing
    partial = _story(tmp_path)
    assert "#### legacy-b00-r1 — " in partial and "folded 0 of 2 blocks" in partial
    assert f"legacy-b00-r{rooms['alpha']}" not in partial and "Alpha began." not in partial
    _page(tmp_path, "1", 4, 5)
    _page(tmp_path, "777", 4, 4)
    folded = _story(tmp_path)
    assert "legacy-b00-" not in folded and "folded 1 of 2 blocks" in folded
    # A record without rows folds only through a part, and the part then tells that period.
    assert "#### legacy-b01-r1 — " in folded and "  Main was quiet." in folded
    part = _part(tmp_path, "1", ["legacy-b01-r1"], text="Main was quiet that day.")
    assert "legacy-b01-r1 — " not in _story(tmp_path) and f"part {part}" in _story(tmp_path)
    assert "Main was quiet." not in _story(tmp_path) and "Main was quiet that day." in _story(tmp_path)


def test_the_story_always_ends_with_the_pages_line_and_the_progress_line_only_while_folding(tmp_path):
    """The story's last line is permanent: my pages (and the date I last sealed one), a helper's drafts,
    the parts — zeros say zeros, a helper's draft is not mine, a page folded into a part still counts.
    The folding progress is a separate line that leaves when the old memory is all folded."""
    rooms = shared.world(tmp_path)
    lines = _story(tmp_path).split("\n")
    assert lines[-2] == "Pages sealed by me: 0; drafted by a helper: 0; parts: 0." and lines[-3] == ""
    assert lines[-1].startswith("Story status: the helper retelling is folded 0 of 2 blocks; 6 retold records are still open "
                                "(12 rows,") and "helper route " in lines[-1] and "pages sealed" not in lines[-1]
    _page(tmp_path, str(rooms["alpha"]), 6, 7, author=HELPER)
    assert _story(tmp_path).split("\n")[-2] == "Pages sealed by me: 0; drafted by a helper: 1; parts: 0."
    _fold_block_zero(tmp_path, rooms)
    _page(tmp_path, str(rooms["beta"]), 8, 9)
    _part(tmp_path, "1", ["legacy-b01-r1"])
    _part(tmp_path, "1", [_page(tmp_path, "1", 10, 11)])
    done = _story(tmp_path)
    assert "Story status:" not in done and "Old memory retold" not in done  # the visible end of the transition
    store = ChronicleStore(tmp_path)
    latest = max(str(record["ts"])[:10] for record in store.records(kinds=("page",)) if record["author"]["kind"] == "mind")
    assert done.split("\n")[-1] == f"Pages sealed by me: 5 (latest {latest}); drafted by a helper: 1; parts: 2."
    # All folded through parts alone: no page yet, and the line says so instead of falling silent.
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    quiet_rooms = shared.world(quiet)
    for unit in ("b00-r1", f"b00-r{quiet_rooms['alpha']}", "b00-r777", "b01-r1", f"b01-r{quiet_rooms['alpha']}",
                 f"b01-r{quiet_rooms['beta']}"):
        _part(quiet, unit.split("-r")[1], [f"legacy-{unit}"])
    assert "Story status:" not in _story(quiet)
    assert _story(quiet).split("\n")[-1] == "Pages sealed by me: 0; drafted by a helper: 0; parts: 6."
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    shared.world(fresh, legacy=False)
    assert _story(fresh).split("\n")[-3:] == [mv._STORY_INTRO, "", "Pages sealed by me: 0; drafted by a helper: 0; parts: 0."]
    assert "Story status:" not in _story(fresh) and "No page or part is sealed yet." not in _story(fresh)


def test_a_helper_refusal_receipt_is_one_line_after_the_status_and_absent_without_one(tmp_path):
    rooms = shared.world(tmp_path)
    assert "A helper could not fold" not in _story(tmp_path)
    unit = f"legacy-b01-r{rooms['beta']}"
    receipt = {"input_sha256": "e" * 64, "kind": "context_overflow", "at": "2026-10-03T00:00:00+00:00",
               "response_ref": {"path": "x.md", "read": {"tool": "read_file", "arguments": {
                   "root": "runtime_data", "path": "task_results/fallback/x.md"}}}}
    assert ChronicleStore(tmp_path).publish([], scan_state={"fallback_refusals": {unit: receipt}}).ok
    lines = _story(tmp_path).split("\n")
    refusal = [line for line in lines if line.startswith("A helper could not fold")]
    assert refusal == ["A helper could not fold: Beta; 2026-09-02 00:02 → 2026-09-02 00:03; context_overflow; "
                       "its answer: read_file(root='runtime_data', path='task_results/fallback/x.md')"]
    assert lines.index(refusal[0]) == next(i for i, line in enumerate(lines) if line.startswith("Story status:")) + 1
    # A receipt without a readable answer says so; the retelling's own id is not the helper's answer.
    bare = {**receipt, "response_ref": {}}
    assert ChronicleStore(tmp_path).publish([], scan_state={"fallback_refusals": {unit: bare}}).ok
    refusal = [line for line in _story(tmp_path).split("\n") if line.startswith("A helper could not fold")]
    assert refusal == ["A helper could not fold: Beta; 2026-09-02 00:02 → 2026-09-02 00:03; context_overflow; "
                       "its answer was not retained"]


def test_drafts_rejections_corrections_and_indented_texts(tmp_path):
    shared.world(tmp_path)
    store = ChronicleStore(tmp_path)
    draft = _page(tmp_path, "1", 10, 10, author=HELPER, text="A helper's page.\n## Not a section")
    text = _story(tmp_path)
    assert "(draft by a helper (Light), not yet accepted or rejected by me)" in text
    assert "  A helper's page.\n  ## Not a section" in text and "\n## Not a section" not in text
    assert store.decide(draft, False, shared.MIND, "wrong reading").ok
    assert draft not in _story(tmp_path)  # a rejected draft does not act
    accepted = _page(tmp_path, "1", 10, 10, author=HELPER, text="A better helper page.")
    assert store.decide(accepted, True, shared.MIND, "right").ok
    assert "(drafted by a helper (Light), accepted by me)" in _story(tmp_path)
    mine = _page(tmp_path, "1", 11, 12, text="My own page.")
    result = store.correct(mine, "My own page, corrected.", shared.MIND, expected_sequence=store.room_head("1"))
    text = _story(tmp_path)
    # The correction stands under the page's own words, signed and dated; nothing is replaced or re-announced.
    signed = f"  My own page.\n\n  [correction {result.record['id']} by mind (root, task t1), {result.record['ts'][:10]}]\n  My own page, corrected."
    assert signed in text and "corrected by me" not in text
    assert text.count("(draft by a helper") == 0
    # A delegated child's draft is signed by the child, not by Light.
    child = {"kind": "helper", "task_id": "kid00001", "route": {}, "focus": {"role": "child", "task_id": "kid00001"}}
    _page(tmp_path, "1", 19, 19, author=child, text="A child's page.")
    text = _story(tmp_path)
    assert "(draft by a helper (child, task kid00001), not yet accepted or rejected by me)" in text
    assert text.count("(draft by a helper") == 1 and "(draft by a helper (Light)" not in text


def test_the_story_is_byte_identical_for_every_integrator_and_changes_only_with_the_chronicle(tmp_path, monkeypatch):
    rooms = shared.world(tmp_path)
    _page(tmp_path, str(rooms["alpha"]), 13, 13)
    tasks = [MAIN, {"id": "rootA001", "chat_id": rooms["alpha"]}, {"id": "rootB001", "chat_id": rooms["beta"]},
             {"id": "wake0001", "chat_id": 1, "metadata": {"usage_category": "consciousness"}},
             {"id": "pres0001", "chat_id": 555, "metadata": {"presence": {"binding_id": "b"}}},
             {"id": "kid00001", "chat_id": 1, "delegation_role": "subagent"}]
    clock = iter(f"2026-10-0{n}T00:00:00+00:00" for n in range(1, 9))
    monkeypatch.setattr("ouroboros.utils.utc_now_iso", lambda: next(clock))
    stories = {task["id"]: _story(tmp_path, task) for task in tasks}
    assert len(set(stories.values())) == 1, stories.keys()  # a child reads the same bytes: no block is privileged
    story = stories["turn0001"]
    assert "  Main talk." in story and "  Alpha worked." in story and "memory_read(node_id='legacy-" not in story
    assert story == _story(tmp_path, {**KID, "id": "kid00002", "chat_id": rooms["beta"]})
    assert '{"' not in story and " ago" not in story and "captured" not in story.lower()
    assert not [line for line in story.split("\n") if re.match(r"\s*\d+[.)] ", line)]  # no ordinals
    assert "turn0001" not in story and "rootA001" not in story
    shared.append(tmp_path / "logs" / "chat.jsonl", shared.msg("2026-09-04T00:00:00+00:00", "a new word"))
    (tmp_path / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "knowledge" / "topic.md").write_text("a knowledge note", encoding="utf-8")
    assert _story(tmp_path) == story  # a chat row or a knowledge note is not the story
    _page(tmp_path, str(rooms["beta"]), 14, 14)
    assert _story(tmp_path) != story  # a new page is
    nanny = {"id": "nan00001", "chat_id": 1, "delegation_role": "subagent",
             "configured_subagent": {"route": {"kind": "agent_session"}}}
    assert _story(tmp_path, nanny) == ""


def test_every_story_reads_every_unfolded_retold_record_whole_and_no_block_is_privileged(tmp_path):
    """Every unfolded record of the old retelling is whole in the story of Main, a root, consciousness,
    Presence and a child, whatever its block: the first block has no entitlement of its own and a later
    one is not a pointer by number. The room page never repeats what the story shows whole; a nanny has no
    story and reads its own room's retold records on its room page. Only the floor (F3) or my selection of an
    account turns a retold record into an address."""
    rooms = shared.world(tmp_path)
    alpha = rooms["alpha"]
    first = (f"#### legacy-b00-r{alpha} — 2026-09-01 00:02 → 2026-09-01 00:05 — Alpha — 3 rows retold in 12 chars\n"
             "  Alpha began.")
    later = (f"#### legacy-b01-r{alpha} — 2026-09-02 00:00 → 2026-09-02 00:01 — Alpha — 2 rows retold in 13 chars\n"
             "  Alpha worked.")
    root = {"id": "rootA001", "chat_id": alpha}
    for task in (MAIN, root, {"id": "wake0001", "chat_id": 1, "metadata": {"usage_category": "consciousness"}},
                 {"id": "pres0001", "chat_id": 555, "metadata": {"presence": {"binding_id": "b"}}}, KID):
        text = _story(tmp_path, task)
        assert first in text and later in text and "  Beta asked." in text, task["id"]
        assert "memory_read(node_id='legacy-" not in text, task["id"]  # nothing retold is a pointer by its block
        assert text.index(first) < text.index(later), task["id"]
    nanny = {"id": "nan00001", "chat_id": 1, "delegation_role": "subagent",
             "configured_subagent": {"route": {"kind": "agent_session"}}}
    assert _story(tmp_path, nanny) == "" and "Alpha began." not in mv.render_room(_snapshot(tmp_path, nanny))
    assert "  Main talk." in mv.render_room(_snapshot(tmp_path, nanny))  # its own room, on its room page
    for task in (root, {**KID, "id": "kid00003", "chat_id": alpha}):
        room = mv.render_room(_snapshot(tmp_path, task))
        assert "Alpha began." not in room and "Alpha worked." not in room, task["id"]  # whole in the story already
    # The floor, not the block number, makes a room's records one line; the horizon stays in it.
    level = mv.FloorLevel((("F3", (str(alpha),)),))
    short = _story(tmp_path).replace(first, "").replace(later, "")
    taken = mv.render_story(_snapshot(tmp_path), level)
    assert "Alpha began." not in taken and "Alpha worked." not in taken and "  Main talk." in taken
    assert (f"- Alpha; 2026-09-01 00:02 → 2026-09-02 00:01; 2 retold records in 25 chars: legacy-b00-r{alpha}, "
            f"legacy-b01-r{alpha}; memory_read(node_id=<id>) reads each") in taken.split("\n")
    assert short != taken  # the room's line stands where its records stood


def test_an_import_not_completed_names_its_reason_and_the_untouched_old_file(tmp_path, monkeypatch):
    shared.world(tmp_path, activate=False)
    monkeypatch.setattr(ChronicleStore, "ensure_activated",
                        lambda self, **kw: {"kind": "import_pending", "reason": "legacy_memory_lock_busy"})
    text = _story(tmp_path)
    assert text.split("\n")[0] == "## My story — unavailable now (legacy_memory_lock_busy)"
    assert "read_file(root='runtime_data', path='memory/dialogue_blocks.json')" in text


def test_gaps_unknown_ranges_and_the_flat_file_are_pointers_with_honest_periods(tmp_path):
    shared.world(tmp_path, cursor=False, flat="The flat old summary.")
    lines = _story(tmp_path).split("\n")
    gap = [line for line in lines if line.startswith("- memory gap: ")]
    assert len(gap) == 1 and gap[0].startswith(
        "- memory gap: Unknown provenance [legacy mixed record]; period known from the retelling text only; "
        "the old cursor file is missing while legacy blocks exist; memory_read(node_id='legacy-cursor-gap-")
    # Rows not established: the old writer's own count, then the length; the flat file has only its length.
    assert ("#### legacy-b00-r1 — 2026-09-01 00:00 - 00:05 (block period) — Main — 4 messages retold (the old "
            "writer's count) in 10 chars") in lines
    assert _story(tmp_path, KID) == _story(tmp_path)
    assert any(line.startswith("#### legacy-flat-") and line.endswith(
        " — period known from the retelling text only — Unknown provenance [legacy mixed record] — retold in 21 chars")
        for line in lines) and "  The flat old summary." in lines
    short = mv.render_story(_snapshot(tmp_path), mv.FloorLevel((("F3", ("1", "legacy")),))).split("\n")
    assert ("- Main; 2026-09-01 00:00 - 00:05 (block period); 2026-09-02 00:00 - 00:03 (block period); 2 retold records "
            "in 25 chars: legacy-b00-r1, legacy-b01-r1; memory_read(node_id=<id>) reads each") in short
    assert any(line.startswith("- Unknown provenance [legacy mixed record]; period known from the retelling text "
                               "only; 1 retold record in 21 chars: legacy-flat-") for line in short)


def test_a_pages_verified_quotes_stand_under_its_text_and_a_page_without_quotes_shows_none(tmp_path):
    """Regression: a helper writes people's words only
    through its quotes, so the story shows each verified quote under the page's text and the
    floor measures it with the page; a page without quotes renders as before."""
    from ouroboros.chronicle_import import row_lineage
    from ouroboros.tools.chronicle import _quote_resolver, page_covers

    shared.world(tmp_path)
    plain = _page(tmp_path, "1", 10, 10, author=HELPER, text="The room started.")
    addresses = {pos: address for address, _row, pos in chat_chain.iter_rows(tmp_path)}
    covers = page_covers(tmp_path, "1", from_addr=addresses[11], to_addr=addresses[11])["covers"]
    quote = {"address": chat_chain.format_address(addresses[11]), "text": "next please", "speaker": "human"}
    result = ChronicleStore(tmp_path).publish_page(room_id="1", text="The owner asked to go on.", covers=covers,
                                                   author=HELPER, quotes=[quote],
                                                   quote_resolver=_quote_resolver(tmp_path, row_lineage(tmp_path)))
    assert result.ok, result
    text = _story(tmp_path)
    block = text.split(f"page {result.record['id']}", 1)[1]
    assert block.split("\n")[1:3] == ["  The owner asked to go on.", f"- quote (human, {quote['address']}): next please"]
    plain_block = text.split(f"page {plain}", 1)[1].split("### ", 1)[0]
    assert "The room started." in plain_block and "- quote (" not in plain_block
