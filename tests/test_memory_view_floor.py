"""The owner's single start budget, ordered forms and protected commitments."""

import pytest

from ouroboros import context_budget as cb, memory_floor as mf, memory_view as mv
from tests import _memory_view_synthetic as syn
from tests import _memory_inventory_shared as shared


def tokens(snapshot, level=mv.FULL_VIEW):
    return mf.view_tokens(mv.render_story(snapshot, level)) + mf.view_tokens(mv.render_room(snapshot, level))


def frame(bound, physical=10**9):
    return {"margin": bound, "physical": physical, "budget": None}


def test_role_budget_is_one_frame_and_reply_is_charged_once():
    for role, facts, levels in (("integrator", {}, 2), ("consciousness", {"direct_chat": True}, 1),
                                ("integrator", {"direct_chat": True}, 1), ("child", {}, 1)):
        actual = mf.floor_allowances(window_tokens=200_000, output_reserve_tokens=65_536,
            non_memory_tokens=10_000, work_room_levels=cb.start_room_levels(role, **facts))
        assert actual["margin"] == 200_000 - 65_536 - 10_000 - levels * 25_000
        assert actual["physical"] == 200_000 - 65_536 - 10_000
    # Input-only limits use total-equivalent for physical arithmetic and the stated I for B/8.
    actual = mf.floor_allowances(window_tokens=265_536, boundary_tokens=200_000,
                                output_reserve_tokens=65_536, non_memory_tokens=0)
    assert actual["margin"] == 175_000 and actual["physical"] == 200_000
    unknown = mf.floor_allowances(window_tokens=None, output_reserve_tokens=65_536, non_memory_tokens=10)
    assert unknown == {"margin": None, "physical": None, "budget": None}


def test_story_openings_precede_untold_current_replies():
    replies = [syn.spoken(1, "ouroboros", 5_000), syn.spoken(2, "ouroboros", 50_000)]
    snapshot = syn.snapshot(story=syn.pages(2, 12_000), room_facts=syn.room("1", lane1=replies))
    assert mf.LADDER == ("F1", "F6w", "F6g", "F3", "F2", "F4", "F7p", "F5l", "F5", "F5a", "F7", "F7a", "F8", "F8a", "F6a")
    assert mf.fit_memory_view(snapshot, frame(tokens(snapshot))) == mv.FULL_VIEW
    level = mf.fit_memory_view(snapshot, frame(tokens(snapshot) - 100))
    assert level.addressed == (("F5l", ("page-000",)),)
    assert replies[0]["line"] in mv.render_room(snapshot, level)
    assert replies[1]["line"] in mv.render_room(snapshot, level)
    assert mv.render_story(snapshot, level) != mv.render_story(snapshot)
    owner_target = mf.fit_memory_view(snapshot, {"margin": 10**9, "physical": 10**9, "budget": 0})
    assert "F2" in dict(owner_target.steps) and "F5" in dict(owner_target.steps)


def test_governing_unanswered_and_uncovered_owner_rows_have_distinct_protection():
    rows = [{**syn.spoken(i, "human", 5_000), "owner_side": True, "immune": i == 2,
             "carried": i == 0} for i in range(3)]
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=rows))
    level = mf.fit_memory_view(snapshot, frame(0))
    assert dict(level.addressed)["F7"] == (rows[0]["address"],)
    text = mv.render_room(snapshot, level)
    assert rows[0]["line"] not in text and all(r["line"] in text for r in rows[1:])
    physical = mf.fit_memory_view(snapshot, frame(0, 0))
    assert dict(physical.addressed)["F7"] == tuple(r["address"] for r in rows[:2])
    assert rows[2]["line"] in mv.render_room(snapshot, physical)
    text = mf.render_view_for_mode(snapshot, mode="max", owner_mode="max", window_tokens=80_000,
        known_window=True, output_reserve=65_536, ratio=1, non_memory_tokens=10_000).room
    assert "Start room is smaller" in text


def test_pages_go_before_uncovered_owner_rows_and_the_oldest_rows_go_only_when_pages_are_exhausted():
    rows = [{**syn.spoken(i, "human", 5_000), "owner_side": True} for i in range(2)]
    story = syn.pages(2, 12_000)
    snapshot = syn.snapshot(story=story, room_facts=syn.room("1", lane1=rows))
    pages_cut = tokens(snapshot, mv.FloorLevel((("F5", tuple(p["id"] for p in story)),)))
    level = mf.fit_memory_view(snapshot, frame(0, pages_cut))
    assert dict(level.addressed) == {"F5": tuple(p["id"] for p in story)}
    assert all(r["line"] in mv.render_room(snapshot, level) for r in rows)
    oldest = mf.fit_memory_view(snapshot, frame(0, pages_cut - 1))
    assert dict(oldest.addressed)["F7"] == (rows[0]["address"],) and rows[1]["line"] in mv.render_room(snapshot, oldest)
    both = mf.fit_memory_view(snapshot, frame(0, 0))
    assert dict(both.addressed)["F7"] == tuple(r["address"] for r in rows)


def test_a_project_roots_sealed_owner_rows_fold_only_under_physical_pressure_oldest_first_into_dated_addresses():
    """F7 holds a project root's rows of its human that left the open conversation (``project_words``) before the
    open ones: whole under margin pressure, the oldest first into its dated address (never a retelling) only under
    physical pressure; a store gap line is no row; a mind's own fold of one needs its authored note as any
    uncovered commitment does."""
    def sealed_row(i):
        stamp = f"[2026-09-0{i + 1}T00:00:00+00:00 · owner · room_message · chat 1 / s{i}]"
        address = f"row:1@2026-09-0{i + 1}T00:00:00+00:00#{i:012x}"
        return {"address": address, "ref": f"chat 1 / s{i}", "ts": f"2026-09-0{i + 1}T00:00:00+00:00",
                "line": f"{stamp}\n  " + "w" * 5_000, "short": f"{stamp} (words, 5000 chars — read by address {address})"}

    sealed = [sealed_row(i) for i in range(2)]
    gap = {"line": "[Host fact: my human's rows of room 1 could not be read here (OSError); chat_history reads that room.]"}
    rows = [{**syn.spoken(i, "human", 5_000), "owner_side": True} for i in range(2)]
    snapshot = syn.snapshot(room_facts={**syn.room("1", lane1=rows), "project_words": [*sealed, gap]})
    assert mf.sealed_owner_rows(snapshot.room) == sealed
    full = mv.render_room(snapshot)
    assert all(item["line"] in full for item in sealed) and gap["line"] in full
    assert full.index(sealed[1]["line"]) < full.index("### Open conversation") < full.index(rows[0]["line"])
    soft = mf.fit_memory_view(snapshot, frame(0))  # margin pressure alone: people's words stay whole
    assert "F7" not in dict(soft.addressed) and all(item["line"] in mv.render_room(snapshot, soft) for item in sealed)
    oldest = mf.fit_memory_view(snapshot, frame(0, tokens(snapshot, soft) - 100))
    assert dict(oldest.addressed)["F7"] == (sealed[0]["address"],)
    text = mv.render_room(snapshot, oldest)
    assert sealed[0]["short"] in text and sealed[0]["line"] not in text and gap["line"] in text
    assert sealed[1]["line"] in text and all(row["line"] in text for row in rows)
    physical = mf.fit_memory_view(snapshot, frame(0, 0))
    assert dict(physical.addressed)["F7"] == (*(item["address"] for item in sealed), *(row["address"] for row in rows))
    with pytest.raises(ValueError, match="authored working_note"):
        mf.level_with(snapshot, mv.FULL_VIEW, [f"F7:{sealed[0]['address']}"])
    noted = mf.level_with(snapshot, mv.FULL_VIEW, [f"F7:{sealed[0]['address']}"], owner_note=True)
    assert dict(noted.addressed)["F7"] == (sealed[0]["address"],)


@pytest.mark.parametrize("lane1, owners", [
    ([{**syn.spoken(i, "human", 8_000), "owner_side": True} for i in range(3)], True),
    ([syn.spoken(i, "ouroboros", 1_000) for i in range(400)], False),
])
def test_the_start_room_notice_names_owner_words_only_when_they_keep_the_view_above_it(lane1, owners):
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=lane1))
    fit = mf.render_view_for_mode(snapshot, mode="max", owner_mode="max", window_tokens=200_000,
        known_window=True, output_reserve=65_536, ratio=1, non_memory_tokens=104_464)
    text, facts = fit.room, fit.facts
    floor = facts["floor"]
    assert (floor["allowance_tokens"], floor["physical_allowance_tokens"]) == (5_000, 30_000)
    assert ("Start room is smaller than requested" in text) is owners
    assert ("uncovered owner words remain in view" in text) is owners
    assert "shortest forms" not in text
    if not owners:
        assert floor['steps']['F7a'] == 1 and 0 < floor['steps']['F2'] <= 400
        assert mf.view_tokens(text) <= floor['allowance_tokens']
    assert "F7" not in dict(floor["steps"])


def test_other_conversations_fold_their_words_then_a_group_to_its_line_and_only_last_all_to_one_count():
    from ouroboros import memory_view_conversations as conversations

    groups = [syn.conversation(i, notes=1, words=3, word_chars=2_000) for i in range(164)]
    snapshot = syn.snapshot(conversations=groups)
    first = groups[0]
    words = tuple(word["id"] for word in first["rooms"][0]["words"])
    full = mv.render_room(snapshot)
    worded = mv.render_room(snapshot, mv.FloorLevel((("F6w", words),)))
    lined = mv.render_room(snapshot, mv.FloorLevel((("F6g", (first["key"],)),)))
    keys = tuple(group["key"] for group in groups)
    counted = mv.render_room(snapshot, mv.FloorLevel((("F6g", keys), ("F6a", keys))))
    assert len(counted) < len(lined) < len(full) and len(worded) < len(full)
    assert "from=<first>, to=<last>" in worded and first["rooms"][0]["words"][0]["line"] not in worded
    assert first["rooms"][0]["words"][0]["address"] in worded
    assert conversations.identity_line(first) in lined and first["rooms"][0]["notes"][0]["text"] in full
    assert "164 conversations by count: 164 open rooms" in counted
    assert "memory_read(room_id='*', start=0)" in counted and counted.count("No recorded consideration:") == 164
    order = [(step, ident) for step, ident, *_ in mf.floor_elements(snapshot) if step in ("F6w", "F6g", "F6a")]
    assert [step for step, _ident in order][:3] == ["F6w"] * 3 and order[-1] == ("F6a", "unread")
    level = mf.fit_memory_view(snapshot, frame(tokens(snapshot) - 1_000))  # the oldest words fold first, alone
    taken = dict(level.addressed)
    assert set(taken) == {"F6w"} and taken["F6w"] == tuple(ident for step, ident in order if step == "F6w")[:len(taken["F6w"])]


def test_giveback_keeps_a_reply_prefix_and_never_returns_kept_choices():
    replies = [syn.spoken(1, "ouroboros", 1_000), syn.spoken(2, "ouroboros", 20_000)]
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=replies))
    bound = tokens(snapshot) - 2_000
    level = mf.fit_memory_view(snapshot, frame(bound))
    assert dict(level.addressed)["F2"] == tuple(r["address"] for r in replies)
    kept = mv.FloorLevel((("F2", (replies[0]["address"],)),))
    later = mf.fit_memory_view(snapshot, frame(10**9), kept=kept)
    assert later == kept
    facts = mf.view_facts(snapshot, level, window_tokens=200_000, mode="max", allowances={})
    assert mf.kept_level(snapshot, facts) == level  # counts alone cannot reconstruct a give-back choice


def test_floor_fact_records_index_runs_against_its_element_order():
    """One representation of the selected forms: index runs in each step's element order, of the forms the fact names,
    exact for a give-back hole and for the mind's selection (F5l to F5, F8 labels to F8a); a fact written before the
    forms were versioned and an older fact's ids still read. A fact of counts alone, of another element order or of
    other forms is read anew, never as a step's first elements. The fact keeps one size whatever the history
    holds, and the trace keeps what a shortage reads without any element id."""
    import json
    from ouroboros.memory_inventory import VIEW_TRACE_KEY, shortage_from_trace

    replies = [syn.spoken(1, "ouroboros", 1_000), syn.spoken(2, "ouroboros", 20_000)]
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=replies))
    hole = mf.level_with(snapshot, mv.FULL_VIEW, ["F2:" + replies[1]["address"]])  # authored selection may have a hole
    facts = mf.view_facts(snapshot, hole, window_tokens=200_000, mode="max", allowances={})
    assert facts["floor"]["runs"] == {"F2": [[1, 1]]} and facts["floor"]["forms_version"] == mf.FORMS_VERSION == 2
    assert "addressed" not in facts["floor"] and mf.kept_level(snapshot, facts) == hole
    unversioned = {key: value for key, value in facts["floor"].items() if key != "forms_version"}
    assert mf.kept_level(snapshot, {"floor": unversioned}) == mv.FULL_VIEW
    older = {"floor": {**unversioned, "addressed": hole.addressed, "runs": None, "element_order": None}}
    assert mf.kept_level(snapshot, {"floor": {k: v for k, v in older["floor"].items() if k != "runs"}}) == mv.FULL_VIEW
    for anew in ({"steps": {"F2": 1}}, {**facts["floor"], "element_order": "another order"},
                 {**facts["floor"], "forms_version": mf.FORMS_VERSION + 1},
                 {**older["floor"], "forms_version": mf.FORMS_VERSION + 1}):
        assert mf.kept_level(snapshot, {"floor": anew}) == mv.FULL_VIEW  # never the first reply by its count
        budget = mf.kept_level(snapshot, {"floor": {**anew, "steps": {"F2": 1}, "by_budget": 1}})
        assert budget == mv.FloorLevel(by_budget=1)  # the owner's mode budget still frames the walk, no element kept
    room = syn.room("1", notes=3, lane1=[syn.spoken(i, "ouroboros", 400) for i in range(4)])
    chosen = syn.snapshot(room_facts=room, story=syn.pages(4, 400))
    pages, notes = [page["id"] for page in chosen.story], [note["id"] for note in room["notes"]]
    mind = mf.level_with(chosen, mv.FULL_VIEW, [f"F5l:{pages[0]}", f"F5l:{pages[2]}", f"F2:{room['lane1'][1]['address']}",
                                                *(f"F8:{ident}" for ident in notes)])
    mind = mf.level_with(chosen, mind, [f"F5:{pages[2]}", "F8a:1"])
    assert dict(mind.addressed) == {"F2": (room["lane1"][1]["address"],), "F5l": (pages[0],), "F5": (pages[2],),
                                    "F8": tuple(notes), "F8a": ("1",)}
    assert mf.kept_level(chosen, mf.view_facts(chosen, mind, window_tokens=92_000, mode="max", allowances={})) == mind
    sizes = []
    for words in (15, 1_500):  # 150 and 15,000 words of ten other conversations, every one by address
        many = syn.snapshot(room_facts=syn.room("1"), conversations=[syn.conversation(i, words=words) for i in range(10)])
        level = mf.fit_memory_view(many, frame(0))
        assert dict(level.steps)["F6w"] == 10 * words
        trace = {VIEW_TRACE_KEY: mf.trace_facts(mf.view_receipt(many, "", "", mf.view_facts(
            many, level, window_tokens=92_000, mode="max", allowances={})))}
        text = json.dumps(trace)
        assert shortage_from_trace(trace) is None  # the global unread count is a window fact, not a retelling debt
        assert many.conversations[0]["rooms"][0]["words"][0]["id"] not in text
        sizes.append(len(text))
    assert sizes[1] < 1.2 * sizes[0], f"floor fact bytes at 150 and 15,000 words: {sizes}"


def test_story_cuts_old_pages_before_fresh_retellings_without_changing_display(tmp_path):
    from ouroboros.chronicle_store import ChronicleStore
    from tests.test_chronicle_tools import addr, chat, ctx_for, ts, write

    store, ctx = ChronicleStore(tmp_path), ctx_for(tmp_path)
    store.ensure_activated()
    ids = []
    for n in range(3):
        row = chat(tmp_path, [{"chat_id": 1, "direction": "in", "ts": ts(n), "text": f"Words {n}"}])[0]
        ids.append(write(ctx, kind="page", text=f"Page {n}. " + "p" * 8_000,
                         covers={"from": addr(row), "to": addr(row)})["node_id"])
    part = write(ctx, kind="part", text="Fresh retelling. " + "r" * 8_000, member_ids=ids[1:],
                 expected_sequence=store.room_head("1"))["node_id"]
    snapshot = mv.capture_memory_view(tmp_path, {"id": "next0001", "chat_id": 1}, mv.ROLE_DEFAULTS["integrator"])
    assert [entry["id"] for entry in snapshot.story] == [part, ids[0]]  # cache-friendly display
    level = mf.fit_memory_view(snapshot, frame(tokens(snapshot) - 100))
    assert dict(level.addressed)["F5l"] == (ids[0],)
    story = mv.render_story(snapshot, level)
    assert "Fresh retelling." in story and "Page 0." in story and "p" * 8_000 not in story


def test_page_sentence_rung_uses_covered_age_even_when_publication_order_differs():
    old, recent = [{**page, "floor_order": [n, n + 1]} for n, page in enumerate(syn.pages(2, 8_000))]
    snapshot = syn.snapshot(story=[recent, old])
    level = mf.fit_memory_view(snapshot, frame(tokens(snapshot) - 100))
    assert dict(level.addressed) == {"F5l": (old["id"],)}
    story = mv.render_story(snapshot, level)
    assert recent["text"] in story and old["text"] not in story
    assert story.index("Page 1.") < story.index("Page 0.")  # display remains publication order
    elements = mf.floor_elements(snapshot)
    assert [ident for step, ident, *_ in elements if step == "F5"] == [old["id"], recent["id"]]


def test_story_giveback_does_not_restore_an_old_small_page_over_a_newer_retelling():
    old = {**syn.pages(1, 1_000)[0], "floor_order": [0, 1]}
    part = {**syn.pages(1, 20_000)[0], "id": "fresh-part", "kind": "part", "floor_order": [1, 4]}
    snapshot = syn.snapshot(story=[part, old])
    level = mf.fit_memory_view(snapshot, frame(tokens(snapshot) - 2_000))
    assert dict(level.addressed)["F5l"] == (old["id"], part["id"])
    assert mf.fit_memory_view(snapshot, frame(tokens(snapshot))) == mv.FULL_VIEW


def test_capture_protects_governing_words_and_does_not_recapture_for_fitting(tmp_path):
    shared.world(tmp_path)
    task = {"id": "turn0001", "chat_id": 1, "type": "task", "_is_direct_chat": True,
            "governing_owner_words": [{"text": "next please", "ref": "chat 1 / m1"}]}
    spec = mv.view_spec_for_task(task, tmp_path)
    snapshot = mv.capture_memory_view(tmp_path, task, spec)
    assert snapshot.active and snapshot.spec.work_room_levels == 1
    assert any(i.get("owner_side") and i.get("immune") for i in snapshot.room["lane1"])
    assert mv.snapshot_from_json(mv.snapshot_json(snapshot)) == snapshot
    before = mv.snapshot_json(snapshot)
    mf.fit_memory_view(snapshot, frame(0, 0))
    assert mv.snapshot_json(snapshot) == before


@pytest.mark.parametrize("governing, immune", [
    ([{"text": "да"}], {"m5"}),
    ([{"text": "да", "ref": "chat 1 / m3"}], {"m3"}),
    ([{"source": "initial_user"}, {"text": "да"}], {"m5"}),
])
def test_governing_words_protect_rows_by_identity_never_by_substring(tmp_path, governing, immune):
    shared.world(tmp_path)
    words = {"m3": "когда это будет готово?", "m4": "всегда проверяй логи", "m5": "да"}
    shared.append(tmp_path / "logs" / "chat.jsonl", *[shared.msg(f"2026-09-04T00:0{i}:00+00:00", text, client_message_id=key)
                                                      for i, (key, text) in enumerate(words.items())],
                  shared.msg("2026-09-04T00:09:00+00:00", "answered", direction="out", task_id="t4"))
    task = {"id": "turn0002", "chat_id": 1, "type": "task", "_is_direct_chat": True, "governing_owner_words": governing}
    lane1 = mv.capture_memory_view(tmp_path, task, mv.view_spec_for_task(task, tmp_path)).room["lane1"]
    assert {key for key in words for item in lane1 if item["ref"] == f"chat 1 / {key}" and item["immune"]} == immune


@pytest.mark.parametrize("names, expected", [(None, ""), (("memory_read",), ""),
    (("enable_tools",), "reachable through enable_tools"), (("list_available_tools",), "my parent task can read")])
def test_floor_receipt_names_reachable_sources_and_never_promises_lossless_recall(names, expected):
    note = mf.floor_note(mv.FloorLevel((("F5", ("p",)),)), window_tokens=200_000, mode="nano", tool_names=names)
    assert expected in note and "Nothing is lost" not in note and "brings it back" not in note
    assert "Read sources with memory_read" in note
    assert "chronicle_write" not in note


def test_list_forms_do_not_address_new_or_immune_rows_and_stay_silent_when_fit():
    rows = [syn.spoken(i, 'ouroboros', 1000) for i in range(12)]
    rows[4] = {**syn.spoken(4, 'human', 1000), 'owner_side': True, 'immune': True}
    snap = syn.snapshot(room_facts=syn.room(lane1=rows), story=syn.pages(10, 2000))
    held = mf.level_with(snap, mv.FULL_VIEW, ['F2:' + r['address'] for r in rows[:4]])
    selected = mf.level_with(snap, held, ['F7a:1'])
    text = mv.render_room(snap, selected)
    assert '4 earlier rows, oldest first, by address' in text
    assert f"from='{rows[0]['address']}', to='{rows[3]['address']}'" in text
    assert all(row['line'] in text for row in rows[4:])
    assert mf.fit_memory_view(snap, frame(tokens(snap))) == mv.FULL_VIEW
    assert mf.fit_memory_view(snap, frame(10**9), kept=selected) == selected
    assert mf.kept_level(snap, mf.view_facts(snap, selected, window_tokens=92000, mode='nano', allowances={})) == selected
    note = mf.floor_note(selected, window_tokens=92000, mode='nano')
    assert 'aggregate row ranges' in note


def note_snapshot():
    room = syn.room(notes=6, lane1=[syn.spoken(0, 'human', 8000),
                                  *(syn.spoken(i, 'ouroboros', 8000) for i in range(1, 13))])
    room['lane1'][0].update(owner_side=True, carried=False)
    for index, note in enumerate(room['notes']):
        note['text'] = f'Label {index}.\n' + 'Authored detail. ' * 240
    return syn.snapshot(room_facts=room, story=syn.pages(5, 8000), conversations=[syn.conversation(0, notes=1)])


@pytest.mark.parametrize('physical', [None, 10**9])
def test_held_notes_follow_start_targets_and_requested_savings(physical):
    snap = note_snapshot()
    level = mf.fit_memory_view(snap, {'margin': 0, 'budget': 0, 'physical': physical}, saving=10**9)
    assert {'F8', 'F8a'} <= dict(level.steps).keys()
    assert all(note['id'] in mv.render_room(snap, level) for note in snap.room['notes'])
    roomy = mf.fit_memory_view(snap, {'margin': tokens(snap), 'budget': None, 'physical': physical})
    assert roomy == mv.FULL_VIEW


def test_notes_yield_late_to_shortage_without_taking_uncovered_owner_rows():
    snap = note_snapshot()
    prior = mf.level_with(snap, mv.FULL_VIEW, [f'{s}:{i}' for s, i, *_ in mf.floor_elements(snap)
                                             if s not in ('F8', 'F8a', 'F7')], owner_note=True)
    bound = tokens(snap, prior) - 1
    level = mf.fit_memory_view(snap, frame(bound), kept=prior)
    assert dict(level.addressed)['F8'] == (snap.room['notes'][0]['id'],)
    assert 'F8a' not in dict(level.steps)
    assert all(dict(level.addressed)[step] == ids for step, ids in prior.addressed)
    assert tokens(snap, level) <= bound
    assert snap.room['lane1'][0]['line'] in mv.render_room(snap, level)
    assert all(mv._note_lines([note])[0] in mv.render_room(snap, level) for note in snap.room['notes'][1:])
    facts = mf.view_facts(snap, level, window_tokens=92000, mode='nano', allowances={})
    assert mf.kept_level(snap, facts) == level
    assert '1 unsealed notes by label' in mf.floor_note(level, window_tokens=92000, mode='nano')


def test_notes_aggregate_only_when_all_the_label_lines_still_do_not_fit():
    snap = note_snapshot()
    lines = mf.level_with(snap, mv.FULL_VIEW, [f'{s}:{i}' for s, i, *_ in mf.floor_elements(snap)
                                             if s not in ('F8a', 'F7')], owner_note=True)  # remote counts precede note lists
    at_lines = mf.fit_memory_view(snap, frame(tokens(snap, lines)))
    assert dict(at_lines.steps)['F8'] == 6 and 'F8a' not in dict(at_lines.steps)
    level = mf.fit_memory_view(snap, frame(tokens(snap, lines) - 1), kept=lines)
    assert dict(level.steps)['F8'] == 6 and dict(level.steps)['F8a'] == 1
    text = mv.render_room(snap, level)
    assert mv._notes_line(snap.room['notes']) in text
    assert tokens(snap, level) < tokens(snap, lines)
    assert 'aggregate note lists' in mf.floor_note(level, window_tokens=92000, mode='nano')
    assert all(note['id'] in text for note in snap.room['notes'])
    assert mf.fit_memory_view(snap, frame(10**9), kept=level) == level


def test_note_forms_are_attributed_to_the_owner_target():
    snap = syn.snapshot(room_facts=syn.room(notes=1))
    snap.room['notes'][0]['text'] = 'Held meaning. ' + 'detail ' * 1000
    level = mf.fit_memory_view(snap, {'margin': 0, 'budget': 0, 'physical': 500})
    assert dict(level.steps)['F8'] == 1 and level.by_budget > 0
    fit = mf.render_view_for_mode(snap, mode='nano', owner_mode='nano', window_tokens=1000000,
        known_window=True, output_reserve=65536, ratio=1, non_memory_tokens=66000)
    room, facts = fit.room, fit.facts
    assert 'only physical pressure can narrow them' not in room
    assert facts['floor']['steps']['F8'] == 1 and facts['floor']['by_budget'] > 0


def test_short_notes_still_have_selectable_forms_and_a_saving_aggregate():
    snap = syn.snapshot(room_facts=syn.room(notes=20))
    addresses = [f'F8:{note["id"]}' for note in snap.room['notes']]
    lines = mf.level_with(snap, mv.FULL_VIEW, addresses)
    assert dict(lines.steps)['F8'] == 20
    assert tokens(snap, lines) > tokens(snap)
    assert mf.fit_memory_view(snap, frame(tokens(snap), tokens(snap))) == mv.FULL_VIEW
    aggregate = mf.fit_memory_view(snap, frame(0, tokens(snap) - 1))
    assert dict(aggregate.steps)['F8a'] == 1 and dict(aggregate.steps)['F8'] == 20
    assert tokens(snap, aggregate) < tokens(snap)
    restored = mf.level_with(snap, aggregate, [address + ':whole' for address in addresses])
    assert restored == mv.FULL_VIEW


def test_new_forms_do_not_inflate_a_short_notes_minimum_bound(monkeypatch):
    snap = syn.snapshot(room_facts=syn.room(notes=1))
    after = tokens(snap, mf.fit_memory_view(snap, frame(0, 0)))  # the floor's own walk with no room left
    with monkeypatch.context() as before:
        before.setattr(mf, 'LADDER', tuple(step for step in mf.LADDER if step not in ('F8', 'F8a')))
        whole = tokens(snap, mf.fit_memory_view(snap, frame(0, 0)))
    assert after == whole


def test_retold_words_fold_before_considered_ones_and_other_rooms_owner_words_before_this_rooms():
    """F6w folds told work, unread facts, then considered words; F7 takes my
    human's words of my other conversations before this room's; both only as physics demands."""
    group = syn.conversation(0, words=3)
    words = group["rooms"][0]["words"]
    for word, rank in zip(words, (2, 0, 1)):  # the oldest nobody considered, the middle retold, the newest considered
        word["rank"] = rank
        if rank == 0:
            word.update(kind="ouroboros", task_id="told", gist={"summary": "The reply settled it.",
                "room_id": group["rooms"][0]["room_id"], "address": word["address"], "ts": word["ts"]})
    owner = syn.conversation(1, words=2, owner=True)
    room = syn.room(lane1=[syn.spoken(0, "human", 300), syn.spoken(1, "ouroboros", 300)])
    snapshot = syn.snapshot(room_facts=room, conversations=[group, owner])
    order = [(step, ident) for step, ident, *_ in mf.floor_elements(snapshot)]
    assert [ident for step, ident in order if step == "F6w"] == [words[1]["id"], words[0]["id"], words[2]["id"], "bare:" + words[2]["id"]]
    theirs = [word["id"] for word in owner["rooms"][0]["words"]]
    f7 = [ident for step, ident in order if step == "F7"]
    assert f7[:len(theirs)] == theirs and room["lane1"][0]["address"] in f7[len(theirs):]
    assert not set(theirs) & {ident for step, ident in order if step == "F6w"}  # my human's words never fold early


def test_my_humans_words_of_other_conversations_keep_the_work_room_while_this_rooms_uncovered_words_spend_it():
    """My human's words of my other conversations are ordinary F7 elements of the walk: oldest first, after every rung
    but the last, they fold to keep the work room the start reserves (the margin) and to pay a requested saving, with
    few measures however many there are (240 rows here, ~1,700 on a real install). This room's uncovered owner words
    keep their right to the work room: whole above the margin, folded only at physical pressure."""
    owner = [syn.conversation(i, notes=1, words=4, word_chars=1_500, owner=True) for i in range(60)]
    here = {**syn.spoken(0, "human", 300), "owner_side": True}
    snapshot = syn.snapshot(room_facts=syn.room(lane1=[here]), story=syn.pages(3, 4_000), conversations=owner)
    calls = []

    def measure(level):
        calls.append(level)
        return tokens(snapshot, level)

    full = measure(mv.FULL_VIEW)
    level = mf.fit_memory_view(snapshot, {"margin": full // 3, "physical": full + 10, "budget": None}, measure=measure)
    theirs = [word["id"] for group in owner for word in group["rooms"][0]["words"]]
    folded = dict(level.addressed).get("F7", ())
    assert folded and tuple(theirs[:len(folded)]) == folded and tokens(snapshot, level) <= full // 3  # oldest first
    assert here["address"] not in folded and "F6a" in dict(level.steps)  # owner addresses follow the common count
    assert len(calls) < 30, len(calls)
    words = tuple(word["id"] for word in owner[0]["rooms"][0]["words"])  # a group's line still saves once its owner
    lined = mv.FloorLevel((("F6g", ("all:" + owner[0]["key"],)), ("F7", words)))
    assert all(mv._row_pointer(w, "words") in mv.render_room(snapshot, lined) for w in owner[0]["rooms"][0]["words"])
    paid = mf.fit_memory_view(snapshot, {"margin": None, "physical": full + 10, "budget": None}, saving=full // 2)
    assert dict(paid.addressed).get("F7") and tokens(snapshot, paid) <= full - full // 2  # a refusal's saving
    rows = [{**syn.spoken(i, "human", 4_000), "owner_side": True} for i in range(3)]
    alone = syn.snapshot(room_facts=syn.room(lane1=rows))
    whole = tokens(alone)
    kept = mf.fit_memory_view(alone, {"margin": whole // 2, "physical": whole + 10, "budget": None})
    assert "F7" not in dict(kept.steps) and tokens(alone, kept) > whole // 2  # this room's: the work room is theirs
    pressed = mf.fit_memory_view(alone, {"margin": whole // 2, "physical": whole - 100, "budget": None})
    assert dict(pressed.addressed)["F7"] == (rows[0]["address"],)  # and physics folds them, oldest first



def test_the_last_rung_keeps_my_humans_folded_words_of_other_conversations_by_their_addresses():
    """At the last rung (F6a, the groups to one count) my human's words of my other conversations that F7 folded keep
    their addresses, and one F7 did not fold stays whole. With a margin below the counted form, the addresses stay
    above it: they occupy the work room up to the physical bound, and no rung deletes them."""
    groups = [syn.conversation(i, words=6, word_chars=2_000, owner=True) for i in range(8)]
    snapshot = syn.snapshot(room_facts=syn.room(), conversations=groups)
    words = [word for group in groups for word in group["rooms"][0]["words"]]
    keys = tuple(group["key"] for group in groups)
    counted = mv.FloorLevel((("F6g", keys), ("F7", tuple(word["id"] for word in words)), ("F6a", keys)))
    full = tokens(snapshot)
    level = mf.fit_memory_view(snapshot, {"margin": tokens(snapshot, counted) - 1, "physical": full + 10, "budget": None})
    text = mv.render_room(snapshot, level)
    assert dict(level.steps).get("F6a") and len(dict(level.addressed)["F7"]) == len(words)  # the last rung was taken
    assert all(mv._row_pointer(word, "words") in text for word in words)  # every folded word keeps its address
    assert tokens(snapshot, counted) - 1 < tokens(snapshot, level) <= full + 10  # above the margin, in the window
    half = mv.FloorLevel((("F6g", keys), ("F7", tuple(word["id"] for word in words[:24])), ("F6a", keys)))
    shown = mv.render_room(snapshot, half)
    assert all(mv._row_pointer(word, "words") in shown for word in words[:24])
    assert all(word["line"] in shown for word in words[24:])  # one F7 did not fold stays whole
