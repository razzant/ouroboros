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
    for role, facts, levels in (("integrator", {}, 2), ("consciousness", {"direct_chat": True}, 2),
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


def test_order_conversation_before_my_pages_and_oldest_reply_first():
    replies = [syn.spoken(1, "ouroboros", 5_000), syn.spoken(2, "ouroboros", 50_000)]
    snapshot = syn.snapshot(story=syn.pages(2, 12_000), room_facts=syn.room("1", lane1=replies))
    assert mf.LADDER == ("F1", "F1b", "F6r", "F6", "F6a", "F3", "F2", "F4", "F7p", "F7", "F7a", "F5", "F5a", "F8", "F8a")
    assert mf.fit_memory_view(snapshot, frame(tokens(snapshot))) == mv.FULL_VIEW
    level = mf.fit_memory_view(snapshot, frame(tokens(snapshot) - 100))
    assert level.addressed == (("F2", (replies[0]["address"],)),)
    assert replies[0]["line"] not in mv.render_room(snapshot, level)
    assert replies[1]["line"] in mv.render_room(snapshot, level)
    assert mv.render_story(snapshot, level) == mv.render_story(snapshot)
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
    _, text, _ = mf.render_view_for_mode(snapshot, mode="max", owner_mode="max", window_tokens=80_000,
        known_window=True, output_reserve=65_536, ratio=1, non_memory_tokens=10_000)
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


@pytest.mark.parametrize("lane1, owners", [
    ([{**syn.spoken(i, "human", 8_000), "owner_side": True} for i in range(3)], True),
    ([syn.spoken(i, "ouroboros", 1_000) for i in range(400)], False),
])
def test_the_start_room_notice_names_owner_words_only_when_they_keep_the_view_above_it(lane1, owners):
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=lane1))
    _, text, facts = mf.render_view_for_mode(snapshot, mode="max", owner_mode="max", window_tokens=200_000,
        known_window=True, output_reserve=65_536, ratio=1, non_memory_tokens=104_464)
    floor = facts["floor"]
    assert (floor["allowance_tokens"], floor["physical_allowance_tokens"]) == (5_000, 30_000)
    assert ("Start room is smaller than requested" in text) is owners
    assert ("uncovered owner words remain in view" in text) is owners
    assert "shortest forms" not in text
    if not owners:
        assert floor['steps']['F7a'] == 1 and 0 < floor['steps']['F2'] <= 400
        assert mf.view_tokens(text) <= floor['allowance_tokens']
    assert "F7" not in dict(floor["steps"])


def test_other_rooms_have_range_directory_and_bounded_aggregate():
    rooms = [syn.live_room(i, notes=1, words=3, word_chars=2_000) for i in range(164)]
    snapshot = syn.snapshot(live=rooms)
    ident = rooms[0]["room_id"]
    full = mv.render_room(snapshot)
    ranged = mv.render_room(snapshot, mv.FloorLevel((("F6r", (ident,)),)))
    directory = mv.render_room(snapshot, mv.FloorLevel((("F6", (ident,)),)))
    aggregate = mv.render_room(snapshot, mv.FloorLevel((("F6a", tuple(r["room_id"] for r in rooms)),)))
    assert len(aggregate) < len(directory) < len(ranged) < len(full)
    assert "from='" in ranged and rooms[0]["words"][0]["line"] not in ranged
    assert mv._directory_room(rooms[0]) in directory
    assert "164 open rooms" in aggregate and "memory_read(room_id='*', start=0)" in aggregate
    assert len(aggregate) < 700


def test_giveback_restores_small_earlier_elements_but_never_kept_choices():
    replies = [syn.spoken(1, "ouroboros", 1_000), syn.spoken(2, "ouroboros", 20_000)]
    snapshot = syn.snapshot(room_facts=syn.room("1", lane1=replies))
    bound = tokens(snapshot) - 2_000
    level = mf.fit_memory_view(snapshot, frame(bound))
    assert dict(level.addressed)["F2"] == (replies[1]["address"],)
    kept = mv.FloorLevel((("F2", (replies[0]["address"],)),))
    later = mf.fit_memory_view(snapshot, frame(10**9), kept=kept)
    assert later == kept
    facts = mf.view_facts(snapshot, level, window_tokens=200_000, mode="max", allowances={})
    assert mf.kept_level(snapshot, facts) == level  # counts alone cannot reconstruct a give-back choice


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
    ([{"text": "всегда проверяй логи"}], {"m4"}),
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
    assert "memory_read reads each" in note
    assert ("chronicle_write" in note) == (names is None or "enable_tools" in names)


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
    return syn.snapshot(room_facts=room, story=syn.pages(5, 8000), live=[syn.live_room(0, notes=1)])


@pytest.mark.parametrize('physical', [None, 10**9])
def test_held_notes_ignore_start_targets_and_requested_savings(physical):
    snap = note_snapshot()
    level = mf.fit_memory_view(snap, {'margin': 0, 'budget': 0, 'physical': physical}, saving=10**9)
    assert not mf.WINDOW_ONLY.intersection(dict(level.steps))
    assert all(mv._note_lines([note])[0] in mv.render_room(snap, level) for note in snap.room['notes'])


def test_notes_yield_after_address_lists_and_uncovered_rows_only_to_the_window():
    snap = note_snapshot()
    prior = mf.level_with(snap, mv.FULL_VIEW, [f'{s}:{i}' for s, i, *_ in mf.floor_elements(snap)
                                             if s not in mf.WINDOW_ONLY], owner_note=True)
    physical = tokens(snap, prior) - 1
    level = mf.fit_memory_view(snap, frame(0, physical))
    assert dict(level.addressed)['F8'] == (snap.room['notes'][0]['id'],)
    assert 'F8a' not in dict(level.steps)
    assert all(dict(level.addressed)[step] == ids for step, ids in prior.addressed)
    assert tokens(snap, level) <= physical
    assert all(mv._note_lines([note])[0] in mv.render_room(snap, level) for note in snap.room['notes'][1:])
    facts = mf.view_facts(snap, level, window_tokens=92000, mode='nano', allowances={})
    assert mf.kept_level(snap, facts) == level
    assert '1 unsealed notes by label' in mf.floor_note(level, window_tokens=92000, mode='nano')


def test_notes_aggregate_only_when_all_the_label_lines_still_do_not_fit():
    snap = note_snapshot()
    lines = mf.level_with(snap, mv.FULL_VIEW, [f'{s}:{i}' for s, i, *_ in mf.floor_elements(snap)
                                             if s != 'F8a'], owner_note=True)
    at_lines = mf.fit_memory_view(snap, frame(0, tokens(snap, lines)))
    assert dict(at_lines.steps)['F8'] == 6 and 'F8a' not in dict(at_lines.steps)
    level = mf.fit_memory_view(snap, frame(0, tokens(snap, lines) - 1))
    assert dict(level.steps)['F8'] == 6 and dict(level.steps)['F8a'] == 1
    text = mv.render_room(snap, level)
    assert mv._notes_line(snap.room['notes']) in text
    assert tokens(snap, level) < tokens(snap, lines)
    assert 'aggregate note lists' in mf.floor_note(level, window_tokens=92000, mode='nano')
    assert all(note['id'] in text for note in snap.room['notes'])
    assert mf.fit_memory_view(snap, frame(10**9), kept=level) == level


def test_note_forms_are_not_attributed_to_the_owner_target():
    snap = syn.snapshot(room_facts=syn.room(notes=1))
    snap.room['notes'][0]['text'] = 'Held meaning. ' + 'detail ' * 1000
    level = mf.fit_memory_view(snap, {'margin': 0, 'budget': 0, 'physical': 500})
    assert dict(level.steps)['F8'] == 1 and level.by_budget == 0
    _, room, facts = mf.render_view_for_mode(snap, mode='nano', owner_mode='nano', window_tokens=1000000,
        known_window=True, output_reserve=65536, ratio=1, non_memory_tokens=66000)
    assert 'only physical pressure can narrow them' in room
    assert not mf.WINDOW_ONLY.intersection(facts['floor']['steps'])


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
    after = mf.minimal_view_tokens(snap, window_tokens=92000)
    with monkeypatch.context() as before:
        before.setattr(mf, 'LADDER', tuple(step for step in mf.LADDER if step not in mf.WINDOW_ONLY))
        whole = mf.minimal_view_tokens(snap, window_tokens=92000)
    assert after == whole
