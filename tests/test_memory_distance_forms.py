"""Distance order, clean starts, work tellings and final receipts of version 2 forms."""
from copy import deepcopy

import pytest

from ouroboros import memory_floor as mf, memory_forms as forms, memory_view as mv
from ouroboros import memory_view_work as work
from tests import _memory_view_synthetic as syn


def size(snap, level=mv.FULL_VIEW):
    return mf.tokens_of(mv.story_chars(snap, level)) + mf.tokens_of(mv.room_chars(snap, level))


def frame(bound):
    return {"margin": bound, "physical": 10**9, "budget": None}


def completed_room():
    rows = [dict(syn.spoken(i, "ouroboros", 3000), task_id="done") for i in range(3)]
    room = syn.room(lane1=rows)
    gist = {"summary": "The build now holds all decisions. Two details remain.", "ts": rows[-1]["ts"],
            "address": rows[-1]["address"], "room_id": "2001"}
    task = syn.task_line(0)
    task.update(task="done", line=task["line"] + " — summary: «" + gist["summary"] + "»")
    group = syn.conversation(1, words=2, word_chars=3000)
    group["rooms"][0]["tasks"] = [deepcopy(task)]
    for word in group["rooms"][0]["words"]:
        word.update(kind="ouroboros", task_id="done")
    room["lane2"] = [task]
    work.bind(room, [group], {"done": gist})
    return syn.snapshot(room_facts=room, conversations=[group]), gist


def test_one_work_telling_across_rooms_and_progress_before_final():
    snap, gist = completed_room()
    rows = snap.room["lane1"]
    assert mv.render_room(snap).count(gist["summary"]) == 1
    progress = mf.level_with(snap, mv.FULL_VIEW, ["F2:work:done:progress"])
    text = mv.render_room(snap, progress)
    assert text.count(gist["summary"]) == 1
    assert all(r["line"] not in text for r in rows[:-1]) and rows[-1]["line"] in text
    assert "2 replies, 6000 chars" in text and text.index("work done") < text.index(rows[-1]["line"])
    final = mf.level_with(snap, progress, ["F2:work:done:final"])
    text = mv.render_room(snap, final)
    assert text.count(gist["summary"]) == 1 and not any(r["line"] in text for r in rows)
    assert "3 replies, 9000 chars" in text and dict(final.addressed)["F2"] == ("work:done:final",)
    assert mf.view_facts(snap, final, window_tokens=200000, mode="max", allowances={})["floor"]["newest_addressed_row"]


@pytest.mark.parametrize("swap", [False, True], ids=["P-A", "P-B"])
def test_distance_order_has_two_adjacent_alternatives(monkeypatch, swap):
    snap, _ = completed_room()
    snap.conversations[0]["rooms"][0]["words"][0].pop("gist")
    stages = list(forms.STAGES)
    if swap:
        i = stages.index("work_final")
        stages[i:i+2] = reversed(stages[i:i+2])
        monkeypatch.setattr(forms, "STAGES", tuple(stages))
    order = [(s, i) for s, i, *_ in mf.floor_elements(snap)]
    assert order.index(("F2", "work:done:progress")) < order.index(("F6g", "all:project:2001"))
    assert (order.index(("F2", "work:done:final")) < order.index(("F6g", "last:project:2001"))) is not swap
    assert not any(s == "events" for s, _ in order)


def test_clean_guard_and_window_fact_use_the_same_ordinary_view():
    snap, _ = completed_room()
    level = mf.fit_memory_view(snap, frame(size(snap, mf.ordinary_level(snap)) + 10))
    assert {(s, i) for s, ids in level.addressed for i in ids} <= forms.ordinary_pairs(snap)
    assert not forms.form_counts(snap, level)["emergency"]
    assert forms.form_counts(snap, mf.fit_memory_view(snap, frame(0)))["emergency"]
    shown = mf.render_view_for_mode(snap, mode="low", owner_mode="max", window_tokens=200000,
        known_window=True, output_reserve=10000, non_memory_tokens=164999, ratio=1)
    assert shown.facts["floor"]["ordinary_unfit"]


def test_sole_task_gist_is_not_an_ordinary_fact():
    task = syn.task_line(0)
    task["gist"] = {"summary": "Only telling.", "room_id": "1", "address": task["last"]}
    task["line"] += " — summary: «Only telling.»"
    snap = syn.snapshot(room_facts={**syn.room(), "lane2": [task]})
    assert ("F1", task["first"]) not in forms.ordinary_pairs(snap)
    level = mf.fit_memory_view(snap, frame(0))
    assert forms.form_counts(snap, level)["emergency"]["sole task gists"] == {"lines": 1, "sources": 1}


def test_return_keeps_an_oldest_prefix_of_counted_conversations():
    groups = [syn.conversation(i, words=3, word_chars=n) for i, n in enumerate([40, 8000, 30, 4000, 500])]
    snap = syn.snapshot(conversations=groups)
    all_counted = mv.FloorLevel((("F6a", tuple(g["key"] for g in groups)),))
    for bound in range(size(snap, all_counted) + 20, size(snap), 300):
        level = mf.fit_memory_view(snap, frame(bound))
        chosen = [i for i, g in enumerate(groups) if g["key"] in dict(level.addressed).get("F6a", ())]
        assert chosen == list(range(len(chosen)))


def test_two_protected_replies_yield_after_room_emergencies_before_counts():
    rows = [dict(syn.spoken(i, "ouroboros", 4000), protected_reply=i in (0, 2)) for i in range(3)]
    groups = [syn.conversation(0, words=2, word_chars=3000)]
    groups[0]["rooms"][0]["words"][0].update(rank=1)
    snap = syn.snapshot(room_facts=syn.room(lane1=rows), conversations=groups)
    order = [(s, i) for s, i, *_ in mf.floor_elements(snap)]
    first_count = order.index(("F6a", groups[0]["key"]))
    for row in (rows[0], rows[2]):
        assert order.index(("F2", rows[1]["address"])) < order.index(("F2", row["address"])) < first_count
    boundary = mf.level_with(snap, mv.FULL_VIEW, [f"{s}:{i}" for s, i in order[:order.index(("F2", rows[0]["address"]))]])
    text = mv.render_room(snap, boundary)
    assert rows[0]["line"] in text and rows[2]["line"] in text and rows[1]["line"] not in text


def test_ladder_override_rebuilds_table_with_only_requested_steps(monkeypatch):
    snap = syn.snapshot(story=syn.pages(2, 3000), room_facts=syn.room(lane1=[syn.spoken(1, "ouroboros", 1000)]))
    all_elements = mf.floor_elements(snap)
    monkeypatch.setattr(mf, "LADDER", ("F5",))
    assert mf.floor_elements(snap) == tuple(e for e in all_elements if e[0] == "F5")
    assert set(dict(mf.fit_memory_view(snap, frame(0)).steps)) == {"F5"}


def test_no_empty_unread_element_and_one_window_line_for_many_rooms():
    groups = [syn.conversation(i, words=2) for i in range(3)]
    snap = syn.snapshot(conversations=groups)
    level = mv.FloorLevel((("F6a", (*(g["key"] for g in groups), "unread")),))
    text = mv.render_room(snap, level)
    assert text.count("Emergency window form:") == 1 and "6 words without recorded consideration" in text
    counts = forms.form_counts(snap, level)
    assert counts["window"]["unread count"] == {"lines": 1, "sources": 6} and "unread" not in counts["host"]
    dry = syn.snapshot(conversations=[syn.conversation(3)])
    assert ("F6a", "unread") not in {(s, i) for s, i, *_ in mf.floor_elements(dry)}


def test_final_counts_and_holding_cover_sources_without_counting_swallowed_forms():
    room = syn.room(lane1=[syn.spoken(i, "ouroboros", 3000) for i in range(8)], notes=1, legacy=1, legacy_chars=500)
    pages = syn.pages(3, 3000)
    pages[0].update(kind="account", selection="selected-account", sources=[])
    pages[1].update(kind="part", lead_rule=1)
    snap = syn.snapshot(room_facts=room, story=pages, conversations=[syn.conversation(1, words=2, owner=True)])
    level = mf.fit_memory_view(snap, frame(0))
    counts = forms.form_counts(snap, level)
    assert "room openings" not in counts["emergency"]
    assert counts["emergency"]["room ranges"] == {"lines": 1, "sources": 8}
    held = forms.holding_room(snap, mv.FULL_VIEW)
    assert {"accounts", "parts", "pages without part", "records without lead", "untold in this room",
            "owner words in other rooms", "notes", "room records"} <= held["tokens"].keys()
    assert len(held["largest"]) == 3 and all(e["reader"].startswith("memory_read(") for e in held["largest"])
    facts = mf.view_facts(snap, level, window_tokens=200000, mode="low", allowances={})["floor"]
    assert facts["forms"] == counts and facts["holding"] == forms.holding_room(snap, level)
    note = mf.floor_note(level, window_tokens=200000, mode="low", snapshot=snap)
    assert all(x in note for x in ("my telling", "host facts", "emergency without", "Holding the room"))


@pytest.mark.parametrize("row,author,task", [
    ({"task_id": "direct"}, {}, "direct"),
    ({"transport": {"origin": {"task_id": "origin"}}}, {}, "origin"),
    ({"type": "relay_entry", "task_id": "recipient", "relay": {"sender": {"task_id": "sender"}}}, {}, "sender"),
    ({"type": "relay_entry", "task_id": "recipient"}, {}, ""),
    ({"task_id": "publisher"}, {"via": "own_account"}, ""),
])
def test_work_key_is_the_author_not_a_relay_recipient(row, author, task):
    assert work.speech_facts(row, author)["task_id"] == task


def test_two_finished_threads_keep_their_gists_and_unread_fold_owner_lines():
    from ouroboros import memory_view_conversations as conv
    groups = [syn.conversation(i, words=2, word_chars=2000) for i in (1, 2)]
    summaries = {}
    for i, group in enumerate(groups):
        room = group['rooms'][0]
        own = room['words'][0]
        own.update(kind='ouroboros', task_id=f'task-{i}')
        task = dict(syn.task_line(i), task=f'task-{i}')
        gist = dict(summary=f'Unique result {i}. More detail.', room_id=room['room_id'],
                    address=task['last'], ts=task['last_ts'])
        task['line'] += ' — summary: «' + gist['summary'] + '»'
        room['tasks'] = [task]
        summaries[f'task-{i}'] = gist
    group = {**groups[0], 'rooms': [g['rooms'][0] for g in groups]}
    room = group['rooms'][0]
    room['folded'] = dict(task='previous-work', sentence='A previous reading settled.', ts=room['words'][0]['ts'],
                          count=3, **{'from': 'row:old-first', 'to': 'row:old-last'})
    owner = dict(syn.conversation(9, words=1, owner=True)['rooms'][0]['words'][0])
    room['words'].append(owner)
    work.bind(None, [group], summaries)
    snap = syn.snapshot(conversations=[group])
    for form in (1, 2, 3):
        text = conv.render_group(group, set(), set(), form)
        assert text.count('No recorded consideration:') == 2
        assert '3 words read and folded' in text and owner['line'] in text
        assert 'Unique result 1.' in text and ('Unique result 0.' in text) is (form != 3)
        assert 'More detail.' not in text
    level = mv.FloorLevel((('F6g', ('last:' + group['key'],)), ('F6a', (group['key'],))))
    counts = forms.form_counts(snap, level)
    text = mv.render_room(snap, level)
    assert 'read-and-folded rows by count' in text and 'previous reading' not in text
    assert 'last gist' not in counts['emergency'] and 'read and folded' not in counts['retold']
    assert counts['host']['unread'] == {'lines': 2, 'sources': 2}
    assert counts['emergency']['conversation count']['lines'] == 1


def test_capture_protects_last_outgoing_and_reply_target_of_unanswered_owner(tmp_path):
    from tests import _memory_inventory_shared as shared
    from tests.test_chronicle_tools import addr
    shared.world(tmp_path)
    first = shared.msg('2026-10-20T01:00:00+00:00', 'Earlier answer.', direction='out', task_id='reply-a')
    last = shared.msg('2026-10-20T01:02:00+00:00', 'Latest answer.', direction='out', task_id='reply-b')
    ask = shared.msg('2026-10-20T01:03:00+00:00', 'Explain the earlier answer.', reply_to={'address': addr(first)})
    shared.append(tmp_path / 'logs' / 'chat.jsonl', first, last, ask)
    task = {'id': 'next-turn', 'chat_id': 1, '_is_direct_chat': True}
    snap = mv.capture_memory_view(tmp_path, task, mv.view_spec_for_task(task, tmp_path))
    protected = {r['address'] for r in snap.room['lane1'] if r.get('protected_reply')}
    assert protected == {addr(first), addr(last)}
    assert next(r for r in snap.room['lane1'] if r['address'] == addr(ask))['immune']


def test_a_record_without_a_first_sentence_cannot_be_an_ordinary_lead():
    room = syn.room(notes=1)
    room['notes'][0].update(text='', lead_rule=1)
    snap = syn.snapshot(room_facts=room)
    ident = room['notes'][0]['id']
    assert ('F8', ident) not in forms.ordinary_pairs(snap)
    counts = forms.form_counts(snap, mv.FloorLevel((('F8', (ident,)),)))
    assert not counts['retold'] and counts['emergency']['notes openings']['sources'] == 1


def test_report_counts_a_work_telling_and_fold_read_once_in_each_visible_form():
    from devtools.context_fit_report import view_counts

    snap, _ = completed_room()
    remote = snap.conversations[0]['rooms'][0]
    remote['folded'] = dict(task='earlier', sentence='An earlier reading settled.', ts='2026-10-01',
                            count=3, **{'from': 'row:old-first', 'to': 'row:old-last'})
    whole = view_counts(snap, mv.FULL_VIEW)['retellings']
    assert whole == {'task summary': 1, 'fold_read': 1}
    for suffix in ('progress', 'final'):
        level = mf.level_with(snap, mv.FULL_VIEW, ['F2:work:done:' + suffix])
        assert view_counts(snap, level)['retellings'] == {'works': 1, 'fold_read': 1}
    counted = mv.FloorLevel((('F6a', (snap.conversations[0]['key'],)),))
    assert view_counts(snap, counted)['retellings'] == {'task summary': 1}
