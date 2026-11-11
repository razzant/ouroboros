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
    remote_task = deepcopy(task)
    remote_task["line"] = task["line"].replace(gist["summary"], work.first_sentence(gist["summary"]))
    group["rooms"][0]["tasks"] = [remote_task]
    for word in group["rooms"][0]["words"]:
        word.update(kind="ouroboros", task_id="done")
    room["lane2"] = [task]
    work.bind(room, [group], {"done": gist})
    return syn.snapshot(room_facts=room, conversations=[group]), gist


def test_one_work_telling_per_block_and_progress_before_final():
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


@pytest.mark.parametrize("remote_task", [False, True])
def test_replaced_words_are_told_once_in_their_own_block_at_every_level(remote_task):
    from ouroboros import memory_view_conversations as conv

    snap, gist = completed_room()
    group = snap.conversations[0]
    remote = group["rooms"][0]
    if not remote_task:
        remote["tasks"] = []
    else:
        group["rooms"].append({**deepcopy(remote), "room_id": "other-thread", "words": []})
    other = deepcopy(remote["words"][-1])
    other.update(id="other-work", task_id="later", work_id="later", gist=dict(
        gist, summary="A later work finished.", ts="2027-01-01"))
    remote["words"].append(other)
    order = [f"{s}:{i}" for s, i, *_ in mf.floor_elements(snap)]
    levels = [mf.level_with(snap, mv.FULL_VIEW, order[:i]) for i in range(len(order) + 1)]
    for level in [*levels, *reversed(levels)]:
        gone = {s: set(ids) for s, ids in level.addressed}
        form = conv.group_form(group, gone)
        counted = group["key"] in gone.get("F6a", ())
        receipt = forms.form_counts(snap, level)
        if counted:
            assert receipt["emergency"]["conversation count"]["sources"] >= len(remote["words"])
            continue
        text = conv.render_group(group, gone.get("F6w", set()), gone.get("F7", set()), form)
        if form == 3:
            assert gist["summary"] not in text
            assert receipt["emergency"]["last gist"]["sources"] >= 1
        else:
            replaced = any(w["id"] in gone.get("F6w", ()) for w in remote["words"][:2])
            assert text.count(work.first_sentence(gist["summary"])) == int(bool(form or replaced or remote_task))
            if form or replaced and not remote_task:
                assert f"[{gist['address']}]" in text


def test_room_range_is_physical_and_returns_surviving_task_tellings():
    snap, gist = completed_room()
    assert forms.stage_of(snap, "F7a", snap.room["room_id"]) == "room_ranges"
    assert forms.STAGES.index("physical_rows") < forms.STAGES.index("room_ranges") < forms.STAGES.index("owner")
    final = mf.level_with(snap, mv.FULL_VIEW, ["F2:work:done:final"])
    aggregate = mf.level_with(snap, final, ["F7a:" + snap.room["room_id"]])
    task_count = mf.level_with(snap, mv.FULL_VIEW, ["F1:" + snap.room["lane2"][0]["first"]])
    for level in (task_count, final, aggregate, final, task_count, mv.FULL_VIEW):
        text = mv._room_text(snap.room, dict(level.addressed))
        assert text.count(gist["summary"]) == 1
        assert ("work done]" in text) is (level == final)
    counts = forms.form_counts(snap, aggregate)
    assert counts["retold"]["task gists"] == {"lines": 1, "sources": 1}
    assert "works" not in counts["retold"]
    assert forms.holding_room(snap, aggregate)["tokens"]["task gists"] > 0
    facts = mf.view_facts(snap, aggregate, window_tokens=200000, mode="low", allowances={})["floor"]
    assert facts["forms"] == counts
    assert "task gists" in mf.floor_note(aggregate, window_tokens=200000, mode="low", snapshot=snap)
    counted = mf.level_with(snap, aggregate, ["F1:" + snap.room["lane2"][0]["first"]])
    assert gist["summary"] not in mv._room_text(snap.room, dict(counted.addressed))


@pytest.mark.parametrize("remote", [False, True], ids=["two-terminals", "fold-read"])
def test_task_telling_is_built_from_the_same_gist_and_omitted_structurally(tmp_path, remote):
    from tests import _memory_inventory_shared as shared

    rooms = shared.world(tmp_path)
    chat = rooms["beta"] if remote else 1
    old, latest = "An earlier outcome.", "The final outcome holds. Its detail stays here."
    rows = [shared.msg("2026-10-20T01:00:00+00:00", "A substantial reply. " * 150,
                       chat_id=chat, direction="out", task_id="done"),
            shared.msg("2026-10-20T01:01:00+00:00", "Completed.", chat_id=chat, direction="system",
                       task_id="done", type="task_summary", summary_kind="terminal_root_projection", wake_summary=old),
            shared.msg("2026-10-20T01:02:00+00:00", "Completed again.", chat_id=chat, direction="system",
                       task_id="done", type="task_summary" if remote else "project_completion_summary",
                       summary_kind="terminal_root_projection", wake_summary=latest, fold_read=remote)]
    shared.append(tmp_path / "logs" / "chat.jsonl", *rows)
    task = {"id": "next-turn", "chat_id": 1, "_is_direct_chat": True}
    snap = mv.capture_memory_view(tmp_path, task, mv.view_spec_for_task(task, tmp_path))
    if remote:
        group = next(g for g in snap.conversations if any(r["room_id"] == str(chat) for r in g["rooms"]))
        from ouroboros import memory_view_conversations as conv
        text = conv.render_group(group, set(), set(), 0)
        assert any(r["folded"] for r in group["rooms"]), [(r["room_id"], r["folded"]) for r in group["rooms"]]
        assert text.count(work.first_sentence(latest)) == 1
        projection = next(t for r in group["rooms"] for t in r["tasks"] if t["task"] == "done")
    else:
        level = mf.level_with(snap, mv.FULL_VIEW, ["F2:work:done:final"])
        text = mv._room_text(snap.room, dict(level.addressed))
        assert old not in text and text.count(latest) == 1
        projection = next(t for t in snap.room["lane2"] if t["task"] == "done")
        assert latest in projection["line"]
    assert "get_task_result(task_id='done')" in projection["line_without_gist"]
    assert work.first_sentence(latest) not in projection["line_without_gist"]


@pytest.mark.parametrize("fixed", ["repeat", "place_fact"])
def test_repeat_and_place_fact_lines_have_no_individual_concession(fixed):
    from ouroboros import memory_view_conversations as conv

    group = syn.conversation(1, words=1)
    word = group["rooms"][0]["words"][0]
    word.update(line="A fixed fact of this place.", text="A fixed fact of this place.")
    word.update({"again": "row:earlier"} if fixed == "repeat" else {"kind": "place_fact"})
    snap = syn.snapshot(conversations=[group])
    assert not any(s == "F6w" for s, _i, *_rest in mf.floor_elements(snap))
    assert word["line"] in conv.render_group(group, set(), set(), 0)
    for form in (1, 2, 3):
        assert word["line"] not in conv.render_group(group, set(), set(), form)
    counted = mv.FloorLevel((("F6a", (group["key"],)),))
    assert word["line"] not in mv.render_room(snap, counted)
    assert "place facts" not in forms.form_counts(snap, counted)["host"]


def test_emergency_conversation_line_names_omitted_notes_only_when_present():
    from ouroboros import memory_view_conversations as conv

    for notes in (0, 2):
        group = syn.conversation(1, notes=notes)
        for form in (2, 3):
            text = conv.render_group(group, set(), set(), form)
            emergency = next(line for line in text.splitlines() if line.startswith("Emergency:"))
            assert ("2 notes by count" in emergency) is bool(notes)
            assert "0 notes" not in emergency


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


def test_unread_summary_does_not_block_return_of_nearest_conversations():
    groups = [syn.conversation(i, words=5, word_chars=1000) for i in range(10)]
    snap = syn.snapshot(conversations=groups)
    keys = tuple(g["key"] for g in groups)
    restored = mv.FloorLevel((("F6g", keys), ("F6a", (*keys[:-1], "unread"))))
    bound = size(snap, restored) + 60
    level = mf.fit_memory_view(snap, frame(bound))
    chosen = dict(level.addressed).get("F6a", ())
    assert "unread" in chosen
    counted = [i for i, key in enumerate(keys) if key in chosen]
    assert 0 < len(counted) < len(keys)
    assert counted == list(range(len(counted)))
    assert size(snap, level) <= bound


def test_unread_summary_is_not_taken_without_a_saving():
    group = syn.conversation(1, words=1, word_chars=1000)
    group["rooms"][0]["words"][0].update(address="row:a", author="Ann", text="Hello.\n" + "A later paragraph. " * 20)
    snap = syn.snapshot(conversations=[group])
    counted = mv.FloorLevel((("F6a", (group["key"],)),))
    global_line = mv.FloorLevel((("F6a", (group["key"], "unread")),))
    assert size(snap, global_line) >= size(snap, counted)
    level = mf.fit_memory_view(snap, frame(0))
    assert group["key"] in dict(level.addressed)["F6a"]
    assert "unread" not in dict(level.addressed)["F6a"]


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
    assert len(held["largest"]) == 3 and all(reader.startswith("memory_read(") for _tokens, reader in held["largest"])
    facts = mf.view_facts(snap, level, window_tokens=200000, mode="low", allowances={})["floor"]
    assert facts["forms"] == counts and facts["holding"] == forms.holding_room(snap, level)
    note = mf.floor_note(level, window_tokens=200000, mode="low", snapshot=snap)
    assert all(x in note for x in ("my telling", "host facts", "emergency without", "Holding the room"))


def test_view_facts_do_not_publish_working_parts_and_keep_the_note_bytes(tmp_path):
    from types import SimpleNamespace
    from ouroboros import observability, working_checkpoint

    snap = syn.snapshot(story=syn.pages(3, 40))
    level = mv.FloorLevel((("F5l", ("page-000",)),))
    note = (
        "### Physical floor\n"
        "Low cuts for the window (200000 tokens). Reduced forms — my telling: none; host facts: none; "
        "emergency without my telling: story openings: 2 lines/1 sources. Read sources with memory_read.\n"
        "Holding the room (tokens): pages without part 91, records without lead 91.\n"
        "31 tokens: memory_read(node_id='page-000')\n"
        "30 tokens: memory_read(node_id='page-001')\n"
        "30 tokens: memory_read(node_id='page-002')"
    )
    assert mf.floor_note(level, window_tokens=200000, mode="low", snapshot=snap).encode() == note.encode()
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="receipt", task_attempt=1)
    state = {"messages": [{"role": "user", "content": "Frozen working message"}], "trace": {}}
    encode = observability._json_bytes
    assert working_checkpoint._encode_working(ctx, state, encode)[2] == 1
    for candidate in (mv.FULL_VIEW, level, mf.fit_memory_view(snap, frame(0))):
        facts = mf.view_facts(snap, candidate, window_tokens=200000, mode="low", allowances={})
        state["trace"]["memory_view"] = facts
        _data, _whole, written = working_checkpoint._encode_working(ctx, state, encode)
        assert written == 0, "view facts (including forms, holding and counts) must stay inline"


@pytest.mark.parametrize("row,author,task", [
    ({"task_id": "direct"}, {}, "direct"),
    ({"transport": {"origin": {"task_id": "origin"}}}, {}, "origin"),
    ({"type": "relay_entry", "task_id": "recipient", "relay": {"sender": {"task_id": "sender"}}}, {}, "sender"),
    ({"type": "relay_entry", "task_id": "recipient"}, {}, ""),
    ({"task_id": "arrival-turn"}, {"via": "own_account"}, ""),
    ({"type": "presence_delivery", "direction": "out", "transport": {"origin": {"task_id": "sender"}}}, {}, "sender"),
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
    assert whole == {'task summary': 2, 'fold_read': 1}
    for suffix in ('progress', 'final'):
        level = mf.level_with(snap, mv.FULL_VIEW, ['F2:work:done:' + suffix])
        assert view_counts(snap, level)['retellings'] == {'task summary': 1, 'works': 1, 'fold_read': 1}
    counted = mv.FloorLevel((('F6a', (snap.conversations[0]['key'],)),))
    assert view_counts(snap, counted)['retellings'] == {'task summary': 1}


@pytest.mark.parametrize("form", [1, 2, 3])
def test_conversation_lines_keep_quiet_rooms_only_in_the_identity_count(form):
    from ouroboros import memory_view_conversations as conv
    group = syn.conversation(1, words=1, notes=1)
    group["rooms"][0]["notes"][0]["lead_rule"] = 1
    quiet = syn.conversation(2, engaged=False)["rooms"][0]
    group["rooms"].append(quiet)
    full = conv.render_group(group, set(), set(), 0)
    line = conv.render_group(group, set(), set(), form)
    assert quiet["silent"] in full
    assert quiet["silent"] not in line and quiet["label"] not in line
    assert "1 other;" in line
    assert "0 works" not in line
    snap = syn.snapshot(conversations=[group])
    ident = ("" if form == 1 else "all:" if form == 2 else "last:") + group["key"]
    counts = forms.form_counts(snap, mv.FloorLevel((("F6g", (ident,)),)))
    assert "silent rooms" not in counts["host"]


def test_repeats_never_add_unread_sources_but_unconsidered_words_do():
    from ouroboros import memory_view_conversations as conv
    for genuine in (False, True):
        group = syn.conversation(1, words=2)
        remote = group["rooms"][0]
        remote["words"][0].update(again="row:earlier", line="Already delivered here.")
        if not genuine:
            remote["words"].pop()
        snap = syn.snapshot(conversations=[group])
        for form in (1, 2, 3):
            text = conv.render_group(group, set(), set(), form)
            assert ("No recorded consideration: 1 words;" in text) is genuine
            assert "2 words;" not in text
        for aggregate in (False, True):
            level = mv.FloorLevel((("F6a", (group["key"], "unread") if aggregate else (group["key"],)),))
            counted = forms.form_counts(snap, level)
            entry = counted["window"].get("unread count") if aggregate else counted["host"].get("unread")
            assert (entry or {}).get("sources", 0) == int(genuine)


@pytest.mark.parametrize("kind", ["work", "gist", "unread"])
@pytest.mark.parametrize("number", [1, 2])
def test_compact_source_heads_read_the_named_rows_and_introduce_the_reader_once(tmp_path, kind, number):
    import re
    from ouroboros.chat_chain import format_address, parse_address, row_address
    from ouroboros import memory_view_conversations as conv
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.registry import ToolContext
    from tests import _memory_inventory_shared as shared

    shared.world(tmp_path)
    room_id = 1 if kind == "work" else 777
    rows = [shared.msg(f"2026-10-20T01:0{i}:00+00:00", f"Source message {i}. " * 80, chat_id=room_id,
                      direction="in" if kind == "unread" else "out", task_id="done",
                      **({"source": "presence:telegram", "transport": {"provider": "telegram", "actor": {"display_name": "Ann"}}}
                         if kind == "unread" else {})) for i in range(number)]
    terminal = shared.msg("2026-10-20T01:03:00+00:00", "Completed.", chat_id=room_id, direction="system",
                          task_id="done", type="task_summary", summary_kind="terminal_root_projection",
                          wake_summary="Completed outcome. A second sentence.")
    shared.append(tmp_path / "logs" / "chat.jsonl", *rows, *([] if kind == "unread" else [terminal]))
    task = {"id": "next-turn", "chat_id": 1, "_is_direct_chat": True}
    snap = mv.capture_memory_view(tmp_path, task, mv.view_spec_for_task(task, tmp_path))
    if kind == "work":
        before = mv._room_text(snap.room, {})
        text = mv._room_text(snap.room, {"F2": {"work:done:final"}})
        line = next(line for line in text.splitlines() if "work done]" in line)
        assert f"{number} replies," in line
        assert "Completed work replies may stand as their gist." in text
    else:
        group = next(g for g in snap.conversations if any(r["room_id"] == str(room_id) for r in g["rooms"]))
        before = "\n\n".join(conv.pieces([group], {}, lambda _key, make: make()))
        text = "\n\n".join(conv.pieces([group], {"F6g": {"all:" + group["key"]}}, lambda _key, make: make()))
        line = next(line for line in text.splitlines() if ("«Completed outcome." in line if kind == "gist"
                                                         else line.startswith("No recorded consideration:")))
    reader = "from=<first>, to=<last>"
    assert reader not in before and text.count(reader) == 1
    assert "memory_read(" not in line
    expected = [format_address(row_address(row)) for row in ([terminal] if kind == "gist" else rows)]
    found = re.findall(r"row:\d+@[^\s;'\])]+#[a-f0-9]+", line.split("«", 1)[0])
    assert found == ([expected[0]] if len(expected) == 1 else [expected[0], expected[-1]])
    assert all(line.count(parse_address(address)["ts"]) == 1 for address in found)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="reader", current_chat_id=1)
    read = _memory_read(ctx, room_id=str(parse_address(found[0])["chat_id"]), rows=True,
                        **{"from": found[0], "to": found[-1]})
    assert "complete: no further rows" in read
    assert re.findall(r"row:\d+@[^\s;'\])]+#[a-f0-9]+", "\n".join(read.splitlines()[2:])) == expected


def test_work_range_stands_at_the_last_reply_without_repeating_its_time():
    snap, _gist = completed_room()
    middle = syn.spoken(1, "human", 50)
    snap.room["lane1"] = [snap.room["lane1"][0], middle, snap.room["lane1"][-1]]
    rows = [r for r in snap.room["lane1"] if r["kind"] == "ouroboros"]
    text = mv._room_text(snap.room, {"F2": {"work:done:final"}})
    line = next(line for line in text.splitlines() if "work done]" in line)
    assert text.index(middle["line"]) < text.index(line)
    assert f"{rows[0]['address']} → {rows[-1]['address']}" in line.split("«", 1)[0]
    assert line.count(rows[0]["ts"]) == line.count(rows[-1]["ts"]) == 1


def test_task_gist_reader_uses_the_source_room_when_its_terminal_is_elsewhere(tmp_path):
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.tools.chronicle import _memory_read
    from ouroboros.tools.registry import ToolContext
    from tests import _memory_inventory_shared as shared

    shared.world(tmp_path)
    terminal = shared.msg('2026-10-20T02:00:00+00:00', 'Terminal in another room.', chat_id=777,
                          direction='system', task_id='done', type='task_summary',
                          summary_kind='terminal_root_projection', wake_summary='A completed result.')
    shared.append(tmp_path / 'logs/chat.jsonl', terminal)
    snap, _gist = completed_room()
    address = format_address(row_address(terminal))
    work.bind(snap.room, [], {'done': dict(summary=terminal['wake_summary'], address=address, room_id='777')})
    text = mv._room_text(snap.room, {'F1': {snap.room['lane2'][0]['first']}})
    assert f'[{address}; task done]' in text
    assert 'memory_read(room_id=<room from address>, rows=true, from=<address>, to=<address>)' in text
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id='reader', current_chat_id=1)
    read = _memory_read(ctx, room_id='777', rows=True, **{'from': address, 'to': address})
    assert terminal['text'] in read and 'complete: no further rows' in read
