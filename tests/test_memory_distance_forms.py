"""Distance order, clean starts, work tellings and final receipts of version 2 forms."""
from copy import deepcopy

import pytest

from ouroboros import memory_floor as mf, memory_forms as forms, memory_view as mv
from ouroboros import memory_view_work as work
from tests import _memory_view_synthetic as syn


def test_remote_work_ranges_stay_in_their_source_rooms_and_read_those_replies(tmp_path):
    from ouroboros import chat_chain, dialogue_store as ds
    from ouroboros import memory_view_conversation_forms as conv
    from tests import _memory_inventory_shared as shared
    from tests.test_memory_waiting import _event_row, _event_group, view, _ctx, _memory_read

    shared.world(tmp_path)
    rows = []
    for i, thread in enumerate(("first", "second", "first"), 1):
        row = _event_row("harbor", thread, i, direction="out", type="presence_delivery",
                         task_id="range001", text=f"Outward answer number {i}.")
        ds.append(tmp_path, "chat", row)
        rows.append(row)
    ds.append(tmp_path, "chat", _event_row("harbor", "second", 4, direction="system", type="task_summary",
              task_id="range001", wake_summary="The shared finding. Extra detail.", status="completed"))
    group = _event_group(view(tmp_path))
    address = lambda row: chat_chain.format_address(chat_chain.row_address(row))
    expected = [work.row_range(address(rows[0]), address(rows[2])), address(rows[1])]
    # Membership in another room must not change the address's room or add another range.
    original = next(w for r in group["rooms"] for w in r["words"] if w["address"] == address(rows[0]))
    other = next(r for r in group["rooms"] if r["room_id"] != str(rows[0]["chat_id"]))
    other["words"].append(dict(original, id="membership-copy", again=original["address"]))
    data = conv._sources(group)
    assert data["replies"]["range001"] == ", ".join(expected)
    for form in (1, 2, 3):
        assert f"[{', '.join(expected)}] «The shared finding.»" in conv.render_group(group, set(), set(), form)
    for span in expected:
        endpoints = span.split(" → ")
        first, last = endpoints[0], endpoints[-1]
        room = chat_chain.parse_address(first)["chat_id"]
        assert chat_chain.parse_address(last)["chat_id"] == room
        text = _memory_read(_ctx(tmp_path, "root0001"), room_id=str(room), rows=True, **{"from": first, "to": last})
        for row in rows:
            assert (row["text"] in text) is (row["chat_id"] == room)
    whole = conv.render_group(group, set(), set(), 0)
    assert "get_task_result(task_id='range001')" in whole
    assert all(row["text"] in whole for row in rows)


@pytest.mark.parametrize("short", ["F5l", "F5", "F5a"])
def test_short_story_forms_keep_human_quotes_until_the_owner_stage(short):
    page = syn.pages(1, 3000)[0]
    human = dict(speaker="human", address=syn.spoken(1, "human", 40)["address"], text="Keep the original promise.")
    mine = dict(speaker="ouroboros", address=syn.spoken(2, "ouroboros", 40)["address"], text="I will finish this.")
    page.update(quotes=[human, mine], lead_rule=True)
    snap = syn.snapshot(story=[page])
    captured = mv.snapshot_json(snap)
    whole = mv.render_story(snap)
    assert mv.quote_line(human) in whole and mv.quote_line(mine) in whole
    steps = (("F5", (page["id"],)), ("F5a", ("story",))) if short == "F5a" else ((short, (page["id"],)),)
    level = mv.FloorLevel(steps)
    text = mv.render_story(snap, level)
    assert text.count(mv.quote_line(human)) == 1
    assert mv.quote_line(mine) not in text and page["text"] not in text
    assert mv.story_chars(snap, level) == mv.json_chars(text)
    quotes = [(ident, full) for step, ident, full, reduced in mf.floor_elements(snap)
              if step == "F7" and full == mv.quote_line(human) and not reduced]
    assert len(quotes) == 1
    ident = quotes[0][0]
    assert forms.stage_of(snap, "F7", ident) == "owner"
    final = mf.level_with(snap, level, ["F7:" + ident])
    text = mv.render_story(snap, final)
    assert mv.quote_line(human) not in text
    assert mv.story_chars(snap, final) == mv.json_chars(text)
    assert forms.form_counts(snap, final)["emergency"]["story quotes"] == {"lines": 0, "sources": 1}
    assert mv.snapshot_json(snap) == captured


@pytest.mark.parametrize("hidden", ["none", "notes", "words", "older gist"])
def test_emergency_conversation_forms_require_hidden_sources(hidden):
    from ouroboros import memory_view_conversations as conv

    snap, gist = completed_room()
    group = snap.conversations[0]
    remote = group["rooms"][0]
    if hidden == "notes":
        remote["notes"] = syn.conversation(8, notes=1)["rooms"][0]["notes"]
    elif hidden == "words":
        remote["words"][0].pop("gist")
    elif hidden == "older gist":
        remote["tasks"].append(dict(remote["tasks"][0], task="later", work_id="later",
                                    gist=dict(gist, summary="Another completed work.", ts="2027-01-01")))
    elements = {i for s, i, *_ in mf.floor_elements(snap) if s == "F6g"}
    assert ("all:" + group["key"] in elements) is (hidden in ("notes", "words"))
    assert ("last:" + group["key"] in elements) is (hidden != "none")
    for form, prefix in ((2, "all:"), (3, "last:")):
        level = mv.FloorLevel((("F6g", (prefix + group["key"],)),))
        text = conv.render_group(group, set(), set(), form)
        missing = hidden in ("notes", "words") or hidden == "older gist" and form == 3
        assert ("Emergency:" in text) is missing
        assert bool(forms.form_counts(snap, level)["emergency"]) is missing


@pytest.mark.parametrize("extra", [0, 20])
def test_walk_does_not_take_a_form_without_savings(monkeypatch, extra):
    snap = syn.snapshot(story=syn.pages(1, 3000))
    monkeypatch.setattr(mf, "LADDER", ("F5",))
    table = mf._elements(snap)
    table.__dict__["priced"] = (("F5", "page-000", 20, 20 + extra),)
    level = mf.fit_memory_view(snap, frame(0))
    assert level == mv.FULL_VIEW


def test_first_counted_conversation_can_pay_the_shared_header(monkeypatch):
    groups = [syn.conversation(i) for i in range(3)]
    snap = syn.snapshot(conversations=groups)
    monkeypatch.setattr(mf, "LADDER", ("F6a",))
    # The first count costs more; the second amortizes the shared header.
    def price(level):
        return (300, 310, 200, 100)[len(dict(level.addressed).get("F6a", ()))]
    level = mf.fit_memory_view(snap, frame(250), measure=price)
    assert dict(level.addressed)["F6a"] == tuple(g["key"] for g in groups[:2])


def test_three_sentence_completion_is_whole_here_and_one_sentence_in_another_conversation(tmp_path):
    from ouroboros import dialogue_store as ds, memory_view_conversations as conv
    from tests import _memory_inventory_shared as shared
    rooms = shared.world(tmp_path)
    gist = "The API now preserves its callers. The migration has passed. I left one measured follow-up."
    for row in [shared.msg("2026-09-09T01:01:00Z", "I delivered the change.", direction="out",
                           task_id="done001", root_task_id="done001", delegation_role="root"),
                shared.msg("2026-09-09T01:02:00Z", "Done", direction="system", type="task_summary",
                           task_id="done001", root_task_id="done001", summary_kind="terminal_root_projection",
                           status="completed", wake_summary=gist)]:
        ds.append(tmp_path, "chat", row)
    def capture(chat):
        task = {"id": "reader001", "chat_id": chat}
        return mv.capture_memory_view(tmp_path, task, mv.view_spec_for_task(task, tmp_path))
    own = capture(1)
    task = next(t for t in own.room["lane2"] if t["task"] == "done001")
    assert gist in task["line"]
    short = mv.render_room(own, mv.FloorLevel((("F2", ("work:done001:final",)),)))
    assert gist in short and "; work done001]" in short
    remote = next(g for g in capture(rooms["alpha"]).conversations if g["key"] == "main")
    line = next(t["line"] for r in remote["rooms"] for t in r["tasks"] if t["task"] == "done001")
    assert work.first_sentence(gist) in line and "The migration has passed." not in line
    assert "The migration has passed." not in conv.render_group(remote, set(), set(), 0)


def test_page_and_note_leads_share_the_sentence_boundary_without_rewriting_the_words():
    from ouroboros.text_sentences import first_sentence
    text = "A thought on\ntwo lines. Another sentence! A third?"
    page = dict(syn.pages(1, 500)[0], text=text, lead_rule=True)
    note = {"id": "note1", "role": "root", "date": "2026-11-11", "text": text, "lead_rule": True}
    assert mv._indented(first_sentence(text)) in mv._page_label(page)
    assert ' '.join(first_sentence(text).split()) in mv._note_pointer(note)
    assert "Another sentence!" not in mv._page_label(page) + mv._note_pointer(note)


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


def test_a_told_word_yielding_under_its_task_line_is_counted_as_my_telling():
    """rev_fable REV2 7: a told word of another conversation that yielded under its work's task line, whose sentence
    stands in the same block, was in no bin of the floor note ("my telling: none; host facts: none; emergency…:
    none"). It is my telling, counted once; a word whose gist has no task line here keeps the work-gist count."""
    snap, gist = completed_room()
    level = mf.fit_memory_view(snap, frame(size(snap) - 1))
    assert [step for step, _ids in level.addressed] == ["F6w"] and len(dict(level.addressed)["F6w"]) == 1
    counts = forms.form_counts(snap, level)
    assert counts["retold"] == {"task gists": {"lines": 1, "sources": 1}} and not counts["emergency"]
    assert work.first_sentence(gist["summary"]) in mv.render_room(snap, level)
    note = mf.floor_note(level, window_tokens=size(snap), mode="max", snapshot=snap)
    assert "my telling: task gists: 1 lines/1 sources;" in note
    other, _gist = completed_room()
    other.conversations[0]["rooms"][0]["tasks"] = []  # no task line carries the gist: its gist line tells the word
    counts = forms.form_counts(other, mf.fit_memory_view(other, frame(size(other) - 1)))
    assert "task gists" not in counts["retold"] and counts["retold"]["work gists"] == {"lines": 1, "sources": 1}


@pytest.mark.parametrize("replies", [False, True])
def test_remote_gist_addresses_its_replies_here_or_its_telling_when_none(replies):
    from ouroboros import memory_view_conversation_forms as conv
    snap, gist = completed_room()
    group = snap.conversations[0]
    remote = group["rooms"][0]
    words = remote["words"]
    address = work.row_range(words[0]["address"], words[-1]["address"]) if replies else gist["address"]
    if not replies:
        remote["words"] = []
    for form in (1, 2, 3):
        text = conv.render_group(group, set(), set(), form)
        assert f"[{address}] «{work.first_sentence(gist['summary'])}»" in text
        if replies:
            assert gist["address"] not in text
    whole = conv.render_group(group, set(), set(), 0)
    assert remote["tasks"][0]["line"] in whole
    assert all(word["line"] in whole for word in remote["words"])
    # A relay's telling belongs to the delivery mark, never to its sender's terminal.
    word = dict(words[0], relay_summary="I sent the public outcome.")
    remote.update(words=[word], tasks=[])
    work.bind(None, [group], {"done": gist})
    assert f"[{word['address']}] «I sent the public outcome.»" in conv.render_group(group, set(), set(), 1)


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
            assert receipt["emergency"]["untold and older gists by count"]["sources"] >= 1
        else:
            replaced = any(w["id"] in gone.get("F6w", ()) for w in remote["words"][:2])
            assert text.count(work.first_sentence(gist["summary"])) == int(bool(form or replaced or remote_task))
            if form or replaced and not remote_task:
                address = work.row_range(remote["words"][0]["address"], remote["words"][1]["address"]) if form else gist["address"]
                assert f"[{address}]" in text


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
            emergency = next((line for line in text.splitlines() if line.startswith("Emergency:")), "")
            assert bool(emergency) is bool(notes)
            assert ("2 notes by count" in emergency) is bool(notes)
            assert "0 notes" not in emergency and "0 untold" not in emergency  # rev_grok REV1 R4: a zero names nothing
    group = syn.conversation(1)
    group["rooms"][0]["tasks"] = [{"task": f"w{n}", "line": f"[t; host; task w{n}] Done", "pos": n, "gist": {
        "summary": f"Work {n} finished.", "address": f"row:2001@2026-09-0{n + 1}T00:00:00+00:00#{n:012x}",
        "ts": f"2026-09-0{n + 1}T00:00:00+00:00"}} for n in range(2)]
    text = conv.render_group(group, set(), set(), 3)
    assert "Emergency: 2 works, only the last gist shown." in text and "untold" not in text


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


@pytest.mark.parametrize("reply_here", [False, True])
def test_a_task_whose_only_rows_here_are_relay_notes_keeps_its_gist_until_a_counted_emergency(reply_here):
    """rev_fable REV2 F4: a relay note carries its sender's task id but is its own work (``relay:<address>``), so a
    task whose only rows here were such notes counted as answered: F1 took its line at the ordinary task-facts stage
    and its gist, printed nowhere, counted as a host fact. That gist is the task's sole telling here now; a task whose
    own reply stands here stays an ordinary fact beside its work."""
    task = syn.task_line(0)
    task.update(task="sender", gist={"summary": "Only telling.", "room_id": "1", "address": task["last"]})
    task["line"] += " — summary: «Only telling.»"
    rows = [dict(syn.spoken(1, "ouroboros", 400), task_id="sender", relay_summary="Passed on a limit.")]
    rows += [dict(syn.spoken(2, "ouroboros", 400), task_id="sender")] if reply_here else []
    room = {**syn.room(lane1=rows), "lane2": [task]}
    work.bind(room, [], {"sender": task["gist"]})
    snap = syn.snapshot(room_facts=room)
    assert rows[0]["work_id"] == "relay:" + rows[0]["address"]
    assert forms.stage_of(snap, "F1", task["first"]) == ("task_facts" if reply_here else "task_only")
    counts = forms.form_counts(snap, mf.fit_memory_view(snap, frame(0)))
    if reply_here:
        assert counts["host"]["task facts"] == {"lines": 1, "sources": 1} and "sole task gists" not in counts["emergency"]
    else:
        assert counts["emergency"]["sole task gists"] == {"lines": 1, "sources": 1} and "task facts" not in counts["host"]
        assert ("F1", task["first"]) not in forms.ordinary_pairs(snap)


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
    assert {"accounts", "parts", "pages without part", "untold in this room",
            "owner words in other rooms", "notes", "room records"} <= held["tokens"].keys()
    assert "records without lead" not in held["tokens"] and held["without_lead"]
    assert len(held["largest"]) == 3 and all(": memory_read(" in reader for _tokens, reader in held["largest"])
    facts = mf.view_facts(snap, level, window_tokens=200000, mode="low", allowances={})["floor"]
    assert facts["forms"] == counts and facts["holding"] == forms.holding_room(snap, level)
    note = mf.floor_note(level, window_tokens=200000, mode="low", snapshot=snap)
    assert all(x in note for x in ("my telling", "host facts", "emergency without", "Holding the room"))


def test_holding_counts_hidden_untold_words_and_ranks_units_with_readers():
    from ouroboros.memory_form_facts import holding_lines
    remote = syn.conversation(1, words=2, word_chars=50000)
    for word in remote["rooms"][0]["words"]:
        word["rank"] = 1
    owner = syn.conversation(2, words=1, word_chars=20000, owner=True)
    room = syn.room(lane1=[syn.spoken(1, "ouroboros", 60000)])
    snap = syn.snapshot(room_facts=room, story=syn.pages(3, 500), conversations=[remote, owner])
    expected = mf.view_tokens("\n".join(w["line"] for w in remote["rooms"][0]["words"]))
    for level in (mv.FULL_VIEW, mv.FloorLevel((("F6g", ("all:" + remote["key"],)),)),
                  mv.FloorLevel((("F6a", (remote["key"],)),))):
        held = forms.holding_room(snap, level)
        assert held["tokens"]["untold in other conversations"] == expected
        assert held["counts"]["untold in other conversations"] == {"conversations": 1, "counted_rows": 2 if level.addressed else 0}
        # The number is what folding them would tell, the whole text, though most may stand by count (rev_fable 7).
        rows = "; 2 rows by count" if level.addressed else ""
        for line in (holding_lines(held)[0], holding_lines(held, short=True)[0]):
            assert f"untold in other conversations {expected:,} at whole-text size" in line
            assert line.count("at whole-text size") == 1
        assert f"at whole-text size (1 conversations{rows})" in holding_lines(held)[0]
        assert list(held["tokens"].values()) == sorted(held["tokens"].values(), reverse=True)
        largest = held["largest"]
        assert [text.split(", ", 1)[0] for _n, text in largest] == [
            "untold in other conversations", "untold in this room", "owner words in other rooms"]
        assert f"{remote['label']}: memory_read(room_id='2001', rows=true, from='{remote['rooms'][0]['words'][0]['address']}'" in largest[0][1]
        assert f"from='{room['lane1'][0]['address']}'" in largest[1][1]
        assert f"{owner['label']}: memory_read(room_id='2002', rows=true, from='{owner['rooms'][0]['words'][0]['address']}'" in largest[2][1]


@pytest.mark.parametrize("mode,physical", [("low", ""), ("nano", ""), ("low", "F7"), ("low", "F6a")])
def test_holding_note_keeps_unit_readers_only_above_the_physical_floor(mode, physical):
    page = syn.pages(1, 4000)[0]
    remote = syn.conversation(1, words=2, word_chars=5000)
    for word in remote["rooms"][0]["words"]:
        word["rank"] = 1
    room = syn.room(lane1=[syn.spoken(1, "human", 1000)])
    snap = syn.snapshot(story=[page], room_facts=room, conversations=[remote])
    cuts = [("F5", (page["id"],))]
    if physical == "F7":
        cuts.append(("F7", (room["lane1"][0]["address"],)))
    elif physical == "F6a":
        cuts.append(("F6a", (remote["key"],)))
    note = mf.floor_note(mv.FloorLevel(tuple(cuts)), mode=mode, snapshot=snap,
                         window_tokens=200000, tool_names=["enable_tools"])
    holding = note[note.index("Holding the room (tokens):"):].splitlines()
    assert "untold in other conversations" in holding[0]
    compact = mode == "nano" or bool(physical)
    assert any("tokens, " in line and "memory_read(" in line for line in holding) is not compact
    assert ("(" in holding[0].removeprefix("Holding the room (tokens):")) is not compact
    if mode == "nano":
        assert "memory_read is reachable through enable_tools" in note


def test_holding_records_have_one_kind_and_one_reader_even_when_present_in_both_blocks():
    page = syn.pages(1, 4000)[0]
    room = syn.room(under=1)
    room["under_parts"] = [dict(page, part="part1", standing=False)]
    snap = syn.snapshot(story=[page], room_facts=room)
    held = forms.holding_room(snap, mv.FULL_VIEW)
    assert "records without lead" not in held["tokens"]
    assert held["without_lead"] == {"pages without part": 1}
    assert len(held["largest"]) == 1
    assert held["largest"][0][0] == held["tokens"]["pages without part"]
    assert "memory_read(node_id='page-000')" in held["largest"][0][1]


def test_holding_largest_units_merge_kinds_with_the_same_reader():
    from dataclasses import replace
    snap, _gist = completed_room()
    room = snap.room
    room["lane1"].append(syn.spoken(5, "human", 4000))
    room["lane2"] = []
    snap = replace(snap, conversations=[], story=syn.pages(2, 1500))
    held = forms.holding_room(snap, mv.FULL_VIEW)
    units = held["largest"]
    readers = [description.split(": memory_read(", 1)[1] for _n, description in units]
    assert len(readers) == len(set(readers)) == 3
    own = next((tokens, description) for tokens, description in units if "rows=true" in description)
    assert "replies of completed works" in own[1] and "untold in this room" in own[1]
    assert own[0] == held["tokens"]["replies of completed works"] + held["tokens"]["untold in this room"]
    assert all(any(f"memory_read(node_id='page-{i:03d}')" in description for _n, description in units[1:]) for i in range(2))
    # Different readers remain separate even if their source kind is identical.
    assert units[1][1] != units[2][1] and all("pages without part" in unit[1] for unit in units[1:])


def test_holding_includes_whole_remote_replies_until_their_gist_replaces_them():
    snap, _gist = completed_room()
    snap.room["lane1"] = []
    group = snap.conversations[0]
    whole = forms.holding_room(snap, mv.FULL_VIEW)
    expected = mf.view_tokens("\n".join(w["line"] for w in group["rooms"][0]["words"]))
    assert whole["tokens"]["replies of completed works"] == expected
    line = mv.FloorLevel((("F6g", (group["key"],)),))
    reduced = forms.holding_room(snap, line)
    assert "replies of completed works" not in reduced["tokens"]
    assert reduced["tokens"]["work gists"] > 0


def test_holding_note_names_a_sole_task_gist_lost_even_without_another_cut():
    snap, _gist = completed_room()
    snap.room["lane1"] = []
    task = snap.room["lane2"][0]
    level = mv.FloorLevel((("F1", (task["first"],)),))
    for chosen in (mv.FULL_VIEW, level):
        note = mf.floor_note(chosen, snapshot=snap, mode="low", window_tokens=200000)
        assert ("Holding the room" in note) is bool(chosen.addressed)


@pytest.mark.parametrize("cause", ["crowded", "route", "held"])
def test_holding_note_also_names_memory_lowering_without_any_replacement(cause):
    from tests.test_memory_low_start import _room
    snap = syn.snapshot(story=syn.pages(1, 500))
    text = mf.floor_note(mv.FULL_VIEW, snapshot=snap, window_tokens=1000000, mode="low",
                         lowered_from="max", max_room=_room(cause))
    assert ("Holding the room" in text) is (cause == "crowded")
    assert "Reduced forms" not in text
    group = syn.conversation(1, words=2)
    snap = syn.snapshot(conversations=[group])
    unread = mv.FloorLevel((("F6a", (group["key"], "unread")),))
    text = mf.floor_note(unread, snapshot=snap, window_tokens=200000, mode="low")
    assert "Window emergency over unread words" in text and "Holding the room" not in text


@pytest.mark.parametrize("source,needs_telling", [("unread", False), ("repeat", False), ("place_fact", False), ("considered", True)])
def test_a_conversation_count_keeps_its_category_but_only_telling_loss_names_holding(source, needs_telling):
    group = syn.conversation(1, words=2)
    for word in group["rooms"][0]["words"]:
        if source == "repeat":
            word["again"] = True
        elif source == "place_fact":
            word["kind"] = "place_fact"
        elif source == "considered":
            word["rank"] = 1
    snap = syn.snapshot(conversations=[group])
    level = mv.FloorLevel((("F6a", (group["key"],)),))
    assert "conversation count" in forms.form_counts(snap, level)["emergency"]
    note = mf.floor_note(level, snapshot=snap, mode="low", window_tokens=200000)
    assert ("Holding the room" in note) is needs_telling


def test_view_facts_do_not_publish_working_parts_and_keep_the_note_bytes(tmp_path):
    from types import SimpleNamespace
    from ouroboros import observability, working_checkpoint

    snap = syn.snapshot(story=syn.pages(3, 40))
    level = mv.FloorLevel((("F5l", ("page-000",)),))
    note = (
        "### Physical floor\n"
        "Low cuts for the window (200000 tokens). Reduced forms — my telling: none; host facts: none; "
        "emergency without my telling: story openings: 2 lines/1 sources. Read sources with memory_read.\n"
        "Holding the room (tokens): pages without part 91 (3 records without lead).\n"
        "31 tokens, pages without part, Main page-000: memory_read(node_id='page-000')\n"
        "30 tokens, pages without part, Main page-001: memory_read(node_id='page-001')\n"
        "30 tokens, pages without part, Main page-002: memory_read(node_id='page-002')"
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


OWNER_LIMIT = "Do not touch the invoices until Friday; only the March batch may move."


def _relayed_limit(root, *, quoted):
    """My human speaks in Main and that row is sealed by a page; the mind then relays a note into a Project's
    running work with a one-line summary, quoting those words or not. The Project root's captured view."""
    import dataclasses
    import json
    from ouroboros import dialogue_store as ds
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.owner_relay import owner_quote, relay_envelope
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.chronicle import _chronicle_write
    from tests import _memory_inventory_shared as shared
    from tests.test_late_review_marks import _ctx
    shared.world(root)
    beta = create_project(root, "beta", name="Beta")
    write_task_result(root, "recipient", "running", chat_id=1, title="recipient")
    bind_task_to_project(root, "recipient", beta["id"], beta["chat_id"], origin={"absent": "system"})
    owner = {"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:01:00Z", "client_message_id": "owner-limit",
             "text": OWNER_LIMIT}
    ds.append(root, "chat", owner)
    relay = {"chat_id": beta["chat_id"], "direction": "out", "type": "relay_entry", "task_id": "recipient",
             "ts": "2026-10-01T00:02:00Z", "client_message_id": "relay:limit:recipient",
             "text": "A limit from my human for this work. " + "Context of the note. " * 40,
             "relay": {**relay_envelope("limit", [owner_quote(root, owner)] if quoted else [], chat_id=1,
                                        summary="Passed on a limit about invoices."), "sender": {"task_id": "teller"}}}
    ds.append(root, "chat", relay)
    ds.append(root, "chat", {"chat_id": beta["chat_id"], "direction": "out", "task_id": "recipient",
                             "root_task_id": "recipient", "delegation_role": "root", "ts": "2026-10-01T00:05:00Z",
                             "text": "Working on the batch now. " + "More. " * 60})
    sealed = json.loads(_chronicle_write(_ctx(root, "root0001"), kind="page", room_id="1",
                                         text="The owner set a limit for the Beta work. Details in the rows.",
                                         covers={"to": format_address(row_address(owner))}))
    assert sealed["ok"], sealed
    snap = mv.capture_memory_view(root, {"id": "recipient", "chat_id": beta["chat_id"]}, dataclasses.replace(
        mv.ROLE_DEFAULTS["integrator"], room_id=str(beta["chat_id"])))
    return snap, next(r for r in snap.room["lane1"] if r["address"] == format_address(row_address(relay)))


@pytest.mark.parametrize("quoted", [True, False], ids=["owner-quoted", "note-only"])
def test_a_relay_quoting_my_humans_words_keeps_them_whole_at_every_ordinary_stage(tmp_path, quoted):
    """rev_fable REV2 F1: a relay's summary became the gist of the row that also holds my human's quoted words, so a
    clean start replaced them by the relay summary when their own row was sealed. Such a relay belongs to no work and
    stands whole through the ordinary stages; it yields only in counted emergencies. A note quoting nobody still folds
    into its own summary."""
    snap, row = _relayed_limit(tmp_path, quoted=quoted)
    ordinary = mf.ordinary_level(snap)
    text = mv.render_story(snap, ordinary) + "\n" + mv.render_room(snap, ordinary)
    assert text.count(OWNER_LIMIT) == int(quoted) and text.count(row["address"]) >= 1
    assert text.count("«Passed on a limit about invoices.»") == int(not quoted)
    assert ("owner_quoted" in row, "gist" in row) == (quoted, not quoted)
    if quoted:
        assert not mf.form_counts(snap, ordinary)["emergency"] and row["line"] in text
        # rev_opus REV3D: past the ordinary stages the relay's opening and address yield with my human's words, last.
        assert {forms.stage_of(snap, "F2", ident) for ident in (row["address"], "bare:" + row["address"])} == {"owner"}
        whole = mf.tokens_of(mv.story_chars(snap)) + mf.tokens_of(mv.room_chars(snap))
        def fit(short):
            fitted = mf.fit_memory_view(snap, {"margin": whole - short, "physical": 10**9, "budget": None})
            return dict(fitted.addressed), mv.render_story(snap, fitted) + "\n" + mv.render_room(snap, fitted)
        # A short gap is paid by forms of earlier stages: my last reply here, the other rooms' lines.
        taken, shown = fit(150)
        assert OWNER_LIMIT in shown and row["address"] not in taken.get("F2", ()), taken
        # The relay yields only once every earlier stage has given what it can: the other rooms' counts and my
        # last reply's address went first.
        taken, shown = fit(1200)
        assert "bare:" + row["address"] in taken.get("F2", ()) and OWNER_LIMIT not in shown, taken
        assert len(taken.get("F6a", ())) == 3 and any(i != "bare:" + row["address"] for i in taken["F2"]), taken
    else:
        assert mf.form_counts(snap, ordinary)["retold"]["works"]["sources"] == 1 and row["line"] not in text


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


def test_holding_formats_only_the_three_units_it_shows(monkeypatch):
    from ouroboros import memory_form_facts as facts

    formatted = []
    class Reader(str):
        def __format__(self, spec):
            formatted.append(str(self))
            return super().__format__(spec)

    original = facts._hold_conversations
    def observe(snapshot, gone, hold, price, counts):
        def tracked(kind, text, reader, **kwargs):
            return hold(kind, text, Reader(reader), **kwargs)
        return original(snapshot, gone, tracked, price, counts)
    monkeypatch.setattr(facts, "_hold_conversations", observe)
    snapshot = syn.snapshot(room_facts=syn.room(), conversations=[
        syn.conversation(i, words=1, word_chars=500+i) for i in range(30)])
    held = facts.holding_room(snapshot, mv.FULL_VIEW)
    assert len(held["largest"]) == len(formatted) == 3
    assert len(set(formatted)) == 3
    assert all(reader in description for reader, (_tokens, description) in zip(formatted, held["largest"]))
    assert list(held["tokens"].values()) == sorted(held["tokens"].values(), reverse=True)


@pytest.mark.parametrize("source", ["capture", "shared"])
@pytest.mark.parametrize("shortest", [False, True])
def test_repeated_fits_and_renders_keep_the_captured_payload_and_borrowed_rows(tmp_path, source, shortest):
    import dataclasses
    import json
    from tests import _memory_inventory_shared as shared

    if source == "capture":
        shared.world(tmp_path)
        snapshot = mv.capture_memory_view(tmp_path, {"id": "turn0001", "chat_id": 1}, mv.ROLE_DEFAULTS["integrator"])
    else:
        snapshot = syn.live_scale()
        snapshot.story[0]["quotes"] = [dict(speaker="human", text="Keep this exact promise.",
                                          address=syn.spoken(1, "human", 40)["address"])]
    aliases = {field.name: getattr(snapshot, field.name) for field in dataclasses.fields(snapshot)}
    before = deepcopy(aliases)
    canonical = mv.snapshot_json(snapshot)
    views = []
    for _ in range(2):
        shown = mf.render_view_for_mode(snapshot, mode="low", owner_mode="max", window_tokens=200000,
            known_window=True, output_reserve=10000, ratio=1, non_memory_tokens=180000, shortest=shortest)
        # Each fit is printed independently; prior prices and receipt caches are already populated on the second.
        views.append((shown.story, shown.room, shown.note, json.dumps(shown.facts, sort_keys=True, ensure_ascii=False)))
        assert mv.snapshot_json(snapshot) == canonical
        assert aliases == before
        assert all(getattr(snapshot, name) is value for name, value in aliases.items())
    assert views[0] == views[1]
    assert "_holding_conversations" in snapshot.__dict__
    if source == "shared":
        assert snapshot.__dict__["_story_quotes"]
        assert "carried" not in snapshot.story[0]["quotes"][0]
