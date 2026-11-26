"""Which form each kind of memory takes when the floor takes it, on fixtures of real words (``_memory_view_synthetic``'s
work room, story and place threads): today's forms, so a change to them shows as a change of these expectations."""
from ouroboros import memory_floor as mf, memory_view as mv, memory_view_conversations as conversations
from tests import _memory_view_synthetic as syn


def shorter(snapshot, step):
    """``{element id: the form the floor gives it}`` of one step, in its walk order."""
    return {ident: short for each, ident, _whole, short in mf.floor_elements(snapshot) if each == step}


def test_unretold_replies_have_openings_then_bare_addresses_on_the_physical_floor():
    snapshot = syn.snapshot(room_facts=syn.work_room())
    replies = [r for r in snapshot.room["lane1"] if r["kind"] == "ouroboros"]
    forms = shorter(snapshot, "F2")
    assert {ident.removeprefix("bare:") for ident in forms} == {r["address"] for r in replies}
    for row in replies:
        assert forms["bare:" + row["address"]] == mv._row_pointer(row, "my reply")
        if row["address"] in forms:
            assert "opening" in forms[row["address"]] and row["head"] in forms[row["address"]]
    level = mv.FloorLevel((("F2", tuple(i for i in forms if i.startswith("bare:"))),))
    text = mv.render_room(snapshot, level)
    assert not any(row["line"] in text for row in replies)
    assert snapshot.room["lane1"][-1]["line"] in text


def test_record_openings_are_emergencies_unless_the_writer_received_the_lead_rule():
    from ouroboros import memory_forms
    story = syn.told_story()
    for entry in story:
        if entry.get("lead"):
            entry["lead_rule"] = 1
        if entry.get("kind") != "legacy":
            entry["text"] += " More recorded detail." * 100
    snapshot = syn.snapshot(story=story)
    labels = shorter(snapshot, "F5l")
    assert set(labels) == {e["id"] for e in story if e["kind"] != "legacy"}
    for entry in story[1:]:
        ordinary = "lead_rule" in entry
        assert (memory_forms.stage_of(snapshot, "F5l", entry["id"]) == "story_ordinary") is ordinary
        assert ("record opening, no lead" in labels[entry["id"]]) is not ordinary
    assert set(shorter(snapshot, "F5")) == {e["id"] for e in story[1:]}
    assert story[0]["text"] in mv.render_story(snapshot)


def test_a_legacy_conversation_without_gist_facts_has_only_host_counts():
    both = syn.place_threads()[0]
    sentences = [task["line"].split("«")[1].split("»")[0] for room in both["rooms"] for task in room["tasks"]]
    whole, line = (conversations.render_group(both, set(), set(), as_line) for as_line in (False, True))
    assert len(sentences) == 2 and all(sentence in whole for sentence in sentences)
    assert not any(sentence in line for sentence in sentences) and "2 rooms where a person or I spoke" in line


def test_saved_conversations_keep_their_captured_words_and_unread_arrivals_yield_first():
    snapshot = syn.snapshot(conversations=syn.place_threads())
    words = {word["id"]: word for group in snapshot.conversations for room in group["rooms"] for word in room["words"]}
    order = list(shorter(snapshot, "F6w"))
    # Unread arrivals now yield before untold considered words; tiny rows can
    # stay whole until their final bare-address form if an opening would grow.
    ordinary = [ident for ident in order if not ident.startswith("bare:")]
    assert [words[ident]["rank"] for ident in ordinary[:2]] == [2, 2]
    assert all(ident.removeprefix("bare:") in words for ident in order)
    assert {ident.removeprefix("bare:") for ident in order} == set(words)
    text = mv.render_room(snapshot)  # whole: my account's publications, a bot's labels, a stranger's question
    assert all(word["line"] in text for word in words.values()) and "1 words nobody considered" in text
    lines = [conversations.identity_line(group) for group in snapshot.conversations[1:4]]
    assert all("1 room where a person or I spoke" in line for line in lines)


def _finished_works_room():
    """This room with finished works of every coverage: (a) a one-reply work whose summary is longer than the reply,
    (b) a two-reply work whose work line is shorter than its replies, (c) a three-reply work whose final line is not
    shorter but whose intermediate line is (the capture gives a work one summary, so (c)'s rows carry two here to
    reach that branch), (d) a reply without a summary, (e) the last reply, protected, (f) three short rows of my
    human's and (g) my human's words after the last reply, immune."""
    def reply(i, text, work="", summary="", **facts):
        row = syn.said(i, "ouroboros", text, **facts)
        if work:
            row.update(task_id=work, work_id=work, gist={"summary": summary, "ts": row["ts"], "room_id": "1",
                                                         "address": row["address"]})
        return row

    short = "Replies of a finished work yield like my other replies when no shorter work line covers them."
    lane1 = [syn.said(0, "human", "Look at the floor.", owner_side=True),
             reply(1, syn.prose(400, "mine"), "t201", syn.prose(900, "page")),
             syn.said(2, "human", "And the work lines.", owner_side=True),
             reply(3, syn.prose(1_500, "mine"), "t202", short), reply(4, syn.prose(1_500, "mine"), "t202", short),
             reply(5, syn.prose(1_200, "mine"), "t203", short), reply(6, syn.prose(1_200, "mine"), "t203", short),
             reply(7, syn.prose(400, "mine"), "t203", syn.prose(3_000, "page")),
             syn.said(8, "human", "Thanks, go on.", owner_side=True),
             syn.said(9, "ouroboros", syn.prose(600, "mine")),
             syn.said(10, "ouroboros", syn.prose(600, "mine"), protected_reply=True),
             syn.said(11, "human", "Yes, fold them like the rest.", owner_side=True, immune=True)]
    rows = dict(zip("f a f2 b1 b2 c1 c2 c3 f3 d e g".split(), lane1))
    return syn.snapshot(room_facts=syn.room("1", lane1=lane1)), rows


def _base_forms(monkeypatch):
    """The floor as it stood before such replies took the plain reply forms: no form for a reply with a summary."""
    from ouroboros import memory_forms
    listed = memory_forms.list_elements
    def without(snapshot):
        gist = {r["address"] for r in (snapshot.room or {}).get("lane1", ()) if r.get("gist")}
        return [e for e in listed(snapshot) if not (e[0] == "F2" and e[1].removeprefix("bare:") in gist)]
    monkeypatch.setattr(mf, "_list_elements", without)


def _form(text, row):
    """The form a reply of mine takes in a rendered room: whole, its opening, its address, or inside the range line."""
    for form, shown in (("whole", row["line"]), ("opening", mv._row_opening(row, "my reply")),
                        ("address", mv._row_pointer(row, "my reply"))):
        if shown in text:
            return form
    return "range"


def test_a_gist_reply_no_work_line_covers_takes_the_plain_reply_forms():
    from ouroboros import memory_forms
    snapshot, rows = _finished_works_room()
    listed = memory_forms.list_elements(snapshot)
    f2 = {ident for step, ident, *_ in listed if step == "F2"}
    for name in ("a", "c3", "d"):
        assert {rows[name]["address"], "bare:" + rows[name]["address"]} <= f2, name
        assert memory_forms.stage_of(snapshot, "F2", rows[name]["address"]) == "room_emergency"
        assert memory_forms.stage_of(snapshot, "F2", "bare:" + rows[name]["address"]) == "physical_rows"
    assert {"work:t202:final", "work:t203:progress"} <= f2 and not {"work:t201:final", "work:t203:final"} & f2
    covered = {r["address"] for name, r in rows.items() if name in ("b1", "b2", "c1", "c2")}
    plain = {ident.removeprefix("bare:") for ident in f2 if not ident.startswith("work:")}
    neither = {r["address"] for r in rows.values()} - covered - plain
    assert not covered & plain and not covered & neither and not plain & neither
    assert plain == {rows[n]["address"] for n in ("a", "c3", "d", "e")}


def test_at_the_floor_such_replies_are_inside_the_range_line_and_are_not_counted_as_read(monkeypatch):
    from ouroboros import memory_reading
    from ouroboros.memory_view_work import folded_work_rows
    frame = dict(mode="low", owner_mode="max", window_tokens=200_000, known_window=True, output_reserve=10_000,
                 ratio=1.0, non_memory_tokens=150_000)
    snapshot, rows = _finished_works_room()
    shown = mf.render_view_for_mode(snapshot, **frame, shortest=True)
    text, gone = shown.room, dict(shown.level.addressed)
    assert _form(text, rows["a"]) == "range" and "F7a" in gone
    folded = {i.removeprefix("bare:") for i in gone.get("F2", ())} | folded_work_rows(snapshot.room, gone)[0]
    mine = [r for r in snapshot.room["lane1"] if r["kind"] == "ouroboros" and r["address"] in folded]
    assert rows["a"] in mine and f"({len(mine)} my replies, " in text
    assert rows["a"]["pos"] not in memory_reading.whole_positions(snapshot.room, gone)
    with monkeypatch.context() as base:
        _base_forms(base)
        before, before_rows = _finished_works_room()
        earlier = mf.render_view_for_mode(before, **frame, shortest=True).room
    assert rows["a"]["line"] in earlier  # the base kept it whole at the floor
    def lines(room, row):
        return [x for x in room.splitlines() if row["address"] in x and "earlier rows" not in x]
    for name in ("b1", "b2", "e", "f", "f2", "f3", "g"):
        assert lines(text, rows[name]) == lines(earlier, before_rows[name]), name


def test_such_replies_yield_where_a_plain_own_reply_yields_and_nowhere_else(monkeypatch):
    from ouroboros import context_budget
    wide = dict(window_tokens=1_000_000, known_window=True, output_reserve=10_000, ratio=1.0)
    snapshot, rows = _finished_works_room()
    held = mf.render_view_for_mode(snapshot, mode="max", owner_mode="max", non_memory_tokens=20_000, **wide).room
    with monkeypatch.context() as base:
        _base_forms(base)
        today = mf.render_view_for_mode(_finished_works_room()[0], mode="max", owner_mode="max",
                                        non_memory_tokens=20_000, **wide).room
    assert held == today and rows["a"]["line"] in held
    low = mf.render_view_for_mode(_finished_works_room()[0], mode="low", owner_mode="low",
                                  non_memory_tokens=context_budget.OWNER_LOW_TARGET_TOKENS, **wide).room
    assert rows["a"]["line"] in low  # Low's target spares this room
    nano = mf.render_view_for_mode(_finished_works_room()[0], mode="nano", owner_mode="nano",
                                   non_memory_tokens=context_budget.OWNER_NANO_TARGET_TOKENS, **wide).room
    assert _form(nano, rows["a"]) == _form(nano, rows["d"]) != "whole"
