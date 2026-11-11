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
