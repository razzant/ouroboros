"""Which form each kind of memory takes when the floor takes it, on fixtures of real words (``_memory_view_synthetic``'s
work room, story and place threads): today's forms, so a change to them shows as a change of these expectations."""
from ouroboros import memory_floor as mf, memory_view as mv, memory_view_conversations as conversations
from tests import _memory_view_synthetic as syn


def shorter(snapshot, step):
    """``{element id: the form the floor gives it}`` of one step, in its walk order."""
    return {ident: short for each, ident, _whole, short in mf.floor_elements(snapshot) if each == step}


def test_a_reply_yields_to_a_bare_stub_whatever_its_work_and_a_closing_sentence_leaves_with_its_task_line():
    snapshot = syn.snapshot(room_facts=syn.work_room())
    lane1, lane2 = snapshot.room["lane1"], snapshot.room["lane2"]
    replies = [row for row in lane1 if row["kind"] == "ouroboros"]
    stubs = shorter(snapshot, "F2")
    # One element a reply, oldest first: a finished work's intermediate and final replies, a long one without a closing
    # sentence, a running work's, one whose closing sentence another room holds; each stub has only size and address.
    assert list(stubs) == [row["address"] for row in replies]
    assert all(stubs[row["address"]] == f"{row['head']} (my reply, {row['chars']} chars — read by address)"
               for row in replies)
    answered = lane1[-1]  # my human's unanswered words stay whole; the reply they answer yields like any other
    assert answered["reply_to"] in stubs and answered["address"] not in shorter(snapshot, "F7")
    level = mv.FloorLevel((("F1", tuple(item["first"] for item in lane2)), ("F2", tuple(stubs))))
    text = mv.render_room(snapshot, level)
    assert "summary: «" in mv.render_room(snapshot) and "summary: «" not in text and "3 earlier tasks" in text
    assert answered["line"] in text and not any(row["line"] in text for row in replies)


def test_a_page_yields_its_first_sentence_whatever_its_lead_and_a_part_or_a_page_without_one_only_its_address():
    story = syn.told_story()
    snapshot = syn.snapshot(story=story)
    old, part, pages = story[0], story[1], story[2:]
    labels = shorter(snapshot, "F5l")
    # Asked to lead or not, written by the old code between two of the new or late by an old focus: one form.
    assert list(labels) == [page["id"] for page in pages if "\n" not in page["text"]]
    assert {page["id"] for page in pages if page.get("lead")} < set(labels) and part["id"] not in labels
    decided = pages[1]  # the decision after its topical first sentence leaves the view with the rest of the page
    assert labels[decided["id"]].endswith("\n  Nova memory audit, November.")
    assert "Decision" in decided["text"] and "Decision" not in labels[decided["id"]]
    assert set(shorter(snapshot, "F5")) == {entry["id"] for entry in story[1:]}
    assert list(shorter(snapshot, "F3")) == [old["room_id"]] and old["text"] in mv.render_story(snapshot)


def test_a_conversation_line_counts_its_threads_and_drops_every_closing_sentence():
    both = syn.place_threads()[0]
    sentences = [task["line"].split("«")[1].split("»")[0] for room in both["rooms"] for task in room["tasks"]]
    whole, line = (conversations.render_group(both, set(), set(), as_line) for as_line in (False, True))
    assert len(sentences) == 2 and all(sentence in whole for sentence in sentences)
    assert not any(sentence in line for sentence in sentences) and "2 rooms where my human or I spoke" in line


def test_my_own_accounts_arrivals_and_a_bots_words_stand_as_a_conversation_and_a_strangers_reply_folds_last():
    snapshot = syn.snapshot(conversations=syn.place_threads())
    words = {word["id"]: word for group in snapshot.conversations for room in group["rooms"] for word in room["words"]}
    order = list(shorter(snapshot, "F6w"))
    assert sorted(order) == sorted(words) and [words[ident]["rank"] for ident in order] == [1] * 6 + [2] * 2
    assert [words[ident]["head"].split("; ")[1] for ident in order[-2:]] == ["github-actions[bot]", "Sasha Petrova"]
    text = mv.render_room(snapshot)  # whole: my account's publications, a bot's labels, a stranger's question
    assert all(word["line"] in text for word in words.values()) and "1 words nobody considered" in text
    lines = [conversations.identity_line(group) for group in snapshot.conversations[1:4]]
    assert all("1 room where my human or I spoke" in line for line in lines)
