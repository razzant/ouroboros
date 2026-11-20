"""The message my human replied to travels beside their words through every transfer of them: a drafted start, the
terminal evidence, Continue, mail into running work and a relay. It stays typed, never counts as my human's words and
owes no reply; when its words cannot be read its address stays, and a place connector's platform id says honestly
that no host address reads it. One message is one corpus row by its identity, and an empty id names none.

Built on the review's reproductions (RV1W Astra F1, Grok F1, Opus F7).
"""
from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store
from ouroboros.chat_chain import format_address, row_address
from ouroboros.loop_messages import _initialize_owner_directives, _record_owner_directive, owner_input_revision
from ouroboros.owner_words import FIELD, REPLY_TARGET, SCHEDULE_TEXT, governing_words_for_schedule

PROPOSAL = "The exact proposed specification, step by step."


@pytest.mark.parametrize("legacy", ["old record", ["old record"], 42])
def test_legacy_reply_reference_shape_does_not_break_context(legacy):
    from ouroboros.context import _owner_message_facts

    assert "reply_to" not in _owner_message_facts({"reply_reference": legacy})
    reference = {"address": "row:1@example", "excerpt": {"text": "Earlier words", "start_utf16": 0}}
    assert _owner_message_facts({"reply_reference": reference})["reply_to"] == reference


def _proposal(root, text=PROPOSAL):
    dialogue_store.append(root, "chat", {"ts": "2026-11-04T09:00:00+00:00", "direction": "out", "chat_id": 1,
                                         "text": text, "client_message_id": "proposal"})
    stored = next(entry.row for entry in dialogue_store.iter_probes(root, "chat", [("client_message_id", "proposal")]))
    return {"address": format_address(row_address(stored))}


def _door_run(tmp_path):
    from tests.test_owner_words_carried import _owner_root

    ctx = _owner_root(tmp_path, {"origin_message_ref": {"chat_id": 1, "client_message_id": "replying-owner"},
                                 "origin_message_text": "Do this."})
    ctx.task_metadata["reply_reference"] = _proposal(ctx.drive_root)
    _initialize_owner_directives(ctx, [{"role": "user", "content": "Do this."}])
    return ctx


def _targets(ctx):
    return [row["text"] for row in governing_words_for_schedule(ctx)[FIELD] if row["source"] == REPLY_TARGET]


@pytest.mark.parametrize("routed", [False, True])
def test_a_reply_target_survives_a_drafted_start_into_its_helper(tmp_path, routed):
    from ouroboros.tools.control_routing import _attach_drafted_objective, _attach_origin_from_metadata

    ctx = _door_run(tmp_path)
    assert ctx._owner_directives[0]["reply_to"]["text"] == PROPOSAL
    if routed:  # start_task from this turn: the door's row becomes the start's origin message
        event = {}
        _attach_origin_from_metadata(ctx, event)
        _attach_drafted_objective(ctx, event)
        [origin] = [row for row in event["owner_corpus"] if row["source"] == "origin_message"]
        assert origin["client_message_id"] == "replying-owner" and origin["reply_to"]["text"] == PROPOSAL
        ctx = SimpleNamespace(task_id="new-root", task_metadata=event)
        _initialize_owner_directives(ctx, [{"role": "user", "content": "Implement the plan."}])
        assert owner_input_revision(ctx) == 0, "carried words owe no reply"
    assert _targets(ctx) == [PROPOSAL], "the helper reads the message my human replied to, under its author"


def test_a_reply_target_rides_the_requirements_into_a_continue_successor(tmp_path):
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources
    from ouroboros.review_evidence_sections import _accept_owner_directives
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result

    ctx = _door_run(tmp_path)
    later = {"address": "row:1@2026-11-04T09:30:00+00:00#0123456789ab", "author": "Sasha", "text": "Sasha's draft."}
    _record_owner_directive(ctx, source="owner_mailbox", content="Keep the old API too.", msg_id="mail-1",
                            origin={"client_message_id": "owner-2"}, reply_to=later)
    requirements = _accept_owner_directives(ctx, None, "root-1")
    assert [row.get("reply_to", {}).get("text") for row in requirements] == [PROPOSAL, "Sasha's draft."]
    assert requirements[1]["client_message_id"] == "owner-2"
    root = ctx.drive_root
    write_task_result(root, "root-1", STATUS_CANCELLED, chat_id=1, root_task_id="root-1", result="interrupted",
                      origin_message_text="Do this.", origin_message_ref={"chat_id": 1, "client_message_id": "replying-owner"},
                      review_evidence={"task_inputs": {"owner_requirements_and_decisions": requirements}},
                      owner_mailbox={"rows": [], "read_complete": True})
    sources = owner_sources(root, load_task_result(root, "root-1"), "root-1")
    successor = SimpleNamespace(task_id="successor", task_metadata={
        "objective_author": {"kind": "continuation", "predecessor_task_id": "root-1"},
        "owner_corpus": owner_corpus_rows(sources)})
    _initialize_owner_directives(successor, [{"role": "user", "content": "[CONTINUE] work order"}])
    assert _targets(successor) == [PROPOSAL, "Sasha's draft."]
    assert [row.get("client_message_id") for row in successor._owner_directives] == ["replying-owner", "owner-2"]


@pytest.mark.parametrize("source", ["initial_user", "owner_mailbox"])
def test_a_scheduled_run_continues_with_the_words_and_reply_target_it_carried(tmp_path, source):
    """RV2W Astra R1: a follow-up root has no door of its own, so its carried door rows are my human's words, not
    its first text: the real registration, fired run, terminal evidence and Continue seed keep them and their target."""
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources, work_order_text
    from ouroboros.review_evidence_sections import _accept_owner_directives
    from ouroboros.tools.followup import _handle_schedule_followup
    from supervisor import queue as schedules
    from tests.test_schedule_continue_basis import receipt

    ctx = _door_run(tmp_path)
    if source == "owner_mailbox":
        reply = ctx._owner_directives[0]["reply_to"]
        ctx._owner_directives = []
        _record_owner_directive(ctx, source=source, content="Do this.", msg_id="mail-1", reply_to=reply)
    assert _handle_schedule_followup(ctx, relation="independent", objective="Check the plan weekly.",
                                     cron="0 9 * * 1").startswith("FOLLOWUP_SCHEDULED")
    [registered] = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
    fired = schedules._task_from_schedule(registered)
    live = SimpleNamespace(task_id="root", task_metadata=fired["metadata"])
    _initialize_owner_directives(live, [{"role": "user", "content": fired["text"]}])
    assert _targets(live) == [PROPOSAL] and owner_input_revision(live) == 0
    record = receipt()  # the scheduler's accepted receipt shape, holding the generated task and its metadata
    scheduled = record["schedule_admission"]["task"]
    scheduled.update(text=fired["text"])
    scheduled["metadata"].update(fired["metadata"])
    record.update(metadata=scheduled["metadata"], owner_mailbox={"rows": [], "read_complete": True},
                  review_evidence={"task_inputs": {"owner_requirements_and_decisions":
                                                  _accept_owner_directives(live, None, "root")}})
    sources = owner_sources(ctx.drive_root, record, "root")
    assert sources["gaps"] == [] and [row["content"] for row in sources["later"]] == ["Do this."]
    assert "Do this." in work_order_text("root", "interrupted_retry_lost", sources)
    successor = SimpleNamespace(task_id="successor", task_metadata={
        "objective_author": {"kind": "continuation", "predecessor_task_id": "root"},
        "owner_corpus": owner_corpus_rows(sources)})
    _initialize_owner_directives(successor, [{"role": "user", "content": "[CONTINUE] work order"}])
    assert _targets(successor) == [PROPOSAL], "the successor reads the message my human replied to"
    assert [row["content"] for row in successor._owner_directives if row["source"] != SCHEDULE_TEXT] == ["Do this."]
    # Beside it the schedule's own text rides as the assignment, a separate source, never my human's words.
    assert [row["content"] for row in successor._owner_directives
            if row["source"] == SCHEDULE_TEXT] == ["Check the plan weekly."]


@pytest.mark.parametrize("select", [False, True])
def test_mail_into_running_work_keeps_the_message_it_replies_to(tmp_path, select):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.projects_registry import create_project
    from ouroboros.server_owner_routing import _route_project_chat_to_running_task
    from tests.test_project_routing_v664 import _ctx

    chat_id = int(create_project(tmp_path, "reply-room")["chat_id"])
    task = {"id": "work-root", "chat_id": chat_id, "root_task_id": "work-root", "delegation_role": "root",
            "drive_root": str(tmp_path)}  # its mailbox lives on this drive
    from ouroboros.chat_references import ingress_reply
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, OWNER_WORDS_KIND

    passage = "Blue lanterns bloom beside the observatory. " * 5 + "END"
    source = f"Map: **{passage}**\nNight route." if select else PROPOSAL
    reference = _proposal(tmp_path, source)
    if select:
        reference = ingress_reply(tmp_path, {"reply_to": reference["address"],
                                             "reply_excerpt": {"text": passage, "start_utf16": 5}})
    routed = _route_project_chat_to_running_task(_ctx(tmp_path, running={"work-root": {"task": task}}), chat_id,
                                                 "Yes, but skip step two.", "owner-reply",
                                                 task_metadata={"reply_reference": reference})
    assert routed == "work-root"
    assert _mailbox_path(tmp_path, "work-root").read_text(encoding="utf-8").count(reference["address"]) == 1
    reader = SimpleNamespace(task_id="work-root", drive_root=tmp_path, budget_drive_root=tmp_path, current_chat_id=chat_id,
                             task_metadata={}, pending_events=[], event_queue=None, drive_logs=lambda: tmp_path / "logs")
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "work-root", None, set(), owner_ctx=reader)
    [row] = reader._owner_directives
    assert row["content"].startswith("Yes, but skip step two.") and row["reply_to"]["text"] == source
    if select:
        [citation] = [m["content"] for m in messages if m.get(HOST_CONTEXT_KIND_KEY) == "reply_citation"]
        [owner] = [m["content"] for m in messages if m.get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND]
        assert passage in citation and source in citation and "client-reported" in citation
        assert reference["address"] in citation and "Ouroboros" in citation and "UTF-16 offset 5" in citation
        assert "Yes, but skip step two." in owner and passage not in owner
        assert row["reply_to"]["excerpt"]["text"] == passage
    assert owner_input_revision(reader) == 1, "my human's own words still owe their reply; the target owes none"


def test_a_relayed_quote_keeps_the_message_it_replies_to(tmp_path):
    from ouroboros.owner_relay import inject_relay
    from tests.test_owner_relay import context

    reference = _proposal(tmp_path)
    quoted = {"chat_id": 1, "direction": "in", "ts": "2026-11-04T10:00:00Z", "text": "Do it as proposed.",
              "client_message_id": "owner-quoted", "reply_to": reference}
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    inject_relay(receiver, {"owner_rows": [quoted], "source_task_id": "teller", "text": "note",
                            "relay": {"relay_id": "r1"}}, [])
    [row] = receiver._owner_directives
    assert row["carried"] is True and row["reply_to"]["text"] == PROPOSAL


def test_native_steering_separates_sender_selection_from_host_reference(tmp_path):
    import json
    from ouroboros.delegate_interactions import _placement_text
    from ouroboros.dialogue_provenance import excerpt_citation

    full = PROPOSAL + " Keep the remaining notes. " * 12
    target = {**_proposal(tmp_path, full), "author_label": "Ouroboros", "snippet": PROPOSAL}
    reference = {**target, "excerpt": {"text": "Sender's rendered words", "start_utf16": 4}}
    quote = {"source": {"chat_id": 1, "ts": "2026-11-04T12:00:00Z"},
             "owner_row": {"text": "Explain the observation.", "reply_to": reference, "_owner_source_root": tmp_path}}
    frame, marker = _placement_text(quote, "owner-handle", "child")
    header, host, citation = frame.splitlines()[-3:]
    assert header.startswith("[Host reference:") and json.loads(host) == {**target, "text": full}
    assert citation == excerpt_citation(reference)
    assert "Explain the observation." in frame and marker["address"] == "owner-handle"


def test_inherited_safety_evidence_keeps_each_replies_selection(tmp_path):
    from ouroboros.dialogue_provenance import excerpt_citation
    from tests.test_owner_words_carried import _owner_root
    from tests.test_safety_focus_bundle import _sources

    parent = _owner_root(tmp_path)
    source = {**_proposal(parent.drive_root), "author": "Ouroboros", "text": PROPOSAL}
    expected = {}
    for index, word in enumerate(("exact", "step")):
        reply = {**source, "excerpt": {"text": word, "start_utf16": PROPOSAL.index(word)}}
        question = f"Explain observation {index}."
        _record_owner_directive(parent, source="owner_mailbox", msg_id=f"msg-{index}", content=question, reply_to=reply)
        expected[question] = excerpt_citation(reply)
    child = _owner_root(tmp_path, governing_words_for_schedule(parent))
    child._owner_directives = []
    rows = _sources(child)["occasion"]["sources"]
    assert len(rows) == 2
    assert all(row["reply_to"] == source for row in rows)
    assert {row["content"]: row["reply_citation"] for row in rows} == expected


@pytest.mark.parametrize("selected_first", [False, True])
@pytest.mark.parametrize("source_read", ["readable", "unavailable", "unresolved"])
@pytest.mark.parametrize("source_kind", ["project", "media"])
def test_whole_reply_and_selected_passage_keep_their_own_source_in_a_helper(tmp_path, selected_first, source_read, source_kind):
    from ouroboros.chat_references import ingress_reply
    from ouroboros.dialogue_provenance import excerpt_citation
    from ouroboros.owner_relay import _rows_to_carry
    from ouroboros.owner_words import UNREAD_TARGET, render_owner_words, reply_target, safety_sources
    from tests.test_owner_words_carried import _owner_root

    parent = _owner_root(tmp_path)
    root = parent.drive_root
    answer = "Map: **North trail**. Return at dawn."
    if source_kind == "project":
        source = {"direction": "system", "type": "project_completion_summary", "text": "Project completed",
                  "completion_answer": answer}
    else:
        # In this branch the author label is the same for both views; the text is not.
        source = {"direction": "out", "type": "photo", "text": "[Photo]", "caption": answer}
    saved = dialogue_store.append(root, "chat", {**source, "chat_id": 1, "ts": "2026-11-04T09:00:00+00:00"}).row
    address = format_address(row_address(saved))
    excerpt = {"text": "North trail", "start_utf16": 5}
    selected = ingress_reply(root, {"reply_to": address, "reply_excerpt": excerpt})
    if source_read == "unresolved":
        selected = {**selected, "unresolved": True}
    whole = reply_target(root, {"address": address})
    selection = reply_target(root if source_read == "readable" else None, selected)
    messages = [("About the whole notice.", whole), ("About the north trail.", selection)]
    if selected_first:
        messages.reverse()
    for i, (message, target) in enumerate(messages):
        _record_owner_directive(parent, source="owner_mailbox", msg_id=f"owner-{i}", content=message, reply_to=target)

    child = _owner_root(tmp_path, governing_words_for_schedule(parent))
    rows = governing_words_for_schedule(child)[FIELD]  # by-value round trip, without consulting history for the quote
    rendered = render_owner_words(rows, audience="child")
    author = "" if source_read == "unresolved" else "Ouroboros"
    body = answer if source_read == "readable" else ""
    assert (f"[{whole['author']}: the message my human replied to, not my human's words · {address}]"
            f"\n{source['text']}") in rendered
    assert (f"[{author or 'author not recorded'}: the message my human replied to, not my human's words · {address}]"
            f"\n{body or UNREAD_TARGET}") in rendered
    assert excerpt_citation(selection) in rendered

    # Both the structured input projection and the live carriage reader must keep the right form.
    by_words = {row["content"]: row for row in safety_sources(child)}
    assert by_words["About the whole notice."]["reply_to"] == whole
    picked = by_words["About the north trail."]
    assert picked["reply_to"] == {"address": address, "author": author, "text": body}
    assert picked["reply_citation"] == excerpt_citation(selection)
    carried, = [quote["owner_row"]["reply_to"] for _, _, quote, _, _ in _rows_to_carry(child, root)
                if quote["text"] == "About the north trail."]
    assert (carried["address"], carried["author"], carried["text"], carried["excerpt"]) == (address, author, body, excerpt)
    assert carried.get("role") == (None if source_read == "unresolved" else "assistant"), "the host's role rides on"


def test_a_place_reply_names_honestly_what_the_host_cannot_read():
    from ouroboros.owner_words import UNREAD_PLATFORM, UNREAD_TARGET, _row, render_owner_words

    place = _row("", source=REPLY_TARGET, carrier="ctx", ref="platform message 42 in this room", author="")
    chat = _row("", source=REPLY_TARGET, carrier="ctx", ref="row:1@2026-11-04T09:00:00+00:00#0123456789ab", author="")
    text = render_owner_words([place, chat], audience="child")
    assert UNREAD_PLATFORM in text and text.count(UNREAD_TARGET) == 1, "only a host address claims to read the words"
    assert "author not recorded: the message my human replied to, not my human's words" in text


@pytest.mark.parametrize("variant,readable_first", [
    ("match", False), ("match", True), ("other_offset", False), ("other_author", False),
    ("other_text", False), ("other_address", False), ("whole_notice", False),
])
def test_two_carriers_of_one_reply_keep_the_full_source_of_the_same_selection(variant, readable_first):
    from ouroboros.owner_words import REPLY_TARGET, _deduped, _words_rows

    # The saved offsets are in the rendered text, whereas the by-value original includes Markdown.
    address = "row:1@2026-11-04T09:00:00+00:00#0123456789ab"
    picked = {"text": "oak", "start_utf16": 7}  # second oak in '🌲 oak oak'
    target = {"address": address, "author": "Ouroboros", "excerpt": picked}
    body = "🌲 **oak** [oak](https://example.org)"
    other = {**target, "text": body}
    if variant == "other_offset":
        other["excerpt"] = {**picked, "start_utf16": 3}
    elif variant == "other_author":
        other["author"] = "Another author"
    elif variant == "other_text":
        other["excerpt"] = {**picked, "text": "oak tree"}
    elif variant == "other_address":
        other["address"] = address.replace("0123456789ab", "1123456789ab")
    elif variant == "whole_notice":
        other.pop("excerpt")

    def carrier(reply_to, name):
        return _words_rows({"source": "owner_mailbox", "content": "Which oak?", "msg_id": "question", "reply_to": reply_to},
                           carrier=name, task_id="work")

    unread, readable = carrier(target, "inherited"), carrier(other, "held")
    rows = _deduped([*readable, *unread] if readable_first else [*unread, *readable])
    assert (rows[0]["text"], rows[0]["carrier"]) == ("Which oak?", "held" if readable_first else "inherited")
    citation = rows[0]["reply_citation"]
    assert citation["excerpt"] == picked
    targets = [r for r in rows if r["source"] == REPLY_TARGET and r.get("reply_view") == citation["reply_view"]]
    assert [r["text"] for r in targets] == ([body] if variant == "match" else [""])


@pytest.mark.parametrize("readable_first", [False, True])
@pytest.mark.parametrize("author", [{"direction": "in", "source": "web", "user_id": 1}, {"direction": "out", "task_id": "abc123"}],
                         ids=["owner", "ouroboros"])
def test_two_readers_labels_for_one_author_keep_the_selected_source(tmp_path, author, readable_first):
    """The door labels the replied row's author for the page (You, Ouroboros); a reader of the row signs it for the
    record (User [source=web], Ouroboros (task …)). Carriers of one selected reply that differ only so are one
    selection, and its readable source stays; another actor's, passage's or message's record is never merged in."""
    from ouroboros.chat_references import ingress_reply
    from ouroboros.owner_words import _deduped, _words_rows, reply_target, task_governing_words

    words = "Meet at the north gate."
    saved = dialogue_store.append(tmp_path, "chat", {"chat_id": 1, "ts": "2026-11-04T09:00:00+00:00", "text": words,
                                                     **author}).row
    address = format_address(row_address(saved))
    reference = ingress_reply(tmp_path, {"reply_to": address, "reply_excerpt": {"text": "north gate", "start_utf16": 12}})
    unread, readable = reply_target(None, reference), reply_target(tmp_path, reference)
    assert unread["author"] != readable["author"] and unread["role"] == readable["role"] == reference["role"]
    assert (unread["text"], readable["text"]) == ("", words)

    def carrier(target, name):
        return _words_rows({"source": "owner_mailbox", "content": "Which gate?", "msg_id": "question", "reply_to": target},
                           carrier=name, task_id="work")

    def merged(held):
        inherited = task_governing_words({FIELD: carrier(unread, "inherited")})[0]  # by value, as a helper inherits it
        held = carrier(held, "held")
        rows = _deduped([*held, *inherited] if readable_first else [*inherited, *held])
        return rows[0]["reply_citation"], [(row["author"], row["text"]) for row in rows if row["source"] == REPLY_TARGET]

    citation, targets = merged(readable)
    assert (citation["address"], citation["excerpt"], citation["reply_view"]) == (address, reference["excerpt"], "authored")
    assert targets == [(readable["author"], words)]
    other_role = "assistant" if readable["role"] == "user" else "user"
    for different in ({"role": other_role}, {"excerpt": {"text": "north gate", "start_utf16": 13}},
                      {"address": address[:-1] + ("0" if address[-1] != "0" else "1")}):
        if not readable_first:  # the first carrier's own record stands either way; it adopts no other's source
            citation, targets = merged({**readable, **different})
            assert citation["reply_view"] == "unread" and targets == [(unread["author"], "")], different


@pytest.mark.parametrize("author", ["Ouroboros", ""])
def test_held_selection_passes_its_recorded_author_to_the_next_helper(tmp_path, monkeypatch, author):
    from ouroboros.owner_words import render_owner_words
    from tests.test_owner_words_carried import _owner_root

    parent = _owner_root(tmp_path)
    address = "row:1@2026-11-04T09:00:00+00:00#0123456789ab"
    reply = {"address": address, "excerpt": {"text": "boat", "start_utf16": 4},
             **({"author_label": author} if author else {"unresolved": True})}
    parent._held_owner_messages = {"held": {"chat_id": 1, "direction": "in", "source": "web", "user_id": 1,
        "ts": "2026-11-04T09:01:00+00:00", "client_message_id": "your-reply", "text": "Explain this.", "reply_to": reply}}
    monkeypatch.setattr("ouroboros.chat_chain.resolve_row", lambda *_: pytest.fail("the target was to be carried by value"))
    child = _owner_root(tmp_path, governing_words_for_schedule(parent))
    grandchild = governing_words_for_schedule(child)[FIELD]
    text = render_owner_words(grandchild, audience="child")
    assert f"Sender-selected passage of {author or 'author not recorded'} at {address}" in text
    assert [row["text"] for row in grandchild if row["source"] == REPLY_TARGET] == [""]


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_message_id_never_merges_two_messages(empty):
    ctx = SimpleNamespace(task_id="run")
    for index in (1, 2):
        _record_owner_directive(ctx, source="owner_mailbox", content=f"message {index}", msg_id=empty or "",
                                origin={"client_message_id": empty})
    _record_owner_directive(ctx, source="owner_mailbox", content="the same message again", msg_id="",
                            origin={"client_message_id": "c-3"})
    _record_owner_directive(ctx, source="owner_mailbox", content="the same message again", msg_id="",
                            origin={"client_message_id": "c-3"})
    assert [row["content"] for row in ctx._owner_directives] == ["message 1", "message 2", "the same message again"]
