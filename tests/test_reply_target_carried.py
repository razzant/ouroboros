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


def test_a_run_started_by_a_selected_reply_hands_its_helper_the_captured_source_while_history_is_unreadable(
        tmp_path, monkeypatch):
    """The run the owner's selected Reply started holds the delivery's reference, which carries the source captured at
    acceptance: its helper reads that source's exact author, whole text and files by value, and the compact reference
    a replay holds gives it the address alone."""
    from ouroboros.chat_references import ingress_reply, stored_reply
    from ouroboros.owner_words import UNREAD_TARGET, render_owner_words
    from ouroboros.store import StoreError

    ctx = _door_run(tmp_path)
    full = "Map: **North trail**. Return at dawn. " * 12
    saved = dialogue_store.append(ctx.drive_root, "chat", {"ts": "2026-11-04T09:00:30+00:00", "direction": "out", "chat_id": 1,
                                                          "text": full, "client_message_id": "trail-map"}).row
    address = format_address(row_address(saved))
    delivered = ingress_reply(ctx.drive_root, {"reply_to": address, "reply_excerpt": {"text": "North trail", "start_utf16": 11}})
    assert delivered["text"] == full and "text" not in stored_reply(delivered)

    def unavailable(*_args, **_kwargs):
        raise StoreError("history unavailable")

    monkeypatch.setattr("ouroboros.chat_chain.resolve_row", unavailable)
    for reference, expected in ((delivered, full), (stored_reply(delivered), "")):
        ctx.task_metadata["reply_reference"] = reference
        ctx._owner_directives = []
        _initialize_owner_directives(ctx, [{"role": "user", "content": "Do this."}])
        [target] = [row for row in governing_words_for_schedule(ctx)[FIELD] if row["source"] == REPLY_TARGET]
        assert target["author"] == "Ouroboros" and target["text"] == expected
        rendered = render_owner_words(governing_words_for_schedule(ctx)[FIELD], audience="child")
        assert (full in rendered) is bool(expected) and (UNREAD_TARGET in rendered) is not bool(expected)


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
        full, citation = [m["content"] for m in messages if m.get(HOST_CONTEXT_KIND_KEY) == "reply_citation"]
        [owner] = [m["content"] for m in messages if m.get(HOST_CONTEXT_KIND_KEY) == OWNER_WORDS_KIND]
        assert source in full and reference["address"] in full and "Ouroboros" in full, "the whole source, then the selection"
        assert passage in citation and "client-reported" in citation
        assert reference["address"] in citation and "UTF-16 offset 5" in citation
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


@pytest.mark.parametrize("history", ["absent", "unreadable"])
def test_a_relayed_selection_keeps_the_source_it_carries_when_history_cannot_be_read(tmp_path, monkeypatch, history):
    """A relay brings the selected message's words and files by value. The receiver reads history again only for
    what the relay did not bring: an unreadable row never empties what arrived with the quote."""
    from ouroboros.owner_relay import inject_relay
    from ouroboros.owner_words import reply_target
    from ouroboros.store import StoreError
    from tests.test_owner_relay import context

    full = "Map: **North trail**. Return at dawn."
    files = [{"name": "map.pdf", "mime": "application/pdf", "size": 55}]
    address = "row:1@2026-11-04T09:00:00+00:00#0123456789ab"  # the store holds no row of that address
    held = {"address": address, "author": "Ouroboros", "author_label": "Ouroboros", "role": "assistant", "text": full,
            "attachments": files, "excerpt": {"text": "North trail", "start_utf16": 5}}
    if history == "unreadable":
        def unreadable(*_args, **_kwargs):
            raise StoreError("history unavailable")

        monkeypatch.setattr("ouroboros.chat_chain.resolve_row", unreadable)
    for root in (tmp_path, None):
        target = reply_target(root, held)
        assert (target["text"], target["attachments"], target["author"], target["excerpt"]) == (
            full, files, "Ouroboros", held["excerpt"])
    whole = {key: value for key, value in held.items() if key != "excerpt"}
    assert reply_target(tmp_path, whole)["text"] == "", "only a selection carries its source by value"

    quoted = {"chat_id": 1, "direction": "in", "ts": "2026-11-04T10:00:00Z", "text": "Which trail?",
              "client_message_id": "owner-quoted", "reply_to": held}
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    frames = []
    inject_relay(receiver, {"owner_rows": [quoted], "source_task_id": "teller", "text": "note",
                            "relay": {"relay_id": "r1"}}, frames)
    [row] = receiver._owner_directives
    assert row["content"] == "Which trail?" and full not in row["content"], "the quoted source is never a new command"
    assert (row["reply_to"]["text"], row["reply_to"]["attachments"]) == (full, files)
    frame = frames[0]["content"]
    assert full in frame and "map.pdf" in frame and f"Sender-selected passage of Ouroboros at {address}" in frame


def test_selected_replies_to_one_source_keep_each_citation_and_hold_the_source_once(tmp_path):
    from ouroboros.chat_references import ingress_reply
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY, REPLY_CITATION_KIND
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.projects_registry import create_project
    from ouroboros.server_owner_routing import _route_project_chat_to_running_task
    from tests.test_project_routing_v664 import _ctx

    chat_id = int(create_project(tmp_path, "citation-room")["chat_id"])
    task = {"id": "work-root", "chat_id": chat_id, "root_task_id": "work-root", "delegation_role": "root",
            "drive_root": str(tmp_path)}
    source = "Blue lanterns bloom beside the observatory. Return at dawn. END"
    address = _proposal(tmp_path, source)["address"]
    picks = [{"text": "Blue lanterns", "start_utf16": 0}, {"text": "dawn", "start_utf16": source.index("dawn")},
             {"text": "END", "start_utf16": source.index("END")}]
    for index, pick in enumerate(picks):
        reference = ingress_reply(tmp_path, {"reply_to": address, "reply_excerpt": pick})
        assert _route_project_chat_to_running_task(_ctx(tmp_path, running={"work-root": {"task": task}}), chat_id,
                                                   f"Question {index}.", f"owner-reply-{index}",
                                                   task_metadata={"reply_reference": reference}) == "work-root"
    reader = SimpleNamespace(task_id="work-root", drive_root=tmp_path, budget_drive_root=tmp_path, current_chat_id=chat_id,
                             task_metadata={}, pending_events=[], event_queue=None, drive_logs=lambda: tmp_path / "logs")
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), tmp_path, "work-root", None, set(), owner_ctx=reader)
    cited = [m for m in messages if m.get(HOST_CONTEXT_KIND_KEY) == REPLY_CITATION_KIND]
    bound = [m for m in cited if HOST_TEXT_SHA_KEY in m]
    frames = [m["content"] for m in cited if HOST_TEXT_SHA_KEY not in m]
    assert len(frames) == 3, "every selected passage keeps its own citation"
    for pick, frame in zip(picks, frames):
        assert f'"{pick["text"]}"' in frame and f"UTF-16 offset {pick['start_utf16']}" in frame
        assert address in frame, "each names the source it selects from"
    [full] = [m["content"] for m in bound]
    assert source in full and "Ouroboros" in full and address in full, "the source's words stand once, under its author"
    assert cited[0] is bound[0], "the source leads the first selection"


def test_a_source_known_with_more_is_shown_whole_again_never_hidden_behind_an_earlier_reading():
    """One address, several readings: only a host row bound to the very same text (address, author, words, files)
    stands in for the source. A file known later, a changed row and a look-alike no host bound each show it again."""
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY, REPLY_CITATION_KIND
    from ouroboros.loop_round_limits import _append_reply_context

    address = "row:1@2026-11-04T09:00:00+00:00#0123456789ab"
    files = [{"name": "map.pdf", "mime": "application/pdf", "size": 55}]
    first = {"address": address, "author": "Ouroboros", "text": "Map: North trail."}

    def pick(word, **more):
        return {**first, **more, "excerpt": {"text": word, "start_utf16": first["text"].index(word)}}

    def sources(rows):
        return [row for row in rows if HOST_TEXT_SHA_KEY in row]

    messages = []
    for target in (pick("Map"), pick("North")):
        _append_reply_context(messages, target, None)
    assert len(sources(messages)) == 1 and len(messages) == 3, "the same reading is held once, each selection kept"
    assert messages[0][HOST_CONTEXT_KIND_KEY] == REPLY_CITATION_KIND and "map.pdf" not in messages[0]["content"]

    _append_reply_context(messages, pick("trail", attachments=files), None)  # its files were unknown at the first reading
    assert [("map.pdf" in row["content"]) for row in sources(messages)] == [False, True], "the file is not lost"
    _append_reply_context(messages, pick("Map", attachments=files), None)
    assert len(sources(messages)) == 2, "and now held in full, once"

    messages[0]["content"] += " (trimmed)"  # a row changed since the host produced it proves nothing
    _append_reply_context(messages, pick("North"), None)
    assert len(sources(messages)) == 3 and sources(messages)[-1]["content"] == sources(messages)[0]["content"][:-len(" (trimmed)")]

    lookalike = [{"role": "user", HOST_CONTEXT_KIND_KEY: REPLY_CITATION_KIND, "content": sources(messages)[-1]["content"]}]
    _append_reply_context(lookalike, pick("North"), None)  # the same words, but no host binding
    assert len(sources(lookalike)) == 1 and lookalike[0] is not sources(lookalike)[0], "text alone is no identity"


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
@pytest.mark.parametrize("source_read", ["readable", "captured", "unavailable", "unresolved"])
@pytest.mark.parametrize("source_kind", ["project", "media"])
def test_whole_reply_and_selected_passage_keep_their_own_source_in_a_helper(tmp_path, selected_first, source_read, source_kind):
    from ouroboros.chat_references import ingress_reply, stored_reply
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
    selected = ingress_reply(root, {"reply_to": address, "reply_excerpt": excerpt})  # the delivery's: it carries its source
    if source_read in ("unavailable", "unresolved"):  # what an older row or a replay holds: the compact reference alone
        selected = stored_reply(selected)
    if source_read == "unresolved":
        selected = {**selected, "unresolved": True}
    whole = reply_target(root, {"address": address})
    selection = reply_target(root if source_read == "readable" else None, selected)  # "captured": history is not read
    messages = [("About the whole notice.", whole), ("About the north trail.", selection)]
    if selected_first:
        messages.reverse()
    for i, (message, target) in enumerate(messages):
        _record_owner_directive(parent, source="owner_mailbox", msg_id=f"owner-{i}", content=message, reply_to=target)

    child = _owner_root(tmp_path, governing_words_for_schedule(parent))
    rows = governing_words_for_schedule(child)[FIELD]  # by-value round trip, without consulting history for the quote
    rendered = render_owner_words(rows, audience="child")
    author = "" if source_read == "unresolved" else "Ouroboros"
    body = answer if source_read in ("readable", "captured") else ""
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
    from ouroboros.chat_references import ingress_reply, stored_reply
    from ouroboros.owner_words import _deduped, _words_rows, reply_target, task_governing_words

    words = "Meet at the north gate."
    saved = dialogue_store.append(tmp_path, "chat", {"chat_id": 1, "ts": "2026-11-04T09:00:00+00:00", "text": words,
                                                     **author}).row
    address = format_address(row_address(saved))
    # The compact reference: what the saved row, history and a replay hold (the delivery's own carries the source whole).
    reference = stored_reply(ingress_reply(tmp_path, {"reply_to": address,
                                                      "reply_excerpt": {"text": "north gate", "start_utf16": 12}}))
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


@pytest.mark.parametrize("carried", [False, True], ids=["relay", "carriage"])
@pytest.mark.parametrize("explicit", [False, True], ids=["carry-rows", "quote-rows"])
def test_held_owner_row_carries_the_captured_source_without_rewriting_its_address(tmp_path, monkeypatch, carried, explicit):
    from ouroboros.owner_relay import _rows_to_carry, inject_relay, owner_row_address, owner_source_row, quote_rows
    from ouroboros.owner_words import reply_target
    from ouroboros.store import StoreError
    from tests.test_owner_relay import context

    full = "Map: **North trail**. Return at dawn. " * 12
    files = [{"name": "trail.pdf", "mime": "application/pdf", "size": 55}]
    reference = {**_proposal(tmp_path, full), "author_label": "Ouroboros", "role": "assistant",
                 "snippet": "Map: North trail…", "excerpt": {"text": "North trail", "start_utf16": 5}}
    captured = {**reply_target(tmp_path, reference), "attachments": files}
    owner = {"chat_id": 1, "direction": "in", "source": "web", "user_id": 1,
             "ts": "2026-11-04T09:01:00Z", "client_message_id": "owner-held", "text": "Which trail?",
             "reply_to": reference}
    parent = context(tmp_path)
    parent._held_owner_messages = {owner_row_address(owner): owner}
    _record_owner_directive(parent, source="owner_mailbox", content=owner["text"], msg_id="owner-held",
                            origin={"chat_id": 1, "client_message_id": "owner-held"}, reply_to=captured)

    def unavailable(*_args, **_kwargs):
        raise StoreError("history unavailable")

    monkeypatch.setattr("ouroboros.chat_chain.resolve_row", unavailable)
    if explicit:
        address = owner_row_address(owner)
        monkeypatch.setattr("ouroboros.chat_chain.resolve_row", lambda *_: pytest.fail("the captured source needs no history read"))
        [quote], error = quote_rows(parent, [address])
        assert not error
    else:
        _, address, quote, _, _ = _rows_to_carry(parent, tmp_path)[0]
    internal = quote["owner_row"]["_owner_reply_target"]  # the source rides the internal row, never the client's fields
    assert internal["text"] == full and internal["attachments"] == files
    assert "text" not in quote["reply_to"] and "attachments" not in quote["reply_to"], "a client reads the preview alone"
    assert quote["owner_row"]["reply_to"] == reference, "the accepted row is not rewritten with projection fields"
    assert owner_source_row(quote["owner_row"]) == owner and owner_row_address(quote["owner_row"]) == address
    monkeypatch.setattr("ouroboros.chat_chain.resolve_row", lambda *_: pytest.fail("the captured source needs no history read"))
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    frames = []
    entry = {"owner_rows": [quote["owner_row"]], "source_task_id": "teller", "text": "note",
             "relay": {"relay_id": "held-source", "quotes": [quote]},
             **({"host_carried": {"went_to": ""}} if carried else {})}
    inject_relay(receiver, entry, frames)
    target = governing_words_for_schedule(receiver)[FIELD]
    [source] = [row for row in target if row["source"] == REPLY_TARGET]
    assert source["text"] == full and source["attachments"] == files
    assert full in frames[0]["content"] and "trail.pdf" in frames[0]["content"]
    if not carried:
        assert receiver._owner_directives[0]["reply_to"] == captured


@pytest.mark.parametrize("capture", ["absent", "other-owner", "other-source", "other-selection", "other-role"])
def test_explicit_quote_never_borrows_a_source_not_captured_for_its_owner_selection(tmp_path, monkeypatch, capture):
    from ouroboros.owner_relay import owner_row_address, quote_rows
    from tests.test_owner_relay import context

    reference = {"address": "row:1@source#0123456789ab", "author_label": "Mira", "role": "user",
                 "snippet": "North…", "excerpt": {"text": "North", "start_utf16": 0}}
    owner = {"chat_id": 1, "direction": "in", "source": "web", "user_id": 1,
             "ts": "2026-11-04T09:01:00Z", "client_message_id": "owner-held", "text": "Which trail?",
             "reply_to": reference}
    handle, parent = owner_row_address(owner), context(tmp_path)
    parent._held_owner_messages = {handle: owner}
    if capture != "absent":
        target = {**reference, "author": "Mira", "text": "North trail. Unrelated full body."}
        if capture == "other-source":
            target["address"] = "row:1@source#abcdef012345"
        elif capture == "other-selection":
            target["excerpt"] = {"text": "trail", "start_utf16": 6}
        elif capture == "other-role":
            target["role"] = "assistant"
        _record_owner_directive(parent, source="owner_mailbox", content=owner["text"], msg_id="captured",
                                origin={"chat_id": 1, "client_message_id": "other" if capture == "other-owner" else "owner-held"},
                                reply_to=target)
    monkeypatch.setattr("ouroboros.chat_chain.resolve_row", lambda *_: pytest.fail("an explicit quote must not fetch new history"))
    [quote], error = quote_rows(parent, [handle])
    assert not error and quote["reply_to"] == reference
    assert quote["owner_row"] == {**owner, "_owner_source_root": str(tmp_path)}
    assert parent._held_owner_messages == {handle: owner}, "quote preparation changes neither row nor held state"


def test_a_relay_envelope_shows_clients_only_the_preview_and_keeps_the_forwarded_words_and_files(tmp_path):
    """The room, history and every client read ``relay.reply_to`` and each ``quotes[i].reply_to``: the preview alone.
    The selected passage's source rides the internal owner row; the forwarded owner message keeps its own words and
    files whole, whatever the source carries."""
    import json

    from ouroboros.owner_relay import owner_quote, relay_envelope

    full = "Map: **North trail**. Return at dawn. " * 12
    own_words = "Which trail do we take? " * 30
    own_files = [{"name": "photo.png", "mime": "image/png", "size": 9}]
    source_files = [{"name": "trail.pdf", "mime": "application/pdf", "size": 55}]
    excerpt = {"text": "North trail", "start_utf16": 5}
    reference = {"address": "row:1@2026-11-04T09:00:00+00:00#0123456789ab", "role": "assistant", "author_label": "Ouroboros",
                 "snippet": "Map: North trail…", "excerpt": excerpt}
    internal = {"address": reference["address"], "author": "Ouroboros", "excerpt": excerpt, "role": "assistant",
                "text": full, "attachments": source_files}
    owner = {"chat_id": 1, "direction": "in", "source": "web", "user_id": 1, "ts": "2026-11-04T09:01:00Z",
             "client_message_id": "owner-held", "text": own_words, "attachments": own_files, "reply_to": reference}
    quote = owner_quote(tmp_path, {**owner, "_owner_reply_target": internal}, address="handle")
    assert quote["owner_row"]["_owner_reply_target"] == internal, "the source stays on the internal row, whole"
    envelope = relay_envelope("relay-1", [quote], chat_id=1)
    for shown in (envelope, envelope["quotes"][0]):
        assert (shown["text"], shown["attachments"]) == (own_words, own_files), "the forwarded words and files are not cut"
        assert shown["reply_to"]["address"] == reference["address"] and shown["reply_to"]["excerpt"] == excerpt
        assert "text" not in shown["reply_to"] and "attachments" not in shown["reply_to"], "preview fields only"
        assert "owner_row" not in shown
    assert full not in json.dumps(envelope) and "trail.pdf" not in json.dumps(envelope)
    plain = owner_quote(tmp_path, {**owner, "reply_to": reference}, address="handle")
    assert plain["reply_to"] == reference, "a compact reference is its own preview"


@pytest.mark.parametrize("source_kind", ["assistant", "owner-with-files", "assistant-document"])
@pytest.mark.parametrize("delivery", ["web", "web-retry", "local-retry", "recovery"])
def test_a_selected_passage_reaches_every_member_whole_at_receipt_and_the_next_helper_without_history(
        tmp_path, monkeypatch, source_kind, delivery):
    """Immediate delivery from its first step: the owner's Reply to a passage goes through the real ingress, which
    captures the source's words and files before it saves the row. A retry taking an owed dispatch and startup recovery
    must use that capture. The source is unavailable before the host hands it to the tree's members, but every member
    and its next helper still get the exact author, whole text and files. Clients see only the compact reference."""
    import json

    from ouroboros.chat_references import stored_reply
    from ouroboros.owner_relay import carry_row_at_receipt, held_owner_messages
    from ouroboros.owner_words import UNREAD_TARGET, attachments_line, render_owner_words, reply_target
    from ouroboros.store import StoreError
    from ouroboros.task_results import write_task_result
    from supervisor import message_bus
    from supervisor.message_ingress import accepted_chat_message
    from tests.test_chat_attachments import _rows, _web_bridge
    from tests.test_ingress_undispatched_boundary import _facts, _land_then_fail, _new_process
    from tests.test_owner_words_at_receipt import _mail, _nanny, _round_top, _said
    from tests.test_owner_words_carried import _owner_root

    write_task_result(tmp_path, "t-root", "running", delegation_role="root", root_task_id="t-root")
    for task_id, status, parent in (("t-nanny", "running", "t-root"), ("t-child", "scheduled", "t-nanny")):
        write_task_result(tmp_path, task_id, status, delegation_role="subagent", parent_task_id=parent, root_task_id="t-root")
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    full = "Map: **North trail**. Return at dawn. " * 12
    files = [{"name": "trail.pdf", "mime": "application/pdf", "size": 55}]
    url = ""
    if source_kind == "assistant":
        source = message_bus.log_chat("out", 1, 0, full, ts="2026-11-09T09:00:00Z")
        files = []
    elif source_kind == "assistant-document":
        url = "/api/tasks/t-root/artifacts/trail.pdf"
        payload = b"synthetic document payload"
        ok, note = bridge.send_document(1, payload, filename="trail.pdf", caption=full, mime="application/pdf",
                                        download_url=url, task_id="t-root")
        assert ok, note
        [source] = _rows(tmp_path)
        assert not source.get("attachments") and "file_base64" not in source
        files = [{"name": "trail.pdf", "mime": "application/pdf", "size": len(payload), "download_url": url}]
    else:
        source = message_bus.log_chat("in", 1, 1, f"{full}\n\n[Attached file: trail.pdf]", source="web",
                                      ts="2026-11-09T09:00:00Z", message_meta={"attachments": files})
    address = format_address(row_address(source))
    excerpt = {"text": "North trail", "start_utf16": 11}

    def unavailable_source(*_args, **_kwargs):
        raise StoreError("the quoted source is no longer readable")

    def send():
        args = {"client_message_id": "owner-reply", "task_metadata": {"reply_to": address, "reply_excerpt": excerpt}}
        if delivery == "local-retry":
            return message_bus.accept_local_message(bridge, tmp_path, "Which trail?", chat_id=1, user_id=1,
                                                    source="web", **args)
        return bridge.handle_web_message("Which trail?", **args)

    # 1. Capture at the first acceptance; after a lost acknowledgement the source is gone before the takeover.
    if delivery == "web":
        send()
    else:
        restore = _land_then_fail(monkeypatch)
        with pytest.raises(RuntimeError, match="acknowledgement was lost"):
            send()
        restore()
        _new_process(monkeypatch)
        assert _facts(tmp_path, 1, "owner-reply")["state"] == "pending" and bridge._inbox.empty()
        with monkeypatch.context() as missing:
            missing.setattr("ouroboros.chat_chain.resolve_row", unavailable_source)
            missing.setattr("ouroboros.chat_references.resolve_row", unavailable_source)
            if delivery == "recovery":
                assert message_bus.recover_pending_dispatches(bridge, tmp_path) == 1
            else:
                send()
            send()
        assert _facts(tmp_path, 1, "owner-reply")["state"] == "completed" and bridge._inbox.qsize() == 1
    delivered = bridge._inbox.get_nowait()["task_metadata"]["reply_reference"]
    saved = accepted_chat_message(tmp_path, 1, "owner-reply")
    assert delivered["text"] == full and delivered.get("attachments", []) == files, "captured before the row was saved"
    assert saved["reply_to"] == stored_reply(delivered), "the row stays compact"
    if delivery != "local-retry":
        assert echoes[-1]["reply_to"] == saved["reply_to"], "the client's echo stays compact"
    expected = reply_target(tmp_path, delivered)  # the record's own reading of the source, while history reads
    assert expected["text"] == full and expected.get("attachments", []) == files and expected["author"]
    label, role = ("You", "user") if source_kind == "owner-with-files" else ("Ouroboros", "assistant")
    assert expected["role"] == delivered["role"] == role and delivered["author_label"] == label

    # 2. It becomes unreadable before anyone gets the carried copy or attempts another history read.
    monkeypatch.setattr("ouroboros.chat_chain.resolve_row", unavailable_source)
    carry_row_at_receipt(tmp_path, "t-root", 1, "owner-reply", reply=delivered)
    for member in ("t-nanny", "t-child"):
        [entry] = _mail(tmp_path, member).values()
        assert entry["host_carried"] == {"went_to": ""}
        target = entry["owner_rows"][0]["_owner_reply_target"]
        assert target["text"] == full and target.get("attachments", []) == files
        assert (target["author"], target["role"]) == (label, role), "history can no longer expand the accepted author label"
        assert full not in json.dumps(entry["relay"]), "no client-visible field holds the source"
        assert not url or url not in json.dumps(entry["relay"])
    nanny, messages = _nanny(tmp_path), [{"role": "user", "content": "your assignment"}]
    controls = _round_top(nanny, messages)
    assert controls["owner_carried_addressed"] == [format_address(row_address(saved))]
    said = _said(messages)
    assert full in said and "Which trail?" in said and (not files or "trail.pdf" in said)
    assert not url or url in said
    held = held_owner_messages(nanny)[format_address(row_address(saved))]
    assert held["_owner_reply_target"]["text"] == full and held["_owner_reply_target"].get("attachments", []) == files

    # 3. The next helper, and the one after it, read the accepted label and the whole text and files by value.
    child = _owner_root(tmp_path, governing_words_for_schedule(nanny))
    for rows in (governing_words_for_schedule(nanny)[FIELD], governing_words_for_schedule(child)[FIELD]):
        [target] = [row for row in rows if row["source"] == REPLY_TARGET]
        assert (target["author"], target["text"], target.get("attachments", [])) == (label, full, files)
        rendered = render_owner_words(rows, audience="child")
        assert f"[{label}: the message my human replied to, not my human's words · {address}]\n{full}" in rendered
        assert UNREAD_TARGET not in rendered and attachments_line(files) in rendered + ("" if files else attachments_line(files))
        assert not url or url in rendered
        assert "Sender-selected passage of" in rendered and json.dumps(excerpt["text"]) in rendered


@pytest.mark.parametrize("standing", ["addressed", "context"])
def test_a_later_copy_completes_a_held_selection_once_without_raising_its_standing_or_repeating_words(
        tmp_path, monkeypatch, standing):
    """A member that holds a selected reply without its source (history was unreadable when the row reached it) takes
    the source from a later copy of the same address exactly once, whichever standing wrote either copy: only the
    source moves. The row stays what it was to its holder (addressed to this work, or context that went elsewhere),
    no words are shown again and no closing control fires."""
    from ouroboros.owner_relay import (ROUTED_BY_HEAD, captured_selection, carry_at_receipt, carry_row_at_receipt,
                                       held_owner_messages, owner_quote, owner_row_address)
    from ouroboros.store import StoreError
    from ouroboros.task_results import write_task_result
    from supervisor import message_bus
    from supervisor.message_ingress import accepted_chat_message
    from tests.test_chat_attachments import _web_bridge
    from tests.test_owner_words_at_receipt import _mail, _nanny, _round_top, _said

    write_task_result(tmp_path, "t-root", "running", delegation_role="root", root_task_id="t-root")
    write_task_result(tmp_path, "t-nanny", "running", delegation_role="subagent", parent_task_id="t-root",
                      root_task_id="t-root")
    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    full = "Map: **North trail**. Return at dawn. " * 12
    source = message_bus.log_chat("out", 1, 0, full, ts="2026-11-09T09:00:00Z")
    bridge.handle_web_message("Which trail?", client_message_id="owner-reply", task_metadata={
        "reply_to": format_address(row_address(source)), "reply_excerpt": {"text": "North trail", "start_utf16": 11}})
    delivered = bridge._inbox.get_nowait()["task_metadata"]["reply_reference"]
    row = accepted_chat_message(tmp_path, 1, "owner-reply")
    address = owner_row_address(row)
    went = "" if standing == "addressed" else ROUTED_BY_HEAD

    def carry(reply):
        """One carriage of the owner's row: at receipt (addressed), or as the context another work received."""
        if standing == "addressed":
            return carry_row_at_receipt(tmp_path, "t-root", 1, "owner-reply", reply=reply)
        held = {**row, "_owner_source_root": str(tmp_path)}
        if selection := captured_selection(tmp_path, row, reply):
            held["_owner_reply_target"] = selection
        return carry_at_receipt(tmp_path, "t-root", [owner_quote(tmp_path, held, address=address)], went_to=went)

    def unavailable(*_args, **_kwargs):
        raise StoreError("history unavailable")

    # The first copy reaches the member while history is unreadable and carries no source: it holds the row alone.
    with monkeypatch.context() as unread:
        unread.setattr("ouroboros.chat_chain.resolve_row", unavailable)
        carry(None)
    [first] = _mail(tmp_path, "t-nanny").values()
    assert first["host_carried"] == {"went_to": went} and "_owner_reply_target" not in first["owner_rows"][0]
    nanny, messages = _nanny(tmp_path), [{"role": "user", "content": "your assignment"}]
    controls = _round_top(nanny, messages)
    assert ("owner_carried_addressed" in controls) == (standing == "addressed")
    held = held_owner_messages(nanny)[address]
    assert "_owner_reply_target" not in held and held["carried"]["went_to"] == went
    assert _said(messages).count("Which trail?") == 1

    # A later copy, now with the source the delivery captured, completes it once.
    carry(delivered)
    completing = first["msg_id"].replace(":carry:" if standing == "addressed" else ":carry-context:", ":carry-source:")
    assert sorted(_mail(tmp_path, "t-nanny")) == sorted([first["msg_id"], completing])
    later_messages = [{"role": "user", "content": "your assignment"}]
    controls = _round_top(nanny, later_messages)
    held = held_owner_messages(nanny)[address]
    assert held["_owner_reply_target"]["text"] == full and held["carried"] == {**held["carried"], "went_to": went}
    assert "owner_carried_addressed" not in controls and "Which trail?" not in _said(later_messages), "nothing shown again"
    carry(delivered)  # a retried carriage writes nothing more
    assert len(_mail(tmp_path, "t-nanny")) == 2


@pytest.mark.parametrize("same_passage", [True, False])
@pytest.mark.parametrize("renamed_row", [False, True], ids=["receipt", "drain-after-rename"])
def test_a_later_copy_completes_a_selection_inherited_at_birth_without_an_earlier_mail(
    tmp_path, monkeypatch, same_passage, renamed_row,
):
    from ouroboros.owner_relay import (captured_selection, carry_row_at_receipt, carry_to_members, held_owner_messages,
                                      owner_row_address)
    from ouroboros.owner_words import UNREAD_TARGET, render_owner_words
    from ouroboros.store import StoreError
    from ouroboros.task_results import write_task_result
    from supervisor import message_bus
    from supervisor.message_ingress import accepted_chat_message
    from tests.test_chat_attachments import _web_bridge
    from tests.test_owner_words_at_receipt import _mail, _nanny, _round_top, _said
    from tests.test_owner_words_carried import _owner_root

    write_task_result(tmp_path, "t-root", "running", delegation_role="root", root_task_id="t-root")
    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    full = "North trail. Keep going. North trail. Turn back. " * 6
    source = message_bus.log_chat("out", 1, 0, full, ts="2026-11-09T09:00:00Z")
    bridge.handle_web_message("Which trail?", client_message_id="owner-reply", task_metadata={
        "reply_to": format_address(row_address(source)), "reply_excerpt": {"text": "North trail", "start_utf16": 0}})
    delivered = bridge._inbox.get_nowait()["task_metadata"]["reply_reference"]
    row = accepted_chat_message(tmp_path, 1, "owner-reply")
    address = owner_row_address(row)
    preview = dict(row["reply_to"])
    if not same_passage:
        preview["excerpt"] = {"text": "North trail", "start_utf16": full.index("North trail", 1)}
    if renamed_row:
        root = _nanny(tmp_path)
        root.task_id = "t-root"
        root.task_metadata = {"root_task_id": "t-root", "delegation_role": "root"}
        _record_owner_directive(root, source="owner_mailbox", content="Which trail?", msg_id="mail-1",
                                origin={"client_message_id": "owner-reply"},
                                reply_to={**preview, "author": preview.get("author_label", "")})
    else:
        root = _owner_root(tmp_path)
        root._held_owner_messages = {address: {**row, "_owner_source_root": str(tmp_path)}}
    inherited = governing_words_for_schedule(root)
    if not renamed_row and not same_passage:
        # The accepted row keeps its exact physical identity (unchanged digest); only its
        # presentation-only inherited citation differs from the raw row's own excerpt.
        inherited[FIELD][0]["reply_citation"]["excerpt"] = preview["excerpt"]
    assert inherited[FIELD][0]["reply_citation"]["reply_view"] == "unread"
    if renamed_row:
        assert inherited[FIELD][0]["ref"] == "msg mail-1"
    write_task_result(tmp_path, "t-nanny", "running", delegation_role="subagent", parent_task_id="t-root",
                      root_task_id="t-root", **inherited)
    nanny = _nanny(tmp_path)
    nanny.task_metadata.update(inherited)
    assert _mail(tmp_path, "t-nanny") == {} and held_owner_messages(nanny) == {}
    opening = render_owner_words(inherited[FIELD], audience="child")
    assert UNREAD_TARGET in opening and opening.count("Which trail?") == 1
    messages = [{"role": "system", "content": opening}, {"role": "user", "content": "your assignment"}]

    def unavailable(*_args, **_kwargs):
        raise StoreError("history unavailable")

    monkeypatch.setattr("ouroboros.chat_chain.resolve_row", unavailable)
    if renamed_row:
        selection = captured_selection(tmp_path, row, delivered)
        assert selection and selection["text"] == full
        root._held_owner_messages = {address: {**row, "_owner_reply_target": selection,
                                               "_owner_source_root": str(tmp_path)}}
        assert governing_words_for_schedule(root)[FIELD][0]["ref"] == "chat 1 / owner-reply"
    for _ in range(2):
        if renamed_row:
            carry_to_members(root, delivered=True)
        else:
            carry_row_at_receipt(tmp_path, "t-root", 1, "owner-reply", reply=delivered)
    assert list(_mail(tmp_path, "t-nanny")) == ([f"relay:carry-source:{address}:t-nanny"] if same_passage else [])
    controls = _round_top(nanny, messages)
    assert "owner_carried_addressed" not in controls and _said(messages).count("Which trail?") == 1
    rows = governing_words_for_schedule(nanny)[FIELD]
    [source] = [item for item in rows if item["source"] == REPLY_TARGET]
    assert source["text"] == (full if same_passage else "")
    assert rows[0]["ref"] == inherited[FIELD][0]["ref"] and rows[0]["carrier"] == inherited[FIELD][0]["carrier"]
    assert render_owner_words(rows, audience="child").count("Which trail?") == 1
    restored = _nanny(tmp_path)
    restored.task_metadata.update(inherited)
    held_owner_messages(restored)
    assert governing_words_for_schedule(restored)[FIELD] == rows, "the complete source survives the recipient's restore"
    if renamed_row and same_passage:
        # A child whose own preview predates the rename can also receive the supplement from this holder.
        write_task_result(tmp_path, "t-child", "scheduled", delegation_role="subagent", parent_task_id="t-nanny",
                          root_task_id="t-root", **inherited)
        carry_to_members(restored, delivered=True)
        [key] = _mail(tmp_path, "t-child")
        assert key == f"relay:carry-source:{address}:t-child"
