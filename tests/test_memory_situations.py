"""Situation names use the memory view's role and exact host-bound room identities."""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

from ouroboros import dialogue_import, dialogue_store, places
from ouroboros.consciousness_authority import consciousness_origin_metadata
from ouroboros.memory_view import view_room_id
from ouroboros.memory_view_events import CARRIED_NAMES, carried_room, helper_names, situation_names
from ouroboros.places import connect_place, conversation_chat_id
from ouroboros.presence_bindings import conversation_key
from ouroboros.presence_context import focus_metadata
from ouroboros.presence_events import PresenceTurnEvent
from ouroboros.projects_registry import create_project
from ouroboros.tools.knowledge import _knowledge_read, _knowledge_write
from ouroboros.tools.registry import ToolContext
from tests._dialogue import seed


def test_knowledge_receipts_show_the_project_room_in_integrating_foci_and_are_quiet_for_helpers(tmp_path):
    project = create_project(tmp_path, "nova", name="Nova")
    room = project["chat_id"]
    exact = f"This situation's names: room:{room} (Project Nova)"
    fields = {"repo_dir": tmp_path, "drive_root": tmp_path, "current_chat_id": room}
    integrating = ToolContext(**fields, task_id="root")
    child = ToolContext(**fields, task_id="child", task_contract={"lineage": {"delegation_role": "subagent"}})
    nanny = ToolContext(**fields, task_id="nanny", task_metadata={"configured_subagent": {"route": {"kind": "agent_session"}}})

    first = _knowledge_read(integrating, "unwritten", scope="global")
    assert "Topic 'unwritten' not found" in first and first.splitlines().count(exact) == 1
    assert "This situation's names:" not in _knowledge_read(child, "unwritten", scope="global")

    for index, ctx in enumerate((integrating, child, nanny)):
        topic = f"topic-{index}"
        content = f"---\nclaims: ['room:{room}', 'place:missing']\n---\nRoom facts."
        written = _knowledge_write(ctx, topic, content=content, mode="overwrite", scope="global")
        read = _knowledge_read(ctx, topic, scope="global")
        assert written.startswith("✅") and read.endswith("Room facts.")
        assert f"Present situations holding this note: room:{room}" in written
        assert "Claims with no matching situation found: 'place:missing'" in written
        assert written.splitlines().count(exact) == read.splitlines().count(exact) == int(index == 0)
        if index:
            assert "This situation's names:" not in written and "This situation's names:" not in read


def test_place_names_use_exact_carried_components_and_no_thread_name_without_a_thread(tmp_path):
    place = connect_place(tmp_path, connector="fixture", provider="chat", label="Nova Team", place_id="p/%")
    route = {"provider": "chat", "account_id": "A/%:B@team", "conversation_id": "C/%", "thread_id": "T/%"}
    key = conversation_key(**route)
    chat = conversation_chat_id(key)
    event = PresenceTurnEvent(source_event_id="event-1", conversation_key=key, actor={"id": "member"},
                              conversation={"name": "Nova discussion"}, message={}, text="Status", **route)
    admission = SimpleNamespace(place=place, binding_id=place.place_id, allowance={}, circle=None)
    metadata = focus_metadata(admission, event, "place", chat)
    metadata["presence"] = {"place_id": place.place_id, "event": {**route, "conversation": event.conversation}}
    names = (f"room:{chat}", "place:p%2F%25", "conversation:chat/A%2F%25:B@team/C%2F%25",
             "thread:chat/A%2F%25:B@team/C%2F%25/T%2F%25")
    assert situation_names({"metadata": metadata}, str(chat)) == names
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="place", task_metadata=metadata)
    reply = _knowledge_read(ctx, "unwritten", scope="global")
    assert all(name in reply for name in names)
    assert "place:p%2F%25 (Nova Team)" in reply
    assert f"{names[2]} (Nova discussion)" in reply

    def conversation_label(conversation):
        fields = deepcopy(metadata)
        fields["presence"]["event"]["conversation"] = conversation
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="place", task_metadata=fields)
        return next(line for line in _knowledge_read(ctx, "unwritten", scope="global").splitlines()
                    if line.startswith("This situation's names:"))

    # The label is a string or nothing: a list, a mapping or a number under a label key never renders.
    assert f"{names[2]}, {names[3]}" in conversation_label({"title": ["Nova", "talk"], "name": {"first": "Nova"}, "username": 7})
    assert f"{names[2]} (nova_team), {names[3]}" in conversation_label({"title": ["Nova", "talk"], "username": "nova_team"})

    carried = consciousness_origin_metadata(metadata, started_root=True)
    assert "presence" not in carried
    ordinary_task = {"type": "task", "metadata": carried}
    assert situation_names(ordinary_task, "1") == ("room:1", *names[1:])
    for thread, expected in (("", names[:-1]), ("0", (*names[:-1], names[-1].rsplit("/", 1)[0] + "/0"))):
        focus = {"metadata": {**carried, "focus_transport": {**route, "thread_id": thread}}}
        assert situation_names(focus, str(chat)) == expected
    assert situation_names({"type": "task"}, "1") == ("room:1",)


def test_a_helper_carries_its_place_turns_names_by_value_and_reads_their_room(tmp_path):
    """A place turn's helper reads the names its scheduler stated, unchanged down the tree; one that carries none
    computes the old mixture (Main's room and the place's names) and passes nothing on."""
    place = connect_place(tmp_path, connector="fixture", provider="chat", label="Nova Team", place_id="team")
    route = {"provider": "chat", "account_id": "team", "conversation_id": "nova", "thread_id": ""}
    key = conversation_key(**route)
    chat = conversation_chat_id(key)
    event = PresenceTurnEvent(source_event_id="event-1", conversation_key=key, actor={"id": "member"},
                              conversation={"name": "Nova"}, message={}, text="Status", **route)
    admission = SimpleNamespace(place=place, binding_id=place.place_id, allowance={}, circle=None)
    metadata = {**focus_metadata(admission, event, "place", chat), "presence": {"place_id": "team", "event": route}}
    turn = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="place", task_metadata=metadata)
    names = (f"room:{chat}", "place:team", "conversation:chat/team/nova")
    assert helper_names(turn, tmp_path) == names  # the place turn's own names: its room is its conversation

    inherited = {**consciousness_origin_metadata(metadata), "delegation_role": "subagent"}
    child = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="child",
                        task_metadata={**inherited, CARRIED_NAMES: list(names)})
    for task in ({"metadata": child.task_metadata}, dict(child.task_metadata)):  # a record, and the flat metadata
        assert situation_names(task, "1") == names and carried_room(task) == str(chat)
        assert view_room_id(task, None, tmp_path, project_chat_ids=()) == str(chat)
    assert helper_names(child, tmp_path) == names  # a grandchild inherits them unchanged

    uncarried = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="old-child", task_metadata=inherited)
    assert situation_names({"metadata": inherited}, "1") == ("room:1", *names[1:]) and carried_room(inherited) is None
    assert view_room_id({"metadata": inherited}, None, tmp_path, project_chat_ids=()) == "1"
    assert helper_names(uncarried, tmp_path) == ()


def test_legacy_conversation_keys_resolve_only_unambiguous_components():
    focus = {"place_id": "team", "conversation_key": "chat:A/@%:C:0"}
    known = ("room:1", "place:team")
    assert situation_names({"focus": focus}, "1") == (*known, "conversation:chat/A%2F@%25/C")
    with_thread = {**focus, "conversation_key": "chat:A/@%:C:17"}
    assert situation_names({"focus": with_thread}, "1") == (*known, "conversation:chat/A%2F@%25/C",
                                                            "thread:chat/A%2F@%25/C/17")

    ambiguous = {**focus, "conversation_key": "chat:A:/@%:C:0"}
    assert situation_names({"focus": ambiguous}, "1") == known
    structured = {"provider": "chat", "account_id": "A:/@%", "conversation_id": "C", "thread_id": "0"}
    assert situation_names({"metadata": {"focus": ambiguous, "focus_transport": structured}}, "1") == (
        *known, "conversation:chat/A:%2F@%25/C", "thread:chat/A:%2F@%25/C/0")


def test_place_transport_stays_out_of_safety_and_continue_bytes_but_reaches_successor_receipts(tmp_path, monkeypatch):
    from ouroboros import safety
    from ouroboros.owner_continue import _wake_opening, work_order_text
    from supervisor.continuation_admission import _successor_task

    monkeypatch.setattr(safety, "utc_now_iso", lambda *_args: "2026-11-18T12:00:00+00:00")
    place = connect_place(tmp_path, connector="fixture", provider="chat", label="Nova Team", place_id="team")
    route = {"provider": "chat", "account_id": "acct:/west@team", "conversation_id": "project/%", "thread_id": "w/17"}
    key = conversation_key(**route)
    chat = conversation_chat_id(key)
    event = PresenceTurnEvent(source_event_id="event-1", conversation_key=key, actor={"id": "member", "display_name": "Sam"},
                              conversation={"name": "Nova discussion"}, message={}, text="Owner's opening words", **route)
    admission = SimpleNamespace(place=place, binding_id=place.place_id, allowance={}, circle=None)
    metadata = focus_metadata(admission, event, "transport-predecessor", chat)
    metadata["presence"] = {"place_id": place.place_id, "event": {**route, "conversation": event.conversation},
                            "observed_text": event.text}
    base = deepcopy(metadata)
    base.pop("focus_transport", None)
    base["focus"].pop("transport", None)
    retained = deepcopy(base)
    retained["focus"]["transport"] = route
    assert metadata["focus_transport"] == route and metadata["focus"] == base["focus"]
    assert not (tmp_path / "memory" / "knowledge").exists()

    def check_prompt(meta):
        ctx = SimpleNamespace(task_metadata=meta, task_id="transport-predecessor", drive_root=tmp_path)
        return safety._build_check_prompt("run_command", {"cmd": ["echo", "hello"]},
                                          [{"role": "user", "content": "The opening frame"}], ctx=ctx).encode()

    def successor(meta):
        gaps = []
        wake = _wake_opening(meta, "transport-predecessor", gaps)
        assert wake and not gaps
        sources = {"wake": wake, "later": [{"source": "owner_mailbox", "content": "Later owner words, verbatim."}]}
        task = _successor_task(SimpleNamespace(DRIVE_ROOT=tmp_path), "transport-predecessor", {"metadata": meta},
                               {"successor_task_id": "transport-successor", "chat_id": 1}, {"cause": "cancelled"}, sources, {})
        return task, sources

    before = check_prompt(base)
    assert check_prompt(metadata) == before
    evidence = json.loads(before.decode().split("Host evidence (provenance is not consent):\n", 1)[1]
                          .split("\nProposed tool call:", 1)[0])
    assert evidence["occasion"]["focus"] == metadata["focus"]
    assert b'"display_name":"Sam"' in before or b'"display_name": "Sam"' in before
    base_task, base_sources = successor(base)
    carried, _ = successor(metadata)
    legacy, _ = successor(retained)
    assert carried["text"].encode() == legacy["text"].encode() == base_task["text"].encode()
    assert event.text in carried["text"] and "Later owner words, verbatim." in carried["text"]
    saved_sources = deepcopy(base_sources)
    saved_sources["wake"]["focus"]["transport"] = route
    assert work_order_text("transport-predecessor", "cancelled", saved_sources) == work_order_text(
        "transport-predecessor", "cancelled", base_sources)

    names = ("conversation:chat/acct:%2Fwest@team/project%2F%25", "thread:chat/acct:%2Fwest@team/project%2F%25/w%2F17")
    for task, expected in ((carried, True), (legacy, True), (base_task, False)):
        assert "presence" not in task["metadata"]
        assert "transport" not in task["metadata"]["focus"]
        assert task["metadata"].get("focus_transport") == (route if expected else None)
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"], task_metadata=task["metadata"])
        reply = _knowledge_read(ctx, "unwritten", scope="global")
        assert all((name in reply) == expected for name in names)


def _append_witness(root, conversation, *, account="T", thread="17", direction="in", state=None, stored_room=None):
    route = {"provider": "chat", "account_id": account, "conversation_id": conversation, "thread_id": thread}
    chat = conversation_chat_id(conversation_key(**route))
    row = {"chat_id": chat if stored_room is None else stored_room, "transport": route, "direction": direction,
           "type": "place_arrival" if state is None else "presence_delivery", "text": f"{conversation} evidence"}
    if state is not None:
        route["delivery"] = {"state": state}
    dialogue_store.append(root, "chat", row)
    return [f"room:{chat}", f"conversation:chat/{account}/{conversation}", f"thread:chat/{account}/{conversation}/{thread}"]


def _write_claims(root, names, topic="nova"):
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="receipt-writer")
    return _knowledge_write(ctx, topic, mode="overwrite", content=f"---\nclaims: {json.dumps(names)}\n---\nNotes.")


def _claim_diagnostic(reply, prefix):
    return next((line.removeprefix(prefix) for line in reply.splitlines() if line.startswith(prefix)), None)


def test_only_positive_typed_routes_that_derive_their_rooms_match_claims(tmp_path):
    # An account-wide place specifies no conversation; typed dialogue is the witness.
    connect_place(tmp_path, connector="fixture", provider="chat", account_id="T", place_id="team")
    arrived = _append_witness(tmp_path, "inbound", thread="0")
    accepted = _append_witness(tmp_path, "accepted", direction="out", state="accepted")
    delivered = _append_witness(tmp_path, "delivered", account="acct ", direction="out", state="delivered")
    for names in (arrived, accepted, delivered):
        dialogue_store.append(tmp_path, "chat", {"chat_id": int(names[0][5:]), "direction": "in", "text": "No transport here"})

    other_room = 515151
    wrong = _append_witness(tmp_path, "wrong-room", stored_room=other_room)
    failed = _append_witness(tmp_path, "failed", direction="out", state="failed")
    unaccepted = _append_witness(tmp_path, "pending", direction="system", state="accepted")
    trimmed = conversation_chat_id(conversation_key("chat", "acct", "delivered", "17"))
    unread = [*wrong, f"room:{other_room}", *failed, *unaccepted,
              f"room:{trimmed}", "conversation:chat/acct/delivered", "thread:chat/acct/delivered/17"]
    live = ["room:1", "place:team", *arrived, *accepted, *delivered]
    reply = _write_claims(tmp_path, live + unread)
    assert reply.startswith("✅")
    assert _claim_diagnostic(reply, "Present situations holding this note: ") == ", ".join(live)
    assert _claim_diagnostic(reply, "Claims with no matching situation found: ") == ", ".join(map(repr, unread))
    assert "unverified" not in reply


def test_claim_receipt_distinguishes_unreadable_evidence_from_proved_absence(tmp_path, monkeypatch):
    connect_place(tmp_path, connector="fixture", provider="chat", account_id="T", place_id="team")
    place_path = places.places_path(tmp_path)
    registry = place_path.read_bytes()
    route = _append_witness(tmp_path, "heard")
    external = [*route, "room:343434"]
    names = ["room:1", "place:team", "place:missing", *external, "other"]
    matched_prefix = "Present situations holding this note: "
    unmatched_prefix = "Claims with no matching situation found: "
    unknown_prefix = "Claims with unverified situations: "

    readable = _write_claims(tmp_path, names, topic="readable")
    assert _claim_diagnostic(readable, matched_prefix) == ", ".join(["room:1", "place:team", *route])
    assert _claim_diagnostic(readable, unmatched_prefix) == "'place:missing', 'room:343434', 'other'"
    assert unknown_prefix not in readable

    place_path.write_bytes(b"Unreadable registry")
    missing_places = _write_claims(tmp_path, names, topic="missing-places")
    assert _claim_diagnostic(missing_places, matched_prefix) == ", ".join(["room:1", *route])
    assert _claim_diagnostic(missing_places, unmatched_prefix) == "'room:343434', 'other'"
    assert _claim_diagnostic(missing_places, unknown_prefix) == "'place:team', 'place:missing'"
    place_path.write_bytes(registry)

    def unavailable(*_args, **_kwargs):
        raise dialogue_store.StoreLost("unreadable dialogue store")

    monkeypatch.setattr(dialogue_store, "transport_witnesses", unavailable)
    missing_dialogue = _write_claims(tmp_path, names, topic="missing-dialogue")
    assert _claim_diagnostic(missing_dialogue, matched_prefix) == "room:1, place:team"
    assert _claim_diagnostic(missing_dialogue, unmatched_prefix) == "'place:missing', 'other'"
    assert _claim_diagnostic(missing_dialogue, unknown_prefix) == ", ".join(map(repr, external))
    solely_unknown = _write_claims(tmp_path, route, topic="unknown")
    assert _claim_diagnostic(solely_unknown, matched_prefix) == "unverified"
    assert unmatched_prefix not in solely_unknown


def test_claim_receipt_leaves_archive_routes_unverified_until_import(tmp_path):
    route = {"provider": "chat", "account_id": "T", "conversation_id": "archive", "thread_id": "17"}
    chat = conversation_chat_id(conversation_key(**route))
    delivery = {"chat_id": chat, "direction": "out", "type": "presence_delivery", "text": "Earlier delivery",
                "transport": {**route, "delivery": {"state": "delivered"}}}
    seed(tmp_path, archives={"chat_20261118T000000.jsonl": [delivery]})
    dialogue_store.ensure(tmp_path)
    [segment] = dialogue_import.pending_segments(tmp_path)
    names = [f"room:{chat}", "conversation:chat/T/archive", "thread:chat/T/archive/17"]
    missing = "room:343434"
    matched_prefix = "Present situations holding this note: "
    unmatched_prefix = "Claims with no matching situation found: "
    unknown_prefix = "Claims with unverified situations: "

    pending = _write_claims(tmp_path, [*names, missing], topic="pending")
    assert pending.startswith("✅")
    assert _claim_diagnostic(pending, matched_prefix) == "unverified"
    assert _claim_diagnostic(pending, unknown_prefix) == ", ".join(map(repr, [*names, missing]))
    assert unmatched_prefix not in pending

    dialogue_import.import_segment(tmp_path, "chat", segment["name"])
    imported = _write_claims(tmp_path, [*names, missing], topic="imported")
    assert imported.startswith("✅")
    assert _claim_diagnostic(imported, matched_prefix) == ", ".join(names)
    assert _claim_diagnostic(imported, unmatched_prefix) == repr(missing)
    assert unknown_prefix not in imported
