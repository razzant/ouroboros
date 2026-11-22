"""Reply facts and addresses name persisted rows, independently of their transport/view."""
import json

import pytest

from ouroboros import dialogue_store
from ouroboros.chat_chain import format_address, row_address
from ouroboros.chat_references import reference_from_row, resolve_reply
from supervisor import message_bus
from tests import _dialogue as dialogue_test
from tests.test_chat_attachments import _history, _web_bridge


def test_reply_is_host_resolved_and_replay_uses_the_same_address(tmp_path, monkeypatch):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    original = message_bus.log_chat("out", 1, 0, "  One\n\t" + "word " * 80, ts="2026-10-01T00:00:00Z")
    address = format_address(row_address(original))
    bridge.handle_web_message("this one", client_message_id="reply", task_metadata={"reply_to": address})
    row = dialogue_test.chat_rows(tmp_path)[-1]
    assert row["reply_to"]["snippet"] == ("One " + "word " * 80)[:159].rstrip() + "…"  # a cut preview says so
    assert row["reply_to"]["chars"] == len(original["text"]), "and the reference keeps how long the whole message is"
    assert row["reply_to"]["role"] == "assistant"
    assert echoes[-1]["reply_to"] == row["reply_to"]
    assert echoes[-1]["row_address"] == format_address(row_address(row))
    replay = next(r for r in _history(tmp_path) if r.get("client_message_id") == "reply")
    assert replay["row_address"] == echoes[-1]["row_address"]
    assert replay["reply_to"] == row["reply_to"]
    queued = bridge._inbox.get_nowait()
    assert queued["task_metadata"]["reply_reference"] == row["reply_to"]
    bridge.handle_web_message("this one", client_message_id="reply", task_metadata={"reply_to": address})
    assert bridge._inbox.empty()
    with pytest.raises(ValueError, match="different message"):
        bridge.handle_web_message("this one", client_message_id="reply", task_metadata={"reply_to": "task:other"})


def test_reply_survives_rotation_and_missing_reference_stays_explicit(tmp_path):
    row = {"chat_id": 1, "ts": "2026-10-01T00:00:00Z", "direction": "in", "text": "yes"}
    dialogue_test.seed(tmp_path, archives={"chat_20261002T000000.jsonl": [row]})
    address = format_address(row_address(row))
    assert resolve_reply(tmp_path, address)["unresolved"] is True  # still importing: unknown, never a failure
    dialogue_test.import_all(tmp_path)
    assert resolve_reply(tmp_path, address)["snippet"] == "yes"
    assert resolve_reply(tmp_path, "task:missing") == {"address": "task:missing", "unresolved": True}
    assert resolve_reply(tmp_path, address[:-12] + "000000000000")["unresolved"] is True
    with pytest.raises(ValueError, match="string"):
        resolve_reply(tmp_path, {"snippet": "forged"})


@pytest.mark.parametrize("extra", [{}, {"reply_excerpt": {"text": "this", "start_utf16": 5}}])
def test_skill_ingress_shares_reply_and_address_with_web(tmp_path, monkeypatch, extra):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    row = message_bus.log_chat("out", 1, 0, "Pick this", ts="2026-10-01T00:00:00Z")
    address = format_address(row_address(row))
    accepted, rejoined = message_bus.accept_local_message(
        bridge, tmp_path, "yes", chat_id=1, user_id=1, client_message_id="skill-reply",
        source="skill:telegram", task_metadata={"reply_to": address, **extra})
    assert not rejoined and accepted["reply_to"]["address"] == address
    assert accepted["reply_to"].get("excerpt") == extra.get("reply_excerpt")
    queued = bridge._inbox.get_nowait()
    message_bus.record_inbound_message(bridge, queued, chat_id=1, user_id=1,
        client_message_id="skill-reply", text="yes", ts=accepted["ts"])
    assert echoes[-1]["row_address"] == format_address(row_address(accepted))
    assert echoes[-1]["reply_to"] == accepted["reply_to"]


def test_progress_address_resolves_without_hashing_projection_fields(tmp_path):
    from ouroboros.chat_references import projection_fields
    row = {"chat_id": 1, "ts": "2026-10-01T01:00:00Z", "task_id": "worker", "text": "Testing"}
    dialogue_store.append(tmp_path, "progress", row)
    address = projection_fields(row)["row_address"]
    assert resolve_reply(tmp_path, address)["snippet"] == "Testing"
    assert resolve_reply(tmp_path, address + "0")["unresolved"] is True


def test_answer_row_uses_the_turns_host_origin(tmp_path, monkeypatch):
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    origin = message_bus.log_chat("in", 1, 1, "Original request", client_message_id="origin")
    write_task_result(tmp_path, "work", "running", origin_message_ref=build_owner_message_ref(
        chat_id=1, client_message_id="origin", text=origin["text"], ts=origin["ts"]))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    message_bus.send_with_budget(1, "Answer", task_id="work")
    row = dialogue_test.chat_rows(tmp_path)[-1]
    assert row["reply_to"]["address"] == format_address(row_address(origin))
    assert echoes[-1]["row_address"] == format_address(row_address(row))
    assert echoes[-1]["reply_to"] == row["reply_to"]


@pytest.mark.parametrize("completion", [False, True], ids=["answer", "completion"])
def test_answer_row_of_work_a_card_answer_opened_quotes_the_owners_choice(tmp_path, monkeypatch, completion):
    """Work started on my human's card answer (``start_task`` ``owner_source``) answers in Main under a Reply that reads
    their choice, never the host frame its System row stores; the row's role, address and room rule stay its own,
    and its replay is the same reference. A relay's explicit Reply keeps its own target."""
    from ouroboros.owner_quiz import quiz_answer_frame
    from ouroboros.owner_relay import owner_row_source
    from ouroboros.projects_registry import create_project
    from ouroboros.room_membership import reply_target_home
    from ouroboros.task_results import write_task_result
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    block = {"quiz_id": "q-palette", "question": "Which colors should the table legend use?",
             "options": ["Use blue and gold", "Use gray and green"], "state": "answered",
             "answered_index": 0, "asked_at": "2024-04-09T10:05:00+00:00", "answered_at": "2024-04-09T10:14:00+00:00"}
    room = create_project(tmp_path, "palette")["chat_id"]
    answer = message_bus.log_chat("system", room, 0, quiz_answer_frame(block, 0, ""), ts=block["answered_at"],
                                  source="owner_quiz_answer", task_id="asker", client_message_id="quiz_answer:asker:q-palette",
                                  record_type="quiz_answer", quiz=dict(block))
    result = write_task_result(tmp_path, "work", "completed" if completion else "running",
                               origin_message_ref=owner_row_source(answer)[0])
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    delivery = {}
    if completion:
        from ouroboros.project_completion_facts import completion_facts
        facts = completion_facts(tmp_path, "work", {}, result, {})
        assert facts["reply_to"] == reference_from_row(answer)
        delivery = {"role": "system", "system_type": "project_completion_summary", "progress_meta": facts}
    message_bus.send_with_budget(1, "The table legend is ready.", task_id="work", **delivery)
    row = dialogue_test.chat_rows(tmp_path)[-1]
    assert row["reply_to"] == {**reference_from_row(answer), "snippet": "1. Use blue and gold"}
    assert row["reply_to"]["address"] == format_address(row_address(answer)) and "chars" not in row["reply_to"]
    assert row["reply_to"]["quote_role"] == "user"
    assert (row["reply_to"]["role"], row["reply_to"]["author_label"], row["reply_to"]["kind"]) == ("system", "You", "system")
    assert row["reply_to"]["target_home"] == reply_target_home(answer) and echoes[-1]["reply_to"] == row["reply_to"]
    assert next(r for r in _history(tmp_path) if r["text"] == row["text"])["reply_to"] == row["reply_to"]
    assert resolve_reply(tmp_path, row["reply_to"]["address"]) == row["reply_to"]
    secret = reference_from_row({**answer, "quiz": {**block, "secret_request": {"pluginName": "x"},
                                                     "secret_outcome": {"event": "secret_filled"}}})
    assert secret["snippet"].startswith("[") and secret["author_label"] == "System"  # a form's outcome: no words of theirs
    assert "quote_role" not in secret
    assert "quote_role" not in reference_from_row({**answer, "source": "notice", "sender_label": "You"})
    explicit = {"address": "row:1@2024-04-09T10:15:00+00:00#0123456789ab", "role": "user", "snippet": "Relayed words"}
    message_bus.send_with_budget(1, "Answering the relayed note.", task_id="work",
                                 progress_meta={"reply_to": explicit, "reply_to_relay": "relay-1"})
    assert dialogue_test.chat_rows(tmp_path)[-1]["reply_to"] == explicit


@pytest.mark.parametrize("stored_chat,room", [(None, 1), (0, 0)])
def test_retained_owner_row_keeps_its_original_bytes_and_normalized_room(tmp_path, stored_chat, room):
    from types import SimpleNamespace
    from ouroboros.owner_relay import origin_snapshot, owner_quote, owner_row_source, retain_owner_message
    from ouroboros.owner_source import resolve_owner_source
    from ouroboros.owner_words import carried_owner_corpus

    row = {"direction": "in", "text": "Count the seedlings", "client_message_id": "garden",
           "chat_id": stored_chat, "ts": "2024-04-08T13:45:00Z"}
    dialogue_store.append(tmp_path, "chat", row)
    ctx = SimpleNamespace(drive_root=tmp_path, current_chat_id=room, task_metadata={})
    held = retain_owner_message(ctx, client_message_id="garden")
    assert held == row
    address = row_address(row)
    ref, _ = owner_row_source(held)
    quote = owner_quote(tmp_path, held)
    snapshot = origin_snapshot(held, tmp_path, include_reply=True)
    assert snapshot["projectable"] is True
    assert snapshot["reply"] == reference_from_row(row)
    assert quote["owner_row"]["chat_id"] == stored_chat
    assert ref["row_sha256"] == address["row_sha256"]
    assert quote["source"]["chat_id"] == ref["chat_id"] == snapshot["reply"]["chat_id"] == room
    assert quote["source"]["address"] == snapshot["reply"]["address"] == format_address(address)
    [opening] = carried_owner_corpus(ctx, source=held)
    assert (opening["chat_id"], opening["row_sha256"]) == (str(room), address["row_sha256"])
    resolved = resolve_owner_source(ctx, {"kind": "chat", "ref": ref})
    assert resolved["ref"]["chat_id"] == room and resolved["ref"]["text_sha256"] == ref["text_sha256"]
    ctx.current_chat_id = 73
    assert resolve_owner_source(ctx, {"kind": "chat", "ref": ref}) == {}


@pytest.mark.parametrize("private", [False, True])
def test_automatic_reply_follows_source_visibility_in_the_selected_room(tmp_path, monkeypatch, private):
    from ouroboros.owner_relay import origin_snapshot, owner_row_source
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.task_results import write_task_result

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    row = {"direction": "in", "chat_id": 1, "task_id": "seed-note", "text": "Count the seedlings",
           "client_message_id": "garden", "ts": "2024-04-08T13:45:00Z"}
    if private:
        row["presence_contact"] = {"task_id": "seed-note", "binding_id": "seeds", "source_event_id": "garden"}
    dialogue_store.append(tmp_path, "chat", row)
    ref, text = owner_row_source(row)
    snap = origin_snapshot(row, tmp_path, include_reply=True)
    room = bind_task_to_project(tmp_path, "count", "garden", origin={"ref": ref, "text": text})["project_chat_id"]
    write_task_result(tmp_path, "count", "running", origin_message_ref=ref, metadata={"origin_message_snapshot": snap})
    assert message_bus.send_with_budget(1, "Counted", task_id="count")["chat_id"] == room
    stored = dialogue_test.chat_rows(tmp_path)[-1]
    live = echoes[-1]
    file_row = message_bus.log_chat("out", room, 0, "Seed table", task_id="count", record_type="document")
    # An older outgoing row kept just the generated preview. Read it in the task's current room.
    dialogue_store.append(tmp_path, "chat", {"direction": "out", "chat_id": room, "task_id": "count",
                          "ts": "2024-04-10T10:00:00Z", "text": "Earlier count", "reply_to": reference_from_row(row)})
    history = _history(tmp_path, room)
    replay = next(r for r in history if r["text"] == "Counted")
    older = next(r for r in history if r["text"] == "Earlier count")
    for projection in (stored, file_row, live, replay, older):
        if private:
            assert "reply_to" not in projection
        else:
            assert projection["reply_to"] == reference_from_row(row)


@pytest.mark.parametrize("client_id", ["garden", ""], ids=["named", "idless"])
def test_main_completion_quotes_the_selected_owner_message_in_its_project(tmp_path, monkeypatch, client_id):
    from ouroboros.owner_relay import owner_row_source
    from ouroboros.project_completion_facts import completion_facts
    from ouroboros.project_dialogue import MAIN_NOTICE_TYPE
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    room = create_project(tmp_path, "garden")["chat_id"]
    source = message_bus.log_chat("in", room, 1, "Count the seedlings", client_message_id=client_id)
    ref, _ = owner_row_source(source)
    result = write_task_result(tmp_path, "count", "completed", origin_message_ref=ref)
    other = bind_task_to_project(tmp_path, "count", "inventory", origin={"absent": "system"})["project_chat_id"]
    facts = completion_facts(tmp_path, "count", {}, result, {})
    quote = reference_from_row(source)
    assert facts["reply_to"] == quote
    message_bus.send_with_budget(1, "Count complete", role="system", task_id="count", system_type="project_completion_summary",
                                 progress_meta=facts)
    assert dialogue_test.chat_rows(tmp_path)[-1]["reply_to"] == echoes[-1]["reply_to"] == quote
    history = _history(tmp_path)
    assert next(r for r in history if r["system_type"] == "project_completion_summary")["reply_to"] == quote
    assert all(r.get("row_address") != quote["address"] for r in history)
    # A notice shared with the work's Project has one live frame for both rooms.
    message_bus.send_with_budget(1, "Inventory update", task_id="count", system_type=MAIN_NOTICE_TYPE)
    assert other in echoes[-1]["also_rooms"] and "reply_to" not in echoes[-1]
    assert "reply_to" not in next(r for r in _history(tmp_path, other) if r["system_type"] == MAIN_NOTICE_TYPE)


@pytest.mark.parametrize("source", ["presence", "skill_repair", "absent"])
def test_a_frozen_main_completion_cannot_make_an_unverified_project_quote_visible(tmp_path, monkeypatch, source):
    from ouroboros.projects_registry import create_project

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    room = create_project(tmp_path, "garden")["chat_id"]
    original = {"direction": "in", "chat_id": room, "ts": "2024-04-08T13:45:00Z", "text": "Seed count"}
    if source != "absent":
        original["source"] = source
        dialogue_store.append(tmp_path, "chat", original)
    quote = reference_from_row(original)
    message_bus.send_with_budget(1, "Count complete", role="system", task_id="count", system_type="project_completion_summary",
                                 progress_meta={"reply_to": quote})
    assert "reply_to" not in dialogue_test.chat_rows(tmp_path)[-1] and "reply_to" not in echoes[-1]
    # History from an older writer may already carry the preview.
    dialogue_store.append(tmp_path, "chat", {"direction": "system", "type": "project_completion_summary", "chat_id": 1,
                          "task_id": "previous", "ts": "2024-04-09T13:45:00Z", "text": "Earlier count", "reply_to": quote})
    replay = next(row for row in _history(tmp_path) if row["task_id"] == "previous")
    assert "reply_to" not in replay


def _reply_to(bridge, stored, cmid):
    address = format_address(row_address(stored))
    bridge.handle_web_message("About this", client_message_id=cmid, task_metadata={"reply_to": address})
    return address, bridge._inbox.get_nowait()["task_metadata"]["reply_reference"]


def test_reply_to_authored_completion_mirror_quotes_the_shown_answer(tmp_path, monkeypatch):
    from ouroboros.context import build_runtime_section
    from tests._context_shared import _make_health_env
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    answer = "ACTUAL AUTHORED ANSWER:\n\n the exact result the owner sees. " + "more " * 60
    stored = message_bus.log_chat("system", 1, 0, "Project / Done / open room", ts="2026-10-17T08:00:00Z",
                                  record_type="project_completion_summary",
                                  message_meta={"completion_answer": answer, "project_id": "project-a"})
    address, reference = _reply_to(bridge, stored, "mirror-reply")
    assert reference["address"] == address and reference["kind"] == "completion"
    assert reference["role"] == "assistant" and reference["author_label"] == "Ouroboros"
    assert reference["snippet"] == " ".join(answer.split())[:159].rstrip() + "…" and reference["chars"] == len(answer)
    assert echoes[-1]["reply_to"] == reference
    replay = next(r for r in _history(tmp_path) if r.get("client_message_id") == "mirror-reply")
    assert replay["reply_to"] == reference
    task = {"id": "reply", "type": "task", "metadata": {"reply_reference": reference}}
    model = json.loads(build_runtime_section(_make_health_env(tmp_path / "env"), task).split("\n\n", 1)[1])
    assert model["reply_to"]["snippet"] == reference["snippet"]
    raw = dialogue_test.chat_rows(tmp_path)[0]  # the stored row, as the dialogue store keeps it
    assert raw["text"] == "Project / Done / open room" and raw["direction"] == "system"
    assert raw["completion_answer"] == answer and format_address(row_address(raw)) == address


@pytest.mark.parametrize("direction,record_type,meta,role", [
    ("system", "task_error", {}, "system"),
    ("system", "project_completion_summary", {"completion_answer": "", "project_id": "project-a"}, "system"),
    ("out", "", {}, "assistant"),
])
def test_reply_to_other_rows_keeps_their_own_text_and_voice(tmp_path, monkeypatch, direction, record_type, meta, role):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    stored = message_bus.log_chat(direction, 1, 0, "Row  text\nas stored", ts="2026-10-17T08:00:00Z",
                                  record_type=record_type, message_meta=meta)
    address, reference = _reply_to(bridge, stored, "plain-reply")
    assert reference["address"] == address and reference["snippet"] == "Row text as stored"
    assert reference["role"] == role and echoes[-1]["reply_to"] == reference
    # Only a Project lifecycle row speaks with its authored answer, as in Main.
    other = reference_from_row({**stored, "direction": "system", "type": "task_error", "completion_answer": "Hidden"})
    assert other["role"] == "system" and other["snippet"] == "Row text as stored"


def test_an_outgoing_rows_own_name_travels_beside_the_owner_correlation(tmp_path, monkeypatch):
    """A relay is saved under its own name; its live frame carries that name, never the owner's id."""
    from types import SimpleNamespace

    from ouroboros.chat_references import projection_fields
    from ouroboros.owner_relay import publish_relay, write_relay
    from ouroboros.task_finalization import host_operation_reply_kwargs
    entry, _ = write_relay(tmp_path, "receiver", relay_id="r1", note="Same words", quotes=[], chat_id=1,
                           teller_task_id="teller")
    sent = []
    publish_relay(tmp_path, "receiver", entry, SimpleNamespace(send_message=lambda chat_id, text, **kw: sent.append(kw)))
    assert sent[0]["progress_meta"]["row_client_message_id"] == entry["msg_id"]
    assert "client_message_id" not in sent[0]["progress_meta"], "the owner's correlation field stays unused"
    stored = next(found.row for found in dialogue_store.rows_by_client_id(tmp_path, entry["msg_id"]))
    assert projection_fields(stored)["row_client_message_id"] == entry["msg_id"], "history projects the same name"
    # The owner's own row keeps its id in client_message_id alone; a row saved without a name gains nothing.
    assert "row_client_message_id" not in projection_fields({**stored, "direction": "in"})
    assert "row_client_message_id" not in projection_fields({**stored, "client_message_id": ""})
    assert "row_client_message_id" not in projection_fields({**stored, "is_progress": True}), "progress merges meta"
    # A host reply names the owner's submission on its live frame; its saved row has no name of its own.
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    message_bus.send_with_budget(1, "Restarting", role="system", system_type="command_reply",
                                 **host_operation_reply_kwargs("web", "", "owner-cmid"))
    frame = next(echo for echo in echoes if echo.get("content") == "Restarting")
    assert frame["client_message_id"] == "owner-cmid" and frame["row_address"]
    assert "row_client_message_id" not in frame, "the owner's correlation never becomes the reply's own name"


def test_a_call_opening_row_names_itself_on_its_live_echo():
    """The call-open row is ingress the host wrote: its live echo must carry its own name, or the chat cannot fold
    it into the call card and the owner sees the opening as a System bubble beside every call (27.10)."""
    from ouroboros.chat_references import projection_fields
    opening = {"direction": "in", "role": "system", "system_type": "call_open", "chat_id": 1, "ts": "t",
               "client_message_id": "call:c1:open", "host_fact": {"kind": "call_open", "call_id": "c1"}, "text": "Cue"}
    assert projection_fields(opening)["row_client_message_id"] == "call:c1:open"
    owner = {**opening, "role": "user", "system_type": ""}
    owner.pop("host_fact")
    assert "row_client_message_id" not in projection_fields(owner), "the owner's own row keeps its id alone"


# --- the owner bot's Reply: one room decision on its accepted row, one physical address for its own turn ------------

def seed_bundled_bridge(root):
    """The bundled Telegram bridge as a seeded install holds it (``skills/native/telegram``, its seed marker): only
    that skill's Replies have their room decided (``chat_references.bundled_bridge``)."""
    from pathlib import Path

    target = Path(root) / "skills" / "native" / "telegram"
    target.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    (target / "SKILL.md").write_text((repo / "skills" / "telegram" / "SKILL.md").read_text(encoding="utf-8"),
                                     encoding="utf-8")
    (target / ".seed-origin").write_text("seeded_from=test\n", encoding="utf-8")
    return target


def _reply_world(root):
    """A Project whose converted root has an admitted helper, as the room rule reads them, beside the bundled bridge."""
    from ouroboros import projects_registry as pr
    from ouroboros.task_results import write_task_result

    seed_bundled_bridge(root)

    project = pr.create_project(root, "reply-room", name="Reply room")
    write_task_result(root, "project-root", "running", root_task_id="project-root", chat_id=1)
    pr.bind_task_to_project(root, "project-root", project["id"], origin={"absent": "system"})
    write_task_result(root, "helper", "running", delegation_role="subagent", parent_task_id="project-root",
                      root_task_id="project-root", chat_id=1)
    return project


def _bot_message(bridge, root, address, *, mid, text="About this", own=False):
    """The owner bot's message 'mid' as /chat/inject accepts it from the bundled bridge: a Reply to a bot message
    names his own message in its transport (``owner_message_id``), a Reply to his own words does not."""
    facts = {"owner_message_id": mid} if address and not own else {}
    row, _ = message_bus.accept_local_message(
        bridge, root, text, chat_id=42, user_id=42, client_message_id=f"tg:123:42:{mid}", source="skill:telegram",
        transport={"kind": "telegram", "conversation_id": "42", "bot_id": "123", **facts},
        **({"task_metadata": {"reply_to": address}} if address else {}))
    return row


def _drain(bridge):
    return [bridge._inbox.get_nowait() for _ in range(bridge._inbox.qsize())]


@pytest.mark.parametrize("target, expected", [
    ("own_words", "ordinary"), ("own_words_unread", "undecided"), ("answer_here", "ordinary"), ("web_main", "ordinary"),
    ("project_card", "project"), ("helper_notice", "project"), ("project_question", "ordinary"),
    # Unknown is not confirmed: a target the store cannot find now stays this chat's ordinary turn, never refused,
    # and no room is recorded for it (``undecided``): nothing claims more than the host could name.
    ("missing_card", "undecided"), ("unresolved_here", "undecided"), ("unresolved_elsewhere", "undecided"),
    ("retired_project", "refused"),
])
def test_the_owner_bots_reply_takes_its_room_once_on_its_accepted_row(tmp_path, monkeypatch, target, expected):
    from ouroboros import projects_registry as pr
    from ouroboros.task_results import write_task_result

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)
    room = project["chat_id"]
    if target == "own_words":
        address = format_address(row_address(_bot_message(bridge, tmp_path, "", mid=10, text="My earlier words")))
    elif target == "answer_here":
        write_task_result(tmp_path, "telegram-turn", "completed", chat_id=42)
        address = format_address(row_address(message_bus.log_chat("out", 42, 0, "An answer here", task_id="telegram-turn")))
    elif target == "web_main":
        address = format_address(row_address(message_bus.log_chat("out", 1, 0, "A web answer")))
    elif target in ("project_card", "retired_project"):
        address = "task:project-root"
    elif target == "helper_notice":
        address = format_address(row_address(message_bus.log_chat(
            "system", room, 0, "The helper stopped", task_id="helper", record_type="terminal_incident",
            message_meta={"role": "system", "system_type": "terminal_incident"})))
    elif target == "project_question":  # an earlier part of a long Project question: its card is the answer's door
        address = format_address(row_address(message_bus.log_chat(
            "out", room, 0, "Which tables?", task_id="project-root", record_type="quiz",
            client_message_id="quiz_publication:project-root:q1", quiz={"quiz_id": "q1", "state": "open"})))
    elif target == "missing_card":
        address = "task:gone"
    else:  # a row the store cannot read now: his own words stay his, a bot message is never placed by its numbers
        address = f"row:{room if target == 'unresolved_elsewhere' else 42}@2026-11-20T00:00:00Z#000000000000"
    if target == "retired_project":
        pr.begin_project_deletion(tmp_path, project["id"])
        pr.complete_project_deletion(tmp_path, project["id"])
    _drain(bridge)
    reference = _bot_message(bridge, tmp_path, address, mid=50, own=target.startswith("own_words"))["reply_to"]
    assert reference["address"] == address
    if expected == "undecided":
        assert "telegram_room_chat_id" not in reference, "an ordinary turn; no room the host could not name"
    else:  # 0: a quote of this conversation or Main, never a destination; decided once, read alike everywhere
        assert reference["telegram_room_chat_id"] == {"ordinary": 0, "project": room, "refused": None}[expected]
    [queued] = _drain(bridge)
    assert queued["task_metadata"]["reply_reference"] == reference, "the dispatch reads the decision the row keeps"
    if expected == "project":
        pr.begin_project_deletion(tmp_path, project["id"])  # a later change of the room is the dispatch's to see
    again, rejoined = message_bus.accept_local_message(
        bridge, tmp_path, "About this", chat_id=42, user_id=42, client_message_id="tg:123:42:50",
        source="skill:telegram", transport={"kind": "telegram", "conversation_id": "42"},
        task_metadata={"reply_to": address})
    assert rejoined and again["reply_to"] == reference, "a redelivery rejoins; the decision is never taken again"


def test_only_the_accepted_exact_reply_gives_its_turn_a_physical_address(tmp_path, monkeypatch):
    from ouroboros.chat_references import (exact_reply_transport, origin_reply, proven_address, telegram_root_address,
                                           with_turn_address)
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import write_task_result

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)
    exact = _bot_message(bridge, tmp_path, "task:helper", mid=50)
    plain = _bot_message(bridge, tmp_path, "", mid=51, text="Just words")
    address = {"kind": "telegram", "conversation_id": "42", "bot_id": "123"}
    assert exact_reply_transport(exact) == {**address, "host_address": "reply", "exact_reply": {"owner_message_id": 50}}
    assert exact_reply_transport(plain) == {}, "an ordinary message keeps the room's binding, as before"
    assert exact_reply_transport({**exact, "source": "skill:other"}) == {}, "another connector's fields grant nothing"
    ref = build_owner_message_ref(chat_id=42, client_message_id="tg:123:42:50", ts=exact["ts"], text=exact["text"])
    assert origin_reply(tmp_path, "exact-turn", {"origin_message_ref": ref})["address"] == \
        format_address(row_address(exact))
    # The turn carries the address its acceptance gave it; a sender's value under that key never stands.
    assert with_turn_address({"project_id": "p"}, exact) == {"project_id": "p", "exact_reply_transport": {
        **address, "host_address": "reply", "exact_reply": {"owner_message_id": 50}}}
    assert with_turn_address({"exact_reply_transport": {"kind": "telegram"}}, plain) == {}
    # A helper of a tree that began there learns where the phone is: kind, chat and bot, never his message's anchor.
    write_task_result(tmp_path, "exact-turn", "running", chat_id=project["chat_id"], origin_message_ref=ref)
    assert telegram_root_address(tmp_path, "exact-turn") == {**address, "host_address": "root"}
    write_task_result(tmp_path, "web-turn", "running", chat_id=project["chat_id"])
    assert telegram_root_address(tmp_path, "web-turn") == {}
    # A row accepted before the bridge sent its bot proves one only by its bot-qualified id, in exact agreement.
    before = {"kind": "telegram", "conversation_id": "42", "sender_label": "Telegram (Owner)"}
    for client_message_id, proven in (("tg:123:42:52", address), ("tg:42:53", {}), ("tg:123:43:54", {})):
        assert proven_address({**plain, "client_message_id": client_message_id, "transport": before}) == proven
    assert proven_address({**plain, "transport": {**plain["transport"], "bot_id": "999"}}) == {}, "its id names 123"
    assert proven_address({**plain, "chat_id": 1}) == {}, "accepted in another chat than the one it names"


def _route(bridge, root, monkeypatch, row, *, open_turn="", live=None, chat=42, real_metadata=False, real_ack=False,
           source_channel="skill:telegram"):
    """The accepted row through the owner router, as server.py hands it over: (turns, follow-ups, receipts).
    ``live``: the live roots each chat may address (``addressable_roots``), by chat id."""
    from types import SimpleNamespace

    from ouroboros import server_owner_routing as routing
    from ouroboros.project_dialogue import build_owner_message_ref

    [queued] = _drain(bridge)
    started, followups, receipts = [], [], []

    def running_work(*_args, **_kwargs):
        if source_channel == "web":
            return ""
        pytest.fail("the owner bot's words are never a running Project work's input")

    monkeypatch.setattr(routing, "_route_project_chat_to_running_task", running_work)
    monkeypatch.setattr(routing, "_route_main_followup",
                        lambda _ctx, chat, *_args, **_kwargs: followups.append(chat) or open_turn)
    monkeypatch.setattr(routing, "addressable_roots", lambda _ctx: SimpleNamespace(
        rows=lambda chat=None: [dict(root) for root in (live or {}).get(chat, [])], gap=None))
    if not real_metadata:
        monkeypatch.setattr(routing, "_decision_turn_metadata", lambda _ctx, _room, _cmid, meta, _roots, **_kw: meta)
    if real_ack:
        monkeypatch.setattr(bridge, "send_routing_ack", lambda chat, **fields: receipts.append({"chat_id": chat, **fields}))
    else:
        monkeypatch.setattr(routing, "_record_routing_receipt", lambda _bridge, _ctx, **fields: receipts.append(fields))
    monkeypatch.setattr("ouroboros.owner_relay.carry_row_at_receipt", lambda *_args, **_kwargs: None)
    source = build_owner_message_ref(chat_id=chat, client_message_id=row["client_message_id"], ts=row["ts"],
                                     text=row["text"])
    ctx = SimpleNamespace(DRIVE_ROOT=root, handle_chat_direct=lambda *args, **kwargs: started.append((args, kwargs)))
    routing._route_owner_message(bridge, ctx, {
        "chat_id": chat, "text": row["text"], "log_text": row["text"], "image_caption": "", "image_data": None,
        "client_message_id": row["client_message_id"], "task_constraint": None, "origin_message_ref": source,
        "task_metadata": {**(queued.get("task_metadata") or {}), "_host_operation": True},
        "source": source_channel, "received_at": row["ts"], "accepted_source_row": row})
    return started, followups, receipts


@pytest.mark.parametrize("conversation", ["new", "open"])
def test_plain_words_and_a_reply_to_words_stay_one_conversation(tmp_path, monkeypatch, conversation):
    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    _reply_world(tmp_path)
    earlier = _bot_message(bridge, tmp_path, "", mid=10, text="Check the parser")
    _drain(bridge)
    open_turn = "turn-open" if conversation == "open" else ""
    outcomes = {}
    for kind, address, mid in (("plain", "", 11), ("reply", format_address(row_address(earlier)), 12)):
        row = _bot_message(bridge, tmp_path, address, mid=mid, text="and on Windows too", own=True)
        if address:
            assert row["reply_to"]["address"] == address and row["reply_to"]["role"] == "user"
            assert row["reply_to"]["telegram_room_chat_id"] == 0, "the quote rides along; it routes nothing"
        started, followups, receipts = _route(bridge, tmp_path, monkeypatch, row, open_turn=open_turn)
        outcomes[kind] = ([(args[0], kwargs["task_metadata"].get("project_id", "")) for args, kwargs in started],
                          followups, [receipt["action"] for receipt in receipts])
    turns = [] if open_turn else [(42, "")]
    saved = ["mailbox_delivery"] if open_turn else []
    assert outcomes["plain"] == outcomes["reply"] == (turns, [42], saved), "the same physical conversation, alike"


@pytest.mark.parametrize("target", ["project", "retired"])
def test_a_confirmed_project_reply_opens_its_own_turn_there_and_an_unavailable_one_is_refused(tmp_path, monkeypatch,
                                                                                            target):
    from ouroboros import projects_registry as pr

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)
    row = _bot_message(bridge, tmp_path, "task:helper", mid=50, text="Why did it stop?")
    assert row["reply_to"]["telegram_room_chat_id"] == project["chat_id"]
    if target == "retired":  # retired after its acceptance: the dispatch reads its room again
        pr.begin_project_deletion(tmp_path, project["id"])
    live = {project["chat_id"]: [{"task_id": "project-root"}], 42: [{"task_id": "telegram-turn"}]}
    started, followups, receipts = _route(bridge, tmp_path, monkeypatch, row, open_turn="turn-open", live=live)
    assert followups == [], "never this chat's running turn, nor the helper it asked about"
    carried = {tid: [entry for entry in _mail(tmp_path, tid) if entry.get("host_carried")]
               for tid in ("project-root", "telegram-turn")}
    if target == "retired":
        assert started == [] and [(r["chat_id"], r["status"], r["reason"]) for r in receipts] == [
            (42, "project_unavailable", "reply_target_unavailable")]
        assert carried == {"project-root": [], "telegram-turn": []}, "no Project, no context for anyone"
        return
    [(args, kwargs)] = started
    assert args[0] == project["chat_id"] and kwargs["task_metadata"]["project_id"] == project["id"]
    assert kwargs["task_metadata"]["origin_message_ref"]["client_message_id"] == "tg:123:42:50"
    assert kwargs["task_metadata"]["reply_reference"]["telegram_room_chat_id"] == project["chat_id"]
    assert receipts == []
    # The Project's other live root hears it as context, read where he wrote it; his private chat's turn does not.
    [context] = carried["project-root"]
    assert carried["telegram-turn"] == [] and context["host_carried"] == {"went_to": "routed:head"}
    assert [(row["client_message_id"], row["chat_id"]) for row in context["owner_rows"]] == [("tg:123:42:50", 42)]
    assert context["relay"]["quotes"][0]["source"]["chat_id"] == 42


def _mail(root, task_id):
    """The entries a work's mailbox holds."""
    from ouroboros.owner_mailbox import _mailbox_path, mailbox_lines

    path = _mailbox_path(root, task_id)
    return [json.loads(line) for line in mailbox_lines(path.read_text(encoding="utf-8"))] if path.exists() else []


def test_a_reply_accepted_before_this_version_keeps_the_ordinary_path(tmp_path, monkeypatch):
    """Work begun before the update dispatches as it was accepted: a reference without the decision routes nothing."""
    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    _reply_world(tmp_path)
    row = _bot_message(bridge, tmp_path, "task:helper", mid=50)
    [queued] = _drain(bridge)
    reference = {key: value for key, value in queued["task_metadata"]["reply_reference"].items()
                 if key != "telegram_room_chat_id"}
    bridge._inbox.put({**queued, "task_metadata": {**queued["task_metadata"], "reply_reference": reference}})
    started, followups, receipts = _route(bridge, tmp_path, monkeypatch, row)
    assert followups == [42] and receipts == []
    assert [(args[0], kwargs["task_metadata"].get("project_id", "")) for args, kwargs in started] == [(42, "")]


@pytest.mark.parametrize("ingress", ["bundled", "botless", "same_name_elsewhere", "not_seeded"])
def test_only_the_bundled_bridge_under_a_bot_qualified_id_has_its_reply_room_decided(tmp_path, monkeypatch, ingress):
    """A skill that merely bears the name, or an id that proves no bot, keeps the ordinary reference."""
    import shutil

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)
    if ingress == "same_name_elsewhere":  # another source's skill of that name beside the seeded one
        shutil.copytree(tmp_path / "skills" / "native" / "telegram", tmp_path / "skills" / "external" / "telegram")
    if ingress == "not_seeded":  # only a copy from another source holds the name
        shutil.move(str(tmp_path / "skills" / "native" / "telegram"), str(tmp_path / "skills" / "external" / "telegram"))
    client_message_id = "tg:42:50" if ingress == "botless" else "tg:123:42:50"
    row, _ = message_bus.accept_local_message(
        bridge, tmp_path, "About this", chat_id=42, user_id=42, client_message_id=client_message_id,
        source="skill:telegram", transport={"kind": "telegram", "conversation_id": "42", "owner_message_id": 50},
        task_metadata={"reply_to": "task:helper"})
    if ingress == "bundled":
        assert row["reply_to"]["telegram_room_chat_id"] == project["chat_id"]
    else:
        assert row["reply_to"]["address"] == "task:helper" and "telegram_room_chat_id" not in row["reply_to"]


def test_a_busy_store_during_the_room_decision_accepts_nothing_and_the_next_try_decides_afresh(tmp_path, monkeypatch):
    from ouroboros import chat_references
    from ouroboros.store import StoreBusy

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)
    decide = chat_references.telegram_reply_room
    tries = []

    def busy_once(*args, **kwargs):
        tries.append(1)
        if len(tries) == 1:
            raise StoreBusy("room facts busy")
        return decide(*args, **kwargs)

    monkeypatch.setattr(chat_references, "telegram_reply_room", busy_once)
    from supervisor.message_ingress import NotAccepted

    with pytest.raises(NotAccepted) as refused:
        _bot_message(bridge, tmp_path, "task:helper", mid=50)
    assert isinstance(refused.value.__cause__, StoreBusy)
    assert message_bus.accepted_chat_message(tmp_path, 42, "tg:123:42:50") is None, "nothing accepted, nothing refused"
    assert _bot_message(bridge, tmp_path, "task:helper", mid=50)["reply_to"]["telegram_room_chat_id"] == project["chat_id"]


def test_a_telegram_chat_bearing_a_projects_number_stays_its_own_conversation(tmp_path, monkeypatch):
    """The private chat's physical id equals another Project's local id: the owner's words, ordinary Reply and an
    exact Reply's row never join that Project; the exact Reply joins the room its turn ran in, in memory too."""
    from ouroboros import projects_registry as pr
    from ouroboros.memory_inventory import row_meta
    from ouroboros.projects_registry import RoomResolver
    from ouroboros.room_membership import row_rooms

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)
    twin = pr.create_project(tmp_path, "number-twin", name="Number twin")
    chat = twin["chat_id"]  # the Telegram chat's id, by coincidence

    def says(mid, text, reply=""):
        row, _ = message_bus.accept_local_message(
            bridge, tmp_path, text, chat_id=chat, user_id=chat, client_message_id=f"tg:123:{chat}:{mid}",
            source="skill:telegram", transport={"kind": "telegram", "conversation_id": str(chat), "bot_id": "123",
                                                **({"owner_message_id": mid} if reply else {})},
            **({"task_metadata": {"reply_to": reply}} if reply else {}))
        return row

    plain = says(10, "Plan the next step")
    own = says(11, "and on Windows too", reply=format_address(row_address(plain)))
    answer = message_bus.log_chat("out", chat, 0, "An answer in this chat", task_id="telegram-turn")
    to_answer = says(12, "Why?", reply=format_address(row_address(answer)))
    exact = says(13, "Why did it stop?", reply="task:helper")
    assert own["reply_to"]["telegram_room_chat_id"] == to_answer["reply_to"]["telegram_room_chat_id"] == 0
    assert exact["reply_to"]["telegram_room_chat_id"] == project["chat_id"]
    rooms, projects = RoomResolver(tmp_path), {project["chat_id"], chat}
    for row in (plain, own, to_answer):
        assert row_rooms(chat, row, projects, rooms) == {1} and row_rooms(chat, row_meta(row), projects, rooms) == {1}
    assert row_rooms(chat, exact, projects, rooms) == row_rooms(chat, row_meta(exact), projects, rooms) == {
        1, project["chat_id"]}, "its logical room, never the number's"
    _drain(bridge)
    started, followups, _receipts = _route(bridge, tmp_path, monkeypatch, says(14, "Next?"), chat=chat)
    assert followups == [chat] and [(args[0], kwargs["task_metadata"].get("project_id", ""))
                                    for args, kwargs in started] == [(chat, "")], "an ordinary turn, not the twin's"


def test_a_refused_messages_end_survives_its_handover_record(tmp_path, monkeypatch):
    """The router may refuse while the handover is still open: ``complete_dispatch`` shrinks the record to its
    dispatcher and process and keeps the refusal; a row nobody refused keeps no such field."""
    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)

    def refuse_during_handover(_text, **message):
        assert dialogue_store.refuse_dispatch(tmp_path, 42, message["client_message_id"], "Not started: gone")

    message_bus.accept_local_message(bridge, tmp_path, "Is it still running?", chat_id=42, user_id=42,
                                     client_message_id="tg:123:42:62", source="skill:telegram",
                                     dispatch=refuse_during_handover)
    facts = dialogue_store.dispatch_facts(tmp_path, 42, "tg:123:42:62")
    assert (facts["state"], facts["refused"]) == ("completed", "Not started: gone")
    message_bus.accept_local_message(bridge, tmp_path, "Plain", chat_id=42, user_id=42,
                                     client_message_id="tg:123:42:63", source="skill:telegram")
    assert "refused" not in dialogue_store.dispatch_facts(tmp_path, 42, "tg:123:42:63")


def test_an_exact_replys_turn_keeps_its_whole_address_while_its_source_row_cannot_be_read(tmp_path, monkeypatch):
    """The turn took its address, strict anchor included, at its start: its answer, media, progress and a receipt
    naming no task go there with the store unreadable; a turn that carries none keeps the room's binding."""
    from ouroboros.chat_references import answer_reference, turn_reply_transport
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.store import StoreBusy
    from ouroboros.task_finalization import host_operation_reply_kwargs
    from ouroboros.task_results import write_task_result

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)
    exact = _bot_message(bridge, tmp_path, "task:helper", mid=50)
    [queued] = _drain(bridge)
    begun = queued["task_metadata"]  # what the router hands the turn it starts
    carried = begun["exact_reply_transport"]
    assert carried == {"kind": "telegram", "conversation_id": "42", "bot_id": "123", "host_address": "reply",
                       "exact_reply": {"owner_message_id": 50}}
    ref = build_owner_message_ref(chat_id=42, client_message_id="tg:123:42:50", ts=exact["ts"], text=exact["text"])
    write_task_result(tmp_path, "exact-turn", "running", chat_id=project["chat_id"], origin_message_ref=ref,
                      metadata=begun)
    write_task_result(tmp_path, "plain-turn", "running", chat_id=project["chat_id"], origin_message_ref=ref)

    def busy(*_args, **_kwargs):
        raise StoreBusy("busy")

    monkeypatch.setattr("ouroboros.project_dialogue.resolve_owner_message_source", busy)
    assert turn_reply_transport(tmp_path, "exact-turn") == carried, "its anchor too, no row read"
    assert "exact_reply" not in turn_reply_transport(tmp_path, "exact-turn", anchor=False), "progress: no anchor"
    assert turn_reply_transport(tmp_path, "plain-turn") == {}, "no carried address: the room's binding, as before"
    meta = {}
    answer_reference(tmp_path, meta, "exact-turn", is_progress=False)
    assert meta == {"transport": carried}, "the quote waits for the store; the address does not"
    receipt = host_operation_reply_kwargs(ref, "failed", "tg:123:42:50", begun["exact_reply_transport"])
    assert receipt["progress_meta"]["transport"] == carried, "a Restart receipt naming no task is told there too"


def test_a_late_telegram_answer_keeps_its_rooms_binding_and_answers_where_it_came_from(tmp_path, monkeypatch):
    """A question shown in the private chat by the mirror, not begun by an exact Reply: its late answer is delivered
    into the question's room, leaves that room's binding as it was, and its turn answers through the bot and chat
    the answer came from. A web answer, or one recorded before the surface was kept, claims no surface."""
    from ouroboros.chat_references import keeps_binding, turn_reply_transport
    from ouroboros.gateway.task_decision import _answer_surface, deliver_as_owner_message
    from ouroboros.owner_quiz import quiz_states, record_answered, record_asked
    from supervisor import queue

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    project = _reply_world(tmp_path)
    room = project["chat_id"]
    surface = {"kind": "telegram", "conversation_id": "42", "bot_id": "123"}
    for quiz, source in (("q-tg", "skill:telegram"), ("q-web", "web"), ("q-old", "skill:telegram")):
        record_asked(tmp_path, "project-root", quiz_id=quiz, question="Orders or users?", options=["Orders", "Users"],
                     assumption="orders meanwhile")
        body = {"surface": surface} if quiz != "q-old" else {}  # an older bridge relayed none
        record_answered(tmp_path, "project-root", quiz_id=quiz, option_index=0, request_id=f"tg:123:{quiz}",
                        answer_source=source, answer_surface=_answer_surface(tmp_path, body, source))
    assert _answer_surface(tmp_path, {"surface": surface}, "web") is None
    slack = {"kind": "slack", "conversation_id": "C1"}
    turns = {}
    for quiz, source in (("q-tg", "skill:telegram"), ("q-web", "web"), ("q-old", "skill:telegram")):
        bridge._chat_transports[room] = dict(slack)
        block = quiz_states(tmp_path, "project-root")[quiz]
        late = {"late_answer": {"task_id": "project-root", "quiz_id": quiz,
                                "answer_snapshot": {"task_id": "project-root", "quiz": {**block, "answered_after_terminal": True}}}}
        assert deliver_as_owner_message(tmp_path, room, "Orders", f"quiz_late_answer:project-root:{quiz}", source=source,
                                        task_metadata=late)[0]
        [*_, queued] = _drain(bridge)
        assert keeps_binding(queued) is (source == "skill:telegram")
        bridge.activate_update_transport(queued)
        assert bridge.chat_transport(room) == (slack if source == "skill:telegram" else {}), "web keeps its old rule"
        queue.RUNNING[f"late-{quiz}"] = {"task": {"id": f"late-{quiz}", "metadata": dict(queued["task_metadata"])}}
        turns[quiz] = turn_reply_transport(tmp_path, f"late-{quiz}")
    assert turns == {"q-tg": {**surface, "host_address": "reply"}, "q-web": {}, "q-old": {}}


def test_a_proven_project_work_bearing_the_private_chats_number_stays_that_projects(tmp_path, monkeypatch):
    """The Project's own visible card, whose room number equals the private chat's id by chance: a Reply to it is
    that Project's turn, and its row is that Project's; ordinary words there stay this conversation's."""
    from ouroboros import projects_registry as pr
    from ouroboros.projects_registry import RoomResolver
    from ouroboros.room_membership import row_rooms
    from ouroboros.task_results import write_task_result

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    _reply_world(tmp_path)
    twin = pr.create_project(tmp_path, "number-twin", name="Number twin")
    chat = twin["chat_id"]
    write_task_result(tmp_path, "twin-root", "running", root_task_id="twin-root", chat_id=1)
    pr.bind_task_to_project(tmp_path, "twin-root", twin["id"], origin={"absent": "system"})

    def says(mid, text, reply=""):
        row, _ = message_bus.accept_local_message(
            bridge, tmp_path, text, chat_id=chat, user_id=chat, client_message_id=f"tg:123:{chat}:{mid}",
            source="skill:telegram", transport={"kind": "telegram", "conversation_id": str(chat), "bot_id": "123",
                                                **({"owner_message_id": mid} if reply else {})},
            **({"task_metadata": {"reply_to": reply}} if reply else {}))
        return row

    card = says(20, "About the twin's work", reply="task:twin-root")
    assert card["reply_to"]["telegram_room_chat_id"] == chat, "a proven Project work keeps its own room"
    rooms = RoomResolver(tmp_path)
    assert row_rooms(chat, card, {chat}, rooms) == {1, chat}
    started, followups, receipts = _route(bridge, tmp_path, monkeypatch, card, chat=chat)
    assert followups == [] and receipts == [] and [(args[0], kwargs["task_metadata"]["project_id"])
                                                    for args, kwargs in started] == [(chat, twin["id"])]
    plain = says(21, "Plan the next step")
    assert row_rooms(chat, plain, {chat}, rooms) == {1}
    started, followups, _receipts = _route(bridge, tmp_path, monkeypatch, plain, chat=chat)
    assert followups == [chat] and [kwargs["task_metadata"].get("project_id", "") for _args, kwargs in started] == [""]


def test_d6_web_project_and_physical_telegram_keep_distinct_lanes_at_the_same_number(tmp_path, monkeypatch):
    from supervisor import state

    bridge, _ = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)
    chat = project["chat_id"]
    monkeypatch.setattr(state, "load_state", lambda: {"owner_external_chat_id": chat})
    for mid, address, lane in ((20, "task:helper", "project"), (21, "", "main")):
        row, _ = message_bus.accept_local_message(
            bridge, tmp_path, "Check the report", chat_id=chat, user_id=chat, client_message_id=f"tg:123:{chat}:{mid}",
            source="skill:telegram", transport={"kind": "telegram", "conversation_id": str(chat), "bot_id": "123",
                                                "owner_message_id": mid}, task_metadata={"reply_to": address})
        started, followups, receipts = _route(bridge, tmp_path, monkeypatch, row, chat=chat, real_metadata=True)
        [(_, kwargs)] = started
        meta = kwargs["task_metadata"]
        assert meta["routing_contract"]["source_lane"] == lane
        assert (meta.get("project_id") or "") == (project["id"] if address else "")
        assert followups == ([] if address else [chat]) and receipts == []
    row, _ = message_bus.accept_local_message(bridge, tmp_path, "From the Project tab", chat_id=chat, user_id=chat,
                                              client_message_id="web-project-reply", source="web")
    started, followups, _ = _route(bridge, tmp_path, monkeypatch, row, chat=chat, real_metadata=True,
                                   source_channel="web")
    [(_, kwargs)] = started
    meta = kwargs["task_metadata"]
    assert followups == [] and meta["project_id"] == project["id"]
    assert meta["routing_contract"]["source_lane"] == "project"


def test_d4_confirmed_project_context_goes_to_members_only_at_a_colliding_number(tmp_path, monkeypatch):
    bridge, _ = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)  # project-root was admitted in Main and has since been bound here
    chat = project["chat_id"]
    row, _ = message_bus.accept_local_message(
        bridge, tmp_path, "About the report", chat_id=chat, user_id=chat, client_message_id=f"tg:123:{chat}:50",
        source="skill:telegram", transport={"kind": "telegram", "conversation_id": str(chat), "bot_id": "123",
                                           "owner_message_id": 50}, task_metadata={"reply_to": "task:helper"})
    assert row["reply_to"]["telegram_room_chat_id"] == chat
    live = {chat: [{"task_id": "private-turn"}, {"task_id": "project-root"},
                   {"task_id": "admitted-project", "project_id": project["id"]}]}
    started, followups, receipts = _route(bridge, tmp_path, monkeypatch, row, live=live, chat=chat)
    assert len(started) == 1 and not followups and not receipts
    assert _mail(tmp_path, "private-turn") == []
    for tid in ("project-root", "admitted-project"):
        [context] = [entry for entry in _mail(tmp_path, tid) if entry.get("host_carried")]
        assert context["owner_rows"][0]["client_message_id"] == row["client_message_id"]


def test_c5_refusal_transport_survives_an_unreadable_quote(tmp_path, monkeypatch):
    from ouroboros import project_dialogue
    from ouroboros.projects_registry import begin_project_deletion
    from ouroboros.store import StoreBusy

    bridge, _ = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)
    row = _bot_message(bridge, tmp_path, "task:helper", mid=50)
    begin_project_deletion(tmp_path, project["id"])
    def busy(*_args, **_kwargs):
        raise StoreBusy("reading the display quote")
    monkeypatch.setattr(project_dialogue, "resolve_owner_message_source", busy)
    started, followups, receipts = _route(bridge, tmp_path, monkeypatch, row, real_ack=True)
    assert not started and not followups
    [receipt] = receipts
    assert receipt["status"] == "project_unavailable"
    assert receipt["reply_to"] == {}
    assert receipt["transport"] == {"kind": "telegram", "conversation_id": "42", "bot_id": "123",
                                    "host_address": "reply"}


@pytest.mark.parametrize("inactive", [False, True])
def test_c7_connector_command_reply_only_refuses_an_inactive_reservation(tmp_path, monkeypatch, inactive):
    from ouroboros.projects_registry import begin_project_deletion

    bridge, _ = _web_bridge(tmp_path, monkeypatch)
    project = _reply_world(tmp_path)
    chat = project["chat_id"]
    if inactive:
        begin_project_deletion(tmp_path, project["id"])
    else:
        chat += 1
    other = message_bus.log_chat("system", chat, 0, "Status OK", source="skill:other", record_type="command_reply")
    address = format_address(row_address(other))
    row = _bot_message(bridge, tmp_path, address, mid=50, text="What does that mean?")
    assert row["reply_to"]["address"] == address
    assert row["reply_to"]["telegram_room_chat_id"] == (None if inactive else 0)


def test_a3_a4_paged_main_keeps_both_sides_of_physical_telegram_at_a_colliding_number(tmp_path, monkeypatch):
    from ouroboros.chat_chain import room_keys
    from ouroboros.project_dialogue import build_owner_message_ref
    from tests._dialogue import chat_rows

    bridge, _ = _web_bridge(tmp_path, monkeypatch)
    chat = _reply_world(tmp_path)["chat_id"]
    mid = f"tg:123:{chat}:20"
    row, _ = message_bus.accept_local_message(
        bridge, tmp_path, "Physical conversation", chat_id=chat, user_id=chat, client_message_id=mid,
        source="skill:telegram", transport={"kind": "telegram", "conversation_id": str(chat), "bot_id": "123"})
    ref = build_owner_message_ref(chat_id=chat, client_message_id=mid, ts=row["ts"], text=row["text"])
    assert ref
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    message_bus.send_with_budget(chat, "Physical conversation answer", task_id="ordinary-turn",
                                 progress_meta={"origin_message_ref": ref})
    message_bus.log_chat("in", chat, 0, "Written in project", source="web", client_message_id="web-room-row")
    [saved] = [row for row in chat_rows(tmp_path) if row.get("text") == "Physical conversation answer"]
    assert saved["chat_id"] == chat and saved.get("logical_room_chat_id") == 1, saved
    main = _history(tmp_path, 1)
    project = _history(tmp_path, chat)
    main_ids = {row.get("client_message_id") for row in main}
    project_ids = {row.get("client_message_id") for row in project}
    assert mid in main_ids and "web-room-row" not in main_ids
    assert mid not in project_ids and "web-room-row" in project_ids
    [answer] = [row for row in main if row.get("text") == "Physical conversation answer"]
    assert answer["text"] == saved["text"]
    assert all(row.get("text") != "Physical conversation answer" for row in project)
    assert ("telegram_inbound_chat", chat) in room_keys(tmp_path, 1)[0]


def test_a6_appearance_maintenance_builds_the_selective_index_for_existing_rows(tmp_path, monkeypatch):
    from ouroboros import dialogue_store as ds, room_appearances

    bridge, _ = _web_bridge(tmp_path, monkeypatch)
    chat = _reply_world(tmp_path)["chat_id"]
    mid = f"tg:123:{chat}:20"
    message_bus.accept_local_message(
        bridge, tmp_path, "Physical conversation", chat_id=chat, user_id=chat, client_message_id=mid,
        source="skill:telegram", transport={"kind": "telegram", "conversation_id": str(chat), "bot_id": "123"})
    message_bus.log_chat("in", chat, 0, "Written in project", source="web", client_message_id="web-row")
    room_appearances.run(tmp_path, lambda: False)
    assert not room_appearances.owed(tmp_path)
    with ds.transaction(tmp_path, write=True) as conn:
        conn.execute("DROP INDEX dialogue_telegram_inbound")  # synthetic database from an older body
    assert room_appearances.owed(tmp_path)
    assert mid in {row.get("client_message_id") for row in _history(tmp_path, 1)}
    room_appearances.run(tmp_path, lambda: False)
    with ds.transaction(tmp_path, write=False) as conn:
        assert ds.has_telegram_inbound_index(conn)
        records = conn.execute("SELECT client_message_id FROM dialogue INDEXED BY dialogue_telegram_inbound "
                               "WHERE stream='chat' AND chat_id=? AND direction='in' AND client_message_id GLOB 'tg:*'",
                               (chat,)).fetchall()
    assert [record[0] for record in records] == [mid]
    assert not room_appearances.owed(tmp_path)


def test_c4_room_binding_discards_turn_and_root_marks():
    bridge = message_bus.LocalChatBridge()
    bridge.activate_update_transport({"chat_id": 42, "source": "skill:telegram", "transport": {
        "kind": "telegram", "conversation_id": "42", "host_address": "reply", "exact_reply": {},
        "telegram_root": {"kind": "telegram", "host_address": "root"}}})
    assert bridge.chat_transport(42) == {"kind": "telegram", "conversation_id": "42"}


def test_c12_message_bus_keeps_its_reply_binding_dependency_lazy():
    import ast
    import inspect
    import textwrap

    def imports_binding(node):
        return isinstance(node, ast.ImportFrom) and node.module == "ouroboros.chat_references" and any(
            alias.name == "keeps_binding" for alias in node.names)

    assert not any(imports_binding(node) for node in ast.parse(inspect.getsource(message_bus)).body)
    body = ast.parse(textwrap.dedent(inspect.getsource(message_bus.LocalChatBridge.activate_update_transport)))
    assert any(imports_binding(node) for node in ast.walk(body))


def test_a_reply_to_a_system_row_names_system_on_every_surface_never_you(tmp_path, monkeypatch):
    """A call's opening enters by the owner's door (``direction: in``) yet is the host's row (``role: system``): a Reply
    to it signs System on the reference, on the row's live frame and on its replay, and nobody can quote a passage of it."""
    from ouroboros.chat_references import ReplyExcerptNotAuthored, ingress_reply, projection_fields
    from ouroboros.owner_words import reply_target

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    bridge.handle_web_message("Cue: greet the caller.", client_message_id="call:c1:open",
                              task_metadata={"host_fact": {"kind": "call_open", "call_id": "c1"}})
    opening = next(echo for echo in echoes if echo.get("client_message_id") == "call:c1:open")
    stored = dialogue_test.chat_rows(tmp_path)[-1]
    assert (stored["direction"], stored["role"]) == ("in", "system")
    assert opening["role"] == "system" and opening["reply_author_label"] == projection_fields(stored)["reply_author_label"] == "System"
    replay = next(row for row in _history(tmp_path) if row.get("client_message_id") == "call:c1:open")
    assert (replay["role"], replay["reply_author_label"]) == ("system", "System")
    reference = resolve_reply(tmp_path, opening["row_address"])
    assert (reference["role"], reference["author_label"]) == ("system", "System")
    bridge.handle_web_message("About the call.", client_message_id="about-call", task_metadata={"reply_to": opening["row_address"]})
    echo = next(echo for echo in echoes if echo.get("client_message_id") == "about-call")
    assert (echo["reply_to"]["role"], echo["reply_to"]["author_label"]) == ("system", "System")
    assert reply_target(tmp_path, echo["reply_to"])["author"] != "You", "the model reads the host's row, never my human's"
    with pytest.raises(ReplyExcerptNotAuthored):
        ingress_reply(tmp_path, {"reply_to": opening["row_address"], "reply_excerpt": {"text": "Cue", "start_utf16": 0}})
    # The owner's own row, beside it, still signs You: only the host's rows changed.
    own = message_bus.log_chat("in", 1, 1, "Meet at noon.", ts="2026-10-01T00:00:00Z", source="web")
    assert projection_fields(own)["reply_author_label"] == "You" and resolve_reply(
        tmp_path, format_address(row_address(own)))["role"] == "user"


@pytest.mark.parametrize("comment", ["", "  Follow the creek at dawn.  "])
def test_late_quiz_bubble_is_selectable_only_when_it_contains_entered_words(tmp_path, monkeypatch, comment):
    from ouroboros.chat_references import ReplyExcerptNotAuthored, ingress_reply
    from ouroboros.gateway.task_decision import deliver_as_owner_message
    from ouroboros.owner_quiz import late_answer_owner_text
    from ouroboros.owner_words import reply_target

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    block = {"quiz_id": "trail", "state": "answered", "answered_after_terminal": True,
             "answered_index": 0, "options": ["Take the ridge"], "comment": comment}
    answer = {"task_id": "trail-work", "quiz_id": "trail", "answer_snapshot": {"task_id": "trail-work", "quiz": block}}
    text = late_answer_owner_text(block)
    assert deliver_as_owner_message(tmp_path, 1, text, "quiz_late_answer:trail-work:trail",
                                    task_metadata={"late_answer": answer}) == (True, "")
    [stored] = dialogue_test.chat_rows(tmp_path)
    assert stored["late_answer"] == answer and stored["text"] == text
    [live] = echoes
    [history] = [r for r in _history(tmp_path) if r.get("client_message_id") == stored["client_message_id"]]
    assert live["content"] == history["text"] == text  # don't change what a card submission displayed
    assert live["role"] == history["role"] == "user"
    assert live.get("reply_excerpt_disabled", False) == history.get("reply_excerpt_disabled", False) == (not comment)
    address = live["row_address"]
    excerpt = {"text": "creek" if comment else "ridge", "start_utf16": text.index("creek" if comment else "ridge")}
    request = {"reply_to": address, "reply_excerpt": excerpt}
    if comment:
        reply = ingress_reply(tmp_path, request)
        assert reply["role"] == "user" and reply["excerpt"] == excerpt
        assert reply_target(tmp_path, reply)["text"] == comment
    else:
        with pytest.raises(ReplyExcerptNotAuthored):
            ingress_reply(tmp_path, request)
        reply = resolve_reply(tmp_path, address)
        # The bubble is still the owner's: a whole-row Reply names its author as it did; only the preview is the host's account.
        assert reply["role"] == "user" and reply["author_label"] == "You" and reply["kind"] == "message"
        assert "wrote no words" in reply["snippet"]
        target = reply_target(tmp_path, reply)
        assert "wrote no words" in target["text"] and "the text of task trail-work" in target["text"]
        assert target["author"] == "System", "a model reads the account as the host's"


def test_selected_reply_preserves_source_author_citation_and_replay_identity(tmp_path, monkeypatch):
    from ouroboros.chat_references import stored_reply
    from ouroboros.dialogue_provenance import quote_words, relation_text
    from ouroboros.owner_words import reply_target

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    source = "Amber **moon**. See the [map](atlas)."
    original = message_bus.log_chat("out", 1, 0, source, ts="2026-10-01T00:00:00Z")
    address = format_address(row_address(original))
    # A client can claim arbitrary rendered words. Only the addressed source can speak for its author.
    excerpt = {"text": "A different passage.\n  " + "Snow-covered hills. " * 12, "start_utf16": 7}
    meta = {"reply_to": address, "reply_excerpt": {**excerpt, "author": "Pretender", "role": "user"}}
    bridge.handle_web_message("Trace this path.", client_message_id="selected", task_metadata=meta)
    stored = dialogue_test.chat_rows(tmp_path)[-1]
    ref = stored["reply_to"]
    assert ref == {**reference_from_row(original), "excerpt": excerpt}
    delivered = bridge._inbox.get_nowait()["task_metadata"]["reply_reference"]
    assert ref == echoes[-1]["reply_to"] == stored_reply(delivered), "the row and its echo keep the compact reference"
    assert delivered["text"] == source, "delivery carries the source whole, captured before the owner row was saved"
    assert ref == next(r for r in _history(tmp_path) if r.get("client_message_id") == "selected")["reply_to"]
    assert json.dumps(excerpt["text"], ensure_ascii=False) in relation_text(stored)
    assert quote_words(stored) == stored["text"] == "Trace this path."
    assert reply_target(tmp_path, ref)["text"] == source
    monkeypatch.setattr("ouroboros.chat_references.resolve_row", lambda *args: (None, {}))
    bridge.handle_web_message("Trace this path.", client_message_id="selected", task_metadata=meta)
    for replacement in ({}, {"reply_excerpt": {**excerpt, "start_utf16": 2}}, {"reply_excerpt": {**excerpt, "text": "Amber"}}):
        with pytest.raises(ValueError, match="different message"):
            bridge.handle_web_message("Trace this path.", client_message_id="selected",
                                      task_metadata={"reply_to": address, **replacement})
    with pytest.raises(ValueError, match="unavailable"):
        bridge.handle_web_message("Trace this path.", client_message_id="fresh", task_metadata=meta)
    assert bridge._inbox.empty()


@pytest.mark.parametrize("words", ["Meet here", "Meet here\n\n[Attached file: map.pdf]"])
def test_selected_web_attachment_source_keeps_only_the_author_caption(tmp_path, monkeypatch, words):
    from ouroboros.chat_references import ingress_reply
    from ouroboros.owner_words import reply_target

    _web_bridge(tmp_path, monkeypatch)
    tail = "[Attached file: map.pdf]"
    file = {"name": "map.pdf", "mime": "application/pdf", "size": 55}
    source = message_bus.log_chat("in", 1, 1, f"{words}\n\n{tail}", source="web",
                                  message_meta={"attachments": [file]})
    address = format_address(row_address(source))
    selection = {"text": "Meet here", "start_utf16": 0}
    whole = ingress_reply(tmp_path, {"reply_to": address})
    ref = ingress_reply(tmp_path, {"reply_to": address, "reply_excerpt": selection})
    assert ref["snippet"] == " ".join(words.split())
    assert tail in reply_target(tmp_path, whole)["text"]  # whole-row Reply did not change
    target = reply_target(tmp_path, ref)
    assert target["text"] == words and target["attachments"][0]["name"] == "map.pdf"


@pytest.mark.parametrize("excerpt", [None, {}, {"text": " ", "start_utf16": 0},
                                     {"text": "a", "start_utf16": False}, {"text": "a", "start_utf16": -1},
                                     {"text": "a", "start_utf16": 2**53}, {"text": "\ud800", "start_utf16": 1}])
def test_invalid_excerpt_never_downgrades_to_an_ordinary_reply(excerpt):
    from ouroboros.chat_references import reply_identity

    with pytest.raises(ValueError, match="reply_excerpt"):
        reply_identity({"reply_to": "row:1@2026-10-01T00:00:00Z#0123456789ab", "reply_excerpt": excerpt})


def test_excerpt_of_project_answer_keeps_runtime_citation_and_whole_authored_source(tmp_path, monkeypatch):
    from ouroboros.context import build_runtime_section
    from ouroboros.owner_words import reply_target
    from tests._context_shared import _make_health_env

    _web_bridge(tmp_path, monkeypatch)
    selected = "The star chart is ready. " * 10 + "Finish."
    answer = f"**Atlas**\n\n{selected}"
    mirror = message_bus.log_chat("system", 1, 0, "Observatory update", record_type="project_completion_summary",
                                  message_meta={"completion_answer": answer, "project_id": "observatory"})
    from ouroboros.chat_references import ingress_reply
    excerpt = {"text": selected, "start_utf16": 6}
    ref = ingress_reply(tmp_path, {"reply_to": format_address(row_address(mirror)), "reply_excerpt": excerpt})
    task = {"id": "chart", "type": "task", "metadata": {"reply_reference": ref}}
    model = json.loads(build_runtime_section(_make_health_env(tmp_path / "env"), task).split("\n\n", 1)[1])
    assert model["reply_to"]["excerpt"] == excerpt and "client-reported" in model["reply_excerpt_note"]
    target = reply_target(tmp_path, ref)
    assert target == {"address": ref["address"], "author": "Ouroboros", "text": answer, "excerpt": excerpt,
                      "role": ref["role"]} and ref["role"] == "assistant"
    question = message_bus.log_chat("out", 1, 0, "Which chart?", record_type="quiz")
    with pytest.raises(ValueError, match="authored chat message"):
        ingress_reply(tmp_path, {"reply_to": format_address(row_address(question)), "reply_excerpt": excerpt})


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("initial_width", [12, 35, 64])
def test_reply_digest_acceptance_and_retry_verify_the_whole_supplied_prefix(tmp_path, monkeypatch, selected, initial_width):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    source = message_bus.log_chat("out", 1, 0, "A harbor.", record_type="proactive_message")
    row = row_address(source)
    prefix = format_address(row).rsplit("#", 1)[0] + "#"
    digest = row["row_sha256"]
    fields = {"reply_excerpt": {"text": "harbor", "start_utf16": 2}} if selected else {}
    def send(sha):
        bridge.handle_web_message("Explain", client_message_id="same-reply",
                                  task_metadata={"reply_to": prefix + sha, **fields})
    wrong = digest[:-1] + ("1" if digest[-1] == "0" else "0")
    if selected:
        with pytest.raises(ValueError, match="unavailable"):
            send(wrong)
    send(digest[:initial_width])
    assert echoes[-1]["reply_to"]["address"] == prefix + digest[:12]
    assert echoes[-1]["reply_to"]["snippet"] == "A harbor."
    monkeypatch.setattr("ouroboros.chat_references.resolve_row", lambda *_: pytest.fail("accepted digest lost"))
    for length in (12, 36, 64):
        send(digest[:length])
    with pytest.raises(ValueError, match="different message"):
        send(wrong)
    assert bridge._inbox.qsize() == 1


def test_older_reply_with_only_a_short_address_requires_the_source_for_more_digits(tmp_path, monkeypatch):
    from ouroboros.chat_references import same_reply

    _web_bridge(tmp_path, monkeypatch)
    source = message_bus.log_chat("out", 1, 0, "Earlier source.")
    row = row_address(source)
    short = format_address(row)
    prefix = short.rsplit("#", 1)[0] + "#"
    digest = row["row_sha256"]
    wrong = digest[:-1] + ("1" if digest[-1] == "0" else "0")
    saved = {"address": short}
    assert same_reply(tmp_path, saved, {"address": prefix + digest})
    assert not same_reply(tmp_path, saved, {"address": prefix + wrong})
    monkeypatch.setattr("ouroboros.chat_references.resolve_history_row", lambda *_: (None, {"state": "absent"}))
    assert same_reply(tmp_path, saved, {"address": short})
    assert not same_reply(tmp_path, saved, {"address": prefix + digest})


@pytest.mark.parametrize("fields,expected", [
    ({"transport": {"actor": {"display_name": "Mira", "platform_actor_id": "42"}}}, "Mira"),
    ({"transport": {"actor": {"platform_actor_id": "42"}}}, "42"),
    ({"sender_label": "Sasha", "transport": {"actor": {"display_name": "Mira"}}}, "Sasha"),
    ({"source": "web"}, "You"),
    ({"source": "presence:telegram", "transport": {"provider": "telegram"}}, "author not recorded"),
], ids=["actor-name", "actor-id", "sender-label-first", "owner-door", "unrecorded-place-author"])
def test_a_replied_row_is_signed_with_the_author_its_record_names(tmp_path, monkeypatch, fields, expected):
    """Without a ``sender_label`` the actor its transport recorded is the author; only a row of the owner's own door
    is You, and a place row whose author nothing records is not."""
    from ouroboros.chat_references import ingress_reply, projection_fields
    from ouroboros.gateway.history import _history_identity

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    source = message_bus.log_chat("in", 1, 1, "Meet at noon.", ts="2026-10-01T00:00:00Z", **{"source": "skill:telegram", **fields})
    assert projection_fields(source)["reply_author_label"] == expected
    assert _history_identity({**source, **projection_fields(source)})["reply_author_label"] == expected
    if not fields.get("source"):  # a skill's live echo; web is echoed at acceptance, places use their bound room
        message_bus.record_inbound_message(bridge, {"source": "skill:telegram", **fields}, chat_id=1, user_id=1,
                                           client_message_id="speaker-live", text="Meet at noon.", ts="2026-10-01T00:00:01Z")
        assert echoes[-1]["reply_author_label"] == expected
        replay = next(row for row in _history(tmp_path) if row.get("client_message_id") == "speaker-live")
        assert replay["reply_author_label"] == expected and replay["row_address"] == echoes[-1]["row_address"]
    whole = resolve_reply(tmp_path, format_address(row_address(source)))
    assert whole["role"] == "user" and whole["author_label"] == expected
    selected = ingress_reply(tmp_path, {"reply_to": whole["address"], "reply_excerpt": {"text": "noon", "start_utf16": 8}})
    assert selected["author_label"] == expected and selected["excerpt"] == {"text": "noon", "start_utf16": 8}


def test_a_late_answer_stored_without_its_snapshot_echoes_selectable_as_history_replays_it(tmp_path, monkeypatch):
    """The card's answer of an older late answer lives in the store: a skill's live echo reads it as history does, so
    its entered words are selectable at once, not only after a reload."""
    from ouroboros.owner_quiz import late_answer_owner_text

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    block = {"quiz_id": "trail", "state": "answered", "answered_after_terminal": True,
             "answered_index": 0, "options": ["Take the ridge"], "comment": "Follow the creek at dawn."}
    monkeypatch.setattr("ouroboros.owner_quiz.quiz_states", lambda root, task_id: {"trail": block})
    text = late_answer_owner_text(block)
    accepted, rejoined = message_bus.accept_local_message(
        bridge, tmp_path, text, chat_id=1, user_id=1, client_message_id="late-skill", source="skill:telegram",
        task_metadata={"late_answer": {"task_id": "trail-work", "quiz_id": "trail"}})
    assert not rejoined and "answer_snapshot" not in accepted["late_answer"]
    message_bus.record_inbound_message(bridge, bridge._inbox.get_nowait(), chat_id=1, user_id=1,
                                       client_message_id="late-skill", text=text, ts=accepted["ts"])
    [history] = [r for r in _history(tmp_path) if r.get("client_message_id") == "late-skill"]
    assert "reply_excerpt_disabled" not in history
    assert "reply_excerpt_disabled" not in echoes[-1], "the live echo reads the same store as the replay"


def test_a_checksum_only_legacy_binding_selects_the_main_original_in_its_project(tmp_path):
    """An ordinary legacy Project's saved binding names its opening only by Main and the words' checksum. The
    public in-process reader admits that candidate for the row's source room, and the batched physical resolver
    with the membership rule decide it is the one original: the quote is selected, not dropped. A binding that
    names another source room is not admitted for this row."""
    from ouroboros.chat_references import visible_source_reply
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "garden")
    original = {"direction": "in", "chat_id": 1, "client_message_id": "garden",
                "ts": "2024-04-08T13:45:00Z", "text": "Count the seedlings"}
    dialogue_store.append(tmp_path, "chat", original)
    legacy_ref = build_owner_message_ref(chat_id=1, client_message_id="", ts="", text=original["text"])
    bindings = {"legacy": {"project_id": project["id"], "project_chat_id": project["chat_id"],
                           "source_ref": legacy_ref, "source_text": original["text"]}}
    (tmp_path / "state" / "project_task_bindings.json").write_text(json.dumps({"bindings": bindings}), encoding="utf-8")
    quote = visible_source_reply(tmp_path, legacy_ref, project["chat_id"])
    assert quote["snippet"] == "Count the seedlings" and quote["role"] == "user"
    assert quote["address"] == format_address(row_address(original))
    bindings["legacy"]["source_ref"] = {**legacy_ref, "chat_id": 2}  # another source room: not admitted
    (tmp_path / "state" / "project_task_bindings.json").write_text(json.dumps({"bindings": bindings}), encoding="utf-8")
    assert visible_source_reply(tmp_path, legacy_ref, project["chat_id"]) == {}
