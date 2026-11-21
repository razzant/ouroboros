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


def test_skill_ingress_shares_reply_and_address_with_web(tmp_path, monkeypatch):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    row = message_bus.log_chat("out", 1, 0, "Pick this", ts="2026-10-01T00:00:00Z")
    address = format_address(row_address(row))
    accepted, rejoined = message_bus.accept_local_message(
        bridge, tmp_path, "yes", chat_id=1, user_id=1, client_message_id="skill-reply",
        source="skill:telegram", task_metadata={"reply_to": address})
    assert not rejoined and accepted["reply_to"]["address"] == address
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


def _route(bridge, root, monkeypatch, row, *, open_turn="", live=None, chat=42):
    """The accepted row through the owner router, as server.py hands it over: (turns, follow-ups, receipts).
    ``live``: the live roots each chat may address (``addressable_roots``), by chat id."""
    from types import SimpleNamespace

    from ouroboros import server_owner_routing as routing
    from ouroboros.project_dialogue import build_owner_message_ref

    [queued] = _drain(bridge)
    started, followups, receipts = [], [], []

    def running_work(*_args, **_kwargs):
        pytest.fail("the owner bot's words are never a running Project work's input")

    monkeypatch.setattr(routing, "_route_project_chat_to_running_task", running_work)
    monkeypatch.setattr(routing, "_route_main_followup",
                        lambda _ctx, chat, *_args, **_kwargs: followups.append(chat) or open_turn)
    monkeypatch.setattr(routing, "addressable_roots", lambda _ctx: SimpleNamespace(
        rows=lambda chat=None: [dict(root) for root in (live or {}).get(chat, [])], gap=None))
    monkeypatch.setattr(routing, "_decision_turn_metadata", lambda _ctx, _room, _cmid, meta, _roots: meta)
    monkeypatch.setattr(routing, "_record_routing_receipt", lambda _bridge, _ctx, **fields: receipts.append(fields))
    monkeypatch.setattr("ouroboros.owner_relay.carry_row_at_receipt", lambda *_args, **_kwargs: None)
    source = build_owner_message_ref(chat_id=chat, client_message_id=row["client_message_id"], ts=row["ts"],
                                     text=row["text"])
    ctx = SimpleNamespace(DRIVE_ROOT=root, handle_chat_direct=lambda *args, **kwargs: started.append((args, kwargs)))
    routing._route_owner_message(bridge, ctx, {
        "chat_id": chat, "text": row["text"], "log_text": row["text"], "image_caption": "", "image_data": None,
        "client_message_id": row["client_message_id"], "task_constraint": None, "origin_message_ref": source,
        "task_metadata": {**(queued.get("task_metadata") or {}), "_host_operation": True},
        "source": "skill:telegram", "received_at": row["ts"]})
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
    with pytest.raises(StoreBusy):
        _bot_message(bridge, tmp_path, "task:helper", mid=50)
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
