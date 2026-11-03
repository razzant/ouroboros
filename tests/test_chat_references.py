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
    assert row["reply_to"]["snippet"] == ("One " + "word " * 80)[:160]
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
    assert reference["snippet"] == " ".join(answer.split())[:160]
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
    ("system", "project_completion_summary", {"project_id": "project-a"}, "system"),
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
