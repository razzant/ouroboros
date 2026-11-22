"""A receipt states where its act landed once, and its saved row, its live frame and a reload agree.

An act that named an explicit owner source (a reference of any shape, a selected source, the door's suppression) is
saved with the exact address its source resolves to, else with ``source_row_address: ""``: the host's stated "no row
wears this". That is never read as an older receipt, which an OMITTED key stays: one authoritative legacy match may
place it later. Synthetic words only.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

ROW = {"direction": "in", "chat_id": 1, "client_message_id": "cm-1", "ts": "2024-04-08T13:45:00Z",
       "text": "Count the seeds"}


def _stored(tmp_path, row=ROW):
    """The synthetic owner row in the canonical store: its address and the host's own computed reference."""
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.owner_relay import owner_row_source
    from tests import _dialogue

    _dialogue.seed(tmp_path, live={"chat": [row]})
    _dialogue.import_all(tmp_path)
    [stored] = _dialogue.chat_rows(tmp_path)
    return format_address(row_address(row)), owner_row_source(stored)[0]


def _bridge(root, monkeypatch):
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    bridge = message_bus.LocalChatBridge({})
    frames: list = []
    bridge._broadcast_fn = frames.append
    return bridge, frames


def _weak_ref(message="cm-1"):
    """An older, valid reference (no row digest) of the synthetic row."""
    from ouroboros.project_dialogue import build_owner_message_ref

    return build_owner_message_ref(chat_id=1, client_message_id=message, ts=ROW["ts"], text=ROW["text"])


def _saved(root):
    return [json.loads(line) for line in (root / "logs" / "chat_annotations.jsonl").read_text(encoding="utf-8").splitlines()]


def _frames(frames):
    return [f for f in frames if f.get("type") == "message_annotation"]


def _emit(root, bridge, evt, **act):
    from supervisor.events_project_routing import _emit_routing_receipt

    return _emit_routing_receipt(SimpleNamespace(DRIVE_ROOT=root, bridge=bridge),
                                 {"client_message_id": "cm-1", "routing_token": "act-1", "chat_id": 1, **evt},
                                 **{"action": "promote_chat_to_task", "target": "work-1", "target_label": "Seeds",
                                    "status": "scheduled", **act})


@pytest.mark.parametrize("unplaced", [False, True])
def test_an_older_receipt_is_placed_by_one_match_but_a_saved_unplaced_one_never_is(tmp_path, unplaced):
    from ouroboros.chat_annotations import addressed_chat_annotations, append_chat_annotation, message_receipts

    address, _ = _stored(tmp_path)
    append_chat_annotation(tmp_path, "cm-1", action="promote_chat_to_task", target="work-1", status="scheduled",
                           routing_token="act-1", source_chat_id=1, source_unplaced=unplaced)
    [row] = _saved(tmp_path)
    assert ("source_row_address" in row) is unplaced and row.get("source_row_address", "") == ""  # the raw row stays as written
    placed = addressed_chat_annotations(tmp_path, by_token=True)
    receipts = message_receipts(tmp_path, [(1, "cm-1", "")])
    if unplaced:
        assert placed == {}
        assert receipts == []  # an unplaced act is no evidence about the one row the legacy lookup matches
    else:
        assert list(placed) == [(1, "cm-1", address, "act-1")]  # the address is the projection's key, not the raw row's
        assert "source_row_address" not in placed[1, "cm-1", address, "act-1"]
        assert [r["status"] for _, r in receipts] == ["scheduled"]


@pytest.mark.parametrize("unplaced_first", [False, True])
def test_an_omitted_and_a_stated_unplaced_row_of_one_act_never_share_a_key(tmp_path, unplaced_first):
    """One correlation key (room, client id, token) holds both rows: neither erases the other's act, and only the
    omitted one is an older receipt."""
    from ouroboros.chat_annotations import _annotation_index, addressed_chat_annotations, append_chat_annotation

    address, _ = _stored(tmp_path)
    acts = [("scheduled", False), ("unconfirmed", True)]
    for status, unplaced in (acts[::-1] if unplaced_first else acts):
        append_chat_annotation(tmp_path, "cm-1", action="promote_chat_to_task", target="work-1", status=status,
                               routing_token="act-1", source_chat_id=1, source_unplaced=unplaced)
    index = _annotation_index(pathlib.Path(tmp_path) / "logs" / "chat_annotations.jsonl", by_token=True)
    assert sorted(index, key=str) == sorted([(1, "cm-1", "act-1", ""), (1, "cm-1", "act-1", None)], key=str)
    placed = addressed_chat_annotations(tmp_path, by_token=True)
    assert [(key[2], row["status"]) for key, row in placed.items()] == [(address, "scheduled")]


def test_a_placed_source_wears_the_same_receipt_live_and_saved(tmp_path, monkeypatch):
    from ouroboros.chat_annotations import addressed_chat_annotations, routing_source_address

    address, ref = _stored(tmp_path)
    assert routing_source_address(ref, 1, "cm-1") == address
    bridge, frames = _bridge(tmp_path, monkeypatch)
    _emit(tmp_path, bridge, {"source_ref": ref})
    [frame] = _frames(frames)
    [row] = _saved(tmp_path)
    assert frame["source_row_address"] == row["source_row_address"] == address
    assert list(addressed_chat_annotations(tmp_path)) == [(1, "cm-1", address)]


def test_a_weak_reference_the_canonical_store_resolves_is_placed_live_and_saved(tmp_path, monkeypatch):
    """An older reference (no row digest) that exactly one stored row answers is placed by that row's address, once."""
    from ouroboros.chat_annotations import addressed_chat_annotations

    address, _ = _stored(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    _emit(tmp_path, bridge, {"source_ref": _weak_ref()})
    [frame] = _frames(frames)
    [row] = _saved(tmp_path)
    assert frame["source_row_address"] == row["source_row_address"] == address
    assert list(addressed_chat_annotations(tmp_path)) == [(1, "cm-1", address)]


@pytest.mark.parametrize("case", ["unavailable_canonical_row", "suppressed"])
def test_a_source_nothing_can_verify_is_saved_and_sent_unplaced_never_as_an_older_receipt(tmp_path, monkeypatch, case):
    """The canonical store holds no row for the valid reference (or the door suppressed the origin), while another
    stored message could share the client id: nothing wears the receipt, live or after a reload."""
    from ouroboros.chat_annotations import addressed_chat_annotations
    from tests import _dialogue

    _dialogue.seed(tmp_path, live={"chat": [{**ROW, "ts": "2024-04-08T09:00:00Z"}]})  # same client id, another message
    _dialogue.import_all(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    receipt = _emit(tmp_path, bridge, {"source_ref": _weak_ref()} if case == "unavailable_canonical_row"
                    else {"origin_suppressed": True})
    [row] = _saved(tmp_path)
    assert row["client_message_id"] == "cm-1" and row["source_row_address"] == ""
    assert addressed_chat_annotations(tmp_path) == {}
    [frame] = _frames(frames)
    assert frame["source_row_address"] == "" and frame["client_message_id"] == "cm-1"
    assert receipt["persisted"] and receipt["fields"]["source_row_address"] == ""


def test_an_act_that_named_no_source_keeps_its_live_frame_and_one_match_placement(tmp_path, monkeypatch):
    from ouroboros.chat_annotations import addressed_chat_annotations

    address, _ = _stored(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    _emit(tmp_path, bridge, {})
    [frame] = _frames(frames)
    [row] = _saved(tmp_path)
    assert frame["client_message_id"] == "cm-1" and "source_row_address" not in frame
    assert "source_row_address" not in row
    assert list(addressed_chat_annotations(tmp_path)) == [(1, "cm-1", address)]


def test_the_ingress_attested_unkept_echo_stays_distinct_from_a_stated_unplaced_receipt(tmp_path):
    """The direct path's own fact (no row was kept for these words) answers the bubble just sent by client id, with no
    address invented and a row its room alone may place: neither a stated "no row" nor a legacy guess."""
    from ouroboros.server_owner_routing import _record_routing_receipt
    from tests._main_followup import _Bridge

    bridge = _Bridge()
    _record_routing_receipt(bridge, SimpleNamespace(DRIVE_ROOT=tmp_path), chat_id=1, client_message_id="cm-1",
                            action="mailbox_delivery", target="t1", status="delivered", source_ref=_weak_ref(),
                            ingress_unkept=True)
    [ack] = bridge.acks
    assert ack["client_message_id"] == "cm-1" and "source_row_address" not in ack
    [row] = _saved(tmp_path)
    assert "source_row_address" not in row


def test_a_picker_on_an_unplaced_source_keeps_its_addressless_decisions(tmp_path, monkeypatch):
    bridge, frames = _bridge(tmp_path, monkeypatch)
    _emit(tmp_path, bridge, {"routing_token": "pick-1", "origin_suppressed": True}, action="route_decision",
          status="needs_manual_target", options=[{"action": "steer_task", "task_id": "work-1"}])
    [row] = _saved(tmp_path)
    assert "source_row_address" not in row and row["options"]
    [frame] = _frames(frames)
    assert frame["status"] == "needs_manual_target" and "source_row_address" not in frame


@pytest.mark.parametrize("act", ["owner_steer", "task_steer", "host_promote", "tool_promote"])
def test_an_unplaced_refusal_selects_the_system_row_only_where_its_path_writes_one(tmp_path, monkeypatch, act):
    """The ACK is not the carrier of a notice no row can wear when the act's own refusal writer publishes the System row
    (``steering._refusal_row``, ``_notify_host_initiated_refusal``); a task-issued act keeps its ACK, it narrates.
    The suppressed origin names no Reply."""
    bridge, frames = _bridge(tmp_path, monkeypatch)
    steer = act.endswith("steer")
    evt = {"origin_suppressed": True,
           **({"relay_target": {"id": "work-1", "title": "Seeds"}} if steer else {}),
           **({"issuer": {"kind": "task", "task_id": "child"}} if act == "task_steer" else {}),
           **({"host_initiated": True} if act == "host_promote" else {})}
    receipt = _emit(tmp_path, bridge, evt, **({"action": "steer_task", "status": "rejected", "reason": "cancel_pending"}
                                              if steer else {"status": "needs_manual_target", "reason": "empty_objective"}))
    [frame] = _frames(frames)
    assert frame["source_row_address"] == "" and frame.get("cause")
    carrier = "routing_ack" if act in ("task_steer", "tool_promote") else "system"
    assert frame["routing_notice_carrier"] == receipt["routing_notice_carrier"] == carrier
    assert "reply_to" not in frame


def test_an_answered_quiz_refusal_is_one_visible_row_from_the_existing_host_path(tmp_path, monkeypatch):
    """An owner steer with words, refused while its work is cancelling, on an answered quiz's own row (the system row
    ``quiz_answer`` the host logs): the ACK names the System row as carrier and the existing writer publishes exactly
    that row, once. The answer row is shown on its question card, never as a bubble an ACK could wear."""
    from ouroboros.owner_quiz import quiz_answer_frame, record_answered, record_asked
    from ouroboros.owner_relay import owner_row_address, owner_row_source
    from supervisor import message_bus
    from supervisor.steering import _handle_steer_task

    asker, quiz, message = "sail-maker", "cloth-choice", "quiz_answer:sail-maker:cloth-choice"
    record_asked(tmp_path, asker, quiz_id=quiz, question="Which cloth?", options=["Blue linen"])
    block = record_answered(tmp_path, asker, quiz_id=quiz, option_index=0, comment="  sail at dusk  ",
                            request_id="cloth-press")["block"]
    answer = message_bus.log_chat("system", 1, 0, quiz_answer_frame(block, 0, block.get("comment", ""), asker=asker),
                                  ts=block["answered_at"], source="owner_quiz_answer", task_id=asker,
                                  record_type="quiz_answer", client_message_id=message, quiz=block, drive_root=tmp_path)
    ref, _ = owner_row_source(answer)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    notices = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge, PENDING=[], RUNNING={},
                          send_with_budget=lambda chat, text, **kw: notices.append((chat, text, kw)))
    monkeypatch.setattr("ouroboros.cancel_intents.cancel_pending", lambda *_: True)
    _handle_steer_task({"chat_id": 1, "client_message_id": message, "target_task_id": "gardener",
                        "message": "Also order the blue linen", "routing_token": "act-1", "source_ref": ref}, ctx)
    [frame] = _frames(frames)
    assert frame["source_row_address"] == owner_row_address(answer) and frame["routing_notice_carrier"] == "system"
    [(chat, text, notice)] = notices
    assert chat == 1 and frame["cause"] in text and notice["system_type"] == "steer_not_delivered"
    assert notice["progress_meta"]["routing_notice_carrier"] == "system"


def test_an_unplaced_attachment_only_notice_of_a_started_act_keeps_its_ack_as_carrier(tmp_path, monkeypatch):
    """No refusal writer runs for a successful start, so its attachment note is never moved to a System row."""
    bridge, frames = _bridge(tmp_path, monkeypatch)
    manifest = [{"status": "rejected", "label": "notes.txt", "reason": "too_large"}]
    _emit(tmp_path, bridge, {"origin_suppressed": True, "host_initiated": True}, attachment_manifest=manifest)
    [frame] = _frames(frames)
    assert frame["source_row_address"] == "" and [row["label"] for row in frame["attachment_manifest"]] == ["notes.txt"]
    assert frame["routing_notice_carrier"] == "routing_ack" and "reply_to" not in frame


def test_the_bus_sends_a_stated_empty_address_and_omits_an_absent_one(tmp_path, monkeypatch):
    bridge, frames = _bridge(tmp_path, monkeypatch)
    bridge.send_routing_ack(1, client_message_id="cm-1", action="steer_task", status="delivered")
    bridge.send_routing_ack(1, client_message_id="cm-1", action="steer_task", status="delivered", source_row_address="")
    bridge.send_routing_ack(1, client_message_id="cm-1", action="steer_task", status="delivered", source_row_address="row:1@a#b")
    assert ["source_row_address" in f for f in _frames(frames)] == [False, True, True]
    assert [f.get("source_row_address") for f in _frames(frames)] == [None, "", "row:1@a#b"]
