"""``/presence/turn`` takes connector facts inside ``event.message`` on any core; the event's keys stay frozen.

Version-agnostic on purpose: this module imports nothing a 7.10 core without the message-facts renderer
lacks, so the same file runs against the core a connector meets. An older core keeps the facts on the
event record and ignores them; a newer one also states them in the focus frame. Either way the person's
words stay the event's ``text`` and an unknown top-level key is still refused. Synthetic events only.
"""
from __future__ import annotations

import copy
from dataclasses import asdict

import pytest
from starlette.testclient import TestClient

from ouroboros import places as p
from ouroboros.gateway import host_service
from ouroboros.presence_runner import PresenceTurnEvent, _live_task_rows, run_presence_turn
from ouroboros.task_results import write_task_result

TOO_LARGE = {"kind": "document", "file_id": "BQAC-synthetic-2", "file_unique_id": "AgAD-synthetic-2",
             "file_name": "dump.zip", "mime_type": "application/zip", "file_size": 314_572_800, "status": "too_large",
             "note": ("Not staged: larger than the Bot API's 20 MiB download limit. "
                      "The Telegram message reference is retained.")}
# telegram-bot 0.10.6: a caption-less album item whose only file is above the Bot API download limit.
MESSAGE_0106 = {"event_kind": "message", "message_id": 4242, "sent_at_epoch": 1793870400,
                "sent_at": "2026-11-05T09:20:00+00:00", "reply_to_message_id": None, "attachments": [TOO_LARGE],
                "media_group_id": "13579"}
# The 0.12.0 seam: every new fact lives inside event.message.
MESSAGE_0120 = {
    "event_kind": "message", "message_id": 4243, "sent_at": "2026-11-05T09:21:00+00:00",
    "attachments": [TOO_LARGE, {"kind": "video_note", "file_name": "round.mp4", "mime_type": "video/mp4",
                                "file_size": 1_048_576, "status": "failed", "note": "Telegram refused the download."},
                    {"kind": "animation", "file_name": "loop.gif", "mime_type": "image/gif", "file_size": 4096,
                     "status": "unsupported", "note": "This connector does not stage animations."}],
    "non_text": [{"kind": "poll", "description": 'poll "Lunch?": Pizza; Sushi (anonymous, one answer)'},
                 {"kind": "sticker", "description": "sticker 😀"}],
    "forward_origin": {"type": "channel", "date": 1793870400, "message_id": 77,
                       "chat": {"id": -100123, "title": "Synthetic Blog", "username": "synthetic_blog"}},
    "is_automatic_forward": True, "media_group_id": "24680", "media_group_position": 1,
    "reply_to_message_id": 41, "reply_to_message": {"message_id": 41, "from": {"id": 7, "first_name": "Bot"},
                                                    "text": "Earlier words"},
    "quote": {"text": "Earlier", "position": 0},
    "entities": [{"type": "text_link", "offset": 0, "length": 4, "url": "https://disk.example/d/synthetic"}],
}


@pytest.fixture
def turn_door(tmp_path, monkeypatch):
    p.connect_place(tmp_path, connector="bridge", provider="chat", account_id="a", conversation_id="r",
                    wakes=["any_message"])
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance",
                        lambda _root: {"status": "available", "remaining_usd": 2, "settled_usd": 1})
    monkeypatch.setattr("ouroboros.presence_context._own_work_section", lambda *_args: "Already started: task1")

    async def authenticated(*_args):
        return "bridge", {}

    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    seen = []

    class Model:
        def handle_task(self, task):
            seen.append(copy.deepcopy(task))
            write_task_result(tmp_path, task["id"], "completed", metadata={
                **task["metadata"], "presence_outcome": "silent", "presence_result_text": ""},
                terminal_origin="model_final", result="")
            return [{"type": "presence_result", "outcome": "silent", "text": ""}]

    event = PresenceTurnEvent("event-wire", "chat", "a", "r", "", "chat:a:r", {"platform_actor_id": "person"},
                              {}, {}, "", facts=("any_message",))
    payload = asdict(event)
    for key in ("delivery_reporting_version", "continuation_version", "identity"):
        payload.pop(key)
    payload["facts"] = ["any_message"]
    with TestClient(host_service.create_host_service_app(tmp_path)) as client:
        client.app.state.host_service_context.presence_runner = lambda **kw: run_presence_turn(
            repo_dir=tmp_path, drive_root=tmp_path, agent_factory=lambda **_kw: Model(), **kw)
        yield tmp_path, client, payload, seen


def _post(client, payload, **extra):
    return client.post("/presence/turn", json={"event": payload, "staged_files": [], "continuation_version": 1,
                                               "delivery_reporting_version": 1, **extra})


@pytest.mark.parametrize("message,text", [(MESSAGE_0106, ""), (MESSAGE_0120, "Look at this")],
                         ids=["telegram-bot-0.10.6", "telegram-bot-0.12.0"])
def test_message_facts_pass_the_turn_door_whole_and_leave_the_words_alone(turn_door, message, text):
    root, client, payload, seen = turn_door
    payload = {**payload, "source_event_id": f"wire-{message['message_id']}", "message": message, "text": text}
    response = _post(client, payload)
    assert response.status_code == 200, response.text
    assert response.json()["outcome"] == "silent"
    task = seen[-1]
    presence = task["metadata"]["presence"]
    assert presence["event"]["message"] == message and presence["observed_text"] == text
    rows = _live_task_rows(root, task["id"], presence["event"]["conversation_key"])
    arrival = next(row for row in rows if row.get("direction") == "in")
    assert arrival["text"] == text and arrival["transport"]["message"] == message
    # The same event keeps one identity: a resend answers from its durable turn, no second model run.
    again = _post(client, payload)
    assert again.status_code == 200 and again.json()["turn_ref"] == response.json()["turn_ref"] == task["id"]
    assert len(seen) == 1


def test_top_level_event_and_payload_keys_stay_frozen(turn_door):
    _root, client, payload, seen = turn_door
    payload = {**payload, "message": MESSAGE_0106}
    extra_event = _post(client, {**payload, "non_text": MESSAGE_0120["non_text"]})
    assert extra_event.status_code == 400 and extra_event.json()["code"] == "presence_event_invalid"
    assert extra_event.json()["disposition"] == "rejected"
    extra_payload = _post(client, payload, attachments=[TOO_LARGE])
    assert extra_payload.status_code == 400 and extra_payload.json()["code"] == "presence_payload_invalid"
    assert not seen
