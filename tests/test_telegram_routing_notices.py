"""A routing act's owner notice reaches Telegram once, through the carrier the host selected.

The real producers (the steer handler, the promote publication boundary, both receipt writers)
publish through the real message bus into the real Telegram outbound handler; only the Bot API
wire is a recording fake. The ACK is the normal carrier and replies to the owner's own message
through the exact message map; a refusal's paired System row stays in the web and is skipped
here only when it says the ACK carried it; nothing is deduplicated by text or by id.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest

from tests.test_telegram_format_parity import _load_skill

# The shared native-skill loader replaces its package entries in sys.modules.
pytestmark = pytest.mark.serial

ASK = "Добавь в экспорт оглавление"


class Api:
    def __init__(self, state_dir, data_dir):
        self.state_dir, self.data_dir, self.logs = state_dir, data_dir, []

    def get_state_dir(self):
        return str(self.state_dir)

    def get_runtime_info(self):
        return {"data_dir": str(self.data_dir)}

    def get_settings(self, _keys):
        return {"TELEGRAM_BOT_TOKEN": "test-token"}

    def log(self, level, message):
        self.logs.append((level, message))


@pytest.fixture
def phone(tmp_path, monkeypatch):
    """The owner's Telegram chat 42 behind the real bus; ``calls`` holds every Bot API request."""
    from ouroboros.event_bus import CHAT_OUTBOUND
    from ouroboros.utils import append_jsonl
    from supervisor import message_bus, queue

    plugin, transport = _load_skill()
    data, skill = tmp_path / "data", tmp_path / "skill"
    data.mkdir()
    skill.mkdir()
    (skill / "settings.json").write_text(json.dumps({"TELEGRAM_CHAT_ID": "42"}), encoding="utf-8")
    api = Api(skill, data)
    calls = []

    class Wire:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, url, *, data, files):
            calls.append((url.rsplit("/", 1)[-1], dict(data or {})))
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 100 + len(calls)}})

    monkeypatch.setattr(transport.httpx, "AsyncClient", Wire)
    handle = plugin._make_outbound(api)
    frames = []

    def publish(topic, event):
        if topic == CHAT_OUTBOUND:
            asyncio.run(handle(dict(event)))

    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    for name, value in (("DATA_DIR", data), ("get_bridge", lambda: bridge),
                        ("load_state", lambda: {"owner_id": 42}), ("publish_event", publish)):
        monkeypatch.setattr(message_bus, name, value)
    for name, value in (("DRIVE_ROOT", str(data)), ("RUNNING", {}), ("PENDING", []),
                        ("ADMISSION_RESERVATIONS", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(queue, name, value)
    ctx = SimpleNamespace(DRIVE_ROOT=data, RUNNING={}, PENDING=[], bridge=bridge, append_jsonl=append_jsonl,
                          send_with_budget=message_bus.send_with_budget, persist_queue_snapshot=lambda **_kw: None)
    state = sys.modules[plugin.__package__ + ".lib.telegram_state"]

    def owner(mid, text=ASK, *, accept=True):
        """One owner message of this Telegram chat, through the bridge's map and the real ingress."""
        message = {"message_id": mid, "chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": text}
        state._owner_message_seen(api, 42, mid)
        fields = state._owner_ingress_fields(api, 42, message, mid)
        if not accept:
            return fields
        receiver = SimpleNamespace(enqueue_local_message=lambda *_a, **_k: None)
        row, _ = message_bus.accept_local_message(receiver, data, text, chat_id=42, user_id=42,
                                                  source="skill:telegram", **fields)
        return row

    return SimpleNamespace(
        plugin=plugin, api=api, ctx=ctx, bridge=bridge, data=data, calls=calls, frames=frames, owner=owner,
        sent=lambda: [data for method, data in calls if method == "sendMessage"],
        acks=lambda: [f for f in frames if f.get("type") == "message_annotation"],
        rows=lambda: [f for f in frames if f.get("type") == "chat" and f.get("role") == "system"])


def _live(task_id="t1", **fields):
    return {task_id: {"task": {"id": task_id, "chat_id": 42, "title": "Parser", **fields}}}


def _refuse(reason, phone, monkeypatch):
    """Arrange one real steer refusal; return the reason the handler states."""
    from ouroboros.task_results import write_task_result
    from supervisor import queue

    if reason != "target_unknown":
        phone.ctx.RUNNING.update(_live(**({"delegation_role": "subagent"} if reason == "subagent_target" else {})))
    if reason == "cancel_pending":
        monkeypatch.setattr("ouroboros.cancel_intents.cancel_pending", lambda *_a, **_k: True)
    elif reason == "cancel_in_transaction":  # the cancel lands between the up-front check and the write
        checks = iter([False, True])
        monkeypatch.setattr("ouroboros.cancel_intents.cancel_pending", lambda *_a, **_k: next(checks))
        return "cancel_pending"
    elif reason == "target_closed":
        actor = SimpleNamespace(_owner_message_admission_lock=threading.RLock(), _busy=True,
                                _accepting_owner_messages=False, _current_task_id="t1")
        monkeypatch.setattr("supervisor.workers.get_direct_chat_agent", lambda _tid: actor)
        monkeypatch.setattr("supervisor.workers.direct_chat_turn", lambda tid: {"id": tid, "chat_id": 42, "title": "Parser"})
    elif reason == "target_finished":
        write_task_result(phone.data, "t1", "completed")
    elif reason == "acceptance_fence_sealed":
        monkeypatch.setitem(queue.ACCEPTANCE_FENCES, "t1", {"status": "sealed"})
    elif reason == "mailbox_write_failed":
        def fail(*_a, **_k):
            raise OSError("disk full")
        monkeypatch.setattr("ouroboros.owner_relay.write_relay", fail)
    return reason


def _steer(phone, client_message_id, *, token="tok-1", quotes=None, issuer=None):
    from supervisor.steering import _handle_steer_task

    _handle_steer_task({"type": "steer_task", "target_task_id": "t1", "message": "Hurry, please", "chat_id": 42,
                        "client_message_id": client_message_id, "routing_token": token,
                        "issuer": issuer or {"kind": "owner_turn"}, **({"quotes": quotes} if quotes else {})}, phone.ctx)


def _cause(reason):
    from ouroboros.project_dialogue import routing_refusal_cause

    return routing_refusal_cause("steer_task", "rejected" if reason == "cancel_pending" else "needs_manual_target", reason)


_EARLY = ["target_unknown", "subagent_target", "cancel_pending"]
_LATE = ["target_closed", "target_finished", "acceptance_fence_sealed", "mailbox_write_failed", "cancel_in_transaction"]
_EVERY_REFUSAL = _EARLY + _LATE


@pytest.mark.parametrize("origin", ["quoted", "accepted"])
@pytest.mark.parametrize("reason", _EVERY_REFUSAL)
def test_a_refused_steer_of_an_owner_message_is_one_line_replying_to_that_message(phone, monkeypatch, reason, origin):
    """Early and late refusals alike: one host line, the target and the host's cause, as a native reply
    to the owner's own message even when it is his latest, and no System row for the phone or the web."""
    from ouroboros.chat_references import reference_from_row
    from ouroboros.owner_relay import owner_quote

    row = phone.owner(50)
    stated = _refuse(reason, phone, monkeypatch)
    _steer(phone, row["client_message_id"], quotes=[owner_quote(phone.data, row)] if origin == "quoted" else None)
    label = "Task" if reason == "target_unknown" else "Parser"
    [sent] = phone.sent()
    assert sent["text"] == f"{label} · {_cause(stated)}" and sent["reply_to_message_id"] == 50
    assert "disable_notification" not in sent  # a problem keeps its sound
    [ack] = phone.acks()
    assert ack["reply_to"] == reference_from_row(row) and ack["routing_notice_carrier"] == "routing_ack"
    assert phone.rows() == []
    # Control: an ordinary answer to the same latest message stays an unquoted neighbour.
    asyncio.run(phone.plugin._make_outbound(phone.api)({"text": "On it.", "reply_to": ack["reply_to"]}))
    assert "reply_to_message_id" not in phone.sent()[-1]


@pytest.mark.parametrize("reason", ["subagent_target", "cancel_pending", "cancel_in_transaction"])
def test_an_unlabelled_owner_act_is_told_once_while_the_web_keeps_its_system_row(phone, monkeypatch, reason):
    """A synthetic receipt id annotates no message: the ACK still carries the cause to the phone and the
    paired System row, which says so, stays for the web. That row replies to no owner row, never to the
    target task's own origin."""
    from ouroboros import dialogue_store
    from ouroboros.chat_references import origin_reply
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX, build_owner_message_ref
    from ouroboros.task_results import write_task_result

    old = phone.owner(40, "Почини парсер")
    write_task_result(phone.data, "t1", "running", origin_message_ref=build_owner_message_ref(
        chat_id=42, client_message_id=old["client_message_id"], ts=old["ts"], text=old["text"]))
    assert origin_reply(phone.data, "t1")["address"]  # what a task-keyed row would otherwise borrow
    stated = _refuse(reason, phone, monkeypatch)
    _steer(phone, AGENT_RECEIPT_ID_PREFIX + "relay-1")
    [sent] = phone.sent()
    assert sent["text"] == f"Parser · {_cause(stated)}" and "reply_to_message_id" not in sent
    [ack] = phone.acks()
    assert ack["routing_notice_carrier"] == "routing_ack" and "reply_to" not in ack
    [paired] = phone.rows()
    assert (paired["system_type"], paired["routing_notice_carrier"], paired["reply_to"]) == (
        "steer_not_delivered", "routing_ack", {})
    stored = [entry.row for entry in dialogue_store.newest(phone.data, "chat", limit=5)
              if entry.row.get("type") == "steer_not_delivered"]
    assert len(stored) == 1 and "reply_to" not in stored[0]


@pytest.mark.parametrize("reason", ["target_closed", "target_finished", "acceptance_fence_sealed", "mailbox_write_failed"])
def test_a_late_refusal_of_an_unlabelled_act_still_reaches_the_phone(phone, monkeypatch, reason):
    """A refusal found inside the delivery transaction writes no System row while its ACK is handed over:
    that ACK is the only carrier."""
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX

    _refuse(reason, phone, monkeypatch)
    _steer(phone, AGENT_RECEIPT_ID_PREFIX + "relay-2")
    assert [data["text"] for data in phone.sent()] == [f"Parser · {_cause(reason)}"]
    assert phone.rows() == []


def _no_ack(failure, phone, monkeypatch):
    """Keep this act's ACK from being handed over: its durable row is refused, or the bus raises."""
    if failure == "append_false":
        monkeypatch.setattr("ouroboros.project_dialogue.append_chat_annotation", lambda *_a, **_k: False)
    elif failure == "append_raises":
        def boom(*_a, **_k):
            raise OSError("annotation lock lost")
        monkeypatch.setattr("ouroboros.project_dialogue.append_chat_annotation", boom)
    else:
        def refuse(*_a, **_k):
            raise RuntimeError("bus closed")
        monkeypatch.setattr(phone.bridge, "send_routing_ack", refuse)


@pytest.mark.parametrize("failure", ["append_false", "append_raises", "ack_raises"])
def test_without_a_handed_over_ack_the_paired_system_row_carries_the_notice(phone, monkeypatch, failure):
    """Fail closed for the ACK, never for the owner: no ACK goes out without its durable row, so the
    paired System row selects itself as carrier and the phone shows it. Nothing claims the ACK said it."""
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX

    _refuse("subagent_target", phone, monkeypatch)
    _no_ack(failure, phone, monkeypatch)
    _steer(phone, AGENT_RECEIPT_ID_PREFIX + "relay-3")
    assert phone.acks() == []
    [paired] = phone.rows()
    assert paired["routing_notice_carrier"] == "system"
    assert [data["text"] for data in phone.sent()] == [f"Parser · {_cause('subagent_target')}"]


def _free(lock) -> bool:
    """Whether another thread can take ``lock`` now: an admission lock still held here would refuse it."""
    taken = []

    def probe():
        taken.append(lock.acquire(blocking=False))
        if taken[0]:
            lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return taken[0]


@pytest.mark.parametrize("failure", ["append_false", "append_raises", "ack_raises"])
@pytest.mark.parametrize("reason", _EVERY_REFUSAL)
@pytest.mark.parametrize("act", ["owner_message", "synthetic"])
def test_every_refusal_without_a_handed_over_ack_is_told_once_by_its_system_row(phone, monkeypatch, act, reason, failure):
    """Without a handed-over ACK the receipt selects the System row, and every owner refusal publishes it: early
    or found inside the delivery transaction, labelled or not, once, after the admission locks are released.
    It answers the act's own owner message, or explicitly none, never the target task's origin; the durable
    receipt stays as its writer left it."""
    from ouroboros import dialogue_store
    from ouroboros.chat_references import origin_reply, reference_from_row
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX, build_owner_message_ref, latest_chat_annotations
    from ouroboros.task_results import write_task_result
    from supervisor import queue

    old, row = phone.owner(40, "Почини парсер"), phone.owner(50)
    write_task_result(phone.data, "t1", "running", origin_message_ref=build_owner_message_ref(
        chat_id=42, client_message_id=old["client_message_id"], ts=old["ts"], text=old["text"]))
    assert origin_reply(phone.data, "t1")["address"]  # what a task-keyed row would otherwise borrow
    stated = _refuse(reason, phone, monkeypatch)
    _no_ack(failure, phone, monkeypatch)
    locks = [queue._queue_lock]
    if reason == "target_closed":
        from supervisor.workers import get_direct_chat_agent

        locks.append(get_direct_chat_agent("t1")._owner_message_admission_lock)
    send, free = phone.ctx.send_with_budget, []
    phone.ctx.send_with_budget = lambda *args, **kwargs: free.append(all(map(_free, locks))) or send(*args, **kwargs)
    client_message_id = row["client_message_id"] if act == "owner_message" else AGENT_RECEIPT_ID_PREFIX + "relay-5"
    _steer(phone, client_message_id)
    assert phone.acks() == [] and free == [True]
    [notice] = phone.rows()
    assert (notice["system_type"], notice["routing_notice_carrier"]) == ("steer_not_delivered", "system")
    assert notice["reply_to"] == (reference_from_row(row) if act == "owner_message" else {})
    [sent] = phone.sent()
    assert sent["text"] == notice["content"] == f"{'Task' if reason == 'target_unknown' else 'Parser'} · {_cause(stated)}"
    assert sent.get("reply_to_message_id") == (50 if act == "owner_message" else None)
    [stored] = [entry.row for entry in dialogue_store.newest(phone.data, "chat", limit=5)
                if entry.row.get("type") == "steer_not_delivered"]
    assert stored.get("reply_to", {}).get("address") == notice["reply_to"].get("address")
    annotation = latest_chat_annotations(phone.data).get(client_message_id)
    if failure == "ack_raises":  # written before the ACK raised, and untouched by the fallback
        assert (annotation["reason"], annotation["cause"]) == (stated, _cause(stated))
        assert "routing_notice_carrier" not in annotation
    else:
        assert annotation is None


@pytest.mark.parametrize("failure", [None, "append_false", "ack_raises"])
@pytest.mark.parametrize("reason", ["subagent_target", "target_finished"])
@pytest.mark.parametrize("act", ["owner_message", "synthetic"])
def test_a_task_that_spoke_for_itself_never_gets_the_system_row(phone, monkeypatch, act, reason, failure):
    """A task-issued act reads its typed refusal in its tool result and Logs row. Its ACK on a relayed owner
    message still reaches the phone; without that ACK no System row stands in, early or late."""
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX

    row = phone.owner(50)
    stated = _refuse(reason, phone, monkeypatch)
    if failure:
        _no_ack(failure, phone, monkeypatch)
    _steer(phone, row["client_message_id"] if act == "owner_message" else AGENT_RECEIPT_ID_PREFIX + "relay-6",
           issuer={"kind": "task", "task_id": "teller"})
    assert phone.rows() == []
    told = act == "owner_message" and failure is None
    assert [data["text"] for data in phone.sent()] == ([f"Parser · {_cause(stated)}"] if told else [])


@pytest.mark.parametrize("failure", [None, "append_false", "append_raises", "ack_raises"])
@pytest.mark.parametrize("act", ["owner_message", "synthetic"])
def test_an_accepted_act_stays_quiet_with_or_without_its_ack(phone, monkeypatch, act, failure):
    """Only a refusal has a cause to tell: a landed steer sends nothing and publishes no refusal row."""
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX

    row = phone.owner(50)
    phone.ctx.RUNNING.update(_live())
    if failure:
        _no_ack(failure, phone, monkeypatch)
    _steer(phone, row["client_message_id"] if act == "owner_message" else AGENT_RECEIPT_ID_PREFIX + "relay-7")
    assert phone.sent() == []
    assert [frame for frame in phone.rows() if frame.get("system_type") == "steer_not_delivered"] == []


def test_a_paired_row_that_cannot_be_written_leaves_the_cause_with_the_ack(phone, monkeypatch):
    """The ACK is handed over before its paired row: a row lost on the way never takes the cause with it."""
    from ouroboros.project_dialogue import AGENT_RECEIPT_ID_PREFIX

    def lost(*_a, **_k):
        raise RuntimeError("chat store unavailable")

    phone.ctx.send_with_budget = lost
    _refuse("target_unknown", phone, monkeypatch)
    _steer(phone, AGENT_RECEIPT_ID_PREFIX + "relay-4")
    assert [data["text"] for data in phone.sent()] == [f"Task · {_cause('target_unknown')}"]


def test_two_refused_acts_of_one_owner_message_are_two_notices(phone, monkeypatch):
    """One message can carry several acts (#896): each refusal is its own line, both replying to it."""
    row = phone.owner(50)
    _refuse("target_unknown", phone, monkeypatch)
    _steer(phone, row["client_message_id"], token="tok-a")
    _steer(phone, row["client_message_id"], token="tok-b")
    sent = phone.sent()
    assert [data["text"] for data in sent] == [f"Task · {_cause('target_unknown')}"] * 2
    assert [data["reply_to_message_id"] for data in sent] == [50, 50]


def test_no_accepted_owner_row_means_no_reply_anchor(phone, monkeypatch):
    """The bridge knows message 77 and its ``tg:42:77`` id, but the host accepted no row for it: the notice
    answers nothing rather than a reply guessed from the id's text."""
    fields = phone.owner(77, accept=False)
    _refuse("target_unknown", phone, monkeypatch)
    _steer(phone, fields["client_message_id"])
    [sent] = phone.sent()
    assert fields["client_message_id"] == "tg:42:77" and "reply_to_message_id" not in sent
    assert "reply_to" not in phone.acks()[0]


def test_a_landed_act_says_nothing_but_a_file_it_could_not_attach(phone, monkeypatch):
    from supervisor.events_project_routing import _emit_routing_receipt

    row = phone.owner(50)
    phone.ctx.RUNNING.update(_live())
    _steer(phone, row["client_message_id"])
    assert phone.sent() == [] and "routing_notice_carrier" not in phone.acks()[-1]
    _emit_routing_receipt(phone.ctx, {"client_message_id": row["client_message_id"], "chat_id": 42,
                                      "routing_token": "tok-files"},
                          action="mailbox_delivery", target="t1", status="delivered", attachment_manifest=[
                              {"ordinal": 0, "label": "схема.png", "status": "staged"},
                              {"ordinal": 1, "label": "отчёт.pdf", "status": "rejected", "reason": "source_missing"}])
    [sent] = phone.sent()
    assert sent["text"] == "Not attached to the task: отчёт.pdf" and sent["reply_to_message_id"] == 50


@pytest.mark.parametrize("act", ["project_unavailable", "file_not_attached", "landed"])
def test_the_direct_receipt_names_its_ack_the_carrier_from_the_fields_its_row_was_written_with(phone, act):
    """The server's own receipt writer (a message to a Project that takes no more work, words saved into a
    work's mailbox) states the same carrier as the supervisor's: a refusal or a file not attached is one line
    answering the owner's message, a plain landed delivery says nothing, and neither the carrier nor the
    reply is a field of the durable row."""
    from ouroboros.chat_references import reference_from_row
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.server_owner_routing import _record_routing_receipt

    row = phone.owner(50)
    if act == "project_unavailable":
        receipt = {"action": "project_route", "target": "gone", "status": "project_unavailable",
                   "reason": "project_unavailable"}
    else:
        receipt = {"action": "mailbox_delivery", "target": "t1", "status": "delivered", "attachment_manifest": [
            {"ordinal": 0, "label": "отчёт.pdf", "status": "rejected" if act == "file_not_attached" else "staged"}]}
    _record_routing_receipt(phone.bridge, phone.ctx, chat_id=42, client_message_id=row["client_message_id"], **receipt)
    [ack] = phone.acks()
    stored = latest_chat_annotations(phone.data)[row["client_message_id"]]
    assert {"routing_notice_carrier", "reply_to"}.isdisjoint(stored)
    if act == "landed":
        assert phone.sent() == [] and {"routing_notice_carrier", "reply_to"}.isdisjoint(ack)
        return
    assert (ack["routing_notice_carrier"], ack["reply_to"]) == ("routing_ack", reference_from_row(row))
    [sent] = phone.sent()
    if act == "project_unavailable":
        assert stored["cause"] == ack["cause"] == "Not started: the project is no longer available"
        assert sent["text"] == f"{ack['target_label']} · {ack['cause']}"
    else:
        assert sent["text"] == "Not attached to the task: отчёт.pdf"
    assert sent["reply_to_message_id"] == 50


def _host_promote(row, *, token="tok-pdf"):
    from ouroboros.project_dialogue import build_owner_message_ref

    return {"type": "promote_chat_to_task", "task_id": "task-pdf", "routing_token": token, "objective": row["text"],
            "chat_id": 42, "client_message_id": row["client_message_id"], "routed_from_main": True,
            "host_initiated": True, "source_text": row["text"], "source_ref": build_owner_message_ref(
                chat_id=42, client_message_id=row["client_message_id"], ts=row["ts"], text=row["text"])}


@pytest.mark.parametrize("failure", [None, "append_false", "append_raises", "ack_raises"])
def test_a_refused_host_promote_is_told_once_and_a_replay_tells_nothing(phone, monkeypatch, failure):
    """A Swarm, skill-card or picker promote has its typed System row in the owner's chat. The phone gets
    the ACK's line, replying to his message, and skips that row; without a handed-over ACK the row is the
    line. The same event again is a replay: the existing guard publishes nothing a second time."""
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    monkeypatch.setattr("supervisor.workers._worker_pool_execution_state",
                        lambda *_a, **_k: {"available": False, "disabled_reason": "paused"})
    row = phone.owner(50, "Запусти «Экспорт PDF» отдельным проектом и начни с шаблона страницы")
    if failure:
        _no_ack(failure, phone, monkeypatch)
    outcome = _handle_promote_chat_to_task(_host_promote(row), phone.ctx)
    assert outcome["reason"] == "worker_pool_unavailable" and "_routing_notice_carrier" not in outcome
    [paired] = phone.rows()
    assert paired["system_type"] == "task_not_started" and paired["reply_to"]["address"]
    [sent] = phone.sent()
    assert sent["text"].endswith(" · Not started: no worker is available right now")
    if failure is None:
        assert paired["routing_notice_carrier"] == "routing_ack" and len(phone.acks()) == 1
        assert sent["reply_to_message_id"] == 50
    else:
        assert paired["routing_notice_carrier"] == "system" and phone.acks() == []
        assert sent["text"] == paired["content"] and sent["reply_to_message_id"] == 50
    replay = _handle_promote_chat_to_task(_host_promote(row), phone.ctx)
    assert replay.get("replayed") is True and len(phone.sent()) == 1 and len(phone.rows()) == 1


def test_a_picker_gets_no_invented_reason_and_one_message_with_its_files(phone):
    """A normal picker carries no cause; a refusal that has one leads the same message, and files the host
    could not attach close it: cause, then the choice, then the files, as ONE message."""
    from supervisor.events_project_routing import _emit_routing_receipt

    row = phone.owner(50)
    options = [{"action": "steer_task", "task_id": "t1", "label": "Parser"},
               {"action": "new_task_in_project", "project_id": "p1", "project_name": "Export"}]
    rejected = [{"ordinal": 0, "label": "оглавление.docx", "status": "rejected", "reason": "too_large"}]
    _emit_routing_receipt(phone.ctx, {"client_message_id": row["client_message_id"], "chat_id": 42,
                                      "routing_token": "tok-pick"},
                          action="route_decision", status="needs_manual_target", options=options,
                          attachment_manifest=rejected)
    [ack] = phone.acks()
    assert "cause" not in ack and ack["routing_notice_carrier"] == "routing_ack"
    picker = ("Choose a target for the last message:\n1. Parser\n2. New task in Export\n"
              "Reply with a number, or tap an option in the web chat.")
    [sent] = phone.sent()
    assert sent["text"] == f"{picker}\nNot attached to the task: оглавление.docx" and sent["reply_to_message_id"] == 50
    asyncio.run(phone.plugin._make_outbound(phone.api)({**ack, "cause": "Not started: the attachments could not be staged",
                                                       "target_label": "Export"}))
    assert phone.sent()[-1]["text"] == (f"Export · Not started: the attachments could not be staged\n{picker}"
                                        "\nNot attached to the task: оглавление.docx")
