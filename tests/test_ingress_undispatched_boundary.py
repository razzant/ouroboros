"""The edge of the one replay-safe retry (the durable dispatch obligation of ``supervisor.message_ingress``).

An owner message's accepted row is handed to dispatch exactly once: by its acceptor, by a same-id
retry while the dispatch is still owed (the acceptor's write landed but the handoff never happened),
or by the recovery when the bus opens again. Everything else only rejoins -- no second row, no second
dispatch: a dispatch that was entered and then raised (it may have queued), a changed message under
that id (refused; the owed dispatch waits for the real one), and the losing side of two retries racing
for the one handover. Web frames and named skill deliveries (Host early rejoin included) follow the
same rule. The other half of the same fact: only the process that took a row's dispatch says it was
entered (``ingress_dispatched``); a dispatch an ended process took is unknown -- a crash that kept the
host session included -- while a dispatch still owed reads ``ingress_pending`` in every process.
"""
from __future__ import annotations

import json
import threading

import pytest

from tests.test_chat_attachments import PNG, _history, _inject, _rows, _send_web, _skill_client, _upload, _web_bridge
from tests.test_chat_attachments import files_app as files_app  # noqa: F401 -- fixture re-export


def _new_process(monkeypatch):
    """The host process ended and another took over: its generation id is gone, the store stays."""
    from ouroboros import process_custody

    monkeypatch.setattr(process_custody, "_SESSION_ID", "the-next-host-process")


def _land_then_fail(monkeypatch):
    """The canonical write commits and its acknowledgement is lost: the row is in the store with
    its dispatch still owed, the caller sees an error."""
    from supervisor import message_bus

    real = message_bus.accept_chat

    def land_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("the append's acknowledgement was lost")

    monkeypatch.setattr(message_bus, "accept_chat", land_then_fail)
    return lambda: monkeypatch.setattr(message_bus, "accept_chat", real)


def _facts(tmp_path, chat_id, cmid):
    from ouroboros import dialogue_store

    return dialogue_store.dispatch_facts(tmp_path, chat_id, cmid)


def test_a_landed_row_whose_handoff_was_lost_is_dispatched_once_by_the_first_retry(files_app, tmp_path, monkeypatch):
    """The obligation outlives the process: the saved row reads ``ingress_pending`` in the next
    process too, the first same-id retry performs the one dispatch, later retries rejoin."""
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    photo = _upload(files_app, "one.png", PNG)
    restore = _land_then_fail(monkeypatch)
    with pytest.raises(RuntimeError):
        _send_web(bridge, "Посмотри", [photo])
    restore()
    assert _facts(tmp_path, 1, "cm-1")["state"] == "pending"
    _new_process(monkeypatch)  # the process ended; the owed dispatch did not
    (replayed,) = _history(tmp_path)
    assert replayed["ingress_accepted"] is True and replayed["ingress_pending"] is True, "saved, still owed"
    assert "ingress_dispatched" not in replayed
    _send_web(bridge, "Посмотри", [photo])
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1, "the one dispatch"
    assert echoes[-1]["ingress_accepted"] is True and echoes[-1]["ingress_dispatched"] is True
    _send_web(bridge, "Посмотри", [photo])
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1, "a later retry only rejoins"
    assert _facts(tmp_path, 1, "cm-1")["state"] == "completed"


def test_the_recovery_performs_an_owed_dispatch_once_without_reading_any_chain(tmp_path, monkeypatch):
    """A crash between the commit and the handoff: when the bus opens again the pending obligation
    is selected through its index and dispatched exactly once; retries and a second recovery rejoin."""
    from ouroboros import dialogue_import, utils
    from supervisor import message_bus

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    restore = _land_then_fail(monkeypatch)
    with pytest.raises(RuntimeError):
        _send_web(bridge, "после сбоя", [])
    restore()
    _new_process(monkeypatch)
    reads = []
    real_chain = utils.jsonl_chain_handles
    monkeypatch.setattr(utils, "jsonl_chain_handles", lambda *a, **k: reads.append(a) or real_chain(*a, **k))
    monkeypatch.setattr(dialogue_import, "scan_pending", lambda *a, **k: reads.append(a) or None)
    next_bridge = message_bus.LocalChatBridge()
    assert message_bus.recover_pending_dispatches(next_bridge, tmp_path) == 1
    assert next_bridge._inbox.qsize() == 1 and reads == [], "from the obligation, not from a chain"
    queued = next_bridge.get_updates(0, timeout=0)[0]["message"]
    assert (queued["text"], queued["client_message_id"], queued["accepted_source_row"]["text"]) == ("после сбоя", "cm-1", "после сбоя")
    assert queued["accepted_source_ref"]["client_message_id"] == "cm-1"
    assert message_bus.recover_pending_dispatches(next_bridge, tmp_path) == 0
    assert _facts(tmp_path, 1, "cm-1")["state"] == "completed"
    _send_web(next_bridge, "после сбоя", [])
    assert len(_rows(tmp_path)) == 1 and next_bridge._inbox.qsize() == 0, "the retry rejoins the dispatched row"


def test_the_recovery_abandons_an_obligation_naming_a_dispatcher_this_body_lacks(tmp_path, monkeypatch):
    from ouroboros import dialogue_store
    from supervisor import message_bus

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    restore = _land_then_fail(monkeypatch)
    with pytest.raises(RuntimeError):
        _send_web(bridge, "чужой диспетчер", [])
    restore()
    with dialogue_store.transaction(tmp_path, write=True) as conn:
        conn.execute("UPDATE dialogue SET dispatch_input=json_set(dispatch_input, '$.dispatcher', 'retired') WHERE client_message_id='cm-1'")
    assert message_bus.recover_pending_dispatches(bridge, tmp_path) == 0 and bridge._inbox.qsize() == 0
    assert _facts(tmp_path, 1, "cm-1")["state"] == "abandoned"
    assert dialogue_store.pending_dispatches(tmp_path) == []


def test_a_web_dispatch_that_was_entered_then_raised_is_unknown(tmp_path, monkeypatch):
    """The row landed and the enqueue began: it may have queued, so the obligation is ``dispatched``
    and a retry, text-only or not, only rejoins."""
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    real = bridge.enqueue_local_message

    def enqueue_then_fail(text, **message):
        real(text, **message)
        raise RuntimeError("the queue's acknowledgement was lost")

    monkeypatch.setattr(bridge, "enqueue_local_message", enqueue_then_fail)
    with pytest.raises(RuntimeError):
        _send_web(bridge, "только текст", [])
    monkeypatch.setattr(bridge, "enqueue_local_message", real)
    assert _facts(tmp_path, 1, "cm-1")["state"] == "dispatched"
    assert "ingress_pending" not in _history(tmp_path)[0]
    _send_web(bridge, "только текст", [])
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1, "the possibly queued item is not doubled"
    assert [echo["client_message_id"] for echo in echoes] == ["cm-1"]


def test_a_changed_message_under_an_owed_id_is_refused_and_the_obligation_waits(tmp_path, monkeypatch):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    restore = _land_then_fail(monkeypatch)
    with pytest.raises(RuntimeError):
        _send_web(bridge, "исходные слова", [])
    restore()
    with pytest.raises(ValueError, match="different message"):
        _send_web(bridge, "другие слова", [])
    assert _facts(tmp_path, 1, "cm-1")["state"] == "pending" and bridge._inbox.qsize() == 0
    assert "ingress_dispatched" not in _history(tmp_path)[0], "still owed is not dispatched"
    assert _history(tmp_path)[0]["ingress_pending"] is True
    _send_web(bridge, "исходные слова", [])
    assert bridge._inbox.qsize() == 1 and _facts(tmp_path, 1, "cm-1")["state"] == "completed"
    assert echoes[-1]["ingress_dispatched"] is True and _history(tmp_path)[0]["ingress_dispatched"] is True, "handed over"


def test_two_racing_retries_hand_the_row_over_once(tmp_path, monkeypatch):
    """The owed dispatch is claimed under the ingress lock: of two tabs pressing Send again at once (a
    duplicated tab shares the kept frame and id), exactly one dispatches."""
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    restore = _land_then_fail(monkeypatch)
    with pytest.raises(RuntimeError):
        _send_web(bridge, "дважды", [])
    restore()
    start = threading.Barrier(2)

    def retry():
        start.wait()
        _send_web(bridge, "дважды", [])

    threads = [threading.Thread(target=retry) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1 and len(echoes) == 2


def test_a_host_retry_after_restart_takes_the_owed_dispatch_then_rejoins(tmp_path, monkeypatch):
    """Host early rejoin: after the host session changed, the retry of a row whose dispatch was still
    owed performs it (not ``rejoined``), the operation reads pending in the new process, and a later
    retry rejoins without reaching the queue."""
    from tests.test_chat_inject_attachments import _skill_file
    from supervisor import message_bus

    bridge = message_bus.LocalChatBridge()
    client = _skill_client(tmp_path, bridge)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "state.json").write_text(json.dumps({"session_id": "before"}), encoding="utf-8")
    body = {"text": "scan", "client_message_id": "tg:1",
            "attachments": [{"path": str(_skill_file(tmp_path, "scan.pdf")), "name": "scan.pdf"}]}
    restore = _land_then_fail(monkeypatch)
    assert _inject(client, **body).status_code == 500
    restore()
    assert _facts(tmp_path, 42, "tg:1")["state"] == "pending"
    _new_process(monkeypatch)
    (tmp_path / "state" / "state.json").write_text(json.dumps({"session_id": "after"}), encoding="utf-8")
    taken = _inject(client, **body)
    assert taken.status_code == 202 and taken.json()["status"] == "queued" and "rejoined" not in taken.json()
    assert bridge._inbox.qsize() == 1
    state = client.get("/chat/operations/42:tg:1", headers={"X-Skill-Token": "token"}).json()
    assert state["status"] == "pending", "dispatched by the live process: its queue holds it"
    uploads = sorted(path.name for path in (tmp_path / "uploads").iterdir())
    replay = _inject(client, **body)
    assert replay.status_code == 202 and replay.json()["rejoined"] is True and bridge._inbox.qsize() == 1
    assert sorted(path.name for path in (tmp_path / "uploads").iterdir()) == uploads, "a rejoin copies nothing"
    assert len(_rows(tmp_path)) == 1


def test_a_host_dispatch_that_was_entered_then_raised_is_never_replayed(tmp_path, monkeypatch):
    from ouroboros.gateway import host_service
    from supervisor import message_bus

    bridge = message_bus.LocalChatBridge()
    client = _skill_client(tmp_path, bridge)

    def enqueue_then_fail(target, text, **message):
        target.enqueue_local_message(text, **message)
        raise RuntimeError("the queue's acknowledgement was lost")

    monkeypatch.setattr(host_service, "dispatch_accepted_restart", enqueue_then_fail)
    assert _inject(client, text="ping", client_message_id="tg:2").status_code == 500
    assert _facts(tmp_path, 42, "tg:2")["state"] == "dispatched"
    assert _inject(client, text="ping", client_message_id="tg:2").json()["rejoined"] is True
    assert bridge._inbox.qsize() == 1 and len(_rows(tmp_path)) == 1


def test_only_the_process_that_took_a_dispatch_says_it_was_entered(tmp_path, monkeypatch):
    """The live process's echo and history say ``ingress_dispatched`` for a row it took, rejoins
    included; once it ended neither does, and a retry adds no row and no queue item."""
    from ouroboros.process_custody import current_custody_session_id

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    _send_web(bridge, "слова", [])
    (row,) = _rows(tmp_path)
    assert row["ingress_process"] == current_custody_session_id()
    assert _facts(tmp_path, 1, "cm-1") == {"state": "completed", "process": current_custody_session_id(), "dispatcher": "bus"}
    assert echoes[0]["ingress_dispatched"] is True and _history(tmp_path)[0]["ingress_dispatched"] is True
    _send_web(bridge, "слова", [])
    assert echoes[-1]["ingress_dispatched"] is True and bridge._inbox.qsize() == 1, "its own row, rejoined"
    _new_process(monkeypatch)
    (replayed,) = _history(tmp_path)
    assert replayed["ingress_accepted"] is True and "ingress_dispatched" not in replayed
    assert "ingress_pending" not in replayed, "an ended process's dispatch is unknown, never pending"
    _send_web(bridge, "слова", [])
    assert "ingress_dispatched" not in echoes[-1] and echoes[-1]["ingress_accepted"] is True
    assert "ingress_pending" not in echoes[-1]
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1, "nothing replays"


def test_a_host_operation_from_before_a_crash_reads_lost_though_the_session_survived(tmp_path, monkeypatch):
    """A crash leaves ``state.json``'s session id as it was; the dispatch fact still names the ended
    process, so the operation reads ``lost`` instead of ``pending`` forever, and a retry only rejoins."""
    from ouroboros.process_custody import current_custody_session_id
    from supervisor import message_bus

    bridge = message_bus.LocalChatBridge()
    client = _skill_client(tmp_path, bridge)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "state.json").write_text(json.dumps({"session_id": "kept"}), encoding="utf-8")
    assert _inject(client, text="ping", client_message_id="tg:3").status_code == 202
    assert _rows(tmp_path)[0]["ingress_process"] == current_custody_session_id()

    def read():
        return client.get("/chat/operations/42:tg:3", headers={"X-Skill-Token": "token"}).json()

    assert read()["status"] == "pending", "the live process took it and queued it"
    _new_process(monkeypatch)
    state = read()
    assert (state["status"], state["reason"]) == ("lost", "host_restarted_before_answer")
    replay = _inject(client, text="ping", client_message_id="tg:3")
    assert replay.status_code == 202 and replay.json()["rejoined"] is True
    assert bridge._inbox.qsize() == 1 and len(_rows(tmp_path)) == 1


def test_history_during_append_does_not_claim_dispatch(tmp_path, monkeypatch):
    """The durable row may be read before the acceptance returns to its enqueue caller: that read says
    the dispatch is still owed (``ingress_pending``), never dispatched and never the ended-process
    unknown a client would settle for good."""
    from supervisor import message_bus
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    original = message_bus.accept_chat
    observations = []

    def append_and_read(*args, **kwargs):
        receipt = original(*args, **kwargs)
        observations.append((_history(tmp_path)[0], bridge._inbox.qsize()))
        return receipt

    monkeypatch.setattr(message_bus, "accept_chat", append_and_read)
    _send_web(bridge, "during append", [])
    assert observations[0][1] == 0
    assert "ingress_dispatched" not in observations[0][0]
    assert observations[0][0]["ingress_pending"] is True
    assert _history(tmp_path)[0]["ingress_dispatched"] is True
    assert "ingress_pending" not in _history(tmp_path)[0]
    assert echoes[-1]["ingress_dispatched"] is True and "ingress_pending" not in echoes[-1]


def test_retention_callback_failure_leaves_the_dispatch_owed(tmp_path, monkeypatch):
    from supervisor import message_bus, message_ingress
    bridge, _ = _web_bridge(tmp_path, monkeypatch)

    def refuse():
        raise RuntimeError("retention failed before dispatch")

    with pytest.raises(RuntimeError, match="retention failed"):
        message_bus.accept_local_message(bridge, tmp_path, "retain", chat_id=1, user_id=1,
                                         client_message_id="retention-failure", source="web", retain_inputs=refuse)
    row = message_bus.accepted_chat_message(tmp_path, 1, "retention-failure")
    assert row and message_ingress.accepted_here(row)
    assert not message_ingress.dispatch_entered(row, tmp_path)
    assert _facts(tmp_path, 1, "retention-failure")["state"] == "pending", "the recovery or a retry performs it"
    assert bridge._inbox.qsize() == 0


def test_deferred_dispatch_echo_does_not_claim_future_call(tmp_path, monkeypatch):
    """A caller's own dispatcher is entered after the echo: the echo says this process has not said
    yet (``ingress_pending``), the dispatcher sees its row entered, history says dispatched after."""
    from supervisor import message_ingress
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    observed = []

    def dispatch(text, **message):
        observed.append(message_ingress.dispatch_entered(message["accepted_source_row"]))
        bridge.enqueue_local_message(text, **message)

    bridge.handle_web_message("deferred", client_message_id="deferred", dispatch=dispatch)
    assert "ingress_dispatched" not in echoes[0]
    assert echoes[0]["ingress_pending"] is True, "the echo before a deferred dispatch: this process has not said yet"
    assert observed == [True]
    assert _history(tmp_path)[0]["ingress_dispatched"] is True
    assert _facts(tmp_path, 1, "deferred")["state"] == "completed", "its own dispatcher's handoff returned"
