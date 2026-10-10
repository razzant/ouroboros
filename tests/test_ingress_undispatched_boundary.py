"""The edge of the one replay-safe retry (supervisor.message_ingress ``_UNDISPATCHED``).

An owner message's accepted row is handed to dispatch by a same-id retry ONLY while this
process holds positive proof that the row's canonical write raised before dispatch was entered.
Everything else is unknown and only rejoins — no second row, no second dispatch: a process that
restarted (it holds no proof), a dispatch that was entered and then raised (it may have queued),
a changed message under that id (refused; the proof stays for the real one), and the losing side
of two retries racing for the one handover. Web frames and named skill deliveries (Host early
rejoin included) follow the same rule. The other half of the same fact: only the process that
accepted a row (its ``ingress_process`` stamp) says its dispatch was entered (``ingress_dispatched``);
after a restart a saved row's delivery is unknown — a crash that kept the host session included.
"""
from __future__ import annotations

import json
import threading

import pytest

from tests.test_chat_attachments import PNG, _history, _inject, _rows, _send_web, _skill_client, _upload, _web_bridge
from tests.test_chat_attachments import files_app as files_app  # noqa: F401 -- fixture re-export


@pytest.fixture(autouse=True)
def _no_proof_leaks():
    from ouroboros import chat_uploads
    from supervisor import message_bus

    message_bus._UNDISPATCHED.clear()
    yield
    message_bus._UNDISPATCHED.clear()
    chat_uploads._PENDING.clear()


def _new_process(monkeypatch):
    """The host process ended and another took over: its proofs and its generation id are gone."""
    from ouroboros import process_custody
    from supervisor import message_bus

    message_bus._UNDISPATCHED.clear()
    monkeypatch.setattr(process_custody, "_SESSION_ID", "the-next-host-process")


def _land_then_fail(monkeypatch):
    from supervisor import message_bus

    real = message_bus.log_chat

    def land_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("the append's acknowledgement was lost")

    monkeypatch.setattr(message_bus, "log_chat", land_then_fail)
    return lambda: monkeypatch.setattr(message_bus, "log_chat", real)


def test_after_a_restart_a_landed_web_row_only_rejoins(files_app, tmp_path, monkeypatch):
    """A new process holds no proof: the row is saved (replayed as such), its delivery unknown —
    a retry rejoins it, and nothing is dispatched or written again."""
    from supervisor import message_bus

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    photo = _upload(files_app, "one.png", PNG)
    restore = _land_then_fail(monkeypatch)
    with pytest.raises(RuntimeError):
        _send_web(bridge, "Посмотри", [photo])
    restore()
    assert message_bus.acceptance_undispatched(1, "cm-1")
    _new_process(monkeypatch)  # the process ended: its proof with it
    (replayed,) = _history(tmp_path)
    assert replayed["ingress_accepted"] is True, "saved"
    assert not {"ingress_undispatched", "ingress_dispatched", "ingress_pending"} & set(replayed), "its delivery unknown"
    _send_web(bridge, "Посмотри", [photo])
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 0, "unknown is never replayed"
    assert echoes[-1]["ingress_accepted"] is True
    assert not {"ingress_undispatched", "ingress_dispatched", "ingress_pending"} & set(echoes[-1])


def test_a_web_dispatch_that_was_entered_then_raised_is_unknown(tmp_path, monkeypatch):
    """The row landed and the enqueue began: it may have queued, so no proof is kept and a retry,
    text-only or not, only rejoins."""
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    real = bridge.enqueue_local_message

    def enqueue_then_fail(text, **message):
        real(text, **message)
        raise RuntimeError("the queue's acknowledgement was lost")

    monkeypatch.setattr(bridge, "enqueue_local_message", enqueue_then_fail)
    with pytest.raises(RuntimeError):
        _send_web(bridge, "только текст", [])
    monkeypatch.setattr(bridge, "enqueue_local_message", real)
    assert "ingress_undispatched" not in _history(tmp_path)[0]
    _send_web(bridge, "только текст", [])
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1, "the possibly queued item is not doubled"
    assert [echo["client_message_id"] for echo in echoes] == ["cm-1"]


def test_a_changed_message_under_a_proven_id_is_refused_and_the_proof_waits(tmp_path, monkeypatch):
    from supervisor import message_bus

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    restore = _land_then_fail(monkeypatch)
    with pytest.raises(RuntimeError):
        _send_web(bridge, "исходные слова", [])
    restore()
    with pytest.raises(ValueError, match="different message"):
        _send_web(bridge, "другие слова", [])
    assert message_bus.acceptance_undispatched(1, "cm-1") and bridge._inbox.qsize() == 0
    assert "ingress_dispatched" not in _history(tmp_path)[0], "proven undispatched is not dispatched"
    assert _history(tmp_path)[0]["ingress_undispatched"] is True and "ingress_pending" not in _history(tmp_path)[0]
    _send_web(bridge, "исходные слова", [])
    assert bridge._inbox.qsize() == 1 and not message_bus.acceptance_undispatched(1, "cm-1")
    assert echoes[-1]["ingress_dispatched"] is True and _history(tmp_path)[0]["ingress_dispatched"] is True, "handed over"


def test_two_racing_retries_hand_the_row_over_once(tmp_path, monkeypatch):
    """The proof is taken under the ingress lock: of two tabs pressing Send again at once (a
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


def test_a_host_retry_after_restart_only_rejoins_and_reads_lost(tmp_path, monkeypatch):
    """Host early rejoin, unknown side: after the host session changed the retry answers the
    rejoin without reaching the ingress, nothing is dispatched, and the operation reads ``lost``."""
    from supervisor import message_bus
    from tests.test_chat_inject_attachments import _skill_file

    bridge = message_bus.LocalChatBridge()
    client = _skill_client(tmp_path, bridge)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "state.json").write_text(json.dumps({"session_id": "before"}), encoding="utf-8")
    body = {"text": "scan", "client_message_id": "tg:1",
            "attachments": [{"path": str(_skill_file(tmp_path, "scan.pdf")), "name": "scan.pdf"}]}
    restore = _land_then_fail(monkeypatch)
    assert _inject(client, **body).status_code == 500
    restore()
    message_bus._UNDISPATCHED.clear()
    (tmp_path / "state" / "state.json").write_text(json.dumps({"session_id": "after"}), encoding="utf-8")
    uploads = sorted(path.name for path in (tmp_path / "uploads").iterdir())
    replay = _inject(client, **body)
    assert replay.status_code == 202 and replay.json()["rejoined"] is True and bridge._inbox.qsize() == 0
    assert sorted(path.name for path in (tmp_path / "uploads").iterdir()) == uploads, "a rejoin copies nothing"
    state = client.get("/chat/operations/42:tg:1", headers={"X-Skill-Token": "token"}).json()
    assert (state["status"], state["reason"]) == ("lost", "host_restarted_before_answer")


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
    assert not message_bus.acceptance_undispatched(42, "tg:2")
    assert _inject(client, text="ping", client_message_id="tg:2").json()["rejoined"] is True
    assert bridge._inbox.qsize() == 1 and len(_rows(tmp_path)) == 1


def test_only_the_process_that_accepted_a_row_says_its_dispatch_was_entered(tmp_path, monkeypatch):
    """The live process's echo and history say ``ingress_dispatched`` for a row it took, rejoins
    included; once it ended neither does, and a retry adds no row and no queue item."""
    from ouroboros.process_custody import current_custody_session_id

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    _send_web(bridge, "слова", [])
    (row,) = _rows(tmp_path)
    assert row["ingress_process"] == current_custody_session_id()
    assert echoes[0]["ingress_dispatched"] is True and _history(tmp_path)[0]["ingress_dispatched"] is True
    _send_web(bridge, "слова", [])
    assert echoes[-1]["ingress_dispatched"] is True and bridge._inbox.qsize() == 1, "its own row, rejoined"
    _new_process(monkeypatch)
    (replayed,) = _history(tmp_path)
    assert replayed["ingress_accepted"] is True and "ingress_dispatched" not in replayed
    assert "ingress_pending" not in replayed, "an ended process's row is unknown, never pending"
    _send_web(bridge, "слова", [])
    assert "ingress_dispatched" not in echoes[-1] and echoes[-1]["ingress_accepted"] is True
    assert "ingress_pending" not in echoes[-1]
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1, "nothing replays"


def test_a_host_operation_from_before_a_crash_reads_lost_though_the_session_survived(tmp_path, monkeypatch):
    """A crash leaves ``state.json``'s session id as it was; the row's process stamp still names the
    ended process, so the operation reads ``lost`` instead of ``pending`` forever, and a retry only
    rejoins."""
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
    """The durable row may be read before log_chat returns to its enqueue caller: that read says the
    running process took it and has not yet said (``ingress_pending``), never dispatched and never the
    ended-process unknown a client would settle for good."""
    from supervisor import message_bus
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    original = message_bus.log_chat
    observations = []
    def append_and_read(*args, **kwargs):
        row = original(*args, **kwargs)
        observations.append((_history(tmp_path)[0], bridge._inbox.qsize()))
        return row
    monkeypatch.setattr(message_bus, "log_chat", append_and_read)
    _send_web(bridge, "during append", [])
    assert observations[0][1] == 0
    assert "ingress_dispatched" not in observations[0][0]
    assert observations[0][0]["ingress_pending"] is True and "ingress_undispatched" not in observations[0][0]
    assert _history(tmp_path)[0]["ingress_dispatched"] is True
    assert "ingress_pending" not in _history(tmp_path)[0]
    assert echoes[-1]["ingress_dispatched"] is True and "ingress_pending" not in echoes[-1]


def test_retention_callback_failure_is_not_dispatch_entry(tmp_path, monkeypatch):
    from supervisor import message_bus, message_ingress
    bridge, _ = _web_bridge(tmp_path, monkeypatch)
    def refuse():
        raise RuntimeError("retention failed before dispatch")
    with pytest.raises(RuntimeError, match="retention failed"):
        message_bus.accept_local_message(bridge, tmp_path, "retain", chat_id=1, user_id=1,
            client_message_id="retention-failure", source="web", retain_inputs=refuse)
    row = message_bus.accepted_chat_message(tmp_path, 1, "retention-failure")
    assert row and message_ingress.accepted_here(row)
    assert not message_ingress.dispatch_entered(row)
    assert bridge._inbox.qsize() == 0


def test_deferred_dispatch_echo_does_not_claim_future_call(tmp_path, monkeypatch):
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
