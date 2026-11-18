"""Fixtures for addressed question mail and Continue from an interrupted task."""
import queue
from types import SimpleNamespace

from ouroboros.task_store import patch_document


def drain(root, task_id):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    ctx = SimpleNamespace(task_id=task_id, task_metadata={}, drive_root=root, budget_drive_root=root)
    messages = []
    _drain_incoming_messages(messages, queue.Queue(), root, task_id, None, set(), owner_ctx=ctx)
    return ctx, messages


def stage_question_mail(root, task_id, events=None):
    """Deliver into the normal transcript, leaving the ordinary ACK to its next successful save."""
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from tests._budget_pause_exact_helpers import _loop_ctx

    ctx, limit = _loop_ctx(root, task_id)
    limit.messages[:] = [{"role": "user", "content": "Resume."}]
    _drain_incoming_messages(limit.messages, queue.Queue(), root, task_id, events, limit.owner_msg_seen,
                             owner_ctx=ctx, defer_content_ack=True)
    return ctx, limit


def continue_to(root, state="admitted", *, asker="asker", successor="successor"):
    from ouroboros.owner_continue import binding_sha
    from ouroboros.task_results import write_task_result
    write_task_result(root, successor, status="running", chat_id=1)
    binding = {"predecessor_task_id": asker, "successor_task_id": successor, "action_nonce": "nonce"}
    admission = {"binding": binding, "binding_sha256": binding_sha(binding)}
    patch_document(root, asker, lambda row: {**row, "continued_by": {
        **admission, "state": state, "successor_task_id": successor}})
    patch_document(root, successor, lambda row: {**row, "continuation_admission": admission})


def public_base_closure(root, reason, *, event_id="q"):
    """Produce the old writer's untyped letter on its asker, with the old card's route fields."""
    from ouroboros.owner_mailbox import KIND_QUIZ_CLOSED, write_owner_message
    from ouroboros.owner_quiz import mark_quiz_history_written, record_closed

    block = record_closed(root, "asker", quiz_id="q", actor="owner", reason=reason, request_id="close")["block"]
    # Neither field existed in the public producer. An old reason's bytes alone are not a directive.
    def old_card(row):
        card = row["owner_quiz"]["q"]
        card.pop("owner_wrote_reason", None)
        card["delivery"].pop("first_route_pending")
        return row

    patch_document(root, "asker", old_card)
    frame = (f"[Owner question closed without answer] quiz {event_id} — closed {block['closed_at']}. Reason: {reason}\n"
             "This closes the question; no option was chosen and no approval was given.")
    assert write_owner_message(root, frame, "asker", msg_id="quiz_closed:q", kind=KIND_QUIZ_CLOSED)
    assert mark_quiz_history_written(root, "asker", "q", receipt="mailbox_at")


def scheduled_closure_reader(root, monkeypatch):
    """An installed test queue through which the real explicit Continue can admit its successor."""
    from ouroboros.owner_quiz import record_asked, record_publication
    from ouroboros.task_results import write_task_result
    from tests._budget_pause_exact_helpers import _install_queue

    q, _, _ = _install_queue(root, monkeypatch)
    task = {"id": "asker", "chat_id": 7, "drive_root": str(root)}
    q.RUNNING["asker"] = {"task": task}
    write_task_result(root, "asker", status="running", chat_id=7)
    record_asked(root, "asker", quiz_id="q", question="Keep the old job?", options=["Keep", "Stop"], chat_id=7)
    record_publication(root, "asker", "q", 7)
    return q, _late_bridge(root, monkeypatch)


NONCE = "press-0001-abcdef"


def _interrupted(tmp_path, task_id="pred-1", *, origin=None, reason_code="", **extra):
    from ouroboros.task_results import STATUS_CANCELLED, STATUS_RUNNING, write_task_result

    from ouroboros.usage_admission import original_group_limit
    prior = original_group_limit(tmp_path, task_id)
    initial = {} if prior["source"] == "ledger_first_row" else {"billing_group": {
        "billing_group_id": task_id, "billing_group_limit_usd": 20.0,
        "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "admission-1"}}
    write_task_result(tmp_path, task_id, STATUS_RUNNING, chat_id=7, project_id="", **initial,
                      origin_message_text="Write the Friday report",
                      origin_message_ref={"chat_id": 7, "client_message_id": "m-1"},
                      root_task_id=task_id, title="Friday report", deadline_at="2099-01-01T00:00:00+00:00")
    fields = {"cancel_origin": origin if origin is not None else {"source": "snapshot_restore",
                                                                   "reason": "server_shutdown"}}
    if reason_code:
        fields = {"reason_code": reason_code}
    write_task_result(tmp_path, task_id, STATUS_CANCELLED if not reason_code else "failed",
                      result="interrupted", **fields, **extra)


def _late_bridge(tmp_path, monkeypatch):
    """A real bridge and explicit fixture publication witnesses for named late delivery."""
    import supervisor.message_bus as mb
    from ouroboros.owner_quiz import quiz_states, record_publication
    from ouroboros.task_store import task_facts

    for task_id in task_facts(tmp_path):
        for quiz_id, block in quiz_states(tmp_path, task_id).items():
            if "chat_id" in block:
                record_publication(tmp_path, task_id, quiz_id, block["chat_id"])

    bridge = mb.LocalChatBridge()
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(mb, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mb, "_BRIDGE", bridge)
    return bridge, frames


def runtime(tmp_path, monkeypatch, *, running=True, asker="asker"):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from ouroboros.gateway import task_decision as td
    from ouroboros.owner_quiz import record_asked, record_publication
    from supervisor import queue as q
    task = {"id": asker, "chat_id": 1, "drive_root": str(tmp_path)}
    monkeypatch.setattr(q, "RUNNING", {asker: {"task": task}} if running else {})
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(q, "_task_drive_for_task", lambda task, tid: tmp_path)
    monkeypatch.setattr(td, "request_drive_root", lambda request: tmp_path)
    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, asker, status="running" if running else "completed", chat_id=1)
    record_asked(tmp_path, asker, quiz_id="q", question="Which database?", options=["sqlite", "postgres"],
                 assumption="sqlite", chat_id=1)
    record_publication(tmp_path, asker, "q", 1)
    bridge, frames = _late_bridge(tmp_path, monkeypatch)
    app = Starlette(routes=[Route("/api/decisions", endpoint=td.api_decision_answer, methods=["POST"])])
    return app, bridge, q, task


def _post(app, body):
    from starlette.testclient import TestClient

    return TestClient(app).post("/api/decisions", json=body)
