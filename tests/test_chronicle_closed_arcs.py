"""Only actual closed task/turn sources enter helper episodic reconstruction."""
import json

import pytest

from ouroboros import consolidator as c
from ouroboros.chronicle_store import source_row_id
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.tools.registry import ToolContext
from tests.test_chronicle_consolidation import Helper, setup


def owner_row(chat, ident, text):
    return {"chat_id": chat, "client_message_id": ident, "direction": "in", "task_id": None,
            "ts": "2026-09-30T01:00:00", "text": text}


def source_ref(row):
    return build_owner_message_ref(chat_id=row["chat_id"], client_message_id=row["client_message_id"],
        ts=row["ts"], text=row["text"])


def facts(task, chat):
    return {"chat_id": chat, "task_id": task, "direction": "system", "type": "task_summary",
            "summary_kind": "host_task_facts", "outcome_final": False,
            "outcome_authority": "pre_finalization_host_facts", "text": "", "ts": "2026-09-30T01:01:00"}


@pytest.mark.parametrize("same_room", [False, True])
def test_finishing_b_does_not_close_waiting_a_even_in_same_room(tmp_path, monkeypatch, same_room):
    room_b = 1 if same_room else 22
    a = owner_row(1, "question-a", "Please wait for my decision on A.")
    b = owner_row(room_b, "question-b", "Complete B.")
    waiting = {"chat_id": 1, "task_id": "a", "direction": "out", "text": "A waits for its owner.", "ts": "2026-09-30T01:00:10"}
    answer = {"chat_id": room_b, "task_id": "b", "direction": "out", "text": "B is finished.", "ts": "2026-09-30T01:00:20"}
    closed_fact = facts("b", room_b)
    store, _ctx, chat, blocks, meta = setup(tmp_path, [a, b, waiting, answer, closed_fact])
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    context = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="b")
    # Current actor identity alone, including inline pressure, is not closure.
    c.consolidate(chat, blocks, meta, None, knowledge_context=context)
    assert helper.calls == []
    finished = {"id": "b", "chat_id": room_b, "origin_message_ref": source_ref(b)}
    c.consolidate(chat, blocks, meta, None, knowledge_context=context, completed_task=finished)
    rows = store.records(kinds=["episode"])
    assert len(rows) == 1
    assert set(rows[0]["metadata"]["source_row_ids"]) == {source_row_id(row) for row in [b, answer, closed_fact]}
    assert "Please wait for my decision on A." not in helper.calls[0][0]
    assert not store.scan_state().get("last_consolidated_offset")
    count = len(helper.calls)
    c.consolidate(chat, blocks, meta, None, knowledge_context=context, completed_task=finished)
    assert len(helper.calls) == count  # closed B is retained behind the open A gap

    week_later = {**facts("c", 33), "ts": "2026-10-07T01:00:00"}
    with chat.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(week_later) + "\n")
    c.consolidate(chat, blocks, meta, None, knowledge_context=context,
        completed_task={"id": "c", "chat_id": 33})
    covered = {ident for row in store.records(kinds=["episode"]) for ident in row["metadata"]["source_row_ids"]}
    assert source_row_id(a) not in covered and source_row_id(waiting) not in covered
    assert not store.scan_state().get("last_consolidated_offset")


def test_canonical_terminal_fact_closes_normal_main_turn_without_current_actor_shortcut(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    owner = owner_row(1, "owner-main", "Finish this normal task.")
    answer = {"chat_id": 1, "task_id": "finished", "direction": "out", "text": "The normal task finished."}
    terminal = {**facts("finished", 1), "summary_kind": "terminal_root_projection", "outcome_final": True,
        "outcome_authority": "canonical_task_result_after_finalization", "status": "completed", "outcome_phase": "done"}
    later = owner_row(1, "later-question", "A later question remains open.")
    store, ctx, chat, blocks, meta = setup(tmp_path, [owner, answer, terminal, later])
    write_task_result(tmp_path, "finished", STATUS_COMPLETED, origin_message_ref=source_ref(owner))
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    c.consolidate(chat, blocks, meta, None, knowledge_context=ctx)
    covered = store.records(kinds=["episode"])[0]["metadata"]["source_row_ids"]
    assert set(covered) == {source_row_id(row) for row in [owner, answer, terminal]}
    assert source_row_id(later) not in covered
    assert store.scan_state()["last_consolidated_offset"] == 3


def test_first_post_task_activation_closes_only_its_completed_source(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import post_task_synthesis
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.memory import Memory

    owner = owner_row(1, "finished-owner", "Finish this one task.")
    open_owner = owner_row(1, "still-open", "Wait for my next decision.")
    answer = {"chat_id": 1, "task_id": "finished", "direction": "out", "text": "Finished."}
    rows = [owner, answer, facts("finished", 1), open_owner]
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    store = ChronicleStore(tmp_path)
    assert not store.log_path.exists()
    helper = Helper()
    def free_helper(*args, **kwargs):
        content, usage, knowledge = helper(*args, **kwargs)
        return content, {**usage, "cost": 0, "prompt_tokens": 0}, knowledge
    monkeypatch.setattr(c, "_light_call", lambda *_args: free_helper)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda path: tmp_path / path)
    result = post_task_synthesis._run_chat_consolidation(env, Memory(tmp_path, tmp_path), None,
        {"id": "finished", "chat_id": 1, "origin_message_ref": source_ref(owner)}, logs)
    assert result == ""
    assert store.activation()
    (episode,) = store.records(kinds=["episode"])
    assert set(episode["metadata"]["source_row_ids"]) == {source_row_id(row) for row in rows[:3]}
    assert source_row_id(open_owner) not in episode["metadata"]["source_row_ids"]
    assert store.scan_state()["last_consolidated_offset"] == 3


def test_host_notice_and_settled_command_advance_scan_but_ack_does_not(tmp_path, monkeypatch):
    from supervisor.message_bus import log_chat
    from ouroboros.task_finalization import host_operation_reply_kwargs

    store, ctx, chat, blocks, meta = setup(tmp_path)
    rows = []
    def emit(direction, text, **kwargs):
        row = log_chat(direction, 1, 1, text, drive_root=tmp_path, require_write=True, **kwargs)
        rows.append(row)
        return row
    emit("system", "Evolution cycle queued, review remains open.", record_type="evolution_notice")
    command = emit("in", "Restart please.", client_message_id="restart")
    correlation = host_operation_reply_kwargs(source_ref(command), "completed")
    emit("system", "Restart request handled.", record_type="command_reply",
         message_meta=correlation["progress_meta"])
    waiting = emit("in", "Please wait for my decision.", client_message_id="waiting")
    correlation = host_operation_reply_kwargs(source_ref(waiting))
    emit("system", "Request received, not finished.", record_type="command_reply",
         message_meta=correlation["progress_meta"])
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    c.consolidate(chat, blocks, meta, None, knowledge_context=ctx)
    represented = set(store.records(kinds=["episode"])[0]["metadata"]["source_row_ids"])
    assert represented == {source_row_id(row) for row in rows if row is not waiting}
    assert store.scan_state()["last_consolidated_offset"] == 3
    calls = len(helper.calls)
    c.consolidate(chat, blocks, meta, None, knowledge_context=ctx)
    assert len(helper.calls) == calls


@pytest.mark.parametrize("delivery_status", ["delivered", "unconfirmed"])
def test_mailbox_followup_closes_only_with_delivered_target_boundary(tmp_path, monkeypatch, delivery_status):
    from ouroboros.project_dialogue import append_chat_annotation

    owner = owner_row(1, "follow-up", "Use the corrected requirement.")
    terminal = facts("finished", 1)
    store, ctx, chat, blocks, meta = setup(tmp_path, [owner, terminal])
    assert append_chat_annotation(tmp_path, "follow-up", action="steer_task",
                                  target="finished", status=delivery_status)
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    c.consolidate(chat, blocks, meta, None, knowledge_context=ctx,
                  completed_task={"id": "finished"})
    represented = set(store.records(kinds=["episode"])[0]["metadata"]["source_row_ids"])
    assert (source_row_id(owner) in represented) == (delivery_status == "delivered")
    assert source_row_id(terminal) in represented


@pytest.mark.parametrize("composer", [False, True])
@pytest.mark.parametrize("command", ["/bg start", "/bg stop", "/evolve stop", "/status", "/restart", "/review"])
def test_real_command_ingress_closes_only_settled_operation_or_review_task(tmp_path, monkeypatch, composer, command):
    from types import SimpleNamespace
    import server
    from ouroboros import server_restart
    from ouroboros.task_results import write_task_result
    from supervisor import events_runtime_controls, message_bus, queue, state
    from tests.test_transport_commands import Ctx

    store, ctx, chat, blocks, meta = setup(tmp_path)
    bridge, live, queued, exits = message_bus.LocalChatBridge(), [], [], []
    bridge._broadcast_fn = live.append
    command_ctx = Ctx({"owner_id": 1, "owner_chat_id": 1})
    command_ctx.WORKERS, command_ctx.PENDING, command_ctx.RUNNING = {}, [], {}
    command_ctx.consciousness = SimpleNamespace(start=lambda: "Enabled", stop=lambda: "Disabled")
    command_ctx.send_with_budget = message_bus.send_with_budget
    command_ctx.safe_restart = object()
    command_ctx.queue_deep_self_review_task = queue.queue_deep_self_review_task
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", command_ctx.load_state)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_args: None)
    monkeypatch.setattr(state, "status_text", lambda *_args: "Runtime status")
    monkeypatch.setattr(events_runtime_controls, "persist_consciousness_choice", lambda _on: "")
    monkeypatch.setattr(server, "_owner_evolution_stop", lambda *_args: "OFF", raising=False)
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server_restart, "_safe_restart_serialized", lambda *_args, **_kw: (True, ""))
    monkeypatch.setattr(server_restart, "_stop_owned_work", lambda *_args: [])
    monkeypatch.setattr(server_restart, "_request_restart_exit", lambda **kw: exits.append(kw))
    monkeypatch.setattr(queue, "load_state", command_ctx.load_state)
    monkeypatch.setattr(queue, "enqueue_task", lambda task: (queued.append(task) or task))
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_kw: None)
    monkeypatch.setattr(queue, "send_with_budget", message_bus.send_with_budget)
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    # Same real bridge entrypoint used by WS chat versus header/settings command frames.
    bridge.ui_send(command, broadcast=composer, client_message_id="command-owner")
    server._process_bridge_updates(bridge, 0, command_ctx)
    rows = [json.loads(line) for line in chat.read_text(encoding="utf-8").splitlines()]
    owner = next(row for row in rows if row["direction"] == "in")
    reply = rows[-1]
    assert reply["origin_message_ref"]["client_message_id"] == owner["client_message_id"]
    delivered = [row for row in live if row.get("type") == "chat" and row.get("role") == "system"]
    assert delivered and delivered[-1]["origin_message_ref"] == reply["origin_message_ref"]
    assert not delivered[-1].get("task_id")  # never an invented completed task card
    if command == "/review":
        assert queued and queued[0]["metadata"]["origin_message_ref"] == reply["origin_message_ref"]
        assert "task_terminal_status" not in reply and reply["type"] == "deep_self_review_queued"
        c.consolidate(chat, blocks, meta, None, knowledge_context=ctx)
        covered = {rid for ep in store.records(kinds=["episode"]) for rid in ep["metadata"]["source_row_ids"]}
        assert source_row_id(owner) not in covered and not store.scan_state().get("last_consolidated_offset")
        tid = queued[0]["id"]
        write_task_result(tmp_path, tid, "completed", metadata=queued[0]["metadata"])
        terminal = {**facts(tid, 1), "summary_kind": "terminal_root_projection", "outcome_final": True,
                    "outcome_authority": "canonical_task_result_after_finalization", "status": "completed", "outcome_phase": "done"}
        with chat.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(terminal) + "\n")
        rows.append(terminal)
    else:
        assert reply["task_terminal_status"] == "completed"
        if command == "/restart":
            assert not rows[-2].get("task_terminal_status")
            assert "Restart confirmed." in reply["text"] and "restarted" not in reply["text"].lower()
            assert exits == [{"owner": True}]  # effect stub, never an actual process restart
            assert (tmp_path / "state/owner_restart_no_resume.flag").read_text(encoding="utf-8") == "owner_restart"
    c.consolidate(chat, blocks, meta, None, knowledge_context=ctx)
    covered = {rid for ep in store.records(kinds=["episode"]) for rid in ep["metadata"]["source_row_ids"]}
    assert source_row_id(owner) in covered
    assert store.scan_state()["last_consolidated_offset"] == len(rows)


def test_real_review_refusal_and_routing_picker_close_without_resolving_ambiguous_owner(tmp_path, monkeypatch):
    import server
    from supervisor import message_bus, queue
    from tests.test_transport_commands import Ctx

    store, ctx, chat, blocks, meta = setup(tmp_path)
    bridge = message_bus.LocalChatBridge()
    command_ctx = Ctx({"owner_id": 1, "owner_chat_id": 1})
    command_ctx.send_with_budget = message_bus.send_with_budget
    command_ctx.queue_deep_self_review_task = queue.queue_deep_self_review_task
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", command_ctx.load_state)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_args: None)
    monkeypatch.setattr(queue, "load_state", command_ctx.load_state)
    monkeypatch.setattr(queue, "enqueue_task", lambda task: {**task, "_admission_blocked": "pool_disabled"})
    monkeypatch.setattr(queue, "send_with_budget", message_bus.send_with_budget)
    bridge.ui_send("/review", broadcast=False)
    server._process_bridge_updates(bridge, 0, command_ctx)
    rows = [json.loads(line) for line in chat.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["task_terminal_status"] == "failed" and rows[-1]["origin_message_ref"]
    waiting = message_bus.log_chat("in", 1, 1, "Please choose with me later.", client_message_id="open", drive_root=tmp_path)
    picker = message_bus.log_chat("out", 1, 0, "Choose a destination.", source="routing_picker",
                                 record_type="routing_options", drive_root=tmp_path)
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    c.consolidate(chat, blocks, meta, None, knowledge_context=ctx)
    covered = {rid for ep in store.records(kinds=["episode"]) for rid in ep["metadata"]["source_row_ids"]}
    assert source_row_id(rows[0]) in covered and source_row_id(picker) in covered
    assert source_row_id(waiting) not in covered
    assert store.scan_state()["last_consolidated_offset"] == 2
