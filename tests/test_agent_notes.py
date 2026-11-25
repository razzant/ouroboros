"""Normal program notes and later replies, using the real local receipt and bridge without a network listener."""
from __future__ import annotations

import asyncio
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import httpx
from ouroboros import dialogue_provenance, extension_late_result, place_intake
from ouroboros.context import build_user_content
from ouroboros.conversation_lease import owner_conversation_key
from ouroboros.gateway import agent_note, host_service
from ouroboros.task_results import write_task_result
from ouroboros.tools.tool_context import ToolContext
from supervisor import message_bus
from supervisor.message_ingress import accepted_chat_message


def _serving(ctx):
    """The fictional local route whose answered call ran the sending tool."""
    ctx._current_llm_call_meta = {"model": "local-example", "physical_model": "local-example (local)",
                                  "physical_provider": "local", "dispatch_account": "example-local-binding",
                                  "use_local": True, "processing_preference": ""}  # "" is the explicit Default
    return ctx


def _room(tmp_path, monkeypatch):
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    bridge = message_bus.LocalChatBridge()
    frames, opened, owner_updates = [], [], []
    bridge._broadcast_fn = frames.append
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], BUDGET_ROOT_FENCES={},
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
        update_state=lambda fn: owner_updates.append(fn),
        handle_chat_direct=lambda chat, text, images, **metadata: opened.append((chat, text, metadata)),
    )
    return bridge, ctx, frames, opened, owner_updates


def test_main_note_opens_an_ordinary_conversation_without_owner_authority(tmp_path, monkeypatch):
    import server
    from ouroboros import settings_integrity

    bridge, ctx, frames, opened, owner_updates = _room(tmp_path, monkeypatch)
    monkeypatch.setattr(settings_integrity, "effective_subagents_mode", lambda **_: "swarm")
    original = "  The example comparison is ready.\n  Проверка: всё готово.\n"
    text, author = agent_note.note_request({"text": original,
                                           "author": {"kind": "agent", "label": "Build agent"}})
    row = agent_note._accept(bridge, tmp_path, text, author, {})
    server._process_bridge_updates(bridge, 0, ctx, timeout=0)
    assert owner_updates == []
    assert row["text"] == "Program message received." and row["role"] == "system"
    assert row["system_type"] == "agent_note_received" and "subagents_mode" not in row
    assert dialogue_provenance.agent_note_text(row) == original
    assert not dialogue_provenance.owner_speech(row, dialogue_provenance.row_author(row))
    echo, = [item for item in frames if item.get("type") == "chat"]
    assert echo["sender_label"] == row["sender_label"] and echo["content"] == original
    assert "subagents_mode" not in echo and echo["ingress_accepted"] is True
    assert len(opened) == 1
    chat, accepted, ingress = opened[0]
    assert chat == 1 and accepted == original
    task = {"id": "fedcba98", "chat_id": chat, "text": accepted, "type": "task", "_is_direct_chat": True,
            "metadata": ingress["task_metadata"], "admission_author": ingress["admission_author"]}
    assert dialogue_provenance.run_origin(task)["owner_ingress"] is False
    assert owner_conversation_key(task) == "owner:1"
    content = build_user_content(task)
    assert content.endswith("\n" + original) and '"Build agent", not verified' in content and "not my human's words" in content

    bridge.handle_web_message("  An ordinary owner reply.\n", client_message_id="example-owner-reply")
    owner_row = accepted_chat_message(tmp_path, 1, "example-owner-reply")
    assert owner_row["text"] == "An ordinary owner reply." and owner_row["subagents_mode"] == "swarm"
    assert frames[-1]["client_message_id"] == "example-owner-reply" and frames[-1]["subagents_mode"] == "swarm"


def test_late_report_after_finish_keeps_the_original_allowance(tmp_path, monkeypatch):
    _check_local_return(tmp_path, monkeypatch, "finished")


def test_late_report_during_project_finish_keeps_one_consideration_owed(tmp_path, monkeypatch):
    _check_local_return(tmp_path, monkeypatch, "closing")


def test_late_report_reaches_one_live_project_reader(tmp_path, monkeypatch):
    _check_local_return(tmp_path, monkeypatch, "live")


def test_late_report_rejoins_the_decided_handoff_after_a_busy_receipt_write(tmp_path, monkeypatch):
    _check_local_return(tmp_path, monkeypatch, "finished", busy_receipt=True)


def _check_local_return(tmp_path, monkeypatch, when, *, busy_receipt=False):
    import server
    from ouroboros import dialogue_store
    from ouroboros.conversation_lease import follow_loop_admission, open_root_intake
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.model_wait import EXECUTION_CLOSE_KEY
    from ouroboros.projects_registry import create_project

    bridge, ctx, frames, opened, owner_updates = _room(tmp_path, monkeypatch)
    def remember_turn(chat, text, _images, **metadata):
        opened.append((chat, text, metadata))
        return {"task": {"id": "fedcba98"}}
    ctx.handle_chat_direct = remember_turn
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "state.json").write_text("{}", encoding="utf-8")
    origin = "abcd1234"
    project_id = "letter-room" if when != "finished" else ""
    room = create_project(tmp_path, project_id)["chat_id"] if project_id else 1
    billing = {"billing_group_id": origin, "billing_group_limit_usd": 2,
               "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "2026-11-20T00:00:00Z"}
    contract = {"constraints": "Use the local room only.", "disabled_tools": ["browse_page"]}
    write_task_result(tmp_path, origin, "running", chat_id=room, project_id=project_id,
                      root_task_id=origin, billing_group=billing, task_contract=contract, task_attempt=1,
                      **({"execution_timeout_sec": 90.0} if when == "finished" else {}))
    task = {"id": origin, "root_task_id": origin, "chat_id": room, "project_id": project_id, "drive_root": str(tmp_path)}
    ctx.RUNNING[origin] = {"task": task, "attempt": 1}
    assert open_root_intake(tmp_path, task)
    tool_ctx = _serving(ToolContext(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path,
                                    task_id=origin, task_lifecycle_bound=True))
    call_ref = extension_late_result.mint(tool_ctx, {"skill": "example-letters", "content_hash": "example-build"})
    assert call_ref

    def finish_work():
        write_task_result(tmp_path, origin, "completed", result="Example work finished.", terminal_origin="model_final",
                          outcome_axes={"execution": {"status": "ok"}},
                          **({EXECUTION_CLOSE_KEY: {"sec": 30.0, "attempt": 1}} if when == "finished" else {}))

    if when == "finished":
        finish_work()
    elif when == "closing":
        follow_loop_admission(SimpleNamespace(drive_root=tmp_path, task_id=origin, task_attempt=1, task_metadata={}), False)
    app = host_service.create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    # The fixture supplies an already authenticated connector, not an implementation of token lookup.
    local = app.state.host_service_context
    local.authenticate_token_payload = lambda _token: ("example-letters", {"content_hash": "example-build"})
    local.require_permission = lambda *_args: None
    request = {"call_ref": call_ref, "report_id": "letter-42", "label": "Example collaborator",
               "text": "  Here is the requested example summary.\n  Проверка: всё готово.\n"}

    async def post_report():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://example.invalid") as client:
            return await client.post("/late_result", json=request)

    first = asyncio.run(post_report())
    assert first.status_code == 429 and first.json()["code"] == "retry"
    report = extension_late_result.report_input(request)[0]
    with monkeypatch.context() as settle:
        if busy_receipt:
            from ouroboros.store import StoreBusy

            def unavailable(*args, **kwargs):
                raise StoreBusy("receipt writer busy", reason="contended")

            settle.setattr(dialogue_store, "complete_dispatch", unavailable)
        server._process_bridge_updates(bridge, 0, ctx, timeout=0)
        if busy_receipt:
            assert dialogue_store.dispatch_state(tmp_path, room, report) == "dispatched"
            assert extension_late_result.binding_for_report(tmp_path, report)["handoff"]["outcome"] == "handed_off"
            assert len(opened) == 1
    if when == "closing":
        assert dialogue_store.dispatch_state(tmp_path, room, report) == "pending" and opened == []
        finish_work()
        assert asyncio.run(post_report()).status_code == 429  # resumed obligation has not yet reached the lane
        server._process_bridge_updates(bridge, 0, ctx, timeout=0)
    receipt = asyncio.run(post_report())
    assert receipt.status_code == 200 and receipt.json() == {"ok": True, "report_ref": report, "delivery": "handed_off"}
    server._process_bridge_updates(bridge, 0, ctx, timeout=0)
    row = accepted_chat_message(tmp_path, room, report)
    assert row["late_report_id"] == "letter-42" and dialogue_provenance.agent_note_text(row) == request["text"]
    assert call_ref not in str(row)
    assert owner_updates == []
    echoes = [item for item in frames if item.get("late_report_id") == "letter-42"]
    assert echoes and all(item["chat_id"] == room and item["content"] == request["text"] for item in echoes)
    if project_id:
        assert accepted_chat_message(tmp_path, 1, report) is None
        assert row["kind"] == "room_echo" and all(item["kind"] == "room_echo" for item in echoes)
    if when == "live":
        messages, seen = [], set()
        controls = _drain_incoming_messages(messages, queue.Queue(), tmp_path, origin, None, seen, owner_ctx=tool_ctx)
        assert not controls.get("agent_note") and opened == []  # staged for its own request, not a transcript row
        assert messages == []  # the ordinary transcript cannot carry the letter to another route
        assert "model_inclusion" not in extension_late_result.binding_for_report(tmp_path, report)
        reader = SimpleNamespace(tools=SimpleNamespace(_ctx=tool_ctx), task_id=origin, drive_root=tmp_path,
                                 active_model="local-example", active_use_local=True)
        letter = extension_late_result.take_letter_send(reader, role="main", account=None)
        assert letter is not None and request["text"] in letter.frame
        from ouroboros.physical_attempt_context import PhysicalAttemptContext, with_dispatch_account
        from ouroboros.usage_accounting import AttemptRequest

        measured = PhysicalAttemptContext("owner_max", "max", "cold_estimate", "", "example-round",
                                          None, None, False, False)
        candidate = {"messages": [{"role": "user", "content": letter.frame}]}
        reservation = SimpleNamespace(attempt_id="attempt-note")
        bare = AttemptRequest(model="local-example (local)", provider="local", physical_context=measured,
                              candidate_raw_sha256="example-wire-sha")
        assert not letter.admit(bare, candidate, reservation)  # no serving-instance witness yet
        from dataclasses import replace

        physical = replace(bare, physical_context=with_dispatch_account(measured, "example-local-binding"))
        assert letter.admit(physical, candidate, reservation)
        assert "model_inclusion" not in extension_late_result.binding_for_report(tmp_path, report)
        finished = SimpleNamespace(state="settled", attempt_id="attempt-note", candidate_raw_sha256="example-wire-sha")
        assert letter.observe(finished, {"ledger_attempt_ids": ["attempt-note"]})
        assert extension_late_result.binding_for_report(tmp_path, report)["model_inclusion"]["attempt_id"] == "attempt-note"
        assert not letter.admit(physical, candidate, SimpleNamespace(attempt_id="another-round"))
        assert not _drain_incoming_messages(messages, queue.Queue(), tmp_path, origin, None, seen, owner_ctx=tool_ctx).get("agent_note")
        return
    assert len(opened) == 1
    chat, text, arguments = opened[0]
    assert (chat, text) == (room, request["text"])
    assert arguments["task_metadata"]["billing_group"] == billing
    assert arguments["admission_author"]["late_report_ref"] == report
    from supervisor.worker_chat_lane import _seed_task, _verify_late_turn, prepare_conversation_task

    task = _seed_task(chat, text, None, None, arguments["task_metadata"], arguments["admission_author"])
    _verify_late_turn(tmp_path, task, report, admit=True)
    prepare_conversation_task(None, task, tmp_path)
    if when == "finished":
        assert task["execution_timeout_sec"] == 60.0
        from dataclasses import asdict
        from ouroboros.physical_attempt_context import PhysicalAttemptContext, with_dispatch_account
        from ouroboros.usage_accounting import AttemptRequest

        tool_ctx.task_id, tool_ctx.task_metadata = task["id"], task["metadata"]
        tool_ctx._late_report_ref = report
        tool_ctx._late_result_route = task["metadata"]["_late_result_route"]
        admit = extension_late_result.return_candidate(
            SimpleNamespace(tools=SimpleNamespace(_ctx=tool_ctx), active_use_local=True), None)
        measured = PhysicalAttemptContext("owner_max", "max", "cold_estimate", "", "example-round",
                                          None, None, False, False)
        private = with_dispatch_account(measured, "example-local-binding")
        assert "dispatch_account" not in asdict(private)
        physical = AttemptRequest(model="local-example (local)", provider="local", physical_context=private)
        # Preparation only previews the inherited window. Before the normal RUNNING path
        # claims it, neither candidate may send, even on the same recorded route.
        assert admit(physical) is False
        assert admit(physical) is False  # preparing again still does not start the turn
    assert task["task_contract"]["allowed_resources"]["network"] is False
    content = build_user_content(task)
    assert content.endswith("\n" + request["text"]) and '"Example collaborator", not verified' in content and origin in content


def test_late_report_of_an_inline_reply_rejoins_the_saved_tool_result(tmp_path, monkeypatch):
    from ouroboros import working_checkpoint
    from ouroboros.gateway.host_late_result import _accept
    from ouroboros.tools.tool_result import ToolResult

    bridge, _, _, opened, _ = _room(tmp_path, monkeypatch)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "state.json").write_text("{}", encoding="utf-8")
    origin = "aabb1234"
    write_task_result(tmp_path, origin, "running", chat_id=1, root_task_id=origin)
    ctx = _serving(ToolContext(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path,
                               task_id=origin, task_lifecycle_bound=True))
    route = {"skill": "example-letters", "content_hash": "example-build"}
    ref = extension_late_result.mint(ctx, route, await_tool_result=True)
    body = {"late_answer": {"report_id": "letter-43", "label": "Lio", "text": "  Original example reply.\n"}}
    producer = json.dumps(body)
    fact = extension_late_result.inline_result_fact(ref, body, producer)
    report = extension_late_result.report_input({**body["late_answer"], "call_ref": ref})
    result = ToolResult(status="ok", code="OK", text=producer, meta={"inline_late_result": fact})
    transcript = [{"role": "tool", "tool_call_id": "sample-send", "content": producer}]
    extension_late_result.acknowledge_inline_result(ctx, result, {"result": producer})
    assert extension_late_result.binding_for_report(tmp_path, report[0])["inline"] == {"state": "pending"}
    working_checkpoint.save_ready(SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), messages=transcript, llm_trace={},
                                                 accumulated_usage={}, owner_msg_seen=set(), round_idx=1, tool_schemas=[]))
    service = SimpleNamespace(data_dir=tmp_path, bridge_getter=lambda: bridge)
    assert _accept(service, route["skill"], route["content_hash"], *report) == "tool_result"
    assert accepted_chat_message(tmp_path, 1, report[0]) is None and opened == []


def test_completed_annotated_send_keeps_a_no_answer_receipt_with_a_partial_view(tmp_path, monkeypatch):
    """A genuine tool completion needs result delivery evidence only for a present answer."""
    from ouroboros import working_checkpoint
    from ouroboros.tools.extension_dispatch import _extension_completion

    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "state.json").write_text("{}", encoding="utf-8")
    origin = "aabb3456"
    write_task_result(tmp_path, origin, "running", chat_id=1, root_task_id=origin)
    ctx = _serving(ToolContext(repo_dir=Path(__file__).resolve().parents[1], drive_root=tmp_path,
                               task_id=origin, task_lifecycle_bound=True))
    descriptor = {"skill": "example-letters", "content_hash": "example-build"}
    monkeypatch.setattr(extension_late_result, "eligible", lambda *_args: True)
    with extension_late_result.with_call_ref(ctx):
        extension_late_result.bind_call_ref(ctx, "example_send", descriptor, {})
        ref = extension_late_result.call_ref(ctx)
        producer = json.dumps({"ok": True, "details": "Ordinary process notes. " * 500, "late_answer": None})
        result = _extension_completion(producer, "The request was allowed with a host annotation.", ctx=ctx)
    assert ref and result.code == "SAFETY_WARNING" and result.producer_text == producer
    extension_late_result.acknowledge_inline_result(ctx, result, {"result": producer[:80], "result_partial": True})
    working_checkpoint.save_ready(SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), messages=[], llm_trace={},
                                                 accumulated_usage={}, owner_msg_seen=set(), round_idx=1, tool_schemas=[]))
    assert extension_late_result.binding_for_report(tmp_path, extension_late_result._report_id(ref))["inline"] == {"state": "none"}


def test_wire_letter_reads_ordinary_chat_and_responses_user_text():
    """Plain request data only: the intact note counts in a provider's user input, in either supported wire shape."""
    letter = "[Local reply to a request from task example]\nThe example summary is ready."
    for candidate in ({"messages": [{"role": "system", "content": "Example rules."}, {"role": "user", "content": letter}]},
                      {"messages": [{"role": "user", "content": [{"type": "text", "text": "Earlier.\n" + letter}]}]},
                      {"input": [{"role": "user", "content": [{"type": "input_text", "text": letter}]}]},
                      {"input": [{"role": "user", "content": letter}]}):
        assert extension_late_result._wire_has_letter(candidate, letter)
    for candidate in ({"input": [{"role": "assistant", "content": [{"type": "output_text", "text": letter}]}]},
                      {"messages": [{"role": "user", "content": "Another example note."}]}, {}):
        assert not extension_late_result._wire_has_letter(candidate, letter)


def test_normal_returns_debit_the_same_global_window_and_their_own_source_window():
    from ouroboros.model_wait import execution_close

    root, child_a, child_b = "root-sample", "source-a", "source-b"
    source = {"task_id": root, "status": "completed", "result": "Initial comparison kept."}

    def reserve(turn, scopes, prior_turn=None):
        spent, left = extension_late_result.debit_return_execution(source, scopes, prior_turn)
        # Previewing a finite remainder does not make the next turn its owner.
        assert not extension_late_result.return_time_owned(source, scopes, turn, f"report-{turn}", left)
        source[extension_late_result._SHARED_EXECUTION] = {
            "spent": spent,
            "active": {"task_id": turn, "report_ref": f"report-{turn}", "scopes": sorted(scopes), "seconds": left},
        }
        assert extension_late_result.return_time_owned(source, scopes, turn, f"report-{turn}", left)
        return left

    def finish(turn, time_limit, elapsed):
        clock = execution_close({"execution_timeout_sec": time_limit}, {"attempt": 1, "started_at": 1000.0}, 1000.0 + elapsed)
        return {"task_id": turn, "status": "completed", "task_attempt": 1, "execution_timeout_sec": time_limit,
                "admission_author": {"late_report_ref": f"report-{turn}"}, **clock}

    first = reserve("turn-a1", {root: 70.0, child_a: 35.0})
    assert first == 35.0
    prior = finish("turn-a1", first, 12.0)
    second = reserve("turn-b1", {root: 70.0, child_b: 65.0}, prior)
    assert second == 58.0  # only work in the common window is carried from A
    prior = finish("turn-b1", second, 9.0)
    third = reserve("turn-a2", {root: 70.0, child_a: 35.0}, prior)
    assert third == 23.0  # A did not acquire a fresh local budget or pay B's local debit
    assert source["result"] == "Initial comparison kept."


def test_main_program_mail_in_the_final_drain_gets_one_consideration(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.loop_delivery import _no_tool_final_answer
    from ouroboros.owner_mailbox import PROVENANCE_AGENT_NOTE, write_task_message
    from threading import RLock

    tid, text = "reader-42", "  The comparison is ready.\n"
    author = dialogue_provenance.recorded_author("Build agent")
    address = "row:1@2026-11-23T12:00:00+00:00#0123456789ab"
    assert write_task_message(tmp_path, text, tid, source_task_id="", provenance=PROVENANCE_AGENT_NOTE,
                              msg_id="note-final-42", agent_note={"author": author, "row_address": address})
    messages, seen, held = [], set(), []
    agent = SimpleNamespace(_busy=True, _current_task_id=tid, _accepting_owner_messages=True)
    ctx = SimpleNamespace(task_id=tid, drive_root=tmp_path, task_metadata={}, _owner_directives=[],
                          owner_message_admission_lock=RLock(), owner_message_admission_agent=agent)
    tools = SimpleNamespace(_ctx=ctx)
    limit = SimpleNamespace(messages=messages, task_id=tid, drive_root=tmp_path, drive_logs=tmp_path,
                            event_queue=None, accumulated_usage={})
    monkeypatch.setattr("ouroboros.loop_delivery._prepare_delivery_candidate", lambda content, *_a, **_k: (True, content))
    monkeypatch.setattr("ouroboros.loop_delivery._unread_review_updates", lambda *_a, **_k: ([], False))
    monkeypatch.setattr("ouroboros.loop_delivery.observe_completion_hold", lambda *args: held.append(args[3]))
    for name in ("_enforce_plan_gate", "_compute_subagent_handoff", "_maybe_inject_finalization_nudges",
                 "_finalize_task_services", "_project_child_result_dispositions"):
        monkeypatch.setattr(loop, name, lambda *_a, **_k: None)
    for name in ("_plan_gate_disclosure", "_forced_orphan_note"):
        monkeypatch.setattr(loop, name, lambda *_a, **_k: "")
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", lambda *_a, **_k: False)
    assert _no_tool_final_answer("The task is complete.", limit, {}, tools, queue.Queue(), seen, lambda _: None) is None
    assert held == ["agent_note"] and agent._accepting_owner_messages is True
    data = "\n".join(item["content"] for item in messages)
    assert data.count(text) == 1 and address in data and '"Build agent"' in data
    assert ctx._owner_directives == []
    assert not loop._drain_incoming_messages(messages, queue.Queue(), tmp_path, tid, None, seen, owner_ctx=ctx).get("agent_note")


def test_project_owner_reply_reaches_its_ordinary_reader_beside_a_local_return(tmp_path, monkeypatch):
    from ouroboros import server_owner_routing
    from ouroboros.projects_registry import create_project

    _, ctx, _, _, _ = _room(tmp_path, monkeypatch)
    room = create_project(tmp_path, "example-project")["chat_id"]
    readers = [{"task_id": tid, "room_reader": True} for tid in ("returner", "ordinary")]
    ctx.RUNNING = {"returner": {"task": {"admission_author": {"late_report_ref": "example-return"}}},
                   "ordinary": {"task": {"id": "ordinary"}}}
    observed = []

    def deliver(_ctx, reader, message, client_message_id, **kwargs):
        observed.append((reader["task_id"], message, client_message_id, kwargs.get("note")))
        return reader["task_id"]

    monkeypatch.setattr(server_owner_routing, "_deliver_to_reader", deliver)
    routed = server_owner_routing._route_project_chat_to_running_task(
        ctx, room, "Which example is ready?", "owner-example-7", roots=SimpleNamespace(rows=lambda _: readers))
    assert routed == "ordinary"
    assert observed == [("ordinary", "Which example is ready?", "owner-example-7", None)]


def test_presence_attention_requires_the_durable_quiet_completion(tmp_path):
    """Only the model's accepted silence with no observed or possible response gives «processed without reply»."""
    assert place_intake.attention_view({"reason": "paused"}, {}) == {"decision": "heard"}
    assert place_intake.attention_view({"reason": "conversation_work"}, {}) == {"decision": "handed_off"}
    assert place_intake.attention_view(None, {}) == {"decision": "unknown"}
    started = {"execution_owner": {"kind": "presence"}}
    assert place_intake.attention_view(None, started) == {"decision": "focus_started"}
    # The wake_silent conjunction also covers records-only/tool delivery: it is NOT a conscious "no answer".
    ambiguous = {**started, "status": "completed", "terminal_origin": "model_final", "wake_silent": True,
                 "metadata": {"presence_outcome": "silent"}}
    for outcome in ("silent", "tool_delivered"):
        assert place_intake.attention_view(None, {**ambiguous, "metadata": {"presence_outcome": outcome}}) == {
            "decision": "focus_started"
        }
    # Only the durable explicit-silence fact (the model's own accepted silence, no delivered answer) proves it.
    quiet = {**ambiguous, "presence_explicit_silence": True}
    assert place_intake.attention_view(None, quiet) == {"decision": "completed_silent"}

    # The real producer decides that fact, on a fictional place focus that finished silently.
    from ouroboros import place_reply

    key = "example|acct|conv|0"

    def focus_ctx(selected, calls=()):
        return SimpleNamespace(
            task_id="quiet-1", drive_root=tmp_path, task_contract={},
            task_metadata={"presence": {"place_id": "place-1", "transport_skill": "example",
                                        "event": {"conversation_key": key}}},
            _presence_completion={"outcome": "silent", "message": "", "quiet": True},
            _completion_selected=selected, _place_message_calls=list(calls))

    own_silent = {"action": "finish", "wake_silent": True, "source": "finish_task"}
    usage = {}
    place_reply.accept_focus_completion(focus_ctx(own_silent), usage, "model_final")
    assert usage.get("presence_explicit_silence") is True  # the model's own choice, no unconfirmed send
    assert place_intake.attention_view(None, {**ambiguous, **usage}) == {"decision": "completed_silent"}
    # A planned report's host-set quiet (the schedule door's wake_silent) is never the model's own silence.
    usage = {}
    place_reply.accept_focus_completion(focus_ctx({**own_silent, "source": "presence_finish"}), usage,
                                         "model_final")
    assert "presence_explicit_silence" not in usage
    # A dispatched reply of the same work whose outcome no receipt confirms leaves the state unproven.
    usage = {}
    place_reply.accept_focus_completion(
        focus_ctx(own_silent, calls=[{"tool": "example_send", "at": "2026-11-23T00:00:00Z",
                                       "text": "A fictional reply", "conversation_keys": [key]}]),
        usage, "model_final")
    assert "presence_explicit_silence" not in usage
    assert place_intake.attention_view(None, {**ambiguous, **usage}) == {"decision": "focus_started"}
    # A queued gesture in this conversation is also a possible response; another conversation's is not.
    gesture = {"tool": "example_react", "at": "2026-11-23T00:00:00Z", "gesture": True}
    for target, decision in [(key, "focus_started"), ("example|acct|other|0", "completed_silent")]:
        usage = {}
        place_reply.accept_focus_completion(focus_ctx(own_silent, [{**gesture, "conversation_keys": [target]}]),
                                             usage, "model_final")
        assert place_intake.attention_view(None, {**ambiguous, **usage}) == {"decision": decision}
