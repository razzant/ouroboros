"""Web Main's own Send in owner Nano on the qualified 80k local route, from the row the ingress keeps to the first
counted request: the routed owner turn starts by the serving formatter's exact count with its whole input, the row my
human wrote is said once beside it, a control met by the last count binds before the loop, and a directly required
handbook's refusal names the whole request it needs.

Synthetic: the route answer, the serving formatter and the SDK client (``tests.test_nano_startup_exact``). The kept row,
its routing, the conversation lane's seed and first step, and the fit are the product's own; the formatter only prices
the payload it is handed, so it never stands in for the ingress's provenance.

Reusable by a transport proof: ``routed_web_send`` returns the turn a carrier runs (``run_worker_task`` takes it as
the conversation lane hands it over), and the synthetic server's ``counted`` and ``sent`` lists
(``_install_synthetic_local``) collect its counted and sent payloads.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from ouroboros import context, context_fit, dialogue_store, local_model, memory_view, memory_view_work, nano_minimum
from ouroboros.context_window import WindowLimit, WindowRecord
from ouroboros.owner_relay import owner_row_address
from ouroboros.tools.registry import ToolContext
from tests import test_nano_startup_exact as exact
from tests.test_nano_startup_exact import (
    BASE_TOKENS, DENSE, REPO, UNIT, _bootstrapped_main_drive, _install_synthetic_local, _local_route_resolver,
    _web_main_task_and_ctx,
)

USABLE = 71_808  # the qualified 80k window less Nano's reply floor (``context_budget.NANO_MIN_HEADROOM_TOKENS``)
SEND_TS = "2026-02-01T11:59:00+00:00"
POINTER = "[Exact task input source]"
TURN_NOTE = "this turn's input, delivered whole as its message"


def routed_web_send(drive, monkeypatch, text, *, client_message_id="web-send-1", ts=SEND_TS, reply_to=""):
    """Web Main's Send up to its carrier: the ingress keeps my human's row, routing carries that row's identity
    (``origin_message_ref``, never ``origin_suppressed``) to the conversation lane, which seeds the turn and stamps
    it as ``conversation_lane.admit_turn`` does. Returns ``(turn, row)``: the turn a carrier runs, the row as kept.
    ``reply_to``: the address this Send replies to, resolved as the Web ingress resolves it
    (``chat_references.ingress_reply``) and kept on the row and as the turn's ``reply_reference``."""
    from ouroboros import server_owner_routing, server_routing_context as routing
    from ouroboros.chat_references import ingress_reply
    from ouroboros.owner_relay import owner_row_source
    from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
    from ouroboros.task_status import execution_owner_record
    from supervisor import worker_chat_lane, workers
    from supervisor.conversation_lane import PREPARE_KEY
    from tests.test_project_routing_v664 import _ctx

    reply = ingress_reply(drive, {"reply_to": reply_to}) if reply_to else {}
    row = {"ts": ts, "chat_id": 1, "user_id": 1, "direction": "in", "source": "web",
           "client_message_id": client_message_id, "text": text, **({"reply_to": reply} if reply else {})}
    row = dialogue_store.append(drive, "chat", row).row  # the row as kept, its identity by value
    admitted = []
    monkeypatch.setattr(routing, "_recent_root_results", lambda *_a, **_k: ([], {}))
    server_owner_routing._route_owner_message(
        object(), _ctx(drive, direct=lambda *args, **kwargs: admitted.append((args, kwargs))),
        {"chat_id": 1, "text": text, "client_message_id": client_message_id, "log_text": text, "source": "web",
         "origin_message_ref": owner_row_source(row)[0], **({"task_metadata": {"reply_reference": reply}} if reply else {})})
    [((chat_id, words, image), kwargs)] = admitted
    metadata = kwargs["task_metadata"]
    assert metadata["origin_message_ref"]["row_sha256"] and "origin_suppressed" not in metadata
    monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
    turn = worker_chat_lane._seed_task(chat_id, words, image, kwargs.get("task_constraint"), metadata)
    turn["root_task_id"] = turn["id"]
    turn["_execution_owner"] = {**execution_owner_record(drive, turn, "direct"), "lane": LANE_CONVERSATION,
                                "executor": EXECUTOR_WORKER}
    turn[PREPARE_KEY] = {"admitted_at": 0.0, "quiet": False, "reply_expected": True,
                         "client_message_id": client_message_id, "census": {"direct_turns_at_entry": 0}}
    return turn, row


def _first_request(env, memory, registry, tools, turn):
    """The carrier's first step on the routed turn, then its fit: the messages its first request holds."""
    from supervisor.worker_chat_lane import prepare_conversation_task

    prepare_conversation_task(None, turn, memory.drive_root)
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=memory.drive_root, current_chat_id=1, current_task_type="task",
                      task_id=turn["id"], is_direct_chat=True, task_metadata=turn["metadata"])
    registry.set_context(ctx)
    messages, _ = context.build_llm_messages(env=env, memory=memory, task=turn, ctx=ctx, tool_schemas=tools,
                                             tool_registry=registry, route_resolver=_local_route_resolver())
    return messages, ctx.context_fit_plan


def _runtime_origin(messages):
    system = messages[0]["content"]
    if isinstance(system, list):
        system = "\n\n".join(block["text"] for block in system if block["type"] == "text")
    body = system.split("## Runtime context\n\n", 1)[1]
    return json.JSONDecoder().raw_decode(body)[0]["task_authority_origin"]


def test_routed_web_send_starts_exact_with_its_whole_input_and_its_row_said_once(tmp_path, monkeypatch):
    """15k tokens of words that fit once start whole; said again in the room they would cross the usable input."""
    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    counted, sent, _ = _install_synthetic_local(monkeypatch, memory.drive_root)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 3.0)  # the estimate alone refuses
    memory.identity_path().write_text("# Synthetic Identity\n\n" + DENSE * 2_000, encoding="utf-8")  # 40k exact
    work = "Send from Web Main: " + DENSE * 750 + " END_OF_SEND"  # 15k exact
    turn, row = routed_web_send(memory.drive_root, monkeypatch, work)
    messages, plan = _first_request(env, memory, registry, tools, turn)

    once = BASE_TOKENS + 40_000 + 15_000
    assert once <= USABLE < once + 15_000
    assert plan.counter_status == "exact" and [tokens for _, tokens in counted] == [once]
    request = json.dumps(messages, ensure_ascii=False)
    assert work in str(messages[1]["content"]) and POINTER not in request and request.count("END_OF_SEND") == 1
    system = json.dumps(messages[0]["content"], ensure_ascii=False)
    assert f"{owner_row_address(row)}] (words, " in system and TURN_NOTE in system  # the row keeps its place and source
    origin = _runtime_origin(counted[0][0]["messages"])
    assert origin["ref"] == turn["metadata"]["origin_message_ref"]
    assert (origin["row_address"], origin["author"]) == (owner_row_address(row), "owner")
    assert origin["representation"] == "first_user_input" and "text" not in origin
    runtime = next(part for part in plan.core.dynamic_sections if part.section_id == "runtime")
    assert "END_OF_SEND" not in runtime.full_text and "END_OF_SEND" not in runtime.operative_view
    rebound = plan.reproject_for_route(window_tokens=plan.window_tokens, known_window=True, ratio=1.0,
                                       output_reserve=plan.output_reserve_tokens, tool_schemas=tools,
                                       current_messages=messages, start_mode="nano", select_mode="nano",
                                       unsent_start=True).messages_for("nano")
    assert _runtime_origin(rebound) == origin and json.dumps(rebound, ensure_ascii=False).count("END_OF_SEND") == 1
    assert sent == []


def test_only_the_attested_row_yields_an_identical_earlier_message_stays_whole(tmp_path, monkeypatch):
    """Deduplicating one attested source is not merging alike text: my human's earlier message with the same words
    is another row and stays whole; only the row this Send kept stands by its head."""
    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    counted, _, _ = _install_synthetic_local(monkeypatch, memory.drive_root)
    words = "Bring the brass key " + DENSE * 10 + " now."  # 200 exact
    earlier = {"ts": "2026-02-01T11:00:00+00:00", "chat_id": 1, "user_id": 1, "direction": "in", "source": "web",
               "client_message_id": "earlier-same-words", "text": words}
    earlier = dialogue_store.append(memory.drive_root, "chat", earlier).row
    turn, row = routed_web_send(memory.drive_root, monkeypatch, words)
    messages, plan = _first_request(env, memory, registry, tools, turn)

    assert plan.counter_status == "exact" and counted[0][1] == BASE_TOKENS + 400  # the earlier row and the input
    assert json.dumps(messages, ensure_ascii=False).count("Bring the brass key") == 2
    system = json.dumps(messages[0]["content"], ensure_ascii=False)
    assert f"{owner_row_address(row)}] (words, " in system and TURN_NOTE in system
    assert f"{owner_row_address(earlier)}] (words, " not in system and owner_row_address(earlier) in system
    assert _runtime_origin(messages)["row_address"] == owner_row_address(row) != owner_row_address(earlier)


def test_a_substituted_or_changed_input_keeps_the_row_whole(tmp_path, monkeypatch):
    """The row yields only on the qualified local 80k Nano route while the request delivers the turn's own input,
    exactly the row's words, unchanged: an exact-source pointer in its place, an input that frames those words inside
    other text, a transcript whose first input changed, or another mode, route or window keeps them whole in the room."""
    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    _install_synthetic_local(monkeypatch, memory.drive_root)
    phrase = "Keep the green lantern lit"
    words = phrase + " " + UNIT * 50 + "END"
    turn, row = routed_web_send(memory.drive_root, monkeypatch, words)
    _, plan = _first_request(env, memory, registry, tools, turn)
    core = plan.core
    whole = memory_view.snapshot_from_json(core.memory_view_json)
    address = owner_row_address(row)
    assert whole.room[memory_view_work.TURN_INPUT_KEY] == address

    def line(snapshot):
        return next(item["line"] for item in snapshot.room["lane1"] if item["address"] == address)

    def view(view_json, original_input, delivered, current=None, *, preferred="nano", provider="local",
             record=plan.window_record):
        return memory_view_work.turn_input_view(view_json, preferred, provider, record, original_input, delivered,
                                                current)
    original = core.user_content_json
    assert phrase in line(whole)
    once = line(view(core.memory_view_json, original, original))
    head = next(item["head"] for item in whole.room["lane1"] if item["address"] == address)
    assert once.startswith(head + " (words, ") and TURN_NOTE in once and phrase not in once
    assert line(view(core.memory_view_json, original, json.dumps(POINTER + "\n{}"))) == line(whole)
    same = [{"role": "system", "content": []}, {"role": "user", "content": json.loads(original)}]
    changed = [{"role": "system", "content": []}, {"role": "user", "content": json.loads(original) + "\n\nA later note."}]
    assert TURN_NOTE in line(view(core.memory_view_json, original, "", same))
    assert line(view(core.memory_view_json, original, original, changed)) == line(whole)
    framed = json.dumps("Quoting myself: " + words)  # the words inside another input are no proof it is theirs
    assert line(view(core.memory_view_json, framed, framed)) == line(whole)
    big = WindowRecord("fixture", total=WindowLimit(1_000_000, "total", "local_health"), reserve=8_192)
    for other in ({"preferred": "max"}, {"provider": "openrouter"}, {"record": big}):
        assert line(view(core.memory_view_json, original, original, **other)) == line(whole), other


def test_a_web_reply_keeps_its_row_whole_with_what_it_replies_to(tmp_path, monkeypatch):
    """A Reply's row prints what it replies to beside its words, which its input never carries: on the real Web
    Reply path the row is not said by its head, and the earlier row it answers keeps its own place."""
    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    counted, _, _ = _install_synthetic_local(monkeypatch, memory.drive_root)
    earlier = dialogue_store.append(memory.drive_root, "chat", {
        "ts": "2026-02-01T11:00:00+00:00", "chat_id": 1, "user_id": 1, "direction": "in", "source": "web",
        "client_message_id": "quoted-earlier", "text": "The ferry leaves at noon."}).row
    words = "Then book the later one " + DENSE * 10 + " please."  # 200 exact
    turn, row = routed_web_send(memory.drive_root, monkeypatch, words, reply_to=owner_row_address(earlier))
    messages, plan = _first_request(env, memory, registry, tools, turn)

    assert row.get("reply_to") and turn["metadata"].get("reply_reference"), "the Reply the ingress kept"
    assert memory_view_work.TURN_INPUT_KEY not in memory_view.snapshot_from_json(plan.core.memory_view_json).room
    system = json.dumps(messages[0]["content"], ensure_ascii=False)
    assert plan.counter_status == "exact" and "[Reply to" in system and TURN_NOTE not in system
    assert counted[0][1] == BASE_TOKENS + 400  # its row whole and its input
    assert json.dumps(messages, ensure_ascii=False).count("Then book the later one") == 2
    assert _runtime_origin(messages)["ref"] == turn["metadata"]["origin_message_ref"]
    assert "text" not in _runtime_origin(messages) and owner_row_address(earlier) in system


def test_the_start_candidate_is_my_own_focus_on_the_qualified_local_main_route():
    """Send in Main is a root of my own focus; a wake, Presence or a helper never takes this start."""
    from ouroboros.consciousness_authority import CONSCIOUSNESS_CATEGORY

    route = {"provider": "local", "use_local": True, "model_role": "main"}
    record = WindowRecord("fixture", total=WindowLimit(80_000, "total", "local_health"), reserve=8_192)
    turn = {"type": "task", "chat_id": 1, "_is_direct_chat": True, "metadata": {"work_kind": "owner"}}
    assert nano_minimum.local_root_start_candidate(turn, route, record)
    assert nano_minimum.local_root_start_candidate({**turn, "_is_direct_chat": False}, route, record)
    for other in ({"usage_category": CONSCIOUSNESS_CATEGORY}, {"_presence_turn": True}, {"parent_task_id": "root-1"},
                  {"delegation_role": "subagent"}):
        assert not nano_minimum.local_root_start_candidate({**turn, **other}, route, record)
    big = WindowRecord("fixture", total=WindowLimit(1_000_000, "total", "local_health"), reserve=8_192)
    assert not nano_minimum.local_root_start_candidate(turn, route, big)


CONTROLS = ["pause", "stop", "deadline", "wrap_up"]


def _routed_send_meets_a_control(tmp_path, monkeypatch, control, **counter):
    """``OuroborosAgent.handle_task`` of the routed Send whose one count runs while ``control`` lands (``counter``
    shapes the synthetic serving formatter). Nothing is sent and no counted view becomes a loop's state: Pause parks
    the original input, a pooled Stop is the supervisor's to settle, a deadline or Wrap up ends the turn unstarted
    with no answer to finalize."""
    import queue as stdqueue
    from functools import partial

    from ouroboros import agent_task_pipeline, budget_pause, cancel_intents, model_concurrency, owner_mailbox, owner_pause
    from ouroboros import subagent_runtime, usage_store, utils, working_checkpoint
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.owner_wait import worker_owner_wait
    from ouroboros.settings_integrity import task_settings_snapshot
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.worker_chat_lane import prepare_conversation_task

    env, memory, registry, _ = _bootstrapped_main_drive(tmp_path, monkeypatch)
    drive = memory.drive_root
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(drive / "settings.json"))
    monkeypatch.setenv(utils.WORKER_PROCESS_ENV, "1")  # a pooled carrier: its model wait holds the worker slot
    counted, sent, _ = _install_synthetic_local(monkeypatch, drive, **counter)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 3.0)
    monkeypatch.setattr(model_concurrency, "_chosen_cap", lambda: None)
    monkeypatch.setattr(subagent_runtime, "apply_task_start_settings_or_disclose",
                        lambda *_a, **_k: task_settings_snapshot({}, os.environ))
    monkeypatch.setattr("ouroboros.project_naming.spawn_turn_namer", lambda *_a, **_k: None)
    monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.owner_wait.context_start_route",
                        lambda _ctx, _task, _saved, effort, **_kw: (_local_route_resolver(), effort))

    work = "Send met by a control in its count: " + DENSE * 1_000 + " SEND_END"  # 20k exact: fits on its one count
    turn, _ = routed_web_send(drive, monkeypatch, work)
    prepare_conversation_task(None, turn, drive)
    turn["metadata"]["lead"] = {"source": "room", "model": "local-model", "use_local": True}
    task_id = turn["id"]
    write_task_result(drive, task_id, "requested", chat_id=1, billing_group={
        "billing_group_id": task_id, "billing_group_limit_usd": 20.0,
        "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "cap-1"})
    usage_store.migrate_from_journal(drive)

    def arrive():
        if control == "pause":
            owner_pause.install_fence(drive, task_id, request_id=f"{task_id}-press")
        elif control == "stop":
            cancel_intents.request_cancel(drive, task_id, reason="owner stop", source="http_single")
        elif control == "wrap_up":
            assert owner_mailbox.write_owner_message(drive, "wrap up now", task_id, msg_id="wrap-1",
                                                     kind=owner_mailbox.KIND_FINALIZE_NOW)
        else:
            turn["deadline_at"] = "2000-01-01T00:00:00+00:00"

    manager = local_model.get_manager()

    def counted_while_a_control_lands(payload, **kwargs):
        observed = manager.measure_prepared_input(payload, **kwargs)
        arrive()  # accepted while this count ran: the count itself succeeds, and its input fits
        return observed

    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        measure_prepared_input=counted_while_a_control_lands, serving_context_evidence=manager.serving_context_evidence))
    ready_seals = []
    real_save_ready = working_checkpoint.save_ready
    monkeypatch.setattr(working_checkpoint, "save_ready",
                        lambda limit_ctx: ready_seals.append(limit_ctx) or real_save_ready(limit_ctx))
    frames = []
    bus = SimpleNamespace(put=lambda event, *_a, **_k: frames.append(event),
                          put_nowait=lambda event, *_a, **_k: frames.append(event))
    agent = OuroborosAgent(Env(repo_dir=REPO, drive_root=drive), event_queue=bus)
    agent.owner_wait_callback = partial(worker_owner_wait, 0, stdqueue.Queue(), bus)
    events = agent.handle_task(turn)

    assert len(counted) == 1 and work in json.dumps(counted[0][0]["messages"], ensure_ascii=False), "the real count"
    assert counted[0][1] <= USABLE and sent == [] and ready_seals == []
    assert not agent._busy, "the interrupted turn released its agent"
    assert (load_task_result(drive, task_id) or {}).get("reason_code") != "task_exception"
    assert not budget_pause.budget_pause_row(drive, task_id), "no exact continuation of a counted view"
    if control == "pause":
        [event] = events
        paused = event["resource_limit"]
        assert event["type"] == "budget_pause" and paused["before_task_start"] is True and not paused.get("checkpoint")
        assert event["_is_direct_chat"] is True and event["task"]["text"] == work, "the original input parks"
    elif control == "stop":
        assert events == [] and cancel_intents.active_intent(drive, task_id), "custody settles the Stop"
    else:
        [metrics] = [event for event in events if event.get("type") == "task_metrics"]
        assert metrics["reason_code"] == ("deadline_local" if control == "deadline" else "owner_requested_finalization")


@pytest.mark.parametrize("control", CONTROLS)
def test_a_control_met_by_the_last_count_binds_before_the_loop(tmp_path, monkeypatch, provider_catalog_offline, control):
    """The Send's one exact count fits; the control accepted while it ran binds before the loop."""
    _routed_send_meets_a_control(tmp_path, monkeypatch, control)


@pytest.mark.parametrize("control", CONTROLS)
def test_a_control_met_by_an_unproven_count_binds_before_the_estimate(tmp_path, monkeypatch, provider_catalog_offline,
                                                                     control):
    """The count ran, but the server's launched window cannot be proven (a padded allocation), so the ordinary
    estimate would decide the start: the control accepted while that count ran binds before it."""
    _routed_send_meets_a_control(tmp_path, monkeypatch, control, launched_n_ctx=65_536)


def test_required_handbook_refusal_counts_the_whole_request_it_needs(tmp_path, monkeypatch):
    """The view fits without the directly required handbook and not with it: the refusal's input is the whole counted
    request, its handbook part named beside the rest, so its deficit is the real one."""
    from ouroboros.agent_dispatch import _context_capacity_terminal

    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    counted, sent, _ = _install_synthetic_local(monkeypatch, memory.drive_root)
    # A sentence under a chapter's subheading: only the whole book holds it, never the map's introductions.
    witness = "Name the authority, fact identity/scope and affected producers/consumers"
    assert witness in (REPO / "docs/development/03-module-size-and-complexity.md").read_text(encoding="utf-8")
    assert witness not in (REPO / "docs/DEVELOPMENT.md").read_text(encoding="utf-8")
    priced = exact._controlled_tokens
    book = 30_000
    monkeypatch.setattr(exact, "_controlled_tokens",
                        lambda blob: priced(blob) + (book if witness.encode("utf-8") in blob else 0))
    memory.identity_path().write_text("# Synthetic Identity\n\n" + DENSE * 2_000, encoding="utf-8")  # 40k exact
    task, ctx = _web_main_task_and_ctx(env, memory, registry, task_id="required-book",
                                       text="Book case: " + DENSE * 50)  # 1k exact
    task["task_contract"] = {"context_requires_development": True}
    with pytest.raises(context_fit.MinimumViewUnfit) as refused:
        context.build_llm_messages(env=env, memory=memory, task=task, ctx=ctx, tool_schemas=tools,
                                   tool_registry=registry, route_resolver=_local_route_resolver())

    unfit, whole = refused.value, BASE_TOKENS + 41_000 + book
    assert whole - book <= USABLE < whole
    assert max(tokens for _, tokens in counted) == whole and min(tokens for _, tokens in counted) == whole - book
    assert (unfit.required_book_tokens, unfit.fixed_input_tokens, unfit.estimated_input_tokens) == (book, whole, whole)
    assert unfit.deficit_tokens == whole + unfit.response_reserve_tokens - unfit.window_tokens > 0
    notice = _context_capacity_terminal(unfit)[0]
    assert f"input {whole:,} (core, request and tools {whole - book:,}; handbook {book:,})" in notice
    assert sent == []
