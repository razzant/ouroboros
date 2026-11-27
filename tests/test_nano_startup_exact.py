"""Compact synthetic tests for exact local Nano startup admission, completeness gates and scope isolation."""
from __future__ import annotations

import ast
import datetime as dt
from dataclasses import replace
import hashlib
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import (
    artifacts, config, context, context_fit, deployment_policy, dialogue_store,
    knowledge, llm_local, local_model, memory as memory_mod, memory_view,
    nano_minimum, send_clock, skill_catalogue,
)
from ouroboros import usage_accounting as usage
from ouroboros.agent import Env
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.context_window import WindowLimit, WindowRecord
from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import _PhysicalSendNotStarted
from ouroboros.local_model_server import input_fingerprint
from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS
from ouroboros.memory import Memory
from ouroboros.model_concurrency import ModelSlotCancelled
from ouroboros.nano_publication import (
    MemoryChange, NanoPublicationRefused, QUALIFIED_WINDOW,
    local_root_publication, read_root_sample, record_root_sample,
)
from ouroboros.send_clock import SendClockPolicy, main_send_scope
from ouroboros.skill_discovery_reuse import WalkedSkills
from ouroboros.tool_policy import initial_tool_schemas
from ouroboros.tools.registry import ToolContext, ToolRegistry

REPO = Path(__file__).resolve().parents[1]
PORT = 8767
WINDOW = QUALIFIED_WINDOW
FIXED_NOW = dt.datetime(2026, 2, 1, 12, 0, tzinfo=dt.timezone.utc)
MIND_AUTHOR = {"kind": "mind", "task_id": "seed-mark", "focus": {"role": "root", "task_id": "seed-mark"}}
COUNTER = {"source": "serving_llama_formatter", "formatter": "SyntheticLlamaFormatter",
           "library_version": "synthetic-1.0", "chat_format": "synthetic-qwen"}
# Controlled markers priced solely by explicit marker counts in the prepared payload
# (32-char UNIT -> ~8 estimator tokens vs +1 exact formatter token; 4-char DENSE -> ~1 estimator token vs +20 exact).
UNIT = "synthetic_compact_token_unit_x_ "
DENSE = "!#z?"
BASE_TOKENS = 10_000


def _controlled_tokens(blob: bytes) -> int:
    return BASE_TOKENS + blob.count(UNIT.encode("utf-8")) + blob.count(DENSE.encode("utf-8")) * 20


def _install_synthetic_local(monkeypatch, drive_root: Path, *, launched_n_ctx: int = WINDOW,
                             raw_window: int = WINDOW, serving_confirmed: bool = True, fail_reason: str | None = None):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Synthetic Nano startup test tried to open a real socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(send_clock, "_now", lambda: FIXED_NOW)
    counted, sent, manifests = [], [], []

    def measure_prepared_input(payload, *, expected_port=None):
        assert expected_port in (None, PORT)
        if fail_reason is not None:
            counted.append((payload, None))
            return {"supported": False, "input_is_exact": False, "reason": fail_reason}
        blob = json.dumps({key: payload.get(key) for key in ("messages", "tools", "tool_choice")},
                          ensure_ascii=False, sort_keys=True).encode("utf-8")
        tokens = _controlled_tokens(blob)
        counted.append((payload, tokens))
        return {
            "supported": True, "input_is_exact": True, "input_tokens": tokens,
            "context_window": WINDOW, "raw_context_window": raw_window, "process_id": 31,
            "native_input_sha256": input_fingerprint(payload),
            "output_limit_enforced": True, "reasoning_included_in_limit": True,
            "tokenizer_template_provenance": {**COUNTER, "launched_n_ctx": launched_n_ctx,
                                              "rendered_prompt_sha256": hashlib.sha256(blob).hexdigest()},
        }

    def create(**body):
        sent.append(body)
        return SimpleNamespace(model_dump=lambda: {"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    client = SimpleNamespace(base_url=f"http://127.0.0.1:{PORT}/v1",
                             chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(local_model, "local_dispatch_port", lambda: PORT)
    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        measure_prepared_input=measure_prepared_input,
        serving_context_evidence=lambda **_: (
            {"process_id": 31, "context_window": WINDOW, "confirmed": True,
             "source": "published_server_arguments", "port": PORT} if serving_confirmed else
            {"process_id": None, "context_window": None, "confirmed": False, "source": "serving_window_unobserved"}
        ),
    ))
    monkeypatch.setattr(LLMClient, "_get_local_client", lambda _self: client)
    monkeypatch.setattr(llm_local, "limit_physical_send", lambda _target, send: send)

    def execute(request, send, before):
        before(SimpleNamespace(attempt_id="synthetic-attempt", drive_root=drive_root))
        manifests.append(request.physical_context.nano_floor_evidence)
        return send()

    monkeypatch.setattr(llm_local, "_execute_candidate", execute)
    return counted, sent, manifests


def _local_route_resolver(window: int = WINDOW, *, response_limit: dict | None = None,
                          working_reserve_tokens: int | None = None, window_record: WindowRecord | None = None):
    facts = {"working_reserve_tokens": working_reserve_tokens} if working_reserve_tokens is not None else {}
    evidence = SimpleNamespace(route_fp="synthetic-local-main", status="confirmed", stale=False, window_tokens=window,
                               route_facts=facts, response_limit=response_limit or {}, window_record=window_record,
                               source="published_server_arguments")
    route = {"model": "local-model", "provider": "local", "use_local": True, "model_role": "main"}
    return lambda *_a, **_kw: (route, evidence)


def _bootstrapped_main_drive(tmp_path: Path, monkeypatch):
    drive = tmp_path / "drive"
    drive.mkdir(parents=True, exist_ok=True)
    env, memory = Env(repo_dir=REPO, drive_root=drive), Memory(drive_root=drive, repo_dir=REPO)
    memory.ensure_files()
    ChronicleStore(drive).ensure_activated()
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "nano")
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setattr(deployment_policy, "organization_rules", lambda: "Synthetic organization baseline rule.")
    normal_setting = config.runtime_setting
    monkeypatch.setattr(config, "runtime_setting",
                        lambda key, *a, **kw: "Synthetic owner baseline rule."
                        if key == "OUROBOROS_SAFETY_OWNER_RULES" else normal_setting(key, *a, **kw))
    registry = ToolRegistry(REPO, drive)
    return env, memory, registry, initial_tool_schemas(registry, context_mode="nano")


def _web_main_task_and_ctx(env: Env, memory: Memory, registry: ToolRegistry, *, task_id: str, text: str):
    meta = {"source": "web", "delegation_role": "root", "budget_drive_root": str(memory.drive_root)}
    task = {"id": task_id, "type": "task", "source": "web", "chat_id": 1, "delegation_role": "root",
            "text": text, "budget_drive_root": str(memory.drive_root), "metadata": dict(meta)}
    ctx = ToolContext(repo_dir=env.repo_dir, drive_root=memory.drive_root, current_chat_id=1,
                      current_task_type="task", task_id=task_id, is_direct_chat=False, task_metadata=dict(meta))
    registry.set_context(ctx)
    return task, ctx


def _dispatch_round_one(plan, messages, tools):
    frame = usage.PhysicalAttemptContext(
        "owner_nano", "nano", "cold_estimate", "synthetic-local-main", "cold_start",
        WINDOW, WINDOW, False, False, preferred_mode="nano",
        nano_floor_plan=(plan, "cold_start"), response_reserve_tokens=8_192,
    )
    with usage.bind_physical_attempt_context(frame), main_send_scope(SendClockPolicy()):
        return LLMClient(api_key="unused")._chat_local(
            messages, plan.start_tool_schemas("nano") or tools, MAIN_LOOP_MAX_TOKENS, "auto",
        )


def test_web_main_builder_to_prepared_payload_when_estimator_overcounts(tmp_path, monkeypatch):
    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    counted, sent, manifests = _install_synthetic_local(monkeypatch, memory.drive_root)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.6)

    memory.identity_path().write_text("# Synthetic Identity\n\n" + UNIT * 1_000, encoding="utf-8")
    overview_addr = knowledge.resolve_knowledge_address(memory.drive_root, "overview", "global")
    overview_addr.path.parent.mkdir(parents=True, exist_ok=True)
    overview_addr.path.write_text("# Overview\n\nSynthetic shared orientation " + UNIT * 1_000, encoding="utf-8")
    note_addr = knowledge.resolve_knowledge_address(memory.drive_root, "synthetic-note", "global")
    note_addr.path.write_text(
        "---\nsummary: " + json.dumps("Synthetic topic summary " + UNIT * 500) + "\n---\n# Note\n\nBody.\n",
        encoding="utf-8",
    )
    memory.append_scratchpad_block("Synthetic working block " + UNIT * 1_000)
    assert ChronicleStore(memory.drive_root).mark(
        {"kind": "task", "task_id": "seed-mark"}, "Synthetic active mark " + UNIT * 500, MIND_AUTHOR, room_id="1",
    ).ok
    owner_rule_text = "Synthetic owner standing rule " + UNIT * 1_000
    normal_setting = config.runtime_setting
    monkeypatch.setattr(config, "runtime_setting",
                        lambda key, *a, **kw: owner_rule_text
                        if key == "OUROBOROS_SAFETY_OWNER_RULES" else normal_setting(key, *a, **kw))

    work_70k = "Assignment: " + UNIT * 55_000 + "END"
    task_70k, ctx_70k = _web_main_task_and_ctx(env, memory, registry, task_id="start-70000", text=work_70k)
    msgs_70k, cap_70k = context.build_llm_messages(
        env=env, memory=memory, task=task_70k, ctx=ctx_70k,
        tool_schemas=tools, tool_registry=registry, route_resolver=_local_route_resolver(),
    )
    plan_70k = ctx_70k.context_fit_plan
    assert plan_70k.output_reserve_tokens == 20_000 and plan_70k.counter_status == "exact"
    assert len(counted) == 1 and counted[-1][1] == 70_000
    assert cap_70k["context_fit"]["nano_calibrated_tokens"] > WINDOW
    assert cap_70k["context_fit"]["counter_status"] == "exact"
    assert work_70k in msgs_70k[1]["content"] and "[Exact task input source]" not in msgs_70k[1]["content"]
    assert not (memory.drive_root / "artifacts" / "actors" / "start-70000").exists()
    system_blob = json.dumps(msgs_70k[0]["content"], ensure_ascii=False)
    for marker in ("## BIBLE.md", "# Synthetic Identity", "Synthetic shared orientation", "Synthetic topic summary",
                   "Synthetic working block", "Synthetic active mark", "Synthetic owner standing rule",
                   "Synthetic organization baseline rule."):
        assert marker in system_blob

    answer, _ = _dispatch_round_one(plan_70k, msgs_70k, tools)
    assert answer["content"] == "ok" and len(sent) == 1
    assert any(work_70k in str(row.get("content", "")) for row in sent[0]["messages"] if row.get("role") == "user")
    assert manifests[-1]["status"] == "exact" and manifests[-1]["scope"] == "cold_start" and manifests[-1]["room"] == "main"
    assert manifests[-1]["memory_tokens"] == 5_000
    record_root_sample(memory.drive_root, {**manifests[-1], "port": PORT})
    monkeypatch.setattr("ouroboros.nano_host_publication.current_rules_digest", lambda: manifests[-1]["rules_sha256"])
    assert read_root_sample(memory.drive_root, provider="local")["measurement"] == "exact"

    note_addr.path.unlink()
    for target_tokens, expects_pointer in ((71_808, False), (71_809, True)):
        counted.clear()
        work = "Boundary: " + UNIT * (target_tokens - BASE_TOKENS - 4_500) + "END"
        t, c = _web_main_task_and_ctx(env, memory, registry, task_id=f"case-{target_tokens}", text=work)
        msgs, _ = context.build_llm_messages(env=env, memory=memory, task=t, ctx=c, tool_schemas=tools,
                                             tool_registry=registry, route_resolver=_local_route_resolver())
        assert counted[0][1] == target_tokens and counted[-1][1] <= 71_808
        assert ("[Exact task input source]" in msgs[1]["content"]) is expects_pointer
        assert (work in msgs[1]["content"]) is (not expects_pointer)


def test_native_shortage_keeps_marks_history_and_full_index_despite_estimator(tmp_path, monkeypatch):
    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    counted, sent, _ = _install_synthetic_local(monkeypatch, memory.drive_root)
    from ouroboros import context_start_bound

    monkeypatch.setattr(context_start_bound, "_lowest_density", lambda *_: 1.0)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 2.0)
    # The lower bound alone overflows for the estimator. The physical overflow
    # instead comes from the soon-addressable WORLD, not from this identity.
    memory.identity_path().write_text("# Identity\n\n" + UNIT * 15_000, encoding="utf-8")
    memory.world_path().write_text("# World\n\nWORLD_PHYSICAL_COST " + DENSE * 2_400, encoding="utf-8")
    note = knowledge.resolve_knowledge_address(memory.drive_root, "selection-witness", "global")
    note.path.parent.mkdir(parents=True, exist_ok=True)
    note.path.write_text('---\nsummary: "Full index preserves the blue notebook."\n---\n# Witness\n', encoding="utf-8")
    assert ChronicleStore(memory.drive_root).mark(
        {"kind": "task", "task_id": "seed-mark"}, "Active mark: keep the brass key.", MIND_AUTHOR, room_id="1",
    ).ok
    dialogue_store.append(memory.drive_root, "chat", {
        "ts": "2026-02-01T11:00:00+00:00", "chat_id": 1, "user_id": 1, "direction": "in", "source": "web",
        "client_message_id": "required-owner-history", "text": "The train leaves from platform seven.",
    })
    task, ctx = _web_main_task_and_ctx(env, memory, registry, task_id="counter-rung", text="Find the key.")
    rows, _ = context.build_llm_messages(env, memory, task, ctx=ctx, tool_schemas=tools,
                                         tool_registry=registry, route_resolver=_local_route_resolver())
    facts = ctx.context_fit_plan.projection("nano").memory_facts
    assert ctx.context_fit_plan.counter_status == "exact" and counted[0][1] > 71_808 >= counted[-1][1]
    assert "world" in {row["section_id"] for row in facts["host_sections_addressed"]}
    assert "knowledge_index" not in {row["section_id"] for row in facts["host_sections_addressed"]}
    for blob in (json.dumps(rows[0], ensure_ascii=False), json.dumps(counted[-1][0], ensure_ascii=False)):
        assert "brass key" in blob and "platform seven" in blob and "blue notebook" in blob
        assert "WORLD_PHYSICAL_COST" not in blob
    assert not sent


def test_world_reader_after_identical_overflows_preserves_owner_words_and_clean_refusal(tmp_path, monkeypatch):
    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    counted, sent, _ = _install_synthetic_local(monkeypatch, memory.drive_root)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.5)

    memory.world_path().write_text("# World\n\n## Host Topology\n" + DENSE * 800 + "\n\n## Tail\nWorld tail.\n", encoding="utf-8")
    dialogue_store.append(memory.drive_root, "chat", {
        "ts": "2026-02-01T11:00:00+00:00", "chat_id": 1, "user_id": 1, "direction": "in", "source": "web",
        "client_message_id": "owner-uncovered-1", "text": "Earlier uncovered owner word must stay in view: " + DENSE * 200,
    })
    dialogue_store.append(memory.drive_root, "chat", {
        "ts": "2026-02-01T11:01:00+00:00", "chat_id": 1, "direction": "out", "source": "web", "text": "Acknowledged.",
    })
    dialogue_store.append(memory.drive_root, "chat", {
        "ts": "2026-02-01T10:00:00+00:00", "chat_id": 99, "user_id": 1, "direction": "in", "source": "web",
        "client_message_id": "other-room-1", "text": "other_room_padding " * 8_000,
    })

    # 1a. Exercise the exact walk when the first two rungs render the same oversized input.
    #     The normal builder may address WORLD on either rung as its other fixed parts grow (1b).
    work_fit = "Whole assignment rescued by WORLD reader: " + DENSE * 2_500 + " END_WORK"
    t_id, c_id = _web_main_task_and_ctx(env, memory, registry, task_id="identical-overflow", text=work_fit)
    base_core = context._capture_context_core(env, memory, t_id, None, c_id, registry)
    snap = memory_view.snapshot_from_json(base_core.memory_view_json)
    only_world_core = replace(
        base_core, reference_books=(), dynamic_head_sections=(),
        semi_stable_sections=tuple(p for p in base_core.semi_stable_sections if p.section_id in ("identity", "owner_rules", "world")),
        dynamic_sections=tuple(p for p in base_core.dynamic_sections if p.section_id == "scratchpad"),
        memory_view_json=memory_view.snapshot_json(replace(snap, conversations=())),
    )
    seen_trial_hashes, orig_count_io = [], LLMClient._local_count_io

    def spied_count_io(self, rows, schemas, caller_max, **kw):
        actual, ceiling, prepare, measure = orig_count_io(self, rows, schemas, caller_max, **kw)
        seen_trial_hashes.append(input_fingerprint(actual))

        def spied_prepare(cand_rows):
            payload = prepare(cand_rows)
            seen_trial_hashes.append(input_fingerprint(payload))
            return payload

        return actual, ceiling, spied_prepare, measure

    exact_start = nano_minimum._exact_start_fit

    def repeat_overflow(core, state, schemas, project, snapshot, user_json):
        uncut = None

        def two_identical_rungs(frame, text, **kwargs):
            nonlocal uncut
            if kwargs.get("floor_mode") in ("keep_owner", "before_f7") and not kwargs.get("exact_local_pressure"):
                return uncut
            rendered = project(frame, text, **kwargs)
            if uncut is None:
                uncut = rendered
            return rendered

        return exact_start(core, state, schemas, two_identical_rungs, snapshot, user_json)

    with monkeypatch.context() as first_two:
        first_two.setattr(LLMClient, "_local_count_io", spied_count_io)
        first_two.setattr(nano_minimum, "_exact_start_fit", repeat_overflow)
        id_plan = context_fit.build_context_fit_plan(env, only_world_core, t_id, preferred_mode="nano",
                                                     route_resolver=_local_route_resolver(), tool_schemas=tools)
    first, repeated_a, repeated_b, trial, retained = seen_trial_hashes
    assert first == repeated_a == repeated_b and trial == retained != first
    assert [tokens for _, tokens in counted] == [80_000, 64_000]
    id_msgs = id_plan.messages_for("nano")
    assert work_fit in id_msgs[1]["content"] and "[Exact task input source]" not in id_msgs[1]["content"]
    assert "Earlier uncovered owner word must stay in view:" in json.dumps(id_msgs[0]["content"], ensure_ascii=False)
    assert "F7" not in (id_plan.projection("nano").memory_facts.get("floor", {}).get("steps") or {})

    # 1b. Full build_llm_messages: WORLD -> reader fits before F7, retaining WORLD source without predicted leaks.
    counted.clear()
    t_world, c_world = _web_main_task_and_ctx(env, memory, registry, task_id="world-reader-fit", text=work_fit)
    msgs_world, _ = context.build_llm_messages(env=env, memory=memory, task=t_world, ctx=c_world, tool_schemas=tools,
                                               tool_registry=registry, route_resolver=_local_route_resolver())
    assert counted[0][1] == 80_000 and counted[-1][1] == 64_000
    assert work_fit in msgs_world[1]["content"] and "[Exact task input source]" not in msgs_world[1]["content"]
    assert "Earlier uncovered owner word must stay in view:" in json.dumps(msgs_world[0]["content"], ensure_ascii=False)
    facts_world = c_world.context_fit_plan.projection("nano").memory_facts
    assert "F7" not in (facts_world.get("floor", {}).get("steps") or {}) and facts_world["floor"]["window_tokens"] == WINDOW
    addressed = {row["section_id"]: row for row in facts_world["host_sections_addressed"]}
    assert "world" in addressed and not any(row.get("predicted") for row in addressed.values())
    assert "World tail." in artifacts.read_actor_source_bytes(
        memory.drive_root, "world-reader-fit", addressed["world"]["source_ref"],
    ).decode("utf-8")

    # 2. When pre-F7 + WORLD -> reader still overflows (74,000 > 71,808), F7 folds (-4,000 -> 70,000) with real 80k window.
    counted.clear()
    work_f7 = "Needs F7 after WORLD reader: " + DENSE * 3_000
    t_f7, c_f7 = _web_main_task_and_ctx(env, memory, registry, task_id="f7-after-world", text=work_f7)
    msgs_f7, _ = context.build_llm_messages(env=env, memory=memory, task=t_f7, ctx=c_f7, tool_schemas=tools,
                                            tool_registry=registry, route_resolver=_local_route_resolver())
    sys_f7 = json.dumps(msgs_f7[0]["content"], ensure_ascii=False)
    assert work_f7 in msgs_f7[1]["content"] and counted[-1][1] == 70_000
    assert f"the window ({WINDOW} tokens)" in sys_f7 and "the window (1 tokens)" not in sys_f7
    assert c_f7.context_fit_plan.projection("nano").memory_facts["floor"]["window_tokens"] == WINDOW

    # 3. Failed retain_section on host-world leaves WORLD whole without predicted leaks, recounts, and folds F7 to fit.
    orig_store = artifacts.store_actor_source_bytes
    monkeypatch.setattr(artifacts, "store_actor_source_bytes",
                        lambda root, actor, *, category, source_id, data, extension="txt", **kw:
                        (_ for _ in ()).throw(OSError("disk full")) if source_id == "host-world"
                        else orig_store(root, actor, category=category, source_id=source_id, data=data, extension=extension, **kw))
    t_fail, c_fail = _web_main_task_and_ctx(env, memory, registry, task_id="world-retain-fail", text="Work " + DENSE * 2_200)
    msgs_fail, _ = context.build_llm_messages(env=env, memory=memory, task=t_fail, ctx=c_fail, tool_schemas=tools,
                                              tool_registry=registry, route_resolver=_local_route_resolver())
    fail_rows = c_fail.context_fit_plan.projection("nano").memory_facts.get("host_sections_addressed", [])
    assert not any(row.get("predicted") or row.get("section_id") == "world" for row in fail_rows)
    assert "World tail." in json.dumps(msgs_fail[0]["content"], ensure_ascii=False)
    assert counted[-1][1] == 70_000 and sent == []
    monkeypatch.setattr(artifacts, "store_actor_source_bytes", orig_store)

    # 4. Unfitting mandatory minimum refuses cleanly with MinimumViewUnfit after measuring the actual minimum projection.
    monkeypatch.setattr(deployment_policy, "organization_rules", lambda: "Mandatory org rule " + DENSE * 3_500)
    t_unfit, c_unfit = _web_main_task_and_ctx(env, memory, registry, task_id="unfit-min", text="Work " + DENSE * 1_000)
    with pytest.raises(context_fit.MinimumViewUnfit) as exc_info:
        context.build_llm_messages(env=env, memory=memory, task=t_unfit, ctx=c_unfit, tool_schemas=tools,
                                   tool_registry=registry, route_resolver=_local_route_resolver())
    assert exc_info.value.measurement_density == 1.0 and exc_info.value.deficit_tokens > 0


def test_small_ceiling_input_cap_and_counter_unavailability_window_separation(tmp_path, monkeypatch):
    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    counted, _, _ = _install_synthetic_local(monkeypatch, memory.drive_root)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 3.0)

    def build_tokens(target: int, tag: str, resolver):
        counted.clear()
        work = "Work: " + UNIT * (target - BASE_TOKENS) + "END"
        t, c = _web_main_task_and_ctx(env, memory, registry, task_id=tag, text=work)
        msgs, cap = context.build_llm_messages(env=env, memory=memory, task=t, ctx=c, tool_schemas=tools,
                                               tool_registry=registry, route_resolver=resolver)
        return work, msgs, cap, c.context_fit_plan

    work_c, msgs_c, _, plan_c = build_tokens(
        74_000, "small-c", _local_route_resolver(response_limit={"max_output_tokens": 4_096, "source": "metadata"}),
    )
    assert plan_c.output_reserve_tokens == 4_096 and work_c in msgs_c[1]["content"]

    work_res = _local_route_resolver(working_reserve_tokens=12_000)
    w_fit, m_fit, _, _ = build_tokens(68_000, "wr-68000", work_res)
    _, m_over, _, _ = build_tokens(68_001, "wr-68001", work_res)
    assert w_fit in m_fit[1]["content"] and "[Exact task input source]" in m_over[1]["content"]

    declared = (WindowLimit(65_000, "input", "user_setting"), WindowLimit(WINDOW, "total", "user_setting"))
    capped_record = WindowRecord("synthetic-local-main", input=declared[0], total=declared[1], reserve=8_192, declared=declared)
    cap_res = _local_route_resolver(window_record=capped_record)
    w_in_fit, m_in_fit, _, _ = build_tokens(65_000, "in-65000", cap_res)
    _, m_in_over, _, _ = build_tokens(65_001, "in-65001", cap_res)
    assert w_in_fit in m_in_fit[1]["content"] and "[Exact task input source]" in m_in_over[1]["content"]

    for control_exc in (ModelSlotCancelled("cancelled"), _PhysicalSendNotStarted("owner_pause")):
        monkeypatch.setattr(llm_local, "measurement_slot", lambda _t, *, wait=False, _e=control_exc: (_ for _ in ()).throw(_e))
        t_ctrl, c_ctrl = _web_main_task_and_ctx(env, memory, registry, task_id="ctrl", text="Work " + UNIT * 20_000)
        with pytest.raises(type(control_exc)):
            context.build_llm_messages(env=env, memory=memory, task=t_ctrl, ctx=c_ctrl,
                                       tool_schemas=tools, tool_registry=registry, route_resolver=cap_res)
    from ouroboros.model_concurrency import measurement_slot as real_measurement_slot
    monkeypatch.setattr(llm_local, "measurement_slot", real_measurement_slot)

    for kw in ({"fail_reason": "measurement_unavailable:ConnectionError"}, {"serving_confirmed": False}):
        cnt, snt, _ = _install_synthetic_local(monkeypatch, memory.drive_root, **kw)
        work_down = "Whole input preserved during counter outage: " + UNIT * 25_000
        t_down, c_down = _web_main_task_and_ctx(env, memory, registry, task_id="counter-down", text=work_down)
        m_down, cap_down = context.build_llm_messages(env=env, memory=memory, task=t_down, ctx=c_down, tool_schemas=tools,
                                                      tool_registry=registry, route_resolver=cap_res)
        plan_down = c_down.context_fit_plan
        assert snt == [] and len(cnt) <= 1
        assert work_down in m_down[1]["content"] and "[Exact task input source]" not in m_down[1]["content"]
        assert plan_down.counter_status == "unavailable" and cap_down["context_fit"]["counter_status"] == "unavailable"
        assert plan_down.status == "confirmed" and cap_down["context_fit"]["evidence_status"] == "confirmed"
        assert plan_down.window_record.input.tokens == 65_000 and plan_down.window_record.total.tokens == WINDOW

    # Unlike the exact cases, this one must fit by the ordinary estimator after
    # addressing the task input. Its fixed frame alone exceeds 80k at the test's 3x calibration.
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    _install_synthetic_local(monkeypatch, memory.drive_root, launched_n_ctx=65_536, raw_window=WINDOW)
    t_pad, c_pad = _web_main_task_and_ctx(env, memory, registry, task_id="padded", text=UNIT * 30_000)
    m_pad, _ = context.build_llm_messages(env=env, memory=memory, task=t_pad, ctx=c_pad, tool_schemas=tools,
                                          tool_registry=registry, route_resolver=_local_route_resolver())
    assert c_pad.context_fit_plan.counter_status == "effective_window_unproven"
    assert "[Exact task input source]" in m_pad[1]["content"]


@pytest.mark.parametrize("control", ["release", "pause", "stop", "deadline", "wrap_up"])
def test_start_count_waits_on_a_busy_slot_and_hands_its_control_to_the_loop(tmp_path, monkeypatch, provider_catalog_offline, control):
    """``OuroborosAgent.handle_task`` of a fresh Web/Main Nano root whose exact start count finds the built-in
    server's one slot held by a generation in this test's own queue. The count waits; at its first failed acquisition
    either the holder releases (the same real exact count, then the whole input is sent) or the task's own control
    arrives: Pause parks the original input, Resume measures it anew; a pooled Stop lets the supervisor settle it;
    deadline and Wrap up have no admitted answer to finalize. No preview enters a loop or ready checkpoint.
    Synthetic: the route answer, the serving formatter and the SDK client."""
    import queue as stdqueue
    import time as clock
    from functools import partial

    from ouroboros import agent_task_pipeline, budget_pause, cancel_intents, model_concurrency, owner_mailbox, owner_pause
    from ouroboros import subagent_runtime, usage_store, utils, working_checkpoint
    from ouroboros.agent import OuroborosAgent
    from ouroboros.model_wait import current_model_wait
    from ouroboros.owner_wait import worker_owner_wait
    from ouroboros.settings_integrity import task_settings_snapshot
    from ouroboros.task_results import load_task_result, write_task_result

    env, memory, registry, _ = _bootstrapped_main_drive(tmp_path, monkeypatch)
    drive = memory.drive_root
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(drive))  # the model queue lives under it: this test's own
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(drive / "settings.json"))
    monkeypatch.setenv(utils.WORKER_PROCESS_ENV, "1")  # a pooled worker: its model wait holds the worker slot
    counted, sent, _ = _install_synthetic_local(monkeypatch, drive)
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 3.0)
    monkeypatch.setattr(model_concurrency, "_chosen_cap", lambda: None)  # no number chosen: the server's one slot
    monkeypatch.setattr(subagent_runtime, "apply_task_start_settings_or_disclose",
                        lambda *_a, **_k: task_settings_snapshot({}, os.environ))
    monkeypatch.setattr("ouroboros.project_naming.spawn_turn_namer", lambda *_a, **_k: None)
    monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.owner_wait.context_start_route",
                        lambda _ctx, _task, _saved, effort, **_kw: (_local_route_resolver(), effort))

    task_id = f"nano-start-{control}"
    work = "Nano start behind a busy slot: " + UNIT * 20_000 + " FIRST_INPUT_END"
    task, _ = _web_main_task_and_ctx(env, memory, registry, task_id=task_id, text=work)
    task.update(_attempt=1, root_task_id=task_id)
    task["metadata"]["lead"] = {"source": "room", "model": "local-model", "use_local": True}
    write_task_result(drive, task_id, "requested", chat_id=1, billing_group={
        "billing_group_id": task_id, "billing_group_limit_usd": 20.0,
        "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "cap-1"})
    usage_store.migrate_from_journal(drive)
    if control == "pause":
        from tests._budget_pause_exact_helpers import _install_queue, _supervisor_ctx
        from supervisor.events_budget import _handle_budget_pause

        queue, _, workers = _install_queue(drive, monkeypatch)
        worker = SimpleNamespace(busy_task_id=task_id)
        task["worker_id"] = 4
        workers.WORKERS[4] = worker
        workers.RUNNING[task_id] = {"task": task, "attempt": 1, "worker_id": 4}

    ready_seals = []
    real_save_ready = working_checkpoint.save_ready

    def observe_ready(limit_ctx):
        ready_seals.append(limit_ctx)
        return real_save_ready(limit_ctx)

    monkeypatch.setattr(working_checkpoint, "save_ready", observe_ready)
    holding = [model_concurrency.model_call_slot("local-model", True, route=model_concurrency.built_in_route(PORT))]
    holding[0].__enter__()  # a generation holds the built-in server's one slot
    waits = []

    def first_failed_acquisition(_delay):
        waits.append(list(counted))
        if len(waits) > 1:
            pytest.fail("The blocked counter kept waiting after release or control")
        if control == "release":
            holding.pop().__exit__(None, None, None)
        elif control == "pause":
            owner_pause.install_fence(drive, task_id, request_id=f"{task_id}-press")
        elif control == "stop":
            cancel_intents.request_cancel(drive, task_id, reason="owner stop", source="http_single")
        elif control == "wrap_up":
            assert owner_mailbox.write_owner_message(drive, "wrap up now", task_id, msg_id="wrap-1",
                                                     kind=owner_mailbox.KIND_FINALIZE_NOW)
        else:
            current_model_wait().tool_context.task_metadata["deadline_at"] = "2000-01-01T00:00:00+00:00"

    monkeypatch.setattr(model_concurrency, "time", SimpleNamespace(
        monotonic=clock.monotonic, time=clock.time, sleep=first_failed_acquisition))
    frames = []
    bus = SimpleNamespace(put=lambda event, *_a, **_k: frames.append(event),
                          put_nowait=lambda event, *_a, **_k: frames.append(event))
    agent = OuroborosAgent(Env(repo_dir=REPO, drive_root=drive), event_queue=bus)
    agent.owner_wait_callback = partial(worker_owner_wait, 0, stdqueue.Queue(), bus)
    try:
        if control == "pause":
            from supervisor.worker_process import run_worker_task

            run_worker_task(agent, task, wid=4, out_q=bus, repo_dir=str(REPO), drive_root=str(drive))
            assert not any(frame.get("type") == "task_done" for frame in frames)
            events = [frame for frame in frames if frame.get("type") == "budget_pause"]
        else:
            events = agent.handle_task(task)
    finally:
        for still in holding:
            still.__exit__(None, None, None)

    assert waits == [[]], "the count waited on the busy slot before counting anything"
    assert (load_task_result(drive, task_id) or {}).get("reason_code") != "task_exception"
    if control != "release":
        assert counted == sent == ready_seals == []
        assert not agent._busy, "the interrupted worker released its turn"
    if control == "pause":
        [event] = events
        paused = event["resource_limit"]
        assert event["type"] == "budget_pause" and paused["before_task_start"] is True
        assert not paused.get("checkpoint") and not budget_pause.budget_pause_row(drive, task_id)
        _handle_budget_pause(event, _supervisor_ctx(drive, workers, queue, [], []))
        [pending] = workers.PENDING
        assert pending["text"] == work and pending["id"] == task_id and pending["_attempt"] == 1
        assert not workers.RUNNING and worker.busy_task_id is None and owner_pause.fence_closed(owner_pause.read_fence(drive, task_id))
        granted = queue.resume_budget_paused_task(task_id)
        assert granted["ok"] and granted["owner_pause_released"], granted
        [resumed] = workers.PENDING
        assert not resumed.get("_budget_pause_resume")
        # Exercise the resumed worker on the same agent, not a separately constructed fresh task.
        agent.handle_task(resumed)
    elif control == "stop":
        assert events == [] and cancel_intents.active_intent(drive, task_id), "custody settles the Stop"
        assert not budget_pause.budget_pause_row(drive, task_id)
    elif control in {"deadline", "wrap_up"}:
        [metrics] = [event for event in events if event.get("type") == "task_metrics"]
        expected = "deadline_local" if control == "deadline" else "owner_requested_finalization"
        assert metrics["reason_code"] == expected
        assert not budget_pause.budget_pause_row(drive, task_id)
    if control in {"release", "pause"}:
        assert agent.tools._ctx.context_fit_plan.counter_status == "exact"
        assert load_task_result(drive, task_id)["status"] == "completed"
        assert any(work in str(row.get("content", "")) for row in counted[0][0]["messages"]), "the real exact count"
        assert len(sent) == 1 and any(work in str(row.get("content", "")) for row in sent[0]["messages"])
        assert "[Exact task input source]" not in json.dumps(sent[0]["messages"])


def test_known_empty_and_single_read_typed_failures(tmp_path, monkeypatch):
    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    _install_synthetic_local(monkeypatch, memory.drive_root)
    monkeypatch.setattr(deployment_policy, "organization_rules", lambda: "")
    normal_setting = config.runtime_setting
    monkeypatch.setattr(config, "runtime_setting",
                        lambda key, *a, **kw: "" if key == "OUROBOROS_SAFETY_OWNER_RULES" else normal_setting(key, *a, **kw))
    memory.world_path().write_text("", encoding="utf-8")
    bp = memory.scratchpad_blocks_path()
    bp.write_text("[]", encoding="utf-8")
    task, ctx = _web_main_task_and_ctx(env, memory, registry, task_id="completeness", text="Check.")

    def fit(mode="nano", *, used_env=env, used_ctx=ctx):
        return context.build_context_fit_plan(used_env, memory, task, preferred_mode=mode, ctx=used_ctx,
                                              tool_schemas=tools, tool_registry=registry, route_resolver=_local_route_resolver())

    plan_empty = fit()
    snap_empty = memory_view.snapshot_from_json(plan_empty.core.memory_view_json)
    assert plan_empty.core.read_failures == () and nano_minimum.completeness_gap(plan_empty.core, snap_empty) == ""

    quoted = ("Shared understanding source unavailable: prose. Knowledge navigation unavailable: ValueError. "
              "Source unreadable here (OSError): memory/registry.md. Scratchpad source unreadable (ValueError). "
              "Identity source unreadable (OSError). (row unreadable by its address)")
    memory.identity_path().write_text("# Identity\n\n" + quoted, encoding="utf-8")
    memory.append_scratchpad_block("Historical quote: " + quoted)
    dialogue_store.append(memory.drive_root, "chat", {"ts": "2026-02-01T11:30:00+00:00", "chat_id": 1, "user_id": 1,
                                                      "direction": "in", "source": "web", "client_message_id": "q1", "text": quoted})

    orig_read_text, block_reads = Path.read_text, 0

    def counting_read_text(self, *a, **kw):
        nonlocal block_reads
        block_reads += int(self == bp)
        return orig_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    for mode in ("max", "low", "nano"):
        block_reads = 0
        built = fit(mode)
        assert block_reads == 1 and built.core.read_failures == ()
    monkeypatch.setattr(Path, "read_text", orig_read_text)
    assert LLMClient(api_key="unused")._local_publication_floor(built, 16_384, clock_at=FIXED_NOW).count()["status"] == "exact"

    memory.scratchpad_path().write_text("# Scratchpad\n\nPreserved markdown fallback.\n", encoding="utf-8")
    for bad_blocks in ("{}", "{corrupt"):
        bp.write_text(bad_blocks, encoding="utf-8")
        bad_plan = fit()
        assert "scratchpad" in bad_plan.core.read_failures
        assert "Preserved markdown fallback." in json.dumps(bad_plan.projection("low").system_message(), ensure_ascii=False)
        floor_obs = LLMClient(api_key="unused")._local_publication_floor(bad_plan, 16_384, clock_at=FIXED_NOW).count()
        assert floor_obs["status"] == "unknown" and floor_obs["reason_code"] == nano_minimum.SOURCE_READ_INCOMPLETE

    bp.write_text("[]", encoding="utf-8")
    orig_open = os.open
    monkeypatch.setattr(memory_mod.os, "open",
                        lambda path, flags, mode=0o777: (_ for _ in ()).throw(OSError("lock denied"))
                        if str(path).endswith("scratchpad_blocks.json.lock") else orig_open(path, flags, mode))
    lock_fail_plan = fit()
    assert "scratchpad" in lock_fail_plan.core.read_failures
    assert "Preserved markdown fallback." in json.dumps(lock_fail_plan.projection("max").system_message(), ensure_ascii=False)
    monkeypatch.setattr(context, "_context_fit_route", lambda *_a, **_kw: _local_route_resolver()())
    pub = local_root_publication(env, memory, task, author_ctx=ctx, root_ctx=ctx, tool_registry=registry,
                                 client=LLMClient(api_key="unused"), tool_schemas=tools, caller_max=16_384,
                                 scope_revision=lambda: "rev-1")
    id_before = memory.identity_path().read_bytes()
    with pytest.raises(NanoPublicationRefused):
        with pub.publication(MemoryChange(memory.identity_path(), id_before, id_before + b"\nMore.\n")):
            pytest.fail("publication must stay closed on incomplete source read")
    monkeypatch.setattr(memory_mod.os, "open", orig_open)

    no_bible_repo = tmp_path / "repo_no_bible"
    no_bible_repo.mkdir()
    for name in ("prompts", "docs"):
        (no_bible_repo / name).symlink_to(REPO / name)
    assert "bible" in fit(used_env=replace(env, repo_dir=no_bible_repo), used_ctx=replace(ctx, repo_dir=no_bible_repo)).core.read_failures

    orig_discover = skill_catalogue.discover_context_skills
    monkeypatch.setattr(skill_catalogue, "discover_context_skills", lambda _root: WalkedSkills([], ("custom: PermissionError",)))
    assert fit().core.read_failures == ("installed_skills",)
    monkeypatch.setattr(skill_catalogue, "discover_context_skills", orig_discover)

    bad_note = knowledge.resolve_knowledge_address(memory.drive_root, "broken-utf8", "global")
    bad_note.path.parent.mkdir(parents=True, exist_ok=True)
    bad_note.path.write_bytes(b"\xff\xfe\x00invalid")
    assert "knowledge_index" in fit().core.read_failures
    bad_note.path.unlink()

    unreadable_snap = replace(snap_empty, room={**(snap_empty.room or {}), "lane1": [{"text_sha256": "", "line": "x"}]})
    assert nano_minimum.completeness_gap(plan_empty.core, unreadable_snap) == "source_read_incomplete:room_rows"

    memory.identity_path().write_bytes(b"\xff\xfe\x00invalid-identity")
    for mode in ("max", "low", "nano"):
        with pytest.raises(UnicodeDecodeError):
            fit(mode)


def test_scope_isolation_and_domain_boundary(tmp_path, monkeypatch):
    llm_local_ast = ast.parse((REPO / "ouroboros/llm_local.py").read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(llm_local_ast) if isinstance(node, ast.ImportFrom) and node.module} | {
        alias.name for node in ast.walk(llm_local_ast) if isinstance(node, ast.Import) for alias in node.names
    }
    assert not any("tool_policy" in name or name.startswith("ouroboros.tools") for name in imported)

    env, memory, registry, tools = _bootstrapped_main_drive(tmp_path, monkeypatch)
    counted, _, _ = _install_synthetic_local(monkeypatch, memory.drive_root)
    # These profiles test scope, not estimator overflow; give their ordinary path a usable input.
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    work = "Out-of-scope control: " + UNIT * 100
    remote = lambda *_a, **_kw: (  # noqa: E731
        {"model": "remote-model", "provider": "ollama", "model_role": "main"},
        SimpleNamespace(route_fp="remote", status="confirmed", stale=False, window_tokens=WINDOW, route_facts={}, source="probe"),
    )

    def run_case(task, ctx, resolver=None):
        registry.set_context(ctx)
        return context.build_llm_messages(env=env, memory=memory, task=task, ctx=ctx, tool_schemas=tools,
                                          tool_registry=registry, route_resolver=resolver or _local_route_resolver())

    direct_ctx = ToolContext(repo_dir=env.repo_dir, drive_root=memory.drive_root, current_chat_id=1,
                             current_task_type="task", task_id="direct-1", is_direct_chat=True, task_metadata={})
    run_case({"id": "direct-1", "type": "task", "source": "web", "chat_id": 1, "_is_direct_chat": True, "text": work}, direct_ctx)
    probe = SimpleNamespace(active_use_local=True, task_type="task", tools=SimpleNamespace(_ctx=direct_ctx),
                            accumulated_usage={}, round_idx=0)
    # My human's direct turn in Main starts by the exact count too (tests/test_nano_web_main_entry.py); the floor's
    # publication sample still observes queued roots alone.
    assert direct_ctx.context_fit_plan.counter_status == "exact" and len(counted) == 1
    assert nano_minimum.measurement_episode(probe, direct_ctx.context_fit_plan) == ""
    counted.clear()

    proj_ctx = ToolContext(repo_dir=env.repo_dir, drive_root=memory.drive_root, current_chat_id=2, project_id="p1",
                           current_task_type="task", task_id="proj-1", is_direct_chat=False, task_metadata={"project_id": "p1"})
    run_case({"id": "proj-1", "type": "task", "source": "web", "chat_id": 2, "project_id": "p1", "delegation_role": "root", "text": work}, proj_ctx)
    assert nano_minimum._origin(proj_ctx.context_fit_plan)["room"] == "project"

    sub_ctx = ToolContext(repo_dir=env.repo_dir, drive_root=memory.drive_root, current_chat_id=1,
                          current_task_type="task", task_id="sub-1", task_metadata={"delegation_role": "subagent"})
    run_case({"id": "sub-1", "type": "task", "chat_id": 1, "parent_task_id": "root-1", "delegation_role": "subagent", "text": work}, sub_ctx)
    run_case(*_web_main_task_and_ctx(env, memory, registry, task_id="remote-1", text=work), resolver=remote)

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    run_case(*_web_main_task_and_ctx(env, memory, registry, task_id="low-1", text=work))

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "nano")
    t_res, c_res = _web_main_task_and_ctx(env, memory, registry, task_id="resume-empty", text=work)
    c_res._held_note_capture = {"entries": []}
    resumed_msgs, _ = run_case(t_res, c_res)
    assert counted == [] and work in resumed_msgs[1]["content"]
