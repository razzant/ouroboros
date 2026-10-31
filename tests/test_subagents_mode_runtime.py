"""Room Off refuses new helpers while admitted work and the review pool still run."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import config, subagent_runtime as runtime
from ouroboros.composer_cognition import update_project_cognition
from ouroboros.projects_registry import create_project
from ouroboros.settings_integrity import live_subagents_mode, task_settings_scope
from tests.test_project_cognition_runtime import snapshot
from tests.test_task_settings_snapshot import document, isolated_runtime, save_owner  # noqa: F401


def configured():
    values = document("old")
    catalog = json.loads(values["OUROBOROS_SUBAGENTS"])
    catalog["items"][0]["delivery"] = "packet"
    catalog["items"].append({"subagent_id": "session", "recommended_use": "Implementation",
        "route": {"kind": "agent_session", "target_id": "codex=gpt-5.6-sol", "credential_profile_id": "profile-1"}})
    values["OUROBOROS_SUBAGENTS"] = json.dumps(catalog)
    save_owner(values)
    create_project(config.DATA_DIR, "alpha")
    return values


def test_live_off_refuses_schedule_and_session_but_same_task_review_runs(monkeypatch):
    from ouroboros import review_substrate
    from ouroboros.tools import plan_review_runtime
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    from ouroboros.tools import delegate
    from ouroboros.delegate_shared import delegate_result

    configured()
    task = {"id": "parent", "project_id": "alpha"}
    ctx = ToolContext(repo_dir=config.REPO_DIR, drive_root=config.DATA_DIR, task_id="parent")
    ctx.task_metadata = {"project_id": "alpha"}
    registry = ToolRegistry(repo_dir=config.REPO_DIR, drive_root=config.DATA_DIR)
    registry.set_context(ctx)
    started, reviewed = [], []

    def start(*args, **kwargs):
        started.append(True)
        return delegate_result({"status": "started", "run_id": "existing-run"})

    def review(request, *, slots, **kwargs):
        reviewed.extend(slot.slot_id for slot in slots)
        return SimpleNamespace(actors=[{"slot_id": s.slot_id, "model": s.model, "status": "ok",
            "raw_text": '[{"item":"x","verdict":"PASS","severity":"advisory","reason":"checked"}]',
            "usage": {}, "prompt_ref": {}, "response_ref": {}} for s in slots])

    monkeypatch.setattr(delegate, "_delegate_start", start)
    monkeypatch.setattr(review_substrate, "run_review_request", review)
    monkeypatch.setattr(plan_review_runtime, "LLMClient", lambda *a, **k: object())
    with task_settings_scope(snapshot(task)):
        admitted, _ = runtime.select_subagent_snapshot(config.runtime_settings(), subagent_id="session")
        for mode in ("auto", "swarm", "off", "auto"):
            update_project_cognition(config.DATA_DIR, "alpha", subagents_mode=mode)
            assert live_subagents_mode() == mode
            result = runtime.exact_start(ctx, "work", {"subagent_id": "session"})
            if mode == "off":
                assert "subagents_off_in_room" in result.text
                assert runtime.SUBAGENTS_OFF_DETAIL in result.text
                scheduled = registry.execute_result("schedule_subagent", {
                    "subagent_id": "critic", "objective": "Check", "expected_output": "Findings"})
                assert scheduled.meta["reason"] == "subagents_off_in_room"
                assert not runtime.current_model_visible_subagent_catalog()
                assert runtime.validate_subagent_snapshot(admitted) == admitted
                slots = plan_review_runtime.plan_review_slots()
                raw = asyncio.run(plan_review_runtime.run_plan_review_slots(
                    ctx, slots, system_prompt="Review", user_content="Changes"))
                assert reviewed == ["critic"] and raw[0]["slot_id"] == "critic"
                assert not raw[0]["error"]
            else:
                assert "started" in result.text
                assert runtime.select_subagent_snapshot(config.runtime_settings(), subagent_id="critic")[0]
                assert runtime.current_model_visible_subagent_catalog()
        assert len(started) == 3


@pytest.mark.parametrize("task", [
    {"type": "evolution", "metadata": {"work_kind": "owner"}},
    {"metadata": {"evolution_root_task_id": "evolve", "work_kind": "owner"}},
    {"metadata": {"initiator": "consciousness"}},
    {"metadata": {"work_kind": "consciousness"}},
])
def test_autonomous_work_keeps_its_own_controls(task):
    configured()
    update_project_cognition(config.DATA_DIR, "alpha", subagents_mode="off")
    task = {**task, "id": "autonomous", "project_id": "alpha"}
    with task_settings_scope(snapshot(task)):
        assert live_subagents_mode() == "auto"
        assert runtime.select_subagent_snapshot(config.runtime_settings(), subagent_id="critic")[0]
    assert live_subagents_mode(task={"id": "owner", "project_id": "alpha"}) == "off"


def test_an_admitted_child_finishes_its_bound_session_after_off(monkeypatch):
    from ouroboros.tools import delegate
    from ouroboros.delegate_shared import delegate_result

    configured()
    with task_settings_scope(snapshot({"id": "child", "project_id": "alpha"})):
        admitted, _ = runtime.select_subagent_snapshot(config.runtime_settings(), subagent_id="session")
        ctx = SimpleNamespace(task_id="child", drive_root=config.DATA_DIR,
                              _configured_actor_bootstrap={"zero_run_receipt_recorded": False})
        monkeypatch.setattr("ouroboros.subagent_bootstrap._durable_zero_run_receipt", lambda *a, **k: {})
        monkeypatch.setattr(delegate, "_delegate_start", lambda *a, **k: delegate_result({"status": "started", "run_id": "run"}))
        update_project_cognition(config.DATA_DIR, "alpha", subagents_mode="off")
        result = runtime.exact_start(ctx, "admitted work", {"snapshot": admitted, "compiled_work_order": True})
        assert "started" in result.text and ctx._configured_actor_bootstrap["physical_started"]
        assert "subagents_off_in_room" in runtime.exact_start(ctx, "new work", {"subagent_id": "session"}).text


@pytest.mark.parametrize("mode", ["off", "auto", "swarm"])
@pytest.mark.parametrize("inputs", ["shared", "declared"])
def test_both_context_builders_omit_only_the_off_catalog(mode, inputs, monkeypatch, tmp_path):
    from ouroboros.context import _capture_context_core
    from ouroboros.memory import Memory
    from tests.test_available_subagents_catalog import _context_env

    values = configured()
    values["OUROBOROS_SUBAGENTS_MODE"] = mode
    save_owner(values)
    env = _context_env(tmp_path)
    task = {"id": "task", "type": "task", "text": "work", "delegation_role": "subagent",
            "task_contract": {"input_sources": inputs},
            "configured_subagent": {"route": {"kind": "api_model", "target_id": "fixture"}}}
    core = _capture_context_core(env, Memory(drive_root=tmp_path), task, None, None)
    assert ("## Available subagents" in core.semi_stable_text) is (mode != "off")
    assert "Review" in core.semi_stable_text


@pytest.mark.parametrize("mode", ["off", "auto", "swarm"])
@pytest.mark.parametrize("presence", [False, True], ids=["owner", "presence"])
def test_host_rechecks_room_mode_on_delayed_admission(mode, presence, monkeypatch):
    from supervisor import events, events_schedule_task
    from ouroboros.dialogue_provenance import PRESENCE_BINDING_AUTHORITY_KEY
    from tests.test_nested_rights_depth import _fake_ctx, _schedule_event
    from ouroboros.task_results import load_task_result

    values = configured()
    values["OUROBOROS_SUBAGENTS_MODE"] = mode
    save_owner(values)
    enqueued = []
    ctx = _fake_ctx(config.DATA_DIR, enqueued)
    event = _schedule_event("new-child", "parent", drive_root=config.DATA_DIR)
    carrier = {PRESENCE_BINDING_AUTHORITY_KEY: {"binding_id": "place"}} if presence else {}
    event.update(carrier)
    event["origin_metadata"] = {"work_kind": "owner"}
    refused = mode == "off" and not presence
    with task_settings_scope(snapshot({"id": "parent", "metadata": {**event["origin_metadata"], **carrier}})):
        if refused:
            with pytest.raises(runtime.SubagentSelectionError, match="Subagents are off in this chat"):
                runtime._require_room_subagents()
        else:
            runtime._require_room_subagents()

    checks = []
    read_mode = events_schedule_task._subagent_room_mode

    def observe(host, task):
        result = read_mode(host, task)
        checks.append(result)
        return result

    monkeypatch.setattr(events_schedule_task, "_subagent_room_mode", observe)
    events._handle_schedule_task(event, ctx)
    if refused:
        assert checks == [("off", None)]
        assert not enqueued
        assert load_task_result(config.DATA_DIR, "new-child")["reason_code"] == "subagents_off_in_room"
    else:
        assert checks == [("auto" if presence else mode, None)] * 2
        assert len(enqueued) == 1
        if presence:
            assert enqueued[0]["metadata"][PRESENCE_BINDING_AUTHORITY_KEY] == carrier[PRESENCE_BINDING_AUTHORITY_KEY]


def test_global_settings_off_withholds_new_admission_from_running_actor():
    from ouroboros.tools.registry import ToolContext

    values = configured()
    ctx = ToolContext(repo_dir=config.REPO_DIR, drive_root=config.DATA_DIR, task_id="parent")
    with task_settings_scope(snapshot({"id": "parent", "project_id": "alpha"})):
        catalog = json.loads(values["OUROBOROS_SUBAGENTS"])
        catalog["enabled"] = False
        save_owner({**values, "OUROBOROS_SUBAGENTS": json.dumps(catalog)})
        assert not runtime.current_model_visible_subagent_catalog()
        result = runtime.exact_start(ctx, "new work", {"subagent_id": "session"})
        assert "subagents_disabled" in result.text


def test_room_switch_reaches_next_round_and_next_admission_without_rewriting_prefix(monkeypatch, tmp_path):
    from ouroboros import loop
    from ouroboros.loop_messages import append_context_facts
    from tests.test_round_facts_line import _facts_ctx
    from tests.test_loop_compaction import _fit
    from ouroboros.tools import delegate
    from ouroboros.tools.registry import ToolContext
    from ouroboros.delegate_shared import delegate_result

    configured()
    ctx = _facts_ctx(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *a, **k: _fit())
    monkeypatch.setattr(delegate, "_delegate_start", lambda *a, **k: delegate_result({"status": "started", "run_id": "run"}))
    child_ctx = ToolContext(repo_dir=config.REPO_DIR, drive_root=config.DATA_DIR, task_id="parent")
    prefix = list(ctx.messages)
    with task_settings_scope(snapshot({"id": "parent", "project_id": "alpha"})):
        for mode in ("auto", "swarm", "off", "auto"):
            update_project_cognition(config.DATA_DIR, "alpha", subagents_mode=mode)
            ctx.round_idx += 1
            assert append_context_facts(ctx, money={})
            assert ctx.messages[-1]["content"].splitlines()[-1] == f"Subagents: {mode.title()} — set by my human for this chat"
            assert "cache_control" not in ctx.messages[-1]
            result = runtime.exact_start(child_ctx, "new work", {"subagent_id": "session"})
            assert ("subagents_off_in_room" in result.text) is (mode == "off")
            assert ("started" in result.text) is (mode != "off")
        assert ctx.messages[:len(prefix)] == prefix
        # A late owner switch also refreshes an unsent prepare in the SAME round.
        update_project_cognition(config.DATA_DIR, "alpha", subagents_mode="swarm")
        assert append_context_facts(ctx, money={})
        assert "Subagents: Swarm" in ctx.messages[-1]["content"]
        assert not append_context_facts(ctx, money={})


@pytest.mark.parametrize("mode", ["off", "auto", "swarm"])
def test_loading_legacy_review_slots_never_reads_the_room(mode, monkeypatch):
    from ouroboros import review_pool_migration, settings_integrity
    from ouroboros.reviewer_slot_config import review_pool_rows
    from tests.test_review_pool_migration import anton_document

    def unexpected_room_read(*args, **kwargs):
        pytest.fail("Review slot resolution must not read the room or reload settings")

    monkeypatch.setattr(review_pool_migration, "_MIGRATIONS_SEEN", {})
    values = {**anton_document(), "OUROBOROS_SUBAGENTS_MODE": mode}
    config.SETTINGS_PATH.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(settings_integrity, "live_subagents_mode", unexpected_room_read)
    monkeypatch.setattr(settings_integrity, "actor_subagents_availability", unexpected_room_read)
    loaded = config.load_settings()
    assert len(review_pool_rows(loaded)) == 3
    assert len(config.review_pool_migrations_seen()) == 1
    assert config.load_settings()["OUROBOROS_SUBAGENTS"] == loaded["OUROBOROS_SUBAGENTS"]
    assert len(config.review_pool_migrations_seen()) == 1


def test_round_fact_names_settings_lock_and_omits_autonomous_scope():
    from ouroboros.loop_messages import subagents_fact_line

    values = configured()
    values.update(OUROBOROS_SUBAGENTS_MODE="swarm", OUROBOROS_SUBAGENTS='{"enabled":false,"items":[]}')
    save_owner(values)
    with task_settings_scope(snapshot({"id": "owner"})):
        assert subagents_fact_line() == "Subagents: Off — off in Settings"
        assert live_subagents_mode() == "swarm"
    with task_settings_scope(snapshot({"id": "evolve", "type": "evolution"})):
        assert subagents_fact_line() == ""
