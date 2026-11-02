"""Room preferences bind each actor, survive cold return and stay live for new children."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ouroboros import config
from ouroboros.composer_cognition import update_project_cognition
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.settings_integrity import (
    actor_settings_snapshot, captured_cognition, live_effort_range, restore_cognition,
    task_settings_scope, task_settings_snapshot,
)
from tests.test_task_settings_snapshot import admitted_agent, document, isolated_runtime, save_owner  # noqa: F401


def snapshot(task):
    from ouroboros.subagent_runtime import apply_task_start_settings

    return actor_settings_snapshot(apply_task_start_settings(), task, config.DATA_DIR)


def choice(context, primary, low, high):
    return {"context_mode": context, "effort_range": {"min": low, "recommended": primary, "max": high}}


def set_room(project, cognition):
    update_project_cognition(config.DATA_DIR, project, **cognition)


def test_parallel_real_task_entries_keep_main_and_two_projects_independent():
    save_owner(document("old"))
    create_project(config.DATA_DIR, "alpha")
    create_project(config.DATA_DIR, "beta")
    alpha = choice("max", "ultra", "low", "low")
    beta = choice("nano", "low", "max", "ultra")
    set_room("alpha", alpha)
    set_room("beta", beta)
    barrier = threading.Barrier(3)

    def body(task):
        before = captured_cognition()
        barrier.wait(10)
        assert captured_cognition() == before
        assert config.get_context_mode() == before["context_mode"]
        assert config.resolve_effort("task") == before["effort_range"]["recommended"]
        return before

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(admitted_agent(body, []).handle_task, {"id": "task-" + pid, "project_id": pid})
                   for pid in ("", "alpha", "beta")]
        main, a, b = [future.result(timeout=20) for future in futures]
    assert a == alpha and b == beta
    assert main["context_mode"] == "max"
    assert main["effort_range"]["recommended"] == "high"
    assert config.get_context_mode() == "max"  # no project value was published to os.environ


def test_current_room_reaches_new_delegate_and_review_but_not_the_running_parent():
    from ouroboros.reviewer_slot_config import review_pool_slots
    from ouroboros.subagent_runtime import _leaf_effort_choice

    values = document("old")
    row = json.loads(values["OUROBOROS_SUBAGENTS"])
    row["items"][0]["effort"] = ""  # Auto; explicit row pins retain their separate precedence
    values["OUROBOROS_SUBAGENTS"] = json.dumps(row)
    save_owner(values)
    create_project(config.DATA_DIR, "alpha")
    create_project(config.DATA_DIR, "beta")
    before = choice("max", "ultra", "medium", "high")
    later = choice("nano", "low", "xhigh", "max")
    set_room("alpha", before)
    with task_settings_scope(snapshot({"id": "parent", "project_id": "alpha"})):
        set_room("alpha", later)
        set_room("beta", choice("low", "none", "none", "low"))
        assert captured_cognition() == before
        assert live_effort_range() == later["effort_range"]
        direct = _leaf_effort_choice(SimpleNamespace(task_metadata={}),
                                    {"route": {"kind": "api_model", "target_id": "openai::probe"}}, "", None, "")
        assert direct["applied"] == "xhigh"  # omitted helper = primary clamped to its own room bounds
        assert review_pool_slots()[0].effort == "max"
        with task_settings_scope(snapshot({"id": "child", "project_id": "alpha", "parent_task_id": "parent"})):
            assert captured_cognition() == later
        assert captured_cognition() == before


def test_converted_roots_and_existing_descendants_resolve_the_current_room():
    save_owner(document("old"))
    create_project(config.DATA_DIR, "alpha")
    later = choice("nano", "low", "max", "ultra")
    set_room("alpha", later)
    child = {"id": "child", "parent_task_id": "root", "root_task_id": "root", "delegation_role": "subagent"}
    with task_settings_scope(snapshot(child)):
        original = captured_cognition()
        bind_task_to_project(config.DATA_DIR, "root", "alpha", origin={"absent": "system"})
        assert live_effort_range() == later["effort_range"]
        assert captured_cognition() == original
        grandchild = {"id": "grandchild", "parent_task_id": "child", "root_task_id": "root"}
        with task_settings_scope(snapshot(grandchild)):
            assert captured_cognition() == later


def test_presence_and_unregistered_workspace_keep_main_even_with_a_project_path(tmp_path):
    save_owner(document("old"))
    row = create_project(config.DATA_DIR, "alpha", working_dir=str(tmp_path))
    set_room("alpha", choice("nano", "none", "none", "low"))
    for task in ({"id": "presence", "_presence_turn": True, "project_id": "alpha", "chat_id": row["chat_id"]},
                 {"id": "headless", "workspace_root": str(tmp_path)}):
        with task_settings_scope(snapshot(task)):
            assert config.get_context_mode() == "max"


def test_checkpoint_carries_only_cognition_and_cold_restore_keeps_it_while_children_read_current():
    from ouroboros.owner_wait import continuation_state, restore_continuation_state

    save_owner(document("old"))
    create_project(config.DATA_DIR, "alpha")
    before = choice("max", "ultra", "low", "medium")
    later = choice("nano", "none", "high", "max")
    set_room("alpha", before)
    task = {"id": "parent", "project_id": "alpha"}
    ctx = SimpleNamespace(task_id="parent", active_context_mode="low", active_effort="ultra")
    with task_settings_scope(snapshot(task)):
        state = continuation_state(ctx, [], {}, {}, 3, [], set())
    assert state["cognition"] == before
    assert set(state["cognition"]) == {"context_mode", "effort_range"}
    assert state["route"]["active_context_mode"] == "low"  # derived view is a separate fact
    set_room("alpha", later)
    with task_settings_scope(snapshot(task)):
        assert captured_cognition() == later
        restore_continuation_state(SimpleNamespace(_ctx=ctx), state, [], {}, {}, set())
        assert captured_cognition() == before
        assert ctx.active_context_mode == "low"
        assert live_effort_range() == later["effort_range"]
    assert config.get_context_mode() == "max"


@pytest.mark.parametrize("carrier,loader", [
    ("_owner_wait_resume", "ouroboros.owner_wait.load_owner_wait"),
    ("_budget_pause_resume", "ouroboros.budget_pause.load_budget_pause"),
    ("_working_recovery", "ouroboros.working_checkpoint.load_recovery"),
])
def test_prepare_restores_cognition_before_dispatch_and_building_context(monkeypatch, carrier, loader):
    from ouroboros import agent

    save_owner(document("new"))
    before = choice("max", "ultra", "low", "medium")
    monkeypatch.setattr("ouroboros.owner_wait.load_owner_wait", lambda *a: {})
    monkeypatch.setattr(loader, lambda *a: {"cognition": before})
    monkeypatch.setattr(agent, "validate_task_authority_sources", lambda *a: None)
    monkeypatch.setattr(agent, "attach_task_contract", lambda task: task)
    monkeypatch.setattr(agent, "apply_consciousness_authority", lambda task: task)

    class ReachedDispatch(Exception):
        pass

    def dispatch(task):
        assert captured_cognition() == before
        assert config.get_context_mode() == "max"
        assert config.resolve_effort("task") == "ultra"
        raise ReachedDispatch

    monkeypatch.setattr(agent, "resolve_dispatch_axes", dispatch)
    obj = object.__new__(agent.OuroborosAgent)
    obj.env = SimpleNamespace(drive_root=config.DATA_DIR, drive_path=lambda part: config.DATA_DIR / part)
    task = {"id": "continued", carrier: {"source_ref": "fixture"}}
    with task_settings_scope(snapshot(task)), pytest.raises(ReachedDispatch):
        obj._prepare_task_context(task)


def test_old_checkpoint_without_cognition_retains_existing_fallback_and_scope_exits_cleanly():
    baseline = task_settings_snapshot({}, {})
    with task_settings_scope(baseline):
        before = captured_cognition()
        restore_cognition(None)
        assert captured_cognition() == before
    assert captured_cognition() is None


def test_unrelated_malformed_project_does_not_block_main_or_a_healthy_actor():
    save_owner(document("old"))
    create_project(config.DATA_DIR, "good")
    create_project(config.DATA_DIR, "bad")
    path = config.DATA_DIR / "state/projects.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    next(row for row in rows["projects"] if row["id"] == "bad")["working_dir"] = []
    path.write_text(json.dumps(rows), encoding="utf-8")
    for pid in ("", "good"):
        with task_settings_scope(snapshot({"id": "actor-" + pid, "project_id": pid})):
            assert config.get_context_mode() == "max"
            assert live_effort_range()["recommended"] == "high"
    with pytest.raises(ValueError, match="malformed routing"):
        snapshot({"id": "bad-actor", "project_id": "bad"})


@pytest.mark.parametrize("exists", [True, False])
@pytest.mark.parametrize("primary,low,high", [("ultra", "low", "ultra"), ("none", "none", "high")])
def test_env_only_legacy_effort_preserves_missing_bounds_at_every_start(monkeypatch, exists, primary, low, high):
    from ouroboros.settings_scales import effort_range
    from ouroboros.subagent_runtime import apply_task_start_settings

    if exists:
        config.SETTINGS_PATH.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", primary)
    expected = {"min": low, "recommended": primary, "max": high}
    for _ in range(2):
        assert effort_range(config.load_settings()) == expected
        with task_settings_scope(apply_task_start_settings()):
            assert effort_range() == expected


@pytest.mark.parametrize("independent", [False, True])
def test_complete_env_effort_triple_uses_the_saved_format(monkeypatch, independent):
    from ouroboros.settings_scales import EFFORT_FORMAT_KEY, effort_range

    if independent:
        config.SETTINGS_PATH.write_text(json.dumps({EFFORT_FORMAT_KEY: 1}), encoding="utf-8")
    for key, value in {"OUROBOROS_EFFORT_MIN": "low", "OUROBOROS_EFFORT_TASK": "ultra",
                       "OUROBOROS_EFFORT_MAX": "medium"}.items():
        monkeypatch.setenv(key, value)
    assert effort_range(config.load_settings()) == {
        "min": "low", "recommended": "ultra", "max": "medium" if independent else "ultra"}


@pytest.mark.parametrize("exists", [True, False])
def test_forwarded_nano_is_main_and_new_projects_choice_and_survives_restore(monkeypatch, exists):
    from ouroboros.composer_cognition import read_cognition

    if exists:
        config.SETTINGS_PATH.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "nano")
    for _ in range(2):
        with task_settings_scope(snapshot({"id": "main"})):
            assert config.get_context_mode() == config.get_owner_context_mode() == "nano"
            saved = captured_cognition()
    create_project(config.DATA_DIR, "from-nano")
    assert read_cognition(config.DATA_DIR, "from-nano")["context_mode"] == "nano"
    with task_settings_scope(snapshot({"id": "project", "project_id": "from-nano"})):
        # A project creating another room still copies Main, not its own overlay.
        set_room("from-nano", choice("low", "none", "low", "medium"))
        create_project(config.DATA_DIR, "second")
    assert read_cognition(config.DATA_DIR, "second")["context_mode"] == "nano"
    save_owner({"OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false"})
    with task_settings_scope(snapshot({"id": "resumed"})):
        assert config.get_context_mode() == "max"
        restore_cognition(saved)
        assert config.get_context_mode() == config.get_owner_context_mode() == "nano"


def test_main_copy_prefers_explicit_disk_choice_over_forwarded_context(monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "nano")
    save_owner({"OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false"})
    with task_settings_scope(snapshot({"id": "main"})):
        assert config.get_context_mode() == "max"
    assert create_project(config.DATA_DIR, "from-disk")["cognition"]["context_mode"] == "max"


def test_failed_reload_and_ambiguous_low_preserve_runtime_presence_on_main():
    prior = {"OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_EFFORT_TASK": "ultra"}
    failed_reload = actor_settings_snapshot(task_settings_snapshot({}, prior), {"id": "main"}, config.DATA_DIR)
    with task_settings_scope(failed_reload):
        assert config.get_context_mode() == "low" and config.get_owner_context_mode() == "max"
        assert config.resolve_effort("task") == "ultra"

        saved = captured_cognition()
    with task_settings_scope(task_settings_snapshot({}, {})):
        restore_cognition(saved)
        assert config.get_context_mode() == "low" and config.get_owner_context_mode() == "max"
        assert config.resolve_effort("task") == "ultra"


@pytest.mark.parametrize("project_id", ["", "alpha"])
def test_subagents_changes_are_live_while_context_effort_and_resume_stay_snapshotted(project_id):
    from ouroboros.settings_integrity import live_subagents_mode

    save_owner(document("old"))
    create_project(config.DATA_DIR, "alpha")
    task = {"id": "parent", "project_id": project_id}
    with task_settings_scope(snapshot(task)):
        captured = captured_cognition()
        assert live_subagents_mode() == "auto"
        for mode in ("swarm", "off", "auto"):
            if project_id:
                update_project_cognition(config.DATA_DIR, project_id, subagents_mode=mode)
            else:
                save_owner({**config.load_settings(), "OUROBOROS_SUBAGENTS_MODE": mode})
            assert live_subagents_mode() == mode
            restore_cognition(captured)
            assert captured_cognition() == captured and live_subagents_mode() == mode
            assert "subagents_mode" not in captured
