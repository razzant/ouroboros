"""Presence's owner-selected working folder keeps one canonical memory."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.presence_admission import admit_presence_turn
from tests.frozen_presence_capabilities import (
    PresenceState,
    PresenceStateError,
    load_presence_state,
    presence_state_fingerprint,
    save_presence_state,
)
from ouroboros.presence_runner import _build_task
from ouroboros.project_facts import resolve_project_id
from ouroboros.skill_loader import SkillReviewState, load_skill, save_enabled, save_review_state
from ouroboros.tools.presence import _configure_presence
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_presence_admission import _binding
from tests.test_presence_runner import _event

pytestmark = pytest.mark.serial


@pytest.fixture
def installed(tmp_path, monkeypatch):
    repo, data, workspace = (tmp_path / name for name in ("repo", "data", "workspace"))
    for path in (repo, data, workspace):
        path.mkdir()
    monkeypatch.setattr("ouroboros.config.DATA_DIR", data)
    from supervisor import queue, state

    monkeypatch.setattr(state, "DRIVE_ROOT", data)
    monkeypatch.setattr(state, "STATE_PATH", data / "state" / "state.json")
    monkeypatch.setattr(queue, "DRIVE_ROOT", data)
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")
    skill_dir = data / "skills" / "external" / "community-helper"
    skill_dir.mkdir(parents=True)
    tools = ("read_file", "write_file", "run_script", "schedule_subagent")
    requests = "".join(
        f"    - id: {name}\n      kind: tool\n      required: true\n      purpose: Use {name}.\n"
        for name in tools
    )
    (skill_dir / "SKILL.md").write_text(
        "---\nname: community-helper\ndescription: Generic workspace fixture.\n"
        "version: 0.1.0\ntype: instruction\npresence:\n"
        "  instructions: Work in the selected folder.\n  capability_requests:\n"
        + requests
        + "    - id: files\n      kind: resource\n      required: true\n"
        "      purpose: Work with files.\n      operations: [read, write, shell]\n"
        "---\n# Community helper\n",
        encoding="utf-8",
    )
    skill = load_skill(skill_dir, data)
    save_enabled(data, skill.name, True)
    save_review_state(data, skill.name, SkillReviewState(status="pass", content_hash=skill.content_hash))
    ctx = ToolContext(repo_dir=repo, drive_root=data)
    binding = _binding(data)
    from ouroboros.places import load_places, edit_place
    place = load_places(data)[0]
    edit_place(data, place.place_id, "set_working_folder", author="owner", working_folder=str(workspace))
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available", "remaining_usd": 10})
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [])
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "full")
    return SimpleNamespace(repo=repo, data=data, workspace=workspace, ctx=ctx,
                           skill=skill, binding=binding, skill_dir=skill_dir)


def _admit(installed):
    return admit_presence_turn(
        drive_root=installed.data, authenticated_transport_skill=installed.binding.transport_skill,
        binding_id=installed.binding.binding_id, global_max_rounds=20,
    )


def _context(installed, task):
    return ToolContext(
        repo_dir=installed.repo, system_repo_dir=installed.repo, drive_root=installed.data,
        task_id=task["id"], current_chat_id=task["chat_id"],
        task_contract=task["task_contract"], task_metadata=task["metadata"],
        workspace_root=Path(task["workspace_root"]), workspace_mode=task["workspace_mode"],
        memory_mode=task["memory_mode"], project_id=resolve_project_id(task),
    )


def test_legacy_state_shape_and_fingerprint_stay_identical(tmp_path):
    payload = {
        "schema_version": 1, "selections": [],
        "runtime_overrides": {"model_slot": None, "inline_max_rounds": None},
    }
    from tests.frozen_presence_capabilities import _domain_sha256

    expected = _domain_sha256("ouroboros.presence.state.v1", payload)
    assert presence_state_fingerprint(PresenceState()) == expected
    saved = save_presence_state(tmp_path, "helper", PresenceState(), expected_state_fingerprint=expected)
    assert saved.workspace_root == ""
    from ouroboros.contracts.skill_payload_policy import PRESENCE_PROFILE_STATE_FILENAME

    path = tmp_path / "state" / "skills" / "helper" / PRESENCE_PROFILE_STATE_FILENAME
    before = path.read_bytes()
    assert json.loads(before) == payload
    assert load_presence_state(tmp_path, "helper") == PresenceState()
    assert path.read_bytes() == before
    assert presence_state_fingerprint(replace(saved, workspace_root=str(tmp_path))) != expected
    with pytest.raises(PresenceStateError):
        PresenceState(workspace_root="relative/folder")


def test_owner_changes_preserve_admitted_snapshot(installed):
    original = _admit(installed)
    other = installed.workspace.with_name("other")
    other.mkdir()
    _configure_presence(installed.ctx, "set_working_folder", place_id=original.place.place_id, working_folder=str(other))
    updated = _admit(installed)
    assert updated.workspace_root == str(other)
    assert original.workspace_root == str(installed.workspace)
    with pytest.raises(FrozenInstanceError):
        original.workspace_root = str(other)
    task = _build_task(original, _event(), drive_root=installed.data, staged_files=())
    assert task["task_contract"]["workspace"] == {"root": str(installed.workspace), "mode": "external"}
    _configure_presence(installed.ctx, "set_working_folder", place_id=original.place.place_id, working_folder="")
    cleared = _build_task(_admit(installed), _event(), drive_root=installed.data, staged_files=())
    assert "workspace_root" not in cleared and "memory_mode" not in cleared


def test_folder_remains_an_address_even_if_temporarily_unavailable(installed):
    original = _admit(installed)
    installed.workspace.rmdir()
    assert _admit(installed).workspace_root == original.workspace_root


def test_admitted_registry_reads_writes_and_runs_script_in_external_folder(installed):
    core = installed.repo / "unchanged.txt"
    core.write_text("system content", encoding="utf-8")
    before = hashlib.sha256(core.read_bytes()).hexdigest()
    task = _build_task(_admit(installed), _event(), drive_root=installed.data, staged_files=())
    ctx = _context(installed, task)
    registry = ToolRegistry(repo_dir=installed.repo, drive_root=installed.data)
    registry.set_context(ctx)
    assert ctx.active_repo_dir() == installed.workspace
    assert ctx.project_id == ""
    written = registry.execute("write_file", {"path": "report.txt", "content": "folder content"})
    assert "Written" in written, written
    read = registry.execute("read_file", {"path": "report.txt"})
    assert "folder content" in read, read
    script = registry.execute("run_script", {
        "interpreter": sys.executable,
        "script": "from pathlib import Path\nPath('script.txt').write_text(str(Path.cwd()))\nprint(Path.cwd())\n",
    })
    assert str(installed.workspace) in script, script
    assert (installed.workspace / "script.txt").read_text() == str(installed.workspace)
    result = registry.execute("knowledge_write", {"topic": "workspace-experience", "content": "Remember this work."})
    assert "Remember this work." in (installed.data / "memory" / "knowledge" / "workspace-experience.md").read_text(), result
    assert not (installed.data / "projects").exists()
    assert not (installed.data / "state" / "headless_tasks").exists()
    assert sorted(path.name for path in installed.repo.iterdir()) == ["unchanged.txt"]
    assert hashlib.sha256(core.read_bytes()).hexdigest() == before


def test_focus_promotion_uses_ordinary_scope_and_followup_keeps_admitted_folder(installed, monkeypatch):
    from ouroboros.tools.control import _build_child_subagent_contract, _start_task
    from ouroboros.tools.followup import _handle_schedule_followup
    from supervisor import queue, workers
    from tests.test_promote_chat_flow import _confirm_promote

    task = _build_task(_admit(installed), _event(), drive_root=installed.data, staged_files=())
    from ouroboros.usage_admission import task_billing_fields
    from ouroboros.task_results import write_task_result
    billing = task_billing_fields(task, task["id"], 0.75, installed.data, pin_initial=True)
    write_task_result(installed.data, task["id"], "running", root_task_id=task["id"], metadata=task["metadata"])
    ctx = _context(installed, task)
    _confirm_promote(monkeypatch)
    monkeypatch.setattr(workers, "DRIVE_ROOT", installed.data)
    monkeypatch.setattr(workers, "REPO_DIR", installed.repo)
    result = _start_task(ctx, "Finish the report", title="Report finish", workspace_root=str(installed.workspace),
                                   project_name="Selected")
    assert result.startswith("OK: task"), result
    event = ctx.pending_events[0]
    enqueued = []
    outcome = workers.promote_chat_to_task(event, SimpleNamespace(
        enqueue_task=lambda row: enqueued.append(row) or row,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    ))
    assert outcome["status"] == "scheduled", outcome
    promoted = enqueued[0]
    assert promoted["workspace_root"] == str(installed.workspace)
    assert promoted["memory_mode"] == "forked"
    assert promoted.get("drive_root") and promoted["project_id"] == "selected"
    assert resolve_project_id(promoted) == "selected"

    queue.init(installed.data)
    queue.init_queue_refs([], {}, {"value": 0})
    for params in ({"run_at": "2030-01-01T00:00:00Z"}, {"cron": "0 9 * * *", "timezone": "UTC"}):
        result = _handle_schedule_followup(ctx, objective="Revisit the report", relation="related", **params)
        assert result.startswith("FOLLOWUP_SCHEDULED"), result
    # A future configuration cannot retarget work already admitted/scheduled.
    _configure_presence(installed.ctx, "set_working_folder", place_id=_admit(installed).place.place_id, working_folder="")
    for record in queue.list_scheduled_tasks(installed.data)["tasks"]:
        assert record["followup_relation"]["kind"] == "related"
        assert record["followup_relation"]["billing_group"] == {
            key: value for key, value in billing.items() if key.startswith("billing_group_")}
        scheduled = queue._task_from_schedule(record)
        assert scheduled["workspace_root"] == str(installed.workspace)
        assert scheduled["workspace_mode"] == "external" and scheduled["memory_mode"] == "shared"
        assert scheduled["task_contract"]["workspace"] == task["task_contract"]["workspace"]
        assert not scheduled.get("drive_root") and resolve_project_id(scheduled) == ""
    child = _build_child_subagent_contract({
        "tid": "child", "objective": "Inspect report", "expected_output": "Findings",
        "parent_contract": promoted["task_contract"],
        "workspace_root": str(installed.workspace), "workspace_mode": "external",
    })
    assert child["workspace"] == task["task_contract"]["workspace"]
    assert "capability_ceiling" not in child and "capability_ceiling" not in task["task_contract"]
    assert (installed.data / "state" / "headless_tasks").exists()


def test_explicit_project_choice_still_wins_over_presence_folder(installed):
    task = _build_task(_admit(installed), _event(), drive_root=installed.data, staged_files=())
    assert resolve_project_id({**task, "project_id": "chosen-project"}) == "chosen-project"
    ordinary = {"workspace_root": str(installed.workspace)}
    assert resolve_project_id(ordinary).startswith("proj_")


def test_promotion_with_an_unusable_presence_folder_returns_the_typed_repair_and_sends_nothing(
    installed, monkeypatch,
):
    """A requested folder vanishes before ordinary promotion admission.

    The typed repair names that request; admission still sends no public text.
    """
    import shutil

    from ouroboros.tools.control import _start_task
    from supervisor import workers
    from tests.test_promote_chat_flow import _confirm_promote

    task = _build_task(_admit(installed), _event(), drive_root=installed.data, staged_files=())
    ctx = _context(installed, task)
    _confirm_promote(monkeypatch)
    monkeypatch.setattr(workers, "DRIVE_ROOT", installed.data)
    monkeypatch.setattr(workers, "REPO_DIR", installed.repo)
    result = _start_task(ctx, "Finish the report", title="Report finish", workspace_root=str(installed.workspace))
    assert result.startswith("OK: task"), result
    event = ctx.pending_events[0]
    shutil.rmtree(installed.workspace)  # the admitted folder disappears before admission

    sent, enqueued = [], []
    outcome = workers.promote_chat_to_task(event, SimpleNamespace(
        enqueue_task=lambda row: enqueued.append(row) or row,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
        send_with_budget=lambda *args, **kwargs: sent.append((args, kwargs)),
    ))

    assert (outcome["status"], outcome["reason"]) == ("needs_manual_target", "workspace_unusable")
    assert "This task asked for" in outcome["detail"] and "explicitly" in outcome["detail"]
    assert str(installed.workspace) in outcome["detail"]
    assert "Presence profile" not in outcome["detail"]
    assert sent == [] and enqueued == []
