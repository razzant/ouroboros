from __future__ import annotations

import json
from types import SimpleNamespace

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from supervisor import evolution_lifecycle as lifecycle


def test_consciousness_model_defaults_to_main(monkeypatch):
    from ouroboros.config import get_consciousness_model

    monkeypatch.setenv("OUROBOROS_MODEL", "openai/gpt-5.5")
    monkeypatch.delenv("OUROBOROS_MODEL_CONSCIOUSNESS", raising=False)
    assert get_consciousness_model() == "openai/gpt-5.5"
    monkeypatch.setenv("OUROBOROS_MODEL_CONSCIOUSNESS", "anthropic/claude-opus-4.8")
    assert get_consciousness_model() == "anthropic/claude-opus-4.8"


def test_evolution_campaign_text_includes_objective(tmp_path, monkeypatch):
    from supervisor import queue
    from supervisor import state as supervisor_state

    supervisor_state.init(tmp_path)
    queue.init(tmp_path)
    queue.start_evolution_campaign("Improve scheduler observability", source="test")

    text = queue.build_evolution_task_text(3)

    assert "EVOLUTION CAMPAIGN" in text
    assert "Improve scheduler observability" in text
    assert "normal advisory + triad + scope review flow" in text


def test_evolution_campaign_pause_resume_preserves_history(tmp_path):
    from supervisor import queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    first = queue.start_evolution_campaign("Improve scheduler observability", source="test")
    live = state.load_state()
    live["evolution_mode_enabled"] = True
    state.save_state(live)
    transaction = lifecycle.begin_evolution_transaction("task1", cycle=1, campaign=first)
    lifecycle.update_evolution_campaign_after_task(
        "task1",
        cost_usd=0.5,
        outcome_axes={"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}},
        rounds=3,
        transaction=transaction,
    )
    queue.pause_evolution_campaign("pause")
    resumed = queue.start_evolution_campaign("", source="test")

    assert resumed["id"] == first["id"]
    assert resumed["cycles_done"] == 1
    assert resumed["history"][0]["task_id"] == "task1"
    # ABI-3 (fix-round-3): the history row producer stamps the honest cost
    # name — this row reaches /api/state (the legacy kwarg above is the
    # internal lifecycle API parameter, not a persisted key).
    assert resumed["history"][0]["accounted_upper_bound_usd"] == 0.5
    assert "cost_usd" not in resumed["history"][0]


def test_evolution_auto_stop_pauses_campaign(tmp_path, monkeypatch):
    from supervisor import queue
    from supervisor import state as supervisor_state

    supervisor_state.init(tmp_path)
    monkeypatch.setattr(queue, "send_with_budget", lambda *args, **kwargs: None)
    queue.init(tmp_path)
    queue.init_queue_refs([], {}, {"value": 0})
    queue.start_evolution_campaign("Improve", source="test")
    st = supervisor_state.load_state()
    st["evolution_mode_enabled"] = True
    st["owner_chat_id"] = 1
    st["evolution_consecutive_failures"] = 3
    supervisor_state.save_state(st)

    queue.enqueue_evolution_task_if_needed()

    assert queue.get_evolution_status_snapshot()["campaign"]["status"] == "paused"


def test_evolution_enqueue_attaches_lightweight_transaction(tmp_path, monkeypatch):
    from supervisor import queue
    from supervisor import state as supervisor_state

    supervisor_state.init(tmp_path)
    monkeypatch.setattr(queue, "send_with_budget", lambda *args, **kwargs: None)
    queue.init(tmp_path)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    queue.start_evolution_campaign("Improve", source="test")
    st = supervisor_state.load_state()
    st["evolution_mode_enabled"] = True
    st["owner_chat_id"] = 1
    supervisor_state.save_state(st)

    queue.enqueue_evolution_task_if_needed()

    assert len(pending) == 1
    tx = pending[0]["metadata"]["evolution_transaction"]
    assert tx["transaction_id"]
    assert tx["task_id"] == pending[0]["id"]
    assert tx["base_head"] or tx["base_head"] == ""
    campaign = queue.get_evolution_status_snapshot()["campaign"]
    assert campaign["active_transaction"]["transaction_id"] == tx["transaction_id"]


def test_evolution_task_completion_preserves_live_transaction_updates(tmp_path):
    from supervisor import queue
    from supervisor import state as supervisor_state

    supervisor_state.init(tmp_path)
    queue.init(tmp_path)
    campaign = queue.start_evolution_campaign("Improve", source="test")
    st = supervisor_state.load_state()
    st["evolution_mode_enabled"] = True
    supervisor_state.save_state(st)
    stale = queue.begin_evolution_transaction("task1", cycle=1, campaign=campaign)
    assert lifecycle.record_evolution_commit(
        campaign["id"], stale["transaction_id"], "task1", "abc123",
    )["ok"] is True
    lifecycle.update_evolution_transaction(
        "task1",
        preflight_status="passed",
        triad_scope_status="passed",
        push_status="skipped_or_failed",
    )

    recorded = lifecycle.update_evolution_campaign_after_task(
        "task1",
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}},
        rounds=3,
        transaction=stale,
    )
    campaign = queue.get_evolution_status_snapshot()["campaign"]

    assert recorded["accepted"] is True
    assert recorded["persisted"] is True
    assert recorded["transaction"]["commit_sha"] == "abc123"
    assert campaign["history"][0]["transaction"]["commit_sha"] == "abc123"
    assert campaign.get("transaction_history", []) == []
    assert campaign["active_transaction"]["commit_sha"] == "abc123"
    assert campaign["active_transaction"]["restart_required"] is True
    assert campaign["active_transaction"]["cycle_outcome"] == "waiting_for_restart"
    assert (tmp_path / "state" / "pending_restart_verify.json").is_file()
    assert int(campaign.get("absorbed_cycles_done") or 0) == 0
    replay_recorded = lifecycle.update_evolution_campaign_after_task(
        "task1",
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}},
        rounds=3,
        transaction=stale,
    )
    campaign = queue.get_evolution_status_snapshot()["campaign"]
    assert replay_recorded["replay"] is True
    assert replay_recorded["transaction"]["commit_sha"] == "abc123"
    assert len(campaign["history"]) == 1
    assert campaign["cycles_done"] == 1

    lifecycle.complete_evolution_campaign("previous scenario complete", cleanup_worktree=False)
    campaign = queue.start_evolution_campaign("Improve", source="test")
    no_durability = queue.begin_evolution_transaction("task2", cycle=2, campaign=campaign)
    lifecycle.update_evolution_campaign_after_task(
        "task2",
        cost_usd=0.0,
        outcome_axes={"execution": {"status": "infra_failed"}, "objective": {"status": "not_evaluated"}},
        rounds=0,
        transaction=no_durability,
    )
    campaign = queue.get_evolution_status_snapshot()["campaign"]
    assert "active_transaction" not in campaign
    assert campaign["transaction_history"][-1]["task_id"] == "task2"
    assert campaign["transaction_history"][-1]["cycle_outcome"] == "no_op"


def test_terminal_evolution_event_without_running_metadata_updates_transaction(tmp_path):
    from supervisor import queue
    from supervisor import state as supervisor_state
    from supervisor.events import _handle_task_done
    from ouroboros.evolution_checkpoints import CHECKPOINTS_REL
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from ouroboros.utils import iter_jsonl_objects

    supervisor_state.init(tmp_path)
    queue.init(tmp_path)
    campaign = queue.start_evolution_campaign("Improve", source="test")
    st = supervisor_state.load_state()
    st["evolution_mode_enabled"] = True
    supervisor_state.save_state(st)
    tx = queue.begin_evolution_transaction("task-cancel", cycle=1, campaign=campaign)
    write_task_result(tmp_path, "task-cancel", STATUS_CANCELLED, cost_usd=1.0, total_rounds=1)

    ctx = SimpleNamespace(
        RUNNING={},
        WORKERS={},
        DRIVE_ROOT=tmp_path,
        REPO_DIR=tmp_path,
        load_state=supervisor_state.load_state,
        save_state=supervisor_state.save_state,
        append_jsonl=supervisor_state.append_jsonl,
        persist_queue_snapshot=lambda reason="": None,
        bridge=SimpleNamespace(push_log=lambda event: None),
    )

    evt = {
        "type": "task_done",
        "task_id": "task-cancel",
        "task_type": "evolution",
        "result_status": "cancelled",
        "metadata": {"evolution_transaction": tx},
    }

    _handle_task_done(evt, ctx)
    _handle_task_done(evt, ctx)

    campaign = queue.get_evolution_status_snapshot()["campaign"]
    assert campaign["history"][0]["task_id"] == "task-cancel"
    assert campaign["history"][0]["transaction"]["transaction_id"] == tx["transaction_id"]
    assert "active_transaction" not in campaign
    assert campaign["transaction_history"][-1]["task_id"] == "task-cancel"
    assert campaign["transaction_history"][-1]["cycle_outcome"] == "no_op"
    assert len(campaign["history"]) == 1
    assert campaign["cycles_done"] == 1
    assert int(supervisor_state.load_state().get("evolution_consecutive_failures") or 0) == 1
    assert len(list(iter_jsonl_objects(tmp_path / CHECKPOINTS_REL))) == 1


def test_degraded_evolution_axes_count_as_failure(tmp_path):
    from supervisor import queue
    from supervisor import state as supervisor_state
    from supervisor.events import _handle_task_done
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result

    supervisor_state.init(tmp_path)
    queue.init(tmp_path)
    campaign = queue.start_evolution_campaign("Improve", source="test")
    st = supervisor_state.load_state()
    st["evolution_mode_enabled"] = True
    supervisor_state.save_state(st)
    tx = queue.begin_evolution_transaction("task-degraded", cycle=1, campaign=campaign)
    write_task_result(
        tmp_path,
        "task-degraded",
        STATUS_COMPLETED,
        cost_usd=1.0,
        total_rounds=1,
        outcome_axes={
            "lifecycle": {"status": "completed"},
            "execution": {"status": "degraded", "reason_code": "verification_failed"},
            "objective": {"status": "not_evaluated", "source": "none"},
            "artifacts": {"status": "not_applicable"},
            "review": {"status": "skipped"},
        },
    )

    ctx = SimpleNamespace(
        RUNNING={},
        WORKERS={},
        DRIVE_ROOT=tmp_path,
        REPO_DIR=tmp_path,
        load_state=supervisor_state.load_state,
        save_state=supervisor_state.save_state,
        append_jsonl=supervisor_state.append_jsonl,
        persist_queue_snapshot=lambda reason="": None,
        bridge=SimpleNamespace(push_log=lambda event: None),
    )

    _handle_task_done(
        {
            "type": "task_done",
            "task_id": "task-degraded",
            "task_type": "evolution",
            "status": STATUS_COMPLETED,
            "metadata": {"evolution_transaction": tx},
        },
        ctx,
    )

    assert int(supervisor_state.load_state().get("evolution_consecutive_failures") or 0) == 1


def test_cron_schedule_enqueues_once_when_due(tmp_path, monkeypatch):
    from supervisor import queue

    queue.init(tmp_path)
    pending = []
    running = {}
    seq = {"value": 0}
    queue.init_queue_refs(pending, running, seq)
    queue.upsert_scheduled_task({
        "id": "hourly",
        "name": "Hourly",
        "enabled": True,
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "next_run_at": "2000-01-01T00:00:00+00:00",
        "task": {
            "type": "task",
            "text": "scheduled work",
            "expected_output": "Scheduled report",
            "constraints": "No web",
            "allowed_resources": {"web": "false"},
            "deadline_at": "2026-06-04T12:00:00Z",
            "task_contract": {"success_criteria": ["report delivered"]},
        },
    })

    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()

    assert len(pending) == 1
    assert pending[0]["text"] == "scheduled work"
    assert pending[0]["type"] == "task"
    assert pending[0]["root_task_id"] == pending[0]["id"]
    assert pending[0]["actor_id"] == "scheduler"
    assert pending[0]["delegation_role"] == "root"
    assert pending[0]["expected_output"] == "Scheduled report"
    assert pending[0]["constraints"] == "No web"
    assert pending[0]["allowed_resources"] == {"web": False}
    assert pending[0]["deadline_at"] == "2026-06-04T12:00:00Z"
    # (W2) success_criteria is an input ALIAS: it arrives normalized into
    # acceptance_claims (one concept, one carrier), not double-persisted.
    contract = pending[0]["task_contract"]
    assert [c["claim"] for c in contract["acceptance_claims"]] == ["report delivered"]
    assert contract["success_criteria"] == []
    assert pending[0]["metadata"]["schedule_id"] == "hourly"


def test_cron_schedule_admission_refusal_is_terminal(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_FAILED, load_task_result
    from supervisor import queue

    queue.init(tmp_path)
    queue.init_queue_refs([], {}, {"value": 0})
    queue.upsert_scheduled_task({
        "id": "blocked-cron",
        "name": "Blocked cron",
        "enabled": True,
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "next_run_at": "2000-01-01T00:00:00+00:00",
        "task": {"type": "task", "text": "must not become a phantom"},
    })
    admitted_ids = []

    def _blocked(task, front=False):
        admitted_ids.append(task["id"])
        return {**task, "_admission_blocked": "project_routing_fence"}

    monkeypatch.setattr(queue, "enqueue_task", _blocked)
    queue.check_scheduled_tasks()

    assert len(admitted_ids) == 1
    result = load_task_result(tmp_path, admitted_ids[0])
    assert result["status"] == STATUS_FAILED
    assert result["reason_code"] == "project_routing_fence"
    schedule = queue.list_scheduled_tasks()["tasks"][0]
    assert schedule["failure_count"] == 1
    assert "project_routing_fence" in schedule["last_error"]


def test_scheduled_task_without_owner_chat_is_headless_safe(tmp_path):
    from supervisor import queue
    from supervisor import state as supervisor_state
    from ouroboros.task_results import load_task_result

    supervisor_state.init(tmp_path)
    queue.init(tmp_path)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    queue.upsert_scheduled_task({
        "id": "headless",
        "name": "Headless",
        "enabled": True,
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "next_run_at": "2000-01-01T00:00:00+00:00",
        "task": {"type": "task", "text": "scheduled work"},
    })

    queue.check_scheduled_tasks()

    assert pending[0]["chat_id"] == 0
    assert load_task_result(tmp_path, pending[0]["id"])["status"] == "scheduled"


def test_schedules_api_validates_five_field_cron(tmp_path):
    from ouroboros.gateway.schedules import api_schedules_delete, api_schedules_list, api_schedules_upsert
    from supervisor import queue

    queue.init(tmp_path)
    app = Starlette(routes=[
        Route("/api/schedules", endpoint=api_schedules_list, methods=["GET"]),
        Route("/api/schedules", endpoint=api_schedules_upsert, methods=["POST"]),
        Route("/api/schedules/{schedule_id}", endpoint=api_schedules_delete, methods=["DELETE"]),
    ])
    app.state.drive_root = tmp_path
    client = TestClient(app)

    bad = client.post("/api/schedules", json={"name": "bad", "trigger": {"type": "cron", "expr": "* * * *"}})
    assert bad.status_code == 400
    bad_semantic = client.post("/api/schedules", json={"name": "bad-semantic", "trigger": {"type": "cron", "expr": "61 * * * *"}})
    assert bad_semantic.status_code == 400
    good = client.post("/api/schedules", json={"id": "ok", "name": "ok", "trigger": {"type": "cron", "expr": "* * * * *"}})
    assert good.status_code == 200
    assert client.get("/api/schedules").json()["tasks"][0]["name"] == "ok"
    disabled = client.post("/api/schedules", json={"name": "bad-bool", "enabled": "false", "trigger": {"type": "cron", "expr": "* * * * *"}})
    assert disabled.status_code == 400
    workspace = client.post("/api/schedules", json={
        "name": "bad-workspace",
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "task": {"workspace_root": "/tmp/project", "text": "nope"},
    })
    assert workspace.status_code == 400
    nested_reserved = client.post("/api/schedules", json={
        "name": "bad-metadata",
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "task": {"text": "nope", "metadata": {"delegation_role": "subagent"}},
    })
    assert nested_reserved.status_code == 400
    nested_actor = client.post("/api/schedules", json={
        "name": "bad-actor",
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "task": {"text": "nope", "metadata": {"actor_id": "forged"}},
    })
    assert nested_actor.status_code == 400
    bad_metadata_type = client.post("/api/schedules", json={
        "name": "bad-metadata-type",
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "task": {"text": "nope", "metadata": "forged"},
    })
    assert bad_metadata_type.status_code == 400
    internal_type = client.post("/api/schedules", json={
        "name": "bad-type",
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "task": {"type": "evolution", "text": "nope"},
    })
    assert internal_type.status_code == 400
    bad_id = client.post("/api/schedules", json={
        "id": "bad/id",
        "name": "bad-id",
        "trigger": {"type": "cron", "expr": "* * * * *"},
    })
    assert bad_id.status_code == 400
    assert client.delete("/api/schedules/ok").json()["ok"] is True
    assert client.get("/api/schedules").json()["tasks"] == []


def test_skill_manifest_parses_scheduled_tasks():
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text

    manifest = parse_skill_manifest_text("""---
name: cron-demo
description: Cron demo
version: 0.1.0
type: extension
entry: plugin.py
permissions: [supervised_task]
scheduled_tasks:
  - name: refresh
    cron: "0 * * * *"
    timezone: Europe/Moscow
---
body
""")

    assert manifest.scheduled_tasks[0]["name"] == "refresh"
    assert manifest.validate() == []


def test_skill_manifest_rejects_invalid_scheduled_task_cron():
    from ouroboros.contracts.skill_manifest import SkillManifestError, parse_skill_manifest_text
    import pytest

    with pytest.raises(SkillManifestError):
        parse_skill_manifest_text("""---
name: cron-demo
description: Cron demo
version: 0.1.0
type: extension
entry: plugin.py
permissions: [supervised_task]
scheduled_tasks:
  - name: refresh
    cron: "61 * * * *"
---
body
""")


def test_skill_manifest_rejects_unsafe_scheduled_task_name():
    from ouroboros.contracts.skill_manifest import SkillManifestError, parse_skill_manifest_text
    import pytest

    with pytest.raises(SkillManifestError):
        parse_skill_manifest_text("""---
name: cron-demo
description: Cron demo
version: 0.1.0
type: extension
entry: plugin.py
permissions: [supervised_task]
scheduled_tasks:
  - name: "bad`name"
    cron: "0 * * * *"
---
body
""")


def test_skill_schedules_sync_into_core_scheduler(tmp_path):
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from supervisor import queue

    queue.init(tmp_path)
    manifest = parse_skill_manifest_text("""---
name: cron-demo
description: Cron demo
version: 0.1.0
type: extension
entry: plugin.py
permissions: [supervised_task]
scheduled_tasks:
  - name: refresh
    cron: "0 * * * *"
---
body
""")
    skill = SimpleNamespace(
        name="cron-demo",
        manifest=manifest,
        enabled=True,
        load_error="",
        content_hash="abc",
        review=SimpleNamespace(status="pass", is_stale_for=lambda _hash: False),
    )

    report = queue.sync_skill_schedules([skill])
    schedules = queue.list_scheduled_tasks()["tasks"]

    assert report["changed"] is True
    assert schedules[0]["id"] == "skill-cron-demo-refresh"
    assert schedules[0]["enabled"] is True
    assert schedules[0]["trigger"]["expr"] == "0 * * * *"


def test_skill_schedule_sync_refreshes_next_run_on_cron_change(tmp_path):
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from supervisor import queue

    queue.init(tmp_path)

    def make_skill(cron: str, content_hash: str):
        manifest = parse_skill_manifest_text(f"""---
name: cron-demo
description: Cron demo
version: 0.1.0
type: extension
entry: plugin.py
permissions: [supervised_task]
scheduled_tasks:
  - name: refresh
    cron: "{cron}"
---
body
""")
        return SimpleNamespace(
            name="cron-demo",
            manifest=manifest,
            enabled=True,
            load_error="",
            content_hash=content_hash,
            review=SimpleNamespace(status="pass", is_stale_for=lambda _hash: False),
        )

    queue.sync_skill_schedules([make_skill("0 * * * *", "a")])
    first = queue.list_scheduled_tasks()["tasks"][0]["next_run_at"]
    queue.sync_skill_schedules([make_skill("30 * * * *", "b")])
    second = queue.list_scheduled_tasks()["tasks"][0]["next_run_at"]

    assert first != second
    assert queue.list_scheduled_tasks()["tasks"][0]["trigger"]["expr"] == "30 * * * *"


def test_schedule_slug_avoids_long_name_collisions():
    from ouroboros.schedule_contract import schedule_slug

    a = schedule_slug("skill", "x" * 100, "a")
    b = schedule_slug("skill", "x" * 100, "b")
    assert a != b
    assert len(a) <= 81


def test_frontend_evolution_and_consciousness_controls_are_present():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    evolution = (root / "web" / "modules" / "evolution.js").read_text(encoding="utf-8")
    settings_ui = (root / "web" / "modules" / "settings_ui.js").read_text(encoding="utf-8")
    settings = (root / "web" / "modules" / "settings.js").read_text(encoding="utf-8")

    assert "evo-start" in evolution
    assert "evo-stop" in evolution
    assert "/evolve on" in evolution
    assert "/evolve off" in evolution
    # Start button is hard-disabled in light mode (self-modification gate).
    assert "runtime.runtime_mode" in evolution
    assert "startBtn.disabled = isLightMode" in evolution
    assert "s-model-consciousness" in settings_ui
    assert "s-local-consciousness" in settings_ui
    assert "OUROBOROS_EFFORT_CONSCIOUSNESS', 'high'" in settings


def test_evolution_checkpoint_records_and_reads(tmp_path):
    from ouroboros.evolution_checkpoints import CHECKPOINTS_REL, append_evolution_checkpoint
    from ouroboros.utils import iter_jsonl_objects

    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "identity.md").write_text("id", encoding="utf-8")

    append_evolution_checkpoint(
        tmp_path,
        repo,
        task_id="evo1",
        campaign={"id": "camp", "objective": "Improve"},
        outcome_axes={"execution": {"status": "ok"}, "objective": {"status": "not_evaluated"}},
        cost_usd=1.25,
        rounds=3,
        transaction={"transaction_id": "tx1", "commit_sha": "abc"},
    )

    rows = list(iter_jsonl_objects(tmp_path / CHECKPOINTS_REL))
    assert rows[0]["task_id"] == "evo1"
    assert rows[0]["campaign_id"] == "camp"
    assert rows[0]["rounds"] == 3
    assert rows[0]["transaction"]["transaction_id"] == "tx1"


def test_solve_capability_digest_joins_taskdone_and_resolution_rows(tmp_path):
    """Block 5C: the digest joins task-done checkpoints with later cycle_outcome
    tags (last wins per task) and renders absorbed vs failed objective history."""
    from ouroboros.evolution_checkpoints import (
        append_cycle_outcome_checkpoint,
        append_evolution_checkpoint,
        build_solve_capability_digest,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "memory").mkdir()

    assert build_solve_capability_digest(tmp_path) == ""  # no history yet

    # Cycle 1: committed, recorded waiting_for_restart at task-done, absorbed at boot.
    append_evolution_checkpoint(
        tmp_path, repo, task_id="evo1",
        campaign={"id": "c1", "objective": "Harden the review loop"},
        outcome_axes={"execution": {"status": "ok"}}, cost_usd=2.5, rounds=12,
        transaction={"transaction_id": "tx1", "commit_sha": "abc123def456", "cycle_outcome": "waiting_for_restart"},
    )
    append_cycle_outcome_checkpoint(
        tmp_path,
        campaign={"id": "c1", "objective": "Harden the review loop"},
        transaction={"task_id": "evo1", "transaction_id": "tx1", "commit_sha": "abc123def456", "cycle_outcome": "absorbed"},
        source="boot_reconcile",
    )
    # Cycle 2: honest no-op at task-done.
    append_evolution_checkpoint(
        tmp_path, repo, task_id="evo2",
        campaign={"id": "c1", "objective": "Vague mega-refactor"},
        outcome_axes={"execution": {"status": "ok"}}, cost_usd=0.5, rounds=4,
        transaction={"transaction_id": "tx2", "commit_sha": "", "cycle_outcome": "no_op"},
    )

    digest = build_solve_capability_digest(tmp_path)
    assert "absorbed=1" in digest and "no_op=1" in digest
    assert "ABSORBED: Harden the review loop" in digest
    assert "abc123def4" in digest  # commit sha shortened
    assert "NO_OP: Vague mega-refactor" in digest

    # Long objectives carry an explicit truncation marker (no silent [:N]).
    append_evolution_checkpoint(
        tmp_path, repo, task_id="evo3",
        campaign={"id": "c1", "objective": "X" * 300},
        outcome_axes={"execution": {"status": "ok"}}, cost_usd=0.1, rounds=1,
        transaction={"transaction_id": "tx3", "commit_sha": "", "cycle_outcome": "no_op"},
    )
    assert "…[truncated; full objective in the ledger]" in build_solve_capability_digest(tmp_path)


def _make_git_repo(repo):
    import subprocess

    def _git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
            cwd=str(repo), check=True, capture_output=True, text=True,
        )
    repo.mkdir(parents=True, exist_ok=True)
    _git("init", "-b", "ouroboros")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "base")
    return _git


def test_no_op_cycle_resets_dirty_worktree_to_base_with_recovery_refs(tmp_path, monkeypatch):
    """Block 5D2: a no_op cycle deterministically restores the worktree to the
    transaction's base_head — dirty files stashed, ahead-HEAD preserved as a
    local branch, both recorded on the transaction (P1: no silent loss)."""
    import subprocess

    from supervisor import git_ops, queue
    from supervisor import state as supervisor_state

    repo = tmp_path / "repo"
    _git = _make_git_repo(repo)
    base_head = _git("rev-parse", "HEAD").stdout.strip()

    # Simulate cycle leftovers: an unreviewed local commit + dirty/untracked files.
    (repo / "wip.txt").write_text("unreviewed\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "unreviewed leftover")
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    git_ops.init(repo, tmp_path, "")
    queue.init(tmp_path)
    queue.RUNNING.clear()

    campaign = queue.start_evolution_campaign("Improve", source="test")
    st = supervisor_state.load_state()
    st["evolution_mode_enabled"] = True
    supervisor_state.save_state(st)
    tx = queue.begin_evolution_transaction("task-noop", cycle=1, campaign=campaign)
    assert tx["base_head"]  # begin records the CURRENT head (after leftover commit)
    # Rewrite base to the true cycle base for the scenario.
    lifecycle.update_evolution_transaction("task-noop", base_head=base_head)

    recorded = lifecycle.update_evolution_campaign_after_task(
        "task-noop", cost_usd=0.1,
        outcome_axes={"execution": {"status": "ok"}}, rounds=2,
        transaction=tx,
    )

    recorded_tx = recorded["transaction"]
    assert recorded_tx["cycle_outcome"] == "no_op"
    assert recorded_tx["cleanup_status"] == "reset_to_base"
    assert recorded_tx["cleanup_stash"].startswith("evolution-cycle-cleanup-")
    assert recorded_tx["cleanup_preserved_ref"].startswith("evolution-leftover-")
    # Worktree is back at base and clean.
    head_now = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True).stdout.strip()
    status_now = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo), capture_output=True, text=True).stdout.strip()
    assert head_now == base_head
    assert status_now == ""
    # Recovery refs exist: preserved branch points at the leftover commit; stash kept the dirty file.
    branches = subprocess.run(["git", "branch", "--list", "evolution-leftover-*"], cwd=str(repo), capture_output=True, text=True).stdout
    assert "evolution-leftover-" in branches
    stashes = subprocess.run(["git", "stash", "list"], cwd=str(repo), capture_output=True, text=True).stdout
    assert "evolution-cycle-cleanup-" in stashes


def test_no_op_cleanup_skips_when_other_tasks_running_or_already_clean(tmp_path):
    """Block 5D2 gates: concurrent tasks in the shared worktree block the reset;
    a clean at-base tree is a recorded no-op."""
    from supervisor import git_ops, queue

    repo = tmp_path / "repo"
    _make_git_repo(repo)
    git_ops.init(repo, tmp_path, "")
    queue.init(tmp_path)

    # Other task running → skip.
    queue.RUNNING.clear()
    queue.RUNNING["other-task"] = {"task": {"id": "other-task"}}
    tx1 = {"transaction_id": "t1", "base_head": "0" * 40}
    lifecycle._cleanup_worktree_after_cycle(tx1, "task-a")
    assert tx1["cleanup_status"] == "skipped_other_tasks_running"
    queue.RUNNING.clear()

    # Clean tree at base → already_clean, nothing destroyed.
    head = git_ops.git_capture(["git", "rev-parse", "HEAD"])[1].strip()
    tx2 = {"transaction_id": "t2", "base_head": head}
    lifecycle._cleanup_worktree_after_cycle(tx2, "task-b")
    assert tx2["cleanup_status"] == "already_clean"

    # No recorded base → skip.
    tx3 = {"transaction_id": "t3"}
    lifecycle._cleanup_worktree_after_cycle(tx3, "task-c")
    assert tx3["cleanup_status"] == "skipped_no_base"


def test_evolution_cleanup_never_mutates_checkout_owned_by_update(monkeypatch):
    from supervisor import git_ops, update_merge, workers

    lock = object()
    released = []
    monkeypatch.setattr(update_merge, "acquire_update_lock", lambda: lock)
    monkeypatch.setattr(update_merge, "release_update_lock", released.append)
    monkeypatch.setattr(
        workers, "repo_writer_admission_closed", lambda: "managed_update:smart"
    )
    monkeypatch.setattr(
        git_ops,
        "git_capture",
        lambda _cmd: (_ for _ in ()).throw(
            AssertionError("evolution cleanup must not touch git behind writer fence")
        ),
    )
    tx = {"base_head": "a" * 40, "transaction_id": "evo-update-race"}

    lifecycle._cleanup_worktree_after_cycle(tx, "evolution-task")

    assert tx["cleanup_status"] == "skipped_repo_writer_admission_closed"
    assert tx["cleanup_deferred_reason"] == "managed_update:smart"
    assert released == [lock]


def test_evolution_cleanup_skips_when_update_lock_is_busy(monkeypatch):
    from supervisor import git_ops, update_merge

    monkeypatch.setattr(
        update_merge,
        "acquire_update_lock",
        lambda: (_ for _ in ()).throw(RuntimeError("busy")),
    )
    monkeypatch.setattr(
        git_ops,
        "git_capture",
        lambda _cmd: (_ for _ in ()).throw(
            AssertionError("busy update lock must stop cleanup before git")
        ),
    )
    tx = {"base_head": "a" * 40, "transaction_id": "evo-update-lock-race"}

    lifecycle._cleanup_worktree_after_cycle(tx, "evolution-task")

    assert tx["cleanup_status"] == "skipped_managed_update_lock_busy"


def test_evolution_restart_uses_local_commit_not_origin_and_blocks_dirty_tree(tmp_path):
    from ouroboros.tools.control import _request_restart
    from ouroboros.tools.registry import ToolContext
    from supervisor import evolution_lifecycle, queue
    from supervisor import state as supervisor_state

    repo = tmp_path / "repo"
    repo.mkdir()
    _git = lambda *args: __import__("subprocess").run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)
    _git("init")
    _git("checkout", "-b", "ouroboros")
    (repo / "file.txt").write_text("ok\n", encoding="utf-8")
    _git("add", ".")
    __import__("subprocess").run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )
    head = __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout.strip()
    supervisor_state.init(tmp_path)
    queue.init(tmp_path)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    st = supervisor_state.load_state()
    st["evolution_mode_enabled"] = True
    supervisor_state.save_state(st)
    tx = evolution_lifecycle.begin_evolution_transaction("evo", cycle=1, campaign=campaign)
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], "evo", head,
    )["ok"] is True
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=tmp_path,
        current_task_type="evolution",
        task_id="evo",
        task_metadata={"evolution_transaction": tx},
    )
    ctx.last_reviewed_commit_sha = head

    result = _request_restart(ctx, "   ")

    assert "Restart requested" in result
    marker = json.loads((tmp_path / "state" / "pending_restart_verify.json").read_text())
    assert marker["reason"] == "agent_requested_restart"
    assert ctx.pending_restart_is_evolution is True
    (repo / "file.txt").write_text("dirty\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path, current_task_type="evolution")

    result = _request_restart(ctx, "dirty")

    assert "RESTART_BLOCKED" in result
    assert "exact local commit receipt" in result


def test_toggle_evolution_tool_accepts_objective(tmp_path, monkeypatch):
    from ouroboros.tools.control import _toggle_evolution

    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    pending = []
    result = _toggle_evolution(SimpleNamespace(pending_events=pending), True, objective="Improve cron")

    assert "ON" in result
    assert pending[0]["objective"] == "Improve cron"


def test_toggle_evolution_tool_refuses_in_light_mode(monkeypatch):
    from ouroboros.tools.control import _toggle_evolution

    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "light")
    pending = []
    result = _toggle_evolution(SimpleNamespace(pending_events=pending), True)

    # The tool's own result reflects the block; no event is queued.
    assert "light" in result.lower()
    assert pending == []


def test_memory_provenance_records_old_and_new_content(tmp_path):
    from ouroboros.tools.control import _update_identity
    from ouroboros.tools.knowledge import _knowledge_write
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    _knowledge_write(ctx, "facts", "old", mode="overwrite")
    _knowledge_write(ctx, "facts", "new", mode="overwrite")
    history = [
        json.loads(line)
        for line in (tmp_path / "memory" / "knowledge_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert history[-1]["old_content"] == "old"
    assert history[-1]["new_content"] == "new"

    _update_identity(ctx, "I am v1 with enough detail to satisfy the identity update length gate.")
    _update_identity(ctx, "I am v2 with enough detail to satisfy the identity update length gate.")
    identity_history = [
        json.loads(line)
        for line in (tmp_path / "memory" / "identity_journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert identity_history[-1]["old_content"].startswith("I am v1")
    assert identity_history[-1]["new_content"].startswith("I am v2")


def test_evolution_block_reason_depends_on_runtime_mode(monkeypatch):
    from supervisor import queue

    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "light")
    blocked = queue.evolution_block_reason()
    assert blocked
    assert "advanced" in blocked and "pro" in blocked

    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    assert queue.evolution_block_reason() == ""
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "pro")
    assert queue.evolution_block_reason() == ""


def test_enqueue_evolution_blocked_in_light_mode(tmp_path, monkeypatch):
    from supervisor import queue
    from supervisor import state as supervisor_state

    supervisor_state.init(tmp_path)
    sent = []
    monkeypatch.setattr(queue, "send_with_budget", lambda chat_id, text, *a, **k: sent.append(text))
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "light")
    queue.init(tmp_path)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    queue.start_evolution_campaign("Improve", source="test")
    st = supervisor_state.load_state()
    st["evolution_mode_enabled"] = True
    st["owner_chat_id"] = 1
    supervisor_state.save_state(st)

    queue.enqueue_evolution_task_if_needed()

    assert pending == []
    assert supervisor_state.load_state()["evolution_mode_enabled"] is False
    assert queue.get_evolution_status_snapshot()["campaign"]["status"] == "paused"
    assert any("light" in m.lower() for m in sent)


def test_enqueue_evolution_omits_duplicate_cycle_message(tmp_path, monkeypatch):
    from supervisor import queue
    from supervisor import state as supervisor_state

    supervisor_state.init(tmp_path)
    monkeypatch.setattr(supervisor_state, "TOTAL_BUDGET_LIMIT", 100.0)
    sent = []
    monkeypatch.setattr(queue, "send_with_budget", lambda chat_id, text, *a, **k: sent.append(text))
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    queue.init(tmp_path)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    queue.start_evolution_campaign("Improve", source="test")
    st = supervisor_state.load_state()
    st["evolution_mode_enabled"] = True
    st["owner_chat_id"] = 1
    st["spent_usd"] = 0.0
    supervisor_state.save_state(st)

    queue.enqueue_evolution_task_if_needed()

    assert len(pending) == 1
    assert pending[0]["type"] == "evolution"
    # The generic "... task started." lifecycle message is the only start bubble;
    # the redundant "Evolution #N: <id>" enqueue bubble is gone.
    assert not any("Evolution #" in m for m in sent)


def test_marketplace_helper_resyncs_skill_schedules(tmp_path, monkeypatch):
    from ouroboros.gateway import marketplace

    calls = []
    monkeypatch.setattr("supervisor.queue.resync_skill_schedules", lambda dr: calls.append(dr))

    marketplace._resync_skill_schedules_quiet(tmp_path)

    assert calls == [tmp_path]


def test_skill_schedule_sync_removes_vanished_skill_schedule(tmp_path):
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from supervisor import queue

    queue.init(tmp_path)
    manifest = parse_skill_manifest_text("""---
name: cron-demo
description: Cron demo
version: 0.1.0
type: extension
entry: plugin.py
permissions: [supervised_task]
scheduled_tasks:
  - name: refresh
    cron: "0 * * * *"
---
body
""")
    skill = SimpleNamespace(
        name="cron-demo", manifest=manifest, enabled=True, load_error="",
        content_hash="abc",
        review=SimpleNamespace(status="pass", is_stale_for=lambda _h: False),
    )

    queue.sync_skill_schedules([skill])
    assert any(t.get("source") == "skill_manifest" for t in queue.list_scheduled_tasks()["tasks"])

    # Skill (or its scheduled_task) vanished → the record is removed entirely,
    # not left as a disabled tombstone.
    queue.sync_skill_schedules([])
    assert queue.list_scheduled_tasks()["tasks"] == []


def test_skill_schedule_sync_preserves_ambiguous_identity_rows(tmp_path):
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from supervisor import queue

    queue.init(tmp_path)
    manifest = parse_skill_manifest_text("""---
name: cron-demo
description: Cron demo
version: 0.1.0
type: extension
entry: plugin.py
permissions: [supervised_task]
scheduled_tasks:
  - name: refresh
    cron: "0 * * * *"
---
body
""")
    skill = SimpleNamespace(
        name="cron-demo", manifest=manifest, enabled=True, load_error="",
        content_hash="abc", identity_collision=False,
        review=SimpleNamespace(status="pass", is_stale_for=lambda _h: False),
    )
    queue.sync_skill_schedules([skill])
    before = queue.list_scheduled_tasks()["tasks"]

    collision = SimpleNamespace(
        name="cron-demo", identity_collision=True,
        manifest=SimpleNamespace(scheduled_tasks=[]),
    )
    unique = SimpleNamespace(
        name="other", manifest=manifest, enabled=True, load_error="",
        content_hash="def", identity_collision=False,
        review=SimpleNamespace(status="pass", is_stale_for=lambda _h: False),
    )
    report = queue.sync_skill_schedules([collision, unique])
    after = queue.list_scheduled_tasks()["tasks"]

    assert report["changed"] is True
    assert next(item for item in after if item["id"] == "skill-cron-demo-refresh") == before[0]
    assert any(item["id"] == "skill-other-refresh" for item in after)


def test_due_skill_schedule_does_not_run_while_identity_is_ambiguous(
    tmp_path, monkeypatch,
):
    from supervisor import queue, state

    state.init(tmp_path)
    queue.init(tmp_path)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    manifest = """---
name: cron-demo
description: Cron demo
version: 0.1.0
type: extension
entry: plugin.py
permissions: [supervised_task]
scheduled_tasks:
  - name: refresh
    cron: "* * * * *"
---
body
"""
    data_skill = tmp_path / "skills" / "external" / "cron-demo"
    user_skill = tmp_path / "user-skills" / "cron-demo"
    for skill_dir in (data_skill, user_skill):
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
        (skill_dir / "plugin.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(user_skill.parent))
    queue.upsert_scheduled_task({
        "id": "skill-cron-demo-refresh",
        "name": "cron-demo/refresh",
        "enabled": True,
        "source": "skill_manifest",
        "skill": "cron-demo",
        "trigger": {"type": "cron", "expr": "* * * * *"},
        "next_run_at": "2000-01-01T00:00:00+00:00",
        "task": {"type": "task", "text": "run cron-demo"},
    })

    queue.resync_skill_schedules(tmp_path)
    before = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    from supervisor import queue_schedules
    monkeypatch.setattr(queue_schedules, "_last_skill_schedule_sync", queue.time.monotonic())
    queue.check_scheduled_tasks()

    assert pending == []
    assert queue.list_scheduled_tasks(tmp_path)["tasks"] == [before]
    assert not (tmp_path / "task_results").exists()


def test_reflection_extract_trailing_json_parses_memory_and_backlog():
    from ouroboros.reflection import _extract_trailing_json

    text = (
        "Reflection body here.\n"
        'MEMORY_ACTIONS_JSON: [{"type": "scratchpad_append", "content": "note"}]\n'
        'BACKLOG_CANDIDATES_JSON: [{"summary": "s", "evidence": "e"}]'
    )
    body_after_backlog, backlog = _extract_trailing_json(text, "BACKLOG_CANDIDATES_JSON:")
    reflection_text, memory = _extract_trailing_json(body_after_backlog, "MEMORY_ACTIONS_JSON:")

    assert backlog == [{"summary": "s", "evidence": "e"}]
    assert memory == [{"type": "scratchpad_append", "content": "note"}]
    assert reflection_text.strip() == "Reflection body here."


def test_reflection_extract_trailing_json_is_order_independent():
    from ouroboros.reflection import _extract_trailing_json

    # Markers emitted in the reverse of the documented order must still both parse
    # (no silent memory-action loss).
    text = (
        "Reflection body.\n"
        'BACKLOG_CANDIDATES_JSON: [{"summary": "s", "evidence": "e"}]\n'
        'MEMORY_ACTIONS_JSON: [{"type": "knowledge_write", "content": "c", "topic": "t"}]'
    )
    after_backlog, backlog = _extract_trailing_json(text, "BACKLOG_CANDIDATES_JSON:")
    reflection_text, memory = _extract_trailing_json(after_backlog, "MEMORY_ACTIONS_JSON:")

    assert backlog == [{"summary": "s", "evidence": "e"}]
    assert memory == [{"type": "knowledge_write", "content": "c", "topic": "t"}]
    assert reflection_text.strip() == "Reflection body."


def test_reflection_validate_memory_actions_filters_types():
    from ouroboros.reflection import _validate_memory_actions

    raw = [
        {"type": "scratchpad_append", "content": "note"},
        {"type": "knowledge_write", "content": "fact", "topic": "facts"},
        {"type": "knowledge_write", "content": "no topic"},
        {"type": "identity_update_candidate", "content": "refine"},
        {"type": "delete_everything", "content": "nope"},
        {"type": "scratchpad_append", "content": "   "},
    ]
    actions = _validate_memory_actions(raw, "task1")

    assert [a["type"] for a in actions] == [
        "scratchpad_append", "knowledge_write", "identity_update_candidate",
    ]
    assert actions[1]["topic"] == "facts"
    assert all(a["task_id"] == "task1" for a in actions)


def test_apply_memory_actions_writes_to_parent_drive(tmp_path):
    from ouroboros.reflection import apply_memory_actions

    env = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path)
    actions = [
        {"type": "scratchpad_append", "content": "durable note", "task_id": "t1"},
        {"type": "knowledge_write", "content": "reusable fact", "topic": "review_process", "task_id": "t1"},
        {"type": "identity_update_candidate", "content": "I value rigor", "task_id": "t1"},
    ]

    assert apply_memory_actions(env, actions) == 3

    knowledge = (tmp_path / "memory" / "knowledge" / "review_process.md").read_text(encoding="utf-8")
    assert "reusable fact" in knowledge
    assert (tmp_path / "memory" / "knowledge_history.jsonl").exists()

    scratchpad = (tmp_path / "memory" / "scratchpad.md").read_text(encoding="utf-8")
    assert "durable note" in scratchpad
    assert "IDENTITY UPDATE CANDIDATE" in scratchpad
    # Identity is never auto-written; the candidate stays in the scratchpad only.
    identity_path = tmp_path / "memory" / "identity.md"
    assert not identity_path.exists() or "I value rigor" not in identity_path.read_text(encoding="utf-8")


def test_runtime_context_includes_schedule_digest(tmp_path):
    from ouroboros.context import build_runtime_section

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "scheduled_tasks.json").write_text(json.dumps({
        "tasks": [{
            "id": "hourly", "name": "Hourly", "enabled": True,
            "trigger": {"type": "cron", "expr": "0 * * * *"},
            "timezone": "Europe/Moscow", "next_run_at": "2030-01-01T00:00:00+00:00",
        }]
    }), encoding="utf-8")
    env = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path,
        drive_path=lambda rel: tmp_path / rel,
    )

    section = build_runtime_section(env, {"id": "t1", "type": "task"})

    assert "scheduled_tasks" in section
    assert "hourly" in section
    assert "0 * * * *" in section


def test_cli_evolve_start_refused_in_light_mode(monkeypatch):
    from ouroboros import cli

    class FakeClient:
        def __init__(self):
            self.posts = []

        def request(self, method, path, body=None):
            if method == "GET" and path == "/api/state":
                return {"runtime_mode": "light"}
            self.posts.append((method, path, body))
            return {"status": "ok"}

    fake = FakeClient()
    monkeypatch.setattr(cli, "_client", lambda args: fake)
    monkeypatch.setattr(cli, "_print_json", lambda *a, **k: None)

    rc = cli._evolve_command(SimpleNamespace(evolve_command="start", objective=[]))

    assert rc == 1
    # Never POSTed the /evolve on command in light mode.
    assert fake.posts == []


def test_assign_tasks_cancels_pending_evolution_in_light_mode(tmp_path, monkeypatch):
    import supervisor.workers as workers
    import supervisor.queue as queue
    from supervisor import state as supervisor_state
    from ouroboros.task_results import load_task_result

    supervisor_state.init(tmp_path)
    monkeypatch.setattr(supervisor_state, "TOTAL_BUDGET_LIMIT", 100.0)
    st = supervisor_state.load_state()
    st["spent_usd"] = 0.0
    supervisor_state.save_state(st)

    orig_drive = workers.DRIVE_ROOT
    orig_q_drive = queue.DRIVE_ROOT
    workers.DRIVE_ROOT = tmp_path
    queue.DRIVE_ROOT = tmp_path
    workers.WORKERS.clear()
    workers.RUNNING.clear()
    workers.PENDING[:] = [{"id": "evo1", "type": "evolution", "text": "x"}]
    queue.PENDING = workers.PENDING
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "light blocked")
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)

    try:
        workers.assign_tasks()
    finally:
        workers.DRIVE_ROOT = orig_drive
        queue.DRIVE_ROOT = orig_q_drive
        workers.PENDING[:] = []

    assert all(t.get("type") != "evolution" for t in workers.PENDING)
    assert load_task_result(tmp_path, "evo1")["status"] == "cancelled"


def test_toggle_evolution_tool_blocked_in_light_mode(monkeypatch):
    from supervisor import events

    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "light")
    state = {"owner_chat_id": 7}
    sent = []
    ctx = SimpleNamespace(
        load_state=lambda: state,
        save_state=lambda s: state.update(s),
        send_with_budget=lambda chat_id, text, *a, **k: sent.append((chat_id, text)),
        PENDING=[],
        sort_pending=lambda: None,
        persist_queue_snapshot=lambda **k: None,
    )

    events._handle_toggle_evolution({"enabled": True, "objective": "x"}, ctx)

    assert state.get("evolution_mode_enabled") is not True
    assert sent and "light" in sent[0][1].lower()
