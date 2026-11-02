"""Composition regressions for nested coordination over the PR #316 target."""

from __future__ import annotations

import types

import pytest

pytestmark = pytest.mark.serial


def test_source_created_promote_announces_once_and_persists_admitted_contract(
    monkeypatch, tmp_path,
):
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import load_task_result
    from supervisor.events import _handle_promote_chat_to_task

    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "projects"))
    project_id = "source-created-project"
    assert create_project(tmp_path, project_id, name="Source Created")["created"] is True
    pending = []
    announcements = []

    def enqueue(task):
        admitted = dict(task)
        admitted["task_contract"] = {
            **task["task_contract"],
            "source": "queue_admitted_composition_test",
        }
        pending.append(admitted)
        return admitted

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(
        "supervisor.terminal_delivery.enqueue_terminal_delivery",
        lambda _root, event, **_kwargs: announcements.append(dict(event)) or True,
    )
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        WORKERS={0: types.SimpleNamespace()},
        PENDING=pending,
        bridge=None,
        enqueue_task=enqueue,
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
        append_jsonl=lambda *_args, **_kwargs: None,
    )

    outcome = _handle_promote_chat_to_task(
        {
            "type": "promote_chat_to_task",
            "task_id": "source-created-root",
            "routing_token": "source-created-token",
            "objective": "Inspect the prepared source",
            "project_id": project_id,
            "project_name": "Source Created",
            "source": "https://github.com/example/source-created.git",
            "_source_prepared": True,
            "_source_created": True,
            "workspace": "none",
            "chat_id": 1,
        },
        ctx,
    )

    assert outcome["status"] == "scheduled"
    assert "_admitted_task_contract" not in outcome
    assert [row["system_type"] for row in announcements] == ["project_started"]
    stored = load_task_result(tmp_path, "source-created-root")
    assert stored["root_task_id"] == "source-created-root"
    assert stored["delegation_role"] == "root"
    assert stored["task_contract"] == pending[0]["task_contract"]
    assert stored["task_contract"]["source"] == "queue_admitted_composition_test"


def test_swarm_host_child_zero_run_is_fanout_and_depth_stays_host_visible(tmp_path):
    from ouroboros.depth_evidence import build_depth_summary
    from ouroboros.outcomes import append_verification_receipt, read_verification_receipts
    from ouroboros.task_finalization import build_swarm_efficiency
    from ouroboros.tools.control import _emit_swarm_fanout

    logs = tmp_path / "logs"
    logs.mkdir()
    ctx = types.SimpleNamespace(drive_logs=lambda: logs, _last_wave_ts=0.0)
    _emit_swarm_fanout(
        ctx,
        parent_task_id="root",
        root_task_id="root",
        depth=1,
        task_group_id="root-wave",
        task_ids=["child-zero-run"],
        role="researcher",
        requested_model_lane="auto",
        objective="Inspect independently",
        emitted_live=True,
    )
    assert append_verification_receipt(tmp_path, "child-zero-run", {
        "status": "declared",
        "contract_kind": "delegation_zero_run",
        "zero_run": True,
        "zero_run_decision": "complete",
        "zero_run_basis": "The host child completed without a physical leaf.",
        "physical_run_started": False,
    })

    rollup = build_swarm_efficiency(
        types.SimpleNamespace(drive_root=tmp_path),
        {"id": "root", "metadata": {"subagents_mode": "swarm"}},
    )
    assert rollup["subagent_count"] == 1
    assert rollup["intent_source"] == "swarm"
    assert rollup.get("status") != "no_fanout_observed"
    [receipt] = read_verification_receipts(tmp_path, "child-zero-run")
    assert receipt["contract_kind"] == "delegation_zero_run"

    root_contract = {"delegation_budget": {
        "depth_remaining": 3,
        "depth_provenance": {
            "requested_depth": 3,
            "permitted_depth": 3,
            "attempted_depth": 0,
            "achieved_depth": 0,
        },
    }}
    child_status = {"depth_provenance": {
        "requested_depth": 3,
        "permitted_depth": 3,
        "attempted_depth": 1,
        "achieved_depth": 1,
    }}
    summary = build_depth_summary(root_contract, [child_status])
    assert summary["achieved_depth"] == 1
    assert summary["host_visible_only"] is True
