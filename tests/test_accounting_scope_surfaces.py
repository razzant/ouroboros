from __future__ import annotations

import asyncio
from unittest.mock import patch

from ouroboros import project_naming
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope


class _ScopedNamingClient:
    def __init__(self, captured):
        self.captured = captured

    def chat(self, **_kwargs):
        self.captured.append(current_usage_scope())
        return {"content": "Scoped title"}, {"cost": 0.25}


def test_project_namer_reconstructs_persisted_budget_scope_without_double_increment(
    tmp_path,
    monkeypatch,
):
    budget_root = tmp_path / "budget-root"
    budget_root.mkdir()
    write_task_result(
        tmp_path,
        "task-1",
        STATUS_RUNNING,
        root_task_id="root-1",
        parent_task_id="parent-1",
        budget_drive_root=str(budget_root),
    )
    monkeypatch.setenv("TOTAL_BUDGET", "9")
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "3")
    monkeypatch.setattr(project_naming, "_light_naming_model", lambda: "openai/test-light")
    captured = []

    with patch("supervisor.state.update_budget_from_usage") as legacy_increment:
        title = project_naming.llm_project_name(
            "Build the thing",
            use_local=True,
            llm_client=_ScopedNamingClient(captured),
            drive_root=tmp_path,
            task_id="task-1",
        )

    assert title == "Scoped title"
    legacy_increment.assert_not_called()
    scope = captured[0]
    assert scope.drive_root == str(budget_root)
    assert scope.task_id == "task-1"
    assert scope.root_task_id == "root-1"
    assert scope.parent_task_id == "parent-1"
    assert scope.category == "project_naming"
    assert scope.source == "project_naming"
    assert scope.global_limit_usd == 9
    assert scope.root_limit_usd == 3


def test_async_project_namer_preserves_active_tree_scope_across_to_thread(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(project_naming, "_light_naming_model", lambda: "openai/test-light")
    captured = []
    active = UsageScope(
        drive_root=tmp_path,
        task_id="active-task",
        root_task_id="active-root",
        parent_task_id="active-parent",
        category="task",
        source="agent.task",
        global_limit_usd=0.0,
        root_limit_usd=0.0,
    )

    with patch("supervisor.state.update_budget_from_usage") as legacy_increment:
        with usage_scope(active):
            title = asyncio.run(project_naming.llm_project_name_async(
                "Build the async thing",
                timeout_sec=2,
                use_local=True,
                llm_client=_ScopedNamingClient(captured),
                drive_root=tmp_path,
                task_id="ignored-because-active",
            ))

    assert title == "Scoped title"
    legacy_increment.assert_not_called()
    scope = captured[0]
    assert scope.task_id == "active-task"
    assert scope.root_task_id == "active-root"
    assert scope.parent_task_id == "active-parent"
    assert scope.category == "project_naming"
    assert scope.source == "project_naming"
    # Explicit zero rails must survive the category/source replacement.
    assert scope.global_limit_usd == 0.0
    assert scope.root_limit_usd == 0.0


def test_project_naming_scope_accepts_unlimited_total_budget(tmp_path, monkeypatch):
    from ouroboros.settings_setup_contract import resolve_total_budget_usd

    monkeypatch.setenv("TOTAL_BUDGET", "0")
    assert resolve_total_budget_usd() is None

    scope = project_naming._project_naming_usage_scope(tmp_path, "naming-task")

    assert scope.global_limit_usd is None
