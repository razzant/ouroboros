from __future__ import annotations

import asyncio
import json
from unittest.mock import Mock, patch

import pytest

from ouroboros import project_naming
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope


class _ScopedNamingClient:
    def __init__(self, captured):
        self.captured = captured

    def chat(self, **_kwargs):
        self.captured.append(current_usage_scope())
        return {"content": "Scoped title"}, {"cost": 0.25}


_MAIN = "openai@absent::main"
_CLOUD = "openai@fixture::cloud"
_LOCAL = "spare (local)"


@pytest.mark.parametrize("light, local_light, local_main, fallback, explicit, expected", [
    ("", False, False, _LOCAL, None, (_LOCAL, True)),
    ("", True, False, _CLOUD, None, (_MAIN, True)),
    (_CLOUD, False, False, _LOCAL, None, (_CLOUD, False)),
    ("", False, True, "", None, (_MAIN, True)),
    (_CLOUD, False, False, "", True, (_CLOUD, True)),
    # An explicit cloud override of a local Light binding sends the slot's credentialed cloud model
    # (light -> fallback -> main): the unresolved local id never reaches a cloud send.
    ("", True, False, _CLOUD, False, (_CLOUD, False)),
    ("openai@absent::light", True, False, _CLOUD, False, (_CLOUD, False)),
    (_LOCAL, False, True, _CLOUD, False, (_CLOUD, False)),
    # An explicit local override keeps the route's own model as its label: a local call never sends it.
    (_CLOUD, True, False, "", True, (_CLOUD, True)),
    ("openai@absent::light", False, False, _CLOUD, True, ("openai@absent::light", True)),
])
def test_project_naming_sends_and_records_the_resolved_light_route(
    tmp_path, monkeypatch, light, local_light, local_main, fallback, explicit, expected,
):
    from ouroboros import config, llm_observability, model_concurrency

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "OUROBOROS_MODEL": _MAIN, "OUROBOROS_MODEL_LIGHT": light, "USE_LOCAL_LIGHT": local_light,
        "USE_LOCAL_MAIN": local_main, "OUROBOROS_MODEL_FALLBACKS": fallback,
        "OUROBOROS_CONNECTIONS": [{"kind": "openai", "name": "fixture", "api_key": "fixture"}],
    }), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    scopes = []
    client = Mock(wraps=_ScopedNamingClient(scopes))
    with patch.object(model_concurrency, "model_call_scope", wraps=model_concurrency.model_call_scope) as slot, \
         patch.object(llm_observability, "persist_observed_call") as observed:
        title = project_naming.llm_project_name("Owner request", use_local=explicit, llm_client=client,
                                                drive_root=tmp_path, task_id="title")
    assert title == "Scoped title"
    assert slot.call_args.args == expected
    sent = client.chat.call_args.kwargs
    manifest = observed.call_args.kwargs["manifest"]
    assert (sent["model"], sent["use_local"]) == (manifest["model"], manifest["use_local"]) == expected
    assert sent["model_role"] == "light" and scopes[0].category == "project_naming"


def test_explicit_remote_naming_with_only_local_models_keeps_the_heuristic(tmp_path, monkeypatch):
    from ouroboros import config

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "OUROBOROS_MODEL": _LOCAL, "OUROBOROS_MODEL_LIGHT": _LOCAL,
        "USE_LOCAL_MAIN": True, "OUROBOROS_MODEL_FALLBACKS": "",
    }), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", settings)
    client = Mock()
    assert project_naming.llm_project_name("Owner request", use_local=False, llm_client=client) == "Owner request"
    client.chat.assert_not_called()


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
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "openai/test-light")
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
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "openai/test-light")
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
