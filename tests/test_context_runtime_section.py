"""The runtime section and the user content the context builder emits.

Split verbatim out of ``tests/test_context.py`` by theme. This module owns the
force-plan notice that must not rewrite the user's text, the ephemeral force plan that
only routes, the light-mode rule and filesystem affordances the runtime section states,
the workspace rules that preserve system review/commit authority, the host routing
manifest and manual contract, the improvement backlog digest, and the runtime_env
block.
"""

from __future__ import annotations

import inspect
import json

import pytest

from ouroboros.context import build_runtime_section, build_user_content

from tests._context_shared import _make_health_env

def test_build_llm_messages_has_no_recorder_only_soft_cap_chain():
    from ouroboros import context as context_module
    from ouroboros.context import build_llm_messages

    assert "soft_cap_tokens" not in inspect.signature(build_llm_messages).parameters
    assert not hasattr(context_module, "apply_message_token_soft_cap")
    source = inspect.getsource(build_llm_messages)
    assert "estimated_tokens_before" not in source
    assert "trimmed_sections" not in source
    assert "context_fit" in source


@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_force_plan_metadata_adds_structured_notice_without_rewriting_user_text(
    monkeypatch, enforcement,
):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    content = build_user_content(
        {
            "text": "Fix the marketplace retry flow.",
            "metadata": {"force_plan": True, "force_plan_source": "swarm"},
        }
    )

    assert content.startswith("[SWARM_INITIATIVE]")
    assert "Source: swarm." in content
    assert f"Resolved review enforcement: {enforcement}." in content
    assert "Under blocking" in content
    assert "non-mutating preparation" in content
    assert "begin implementation only after review closes" in content
    # Fan-out integration mechanics (owner-approved, 2026-08-05): parallel
    # children cannot see each other's edits, so a plan gives them disjoint
    # write regions or plans the parent synthesis for the expected overlap.
    assert "cannot see each other's edits" in content
    assert "disjoint write regions" in content
    assert content.rstrip().endswith("Fix the marketplace retry flow.")


def test_ephemeral_force_plan_is_routing_only_and_transfers_work():
    content = build_user_content({
        "text": "Fix the marketplace retry flow.",
        "_ephemeral_turn": True,
        "metadata": {"force_plan": True, "force_plan_source": "swarm"},
    })

    assert content.startswith("[SWARM_ROUTING_INTENT]")
    assert "exactly one NEW managed root" in content
    assert "do not execute it" in content
    assert content.rstrip().endswith("Fix the marketplace retry flow.")


def test_runtime_section_includes_light_runtime_mode_rule(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "light")
    section = build_runtime_section(env, {"id": "task-1", "type": "task"})
    payload = json.loads(section.split("\n\n", 1)[1])

    assert payload["runtime_mode"] == "light"
    assert "forbids Ouroboros repo mutation" in payload["runtime_mode_rule"]
    assert "user_files" in payload["runtime_mode_rule"]
    assert "artifact_store" in payload["runtime_mode_rule"]
    assert "explicit scoped skill-payload work/repair" in payload["runtime_mode_rule"]
    assert "runtime_data/uploads" in payload["runtime_mode_rule"]


def test_runtime_section_includes_filesystem_affordances_with_ctx(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext

    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "light")
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path)

    section = build_runtime_section(env, {"id": "task-1", "type": "task"}, ctx=ctx)
    payload = json.loads(section.split("\n\n", 1)[1])
    fs = payload["capabilities"]["filesystem"]

    assert fs["profile"] == "self_modification"
    assert "runtime_data" in fs["searchable_roots"]
    assert "task_drive" not in fs["searchable_roots"]
    assert "task_drive" in fs["allowed_shell_cwd_roots"]
    assert "status" in fs["git_readonly_subcommands"]
    assert "active_workspace" in fs["light_gated_roots"]


def test_runtime_section_external_workspace_includes_user_files_shell_affordance(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext

    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    drive = tmp_path / "data"
    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    drive.mkdir()
    repo.mkdir(exist_ok=True)
    workspace.mkdir(exist_ok=True)
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=drive,
        workspace_root=workspace,
        workspace_mode="external",
    )

    section = build_runtime_section(env, {"id": "task-1", "type": "task"}, ctx=ctx)
    payload = json.loads(section.split("\n\n", 1)[1])
    fs = payload["capabilities"]["filesystem"]

    assert fs["profile"] == "external_workspace_task"
    assert "user_files" in fs["allowed_shell_cwd_roots"]


def test_runtime_section_workspace_rule_preserves_system_review_commit_authority(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    section = build_runtime_section(
        env,
        {
            "id": "task-1",
            "type": "task",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "forked",
        },
    )
    rule = json.loads(section.split("\n\n", 1)[1])["active_workspace"]["rule"]

    assert "default to the active workspace" in rule
    assert "explicit typed root/cwd" in rule
    assert "self-review/commit tools remain available" in rule
    assert "self-review/commit tools are unavailable" not in rule


def test_runtime_section_omits_light_rule_for_advanced(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    section = build_runtime_section(env, {"id": "task-1", "type": "task"})
    payload = json.loads(section.split("\n\n", 1)[1])

    assert payload["runtime_mode"] == "advanced"
    assert "runtime_mode_rule" not in payload


def test_runtime_section_includes_non_workspace_memory_boundary(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    section = build_runtime_section(
        env,
        {
            "id": "task-1",
            "type": "task",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "child"),
            "child_drive_root": str(tmp_path / "child"),
            "budget_drive_root": str(tmp_path / "data"),
        },
    )
    payload = json.loads(section.split("\n\n", 1)[1])
    assert payload["task"]["memory_mode"] == "forked"
    assert payload["task"]["child_drive_root"].endswith("child")
    assert payload["task"]["budget_drive_root"].endswith("data")


def test_runtime_section_exposes_host_routing_manifest_and_manual_contract(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    task = {
        "id": "decision-1",
        "type": "task",
        "metadata": {
            "current_chat": {
                "chat_id": 1,
                "running_tasks": [],
                "addressable_root_tasks": [{"task_id": "pending-1", "status": "pending"}],
            },
            "main_routing_manifest": {
                "projects": [{"project_id": "racer", "name": "Racer"}],
                "root_tasks": [{"task_id": "pending-1", "status": "pending"}],
            },
            "routing_contract": {
                "source_lane": "main",
                "on_uncertain_or_invalid_target": "needs_manual_target",
                "manual_options": [{"task_id": "pending-1"}],
            },
        },
    }

    payload = json.loads(build_runtime_section(env, task).split("\n\n", 1)[1])

    assert payload["current_chat"]["addressable_root_tasks"][0]["task_id"] == "pending-1"
    assert payload["main_routing_manifest"]["projects"][0]["project_id"] == "racer"
    assert payload["routing_contract"]["on_uncertain_or_invalid_target"] == "needs_manual_target"


def test_runtime_section_includes_improvement_backlog_digest(tmp_path):
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory

    class FakeEnv:
        def drive_path(self, p):
            return tmp_path / p

        def repo_path(self, p):
            return tmp_path / "repo" / p

        @property
        def repo_dir(self):
            return tmp_path / "repo"

        @property
        def drive_root(self):
            return tmp_path

    (tmp_path / "repo" / "prompts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    (tmp_path / "repo" / "prompts" / "SYSTEM.md").write_text("System prompt", encoding="utf-8")
    (tmp_path / "repo" / "BIBLE.md").write_text("Bible", encoding="utf-8")
    (tmp_path / "repo" / "README.md").write_text("README", encoding="utf-8")
    (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text('# Ouroboros v1.2.3', encoding="utf-8")
    (tmp_path / "repo" / "docs" / "DEVELOPMENT.md").write_text('# Dev', encoding="utf-8")
    (tmp_path / "repo" / "docs" / "CHECKLISTS.md").write_text('Checklist', encoding="utf-8")
    (tmp_path / "repo" / "VERSION").write_text("1.2.3", encoding="utf-8")
    (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
    (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
    (tmp_path / "memory" / "identity.md").write_text("I am Ouroboros", encoding="utf-8")
    (tmp_path / "memory" / "scratchpad.md").write_text("scratchpad", encoding="utf-8")
    (tmp_path / "memory" / "knowledge" / "improvement-backlog.md").write_text(
        "# Improvement Backlog\n\n### ibl-1\n- status: open\n- created_at: 2026-04-14T09:00:00+00:00\n- source: execution_reflection\n- category: process\n- task_id: task-1\n- requires_plan_review: yes\n- fingerprint: fp-1\n- summary: Reduce recurring task friction around REVIEW_BLOCKED\n",
        encoding="utf-8",
    )

    messages, _ = build_llm_messages(
        env=FakeEnv(),
        memory=Memory(drive_root=tmp_path),
        task={"id": "task-a", "type": "task", "text": "hello"},
    )
    dynamic_text = messages[0]["content"][2]["text"]
    assert "## Improvement Backlog" in dynamic_text
    assert "Reduce recurring task friction around REVIEW_BLOCKED" in dynamic_text


class TestRuntimeEnvSection:
    """build_runtime_section: runtime_env carries presentation + platform, and
    the per-message owner_client fact renders beside it (is_desktop retired)."""

    def _make_env(self, tmp_path):
        class FakeEnv:
            repo_dir = tmp_path / "repo"
            drive_root = tmp_path

            def drive_path(self, p):
                return tmp_path / p

        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        (tmp_path / "state" / "state.json").write_text(
            '{"spent_usd": 0}', encoding="utf-8"
        )
        return FakeEnv()

    def test_runtime_env_presentation_absent_means_web(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        monkeypatch.delenv("OUROBOROS_PRESENTATION", raising=False)
        env = self._make_env(tmp_path)
        section = build_runtime_section(env, {"id": "t1", "type": "task"})
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert "runtime_env" in data
        assert "platform" in data["runtime_env"]
        assert isinstance(data["runtime_env"]["platform"], str)
        assert data["runtime_env"]["presentation"] == "web"
        # The dead is_desktop flag is retired; presentation replaced it.
        assert "is_desktop" not in data["runtime_env"]

    def test_runtime_env_presentation_from_launcher_export(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        for value in ("desktop_window", "browser_fallback"):
            monkeypatch.setenv("OUROBOROS_PRESENTATION", value)
            env = self._make_env(tmp_path)
            section = build_runtime_section(env, {"id": "t2", "type": "task"})
            data = json.loads(section.split("## Runtime context\n\n", 1)[1])
            assert data["runtime_env"]["presentation"] == value

    def test_owner_client_rendered_from_metadata(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        monkeypatch.delenv("OUROBOROS_PRESENTATION", raising=False)
        env = self._make_env(tmp_path)
        fact = {"pywebview": True, "ua": "TestShell/1.0", "viewport": {"w": 1200, "h": 800}}
        section = build_runtime_section(
            env, {"id": "t3", "type": "task", "metadata": {"client_surface": fact}}
        )
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert data["owner_client"] == fact
        assert "SENT" in data["owner_client_note"]

    def test_owner_client_absent_is_a_gap_not_a_default(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        env = self._make_env(tmp_path)
        section = build_runtime_section(env, {"id": "t4", "type": "task"})
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert "owner_client" not in data
        assert "owner_client_note" not in data

    def test_owner_client_channel_fact_stamped_by_external_admission(self, tmp_path):
        from ouroboros.context import build_runtime_section

        env = self._make_env(tmp_path)
        # /api/tasks and CLI STAMP the channel fact at admission; the renderer
        # reads only the producer-assembled fact.
        section = build_runtime_section(
            env, {"id": "t5", "type": "task", "metadata": {"client_surface": {"channel": "cli"}}}
        )
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert data["owner_client"] == {"channel": "cli"}

    def test_owner_client_never_inferred_from_metadata_source(self, tmp_path):
        from ouroboros.context import build_runtime_section

        env = self._make_env(tmp_path)
        # metadata.source is OVERLOADED (scheduler writes scheduled_task /
        # skill_scheduled_task): the renderer must never dress it up as an
        # owner surface — no producer stamp, no fact (codex scope round 2 N1).
        for source in ("cli", "scheduled_task", "skill_scheduled_task", "web"):
            section = build_runtime_section(
                env, {"id": "t6", "type": "task", "metadata": {"source": source}}
            )
            data = json.loads(section.split("## Runtime context\n\n", 1)[1])
            assert "owner_client" not in data, f"source={source!r} must not render"
        # Internal producers use top-level task["source"], never rendered.
        section = build_runtime_section(
            env, {"id": "t7", "type": "task", "source": "promote_chat_to_task"}
        )
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert "owner_client" not in data


def _delegation_data_root(tmp_path, monkeypatch):
    root = tmp_path / "delegation_data_root"
    (root / "state").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    return root


def _delegation_fact(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    section = build_runtime_section(env, {"id": "task-1", "type": "task"})
    payload = json.loads(section.split("\n\n", 1)[1])
    return payload["capabilities"]


def test_delegation_fact_carries_configured_route_and_historical_rows(tmp_path, monkeypatch):
    root = _delegation_data_root(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "claudexor=opus-5:high")
    (root / "state" / "reviewer_slot_last_execution.json").write_text(json.dumps({
        "triad_1": {
            "ts": "2026-08-18T01:02:03+00:00",
            "surface": "triad",
            "status": "ok",
            "effective": {"route": "agent_session:claudexor", "model": "opus-5"},
        },
        "triad_2": {
            "ts": "2026-08-18T01:02:04+00:00",
            "surface": "triad",
            "status": "error",
            # B1 typed facts: a dated window carries reset_at, an undated one
            # only the code — both must surface independently.
            "failure_code": "subscription_window_exhausted",
            "reset_at": "2026-08-18T09:20:00+00:00",
        },
    }), encoding="utf-8")
    (root / "state" / "subagent_last_delegation.json").write_text(json.dumps({
        "ts": "2026-08-18T02:00:00+00:00",
        "route": "claudexor",
        "requested_model": "opus-5",
        "applied_model": "claude-opus-5",
        "run_id": "run-1",
    }), encoding="utf-8")

    capabilities = _delegation_fact(tmp_path, monkeypatch)
    delegation = capabilities["delegation"]

    assert delegation["configured_route"] == {
        "harness": "claudexor", "model": "opus-5", "effort": "high",
    }
    rows = {row["slot"]: row for row in delegation["reviewer_slots_last"]}
    assert rows["triad_1"]["outcome"] == "ok"
    assert "failure_code" not in rows["triad_1"]
    assert rows["triad_2"]["outcome"] == "failed"
    assert rows["triad_2"]["failure_code"] == "subscription_window_exhausted"
    assert rows["triad_2"]["reset_at"] == "2026-08-18T09:20:00+00:00"
    # Per-row label is the timestamp only; the verbatim historical disclaimer
    # lives ONCE in the note (review fix 12), never repeated per row.
    assert rows["triad_1"]["observed"] == "last observed at 2026-08-18T01:02:03+00:00"
    last = delegation["subagent_last_delegation"]
    assert last["route"] == "claudexor"
    assert last["applied_model"] == "claude-opus-5"
    assert last["observed"] == "last observed at 2026-08-18T02:00:00+00:00"
    assert "historical" not in rows["triad_1"]["observed"]
    # The prompt-visible note teaches the semantics ONCE: rows are history, live
    # facts come from plan-review waves and typed delegate refusals.
    assert "historical, not live health" in delegation["note"]
    assert "plan-review wave rows" in delegation["note"]
    assert "typed" in delegation["note"] and "refusal" in delegation["note"]
    assert "never healthy" in delegation["note"]


def test_delegation_fact_undated_window_code_surfaces_without_reset(tmp_path, monkeypatch):
    root = _delegation_data_root(tmp_path, monkeypatch)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)
    (root / "state" / "reviewer_slot_last_execution.json").write_text(json.dumps({
        "scope": {
            "ts": "2026-08-18T03:00:00+00:00",
            "status": "error",
            "failure_code": "credential_pool_exhausted",
        },
    }), encoding="utf-8")

    delegation = _delegation_fact(tmp_path, monkeypatch)["delegation"]

    (row,) = delegation["reviewer_slots_last"]
    assert row["failure_code"] == "credential_pool_exhausted"
    assert "reset_at" not in row
    assert row["outcome"] == "failed"


def test_delegation_fact_absent_files_mean_absent_observations_not_health(tmp_path, monkeypatch):
    _delegation_data_root(tmp_path, monkeypatch)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)

    delegation = _delegation_fact(tmp_path, monkeypatch)["delegation"]

    assert delegation["configured_route"] == "not configured"
    assert "reviewer_slots_last" not in delegation
    assert "subagent_last_delegation" not in delegation
    # Nothing in the fact may read as a live-health claim.
    assert "healthy" not in json.dumps(
        {k: v for k, v in delegation.items() if k != "note"})


def test_delegation_fact_failure_never_drops_capability_digest(tmp_path, monkeypatch):
    _delegation_data_root(tmp_path, monkeypatch)

    def _boom():
        raise RuntimeError("reader exploded")

    monkeypatch.setattr(
        "ouroboros.reviewer_slot_config.reviewer_slot_last_executions", _boom)

    capabilities = _delegation_fact(tmp_path, monkeypatch)

    assert "delegation" not in capabilities
    # The surrounding digest survives intact.
    assert "allow_mutative_subagents" in capabilities
    assert "write_surfaces" in capabilities
