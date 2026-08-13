"""Tests for ouroboros.context health invariants."""

from __future__ import annotations

import json

import pytest

from ouroboros.context import build_health_invariants, build_runtime_section, build_user_content


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


class TestCacheHitRateInvariant:
    def _make_env(self, tmp_path, events_lines):
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

        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "docs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "VERSION").write_text("1.2.3", encoding="utf-8")
        (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
        (tmp_path / "repo" / "web").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "web" / "package.json").write_text('{"version": "1.2.3"}', encoding="utf-8")
        (tmp_path / "repo" / "README.md").write_text('version-1.2.3', encoding="utf-8")
        (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text('# Ouroboros v1.2.3', encoding="utf-8")
        (tmp_path / "repo" / "docs" / "DEVELOPMENT.md").write_text('# Dev', encoding="utf-8")
        (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0, "budget_drift_alert": false}', encoding="utf-8")
        (tmp_path / "memory" / "identity.md").write_text('x' * 300, encoding="utf-8")
        (tmp_path / "memory" / "scratchpad.md").write_text('x' * 300, encoding="utf-8")
        (tmp_path / "logs" / "events.jsonl").write_text("\n".join(events_lines) + "\n", encoding="utf-8")
        return FakeEnv()

    def test_cache_hit_rate_good(self, tmp_path):
        lines = [json.dumps({"type": "llm_round", "prompt_tokens": 1000, "cached_tokens": 600}) for _ in range(15)]
        env = self._make_env(tmp_path, lines)
        result = build_health_invariants(env)
        assert "cache hit rate" in result.lower()
        assert "60%" in result or "60.0%" in result

    def test_cache_hit_rate_warning_below_30(self, tmp_path):
        lines = [json.dumps({"type": "llm_round", "prompt_tokens": 1000, "cached_tokens": 200}) for _ in range(15)]
        env = self._make_env(tmp_path, lines)
        result = build_health_invariants(env)
        assert "LOW CACHE HIT RATE" in result


def _make_health_env(tmp_path, events_lines=None):
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

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "prompts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "archive" / "rescue").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "VERSION").write_text("1.2.3", encoding="utf-8")
    (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
    (tmp_path / "repo" / "web").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "web" / "package.json").write_text('{"version": "1.2.3"}', encoding="utf-8")
    (tmp_path / "repo" / "README.md").write_text('version-1.2.3', encoding="utf-8")
    (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text('# Ouroboros v1.2.3', encoding="utf-8")
    (tmp_path / "repo" / "docs" / "DEVELOPMENT.md").write_text('# Dev', encoding="utf-8")
    (tmp_path / "repo" / "prompts" / "CONSCIOUSNESS.md").write_text('Prompt text', encoding="utf-8")
    (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0, "budget_drift_alert": false}', encoding="utf-8")
    (tmp_path / "memory" / "identity.md").write_text('x' * 300, encoding="utf-8")
    (tmp_path / "memory" / "scratchpad.md").write_text('x' * 300, encoding="utf-8")
    event_lines = events_lines or []
    (tmp_path / "logs" / "events.jsonl").write_text("\n".join(event_lines) + ("\n" if event_lines else ""), encoding="utf-8")
    return FakeEnv()


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


def test_health_invariants_reports_remote_context_overflow(tmp_path):
    env = _make_health_env(
        tmp_path,
        [json.dumps({"type": "remote_context_overflow", "model": "provider/model"})],
    )

    result = build_health_invariants(env)

    assert "REMOTE CONTEXT OVERFLOW" in result
    assert "provider/model x1" in result


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


class TestAdditionalHealthInvariantCoverage:
    def test_version_desync_warning(self, tmp_path):
        env = _make_health_env(tmp_path)
        (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.4"', encoding="utf-8")

        result = build_health_invariants(env)
        assert "VERSION DESYNC" in result
        assert "pyproject.toml=1.2.4" in result

    def test_web_package_version_desync_warning(self, tmp_path):
        env = _make_health_env(tmp_path)
        (tmp_path / "repo" / "web" / "package.json").write_text('{"version": "1.2.4"}', encoding="utf-8")

        result = build_health_invariants(env)
        assert "VERSION DESYNC" in result
        assert "web/package.json=1.2.4" in result

    def test_rc_pep440_pyproject_does_not_warn(self, tmp_path):
        env = _make_health_env(tmp_path)
        (tmp_path / "repo" / "VERSION").write_text("4.50.0-rc.2", encoding="utf-8")
        (tmp_path / "repo" / "pyproject.toml").write_text('version = "4.50.0rc2"', encoding="utf-8")
        (tmp_path / "repo" / "web" / "package.json").write_text('{"version": "4.50.0-rc.2"}', encoding="utf-8")
        (tmp_path / "repo" / "README.md").write_text(
            "[![Version 4.50.0-rc.2](https://img.shields.io/badge/version-4.50.0--rc.2-green.svg)](VERSION)",
            encoding="utf-8",
        )
        (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text(
            "# Ouroboros v4.50.0-rc.2",
            encoding="utf-8",
        )

        result = build_health_invariants(env)
        assert "VERSION DESYNC" not in result

    def test_rc_badge_url_mismatch_warns(self, tmp_path):
        env = _make_health_env(tmp_path)
        (tmp_path / "repo" / "VERSION").write_text("4.50.0-rc.2", encoding="utf-8")
        (tmp_path / "repo" / "pyproject.toml").write_text('version = "4.50.0rc2"', encoding="utf-8")
        (tmp_path / "repo" / "web" / "package.json").write_text('{"version": "4.50.0-rc.2"}', encoding="utf-8")
        (tmp_path / "repo" / "README.md").write_text(
            "[![Version 4.50.0-rc.2](https://img.shields.io/badge/version-4.50.0-rc.2-green.svg)](VERSION)",
            encoding="utf-8",
        )
        (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text(
            "# Ouroboros v4.50.0-rc.2",
            encoding="utf-8",
        )

        result = build_health_invariants(env)
        assert "VERSION DESYNC" in result
        assert "README badge URL token" in result

    def test_duplicate_processing_warning(self, tmp_path):
        env = _make_health_env(tmp_path)
        (tmp_path / "logs" / "events.jsonl").write_text(
            json.dumps({
                "type": "owner_message_injected",
                "text": "same message",
                "task_id": "task-a",
            }) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "logs" / "supervisor.jsonl").write_text(
            json.dumps({
                "event_type": "owner_message_injected",
                "text": "same message",
                "task_id": "task-b",
            }) + "\n",
            encoding="utf-8",
        )

        result = build_health_invariants(env)
        assert "DUPLICATE PROCESSING" in result
        assert "task-a" in result
        assert "task-b" in result

    def test_provider_and_overflow_warnings(self, tmp_path):
        env = _make_health_env(
            tmp_path,
            events_lines=[
                json.dumps({"type": "llm_api_error", "model": "openai/gpt-5.5"}),
                json.dumps({"type": "local_context_overflow", "model": "local/qwen"}),
            ],
        )

        result = build_health_invariants(env)
        assert "PROVIDER/ROUTING ERRORS" in result
        assert "openai/gpt-5.5 x1" in result
        assert "LOCAL CONTEXT OVERFLOW" in result
        assert "local/qwen x1" in result

    def test_rescue_snapshot_warning(self, tmp_path):
        env = _make_health_env(tmp_path)
        rescue_dir = tmp_path / "archive" / "rescue" / "2026-04-14-test"
        rescue_dir.mkdir(parents=True, exist_ok=True)
        (rescue_dir / "rescue_meta.json").write_text("{}", encoding="utf-8")
        (rescue_dir / "changes.diff").write_text("diff", encoding="utf-8")

        result = build_health_invariants(env)
        assert "RESCUE SNAPSHOT AVAILABLE" in result
        assert "2026-04-14-test" in result


def _grow_file(path, size: int) -> None:
    """Create a file whose st_size is exactly `size` without writing `size` bytes."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.truncate(path, size)


def _grow_ledger(path, size: int) -> None:
    """Grow a synthetic usage ledger WITHOUT triggering tail quarantine.

    build_health_invariants reads the ledger (budget-drift check) BEFORE the
    hot-store stat. A single torn tail row would be QUARANTINED there — the
    substrate ftruncates the file — shrinking st_size before the check under
    test runs. Corruption BEFORE the tail instead raises UsageLedgerCorrupt
    without mutating the file, degrading the budget check to its established
    "COST ACCOUNTING UNAVAILABLE" path while st_size stays exactly `size`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = b"not json\n"
    path.write_bytes(row * (size // len(row)) + b"x" * (size % len(row)))


class TestHotStoreGrowthInvariant:
    def test_progress_growth_warns_above_threshold(self, tmp_path):
        from ouroboros.context_budget import PROGRESS_LOG_WARN_BYTES

        env = _make_health_env(tmp_path)
        _grow_file(tmp_path / "logs" / "progress.jsonl", PROGRESS_LOG_WARN_BYTES + 1)

        result = build_health_invariants(env)
        assert "HOT STORE GROWTH" in result
        assert "logs/progress.jsonl" in result
        assert "rotation" in result  # remediation pointer

    def test_exactly_at_threshold_stays_silent(self, tmp_path):
        from ouroboros.context_budget import PROGRESS_LOG_WARN_BYTES

        env = _make_health_env(tmp_path)
        _grow_file(tmp_path / "logs" / "progress.jsonl", PROGRESS_LOG_WARN_BYTES)

        result = build_health_invariants(env)
        assert "HOT STORE GROWTH" not in result

    def test_ledger_growth_warns_with_lock_remediation(self, tmp_path):
        from ouroboros.context_budget import USAGE_LEDGER_WARN_BYTES

        env = _make_health_env(tmp_path)
        _grow_ledger(tmp_path / "state" / "usage_attempts.jsonl", USAGE_LEDGER_WARN_BYTES + 1)

        result = build_health_invariants(env)
        assert "HOT STORE GROWTH" in result
        assert "state/usage_attempts.jsonl" in result
        assert "monetary lock" in result

    def test_events_and_tools_thresholds_are_generous_but_live(self, tmp_path):
        from ouroboros.context_budget import EVENTS_LOG_WARN_BYTES, TOOLS_LOG_WARN_BYTES

        env = _make_health_env(tmp_path)
        _grow_file(tmp_path / "logs" / "events.jsonl", EVENTS_LOG_WARN_BYTES + 1)
        _grow_file(tmp_path / "logs" / "tools.jsonl", TOOLS_LOG_WARN_BYTES + 1)

        result = build_health_invariants(env)
        assert result.count("HOT STORE GROWTH") == 2
        assert "logs/events.jsonl" in result
        assert "logs/tools.jsonl" in result

    def test_isolated_benchmark_sentinel_suppresses_warnings(self, tmp_path):
        from supervisor.state import ISOLATED_BENCHMARK_SENTINEL
        from ouroboros.context_budget import PROGRESS_LOG_WARN_BYTES, USAGE_LEDGER_WARN_BYTES

        env = _make_health_env(tmp_path)
        _grow_file(tmp_path / "logs" / "progress.jsonl", PROGRESS_LOG_WARN_BYTES + 1)
        _grow_ledger(tmp_path / "state" / "usage_attempts.jsonl", USAGE_LEDGER_WARN_BYTES + 1)
        (tmp_path / ISOLATED_BENCHMARK_SENTINEL).write_text("isolated\n", encoding="utf-8")

        result = build_health_invariants(env)
        assert "HOT STORE GROWTH" not in result

    def test_absent_stores_stay_silent(self, tmp_path):
        env = _make_health_env(tmp_path)

        result = build_health_invariants(env)
        assert "HOT STORE GROWTH" not in result


class TestAdvisoryReviewStatusInContext:
    """Tests that advisory review status appears in LLM context when runs exist."""

    def _make_env(self, tmp_path):
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

        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "docs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "VERSION").write_text("1.2.3", encoding="utf-8")
        (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
        (tmp_path / "repo" / "README.md").write_text('version-1.2.3', encoding="utf-8")
        (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text('# Ouroboros v1.2.3', encoding="utf-8")
        (tmp_path / "repo" / "docs" / "DEVELOPMENT.md").write_text('# Dev', encoding="utf-8")
        (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0, "budget_drift_alert": false}', encoding="utf-8")
        (tmp_path / "memory" / "identity.md").write_text('x' * 300, encoding="utf-8")
        (tmp_path / "memory" / "scratchpad.md").write_text('x' * 300, encoding="utf-8")
        return FakeEnv()

    def test_advisory_status_in_build_llm_messages(self, tmp_path):
        """format_status_section returns non-empty string when runs exist."""
        from ouroboros.review_state import (
            AdvisoryReviewState, AdvisoryRunRecord, save_state, format_status_section
        )
        state = AdvisoryReviewState()
        state.add_run(AdvisoryRunRecord(
            snapshot_hash="abc123",
            commit_message="test commit",
            status="fresh",
            ts="2026-01-01T00:00:00",
            items=[{"item": "bible_compliance", "verdict": "PASS", "severity": "critical", "reason": "ok"}],
        ))
        save_state(tmp_path, state)

        loaded = __import__("ouroboros.review_state", fromlist=["load_state"]).load_state(tmp_path)
        section = format_status_section(loaded)
        assert "Advisory Pre-Review Status" in section
        assert "FRESH" in section
        assert "abc123" in section

    def test_advisory_status_empty_when_no_runs(self, tmp_path):
        """format_status_section returns 'No advisory runs' when state is empty."""
        from ouroboros.review_state import AdvisoryReviewState, format_status_section
        state = AdvisoryReviewState()
        section = format_status_section(state)
        assert "No advisory runs" in section

    def test_review_continuity_context_surfaces_live_gate_and_continuation(self, tmp_path):
        from ouroboros.agent_task_pipeline import build_review_context
        from ouroboros.context import build_llm_messages
        from ouroboros.memory import Memory
        from ouroboros.review_state import (
            AdvisoryReviewState,
            AdvisoryRunRecord,
            CommitAttemptRecord,
            compute_snapshot_hash,
            make_repo_key,
            save_state,
        )
        from ouroboros.task_continuation import ReviewContinuation, save_review_continuation
        from ouroboros.task_results import STATUS_COMPLETED, write_task_result

        env = self._make_env(tmp_path)
        (tmp_path / "repo" / ".git").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "prompts").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "prompts" / "SYSTEM.md").write_text("System", encoding="utf-8")
        (tmp_path / "repo" / "BIBLE.md").write_text("Bible", encoding="utf-8")
        (tmp_path / "repo" / "docs" / "CHECKLISTS.md").write_text("Checklist", encoding="utf-8")
        (tmp_path / "repo" / "tracked.py").write_text("print('hi')\n", encoding="utf-8")

        repo_key = make_repo_key(tmp_path / "repo")
        snapshot_hash = compute_snapshot_hash(tmp_path / "repo")
        state = AdvisoryReviewState()
        state.add_run(AdvisoryRunRecord(
            snapshot_hash=snapshot_hash,
            commit_message="test commit",
            status="bypassed",
            ts="2026-04-07T09:59:00+00:00",
            repo_key=repo_key,
            bypass_reason="manual audit override",
        ))
        state.advisory_runs[-1].status = "stale"
        state.last_stale_from_edit_ts = "2026-04-07T10:00:00+00:00"
        state.last_stale_reason = "edit_text mutated tracked.py"
        state.last_stale_repo_key = repo_key
        state.record_attempt(CommitAttemptRecord(
            ts="2026-04-07T10:01:00+00:00",
            commit_message="blocked commit",
            status="blocked",
            repo_key=repo_key,
            tool_name="commit_reviewed",
            task_id="task-old",
            attempt=1,
            critical_findings=[{
                "item": "tests_affected",
                "reason": "Fix the failing test before commit",
                "severity": "critical",
                "verdict": "FAIL",
            }],
            readiness_warnings=["Review was blocked and needs follow-up."],
        ))
        save_state(tmp_path, state)

        save_review_continuation(
            tmp_path,
            ReviewContinuation(
                task_id="task-old",
                source="blocked_review",
                stage="blocking_review",
                repo_key=repo_key,
                tool_name="commit_reviewed",
                attempt=1,
                block_reason="critical_findings",
                critical_findings=[{
                    "item": "tests_affected",
                    "reason": "Fix the failing test before commit",
                    "severity": "critical",
                    "verdict": "FAIL",
                }],
                readiness_warnings=["Review was blocked and needs follow-up."],
            ),
            expect_task_id="task-old",
        )
        write_task_result(
            tmp_path,
            "task-old",
            STATUS_COMPLETED,
            result="Commit blocked by review.",
        )

        messages, _ = build_llm_messages(
            env=env,
            memory=Memory(drive_root=tmp_path),
            task={"id": "task-new", "type": "task", "text": "continue"},
            review_context_builder=lambda: build_review_context(env),
        )
        dynamic_text = messages[0]["content"][2]["text"]

        assert "## Review Continuity" in dynamic_text
        assert "repo_commit_ready=no" in dynamic_text
        assert "retry_anchor=commit_readiness_debt" in dynamic_text
        assert "Commit-readiness debt" in dynamic_text
        assert "bypass_reason=manual audit override" in dynamic_text
        assert "stale_marker=2026-04-07T10:00:00" in dynamic_text
        assert "### Open review continuations" in dynamic_text
        assert "critical_finding=tests_affected: Fix the failing test before commit" in dynamic_text
        assert "### Historical review ledger" in dynamic_text
        assert "## Scratchpad" in dynamic_text
        assert dynamic_text.index("## Scratchpad") < dynamic_text.index("## Drive state")
        assert dynamic_text.index("## Runtime context") < dynamic_text.index("## Review Continuity")

    def test_review_continuity_context_ignores_foreign_repo_obligations(self, tmp_path):
        from ouroboros.agent_task_pipeline import build_review_context
        from ouroboros.review_state import (
            AdvisoryReviewState,
            AdvisoryRunRecord,
            CommitAttemptRecord,
            compute_snapshot_hash,
            make_repo_key,
            save_state,
        )

        env = self._make_env(tmp_path)
        repo_a = tmp_path / "repo"
        repo_b = tmp_path / "repo-other"
        (repo_a / ".git").mkdir(parents=True, exist_ok=True)
        (repo_b / ".git").mkdir(parents=True, exist_ok=True)
        (repo_a / "tracked.py").write_text("print('repo a')\n", encoding="utf-8")
        (repo_b / "tracked.py").write_text("print('repo b')\n", encoding="utf-8")

        repo_a_key = make_repo_key(repo_a)
        repo_b_key = make_repo_key(repo_b)
        state = AdvisoryReviewState()
        state.add_run(AdvisoryRunRecord(
            snapshot_hash=compute_snapshot_hash(repo_a),
            commit_message="repo a ready",
            status="fresh",
            ts="2026-04-07T10:00:00+00:00",
            repo_key=repo_a_key,
        ))
        state.record_attempt(CommitAttemptRecord(
            ts="2026-04-07T10:01:00+00:00",
            commit_message="repo b blocked",
            status="blocked",
            repo_key=repo_b_key,
            tool_name="commit_reviewed",
            task_id="task-b",
            attempt=1,
            block_reason="critical_findings",
            critical_findings=[{
                "item": "foreign_issue",
                "reason": "other repo only",
                "severity": "critical",
                "verdict": "FAIL",
            }],
        ))
        save_state(tmp_path, state)

        dynamic_text = build_review_context(env)
        assert "repo_commit_ready=yes" in dynamic_text
        assert "foreign_issue" not in dynamic_text
        assert "repo b blocked" not in dynamic_text

    def test_review_continuity_context_keeps_open_obligations_without_runs(self, tmp_path):
        from ouroboros.agent_task_pipeline import build_review_context
        from ouroboros.review_state import (
            AdvisoryReviewState,
            ObligationItem,
            make_repo_key,
            save_state,
        )

        env = self._make_env(tmp_path)
        (tmp_path / "repo" / ".git").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "tracked.py").write_text("print('hi')\n", encoding="utf-8")

        repo_key = make_repo_key(tmp_path / "repo")
        state = AdvisoryReviewState(
            open_obligations=[
                ObligationItem(
                    obligation_id="obl-0001",
                    item="tests_affected",
                    severity="critical",
                    reason="Coverage still missing",
                    source_attempt_ts="2026-04-07T10:00:00+00:00",
                    source_attempt_msg="blocked commit",
                    repo_key=repo_key,
                    fingerprint="finding:tests_affected:abc123",
                )
            ]
        )
        save_state(tmp_path, state)

        dynamic_text = build_review_context(env)
        assert "## Review Continuity" in dynamic_text
        assert "open_obligations=1" in dynamic_text
        assert "[obl-0001] tests_affected: Coverage still missing" in dynamic_text

    def test_review_continuity_context_keeps_all_debt_evidence(self, tmp_path):
        from ouroboros.agent_task_pipeline import build_review_context
        from ouroboros.review_state import (
            AdvisoryReviewState,
            CommitReadinessDebtItem,
            make_repo_key,
            save_state,
        )

        env = self._make_env(tmp_path)
        (tmp_path / "repo" / ".git").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "tracked.py").write_text("print('hi')\n", encoding="utf-8")

        repo_key = make_repo_key(tmp_path / "repo")
        state = AdvisoryReviewState(
            commit_readiness_debts=[
                CommitReadinessDebtItem(
                    debt_id="debt-0001",
                    category="repeated_obligation",
                    title="Commit readiness debt",
                    summary="Repeated tests blocker",
                    repo_key=repo_key,
                    source_obligation_ids=["obl-0001"],
                    evidence=[
                        "first evidence",
                        "second evidence",
                        "third evidence",
                    ],
                )
            ]
        )
        save_state(tmp_path, state)

        dynamic_text = build_review_context(env)
        assert "first evidence" in dynamic_text
        assert "second evidence" in dynamic_text
        assert "third evidence" in dynamic_text


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
    """build_runtime_section includes runtime_env with platform and is_desktop."""

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

    def test_runtime_env_present(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        monkeypatch.delenv("OUROBOROS_DESKTOP_MODE", raising=False)
        env = self._make_env(tmp_path)
        section = build_runtime_section(env, {"id": "t1", "type": "task"})
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert "runtime_env" in data
        assert "platform" in data["runtime_env"]
        assert isinstance(data["runtime_env"]["platform"], str)
        assert data["runtime_env"]["is_desktop"] is False

    def test_runtime_env_desktop_flag(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        monkeypatch.setenv("OUROBOROS_DESKTOP_MODE", "1")
        env = self._make_env(tmp_path)
        section = build_runtime_section(env, {"id": "t2", "type": "task"})
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert data["runtime_env"]["is_desktop"] is True


# ===========================================================================
# Memory / consolidation offset behavior (merged from former
# test_context_memory_overhaul.py).  Inspect-only `limit=50` / `limit=1000`
# source-string pins were dropped — behavioral coverage below already
# exercises the offset path.  test_no_identity_truncation_in_consolidator_
# prompts was also dropped (inspect-only); identity-truncation is covered
# behaviorally by consolidator tests.
# ===========================================================================


def test_recent_chat_starts_after_consolidated_offset(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    memory_dir = tmp_path / "memory"
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {"ts": f"2026-03-19T16:{i:02d}:00Z", "direction": "in", "username": "User", "text": f"msg-{i}"}
        for i in range(5)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)
    (memory_dir / "dialogue_meta.json").write_text(
        json.dumps({
            "last_consolidated_offset": 3,
            "chat_log_signature": memory.jsonl_generation_signature("chat.jsonl"),
        }),
        encoding="utf-8",
    )

    sections = build_recent_sections(memory, env=None)
    combined = "\n\n".join(sections)

    assert "msg-0" not in combined
    assert "msg-1" not in combined
    assert "msg-2" not in combined
    assert "msg-3" in combined
    assert "msg-4" in combined


def test_recent_chat_main_includes_all_threads_full_awareness(tmp_path):
    """Full project awareness (v6.32.0): the one identity's main/global context
    sees its WHOLE conversation — main + project threads alike (BIBLE P1, one
    awareness across direct chat, project rooms, and consciousness). Project chat
    is part of the one mind's memory, NOT partitioned out; only A2A virtual
    transport is excluded (covered elsewhere)."""
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    project = create_project(tmp_path, "racer")
    project_chat = int(project["chat_id"])
    transport_chat = 555000111  # large NON-project id (e.g. a Telegram mirror)

    entries = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": "main-keep"},
        {"chat_id": project_chat, "direction": "in", "username": "User", "text": "project-visible"},
        {"chat_id": transport_chat, "direction": "in", "username": "User", "text": "transport-keep"},
        {"direction": "in", "username": "User", "text": "legacy-keep"},  # no chat_id -> main
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    combined = "\n\n".join(build_recent_sections(Memory(drive_root=tmp_path), env=None))

    assert "main-keep" in combined
    assert "legacy-keep" in combined
    assert "transport-keep" in combined
    assert "project-visible" in combined  # full awareness: the one mind sees project chat


def test_recent_chat_for_project_thread_shows_only_its_own_thread(tmp_path):
    """A project TASK gets a FOCUSED working view of its own thread (full
    awareness, v6.32.0): its "## Recent chat" is its own project thread, not the
    штаб's main chat nor a sibling project's chat, so cross-project noise does not
    bloat its working context. This is focus, not memory isolation — the one mind
    still sees everything via the main/background path. Pins that thread_chat_id
    selects the project's own raw tail rather than the main consolidation stream."""
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    proj_a = create_project(tmp_path, "racer")
    proj_b = create_project(tmp_path, "research")
    chat_a = int(proj_a["chat_id"])
    chat_b = int(proj_b["chat_id"])

    entries = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": "main-stab-chat"},
        {"chat_id": chat_a, "direction": "in", "username": "User", "text": "project-a-own-thread"},
        {"chat_id": chat_b, "direction": "in", "username": "User", "text": "project-b-sibling"},
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    combined = "\n\n".join(build_recent_sections(
        Memory(drive_root=tmp_path), env=None, thread_chat_id=chat_a))

    assert "project-a-own-thread" in combined   # its own thread is visible
    assert "project-b-sibling" not in combined  # sibling project not in focused view
    assert "main-stab-chat" not in combined     # main chat not in focused project view


def test_project_workpad_and_journal_not_silently_sliced(tmp_path, monkeypatch):
    """BIBLE P1 (no silent truncation): project cognitive artifacts are not
    prefix-sliced into context. The workpad rides in FULL; journal milestones show
    full text (no per-row [:N]) with a visible journal_read pointer for older."""
    import types

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.context import build_knowledge_sections
    from ouroboros.project_facts import project_journal_path, project_workpad_path
    from ouroboros.utils import append_jsonl

    pid = "builder"
    wp = project_workpad_path(pid)
    wp.parent.mkdir(parents=True, exist_ok=True)
    tail = "WORKPAD_TAIL_MARKER"
    wp.write_text("A" * 20_000 + tail, encoding="utf-8")  # > old 12_000 slice
    append_jsonl(project_journal_path(pid), {
        "ts": "2026-06-14T00:00:00Z", "kind": "checkpoint", "text": "M" * 600,  # > old 200 slice
    })

    env = types.SimpleNamespace(drive_path=lambda rel: tmp_path / rel)
    combined = "\n\n".join(build_knowledge_sections(env, project_id=pid))

    assert tail in combined          # full workpad, not prefix-sliced to 12_000
    assert ("M" * 600) in combined   # full journal milestone, not sliced to 200


def test_append_journal_milestone_bounds_over_limit_with_pointer(tmp_path, monkeypatch):
    """An AUTOMATIC completion milestone honors the journal's durable per-row cap:
    over-limit text is bounded with a VISIBLE pointer (recorded, never silently
    sliced nor dropped) — same _MAX_TEXT_CHARS contract as the journal_write tool,
    so emit_task_results cannot append a raw unbounded row."""
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.project_facts import project_journal_path
    from ouroboros.tools.project_journal import _MAX_TEXT_CHARS, append_journal_milestone
    from ouroboros.utils import iter_jsonl_objects

    pid = "lh"
    append_journal_milestone(pid, "done", "Z" * (_MAX_TEXT_CHARS + 500), task_id="t1")
    rows = [r for r in iter_jsonl_objects(project_journal_path(pid)) if isinstance(r, dict)]
    assert len(rows) == 1                      # recorded (not dropped/rejected)
    txt = rows[0]["text"]
    assert len(txt) <= _MAX_TEXT_CHARS         # honors the durable per-row contract
    assert "task_results" in txt               # VISIBLE pointer to the full text


def test_low_mode_preserves_full_unconsolidated_dialogue_suffix(tmp_path, monkeypatch):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    memory_dir = tmp_path / "memory"
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    fresh_count = 305
    entries = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"consolidated-{i}"}
        for i in range(3)
    ] + [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"fresh-{i}"}
        for i in range(fresh_count)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)
    (memory_dir / "dialogue_meta.json").write_text(
        json.dumps({
            "last_consolidated_offset": 3,
            "chat_log_signature": memory.jsonl_generation_signature("chat.jsonl"),
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")

    combined = "\n\n".join(build_recent_sections(memory, env=None))

    assert "consolidated-0" not in combined
    assert "fresh-0" in combined
    assert f"fresh-{fresh_count - 1}" in combined


def test_low_mode_without_consolidation_keeps_max_raw_dialogue_tail(tmp_path, monkeypatch):
    from ouroboros.context import build_recent_sections
    from ouroboros.context_budget import MAX_RECENT_CHAT_TAIL
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fresh_count = 305
    entries = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"fresh-{i}"}
        for i in range(fresh_count)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")

    combined = "\n\n".join(build_recent_sections(Memory(drive_root=tmp_path), env=None))

    assert fresh_count < MAX_RECENT_CHAT_TAIL
    assert "fresh-0" in combined
    assert f"fresh-{fresh_count - 1}" in combined


def test_recent_chat_offset_uses_filtered_dialogue_entries(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    memory_dir = tmp_path / "memory"
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": "consolidated-0"},
        {"chat_id": -1, "direction": "in", "username": "Agent", "text": "a2a-noise"},
        {"chat_id": 1, "direction": "in", "username": "User", "text": "consolidated-1"},
        {"chat_id": 1, "direction": "in", "username": "User", "text": "fresh"},
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)
    (memory_dir / "dialogue_meta.json").write_text(
        json.dumps({
            "last_consolidated_offset": 2,
            "chat_log_signature": memory.jsonl_generation_signature("chat.jsonl"),
        }),
        encoding="utf-8",
    )

    combined = "\n\n".join(build_recent_sections(memory, env=None))

    assert "consolidated-0" not in combined
    assert "consolidated-1" not in combined
    assert "a2a-noise" not in combined
    assert "fresh" in combined


def test_recent_chat_ignores_stale_consolidation_offset_after_rotation(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    memory_dir = tmp_path / "memory"
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    initial = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"early-{i}"}
        for i in range(3)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in initial) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)
    stale_signature = memory.jsonl_generation_signature("chat.jsonl")
    (memory_dir / "dialogue_meta.json").write_text(
        json.dumps({
            "last_consolidated_offset": 3,
            "chat_log_signature": stale_signature,
        }),
        encoding="utf-8",
    )

    rotated = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"post-rotate-{i}"}
        for i in range(2)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in rotated) + "\n",
        encoding="utf-8",
    )

    combined = "\n\n".join(build_recent_sections(memory, env=None))

    # Rotation invalidates the stale offset; rotated entries appear.
    assert "post-rotate-0" in combined
    assert "post-rotate-1" in combined


def test_recent_chat_keeps_offset_when_same_log_gets_appended(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    memory_dir = tmp_path / "memory"
    logs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    initial = [
        {"chat_id": 1, "direction": "in", "username": "User", "text": f"old-{i}"}
        for i in range(3)
    ]
    (logs_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in initial) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)
    (memory_dir / "dialogue_meta.json").write_text(
        json.dumps({
            "last_consolidated_offset": 3,
            "chat_log_signature": memory.jsonl_generation_signature("chat.jsonl"),
        }),
        encoding="utf-8",
    )

    with open(logs_dir / "chat.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"chat_id": 1, "direction": "in", "username": "User", "text": "new"}) + "\n")

    combined = "\n\n".join(build_recent_sections(memory, env=None))

    assert "old-0" not in combined
    assert "new" in combined


def test_world_profile_is_loaded_with_stable_memory(tmp_path):
    from ouroboros.context import build_memory_sections
    from ouroboros.memory import Memory

    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "WORLD.md").write_text("world-profile-data", encoding="utf-8")
    memory = Memory(drive_root=tmp_path)

    sections = build_memory_sections(memory)
    combined = "\n\n".join(sections)

    assert "world-profile-data" in combined


def test_retired_dialogue_summary_remains_visible_when_blocks_exist(tmp_path):
    from ouroboros.context import build_memory_sections
    from ouroboros.memory import Memory

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "dialogue_summary.md").write_text("legacy dialogue", encoding="utf-8")
    (memory_dir / "dialogue_blocks.json").write_text(
        json.dumps([{"content": "new dialogue block"}]),
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)

    combined = "\n\n".join(build_memory_sections(memory, partition="volatile"))

    assert "## Dialogue History" in combined
    assert "new dialogue block" in combined
    assert "## Legacy Dialogue Summary (retired flat format, read-only fallback)" in combined
    assert "legacy dialogue" in combined


def test_retired_dialogue_summary_fallback_preserves_continuity_without_blocks(tmp_path):
    from ouroboros.context import build_memory_sections
    from ouroboros.memory import Memory

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "dialogue_summary.md").write_text("legacy dialogue only", encoding="utf-8")
    memory = Memory(drive_root=tmp_path)

    combined = "\n\n".join(build_memory_sections(memory, partition="volatile"))

    assert "## Legacy Dialogue Summary (retired flat format, read-only fallback)" in combined
    assert "legacy dialogue only" in combined


def test_recent_sections_filter_process_logs_by_task_id(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "progress.jsonl").write_text(
        "\n".join([
            json.dumps({"task_id": "task-a", "text": "in-scope"}),
            json.dumps({"task_id": "task-b", "text": "out-of-scope"}),
        ]) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "tools.jsonl").write_text(
        "\n".join([
            json.dumps({"task_id": "task-a", "tool": "shell"}),
            json.dumps({"task_id": "task-b", "tool": "shell"}),
        ]) + "\n",
        encoding="utf-8",
    )

    memory = Memory(drive_root=tmp_path)
    sections = build_recent_sections(memory, env=None, task_id="task-a")
    combined = "\n\n".join(sections)
    assert "in-scope" in combined
    assert "out-of-scope" not in combined


def test_installed_skills_section_includes_warnings_verdict(tmp_path, monkeypatch):
    from ouroboros.context import _build_installed_skills_section

    class FakeEnv:
        drive_root = tmp_path

    monkeypatch.setattr(
        "ouroboros.skill_loader.summarize_skills",
        lambda _root: {
            "skills": [
                {
                    "name": "weather",
                    "type": "script",
                    "enabled": True,
                    "review_status": "warnings",
                    "executable_review": True,
                    "review_stale": False,
                    "description": "Weather helper",
                }
            ]
        },
    )

    section = _build_installed_skills_section(FakeEnv())

    assert "## Installed Skills" in section
    assert "weather" in section
    assert "warnings" in section


def test_health_invariants_come_first_in_dynamic_context(tmp_path):
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
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    (tmp_path / "repo" / "prompts" / "SYSTEM.md").write_text("System prompt", encoding="utf-8")
    (tmp_path / "repo" / "BIBLE.md").write_text("Bible", encoding="utf-8")
    (tmp_path / "repo" / "README.md").write_text("README", encoding="utf-8")
    (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros v1.2.3", encoding="utf-8")
    (tmp_path / "repo" / "docs" / "DEVELOPMENT.md").write_text(
        "### File Size Budgets\n| Path | Budget chars |\n|------|--------------|\n| memory/identity.md | 1000 |\n",
        encoding="utf-8",
    )
    (tmp_path / "repo" / "docs" / "CHECKLISTS.md").write_text("Checklist", encoding="utf-8")
    (tmp_path / "repo" / "VERSION").write_text("1.2.3", encoding="utf-8")
    (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
    (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0, "budget_drift_alert": false}', encoding="utf-8")
    (tmp_path / "memory" / "identity.md").write_text("x" * 950, encoding="utf-8")
    (tmp_path / "memory" / "scratchpad.md").write_text("scratchpad", encoding="utf-8")

    messages, _cap_info = build_llm_messages(
        env=FakeEnv(),
        memory=Memory(drive_root=tmp_path),
        task={"id": "task-a", "type": "task", "text": "hello"},
    )

    dynamic_text = messages[0]["content"][2]["text"]
    assert dynamic_text.startswith("## Health Invariants")
    assert dynamic_text.index("## Health Invariants") < dynamic_text.index("## Drive state")


def test_health_invariants_come_first_in_background_consciousness_context(tmp_path):
    from ouroboros.consciousness import BackgroundConsciousness

    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    (repo_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (repo_dir / "docs").mkdir(parents=True, exist_ok=True)
    (drive_root / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
    (drive_root / "logs").mkdir(parents=True, exist_ok=True)
    (drive_root / "state").mkdir(parents=True, exist_ok=True)

    (repo_dir / "prompts" / "CONSCIOUSNESS.md").write_text("Consciousness prompt", encoding="utf-8")
    (repo_dir / "BIBLE.md").write_text("Bible", encoding="utf-8")
    (repo_dir / "VERSION").write_text("1.2.3", encoding="utf-8")
    (repo_dir / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
    (repo_dir / "README.md").write_text("README", encoding="utf-8")
    (repo_dir / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros v1.2.3", encoding="utf-8")
    (repo_dir / "docs" / "DEVELOPMENT.md").write_text(
        "### File Size Budgets\n| Path | Budget chars |\n|------|--------------|\n| memory/identity.md | 1000 |\n",
        encoding="utf-8",
    )
    (drive_root / "state" / "state.json").write_text('{"spent_usd": 0, "budget_drift_alert": false}', encoding="utf-8")
    (drive_root / "memory" / "identity.md").write_text("x" * 950, encoding="utf-8")
    (drive_root / "memory" / "scratchpad.md").write_text("scratchpad", encoding="utf-8")
    (drive_root / "logs" / "chat.jsonl").write_text("", encoding="utf-8")
    (drive_root / "logs" / "progress.jsonl").write_text("", encoding="utf-8")
    (drive_root / "logs" / "tools.jsonl").write_text("", encoding="utf-8")
    (drive_root / "logs" / "events.jsonl").write_text("", encoding="utf-8")
    (drive_root / "logs" / "supervisor.jsonl").write_text("", encoding="utf-8")
    (drive_root / "logs" / "task_reflections.jsonl").write_text("", encoding="utf-8")

    bg = BackgroundConsciousness(
        drive_root=drive_root,
        repo_dir=repo_dir,
        event_queue=None,
        owner_chat_id_fn=lambda: None,
    )

    text = bg._build_context()
    assert text.index("## Health Invariants") < text.index("## Drive state")


def test_drive_state_section_is_typed_projection_with_pointer(tmp_path):
    """W3 adjacent (a): the Drive state section projects the fields the agent
    reasons about and NAMES the omitted internal caches with an on-demand
    pointer (P1: disclosed omission) instead of dumping state.json wholesale —
    the budget narrative stays with the usage-accounting authority in the
    Runtime section."""
    import json

    from ouroboros.context import _drive_state_section

    class FakeEnv:
        def drive_path(self, p):
            return tmp_path / p

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "state.json").write_text(json.dumps({
        "session_id": "abc123",
        "current_branch": "ouroboros",
        "evolution_mode_enabled": False,
        "budget_drift_alert": True,
        "budget_drift_pct": 48.05,
        "spent_usd": 1699.3,
        "managed_update_cache": {"latest_sha": "x" * 40, "latest_message": "big"},
        "usage_accounting": {"settled_usd": 1633.1},
        "openrouter_last_check_call": 5750,
    }), encoding="utf-8")

    section = _drive_state_section(FakeEnv())

    assert section.startswith("## Drive state")
    assert '"session_id": "abc123"' in section
    assert '"budget_drift_alert": true' in section
    # Internal caches / duplicated spend narrative are OMITTED but NAMED.
    assert '"managed_update_cache"' not in section
    assert '"usage_accounting"' not in section
    assert '"spent_usd"' not in section
    for named in ("managed_update_cache", "usage_accounting", "spent_usd", "openrouter_last_check_call"):
        assert named in section  # named in the omission note
    assert "read_file(root='runtime_data', path='state/state.json')" in section

    # Missing/empty file: still a valid section, no omission note needed.
    (tmp_path / "state" / "state.json").unlink()
    empty = _drive_state_section(FakeEnv())
    assert empty.startswith("## Drive state")
    assert "read_file" not in empty


def test_review_ledger_caps_runs_and_attempts_with_omission_notes(tmp_path):
    """W3 adjacent (b): the historical review ledger rides into EVERY task's
    context — cap runs/attempts at the 5 most recent with EXPLICIT omission
    notes (the continuation pattern) and truncate commit messages; the full
    ledger stays behind review_status."""
    from ouroboros.review_state import (
        AdvisoryReviewState,
        AdvisoryRunRecord,
        CommitAttemptRecord,
        format_status_section,
    )

    state = AdvisoryReviewState()
    long_msg = "feat: " + ("y" * 2000)
    for i in range(8):
        state.add_run(AdvisoryRunRecord(
            snapshot_hash=f"hash{i:04d}00000000",
            commit_message=long_msg if i == 7 else f"commit {i}",
            status="fresh",
            ts=f"2026-01-0{i + 1}T00:00:00",
        ))
    for i in range(8):
        state.record_attempt(CommitAttemptRecord(
            status="succeeded",
            commit_message=f"attempt commit {i}",
            ts=f"2026-01-0{i + 1}T01:00:00",
            attempt=i + 1,
        ))

    section = format_status_section(state)

    assert "3 older advisory run(s) omitted" in section
    assert "3 older attempt(s) omitted" in section
    assert "review_status" in section
    assert "hash0007" in section       # newest kept
    assert "hash0000" not in section   # oldest omitted
    assert "attempt commit 7" in section
    assert "attempt commit 0" not in section
    # The 2000-char commit message is display-truncated with the explicit notice.
    assert "y" * 2000 not in section
    assert "truncated at 300 chars" in section


def test_settled_continuations_retire_after_age_window(tmp_path):
    """W3 adjacent (b): a continuation whose owning task SETTLED and that sat
    un-resumed past the age window is archived (durable move, never deleted);
    fresh settled records stay — they are the designed cross-task resume
    pointer."""
    from ouroboros.task_continuation import (
        ReviewContinuation,
        archived_continuation_dir,
        continuation_path,
        list_review_continuations,
        retire_settled_continuations,
        save_review_continuation,
    )

    old = save_review_continuation(tmp_path, ReviewContinuation(
        task_id="oldtask", source="commit_blocked", stage="review"))
    # Age the record past the window (rewrite the stored timestamps).
    import json as _json
    path = continuation_path(tmp_path, "oldtask")
    data = _json.loads(path.read_text(encoding="utf-8"))
    data["created_ts"] = data["updated_ts"] = "2026-01-01T00:00:00+00:00"
    path.write_text(_json.dumps(data), encoding="utf-8")

    save_review_continuation(tmp_path, ReviewContinuation(
        task_id="freshtask", source="commit_blocked", stage="review"))

    settled = {"oldtask": True, "freshtask": True, "runningtask": False}
    retired = retire_settled_continuations(tmp_path, is_settled=lambda tid: settled.get(tid, False))

    assert retired == ["oldtask"]
    assert not continuation_path(tmp_path, "oldtask").exists()
    assert (archived_continuation_dir(tmp_path) / "oldtask.json").exists()  # durable, not deleted
    remaining, _corrupt = list_review_continuations(tmp_path)
    assert [c.task_id for c in remaining] == ["freshtask"]

    # An old continuation of a NON-settled task stays put.
    save_review_continuation(tmp_path, ReviewContinuation(
        task_id="runningtask", source="commit_blocked", stage="review"))
    path = continuation_path(tmp_path, "runningtask")
    data = _json.loads(path.read_text(encoding="utf-8"))
    data["created_ts"] = data["updated_ts"] = "2026-01-01T00:00:00+00:00"
    path.write_text(_json.dumps(data), encoding="utf-8")
    assert retire_settled_continuations(tmp_path, is_settled=lambda tid: settled.get(tid, False)) == []
    assert continuation_path(tmp_path, "runningtask").exists()
    assert old.task_id == "oldtask"


def test_settled_continuation_with_open_obligations_survives_age_retirement(tmp_path):
    """A settled FAILED task whose continuation records obligations that are
    STILL open in the review ledger is genuinely unresolved review work: age
    must not archive it out of context (P1/P3). A same-age settled sibling with
    no open markers still retires — the noise-reduction path stays."""
    import json as _json

    from ouroboros.agent_task_pipeline import build_review_context
    from ouroboros.review_state import (
        AdvisoryReviewState,
        ObligationItem,
        make_repo_key,
        save_state,
    )
    from ouroboros.task_continuation import (
        ReviewContinuation,
        archived_continuation_dir,
        continuation_path,
        save_review_continuation,
    )

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

    env = FakeEnv()
    (tmp_path / "repo" / ".git").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "tracked.py").write_text("print('hi')\n", encoding="utf-8")
    repo_key = make_repo_key(tmp_path / "repo")

    def _aged_continuation(task_id, obligation_ids):
        save_review_continuation(tmp_path, ReviewContinuation(
            task_id=task_id, source="commit_blocked", stage="review",
            block_reason="critical_findings", obligation_ids=obligation_ids))
        path = continuation_path(tmp_path, task_id)
        data = _json.loads(path.read_text(encoding="utf-8"))
        data["created_ts"] = data["updated_ts"] = "2026-01-01T00:00:00+00:00"
        path.write_text(_json.dumps(data), encoding="utf-8")

    _aged_continuation("unresolvedtask", ["obl-open-1"])
    _aged_continuation("closedtask", ["obl-long-gone"])
    task_results = tmp_path / "task_results"
    task_results.mkdir(parents=True, exist_ok=True)
    for tid in ("unresolvedtask", "closedtask"):
        (task_results / f"{tid}.json").write_text(
            _json.dumps({"id": tid, "status": "failed"}), encoding="utf-8")

    state = AdvisoryReviewState(open_obligations=[
        ObligationItem(
            obligation_id="obl-open-1",
            item="tests_affected",
            severity="critical",
            reason="Coverage still missing",
            source_attempt_ts="2026-01-01T00:00:00+00:00",
            source_attempt_msg="blocked commit",
            repo_key=repo_key,
            fingerprint="finding:tests_affected:abc123",
        )
    ])
    save_state(tmp_path, state)

    dynamic_text = build_review_context(env)

    # Unresolved work survives the age window and stays in cognitive context.
    assert continuation_path(tmp_path, "unresolvedtask").exists()
    assert "task=unresolvedtask" in dynamic_text
    # The provably-closed sibling still rides the age path (durable, disclosed).
    assert not continuation_path(tmp_path, "closedtask").exists()
    assert (archived_continuation_dir(tmp_path) / "closedtask.json").exists()
    assert "closedtask" in dynamic_text  # transient archive disclosure line
