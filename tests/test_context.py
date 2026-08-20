"""The health invariants ouroboros.context builds, and where they must appear.

This module owns the cache hit-rate invariant, the remote context overflow it reports,
the hot-store growth it watches, the rest of the invariant coverage, and the rule that
the invariants come first in both the dynamic and the background-consciousness context.

The runtime section, the advisory review status, the memory/consolidation sections and
the drive-state projection were split verbatim into
``tests/test_context_runtime_section.py``, ``tests/test_context_advisory_review.py``,
``tests/test_context_memory.py`` and ``tests/test_context_drive_state.py``; the health
environment builder they share lives in ``tests/_context_shared.py``.
"""

from __future__ import annotations

import json


from ouroboros.context import build_health_invariants

from tests._context_shared import _make_health_env



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


def test_health_invariants_reports_remote_context_overflow(tmp_path):
    env = _make_health_env(
        tmp_path,
        [json.dumps({"type": "remote_context_overflow", "model": "provider/model"})],
    )

    result = build_health_invariants(env)

    assert "REMOTE CONTEXT OVERFLOW" in result
    assert "provider/model x1" in result


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

    def test_scheduled_tasks_store_growth_warns_with_receipt_remediation(self, tmp_path):
        """The one-shot follow-up receipts (B2b W=A) made this whole-document
        store grow with every fired follow-up; the scheduler re-parses and
        rewrites it on every tick under the queue lock."""
        from ouroboros.context_budget import SCHEDULED_TASKS_WARN_BYTES

        env = _make_health_env(tmp_path)
        _grow_file(tmp_path / "state" / "scheduled_tasks.json", SCHEDULED_TASKS_WARN_BYTES + 1)

        result = build_health_invariants(env)
        assert "HOT STORE GROWTH" in result
        assert "state/scheduled_tasks.json" in result
        assert "receipts" in result  # remediation pointer

    def test_absent_stores_stay_silent(self, tmp_path):
        env = _make_health_env(tmp_path)

        result = build_health_invariants(env)
        assert "HOT STORE GROWTH" not in result


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
