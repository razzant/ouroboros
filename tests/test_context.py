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

import pytest

from ouroboros.context import build_health_invariants, build_runtime_section

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

    def _emit_producer_rounds(self, tmp_path, reported, count=6):
        """Rounds in the PRODUCER's exact shape.

        `loop_llm_call.call_llm_with_retry` is the only emitter of `llm_round`,
        so a synthesized row proves nothing about what the health line reads:
        the first version of this rule was inert because the producer stamped
        `cached_tokens` on every event, including rounds where the provider had
        reported no cache at all. `reported=None` is a provider that says
        nothing about caching.
        """
        from ouroboros.loop_llm_call import call_llm_with_retry

        class _LLM:
            def chat(self, **_kwargs):
                usage = {"provider": "openrouter", "resolved_model": "m",
                         "prompt_tokens": 1000, "completion_tokens": 10, "cost": 0.0}
                if reported is not None:
                    usage["cached_tokens"] = reported
                return {"content": "ok"}, usage

        for index in range(count):
            call_llm_with_retry(
                _LLM(), [{"role": "user", "content": "hi"}], "m", None, "medium", 1,
                tmp_path / "logs", "cache-probe", index, None, {}, "task", False,
            )

    def test_no_provider_reported_a_cache_leaves_the_share_unknown(self, tmp_path):
        """Absence is not a measured zero: a run whose provider never reported a
        cache rendered as an honest 0% and read as a caching regression nobody
        had measured."""
        from ouroboros.context_health import _compute_cache_hit_rate

        env = self._make_env(tmp_path, [])
        self._emit_producer_rounds(tmp_path, None)
        assert _compute_cache_hit_rate(env) is None
        assert "cache hit rate" not in build_health_invariants(env).lower()

    def test_an_explicitly_reported_zero_is_still_a_real_zero(self, tmp_path):
        from ouroboros.context_health import _compute_cache_hit_rate

        env = self._make_env(tmp_path, [])
        self._emit_producer_rounds(tmp_path, 0)
        assert _compute_cache_hit_rate(env) == 0.0
        assert "LOW CACHE HIT RATE" in build_health_invariants(env)

    def test_a_window_with_fewer_than_five_reporting_rounds_stays_unknown(self, tmp_path):
        """The five-round threshold counts MEASUREMENTS, not rounds. Three
        reporters beside three silent rounds are still too thin a sample to
        publish a share, so the invariant says nothing rather than a number the
        window cannot support."""
        from ouroboros.context_health import _compute_cache_hit_rate

        env = self._make_env(tmp_path, [])
        self._emit_producer_rounds(tmp_path, None, count=3)
        self._emit_producer_rounds(tmp_path, 600, count=3)
        assert _compute_cache_hit_rate(env) is None

    def test_silent_rounds_stay_out_of_the_reporters_denominator(self, tmp_path):
        """A mixed install reads the reporters' own ratio.

        Charging a silent round's prompt tokens to the denominator turned a
        provider that never measured a cache into measured misses: the share
        collapsed and "LOW CACHE HIT RATE" fired for a regression no round had
        measured. Five reporters at 600 cached of 1000 prompt are 60%, whatever
        the silent rounds beside them spent."""
        from ouroboros.context_health import _compute_cache_hit_rate

        env = self._make_env(tmp_path, [])
        self._emit_producer_rounds(tmp_path, None, count=3)
        self._emit_producer_rounds(tmp_path, 600, count=5)
        assert _compute_cache_hit_rate(env) == 0.6

    @pytest.mark.parametrize("reports, expected_rate", [
        ([None] * 6, None),
        ([0] * 6, 0.0),
        ([600] * 6, 0.6),
        ([None] * 10 + [600] * 5, 0.6),
    ], ids=["unknown", "measured_zero", "measured_hit", "mixed"])
    def test_nullable_adapter_cache_reaches_durable_and_live_rounds(
        self, tmp_path, reports, expected_rate,
    ):
        """The real adapter uses a present null key for an unreported cache.

        Omitted-key fixtures alone missed the producer turning that null into
        zero. Both round events must preserve the measurement before health
        computes its share over the reporting rounds.
        """
        from queue import Queue
        from ouroboros.context_health import _compute_cache_hit_rate
        from ouroboros.llm_claudexor import _usage
        from ouroboros.loop_llm_call import call_llm_with_retry

        env = self._make_env(tmp_path, [])
        events = Queue()
        accumulated = {}

        class LLM:
            def chat(self, **_kwargs):
                counters = {"input_tokens": 1000, "output_tokens": 10}
                if reported is not None:
                    counters["cached_input_tokens"] = reported
                usage, cost, final = _usage({"usage": counters})
                usage.update(provider="claudexor", resolved_model="claudexor/probe",
                             cost=cost, cost_final=final)
                return {"content": "Completed synthetic round."}, usage

        for index, reported in enumerate(reports, 1):
            message, _cost = call_llm_with_retry(
                LLM(), [{"role": "user", "content": "Check the report."}],
                "claudexor/probe", None, "medium", 1, tmp_path / "logs",
                "nullable-cache", index, events, accumulated,
            )
            assert message["content"] == "Completed synthetic round."
        rows = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()
                if line.strip()]
        durable = [row["cached_tokens"] for row in rows if row.get("type") == "llm_round"]
        live = []
        while not events.empty():
            event = events.get_nowait()
            if event.get("type") == "log_event" and event["data"].get("type") == "llm_round_finished":
                live.append(event["data"]["cached_tokens"])
        assert durable == reports
        assert live == reports
        assert _compute_cache_hit_rate(env) == expected_rate


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

    def test_rotated_log_thresholds_are_regression_tripwires(self, tmp_path):
        """events/tools/supervisor/task_reflections rotate on the supervisor
        tick (CPL4-C1..C4); their thresholds fire only when rotation is broken."""
        from ouroboros.context_budget import (
            EVENTS_LOG_WARN_BYTES,
            SUPERVISOR_LOG_WARN_BYTES,
            TASK_REFLECTIONS_LOG_WARN_BYTES,
            TOOLS_LOG_WARN_BYTES,
        )

        env = _make_health_env(tmp_path)
        _grow_file(tmp_path / "logs" / "events.jsonl", EVENTS_LOG_WARN_BYTES + 1)
        _grow_file(tmp_path / "logs" / "tools.jsonl", TOOLS_LOG_WARN_BYTES + 1)
        _grow_file(tmp_path / "logs" / "supervisor.jsonl", SUPERVISOR_LOG_WARN_BYTES + 1)
        _grow_file(
            tmp_path / "logs" / "task_reflections.jsonl",
            TASK_REFLECTIONS_LOG_WARN_BYTES + 1,
        )

        result = build_health_invariants(env)
        assert result.count("HOT STORE GROWTH") == 4
        assert "logs/events.jsonl" in result
        assert "logs/tools.jsonl" in result
        assert "logs/supervisor.jsonl" in result
        assert "logs/task_reflections.jsonl" in result
        assert "rotation is broken or missing" in result

    def test_events_archive_chain_growth_warns(self, tmp_path):
        """Custody replay walks the whole events chain; the pre-rotation 100MB
        replay-degradation signal now watches live + archive segments."""
        from ouroboros.context_budget import EVENTS_ARCHIVE_SCAN_WARN_BYTES

        env = _make_health_env(tmp_path)
        segment = tmp_path / "archive" / "events_20260101T000000.jsonl"
        segment.parent.mkdir(parents=True, exist_ok=True)
        with segment.open("wb") as fh:  # sparse: size matters, bytes do not
            fh.seek(EVENTS_ARCHIVE_SCAN_WARN_BYTES)
            fh.write(b"x")

        result = build_health_invariants(env)
        assert "HOT STORE GROWTH" in result
        assert "events chain" in result
        assert "never deleted" in result
        assert "Legacy segments retain inline delegated request bodies" in result
        assert "without shrinking existing history" in result
        assert segment.stat().st_size == EVENTS_ARCHIVE_SCAN_WARN_BYTES + 1

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
    # Knowledge and my rooms lead the changing block; the runtime sections follow, health first.
    assert dynamic_text.startswith("## Shared understanding")
    assert dynamic_text.index("## This room (Main)") < dynamic_text.index("## Health Invariants")
    assert dynamic_text.index("## Health Invariants") < dynamic_text.index("## Scratchpad")
    assert dynamic_text.index("## Health Invariants") < dynamic_text.index("## Drive state")


def test_health_invariants_come_first_in_a_consciousness_wake_context(tmp_path):
    """A wake-up is an ordinary Main turn: the same builder, the same section order."""
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory

    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    (repo_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (repo_dir / "docs").mkdir(parents=True, exist_ok=True)
    (drive_root / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
    (drive_root / "logs").mkdir(parents=True, exist_ok=True)
    (drive_root / "state").mkdir(parents=True, exist_ok=True)

    (repo_dir / "prompts" / "SYSTEM.md").write_text("System prompt", encoding="utf-8")
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

    class FakeEnv:
        def drive_path(self, p):
            return drive_root / p

        def repo_path(self, p):
            return repo_dir / p

        @property
        def repo_dir(self):
            return repo_dir

        @property
        def drive_root(self):
            return drive_root

    messages, _cap_info = build_llm_messages(
        env=FakeEnv(),
        memory=Memory(drive_root=drive_root, repo_dir=repo_dir),
        task={"id": "wake1", "type": "task", "text": "[Wake-up · heartbeat]", "_is_direct_chat": True,
              "metadata": {"initiator": "consciousness", "usage_category": "consciousness"}},
    )

    dynamic_text = messages[0]["content"][2]["text"]
    assert dynamic_text.startswith("## Shared understanding")
    assert "## This room" not in dynamic_text  # a wake has no current room
    assert dynamic_text.index("## Health Invariants") < dynamic_text.index("## Scratchpad")
    assert dynamic_text.index("## Health Invariants") < dynamic_text.index("## Drive state")


def test_project_room_keeps_an_older_own_directive_beside_sibling_traffic(tmp_path, monkeypatch):
    """Sibling traffic cannot hide an older own-room directive: the room's open rows are its own."""
    from ouroboros.projects_registry import create_project
    from tests._memory_view_context import blocks, section
    from tests.test_cache_optimization import _make_env_and_memory

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    env, memory = _make_env_and_memory(tmp_path)
    root = memory.drive_root
    own = create_project(root, "own", name="Own")
    sibling = create_project(root, "sibling", name="Sibling")
    own_chat, sibling_chat = int(own["chat_id"]), int(sibling["chat_id"])
    decisive = "CLAUDEXOR_ONLY_AND_THREE_LEVEL_NESTING"
    (root / "archive").mkdir(parents=True, exist_ok=True)
    (root / "archive" / "chat_20260820T010000.jsonl").write_text(
        json.dumps({"chat_id": own_chat, "direction": "in", "ts": "2026-08-20T01:00:00+00:00", "text": decisive}) + "\n",
        encoding="utf-8",
    )
    (root / "logs" / "chat.jsonl").write_text("".join(
        json.dumps({"chat_id": sibling_chat, "direction": "in", "ts": "2026-08-21T00:00:00+00:00",
                    "text": f"noise-{i}"}) + "\n" for i in range(4500)), encoding="utf-8")

    _a, _b, changing, _cap = blocks(env, memory, {"id": "own-task", "chat_id": own_chat})
    room = section(changing, f"## This room (Project Own [chat_id={own_chat}])")
    assert decisive in room
    assert "noise-" not in changing  # the sibling room is one line in ## Live rooms
    assert f"Project Sibling [chat_id={sibling_chat}]" in section(changing, "## Live rooms")
    assert str(root) not in room


def test_project_room_keeps_the_retention_proof_cross_room_directive_once(tmp_path):
    """The owner's words that started a Project stay visible once: in the open conversation of the
    room they were written in, else under ``Words that started this work``."""
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from tests._memory_view_context import blocks, section
    from tests.test_cache_optimization import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    root = memory.drive_root
    project = create_project(root, "cat-tower", name="Cat tower")
    project_chat = int(project["chat_id"])
    directive = "d" * 500 + " CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3"
    ref = build_owner_message_ref(chat_id=1, client_message_id="cat-origin", ts="2026-08-21T00:00:00Z", text=directive)
    bind_task_to_project(root, "cat-root", "cat-tower", project_chat, origin={"ref": ref, "text": directive})
    source_row = {**ref, "direction": "in", "text": directive}
    follow = {"chat_id": project_chat, "direction": "in", "ts": "2026-08-21T00:01:00Z", "text": "continue"}
    (root / "logs" / "chat.jsonl").write_text(json.dumps(source_row) + "\n" + json.dumps(follow) + "\n",
                                              encoding="utf-8")
    label = f"## This room (Project Cat tower [chat_id={project_chat}])"
    _a, _b, present, _cap = blocks(env, memory, {"id": "cat-root", "chat_id": project_chat})
    room = section(present, label)
    assert room.count("CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3") == 1  # an open row of this room
    assert "### Words that started this work" not in room

    origin = root / "memory" / "chronicle"
    assert origin.is_dir()  # the first capture activated the chronicle
    (root / "logs" / "chat.jsonl").write_text(json.dumps(follow) + "\n", encoding="utf-8")
    _a, _b, later, _cap = blocks(env, memory, {"id": "cat-root", "chat_id": project_chat})
    room = section(later, label)
    assert room.count("CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3") == 1  # retention-proof: the binding kept it
    assert "### Words that started this work (retention-proof)" in room


def test_chat_history_surfaces_malformed_gap_even_when_search_matches_nothing(tmp_path):
    from ouroboros.memory import Memory

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "chat.jsonl").write_bytes(
        b'{"direction":"in","text":"valid"}\n{"direction":"in","text":"decisive-tail"\n'
    )

    result = Memory(tmp_path).chat_history(count=20, search="absent-query")

    assert "no observed messages matching query" in result
    assert "completeness unknown" in result
    assert "jsonl_malformed" in result


def test_archive_only_chat_chain_is_complete_while_live_file_is_absent(tmp_path):
    from ouroboros.memory import Memory

    archive = tmp_path / "archive"
    archive.mkdir(parents=True)
    (archive / "chat_20260820T010000.jsonl").write_text(
        json.dumps({"direction": "in", "text": "archive-only"}) + "\n",
        encoding="utf-8",
    )

    entries, coverage = Memory(drive_root=tmp_path).read_chat_generations()

    assert [entry["text"] for entry in entries] == ["archive-only"]
    assert coverage["gaps"] == []


def test_runtime_section_carries_official_update_fact(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    # Patched WHERE IT IS USED: context.py binds the name at import, so patching the
    # defining module would leave this test asserting the real projection's own answer
    # and proving nothing about the injection.
    monkeypatch.setattr(
        "ouroboros.context.official_update_projection",
        lambda head: {"status": "update_available", "running": {"sha": head}, "letter": {"state": "ready"}},
    )
    section = build_runtime_section(env, {"id": "task-1", "type": "task"})
    payload = json.loads(section.split("\n\n", 1)[1])

    assert payload["official_update"]["status"] == "update_available"
    assert payload["official_update"]["letter"] == {"state": "ready"}
    assert payload["official_update"]["running"]["sha"] == payload["git_head"], "the fact reads THIS repo's HEAD"
