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


def test_project_recent_chat_filters_archives_before_recent_bound(tmp_path, monkeypatch):
    """Sibling traffic cannot hide an older own-project directive before filtering."""
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    logs = tmp_path / "logs"
    archive = tmp_path / "archive"
    logs.mkdir(parents=True)
    archive.mkdir(parents=True)
    own = create_project(tmp_path, "own")
    sibling = create_project(tmp_path, "sibling")
    own_chat = int(own["chat_id"])
    sibling_chat = int(sibling["chat_id"])
    decisive = "CLAUDEXOR_ONLY_AND_THREE_LEVEL_NESTING"
    (archive / "chat_20260820T010000.jsonl").write_text(
        json.dumps({"chat_id": own_chat, "direction": "in", "text": decisive}) + "\n",
        encoding="utf-8",
    )
    (logs / "chat.jsonl").write_text(
        "\n".join(
            json.dumps({"chat_id": sibling_chat, "direction": "in", "text": f"noise-{i}"})
            for i in range(4500)
        ) + "\n",
        encoding="utf-8",
    )

    combined = "\n\n".join(build_recent_sections(
        Memory(drive_root=tmp_path), env=None, thread_chat_id=own_chat,
    ))

    assert decisive in combined
    assert "noise-4499" not in combined
    assert '\"omitted_matching_rows\": 0' in combined
    assert "chat_history(count, offset, search)" in combined
    assert str(tmp_path) not in combined


def test_project_context_keeps_retention_proof_cross_thread_cat_directive_once(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project, create_project

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    project = create_project(tmp_path, "cat-tower")
    project_chat = int(project["chat_id"])
    directive = "d" * 500 + " CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3"
    ref = build_owner_message_ref(
        chat_id=1, client_message_id="cat-origin", ts="2026-08-21T00:00:00Z", text=directive,
    )
    bind_task_to_project(
        tmp_path, "cat-root", "cat-tower", project_chat,
        origin={"ref": ref, "text": directive},
    )
    source_row = {**ref, "direction": "in", "text": directive}
    (logs / "chat.jsonl").write_text(
        json.dumps(source_row) + "\n"
        + json.dumps({"chat_id": project_chat, "direction": "in", "text": "continue"}) + "\n",
        encoding="utf-8",
    )
    present = "\n\n".join(build_recent_sections(
        Memory(drive_root=tmp_path), env=None, thread_chat_id=project_chat,
    ))
    assert present.count("CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3") == 1

    (logs / "chat.jsonl").write_text(
        json.dumps({"chat_id": project_chat, "direction": "in", "text": "continue"}) + "\n",
        encoding="utf-8",
    )

    combined = "\n\n".join(build_recent_sections(
        Memory(drive_root=tmp_path), env=None, thread_chat_id=project_chat,
    ))

    assert directive in combined
    assert combined.count("CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3") == 1
    assert "Project owner origins (retention-proof bindings)" in combined


def test_automatic_recent_context_reads_only_bounded_generation_suffix(tmp_path, monkeypatch):
    import pathlib

    from ouroboros.memory import Memory, _AUTOMATIC_CHAT_GENERATIONS

    logs, archive = tmp_path / "logs", tmp_path / "archive"
    logs.mkdir(parents=True)
    archive.mkdir()
    for index in range(10):
        (archive / f"chat_20260820T{index:02d}0000.jsonl").write_text(
            json.dumps({"direction": "in", "text": f"archive-{index}"}) + "\n",
            encoding="utf-8",
        )
    (logs / "chat.jsonl").write_text(
        json.dumps({"direction": "in", "text": "live-latest"}) + "\n",
        encoding="utf-8",
    )
    memory = Memory(tmp_path)
    reads = []
    original = memory._read_chat_generation

    def _counted(path, **kwargs):
        reads.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(memory, "_read_chat_generation", _counted)
    entries, coverage = memory.read_unconsolidated_chat({}, 20)

    assert len(reads) <= _AUTOMATIC_CHAT_GENERATIONS
    assert entries[-1]["text"] == "live-latest"
    assert coverage["omitted_matching_rows_unknown"] is True
    assert any(gap["kind"] == "unscanned_unconsolidated_generations" for gap in coverage["gaps"])
    assert [pathlib.Path(row["path"]).name for row in coverage["generations"]] == [
        "chat_20260820T080000.jsonl", "chat_20260820T090000.jsonl", "chat.jsonl",
    ]


def test_automatic_recent_context_materializes_a_bounded_row_suffix(tmp_path):
    from ouroboros.memory import Memory

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "chat.jsonl").write_text(
        "".join(
            json.dumps({"direction": "in", "text": f"row-{index}"}) + "\n"
            for index in range(20_000)
        ),
        encoding="utf-8",
    )

    entries, coverage = Memory(tmp_path).read_unconsolidated_chat({}, 1)

    assert [entry["text"] for entry in entries] == ["row-19999"]
    assert coverage["generations"][0]["rows"] <= 100
    assert coverage["omitted_matching_rows_unknown"] is True
    assert any(
        gap["kind"] in {"generation_prefix_unscanned", "generation_tail_rows_unscanned"}
        for gap in coverage["gaps"]
    )


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


def test_main_recent_chat_resumes_unconsolidated_archived_generation(tmp_path):
    from ouroboros.context import build_recent_sections
    from ouroboros.memory import Memory
    from ouroboros.utils import jsonl_generation_signature

    logs = tmp_path / "logs"
    archive = tmp_path / "archive"
    memory_dir = tmp_path / "memory"
    logs.mkdir(parents=True)
    archive.mkdir(parents=True)
    memory_dir.mkdir(parents=True)
    old = archive / "chat_20260820T010000.jsonl"
    old.write_text(
        "\n".join(json.dumps({"direction": "in", "text": f"old-{i}"}) for i in range(5)) + "\n",
        encoding="utf-8",
    )
    (logs / "chat.jsonl").write_text(
        json.dumps({"direction": "in", "text": "new-live"}) + "\n",
        encoding="utf-8",
    )
    (memory_dir / "dialogue_meta.json").write_text(json.dumps({
        "last_consolidated_offset": 3,
        "chat_log_signature": jsonl_generation_signature(old),
    }), encoding="utf-8")

    combined = "\n\n".join(build_recent_sections(Memory(drive_root=tmp_path), env=None))

    assert "old-0" not in combined
    assert "old-2" not in combined
    assert "old-3" in combined
    assert "old-4" in combined
    assert "new-live" in combined
    assert '\"gaps\": []' in combined


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


def test_missing_cursor_generation_hot_path_never_replays_full_archive(tmp_path, monkeypatch):
    from ouroboros.memory import Memory

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "chat.jsonl").write_text(
        json.dumps({"direction": "in", "text": "bounded-live"}) + "\n",
        encoding="utf-8",
    )
    memory = Memory(drive_root=tmp_path)

    def _forbidden_full_replay(**_kwargs):
        raise AssertionError("automatic gap recovery must not scan every archive generation")

    monkeypatch.setattr(memory, "read_chat_generations", _forbidden_full_replay)
    entries, coverage = memory.read_unconsolidated_chat({
        "last_consolidated_offset": 50,
        "chat_log_signature": {"first_line_sha256": "f" * 64, "size": 999},
    }, 20)

    assert [entry["text"] for entry in entries] == ["bounded-live"]
    assert coverage["omitted_matching_rows_unknown"] is True
    assert coverage["gaps"][0]["kind"] == "consolidation_cursor_generation_missing"
    assert coverage["reader"] == "chat_history(count, offset, search)"


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
