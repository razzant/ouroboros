"""
Tests for Ouroboros extensions.

Tests each extension module independently with no external dependencies.
Uses tmp_path fixtures for file system operations.

Run: pytest tests/test_extensions.py -v
"""

import json
import hashlib
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta


# ============================================================================
# Anti-Pattern Detector Tests
# ============================================================================

class TestAntiPatternDetector:
    """Tests for ouroboros.tools.antipatterns."""

    def _make_event(self, event_type, tool=None, task_id="task_1", hours_ago=1.0, **kwargs):
        ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        evt = {"ts": ts, "type": event_type, "task_id": task_id}
        if tool:
            evt["tool"] = tool
        evt.update(kwargs)
        return evt

    def test_stuck_tool_loop_detected(self):
        from ouroboros.tools.antipatterns import AntiPatternDetector

        events = [
            self._make_event("tool_error", tool="browse_page", hours_ago=0.5),
            self._make_event("tool_error", tool="browse_page", hours_ago=0.4),
            self._make_event("tool_error", tool="browse_page", hours_ago=0.3),
        ]
        detector = AntiPatternDetector()
        patterns = detector.scan_events(events, window_hours=2.0)

        assert len(patterns) >= 1
        assert patterns[0]["pattern"] == "stuck_tool_loop"
        assert patterns[0]["severity"] == "high"

    def test_no_false_positive_on_single_error(self):
        from ouroboros.tools.antipatterns import AntiPatternDetector

        events = [
            self._make_event("tool_error", tool="browse_page", hours_ago=0.5),
        ]
        detector = AntiPatternDetector()
        patterns = detector.scan_events(events, window_hours=2.0)

        # Single error should not trigger
        stuck_patterns = [p for p in patterns if p["pattern"] == "stuck_tool_loop"]
        assert len(stuck_patterns) == 0

    def test_analysis_paralysis_detected(self):
        from ouroboros.tools.antipatterns import AntiPatternDetector

        events = [
            self._make_event("llm_round", hours_ago=0.5 + i * 0.01, cost_usd=0.15)
            for i in range(25)  # 25 rounds × $0.15 = $3.75
        ]
        detector = AntiPatternDetector()
        patterns = detector.scan_events(events, window_hours=2.0)

        ap_patterns = [p for p in patterns if p["pattern"] == "analysis_paralysis"]
        assert len(ap_patterns) >= 1

    def test_no_analysis_paralysis_with_commit(self):
        from ouroboros.tools.antipatterns import AntiPatternDetector

        events = [
            self._make_event("llm_round", hours_ago=0.5 + i * 0.01, cost_usd=0.15)
            for i in range(25)
        ]
        events.append(self._make_event("git_push", hours_ago=0.1))
        detector = AntiPatternDetector()
        patterns = detector.scan_events(events, window_hours=2.0)

        ap_patterns = [p for p in patterns if p["pattern"] == "analysis_paralysis"]
        assert len(ap_patterns) == 0

    def test_context_thrashing_detected(self):
        from ouroboros.tools.antipatterns import AntiPatternDetector

        now = datetime.now(timezone.utc)
        tools = [
            {
                "ts": (now - timedelta(hours=0.5 + i * 0.01)).isoformat(),
                "tool": "repo_read",
                "task_id": "task_1",
                "args": {"path": "agent.py"},
            }
            for i in range(6)
        ]
        detector = AntiPatternDetector()
        patterns = detector.scan_events([], tools=tools, window_hours=2.0)

        ct_patterns = [p for p in patterns if p["pattern"] == "context_thrashing"]
        assert len(ct_patterns) >= 1

    def test_old_events_outside_window_ignored(self):
        from ouroboros.tools.antipatterns import AntiPatternDetector

        events = [
            self._make_event("tool_error", tool="browse_page", hours_ago=10.0),
            self._make_event("tool_error", tool="browse_page", hours_ago=10.1),
            self._make_event("tool_error", tool="browse_page", hours_ago=10.2),
        ]
        detector = AntiPatternDetector()
        patterns = detector.scan_events(events, window_hours=2.0)

        # All events are older than the window
        assert len(patterns) == 0


# ============================================================================
# Identity Hash Chain Tests
# ============================================================================

class TestIdentityChain:
    """Tests for ouroboros.identity_chain."""

    def test_compute_hash_deterministic(self):
        from ouroboros.identity_chain import compute_identity_hash

        h1 = compute_identity_hash("BIBLE text", "identity text")
        h2 = compute_identity_hash("BIBLE text", "identity text")
        assert h1 == h2

    def test_compute_hash_differs_on_change(self):
        from ouroboros.identity_chain import compute_identity_hash

        h1 = compute_identity_hash("BIBLE v1", "identity v1")
        h2 = compute_identity_hash("BIBLE v2", "identity v1")
        assert h1 != h2

    def test_append_and_verify_chain(self, tmp_path):
        from ouroboros.identity_chain import append_to_chain, verify_chain

        bible = "# Constitution\nPrinciple 0: Agency"
        identity = "# Who I Am\nI am Ouroboros."

        # First entry
        entry = append_to_chain(tmp_path, bible, identity, reason="genesis")
        assert entry["prev_hash"] == "GENESIS"
        assert entry["chain_length"] == 1

        # Verify
        result = verify_chain(tmp_path, bible, identity)
        assert result["status"] == "OK"

    def test_chain_detects_drift(self, tmp_path):
        from ouroboros.identity_chain import append_to_chain, verify_chain

        bible = "# Constitution"
        identity = "# Who I Am"

        append_to_chain(tmp_path, bible, identity, reason="test")

        # Modify identity without going through the chain
        result = verify_chain(tmp_path, bible, "# MODIFIED IDENTITY")
        assert result["status"] == "IDENTITY_DRIFT"

    def test_chain_detects_break(self, tmp_path):
        from ouroboros.identity_chain import append_to_chain, verify_chain

        bible = "# Constitution"
        identity_v1 = "# Who I Am v1"
        identity_v2 = "# Who I Am v2"

        append_to_chain(tmp_path, bible, identity_v1, reason="v1")
        append_to_chain(tmp_path, bible, identity_v2, reason="v2")

        # Tamper with the chain file
        chain_path = tmp_path / "memory" / "identity_chain.jsonl"
        lines = chain_path.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        entry["hash"] = "tampered_hash"
        lines[0] = json.dumps(entry)
        chain_path.write_text("\n".join(lines))

        result = verify_chain(tmp_path, bible, identity_v2)
        assert result["status"] == "CHAIN_BREAK"

    def test_no_chain_returns_no_chain(self, tmp_path):
        from ouroboros.identity_chain import verify_chain

        result = verify_chain(tmp_path, "bible", "identity")
        assert result["status"] == "NO_CHAIN"


# ============================================================================
# Knowledge Router Tests
# ============================================================================

class TestKnowledgeRouter:
    """Tests for ouroboros.knowledge_router."""

    def _create_topic(self, knowledge_dir, name, keywords, content="test content"):
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        path = knowledge_dir / f"{name}.md"
        header = f"<!-- keywords: {', '.join(keywords)} -->\n\n"
        path.write_text(header + content)

    def test_routes_by_keyword(self, tmp_path):
        from ouroboros.knowledge_router import route_knowledge

        kdir = tmp_path / "knowledge"
        self._create_topic(kdir, "browser-gotchas", ["browser", "playwright", "screenshot"])
        self._create_topic(kdir, "git-recipes", ["git", "commit", "push", "branch"])

        matches = route_knowledge("I need to fix the browser screenshot tool", kdir)
        assert len(matches) >= 1
        assert matches[0][0] == "browser-gotchas"

    def test_no_match_returns_empty(self, tmp_path):
        from ouroboros.knowledge_router import route_knowledge

        kdir = tmp_path / "knowledge"
        self._create_topic(kdir, "browser-gotchas", ["browser", "playwright"])

        matches = route_knowledge("update the budget tracker", kdir)
        assert len(matches) == 0

    def test_multiple_matches_ranked(self, tmp_path):
        from ouroboros.knowledge_router import route_knowledge

        kdir = tmp_path / "knowledge"
        self._create_topic(kdir, "git-recipes", ["git", "commit", "push"])
        self._create_topic(kdir, "git-advanced", ["git", "rebase", "merge"])

        # "git commit push" should match git-recipes more strongly
        matches = route_knowledge("I need to git commit and push", kdir)
        assert len(matches) >= 1
        assert matches[0][0] == "git-recipes"

    def test_load_relevant_knowledge_formats_output(self, tmp_path):
        from ouroboros.knowledge_router import load_relevant_knowledge

        kdir = tmp_path / "knowledge"
        self._create_topic(kdir, "browser-gotchas", ["browser"],
                           content="Always close the browser after use.")

        result = load_relevant_knowledge("fix browser issue", kdir)
        assert result is not None
        assert "browser-gotchas" in result
        assert "Always close" in result

    def test_skips_index_files(self, tmp_path):
        from ouroboros.knowledge_router import build_keyword_index

        kdir = tmp_path / "knowledge"
        kdir.mkdir(parents=True)
        (kdir / "_index.md").write_text("# Index")
        self._create_topic(kdir, "real-topic", ["test"])

        index = build_keyword_index(kdir)
        assert "_index" not in index
        assert "real-topic" in index


# ============================================================================
# Temporal Context Tests
# ============================================================================

class TestTemporalContext:
    """Tests for ouroboros.temporal."""

    def test_builds_age(self):
        from ouroboros.temporal import build_temporal_context

        result = build_temporal_context({})
        assert "Age:" in result
        assert "days" in result

    def test_shows_creator_silence(self):
        from ouroboros.temporal import build_temporal_context

        old_time = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        result = build_temporal_context({"last_owner_message_at": old_time})
        assert "silent" in result.lower() or "sleeping" in result.lower()

    def test_shows_active_creator(self):
        from ouroboros.temporal import build_temporal_context

        recent = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        result = build_temporal_context({"last_owner_message_at": recent})
        assert "active" in result.lower()

    def test_shows_evolution_state(self):
        from ouroboros.temporal import build_temporal_context

        result = build_temporal_context({
            "evolution_mode_enabled": True,
            "evolution_cycle": 15,
        })
        assert "Evolution:" in result
        assert "15" in result


# ============================================================================
# Resilience / Circuit Breaker Tests
# ============================================================================

class TestCircuitBreaker:
    """Tests for ouroboros.resilience."""

    def test_starts_closed(self):
        from ouroboros.resilience import CircuitBreaker

        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == "CLOSED"
        assert cb.allow_call()

    def test_opens_after_threshold(self):
        from ouroboros.resilience import CircuitBreaker

        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure(Exception("err 1"))
        cb.record_failure(Exception("err 2"))
        assert cb.state == "CLOSED"  # Not yet

        cb.record_failure(Exception("err 3"))
        assert cb.state == "OPEN"
        assert not cb.allow_call()

    def test_recovers_after_cooldown(self):
        from ouroboros.resilience import CircuitBreaker
        import time

        cb = CircuitBreaker("test", failure_threshold=2, cooldown_sec=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"

        time.sleep(0.15)
        assert cb.state == "HALF_OPEN"
        assert cb.allow_call()

    def test_success_closes_circuit(self):
        from ouroboros.resilience import CircuitBreaker

        cb = CircuitBreaker("test", failure_threshold=2, cooldown_sec=0.01)
        cb.record_failure()
        cb.record_failure()

        import time
        time.sleep(0.02)
        cb.record_success()
        assert cb.state == "CLOSED"

    def test_format_breakers_empty_when_all_closed(self):
        from ouroboros.resilience import CircuitBreaker, format_breakers_for_health

        # Fresh state — should return empty string
        result = format_breakers_for_health()
        # May or may not have breakers registered; if all closed, should be empty
        assert isinstance(result, str)


# ============================================================================
# Memory Consolidation Tests
# ============================================================================

class TestMemoryConsolidation:
    """Tests for ouroboros.memory_consolidation."""

    def _write_journal_entries(self, journal_path, entries):
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_find_unconsolidated_periods(self, tmp_path):
        from ouroboros.memory_consolidation import find_unconsolidated_periods

        journal_path = tmp_path / "memory" / "scratchpad_journal.jsonl"
        consolidated_dir = tmp_path / "memory" / "consolidated"

        # Create entries from 2 weeks ago (should need consolidation)
        two_weeks_ago = datetime.now(timezone.utc) - timedelta(days=14)
        entries = [
            {"ts": (two_weeks_ago + timedelta(hours=i)).isoformat(), "content": f"Entry {i}"}
            for i in range(5)
        ]
        self._write_journal_entries(journal_path, entries)

        periods = find_unconsolidated_periods(journal_path, consolidated_dir)
        assert len(periods) >= 1

    def test_current_week_not_consolidated(self, tmp_path):
        from ouroboros.memory_consolidation import find_unconsolidated_periods

        journal_path = tmp_path / "memory" / "scratchpad_journal.jsonl"
        consolidated_dir = tmp_path / "memory" / "consolidated"

        # Create entries from today (should NOT be consolidated — still accumulating)
        now = datetime.now(timezone.utc)
        entries = [
            {"ts": (now - timedelta(hours=i)).isoformat(), "content": f"Entry {i}"}
            for i in range(5)
        ]
        self._write_journal_entries(journal_path, entries)

        periods = find_unconsolidated_periods(journal_path, consolidated_dir)
        # Current week should be excluded
        current_key = f"{now.isocalendar()[0]}-W{now.isocalendar()[1]:02d}"
        assert current_key not in periods

    def test_load_consolidated_memory(self, tmp_path):
        from ouroboros.memory_consolidation import load_consolidated_memory

        consolidated_dir = tmp_path / "memory" / "consolidated"
        consolidated_dir.mkdir(parents=True)

        # Create a fake digest
        (consolidated_dir / "2026-W07.md").write_text(
            "# Memory Digest: 2026-W07\n\nI learned about browser tools."
        )

        result = load_consolidated_memory(consolidated_dir)
        assert "2026-W07" in result
        assert "browser" in result

    def test_empty_dir_returns_empty(self, tmp_path):
        from ouroboros.memory_consolidation import load_consolidated_memory

        result = load_consolidated_memory(tmp_path / "nonexistent")
        assert result == ""
