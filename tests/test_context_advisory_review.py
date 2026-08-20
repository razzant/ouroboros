"""How an advisory review status is presented inside the context.

Split verbatim out of ``tests/test_context.py`` by theme. This module owns the advisory
review status block the context carries and everything it must and must not claim about
a run.
"""

from __future__ import annotations






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
