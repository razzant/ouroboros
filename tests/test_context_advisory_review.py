"""How the review status is presented inside the context.

Split verbatim out of ``tests/test_context.py`` by theme. This module owns the review
status block the context carries and everything it must and must not claim; the retired
advisory lane's archived rows claim nothing there (``tests/test_advisory_run_archive.py``).
"""

from __future__ import annotations






class TestAdvisoryReviewStatusInContext:
    """Tests that the review status reaches the LLM context, and archived runs never do."""

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

    def test_archived_advisory_rows_alone_render_no_review_history(self, tmp_path):
        """The retired lane's rows are an archive: alone they render as an empty history."""
        from ouroboros.review_state import AdvisoryReviewState, format_status_section, load_state, save_state
        state = AdvisoryReviewState(advisory_runs=[{
            "snapshot_hash": "abc123", "commit_message": "test commit", "status": "fresh",
            "ts": "2026-01-01T00:00:00",
            "items": [{"item": "bible_compliance", "verdict": "PASS", "severity": "critical", "reason": "ok"}],
        }])
        save_state(tmp_path, state)

        section = format_status_section(load_state(tmp_path))
        assert section == format_status_section(AdvisoryReviewState())
        assert "FRESH" not in section and "abc123" not in section

    def test_advisory_status_empty_when_no_runs(self, tmp_path):
        """format_status_section names the empty history when state is empty."""
        from ouroboros.review_state import AdvisoryReviewState, format_status_section
        state = AdvisoryReviewState()
        section = format_status_section(state)
        assert "No reviewed attempts recorded yet." in section

    def test_review_continuity_context_surfaces_live_gate_and_continuation(self, tmp_path):
        from ouroboros.agent_task_pipeline import build_review_context
        from ouroboros.context import build_llm_messages
        from ouroboros.memory import Memory
        from ouroboros.review_state import (
            AdvisoryReviewState,
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
        state.advisory_runs.append({  # an archived row of the retired lane
            "snapshot_hash": snapshot_hash, "commit_message": "test commit", "status": "stale",
            "ts": "2026-04-07T09:59:00+00:00", "repo_key": repo_key, "bypass_reason": "manual audit override",
        })
        from ouroboros import review_ledger

        review_ledger.write_record(tmp_path, {  # the preflight look the marker below stales
            "record_id": review_ledger.new_record_id(), "state": "settled", "surface": "preflight",
            "ts": "2026-04-07T09:58:00+00:00", "verdict": {"aggregate": "PASS"},
            "subject": {"root": str(tmp_path / "repo")}, "rows": []})
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
                review_record_id="rv-old",
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
        dynamic_text = messages[0]["content"][-1]["text"]

        assert "## Review Continuity" in dynamic_text
        # Neither a readiness axis (decision 3A) nor a debt or retry anchor is projected beside the facts.
        for gone in ("retry_anchor", "commit_readiness_debt", "Commit-readiness debt"):
            assert gone not in dynamic_text, gone
        assert "stale_marker=2026-04-07T10:00:00" in dynamic_text
        for retired in ("Advisory readiness", "repo_commit_ready", "advisory_status=", "snapshot_hash=",
                        "bypass_reason=", "no advisory run matches", "manual audit override"):
            assert retired not in dynamic_text, retired
        assert "### Open review continuations" in dynamic_text
        assert "critical_finding=tests_affected: Fix the failing test before commit" in dynamic_text
        assert "review_record_id=rv-old" in dynamic_text  # the record holds every seat's answer as written
        assert "### Historical review ledger" in dynamic_text
        assert "## Scratchpad" in dynamic_text
        assert dynamic_text.index("## Scratchpad") < dynamic_text.index("## Drive state")
        assert dynamic_text.index("## Runtime context") < dynamic_text.index("## Review Continuity")

    def test_review_continuity_shows_earlier_obligations_until_a_commit_lands_never_as_current(self, tmp_path):
        """An open obligation this checkout recorded before obligations retired shows once in each
        Review Continuity view (the fallback section's too), a foreign checkout's never, beside the continuation's
        record address; no obligation count, debt or retry anchor is projected; a commit that lands here after it
        ends the block."""
        import json

        from ouroboros.agent_task_pipeline import build_review_context
        from ouroboros.context_sections import review_status_host_section
        from ouroboros.review_state import CommitAttemptRecord, load_state, make_repo_key, update_state
        from ouroboros.task_continuation import ReviewContinuation, save_review_continuation

        env = self._make_env(tmp_path)
        (tmp_path / "repo" / ".git").mkdir(parents=True, exist_ok=True)
        repo_key = make_repo_key(tmp_path / "repo")
        recorded = "2026-04-07T10:00:00+00:00"

        def fallback():  # the section a context shows when the builder above yields nothing
            return review_status_host_section(load_state(tmp_path), tmp_path / "repo", tmp_path)
        mine = {"obligation_id": "obl-0003", "item": "tests_affected", "severity": "critical",
                "reason": "Coverage still missing", "source_attempt_ts": recorded, "source_attempt_msg": "blocked",
                "status": "still_open", "resolved_by": "", "repo_key": repo_key,
                "fingerprint": "finding:tests_affected:abc", "created_ts": recorded, "updated_ts": recorded}
        foreign = {**mine, "obligation_id": "obl-0004", "item": "foreign_issue", "repo_key": str(tmp_path / "other")}
        debt = {"debt_id": "crd-0001", "category": "readiness_warning", "summary": "Review was blocked",
                "status": "queued", "repo_key": repo_key, "fingerprint": "readiness_warning:attempt:abc"}
        (tmp_path / "state" / "advisory_review.json").write_text(json.dumps({  # what a body with Blocking left
            "state_version": 3, "attempts": [], "open_obligations": [mine, foreign], "next_obligation_seq": 5,
            "commit_readiness_debts": [debt], "next_commit_readiness_debt_seq": 2}), encoding="utf-8")
        save_review_continuation(tmp_path, ReviewContinuation(
            task_id="task-old", source="blocked_review", stage="blocking_review", repo_key=repo_key,
            tool_name="commit_reviewed", attempt=1, block_reason="critical_findings", review_record_id="rv-0001",
            critical_findings=[{"item": "tests_affected", "reason": "Fix the failing test", "severity": "critical",
                                "verdict": "FAIL"}]))

        section = build_review_context(env)
        for view in (section.full_text, section.operative_view):
            assert view.count("- earlier_open_obligations=1 (recorded before obligations retired") == 1
            assert view.count("[obl-0003] tests_affected: Coverage still missing") == 1
            assert "review_record_id=rv-0001" in view
            for gone in ("obl-0004", "foreign_issue", "crd-0001", "commit_readiness_debt", "retry_anchor"):
                assert gone not in view, gone
            assert "open_obligations=" not in view.replace("earlier_open_obligations=", "")
        for view in (fallback().full_text, fallback().operative_view):
            assert view.count("[obl-0003] tests_affected: Coverage still missing") == 1 and "obl-0004" not in view

        update_state(tmp_path, lambda state: state.record_attempt(CommitAttemptRecord(
            ts="2026-04-08T10:00:00+00:00", commit_message="landed", status="succeeded", repo_key=repo_key,
            task_id="task-new", attempt=1)))
        landed = build_review_context(env)
        for view in (landed.full_text, landed.operative_view):
            assert "earlier_open_obligations" not in view and "obl-0003" not in view
            assert "review_record_id=rv-0001" in view  # the continuation follows its own rule
        for view in (fallback().full_text, fallback().operative_view):
            assert "obl-0003" not in view
