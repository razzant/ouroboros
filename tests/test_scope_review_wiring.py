"""How the scope review is wired into the surrounding stack.

Split by theme out of the original ``tests/test_scope_review.py`` giant. This
module owns the module surface and its neighbours: the scope-review module
structure, workspace-root refusal, review_state path-aware freshness, the
enriched triad, git.py wiring, shared LLM routing and the advisory schema.
"""

import inspect
import pathlib
import subprocess

import pytest

from tests._scope_review_shared import _get_module

def test_review_thoroughness_is_count_free_and_evidence_bound():
    helpers = _get_module("ouroboros.tools.review_helpers")
    block = helpers.REVIEW_THOROUGHNESS_BLOCK

    assert "5 bugs" not in block
    assert "zero, one, or many findings are all valid" in block
    assert "Never invent a finding to increase the count" in block


def test_scope_review_uses_active_subject_and_system_governance(tmp_path, monkeypatch):
    mod = _get_module("ouroboros.tools.scope_review")
    registry = _get_module("ouroboros.tools.registry")
    governance = tmp_path / "system"
    subject = tmp_path / "subject"
    drive = tmp_path / "data"
    governance.mkdir()
    subject.mkdir()
    drive.mkdir()
    captured = {}

    def fake_build(repo_dir, _message, **kwargs):
        captured["subject"] = pathlib.Path(repo_dir)
        captured["governance"] = pathlib.Path(kwargs["context"].governance_repo_dir)
        return None, mod._TouchedContextStatus(status="empty")

    monkeypatch.setattr(mod, "_build_scope_prompt", fake_build)
    ctx = registry.ToolContext(
        repo_dir=governance,
        system_repo_dir=governance,
        workspace_root=subject,
        workspace_mode="external",
        drive_root=drive,
    )

    mod.run_scope_review(ctx, "review external subject", scope_model="test-scope")

    assert captured == {
        "subject": subject.resolve(),
        "governance": governance.resolve(),
    }


def test_scope_review_refuses_ambiguous_workspace_root(tmp_path):
    mod = _get_module("ouroboros.tools.scope_review")
    registry = _get_module("ouroboros.tools.registry")
    system = tmp_path / "system"
    subject = tmp_path / "subject"
    drive = tmp_path / "data"
    system.mkdir()
    subject.mkdir()
    drive.mkdir()
    ctx = registry.ToolContext(
        repo_dir=system,
        system_repo_dir=system,
        workspace_root=subject,
        workspace_mode="",
        drive_root=drive,
    )

    result = mod.run_scope_review(ctx, "must not inspect the wrong repo")

    assert result.blocked is True
    assert result.status == "error"
    assert "workspace_root is set without workspace_mode" in result.block_message


def test_managed_resolver_enables_binary_metadata_context(tmp_path, monkeypatch):
    mod = _get_module("ouroboros.tools.scope_review")
    registry = _get_module("ouroboros.tools.registry")
    registry_guards = _get_module("ouroboros.tools.registry_guards")
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    captured = {}

    def fake_build(_repo_dir, _message, **kwargs):
        captured["represent_binary"] = kwargs["context"].represent_binary
        return None, mod._TouchedContextStatus(status="empty")

    monkeypatch.setattr(mod, "_build_scope_prompt", fake_build)
    monkeypatch.setattr(
        registry_guards, "_authorized_managed_update_resolver", lambda _ctx: True
    )
    monkeypatch.setattr(
        registry, "_authorized_managed_update_resolver", lambda _ctx: True
    )
    ctx = registry.ToolContext(repo_dir=repo, drive_root=drive, task_id="resolver")

    result = mod.run_scope_review(ctx, "review assisted update", scope_model="test")

    assert result.blocked is True
    assert captured == {"represent_binary": True}

class TestScopeReviewModule:
    # test_scope_review_imports removed in v5.15.x — pure callable-existence
    # check. The fail-closed test below already imports the module, and the
    # behavioral integration tests exercise run_scope_review end-to-end.

    def test_scope_review_fail_closed_design(self):
        """run_scope_review must be fail-closed: errors return blocking strings."""
        mod = _get_module("ouroboros.tools.scope_review")
        source = inspect.getsource(mod.run_scope_review)
        assert "SCOPE_REVIEW_BLOCKED" in source
        assert "fail" in source.lower() or "block" in source.lower()

    def test_scope_review_default_is_terra(self):
        mod = _get_module("ouroboros.tools.scope_review")
        assert "gpt-5.6-terra" in mod._SCOPE_MODEL_DEFAULT
        # Verify the getter returns the shipped default when no override env var is set
        import os
        if not os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL"):
            assert "gpt-5.6-terra" in mod._get_scope_model()
        # else: env override is active — default check not applicable in this env

    def test_scope_review_model_configurable_via_env(self):
        """OUROBOROS_SCOPE_REVIEW_MODEL env overrides the default."""
        mod = _get_module("ouroboros.tools.scope_review")
        import os
        old = os.environ.get("OUROBOROS_SCOPE_REVIEW_MODEL")
        old_plural = os.environ.get("OUROBOROS_SCOPE_REVIEW_MODELS")
        try:
            os.environ.pop("OUROBOROS_SCOPE_REVIEW_MODELS", None)
            os.environ["OUROBOROS_SCOPE_REVIEW_MODEL"] = "google/gemini-2.5-pro"
            assert mod._get_scope_model() == "google/gemini-2.5-pro"
        finally:
            if old is None:
                os.environ.pop("OUROBOROS_SCOPE_REVIEW_MODEL", None)
            else:
                os.environ["OUROBOROS_SCOPE_REVIEW_MODEL"] = old
            if old_plural is None:
                os.environ.pop("OUROBOROS_SCOPE_REVIEW_MODELS", None)
            else:
                os.environ["OUROBOROS_SCOPE_REVIEW_MODELS"] = old_plural

    def test_scope_review_effort_configurable(self):
        """OUROBOROS_EFFORT_SCOPE_REVIEW should resolve via resolve_effort."""
        from ouroboros.config import resolve_effort
        import os
        old = os.environ.get("OUROBOROS_EFFORT_SCOPE_REVIEW")
        try:
            os.environ["OUROBOROS_EFFORT_SCOPE_REVIEW"] = "low"
            assert resolve_effort("scope_review") == "low"
            assert resolve_effort("scope-review") == "low"
        finally:
            if old is None:
                os.environ.pop("OUROBOROS_EFFORT_SCOPE_REVIEW", None)
            else:
                os.environ["OUROBOROS_EFFORT_SCOPE_REVIEW"] = old

    def test_scope_prompt_includes_scope_checklist(self):
        """_build_scope_prompt must load the scope checklist, not the repo checklist."""
        mod = _get_module("ouroboros.tools.scope_review")
        source = inspect.getsource(mod._build_scope_prompt)
        assert "Intent / Scope Review Checklist" in source

    def test_scope_prompt_includes_generated_scope_atlas(self):
        # scope_review now uses the bounded generated Atlas instead of the legacy full pack.
        # The call is in _gather_scope_packs which _build_scope_prompt delegates to.
        mod = _get_module("ouroboros.tools.scope_review")
        source = inspect.getsource(mod._gather_scope_packs)
        assert "compile_review_context_atlas" in source
        assert "ReviewContextAtlasRequest" in source
        assert "fixed_prompt_tokens" in source

    def test_scope_prompt_fails_closed_on_atlas_inventory_error(self, tmp_path, monkeypatch):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8"
        )
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "a.py").write_text("aaa", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@o", "-c", "user.name=T", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "a.py").write_text("bbb", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        scope_pack = _get_module("ouroboros.tools.scope_review_pack")
        monkeypatch.setattr(
            scope_pack,
            "compile_review_context_atlas",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("inventory failed")),
        )
        with pytest.raises(RuntimeError, match="inventory failed"):
            mod._build_scope_prompt(tmp_path, "test msg")

    def test_scope_prompt_keeps_literal_atlas_placeholder_in_touched_content(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / "CHECKLISTS.md").write_text(
            "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8"
        )
        (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
        (tmp_path / "a.py").write_text("aaa", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "-c", "user.email=t@o", "-c", "user.name=T", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True,
        )
        (tmp_path / "a.py").write_text("print('__GENERATED_SCOPE_ATLAS_PENDING__')\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

        mod = _get_module("ouroboros.tools.scope_review")
        prompt, status = mod._build_scope_prompt(tmp_path, "test msg")
        assert status is None
        current_section = prompt[prompt.index("## Current touched files"):prompt.index("## Wider repository context")]
        assert "__GENERATED_SCOPE_ATLAS_PENDING__" in current_section

# ---------------------------------------------------------------------------
# review_state path-aware freshness
# ---------------------------------------------------------------------------

class TestPathAwareFreshness:
    def test_snapshot_hash_stable_without_message(self, tmp_path):
        """Snapshot hash should NOT change when only commit_message changes."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        rs = _get_module("ouroboros.review_state")
        h1 = rs.compute_snapshot_hash(tmp_path, "message A")
        h2 = rs.compute_snapshot_hash(tmp_path, "message B")
        # Hash now based on code only — should be SAME for different messages
        assert h1 == h2

    def test_snapshot_hash_changes_with_file_content(self, tmp_path):
        """Snapshot hash must change when file content changes."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "file.py").write_text("v1", encoding="utf-8")
        subprocess.run(["git", "add", "file.py"], cwd=str(tmp_path), capture_output=True)
        rs = _get_module("ouroboros.review_state")
        h1 = rs.compute_snapshot_hash(tmp_path, "msg")
        # Modify file
        (tmp_path / "file.py").write_text("v2", encoding="utf-8")
        h2 = rs.compute_snapshot_hash(tmp_path, "msg")
        assert h1 != h2

    def test_path_scoped_hash(self, tmp_path):
        """When paths= is provided, only those files affect the hash."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "a.py").write_text("aaa", encoding="utf-8")
        (tmp_path / "b.py").write_text("bbb", encoding="utf-8")
        rs = _get_module("ouroboros.review_state")
        h_a = rs.compute_snapshot_hash(tmp_path, paths=["a.py"])
        h_b = rs.compute_snapshot_hash(tmp_path, paths=["b.py"])
        assert h_a != h_b

    def test_stale_lifecycle(self):
        """add_run marks previous non-matching fresh runs as stale."""
        rs = _get_module("ouroboros.review_state")
        state = rs.AdvisoryReviewState()
        run1 = rs.AdvisoryRunRecord(
            snapshot_hash="hash1", commit_message="m1",
            status="fresh", ts="2026-01-01T00:00:00",
        )
        state.add_run(run1)
        assert state.advisory_runs[0].status == "fresh"

        run2 = rs.AdvisoryRunRecord(
            snapshot_hash="hash2", commit_message="m2",
            status="fresh", ts="2026-01-01T01:00:00",
        )
        state.add_run(run2)
        assert state.advisory_runs[0].status == "stale"  # hash1 became stale
        assert state.advisory_runs[1].status == "fresh"   # hash2 is fresh

# ---------------------------------------------------------------------------
# Triad review enrichment
# ---------------------------------------------------------------------------

class TestTriadReviewEnriched:
    def test_triad_prompt_has_touched_files_placeholder(self):
        """The dynamic review prompt template must include current_files_section."""
        mod = _get_module("ouroboros.tools.review")
        assert "{current_files_section}" in mod._REVIEW_PROMPT_TEMPLATE_DYNAMIC

    def test_triad_prompt_has_goal_section(self):
        """The dynamic review prompt template must include goal_section (the
        per-commit tail; the stable prefix carries the cache marker)."""
        mod = _get_module("ouroboros.tools.review")
        assert "{goal_section}" in mod._REVIEW_PROMPT_TEMPLATE_DYNAMIC
        assert "{goal_section}" not in mod._REVIEW_PROMPT_TEMPLATE_STABLE

    def test_run_unified_review_accepts_goal_scope(self):
        """_run_unified_review must accept goal and scope keyword args."""
        mod = _get_module("ouroboros.tools.review")
        sig = inspect.signature(mod._run_unified_review)
        assert "goal" in sig.parameters
        assert "scope" in sig.parameters

# ---------------------------------------------------------------------------
# git.py wiring
# ---------------------------------------------------------------------------

class TestGitWiring:
    def test_repo_commit_schema_has_goal_scope(self):
        git = _get_module("ouroboros.tools.git")
        tools = git.get_tools()
        commit = next(t for t in tools if t.name == "commit_reviewed")
        props = commit.schema["parameters"]["properties"]
        assert "goal" in props
        assert "scope" in props

    def test_repo_commit_push_accepts_goal_scope(self):
        git = _get_module("ouroboros.tools.git")
        sig = inspect.signature(git._repo_commit_push)
        assert "goal" in sig.parameters
        assert "scope" in sig.parameters

    def test_scope_review_wired_in_commit(self):
        """The shared reviewed stage must call the parallel review helper."""
        git = _get_module("ouroboros.tools.git")
        source = inspect.getsource(git._run_reviewed_stage_cycle)
        assert "_run_parallel_review" in source
        # The parallel helper must contain both triad and scope review
        parallel_source = inspect.getsource(git._run_parallel_review)
        assert "run_scope_review" in parallel_source
        assert "_run_unified_review" in parallel_source
        # ThreadPoolExecutor must be used for parallel execution
        assert "ThreadPoolExecutor" in parallel_source

    def test_repo_commit_not_bypass_scope(self):
        """repo_commit must reach scope review via the shared stage helper."""
        git = _get_module("ouroboros.tools.git")
        source = inspect.getsource(git._repo_commit_push)
        assert "_run_reviewed_stage_cycle" in source
        shared_source = inspect.getsource(git._run_reviewed_stage_cycle)
        assert "_check_advisory_freshness" in shared_source
        assert "_run_parallel_review" in shared_source
        parallel_source = inspect.getsource(git._run_parallel_review)
        assert "run_scope_review" in parallel_source
        assert "ThreadPoolExecutor" in parallel_source

    def test_parallel_execution_both_always_run(self):
        """Both triad and scope futures are always submitted regardless of each other's result."""
        git = _get_module("ouroboros.tools.git")
        source = inspect.getsource(git._run_parallel_review)
        # Both submissions must be present before any result() call
        submit_triad = source.find("triad_fut = pool.submit")
        submit_scope = source.find("scope_fut = pool.submit")
        result_triad = source.find("triad_fut.result()")
        result_scope = source.find("scope_fut.result()")
        # Both must be submitted, and submissions must precede result() calls
        assert submit_triad > 0
        assert submit_scope > 0
        assert result_triad > 0
        assert result_scope > 0
        # Both submitted before any result() is collected
        assert submit_triad < result_triad
        assert submit_scope < result_scope

    def test_aggregated_verdict_both_blockers_shown(self):
        """When both triad and scope block, both messages must appear in combined output."""
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        triad_error = "⚠️ REVIEW_BLOCKED: triad finding"
        scope_blocked = scope_mod.ScopeReviewResult(
            blocked=True,
            block_message="⚠️ SCOPE_REVIEW_BLOCKED: scope finding",
            critical_findings=[{"verdict": "FAIL", "item": "intent_alignment",
                                "severity": "critical", "reason": "scope blocked", "model": "test"}],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                triad_error, scope_blocked, "critical_findings", [], ctx,
                "test commit", 0.0, ctx.repo_dir)
        assert blocked
        assert "triad finding" in combined_msg
        assert "scope finding" in combined_msg
        assert "Both triad review AND scope review" in combined_msg
        assert len(findings) == 1

    def test_triad_advisory_included_when_scope_blocks(self):
        """When triad passes but has advisory findings and scope blocks, all findings appear."""
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        scope_blocked = scope_mod.ScopeReviewResult(
            blocked=True,
            block_message="⚠️ SCOPE_REVIEW_BLOCKED: scope critical finding",
            critical_findings=[{"verdict": "FAIL", "item": "intent_alignment",
                                "severity": "critical", "reason": "scope blocked", "model": "test"}],
        )
        triad_advisory = [{"item": "context_building", "reason": "advisory note"}]
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                None, scope_blocked, "scope_blocked", triad_advisory, ctx,
                "test commit", 0.0, ctx.repo_dir)
        assert blocked
        assert "scope critical finding" in combined_msg
        assert "advisory note" in combined_msg
        assert len(findings) == 1

    def test_advisory_mode_scope_criticals_not_in_blocking_findings(self):
        """Advisory-mode scope critical findings must NOT be added to _combined_findings."""
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        # Triad blocks; scope does NOT block but has critical findings (advisory enforcement)
        triad_error = "⚠️ REVIEW_BLOCKED: triad issue"
        scope_advisory_crit = scope_mod.ScopeReviewResult(
            blocked=False,  # advisory mode — not blocked
            block_message="",
            critical_findings=[{"verdict": "FAIL", "item": "intent_alignment",
                                "severity": "critical", "reason": "advisory-only scope note", "model": "test"}],
            advisory_findings=[],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                triad_error, scope_advisory_crit, "critical_findings", [], ctx,
                "test commit", 0.0, ctx.repo_dir)
        assert blocked
        # Advisory-mode scope criticals must NOT appear in durable blocking findings
        assert all(f.get("item") != "intent_alignment" for f in findings), \
            "Advisory-mode scope criticals must not be recorded as blocking findings"
        # But should appear in scope_advisory_items for visibility
        assert any(
            (isinstance(item, dict) and item.get("item") == "intent_alignment")
            or (isinstance(item, str) and "intent_alignment" in item)
            for item in scope_adv
        )

    def test_scope_advisory_visible_on_successful_commit(self):
        """Non-blocking scope advisory findings must be returned even when commit is not blocked."""
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        # Scope passes (not blocked) but has advisory findings
        scope_advisory = scope_mod.ScopeReviewResult(
            blocked=False,
            block_message="",
            critical_findings=[],
            advisory_findings=[{"verdict": "PASS", "item": "architecture_fit",
                                "severity": "advisory", "reason": "minor concern", "model": "test"}],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                None, scope_advisory, "", [], ctx, "test commit", 0.0, ctx.repo_dir)
        # Should NOT block
        assert not blocked
        assert combined_msg is None
        # But scope advisory items must be returned for caller to surface
        assert len(scope_adv) > 0
        assert any(
            (isinstance(item, dict) and item.get("item") == "architecture_fit")
            or (isinstance(item, str) and "architecture_fit" in item)
            for item in scope_adv
        )

    @pytest.mark.parametrize("crit_item", sorted(_get_module("ouroboros.tools.scope_review")._SCOPE_REQUIRED_ITEMS))
    def test_aggregation_does_not_block_on_advisory_scope_criticals(self, crit_item):
        """NW-2 guardrail (aggregation seam): a 58a52c4-class hardcode could be
        re-introduced downstream in aggregate_review_verdict instead of in
        scope_review.py. With no triad error and a non-blocked scope result that
        merely CARRIES a critical finding (advisory pass-through), the aggregator
        must NOT flip to blocked for ANY item id.
        """
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        scope_advisory_crit = scope_mod.ScopeReviewResult(
            blocked=False,
            block_message="",
            critical_findings=[{"verdict": "FAIL", "item": crit_item,
                                "severity": "critical", "reason": "advisory-only scope note", "model": "test"}],
            advisory_findings=[],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                None, scope_advisory_crit, "", [], ctx, "test commit", 0.0, ctx.repo_dir)
        assert not blocked, (
            f"aggregation must NOT block on an advisory-pass-through scope critical "
            f"for item {crit_item!r}; a per-item always-block hardcode would fail here"
        )
        assert combined_msg is None

    def test_scope_review_skipped_surfaces_through_aggregation_path(self):
        """Budget-skip advisories must survive aggregation and caller-side surfacing."""
        import types
        import unittest.mock as mock
        scope_mod = _get_module("ouroboros.tools.scope_review")
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        scope_advisory = scope_mod.ScopeReviewResult(
            blocked=False,
            block_message="",
            critical_findings=[],
            advisory_findings=[{
                "verdict": "FAIL",
                "item": "scope_review_skipped",
                "severity": "advisory",
                "reason": "⚠️ SCOPE_REVIEW_SKIPPED: Full scope-review prompt exceeds budget.",
                "model": "scope_reviewer",
            }],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            blocked, combined_msg, block_reason, findings, scope_adv = pr_mod.aggregate_review_verdict(
                None, scope_advisory, "", [], ctx, "test commit", 0.0, ctx.repo_dir)

        if scope_adv:
            ctx._review_advisory.extend(scope_adv)

        assert not blocked
        assert combined_msg is None
        assert findings == []
        assert any(
            (isinstance(item, dict) and item.get("item") == "scope_review_skipped")
            or (isinstance(item, str) and "scope_review_skipped" in item)
            for item in scope_adv
        )
        assert any(
            (isinstance(item, dict) and item.get("item") == "scope_review_skipped")
            or (isinstance(item, str) and "scope_review_skipped" in item)
            for item in ctx._review_advisory
        )

    def test_triad_crash_resets_stale_findings(self):
        """If triad crashes, stale ctx findings from prior attempt must not bleed into current run."""
        import types
        import unittest.mock as mock
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        # Seed stale fields from a previous attempt
        ctx = types.SimpleNamespace(
            repo_dir=None,
            _last_review_block_reason="critical_findings",
            _last_review_critical_findings=[
                {"verdict": "FAIL", "item": "secrets_check", "severity": "critical",
                 "reason": "stale from prior run", "model": "old-model"}
            ],
            _review_advisory=[],
            _review_history=[],
            _scope_review_history={},
        )
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            with mock.patch("ouroboros.tools.review._run_unified_review",
                            side_effect=RuntimeError("triad crashed")):
                with mock.patch("ouroboros.tools.scope_review.run_scope_review") as mock_scope:
                    from ouroboros.tools.scope_review import ScopeReviewResult
                    mock_scope.return_value = ScopeReviewResult(blocked=False)
                    review_err, scope_result, triad_block_reason, _ = pr_mod.run_parallel_review(
                        ctx, "test commit")
        # Triad crash must yield infra_failure reason, not the stale critical_findings
        assert triad_block_reason == "infra_failure"
        # Stale findings must be cleared — no bleed-through to aggregate
        assert ctx._last_review_critical_findings == []
        assert "crashed" in review_err

    def test_scope_crash_resets_stale_actor_records(self):
        """If scope crashes, current raw evidence must not reuse previous scope actors."""
        import types
        import unittest.mock as mock
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        ctx = types.SimpleNamespace(
            repo_dir=None,
            _last_review_block_reason="",
            _last_review_critical_findings=[],
            _review_advisory=[],
            _review_history=[],
            _scope_review_history={},
            _last_scope_raw_results=[
                {"slot_id": "stale", "model_id": "old-scope", "status": "responded"}
            ],
        )
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            with mock.patch("ouroboros.tools.review._run_unified_review", return_value=None):
                with mock.patch.object(pr_mod, "run_scope_review", side_effect=RuntimeError("scope crashed")):
                    review_err, scope_result, triad_block_reason, _ = pr_mod.run_parallel_review(
                        ctx, "test commit")

        assert review_err is None
        assert triad_block_reason == ""
        assert scope_result.blocked is True
        assert scope_result.status == "error"
        assert ctx._last_scope_raw_results
        assert ctx._last_scope_raw_results[0]["status"] == "error"
        assert ctx._last_scope_raw_results[0]["slot_id"] == "scope_slot_error"
        assert ctx._last_scope_raw_results[0]["model_id"] != "old-scope"
        assert ctx._last_scope_raw_result["raw_results"][0]["status"] == "error"

    def test_advisory_freshness_path_aware(self):
        """_check_advisory_freshness must accept paths parameter."""
        git = _get_module("ouroboros.tools.git")
        sig = inspect.signature(git._check_advisory_freshness)
        assert "paths" in sig.parameters

# ---------------------------------------------------------------------------
# LLM routing validation (Phase 3, item 6)
# ---------------------------------------------------------------------------

class TestSharedLLMRouting:
    def test_triad_review_uses_llm_client(self):
        """Triad review (_query_model) must use LLMClient, not ad-hoc HTTP."""
        mod = _get_module("ouroboros.tools.review")
        source = inspect.getsource(mod._query_model)
        assert "LLMClient" in source or "llm_client" in source.lower()
        # Must NOT use requests or httpx directly
        assert "requests.post" not in source
        assert "httpx" not in source

    def test_triad_emits_llm_usage_events(self):
        """Triad review must use the shared review usage emitter."""
        mod = _get_module("ouroboros.tools.review")
        source = inspect.getsource(mod._multi_model_review_async)
        assert "emit_review_usage" in source
        helper = inspect.getsource(_get_module("ouroboros.tools.review_helpers").emit_review_usage)
        assert "llm_usage" in helper
        assert "emit_review_event" in helper

    def test_scope_review_uses_llm_client(self):
        """Scope review must use LLMClient for its model call.

        LLMClient is used in _call_scope_llm (called by run_scope_review),
        so we check the whole module for its presence rather than just
        the top-level run_scope_review function.
        """
        mod = _get_module("ouroboros.tools.scope_review")
        # LLMClient is instantiated in _call_scope_llm which run_scope_review delegates to
        source = inspect.getsource(mod._call_scope_llm)
        assert "LLMClient" in source

    def test_scope_review_emits_usage_once_via_substrate(self):
        """Scope usage is emitted exactly ONCE, by the shared review substrate.

        The former job-level re-emit in run_scope_review duplicated every scope
        call in llm_usage telemetry without ledger_attempt_ids (v6.69.0 dedup):
        the substrate per-slot emission is the single telemetry source.
        """
        mod = _get_module("ouroboros.tools.scope_review")
        source = inspect.getsource(mod)
        assert 'source="scope_review")' not in source  # no job-level re-emit
        substrate = inspect.getsource(_get_module("ouroboros.review_substrate"))
        assert 'source=f"review_substrate:{request.surface}"' in substrate
        helper = inspect.getsource(_get_module("ouroboros.tools.review_helpers").emit_review_usage)
        assert "llm_usage" in helper
        assert "emit_review_event" in helper

# ---------------------------------------------------------------------------
# Advisory schema enrichment
# ---------------------------------------------------------------------------

class TestAdvisorySchemaEnriched:
    def test_advisory_schema_has_goal_scope_paths(self):
        adv = _get_module("ouroboros.tools.claude_advisory_review")
        tools = adv.get_tools()
        adv_tool = next(t for t in tools if t.name == "advisory_review")
        props = adv_tool.schema["parameters"]["properties"]
        assert "goal" in props
        assert "scope" in props
        assert "paths" in props

    def test_advisory_prompt_uses_section_loader(self):
        """Advisory prompt builder must use precise section loader, not full CHECKLISTS.md."""
        adv = _get_module("ouroboros.tools.claude_advisory_review")
        source = inspect.getsource(adv._build_advisory_prompt)
        assert "load_checklist_section" in source

    def test_advisory_no_blind_truncation(self):
        """Advisory must not silently truncate raw_result."""
        adv = _get_module("ouroboros.tools.claude_advisory_review")
        source = inspect.getsource(adv._handle_advisory_pre_review)
        assert "raw_result[:4000]" not in source
