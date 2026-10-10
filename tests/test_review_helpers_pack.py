"""Tests for the review stack's shared helpers and the packet seat.

Verifies:
- Checklist section loader extracts exact sections
- Goal/scope precedence: goal > scope > commit_message > fallback
- Touched-file pack builds correctly (the triad packet's own evidence)
- Path-aware freshness
- Stale marking lifecycle
- repo_commit doesn't bypass the one review wave
- The wave's verdict projection (``aggregate_review_verdict``)
- review_helpers imports cleanly (no circular deps)

The two-part brief (Part 2 — the coupling questions) and the retrieving seat's
window sizing live in tests/test_review_brief_coupling.py.
"""

import importlib
import inspect
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_module(name):
    sys.path.insert(0, REPO)
    return importlib.import_module(name)


def test_review_thoroughness_is_count_free_and_evidence_bound():
    helpers = _get_module("ouroboros.tools.review_helpers")
    block = helpers.REVIEW_THOROUGHNESS_BLOCK

    assert "5 bugs" not in block
    assert "zero, one, or many findings are all valid" in block
    assert "Never invent a finding to increase the count" in block


# ---------------------------------------------------------------------------
# review_helpers tests
# ---------------------------------------------------------------------------

class TestChecklistSectionLoader:
    def test_loads_change_review_sections_by_layer(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        core = mod.load_checklist_section("Change Review Checklist")
        assert "## Change Review Checklist" in core
        assert "secrets_check" in core and "bible_compliance" not in core
        body = mod.load_checklist_section("Ouroboros Body Layer")
        assert "bible_compliance" in body
        # Must NOT contain scope checklist
        assert "Coupling questions" not in core + body

    def test_loads_scope_section(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        section = mod.load_checklist_section("Coupling questions")
        assert "## Coupling questions" in section
        assert "intent_alignment" in section
        # Must NOT contain change-review checklist items
        assert "## Change Review Checklist" not in section
        assert "## Ouroboros Body Layer" not in section

    def test_raises_on_missing_section(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        with pytest.raises(ValueError):
            mod.load_checklist_section("Nonexistent Section")


class TestGoalSection:
    def test_goal_section_has_source(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        section = mod.build_goal_section(goal="fix bug", scope="", commit_message="msg")
        assert "Source: goal" in section
        assert "fix bug" in section

    def test_scope_section_empty_when_no_scope(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        section = mod.build_scope_section()
        assert section == ""

    def test_scope_section_present_when_scope(self):
        mod = _get_module("ouroboros.tools.review_helpers")
        section = mod.build_scope_section(scope="only review.py")
        assert "only review.py" in section
        assert "IMPORTANT" in section


class TestTouchedFilePack:
    def test_reads_existing_files(self, tmp_path):
        (tmp_path / "a.py").write_text("print('hello')", encoding="utf-8", newline="\n")
        (tmp_path / "b.md").write_text("# readme", encoding="utf-8", newline="\n")
        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_touched_file_pack(tmp_path, ["a.py", "b.md"])
        assert "a.py" in pack
        assert "print('hello')" in pack
        assert "b.md" in pack
        assert omitted == []

    def test_skips_binary_files(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_touched_file_pack(tmp_path, ["image.png"])
        assert "image.png" in omitted
        assert "```" not in pack or "image.png" not in pack.split("```")[1] if "```" in pack else True

    def test_represents_binary_with_exact_git_metadata(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@ouroboros"],
            cwd=str(tmp_path), check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "TestBot"],
            cwd=str(tmp_path), check=True,
        )
        binary = tmp_path / "native.so"
        binary.write_bytes(b"old\x00payload")
        subprocess.run(["git", "add", "-f", "native.so"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=str(tmp_path), check=True)
        binary.write_bytes(b"new\x00payload")
        subprocess.run(["git", "add", "native.so"], cwd=str(tmp_path), check=True)

        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_touched_file_pack(
            tmp_path, ["native.so"], represent_binary=True
        )

        assert omitted == []
        assert "staged blob" in pack
        assert "pre-merge HEAD blob" in pack
        assert "official MERGE_HEAD blob" in pack
        assert "unknown" not in pack

    def test_binary_metadata_without_stage_zero_stays_omitted(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
        (tmp_path / "native.so").write_bytes(b"unstaged\x00payload")

        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_touched_file_pack(
            tmp_path, ["native.so"], represent_binary=True
        )

        assert omitted == ["native.so"]
        assert "no readable stage-0" in pack

    def test_staged_binary_deletion_has_exact_parent_metadata(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@ouroboros"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.name", "TestBot"], cwd=str(tmp_path), check=True)
        binary = tmp_path / "logo.png"
        binary.write_bytes(b"png\x00payload")
        subprocess.run(["git", "add", "logo.png"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "rm", "logo.png"], cwd=str(tmp_path), check=True)

        helpers = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = helpers.build_touched_file_pack(
            tmp_path, ["logo.png"], represent_binary=True
        )
        assert omitted == []
        assert "staged blob: `absent (deletion)`" in pack
        assert "pre-merge HEAD:" in pack

    def test_extensionless_binary_deletion_has_exact_parent_metadata(self, tmp_path):
        subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@ouroboros"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "config", "user.name", "TestBot"], cwd=str(tmp_path), check=True)
        binary = tmp_path / "firmware"
        binary.write_bytes(b"firmware\x00payload")
        subprocess.run(["git", "add", "firmware"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "commit", "-m", "base"], cwd=str(tmp_path), check=True)
        subprocess.run(["git", "rm", "firmware"], cwd=str(tmp_path), check=True)

        helpers = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = helpers.build_touched_file_pack(
            tmp_path, ["firmware"], represent_binary=True
        )
        assert omitted == []
        assert "staged blob: `absent (deletion)`" in pack
        assert "pre-merge HEAD:" in pack

    def test_omits_large_files(self, tmp_path):
        # _FILE_SIZE_LIMIT is now 1MB; write a file slightly above that threshold
        (tmp_path / "huge.py").write_bytes(b"x" * (1_048_576 + 1))
        mod = _get_module("ouroboros.tools.review_helpers")
        pack, omitted = mod.build_touched_file_pack(tmp_path, ["huge.py"])
        assert "huge.py" in omitted
        assert "omitted" in pack.lower()



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
        (tmp_path / "file.py").write_text("v1", encoding="utf-8", newline="\n")
        subprocess.run(["git", "add", "file.py"], cwd=str(tmp_path), capture_output=True)
        rs = _get_module("ouroboros.review_state")
        h1 = rs.compute_snapshot_hash(tmp_path, "msg")
        # Modify file
        (tmp_path / "file.py").write_text("v2", encoding="utf-8", newline="\n")
        h2 = rs.compute_snapshot_hash(tmp_path, "msg")
        assert h1 != h2

    def test_path_scoped_hash(self, tmp_path):
        """When paths= is provided, only those files affect the hash."""
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        (tmp_path / "a.py").write_text("aaa", encoding="utf-8", newline="\n")
        (tmp_path / "b.py").write_text("bbb", encoding="utf-8", newline="\n")
        rs = _get_module("ouroboros.review_state")
        h_a = rs.compute_snapshot_hash(tmp_path, paths=["a.py"])
        h_b = rs.compute_snapshot_hash(tmp_path, paths=["b.py"])
        assert h_a != h_b


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

    def test_one_wave_wired_in_commit(self):
        """The shared reviewed stage must call the parallel review helper, which
        runs ONE wave (Q25-A two-phase contract: assembly, then dispatch) — the
        former second scope dispatch is gone with the scope role."""
        git = _get_module("ouroboros.tools.git")
        # `_run_reviewed_stage_cycle` runs the one cycle body under the commit's panel.
        assert "_reviewed_stage_cycle(" in inspect.getsource(git._run_reviewed_stage_cycle)
        source = inspect.getsource(_get_module("ouroboros.tools.git_review_cycle")._reviewed_stage_cycle)
        assert "_run_parallel_review" in source
        parallel_source = inspect.getsource(git._run_parallel_review)
        assert "_prepare_unified_review" in parallel_source
        assert "_dispatch_unified_review" in parallel_source
        assert "run_scope_review" not in parallel_source and "_run_scope" not in parallel_source
        # ThreadPoolExecutor must be used for the dispatch under a copied context
        assert "ThreadPoolExecutor" in parallel_source

    def test_repo_commit_not_bypass_the_wave(self):
        """repo_commit must reach the review wave via the shared stage helper."""
        git = _get_module("ouroboros.tools.git")
        source = inspect.getsource(git._repo_commit_push)
        assert "_run_reviewed_stage_cycle" in source
        assert "_reviewed_stage_cycle(" in inspect.getsource(git._run_reviewed_stage_cycle)
        shared_source = inspect.getsource(_get_module("ouroboros.tools.git_review_cycle")._reviewed_stage_cycle)
        # The free checks, the tests and the author's optional one-row look live in
        # the extracted gate helper the stage cycle calls before any paid dispatch.
        assert "_preflight_and_tests_gate" in shared_source
        assert "run_commit_preflight" in inspect.getsource(git._preflight_and_tests_gate)
        assert "_run_parallel_review" in shared_source

    def test_assembly_precedes_admission_precedes_dispatch(self):
        """Q25=A ordering for the one wave: every seat's brief is assembled first,
        the whole wave is admitted (money) next, and only then is the dispatch
        submitted to the pool — each step before the result() collection."""
        git = _get_module("ouroboros.tools.git")
        source = inspect.getsource(git._run_parallel_review)
        prepare = source.find("_prepare_unified_review(")
        admit = source.find("admit_commit_gate_wave(")
        submit = source.find("copy_context().run, _dispatch_unified_review")
        result = source.find("future.result()")
        for position in (prepare, admit, submit, result):
            assert position > 0
        assert prepare < admit < submit < result

    def test_aggregated_verdict_carries_the_coupling_note_beside_the_block(self):
        """When the wave blocks for a reason other than critical findings and the
        coupling question carries findings, the projection appends the coupling
        note to the block message and lists the findings as coupling items."""
        import types
        from ouroboros.review_ledger import CouplingOutcome
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        review_err = "⚠️ REVIEW_BLOCKED: Only 1 of 3 review models responded successfully"
        coupling = CouplingOutcome(
            verdict="FAIL", blocked=True, status="responded",
            critical_findings=[{"verdict": "FAIL", "item": "intent_alignment",
                                "severity": "critical", "reason": "scope blocked", "model": "test"}],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[{"item": "x"}], _review_advisory=[])
        blocked, combined_msg, block_reason, findings, coupling_items = pr_mod.aggregate_review_verdict(
            review_err, coupling, "review_quorum", [], ctx, "test commit", 0.0, ctx.repo_dir)
        assert blocked and block_reason == "review_quorum"
        assert "Only 1 of 3" in combined_msg and "scope blocked" in combined_msg
        assert findings == [{"item": "x"}]
        assert coupling_items == [{"severity": "critical", "tag": "coupling", "item": "intent_alignment",
                                   "reason": "scope blocked", "verdict": "FAIL"}]

    def test_advisory_included_when_the_wave_blocks(self):
        """Advisory findings of the wave appear in the block message."""
        import types
        from ouroboros.review_ledger import CouplingOutcome
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        advisory = [{"item": "context_building", "reason": "advisory note"}]
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        blocked, combined_msg, _reason, findings, _items = pr_mod.aggregate_review_verdict(
            "⚠️ REVIEW_BLOCKED: review NOT_PERFORMED", CouplingOutcome(), "coupling_not_performed",
            advisory, ctx, "test commit", 0.0, ctx.repo_dir)
        assert blocked
        assert "NOT_PERFORMED" in combined_msg and "advisory note" in combined_msg
        assert findings == []

    def test_coupling_advisory_visible_on_successful_commit(self):
        """Non-blocking coupling advisory findings are returned even when the
        wave does not block (the caller surfaces them)."""
        import types
        from ouroboros.review_ledger import CouplingOutcome
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        coupling = CouplingOutcome(
            verdict="PASS", blocked=False, status="responded",
            advisory_findings=[{"verdict": "FAIL", "item": "architecture_fit",
                                "severity": "advisory", "reason": "minor concern", "model": "test"}],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        blocked, combined_msg, _reason, findings, coupling_items = pr_mod.aggregate_review_verdict(
            None, coupling, "", [], ctx, "test commit", 0.0, ctx.repo_dir)
        assert not blocked and combined_msg is None and findings == []
        assert [item["item"] for item in coupling_items] == ["architecture_fit"]
        assert coupling_items[0]["severity"] == "advisory" and coupling_items[0]["tag"] == "coupling"

    @pytest.mark.parametrize("crit_item", sorted(_get_module("ouroboros.tools.scope_review_contract").SCOPE_REQUIRED_ITEMS))
    def test_projection_never_blocks_on_its_own(self, crit_item):
        """NW-2 guardrail (projection seam): the wave's verdict is reduced ONCE
        (``review_ledger.reduce_verdict``) and projected here; with no gate
        verdict (``review_err`` None) the projection must NOT flip to blocked
        for ANY coupling item it merely carries — a 58a52c4-class per-item
        always-block hardcode would fail here."""
        import types
        from ouroboros.review_ledger import CouplingOutcome
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        coupling = CouplingOutcome(
            verdict="FAIL", blocked=False, status="responded",
            critical_findings=[{"verdict": "FAIL", "item": crit_item,
                                "severity": "critical", "reason": "advisory-only note", "model": "test"}],
        )
        ctx = types.SimpleNamespace(
            repo_dir=None, _last_review_critical_findings=[], _review_advisory=[])
        blocked, combined_msg, _reason, _findings, coupling_items = pr_mod.aggregate_review_verdict(
            None, coupling, "", [], ctx, "test commit", 0.0, ctx.repo_dir)
        assert not blocked and combined_msg is None
        assert coupling_items[0]["item"] == crit_item

    def test_assembly_crash_resets_stale_findings(self):
        """If the wave's assembly crashes, stale ctx findings from a prior attempt
        must not bleed into the current run."""
        import types
        import unittest.mock as mock
        pr_mod = _get_module("ouroboros.tools.parallel_review")

        # Seed stale fields from a previous attempt
        ctx = types.SimpleNamespace(
            repo_dir=None, drive_root=None, task_id="stale",
            _last_review_block_reason="critical_findings",
            _last_review_critical_findings=[
                {"verdict": "FAIL", "item": "secrets_check", "severity": "critical",
                 "reason": "stale from prior run", "model": "old-model"}
            ],
            _review_advisory=[],
            _review_history=[],
        )
        with mock.patch.object(pr_mod, "run_cmd", return_value=""):
            with mock.patch("ouroboros.tools.review._prepare_unified_review",
                            side_effect=RuntimeError("assembly crashed")):
                review_err, coupling, block_reason, _ = pr_mod.run_parallel_review(ctx, "test commit")
        # The crash must yield infra_failure, not the stale critical_findings
        assert block_reason == "infra_failure"
        assert ctx._last_review_critical_findings == []
        assert "crashed" in review_err
        assert coupling is None  # no seat was asked anything

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

    def test_review_emits_llm_usage_once_via_substrate(self):
        """Review usage is emitted exactly ONCE, by the shared review substrate.

        The former job-level re-emits (in _multi_model_review_async and in the
        retired scope role) doubled every call in llm_usage telemetry and
        mislabelled a delegated session's provider: the substrate per-slot
        emission is the single source for every seat of the one wave.
        """
        source = inspect.getsource(_get_module("ouroboros.tools.review"))
        assert 'source="review"' not in source  # no job-level re-emit
        for module in ("ouroboros.tools.review_multi_model", "ouroboros.tools.parallel_review",
                       "ouroboros.tools.review_brief_coupling"):
            assert 'source="scope_review")' not in inspect.getsource(_get_module(module))
        substrate = inspect.getsource(_get_module("ouroboros.review_substrate"))
        assert 'source=f"review_substrate:{request.surface}"' in substrate
        helper = inspect.getsource(_get_module("ouroboros.tools.review_helpers").emit_review_usage)
        assert "llm_usage" in helper
        assert "emit_review_event" in helper


class TestTriadPromptAntiPatternLock:
    """v4.34.0: triad pre-commit review prompt now also carries the
    Anti pattern-lock guard. Scope and triad must stay symmetric so
    semantic breadth is guarded without pressuring either surface to invent findings.
    """

    def test_triad_template_has_anti_pattern_lock_guard(self):
        mod = _get_module("ouroboros.tools.review")
        tpl = mod._REVIEW_PROMPT_TEMPLATE_STABLE + mod._REVIEW_PROMPT_TEMPLATE_DYNAMIC
        assert "Anti pattern-lock guard" in tpl
        assert "exactly one FAIL" not in tpl
        guard = _get_module("ouroboros.tools.review_helpers").anti_pattern_lock_guard("body")
        # Normalize whitespace so prompt reflow doesn't break the contract.
        import re
        flat = re.sub(r"\s+", " ", f"{tpl}\n{guard}")
        assert "zero or one FAIL is valid" in flat
        assert "numeric finding quota" in flat
        # Accept any casing — "different concern class" / "DIFFERENT concern class"
        assert "concern class" in flat.lower()
        assert "second pass" in flat.lower()


def test_default_context_mode_is_max_and_generic_settings_merge_preserves_it(monkeypatch):
    """The default and ordinary settings writer retain their explicit contract."""
    from ouroboros import config
    from ouroboros.gateway.settings import _merge_settings_payload

    assert config.SETTINGS_DEFAULTS["OUROBOROS_CONTEXT_MODE"] == "max"
    monkeypatch.delenv("OUROBOROS_CONTEXT_MODE", raising=False)
    assert config.get_context_mode() == "max"

    merged = _merge_settings_payload({"OUROBOROS_CONTEXT_MODE": "max"},
                                     {"OUROBOROS_CONTEXT_MODE": "low"})
    assert merged["OUROBOROS_CONTEXT_MODE"] == "max"


class TestTriadPackExclusions:
    """The triad pack's disclosed exclusion classes (review economics, D-06a).

    The builder takes the advisory seam's ``exclude_paths`` shape and marks an
    excluded path ONCE; ``triad_pack_exclusions`` names exactly two classes the
    host can back — span-only release carriers on a VERSION-staged commit
    (``release_sync`` carrier SSOT) and governance docs byte-identical to the
    inlined prefix copy — and returns the disclosure note the caller appends."""

    def test_exclude_paths_withhold_the_text_with_one_marker(self, tmp_path):
        mod = _get_module("ouroboros.tools.review_helpers")
        # Oversize AND excluded: the exclusion marker wins, never two markers.
        (tmp_path / "uv.lock").write_bytes(b"x" * (1_048_576 + 1))
        (tmp_path / "a.py").write_text("print('kept')", encoding="utf-8", newline="\n")
        pack, omitted = mod.build_touched_file_pack(
            tmp_path, ["uv.lock", "a.py"], exclude_paths={"uv.lock"})
        assert omitted == ["uv.lock"]
        assert pack.count("### uv.lock") == 1
        assert "withheld by the caller's exclusion note" in pack
        assert "byte limit" not in pack and "xxxx" not in pack
        assert "print('kept')" in pack
        # The default is byte-identical to the pre-exclusion builder.
        pack_default, omitted_default = mod.build_touched_file_pack(tmp_path, ["a.py"])
        assert omitted_default == [] and "print('kept')" in pack_default

    @staticmethod
    def _carrier_repo(tmp_path, *, with_lock=True):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(repo), check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=str(repo), check=True)
        (repo / "VERSION").write_text("1.0.0\n", encoding="utf-8", newline="\n")
        (repo / "pyproject.toml").write_text(
            '[project]\nname = "ouroboros"\nversion = "1.0.0"\n', encoding="utf-8", newline="\n")
        if with_lock:
            (repo / "uv.lock").write_text(_uv_lock_text("1.0.0"), encoding="utf-8", newline="\n")
        (repo / "docs").mkdir()
        (repo / "docs" / "ARCHITECTURE.md").write_text(
            "# Ouroboros v1.0.0 — Architecture\n\nArchitecture body.\n", encoding="utf-8", newline="\n")
        (repo / "docs" / "DEVELOPMENT.md").write_text("# DEV\n\nHandbook body.\n", encoding="utf-8", newline="\n")
        (repo / "app.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=str(repo), check=True)
        return repo

    @staticmethod
    def _staged_paths(repo):
        out = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=str(repo),
                             check=True, capture_output=True, text=True).stdout
        return [line for line in out.splitlines() if line]

    def test_span_only_carriers_and_prefix_duplicates_are_cut_on_a_version_bump(self, tmp_path):
        mod = _get_module("ouroboros.tools.review_file_pack")
        repo = self._carrier_repo(tmp_path)
        (repo / "VERSION").write_text("1.0.1\n", encoding="utf-8", newline="\n")
        (repo / "uv.lock").write_text(_uv_lock_text("1.0.1"), encoding="utf-8", newline="\n")
        # pyproject: version bump PLUS a dependency edit outside its span.
        (repo / "pyproject.toml").write_text(
            '[project]\nname = "ouroboros"\nversion = "1.0.1"\ndependencies = ["httpx"]\n',
            encoding="utf-8", newline="\n")
        (repo / "docs" / "ARCHITECTURE.md").write_text(
            "# Ouroboros v1.0.1 — Architecture\n\nArchitecture body.\n", encoding="utf-8", newline="\n")
        (repo / "docs" / "DEVELOPMENT.md").write_text("# DEV\n\nHandbook body, revised.\n", encoding="utf-8", newline="\n")
        (repo / "app.py").write_text("x = 2\n", encoding="utf-8", newline="\n")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
        paths = self._staged_paths(repo)
        dev_text = (repo / "docs" / "DEVELOPMENT.md").read_text(encoding="utf-8")

        excluded, note = mod.triad_pack_exclusions(
            repo, paths, prefix_texts={"docs/DEVELOPMENT.md": dev_text, "docs/DESIGN.md": ""})

        assert excluded == {"VERSION", "uv.lock", "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md"}
        assert "pyproject.toml" not in excluded and "app.py" not in excluded
        assert note.startswith("⚠️ PACK EXCLUSION NOTE: full text withheld for 4 touched file(s)")
        assert "VERSION_CARRIER_SPANS" in note and "version_carrier_desyncs" in note
        assert "uv.lock" in note and "byte-identical" in note and "docs/DEVELOPMENT.md" in note
        # The pack renders the cut through the builder's own marker + omitted list.
        pack, omitted = mod.build_touched_file_pack(repo, paths, exclude_paths=excluded)
        assert set(omitted) == excluded
        assert "httpx" in pack and "x = 2" in pack  # kept texts
        assert "Handbook body, revised." not in pack and "editable" not in pack  # withheld texts

    def test_without_version_staged_carriers_keep_their_text(self, tmp_path):
        """The carrier class is a release-bump mechanism: no VERSION staged, no
        carrier cut (the preflight carrier gate did not run); the prefix-dedup
        class is independent of it."""
        mod = _get_module("ouroboros.tools.review_file_pack")
        repo = self._carrier_repo(tmp_path)
        (repo / "uv.lock").write_text(_uv_lock_text("1.0.1"), encoding="utf-8", newline="\n")
        (repo / "docs" / "DEVELOPMENT.md").write_text("# DEV\n\nHandbook body, revised.\n", encoding="utf-8", newline="\n")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
        paths = self._staged_paths(repo)
        dev_text = (repo / "docs" / "DEVELOPMENT.md").read_text(encoding="utf-8")

        excluded, note = mod.triad_pack_exclusions(
            repo, paths, prefix_texts={"docs/DEVELOPMENT.md": dev_text})
        assert excluded == {"docs/DEVELOPMENT.md"}
        assert "release carrier" not in note and "byte-identical" in note
        # A prefix copy with DIFFERENT bytes (or none) keeps the doc's full text.
        assert mod.triad_pack_exclusions(
            repo, paths, prefix_texts={"docs/DEVELOPMENT.md": "# DEV\n\nOther bytes.\n"}) == (set(), "")
        assert mod.triad_pack_exclusions(repo, paths, prefix_texts={}) == (set(), "")

    def test_a_carrier_new_at_head_keeps_its_text(self, tmp_path):
        mod = _get_module("ouroboros.tools.review_file_pack")
        repo = self._carrier_repo(tmp_path, with_lock=False)
        (repo / "VERSION").write_text("1.0.1\n", encoding="utf-8", newline="\n")
        (repo / "uv.lock").write_text(_uv_lock_text("1.0.1"), encoding="utf-8", newline="\n")
        subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True)
        excluded, _note = mod.triad_pack_exclusions(
            repo, self._staged_paths(repo), prefix_texts={})
        assert excluded == {"VERSION"}


def _uv_lock_text(version):
    return (
        'version = 1\n\n[[package]]\nname = "ouroboros"\n'
        f'version = "{version}"\nsource = {{ editable = "." }}\n\n'
        '[[package]]\nname = "httpx"\nversion = "0.27.0"\n'
    )
