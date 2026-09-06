from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.tools.review_context_atlas import (
    ATLAS_ASSEMBLY_FAILURE_STATUSES,
    ReviewContextAtlasRequest,
    atlas_assembly_failed,
    atlas_assembly_failure_reason,
    atlas_hard_budget_overflowed,
    atlas_unassembled_required,
    compile_review_context_atlas,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _coverage(pack):
    return {row["path"]: row for row in pack.manifest["coverage"]}


def test_atlas_accounts_for_every_tracked_path_and_excludes_unrelated_tests(tmp_path):
    _write(tmp_path / "app.py", "import helper\n\ndef run():\n    return helper.value()\n")
    _write(tmp_path / "helper.py", "def value():\n    return 42\n")
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(tmp_path / "pkg" / "main.py", "from .helper import thing\n\nanswer = thing()\n")
    _write(tmp_path / "pkg" / "helper.py", "def thing():\n    return 7\n")
    _write(tmp_path / "tests" / "test_app.py", "def test_app():\n    assert True\n")
    _write(tmp_path / "docs" / "CHECKLISTS.md", "canonical checklist\n")

    tracked = (
        "app.py",
        "helper.py",
        "pkg/__init__.py",
        "pkg/main.py",
        "pkg/helper.py",
        "tests/test_app.py",
        "docs/CHECKLISTS.md",
    )
    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=tracked,
            anchors=("app.py",),
            already_included=frozenset({"docs/CHECKLISTS.md"}),
            fixed_prompt_tokens=100,
            target_total_tokens=20_000,
            hard_total_tokens=25_000,
            include_tests=False,
        )
    )

    coverage = _coverage(pack)
    assert set(coverage) == set(tracked)
    assert coverage["docs/CHECKLISTS.md"]["disposition"] == "already_included"
    assert coverage["tests/test_app.py"]["disposition"] == "excluded_test"
    assert "pkg.helper" in coverage["pkg/main.py"]["imports"]
    assert "def test_app" not in pack.text


def test_atlas_include_tests_allows_test_files(tmp_path):
    _write(tmp_path / "tests" / "test_app.py", "def test_app():\n    assert True\n")

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("tests/test_app.py",),
            anchors=("tests/test_app.py",),
            fixed_prompt_tokens=100,
            target_total_tokens=20_000,
            hard_total_tokens=25_000,
            include_tests=True,
        )
    )

    coverage = _coverage(pack)
    assert coverage["tests/test_app.py"]["disposition"] == "full"
    assert "def test_app" in pack.text


def test_atlas_compact_manifest_keeps_full_coverage_out_of_prompt(tmp_path):
    _write(tmp_path / "app.py", "import helper\n\nprint(helper.VALUE)\n")
    _write(tmp_path / "helper.py", "VALUE = 42\n")
    _write(tmp_path / "other.py", "def unused():\n    return 'ok'\n")

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("app.py", "helper.py", "other.py"),
            anchors=("app.py",),
            fixed_prompt_tokens=100,
            target_total_tokens=20_000,
            hard_total_tokens=25_000,
            compact_manifest=True,
        )
    )

    assert pack.manifest["compact_manifest_in_prompt"] is True
    assert {row["path"] for row in pack.manifest["coverage"]} == {
        "app.py",
        "helper.py",
        "other.py",
    }
    assert '"coverage": [' not in pack.text
    assert '"coverage_in_prompt": "compact_full_index_plus_bounded_samples"' in pack.text
    assert '"coverage_samples"' in pack.text
    assert '"coverage_sample_counts"' in pack.text
    assert '"coverage_index_count": 3' in pack.text
    assert "### Compact full coverage index" in pack.text
    for rel_path in ("app.py", "helper.py", "other.py"):
        assert f"\t{rel_path}" in pack.text
    assert "compact coverage mode" in pack.text


def test_atlas_force_includes_protected_workflow_even_under_skipped_github_dir(tmp_path):
    _write(tmp_path / ".github" / "workflows" / "ci.yml", "name: CI\n")
    _write(tmp_path / "ouroboros" / "tools" / "review_context_atlas.py", "ATLAS = True\n")
    _write(tmp_path / "ouroboros" / "tools" / "plan_review_runtime.py", "RUNTIME = True\n")
    _write(tmp_path / "assets" / "logo.txt", "asset text\n")
    _write(tmp_path / "main.py", "print('main')\n")

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=(
                ".github/workflows/ci.yml",
                "ouroboros/tools/review_context_atlas.py",
                "ouroboros/tools/plan_review_runtime.py",
                "assets/logo.txt",
                "main.py",
            ),
            fixed_prompt_tokens=100,
            target_total_tokens=20_000,
            hard_total_tokens=25_000,
        )
    )

    coverage = _coverage(pack)
    assert coverage[".github/workflows/ci.yml"]["disposition"] == "full"
    assert coverage["ouroboros/tools/review_context_atlas.py"]["disposition"] == "full"
    assert coverage["ouroboros/tools/plan_review_runtime.py"]["disposition"] == "full"
    assert "name: CI" in pack.text
    assert coverage["assets/logo.txt"]["disposition"] == "excluded_dir"
    assert "asset text" not in pack.text


def test_atlas_force_includes_the_review_execution_seam(tmp_path):
    """The v6.87.21 seam module is review-stack surface, not an ordinary dependency.

    ``review_execution.py`` owns the route vocabulary, transport dispatch and
    api_chat prompt rendering for review slots; a broad review pack that
    budget-drops it hides the review stack's own execution layer from the
    reviewers guarding it. Pinned at both levels: the membership predicate and
    the compiled pack's disposition.
    """
    from ouroboros.tools.review_context_atlas import _is_force_include

    assert _is_force_include("ouroboros/review_execution.py") is True

    _write(tmp_path / "ouroboros" / "review_execution.py", "SEAM = True\n")
    _write(tmp_path / "main.py", "print('main')\n")
    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("ouroboros/review_execution.py", "main.py"),
            fixed_prompt_tokens=100,
            target_total_tokens=20_000,
            hard_total_tokens=25_000,
        )
    )
    assert _coverage(pack)["ouroboros/review_execution.py"]["disposition"] == "full"
    assert "SEAM = True" in pack.text


def test_atlas_devtools_manifest_only_unless_touched(tmp_path):
    _write(tmp_path / "devtools" / "benchmarks" / "programbench" / "run.py", "VALUE = 'devtools full text'\n")
    _write(tmp_path / "ouroboros" / "core.py", "print('core')\n")

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("devtools/benchmarks/programbench/run.py", "ouroboros/core.py"),
            anchors=("ouroboros/core.py",),
            fixed_prompt_tokens=100,
            target_total_tokens=20_000,
            hard_total_tokens=25_000,
        )
    )

    coverage = _coverage(pack)
    assert coverage["devtools/benchmarks/programbench/run.py"]["disposition"] == "excluded_dir"
    assert "devtools full text" not in pack.text
    assert coverage["ouroboros/core.py"]["disposition"] == "full"

    touched = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("devtools/benchmarks/programbench/run.py",),
            anchors=("devtools/benchmarks/programbench/run.py",),
            fixed_prompt_tokens=100,
            target_total_tokens=20_000,
            hard_total_tokens=25_000,
        )
    )

    touched_coverage = _coverage(touched)
    assert touched_coverage["devtools/benchmarks/programbench/run.py"]["disposition"] == "full"
    assert "devtools full text" in touched.text


def test_atlas_marks_sensitive_binary_oversized_and_vendored_files(tmp_path):
    _write(tmp_path / ".env.example", "TOKEN=secret\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x00")
    _write(tmp_path / "script.min.js", "minified();\n")
    (tmp_path / "huge.py").write_bytes(b"x" * (1_048_576 + 1))
    normal_source = "\n".join(f"import pkg_{idx}" for idx in range(30))
    normal_source += '\nDATABASE_URL = "postgres://alice:secretpw@db.local/app"\n'
    normal_source += "\n".join(f"def f_{idx}():\n    return {idx}\n" for idx in range(20))
    _write(tmp_path / "normal.py", normal_source)

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=(".env.example", "image.png", "script.min.js", "huge.py", "normal.py"),
            fixed_prompt_tokens=100,
            target_total_tokens=20_000,
            hard_total_tokens=25_000,
        )
    )

    coverage = _coverage(pack)
    assert coverage[".env.example"]["disposition"] == "sensitive"
    assert coverage[".env.example"]["sha256"] == ""
    assert coverage[".env.example"]["size"] == 0
    assert coverage["image.png"]["disposition"] == "binary_media"
    assert coverage["script.min.js"]["disposition"] == "vendored_minified"
    assert coverage["huge.py"]["disposition"] == "oversized"
    assert coverage["normal.py"]["disposition"] == "full"
    assert coverage["normal.py"]["imports_total"] == 30
    assert coverage["normal.py"]["symbols_total"] >= 20
    assert len(coverage["normal.py"]["imports"]) <= 12
    assert "secretpw" not in pack.text
    assert "postgres://***REDACTED***@db.local/app" in pack.text


def test_atlas_respects_total_prompt_target_and_reports_budget_manifest_only(tmp_path):
    tracked = []
    for idx in range(8):
        rel = f"pkg/mod_{idx}.py"
        tracked.append(rel)
        _write(tmp_path / rel, ("def f():\n    return 'x'\n" * 120))

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=tuple(tracked),
            fixed_prompt_tokens=100,
            target_total_tokens=5_000,
            hard_total_tokens=8_000,
        )
    )

    assert pack.manifest["estimated_total_tokens"] <= 8_000
    assert pack.manifest["selected_count"] < len(tracked)
    assert any(row["disposition"] == "manifest_only" for row in pack.manifest["coverage"])
    assert pack.status in {"budget_constrained", "ok"}

    _write(tmp_path / "BIBLE.md", "constitution\n" * 500)
    overflow = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("BIBLE.md",),
            fixed_prompt_tokens=100,
            target_total_tokens=300,
            hard_total_tokens=350,
        )
    )
    # Even the content-free manifest exceeds this micro budget (hard context
    # allowance is 0 after fixed+headroom) — only then budget_exceeded survives.
    assert overflow.status == "budget_exceeded"
    assert atlas_assembly_failed(overflow)
    assert _coverage(overflow)["BIBLE.md"]["disposition"] == "budget_omitted"


def test_atlas_required_overflow_is_an_assembly_failure_not_a_smaller_pack(tmp_path):
    """BIBLE P3 scope floor: a REQUIRED artifact that does not fit fails the
    ASSEMBLY. The pack must not come back reviewable (`budget_constrained`) with
    the artifact quietly downgraded to a manifest row — the row is disclosure
    that ACCOMPANIES the refusal, never a substitute for it."""
    _write(tmp_path / "BIBLE.md", "constitution\n" * 3000)  # ~9K tokens > hard allowance
    _write(tmp_path / "small.py", "def f():\n    return 'x'\n" * 30)

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("BIBLE.md", "small.py"),
            fixed_prompt_tokens=100,
            target_total_tokens=4_000,
            hard_total_tokens=10_000,
        )
    )

    coverage = _coverage(pack)
    # The refusal.
    assert pack.status == "required_artifact_omitted"
    assert atlas_assembly_failed(pack)
    assert pack.status in ATLAS_ASSEMBLY_FAILURE_STATUSES
    # The disclosure that accompanies it: typed, naming artifact AND reason,
    # carried by the ONE reader every consumer uses. The durable manifest is that
    # carrier — a parallel typed attribute would be a second copy of one fact,
    # and every production reader already goes through the manifest.
    assert [row["path"] for row in atlas_unassembled_required(pack.manifest)] == ["BIBLE.md"]
    assert [row["path"] for row in pack.manifest["unassembled_required"]] == ["BIBLE.md"]
    assert not hasattr(pack, "unassembled_required")
    # Successor pin: the reason states the honest arithmetic (remaining budget
    # after higher-priority content and the measured manifest render), not the
    # old ambiguous "file exceeded the hard budget" claim.
    assert "does not fit the atlas hard budget" in pack.manifest["unassembled_required"][0]["reason"]
    assert "remain after higher-priority content and the rendered manifest" in (
        pack.manifest["unassembled_required"][0]["reason"]
    )
    assert "BIBLE.md" in atlas_assembly_failure_reason(pack)
    assert coverage["BIBLE.md"]["disposition"] == "budget_omitted"
    assert coverage["small.py"]["disposition"] == "full"
    assert pack.manifest["estimated_total_tokens"] <= 10_000


def test_atlas_required_removed_by_shrink_wave_is_the_same_assembly_failure(tmp_path):
    """The twin branch: a required artifact that PASSED selection and was then
    dropped by the hard-budget shrink wave is the identical failure — one
    branch fixed while its sibling degrades silently is the whole defect.

    Successor construction: with the manifest render now MEASURED and charged
    at admission, the wave fires only on residual estimator drift (the
    selected-rows part of the manifest preview grows with each admission), so
    the scenario uses many small required files whose per-admission row cost
    accumulates past the greedy slack."""
    tracked = []
    for idx in range(30):
        rel = f"prompts/section_{idx:02d}/prompt_body_file_{idx:02d}.md"
        tracked.append(rel)
        _write(tmp_path / rel, "prompt line\n" * 60)

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=tuple(tracked),
            fixed_prompt_tokens=100,
            target_total_tokens=11_750,
            hard_total_tokens=12_750,
        )
    )

    omitted_reasons = {row["path"]: row["reason"] for row in pack.manifest["unassembled_required"]}
    assert omitted_reasons, "shrink wave must have dropped at least one required prompt file"
    assert any(
        "admitted, then removed because the rendered atlas exceeded the hard budget" in reason
        for reason in omitted_reasons.values()
    ), f"expected the shrink-wave branch, got: {omitted_reasons}"
    assert pack.status == "required_artifact_omitted"
    assert atlas_assembly_failed(pack)


def test_atlas_mixed_failure_reports_both_causes_without_losing_disclosure(tmp_path):
    """The MIXED assembly failure: required candidates are marked budget_omitted
    BEFORE the rendered content-free atlas is tested against the hard budget, so
    one pack can carry status="budget_exceeded" AND non-empty
    manifest["unassembled_required"] simultaneously. The reason must then render
    BOTH causes — treating the required rows as an exclusive discriminator
    suppresses the overflow and prescribes a remedy that cannot resolve it."""
    # Identical shape to the micro-budget case in
    # test_atlas_respects_total_prompt_target_and_reports_budget_manifest_only:
    # the hard context allowance is 0 after fixed+headroom, so even the
    # manifest-only pack overflows — while required BIBLE.md was already dropped.
    _write(tmp_path / "BIBLE.md", "constitution\n" * 500)
    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("BIBLE.md",),
            fixed_prompt_tokens=100,
            target_total_tokens=300,
            hard_total_tokens=350,
        )
    )

    # The mixed state itself: both facts, one pack.
    assert pack.status == "budget_exceeded"
    assert atlas_assembly_failed(pack)
    assert atlas_hard_budget_overflowed(pack.manifest)
    rows = atlas_unassembled_required(pack.manifest)
    # The required-artifact disclosure is NOT lost to the overflow…
    assert [row["path"] for row in rows] == ["BIBLE.md"]
    # …and the overflow is NOT suppressed behind the required rows: the shared
    # reason renders BOTH causes for every consumer (scope, plan, deep review).
    reason = atlas_assembly_failure_reason(pack)
    assert "BIBLE.md" in reason
    assert "required artifact could not be assembled" in reason
    assert "exceeded hard budget" in reason

    # The halves that must NOT change: each single-cause failure still reports
    # exactly its own cause, nothing extra.
    pure_missing = SimpleNamespace(
        status="required_artifact_omitted",
        manifest={
            "status": "required_artifact_omitted",
            "estimated_total_tokens": 900,
            "unassembled_required": [{"path": "BIBLE.md", "reason": "required file exceeded the atlas hard budget"}],
        },
    )
    missing_reason = atlas_assembly_failure_reason(pure_missing)
    assert "BIBLE.md" in missing_reason
    assert "exceeded hard budget (~" not in missing_reason
    assert not atlas_hard_budget_overflowed(pure_missing.manifest)
    pure_overflow = SimpleNamespace(
        status="budget_exceeded",
        manifest={"status": "budget_exceeded", "estimated_total_tokens": 950_000, "unassembled_required": []},
    )
    overflow_reason = atlas_assembly_failure_reason(pure_overflow)
    assert "required artifact" not in overflow_reason
    assert "exceeded hard budget" in overflow_reason
    assert atlas_hard_budget_overflowed(pure_overflow.manifest)


def test_atlas_navigational_omission_stays_legal(tmp_path):
    """The counterpart the owner preserved: manifest-only NAVIGATION entries for
    non-required files are a lossless map, not an assembly failure."""
    for idx in range(8):
        _write(tmp_path / f"pkg/mod_{idx}.py", "def f():\n    return 'x'\n" * 120)

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=tuple(f"pkg/mod_{idx}.py" for idx in range(8)),
            fixed_prompt_tokens=100,
            target_total_tokens=5_000,
            hard_total_tokens=8_000,
        )
    )

    assert any(row["disposition"] == "manifest_only" for row in pack.manifest["coverage"])
    assert pack.manifest["unassembled_required"] == []
    assert not atlas_assembly_failed(pack)
    assert pack.status == "budget_constrained"


def test_atlas_oversized_required_artifact_is_a_typed_assembly_failure(tmp_path):
    """XR-1: `_build_file_facts` used to classify a file over the per-file 1MB
    cap as `oversized` and return BEFORE requiredness was computed, so a
    REQUIRED artifact (here a force-included `prompts/` file) was silently
    dropped while the pack assembled and review proceeded — bypassing the
    BIBLE P3 typed-failure guarantee. Requiredness is now decided before any
    disposition can drop an artifact, and the drop lands on the ONE existing
    predicate (`budget_omitted` + `required_artifact_omitted`)."""
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "huge.md").write_bytes(b"x" * (1_048_576 + 1))
    _write(tmp_path / "ok.py", "print(1)\n")

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("prompts/huge.md", "ok.py"),
        )
    )

    assert pack.status == "required_artifact_omitted"
    assert atlas_assembly_failed(pack)
    coverage = _coverage(pack)
    assert coverage["prompts/huge.md"]["disposition"] == "budget_omitted"
    assert "per-file atlas cap" in coverage["prompts/huge.md"]["reason"]
    assert [row["path"] for row in atlas_unassembled_required(pack.manifest)] == [
        "prompts/huge.md"
    ]
    # The legal half is untouched: a NON-required oversized file stays a plain
    # coverage row (pinned by
    # test_atlas_marks_sensitive_binary_oversized_and_vendored_files).


def test_atlas_diff_only_required_artifact_is_a_typed_assembly_failure(tmp_path):
    """XR-4: the `diff_only_included` classification returned before
    requiredness was computed, so an artifact owed IN FULL regardless of the
    change (prompts/, contracts/, protected + review stack, canonical docs)
    whose full snapshot the ladder dropped to diff-only left NO typed failure:
    the full text was in neither the fixed prompt nor the atlas, and review
    proceeded. A merely-touched file on the same path keeps the sanctioned
    disclosed degradation (guaranteed-fit ladder step 4) — its complete
    change-evidence is the staged diff itself."""
    _write(tmp_path / "prompts" / "touched.md", "p\n" * 200)
    _write(tmp_path / "module.py", "q = 1\n" * 200)

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("prompts/touched.md", "module.py"),
            anchors=("prompts/touched.md", "module.py"),
            already_included=frozenset({"prompts/touched.md", "module.py"}),
            diff_only_included=frozenset({"prompts/touched.md", "module.py"}),
        )
    )

    assert pack.status == "required_artifact_omitted"
    assert atlas_assembly_failed(pack)
    coverage = _coverage(pack)
    assert coverage["prompts/touched.md"]["disposition"] == "budget_omitted"
    assert "diff-only" in coverage["prompts/touched.md"]["reason"]
    # The merely-touched module keeps the legal disclosed diff-only row.
    assert coverage["module.py"]["disposition"] == "already_included"
    assert "full snapshot omitted" in coverage["module.py"]["reason"]
    assert [row["path"] for row in atlas_unassembled_required(pack.manifest)] == [
        "prompts/touched.md"
    ]


def test_atlas_includes_anchored_test_file_instead_of_silently_excluding_it(tmp_path):
    """Sibling-path sweep of XR-1/XR-4: `excluded_test` exempted force-include
    but NOT anchors (while `excluded_dir` exempts both), so a REQUIRED-by-anchor
    test file was silently droppable with `include_tests=False`. Tests are
    excludable only when UNRELATED to the change (BIBLE P3); an anchor is
    related by definition."""
    _write(tmp_path / "tests" / "test_mod.py", "def test_ok():\n    assert True\n")
    _write(tmp_path / "tests" / "test_other.py", "def test_other():\n    assert True\n")
    _write(tmp_path / "mod.py", "x = 1\n")

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("tests/test_mod.py", "tests/test_other.py", "mod.py"),
            anchors=("tests/test_mod.py",),
            include_tests=False,
        )
    )

    coverage = _coverage(pack)
    assert coverage["tests/test_mod.py"]["disposition"] == "full"
    # Unrelated wider tests stay excluded by policy, exactly as before.
    assert coverage["tests/test_other.py"]["disposition"] == "excluded_test"
    assert not atlas_assembly_failed(pack)


def test_atlas_centrality_scores_default_off_is_identical(tmp_path):
    """Empty centrality_scores (the scope/plan path) must produce byte-identical
    selection to the heuristic baseline — D2 is strictly additive."""
    tracked = [f"mod_{i}.py" for i in range(6)]
    for rel in tracked:
        _write(tmp_path / rel, ("def f():\n    return 'x'\n" * 60))

    def _compile(**extra):
        return compile_review_context_atlas(
            ReviewContextAtlasRequest(
                repo_dir=tmp_path,
                tracked_paths=tuple(tracked),
                fixed_prompt_tokens=100,
                target_total_tokens=4_000,
                hard_total_tokens=6_000,
                **extra,
            )
        )

    baseline = _compile()
    explicit_empty = _compile(centrality_scores={})
    assert [r.rel_path for r in baseline.selected] == [r.rel_path for r in explicit_empty.selected]
    assert baseline.text == explicit_empty.text


def test_atlas_centrality_scores_boost_selection_order(tmp_path):
    """A centrality bonus must pull a hub module into the bounded selection
    ahead of equal-sized peers, without touching required/anchor tiers."""
    tracked = [f"mod_{i}.py" for i in range(6)]
    for rel in tracked:
        _write(tmp_path / rel, ("def f():\n    return 'x'\n" * 60))

    # Target budget fits only ~3 of 6 equal-sized files (selection pressure);
    # hard budget leaves real headroom past the atlas's 5K hard reserve.
    boosted = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=tuple(tracked),
            fixed_prompt_tokens=100,
            target_total_tokens=2_300,
            hard_total_tokens=7_000,
            centrality_scores={"mod_5.py": 600.0},
        )
    )

    selected = [r.rel_path for r in boosted.selected]
    assert selected, "tight budget must still select at least one file"
    assert selected[0] == "mod_5.py", "the centrality-boosted hub must be picked first"
    assert "mod_2.py" not in selected, "the boost must displace an unboosted peer"
    cov = _coverage(boosted)
    assert "graph_centrality" in cov["mod_5.py"]["reason"]


def _not_assembled_pack():
    """A pack whose REQUIRED artifact could not be assembled (BIBLE P3 refusal)."""
    return SimpleNamespace(
        text="## Generated Scope Atlas\n(remainder without ouroboros/llm.py)\n",
        manifest={
            "estimated_total_tokens": 500_000,
            "unassembled_required": [
                {"path": "ouroboros/llm.py", "reason": "required file exceeded the atlas hard budget"}
            ],
        },
        status="required_artifact_omitted",
        selected=(),
        omitted=(),
        token_count=500_000,
    )


def test_scope_review_refuses_a_pack_that_did_not_assemble(monkeypatch, tmp_path):
    """Consumer 1 of 3. Scope review must not review the remainder: the atlas
    section is refused, and the manifest disclosure is recorded BEFORE the
    refusal so it accompanies it rather than replacing it."""
    from ouroboros.tools import scope_review as sr

    monkeypatch.setattr(sr, "compile_review_context_atlas", lambda req: _not_assembled_pack())
    sr._SCOPE_CONTEXT_MANIFEST.set({})

    with pytest.raises(sr._ScopeAtlasNotAssembled) as excinfo:
        sr._gather_scope_packs(tmp_path, ["ouroboros/llm.py"])

    assert "ouroboros/llm.py" in str(excinfo.value)
    manifest = sr._current_scope_context_manifest()
    assert manifest["unassembled_required"][0]["path"] == "ouroboros/llm.py"


def test_deep_self_review_refuses_a_pack_that_did_not_assemble(monkeypatch, tmp_path):
    """Consumer 2 of 3. Same predicate, same refusal — no pack text is returned
    for a review that could not assemble a required artifact."""
    from ouroboros import deep_self_review as dsr

    monkeypatch.setattr(dsr, "_dulwich_tracked_paths", lambda repo_dir: (["main.py"], []))
    monkeypatch.setattr(dsr, "compile_review_context_atlas", lambda req: _not_assembled_pack())

    pack_text, stats = dsr.build_review_pack(tmp_path, tmp_path / "drive")

    assert pack_text == ""
    assert "ouroboros/llm.py" in stats["skipped"][0]
    assert stats["context_manifest"]["unassembled_required"][0]["path"] == "ouroboros/llm.py"


def test_required_beyond_diff_is_one_public_definition_for_every_consumer():
    """The class owed IN FULL is exported so the scope ladder (which chooses what
    to degrade) and the assembler (which refuses a degraded required artifact)
    read ONE definition. If they drift, the ladder degrades a path the assembler
    can only refuse — a manufactured, unfixable budget deficit."""
    from ouroboros.tools.review_context_atlas import atlas_required_beyond_diff

    # force-include: prompts/, contracts/, protected runtime, review stack
    assert atlas_required_beyond_diff("prompts/x.md")
    assert atlas_required_beyond_diff("ouroboros/contracts/thing.py")
    assert atlas_required_beyond_diff(".github/workflows/ci.yml")
    assert atlas_required_beyond_diff("ouroboros/tools/scope_review.py")
    # canonical context docs (the second clause — BIBLE.md is also force-include,
    # so ARCHITECTURE.md is what proves the clause is not dead)
    assert atlas_required_beyond_diff("BIBLE.md")
    assert atlas_required_beyond_diff("docs/ARCHITECTURE.md")
    # a merely-touched ordinary file may legally degrade to diff-only
    assert not atlas_required_beyond_diff("module.py")
    assert not atlas_required_beyond_diff("tests/test_thing.py")


def test_atlas_charges_the_measured_manifest_render_at_admission(tmp_path):
    """Issue #284 pack arithmetic: the rendered manifest/index skeleton is part
    of the pack, so admission charges it up front instead of reserving a blind
    constant. Observable contract: the assembled pack respects the hard budget
    WITHOUT wave-evicting content that was admitted against a budget the render
    then consumed — for a comfortably-sized repo no coverage row may carry an
    'admitted, then removed' reason, and the total honors the hard cap."""
    tracked = []
    for idx in range(12):
        rel = f"pkg/module_{idx:02d}.py"
        tracked.append(rel)
        _write(tmp_path / rel, "def f():\n    return 'x'\n" * 80)

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=tuple(tracked),
            fixed_prompt_tokens=100,
            target_total_tokens=6_000,
            hard_total_tokens=9_000,
        )
    )

    assert pack.manifest["estimated_total_tokens"] <= 9_000
    assert pack.status in {"budget_constrained", "ok", "under_target"}
    reasons = [row["reason"] for row in pack.manifest["coverage"]]
    assert not any("admitted, then removed" in reason for reason in reasons), reasons
    # Files that did not fit were refused AT ADMISSION with the honest reason.
    not_selected = [r for r in reasons if "not selected within atlas target budget" in r]
    assert not_selected, "the target budget must have refused at least one candidate"


def test_atlas_admission_reason_reports_remaining_capacity_not_a_file_claim(tmp_path):
    """Honest per-file diagnostics: a required file refused at admission names
    the REMAINDER it did not fit (after higher-priority content and the
    rendered manifest), never the ambiguous claim that the file alone exceeded
    the whole hard budget."""
    _write(tmp_path / "prompts" / "big_a.md", "prompt line\n" * 900)
    _write(tmp_path / "prompts" / "big_b.md", "prompt line\n" * 900)

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=("prompts/big_a.md", "prompts/big_b.md"),
            fixed_prompt_tokens=100,
            target_total_tokens=8_000,
            hard_total_tokens=9_000,
        )
    )

    rows = pack.manifest["unassembled_required"]
    assert rows, "one of the two required prompts must not fit"
    for row in rows:
        assert "does not fit the atlas hard budget" in row["reason"]
        assert "remain after higher-priority content and the rendered manifest" in row["reason"]


def test_atlas_compact_manifest_is_the_default_and_collapses_excluded_rows(tmp_path):
    """Approved with the #284 fixes: compact coverage is the default prompt
    form, and policy-excluded classes collapse to per-directory count rows in
    the compact index (the durable manifest keeps every per-path row)."""
    assert ReviewContextAtlasRequest(repo_dir=tmp_path).compact_manifest is True

    _write(tmp_path / "app.py", "import helper\n\nprint(helper.VALUE)\n")
    _write(tmp_path / "helper.py", "VALUE = 42\n")
    for idx in range(4):
        _write(tmp_path / "tests" / f"test_mod_{idx}.py", "def test_ok():\n    assert True\n")
    _write(tmp_path / "assets" / "logo.png", "not really a png")

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path,
            tracked_paths=(
                "app.py",
                "helper.py",
                "tests/test_mod_0.py",
                "tests/test_mod_1.py",
                "tests/test_mod_2.py",
                "tests/test_mod_3.py",
                "assets/logo.png",
            ),
            anchors=("app.py",),
            fixed_prompt_tokens=100,
            target_total_tokens=20_000,
            hard_total_tokens=25_000,
        )
    )

    # Excluded classes are collapsed to one per-directory row in the index…
    assert "excluded_test\ttests/ (4 files)" in pack.text
    assert "excluded_dir\tassets/ (1 files)" in pack.text  # assets/ is a skip-dir class
    assert "\ttests/test_mod_0.py" not in pack.text
    # …reviewable rows keep their per-path lines…
    assert "\tapp.py" in pack.text
    assert "\thelper.py" in pack.text
    # …and the durable manifest still carries every per-path coverage row.
    assert {row["path"] for row in pack.manifest["coverage"]} == {
        "app.py",
        "helper.py",
        "tests/test_mod_0.py",
        "tests/test_mod_1.py",
        "tests/test_mod_2.py",
        "tests/test_mod_3.py",
        "assets/logo.png",
    }


def test_scope_ladder_never_hands_required_beyond_diff_paths_to_diff_only(tmp_path):
    """The self-defeating rung is removed: the atlas refuses a diff-only
    required artifact by design, so `_degradable_diff_only_paths` must never
    offer an atlas-required-beyond-diff path to the diff-only tier."""
    from ouroboros.tools import scope_review as sr

    _write(tmp_path / "prompts" / "SYSTEM.md", "system prompt\n")
    _write(tmp_path / "ouroboros" / "mod.py", "x = 1\n")
    _write(tmp_path / "docs" / "ARCHITECTURE.md", "arch\n")

    degradable = sr._degradable_diff_only_paths(
        tmp_path,
        ["prompts/SYSTEM.md", "ouroboros/mod.py", "docs/ARCHITECTURE.md"],
        [],
        [],
    )

    assert "ouroboros/mod.py" in degradable
    assert "prompts/SYSTEM.md" not in degradable
    assert "docs/ARCHITECTURE.md" not in degradable


def test_canonical_context_docs_membership_includes_design():
    """Drift guard for the three synchronized canonical-doc lists: the atlas
    copy must carry docs/DESIGN.md like scope_review's tuple and the external
    substrate set (fable p0-review note, 2026-08-31)."""
    from ouroboros.tools.review_context_atlas import _CANONICAL_CONTEXT_DOCS

    assert "docs/DESIGN.md" in _CANONICAL_CONTEXT_DOCS


def test_atlas_diff_only_reason_override_is_the_callers_typed_omission(tmp_path):
    """A `diff_only_included` row omitted BY DESIGN (the scope pack's span-only
    release carriers) carries the caller's reason instead of the budget one —
    same disposition, same selection; the required-artifact escalation ignores
    the override, so a by-design reason can never launder a refusal."""
    _write(tmp_path / "uv.lock", "lock\n" * 50)
    _write(tmp_path / "module.py", "q = 1\n" * 50)
    _write(tmp_path / "prompts" / "touched.md", "p\n" * 50)
    every = ("uv.lock", "module.py", "prompts/touched.md")

    pack = compile_review_context_atlas(
        ReviewContextAtlasRequest(
            repo_dir=tmp_path, tracked_paths=every, anchors=every,
            already_included=frozenset(every), diff_only_included=frozenset(every),
            diff_only_reasons={"uv.lock": "BY-DESIGN-REASON", "prompts/touched.md": "BY-DESIGN-REASON"},
        )
    )

    coverage = _coverage(pack)
    assert coverage["uv.lock"]["disposition"] == "already_included"
    assert coverage["uv.lock"]["reason"] == "BY-DESIGN-REASON"
    assert "full snapshot omitted" in coverage["module.py"]["reason"]
    assert coverage["prompts/touched.md"]["disposition"] == "budget_omitted"
    assert [row["path"] for row in atlas_unassembled_required(pack.manifest)] == ["prompts/touched.md"]
