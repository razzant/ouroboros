"""The guaranteed-fit ladder degrades the pack honestly.

Split by theme out of the original ``tests/test_scope_review.py`` giant. This
module owns the degradation ladder: what may be degraded to diff-only and in
which order, what never may (canonical docs, required artifacts, deleted
non-tests), the oversize terminals that name their causes, and the aggregated
ladder-step record.
"""

import subprocess

import pytest

def test_ladder_steps_are_recorded_once_aggregated(tmp_path, monkeypatch):
    """RS5: the guaranteed-fit ladder leaves ONE aggregated field in the existing
    context manifest — not an event per step, and not silence."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(scope_pack, "run_cmd", lambda cmd, cwd=None: (
        "M\ta.py" if "--name-status" in cmd else "diff --git a/a.py b/a.py\n+x = 1\n"
    ))
    monkeypatch.setattr(scope_pack, "capture_staged_diff",
                        lambda _repo, **_k: "diff --git a/a.py b/a.py\n+x = 1\n")
    monkeypatch.setattr(scope_pack, "_gather_scope_packs", lambda *a, **k: "ATLAS")
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_k: 900_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None and prompt
    manifest = sr._current_scope_context_manifest()
    steps = manifest.get("ladder_steps")
    assert isinstance(steps, list) and len(steps) == 1
    assert steps[0]["step"] == "full_atlas"
    assert set(steps[0]) >= {"tokens_before", "tokens_after", "diff_only_files", "deficit"}


def _repo_with_oversized_required_prompt(tmp_path, required_bytes=935_000):
    """A repo whose UNCHANGED `prompts/` artifact cannot fit any atlas budget."""
    import subprocess

    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    (tmp_path / "docs" / "DEVELOPMENT.md").write_text("dev guide\n", encoding="utf-8")
    (tmp_path / "BIBLE.md").write_text("constitution\n", encoding="utf-8")
    (tmp_path / "prompts").mkdir(exist_ok=True)
    # Force-included by prefix => `required`, and never touched by this commit.
    (tmp_path / "prompts" / "huge.md").write_text("x" * required_bytes, encoding="utf-8")
    (tmp_path / "ok.py").write_text("print(1)\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    (tmp_path / "ok.py").write_text("print(2)\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)


def test_unassembled_required_terminal_names_the_artifact_not_a_phantom_overflow(
    tmp_path, monkeypatch,
):
    """BIBLE P1/P3. The ladder terminates on TWO different failures. When it ends
    because a REQUIRED artifact never assembled, the owner-facing block must say
    so — not reuse the irreducible-prompt story, whose own quoted token count
    contradicts the budget it claims to exceed, and whose remedy ("split the
    staged diff") cannot shrink an UNCHANGED artifact. The refusal is also a
    ladder STEP: a terminal with an empty trace explains nothing after the fact."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    _repo_with_oversized_required_prompt(tmp_path)
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 200_000)
    for _module in (sr, scope_pack):  # the ladder terminal and the block message read it
        monkeypatch.setattr(_module, "_scope_window",
                            lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert prompt is None
    assert status.status == "fixed_overflow"          # authority branch unchanged
    assert status.unassembled_required == ["prompts/huge.md"]  # cause carried
    # The trace records the refusal steps, naming what did not assemble.
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    assert [s for s in steps if s["step"] == "atlas_refused"], steps
    assert steps[0]["unassembled_required"] == ["prompts/huge.md"]

    result = sr._handle_prompt_signals(prompt, status, input_limit=200_000, scope_model="")
    assert "prompts/huge.md" in result.block_message
    # The false cause and its self-contradicting comparison are gone.
    assert "irreducible scope prompt" not in result.block_message
    assert f"({200_000})" not in result.block_message
    assert "Split the commit into smaller staged diffs" not in result.block_message


def test_sub_floor_terminal_reports_the_same_cause_as_the_1m_terminal(monkeypatch):
    """The twin: `budget_exceeded` (sub-floor reviewer) is the other authority
    branch of the same terminal and made the identical false claim. Fixing one
    branch and leaving its sibling is the defect, not the fix. The genuine
    overflow wording must survive on both."""
    from ouroboros.tools import scope_review as sr

    monkeypatch.setattr(
        sr, "_scope_window", lambda _m, **_k: sr.ReviewerWindow(window_tokens=200_000, status="confirmed"),
    )
    missing = sr._TouchedContextStatus(
        status="budget_exceeded", token_count=3_672,
        unassembled_required=["prompts/huge.md"],
    )
    result = sr._handle_prompt_signals(None, missing, input_limit=200_000, scope_model="m/x")
    assert "prompts/huge.md" in result.block_message
    assert "prompts/huge.md" in result.advisory_findings[0]["reason"]
    assert "cannot fit the irreducible scope prompt" not in result.block_message
    assert "Full scope-review prompt" not in result.advisory_findings[0]["reason"]

    # The half that must NOT change: a real overflow still reports an overflow.
    overflow = sr._TouchedContextStatus(status="budget_exceeded", token_count=990_000)
    plain = sr._handle_prompt_signals(None, overflow, input_limit=200_000, scope_model="m/x")
    assert "irreducible scope prompt" in plain.block_message
    assert "~990000 estimated tokens" in plain.advisory_findings[0]["reason"]


def test_mixed_terminal_reports_both_causes_and_the_mixed_remedy(tmp_path, monkeypatch):
    """The MIXED terminal: the refusal that dropped a required artifact was itself
    a hard-budget overflow (even the content-free manifest did not fit beside the
    fixed prompt). Reporting only the missing artifact prescribes
    ATLAS_MISSING_ARTIFACT_REMEDY — "narrowing the reviewed change cannot help" —
    which is false for, and cannot resolve, the overflow half. Both causes ride
    the terminal, the trace, and the owner-facing block."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack
    from ouroboros.tools.review_context_atlas import ATLAS_MIXED_ASSEMBLY_REMEDY

    _repo_with_oversized_required_prompt(tmp_path)
    # An input budget so small the atlas hard allowance is zero: ANY rendered
    # manifest overflows, while required prompts/huge.md was already dropped.
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 6_000)
    for _module in (sr, scope_pack):  # the ladder terminal and the block message read it
        monkeypatch.setattr(_module, "_scope_window",
                            lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert prompt is None
    assert status.status == "fixed_overflow"          # authority branch unchanged
    assert status.unassembled_required == ["prompts/huge.md"]  # cause 1 carried
    assert status.atlas_overflowed is True                     # cause 2 carried
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    refused = [s for s in steps if s["step"] == "atlas_refused"]
    assert refused, steps
    assert refused[0]["atlas_overflowed"] is True

    result = sr._handle_prompt_signals(prompt, status, input_limit=6_000, scope_model="")
    # Both causes are rendered — neither shadows the other…
    assert "prompts/huge.md" in result.block_message
    assert "content-free atlas manifest" in result.block_message
    # …and the remedy is the mixed one, not the single-cause half-truth that
    # cannot resolve the overflow.
    assert ATLAS_MIXED_ASSEMBLY_REMEDY in result.block_message
    assert "narrowing the reviewed change cannot help" not in result.block_message


def test_mixed_sub_floor_terminal_reports_the_same_two_causes(monkeypatch):
    """The sub-floor authority branch is the twin surface of the same terminal:
    it must render the identical mixed cause and remedy."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools.review_context_atlas import ATLAS_MIXED_ASSEMBLY_REMEDY

    monkeypatch.setattr(
        sr, "_scope_window", lambda _m, **_k: sr.ReviewerWindow(window_tokens=200_000, status="confirmed"),
    )
    mixed = sr._TouchedContextStatus(
        status="budget_exceeded", token_count=3_672,
        unassembled_required=["prompts/huge.md"], atlas_overflowed=True,
    )
    result = sr._handle_prompt_signals(None, mixed, input_limit=200_000, scope_model="m/x")
    for text in (result.block_message, result.advisory_findings[0]["reason"]):
        assert "prompts/huge.md" in text
        assert "content-free atlas manifest" in text
    assert ATLAS_MIXED_ASSEMBLY_REMEDY in result.advisory_findings[0]["reason"]
    assert "narrowing the reviewed change cannot help" not in result.advisory_findings[0]["reason"]


def test_diff_only_degradation_is_not_reported_as_fully_included(tmp_path, monkeypatch):
    """P1. When the ladder drops a touched file's full snapshot, the durable
    coverage manifest must say so. `already_included` was re-derived from ALL
    touched paths instead of the surviving `kept` set that `_render_touched_section`
    owns, so a file whose snapshot had just been removed was still recorded as
    "included in fixed prompt context" — a claim the prompt itself contradicts."""
    import subprocess

    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    # Two equal-sized big files: the ladder degrades only the first.
    (tmp_path / "big_a.py").write_text("x = 1\n" * 40_000, encoding="utf-8")
    (tmp_path / "big_b.py").write_text("y = 1\n" * 40_000, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    # Tiny change inside huge files: the diff fits, the snapshots do not.
    (tmp_path / "big_a.py").write_text("z = 0\n" + "x = 1\n" * 39_999, encoding="utf-8")
    (tmp_path / "big_b.py").write_text("z = 0\n" + "y = 1\n" * 39_999, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 120_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None and prompt is not None
    assert "TOUCHED FILE BUDGET DEGRADATION NOTE" in prompt and "- big_a.py" in prompt
    assert "### big_b.py" in prompt  # this one kept its snapshot
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    # The degraded file is disclosed as diff-only, the intact one is not.
    assert "full snapshot omitted" in rows["big_a.py"]["reason"]
    assert rows["big_b.py"]["reason"] == "included in fixed prompt context"


def test_design_skipped_touched_test_is_not_claimed_as_fully_included(tmp_path):
    """XG-1R4.1 / P1. `_gather_scope_packs` used to derive `already_included` from
    the touched LIST (`all_touched_paths`), while `_build_scope_prompt` omits the
    full snapshots of touched TESTS by design (`current_skipped_by_design`). The
    durable row for a touched test therefore read "included in fixed prompt
    context" while NO full snapshot of it existed anywhere in the pack — the same
    false-coverage-claim class as XR-4/XG-1R.4, on the last surface still
    deriving the claim instead of being told it.

    `already_included` is now the CONSERVATIVE set the fixed part really carries,
    so the touched test falls through to the atlas, where (being an anchor, hence
    related to the change and not excludable under BIBLE P3) it is supplied in
    FULL exactly once. FULL is the spacious-budget DEFAULT, not a guarantee:
    under budget pressure the guaranteed-fit ladder may degrade a touched test
    to diff-only (the constrained sibling below) — constitutionally sound
    because the test's complete changes ride the staged diff. The invariant
    under test is the general one: a coverage row may never claim content the
    pack does not contain."""
    import subprocess

    from ouroboros.tools import scope_review as sr

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    # An INDENTED marker far from the change: unreachable through the staged diff
    # (outside any -U3 hunk, and never picked as git's hunk-header funcname), so
    # finding it in the prompt proves a real full snapshot.
    body = ["def test_a():", "    UNCHANGED_TEST_BODY_MARKER_ZZZ = 1"]
    body += [f"    filler_{idx} = {idx}" for idx in range(40)]
    body += ["    assert True"]
    (tmp_path / "tests" / "test_thing.py").write_text(
        "\n".join(body) + "\n", encoding="utf-8",
    )
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    (tmp_path / "tests" / "test_thing.py").write_text(
        "\n".join(body) + "\n\n\ndef test_b():\n    assert True\n", encoding="utf-8",
    )
    (tmp_path / "mod.py").write_text("x = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None and prompt
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    row = rows["tests/test_thing.py"]
    # The false claim is gone…
    assert row["reason"] != "included in fixed prompt context"
    assert row["disposition"] != "already_included"
    # …and the pack really carries what the row now says: the atlas supplied the
    # touched test in full, so the unchanged body reached the reviewer.
    assert row["disposition"] == "full"
    assert "UNCHANGED_TEST_BODY_MARKER_ZZZ" in prompt
    # The dedup note still explains the non-duplication in the fixed part.
    assert "DEDUPLICATION NOTE" in prompt and "- tests/test_thing.py" in prompt
    # The touched non-test keeps its true `already_included` claim.
    assert rows["mod.py"]["reason"] == "included in fixed prompt context"
    # Spacious budget: the ladder never reached for the test — no degradation.
    assert "TOUCHED FILE BUDGET DEGRADATION NOTE" not in prompt


def _ladder_repo(tmp_path, files: dict, changes: dict):
    """Init a git repo with ``files``, then stage ``changes``.

    Values: str -> text file, bytes -> binary file, None -> delete the path
    (``git add .`` stages removals too)."""
    import subprocess

    def _put(rel, content):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            path.unlink()
        elif isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")

    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    for rel, content in files.items():
        _put(rel, content)
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    for rel, content in changes.items():
        _put(rel, content)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)


# ~90K estimated tokens of test body; the INDENTED marker is unreachable through
# the staged diff (outside any -U3 hunk of an end-of-file change, and never a
# hunk-header funcname), so its absence from the prompt proves the full
# snapshot is really gone from the whole pack.
_BIG_TEST_BODY = "\n".join(
    ["def test_big():", "    UNCHANGED_BIG_TEST_MARKER_QQQ = 1"]
    + ["    filler = 1"] * 24_000
    + ["    assert True"]
) + "\n"
_BIG_TEST_CHANGED = _BIG_TEST_BODY + "\n\ndef test_added():\n    assert True\n"


def test_constrained_budget_degrades_touched_test_to_diff_only(tmp_path, monkeypatch):
    """Phase L, the constrained sibling of the spacious pin above. A touched
    test is filtered out of the fixed part's snippets by design, yet the atlas
    is owed it as a FULL anchor — and the ladder built its degradable set only
    from `current_context_paths`, so one oversized touched test structurally
    sank pack assembly with `required_artifact_omitted` although its complete
    changes sat in the staged diff. Touched tests now ride the ladder's free
    tier: under pressure they degrade to diff-only via the existing
    `diff_only_included` mechanism, assembly SUCCEEDS, and the manifest row
    carries the disclosure."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    _ladder_repo(
        tmp_path,
        files={"tests/test_big.py": _BIG_TEST_BODY, "mod.py": "x = 1\n"},
        changes={"tests/test_big.py": _BIG_TEST_CHANGED, "mod.py": "x = 2\n"},
    )
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    # Assembly SUCCEEDS — the oversized touched test no longer sinks the pack.
    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The ladder degraded the test, disclosed in the prompt…
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_big.py" in note
    # …and the full snapshot is truly gone from the whole pack.
    assert "UNCHANGED_BIG_TEST_MARKER_QQQ" not in prompt
    # The dedup note no longer lists it: that would claim an atlas snapshot.
    assert "CURRENT FILE CONTEXT DEDUPLICATION NOTE" not in prompt
    # The durable coverage row carries the diff-only disclosure — not a false
    # full-inclusion claim, not a required omission.
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    row = rows["tests/test_big.py"]
    assert row["disposition"] == "already_included"
    assert "changes included" in row["reason"]
    assert "full snapshot omitted" in row["reason"]
    # The small touched module keeps its true full-snapshot claim.
    assert rows["mod.py"]["reason"] == "included in fixed prompt context"


def test_touched_test_degrades_before_the_required_tier_and_zero_context_diff(
    tmp_path, monkeypatch,
):
    """Ordering. Touched tests join the FREE tier of the two-tier ladder sort
    (`atlas_required_beyond_diff` is False for tests/), so under deficit the
    big touched test degrades BEFORE the ladder reaches for the -U0 rung or
    the required tier: the required-beyond-diff artifact keeps its full
    snapshot in the fixed part and no zero-context-diff step is recorded."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    _ladder_repo(
        tmp_path,
        files={
            "tests/test_big.py": _BIG_TEST_BODY,
            # ~10K tokens: fits the fixed part; owed in full regardless of size.
            "prompts/mini_prompt.md": "word here\n" * 4_000,
        },
        changes={
            "tests/test_big.py": _BIG_TEST_CHANGED,
            "prompts/mini_prompt.md": "CHANGED\n" + "word here\n" * 3_999,
        },
    )
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The required-beyond-diff artifact kept its full snapshot in the fixed part…
    assert "### prompts/mini_prompt.md" in prompt
    # …the degradation note names the test, and only the test…
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_big.py" in note
    assert "prompts/mini_prompt.md" not in note
    # …and the ladder never needed the -U0 rung: the free tier covered it.
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    assert steps and not any(s.get("zero_context_diff") for s in steps), steps
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["prompts/mini_prompt.md"]["reason"] == "included in fixed prompt context"
    assert "full snapshot omitted" in rows["tests/test_big.py"]["reason"]


def test_canonical_doc_is_never_ladder_degraded_to_diff_only(tmp_path, monkeypatch):
    """The boundary of Phase L: only tests/ paths joined the degradable set. A
    touched CANONICAL doc is owed in full through the fixed part's
    canonical-docs section (`atlas_required_beyond_diff` is True), so when it
    alone overflows the budget the ladder exhausts its free rungs (including
    -U0) and fails CLOSED — it never hands the doc to diff-only."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    _ladder_repo(
        tmp_path,
        files={
            # ~84K tokens injected whole into the canonical-docs section.
            "docs/ARCHITECTURE.md": "arch doc line\n" * 24_000,
            "mod.py": "x = 1\n",
        },
        changes={
            "docs/ARCHITECTURE.md": "CHANGED\n" + "arch doc line\n" * 23_999,
            "mod.py": "x = 2\n",
        },
    )
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 45_000)
    monkeypatch.setattr(scope_pack, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert prompt is None
    assert status.status == "fixed_overflow"
    # The canonical doc never became a required omission via diff-only…
    assert status.unassembled_required == []
    assert status.atlas_overflowed is True
    # …its coverage row keeps the truthful fixed-part claim…
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["docs/ARCHITECTURE.md"]["reason"] == "included in fixed prompt context"
    # …and the ladder really exhausted the free rungs (-U0 attempted) first.
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    assert any(s.get("zero_context_diff") for s in steps), steps


def test_binary_test_fixture_is_never_degraded_to_diff_only(tmp_path, monkeypatch):
    """A binary fixture under tests/ has NO changes in the staged text diff
    (`git diff --cached` renders "Binary files differ"), so the diff-only
    disclosure "changes included in the fixed staged diff" would be a false
    claim. Binary staged test paths (`staged_path_is_binary`) stay out of the
    degradable tier; the fixture remains the atlas's business (typed
    `binary_media` row). The fixture makes the binary the LARGEST touched test,
    so a candidates list without the binary filter would degrade the binary
    first and fail the note assertions below."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    _ladder_repo(
        tmp_path,
        files={
            "tests/fixture.bin": bytes(range(256)) * 1_600,  # ~400KB, biggest
            "tests/test_big.py": _BIG_TEST_BODY,
        },
        changes={
            "tests/fixture.bin": b"\x00CHANGED" + bytes(range(256)) * 1_600,
            "tests/test_big.py": _BIG_TEST_CHANGED,
        },
    )
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The TEXT test rode the diff-only rung; the binary fixture did not.
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_big.py" in note
    assert "tests/fixture.bin" not in note
    # The binary fixture stays honestly delegated to the atlas (typed row,
    # never the diff-only "changes included" claim).
    dedup = prompt.split("## CURRENT FILE CONTEXT DEDUPLICATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/fixture.bin" in dedup
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["tests/fixture.bin"]["disposition"] == "binary_media"
    assert "full snapshot omitted" not in rows["tests/fixture.bin"]["reason"]
    assert "full snapshot omitted" in rows["tests/test_big.py"]["reason"]


def test_deleted_text_test_degrades_to_diff_only_under_pressure(tmp_path, monkeypatch):
    """A deleted TEXT test inlines its whole HEAD snapshot into the fixed part
    (`_inline_deleted_file_pack`) and the ladder had no rung for it — pure
    pressure with no relief, although a text deletion's complete content is
    already the staged diff's own minus-lines. Deleted text tests now join the
    same degradable tier: under pressure the HEAD inline is replaced by a
    disclosed omission marker. Binary deletions never qualify (their content is
    not in the text diff). NB: mod.py co-degrades here because the
    refusal-branch deficit has a pre-existing 50K floor that outsizes the
    deleted test — a ladder property, not an effect of this fix."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    # ~30K tokens of deleted test: inline + its diff minus-lines both ride the
    # fixed part (~65K total), overflowing the 45K budget; dropping the inline
    # (~30K) brings the prompt back under it.
    gone_body = "\n".join(
        ["def test_gone():", "    DELETED_TEST_HEAD_MARKER_WWW = 1"]
        + ["    filler = 1"] * 8_000
        + ["    assert True"]
    ) + "\n"
    _ladder_repo(
        tmp_path,
        files={
            "tests/test_gone.py": gone_body,
            "tests/gone.bin": bytes(range(256)) * 4,  # small binary deletion
            "mod.py": "x = 1\n",
        },
        changes={
            "tests/test_gone.py": None,
            "tests/gone.bin": None,
            "mod.py": "x = 2\n",
        },
    )
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The deleted text test was degraded: no HEAD inline, disclosed marker
    # instead, and the degradation note names it.
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_gone.py" in note
    assert "full HEAD snapshot omitted" in prompt
    assert "*(DELETED — content from HEAD)*" not in prompt
    # The binary deletion did NOT ride the diff-only rung: it keeps its own
    # typed suppression marker (its content is not in the text diff).
    assert "tests/gone.bin" not in note
    assert "### tests/gone.bin\n\n*(DELETED — " in prompt
    assert "content suppressed" in prompt


def test_degraded_test_gets_no_false_atlas_delegation_phrase(tmp_path, monkeypatch):
    """The dedup note used to promise unconditionally that a touched test's
    "full snapshot appears once in the generated atlas" — false the moment the
    ladder degrades that test to diff-only. The reviewer-facing phrase is now
    conditional and per-file: full-delegated tests stay listed in the dedup
    note, budget-degraded ones move to the degradation note, and the
    unconditional phrase is gone."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    _ladder_repo(
        tmp_path,
        files={
            "tests/test_huge.py": _BIG_TEST_BODY,
            "tests/test_tiny.py": "def test_tiny():\n    TINY_KEPT_MARKER_JJJ = 1\n",
        },
        changes={
            "tests/test_huge.py": _BIG_TEST_CHANGED,
            "tests/test_tiny.py": (
                "def test_tiny():\n    TINY_KEPT_MARKER_JJJ = 1\n    assert True\n"
            ),
        },
    )
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The unconditional promise is gone from the reviewer-facing prompt…
    assert "appears once in the generated atlas" not in prompt
    # …the conditional wording rides the dedup note, which lists ONLY the
    # test still delegated in full…
    dedup = prompt.split("## CURRENT FILE CONTEXT DEDUPLICATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_tiny.py" in dedup
    assert "tests/test_huge.py" not in dedup
    assert "move to the degradation note instead" in prompt
    # …and each file's actual disposition backs the wording: tiny delegated in
    # full by the atlas, huge disclosed as diff-only.
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_huge.py" in note
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["tests/test_tiny.py"]["disposition"] == "full"
    assert "full snapshot omitted" in rows["tests/test_huge.py"]["reason"]


def test_deleted_non_test_file_is_never_degraded(tmp_path, monkeypatch):
    """Boundary pin for the deleted branch: ONLY tests/ deletions join the
    degradable tier. A deleted ordinary module keeps its HEAD inline even under
    pressure (here it alone overflows the budget, so the ladder exhausts its
    rungs and fails CLOSED) — a mutation dropping the tests/-filter from the
    deleted branch would degrade it and assemble, flipping every assert below.
    The named `diff_only_paths` ladder trace proves it was never degraded."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    gone_body = "\n".join(["def helper():"] + ["    filler = 1"] * 8_000) + "\n"
    _ladder_repo(
        tmp_path,
        files={"mod_big.py": gone_body, "mod.py": "x = 1\n"},
        changes={"mod_big.py": None, "mod.py": "x = 2\n"},
    )
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 45_000)
    monkeypatch.setattr(scope_pack, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert prompt is None
    assert status.status == "fixed_overflow"
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    assert steps
    for step in steps:
        assert "mod_big.py" not in (step.get("diff_only_paths") or []), step


def test_deleted_test_token_estimate_orders_largest_first(tmp_path, monkeypatch):
    """The cat-file size fallback in `_touched_token_estimate` is load-bearing:
    deleted paths have no worktree stat, and a fallback that returned 0 would
    (a) sort every deleted test LAST instead of largest-first and (b) count 0
    freed tokens per pop, so the loop would drain the whole tier. With honest
    estimates the LARGER deleted test alone covers the deficit and the smaller
    one keeps its HEAD inline."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    big_gone = "\n".join(["def test_gone_big():"] + ["    filler = 1"] * 16_000) + "\n"
    small_gone = "\n".join(["def test_gone_small():"] + ["    filler = 1"] * 2_100) + "\n"
    _ladder_repo(
        tmp_path,
        files={
            "tests/test_gone_big.py": big_gone,
            "tests/test_gone_small.py": small_gone,
            "mod.py": "x = 1\n",
        },
        changes={
            "tests/test_gone_big.py": None,
            "tests/test_gone_small.py": None,
            "mod.py": "x = 2\n",
        },
    )
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 135_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # Guarded extraction: a missing note must fail as an assert, not IndexError.
    assert "## TOUCHED FILE BUDGET DEGRADATION NOTE" in prompt
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    # Largest-first: the big deleted test degraded, the small one did not.
    assert "- tests/test_gone_big.py" in note
    assert "tests/test_gone_small.py" not in note
    assert "### tests/test_gone_big.py\n\n*(DELETED — full HEAD snapshot omitted" in prompt
    assert "### tests/test_gone_small.py\n\n*(DELETED — content from HEAD)*" in prompt


def test_oversized_deleted_test_keeps_suppressed_marker_not_diff_only(
    tmp_path, monkeypatch,
):
    """A deleted test over `_DELETED_INLINE_MAX_BYTES` is ALREADY a suppressed
    marker — its inline never weighed on the budget, so degrading it would free
    phantom tokens (the HEAD-blob estimate, ~277K here) and misattribute the
    relief. The size guard keeps it out of the degradable tier: pressure is
    relieved by the genuine candidate (the big CURRENT test), and the oversized
    deletion keeps its own typed suppression marker."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    huge_gone = "\n".join(["def test_huge_gone():"] + ["    filler = 1"] * 74_000) + "\n"
    assert len(huge_gone.encode()) > 1_048_576  # over the inline cap
    _ladder_repo(
        tmp_path,
        files={"tests/test_huge_gone.py": huge_gone, "tests/test_big.py": _BIG_TEST_BODY},
        changes={"tests/test_huge_gone.py": None, "tests/test_big.py": _BIG_TEST_CHANGED},
    )
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 330_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The oversized deletion kept its typed suppression marker…
    assert "content > 1024 KB; suppressed" in prompt
    assert "## TOUCHED FILE BUDGET DEGRADATION NOTE" in prompt
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    # …the genuine candidate carried the degradation, not the phantom one.
    assert "- tests/test_big.py" in note
    assert "tests/test_huge_gone.py" not in note
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    for step in steps:
        assert "tests/test_huge_gone.py" not in (step.get("diff_only_paths") or []), step


def test_a_renamed_test_fixture_is_not_degraded(tmp_path, monkeypatch):
    """Conservative rename guard: a renamed path's staged diff may carry only a
    rename header (no content hunks), so degrading it to diff-only could hide
    its content entirely. Renamed touched tests keep their snapshot; the plain
    modified test still rides the diff-only rung."""
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    _ladder_repo(
        tmp_path,
        files={"tests/old_name.bin": bytes(range(256)) * 1_600,
               "tests/test_big.py": _BIG_TEST_BODY},
        changes={"tests/test_big.py": _BIG_TEST_CHANGED},
    )
    subprocess.run(["git", "mv", "tests/old_name.bin", "tests/new_name.bin"],
                   cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 45_000)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    assert "- tests/test_big.py" in note
    assert "new_name.bin" not in note and "old_name.bin" not in note


def test_staged_diff_capture_survives_non_utf8_text(tmp_path, monkeypatch):
    """Git calls NUL-free non-UTF-8 content TEXT, so those bytes ride ordinary diff
    lines. The old strict-UTF-8 text capture raised on them and the review continued
    on a "(failed to get staged diff)" placeholder — with the ladder able to degrade
    a touched test to diff-only, that placeholder can be a file's only evidence. The
    bytes now arrive reversibly escaped and the pack assembles."""
    from ouroboros.tools import scope_review as sr

    _ladder_repo(tmp_path, files={"tests/test_bytes.py": "x = 1\n"}, changes={"mod.py": "y = 1\n"})
    (tmp_path / "tests" / "test_bytes.py").write_bytes(b"x = 1  # latin caf\xe9\n")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert status is None, getattr(status, "unassembled_required", status)
    assert "\\xe9" in prompt  # reversible escape, not a U+FFFD flattening
    assert "�" not in prompt
    assert "failed to get staged diff" not in prompt


def test_unavailable_staged_diff_blocks_instead_of_reviewing_a_placeholder(tmp_path, monkeypatch):
    """When the canonical staged diff cannot be captured at all, prompt assembly
    fails with a RuntimeError — the type `scope_review`'s own caller already turns
    into a blocked, fail-closed result — instead of sending an authoritative review
    a placeholder that says the evidence is missing."""
    from ouroboros.tools import review_binary_context as rbc
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    _ladder_repo(tmp_path, files={"mod.py": "x = 1\n"}, changes={"mod.py": "x = 2\n"})

    def broken(*_a, **_k):
        raise rbc.StagedDiffUnavailable("staged diff capture failed (rc 128): fatal")

    monkeypatch.setattr(scope_pack, "capture_staged_diff", broken)

    assert issubclass(rbc.StagedDiffUnavailable, RuntimeError)
    with pytest.raises(RuntimeError):
        sr._build_scope_prompt(tmp_path, "test commit")


def test_ladder_cannot_degrade_a_required_beyond_diff_artifact_to_diff_only(
    tmp_path, monkeypatch,
):
    """XR-4 end-to-end. The guaranteed-fit ladder degrades the LARGEST touched
    files to diff-only; when that file is an artifact owed in full regardless of
    the change (here a `prompts/` file), the atlas used to accept the declared
    drop without a typed failure and scope review PROCEEDED with the prompt's
    full snapshot in neither the fixed prompt nor the atlas. Now the atlas
    refuses (BIBLE P3), the refusal is a recorded ladder step, and the terminal
    names the artifact — review does not proceed on the remainder."""
    import subprocess

    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    (tmp_path / "prompts").mkdir()
    # The prompts/ artifact is the LARGEST touched file: degraded first.
    (tmp_path / "prompts" / "big_prompt.md").write_text(
        "word here\n" * 45_000, encoding="utf-8",
    )
    (tmp_path / "big_b.py").write_text("y = 1\n" * 40_000, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    # Tiny changes inside huge files: the diff fits, the snapshots do not.
    (tmp_path / "prompts" / "big_prompt.md").write_text(
        "CHANGED\n" + "word here\n" * 44_999, encoding="utf-8",
    )
    (tmp_path / "big_b.py").write_text("z = 0\n" + "y = 1\n" * 39_999, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 120_000)
    monkeypatch.setattr(scope_pack, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    assert prompt is None
    assert status.status == "fixed_overflow"          # authority branch unchanged
    assert status.unassembled_required == ["prompts/big_prompt.md"]
    steps = sr._current_scope_context_manifest().get("ladder_steps") or []
    refused = [s for s in steps if s["step"] == "atlas_refused"]
    assert refused, steps
    # The refusal naming the artifact is a recorded ladder step (the FIRST
    # refusal may be the pre-degradation hard-budget one with no named rows).
    assert any(
        s["unassembled_required"] == ["prompts/big_prompt.md"] for s in refused
    ), refused
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["prompts/big_prompt.md"]["disposition"] == "budget_omitted"
    # An ORDINARY touched file may still ride the disclosed diff-only step
    # (pinned by test_diff_only_degradation_is_not_reported_as_fully_included).
    # ORDERING: the tier sort only reorders, so the pop loop must also refuse to cross
    # into the required tier until the zero-context rung has been tried — degrading a
    # required artifact provably cannot buy a fitting pack, while -U0 still might. Every
    # recorded step that already shows the artifact degraded must show -U0 attempted.
    for step in steps:
        if step.get("unassembled_required") and step.get("diff_only_files"):
            assert step["zero_context_diff"] is True, step
    assert any(step.get("zero_context_diff") for step in steps), steps


def test_ladder_degrades_ordinary_files_before_a_required_artifact(tmp_path, monkeypatch):
    """Defect B. The ladder sorted ALL touched paths by size with no requiredness
    filter, so the LARGEST file was degraded first even when it was an artifact
    owed in full. Degrading one of those can never buy a fitting pack — the atlas
    turns it into an assembly refusal (`required_artifact_omitted`), which the
    ladder then reads as a further deficit and degrades further. The deficit was
    manufactured: the ordinary files alone covered it.

    The fixture is deliberately shaped so a size-only sort CANNOT pass it: the
    required artifact is the LARGEST touched file (~40K tokens), while the three
    ordinary files are individually smaller (~20K each) but collectively cover
    the deficit. Pre-fix this reaches the terminal with
    `status == "fixed_overflow"` and `unassembled_required ==
    ["prompts/large_prompt.md"]`; post-fix the three ordinary files degrade, the
    prompt's full snapshot survives, and the pack assembles."""
    import subprocess

    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_pack as scope_pack

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "CHECKLISTS.md").write_text(
        "## Intent / Scope Review Checklist\n\nplaceholder\n", encoding="utf-8",
    )
    (tmp_path / "prompts").mkdir()
    # ~40K touched tokens: the LARGEST touched file, and owed in full.
    (tmp_path / "prompts" / "large_prompt.md").write_text(
        "word here\n" * 16_000, encoding="utf-8",
    )
    ordinary = ["a_mod.py", "b_mod.py", "c_mod.py"]
    for name in ordinary:
        # ~20K touched tokens each: individually smaller than the artifact,
        # together (~60K) more than the ~50K deficit.
        (tmp_path / name).write_text("y = 1\n" * 13_334, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True,
    )
    # Tiny changes inside big files: the diff fits, the snapshots do not.
    (tmp_path / "prompts" / "large_prompt.md").write_text(
        "CHANGED\n" + "word here\n" * 15_999, encoding="utf-8",
    )
    for name in ordinary:
        (tmp_path / name).write_text("z = 0\n" + "y = 1\n" * 13_333, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)

    monkeypatch.setattr(scope_pack, "_effective_scope_input_limit", lambda **_kw: 90_000)
    monkeypatch.setattr(scope_pack, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(window_tokens=1_000_000, status="confirmed"))

    prompt, status = sr._build_scope_prompt(tmp_path, "test commit")

    # The pack assembles: the deficit was always coverable by optional content.
    assert status is None, getattr(status, "unassembled_required", status)
    assert prompt
    # The required artifact keeps its full snapshot in the fixed part…
    assert "### prompts/large_prompt.md" in prompt
    # …and the disclosed degradation names the ordinary files, and only those.
    note = prompt.split("## TOUCHED FILE BUDGET DEGRADATION NOTE", 1)[1].split("\n\n", 1)[0]
    for name in ordinary:
        assert f"- {name}" in note, note
    assert "prompts/large_prompt.md" not in note, note
    rows = {r["path"]: r for r in sr._current_scope_context_manifest()["coverage"]}
    assert rows["prompts/large_prompt.md"]["disposition"] == "already_included"


def test_cold_start_sizes_down_and_passes_instead_of_400ing(tmp_path, monkeypatch):
    """RS4 anti-regression: with NO observation for an unknown model the cold-start
    density must make the cap SMALLER than the historical optimistic one (pack passes),
    never larger (pack draws a deterministic provider 400)."""
    from ouroboros.capability_evidence import _DENSITY_MEMO, record_token_density
    from ouroboros.tools import scope_review as sr
    from ouroboros.tools import scope_review_budget as scope_budget

    _DENSITY_MEMO.clear()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(scope_budget, "_scope_window",
                        lambda _m, **_k: sr.ReviewerWindow(1_000_000, "confirmed"))

    cold = sr._effective_scope_input_limit(scope_model="unknown/brand-new-model")
    assert 0 < cold <= sr._SCOPE_INPUT_TOKEN_LIMIT, (
        "a cold start must never be LOOSER than the historical absolute-margin cap"
    )
    # A pack sized at the cold cap still fits the reviewer's real window even at the
    # conservative density, so the first call is not a guaranteed 400.
    from ouroboros.capability_evidence import COLD_START_TOKEN_DENSITY
    assert int(cold * COLD_START_TOKEN_DENSITY) + sr._SCOPE_MAX_TOKENS <= 1_000_000

    # A later genuine measurement is what changes the number — and it is disclosed as
    # `measured` provenance rather than silently replacing an assumption.
    record_token_density(
        tmp_path, "unknown/brand-new-model", prompt_chars=4_000_000, prompt_tokens=1_100_000,
    )
    from ouroboros.capability_evidence import resolve_token_density
    assert resolve_token_density(tmp_path, "unknown/brand-new-model")[1] == "measured"
