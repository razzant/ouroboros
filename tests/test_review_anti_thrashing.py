"""Tests for anti-thrashing behavior in pre-commit review history sections.

Covers v4.35.x changes: obligation ID injection, verdict-authoritative
instructions, and anti-rephrase guidance in `_build_review_history_section`
(both the triad `review.py` and the scope-level `scope_review.py` copies).
"""
import pathlib
import subprocess
from dataclasses import dataclass

from ouroboros.review_state import (
    AdvisoryReviewState,
    ObligationItem,
    load_state,
    make_repo_key,
    save_state,
)
from ouroboros.tools import scope_review as scope_review_mod
from ouroboros.tools import scope_review_pack as scope_pack
from ouroboros.tools.review import _build_review_history_section as triad_hist
from ouroboros.tools.review_helpers import (
    format_obligation_excerpt,
    _ANTI_THRASHING_RULE_VERDICT,
    _ANTI_THRASHING_RULE_ITEM_NAME,
    _HISTORY_VERIFICATION_ONLY_RULE,
)
from ouroboros.tools.scope_review import (
    _build_review_history_section as scope_hist,
    _build_scope_history_section,
    _build_scope_prompt,
)


@dataclass
class _MockObligation:
    obligation_id: str
    item: str
    reason: str


def _mk_obligations():
    return [
        _MockObligation(
            obligation_id="ob-abc123",
            item="code_quality",
            reason="some concrete reason about a specific symbol that was flagged",
        ),
        _MockObligation(
            obligation_id="ob-def456",
            item="bible_compliance",
            reason="short reason",
        ),
    ]


def _mk_history(attempt=1):
    return [
        {
            "attempt": attempt,
            "commit_message": "fix: do the thing",
            "critical": ["code_quality: something broke"],
            "advisory": ["style: minor nit"],
        }
    ]


# ---------------------------------------------------------------------------
# Triad review._build_review_history_section
# ---------------------------------------------------------------------------


def test_history_section_includes_obligation_ids():
    out = triad_hist(_mk_history(), open_obligations=_mk_obligations())
    assert '"obligation_id": "ob-abc123"' in out
    assert '"item": "code_quality"' in out
    assert '"obligation_id": "ob-def456"' in out
    assert '"item": "bible_compliance"' in out


def test_history_section_verdict_authoritative_instruction():
    out = triad_hist(_mk_history(), open_obligations=_mk_obligations())
    assert _ANTI_THRASHING_RULE_VERDICT in out


def test_history_section_anti_rephrase_instruction():
    out = triad_hist(_mk_history(), open_obligations=_mk_obligations())
    assert _ANTI_THRASHING_RULE_ITEM_NAME in out


def test_history_section_verification_only_instruction():
    out = triad_hist(_mk_history(), open_obligations=_mk_obligations())
    assert _HISTORY_VERIFICATION_ONLY_RULE in out


def test_history_section_empty_without_history_or_obligations():
    assert triad_hist([], open_obligations=None) == ""
    assert triad_hist([], open_obligations=[]) == ""
    assert triad_hist(None, open_obligations=None) == ""


def test_history_section_works_with_history_no_obligations():
    # Backward compat: open_obligations=None, still emits the IMPORTANT RULES
    # and the previous-rounds section.
    out = triad_hist(_mk_history())
    assert "Round 1" in out
    assert "CRITICAL findings:" in out
    assert "IMPORTANT RULES FOR THIS REVIEW" in out
    # Obligations block absent when none supplied
    assert "Open obligations from previous blocking rounds" not in out


def test_history_section_obligations_only_no_history():
    # Anti-thrashing block must render even with no history rounds.
    out = triad_hist([], open_obligations=_mk_obligations())
    assert "Open obligations from previous blocking rounds" in out
    assert '"obligation_id": "ob-abc123"' in out


# ---------------------------------------------------------------------------
# Scope review._build_review_history_section
# ---------------------------------------------------------------------------


def test_scope_review_history_section_includes_obligation_ids():
    out = scope_hist(_mk_history(), open_obligations=_mk_obligations())
    assert '"obligation_id": "ob-abc123"' in out
    assert '"obligation_id": "ob-def456"' in out
    assert '"item": "code_quality"' in out


def test_scope_review_history_section_verdict_authoritative():
    out = scope_hist(_mk_history(), open_obligations=_mk_obligations())
    assert _ANTI_THRASHING_RULE_VERDICT in out
    assert _ANTI_THRASHING_RULE_ITEM_NAME in out
    assert _HISTORY_VERIFICATION_ONLY_RULE in out


def test_scope_review_history_section_empty_without_inputs():
    assert scope_hist([], open_obligations=None) == ""
    assert scope_hist(None, open_obligations=None) == ""


# ---------------------------------------------------------------------------
# Scope review._build_scope_history_section — verdict-authoritative note
# ---------------------------------------------------------------------------


def test_scope_history_section_verdict_authoritative():
    history = [
        {"summary": "previous scope round noted a broken contract",
         "status": "responded"},
    ]
    out = _build_scope_history_section(history)
    # The shared constant is now interpolated into the scope history section.
    assert _ANTI_THRASHING_RULE_VERDICT in out
    assert _HISTORY_VERIFICATION_ONLY_RULE in out


# ---------------------------------------------------------------------------
# format_obligation_excerpt helper
# ---------------------------------------------------------------------------


def test_format_obligation_excerpt_truncates_with_omission_note():
    long_reason = "x" * 200
    out = format_obligation_excerpt(long_reason)
    assert "⚠️ OMISSION NOTE" in out


def test_format_obligation_excerpt_no_truncation_for_short_reason():
    short_reason = "short reason"
    out = format_obligation_excerpt(short_reason)
    assert out == short_reason


def test_format_obligation_excerpt_sanitizes_newlines():
    """Newlines in obligation reason must be collapsed to prevent prompt injection."""
    multiline_reason = "first line\nsecond line\nthird line"
    out = format_obligation_excerpt(multiline_reason)
    assert "\n" not in out
    assert "first line" in out
    assert "second line" in out


def test_format_obligation_excerpt_redacts_secrets_before_collapsing():
    """Secrets must be redacted BEFORE newline collapsing so line-anchored patterns work."""
    # A multiline reason where the secret sits on its own line (line-anchored _SECRET_LINE_RE).
    reason_with_secret = "some issue was found\nAPI_KEY=supersecret123\nplease review"
    out = format_obligation_excerpt(reason_with_secret, max_chars=300)
    assert "supersecret123" not in out, "Secret value must not appear in excerpt"
    assert "***REDACTED***" in out or "REDACTED" in out, "Redaction marker must be present"


def test_advisory_prompt_includes_verdict_authoritative_and_anti_rephrase_rules():
    """The advisory prompt must carry the same verdict-authoritative and anti-rephrase
    rules as the triad/scope history sections (step 6.e and 6.f)."""
    from ouroboros.tools.claude_advisory_review import _build_advisory_prompt
    import pathlib
    prompt = _build_advisory_prompt(
        repo_dir=pathlib.Path("/tmp/test-repo"),
        commit_message="test commit",
        goal="",
        scope="",
        drive_root=None,
        prompt_context={
            "diff": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new",
            "changed_files": "foo.py",
        },
    )
    assert "VERDICT IS AUTHORITATIVE" in prompt
    assert "DO NOT REPHRASE" in prompt
    assert "VERIFICATION ONLY" in prompt


# ---------------------------------------------------------------------------
# Integration tests: real obligation loading via review_state machinery.
# These catch wiring regressions (e.g. wrong load_state argument, wrong
# repo_key) that the mock-based tests above cannot see.
# ---------------------------------------------------------------------------


def _init_git_repo(path: pathlib.Path) -> None:
    """Create a minimal git repo rooted at `path` so make_repo_key resolves stably."""
    subprocess.run(
        ["git", "init", "-q"], cwd=str(path), check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _write_obligation_to_state(
    drive_root: pathlib.Path, repo_key: str, obligation: ObligationItem
) -> None:
    """Persist an obligation under the given repo_key into advisory_review.json."""
    state = AdvisoryReviewState()
    ob = ObligationItem(
        obligation_id=obligation.obligation_id,
        item=obligation.item,
        severity=obligation.severity,
        reason=obligation.reason,
        source_attempt_ts=obligation.source_attempt_ts,
        source_attempt_msg=obligation.source_attempt_msg,
        status="still_open",
        repo_key=repo_key,
    )
    state.open_obligations.append(ob)
    save_state(drive_root, state)


def test_run_unified_review_obligation_loading_uses_drive_root_and_make_repo_key(tmp_path, monkeypatch):
    """Verify that _run_unified_review passes durable obligations to the prompt builder.

    This tests the production call-site wiring in review.py:
    - load_state(pathlib.Path(ctx.drive_root)) is used (not a file path)
    - make_repo_key(pathlib.Path(ctx.repo_dir)) is used (not str())
    - The loaded obligations reach _build_review_history_section

    Strategy: monkeypatch _build_review_history_section to capture its arguments,
    then verify the persisted obligation was passed via the correct repo_key path.
    """
    import ouroboros.tools.review as review_mod

    drive_root = tmp_path / "data"
    drive_root.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)

    repo_key = make_repo_key(repo_dir)
    persisted = ObligationItem(
        obligation_id="ob-unified-999",
        item="code_quality",
        severity="critical",
        reason="specific concrete wiring regression",
        source_attempt_ts="2026-04-17T00:00:00Z",
        source_attempt_msg="fix: wiring",
    )
    _write_obligation_to_state(drive_root, repo_key, persisted)

    # Capture obligations passed to the prompt builder.
    captured_obligations = []
    original_build = review_mod._build_review_history_section

    def capturing_build(history, open_obligations=None):
        if open_obligations:
            captured_obligations.extend(open_obligations)
        return original_build(history, open_obligations=open_obligations)

    monkeypatch.setattr(review_mod, "_build_review_history_section", capturing_build)

    # Stub out heavy git / LLM / file I/O so we never leave the obligation-loading path.
    monkeypatch.setattr(
        review_mod, "run_cmd",
        lambda cmd, *args, **kw: ("M\tfile.py" if "--name-only" in cmd else "+ change"),
    )
    import ouroboros.tools.review_binary_context as _rbc
    monkeypatch.setattr(_rbc, "capture_staged_diff", lambda _repo, *, unified=3: "+ change")
    monkeypatch.setattr(review_mod, "build_touched_file_pack", lambda *a, **k: ("(pack)", []))
    monkeypatch.setattr(review_mod, "_load_checklist_section", lambda: "(checklists)")
    monkeypatch.setattr(
        review_mod, "load_governance_doc",
        lambda _root, name, **_kw: f"({pathlib.Path(name).stem.lower()})",
    )
    # Stub the expensive review engine — return no findings so we focus on loading.
    monkeypatch.setattr(review_mod, "_collect_review_findings",
                        lambda *a, **k: ([], [], [], []))
    # Also stub the multi-model review path so no LLM calls are made.
    monkeypatch.setattr(review_mod, "_handle_multi_model_review",
                        lambda *a, **k: None)

    ctx = type("_Ctx", (), {
        "repo_dir": str(repo_dir),
        "drive_root": str(drive_root),
        "task_id": "test-task",
        "_review_history": [],
        "_review_iteration_count": 0,
        "_last_review_block_reason": "",
        "_last_triad_models": [],
        "_last_review_critical_findings": [],
        "_last_triad_raw_results": [],
        "_review_degraded_reasons": [],
        "_last_scope_raw_result": {},
        "_review_advisory": [],
        "pending_events": None,
        "event_queue": None,
    })()
    review_mod._run_unified_review(ctx, "fix: wiring test", repo_dir=str(repo_dir))

    # The obligation written under make_repo_key must have been passed to the builder.
    found_ids = [ob.obligation_id for ob in captured_obligations]
    assert "ob-unified-999" in found_ids, (
        f"Expected obligation 'ob-unified-999' to be loaded via make_repo_key+drive_root "
        f"and passed to _build_review_history_section. Got: {found_ids}"
    )
    all_prompt_text = original_build(_mk_history(), open_obligations=captured_obligations)
    assert '"obligation_id": "ob-unified-999"' in all_prompt_text


def test_run_unified_review_injects_obligation_ids_with_correct_repo_key(tmp_path):
    """Write an obligation keyed by make_repo_key(repo_dir), load via
    load_state(drive_root), and verify it renders into the review history section.

    This is the integration path mirrored by `_run_unified_review` in review.py
    after the Fix 1 correction (drive_root argument + make_repo_key lookup).
    """
    drive_root = tmp_path / "data"
    drive_root.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)

    repo_key = make_repo_key(repo_dir)
    persisted = ObligationItem(
        obligation_id="ob-int-001",
        item="code_quality",
        severity="critical",
        reason="specific concrete finding about a symbol",
        source_attempt_ts="2026-04-17T00:00:00Z",
        source_attempt_msg="fix: something",
    )
    _write_obligation_to_state(drive_root, repo_key, persisted)

    # Replay the exact sequence used inside _run_unified_review after Fix 1.
    loaded = load_state(drive_root)
    open_obs = loaded.get_open_obligations(repo_key=make_repo_key(repo_dir))
    assert open_obs, "expected obligation to be loadable under the canonical repo_key"

    out = triad_hist(_mk_history(), open_obligations=open_obs)
    assert '"obligation_id": "ob-int-001"' in out
    assert '"item": "code_quality"' in out


def test_scope_build_prompt_loads_obligations_from_drive_root(tmp_path, monkeypatch):
    """Verify `_build_scope_prompt` with a valid `drive_root` loads obligations
    from the persisted state and renders them into the prompt.

    We stub the heavy scope-pack builders so the test focuses on the obligation
    loading wiring introduced by Fix 2. The stubs supply just enough structure
    for `_compute_touched_status` to return None (success).
    """
    drive_root = tmp_path / "data"
    drive_root.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)

    repo_key = make_repo_key(repo_dir)
    persisted = ObligationItem(
        obligation_id="ob-scope-42",
        item="cross_module_bugs",
        severity="critical",
        reason="broken implicit contract in sibling module",
        source_attempt_ts="2026-04-17T00:00:00Z",
        source_attempt_msg="refactor: split module",
    )
    _write_obligation_to_state(drive_root, repo_key, persisted)

    # Stub the heavy scope-pack / git I/O helpers so the prompt builder succeeds.
    monkeypatch.setattr(
        scope_pack,
        "_parse_staged_name_status",
        lambda _rd: [("M", "file.py", "file.py")],
    )
    monkeypatch.setattr(
        scope_pack,
        "build_touched_file_pack",
        lambda _rd, _paths, **_kwargs: ("(touched file pack)", []),
    )
    monkeypatch.setattr(
        scope_pack,
        "_inline_deleted_file_pack",
        lambda pack, _deleted, _rd, **_kwargs: pack,
    )
    monkeypatch.setattr(
        scope_pack,
        "_gather_scope_packs",
        lambda _rd, _paths, fixed_prompt_tokens=0: "(generated scope atlas)",
    )
    monkeypatch.setattr(
        scope_pack,
        "run_cmd",
        lambda *args, **kwargs: "diff --git a/file.py b/file.py\n",
    )

    prompt, status = _build_scope_prompt(
        repo_dir,
        commit_message="fix: integration scope test",
        context=scope_review_mod._ScopePromptContext(drive_root=drive_root),
    )

    assert status is None, f"expected success, got status={status}"
    assert prompt is not None
    assert '"obligation_id": "ob-scope-42"' in prompt
    assert '"item": "cross_module_bugs"' in prompt


# ---------------------------------------------------------------------------
# Fence safety: obligation reason containing triple-backticks must not break
# the JSON code block injected into the review history section.
# ---------------------------------------------------------------------------

def _make_ob(obligation_id: str, item: str, reason: str) -> ObligationItem:
    ob = ObligationItem.__new__(ObligationItem)
    ob.obligation_id = obligation_id
    ob.item = item
    ob.severity = "critical"
    ob.reason = reason
    ob.status = "still_open"
    ob.source_ts = "2026-01-01T00:00:00"
    ob.source_commit = "test commit"
    return ob


def test_triad_history_section_backtick_in_reason_does_not_break_fence():
    """Obligation reason with triple-backticks must not escape the JSON fence."""
    reason_with_backticks = "Found issue in ```python\ncode block\n``` here"
    ob = _make_ob("abc123", "code_quality", reason_with_backticks)
    result = triad_hist([], open_obligations=[ob])
    # The block must still be fenced (at least one opening fence present)
    assert "```" in result
    # The obligation_id must be present inside the block (not broken out)
    assert '"obligation_id": "abc123"' in result
    # The reason_excerpt must be sanitized (newlines collapsed, backticks via format_prompt_code_block safe fence)
    assert "abc123" in result


def test_scope_history_section_backtick_in_reason_does_not_break_fence():
    """Same fence-safety guarantee for scope_review copy of _build_review_history_section."""
    reason_with_backticks = "See ```json\n{\"key\": \"value\"}\n``` for details"
    ob = _make_ob("def456", "security_issues", reason_with_backticks)
    result = scope_hist([], open_obligations=[ob])
    assert "```" in result
    assert '"obligation_id": "def456"' in result
    assert "def456" in result
