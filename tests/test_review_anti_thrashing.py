"""Tests for anti-thrashing behavior in pre-commit review history sections.

Covers the verdict-authoritative and verification-only instructions in the ONE
shared `build_review_history_section` — as the packet seat (`review.py`) and the
two-part brief (`review_brief_coupling.py`) both render it — plus the Part-2
coupling-history block, and that earlier rounds reach the next wave as written,
with no obligation layer on top (step 5).
"""
import json
import pathlib
import subprocess

from ouroboros.review_state import make_repo_key
from ouroboros.tools.review import _build_review_history_section as triad_hist
from ouroboros.tools.review_helpers import (
    _ANTI_THRASHING_RULE_VERDICT,
    _HISTORY_VERIFICATION_ONLY_RULE,
)
from ouroboros.tools.review_brief_coupling import build_coupling_history_section as _build_scope_history_section
from ouroboros.tools.review_helpers import build_review_history_section as scope_hist


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


def test_history_section_verdict_authoritative_instruction():
    out = triad_hist(_mk_history())
    assert _ANTI_THRASHING_RULE_VERDICT in out


def test_history_section_verification_only_instruction():
    out = triad_hist(_mk_history())
    assert _HISTORY_VERIFICATION_ONLY_RULE in out


def test_history_section_empty_without_history():
    assert triad_hist([]) == ""
    assert triad_hist(None) == ""


def test_history_section_renders_rounds_and_rules():
    out = triad_hist(_mk_history())
    assert "Round 1" in out
    assert "CRITICAL findings:" in out
    assert "IMPORTANT RULES FOR THIS REVIEW" in out
    assert "Open obligations" not in out and "obligation_id" not in out


# ---------------------------------------------------------------------------
# The two-part brief's Part-1 history (the same shared builder)
# ---------------------------------------------------------------------------


def test_scope_review_history_section_verdict_authoritative():
    out = scope_hist(_mk_history())
    assert _ANTI_THRASHING_RULE_VERDICT in out
    assert _HISTORY_VERIFICATION_ONLY_RULE in out


def test_scope_review_history_section_empty_without_inputs():
    assert scope_hist([]) == ""
    assert scope_hist(None) == ""


# ---------------------------------------------------------------------------
# review_brief_coupling.build_coupling_history_section — verdict-authoritative note
# ---------------------------------------------------------------------------


def test_coupling_history_section_verdict_authoritative():
    history = [
        {"summary": "previous coupling round noted a broken contract",
         "status": "responded"},
    ]
    out = _build_scope_history_section(history)
    # The shared constant is interpolated into the Part-2 history block.
    assert _ANTI_THRASHING_RULE_VERDICT in out
    assert _HISTORY_VERIFICATION_ONLY_RULE in out


# ---------------------------------------------------------------------------
# Earlier rounds reach the next wave as written (step 5 home).
# ---------------------------------------------------------------------------


def _init_git_repo(path: pathlib.Path) -> None:
    """Create a minimal git repo rooted at `path` so make_repo_key resolves stably."""
    subprocess.run(
        ["git", "init", "-q"], cwd=str(path), check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def test_the_next_wave_reads_earlier_rounds_as_written_without_an_obligation_layer(tmp_path):
    """Home (step 5): with a state file that still holds an open obligation for the checkout (left by a body
    with Blocking), the packet section's owner and the retrieving brief carry an earlier round's finding verbatim
    under the verdict and verification-only rules, and carry neither the stored obligation, its id nor the
    item-name rule; with no earlier round neither renders a history."""
    from ouroboros.tools import review_brief_coupling as brief_mod
    from ouroboros.tools.review_helpers import review_history_brief

    drive_root, repo_dir = tmp_path / "data", tmp_path / "repo"
    (drive_root / "state").mkdir(parents=True)
    repo_dir.mkdir()
    _init_git_repo(repo_dir)
    stored = {"obligation_id": "obl-0007", "item": "code_quality", "severity": "critical",
              "reason": "ARCHIVED-OBLIGATION wording", "source_attempt_ts": "2026-10-01T00:00:00Z",
              "source_attempt_msg": "fix: earlier", "status": "still_open", "repo_key": make_repo_key(repo_dir)}
    (drive_root / "state" / "advisory_review.json").write_text(json.dumps(
        {"state_version": 3, "attempts": [], "open_obligations": [stored], "next_obligation_seq": 8}), encoding="utf-8")
    finding = "the retry loop re-reads the stale cache on line 40"
    rounds = [{"attempt": 1, "commit_message": "fix: retry", "critical": [
        {"item": "code_quality", "reason": finding, "severity": "critical", "verdict": "FAIL"}]}]

    packet = review_history_brief(rounds, drive_root=drive_root, repo_root=repo_dir, task_id="t-1")
    brief, _manifest = brief_mod.build_retrieving_brief(repo_dir, brief_mod.BriefInputs(
        commit_message="fix: retry", drive_root=drive_root, intent=brief_mod.BriefIntent(review_history=rounds)))
    for text in (packet, brief):
        assert finding in text and _ANTI_THRASHING_RULE_VERDICT in text and _HISTORY_VERIFICATION_ONLY_RULE in text
        for gone in ("ARCHIVED-OBLIGATION", "obl-0007", "Open obligations from previous",
                     "Do NOT rephrase prior findings", "reuse that exact"):
            assert gone not in text, gone

    assert review_history_brief([], drive_root=drive_root, repo_root=repo_dir, task_id="t-1") == ""
    bare, _manifest = brief_mod.build_retrieving_brief(repo_dir, brief_mod.BriefInputs(
        commit_message="fix: retry", drive_root=drive_root))
    assert "Previous review rounds" not in bare and "ARCHIVED-OBLIGATION" not in bare


# ---------------------------------------------------------------------------
# Fence safety: a round's recorded details containing triple-backticks must not
# break the JSON code block the review history section renders them in.
# ---------------------------------------------------------------------------

def _round_with_detail(detail: str) -> list:
    return [{"attempt": 1, "commit_message": "fix: fence", "critical": [], "note": detail}]


def test_triad_history_section_backtick_in_a_round_detail_does_not_break_fence():
    """A recorded detail with triple-backticks gets a longer fence; its JSON stays whole."""
    detail = "Found issue in ```python\ncode block\n``` here"
    result = triad_hist(_round_with_detail(detail))
    block = result.split("````json\n", 1)[1].split("\n````", 1)[0]
    assert json.loads(block) == {"note": detail}


def test_brief_history_section_backtick_in_a_round_detail_does_not_break_fence():
    """Same fence-safety guarantee as the two-part brief renders it."""
    detail = "See ```json\n{\"key\": \"value\"}\n``` for details"
    result = scope_hist(_round_with_detail(detail))
    block = result.split("````json\n", 1)[1].split("\n````", 1)[0]
    assert json.loads(block) == {"note": detail}


