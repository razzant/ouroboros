"""The diff pulse: a change's size since its contribution base, a fact and one fixed
question for the mind while it edits its body, at the commit gate and atop every
reviewer's brief of a body subject; never a threshold, a warning or a block
(``ouroboros/review.py::diff_pulse``)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from ouroboros.review import DIFF_PULSE_QUESTION, diff_pulse
from ouroboros.tools.registry import ToolContext
from tests.test_reference_book_balance import _registry
from tests.test_review_change_end_to_end import GOAL, SCOPE, _brief_text, staged_body  # noqa: F401 -- the fixture

pytestmark = pytest.mark.serial  # real git subprocesses

OFFICIAL = "https://github.com/razzant/ouroboros.git"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                                   cwd=root).decode("utf-8").strip()


def _write(root: Path, path: str, lines: int, text: str = "x = 1\n") -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text * lines, encoding="utf-8", newline="\n")


def _body(root: Path, *, official: bool = True) -> Path:
    """A body checkout whose official development ref is cached at its first commit."""
    root.mkdir(parents=True)
    _write(root, "ouroboros/core.py", 10)
    _write(root, "notes.md", 1, "notes\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    if official:
        _git(root, "remote", "add", "canonical", OFFICIAL)
        _git(root, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    return root


def test_the_pulse_counts_every_wave_since_the_contribution_base(tmp_path):
    repo = _body(tmp_path / "body")
    base = _git(repo, "rev-parse", "HEAD")
    # Wave 1 creates a module and grows another, and is committed.
    _write(repo, "ouroboros/custody.py", 285, "y = 2\n")
    _write(repo, "ouroboros/core.py", 15)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "wave 1")
    # Wave 2 is still on disk: one more line and an untracked new module.
    _write(repo, "ouroboros/core.py", 16)
    _write(repo, "ouroboros/fresh.py", 4, "z = 3\n")
    # What the last wave alone shows hides the module wave 1 created.
    assert "custody.py" not in _git(repo, "diff", "--name-only", "HEAD")

    fact, question = diff_pulse(repo, ["ouroboros/core.py"]).split("\n")
    assert fact.startswith(f"Diff pulse since canonical/ouroboros merge-base {base[:12]}: ")
    assert "files 3, lines +295/-0 (net +295)" in fact
    assert "new files: ouroboros/custody.py, ouroboros/fresh.py;" in fact
    assert fact.endswith("module lines (target 1000, cap 1600): ouroboros/core.py 16 (+6/-0).")
    assert question == DIFF_PULSE_QUESTION

    # The index a commit or a review subject holds: the untracked module is not in it.
    _git(repo, "add", "ouroboros/core.py")
    staged = diff_pulse(repo, staged=True)
    assert "files 2, lines +291/-0 (net +291); new files: ouroboros/custody.py;" in staged
    assert "ouroboros/custody.py 285 (+285/-0), ouroboros/core.py 16 (+6/-0)." in staged
    assert "fresh.py" not in staged


def test_without_a_cached_official_ref_the_pulse_says_so_and_measures_from_head(tmp_path):
    repo = _body(tmp_path / "fork", official=False)
    _write(repo, "ouroboros/core.py", 12)
    _git(repo, "commit", "-qam", "committed growth")
    _write(repo, "notes.md", 2, "notes\n")
    unknown = "Diff pulse since HEAD (contribution base unavailable): files 1, lines +1/-0 (net +1);"
    assert diff_pulse(repo).startswith(unknown)
    # A remote's name carries no authority: a fork's ouroboros branch is not the official line.
    _git(repo, "remote", "add", "origin", "https://github.com/contributor/ouroboros.git")
    _git(repo, "update-ref", "refs/remotes/origin/ouroboros", "HEAD~1")
    assert diff_pulse(repo).startswith(unknown)
    _git(repo, "remote", "add", "canonical", OFFICIAL)
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD~1")
    assert "files 2, lines +3/-0 (net +3);" in diff_pulse(repo)


def test_an_unmeasurable_root_has_no_pulse_and_raises_nothing(tmp_path):
    plain = tmp_path / "plain"
    _write(plain, "ouroboros/core.py", 3)
    assert diff_pulse(plain) == "" and diff_pulse(tmp_path / "missing") == ""


def test_one_shape_and_one_question_for_any_size(tmp_path):
    """No size is a trigger: a one-line and a five-thousand-line growth of the same
    module read alike, with the same question and nothing that warns."""
    shapes = []
    for name, grown in (("small", 1), ("huge", 5000)):
        repo = _body(tmp_path / name)
        _write(repo, "ouroboros/core.py", 10 + grown)
        pulse = diff_pulse(repo)
        fact, question = pulse.split("\n")
        assert question == DIFF_PULSE_QUESTION and "⚠" not in pulse and f"+{grown}/-0" in fact
        shapes.append(re.sub(r"\d+", "N", re.sub(r"merge-base [0-9a-f]{12}", "merge-base SHA", fact)))
    assert shapes[0] == shapes[1]


def test_list_bounds_are_presentation_and_count_what_they_leave_out(tmp_path):
    repo = _body(tmp_path / "body")
    for index in range(3):
        _write(repo, f"ouroboros/m{index}.py", index + 1)
    bounded = diff_pulse(repo, limit=2)
    assert "files 3, lines +6/-0 (net +6); new files: ouroboros/m0.py, ouroboros/m1.py (+1 more);" in bounded
    assert "ouroboros/m2.py 3 (+3/-0), ouroboros/m1.py 2 (+2/-0) (+1 more)." in bounded  # most grown first
    whole = diff_pulse(repo)
    assert "new files: ouroboros/m0.py, ouroboros/m1.py, ouroboros/m2.py;" in whole and "more" not in whole


CODE = {
    "write_file": {"path": "ouroboros/loop.py", "content": "x = 1\ny = 2\n"},
    "edit_text": {"path": "ouroboros/loop.py", "old_str": "x = 1", "new_str": "x = 1\ny = 2"},
    "apply_patch": {"patch": "*** Update File: ouroboros/loop.py\n-x = 1\n+x = 1\n+y = 2\n"},
    "edit_batch": {"edits": [{"path": "ouroboros/loop.py", "old_str": "x = 1", "new_str": "x = 1\ny = 2"}]},
}


@pytest.mark.parametrize("tool", sorted(CODE))
def test_a_body_edit_carries_the_pulse_and_a_foreign_project_edit_does_not(tmp_path, monkeypatch, tool):
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    body = str(_registry(tmp_path / "a", external=False).execute(tool, CODE[tool]))
    assert body.startswith("✅"), body[:300]
    assert "Diff pulse since canonical/ouroboros merge-base" in body and "ouroboros/loop.py 2 (+1/-0)" in body
    assert DIFF_PULSE_QUESTION in body and "Text growth" not in body  # no book or ruled text was touched
    # A user's project with the same layout and the official remote is not the body.
    foreign = str(_registry(tmp_path / "b", external=True).execute(tool, CODE[tool]))
    assert foreign.startswith("✅") and "Diff pulse" not in foreign, foreign[:300]


def test_the_commit_gate_shows_the_pulse_and_no_size_changes_what_it_decides(tmp_path):
    from ouroboros.tools import commit_gate

    seen = {}
    # The huge diff is well past the retired 400,000-character "Large diff" warning.
    for name, grown in (("small", 1), ("huge", 70_000)):
        repo = _body(tmp_path / name)
        _write(repo, "ouroboros/core.py", 10 + grown)
        _git(repo, "add", "-A")
        progress: list = []
        ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / name / "data", task_id="t",
                          emit_progress_fn=progress.append)
        result = commit_gate.deterministic_preflight(ctx, "refactor: core", ["ouroboros/core.py"])
        [size] = [line for line in progress if line.startswith("Size (information):")]
        assert size.endswith(diff_pulse(repo, staged=True)) and f"ouroboros/core.py {10 + grown} (+{grown}/-0)" in size
        seen[name] = (result, [line for line in progress if line is not size])
    assert seen["small"] == seen["huge"]  # neither the outcome nor any warning depends on the size


def test_only_an_ordinary_body_subject_carries_the_pulse_into_a_brief(tmp_path):
    from ouroboros.tools.review_helpers import build_goal_section, review_subject_pulse

    repo = _body(tmp_path / "body")
    _write(repo, "ouroboros/core.py", 11)
    _git(repo, "add", "-A")
    assert review_subject_pulse(repo, "body") == diff_pulse(repo, staged=True) != ""
    assert review_subject_pulse(repo, "core") == "" and review_subject_pulse(repo, "body", managed=object()) == ""
    plain = build_goal_section("Ship it", "", "fix: ship")
    assert build_goal_section("Ship it", "", "fix: ship", diff_pulse="") == plain
    assert build_goal_section("Ship it", "", "fix: ship", "", "## Exhibit", "PULSE") == plain.replace(
        "forgotten.", "forgotten.\n\n\n## Exhibit\n\n\nPULSE", 1)


def test_every_seat_of_a_body_wave_reads_the_pulse_and_a_core_brief_does_not(
        staged_body, tmp_path, monkeypatch):  # noqa: F811 -- the imported fixture
    import ouroboros.review_substrate as substrate
    from ouroboros import review_ledger
    from ouroboros.tools.review_admission import build_two_part_brief
    from ouroboros.tools.review_change import run_review_change
    from ouroboros.tools.review_subject import ReviewSubjectSpec, freeze_subject
    from tests import _contributor_packet_shared as shared

    repo = Path(staged_body["repo"])
    shared.git(repo, "remote", "add", "canonical", OFFICIAL)
    shared.git(repo, "update-ref", "refs/remotes/canonical/ouroboros", staged_body["base_sha"])
    sent: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(sent))
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "drive")
    result = run_review_change(ctx, root="system_repo", surface="change", goal=GOAL, scope=SCOPE, subject="index")
    assert result["state"] == "settled", result
    pulse = diff_pulse(repo, staged=True)
    fact = pulse.split("\n")[0]
    assert fact.startswith(f"Diff pulse since canonical/ouroboros merge-base {staged_body['base_sha'][:12]}")
    assert sorted(brief["slot_id"] for brief in sent) == ["s1", "t1", "t2"]
    for brief in sent:  # the packet seat and both retrieving seats
        assert _brief_text(brief).count(pulse) == 1, brief["slot_id"]

    record = review_ledger.load_record(ctx.drive_root, result["record_id"])
    row = next(row for row in record["rows"] if row["seat_id"] == "t1")
    seat = {"slot_id": "t1", "model": row["requested"]["model"], "route": row["requested"]["route"],
            "retrieves": False, "session_profile": row["requested"]["profile"], "subagent_id": row["subagent_id"]}
    frozen = freeze_subject(ctx, ReviewSubjectSpec(root_kind="system_repo", root=str(repo), kind="index",
                                                   governance_root=str(repo), surface="change", layer="body"))
    assert pulse in build_two_part_brief(frozen, seat, goal=GOAL, scope=SCOPE)["system"]
    assert fact not in build_two_part_brief(frozen, seat, layer="core", goal=GOAL, scope=SCOPE)["system"]
