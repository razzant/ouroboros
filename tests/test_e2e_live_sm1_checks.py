"""SM1's clean-worktree check after the reviewed commit: the runtime's own transient scratch under ``.ouroboros/``
(``run_script`` in the active workspace, tools/shell.py) is recorded but does not fail the check — the post-task
evolution cycle starts seconds after the commit in the same clone (rc.15 run3, SM1_a1, issue #701) — while any other
untracked or modified path still does."""
from __future__ import annotations

import pathlib
import subprocess

from devtools.e2e_live import scenarios


def _repo(root: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for key, value in (("user.name", "t"), ("user.email", "t@example.com")):
        subprocess.run(["git", "-C", str(root), "config", key, value], check=True)
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "base"], check=True)
    return root


def test_the_runtimes_transient_scratch_is_recorded_but_does_not_fail_the_clean_check(tmp_path):
    root = _repo(tmp_path)
    assert scenarios.worktree_after_commit(root) == (True, "", [])
    (root / ".ouroboros" / "tmp_scripts").mkdir(parents=True)
    (root / ".ouroboros" / "tmp_scripts" / "script_deadbeef.py").write_text("print(1)\n", encoding="utf-8")
    clean, porcelain, transient = scenarios.worktree_after_commit(root)
    assert clean is True and porcelain == "?? .ouroboros/" and transient == ["?? .ouroboros/"]


def test_any_other_untracked_or_modified_path_still_fails_the_clean_check(tmp_path):
    root = _repo(tmp_path)
    (root / ".ouroboros" / "tmp_scripts").mkdir(parents=True)
    (root / ".ouroboros" / "tmp_scripts" / "script_deadbeef.py").write_text("print(1)\n", encoding="utf-8")
    (root / "stray.txt").write_text("x\n", encoding="utf-8")
    clean, porcelain, transient = scenarios.worktree_after_commit(root)
    assert clean is False and "?? stray.txt" in porcelain and transient == ["?? .ouroboros/"]
    (root / "stray.txt").unlink()
    (root / "a.txt").write_text("changed\n", encoding="utf-8")
    clean, porcelain, transient = scenarios.worktree_after_commit(root)
    assert clean is False and "M a.txt" in porcelain and transient == ["?? .ouroboros/"]   # ``_git`` strips the leading column


def test_the_landing_commit_call_records_its_skip_flags():
    """rc.15 run2/run3: two SM1 lanes landed through review_rebuttal + skip_advisory_review=True; the documented
    path has no skip flags, so the landing call's truthy ``skip_*`` arguments are a recorded fact and a check."""
    rows = [{"tool": "commit_reviewed", "result_preview": "⚠️ REVIEW_BLOCKED (attempt 1)", "args": {"skip_advisory_review": True}},
            {"tool": "commit_reviewed", "result_preview": "OK: committed to ouroboros: x",
             "args": {"skip_advisory_review": True, "skip_tests": False, "paths": ["a"]}},
            {"tool": "read_file", "result_preview": "OK", "args": {"skip_advisory_review": True}}]
    facts = scenarios.commit_refusal_facts({}, rows, {})
    assert facts["landing_skip_flags"] == ["skip_advisory_review"]
    clean = [{"tool": "commit_reviewed", "result_preview": "OK: committed", "args": {"paths": ["a"]}}]
    assert scenarios.commit_refusal_facts({}, clean, {})["landing_skip_flags"] == []
    assert scenarios.commit_refusal_facts({}, [], {})["landing_skip_flags"] == []
