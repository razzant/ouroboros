"""The diff pulse: a change's size since its contribution base, a fact and one fixed
question for the mind while it edits its body, at the commit gate and atop every
reviewer's brief of a body subject; never a threshold, a warning or a block
(``ouroboros/review.py::diff_pulse``)."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from ouroboros.review import DIFF_PULSE_QUESTION, diff_pulse
from ouroboros.tools.registry import ToolContext
from tests._cases import case_folders
from tests._reference_book_fixtures import BODY, CHAPTER, ENTRY
from tests._reference_book_fixtures import _registry
from tests._review_change_fixtures import GOAL, SCOPE, _brief_text, staged_body  # noqa: F401 -- the fixture


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

    fact, shape, question = diff_pulse(repo, ["ouroboros/core.py"]).split("\n")  # an edit's own file: no directory fact
    assert fact.startswith(f"Diff pulse since canonical/ouroboros merge-base {base[:12]}: ")
    assert "files 3, lines +295/-0 (net +295)" in fact
    assert "new files: ouroboros/custody.py, ouroboros/fresh.py;" in fact
    assert fact.endswith("module lines (target 1000, cap 1600): ouroboros/core.py 16 (+6/-0).")
    assert shape == "Code shape: module handles +0/-0, calls through them +0/-0, re-exported names +0/-0."
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


def test_with_no_readable_diff_the_touched_books_keep_their_balance(tmp_path):
    """Neither a contribution base nor HEAD can be diffed (a body checkout before its first commit): no fact
    line, yet the touched books keep the balance Git can measure instead of disappearing with the pulse."""
    from ouroboros.reference_books import book_balance_note

    repo = tmp_path / "unborn"
    for path, text in {"docs/ARCHITECTURE.md": ENTRY, CHAPTER: BODY}.items():
        _write(repo, path, 1, text)
    _git(repo, "init", "-q")
    note = book_balance_note(repo, [CHAPTER])
    assert "n/a vs HEAD; contribution unknown" in note
    assert diff_pulse(repo, [CHAPTER]) == note and diff_pulse(repo) == ""  # unshown, no book is known to be touched


def test_a_staged_pulse_sizes_modules_and_books_from_the_index_not_later_worktree_edits(tmp_path):
    """The staged pulse diffs the index, so its module sizes and book balance read the index too: an
    unstaged edit made after staging (more lines, more prose, or a revert of the staged growth) must
    neither mask nor inflate what the commit would hold."""
    from tests._reference_book_fixtures import _book_repo

    repo = _book_repo(tmp_path / "body")
    staged_prose, unstaged = "Staged reason.\n", "Unstaged worktree prose, much longer than the staged line.\n"
    (repo / CHAPTER).write_text(BODY + staged_prose, encoding="utf-8", newline="\n")
    _write(repo, "ouroboros/loop.py", 4)
    _git(repo, "add", CHAPTER, "ouroboros/loop.py")
    (repo / CHAPTER).write_text(BODY + staged_prose + unstaged, encoding="utf-8", newline="\n")
    _write(repo, "ouroboros/loop.py", 40)

    staged = diff_pulse(repo, [CHAPTER, "ouroboros/loop.py"], staged=True)
    assert f"+{len(staged_prose)} B vs HEAD" in staged
    assert f"+{len(staged_prose) + len(unstaged)} B vs HEAD" not in staged
    assert "ouroboros/loop.py 4 (+3/-0)" in staged and "ouroboros/loop.py 40" not in staged
    worktree = diff_pulse(repo, [CHAPTER, "ouroboros/loop.py"])
    assert f"+{len(staged_prose) + len(unstaged)} B vs HEAD" in worktree and "ouroboros/loop.py 40 (+39/-0)" in worktree

    # An unstaged revert cannot hide staged growth either.
    (repo / CHAPTER).write_text(BODY, encoding="utf-8", newline="\n")
    assert f"+{len(staged_prose)} B vs HEAD" in diff_pulse(repo, [CHAPTER], staged=True)


@pytest.mark.skipif(os.name == "nt", reason="a literal backslash cannot be part of a Windows file name")
def test_a_staged_name_the_module_inventory_rejects_is_counted_not_sized_and_never_raises(tmp_path):
    """The staged pulse is a fact: a valid POSIX file name the gated-module inventory rejects (a literal
    backslash) is counted in the change and left unsized, as the worktree pulse does, and the commit's
    preflight carries on."""
    repo = _body(tmp_path / "body")
    for name in ("ordinary.md", "notes\\draft.md", "odd\\module.py"):
        (repo / name).write_text("A note\n", encoding="utf-8", newline="\n")
        _git(repo, "add", "--", name)
    staged = diff_pulse(repo, staged=True)
    assert "files 3," in staged and "odd\\module.py 1" not in staged
    worktree = diff_pulse(repo)
    assert "files 3," in worktree


def test_staged_constitution_and_prompts_are_sized_from_the_index_and_a_conflict_is_unmeasured(tmp_path):
    """The ruled texts (BIBLE.md and prompts/) follow the index like the books: an unstaged edit after staging
    neither hides nor inflates the staged balance; during an unresolved merge the prompts are unmeasured, never a
    false deletion, and they are measured again once the conflict is staged resolved."""
    from ouroboros.reference_books import UnmeasuredText, book_balance_note, ruled_text_sizes

    repo = _body(tmp_path / "body")
    (repo / "prompts").mkdir()
    for name, text in (("BIBLE.md", "# Bible\n"), ("prompts/base.md", "kept\n"), ("prompts/other.md", "base\n")):
        (repo / name).write_text(text, encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "ruled texts")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    (repo / "BIBLE.md").write_text("# Bible\nStaged.\n", encoding="utf-8", newline="\n")
    _git(repo, "add", "BIBLE.md")
    (repo / "BIBLE.md").write_text("# Bible\n", encoding="utf-8", newline="\n")  # an unstaged revert
    assert "+8 B vs HEAD" in book_balance_note(repo, ["BIBLE.md"], staged=True)
    assert "+8 B vs HEAD" not in book_balance_note(repo, ["BIBLE.md"])
    _git(repo, "commit", "-qm", "bible")

    branch = _git(repo, "branch", "--show-current")
    _git(repo, "checkout", "-qb", "theirs")
    (repo / "prompts/other.md").write_text("theirs\n", encoding="utf-8", newline="\n")
    _git(repo, "commit", "-qam", "theirs")
    _git(repo, "checkout", "-q", branch)
    (repo / "prompts/other.md").write_text("ours\n", encoding="utf-8", newline="\n")
    _git(repo, "commit", "-qam", "ours")
    merged = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "merge", "theirs"], cwd=repo,
                            capture_output=True)
    assert merged.returncode == 1 and _git(repo, "ls-files", "-u", "--", "prompts/other.md")
    with pytest.raises(UnmeasuredText, match="holds prompts/other.md unmerged"):  # there and unread: never "absent"
        ruled_text_sizes(repo, "prompts", staged=True)
    assert "balance unavailable" in book_balance_note(repo, ["prompts/other.md"], staged=True)
    (repo / "prompts/other.md").write_text("resolved\n", encoding="utf-8", newline="\n")
    _git(repo, "add", "prompts/other.md")
    assert ruled_text_sizes(repo, "prompts", staged=True) == {"prompts/base.md": 5, "prompts/other.md": 9}


def test_one_shape_and_one_question_for_any_size(tmp_path):
    """No size is a trigger: a one-line and a five-thousand-line growth of the same
    module read alike, with the same question and nothing that warns."""
    shapes = []
    for name, grown in (("small", 1), ("huge", 5000)):
        repo = _body(tmp_path / name)
        _write(repo, "ouroboros/core.py", 10 + grown)
        pulse = diff_pulse(repo)
        fact, shape, question = pulse.split("\n")
        assert question == DIFF_PULSE_QUESTION and "⚠" not in pulse and f"+{grown}/-0" in fact
        shapes.append(re.sub(r"\d+", "N", re.sub(r"merge-base [0-9a-f]{12}", "merge-base SHA", fact + shape)))
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


SUITE = """import pytest


@pytest.mark.serial
def test_kept():
    pass


@pytest.mark.parametrize("value", [1, 2, 3])
def test_rows(value):
    pass
"""


def _suite_repo(root: Path) -> Path:
    """A body whose official ref holds a test module with four declared items and one with two."""
    repo = _body(root)
    _write(repo, "tests/test_old.py", 1, SUITE)
    _write(repo, "tests/test_gone.py", 1, "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n")
    _write(repo, "tests/_world.py", 1, "world = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "suite")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    return repo


def test_a_changed_test_source_adds_what_the_suite_declares_and_records(tmp_path, monkeypatch):
    from ouroboros import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    repo = _suite_repo(tmp_path / "body")
    grown = SUITE.replace("[1, 2, 3]", "[1, 2, 3, 4]") + "\n\n@pytest.mark.parametrize('x', range(9))\ndef test_computed(x):\n    pass\n"
    _write(repo, "tests/test_old.py", 1, grown)  # one row and one computed list: +2 items
    (repo / "tests/test_gone.py").unlink()  # -2 items
    _write(repo, "tests/test_new.py", 1, 'import pytest\n\npytestmark = pytest.mark.serial("cost-stand")\n'
                                         '@pytest.mark.parametrize("a", [1, 2])\n@pytest.mark.parametrize("b", [1, 2, 3])\n'
                                         'def test_x(a, b):\n    pass\n# @pytest.mark.serial("a comment")\n')  # +6 items
    _write(repo, "web/tests/view.test.js", 2, "test();\n")
    _write(repo, "tests/_world.py", 3, "world = 1\n")
    _write(repo, "ouroboros/core.py", 11)

    fact, *_ = diff_pulse(repo).split("\n")
    assert "module lines (target 1000, cap 1600): ouroboros/core.py 11 (+1/-0); tests: items +8/-2, new modules 2, " in fact
    assert "serial marks +1 (cost-stand), recorded test-seconds unknown, test module lines: " in fact
    assert fact.endswith("test module lines: tests/test_new.py 8 (+8/-0), tests/test_old.py 16 (+6/-1), "
                         "tests/_world.py 3 (+2/-0), web/tests/view.test.js 2 (+2/-0).")  # a deleted module has no size

    # The table of the last complete battery: the slowest system it saw, and what it never saw.
    state = tmp_path / "data" / "state"
    state.mkdir(parents=True)
    table = state / "test-durations.json"
    table.write_text(json.dumps({"format": 1, "platforms": {
        "darwin": {"tests/test_old.py": 2500}, "linux": {"tests/test_old.py": 1000, "tests/test_gone.py": 1500}}}),
        encoding="utf-8")
    assert "4 recorded test-seconds of changed modules (1 not recorded), test module lines: " in diff_pulse(repo)
    table.write_text("{", encoding="utf-8")
    assert "recorded test-seconds unknown" in diff_pulse(repo)  # unknown is never zero
    for value in ('"2500"', "true", "NaN", '{"ms": 1}', "9" * 310, "-3000", "[" * 10000 + "]" * 10000):  # unknown, no raise
        table.write_text('{"format": 1, "platforms": {"darwin": {"tests/test_old.py": %s}}}' % value, encoding="utf-8")
        assert "recorded test-seconds unknown" in diff_pulse(repo)
        # A valid sample on another system never hides the unreadable one.
        table.write_text('{"format": 1, "platforms": {"darwin": {"tests/test_old.py": %s}, '
                         '"linux": {"tests/test_old.py": 2000}}}' % value, encoding="utf-8")
        assert "recorded test-seconds unknown" in diff_pulse(repo)
    table.write_text('{"format": 1, "platforms": {"darwin": [], "linux": {"tests/test_old.py": 2000}}}', encoding="utf-8")
    assert "recorded test-seconds unknown" in diff_pulse(repo)  # a system without a file map is no table
    table.unlink()

    # An edit shows its own files, and a shown file outside the suite adds nothing.
    assert "tests: items +2/-0, new modules 0, serial marks +0, " in diff_pulse(repo, ["tests/test_old.py"])
    assert "; tests:" not in diff_pulse(repo, ["ouroboros/core.py"])
    # The index a commit or a review subject holds, not the worktree.
    _git(repo, "add", "tests/test_gone.py")
    assert "tests: items +0/-2, new modules 0," in diff_pulse(repo, staged=True)
    # A test module that does not parse mid-edit is unread, never counted as empty.
    _write(repo, "tests/test_old.py", 1, "def broken(:\n")
    assert "tests: items +0/-0 (1 unreadable)," in diff_pulse(repo, ["tests/test_old.py"])


def test_the_suite_facts_read_each_side_with_one_git_call(tmp_path, monkeypatch):
    """A `git show` per changed test module cost a 2,000-module PR's commit fifty seconds of pulse: both sides
    are read with one `git cat-file --batch` each (the worktree side needs none), and a path with whitespace,
    which batch input would split, is read alone."""
    from ouroboros import review

    repo = _suite_repo(tmp_path / "body")
    for index in range(12):
        _write(repo, f"tests/test_many_{index}.py", 1, SUITE)
    _write(repo, "tests/test_old.py", 1, SUITE + "\n\ndef test_more():\n    pass\n")
    _git(repo, "add", "-A")
    calls, real = [], review.subprocess.run

    def counted(argv, *args, **kwargs):
        calls.append(list(argv))
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(review.subprocess, "run", counted)
    declared = review._declared_tests(SUITE, "tests/test_x.py")[0]
    assert f"tests: items +{12 * declared + 1}/-0, new modules 12," in diff_pulse(repo, staged=True)
    batched = [argv for argv in calls if "cat-file" in argv]
    assert len(batched) == 2 and not any("show" in argv for argv in calls)  # the base side and the index side
    assert "tests: items +1/-0, new modules 0," in diff_pulse(repo, ["tests/test_old.py"])  # the worktree side: one batch
    assert [argv for argv in calls if "cat-file" in argv][2:] == [batched[0]]
    _write(repo, "tests/test_spaced name.py", 1, SUITE)
    _git(repo, "add", "-A")
    calls.clear()
    assert "new modules 13," in diff_pulse(repo, staged=True)
    assert sum("show" in argv for argv in calls) == 2 and len([argv for argv in calls if "cat-file" in argv]) == 2


def test_a_bare_mark_shows_and_a_mark_kept_or_moved_adds_nothing(tmp_path):
    repo = _suite_repo(tmp_path / "body")
    _write(repo, "tests/test_old.py", 1, SUITE.replace("def test_rows", "@pytest.mark.serial\ndef test_rows"))
    assert "serial marks +1 (bare), " in diff_pulse(repo)
    _write(repo, "tests/test_old.py", 1, SUITE + "\n\ndef test_more():\n    pass\n")
    assert "tests: items +1/-0, new modules 0, serial marks +0, " in diff_pulse(repo)


def test_test_lines_are_a_fact_for_any_size_and_no_module_limit_names_them(tmp_path):
    shapes = []
    for name, rows in (("small", 1), ("huge", 3000)):
        repo = _suite_repo(tmp_path / name)
        _write(repo, "tests/test_old.py", 1, SUITE + "\n\ndef test_more():\n    pass\n" * rows)
        pulse = diff_pulse(repo)
        assert "⚠" not in pulse and "module lines (target 1000, cap 1600): none; tests:" in pulse
        shapes.append(re.sub(r"\d+", "N", re.sub(r"merge-base [0-9a-f]{12}", "merge-base SHA", pulse)))
    assert shapes[0] == shapes[1]


LEAF = '''def _core():
    """The parent, read at call time."""
    from ouroboros import core

    return core


def step():
    return _core().x
'''


def test_a_changed_python_source_adds_its_code_shape_and_an_unparsed_one_is_unknown(tmp_path):
    """Module handles and the calls through them, re-exports, a test's patched product-module names and
    `git init` runs, since the contribution base (``code_shape_facts``): facts beside the sizes, never a gate."""
    repo = _body(tmp_path / "body")
    _write(repo, "ouroboros/leaf.py", 1, LEAF)
    _write(repo, "ouroboros/facade.py", 1, "from ouroboros.leaf import step  # noqa: F401\n")
    _write(repo, "tests/test_leaf.py", 1, "def test_step(monkeypatch):\n    pass\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "leaf")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    _write(repo, "ouroboros/leaf.py", 1, LEAF + "\n\ndef twice():\n    return _core().x + _core().y\n")
    _write(repo, "ouroboros/again.py", 1, LEAF)  # an untracked new leaf
    (repo / "ouroboros/facade.py").unlink()
    _write(repo, "tests/test_leaf.py", 1, 'import subprocess\n\n\ndef test_step(monkeypatch, tmp_path):\n'
                                          '    monkeypatch.setattr("ouroboros.core.x", 2)\n'
                                          '    subprocess.run(["git", "init", "-q"], cwd=tmp_path)\n')

    fact, shape, _directory, question = diff_pulse(repo).split("\n")  # a new leaf: the directory fact stands third
    assert shape == ("Code shape: module handles +1/-0, calls through them +3/-0, re-exported names +0/-1; "
                     "tests: patched product-module names +1/-0, git init call sites +1/-0.")
    assert "⚠" not in shape and question == DIFF_PULSE_QUESTION
    # An edit shows its own files; a file outside Python adds no line.
    assert "\nCode shape: module handles +0/-0, calls through them +2/-0, re-exported names +0/-0.\n" in diff_pulse(
        repo, ["ouroboros/leaf.py"])
    assert "Code shape" not in diff_pulse(repo, ["notes.md"])
    # The index a commit or a review subject holds: the untracked leaf and the unstaged test are not in it.
    _git(repo, "add", "ouroboros/leaf.py", "ouroboros/facade.py")
    assert "\nCode shape: module handles +0/-0, calls through them +2/-0, re-exported names +0/-1.\n" in diff_pulse(
        repo, staged=True)
    # A module that does not parse mid-edit is unknown, never zero.
    _write(repo, "ouroboros/leaf.py", 1, "def broken(:\n")
    unknown = "Code shape: module handles, calls through them, re-exported names unknown (1 unreadable: ouroboros/leaf.py)."
    assert unknown in diff_pulse(repo, ["ouroboros/leaf.py"])


def test_an_unread_side_is_unknown_never_a_deletion_and_a_deleted_module_returns_from_no_handle(
        tmp_path, monkeypatch):
    """A side Git does not give, on a delete-only or mode-only edit, is unknown, not removals; a module
    the change deletes is no module its handles return; an ordinary source parses to known counts while
    MemoryError or RecursionError from ast.parse leaves both suite and shape unknown."""
    from ouroboros import review

    repo = _body(tmp_path / "body")
    _write(repo, "ouroboros/leaf.py", 1, LEAF)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "leaf")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    _write(repo, "ouroboros/leaf.py", 1, LEAF.replace('    """The parent, read at call time."""\n', ""))  # -1 line
    _git(repo, "add", "-A")
    unread = "Code shape: module handles, calls through them, re-exported names unknown (1 unreadable: ouroboros/leaf.py)."
    real = review._blob_texts

    def index_unread(root, specs):
        return {spec: None if spec.startswith(":") else text for spec, text in real(root, specs).items()}

    monkeypatch.setattr(review, "_blob_texts", index_unread)
    assert unread in diff_pulse(repo, staged=True)
    monkeypatch.setattr(review, "_blob_texts", real)
    assert "Code shape: module handles +0/-0, calls through them +0/-0, re-exported names +0/-0." in diff_pulse(
        repo, staged=True)
    (repo / "ouroboros/core.py").unlink()
    _git(repo, "add", "-A")
    assert "Code shape: module handles +0/-1, calls through them +0/-1, re-exported names +0/-0." in diff_pulse(
        repo, staged=True)

    test_src, product_src = "def test_ok():\n    pass\n", "X = 1\n"
    _write(repo, "tests/test_sample.py", 1, test_src)
    _write(repo, "ouroboros/sample.py", 1, product_src)
    known_fact, known_shape, *_ = diff_pulse(repo, ["tests/test_sample.py", "ouroboros/sample.py"]).split("\n")
    assert "tests: items +1/-0," in known_fact and "unreadable" not in known_fact
    assert known_shape == (
        "Code shape: module handles +0/-0, calls through them +0/-0, re-exported names +0/-0; "
        "tests: patched product-module names +0/-0, git init call sites +0/-0.")

    real_parse = ast.parse

    def fail_parse(*args, **kwargs):
        source = args[0] if args else kwargs.get("source")
        if source == product_src:
            raise MemoryError("simulated parse memory limit")
        if source == test_src:
            raise RecursionError("simulated parse recursion limit")
        return real_parse(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(ast, "parse", fail_parse)
        unknown_fact, unknown_shape, *_ = diff_pulse(repo, ["tests/test_sample.py", "ouroboros/sample.py"]).split("\n")
    assert "tests: items +0/-0 (1 unreadable)," in unknown_fact
    assert unknown_shape == (
        "Code shape: module handles, calls through them, re-exported names unknown (1 unreadable: "
        "ouroboros/sample.py); tests: patched product-module names, git init call sites unknown (1 unreadable: "
        "tests/test_sample.py).")


def test_the_suite_reads_the_same_side_presence_as_the_shape_line(tmp_path, monkeypatch):
    """A test module's side that exists but was not read is unreadable to the suite's items, as to the shape
    line, never zero items; only a real deletion removes what it declared."""
    from ouroboros import review

    repo = _suite_repo(tmp_path / "body")
    _write(repo, "tests/test_old.py", 1, SUITE + "# one more line\n")
    _git(repo, "add", "-A")
    real = review._blob_texts
    for index_side in (True, False):  # the index side unread, then the base side
        monkeypatch.setattr(review, "_blob_texts", lambda root, specs, index_side=index_side: {
            spec: None if spec.startswith(":") == index_side else text for spec, text in real(root, specs).items()})
        pulse = diff_pulse(repo, ["tests/test_old.py"], staged=True)
        assert "tests: items +0/-0 (1 unreadable), new modules 0," in pulse
        assert "git init call sites unknown (1 unreadable: tests/test_old.py)." in pulse
    monkeypatch.setattr(review, "_blob_texts", real)
    assert "tests: items +0/-0, new modules 0," in diff_pulse(repo, ["tests/test_old.py"], staged=True)
    _git(repo, "rm", "-q", "tests/test_gone.py")
    assert "tests: items +0/-2, new modules 0," in diff_pulse(repo, ["tests/test_gone.py"], staged=True)


CODE = {
    "write_file": {"path": "ouroboros/loop.py", "content": "x = 1\ny = 2\n"},
    "edit_text": {"path": "ouroboros/loop.py", "old_str": "x = 1", "new_str": "x = 1\ny = 2"},
    "apply_patch": {"patch": "*** Update File: ouroboros/loop.py\n-x = 1\n+x = 1\n+y = 2\n"},
    "edit_batch": {"edits": [{"path": "ouroboros/loop.py", "old_str": "x = 1", "new_str": "x = 1\ny = 2"}]},
}


def test_a_body_edit_carries_the_pulse_and_a_foreign_project_edit_does_not(subtests, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for tool in sorted(CODE):
        tmp_path = next(case_paths)
        with subtests.test(tool=tool), monkeypatch.context() as monkeypatch:
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


# ---------------------------------------------------------------- the body's manuals beside a change (BIBLE P1)
_OWNERS = ('[domains]\nD01 = "Core"\n\n[references]\nD01 = ["docs/architecture/01-core.md", "docs/DESIGN.md"]\n'
           '"web/" = ["docs/architecture/03-web.md"]\n\n[modules]\n"ouroboros/core.py" = "D01"\n')
_INVARIANTS = ("# 10. Key Invariants\n\nThe rules.\n\n1. **One owner.** `ouroboros/core.py` owns x.\n"
               "2. **Another file.** `tools/core.py` is not that module.\n3. **Again.** `core.py` keeps y.\n")


def _owned_body(root: Path, chapters: dict) -> Path:
    repo = _body(root)
    files = {"ouroboros/domains.toml": _OWNERS, "docs/architecture/10-key-invariants.md": _INVARIANTS,
             "web/app.js": "// app\n", **chapters}
    for path, text in files.items():
        _write(repo, path, 1, text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "manuals")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    return repo


def test_the_pulse_names_each_touched_files_owning_manuals_and_whether_the_change_touches_them(tmp_path):
    """One row per file, never a union: the reader sees which file left which manual untouched, and which key
    invariants name the file (a mention, not a judgment). A fact beside the change, never a gate."""
    repo = _owned_body(tmp_path / "body", {"docs/architecture/01-core.md": "# 1. Core\n\nThe core.\n"})
    _write(repo, "ouroboros/core.py", 12)
    _write(repo, "web/app.js", 2, "// app\n")
    _write(repo, "docs/architecture/01-core.md", 1, "# 1. Core\n\nThe core, restated.\n")
    pulse = diff_pulse(repo)
    assert ("- ouroboros/core.py: docs/architecture/01-core.md (in this change), docs/DESIGN.md (not in this change); "
            "key invariants that name it: 1, 3") in pulse
    assert "- web/app.js: docs/architecture/03-web.md (not in this change)" in pulse
    owners = pulse.split("Owning manuals")[1].split("Removed governed prose")[0]
    assert owners.count("\n- ") == 2 and "notes.md" not in owners and "01-core.md:" not in owners  # only declared owners' files
    # Staged, the owners are the index's: a later unstaged manifest edit changes nothing the commit would hold.
    _git(repo, "add", "ouroboros/core.py")
    _write(repo, "ouroboros/domains.toml", 1, _OWNERS.replace("01-core.md", "09-other.md"))
    staged = diff_pulse(repo, ["ouroboros/core.py"], staged=True)
    assert "- ouroboros/core.py: docs/architecture/01-core.md (not in this change)" in staged and "09-other" not in staged
    # A tree with no manifest (any other repository) says nothing about owners.
    assert "Owning manuals" not in diff_pulse(_touched(_body(tmp_path / "plain")))


def test_a_change_that_adds_a_runtime_module_says_whether_the_module_directory_is_in_it(tmp_path):
    """The hand-kept module directory beside a new module: the pulse says whether the directory file is in the
    change, as a literal fact about the diff (no veto), names every added module, and says nothing when the change
    adds none: a new test, a new document or an edit of an existing module is no new runtime module."""
    directory = "docs/architecture/01-module-directory.md"
    repo = _owned_body(tmp_path / "body", {directory: "# 1.1 Module Directory\n\nRows.\n"})
    line = "New runtime modules: {}; the hand-kept module directory `" + directory + "` is {}."
    _write(repo, "ouroboros/core.py", 11)
    _write(repo, "tests/test_new.py", 1)
    _write(repo, "docs/new.md", 1, "notes\n")
    assert "New runtime modules" not in diff_pulse(repo) and directory not in diff_pulse(repo)  # nothing added
    _write(repo, "ouroboros/fresh.py", 1)
    _write(repo, "supervisor/also.py", 1)
    assert line.format("ouroboros/fresh.py, supervisor/also.py", "not in this change") in diff_pulse(repo)
    _write(repo, directory, 1, "# 1.1 Module Directory\n\nRows.\n- fresh.py: a new row.\n")
    assert line.format("ouroboros/fresh.py, supervisor/also.py", "in this change") in diff_pulse(repo)
    # It follows the shown files like its sibling notes: an edit's receipt for another file does not repeat it.
    assert "runtime modules" not in diff_pulse(repo, ["ouroboros/core.py"])
    assert line.format("ouroboros/fresh.py", "in this change") in diff_pulse(repo, ["ouroboros/fresh.py"])
    # Staged, the fact is the index's: only what the commit would add, and the directory only if staged with it.
    _git(repo, "add", "ouroboros/fresh.py")
    staged = diff_pulse(repo, ["ouroboros/fresh.py"], staged=True)
    assert line.format("ouroboros/fresh.py", "not in this change") in staged
    # A move is told as a move (Git's own rename detection), beside what is new; a deletion alone says nothing.
    _git(repo, "mv", "ouroboros/core.py", "ouroboros/moved.py")
    moved = diff_pulse(repo, staged=True)
    assert ("New runtime modules: ouroboros/fresh.py; moved runtime modules: ouroboros/moved.py (from ouroboros/core.py); "
            f"the hand-kept module directory `{directory}` is not in this change.") in moved
    _git(repo, "reset", "-q", "--hard")
    _git(repo, "rm", "-q", "ouroboros/core.py")
    assert "runtime modules" not in diff_pulse(repo, staged=True)


def _touched(repo: Path) -> Path:
    _write(repo, "ouroboros/core.py", 11)
    return repo


def test_the_pulse_counts_every_removed_line_of_governed_prose_and_says_what_it_does_not_show(tmp_path):
    """A paragraph moved between changed documents is quiet; one deleted is counted even where a twin survives
    elsewhere; nothing is cut silently: a long line states the characters left out, a long list the lines left
    out, and the locator names every changed governed document."""
    long = "Why this holds: " + "reason " * 60
    keep = [f"Kept rule {n}." for n in range(3)]
    gone = [f"Dropped reason {n}." for n in range(13)]
    fence = "```\nFenced example.\n\nFenced twin.\n```"  # not prose; the lines after its close are again
    one = "# 1. Core\n\n" + "\n\n".join(["Moved paragraph.", "Twin line.", long, *keep, *gone[:6], fence, *gone[6:]]) + "\n"
    two = "# 2. Other\n\nTwin line.\n"
    repo = _owned_body(tmp_path / "body", {"docs/architecture/01-core.md": one, "docs/architecture/02-other.md": two})
    _write(repo, "docs/architecture/01-core.md", 1, "# 1. Core\n\n" + "\n\n".join(keep) + "\n")
    _write(repo, "docs/architecture/02-other.md", 1, two + "\nMoved paragraph.\n")
    pulse = diff_pulse(repo, ["docs/architecture/01-core.md"])
    assert "Removed governed prose: 15 lines of the base are gone from the changed documents:" in pulse
    assert "Moved paragraph." not in pulse and "- docs/architecture/01-core.md: Twin line." in pulse
    assert f"- docs/architecture/01-core.md: {long.strip()[:240]} … (+{len(long.strip()) - 240} characters)" in pulse
    assert "(3 more removed lines are not shown here; all of them: `git diff " in pulse and "Fenced" not in pulse
    assert "-- docs/architecture/01-core.md docs/architecture/02-other.md`)" in pulse
    # A change that removes nothing, and an edit that touches no governed document, say nothing.
    assert "Removed governed prose" not in diff_pulse(repo, ["ouroboros/core.py"])
    _git(repo, "checkout", "-q", "--", "docs")
    _write(repo, "docs/architecture/01-core.md", 1, one + "\nAdded reason.\n")
    assert "Removed governed prose" not in diff_pulse(repo, ["docs/architecture/01-core.md"])


def test_every_owned_file_has_its_row_a_governed_prompt_keeps_its_owner_and_a_removed_file_its_base_owner(tmp_path):
    """No owned file is cut to a count: six owned modules give six rows. A prompt with a declared owner is a file
    like any other here; a removed file keeps the owner the base assigned it, and an invariant that names a file
    is stated whether or not the manifest declares an owner for it."""
    owned = [f"ouroboros/m{n}.py" for n in range(6)]
    manifest = _OWNERS.replace('"web/" =', '"prompts/" = ["docs/development/04-gov.md"]\n"web/" =') + "".join(
        f'"{path}" = "D01"\n' for path in [*owned, "ouroboros/gone.py"])
    invariants = _INVARIANTS + "4. **Notes.** `notes.md` holds the notes.\n"
    repo = _owned_body(tmp_path / "body", {"ouroboros/domains.toml": manifest, "ouroboros/gone.py": "x = 1\n",
                                           "docs/architecture/10-key-invariants.md": invariants,
                                           "prompts/SYSTEM.md": "# System\n\nBe brief.\n"})
    for path in owned:
        _write(repo, path, 1)
    _write(repo, "prompts/SYSTEM.md", 1, "# System\n\nBe brief and kind.\n")
    _write(repo, "notes.md", 2, "notes\n")
    _write(repo, "ouroboros/domains.toml", 1, manifest.replace('"ouroboros/gone.py" = "D01"\n', ""))
    (repo / "ouroboros/gone.py").unlink()
    owners = diff_pulse(repo).split("Owning manuals")[1].split("Removed governed prose")[0]
    for path in [*owned, "ouroboros/gone.py"]:
        assert f"- {path}: docs/architecture/01-core.md (not in this change), docs/DESIGN.md (not in this change)\n" in owners
    assert "- prompts/SYSTEM.md: docs/development/04-gov.md (not in this change)" in owners
    assert "- notes.md: no owning manual declared; key invariants that name it: 4" in owners
    assert owners.count("\n- ") == 9 and "more files" not in owners and "ouroboros/domains.toml:" not in owners


def test_a_governed_side_that_was_not_read_is_unknown_prose_and_a_deleted_document_is_removed(tmp_path, monkeypatch):
    """Missing bytes are not deleted words: a present document whose side could not be read is named, with the
    locator, and counted as neither kept nor removed; a document the change deletes is removed prose."""
    from ouroboros import review

    chapter = "docs/architecture/01-core.md"
    repo = _owned_body(tmp_path / "body", {chapter: "# 1. Core\n\nOne rule.\n\nTwo rule.\n"})
    _write(repo, chapter, 1, "# 1. Core\n\nOne rule.\n")
    _git(repo, "add", "-A")
    real = review._blob_texts
    monkeypatch.setattr(review, "_blob_texts", lambda root, specs: {
        spec: None if spec.startswith(":") else text for spec, text in real(root, specs).items()})
    unread = diff_pulse(repo, [chapter], staged=True)
    assert (f"Removed governed prose unknown (a side was not read here, nothing is counted as removed): {chapter}; "
            f"read them with `git diff --cached ") in unread and "of the base are gone" not in unread
    monkeypatch.setattr(review, "_blob_texts", real)
    read = diff_pulse(repo, [chapter], staged=True)
    assert f"1 line of the base is gone from the changed documents:\n- {chapter}: Two rule.\n" in read
    assert "prose unknown" not in read
    _git(repo, "rm", "-qf", chapter)
    monkeypatch.setattr(review, "_blob_texts", lambda root, specs: {
        spec: None if spec.startswith(":") else text for spec, text in real(root, specs).items()})
    gone = diff_pulse(repo, [chapter], staged=True)
    assert "3 lines of the base are gone" in gone and "prose unknown" not in gone  # an absent side is absent


def test_a_manifest_of_the_wrong_shape_is_one_named_line_and_a_body_write_still_succeeds(tmp_path, monkeypatch):
    """The pulse is information: valid TOML of the wrong shape is an invalid manifest (one ValueError for every
    reader), named once in the receipt of a write that happened, never a tool error; a sound manifest names owners."""
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    broken = _registry(tmp_path / "broken", external=False)
    for path, content in (("ouroboros/domains.toml", "modules = 1\n"), ("ouroboros/loop.py", "x = 2\n")):
        said = str(broken.execute("write_file", {"path": path, "content": content}))
        assert said.startswith("✅"), said[:300]
        assert ("Owning manuals unavailable: ouroboros/domains.toml section [modules] must be a table; "
                "no owner is named.") in said
    sound = _registry(tmp_path / "sound", external=False)
    assert str(sound.execute("write_file", {"path": "ouroboros/domains.toml", "content": (
        '[domains]\nD01 = "Core"\n[references]\nD01 = ["docs/architecture/01-one.md"]\n'
        '[modules]\n"ouroboros/loop.py" = "D01"\n')})).startswith("✅")
    said = str(sound.execute("write_file", {"path": "ouroboros/loop.py", "content": "x = 2\n"}))
    assert said.startswith("✅") and "unavailable" not in said
    assert "- ouroboros/loop.py: docs/architecture/01-one.md (not in this change)" in said
