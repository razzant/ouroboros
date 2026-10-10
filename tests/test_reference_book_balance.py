"""The reference-book balance: a fact on every local surface, never a gate (BIBLE P3 c5).

The official CI lane (tests/test_reference_book_budgets.py) is what refuses a grown book;
here the same measurement reaches Ouroboros when it writes book text, plans book work,
runs readiness before a paid reviewer and asks for codebase health.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from ouroboros.reference_books import (
    BOOK_ENTRYPOINTS,
    BOOK_GROWTH_RULE,
    book_balance_note,
    book_balances,
    book_plan_fact,
    book_source_sizes,
    compose_book,
    composed_size,
    load_reference_book,
    render_book_balance,
)
from tests.test_plan_review_engine import CLEAN, DECK_SPEC, _call, harness as _engine_harness

REPO = pathlib.Path(__file__).resolve().parents[1]
CHAPTER = "docs/architecture/01-one.md"
ENTRY = "# Architecture\n\nThe map.\n\n## Chapters\n\n- [One](architecture/01-one.md)\n"
BODY = "# One\n\nThe first chapter.\n\nP1 honest and long enough to shorten.\n"
harness = _engine_harness


def _git(root, *args):
    return subprocess.check_output(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                                   cwd=root).decode("utf-8").strip()


def _book_repo(root: pathlib.Path, *, official: bool = True) -> pathlib.Path:
    for path, text in {BOOK_ENTRYPOINTS["architecture"]: ENTRY, CHAPTER: BODY,
                       "ouroboros/reference_books.py": "# the book contract\n", "ouroboros/loop.py": "x = 1\n",
                       "notes.md": "notes\n"}.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8", newline="\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    if official:
        _git(root, "remote", "add", "canonical", "https://github.com/razzant/ouroboros.git")
        _git(root, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    return root


def test_composed_size_is_the_composed_book_on_the_real_tree():
    for book_id in BOOK_ENTRYPOINTS:
        composed = len(compose_book(load_reference_book(REPO, book_id)).encode("utf-8"))
        assert composed_size(book_source_sizes(REPO, book_id)) == composed


@pytest.mark.serial
@pytest.mark.parametrize("remote,url", [
    ("canonical", "https://github.com/razzant/ouroboros.git"),
    ("renamed", "git@github.com:razzant/ouroboros.git"),
    ("origin", "ssh://git@GitHub.com/RAZZANT/Ouroboros.git/"),
])
def test_published_feature_keeps_contribution_growth(tmp_path, remote, url):
    clone = _book_repo(tmp_path / "clone", official=False)
    base = _git(clone, "rev-parse", "HEAD")
    _git(clone, "remote", "add", remote, url)
    target = f"refs/remotes/{remote}/ouroboros"
    _git(clone, "update-ref", target, base)
    # Names carry no authority: managed is a fork; publication has caught up
    # to HEAD on the author's feature branch, exactly the old zero-debt bug.
    _git(clone, "remote", "add", "managed", "https://github.com/contributor/ouroboros.git")
    fork = "managed" if remote == "origin" else "origin"
    if fork == "origin":
        _git(clone, "remote", "add", fork, "https://github.com/contributor/ouroboros.git")
    _git(clone, "checkout", "-qb", "feature")
    (clone / CHAPTER).write_text(BODY + "Committed growth.\n", encoding="utf-8", newline="\n")
    _git(clone, "commit", "-qam", "grow")
    for ref in (f"refs/remotes/{fork}/feature", "refs/remotes/managed/ouroboros"):
        _git(clone, "update-ref", ref, "HEAD")
    _git(clone, "branch", f"--set-upstream-to={fork}/feature")
    assert _git(clone, "rev-parse", "@{upstream}") == _git(clone, "rev-parse", "HEAD")
    (published,) = book_balances(clone, [CHAPTER])
    assert published.vs_head == 0 and published.vs_upstream == published.owed == len("Committed growth.\n")
    # Advance the official target independently: the measured base must be the
    # common ancestor, not this tip and not the feature-tracking upstream.
    tree = _git(clone, "rev-parse", f"{base}^{{tree}}")
    tip = _git(clone, "commit-tree", tree, "-p", base, "-m", "official advance")
    _git(clone, "update-ref", target, tip)
    (clone / CHAPTER).write_text(BODY + "Committed growth.\nMore.\n", encoding="utf-8", newline="\n")
    (balance,) = book_balances(clone, [CHAPTER])
    assert balance.vs_head == len("More.\n")
    assert balance.vs_upstream == len("Committed growth.\nMore.\n") == balance.owed
    assert balance.upstream == target and balance.merge_base == base
    assert target in render_book_balance(balance) and base in render_book_balance(balance)
    assert balance.changed == ((CHAPTER, len("More.\n")),)
    assert book_balances(clone, ["notes.md"]) == []  # no book source touched, nothing measured


@pytest.mark.serial
def test_linked_body_candidate_without_upstream_keeps_committed_growth(tmp_path):
    serving = _book_repo(tmp_path / "serving")
    base = _git(serving, "rev-parse", "HEAD")
    (serving / CHAPTER).write_text(BODY + "Inherited growth.\n", encoding="utf-8", newline="\n")
    _git(serving, "commit", "-qam", "serving growth")
    candidate = tmp_path / "candidate"
    _git(serving, "worktree", "add", "-qb", "candidate/task", str(candidate), "HEAD")
    assert (candidate / ".git").is_file()
    assert subprocess.run(["git", "rev-parse", "@{upstream}"], cwd=candidate, capture_output=True).returncode
    (balance,) = book_balances(candidate, [CHAPTER])
    assert balance.vs_head == 0 and balance.vs_upstream == len("Inherited growth.\n")
    assert balance.merge_base == base and balance.upstream == "refs/remotes/canonical/ouroboros"


@pytest.mark.serial
@pytest.mark.parametrize("baseline", ["absent_remote", "fork", "missing_ref", "unrelated", "missing_book"])
def test_unknown_contribution_never_means_paid_even_with_measured_head(tmp_path, baseline):
    repo = _book_repo(tmp_path / "repo", official=False)
    if baseline != "absent_remote":
        url = "https://github.com/fork/ouroboros.git" if baseline == "fork" else "https://github.com/razzant/ouroboros.git"
        _git(repo, "remote", "add", "managed", url)
        if baseline != "missing_ref":
            base = _git(repo, "rev-parse", "HEAD")
            if baseline == "unrelated":
                base = _git(repo, "commit-tree", "HEAD^{tree}", "-m", "unrelated root")
            elif baseline == "missing_book":
                _git(repo, "rm", "--cached", "-q", CHAPTER, BOOK_ENTRYPOINTS["architecture"])
                tree = _git(repo, "write-tree")
                _git(repo, "read-tree", "HEAD")
                base = _git(repo, "commit-tree", tree, "-m", "before books")
                head = _git(repo, "commit-tree", "HEAD^{tree}", "-p", base, "-m", "add books")
                _git(repo, "update-ref", "HEAD", head)
            _git(repo, "update-ref", "refs/remotes/managed/ouroboros", base)
    for text in (BODY, BODY.replace(" and long enough to shorten", ""), BODY + "Added.\n"):
        (repo / CHAPTER).write_text(text, encoding="utf-8", newline="\n")
        (balance,) = book_balances(repo, [CHAPTER])
        assert balance.vs_head == len(text.encode()) - len(BODY.encode())
        assert balance.vs_upstream is None
        note = book_balance_note(repo, [CHAPTER])
        assert "contribution unknown" in note and "nothing owed" not in note


@pytest.mark.serial
@pytest.mark.parametrize("suffix", ["", "Short.\n"])
def test_committed_growth_can_be_paid_in_the_worktree(tmp_path, suffix):
    repo = _book_repo(tmp_path)
    (repo / CHAPTER).write_text(BODY + "Committed growth.\n", encoding="utf-8", newline="\n")
    _git(repo, "commit", "-qam", "growth")
    text = BODY if not suffix else "# One\n\nThe first chapter.\n\n" + suffix
    (repo / CHAPTER).write_text(text, encoding="utf-8", newline="\n")
    (balance,) = book_balances(repo, [CHAPTER])
    assert balance.vs_head < 0 and balance.vs_upstream <= 0 and balance.owed == 0
    assert "nothing owed" in book_balance_note(repo, [CHAPTER])


@pytest.mark.serial
@pytest.mark.parametrize("url", [
    "/tmp/github.com/razzant/ouroboros.git", "file://github.com/razzant/ouroboros.git",
    "https://example.invalid/razzant/ouroboros.git", "https://github.com.example.invalid/razzant/ouroboros.git",
    "https://github.com/razzant/ouroboros-extra.git", "https://github.com:bad/razzant/ouroboros.git",
])
def test_remote_host_and_exact_repository_matter_even_with_an_official_push_url(tmp_path, url):
    repo = _book_repo(tmp_path, official=False)
    _git(repo, "remote", "add", "managed", url)
    _git(repo, "remote", "set-url", "--push", "managed", "https://github.com/razzant/ouroboros.git")
    _git(repo, "update-ref", "refs/remotes/managed/ouroboros", "HEAD")
    (balance,) = book_balances(repo, [CHAPTER])
    assert balance.vs_head == 0 and balance.vs_upstream is None and balance.upstream == ""
    assert "contribution unknown" in book_balance_note(repo, [CHAPTER])


@pytest.mark.serial
def test_membership_and_utf8_bytes_include_composition_separators(tmp_path):
    repo = _book_repo(tmp_path)
    second = "docs/architecture/02-two.md"
    entry = ENTRY + "- [Two](architecture/02-two.md)\n"
    body = "# Two\n\nКратко.\n"
    (repo / BOOK_ENTRYPOINTS["architecture"]).write_text(entry, encoding="utf-8")
    (repo / CHAPTER).write_text(BODY.replace(" and long enough to shorten", ""), encoding="utf-8")
    (repo / second).write_text(body, encoding="utf-8")
    expected = len(compose_book(load_reference_book(repo, "architecture")).encode("utf-8"))
    (balance,) = book_balances(repo, [second])
    base_size = len(ENTRY.encode()) + len(BODY.encode()) + 2
    assert balance.size == expected
    assert balance.vs_head == balance.vs_upstream == expected - base_size
    assert sum(delta for _, delta in balance.changed) + 2 == balance.vs_head


@pytest.mark.serial
def test_balance_helper_does_not_import_runtime_state(tmp_path):
    repo = _book_repo(tmp_path)
    code = ("import sys; from pathlib import Path; from ouroboros.reference_books import book_balances; "
            "assert book_balances(Path(sys.argv[1]))[0].vs_upstream == 0; "
            "assert not any(m == 'supervisor' or m.startswith('supervisor.') or m in "
            "{'ouroboros.config', 'ouroboros.settings_integrity', 'ouroboros.review_state', "
            "'ouroboros.runtime_mode_policy'} for m in sys.modules)")
    subprocess.run([sys.executable, "-B", "-c", code, str(repo)], cwd=REPO, check=True, capture_output=True)


@pytest.mark.serial
def test_note_states_the_rule_when_owed_and_one_quiet_line_when_paid(tmp_path):
    repo = _book_repo(tmp_path)
    (repo / CHAPTER).write_text(BODY + "Added.\n", encoding="utf-8", newline="\n")
    grown = book_balance_note(repo, [CHAPTER])
    assert "Architecture book" in grown and "+7 B vs HEAD" in grown and BOOK_GROWTH_RULE in grown
    (repo / CHAPTER).write_text(BODY.replace(" and long enough to shorten", ""), encoding="utf-8", newline="\n")
    paid = book_balance_note(repo, [CHAPTER])
    assert paid.startswith("ℹ️ Reference book:") and "nothing owed" in paid
    assert BOOK_GROWTH_RULE not in paid and "\n" not in paid
    assert book_balance_note(repo, ["notes.md"]) == ""


@pytest.mark.serial
def test_note_never_calls_an_unmeasured_book_paid(tmp_path):
    repo = tmp_path / "fresh"
    repo.mkdir()
    (repo / "notes.md").write_text("notes\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "no book yet")
    for path, text in {BOOK_ENTRYPOINTS["architecture"]: ENTRY, CHAPTER: BODY}.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8", newline="\n")
    unmeasured = book_balance_note(repo, [CHAPTER])
    assert "n/a vs HEAD" in unmeasured and "nothing owed" not in unmeasured
    missing = ENTRY + "- [Two](architecture/02-missing.md)\n"  # a chapter list naming an absent file
    (repo / BOOK_ENTRYPOINTS["architecture"]).write_text(missing, encoding="utf-8", newline="\n")
    assert "balance unavailable" in book_balance_note(repo, [CHAPTER])
    measured = _book_repo(tmp_path / "measured")
    assert "nothing owed" in book_balance_note(measured, [CHAPTER])


@pytest.mark.serial
@pytest.mark.parametrize("missing_book", ["architecture", "development"])
@pytest.mark.parametrize("growth", ["", "Added.\n"])
def test_note_discloses_an_unavailable_book_beside_a_measured_book(tmp_path, missing_book, growth):
    repo = _book_repo(tmp_path)
    development = "docs/development/01-one.md"
    (repo / development).parent.mkdir(parents=True)
    (repo / development).write_text(BODY, encoding="utf-8", newline="\n")
    (repo / BOOK_ENTRYPOINTS["development"]).write_text(
        ENTRY.replace("Architecture", "Development").replace("architecture/", "development/"),
        encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "both books")
    _git(repo, "update-ref", "refs/remotes/canonical/ouroboros", "HEAD")
    paths = [CHAPTER, development]
    missing = CHAPTER if missing_book == "architecture" else development
    readable = development if missing_book == "architecture" else CHAPTER
    (repo / readable).write_text(BODY + growth, encoding="utf-8", newline="\n")
    complete = book_balance_note(repo, paths)
    readable_only = book_balance_note(repo, [readable])
    assert "unavailable" not in complete
    assert ("nothing owed" in complete) is (not growth)
    assert (BOOK_GROWTH_RULE in complete) is bool(growth)

    (repo / missing).unlink()
    partial = book_balance_note(repo, paths)
    assert f"{missing_book.title()} book balance unavailable" in partial
    assert render_book_balance(book_balances(repo, [readable])[0]) in partial
    assert "nothing owed" not in partial
    assert (BOOK_GROWTH_RULE in partial) is bool(growth)
    assert book_balance_note(repo, [readable]) == readable_only
    assert book_balance_note(repo, ["notes.md"]) == ""

    (repo / missing).write_text(BODY, encoding="utf-8", newline="\n")
    assert book_balance_note(repo, paths) == complete


def _registry(tmp_path, *, external: bool):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    system = _book_repo(tmp_path / "system")
    drive = tmp_path / "drive"
    drive.mkdir()
    registry = ToolRegistry(repo_dir=system, drive_root=drive)
    if external:  # a user's own project that happens to carry the same docs layout
        project = _book_repo(tmp_path / "project")
        registry.set_context(ToolContext(repo_dir=system, system_repo_dir=system, drive_root=drive,
                                         workspace_root=project, workspace_mode="external"))
    return registry


GROW = {
    "write_file": {"path": CHAPTER, "content": BODY + "Added.\n"},
    "edit_text": {"path": CHAPTER, "old_str": "P1 honest", "new_str": "P1 honest, added"},
    "apply_patch": {"patch": f"*** Update File: {CHAPTER}\n-P1 honest and long enough to shorten.\n"
                             "+P1 honest and long enough to shorten, added.\n"},
    "edit_batch": {"edits": [{"path": CHAPTER, "old_str": "P1 honest", "new_str": "P1 honest, added"}]},
}


@pytest.mark.serial
@pytest.mark.parametrize("tool", sorted(GROW))
def test_a_body_book_edit_carries_the_balance(tmp_path, monkeypatch, tool):
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    result = str(_registry(tmp_path, external=False).execute(tool, GROW[tool]))
    assert result.startswith("✅"), result[:300]
    assert "ℹ️ Reference books:" in result and BOOK_GROWTH_RULE in result


@pytest.mark.serial
@pytest.mark.parametrize("tool", sorted(GROW))
def test_a_foreign_project_or_a_non_book_path_gets_no_balance(tmp_path, monkeypatch, tool):
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *args, **kwargs: (True, ""))
    foreign = str(_registry(tmp_path / "a", external=True).execute(tool, GROW[tool]))
    assert foreign.startswith("✅") and "Reference book" not in foreign, foreign[:300]
    plain = {"write_file": {"path": "notes.md", "content": "notes, more\n"},
             "edit_text": {"path": "notes.md", "old_str": "notes", "new_str": "notes, more"},
             "apply_patch": {"patch": "*** Update File: notes.md\n-notes\n+notes, more\n"},
             "edit_batch": {"edits": [{"path": "notes.md", "old_str": "notes", "new_str": "notes, more"}]}}[tool]
    body = str(_registry(tmp_path / "b", external=False).execute(tool, plain))
    assert body.startswith("✅") and "Reference book" not in body, body[:300]


def test_plan_fact_names_book_sources_and_new_modules_only(tmp_path):
    repo = _book_repo(tmp_path)
    chapter = book_plan_fact(repo, [CHAPTER])
    assert chapter.startswith("FACT:") and "docs/ARCHITECTURE.md" in chapter and BOOK_GROWTH_RULE in chapter
    module = book_plan_fact(repo, ["ouroboros/new_module.py"])
    assert "new module(s) ouroboros/new_module.py" in module and "Architecture-book source" in module
    assert book_plan_fact(repo, ["ouroboros/loop.py", "notes.md", "tests/test_new.py"]) == ""


def test_plan_task_carries_the_book_fact_for_a_system_repo_plan_only(harness):
    harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    chapter = {**DECK_SPEC, "affected_paths": [str(harness.system / "docs" / "architecture" / "06-agent-core.md")]}
    assert "FACT: affected_paths name book sources of docs/ARCHITECTURE.md" in _call(harness.make_ctx(), spec=chapter)
    module = {**DECK_SPEC, "affected_paths": [str(harness.system / "ouroboros" / "brand_new.py")]}
    assert "new module(s) ouroboros/brand_new.py" in _call(harness.make_ctx(task_id="task-2"), spec=module)
    existing = {**DECK_SPEC, "affected_paths": [str(harness.system / "ouroboros" / "loop.py")]}
    assert "FACT: affected_paths" not in _call(harness.make_ctx(task_id="task-3"), spec=existing)
    workspace = {**DECK_SPEC, "affected_paths": [str(harness.workspace / "docs" / "architecture" / "01-x.md")]}
    assert "FACT: affected_paths" not in _call(harness.make_ctx(task_id="task-4"), spec=workspace)


def test_plan_task_measures_a_bound_body_candidate(harness):
    import shutil

    harness.install({"s1": CLEAN, "s2": CLEAN, "s3": CLEAN})
    candidate = harness.system.parent / "candidate"
    shutil.copytree(harness.system, candidate)

    def bound(task_id):
        ctx = harness.make_ctx(active_workspace=False, task_id=task_id)
        ctx.serving_repo_dir = harness.system
        ctx.repo_dir = ctx.system_repo_dir = candidate
        ctx.task_metadata["body_candidate"] = {"path": str(candidate)}
        return ctx

    chapter = {**DECK_SPEC, "affected_paths": ["docs/architecture/01-one.md"]}
    assert "FACT: affected_paths name book sources of docs/ARCHITECTURE.md" in _call(bound("c-1"), spec=chapter)
    module = {**DECK_SPEC, "affected_paths": ["supervisor/brand_new.py"]}
    assert "new module(s) supervisor/brand_new.py" in _call(bound("c-2"), spec=module)
    existing = {**DECK_SPEC, "affected_paths": ["ouroboros/loop.py"]}
    assert "FACT: affected_paths" not in _call(bound("c-3"), spec=existing)


@pytest.mark.serial
def test_readiness_warns_on_a_grown_touched_book_only(tmp_path):
    from ouroboros.tools.review_helpers import check_worktree_readiness

    repo = _book_repo(tmp_path / "body")
    (repo / CHAPTER).write_text(BODY + "Added.\n", encoding="utf-8", newline="\n")
    grown = [w for w in check_worktree_readiness(repo) if "Architecture book" in w]
    assert len(grown) == 1 and grown[0].startswith("official CI will enforce:") and BOOK_GROWTH_RULE in grown[0]
    (repo / CHAPTER).write_text(BODY.replace(" and long enough to shorten", ""), encoding="utf-8", newline="\n")
    assert not [w for w in check_worktree_readiness(repo) if "book" in w]
    foreign = _book_repo(tmp_path / "foreign")
    (foreign / "ouroboros" / "reference_books.py").unlink()
    _git(foreign, "commit", "-qam", "no book contract")
    (foreign / CHAPTER).write_text(BODY + "Added.\n", encoding="utf-8", newline="\n")
    assert not [w for w in check_worktree_readiness(foreign) if "book" in w]
