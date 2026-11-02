"""DEVELOPMENT.md follows the subject of the work (BIBLE P1, owner 6A 27.10; D-DEV 08.08).

The task's own ``context_requires_development`` wins either way and absent is not false; P1's surfaces (a
delegated child, an external API/CLI/scheduled source or actor) get the pointer unless the task asks for the
handbook; self-body work keeps it; a bound tree ``body_fact`` proves foreign gets the pointer; an unbound task
or a tree whose body fact is unknown keeps the book. In Max without the book, its chapter map stands in.
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.context_input_selection import development_resident


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})


@pytest.fixture
def trees(tmp_path):
    system = tmp_path / "system"
    system.mkdir()
    _git(system, "init", "-q")
    (system / "README.md").write_text("body", encoding="utf-8")
    _git(system, "add", "README.md")
    _git(system, "commit", "-q", "-m", "body")
    _git(system, "worktree", "add", "-q", "--detach", str(tmp_path / "body-worktree"))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    _git(foreign, "init", "-q")
    _git(foreign, "remote", "add", "origin", "https://github.com/someone/elsewhere.git")
    documents = tmp_path / "documents"
    documents.mkdir()
    ctx = SimpleNamespace(repo_dir=system, system_repo_dir=system, drive_root=tmp_path / "data")
    return ctx, {"body": tmp_path / "body-worktree", "foreign": foreign, "documents": documents}


@pytest.mark.parametrize("tree,resident", [("body", True), ("foreign", False), ("documents", True), ("missing", True)])
def test_only_a_bound_tree_proven_foreign_gets_the_chapter_map(trees, tree, resident):
    """A body worktree keeps the book, a tree with someone else's remote gets the map, and a tree whose body fact
    is unknown (no git, or no such folder) keeps the book: missing facts never narrow it (6A)."""
    ctx, paths = trees
    root = str(paths.get(tree) or paths["documents"].parent / "no-such-folder")
    task = {"id": "t", "type": "task", "workspace_root": root, "workspace_mode": "external"}
    assert development_resident(task, ctx) is resident
    assert development_resident({**task, "context_requires_development": not resident}, ctx) is (not resident)


def test_a_body_fact_that_fails_keeps_the_book(trees, monkeypatch):
    from ouroboros import review_body_fact

    ctx, paths = trees
    monkeypatch.setattr(review_body_fact, "body_fact", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("git gone")))
    task = {"type": "task", "workspace_root": str(paths["foreign"]), "workspace_mode": "external"}
    assert development_resident(task, ctx) is True


def test_without_a_binding_the_subject_is_unknown_and_the_book_stays(trees):
    ctx, _paths = trees
    for task in ({"type": "task"}, {"type": "task", "_is_direct_chat": True}, {"project_id": "p", "type": "task"}):
        assert development_resident(task, ctx) is True
    assert development_resident({"type": "task", "context_requires_development": False}, ctx) is False


@pytest.mark.parametrize("surface", [{"delegation_role": "subagent"}, {"metadata": {"source": "api_task"}},
                                     {"metadata": {"source": "scheduled_task"}}, {"actor_id": "cli"}])
def test_p1_surfaces_get_the_pointer_unless_the_task_asks(trees, surface):
    ctx, _paths = trees
    task = {"type": "task", **surface}
    assert development_resident(task, ctx) is False
    assert development_resident({**task, "context_requires_self_body_docs": True}, ctx) is True
    assert development_resident({**task, "task_contract": {"context_requires_self_body_docs": True}}, ctx) is True
    assert development_resident({**task, "context_requires_development": True}, ctx) is True
    # A contract's default false self-body flag is no answer: it never narrows on its own.
    assert development_resident({"type": "task", "task_contract": {"context_requires_self_body_docs": False}}, ctx)


def test_self_body_work_keeps_the_book_on_a_foreign_binding_and_an_explicit_no_still_wins(trees):
    ctx, paths = trees
    for kind in ("evolution", "deep_self_review", "review"):
        task = {"type": kind, "workspace_root": str(paths["foreign"]), "workspace_mode": "external"}
        assert development_resident(task, ctx) is True
        assert development_resident({**task, "context_requires_development": False}, ctx) is False


def test_max_without_the_handbook_carries_its_chapter_map_in_the_handbooks_own_block():
    from ouroboros.context_layout import reference_doc_sections

    development = "# DEVELOPMENT.md\n\n## Review\n\nRead every source.\n\n## Naming\n\nNames say what they are.\n"
    env = SimpleNamespace(repo_dir=None)
    with_book = reference_doc_sections(env, context_mode="max", include_development=True,
                                       architecture_text="# ARCH\n\nbody", development_text=development)
    without = reference_doc_sections(env, context_mode="max", include_development=False,
                                     architecture_text="# ARCH\n\nbody", development_text=development)
    assert with_book[0] == without[0]  # the common sections stay shared across tasks (one cached prefix)
    assert with_book[1] == "## DEVELOPMENT.md\n\n" + development
    assert without[1].startswith("## DEVELOPMENT.md (navigation map)") and "Review" in without[1]
    assert "Read every source." not in without[1]


@pytest.mark.parametrize("flag", [True, False, None])
def test_start_task_carries_an_explicit_answer_to_the_admitted_task_and_absence_stays_absent(tmp_path, monkeypatch, flag):
    from tests.test_root_effort_ingress import _promoted

    task, _row, _workers = _promoted(tmp_path, monkeypatch, **({} if flag is None else {"context_requires_development": flag}))
    assert task.get("context_requires_development", "absent") == ("absent" if flag is None else flag)
