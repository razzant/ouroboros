"""The body predicate decides the checklist layer from git facts and the
install manifest alone — never from a directory, branch or remote name.

Every case builds real git repositories under ``tmp_path``; the system
repository is a plain checkout that may carry an ``origin`` (the install's own
fork) and a manifest with a configured ``managed_remote_url``.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from ouroboros import review_body_fact as rbf
from ouroboros.review_body_fact import BodyFact, body_fact, layer_for, normalize_remote_url

MANAGED_URL = "https://github.com/razzant/ouroboros.git"
FORK_URL = "git@github.com:someone/ouroboros-fork.git"
FOREIGN_URL = "https://github.com/other/project.git"


def _git(cwd: pathlib.Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=T", *args],
        cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8")
    return proc.stdout.strip()


def _repo(path: pathlib.Path, *, remote: str = "") -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "main")
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "base")
    if remote:
        _git(path, "remote", "add", "origin", remote)
    return path


@pytest.fixture()
def system(tmp_path: pathlib.Path) -> pathlib.Path:
    """The installed body: a checkout whose ``origin`` is the install's fork."""
    return _repo(tmp_path / "system", remote=FORK_URL)


def _fact(root, system, **kwargs) -> BodyFact:
    kwargs.setdefault("manifest", {"managed_remote_url": MANAGED_URL})
    kwargs.setdefault("data_dir", system.parent / "data")
    return body_fact(root, system_repo=system, **kwargs)


# --- the body, each way it is recognized ---------------------------------------

def test_the_system_repository_and_paths_inside_it_are_the_body_by_dir(system):
    fact = _fact(system, system)
    assert (fact.body, fact.how) == ("true", "dir")
    # A path inside the body is the body whether or not it exists yet
    # (creating ouroboros/new_module.py IS self-modification).
    nested = _fact(system / "ouroboros" / "tools", system)
    assert (nested.body, nested.how) == ("true", "dir")
    assert layer_for(nested) == "body"


def test_a_worktree_of_the_body_is_recognized_by_its_common_git_dir(system, tmp_path):
    worktree = tmp_path / "copies" / "wt"
    worktree.parent.mkdir()
    _git(system, "worktree", "add", "-q", str(worktree), "-b", "feature")

    fact = _fact(worktree, system)

    assert (fact.body, fact.how) == ("true", "git_common_dir")


def test_a_local_clone_chain_reaches_the_body_within_three_hops(system, tmp_path):
    first = tmp_path / "c1"
    second = tmp_path / "c2"
    third = tmp_path / "c3"
    _git(tmp_path, "clone", "-q", str(system), str(first))
    _git(tmp_path, "clone", "-q", str(first), str(second))
    _git(tmp_path, "clone", "-q", str(second), str(third))

    direct = _fact(first, system)
    assert (direct.body, direct.how) == ("true", "remote_chain")
    assert "dir" in direct.detail  # the hop that closed the chain is named
    deep = _fact(third, system)
    assert (deep.body, deep.how) == ("true", "remote_chain")


def test_a_chain_longer_than_the_depth_bound_is_unknown_not_guessed(system, tmp_path):
    previous = system
    for name in ("c1", "c2", "c3", "c4"):
        clone = tmp_path / name
        _git(tmp_path, "clone", "-q", str(previous), str(clone))
        previous = clone

    fact = _fact(previous, system)

    assert (fact.body, fact.how) == ("unknown", "unknown")
    assert "chain depth 3 exhausted" in fact.detail


def test_a_remote_at_the_configured_managed_url_is_the_body_whatever_its_spelling(system, tmp_path):
    for spelling in ("git@github.com:razzant/ouroboros.git", "ssh://git@GitHub.com/razzant/ouroboros",
                     "https://GITHUB.com/razzant/ouroboros/"):
        root = _repo(tmp_path / spelling.replace("/", "_").replace(":", "_"), remote=spelling)
        fact = _fact(root, system)
        assert (fact.body, fact.how) == ("true", "managed_remote"), spelling
    assert layer_for(fact) == "body"


def test_the_managed_url_is_the_installs_configured_source_not_a_fixed_official_one(system, tmp_path):
    mirror = _repo(tmp_path / "mirror-clone", remote="https://mirror.example.org/forks/ouro.git")

    assert _fact(mirror, system, manifest={"managed_remote_url": "git@mirror.example.org:forks/ouro"}).how == "managed_remote"
    foreign = _fact(mirror, system)  # the default manifest points elsewhere
    assert (foreign.body, foreign.how) == ("false", "remote_chain")


def test_a_remote_at_the_installs_own_origin_is_the_body_as_install_fork(system, tmp_path):
    root = _repo(tmp_path / "fork-clone", remote="https://github.com/someone/ouroboros-fork")

    fact = _fact(root, system)

    assert (fact.body, fact.how) == ("true", "install_fork")


def test_an_install_without_a_managed_url_is_still_the_body_by_dir_and_common_git_dir(system, tmp_path):
    worktree = tmp_path / "wt2"
    _git(system, "worktree", "add", "-q", str(worktree), "-b", "other")

    assert _fact(system, system, manifest={}).how == "dir"
    assert _fact(worktree, system, manifest={}).how == "git_common_dir"


def test_a_registered_copy_is_judged_by_its_recorded_source(system, tmp_path):
    data_dir = tmp_path / "data"
    body_copy = _repo(tmp_path / "copy-of-body")
    foreign_copy = _repo(tmp_path / "copy-of-other")
    (data_dir / "state").mkdir(parents=True)
    (data_dir / "state" / "subagent_worktrees.json").write_text(json.dumps({"worktrees": [
        {"path": str(body_copy), "repo_dir": str(system), "source_is_system_repo": True,
         "base_sha": "a" * 40, "kind": "self_worktree"},
        {"path": str(foreign_copy), "repo_dir": str(tmp_path / "elsewhere"),
         "source_is_system_repo": False, "base_sha": "b" * 40, "kind": "self_worktree"},
    ]}), encoding="utf-8")

    body = _fact(body_copy, system, data_dir=data_dir)
    other = _fact(foreign_copy, system, data_dir=data_dir)

    assert (body.body, body.how) == ("true", "copy_origin")
    assert (other.body, other.how) == ("false", "copy_origin")


# --- not the body, and not known --------------------------------------------------

def test_a_foreign_origin_is_false_and_names_the_remote(system, tmp_path):
    root = _repo(tmp_path / "foreign", remote=FOREIGN_URL)

    fact = _fact(root, system)

    assert (fact.body, fact.how) == ("false", "remote_chain")
    assert FOREIGN_URL in fact.detail
    assert layer_for(fact) == "core"


def test_the_branch_name_never_takes_part(system, tmp_path):
    root = _repo(tmp_path / "named-like-ours", remote=FOREIGN_URL)
    _git(root, "checkout", "-q", "-b", "ouroboros")

    assert _fact(root, system).body == "false"


def test_no_git_is_unknown(system, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "file.txt").write_text("x\n", encoding="utf-8")

    fact = _fact(plain, system)

    assert (fact.body, fact.how) == ("unknown", "unknown")
    assert "not a git repository" in fact.detail
    missing = _fact(tmp_path / "does-not-exist", system)
    assert (missing.body, missing.how) == ("unknown", "unknown")


def test_a_repository_without_remote_or_copy_origin_is_unknown(system, tmp_path):
    root = _repo(tmp_path / "orphan")

    fact = _fact(root, system)

    assert (fact.body, fact.how) == ("unknown", "unknown")
    assert "no remote" in fact.detail and "no copy origin" in fact.detail


def test_a_git_failure_is_unknown_never_false(system, tmp_path, monkeypatch):
    root = _repo(tmp_path / "timeout", remote=FOREIGN_URL)

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=rbf.GIT_TIMEOUT_SEC)

    monkeypatch.setattr(rbf.subprocess, "run", _timeout)
    fact = _fact(root, system)

    assert (fact.body, fact.how) == ("unknown", "unknown")
    assert "TimeoutExpired" in fact.detail


# --- the mind's argument -----------------------------------------------------------

def test_treat_as_body_raises_only_unknown_and_records_the_raise(system, tmp_path):
    orphan = _repo(tmp_path / "orphan2")
    foreign = _repo(tmp_path / "foreign2", remote=FOREIGN_URL)

    raised = _fact(orphan, system, treat_as_body=True)
    assert (raised.body, raised.how) == ("true", "unknown")
    assert "treat_as_body" in raised.detail and layer_for(raised) == "body"

    assert _fact(foreign, system, treat_as_body=True) == _fact(foreign, system)
    assert _fact(system, system, treat_as_body=True) == _fact(system, system)
    assert _fact(system, system, treat_as_body=False).body == "true"  # nothing lowers a body


# --- the URL spelling -----------------------------------------------------------------

def test_remote_url_normalization_identifies_one_remote_under_many_spellings():
    same = {normalize_remote_url(url) for url in (
        "https://github.com/razzant/ouroboros.git", "https://GitHub.com/razzant/ouroboros/",
        "git@github.com:razzant/ouroboros.git", "ssh://git@github.com/razzant/ouroboros",
        "https://user:token@github.com/razzant/ouroboros.git")}
    assert same == {"github.com/razzant/ouroboros"}
    assert normalize_remote_url("https://host:8443/a/b.git") == "host:8443/a/b"
    assert normalize_remote_url("") == ""
    # A local path is followed, never compared: it keeps its own spelling.
    assert normalize_remote_url("/srv/git/ouroboros") == "/srv/git/ouroboros"
    assert rbf._is_local_path("/srv/git/ouroboros") and rbf._is_local_path("../ouro")
    assert rbf._is_local_path("file:///srv/git/ouroboros") and rbf._is_local_path("C:\\git\\ouro")
    assert not rbf._is_local_path("git@github.com:razzant/ouroboros.git")
    assert not rbf._is_local_path("https://github.com/razzant/ouroboros.git")


def test_the_vocabulary_is_closed():
    assert rbf.BODY_FACT_VALUES == ("true", "false", "unknown")
    assert set(rbf.HOW_VALUES) == {"dir", "git_common_dir", "remote_chain", "managed_remote",
                                   "install_fork", "copy_origin", "unknown"}
    assert rbf.CHECKLIST_LAYERS == ("core", "body")
