"""The body predicate decides the checklist layer from git facts and the
install manifest alone — never from a directory, branch or remote name.

Every case builds real git repositories under ``tmp_path``; the system
repository is a plain checkout that may carry an ``origin`` (the install's own
fork) and a manifest with a configured ``managed_remote_url``.
"""

from __future__ import annotations

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


def test_a_source_install_managed_remote_is_used_without_a_manifest(system, tmp_path):
    """A source checkout's maintained managed remote is the configured source."""
    _git(system, "remote", "add", "managed", MANAGED_URL)
    official = _repo(tmp_path / "official-source-clone", remote=MANAGED_URL)

    fact = body_fact(official, system_repo=system)

    assert (fact.body, fact.how) == ("true", "managed_remote")
    foreign = _repo(tmp_path / "foreign-source-clone", remote=FOREIGN_URL)
    assert body_fact(foreign, system_repo=system).body == "false"


def test_the_managed_url_is_the_installs_configured_source_not_a_fixed_official_one(system, tmp_path):
    mirror = _repo(tmp_path / "mirror-clone", remote="https://mirror.example.org/forks/ouro.git")

    assert _fact(mirror, system, manifest={"managed_remote_url": "git@mirror.example.org:forks/ouro"}).how == "managed_remote"
    foreign = _fact(mirror, system)  # the default manifest points elsewhere
    assert (foreign.body, foreign.how) == ("false", "remote_chain")


@pytest.mark.parametrize("legacy", ["https://github.com/joi-lab/ouroboros-desktop",
                                    "https://github.com/joi-lab/ouroboros"])
def test_a_former_official_home_in_the_manifest_resolves_like_the_updater(system, tmp_path, legacy):
    """Owner 2026-10-11 №3=A: builds before v6.8.0 keep an archived joi-lab address
    in their manifest while the updater follows the official repository, so a
    clone of the official repository is the body; a custom mirror still is not."""
    official = _repo(tmp_path / "official-clone", remote=MANAGED_URL)
    fact = _fact(official, system, manifest={"managed_remote_url": legacy})
    assert (fact.body, fact.how) == ("true", "managed_remote")
    mirror = _repo(tmp_path / "mirror-clone", remote="https://mirror.example.org/forks/ouro.git")
    assert _fact(mirror, system, manifest={"managed_remote_url": legacy}).body == "false"


@pytest.mark.parametrize("legacy", ["https://github.com/joi-lab/ouroboros.git",
                                    "https://github.com/joi-lab/ouroboros-desktop"])
def test_an_install_cloned_from_a_former_home_still_recognizes_its_own_origin(tmp_path, legacy):
    """The former-home mapping applies to the managed source only: a source
    install whose origin is the old joi-lab address keeps its clones as the body."""
    system = _repo(tmp_path / "legacy-system", remote=legacy)
    clone = _repo(tmp_path / "legacy-clone", remote=legacy)
    fact = _fact(clone, system, manifest={})
    assert (fact.body, fact.how) == ("true", "install_fork")


def test_a_remote_at_the_installs_own_origin_is_the_body_as_install_fork(system, tmp_path):
    root = _repo(tmp_path / "fork-clone", remote="https://github.com/someone/ouroboros-fork")

    fact = _fact(root, system)

    assert (fact.body, fact.how) == ("true", "install_fork")


def test_an_install_without_a_managed_url_is_still_the_body_by_dir_and_common_git_dir(system, tmp_path):
    worktree = tmp_path / "wt2"
    _git(system, "worktree", "add", "-q", str(worktree), "-b", "other")

    assert _fact(system, system, manifest={}).how == "dir"
    assert _fact(worktree, system, manifest={}).how == "git_common_dir"


def test_a_real_partial_clone_is_the_body_through_its_annotated_remote(system, tmp_path, monkeypatch):
    """#3004: ``git remote -v`` shows a partial clone's fetch URL as ``<url> (fetch) [blob:none]``. The predicate
    reads the remote's configured URL, so that display hides neither a local chain nor the configured source."""
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")  # nothing here may reach a network
    _git(system, "config", "uploadpack.allowFilter", "true")
    partial = tmp_path / "partial"
    _git(tmp_path, "clone", "-q", "--filter=blob:none", system.as_uri(), str(partial))
    _git(partial, "remote", "set-url", "origin", str(system))
    major, minor = map(int, _git(tmp_path, "--version").split()[2].split(".")[:2])
    shown = " [blob:none]" if (major, minor) >= (2, 37) else ""  # Git writes the filter there since 2.37
    assert f"origin\t{system} (fetch){shown}" in _git(partial, "remote", "-v").splitlines()
    chained = _fact(partial, system)
    assert (chained.body, chained.how) == ("true", "remote_chain")
    _git(partial, "remote", "set-url", "origin", MANAGED_URL)
    fact = _fact(partial, system)
    assert (fact.body, fact.how) == ("true", "managed_remote") and MANAGED_URL in fact.detail


def _filtered(root: pathlib.Path, name: str = "origin") -> None:
    """The partial-clone filter Git shows after the fetch URL in ``git remote -v`` (``[blob:none]``)."""
    _git(root, "config", f"remote.{name}.partialclonefilter", "blob:none")


def _body_clone(root: pathlib.Path, system: pathlib.Path) -> None:
    _git(root, "clone", "-q", str(system), str(root / "origin"))  # a folder named like the remote


def _shape(name: str, root: pathlib.Path, system: pathlib.Path, monkeypatch) -> None:
    if name == "foreign":
        _git(root, "remote", "add", "origin", FOREIGN_URL)
        _filtered(root)
    elif name == "body_beside_plain_foreign":
        _git(root, "remote", "add", "acme", FOREIGN_URL)
        _git(root, "remote", "add", "origin", MANAGED_URL)
        _filtered(root)
    elif name == "insteadof":
        _git(root, "remote", "add", "origin", "gh:razzant/ouroboros")
        _git(root, "config", "url.https://github.com/.insteadOf", "gh:")
        _filtered(root)
    elif name == "first_url_decides":
        _git(root, "remote", "add", "origin", FOREIGN_URL)
        _git(root, "config", "--add", "remote.origin.url", MANAGED_URL)
        _git(root, "config", "remote.origin.pushurl", MANAGED_URL)
        _filtered(root)
    elif name in ("push_only_name_rewritten", "reset_url_name_rewritten"):
        if name == "push_only_name_rewritten":
            _git(root, "config", "remote.origin.pushurl", MANAGED_URL)
        else:
            _git(root, "config", "remote.origin.url", FOREIGN_URL)
            _git(root, "config", "--add", "remote.origin.url", "")  # an empty value clears the list
        _git(root, "config", f"url.{MANAGED_URL}.insteadOf", "origin")  # Git's fallback would name the body
    elif name == "push_only_beside_body_folder":
        _body_clone(root, system)
        _git(root, "config", "remote.origin.pushurl", FOREIGN_URL)
    elif name == "same_name_local_url":
        _body_clone(root, system)
        _git(root, "remote", "add", "origin", "origin")
        _filtered(root)
    elif name == "dotted_name":
        _git(root, "remote", "add", "team/my.fork", MANAGED_URL)
        _filtered(root, "team/my.fork")
    elif name == "slash_name_git_skips":  # named like the path of a body clone; Git skips the remote (warning)
        clone = root.parent / f"{root.name}-body"
        _git(root.parent, "clone", "-q", str(system), str(clone))
        _git(root, "config", f"remote./{clone.as_posix().lstrip('/')}.url", FOREIGN_URL)
    elif name == "former_home":
        _git(root, "remote", "add", "origin", "https://github.com/joi-lab/ouroboros-desktop")
        _filtered(root)
    elif name == "global_only":
        config = root.parent / f"{root.name}.gitconfig"
        config.write_text(f'[remote "upstream"]\n\turl = {MANAGED_URL}\n\tpartialclonefilter = blob:none\n',
                          encoding="utf-8")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))


FETCH_URL_SHAPES = {
    "foreign": ("false", "remote_chain"),  # a filtered foreign remote is foreign, not unknown
    "body_beside_plain_foreign": ("true", "managed_remote"),  # a hidden body remote no longer leaves `false`
    "insteadof": ("true", "managed_remote"),  # Git's url.<base>.insteadOf rewrite applies
    "first_url_decides": ("false", "remote_chain"),  # the first URL fetches; later URLs and the pushurl do not
    "push_only_name_rewritten": ("unknown", "unknown"),  # Git's own-name fallback is never a fetch URL
    "reset_url_name_rewritten": None,  # as the installed Git reads its empty last URL, below
    "push_only_beside_body_folder": ("unknown", "unknown"),  # ... nor a folder followed under that name
    "same_name_local_url": ("true", "remote_chain"),  # an explicit URL equal to its name is followed
    "dotted_name": ("true", "managed_remote"),  # a name keeps its dots and inner slash
    "slash_name_git_skips": ("unknown", "unknown"),  # never the name answered as a URL and followed
    "former_home": ("true", "managed_remote"),
    "global_only": ("true", "managed_remote"),  # every config scope Git reads, as `git remote -v` lists
}


@pytest.mark.parametrize("shape", list(FETCH_URL_SHAPES))
def test_a_remote_is_judged_by_its_fetch_url_as_git_resolves_it(system, tmp_path, monkeypatch, shape):
    """Each row is one shape of real Git configuration; the filtered ones would show ``[blob:none]``."""
    root = _repo(tmp_path / shape)
    _shape(shape, root, system, monkeypatch)
    fact = _fact(root, system)
    expected = FETCH_URL_SHAPES[shape]
    if expected is None:  # Git 2.46+ clears the list at the empty value; an older Git keeps the URL before it,
        kept = f"{FOREIGN_URL} (fetch)" in _git(root, "remote", "-v")  # and only then lists it as the fetch URL
        expected = ("false", "remote_chain") if kept else ("unknown", "unknown")
    assert (fact.body, fact.how) == expected, fact.detail
    if shape == "dotted_name":
        assert "remote team/my.fork " in fact.detail
    if expected[0] == "unknown":
        assert "has no remote" in fact.detail


def test_a_registered_copy_is_judged_by_its_recorded_source(system, tmp_path):
    from ouroboros import subagent_worktrees

    data_dir = tmp_path / "data"
    body_copy = _repo(tmp_path / "copy-of-body")
    foreign_copy = _repo(tmp_path / "copy-of-other")
    for row in ({"path": str(body_copy), "repo_dir": str(system), "source_is_system_repo": True,
                 "base_sha": "a" * 40, "kind": "self_worktree"},
                {"path": str(foreign_copy), "repo_dir": str(tmp_path / "elsewhere"),
                 "source_is_system_repo": False, "base_sha": "b" * 40, "kind": "self_worktree"}):
        subagent_worktrees._put_record(row, data_dir)  # each copy's own record file

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


@pytest.mark.parametrize("failing", ["", "config", "ls-remote"])  # every call, or either call that reads the remotes
def test_a_git_failure_is_unknown_never_false(system, tmp_path, monkeypatch, failing):
    root = _repo(tmp_path / "timeout", remote=FOREIGN_URL)
    real_run = subprocess.run

    def _timeout(argv, *args, **kwargs):
        if failing and argv[1] != failing:
            return real_run(argv, *args, **kwargs)
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


def test_a_name_the_filesystem_refuses_is_a_failed_probe_not_an_error(tmp_path):
    # An over-long component or an embedded NUL is a failed probe (None), never an exception (#2918).
    assert rbf.repository_root(tmp_path / ("x" * 300) / "new.py") is None
    assert rbf.repository_root(str(tmp_path / "bad\x00name.py")) is None
