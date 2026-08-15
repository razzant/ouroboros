# tests/test_execution_facts.py — the placement-neutral fact accessor (RWS v2 §3.1).
#
# Two contracts are pinned here:
#   1. LocalExecutionFacts is byte-identical to the direct pathlib/subprocess
#      probes the guards do today (canonicalization, exists/is_dir semantics,
#      git worktree identity, interpreter provability). If it drifts, converting
#      a guard to facts would silently change local behavior.
#   2. An ssh placement raises the TYPED SSH_FACTS_UNAVAILABLE for every fact —
#      never a Home probe standing in for a remote one.
import pathlib
import subprocess
import sys
import types

import pytest

from ouroboros import execution_facts
from ouroboros.execution_facts import (
    SSH_FACTS_UNAVAILABLE,
    ExecutionFacts,
    LOCAL_FACTS,
    LocalExecutionFacts,
    RemoteExecutionFacts,
    RemoteFactsUnavailableError,
    facts_for_ref,
)
from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY


def _ctx(metadata=None):
    return types.SimpleNamespace(task_metadata=dict(metadata or {}))


def _local_ctx(root: pathlib.Path):
    return _ctx({SEALED_WORKSPACE_REF_KEY: {"kind": "local", "local_root": str(root)}})


def _ssh_ctx():
    return _ctx({
        SEALED_WORKSPACE_REF_KEY: {
            "kind": "ssh",
            "connection_id": "conn-1",
            "remote_root": "/srv/work/app",
            "workspace_id": "ws-1",
        }
    })


def _facts_for(ctx):
    """Test-local convenience: what a caller does with a ctx it has not read yet.

    Production has exactly one accessor constructor (`facts_for_ref`, fed by the
    prepare phase's single placement read); this helper keeps the SELECTION tests
    about selection without reintroducing a second door into the placement.
    """
    from ouroboros.workspace_ref import workspace_ref_for

    return facts_for_ref(workspace_ref_for(ctx))


# ── selection ────────────────────────────────────────────────────────────────
def test_facts_for_selects_local_for_local_and_legacy_placement(tmp_path):
    assert _facts_for(_local_ctx(tmp_path)) is LOCAL_FACTS
    assert _facts_for(_ctx()) is LOCAL_FACTS  # legacy: no sealed ref at all
    assert _facts_for(types.SimpleNamespace()) is LOCAL_FACTS


def test_facts_for_selects_remote_accessor_for_ssh_placement():
    facts = _facts_for(_ssh_ctx())
    assert isinstance(facts, RemoteExecutionFacts)
    assert facts.placement == "ssh"
    assert facts.ref.connection_id == "conn-1"


def test_both_accessors_satisfy_the_protocol():
    assert isinstance(LOCAL_FACTS, ExecutionFacts)
    assert isinstance(_facts_for(_ssh_ctx()), ExecutionFacts)


# ── local parity with the direct probes ──────────────────────────────────────
def test_canonical_path_matches_expanduser_resolve(tmp_path):
    nested = tmp_path / "a" / ".." / "b" / "c.txt"
    assert LOCAL_FACTS.canonical_path(nested) == str(
        pathlib.Path(nested).expanduser().resolve(strict=False)
    )
    assert LOCAL_FACTS.canonical_path("~") == str(pathlib.Path("~").expanduser().resolve(strict=False))
    # expanduser is identity for absolute/relative spellings: bare resolve parity.
    assert LOCAL_FACTS.canonical_path(tmp_path) == str(pathlib.Path(tmp_path).resolve(strict=False))


@pytest.mark.parametrize("kind", ["dir", "file", "missing"])
def test_path_fact_kind_matches_pathlib_predicates(tmp_path, kind):
    if kind == "dir":
        target = tmp_path / "d"
        target.mkdir()
    elif kind == "file":
        target = tmp_path / "f.txt"
        target.write_text("abcd", encoding="utf-8")
    else:
        target = tmp_path / "nope"
    fact = LOCAL_FACTS.path_fact(target)
    assert fact.kind == kind
    assert fact.exists is target.exists()
    assert fact.is_dir is target.is_dir()
    assert fact.is_file is target.is_file()
    assert fact.canonical == str(target.resolve(strict=False))
    assert fact.requested == str(target)
    # size is the stat size for anything that exists, -1 for a missing path.
    assert fact.size == (target.stat().st_size if kind != "missing" else -1)


def test_path_fact_follows_symlinks_like_is_dir_and_flags_the_link(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    fact = LOCAL_FACTS.path_fact(link)
    assert fact.kind == "dir" and fact.is_dir is link.is_dir()
    assert fact.symlink is True
    assert fact.canonical == str(real.resolve(strict=False))


def test_broken_symlink_reads_as_missing_like_exists(tmp_path):
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "absent")
    fact = LOCAL_FACTS.path_fact(link)
    assert link.exists() is False
    assert fact.exists is False and fact.kind == "missing"
    assert fact.symlink is True


def test_path_facts_fans_out_in_order(tmp_path):
    (tmp_path / "one").mkdir()
    facts = LOCAL_FACTS.path_facts([tmp_path / "one", tmp_path / "two"])
    assert [f.kind for f in facts] == ["dir", "missing"]


def test_git_fact_matches_the_admission_probe(tmp_path):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True, capture_output=True)
    top = LOCAL_FACTS.git_fact(repo)
    assert top.is_worktree and top.is_worktree_root
    assert top.toplevel == str(repo.resolve(strict=False))
    # A subdirectory IS in a worktree but is NOT the worktree root — the exact
    # distinction workspace_admission.validate_workspace_root enforces.
    sub = LOCAL_FACTS.git_fact(repo / "sub")
    assert sub.is_worktree and not sub.is_worktree_root
    assert sub.toplevel == str(repo.resolve(strict=False))


def test_git_fact_on_a_non_worktree_is_not_an_error(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    fact = LOCAL_FACTS.git_fact(plain)
    assert fact.toplevel == "" and not fact.is_worktree
    assert fact.cwd == str(plain.resolve(strict=False))


def test_git_fact_swallows_probe_failure_as_not_a_worktree(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("git missing")

    monkeypatch.setattr(execution_facts.subprocess, "run", boom)
    assert LOCAL_FACTS.git_fact(tmp_path).toplevel == ""


def test_interpreter_fact_delegates_to_the_single_probe():
    from ouroboros.python_interpreter import usable_executable

    fact = LOCAL_FACTS.interpreter_fact(sys.executable)
    assert fact.requested == sys.executable
    assert fact.resolved == usable_executable(sys.executable)
    assert fact.usable is True
    missing = LOCAL_FACTS.interpreter_fact("definitely-not-an-interpreter-xyz")
    assert missing.resolved == "" and missing.usable is False


# ── ssh placement: typed refusal on every fact ───────────────────────────────
@pytest.mark.parametrize(
    "call",
    [
        lambda f: f.canonical_path("/srv/work/app"),
        lambda f: f.path_fact("/srv/work/app"),
        lambda f: f.path_facts(["/srv/work/app"]),
        lambda f: f.git_fact("/srv/work/app"),
        lambda f: f.interpreter_fact("python3"),
    ],
)
def test_every_remote_fact_fails_typed(call):
    facts = _facts_for(_ssh_ctx())
    with pytest.raises(RemoteFactsUnavailableError) as excinfo:
        call(facts)
    assert excinfo.value.code == SSH_FACTS_UNAVAILABLE
    text = str(excinfo.value)
    assert text.startswith(SSH_FACTS_UNAVAILABLE)
    # The refusal names the placement identity, never a Home path.
    assert "conn-1" in text and "ws-1" in text


def test_remote_facts_never_fall_back_to_a_home_probe(tmp_path, monkeypatch):
    """A remote fact must not be answerable by probing Home: patching the local
    accessor's probes to explode proves the remote path never reaches them."""

    def boom(*args, **kwargs):
        raise AssertionError("remote facts must not probe the Home filesystem")

    monkeypatch.setattr(LocalExecutionFacts, "path_fact", boom)
    monkeypatch.setattr(LocalExecutionFacts, "git_fact", boom)
    facts = _facts_for(_ssh_ctx())
    with pytest.raises(RemoteFactsUnavailableError):
        facts.path_fact(str(tmp_path))
    with pytest.raises(RemoteFactsUnavailableError):
        facts.git_fact(str(tmp_path))
