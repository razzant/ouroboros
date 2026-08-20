"""S6 C3/C4 — the private-snapshot registry when it cannot be read or written.

``state/subagent_worktrees.json`` is the third durable registry in the family
(beside ``state/cancel_intents.json`` and ``state/terminal_deliveries.json``)
and it now answers "malformed" the way they do: absent stays an ordinary empty
registry, malformed refuses the mutation, keeps the bytes and discloses one
typed ``subagent_worktree_registry_corrupt`` event. Pre-fix a live snapshot
read as missing, the startup GC reported a clean sweep, and the next
reconciliation overwrote the malformed bytes with a valid empty registry —
stranding the checkout and the ``refs/ouroboros/delegated/*`` ref that pins its
baseline with nothing left naming them.

C4 pins the write half: a failed registration now removes the checkout AND the
baseline ref on the Git branch, the symmetry the payload sibling already had
(``tests/test_delegated_skill_payload.py::test_registry_save_failure_leaves_no_orphan_snapshot_dir``).
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from ouroboros import subagent_worktrees as wt


MALFORMED = '"not a registry"'


def _git(cwd, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check,
    )


def _seed_target(tmp_path: pathlib.Path) -> pathlib.Path:
    target = tmp_path / "target"
    target.mkdir()
    _git(target, "init")
    (target / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(target, "add", "-A")
    _git(target, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed")
    return target


def _registry(data_dir: pathlib.Path) -> pathlib.Path:
    return data_dir / "state" / "subagent_worktrees.json"


def _events(data_dir: pathlib.Path):
    path = data_dir / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _snapshot(tmp_path, snapshot_id="snapS6"):
    """One registered delegated execution snapshot; returns (target, snaps, data, handle)."""
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    handle = wt.provision_execution_snapshot(
        target_root=target, task_id="t1", snapshot_id=snapshot_id,
        worktree_root=snaps, data_dir=data)
    return target, snaps, data, handle


# ---------------------------------------------------------------------------
# C3 — absent is empty, malformed is refused
# ---------------------------------------------------------------------------


def test_c3_absent_is_an_ordinary_empty_registry_in_both_modes(tmp_path):
    """The strictness must separate ABSENT from MALFORMED: a never-written
    registry is the first-write case, never a refusal."""
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)

    assert wt._load_registry(data_dir=data) == []
    assert wt._load_registry(data_dir=data, strict=True) == []
    assert wt.list_worktrees(data_dir=data) == []
    assert wt.find_execution_snapshot("nothing", data_dir=data) is None


@pytest.mark.parametrize(
    "payload", [MALFORMED, '{"worktrees": "nope"}', '{"worktrees": [', "\x00\x01"],
)
def test_c3_every_malformed_shape_is_refused_by_the_strict_read(tmp_path, payload):
    """C3/O2: malformed is a fact of its own — the soft read still answers
    empty for the UI listing, the strict read raises for anything that writes."""
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    _registry(data).write_text(payload, encoding="utf-8")

    assert wt._load_registry(data_dir=data) == [], "the inspection read stays soft"
    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt._load_registry(data_dir=data, strict=True)
    assert _registry(data).read_text(encoding="utf-8") == payload, "bytes are kept"
    assert any(
        row.get("type") == "subagent_worktree_registry_corrupt"
        for row in _events(data)
    ), "the refusal is disclosed durably"


def test_c3_a_live_snapshot_is_not_reported_missing_over_a_malformed_registry(tmp_path):
    """C3/O2: the lookup that decides "does this binding still exist?" must not
    answer "no" from a file it could not read — the checkout and its pinned
    baseline ref are right there, and a false "missing" sends the caller off to
    provision a replacement."""
    target, _snaps, data, handle = _snapshot(tmp_path)
    assert wt.find_execution_snapshot("snapS6", data_dir=data) is not None
    _registry(data).write_text(MALFORMED, encoding="utf-8")

    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt.find_execution_snapshot("snapS6", data_dir=data)
    assert pathlib.Path(handle.path).is_dir()
    assert _git(target, "rev-parse", handle.baseline_ref).stdout.strip() == handle.baseline_sha


def test_c3_the_startup_gc_refuses_to_sweep_an_unreadable_registry(tmp_path):
    """C3/O2: a destructive GC over an unknowable keep-set is exactly the case
    the delegated-snapshot prune already fails closed on when the custody log
    is unreadable (`server.py`). Reporting a clean sweep instead was the lie."""
    _target, snaps, data, handle = _snapshot(tmp_path)
    _registry(data).write_text(MALFORMED, encoding="utf-8")

    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt.prune_execution_snapshots(set(), worktree_root=snaps, data_dir=data)
    assert pathlib.Path(handle.path).is_dir(), "nothing was removed"


def test_c3_prune_orphans_never_overwrites_a_malformed_registry(tmp_path):
    """C3/O2, the destructive half: startup reconciliation used to rewrite the
    malformed bytes as a valid EMPTY registry, taking the only record of the
    checkout and its pinned ref with it. It now refuses and keeps the bytes."""
    target, snaps, data, handle = _snapshot(tmp_path)
    _registry(data).write_text(MALFORMED, encoding="utf-8")

    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt.prune_orphans(worktree_root=snaps, data_dir=data)

    assert _registry(data).read_text(encoding="utf-8") == MALFORMED, "recovery material"
    assert pathlib.Path(handle.path).is_dir()
    assert _git(target, "rev-parse", handle.baseline_ref, check=False).returncode == 0


def test_c3_a_new_snapshot_does_not_overwrite_a_malformed_registry(tmp_path):
    """C3/O2: provisioning reads-appends-writes, so a soft read would replace a
    malformed registry with one holding only the new row. It refuses instead —
    and the refused attempt leaves nothing behind (the O3 cleanup path)."""
    target, snaps, data, first = _snapshot(tmp_path, snapshot_id="snapOne")
    _registry(data).write_text(MALFORMED, encoding="utf-8")

    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt.provision_execution_snapshot(
            target_root=target, task_id="t1", snapshot_id="snapTwo",
            worktree_root=snaps, data_dir=data)

    assert _registry(data).read_text(encoding="utf-8") == MALFORMED
    assert pathlib.Path(first.path).is_dir(), "the registered snapshot is untouched"
    assert not (snaps / "dlg_t1_snapTwo").exists(), "the refused attempt cleans up"
    assert _git(
        target, "rev-parse", "refs/ouroboros/delegated/snapTwo", check=False,
    ).returncode != 0, "and leaves no pinned ref"


def test_c3_the_healthy_registry_paths_are_unchanged(tmp_path):
    """The fix must not cost the ordinary lifecycle anything."""
    target, snaps, data, handle = _snapshot(tmp_path)

    assert wt.find_execution_snapshot("snapS6", data_dir=data)["path"] == handle.path
    assert wt.prune_execution_snapshots(
        {"snapS6"}, worktree_root=snaps, data_dir=data,
    ) == {"removed": [], "kept": ["snapS6"]}
    assert wt.prune_orphans(worktree_root=snaps, data_dir=data) == {"removed": 0, "kept": 1}
    assert wt.remove_execution_snapshot("snapS6", worktree_root=snaps, data_dir=data) is True
    assert wt.list_worktrees(data_dir=data) == []


# ---------------------------------------------------------------------------
# C4 — a registry write that fails on the Git branch
# ---------------------------------------------------------------------------


def test_c4_a_git_branch_registry_write_failure_leaves_no_worktree_or_ref(
    tmp_path, monkeypatch,
):
    """C4/O3: registration is inside the cleanup scope on BOTH branches now.
    A failed registry write removes the checkout and deletes the baseline ref
    it pinned, instead of leaving a snapshot nothing can name plus a ref that
    holds its commit against git's own GC."""
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"

    def _boom(*_a, **_k):
        raise OSError("registry disk full")

    monkeypatch.setattr(wt, "_save_registry", _boom)
    with pytest.raises(OSError, match="registry disk full"):
        wt.provision_execution_snapshot(
            target_root=target, task_id="t1", snapshot_id="snapFail",
            worktree_root=snaps, data_dir=data)

    leftovers = sorted(p.name for p in snaps.glob("dlg_*")) if snaps.exists() else []
    assert leftovers == [], leftovers
    assert _git(
        target, "rev-parse", "refs/ouroboros/delegated/snapFail", check=False,
    ).returncode != 0, "the baseline ref is gone with the checkout"
    assert _git(target, "worktree", "list").stdout.count("dlg_t1_snapFail") == 0


def test_c4_b_the_acting_worktree_branch_cleans_up_on_registry_failure(
    tmp_path, monkeypatch,
):
    """C4/O3, the third provisioning branch: ``provision_worktree`` creates a
    checkout AND a task branch before it registers either. A corrupt registry
    (strict read) or a failed write must remove both — otherwise every retry
    strands one more unreclaimable worktree+branch pair, without bound."""
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"

    # Leg 1: corrupt registry — the strict read refuses AFTER the checkout
    # exists; the refused attempt must leave neither checkout nor branch.
    (data / "state").mkdir(parents=True)
    _registry(data).write_text(MALFORMED, encoding="utf-8")
    with pytest.raises(wt.SubagentWorktreeRegistryCorrupt):
        wt.provision_worktree(
            repo_dir=target, task_id="acting9", worktree_root=snaps, data_dir=data)
    assert not (snaps / "acting9").exists(), "the refused attempt cleans up"
    assert _git(
        target, "rev-parse", "--verify", f"{wt._BRANCH_PREFIX}acting9", check=False,
    ).returncode != 0, "and takes the task branch with it"
    assert _registry(data).read_text(encoding="utf-8") == MALFORMED

    # Leg 2: registry write failure over a healthy registry — same symmetry.
    _registry(data).unlink()

    def _boom(*_a, **_k):
        raise OSError("registry disk full")

    monkeypatch.setattr(wt, "_save_registry", _boom)
    with pytest.raises(OSError, match="registry disk full"):
        wt.provision_worktree(
            repo_dir=target, task_id="acting9", worktree_root=snaps, data_dir=data)
    assert not (snaps / "acting9").exists()
    assert _git(
        target, "rev-parse", "--verify", f"{wt._BRANCH_PREFIX}acting9", check=False,
    ).returncode != 0
    assert _git(target, "worktree", "list").stdout.count("acting9") == 0


# ---------------------------------------------------------------------------
# Disclosure — the registry's own shape
# ---------------------------------------------------------------------------


def test_the_registry_has_no_version_and_two_kind_discriminated_shapes(tmp_path):
    """Disclosure (MIGRATION_v7.md): unlike its two sibling registries this one
    carries NO ``schema_version``, and its two row shapes are told apart only
    by ``kind == "delegated_exec"``. A future format change has to add the
    discriminator it lacks before it can migrate anything.
    """
    target = _seed_target(tmp_path)
    snaps, data = tmp_path / "snaps", tmp_path / "data"
    wt.provision_worktree(
        repo_dir=target, task_id="acting1", worktree_root=snaps, data_dir=data)
    wt.provision_execution_snapshot(
        target_root=target, task_id="t1", snapshot_id="snapKind",
        worktree_root=snaps, data_dir=data)

    envelope = json.loads(_registry(data).read_text(encoding="utf-8"))
    assert set(envelope) == {"worktrees"}, "no version field to dispatch on"
    kinds = [row.get("kind", "") for row in envelope["worktrees"]]
    assert sorted(kinds) == ["", "delegated_exec"], (
        "the acting-worktree row carries no kind at all; absence IS the shape"
    )
