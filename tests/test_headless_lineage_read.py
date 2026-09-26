"""A child reads its lineage's task files on the drive they live on (#1260).

Regression for a forked parent on its own headless drive
(``data/state/headless_tasks/<parent>/data``). The fixture derives the address
from the host helper and creates known bytes there before calling the real
``read_file``: previously ``lineage_read_base`` looked for lineage task roots
only on the canonical data root and on the child's OWN drives. Now each
``lineage_task_ids`` member is also looked up on ITS OWN headless drive
(``headless.task_state_dir(canonical, id) / "data"``): that drive pairs with that
id only (no drive x id cross-product), nothing is found by walking the disk, a
headless task root reached through a symlink is not a lineage base, and the grant
stays READ-only. The canonical-drive lineage read is unchanged.
"""
from __future__ import annotations

import pathlib
import shutil
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.headless import task_state_dir
from ouroboros.tool_access import _resolve_target_in_selected_base, lineage_read_base
from ouroboros.tools.registry import ToolContext, ToolRegistry

PARENT = "p07499dc017c01f83"
ROOT = "r5173b7c3c15d4c0b"
CHILD = "c11ae4fd0aa111111"
SIBLING = "s339e97de0b222222"
STRANGER = "x8b8af5e9cc333333"
SUCCESSOR = "s2f0a1b2c3d4e5f60"
PRED = "pred-1"
GRANDPRED = "pred-0"


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _symlink_or_skip(link: pathlib.Path, target: pathlib.Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")


@pytest.fixture
def geometry(tmp_path, monkeypatch):
    """Every task runs forked on its own headless drive under the canonical root,
    as on a live install; the parent and root also have canonical-drive files."""
    home, repo, canonical = tmp_path / "home", tmp_path / "repo", tmp_path / "data"
    for path in (home, repo, canonical):
        path.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    _write(repo / "README.md", "repo readme\n")

    def drive(task_id):
        path = task_state_dir(canonical, task_id) / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path

    geo = SimpleNamespace(home=home, repo=repo, canonical=canonical, outside=tmp_path / "outside", drive=drive)
    geo.parent_brief = _write(drive(PARENT) / "task_drives" / PARENT / "brief.md", "PARENT_HEADLESS_BRIEF\n")
    geo.parent_canonical = _write(canonical / "task_drives" / PARENT / "canon.md", "PARENT_CANONICAL_BYTES\n")
    geo.root_report = _write(
        task_artifact_dir_path(drive(ROOT), ROOT) / "report.txt", "ROOT_HEADLESS_ARTIFACT\n")
    geo.root_canonical = _write(
        task_artifact_dir_path(canonical, ROOT) / "final.txt", "ROOT_CANONICAL_ARTIFACT\n")
    geo.sibling_notes = _write(drive(SIBLING) / "task_drives" / SIBLING / "notes.txt", "SIBLING_BYTES\n")
    geo.stranger_out = _write(
        task_artifact_dir_path(drive(STRANGER), STRANGER) / "out.txt", "STRANGER_BYTES\n")
    # The parent's drive holds a directory named after the ROOT: a drive pairs with its own id only.
    geo.cross = _write(drive(PARENT) / "task_drives" / ROOT / "x.txt", "CROSS_BYTES\n")
    geo.pred_notes = _write(drive(PRED) / "task_drives" / PRED / "notes.md", "PRED_HEADLESS_BYTES\n")
    geo.grand_notes = _write(drive(GRANDPRED) / "task_drives" / GRANDPRED / "notes.md", "GRANDPRED_BYTES\n")
    return geo


def child_registry(geo):
    """A read-only child of PARENT under ROOT, on its own headless drive."""
    ctx = ToolContext(repo_dir=geo.repo, drive_root=geo.drive(CHILD), task_id=CHILD)
    ctx.budget_drive_root = str(geo.canonical)
    ctx.task_metadata = {
        "delegation_role": "subagent", "parent_task_id": PARENT, "root_task_id": ROOT,
        "budget_drive_root": str(geo.canonical),
    }
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    registry = ToolRegistry(repo_dir=geo.repo, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry, ctx


def continuation_registry(geo):
    """A top-level forked continuation of PRED (whose own predecessor is GRANDPRED)."""
    ctx = ToolContext(
        repo_dir=geo.repo, drive_root=geo.drive(SUCCESSOR), task_id=SUCCESSOR, memory_mode="forked",
        budget_drive_root=str(geo.canonical),
    )
    ctx.task_metadata = {"memory_mode": "forked", "budget_drive_root": str(geo.canonical)}
    ctx.task_contract = {"predecessor_authority": {
        "kind": "bounded_continuation_envelope", "status": "done", "previous_task_id": GRANDPRED,
        "source": {"kind": "task_result", "task_id": PRED, "human_label": f"task {PRED}",
                   "tool": "get_task_result", "arguments": {"task_id": PRED, "include_authority": True}},
    }}
    registry = ToolRegistry(repo_dir=geo.repo, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry, ctx


def _reads(registry, root, path):
    """The same absolute path read with the root NAMED and with the root INFERRED."""
    return (registry.execute("read_file", {"root": root, "path": str(path)}),
            registry.execute("read_file", {"path": str(path)}))


# --- the parent's and the root's headless task files are readable ----------------

def test_child_reads_its_headless_parents_brief_named_and_inferred(geometry):
    registry, ctx = child_registry(geometry)

    for out in _reads(registry, "task_drive", geometry.parent_brief):
        assert "PARENT_HEADLESS_BRIEF" in out, out
        assert ctx.last_read_view["opened_root"] == "task_drive"
        assert ctx.last_read_view["target"] == str(geometry.parent_brief.resolve())
    for out in _reads(registry, "task_drive", geometry.parent_canonical):  # canonical drive unchanged
        assert "PARENT_CANONICAL_BYTES" in out, out
    assert lineage_read_base(ctx, "task_drive", geometry.parent_brief) == geometry.parent_brief.parent.resolve()


def test_child_reads_its_headless_roots_artifact_named_and_inferred(geometry):
    registry, ctx = child_registry(geometry)

    for out in _reads(registry, "artifact_store", geometry.root_report):
        assert "ROOT_HEADLESS_ARTIFACT" in out, out
        assert ctx.last_read_view["opened_root"] == "artifact_store"
    for out in _reads(registry, "artifact_store", geometry.root_canonical):
        assert "ROOT_CANONICAL_ARTIFACT" in out, out


# --- nothing outside the lineage, and no drive x id cross-product ----------------

def test_sibling_stranger_and_cross_paired_headless_files_stay_refused(geometry):
    registry, ctx = child_registry(geometry)

    for root, path, secret in (
        ("task_drive", geometry.sibling_notes, "SIBLING_BYTES"),
        ("artifact_store", geometry.stranger_out, "STRANGER_BYTES"),
        ("task_drive", geometry.cross, "CROSS_BYTES"),
    ):
        named, inferred = _reads(registry, root, path)
        assert secret not in named and f"outside selected root={root}" in named, named
        assert secret not in inferred and "outside selected root=active_workspace" in inferred, inferred
        assert lineage_read_base(ctx, root, path) is None


def test_a_symlink_inside_the_parents_headless_drive_does_not_escape(geometry):
    registry, _ctx = child_registry(geometry)
    link = geometry.parent_brief.parent / "escape"
    _symlink_or_skip(link, geometry.sibling_notes.parent)

    for out in _reads(registry, "task_drive", link / "notes.txt"):
        assert "SIBLING_BYTES" not in out and "outside selected root" in out, out


@pytest.mark.parametrize("linked", ["task_drive", "headless_drive"])
def test_a_symlinked_headless_task_root_is_not_a_lineage_base(geometry, linked):
    """The ROOT's headless drive (or its task_drive) points elsewhere: the lexical
    lineage address must not become a door to whatever the link names."""
    registry, ctx = child_registry(geometry)
    root_drive = task_state_dir(geometry.canonical, ROOT) / "data"
    _write(geometry.outside / "task_drives" / ROOT / "notes.txt", "ESCAPED_BYTES\n")
    if linked == "headless_drive":
        shutil.rmtree(root_drive)
        _symlink_or_skip(root_drive, geometry.outside)
    else:
        (root_drive / "task_drives").mkdir(parents=True)
        _symlink_or_skip(root_drive / "task_drives" / ROOT, geometry.outside / "task_drives" / ROOT)
    target = root_drive / "task_drives" / ROOT / "notes.txt"

    out = registry.execute("read_file", {"root": "task_drive", "path": str(target)})

    assert "ESCAPED_BYTES" not in out and "outside selected root=task_drive" in out, out
    assert lineage_read_base(ctx, "task_drive", target) is None


# --- the named predecessor, one hop, and the READ-only grant ---------------------

def test_continuation_reads_its_headless_predecessor_but_not_the_hop_before(geometry):
    registry, _ctx = continuation_registry(geometry)

    for out in _reads(registry, "task_drive", geometry.pred_notes):
        assert "PRED_HEADLESS_BYTES" in out, out
    for out in _reads(registry, "task_drive", geometry.grand_notes):
        assert "GRANDPRED_BYTES" not in out and "outside selected root" in out, out


def test_a_headless_lineage_drive_is_read_only_while_the_own_drive_stays_writable(geometry):
    registry, ctx = continuation_registry(geometry)
    target = geometry.pred_notes
    before = target.read_text(encoding="utf-8")

    write = registry.execute("write_file", {"root": "task_drive", "path": str(target), "content": "x"})
    edit = registry.execute("edit_text", {"root": "task_drive", "path": str(target),
                                          "old_str": "PRED", "new_str": "MINE"})
    own = registry.execute("write_file", {"root": "task_drive", "path": "scratch.txt", "content": "mine"})

    for out in (write, edit):
        assert out.startswith("⚠️") and "outside selected root=task_drive" in out, out
    assert target.read_text(encoding="utf-8") == before
    assert not own.startswith("⚠️"), own
    own_base = geometry.drive(SUCCESSOR) / "task_drives" / SUCCESSOR
    assert (own_base / "scratch.txt").read_text(encoding="utf-8") == "mine"
    assert _resolve_target_in_selected_base(
        ctx, root="task_drive", base_path=own_base, path=str(target), operation="read") == target.resolve()
    for operation in ("write", "edit"):
        with pytest.raises(ValueError):
            _resolve_target_in_selected_base(
                ctx, root="task_drive", base_path=own_base, path=str(target), operation=operation)
