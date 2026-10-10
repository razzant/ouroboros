"""A top-level continuation reads the ONE predecessor it continues (owner 3A, #1232).

Measured on a live install: a continuation admitted with ``memory_mode=forked``
runs on its own headless drive (``data/state/headless_tasks/<id>/data``), called
``get_task_result(include_authority=True)`` on its predecessor, saw a registered
artifact path plus hash, and ``read_file`` on that path was refused ("outside
artifact root"). The lineage rule (T4=A, #1105) named the actor's own, parent's
and root's task files; the predecessor was not in that set. Now
``lineage_task_ids`` appends the predecessor's validated id from the ONE carrier a
running ToolContext has: ``task_contract["predecessor_authority"]["source"]
["task_id"]``, minted by ``validate_task_authority_sources`` at startup binding
(``agent_startup_checks``) and carried into the contract by
``build_task_contract``; ``task_metadata`` is read only when no contract rides
(the agent never copies the envelope into metadata). One hop only: the
envelope's ``previous_task_id`` is never read. READ only: a write into the
predecessor's drive stays refused while the task's own drive stays writable.
The read follows the source the actor carries: a delegated child's predecessor
brief retains its parent's source and reads that predecessor's files too,
read-only, exactly as it reads its parent's and root's; a child without the
brief keeps parent/root.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tool_access import (
    _resolve_target_in_selected_base,
    lineage_read_base,
    lineage_task_ids,
)
from ouroboros.tools.registry import ToolContext, ToolRegistry

SUCCESSOR = "s2f0a1b2c3d4e5f60"
PRED = "pred-1"
GRANDPRED = "pred-0"
OTHER = "other-task"
PARENT = "p07499dc017c01f83"
ROOT = "r5173b7c3c15d4c0b"
CHILD = "c11ae4fd0aa111111"


def host_pointer(task_id):
    """The exact host-issued actor pointer ``valid_task_result_authority_source`` accepts."""
    return {
        "kind": "task_result",
        "task_id": task_id,
        "human_label": f"task {task_id}",
        "tool": "get_task_result",
        "arguments": {"task_id": task_id, "include_authority": True},
    }


def predecessor_envelope(task_id, *, previous=""):
    """The shape startup binding mints: a bounded envelope whose chain cursor names
    the hop BEFORE the predecessor and whose pull pointer rides LAST under ``source``."""
    return {
        "kind": "bounded_continuation_envelope",
        "status": "done",
        "previous_task_id": previous,
        "source": host_pointer(task_id),
    }


@pytest.fixture
def geometry(tmp_path, monkeypatch):
    """The predecessor's task files live on the CANONICAL data root; the
    continuation runs on its own forked (headless) drive; the owner home is a
    fake tmp home. The predecessor's own predecessor and a stranger have files
    too, so the one-hop and no-walking rules have something to refuse."""
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    canonical = tmp_path / "data"
    headless = tmp_path / "state" / "headless_tasks" / SUCCESSOR / "data"
    for path in (home, repo, canonical, headless):
        path.mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    (repo / "README.md").write_text("repo readme\n", encoding="utf-8")
    pred_drive = canonical / "task_drives" / PRED
    pred_drive.mkdir(parents=True)
    (pred_drive / "notes.md").write_text("PRED_DRIVE_BYTES\n", encoding="utf-8")
    pred_artifacts = task_artifact_dir_path(canonical, PRED, create=True)
    (pred_artifacts / "report.pdf.txt").write_text("PRED_ARTIFACT_BYTES\n", encoding="utf-8")
    grand_drive = canonical / "task_drives" / GRANDPRED
    grand_drive.mkdir(parents=True)
    (grand_drive / "notes.md").write_text("GRANDPRED_BYTES\n", encoding="utf-8")
    other_drive = canonical / "task_drives" / OTHER
    other_drive.mkdir(parents=True)
    (other_drive / "notes.md").write_text("OTHER_BYTES\n", encoding="utf-8")
    other_artifacts = task_artifact_dir_path(canonical, OTHER, create=True)
    (other_artifacts / "out.txt").write_text("OTHER_ARTIFACT_BYTES\n", encoding="utf-8")
    return SimpleNamespace(
        home=home, repo=repo, canonical=canonical, headless=headless,
        pred_drive=pred_drive, pred_artifacts=pred_artifacts, grand_drive=grand_drive,
        other_drive=other_drive, other_artifacts=other_artifacts,
    )


def continuation_registry(geo, *, predecessor=PRED):
    """A top-level continuation of ``predecessor`` admitted with ``memory_mode=forked``:
    it executes on its headless drive while the canonical root stays its budget root.
    ``predecessor=None`` gives the same task without any predecessor authority."""
    ctx = ToolContext(
        repo_dir=geo.repo, drive_root=geo.headless, task_id=SUCCESSOR, memory_mode="forked",
        budget_drive_root=str(geo.canonical),
    )
    ctx.task_metadata = {"memory_mode": "forked", "budget_drive_root": str(geo.canonical)}
    ctx.task_contract = {"predecessor_authority": predecessor_envelope(
        predecessor, previous=GRANDPRED)} if predecessor else {}
    registry = ToolRegistry(repo_dir=geo.repo, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry, ctx


def child_registry(geo):
    """A delegated read-only child of PARENT under ROOT whose contract carries the
    predecessor source through the real child-contract builder."""
    from ouroboros.tools.control_scheduling import _build_child_subagent_contract

    ctx = ToolContext(repo_dir=geo.repo, drive_root=geo.headless, task_id=CHILD)
    ctx.budget_drive_root = str(geo.canonical)
    ctx.task_metadata = {
        "delegation_role": "subagent",
        "parent_task_id": PARENT,
        "root_task_id": ROOT,
        "budget_drive_root": str(geo.canonical),
    }
    ctx.task_contract = _build_child_subagent_contract({
        "tid": CHILD, "parent_task_id": PARENT, "root_task_id": ROOT,
        "parent_contract": {"predecessor_authority": predecessor_envelope(PRED)},
    })
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent", allow_enable=False)
    registry = ToolRegistry(repo_dir=geo.repo, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry, ctx


# --- (a) the continuation reads its predecessor's task files from a forked drive ---

def test_continuation_reads_its_predecessors_task_drive_from_a_forked_drive(geometry):
    registry, ctx = continuation_registry(geometry)
    target = geometry.pred_drive / "notes.md"

    out = registry.execute("read_file", {"root": "task_drive", "path": str(target)})

    assert "PRED_DRIVE_BYTES" in out, out
    assert out.startswith("# task_drive:"), out
    assert ctx.last_read_view["opened_root"] == "task_drive"
    assert ctx.last_read_view["target"] == str(target.resolve())


def test_continuation_reads_its_predecessors_registered_artifact(geometry):
    """The live shape: the artifact path ``get_task_result(include_authority=True)``
    registered, read with the artifact root named and with no root at all."""
    registry, ctx = continuation_registry(geometry)
    target = geometry.pred_artifacts / "report.pdf.txt"

    named = registry.execute("read_file", {"root": "artifact_store", "path": str(target)})
    bare = registry.execute("read_file", {"path": str(target)})

    assert "PRED_ARTIFACT_BYTES" in named and named.startswith("# artifact_store:"), named
    assert "PRED_ARTIFACT_BYTES" in bare, bare
    assert ctx.last_read_view["opened_root"] == "artifact_store"
    assert lineage_read_base(ctx, "artifact_store", target) == geometry.pred_artifacts.resolve()
    assert lineage_read_base(ctx, "task_drive", geometry.pred_drive / "notes.md") == geometry.pred_drive.resolve()


# --- (b) a stranger's files and the hop before the predecessor stay refused -------

def test_a_strangers_drive_and_the_predecessors_own_predecessor_stay_refused(geometry):
    registry, ctx = continuation_registry(geometry)

    other = registry.execute("read_file", {"root": "task_drive", "path": str(geometry.other_drive / "notes.md")})
    other_artifact = registry.execute(
        "read_file", {"root": "artifact_store", "path": str(geometry.other_artifacts / "out.txt")})
    grand = registry.execute("read_file", {"root": "task_drive", "path": str(geometry.grand_drive / "notes.md")})

    assert "OTHER_BYTES" not in other and "outside selected root=task_drive" in other, other
    assert "OTHER_ARTIFACT_BYTES" not in other_artifact, other_artifact
    assert "outside selected root=artifact_store" in other_artifact, other_artifact
    # ONE hop: the envelope's ``previous_task_id`` (GRANDPRED) is never a lineage member.
    assert "GRANDPRED_BYTES" not in grand and "outside selected root=task_drive" in grand, grand
    assert GRANDPRED not in lineage_task_ids(ctx) and OTHER not in lineage_task_ids(ctx)


# --- (c) the predecessor grant is a READ grant ------------------------------------

def test_a_write_into_the_predecessors_drive_is_refused_while_the_own_drive_stays_writable(geometry):
    registry, ctx = continuation_registry(geometry)
    target = geometry.pred_drive / "notes.md"
    before = target.read_text(encoding="utf-8")

    refused = registry.execute("write_file", {"root": "task_drive", "path": str(target), "content": "x"})
    own = registry.execute("write_file", {"root": "task_drive", "path": "scratch.txt", "content": "mine"})

    assert refused.startswith("⚠️") and "outside selected root=task_drive" in refused, refused
    assert target.read_text(encoding="utf-8") == before
    assert not own.startswith("⚠️"), own
    assert (geometry.headless / "task_drives" / SUCCESSOR / "scratch.txt").read_text(encoding="utf-8") == "mine"
    own_base = geometry.headless / "task_drives" / SUCCESSOR
    assert _resolve_target_in_selected_base(
        ctx, root="task_drive", base_path=own_base, path=str(target), operation="read") == target.resolve()
    for operation in ("write", "edit"):
        with pytest.raises(ValueError):
            _resolve_target_in_selected_base(
                ctx, root="task_drive", base_path=own_base, path=str(target), operation=operation)


# --- (d) without predecessor authority nothing changes ----------------------------

def test_a_task_without_predecessor_authority_keeps_todays_lineage(geometry):
    registry, ctx = continuation_registry(geometry, predecessor=None)

    assert lineage_task_ids(ctx) == (SUCCESSOR,)
    out = registry.execute("read_file", {"root": "task_drive", "path": str(geometry.pred_drive / "notes.md")})
    assert "PRED_DRIVE_BYTES" not in out and "outside selected root=task_drive" in out, out

    ctx.task_metadata.update({"parent_task_id": PARENT, "root_task_id": ROOT})
    assert lineage_task_ids(ctx) == (SUCCESSOR, PARENT, ROOT)
    ctx.task_metadata["root_task_id"] = PARENT  # parent IS the root: no duplicate
    assert lineage_task_ids(ctx) == (SUCCESSOR, PARENT)
    ctx.task_metadata["parent_task_id"] = "../escape"  # malformed ids are dropped, not guessed
    ctx.task_metadata["root_task_id"] = ""
    assert lineage_task_ids(ctx) == (SUCCESSOR,)
    top = ToolContext(repo_dir=geometry.repo, drive_root=geometry.canonical, task_id=ROOT)
    assert lineage_task_ids(top) == (ROOT,)


def test_lineage_task_ids_names_the_predecessor_once_after_parent_and_root(geometry):
    _registry, ctx = continuation_registry(geometry)
    assert lineage_task_ids(ctx) == (SUCCESSOR, PRED)

    ctx.task_metadata.update({"parent_task_id": PARENT, "root_task_id": ROOT})
    assert lineage_task_ids(ctx) == (SUCCESSOR, PARENT, ROOT, PRED)

    ctx.task_metadata["root_task_id"] = PRED  # the predecessor IS the root: no duplicate
    assert lineage_task_ids(ctx) == (SUCCESSOR, PARENT, PRED)

    # A malformed or missing pointer id is dropped, never guessed from the envelope.
    ctx.task_metadata = {}
    ctx.task_contract["predecessor_authority"]["source"]["task_id"] = "../escape"
    assert lineage_task_ids(ctx) == (SUCCESSOR,)
    ctx.task_contract["predecessor_authority"]["source"] = "not-a-pointer"
    assert lineage_task_ids(ctx) == (SUCCESSOR,)
    ctx.task_contract["predecessor_authority"] = "not-an-envelope"
    assert lineage_task_ids(ctx) == (SUCCESSOR,)


def test_metadata_carries_the_envelope_only_when_no_contract_rides(geometry):
    """The contract is the carrier on a running ToolContext (the agent never copies
    the envelope into metadata); a context with no contract at all is read from
    metadata, and a present contract is never second-guessed by metadata."""
    _registry, ctx = continuation_registry(geometry, predecessor=None)
    ctx.task_metadata["predecessor_authority"] = predecessor_envelope(PRED)
    assert lineage_task_ids(ctx) == (SUCCESSOR, PRED)

    ctx.task_contract = {"predecessor_authority": predecessor_envelope(OTHER)}
    assert lineage_task_ids(ctx) == (SUCCESSOR, OTHER)
    ctx.task_contract = {"lineage": {"delegation_role": "root"}}
    assert lineage_task_ids(ctx) == (SUCCESSOR,)


# --- (e) a delegated child keeps parent/root even with an inherited envelope ------

def test_a_delegated_child_inherits_the_continuations_predecessor_read(geometry):
    """The brief's source preserves the existing one-hop read grant."""
    registry, ctx = child_registry(geometry)

    assert lineage_task_ids(ctx) == (CHILD, PARENT, ROOT, PRED)
    out = registry.execute("read_file", {"root": "task_drive", "path": str(geometry.pred_drive / "notes.md")})
    assert "PRED_DRIVE_BYTES" in out, out
    assert lineage_read_base(ctx, "task_drive", geometry.pred_drive / "notes.md") is not None

    ctx.task_contract.pop("predecessor_authority", None)
    ctx.task_metadata.pop("predecessor_authority", None)
    assert lineage_task_ids(ctx) == (CHILD, PARENT, ROOT)
