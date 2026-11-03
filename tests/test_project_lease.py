"""Multi-project lease + registry + chat-id policy (v6.32.0)."""

from __future__ import annotations
import pathlib

import pytest

from ouroboros.contracts.chat_id_policy import (
    PROJECT_CHAT_ID_MIN,
    WEB_UI_CHAT_ID,
    is_a2a_chat_id,
    is_project_chat_id,
    project_chat_id,
)
from ouroboros.project_lease import candidate_is_leasable, running_project_ids


def _task(project_id="", role="", tid="t1"):
    task = {"id": tid, "type": "task"}
    if project_id:
        task["project_id"] = project_id
    if role:
        task["delegation_role"] = role
    return task


def _meta(task):
    """Production RUNNING value shape: meta dict wrapping the task."""
    return {"task": task, "worker_id": 0, "last_heartbeat_at": 1.0}


def test_running_project_ids_counts_top_level_scoped_tasks_only():
    # Mix the PRODUCTION meta shape (workers.py RUNNING values) with bare task
    # dicts — running_project_ids must unwrap meta and still count both.
    running = [
        _meta(_task("alpha")),               # production shape
        _task("beta"),                       # bare task dict
        _meta(_task("", tid="plain")),       # unscoped: no lane
        _meta(_task("gamma", role="subagent")),  # swarm member: no lease of its own
        "garbage",
        None,
    ]
    assert running_project_ids(running) == {"alpha", "beta"}


def test_running_project_ids_unwraps_production_meta_shape():
    """Regression for the inert-lease bug: RUNNING.values() are meta dicts."""
    running = {"t1": _meta(_task("racer"))}.values()
    ids = running_project_ids(running)
    assert ids == {"racer"}
    assert candidate_is_leasable(_task("racer", tid="t2"), ids) is False


def test_candidate_is_leasable_matrix():
    leased = {"alpha"}
    # Unscoped tasks never serialize.
    assert candidate_is_leasable(_task(""), leased) is True
    # A second writer for a leased project waits.
    assert candidate_is_leasable(_task("alpha"), leased) is False
    # A different project proceeds in parallel.
    assert candidate_is_leasable(_task("beta"), leased) is True
    # The leased project's OWN subagents must not deadlock the swarm.
    assert candidate_is_leasable(_task("alpha", role="subagent"), leased) is True


def test_project_chat_id_policy():
    assert is_project_chat_id(WEB_UI_CHAT_ID) is False
    assert is_project_chat_id(-5) is False
    cid = project_chat_id("my-game")
    assert cid >= PROJECT_CHAT_ID_MIN
    assert is_project_chat_id(cid) is True
    assert is_a2a_chat_id(cid) is False
    # Deterministic and id-sensitive.
    assert project_chat_id("my-game") == cid
    assert project_chat_id("other") != cid
    # Empty scope falls back to the main chat.
    assert project_chat_id("") == WEB_UI_CHAT_ID


def test_registry_create_idempotent_and_summary(tmp_path):
    from ouroboros.projects_registry import (
        create_project,
        get_project,
        list_projects,
        projects_summary,
    )

    entry = create_project(tmp_path, "racer", name="Cyber Racer")
    assert entry["id"] == "racer"
    assert "status" not in entry  # statuses removed (v6.33.0)
    assert entry["chat_id"] == project_chat_id("racer")

    again = create_project(tmp_path, "racer", name="ignored on existing")
    assert again["name"] == "Cyber Racer"
    assert len(list_projects(tmp_path)) == 1

    rows = projects_summary(tmp_path)
    assert rows and rows[0]["id"] == "racer" and rows[0]["chat_id"] == entry["chat_id"]
    assert "status" not in rows[0]
    assert get_project(tmp_path, "missing") is None


def test_registry_reconcile_registers_existing_stores_never_prunes(tmp_path):
    from ouroboros.projects_registry import create_project, list_projects, reconcile_projects

    create_project(tmp_path, "kept")
    (tmp_path / "projects" / "legacy-store" / "knowledge").mkdir(parents=True)

    added = reconcile_projects(tmp_path)

    assert added == 1
    ids = {p["id"] for p in list_projects(tmp_path)}
    assert ids == {"kept", "legacy-store"}
    # Second run is a no-op (idempotent) and nothing is pruned.
    assert reconcile_projects(tmp_path) == 0
    assert {p["id"] for p in list_projects(tmp_path)} == ids


_GHOST_ID = "proj_deadbeef1234"
_TOMB_ID = "proj_deadtomb0001"


def _store(drive_root, project_id):
    """Materialize the on-disk shape project_facts creates for a project store."""
    (drive_root / "projects" / project_id / "knowledge").mkdir(parents=True)


def _write_bindings(drive_root, rows):
    """Write the durable task->project bindings file directly.

    bind_task_to_project would CREATE the registry row it binds to, which is
    exactly the state these tests must not have: the guard only matters for a
    store whose registry row is missing.
    """
    import json

    state = drive_root / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "project_task_bindings.json").write_text(
        json.dumps({"bindings": rows}), encoding="utf-8"
    )


def _read_bindings(drive_root):
    import json

    return json.loads(
        (drive_root / "state" / "project_task_bindings.json").read_text(encoding="utf-8")
    )["bindings"]


def _binding(project_id, task_id="t-1"):
    return {
        "task_id": task_id,
        "project_id": project_id,
        "project_chat_id": project_chat_id(project_id),
        "bound_at": "2026-08-26T00:00:00Z",
        "origin_absent": "post_hoc_unresolved",
    }


def test_reconcile_skips_workspace_derived_store_without_a_binding(tmp_path):
    """A proj_<hash> store minted by project_facts for a workspace path is not a
    project room: with no durable binding it must never become a sidebar row."""
    from ouroboros.projects_registry import list_projects, reconcile_projects

    _store(tmp_path, _GHOST_ID)

    assert reconcile_projects(tmp_path) == 0
    assert list_projects(tmp_path) == []
    # Idempotent: the ghost is skipped again on the next 300s tick, never pruned.
    assert reconcile_projects(tmp_path) == 0
    assert (tmp_path / "projects" / _GHOST_ID / "knowledge").is_dir()


def test_reconcile_registers_workspace_derived_store_with_a_binding(tmp_path):
    """A durable task binding is the proof of user ownership that admits the
    store — the registry-row-lost recovery case."""
    from ouroboros.projects_registry import list_projects, reconcile_projects

    _store(tmp_path, _GHOST_ID)
    _write_bindings(tmp_path, {"t-1": _binding(_GHOST_ID)})

    assert reconcile_projects(tmp_path) == 1
    assert {p["id"] for p in list_projects(tmp_path)} == {_GHOST_ID}


def test_reconcile_still_registers_named_store_beside_a_skipped_ghost(tmp_path):
    """The guard is prefix-scoped: an ordinary named store registers exactly as
    before, even while an unbound proj_ store in the same scan is skipped."""
    from ouroboros.projects_registry import list_projects, reconcile_projects

    _store(tmp_path, "legacy-store")
    _store(tmp_path, _GHOST_ID)

    assert reconcile_projects(tmp_path) == 1
    assert {p["id"] for p in list_projects(tmp_path)} == {"legacy-store"}


def test_reconcile_never_prunes_existing_rows_while_skipping_a_ghost(tmp_path):
    """Function contract: reconcile only ever ADDS. A pass that skips a ghost
    must leave every pre-existing row untouched."""
    from ouroboros.projects_registry import create_project, list_projects, reconcile_projects

    create_project(tmp_path, "kept", name="Kept Project")
    _store(tmp_path, _GHOST_ID)

    assert reconcile_projects(tmp_path) == 0
    rows = list_projects(tmp_path)
    assert [p["id"] for p in rows] == ["kept"]
    assert rows[0]["name"] == "Kept Project"


def test_reconcile_survives_malformed_binding_rows(tmp_path):
    """A malformed legacy binding row must fail closed for ITSELF only: the
    reconcile completes, named stores still register, unbound ghosts stay out."""
    from ouroboros.projects_registry import list_projects, reconcile_projects

    _store(tmp_path, "legacy-store")
    _store(tmp_path, _GHOST_ID)
    _store(tmp_path, "proj_00000000cafe")
    _write_bindings(tmp_path, {
        "t-str": "not-a-dict",
        "t-noid": {"task_id": "t-noid", "bound_at": "2026-08-26T00:00:00Z"},
        "t-null": {"task_id": "t-null", "project_id": None},
        # A LIST-valued project_id would sanitize through str() into a
        # valid-looking id ("['proj_00000000cafe']" -> "proj_00000000cafe"),
        # falsely vouching for that ghost — the type check must skip it.
        "t-list": {"task_id": "t-list", "project_id": ["proj_00000000cafe"]},
        "t-ok": _binding(_GHOST_ID, task_id="t-ok"),
    })

    # legacy-store (named) + the ONE ghost a valid row vouches for; the
    # list-vouched ghost stays out.
    assert reconcile_projects(tmp_path) == 2
    assert {p["id"] for p in list_projects(tmp_path)} == {"legacy-store", _GHOST_ID}


def test_reconcile_leaves_a_tombstoned_workspace_store_tombstoned(tmp_path):
    """Deletion PRESERVES bindings, so a tombstoned proj_ store still carries
    the very binding this guard accepts as ownership proof — the guard alone
    would re-admit it. The pre-existing `known` check runs FIRST and is what
    keeps it dead: a tombstoned id is never resurrected as active."""
    from ouroboros.projects_registry import (
        PROJECT_TOMBSTONED,
        begin_project_deletion,
        complete_project_deletion,
        create_project,
        list_projects,
        list_reserved_projects,
        reconcile_projects,
    )

    create_project(tmp_path, _TOMB_ID)
    _store(tmp_path, _TOMB_ID)
    _write_bindings(tmp_path, {"t-tomb": _binding(_TOMB_ID, task_id="t-tomb")})
    begin_project_deletion(tmp_path, _TOMB_ID)
    complete_project_deletion(tmp_path, _TOMB_ID)

    # The binding survived the deletion (nothing removed it).
    assert _read_bindings(tmp_path)["t-tomb"]["project_id"] == _TOMB_ID

    assert reconcile_projects(tmp_path) == 0
    assert list_projects(tmp_path) == []
    reserved = {p["id"]: p["lifecycle"] for p in list_reserved_projects(tmp_path)}
    assert reserved == {_TOMB_ID: PROJECT_TOMBSTONED}


@pytest.mark.parametrize("witnessed", [False, True])
def test_reconcile_recovers_a_marked_real_room_with_its_name(tmp_path, witnessed):
    """The recovery case the binding gate alone cannot cover: a REAL room named in
    Cyrillic hashes into the same proj_<hash> namespace as workspace stores (its id
    carries no binding when the owner created it by name). Once a reconcile tick has
    stamped the store's `.project.json`, losing the registry row is recoverable WITH
    the room's display name — better than the pre-guard behavior, which resurrected
    it under the machine name. Reconstruction creates a new execution incarnation;
    neither a store marker nor the commit witness certifies the old assignment."""
    from ouroboros import projects_registry as registry
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.projects_registry import (
        _registry_witness_path, create_project, list_projects, reconcile_projects,
    )

    pid = project_id_from_display_name("динозавры")
    assert pid.startswith("proj_")
    born = create_project(tmp_path, pid, name="динозавры")["created_at"]
    _store(tmp_path, pid)
    # One tick with the row alive stamps the marker into the existing store.
    assert reconcile_projects(tmp_path) == 0
    assert (tmp_path / "projects" / pid / ".project.json").is_file()

    # Registry catastrophe: the rows are lost wholesale.
    prior = registry.project_admission_view(tmp_path, pid, frozen=True)
    (tmp_path / "state" / "projects.json").unlink()
    if not witnessed:
        _registry_witness_path(tmp_path).unlink()  # a loss before any witness stamp

    assert reconcile_projects(tmp_path) == 1
    rows = list_projects(tmp_path)
    assert [p["id"] for p in rows] == [pid]
    assert rows[0]["name"] == "динозавры"
    assert rows[0]["origin"] == "owner"
    assert rows[0]["created_at"] == born, "recovery restores provenance, not a fresh mint"
    assert rows[0]["routing_incarnation"] != prior["project"]["routing_incarnation"]
    with pytest.raises(registry.ProjectAdmissionError, match="changed"):
        with registry.project_admission_guard(tmp_path, prior):
            pytest.fail("Reconstruction must not authorize the old prepared work")


def test_marker_maintenance_never_mkdirs_and_skips_reconcile_origin(tmp_path):
    """Two negative halves of the marker contract: a file-less room gets NO store
    directory minted for it (the marker waits until the store materializes), and a
    reconcile-originated row never mints recovery evidence for itself (else a
    pre-guard ghost row would immortalize its ghost through the marker)."""
    from ouroboros.projects_registry import create_project, reconcile_projects

    create_project(tmp_path, "fileless", name="File-less room")
    _store(tmp_path, "legacy-store")

    assert reconcile_projects(tmp_path) == 1  # legacy-store, origin=reconcile

    assert not (tmp_path / "projects" / "fileless").exists()
    assert not (tmp_path / "projects" / "legacy-store" / ".project.json").exists()


def test_marker_follows_a_rename_on_the_next_tick(tmp_path):
    """update_project may rename a room; the maintenance half refreshes the stale
    marker on the next tick so recovery restores the CURRENT name."""
    import json

    from ouroboros.projects_registry import create_project, reconcile_projects, update_project

    create_project(tmp_path, _GHOST_ID, name="Before")
    _store(tmp_path, _GHOST_ID)
    reconcile_projects(tmp_path)
    update_project(tmp_path, _GHOST_ID, name="After")
    reconcile_projects(tmp_path)

    marker = json.loads(
        (tmp_path / "projects" / _GHOST_ID / ".project.json").read_text(encoding="utf-8")
    )
    assert marker["name"] == "After"


def test_deleted_room_stays_dead_even_after_registry_file_loss(tmp_path):
    """Tombstoning removes the registry-authored `.project.json` (rows AND tombstones
    live in the one projects.json, so a stale marker would resurrect the deleted room
    as active in exactly the registry-loss scenario recovery exists for). The store
    itself — the owner's memory — stays preserved."""
    from ouroboros.projects_registry import (
        begin_project_deletion,
        complete_project_deletion,
        create_project,
        list_projects,
        reconcile_projects,
    )

    create_project(tmp_path, _TOMB_ID, name="Dead Room")
    _store(tmp_path, _TOMB_ID)
    reconcile_projects(tmp_path)  # stamps the marker while the row is ACTIVE
    assert (tmp_path / "projects" / _TOMB_ID / ".project.json").is_file()

    begin_project_deletion(tmp_path, _TOMB_ID)
    complete_project_deletion(tmp_path, _TOMB_ID)
    assert not (tmp_path / "projects" / _TOMB_ID / ".project.json").exists()
    assert (tmp_path / "projects" / _TOMB_ID / "knowledge").is_dir()  # memory preserved

    (tmp_path / "state" / "projects.json").unlink()  # registry catastrophe
    assert reconcile_projects(tmp_path) == 0
    assert list_projects(tmp_path) == []


def test_one_unwritable_store_does_not_abort_the_reconcile(tmp_path):
    """Per-row isolation: a store the marker cannot be written into keeps its own
    marker pending for the next tick, while registrations still run."""
    import os

    from ouroboros.projects_registry import create_project, list_projects, reconcile_projects

    create_project(tmp_path, "sealed", name="Sealed")
    _store(tmp_path, "sealed")
    _store(tmp_path, "legacy-store")
    os.chmod(tmp_path / "projects" / "sealed", 0o555)
    try:
        assert reconcile_projects(tmp_path) == 1  # legacy-store still registers
        assert {p["id"] for p in list_projects(tmp_path)} == {"sealed", "legacy-store"}
    finally:
        os.chmod(tmp_path / "projects" / "sealed", 0o755)


def test_journal_and_workpad_roundtrip(tmp_path, monkeypatch):
    import types

    # Scope the project store to tmp_path WITHOUT importlib.reload(config): a
    # reload permanently rebinds ouroboros.config.DATA_DIR for the rest of the
    # pytest process (monkeypatch restores only the env var, not the reloaded
    # module), polluting later tests. project_facts reads config.DATA_DIR at call
    # time, so monkeypatch.setattr (auto-restored) is sufficient and isolated.
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.tools import project_journal as pj

    ctx = types.SimpleNamespace(project_id="racer", task_id="t-9", drive_root=tmp_path)
    tools = {t.name: t for t in pj.get_tools()}

    out = tools["journal_write"].handler(ctx, kind="start", text="Bootstrapping the racer")
    assert out.startswith("OK:")
    out = tools["journal_write"].handler(ctx, kind="bogus", text="x")
    assert "TOOL_ARG_ERROR" in out
    listing = tools["journal_read"].handler(ctx)
    assert "Bootstrapping the racer" in listing and "START" in listing

    assert tools["workpad_write"].handler(ctx, content="## plan\n- wheels").startswith("OK:")
    assert "wheels" in tools["workpad_read"].handler(ctx)

    digest = pj.journal_tail_digest("racer")
    assert "Bootstrapping the racer" in digest

    # Unscoped ctx without explicit id refuses honestly.
    bare = types.SimpleNamespace(project_id="", task_id="t", drive_root=tmp_path)
    assert "no project scope" in tools["journal_write"].handler(bare, kind="note", text="x")


@pytest.mark.parametrize("failure", ["torn_temp_write", "failed_replace"])
def test_a_failed_workpad_replacement_keeps_the_whole_previous_page(tmp_path, monkeypatch, failure):
    """#1854: a replacement that dies mid-write or at the swap leaves the complete
    previous page and says so; it never exposes an empty or torn page or a stray temp."""
    import errno
    import os
    import types

    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    from ouroboros.project_facts import project_workpad_path
    from ouroboros.tools import project_journal as pj

    ctx = types.SimpleNamespace(project_id="racer", task_id="t-9", drive_root=tmp_path)
    assert pj._workpad_write(ctx, "## old plan\n- wheels\n").startswith("OK:")

    real_write_bytes = pathlib.Path.write_bytes

    def torn_write(self, data):
        real_write_bytes(self, data[:5])  # the first bytes land, then the disk fills
        raise OSError(errno.ENOSPC, "No space left on device")

    def failed_replace(src, dst):
        raise OSError(errno.EIO, "I/O error")

    with monkeypatch.context() as patched:
        if failure == "torn_temp_write":
            patched.setattr(pathlib.Path, "write_bytes", torn_write)
        else:
            patched.setattr(os, "replace", failed_replace)
        failed = pj._workpad_write(ctx, "## new plan\n- engine\n")
    assert failed.startswith("⚠️ TOOL_ERROR (workpad_write)")
    assert pj._workpad_read(ctx) == "## old plan\n- wheels\n"
    assert [p.name for p in project_workpad_path("racer").parent.iterdir()] == ["workpad.md"]

    assert pj._workpad_write(ctx, "## new plan\n- engine\n").startswith("OK:")
    assert pj._workpad_read(ctx) == "## new plan\n- engine\n"
    assert [p.name for p in project_workpad_path("racer").parent.iterdir()] == ["workpad.md"]


def test_stale_tombstone_marker_is_swept_by_the_next_reconcile_tick(tmp_path, monkeypatch):
    """Convergence: if the marker unlink at tombstone time failed (or predates it),
    the maintenance pass removes it on a later tick, so registry loss after that can
    never resurrect the deleted room through the marker."""
    from ouroboros.projects_registry import (
        begin_project_deletion,
        complete_project_deletion,
        create_project,
        list_projects,
        reconcile_projects,
    )

    create_project(tmp_path, _TOMB_ID, name="Sticky Marker")
    _store(tmp_path, _TOMB_ID)
    reconcile_projects(tmp_path)
    marker = tmp_path / "projects" / _TOMB_ID / ".project.json"
    assert marker.is_file()

    real_unlink = pathlib.Path.unlink
    def _failing_unlink(self, *a, **k):
        if self.name == ".project.json":
            raise OSError("transient")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(pathlib.Path, "unlink", _failing_unlink)
    begin_project_deletion(tmp_path, _TOMB_ID)
    complete_project_deletion(tmp_path, _TOMB_ID)
    assert marker.is_file(), "unlink failed; the stale marker survived deletion"
    monkeypatch.setattr(pathlib.Path, "unlink", real_unlink)

    reconcile_projects(tmp_path)  # the sweep retries the removal
    assert not marker.exists()
    (tmp_path / "state" / "projects.json").unlink()
    assert reconcile_projects(tmp_path) == 0
    assert list_projects(tmp_path) == []
