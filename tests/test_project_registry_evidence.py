"""The Project registry's first use and recovery on synthetic roots."""
import errno
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from ouroboros import dialogue_history_evidence as evidence
from ouroboros import dialogue_store as dialogue, projects_registry as projects, store
from ouroboros.contracts.chat_id_policy import project_chat_id


def require_writable_lock_descriptors(monkeypatch, refusal):
    import fcntl
    from ouroboros import platform_layer as platform

    acquire = platform.file_lock_exclusive_nb

    def writable_lock(fd):
        if (fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE) == os.O_RDONLY:
            raise OSError(refusal, "Volume requires a writable descriptor for exclusive locking")
        acquire(fd)

    monkeypatch.setattr(platform, "file_lock_exclusive_nb", writable_lock)


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory flock")
@pytest.mark.parametrize("previously_certified", [False, True])
@pytest.mark.parametrize("refusal", [errno.EBADF, errno.EACCES, errno.EISDIR, errno.EINVAL, errno.ENOTSUP])
def test_writable_file_lock_still_saves_when_read_only_descriptors_refuse_exclusion(
        tmp_path, monkeypatch, previously_certified, refusal):
    from ouroboros import project_registry_recovery as recovery

    path, witness = recovery.registry_path(tmp_path), recovery.witness_path(tmp_path)
    if previously_certified:
        projects.create_project(tmp_path, "room", name="Original")
        assert recovery._released_proof(path)
    with monkeypatch.context() as volume:
        require_writable_lock_descriptors(volume, refusal)
        if previously_certified:
            projects.update_project(tmp_path, "room", name="Current")
        else:
            projects.create_project(tmp_path, "room", name="Current")
        assert projects.get_project(tmp_path, "room")["name"] == "Current"
        assert witness.is_file() and recovery._released_proof(path) is None

    # Even when the volume qualifies again, an earlier copy cannot authorize restoration.
    path.rename(tmp_path / "saved-primary.json")
    projects.reconcile_projects(tmp_path)
    assert not path.exists() and recovery.recovery_notice(tmp_path) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory flock")
def test_handoff_leaves_a_free_name_when_the_old_writer_cannot_reclaim_a_certificate(tmp_path, monkeypatch):
    from ouroboros import platform_layer as platform, project_registry_recovery as recovery

    projects.create_project(tmp_path, "room")
    path = recovery.registry_path(tmp_path)
    lock = path.with_suffix(".json.lock")
    saved = path.read_bytes()
    assert recovery._released_proof(path)
    require_writable_lock_descriptors(monkeypatch, errno.EBADF)
    recovery.hand_off(tmp_path)
    assert not lock.exists()
    fd = platform.acquire_exclusive_file_lock(lock, timeout_sec=0)
    assert fd is not None
    platform.release_exclusive_file_lock(lock, fd)
    assert path.read_bytes() == recovery.witness_path(tmp_path).read_bytes() == saved


@pytest.mark.parametrize("kind", ["project_started", "project_handoff", "project_completion_summary", "project_closure"])
def test_saved_project_notice_preserves_folderless_history(tmp_path, kind):
    chat = project_chat_id("folderless") if kind == "project_closure" else 1
    receipt = dialogue.append(tmp_path, "chat", {
        "type": kind, "chat_id": chat, "role": "system", "direction": "out", "text": "Saved notice",
        **({} if kind == "project_closure" else {"project_id": "folderless"}),
    })
    assert projects.reconcile_projects(tmp_path) == 0
    with pytest.raises(FileNotFoundError):
        projects.reserved_project_chat_ids(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError):
        projects.create_project(tmp_path, "folderless")
    assert dialogue.row_at(tmp_path, "chat", receipt.seq).row["text"] == "Saved notice"
    assert not projects._registry_path(tmp_path).exists()


def test_main_and_external_dialogue_allow_first_project(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    for chat, source in ((1, "web"), (project_chat_id("remote"), "skill:telegram")):
        dialogue.append(tmp_path, "chat", {
            "direction": "in", "chat_id": chat, "source": source, "user_id": 1,
            "text": "We discussed a project_started row",
        })
    database = dialogue.path(tmp_path)
    names = set(database.parent.iterdir())
    snapshots = []
    connect = store.counted_connect

    def open_snapshot(target, path, **kwargs):
        snapshots.append(target)
        return connect(target, path, **kwargs)

    with monkeypatch.context() as inspection:
        inspection.setattr(store, "counted_connect", open_snapshot)
        inspection.setattr(dialogue, "ensure", lambda *_: pytest.fail("lookup requested cutover"))
        inspection.setattr(store, "name_lock", lambda *a, **kw: pytest.fail("lookup requested a writer lock"))
        assert projects.reserved_project_chat_ids(tmp_path, strict=True) == set()
    assert len(snapshots) == 1 and "mode=ro" in snapshots[0]
    assert set(database.parent.iterdir()) == names
    room = projects.create_project(tmp_path, "first")
    assert projects.reserved_project_chat_ids(tmp_path, strict=True) == {room["chat_id"]}


@pytest.mark.parametrize("case", ["bound_answer", "bound_press", "primary_owner_chat", "different_binding",
                                  "unconfirmed_binding", "ordinary_web"])
def test_host_delivered_web_decision_is_not_project_evidence_only_in_a_bound_owner_chat(tmp_path, case):
    """A late answer or a press on an ended author's card is accepted, typed, in the card's own chat. In a chat the
    owner bindings prove to be the owner's (the external slot or the primary chat), a previously empty profile keeps
    its normal Main and first Project; untyped web ingress there, another chat or an unconfirmed binding does not."""
    import asyncio
    import json
    from types import SimpleNamespace
    from ouroboros.gateway.history import make_chat_history_endpoint
    from supervisor.message_ingress import accept_chat
    from supervisor.state_initialization import complete, mark_pending

    external_chat = 70_001
    identity = mark_pending(tmp_path, origin="first_boot")
    assert complete(tmp_path, identity, adopted=False)
    state = {"initialization_id": identity, "owner_external_id": 17,
             "owner_external_chat_id": external_chat if case != "different_binding" else 70_002}
    if case == "primary_owner_chat":
        state.update(owner_external_chat_id=None, owner_chat_id=external_chat)
    if case == "unconfirmed_binding":
        state["_recovery"] = {"unconfirmed": ["owner_external_chat_id"]}
    (tmp_path / "state/state.json").write_text(json.dumps(state), encoding="utf-8")
    dialogue.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "source": "web", "text": "Main text"})
    late = {"task_id": "earlier-work", "quiz_id": "q", "answer_snapshot": {
        "task_id": "earlier-work", "quiz": {"quiz_id": "q", "state": "answered", "answered_after_terminal": True,
                                             "publication_chat_id": external_chat, "publication_evidence": "pre_exposure"}}}
    press = {"task_id": "earlier-work", "card_id": "card_0123456789abcdef", "request_id": "press-1"}
    typed = {"ordinary_web": {}, "bound_press": {"card_press": press}}.get(case, {"late_answer": late})
    receipt = accept_chat(external_chat, 1, "An answer", drive_root=tmp_path, source="web", client_message_id="synthetic",
                          message_meta=typed)
    assert receipt.row.get("late_answer") == typed.get("late_answer")
    assert receipt.row.get("card_press") == typed.get("card_press")
    if case in {"bound_answer", "bound_press", "primary_owner_chat"}:
        response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={})))
        assert response.status_code == 200
        assert "Main text" in [row["text"] for row in json.loads(response.body)["messages"]]
        assert projects.create_project(tmp_path, "first")["created"]
    else:
        with pytest.raises(FileNotFoundError):
            projects.create_project(tmp_path, "first")
        assert not projects._registry_path(tmp_path).exists()
    assert dialogue.row_at(tmp_path, "chat", receipt.seq).row == receipt.row


@pytest.mark.parametrize("pending_reset", [False, True])
def test_non_main_web_ingress_keeps_its_room_separate(tmp_path, pending_reset):
    from ouroboros.gateway.history import RegistryUnavailable, _project_history_context
    from ouroboros.store import StoreError
    from supervisor.state_initialization import mark_pending

    chat = project_chat_id("former")
    saved = dialogue.append(tmp_path, "chat", {
        "direction": "in", "chat_id": chat, "user_id": 1, "source": "web", "text": "In this room",
    })
    if pending_reset:
        mark_pending(tmp_path, origin="owner_reset")
        (tmp_path / "projects/former").mkdir(parents=True)
    projects.reconcile_projects(tmp_path)
    with pytest.raises(StoreError) as refused:
        _project_history_context(tmp_path, 1)
    assert isinstance(refused.value, RegistryUnavailable) is not pending_reset
    with pytest.raises(FileNotFoundError):
        projects.create_project(tmp_path, "former")
    assert not projects._registry_path(tmp_path).exists()
    assert dialogue.row_at(tmp_path, "chat", saved.seq).row["chat_id"] == chat


@pytest.mark.parametrize("protocol", ["name", "closed_wal"])
@pytest.mark.parametrize("entry", ["startup", "create", "prepared_create"])
def test_first_project_uses_existing_main_history_and_preserves_workspace_shelf(tmp_path, monkeypatch, protocol, entry):
    from ouroboros.server_boot import reconcile_projects

    basis = (projects.project_admission_view(tmp_path, "first", allow_unregistered=True, frozen=True)
             if entry == "prepared_create" else None)
    if protocol == "name":
        monkeypatch.setattr(dialogue, "probed_tier", lambda *_: store.TIER_NAME)
    else:
        monkeypatch.setattr(store, "_journal_mode", lambda *_: "WAL")
    saved = dialogue.append(tmp_path, "chat", {"chat_id": 1, "direction": "in", "text": "Saved"})
    database = dialogue.path(tmp_path)
    store.close_connections(database)
    if protocol == "closed_wal":
        assert store.wal_file(database)
        assert not any(database.with_name(database.name + ext).exists() for ext in ("-wal", "-shm"))
    else:
        assert dialogue.header_tier(tmp_path) == store.TIER_NAME
    shelf = tmp_path / "projects/proj_0123456789ab/knowledge"
    shelf.mkdir(parents=True)
    fact = shelf / "note.md"
    fact.write_bytes(b"workspace fact")
    monkeypatch.setattr(dialogue, "ensure", lambda *_: pytest.fail("bootstrap requested cutover"))
    assert projects.reserved_project_chat_ids(tmp_path, strict=True) == set()
    assert not projects._registry_path(tmp_path).exists()
    assert not projects._registry_witness_path(tmp_path).exists()

    if entry == "startup":
        reconcile_projects(tmp_path)
        assert projects._registry_path(tmp_path).is_file()
        assert projects.reserved_project_chat_ids(tmp_path, strict=True) == set()
    first = projects.create_project(tmp_path, "first", admission_basis=basis)
    assert first["created"] and projects.reserved_project_chat_ids(tmp_path, strict=True) == {first["chat_id"]}
    assert dialogue.row_at(tmp_path, "chat", saved.seq).row["text"] == "Saved"
    assert fact.read_bytes() == b"workspace fact"


@pytest.mark.parametrize("phase", ["new", "pending"])
def test_positive_initialization_with_existing_name_tier_store(tmp_path, phase):
    from supervisor import state_initialization

    dialogue.create(tmp_path, tier=store.TIER_NAME)
    prior = state_initialization.mark_pending(tmp_path, origin="first_boot") if phase == "pending" else None
    decision = state_initialization.initialization_decision(tmp_path)
    assert decision["create"] and (prior is None or decision["initialization_id"] == prior)
    assert projects._registry_path(tmp_path).is_file()
    assert projects.reserved_project_chat_ids(tmp_path, strict=True) == set()
    assert projects.create_project(tmp_path, "first")["created"]


@pytest.mark.parametrize("survivor", [None, "chat", "progress"])
def test_providerless_boot_holds_a_legacy_pending_reset(tmp_path, survivor):
    from ouroboros.server_boot import reconcile_projects
    from supervisor import state_initialization as initialization

    shelf = tmp_path / "projects/retained/knowledge"
    shelf.mkdir(parents=True)
    if survivor:
        dialogue.append(tmp_path, survivor, {"chat_id": 1, "text": "Earlier profile"})
    identity = initialization.mark_pending(tmp_path, origin="owner_reset")
    reconcile_projects(tmp_path)
    assert not projects._registry_path(tmp_path).exists()
    assert initialization.initialization_decision(tmp_path)["reason"] == "reset_cleanup_unconfirmed"
    assert not initialization.complete(tmp_path, identity, adopted=True)
    assert initialization.read_witness(tmp_path)[1]["phase"] == "pending"
    with pytest.raises(FileNotFoundError):
        projects.create_project(tmp_path, "new")
    assert shelf.is_dir()


@pytest.mark.parametrize("retained_shelf", [False, True])
@pytest.mark.parametrize("receipt", ["recorded", "unreadable", "absent"])
def test_completed_reset_recreates_a_missing_registry_only_without_its_receipt_on_first_use_evidence(
        tmp_path, retained_shelf, receipt):
    """A Reset that recorded its cleanup receipt published its own registry: losing it is never another first use,
    and a receipt field present but unreadable is unknown. Only an older Reset, which wrote no such field and no
    registry, gets a first Project, exactly when the ordinary first-use evidence allows one (no retained shelf)."""
    import json
    from supervisor.state_initialization import mark_pending, read_witness, witness_path

    if retained_shelf:
        (tmp_path / "projects/retained").mkdir(parents=True)
    mark_pending(tmp_path, origin="owner_reset")
    _, previous = read_witness(tmp_path)
    completed = {**previous, "phase": "complete"}
    if receipt != "absent":
        identity = previous["initialization_id"] if receipt == "recorded" else "another-identity"
        completed["reset_receipt"] = {"initialization_id": identity, "deleted": [], "retained_logs": []}
    witness_path(tmp_path).write_text(json.dumps(completed), encoding="utf-8")
    assert read_witness(tmp_path)[0] == "ok"
    projects.reconcile_projects(tmp_path)
    if receipt != "absent" or retained_shelf:
        with pytest.raises(FileNotFoundError):
            projects.create_project(tmp_path, "new")
        assert not projects._registry_path(tmp_path).exists()
    else:
        assert projects._registry_path(tmp_path).is_file()
        assert projects.create_project(tmp_path, "new")["created"]
    assert (tmp_path / "projects/retained").is_dir() is retained_shelf


def test_restore_receipt_rules_out_first_use(tmp_path):
    primary = projects._registry_path(tmp_path)
    primary.parent.mkdir(parents=True)
    receipt = primary.with_name("projects.json.restore")
    receipt.write_bytes(b"previously restored")
    projects.reconcile_projects(tmp_path)
    with pytest.raises(FileNotFoundError):
        projects.create_project(tmp_path, "new")
    assert not primary.exists() and receipt.read_bytes() == b"previously restored"


@pytest.mark.parametrize("witness", ["invalid", "unreadable"])
def test_an_unconfirmed_initialization_witness_neither_restores_nor_admits(tmp_path, witness):
    """A valid registry certificate proves the revision, not which profile owns it: while the initialization
    witness cannot be read or parsed, strict reads refuse, maintenance restores nothing and no new Project is
    created. A repaired witness resumes ordinary operation, the certified restore included."""
    import json
    from ouroboros import project_registry_recovery as recovery
    from ouroboros.room_archive import registry_reservations
    from ouroboros.store import StoreError
    from supervisor.state_initialization import witness_path

    room = projects.create_project(tmp_path, "kept")
    path = projects._registry_path(tmp_path)
    saved = path.read_bytes()
    target = witness_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if witness == "invalid":
        target.write_text("{not json", encoding="utf-8")
    else:
        target.mkdir()  # a directory where the witness belongs: its read fails
    assert recovery.history_generation(tmp_path) == "unknown"
    # A readable primary authorizes no room while the generation is unconfirmed.
    with pytest.raises(FileNotFoundError):
        projects.reserved_project_chat_ids(tmp_path, strict=True)
    assert projects.readable_rooms(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        projects.create_project(tmp_path, "new")
    path.unlink()
    projects.reconcile_projects(tmp_path)
    assert not path.exists() and recovery.recovery_notice(tmp_path) is None
    with pytest.raises(StoreError):
        registry_reservations(tmp_path)
    if witness == "unreadable":
        target.rmdir()
    target.write_text(json.dumps({"initialization_id": "repaired", "phase": "complete",
                                  "origin": "first_boot"}), encoding="utf-8")
    projects.reconcile_projects(tmp_path)
    assert path.read_bytes() == saved
    assert projects.reserved_project_chat_ids(tmp_path, strict=True) == {room["chat_id"]}


def test_a_recorded_project_appearance_keeps_a_room_unknown(tmp_path, monkeypatch):
    from ouroboros.server_boot import reconcile_projects

    monkeypatch.setattr(dialogue, "probed_tier", lambda *_: store.TIER_NAME)
    receipt = dialogue.append(tmp_path, "chat", {"chat_id": 1, "type": "main_notice", "text": "Saved in both rooms"},
                              appearances=[f"room:{project_chat_id('former')}"])
    reconcile_projects(tmp_path)
    with pytest.raises(FileNotFoundError):
        projects.reserved_project_chat_ids(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError):
        projects.create_project(tmp_path, "former")
    assert not projects._registry_path(tmp_path).exists()
    assert dialogue.row_at(tmp_path, "chat", receipt.seq).row["text"] == "Saved in both rooms"


@pytest.mark.parametrize("existing", [False, True])
def test_first_project_defers_to_pending_chat_import_without_cutover(tmp_path, existing):
    import asyncio
    import json
    from types import SimpleNamespace
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.server_boot import reconcile_projects
    from tests._dialogue import import_all, write_segment

    if existing:
        dialogue.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "text": "First row"})
    chain = write_segment(tmp_path / "logs/chat.jsonl", [{
        "ts": "2026-01-01T00:00:00Z", "direction": "in", "chat_id": 1, "user_id": 1,
        "source": "web", "text": "Before boot",
    }])
    saved = chain.read_bytes()
    with pytest.raises(FileNotFoundError):
        projects.reserved_project_chat_ids(tmp_path, strict=True)
    reconcile_projects(tmp_path)
    with pytest.raises(FileNotFoundError):
        projects.create_project(tmp_path, "held")
    assert chain.read_bytes() == saved and dialogue.path(tmp_path).exists() == existing
    import_all(tmp_path)
    projects.reconcile_projects(tmp_path)  # maintenance after the import, without a second boot or a write from Main
    assert projects._registry_path(tmp_path).is_file()
    assert projects.reserved_project_chat_ids(tmp_path, strict=True) == set()
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={})))
    assert response.status_code == 200
    assert "Before boot" in [row["text"] for row in json.loads(response.body)["messages"]]
    assert projects.create_project(tmp_path, "first")["created"] is True


def test_an_established_legacy_root_with_a_torn_line_keeps_its_ordinary_first_use(tmp_path):
    """A line an older body left torn imports as a permanently skipped interval: never a row, so never
    Project-history evidence. An established root with one recovers its ordinary first use through the real
    maintenance consumers -- registry published, Main readable, first Project creatable -- instead of staying
    locked out of its own history."""
    import asyncio
    import json
    from types import SimpleNamespace
    from ouroboros import maintenance
    from ouroboros.gateway.history import make_chat_history_endpoint
    from tests._dialogue import write_segment

    write_segment(tmp_path / "logs/chat.jsonl", [
        {"ts": "2026-01-01T00:00:00Z", "direction": "in", "chat_id": 1, "user_id": 1,
         "source": "web", "text": "Ordinary Main"},
        '{"ts": "2026-01-01T00:00:01Z", "dire',  # torn: permanently skipped, never a row
    ])
    due, run = maintenance.JOBS["dialogue_import"]
    assert due(tmp_path, {}, {}, False), "a legacy chain without a store is the runner's own cutover case"
    assert run(tmp_path, lambda: False)["state"] == "done"
    with dialogue.transaction(tmp_path, write=False) as conn:
        assert conn.execute("SELECT count(*) AS n FROM skipped WHERE stream='chat'").fetchone()["n"] == 1
    projects.reconcile_projects(tmp_path)  # the maintenance tick that publishes the first registry
    assert projects._registry_path(tmp_path).is_file()
    assert projects.reserved_project_chat_ids(tmp_path, strict=True) == set()
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={})))
    assert response.status_code == 200
    assert "Ordinary Main" in [row["text"] for row in json.loads(response.body)["messages"]]
    assert projects.create_project(tmp_path, "first")["created"]


def test_a_providerless_legacy_profile_reaches_a_readable_main_through_maintenance(tmp_path):
    """No settings, no provider, no store, no registry -- only the chain files an older body left. The real
    maintenance consumers (the dispatcher's due check and its runner; no test-initiated import) reach a
    readable Main and an ordinary first Project: the runner opens with the cutover, so the import never
    depends on the provider-gated boot step."""
    import asyncio
    import json
    from types import SimpleNamespace
    from ouroboros import maintenance
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.room_archive import archive_needed
    from tests._dialogue import seed

    seed(tmp_path, archives={"chat_20260101T000000.jsonl": [
             {"ts": "2026-01-01T00:00:00Z", "direction": "in", "chat_id": 1, "user_id": 1,
              "source": "web", "text": "Archived before boot"}]},
         live={"chat": [{"ts": "2026-01-02T00:00:00Z", "direction": "in", "chat_id": 1, "user_id": 1,
                         "source": "web", "text": "Live before boot"}]})
    assert not (tmp_path / "settings.json").exists() and not dialogue.path(tmp_path).exists()
    due, run = maintenance.JOBS["dialogue_import"]
    assert due(tmp_path, {}, {}, False)
    assert run(tmp_path, lambda: False)["state"] == "done"
    assert not due(tmp_path, {}, {}, False), "an imported store owes the job nothing"
    projects.reconcile_projects(tmp_path)
    assert projects._registry_path(tmp_path).is_file()
    assert not archive_needed(tmp_path)
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={})))
    assert response.status_code == 200
    texts = [row["text"] for row in json.loads(response.body)["messages"]]
    assert "Archived before boot" in texts and "Live before boot" in texts
    assert projects.create_project(tmp_path, "first")["created"]
    bare = tmp_path / "bare-root"
    bare.mkdir()
    assert not due(bare, {}, {}, False), "a root without chain files has nothing to cut over"


@pytest.mark.parametrize("tails", [True, False])
def test_imported_chat_segments_stay_complete_and_an_unplanned_one_is_pending_in_both_readers(
        tmp_path, monkeypatch, tails):
    """While the importer's recorded archive stamp holds, first use lists nothing. Another stream's rotation moves the
    stamp: each chat segment the import ledger records as imported, at its recorded size, stays complete, with or
    without the plan's optional tails, and no segment's bytes are read. One whose size changed, or one the ledger
    never recorded, is an input still pending: Main's history and the archive both answer 503 ``migrating``."""
    import asyncio
    import builtins
    import io
    import json
    import pathlib
    from types import SimpleNamespace
    from ouroboros.gateway.chat_archive import make_chat_archive_endpoint
    from ouroboros.gateway.history import make_chat_history_endpoint
    from tests._dialogue import import_all, seed, write_segment

    row = {"ts": "2026-01-01T00:00:00Z", "direction": "in", "chat_id": 1, "user_id": 1, "source": "web"}
    seed(tmp_path, archives={"chat_20260101T000000.jsonl": [{**row, "text": "Archived Main"}]})
    import_all(tmp_path)
    if not tails:  # an older plan recorded only the stamp
        with dialogue.transaction(tmp_path, write=True) as conn:
            conn.execute("DELETE FROM meta WHERE key='archive_tails'")
    archive = tmp_path / "archive"
    imported = archive / "chat_20260101T000000.jsonl"
    saved = imported.read_bytes()
    listing = os.scandir

    def unlisted(path=".", *args):
        if pathlib.Path(path) == archive:
            pytest.fail("listed an archive whose recorded stamp still holds")
        return listing(path, *args)

    with monkeypatch.context() as warm:
        warm.setattr(os, "scandir", unlisted)
        assert projects.reserved_project_chat_ids(tmp_path, strict=True) == set()

    def reserved_without_segment_bytes():
        with monkeypatch.context() as unread:
            for module in (builtins, io, os):
                def opener(file, *args, _open=module.open, **kwargs):
                    if not isinstance(file, int) and pathlib.Path(os.fsdecode(file)).parent == archive:
                        pytest.fail(f"read a segment's bytes: {file}")
                    return _open(file, *args, **kwargs)
                unread.setattr(module, "open", opener)
            return projects.reserved_project_chat_ids(tmp_path, strict=True)

    def rotate(name, text):
        moved = archive.stat().st_mtime_ns + 1_000_000_000  # a coarse clock cannot hide the directory's change
        write_segment(archive / name, [{**row, "text": text}])
        os.utime(archive, ns=(moved, moved))

    rotate("progress_20260102T000000.jsonl", "Unrelated progress")
    assert reserved_without_segment_bytes() == set()
    assert reserved_without_segment_bytes() == set(), "a re-read answers the same"

    write_segment(imported, [{**row, "text": "Appended after its import"}])
    with pytest.raises(evidence.ProjectHistoryImportPending):
        reserved_without_segment_bytes()
    imported.write_bytes(saved)
    assert reserved_without_segment_bytes() == set()

    rotate("chat_20260103T000000.jsonl", "Rotated by another body")
    with pytest.raises(evidence.ProjectHistoryImportPending):
        reserved_without_segment_bytes()
    history = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={})))
    assert history.status_code == 503 and json.loads(history.body)["reason_code"] == "migrating"
    held = asyncio.run(make_chat_archive_endpoint(tmp_path)(SimpleNamespace(query_params={})))
    assert (held.status_code, json.loads(held.body)) == (503, {"error": "migrating", "reason_code": "migrating"})
    assert not projects._registry_path(tmp_path).exists()


def test_maintenance_retries_a_busy_first_use_then_stops_inspecting_dialogue(tmp_path, monkeypatch):
    from ouroboros.server_boot import reconcile_projects

    dialogue.create(tmp_path, tier=store.TIER_NAME)
    with monkeypatch.context() as busy:
        def unavailable(*args, **kwargs):
            raise store.StoreBusy("another reader")
        busy.setattr(dialogue, "transaction", unavailable)
        reconcile_projects(tmp_path)
        assert not projects._registry_path(tmp_path).exists()
    projects.reconcile_projects(tmp_path)
    assert projects._registry_path(tmp_path).is_file()
    monkeypatch.setattr(evidence, "could_lack_project_history", lambda *a, **kw: pytest.fail("inspected history again"))
    projects.reconcile_projects(tmp_path)
    assert projects.reserved_project_chat_ids(tmp_path, strict=True) == set()


@pytest.mark.parametrize("shelf_id", ["proj_0123456789ab", "proj_team-local"])
def test_unbound_workspace_shelf_stays_unregistered_without_a_marker(tmp_path, shelf_id):
    shelf = tmp_path / "projects" / shelf_id / "knowledge"
    shelf.mkdir(parents=True)
    fact = shelf / "fact.md"
    fact.write_bytes(b"workspace fact")
    assert projects.reserved_project_chat_ids(tmp_path, strict=True) == set()
    room = projects.create_project(tmp_path, "first")
    assert projects.reconcile_projects(tmp_path) == 0
    assert projects.reserved_project_chat_ids(tmp_path, strict=True) == {room["chat_id"]}
    assert fact.read_bytes() == b"workspace fact"


@pytest.mark.parametrize("shelf_id", ["proj_0123456789ab", "proj_team-local"])
def test_workspace_shaped_shelf_with_project_marker_keeps_old_reservation_unknown(tmp_path, shelf_id):
    shelf = tmp_path / "projects" / shelf_id
    shelf.mkdir(parents=True)
    marker = shelf / ".project.json"
    marker.write_text(f'{{"id":"{shelf_id}","name":"Previous room"}}', encoding="utf-8")
    assert projects.reconcile_projects(tmp_path) == 0
    with pytest.raises(FileNotFoundError):
        projects.create_project(tmp_path, "first")
    assert marker.is_file() and not projects._registry_path(tmp_path).exists()


@pytest.mark.parametrize("reservation", ["active", "deleting", "tombstoned", "legacy"])
def test_new_registration_keeps_the_first_project_chat_reservation(tmp_path, monkeypatch, reservation):
    import json

    first = projects.create_project(tmp_path, "kept")
    path = projects._registry_path(tmp_path)
    if reservation in {"deleting", "tombstoned"}:
        projects.begin_project_deletion(tmp_path, "kept")
        if reservation == "tombstoned":
            projects.complete_project_deletion(tmp_path, "kept")
    elif reservation == "legacy":
        original = json.loads(path.read_bytes())
        del original["projects"][0]["chat_id"]  # older row reserves its derived default
        projects._save(tmp_path, original)
    saved = path.read_bytes()
    monkeypatch.setattr(projects, "project_chat_id", lambda pid: first["chat_id"] if pid == "new" else project_chat_id(pid))
    with pytest.raises(ValueError, match="already reserved"):
        projects.create_project(tmp_path, "new")
    assert path.read_bytes() == saved
    assert projects.reserved_project_chat_ids(tmp_path, strict=True) == {first["chat_id"]}
    if reservation in {"active", "legacy"}:
        assert not projects.create_project(tmp_path, "kept")["created"]


@pytest.mark.parametrize("entry", ["create", "startup", "returned_primary"])
def test_first_registration_can_publish_without_link_or_exclusive_rename(tmp_path, monkeypatch, entry):
    import json
    from ouroboros import platform_layer as platform, project_registry_recovery as recovery

    path = recovery.registry_path(tmp_path)
    other_primary = b'{"projects": []}\n'

    def unavailable_link(*args):
        raise OSError(errno.ENOTSUP, "this volume does not link files")

    def unavailable_rename(source, target, *, exchange=False):
        if entry == "returned_primary" and not exchange:
            target.write_bytes(other_primary)  # another primary is already present at the replace precheck
        raise OSError(errno.ENOTSUP, "this volume does not support rename modes")

    monkeypatch.setattr(os, "link", unavailable_link)
    monkeypatch.setattr(recovery, "rename_exclusive", unavailable_rename)
    if entry == "returned_primary":
        with pytest.raises(FileNotFoundError):
            projects.create_project(tmp_path, "first")
        assert path.read_bytes() == other_primary
    elif entry == "startup":
        recovery.initialize_new_root(tmp_path)
        assert json.loads(path.read_bytes())["projects"] == []
    first = projects.create_project(tmp_path, "first")
    assert first["created"] and projects.reserved_project_chat_ids(tmp_path, strict=True) == {first["chat_id"]}
    qualified = os.name == "nt" and platform.kernel_file_locks_enforced(path.with_name(path.name + ".lock"))
    assert recovery.witness_path(tmp_path).read_bytes() == (path.read_bytes() if qualified else b"")
    assert recovery.recovery_notice(tmp_path) is None


def test_strict_restore_waits_for_no_clobber_publication(tmp_path, monkeypatch):
    from ouroboros import project_registry_recovery as recovery, utils

    projects.create_project(tmp_path, "kept")
    path = recovery.registry_path(tmp_path)
    saved = path.read_bytes()
    assert recovery._released_proof(path)
    path.rename(tmp_path / "kept-primary.json")
    native, replace = recovery.rename_exclusive, utils.replace_atomic

    def unavailable_link(*args):
        raise OSError(errno.ENOTSUP, "this volume does not link files")

    def exchange_only(source, target, *, exchange=False):
        if not exchange:
            raise OSError(errno.ENOTSUP, "this volume does not support no-replace rename")
        return native(source, target, exchange=True)

    def ordinary_replace(source, target, **kwargs):
        assert target != path, "restoration must never use ordinary replace"
        return replace(source, target, **kwargs)

    with monkeypatch.context() as unavailable:
        unavailable.setattr(os, "link", unavailable_link)
        unavailable.setattr(recovery, "rename_exclusive", exchange_only)
        unavailable.setattr(utils, "replace_atomic", ordinary_replace)
        with pytest.raises(OSError) as refused:
            recovery.reconcile(tmp_path)
        assert refused.value.errno == errno.ENOTSUP
        assert not path.exists() and recovery.recovery_notice(tmp_path) is None
        assert recovery.witness_path(tmp_path).read_bytes() == saved
    projects.reconcile_projects(tmp_path)
    assert path.read_bytes() == saved and recovery.recovery_notice(tmp_path)


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("qualified", [False, True])
def test_older_noop_preserves_maintenance_for_the_actual_primary(tmp_path, monkeypatch, restored, qualified):
    from ouroboros import platform_layer as platform, project_registry_recovery as recovery

    projects.create_project(tmp_path, "kept")
    path = recovery.registry_path(tmp_path)
    saved = path.read_bytes()
    if restored:
        path.unlink()
        projects.reconcile_projects(tmp_path)
    assert path.read_bytes() == recovery.witness_path(tmp_path).read_bytes() == saved
    lock = path.with_suffix(".json.lock")
    fd = platform.acquire_exclusive_file_lock(lock, timeout_sec=0)
    assert fd is not None
    platform.release_exclusive_file_lock(lock, fd)
    if not qualified:
        monkeypatch.setattr(platform, "kernel_file_locks_enforced", lambda *_: False)
    (tmp_path / "projects/legacy/knowledge").mkdir(parents=True)
    assert projects.reconcile_projects(tmp_path) == (0 if restored else 1)
    assert {row["id"] for row in projects.list_projects(tmp_path)} == ({"kept"} if restored else {"kept", "legacy"})


@pytest.mark.parametrize("published", [False, True])
def test_cold_restore_keeps_its_certificate_across_interruption(tmp_path, monkeypatch, published):
    from ouroboros import project_registry_recovery as recovery

    projects.create_project(tmp_path, "kept")
    path = recovery.registry_path(tmp_path)
    saved = path.read_bytes()
    path.unlink()
    swap = recovery.rename_exclusive
    publish = recovery._publish_new_file

    def interrupted_release(source, target, *, exchange=False):
        if exchange and target.is_dir():
            raise OSError("interrupted with the certificate still private")
        return swap(source, target, exchange=exchange)

    def interrupted_publish(source, target):
        if published:
            publish(source, target)
        raise OSError("interrupted recovery")

    with monkeypatch.context() as cut:
        cut.setattr(recovery, "rename_exclusive", interrupted_release)
        cut.setattr(recovery, "_publish_new_file", interrupted_publish)
        projects.reconcile_projects(tmp_path)
    if os.name != "nt":
        assert path.with_suffix(".json.lock").is_dir()
    assert path.exists() == published
    if published:
        assert path.read_bytes() == saved
        path.unlink()  # another loss before a healthy tick could certify the primary anew
    projects.reconcile_projects(tmp_path)
    assert path.read_bytes() == saved and recovery.recovery_notice(tmp_path)


def test_waiting_noop_keeps_the_commit_that_precedes_it(tmp_path, monkeypatch):
    from ouroboros import platform_layer as platform, project_registry_recovery as recovery

    projects.create_project(tmp_path, "kept")
    path = recovery.registry_path(tmp_path)
    waiting = Event()
    acquire = platform.file_lock_exclusive_nb

    def observe_contention(fd):
        try:
            acquire(fd)
        except BlockingIOError:
            waiting.set()
            raise

    def noop():
        with recovery.writer_lock(path, timeout=10):
            pass

    monkeypatch.setattr(platform, "file_lock_exclusive_nb", observe_contention)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with recovery.writer_lock(path):
            projects.update_project(tmp_path, "kept", name="Latest")
            waiter = pool.submit(noop)
            assert waiting.wait(5)
        waiter.result(timeout=10)
    saved = path.read_bytes()
    path.unlink()
    projects.reconcile_projects(tmp_path)
    assert path.read_bytes() == saved and projects.get_project(tmp_path, "kept")["name"] == "Latest"


@pytest.mark.parametrize("returned", [False, True])
def test_save_certifies_the_bytes_of_its_own_replacement(tmp_path, monkeypatch, returned):
    from ouroboros import project_registry_recovery as recovery, utils

    projects.create_project(tmp_path, "kept")
    path = recovery.registry_path(tmp_path)
    replace = utils.replace_atomic

    def publish(source, target):
        replace(source, target)
        if returned and target == path:
            other = target.with_name("returned.json")
            other.write_bytes(b'{"projects":[]}')
            replace(other, target)

    with monkeypatch.context() as writing:
        writing.setattr(recovery, "read_primary", lambda *_: pytest.fail("save reread its serialized registry"))
        writing.setattr(utils, "replace_atomic", publish)
        projects.update_project(tmp_path, "kept", name="Latest")
    saved = path.read_bytes()
    path.unlink()
    projects.reconcile_projects(tmp_path)
    if returned:
        assert saved == b'{"projects":[]}' and not path.exists()
    else:
        assert path.read_bytes() == saved and projects.get_project(tmp_path, "kept")["name"] == "Latest"


def test_the_web_ingress_proof_is_recomputed_only_when_its_sources_move(tmp_path, monkeypatch):
    """While the registry is absent every strict read re-proves first use, and the ingress scan inside the
    proof reads every non-Main inbound row. One answer per identity of everything the scan reads (store and
    WAL, owner bindings, initialization witness) keeps the hot re-proofs off the growing history; a moved
    source re-reads all of it."""
    import json
    from supervisor.message_ingress import accept_chat
    from supervisor.state_initialization import complete, mark_pending

    monkeypatch.setattr(store, "_journal_mode", lambda *_: "PERSIST")
    scans = []
    real = evidence._scan_web_project_ingress

    def counted(conn, root):
        answer = real(conn, root)
        scans.append(answer)
        return answer

    monkeypatch.setattr(evidence, "_scan_web_project_ingress", counted)
    external = project_chat_id("remote")
    identity = mark_pending(tmp_path, origin="first_boot")
    assert complete(tmp_path, identity, adopted=False)
    state = tmp_path / "state" / "state.json"
    state.write_text(json.dumps({"initialization_id": identity,
                                 "owner_external_chat_id": external + 1}), encoding="utf-8")
    late = {"task_id": "earlier-work", "quiz_id": "q", "answer_snapshot": {
        "task_id": "earlier-work", "quiz": {"quiz_id": "q", "state": "answered", "answered_after_terminal": True,
                                             "publication_chat_id": external, "publication_evidence": "pre_exposure"}}}
    accept_chat(external, 1, "An answer", drive_root=tmp_path, source="web", client_message_id="synthetic",
                message_meta={"late_answer": late})
    # The owner bindings name another chat: the decision stays ambiguous and first use is refused.
    assert evidence.could_lack_project_history(tmp_path) is False
    assert evidence.could_lack_project_history(tmp_path) is False
    assert scans == [True], "unchanged sources reuse the one proof"
    state.write_text(json.dumps({"initialization_id": identity,
                                 "owner_external_chat_id": external}), encoding="utf-8")
    assert evidence.could_lack_project_history(tmp_path) is True, "the bound owner chat clears the decision"
    assert scans == [True, False], "the moved owner bindings re-read all"
    dialogue.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "text": "a Main row"})
    assert evidence.could_lack_project_history(tmp_path) is True
    assert len(scans) == 3, "the moved store re-reads all"


def test_ingress_proof_reads_one_cached_answer_and_never_mixes_source_generations(tmp_path, monkeypatch):
    """One lookup tolerates a concurrent cache clear. Before and after a scan the captured key
    must still name every source: otherwise only ambiguity is safe, and no answer is memoized."""

    proof = ("synthetic-dialogue",)

    class OneRead(dict):
        def __init__(self):
            super().__init__({proof: False})
            self.reads = 0

        def get(self, key, default=None):
            self.reads += 1
            return super().get(key, default)

        def __contains__(self, key):
            raise AssertionError("the cached answer must use one get, not contains followed by get")

    cached = OneRead()
    monkeypatch.setattr(evidence, "_INGRESS_PROOFS", cached)
    monkeypatch.setattr(evidence, "ingress_proof_key", lambda _root: proof)
    assert evidence.could_have_web_project_ingress(object(), tmp_path, proof) is False
    assert cached.reads == 1

    monkeypatch.setattr(evidence, "ingress_proof_key", lambda _root: ("another-generation",))
    assert evidence.could_have_web_project_ingress(object(), tmp_path, proof) is True, "the sources moved before reading"

    calls = []

    def read_key(root):
        calls.append(("key", root))
        return ("another-generation",) if ("scan", root) in calls else proof

    def scan(_conn, root):
        calls.append(("scan", root))
        return False

    monkeypatch.setattr(evidence, "ingress_proof_key", read_key)
    monkeypatch.setattr(evidence, "_INGRESS_PROOFS", {})
    monkeypatch.setattr(evidence, "_scan_web_project_ingress", scan)
    assert evidence.could_have_web_project_ingress(object(), tmp_path, proof) is True, "the sources moved during the scan"
    assert calls == [("key", tmp_path), ("scan", tmp_path), ("key", tmp_path)]
    assert evidence._INGRESS_PROOFS == {}, "a mixed-generation absence is never memoized"
