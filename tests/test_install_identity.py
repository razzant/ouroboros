"""Stored paths grant no cross-install write authority; isolated and own drives still work."""
import json
import logging
import pathlib
import shutil
from types import SimpleNamespace

import pytest

from ouroboros import install_identity as identity, store
from ouroboros.task_results import load_task_result, write_task_result


def boot(monkeypatch, root):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    return identity.initialize_install()


@pytest.fixture
def installs(tmp_path, monkeypatch):
    a, b = tmp_path / "A", tmp_path / "B"
    boot(monkeypatch, a)
    write_task_result(a, "task", "completed", result="Original result", artifact_status="ready",
                      root_task_id="task", root_phase_checkpoint={"post_task_synthesis": "pending_once"})
    boot(monkeypatch, b)
    yield a, b
    store.close_connections()


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
@pytest.mark.parametrize("service", [False, True])
def test_foreign_store_refuses_before_any_connection_but_local_history_reads(
        installs, monkeypatch, journal, service):
    a, b = installs
    monkeypatch.setattr(store, "_journal_mode", lambda *_: journal)
    with store.open_store(b, write=True):
        pass  # The import owner publishes before service-mode transactions.
    connections, real = [], store.connection

    def connect(path, *args, **kwargs):
        connections.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(store, "connection", connect)
    with pytest.raises(identity.StoreForeign, match="belongs to another Ouroboros install") as refused:
        with store.open_store(a, write=True, service=service):
            pytest.fail("foreign write transaction opened")
    assert refused.value.reason == "foreign_install_root" and connections == []
    with pytest.raises(identity.StoreForeign):
        with store.open_store(a, write=False):
            pytest.fail("read transaction opened a foreign rw connection")
    assert connections == []
    with store.open_store(b, write=True, service=service) as txn:
        txn.set_meta("own_write", journal)
        assert txn.conn.execute("PRAGMA journal_mode").fetchone()[0] == journal.lower()
    with store.open_store(b, write=False) as txn:
        assert txn.meta("own_write") == journal


def test_empty_own_read_does_not_create_install_identity(tmp_path, monkeypatch):
    root = tmp_path / "fresh"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    with store.open_store(root, write=False) as txn:
        assert txn.task_ids() == []
    assert not root.exists()


@pytest.mark.parametrize("kind", ["custody", "mail", "projection"])
@pytest.mark.parametrize("layout", ["", "task_drives/task", "state/headless_tasks/task/data"])
def test_foreign_locks_never_create_a_lock_file(installs, kind, layout):
    from ouroboros.task_custody import task_custody_lock, task_mail_lock
    from ouroboros.terminal_projection import _publication_lock

    a, b = installs
    lock = {"custody": task_custody_lock, "mail": task_mail_lock, "projection": _publication_lock}[kind]
    foreign = a / layout
    before = {path.relative_to(a) for path in a.rglob("*")}
    with pytest.raises(identity.StoreForeign):
        with lock(foreign, "task"):
            pytest.fail("foreign lock acquired")
    assert {path.relative_to(a) for path in a.rglob("*")} == before
    with lock(b / layout, "task") as acquired:
        assert acquired


def test_projection_prepare_and_mailbox_refuse_foreign_effects(installs):
    from ouroboros.terminal_projection import _prepare
    from ouroboros.owner_mailbox import _mailbox_path, write_task_message

    a, b = installs
    with pytest.raises(identity.StoreForeign):
        _prepare(a, "task", {}, {})
    before = {path.relative_to(a) for path in a.rglob("*")}
    with pytest.raises(identity.StoreForeign):
        write_task_message(a, "must stay here", "task", source_task_id="sender")
    assert {path.relative_to(a) for path in a.rglob("*")} == before
    assert not _mailbox_path(a, "task").exists()
    assert write_task_message(b, "addressed", "task", source_task_id="sender")
    assert "addressed" in _mailbox_path(b, "task").read_text()


def test_owned_process_set_rejects_foreign_root_and_accepts_own_child(installs):
    from ouroboros import owned_shutdown

    a, b = installs
    entry = {"pid": 123, "purpose": "test", "scope": "task", "owner_task": "task", "fingerprint": {}}
    with pytest.raises(identity.StoreForeign):
        owned_shutdown.record_ledgered_process(a, entry)
    assert not owned_shutdown.owned_processes_path(a).exists()
    child = b / "task_drives" / "task"
    assert owned_shutdown.record_ledgered_process(child, entry)
    assert owned_shutdown.installation_root(child) == b
    assert owned_shutdown.owned_records(b)[0]["drive_root"] == str(child)


def test_copied_owned_process_pointer_never_reaches_the_stop_owner(installs, monkeypatch):
    from ouroboros import owned_shutdown

    a, b = installs
    called = []
    monkeypatch.setattr(owned_shutdown, "_stop_ledgered", lambda *args: called.append(args) or True)
    with pytest.raises(identity.StoreForeign):
        owned_shutdown._stop_entry(b, {"drive_root": str(a), "kind": "supervised"}, 0, set())
    assert called == []
    assert owned_shutdown._stop_entry(b, {"drive_root": str(b), "kind": "supervised"}, 0, set())
    assert len(called) == 1



@pytest.mark.parametrize("kind", ["foreground", "service"])
@pytest.mark.parametrize("pointer", ["drive_root", "record_path"])
def test_copied_registry_paths_never_reach_executor_load_stop_or_forget(installs, monkeypatch, kind, pointer):
    from ouroboros import owned_shutdown, workspace_executor as executor

    a, b = installs
    own_path = b / "state" / executor._PROCESS_STATE_DIR / "own.json"
    foreign = {"record_id": "foreign", "kind": kind, "drive_root": str(b),
               "record_path": str(own_path.with_name("foreign.json"))}
    foreign[pointer] = str(a if pointer == "drive_root" else a / "state" / "processes" / "foreign.json")
    own = {"record_id": "own", "kind": kind, "drive_root": str(b), "record_path": str(own_path)}
    owned_shutdown.owned_processes_path(b).write_text(json.dumps({"records": {"foreign": foreign, "own": own}}))
    loaded, stopped, forgotten = [], [], []
    monkeypatch.setattr(executor, "_FOREGROUND", {})
    monkeypatch.setattr(executor, "_SERVICES", {})
    monkeypatch.setattr(executor, "_load_process_record",
                        lambda path: loaded.append(path) or {"record_type": kind, "id": path.stem})
    monkeypatch.setattr(executor, "_valid_process_record", lambda *a, **k: True)
    monkeypatch.setattr(executor, "_stop_record_process", lambda path, *a, **k: stopped.append(path) or True)
    monkeypatch.setattr(executor, "_forget_process", lambda path: forgotten.append(path))
    cleanup = executor.kill_all_foreground if kind == "foreground" else executor._kill_durable_service_records
    assert len(cleanup(b)) == 1
    assert loaded == stopped == forgotten == [own_path]
    assert "foreign" in {entry["record_id"] for entry in owned_shutdown.owned_records(b)}


@pytest.mark.parametrize("relocation", ["copy", "move", "original"])
def test_legacy_process_import_requires_original_install_provenance(tmp_path, monkeypatch, relocation):
    from ouroboros import owned_shutdown, workspace_executor as executor

    a, b = tmp_path / "A", tmp_path / "B"
    boot(monkeypatch, a)
    payload = {"record_type": "service", "service_id": "task:svc", "task_id": "task", "name": "svc",
               "executor_type": "docker_exec", "executor_id": "docker", "container_name": "bench",
               "backend_pid": "4242", "host_pid": 0}
    path = executor._register_process(a, payload)
    before = path.read_bytes()
    owned_shutdown.owned_processes_path(a).unlink()  # legacy installation, before its one-time import
    if relocation == "copy":
        shutil.copytree(a, b)
    elif relocation == "move":
        a.rename(b)
    else:
        b = a
    boot(monkeypatch, b)
    loaded, real = [], executor._load_process_record
    monkeypatch.setattr(executor, "_load_process_record", lambda path: loaded.append(path) or real(path))
    copied_path = b / path.relative_to(a)
    assert owned_shutdown.import_inherited_records(b) == (1 if relocation == "original" else 0)
    assert loaded == ([copied_path] if relocation == "original" else [])
    mark = json.loads(owned_shutdown.owned_processes_path(b).read_text())["inherited_import"]
    if relocation != "original":
        assert mark["reason_code"] == "foreign_install_root"
        assert not owned_shutdown.owned_records(b)
    assert copied_path.read_bytes() == before
    assert owned_shutdown.import_inherited_records(b) is None
    own_path = executor._register_process(b, payload)
    assert own_path in owned_shutdown.executor_record_paths(b, "service")


def test_foreign_stop_does_not_start_this_installs_shutdown_grace(installs, monkeypatch):
    from ouroboros import owned_shutdown

    a, b = installs
    stop, stamped = owned_shutdown._Stop(), []
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", stop)
    monkeypatch.setattr(owned_shutdown, "_stamp", lambda root, *a, **k: stamped.append(root))
    with pytest.raises(identity.StoreForeign):
        owned_shutdown.begin_owned_stop(a)
    assert stop.deadline is None and not stop.started and stamped == []
    owned_shutdown.begin_owned_stop(b)
    assert stop.deadline is not None and stamped == [b]


@pytest.mark.parametrize("pointer", identity.TASK_ROOT_FIELDS)
def test_foreign_synthesis_member_rests_without_reading_its_authority(installs, monkeypatch, pointer):
    from ouroboros import agent_task_pipeline as pipeline, obligations

    a, b = installs
    write_task_result(b, "task", "completed", result="Copied answer", root_task_id="task",
                      **{pointer: str(a)}, root_phase_checkpoint={"post_task_synthesis": "pending_once"})
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: pytest.fail("foreign dispatch"))
    bodies, real = [], store.Txn.body_text

    def body(txn, tid):
        assert pathlib.Path(txn.root) != a, "foreign task was polled"
        bodies.append(tid)
        return real(txn, tid)

    monkeypatch.setattr(store.Txn, "body_text", body)
    assert pipeline.recover_pending_root_post_task_synthesis(b) == 0
    with store.open_store(b, write=False) as txn:
        saved = txn.load("task")
        mark = txn.get_row("duty_rest", "synthesis_recovery:task")
        revision = txn.revision()
    assert saved["reason_code"] == "foreign_install_root"
    assert str(a) in saved["reason_detail"] and "move it explicitly" in saved["reason_detail"]
    assert mark["waits_on"] == ["foreign_install_root"]
    assert "task" in obligations.members(b, "synthesis")
    bodies.clear()
    for _ in range(3):
        assert pipeline.recover_pending_root_post_task_synthesis(b) == 0
    assert bodies == []
    with store.open_store(b, write=False) as txn:
        assert txn.revision() == revision


def test_projection_duty_rests_a_foreign_saved_child(installs, monkeypatch):
    from ouroboros import terminal_projection as projection

    a, b = installs
    write_task_result(b, "task", "completed", result="Copied result", root_task_id="task",
                      drive_root=str(a), artifact_status="ready")
    assert projection.reconcile_terminal_projections(b) == 0
    with store.open_store(b, write=False) as txn:
        assert txn.get_row("duty_rest", "terminal_projection:task")["waits_on"] == ["foreign_install_root"]
    monkeypatch.setattr(projection, "settle_terminal_projection", lambda *a, **k: pytest.fail("rested member polled"))
    assert projection.reconcile_terminal_projections(b) == 0


def test_headless_refuses_foreign_child_before_copy_or_cleanup(installs, monkeypatch):
    from ouroboros import headless

    a, b = installs
    cache = a / "state" / "code_intel"
    cache.mkdir()
    (cache / "keep").write_text("original")
    monkeypatch.setattr(headless, "copy_child_task_result", lambda *a, **k: pytest.fail("foreign copy"))
    report = headless.prepare_terminal_task_files(b, {"id": "task", "drive_root": str(a)})
    assert report["reason_code"] == "foreign_install_root"
    assert (cache / "keep").read_text() == "original"


def test_own_child_terminal_files_settle_and_bare_root_still_writes(installs, tmp_path):
    from ouroboros import headless
    from ouroboros.post_task_checkpoint import root_checkpoint_roots

    _a, b = installs
    child = b / "state" / "headless_tasks" / "child" / "data"
    write_task_result(child, "child", "completed", result="Own child result", artifact_status="ready")
    report = headless.prepare_terminal_task_files(b, {"id": "child", "drive_root": str(child)})
    assert not report["error"] and report["result"]["result"] == "Own child result"
    assert root_checkpoint_roots(SimpleNamespace(drive_root=child), {"budget_drive_root": str(b)}) == [b]
    bare = tmp_path / "isolated"
    write_task_result(bare, "isolated", "completed", result="Still supported")
    assert identity.owned_root(bare) and not (bare / identity.IDENTITY_PATH).exists()
    assert load_task_result(bare, "isolated")["result"] == "Still supported"


@pytest.mark.parametrize("move", [False, True])
def test_copied_or_moved_root_mints_its_own_identity_once(tmp_path, monkeypatch, caplog, move):
    a, b = tmp_path / "A", tmp_path / "B"
    original = boot(monkeypatch, a)
    if move:
        a.rename(b)
    else:
        shutil.copytree(a, b)
    with caplog.at_level(logging.WARNING, logger=identity.__name__):
        current = boot(monkeypatch, b)
        assert identity.initialize_install() == current
    assert current["id"] != original["id"] and current["root"] == str(b)
    assert len(caplog.records) == 1 and "separate Ouroboros install" in caplog.text
    assert identity.owned_root(b)
    write_task_result(b, "own", "completed", result="Works after relocation")
    if not move:
        assert json.loads((a / identity.IDENTITY_PATH).read_text()) == original
        assert not identity.owned_root(a)
    else:
        assert not a.exists()
        with pytest.raises(identity.StoreForeign):
            write_task_result(a, "must-not-recreate", "completed")
        assert not a.exists()
        c = tmp_path / "C"
        store.close_connections()
        b.rename(c)
        boot(monkeypatch, c)
        assert identity.owned_root(c)
        assert not identity.owned_root(a) and not identity.owned_root(b)
        assert not a.exists() and not b.exists()


def test_unreadable_identity_never_becomes_an_isolated_root(installs):
    a, b = installs
    (a / identity.IDENTITY_PATH).write_text("broken")
    with pytest.raises(store.StoreError, match="identity unreadable"):
        identity.owned_root(a / "task_drives" / "task")
    assert identity.owned_root(b)


def test_terminal_money_never_reaches_a_foreign_ledger_root(installs, monkeypatch):
    """A copied custody row whose ledger root is the original install keeps its money owed here."""
    from ouroboros import delegate_collection as body
    from ouroboros import delegate_custody as custody
    from ouroboros import delegate_custody_current as current
    from ouroboros import usage_accounting

    a, b = installs
    calls = []
    monkeypatch.setattr(usage_accounting, "record_subscription_session", lambda *args, **kw: calls.append(kw))
    detail = {"summary": {"runId": "run-a", "state": "succeeded", "spendUsd": 0.1, "spendEstimated": False,
                          "effectiveAccess": "readonly"}, "lastSeq": 4,
              "primaryOutput": {"text": "answer", "bytes": 6, "truncated": False}}
    for rid, ledger in (("run-a", a), ("run-b", b)):
        entry = custody.RunCustody(run_id=rid, task_id="task-" + rid, invocation_id="inv-" + rid, route_id="test",
                                   model="test", access="readonly", mode="ask", ledger_root=str(ledger))
        assert custody.record_started(b, entry)
        body.collect_terminal(b, rid, detail={**detail, "summary": {**detail["summary"], "runId": rid}})
    assert [kw["drive_root"] for kw in calls] == [b]  # only this install's own ledger
    assert not current.run(b, "run-a").ledger_recorded
    assert current.run(b, "run-b").ledger_recorded
    assert not (a / "state" / "usage.sqlite").exists()


def test_a_restored_row_pointing_at_another_install_never_launches_and_says_why(installs, monkeypatch):
    """B's queue may hold a copied row whose stored roots are A's: B never launches it, and B's own
    row for that task names A (status unchanged); a row on B's own roots stays eligible."""
    from supervisor import workers, worker_assignment as assignment
    from supervisor import workspace_preparation

    a, b = installs
    write_task_result(b, "copied", "queued", result="Queued in the copy", drive_root=str(a))
    write_task_result(b, "own", "queued", result="Queued here", drive_root=str(b))
    monkeypatch.setattr(workers, "DRIVE_ROOT", b, raising=False)
    submitted = []
    generation = SimpleNamespace(closed=lambda: False, submit=lambda job, lane: submitted.append((job, lane)) or True)
    monkeypatch.setattr(workspace_preparation, "current_generation", lambda: generation)
    copied = {"id": "copied", "drive_root": str(a), "budget_drive_root": str(a)}
    own = {"id": "own", "drive_root": str(b)}
    assert assignment._foreign_roots_refused(copied) is True
    assert assignment._foreign_roots_refused(copied) is True  # the reason is written once, not per pass
    assert assignment._foreign_roots_refused(own) is False
    assert [lane for _job, lane in submitted] == [workspace_preparation.ADMISSION_LANE]
    submitted[0][0]()
    row = load_task_result(b, "copied")
    assert row["status"] == "queued" and row["reason_code"] == "foreign_install_root"
    assert str(a) in row["reason_detail"]
    assert "reason_code" not in (load_task_result(b, "own") or {})


def test_forward_to_worker_names_a_foreign_child_mailbox_and_writes_nothing_there(installs):
    """A parent in B whose copied child row records A's drive gets a typed refusal; A gains no file."""
    from ouroboros.task_results import STATUS_RUNNING
    from ouroboros.tools.core import _forward_to_worker

    a, b = installs
    write_task_result(b, "parent1", STATUS_RUNNING, root_task_id="parent1", result="running")
    write_task_result(b, "copied-child", STATUS_RUNNING, child_drive_root=str(a), parent_task_id="parent1",
                      root_task_id="parent1", result="running")
    own_child = b / "state" / "headless_tasks" / "own-child" / "data"
    write_task_result(b, "own-child", STATUS_RUNNING, child_drive_root=str(own_child), parent_task_id="parent1",
                      root_task_id="parent1", result="running")
    ctx = SimpleNamespace(drive_root=b, task_id="parent1")
    before = {path.relative_to(a) for path in a.rglob("*")}
    refused = _forward_to_worker(ctx, "copied-child", "do not cross installs")
    assert "TASK_FOREIGN_INSTALL" in refused and "NOT delivered" in refused and str(a) in refused
    assert {path.relative_to(a) for path in a.rglob("*")} == before
    delivered = _forward_to_worker(ctx, "own-child", "stay here")
    assert "Message forwarded" in delivered
    assert "stay here" in (own_child / "memory" / "owner_mailbox" / "own-child.jsonl").read_text(encoding="utf-8")


def test_a_copied_main_debt_never_stops_this_install_replaying_its_own(installs):
    from supervisor import terminal_delivery as td

    a, b = installs
    write_task_result(b, "foreign", "completed", result="Copied answer", root_task_id="foreign",
                      drive_root=str(a), artifact_status="ready")
    write_task_result(b, "own", "completed", result="Own answer", root_task_id="own")
    old = "2000-01-01T00:00:00+00:00"
    pending = {
        "project-completion:foreign": {"chat_id": 1, "task_id": "foreign", "text": "copied main card",
                                        "terminal_projection_token": "tok", "registered_at": old, "ts": old},
        "final:own:abc": {"chat_id": 1, "task_id": "own", "text": "Own answer", "registered_at": old, "ts": old},
    }
    registry = b / "state" / "terminal_deliveries.json"
    registry.write_text(json.dumps({"schema_version": 3, "delivered": [], "pending": pending}), encoding="utf-8")
    puts = []
    bus = SimpleNamespace(put_unless_outstanding=lambda event: puts.append(event) or True,
                          delivery_outstanding=lambda _did: False)
    assert td.replay_pending_deliveries(b, event_queue=bus) == ["final:own:abc"]
    assert [event["delivery_id"] for event in puts] == ["final:own:abc"]
    assert {row["delivery_id"] for row in td.pending_deliveries(b)} == set(pending)  # the copied debt stays retained


def test_this_installs_shutdown_door_never_reads_its_identity(installs, monkeypatch):
    from ouroboros import owned_shutdown

    a, b = installs
    (b / identity.IDENTITY_PATH).write_text("{not json", encoding="utf-8")
    stop, stamped = owned_shutdown._Stop(), []
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", stop)
    monkeypatch.setattr(owned_shutdown, "_stamp", lambda root, *_a, **_k: stamped.append(root))
    owned_shutdown.begin_owned_stop(b)  # an unreadable identity never blocks this install's own stop
    assert stop.deadline is not None and stamped == [b]
    with pytest.raises(store.StoreError):
        owned_shutdown.installation_root(a / "task_drives" / "task")  # another root still needs a readable verdict


def test_one_directory_under_another_spelling_keeps_its_identity(tmp_path, monkeypatch):
    root = tmp_path / "Data"
    first = boot(monkeypatch, root)
    variant = tmp_path / "DATA"
    if not variant.exists():
        pytest.skip("case-sensitive filesystem: two spellings are two directories")
    assert boot(monkeypatch, variant)["id"] == first["id"]  # no copy is minted for a spelling
    assert identity.owned_root(root / "task_drives" / "task")  # pointers in the first spelling stay own
