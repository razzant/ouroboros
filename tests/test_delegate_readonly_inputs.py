"""Readonly session inputs preserve native read policy at the actual start seam."""
import copy
import hashlib
import json
import os
import pathlib
import shutil
import stat
import sys

import pytest

from tests.test_delegated_skill_payload import _payload_ctx, _StartStub


def test_readonly_source_permissions_do_not_prevent_exact_session_input(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon, safety
    from ouroboros.tools.registry import ToolRegistry

    ctx = _payload_ctx(tmp_path, monkeypatch)
    ctx.task_metadata.update(resource_intent={"kind": "explicit_none"}, parent_task_id="parent")
    parent = tmp_path / "data/task_drives/parent"
    parent.mkdir(parents=True)
    source = parent / "notes.txt"
    token = "ghp_" + "a" * 32
    source.write_text("Parent input with " + token, encoding="utf-8")
    source.chmod(0o444)
    binary = parent / "fixture.bin"
    binary.write_bytes(b"\xff\x00" + token.encode() + b"\r\n")
    owner_secret = ctx.drive_root / "settings.json"
    owner_secret.write_text('{"owner": "control"}', encoding="utf-8")
    secret_alias = parent / "ordinary.txt"
    secret_alias.hardlink_to(owner_secret)
    protected = parent / "oracle.txt"
    protected.write_text("black box bytes", encoding="utf-8")
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [{
        "id": "oracle", "role": "black_box_reference", "paths": [str(protected)],
    }]}}
    native = copy.copy(ctx)
    native.task_constraint = {"mode": "local_readonly_subagent"}
    reader = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    reader.set_context(native)
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    readable = reader.execute("read_file", {"root": "task_drive", "path": str(source)})
    assert "Parent input with " + token in readable
    assert "control" in reader.execute("read_file", {"root": "task_drive", "path": str(secret_alias)})
    assert "BLOCKED" in reader.execute("read_file", {"root": "task_drive", "path": str(protected)})
    seen = {}

    class Gateway(_StartStub):
        def start_run(self, request, **kwargs):
            scope = pathlib.Path(request["scope"]["root"])
            manifest_text = (scope / "inputs.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            sources = {row["source"] for row in manifest}
            assert str(secret_alias) in sources and str(protected) not in sources
            [note] = [row for row in manifest if row["source"] == str(source)]
            delivered = (scope / note["local"]).read_bytes()
            assert delivered == source.read_bytes()
            assert "masked_spans" not in note
            assert note["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
            assert note["sha256"] == hashlib.sha256(delivered).hexdigest()
            [binary_row] = [row for row in manifest if row["source"] == str(binary)]
            assert (scope / binary_row["local"]).read_bytes() == binary.read_bytes()
            assert binary_row["source_sha256"] == binary_row["sha256"]
            return super().start_run(request, **kwargs)

    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: Gateway(seen))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    try:
        result = registry.execute_result("delegate_start", {
            "subagent_id": "payload-session", "access": "readonly", "prompt": "Read the parent notes.",
        })
        assert result.status == "ok", result.text
        assert json.loads(result.text).get("run_id") == "run-p1", result.text
        assert source.read_text(encoding="utf-8") == "Parent input with " + token
    finally:
        source.chmod(0o644)


def test_input_copy_failure_is_typed_unrun_without_pending_engine_custody(tmp_path, monkeypatch):
    from ouroboros import artifacts, claudexor_daemon, delegate_custody, safety
    from ouroboros.tools.registry import ToolRegistry

    ctx = _payload_ctx(tmp_path, monkeypatch)
    ctx.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    (ctx.active_repo_dir() / "notes.txt").write_text("permitted input", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: _StartStub(seen))
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))

    def unavailable(*args, **kwargs):
        raise OSError("input verification failed")

    monkeypatch.setattr(artifacts, "stream_artifact_file", unavailable)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = json.loads(registry.execute("delegate_start", {
        "subagent_id": "payload-session", "access": "readonly", "prompt": "Read notes.",
    }))
    assert result["reason"] == "readonly_inputs_unavailable" and result["definitely_unrun"] is True
    assert "request" not in seen and not delegate_custody.pending_invocations(ctx.drive_root)


@pytest.mark.parametrize("change", ["edited", "deleted"])
def test_unknown_start_replays_same_exact_inputs_and_durable_invocation(tmp_path, monkeypatch, change):
    from ouroboros import claudexor_daemon, delegate_custody, safety
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools.registry import ToolRegistry

    ctx = _payload_ctx(tmp_path, monkeypatch)
    ctx.task_metadata["resource_intent"] = {"kind": "explicit_none"}
    source = ctx.active_repo_dir() / "notes.txt"
    source.write_text("original input", encoding="utf-8")
    requests = []

    class Gateway(_StartStub):
        def start_run(self, request, **kwargs):
            requests.append((request, kwargs))
            if len(requests) == 1:
                raise ClaudexorUnavailable("daemon_unreachable", "start response lost")
            scope = pathlib.Path(request["scope"]["root"])
            [note] = [row for row in json.loads((scope / "inputs.json").read_text(encoding="utf-8"))
                      if row["source"] == str(source)]
            assert (scope / note["local"]).read_text(encoding="utf-8") == "original input"
            return super().start_run(request, **kwargs)

    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: Gateway({}))
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = json.loads(registry.execute("delegate_start", {
        "subagent_id": "payload-session", "access": "readonly", "prompt": "Read notes.",
    }))
    invocation = result["pending_invocation_id"]
    source.write_text("later parent revision", encoding="utf-8") if change == "edited" else source.unlink()
    result = json.loads(registry.execute("delegate_start", {"retry_of": invocation, "prompt": "Read notes."}))
    assert result["run_id"] == "run-p1" and result["idempotent_recovery"] is True
    assert requests[0] == requests[1]
    assert delegate_custody.replay(ctx.drive_root)["run-p1"].invocation_id == invocation
    assert not delegate_custody.pending_invocations(ctx.drive_root)


# -- #2945: lineage copies through the one verified copier, at the real start seam -------------------------------

ROOT = "root-r"
SCOPE = ("evidence", "test", "wire")  # where a record_compression pass may one day look
TEXT = b"A finished record: the same line again keeps it compressible.\n"


def _root_materials(data):
    """A root task's drive and artifacts as delegated work leaves them: binary, Git with read-only packs, QA, and
    records a later compression pass may want (run evidence, test evidence, the call wire journal, small today)."""
    from ouroboros.task_results import write_task_result

    data.mkdir(parents=True, exist_ok=True)
    write_task_result(data, ROOT, "running", root_task_id=ROOT, billing_group={  # its children's billing lineage
        "billing_group_id": ROOT, "billing_group_limit_usd": 10.0,
        "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "rev-1"})
    drive, store = data / "task_drives" / ROOT, data / "task_results" / "artifacts" / ROOT
    files = {"binary": (drive / "fixture.bin", os.urandom(300_000)),
             "pack": (drive / "repo" / ".git" / "objects" / "pack" / "pack-1.pack", os.urandom(200_000)),
             "head": (drive / "repo" / ".git" / "HEAD", b"ref: refs/heads/main\n"),
             "qa": (drive / "qa" / "results.json", b'{"passed": 12}\n'),
             "report": (store / "report.md", b"# Report\n" * 9000),
             "evidence": (store / "delegated_runs" / "run-1" / "evidence" / "out.log", TEXT * 2048),
             "test": (store / "source_handles" / "test-evidence" / "run-1" / "stdout.bin", TEXT * 2048),
             "wire": (store / "conversation-wire.jsonl", b'{"seq": 1}\n')}
    for path, raw in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    if os.name != "nt":  # as Git leaves a pack: an ordinary read-only file that still clones
        files["pack"][0].chmod(0o444)
    return {name: path for name, (path, _raw) in files.items()}


def _lineage_ctx(tmp_path, monkeypatch, task_id, parent):
    ctx = _payload_ctx(tmp_path, monkeypatch)
    ctx.task_id = task_id
    ctx.task_metadata = {"root_task_id": ROOT, "parent_task_id": parent, "resource_intent": {"kind": "explicit_none"}}
    return ctx


class _Reader(_StartStub):
    """The engine double reads every staged copy when the start reaches it, as the leaf would, and keeps that view."""

    def __init__(self):
        super().__init__({})
        self.views = []

    def start_run(self, request, *, idempotency_key=""):
        scope = pathlib.Path(request["scope"]["root"])
        rows = json.loads((scope / "inputs.json").read_text(encoding="utf-8"))
        self.views.append({"scope": scope, "rows": rows, "request": request,
                           "bytes": {row["source"]: (scope / row["local"]).read_bytes() for row in rows}})
        return {"runId": f"run-{len(self.views)}", "runDir": "/tmp/run"}


class _Meter:
    """What the copier physically did: each clone attempt's answer, and the bytes each stream wrote into a sink."""

    def __init__(self, monkeypatch):
        from ouroboros import artifacts

        self.clones, self.streamed = [], []
        clone, stream = artifacts.clone_file, artifacts.stream_artifact_file

        def counted_clone(source, destination):
            done = clone(source, destination)
            self.clones.append((str(source), done))
            return done

        def counted_stream(path, sink=None, **kwargs):
            measured = stream(path, sink, **kwargs)
            if sink is not None:
                self.streamed.append((str(path), measured["size"]))
            return measured

        monkeypatch.setattr(artifacts, "clone_file", counted_clone)
        monkeypatch.setattr(artifacts, "stream_artifact_file", counted_stream)


def _start_as(tmp_path, monkeypatch, engine, task_id, parent):
    from ouroboros import claudexor_daemon, safety
    from ouroboros.tools.registry import ToolRegistry

    ctx = _lineage_ctx(tmp_path, monkeypatch, task_id, parent)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: engine)
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return json.loads(registry.execute("delegate_start", {
        "subagent_id": "payload-session", "access": "readonly", "prompt": "Read the lineage."}))


@pytest.mark.parametrize("future_compression", [True, False], ids=["compression-capable", "no-future-compression"])
def test_siblings_and_a_nested_child_each_start_once_with_exact_independent_copies(tmp_path, monkeypatch, future_compression):
    from ouroboros import artifact_storage

    # Exercise the capability boundary, not a hardcoded platform name: when the
    # job cannot compress here, withholding a clone protects no later saving.
    monkeypatch.setattr(artifact_storage, "compression_supported", lambda: future_compression)
    data = tmp_path / "data"
    materials = _root_materials(data)
    expected = {str(path): path.read_bytes() for path in materials.values()}
    notes = data / "task_drives" / "c-one" / "notes.md"
    notes.parent.mkdir(parents=True)
    notes.write_bytes(b"the first sibling's notes\n")
    engine, meter = _Reader(), _Meter(monkeypatch)
    for task_id, parent in (("c-one", ROOT), ("c-two", ROOT), ("g-one", "c-one")):  # each starts exactly once
        assert _start_as(tmp_path, monkeypatch, engine, task_id, parent)["status"] == "started"
    assert [view["scope"].parents[1].name for view in engine.views] == ["c-one", "c-two", "g-one"]
    for view in engine.views:
        assert {source: view["bytes"][source] for source in expected} == expected, "every file, byte for byte"
        new_file = stat.S_IMODE(view["scope"].stat().st_mode) & 0o666
        for row in view["rows"]:
            local = view["scope"] / row["local"]
            assert row["sha256"] == row["source_sha256"] == hashlib.sha256(view["bytes"][row["source"]]).hexdigest()
            assert row["size"] == row["source_size"] == len(view["bytes"][row["source"]])
            info = os.lstat(local)
            assert info.st_nlink == 1 and info.st_ino != os.lstat(row["source"]).st_ino, "its own inode"
            assert stat.S_IMODE(info.st_mode) == new_file, "the folder's new-file mode, not the source's"
    assert [str(notes) in view["bytes"] for view in engine.views] == [True, False, True], "own, sibling, parent"
    lineage = {source for view in engine.views for source in view["bytes"]}
    records = {str(materials[name]) for name in SCOPE} if future_compression else set()
    assert not {source for source, _done in meter.clones} & records, "a record a pass may compress is never cloned"
    if not future_compression:
        assert {source for source, _done in meter.clones} == lineage, "no future compression to protect"
    assert sum(size for source, size in meter.streamed if source in records) == 3 * sum(map(len, (
        expected[source] for source in records))), "streamed into each start's copy, once per start"
    if sys.platform == "darwin":  # APFS: every other lineage file is cloned, and none of it is streamed
        assert {source for source, done in meter.clones if done} == lineage - records
        assert not [source for source, _size in meter.streamed if source in lineage - records]
        assert {name for name, path in materials.items() if artifact_storage._may_share(path)} == set(
            materials) - (set(SCOPE) if future_compression else set())
    # Independent inodes: an in-place rewrite of a source and a reader's write into one copy change nothing else.
    binary = materials["binary"]
    copies = [view["scope"] / next(row["local"] for row in view["rows"] if row["source"] == str(binary))
              for view in engine.views]
    with binary.open("r+b") as handle:
        handle.write(b"\x00" * 4096)  # the same size, other bytes
    with copies[0].open("r+b") as handle:
        handle.write(b"written by the first reader")
    assert copies[1].read_bytes() == copies[2].read_bytes() == expected[str(binary)]
    assert binary.read_bytes()[4096:] == expected[str(binary)][4096:] and copies[0].read_bytes() != binary.read_bytes()


@pytest.mark.skipif(sys.platform != "darwin", reason="real APFS clones and transparent compression")
def test_a_staged_record_stays_compressible_while_a_cloned_neighbour_cannot_be(tmp_path, monkeypatch):
    """APFS marks a clone's source may-share for good: the floor then keeps it (``shared``). The records a pass
    may compress stream, so after the staged scope retires the real job still compresses them, the small wire
    journal grown afterwards included; the cloned report, by design outside that scope, says why."""
    from ouroboros import artifact_storage, maintenance, record_compression
    from ouroboros.task_results import write_task_result

    data = tmp_path / "data"
    materials = _root_materials(data)
    assert not any(artifact_storage._may_share(path) for path in materials.values())
    engine = _Reader()
    assert _start_as(tmp_path, monkeypatch, engine, "c-one", ROOT)["status"] == "started"
    assert artifact_storage._may_share(materials["report"]) is True
    assert not any(artifact_storage._may_share(materials[name]) for name in SCOPE)
    shutil.rmtree(engine.views[0]["scope"])  # the staged scope retires with its drive, its clones with it
    assert artifact_storage.compression_outcome(materials["report"]) == "shared", "the mark outlives its clone"
    with materials["wire"].open("ab") as handle:
        handle.write(TEXT * 2048)  # the call's journal grows past the floor after the start
    identities = {name: hashlib.sha256(materials[name].read_bytes()).hexdigest() for name in SCOPE}
    write_task_result(data, ROOT, "completed")
    monkeypatch.setattr(record_compression, "_queue_live", lambda root: (lambda task_id: False))
    assert maintenance.run_job(data, record_compression.JOB, should_stop=lambda: False)["state"] == "done"
    for name in SCOPE:
        assert os.lstat(materials[name]).st_flags & stat.UF_COMPRESSED, name
        assert hashlib.sha256(materials[name].read_bytes()).hexdigest() == identities[name]
    # A later start can preserve already-compressed records as clones: the floor
    # reports compressed_already before shared, so there is no future saving to block.
    meter = _Meter(monkeypatch)
    assert _start_as(tmp_path, monkeypatch, engine, "c-two", ROOT)["status"] == "started"
    view = engine.views[-1]
    for name in SCOPE:
        source = materials[name]
        local = view["scope"] / next(row["local"] for row in view["rows"] if row["source"] == str(source))
        assert (str(source), True) in meter.clones and str(source) not in {path for path, _size in meter.streamed}
        assert os.lstat(local).st_flags & stat.UF_COMPRESSED
        assert artifact_storage.compression_outcome(source) == "compressed_already"
        assert hashlib.sha256(local.read_bytes()).hexdigest() == identities[name]
        with local.open("r+b") as handle:
            handle.write(b"reader-local change")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == identities[name]


def test_store_files_and_flagged_sources_stream_exactly_into_removable_copies(tmp_path, monkeypatch):
    """A SQLite file streams (a clone opens and closes it, dropping this process's locks); an immutable source
    streams on macOS (its clone would carry the flag and refuse its own removal). Either way the copy is exact
    and the whole scope goes with its folder."""
    import sqlite3

    drive = _root_materials(tmp_path / "data")["binary"].parent
    database = drive / "state.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("create table facts (value text)")
        connection.execute("insert into facts values ('kept')")
    connection.close()
    locked = drive / "locked.txt"
    locked.write_bytes(b"an immutable source\n" * 4096)
    flag = getattr(stat, "UF_IMMUTABLE", 0) if sys.platform == "darwin" else 0
    if flag:
        os.chflags(locked, flag)
    engine, meter = _Reader(), _Meter(monkeypatch)
    try:
        assert _start_as(tmp_path, monkeypatch, engine, "c-one", ROOT)["status"] == "started"
        [view] = engine.views
        assert view["bytes"][str(database)] == database.read_bytes() and view["bytes"][str(locked)] == locked.read_bytes()
        assert (str(database), database.stat().st_size) in meter.streamed
        assert str(database) not in {source for source, _done in meter.clones}, "a store file is never cloned"
        if flag:
            assert (str(locked), locked.stat().st_size) in meter.streamed and (str(locked), True) not in meter.clones
        local = view["scope"] / next(row["local"] for row in view["rows"] if row["source"] == str(locked))
        assert not getattr(os.lstat(local), "st_flags", 0)
        shutil.rmtree(view["scope"])
        assert not view["scope"].exists() and database.exists() and locked.exists()
    finally:
        if flag:
            os.chflags(locked, 0)


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="a POSIX mode a non-root reader cannot pass")
def test_an_unreadable_lineage_file_refuses_the_start_unrun_with_nothing_sent(tmp_path, monkeypatch):
    from ouroboros import delegate_custody

    data = tmp_path / "data"
    materials = _root_materials(data)
    materials["qa"].chmod(0)
    engine = _Reader()
    try:
        result = _start_as(tmp_path, monkeypatch, engine, "c-one", ROOT)
    finally:
        materials["qa"].chmod(0o644)
    assert result["reason"] == "readonly_inputs_unavailable" and result["definitely_unrun"] is True, result
    assert engine.views == [] and not delegate_custody.pending_invocations(data)
    assert "input_files" not in result, "nothing claims to be prepared"
