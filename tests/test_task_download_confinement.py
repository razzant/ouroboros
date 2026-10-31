"""#1297 consumer qualification. Real loopback HTTP, no backend emulation.

Run via safe_test. On Windows the positive download tests deliberately fail
while task_archive.CONFINED remains false; do not turn that gate into a skip.
Only the lexical-parser test below emulates Windows, and proves no native I/O.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from hashlib import sha256
from pathlib import Path

import httpx
import pytest
import uvicorn

from ouroboros import artifacts, confined_files, evidence_packs
from ouroboros.gateway import task_archive
from ouroboros.task_custody import own_child_drives, store_artifact_view, task_artifact_stores
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_task_file_serving import TASK, _client, _split_child


@contextlib.contextmanager
def _http(data):
    app = _client(data).app
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "isolated HTTP fixture did not start"
        with httpx.Client(base_url=f"http://127.0.0.1:{sock.getsockname()[1]}", timeout=10) as client:
            yield client
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive(), "HTTP fixture did not stop"


def _junction_or_link(link, target):
    if sys.platform == "win32":
        # Directory junctions on the local test volume do not need symlink privilege.
        result = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        link.symlink_to(target, target_is_directory=True)


def _url(name):
    return f"/api/tasks/{TASK}/artifacts/{name}"


def test_real_http_canonical_child_head_range_zip_and_identity(tmp_path):
    data, child, store = _split_child(tmp_path)
    with _http(data) as client:
        nested = client.get(_url("summary.txt"), params={"relpath": "reports/a/summary.txt"})
        assert (nested.status_code, nested.content) == (200, b"alpha")
        assert nested.headers["x-ouroboros-artifact-identity"] == "verified"
        assert nested.headers["x-ouroboros-artifact-sha256"] == sha256(b"alpha").hexdigest()
        head = client.head(_url("summary.txt"), params={"relpath": "reports/a/summary.txt"})
        assert (head.status_code, head.content, head.headers["content-length"]) == (200, b"", "5")
        partial = client.get(_url("summary.txt"), params={"relpath": "reports/a/summary.txt"},
                             headers={"range": "bytes=1-3"})
        assert (partial.status_code, partial.content, partial.headers["content-range"]) == (206, b"lph", "bytes 1-3/5")
        assert client.get(_url("summary.txt")).status_code == 409
        archive = client.get(_url("reports.zip"), params={"archive": "reports"})
        assert archive.status_code == 200, archive.text
        with zipfile.ZipFile(io.BytesIO(archive.content)) as zipped:
            assert zipped.namelist() == ["reports/a/summary.txt", "reports/b/summary.txt"]
            assert zipped.read("reports/b/summary.txt") == b"beta"
        detail = client.get(f"/api/tasks/{TASK}")
        assert detail.json()["artifact_archives"]["reports"]["files"] == 2
        (store / "reports/a/summary.txt").write_bytes(b"other")
        changed = client.get(_url("summary.txt"), params={"relpath": "reports/a/summary.txt"})
        assert (changed.status_code, changed.json()["reason_code"]) == (409, "artifact_identity_changed")
        canonical = artifacts.task_artifact_dir_path(data, TASK, create=True) / "reports/a/summary.txt"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"alpha")
        assert client.get(_url("summary.txt"), params={"relpath": "reports/a/summary.txt"}).content == b"alpha"


def test_real_http_chat_media_and_all_source_branches(tmp_path):
    data = tmp_path / "data"
    media = artifacts.store_chat_media_bytes(data, TASK, b"image bytes", "image/png")
    plain = b'{"observation":"exact"}'
    published = artifacts.store_actor_source_bytes(data, TASK, category="tool_results", source_id="result",
                                                   data=plain, extension="json")
    delegated = artifacts.store_actor_source_bytes(data, TASK, category="delegated_activity", source_id="run",
                                                   data=b'{"event":1}\n', extension="jsonl")
    acceptance_raw = json.dumps({"authority": "host_root", "request": {
        "surface": "task_acceptance", "task_id": TASK}}).encode()
    acceptance = artifacts.store_actor_source_bytes(data, TASK, category="context_checkpoints",
                                                     source_id="acceptance", data=acceptance_raw, extension="json")
    write_task_result(data, TASK, "completed", completion_observations={"source_ref": published})
    with _http(data) as client:
        image = client.get(_url(media["name"]))
        assert (image.status_code, image.content, image.headers["content-type"]) == (200, b"image bytes", "image/png")
        for ref, expected in ((published, plain), (delegated, b'{"event":1}\n'), (acceptance, acceptance_raw)):
            response = client.get(_url(Path(ref["path"]).name), params={"source": ref["path"]})
            assert (response.status_code, response.content) == (200, expected)
        Path(media["path"]).write_bytes(b"tampered")
        assert client.get(_url(media["name"])).status_code == 404


def test_real_http_host_configured_root_junction_is_accepted(tmp_path):
    data = tmp_path / "physical"
    ref = artifacts.store_task_artifact_bytes(data, TASK, "ready.txt", b"owner configured root")
    alias = tmp_path / "configured"
    _junction_or_link(alias, data)
    with _http(alias) as client:
        response = client.get(_url(ref["path"]))
        assert (response.status_code, response.content) == (200, b"owner configured root")


def test_real_http_configured_alias_root_produces_and_serves_recorded_identities(tmp_path):
    """The owner's data root is a link; producers and server both use that spelling."""
    physical = tmp_path / "physical"
    physical.mkdir()
    _junction_or_link(tmp_path / "data", physical)
    data, child, store = _split_child(tmp_path)  # every producer records the configured spelling
    rows = load_task_result(child, TASK)["artifacts"]
    next(row for row in rows if row.get("relpath") == "reports/b/summary.txt")["errors"] = ["copy failed"]
    write_task_result(child, TASK, "completed", artifacts=rows)
    artifacts.store_task_artifact_bytes(data, TASK, "ready.txt", b"configured root")
    (artifacts.task_artifact_dir_path(data, TASK) / "notes.txt").write_bytes(b"canonical capture")
    canonical = [row for row in artifacts.collect_task_artifact_records(data, TASK) if row["name"] == "notes.txt"]
    write_task_result(data, TASK, "running", artifacts=canonical)
    admitted = own_child_drives(data, TASK, store_row=load_task_result(data, TASK))
    assert admitted == [physical.resolve() / child.relative_to(data)]
    with _http(data) as client:
        ready = client.get(_url("ready.txt"))
        assert (ready.status_code, ready.content) == (200, b"configured root")
        assert ready.headers["x-ouroboros-artifact-identity"] == "verified"
        notes = client.get(_url("notes.txt"))
        assert (notes.status_code, notes.headers["x-ouroboros-artifact-identity"]) == (200, "verified")
        detail = client.get(f"/api/tasks/{TASK}").json()
        listed = sorted(row.get("relpath") or row["name"] for row in detail["artifacts"])
        assert listed == ["notes.txt", "ready.txt", "reports/a/summary.txt", "reports/b/summary.txt"], listed
        failed = next(row for row in detail["artifacts"] if row.get("relpath") == "reports/b/summary.txt")
        assert failed["errors"] == ["copy failed"]
        archive = detail["artifact_archives"]["reports"]
        assert (archive["files"], archive["excluded"]) == (1, 1)
        archived = client.get(_url("reports.zip"), params={"archive": "reports"})
        with zipfile.ZipFile(io.BytesIO(archived.content)) as zipped:
            assert zipped.namelist() == ["reports/a/summary.txt"]  # the failed capture stays out
        nested = client.get(_url("summary.txt"), params={"relpath": "reports/a/summary.txt"})
        assert (nested.status_code, nested.headers["x-ouroboros-artifact-identity"]) == (200, "verified")
        (store / "reports/a/summary.txt").write_bytes(b"other")
        changed = client.get(_url("summary.txt"), params={"relpath": "reports/a/summary.txt"})
        assert (changed.status_code, changed.json()["reason_code"]) == (409, "artifact_identity_changed")


def test_real_http_unicode_filename_keeps_exact_bytes(tmp_path):
    data = tmp_path / "data"
    name, raw = "данные-🧪.txt", "точные байты\r\n".encode()
    artifacts.store_task_artifact_bytes(data, TASK, name, raw)
    with _http(data) as client:
        response = client.get(_url(name))
        assert (response.status_code, response.content) == (200, raw)


def test_real_http_packed_source_preserves_placement_index_authority(tmp_path):
    data = tmp_path / "data"
    raw = b'{"complete":"packed checkpoint"}'
    ref = artifacts.store_actor_source_bytes(data, TASK, category="context_checkpoints", source_id="checkpoint",
                                             data=raw, extension="json")
    write_task_result(data, TASK, "completed", completion_observations={"source_ref": ref})
    loose = artifacts.task_artifact_dir_path(data, TASK) / ref["path"]
    pack, sidecar = evidence_packs.write_pack(data, TASK, 1, [{"sha256": ref["sha256"], "size": len(raw),
        "kind": evidence_packs.CHECKPOINT, "path": ref["path"], "source": loose}])
    loose.unlink()
    index = evidence_packs.packs_dir(data) / evidence_packs.INDEX_NAME
    with _http(data) as client:
        response = client.get(_url(Path(ref["path"]).name), params={"source": ref["path"]})
        assert (response.status_code, response.content) == (200, raw)
        assert not index.exists()  # the task's own sidecar answered; the shared index is the packer's to build
        evidence_packs.index_upsert(data, pack, sidecar)  # as the packer does before it unlinks a loose handle
        # A warm placement remains usable if its sidecar is damaged.
        (index.parent / evidence_packs.sidecar_name(TASK, 1)).write_text("{")
        cached = client.get(_url(Path(ref["path"]).name), params={"source": ref["path"]})
        assert (cached.status_code, cached.content) == (200, raw)
        pack.write_bytes(b"broken xz")
        assert client.get(_url(Path(ref["path"]).name), params={"source": ref["path"]}).status_code == 404


@pytest.mark.parametrize("component", ["task_results", "artifacts", "task", "nested"])
def test_real_http_reparse_components_do_not_serve_external_bytes(tmp_path, component):
    data = tmp_path / "data"
    store = artifacts.task_artifact_dir_path(data, TASK, create=True)
    nested = store / "reports"
    nested.mkdir()
    payload = nested / "report.txt"
    payload.write_bytes(b"original")
    write_task_result(data, TASK, "completed", artifacts=artifacts.collect_task_artifact_records(data, TASK))
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = {"task_results": data / "task_results", "artifacts": store.parent,
              "task": store, "nested": nested}[component]
    relative = payload.relative_to(victim)
    planted = outside / relative
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_bytes(b"EXTERNAL-CANARY")
    victim.rename(victim.with_name(victim.name + ".held"))
    _junction_or_link(victim, outside)
    with _http(data) as client:
        response = client.get(_url("report.txt"), params={"relpath": "reports/report.txt"})
        assert response.status_code != 200
        assert b"EXTERNAL-CANARY" not in response.content
        archive = client.get(_url("reports.zip"), params={"archive": "reports"})
        assert archive.status_code != 200
        detail = client.get(f"/api/tasks/{TASK}")
        assert "EXTERNAL-CANARY" not in detail.text


def test_manifest_leaf_link_does_not_supply_registration(tmp_path):
    data = tmp_path / "data"
    store = artifacts.task_artifact_dir_path(data, TASK, create=True)
    outside = tmp_path / "external-manifest.json"
    outside.write_text(json.dumps({"artifacts": {"outside.txt": {
        "name": "outside.txt", "path": str(store / "outside.txt"), "sha256": "a" * 64,
        "immutable": True, "size": 1}}}))
    try:
        (store / ".artifact_manifest.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"native file-symlink fixture unavailable: {exc}")
    assert artifacts.registered_task_artifact(data, TASK, "outside.txt") is None
    assert store_artifact_view(data, TASK, []) == []


def test_real_http_directory_swap_after_member_selection(tmp_path, monkeypatch):
    data, _child, store = _split_child(tmp_path)
    outside = tmp_path / "outside"
    (outside / "a").mkdir(parents=True)
    (outside / "a/summary.txt").write_bytes(b"EXTERNAL-CANARY")
    original = task_archive._archive_members

    def swap(*args, **kwargs):
        members = original(*args, **kwargs)
        (store / "reports").rename(store / "reports.old")
        _junction_or_link(store / "reports", outside)
        return members

    monkeypatch.setattr(task_archive, "_archive_members", swap)
    with _http(data) as client:
        response = client.get(_url("reports.zip"), params={"archive": "reports"})
        assert response.status_code == 404, response.text
        assert b"EXTERNAL-CANARY" not in response.content


def test_selectors_manifests_and_listings_do_not_resolve_descendants(tmp_path, monkeypatch):
    data = tmp_path / "data"
    store = artifacts.task_artifact_dir_path(data, TASK, create=True)
    (store / "ready.txt").write_bytes(b"ready")
    child = data / "task_drives" / TASK
    child.mkdir(parents=True)
    original = Path.resolve

    def only_anchor(path, *args, **kwargs):
        assert path == data, f"pre-open descendant resolve: {path}"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", only_anchor)
    stores = task_artifact_stores(data, TASK)
    assert own_child_drives(data, TASK) == [child]
    assert task_archive.task_artifact_location(stores, store / "ready.txt")[2] == "ready.txt"
    assert task_archive.task_artifact_location(stores, "//invalid-host/share/secret") is None
    assert artifacts.registered_task_artifact(data, TASK, "ready.txt") is None
    assert [row["name"] for row in store_artifact_view(data, TASK, [])] == ["ready.txt"]


def test_windows_selector_parser_portable_logic_only(monkeypatch):
    monkeypatch.setattr(confined_files, "IS_WINDOWS", True)
    for value in ("NUL", "NUL.txt", "COM¹.bin", "x:y", "file.", "file ", "../secret", "a//b", "a\\b",
                  "//host/share", "a/CONOUT$", "a/LPT².txt", "a/../b", "a/./b"):
        assert confined_files.relative_parts(value) == [], value
    assert confined_files.relative_parts("reports/данные.txt") == ["reports", "данные.txt"]
    native_name = "数" * 100 + ".txt"
    assert confined_files.relative_parts(native_name) == [native_name]
    assert confined_files.plain_name(native_name) == ""  # existing upload byte limit
    assert confined_files.relative_parts("🧪" * 128) == []
    # Windows walk entries keep the Win32 naming rules that every path consumer uses.
    for value in ("a\\b", "x:y", "a/b", "", ".", "..", "nul\x00", "数" * 256, "NUL.txt", "file.", "COM¹.bin"):
        assert not confined_files.native_name(value), value
    assert confined_files.native_name(native_name)
    monkeypatch.setattr(confined_files, "IS_WINDOWS", False)
    for value in ("a\\b", "x:y", "NUL", "file. "):  # one POSIX entry each
        assert confined_files.native_name(value), value
    assert confined_files.relative_parts("a\\b") == []  # the HTTP selector still refuses it


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS name longer than POSIX UTF-8 byte limit")
def test_native_windows_long_unicode_filename_and_listing(tmp_path):
    data, name = tmp_path / "data", "数" * 100 + ".txt"
    store = artifacts.task_artifact_dir_path(data, TASK, create=True)
    (store / name).write_bytes(b"native unicode bytes")
    rows = artifacts.collect_task_artifact_records(data, TASK)
    assert [row["name"] for row in rows] == [name]
    write_task_result(data, TASK, "completed", artifacts=rows)
    with _http(data) as client:
        response = client.get(_url(name))
        assert (response.status_code, response.content) == (200, b"native unicode bytes")


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows HTTP selector semantics")
def test_native_windows_device_and_stream_selectors_stop_before_discovery(tmp_path, monkeypatch):
    from ouroboros.gateway import tasks

    def unexpected(*args, **kwargs):
        pytest.fail("invalid Windows selector reached metadata/store discovery")

    monkeypatch.setattr(tasks, "task_artifact_stores", unexpected)
    with _http(tmp_path / "data") as client:
        for name in ("NUL", "NUL.txt", "COM¹.txt", "a:stream", "CONOUT$", "file."):
            assert client.get(_url(name)).status_code == 400, name
        for source in ("a/COM1.txt", "a/value:stream", "//host/share/secret", "a/../secret"):
            assert client.get(_url("secret"), params={"source": source}).status_code == 404, source


def test_confined_native_directory_and_regular_file_handles(tmp_path):
    root = tmp_path / "root"
    (root / "a/b").mkdir(parents=True)
    (root / "a/b/value").write_bytes(b"held bytes")
    with confined_files.confined_directory(root, ["a", "b"]) as directory:
        assert directory.entries() == [("value", False)]
        handle, observed = directory.open("value")
        with handle:
            assert handle.read() == b"held bytes"
            assert observed.st_size == 10
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "value").write_bytes(b"EXTERNAL-CANARY")
    _junction_or_link(root / "link", outside)
    with pytest.raises(OSError):
        confined_files.open_confined_file(root, ["link", "value"])


def test_held_parent_rename_cannot_redirect_the_read(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    (root / "parent").mkdir(parents=True)
    outside.mkdir()
    (root / "parent/value").write_bytes(b"held bytes")
    (outside / "value").write_bytes(b"EXTERNAL-CANARY")
    with confined_files.confined_directory(root, ["parent"]) as directory:
        # Like a POSIX dirfd, a held walk directory shares rename on every platform;
        # the held object, not the name it had, stays the authority for its opens.
        (root / "parent").rename(root / "moved")
        _junction_or_link(root / "parent", outside)
        handle, _ = directory.open("value")
        with handle:
            assert handle.read() == b"held bytes"
        assert directory.entries() == [("value", False)]


def test_held_walk_directories_do_not_block_removal_and_release_cleanly(tmp_path):
    root = tmp_path / "root"
    (root / "store/sub").mkdir(parents=True)
    (root / "store/sub/value").write_bytes(b"held bytes")
    with confined_files.confined_directory(root, ["store", "sub"]) as directory:
        assert directory.entries() == [("value", False)]
        # Another reader or the reaper removes what this walk holds; the hold refuses nothing.
        (root / "store/sub/value").unlink()
        (root / "store/sub").rmdir()
        with pytest.raises(OSError):
            directory.open("value")  # a removed directory addresses nothing new
    shutil.rmtree(root / "store")  # released: nothing of the walk is still held
    assert not (root / "store").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX entry names and special files")
def test_posix_native_names_are_captured_and_special_entries_skipped(tmp_path, monkeypatch):
    from ouroboros.task_finalization import rescued_files_fact

    data = tmp_path / "data"
    store = artifacts.task_artifact_dir_path(data, TASK, create=True)
    (store / "deep/a\\b").mkdir(parents=True)
    (store / "win\\style.txt").write_bytes(b"one POSIX name")
    (store / "deep/a\\b/x.txt").write_bytes(b"nested")
    os.mkfifo(store / "pipe")
    os.mkfifo(store / "deep/pipe")
    monkeypatch.chdir(store)  # a socket path must stay short
    with socket.socket(socket.AF_UNIX) as server:
        server.bind("sock")
        expected = ["deep/a\\b/x.txt", "win\\style.txt"]
        for measure in (True, False):
            rows = artifacts.collect_task_artifact_records(data, TASK, measure=measure, strict=True)
            assert sorted(row.get("relpath") or row["name"] for row in rows) == expected, measure
        fact = rescued_files_fact(TASK, [store])
        assert (fact["state"], fact["count"]) == ("positive", 2)
        captured = artifacts.collect_task_artifact_records(data, TASK)
        write_task_result(data, TASK, "completed", artifacts=captured)
        view = store_artifact_view(data, TASK, captured)
        assert sorted(row.get("relpath") or row["name"] for row in view) == expected
        assert all(row.get("sha256") for row in view), "a recorded identity is kept, never re-listed"
        assert confined_files.relative_parts("win\\style.txt") == []  # HTTP selectors are unchanged
        with _http(data) as client:
            assert client.get(_url("win%5Cstyle.txt")).status_code == 400
        locked = store / "deep/locked"
        locked.mkdir()
        locked.chmod(0)
        try:
            if os.geteuid() != 0:  # a real I/O failure still keeps the strict listing from vouching
                with pytest.raises(PermissionError):
                    artifacts.collect_task_artifact_records(data, TASK, measure=False, strict=True)
                assert rescued_files_fact(TASK, [store])["state"] == "unknown"
        finally:
            locked.chmod(0o700)
