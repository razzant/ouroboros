"""TZ-1 V12: a task's recorded files leave through one confined descriptor - the exact nested
file (``?relpath=``), or one recorded directory as a ZIP (``?archive=``) - read from the task's
OWN stores only (canonical, then its own child drive's), with the detail's
``artifact_archives`` projection. A bare name never picks a nested file; forged rows, escaping
symlinks and components swapped after attribution are refused or excluded, never followed; a
captured file serves only the bytes it verified; a platform without directory-relative
no-follow opens answers a typed 503 (owner decision, issue #1297)."""

from __future__ import annotations

import errno
import io
import os
import stat
import threading
import zipfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import artifacts, headless
from ouroboros.gateway import task_archive
from ouroboros.task_custody import task_artifact_stores
from ouroboros.task_results import write_task_result

TASK = "childpub"
URL = f"/api/tasks/{TASK}/artifacts/reports.zip"
SECRET = "outside secret"
confined = pytest.mark.skipif(not task_archive.CONFINED, reason="needs directory-relative no-follow opens")


def _split_child(tmp_path, *, status="completed"):
    """A canonical subagent row plus its own headless child drive holding two nested
    deliverables that share one basename."""
    data = tmp_path / "data"
    child = headless.prepare_task_drive(data, TASK, "empty")
    store = artifacts.task_artifact_dir_path(child, TASK, create=True)
    for sub, text in (("a", "alpha"), ("b", "beta")):
        (store / "reports" / sub).mkdir(parents=True)
        (store / "reports" / sub / "summary.txt").write_text(text, encoding="utf-8")
    write_task_result(child, TASK, status, result="child done", artifacts=artifacts.collect_task_artifact_records(child, TASK),
                      artifact_status="ready")
    write_task_result(data, TASK, "running", child_drive_root=str(child), delegation_role="subagent",
                      parent_task_id="parent1", root_task_id="parent1")
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    return data, child, store


def _client(data):
    from ouroboros.gateway.tasks import api_task_artifact, api_task_get

    app = Starlette(routes=[
        Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"]),
        Route("/api/tasks/{task_id}/artifacts/{name}", endpoint=api_task_artifact, methods=["GET", "HEAD"]),
    ])
    app.state.drive_root = data
    return TestClient(app)


def _zip(response):
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert int(response.headers["content-length"]) == len(response.content)
    return zipfile.ZipFile(io.BytesIO(response.content))


def _link_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


def _spy_spools(monkeypatch):
    spools, real = [], task_archive.tempfile.TemporaryFile

    def spy(*args, **kwargs):
        spool = real(*args, **kwargs)
        spools.append(spool)
        return spool
    monkeypatch.setattr(task_archive.tempfile, "TemporaryFile", spy)
    return spools


def _swap_after_members(monkeypatch, swap):
    """The attacker's move right after the members were chosen from their stats and before the
    first open: the window a path-based open would lose."""
    real = task_archive._archive_members

    def raced(*args, **kwargs):
        members = real(*args, **kwargs)
        swap()
        return members
    monkeypatch.setattr(task_archive, "_archive_members", raced)


@confined
def test_exact_nested_selection_reads_both_own_stores_read_only(tmp_path):
    data, child, store = _split_child(tmp_path)
    client = _client(data)
    url = f"/api/tasks/{TASK}/artifacts/summary.txt"

    ambiguous = client.get(url)
    assert ambiguous.status_code == 409
    assert ambiguous.json() == {
        "error": "artifact name matches several nested files; select one with ?relpath=",
        "reason_code": "artifact_name_ambiguous", "task_id": TASK, "artifact": "summary.txt",
        "relpaths": ["reports/a/summary.txt", "reports/b/summary.txt"]}
    exact = client.get(url, params={"relpath": "reports/b/summary.txt"})
    assert (exact.status_code, exact.text) == (200, "beta")
    (store / "reports/b/summary.txt").unlink()
    write_task_result(child, TASK, "completed", artifacts=artifacts.collect_task_artifact_records(child, TASK))
    assert client.get(url).status_code == 404  # a sole nested basename is no bare-name route
    for bad in ("../a/summary.txt", "reports//summary.txt", "/reports/a/summary.txt", "reports/./summary.txt",
                "reports/a/other.txt", "reports\\a\\summary.txt", ""):
        refused = client.get(url, params={"relpath": bad})
        assert (refused.status_code, refused.json()["reason_code"]) == (400, "artifact_relpath_invalid"), bad
    assert client.get(url, params={"relpath": "reports/a/summary.txt", "source": "x"}).status_code == 400
    assert client.get(url, params={"relpath": "reports/c/summary.txt"}).status_code == 404
    (store / "summary.txt").write_text("top", encoding="utf-8")
    write_task_result(child, TASK, "completed", artifacts=artifacts.collect_task_artifact_records(child, TASK))
    assert (client.get(url).status_code, client.get(url).text) == (200, "top")
    detail = client.get(f"/api/tasks/{TASK}").json()
    assert sorted(row.get("relpath", row["name"]) for row in detail["artifacts"]) == [
        "reports/a/summary.txt", "summary.txt"]
    assert not artifacts.task_artifact_dir_path(data, TASK).exists()  # nothing created or copied
    headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(child)})
    (store / "reports/a/summary.txt").unlink()
    after = client.get(url, params={"relpath": "reports/a/summary.txt"})
    assert (after.status_code, after.text) == (200, "alpha")  # the canonical copy at its relpath


@confined
def test_forged_rows_and_escaping_links_authorize_no_read(tmp_path):
    data = tmp_path / "data"
    sibling = headless.prepare_task_drive(data, "sibling1", "empty")
    planted = artifacts.task_artifact_dir_path(sibling, TASK, create=True) / "secret.txt"
    planted.write_text("sibling secret", encoding="utf-8")
    write_task_result(data, TASK, "completed", child_drive_root=str(sibling), drive_root=str(sibling),
                      artifacts=[{"name": "secret.txt", "path": str(planted)}],
                      metadata={"drive_root": str(sibling), "child_drive_root": str(sibling)})
    client = _client(data)
    assert task_artifact_stores(data, TASK) == [artifacts.task_artifact_dir_path(data.resolve(), TASK)]
    refused = client.get(f"/api/tasks/{TASK}/artifacts/secret.txt")
    assert refused.status_code == 500 and "sibling" not in refused.text
    assert client.get(f"/api/tasks/{TASK}/artifacts/secret.txt", params={"relpath": "secret.txt"}).status_code == 404

    data2, child, store = _split_child(tmp_path / "two")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(SECRET, encoding="utf-8")
    _link_or_skip(store / "link.txt", outside / "secret.txt")
    write_task_result(child, TASK, "completed", artifacts=[{"name": "link.txt", "path": str(store / "link.txt")}])
    client2 = _client(data2)
    escaped = client2.get(f"/api/tasks/{TASK}/artifacts/link.txt")  # a child row escaping its store is not listed
    assert escaped.status_code == 404 and SECRET not in escaped.text
    assert client2.get(f"/api/tasks/{TASK}/artifacts/link.txt", params={"relpath": "link.txt"}).status_code == 404


@confined
def test_an_immutable_download_serves_only_its_verified_bytes(tmp_path, monkeypatch):
    data = tmp_path / "data"
    source = tmp_path / "report.txt"
    source.write_text("captured", encoding="utf-8")
    record = artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=data, task_id=TASK), source, immutable=True)
    write_task_result(data, TASK, "completed", artifacts=[record])
    client = _client(data)
    url = f"/api/tasks/{TASK}/artifacts/{record['name']}"
    real = artifacts.stream_artifact_file

    def swap_after_verify(*args, **kwargs):
        measured = real(*args, **kwargs)
        Path(record["path"]).write_text("swapped!", encoding="utf-8")  # after verification
        return measured
    monkeypatch.setattr(task_archive.artifact_store, "stream_artifact_file", swap_after_verify)
    assert client.get(url).text == "captured"
    monkeypatch.undo()
    refused = client.get(url)  # the pathname now holds other bytes: nothing unverified leaves
    assert (refused.status_code, refused.json()["reason_code"]) == (404, "artifact_unverified")
    assert "swapped" not in refused.text


@confined
def test_the_descriptor_response_answers_head_and_ranges_from_one_open(tmp_path, monkeypatch):
    data, child, store = _split_child(tmp_path)
    (store / "blob.bin").write_bytes(b"0123456789")
    write_task_result(child, TASK, "completed", artifacts=artifacts.collect_task_artifact_records(child, TASK))
    client = _client(data)
    url = f"/api/tasks/{TASK}/artifacts/blob.bin"

    full = client.get(url)
    assert full.content == b"0123456789" and full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-length"] == "10" and full.headers["etag"] and full.headers["last-modified"]
    head = client.head(url)
    assert head.status_code == 200 and head.content == b"" and head.headers["content-length"] == "10"
    part = client.get(url, headers={"range": "bytes=2-4"})
    assert (part.status_code, part.content, part.headers["content-range"]) == (206, b"234", "bytes 2-4/10")
    suffix = client.get(url, headers={"range": "bytes=-3"})
    assert (suffix.status_code, suffix.content) == (206, b"789")
    assert client.get(url, headers={"range": "bytes=20-30"}).status_code == 416
    assert client.get(url, headers={"range": "bytes=0-1,4-5"}).content == b"0123456789"  # several: served whole


@confined
def test_chat_media_serves_only_bytes_that_hash_to_its_name(tmp_path):
    data = tmp_path / "data"
    payload = b"\x89PNG fake image"
    name = f"chat-media-{sha256(payload).hexdigest()}.png"
    media = artifacts.task_artifact_dir_path(data, TASK, create=True) / "chat_media" / name
    media.parent.mkdir(parents=True)
    media.write_bytes(payload)
    client = _client(data)
    assert client.get(f"/api/tasks/{TASK}/artifacts/{name}").content == payload
    media.write_bytes(b"tampered")
    assert client.get(f"/api/tasks/{TASK}/artifacts/{name}").status_code == 404


@confined
def test_an_svg_task_file_keeps_bytes_and_view_but_gets_no_script_or_origin(tmp_path):
    """A generated SVG reaches the artifact list and send_file's card Open as a registered
    task file (chat media is raster/video only). Exact bytes, image type and Range reads
    stay; the chat-upload inline policy (sandbox CSP, nosniff) withholds script and this
    origin. Raster headers are unchanged."""
    data = tmp_path / "data"
    svg = (b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8"/>'
           b'<script>fetch("/api/state")</script></svg>')
    png = b"\x89PNG\r\n\x1a\n raster"
    media = artifacts.task_artifact_dir_path(data, TASK, create=True) / "chat_media"
    media.mkdir()
    raster_name = f"chat-media-{sha256(png).hexdigest()}.png"
    (media / raster_name).write_bytes(png)
    generated = artifacts.store_task_artifact_bytes(data, TASK, f"image-{sha256(svg).hexdigest()}.svg", svg,
                                                    kind="generated_image")
    client = _client(data)
    url = f"/api/tasks/{TASK}/artifacts/{generated['path']}"
    opened = client.get(url)
    assert opened.content == svg and opened.headers["content-type"].startswith("image/svg+xml")
    assert opened.headers["content-security-policy"] == "sandbox"
    assert opened.headers["x-content-type-options"] == "nosniff"
    part = client.get(url, headers={"range": "bytes=0-3"})
    assert (part.status_code, part.content, part.headers["content-security-policy"]) == (206, b"<svg", "sandbox")
    raster = client.get(f"/api/tasks/{TASK}/artifacts/{raster_name}")
    assert raster.content == png and raster.headers["content-type"] == "image/png"
    assert "content-security-policy" not in raster.headers and "x-content-type-options" not in raster.headers


def test_without_confined_opens_files_and_archives_fail_closed(tmp_path, monkeypatch):
    data, child, _store = _split_child(tmp_path)
    client = _client(data)
    monkeypatch.setattr(task_archive, "CONFINED", False)

    assert client.get(f"/api/tasks/{TASK}").json()["artifact_archives"] == {
        "reports": {"name": "reports.zip", "files": 0, "size": 0, "excluded": 2, "available": False}}
    refused = client.get(URL, params={"archive": "reports"})
    assert (refused.status_code, refused.json()["reason_code"]) == (503, "artifact_archive_unavailable")
    plain = client.get(f"/api/tasks/{TASK}/artifacts/summary.txt", params={"relpath": "reports/a/summary.txt"})
    assert (plain.status_code, plain.json()["reason_code"]) == (503, "artifact_unavailable")
    assert client.get(URL, params={"archive": "../reports"}).status_code == 400  # refusals come first


@confined
def test_directory_archive_keeps_relative_paths_one_member_per_relpath_canonical_first(tmp_path, monkeypatch):
    data, child, store = _split_child(tmp_path)
    client = _client(data)

    assert client.get(f"/api/tasks/{TASK}").json()["artifact_archives"] == {
        "reports": {"name": "reports.zip", "files": 2, "size": 9, "excluded": 0, "available": True}}
    response = client.get(URL, params={"archive": "reports"})
    archive = _zip(response)
    assert response.headers["content-disposition"] == 'attachment; filename="reports.zip"'
    assert archive.namelist() == ["reports/a/summary.txt", "reports/b/summary.txt"]
    assert archive.read("reports/b/summary.txt") == b"beta"
    assert _zip(client.get(f"/api/tasks/{TASK}/artifacts/a.zip", params={"archive": "reports/a"})).namelist() == [
        "a/summary.txt"]
    canonical = artifacts.task_artifact_dir_path(data, TASK, create=True)
    (canonical / "reports/a").mkdir(parents=True)
    (canonical / "reports/a/summary.txt").write_text("alpha", encoding="utf-8")  # a relocated copy: same bytes
    opened = []
    real_open = task_archive._open_member
    monkeypatch.setattr(task_archive, "_open_member", lambda parents, route: opened.append(route) or real_open(parents, route))
    merged = _zip(client.get(URL, params={"archive": "reports"}))
    assert merged.namelist() == ["reports/a/summary.txt", "reports/b/summary.txt"]
    assert merged.read("reports/a/summary.txt") == b"alpha" and opened[0][1][0] == "task_results"  # canonical first
    (canonical / "reports/a/summary.txt").write_text("alpha canonical", encoding="utf-8")  # other bytes than the record
    refused = client.get(URL, params={"archive": "reports"})
    assert (refused.status_code, refused.json()["reason_code"], refused.json()["member"]) == (
        404, "artifact_archive_unverified", "reports/a/summary.txt")  # never new bytes under the recorded identity
    (canonical / ".github").mkdir()
    (canonical / ".github/ci.yml").write_text("name: CI", encoding="utf-8")
    dotted = _zip(client.get(f"/api/tasks/{TASK}/artifacts/.github.zip", params={"archive": ".github"}))
    assert dotted.namelist() == [".github/ci.yml"]


@confined
def test_directory_archive_excludes_and_counts_rows_it_cannot_serve(tmp_path):
    data, child, store = _split_child(tmp_path)
    (store / "reports/b/summary.txt").unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(SECRET, encoding="utf-8")
    _link_or_skip(store / "reports/link.txt", outside / "secret.txt")
    for name in ("failed.txt", "errored.txt"):
        (store / "reports" / name).write_text(name, encoding="utf-8")
    def ready(path, **extra):
        return {"name": path.name, "path": str(path), "status": "ready", "errors": [], **extra}
    rows = [ready(store / "reports/a/summary.txt"), ready(store / "reports/link.txt", relpath="reports/link.txt"),
            ready(store / "reports/gone.txt"), {**ready(store / "reports/failed.txt"), "status": "failed"},
            {**ready(store / "reports/errored.txt"), "errors": ["copy failed"]}]
    write_task_result(child, TASK, "completed", artifacts=rows)
    client = _client(data)

    # The escaping link row never enters the view; gone, failed and errored rows are counted out.
    assert client.get(f"/api/tasks/{TASK}").json()["artifact_archives"] == {
        "reports": {"name": "reports.zip", "files": 1, "size": 5, "excluded": 3, "available": True}}
    archive = _zip(client.get(URL, params={"archive": "reports"}))
    assert archive.namelist() == ["reports/a/summary.txt"]


@confined
def test_directory_archive_verifies_captures_and_types_every_refusal(tmp_path, monkeypatch):
    data, child, store = _split_child(tmp_path)
    rows = [artifacts.artifact_record(store / "reports/a/summary.txt"),
            {**artifacts.artifact_record(store / "reports/b/summary.txt"), "immutable": True}]
    write_task_result(child, TASK, "completed", artifacts=rows)
    client = _client(data)
    spools = _spy_spools(monkeypatch)
    (store / "reports/a/summary.txt").write_text("alpha v2", encoding="utf-8")  # a mutable row, not re-recorded
    stale = client.get(URL, params={"archive": "reports"})
    assert (stale.status_code, stale.json()["member"]) == (404, "reports/a/summary.txt")
    rows[0] = artifacts.artifact_record(store / "reports/a/summary.txt")  # re-recorded: the new identity serves
    write_task_result(child, TASK, "completed", artifacts=rows)
    assert _zip(client.get(URL, params={"archive": "reports"})).read("reports/a/summary.txt") == b"alpha v2"
    (store / "reports/b/summary.txt").write_text("beta v2", encoding="utf-8")
    refused = client.get(URL, params={"archive": "reports"})
    assert refused.json() == {"error": "archive member is missing, changed while read, or failed its capture verification",
                              "reason_code": "artifact_archive_unverified", "task_id": TASK,
                              "artifact": "reports.zip", "directory": "reports", "member": "reports/b/summary.txt"}
    assert len(spools) == 3 and all(spool.closed for spool in spools)
    for name, params in (("reports.zip", {"archive": "reports", "relpath": "reports.zip"}),
                         ("reports.zip", {"archive": "../reports"}), ("reports.zip", {"archive": ""}),
                         ("other.zip", {"archive": "reports"}), ("reports", {"archive": "reports"})):
        bad = client.get(f"/api/tasks/{TASK}/artifacts/{name}", params=params)
        assert (bad.status_code, bad.json()["reason_code"]) == (400, "artifact_archive_invalid"), params

    def no_spool(*_args, **_kwargs):
        raise OSError(errno.EMFILE, "too many open files")
    monkeypatch.setattr(task_archive.tempfile, "TemporaryFile", no_spool)
    failed = client.get(URL, params={"archive": "reports"})
    assert (failed.status_code, failed.json()["reason_code"]) == (503, "artifact_archive_unavailable")


@confined
def test_a_directory_swapped_for_a_symlink_after_the_stat_is_refused_never_followed(tmp_path, monkeypatch):
    data, child, store = _split_child(tmp_path)
    client = _client(data)
    outside = tmp_path / "outside" / "a"
    outside.mkdir(parents=True)
    (outside / "summary.txt").write_text(SECRET, encoding="utf-8")

    def swap():
        (store / "reports/a").rename(tmp_path / "moved-a")
        _link_or_skip(store / "reports/a", outside, directory=True)
    _swap_after_members(monkeypatch, swap)
    refused = client.get(URL, params={"archive": "reports"})
    assert (refused.status_code, refused.json()["reason_code"]) == (404, "artifact_archive_unverified")
    assert SECRET not in refused.text


@pytest.mark.skipif(not task_archive.CONFINED or not hasattr(os, "mkfifo"), reason="confined opens and FIFOs required")
def test_a_member_swapped_for_a_fifo_is_refused_without_blocking(tmp_path, monkeypatch):
    data, child, store = _split_child(tmp_path)
    client = _client(data)
    target = store / "reports/a/summary.txt"

    def swap():
        target.unlink()
        os.mkfifo(target)
    _swap_after_members(monkeypatch, swap)
    answer = []
    worker = threading.Thread(target=lambda: answer.append(client.get(URL, params={"archive": "reports"})), daemon=True)
    worker.start()
    worker.join(timeout=15)
    assert not worker.is_alive() and answer, "opening the FIFO blocked the archive build"
    assert (answer[0].status_code, answer[0].json()["reason_code"]) == (404, "artifact_archive_unverified")
    assert stat.S_ISFIFO(os.lstat(target).st_mode)


@pytest.mark.serial
@confined
def test_real_http_consumer_downloads_a_nested_file_and_a_directory_zip_off_the_loop(tmp_path):
    """The real server path (uvicorn + the HTTP client) for the two V12 addresses the UI builds."""
    import asyncio
    import socket
    import time

    import httpx
    import uvicorn

    data, child, _store = _split_child(tmp_path)
    app = _client(data).app
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        end = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < end:
            time.sleep(0.01)
        assert server.started
        base = f"http://127.0.0.1:{sock.getsockname()[1]}/api/tasks/{TASK}"

        async def fetch():
            async with httpx.AsyncClient(timeout=10) as client:
                nested = await client.get(f"{base}/artifacts/summary.txt", params={"relpath": "reports/a/summary.txt"})
                archive = await client.get(f"{base}/artifacts/reports.zip", params={"archive": "reports"})
                return nested, archive
        nested, archive = asyncio.run(fetch())
        assert (nested.status_code, nested.text) == (200, "alpha")
        assert zipfile.ZipFile(io.BytesIO(archive.content)).namelist() == ["reports/a/summary.txt",
                                                                            "reports/b/summary.txt"]
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()


@confined
@pytest.mark.parametrize("immutable", [False, True])
def test_an_empty_file_still_completes_the_response(tmp_path, immutable):
    """F8: a zero-byte file (mutable, or a verified capture through the spool) answers GET
    with a complete empty body, HEAD with its length, and every range as unsatisfiable."""
    data, child, store = _split_child(tmp_path)
    (store / "empty.bin").write_bytes(b"")
    row = artifacts.artifact_record(store / "empty.bin")
    write_task_result(child, TASK, "completed", artifacts=[{**row, "immutable": True} if immutable else row])
    client = _client(data)
    url = f"/api/tasks/{TASK}/artifacts/empty.bin"

    full = client.get(url)
    assert (full.status_code, full.content, full.headers["content-length"]) == (200, b"", "0")
    head = client.head(url)
    assert (head.status_code, head.content, head.headers["content-length"]) == (200, b"", "0")
    assert client.get(url, headers={"range": "bytes=0-0"}).status_code == 416
    assert client.get(url, headers={"range": "bytes=-1"}).status_code == 416


@confined
def test_a_failed_spool_allocation_closes_the_member_it_opened(tmp_path, monkeypatch):
    """A verified download opens the member first; when the private spool cannot be
    allocated the open descriptor is closed with the typed refusal, never leaked."""
    data, child, store = _split_child(tmp_path)
    write_task_result(child, TASK, "completed",
                      artifacts=[{**artifacts.artifact_record(store / "reports/a/summary.txt"), "immutable": True}])
    client = _client(data)
    handles = []
    real_open = task_archive._open_member

    def spy_open(*args, **kwargs):
        handle, observed = real_open(*args, **kwargs)
        handles.append(handle)
        return handle, observed

    def no_spool(*_args, **_kwargs):
        raise OSError(errno.EMFILE, "too many open files")

    monkeypatch.setattr(task_archive, "_open_member", spy_open)
    monkeypatch.setattr(task_archive.tempfile, "TemporaryFile", no_spool)
    refused = client.get(f"/api/tasks/{TASK}/artifacts/summary.txt", params={"relpath": "reports/a/summary.txt"})
    assert (refused.status_code, refused.json()["reason_code"]) == (503, "artifact_unavailable")
    assert len(handles) == 1 and handles[0].closed


@confined
def test_a_store_reached_through_a_link_serves_and_lists_nothing(tmp_path):
    """Path safety: a store or a parent component swapped for a symlink belongs to no
    store: the view lists nothing through it and no byte leaves through it."""
    data, child, store = _split_child(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "summary.txt").write_text(SECRET, encoding="utf-8")
    (store / "reports/a").rename(tmp_path / "moved-a")
    _link_or_skip(store / "reports/a", outside, directory=True)
    client = _client(data)
    detail = client.get(f"/api/tasks/{TASK}").json()
    assert "reports/a/summary.txt" not in {row.get("relpath") for row in detail["artifacts"]}
    refused = client.get(f"/api/tasks/{TASK}/artifacts/summary.txt", params={"relpath": "reports/a/summary.txt"})
    assert refused.status_code == 404 and SECRET not in refused.text
    linked_store = data / "task_results" / "artifacts"
    linked_store.mkdir(parents=True, exist_ok=True)
    _link_or_skip(linked_store / TASK, outside, directory=True)
    refused = client.get(f"/api/tasks/{TASK}/artifacts/summary.txt")
    assert refused.status_code in {404, 409} and SECRET not in refused.text


@confined
def test_a_measured_files_changed_bytes_are_refused_with_the_recorded_digest_and_unmeasured_bytes_are_labelled(tmp_path):
    """A row that records a digest (immutable or not) is served only when its bytes still match
    it: a same-length replacement answers 409 ``artifact_identity_changed`` naming the recorded
    digest, never the new bytes under the old identity. A listing without a digest streams its
    current bytes and says so (``x-ouroboros-artifact-identity: unmeasured``). A directory ZIP
    applies the same rule to each member."""
    data, child, store = _split_child(tmp_path)
    client = _client(data)
    url = f"/api/tasks/{TASK}/artifacts/summary.txt"
    (store / "reports/a/summary.txt").write_text("ALPHA", encoding="utf-8")  # same length as "alpha"

    refused = client.get(url, params={"relpath": "reports/a/summary.txt"})
    assert refused.status_code == 409, refused.text
    body = refused.json()
    assert body["reason_code"] == "artifact_identity_changed" and body["task_id"] == TASK
    assert body["recorded_sha256"] == sha256(b"alpha").hexdigest() and body["recorded_size"] == 5
    assert "ALPHA" not in refused.text

    served = client.get(url, params={"relpath": "reports/b/summary.txt"})
    assert served.text == "beta" and served.headers["x-ouroboros-artifact-identity"] == "verified"
    assert served.headers["x-ouroboros-artifact-sha256"] == sha256(b"beta").hexdigest()

    (store / "reports" / "c").mkdir()
    (store / "reports/c/summary.txt").write_text("gamma", encoding="utf-8")  # nobody recorded it
    current = client.get(url, params={"relpath": "reports/c/summary.txt"})
    assert current.text == "gamma" and current.headers["x-ouroboros-artifact-identity"] == "unmeasured"
    assert "x-ouroboros-artifact-sha256" not in current.headers

    archive = client.get(URL, params={"archive": "reports"})
    assert (archive.status_code, archive.json()["reason_code"]) == (404, "artifact_archive_unverified")
    assert archive.json()["member"] == "reports/a/summary.txt"


@confined
def test_a_stale_accepted_disposition_cannot_certify_bytes_the_store_no_longer_serves(tmp_path):
    """The parent's disposition hash covers recorded identities and a read stays pure (no hash
    on read), so an unrecorded byte change leaves it accepted; the store makes that honest by
    refusing to serve the changed bytes under the recorded identity, so the disposition never
    certifies bytes a consumer can obtain."""
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256, _current_child_result_disposition
    from ouroboros.tools.task_tree import _tree_note

    data = tmp_path / "data"
    artifact_dir = artifacts.task_artifact_dir_path(data, TASK, create=True)
    report = artifact_dir / "report.md"
    report.write_bytes(b"version one\n")  # bytes: the recorded identity below is of exactly these
    write_task_result(data, TASK, "completed", parent_task_id="parent1", root_task_id="parent1",
                      delegation_role="subagent", result="artifact-backed", artifacts=[artifacts.artifact_record(report)])
    parent = SimpleNamespace(drive_root=str(data), budget_drive_root=str(data), task_id="parent1", role="orchestrator",
                             task_metadata={"budget_drive_root": str(data), "root_task_id": "parent1"})
    shown = _child_result_sha256(load_effective_task_result(data, TASK))
    payload = {"type": "child_result_disposition", "child_task_id": TASK, "disposition": "integrated",
               "child_result_sha256": shown}
    assert _tree_note(parent, "decision", "integrated the report", payload=payload).startswith("OK:")

    report.write_bytes(b"version two\n")  # same length, no re-record
    view = load_effective_task_result(data, TASK)
    assert _child_result_sha256(view) == shown and _current_child_result_disposition(view) == "integrated"
    refused = _client(data).get(f"/api/tasks/{TASK}/artifacts/report.md")
    assert refused.status_code == 409 and refused.json()["reason_code"] == "artifact_identity_changed"
    assert refused.json()["recorded_sha256"] == sha256(b"version one\n").hexdigest()
    assert "version two" not in refused.text
