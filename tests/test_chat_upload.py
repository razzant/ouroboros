"""Tests for the /api/chat/upload endpoint."""
import io
import pathlib
import pytest
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Route
import sys

# Ensure repo root is on path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


@pytest.fixture
def client(tmp_path, monkeypatch):
    import ouroboros.gateway.files as upload_api
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    app = Starlette(routes=[
        Route("/api/chat/upload", endpoint=upload_api.api_chat_upload, methods=["POST"]),
        Route("/api/chat/upload", endpoint=upload_api.api_chat_upload_delete, methods=["DELETE"]),
    ])
    with TestClient(app) as c:
        yield c


def test_upload_success(client, tmp_path):
    data = b"hello world"
    resp = client.post("/api/chat/upload", files={"file": ("test.txt", io.BytesIO(data), "text/plain")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    # stored filename is unique: full UUID hex (32 chars) prefix + underscore + safe_base
    fname = body["filename"]
    assert fname.endswith("_test.txt")
    prefix = fname[: fname.index("_test.txt")]
    assert len(prefix) == 32, f"Expected 32-char UUID hex prefix, got {len(prefix)}: {prefix!r}"
    assert all(c in "0123456789abcdef" for c in prefix), "UUID prefix must be lowercase hex"
    assert body["display_name"] == "test.txt"
    assert body["size"] == len(data)
    dest = tmp_path / "uploads" / body["filename"]
    assert dest.exists()
    assert dest.read_bytes() == data


def test_upload_missing_file(client):
    resp = client.post("/api/chat/upload", data={})
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


def test_upload_same_name_twice_succeeds(client, tmp_path):
    """Same display name can be uploaded multiple times — each gets a unique stored name."""
    data = b"x"
    r1 = client.post("/api/chat/upload", files={"file": ("dup.txt", io.BytesIO(data), "text/plain")})
    r2 = client.post("/api/chat/upload", files={"file": ("dup.txt", io.BytesIO(data), "text/plain")})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["filename"] != r2.json()["filename"]


def test_upload_filename_sanitization(client, tmp_path):
    """Path traversal attempt should be neutralized."""
    data = b"evil"
    resp = client.post("/api/chat/upload", files={"file": ("../../evil.txt", io.BytesIO(data), "text/plain")})
    assert resp.status_code == 200
    body = resp.json()
    # basename strips directory traversal; stored name has uuid prefix
    assert "/" not in body["filename"]
    assert ".." not in body["filename"]
    assert body["display_name"] == "evil.txt"
    dest = tmp_path / "uploads" / body["filename"]
    assert dest.exists()


def test_upload_spaces_in_filename(client, tmp_path):
    data = b"content"
    resp = client.post("/api/chat/upload", files={"file": ("my file name.txt", io.BytesIO(data), "text/plain")})
    assert resp.status_code == 200
    assert " " not in resp.json()["filename"]
    assert " " not in resp.json()["display_name"]


@pytest.mark.parametrize("name", ["a" * 196 + ".txt", "ж" * 100 + ".txt"], ids=["ascii200", "utf8"])
def test_valid_long_upload_names_keep_exact_bytes_on_both_ingresses(client, tmp_path, name):
    from hashlib import sha256
    from typing import get_type_hints
    from ouroboros.gateway.contracts import UploadResponse
    from ouroboros.gateway.files import store_chat_upload

    payload = b"complete uploaded content"
    source = tmp_path / name
    source.write_bytes(payload)
    host_copy = store_chat_upload(source, data_dir=tmp_path / "host")
    assert host_copy.name.endswith("_" + name)
    assert host_copy.read_bytes() == payload
    response = client.post("/api/chat/upload", files={"file": (name, io.BytesIO(payload), "text/plain")})
    assert response.status_code == 200
    body = response.json()
    assert body["display_name"] == name
    assert body["sha256"] == sha256(payload).hexdigest()
    assert set(body) == set(get_type_hints(UploadResponse))
    assert (tmp_path / "uploads" / body["filename"]).read_bytes() == payload
    assert not list((tmp_path / "uploads").glob(".*.tmp"))


def test_upload_invalid_content_length(client):
    """Non-numeric Content-Length should not cause a 500; treated as 0 (unknown)."""
    import io
    data = b"hello"
    resp = client.post(
        "/api/chat/upload",
        files={"file": ("cl_test.txt", io.BytesIO(data), "text/plain")},
        headers={"content-length": "abc"},
    )
    # Should succeed (or fail with a data error), not crash with 500
    assert resp.status_code in (200, 400)


def test_upload_lifecycle_delete_removes_file(client, tmp_path):
    """Lifecycle: upload succeeds, then DELETE removes the file.
    This documents the server-side contract: uploaded files persist until
    explicitly deleted. The JS only uploads when WebSocket is OPEN, so
    orphan files cannot occur via the queued-send path (offline upload is rejected).
    """
    data = b"test content"
    # Step 1: upload succeeds
    up = client.post("/api/chat/upload", files={"file": ("lifecycle.txt", io.BytesIO(data), "text/plain")})
    assert up.status_code == 200
    body = up.json()
    assert body["ok"] is True
    stored_name = body["filename"]
    dest = tmp_path / "uploads" / stored_name
    assert dest.exists(), "File must exist after upload"

    # Step 2: delete (e.g. user removes attachment before sending)
    del_resp = client.request(
        "DELETE",
        "/api/chat/upload",
        data=__import__("json").dumps({"filename": stored_name}),
        headers={"Content-Type": "application/json"},
    )
    assert del_resp.status_code == 200
    assert del_resp.json()["ok"] is True
    assert not dest.exists(), "Deleted file must be gone"


def test_upload_large_file_uses_disk_custody(client, tmp_path):
    """The former50MiB transport cap must not reject ordinary task inputs."""
    from hashlib import sha256

    source = tmp_path / "large.bin"
    block = b"x" * (1024 * 1024)
    expected = sha256()
    with source.open("wb") as handle:
        for _ in range(51):
            handle.write(block)
            expected.update(block)
    with source.open("rb") as handle:
        resp = client.post("/api/chat/upload", files={"file": ("big.bin", handle, "application/octet-stream")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["size"] == 51 * 1024 * 1024
    assert body["sha256"] == expected.hexdigest()
    destination = tmp_path / "uploads" / body["filename"]
    with destination.open("rb") as handle:
        actual = sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            actual.update(chunk)
    assert actual.hexdigest() == expected.hexdigest()
    assert not list(destination.parent.glob("*.uploading"))


def _delete(client, payload):
    """Helper: send DELETE /api/chat/upload with JSON body."""
    import json as _json
    return client.request(
        "DELETE",
        "/api/chat/upload",
        data=_json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def test_delete_success(client, tmp_path):
    """Upload then delete — file should be removed."""
    data = b"deleteme"
    up = client.post("/api/chat/upload", files={"file": ("todelete.txt", io.BytesIO(data), "text/plain")})
    stored_name = up.json()["filename"]
    assert (tmp_path / "uploads" / stored_name).exists()
    resp = _delete(client, {"filename": stored_name})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert not (tmp_path / "uploads" / stored_name).exists()


def test_delete_not_found(client):
    """Delete non-existent file returns 404."""
    resp = _delete(client, {"filename": "nonexistent.txt"})
    assert resp.status_code == 404
    assert resp.json()["ok"] is False


def test_delete_path_traversal(client, tmp_path):
    """Filename with path separators must be rejected."""
    resp = _delete(client, {"filename": "../evil.txt"})
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


def test_delete_missing_filename(client):
    """Missing filename field returns 400."""
    resp = _delete(client, {})
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


def test_delete_dot_filename(client):
    """Filename '.' must be rejected with 400, not cause IsADirectoryError."""
    resp = _delete(client, {"filename": "."})
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


def test_delete_dotdot_filename(client):
    """Filename '..' must be rejected with 400, not resolve to parent dir."""
    resp = _delete(client, {"filename": ".."})
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


def test_upload_file_persists_for_queued_message(client, tmp_path):
    """Uploaded file remains on server after upload, even if the WS message
    is queued for later delivery (offline reconnect path).

    Contract: upload is server-side durable. The client JS only uploads when
    WebSocket is OPEN at send time. If the WS drops after upload completes but
    before the message is delivered, the queued message will reference a path
    that still exists on the server — the file is NOT deleted by the upload
    endpoint or any queuing logic. This test verifies the server-side half
    of that contract.
    """
    data = b"queued message attachment"
    # Simulate: upload succeeds (WS was OPEN when sendMessage ran)
    up = client.post("/api/chat/upload", files={"file": ("queued.txt", io.BytesIO(data), "text/plain")})
    assert up.status_code == 200
    body = up.json()
    assert body["ok"] is True
    stored_name = body["filename"]
    dest = tmp_path / "uploads" / stored_name

    # File must exist immediately after upload — not deleted by any queuing logic.
    assert dest.exists(), "Uploaded file must persist for queued message delivery"
    assert dest.read_bytes() == data

    # Simulate: reconnect delivers the queued message. File is still there.
    assert dest.exists(), "File must still exist when reconnected message is delivered"

    # Only explicit DELETE removes it (e.g. user cancels attachment before sending).
    del_resp = _delete(client, {"filename": stored_name})
    assert del_resp.status_code == 200
    assert not dest.exists(), "File removed only by explicit DELETE"


def test_upload_parse_error_returns_400(client, monkeypatch):
    """If form parsing raises a general exception (e.g. disconnect), we return 400."""
    from starlette.requests import Request

    async def mock_form(self):
        raise RuntimeError("Unexpected disconnect or parse error")

    monkeypatch.setattr(Request, "form", mock_form)

    resp = client.post("/api/chat/upload", files={"file": ("test.txt", io.BytesIO(b"data"), "text/plain")})
    assert resp.status_code == 400
    assert resp.json()["ok"] is False
    assert "Unexpected disconnect" in resp.json()["error"]


@pytest.mark.parametrize(("cancel_mode", "copy_fails"), [("asyncio", False), ("anyio", True)])
def test_cancelled_upload_waits_for_its_copy_before_closing_spool(tmp_path, monkeypatch, copy_fails, cancel_mode):
    import anyio
    import asyncio
    import tempfile
    import threading
    from types import SimpleNamespace
    from starlette.datastructures import UploadFile
    from ouroboros.gateway import files

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    spool = tempfile.SpooledTemporaryFile(max_size=10)
    spool.write(b"complete file")
    upload = UploadFile(spool, filename="cancelled.bin")
    original = files._store_chat_upload
    def held_copy(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        try:
            if copy_fails:
                raise OSError("controlled disk copy failure")
            return original(*args, **kwargs)
        finally:
            finished.set()
    monkeypatch.setattr(files, "_store_chat_upload", held_copy)
    async def form():
        return {"file": upload}
    scopes = []
    cancelled = []
    async def request():
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            try:
                await files.api_chat_upload(SimpleNamespace(form=form))
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
    async def run():
        copying = asyncio.create_task(request())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            scopes[0].cancel() if cancel_mode == "anyio" else copying.cancel()
            await asyncio.sleep(0)
            if cancel_mode == "asyncio":
                copying.cancel()
            await asyncio.sleep(0)
            assert not copying.done() and not spool.closed
        finally:
            release.set()
            try:
                await copying
            except asyncio.CancelledError:
                assert cancel_mode == "asyncio"
        assert cancelled == [True] and finished.is_set() and spool.closed
        saved = list((tmp_path / "uploads").glob("*"))
        assert len(saved) == (0 if copy_fails else 1)
        if saved:
            assert saved[0].name.endswith("_cancelled.bin")
            assert saved[0].read_bytes() == b"complete file"
        assert not list((tmp_path / "uploads").glob(".*.tmp"))
    try:
        asyncio.run(run())
    finally:
        spool.close()
