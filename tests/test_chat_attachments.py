"""Owner chat attachments: measured refs, the confined upload route, the pending guard,
one canonical row with the same views live and on replay, and skill/transport ingress.

The Windows handle proof is exercised here against a FAKE Win32 file API: that proves
the decision logic only. Only a real Windows run (the ``win32`` tests below, run by the
Windows leg of ``full-test``) qualifies the ctypes binding itself: streams, reparse points,
a held directory's rename, a file moved out while opened, and every handle closed.
"""
from __future__ import annotations

import asyncio
import base64
import errno
import hashlib
import io
import json
import os
import pathlib
import sys
import threading
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import chat_uploads, confined_files

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 56
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 60


@pytest.fixture(autouse=True)
def _fresh_pending():
    from supervisor import message_bus

    chat_uploads._PENDING.clear()
    yield
    chat_uploads._PENDING.clear()
    getattr(message_bus, "_UNDISPATCHED", {}).clear()


@pytest.fixture
def files_app(tmp_path, monkeypatch):
    import ouroboros.gateway.files as files

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    app = Starlette(routes=[
        Route("/api/chat/upload", files.api_chat_upload, methods=["POST"]),
        Route("/api/chat/upload", files.api_chat_upload_delete, methods=["DELETE"]),
        Route("/api/files/download", files.api_files_download),
    ])
    with TestClient(app) as client:
        yield client


def _upload(client, name: str, data: bytes) -> dict:
    response = client.post("/api/chat/upload", files={"file": (name, io.BytesIO(data), "application/octet-stream")})
    assert response.status_code == 200, response.text
    return response.json()


# --- naming and byte detection -------------------------------------------------------

@pytest.mark.parametrize("head,name,expected", [
    (PNG, "x.jpg", ("image/png", "image")),
    (JPEG, "photo", ("image/jpeg", "image")),
    (b"GIF89a" + b"\x00" * 10, "a.gif", ("image/gif", "image")),
    (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "a.webp", ("image/webp", "image")),
    (b"RIFF\x00\x00\x00\x00WAVEfmt ", "a.wav", ("audio/wav", "audio")),
    (b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic", "a.heic", ("image/heic", "image")),
    (b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1miaf", "a.avif", ("image/avif", "image")),
    (b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00M4A isommp42", "a.m4a", ("audio/mp4", "audio")),
    (b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41", "a.mp4", ("video/mp4", "video")),
    (b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41", "voice.m4a", ("audio/mp4", "audio")),
    (b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00qt  ", "a.mov", ("video/quicktime", "video")),
    (b"\x1aE\xdf\xa3\x9fB\x86\x81\x01B\xf7\x81\x01B\xf2\x81\x04B\xf3\x81\x08B\x82\x84webm", "a.webm", ("video/webm", "video")),
    (b"ID3\x04\x00", "a.mp3", ("audio/mpeg", "audio")),
    (b"OggS\x00\x02", "a.ogg", ("audio/ogg", "audio")),
    (b"\x00\x00\x00\x18ftypzzzz\x00\x00\x00\x00zzzz", "a.mp4", ("application/octet-stream", "file")),
    (b"<svg xmlns='http://www.w3.org/2000/svg'>", "a.svg", ("image/svg+xml", "file")),
    (b"<html><script>alert(1)</script>", "a.png", ("application/octet-stream", "file")),
    (b"<html>", "a.html", ("text/html", "file")),
    (b"%PDF-1.7", "a.pdf", ("application/pdf", "file")),
    (b"\xff\xfb\x90\x00", "a.bin", ("application/octet-stream", "file")),
])
def test_media_kind_is_proven_from_bytes_and_conservative(head, name, expected):
    assert chat_uploads.detect_media(head, name) == expected


def test_stored_names_are_plain_on_every_platform_and_keep_their_extension():
    assert chat_uploads.safe_upload_name("../../my photo.png") == "my_photo.png"
    assert chat_uploads.safe_upload_name("a:b?*|.pdf") == "a_b___.pdf"
    assert chat_uploads.safe_upload_name("CON.txt") == "upload.txt"
    assert chat_uploads.safe_upload_name("notes. ") == "notes"
    long = chat_uploads.safe_upload_name("Фото" * 80 + ".jpeg")
    assert long.endswith(".jpeg") and len(long.encode("utf-8")) <= 222
    assert len(chat_uploads.new_upload_id("x" * 5000 + ".txt").encode("utf-8")) <= 255
    for bad in ("../x", "0" * 32 + "_../x", "0" * 32 + "_a:stream", "0" * 31 + "_a", "0" * 32 + "_a\\b", ""):
        assert chat_uploads.upload_id(bad) == "", bad


def test_a_name_is_bounded_in_code_points_the_bound_the_browser_mirrors():
    """web/tests/chat_attachment_values.test.js keeps these same names: the browser's display
    bound counts code points too, so it passes a server name whole and never splits a pair."""
    name = chat_uploads.safe_upload_name("a" * 180 + "😀" * 10 + ".png")
    assert name == "a" * 180 + "😀" * 9 + ".png" and len(name) == 193
    assert len(name.encode("utf-16-le")) // 2 == 202, "past 200 UTF-16 units"
    view = chat_uploads.attachment_view({"upload": f"{'0' * 32}_{name}", "name": name, "kind": "image", "size": 1})
    assert view["available"] and view["name"] == name
    assert chat_uploads.attachment_view({"name": "😀" * 201})["name"] == "😀" * 200


# --- confined open ------------------------------------------------------------------

@pytest.mark.skipif(not confined_files.POSIX_CONFINED, reason="POSIX directory-relative opens")
def test_posix_open_refuses_links_fifos_directories_and_names(tmp_path):
    (tmp_path / "plain.png").write_bytes(PNG)
    handle, observed = confined_files.open_regular_file(tmp_path, "plain.png")
    with handle:
        assert handle.read() == PNG and observed.st_size == len(PNG)
    (tmp_path / "link.png").symlink_to(tmp_path / "plain.png")
    with pytest.raises(OSError) as link:
        confined_files.open_regular_file(tmp_path, "link.png")
    assert link.value.errno in (errno.ELOOP, errno.EMLINK)
    (tmp_path / "dir.png").mkdir()
    with pytest.raises(OSError):
        confined_files.open_regular_file(tmp_path, "dir.png")
    os.mkfifo(tmp_path / "fifo.png")
    done = threading.Event()
    def opener():
        try:
            confined_files.open_regular_file(tmp_path, "fifo.png")
        except OSError as exc:
            assert exc.errno is None  # refused like a swap, never opened for reading
            done.set()
    threading.Thread(target=opener, daemon=True).start()
    assert done.wait(5), "a planted FIFO must never block the open"
    for name in ("../plain.png", "a/b", "", "CON", "x.", "a:b"):
        with pytest.raises(OSError) as invalid:
            confined_files.open_regular_file(tmp_path, name)
        assert invalid.value.errno == errno.EINVAL, name


class FakeWin32:
    """A fake of ``_Win32Files``: decision logic only, never a Windows qualification."""

    def __init__(self, tmp_path, *, reparse=False, file_type_disk=True, final=None):
        self.real = tmp_path / "file.bin"
        self.real.write_bytes(b"windows bytes")
        self.reparse, self.disk, self.final, self.closed, self.opened = reparse, file_type_disk, final, [], []

    def open(self, path, *, directory):
        self.opened.append((path, directory))
        return 100 + len(self.opened)

    def require_plain(self, handle, *, directory):
        if self.reparse and not directory:
            raise OSError(errno.ELOOP, "a reparse point is not followed")
        if not directory and not self.disk:
            raise OSError("not an ordinary disk file")

    def final_path(self, handle):
        return "\\\\?\\C:\\Data\\uploads" if handle == 101 else (self.final or "\\\\?\\c:\\data\\UPLOADS\\name.png")

    def to_fd(self, handle):
        return os.open(self.real, os.O_RDONLY)

    def close(self, handle):
        self.closed.append(handle)


def test_windows_handle_proof_logic_against_a_fake_api(tmp_path):
    ok = FakeWin32(tmp_path)
    handle, observed = confined_files.open_regular_file("C:/Data/uploads", "name.png", win32=ok)
    with handle:
        assert handle.read() == b"windows bytes" and observed.st_size == 13
    assert ok.closed == [101], "the held directory is closed; the file handle now belongs to the fd"
    assert ok.opened[0][1] is True and ok.opened[1][1] is False
    for api, expected in ((FakeWin32(tmp_path, reparse=True), errno.ELOOP),
                          (FakeWin32(tmp_path, file_type_disk=False), None),
                          (FakeWin32(tmp_path, final="\\\\?\\C:\\Elsewhere\\name.png"), None),
                          (FakeWin32(tmp_path, final="\\\\?\\UNC\\server\\share\\name.png"), None)):
        with pytest.raises(OSError) as refused:
            confined_files.open_regular_file("C:/Data/uploads", "name.png", win32=api)
        assert refused.value.errno == expected
        assert sorted(api.closed) == [101, 102], "every handle is closed on refusal"
    with pytest.raises(OSError) as device:
        confined_files.open_regular_file("C:/Data/uploads", "NUL.txt", win32=FakeWin32(tmp_path))
    assert device.value.errno == errno.EINVAL


class RecordingWin32:
    """The REAL kernel32 binding, recording every handle it opens and the file object it names;
    ``after_open`` runs right after each open (a racing writer). It records, never raises, so no
    handle can leak."""

    def __init__(self, after_open=None):
        self.api, self.after_open, self.handles, self.objects = confined_files._Win32Files.load(), after_open, [], {}

    def open(self, path, *, directory):
        handle = self.api.open(path, directory=directory)
        self.handles.append(handle)
        self.objects[handle] = self._identity(handle)
        if self.after_open:
            self.after_open(directory)
        return handle

    def __getattr__(self, name):
        return getattr(self.api, name)

    def _identity(self, handle):
        """(volume, file index) of the file HANDLE names now, or None when the query fails."""
        info = self.api._info_type()
        if not self.api._info(handle, self.api._ctypes.byref(info)):
            return None
        return info.volume, info.index_high, info.index_low

    def _kind(self, handle):
        """The kernel object type HANDLE names now (``File`` for a file or directory), or None when the
        query fails."""
        import ctypes
        from ctypes import wintypes

        query = ctypes.WinDLL("ntdll").NtQueryObject
        query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.ULONG, ctypes.c_void_p]
        query.restype = ctypes.c_long
        buffer = (ctypes.c_void_p * 512)()  # PUBLIC_OBJECT_TYPE_INFORMATION opens with a UNICODE_STRING
        if query(handle, 2, buffer, ctypes.sizeof(buffer), None) < 0 or not buffer[1]:  # ObjectTypeInformation
            return None
        return ctypes.wstring_at(buffer[1], ctypes.c_ushort.from_buffer(buffer).value // 2)

    def leaked(self):
        """Handles still open now. ``GetHandleInformation`` fails on a closed one, but Windows hands a
        closed handle's value to the next kernel object (CPython 3.10's locks, a BufferedReader's too,
        are semaphores), so an open value is released only when proven to name something else: a
        non-file object, or another file. A query that fails proves nothing: still open."""
        import ctypes
        from ctypes import wintypes

        info = ctypes.WinDLL("kernel32", use_last_error=True).GetHandleInformation
        info.argtypes, info.restype = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL
        return [h for h in self.handles if info(h, ctypes.byref(wintypes.DWORD())) and self._kind(h) in (None, "File")
                and (self.objects[h] is None or self._identity(h) in (None, self.objects[h]))]


def _windows_uploads(tmp_path):
    uploads, outside = tmp_path / "uploads", tmp_path / "outside"
    uploads.mkdir()
    outside.mkdir()
    (uploads / "plain.png").write_bytes(PNG)
    (outside / "plain.png").write_bytes(b"OUTSIDE")
    return uploads, outside


@pytest.mark.skipif(sys.platform != "win32", reason="real Windows handles (Windows CI leg)")
def test_windows_leak_probe_reports_a_held_handle_and_releases_only_a_proven_reuse(tmp_path):
    """Controls for the probe the real-handle tests trust: ``leaked() == []`` there means something."""
    import _winapi

    uploads, _outside = _windows_uploads(tmp_path)
    api = RecordingWin32()
    held = api.open(str(uploads / "plain.png"), directory=False)
    try:
        assert api._kind(held) == "File" and api.objects[held] is not None
        assert api.leaked() == [held], "a real handle still held is reported"
        for unknown in ("_identity", "_kind"):  # a query that fails is no proof of release
            setattr(api, unknown, lambda _handle: None)
            assert api.leaked() == [held], unknown
        del api._identity, api._kind
        recorded, api.objects[held] = api.objects[held], (0, 0, 0)
        assert api.leaked() == [], "the value now names another file: released"
        api.objects[held] = recorded
        other = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False, os.getpid())  # a live non-file object
        try:
            api.handles.append(other)
            api.objects[other] = recorded  # as if the file's value had been handed to it
            assert api._kind(other) == "Process" and api.leaked() == [held], "a value naming a non-file is released"
        finally:
            _winapi.CloseHandle(other)
    finally:
        api.close(held)
    assert api.leaked() == []


@pytest.mark.skipif(sys.platform != "win32", reason="real Windows handles (Windows CI leg)")
def test_windows_real_handles_refuse_reparse_points_and_streams_and_close_every_handle(tmp_path):
    import _winapi

    uploads, outside = _windows_uploads(tmp_path)
    with open(f"{uploads / 'plain.png'}:hidden", "wb") as stream:  # an alternate data stream
        stream.write(b"STREAM BYTES")
    api = RecordingWin32()
    handle, observed = confined_files.open_regular_file(uploads, "plain.png", win32=api)
    with handle:
        assert handle.read() == PNG and observed.st_size == len(PNG), "only the file's own data, never a stream"
    assert len(api.handles) == 2 and api.leaked() == [], "the held directory closed; the file closed with its fd"
    for name in ("plain.png:hidden", "plain.png::$DATA", "plain.png:hidden:$DATA", "NUL", "con.png"):
        with pytest.raises(OSError) as named:
            confined_files.open_regular_file(uploads, name)
        assert named.value.errno == errno.EINVAL, name  # refused by name, before any handle

    _winapi.CreateJunction(str(outside), str(uploads / "junction.png"))
    os.symlink(outside / "plain.png", uploads / "link.png")
    _winapi.CreateJunction(str(uploads), str(tmp_path / "via"))
    # A directory junction named as the file is refused by the open itself (a directory needs
    # backup semantics) or as a reparse point; a file symlink, and a junction standing in for
    # the trusted directory, as reparse points.
    for directory, name, refusals in ((uploads, "junction.png", {errno.EACCES, errno.ELOOP}),
                                      (uploads, "link.png", {errno.ELOOP}),
                                      (tmp_path / "via", "plain.png", {errno.ELOOP})):
        api = RecordingWin32()
        with pytest.raises(OSError) as reparse:
            confined_files.open_regular_file(directory, name, win32=api)
        assert reparse.value.errno in refusals, (directory, name, reparse.value)
        assert api.handles and api.leaked() == [], f"a refusal closes every handle it opened: {name}"


@pytest.mark.skipif(sys.platform != "win32", reason="real Windows handles (Windows CI leg)")
def test_windows_real_handles_hold_the_directory_and_refuse_a_file_moved_out(tmp_path):
    uploads, outside = _windows_uploads(tmp_path)
    renames = []

    def rename_directory(directory):
        if directory:  # the held directory can be neither renamed nor deleted while held
            try:
                os.rename(uploads, tmp_path / "moved")
                renames.append("renamed")
            except PermissionError:
                renames.append("refused")

    api = RecordingWin32(rename_directory)
    try:
        handle, _observed = confined_files.open_regular_file(uploads, "plain.png", win32=api)
    finally:
        assert renames == ["refused"], "the held directory must refuse its rename"
    with handle:
        assert handle.read() == PNG
    assert api.leaked() == []
    os.rename(uploads, tmp_path / "moved")  # released: nothing of the open is still held
    os.rename(tmp_path / "moved", uploads)

    moves = []

    def move_file_out(directory):
        if not directory:  # the file shares delete, so it can move; its final path then differs
            try:
                os.rename(uploads / "plain.png", outside / "moved.png")
                moves.append("moved")
            except OSError as exc:
                moves.append(repr(exc))

    api = RecordingWin32(move_file_out)
    try:
        with pytest.raises(OSError, match="left its directory") as moved:
            confined_files.open_regular_file(uploads, "plain.png", win32=api)
    finally:
        assert moves == ["moved"], f"a file opened with delete sharing can be moved: {moves}"
    assert moved.value.errno is None and len(api.handles) == 2 and api.leaked() == []
    os.remove(outside / "moved.png")  # no handle of the refused open is left on it


# --- the upload route ---------------------------------------------------------------

def test_upload_returns_the_measured_view_and_keeps_the_model_mime(files_app):
    body = _upload(files_app, "shot one.png", PNG)
    assert body["mime"] == "image/png" and body["display_name"] == "shot_one.png"
    assert body["view"] == {"name": "shot_one.png", "kind": "image", "mime": "image/png", "size": len(PNG),
                            "available": True, "url": "/api/files/download?upload=" + body["filename"]}
    assert body["sha256"] == hashlib.sha256(PNG).hexdigest()


def test_download_branch_serves_media_inline_and_everything_else_as_download(files_app, tmp_path, monkeypatch):
    image = _upload(files_app, "a.png", PNG)
    heic = _upload(files_app, "b.heic", b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 40)
    for name, data in (("c.html", b"<html><script>x</script>"), ("d.svg", b"<svg onload='x'/>"),
                       ("e.pdf", b"%PDF-1.4"), ("f.bin", b"\x00\x01")):
        served = files_app.get(_upload(files_app, name, data)["view"]["url"])
        assert served.status_code == 200 and served.content == data
        assert served.headers["content-type"] == "application/octet-stream", name
        assert served.headers["content-disposition"].startswith("attachment"), name
        assert served.headers["content-security-policy"] == "default-src 'none'; sandbox"
        assert served.headers["x-content-type-options"] == "nosniff"
    inline = files_app.get(image["view"]["url"])
    assert inline.headers["content-type"] == "image/png" and inline.headers["content-disposition"].startswith("inline")
    assert inline.headers["cache-control"] == "private, max-age=86400"
    assert inline.headers["cross-origin-resource-policy"] == "same-origin"
    assert files_app.get(heic["view"]["url"]).headers["content-type"] == "image/heic"
    # The Files root is another authority: re-rooting it changes nothing here.
    monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(tmp_path / "elsewhere"))
    (tmp_path / "elsewhere").mkdir()
    assert files_app.get(image["view"]["url"]).status_code == 200


def test_download_branch_range_head_and_refusals(files_app, tmp_path, monkeypatch):
    from ouroboros import artifacts

    payload = bytes(range(256)) * 4096  # 1 MiB
    stored = _upload(files_app, "big.bin", payload)
    url = stored["view"]["url"]
    monkeypatch.setattr(artifacts, "stream_artifact_file", lambda *a, **k: pytest.fail("no whole-file hash per read"))
    tail = files_app.get(url, headers={"Range": "bytes=1048000-1048575"})
    assert tail.status_code == 206 and tail.content == payload[1048000:]
    assert tail.headers["content-range"] == "bytes 1048000-1048575/1048576"
    assert files_app.get(url, headers={"Range": "bytes=2000000-"}).status_code == 416
    head = files_app.head(url)
    assert head.status_code == 200 and head.headers["content-length"] == str(len(payload)) and not head.content
    assert files_app.get("/api/files/download?upload=../../etc/passwd").status_code == 400
    assert files_app.get(url + "&path=big.bin").status_code == 400
    assert files_app.get(url + "&upload=" + stored["filename"]).status_code == 400
    assert files_app.get("/api/files/download?upload=" + "0" * 32 + "_gone.png").status_code == 404
    uploads = tmp_path / "uploads"
    (uploads / ("1" * 32 + "_link.png")).symlink_to("/etc/hosts")
    assert files_app.get("/api/files/download?upload=" + "1" * 32 + "_link.png").status_code == 404
    if hasattr(os, "mkfifo"):
        os.mkfifo(uploads / ("2" * 32 + "_fifo.png"))
        assert files_app.get("/api/files/download?upload=" + "2" * 32 + "_fifo.png").status_code == 404


def test_upload_route_keeps_the_network_password_gate(tmp_path, monkeypatch):
    import ouroboros.gateway.files as files
    from ouroboros import server_auth

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server_auth, "get_configured_network_password", lambda: "secret")
    stored, _ref = chat_uploads.store_upload(PNG, "a.png", data_dir=tmp_path)
    app = server_auth.NetworkAuthGate(Starlette(routes=[Route("/api/files/download", files.api_files_download)]))
    response = TestClient(app).get("/api/files/download?upload=" + stored.name)  # "testclient" is not loopback
    assert response.status_code == 401


# --- the pending guard --------------------------------------------------------------

def test_only_a_pending_upload_can_be_deleted_and_acceptance_wins(files_app, tmp_path):
    def delete(name):
        return files_app.request("DELETE", "/api/chat/upload", json={"filename": name})

    pending = _upload(files_app, "a.png", PNG)
    assert delete(pending["filename"]).status_code == 200
    assert not (tmp_path / "uploads" / pending["filename"]).exists()
    accepted = _upload(files_app, "b.png", PNG)
    refs = chat_uploads.claim_refs(chat_uploads.refs_for_frame([{"filename": accepted["filename"]}]), tmp_path)
    assert refs[0]["kind"] == "image"
    refused = delete(accepted["filename"])
    assert refused.status_code == 409 and (tmp_path / "uploads" / accepted["filename"]).is_file()
    # Deleted while pending, then named by a frame: the row records it unavailable.
    raced = _upload(files_app, "c.png", PNG)
    measured = chat_uploads.refs_for_frame([{"filename": raced["filename"], "display_name": "c.png"}])
    assert delete(raced["filename"]).status_code == 200
    assert chat_uploads.claim_refs(measured, tmp_path) == [chat_uploads.unavailable_ref("c.png")]
    # After a restart nothing is pending: fail closed, the copy stays.
    survivor = _upload(files_app, "d.png", PNG)
    chat_uploads._PENDING.clear()
    assert delete(survivor["filename"]).status_code == 409
    assert delete("nonexistent.txt").status_code == 404


def test_a_frame_naming_an_unknown_upload_records_an_honest_unavailable_ref(tmp_path):
    refs = chat_uploads.refs_for_frame([{"filename": "0" * 32 + "_ghost.png", "display_name": "ghost.png"},
                                        {"filename": "../etc/passwd"}, "junk"], tmp_path)
    assert refs == [chat_uploads.unavailable_ref("ghost.png"), chat_uploads.unavailable_ref("../etc/passwd"),
                    chat_uploads.unavailable_ref("")]
    assert [view["available"] for view in chat_uploads.attachment_views(refs)] == [False, False, False]
    assert all("url" not in view for view in chat_uploads.attachment_views(refs))


# --- one canonical row: own bubble, echo and history alike ------------------------

def _history(tmp_path, chat_id=1):
    from ouroboros.gateway.history import make_chat_history_endpoint

    (tmp_path / "logs" / "progress.jsonl").touch()
    endpoint = make_chat_history_endpoint(tmp_path)
    query = {"limit": "20", **({"chat_id": str(chat_id)} if chat_id != 1 else {})}
    response = asyncio.run(endpoint(SimpleNamespace(query_params=query)))
    return json.loads(response.body.decode("utf-8"))["messages"]


def test_web_acceptance_records_refs_and_echo_equals_history(files_app, tmp_path, monkeypatch):
    from ouroboros.gateway import ws
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ws, "DATA_DIR", tmp_path)
    bridge = message_bus.LocalChatBridge()
    echoes = []
    bridge._broadcast_fn = echoes.append
    first, second = _upload(files_app, "one.png", PNG), _upload(files_app, "two.pdf", b"%PDF-1.4")
    text = "Посмотри\n\n[Attached file: one.png]\n[Attached file: two.pdf]"
    frame = [{"filename": item["filename"], "display_name": item["display_name"], "mime": item["mime"]}
             for item in (first, second)]
    send_kwargs = dict(broadcast=True, sender_session_id="s", client_message_id="cm-1", chat_id=1,
                       task_metadata={"chat_attachment_uploads": ws._chat_attachment_uploads(frame)})
    ws._accept_with_attachments(bridge, text, send_kwargs, frame)

    rows = [json.loads(line) for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["text"] == text, "canonical text and model input are unchanged"
    assert [ref["name"] for ref in rows[0]["attachments"]] == ["one.png", "two.pdf"]
    assert all(set(ref) == {"upload", "name", "mime", "kind", "size", "sha256", "mtime_ns"} for ref in rows[0]["attachments"])
    assert echoes[0]["attachments"] == [first["view"], second["view"]], "own bubble view == echo view"
    replayed = [row for row in _history(tmp_path) if row.get("client_message_id") == "cm-1"]
    assert replayed[0]["attachments"] == echoes[0]["attachments"] and replayed[0]["text"] == text
    queued = bridge._inbox.get_nowait()
    specs = queued["task_metadata"]["chat_attachment_uploads"]
    assert [spec["label"] for spec in specs] == ["one.png", "two.pdf"]
    assert [(spec["size"], spec["sha256"]) for spec in specs] == [(ref["size"], ref["sha256"]) for ref in rows[0]["attachments"]]
    deleted = files_app.request("DELETE", "/api/chat/upload", json={"filename": first["filename"]})
    assert deleted.status_code == 409, "an accepted original is never removed by composer cleanup"


# --- skill and transport ingress ---------------------------------------------------

def _skill_client(tmp_path, bridge):
    from tests.test_chat_inject_attachments import _client

    return _client(tmp_path, bridge)


def test_transport_inline_photo_is_parked_once_and_still_reaches_vision(tmp_path, monkeypatch):
    from supervisor import message_bus
    from tests.test_host_service_api import FakeBridge

    bridge = FakeBridge()
    from tests.test_live_image_delivery import pixels
    original = pixels("JPEG")
    photo = base64.b64encode(original).decode("ascii")
    response = _skill_client(tmp_path, bridge).post(
        "/chat/inject", headers={"X-Skill-Token": "token"},
        json={"text": "", "image_base64": photo, "image_mime": "image/jpeg", "image_caption": "кадр",
              "chat_id": 42, "user_id": 42, "source": "telegram", "transport": {"kind": "telegram"}})
    assert response.status_code == 202
    message = bridge.messages[0]
    assert message["image_base64"] == photo, "vision input is unchanged"
    (spec,) = message["task_metadata"]["chat_attachment_uploads"]
    (ref,) = message["task_metadata"]["chat_attachments"]
    assert ref["name"] == "photo.jpg" and ref["kind"] == "image" and ref["sha256"] == hashlib.sha256(original).hexdigest()
    assert spec["sha256"] == ref["sha256"] and spec["size"] == len(original)
    assert pathlib.Path(spec["path"]).read_bytes() == original
    assert (tmp_path / "uploads" / ref["upload"]).read_bytes() == original
    assert len(list((tmp_path / "uploads").iterdir())) == 1

    # The worker's canonical writer: one row with the ref, one echo frame without a second photo.
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    frames = []
    real = message_bus.LocalChatBridge()
    real._broadcast_fn = frames.append
    message_bus.record_inbound_message(real, {**message, "text": "", "source": "skill:telegram"}, chat_id=42,
                                       user_id=42, client_message_id="host-1", text="кадр", ts="2026-10-05T00:00:00Z")
    (frame,) = frames
    assert frame["type"] == "chat" and frame["image_base64"] == "" and frame["attachments"][0]["kind"] == "image"
    (row,) = [json.loads(line) for line in (tmp_path / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    assert row["attachments"][0]["upload"] == ref["upload"]
    assert _history(tmp_path, 42)[0]["attachments"] == frame["attachments"]


def test_retry_identity_includes_attachment_content_not_stored_names(tmp_path):
    from supervisor import message_bus
    from tests.test_chat_inject_attachments import _skill_file

    bridge = message_bus.LocalChatBridge()
    client = _skill_client(tmp_path, bridge)
    source = _skill_file(tmp_path, "scan.pdf", b"%PDF first")
    body = {"text": "the scan", "chat_id": 42, "user_id": 42, "client_message_id": "tg:1",
            "attachments": [{"path": str(source), "name": "scan.pdf"}]}
    assert client.post("/chat/inject", headers={"X-Skill-Token": "token"}, json=body).status_code == 202
    replay = client.post("/chat/inject", headers={"X-Skill-Token": "token"}, json=body)
    assert replay.status_code == 202 and replay.json()["rejoined"] is True, "a new stored name is the same message"
    assert len(list((tmp_path / "uploads").iterdir())) == 1
    source.write_bytes(b"%PDF different")
    changed = client.post("/chat/inject", headers={"X-Skill-Token": "token"}, json=body)
    assert changed.status_code == 409, "same id, same words, different file is a different message"
    photo = {**body, "client_message_id": "tg:2", "attachments": [],
             "image_base64": base64.b64encode(JPEG).decode("ascii")}
    assert client.post("/chat/inject", headers={"X-Skill-Token": "token"}, json=photo).status_code == 202
    other = {**photo, "image_base64": base64.b64encode(PNG).decode("ascii")}
    assert client.post("/chat/inject", headers={"X-Skill-Token": "token"}, json=other).status_code == 409
    assert bridge._inbox.qsize() == 2


def test_same_message_compares_words_and_ordered_attachment_identity():
    row = {"text": "a  b", "attachments": [{"sha256": "1", "name": "x.png"}, {"sha256": "2", "name": "y.pdf"}]}
    assert chat_uploads.same_message(row, "a b", [{"sha256": "1", "name": "x.png", "upload": "new"},
                                                  {"sha256": "2", "name": "y.pdf"}])
    assert not chat_uploads.same_message(row, "a b", list(reversed(row["attachments"])))
    assert not chat_uploads.same_message(row, "a b", [])
    assert not chat_uploads.same_message({"text": "a b"}, "a b", row["attachments"])


# --- every attachment, the owner's words, one message per web id ---------------------

def _web_bridge(tmp_path, monkeypatch):
    from ouroboros.gateway import ws
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ws, "DATA_DIR", tmp_path)
    bridge = message_bus.LocalChatBridge()
    echoes = []
    bridge._broadcast_fn = echoes.append
    return bridge, echoes


def _send_web(bridge, text, uploads, cmid="cm-1", **extra):
    """One composer frame through the socket's own acceptance (``ws._accept_with_attachments``)."""
    from ouroboros.gateway import ws

    frame = [{"filename": item["filename"], "display_name": item["display_name"], "mime": item["mime"]}
             for item in uploads]
    send_kwargs = dict(broadcast=True, sender_session_id="s", client_message_id=cmid, chat_id=1,
                       task_metadata={"chat_attachment_uploads": ws._chat_attachment_uploads(frame)}, **extra)
    if frame:
        ws._accept_with_attachments(bridge, text, send_kwargs, frame)
    else:
        bridge.ui_send(text, **send_kwargs)


def _rows(tmp_path):
    path = tmp_path / "logs" / "chat.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def test_every_attachment_is_recorded_echoed_and_replayed_past_the_text_tail(files_app, tmp_path, monkeypatch):
    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    uploads = [_upload(files_app, f"shot-{index:02d}.png", PNG) for index in range(27)]
    uploads += [_upload(files_app, f"note-{index}.txt", b"note") for index in range(3)]
    # The model text names 25 files and counts the rest; the structured refs keep all 30.
    tail = "\n".join([f"[Attached file: {item['display_name']}]" for item in uploads[:25]] + ["[5 more attached files]"])
    _send_web(bridge, "Тридцать файлов\n\n" + tail, uploads)

    (row,) = _rows(tmp_path)
    assert [ref["name"] for ref in row["attachments"]] == [item["display_name"] for item in uploads]
    assert echoes[0]["attachments"] == [item["view"] for item in uploads], "no attachment is dropped"
    assert _history(tmp_path)[0]["attachments"] == echoes[0]["attachments"]
    queued = bridge._inbox.get_nowait()
    assert len(queued["task_metadata"]["chat_attachment_uploads"]) == 30


def test_a_placeholder_is_marked_only_when_the_host_wrote_it(tmp_path, monkeypatch):
    from supervisor import message_bus

    bridge = message_bus.LocalChatBridge()
    client = _skill_client(tmp_path, bridge)
    photo = base64.b64encode(JPEG).decode("ascii")
    for cmid, text in (("tg:host", ""), ("tg:owner", "(image attached)")):
        response = client.post("/chat/inject", headers={"X-Skill-Token": "token"}, json={
            "text": text, "image_base64": photo, "chat_id": 42, "user_id": 42, "client_message_id": cmid})
        assert response.status_code == 202, response.text
    host, owner = _rows(tmp_path)
    assert host["text"] == owner["text"] == "(image attached)", "canonical text unchanged"
    assert host.get("text_placeholder") is True and "text_placeholder" not in owner
    replayed = {row["client_message_id"]: row for row in _history(tmp_path, 42)}
    assert replayed["tg:host"].get("text_placeholder") is True
    assert "text_placeholder" not in replayed["tg:owner"], "the owner's own words stay the owner's"

    # The dequeued (unnamed) writer and the web writer mark the same way.
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    ref = host["attachments"]
    for cmid, raw in (("host-1", ""), ("host-2", "(file attached)")):
        message_bus.record_inbound_message(
            bridge, {"text": raw, "source": "skill:telegram", "task_metadata": {"chat_attachments": ref}},
            chat_id=42, user_id=42, client_message_id=cmid, text="(file attached)", ts="2026-10-05T00:00:00Z")
    bridge.ui_send("(file attached)", client_message_id="web-1", task_metadata={"chat_attachments": ref})
    marked = {row["client_message_id"]: row.get("text_placeholder") for row in _rows(tmp_path)[2:]}
    assert marked == {"host-1": True, "host-2": None, "web-1": None}


def test_a_redelivered_web_frame_rejoins_its_message_and_a_changed_one_is_refused(files_app, tmp_path, monkeypatch):
    from supervisor import message_bus

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    photo = _upload(files_app, "one.png", PNG)
    text = "Посмотри\n\n[Attached file: one.png]"
    _send_web(bridge, text, [photo])
    _send_web(bridge, text, [photo])  # the same frame dispatched again
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1, "no second row, queue item or task"
    assert echoes[1] == echoes[0], "the rejoin re-echoes the accepted row: same time, views and id"
    # A fresh stored copy of the same bytes under the same name is still the same message.
    copy = _upload(files_app, "one.png", PNG)
    _send_web(bridge, text, [copy])
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1

    changed = _upload(files_app, "one.png", JPEG)
    for words, uploads in ((text, [changed]), ("Другие слова\n\n[Attached file: one.png]", [photo]), (text, [])):
        with pytest.raises(ValueError, match="different message"):
            _send_web(bridge, words, uploads)
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1 and len(echoes) == 3
    for spare in (changed, copy):  # neither the refused nor the rejoining frame claimed its upload
        assert files_app.request("DELETE", "/api/chat/upload", json={"filename": spare["filename"]}).status_code == 200
    assert files_app.request("DELETE", "/api/chat/upload", json={"filename": photo["filename"]}).status_code == 409

    dispatched = []
    for _ in range(2):  # an accepted Restart is dispatched once
        _send_web(bridge, "/restart", [], cmid="cm-2", dispatch=lambda _text, **message: dispatched.append(message))
    assert [message["client_message_id"] for message in dispatched] == ["cm-2"]
    message_bus.log_chat("in", 1, 0, "from a skill", source="skill:telegram", client_message_id="cm-3",
                         drive_root=tmp_path, require_write=True)
    with pytest.raises(ValueError, match="another source"):
        _send_web(bridge, "from a skill", [], cmid="cm-3")


def test_the_accepted_row_lookup_skips_only_segments_that_cannot_name_the_id(tmp_path):
    from supervisor import message_bus

    def line(cmid, text, *, ascii_only=False, chat_id=1):
        return json.dumps({"direction": "in", "chat_id": chat_id, "client_message_id": cmid, "text": text},
                          ensure_ascii=ascii_only) + "\n"

    (tmp_path / "archive").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "archive" / "chat_20261001T000000.jsonl").write_text(
        line("msg-1-1", "archived") + line('тест"2', "escaped", ascii_only=True), encoding="utf-8")
    (tmp_path / "logs" / "chat.jsonl").write_text(line("msg-1-10", "live"), encoding="utf-8")
    assert message_bus.accepted_chat_message(tmp_path, 1, "msg-1-1")["text"] == "archived"
    assert message_bus.accepted_chat_message(tmp_path, 1, 'тест"2')["text"] == "escaped", "escaped ids are parsed"
    assert message_bus.accepted_chat_message(tmp_path, 1, "msg-1-10")["text"] == "live"
    assert message_bus.accepted_chat_message(tmp_path, 1, "msg-1") is None, "a substring hit is only a candidate"
    assert message_bus.accepted_chat_message(tmp_path, 2, "msg-1-1") is None


# --- a transport's inline photo: bounded, valid, and seen by the model beside files ----

def _inject(client, **body):
    return client.post("/chat/inject", headers={"X-Skill-Token": "token"},
                       json={"chat_id": 42, "user_id": 42, **body})


def test_an_inline_photo_is_refused_by_its_exact_decoded_size_before_any_decode(tmp_path, monkeypatch):
    """The bound is the decoded size, padding counted, judged from the base64 length alone:
    a photo one or two bytes past it is refused before decode; the decode re-checks it."""
    from ouroboros.gateway import host_service
    from tests.test_host_service_api import FakeBridge

    bridge = FakeBridge()
    client = _skill_client(tmp_path, bridge)
    monkeypatch.setattr(host_service, "_INLINE_IMAGE_MAX", len(JPEG))
    exact, over_one, over_two = (base64.b64encode(JPEG + b"\x00" * extra).decode("ascii") for extra in (0, 1, 2))
    assert (exact.endswith("=="), over_one.endswith("="), len(over_two) == len(exact)) == (True, True, True)
    with monkeypatch.context() as patch:
        patch.setattr(host_service, "_inline_image_bytes", lambda _payload: pytest.fail("decoded before the size check"))
        for encoded in (over_one, over_two, "\n".join(over_two[i:i + 16] for i in range(0, len(over_two), 16))):
            too_large = _inject(client, text="", image_base64=encoded, client_message_id="tg:big")
            assert too_large.status_code == 413, too_large.text
    with monkeypatch.context() as patch:  # the decoded backstop, should the length judgement ever pass it
        patch.setattr(host_service, "_inline_image_too_large", lambda _payload: False)
        backstop = _inject(client, text="", image_base64=over_one)
        assert backstop.status_code == 400 and "at most" in backstop.json()["error"]
    assert _inject(client, text="", image_base64="not*base64!").status_code == 400
    assert _inject(client, text="", image_base64=exact[:8] + "\u00a0" + exact[8:]).status_code == 400, "only line wrapping is unwrapped"
    assert bridge.messages == [] and not list((tmp_path / "uploads").glob("*")), "nothing parked or queued"
    wrapped = "\n".join(exact[i:i + 16] for i in range(0, len(exact), 16))  # MIME-style line breaks
    assert _inject(client, text="", image_base64=wrapped, image_mime="image/jpeg").status_code == 202
    (ref,) = bridge.messages[0]["task_metadata"]["chat_attachments"]
    assert (tmp_path / "uploads" / ref["upload"]).read_bytes() == JPEG


def test_inline_bytes_that_prove_no_image_are_an_ordinary_file_never_vision(tmp_path, monkeypatch):
    """A transport's ``image_base64`` that is HTML or a video is no photo: it is kept and staged as
    an ordinary file named from its bytes, and vision never sees it under the transport's label."""
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    bridge = message_bus.LocalChatBridge()
    client = _skill_client(tmp_path, bridge)
    mp4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00" * 32
    for cmid, data in (("tg:html", b"<html><script>alert(1)</script></html>"), ("tg:clip", mp4)):
        body = {"text": "", "image_base64": base64.b64encode(data).decode("ascii"), "image_mime": "image/jpeg",
                "client_message_id": cmid}
        assert _inject(client, **body).status_code == 202
        replay = _inject(client, **body)
        assert replay.status_code == 202 and replay.json()["rejoined"] is True, "its retry is the same message"
    html, clip = (bridge._inbox.get_nowait() for _ in range(2))
    assert bridge._inbox.empty()
    for queued, name, kind in ((html, "attachment.bin", "file"), (clip, "attachment.mp4", "video")):
        assert queued["image_base64"] == "" and queued["image_mime"] == "", "no false vision"
        (ref,) = queued["task_metadata"]["chat_attachments"]
        (spec,) = queued["task_metadata"]["chat_attachment_uploads"]
        assert (ref["name"], ref["kind"], spec["label"]) == (name, kind, name)
        assert (spec["size"], spec["sha256"]) == (ref["size"], ref["sha256"])
    rows = _rows(tmp_path)
    assert [(row["text"], row.get("text_placeholder")) for row in rows] == [("(file attached)", True)] * 2


def test_an_inline_photo_beside_files_reaches_the_model_with_them(tmp_path, monkeypatch):
    import queue

    import supervisor.workers as workers
    from tests.test_attachment_staging import _FakeChatAgent, _drive
    from tests.test_chat_inject_attachments import _skill_file
    from tests.test_host_service_api import FakeBridge

    bridge = FakeBridge()
    client = _skill_client(tmp_path, bridge)
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
    scan = _skill_file(tmp_path, "scan.pdf", b"%PDF-1.4 scan")
    response = _inject(client, text="оба", image_base64=base64.b64encode(png).decode("ascii"), image_mime="image/jpeg",
                       attachments=[{"path": str(scan), "name": "scan.pdf"}])
    assert response.status_code == 202, response.text
    message = bridge.messages[0]
    assert message["image_mime"] == "image/png", "vision gets the type the bytes prove, not the transport's label"
    specs = message["task_metadata"]["chat_attachment_uploads"]
    assert [spec["label"] for spec in specs] == ["photo.png", "scan.pdf"], "the photo is staged with the files"

    drive = _drive(tmp_path / "worker")
    monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
    monkeypatch.setattr(workers, "get_event_q", lambda: queue.Queue())
    agent = _FakeChatAgent()
    workers._run_chat_task(agent, 42, "оба", (message["image_base64"], message["image_mime"]),
                           task_metadata=message["task_metadata"])
    images = agent.task["attachment_images"]
    assert [row["label"] for row in images] == ["photo.png"] and images[0]["mime"] == "image/png"
    assert "image_base64" not in agent.task, "staged once, never also injected inline"
    # Alone, the photo uses the same original-retaining staging rail, once.
    _inject(client, text="", image_base64=base64.b64encode(png).decode("ascii"))
    alone = bridge.messages[1]
    spec, = alone["task_metadata"]["chat_attachment_uploads"]
    assert pathlib.Path(spec["path"]).read_bytes() == png
    workers._run_chat_task(agent, 42, "", (alone["image_base64"], alone["image_mime"]),
                           task_metadata=alone["task_metadata"])
    from ouroboros.context import build_user_content
    image, = [block for block in build_user_content(agent.task) if block["type"] == "image_url"]
    assert base64.b64decode(image["image_url"]["url"].split(",", 1)[1]) == png
    assert len(agent.task["attachment_images"]) == 1 and "image_base64" not in agent.task


def test_replay_shows_an_upload_deleted_or_rewritten_since_as_unavailable(files_app, tmp_path, monkeypatch):
    """Replay compares each upload with its recorded stat witness (size and mtime): a delete, a
    replace and a same-size rewrite all show unavailable, and nothing is hashed to say so."""
    import ouroboros.artifacts as artifacts

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    names = ("kept.png", "gone.png", "edited.png", "same.png", "swapped.png")
    kept, gone, edited, same, swapped = (_upload(files_app, name, PNG) for name in names)
    _send_web(bridge, "пять", [kept, gone, edited, same, swapped])
    (row,) = _rows(tmp_path)
    witness = {ref["name"]: ref["mtime_ns"] for ref in row["attachments"]}
    uploads = tmp_path / "uploads"
    (uploads / gone["filename"]).unlink()  # e.g. through the Files API
    (uploads / edited["filename"]).write_bytes(PNG + b"rewritten")
    (uploads / same["filename"]).write_bytes(PNG[:-1] + b"!")  # same size, different bytes, later
    os.utime(uploads / same["filename"], ns=(witness["same.png"] + 10**9, witness["same.png"] + 10**9))
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(PNG)
    os.replace(replacement, uploads / swapped["filename"])  # another file, same size
    os.utime(uploads / swapped["filename"], ns=(witness["swapped.png"] + 10**9, witness["swapped.png"] + 10**9))
    with monkeypatch.context() as patch:
        for owner, name in ((chat_uploads, "open_upload"), (chat_uploads, "measure_upload"), (artifacts, "stream_artifact_file")):
            patch.setattr(owner, name, lambda *_a, **_k: pytest.fail("replay opened or hashed an upload"))
        views = _history(tmp_path)[0]["attachments"]
    assert [view["available"] for view in views] == [True, False, False, False, False], "no stale actions on replay"
    assert views[0] == echoes[0]["attachments"][0] and "url" not in views[1] and views[2]["name"] == "edited.png"
    unwitnessed = {key: value for key, value in row["attachments"][0].items() if key != "mtime_ns"}
    assert chat_uploads.attachment_views([unwitnessed], tmp_path)[0]["available"] is False, "no witness, no claim"


def test_a_failed_web_write_keeps_its_uploads_and_one_retry_hands_a_proven_undispatched_row_over(
        files_app, tmp_path, monkeypatch):
    """A refused canonical write keeps the attachment originals (a failed append may still
    have left bytes, so they stay claimed, never deletable); the sender's retry under the
    same id then lands once. If the row DID land before the failure, this process holds
    positive proof that it never reached dispatch: history says so (``ingress_undispatched``),
    and the ONE explicit same-id retry hands that row over — one row, one queue item bound to
    it; any later retry only rejoins (the same rule as a named skill delivery)."""
    from supervisor import message_bus

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    photo, second = _upload(files_app, "one.png", PNG), _upload(files_app, "two.png", PNG)
    text = "Посмотри\n\n[Attached file: one.png]"
    real = message_bus.log_chat

    def refuse(*_args, **_kwargs):
        raise RuntimeError("canonical message acceptance could not be persisted")

    monkeypatch.setattr(message_bus, "log_chat", refuse)
    with pytest.raises(RuntimeError):
        _send_web(bridge, text, [photo])
    assert _rows(tmp_path) == [] and bridge._inbox.qsize() == 0 and echoes == []
    assert files_app.request("DELETE", "/api/chat/upload", json={"filename": photo["filename"]}).status_code == 409
    monkeypatch.setattr(message_bus, "log_chat", real)
    _send_web(bridge, text, [photo])
    (row,) = _rows(tmp_path)
    assert row["attachments"][0]["upload"] == photo["filename"] and "unavailable" not in row["attachments"][0]
    assert bridge._inbox.qsize() == 1 and len(echoes) == 1

    def land_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("the append's acknowledgement was lost")

    monkeypatch.setattr(message_bus, "log_chat", land_then_fail)
    with pytest.raises(RuntimeError):
        _send_web(bridge, "Второй", [second], cmid="cm-2")
    monkeypatch.setattr(message_bus, "log_chat", real)
    assert bridge._inbox.qsize() == 1 and len(echoes) == 1, "nothing of cm-2 was dispatched or echoed"
    replayed = {row["client_message_id"]: row for row in _history(tmp_path)}
    assert replayed["cm-2"]["ingress_accepted"] is True and replayed["cm-2"]["ingress_undispatched"] is True
    assert "ingress_undispatched" not in replayed["cm-1"]
    bridge._inbox.get_nowait()
    for _ in range(2):  # Send again, then a duplicated tab's Send again
        _send_web(bridge, "Второй", [second], cmid="cm-2")
    rows = _rows(tmp_path)
    assert [row["client_message_id"] for row in rows] == ["cm-1", "cm-2"], "no duplicate row"
    assert [echo["client_message_id"] for echo in echoes] == ["cm-1", "cm-2", "cm-2"] and echoes[1] == echoes[2]
    assert echoes[1]["ingress_accepted"] is True and "ingress_undispatched" not in echoes[1]
    (item,) = [bridge._inbox.get_nowait() for _ in range(bridge._inbox.qsize())]
    assert item["accepted_source_row"] == rows[1], "the one dispatch is the accepted row, not a second message"
    assert message_bus.record_inbound_message(bridge, item, chat_id=1, user_id=1, client_message_id="cm-2",
                                              text=rows[1]["text"], ts="later")["ts"] == rows[1]["ts"]
    assert len(_rows(tmp_path)) == 2, "dequeue validated the witness; it wrote nothing"
    assert not any(row.get("ingress_undispatched") for row in _history(tmp_path)), "handed over: plain saved"
    assert (tmp_path / "uploads" / second["filename"]).exists(), "the accepted row's bytes are kept"


def test_an_unreadable_chat_archive_refuses_the_frame_instead_of_guessing_it_new(files_app, tmp_path, monkeypatch):
    """Absent history is a new message; history that cannot be READ is unknown: the frame is
    refused before any claim, row, queue item or echo, so a redelivery can never log twice.
    A host process's first lookup folds the whole retained chain (``message_ingress._AcceptedIds``)."""
    import ouroboros.utils as utils
    from supervisor import message_ingress

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    photo = _upload(files_app, "one.png", PNG)
    text = "Посмотри\n\n[Attached file: one.png]"
    _send_web(bridge, "первое", [])  # a fresh install has no chat.jsonl: absent, not unreadable
    real = message_ingress.jsonl_chain_handles

    def unreadable(*_args, **_kwargs):
        raise utils.JsonlChainUnreadable("archive segment could not be opened")

    message_ingress.reset_accepted_ids()  # the next host process: nothing folded yet
    monkeypatch.setattr(message_ingress, "jsonl_chain_handles", unreadable)
    with pytest.raises(OSError):
        _send_web(bridge, text, [photo], cmid="cm-2")
    assert len(_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1 and len(echoes) == 1
    assert photo["filename"] in chat_uploads._PENDING, "nothing was claimed"
    monkeypatch.setattr(message_ingress, "jsonl_chain_handles", real)
    _send_web(bridge, text, [photo], cmid="cm-2")
    _send_web(bridge, text, [photo], cmid="cm-2")  # and a redelivery still rejoins
    assert [row["client_message_id"] for row in _rows(tmp_path)][1:] == ["cm-2"] and bridge._inbox.qsize() == 2


def test_worker_staging_copies_only_the_bytes_the_row_records(files_app, tmp_path, monkeypatch):
    """Each staging spec carries its ref's measured identity: bytes swapped under the upload path
    after acceptance (rewritten, or a link to another file) are refused at staging, not staged."""
    from ouroboros.artifacts import stage_task_attachments

    bridge, _echoes = _web_bridge(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    kept, rewritten, linked = (_upload(files_app, name, PNG) for name in ("kept.png", "rewritten.png", "linked.png"))
    _send_web(bridge, "три", [kept, rewritten, linked, {"filename": "0" * 32 + "_ghost.png", "display_name": "ghost.png",
                                                       "mime": "image/png"}])
    specs = bridge._inbox.get_nowait()["task_metadata"]["chat_attachment_uploads"]
    assert specs[3]["path"] == "", "an unavailable ref stages nothing"
    (tmp_path / "uploads" / rewritten["filename"]).write_bytes(JPEG)
    outside = tmp_path / "outside.png"
    outside.write_bytes(JPEG)
    (tmp_path / "uploads" / linked["filename"]).unlink()
    try:
        os.symlink(outside, tmp_path / "uploads" / linked["filename"])
    except OSError:  # no symlink privilege (Windows): the same swap as a plain replacement
        os.replace(outside, tmp_path / "uploads" / linked["filename"])
    (tmp_path / "drive").mkdir()
    manifest = stage_task_attachments(tmp_path / "drive", "task-1", specs)
    assert [(row["status"], row.get("reason", "")) for row in manifest] == [
        ("staged", ""), ("rejected", "copy_failed"), ("rejected", "copy_failed"), ("rejected", "invalid_path")]
    assert pathlib.Path(manifest[0]["abs_path"]).read_bytes() == PNG


# --- custody: an unnamed handoff, an acceptance write that raised, the placeholder mark -------

def test_an_unnamed_inject_keeps_its_copies_once_handoff_began(tmp_path, monkeypatch):
    """Custody passes before dispatch: a dispatch that enqueued and then raised answers 500,
    and the queued message still names copies that exist (never deleted under queued work)."""
    from ouroboros.gateway import host_service
    from tests.test_chat_inject_attachments import _skill_file
    from tests.test_host_service_api import FakeBridge

    bridge = FakeBridge()
    client = _skill_client(tmp_path, bridge)

    def enqueue_then_fail(target, text, **message):
        target.enqueue_local_message(text, **message)
        raise RuntimeError("acknowledgement lost after the handoff")

    monkeypatch.setattr(host_service, "dispatch_accepted_restart", enqueue_then_fail)
    response = _inject(client, text="scan", attachments=[{"path": str(_skill_file(tmp_path, "scan.pdf")), "name": "scan.pdf"}])
    assert response.status_code == 500
    (spec,) = bridge.messages[0]["task_metadata"]["chat_attachment_uploads"]
    assert pathlib.Path(spec["path"]).read_bytes() == b"%PDF-1.4 hello", "the queued work's input is kept"


def test_a_named_acceptance_whose_write_raised_reads_lost_until_one_retry_hands_it_over(tmp_path, monkeypatch):
    """The row landed, then the append raised: the inputs and row stay and the operation reads ``lost``
    (``acceptance_write_failed``), not ``pending`` forever. That is this process's proof the row never
    entered dispatch, so the early rejoin does not answer for it: the next same-id retry reaches the
    named ingress, which hands the row over ONCE (``queued``; the echo shows the row's own refs, the
    retry's identical copies feed staging); every later retry only rejoins."""
    from supervisor import message_bus
    from tests.test_chat_inject_attachments import _skill_file

    bridge = message_bus.LocalChatBridge()
    client = _skill_client(tmp_path, bridge)
    real = message_bus.log_chat

    def land_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("the append's acknowledgement was lost")

    body = {"text": "scan", "client_message_id": "tg:9",
            "attachments": [{"path": str(_skill_file(tmp_path, "scan.pdf")), "name": "scan.pdf"}]}
    monkeypatch.setattr(message_bus, "log_chat", land_then_fail)
    assert _inject(client, **body).status_code == 500
    monkeypatch.setattr(message_bus, "log_chat", real)
    (row,) = _rows(tmp_path)
    assert (tmp_path / "uploads" / row["attachments"][0]["upload"]).exists(), "the accepted bytes are kept"
    state = client.get("/chat/operations/42:tg:9", headers={"X-Skill-Token": "token"}).json()
    assert (state["status"], state["reason"]) == ("lost", "acceptance_write_failed") and bridge._inbox.qsize() == 0
    assert _inject(client, **body).json() == {"ok": True, "status": "queued", "operation_ref": "42:tg:9"}
    assert _inject(client, **body).json()["rejoined"] is True
    assert bridge._inbox.qsize() == 1 and len(_rows(tmp_path)) == 1, "one row, one dispatch"
    item = bridge._inbox.get_nowait()
    (spec,) = item["task_metadata"]["chat_attachment_uploads"]
    assert pathlib.Path(spec["path"]).read_bytes() == b"%PDF-1.4 hello" and spec["sha256"] == row["attachments"][0]["sha256"]
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    message_bus.record_inbound_message(bridge, item, chat_id=42, user_id=42, client_message_id="tg:9",
                                       text=row["text"], ts="later")
    assert frames[0]["attachments"] == chat_uploads.attachment_views(row["attachments"]) and len(_rows(tmp_path)) == 1
    assert client.get("/chat/operations/42:tg:9", headers={"X-Skill-Token": "token"}).json()["status"] != "lost"
    # A write that raised WITHOUT landing leaves no row: the retry is a fresh, ordinary acceptance.
    monkeypatch.setattr(message_bus, "log_chat", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("disk")))
    assert _inject(client, text="later", client_message_id="tg:10").status_code == 500
    monkeypatch.setattr(message_bus, "log_chat", real)
    assert _inject(client, text="later", client_message_id="tg:10").status_code == 202
    assert bridge._inbox.qsize() == 1 and not message_bus.acceptance_undispatched(42, "tg:10")


def test_live_echoes_carry_the_rows_placeholder_mark(files_app, tmp_path, monkeypatch):
    """An echo is the accepted row as history replays it: when the host wrote the text (no words
    were sent), the echo has that text AND ``text_placeholder``; the owner's words never do."""
    from supervisor import message_bus

    bridge, echoes = _web_bridge(tmp_path, monkeypatch)
    photo = _upload(files_app, "one.png", PNG)
    _send_web(bridge, "   ", [photo], image_base64="eA==", image_caption="[user attachment: one.png]")
    _send_web(bridge, "(image attached)", [photo], cmid="cm-2")
    host, owner = _rows(tmp_path)
    assert (host["text"], host.get("text_placeholder")) == ("[user attachment: one.png]", True)
    assert "text_placeholder" not in owner, "the owner's literal words stay the owner's"
    assert [(echo["content"], echo.get("text_placeholder")) for echo in echoes] == [
        ("[user attachment: one.png]", True), ("(image attached)", None)]
    replayed = {row["client_message_id"]: row for row in _history(tmp_path)}
    assert [(replayed[cmid]["text"], replayed[cmid].get("text_placeholder")) for cmid in ("cm-1", "cm-2")] == [
        (echo["content"], echo.get("text_placeholder")) for echo in echoes]

    frames = []
    bridge._broadcast_fn = frames.append
    refs = host["attachments"]
    for cmid, raw in (("host-1", ""), ("host-2", "кадр")):
        message_bus.record_inbound_message(
            bridge, {"text": raw, "source": "skill:telegram", "task_metadata": {"chat_attachments": refs}},
            chat_id=42, user_id=42, client_message_id=cmid, text=raw or "(image attached)", ts="2026-10-05T00:00:00Z")
    assert [(frame["content"], frame.get("text_placeholder")) for frame in frames] == [
        ("(image attached)", True), ("кадр", None)]
    replayed = {row["client_message_id"]: row for row in _history(tmp_path, 42)}
    assert [(replayed[cmid]["text"], replayed[cmid].get("text_placeholder")) for cmid in ("host-1", "host-2")] == [
        (frame["content"], frame.get("text_placeholder")) for frame in frames]
