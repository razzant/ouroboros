"""Storage Remove over a REAL server and a REAL network: an admitted 202 whose body is cut mid-flight.

A small relay sits between the browser and the isolated server. It lets the POST reach the server, which starts the
removal, then answers 202 with a Content-Length larger than the body it sends and closes: Chromium resolves the
fetch and fails reading the body (ERR_CONTENT_LENGTH_MISMATCH). The tray must stay with the lost-answer sentence
(nothing is sent again) and say what the run did; a whole 202 keeps the ordinary flow.
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import threading

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as _server

direct_server_with_data = _server
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

OUT = pathlib.Path(os.environ.get("QA_OUT", "")) if os.environ.get("QA_OUT") else None
FOLDERS = 3
LOST = "The answer to Remove was lost; nothing is sent again."


def _read_head(sock):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(65536)
        if not chunk:
            break
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    return head, rest


def _dechunk(raw):
    body = b""
    while raw:
        size, _, raw = raw.partition(b"\r\n")
        count = int(size.split(b";")[0] or b"0", 16)
        if not count:
            break
        body, raw = body + raw[:count], raw[count + 2:]
    return body


class Relay:
    """Forwards every request to the server; cuts the answer to POST /api/storage/free when `cut` is set.

    Every socket is bounded (a stalled peer times out) and owned: `close()` shuts them all and joins the threads."""

    WAIT = 30  # seconds one peer may stall while a request or an answer is read

    def __init__(self, upstream_port, cut):
        self.upstream, self.cut = upstream_port, cut
        self.free_posts = 0
        self.cuts = 0
        self._lock = threading.Lock()
        self._socks, self._threads = set(), []
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(64)
        self.port = self.listener.getsockname()[1]
        self._spawn(self._accept)

    def _own(self, sock):
        with self._lock:
            self._socks.add(sock)
        return sock

    def _spawn(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def close(self):
        with self._lock:
            socks, threads = [self.listener, *self._socks], list(self._threads)
        for sock in socks:
            for shut in (lambda: sock.shutdown(socket.SHUT_RDWR), sock.close):
                try:
                    shut()
                except OSError:
                    pass
        for thread in threads:
            thread.join(timeout=10)
        assert not [thread for thread in threads if thread.is_alive()], "the relay leaked a thread"

    def _accept(self):
        while True:
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            self._spawn(self._serve, self._own(client))

    def _swap(self, raw):
        return raw.replace(f"127.0.0.1:{self.port}".encode(), f"127.0.0.1:{self.upstream}".encode())

    def _serve(self, client):
        out = None
        try:
            client.settimeout(self.WAIT)
            head, rest = _read_head(client)
            if not head:
                return
            lines = self._swap(head).split(b"\r\n")
            upgrade = any(line.lower().startswith(b"upgrade:") for line in lines)
            length = next((int(line.split(b":")[1]) for line in lines if line.lower().startswith(b"content-length:")), 0)
            while len(rest) < length:
                chunk = client.recv(65536)
                if not chunk:  # the browser went away mid-request: nothing to forward
                    return
                rest += chunk
            if not upgrade:
                lines = [line for line in lines if not line.lower().startswith(b"connection:")] + [b"Connection: close"]
            out = self._own(socket.create_connection(("127.0.0.1", self.upstream), timeout=self.WAIT))
            out.sendall(b"\r\n".join(lines) + b"\r\n\r\n" + rest)
            target = lines[0].split(b" ")
            if target[1] == b"/api/storage/free" and target[0] == b"POST":
                self.free_posts += 1
                if self.cut:
                    self._cut_answer(out, client)
                    return
            if upgrade:
                client.settimeout(None)  # a socket upgrade idles; close() ends it
                out.settimeout(None)
                self._spawn(self._pipe, client, out)
            self._pipe(out, client)
        except OSError:
            pass
        finally:
            for sock in (client, out):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

    @staticmethod
    def _pipe(source, sink):
        try:
            while chunk := source.recv(65536):
                sink.sendall(chunk)
        except OSError:
            pass
        finally:
            for sock in (source, sink):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _cut_answer(self, out, client):
        head, rest = _read_head(out)
        while chunk := out.recv(65536):
            rest += chunk
        out.close()
        lines = head.split(b"\r\n")
        body = _dechunk(rest) if any(line.lower() == b"transfer-encoding: chunked" for line in lines) else rest
        lines = [line for line in lines if not line.lower().startswith((b"content-length:", b"transfer-encoding:"))]
        lines.append(f"Content-Length: {len(body) + 200}".encode())
        assert lines[0].split(b" ")[1] == b"202", lines[0]
        self.cuts += 1
        client.sendall(b"\r\n".join(lines) + b"\r\n\r\n" + body)  # then closed: 200 bytes short


def _seed(data):
    from ouroboros.task_results import write_task_result

    paths = []
    for index in range(FOLDERS):
        task_id = f"w-{index:03d}"
        folder = data / "task_drives" / task_id
        folder.mkdir(parents=True)
        (folder / "s.bin").write_bytes(b"s" * 64)
        write_task_result(data, task_id, "completed", title=f"Sample {index:03d}", child_drive_root=str(folder))
        paths.append(folder)
    return paths


def _shot(page, name):
    if OUT:
        OUT.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(OUT / f"{name}.png"))


@pytest.mark.parametrize("cut", [True, False], ids=["cut", "whole"])
def test_cut(direct_server_with_data, cut):
    from playwright.sync_api import sync_playwright

    kind = "cut" if cut else "whole"
    server = direct_server_with_data
    data = server["data_dir"]
    server["stop_server"]()
    paths = _seed(data)
    server["start_server"]()
    relay = Relay(int(server["url"].rsplit(":", 1)[1]), cut)
    events, pool_pages, failed, console = [], [], [], []  # events: the storage requests, in the order the page sent them
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1232, "height": 880})
            page.on("request", lambda req: events.append((req.method, req.url.split("/api/", 1)[1], req.post_data))
                    if "/api/storage" in req.url else None)
            page.on("response", lambda res: pool_pages.append(res) if "view=pool" in res.url else None)
            page.on("requestfailed", lambda req: failed.append((req.method, req.url, req.failure)))
            page.on("console", lambda msg: console.append(f"{msg.type}: {msg.text}"))
            try:
                page.goto(f"http://127.0.0.1:{relay.port}/", wait_until="domcontentloaded")
                page.wait_for_selector("#page-chat")
                page.click('[data-nav-page="dashboard"]')
                page.click('[data-dashboard-tab="storage"]')
                page.wait_for_selector("[data-storage-panel]:not([hidden])")
                page.click("[data-storage-door]")
                page.wait_for_selector('[data-storage-tray] [data-free-go]:not([aria-disabled="true"])', timeout=120_000)
                assert f"Remove {FOLDERS} items" in page.locator("[data-storage-tray]").inner_text()
                page.click("[data-storage-tray] [data-free-go]")
                if cut:
                    page.wait_for_selector(f"text={LOST}", timeout=30_000)
                    _shot(page, "cut-1-lost-answer")
                # The recovered state, not just the Done button: every folder removed and none still being deleted.
                page.wait_for_function(
                    "() => { const t = document.querySelector('[data-storage-tray]').innerText;"
                    f" return t.includes('Removed {FOLDERS} items.') && !t.includes('still being deleted'); }}", timeout=120_000)
                page.wait_for_selector("[data-storage-tray] [data-free-done]", timeout=30_000)
                tray = page.locator("[data-storage-tray]").inner_text()
                _shot(page, f"{kind}-2-settled")
                previewed = {item["id"] for res in pool_pages for item in res.json()["items"]}
                page.locator('[data-free-disclosure="each"] > summary').click()
                page.wait_for_selector("[data-outcome]")
                outcomes = page.locator("[data-outcome]").evaluate_all("rows => rows.map((row) => row.getAttribute('data-outcome'))")
            except BaseException:
                _shot(page, f"{kind}-failure")
                raise
            finally:
                if OUT:
                    OUT.mkdir(parents=True, exist_ok=True)
                    (OUT / f"{kind}-console.json").write_text(json.dumps(
                        {"console": console, "requestfailed": failed, "events": events}, indent=1, default=str))
                browser.close()
        posts = [json.loads(body) for method, url, body in events if method == "POST" and url == "storage/free"]
        assert len(posts) == 1 and relay.free_posts == 1, (posts, relay.free_posts)
        assert sorted(posts[0]) == ["items", "plan"]
        expected = {f"direct:w-{index:03d}" for index in range(FOLDERS)}
        assert previewed == expected and set(posts[0]["items"]) == previewed and len(posts[0]["items"]) == FOLDERS, (previewed, posts[0])
        assert outcomes == ["removed"] * FOLDERS, outcomes
        assert "Remove 3 items" not in tray and "Nothing was removed" not in tray, tray
        assert not any(path.exists() for path in paths)
        if cut:
            assert relay.cuts == 1
            assert any(method == "POST" and "ERR_CONTENT_LENGTH_MISMATCH" in str(failure) for method, url, failure in failed
                       if url.endswith("/api/storage/free")), failed
            after = events[[i for i, event in enumerate(events) if event[:2] == ("POST", "storage/free")][0] + 1:]
            assert any(method == "GET" and url.split("?")[0] == "storage" for method, url, _ in after), "no status read after the lost answer"
            assert not any(method == "POST" and url == "storage/free" for method, url, _ in after), "a second Remove on the same plan"
        else:
            assert not failed or all("storage/free" not in req_url for _, req_url, _ in failed), failed
    finally:
        relay.close()
