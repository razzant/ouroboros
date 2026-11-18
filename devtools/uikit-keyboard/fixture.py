"""Read-only loopback fixture for the checked-out SPA. No running product or account."""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import runpy
import select
import socketserver
import threading
from urllib.parse import parse_qs, unquote, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEB = ROOT / "web"
CASES = ("main-reading", "main-latest", "project-ready", "project-held")
PROJECT = dict(id="trace-project", name="Sample drawing", chat_id=101, lifecycle="active",
               visible_revision=0, last_message_at="2026-11-01T00:59:00Z")
CARD = 7


def packet(history, room):
    words = [f"Synthetic room {room}, entry {i}. " + "A note about the sample drawing. " * 2
             for i in range(10)]
    page = history(words, start=0 if room == 1 else 20)
    for row in page["messages"]:
        row["chat_id"] = room
    if room == 1:
        card = page["messages"][CARD]
        card.update(row_address=f"row:1@{card['ts']}#000000000008", role="system",
                    system_type="project_completion_summary", task_id="motion-sample",
                    project_id=PROJECT["id"], project_name=PROJECT["name"], task_name="Synthetic drawing",
                    completion_answer="\n\n".join(f"Sample paragraph {i}. The blue circle is inside the square."
                                                  for i in range(1, 9)))
    return page


def inputs(root):
    shared = runpy.run_path(str(root / "tests/_phone_room_shared.py"))
    sdk = shared["telegram_sdk"](insets={"top": 47, "bottom": 34, "left": 20, "right": 12}, content={"top": 46})
    pages = {str(room): packet(shared["history"], room) for room in (1, PROJECT["chat_id"])}
    data = json.dumps([PROJECT, sdk, pages], sort_keys=True, separators=(",", ":")).encode()
    return sdk, pages, hashlib.sha256(data).hexdigest()


def page_html(template, sdk, case):
    # The sidecar's already-open SPA and the production host adapter. Keep its viewport meta first.
    if '<html lang="en"' not in template or '<script src="/static/theme.js">' not in template:
        raise ValueError("SPA entry changed")
    src = template.replace('<html ', '<html data-ouroboros-host="telegram" ', 1)
    project = case.startswith("project-")
    config = {"scenario": case, "projectId": PROJECT["id"], "room": PROJECT["chat_id"] if project else 1,
              "prefix": f"pchat-{PROJECT['id']}" if project else "chat", "anchorKey": f"chat:{CARD + (20 if project else 0)}"}
    boot = f"<script>{sdk}</script>\n<script>window.traceConfig={json.dumps(config)};</script>\n"
    return src.replace('<script src="/static/theme.js">',
                       boot + '<script src="/probe/recorder.js"></script>\n<script src="/static/theme.js">', 1)


def api(path, query, pages):
    if path == "/api/onboarding":
        return 204, None
    if path == "/api/chat/history":
        room = query.get("chat_id", ["1"])
        if len(room) == 1 and set(query) <= {"chat_id"} and room[0] in pages:
            return 200, pages[room[0]]
        return 404, {}
    if path == "/api/projects":
        matches = (query.get("id", [PROJECT["id"]]) == [PROJECT["id"]] and
                   query.get("chat_id", [str(PROJECT["chat_id"])]) == [str(PROJECT["chat_id"])])
        return 200, {"projects": [PROJECT] if matches else []}
    bodies = {
        "/api/state": {"sha": "", "supervisor_ready": True, "active_chat_activities": [],
                       "projects": [PROJECT], "project_chat_ids": [PROJECT["chat_id"]], "task_bindings": {}},
        "/api/health": {"ok": True, "version": "fixture"},
        "/api/ui/i18n": {"language": "en", "entries": {}}, "/api/ui/preferences": {},
        "/api/settings": {"settings": {}}, "/api/ui/contributions": {"contributions": []},
        "/api/model-catalog": {"items": [], "model_sources": [], "errors": []},
        "/api/tasks": {"tasks": []}, "/api/widgets": {"ui_tabs": []},
        "/api/schedules": {"schedules": []}, "/api/update/status": {"managed": False},
    }
    return (200, bodies[path]) if path in bodies else (404, {})


def static_asset(path):
    if not path.startswith("/static/"):
        return None
    file = (WEB / unquote(path[len("/static/"):])).resolve()
    if WEB.resolve() not in file.parents or not file.is_file():
        return None
    if file.suffix not in {".js", ".css", ".png", ".jpg", ".svg", ".woff", ".woff2", ".ttf", ".json"}:
        return None
    return file


class Fixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def respond(self, status, body=b"", mime="application/json", headers=()):
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def note(self, case, name):
        if case in CASES:
            with self.server.lock:
                self.server.reads[case][name] += 1

    def do_GET(self):
        origin = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != origin:
            return self.respond(421)
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        cookie = SimpleCookie(self.headers.get("Cookie", "")).get("motion_case")
        case = cookie.value if cookie else None
        if url.path == "/":
            case = query.get("case", [""])[0]
            if case not in CASES:
                return self.respond(404)
            headers = [("Set-Cookie", f"motion_case={case}; Path=/; HttpOnly; SameSite=Strict"),
                       ("Content-Security-Policy", f"default-src 'self'; connect-src 'self' ws://{origin}; "
                        "script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                        "img-src 'self' data: blob:; object-src 'none'; base-uri 'none'")]
            self.note(case, "document")
            return self.respond(200, self.server.pages[case], "text/html; charset=utf-8", headers)
        if url.path == "/probe/recorder.js":
            return self.respond(200, self.server.recorder, "text/javascript")
        if url.path == "/ws" and self.headers.get("Origin") == f"http://{origin}":
            return self.heartbeat(case)
        if url.path == "/probe/reads" and self.client_address[0] == "127.0.0.1":
            with self.server.lock:
                body = {"cases": self.server.reads, "fixtureSha256": self.server.fixture_digest}
                return self.respond(200, json.dumps(body).encode())
        if url.path.startswith("/api/"):
            status, data = api(url.path, query, self.server.histories)
            if status < 300:
                name = url.path
                if name == "/api/chat/history":
                    name += "/" + query.get("chat_id", ["1"])[0]
                if name == "/api/projects" and "id" in query:
                    name += "/by-id"
                self.note(case, name)
            else:
                self.note(case, "unserved-api-reads")
            return self.respond(status, json.dumps(data).encode() if data is not None else b"")
        file = static_asset(url.path)
        if file:
            if url.path in {"/static/app.js", "/static/style.css"}:
                self.note(case, url.path)
            return self.respond(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        self.respond(404)

    def heartbeat(self, case):
        # The only application frame on this connection is a fixture heartbeat. No write handlers.
        nonce = self.headers.get("Sec-WebSocket-Key", "")
        if case not in CASES or self.headers.get("Upgrade", "").lower() != "websocket" or len(nonce) != 24:
            return self.respond(400)
        accept = hashlib.sha1((nonce + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", base64.b64encode(accept).decode())
        self.end_headers()
        self.close_connection = True
        beat = b'{"type":"heartbeat"}'
        def send(op, data):
            self.wfile.write(bytes([0x80 | op, len(data)]) + data)
        try:
            send(1, beat)
            self.note(case, "/ws")
            self.connection.settimeout(3)
            while True:
                if not select.select([self.connection], [], [], 8)[0]:
                    send(1, beat)
                    continue
                header = self.rfile.read(2)
                if len(header) != 2 or not header[1] & 0x80:
                    return
                op, size = header[0] & 15, header[1] & 127
                if size in (126, 127):
                    size = int.from_bytes(self.rfile.read(2 if size == 126 else 8), "big")
                if size > 65536 or (op >= 8 and size > 125):
                    return
                mask, data = self.rfile.read(4), self.rfile.read(size)
                if len(mask) != 4 or len(data) != size:
                    return
                if op in (8, 9):
                    send(8 if op == 8 else 10, bytes(v ^ mask[i % 4] for i, v in enumerate(data)))
                if op == 8:
                    return
        except OSError:
            pass

    def do_POST(self):
        # The read-receipt owner may acknowledge a paint. The synthetic room's revision is zero.
        if urlsplit(self.path).path == "/api/ui/preferences" and self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}":
            self.close_connection = True
            self.respond(200, b"{}", headers=[("Connection", "close")])
        else:
            self.close_connection = True
            self.respond(405, headers=[("Connection", "close")])


class LoopbackServer(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind names the bound address with socket.getfqdn, a reverse lookup this fixture never
        # uses; on some hosts it waits on DNS. Only 127.0.0.1 is ever served.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port_file", type=Path)
    args = parser.parse_args()
    sdk, pages, digest = inputs(ROOT)
    template = (WEB / "index.html").read_text(encoding="utf-8")
    with LoopbackServer(("127.0.0.1", 0), Fixture) as server:
        server.pages = {case: page_html(template, sdk, case).encode() for case in CASES}
        server.histories = pages
        server.fixture_digest = digest
        server.recorder = (HERE / "recorder.js").read_bytes()
        server.lock = threading.Lock()
        server.reads = defaultdict(Counter)
        pending = args.port_file.with_suffix(".tmp")
        pending.write_text(str(server.server_port))
        pending.replace(args.port_file)
        server.serve_forever()


if __name__ == "__main__":
    main()
