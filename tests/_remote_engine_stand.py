"""A remote engine stand for tests (``ouroboros.remote_engines``): loaded as a pytest plugin, it provides ``host``.

The process boundary is real: a stand-in for OpenSSH (a script that records its argv, runs the "remote" command with
``/bin/sh`` on this machine and holds ``-L`` forwards with a socket relay) and a loopback HTTP engine that answers
only its bearer token. That stand-in is a POSIX script run through ``/bin/sh``, so on Windows every test that takes
``host`` is skipped (the code under test is the same there; only this stand-in is not). The "host" is a temporary folder: its engine home holds the descriptor and token a running
engine writes, its ``remote_root`` receives the staged copies. No real SSH, host, engine or account is used.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from ouroboros import remote_engines
from ouroboros.gateways import claudexor as wire

STORE = "store-remote-1"
TOKEN = "remote-engine-secret-token"

_FAKE_SSH = r'''
import json, os, socket, sys, threading
args = sys.argv[1:]
options, forward, no_command, destination, rest = [], "", False, "", []
i = 0
while i < len(args):
    a = args[i]
    if a == "-o":
        options.append(args[i + 1]); i += 2; continue
    if a == "-L":
        forward = args[i + 1]; i += 2; continue
    if a in ("-T", "-N"):
        no_command = no_command or a == "-N"; i += 1; continue
    destination, rest = a, args[i + 1:]
    break
with open(os.environ["FAKE_SSH_LOG"], "a") as log:
    log.write(json.dumps({"destination": destination, "forward": forward, "no_command": no_command,
                          "command": rest, "options": options}) + "\n")
if destination.startswith("deny@"):
    sys.stderr.write("Permission denied (publickey).\n"); sys.exit(255)
if destination.startswith("unknown@"):
    sys.stderr.write("Host key verification failed.\n"); sys.exit(255)
if forward:
    bind, lport, rhost, rport = forward.rsplit(":", 3)
    server = socket.socket(); server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind, int(lport))); server.listen(32)
    def pump(src, dst):
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try: s.shutdown(socket.SHUT_RDWR)
                except OSError: pass
    while True:
        conn, _ = server.accept()
        try:
            upstream = socket.create_connection((rhost.strip("[]"), int(rport)))
        except OSError:
            conn.close(); continue
        threading.Thread(target=pump, args=(conn, upstream), daemon=True).start()
        threading.Thread(target=pump, args=(upstream, conn), daemon=True).start()
os.execv("/bin/sh", ["/bin/sh", "-c", rest[-1]])
'''


class _Engine(BaseHTTPRequestHandler):
    """A loopback engine: bearer-checked, store-proving, recording every request it answers. It keeps the runs it
    started (idempotent by key), their task groups, run folders and artifacts, and its account roster, as the
    engine's own routes answer them (``GET /v2/runs``, ``/v2/task-groups/:id/runs``, ``/v2/runs/:id/artifacts``
    and ``/artifacts/<path>``, ``/v2/credential-profiles``)."""

    def log_message(self, *_args):
        pass

    def _reply(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _bytes(self, data, kind="application/octet-stream"):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _summary(self, run):
        return {"runId": run["runId"], "jobId": run["runId"], "state": self.server.state.get("run_states", {}).get(
            run["runId"], self.server.state.get("run_state", "completed")), "createdAt": run["createdAt"],
            "project": {"kind": "project", "root": run["root"]}, "executionRoot": run["executionRoot"],
            "prompt": run["prompt"]}

    def _handle(self, method):
        state = self.server.state
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        body = json.loads(raw) if raw and method != "PUT" else None
        url = urlparse(self.path)
        query = parse_qs(url.query)
        state["calls"].append({"method": method, "path": url.path, "query": query, "body": body,
                               "key": self.headers.get("Idempotency-Key")})
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            return self._reply(401, {"code": "unauthorized", "message": "bad token"})
        runs = state.setdefault("runs", [])
        if url.path == "/v2/handshake":
            return self._reply(200, {"compatible": True, "protocolMajor": wire.CLAUDEXOR_PROTOCOL_MAJOR,
                                     "engine": {"version": "5.1.0", "sha": "7c611ec5"}, "servingMode": "normal",
                                     "storeId": state["store"]})
        if url.path == "/v2/harnesses":
            return self._reply(200, {"harnesses": [{"id": "codex"}]})
        if url.path == "/v2/operations":
            served = [{"method": "GET", "path": "/v2/task-groups/:id/runs"}] if state.get("task_groups", True) else []
            if state.get("run_lookup"):  # the noncreating read of an accepted start (``delegate_accepted``)
                served.append({"method": "POST", "path": "/v2/runs/lookup"})
            return self._reply(200, {"operations": served})
        if url.path == "/v2/runs/lookup" and method == "POST":
            # The engine's own contract (``tests/test_delegate_accepted_lookup.Engine``): the command the key holds,
            # read with that start's own body; the same key under another body is a conflict, never a match.
            key = self.headers.get("Idempotency-Key")
            held = next((run for run in runs if run["key"] == key), None)
            if held is None:
                return self._reply(200, {"status": "not_found", "storeId": state["store"]})
            if held["body"] != body:
                return self._reply(409, {"code": "idempotency_conflict", "message": "the key holds another request"})
            return self._reply(200, {"status": "found", "storeId": state["store"], "handle": {
                "jobId": held["runId"], "runId": held["runId"], "runDir": f"/host/runs/{held['runId']}",
                "state": state.get("run_states", {}).get(held["runId"], state.get("run_state", "completed"))}})
        if url.path == "/v2/credential-profiles":
            if state.get("profiles_error"):
                return self._reply(503, {"code": "accounts_unavailable", "message": "roster unreadable"})
            return self._reply(200, state.get("profiles", {"profiles": [], "accountPools": []}))
        if url.path == "/v2/model-sources":
            return self._reply(200, {"sources": [{"id": "codex", "label": "Codex", "credentialHarness": "codex"}]})
        if url.path == "/v2/model-sources/codex/models":
            return self._reply(200, {"source": "codex", "credentialProfileId": "remote-main",
                                     "models": [{"id": "gpt-6-remote"}]})
        if url.path == "/v2/trust" and method == "GET":
            root = query.get("repoRoot", [""])[0]
            return self._reply(200, {"entries": [{"repoRoot": root, "path": root, "allowFullAccess": False,
                                                  "accessDefault": "workspace_write", "source": "default"}]})
        if url.path == "/v2/trust" and method == "POST":
            return self._reply(200, {"repoRoot": body["repoRoot"], "path": body["repoRoot"], "allowFullAccess": True,
                                     "accessDefault": "workspace_write"})
        if url.path == "/v2/runs" and method == "POST":
            if state.get("refuse_start"):  # the engine's own answer to the POST, ``(status, code)``: nothing starts
                status, code = state["refuse_start"]
                return self._reply(status, {"code": code, "message": f"the engine refused this start ({code})"})
            key = self.headers.get("Idempotency-Key")
            prior = next((run for run in runs if run["key"] == key), None)
            if prior is None:
                scope, execution = body.get("scope") or {}, body.get("execution") or {}
                prior = {"runId": f"run-remote-{len(runs) + 1}", "key": key, "taskGroupId": body.get("taskGroupId"),
                         "root": scope.get("root"), "executionRoot": execution.get("workspaceRoot") or scope.get("root"),
                         "prompt": body.get("prompt"), "createdAt": f"2026-11-11T00:00:{len(runs):02d}Z", "body": body}
                runs.append(prior)
                if state.get("lose_start_answer"):  # accepted, then the answer is lost on the way back
                    self.close_connection = True
                    return
            return self._reply(200, {"runId": prior["runId"], "runDir": f"/host/runs/{prior['runId']}",
                                     "storeId": state["store"]})
        if url.path == "/v2/runs" and method == "GET":
            return self._reply(200, {"runs": [self._summary(run) for run in reversed(runs)], "nextCursor": None})
        if url.path.startswith("/v2/task-groups/") and url.path.endswith("/runs"):
            expected = query.get("expectedStoreId", [""])[0]
            if expected and expected != state["store"]:
                return self._reply(409, {"code": "store_identity_mismatch", "message": "another store"})
            group = url.path.split("/")[3]
            members = [{"runId": run["runId"], "jobId": run["runId"], "createdAt": run["createdAt"]}
                       for run in runs if run.get("taskGroupId") == group]
            return self._reply(200, {"taskGroupId": group, "storeId": state["store"], "members": members,
                                     "hasMore": False, "nextCursor": None})
        if url.path.startswith("/v2/runs/") and url.path.endswith("/control"):
            if state.get("hang_control") is not None:  # accepted; no answer until the test releases it
                state["hang_control"].wait(30)
            return self._reply(200, {"accepted": True, "status": "cancelling"})
        if url.path.startswith("/v2/uploads"):  # a model request's upload: what reached the engine is kept
            uploads = state.setdefault("uploads", {})
            parts = url.path.split("/")
            if url.path == "/v2/uploads" and method == "POST":
                upload = uploads.setdefault(self.headers.get("Idempotency-Key"), {
                    "uploadId": f"up-{len(uploads) + 1}", "bytes": None, "size": body.get("sizeBytes")})
                return self._reply(200, {"uploadId": upload["uploadId"], "state": "open"})
            upload = next((item for item in uploads.values() if item["uploadId"] == parts[3]), None)
            if upload is None:
                return self._reply(404, {"code": "upload_not_found", "message": url.path})
            if method == "PUT":
                upload["bytes"] = raw
                return self._reply(200, {"uploadId": upload["uploadId"], "state": "uploaded"})
            if url.path.endswith("/finalize"):
                import hashlib

                return self._reply(200, {"resourceId": f"res-{upload['uploadId']}", "purpose": "model",
                                         "sha256": "sha256:" + hashlib.sha256(upload["bytes"] or b"").hexdigest(),
                                         "sizeBytes": len(upload["bytes"] or b"")})
            return self._reply(200, {"uploadId": upload["uploadId"],
                                     "state": "uploaded" if upload["bytes"] is not None else "open"})
        if url.path.startswith("/v2/model-operations/") and url.path.endswith("/control"):
            return self._reply(200, {"id": url.path.split("/")[3], "state": "cancelling"})
        if url.path.startswith("/v2/runs/") and url.path.endswith("/artifacts"):
            run_id = url.path.split("/")[3]
            if state.get("artifacts_list_error"):
                return self._reply(404, {"code": "not_found", "message": "no artifact listing"})
            rows = [{"path": path, "kind": "file", "bytes": len(data) if isinstance(data, bytes) else 1}
                    for (rid, path), data in state.get("artifacts", {}).items() if rid == run_id]
            return self._reply(200, {"runId": run_id, "artifacts": rows})
        if url.path.startswith("/v2/runs/") and "/artifacts/" in url.path:
            run_id, _, path = url.path[len("/v2/runs/"):].partition("/artifacts/")
            artifact = state.get("artifacts", {}).get((run_id, path))
            if artifact is None:
                return self._reply(404, {"code": "artifact_not_found", "message": path})
            if isinstance(artifact, dict) and artifact.get("status"):
                return self._reply(int(artifact["status"]), {"code": artifact["code"], "message": artifact["code"]})
            return self._bytes(artifact)
        if url.path.startswith("/v2/runs/"):
            run_id = url.path.split("/")[3]
            summary = {"state": state.get("run_states", {}).get(run_id, state.get("run_state", "completed")),
                       "runDir": f"/host/runs/{run_id}", "workProduct": {"kind": "git", "path": "final/patch.diff"},
                       **state.get("run_details", {}).get(run_id, {})}
            return self._reply(200, {"summary": summary})
        return self._reply(404, {"code": "not_modeled", "message": url.path})

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_PUT(self):
        self._handle("PUT")


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A fake OpenSSH on a real process boundary, a running loopback engine and its host folders."""
    if sys.platform == "win32":
        pytest.skip("the stand-in OpenSSH is a POSIX script that runs the remote command through /bin/sh")
    tools = tmp_path / "bin"
    tools.mkdir()
    ssh = tools / "ssh"
    ssh.write_text(f"#!{sys.executable}\n{_FAKE_SSH}", encoding="utf-8")
    ssh.chmod(0o755)
    log = tmp_path / "ssh.log"
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    monkeypatch.setattr(remote_engines, "_ssh_executable", lambda: str(ssh))
    state = {"calls": [], "store": STORE}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    server.state = state
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    engine_home = tmp_path / "host" / "engine"
    (engine_home / "daemon").mkdir(parents=True)
    token = engine_home / "daemon" / "token"
    token.write_text(TOKEN, encoding="utf-8")
    descriptor = engine_home / "daemon" / "control-api.json"
    descriptor.write_text(json.dumps({"host": "127.0.0.1", "port": server.server_address[1],
                                      "tokenPath": str(token)}), encoding="utf-8")
    remote_root = tmp_path / "host" / "runs"
    workspace = tmp_path / "host" / "project"
    workspace.mkdir(parents=True)
    rows = [{"name": "vps", "label": "Build box", "ssh_host": "ouroboros@vds", "engine_home": str(engine_home),
             "remote_root": str(remote_root), "workspaces": [str(workspace)]}]
    monkeypatch.setenv(remote_engines.ENGINE_CONNECTIONS_KEY, json.dumps(rows))

    def ssh_calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    def set_rows(new_rows):
        monkeypatch.setenv(remote_engines.ENGINE_CONNECTIONS_KEY, json.dumps(new_rows))

    yield SimpleNamespace(state=state, rows=rows, set_rows=set_rows, ssh_calls=ssh_calls, engine_home=engine_home,
                          remote_root=remote_root, workspace=workspace, descriptor=descriptor, server=server)
    for name in ("vps", "other"):
        remote_engines.retire(name)
    server.shutdown()
    server.server_close()


def tunnel_pids(name: str) -> list:
    published = remote_engines.published(name)
    return [int(published["pid"])] if published and published.get("pid") else []


def alive(pid: int) -> bool:
    """A live process; an exited child its parent has not reaped yet (a zombie) is not. Observed through the platform
    seam, never ``os.kill(pid, 0)`` (on Windows that delivers a console event to the pid's whole group)."""
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.process_containment import pid_is_zombie

    return pid_is_alive(pid) and not pid_is_zombie(pid)
