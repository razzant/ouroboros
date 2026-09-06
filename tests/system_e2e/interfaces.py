"""Suite actors beyond the loopback stub models (plan §8).

``FakeClaudexorDaemon`` (landed with the Ф4 wave-3b delegated-transport lane) is a
loopback claudexord imitation serving the EXACT client contract this tree's
``ouroboros/gateways/claudexor.py`` speaks: the protocol-3 authenticated handshake,
the capability/quota answers ``subagent_route_health.route_health`` reads, project
registration with Idempotency-Key, ``POST /v2/runs`` with the engine's replay
check (same key + byte-identical body → the ORIGINAL handle; same key + different
digest → 409 ``idempotency_conflict``), run detail with the ``summary`` facts the
custody settler consumes, the cancel control verb, and (wave 4) the interactive
question surface — ``pendingInteractions`` on the detail plus the
``POST /v2/runs/:id/interactions/:iid/answer`` verb ``delegate_answer`` speaks,
with its typed delivered/already_resolved/rejected statuses. Behavior is scripted
PER RUN by markers in the POSTed prompt (success / hang / typed refusal / ask)
plus the pinned-profile refusal, and (the mutating wave) the applied facts a
WRITING run produces: the edits themselves, made inside the private execution
snapshot the start body names, and the ``attempts/<id>/attempt.yaml`` containment
record ``gateways/claudexor.py::attempt_containment`` reads. So one daemon serves
every delegated-transport scenario without a second boot. It records every request
(method, path, idempotency key, body) for wire-truth assertions.

``PlaywrightUIClient`` stays an interface stub until the gateway/UI-truth wave
lands: instantiating it is a scenario bug, and it refuses loudly.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

_NOT_LANDED = (
    "{name} is an interface stub: its implementation lands with the {lane} wave of "
    "the Ф4 integration suite (plan §8). Write the scenario against this surface, "
    "but do not enable it before the lane lands — see tests/system_e2e/ and "
    "docs/v7next/LEDGER_CORRECTIONS.md (F4 lane 1)."
)

# Prompt markers a scenario plants to script ONE run's behavior. Chosen so they can
# ride inside an ordinary delegate_start prompt verbatim.
FAKE_HANG_MARKER = "[FAKE:HANG]"       # the run never reaches a terminal state
FAKE_REFUSE_MARKER = "[FAKE:REFUSE]"   # the start POST is refused 400, typed
FAKE_ASK_MARKER = "[FAKE:ASK]"         # the run asks ONE question and waits for the answer
# The mutating marker: the run EDITS the workspace its start body was given, exactly
# as a real harness would, and records the applied containment facts of its attempt.
# Nothing else in the fake changes for it — a mutating run is an ordinary run whose
# harness happened to write, which is the shape the integration path must survive.
FAKE_MUTATE_MARKER = "[FAKE:MUTATE]"

# The edits a [FAKE:MUTATE] run makes when a scenario does not script its own:
# one NEW untracked text file (eligible for capture) and one rewrite of a tracked
# file — the two classes ``workspace_patch_capture`` distinguishes.
DEFAULT_WORKSPACE_EDITS = {
    "delegated_new.txt": "written by the delegated run\n",
    "tracked.txt": "one\ntwo\nthree\n",
}

# The one scripted question a [FAKE:ASK] run raises — the exact
# ``ControlPendingInteraction`` wire keys ``pending_interactions()`` consumes.
FAKE_QUESTION_TEXT = "Which port should the fake service use?"
FAKE_ANSWER_LABEL = "8080"


def _fake_pending_interaction(run_id: str, harness_id: str) -> Dict[str, Any]:
    return {
        "interactionId": "int-" + run_id[:8],
        "runId": run_id,
        "attemptId": "a01",
        "harnessId": harness_id,
        "sourceTool": "AskUserQuestion",
        "questions": [{
            "id": "q1",
            "question": FAKE_QUESTION_TEXT,
            "header": "Port",
            "options": [{"label": FAKE_ANSWER_LABEL, "description": "the default"},
                        {"label": "9090", "description": None}],
            "multi_select": False,
        }],
        "requestedAt": "2026-09-01T00:00:00Z",
        # None = "waits until answered": the honest scripting for a scenario
        # that WILL answer (a non-null timeout would promise an engine-side
        # benign decline this fake never performs).
        "timeoutAt": None,
    }


def _tree_engine_identity() -> tuple:
    """(version, build_sha) the fake reports — the TREE'S OWN runtime pin.

    ``OwnedClaudexorDaemon.ensure_running`` returns an attached live endpoint
    WITHOUT touching the managed runtime only when the handshake identity equals
    the tracked pin exactly; any mismatch walks into ``runtime_manager.ensure()``,
    whose repair path downloads the pinned archive — network egress the keyless
    lane must never take. Reporting the pin's own identity keeps every scenario
    on the attach fast path, and keeps doing so when upstream bumps the pin.
    """
    try:
        from ouroboros.claudexor_runtime import load_runtime_pin

        pin = load_runtime_pin()
        if pin is not None:
            return str(pin.version), str(pin.build_sha)
    except Exception:
        pass
    return "3.99.0", "fakesha0"


class FakeClaudexorDaemon:
    """Loopback claudexord imitation (handshake / caps / quota / scripted runs).

    The contract facts served here are pinned against the REAL client by the
    default-lane ``test_fake_daemon_*`` contract tests (the gateway itself talks
    to this fake), so drift between the fake and ``gateways/claudexor.py`` is a
    named failure, not a silently green scenario.
    """

    def __init__(self, *, harness_id: str = "fake-harness",
                 engine_version: str = "",
                 engine_build_sha: str = "",
                 applied_model: str = "mock-model-echo",
                 applied_profile: str = "fake-profile-1",
                 ghost_profile: str = "ghost-profile",
                 workspace_edits: Optional[Dict[str, str]] = None,
                 runs_dir: Optional[pathlib.Path] = None) -> None:
        self.harness_id = str(harness_id)
        pin_version, pin_sha = _tree_engine_identity()
        self.engine_version = str(engine_version or pin_version)
        self.engine_build_sha = str(engine_build_sha or pin_sha)
        self.applied_model = str(applied_model)
        self.applied_profile = str(applied_profile)
        self.ghost_profile = str(ghost_profile)
        self.workspace_edits = dict(
            DEFAULT_WORKSPACE_EDITS if workspace_edits is None else workspace_edits)
        self.token = "fake-daemon-token-" + uuid.uuid4().hex
        self._runs_dir = pathlib.Path(runs_dir) if runs_dir else None
        self._lock = threading.Lock()
        self.requests: List[Dict[str, Any]] = []
        self.runs: Dict[str, Dict[str, Any]] = {}
        self._projects: Dict[str, str] = {}          # root -> project id
        self._replay: Dict[str, Dict[str, Any]] = {}  # idempotency key -> {digest, kind, payload}
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # pragma: no cover - stdlib noise
                return

            def _body(self) -> Any:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                if not raw:
                    return {}
                try:
                    return json.loads(raw.decode("utf-8"))
                except ValueError:
                    return {}

            def _send(self, payload: Any, status: int = 200) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method: str) -> None:
                body = self._body() if method in ("POST", "PATCH") else {}
                record = {
                    "method": method,
                    "path": self.path,
                    "idempotency_key": str(self.headers.get("Idempotency-Key") or ""),
                    "protocol_major": str(self.headers.get("X-Claudexor-Protocol-Major") or ""),
                    "body": body,
                }
                with outer._lock:
                    outer.requests.append(record)
                    if self.headers.get("Authorization") != f"Bearer {outer.token}":
                        return self._send({"code": "unauthorized",
                                           "message": "bad bearer token"}, 401)
                    status, payload = outer._route(method, self.path, record)
                return self._send(payload, status)

            def do_GET(self):     # noqa: N802 - stdlib callback name
                self._handle("GET")

            def do_POST(self):    # noqa: N802 - stdlib callback name
                self._handle("POST")

            def do_DELETE(self):  # noqa: N802 - stdlib callback name
                self._handle("DELETE")

            def do_PATCH(self):   # noqa: N802 - stdlib callback name
                self._handle("PATCH")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # -- lifecycle -------------------------------------------------------------

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> "FakeClaudexorDaemon":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "FakeClaudexorDaemon":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    def install(self, config_dir: pathlib.Path) -> pathlib.Path:
        """Write the ``daemon/control-api.json`` descriptor + token file under
        *config_dir* — the exact layout ``discover_daemon_at`` reads, and (when
        *config_dir* is ``<data_root>/claudexor``) the owned-daemon location the
        server's default discovery prefers once provisioned (D30)."""
        daemon_dir = pathlib.Path(config_dir) / "daemon"
        daemon_dir.mkdir(parents=True, exist_ok=True)
        token_path = daemon_dir / "token"
        token_path.write_text(self.token, encoding="utf-8")
        token_path.chmod(0o600)
        if self._runs_dir is None:
            self._runs_dir = pathlib.Path(config_dir) / "runs"
        (daemon_dir / "control-api.json").write_text(json.dumps({
            "host": "127.0.0.1", "port": self.port, "tokenPath": str(token_path),
        }), encoding="utf-8")
        return daemon_dir / "control-api.json"

    # -- recorded-wire helpers -------------------------------------------------

    def calls(self, method: str = "", path_prefix: str = "") -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.requests
                    if (not method or row["method"] == method)
                    and (not path_prefix or row["path"].startswith(path_prefix))]

    def run_start_posts(self) -> List[Dict[str, Any]]:
        return [row for row in self.calls("POST", "/v2/runs")
                if row["path"].rstrip("/") == "/v2/runs"]

    # -- routing ---------------------------------------------------------------

    def _route(self, method: str, path: str, record: Dict[str, Any]) -> tuple:
        """Answer one authenticated request. Caller holds the lock."""
        body = record["body"]
        clean = path.split("?", 1)[0].rstrip("/")
        parts = [p for p in clean.split("/") if p]
        if method == "POST" and clean == "/v2/handshake":
            return 200, {"compatible": True, "protocolMajor": 3,
                         "engine": {"version": self.engine_version,
                                    "sha": self.engine_build_sha}}
        if method == "GET" and clean == "/v2/agent-capabilities":
            return 200, {"harnesses": [self._harness_row()]}
        if method == "GET" and clean == "/v2/harnesses":
            return 200, {"harnesses": [self._harness_row()]}
        if method == "GET" and clean == "/v2/quota":
            return 200, {"snapshots": [], "absences": []}
        if method == "GET" and clean == "/v2/settings":
            return 200, {"harnesses": {}}
        if method == "POST" and clean == "/v2/settings":
            return 200, {}
        if clean == "/v2/projects":
            if method == "GET":
                return 200, {"projects": [
                    {"id": pid, "root": root} for root, pid in self._projects.items()]}
            if method == "POST":
                if not record["idempotency_key"]:
                    return 400, {"code": "missing_idempotency_key",
                                 "message": "project registration requires Idempotency-Key"}
                root = str(body.get("root") or "")
                if not root:
                    return 400, {"code": "invalid_request", "message": "root is required"}
                pid = self._projects.get(root) or ("proj" + uuid.uuid4().hex[:12])
                self._projects[root] = pid
                return 200, {"id": pid}
        if method == "DELETE" and len(parts) == 3 and parts[:2] == ["v2", "projects"]:
            for root, pid in list(self._projects.items()):
                if pid == parts[2]:
                    del self._projects[root]
                    return 200, {"removed": True}
            return 404, {"code": "project_not_found", "message": "no such project"}
        if method == "POST" and clean == "/v2/runs":
            return self._start_run(record)
        if len(parts) >= 3 and parts[:2] == ["v2", "runs"]:
            run = self.runs.get(parts[2])
            if run is None:
                return 404, {"code": "run_not_found", "message": "no such run"}
            if method == "GET" and len(parts) == 3:
                return 200, self._detail(run)
            if (method == "POST" and len(parts) == 6
                    and parts[3] == "interactions" and parts[5] == "answer"):
                # The delegate_answer verb. Typed statuses at ANY http code are
                # the client contract (answer_interaction accepts a body whose
                # ``status`` is one of delivered/not_found/already_resolved/
                # rejected regardless of the code).
                iid = parts[4]
                if not any(str(row.get("interactionId")) == iid for row in run["pending"]):
                    return 409, {"accepted": False, "status": "already_resolved",
                                 "message": "no such pending interaction"}
                rows = body.get("answers")
                if not isinstance(rows, list) or not rows:
                    return 400, {"accepted": False, "status": "rejected",
                                 "message": "answers must be a non-empty list"}
                run["answers"].append({"interaction_id": iid, "answers": rows})
                run["pending"] = [row for row in run["pending"]
                                  if str(row.get("interactionId")) != iid]
                return 200, {"accepted": True, "status": "delivered"}
            if method == "POST" and len(parts) == 4 and parts[3] == "control":
                control = body.get("control") if isinstance(body.get("control"), dict) else {}
                if str(control.get("kind") or "") == "cancel":
                    if run["state"] not in ("succeeded", "cancelled", "failed"):
                        run["state"] = "cancelled"
                    run["cancel_reason"] = str(control.get("reason") or "")
                    return 200, {"accepted": True, "status": "cancelling"}
                return 400, {"code": "unknown_control", "message": "unsupported control kind"}
        return 404, {"code": "not_found", "message": f"no route for {method} {clean}"}

    def _harness_row(self) -> Dict[str, Any]:
        return {"id": self.harness_id, "enabled": True,
                "accessProfilesSupported": ["readonly", "workspace_write",
                                            "external_sandbox_full"]}

    def _start_run(self, record: Dict[str, Any]) -> tuple:
        body = record["body"]
        key = record["idempotency_key"]
        if not key:
            return 400, {"code": "missing_idempotency_key",
                         "message": "run start requires Idempotency-Key"}
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
        replayed = self._replay.get(key)
        if replayed is not None:
            if replayed["digest"] != digest:
                return 409, {"code": "idempotency_conflict",
                             "message": "Idempotency-Key replayed with a different request digest"}
            return replayed["status"], json.loads(json.dumps(replayed["payload"]))

        def _remember(status: int, payload: Dict[str, Any]) -> tuple:
            self._replay[key] = {"digest": digest, "status": status, "payload": payload}
            return status, payload

        harnesses = body.get("harnesses")
        if harnesses != [self.harness_id] or body.get("primaryHarness") != self.harness_id:
            return _remember(404, {"code": "route_not_found",
                                   "message": f"this daemon serves only {self.harness_id!r}"})
        prompt = str(body.get("prompt") or "")
        if FAKE_REFUSE_MARKER in prompt:
            return _remember(400, {"code": "fake_route_refused",
                                   "message": "scripted typed refusal of this start"})
        pinned = str(body.get("credentialProfileId") or "")
        if pinned and pinned == self.ghost_profile:
            return _remember(409, {"code": "credential_profile_unknown",
                                   "message": f"profile {pinned!r} is not registered"})
        rid = uuid.uuid4().hex[:16]
        run_dir = ""
        if self._runs_dir is not None:
            run_dir = str(self._runs_dir / rid)
            pathlib.Path(run_dir).mkdir(parents=True, exist_ok=True)
        self.runs[rid] = {
            "id": rid, "state": "running", "polls": 0,
            "hang": FAKE_HANG_MARKER in prompt,
            "access": str(body.get("access") or ""),
            "run_dir": run_dir, "body": body, "cancel_reason": "",
            "pending": ([_fake_pending_interaction(rid, self.harness_id)]
                        if FAKE_ASK_MARKER in prompt else []),
            "answers": [], "workspace_written": [],
        }
        if FAKE_MUTATE_MARKER in prompt:
            self._perform_run_work(self.runs[rid])
        return _remember(200, {"runId": rid, "runDir": run_dir})

    # -- the mutating half: what a WRITING harness leaves behind ---------------

    def _perform_run_work(self, run: Dict[str, Any]) -> None:
        """Do a mutating run's work the way a real harness does: edit the workspace
        the start body named, then record the attempt's APPLIED facts.

        The workspace is ``execution.workspaceRoot`` — the PRIVATE execution snapshot
        Ouroboros provisioned — and never ``scope.root``, which stays the live tree the
        engine may only read. A fake writing to ``scope.root`` would be a fake that
        breaks the isolation the scenarios exist to prove, so the two are read from
        their own keys and the live root is never touched here.
        """
        execution = run["body"].get("execution")
        execution = execution if isinstance(execution, dict) else {}
        workspace = pathlib.Path(str(execution.get("workspaceRoot") or ""))
        if str(workspace) and workspace.is_dir():
            for rel, text in self.workspace_edits.items():
                target = workspace.joinpath(*pathlib.PurePosixPath(rel).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                run["workspace_written"].append(rel)
        self._write_attempt_record(run)

    def _write_attempt_record(self, run: Dict[str, Any]) -> None:
        """``<runDir>/attempts/a01/attempt.yaml`` in Claudexor's applied-facts shape.

        The ONLY evidence Ouroboros has that the harness HOME was scoped and that an
        OS boundary was actually applied (``attempt_containment``): the HOME pair is
        projected onto no ``/v2`` response. The mechanism is written WITH the path the
        policy was proved against, because a mechanism without its proof is read as no
        boundary at all — the fake must not be able to claim a containment the reader
        would refuse to believe from a real engine.
        """
        if not run["run_dir"]:
            return
        attempt_dir = pathlib.Path(run["run_dir"]) / "attempts" / "a01"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        home_dir = str(pathlib.Path(run["run_dir"]) / "home")
        record = {
            "attempt_id": "a01",
            "harness_id": self.harness_id,
            "harness_home_isolated": True,
            "harness_home_dir": home_dir,
            "confinement_mechanism": "fake-sandbox",
            "confinement_verified_denied_path": str(
                pathlib.Path(run["run_dir"]).parent / "daemon"),
        }
        (attempt_dir / "attempt.yaml").write_text(
            "\n".join(f"{key}: {json.dumps(value)}" for key, value in record.items()) + "\n",
            encoding="utf-8")

    def _detail(self, run: Dict[str, Any]) -> Dict[str, Any]:
        run["polls"] += 1
        # A run with a pending interaction WAITS (state stays running) until
        # the answer verb clears it; the very next poll then flips terminal.
        if run["state"] == "running" and not run["hang"] and not run["pending"]:
            run["state"] = "succeeded"
        state = run["state"]
        terminal = state in ("succeeded", "cancelled", "failed")
        summary: Dict[str, Any] = {
            "state": state,
            "model": self.applied_model,
            "effectiveAccess": run["access"],
            "waitingOnUser": bool(run["pending"]),
            "runDir": run["run_dir"],
        }
        detail: Dict[str, Any] = {
            "id": run["id"],
            "lastSeq": 3 if terminal else run["polls"],
            "summary": summary,
            "pendingInteractions": [json.loads(json.dumps(row)) for row in run["pending"]],
        }
        if terminal:
            summary.update({
                "spendUsd": 0.0, "spendEstimated": False,
                "inputTokens": 120, "outputTokens": 40, "cachedInputTokens": 0,
                "authRoute": {"profileId": self.applied_profile},
            })
            text = f"FAKE_RUN_RESULT {run['id']}: assignment complete."
            detail["outcomeBanner"] = state
            detail["finalSummary"] = "Fake delegated run finished."
            detail["primaryOutput"] = {
                "text": text, "bytes": len(text.encode("utf-8")),
                "truncated": False, "path": "output/final.md",
            }
        return detail


class PlaywrightUIClient:
    """Real-browser client over an isolated server's web UI (gateway/UI truth)."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(_NOT_LANDED.format(
            name="PlaywrightUIClient", lane="gateway/UI-truth"))

    def open(self) -> "PlaywrightUIClient":  # pragma: no cover - unreachable
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - unreachable
        raise NotImplementedError
