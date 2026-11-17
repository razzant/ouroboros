"""S39 — both container modes on real Docker (the ``container`` deep-integration lane).

Keyless. The lifecycle scenarios run providerless (the gateway serves without a
supervisor); the consumer scenarios configure the harness's own loopback stub models,
which a ``--network none`` container reaches only through ``_ModelRelay`` over one
``docker exec``'s stdio, so no container needs a route to the host or the internet for a
model. Every assertion reads durable facts — the serving checkout's HEAD, the server's
interpreter, the runtime venv's marker, files on the volumes, API results, container exit
codes and restart counts, the stub's wire record — never a model answer's quality.
Recipes: packaging/docker/compose.yaml; contract: docs/DEPLOYMENT.md.

Opt-in only, and never against an implicit Docker daemon (exactly one target):

    OUROBOROS_E2E_DEEP=container
    OUROBOROS_CONTAINER_DOCKER_CONTEXT=<a named, dedicated context; "default" is refused>
      or OUROBOROS_CONTAINER_DOCKER_HOST=<the dedicated daemon's address>
    OUROBOROS_CONTAINER_DOCKER_CONFIG=<docker client config directory>      (optional)
    OUROBOROS_CONTAINER_IMAGE=ouroboros-web            (immutable image, built from this tree)
    OUROBOROS_CONTAINER_IMAGE_SHA=<commit the immutable image was built from>  (optional)
    OUROBOROS_CONTAINER_MANAGED_IMAGE=ouroboros-managed (``--target managed`` over a released seed)
    OUROBOROS_CONTAINER_SEED_MANIFEST=<seed dir>/repo_bundle_manifest.json  (first-boot identity)
    OUROBOROS_CONTAINER_CANDIDATE_GIT=<host repository, candidate on branch ouroboros>  (Update)
    OUROBOROS_CONTAINER_EVIDENCE=<file>   (optional: JSON lines of every observed fact)
    OUROBOROS_CONTAINER_KEEP=1            (optional: keep this run's containers and volumes)

Run from the repository root; set test controls AFTER safe_test's environment isolation:

    python -I -S scripts/safe_test.py --temp-parent /tmp -- env \
      OUROBOROS_E2E_DEEP=container \
      OUROBOROS_CONTAINER_DOCKER_CONTEXT=ouroboros-qa \
      OUROBOROS_CONTAINER_IMAGE=ouroboros-web \
      OUROBOROS_CONTAINER_MANAGED_IMAGE=ouroboros-managed \
      OUROBOROS_CONTAINER_SEED_MANIFEST=/absolute/seed/repo_bundle_manifest.json \
      OUROBOROS_CONTAINER_CANDIDATE_GIT="$PWD" \
      OUROBOROS_CONTAINER_EVIDENCE="$PWD/s39.jsonl" \
      "$PWD/.venv/bin/python" -m pytest tests/system_e2e/test_managed_container.py \
      -m integration --junitxml=s39.junit.xml

Substitute your dedicated context and seed path; HOST/CONFIG may replace CONTEXT as above.
No host path is mounted into a container: the candidate repository and the Update's wheel
are streamed into a volume. Every container and volume carries this run's id and is
removed by exact name at the end. The plumbing (relay, in-container API client, wheel,
dependency commit, daemon selection) has default-lane tests in test_container_driver_plumbing.py.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import pathlib
import shlex
import socket
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
import zipfile

import pytest

from ouroboros.configured_subagents import SUBAGENTS_SETTING
from ouroboros.settings_setup_contract import ONBOARDING_COMPLETED_KEY, secret_provider_setting_keys
from tests.system_e2e.harness import (
    LANE_CONTAINER,
    MOCK_SLUG,
    ReplayModel,
    ScriptedStubModel,
    assert_settings_keyless,
    body_text,
    keyless_review_rows,
    keyless_settings,
    require_lane,
    supervisor_state_is_ready,
)
from tests.system_e2e.test_system_scenarios_w2 import (
    S6_CHILD_MARKER,
    S6_CHILD_SLUG,
    S6_FIXTURE,
    S6_PARENT_MARKER,
    _s6_slot_binder,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
VENV_PYTHON = "/ouroboros/body/venv/bin/python"
BODY_REPO = "/ouroboros/body/repo"
MANAGED_DATA = "/ouroboros/data"
RUN_ID = uuid.uuid4().hex[:8]
TERMINAL = {"completed", "failed", "cancelled", "rejected_duplicate"}
_CREATED: dict = {"containers": [], "volumes": []}

# The container-side end of the model relay: loopback inside the container, never one of
# Ouroboros's own ports. Settings name it, so the keyless loopback rule holds there too.
RELAY_PORT = 18080
RELAY_BASE_URL = f"http://127.0.0.1:{RELAY_PORT}/v1"
STUB_KEY = "stub-key-not-a-credential"

_SERVER_FACTS = r'''
import json, os, subprocess
servers = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        argv = [part.decode() for part in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if part]
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        continue
    if any(part.endswith("server.py") for part in argv):
        head = subprocess.run(["git", "-C", cwd, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        servers.append({"pid": int(pid), "argv": argv, "cwd": cwd, "head": head})
print(json.dumps(servers))
'''

# One HTTP request from inside the container to the port its server published. A
# ``--network none`` container has no published port, and the port file is the endpoint
# every in-container consumer reads.
_API_CLIENT = r'''
import base64, json, sys, urllib.error, urllib.request
ask = json.load(sys.stdin)
with open(ask["port_file"], encoding="utf-8") as stream:
    port = int(stream.read().strip())
data = None if ask["payload"] is None else json.dumps(ask["payload"]).encode()
request = urllib.request.Request(f"http://127.0.0.1:{port}{ask['path']}", data=data, method=ask["method"],
                                 headers={"Content-Type": "application/json"})
try:
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=ask["timeout"]) as answer:
        status, body = answer.status, answer.read()
except urllib.error.HTTPError as error:
    status, body = error.code, error.read()
print(json.dumps({"port": port, "status": status, "body": base64.b64encode(body).decode()}))
'''

# The relay's container end: accepts loopback connections and multiplexes them over its
# stdio as frames of (kind, stream id, length): 0 ready, 1 open, 2 data, 3 close.
_RELAY_PROGRAM = r'''
import socket, struct, sys, threading
out, inp, lock, conns = sys.stdout.buffer, sys.stdin.buffer, threading.Lock(), {}
def send(kind, sid, data=b""):
    with lock:
        out.write(struct.pack(">BII", kind, sid, len(data)) + data)
        out.flush()
def pump(sid, conn):
    try:
        while True:
            data = conn.recv(65536)
            if not data:
                break
            send(2, sid, data)
    except OSError:
        pass
    send(3, sid)
def serve(listener):
    sid = 0
    while True:
        conn, _ = listener.accept()
        sid += 1
        conns[sid] = conn
        send(1, sid)
        threading.Thread(target=pump, args=(sid, conn), daemon=True).start()
def read(size):
    data = b""
    while len(data) < size:
        chunk = inp.read(size - len(data))
        if not chunk:
            raise SystemExit(0)
        data += chunk
    return data
listener = socket.create_server(("127.0.0.1", int(sys.argv[1])))
threading.Thread(target=serve, args=(listener,), daemon=True).start()
send(0, 0)
while True:
    kind, sid, size = struct.unpack(">BII", read(9))
    data = read(size) if size else b""
    conn = conns.get(sid)
    if conn is None:
        continue
    if kind == 2:
        try:
            conn.sendall(data)
        except OSError:
            pass
    elif kind == 3:
        conns.pop(sid, None)
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()
'''


# ---------------------------------------------------------------------------
# The one Docker daemon this lane may use
# ---------------------------------------------------------------------------

_AMBIENT_DOCKER_KEYS = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")


def _docker_base() -> list:
    """The docker CLI bound to the one daemon named for this lane; ambient selection is never used."""
    context = os.environ.get("OUROBOROS_CONTAINER_DOCKER_CONTEXT", "").strip()
    host = os.environ.get("OUROBOROS_CONTAINER_DOCKER_HOST", "").strip()
    config = os.environ.get("OUROBOROS_CONTAINER_DOCKER_CONFIG", "").strip()
    if bool(context) == bool(host) or context == "default":
        pytest.skip("set exactly one of OUROBOROS_CONTAINER_DOCKER_CONTEXT (a named, dedicated context; "
                    "'default' is refused) or OUROBOROS_CONTAINER_DOCKER_HOST")
    return ["docker", *(["--config", config] if config else []),
            *(["--context", context] if context else ["--host", host])]


def _docker_env() -> dict:
    return {key: value for key, value in os.environ.items() if key not in _AMBIENT_DOCKER_KEYS}


def _docker(*args: str, check: bool = True, timeout: float = 600, stdin=None) -> subprocess.CompletedProcess:
    """Run one docker command; ``stdin`` is bytes or an open binary file streamed to it."""
    command = [*_docker_base(), *args]
    feed = {"input": stdin} if isinstance(stdin, bytes) else {"stdin": stdin}
    run = subprocess.run(command, capture_output=True, timeout=timeout, env=_docker_env(), **feed)
    run.stdout, run.stderr = run.stdout.decode("utf-8", "replace"), run.stderr.decode("utf-8", "replace")
    if check and run.returncode != 0:
        raise AssertionError(f"{' '.join(command[:12])} -> {run.returncode}\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}")
    return run


def _evidence(step: str, **facts) -> dict:
    row = {"run": RUN_ID, "step": step, "ts": time.time(), **facts}
    print(json.dumps(row, default=str), flush=True)
    target = os.environ.get("OUROBOROS_CONTAINER_EVIDENCE", "").strip()
    if target:
        with open(target, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
    return row


@pytest.fixture(scope="module")
def _lane():
    require_lane(LANE_CONTAINER)
    _docker_base()
    yield
    if os.environ.get("OUROBOROS_CONTAINER_KEEP") == "1":
        _evidence("kept", **_CREATED)
        return
    for name in _CREATED["containers"]:
        _docker("rm", "-f", name, check=False)
    for name in _CREATED["volumes"]:
        _docker("volume", "rm", name, check=False)


def _name(role: str) -> str:
    return f"ob-qa-{RUN_ID}-{role}"


def _volume(role: str) -> str:
    name = _name(role)
    _docker("volume", "create", name)
    _CREATED["volumes"].append(name)
    return name


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _image(kind: str) -> str:
    key = "OUROBOROS_CONTAINER_MANAGED_IMAGE" if kind == "managed" else "OUROBOROS_CONTAINER_IMAGE"
    return os.environ.get(key, "").strip() or ("ouroboros-managed" if kind == "managed" else "ouroboros-web")


def _recipe(service: str) -> dict:
    """One service of the executable recipe carrier, so the lane qualifies the recipe itself."""
    yaml = pytest.importorskip("yaml")
    recipe = yaml.safe_load((REPO / "packaging" / "docker" / "compose.yaml").read_text(encoding="utf-8"))
    return recipe["services"][service]


def _run(role: str, kind: str, mounts: list, *, args=(), env=None, restart="on-failure", network=""):
    """Start one detached container; returns (name, published host port or None under ``network="none"``)."""
    name, port = _name(role), (None if network == "none" else _free_port())
    command = ["run", "-d", "--name", name, "--init", "--read-only", "--tmpfs", "/tmp", "--restart", restart]
    command += ["--network", "none"] if network == "none" else ["-p", f"127.0.0.1:{port}:8765"]
    for volume, target in mounts:
        command += ["-v", f"{volume}:{target}"]
    for key, value in (env or {}).items():
        command += ["-e", f"{key}={value}"]
    command += [_image(kind), *args]
    _docker(*command)
    _CREATED["containers"].append(name)
    return name, port


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as response:  # noqa: S310
            return response.status == 200
    except Exception:
        return False


def _wait_healthy(name: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not _healthy(port):
        state = _state(name)
        if state.get("Status") == "exited" or time.monotonic() > deadline:
            logs = _docker("logs", "--tail", "200", name, check=False)
            raise AssertionError(f"{name} never became healthy ({state}):\n{logs.stdout}\n{logs.stderr}")
        time.sleep(2)


def _state(name: str) -> dict:
    run = _docker("inspect", "--format", "{{json .State}}|{{.RestartCount}}", name, check=False)
    if run.returncode != 0:
        return {}
    state, _, restarts = run.stdout.strip().partition("|")
    return {**json.loads(state), "RestartCount": int(restarts or 0)}


def _wait_exited(name: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _state(name)
        if state.get("Status") == "exited" and not state.get("Restarting"):
            return state
        time.sleep(1)
    raise AssertionError(f"{name} did not exit: {_state(name)}")


def _exec(name: str, *command: str) -> str:
    return _docker("exec", name, *command).stdout.strip()


def _server_facts(name: str) -> list:
    return json.loads(_exec(name, "python", "-c", _SERVER_FACTS))


def _body_facts(name: str) -> dict:
    servers = _server_facts(name)
    marker = json.loads(_exec(name, "cat", "/ouroboros/body/venv/ouroboros-runtime-venv.json"))
    return {"head": _exec(name, "git", "-C", BODY_REPO, "rev-parse", "HEAD"),
            "server_interpreters": sorted({server["argv"][0] for server in servers}),
            "serving_heads": sorted({server["head"] for server in servers}),
            "venv_marker": marker}


def _seed_sha() -> str:
    path = os.environ.get("OUROBOROS_CONTAINER_SEED_MANIFEST", "").strip()
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))["source_sha"] if path else ""


def _boot_timeout() -> float:
    return float(os.environ.get("OUROBOROS_CONTAINER_BOOT_TIMEOUT", "1800"))


def _post(port: int, path: str, payload: dict, timeout: float = 60) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8") or "{}")


def _helper(volume_mounts: list, script: str, kind: str = "managed", stdin=None) -> str:
    """A throwaway networkless container of the same image, used only to read or seed a volume."""
    command = ["run", "--rm", *(["-i"] if stdin is not None else []), "--network", "none", "--entrypoint", "sh"]
    for volume, target in volume_mounts:
        command += ["-v", f"{volume}:{target}"]
    return _docker(*command, _image(kind), "-c", script, stdin=stdin).stdout


def _put_file(volume_mounts: list, path: str, data: bytes, kind: str = "managed") -> None:
    quoted = shlex.quote(path)
    _helper(volume_mounts, f"mkdir -p \"$(dirname {quoted})\" && umask 077 && cat > {quoted}", kind, stdin=data)


# ---------------------------------------------------------------------------
# Talking to a server inside its container, and the model relay
# ---------------------------------------------------------------------------

def _api_call(name: str, data_dir: str, method: str, path: str, payload=None, timeout: float = 60.0) -> tuple:
    """``(status, body)`` of one request made inside ``name`` to its published port; status 0: no answer."""
    ask = {"port_file": f"{data_dir}/state/server_port", "method": method, "path": path,
           "payload": payload, "timeout": timeout}
    run = _docker("exec", "-i", name, "python", "-c", _API_CLIENT, check=False, timeout=timeout + 60,
                  stdin=json.dumps(ask).encode())
    if run.returncode != 0:
        return 0, run.stderr[-800:].encode()
    answer = json.loads(run.stdout)
    return answer["status"], base64.b64decode(answer["body"])


def _api_json(api, method: str, path: str, payload=None, timeout: float = 60.0) -> dict:
    status, body = api(method, path, payload, timeout)
    assert status == 200, f"{method} {path} -> {status}: {body[-800:]!r}"
    return json.loads(body or b"{}")


def _wait_served(name: str, data_dir: str, timeout: float, *, supervisor: bool = True) -> dict:
    """Poll the server inside ``name`` until it answers (and, by default, its supervisor is ready)."""
    deadline = time.monotonic() + timeout
    while True:
        status, body = _api_call(name, data_dir, "GET", "/api/state" if supervisor else "/api/health", timeout=10)
        if status == 200:
            state = json.loads(body or b"{}")
            if not supervisor or supervisor_state_is_ready(state):
                return state
        container = _state(name)
        if container.get("Status") == "exited" or time.monotonic() > deadline:
            logs = _docker("logs", "--tail", "200", name, check=False)
            raise AssertionError(f"{name} never served ({container}):\n{logs.stdout}\n{logs.stderr}")
        time.sleep(3)


def _stub_port(model) -> int:
    return int(urllib.parse.urlsplit(model.base_url).port)


def _read_exact(stream, size: int):
    data = b""
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _close(sock) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


class _ModelRelay:
    """A container's loopback model endpoint, carried to a host stub over one ``docker exec``'s stdio.

    The scenario model stays the harness's own loopback stub on the host; the container
    reaches it only through this pipe, so ``--network none`` holds and no route from a
    container to the host is opened. Frames are those of ``_RELAY_PROGRAM``."""

    def __init__(self, container: str, upstream_port: int, command=None) -> None:
        self.command = command or [*_docker_base(), "exec", "-i", container, "python", "-u", "-c",
                                   _RELAY_PROGRAM, str(RELAY_PORT)]
        self.upstream_port = upstream_port
        self.streams = 0
        self.proc = None
        self._upstreams: dict = {}
        self._lock = threading.Lock()

    def _ready(self, timeout: float) -> bool:
        """The ready frame within ``timeout``; the read runs on a thread, so a stalled exec cannot outlast it."""
        frames: list = []
        reader = threading.Thread(target=lambda: frames.append(_read_exact(self.proc.stdout, 9)), daemon=True)
        reader.start()
        reader.join(timeout)
        return frames == [struct.pack(">BII", 0, 0, 0)]

    def __enter__(self) -> "_ModelRelay":
        deadline = time.monotonic() + 120
        while True:  # a container still starting refuses the exec; its program answers "ready" once listening
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, env=_docker_env())
            if self._ready(max(1.0, min(30.0, deadline - time.monotonic()))):
                break
            self.proc.kill()
            self.proc.wait(timeout=30)
            if time.monotonic() > deadline:
                raise AssertionError(f"the model relay never started within 120s: {self.command[:6]}")
            time.sleep(2)
        threading.Thread(target=self._down, daemon=True).start()
        return self

    def _down(self) -> None:
        while (header := _read_exact(self.proc.stdout, 9)) is not None:
            kind, sid, size = struct.unpack(">BII", header)
            data = _read_exact(self.proc.stdout, size) if size else b""
            if data is None:
                break
            if kind == 1:
                self.streams += 1
                try:
                    upstream = socket.create_connection(("127.0.0.1", self.upstream_port), timeout=30)
                    upstream.settimeout(None)
                except OSError:
                    self._send(3, sid)
                    continue
                self._upstreams[sid] = upstream
                threading.Thread(target=self._up, args=(sid, upstream), daemon=True).start()
            elif kind == 2 and sid in self._upstreams:
                try:
                    self._upstreams[sid].sendall(data)
                except OSError:
                    pass
            elif kind == 3 and (upstream := self._upstreams.pop(sid, None)) is not None:
                _close(upstream)
        for upstream in list(self._upstreams.values()):
            _close(upstream)

    def _up(self, sid: int, upstream) -> None:
        try:
            while data := upstream.recv(65536):
                self._send(2, sid, data)
        except OSError:
            pass
        self._send(3, sid)

    def _send(self, kind: int, sid: int, data: bytes = b"") -> None:
        with self._lock:
            try:
                self.proc.stdin.write(struct.pack(">BII", kind, sid, len(data)) + data)
                self.proc.stdin.flush()
            except (OSError, ValueError):
                pass

    def __exit__(self, *_exc) -> None:
        with self._lock:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)


# ---------------------------------------------------------------------------
# B — the keyless consumer flow (onboarding, project, task + child, artifact)
# ---------------------------------------------------------------------------

B_PROJECT = "S39 consumer room"
B_ARTIFACT = "s39-consumer-report.txt"
B_ARTIFACT_TEXT = "S39 consumer report: written by the root after it absorbed its scout.\n"
B_TASK = "Delegate the repository survey to your scout, absorb it, write the report, then finish."
_S6_ROOT_FINAL = ("root", "mock-model|tools", 4)
# S6's exact parent/child tree (wait, exact-hash disposition) plus one artifact write.
B_FIXTURE = {
    **{key: step for key, step in S6_FIXTURE.items() if key != _S6_ROOT_FINAL},
    _S6_ROOT_FINAL: {"tool": "write_file", "arguments": {
        "root": "artifact_store", "path": B_ARTIFACT, "content": B_ARTIFACT_TEXT}},
    ("root", "mock-model|tools", 5): S6_FIXTURE[_S6_ROOT_FINAL],
}


def _onboarding_payload() -> dict:
    """What the onboarding wizard posts for a keyless install: the loopback stub as an
    OpenAI-compatible provider and an owner-edited catalog (the S6 scout plus keyless reviewers)."""
    scout = {"subagent_id": "mock-child", "recommended_use": "Read-only scout for the S39 consumer flow.",
             "route": {"kind": "api_model", "target_id": S6_CHILD_SLUG}, "effort": "low"}
    return {
        "OPENAI_COMPATIBLE_BASE_URL": RELAY_BASE_URL,
        "OPENAI_COMPATIBLE_API_KEY": STUB_KEY,
        "OUROBOROS_MODEL": MOCK_SLUG,
        "OUROBOROS_MODEL_LIGHT": MOCK_SLUG,
        "OUROBOROS_RUNTIME_MODE": "light",
        "TOTAL_BUDGET": 10.0,
        "OUROBOROS_PER_TASK_COST_USD": 10.0,
        "skipSubscriptionPresets": True,
        SUBAGENTS_SETTING: {"enabled": True, "items": [scout, *keyless_review_rows()]},
    }


def _assert_onboarded_keyless(saved: dict) -> None:
    """An onboarded document keeps provider defaults (endpoints, scopes); keyless means its only
    secret is the stub's placeholder and its model endpoint the container-loopback relay."""
    secrets = {key: saved[key] for key in secret_provider_setting_keys() if str(saved.get(key) or "").strip()}
    assert secrets == {"OPENAI_COMPATIBLE_API_KEY": STUB_KEY}, sorted(secrets)
    assert saved.get("OPENAI_COMPATIBLE_BASE_URL") == RELAY_BASE_URL, saved.get("OPENAI_COMPATIBLE_BASE_URL")


def _wait_api(api, predicate, timeout: float, what: str):
    deadline, last = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(2)
    raise AssertionError(f"{what} within {timeout:.0f}s (last: {last!r})")


def _supervisor_ready(api) -> dict:
    status, body = api("GET", "/api/state", None, 10)
    state = json.loads(body or b"{}") if status == 200 else {}
    return state if supervisor_state_is_ready(state) else {}


def _task_rows(api) -> dict:
    """Every task's lineage row, as the task list projects it from the durable store."""
    listed = _api_json(api, "GET", "/api/tasks?limit=0").get("tasks") or []
    keep = ("status", "parent_task_id", "root_task_id", "delegation_role", "result")
    return {str(row.get("task_id")): {key: row.get(key) for key in keep} for row in listed if isinstance(row, dict)}


def _terminal_row(api, task_id: str) -> dict:
    row = _task_rows(api).get(task_id) or {}
    return row if row.get("status") in TERMINAL else {}


def _consumer_flow(api, settings) -> dict:
    """Onboarding; a project placed on the work root; its root task, child and artifact.
    Returns the facts a restart must keep.
    ``api(method, path, payload, timeout)`` answers ``(status, body)``; ``settings()`` reads the
    saved settings document."""
    completed = _api_json(api, "POST", "/api/onboarding/complete", _onboarding_payload(), 180)
    assert completed.get("ok") is True, completed
    _wait_api(api, lambda: _supervisor_ready(api), 300, "the onboarded supervisor was not ready")
    saved = settings()
    _assert_onboarded_keyless(saved)
    assert saved.get(ONBOARDING_COMPLETED_KEY), "the completion is durable"
    # Then an ordinary Settings save, to S6's keyless posture: a canned native reviewer answers
    # without the reads its coverage requires, so a task acceptance review would loop the strict tree.
    # Images bind 0.0.0.0: an ordinary web Settings save supplies its required network password.
    review = _api_json(api, "POST", "/api/settings", {
        "OUROBOROS_TASK_REVIEW_MODE": "off", "OUROBOROS_NETWORK_PASSWORD": f"s39-local-test-{RUN_ID}"}, 120)
    assert review.get("saved") is not False, review
    _wait_api(api, lambda: settings().get("OUROBOROS_TASK_REVIEW_MODE") == "off", 30, "the Settings save never landed")

    project = _api_json(api, "POST", "/api/projects", {"name": B_PROJECT, "with_workspace": True}).get("project") or {}
    assert project.get("id") and project.get("working_dir"), project

    task_id = str(_api_json(api, "POST", "/api/tasks", {
        "description": B_TASK, "project_id": project["id"], "memory_mode": "forked",
        "actor_id": "s39-driver", "source": "s39-driver",
        "metadata": {"source": "s39-driver", "delegation_role": "root"}}).get("task_id") or "")
    assert task_id
    root = _wait_api(api, lambda: _terminal_row(api, task_id), 600, f"task {task_id} reached no terminal")
    assert root["status"] == "completed" and S6_PARENT_MARKER in str(root.get("result") or ""), root
    children = {tid: row for tid, row in _task_rows(api).items() if row.get("parent_task_id") == task_id}
    assert len(children) == 1, children
    (child_id, child), = children.items()
    assert child["status"] == "completed" and child.get("delegation_role") == "subagent", child
    assert S6_CHILD_MARKER in str(child.get("result") or ""), child
    status, artifact = api("GET", f"/api/tasks/{task_id}/artifacts/{B_ARTIFACT}", None, 60)
    assert status == 200 and artifact == B_ARTIFACT_TEXT.encode(), (status, artifact[-400:])
    public = _api_json(api, "GET", f"/api/tasks/{task_id}")
    assert public.get("status") == "completed" and public.get("project_id") == project["id"], public
    child_public = _api_json(api, "GET", f"/api/tasks/{child_id}")
    assert child_public.get("project_id") == project["id"], child_public
    return {"task_id": task_id, "child_id": child_id, "project": project, "artifact": artifact,
            "result": root.get("result"), "onboarded": saved.get(ONBOARDING_COMPLETED_KEY)}


def _consumer_facts_survive(api, settings, facts: dict) -> None:
    """After a restart: the same durable task tree, artifact, project and settings."""
    rows = _task_rows(api)
    root = rows.get(facts["task_id"]) or {}
    assert root.get("status") == "completed" and root.get("result") == facts["result"], root
    assert (rows.get(facts["child_id"]) or {}).get("status") == "completed", rows.get(facts["child_id"])
    public = _api_json(api, "GET", f"/api/tasks/{facts['task_id']}")
    assert public.get("status") == "completed" and public.get("project_id") == facts["project"]["id"], public
    status, artifact = api("GET", f"/api/tasks/{facts['task_id']}/artifacts/{B_ARTIFACT}", None, 60)
    assert status == 200 and artifact == facts["artifact"], (status, artifact[-400:])
    projects = _api_json(api, "GET", "/api/projects").get("projects") or []
    kept = [row for row in projects if row.get("id") == facts["project"]["id"]]
    assert kept and kept[0].get("working_dir") == facts["project"]["working_dir"], projects
    saved = settings()
    assert saved.get(ONBOARDING_COMPLETED_KEY) == facts["onboarded"] and saved.get("OUROBOROS_TASK_REVIEW_MODE") == "off"


@pytest.mark.integration
@pytest.mark.usefixtures("_lane")
def test_s39_the_immutable_image_serves_a_keyless_consumer_from_onboarding_through_restart():
    recipe = _recipe("image")
    env = {key: str(value) for key, value in recipe["environment"].items()}
    data_dir = env["OUROBOROS_DATA_DIR"]
    mounts = [(_volume(f"consumer-{index}"), volume.split(":", 1)[1]) for index, volume in enumerate(recipe["volumes"])]
    name, _none = _run("consumer", "image", mounts, env=env, network="none")
    _wait_served(name, data_dir, 600, supervisor=False)
    assert _exec(name, "sh", "-c", f"test -e {data_dir}/settings.json && echo present || echo absent") == "absent"
    assert _exec(name, "cat", f"{data_dir}/state/server_port") == "8765", "the published port is the requested one"

    def api(method, path, payload=None, timeout=60.0):
        return _api_call(name, data_dir, method, path, payload, timeout)

    def settings():
        return json.loads(_exec(name, "cat", f"{data_dir}/settings.json"))

    model = ReplayModel(B_FIXTURE, slot_binder=_s6_slot_binder, model_ids=["mock-model", "mock-child"])
    with model:
        with _ModelRelay(name, _stub_port(model)) as relay:
            facts = _consumer_flow(api, settings)
        before = _server_facts(name)
        assert {server["cwd"] for server in before} == {"/app"}, before
        expected = os.environ.get("OUROBOROS_CONTAINER_IMAGE_SHA", "").strip()
        if expected:
            assert {server["head"] for server in before} == {expected}, before
        assert facts["project"]["working_dir"].startswith(env["OUROBOROS_SUBAGENT_PROJECTS_ROOT"] + "/"), facts
        _evidence("consumer-flow", task=facts["task_id"], child=facts["child_id"], streams=relay.streams,
                  project=facts["project"].get("working_dir"), servers=before)

        _docker("restart", "-t", "30", name)
        _wait_served(name, data_dir, 600)
        with _ModelRelay(name, _stub_port(model)):  # nothing may call it: a completed tree is not re-run
            _consumer_facts_survive(api, settings, facts)
            time.sleep(10)
        after = _server_facts(name)
        _evidence("consumer-restarted", servers=after, calls=model.kinds())
        assert {server["head"] for server in after} == {server["head"] for server in before}
        model.assert_consumed()


# ---------------------------------------------------------------------------
# Managed lifecycle: seed boot, recreation, Panic, layouts, shared data
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.usefixtures("_lane")
def test_s39_managed_parent_volume_survives_recreation_and_panic_needs_the_owner():
    volume = _volume("parent")
    first, port = _run("parent-1", "managed", [(volume, "/ouroboros")])
    _wait_healthy(first, port, _boot_timeout())
    booted = _evidence("managed-first-boot", **_body_facts(first))
    if _seed_sha():
        assert booted["head"] == _seed_sha(), "first boot runs the released seed's commit"
    assert booted["serving_heads"] == [booted["head"]], "the server runs the body checkout"
    assert booted["server_interpreters"] == [VENV_PYTHON], "the body runs in the persistent runtime venv"
    assert booted["venv_marker"]["state"] == "ready"
    _exec(first, "sh", "-c", "for root in data work home; do mkdir -p /ouroboros/$root"
                             " && echo kept > /ouroboros/$root/qa-sentinel; done")

    _docker("rm", "-f", first)
    second, port = _run("parent-2", "managed", [(volume, "/ouroboros")])
    _wait_healthy(second, port, 600)
    recreated = _evidence("managed-recreated", **_body_facts(second))
    assert recreated["head"] == booted["head"] and recreated["venv_marker"] == booted["venv_marker"]
    assert recreated["server_interpreters"] == [VENV_PYTHON]
    for root in ("data", "work", "home"):
        assert _exec(second, "cat", f"/ouroboros/{root}/qa-sentinel") == "kept"

    try:
        _post(port, "/api/command", {"cmd": "/panic"}, timeout=30)
    except Exception:
        pass  # the hard exit may close the connection before the answer
    stopped = _wait_exited(second, 120)
    time.sleep(15)
    settled = _state(second)
    _evidence("managed-panic", exit=stopped.get("ExitCode"), restarts=settled.get("RestartCount"),
              status=settled.get("Status"))
    assert settled.get("Status") == "exited" and settled.get("RestartCount") == stopped.get("RestartCount")
    assert _helper([(volume, "/ouroboros")], "cat /ouroboros/data/state/panic_stop.flag").strip() == "panic"

    _docker("start", second)  # an ordinary start keeps automatic intent: Panic holds
    held = _wait_exited(second, 120)
    _evidence("managed-automatic-start-after-panic", exit=held.get("ExitCode"))
    assert held.get("ExitCode") == 0 and not _healthy(port)

    owner, port = _run("parent-owner", "managed", [(volume, "/ouroboros")], args=["--launch-intent", "owner"],
                       restart="no")
    _wait_healthy(owner, port, 600)
    _evidence("managed-owner-resume", **_body_facts(owner))
    _docker("stop", "-t", "30", owner)
    assert _wait_exited(owner, 60).get("ExitCode") == 0, "an ordinary stop is clean"


@pytest.mark.integration
@pytest.mark.usefixtures("_lane")
def test_s39_managed_split_volumes_boot_and_a_mounted_repository_is_refused_untouched():
    mounts = [(_volume(kind), f"/ouroboros/{kind}") for kind in ("body", "data", "work", "home")]
    split, port = _run("split", "managed", mounts)
    _wait_healthy(split, port, _boot_timeout())
    facts = _evidence("managed-split-boot", **_body_facts(split))
    assert facts["server_interpreters"] == [VENV_PYTHON] and facts["venv_marker"]["state"] == "ready"
    assert facts["serving_heads"] == [facts["head"]]
    _docker("rm", "-f", split)

    repo_volume = _volume("mounted-repo")
    _helper([(repo_volume, "/v")], "echo keep > /v/keep.txt")
    refused, _port = _run("mounted-repo", "managed", [(_volume("mounted-parent"), "/ouroboros"),
                          (repo_volume, BODY_REPO)], restart="no")
    state = _wait_exited(refused, 300)
    logs = _docker("logs", refused, check=False)
    _evidence("managed-mounted-repo", exit=state.get("ExitCode"), tail=(logs.stdout + logs.stderr)[-1500:])
    assert state.get("ExitCode") != 0
    assert "mount point without a checkout" in logs.stdout + logs.stderr
    assert _helper([(repo_volume, "/v")], "ls -A /v").split() == ["keep.txt"], "nothing was moved"


@pytest.mark.integration
@pytest.mark.usefixtures("_lane")
def test_s39_a_second_container_on_shared_data_is_refused():
    volume = _volume("shared")
    serving, port = _run("shared-1", "managed", [(volume, "/ouroboros")])
    _wait_healthy(serving, port, _boot_timeout())
    second, _port = _run("shared-2", "managed", [(volume, "/ouroboros")], restart="no")
    state = _wait_exited(second, 300)
    logs = _docker("logs", second, check=False)
    _evidence("managed-shared-data", exit=state.get("ExitCode"), tail=(logs.stdout + logs.stderr)[-800:])
    assert state.get("ExitCode") == 0 and "Another instance already running" in logs.stdout + logs.stderr
    assert _healthy(port)

    data = _volume("image-data")
    immutable, image_port = _run("image-1", "image", [(data, "/data")],
                                 env={"OUROBOROS_DATA_DIR": "/data", "OUROBOROS_LAUNCH_INTENT": "automatic"})
    _wait_healthy(immutable, image_port, 600)
    rival, _rival_port = _run("image-2", "image", [(data, "/data")], env={"OUROBOROS_DATA_DIR": "/data"},
                              restart="no")
    state = _wait_exited(rival, 120)
    logs = _docker("logs", rival, check=False)
    _evidence("image-shared-data", exit=state.get("ExitCode"), tail=(logs.stdout + logs.stderr)[-800:])
    assert state.get("ExitCode") == 2 and "another Ouroboros server serves this data directory" in logs.stdout + logs.stderr
    assert _healthy(image_port)


# ---------------------------------------------------------------------------
# F — an Update adding a local-wheel dependency, offline recreation, rollback
# ---------------------------------------------------------------------------

F_WHEEL = "ouroboros_s39_marker-1.0.0-py3-none-any.whl"
F_MARK = f"S39_LOCAL_WHEEL_{RUN_ID}"
F_REQUIREMENT = f"ouroboros-s39-marker @ file:///qa/wheels/{F_WHEEL}"
F_TASK = "Run the S39 dependency check: import ouroboros_s39_marker in the runtime and report its MARK."
# A bare ``python`` in run_command is the agent's own interpreter (OUROBOROS_AGENT_PYTHON, which
# server.py sets from its sys.executable): in the managed body, the runtime venv.
F_IMPORT_STEP = {"tool": "run_command", "arguments": {
    "cmd": ["python", "-c", "import ouroboros_s39_marker as marker, sys; print(marker.MARK, sys.executable)"],
    "cwd": "task_drive"}}


def _marker_wheel(mark: str) -> bytes:
    """A pure-Python wheel built from bytes: the dependency the Update adds, installable offline."""
    dist = "ouroboros_s39_marker-1.0.0.dist-info"
    files = {
        "ouroboros_s39_marker/__init__.py": f"MARK = {mark!r}\n".encode(),
        f"{dist}/METADATA": b"Metadata-Version: 2.1\nName: ouroboros-s39-marker\nVersion: 1.0.0\n",
        f"{dist}/WHEEL": b"Wheel-Version: 1.0\nGenerator: ouroboros-s39\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = "".join(
        f"{path},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()},{len(data)}\n"
        for path, data in files.items()) + f"{dist}/RECORD,,\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as wheel:
        for path, data in {**files, f"{dist}/RECORD": record.encode()}.items():
            wheel.writestr(path, data)
    return buffer.getvalue()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _dependency_commit(candidate: pathlib.Path, scratch: pathlib.Path) -> tuple:
    """Clone the candidate and commit one runtime-lock line naming the local wheel; returns
    (clone, the candidate's SHA, the dependency commit's SHA)."""
    work = scratch / "candidate"
    _git("clone", "--quiet", "--branch", "ouroboros", str(candidate), str(work))
    base = _git("-C", str(work), "rev-parse", "HEAD")
    lock = work / "requirements-runtime.lock"
    text = lock.read_text(encoding="utf-8")
    lock.write_text(text + ("" if text.endswith("\n") else "\n") + F_REQUIREMENT + "\n", encoding="utf-8")
    _git("-C", str(work), "-c", "user.name=S39 Driver", "-c", "user.email=s39@e2e.invalid",
         "commit", "--quiet", "-am", "test: S39 local-wheel dependency")
    return work, base, _git("-C", str(work), "rev-parse", "HEAD")


def _stage_candidate(candidate: pathlib.Path, scratch: pathlib.Path) -> tuple:
    """Stream the candidate plus its dependency commit, and the wheel, into a fresh volume;
    returns (volume, the candidate's SHA, the dependency commit's SHA). Docker may run in a VM
    that sees none of the host's paths, so nothing is bind-mounted."""
    work, base, target = _dependency_commit(candidate, scratch)
    bundle = scratch / "candidate.bundle"
    _git("-C", str(work), "bundle", "create", str(bundle), "HEAD", "refs/heads/ouroboros")
    volume = _volume("qa")
    with bundle.open("rb") as stream:
        _helper([(volume, "/qa")], "cat > /qa/candidate.bundle && git clone --quiet --bare /qa/candidate.bundle "
                                   "/qa/candidate.git && rm /qa/candidate.bundle", stdin=stream)
    _helper([(volume, "/qa")], f"mkdir -p /qa/wheels && cat > /qa/wheels/{F_WHEEL}", stdin=_marker_wheel(F_MARK))
    return volume, base, target


def _wait_new_generation(name: str, previous: set, served, timeout: float = 1800) -> None:
    """Wait until no server process of ``name`` is one of ``previous`` and ``served()`` holds: the
    checkout's HEAD moves before the old server exits, so only a new PID proves the new code runs."""
    deadline = time.monotonic() + timeout
    while True:
        pids = {server["pid"] for server in _server_facts(name)}
        if pids and not pids & previous and served():
            return
        assert time.monotonic() < deadline, f"{name} never served a new generation (servers {sorted(pids)})"
        time.sleep(3)


def _import_marker(name: str) -> str:
    return _exec(name, VENV_PYTHON, "-c", "import ouroboros_s39_marker as marker; print(marker.MARK)")


@pytest.mark.integration
@pytest.mark.usefixtures("_lane")
def test_s39_an_update_adding_a_local_wheel_survives_offline_recreation_and_rolls_back(tmp_path):
    candidate = os.environ.get("OUROBOROS_CONTAINER_CANDIDATE_GIT", "").strip()
    if not candidate:
        pytest.skip("set OUROBOROS_CONTAINER_CANDIDATE_GIT to a host repository with the candidate on 'ouroboros'")
    qa, base_sha, target = _stage_candidate(pathlib.Path(candidate), tmp_path)
    volume = _volume("update")
    stub = ScriptedStubModel([F_IMPORT_STEP])
    # Preseeded settings: the stub through the relay, and the development channel's branch.
    settings = keyless_settings(stub, OPENAI_COMPATIBLE_BASE_URL=RELAY_BASE_URL, OUROBOROS_UPDATE_CHANNEL="development")
    assert_settings_keyless(settings)
    _put_file([(volume, "/ouroboros")], f"{MANAGED_DATA}/settings.json", json.dumps(settings, indent=2).encode())

    first, port = _run("update-1", "managed", [(volume, "/ouroboros"), (qa, "/qa:ro")])
    _wait_healthy(first, port, _boot_timeout())
    before = _evidence("update-seed", **_body_facts(first))
    if _seed_sha():
        assert before["head"] == _seed_sha()
    _exec(first, "sh", "-c", f"echo kept > {MANAGED_DATA}/qa-sentinel")
    official = _exec(first, "git", "-C", BODY_REPO, "remote", "get-url", "managed").removesuffix(".git")
    for spelling in (official, f"{official}.git"):  # the Update fetches the official remote: serve it locally
        _exec(first, "git", "-C", BODY_REPO, "config", "--add", "url./qa/candidate.git.insteadOf", spelling)
    seed_servers = {server["pid"] for server in _server_facts(first)}
    plan = _post(port, "/api/update/preflight", {}, timeout=300).get("merge_plan") or {}
    _evidence("update-preflight", plan={k: plan.get(k) for k in ("available", "kind", "base_sha", "target_sha")})
    assert plan.get("target_sha") == target, plan
    applied = _post(port, "/api/update/apply", {"strategy": "auto_merge", "expected_base_sha": plan.get("base_sha"),
                                                 "expected_target_sha": target}, timeout=900)
    assert applied.get("status") == "ok" and applied.get("restarting") is True, applied
    _wait_new_generation(first, seed_servers, lambda: _healthy(port))
    updated = _evidence("update-landed", **_body_facts(first))
    assert updated["serving_heads"] == [target] and updated["server_interpreters"] == [VENV_PYTHON]
    assert updated["venv_marker"]["updated_at"] == before["venv_marker"]["updated_at"], \
        "the venv was updated in place, never recreated"
    assert _import_marker(first) == F_MARK, "exit 42's dependency sync installed the Update's wheel"

    # Recreation without the wheel's volume, without a network and with pip's index disabled.
    _docker("rm", "-f", first)
    second, _none = _run("update-2", "managed", [(volume, "/ouroboros")], network="none", env={"PIP_NO_INDEX": "1"})

    def api(method, path, payload=None, timeout=60.0):
        return _api_call(second, MANAGED_DATA, method, path, payload, timeout)

    with stub, _ModelRelay(second, _stub_port(stub)):
        _wait_served(second, MANAGED_DATA, 900)
        recreated = _evidence("update-recreated-offline", **_body_facts(second))
        assert recreated["head"] == target and recreated["serving_heads"] == [target]
        assert recreated["venv_marker"] == updated["venv_marker"]
        task_id = str(_api_json(api, "POST", "/api/tasks", {
            "description": F_TASK, "memory_mode": "forked", "actor_id": "s39-driver", "source": "s39-driver",
            "metadata": {"source": "s39-driver", "delegation_role": "root"}}).get("task_id") or "")
        task = _wait_api(api, lambda: (lambda row: row if row.get("status") in TERMINAL else {})(
            _api_json(api, "GET", f"/api/tasks/{task_id}")), 600, f"task {task_id} never finished")
        imported = [text for kind, body in list(stub.calls) if F_MARK in (text := body_text(body))]
        _evidence("update-task-import", task=task_id, status=task.get("status"), kinds=stub.kinds(),
                  imported=bool(imported))
        assert task.get("status") == "completed", task
        assert imported and VENV_PYTHON in imported[-1], "the task imported the wheel from the body's runtime venv"

        target_servers = {server["pid"] for server in _server_facts(second)}
        rolled = _api_json(api, "POST", "/api/git/rollback", {"target": base_sha}, 300)
        assert rolled.get("restarting") is True, rolled
        _wait_new_generation(second, target_servers, lambda: bool(_supervisor_ready(api)), 900)
        rolled_back = _evidence("update-rolled-back", **_body_facts(second))
    assert rolled_back["serving_heads"] == [base_sha] and rolled_back["server_interpreters"] == [VENV_PYTHON]
    assert F_REQUIREMENT not in _exec(second, "cat", f"{BODY_REPO}/requirements-runtime.lock"), "the old lock is back"
    assert _import_marker(second) == F_MARK, "rollback restores code and its lock, not the package set (disclosed)"
    assert _exec(second, "cat", f"{MANAGED_DATA}/qa-sentinel") == "kept"
