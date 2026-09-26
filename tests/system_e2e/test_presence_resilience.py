"""S33 — Presence under real waits keeps the owner's controls and receipts
answering, and a retried or abandoned turn stays ONE physical turn, as a REAL
consumer of the candidate: the working tree copied by ``candidate_checkout``,
served by a keyless isolated server, driven over the SAME Host Service and
main-app HTTP surfaces a transport and the SPA use, against the scripted
loopback model.

THE TWO MECHANISMS (reproduced on this harness in the sprint's isolated repro):

1. EXECUTOR STARVATION. A Presence turn waiting on the model, on a Presence slot
   or on its conversation's turn kept an ``asyncio.to_thread`` worker, and the
   Host Service shares the main app's event loop and so its ONE default
   executor. Twelve waiting turns on a twelve-thread executor stalled
   ``/api/state``, Host ``/presence/delivery`` receipts and the owner's Stop (no
   cancel intent was minted; the task later *completed*), while ``/api/health``
   — pure async — kept answering 200 in 2 ms.
2. SYNCHRONOUS HOST AUTH ON THE LOOP. Token authentication, permission and
   admission ran synchronously ON the event loop, so one slow authentication
   froze every request of both apps, health included.

HOW THE WAITS ARE BUILT (event-gated, never a timed race):

* the server tree's default executor is pinned to 12 threads (the incident
  host's ``min(32, cpu + 4)``) by a child-only ``sitecustomize`` (``_HOOK_SOURCE``);
* ``OUROBOROS_PRESENCE_MAX_ACTIVE=2``: e00/e01 are HELD at the loopback model by
  ``ModelGate``s (the two active turns), e02-e05 open four more rooms (slot
  waits), e06-e11 are each room's second event (conversation waits);
* e00's HTTP client DISCONNECTS while its turn is at the model and a
  transport-style retry of the SAME event follows (the 13th request);
* the slow authentication is the same hook: it reports every
  ``HostServiceContext.authenticate_token_payload`` to the test's loopback
  ``HostAuthGate``, which HOLDS one skill's call until released (the repro needed
  4 000-file skill payloads for the same effect).

WHAT IS ASSERTED. ``/api/health`` answering is never a pass; it is only recorded
as the diagnosis that tells a blocked loop from a starved executor.

* WHILE every wait is still held, inside the fixture's client windows (10 s
  reads, 15 s Stop): ``/api/state`` answers a ready snapshot and a fresh v1
  receipt is recorded — two rounds, so one lucky free thread cannot pass — and
  the owner's Stop of a held pooled task answers ``ok`` with its durable
  ``requested`` cancel intent and a ``cancelled`` terminal;
* with only the auth held: the same, plus another skill's complete turn;
* after release: every event answers ``message`` with its OWN reply, the retry
  answers e00's reply, a later replay answers the identical projection, the model
  saw every event EXACTLY once, and every recorded receipt is a durable chat row;
* the owner's Panic under the combined load answers inside the read window and
  the whole server tree is gone, with the panic exit code and flag.

KNOWN LIMITS. The knob resizes only DEFAULT-sized pools: a candidate that sizes
the loop's default executor explicitly fails the premise by name instead of
passing on a bigger pool. The hold models a slow AUTHENTICATION; permission and
admission are exercised on the same request path but never held separately. The
stubs prove Host wiring only, never a model's or a transport's behaviour.
"""

from __future__ import annotations

import http.client
import json
import pathlib
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.candidate_checkout import CandidateError, candidate_checkout, require_candidate_interpreter
from tests.system_e2e.harness import (
    LANE_MOCK,
    REPO_ROOT,
    ArtifactOracle,
    KeylessIsolatedServer,
    ModelGate,
    ScriptedStubModel,
    assert_settings_keyless,
    clone_repo,
    keyless_settings,
    message_text,
    require_lane,
    submit_running,
    wait_until,
    write_settings_file,
)

S33_DEFAULT_EXECUTOR_WORKERS = 12   # the incident host; one held event per thread
S33_MAX_ACTIVE = 2
S33_READ_WINDOW_SEC = 10.0          # /api/state and a receipt must answer inside this
S33_STOP_WINDOW_SEC = 15.0          # the owner's Stop must answer inside this
S33_TURN_WINDOW_SEC = 60.0          # a complete unheld turn while only the auth is held
S33_HOLD_SEC = 240.0                # every gate's bound: a broken run fails, never hangs
S33_SETTLE_SEC = 180.0              # after release, every held request answers inside this
S33_PANIC_EXIT_SEC = 60.0

PROVIDER = "s33chat"
ACCOUNT = "s33-account"
BEHAVIOR_SKILL = "s33-presence-profile"
PROBE_SKILL = "s33-probe"
SLOW_SKILL = "s33-slow"
HELD_TRANSPORTS = tuple(f"s33-t{i}" for i in range(4))  # <= 4 in flight each (the Host caps 5)
CONTROL_MARKER = "[S33-CONTROL-TASK]"
# The product's default install ports: Panic sweeps its BOUND port, so the run
# refuses to Panic unless the server provably bound somewhere else.
LIVE_DEFAULT_PORTS = frozenset({8765, 8766, 8767})

# The host frames THIS turn's input with its exact source facts
# (presence_context.frame_presence_user_content: "Source facts: {json}"); the
# event id is ``<provider>:<run>:<label>``, so a model call names its event
# exactly, and a reply or an earlier turn quoted into context never does.
_SOURCE_EVENT_RE = re.compile(r'"source_event_id": "' + re.escape(PROVIDER) + r':([0-9a-f]{8}):([a-z0-9]+)"')

# Loaded ONLY into the isolated server tree (a ``sitecustomize`` on the child's
# PYTHONPATH), never into this process. Each knob is off unless its env var is set:
#   S33_DEFAULT_EXECUTOR_WORKERS — every DEFAULT-sized ThreadPoolExecutor (the
#     loop's lazily created default executor: what ``asyncio.to_thread`` uses)
#     gets this many threads; explicitly sized pools are untouched; every
#     construction is logged to S33_EXECUTOR_LOG as proof of the applied size.
#   S33_AUTH_GATE_URL — ``HostServiceContext.authenticate_token_payload`` reports
#     each authenticated Host request (skill, and whether it ran ON the event-loop
#     thread) to the test's HostAuthGate, which may hold it: the event-gated
#     stand-in for a slow synchronous authentication.
_HOOK_SOURCE = r'''"""S33 fixture hook: isolated server tree only (tests/system_e2e/test_presence_resilience.py)."""
import os
import sys

_WORKERS = os.environ.get("S33_DEFAULT_EXECUTOR_WORKERS", "").strip()
if _WORKERS:
    import json
    import concurrent.futures.thread as _cft

    _original_init = _cft.ThreadPoolExecutor.__init__

    def _init(self, max_workers=None, *args, **kwargs):
        forced = max_workers is None
        _original_init(self, int(_WORKERS) if forced else max_workers, *args, **kwargs)
        log_path = os.environ.get("S33_EXECUTOR_LOG", "").strip()
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"pid": os.getpid(), "forced_default": forced,
                                         "max_workers": self._max_workers}) + "\n")
            except OSError:
                pass

    _cft.ThreadPoolExecutor.__init__ = _init

_GATE = os.environ.get("S33_AUTH_GATE_URL", "").strip()
if _GATE:
    import importlib.abc
    import importlib.machinery

    def _report(skill):
        import asyncio
        import urllib.parse
        import urllib.request

        try:
            asyncio.get_running_loop()
            on_loop = "1"
        except RuntimeError:
            on_loop = "0"
        query = urllib.parse.urlencode({"skill": skill, "on_loop": on_loop})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            opener.open(_GATE + "/auth?" + query, timeout=300).read()
        except Exception:
            pass  # a vanished gate must never change what the product does

    def _patch(module):
        context = getattr(module, "HostServiceContext", None)
        original = getattr(context, "authenticate_token_payload", None)
        if original is None:
            return

        def authenticate_token_payload(self, raw_token):
            skill, payload = original(self, raw_token)
            _report(skill)
            return skill, payload

        context.authenticate_token_payload = authenticate_token_payload

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "ouroboros.gateway.host_service":
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            exec_module = spec.loader.exec_module

            def patched(module):
                exec_module(module)
                _patch(module)

            spec.loader.exec_module = patched
            return spec

    sys.meta_path.insert(0, _Finder())
'''


# ---------------------------------------------------------------------------
# Loopback gates
# ---------------------------------------------------------------------------

class HostAuthGate:
    """The test side of the auth hook: records every authenticated Host request
    and HOLDS the first one of ``hold_skill`` until ``release`` (bounded by
    ``timeout``, like ``ModelGate``), so "authentication is slow right now" is a
    state the scenario controls instead of a latency it races."""

    def __init__(self, *, timeout: float = S33_HOLD_SEC) -> None:
        self.timeout = float(timeout)
        self.hold_skill = ""
        self.arrived = threading.Event()
        self.release = threading.Event()
        self.timed_out = False
        self.calls: list = []            # (skill, ran_on_event_loop) in arrival order
        self._cv = threading.Condition()
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib callback name
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                skill = (query.get("skill") or [""])[0]
                hold = False
                with outer._cv:
                    outer.calls.append((skill, (query.get("on_loop") or ["0"])[0] == "1"))
                    if skill and skill == outer.hold_skill and not outer.arrived.is_set():
                        hold = True
                        outer.arrived.set()
                    outer._cv.notify_all()
                if hold and not outer.release.wait(outer.timeout):
                    outer.timed_out = True
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *_args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def count(self, skills) -> int:
        with self._cv:
            return sum(1 for skill, _ in self.calls if skill in skills)

    def wait_count(self, skills, count: int, timeout: float) -> bool:
        with self._cv:
            return self._cv.wait_for(
                lambda: sum(1 for skill, _ in self.calls if skill in skills) >= count, timeout)

    def on_loop(self) -> dict:
        with self._cv:
            return {skill: on for skill, on in self.calls}

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)


def _user_texts(body: dict) -> list:
    return [message_text(m) for m in (body.get("messages") or [])
            if isinstance(m, dict) and m.get("role") == "user"]


def _event_labels(run: str, body: dict) -> list:
    """The S33 events a model call serves, read from the host's framing of the
    turn's own input (never from the system prompt, which may list other work)."""
    return sorted({label for text in _user_texts(body)
                   for rid, label in _SOURCE_EVENT_RE.findall(text) if rid == run})


def _event_id(run: str, label: str) -> str:
    return f"{PROVIDER}:{run}:{label}"


def _reply(run: str, label: str) -> str:
    return f"S33 reply to event {label} of run {run}."


class _Arrivals:
    """Every model call's S33 event labels, recorded at ARRIVAL — before any
    hold — so a held call is counted while it is still in flight."""

    def __init__(self, run: str) -> None:
        self.run = run
        self.rows: list = []
        self._lock = threading.Lock()

    def __call__(self, body: dict) -> None:
        with self._lock:
            self.rows.append(_event_labels(self.run, body))

    def per_event(self) -> dict:
        counts: dict = {}
        with self._lock:
            for labels in self.rows:
                for label in labels:
                    counts[label] = counts.get(label, 0) + 1
        return counts


class _AllGates:
    def __init__(self, *gates) -> None:
        self.gates = gates

    def __call__(self, body: dict) -> None:
        for gate in self.gates:
            gate(body)


def _reply_step(run: str):
    def step(body: dict) -> dict:
        labels = _event_labels(run, body)
        return {"final": _reply(run, labels[0] if len(labels) == 1 else "unmarked")}
    return step


# ---------------------------------------------------------------------------
# Seeding: the owner-side facts a transport skill and a Presence profile need
# (the same helpers tests/test_host_service_api.py and test_presence_admission.py use)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Transport:
    name: str
    access: str        # the X-Skill-Token value: a fixture string, never a credential
    binding_id: str


def _seed_behavior(data_root: pathlib.Path) -> None:
    from ouroboros.presence_capabilities import (
        PresenceSelection,
        PresenceState,
        PresenceToolTarget,
        presence_state_fingerprint,
        save_presence_state,
    )
    from ouroboros.presence_profile import parse_presence_profile, presence_request_fingerprint
    from ouroboros.skill_loader import SkillReviewState, load_skill, save_enabled, save_review_state

    skill_dir = data_root / "skills" / "external" / BEHAVIOR_SKILL
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {BEHAVIOR_SKILL}\n"
        "description: Neutral S33 presence fixture.\n"
        "version: 0.1.0\n"
        "type: instruction\n"
        "presence:\n"
        "  instructions: Participate helpfully in the selected room. Answer in one short sentence.\n"
        "  context_topics: [community-guidelines]\n"
        "  runtime_defaults:\n"
        "    model_slot: main\n"
        "    inline_max_rounds: 4\n"
        "  capability_requests:\n"
        "    - id: history\n"
        "      kind: tool\n"
        "      required: true\n"
        "      purpose: Read relevant room history.\n"
        "---\n"
        "# S33 presence profile\n",
        encoding="utf-8",
    )
    loaded = load_skill(skill_dir, data_root)
    assert loaded is not None and not loaded.load_error, getattr(loaded, "load_error", "not loaded")
    save_enabled(data_root, loaded.name, True)
    save_review_state(data_root, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    profile = parse_presence_profile(loaded.manifest, skill_dir)
    assert profile is not None
    save_presence_state(
        data_root, loaded.name,
        PresenceState((PresenceSelection(
            presence_request_fingerprint(profile.capability_requests[0]),
            PresenceToolTarget("builtin", "chat_history"),
        ),)),
        expected_state_fingerprint=presence_state_fingerprint(PresenceState()),
    )


def _seed_transport(data_root: pathlib.Path, name: str) -> _Transport:
    from ouroboros.gateway.host_service import AUTH_TOKEN_FILENAME
    from ouroboros.presence_bindings import (
        PresenceBinding,
        PresenceEndpoint,
        new_presence_binding_id,
        save_presence_binding,
    )
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
        save_skill_grants,
    )
    from ouroboros.utils import atomic_write_json

    skill_dir = data_root / "skills" / "external" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: S33 transport fixture\nversion: 0.1\n"
        "type: extension\nentry: plugin.py\npermissions: [presence]\nsubscribe_events: []\n---\n# S33 transport\n",
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text("def register(api):\n    pass\n", encoding="utf-8")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_root, name, SkillReviewState(status="pass", content_hash=content_hash))
    save_enabled(data_root, name, True)
    save_skill_grants(data_root, name, [], content_hash=content_hash, requested_keys=[],
                      granted_permissions=["presence"], requested_permissions=["presence"])
    access = f"s33-{name}-{uuid.uuid4().hex}"
    atomic_write_json(data_root / "state" / "skills" / name / AUTH_TOKEN_FILENAME,
                      {"token": access, "content_hash": content_hash, "issued_at": "s33"})
    binding = save_presence_binding(data_root, PresenceBinding(
        new_presence_binding_id(), name, BEHAVIOR_SKILL,
        PresenceEndpoint(PROVIDER, ACCOUNT, "*", ""),      # account-wide: any room
        PresenceEndpoint(PROVIDER, ACCOUNT, "s33-room", ""),
    ))
    return _Transport(name, access, binding.binding_id)


# ---------------------------------------------------------------------------
# Wire helpers (http.client: no proxy lookup, and an explicit disconnect)
# ---------------------------------------------------------------------------

def _call(port: int, method: str, path: str, payload=None, *, headers=None, timeout: float) -> dict:
    started = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=timeout)
    try:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        conn.request(method, path, body=data,
                     headers={**({"Content-Type": "application/json"} if data is not None else {}),
                              **(headers or {})})
        response = conn.getresponse()
        raw = response.read()
        try:
            body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except ValueError:
            body = {"raw": raw[:300].decode("utf-8", "replace")}
        return {"status": response.status, "body": body if isinstance(body, dict) else {"value": body},
                "error": "", "sec": round(time.monotonic() - started, 3)}
    except TimeoutError:
        return {"status": 0, "body": {}, "error": "timeout", "sec": round(time.monotonic() - started, 3)}
    except (OSError, http.client.HTTPException) as exc:
        return {"status": 0, "body": {}, "error": f"{type(exc).__name__}: {exc}",
                "sec": round(time.monotonic() - started, 3)}
    finally:
        conn.close()


def _event_body(transport: _Transport, event_id: str, room: str, text: str) -> dict:
    return {
        "binding_id": transport.binding_id,
        "delivery_reporting_version": 0,
        "event": {
            "source_event_id": event_id, "provider": PROVIDER, "account_id": ACCOUNT,
            "conversation_id": room, "thread_id": "",
            "conversation_key": "caller-controlled-key-is-ignored",
            "actor": {"platform_actor_id": "user-7", "username": "alex", "display_name": "Alex"},
            "conversation": {"title": "S33 room"}, "message": {"message_id": event_id[-8:]},
            "text": text,
        },
    }


def _receipt(delivery_id: str) -> dict:
    return {
        "schema_version": 1, "delivery_id": delivery_id, "part_id": "0", "state": "delivered",
        "provider": PROVIDER, "account_id": ACCOUNT, "conversation_id": "s33-probe-room", "thread_id": "",
        "text": "S33 probe receipt", "format": "markdown", "message": {"message_id": delivery_id[:8]},
        "origin": {"kind": "automatic", "task_id": "", "source_event_id": ""},
    }


class _Turn(threading.Thread):
    """One transport request for one event, answered whenever the Host answers it."""

    def __init__(self, port: int, transport: _Transport, body: dict, label: str) -> None:
        super().__init__(name=f"s33-turn-{label}", daemon=True)
        self.port, self.transport, self.body, self.label = port, transport, body, label
        self.result: dict = {}

    def run(self) -> None:
        self.result = _call(self.port, "POST", "/presence/turn", self.body,
                            headers={"X-Skill-Token": self.transport.access},
                            timeout=S33_HOLD_SEC + S33_SETTLE_SEC)


class _HookedServer(KeylessIsolatedServer):
    """``KeylessIsolatedServer`` whose child additionally carries the S33 hook env."""

    def __init__(self, clone, data_root, settings_path, *, extra_env: dict) -> None:
        super().__init__(clone, data_root, settings_path)
        self.extra_env = dict(extra_env)

    def _env(self) -> dict:
        env = super()._env()
        env.update(self.extra_env)
        return env


def _jsonl_file(path: pathlib.Path) -> list:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return [row for row in rows if isinstance(row, dict)]


# ---------------------------------------------------------------------------
# The scenario driver
# ---------------------------------------------------------------------------

def _run_s33(candidate, root: pathlib.Path, *, held_turns: bool, slow_auth: bool, panic: bool = False) -> None:
    run = uuid.uuid4().hex[:8]
    root = pathlib.Path(root)
    data_root = root / "data"
    data_root.mkdir(parents=True)
    (root / "home").mkdir()
    hook_dir = root / "hook"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(_HOOK_SOURCE, encoding="utf-8")
    executor_log = root / "executor_pools.jsonl"

    arrivals = _Arrivals(run)
    # The control task's OWN input (other turns' system prompts may list it).
    control_gate = ModelGate(lambda body: bool(body.get("tools")) and CONTROL_MARKER in "".join(_user_texts(body)[:1]),
                             timeout=S33_HOLD_SEC)
    held = {label: ModelGate(lambda body, label=label: _event_labels(run, body) == [label],
                             timeout=S33_HOLD_SEC)
            for label in ("e00", "e01")}
    model_gates = (control_gate, *held.values())
    turns: dict = {}
    receipts: list = []
    probes: dict = {}
    server = None
    with HostAuthGate() as auth_gate, ScriptedStubModel(
            [_reply_step(run)] * 400, gate=_AllGates(arrivals, *model_gates)) as stub:
        try:
            settings = keyless_settings(stub, OUROBOROS_PRESENCE_MAX_ACTIVE=S33_MAX_ACTIVE,
                                        OUROBOROS_MAX_WORKERS=2)
            assert_settings_keyless(settings)
            write_settings_file(data_root / "settings.json", settings)
            _seed_behavior(data_root)
            transports = {name: _seed_transport(data_root, name)
                          for name in (*HELD_TRANSPORTS, SLOW_SKILL, PROBE_SKILL)}
            server = _HookedServer(candidate, data_root, data_root / "settings.json", extra_env={
                "HOME": str(root / "home"),
                "PYTHONPATH": str(hook_dir),
                "S33_DEFAULT_EXECUTOR_WORKERS": str(S33_DEFAULT_EXECUTOR_WORKERS),
                "S33_EXECUTOR_LOG": str(executor_log),
                "S33_AUTH_GATE_URL": auth_gate.url,
                # Inherited by every descendant: the Panic survivor scan's membership.
                "S33_TREE": f"s33-tree-{run}",
            })
            server.start(ready_timeout=300)
            host_port = server.host_service_port
            oracle = ArtifactOracle(server.data_root)

            # PREMISE: the server process's default executor really is 12 threads.
            pools = [row for row in _jsonl_file(executor_log) if row.get("pid") == server.proc.pid]
            assert any(row.get("forced_default") and row.get("max_workers") == S33_DEFAULT_EXECUTOR_WORKERS
                       for row in pools), (
                "premise not established: the server never built a DEFAULT-sized executor at "
                f"{S33_DEFAULT_EXECUTOR_WORKERS} threads (an explicitly sized loop executor bypasses "
                f"the knob): {pools}")

            def diagnosis() -> str:
                on_loop = auth_gate.on_loop()
                return json.dumps({
                    "health": probes.get("health"),
                    "auth_ran_on_event_loop": on_loop,
                    "auth_calls": len(auth_gate.calls),
                    "model_arrivals": arrivals.per_event(),
                    "answered_early": {label: turn.result for label, turn in turns.items()
                                       if not turn.is_alive()},
                    "executor_pools": pools,
                }, default=str)

            # A REAL pooled task held mid model call: the target of the owner's Stop.
            control_id = submit_running(server, f"{CONTROL_MARKER} Say hello in one line and finish.")
            assert control_gate.arrived.wait(120), f"the control task never reached the model: {stub.kinds()}"

            if held_turns:
                rooms = [f"s33-room-{i}" for i in range(6)]
                plan = [(f"e{i:02d}", rooms[i % 6], transports[HELD_TRANSPORTS[i % 4]])
                        for i in range(S33_DEFAULT_EXECUTOR_WORKERS)]
                bodies = {label: _event_body(transport, _event_id(run, label), room,
                                             f"S33 event {label}: say one short sentence.")
                          for label, room, transport in plan}
                by_label = {label: transport for label, _room, transport in plan}

                # e00: sent on a connection this test will abandon mid-turn.
                abandoned = http.client.HTTPConnection("127.0.0.1", host_port, timeout=S33_HOLD_SEC)
                abandoned.request("POST", "/presence/turn", body=json.dumps(bodies["e00"]).encode("utf-8"),
                                  headers={"Content-Type": "application/json",
                                           "X-Skill-Token": by_label["e00"].access})
                assert held["e00"].arrived.wait(120), f"e00 never reached the model: {diagnosis()}"
                turns["e01"] = _Turn(host_port, by_label["e01"], bodies["e01"], "e01")
                turns["e01"].start()
                assert held["e01"].arrived.wait(120), f"e01 never reached the model: {diagnosis()}"
                # The transport gives up on e00 while its turn is at the model ...
                abandoned.sock.shutdown(socket.SHUT_RDWR)
                abandoned.close()
                # ... ten more events queue behind the two active turns, and e00 is retried.
                for label, _room, transport in plan[2:]:
                    turns[label] = _Turn(host_port, transport, bodies[label], label)
                    turns[label].start()
                turns["e00-retry"] = _Turn(host_port, by_label["e00"], bodies["e00"], "e00-retry")
                turns["e00-retry"].start()
                assert auth_gate.wait_count(HELD_TRANSPORTS, len(plan) + 1, 60), (
                    f"the Host authenticated only {auth_gate.count(HELD_TRANSPORTS)} of {len(plan) + 1} "
                    f"held turn requests (a blocked event loop, or the auth seam moved): {diagnosis()}")
                assert arrivals.per_event() == {"e00": 1, "e01": 1}, (
                    f"exactly the two active turns may be at the model: {diagnosis()}")

            if slow_auth:
                auth_gate.hold_skill = SLOW_SKILL
                turns["slow"] = _Turn(host_port, transports[SLOW_SKILL], _event_body(
                    transports[SLOW_SKILL], _event_id(run, "slow"), "s33-slow-room",
                    "S33 event slow: say one short sentence."), "slow")
                turns["slow"].start()
                assert auth_gate.arrived.wait(60), (
                    "premise not established: the auth hook never saw the slow skill's Host "
                    "authentication (HostServiceContext.authenticate_token_payload moved?): "
                    f"{diagnosis()}")

            if panic:
                _assert_panic_tears_down(server, oracle, f"S33_TREE=s33-tree-{run}", diagnosis)
                return

            # Recorded, never asserted: a blocked loop times this out, a starved
            # executor answers it in milliseconds.
            probes["health"] = _call(server.port, "GET", "/api/health", timeout=S33_READ_WINDOW_SEC)
            for round_index in range(2):
                state = _call(server.port, "GET", "/api/state", timeout=S33_READ_WINDOW_SEC)
                probes[f"state-{round_index}"] = state
                assert state["status"] == 200 and state["body"].get("supervisor_ready") is True, (
                    f"/api/state did not answer a ready snapshot inside {S33_READ_WINDOW_SEC:.0f}s "
                    f"while Presence waited (round {round_index}): {state}; {diagnosis()}")
                delivery_id = f"s33-{run}-receipt-{round_index}"
                recorded = _call(host_port, "POST", "/presence/delivery", _receipt(delivery_id),
                                 headers={"X-Skill-Token": transports[PROBE_SKILL].access},
                                 timeout=S33_READ_WINDOW_SEC)
                probes[f"receipt-{round_index}"] = recorded
                assert 200 <= recorded["status"] < 300 and recorded["body"].get("ok") is True \
                    and recorded["body"].get("duplicate") is not True, (
                        f"a fresh receipt was not recorded inside {S33_READ_WINDOW_SEC:.0f}s while "
                        f"Presence waited (round {round_index}): {recorded}; {diagnosis()}")
                receipts.append(delivery_id)

            if slow_auth and not held_turns:
                # Another skill's whole turn completes while one authentication is held.
                fast = _call(host_port, "POST", "/presence/turn", _event_body(
                    transports[HELD_TRANSPORTS[0]], _event_id(run, "fast"), "s33-fast-room",
                    "S33 event fast: say one short sentence."),
                    headers={"X-Skill-Token": transports[HELD_TRANSPORTS[0]].access},
                    timeout=S33_TURN_WINDOW_SEC)
                assert fast["status"] == 200 and fast["body"].get("outcome") == "message" \
                    and fast["body"].get("text") == _reply(run, "fast"), (
                        f"another skill's turn did not complete while one Host authentication was "
                        f"held: {fast}; {diagnosis()}")

            stop = _call(server.port, "POST", f"/api/tasks/{control_id}/cancel", {},
                         timeout=S33_STOP_WINDOW_SEC)
            # Accepted is enough here; the durable intent and the terminal below are the proof.
            assert 200 <= stop["status"] < 300 and stop["body"].get("ok") is True, (
                f"the owner's Stop did not answer inside {S33_STOP_WINDOW_SEC:.0f}s while Presence "
                f"waited: {stop}; {diagnosis()}")
            requested = wait_until(lambda: [
                row for row in oracle.supervisor_rows("cancel_intent")
                if row.get("task_id") == control_id and row.get("event") == "requested"], 10)
            assert requested, f"no durable cancel intent for {control_id}: {oracle.supervisor_rows('cancel_intent')}"
            cancelled = wait_until(
                lambda: oracle.task_result(control_id).get("status") == "cancelled", 30)
            assert cancelled, f"the stopped task did not settle cancelled: {oracle.task_result(control_id)}"

            # Everything above happened INSIDE the hold: nothing had been released,
            # no held request has been answered, and no waiting event reached the model.
            assert not any(gate.release.is_set() for gate in model_gates) and not auth_gate.release.is_set()
            early = {label: turn.result for label, turn in turns.items() if not turn.is_alive()}
            assert not early, f"held Presence requests were answered before release: {early}"
            expected_arrivals = ({"e00": 1, "e01": 1} if held_turns else {}) | (
                {"fast": 1} if slow_auth and not held_turns else {})
            assert arrivals.per_event() == expected_arrivals, (
                f"a waiting event reached the model during the hold: {diagnosis()}")

            auth_gate.release.set()
            for gate in model_gates:
                gate.release.set()
            deadline = time.monotonic() + S33_SETTLE_SEC
            for turn in turns.values():
                turn.join(timeout=max(0.0, deadline - time.monotonic()))
            unanswered = [label for label, turn in turns.items() if turn.is_alive()]
            assert not unanswered, f"held requests never answered after release: {unanswered}"
            assert not auth_gate.timed_out and not any(gate.timed_out for gate in model_gates)

            refs: dict = {}
            for label, turn in turns.items():
                event = "e00" if label == "e00-retry" else label
                body = turn.result.get("body") or {}
                assert turn.result.get("status") == 200 and body.get("ok") is True \
                    and body.get("outcome") == "message" and body.get("text") == _reply(run, event), (
                        f"{label} did not answer its own reply after release: {turn.result}")
                refs[label] = str(body.get("turn_ref") or "")
                stored = oracle.task_result(refs[label])
                assert stored.get("status") == "completed", (label, stored)
            assert len(set(refs.values())) == len(refs), f"two events shared one turn: {refs}"

            if held_turns:
                # The abandoned attempt and its retry were ONE turn; a later replay
                # answers the identical projection without another model call.
                replay = _call(host_port, "POST", "/presence/turn", bodies["e00"],
                               headers={"X-Skill-Token": by_label["e00"].access}, timeout=60)
                assert replay["status"] == 200, replay
                assert {key: replay["body"].get(key) for key in ("outcome", "text", "turn_ref")} == {
                    key: turns["e00-retry"].result["body"].get(key) for key in ("outcome", "text", "turn_ref")
                }, (replay, turns["e00-retry"].result)
            expected = {label: 1 for label in (
                *(bodies if held_turns else ()), *(("slow",) if slow_auth else ()),
                *(("fast",) if slow_auth and not held_turns else ()))}
            assert arrivals.per_event() == expected, (
                f"an event bought more (or fewer) than one model call: {arrivals.per_event()}")

            recorded_ids = {
                str(((row.get("transport") or {}).get("delivery") or {}).get("delivery_id") or "")
                for row in (json.loads(line) for line in oracle.chat_bytes().decode("utf-8").splitlines()
                            if '"presence_delivery"' in line)
            }
            assert set(receipts) <= recorded_ids, (receipts, recorded_ids)
        finally:
            auth_gate.release.set()
            for gate in model_gates:
                gate.release.set()
            for turn in turns.values():
                turn.join(timeout=S33_SETTLE_SEC)
            if server is not None:
                server.stop()
            for turn in turns.values():
                turn.join(timeout=10)


def _assert_panic_tears_down(server, oracle, member: str, diagnosis) -> None:
    from ouroboros.config import PANIC_EXIT_CODE
    from ouroboros.process_containment import pids_with_env_marker

    # Safety precondition, not the claim: Panic sweeps the port its server BOUND
    # and the Host port it reads from settings. Both must provably be this
    # server's own, or the sweep could reach the operator's live install.
    assert not {server.port, server.host_service_port} & LIVE_DEFAULT_PORTS
    assert oracle.server_port() == server.port, "the server's bound port is not provably its own"
    assert pids_with_env_marker(member), "the survivor scan cannot see the live server tree"
    answered = _call(server.port, "POST", "/api/command", {"cmd": "/panic"}, timeout=S33_READ_WINDOW_SEC)
    assert 200 <= answered["status"] < 300, (
        f"the owner's Panic was not accepted inside {S33_READ_WINDOW_SEC:.0f}s under held Presence "
        f"load: {answered}; {diagnosis()}")
    exited = wait_until(lambda: server.proc.poll() is not None, S33_PANIC_EXIT_SEC)
    assert exited, f"Panic did not end the server within {S33_PANIC_EXIT_SEC:.0f}s: {diagnosis()}"
    assert server.proc.returncode == PANIC_EXIT_CODE, server.proc.returncode
    assert (server.data_root / "state" / "panic_stop.flag").is_file()
    gone = wait_until(lambda: pids_with_env_marker(member) == [], 30)
    assert gone, f"Panic left members of the server tree alive: {pids_with_env_marker(member)}"


# ---------------------------------------------------------------------------
# Mock lane
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def s33_candidate(tmp_path_factory):
    """The tree every S33 server runs. Preferred: ONE byte-faithful
    ``candidate_checkout`` of the working tree (uncommitted changes included),
    which needs a dependency-only interpreter. The nightly lane installs the
    checkout editable, which the candidate probe refuses; a CLEAN tree is then
    byte-identical to HEAD, so a HEAD clone serves the same bytes. A dirty tree
    under such an interpreter is refused, never silently tested as HEAD."""
    require_lane(LANE_MOCK)
    try:
        require_candidate_interpreter()
    except CandidateError as exc:
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT,
                               check=True, capture_output=True, text=True).stdout
        if dirty.strip():
            pytest.fail(f"{exc} — and the working tree is dirty, so a HEAD clone would not be the candidate")
        yield clone_repo(tmp_path_factory.mktemp("s33"))
        return
    with candidate_checkout(REPO_ROOT, tmp_path_factory.mktemp("s33") / "candidate",
                            origin_proof=True) as candidate:
        yield candidate


@pytest.mark.integration
@pytest.mark.serial
def test_s33_presence_waits_leave_state_receipts_and_stop_answering(s33_candidate, tmp_path):
    """Mechanism 1 alone: 12 waiting events on a 12-thread default executor."""
    require_lane(LANE_MOCK)
    _run_s33(s33_candidate, tmp_path / "executor", held_turns=True, slow_auth=False)


@pytest.mark.integration
@pytest.mark.serial
def test_s33_held_host_authentication_leaves_the_event_loop_serving(s33_candidate, tmp_path):
    """Mechanism 2 alone: one authentication held, nothing else waiting."""
    require_lane(LANE_MOCK)
    _run_s33(s33_candidate, tmp_path / "auth", held_turns=False, slow_auth=True)


@pytest.mark.integration
@pytest.mark.serial
def test_s33_presence_waits_and_held_authentication_together(s33_candidate, tmp_path):
    require_lane(LANE_MOCK)
    _run_s33(s33_candidate, tmp_path / "both", held_turns=True, slow_auth=True)


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.skipif(sys.platform == "win32", reason="the survivor scan is POSIX-only")
def test_s33_owner_panic_tears_down_under_held_presence_load(s33_candidate, tmp_path):
    require_lane(LANE_MOCK)
    _run_s33(s33_candidate, tmp_path / "panic", held_turns=True, slow_auth=True, panic=True)


# ---------------------------------------------------------------------------
# Default lane: the seams the fixture relies on
# ---------------------------------------------------------------------------

def test_presence_resilience_fixture_seams_still_exist():
    """The auth hook wraps a method by NAME and the scenario attributes model calls
    by marker; a drift in either must be a named failure here, not a silently
    vacuous hold in the mock lane."""
    from ouroboros.presence_context import frame_presence_user_content

    compile(_HOOK_SOURCE, "sitecustomize.py", "exec")
    source = (REPO_ROOT / "ouroboros" / "gateway" / "host_service.py").read_text(encoding="utf-8")
    assert "class HostServiceContext" in source
    assert "def authenticate_token_payload(self, raw_token" in source
    run = "0123abcd"
    body = _event_body(_Transport("t", "a", "b"), _event_id(run, "e00"), "room", "hello")
    framed = frame_presence_user_content(
        {"_presence_turn": True, "metadata": {"presence": {"event": body["event"], "observed_text": "hello"}}},
        f"hello {_reply(run, 'e01')}")
    assert _event_labels(run, {"messages": [
        {"role": "system", "content": f'"source_event_id": "{_event_id(run, "e02")}"'},
        {"role": "user", "content": framed},
    ]}) == ["e00"]
