"""Claudexor v3 control-plane transport.

Pure I/O, like every gateway (docs/DEVELOPMENT.md "Gateway Rules"): discover the
loopback daemon, negotiate the protocol, and translate the typed ``/v2`` surface
into Python primitives. No routing, no policy, no harness identity branches — the
caller asks for a CAPABILITY (an access profile, an opaque route id) and reads the
manifest Claudexor publishes.

Two Claudexor I/O surfaces live here, not one: the HTTP control plane, and the run
tree Claudexor writes on disk. The engine deliberately records some APPLIED facts
only as run artifacts (an attempt's harness HOME is one), so a caller that has to
verify what was enforced has nowhere else to read them — and the path layout is
Claudexor's, which makes it this module's business rather than a policy caller's.

Token custody: the daemon bearer token grants the ENTIRE ``/v2`` surface. It is
read here, held in this process only, and never returned to a caller, written into
a ``ToolContext``, put in a child's environment, or handed to a harness sandbox.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ouroboros.config import (
    CLAUDEXOR_MIN_VERSION,
    CLAUDEXOR_PROTOCOL_MAJOR,
    get_claudexor_quota_refresh_timeout_sec,
)

log = logging.getLogger(__name__)

CONTROL_API_REL = ".claudexor/v3/daemon/control-api.json"
PROTOCOL_HEADER = "X-Claudexor-Protocol-Major"
_CONNECT_TIMEOUT_SEC = 5.0
# The client-wide read default, and the CEILING on any polling/self-bounding caller's ask
# (`delegate_progress.poll_bound` reads it): a per-request value above it is not a bound
# at all, it is a hung read granted more rope than it would have had. Generous on
# purpose: most calls here would rather wait than fail, and a run start can take a while
# to answer.
_READ_TIMEOUT_SEC = 60.0
# The FLOOR under non-strict bounded wait/admission asks. Owned-daemon startup also
# imports it as the CEILING for each fast loopback liveness probe. Every ordinary
# `delegate_wait` poll asks for what its window has left; this is where that narrowing
# stops, because a nearly spent window asking for its own 0.2s turns a healthy daemon
# into a timeout, while the 60s default would outrun the very deadline the wait clamps
# itself to (measured, 4.51s of wall against a 4.2s window, and that was a fast daemon).
# Five seconds is a real answer from a healthy loopback daemon and a rounding error
# against the finalization grace the clamp reserves. It bounds the READ phase only:
# `_request` applies the caller bound to every HTTPX phase. Strict review polls add
# their total wall-clock bound outside this phase-local adapter.
# Passed per request via ``_request(timeout_sec=...)``; it never changes the default.
SHORT_POLL_TIMEOUT_SEC = 5.0
_ATTEMPTS_REL = "attempts"
_ATTEMPT_RECORD = "attempt.yaml"


class ClaudexorUnavailable(RuntimeError):
    """Typed lane refusal: the delegated route cannot run right now.

    Carries the machine-readable ``code`` so callers classify instead of matching
    prose. Never raised for an ordinary in-run failure — only for "this transport
    is not usable".

    ``required_actions`` retains the daemon's TOP-LEVEL ``ControlProblem.requiredActions``
    string list when the refusal carried one (e.g. the reconcile 409's
    ``retry_setup_reconciliation``), bounded to the daemon's own wire limit. It is
    a preserved fact for the typed error seam, not a client action framework.
    """

    def __init__(self, code: str, message: str, *, status_code: int = 0,
                 required_actions: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code = str(code or "claudexor_unavailable")
        self.status_code = int(status_code or 0)
        self.required_actions = tuple(required_actions or ())


# Cross-repo contract (B1): the engine's window-exhausted RunFailure codes. A
# newer engine (rotation PR-A) reports a spent credential POOL under its own
# code; both heal on a timer, so both map onto the exhausted class below — with
# the ORIGINAL code preserved. Any other code stays a generic
# ClaudexorUnavailable: fail-open, old engines emitting code:null included.
WINDOW_EXHAUSTED_CODES = ("subscription_window_exhausted", "credential_pool_exhausted")


class ClaudexorSubscriptionWindowExhausted(ClaudexorUnavailable):
    """The subscription window is spent and heals on a timer, not on payment."""

    def __init__(self, message: str, *, reset_at: str = "", status_code: int = 0,
                 code: str = "subscription_window_exhausted") -> None:
        super().__init__(code, message, status_code=status_code)
        self.reset_at = str(reset_at or "")


@dataclass(frozen=True)
class DaemonEndpoint:
    """Loopback control-plane address. ``token`` stays host-side (see module doc)."""

    host: str
    port: int
    token: str


def engine_at_least(version: str, minimum: str) -> bool:
    """Is the reported engine at or past ``minimum``? THE version-floor predicate.

    One reader, so the handshake's transport floor and a lane's own feature floor
    cannot disagree about what "old enough to refuse" means. An unparsable or absent
    version compares as ``(0,)`` — below every floor, so it fails CLOSED.
    """
    pair: List[tuple] = []
    for value in (version, minimum):
        parts: List[int] = []
        for chunk in str(value or "").split("."):
            digits = "".join(c for c in chunk if c.isdigit())
            parts.append(int(digits) if digits else 0)
        pair.append(tuple(parts or [0]))
    return pair[0] >= pair[1]


def operator_home() -> pathlib.Path:
    """The home ``discover_daemon`` reads the control token from.

    Named once because it is the exact directory a delegated harness must never
    inherit: it holds ``~/.claudexor/v3/daemon/token``, which grants the whole ``/v2``
    surface. Both the discovery below and the applied-isolation check read it here.
    """
    return pathlib.Path(os.path.expanduser("~"))


def discover_daemon(home: Optional[pathlib.Path] = None) -> DaemonEndpoint:
    """Read the daemon descriptor plus the referenced token.

    With an explicit ``home`` this reads that home's ``~/.claudexor/v3`` layout
    verbatim. With none, the OWNED daemon is preferred WHEN PROVISIONED (D30):
    once Ouroboros has spawned its own daemon under the data-plane config dir,
    every default discovery — delegated subagents, review sessions, the account
    surfaces — talks to that one, and the operator's personal daemon is left
    alone. An unprovisioned owned home falls through to the operator layout,
    which is the entire pre-D30 behavior; the cutover is the owner's own
    provisioning action, never a silent boot-time switch.
    """
    if home is None:
        from ouroboros.claudexor_daemon import owned_daemon_provisioned, owned_descriptor_path

        if owned_daemon_provisioned():
            return _endpoint_from_descriptor(owned_descriptor_path())
    root = pathlib.Path(home) if home is not None else operator_home()
    return _endpoint_from_descriptor(root / CONTROL_API_REL)


def discover_daemon_at(config_dir: pathlib.Path) -> DaemonEndpoint:
    """Discovery for an explicit ``CLAUDEXOR_CONFIG_DIR`` root.

    Under an override the override IS the complete relocatable root, so the
    descriptor lives at ``<config_dir>/daemon/control-api.json`` — a different
    shape from the default ``~/.claudexor/v3`` layout ``discover_daemon`` reads.
    """
    return _endpoint_from_descriptor(pathlib.Path(config_dir) / "daemon" / "control-api.json")


def _endpoint_from_descriptor(control_path: pathlib.Path) -> DaemonEndpoint:
    try:
        raw = json.loads(control_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ClaudexorUnavailable(
            "daemon_not_discovered",
            f"Claudexor control-api descriptor not found at {control_path}",
        ) from exc
    except (OSError, ValueError) as exc:
        raise ClaudexorUnavailable(
            "daemon_descriptor_unreadable",
            f"Claudexor control-api descriptor unreadable: {type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(raw, dict):
        raise ClaudexorUnavailable("daemon_descriptor_unreadable", "control-api.json is not an object")
    host = str(raw.get("host") or "").strip()
    token_path = str(raw.get("tokenPath") or "").strip()
    try:
        port = int(raw.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    if not host or port <= 0 or not token_path:
        raise ClaudexorUnavailable(
            "daemon_descriptor_incomplete",
            "control-api.json is missing host/port/tokenPath",
        )
    try:
        # The SAME set as the descriptor read four lines up, and for the same reason: a
        # path out of a JSON descriptor can carry an embedded null and a token file can
        # hold bytes that are not UTF-8, both of which `read_text` raises as `ValueError`
        # (`UnicodeDecodeError` is one), and either escaping here is a traceback where a
        # typed refusal belongs. `RuntimeError` is deliberately NOT in this set, and the
        # v6.87.44 comment claiming it covered a symlink loop was wrong: `read_text` on a
        # loop raises `OSError` (ELOOP) — `resolve()` is what raises `RuntimeError`, which
        # is why `_resolved` in delegate.py catches it and this does not. It was also a
        # hazard, since `ClaudexorUnavailable` IS a `RuntimeError`: the moment this block
        # grew a call that refuses typed, the catch would re-wrap it under this code.
        token = pathlib.Path(token_path).read_text(encoding="utf-8").strip()
    except (OSError, ValueError) as exc:
        raise ClaudexorUnavailable(
            "daemon_token_unreadable",
            f"Claudexor daemon token unreadable: {type(exc).__name__}",
        ) from exc
    if not token:
        raise ClaudexorUnavailable("daemon_token_unreadable", "Claudexor daemon token file is empty")
    if not _is_loopback(host):
        # The token this descriptor points at grants the ENTIRE /v2 control API — start
        # runs, read every artifact, cancel anything. The loopback boundary was
        # documented and never enforced, so anything able to write one file under
        # ~/.claudexor could redirect the bearer to a host it controls: token
        # exfiltration plus authenticated SSRF, from a file write. Refused BEFORE the
        # client exists, so no request can be built against a non-loopback endpoint.
        raise ClaudexorUnavailable(
            "daemon_endpoint_not_loopback",
            f"Claudexor control-api descriptor names the non-loopback host {host!r}. The "
            f"daemon control token grants the whole /v2 surface and is only ever sent to "
            f"the local daemon; refusing rather than shipping it off-host.",
        )
    return DaemonEndpoint(host=host, port=port, token=token)


def _is_loopback(host: str) -> bool:
    """True only for a literal loopback ADDRESS, or the exact name ``localhost``.

    An IP literal is decided by the stdlib (``127.0.0.0/8``, ``::1``, and their
    zone/bracket spellings), never by string prefixes: ``127.0.0.1.evil.com`` is a
    NAME, and ``0x7f.1`` is not one this reader will guess at. Resolution is
    deliberately NOT attempted — a name that resolves to loopback today can resolve
    elsewhere on the next lookup, so only ``localhost`` (which every platform pins to
    loopback) is accepted by name. Everything else is refused.
    """
    import ipaddress

    candidate = str(host or "").strip().strip("[]")
    if not candidate:
        return False
    if candidate.lower() == "localhost":
        return True
    candidate = candidate.split("%", 1)[0]      # drop an IPv6 zone id
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


class ClaudexorGateway:
    """Thin typed client over the Claudexor ``/v2`` control API."""

    def __init__(self, endpoint: Optional[DaemonEndpoint] = None, *, home: Optional[pathlib.Path] = None):
        self._endpoint = endpoint if endpoint is not None else discover_daemon(home)
        self._engine_version = ""
        self._engine_build_sha = ""
        # trust_env=False: a shell HTTP(S)_PROXY must never be able to intercept the
        # loopback control plane (the bearer token rides these requests).
        self._client = httpx.Client(
            base_url=f"http://{self._endpoint.host}:{self._endpoint.port}",
            timeout=httpx.Timeout(_READ_TIMEOUT_SEC, connect=_CONNECT_TIMEOUT_SEC),
            trust_env=False,
            headers={
                "Authorization": f"Bearer {self._endpoint.token}",
                PROTOCOL_HEADER: str(CLAUDEXOR_PROTOCOL_MAJOR),
                "Content-Type": "application/json",
            },
        )

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            log.debug("Claudexor client close failed", exc_info=True)

    def __enter__(self) -> "ClaudexorGateway":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @property
    def engine_version(self) -> str:
        return self._engine_version

    @property
    def engine_build_sha(self) -> str:
        return self._engine_build_sha

    # -- transport -------------------------------------------------------------

    def _request(self, method: str, path: str, *, json_body: Any = None,
                 headers: Optional[Dict[str, str]] = None,
                 timeout_sec: Optional[float] = None) -> Any:
        # ``timeout_sec`` replaces the client's read default for THIS call only, for a
        # caller that is bounding ITSELF — today every `delegate_wait` poll, each asking
        # for what its window has left, floored at ``SHORT_POLL_TIMEOUT_SEC`` and never
        # raised above the default (``delegate_progress.poll_bound``). The caller's
        # bound applies to connect as well as read/write/pool phases. HTTPX treats
        # these as phase-local, so strict total budgeting happens in bounded_poll.
        # Absent, the call is not passed at all rather than passed as None — httpx reads
        # an explicit ``timeout=None`` as "no timeout", which is the opposite of the
        # default it would otherwise inherit.
        if timeout_sec is None:
            bound: Dict[str, Any] = {}
        else:
            bounded = max(0.000001, float(timeout_sec))
            bound = {
                "timeout": httpx.Timeout(
                    bounded,
                    connect=min(_CONNECT_TIMEOUT_SEC, bounded),
                )
            }
        try:
            response = self._client.request(method, path, json=json_body,
                                            headers=headers or None, **bound)
        except httpx.HTTPError as exc:
            raise ClaudexorUnavailable(
                "daemon_unreachable",
                f"Claudexor daemon unreachable: {type(exc).__name__}: {exc}",
            ) from exc
        if response.status_code >= 400:
            raise self._problem(response)
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise ClaudexorUnavailable(
                "malformed_response",
                f"Claudexor returned a non-JSON body for {method} {path}: {exc}",
            ) from exc

    def _problem(self, response: httpx.Response) -> ClaudexorUnavailable:
        """Translate a ControlProblem body into a typed refusal."""
        code = f"http_{response.status_code}"
        message = response.text[:500]
        context: Dict[str, Any] = {}
        required_actions: tuple[str, ...] = ()
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            code = str(body.get("code") or code)
            message = str(body.get("message") or message)
            raw_context = body.get("context")
            context = raw_context if isinstance(raw_context, dict) else {}
            # The daemon serializes `requiredActions` at the ControlProblem TOP LEVEL
            # (`daemon-server` projects the field beside code/message; `problem-safety`
            # bounds the wire list to at most 16 redacted strings of at most 512 chars).
            # It is deliberately NOT read from `context`: no producer puts it there, and
            # a context sniff would resurrect the exact had-it-both-ways bug the reset
            # classification below documents. The bound is mirrored so a foreign body
            # cannot balloon the retained tuple.
            raw_actions = body.get("requiredActions")
            if isinstance(raw_actions, list):
                required_actions = tuple(
                    str(item)[:512] for item in raw_actions[:16]
                    if isinstance(item, str) and item
                )
        # The CODE decides, exactly as every other classification on this seam does.
        # Sniffing `context` for a reset key instead had it both ways: no producer puts
        # `resetsAt`/`resets_at`/`cooldownUntil` in a ControlProblem context (a spent
        # window is reported as a run-detail RunFailure, and `cooldown_until` lives in a
        # quota snapshot), so the transient class was unreachable — while any unrelated
        # refusal that happened to carry one, an `idempotency_conflict` say, would have
        # been announced as a spent subscription window and retried on a timer.
        if code in WINDOW_EXHAUSTED_CODES:
            return ClaudexorSubscriptionWindowExhausted(
                message, reset_at=str(context.get("resetsAt") or ""),
                status_code=response.status_code, code=code,
            )
        return ClaudexorUnavailable(code, message, status_code=response.status_code,
                                    required_actions=required_actions)

    # -- operations ------------------------------------------------------------

    def handshake(self, *, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        """Negotiate protocol major and enforce the minimum engine version.

        ``timeout_sec`` bounds this call the way ``get_run`` is bounded: a caller
        holding a clamped window pays for the OPENING round trip out of that window,
        so an unbounded handshake could spend it before the window began.
        """
        body = self._request(
            "POST", "/v2/handshake", timeout_sec=timeout_sec,
            json_body={"protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR, "client": "ouroboros"},
        )
        if not isinstance(body, dict):
            raise ClaudexorUnavailable("malformed_response", "handshake did not return an object")
        if not body.get("compatible") or int(body.get("protocolMajor") or 0) != CLAUDEXOR_PROTOCOL_MAJOR:
            raise ClaudexorUnavailable(
                "protocol_incompatible",
                f"Claudexor refused protocol major {CLAUDEXOR_PROTOCOL_MAJOR}: {body!r}",
            )
        engine = body.get("engine") if isinstance(body.get("engine"), dict) else {}
        version = str(engine.get("version") or "")
        if not engine_at_least(version, CLAUDEXOR_MIN_VERSION):
            raise ClaudexorUnavailable(
                "engine_too_old",
                f"Claudexor {version or 'unknown'} is older than the required {CLAUDEXOR_MIN_VERSION}",
            )
        self._engine_version = version
        self._engine_build_sha = str(engine.get("sha") or "")
        return body

    def agent_capabilities(self) -> Dict[str, Any]:
        body = self._request("GET", "/v2/agent-capabilities")
        return body if isinstance(body, dict) else {}

    def harnesses(self) -> List[Dict[str, Any]]:
        """GET /v2/harnesses — per-harness status rows WITH the full manifest.

        The agent-capability catalog is a derived projection that deliberately
        drops the manifest's transport flags (``json_schema_output``,
        ``interactive``); this is the surface that still carries them, so
        transport-capability questions are asked here, not of the catalog.
        """
        body = self._request("GET", "/v2/harnesses")
        rows = body.get("harnesses") if isinstance(body, dict) else None
        return [row for row in (rows or []) if isinstance(row, dict)]

    def quota_state(self) -> Dict[str, Any]:
        """GET /v2/quota once, retaining its one-epoch evidence envelope."""
        body = self._request("GET", "/v2/quota")
        return body if isinstance(body, dict) else {}

    def refresh_quota(self) -> Dict[str, Any]:
        """POST /v2/quota once, returning the foreground evidence envelope."""
        return self._request(
            "POST",
            "/v2/quota",
            json_body={},
            timeout_sec=get_claudexor_quota_refresh_timeout_sec(),
        )

    def quota_snapshots(self) -> List[Dict[str, Any]]:
        body = self.quota_state()
        snapshots = body.get("snapshots") if isinstance(body, dict) else None
        return [row for row in (snapshots or []) if isinstance(row, dict)]

    def quota_absences(self) -> List[Dict[str, Any]]:
        """Profiles whose quota could NOT be read (a 429/failed refresh, no login).

        Legacy compatibility projection of the same one-epoch quota envelope.
        An absence is typed evidence that quota could not be read for a profile;
        route health treats that state as unknown and therefore fail-open.
        """
        body = self.quota_state()
        absences = body.get("absences") if isinstance(body, dict) else None
        return [row for row in (absences or []) if isinstance(row, dict)]

    def register_project(self, root: str) -> str:
        """Register a run root and return its project id (idempotent per root).

        Claudexor answers the FIRST run against an unregistered root with
        404 ``project_not_registered``, so registration is a required step, not an
        optimization. Re-registering an existing root returns the existing id.
        """
        body = self._request(
            "POST", "/v2/projects",
            json_body={"root": str(root)},
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )
        project_id = str((body or {}).get("id") or "") if isinstance(body, dict) else ""
        if not project_id:
            raise ClaudexorUnavailable("malformed_response", "project registration returned no id")
        return project_id

    def remove_project(self, project_id: str) -> Dict[str, Any]:
        """Retire a project registration. Non-destructive: artifacts are retained."""
        body = self._request("DELETE", f"/v2/projects/{project_id}")
        return body if isinstance(body, dict) else {}

    def find_project_id(self, root: str) -> str:
        body = self._request("GET", "/v2/projects")
        target = str(root)
        for row in (body or {}).get("projects") or [] if isinstance(body, dict) else []:
            if isinstance(row, dict) and str(row.get("root") or "") == target:
                return str(row.get("id") or "")
        return ""

    def start_run(self, request: Dict[str, Any], *, idempotency_key: str = "") -> Dict[str, Any]:
        """POST /v2/runs with a caller-built, schema-valid request body.

        ``idempotency_key`` is the caller's LOGICAL INVOCATION ID — minted once per
        intended invocation, reused verbatim on a transport retry. The engine's replay
        check (control-api ``handleRunCreate`` → daemon command store) runs BEFORE any
        preflight: a replayed key with the byte-identical request returns the ORIGINAL
        accepted job's handle, and a replayed key with a different request digest is a
        409 ``idempotency_conflict`` — so a retry must reuse the id AND reproduce the
        body exactly. A fresh random key per POST makes an accepted start whose
        response was lost come back as a SECOND live run that nothing knows about; a
        stale content-stable key makes a deliberate re-run come back as the finished
        OLD run. Callers with no invocation identity keep the random default.
        """
        body = self._request(
            "POST", "/v2/runs",
            json_body=dict(request),
            headers={"Idempotency-Key": str(idempotency_key or "") or uuid.uuid4().hex},
        )
        if not isinstance(body, dict):
            raise ClaudexorUnavailable("malformed_response", "run start returned no handle")
        return body

    def create_thread(self, request: Dict[str, Any], *, idempotency_key: str) -> Dict[str, Any]:
        """Create one durable v3 conversation thread.

        Claudexor owns continuity and profile routing. This client only carries
        the strict request and the caller's stable idempotency identity.
        """
        body = self._request(
            "POST", "/v2/threads", json_body=dict(request),
            headers={"Idempotency-Key": str(idempotency_key)},
        )
        if not isinstance(body, dict) or not str(body.get("id") or ""):
            raise ClaudexorUnavailable("malformed_response", "thread create returned no id")
        return body

    def start_thread_turn(
        self, thread_id: str, request: Dict[str, Any], *, idempotency_key: str,
    ) -> Dict[str, Any]:
        """Append one turn through the public v3 thread pipeline."""
        from urllib.parse import quote

        body = self._request(
            "POST", f"/v2/threads/{quote(str(thread_id), safe='')}/turns",
            json_body=dict(request), headers={"Idempotency-Key": str(idempotency_key)},
        )
        if not isinstance(body, dict):
            raise ClaudexorUnavailable("malformed_response", "thread turn returned no handle")
        return body

    def get_thread(self, thread_id: str) -> Dict[str, Any]:
        """Read turns, native-session bindings, and continuity receipts."""
        from urllib.parse import quote

        body = self._request("GET", f"/v2/threads/{quote(str(thread_id), safe='')}")
        return body if isinstance(body, dict) else {}

    def get_run(self, run_id: str, *, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        body = self._request("GET", f"/v2/runs/{run_id}", timeout_sec=timeout_sec)
        return body if isinstance(body, dict) else {}

    def get_run_artifact(self, run_id: str, path: str) -> bytes:
        """GET /v2/runs/:id/artifacts/<path> — the FULL artifact body, raw bytes.

        The run detail's ``primaryOutput.text`` is a bounded 256 KiB PREVIEW
        (control-api ``PRIMARY_OUTPUT_PREVIEW_BYTES``) with ``bytes``/``truncated``
        beside it; this route is where the full file actually lives. The engine serves
        text artifacts through ``redactSecrets`` (so the served length may differ from
        the on-disk ``bytes`` when a secret was rewritten), refuses credential-shaped
        files with a typed 409, caps text at 4 MiB with a 413, and answers a
        retention-reclaimed run with a 410 tombstone — all of which surface here as
        typed ``ClaudexorUnavailable`` refusals, never as a silent empty body.
        """
        from urllib.parse import quote

        try:
            response = self._client.request(
                "GET", f"/v2/runs/{quote(str(run_id), safe='')}/artifacts/{quote(str(path), safe='/')}")
        except httpx.HTTPError as exc:
            raise ClaudexorUnavailable(
                "daemon_unreachable",
                f"Claudexor daemon unreachable: {type(exc).__name__}: {exc}",
            ) from exc
        if response.status_code >= 400:
            raise self._problem(response)
        return response.content

    def answer_interaction(self, run_id: str, interaction_id: str,
                           answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """POST /v2/runs/:id/interactions/:iid/answer — deliver one answer set.

        ``answers`` rows are already in the wire shape (``questionId`` /
        ``selectedLabels`` / ``freeText`` — the strict ``ControlInteractionAnswerRequest``);
        this method is transport, not translation.

        The engine's reply is TYPED at every HTTP status it owns: 200 carries
        ``{accepted, status: "delivered"}``, and a 404/409 refusal carries the SAME
        ``ControlInteractionAnswerResponse`` shape with ``status`` ``not_found`` /
        ``already_resolved`` / ``rejected`` (daemon-server answers the route with the
        parsed response at 200/404/409). Any body carrying one of those statuses is
        returned as the ANSWER it is — an engine's ``already_resolved`` is a fact,
        not an outage. What still raises ``ClaudexorUnavailable``: transport
        failures, a bodyless 404 (``no such run``), the 501 of an engine build with
        no answer service, and any other refusal without a typed status.
        """
        from urllib.parse import quote

        path = (f"/v2/runs/{quote(str(run_id), safe='')}"
                f"/interactions/{quote(str(interaction_id), safe='')}/answer")
        try:
            response = self._client.request("POST", path,
                                            json={"answers": list(answers or [])})
        except httpx.HTTPError as exc:
            raise ClaudexorUnavailable(
                "daemon_unreachable",
                f"Claudexor daemon unreachable: {type(exc).__name__}: {exc}",
            ) from exc
        body: Any = None
        if response.content:
            try:
                body = response.json()
            except ValueError:
                body = None
        if isinstance(body, dict) and str(body.get("status") or "") in (
                "delivered", "not_found", "already_resolved", "rejected"):
            return body
        if response.status_code >= 400:
            raise self._problem(response)
        raise ClaudexorUnavailable(
            "malformed_response",
            f"interaction answer returned no typed status (HTTP {response.status_code})",
        )

    def cancel_run(self, run_id: str, *, reason: str = "") -> Dict[str, Any]:
        control: Dict[str, Any] = {"kind": "cancel"}
        if reason:
            control["reason"] = str(reason)
        body = self._request("POST", f"/v2/runs/{run_id}/control", json_body={"control": control})
        return body if isinstance(body, dict) else {}

    # -- account surfaces (D30: read/translate only; the daemon owns ALL auth
    # logic — profiles, login jobs, device-code custody, verification) ---------

    def credential_profiles(self) -> Dict[str, Any]:
        """GET /v2/credential-profiles — profiles + per-harness native rows.

        The daemon's payload already distinguishes the two verification
        truths the UI must show honestly (Q2-а): ``verification_source``
        ``local_store`` (material present, liveness UNPROVEN) vs ``vendor``
        (the vendor answered a request with this credential)."""
        body = self._request("GET", "/v2/credential-profiles")
        return body if isinstance(body, dict) else {}

    def create_credential_profile(self, harness_id: str, profile_id: str,
                                  display_name: str = "") -> Dict[str, Any]:
        request: Dict[str, Any] = {"harnessId": str(harness_id), "profileId": str(profile_id)}
        if display_name:
            request["displayName"] = str(display_name)
        body = self._request(
            "POST", "/v2/credential-profiles", json_body=request,
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )
        return body if isinstance(body, dict) else {}

    def update_credential_profile(self, harness_id: str, profile_id: str,
                                  *, enabled: bool) -> Dict[str, Any]:
        """PATCH /v2/credential-profiles/:harness/:profileId — the engine's own
        Enabled toggle for a NAMED account (``{enabled}`` is the one
        user-settable routing control the profile row carries).

        Translate-only, like every account surface here: the daemon owns the
        registry row and rotation policy, and its refusal is the answer. The
        route exists on 3.5.0 engines already; unified-model engines serve the
        migrated default logins through it too, because those are ordinary
        registry rows there."""
        from urllib.parse import quote

        body = self._request(
            "PATCH",
            f"/v2/credential-profiles/{quote(str(harness_id), safe='')}"
            f"/{quote(str(profile_id), safe='')}",
            json_body={"enabled": bool(enabled)},
        )
        return body if isinstance(body, dict) else {}

    def delete_credential_profile(self, harness_id: str, profile_id: str) -> Dict[str, Any]:
        """DELETE /v2/credential-profiles/:harness/:profileId — the engine's own
        removal contract for a NAMED account.

        Ouroboros never deletes vendor credential material itself: the daemon
        owns the profile record and whatever it stored for it, so removal is a
        request to that owner and its refusal is the answer. There is no
        counterpart for a native CLI login — that account belongs to the
        vendor's own CLI, and simulating a sign-out here would claim an effect
        this process cannot have."""
        from urllib.parse import quote

        body = self._request(
            "DELETE",
            f"/v2/credential-profiles/{quote(str(harness_id), safe='')}"
            f"/{quote(str(profile_id), safe='')}",
        )
        return body if isinstance(body, dict) else {}

    def harness_models(self, harness_id: str) -> List[Dict[str, Any]]:
        """GET /v2/harnesses/:id/models — the discovered model list (owner
        directive: models are a dropdown fed by discovery, never free input)."""
        from urllib.parse import quote

        body = self._request("GET", f"/v2/harnesses/{quote(str(harness_id), safe='')}/models")
        models = body.get("models") if isinstance(body, dict) else None
        return [row for row in (models or []) if isinstance(row, dict)]

    def setup_job_create(self, request: Dict[str, Any], *, idempotency_key: str = "") -> Dict[str, Any]:
        body = self._request(
            "POST", "/v2/setup/jobs", json_body=dict(request),
            headers={"Idempotency-Key": str(idempotency_key or "") or uuid.uuid4().hex},
        )
        return body if isinstance(body, dict) else {}

    def setup_job_call(self, job_id: str, op: str, *, value: str = "") -> Dict[str, Any]:
        """One job-scoped setup call: ``snapshot`` (GET; the transient
        device-code/oauth_url disclosure rides it and is never journaled),
        ``cancel`` (POST), ``input`` (POST /v2/setup/jobs/{id}/input —
        deliver ONE line of user input, the claude OAuth paste-code, to a
        login job that awaits it; engine 3.3.7+), or ``reconcile``
        (POST /v2/setup/jobs/{id}/reconcile — ask the daemon to prove an
        unconfirmed termination's process group empty; supported floor 3.2.0).

        The input value is live login material, the same custody rule as the
        device code: it rides this loopback request once and is never logged,
        stored, or echoed by this client. An engine that predates the input
        route answers 404, which the caller must treat as a typed capability
        gap, not a bug.
        """
        from urllib.parse import quote

        base = f"/v2/setup/jobs/{quote(str(job_id), safe='')}"
        if op == "snapshot":
            body = self._request("GET", f"{base}/snapshot")
        elif op == "cancel":
            body = self._request("POST", f"{base}/cancel")
        elif op == "input":
            body = self._request("POST", f"{base}/input", json_body={"value": str(value)})
        elif op == "reconcile":
            body = self._request("POST", f"{base}/reconcile")
        else:
            raise ValueError(f"unknown setup job op: {op!r}")
        return body if isinstance(body, dict) else {}

    def operations(self) -> List[Dict[str, Any]]:
        """GET /v2/operations — the engine's own implemented-route catalog.

        The handshake advertises this path (``operationsPath``); the catalog is
        how a caller discovers a capability structurally instead of branching
        on version folklore (verified live: ``{protocolMajor, operations:[{id,
        method, path, ...}]}``).
        """
        body = self._request("GET", "/v2/operations")
        ops = body.get("operations") if isinstance(body, dict) else None
        return [row for row in (ops or []) if isinstance(row, dict)]

    def get_settings(self) -> Dict[str, Any]:
        """GET /v2/settings — the daemon's effective settings snapshot.

        The read half of the rotation reconcile (B3): the snapshot's
        ``harnesses`` map carries each configured harness's
        ``profileLimitAction``, so provisioning patches only what is actually
        missing instead of blind-writing every discovered harness. The route
        has served since the v2 boundary existed, so every engine past
        ``CLAUDEXOR_MIN_VERSION`` answers it.
        """
        body = self._request("GET", "/v2/settings")
        return body if isinstance(body, dict) else {}

    def patch_settings(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """POST /v2/settings — the daemon's own live settings patch (the
        write half of the rotation reconcile, D28/B3)."""
        body = self._request("POST", "/v2/settings", json_body=dict(request))
        return body if isinstance(body, dict) else {}

    def set_secret(self, name: str, value: str) -> Dict[str, Any]:
        """Store one managed secret through the daemon's non-journaled route.

        The value stays in this loopback request and is never returned or logged.
        This is transport only; callers own the choice of managed slot.
        """
        body = self._request(
            "POST", "/v2/secrets",
            json_body={"name": str(name), "value": str(value)},
        )
        return body if isinstance(body, dict) else {}


def pending_interactions(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The run detail's live interactive questions, normalized and complete.

    ``GET /v2/runs/:id`` carries ``pendingInteractions`` — full
    ``ControlPendingInteraction`` rows with the question TEXT, header, options and
    ``multi_select``, not just the ``summary.waitingOnUser`` boolean the old wait
    kept. This is the ONE reader of that wire shape: snake_case keys out, absent
    strings normalized to ``None``/empty, rows without an interaction id dropped
    (an unanswerable row is noise, not a question). Purely shape translation — no
    truncation here; bounding belongs to the delivery layer that knows its budget.
    """
    rows = detail.get("pendingInteractions") if isinstance(detail, dict) else None
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        questions: List[Dict[str, Any]] = []
        for question in row.get("questions") or []:
            if not isinstance(question, dict):
                continue
            questions.append({
                "question_id": str(question.get("id") or ""),
                "question": str(question.get("question") or ""),
                "header": str(question.get("header") or "") or None,
                "options": [
                    {"label": str(option.get("label") or ""),
                     "description": str(option.get("description") or "") or None}
                    for option in (question.get("options") or [])
                    if isinstance(option, dict)
                ],
                "multi_select": bool(question.get("multi_select")),
            })
        interaction_id = str(row.get("interactionId") or "")
        if not interaction_id:
            continue
        out.append({
            "interaction_id": interaction_id,
            "source_tool": str(row.get("sourceTool") or "") or None,
            "requested_at": str(row.get("requestedAt") or ""),
            "timeout_at": str(row.get("timeoutAt") or "") or None,
            "questions": questions,
        })
    return out


# -- applied-fact artifacts ----------------------------------------------------


@dataclass(frozen=True)
class AttemptContainment:
    """What one attempt's harness ACTUALLY ran under, as the engine recorded it.

    TWO axes, one record, because they are two different guarantees and reading only
    the first is how a `~`-redirect gets reported as a sandbox:

    ``home_isolated`` — the scoped ``HOME``. ``None`` when the attempt recorded no such
    fact at all: "unverified", which is not "verified false" and must not be collapsed
    into it by a caller reading a bare boolean. The consequence of a recorded false is a
    CANCELLATION, so absence must stay absence.

    ``boundary_mechanism`` — the OS-enforced filesystem boundary the engine APPLIED
    (``""`` when the attempt named none). Silence collapses to "no boundary" here, and
    deliberately not for the home above: an engine that reports nothing is
    indistinguishable from an engine that applied nothing, and the consequence of
    reading it that way is a DISCLOSURE rather than a refusal. That direction is safe;
    the opposite one would let an unconfined run pass as confined.

    ``confinement_unavailable_reason`` — the engine's own typed explanation for a
    missing boundary (e.g. no mechanism exists for this host), read from the SAME
    attempt artifact. Telemetry that AMPLIFIES the unconfined disclosure — never
    an admission token: an old engine that writes nothing here changes no
    decision, and a reason's presence never excuses a recorded FALSE.
    """

    attempt_id: str
    home_isolated: Optional[bool]
    home_dir: str
    boundary_mechanism: str = ""
    confinement_unavailable_reason: str = ""


def attempt_containment(run_dir: str) -> List[AttemptContainment]:
    """Read every attempt's APPLIED containment facts from the run's own artifacts.

    Claudexor writes ``harness_home_isolated`` / ``harness_home_dir`` and the applied
    boundary (``confinement_mechanism``, with ``confinement_verified_denied_path`` as
    its proof) onto ``<runDir>/attempts/<id>/attempt.yaml``. The HOME pair is projected
    onto no ``/v2`` response, so for that half this file is the only evidence a caller
    has that the confinement it asked for was applied rather than merely requested; the
    BOUNDARY half is also served on the run detail as ``candidates[].confinement`` (since
    3.3.6). The artifact is read for both, because it is where the two meet per attempt.

    The mechanism is read as an OPAQUE string. Which mechanisms exist, and on which
    hosts, is the engine's business — Ouroboros asks what was applied and never which
    OS it is sitting on, so the day a second mechanism ships this reader is unchanged.

    Empty means NO EVIDENCE, never "not isolated": the record is written when an
    attempt finishes, so a young run legitimately has none. Unreadable artifacts are
    skipped for the same reason — a caller distinguishes absence from a recorded false.
    """
    root = pathlib.Path(str(run_dir or "")) / _ATTEMPTS_REL
    try:
        attempt_dirs = sorted(entry for entry in root.iterdir() if entry.is_dir())
    except (OSError, ValueError, RuntimeError):
        return []
    import yaml  # type: ignore

    applied: List[AttemptContainment] = []
    for attempt_dir in attempt_dirs:
        try:
            record = yaml.safe_load((attempt_dir / _ATTEMPT_RECORD).read_text(encoding="utf-8"))
        except (OSError, ValueError, RuntimeError, yaml.YAMLError):
            continue
        if not isinstance(record, dict):
            continue
        raw = record.get("harness_home_isolated")
        # A mechanism is only evidence WITH its proof: 3.3.2 records the mechanism
        # beside the path the policy was executed against and refused, on this host,
        # before the harness spawned. A name on its own is the promise the applied-fact
        # block exists to replace, so it is read as no boundary at all.
        mechanism = str(record.get("confinement_mechanism") or "").strip()
        proven = str(record.get("confinement_verified_denied_path") or "").strip()
        applied.append(AttemptContainment(
            attempt_id=str(record.get("attempt_id") or attempt_dir.name),
            home_isolated=raw if isinstance(raw, bool) else None,
            home_dir=str(record.get("harness_home_dir") or ""),
            boundary_mechanism=mechanism if (mechanism and proven) else "",
            confinement_unavailable_reason=str(
                record.get("confinement_unavailable_reason") or ""
            ).strip(),
        ))
    return applied


__all__ = [
    "AttemptContainment",
    "ClaudexorGateway",
    "ClaudexorSubscriptionWindowExhausted",
    "ClaudexorUnavailable",
    "DaemonEndpoint",
    "WINDOW_EXHAUSTED_CODES",
    "attempt_containment",
    "discover_daemon",
    "discover_daemon_at",
    "engine_at_least",
    "operator_home",
    "pending_interactions",
]
