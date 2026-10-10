"""Ouroboros-owned Claudexor daemon (D30).

Ouroboros runs its OWN ``claudexord`` under a data-plane config dir
(``CLAUDEXOR_CONFIG_DIR`` — the override IS the complete relocatable root:
config, credential profiles, secrets, daemon token, socket, runs). The
operator's personal ``~/.claudexor`` state is never read, never imported and
never touched: coexisting daemons per config dir are the engine's own
first-class seam, and the owner's existing logins stay the owner's (accounts
for the Ouroboros home are logged in fresh, through the daemon's own login
jobs).

Lifecycle belongs to the installation, not the process that first needed it:

* spawn through ``process_custody`` (daemon scope) from whichever process first
  needs the daemon — a task worker included; every worker tree-kill spares the
  ledger's live daemon roots (``supervisor.worker_pool_lifecycle.kill_worker_tree``)
  and both server sweeps retain the purpose's legacy session rows, so neither a
  worker's death nor a server generation change ends the daemon's paid runs;
* ATTACH-IF-ALIVE: a live daemon serving our config dir is attached to. A
  custodied startup without a reachable control endpoint is joined across
  managers/processes; caller wait expiry (``daemon_starting``; a silent live listener:
  ``daemon_not_answering``) never kills it. Engine writer election owns the first-launch race;
* ATTACH-ONLY (``review_run_isolation.attach_home``, the isolated contributor
  review, which starts no engine unattached): another data plane's running engine,
  by marker, loopback descriptor and handshake — never started, prepared, rotated,
  claimed or stopped; a missing or dead one is a typed refusal. Own runs stay cancellable;
* STOP-ONLY-WHAT-IS-PROVABLY-OURS: ``stop`` (Panic) terminates the child THIS
  manager spawned and ledger roots confirmed by our marker and measured
  custody fingerprint, with an authenticated endpoint, a typed transport
  failure, or a positively absent descriptor for our own marked startup — a
  prior generation's or a worker's spawn included. Token refusal,
  invalid discovery and incompatible/malformed replies never permit that
  fallback. Never stop a live responder known only by name or by
  the descriptor port (a foreign daemon on a recycled port stays disclosed, not
  killed). A newer runtime pin is never hot-swapped: the live engine keeps
  serving until a planned restart whose landed checkout pins another engine
  ends it (``server_restart._stop_owned_daemon_for_new_pin``), the owner's
  Restart or Panic stops it, or it exits; the next start selects the pin.

Zero auth logic lives here or anywhere in Ouroboros: login jobs, device-code
custody, verification and rotation are the daemon's own product surface,
reached through the ``/v2`` control API (``gateways/claudexor.py``).
"""

from __future__ import annotations
import json
import re

import logging
import os
import pathlib
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, Literal, Optional

from ouroboros.claudexor_startup_failure import (
    ExitFact,
    build_start_failure_record,
    classified_start_failure_record,
    classify_startup_failure,
    describe_log_interval,
    latch_cleared_row,
    read_startup_log_interval,
    start_failure_detail,
    start_failure_label,
    start_failure_row,
)
from ouroboros.review_run_isolation import attach_home, run_cap_from_env
from ouroboros.config import (
    CLAUDEXOR_STARTUP_WAIT_SEC as _SPAWN_WAIT_SEC,
    CLAUDEXOR_STARTUP_POLL_SEC as _SPAWN_POLL_SEC,
    CLAUDEXOR_ADMISSION_WAIT_SEC as _ADMISSION_WAIT_SEC,
    CLAUDEXOR_ADMISSION_POLL_SEC as _ADMISSION_POLL_SEC,
)

log = logging.getLogger(__name__)

_OWNED_DIR_NAME = "claudexor"
CUSTODY_PURPOSE = "claudexor_daemon"
# Admission, distinct from reachability: a 3.4+ daemon serves the authenticated
# handshake BEFORE its admission gate (the body says `servingMode`), while every
# product route answers 503 `daemon_recovery_only` (retryable) until journal
# recovery completes. Recovery can persist indefinitely (blocked journal
# partitions), so the wait is bounded and ends in the typed refusal the 503
# already produces (D28) — never a silent indefinite wait, never a kill of a
# recovering daemon. The 150 ms cadence is the engine CLI's own.

# Engines at/above this version own the limit-action default themselves
# (kind-aware "auto" semantics, Clawdexor A6): subscription profiles rotate,
# metered API keys fail, and the OWNER's explicit choices always win. Blanket
# "rotate" writes from this side would overwrite that judgment, so reconcile
# skips those engines entirely. Confirmed shipped in the actual 3.6.0 release
# (claudexor 31aa51c9, schema limit_action enum carries kind-aware "auto"), so
# this floor names the real release wave (issue #246); it is deliberately not
# CLAUDEXOR_MIN_VERSION (owner decision 5=A: no floor bump).
_ROTATION_AUTO_SEMANTICS_MIN_VERSION = "3.6.0"
_ROTATION_RECEIPT_NAME = "claudexor_rotation_provisioning.json"
_SETUP_ATTACH_ROLE = "setup_attach"
_SHELL_POSIX = "posix"
_SHELL_POWERSHELL = "powershell"
_TRANSPORT_UNREACHABLE = "transport_unreachable"
# The typed answer of ``OwnedClaudexorDaemon.stop_outcome``: an unconfirmed stop
# is already disclosed (critical log + supervisor row) with custody retained.
DaemonStopOutcome = Literal["stopped", "nothing_to_stop", "unconfirmed"]
from ouroboros.config import (
    CLAUDEXOR_OPERATOR_STOP_TIMEOUT_SEC as _OPERATOR_STOP_TIMEOUT_SEC,
    CLAUDEXOR_STOP_EXIT_WAIT_SEC as _STOP_EXIT_WAIT_SEC,
)


def _handshake_serving_mode(body: Any) -> str:
    """The handshake's explicit admission mode, '' when the engine says nothing.

    Only an EXPLICIT ``recovery_only`` ever counts as recovering: pre-3.4
    engines carry no ``servingMode`` at all, and an absent or unknown value must
    read as normal admission — byte-identical behavior for every engine that
    predates the field.
    """
    if not isinstance(body, dict):
        return ""
    return str(body.get("servingMode") or "").strip().lower()


def owned_config_dir() -> pathlib.Path:
    """The data-plane root the owned daemon lives under."""
    from ouroboros.config import DATA_DIR

    return pathlib.Path(DATA_DIR) / _OWNED_DIR_NAME


def attached_endpoint(home: pathlib.Path) -> Any:
    """The attach-only home's engine (``review_run_isolation.attach_home``), provenance first."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable, discover_daemon_at

    try:
        marker = json.loads((home / OWNERSHIP_MARKER).read_text(encoding="utf-8"))
        marked = pathlib.Path(str(marker.get("data_dir") or "")) if isinstance(marker, dict) else None
        if (not isinstance(marker, dict) or marker.get("owner") != "ouroboros" or marked is None
                or not marked.is_absolute() or (marked / _OWNED_DIR_NAME).resolve() != home.resolve()):
            raise ValueError("marker does not name this home as an Ouroboros-owned engine home")
    except (OSError, ValueError, RuntimeError) as exc:  # RuntimeError: a symlink loop in resolve()
        raise ClaudexorUnavailable(
            "attach_provenance_refused",
            f"attach-only engine home {home} has no matching ownership marker "
            f"({type(exc).__name__}: {exc}); nothing is started in its place",
        ) from exc
    return discover_daemon_at(home)


def owned_descriptor_path() -> pathlib.Path:
    return owned_config_dir() / "daemon" / "control-api.json"


def _descriptor_identity() -> Optional[tuple]:
    """Stat identity of the control descriptor, ``None`` when absent.

    Recorded at spawn and compared when the exit is first observed. The engine
    rewrites ``control-api.json`` IN PLACE (``daemon-startup.ts``), so the tuple
    relies on ``st_mtime_ns``/``st_size`` changing, not on a new inode; a change
    means a descriptor was written DURING that spawn — the second half of the
    latch predicate.
    """
    try:
        stat = owned_descriptor_path().stat()
    except OSError:
        return None
    return (stat.st_ino, stat.st_mtime_ns, stat.st_size)


def owned_daemon_provisioned() -> bool:
    """Has the owner ever provisioned the owned daemon? (descriptor exists)

    This is the D30 cutover predicate: default daemon discovery prefers the
    owned home exactly from the moment this is True, and the moment is an
    owner action (first login/connect), never a silent boot-time switch.
    """
    try:
        return owned_descriptor_path().is_file()
    except OSError:
        return False


OWNERSHIP_MARKER = "ouroboros-owned.json"


def ownership_marker_path() -> pathlib.Path:
    return owned_config_dir() / OWNERSHIP_MARKER


def read_ownership_marker(*, strict: bool = False) -> Dict[str, Any]:
    """The durable claim that THIS data plane provisioned the home ({} = none)."""
    import json

    path = ownership_marker_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if strict and (not isinstance(raw, dict) or not raw):
            raise ValueError("ownership marker is not a nonempty object")
        return raw if isinstance(raw, dict) else {}
    except FileNotFoundError:
        if strict and path.is_symlink():
            raise ValueError("ownership marker is a dangling symlink")
        return {}
    except (OSError, ValueError):
        if strict:
            raise
        return {}


def verify_owned_home(*, require_marker: bool = False) -> str:
    """'' when the home is OURS to manage; a typed reason otherwise.

    Two independent facts, both required before any restart may CLAIM the
    home: the config dir sits under OUR data plane, and the ownership marker
    (when present) names the same data plane. A marker naming a different
    data plane is a FOREIGN home — disclosed, never adopted, never killed.
    A missing marker remains valid for provisioning. ``require_marker`` instead
    demands a positive Ouroboros marker before an attached process can be stopped.
    """
    from ouroboros.config import DATA_DIR

    config_dir = owned_config_dir()
    data_dir = pathlib.Path(DATA_DIR).resolve()
    try:
        config_dir.resolve().relative_to(data_dir)
    except ValueError:
        return f"config dir {config_dir} is outside the data plane {data_dir}"
    # Judge the read itself: a publisher may create a complete marker between
    # an absent read and a later exists() check. That is not malformed data.
    try:
        marker = read_ownership_marker(strict=True)
    except (OSError, ValueError):
        return "owned daemon marker is missing or invalid; stop ownership is unconfirmed"
    marked = str(marker.get("data_dir") or "")
    if (require_marker or marker) and (
        marker.get("owner") != "ouroboros" or not marked
    ):
        return "owned daemon marker is missing or invalid; stop ownership is unconfirmed"
    if marked and pathlib.Path(marked).resolve() != data_dir:
        return (f"ownership marker names a different data plane ({marked}); "
                "this home is not ours to manage")
    return ""


def _write_ownership_marker() -> None:
    """Atomically create missing evidence under the shared JSON publication lock."""
    from ouroboros.config import DATA_DIR
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.utils import update_json_locked, utc_now_iso

    problem = verify_owned_home()
    if problem:
        raise ClaudexorUnavailable("foreign_daemon_home", problem)
    path = ownership_marker_path()

    def create_missing(current: Dict[str, Any]) -> Any:
        # Revalidate inside the lock: another publisher may have claimed the
        # home after the first check. Existing ownership is never rewritten.
        problem = verify_owned_home()
        if problem:
            raise ClaudexorUnavailable("foreign_daemon_home", problem)
        if current:
            return None
        return {
            "owner": "ouroboros", "data_dir": str(pathlib.Path(DATA_DIR).resolve()),
            "provisioned_at": utc_now_iso(),
        }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        update_json_locked(path, create_missing, strict_existing_dict=True,
                           reject_existing_empty_dict=True)
    except ValueError as exc:
        raise ClaudexorUnavailable("foreign_daemon_home", verify_owned_home(require_marker=True) or str(exc)) from exc
    except (OSError, TimeoutError):
        log.warning("ownership marker write failed; attached stop remains unconfirmed", exc_info=True)


def resolve_claudexord() -> str:
    """Compatibility view of the old single-binary resolver."""
    from ouroboros.claudexor_runtime import resolve_external_claudexord

    return resolve_external_claudexord()


def attach_login_shell() -> str:
    """The explicit shell target for the host's copy-paste fallback."""
    from ouroboros.platform_layer import IS_WINDOWS

    return _SHELL_POWERSHELL if IS_WINDOWS else _SHELL_POSIX


def resolve_attach_login_argv(engine: Any) -> list[str]:
    """Resolve the packaged attach role on the exact serving engine.

    A live daemon may intentionally lag the reviewed next-spawn pin.  The
    handshake's version, build SHA and absolute entry therefore select the
    preserved tree, whose own additive probe role is the capability fact.
    Older probes remain readable but advertise no role, so they yield a typed
    unavailable result rather than a bare ``claudexor`` PATH command.
    """
    from ouroboros.claudexor_runtime import ClaudexorRuntimeError, get_runtime_manager
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    row = engine if isinstance(engine, dict) else {}
    try:
        command = get_runtime_manager().resolve_serving_role_command(
            engine_version=str(row.get("version") or ""),
            engine_build_sha=str(row.get("sha") or ""),
            engine_entry=str(row.get("entry") or ""),
            role=_SETUP_ATTACH_ROLE,
        )
    except ClaudexorRuntimeError as exc:
        if exc.code == "runtime_role_unavailable":
            code = "terminal_transport_unsupported"
            status_code = 409
            actions: tuple[str, ...] = ()
        elif exc.code in {
            "runtime_probe_failed",
            "runtime_probe_identity_mismatch",
            "runtime_node_version_mismatch",
        }:
            code = "terminal_transport_probe_failed"
            status_code = 503
            actions = ("retry_setup_login",)
        else:
            code = "terminal_transport_unavailable"
            status_code = 409
            actions = ()
        raise ClaudexorUnavailable(
            code,
            f"the packaged external-terminal recovery is unavailable: {exc}",
            status_code=status_code,
            required_actions=actions,
        ) from exc
    return [*command, "setup", "attach"]


def attach_login_command(job_id: str, *, argv: list[str], shell: str = "") -> str:
    """Render the already-probed packaged attach command for copy/paste."""
    target = str(shell or attach_login_shell())
    args = [str(value) for value in (*argv, str(job_id))]
    env = {
        "CLAUDEXOR_CONFIG_DIR": str(owned_config_dir()),
        # Never let an operator socket redirect the exact packaged entry away
        # from the owned config home.
        "CLAUDEXOR_DAEMON_SOCK": "",
    }
    if target == _SHELL_POSIX:
        assignments = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
        return " ".join([*assignments, *(shlex.quote(arg) for arg in args)])
    if target == _SHELL_POWERSHELL:
        quote = lambda value: "'" + value.replace("'", "''") + "'"
        assignments = [f"$env:{key}={quote(value)}" for key, value in env.items()]
        command = " ".join(["&", *(quote(arg) for arg in args)])
        return "; ".join([*assignments, command])
    raise ValueError(f"unsupported shell target: {target}")


class OwnedClaudexorDaemon:
    """Supervisor for the one Ouroboros-owned daemon (module singleton)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._last_error = ""
        self._engine_version = ""
        # Liveness bookkeeping above; the PROVEN serving version below. Only a
        # successful handshake writes it, so a failed probe cannot retract it.
        self._proven_engine_version = ""
        self._engine_build_sha = ""
        self._silent_listener = False  # last probe: a listener accepted, then sent no answer
        self._generation = 0
        self._stopping = False
        self._panic_requested = False
        self._panic_endpoint = None
        self._panic_targets: tuple = ()
        self._panic_capture_error = "no attested attachment captured"
        self._startup_attempt: Dict[str, Any] = {}
        # Spawn latch (#844): the typed record of the last start that ended
        # with ``ExitFact.failed_without_control``. While set, no caller
        # spawns; the periodic supervisor sweep, a live attach, or a new
        # manager (Restart/Panic) clears it. Not a counter, not a backoff.
        self._last_start_failure: Optional[Dict[str, Any]] = None
        # Rotation reconcile (B3): a non-blocking lock dedups CONCURRENT
        # ensures so they never double-POST settings; nothing else is gated.
        self._rotation_lock = threading.Lock()

    # -- state ------------------------------------------------------------

    def _classify_liveness(self, *, timeout_sec: Optional[float] = None) -> tuple:
        """(endpoint_or_None, state, detail) — the ONE liveness probe.

        The bearer token in OUR descriptor is the identity proof: each home's
        daemon mints its own random token, so an AUTHENTICATED handshake can
        only succeed against the daemon serving our home. An auth refusal
        (401/403) therefore means something ELSE answered on the descriptor's
        stale port — a foreign daemon, alive, not ours: disclosed, never
        killed, never adopted. A transport failure is distinguished privately
        from other stale evidence so explicit stop can use measured custody;
        protocol, discovery and malformed-response failures confer no such
        authority. Public status still projects all those failures as stale.
        """
        from ouroboros.gateways.claudexor import (
            ClaudexorGateway,
            ClaudexorUnavailable,
            discover_daemon_at,
        )
        import httpx

        self._silent_listener = False
        if not owned_daemon_provisioned():
            self._engine_version = self._engine_build_sha = ""
            return None, "not_provisioned", ""
        try:
            endpoint = discover_daemon_at(owned_config_dir())
        except ClaudexorUnavailable as exc:
            return None, "stale", f"{exc.code}: {exc}"
        try:
            with ClaudexorGateway(endpoint) as gateway:
                handshake = (
                    gateway.handshake()
                    if timeout_sec is None
                    else gateway.handshake(timeout_sec=timeout_sec)
                )
                self._engine_version = gateway.engine_version
                self._proven_engine_version = str(gateway.engine_version or "")
                engine = handshake.get("engine") if isinstance(handshake.get("engine"), dict) else {}
                self._engine_build_sha = str(engine.get("sha") or "")
                # Reachable-recovering is still "running" (the handshake proves
                # identity and liveness); admission is a separate, later
                # question answered by `ensure_owned_gateway`'s own handshakes.
            return endpoint, "running", ""
        except ClaudexorUnavailable as exc:
            self._engine_version = self._engine_build_sha = ""
            self._silent_listener = isinstance(exc.__cause__, (httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError))
            status = int(getattr(exc, "status_code", 0) or 0)
            if status in (401, 403):
                return None, "foreign_daemon", (
                    f"{exc.code}: a live daemon answered on the owned home's "
                    "descriptor port but REFUSED our home's token — a foreign "
                    "daemon recycled the port. It is not ours: disclosed, not "
                    "killed; a restart of OUR daemon rewrites the descriptor."
                )
            # Local request/configuration errors and decoded response failures
            # do not prove an unavailable network; received refusals still win.
            if status < 400 and exc.code == "daemon_unreachable" and isinstance(
                exc.__cause__, (httpx.NetworkError, httpx.ConnectTimeout,
                                httpx.ReadTimeout, httpx.WriteTimeout),
            ):
                return None, _TRANSPORT_UNREACHABLE, f"{exc.code}: {exc}"
            return None, "stale", f"{exc.code}: {exc}"

    @property
    def engine_version(self) -> str:
        """Serving version PROVEN by the last SUCCESSFUL handshake here, else ''.

        A request-shape floor reads this before it has a gateway, so a FAILED
        probe leaves it exactly as it was: a handshake timeout, a token refusal
        or a momentarily unreachable socket is silence about the engine, never
        evidence that a version it already served is gone. Blanking it there
        would flip a running caller's request shape between the candidate it
        was priced on and the send, which is a different bug wearing this
        field's clothes. Only another successful handshake replaces the value,
        and that is exactly how a deliberate stop or a planned restart on a new
        pin publishes its engine: the next ``ensure_owned_gateway`` re-proves
        the serving version, and until then the last proven one stands. Engine
        DOWNGRADE is not a supported direction, so a supported replacement can
        only prove the same version or a newer one. ``_engine_version`` stays
        the liveness projection — current probe, blanked on failure — which is
        what status and the pin comparison read.
        """
        return self._proven_engine_version

    def _alive_endpoint(self, *, timeout_sec: Optional[float] = None) -> Optional[Any]:
        """Endpoint of a LIVE daemon on our home, or None. Never spawns."""
        endpoint, state, detail = self._classify_liveness(timeout_sec=timeout_sec)
        if detail:
            self._last_error = detail
        return endpoint

    def status_dict(self) -> Dict[str, Any]:
        """UI status projection; observes capacity without starting the daemon."""
        from ouroboros.claudexor_exit_facts import engine_status_facts
        endpoint, state, detail = self._classify_liveness()
        if detail:
            self._last_error = detail
        ownership_problem = verify_owned_home() if state != "not_provisioned" else ""
        from ouroboros.claudexor_runtime import get_runtime_manager

        runtime_manager = get_runtime_manager()
        runtime = runtime_manager.status(
            running=state == "running",
            engine_version=self._engine_version,
            engine_build_sha=self._engine_build_sha,
        )
        return {
            "state": "stale" if state == _TRANSPORT_UNREACHABLE else state,
            "config_dir": str(owned_config_dir()),
            "engine_version": self._engine_version,
            "engine_build_sha": self._engine_build_sha,
            "self_started": bool(self._proc is not None and self._proc.poll() is None),
            "runtime": runtime,
            "last_error": self._last_error or None,
            # Typed foreign-home disclosure ('' = ours): a marker naming another
            # data plane means we display, and manage, NOTHING here.
            "ownership_problem": ownership_problem or None,
            **engine_status_facts(endpoint if state == "running" and not ownership_problem else None, self._engine_version, self._engine_build_sha),
        }

    # -- lifecycle ----------------------------------------------------------

    def _check_start_generation(self, generation: int) -> None:
        """A Stop retires in-flight callers, never a later explicit start."""
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        if self._panic_requested or self._stopping or generation != self._generation:
            raise ClaudexorUnavailable(
                "daemon_start_cancelled", "owned daemon startup was cancelled by Stop",
                status_code=503,
            )

    def _startup_pids(self) -> set[int]:
        """Join only this lifecycle's existing custody; election stays in the engine."""
        from ouroboros.config import DATA_DIR
        from ouroboros.gateways.claudexor import ClaudexorUnavailable
        from ouroboros.process_custody import live_daemon_root_pids

        try:
            pids = live_daemon_root_pids(
                pathlib.Path(DATA_DIR), retained_purposes={CUSTODY_PURPOSE},
                purposes={CUSTODY_PURPOSE}, strict=True,
            )
        except (OSError, ValueError, TypeError) as exc:
            raise ClaudexorUnavailable(
                "daemon_startup_unknown", "owned daemon custody is unreadable; no replacement was spawned",
                status_code=503,
            ) from exc
        proc = self._proc
        if proc is not None and proc.poll() is None:
            pids.add(proc.pid)
        return pids

    def _accept_endpoint(self, endpoint: Any, generation: int) -> Any:
        self._check_start_generation(generation)
        _write_ownership_marker()
        self._remember_stop_targets(endpoint)
        with self._lock:
            self._check_start_generation(generation)
        # An own child that exited while a live daemon answers — an election
        # loser, or a crash after publishing control — is reaped and ROWED
        # here (Popen.poll reaps it); it never latches against this attach,
        # because a live authenticated daemon (started by anyone) is exactly
        # the fact the latch waited for, so the attach clears it right after.
        self._settle_exited_child()
        with self._lock:
            self._last_error = ""
        self.clear_start_failure_latch(cleared_by="live_daemon_attached")
        return endpoint

    def _settle_exited_child(self) -> Optional[Dict[str, Any]]:
        """Reap this manager's own EXITED child once and harvest its exit fact (#844).

        Called at every spawn decision and at attach, not only when a caller's
        wait expires: with the real crash cadence (V8 dies after the 20 s
        startup window) the waiting caller got ``daemon_starting`` and the
        child died with nobody waiting — the next caller must still find the
        row, the classification and the latch instead of respawning silently.
        A live child, or a joined peer startup (no own handle), settles nothing.
        The exit fact and the latch are taken under the SAME lock as the pop
        (descriptor identity sampled right here), so a second caller in the
        read/classify/row window that follows already meets the latch, never a
        free spawn slot; the classification completes the record afterwards.
        """
        from ouroboros.utils import utc_now_iso

        with self._lock:
            proc, attempt = self._proc, dict(self._startup_attempt)
            returncode = proc.poll() if proc is not None else None
            if returncode is None:
                return None
            self._proc, self._startup_attempt = None, {}
            exit_status = ExitFact(returncode=returncode,
                                   descriptor_written=_descriptor_identity() != attempt.get("descriptor"))
            pending = build_start_failure_record(
                pin_version=str(attempt.get("version") or "unknown"),
                pin_build=str(attempt.get("build_sha") or "unknown"), exit_status=exit_status,
                log_path=str(owned_config_dir() / "daemon.log"), at=utc_now_iso(),
            )
            if exit_status.failed_without_control:
                self._last_start_failure = pending
        return self._record_start_failure(attempt, exit_status, pending)

    def clear_start_failure_latch(self, *, cleared_by: str) -> bool:
        """Release the spawn latch; True when one was set (a durable row names who released it).

        The sweep, owner Refresh and successful attach call this; Restart/Panic build a new manager.
        Another caller may spawn between release and retry; the retry joins that child.
        """
        with self._lock:
            record, self._last_start_failure = self._last_start_failure, None
        if record is None:
            return False
        from ouroboros.utils import utc_now_iso

        self._supervisor_row(latch_cleared_row(record, cleared_by=cleared_by, ts=utc_now_iso()))
        return True

    def _refuse_latched_spawn(self) -> None:
        """Settle an own child that died unwatched; while latched, refuse immediately, spawning nothing.

        Runs at both spawn-decision points (before preparation and right before spawn); ``_spawn``
        re-checks the latch under its own lock, so a peer's settle in between buys no extra spawn.
        """
        self._settle_exited_child()
        with self._lock:
            self._raise_if_latched()

    def _raise_if_latched(self) -> None:
        """Under ``self._lock``: the typed refusal while the spawn latch is set."""
        record = self._last_start_failure
        if record is None:
            return
        from ouroboros.gateways.claudexor import ClaudexorUnavailable
        detail = start_failure_detail(record)
        self._last_error = f"daemon_spawn_failed: {detail}"
        raise ClaudexorUnavailable(
            "daemon_spawn_failed",
            "owned daemon start is latched after a typed startup failure; no new spawn until "
            f"the periodic supervisor sweep retries, the owner refreshes or restarts; {detail}",
            status_code=503,
        )

    def ensure_running(self, *, startup_wait_sec: Optional[float] = None) -> Any:
        """Attach or join an installation-owned startup within this caller's wait.

        A live child or another manager's custodied daemon survives wait expiry.
        Runtime preparation and network waits never hold the management lock.
        Stop retires this manager's current callers before taking its process
        snapshot; a later explicit ensure may start again after Stop completes.
        A live authenticated old engine stays serving while a new pin is staged.
        """
        from ouroboros.claudexor_runtime import ClaudexorRuntimeError, get_runtime_manager
        from ouroboros.gateways.claudexor import SHORT_POLL_TIMEOUT_SEC, ClaudexorUnavailable

        if attach_home() is not None or run_cap_from_env() is not None:  # attached, or an isolated review
            raise ClaudexorUnavailable("attach_only_engine", "an isolated review never starts or prepares an engine")
        with self._lock:
            generation = self._generation
            self._check_start_generation(generation)
        problem = verify_owned_home()
        if problem:
            raise ClaudexorUnavailable("foreign_daemon_home", problem)
        endpoint, state, detail = self._classify_liveness()
        self._check_start_generation(generation)
        runtime_manager = get_runtime_manager()
        if endpoint is not None:
            pin = getattr(runtime_manager, "pin", None)
            if (pin is not None and self._engine_version == getattr(pin, "version", None)
                    and self._engine_build_sha == getattr(pin, "build_sha", None)):
                return self._accept_endpoint(endpoint, generation)
        elif self._startup_pids():
            return self._wait_for_start(generation, startup_wait_sec)
        else:
            self._refuse_latched_spawn()  # before preparation: the refusal is immediate
        if state == "foreign_daemon" and detail:
            log.warning("owned-daemon startup leaves the foreign responder untouched: %s", detail)
        try:
            command = runtime_manager.ensure()
        except ClaudexorRuntimeError as exc:
            self._check_start_generation(generation)
            if endpoint is not None:
                log.warning("managed runtime ensure failed while the owned daemon remains live: %s", exc)
                return self._accept_endpoint(endpoint, generation)
            raise ClaudexorUnavailable(exc.code, str(exc)) from exc
        self._check_start_generation(generation)
        if endpoint is not None:
            return self._accept_endpoint(endpoint, generation)
        # Preparation may take much longer than a concurrent startup. Re-read
        # both owners before spawning; engine writer election closes the remaining
        # first-launch race between independent Python processes.
        problem = verify_owned_home()
        if problem:
            raise ClaudexorUnavailable("foreign_daemon_home", problem)
        endpoint = self._alive_endpoint(timeout_sec=SHORT_POLL_TIMEOUT_SEC)
        self._check_start_generation(generation)
        if endpoint is not None:
            return self._accept_endpoint(endpoint, generation)
        if not self._startup_pids():
            self._refuse_latched_spawn()  # a peer's failure may have latched during preparation
            self._spawn(command, runtime_manager, generation)
        return self._wait_for_start(generation, startup_wait_sec)

    def _spawn(self, command: list[str], runtime_manager: Any, generation: int) -> None:
        """Publish our own startup through the existing marker and process custody."""
        config_dir = owned_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["CLAUDEXOR_CONFIG_DIR"] = str(config_dir)
        # Loopback-only ephemeral port is the engine default; explicitly
        # scrub any operator-level overrides that would cross homes.
        for crossing in ("CLAUDEXOR_DAEMON_SOCK", "CLAUDEXOR_CONTROL_PORT"):
            env.pop(crossing, None)
        command_bin = pathlib.Path(command[0]).parent
        if command_bin.is_dir():
            # Windows materializes os.environ with its native "Path" key; a
            # plain dict lookup of "PATH" misses it and would hand the child
            # a PATH holding only the Node bin dir (the engine then reports
            # git_missing). Prepend onto whichever key the host actually has.
            path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
            # An EMPTY PATH component means the CURRENT WORKING DIRECTORY on
            # POSIX. A host with no PATH (a scrubbed service manager, a bare
            # container unit) would otherwise leave a trailing empty entry
            # here and make CWD an executable search root for a long-lived
            # daemon that shells out to tools of its own. Drop every empty
            # component; order is otherwise preserved exactly.
            inherited = str(env.get(path_key, "") or "")
            composed = [str(command_bin), *inherited.split(os.pathsep)]
            env[path_key] = os.pathsep.join(part for part in composed if part)
        runtime = runtime_manager.status()
        log_path = config_dir / "daemon.log"
        from ouroboros.config import DATA_DIR
        from ouroboros.process_custody import spawn_supervised
        from ouroboros.platform_layer import subprocess_new_group_kwargs

        log.info("Spawning owned claudexord under %s from %s", config_dir, runtime.get("source") or "external")
        _write_ownership_marker()
        with self._lock:
            self._check_start_generation(generation)
            self._raise_if_latched()  # a peer may have settled our dead child since the pre-spawn check
            if self._proc is not None:  # alive: join it; exited and unsettled: the next settle owns its fact
                return
            with open(log_path, "ab") as sink:
                attempt = {
                    "log_start": sink.tell(), "log_identity": (os.fstat(sink.fileno()).st_dev,
                                                              os.fstat(sink.fileno()).st_ino),
                    "version": runtime.get("version") or "unknown",
                    "build_sha": runtime.get("build_sha") or "unknown",
                    "descriptor": _descriptor_identity(),
                }
                self._proc = spawn_supervised(
                    command,
                    drive_root=pathlib.Path(DATA_DIR),
                    purpose=CUSTODY_PURPOSE,
                    scope="daemon",
                    on_spawn=self._publish_child,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=sink,
                    stderr=sink,
                    **subprocess_new_group_kwargs(breakaway_from_job=True),
                )
                self._startup_attempt = {**attempt, "pid": self._proc.pid}

    def _publish_child(self, proc) -> None:
        """Expose the owned handle before custody I/O; retire a raced Panic spawn."""
        self._proc = proc
        if self._panic_requested:
            self.panic_stop(request_only=True)

    def _wait_for_start(self, generation: int, startup_wait_sec: Optional[float]) -> Any:
        from ouroboros.gateways.claudexor import SHORT_POLL_TIMEOUT_SEC, ClaudexorUnavailable

        wait = _SPAWN_WAIT_SEC if startup_wait_sec is None else max(0.0, float(startup_wait_sec))
        deadline = time.monotonic() + wait
        while True:
            self._check_start_generation(generation)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            endpoint = self._alive_endpoint(timeout_sec=min(remaining, SHORT_POLL_TIMEOUT_SEC))
            if endpoint is not None:
                return self._accept_endpoint(endpoint, generation)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(_SPAWN_POLL_SEC, remaining))
        self._check_start_generation(generation)
        pids = self._startup_pids()
        detail = self._startup_diagnostic(pids)
        if pids:  # a live listener that accepted and stayed silent is not answering, not starting
            code, said = (("daemon_not_answering", f"process is alive but did not answer in this {wait:.1f}s wait; retry joins the same daemon")
                          if self._silent_listener else ("daemon_starting", f"is still starting after this {wait:.1f}s wait; retry joins the same startup"))
            self._last_error = f"{code}: {detail}"
            raise ClaudexorUnavailable(code, f"owned daemon {said}; {detail}", status_code=503)
        with self._lock:
            self._check_start_generation(generation)
        failure = self._settle_exited_child()
        with self._lock:  # no own fact here: name the manager's latched state, marked as such
            latched = None if failure is not None else self._last_start_failure
        if failure is not None or latched is not None:
            detail = f"{detail}; {'' if failure else 'latched: '}{start_failure_label(failure or latched)}"
        self._last_error = f"daemon_spawn_failed: {detail}"
        raise ClaudexorUnavailable(
            "daemon_spawn_failed", f"no live owned startup or authenticated endpoint after {wait:.1f}s; {detail}",
            status_code=503,
        )

    def _record_start_failure(self, attempt: Dict[str, Any], exit_status: ExitFact,
                              pending: Dict[str, Any]) -> Dict[str, Any]:
        """Complete the harvested record: classify (diagnostic), row it, finalize the latch (#844).

        Only this manager's own exited child carries an exit fact; a joined
        peer startup that vanished leaves nothing to classify and never latches.
        The latch, if any, was taken with ``pending`` under the lock; it is
        replaced by the classified record only while it is still that object —
        a sweep that released it meanwhile is never re-latched.
        """
        interval, written = read_startup_log_interval(
            pathlib.Path(pending["log_path"]), start=int(attempt["log_start"]),
            identity=attempt["log_identity"])
        record = classified_start_failure_record(
            pending, classification=classify_startup_failure(written, exit_status), log_interval=interval)
        latched = exit_status.failed_without_control
        with self._lock:
            if latched and self._last_start_failure is pending:
                self._last_start_failure = record
        log.warning("Owned Claudexor start failed (latched=%s): %s", latched, start_failure_detail(record))
        self._supervisor_row(start_failure_row(record, latched=latched))
        return record

    def _startup_diagnostic(self, pids: set[int]) -> str:
        """Name current process evidence and its log interval, never an old tail as cause."""
        with self._lock:
            attempt = dict(self._startup_attempt)
            proc, latched = self._proc, self._last_start_failure
        details = [f"stage=waiting_for_control; live_pids={sorted(pids)}"]
        log_path = owned_config_dir() / "daemon.log"
        if attempt:
            details.append(f"spawn_pid={attempt['pid']}; selected_version={attempt['version']}; "
                           f"selected_build_sha={attempt['build_sha']}; "
                           f"poll={proc.poll() if proc is not None else 'unknown'}")
            interval, _ = read_startup_log_interval(  # identity-checked bounds only, no bytes
                log_path, start=int(attempt["log_start"]), identity=attempt["log_identity"], limit=0)
            details.append(describe_log_interval(interval or []))
        elif latched is None:
            details.append("joining another manager; its runtime build is not yet authenticated")
        else:
            details.append("no own startup handle left (settled by a concurrent caller, or a joined startup ended)")
        details.append(f"log={log_path} (shared diagnostic source, not an attributed failure cause)")
        return "; ".join(details)

    def _terminate_child(self) -> bool:
        """Stop our captured child outside the lock; clear only its confirmed handle.

        A child found exited HERE died of this stop's own signalling (the ledger pass
        runs first): cleared, not rowed. An earlier death was settled by ``stop_outcome``.
        """
        with self._lock:
            proc = self._proc
        if proc is None:
            return False
        if proc.poll() is not None:
            with self._lock:
                if self._proc is proc:
                    self._proc = None
                    self._startup_attempt = {}
            return False
        from ouroboros.platform_layer import kill_process_tree

        kill_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("owned daemon child %s stop is unconfirmed; handle retained", proc.pid)
            return False
        with self._lock:
            if self._proc is proc:
                self._proc = None
                self._startup_attempt = {}
        return True

    def reconcile_rotation(self, gateway: Any) -> None:
        """D28 as reconciliation (B3): default the MISSING limit-action
        policies to "rotate", never touching a persisted one.

        The predecessor was a spawn-only best-effort patch: one attempt at
        provisioning, a bare except, and no read-back — so a race with the
        daemon's startup "serving recovery only" window failed it forever,
        attach paths never patched at all, and a harness discovered later was
        never covered. This runs on EVERY owned ``ensure_owned_gateway`` instead
        (owner decision 5=A, literal: no read-path TTL — each ensure does the
        GET, computes the missing set and POSTs conditionally), against the
        gateway that ensure just handshook:

        * GET the effective settings snapshot, then POST only when a
          discovered harness carries NO ``profileLimitAction`` at all — an
          explicitly persisted ``fail``/``ask``/``rotate`` is the owner's (or
          the engine's) word and is never overwritten (owner decision 3=A);
        * skip engines whose version owns kind-aware "auto" defaults (A6+):
          their judgment is strictly better than a blanket "rotate";
        * the non-blocking lock exists purely to dedup CONCURRENT ensures —
          the overlapping caller is covered by the reconcile in flight;
        * ANY failure — the daemon's typed startup "recovery only" refusal
          included — simply retries on the next ensure; no special case;
        * a POST that actually changed policy leaves a durable receipt under
          ``state/`` naming the daemon and the patched harnesses;
        * never patches a home ``verify_owned_home`` rejects (never-adopt).

        Best-effort by contract: raises nothing, so a reconcile hiccup can
        never eat the delegation or login that ensured the daemon.
        """
        if not self._rotation_lock.acquire(blocking=False):
            return  # a concurrent ensure is reconciling right now; it covers us
        try:
            try:
                from ouroboros.gateways.claudexor import engine_at_least

                if engine_at_least(str(getattr(gateway, "engine_version", "") or ""),
                                   _ROTATION_AUTO_SEMANTICS_MIN_VERSION):
                    return
                ownership_problem = verify_owned_home()
                if ownership_problem:
                    log.warning("rotation reconcile refused (never-adopt): %s",
                                ownership_problem)
                    return
                snapshot = gateway.get_settings()
                raw_configured = snapshot.get("harnesses") if isinstance(snapshot, dict) else None
                if not isinstance(raw_configured, dict):
                    # Shape drift (no harnesses table, or not a dict): unknown state
                    # must never read as "nothing persisted" — a blanket POST here
                    # would overwrite judgments this side simply failed to read.
                    log.warning(
                        "rotation reconcile skipped: settings snapshot carries no "
                        "harnesses dict (engine %s)",
                        str(getattr(gateway, "engine_version", "") or "unknown"))
                    return
                configured = raw_configured
                missing = []
                for row in gateway.agent_capabilities().get("harnesses") or []:
                    hid = str(row.get("id") or "") if isinstance(row, dict) else ""
                    if not hid:
                        continue
                    stored = configured.get(hid)
                    action = stored.get("profileLimitAction") if isinstance(stored, dict) else None
                    if not str(action or ""):
                        missing.append(hid)
                if missing:
                    gateway.patch_settings({
                        "harnesses": {hid: {"profileLimitAction": "rotate"} for hid in missing},
                    })
                    self._record_rotation_receipt(
                        str(getattr(gateway, "engine_version", "") or ""), missing)
            except Exception:
                log.warning("rotation reconcile failed; the next ensure retries",
                            exc_info=True)
        finally:
            self._rotation_lock.release()

    def _record_rotation_receipt(self, engine_version: str, patched: list) -> None:
        """Durable half of the reconcile: a settings POST that changed the
        daemon's policy leaves a record naming the daemon identity, the
        patched harnesses and the moment — not just a log line: a typed JSON
        receipt written atomically under ``state/`` beside the policy it
        describes, so an audit reads the fact instead of grepping logs."""
        import json

        from ouroboros.config import DATA_DIR
        from ouroboros.utils import utc_now_iso, write_text_atomic

        path = pathlib.Path(DATA_DIR) / "state" / _ROTATION_RECEIPT_NAME
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomic(path, json.dumps({
                "ts": utc_now_iso(),
                "daemon_config_dir": str(owned_config_dir()),
                "engine_version": str(engine_version or ""),
                "patched_harnesses": sorted(str(h) for h in patched),
                "limit_action": "rotate",
                "reason": "limit_action_absent_defaulted_to_rotate",
            }, ensure_ascii=False, indent=1))
        except OSError as exc:
            # Residual: the POST itself landed — the next ensure's GET sees the values
            # present and correctly skips — so the only gap is this missing receipt.
            log.warning("rotation provisioning receipt write failed at %s: %s",
                        path, exc, exc_info=True)

    def stop(self) -> bool:
        """``stop_outcome`` as Panic reads it: True only for a confirmed stop."""
        return self.stop_outcome() == "stopped"

    def _request_operator_stop(self) -> bool:
        """Use the installed same-home CLI, never provisioning or waking a daemon."""
        from ouroboros.claudexor_runtime import get_runtime_manager
        from ouroboros.platform_layer import subprocess_hidden_kwargs

        try:
            command = get_runtime_manager().resolve_cli_command(require_npm=False)
            if not command:
                return False
            env = dict(os.environ)
            env["CLAUDEXOR_CONFIG_DIR"] = str(owned_config_dir())
            for crossing in ("CLAUDEXOR_DAEMON_SOCK", "CLAUDEXOR_CONTROL_PORT"):
                env.pop(crossing, None)
            result = subprocess.run(
                [*command, "daemon", "stop", "--json"], env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                timeout=_OPERATOR_STOP_TIMEOUT_SEC, **subprocess_hidden_kwargs(),
            )
            receipt = json.loads(result.stdout) if result.returncode == 0 else None
            return bool(isinstance(receipt, dict) and receipt.get("ok") is True
                        and receipt.get("stopped") is True and receipt.get("outcome") in {"exited", "killed"})
        except Exception as exc:
            # CLI stdout/stderr may contain private diagnostics. The existing
            # stop owner independently decides whether measured fallback is valid.
            log.warning("Owned daemon operator stop did not confirm completion (%s)", type(exc).__name__)
            return False

    def _remember_stop_targets(self, endpoint) -> None:
        """Prepare attachment identity during normal admission, never during Panic.

        The authenticated endpoint and owned-home marker authorize only measured
        custody rows. Pin OS references between two full fingerprint checks;
        later requests use those references, never a descriptor PID or port.
        """
        if (endpoint == self._panic_endpoint and self._panic_targets) or self._panic_requested:
            return
        from ouroboros.config import DATA_DIR
        from ouroboros.platform_layer import capture_process_stop_target
        from ouroboros.process_custody import _fingerprint_matches, process_stop_snapshot

        targets = []
        try:
            if problem := verify_owned_home(require_marker=True):
                raise OSError(problem)
            for entry in process_stop_snapshot(pathlib.Path(DATA_DIR), {CUSTODY_PURPOSE}):
                if not _fingerprint_matches(entry, require_measured=True):
                    continue
                target = capture_process_stop_target(int(entry["pid"]))
                if _fingerprint_matches(entry, require_measured=True):
                    target["custody"] = entry
                    targets.append(target)
            self._panic_targets = tuple(targets)
            self._panic_capture_error = "" if targets else "no measured attached process identity"
            self._panic_endpoint = endpoint
            if self._panic_requested:  # publication raced the immediate request
                self.panic_stop(request_only=True)
        except Exception as exc:
            self._panic_capture_error = f"attachment capture failed: {type(exc).__name__}"

    def panic_stop(self, *, request_only: bool = False) -> list:
        """Signal captured children/attachments without manager, CLI or disk waits."""
        from ouroboros.platform_layer import request_process_tree_kill

        self._panic_requested = True
        proc = self._proc
        targets = ([proc] if proc is not None else []) + list(self._panic_targets)
        requests = [request_process_tree_kill(target) for target in targets]
        if not targets and self._panic_capture_error:
            requests.append({"requested": False, "error": self._panic_capture_error})
        if not request_only:
            self.stop_outcome()
        return requests

    def stop_outcome(self) -> DaemonStopOutcome:
        """Stop verified own roots; report every unconfirmed remainder.

        ``"stopped"`` requires a confirmed stop with no remaining custody;
        ``"nothing_to_stop"`` is the quiet empty case; ``"unconfirmed"`` has
        already been disclosed by ``_report_stop_unconfirmed`` (critical log
        plus the supervisor row) with custody retained, so a caller that must
        proceed anyway — the owner's manual Restart — needs no private state
        to tell the two non-stops apart. Lock acquisition is bounded separately
        from HTTP connect/read phases and each root's exit wait; there is no
        promised absolute wall-clock deadline for the whole teardown. A
        self-started Popen handle proves direct ownership. An authenticated
        same-home endpoint permits ordinary CLI shutdown, including a legacy
        attached daemon whose recorded birth was unavailable. Forced signalling
        still requires the measured ledger identity plus an authenticated
        endpoint, typed transport failure, or absent descriptor for a marked
        startup. Network and exit waits run outside the lock. A token refusal
        or invalid discovery permits no attached fallback.
        """
        from ouroboros.config import DATA_DIR
        from ouroboros.gateways.claudexor import SHORT_POLL_TIMEOUT_SEC
        from ouroboros.process_custody import (
            pending_process_stops, pid_is_zombie, process_stop_snapshot, stop_ledgered_processes,
        )
        from ouroboros.platform_layer import collect_descendant_pids, pid_is_alive

        if not self._lock.acquire(timeout=SHORT_POLL_TIMEOUT_SEC):
            self._report_stop_unconfirmed("daemon manager lock unavailable; custody unchanged")
            return "unconfirmed"
        if self._stopping:
            self._lock.release()
            self._report_stop_unconfirmed("owned daemon stop is already in progress")
            return "unconfirmed"
        self._generation += 1
        self._stopping = True
        self._lock.release()
        try:
            root = pathlib.Path(DATA_DIR)
            purposes = {CUSTODY_PURPOSE}
            stopped, unconfirmed = [], []
            self._last_error = ""
            # An unwatched death BEFORE this stop is settled first (rowed, latched when
            # it applies) so Restart/Panic never forget it; a child killed below is not one.
            self._settle_exited_child()
            ownership_problem = verify_owned_home(require_marker=True)
            endpoint, state = None, ""
            if not ownership_problem:
                endpoint, state, detail = self._classify_liveness(timeout_sec=SHORT_POLL_TIMEOUT_SEC)
                if detail:
                    self._last_error = detail
            pre_listener = state == "not_provisioned" and not os.path.lexists(owned_descriptor_path())
            operator_stopped = False
            if endpoint is not None or state == _TRANSPORT_UNREACHABLE or pre_listener:
                expected_entries, observed = [], set()
                try:
                    expected_entries = ([target["custody"] for target in self._panic_targets]
                                        if self._panic_requested else process_stop_snapshot(root, purposes))
                    for entry in expected_entries:
                        pid = int(entry["pid"])
                        observed.add(pid)
                        observed.update(collect_descendant_pids(pid))
                    if self._proc is not None:
                        observed.add(self._proc.pid)
                except Exception:
                    unconfirmed.append("stop target custody could not be observed")
                if endpoint is not None and (not self._panic_requested or endpoint == self._panic_endpoint):
                    operator_stopped = self._request_operator_stop()
                if operator_stopped:
                    # Lease release confirms clean service shutdown; a Node tail
                    # or captured harness child may still be physically alive.
                    deadline = time.monotonic() + _STOP_EXIT_WAIT_SEC
                    remaining = observed
                    while remaining:
                        remaining = {pid for pid in remaining if pid_is_alive(pid) and not pid_is_zombie(pid)}
                        if not remaining or time.monotonic() >= deadline:
                            break
                        time.sleep(0.05)
                else:
                    stopped = stop_ledgered_processes(
                        root, purposes, unconfirmed=unconfirmed, expected_entries=expected_entries,
                    )
            child_stopped = self._terminate_child()
            if operator_stopped:
                remaining = {pid for pid in observed if pid_is_alive(pid) and not pid_is_zombie(pid)}
                if remaining:
                    unconfirmed.append(f"operator stop left process exit unconfirmed: {sorted(remaining)}")
                live_endpoint, _, _ = self._classify_liveness(timeout_sec=SHORT_POLL_TIMEOUT_SEC)
                if live_endpoint is not None:
                    # The CLI's receipt concerns its pinned owner, not a new
                    # daemon another client may have started during shutdown.
                    unconfirmed.append("an authenticated owned endpoint remains after operator stop")
            if ownership_problem and owned_daemon_provisioned() and not child_stopped:
                unconfirmed.append("descriptor ownership is unconfirmed")
            unconfirmed.extend(pending_process_stops(root, purposes))
            if self._proc is not None:
                unconfirmed.append("self-started child exit unconfirmed")
            if endpoint is not None and not stopped and not child_stopped and not operator_stopped:
                unconfirmed.append("authenticated endpoint has no confirmed stopped root")
            if unconfirmed:
                reason = ownership_problem or self._last_error
                if reason:
                    unconfirmed.insert(0, reason)
                self._report_stop_unconfirmed("; ".join(dict.fromkeys(unconfirmed)))
                return "unconfirmed"
            self._last_error = ""
            return "stopped" if (operator_stopped or child_stopped or stopped) else "nothing_to_stop"
        finally:
            with self._lock:
                self._stopping = False

    def _report_stop_unconfirmed(self, detail: str) -> None:
        """The lifecycle owner discloses a failed stop in the existing supervisor log."""
        from ouroboros.utils import utc_now_iso

        self._last_error = detail
        log.critical("Owned Claudexor stop unconfirmed: %s; custody retained", detail)
        self._supervisor_row({"ts": utc_now_iso(), "type": "process_stop_unconfirmed", "reason": detail})

    def _supervisor_row(self, row: Dict[str, Any]) -> None:
        """The manager's one durable disclosure seam: a purpose-stamped supervisor row."""
        from ouroboros.config import DATA_DIR
        from ouroboros.utils import append_jsonl

        append_jsonl(pathlib.Path(DATA_DIR) / "logs" / "supervisor.jsonl",
                     {**row, "purpose": CUSTODY_PURPOSE})


_MANAGER: Optional[OwnedClaudexorDaemon] = None
_MANAGER_LOCK = threading.Lock()


def get_owned_daemon(*, create: bool = True) -> Optional[OwnedClaudexorDaemon]:
    global _MANAGER
    if _MANAGER is not None or not create:
        return _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = OwnedClaudexorDaemon()
        return _MANAGER


def warm_owned_daemon() -> bool:
    """One background ``ensure_owned_gateway`` at server start, provisioned homes only.

    The first delegation after a Restart then finds the daemon serving instead
    of paying its spawn. Nothing new is managed here: a Stop that lands while
    the ensure is still preparing retires it through the existing start
    generation, a failure is logged and left to the first real caller, and an
    unprovisioned home (no descriptor yet) is never touched (``False``).
    """
    if not owned_daemon_provisioned():
        return False

    def warm() -> None:
        try:
            # Warmup is the host's speculative join, so give an already
            # custodied daemon enough room to finish journal admission after
            # its control socket appears.  A normal caller keeps the shorter
            # admission window; expiry still leaves custody intact and the
            # first real caller simply joins the same startup.
            warmup_admission_wait = max(_ADMISSION_WAIT_SEC, _SPAWN_WAIT_SEC * 2.0)
            ensure_owned_gateway(admission_wait_sec=warmup_admission_wait).close()
        except Exception:
            log.info("Owned daemon warmup did not reach readiness; the first caller starts or joins",
                     exc_info=True)

    threading.Thread(target=warm, name="owned-daemon-warmup", daemon=True).start()
    return True


def owned_engine_version() -> str:
    """The serving version proven by the last SUCCESSFUL handshake, no new I/O.

    A feature floor that must be decided BEFORE a gateway exists (the model
    lane freezes its request bytes before it connects) reads the version the
    previous ensure/handshake already proved, and keeps reading the same answer
    while probes fail — several callers probe this singleton concurrently, so a
    transient failure must not change one caller's request shape mid-turn.
    Empty means no handshake has succeeded in this process yet, and every floor
    then fails CLOSED — a process's first model call sends the legacy shape
    rather than probing the control plane again or trusting the next-spawn pin,
    which a live daemon may intentionally lag.
    """
    return get_owned_daemon().engine_version


def read_owned_gateway(*, timeout_sec: Optional[float] = None) -> Any:
    """Connect to the owned engine for metadata, without starting or repairing it.

    Discovery is explicitly owned-only, including on unprovisioned installs.
    Callers own close(); discovery/handshake failures retain their typed refusal.
    An attach-only selection reads the selected home and records no stop target.
    """
    from ouroboros.gateways.claudexor import ClaudexorGateway, discover_daemon_at

    home = attach_home()
    endpoint = attached_endpoint(home) if home is not None else discover_daemon_at(owned_config_dir())
    gateway = ClaudexorGateway(endpoint)
    try:
        gateway.handshake(**({"timeout_sec": timeout_sec} if timeout_sec is not None else {}))
        if home is None:
            get_owned_daemon()._remember_stop_targets(endpoint)
    except Exception:
        gateway.close()
        raise
    return gateway


def ensure_owned_gateway(*, admission_wait_sec: Optional[float] = None,
                         startup_wait_sec: Optional[float] = None) -> Any:
    """Return an authenticated gateway to the lazily ensured owned daemon.

    This is the explicit start/probe seam — the ONE funnel every consumer
    (delegation, review sessions, account surfaces, login) passes through,
    which is why the rotation reconcile rides it: spawn AND owned-attach paths,
    on every ensure, best-effort (see ``reconcile_rotation``); never attach-only.
    The gateway transport itself stays pure I/O; callers own ``close()`` (or
    use it as a context manager). ``stop()`` owns the separate marker, transport
    and process-identity checks for stopping an attached daemon.

    ADMISSION is waited for here — outside the daemon manager's lock, the same
    way for a fresh spawn and an attach. A daemon whose handshake explicitly
    says ``servingMode=recovery_only`` answers every product route 503
    (``daemon_recovery_only``, retryable), so the handshake is re-polled about
    every 150 ms under a wall-clock deadline of ``admission_wait_sec`` seconds
    (default ``_ADMISSION_WAIT_SEC``, resolved at call time so tests can shrink
    it), each poll's read phase bounded by what is left of the window. Expiry
    raises the SAME typed refusal the 503 produces — the dispatch table already
    classifies it (auto → native with a loud marker, pin → blocked) — and the
    recovering daemon is left alive (D28: bounded wait, then typed refusal;
    never a silent indefinite wait, never a kill). ``admission_wait_sec=0`` is
    the zero-wait variant for callers that must not stall on ADMISSION: a
    recovering daemon is an immediate typed refusal there, and the initial
    handshake below is read-bounded by the same small window. The wait bounds
    admission only. ``startup_wait_sec`` separately narrows the caller's control
    readiness wait (default ``config.CLAUDEXOR_STARTUP_WAIT_SEC``); its expiry keeps
    live startup custody: ``daemon_starting``, or ``daemon_not_answering`` for a silent listener.
    Ordinary attach probes retain their transport ceiling; runtime preparation
    is separate from both waits. Neither zero-wait parameter promises zero
    total latency or permission to stop the process.
    An expired/failed admission also skips the reconcile: the recovering
    daemon 503s settings reads anyway, and the next ensure retries it.
    """
    from ouroboros.gateways.claudexor import (
        SHORT_POLL_TIMEOUT_SEC, ClaudexorGateway, ClaudexorUnavailable,
    )

    wait = _ADMISSION_WAIT_SEC if admission_wait_sec is None else max(
        0.0, float(admission_wait_sec))
    daemon = get_owned_daemon()
    home = attach_home()
    if home is not None:  # attach-only: no start, no runtime pin, no ownership claim
        endpoint = attached_endpoint(home)
    else:
        endpoint = (daemon.ensure_running() if startup_wait_sec is None
                    else daemon.ensure_running(startup_wait_sec=startup_wait_sec))
    gateway = ClaudexorGateway(endpoint)
    try:
        # Read-bounded: a daemon that accepts the socket but withholds the
        # handshake must not hold a zero/small-wait caller for the transport's
        # 60s default read — the sweep's whole posture is "skip, next tick".
        body = gateway.handshake(timeout_sec=max(wait, SHORT_POLL_TIMEOUT_SEC))
        deadline = time.monotonic() + wait

        def _expired() -> ClaudexorUnavailable:
            return ClaudexorUnavailable(
                "daemon_recovery_only",
                "the owned daemon is reachable but still admitting only "
                f"recovery work after {wait:.1f}s; its product routes "
                "answer 503 (retryable) until journal recovery completes",
            )

        while _handshake_serving_mode(body) == "recovery_only":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _expired()
            time.sleep(min(_ADMISSION_POLL_SEC, remaining))
            # The WHOLE declared window is usable (proton0 review): the last
            # read gets the thin residue, floored so a loopback handshake can
            # still complete — a daemon that admitted normal work at the
            # window's edge is observed, not discarded. A transport failure
            # inside that final residue counts as expiry (typed, never a
            # transport mislabel); a mid-window one propagates unchanged.
            remaining = deadline - time.monotonic()
            try:
                body = gateway.handshake(
                    timeout_sec=max(remaining, _ADMISSION_POLL_SEC / 3.0))
            except ClaudexorUnavailable:
                if deadline - time.monotonic() <= 0:
                    raise _expired() from None
                raise
    except Exception:
        gateway.close()
        raise
    if home is not None:  # the proven version feeds request-shape floors; host engine settings stay untouched
        daemon._proven_engine_version = str(gateway.engine_version or "")
        return gateway
    daemon.reconcile_rotation(gateway)
    return gateway


__all__ = [
    "OwnedClaudexorDaemon",
    "ownership_marker_path",
    "read_ownership_marker",
    "verify_owned_home",
    "attach_login_command",
    "attach_login_shell",
    "resolve_attach_login_argv",
    "ensure_owned_gateway",
    "owned_engine_version",
    "read_owned_gateway",
    "get_owned_daemon",
    "warm_owned_daemon",
    "DaemonStopOutcome",
    "owned_config_dir",
    "owned_daemon_provisioned",
    "owned_descriptor_path",
    "resolve_claudexord",
]


# --- Connect's vendor-CLI install (domain operation of the owned data plane;
# the accounts gateway only invokes it and translates the typed result) ---
_HARNESS_INSTALL_STDOUT_LIMIT = 64 * 1024
_HARNESS_INSTALL_CORE_FIELDS = frozenset({
    "ok", "dryRun", "exitCode", "target", "harness", "command",
    "installLocation", "installedBinary", "installedVersion", "pinnedVersion",
    "verification",
})
_HARNESS_INSTALL_PROVENANCE_FIELDS = frozenset({"installerSha256", "installerByteLength"})
_LOCAL_INSTALL_VERIFICATIONS = frozenset({
    "release_verified", "deterministic_only", "unattended_unpinned",
})
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def is_immediate_missing_cli_job(job: Dict[str, Any], harness: str, gateway: Any) -> bool:
    """Match only the pinned engine's synchronous missing-vendor-CLI result."""
    if not isinstance(job, dict):
        return False
    outcome = job.get("outcome")
    if (
        not isinstance(job.get("jobId"), str)
        or not job["jobId"]
        or job.get("harness") != harness
        or job.get("action") != "login"
        or job.get("state") != "not_supported"
        or job.get("phase") != "completed"
        or not isinstance(outcome, dict)
        or outcome.get("reason") != "not_supported"
        or "command" not in job
        or job.get("command") is not None
        or job.get("authorization") is not None
        or job.get("nativeCommand") is not None
    ):
        return False

    from ouroboros.claudexor_runtime import get_runtime_manager

    pin = get_runtime_manager().pin
    return bool(
        pin is not None
        and pin.cli_entrypoint is not None
        and gateway.engine_version == pin.version
        and gateway.engine_build_sha == pin.build_sha
    )


def _drain_installer_stdout(pipe: Any, output: bytearray, state: Dict[str, bool]) -> None:
    try:
        while True:
            chunk = pipe.read(8192)
            if not chunk:
                break
            remaining = _HARNESS_INSTALL_STDOUT_LIMIT - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                state["overflow"] = True
    except Exception:
        state["read_error"] = True
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _valid_install_success(payload: Any, harness: str) -> bool:
    if not isinstance(payload, dict):
        return False
    fields = frozenset(payload)
    with_provenance = _HARNESS_INSTALL_CORE_FIELDS | _HARNESS_INSTALL_PROVENANCE_FIELDS
    if fields not in (_HARNESS_INSTALL_CORE_FIELDS, with_provenance):
        return False
    verification = payload.get("verification")
    if (
        payload.get("ok") is not True
        or payload.get("dryRun") is not False
        or type(payload.get("exitCode")) is not int
        or payload["exitCode"] != 0
        or payload.get("target") != "local"
        or payload.get("harness") != harness
        or not isinstance(payload.get("command"), str)
        or not payload["command"]
        or not isinstance(payload.get("installLocation"), str)
        or not payload["installLocation"]
        or not isinstance(payload.get("installedBinary"), str)
        or not os.path.isabs(payload["installedBinary"])
        or not isinstance(payload.get("installedVersion"), str)
        or not payload["installedVersion"].strip()
        or len(payload["installedVersion"]) > 256
        or not isinstance(verification, str)
        or verification not in _LOCAL_INSTALL_VERIFICATIONS
        or (
            verification == "unattended_unpinned"
            and payload.get("pinnedVersion") is not None
        )
        or (
            verification != "unattended_unpinned"
            and not (
                isinstance(payload.get("pinnedVersion"), str)
                and bool(payload["pinnedVersion"])
            )
        )
    ):
        return False
    if fields == with_provenance:
        return bool(
            verification == "unattended_unpinned"
            and isinstance(payload.get("installerSha256"), str)
            and _SHA256_HEX.fullmatch(payload["installerSha256"])
            and type(payload.get("installerByteLength")) is int
            and payload["installerByteLength"] > 0
        )
    return True


def install_missing_harness_cli(harness: str) -> None:
    from ouroboros.claudexor_runtime import ClaudexorRuntimeError, get_runtime_manager
    from ouroboros.config import get_claudexor_harness_install_timeout_sec
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.platform_layer import merge_hidden_kwargs, subprocess_new_group_kwargs
    # The same custody set /panic reaps (isolated_deps._run is the template):
    # an in-flight vendor installer must not survive an emergency stop.
    from ouroboros.tools.shell import _active_subprocesses, _kill_process_group, _subprocess_lock

    try:
        command = get_runtime_manager().ensure_cli_command()
    except ClaudexorRuntimeError as exc:
        raise ClaudexorUnavailable(exc.code, str(exc)) from exc
    if len(command) != 2:
        raise ClaudexorUnavailable(
            "runtime_cli_unavailable", "the exact managed Claudexor CLI is not selectable"
        )
    argv = [
        *command, "harness", "install", harness,
        "--target", "local", "--yes", "--json",
    ]
    # The SAME data-plane binding the owned daemon starts with: the config-dir
    # override is the complete relocatable root (D30), and the cross-home
    # overrides the daemon scrubs must not reach the installer either —
    # otherwise the CLI acts on the operator's personal Claudexor home.
    env = dict(os.environ)
    env["CLAUDEXOR_CONFIG_DIR"] = str(owned_config_dir())
    for crossing in ("CLAUDEXOR_DAEMON_SOCK", "CLAUDEXOR_CONTROL_PORT"):
        env.pop(crossing, None)
    kwargs = merge_hidden_kwargs(subprocess_new_group_kwargs())
    timeout_sec = get_claudexor_harness_install_timeout_sec()
    try:
        # Registration is atomic WITH the spawn: /panic snapshots the tracked
        # set under this same lock, so it can never observe the child alive
        # but untracked (the round-2 reviewer's interleaving).
        with _subprocess_lock:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                **kwargs,
            )
            _active_subprocesses.add(proc)
    except OSError as exc:
        raise ClaudexorUnavailable(
            "harness_install_spawn_failed",
            f"managed Claudexor installer could not start: {type(exc).__name__}",
        ) from exc

    output = bytearray()
    state: Dict[str, bool] = {}
    reader = threading.Thread(
        target=_drain_installer_stdout,
        args=(proc.stdout, output, state),
        name="claudexor-installer-stdout",
        daemon=True,
    )
    reader.start()
    try:
        try:
            exit_code = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired as exc:
            _kill_process_group(proc)
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
            raise ClaudexorUnavailable(
                "harness_install_timeout",
                f"managed Claudexor installer exceeded {timeout_sec:d}s",
            ) from exc
    finally:
        with _subprocess_lock:
            _active_subprocesses.discard(proc)
        # Bounded CLEANUP of an already-finished/killed child's pipe, not a
        # behavioral wait: the drain thread ends when the pipe does.
        reader.join(timeout=10)
        if reader.is_alive():
            try:
                proc.stdout.close()
            except Exception:
                pass
            reader.join(timeout=1)
        if reader.is_alive():
            state["read_error"] = True

    if exit_code != 0:
        raise ClaudexorUnavailable(
            "harness_install_failed", f"managed Claudexor installer exited with code {exit_code}"
        )
    if state.get("overflow") or state.get("read_error"):
        raise ClaudexorUnavailable(
            "harness_install_invalid_response", "managed Claudexor installer output was invalid"
        )
    try:
        payload = json.loads(bytes(output))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ClaudexorUnavailable(
            "harness_install_invalid_response", "managed Claudexor installer returned invalid JSON"
        ) from exc
    if not _valid_install_success(payload, harness):
        raise ClaudexorUnavailable(
            "harness_install_invalid_response", "managed Claudexor installer receipt was invalid"
        )
