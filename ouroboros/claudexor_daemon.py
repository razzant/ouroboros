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
* ATTACH-IF-ALIVE: a live daemon already serving our config dir (a previous
  generation's, custody-pending) is attached to, not duplicated — the engine
  refuses a second daemon on the same socket anyway;
* STOP-ONLY-WHAT-IS-PROVABLY-OURS: ``stop`` (Panic) terminates the child THIS
  manager spawned and ledger roots confirmed by our marker and measured
  custody fingerprint, with an authenticated endpoint or a typed transport
  failure — a prior generation's or a worker's spawn included. Token refusal,
  invalid discovery and incompatible/malformed replies never permit that
  fallback. Never stop a live responder known only by name or by
  the descriptor port (a foreign daemon on a recycled port stays disclosed, not
  killed). A newer runtime pin is staged for the next natural start, never
  hot-swapped. Planned replacement remains deferred pending the engine's
  atomic treatment of setup jobs with unconfirmed termination.

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
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_OWNED_DIR_NAME = "claudexor"
CUSTODY_PURPOSE = "claudexor_daemon"
_SPAWN_WAIT_SEC = 20.0
_SPAWN_POLL_SEC = 0.25
# Admission, distinct from reachability: a 3.4+ daemon serves the authenticated
# handshake BEFORE its admission gate (the body says `servingMode`), while every
# product route answers 503 `daemon_recovery_only` (retryable) until journal
# recovery completes. Recovery can persist indefinitely (blocked journal
# partitions), so the wait is bounded and ends in the typed refusal the 503
# already produces (D28) — never a silent indefinite wait, never a kill of a
# recovering daemon. The 150 ms cadence is the engine CLI's own.
_ADMISSION_WAIT_SEC = 5.0
_ADMISSION_POLL_SEC = 0.15

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


def owned_descriptor_path() -> pathlib.Path:
    return owned_config_dir() / "daemon" / "control-api.json"


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


def read_ownership_marker() -> Dict[str, Any]:
    """The durable claim that THIS data plane provisioned the home ({} = none)."""
    import json

    try:
        raw = json.loads(ownership_marker_path().read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
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
    marker = read_ownership_marker()
    marked = str(marker.get("data_dir") or "")
    path = ownership_marker_path()
    if (require_marker or path.exists() or path.is_symlink()) and (
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
        self._engine_build_sha = ""
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

        if not owned_daemon_provisioned():
            self._engine_version = ""
            self._engine_build_sha = ""
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
                engine = handshake.get("engine") if isinstance(handshake.get("engine"), dict) else {}
                self._engine_build_sha = str(engine.get("sha") or "")
                # Reachable-recovering is still "running" (the handshake proves
                # identity and liveness); admission is a separate, later
                # question answered by `ensure_owned_gateway`'s own handshakes.
            return endpoint, "running", ""
        except ClaudexorUnavailable as exc:
            self._engine_version = ""
            self._engine_build_sha = ""
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

    def _alive_endpoint(self, *, timeout_sec: Optional[float] = None) -> Optional[Any]:
        """Endpoint of a LIVE daemon on our home, or None. Never spawns."""
        endpoint, state, detail = self._classify_liveness(timeout_sec=timeout_sec)
        if detail:
            self._last_error = detail
        return endpoint

    def status_dict(self) -> Dict[str, Any]:
        """UI status projection. Read-only: never spawns."""
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
        }

    # -- lifecycle ----------------------------------------------------------

    def ensure_running(self) -> Any:
        """Attach to a live owned daemon, or (re)start one; returns its endpoint.

        The stale lifecycle, minimal and honest: verify liveness by an
        AUTHENTICATED handshake; a dead daemon whose home carries OUR ownership
        marker is restarted under the same supervision and reconciled (fresh
        discovery + handshake against the rewritten descriptor); a live daemon
        that refuses our token is FOREIGN — disclosed in the typed state, never
        killed, and never a reason not to restart OUR OWN dead daemon, whose
        socket is free by definition. A home whose marker names another data
        plane is refused outright: restarting there would be adoption.

        Raises ClaudexorUnavailable (typed) when the binary is missing, the
        home is not ours, or the spawned daemon never published a live
        descriptor.
        """
        from ouroboros.gateways.claudexor import (
            SHORT_POLL_TIMEOUT_SEC,
            ClaudexorUnavailable,
        )

        with self._lock:
            # Validate before probing or claiming even an already-live endpoint.
            ownership_problem = verify_owned_home()
            if ownership_problem:
                raise ClaudexorUnavailable("foreign_daemon_home", ownership_problem)
            endpoint, state, detail = self._classify_liveness()
            if endpoint is not None:
                _write_ownership_marker()
            if state == "foreign_daemon" and detail:
                # A live foreign daemon sits on our STALE descriptor port. Our
                # own daemon is dead (it would hold that port otherwise), so
                # restarting ours is legitimate: the fresh spawn binds a new
                # ephemeral port and rewrites the descriptor. The foreign one
                # is left untouched and the fact is disclosed, not silenced.
                log.warning("owned-daemon restart proceeding past a foreign "
                            "responder on the stale port: %s", detail)
            from ouroboros.claudexor_runtime import ClaudexorRuntimeError, get_runtime_manager

            runtime_manager = get_runtime_manager()
            if endpoint is not None:
                pin = getattr(runtime_manager, "pin", None)
                if (
                    pin is not None
                    and self._engine_version == getattr(pin, "version", None)
                    and self._engine_build_sha == getattr(pin, "build_sha", None)
                ):
                    # The live, authenticated daemon already serves the exact
                    # pinned identity. Never touch its directory here: a broken
                    # on-disk copy of the SAME target would otherwise trigger a
                    # repair that swaps the serving tree under the running
                    # process. Disk repair happens at the next natural start
                    # through the ordinary ensure path (owner decision 2A:
                    # side-by-side, current work is never touched).
                    return endpoint
            if endpoint is None and self._proc is not None and self._proc.poll() is None:
                raise ClaudexorUnavailable("daemon_stop_unconfirmed", "the previous startup child is still alive; no replacement was spawned")
            try:
                command = runtime_manager.ensure()
            except ClaudexorRuntimeError as exc:
                if endpoint is not None:
                    log.warning(
                        "managed runtime ensure failed while the owned daemon remains live: %s", exc
                    )
                    return endpoint
                raise ClaudexorUnavailable(exc.code, str(exc)) from exc
            if endpoint is not None:
                # A newer managed tree may have been staged above, but a live
                # daemon is never hot-swapped. The next natural start selects it.
                return endpoint
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
            runtime = get_runtime_manager().status()
            log_path = config_dir / "daemon.log"
            from ouroboros.config import DATA_DIR
            from ouroboros.process_custody import spawn_supervised

            log.info("Spawning owned claudexord under %s from %s", config_dir, runtime.get("source") or "external")
            with open(log_path, "ab") as sink:
                self._proc = spawn_supervised(
                    command,
                    drive_root=pathlib.Path(DATA_DIR),
                    purpose=CUSTODY_PURPOSE,
                    scope="daemon",
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=sink,
                    stderr=sink,
                )
            deadline = time.monotonic() + _SPAWN_WAIT_SEC
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # An exited child may only have lost the writer lease to another
                # worker. Child state is not completion authority: the shared,
                # authenticated endpoint or this deadline is.
                # RECONCILE: fresh discovery + AUTHENTICATED handshake against
                # the descriptor the new daemon just wrote — the same identity
                # proof attach uses, so a restart never claims a port it does
                # not hold.
                endpoint = self._alive_endpoint(
                    timeout_sec=min(remaining, SHORT_POLL_TIMEOUT_SEC),
                )
                if endpoint is not None:
                    _write_ownership_marker()
                    if self._proc is not None and self._proc.poll() is not None:
                        self._proc = None
                    self._last_error = ""
                    return endpoint
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(_SPAWN_POLL_SEC, remaining))
            tail = ""
            try:
                tail = log_path.read_bytes()[-500:].decode("utf-8", errors="replace")
            except OSError:
                pass
            # OUR OWN child, and it never became a daemon we can reach: leaving it
            # alive orphans a process holding this config dir, and leaving the handle
            # set makes the NEXT ensure_running spawn a second one beside it. Killed
            # here rather than in `stop()`, which by contract only ever terminates a
            # daemon we successfully started.
            self._terminate_child()
            raise ClaudexorUnavailable(
                "daemon_spawn_failed",
                "the owned claudexord did not publish a live control descriptor "
                f"within {_SPAWN_WAIT_SEC:.0f}s"
                + (f"; log tail: {tail}" if tail else ""),
            )

    # Spawn-path rotation deferral (the sprint's `_admit_spawned` /
    # `run_deferred_rotation` pair) was SUPERSEDED at merge by the mainline's
    # `reconcile_rotation`, which rides EVERY `ensure_owned_gateway` (spawn and
    # attach), is conditional and idempotent, and treats the recovery-window
    # 503 as an ordinary retry-next-ensure failure — the same incident class
    # closed without spawn-time state. REACHABLE stays the whole spawn exit
    # predicate; the bounded admission wait stays in `ensure_owned_gateway`.

    def _terminate_child(self) -> bool:
        """Stop our child; forget its handle only after confirmed exit. Caller holds the lock."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self._proc = None
            return False
        from ouroboros.platform_layer import kill_process_tree

        kill_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("owned daemon child %s stop is unconfirmed; handle retained", proc.pid)
            return False
        self._proc = None
        return True

    def reconcile_rotation(self, gateway: Any) -> None:
        """D28 as reconciliation (B3): default the MISSING limit-action
        policies to "rotate", never touching a persisted one.

        The predecessor was a spawn-only best-effort patch: one attempt at
        provisioning, a bare except, and no read-back — so a race with the
        daemon's startup "serving recovery only" window failed it forever,
        attach paths never patched at all, and a harness discovered later was
        never covered. This runs on EVERY ``ensure_owned_gateway`` instead
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
        """Stop verified own roots; report every unconfirmed remainder.

        False also means nothing needed stopping. True requires a confirmed
        stop with no remaining custody. Lock acquisition is bounded separately
        from HTTP connect/read phases and each root's exit wait; there is no
        promised absolute wall-clock deadline for the whole teardown. A
        self-started Popen handle proves direct ownership; attached roots need
        the owned marker and measured ledger identity, plus either an
        authenticated endpoint or a typed transport failure. An explicit token
        refusal and other unknown identity evidence never permit this fallback.
        """
        from ouroboros.config import DATA_DIR
        from ouroboros.gateways.claudexor import SHORT_POLL_TIMEOUT_SEC
        from ouroboros.process_custody import pending_process_stops, stop_ledgered_processes

        if not self._lock.acquire(timeout=SHORT_POLL_TIMEOUT_SEC):
            self._report_stop_unconfirmed("daemon manager lock unavailable; custody unchanged")
            return False
        try:
            root = pathlib.Path(DATA_DIR)
            purposes = {CUSTODY_PURPOSE}
            stopped, unconfirmed = [], []
            self._last_error = ""
            ownership_problem = verify_owned_home(require_marker=True)
            endpoint, state = None, ""
            if not ownership_problem:
                endpoint, state, detail = self._classify_liveness(timeout_sec=SHORT_POLL_TIMEOUT_SEC)
                if detail:
                    self._last_error = detail
            if endpoint is not None or state == _TRANSPORT_UNREACHABLE:
                stopped = stop_ledgered_processes(root, purposes, unconfirmed=unconfirmed)
            child_stopped = self._terminate_child()
            if ownership_problem and owned_daemon_provisioned() and not child_stopped:
                unconfirmed.append("descriptor ownership is unconfirmed")
            unconfirmed.extend(pending_process_stops(root, purposes))
            if self._proc is not None:
                unconfirmed.append("self-started child exit unconfirmed")
            if endpoint is not None and not stopped and not child_stopped:
                unconfirmed.append("authenticated endpoint has no confirmed stopped root")
            if unconfirmed:
                reason = ownership_problem or self._last_error
                if reason:
                    unconfirmed.insert(0, reason)
                self._report_stop_unconfirmed("; ".join(dict.fromkeys(unconfirmed)))
                return False
            self._last_error = ""
            return child_stopped or bool(stopped)
        finally:
            self._lock.release()

    def _report_stop_unconfirmed(self, detail: str) -> None:
        """The lifecycle owner discloses a failed stop in the existing supervisor log."""
        from ouroboros.config import DATA_DIR
        from ouroboros.utils import append_jsonl, utc_now_iso

        self._last_error = detail
        log.critical("Owned Claudexor stop unconfirmed: %s; custody retained", detail)
        append_jsonl(pathlib.Path(DATA_DIR) / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "process_stop_unconfirmed",
            "purpose": CUSTODY_PURPOSE, "reason": detail,
        })


_MANAGER: Optional[OwnedClaudexorDaemon] = None
_MANAGER_LOCK = threading.Lock()


def get_owned_daemon() -> OwnedClaudexorDaemon:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = OwnedClaudexorDaemon()
        return _MANAGER


def ensure_owned_gateway(*, admission_wait_sec: Optional[float] = None) -> Any:
    """Return an authenticated gateway to the lazily ensured owned daemon.

    This is the explicit start/probe seam — the ONE funnel every consumer
    (delegation, review sessions, account surfaces, login) passes through,
    which is why the rotation reconcile rides it: spawn AND attach paths are
    both covered, on every ensure, best-effort (see ``reconcile_rotation``).
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
    admission only. ``ensure_running`` keeps ordinary attach probes on their
    default transport ceiling, while a spawned daemon is reconciled through
    bounded authenticated handshakes inside its existing startup window.
    An expired/failed admission also skips the reconcile: the recovering
    daemon 503s settings reads anyway, and the next ensure retries it.
    """
    from ouroboros.gateways.claudexor import (
        SHORT_POLL_TIMEOUT_SEC, ClaudexorGateway, ClaudexorUnavailable,
    )

    wait = _ADMISSION_WAIT_SEC if admission_wait_sec is None else max(
        0.0, float(admission_wait_sec))
    daemon = get_owned_daemon()
    endpoint = daemon.ensure_running()
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
    "get_owned_daemon",
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
