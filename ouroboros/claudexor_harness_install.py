"""Connect's repair of a missing vendor CLI through the pinned engine's own installer.

A domain operation of the owned data plane: the exact managed CLI installs the vendor
harness into the owned config home, its JSON receipt is validated strictly, and the
in-flight installer stays in the custody set an emergency stop reaps. The accounts
gateway only invokes it and translates the typed result.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from typing import Any, Dict

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
    from ouroboros.claudexor_daemon import owned_config_dir
    from ouroboros.claudexor_runtime import ClaudexorRuntimeError, get_runtime_manager
    from ouroboros.config import get_claudexor_harness_install_timeout_sec
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.platform_layer import merge_hidden_kwargs, subprocess_new_group_kwargs
    # The same custody set /panic reaps (isolated_deps._run is the template):
    # an in-flight vendor installer must not survive an emergency stop.
    from ouroboros.tools.shell import _active_subprocesses, _kill_process_group, _subprocess_lock
    from ouroboros.tools.shell_process import CommandAdmissionClosed, tracked_start

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
        # Counted before Popen and published as it returns (``tracked_start``): Panic's lock-free
        # request or a begun stop that misses the handle is seen by its publication, which stops it
        # (the round-2 reviewer's interleaving), and Popen never runs under the registry lock.
        with tracked_start() as publish:
            proc = publish(subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                **kwargs,
            ))
    except (OSError, CommandAdmissionClosed) as exc:
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
