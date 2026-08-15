"""Effective OpenSSH configuration validation for the execd channel (RWS v2 §5).

A mandated pre-split: the donor's ``remote_ssh.py`` was AT the 1600-line module
gate before any v2 adaptation, and the half that does not belong to a session is
this one — stateless checks over what OpenSSH *actually* resolves for an alias.

The whole idea is that Ouroboros never parses ``~/.ssh/config``.  System OpenSSH
owns authentication, so the owner's Host blocks, ProxyJump, IdentityFile, agent
and strict ``known_hosts`` rules apply untouched.  What we do instead is build
the exact final argv, ask OpenSSH what that argv resolves to (``ssh -G``), and
refuse to spawn when the resolved configuration would compromise the protocol
channel:

* a surviving ``RemoteCommand`` or a forced TTY would corrupt or block frames;
* ``SetEnv``, or a ``SendEnv`` pattern that matches a key we retain, would push
  Home environment values onto the target;
* forwarding/tunnel directives are not silently inherited by the protocol
  channel — intentional browser forwarding is its own bounded operation.

Two probes, deliberately: one WITH our overrides (does the channel survive the
alias?) and one raw (what did the owner actually ask for?).  Checking only the
overridden view would hide a hostile alias; checking only the raw view would
reject aliases our overrides already neutralize.  Everything here reads ``ssh -G``
output in memory only — raw config, ``SetEnv`` values and IdentityFile paths are
never logged or persisted; durable diagnostics carry bounded names and reasons.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import pathlib
import shutil
import subprocess
from collections.abc import Mapping
from typing import Any

from ouroboros.workspace_diagnostics import (
    RemoteWorkspaceError,
    sanitize_execution_text,
)

# Exactly the keys that may be retained for OpenSSH alias auth/ProxyJump on
# Home. Provider, Home-control, MCP and locale values are absent by default —
# the child environment is CONSTRUCTED, never inherited.
SSH_ENV_KEYS = (
    "HOME",
    "PATH",
    "USER",
    "LOGNAME",
    "SSH_AUTH_SOCK",
    "TMPDIR",
)

_UNSAFE_FORWARD_KEYS = ("localforward", "remoteforward", "dynamicforward", "tunnel")
_FALSEY = {"none", "false", "no"}

_LOG = logging.getLogger(__name__)

# ── the OPERATIONAL SSH bounds (RWS v2 §4.4) ─────────────────────────────────
# The SSOT lives HERE, in the transport module that constructs the `-o` options,
# not in `ouroboros/config.py`. It used to live there and be reached by a
# function-local `from ouroboros.config import get_ssh_timeout_sec`, which put a
# `settings_or_owner_state` module on the transport's dependency list — exactly
# what the §3.3 reverse gate forbids, and invisible to it while the gate only
# read module-scope imports. These values need nothing from Home: they are
# env-only integers with per-kind bounds, deliberately NOT in `SETTINGS_DEFAULTS`
# because they are rare operator repairs rather than Settings controls.
# `config.get_ssh_timeout_sec` now delegates here, so every existing caller keeps
# reading ONE table.
#
# Protocol/frame limits and the fixed 15s lost-lease ceiling are NOT here at all:
# they are a safety contract, not a configuration seam.
_SSH_TIMEOUTS: dict[str, tuple[str, int, int]] = {  # kind: (env key, default, hard max)
    "connect": ("OUROBOROS_SSH_CONNECT_TIMEOUT_SEC", 20, 300),
    "keepalive_interval": ("OUROBOROS_SSH_KEEPALIVE_INTERVAL_SEC", 5, 60),
    "keepalive_count": ("OUROBOROS_SSH_KEEPALIVE_COUNT", 3, 12),
    "bootstrap": ("OUROBOROS_SSH_BOOTSTRAP_TIMEOUT_SEC", 120, 900),
    "admission": ("OUROBOROS_SSH_ADMISSION_TIMEOUT_SEC", 60, 300),
    "reconcile": ("OUROBOROS_SSH_RECONCILE_TIMEOUT_SEC", 20, 120),
    "shutdown": ("OUROBOROS_SSH_SHUTDOWN_TIMEOUT_SEC", 5, 30),
}


def get_ssh_timeout_sec(kind: str) -> int:
    """The single configuration seam for OPERATIONAL SSH bounds (RWS v2 §4.4).

    Env-only by design (Appendix E §4.4). An unknown kind RAISES rather than
    defaulting, so a typo at a transport call site cannot silently borrow another
    phase's timeout.
    """

    if kind not in _SSH_TIMEOUTS:
        raise ValueError(f"unknown SSH timeout kind: {kind}")
    key, default, hard_max = _SSH_TIMEOUTS[kind]
    raw = os.environ.get(key, default)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = default
    if parsed < 1:
        parsed = default
    return max(1, min(parsed, hard_max))


def transport_error(
    code: str,
    message: str,
    *,
    phase: str,
    completion: str = "not_started",
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> RemoteWorkspaceError:
    """Build the typed transport error every remote layer raises."""

    return RemoteWorkspaceError(
        code,
        message,
        phase=phase,
        completion=completion,  # type: ignore[arg-type]
        retryable=retryable,
        details=details,
    )


def safe_text(value: Any, limit: int = 2000) -> str:
    """Bound and de-identify text before it can enter a durable diagnostic."""

    text = sanitize_execution_text(value)
    text = " ".join(text.replace("\x00", "").split())
    home = str(pathlib.Path.home())
    if home and home != "/":
        text = text.replace(home, "<home>")
    return text[:limit]


def minimal_ssh_env() -> dict[str, str]:
    """The constructed child environment: retained keys only, nothing ambient."""

    return {key: os.environ[key] for key in SSH_ENV_KEYS if os.environ.get(key)}


def protocol_ssh_options(*, forwarding: bool) -> list[str]:
    """Return the fixed ``-o`` overrides that make a channel frame-safe."""

    options = [
        "RemoteCommand=none",
        "RequestTTY=no",
        "BatchMode=yes",
        f"ConnectTimeout={get_ssh_timeout_sec('connect')}",
        f"ServerAliveInterval={get_ssh_timeout_sec('keepalive_interval')}",
        f"ServerAliveCountMax={get_ssh_timeout_sec('keepalive_count')}",
        "ForwardAgent=no",
        "ForwardX11=no",
        "PermitLocalCommand=no",
        "ClearAllForwardings=yes",
        "Tunnel=no",
        "ControlMaster=no",
        "ControlPath=none",
        "ControlPersist=no",
    ]
    if forwarding:
        # An owner ControlMaster/ControlPersist must not be reused or left
        # alive by a forward either, and a forward that cannot bind must fail
        # rather than publish a dead port.
        options.extend([
            "ControlMaster=no",
            "ControlPath=none",
            "ControlPersist=no",
            "ExitOnForwardFailure=yes",
        ])
    return [token for option in options for token in ("-o", option)]


def _resolved_config(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(" ")
        if separator:
            result.setdefault(key.strip().lower(), []).append(value.strip())
    return result


def _probe(argv: list[str], *, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=timeout_sec,
        check=False,
        env=minimal_ssh_env(),
    )


def _reject_hostile_channel(config: Mapping[str, list[str]]) -> None:
    if (config.get("remotecommand") or ["none"])[-1].lower() not in {"none", ""}:
        raise transport_error(
            "unsupported_ssh_client",
            "The SSH alias retains a hostile RemoteCommand after override.",
            phase="connect",
        )
    if (config.get("requesttty") or ["false"])[-1].lower() not in {
        "false",
        "no",
        "none",
    }:
        raise transport_error(
            "unsupported_ssh_client",
            "The SSH alias forces a terminal and cannot carry execd frames.",
            phase="connect",
        )


def _reject_environment_forwarding(raw_config: Mapping[str, list[str]]) -> None:
    """Refuse before spawn if the alias could push Home values to the target.

    ``SendEnv=-*``/``SetEnv=-*`` are deliberately NOT used to neutralize this:
    Phase 0 proved they do not generically cancel Host-block values on OpenSSH
    10.2p1, so the only honest answer is to refuse.  An unparseable pattern is
    treated as matching (fail closed); a requested key we do not retain is inert.
    """

    if any(
        value and value.lower() != "none"
        for value in raw_config.get("setenv", [])
    ):
        raise transport_error(
            "unsafe_ssh_environment",
            "The SSH alias configures SetEnv; execd refuses ambient remote variables.",
            phase="connect",
        )
    patterns = [
        token
        for value in raw_config.get("sendenv", [])
        for token in value.split()
    ]
    for env_name in SSH_ENV_KEYS:
        selected = False
        for pattern in patterns:
            negate = pattern.startswith("-")
            candidate = pattern[1:] if negate else pattern
            if candidate and fnmatch.fnmatchcase(env_name, candidate):
                selected = not negate
        if selected and os.environ.get(env_name):
            raise transport_error(
                "unsafe_ssh_environment",
                f"The SSH alias SendEnv pattern matches retained key {env_name}.",
                phase="connect",
            )


def _reject_inherited_forwarding(
    config: dict[str, list[str]],
    raw_config: Mapping[str, list[str]],
    *,
    forwarding: bool,
) -> None:
    configured = {
        key
        for key in _UNSAFE_FORWARD_KEYS
        if any(
            value and value.lower() not in _FALSEY
            for value in raw_config.get(key, [])
        )
    }
    if not configured:
        return
    if forwarding:
        # A deliberate forward channel must carry ONLY the forward we recorded,
        # so an alias that adds its own is refused outright.
        raise transport_error(
            "unsafe_ssh_forwarding",
            "The SSH alias contains forwarding or tunnel directives.",
            phase="connect",
        )
    if (config.get("clearallforwardings") or ["no"])[-1].lower() not in {
        "yes",
        "true",
    } or (config.get("tunnel") or ["no"])[-1].lower() not in _FALSEY:
        raise transport_error(
            "unsafe_ssh_forwarding",
            "The SSH alias retains forwarding after protocol overrides.",
            phase="connect",
        )
    # Neutralized, not silently dropped: the owner asked for something we
    # disabled, and that belongs in the diagnostics they can read.
    _LOG.warning(
        "Ignored SSH alias forwarding directives for execd protocol: %s",
        ",".join(sorted(configured)),
    )
    config["_ouroboros_warning_directives"] = sorted(configured)


def validated_ssh_config(
    alias: str,
    ssh_binary: str | None,
    *,
    forwarding: bool,
) -> tuple[list[str], dict[str, list[str]]]:
    """Return `(argv prefix ending in the alias, effective config)`, or raise.

    The argv is constructed FIRST, then resolved through ``ssh -G`` with those
    exact options, so what is validated is what will run.
    """

    binary = str(ssh_binary or "").strip() or shutil.which("ssh")
    if not binary:
        raise transport_error(
            "ssh_unavailable",
            "OpenSSH client is unavailable.",
            phase="connect",
        )
    base = [binary, "-T", *protocol_ssh_options(forwarding=forwarding)]
    connect_timeout = get_ssh_timeout_sec("connect")
    raw_probe = _probe([binary, "-G", alias], timeout_sec=connect_timeout)
    probe = _probe([*base, "-G", alias], timeout_sec=connect_timeout)
    if probe.returncode != 0 or raw_probe.returncode != 0:
        raise transport_error(
            "unsupported_ssh_client",
            "OpenSSH could not resolve the configured host alias.",
            phase="connect",
            details={"stderr": safe_text(probe.stderr)},
        )
    config = _resolved_config(probe.stdout)
    raw_config = _resolved_config(raw_probe.stdout)
    _reject_hostile_channel(config)
    _reject_environment_forwarding(raw_config)
    _reject_inherited_forwarding(config, raw_config, forwarding=forwarding)
    return [*base, alias], config


def validated_ssh_base_command(
    alias: str,
    ssh_binary: str | None = None,
    *,
    forwarding: bool = False,
) -> list[str]:
    """Return a hostile-alias-checked argv prefix ending in the host alias."""

    command, _config = validated_ssh_config(alias, ssh_binary, forwarding=forwarding)
    return command
