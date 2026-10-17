"""Zoom's raw-audio prerequisites on Linux, checked where the SDK will run.

Zoom documents two: a PulseAudio server for the user the SDK runs as (PipeWire's
pulse service counts) and ``~/.config/zoomus.conf`` with ``system.audio.type=default``.
Source: Zoom Developer Docs, Meeting SDK for Linux, "Use raw data". Without them the
SDK joins and delivers no meeting audio, so no call ever opens.

libpulse reaches the server through ``PULSE_SERVER`` or the socket
``$XDG_RUNTIME_DIR/pulse/native``. Ouroboros passes both variables to its companions
when it has them; a service started without ``XDG_RUNTIME_DIR`` still has the user
manager's ``/run/user/<uid>`` when that exists (the rule ``meeting_audio`` applies).
The probe checks the shell it runs in and the companion checks its own environment,
both through ``check``, so each names exactly what is missing where it looked. Only
paths are inspected; nothing is run, installed or started.

Stdlib only and free of sibling imports.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import MutableMapping, Optional, Tuple

PASS, FAIL = "PASS", "FAIL"
USER_RUNTIME_ROOT = Path("/run/user")


def current_uid() -> Optional[int]:
    return os.getuid() if hasattr(os, "getuid") else None


def locate_server(env: MutableMapping[str, str], *, uid: Optional[int],
                  user_runtime_root: Path = USER_RUNTIME_ROOT) -> Tuple[str, str]:
    """``(where, "")`` for the server libpulse would reach, or ``("", what is missing)``.

    An unset ``XDG_RUNTIME_DIR`` is set in ``env`` to ``<user_runtime_root>/<uid>`` when
    that directory exists, so an SDK loaded after this call looks where this check did.
    """
    declared = str(env.get("PULSE_SERVER") or "").strip()
    if declared:
        return f"PULSE_SERVER ({declared})", ""
    runtime = str(env.get("XDG_RUNTIME_DIR") or "").strip()
    if not runtime:
        candidate = user_runtime_root / str(uid) if uid is not None else None
        if candidate is None or not candidate.is_dir():
            where = f", and {candidate} does not exist" if candidate is not None else ""
            return "", (f"PULSE_SERVER and XDG_RUNTIME_DIR are not set{where} "
                        "(this user has no login session or lingering user manager)")
        runtime = env["XDG_RUNTIME_DIR"] = str(candidate)
    socket = Path(runtime) / "pulse" / "native"
    if socket.exists():
        return str(socket), ""
    return "", (f"PULSE_SERVER is not set and {socket} does not exist "
                "(no PulseAudio or PipeWire pulse server runs for this user)")


def zoomus_conf(home: str) -> Tuple[Path, str]:
    """``(path, what is missing)`` for ``~/.config/zoomus.conf``; ``""`` when it is set."""
    conf = Path(home or "~").expanduser() / ".config" / "zoomus.conf"
    try:
        text = conf.read_text(encoding="utf-8")
    except FileNotFoundError:
        return conf, f"{conf} does not exist (it needs system.audio.type=default)"
    except (OSError, UnicodeDecodeError) as exc:
        return conf, f"{conf} is unreadable ({type(exc).__name__}; it needs system.audio.type=default)"
    if not re.search(r"^\s*system\.audio\.type\s*=\s*default\s*$", text, re.MULTILINE):
        return conf, f"{conf} does not set system.audio.type=default"
    return conf, ""


def check(env: MutableMapping[str, str], *, home: str, uid: Optional[int],
          user_runtime_root: Path = USER_RUNTIME_ROOT) -> Tuple[str, str]:
    """``(PASS|FAIL, detail)``: where the server and the config are, or each missing piece."""
    conf, conf_missing = zoomus_conf(home)
    server, server_missing = locate_server(env, uid=uid, user_runtime_root=user_runtime_root)
    missing = [problem for problem in (conf_missing, server_missing) if problem]
    if missing:
        return FAIL, "; ".join(missing)
    return PASS, f"{conf} sets system.audio.type=default; PulseAudio server: {server}"
