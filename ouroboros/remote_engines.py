"""Named remote Claudexor engines over the owner's SSH: the ``OUROBOROS_ENGINE_CONNECTIONS`` rows, the attach-only
session that reaches each engine, and the authenticated gateway bound to it.

A row is ``{"name", "label"?, "ssh_host", "engine_home"?, "remote_root"?, "enabled"?}`` and holds no secret.
``ssh_host`` is a destination the owner's OpenSSH already reaches (a ``~/.ssh/config`` alias or ``user@host``: keys,
agent, ``known_hosts`` and ProxyJump stay OpenSSH's). ``engine_home`` is the engine's configuration root on that host,
whose ``daemon/control-api.json`` names ``{host, port, tokenPath}``: ``~/.claudexor/v3`` (the default) for a user's
own engine, ``<data>/claudexor`` for an Ouroboros installation's owned engine. ``remote_root`` is the folder under
which each staged run copy lives (``remote_workspace``); a connection without it serves model operations only. The
binding is the engine's home and store, never the account SSH logs in as.

Attach-only. The installation holds one tunnel per connection: it reads the running engine's descriptor and the
token it names over SSH, keeps one supervised ``ssh -N -L`` forward (process custody, daemon scope: it outlives the
process that opened it, never the owner's Disconnect or Panic) and publishes a private loopback descriptor of the
ordinary ``{host, port, tokenPath}`` shape under ``<data>/state/remote_engines/<name>/`` (the token file ``0600``,
the custody the owned daemon's token has), so every process reaches the engine through the existing descriptor
reader, its loopback rule and its generations. Nothing here starts, installs, updates, restarts or stops a remote
engine or any of its processes: an engine that does not answer is ``remote_engine_not_running`` and starting one
stays the host owner's act. ``BatchMode`` and ``StrictHostKeyChecking=yes`` make every prompt (password, passphrase,
MFA, an unknown or changed host key) a typed refusal, never answered here, and a user's ``ControlMaster`` or
``LocalCommand`` configuration is switched off, so no detached SSH process escapes custody. Short commands (the
descriptor, the token, a staged copy) are tracked children that Panic and the generation's stop reach.

The bearer token never reaches a log, a status read, a UI or a model. Disconnecting stops the forward only: the
engine and its runs continue, and stopping one run is that run's own control request on its own engine.

A new native run or model operation on a remote engine records ``engine_ref`` ``{"kind": "remote", "connection",
"ssh_host", "engine_home", "store", "remote_root"}`` before its POST (``execution_domain``); every later operation
reopens that connection and requires the same address, home and store, so a re-pointed row refuses
(``remote_engine_changed``), another connection proving the same store is no substitute, and nothing migrates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import posixpath
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable, _is_loopback

log = logging.getLogger(__name__)

ENGINE_CONNECTIONS_KEY = "OUROBOROS_ENGINE_CONNECTIONS"
REMOTE_ENGINE = "remote"
DEFAULT_ENGINE_HOME = "~/.claudexor/v3"
DESCRIPTOR_REL = "daemon/control-api.json"
# The connection-name grammar of named API connections (``provider_models``); ``default`` and ``local`` stay the
# owned engine's spellings.
CONNECTION_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_RESERVED_NAMES = frozenset({"default", "local"})
# An alias, ``user@host``, ``host:port`` or ``user@[v6]``: argv never passes through a local shell, so the only
# hazard is an option (a leading ``-``) or whitespace.
_DESTINATION = re.compile(r"[A-Za-z0-9_.\[][A-Za-z0-9_.@%+:\[\]-]{0,254}")
_ROW_KEYS = frozenset({"name", "label", "ssh_host", "engine_home", "remote_root", "workspaces", "enabled"})
_LABEL_MAX = 80
# Every prompt is a refusal, and nothing the owner's ssh_config adds may outlive or widen this process's custody.
_SSH_OPTIONS = (
    "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ControlMaster=no",
    "-o", "ControlPath=none", "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no",
    "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
)
_OUTPUT_CAP = 1024 * 1024  # a descriptor, token or receipt: anything larger is not the answer asked for
# The run fact files the existing readers take from a run folder (``attempt_containment``, ``final_attempt_facts``,
# ``review_session_reads``), mirrored from a remote run's artifact routes; never a ``.`` or ``..`` part.
_FACT_FILE = re.compile(r"final/telemetry\.yaml|attempts/(?!\.\.?/)[^/]+/(?:attempt\.yaml|events\.jsonl)")
_STDERR_CAP = 64 * 1024
# A live run's fact snapshot older than this may go once a newer complete one exists (the newest few always stay):
# a reader opens the snapshot its own read returned at once; terminal snapshots are never pruned here.
_LIVE_SNAPSHOT_GRACE_SEC = 600.0
_LIVE_SNAPSHOTS_KEPT = 3


def fact_mirror_root(facts: Mapping[str, Any], run_id: str) -> pathlib.Path:
    """The installation's fact snapshots of ``run_id`` on the engine ``facts`` names (a tunnel's facts or a recorded
    ``engine_ref``): under its connection, keyed by its SSH address, engine home and store, so a re-pointed or
    reinstalled engine's run of the same id never reads another engine's facts."""
    from urllib.parse import quote

    place = json.dumps([str(facts.get(key) or "") for key in ("ssh_host", "engine_home", "store")])
    engine = hashlib.sha256(place.encode("utf-8")).hexdigest()[:16]
    return _tunnel_dir(str(facts.get("connection") or "")) / "runs" / engine / quote(str(run_id), safe="")


def _prune_live_snapshots(folder: pathlib.Path, *, keep: pathlib.Path) -> None:
    """Drop live snapshots beyond the newest few once they are older than the grace (never ``keep``)."""
    try:
        snapshots = sorted((path for path in folder.iterdir() if path.is_dir() and path != keep),
                           key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return
    cutoff = time.time() - _LIVE_SNAPSHOT_GRACE_SEC
    for path in snapshots[_LIVE_SNAPSHOTS_KEPT - 1:]:
        try:
            if path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def _unavailable(code: str, message: str) -> ClaudexorUnavailable:
    return ClaudexorUnavailable(code, message)



# --- the rows ----------------------------------------------------------------------------------------------

def remote_path_error(value: Any, *, home_ok: bool = True) -> str:
    """Why ``value`` is not a usable folder on a remote host ("" when it is): absolute (or ``~/``-relative to the
    SSH user's home), without control characters, empty, ``.`` or ``..`` parts, and never the root itself."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return "is required: an absolute folder on that host"
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        return "must not contain control characters"
    relative = value[2:] if home_ok and value.startswith("~/") else None
    if relative is None and not value.startswith("/"):
        return "must be an absolute path" + (" or start with ~/" if home_ok else "")
    body = relative if relative is not None else value[1:]
    if not body or any(part in ("", ".", "..") for part in body.split("/")):
        return "must name a folder below the root, without empty, '.' or '..' parts"
    return ""


def normalize_engine_connections(raw: Any) -> Tuple[List[dict], str]:
    """``(rows, error)`` for a saved ``OUROBOROS_ENGINE_CONNECTIONS``: the rows as written, or the first problem as
    one owner-facing sentence (nothing saved). Shape only: no host is contacted."""
    key = ENGINE_CONNECTIONS_KEY
    if raw is None:
        return [], ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else []
        except ValueError:
            return [], f"{key} must be a JSON list of engine connections."
    if not isinstance(raw, list):
        return [], f"{key} must be a list of engine connections."
    rows, seen = [], set()
    for index, row in enumerate(raw):
        where = f"{key}[{index}]"
        if not isinstance(row, dict):
            return [], f"{where} must be an object."
        unknown = sorted(set(row) - _ROW_KEYS)
        if unknown:
            return [], f"{where}: unknown field(s) {', '.join(unknown)}."
        name = row.get("name")
        if not isinstance(name, str) or not CONNECTION_NAME.fullmatch(name) or name in _RESERVED_NAMES:
            return [], (f"{where}: name must be 1-32 lowercase letters, digits, '-' or '_', starting with a letter "
                        "or digit, and is neither 'default' nor 'local'.")
        if name in seen:
            return [], f"{where}: the engine connection {name!r} is defined twice."
        seen.add(name)
        host = row.get("ssh_host")
        if not isinstance(host, str) or not _DESTINATION.fullmatch(host):
            return [], (f"{where}: ssh_host must be an SSH destination this computer already reaches, such as an "
                        "alias from ~/.ssh/config or user@host.")
        for name_of_field, home_ok in (("engine_home", True), ("remote_root", False)):  # the engine reads remote_root
            if name_of_field in row and (error := remote_path_error(row[name_of_field], home_ok=home_ok)):
                return [], f"{where}: {name_of_field} {error}."
        workspaces = row.get("workspaces", [])
        if not isinstance(workspaces, list) or len(workspaces) > 32:
            return [], f"{where}: workspaces must be a list of at most 32 absolute folders on that host."
        for item in workspaces:
            if error := remote_path_error(item, home_ok=False):
                return [], f"{where}: workspace {item!r} {error}."
        label = row.get("label", "")
        if not isinstance(label, str) or len(label) > _LABEL_MAX or any(ord(ch) < 32 for ch in label):
            return [], f"{where}: label must be one line of at most {_LABEL_MAX} characters."
        if not isinstance(row.get("enabled", True), bool):
            return [], f"{where}: enabled must be true or false."
        rows.append(dict(row))
    return rows, ""


def engine_connection_rows(settings: Optional[Mapping[str, Any]] = None) -> List[dict]:
    """The saved rows of ``settings`` (else this process's runtime view); an unusable saved value names none."""
    if settings is None:
        from ouroboros.settings_integrity import runtime_setting

        raw = runtime_setting(ENGINE_CONNECTIONS_KEY, None)
    else:
        raw = settings.get(ENGINE_CONNECTIONS_KEY)
    rows, error = normalize_engine_connections(raw)
    if error:
        log.warning("Engine connections unreadable: %s", error)
    return rows if not error else []


def _resolved(row: Mapping[str, Any]) -> dict:
    return {"name": str(row["name"]), "label": str(row.get("label") or "").strip(), "ssh_host": str(row["ssh_host"]),
            "engine_home": str(row.get("engine_home") or DEFAULT_ENGINE_HOME),
            "remote_root": str(row.get("remote_root") or ""), "workspaces": list(row.get("workspaces") or []),
            "enabled": row.get("enabled", True) is not False}


def engine_connection(name: str, settings: Optional[Mapping[str, Any]] = None, *, new_work: bool = True) -> dict:
    """One configured connection with its defaults; a missing row, or a disabled one for ``new_work``, refuses.

    Recorded work keeps reaching a disabled connection (collection, Stop): switching it off takes no new work."""
    row = next((item for item in engine_connection_rows(settings) if item.get("name") == name), None)
    if row is None:
        raise _unavailable("remote_engine_unconfigured", f"No engine connection named {name!r} is configured in "
                           "Settings › Accounts and keys › Remote engines; nothing was sent, and nothing moved to "
                           "another engine.")
    resolved = _resolved(row)
    if new_work and not resolved["enabled"]:
        raise _unavailable("remote_engine_disabled", f"The engine connection {name!r} is switched off, so it takes no "
                           "new work; nothing was sent, and nothing moved to another engine.")
    return resolved


# --- SSH ---------------------------------------------------------------------------------------------------

def _ssh_executable() -> str:
    from ouroboros.platform_layer import bootstrap_process_path

    bootstrap_process_path()
    found = shutil.which("ssh")
    if not found:
        raise _unavailable("remote_ssh_missing", "No ssh program is on this computer's PATH, so no remote engine can "
                           "be reached; install OpenSSH's client.")
    return found


def remote_shell_path(path: str) -> str:
    """``path`` as one word of a POSIX shell on the host: quoted, a ``~/`` prefix read from that shell's ``$HOME``."""
    return f'"$HOME"/{shlex.quote(path[2:])}' if path.startswith("~/") else shlex.quote(path)


def ssh_argv(destination: str, script: str) -> List[str]:
    """One ssh invocation running ``script`` under the host's ``/bin/sh`` (whatever the login shell is), unforwarded."""
    if not _DESTINATION.fullmatch(destination or ""):
        raise _unavailable("remote_engine_invalid", "The engine connection's SSH destination is not usable.")
    return [_ssh_executable(), *_SSH_OPTIONS, "-o", "ClearAllForwardings=yes", destination,
            f"sh -c {shlex.quote(script)}"]


def ssh_failure(code: int, stderr: bytes, what: str) -> ClaudexorUnavailable:
    """A typed SSH or remote-command failure; the text is OpenSSH's or the host's own, bounded, never a token."""
    text = stderr.decode("utf-8", "replace").strip()[-600:]
    if code == 255:
        lowered = text.lower()
        if "host key verification failed" in lowered or "host key for" in lowered:
            return _unavailable("remote_ssh_host_key_unverified", f"{what}: OpenSSH did not verify the host key "
                                f"({text or 'no detail'}). Connect once in a terminal and accept the key you checked.")
        if "permission denied" in lowered or "passphrase" in lowered or "password" in lowered:
            return _unavailable("remote_ssh_needs_interaction", f"{what}: OpenSSH needs an interactive sign-in "
                                f"({text or 'no detail'}); Ouroboros never answers prompts. Load the key into "
                                "ssh-agent or configure key authentication for this alias.")
        return _unavailable("remote_ssh_failed", f"{what}: ssh failed ({text or 'exit 255'}).")
    return _unavailable("remote_command_failed", f"{what}: the host answered exit {code} ({text or 'no detail'}).")


@dataclass
class SshResult:
    code: int
    stdout: bytes
    stderr: bytes


def run_ssh(destination: str, script: str, *, timeout_sec: float, what: str,
            feed: Optional[Callable[[Any], None]] = None, sink: Any = None, check: bool = True) -> SshResult:
    """One bounded SSH command: a tracked child (Panic and the generation's stop reach it), killed at
    ``timeout_sec``. ``feed(stdin)`` streams its input on a thread; ``sink`` receives its whole stdout (otherwise
    at most ``_OUTPUT_CAP`` bytes are kept, more is a typed overflow). A non-zero exit raises unless ``check`` is
    False."""
    from ouroboros.platform_layer import merge_hidden_kwargs, subprocess_new_group_kwargs
    from ouroboros.tools.shell_process import (CommandAdmissionClosed, _kill_process_group, contained_popen,
                                               retire_tracked, tracked_start)

    argv = ssh_argv(destination, script)
    try:
        with tracked_start() as publish:
            proc = publish(contained_popen(
                subprocess.Popen, argv, stdin=subprocess.PIPE if feed else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, **merge_hidden_kwargs(subprocess_new_group_kwargs())))
    except CommandAdmissionClosed as exc:
        raise _unavailable("remote_engine_stopping", f"{what}: this generation is stopping ({exc}); nothing was sent.") from exc
    except OSError as exc:
        raise _unavailable("remote_ssh_unavailable", f"{what}: ssh could not start ({type(exc).__name__}).") from exc
    out, err, state = bytearray(), bytearray(), {}

    def drain_out() -> None:
        try:
            for chunk in iter(lambda: proc.stdout.read(65536), b""):
                if sink is not None:
                    sink.write(chunk)
                elif len(out) + len(chunk) > _OUTPUT_CAP:
                    state["overflow"] = True
                else:
                    out.extend(chunk)
        except Exception as exc:  # a sink failure ends the read; the command is then killed below
            state["read_error"] = f"{type(exc).__name__}: {exc}"

    def drain_err() -> None:
        for chunk in iter(lambda: proc.stderr.read(4096), b""):
            if len(err) < _STDERR_CAP:
                err.extend(chunk[:_STDERR_CAP - len(err)])

    def write_in() -> None:
        try:
            feed(proc.stdin)
        except Exception as exc:
            state["feed_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    threads = [threading.Thread(target=target, name=f"remote-ssh-{index}", daemon=True)
               for index, target in enumerate((drain_out, drain_err, *((write_in,) if feed else ())))]
    for thread in threads:
        thread.start()
    try:
        code = proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(proc)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        raise _unavailable("remote_ssh_timeout", f"{what}: no answer within {timeout_sec:.0f}s; the command was "
                           "stopped on this computer, and its effect on the host is unknown.") from exc
    finally:
        retire_tracked(proc)
        for thread in threads:
            thread.join(timeout=10)
    result = SshResult(code, bytes(out), bytes(err))
    if check and code != 0:  # the host's own words first: a refused stream also breaks its pipe
        raise ssh_failure(code, result.stderr, what)
    if state.get("feed_error") or state.get("read_error"):
        raise _unavailable("remote_transfer_failed", f"{what}: {state.get('feed_error') or state.get('read_error')}")
    if state.get("overflow"):
        raise _unavailable("remote_command_failed", f"{what}: the host's answer exceeded {_OUTPUT_CAP} bytes.")
    return result


# --- the installation's tunnel per connection ---------------------------------------------------------------

_LOCK = threading.Lock()
_STATUS: Dict[str, Dict[str, Any]] = {}
_HELD: Dict[str, Any] = {}  # this process's own Popen of a forward it spawned: reaped here once it is stopped


def tunnel_purpose(name: str) -> str:
    """The process-custody purpose of ``name``'s forward (``process_custody``)."""
    return f"remote_engine_tunnel:{name}"


def _tunnel_dir(name: str) -> pathlib.Path:
    from ouroboros import config

    return pathlib.Path(config.DATA_DIR) / "state" / "remote_engines" / name  # ``remote_stops.TUNNEL_HOME``


def _note(name: str, state: str, **facts: Any) -> None:
    from ouroboros.utils import utc_now_iso

    with _LOCK:
        _STATUS[name] = {"state": state, "at": utc_now_iso(), **facts}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _descriptor(row: dict) -> Dict[str, Any]:
    """The running engine's ``{host, port, tokenPath}`` at ``engine_home``, with the SSH login's uid and the
    descriptor owner's (the engine's user): a read, never a start."""
    from ouroboros.runtime_limits import REMOTE_ENGINE_SSH_COMMAND_SEC

    path = remote_shell_path(f"{row['engine_home'].rstrip('/')}/{DESCRIPTOR_REL}")
    script = (f"f={path}; if [ ! -r \"$f\" ]; then echo 'no readable engine descriptor' >&2; exit 66; fi; "
              "id -u; (stat -c %u \"$f\" 2>/dev/null || stat -f %u \"$f\"); cat \"$f\"")
    what = f"Reading the engine descriptor on {row['name']}"
    answer = run_ssh(row["ssh_host"], script, timeout_sec=REMOTE_ENGINE_SSH_COMMAND_SEC, check=False, what=what)
    if answer.code == 66:
        raise _unavailable("remote_engine_not_running", f"No running engine was found at {row['engine_home']} on "
                           f"{row['name']}: its descriptor is absent or unreadable for this SSH user. Ouroboros only "
                           "attaches to a running engine and never starts one.")
    if answer.code != 0:
        raise ssh_failure(answer.code, answer.stderr, what)
    lines = answer.stdout.decode("utf-8", "replace").splitlines()
    try:
        transport_uid, owner_uid, body = int(lines[0].strip()), int(lines[1].strip()), json.loads("\n".join(lines[2:]))
        host, port, token_path = str(body.get("host") or ""), int(body.get("port") or 0), str(body.get("tokenPath") or "")
    except (IndexError, ValueError, TypeError, AttributeError) as exc:
        raise _unavailable("remote_engine_descriptor_unreadable", f"The engine descriptor on {row['name']} is not "
                           "the {host, port, tokenPath} object an engine writes.") from exc
    if not _is_loopback(host) or not 0 < port < 65536 or not token_path.startswith("/"):
        raise _unavailable("remote_engine_descriptor_unreadable", f"The engine descriptor on {row['name']} names a "
                           "non-loopback address, an invalid port or a relative token path; nothing was forwarded.")
    return {"remote_host": host, "remote_port": port, "remote_token_path": token_path,
            "transport_uid": transport_uid, "engine_owner_uid": owner_uid}


def _token(row: dict, token_path: str) -> str:
    from ouroboros.runtime_limits import REMOTE_ENGINE_SSH_COMMAND_SEC

    answer = run_ssh(row["ssh_host"], f"cat {shlex.quote(token_path)}", timeout_sec=REMOTE_ENGINE_SSH_COMMAND_SEC,
                     what=f"Reading the engine token on {row['name']}")
    token = answer.stdout.decode("utf-8", "replace").strip()
    if not token or any(ch.isspace() for ch in token):
        raise _unavailable("remote_engine_token_unreadable", f"The engine token on {row['name']} is empty or "
                           "malformed; nothing was forwarded.")
    return token


def _write_private(path: pathlib.Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` readable by this user only (the token, like the owned daemon's)."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def published(name: str) -> Optional[dict]:
    """The installation's published tunnel facts for ``name`` (no token), or ``None``: no remote contact."""
    from ouroboros.utils import read_json_dict

    return read_json_dict(_tunnel_dir(name) / "control-api.json")


def _matches(published: Optional[dict], row: dict) -> bool:
    return bool(published) and all(published.get(key) == row[key] for key in ("ssh_host", "engine_home")) and bool(
        published.get("store"))


def _unpublish(name: str) -> None:
    for leaf in ("control-api.json", "token"):
        try:
            (_tunnel_dir(name) / leaf).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            log.warning("Remote engine tunnel file %s/%s could not be removed", name, leaf, exc_info=True)


def _stop_tunnel(name: str, *, timeout_sec: float = 3.0) -> Dict[str, List[Any]]:
    """Stop ``name``'s forward by its measured identity (``process_custody``; the engine is untouched): ``{"stopped":
    [pid], "unconfirmed": [why]}``, plus what custody still holds read strictly (``pending_process_stops``, as the
    owned daemon's stop does): an unreadable ledger or a live row is unconfirmed, never "nothing was running"."""
    from ouroboros import config
    from ouroboros.process_custody import pending_process_stops, stop_ledgered_processes

    root, purposes, unconfirmed = pathlib.Path(config.DATA_DIR), {tunnel_purpose(name)}, []
    try:
        stopped = stop_ledgered_processes(root, purposes, timeout_sec=timeout_sec, unconfirmed=unconfirmed)
    except Exception as exc:
        log.warning("Stopping the remote engine tunnel %s failed", name, exc_info=True)
        return {"stopped": [], "unconfirmed": [f"the stop could not be carried out ({type(exc).__name__})"]}
    with _LOCK:
        held = _HELD.pop(name, None) if stopped else None
    if held is not None:
        try:
            held.wait(timeout=max(0.0, min(timeout_sec, 2.0)))  # reap our own child; another process's is init's
        except Exception:
            pass
    try:
        unconfirmed += [why for why in pending_process_stops(root, purposes) if why not in unconfirmed]
    except Exception as exc:
        unconfirmed.append(f"what process custody still holds could not be read ({type(exc).__name__})")
    return {"stopped": list(stopped), "unconfirmed": list(unconfirmed)}


def _stop_refusal(name: str, outcome: Dict[str, List[Any]], *, reason: str) -> Optional[ClaudexorUnavailable]:
    """``None`` for a confirmed stop; else noted, and the typed refusal to open another forward beside this one."""
    detail = "; ".join(str(why) for why in outcome["unconfirmed"])[:600]
    if not detail:
        return None
    _note(name, "stop_unconfirmed", reason=reason, message=detail)
    return _unavailable("remote_engine_stop_unconfirmed", f"Its tunnel to {name} is not confirmed stopped ({detail}): "
                        "it stays in custody with its descriptor and no second one was opened; Disconnect it again.")


def _open_tunnel(row: dict) -> dict:
    """Attach: read the running engine's descriptor and token, hold one supervised ``ssh -N -L`` (daemon scope:
    it survives this process, not the owner's Disconnect or Panic), publish the private loopback descriptor and
    prove the engine's store through it."""
    from ouroboros import config
    from ouroboros.process_custody import spawn_supervised
    from ouroboros.runtime_limits import REMOTE_ENGINE_FORWARD_READY_SEC
    from ouroboros.utils import atomic_write_json, utc_now_iso

    name, folder = row["name"], _tunnel_dir(row["name"])
    facts = _descriptor(row)
    token = _token(row, facts["remote_token_path"])
    local = _free_port()
    host = facts["remote_host"]
    target = f"[{host}]" if ":" in host else host
    argv = [_ssh_executable(), *_SSH_OPTIONS, "-N", "-o", "ExitOnForwardFailure=yes",
            "-L", f"127.0.0.1:{local}:{target}:{facts['remote_port']}", row["ssh_host"]]
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    # An ordinary member of the installation's process scope (it inherits the marker): Panic's census ends it with
    # every other local process, never after a network step. A stop Panic could not deliver through it is owed and
    # delivered at the next start (``remote_stops``).
    with open(folder / "ssh.log", "ab") as diagnostics:
        proc = spawn_supervised(argv, drive_root=pathlib.Path(config.DATA_DIR), purpose=tunnel_purpose(name),
                                scope="daemon", stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=diagnostics)
    with _LOCK:
        _HELD[name] = proc
    deadline = time.monotonic() + REMOTE_ENGINE_FORWARD_READY_SEC
    ready = False
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            with socket.create_connection(("127.0.0.1", local), timeout=0.5):
                ready = True
                break
        except OSError:
            time.sleep(0.1)
    if not ready:  # the forward that never answered may still run: say so, never "failed and gone"
        if refusal := _stop_refusal(name, _stop_tunnel(name), reason="forward_not_ready"):
            raise refusal
        detail = (folder / "ssh.log").read_bytes()[-600:]
        raise ssh_failure(255, detail, f"Opening the forward to {name}")
    _write_private(folder / "token", token.encode("utf-8"))
    published = {"host": "127.0.0.1", "port": local, "tokenPath": str(folder / "token"), "connection": name,
                 "ssh_host": row["ssh_host"], "engine_home": row["engine_home"], "pid": proc.pid,
                 "connected_at": utc_now_iso(), **{k: v for k, v in facts.items() if k != "remote_token_path"}}
    atomic_write_json(folder / "control-api.json", published)
    try:
        client = RemoteEngineGateway(published)
        try:
            client.handshake()
        finally:
            client.close()
        if not client._store_identity:
            raise _unavailable("remote_engine_identity_unproven", f"The engine on {name} did not prove its store "
                               "identity, so its answers could not be told apart from another engine's.")
    except ClaudexorUnavailable as exc:  # unconfirmed: its descriptor stays as the evidence of what may still run
        if refusal := _stop_refusal(name, _stop_tunnel(name), reason="handshake_failed"):
            raise refusal from exc
        _unpublish(name)
        if exc.code == "daemon_unreachable":
            raise _unavailable("remote_engine_not_running", f"The engine at {row['engine_home']} on {name} does not "
                               "answer through the forward; Ouroboros only attaches to a running engine and never "
                               "starts or restarts one.") from exc
        raise
    published.update(store=client._store_identity, engine_version=client.engine_version,
                     engine_build_sha=client.engine_build_sha)
    atomic_write_json(folder / "control-api.json", published)
    return published


def _alive(published: dict) -> bool:
    """Whether the forward ``published`` names is still the installation's own: the published pid is a live custody
    row of its connection's tunnel purpose, by its measured birth and command (``process_custody``). No socket is
    opened, so a status read touches no host, and a port another local process took after the forward died is never
    mistaken for the tunnel (its next handshake would hand that process the engine's token). The process that spawned
    it answers from its own child, whose pid no other process can hold until it is reaped: no measurement is needed."""
    from ouroboros import config
    from ouroboros.process_custody import process_stop_snapshot

    with _LOCK:
        held = _HELD.get(str(published.get("connection") or ""))
    try:
        pid = int(published.get("pid") or 0)
        if held is not None and held.pid == pid:
            return held.poll() is None
        rows = process_stop_snapshot(pathlib.Path(config.DATA_DIR), {tunnel_purpose(str(published.get("connection")))})
    except (OSError, ValueError):  # an unreadable ledger proves no tunnel
        return False
    return pid > 0 and any(int(row.get("pid") or 0) == pid for row in rows)


def tunnel(name: str, *, new_work: bool = True, settings: Optional[Mapping[str, Any]] = None,
           attach: bool = True) -> dict:
    """The installation's live tunnel facts for ``name`` (no token), attaching once under the connection's lock
    when none answers (never with ``attach=False``: a terminal path spawns nothing); a re-pointed row (another
    destination or engine home) replaces the old tunnel."""
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.runtime_limits import REMOTE_ENGINE_SSH_COMMAND_SEC

    row = engine_connection(name, settings, new_work=new_work)
    current = published(name)
    if _matches(current, row) and _alive(current):
        return current
    if not attach:
        raise _unavailable("remote_engine_offline", f"No tunnel to {name} is open, and this path opens none.")
    folder = _tunnel_dir(name)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    held = acquire_exclusive_file_lock(folder / "attach.lock", timeout_sec=REMOTE_ENGINE_SSH_COMMAND_SEC * 4,
                                       stale_sec=REMOTE_ENGINE_SSH_COMMAND_SEC * 8)
    if held is None:
        raise _unavailable("remote_engine_busy", f"Another attach to {name} is still in progress; try again shortly.")
    try:
        current = published(name)
        if _matches(current, row) and _alive(current):
            return current
        if refusal := _stop_refusal(name, _stop_tunnel(name), reason="reattach"):  # the earlier forward goes first
            raise refusal
        _unpublish(name)
        _note(name, "connecting")
        try:
            current = _open_tunnel(row)
        except ClaudexorUnavailable as exc:
            if exc.code != "remote_engine_stop_unconfirmed":  # that one was noted as what it is
                _note(name, "needs_interaction" if exc.code in {"remote_ssh_needs_interaction",
                                                                "remote_ssh_host_key_unverified"} else "failed",
                      code=exc.code, message=str(exc)[:600])
            raise
        _note(name, "connected", engine_version=current.get("engine_version", ""), store=current.get("store", ""))
        return current
    finally:
        release_exclusive_file_lock(folder / "attach.lock", held)


def retire(name: str, *, reason: str = "disconnected") -> Optional[bool]:
    """Close the installation's tunnel to ``name``; the engine and its runs continue. ``True``: our forward stopped;
    ``False``: none of ours ran; ``None``: unconfirmed (custody row and descriptor kept: it may still be up)."""
    outcome = _stop_tunnel(name)
    if _stop_refusal(name, outcome, reason=reason) is not None:
        return None
    _unpublish(name)
    _note(name, "offline", reason=reason)
    return bool(outcome["stopped"])


def panic_close(names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Panic's last remote step (and the next boot's, for a forward Panic's exit outran): request every tunnel's stop
    by its ledger identity, unwaited, and drop the published tokens for safety. What the stop could not confirm is
    returned and noted, never called closed; its custody row stays for the reaper and the next start."""
    stopped: Dict[str, Any] = {}
    try:
        names = names if names is not None else [str(row.get("name")) for row in engine_connection_rows()]
        for name in names:
            stopped[name] = _stop_tunnel(name, timeout_sec=0)
            _unpublish(name)
            _stop_refusal(name, stopped[name], reason="panic")  # noted when unconfirmed; Panic raises nothing
    except Exception as exc:  # never in a Panic's or a boot's way
        stopped["error"] = type(exc).__name__
    return {"requested": True, "tunnels": stopped}


def status(settings: Optional[Mapping[str, Any]] = None) -> List[dict]:
    """Every configured connection with its last observation (an unconfirmed stop stays so until a confirmed stop
    or a noted reconnect) and the stops Emergency Stop owes (``owed_stops``, ``null``: unreadable), then each removed
    connection still owed one (``removed``: no tunnel and no delivery until a row names it again); no host contact."""
    from ouroboros import config
    from ouroboros.remote_stops import owed_stop_counts

    try:
        owed: Optional[Dict[str, Dict[str, int]]] = owed_stop_counts(config.DATA_DIR)
    except Exception:
        log.warning("Owed remote stops could not be read for the engine status", exc_info=True)
        owed = None
    out = []
    for row in engine_connection_rows(settings):
        resolved = _resolved(row)
        with _LOCK:
            observed = dict(_STATUS.get(resolved["name"]) or {})
        current = published(resolved["name"])
        if _matches(current, resolved) and _alive(current) and observed.get("state") != "stop_unconfirmed":
            observed = {"state": "connected", **{k: current.get(k) for k in (
                "engine_version", "store", "connected_at", "transport_uid", "engine_owner_uid")}}
        elif observed.get("state") in (None, "connected"):
            observed = {**observed, "state": "offline" if observed else "unknown"}
        out.append({**resolved, "observation": observed,
                    "owed_stops": None if owed is None else owed.get(resolved["name"], {"runs": 0, "starts": 0})})
    named = {row["name"] for row in out}
    return out + [{"name": name, "label": name, "ssh_host": "", "engine_home": "", "remote_root": "", "workspaces": [],
                   "enabled": False, "observation": {"state": "removed"}, "owed_stops": counts, "removed": True}
                  for name, counts in sorted((owed or {}).items()) if name not in named]


def check(name: str, *, settings: Optional[Mapping[str, Any]] = None) -> dict:
    """The owner's Check: prove the engine through the live tunnel, attaching only when none answers, and read what
    it serves; failures are facts. A live tunnel and the work on it are left as they are (Disconnect is separate)."""
    try:
        client = gateway(name, new_work=False, settings=settings)
        try:
            harnesses = client.harnesses()
            sources = client.list_model_sources()
        finally:
            client.close()
    except ClaudexorUnavailable as exc:
        return {"name": name, "ok": False, "code": exc.code, "message": str(exc)[:600]}
    facts = client.tunnel
    return {"name": name, "ok": True, "engine_version": facts.get("engine_version", ""), "store": facts.get("store", ""),
            "user_mismatch": facts.get("transport_uid") != facts.get("engine_owner_uid"),
            "harnesses": sorted(str(row.get("id") or "") for row in harnesses if isinstance(row, dict)),
            "model_sources": sorted(str(row.get("id") or "") for row in (sources.get("sources") or [])
                                    if isinstance(row, dict))}


# --- the gateway -------------------------------------------------------------------------------------------

def remote_ref(published: Mapping[str, Any]) -> dict:
    """The binding a record keeps: the connection, its SSH address and engine home, the proven store, the root."""
    return {"kind": REMOTE_ENGINE, "connection": str(published.get("connection") or ""),
            "ssh_host": str(published.get("ssh_host") or ""), "engine_home": str(published.get("engine_home") or ""),
            "store": str(published.get("store") or ""), "remote_root": str(published.get("remote_root") or "")}


class RemoteEngineGateway(ClaudexorGateway):
    """The Claudexor transport to one remote engine through the installation's published tunnel.

    The endpoint is the private loopback descriptor (``_endpoint_from_descriptor``: loopback rule, token file,
    descriptor generation), so store identity holds only while that tunnel is the one published. A run detail's
    ``runDir`` is a path on the host and moves to ``remoteRunDir``. A start's local roots are staged on the host and
    the request names their copies (``remote_workspace.staged_request``): nothing local is registered or granted."""

    stages_workspace = True

    def __init__(self, published: Mapping[str, Any]):
        from ouroboros.gateways.claudexor import _endpoint_from_descriptor

        self.tunnel = dict(published)
        super().__init__(_endpoint_from_descriptor(pathlib.Path(_tunnel_dir(self.tunnel["connection"]))
                                                   / "control-api.json"))
        self.engine_ref: Optional[dict] = None

    def handshake(self, *, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        body = super().handshake(timeout_sec=timeout_sec)
        recorded = str(self.tunnel.get("store") or "")
        if recorded and self._store_identity != recorded:
            raise _unavailable("remote_engine_changed", f"The engine behind {self.tunnel['connection']} now serves "
                               "another store than its tunnel proved; nothing was done.")
        self.engine_ref = remote_ref(self.tunnel) if recorded else None
        return body

    def get_run(self, run_id: str, *, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        """The run detail with its host ``runDir`` moved to ``remoteRunDir``; ``runDir`` then names this
        installation's snapshot of the run's fact files as THIS read found them (``_mirror_facts``), so the existing
        readers see what the engine recorded now and nothing older; a read that is not complete names no
        ``runDir`` (its facts are unknown, never an earlier read's)."""
        detail = super().get_run(run_id, timeout_sec=timeout_sec)
        summary = detail.get("summary") if isinstance(detail, dict) else None
        if isinstance(summary, dict) and "runDir" in summary:
            moved = {**{k: v for k, v in summary.items() if k != "runDir"}, "remoteRunDir": summary["runDir"]}
            folder, facts = self._mirror_facts(run_id, str(summary.get("state") or ""), str(summary["runDir"] or ""))
            detail["summary"] = {**moved, **({"runDir": str(folder)} if folder is not None else {}),
                                 "remoteArtifacts": facts}
        return detail

    def _mirror_facts(self, run_id: str, state: str,
                      remote_dir: str = "") -> Tuple[Optional[pathlib.Path], Dict[str, Any]]:
        """Snapshot the run's fact files (``final/telemetry.yaml``, ``attempts/*/attempt.yaml`` and, once the run is
        terminal, ``attempts/*/events.jsonl``) as one read of ``GET /v2/runs/:id/artifacts`` lists them, each fetched
        whole now (a size never stands for content): a text fact the artifact route refuses for its size (413) is
        read whole from the run's own folder over SSH. A complete read is published under its content digest
        (``live/`` while the run works, ``final/`` once terminal: the snapshot terminal settlement records and later
        readers reopen); one with an unread listing or file publishes nothing and names what was not read."""
        from urllib.parse import quote

        try:
            listing = self._request("GET", f"/v2/runs/{quote(str(run_id), safe='')}/artifacts")
        except ClaudexorUnavailable as exc:
            return None, {"state": "unread", "code": exc.code}
        from ouroboros.delegate_custody import TERMINAL_STATES

        terminal = state in TERMINAL_STATES
        wanted = sorted({str((row or {}).get("path") or "") for row in (listing or {}).get("artifacts") or []
                         if isinstance(row, dict) and row.get("kind") != "directory"
                         and _FACT_FILE.fullmatch(str(row.get("path") or ""))
                         and (terminal or not str(row.get("path")).endswith("events.jsonl"))})
        base = fact_mirror_root(self.tunnel, run_id)
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        import tempfile

        reading = pathlib.Path(tempfile.mkdtemp(prefix=".read-", dir=base))  # this read's own: no shared partial
        try:
            digests, unread = {}, {}
            for path in wanted:
                target = reading / path
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    digests[path] = self._fetch_fact(run_id, path, target, remote_dir)
                except (ClaudexorUnavailable, OSError) as exc:
                    unread[path] = str(getattr(exc, "code", "") or type(exc).__name__)
            if unread:
                return None, {"state": "partial", "unread": unread, "read": sorted(digests)}
            identity = hashlib.sha256(json.dumps({"state": state, "files": sorted(digests.items())},
                                                 sort_keys=True).encode("utf-8")).hexdigest()
            folder = base / ("final" if terminal else "live") / identity[:32]
            folder.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.rename(reading, folder)  # a complete snapshot appears at once; one with the same content stays
            except OSError:
                if not folder.is_dir():
                    raise
            if not terminal:
                _prune_live_snapshots(folder.parent, keep=folder)
            return folder, {"state": "mirrored", "snapshot": identity, "files": dict(sorted(digests.items()))}
        except OSError as exc:
            return None, {"state": "unread", "code": type(exc).__name__}
        finally:
            shutil.rmtree(reading, ignore_errors=True)

    def _fetch_fact(self, run_id: str, path: str, target: pathlib.Path, remote_dir: str) -> str:
        """One fact file's whole bytes into ``target`` and their SHA-256: the artifact route, else, when it refuses
        the file for its size, the run's own folder over SSH (the readers read a local run's file whole too)."""
        try:
            with open(target, "wb") as sink:
                return str(self.stream_run_artifact(run_id, path, sink)["sha256"])
        except ClaudexorUnavailable as exc:
            if exc.status_code != 413 or remote_path_error(remote_dir, home_ok=False):
                raise
        from ouroboros.runtime_limits import REMOTE_ENGINE_TRANSFER_SEC

        with open(target, "wb") as sink:
            run_ssh(self.tunnel["ssh_host"], f"cat {shlex.quote(posixpath.join(remote_dir, path))}",
                    timeout_sec=REMOTE_ENGINE_TRANSFER_SEC, sink=sink,
                    what=f"Reading the run's {path} on {self.tunnel['connection']}")
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def serves(self, method: str, path: str) -> bool:
        """Whether this engine's implemented-route catalog names ``method path`` (``GET /v2/operations``)."""
        return any(str(row.get("method") or "").upper() == method and row.get("path") == path
                   for row in self.operations())

    def task_group_members(self, group_id: str, *, expected_store: Any = "") -> Tuple[List[str], Dict[str, Any]]:
        """``(identities, facts)`` of the task group ``group_id`` as this engine's store lists it now: each member's
        run id, or its job id while it is queued without one (the engine's run routes answer either), and in
        ``facts`` how many members named neither and whether the read stopped at its page bound. The engine lists
        its currently served generations only: an empty or partial answer is never proof that nothing started. It
        refuses another store (``expectedStoreId``) rather than answer for it."""
        from urllib.parse import quote, urlencode

        found, nameless, cursor, complete = [], 0, "", False
        for _page in range(50):
            query = {"limit": 200, **({"expectedStoreId": str(expected_store)} if expected_store else {}),
                     **({"cursor": cursor} if cursor else {})}
            body = self._request("GET", f"/v2/task-groups/{quote(str(group_id), safe='')}/runs?{urlencode(query)}")
            body = body if isinstance(body, dict) else {}
            if expected_store and str(body.get("storeId") or "") != str(expected_store):
                raise _unavailable("remote_engine_changed", "The task group answer came from another store than the "
                                   "one this start recorded; it speaks for nothing here.")
            for row in body.get("members") or []:
                identity = str((row or {}).get("runId") or (row or {}).get("jobId") or "") if isinstance(row, dict) else ""
                if identity:
                    found.append(identity)
                else:
                    nameless += 1
            cursor = str(body.get("nextCursor") or "")
            if not body.get("hasMore") or not cursor:
                complete = True
                break
        facts = {**({"members_without_identity": nameless} if nameless else {}),
                 **({} if complete else {"read_bound_reached": True})}
        return list(dict.fromkeys(found)), facts

    def list_runs(self, *, pages: int = 1) -> List[Dict[str, Any]]:
        """Newest-first run summaries (``GET /v2/runs``), at most ``pages`` pages of 200."""
        from urllib.parse import urlencode

        found, cursor = [], ""
        for _page in range(max(1, pages)):
            body = self._request("GET", "/v2/runs?" + urlencode({"limit": 200, **({"cursor": cursor} if cursor else {})}))
            body = body if isinstance(body, dict) else {}
            found += [row for row in body.get("runs") or [] if isinstance(row, dict)]
            cursor = str(body.get("nextCursor") or "")
            if not cursor:
                break
        return found

    def start_run(self, request: Dict[str, Any], *, idempotency_key: str = "") -> Dict[str, Any]:
        """Stage, then POST. Every refusal before the POST carries ``unsent`` (nothing reached the engine's run
        routes, so a caller that is not replaying may close the start as definite), and a host folder this call
        created for it is removed again; a refusal of the POST itself is the engine's answer, as for any engine. When
        that answer is definite (a 4xx: no run started) the folder this call created goes too; an unanswered POST, a
        5xx and a key the engine already holds (``idempotency_conflict``) keep it, since a run may use it."""
        from ouroboros.remote_workspace import remove_unstarted_folder

        staged: Dict[str, Any] = {}
        try:
            self._record_phase(idempotency_key, "staging")  # durable before anything reaches the host, or nothing does
            staged = self._prepare_start(request, idempotency_key)
            if staged.get("access") == "full":
                super().ensure_full_access(str((staged.get("scope") or {}).get("root") or ""))
            self._record_phase(idempotency_key, "posting")  # durable before the POST leaves, or no POST
        except ClaudexorUnavailable as exc:
            exc.unsent = True  # type: ignore[attr-defined]
            removed = remove_unstarted_folder(self.tunnel, staged) if staged else ""
            if removed:
                exc.args = (f"{exc}{removed}",)
            raise
        try:  # a ``_``-prefixed key is the host's own (``_staged_folder``), never the engine's
            handle = super().start_run({key: value for key, value in staged.items() if not str(key).startswith("_")},
                                       idempotency_key=idempotency_key)
        except ClaudexorUnavailable as exc:
            if 400 <= exc.status_code < 500 and exc.code != "idempotency_conflict":
                if removed := remove_unstarted_folder(self.tunnel, staged):
                    exc.args = (f"{exc}{removed}",)
            raise
        if isinstance(handle, dict) and "runDir" in handle:
            handle = {**{k: v for k, v in handle.items() if k != "runDir"}, "remoteRunDir": handle["runDir"]}
        return handle

    def _record_phase(self, invocation_id: str, phase: str) -> None:
        """The invocation's local phase fact (``REMOTE_START_PHASE``) in the custody root its caller bound
        (``custody_root``; the installation's data root when unbound): ``staging`` before anything reaches the host,
        ``posting`` just before the POST. Both land in one root, so a start whose facts were written elsewhere reads
        as unknown there, never as unsent. A fact that did not land stops the start: nothing more is sent."""
        from ouroboros import config
        from ouroboros import delegate_custody as custody

        if not invocation_id:
            return  # an unkeyed start is refused by its staging (``remote_staging_unkeyed``)
        if not custody.emit(getattr(self, "custody_root", None) or config.DATA_DIR, custody.REMOTE_START_PHASE, {
                "run_id": "", "invocation_id": str(invocation_id), "phase": phase,
                "connection": str(self.tunnel.get("connection") or "")}):
            raise _unavailable("remote_start_phase_unwritten", f"This remote start's {phase} fact could not be saved "
                               "here, so nothing more was sent.")

    def create_thread(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        """Not offered on a remote engine: a thread owns its project's address, which a staged copy cannot give it,
        and its turns cannot carry the task group a lost answer is looked for by. Plan turns and plan reviews refuse
        here, by name, before anything is sent."""
        raise _unavailable("remote_thread_unsupported", "Plan turns and other thread sessions are not available on a "
                           "remote engine yet; choose a local source for plan work. Nothing was sent.")

    start_thread_turn = create_thread

    def lookup_run(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        """The engine's body lookup is never asked of a remote engine: it would send the body this computer recorded,
        while the engine accepted the copy staged on its host (other roots, an appended host path map, its
        ``taskGroupId``). A remote start's accepted-start read (``delegate_accepted.locate``) is its task group and run
        folder on its recorded store instead (``remote_stops.locate_started``); this refusal answers any direct ask."""
        raise ClaudexorUnavailable("remote_lookup_unsupported", "A remote engine's start is not looked up by the body "
                                   "this computer recorded; nothing was read or sent.", status_code=501)

    lookup_thread_turn = lookup_run

    def _prepare_start(self, request: Dict[str, Any], idempotency_key: str) -> Dict[str, Any]:
        """The request this engine can run (``remote_workspace.staged_request``), with ``taskGroupId`` when the
        start is one-shot: the engine's task-group launch metadata refuses a thread turn, a delegated run and a
        ``kind``ed start, so those never carry one."""
        from ouroboros.delegate_continuation import unreturned_predecessor
        from ouroboros.remote_stops import TASK_GROUPS_ROUTE, task_group_id
        from ouroboros.remote_workspace import staged_request

        if request.get("_thread_id") or request.get("threadId"):
            self.create_thread()  # refuses by name
        execution = request.get("execution") if isinstance(request.get("execution"), dict) else {}
        workspace = str((self.engine_ref or {}).get("workspace") or "")
        if execution.get("workspaceKind") == "directory" and (not workspace or execution.get("isolation") != "live"):
            # The engine's own folder workspace applies, discards or edits directly on ITS host: a staged copy's
            # result would never come back here, and in a declared workspace only direct work is what was chosen.
            raise _unavailable("remote_directory_unsupported", "A folder run (not a Git project) cannot run on a "
                               "remote engine from a copy: its edits and its Apply would stay in the copy on that "
                               "host and never reach this computer. Choose a Git project, or one of the engine's "
                               "declared remote workspaces to work there directly; nothing was sent.")
        if unreturned := unreturned_predecessor(getattr(self, "custody_root", None), request):
            raise _unavailable("continuation_remote_result_unread", unreturned)
        one_shot = not any(key in request for key in ("threadId", "delegatedFromRunId", "kind", "_thread_id"))
        grouped = bool(one_shot and idempotency_key and self.serves(*TASK_GROUPS_ROUTE))
        if not grouped and workspace:  # a staged copy is also found by its run folder; a workspace run is not
            raise _unavailable("remote_engine_too_old" if one_shot else "remote_workspace_thread_unsupported",
                               ("This remote engine cannot find a start again by its task group" if one_shot else
                                "A thread turn or delegated start cannot carry the task group a lost answer is "
                                "looked for by") + ", so a run in a remote workspace could not be stopped by "
                               "Emergency Stop if its answer were lost; nothing was started.")
        staged = staged_request(self, request, idempotency_key=str(idempotency_key or ""))
        if grouped:  # saved before the POST: an answer lost to Emergency Stop is looked for on this store, never re-sent
            staged["taskGroupId"] = task_group_id(str(idempotency_key))
        return staged

    def ensure_full_access(self, root: str) -> Dict[str, Any]:
        """Deferred: the grant is made for the run's staged copy when the run starts (``start_run``)."""
        return {"repoRoot": str(root), "allowFullAccess": True, "deferred": "remote_staged_copy"}

    def register_project(self, root: str) -> str:
        raise _unavailable("remote_registration_unsupported", "A remote engine runs only staged copies or declared "
                           "remote workspaces; a local root is never registered there.")

    def project_registration(self, root: str) -> Tuple[str, Optional[bool]]:
        return self.register_project(root), None

    def find_project_id(self, root: str) -> str:
        return ""


def gateway(name: str, *, new_work: bool = True, timeout_sec: Optional[float] = None,
            settings: Optional[Mapping[str, Any]] = None, attach: bool = True) -> RemoteEngineGateway:
    """An authenticated gateway to the remote engine ``name`` (attaching when needed), its handshake done."""
    row = engine_connection(name, settings, new_work=new_work)
    client = RemoteEngineGateway({**tunnel(name, new_work=new_work, settings=settings, attach=attach),
                                  "remote_root": row["remote_root"]})
    try:
        client.handshake(**({"timeout_sec": timeout_sec} if timeout_sec is not None else {}))
        if client.engine_ref is None:
            raise _unavailable("remote_engine_identity_unproven", f"The engine on {name} did not prove its store.")
    except BaseException:
        client.close()
        raise
    return client


def bound_gateway(ref: Mapping[str, Any], *, timeout_sec: Optional[float] = None,
                  attach: bool = True) -> RemoteEngineGateway:
    """The gateway of the remote engine a record is bound to: the same connection AND its recorded SSH address,
    engine home and store. Another connection proving the same store is not a substitute; a re-pointed row refuses."""
    name = str(ref.get("connection") or "")
    row = engine_connection(name, new_work=False)
    for key in ("ssh_host", "engine_home"):
        if ref.get(key) and row[key] != ref[key]:
            raise _unavailable("remote_engine_changed", f"The engine connection {name!r} now names another {key} than "
                               "the one this work was recorded on; the work is kept, not moved.")
    client = gateway(name, new_work=False, timeout_sec=timeout_sec, attach=attach)
    if ref.get("store") and client.store_identity != ref["store"]:
        client.close()
        raise _unavailable("remote_engine_changed", f"The engine behind {name!r} serves another store than the one "
                           "this work was recorded in; its answers speak only for that store, so nothing was done.")
    kept = {key: str(ref[key]) for key in ("remote_root", "workspace") if ref.get(key)}
    if kept:  # a recorded staging root or remote workspace stays the run's, whatever the row says now
        client.tunnel.update(kept)
        client.engine_ref = {**(client.engine_ref or {}), **kept}
    return client


__all__ = [
    "CONNECTION_NAME", "DEFAULT_ENGINE_HOME", "ENGINE_CONNECTIONS_KEY", "REMOTE_ENGINE", "RemoteEngineGateway",
    "bound_gateway", "check", "engine_connection", "engine_connection_rows", "fact_mirror_root", "gateway",
    "normalize_engine_connections", "panic_close", "published", "remote_path_error", "remote_ref", "remote_shell_path",
    "retire", "run_ssh", "status", "tunnel", "tunnel_purpose",
]
