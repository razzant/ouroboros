"""Kernel-state readers for work scopes: the marker grammar, one census, typed identities.

Every process that exists because of work has one custodian that can see it and end it
by measured identity (DEVELOPMENT "Process Custody Rule"). What that custodian reads here
is live kernel state, never what a command line looks like:

* the scope MARKER, an environment entry ``NAME=1`` whose name carries a random 128-bit
  hex id (``OURO_INSTALL_SCOPE_<hex>`` for the installation, ``OURO_TASK_SCOPE_<hex>`` for
  a task; the engine's ``CLAUDEXOR_RUN_SCOPE_<hex>`` is recognised for shedding and read
  from the engine's own index at Panic). Inherited by descendants; Linux reads exact
  NUL-separated ``/proc/<pid>/environ`` entries, but nondumpable/setuid peers may hide them;
  ``OURO_LAUNCH_SCOPE_<hex>`` names one service launch, whose own stop alone reads it;
  on macOS only for non-platform binaries — ``ps -E`` shows no environment for Apple's
  own ``/bin/sh``, ``/bin/sleep``, ``/usr/bin/find`` under SIP, so the marker alone is
  blind to exactly a ``find`` orphaned under a killed shell. ``ps -E`` also prints that
  environment after the argv in ONE text column, matched word by word: a process whose
  argv (or another variable's value) holds the exact word ``NAME=1`` reads as a member
  and is signalled with the scope. Only the random 128-bit name keeps a stranger out —
  practically, not structurally (a disclosed residual);
* each process's pid, ppid, pgid, state and BIRTH: the C-locale ``ps -o lstart`` text,
  spelled from a fixed English table so a Linux ``/proc`` birth compares equal to the
  engine's recorded text whatever the server's locale — the identity a recorded group or
  descendant is re-measured against before any signal. Births compare in ONE spelling,
  ``canonical_birth`` (whitespace collapsed): ``ps``/``ctime`` pad a one-digit day
  (``Oct  1``), the engine's index writes ``Oct 1``, old records keep either. And in ONE
  zone, the one a ``ps`` without ``TZ`` prints (``_birth_zone``): this server's own ``TZ``
  never respells a birth, and a zone it cannot read leaves births unreadable. Residuals, a
  zone-less text's: a system zone change between record and check respells every recorded
  birth, so a live recorded process reads as another one (a zone-free birth needs the
  engine's schema); a DST fall-back hour spells two instants alike; Linux ``btime`` moves
  with clock steps; records written under another effective ``TZ`` read as other processes
  (no migration: the feature is unpublished); the engine's own floors parse in its daemon's ``TZ``.

Every answer about one process is typed: ``alive`` or ``zombie`` with birth and pgid,
``absent`` (ESRCH) or ``unreadable`` (a failed or stalled reader, or a Linux ``/proc`` row
with no readable kernel boot time or zone to spell its birth in); unreadable is never read
as gone. Linux reads ``/proc`` (the documented Docker image ships no ``ps``); macOS reads
ONE ``ps -A`` for the table (``-E`` only when a marker is asked for) and ONE ``ps -p a,b,c``
per identity batch, both under the C locale without ``TZ`` and with every scope marker shed,
so the reader never lists itself. Environment bytes
are never logged or persisted. Windows has no environment census. Membership, the reap and
the durable records live in ``process_scope``.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import secrets
import subprocess
import time
import zoneinfo
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros import platform_layer as _pl

SCOPE_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*_SCOPE_[0-9a-f]{32}$")
SCOPE_VALUE = "1"
INSTALL_SCOPE_PREFIX = "OURO_INSTALL_SCOPE_"
TASK_SCOPE_PREFIX = "OURO_TASK_SCOPE_"
LAUNCH_SCOPE_PREFIX = "OURO_LAUNCH_SCOPE_"  # one service launch's own: its stop's reach, never a task lifetime
RUN_SCOPE_PREFIX = "CLAUDEXOR_RUN_SCOPE_"

_CENSUS_TIMEOUT_SEC = 20
_BIRTH_TIMEOUT_SEC = 5

# Typed per-member answers.
IDENTITY_ALIVE = "alive"
IDENTITY_ZOMBIE = "zombie"
IDENTITY_ABSENT = "absent"
IDENTITY_UNREADABLE = "unreadable"


# --------------------------------------------------------------------------- markers

def mint_marker(prefix: str) -> str:
    """A fresh grammar-shaped scope name: ``<PREFIX><32 random hex digits>``."""
    name = f"{prefix}{secrets.token_hex(16)}"
    if not SCOPE_NAME_RE.match(name):
        raise ValueError(f"scope prefix {prefix!r} does not fit the marker grammar")
    return name


def is_scope_name(name: str) -> bool:
    return bool(SCOPE_NAME_RE.match(str(name or "")))


def is_launch_marker(name: str) -> bool:
    """One service launch's own marker, never a task's or the installation's."""
    return is_scope_name(name) and str(name).startswith(LAUNCH_SCOPE_PREFIX)


def scope_entries(env: Mapping[str, str], *, prefix: str = "") -> Dict[str, str]:
    """The grammar-shaped ``NAME=1`` entries of ``env`` (optionally one role's)."""
    return {key: SCOPE_VALUE for key, value in env.items()
            if is_scope_name(key) and str(value) == SCOPE_VALUE and key.startswith(prefix)}


def shed_scopes(env: Mapping[str, str], *, keep_prefixes: Iterable[str] = (INSTALL_SCOPE_PREFIX,)) -> Dict[str, str]:
    """``env`` without the scope markers a longer-lived custodian must not inherit.

    The managed daemon and a keep/session service keep the installation marker and
    drop task and engine-run markers: inheriting one would end them with the first
    task or run that happened to start them.
    """
    kept = tuple(keep_prefixes)
    return {key: value for key, value in env.items()
            if not is_scope_name(key) or (bool(kept) and key.startswith(kept))}


def installation_scope_entries(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The installation marker entries present in ``env`` (default: this process)."""
    return scope_entries(os.environ if env is None else env, prefix=INSTALL_SCOPE_PREFIX)

# --------------------------------------------------------------------------- identity and census

def canonical_birth(text: object) -> str:
    """The one comparable spelling of a birth: whitespace runs collapsed, as the engine's
    index writes it. Both spellings of a one-digit day name one instant; every typed answer
    (``_Row``, ``_Identity``) carries this form, and a recorded birth is compared,
    de-duplicated and saved anew in it (an old record keeps its bytes until a pass rewrites it)."""
    return " ".join(str(text or "").split())


@dataclass
class _Row:
    pid: int
    ppid: int
    pgid: int
    birth: str          # the C-locale ``ps -o lstart`` text, canonical, compared for equality
    epoch: float        # the same instant as seconds, for "born after the recorded leader"
    state: str = IDENTITY_ALIVE
    markers: Optional[set] = field(default_factory=set)  # None: environment unreadable

    def __post_init__(self) -> None:
        self.birth = canonical_birth(self.birth)


@dataclass
class _Identity:
    state: str          # alive | zombie | absent | unreadable
    birth: str = ""     # canonical, like ``_Row.birth``
    pgid: int = 0

    def __post_init__(self) -> None:
        self.birth = canonical_birth(self.birth)


# The C-locale spelling ``ps -o lstart`` and procps's ``ctime`` print: fixed tables, so the
# server's own LC_TIME can never change how a birth is spelled or compared.
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTHS = {name: index for index, name in enumerate(_MONTH_NAMES, start=1)}
_LSTART = r"[A-Za-z]{3} [A-Za-z]{3} +\d{1,2} \d\d:\d\d:\d\d \d{4}"
_LSTART_RE = re.compile(r"^[A-Za-z]{3} ([A-Za-z]{3}) +(\d{1,2}) (\d\d):(\d\d):(\d\d) (\d{4})$")
_CENSUS_ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(" + _LSTART + r")\s+(\S+)\s*(.*)$")
_IDENTITY_ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(" + _LSTART + r")\s+(\S+)\s*$")


def _ps_env() -> Dict[str, str]:
    # English month/day names whatever the server's locale (the parser has no locale
    # tables); no ``TZ``, like the engine's own ps env, so both spell a birth in the one OS
    # zone (``_birth_zone``); the census's own ps child carries NO scope marker, so it never lists itself.
    env = {key: value for key, value in os.environ.items() if key != "TZ"}
    return shed_scopes({**env, "LC_ALL": "C", "LANG": "C"}, keep_prefixes=())


_BIRTH_ZONE: List[Any] = []


def _birth_zone() -> Any:
    """The zone a birth is spelled and parsed in: the one a ``ps`` without ``TZ`` prints (the
    engine's; ours sheds it, ``_ps_env``). With no ``TZ`` key in this server's environment that
    is libc local time (``None``); with one, empty included, the OS zone read from
    ``/etc/localtime``, cached once proven. ``False`` while that cannot be read: no birth is
    spelled or parsed then, never a guessed UTC, local or zero instant (a failure is retried)."""
    if "TZ" not in os.environ:
        return None
    if not _BIRTH_ZONE:
        try:
            with open("/etc/localtime", "rb") as handle:
                _BIRTH_ZONE.append(zoneinfo.ZoneInfo.from_file(handle, key="localtime"))
        except Exception:
            return False
    return _BIRTH_ZONE[0]


def birth_epoch(text: str) -> float:
    """Seconds since the epoch for one ``lstart`` text in the birth zone; 0.0 when it does not
    parse or the zone is unreadable (a floor recorded but unreadable is never replaced by a
    weaker one: ``process_scope.scope_members``)."""
    match = _LSTART_RE.match(canonical_birth(text))
    zone = _birth_zone() if match else False
    if zone is False:
        return 0.0
    month, day, hour, minute, second, year = match.groups()
    try:
        return _dt.datetime(int(year), _MONTHS[month], int(day), int(hour), int(minute), int(second),
                            tzinfo=zone).timestamp()
    except (KeyError, ValueError, OverflowError):
        return 0.0


def lstart_text(epoch: float) -> str:
    """The C-locale ``ps -o lstart`` spelling of ``epoch`` (procps prints ``ctime`` of
    ``btime + starttime // CLK_TCK``) from the fixed English tables, so a Linux birth read
    from ``/proc`` compares equal (``canonical_birth``) to the engine's recorded text under
    any LC_TIME, in the birth zone; ``""`` while that zone is unreadable (``_birth_zone``)."""
    zone = _birth_zone()
    if zone is False:
        return ""
    when = time.localtime(int(epoch)) if zone is None else _dt.datetime.fromtimestamp(int(epoch), zone).timetuple()
    return (f"{_WEEKDAYS[when.tm_wday]} {_MONTH_NAMES[when.tm_mon - 1]} {when.tm_mday:>2} "
            f"{when.tm_hour:02}:{when.tm_min:02}:{when.tm_sec:02} {when.tm_year}")


def _state_letter(text: str) -> str:
    return IDENTITY_ZOMBIE if str(text or "").startswith("Z") else IDENTITY_ALIVE


# --- Linux: /proc needs no ps (the documented Docker image ships none)

_BOOT_EPOCH: List[float] = []


def _boot_epoch() -> Optional[float]:
    """The kernel's ``btime`` from ``/proc/stat``, cached once read. ``None`` when the file is
    unreadable or holds no positive integer ``btime``: never a zero that would spell a live
    process's birth as 1970 and read it as another process. A failure is not cached."""
    if not _BOOT_EPOCH:
        try:
            with open("/proc/stat", "rb") as handle:
                for line in handle:
                    if line.startswith(b"btime "):
                        if (boot := int(line.split()[1])) > 0:
                            _BOOT_EPOCH.append(float(boot))
                        break
        except (OSError, ValueError, IndexError):
            pass
    return _BOOT_EPOCH[0] if _BOOT_EPOCH else None


def parse_proc_stat(data: bytes, *, boot_epoch: float, hz: int) -> Tuple[str, int, int, float]:
    """``(state, ppid, pgid, birth_epoch)`` from one ``/proc/<pid>/stat`` body (comm is
    parenthesised and may hold spaces or ``)``, so fields follow the LAST ``)``)."""
    fields = data.rpartition(b")")[2].split()
    state = fields[0].decode("ascii", "replace")
    ppid, pgid, start_ticks = int(fields[1]), int(fields[2]), int(fields[19])
    return state, ppid, pgid, float(int(boot_epoch) + start_ticks // max(1, int(hz)))


def _proc_identity(pid: int) -> _Identity:
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as handle:
            data = handle.read()
    except FileNotFoundError:
        return _Identity(IDENTITY_ABSENT)
    except ProcessLookupError:
        return _Identity(IDENTITY_ABSENT)
    except OSError:
        return _Identity(IDENTITY_UNREADABLE)
    boot = _boot_epoch()
    if boot is None:
        return _Identity(IDENTITY_UNREADABLE)  # present, but no birth can be spelled: never a guessed one
    try:
        state, _ppid, pgid, epoch = parse_proc_stat(data, boot_epoch=boot, hz=os.sysconf("SC_CLK_TCK"))
    except (ValueError, IndexError, OSError):
        return _Identity(IDENTITY_UNREADABLE)
    birth = lstart_text(epoch)
    return _Identity(_state_letter(state), birth, pgid) if birth else _Identity(IDENTITY_UNREADABLE)


def _proc_rows(assignments: set, wanted: set) -> Optional[Dict[int, _Row]]:
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    boot = _boot_epoch()
    if boot is None or _birth_zone() is False:
        return None  # no boot time or zone, no comparable birth: the table is unknown, never guessed
    hz = os.sysconf("SC_CLK_TCK")
    rows: Dict[int, _Row] = {}
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(f"/proc/{pid}/stat", "rb") as handle:
                state, ppid, pgid, epoch = parse_proc_stat(handle.read(), boot_epoch=boot, hz=hz)
        except (OSError, ValueError, IndexError):
            continue  # gone between listdir and read, or unreadable: not a row we can name
        rows[pid] = _Row(pid, ppid, pgid, lstart_text(epoch), epoch, _state_letter(state),
                         _proc_markers(pid, assignments, wanted))
    return rows or None


def _proc_markers(pid: int, assignments: set, wanted: set) -> Optional[set]:
    if not assignments:
        return set()
    try:
        with open(f"/proc/{pid}/environ", "rb") as handle:
            entries = set(handle.read().split(b"\0"))
    except OSError:
        return None  # gone, nondumpable or another user's: unknown, not "no"
    present = {entry.decode("utf-8", "replace") for entry in entries}
    return {marker for marker in wanted if f"{marker}={SCOPE_VALUE}" in present}


# --- macOS/BSD: ps is the only kernel-state reader

def _ps_identities(pids: List[int], timeout: float = _BIRTH_TIMEOUT_SEC) -> Dict[int, _Identity]:
    try:
        out = subprocess.run(["ps", "-o", "pid=,pgid=,lstart=,state=", "-p", ",".join(str(pid) for pid in pids)],
                             capture_output=True, text=True, errors="replace", timeout=timeout,
                             env=_ps_env())
    except Exception:
        return {pid: _Identity(IDENTITY_UNREADABLE) for pid in pids}
    # macOS ``ps -p`` exits 1 whenever ANY listed pid is missing (the others' rows still
    # print, and the missing are answered absent below); anything else is no answer.
    if out.returncode not in (0, 1):
        return {pid: _Identity(IDENTITY_UNREADABLE) for pid in pids}
    found: Dict[int, _Identity] = {}
    for line in (out.stdout or "").splitlines():
        match = _IDENTITY_ROW_RE.match(line)
        if match:
            pid, pgid, birth, state = match.groups()
            found[int(pid)] = _Identity(_state_letter(state), birth.strip(), int(pgid))
    return {pid: found.get(pid, _Identity(IDENTITY_ABSENT)) for pid in pids}


def read_identities(pids: Iterable[int], *, timeout: Optional[float] = None) -> Dict[int, _Identity]:
    """One batched identity read (``alive``/``zombie`` with birth and pgid, ``absent`` or
    ``unreadable``) for ``pids``: Linux from ``/proc``, macOS from ONE ``ps -p a,b,c`` (``timeout``
    bounds it; its own bound otherwise)."""
    wanted = sorted({int(pid) for pid in pids if int(pid) > 0})
    if not wanted:
        return {}
    if _pl.IS_WINDOWS:
        return {pid: _Identity(IDENTITY_UNREADABLE) for pid in wanted}
    if os.path.isdir("/proc"):
        return {pid: _proc_identity(pid) for pid in wanted}
    return _ps_identities(wanted, _BIRTH_TIMEOUT_SEC if timeout is None else max(0.0, timeout))


def measure_birth(pid: int) -> str:
    """The LIVE birth text of ``pid``: ``""`` when gone, a zombie, or unmeasurable (a
    liveness answer for callers that only need "is this exact process still running")."""
    identity = read_identities([pid]).get(int(pid))
    return identity.birth if identity is not None and identity.state == IDENTITY_ALIVE else ""


def birth_text(pid: int) -> str:
    """The birth text as the table still shows it, a zombie's included (``""`` when gone
    or unmeasurable): the identity a group RECORD needs for a leader that already exited."""
    identity = read_identities([pid]).get(int(pid))
    return identity.birth if identity is not None and identity.state in (IDENTITY_ALIVE, IDENTITY_ZOMBIE) else ""


def census(markers: Iterable[str] = (), *, env_rows: Any = None,
           timeout: float = _CENSUS_TIMEOUT_SEC) -> Optional[Dict[int, _Row]]:
    """One snapshot of the live process table: pid, ppid, pgid, birth, state and which
    of ``markers`` each process visibly carries. ``None`` when the table could not be
    read (unknown, never empty). Linux reads ``/proc`` (no ``ps`` needed); macOS reads
    the one ``ps -E`` column that mixes argv and environment (module docstring: residual).
    Without markers no environment is read on either system: on macOS that drops ``-E``,
    whose per-process argument-area read dominates the table's cost on a busy host.
    ``env_rows(rows)`` (macOS) names, from the whole table, the only pids whose environment
    a marker read covers, in one ``ps -E -p`` batch; the others report ``markers=None`` (not
    read). Linux reads every ``/proc/<pid>/environ`` as before: a subreaper can adopt an
    orphan there. ``timeout`` bounds the whole read, both ``ps`` of a targeted one (a caller
    inside a shutdown's grace passes its share); expiring is a failed read."""
    if _pl.IS_WINDOWS:
        return None
    wanted = {str(marker) for marker in markers if marker}
    assignments = {f"{marker}={SCOPE_VALUE}" for marker in wanted}
    if os.path.isdir("/proc"):
        return _proc_rows(assignments, wanted)
    if _birth_zone() is False:
        return None  # each row carries the instant its birth names: unknown without the zone
    targeted, began = bool(wanted) and env_rows is not None, time.monotonic()
    columns = ["-E", "-ww", "-A", "-o", "pid=,ppid=,pgid=,lstart=,state=,command="] if wanted and not targeted else [
        "-A", "-o", "pid=,ppid=,pgid=,lstart=,state="]
    try:
        out = subprocess.run(["ps", *columns],
                             capture_output=True, text=True, errors="replace",
                             timeout=timeout, env=_ps_env())
    except Exception:
        return None
    if out.returncode != 0:
        return None
    rows: Dict[int, _Row] = {}
    for line in (out.stdout or "").splitlines():
        match = _CENSUS_ROW_RE.match(line)
        if not match:
            continue
        pid, ppid, pgid, birth, state, rest = match.groups()
        words = set(rest.split())
        rows[int(pid)] = _Row(int(pid), int(ppid), int(pgid), birth.strip(), birth_epoch(birth),
                              _state_letter(state), {marker for marker in wanted if f"{marker}={SCOPE_VALUE}" in words})
    if targeted and rows:
        chosen = sorted({int(pid) for pid in env_rows(rows)} & set(rows))
        left = timeout - (time.monotonic() - began)
        found = (_ps_markers(chosen, wanted, left) if left > 0 else None) if chosen else {}
        if found is None:
            return None  # the environments asked for could not be read: the table is unknown, never clean
        for pid, row in rows.items():
            row.markers = found.get(pid, set()) if pid in found else None
    return rows or None


def _ps_markers(pids: List[int], wanted: set, timeout: float = _CENSUS_TIMEOUT_SEC) -> Optional[Dict[int, set]]:
    """``{pid: markers}`` for the listed pids from one ``ps -E -p`` read (macOS); a pid the read
    no longer shows is absent from the answer; ``None`` when the read itself failed."""
    try:
        out = subprocess.run(["ps", "-E", "-ww", "-o", "pid=,command=", "-p", ",".join(str(pid) for pid in pids)],
                             capture_output=True, text=True, errors="replace", timeout=timeout,
                             env=_ps_env())
    except Exception:
        return None
    if out.returncode not in (0, 1):  # 1: some listed pid is gone, the others still print
        return None
    found: Dict[int, set] = {}
    for line in (out.stdout or "").splitlines():
        head, _, rest = line.strip().partition(" ")
        if head.isdigit():
            words = set(rest.split())
            found[int(head)] = {marker for marker in wanted if f"{marker}={SCOPE_VALUE}" in words}
    return found


def leader_facts(pid: int) -> Tuple[int, str]:
    """``(pgid, birth)`` of a just-spawned session leader, zombie-visible: the shell may
    have exited before its row is written, and ``os.getpgid`` refuses a corpse (ESRCH on
    macOS). Every seam spawns with ``start_new_session``, so a leader the table no longer
    shows at all still led the group that bears its pid (structural); its birth is then
    unrecorded, which rule 2 treats as "cannot vouch for a live leader"."""
    identity = read_identities([pid]).get(int(pid))
    if identity is not None and identity.state in (IDENTITY_ALIVE, IDENTITY_ZOMBIE):
        return (identity.pgid or int(pid)), identity.birth
    return int(pid), ""


__all__ = [
    "IDENTITY_ABSENT", "IDENTITY_ALIVE", "IDENTITY_UNREADABLE", "IDENTITY_ZOMBIE", "INSTALL_SCOPE_PREFIX",
    "LAUNCH_SCOPE_PREFIX", "RUN_SCOPE_PREFIX", "SCOPE_NAME_RE", "SCOPE_VALUE", "TASK_SCOPE_PREFIX", "birth_epoch",
    "birth_text",
    "canonical_birth", "census", "installation_scope_entries", "is_launch_marker", "is_scope_name", "leader_facts",
    "lstart_text",
    "measure_birth", "mint_marker", "parse_proc_stat", "read_identities", "scope_entries", "shed_scopes",
]
