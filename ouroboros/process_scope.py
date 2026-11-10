"""Work-scoped process lifetime: membership, one reap, task scopes, Panic, the engine index.

The custodian's view of a scope on POSIX is the union of three sources, each read from
live kernel state by ``process_census``: the scope MARKER (an environment entry inherited
by every descendant, invisible for macOS platform binaries under SIP); the scope's
RECORDED process groups — every own command, executor foreground command and task-scoped
service is its own group leader (``start_new_session``), recorded at spawn with the
leader's measured birth (zombie-visible: the shell may have exited before its row is
written), so a ``cmd &`` or ``nohup`` child stays reachable after its shell exits; a
recorded group is signalled MEMBER BY MEMBER, never ``killpg`` on a bare number, and only
under the pgid-reuse rule (every candidate born at or after the recorded leader, a live
``pid == G`` carrying the recorded leader birth; otherwise the id was reused and the group
is skipped entirely; disclosed residual: a group that emptied and whose id was recycled as
a new, leaderless group born after the recorded leader passes both rules; a recorded leader
the table omits is read by name, and an unreadable one keeps its group owed, never gone).
Tracked commands retain their waitable leader until whole-tree retirement, preventing reuse
while their custodian lives; and
individually recorded descendants — the engine's captured pids, and every KNOWN member an
earlier pass left unverified or surviving, each checked by pid AND birth.

Every answer about one process is typed: confirmed absent or reused (ESRCH, a corpse, or
a measured birth that differs), ended by our signal, a survivor, or UNVERIFIED (identity
unreadable, a stalled reader, a denied or failed signal); unverified is never ended or
gone. A pending obligation (a task scope record, ``panic_sweep_pending``) retires only on a
recorded outcome: settled for its requested scope, or unsupported coverage. Discovery
must be readable when requested; only installation-wide Panic requires the engine index
absent or read. Ordinary startup retries a pending Panic's saved identities without either discovery source.
Survivors and undecidable identities remain owed; a denied signal (EPERM) is disclosed as
final. Remaining known members stay recorded by pid and birth for the next pass. Inheritance is not a lifetime
transfer: a process handed to a longer-lived custodian sheds every shorter-lived marker at
its spawn env (``shed_scopes``). Environment bytes are never logged or persisted. Windows:
``end_scope`` reports ``unsupported`` as a final disclosure; the Job/TerminateProcess paths
stay the truth.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import logging
import os
import pathlib
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ouroboros import platform_layer as _pl
from ouroboros.process_census import (
    IDENTITY_ABSENT,
    IDENTITY_ALIVE,
    IDENTITY_UNREADABLE,
    IDENTITY_ZOMBIE,
    INSTALL_SCOPE_PREFIX,
    RUN_SCOPE_PREFIX,
    SCOPE_NAME_RE,
    SCOPE_VALUE,
    TASK_SCOPE_PREFIX,
    _Identity,
    _Row,
    birth_epoch,
    birth_text,
    canonical_birth,
    census,
    installation_scope_entries,
    is_scope_name,
    leader_facts,
    lstart_text,
    measure_birth,
    mint_marker,
    parse_proc_stat,
    read_identities,
    scope_entries,
    shed_scopes,
)
from ouroboros.utils import append_jsonl, atomic_write_json, read_json_dict, update_json_locked, utc_now_iso

log = logging.getLogger(__name__)


INSTALLATION_SCOPE_FILE = "installation_scope.json"
PANIC_SWEEP_PENDING_FILE = "panic_sweep_pending.json"
TASK_SCOPES_DIR = "task_scopes"
# The managed engine's durable scope index, under the installation's OWN Claudexor home
# (``claudexor_daemon.owned_config_dir()``: ``<data>/claudexor``), never another root.
ENGINE_RUN_SCOPES_FILE = pathlib.Path("daemon") / "run-scopes.json"
ENGINE_RUN_SCOPES_SCHEMA = 1
# Only these index states let a Panic pass complete; unreadable, invalid or a newer schema
# keeps the pending record while every row that was read is still acted on.
_INDEX_COMPLETE_STATUSES = ("absent", "read")

# Task-end: TERM, a bounded grace, then KILL; Panic sends KILL at once. Bounded rescans
# catch a fork between the snapshot and the signal; nothing here is a polling monitor.
TASK_END_GRACE_SEC = 2.0
_KILL_CONFIRM_SEC = 1.0
_RESCAN_SETTLE_SEC = 0.1
_CLEANUP_RUNNING_STALE_SEC = 60.0

COVERAGE_POSIX = "marker+groups"
COVERAGE_UNSUPPORTED = "unsupported"

# Why a member is unverified (``process_census`` types the identity answers themselves).
UNVERIFIED_IDENTITY = "identity_unreadable"
UNVERIFIED_DENIED = "signal_denied"
UNVERIFIED_FAILED = "signal_failed"
UNVERIFIED_CENSUS = "census_unreadable"  # the table itself could not be read: a known identity stays owed
UNVERIFIED_SPARE_UNKNOWN = "spared_roots_unknown"  # only the parent chain reached it, and daemon roots are unreadable
UNVERIFIED_UNPUBLISHED = "custody_unwritten"  # a launch's chain-only member no custody could name: nothing signalled

# The disclosure the installation census cannot resolve itself (owner decision
# 2026-10-12, accepted and disclosed): on macOS a system binary started with ``&`` by a
# shell that exited before any group or capture was recorded shows no environment and
# belongs to no recorded group; engine harness shells' leftovers are the engine's capture
# set, read from its index below when that index is available.
PANIC_CENSUS_RESIDUAL = ("marker discovery is incomplete: Linux nondumpable/setuid peers may hide environments; "
                        "macOS system binaries started with & by a shell gone before capture are not reached; "
                        "engine leftovers require a readable run-scope index")
PANIC_KNOWN_RESIDUAL = ("known identities only, not a full cleanup: old processes the last installation-wide "
                        "attempt did not reach (late descendants, cutoff or unreadable sources) are not discovered; "
                        "ending an old controller may affect new work attached to it")


# --------------------------------------------------------------------------- installation scope

def _state_dir(data_dir: Any) -> pathlib.Path:
    return pathlib.Path(data_dir) / "state"


def installation_scope_path(data_dir: Any) -> pathlib.Path:
    return _state_dir(data_dir) / INSTALLATION_SCOPE_FILE


def installation_marker(data_dir: Any, *, create: bool = True) -> str:
    """The persisted installation marker of ``data_dir``: minted once, stable across
    server generations, unique per (temporary) installation. ``""`` when absent and
    ``create`` is false."""
    path = installation_scope_path(data_dir)
    record = read_json_dict(path) or {}
    marker = str(record.get("marker") or "")
    if is_scope_name(marker) and marker.startswith(INSTALL_SCOPE_PREFIX):
        return marker
    if not create:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)

    def mint(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        existing = str((current or {}).get("marker") or "")
        if is_scope_name(existing) and existing.startswith(INSTALL_SCOPE_PREFIX):
            return None
        return {"marker": mint_marker(INSTALL_SCOPE_PREFIX), "created_at": utc_now_iso()}

    record = update_json_locked(path, mint)
    return str(record.get("marker") or "")


_PLANTED: Dict[str, Any] = {}


def plant_installation_scope(data_dir: Any, environ: Optional[Dict[str, str]] = None) -> str:
    """Put the installation marker into the server's environment BEFORE any child spawns.

    Idempotent: an exec-restarted generation already carries it. Remembers the
    launcher's pid (the parent at planting) so the Panic census can exclude it.
    """
    marker = installation_marker(data_dir, create=True)
    target = os.environ if environ is None else environ
    target[marker] = SCOPE_VALUE
    if environ is None:
        _PLANTED.update(marker=marker, launcher_pid=os.getppid(), server_pid=os.getpid())
    return marker


def panic_census_exclusions() -> set:
    """Never signalled by the installation census: this server, its launcher, pid 1."""
    return {os.getpid(), int(_PLANTED.get("launcher_pid") or os.getppid()), 1}


def initialize_installation_scope(data_dir: Any) -> str:
    """The one post-admission, pre-child scope initialization every server entry runs:
    retry only the old Panic's saved identities, then plant the marker. An old pending
    record never authorizes discovery of this generation's independent work.
    Idempotent; ``main`` and the imported ASGI ``lifespan`` share it."""
    try:
        complete_pending_panic_sweep(data_dir, exclude={os.getpid(), os.getppid(), 1})
    except Exception:
        log.warning("pending Panic scope sweep failed; its record stays for the next start", exc_info=True)
    return plant_installation_scope(data_dir)


# --------------------------------------------------------------------------- membership

def scope_members(rows: Mapping[int, _Row], *, marker: str = "", markers: Iterable[str] = (),
                  groups: Iterable[Mapping[str, Any]] = (), captured: Iterable[Mapping[str, Any]] = (),
                  scope_start_epoch: float = 0.0, exclude: Iterable[int] = (),
                  descendants: bool = False, marker_targets: Optional[Mapping[str, Any]] = None,
                  spared: Iterable[int] = ()) -> Dict[str, Any]:
    """The scope's live members from one snapshot: marker members ∪ members of the
    recorded groups that pass the pgid-reuse rule ∪ individually recorded descendants
    whose birth still matches (a recorded pid missing from the snapshot is read by the typed
    identity reader, never called gone; each birth recorded for that pid is judged on its
    own). A recorded group leader (pid == G with its measured birth) the snapshot omits is
    read the same way, in the same batch: unreadable leaves the group unjudged and the leader
    unverified, never gone and never a group signalled on a guess; alive with that birth it
    is a member wherever its group went, like a captured descendant. A recorded floor that
    cannot be read as an instant leaves its group unjudged too. ``descendants`` adds every
    row of the same snapshot whose parent chain reaches a member, whatever its group; an
    excluded pid is never a member, so its subtree is entered only through other members.
    Births compare canonically (``canonical_birth``). Bookkeeping for one pass that ends
    several owned processes: each member's ``origins`` are the ``target`` of every group or
    captured entry and of every marker in ``marker_targets`` that reached it (a descendant
    inherits its parent's); ``descended`` holds the members only the parent chain reached.
    A ``held`` captured identity (one a chain reached while the daemon roots were unknown) is
    ``descended`` too, and a member only while its present parent chain stays clear of every
    ``spared`` root: below one it is that root's and no longer owed; a chain the snapshot cannot
    show keeps it unverified as ``spared_roots_unknown``, never a member on a guess.
    Returns ``{"members": {pid: row}, "groups_reused": [pgid], "groups_gone": [pgid],
    "captured_gone": [{pid, birth}], "captured_unverified": [{pid, birth, reason}],
    "origins": {pid: {target}}, "descended": {pid}}`` (an unverified recorded leader or
    group included; these lists keep the entry's ``target`` when it had one)."""
    excluded = {int(pid) for pid in exclude} | {0, 1}
    spared_roots = {int(pid) for pid in spared}
    names = {str(name) for name in [marker, *markers] if name}
    by_marker = dict(marker_targets or {})
    members: Dict[int, _Row] = {}
    origins: Dict[int, set] = {}

    def tag(item: Dict[str, Any], target: Any) -> Dict[str, Any]:  # an untagged entry keeps its shape
        return item if target is None else {**item, "target": target}

    def own(pid: int, target: Any) -> None:
        if target is not None:
            origins.setdefault(int(pid), set()).add(target)

    for row in rows.values():
        if row.pid not in excluded and row.markers and names & row.markers:
            members[row.pid] = row
            for name in names & row.markers:
                own(row.pid, by_marker.get(name))
    known = [(int(item.get("pid") or 0), canonical_birth(item.get("birth")), item.get("target"), bool(item.get("held")))
             for item in captured]
    known = [entry for entry in known if entry[0] > 1 and entry[0] not in excluded and entry[1]]
    recorded = [(int(group.get("pgid") or 0), canonical_birth(group.get("leader_birth")), group) for group in groups]
    # A pid the snapshot omits (gone, or a row a partial table could not read) is asked by
    # name in ONE batch: each recorded descendant and each recorded leader rule 2 must vouch for.
    omitted = {pgid for pgid, leader_birth, _ in recorded
               if pgid > 1 and leader_birth and pgid not in rows and pgid not in excluded}
    asked = {entry[0] for entry in known if entry[0] not in rows} | omitted
    identities = _read_identities(asked) if asked else {}
    unverified: List[Dict[str, Any]] = []
    reused: List[int] = []
    gone: List[int] = []
    for pgid, leader_birth, group in recorded:
        if pgid <= 1:
            continue
        target = group.get("target")
        candidates = [row for row in rows.values() if row.pgid == pgid]
        leader = rows.get(pgid)
        if pgid in omitted:
            # The answer stands in for the omitted row: absence leaves the group to its members
            # and floor; a live or dead pid == G is judged as its row would be.
            answer = identities.get(pgid) or _Identity(IDENTITY_UNREADABLE)
            if answer.state == IDENTITY_UNREADABLE:
                unverified.append(tag({"pid": pgid, "birth": leader_birth, "reason": UNVERIFIED_IDENTITY}, target))
                continue  # owed: the next pass judges this group again
            if answer.state != IDENTITY_ABSENT:
                leader = _Row(pgid, 0, answer.pgid, answer.birth, birth_epoch(answer.birth), answer.state, set())
                if leader.pgid == pgid:
                    candidates.append(leader)
        if (leader is not None and leader_birth and leader.birth == leader_birth
                and leader.state == IDENTITY_ALIVE and pgid not in excluded):
            members[pgid] = leader  # the recorded leader itself, whatever group it is in now: never discharged
            own(pgid, target)
        if not candidates:
            gone.append(pgid)  # an emptied group is simply gone
            continue
        # No member of a recorded group predates its leader, else the group's own recorded
        # floor (the engine's scope_start), else the scope's start; lstart has one-second
        # resolution, hence the tolerance. A floor bounds the members' age; it is never the
        # leader's identity. The first one recorded decides: one that cannot be read as an
        # instant (no proven zone, a malformed text) leaves the group owed, never a weaker floor.
        source = leader_birth or str(group.get("floor") or "")
        floor = birth_epoch(source) if source else float(scope_start_epoch or 0.0)
        if source and not floor:
            unverified.append(tag({"pid": pgid, "birth": leader_birth, "reason": UNVERIFIED_IDENTITY}, target))
            continue
        floor -= 1.0
        # A live pid == G must carry the recorded leader birth. An unrecorded leader birth
        # (the spawn-time read found the leader already reaped) cannot vouch for a live
        # leader: the kernel never hands out a pid that is still a live process-group id,
        # so a live pid == G after our leader was reaped means our group had emptied first
        # and the id was recycled — skipped, disclosed, never signalled on a guess. The
        # miss this leaves (a spawn-time identity read that failed for a leader still ours)
        # is the disclosed residual.
        if (leader is not None and (not leader_birth or leader.birth != leader_birth)) or any(
                row.epoch and row.epoch < floor for row in candidates):
            reused.append(pgid)  # the id names another group now: skipped entirely
            continue
        for row in candidates:
            if row.pid not in excluded:
                members[row.pid] = row
                own(row.pid, target)
    captured_gone: List[Dict[str, Any]] = []  # identities: a pid is resolved per recorded birth, never as a number
    missing: Dict[int, List[Tuple[str, Any, bool]]] = {}
    descended: set = set()

    def below_spared(pid: int) -> Optional[bool]:
        seen, parent = {pid}, rows[pid].ppid
        while parent > 1:
            if parent in spared_roots:
                return True
            if parent in seen or parent not in rows:
                return None  # a parent this snapshot does not show
            seen.add(parent)
            parent = rows[parent].ppid
        return False

    for pid, birth, target, held in known:
        row = rows.get(pid)
        if row is None:
            missing.setdefault(pid, []).append((birth, target, held))  # not in this snapshot: asked for by name above
        elif row.birth != birth:
            captured_gone.append(tag({"pid": pid, "birth": birth}, target))  # the pid names another process now
        elif not held or below_spared(pid) is False:
            members[pid] = row
            own(pid, target)
            if held:
                descended.add(pid)
        elif below_spared(pid) is None:
            unverified.append(tag({"pid": pid, "birth": birth, "reason": UNVERIFIED_SPARE_UNKNOWN}, target))
    # A partial table (Linux: a ``/proc/<pid>/stat`` the census could not read) must not call a
    # known pid gone; the typed reader answers absent, reused, alive or unreadable, and that
    # one answer judges every birth recorded for the pid. A held one alive there has no chain to judge.
    for pid, recorded_births in missing.items():
        identity = identities.get(pid) or _Identity(IDENTITY_UNREADABLE)
        for birth, target, held in recorded_births:
            alive = identity.state == IDENTITY_ALIVE and identity.birth == birth
            if identity.state == IDENTITY_UNREADABLE or (held and alive):
                reason = UNVERIFIED_SPARE_UNKNOWN if held else UNVERIFIED_IDENTITY
                unverified.append(tag({"pid": pid, "birth": birth, "reason": reason}, target))
            elif alive:
                members[pid] = _Row(pid, 0, identity.pgid, birth, birth_epoch(birth), IDENTITY_ALIVE, set())
                own(pid, target)
            else:
                captured_gone.append(tag({"pid": pid, "birth": birth}, target))  # absent, a corpse, or reborn
    if descendants:
        children: Dict[int, List[_Row]] = {}
        for row in rows.values():
            children.setdefault(row.ppid, []).append(row)
        pending = list(members)
        while pending:
            parent = pending.pop()
            for row in children.get(parent, ()):
                if row.pid not in excluded:
                    for target in origins.get(parent, ()):
                        own(row.pid, target)
                if row.pid not in members and row.pid not in excluded:
                    members[row.pid] = row
                    descended.add(row.pid)
                    pending.append(row.pid)
    return {"members": members, "groups_reused": reused, "groups_gone": gone, "captured_gone": captured_gone,
            "captured_unverified": unverified, "origins": origins, "descended": descended}


def _await_exit(targets: Mapping[int, str], deadline: float) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Poll ``{pid: birth}`` with ONE batched identity read per iteration until every
    target is gone or ``deadline``. Returns ``(still alive with the same birth,
    unverified {pid: reason})``: a target whose identity turned unreadable is never read
    as gone."""
    remaining = dict(targets)
    unverified: Dict[int, str] = {}
    while remaining:
        identities = _read_identities(remaining)
        alive: Dict[int, str] = {}
        unverified = {}
        for pid, birth in remaining.items():
            identity = identities.get(pid) or _Identity(IDENTITY_UNREADABLE)
            if identity.state == IDENTITY_UNREADABLE:
                unverified[pid] = UNVERIFIED_IDENTITY
                alive[pid] = birth  # keep asking until the deadline
            elif identity.state == IDENTITY_ALIVE and identity.birth == canonical_birth(birth):
                alive[pid] = birth
            # absent, zombie, or another birth: this exact process is gone
        remaining = alive
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    return {pid: birth for pid, birth in remaining.items() if pid not in unverified}, unverified


def end_scope(*, marker: str = "", markers: Iterable[str] = (), groups: Iterable[Mapping[str, Any]] = (),
              captured: Iterable[Mapping[str, Any]] = (), scope_start_epoch: float = 0.0,
              exclude: Iterable[int] = (), grace_sec: float = TASK_END_GRACE_SEC, term_first: bool = True,
              rescans: int = 2, descendants: bool = False, marker_targets: Optional[Mapping[str, Any]] = None,
              env_rows: Any = None, hold_descended: bool = False, spared: Iterable[int] = (),
              census_timeout: Optional[float] = None, before_signal: Any = None) -> Dict[str, Any]:
    """End every live member of a scope by measured identity; report what is left, typed.

    ``ended``: members gone after our signal. ``already_gone``: members confirmed absent or
    reused (ESRCH, a corpse, or a measured birth that differs) before the signal.
    ``survivors``: members alive with their recorded birth after the last pass.
    ``unverified``: members whose identity could not be read or whose signal was denied
    or failed, and every recorded identity a failed table left unchecked
    (``census_unreadable``) — never counted as ended or gone; an entry whose exact identity
    (pid AND birth) a later scan of the same pass ended or confirmed gone is dropped, while
    another birth on the same pid keeps its own. ``census_failed``: the table could not be
    read and NOTHING is claimed clean. ``settled`` requires readable requested discovery,
    no survivors and no unverified member other than a denied signal (EPERM), which no later
    pass of this installation can change: final, disclosed. ``env_rows`` narrows the marker
    read and ``census_timeout`` bounds it (``census``); ``hold_descended`` leaves members only
    the parent chain reached, held identities included, unsignalled, ``unverified`` as
    ``spared_roots_unknown`` (``spared``: ``scope_members``). ``before_signal(view)`` runs on each
    scan before any signal and returns the pids to leave unsignalled, ``unverified``. With tagged
    entries the outcome also carries ``origins``, ``{pid: [target]}``.
    """
    outcome: Dict[str, Any] = {
        "coverage": COVERAGE_UNSUPPORTED if _pl.IS_WINDOWS else COVERAGE_POSIX,
        "ended": [], "already_gone": [], "survivors": [], "unverified": [],
        "groups_reused": [], "groups_gone": [], "captured_gone": [],
        "census_failed": False, "settled": False, "scans": 0,
    }
    if _pl.IS_WINDOWS:
        return outcome
    # One spelling per recorded identity, so an old padded record and the engine's collapsed
    # one of the same process resolve, de-duplicate and stay owed as one (``canonical_birth``).
    groups = list(groups)
    captured = [{**item, "birth": canonical_birth(item.get("birth"))} for item in captured]
    excluded = set(int(pid) for pid in exclude)
    seen: Dict[int, str] = {}
    resolved: set = set()  # every (pid, birth) this pass ended or confirmed gone: resolution is per identity
    origins: Dict[int, set] = {}

    def own(pid: int, targets: Iterable[Any]) -> None:
        for target in targets:
            if target is not None:
                origins.setdefault(int(pid), set()).add(target)

    for scan in range(max(1, int(rescans) + 1)):
        names = [name for name in [marker, *markers] if name]
        # saved identities alone use the addressed reader below
        share = _grace_share() if census_timeout is not None else None  # each scan: half of what is left
        reading = {**({"env_rows": env_rows} if env_rows is not None else {}),
                   **({"timeout": census_timeout if share is None else min(census_timeout, share)}
                      if census_timeout is not None else {})}
        rows = census(names, **reading) if (names or groups or descendants) else {}
        if rows is None:
            outcome["census_failed"] = True
            # The table could not be read: every recorded identity this pass has not resolved
            # stays owed, including a group leader known before this census. Otherwise the
            # next known-only startup could forget it merely because discovery failed.
            judged = resolved | {(int(item["pid"]), str(item["birth"])) for item in outcome["survivors"]}
            leaders = [{"pid": group.get("pgid"), "birth": canonical_birth(group.get("leader_birth")),
                        "target": group.get("target")} for group in groups]
            for item in [*captured, *leaders]:
                pid, birth = int(item.get("pid") or 0), str(item.get("birth") or "")
                if pid > 1 and pid not in excluded and birth and (pid, birth) not in judged:
                    _add_unverified(outcome, pid, birth, UNVERIFIED_SPARE_UNKNOWN if item.get("held") else UNVERIFIED_CENSUS)
                    own(pid, [item.get("target")])
            break
        outcome["scans"] = scan + 1
        view = scope_members(rows, markers=names, groups=groups, captured=captured,
                             scope_start_epoch=scope_start_epoch, exclude=excluded, descendants=descendants,
                             marker_targets=marker_targets, spared=spared)
        for pid, targets in view["origins"].items():
            own(pid, targets)
        for key in ("groups_reused", "groups_gone"):
            outcome[key] = sorted(set(outcome[key]) | set(view[key]))
        for item in view["captured_gone"]:
            _add_gone(outcome, resolved, "captured_gone", item["pid"], item["birth"])
            own(item["pid"], [item.get("target")])
        for item in view["captured_unverified"]:
            _add_unverified(outcome, item["pid"], item["birth"], item["reason"])
            own(item["pid"], [item.get("target")])
        fresh = {pid: row.birth for pid, row in view["members"].items() if pid not in seen}
        if hold_descended:  # nothing tells such a member from a spared root: never signalled on a guess
            for pid in [pid for pid in fresh if pid in view["descended"]]:
                _add_unverified(outcome, pid, fresh.pop(pid), UNVERIFIED_SPARE_UNKNOWN)
                seen[pid] = view["members"][pid].birth
        if before_signal is not None:
            for pid in before_signal(view).intersection(fresh):
                _add_unverified(outcome, pid, fresh.pop(pid), UNVERIFIED_UNPUBLISHED)
                seen[pid] = view["members"][pid].birth
        if not fresh:
            break  # nothing in the scope, or a quiet rescan after the kill pass
        seen.update(fresh)
        if term_first:
            termed = _signal_pass(fresh, outcome, resolved, graceful=True, gone_bucket="already_gone")
            alive, unverified = _await_exit(termed, time.monotonic() + max(0.0, float(grace_sec)))
            _note_unverified(outcome, unverified, termed)
            _note_ended(outcome, resolved, termed, alive, unverified)
            to_confirm = _signal_pass(alive, outcome, resolved, graceful=False, gone_bucket="ended")  # died of our TERM
        else:
            to_confirm = _signal_pass(fresh, outcome, resolved, graceful=False, gone_bucket="already_gone")
        left, unverified = _await_exit(to_confirm, time.monotonic() + _KILL_CONFIRM_SEC)
        _note_unverified(outcome, unverified, to_confirm)
        _note_ended(outcome, resolved, to_confirm, left, unverified)
        outcome["survivors"].extend({"pid": pid, "birth": birth} for pid, birth in left.items())
        time.sleep(_RESCAN_SETTLE_SEC)
    for key in ("ended", "captured_gone"):
        outcome[key] = sorted(set(outcome[key]))
    # An identity a later scan ended or confirmed gone owes nothing: its earlier unverified
    # entry goes. Another birth on the same pid is another process: its entry stays.
    outcome["unverified"] = [item for item in outcome["unverified"] if (item["pid"], item["birth"]) not in resolved]
    outcome["settled"] = not outcome["census_failed"] and not outcome["survivors"] and all(
        item["reason"] == UNVERIFIED_DENIED for item in outcome["unverified"])
    if origins:
        outcome["origins"] = {pid: sorted(targets, key=str) for pid, targets in origins.items()}
    return outcome


def _signal_pass(targets: Mapping[int, str], outcome: Dict[str, Any], resolved: set, *, graceful: bool,
                 gone_bucket: str) -> Dict[int, str]:
    """Re-measure every target in ONE read, then signal each still-matching member once.
    Returns the members whose signal was accepted. A target found absent, a corpse or
    reborn lands in ``gone_bucket`` (``already_gone`` before any signal of ours, ``ended``
    after our TERM); an unreadable identity, a denied or a failed signal is UNVERIFIED."""
    identities = _read_identities(targets)
    accepted: Dict[int, str] = {}
    for pid, birth in targets.items():
        identity = identities.get(pid) or _Identity(IDENTITY_UNREADABLE)
        if identity.state == IDENTITY_UNREADABLE:
            _add_unverified(outcome, pid, birth, UNVERIFIED_IDENTITY)
            continue
        if identity.state != IDENTITY_ALIVE or identity.birth != canonical_birth(birth):
            _add_gone(outcome, resolved, gone_bucket, pid, birth)  # absent, a corpse, or another process now
            continue
        result = _pl.deliver_signal(pid, graceful=graceful)
        if result == _pl.SIGNAL_SENT:
            accepted[pid] = birth
        elif result == _pl.SIGNAL_ABSENT:
            _add_gone(outcome, resolved, gone_bucket, pid, birth)
        else:
            _add_unverified(outcome, pid, birth, UNVERIFIED_DENIED if result == _pl.SIGNAL_DENIED else UNVERIFIED_FAILED)
    return accepted


def _add_gone(outcome: Dict[str, Any], resolved: set, bucket: str, pid: int, birth: str) -> None:
    """The public buckets list pids; the resolution itself is the exact (pid, birth)."""
    resolved.add((int(pid), str(birth)))
    if pid not in outcome[bucket]:
        outcome[bucket].append(int(pid))


def _note_ended(outcome: Dict[str, Any], resolved: set, signalled: Mapping[int, str], alive: Mapping[int, str],
                unverified: Mapping[int, str]) -> None:
    for pid, birth in signalled.items():
        if pid not in alive and pid not in unverified:
            _add_gone(outcome, resolved, "ended", pid, birth)


def _add_unverified(outcome: Dict[str, Any], pid: int, birth: str, reason: str) -> None:
    if not any(item["pid"] == pid and item["birth"] == birth for item in outcome["unverified"]):
        outcome["unverified"].append({"pid": int(pid), "birth": birth, "reason": reason})


def _note_unverified(outcome: Dict[str, Any], unverified: Mapping[int, str], signalled: Mapping[int, str]) -> None:
    for pid, reason in unverified.items():
        _add_unverified(outcome, pid, signalled.get(pid, ""), reason)


# --------------------------------------------------------------------------- one owned process

_CLEANUP_FACT_KEYS = ("coverage", "reach", "exit_confirmed", "ended", "already_gone", "survivors", "unverified",
                      "census_failed", "custody")
ORPHANS_UNKNOWN_RESIDUAL = ("its leader had exited before this stop: a process that left its group and parent chain "
                            "showing no launch marker (none was recorded, or its binary hides its environment) "
                            "cannot be ruled out")


def end_owned_processes(targets: Iterable[Mapping[str, Any]], *, spared: Optional[Iterable[int]] = (),
                        graceful: bool = True) -> Dict[Any, Dict[str, Any]]:
    """End several owned processes and everything each started, in ONE pass, by measured identity.

    The stop of recorded processes (services, executor records), never of a task or an
    installation: a target is ``{key, pid, birth, floor, markers, known, custody}``, and only its
    own launch ``markers`` are read, so what another launch of the same task carries is never
    reached. Each scan reads one census for all targets and waits once: its members come from
    ``scope_members``, the target's own group under rule 2 (``birth`` vouches for a live
    leader; without it the ISO ``floor`` bounds a leaderless group's members), the process and
    ``known`` identities by pid AND birth (a ``held`` one: ``scope_members``), its markers (on
    macOS read only below launchd's children born since the earliest start: whatever left a
    chain was reparented to one, whichever ancestor hides its environment; every environment
    while ``spared`` is unknown) and every process whose parent chain reaches a member,
    whatever group it moved to. What is left is attributed to the targets whose membership
    holds it (an unattributable leftover to every target). With nothing to judge a target is
    unverified, never signalled, and a pid naming the caller, its parent or its group (a stale
    record's reused pid) is never a target, group or member. ``spared`` daemon roots, and
    held identities below them, are never members; ``None`` (unreadable) holds every member
    only a parent chain reached. ``graceful``: TERM, ``TASK_END_GRACE_SEC``, then KILL;
    otherwise KILL at once, one rescan, its table read bounded inside a running shutdown's
    grace (``_grace_share``). ``exit_confirmed``
    (everything still traceable to the target ended) is stricter than ``settled``: a denied signal
    is final for a scope record, never proof that this work ended. A target's ``custody``, ``(drive
    root, row)`` of its launch, makes this pass the launch's one publication: what could outlive its
    signals is durable before any of them (``_write_ahead``), and its ``exit_confirmed`` holds
    only once what the pass judged was reconciled (``process_custody.settle_launch``). Windows:
    ``unsupported``, its native paths stay the truth."""
    from ouroboros.process_custody import custody_lock_budget, settle_launch

    blind, prepared, terms = _owned_reach(targets, spared)
    judged = {key: _owned_outcome({"coverage": COVERAGE_POSIX, "census_failed": False, "scans": 0,
                                   "unverified": [{"pid": pid, "birth": "", "reason": UNVERIFIED_IDENTITY}]}, reach)
              for key, (pid, reach) in blind.items()}
    if not prepared:
        return judged
    lock_wait = custody_lock_budget()
    outcome = end_scope(**terms, grace_sec=TASK_END_GRACE_SEC if graceful else 0.0, term_first=graceful,
                        rescans=2 if graceful else 1, hold_descended=spared is None,
                        census_timeout=None if graceful else _grace_share(),
                        before_signal=_write_ahead(prepared, spared is None, lock_wait))
    origins, every = outcome.get("origins") or {}, [item["key"] for item in prepared]
    leaders = _unmeasured_leaders(prepared, spared)

    def mine(key: Any, pid: Any) -> bool:
        return key in (origins.get(int(pid)) or every)

    for item in prepared:
        key = item["key"]
        part = {
            "coverage": outcome["coverage"], "census_failed": outcome["census_failed"], "scans": outcome["scans"],
            **{bucket: [pid for pid in outcome[bucket] if mine(key, pid)]
               for bucket in ("ended", "already_gone", "captured_gone")},
            **{bucket: [entry for entry in outcome[bucket] if mine(key, entry["pid"])]
               for bucket in ("survivors", "unverified")},
            "groups_reused": [pid for pid in outcome["groups_reused"] if pid == item["pid"]],
            "groups_gone": [pid for pid in outcome["groups_gone"] if pid == item["pid"]]}
        named = {entry["pid"] for entry in part["survivors"] + part["unverified"]}
        if item["pid"] in leaders and item["pid"] not in named:
            part["unverified"].append({"pid": item["pid"], "birth": "", "reason": UNVERIFIED_IDENTITY})
        judged[key] = _owned_outcome(part, item["reach"])
        if item["custody"]:
            settle_launch(item["custody"], judged[key], item["judged"], lock_wait)
    return judged


def _write_ahead(prepared: List[Dict[str, Any]], held: bool, lock_wait: Any) -> Any:
    """``end_scope``'s ``before_signal``: what could outlive THIS pass's signals is durable in its launch's custody
    before any of them, so no stop loses what it knows. That is every member only a parent chain reached, as this pass
    would owe it (``held`` while the daemon roots are unknown), and every identity the pass was handed: being judged is
    not being persisted, since a retained input may come from a stop whose every write failed. Only a successful write
    counts (nothing is appended for what the ledger already names), and only for this launch; its leader needs none,
    as its row names it, and neither does what the table shows dead. What no custody can name yet (a launch still
    registering, every write failed) holds its whole launch back: nothing of it is signalled, so the chain stays for
    the retry. This orders one stop's write against that stop's own signal only: a stop that never observed a member
    (started after its last table read) can confirm what it saw before another stop's write lands; custody names the
    member from that write on. What was newly written counts as judged by this pass (``settle_launch``)."""
    from ouroboros.process_custody import write_ahead

    every = [item["key"] for item in prepared]

    def name(entry: Mapping[str, Any]) -> Tuple[int, str]:
        return int(entry.get("pid") or 0), canonical_birth(entry.get("birth"))

    def publish(view: Mapping[str, Any]) -> set:
        withheld: set = set()
        dead = {name(entry) for entry in view["captured_gone"]} | {
            name({"pid": pid, "birth": row.birth}) for pid, row in view["members"].items() if row.state != IDENTITY_ALIVE}
        for item in prepared:
            def mine(pid: int, key: Any = item["key"]) -> bool:
                return key in (view["origins"].get(pid) or every)

            chain = [{"pid": pid, "birth": view["members"][pid].birth, **({"held": True} if held else {})}
                     for pid in sorted(view["descended"]) if pid in view["members"] and mine(pid)]
            owed = {name(entry): entry for entry in [*chain, *item["known"]]}  # a retained identity keeps its own flags
            spoken = dead | item["durable"] | {(item["pid"], item["birth"])}
            unwritten = [entry for key, entry in owed.items() if key not in spoken]
            if not unwritten or not item["custody"]:
                continue
            if write_ahead(item["custody"], unwritten, lock_wait()):
                item["durable"].update(name(entry) for entry in unwritten)
                judged = {name(entry) for entry in item["judged"]}
                item["judged"] += [entry for entry in unwritten if name(entry) not in judged]
            else:
                withheld.update(pid for pid in view["members"] if mine(pid))
        return withheld

    return publish


def gone_identities(items: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Those of ``items`` whose exact process is provably gone, by one batched read: absent, a corpse or another birth.
    An unreadable identity is not gone."""
    items = list(items)
    answers = _read_identities({int(item["pid"]) for item in items})

    def gone(item: Mapping[str, Any]) -> bool:
        answer = answers.get(int(item["pid"])) or _Identity(IDENTITY_UNREADABLE)
        return answer.state != IDENTITY_UNREADABLE and not (
            answer.state == IDENTITY_ALIVE and answer.birth == canonical_birth(item.get("birth")))

    return [item for item in items if gone(item)]


def _unmeasured_leaders(prepared: List[Dict[str, Any]], spared: Optional[Iterable[int]]) -> set:
    """The recorded leader pids, never measured, that a live (or unreadable) process still holds: one batched
    read. Each may be its leader hiding its marker, so its target is never confirmed while it lives; nothing
    measured it, so nothing signals it."""
    roots = {int(pid) for pid in spared or ()}
    asked = [item["pid"] for item in prepared if item["unmeasured"] and item["pid"] not in roots]
    return {pid for pid, identity in (_read_identities(asked) if asked else {}).items()
            if identity.state in (IDENTITY_ALIVE, IDENTITY_UNREADABLE)}


def _owned_reach(targets: Iterable[Mapping[str, Any]], spared: Optional[Iterable[int]]) -> Tuple[
        Dict[Any, Tuple[int, List[str]]], List[Dict[str, Any]], Dict[str, Any]]:
    """The one reach of ``end_owned_processes`` and ``observe_owned_processes``: targets with nothing to
    judge (``{key: (pid, reach)}``), the prepared rest and their ONE scope's membership terms."""
    own = {os.getpid(), os.getppid(), _pl.current_process_group_id()}
    blind: Dict[Any, Tuple[int, List[str]]] = {}
    prepared: List[Dict[str, Any]] = []
    for target in targets:
        key, pid = target["key"], int(target["pid"])
        birth, known = canonical_birth(target.get("birth")), list(target.get("known") or ())
        markers = [name for name in target.get("markers") or () if is_scope_name(str(name))]
        floor = target.get("floor") or ""
        mine = pid not in own
        if not mine:  # a reused pid naming this process, its parent or its group: never a target or a group
            birth, floor = "", ""
        floor_text = "" if birth else _iso_floor(floor)
        reach = [*(["group"] if birth or floor_text else []), "descendants", *(["marker"] if markers else []),
                 *(["owed"] if known else [])]
        if not (birth or floor_text or known or markers) and not _pl.IS_WINDOWS:
            blind[key] = (pid, reach)
            continue
        prepared.append({"key": key, "pid": pid, "birth": birth, "floor": floor_text, "markers": markers,
                         "known": known, "reach": reach, "custody": target.get("custody"),
                         # the exact input this pass measures: all it can discharge, with what it writes ahead
                         "judged": [*known, *([{"pid": pid, "birth": birth}] if birth else [])], "durable": set(),
                         "unmeasured": mine and pid > 1 and not birth and not target.get("exited")})
    starts = [birth_epoch(item["birth"] or item["floor"]) for item in prepared]
    since = min(starts) if prepared and all(starts) else 0.0
    marker_targets = {name: item["key"] for item in prepared for name in item["markers"]}
    return blind, prepared, dict(
        groups=[{"pgid": item["pid"], "leader_pid": item["pid"], "leader_birth": item["birth"], "target": item["key"],
                 **({"floor": item["floor"]} if item["floor"] else {})} for item in prepared
                if item["birth"] or item["floor"]],
        captured=[entry for item in prepared for entry in (
            [{"pid": item["pid"], "birth": item["birth"], "target": item["key"]}] if item["birth"] else []) + [
            {**known, "target": item["key"]} for known in item["known"]]],
        markers=list(marker_targets), marker_targets=marker_targets, spared=spared or (),
        exclude={1, *own, *(int(pid) for pid in spared or ())}, descendants=True,
        # Only what lies below a launchd child born since the launch needs its environment read for a marker,
        # unless the hold must tell a marker carrier from a chain-only member: then every one is read.
        env_rows=(lambda rows: _orphaned_since(rows, since - 1.0))
        if marker_targets and since and spared is not None else None)


def observe_owned_processes(targets: Iterable[Mapping[str, Any]], *,
                            spared: Optional[Iterable[int]] = ()) -> Dict[Any, str]:
    """What is left of several owned processes, read for custody readers: no signal, no wait, at most
    ONE census.

    The same targets and reach as ``end_owned_processes``. Each key answers ``live`` (a member its
    reach names is alive, never a zombie: leader by birth, group under rule 2, an owed identity, a marker carrier or
    a descendant of one), ``unknown`` (the table or an identity could not be read, the daemon roots
    are unknown (``spared=None``) for a member only a parent chain reached, or nothing was recorded to
    judge by) or ``settled``. A reused leader pid is another process, never this target's member.
    A leader measured alive with its recorded birth (one batched identity read) already answers
    ``live``; only the other targets need the table."""
    targets = list(targets)
    blind, prepared, _ = _owned_reach(targets, spared)
    states = {key: "unknown" for key in blind}
    if not prepared:
        return states
    if _pl.IS_WINDOWS:
        return {**states, **{item["key"]: "unknown" for item in prepared}}
    roots = {int(pid) for pid in spared or ()}  # a spared daemon root is never a member, on either path
    leaders = read_identities(item["pid"] for item in prepared if item["birth"] and item["pid"] not in roots)
    for item in prepared:
        identity = leaders.get(item["pid"])
        if (item["birth"] and identity is not None and identity.state == IDENTITY_ALIVE
                and identity.birth == item["birth"]):
            states[item["key"]] = "live"
    _, prepared, terms = _owned_reach([target for target in targets if target["key"] not in states], spared)
    if not prepared:
        return states
    rows = census(terms["markers"], **({"env_rows": terms["env_rows"]} if terms["env_rows"] is not None else {}))
    if rows is None:
        return {**states, **{item["key"]: "unknown" for item in prepared}}
    view = scope_members(rows, markers=terms["markers"], groups=terms["groups"], captured=terms["captured"],
                         exclude=terms["exclude"], descendants=True, marker_targets=terms["marker_targets"],
                         spared=terms["spared"])
    origins, every = view["origins"], [item["key"] for item in prepared]
    held = view["descended"] if spared is None else set()
    unmeasured = _unmeasured_leaders(prepared, spared)
    for item in prepared:
        def mine(pid: Any, key: Any = item["key"]) -> bool:
            return key in (origins.get(int(pid)) or every)

        # A zombie runs nothing: its parent (a worker still holding the Popen) has not reaped it yet. The stop already
        # counts it ended (``owned_shutdown._exit_proven``); a reader calling it live would keep an ended launch open.
        live = any(mine(pid) for pid, row in view["members"].items()
                   if pid not in held and row.state != IDENTITY_ZOMBIE)
        unsure = any(mine(pid) for pid in held if pid in view["members"]) or any(
            mine(entry["pid"]) for entry in view["captured_unverified"]) or item["pid"] in unmeasured
        states[item["key"]] = "live" if live else "unknown" if unsure else "settled"
    return states


def _orphaned_since(rows: Mapping[int, _Row], floor: float) -> List[int]:
    """Every pid below (or being) a launchd child born at or after ``floor``. macOS has no
    subreaper: a process whose parent chain broke was reparented to launchd, so whatever still
    carries a launch's marker lies in such an orphan's subtree, whichever ancestor hides its
    environment."""
    top: Dict[int, int] = {}
    for pid in rows:
        path, current = [], pid
        while current not in top:
            parent = rows[current].ppid
            if parent <= 1 or parent not in rows or len(path) > len(rows):
                top[current] = current  # launchd's child, or the highest ancestor the table shows
                break
            path.append(current)
            current = parent
        top.update(dict.fromkeys(path, top[current]))
    return [pid for pid, head in top.items() if rows[head].epoch >= floor]


def _read_identities(pids: Iterable[int]) -> Dict[int, _Identity]:
    """``read_identities`` bounded by a running shutdown's grace share (``_grace_share``); its own bound otherwise."""
    share = _grace_share()
    return read_identities(pids) if share is None else read_identities(pids, timeout=share)


def _grace_share() -> Optional[float]:
    """Half of what is left of a running shutdown's one grace (``owned_shutdown``), for each table
    and identity read a stop makes inside it: reads that never answer still leave the native kill
    and its confirmation inside that grace. ``None`` outside a shutdown: each read's own bound."""
    from ouroboros.owned_shutdown import stop_deadline

    deadline = stop_deadline()
    return None if deadline is None else max(0.2, (deadline - time.monotonic()) / 2)


def _owned_outcome(part: Dict[str, Any], reach: List[str]) -> Dict[str, Any]:
    for bucket in ("ended", "already_gone", "captured_gone", "survivors", "unverified", "groups_reused",
                   "groups_gone"):
        part.setdefault(bucket, [])
    part["reach"] = reach
    part["exit_confirmed"] = (part["coverage"] == COVERAGE_POSIX and not part["census_failed"]
                              and not part["survivors"] and not part["unverified"])
    part["known_members"] = _known_members(part)
    return part


def process_cleanup_facts(outcome: Mapping[str, Any]) -> Dict[str, Any]:
    """The bounded typed facts of one owned stop a result carries: pids and births, never environment."""
    return {key: outcome.get(key) for key in _CLEANUP_FACT_KEYS if key != "custody" or outcome.get(key)}


def end_held_processes(held: Iterable[Mapping[str, Any]], *, others: Iterable[Mapping[str, Any]] = (),
                       others_spared: Optional[Iterable[int]] = (), graceful: bool = True) -> Dict[Any, Dict[str, Any]]:
    """End children this process still holds (``Popen``) and everything each started, in one pass.

    ``held``: ``{key, proc, birth, started_at, markers, known, drive_root, custody}``; ``others``: recorded-only
    targets of ``end_owned_processes`` joining the same pass (their keys answer with its outcome;
    ``others_spared`` adds their daemon roots, ``None`` when unreadable). POSIX:
    ``end_owned_processes`` over each leader the Popen identifies (``birth``, else read now:
    an unreaped pid cannot be reused), sparing the installation's daemon roots. A failed table
    or an unmeasurable live leader stays unconfirmed: an immediate (shutdown) stop then still
    sends the held child's own group the native kill, once what the pass owes is durable (that kill
    breaks a chain like any signal), a graceful one signals nothing so a retry can find the whole tree. A leader gone before the stop leaves its chain broken: an orphan
    with no readable marker is beyond any retry, so the stop confirms what it still reaches and
    ``residual`` names that gap. Windows: the native tree
    kill, and only the leader's exit is confirmed (``unsupported``). Returns ``{key: {"facts",
    "birth", "known"}}``: the facts a stop result carries, the leader's birth and the identities
    the next attempt must check."""
    held, others = list(held), list(others)
    answers: Dict[Any, Dict[str, Any]] = {}
    if _pl.IS_WINDOWS:
        if others:
            answers.update(end_owned_processes(others, spared=others_spared, graceful=graceful))
        for item in held:
            proc = item["proc"]
            if proc.poll() is None:
                _pl.kill_process_tree(proc)
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
            answers[item["key"]] = {"birth": item.get("birth") or "", "known": [], "facts": {
                "coverage": COVERAGE_UNSUPPORTED, "reach": ["native tree"], "exit_confirmed": proc.poll() is not None,
                "residual": "the native tree kill ended it; a descendant that left that tree is not verified"}}
        return answers
    from ouroboros.process_custody import custody_lock_budget, spared_daemon_roots

    targets, gone, roots_of = [], {}, {}
    spared = None if others_spared is None else set(others_spared)
    unread = [item["proc"].pid for item in held if not item.get("birth") and item["proc"].returncode is None]
    # One read for every held leader with no recorded birth.
    births = _read_identities(unread) if unread else {}
    for item in held:
        proc, birth = item["proc"], item.get("birth") or ""
        if not birth and proc.returncode is None and births.get(proc.pid) is not None:
            identity = births[proc.pid]
            birth = identity.birth if identity.state in (IDENTITY_ALIVE, IDENTITY_ZOMBIE) else ""
        gone[item["key"]] = proc.poll() is not None
        started = float(item.get("started_at") or 0.0)
        targets.append({"key": item["key"], "pid": proc.pid, "birth": birth, "markers": item.get("markers") or (),
                        "known": item.get("known") or (), "exited": gone[item["key"]], "custody": item.get("custody"),
                        "floor": _dt.datetime.fromtimestamp(started, _dt.timezone.utc).isoformat() if started else ""})
        drive = str(item.get("drive_root") or "")
        if drive not in roots_of:  # one ledger read per installation, not per service
            roots_of[drive] = spared_daemon_roots(item.get("drive_root"))
        spared = None if roots_of[drive] is None or spared is None else spared | roots_of[drive]
    outcomes = end_owned_processes([*targets, *others], spared=spared, graceful=graceful)
    answers.update({target["key"]: outcomes[target["key"]] for target in others})
    lock_wait = custody_lock_budget()
    for item, target in zip(held, targets):
        key, proc, outcome = item["key"], item["proc"], outcomes[item["key"]]
        measured = bool(target["birth"]) or gone[key]
        if not graceful and (outcome["census_failed"] or not measured) and _owed_is_durable(target, outcome, lock_wait):
            # Shutdown/update keep their old native group kill when nothing could be measured; a stop that
            # can be retried signals nothing, so a later readable table still finds the whole tree.
            _pl.request_process_tree_kill(proc)  # refused once the held child exited
        if not measured:  # alive but unmeasured: nothing above could judge it
            _add_unverified(outcome, proc.pid, "", UNVERIFIED_IDENTITY)
            outcome["exit_confirmed"] = False
        facts = process_cleanup_facts(outcome)
        if gone[key]:
            facts["residual"] = ORPHANS_UNKNOWN_RESIDUAL  # disclosed: no later stop could trace further either
        answers[key] = {"facts": facts, "birth": target["birth"], "known": outcome["known_members"]}
        if facts["exit_confirmed"] and outcome.get("row"):
            # Beside a record step of the same launch, each retires the row it left: whichever publishes last, it goes.
            from ouroboros.owned_shutdown import forget_ledgered_rows
            from ouroboros.process_custody import _lock_wait

            with contextlib.suppress(Exception):  # a record left behind only costs the next stop a look
                forget_ledgered_rows(item["drive_root"], [outcome["row"]], timeout_sec=_lock_wait())
    for item in held:  # a confirmed leader is a corpse to reap; any other is polled, never waited for
        with contextlib.suppress(Exception):
            item["proc"].wait(timeout=5) if answers[item["key"]]["facts"]["exit_confirmed"] else item["proc"].poll()
    return answers


def _owed_is_durable(target: Mapping[str, Any], outcome: Mapping[str, Any], lock_wait: Any) -> bool:
    """May a native kill break this launch's chain: its pass reconciled, or what else it owes is written now."""
    from ouroboros.process_custody import write_ahead

    leader = (int(target["pid"]), canonical_birth(target.get("birth")))
    owed = [item for item in outcome["known_members"] if (int(item["pid"]), canonical_birth(item.get("birth"))) != leader]
    return not target.get("custody") or not owed or bool(outcome.get("row")) or write_ahead(target["custody"], owed, lock_wait())


def held_item(record: Any, proc: Any) -> Dict[str, Any]:
    """One service record's ``end_held_processes`` item: a copy of what it still owes, its launch's identity and the
    custody of that launch, found by its marker even while its registration has not returned its row yet."""
    marker = record.scope_marker
    return {"key": record.service_id, "proc": proc, "birth": record.leader_birth, "started_at": record.started_at,
            "markers": [marker] if marker else [], "known": list(record.known_members), "drive_root": record.drive_root,
            "custody": (record.drive_root, getattr(proc, "_ouroboros_custody", None)
                        or {"pid": proc.pid, "scope_marker": marker}) if record.drive_root and marker else None}


# --------------------------------------------------------------------------- task scopes

def task_scopes_dir(drive_root: Any) -> pathlib.Path:
    return _state_dir(drive_root) / TASK_SCOPES_DIR


def task_scope_path(drive_root: Any, task_id: str) -> pathlib.Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(task_id or ""))
    return task_scopes_dir(drive_root) / f"{safe}.json"


def task_scope_record(drive_root: Any, task_id: str) -> Optional[Dict[str, Any]]:
    return read_json_dict(task_scope_path(drive_root, task_id))


def task_scope_records(drive_root: Any) -> List[Dict[str, Any]]:
    folder = task_scopes_dir(drive_root)
    try:
        paths = sorted(folder.iterdir())
    except OSError:
        return []
    records = []
    for path in paths:
        if path.suffix != ".json":
            continue
        record = read_json_dict(path)
        if record and record.get("task_id") and record.get("marker"):
            records.append(record)
    return records


def _fresh_record(task_id: str, groups: Optional[List[Dict[str, Any]]] = None, previous_markers: Iterable[str] = (),
                  known_members: Iterable[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    return {"task_id": str(task_id), "marker": mint_marker(TASK_SCOPE_PREFIX),
            "minted_at": utc_now_iso(), "minted_epoch": time.time(), "groups": list(groups or []),
            "previous_markers": [name for name in previous_markers if is_scope_name(name)],
            "known_members": [dict(item) for item in known_members]}


def _rotated(task_id: str, current: Dict[str, Any]) -> Dict[str, Any]:
    """A fresh record for a NEW launch of a task whose cleanup already started: the new
    children get a fresh id so the running cleanup cannot reach them, while the groups,
    the markers AND the known members the old record still owed are carried (rule 2, the
    census and the identity reader judge each at the next settlement)."""
    previous = [str(current.get("marker") or ""), *(current.get("previous_markers") or [])]
    return _fresh_record(task_id, current.get("groups") or [], previous, _saved_members(current))


def _record_markers(record: Mapping[str, Any]) -> List[str]:
    return [name for name in [str(record.get("marker") or ""), *(record.get("previous_markers") or [])]
            if is_scope_name(name)]


def _retire_record(path: pathlib.Path, marker: str) -> bool:
    """Remove the record only while it still names ``marker`` (a rotation in between keeps
    the successor's record), under the record's own writer lock."""
    removed = []

    def tombstone(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if str((current or {}).get("marker") or "") != marker:
            return None
        return {**current, "cleanup": {"state": "done"}}

    def unlink(_written: Dict[str, Any]) -> None:
        path.unlink(missing_ok=True)
        removed.append(True)

    try:
        update_json_locked(path, tombstone, after_write=unlink)
    except (OSError, TimeoutError, ValueError):
        return False
    return bool(removed)


def task_scope_marker(drive_root: Any, task_id: str) -> str:
    """The task's marker, minted and written durably BEFORE the first launch under it.

    A record whose cleanup already started is rotated: the successor's children get a
    fresh id so a late cleanup of the ended attempt cannot reach them."""
    path = task_scope_path(drive_root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    def ensure(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not current or not is_scope_name(str(current.get("marker") or "")):
            return _fresh_record(task_id)
        return _rotated(task_id, current) if (current.get("cleanup") or {}).get("state") else None

    return str(update_json_locked(path, ensure).get("marker") or "")


def task_scope_env(drive_root: Any, task_id: str) -> Dict[str, str]:
    """The per-launch entry a task's child receives (never put into ``os.environ``)."""
    if not drive_root or not task_id:
        return {}
    return {task_scope_marker(drive_root, task_id): SCOPE_VALUE}


def task_launch_target(binding: Optional[Mapping[str, Any]] = None, *, drive_root: Any = None,
                       task_id: str = "") -> Optional[tuple]:
    """``(drive_root, task_id)`` the launch belongs to, from the context-bound invocation
    binding (``tool_custody.invocation_binding()``) or the caller's own context; ``None``
    for an unbound/non-task launch. An active invocation is authoritative, including empty
    coordinates; fallback coordinates must come from ``owner_pause._member_coordinates``,
    never a display label. Every real task id, including ``manual``, owns its scope."""
    root, task = ((binding.get("drive_root"), binding.get("task_id")) if binding
                  else (drive_root, task_id))
    if not root or not task:
        return None
    return pathlib.Path(root), str(task)


def scoped_child_env(env: Mapping[str, str], target: Optional[tuple]) -> Dict[str, str]:
    """The child's final environment: a task launch adds its task marker; a launch handed to
    a longer-lived custodian (``target=None``: a keep/session service) sheds every task and
    engine-run marker and keeps the installation marker. Never mutates ``env``."""
    if target is None:
        return shed_scopes(env)
    try:
        return {**env, **task_scope_env(*target)}
    except (OSError, TimeoutError) as exc:
        # This environment is prepared before Popen. A durable marker may exist;
        # the fact is no child launch, not rollback or a side-effect-free tool call.
        exc.process_not_started = True
        raise


def record_task_group(drive_root: Any, task_id: str, pid: int) -> Optional[Dict[str, Any]]:
    """Record the group a just-spawned task child leads, with its measured birth.

    Groups without a live member are pruned on every append (``killpg(G, 0)`` is one
    syscall), so a task that runs thousands of commands keeps a small record."""
    if _pl.IS_WINDOWS or not drive_root or not task_id or int(pid) <= 0:
        return None
    pgid, birth = leader_facts(int(pid))
    if pgid <= 1:
        return None
    entry = {"pgid": pgid, "leader_pid": int(pid), "leader_birth": birth, "recorded_at": utc_now_iso()}
    path = task_scope_path(drive_root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    def append(current: Dict[str, Any]) -> Dict[str, Any]:
        if not current or not is_scope_name(str(current.get("marker") or "")):
            record = _fresh_record(task_id)
        elif (current.get("cleanup") or {}).get("state"):
            record = _rotated(task_id, current)
        else:
            record = dict(current)
        kept = [group for group in record.get("groups") or []
                if int(group.get("pgid") or 0) != pgid and _pl.process_group_is_alive(int(group.get("pgid") or 0))]
        record["groups"] = [*kept, entry]
        return record

    try:
        update_json_locked(path, append)
    except (OSError, TimeoutError, ValueError):
        log.warning("task scope group for %s could not be recorded; task-end cleanup relies on the marker",
                    task_id, exc_info=True)
        return None
    return entry


def task_scope_terminal_evidence(row: Any) -> bool:
    """Positive durable terminal evidence: a truly terminal status that is not a
    reopenable presence placeholder. Running, interrupted, paused, held, handed off,
    unreadable and placeholder rows are NOT terminal."""
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, is_reconciled_presence_placeholder

    if not isinstance(row, dict) or str(row.get("status") or "") not in _TRULY_TERMINAL_STATUSES:
        return False
    return not is_reconciled_presence_placeholder(row)


_CLEANUP_THREADS: List[threading.Thread] = []
_CLEANUP_LOCK = threading.Lock()


def note_task_terminal_transition(drive_root: Any, task_id: str) -> Optional[threading.Thread]:
    """Called by the result writer right after an ACCEPTED transition into a truly
    terminal status landed: the obligation is durable already (the scope record plus
    the terminal row), so it is executed outside the store transaction on a thread.
    Nothing happens for a task that launched nothing (no record)."""
    try:
        if not task_scope_path(drive_root, task_id).exists():
            return None
    except OSError:
        return None
    thread = threading.Thread(target=settle_task_scope, args=(drive_root, task_id),
                              kwargs={"reason": "terminal_transition"}, name=f"task-scope-cleanup-{task_id}",
                              daemon=True)
    with _CLEANUP_LOCK:
        _CLEANUP_THREADS[:] = [item for item in _CLEANUP_THREADS if item.is_alive()]
        _CLEANUP_THREADS.append(thread)
    thread.start()
    return thread


def wait_for_cleanups(timeout_sec: float = 10.0) -> None:
    """Join the cleanup threads this process started (``note_task_terminal_transition``), for a
    test that observes a cleanup's outcome; no production caller waits on them."""
    deadline = time.monotonic() + timeout_sec
    with _CLEANUP_LOCK:
        threads = list(_CLEANUP_THREADS)
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))


_OUTCOME_DIGEST_KEYS = ("coverage", "ended", "already_gone", "survivors", "unverified", "groups_reused",
                        "groups_gone", "captured_gone", "census_failed", "settled")


def outcome_digest(outcome: Mapping[str, Any]) -> str:
    """A digest of the facts that matter (never the clock), so a repeated unchanged
    outcome is recorded once instead of on every sweep."""
    body = json.dumps({key: outcome.get(key) for key in _OUTCOME_DIGEST_KEYS}, sort_keys=True, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _record_task_scope_outcome(drive_root: Any, task_id: str, facts: Mapping[str, Any]) -> bool:
    """Attach the outcome to the task's result row WITHOUT writing a status: the row's
    lifecycle stays whatever the terminal writer made it (C7: this is no status writer)."""
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema
    from ouroboros.task_store import patch_document

    def attach(existing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not existing:
            return None
        require_writable_task_result_schema(existing)
        return stamp_task_result_schema({**existing, "process_scope": dict(facts), "updated_at": utc_now_iso()})

    try:
        stored = patch_document(drive_root, str(task_id), attach)
    except Exception:
        log.warning("task scope outcome for %s could not be recorded on its result", task_id, exc_info=True)
        return False
    return isinstance(stored, dict) and stored.get("process_scope") == dict(facts)


def _unsettled_reason(outcome: Mapping[str, Any]) -> str:
    if outcome.get("census_failed"):
        return "census_failed"
    if outcome.get("survivors"):
        return "survivors"
    if any(item.get("reason") != UNVERIFIED_DENIED for item in outcome.get("unverified") or []):
        return "unverified"
    return ""


def _known_members(outcome: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The members this pass could not end or confirm, by pid AND birth: inputs of the next
    pass, checked like a captured descendant — losing marker visibility never discharges
    one; confirmed absence, a corpse or another birth does. A member held because nothing
    told it from a daemon root is carried ``held``: no signal authority of its own, judged
    again by its parent chain once the roots are known (``scope_members``)."""
    return [{"pid": int(item["pid"]), "birth": str(item["birth"]),
             **({"held": True} if item.get("reason") == UNVERIFIED_SPARE_UNKNOWN else {})}
            for item in [*(outcome.get("survivors") or []), *(outcome.get("unverified") or [])]
            if item.get("birth")]


def _saved_members(record: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{"pid": int(item["pid"]), "birth": str(item["birth"])} for item in (record or {}).get("known_members") or []
            if isinstance(item, dict) and str(item.get("pid") or "").isdigit() and item.get("birth")]


def settle_task_scope(drive_root: Any, task_id: str, *, reason: str = "sweep") -> Optional[Dict[str, Any]]:
    """Execute one task's cleanup obligation on positive terminal evidence only.

    Reads the record and the durable result; a task that is not truly terminal (or whose
    row is a reopenable placeholder, or unreadable) is left alone. The locked claim holds
    only while the record still names the marker read first: a record retired or rotated
    in between is neither recreated nor claimed, and nothing runs. The outcome is attached
    to the result (``process_scope``, no status write) when it changed since the last
    attempt; the record is retired only when this task's requested discovery is readable,
    no survivor or unverified member remains except denied signals, and the outcome was recorded — otherwise it stays,
    with its marker, its group evidence and the known members (pid, birth) the next sweep
    must check, for that sweep. An unsupported platform is a final disclosure.
    """
    path = task_scope_path(drive_root, task_id)
    record = read_json_dict(path)
    if not record:
        return None
    try:
        from ouroboros.task_results import load_task_result

        row = load_task_result(drive_root, str(task_id))
    except Exception:
        return None
    if not task_scope_terminal_evidence(row):
        return None
    started_at = time.time()
    marker = str(record.get("marker") or "")

    def claim(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not current or str(current.get("marker") or "") != marker:
            return None  # retired or rotated since the read: no record to recreate, no successor's to claim
        cleanup = current.get("cleanup") or {}
        if cleanup.get("state") == "running" and started_at - float(cleanup.get("started_epoch") or 0) < _CLEANUP_RUNNING_STALE_SEC:
            return None  # another custodian is on it right now
        return {**current, "cleanup": {**cleanup, "state": "running", "started_epoch": started_at, "reason": reason}}

    try:
        claimed = update_json_locked(path, claim)
    except (OSError, TimeoutError, ValueError):
        return None
    if (claimed.get("cleanup") or {}).get("started_epoch") != started_at:
        return None
    previous_cleanup = dict(claimed.get("cleanup") or {})
    outcome = end_scope(markers=_record_markers(claimed), groups=claimed.get("groups") or [],
                        captured=_saved_members(claimed), scope_start_epoch=float(claimed.get("minted_epoch") or 0.0),
                        exclude={os.getpid(), 1})
    facts = {**outcome, "marker_role": "task", "reason": reason, "settled_at": utc_now_iso()}
    digest = outcome_digest(outcome)
    recorded = digest == str(previous_cleanup.get("last_digest") or "")  # unchanged: recorded already
    if not recorded:
        recorded = _record_task_scope_outcome(drive_root, task_id, facts)
    final = outcome["coverage"] == COVERAGE_UNSUPPORTED  # no census here: disclosed once, never retried
    if recorded and (outcome["settled"] or final):
        _retire_record(path, str(claimed.get("marker") or ""))
        return facts
    pending = {"state": "pending", "reason": _unsettled_reason(outcome) or "outcome_unrecorded",
               "attempts": int(previous_cleanup.get("attempts") or 0) + 1, "attempted_at": utc_now_iso(),
               **({"last_digest": digest} if recorded else {})}
    known = _known_members(outcome)
    try:
        update_json_locked(path, lambda current: {**current, "cleanup": pending, "known_members": known}
                           if str((current or {}).get("marker") or "") == str(claimed.get("marker") or "") else None)
    except (OSError, TimeoutError, ValueError):
        pass
    return facts


def settle_task_scopes(drive_root: Any, live_task_ids: Any = None) -> List[Dict[str, Any]]:
    """The startup/periodic sweep's pass over every outstanding obligation: a record whose
    task has positive durable terminal evidence is settled; a live, unknown, retained or
    paused owner keeps its processes. ``live_task_ids`` is a set or a zero-arg callable."""
    records = task_scope_records(drive_root)
    if not records:
        return []
    live = live_task_ids() if callable(live_task_ids) else live_task_ids
    settled = []
    for record in records:
        task_id = str(record["task_id"])
        if live is not None and task_id in live:
            continue
        try:
            facts = settle_task_scope(drive_root, task_id, reason="sweep")
        except Exception:
            log.debug("task scope sweep failed for %s", task_id, exc_info=True)
            continue
        if facts is not None:
            settled.append({"task_id": task_id, **facts})
    return settled


# --------------------------------------------------------------------------- the engine's index

def engine_run_scopes_path(data_dir: Any) -> pathlib.Path:
    from ouroboros.claudexor_daemon import _OWNED_DIR_NAME

    return pathlib.Path(data_dir) / _OWNED_DIR_NAME / ENGINE_RUN_SCOPES_FILE


def engine_scope_index(data_dir: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The managed engine's durable run-scope index (``daemon/run-scopes.json``,
    ``schema_version`` 1): per entry ``run_id``, ``marker``, ``state``, ``opened_at`` (ISO
    time: the members' age floor while no lstart birth exists), ``scope_start`` (lstart or
    null), ``root_groups[{pgid, leader_pid, leader_birth | null}]``, ``captured[{pid, pgid,
    birth}]``, births as C-locale ``ps -o lstart=`` text. Returns ``(valid entries, disclosure)``: a
    missing, unreadable (I/O, torn JSON, not an object) or newer-schema file yields no
    entries and a typed status; a schema-1 file without a top-level ``entries`` list is
    ``invalid``; a malformed row is disclosed under ``malformed`` (status ``invalid``) while
    the well-formed rows are returned — never an empty scope read as clean, never a raise."""
    path = engine_run_scopes_path(data_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], {"status": "absent", "path": str(path)}
    except OSError as exc:
        return [], {"status": "unreadable", "path": str(path), "error": type(exc).__name__}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        return [], {"status": "unreadable", "path": str(path), "error": type(exc).__name__}
    if not isinstance(data, dict):
        return [], {"status": "unreadable", "path": str(path), "error": "not an object"}
    version = data.get("schema_version")
    if version != ENGINE_RUN_SCOPES_SCHEMA:
        return [], {"status": "unsupported_schema", "path": str(path), "schema_version": version}
    entries = data.get("entries")
    if not isinstance(entries, list):
        return [], {"status": "invalid", "path": str(path),
                    "error": "entries is not a list" if "entries" in data else "entries missing"}
    valid: List[Dict[str, Any]] = []
    malformed: List[Dict[str, Any]] = []
    for position, entry in enumerate(entries):
        defect = _entry_defect(entry)
        if defect:
            malformed.append({"index": position, "reason": defect,
                              "run_id": str(entry.get("run_id") or "") if isinstance(entry, dict) else ""})
        else:
            valid.append(entry)
    disclosure = {"status": "invalid" if malformed else "read", "path": str(path), "entries": len(valid),
                  "states": sorted({str(entry.get("state") or "") for entry in valid})}
    if malformed:
        disclosure["malformed"] = malformed
    return valid, disclosure


_ENTRY_ROWS = {"root_groups": {"pgid": (int,), "leader_pid": (int, None), "leader_birth": (str, None)},
               "captured": {"pid": (int,), "pgid": (int,), "birth": (str,)}}


def _entry_defect(entry: Any) -> str:
    """Why ``entry`` falls outside the schema-1 contract (``""`` when it fits): a
    grammar-shaped ``marker``; ``scope_start`` a string or null; ``root_groups`` and
    ``captured`` lists of objects with integer pids/pgids, string births and a string or
    null ``leader_birth`` (null keeps the entry's floor: ``scope_start``, else
    ``opened_at``); the optional ``leader_identity`` is ignored."""
    if not isinstance(entry, dict):
        return "entry is not an object"
    if not is_scope_name(str(entry.get("marker") or "")):
        return "marker outside the grammar"
    if entry.get("scope_start") is not None and not isinstance(entry["scope_start"], str):
        return "scope_start is not a string"
    for key, shape in _ENTRY_ROWS.items():
        rows = entry.get(key)
        if rows is None:
            continue
        if not isinstance(rows, list):
            return f"{key} is not a list"
        for row in rows:
            if not isinstance(row, dict):
                return f"{key} row is not an object"
            for name, kinds in shape.items():
                value = row.get(name)
                if (value is None and None in kinds) or (str in kinds and isinstance(value, str)) or (
                        int in kinds and isinstance(value, int) and not isinstance(value, bool)):
                    continue
                return f"{key}.{name} malformed"
    return ""


def _engine_scope_targets(entries: Iterable[Mapping[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(markers, groups, captured)`` the engine recorded: every entry's marker through
    the census; its root groups and the groups its captured pids lead (a captured pid
    that is its own group id vouches for that group's leader birth) under rule 2; and
    each captured pid by its recorded birth. A null ``leader_birth`` stays empty — the
    host cannot verify a live ``pid == pgid`` for it, so rule 2 skips that group — while
    the entry's ``scope_start``, else its ISO ``opened_at`` in lstart form (the engine's
    contract: the floor while no lstart birth exists), rides as the group's ``floor``, an
    age bound for the members a leaderless group still holds."""
    markers: List[str] = []
    groups: List[Dict[str, Any]] = []
    captured: List[Dict[str, Any]] = []
    for entry in entries:
        marker = str(entry.get("marker") or "")
        if is_scope_name(marker):
            markers.append(marker)
        floor = str(entry.get("scope_start") or "") or _iso_floor(entry.get("opened_at"))
        for group in entry.get("root_groups") or []:
            if isinstance(group, dict) and int(group.get("pgid") or 0) > 1:
                groups.append({"pgid": int(group["pgid"]), "leader_pid": int(group.get("leader_pid") or 0),
                               "leader_birth": str(group.get("leader_birth") or ""), "floor": floor})
        leaders = {int(item.get("pid") or 0): str(item.get("birth") or "")
                   for item in entry.get("captured") or [] if isinstance(item, dict)}
        for item in entry.get("captured") or []:
            if not isinstance(item, dict):
                continue
            pid, pgid, birth = int(item.get("pid") or 0), int(item.get("pgid") or 0), str(item.get("birth") or "")
            if pid > 1 and birth:
                captured.append({"pid": pid, "birth": birth})
            if pgid > 1 and all(group["pgid"] != pgid for group in groups):
                groups.append({"pgid": pgid, "leader_pid": pgid, "leader_birth": leaders.get(pgid, ""), "floor": floor})
    return markers, groups, captured


def _iso_floor(opened_at: Any) -> str:
    """An entry's ISO ``opened_at`` as the lstart text the membership floor compares (whole
    seconds; a naive time is UTC); ``""`` when absent or not ISO 8601. An instant the birth zone
    cannot spell keeps its ISO text: recorded, unreadable, so its group stays owed."""
    text = str(opened_at or "").strip()
    try:
        when = _dt.datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return ""
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    return lstart_text(when.timestamp()) or text


# --------------------------------------------------------------------------- Panic

def panic_sweep_pending_path(data_dir: Any) -> pathlib.Path:
    return _state_dir(data_dir) / PANIC_SWEEP_PENDING_FILE


def _installation_census(data_dir: Any, *, exclude: Iterable[int], rescans: int,
                         known: Iterable[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    marker = installation_marker(data_dir, create=False)
    task_markers: List[str] = []
    groups: List[Dict[str, Any]] = []
    task_known: List[Dict[str, Any]] = []  # identities task cleanups still owe: Panic's to end too
    for record in task_scope_records(data_dir):
        task_markers.extend(_record_markers(record))
        groups.extend(record.get("groups") or [])
        task_known.extend(_saved_members(record))
    engine_entries, engine = engine_scope_index(data_dir)
    engine_markers, engine_groups, captured = _engine_scope_targets(engine_entries)
    known = list(known)
    outcome = end_scope(marker=marker, markers=[*task_markers, *engine_markers], groups=[*groups, *engine_groups],
                        captured=[*captured, *known, *task_known], exclude=exclude, grace_sec=0.0, term_first=False,
                        rescans=rescans)
    # The index decides completion: a pass whose index was unreadable, invalid or of a newer
    # schema may have missed recorded members, so it is not settled whatever the census said.
    outcome["settled"] = bool(outcome["settled"]) and engine.get("status") in _INDEX_COMPLETE_STATUSES
    outcome.update(marker_role="installation", scope="installation", recorded_groups=len(groups), engine_index=engine,
                   engine_groups=len(engine_groups), engine_captured=len(captured), known_members=len(known),
                   task_known=len(task_known), residual=PANIC_CENSUS_RESIDUAL)
    return outcome


def _compact_outcome(outcome: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: outcome.get(key) for key in ("scope", "pending_record", "coverage", "ended", "already_gone", "survivors", "unverified",
                                              "captured_gone", "census_failed", "settled", "engine_index", "residual")}


def _record_panic_outcome(data_dir: Any, outcome: Mapping[str, Any], *, phase: str) -> bool:
    """The Panic census outcome in the supervisor journal (pid, birth, reason: never environment
    bytes). Only the writer's own ``True`` (a failed write returns ``False``) lets a pending record retire."""
    try:
        return append_jsonl(pathlib.Path(data_dir) / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "panic_scope_census", "phase": phase, **_compact_outcome(outcome)}) is True
    except Exception:
        log.warning("Panic scope census outcome could not be journalled", exc_info=True)
        return False


def _settle_pending_record(data_dir: Any, outcome: Dict[str, Any], *, phase: str) -> None:
    """Retire only a recorded, settled (for its stated scope) or unsupported outcome.
    Otherwise carry the last outcome and remaining pid/birth identities to the next pass."""
    pending = panic_sweep_pending_path(data_dir)
    recorded = _record_panic_outcome(data_dir, outcome, phase=phase)
    final = outcome.get("coverage") == COVERAGE_UNSUPPORTED
    outcome["recorded"] = recorded
    if recorded and (outcome.get("settled") or final):
        try:
            pending.unlink(missing_ok=True)
            outcome["pending_retired"] = True
            return
        except OSError:
            log.warning("panic sweep pending record could not be retired; it is kept", exc_info=True)
    outcome["pending_retired"] = False
    known = _known_members(outcome)
    try:
        update_json_locked(pending, lambda current: {**(current or {}), "last_outcome": _compact_outcome(outcome),
                                                     "known_members": known, "last_attempt_at": utc_now_iso(),
                                                     "last_phase": phase,
                                                     "attempts": int((current or {}).get("attempts") or 0) + 1})
    except (OSError, TimeoutError, ValueError):
        log.warning("panic sweep pending record could not be updated", exc_info=True)


def panic_scope_census(data_dir: Any, *, exclude: Optional[Iterable[int]] = None) -> Dict[str, Any]:
    """Panic's one more bounded request attempt: SIGKILL every live member of the
    installation scope (marker ∪ recorded task groups and the identities their cleanups still
    owe ∪ the engine's recorded runs) by re-measured identity, excluding the server, its
    launcher and pid 1. Write the pending record FIRST. The launcher's observed Panic exit
    may finish this installation-wide attempt; later starts retry only saved identities.
    Earlier known members ride along. Retirement requires a recorded settled/final outcome;
    nothing here makes a later start an extension of this Panic's discovery authority."""
    marker = installation_marker(data_dir, create=False)
    if not marker:
        return {"requested": False, "reason": "no installation marker planted"}
    excluded = set(panic_census_exclusions() if exclude is None else {int(pid) for pid in exclude})
    pending = panic_sweep_pending_path(data_dir)
    known = _saved_members(read_json_dict(pending))
    try:
        pending.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(pending, {"requested_at": utc_now_iso(), "marker_role": "installation",
                                    "server_pid": os.getpid(), "excluded": sorted(excluded), "known_members": known})
    except OSError:
        log.warning("panic sweep pending record could not be written", exc_info=True)
    outcome = _installation_census(data_dir, exclude=excluded, rescans=1, known=known)
    _settle_pending_record(data_dir, outcome, phase="panic")
    return {"requested": True, **outcome}


def _panic_record_readable(record: Optional[Mapping[str, Any]]) -> bool:
    """A malformed member list cannot be filtered into an empty, retireable obligation."""
    if record is None:
        return False
    excluded, known = record.get("excluded", []), record.get("known_members", [])
    return (isinstance(excluded, list) and all(type(pid) is int for pid in excluded)
            and isinstance(known, list) and all(
                isinstance(item, dict) and type(item.get("pid")) is int and item["pid"] > 1
                and isinstance(item.get("birth"), str) and bool(item["birth"].strip()) for item in known))


def complete_pending_panic_sweep(data_dir: Any, *, exclude: Optional[Iterable[int]] = None,
                               installation: bool = False) -> Optional[Dict[str, Any]]:
    """Retry saved pid/birth identities, not membership discovered after manual return.

    Only the launcher's observed Panic-exit branch passes ``installation=True`` to finish
    that active Panic's broad attempt. No persisted field or flag grants it to a later boot.
    Current controller, parent and pid 1 remain excluded. ``None`` means no pending file."""
    pending = panic_sweep_pending_path(data_dir)
    record = read_json_dict(pending)
    if record is None and not pending.exists():
        return None
    readable = _panic_record_readable(record)
    saved = set(record.get("excluded", [])) if readable else set()
    known = _saved_members(record) if readable else []
    excluded = saved | {os.getpid(), os.getppid(), 1} | {int(pid) for pid in (exclude or ())}
    if installation:
        outcome = _installation_census(data_dir, exclude=excluded, rescans=1, known=known)
    else:
        outcome = end_scope(captured=known, exclude=excluded, grace_sec=0.0, term_first=False, rescans=0)
        outcome.update(marker_role="installation", scope="known_members", known_members=len(known),
                       engine_index={"status": "not_read"}, residual=PANIC_KNOWN_RESIDUAL)
    if readable:
        _settle_pending_record(data_dir, outcome, phase="completion")
    else:  # keep the exact unreadable record; only a new Panic may replace its obligation
        outcome.update(pending_record="unreadable", settled=False, pending_retired=False)
        outcome["recorded"] = _record_panic_outcome(data_dir, outcome, phase="completion")
    log.warning("completed the pending Panic scope sweep: ended=%s survivors=%s unverified=%s census_failed=%s "
                "retired=%s", outcome["ended"], outcome["survivors"], outcome["unverified"],
                outcome["census_failed"], outcome.get("pending_retired"))
    return outcome


__all__ = [
    "COVERAGE_POSIX", "COVERAGE_UNSUPPORTED", "ENGINE_RUN_SCOPES_FILE", "INSTALL_SCOPE_PREFIX", "RUN_SCOPE_PREFIX",
    "SCOPE_NAME_RE", "SCOPE_VALUE", "TASK_END_GRACE_SEC", "TASK_SCOPE_PREFIX", "birth_epoch", "birth_text",
    "census", "complete_pending_panic_sweep", "end_held_processes", "end_owned_processes", "end_scope",
    "engine_scope_index", "initialize_installation_scope", "installation_marker", "installation_scope_entries",
    "is_scope_name", "leader_facts", "lstart_text", "measure_birth", "mint_marker", "note_task_terminal_transition",
    "observe_owned_processes",
    "outcome_digest", "panic_census_exclusions", "panic_scope_census", "panic_sweep_pending_path",
    "parse_proc_stat", "plant_installation_scope", "process_cleanup_facts", "read_identities", "record_task_group",
    "scope_entries", "scope_members", "settle_task_scope", "settle_task_scopes", "shed_scopes", "task_scope_env",
    "task_scope_marker", "task_scope_record", "task_scope_records", "task_scope_terminal_evidence",
    "wait_for_cleanups",
]
