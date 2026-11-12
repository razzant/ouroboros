"""Retire review checking copies whose review is over; the same pass migrates older copies.

A copy goes only on positive evidence that its run ended: a receipt of its operation, or
custody, read fresh for that seat, showing that a run started in this copy is terminal and none
still runs or is pending there. Each seat is decided under its cross-process lock, which
workers also take to prepare, refresh and capture it, so a review that starts or ends during
a pass is never deleted under itself. Threads live with their task. A held one-shot copy
stays for a live author or the Skills continuation door, within recorded run limits. Failed
captures retry; marked copies without a complete receipt stay regardless of age. Only
unmarked materialization leftovers expire; a refresh's staging goes once its seat's lock
is free, and an older body's staging inside a seat at once (a pristine copy). Evidence,
markers and the frozen views needed by uncaptured copies stay.

The views pass runs once per process (``views_since``: when the process began), under the
lock (``.views.lock`` beside ``views/``) a freeze takes to reuse or publish a view. It removes
discarded views, and a view that no seat holding a copy needs and whose last use predates the
process, when a finished seat names it or it is older than the retention period. It does not
run after a pass that stopped early, found a seat busy or could not read a seat used within
the retention period.

The same pass retires an isolated checkout (``state/review_checkouts/<token>/``, made by
``review_subject.isolated_checkout``) that its wave kept for open custody or that a crash left,
once no review can still read it: its task is terminal and not live, no active review custody
(``state/advisory_review.json``) names its round, no live review operation runs for the task and
no seat that still holds a copy checks from it. Before judging, the pass repairs, in one locked
write, the attempts that custody still holds whose seats are proven closed (#2710), so a leftover
of an old writer is released by its own proof, never by age. Its owner is the task its ``checkout.json`` names;
a checkout made before the marker is owned only through a recorded round of that state, and goes
only once older than the retention period. Targeted ``git worktree remove`` retires only this
checkout's registration. A rejoin rebuilds a retired checkout at its frozen tree. Before every POST of a read-only
seat's explicit continuation (``review_subject.restore_released_subject``) its checkout is put back under the
custody locks of its marker's task and the deciding task, the locks this removal takes (``owner_keep.gate``), with a
marker naming the deciding task (a live Continue successor holds it as ``task_live``); a removal re-reads the
marker under its gate and keeps a checkout rewritten since it was judged. An unreadable review state or a busy or
unreadable seat keeps every such checkout; an unreadable marker or task row keeps its own. A gate rejoin's reuse
shares no such lock: one starting after the custody judgment can reuse the tree and then lose it during removal.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Mapping, Optional

from ouroboros.owner_keep import OWNER_KEEP, gate, is_kept
from ouroboros.review_checking import (
    VIEWS_LOCK_NAME,
    _RECEIPTED,
    _capture_locked,
    _checking_area_present,
    _receipt,
    _release_area,
    _seat_locked,
    seat_dir,
)
from ouroboros.session_scratch import release_seat_scratch
from ouroboros.unindexed import unindexed
from ouroboros.workspace_preflight import SESSION_SCRATCH_DIRNAME

_PASS_LOCK = threading.Lock()
_PSEUDO_SEAT_OWNERS = frozenset({"api_skill_review", "marketplace_install"})


def seat_scratch_released(data_root: Any, key: str) -> bool:
    """Declared-disposable scratch follows the checking copy, not its evidence receipt.
    A missing home means no copy; an unreadable path is present, so it retains.
    Existing homes are checked under their seat lock by the caller."""
    home = pathlib.Path(data_root) / "state" / "review_checkouts" / "seats" / key[5:]
    return not _checking_area_present(unindexed(home, "area"))


def _last_use(path: pathlib.Path) -> float:
    """ctime too: a copy's mtime is its source's (copystat); a reuse touches a view."""
    try:
        info = path.lstat()
    except OSError:
        return float("inf")  # unknown counts as recent
    return max(info.st_mtime, info.st_ctime)


def _view_containers(views: pathlib.Path, source: Mapping[str, Any]) -> set:
    """The ``views/`` entries a frozen source names (its folder view and evidence views)."""
    paths = [pathlib.Path(str(path)).resolve(strict=False)
             for path in [source.get("view"), *(source.get("evidence_views") or {}).values()] if path]
    return {views / path.relative_to(views).parts[0] for path in paths if path.is_relative_to(views) and path != views}


def _started_after(started: Any, since: Any) -> bool:
    try:
        return datetime.fromisoformat(str(started)) >= datetime.fromisoformat(str(since))
    except (TypeError, ValueError):
        return False  # an unknown start proves nothing about this copy


def _copy_run_state(data_root: Any, task_id: str, area: pathlib.Path, created_at: Any, snapshot: Any = None) -> tuple:
    """One custody read (or the caller's ``snapshot`` of the task's custody): (runs over, continuation window
    open). False/None terminality retains; only runs born in this copy prove completion. Their recorded
    start/cap bound continuation; no new deadline is invented for unbounded or unknown timing."""
    from ouroboros import delegate_custody_current as current
    from ouroboros.delegate_custody import TERMINAL_STATES
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now

    snapshot = current.task_snapshot(data_root, task_id) if snapshot is None else snapshot
    target = str(area.resolve(strict=False))

    def names(root: Any) -> bool:
        return bool(root) and str(pathlib.Path(str(root)).resolve(strict=False)) == target

    runs = [entry for entry in snapshot["runs"].values() if names(entry.execution_root)]
    if any(names(row.get("execution_root")) for row in snapshot["pending"]) or not all(
            entry.terminal_state in TERMINAL_STATES for entry in runs):
        return False, True
    own = [entry for entry in runs if _started_after(entry.started_ts, created_at)]
    ends = []
    for entry in own:
        start, cap = parse_deadline_ts(entry.window_origin), int(getattr(entry, "max_seconds", 0) or 0)
        if start is not None and cap > 0:
            ends.append(start + timedelta(seconds=cap))
    return (True if own else None), (not ends or min(ends) > utc_now())


def _seat_proven_over(data_root: Any, task_id: str, home: pathlib.Path, live: Any, snapshot: Any = None) -> bool:
    """Use a pseudo owner's seat proof when no task row can prove it over."""
    if task_id not in _PSEUDO_SEAT_OWNERS:
        from ouroboros.subagent_worktrees import _task_proven_over

        return _task_proven_over(task_id, data_root, live)
    try:
        marker = json.loads((home / "checking.json").read_text(encoding="utf-8"))
        for area in (home / "area", home / "area.noindex"):
            over, _window = _copy_run_state(data_root, task_id, area, marker.get("created_at"), snapshot)
            if over is True:
                return True
        return False
    except Exception:
        return False


def _proven(data_root: Any, proven: Dict[Any, Any], task_id: str, home: pathlib.Path, live: Any) -> bool:
    """HOME's seat proof, cached in PROVEN: one per task, or one per home for a pseudo owner, whose own copy's
    runs decide it; such an owner's custody snapshot is read once into PROVEN (an unreadable one proves none)."""
    if task_id not in _PSEUDO_SEAT_OWNERS:
        if task_id not in proven:
            proven[task_id] = _seat_proven_over(data_root, task_id, home, live)
        return proven[task_id]
    key, custody = (task_id, home.name), ("custody", task_id)
    if key not in proven:
        if custody not in proven:
            from ouroboros import delegate_custody_current as current

            try:
                proven[custody] = current.task_snapshot(data_root, task_id)
            except Exception:
                proven[custody] = None
        proven[key] = proven[custody] is not None and _seat_proven_over(data_root, task_id, home, live, proven[custody])
    return proven[key]


def _checkout_entries(base: pathlib.Path) -> list:
    """The ``<token>/`` folders ``review_subject.isolated_checkout`` makes (not ``seats/``, ``views/`` or dot names)."""
    try:
        return [entry for entry in sorted(base.iterdir()) if entry.name not in {"seats", "views"}
                and not entry.name.startswith(".") and not entry.is_symlink() and entry.is_dir()]
    except OSError:
        return []


def _review_rounds(data_root: Any, *, repair: bool = False) -> tuple:
    """``(active, owners)`` from the one review state (``state/advisory_review.json``): the checkout
    names of rounds whose custody is still active (the one custody predicate,
    ``review_state._attempt_has_active_review_custody``), and every recorded round's task by its
    checkout name. ``repair`` (the background pass, never Storage → Free, which only reads): the
    attempts the predicate still holds whose exact records prove their seats closed first take that
    repair, in one locked write (``review_change_custody.repair_held_attempts``). A state that cannot
    be read raises: no checkout is judged without it."""
    from ouroboros.review_state import _load_state_unlocked
    from ouroboros.tools.review_subject import checkout_token

    state = _load_state_unlocked(pathlib.Path(data_root), strict_attempt_authority=True)
    if repair:
        from ouroboros.tools.review_change_custody import repair_held_attempts

        state = repair_held_attempts(data_root, state)

    def name(attempt: Any) -> str:
        return checkout_token(attempt.review_retry_key, task_id=attempt.task_id)

    return ({name(row) for row in state.get_active_attempts() if row.review_retry_key},
            {name(row): str(row.task_id) for row in state.attempts if row.review_retry_key and row.task_id})


def _checkout_hold(data_root: Any, entry: pathlib.Path, *, rounds: tuple, sources: set,
                   live: Callable[[str], Optional[bool]], cutoff: float) -> str:
    """Why the isolated checkout ENTRY stays, or ``""`` when no review can still read it."""
    from ouroboros.review_operation import task_has_live_review_operation
    from ouroboros.store import StoreBusy
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.tools.review_subject import CHECKOUT_MARKER

    active, owners = rounds
    try:
        marker = json.loads((entry / CHECKOUT_MARKER).read_text(encoding="utf-8"))
    except FileNotFoundError:
        marker = None  # made before the marker: owned only through a recorded round, and only past the wait
    except (OSError, ValueError):
        return "marker_unreadable"
    if marker is not None and (not isinstance(marker, dict) or marker.get("schema_version") != 1):
        return "marker_unreadable"
    task_id = str(marker.get("task_id") or "") if marker is not None else owners.get(entry.name, "")
    repo = unindexed(entry, "repo").resolve(strict=False)
    if not task_id:
        return "owner_unknown"
    if is_kept(data_root, task_id):
        return OWNER_KEEP
    if entry.name in active:
        return "review_custody_active"
    if any(root == repo or root.is_relative_to(repo) for root in sources):
        return "seat_checking"
    if marker is None and _last_use(entry) >= cutoff:
        return "younger_than_retention"
    try:
        row = load_task_result(data_root, task_id, strict=True) or {}
    except StoreBusy:
        return "store_busy"
    except (OSError, ValueError, KeyError, TypeError):
        return "task_result_unreadable"
    if str(row.get("status") or "") not in SETTLED_STATUSES:
        return "task_not_terminal"
    try:
        alive = live(task_id)
    except StoreBusy:
        return "store_busy"
    if alive is None:
        return "liveness_unknown"
    if alive:
        return "task_live"
    try:
        if task_has_live_review_operation(data_root, task_id, result_loader=lambda tid: row if tid == task_id else (
                load_task_result(data_root, tid, strict=True) or {})):
            return "operation_live"
    except StoreBusy:
        return "store_busy"
    except (OSError, ValueError, KeyError, TypeError):
        return "task_result_unreadable"
    return ""


def _checkout_registration_gone(repo: pathlib.Path) -> bool:
    """A deleted subject leaves a Git pointer to an absent registration; unreadable is not absent."""
    pointer = (repo / ".git").read_text(encoding="utf-8").strip()
    if not pointer.startswith("gitdir: ") or not pointer[8:]:
        return False
    try:
        (repo / pointer[8:]).stat()
    except FileNotFoundError:
        return True
    return False


def _marker_text(entry: pathlib.Path) -> Optional[str]:
    from ouroboros.tools.review_subject import CHECKOUT_MARKER

    try:
        return (entry / CHECKOUT_MARKER).read_text(encoding="utf-8")
    except OSError:
        return None


def _remove_unless_kept(data_root: Any, entry: pathlib.Path, rounds: tuple, judged: Optional[str]) -> str:
    """The removal under the keep gate of the checkout's task, read as ``_checkout_hold`` reads it. ``judged``: the
    marker as the hold judged it; one rewritten since (a released read-only checkout put back for its continuation
    under the same gate, ``review_subject.restore_released_subject``) is a new checkout, kept for the next pass."""
    current = _marker_text(entry)
    try:
        task_id = str(json.loads(current or "").get("task_id") or "")
    except (ValueError, AttributeError):
        task_id = rounds[1].get(entry.name, "")
    with gate(data_root, task_id, timeout_sec=0) as why:
        if not why and _marker_text(entry) != judged:
            why = "rebuilt_since_judged"
        return f"{entry.name}: {why}" if why else _remove_checkout(entry)


def _remove_checkout(entry: pathlib.Path) -> str:
    """Remove this checkout after Git unregisters it or its registration is already gone."""
    from ouroboros.subagent_worktrees import _force_rmtree
    from ouroboros.tools.review_subject import _git_bytes

    repo, common = unindexed(entry, "repo"), None
    if (repo / ".git").is_file():
        rc, out, err = _git_bytes(repo, ["rev-parse", "--git-common-dir"])
        text = out.decode("utf-8", "replace").strip()
        common = (repo / text).resolve(strict=False) if rc == 0 and text else None
        if common is None or not common.is_dir():
            if not _checkout_registration_gone(repo):
                return f"{entry.name}: Git common directory unavailable: {err.strip()}"
        else:
            rc, _out, err = _git_bytes(common, ["--git-dir", str(common), "worktree", "remove", "--force", str(repo)])
            if rc:
                return f"{entry.name}: Git worktree removal failed: {err.strip()}"
    _force_rmtree(entry)
    if entry.exists() or entry.is_symlink():
        return f"{entry.name}: the checkout survived its removal"
    return ""


def _retire_checkouts(data_root: Any, entries: list, live: Callable[[], Any], cutoff: float, sources: set,
                      stopped: Callable[[], bool], holds: Any, report: Dict[str, Any]) -> None:
    """Judge and remove the isolated checkouts ENTRIES; ``sources`` are the roots kept seats check from."""
    try:
        rounds = _review_rounds(data_root, repair=True)
    except Exception as exc:  # unknown custody: every checkout stays
        report["errors"].append(f"review state unreadable: {type(exc).__name__}: {exc}")
        report["partial"] = True
        return
    for entry in entries:
        if stopped():
            report["partial"] = True  # the checkouts not judged keep the row's holds
            return
        try:
            judged = _marker_text(entry)  # what the hold reads; a marker rewritten before removal keeps it
            hold = _checkout_hold(data_root, entry, rounds=rounds, sources=sources,
                                  live=lambda task: task in live(), cutoff=cutoff)
            error = "" if hold else _remove_unless_kept(data_root, entry, rounds, judged)
        except Exception as exc:  # an unreadable marker, row or tree keeps the checkout
            hold, error = "", f"{entry.name}: {type(exc).__name__}: {exc}"
        if error:
            report["errors"].append(error)
        elif hold:
            report["checkouts_kept"] += 1
            if hold != "younger_than_retention":
                holds.note(hold, entry.name)
        else:
            report["checkouts_removed"] += 1


def _seat_hold_cause(home: pathlib.Path, proven: bool) -> str:
    """The exact fact behind a grouped seat-home hold."""
    return "seat_checking" if _checking_area_present(unindexed(home, "area")) else "not_proven_over" if not proven else ""


def finished_seat_homes(data_root: Any, live: Callable[[str], Any], stopped: Callable[[], bool],
                        proven: Optional[Dict[Any, Any]] = None, only: Optional[Any] = None,
                        *, causes: Optional[Dict[pathlib.Path, str]] = None) -> list:
    """For the owner's Storage -> Free and the ``record_compression`` job: ``(home, verdict, task_id)`` per
    seat home (``only``: these home names), ``offer`` when its copy is gone and its task is proven over
    (``proven`` caches that per task, per home for a pseudo owner: ``_proven``), else ``in_use`` or ``unreadable``.
    Pure: markers and task rows are read, no seat lock is taken (a lock writes a file)."""
    seats, proven, found = pathlib.Path(data_root) / "state" / "review_checkouts" / "seats", {} if proven is None else proven, []
    for home in sorted(seats.iterdir()) if seats.is_dir() else ():
        if stopped():
            break
        if home.name.startswith(".") or home.is_symlink() or not home.is_dir() or only is not None and home.name not in only:
            continue
        try:
            task_id = str(json.loads((home / "checking.json").read_text(encoding="utf-8")).get("task_id") or "")
        except (OSError, ValueError, AttributeError):
            if causes is not None:
                causes[home] = "marker_unreadable"
            found.append((home, "unreadable", ""))
            continue
        over = _proven(data_root, proven, task_id, home, live)
        bare = over and not _checking_area_present(unindexed(home, "area"))  # its copy is gone
        if causes is not None and not bare:
            causes[home] = _seat_hold_cause(home, over)
        found.append((home, "offer" if bare else "in_use", task_id))
    return found


@contextlib.contextmanager
def finished_home_held(data_root: Any, home: pathlib.Path, live: Callable[[str], Any], proven: Dict[Any, Any],
                       *, causes: Optional[Dict[pathlib.Path, str]] = None):
    """HOME under its seat lock (a thread seat's is its thread's; never waits): yields ``""`` while the home
    is still finished (its copy gone, its task proven over), else ``busy`` or ``in_use``; the lock is held
    until the caller's block ends, so no worker prepares, refreshes or captures it meanwhile."""
    marker = json.loads((home / "checking.json").read_text(encoding="utf-8"))
    task_id, thread = str(marker.get("task_id") or ""), str(marker.get("thread_seat") or "")
    with _seat_locked(seat_dir(data_root, task_id, f"thread:{thread}") if thread else home, wait=False) as held:
        if not held:
            if causes is not None:
                causes[home] = "seat_busy"
            yield "busy"
            return
        over = _proven(data_root, proven, task_id, home, live)
        if causes is not None and (_checking_area_present(unindexed(home, "area")) or not over):
            causes[home] = _seat_hold_cause(home, over)
        yield "in_use" if _checking_area_present(unindexed(home, "area")) or not over else ""


def retire_seat_home(data_root: Any, home: pathlib.Path, live: Callable[[str], Any], proven: Dict[Any, Any]) -> str:
    """Move one finished seat home (its evidence and marker) into the custody trash the maintenance child
    empties, after re-checking it under the seat lock; returns why it stayed, or ``""``. Its task is proven
    over before the owner's keep gate (the probe takes the supervisor's queue lock, which drive settlement
    holds while it waits for custody); inside, only the recorded terminal status is read again."""
    from ouroboros.body_candidate import _recorded_terminal
    marker = json.loads((home / "checking.json").read_text(encoding="utf-8"))
    task_id = str(marker.get("task_id") or "")
    over = _proven(data_root, proven, task_id, home, live)
    with gate(data_root, task_id, timeout_sec=0) as kept:  # outside the seat lock's file: it covers the move too
        if kept:
            return kept
        if over and task_id not in _PSEUDO_SEAT_OWNERS and not _recorded_terminal(task_id, data_root):
            return "in_use"  # continued since it was proven over: no stale proof deletes it
        with finished_home_held(data_root, home, live, proven) as why:
            if why:
                return why
        trash = pathlib.Path(data_root) / "state" / "custody_trash" / f"{home.name}-{uuid.uuid4().hex[:12]}"
        trash.parent.mkdir(parents=True, exist_ok=True)
        home.replace(trash)  # outside the seat lock: Windows moves no folder with an open file
    return ""


def orphan_review_checkouts(data_root: Any, live: Callable[[str], Any], homes: Optional[list] = None,
                            *, causes: Optional[dict] = None) -> list:
    """For the owner's Storage -> Free: ``(checkout, verdict)`` per isolated review checkout, ``offer``
    when no task owns it (made before the marker, named by no recorded round) or the background pass
    would retire it now, else ``owner_keep``, ``in_use`` or ``unreadable``. Pure, like ``finished_seat_homes``."""
    from ouroboros.store import StoreBusy

    entries = _checkout_entries(pathlib.Path(data_root) / "state" / "review_checkouts")
    if not entries:
        return []
    try:
        rounds = _review_rounds(data_root)
    except Exception:  # unknown custody: every checkout is shown, never offered
        if causes is not None:
            causes.update((entry, "review_state_unreadable") for entry in entries)
        return [(entry, "unreadable") for entry in entries]
    sources = set()
    for home, verdict, _task in finished_seat_homes(data_root, live, lambda: False) if homes is None else homes:
        try:
            root = json.loads((home / "checking.json").read_text(encoding="utf-8")).get("source", {}).get("root")
        except (OSError, ValueError, AttributeError):
            root = None
        if verdict != "offer" and root:  # a seat that may still run can read the checkout it came from
            sources.add(pathlib.Path(str(root)).resolve(strict=False))
    found = []
    for entry in entries:
        try:  # an active round first: its marker may name no task yet
            hold = "review_custody_active" if entry.name in rounds[0] else _checkout_hold(
                data_root, entry, rounds=rounds, sources=sources, live=live, cutoff=float("inf"))
        except StoreBusy:
            hold = "store_busy"
        except Exception:
            hold = "review_state_unreadable"
        if causes is not None and hold:
            causes[entry] = hold
        found.append((entry, "offer" if hold in ("", "owner_unknown") else OWNER_KEEP if hold == OWNER_KEEP else
                      "unreadable" if hold in {"marker_unreadable", "review_state_unreadable", "task_result_unreadable"} else "in_use"))
    return found


def retire_orphan_checkouts(data_root: Any, chosen: list, live: Callable[[str], Any],
                            stop: Callable[[], bool]) -> Dict[pathlib.Path, str]:
    """Remove the listed checkouts ``chosen`` still offered, re-judged once under the pass lock: each
    attempted one's reason for staying, ``""`` once removed (a stop leaves the rest unattempted)."""
    if not _PASS_LOCK.acquire(blocking=False):
        return {entry: "busy" for entry in chosen}
    try:
        causes: dict = {}
        judged = {entry: _marker_text(entry) for entry in chosen}  # before the judgment the removal re-checks
        verdicts, outcome = dict(orphan_review_checkouts(data_root, live, causes=causes)), {}
        rounds: Optional[tuple] = None
        for entry in chosen:
            if stop():
                break
            verdict = verdicts.get(entry, "other")
            if verdict == "offer" and rounds is None:
                try:  # once, for a checkout with no marker; unknown custody keeps every checkout, as the listing does
                    rounds = _review_rounds(data_root)
                except Exception:
                    rounds = ()
            if verdict == "offer" and not rounds:
                outcome[entry] = "review_state_unreadable"
                continue
            outcome[entry] = causes.get(entry, verdict) if verdict != "offer" else "other" if _remove_unless_kept(
                data_root, entry, rounds, judged[entry]) else ""
        return outcome
    finally:
        _PASS_LOCK.release()


def prune_review_checkouts(data_root: Any, live_task_ids: Any, *, views_since: Optional[float] = None,
                           retention_days: Optional[int] = None, now: Optional[float] = None,
                           stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """One pass over ``state/review_checkouts`` (see the module docstring).

    ``live_task_ids`` is a set or a callable asked again for each thread seat; ``stop`` is
    asked before each seat. ``views_done`` reports a completed views pass, ``partial`` one whose
    checkouts were not judged (a busy or unread seat, a stop, an unreadable review state)."""
    from ouroboros import delegate_custody_current as current
    from ouroboros.maintenance_jobs import Holds
    from ouroboros.retention import age_cutoff, get_gc_retention_days

    base = (pathlib.Path(data_root) / "state" / "review_checkouts").resolve(strict=False)
    seats, view_root = base / "seats", base / "views"
    views = views_since is not None
    holds = Holds()  # the shared retirer shape (maintenance_jobs.record_pass): each kept copy by its typed reason
    report: Dict[str, Any] = {"released": 0, "removed": 0, "kept": 0, "busy": 0, "capture_failed": 0,
                              "checkouts_removed": 0, "checkouts_kept": 0, "errors": [], "holds": holds.rows()}
    checkouts = _checkout_entries(base)
    if not (seats.is_dir() or checkouts or views and view_root.is_dir()):
        return {**report, "views_done": views}
    if not _PASS_LOCK.acquire(blocking=False):
        return {**report, "skipped": "pass_in_progress", "rested": True}
    try:
        if current.unreadable(data_root):
            return {**report, "skipped": "custody_unreadable", "rested": True}
        live = live_task_ids if callable(live_task_ids) else (lambda: live_task_ids)
        stopped = stop or (lambda: False)
        if views:  # existing once-per-process recovery pass; no periodic historical scan
            from ouroboros.workspace_preflight import restore_seat_scratch_notes
            restore_seat_scratch_notes(data_root, stop=stopped)
        cutoff = age_cutoff(get_gc_retention_days() if retention_days is None else retention_days, now)
        named, kept_views, unread, busy, sources = set(), set(), [], [], set()

        def remove(path: pathlib.Path) -> None:
            if not _checking_area_present(path):
                return
            error = _release_area(path, {}).get("cleanup_error")
            if error:
                report["errors"].append(error)
            else:
                report["removed"] += 1

        def scratch(home: pathlib.Path, task_id: str) -> None:
            """Under the seat lock: the scratch records of a home whose copy is gone go to the custody trash
            (a capture inside this pass already hands them over with the copy; this covers a copy released
            earlier whose record stayed)."""
            report["errors"].extend(release_seat_scratch(data_root, task_id, home)["errors"])

        def retire(home: pathlib.Path, area: pathlib.Path) -> bool:
            """Under the seat lock: True keeps the copy (and the views it names)."""
            marker = json.loads((home / "checking.json").read_text(encoding="utf-8"))
            # A copy handed to an explicit successor is that holder's: its receipt and capture decide.
            task_id = str(marker.get("task_id") or "")
            operation = str(marker.get("holder_operation_id") or marker.get("operation_id") or "")
            thread = str(marker.get("thread_seat") or "")
            thread_home = bool(thread) and seat_dir(data_root, task_id, f"thread:{thread}").resolve(
                strict=False) == home
            over, window_open = _copy_run_state(data_root, task_id, area, marker.get("created_at"))
            # Finished managed authors cannot continue this seat. The canonical Skills
            # endpoint can; its explicit continuation still enforces the original limits.
            continuation = bool(marker.get("held_for_continuation")) and window_open and (
                task_id == "api_skill_review" or task_id in live())
            if over is False or continuation or thread_home and task_id in live():
                report["kept"] += 1
                holds.note("run_open" if over is False else "task_live", home.name)
                return True
            receipt = _receipt(data_root, task_id, operation) if operation else {}
            if (over or receipt.get("status") == "capture_failed") and operation and receipt.get("status") not in _RECEIPTED:
                _capture_locked(data_root, task_id, operation)  # the run ended unobserved
                receipt = _receipt(data_root, task_id, operation)  # only a durable receipt counts
                if not receipt:  # nothing was saved (a full volume): the copy waits for a later pass
                    report["capture_failed"] += 1
                    holds.note("capture_failed", home.name)
                    return True
            if receipt.get("status") in _RECEIPTED:
                with gate(data_root, task_id, timeout_sec=0) as held:  # the owner's keep holds a captured copy too
                    if held:
                        report["kept"] += 1
                        holds.note(held, home.name)
                        return True
                    receipt = _release_area(area, receipt, marker.get("git"))
                    if not receipt.get("cleanup_error"):
                        scratch(home, task_id)  # the seat's scratch goes with its copy (#2706, #2707)
                if receipt.get("cleanup_error"):
                    report["errors"].append(receipt["cleanup_error"])
                    holds.note("removal_failed", home.name)
                    return True
                report["released"] += 1
                return False
            # Age proves neither execution completion nor retention of unique
            # work. A failed or unknown capture keeps the copy and its basis.
            report["capture_failed" if receipt else "kept"] += 1
            holds.note("capture_failed" if receipt else "capture_unproven", home.name)
            return True

        def staging(entry: pathlib.Path) -> None:
            """``.next-<home>.<nonce>``: a refresh holds its home's lock while it builds one."""
            with _seat_locked(seats / entry.name[len(".next-"):].rsplit(".", 1)[0], wait=False) as held:
                if held:
                    remove(entry)

        def seat(home: pathlib.Path) -> set:
            """One seat home; returns the views it still needs."""
            area = unindexed(home, "area")
            if home.is_symlink():  # never followed; it names no copy this pass could own
                if views:
                    report["errors"].append(f"{home.name}: the seat home is a link")
                return set()
            if not views and not _checking_area_present(area) and not os.path.lexists(home / SESSION_SCRATCH_DIRNAME):
                return set()  # a periodic pass reads (marker included) only seats that still hold a copy or a scratch record
            try:
                marker = json.loads((home / "checking.json").read_text(encoding="utf-8"))
                thread, task_id = marker.get("thread_seat"), marker.get("task_id")
            except (OSError, ValueError, AttributeError):
                thread, task_id = "", ""
            lock_home = seat_dir(data_root, task_id, f"thread:{thread}") if thread else home
            with _seat_locked(lock_home, wait=False) as held:  # same owner for active and relocated thread turns
                if not held:  # a worker is changing it; the views it needs are read in a later pass
                    report["busy"] += 1
                    holds.note("seat_busy", home.name)
                    busy.append(home.name)
                    return set()
                try:
                    source = json.loads((home / "checking.json").read_text(encoding="utf-8")).get("source") or {}
                except FileNotFoundError:
                    if _last_use(area) < cutoff:  # a materialization that never wrote its marker
                        remove(area)
                    for leftover in home.glob(".next-*"):  # an older body's staging: never a reviewer's work
                        remove(leftover)
                    return set()
                except (OSError, ValueError, AttributeError):  # unknown basis must keep the possible frozen views too
                    raise
                views_of = _view_containers(view_root, source)
                named.update(views_of)
                try:
                    for leftover in home.glob(".next-*"):  # an older body's interrupted thread refresh
                        remove(leftover)
                    if not _checking_area_present(area):
                        if not thread:  # a copy already gone holds no scratch (a handoff a crash interrupted)
                            scratch(home, str(task_id or ""))
                        return set()
                    kept = retire(home, area)
                except Exception as exc:  # its views are known and stay; the views pass need not wait for it
                    report["errors"].append(f"{home.name}: {type(exc).__name__}: {exc}")
                    kept = True
                if kept and source.get("root"):  # a copy that stays may still read the checkout it came from
                    sources.add(pathlib.Path(str(source["root"])).resolve(strict=False))
                return views_of if kept else set()

        for home in sorted(seats.iterdir()) if seats.is_dir() else ():
            if stopped():
                return {**report, "skipped": "stopped", "rested": True}
            try:
                if home.name.startswith(".next-"):
                    staging(home)
                elif home.is_symlink() or home.is_dir():  # a stray file (.DS_Store) is not a seat
                    kept_views |= seat(home)
            except Exception as exc:  # a seat whose views are unknown never stops the pass; it keeps every view
                unread.append(home.name)
                report["errors"].append(f"{home.name}: {type(exc).__name__}: {exc}")
        if checkouts and not unread and not busy and not stopped():
            _retire_checkouts(data_root, checkouts, live, cutoff, sources, stopped, holds, report)
        elif checkouts:  # a seat could not be judged: the checkouts wait, and the row keeps their holds
            report["partial"] = True
        if views and not unread and not stopped():
            doomed = []
            with _seat_locked(base, wait=False, name=VIEWS_LOCK_NAME) if view_root.is_dir() else contextlib.nullcontext(
                    True) as held:
                if not held or busy:
                    return {**report, "skipped": "views_busy"}
                for view in sorted(view_root.iterdir()) if view_root.is_dir() else ():
                    if view in kept_views or not (view.is_symlink() or view.is_dir()):  # a stray file is no view
                        continue
                    if view.name.startswith(".discarded-"):
                        doomed.append(view)
                    elif _last_use(view) < views_since and (view in named or _last_use(view) < cutoff):
                        aside = view_root / f".discarded-{view.name.lstrip('.')}.{uuid.uuid4().hex[:8]}"
                        try:
                            view.rename(aside)  # gone at once for a freeze that takes the lock next
                        except OSError as exc:  # held open on Windows: the next process start retries it
                            report["errors"].append(f"{view.name}: {type(exc).__name__}: {exc}")
                            continue
                        doomed.append(aside)
            for view in doomed:
                remove(view)
            report["views_done"] = True
        return report
    except Exception as exc:  # housekeeping never breaks startup or maintenance
        return {**report, "errors": [*report["errors"], f"{type(exc).__name__}: {exc}"]}
    finally:
        _PASS_LOCK.release()
