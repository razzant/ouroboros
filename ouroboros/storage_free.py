"""Dashboard -> Storage -> Temporary space: the owner's free-space action (B).

A pure *preview* lists what the app's own retirers would remove now, in two groups: G1, the work
folders of finished tasks (direct and helper drives whose row is terminal and whose task the
supervisor's liveness probe proves not live), and G2, working copies of finished work (plain
worktrees with the same proof and their unique work proven saved (``copy_capture_proof``), delegated snapshots by the custody GC's own proof
(``subagent_worktrees.snapshot_released``: task proven over, custody closed and readable), own-body
candidates whose owner is terminal with no live lineage process). The preview changes nothing:
no file under the data root (the task rows are read through the read-only fact reader, never the
drive pass's projection fill), no event row, no settlement call; its sizes come from the storage
count child (``python -m ouroboros.storage --size-paths``, one JSON string per line each way)
with progress and Stop, and the plan is held in memory under an id. A folder the child could
not list keeps an unknown size (``None``, never zero) and the group says so. A *run* frees
exactly the previewed items of the chosen groups through the existing retirers with the wait
set to zero (owner decision Q1 = A; the background pass keeps its days): G1 through
``headless.prune_task_drives``/``prune_headless_task_drives`` (``settle_child_drive`` moves a
proven drive into ``state/custody_trash``; the maintenance child deletes it, asked at once
through ``maintenance.child.request``), G2 through ``subagent_worktrees.prune_orphans``/
``prune_execution_snapshots``, one copy per call with a stop check between calls (each call
decides under that copy's record lock, as the custody cadence does: the record, pin and own-body
verdict logic live there, and a move-aside would need a second record state). It runs on its own
thread and stops between items on the owner's Stop, the server's stop or a closed supervisor
generation (``_closed``), with no reconcile latch; concurrency is the retirers' own locks. A
moved folder counts as removed only once its trash entry is gone, "still being deleted"
meanwhile, and "stuck" once the maintenance child's ``custody_trash`` job recorded that the
entry survived its removal (the child retries it at its next run); free space is read
fresh on every projection. Words (labels, loss sentences) live in the client with the other
storage copy; this module reports ids, counts, sizes and typed reasons.

The ordinary action works on exact objects: the *pool* (``storage_free_view``) is every offered work folder,
working copy and declared temporary environment (``copy_environments``), each addressed alone; everything else
is listed with its reason. Selecting by ``items`` retires those objects, each re-judged under its retirer's own
lock and the owner's keep (``owner_keep``); ``groups`` stays for the groups below, with their own words.
``manual`` is a third, separate form: exact finished copies whose work no capture holds, each named with what it
would lose (``view=manual``) and sent with ``accept_loss``; nothing else reaches them (``copy_capture_proof.discard_offer``,
``copy_capture_proof.discard_unsaved_copy``).

Five more groups hold what no background retirer removes, so only the owner's press frees them
(the client starts those holding saved work unchecked): G3, reviewer seat homes of tasks proven
over whose copy is gone (``review_checking_prune.finished_seat_homes``; moved into the custody
trash like G1); G4, isolated review checkouts no task owns (``orphan_review_checkouts``); G5,
folders in the copies' root no copy record names; G6, the environment folder of a copy that is
gone and finished test-run records there; G7, the two Git pins of a writer's saved result whose
task is over without applying or rejecting it (sized by ``git rev-list --disk-usage``; the
record and the saved patch stay, the space returns at Git's next prune).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from ouroboros.storage_free_view import (  # noqa: F401 -- public re-exports
    StorageFreeEntry, StorageFreeItemsResponse, StorageFreeKept, StorageFreeManual, StorageFreePool, UNSAVED_LISTED, VIEWS, manual_ids,
    ordinary_ids, outcomes, page, pool,
)
from ouroboros.utils import utc_now_iso

try:  # Python 3.11+
    from typing import NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import NotRequired, TypedDict  # type: ignore[assignment]

log = logging.getLogger(__name__)

GROUP_WORK_FOLDERS = "work_folders"
GROUP_WORKING_COPIES = "working_copies"
GROUP_REVIEW_EVIDENCE = "review_evidence"
GROUP_REVIEW_CHECKOUTS = "review_checkouts"
GROUP_LOST_COPIES = "lost_copies"
GROUP_SERVICE_FOLDERS = "service_folders"
GROUP_RESULT_PINS = "result_pins"
GROUPS = (GROUP_WORK_FOLDERS, GROUP_WORKING_COPIES, GROUP_REVIEW_EVIDENCE, GROUP_REVIEW_CHECKOUTS,
          GROUP_LOST_COPIES, GROUP_SERVICE_FOLDERS, GROUP_RESULT_PINS)
LARGEST_ITEMS = 20
# A retirer's typed reason -> the owner's reason (the client's sentences): results or messages
# still being saved (late mail and call records that moved under the pass included: the next pass
# carries them), in use again, busy right now, the task record or its mailbox unreadable, a restart.
_KEPT_REASONS = {
    "results_pending": ("child_result_unadopted", "child_refs_pending", "call_inventory_pending", "call_inventory_changed",
                        "inputs_uncustodied", "unread_mailbox_uncustodied", "verification_receipts_uncustodied",
                        "canonical_changed", "drive_occupant_changed", "canonical_not_settled", "post_work_open",
                        "artifacts_not_terminal", "child_drive_mismatch"),
    "in_use": ("task_live", "liveness_unknown", "owner_processes_live", "owner_not_terminal", "task_not_terminal", "parent_not_terminal",
               "not_proven_over", "custody_held", "owner_paused"),
    "busy": ("custody_busy", "mailbox_busy", "store_busy", "busy", "record_busy"),
    "unreadable_record": ("task_result_unreadable", "unread_mailbox_unreadable", "record_unreadable"),
    "capture_unproven": ("capture_unproven", "capture_incomplete"),
    "retained_evidence": ("retained_process_environment",),
    "changed": ("changed", "nothing_unsaved", "nested_repository", "commits_unprotected"),
    "unproven_excluded_inputs": ("unproven_excluded_inputs",),
    "owner_keep": ("owner_keep",),
    "restart": ("generation_closed",),
}
_RETIRER_STAMPS = ("scratch_retention", "retiring", "removal_failed", "removal_failed_identity")  # no change of ownership


class StorageFreeItem(TypedDict):
    id: str
    name: str  # the task's title when it has one, else the folder name
    bytes: Optional[int]


class StorageFreeGroup(TypedDict):
    id: str  # one of GROUPS
    count: int
    inside_wait: int  # finished less than retention_days ago (the background would still wait)
    in_use: int  # live, pending, unknown or non-terminal: offered nowhere, named here
    unreadable: int  # no readable task record (or unreadable custody): shown, never freed
    bytes: Optional[int]
    bytes_complete: bool
    items: List[StorageFreeItem]  # the largest, at most LARGEST_ITEMS


class StorageFreeRun(TypedDict):
    groups: List[str]
    selected: int
    done: int
    removed: int
    removed_bytes: int
    deleting: int  # moved aside, the maintenance child has not deleted them yet
    deleting_bytes: int
    stuck: int  # moved aside and survived the child's removal; retried at the child's next run
    kept: List[StorageFreeKept]
    failed: int  # the removal itself failed; not in ``kept``
    not_started: int  # stopped before it: untouched
    outcomes_ready: bool  # ``view=outcomes`` answers for this run
    free_before: Optional[int]
    free_after: Optional[int]
    stopped: bool
    error: Optional[str]


class StorageFree(TypedDict):
    state: Literal["idle", "listing", "ready", "running", "done", "failed", "stopped"]
    plan: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    done: int  # folders sized so far (listing) or attempted (running)
    total: int
    groups: List[StorageFreeGroup]
    pool: Optional[StorageFreePool]  # the ordinary action's pool: null until a plan exists
    run: Optional[StorageFreeRun]
    error: Optional[str]


class StorageFreeRequest(TypedDict):
    """``POST /api/storage/free``: the plan id the preview answered with and what to free of it:
    exact ``items`` of the ordinary pool, whole ``groups``, or the exact ``manual`` discards with
    ``accept_loss``; one form per request."""

    plan: str
    groups: NotRequired[List[str]]  # ids of GROUPS
    items: NotRequired[List[str]]  # ids of ``view=pool``
    manual: NotRequired[List[str]]  # ids of ``view=manual``
    accept_loss: NotRequired[bool]  # exactly true: the listed unsaved paths of ``manual`` are lost


class _Stopped(Exception):
    """The owner or the generation stopped the preview or the run."""


class _Candidate:
    __slots__ = ("group", "key", "path", "layout", "inside_wait", "bytes", "unreadable", "name", "task", "row",
                 "owner", "kind", "ordinary", "reason", "eligible_after", "manual")

    def __init__(self, group: str, key: str, path: pathlib.Path, layout: str, *, inside_wait: bool = False,
                 task: str = "", row: Optional[Dict[str, Any]] = None, owner: str = "") -> None:
        self.group, self.key, self.path, self.layout, self.inside_wait = group, key, path, layout, inside_wait
        self.bytes: Optional[int] = None  # None until sized, and when the folder could not be listed
        self.unreadable = 0  # entries inside it the sizing could not read
        self.name = key
        self.task = task  # the owner task its name is read from; none: the folder's name
        self.row = row or {}  # a copy record (G2 copies, G7 pins)
        self.owner = owner or task  # the task whose keep holds it; a leftover's name may carry one
        self.kind = {GROUP_WORK_FOLDERS: "work_folder", GROUP_WORKING_COPIES: "working_copy"}.get(group, "other")
        self.ordinary = group in (GROUP_WORK_FOLDERS, GROUP_WORKING_COPIES)  # the ordinary action's pool
        self.reason: Optional[str] = None  # why it stays out of the pool
        self.eligible_after: Optional[str] = None
        self.manual: Optional[Dict[str, Any]] = None  # ``discard_offer``'s answer, its largest paths and ``unsaved_count``,
        # when the owner may discard it by hand


def _sorted_into(root: pathlib.Path, group: Dict[str, Any], verdict: str, candidate: "_Candidate", reason: str = "") -> None:
    """A lister's verdict: offered, or named with its reason; in use or unreadable (shown, never freed)."""
    from ouroboros.owner_keep import OWNER_KEEP, is_kept

    if verdict == "offer" and is_kept(root, candidate.owner):
        verdict, reason = "in_use", OWNER_KEEP
    if verdict == "offer":
        group["candidates"].append(candidate)
        return
    group["unreadable" if verdict == "unreadable" else "in_use"] += 1
    candidate.reason = reason or ("unreadable_record" if verdict == "unreadable" else verdict if verdict == OWNER_KEEP else "in_use")
    candidate.ordinary = False
    group["excluded"].append(candidate)


def _eligible_after(instant: Optional[float], days: int) -> Optional[str]:
    if instant is None:
        return None
    return datetime.fromtimestamp(float(instant) + days * 86400, tz=timezone.utc).isoformat()


def _run_finished(run: pathlib.Path) -> bool:
    """A test-run record whose run ended: its result is written, or its supervisor is provably gone."""
    from ouroboros.platform_layer import pid_provably_gone

    if (run / "result.json").is_file():
        return True
    try:
        pid = json.loads((run / "owner.json").read_text(encoding="utf-8")).get("supervisor_pid")
        return isinstance(pid, int) and not isinstance(pid, bool) and pid_provably_gone(pid)
    except (OSError, ValueError, AttributeError):
        return False


def _pins_size(row: Dict[str, Any]) -> Optional[int]:
    """Bytes of the objects only this copy's pins reach (Git 2.31+); None when Git cannot say."""
    from ouroboros.subagent_worktrees import isolated_git_env

    refs = [str(row.get(key) or "") for key in ("baseline_ref", "result_ref") if row.get(key)]
    try:
        proc = subprocess.run(["git", "rev-list", "--disk-usage", "--objects", *refs, "--not",
                               *(f"--exclude={ref}" for ref in refs), "--all"],
                              cwd=str(row.get("git_dir") or row.get("target_root") or "."), capture_output=True,
                              text=True, timeout=120, env={**isolated_git_env(), "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return int(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip().isdigit() else None


def _free_bytes(root: pathlib.Path) -> Optional[int]:
    try:
        return shutil.disk_usage(root).free
    except OSError:
        return None


def _name_items(root: pathlib.Path, items: List["_Candidate"]) -> None:
    """Every listed object's name: its task's title when it has one, else the folder's; one display read
    (no bulky evidence) per page of ids."""
    from ouroboros.task_store import display_documents

    tasks = sorted({item.task for item in items if item.task})
    documents: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(tasks), 500):
        try:
            documents.update((task, json.loads(text)) for task, text in display_documents(root, tasks[start:start + 500]).items())
        except Exception:
            log.debug("Task titles unreadable: folder names stand", exc_info=True)
    for item in items:
        document = documents.get(item.task) or {}
        item.name = str(document.get("title") or document.get("suggested_name") or item.task or item.path.name).strip() or item.path.name


def _reason(raw: str) -> str:
    raw = str(raw or "")
    if raw.startswith("custody_error"):
        return "unreadable_record" if "ValueError" in raw else "other"
    for reason, codes in _KEPT_REASONS.items():
        if raw in codes:
            return reason
    return "other"


class FreeSpace:
    """The server's one free-space owner: a preview, then a run, one at a time (module docstring).
    ``live`` is the supervisor's liveness probe and ``guard`` its ownership interlock; both resolve
    to the supervisor's own at call time unless a test supplies them."""

    def __init__(self, *, live: Optional[Callable[[str], Optional[bool]]] = None, guard: Any = None,
                 offered: tuple = GROUPS) -> None:
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._cancel = threading.Event()
        self._live, self._guard, self._offered = live, guard, tuple(offered)
        self._thread: Optional[threading.Thread] = None
        self._process = None
        self._state: str = "idle"
        self._plan: Optional[Dict[str, Any]] = None
        self._run: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._progress = {"done": 0, "total": 0}

    # ---- the probe and the interlock ------------------------------------------------------

    def _probe(self) -> Callable[[str], Optional[bool]]:
        if self._live is not None:
            return self._live
        from supervisor.queue import task_settlement_liveness

        return task_settlement_liveness

    def _interlock(self, stop: Callable[[], bool]) -> Any:
        if self._guard is not None:
            return lambda: self._guard(stop=stop)
        from supervisor.queue import task_settlement_interlock

        return lambda: task_settlement_interlock(stop=stop)

    def _closed(self) -> bool:
        """The owner's Stop, the server's stop, or a closed supervisor generation."""
        if self._cancel.is_set() or self._stopped.is_set():
            return True
        try:
            from ouroboros.server_maintenance import _stop_requested

            return bool(_stop_requested())
        except Exception:
            return False

    # ---- the projection ---------------------------------------------------------------------

    def projection(self, data_root: Any) -> StorageFree:
        root = pathlib.Path(data_root)
        with self._lock:
            state, plan, run, error = self._state, self._plan, self._run, self._error
            progress = dict(self._progress)
        groups = [self._group_view(group) for group in (plan or {}).get("groups", [])] if plan else []
        view: StorageFree = {
            "state": state,  # type: ignore[typeddict-item]
            "plan": plan["id"] if plan else None,
            "started_at": (plan or {}).get("started_at"),
            "finished_at": (plan or {}).get("finished_at"),
            "done": progress["done"], "total": progress["total"],
            "groups": groups,
            "pool": pool(plan) if plan else None,
            "run": self._run_view(root, run) if run else None,
            "error": error,
        }
        return view

    @staticmethod
    def _group_view(group: Dict[str, Any]) -> StorageFreeGroup:
        candidates: List[_Candidate] = group["candidates"]
        sized = [item for item in candidates if item.bytes is not None]
        ranked = sorted(sized, key=lambda item: -(item.bytes or 0))[:LARGEST_ITEMS]
        return {
            "id": group["id"], "count": len(candidates),
            "inside_wait": sum(1 for item in candidates if item.inside_wait),
            "in_use": group["in_use"], "unreadable": group["unreadable"],
            "bytes": sum(item.bytes or 0 for item in sized) if sized or not candidates else None,
            "bytes_complete": len(sized) == len(candidates) and not any(item.unreadable for item in sized),
            "items": [{"id": item.key, "name": item.name, "bytes": item.bytes} for item in ranked],
        }

    def items(self, data_root: Any, plan_id: Any, view: Any, cursor: Any, limit: Any, version: Any) -> Dict[str, Any]:
        """One page of ``view`` of the held plan, or ``{"error": <reason code>}``: no disk walk per page."""
        root = pathlib.Path(data_root)
        with self._lock:
            state, plan, run = self._state, self._plan, self._run
        if view not in VIEWS or not isinstance(cursor, int) or cursor < 0 or not isinstance(limit, int) or not 1 <= limit <= 500:
            return {"error": "bad_view"}
        if plan is None or str(plan_id or "") != plan["id"]:
            return {"error": "preview_outdated"}
        if state == "listing":
            return {"error": "listing"}
        rows = outcomes(run, self._settled(root, run)) if view == "outcomes" and run else None
        answer = page(plan, str(view), rows, cursor, limit)
        return {"error": "view_changed"} if version and str(version) != answer["version"] else answer

    def _settled(self, root: pathlib.Path, run: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """The run's folders as they stand now, by id: a moved folder is removed once its trash entry is gone,
        stuck once the ``custody_trash`` job recorded that this very entry survived its removal."""
        from ouroboros.task_custody import custody_trash_entries

        entries = custody_trash_entries(root) if run["moved"] else []
        survived = self._stuck_trash(root) if entries else {}
        waiting = {name.rsplit("-", 1)[0] for name, identity in entries if survived.get(name) != identity}
        blocked = {name.rsplit("-", 1)[0] for name, identity in entries if survived.get(name) == identity}
        deleting = [key for key in run["moved"] if key.split(":", 1)[-1] in waiting]
        stuck = [key for key in run["moved"] if key not in deleting and key.split(":", 1)[-1] in blocked]
        removed = [key for key in run["moved"] if key not in deleting and key not in stuck] + run["removed"]
        return removed, deleting, stuck

    def _run_view(self, root: pathlib.Path, run: Dict[str, Any], settled: Optional[Tuple[List[str], ...]] = None) -> StorageFreeRun:
        """The result as it stands now (``_settled``)."""
        removed, deleting, stuck = settled or self._settled(root, run)
        sizes = run["sizes"]
        return {
            "groups": run["groups"], "selected": run["selected"], "done": run["done"],
            "removed": len(removed), "removed_bytes": sum(sizes.get(key) or 0 for key in removed),
            "deleting": len(deleting), "deleting_bytes": sum(sizes.get(key) or 0 for key in deleting),
            "stuck": len(stuck),
            "kept": [{"reason": reason, "count": count} for reason, count in sorted(run["kept"].items()) if count],
            "failed": len(run["failed"]),
            "not_started": sum(1 for item in run["items"] if item.key not in run["kept_items"] and item.key not in run["failed"]
                               and item.key not in removed and item.key not in deleting and item.key not in stuck),
            "outcomes_ready": True,
            "free_before": run["free_before"],
            "free_after": _free_bytes(root),
            "stopped": run["stopped"], "error": run["error"],
        }

    @staticmethod
    def _stuck_trash(root: pathlib.Path) -> Dict[str, Any]:
        """The ``custody_trash`` job's record of entries that survived its removal (name -> the
        identity it saw); nothing when the row is absent or the store cannot be read now."""
        try:
            from ouroboros import maintenance_jobs

            row = maintenance_jobs.job(root, "custody_trash") or {}
            stuck = row.get("stuck")
            return stuck if isinstance(stuck, dict) else {}
        except Exception:
            return {}

    # ---- lifecycle --------------------------------------------------------------------------

    def _busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def preview(self, data_root: Any, repo_dir: Any) -> bool:
        """Start one preview unless one lists or a run runs; False means the caller joined it."""
        with self._lock:
            if self._stopped.is_set() or self._busy():
                return False
            self._cancel.clear()
            self._state, self._plan, self._run, self._error = "listing", None, None, None
            self._progress = {"done": 0, "total": 0}
            self._thread = threading.Thread(target=self._list, args=(pathlib.Path(data_root), pathlib.Path(repo_dir)),
                                            name="storage-free-preview", daemon=True)
            self._thread.start()
        return True

    def start(self, data_root: Any, repo_dir: Any, plan_id: Any, groups: Any, items: Any = None, manual: Any = None,
              accept_loss: Any = False) -> Dict[str, Any]:
        """Free the previewed objects under PLAN_ID: exact ITEMS of the pool, whole GROUPS, or the exact
        MANUAL discards with ACCEPT_LOSS exactly true: ``{"started": bool, "reason": str}``. Any bad id
        refuses the whole request before any effect."""
        if sum(form is not None for form in (items, groups, manual)) > 1:
            return {"started": False, "reason": "selection_conflict"}
        chosen = [str(group) for group in groups] if isinstance(groups, list) else None
        with self._lock:
            if self._stopped.is_set() or self._busy():
                return {"started": False, "reason": "busy"}
            if self._plan is None or self._state != "ready" or str(plan_id or "") != self._plan["id"]:
                return {"started": False, "reason": "preview_outdated"}
            known = {group["id"] for group in self._plan["groups"]}
            selected: Optional[set] = None
            if manual is not None:
                refusal, discards = self._exact_manual(self._plan, manual, accept_loss)
                if refusal:
                    return {"started": False, "reason": refusal}
                self._cancel.clear()
                self._state, self._error = "running", None
                self._progress = {"done": 0, "total": len(discards)}
                self._thread = threading.Thread(target=self._free, args=(pathlib.Path(data_root), self._plan, [], None, discards),
                                                name="storage-free-run", daemon=True)
                self._thread.start()
                return {"started": True, "reason": ""}
            if items is not None:
                refusal, selected = self._exact(self._plan, items)
                if refusal:
                    return {"started": False, "reason": refusal}
                chosen = [group["id"] for group in self._plan["groups"]
                          if any(item.key in selected for item in group["candidates"])]
            elif not chosen or any(group not in known for group in chosen):
                return {"started": False, "reason": "unknown_group" if chosen else "nothing_selected"}
            self._cancel.clear()
            self._state, self._error = "running", None
            self._progress = {"done": 0, "total": sum(sum(1 for item in group["candidates"] if selected is None or item.key in selected)
                                                     for group in self._plan["groups"] if group["id"] in chosen)}
            self._thread = threading.Thread(target=self._free, args=(pathlib.Path(data_root), self._plan, chosen, selected),
                                            name="storage-free-run", daemon=True)
            self._thread.start()
        return {"started": True, "reason": ""}

    @staticmethod
    def _exact(plan: Dict[str, Any], items: Any) -> Tuple[str, set]:
        """The selection of exact pool ids, or why it is refused; a saved-work group is no ordinary item."""
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            return "unknown_item", set()
        if not items:
            return "nothing_selected", set()
        pooled = ordinary_ids(plan)
        known = {item.key for group in plan["groups"] for item in [*group["candidates"], *group["excluded"]]}
        if any(item not in known for item in items):
            return "unknown_item", set()
        return ("not_offered", set()) if any(item not in pooled for item in items) else ("", set(items))

    @staticmethod
    def _exact_manual(plan: Dict[str, Any], ids: Any, accept_loss: Any) -> Tuple[str, set]:
        """The exact discards, or why they are refused: only ``view=manual`` ids, and only with the loss accepted."""
        if accept_loss is not True:
            return "loss_not_accepted", set()
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            return "unknown_item", set()
        if not ids:
            return "nothing_selected", set()
        known = {item.key for group in plan["groups"] for item in [*group["candidates"], *group["excluded"]]}
        if any(item not in known for item in ids):
            return "unknown_item", set()
        return ("not_offered", set()) if set(ids) - manual_ids(plan) else ("", set(ids))

    def cancel(self) -> bool:
        """Stop the listing (nothing is kept) or the run (what was freed stays freed)."""
        with self._lock:
            if not self._busy():
                return False
            self._cancel.set()
            process = self._process
        if process is not None:
            with contextlib.suppress(OSError):
                process.kill()
        return True

    def stop(self) -> None:
        self._stopped.set()
        self.cancel()

    def live_children(self) -> list:
        process, thread = self._process, self._thread
        if process is not None and process.poll() is None:
            return [process]
        return [thread] if thread is not None and thread.is_alive() else []

    # ---- the preview ------------------------------------------------------------------------

    def _candidates(self, root: pathlib.Path) -> List[Dict[str, Any]]:
        from ouroboros import body_candidate, owner_keep
        from ouroboros.copy_capture_proof import discard_offer, plain_copy_verdict
        from ouroboros.copy_environments import environment_verdict, scratch_owner
        from ouroboros.headless import HEADLESS_TASKS_DIR, TASK_DRIVES_DIR, drive_candidates
        from ouroboros.private_copy_read import _copy_owner, read_records
        from ouroboros.retention import age_cutoff, get_gc_retention_days
        from ouroboros.subagent_worktrees import (
            _record_file, _record_name, _records_dir, _resolve_root, _task_proven_over, snapshot_released,
        )
        from ouroboros.task_status import FINAL_STATUSES
        from ouroboros.test_environment import SCRATCH_MARKER

        live = self._probe()
        days = get_gc_retention_days()
        cutoff = age_cutoff(days)
        groups: List[Dict[str, Any]] = []

        def new_group(group_id: str) -> Dict[str, Any]:
            return {"id": group_id, "candidates": [], "excluded": [], "in_use": 0, "unreadable": 0}

        if GROUP_WORK_FOLDERS in self._offered:
            group = new_group(GROUP_WORK_FOLDERS)
            for layout, base in (("headless", root / HEADLESS_TASKS_DIR), ("direct", root / TASK_DRIVES_DIR)):
                if not base.is_dir():
                    continue
                for task_id, status, at, inside, fact in drive_candidates(base, root, cutoff, self._closed, record=False):
                    if self._closed():
                        raise _Stopped()
                    item = _Candidate(GROUP_WORK_FOLDERS, f"{layout}:{task_id}", base / task_id, layout, inside_wait=inside, task=task_id)
                    item.eligible_after = _eligible_after(at, days)
                    # No readable row: shown, never freed.
                    _sorted_into(root, group, "unreadable" if not fact else "in_use" if status not in FINAL_STATUSES
                                 or live(task_id) is not False else "offer", item)
            groups.append(group)
        read = read_records(root)  # one passive read per copy: no import, no event row; unreadable ones are named
        rows = list(read.rows)
        open_ids: Optional[set] = None
        if any(row.get("kind") == "delegated_exec" for row in rows):
            try:
                from ouroboros.delegate_custody import open_snapshot_ids

                open_ids = open_snapshot_ids(root)
            except Exception:
                log.debug("Delegated custody unreadable: snapshots stay out of the preview", exc_info=True)
        copies_group: Optional[Dict[str, Any]] = None
        if GROUP_WORKING_COPIES in self._offered:
            group = copies_group = new_group(GROUP_WORKING_COPIES)
            for row in rows:
                if self._closed():
                    raise _Stopped()
                path = pathlib.Path(str(row.get("path") or ""))
                if not row.get("path") or not path.exists():
                    continue  # nothing on disk to free; the cadence forgets the row
                kind, task_id = str(row.get("kind") or ""), str(row.get("task_id") or "")
                if kind == "delegated_exec":
                    # Offered by the retirer's own proof (D6): a snapshot whose task is not proven
                    # over, or that custody holds, is in use; unreadable custody is shown, never freed.
                    snapshot = str(row.get("snapshot_id") or "")
                    item = _Candidate(GROUP_WORKING_COPIES, f"snapshot:{snapshot}", path, "delegated", task=task_id, row=row)
                    item.eligible_after = _eligible_after(float(row.get("created_at") or 0), 0)
                    kept = owner_keep.is_kept(root, task_id)  # the proof reads the keep too: name it, not "in use"
                    _sorted_into(root, group, "in_use" if kept else "unreadable" if open_ids is None else "offer"
                                 if snapshot_released(row, open_ids, data_dir=root, live=live) else "in_use", item,
                                 owner_keep.OWNER_KEEP if kept else "")
                    continue
                item = _Candidate(GROUP_WORKING_COPIES, f"{kind or 'plain'}:{_record_name(path)}", path, kind or "plain",
                                  task=task_id, row=row)
                item.eligible_after = _eligible_after(max(float(row.get("created_at") or 0), float(row.get("used_at") or 0)), days)
                over = body_candidate._recorded_terminal(task_id, root) and not body_candidate.live_lineage_processes(root, task_id)
                reason = ""
                if over and kind == "body_candidate":
                    # The retirer's own order (``retention_verdict``), its scratch witness left unwritten.
                    reason = ("retained_evidence" if body_candidate._retained_scratch(row, classify=True, record_witness=False)
                              else "unproven_excluded_inputs" if body_candidate.unique_work(row).get("excluded") else "")
                elif over:
                    over = live(task_id) is False
                    reason = plain_copy_verdict(row, root)["reason"] if over else ""
                    reason = reason if reason == "capture_unproven" else ""
                reason = "owner_keep" if reason and owner_keep.is_kept(root, task_id) else reason  # the keep is what holds it now
                _sorted_into(root, group, "in_use" if not over or reason else "offer", item, reason)
                if reason in ("capture_unproven", "unproven_excluded_inputs"):
                    offer = discard_offer(row, root, live, _resolve_root())  # held for the plan's life: the largest paths only
                    item.manual = None if offer["refusal"] else {**offer, "unsaved": offer["unsaved"][:UNSAVED_LISTED],
                                                                 "unsaved_count": len(offer["unsaved"])}
            groups.append(group)
        if {GROUP_REVIEW_EVIDENCE, GROUP_REVIEW_CHECKOUTS} & set(self._offered):
            from ouroboros import review_checking_prune as seats

            homes = seats.finished_seat_homes(root, live, self._closed)  # one proof per seat task for both groups
            if GROUP_REVIEW_EVIDENCE in self._offered:
                group = new_group(GROUP_REVIEW_EVIDENCE)
                for home, verdict, task_id in homes:
                    _sorted_into(root, group, verdict, _Candidate(GROUP_REVIEW_EVIDENCE, f"seat:{home.name}", home, "seat", task=task_id))
                groups.append(group)
            if GROUP_REVIEW_CHECKOUTS in self._offered:
                group = new_group(GROUP_REVIEW_CHECKOUTS)
                for entry, verdict in seats.orphan_review_checkouts(root, live, homes):
                    _sorted_into(root, group, verdict, _Candidate(GROUP_REVIEW_CHECKOUTS, f"checkout:{entry.name}", entry, "checkout"))
                groups.append(group)
            if self._closed():
                raise _Stopped()
        from ouroboros.unindexed import COPIES

        # Older copies sit at the root, new ones in COPIES, each with its environment and test runs beside it:
        # one name can stand in both, so each id below is the folder's full path (``_record_name``).
        folders = [_resolve_root(), _resolve_root() / COPIES]
        listing = [entry for folder in folders if folder.is_dir() and not folder.is_symlink()
                   and {GROUP_LOST_COPIES, GROUP_SERVICE_FOLDERS} & set(self._offered)
                   for entry in sorted(folder.iterdir()) if entry.name != COPIES]
        recorded = {str(pathlib.Path(str(row["path"])).resolve(strict=False)) for row in rows if row.get("path")}
        named_unreadable: set = set()
        if GROUP_LOST_COPIES in self._offered:
            # A copy's record precedes its folder, so a folder no record names is no copy being made.
            group = new_group(GROUP_LOST_COPIES)
            for entry in listing:
                if (entry.name.startswith(".") or entry.name.endswith(".env") or entry.is_symlink()
                        or not entry.is_dir() or str(entry.resolve(strict=False)) in recorded):
                    continue
                owner = _copy_owner(entry.name)
                item = _Candidate(GROUP_LOST_COPIES, f"lost:{_record_name(entry)}", entry, "lost", owner=owner)
                if read.unlistable or _record_file(entry, root).name in read.unreadable:
                    named_unreadable.add(_record_file(entry, root).name)
                    _sorted_into(root, group, "unreadable", item)  # a record may name it: unread is not unregistered
                    continue
                _sorted_into(root, group, "in_use" if live(owner) is not False else "offer", item)
            groups.append(group)
        if copies_group is not None:  # a record whose folder this listing did not reach is still named
            for name in sorted(read.unreadable - named_unreadable):
                _sorted_into(root, copies_group, "unreadable",
                             _Candidate(GROUP_WORKING_COPIES, f"record:{name}", _records_dir(root) / name, "record"))
        if GROUP_SERVICE_FOLDERS in self._offered:
            group = new_group(GROUP_SERVICE_FOLDERS)
            for entry in listing:  # a copy's environment folder beside it, once the copy is gone
                if entry.name.endswith(".env") and entry.is_dir() and not entry.is_symlink() \
                        and not os.path.lexists(entry.with_name(entry.name[:-len(".env")])):
                    declared = str((scratch_owner(entry) or {}).get("task_id") or "")
                    owner = declared or _copy_owner(entry.name[:-len(".env")])
                    item = _Candidate(GROUP_SERVICE_FOLDERS, f"env:{_record_name(entry)}", entry, "env", owner=owner, task=declared)
                    why = environment_verdict(entry, root, live)
                    item.ordinary = not why
                    if not why:
                        item.kind = "environment"
                        item.eligible_after = _eligible_after((entry / SCRATCH_MARKER).stat().st_mtime, days)
                    _sorted_into(root, group, "in_use" if live(owner) is not False else "offer", item, why if why == "in_use" else "")
            for runs in (folder / ".test-evidence" for folder in folders):  # runs inside copies without a task drive
                for run in sorted(runs.iterdir()) if runs.is_dir() and not runs.is_symlink() else ():
                    if run.is_dir() and not run.is_symlink():
                        _sorted_into(root, group, "offer" if _run_finished(run) else "in_use",
                                     _Candidate(GROUP_SERVICE_FOLDERS, f"test_run:{_record_name(run)}", run, "test_run"))
            groups.append(group)
        if GROUP_RESULT_PINS in self._offered:
            # Held by custody (no disposition) while its task is over and its checkout is gone: only the pins remain.
            group = new_group(GROUP_RESULT_PINS)
            for row in rows:
                snap = str(row.get("snapshot_id") or "")
                if row.get("kind") != "delegated_exec" or not (row.get("baseline_ref") or row.get("result_ref")) \
                        or os.path.lexists(str(row.get("path") or "")) or (open_ids is not None and snap not in open_ids):
                    continue  # a checkout is G2's; a snapshot custody releases goes with the custody GC
                verdict = "unreadable" if open_ids is None else "offer" if _task_proven_over(
                    str(row.get("task_id") or ""), root, live) else "in_use"
                _sorted_into(root, group, verdict, _Candidate(GROUP_RESULT_PINS, f"pins:{snap}", pathlib.Path(
                    str(row.get("path") or "")), "pins", task=str(row.get("task_id") or ""), row=row))
            groups.append(group)
        return groups

    def _size(self, root: pathlib.Path, repo_dir: pathlib.Path, candidates: List[_Candidate]) -> None:
        """Size every candidate folder in the count child and every pin pair through Git; each answer
        lands as it is known."""
        from ouroboros.startup_historical_audit import read_job_lines, spawn_maintenance_job
        from ouroboros.storage import PURPOSE

        by_path = {str(item.path): item for item in candidates if item.layout != "pins"}
        with self._lock:
            self._progress = {"done": 0, "total": len(candidates)}
        for item in candidates:
            if item.layout == "pins":
                if self._closed():
                    raise _Stopped()
                item.bytes = _pins_size(item.row)
                with self._lock:
                    self._progress["done"] += 1
        if not by_path:
            return
        process = spawn_maintenance_job(root, repo_dir, PURPOSE, args=("--size-paths",), stdin=subprocess.PIPE)
        with self._lock:
            self._process = process
        try:
            if self._closed():
                process.kill()
            try:  # one JSON string per line: ASCII on the pipe, the exact path back (storage.main)
                process.stdin.write("".join(f"{json.dumps(path)}\n" for path in by_path).encode("ascii"))
                process.stdin.close()
            except (OSError, ValueError):
                pass

            def on_line(raw: bytes) -> None:
                try:
                    line = json.loads(raw)
                except ValueError:
                    return
                if not isinstance(line, dict):
                    return
                if line.get("event") == "size" and str(line.get("path")) in by_path:
                    item, size = by_path[str(line["path"])], line.get("bytes")
                    item.bytes = size if isinstance(size, int) and not isinstance(size, bool) else None
                    item.unreadable = line.get("unreadable") if isinstance(line.get("unreadable"), int) else 0
                    with self._lock:
                        self._progress["done"] += 1
                elif "status" in line and line.get("status") != "finished":
                    raise RuntimeError(str(line.get("message") or "The sizes could not be read."))

            exit_code = read_job_lines(process, on_line, lambda proc: proc.kill())
        finally:
            with self._lock:
                self._process = None
        if self._closed():
            raise _Stopped()
        if exit_code != 0:
            raise RuntimeError("The sizes could not be read.")

    def _list(self, root: pathlib.Path, repo_dir: pathlib.Path) -> None:
        started = utc_now_iso()
        try:
            groups = self._candidates(root)
            plan = {"id": uuid.uuid4().hex[:12], "started_at": started, "finished_at": None, "groups": groups}
            with self._lock:
                self._plan = plan  # the counts appear at once; the sizes fill in while the state stays listing
            candidates = [item for group in groups for item in group["candidates"]]
            self._size(root, repo_dir, candidates + [item for group in groups for item in group["excluded"] if item.manual])
            _name_items(root, [item for group in groups for item in [*group["candidates"], *group["excluded"]]])
            with self._lock:
                self._plan, self._state = {**plan, "finished_at": utc_now_iso()}, "ready"
        except _Stopped:
            with self._lock:
                self._state, self._plan = "idle", None
        except Exception as exc:
            from ouroboros.launch_roots import WorkRootUnavailable  # its sentence names the setting that decides

            log.warning("Storage free preview failed", exc_info=True)
            with self._lock:
                self._state, self._plan = "failed", None
                authored = isinstance(exc, (RuntimeError, WorkRootUnavailable))
                self._error = str(exc) if authored else f"{type(exc).__name__}"

    # ---- the run ----------------------------------------------------------------------------

    def _free(self, root: pathlib.Path, plan: Dict[str, Any], chosen: List[str], selected: Optional[set] = None,
              discards: Optional[set] = None) -> None:
        from ouroboros.store import supervisor_row

        started = time.monotonic()
        lists = {group["id"]: [item for item in group["candidates"] if selected is None or item.key in selected]
                 for group in plan["groups"] if group["id"] in chosen}
        manual = [item for group in plan["groups"] for item in group["excluded"] if item.key in (discards or ())]
        items = [item for listed in lists.values() for item in listed] + manual
        run: Dict[str, Any] = {"groups": chosen, "selected": len(items), "done": 0, "moved": [], "removed": [], "kept": {},
                               "kept_items": {}, "failed": [], "items": items,
                               "sizes": {item.key: item.bytes for item in items}, "free_before": _free_bytes(root),
                               "stopped": False, "error": None}
        with self._lock:
            self._run = run
        try:
            for group_id, listed in lists.items():
                free = {GROUP_WORK_FOLDERS: self._free_drives, GROUP_WORKING_COPIES: self._free_copies,
                        GROUP_REVIEW_EVIDENCE: self._free_seats, GROUP_REVIEW_CHECKOUTS: self._free_checkouts,
                        GROUP_RESULT_PINS: self._free_pins}.get(group_id, self._free_leftovers)
                free(root, listed, run)
            self._free_manual(root, manual, run)
        except _Stopped:
            run["stopped"] = True
        except Exception as exc:
            log.warning("Storage free run failed", exc_info=True)
            run["error"] = f"{type(exc).__name__}"
        with self._lock:
            self._state = "stopped" if run["stopped"] else "failed" if run["error"] else "done"
            self._plan = {**plan, "finished_at": utc_now_iso()}
        settled = self._settled(root, run)
        view = self._run_view(root, run, settled)
        self._record(root, view, time.monotonic() - started, settled, list(run["failed"]), dict(run["kept_items"]),
                     {item.key: {"unsaved_count": item.manual["unsaved_count"], "unsaved_bytes": item.manual["unsaved_bytes"]}
                      for item in manual})
        supervisor_row(root, "storage_free", {"phase": self._state, "duration_seconds": round(time.monotonic() - started, 2),
                                               **{key: view[key] for key in ("groups", "selected", "removed", "removed_bytes",
                                                                             "deleting", "stuck", "kept", "failed", "not_started",
                                                                             "free_before", "free_after")}})

    @staticmethod
    def _record(root: pathlib.Path, view: StorageFreeRun, seconds: float, settled: Tuple[List[str], ...],
                failed: List[str], kept_ids: Dict[str, str], discards: Optional[Dict[str, Any]] = None) -> None:
        try:
            from ouroboros.server_maintenance import _prune_event

            # The folders by the ids the preview named them with (a work folder: ``layout:task_id``), as the
            # cadence's own prune names what it removed: who removed a task's folder is then one search.
            removed, deleting, stuck = settled
            _prune_event("storage_free", ("removed", "deleting", "kept"), report={
                **view, "duration_seconds": round(seconds, 2), "removed_ids": removed, "deleting_ids": deleting,
                "stuck_ids": stuck, "failed_ids": failed, "kept_ids": kept_ids,
                **({"manual_discards": discards} if discards else {})})
        except Exception:
            log.debug("storage_free event not recorded", exc_info=True)

    def _tick(self, run: Dict[str, Any], count: int) -> None:
        with self._lock:
            run["done"] = count
            self._progress["done"] = count

    @staticmethod
    def _keep(run: Dict[str, Any], item: _Candidate, reason: str) -> None:
        """One object stays with its reason; a removal that failed for no named reason is ``failed``."""
        if reason == "other":
            run["failed"].append(item.key)
            return
        run["kept"][reason] = run["kept"].get(reason, 0) + 1
        run["kept_items"][item.key] = reason

    def _free_drives(self, root: pathlib.Path, candidates: List[_Candidate], run: Dict[str, Any]) -> None:
        from ouroboros import maintenance
        from ouroboros.headless import prune_headless_task_drives, prune_task_drives

        live, stop = self._probe(), self._closed
        guard = self._interlock(stop)
        by_layout: Dict[str, Dict[str, _Candidate]] = {"headless": {}, "direct": {}}
        for item in candidates:
            by_layout[item.layout][item.key.split(":", 1)[1]] = item
        attempted = run["done"]
        for layout, prune in (("headless", prune_headless_task_drives), ("direct", prune_task_drives)):
            selected = by_layout[layout]
            if not selected:
                continue
            if stop():
                raise _Stopped()
            counted = [0]

            def progress(report: Dict[str, Any]) -> None:
                counted[0] = len(report["pruned"]) + len(report["skipped"]) + len(report["errors"])
                self._tick(run, attempted + counted[0])

            report = prune(root, retention_days=0, only=list(selected), live=live, guard=guard, stop=stop, budget=None,
                           progress=progress)
            attempted += len(report["pruned"]) + len(report["skipped"]) + len(report["errors"])
            self._tick(run, attempted)
            reached = set()
            for row in report["pruned"]:
                reached.add(row["task_id"])
                run["moved"].append(selected[row["task_id"]].key)
            for row in report["skipped"]:
                item = selected.get(row["task_id"])
                if item is None:
                    continue
                reached.add(row["task_id"])
                if not item.path.exists():
                    run["moved"].append(item.key)  # the cadence settled it first; its trash may still exist
                else:
                    self._keep(run, item, _reason(row.get("reason")))
            for row in report["errors"]:
                item = selected.get(row["task_id"])
                if item is None:
                    continue
                reached.add(row["task_id"])
                if not item.path.exists():
                    run["moved"].append(item.key)
                else:
                    self._keep(run, item, "unreadable_record" if "ValueError" in str(row.get("error")) else "other")
            if report["deferred"]:
                raise _Stopped()
            for task_id, item in selected.items():
                if task_id in reached:
                    continue  # the pruner's own verdict stands
                attempted += 1
                if not item.path.exists():
                    run["moved"].append(item.key)  # gone before listing; follow the cadence's trash too
                else:
                    self._keep(run, item, "other")
            self._tick(run, attempted)
        if run["moved"]:
            maintenance.child.request()  # the moved drives' deletion is the maintenance child's job

    def _free_copies(self, root: pathlib.Path, candidates: List[_Candidate], run: Dict[str, Any]) -> None:
        """One retirer call per copy, so Stop lands between copies and each record-lock hold covers
        one checkout; a stop leaves the remaining copies as they were."""
        from ouroboros.subagent_worktrees import (
            SubagentWorktreeRegistryCorrupt, WorktreeOpsLockBusy, find_execution_snapshot, prune_execution_snapshots, prune_orphans,
        )

        live, stop = self._probe(), self._closed
        open_ids: Optional[set] = None
        if any(item.layout == "delegated" for item in candidates):
            try:
                from ouroboros.delegate_custody import open_snapshot_ids

                open_ids = open_snapshot_ids(root)
            except Exception:
                open_ids = None  # unreadable custody: no snapshot is freed
        for item in candidates:
            if stop():
                raise _Stopped()
            task_id, path, reasons = item.task, str(item.row.get("path") or ""), {}
            try:
                if item.layout == "delegated":
                    snapshot = item.key.split(":", 1)[1]
                    if open_ids is not None and find_execution_snapshot(snapshot, root) == item.row:
                        prune_execution_snapshots(open_ids, data_dir=root, only=[snapshot], live=live, reasons=reasons)
                else:
                    prune_orphans(data_dir=root, retention_days=0, live=live, only_rows={path: item.row}, reasons=reasons)
            except WorktreeOpsLockBusy:  # one held copy stays with its reason: the run goes on
                reasons[path] = "record_busy"
            except SubagentWorktreeRegistryCorrupt:
                reasons[path] = "record_unreadable"
            if not item.path.exists():
                run["removed"].append(item.key)
            else:
                self._keep(run, item, self._copy_stays(root, item, task_id, live, reasons.get(path, "")))
            self._tick(run, run["done"] + 1)

    def _free_manual(self, root: pathlib.Path, candidates: List[_Candidate], run: Dict[str, Any]) -> None:
        """The owner's exact discards, one retirer call per copy: a copy that stays says the retirer's reason."""
        from ouroboros.copy_capture_proof import discard_unsaved_copy
        from ouroboros.subagent_worktrees import SubagentWorktreeRegistryCorrupt, WorktreeOpsLockBusy

        live = self._probe()
        for item in candidates:
            if self._closed():
                raise _Stopped()
            path, reasons = str(item.row.get("path") or ""), {}
            try:
                discard_unsaved_copy(item.row, item.manual["signature"], data_dir=root, live=live, reasons=reasons)
            except WorktreeOpsLockBusy:
                reasons[path] = "record_busy"
            except SubagentWorktreeRegistryCorrupt:
                reasons[path] = "record_unreadable"
            if not item.path.exists():
                run["removed"].append(item.key)
            else:
                self._keep(run, item, _reason(reasons.get(path, "")))
            self._tick(run, run["done"] + 1)

    @staticmethod
    def _copy_stays(root: pathlib.Path, item: _Candidate, task_id: str, live: Callable[[str], Optional[bool]], named: str = "") -> str:
        """Why a copy the retirer left stays: the owner kept it, its record changed beyond the retirer's own
        stamps, the retirer named a reason (``named``), else its proof read fresh: in use, its work unproven.
        ``other``, a failed removal, only when every proof still holds."""
        from ouroboros import delegate_custody
        from ouroboros.copy_capture_proof import CAPTURE_UNPROVEN, plain_copy_verdict
        from ouroboros.owner_keep import OWNER_KEEP, is_kept
        from ouroboros.private_copy_read import checkout_registration
        from ouroboros.subagent_worktrees import _record_at, _task_proven_over, snapshot_released, SubagentWorktreeRegistryCorrupt

        def owned(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            return {key: value for key, value in (row or {}).items() if key not in _RETIRER_STAMPS}

        if is_kept(root, task_id):
            return OWNER_KEEP
        if checkout_registration(item.path, root) == "unknown":
            return "unreadable_record"
        try:
            if owned(_record_at(item.path, root, op="free_copy")) != owned(item.row):
                return "changed"
        except SubagentWorktreeRegistryCorrupt:
            return "unreadable_record"
        if named:
            return _reason(named)
        if item.layout == "delegated":
            try:
                if delegate_custody.custody_log_unreadable(root):
                    return "unreadable_record"
                return "other" if snapshot_released(item.row, delegate_custody.open_snapshot_ids(root), data_dir=root, live=live) else "in_use"
            except Exception:
                return "unreadable_record"
        if not _task_proven_over(task_id, root, live):
            return "in_use"
        if item.layout == "body_candidate":
            return "other"
        reason = plain_copy_verdict(item.row, root)["reason"]
        return reason if reason == CAPTURE_UNPROVEN else "other"

    def _settle(self, run: Dict[str, Any], item: _Candidate, reason: str, *, moved: bool = False) -> None:
        """One attempted item of G3-G7: removed (or moved into the trash) when ``reason`` is empty, else kept."""
        if reason:
            self._keep(run, item, reason)
        else:
            run["moved" if moved else "removed"].append(item.key)
        self._tick(run, run["done"] + 1)

    def _free_seats(self, root: pathlib.Path, candidates: List[_Candidate], run: Dict[str, Any]) -> None:
        """G3: each seat home, re-checked under its seat lock, moves into the custody trash."""
        from ouroboros import maintenance
        from ouroboros.review_checking_prune import retire_seat_home

        live, proven, moved = self._probe(), {}, False
        for item in candidates:
            if self._closed():
                raise _Stopped()
            try:
                reason = retire_seat_home(root, item.path, live, proven) if item.path.is_dir() else ""
            except ValueError:
                reason = "unreadable_record"
            except OSError:
                reason = "other"
            moved |= not reason and not item.path.exists()
            self._settle(run, item, reason, moved=True)
        if moved:
            maintenance.child.request()  # the moved homes' deletion is the maintenance child's job

    def _free_checkouts(self, root: pathlib.Path, candidates: List[_Candidate], run: Dict[str, Any]) -> None:
        """G4: re-judged once under the review prune's pass lock, then removed one by one."""
        from ouroboros.review_checking_prune import retire_orphan_checkouts

        outcome = retire_orphan_checkouts(root, [item.path for item in candidates], self._probe(), self._closed)
        for item in candidates:
            if item.path in outcome:
                reason = outcome[item.path]
                self._settle(run, item, "unreadable_record" if reason == "unreadable" else reason)
        if len(outcome) < len(candidates):
            raise _Stopped()

    def _free_leftovers(self, root: pathlib.Path, candidates: List[_Candidate], run: Dict[str, Any]) -> None:
        """G5/G6: folders in the copies' root, each re-checked, then removed in this thread (a linked
        worktree also leaves its repository's registration)."""
        from ouroboros.copy_environments import retire_environment
        from ouroboros.owner_keep import gate
        from ouroboros.private_copy_read import checkout_registration
        from ouroboros.review_checking import CheckingAreaUnavailable, _remove_checking_area
        from ouroboros.subagent_worktrees import _deletable, _resolve_root

        live, copies = self._probe(), _resolve_root()
        for item in candidates:
            if self._closed():
                raise _Stopped()
            path, name = item.path, item.path.name
            if item.ordinary and item.layout == "env":  # the declared environment: the one rule the background goes by
                try:
                    self._settle(run, item, retire_environment(path, root, live, copies) if os.path.lexists(path) else "")
                except OSError:
                    self._settle(run, item, "other")
                continue
            registration = checkout_registration(path, root) if item.layout == "lost" else "absent"
            if item.layout == "test_run":
                holds = not _run_finished(path)
            else:
                holds = live(item.owner) is not False or registration == "registered" or (
                    item.layout == "env" and os.path.lexists(path.with_name(name[:-len(".env")])))
            reason = "unreadable_record" if registration == "unknown" else "in_use" if holds else ""
            if not reason and os.path.lexists(path):  # gone already counts as removed
                try:
                    with gate(root, item.owner, timeout_sec=0) as why:
                        reason = why
                        if not why:
                            if not _deletable(path, copies):
                                raise CheckingAreaUnavailable(f"{path} is not inside the copies' root")
                            _remove_checking_area(path)
                except (CheckingAreaUnavailable, OSError):
                    reason = "other"
            self._settle(run, item, reason)

    def _free_pins(self, root: pathlib.Path, candidates: List[_Candidate], run: Dict[str, Any]) -> None:
        """G7: each held result's two pins, dropped under its disposition and record locks."""
        live = self._probe()
        for item in candidates:
            if self._closed():
                raise _Stopped()
            try:
                reason = _release_pins(root, item.key.split(":", 1)[1], live)
            except Exception:
                log.warning("Pins of %s were not released", item.key, exc_info=True)
                reason = "other"
            self._settle(run, item, reason)


def _release_pins(root: pathlib.Path, snapshot_id: str, live: Callable[[str], Optional[bool]]) -> str:
    """Drop one finished writer's baseline and result pins while custody still holds its result
    undisposed; the record says so and keeps the rest, the saved patch stays. Why not, or ``""``."""
    from ouroboros import delegate_custody as custody
    from ouroboros import subagent_worktrees as sw
    from ouroboros.delegate_custody_current import runs
    from ouroboros.owner_keep import gate
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    entry = next((row for row in runs(root).values() if row.snapshot_id == snapshot_id), None)
    if entry is None:
        return "unreadable_record"
    lock = custody.disposition_lock_path(root, entry)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire_exclusive_file_lock(lock, timeout_sec=0, owner_aware_stale=True)
    if fd is None:
        return "busy"  # an apply, a reject or a collection holds this result
    try:
        row = sw.find_execution_snapshot(snapshot_id, root)
        if row is None or not (row.get("baseline_ref") or row.get("result_ref")):
            return ""  # already gone
        if os.path.lexists(str(row.get("path") or "")) or snapshot_id not in custody.open_snapshot_ids(root):
            return "other"  # a checkout came back, or custody let it go: the custody GC owns it now
        if not sw._task_proven_over(str(row.get("task_id") or ""), root, live):
            return "in_use"
        with sw._record_lock(row["path"], root, op="release_pins", task_id=str(row.get("task_id") or "")), \
                gate(root, row.get("task_id"), timeout_sec=0) as why:
            if why:
                return why
            current = sw._record_at(row["path"], root, op="release_pins") or row
            sw._unpin_snapshot(pathlib.Path(str(current.get("git_dir") or current.get("target_root") or ".")), current)
            sw._put_record({**{key: value for key, value in current.items()
                               if key not in ("baseline_ref", "result_ref", "result_sha", "result_tree")},
                            "pins_released": utc_now_iso()}, root)
        return ""
    finally:
        release_exclusive_file_lock(lock, fd)


free = FreeSpace()


__all__ = [
    "GROUPS",
    "GROUP_LOST_COPIES",
    "GROUP_RESULT_PINS",
    "GROUP_REVIEW_CHECKOUTS",
    "GROUP_REVIEW_EVIDENCE",
    "GROUP_SERVICE_FOLDERS",
    "GROUP_WORKING_COPIES",
    "GROUP_WORK_FOLDERS",
    "FreeSpace",
    "StorageFree",
    "StorageFreeGroup",
    "StorageFreeItem",
    "StorageFreeEntry",
    "StorageFreeItemsResponse",
    "StorageFreeKept",
    "StorageFreeManual",
    "StorageFreePool",
    "StorageFreeRequest",
    "StorageFreeRun",
    "free",
]
