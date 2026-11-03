"""Addressed startup file recovery; historical drive discovery belongs to import."""
from pathlib import Path


def startup_tree_exclusions(root):
    """Follow only current recovery candidates and their own child-result stores.

    An unreadable/missing link leaves destructive housekeeping owed. Completed
    results outside this closure are never enumerated to construct exclusions.
    """
    from ouroboros.obligations import members
    from ouroboros.task_custody import own_child_drives
    from ouroboros.task_results import list_task_results, load_task_result

    protected = set()
    try:
        pending = set(members(root, "nonterminal")) | set(members(root, "pending_drives"))
        for facts in members(root, "unknowns").values():
            if not facts.get("task_id"):
                return None
            pending.add(str(facts["task_id"]))
        while pending:
            task_id = pending.pop()
            if task_id in protected:
                continue
            protected.add(task_id)
            canonical = load_task_result(root, task_id, strict=True)
            rows = [canonical] if canonical else []
            for child in own_child_drives(root, task_id):
                rows.extend(list_task_results(child, strict=True))
            if not rows:
                return None
            for row in rows:
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                for field in ("task_id", "parent_task_id", "root_task_id", "retry_task_id", "superseded_by",
                              "original_task_id", "timeout_retry_from"):
                    linked = str(row.get(field) or metadata.get(field) or "")
                    if linked and linked not in protected:
                        pending.add(linked)
    except Exception:
        return None
    return protected


def _drive_stamps(root, task_id):
    """Each own child drive of TASK_ID with its store's identity (the database file and the commits
    its WAL log still holds, ``store.log_identity``): what a resting recovery waits on."""
    from ouroboros.store import log_identity
    from ouroboros.task_custody import own_child_drives

    stamps = []
    for drive in own_child_drives(root, task_id):
        database = drive / "state" / "current.sqlite"
        try:
            info = database.stat()
            stamps.append([str(drive), info.st_ino, info.st_size, info.st_mtime_ns, log_identity(database)])
        except OSError:
            stamps.append([str(drive)])
    return stamps


def recover_terminal_task_files(drive_root, protected):
    """Recover terminal files of addressed ``pending_drives``. A settled member whose last pass found only
    facts a restart cannot change -- no own drive left, or a child whose result never settled -- rests
    (``obligations.DutyRest``) on its row and its drives' store identities, reported as it was found."""
    from ouroboros import obligations as o
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.headless import (
        _child_drive_from_task,
        prepare_terminal_task_files,
        task_requires_workspace_capture,
        terminal_task_files_ready,
    )
    from ouroboros.observability import _has_pending_ref_promotion
    from ouroboros.task_custody import own_child_drives
    from ouroboros.task_results import load_task_result, validate_task_id, write_task_result
    from ouroboros.task_status import SETTLED_STATUSES, effective_task_result

    root = Path(drive_root)
    report = {"recovered": [], "unresolved": [], "protected": sorted(protected), "errors": []}
    try:
        candidates = o.members(root, "pending_drives")
    except Exception as exc:
        report["errors"].append(str(exc))
        report["unresolved"].append("*")
        return report
    stamps = {}
    for task_id in candidates:
        try:
            stamps[task_id] = {"drives": _drive_stamps(root, validate_task_id(task_id))}
        except (OSError, ValueError):
            continue  # examined below, which reports why
    rest = o.DutyRest(root, "drive_recovery", stamps)
    for task_id in sorted(candidates.keys() - protected):
        acted, waits, unresolved = False, set(), len(report["unresolved"])
        try:
            validate_task_id(task_id)
            child = _child_drive_from_task(candidates[task_id])
            canonical = child is None or child.resolve(strict=False) == root.resolve(strict=False)
            rested = rest.resting.get(task_id)
            # A canonical capture never depended on an absent child. Re-examine
            # that old rest only for addressed members with no split-drive source.
            if rested and not (canonical and rested.get("waits_on") == ["child_drive_absent"]):
                if (rested.get("report") or {}).get("unresolved"):
                    report["unresolved"].append(task_id)
                continue
            current = load_task_result(root, task_id, strict=True) or {}
            for child_root in own_child_drives(root, task_id):
                if not child_root.is_dir() or child_root.is_symlink() or child_root.parent.is_symlink():
                    continue
                task = {**current, "id": task_id, "drive_root": str(child_root)}
                ready = terminal_task_files_ready(root, task, current)
                pending = _has_pending_ref_promotion(current.get("child_ref_promotion"))
                if ready:
                    o.drive_finished(root, task, current)
                    acted = True
                    continue
                source = load_task_result(child_root, task_id, strict=True) or {}
                if source.get("status") not in SETTLED_STATUSES:
                    waits.add("child_result_unsettled")
                    if (current.get("status") == "scheduled" and source.get("status") == "running"
                            and source.get("started_at") and not source.get("_is_direct_chat")
                            and not cancel_pending(root, task_id, strict=True)):
                        observed = effective_task_result(root, {**current, "child_drive_root": str(child_root)},
                                                         materialize_artifacts=False, _boot=True)
                        if observed.get("reason_code") == "orphaned_running_after_worker_restart":
                            write_task_result(root, task_id, "running", child_drive_root=str(child_root),
                                budget_drive_root=str(root), started_at=source["started_at"],
                                ts=source.get("ts") or source["started_at"])
                            report.setdefault("rebound", []).append(task_id)
                            acted = True
                    if pending or current.get("status") == "completed":
                        report["unresolved"].append(task_id)
                    continue
                task = {**source, **current, "id": task_id, "drive_root": str(child_root)}
                acted = True
                prepared = prepare_terminal_task_files(root, task)
                settled = prepared.get("result")
                if prepared["error"] or not terminal_task_files_ready(root, task, settled):
                    report["unresolved"].append(task_id)
                else:
                    report["recovered"].append(task_id)
            if current and terminal_task_files_ready(root, {**current, "id": task_id}, current):
                o.drive_finished(root, {**current, "id": task_id}, current)
            elif not acted and current.get("status") in SETTLED_STATUSES:
                child = _child_drive_from_task(current)
                if (not waits and task_requires_workspace_capture(current)
                        and (child is None or child.resolve(strict=False) == root.resolve(strict=False))):
                    # A failed optional fork runs in the canonical store. Its
                    # capture is still owed; a busy publisher must not put it to rest.
                    task = {**current, "id": task_id}
                    prepared = prepare_terminal_task_files(root, task)
                    if prepared["error"] or not terminal_task_files_ready(root, task, prepared.get("result")):
                        report["unresolved"].append(task_id)
                    else:
                        report["recovered"].append(task_id)
                else:
                    rest.wait(task_id, *(waits or {"child_drive_absent"}),
                              report={"unresolved": len(report["unresolved"]) > unresolved})
        except Exception as exc:
            report["unresolved"].append(task_id)
            report["errors"].append(f"{task_id}: {type(exc).__name__}: {exc}")
    rest.commit()
    return report
