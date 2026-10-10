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


def recover_terminal_task_files(drive_root, protected):
    from ouroboros import obligations as o
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.headless import prepare_terminal_task_files, terminal_task_files_ready
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
    for task_id in sorted(candidates.keys() - protected):
        try:
            validate_task_id(task_id)
            current = load_task_result(root, task_id, strict=True) or {}
            for child_root in own_child_drives(root, task_id):
                if not child_root.is_dir() or child_root.is_symlink() or child_root.parent.is_symlink():
                    continue
                task = {**current, "id": task_id, "drive_root": str(child_root)}
                ready = terminal_task_files_ready(root, task, current)
                pending = _has_pending_ref_promotion(current.get("child_ref_promotion"))
                if ready:
                    o.drive_finished(root, task, current)
                    continue
                source = load_task_result(child_root, task_id, strict=True) or {}
                if source.get("status") not in SETTLED_STATUSES:
                    if (current.get("status") == "scheduled" and source.get("status") == "running"
                            and source.get("started_at") and not source.get("_is_direct_chat")
                            and not cancel_pending(root, task_id, strict=True)):
                        observed = effective_task_result(root, {**current, "child_drive_root": str(child_root)},
                                                         materialize_artifacts=False)
                        if observed.get("reason_code") == "orphaned_running_after_worker_restart":
                            write_task_result(root, task_id, "running", child_drive_root=str(child_root),
                                budget_drive_root=str(root), started_at=source["started_at"],
                                ts=source.get("ts") or source["started_at"])
                            report.setdefault("rebound", []).append(task_id)
                    if pending or current.get("status") == "completed":
                        report["unresolved"].append(task_id)
                    continue
                task = {**source, **current, "id": task_id, "drive_root": str(child_root)}
                prepared = prepare_terminal_task_files(root, task)
                settled = prepared.get("result")
                if prepared["error"] or not terminal_task_files_ready(root, task, settled):
                    report["unresolved"].append(task_id)
                else:
                    report["recovered"].append(task_id)
            if current and terminal_task_files_ready(root, {**current, "id": task_id}, current):
                o.drive_finished(root, {**current, "id": task_id}, current)
        except Exception as exc:
            report["unresolved"].append(task_id)
            report["errors"].append(f"{task_id}: {type(exc).__name__}: {exc}")
    return report
