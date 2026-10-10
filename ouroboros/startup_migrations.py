"""Visible lifecycle import/repair of current boot state, never a reader fallback.

Run before producers on every startup door. The completed generation is stamped
only after publication succeeds. Interrupted imports may leave extra candidates;
re-entry classifies the source once again. Unknown records stay addressable and
block destructive tree pruning. ``python -m ouroboros.startup_migrations`` is the
explicit rebuild job, including its historical seal audit (no audit watermark).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from ouroboros import obligations as o
from ouroboros.utils import append_jsonl, read_json_dict, update_json_locked, utc_now_iso

log = logging.getLogger(__name__)
SCHEMA_GENERATION = 7
OBLIGATIONS_GENERATION = 1


def watermarks(root) -> dict:
    path = Path(root) / "state" / "migrations.json"
    if not path.exists():
        return {}
    value = read_json_dict(path)
    if value is None:
        raise ValueError(f"migration watermarks unreadable: {path}")
    return value


def stamp(root, **facts):
    target = Path(root) / "state" / "migrations.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    return update_json_locked(target, lambda current: {**current, **facts}, strict_existing_dict=True)


def _result_records(root):
    """The one enumeration/parse pass, shared with the legacy latch migration."""
    try:
        paths = sorted(path for path in (Path(root) / "task_results").iterdir() if path.suffix == ".json")
    except FileNotFoundError:
        paths = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            yield path.stem, None, type(exc).__name__
            continue
        yield path.stem, row, ""


def migrate_cancel_latches(root, *, records=None, generation=SCHEMA_GENERATION, rebuild=False):
    from ouroboros.cancel_intents import migrate_legacy_cancel_latches

    if not rebuild and watermarks(root).get("cancel_latches_generation") == generation:
        return []
    records = list(_result_records(root)) if records is None else records
    migrated = migrate_legacy_cancel_latches(root, records=records)
    stamp(root, cancel_latches_generation=generation)
    return migrated


def _classify(root, records, *, rebuild=False, replace_results=False):
    from ouroboros.task_result_schema import task_result_schema_refusal
    from ouroboros.task_status import SETTLED_STATUSES
    from ouroboros.headless import terminal_task_files_ready
    from ouroboros.task_custody import own_child_drives

    sets = {name: {} for name in o.BOOT_SETS if name != "custody_open"}
    for tid, row, error in records:
        refusal = error or task_result_schema_refusal(row)
        # The admissible pre-7 latch is stamped by the migration before completion.
        if (refusal == "unstamped_pre_7_0" and isinstance(row, dict)
                and row.get("status") == "cancel_requested" and row.get("task_id") == tid):
            row = {**row, "_schema_version": 1}
            refusal = ""
        if not refusal and str(row.get("task_id") or "") != tid:
            refusal = "task_id_mismatch"
        if refusal:
            sets["unknowns"][f"task:{tid}"] = {"task_id": tid, "path": f"task_results/{tid}.json", "reason": refusal}
            continue
        try:
            facts = o.result_facts(row)
            for name in o.result_memberships(row):
                sets[name][tid] = facts
            # Include inherited incomplete terminal copy-back, not just RUNNING.
            # Only the migration may discover child drives by enumeration below.
            if row.get("status") in {"running", "interrupted"}:
                sets["pending_drives"][tid] = facts
            elif row.get("status") in SETTLED_STATUSES:
                task = {**row, "id": tid}
                if not terminal_task_files_ready(Path(root), task, row):
                    sets["pending_drives"][tid] = facts
        except Exception as exc:
            sets["unknowns"][f"task:{tid}"] = {"task_id": tid, "path": f"task_results/{tid}.json", "reason": type(exc).__name__}
    # Drives can contain a terminal result before the canonical start/result landed.
    for base in (Path(root) / "state/headless_tasks", Path(root) / "task_drives"):
        if base.exists():
            for path in base.iterdir():
                if path.is_dir() and not path.is_symlink():
                    tid = path.name
                    sets["pending_drives"].setdefault(tid, {"task_id": tid,
                        "drives": [str(child) for child in own_child_drives(root, tid)]})
    # Import existing one-time notice receipts once. No ordinary notice reads chat.
    from ouroboros.notice_receipts import import_upgrade_receipts
    sets["upgrade_notices"] = import_upgrade_receipts(root)
    with o.locked(root):
        for name, rows in sets.items():
            try:
                prior = o._read(root, name, missing_ok=True)
            except o.ObligationsUnavailable:
                prior = {}  # an unreadable set is replaced by this complete classification
            if not (rebuild or replace_results) or name in o.RECEIPT_SETS:
                rows.update(prior)  # interrupted publications remain candidates
            elif name == "delegated_runs":
                for tid, facts in prior.items():
                    if facts.get("closed_runs"):
                        rows[tid] = {**rows.get(tid, {}), "task_id": tid, "closed_runs": facts["closed_runs"]}
            o._write(root, name, rows)
    return {name: len(rows) for name, rows in sets.items()}


def prepare_startup_state(root, *, rebuild=False, repo_dir=None, strict=True):
    """Boot may continue with unavailable sets; explicit repair reports its failure."""
    try:
        return _prepare_startup_state(root, rebuild=rebuild)
    except Exception as exc:
        log.error("Startup migration failed; current obligations unavailable; explicit rebuild required", exc_info=True)
        try:
            append_jsonl(Path(root) / "logs/supervisor.jsonl", {"ts": utc_now_iso(), "type": "startup_migration",
                "phase": "failed", "job": "obligations", "error": type(exc).__name__})
        except Exception:
            log.error("Could not persist the startup migration failure", exc_info=True)
        if strict:
            raise
        return {"imported": False, "status": "unavailable"}


def _readable(root, name):
    try:
        with o.locked(root):
            o._read(root, name)
        return True
    except o.ObligationsUnavailable:
        return False


def _prepare_startup_state(root, *, rebuild):
    root = Path(root)
    try:
        marks = watermarks(root)
    except ValueError:
        # A torn watermark file is rewritten empty: every job below then runs and restamps it.
        log.warning("Migration watermarks unreadable; rewriting them and re-running the lifecycle import")
        from ouroboros.utils import atomic_write_json
        atomic_write_json(root / "state" / "migrations.json", {}, trailing_newline=True)
        marks = {}
    saved_sha = (read_json_dict(root / "state/state.json") or {}).get("current_sha")
    # Every aware checkout stamps this SHA. A mismatch means an unaware body
    # ran between us (even at the same major), or a harmless interrupted stamp.
    foreign_body = bool(saved_sha and saved_sha != marks.get("observed_state_sha"))
    if foreign_body:
        stamp(root, cancel_latches_generation=None, obligations_generation=None)
        marks = watermarks(root)
    unreadable = {name for name in o.BOOT_SETS if not _readable(root, name)}  # missing, torn or corrupt
    rebuild_owed = (root / "state" / "obligations" / o.REBUILD_MARK).exists()
    import_needed = (rebuild or rebuild_owed or bool(unreadable) or marks.get("obligations_available") is False
                     or marks.get("obligations_generation") != OBLIGATIONS_GENERATION)
    cancel_needed = marks.get("cancel_latches_generation") != SCHEMA_GENERATION
    if not import_needed and not cancel_needed:
        return {"imported": False}
    stamp(root, obligations_available=False)
    started = time.monotonic()
    append_jsonl(root / "logs/supervisor.jsonl", {"ts": utc_now_iso(), "type": "startup_migration",
                                                "phase": "started", "job": "obligations"})
    records = list(_result_records(root))
    report = (_classify(root, records, rebuild=rebuild, replace_results=foreign_body or rebuild_owed)
              if import_needed else {})
    migrate_cancel_latches(root, records=records, rebuild=rebuild)
    if import_needed:
        from ouroboros.delegate_custody_current import rebuild as rebuild_custody
        report["custody_open"] = rebuild_custody(root, replace=rebuild or "custody_open" in unreadable)
        with o.locked(root):
            report["unknowns"] = len(o._read(root, "unknowns"))
        stamp(root, obligations_generation=OBLIGATIONS_GENERATION)
        (root / "state" / "obligations" / o.REBUILD_MARK).unlink(missing_ok=True)
    append_jsonl(root / "logs/supervisor.jsonl", {"ts": utc_now_iso(), "type": "startup_migration",
        "phase": "completed", "job": "obligations", "counts": report, "duration_seconds": time.monotonic() - started})
    stamp(root, obligations_available=True, **({"observed_state_sha": saved_sha} if saved_sha else {}))
    return {"imported": import_needed, **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_startup_state(args.data_root, rebuild=True)
    from ouroboros.model_send_seal import reconcile_model_send_seals
    report["audit"] = reconcile_model_send_seals(args.data_root)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
