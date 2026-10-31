"""Visible lifecycle import of current boot state, never a reader fallback.

Run before producers on every startup door: ``ensure_current_store`` opens or creates
``state/current.sqlite`` (the one importer; legacy reconcile is a named job in
``store_import``), the admissible legacy cancel latches become intents, and the
boot admissions no document derives (drives present on disk, the one-time upgrade
receipts) join the owner-held rows, and legacy questions bind their publication
address once. ``state/migrations.json`` keeps the watermarks: the store, the
cancel-latch, the question-publication and the retired-set generations.
``python -m ouroboros.startup_migrations`` is the explicit rebuild job, including
its historical seal audit (no audit watermark). An owner-held obligation set whose producer
this body retired (``RETIRED_SETS``) has no reader or remover left, so the first boot of its
generation deletes the rows an older body kept there. Its unsent outbox lines become exhausted
with ``producer_retired`` through the shared storage owner; payloads remain, no delivery is invented.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

from ouroboros.utils import append_jsonl, read_json_dict, update_json_locked, utc_now_iso

log = logging.getLogger(__name__)
SCHEMA_GENERATION = 7
QUESTION_PUBLICATIONS_GENERATION = 1
RETIRED_SETS = ("memory_max",)  # the lost-Max episode's one row, retired 29.10 with its producer
RETIRED_SETS_GENERATION = 2  # also retire the old episode's unsent outbox lines; generation 1 removed only its set


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


def migrate_cancel_latches(root, *, records, generation=SCHEMA_GENERATION, rebuild=False):
    """The admissible ``cancel_requested`` rows of RECORDS (``(task_id, document, error)`` as the
    importer admitted them; ``None`` reads the store's rows) become cancel intents once per
    generation; the explicit rebuild reruns it over the store."""
    from ouroboros.cancel_intents import migrate_legacy_cancel_latches

    if not rebuild and watermarks(root).get("cancel_latches_generation") == generation:
        return []
    migrated = migrate_legacy_cancel_latches(root, records=None if records is None else list(records))
    stamp(root, cancel_latches_generation=generation)
    return migrated


def _legacy_question_publications(root, needed):
    """Import exact publication evidence once; current routing proves no history.

    Conflicting addresses or unstable history stay unknown. Ask-time chat and
    binding timestamps cannot establish whether a queued quiz was rebound.
    """
    from ouroboros.memory import Memory

    if not needed:
        return {}, {}, False
    try:
        rows, coverage = Memory(root).read_chat_generations(
            predicate=lambda row: row.get("type") == "quiz" and row.get("direction") == "out")
    except (OSError, ValueError) as exc:
        rows, coverage = [], {"snapshot_stable": True, "gaps": [
            {"kind": "generation_unreadable", "error": type(exc).__name__}]}
    gaps = coverage.get("gaps") or []
    unstable = not coverage.get("snapshot_stable")
    unreadable = any(gap.get("kind") == "generation_unreadable" for gap in gaps)
    addresses = {}
    for row in rows:
        quiz = row.get("quiz")
        if not isinstance(quiz, dict):
            continue
        key = (str(row.get("task_id") or ""), str(quiz.get("quiz_id") or ""))
        chat_id = row.get("chat_id")
        if key in needed and isinstance(chat_id, int) and not isinstance(chat_id, bool):
            addresses.setdefault(key, set()).add(chat_id)
    publications = {key: next(iter(values)) for key, values in addresses.items()
                    if len(values) == 1 and not unstable}
    unknowns = {f"question:{tid}:{qid}": {"task_id": tid, "quiz_id": qid,
        "reason": ("publication_address_ambiguous" if len(addresses.get((tid, qid), ())) > 1 else
                   "publication_history_unstable" if unstable else
                   "publication_history_unavailable" if unreadable else "publication_address_missing"),
        **({"history_gaps": gaps} if gaps else {})}
        for tid, qid in needed if (tid, qid) not in publications}
    return publications, unknowns, bool(unknowns and (unstable or unreadable))


def migrate_question_publications(root, *, rebuild=False, legacy=False):
    """Questions asked without publication addressing bind it once, from the store's
    ``owner_questions`` members and exact chat evidence (a GET never migrates): those asked
    before it existed, and those an older body asked after a rollback (no delivery record and
    no ``question:`` row yet, discovered by the legacy reconcile job). The rest stay addressed ``question:`` unknowns, never a guessed
    Main; a moving or unreadable history earns one maintenance retry, and the explicit rebuild
    starts a fresh repair. Current generations never read history during boot."""
    from ouroboros import obligations
    from ouroboros.owner_quiz import quiz_needs_retention, record_publication
    from ouroboros.task_store import read_document

    marks = watermarks(root)
    current = marks.get("question_publications_generation") == QUESTION_PUBLICATIONS_GENERATION
    retry = marks.get("question_publication_retry") or {}
    if current and not (rebuild or legacy):
        if retry.get("state") == "pending":
            from ouroboros import maintenance_jobs
            job = maintenance_jobs.job(root, "legacy_reconcile") or {}
            if not job or job.get("state") == "done":
                maintenance_jobs.record(root, "legacy_reconcile", state="pending", reason="question_publication_retry",
                                        cursor=None, done=0, total=None, error=None)
        return {}
    needed = set()
    for tid in obligations.members(root, "owner_questions"):
        try:
            quizzes = (read_document(root, tid) or {}).get("owner_quiz")
        except ValueError:
            continue  # an unreadable member stays flagged: Main discloses its gap
        for qid, block in quizzes.items() if isinstance(quizzes, dict) else ():
            # A current ask carries its own publication debt, which a closure before publication
            # takes over (``publication_pending``); either binds its address before exposure.
            # Only an older body's ask (no such debt) is legacy for this import.
            owed = block.get("delivery") if isinstance(block, dict) else None
            if (quiz_needs_retention(block) and "publication_chat_id" not in block
                    and not (isinstance(owed, dict)
                             and (owed.get("kind") == "publication" or owed.get("publication_pending")))):
                needed.add((tid, str(qid)))
    publications, unknowns, unsettled = _legacy_question_publications(root, needed)
    for (tid, qid), chat_id in publications.items():
        record_publication(root, tid, qid, chat_id, evidence="history")
    for key in obligations.members(root, "unknowns"):
        if key.startswith("question:") and key not in unknowns:
            obligations.remove(root, "unknowns", key)  # re-classified: bound, answered or gone
    for key, facts in unknowns.items():
        obligations.add(root, "unknowns", key, facts)
    attempts = (retry.get("attempts", 0) if current and not rebuild else 0) + 1
    stamp(root, question_publications_generation=QUESTION_PUBLICATIONS_GENERATION, question_publication_retry=(
        {"attempts": attempts, "state": "pending" if attempts < 2 else "rebuild_required"} if unsettled else None))
    return {"bound": len(publications), "unknown": len(unknowns)}


def retire_sets(root, *, rebuild=False) -> int:
    """Delete the rows an older body kept in ``RETIRED_SETS``: the remover their gone producer was.
    Once per generation, stamped after success; a current stamp reads no row (the explicit rebuild
    repeats it, as after a return from an older body that kept such a row again). Returns the rows removed."""
    from ouroboros import obligations
    from ouroboros.terminal_outbox import retire_producer

    if not rebuild and watermarks(root).get("retired_sets_generation") == RETIRED_SETS_GENERATION:
        return 0
    retire_producer(root, delivery_prefix="memory_max:", system_type="memory_max_notice")
    removed = 0
    for name in RETIRED_SETS:
        for identity in obligations.members(root, name):
            obligations.remove(root, name, identity)
            removed += 1
    stamp(root, retired_sets_generation=RETIRED_SETS_GENERATION)
    return removed


def boot_admissions(root) -> dict:
    """Owner-held rows no document derives: a drive directory present on disk owes its terminal
    files (a terminal result can land in the drive before the canonical start/result did), and
    the one-time upgrade notice receipts are imported from chat once."""
    from ouroboros.notice_receipts import import_upgrade_receipts
    from ouroboros.task_custody import own_child_drives

    root = Path(root)
    pending = {}
    for base in (root / "state/headless_tasks", root / "task_drives"):
        if base.exists():
            for path in base.iterdir():
                if path.is_dir() and not path.is_symlink():
                    tid = path.name
                    pending[tid] = {"task_id": tid, "drives": [str(child) for child in own_child_drives(root, tid)]}
    return {"pending_drives": pending, "upgrade_notices": import_upgrade_receipts(root)}


def prepare_startup_state(root, *, rebuild=False, repo_dir=None, strict=True, initialize_journal=False):
    """Boot may continue with an unavailable store (every door then answers typed); the explicit
    repair reports its failure."""
    try:
        return _prepare_startup_state(root, rebuild=rebuild, repo_dir=repo_dir,
                                      initialize_journal=initialize_journal)
    except Exception as exc:
        log.error("Startup import failed; the current-facts store is unavailable", exc_info=True)
        try:
            append_jsonl(Path(root) / "logs/supervisor.jsonl", {"ts": utc_now_iso(), "type": "startup_migration",
                "phase": "failed", "job": "current_store", "error": type(exc).__name__})
        except Exception:
            log.error("Could not persist the startup migration failure", exc_info=True)
        if strict:
            raise
        return {"imported": False, "status": "unavailable"}


def _prepare_startup_state(root, *, rebuild, repo_dir=None, initialize_journal=False):
    from ouroboros.store_import import ensure_current_store

    root = Path(root)
    try:
        watermarks(root)
    except ValueError:
        # A torn watermark file is rewritten empty: every job below then runs and restamps it.
        log.warning("Migration watermarks unreadable; rewriting them and re-running the lifecycle import")
        from ouroboros.utils import atomic_write_json
        atomic_write_json(root / "state" / "migrations.json", {}, trailing_newline=True)
    started = time.monotonic()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir or Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = ""  # unknown current body is recorded; never substitute the previous body's SHA
    report = ensure_current_store(root, rederive=rebuild, body_sha=head, initialize_journal=initialize_journal)
    if rebuild:
        from ouroboros.delegate_custody_current import rebuild as rebuild_custody
        report["custody_open"] = rebuild_custody(root, replace=True)
        report["cancel_latches"] = len(migrate_cancel_latches(root, records=None, rebuild=True))
    if report["status"] in {"unchanged", "completed", "reconciled"}:
        if not rebuild:
            migrate_cancel_latches(root, records=None)  # the generation gate precedes every history read
        try:  # unstamped on failure: the next boot repeats it
            report["question_publications"] = migrate_question_publications(root, rebuild=rebuild)
        except Exception as exc:
            log.warning("Legacy question publication import failed; it reruns at the next start", exc_info=True)
            report["question_publications"] = {"error": type(exc).__name__}
        try:  # unstamped on failure: the next boot repeats it
            if removed := retire_sets(root, rebuild=rebuild):
                report["retired_rows"] = removed
        except Exception:
            log.warning("Retired obligation rows not removed; the next start repeats it", exc_info=True)
    append_jsonl(root / "logs/supervisor.jsonl", {"ts": utc_now_iso(), "type": "startup_migration",
        "phase": "completed", "job": "current_store", "counts": {key: value for key, value in report.items()
                                                                  if key != "drives"},
        "duration_seconds": time.monotonic() - started})
    return {"imported": report.get("status") == "completed", **report}


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
