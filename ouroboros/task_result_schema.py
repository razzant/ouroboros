"""ABI 7.0 task-result schema admission: the stamp, the classifier, and the quarantine.

Every durable task-result row written by this version is stamped
``_schema_version: 1`` (the shared opt-in key from
``contracts/schema_versions.py``). Readers QUARANTINE — never convert — a row
that is unstamped (pre-7.0 history), carries an unknown or future stamp, is
malformed JSON, or still carries the retired pre-7.0
``improvement_policy: "until_deadline"`` contract form the 7.0
acceptance-wallet authority judges malformed (owner decision Q8=B: no legacy
converter, no compat machinery). The bytes move unchanged into
``task_results/quarantine/`` and the read reports "no result". Visibility is
the durable events log ONLY (owner decision 6.3=B: no UI counter, no chat
notice): ONE ``task_results_quarantined`` event per read/scan batch, never
one per file. Retention/GC never touches the quarantine namespace; recovery
is a manual owner action (re-stamp and move the file back).

Exactly ONE carve-out is admitted (owner 4A), and it lives outside this module
in ``cancel_intents.migrate_legacy_cancel_latches``: unstamped rows still
LATCHED at ``cancel_requested`` are re-stamped in place at boot before any
ordinary read, because quarantining them stranded the wedged task with no
terminal at all. It re-writes, it does not convert — the status and every
field are unchanged — and nothing else escapes the quarantine.

Split out of ``ouroboros/task_results.py`` (module-size discipline); the
facade re-exports every name here, and callers keep importing through it.
"""

from __future__ import annotations

import logging
import os
import pathlib
import uuid
from typing import Any, Dict, List

from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
from ouroboros.utils import append_jsonl, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

TASK_RESULT_SCHEMA_VERSION = 1
TASK_RESULT_QUARANTINE_DIR = "quarantine"
QUARANTINED_SCHEMA_REASON = "quarantined_schema"


def task_result_schema_refusal(data: Any) -> str:
    """Classify one stored task-result row for schema admission.

    Returns ``""`` for an admissible row, else the quarantine reason:
    ``malformed`` (unparseable / not an object), ``unstamped_pre_7_0``,
    ``future_schema`` (integer stamp above ours), ``invalid_stamp``
    (non-integer or non-positive stamp), or
    ``retired_contract_until_deadline`` (ledger f30 entry 4: the pre-7.0
    pacing alias whose normalized profile the acceptance-wallet authority
    judges malformed). Classification only — Q8=B deliberately has no
    conversion path.
    """
    if not isinstance(data, dict):
        return "malformed"
    stamp = data.get(SCHEMA_VERSION_KEY)
    if stamp is None:
        return "unstamped_pre_7_0"
    if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 1:
        return "invalid_stamp"
    if stamp > TASK_RESULT_SCHEMA_VERSION:
        return "future_schema"
    contract = data.get("task_contract")
    profile = contract.get("budget_profile") if isinstance(contract, dict) else None
    if isinstance(profile, dict) and str(profile.get("improvement_policy") or "") == "until_deadline":
        return "retired_contract_until_deadline"
    return ""


def require_writable_task_result_schema(existing: Any, path: Any = "") -> None:
    """Refuse to merge current-schema fields into a row another schema owns.

    An UNSTAMPED existing row is writable — a live pre-upgrade task's next
    lifecycle write stamps it (stamp-on-write, not a converter). A row stamped
    with any OTHER version (e.g. a rollback survivor written by a newer
    release) must never be silently downgraded: fail loudly instead.
    """
    stamp = existing.get(SCHEMA_VERSION_KEY) if isinstance(existing, dict) else None
    if stamp is not None and (
        isinstance(stamp, bool) or stamp != TASK_RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            "TASK_RESULT_SCHEMA_REFUSED: refusing to write schema-"
            f"{TASK_RESULT_SCHEMA_VERSION} fields over a row stamped {stamp!r}"
            + (f": {path}" if str(path or "") else "")
        )


def stamp_task_result_schema(row: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp *row* (in place) as written by the current schema; returns it."""
    row[SCHEMA_VERSION_KEY] = TASK_RESULT_SCHEMA_VERSION
    return row


def quarantine_task_result(path: pathlib.Path, reason: str) -> str:
    """Move one inadmissible row into ``task_results/quarantine/``, bytes unchanged.

    Same-directory rename under the row's own write-lock sidecar (the
    ``update_json_locked`` lock), so a concurrent lifecycle write either lands
    before the move — the row is re-checked under the lock and KEPT when it
    became admissible — or re-creates a fresh stamped row after it. Never
    overwrites an occupied quarantine slot. Returns ``"moved"``,
    ``"kept_admissible"`` or ``"failed"``; best-effort — a failure leaves the
    row in place (the triggering read still reports no result).
    """
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )

    lock_path = path.with_name(path.name + ".lock")
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=4.0, stale_sec=90.0)
    if lock_fd is None:
        log.warning("task-result quarantine skipped (lock timeout): %s", path)
        return "failed"
    try:
        if not path.is_file():
            return "failed"  # already quarantined / removed by another reader
        if not task_result_schema_refusal(read_json_dict(path)):
            return "kept_admissible"  # a concurrent write stamped it
        quarantine_dir = path.parent / TASK_RESULT_QUARANTINE_DIR
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        dest = quarantine_dir / path.name
        if dest.exists():
            dest = quarantine_dir / f"{path.stem}.{uuid.uuid4().hex[:8]}{path.suffix}"
        os.rename(path, dest)
        log.warning("Quarantined task result %s -> %s (%s)", path.name, dest, reason)
        return "moved"
    except OSError:
        log.warning("task-result quarantine failed for %s", path, exc_info=True)
        return "failed"
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def emit_quarantine_event(drive_root: Any, quarantined: List[Dict[str, str]]) -> None:
    """ONE durable events-log row per read/scan batch (6.3=B: log-only visibility)."""
    if not quarantined:
        return
    reasons: Dict[str, int] = {}
    for row in quarantined:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    try:
        root = pathlib.Path(drive_root)
        append_jsonl(root / "logs" / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "task_results_quarantined",
            "count": len(quarantined),
            "first_task_id": quarantined[0]["task_id"],
            "reasons": reasons,
            "quarantine_dir": str(root / "task_results" / TASK_RESULT_QUARANTINE_DIR),
        })
    except Exception:
        log.warning("failed to record the task-result quarantine event", exc_info=True)
