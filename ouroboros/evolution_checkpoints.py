"""Append-only checkpoints for evolution progress and later eval curves."""

from __future__ import annotations

import hashlib
import logging
import pathlib
import subprocess
from typing import Any, Dict, Optional

from ouroboros.outcomes import normalize_outcome_axes
from ouroboros.utils import append_jsonl, utc_now_iso


log = logging.getLogger(__name__)

CHECKPOINTS_REL = pathlib.Path("state") / "evolution_checkpoints.jsonl"


def _sha_file(path: pathlib.Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _git_value(repo_dir: pathlib.Path, args: list[str]) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=str(repo_dir), capture_output=True, text=True, timeout=5)
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def append_cycle_outcome_checkpoint(
    drive_root: pathlib.Path,
    *,
    campaign: Dict[str, Any] | None = None,
    transaction: Dict[str, Any] | None = None,
    source: str = "",
    backlog_id: str = "",
) -> None:
    """Tag the ledger when a cycle outcome is decided AFTER task-done.

    A commit-bearing cycle is recorded ``waiting_for_restart`` at task-done;
    the absorbed/abandoned resolution lands later (restart verification or
    boot reconcile) and previously never reached this ledger — the solve-
    capability history could not tell which objectives actually got absorbed.
    Append-only and schema-additive (``kind="cycle_outcome"``); join key with
    the task-done row is ``task_id``.
    """
    tx = transaction if isinstance(transaction, dict) else {}
    entry = {
        "schema_version": 1,
        "kind": "cycle_outcome",
        "ts": utc_now_iso(),
        "source": str(source or ""),
        "task_id": str(tx.get("task_id") or ""),
        "campaign_id": str((campaign or {}).get("id") or tx.get("campaign_id") or ""),
        "campaign_objective": str((campaign or {}).get("objective") or ""),
        "backlog_id": str(backlog_id or ""),
        "cycle_outcome": str(tx.get("cycle_outcome") or ""),
        "abandoned_reason": str(tx.get("abandoned_reason") or ""),
        "commit_sha": str(tx.get("commit_sha") or ""),
        "outcome_axes": normalize_outcome_axes({"outcome_axes": tx.get("outcome_axes") or {}}),
    }
    append_jsonl(pathlib.Path(drive_root) / CHECKPOINTS_REL, entry)


def append_cycle_outcome_tag(
    drive_root: pathlib.Path, *, campaign: Any, transaction: Any,
    source: str, backlog_id: str,
) -> None:
    """Append a resolution tag without letting a ledger failure break the boot.

    The tag is written after the campaign transition that decided the outcome,
    so it can never be part of that transition's atomic write; failing the
    caller here would trade an under-reported digest for a broken restart path.
    Invariant the callers keep: the campaign resolution and this tag are ONE
    critical section under ``locks/state.lock`` (both the boot reconcile and the
    marker restart-verify path append the tag before releasing it), and the
    repair-side reader ``backfill_missing_cycle_outcomes`` takes the same lock —
    so a concurrent booter never sees a resolved campaign whose tag is still
    pending. What a crash inside that section loses, the backfill re-derives.
    """
    try:
        append_cycle_outcome_checkpoint(
            drive_root, campaign=campaign, transaction=transaction,
            source=source, backlog_id=backlog_id,
        )
    except Exception:
        log.debug("Failed to append %s cycle-outcome checkpoint", source, exc_info=True)


def backfill_missing_cycle_outcomes(
    drive_root: pathlib.Path, campaign: Dict[str, Any] | None,
) -> int:
    """Re-derive the cycle-outcome rows a crash lost, and return how many landed.

    The campaign write that resolves a cycle and this ledger's append are NOT
    one atomic write: a crash between them leaves a campaign that says
    ``absorbed`` with no row, and the digest under-reports that cycle forever.
    The resolved transaction itself carries every field the row needs, so the
    row is derivable — this replays the missing ones at boot. Idempotent: a
    task that already has a ``cycle_outcome`` row is skipped, and only
    commit-bearing resolutions (the ones the two crash windows can strand) are
    replayed. ``backlog_id`` is not recoverable after the fact and is left
    empty, exactly as the abandoned path already writes it. Never raises.
    The scan-then-append is only dedupe-safe under ``locks/state.lock`` — the
    lock the resolution writers hold across campaign write + tag — so the boot
    caller (``verify_restart``) runs it there and skips it, rather than running
    it unlocked, when the lock is unavailable.
    """
    import json as _json

    written = 0
    try:
        history = (campaign or {}).get("transaction_history") or []
        pending = {
            str(tx.get("task_id")): tx
            for tx in history
            if isinstance(tx, dict)
            and str(tx.get("task_id") or "")
            and str(tx.get("commit_sha") or "").strip()
            and str(tx.get("cycle_outcome") or "") in {"absorbed", "abandoned"}
        }
        if not pending:
            return 0
        path = pathlib.Path(drive_root) / CHECKPOINTS_REL
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            try:
                row = _json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and str(row.get("kind") or "") == "cycle_outcome":
                pending.pop(str(row.get("task_id") or ""), None)
        for tx in pending.values():
            append_cycle_outcome_checkpoint(
                drive_root, campaign=campaign, transaction=tx,
                source="boot_backfill", backlog_id="",
            )
            written += 1
    except Exception:
        log.debug("Failed to backfill cycle-outcome checkpoints", exc_info=True)
    return written


def build_solve_capability_digest(drive_root: pathlib.Path, *, max_entries: int = 200) -> str:
    """Compact digest of which objectives actually improved the system.

    Joins task-done checkpoints with later ``cycle_outcome`` tags (last wins
    per task) and renders absorbed vs abandoned/no_op history for the
    promotion chooser. Returns "" when there is no usable history.
    """
    import json as _json

    path = pathlib.Path(drive_root) / CHECKPOINTS_REL
    try:
        all_lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    omitted_lines = max(0, len(all_lines) - max_entries)
    lines = all_lines[-max_entries:]
    rows = []
    for line in lines:
        try:
            row = _json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    if not rows:
        return ""

    by_task: Dict[str, Dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        merged = by_task.setdefault(task_id, {})
        if task_id not in order:
            order.append(task_id)
        if str(row.get("kind") or "") == "cycle_outcome":
            merged["cycle_outcome"] = str(row.get("cycle_outcome") or merged.get("cycle_outcome") or "")
            merged["abandoned_reason"] = str(row.get("abandoned_reason") or merged.get("abandoned_reason") or "")
            merged.setdefault("objective", str(row.get("campaign_objective") or ""))
            merged["commit_sha"] = str(row.get("commit_sha") or merged.get("commit_sha") or "")
        else:
            tx = row.get("transaction") if isinstance(row.get("transaction"), dict) else {}
            # Task-done rows: a later cycle_outcome tag wins over the
            # waiting_for_restart recorded at task end.
            merged.setdefault("cycle_outcome", str(tx.get("cycle_outcome") or ""))
            merged["objective"] = str(row.get("campaign_objective") or merged.get("objective") or "")
            merged.setdefault("commit_sha", str(tx.get("commit_sha") or ""))
            merged["rounds"] = int(row.get("rounds") or 0)
            merged["cost_accounting_status"] = str(
                row.get("cost_accounting_status") or "available"
            )
            merged["cost_usd"] = (
                float(row.get("cost_usd")) if row.get("cost_usd") is not None else None
            )
            execution = (row.get("outcome_axes") or {}).get("execution") or {}
            merged["execution"] = str(execution.get("status") or "")

    counts: Dict[str, int] = {}
    absorbed: list[str] = []
    failed: list[str] = []
    for task_id in reversed(order):  # newest first
        info = by_task.get(task_id) or {}
        outcome = str(info.get("cycle_outcome") or "unknown")
        counts[outcome] = counts.get(outcome, 0) + 1
        objective = str(info.get("objective") or "").strip().replace("\n", " ")
        if len(objective) > 110:
            # Explicit omission marker (no silent [:N] of cognitive artifacts);
            # the full objective stays in the ledger row.
            objective = objective[:110] + " …[truncated; full objective in the ledger]"
        if outcome == "absorbed" and len(absorbed) < 8:
            extras = []
            if info.get("commit_sha"):
                extras.append(str(info["commit_sha"])[:10])
            if info.get("rounds"):
                extras.append(f"rounds={info['rounds']}")
            if info.get("cost_accounting_status") == "unavailable":
                extras.append("cost=unavailable")
            elif info.get("cost_usd"):
                extras.append(f"cost=${info['cost_usd']:.2f}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            absorbed.append(f"- ABSORBED: {objective or '(objective unknown)'}{suffix}")
        elif outcome in {"abandoned", "no_op"} and len(failed) < 4:
            reason = str(info.get("abandoned_reason") or "").strip()
            suffix = f" — {reason}" if reason else ""
            failed.append(f"- {outcome.upper()}: {objective or '(objective unknown)'}{suffix}")
    if not counts:
        return ""
    summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    parts = [f"Cycle outcomes ({len(by_task)} recent cycles): {summary}."]
    if omitted_lines:
        # P1: no silent truncation of cognitive history — disclose the window.
        parts.append(
            f"[OMISSION NOTE: digest covers the newest {max_entries} ledger rows; "
            f"{omitted_lines} older rows omitted — full history in state/evolution_checkpoints.jsonl]"
        )
    if absorbed:
        parts.append("Objectives that ACTUALLY got absorbed (reviewed commit survived restart):")
        parts.extend(absorbed)
    if failed:
        parts.append("Recent cycles that did NOT land:")
        parts.extend(failed)
    return "\n".join(parts)


def append_evolution_checkpoint(
    drive_root: pathlib.Path,
    repo_dir: pathlib.Path,
    *,
    task_id: str,
    campaign: Dict[str, Any] | None = None,
    outcome_axes: Dict[str, Any] | None = None,
    cost_usd: Optional[float] = 0.0,
    cost_accounting_status: str = "available",
    rounds: int = 0,
    transaction: Dict[str, Any] | None = None,
) -> None:
    """Persist a lightweight checkpoint after an evolution cycle."""
    memory = pathlib.Path(drive_root) / "memory"
    entry = {
        "schema_version": 1,
        "ts": utc_now_iso(),
        "task_id": str(task_id or ""),
        "campaign_id": str((campaign or {}).get("id") or ""),
        "campaign_objective": str((campaign or {}).get("objective") or ""),
        "git_sha": _git_value(pathlib.Path(repo_dir), ["rev-parse", "HEAD"]),
        "git_branch": _git_value(pathlib.Path(repo_dir), ["rev-parse", "--abbrev-ref", "HEAD"]),
        "identity_sha256": _sha_file(memory / "identity.md"),
        "scratchpad_sha256": _sha_file(memory / "scratchpad.md"),
        "knowledge_index_sha256": _sha_file(memory / "knowledge" / "index-full.md"),
        "outcome_axes": normalize_outcome_axes({"outcome_axes": outcome_axes or {}}),
        "cost_usd": (
            float(cost_usd) if cost_accounting_status == "available" and cost_usd is not None
            else None
        ),
        "cost_accounting_status": (
            "available" if cost_accounting_status == "available" and cost_usd is not None
            else "unavailable"
        ),
        "rounds": int(rounds or 0),
    }
    if isinstance(transaction, dict) and transaction:
        entry["transaction"] = dict(transaction)
    append_jsonl(pathlib.Path(drive_root) / CHECKPOINTS_REL, entry)
