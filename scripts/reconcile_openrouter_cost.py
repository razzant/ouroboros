#!/usr/bin/env python3
"""Inspect selected physical attempts; explicitly fetch and/or apply exact prices.

Source invocation: python scripts/reconcile_openrouter_cost.py --attempt-id ID
Repeat --attempt-id to select more rows. --fetch permits one bounded metadata GET
per distinct ID with no retained price; --apply separately permits accounting.
Neither flag implies the other. No flags means no network or monetary writes.
An existing usage.sqlite is required; this command never scans or imports history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _target() -> dict:
    """Read the existing settings/route owners without load_settings' migrations."""
    from ouroboros import config
    from ouroboros.llm_routing import _ProviderRoutingMixin
    from ouroboros.settings_integrity import read_settings_json_verified

    raw = read_settings_json_verified(config.SETTINGS_PATH) if config.SETTINGS_PATH.exists() else {}
    settings = {"OPENROUTER_API_KEY": (raw or {}).get("OPENROUTER_API_KEY",
                                                    config.runtime_setting("OPENROUTER_API_KEY", ""))}
    return _ProviderRoutingMixin()._resolve_remote_target("openrouter::", settings=settings)


def _custody(root: Path, row: dict) -> str | None:
    """Reuse maintenance's owner/review/post-work limits, never reopen an answer."""
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_open
    from ouroboros.review_operation import task_has_live_review_operation
    from ouroboros.task_status import SETTLED_STATUSES, task_has_live_queue_ownership
    from supervisor import queue
    from supervisor.task_ownership import TaskOwnershipRead

    task_id = str(row.get("task_id") or "")
    if row.get("non_task_operation") is True:
        # Review attribution survives completion. Only an actual operation
        # still holding custody prevents a price-only correction, including
        # one whose author has already published a terminal task result.
        # system:<source> names an accounting scope, not a task-result ID;
        # the real probe producer deliberately uses it without a task owner.
        if task_id and not task_id.startswith("system:") and task_has_live_review_operation(root, task_id):
            return "review_unsettled"
        return None if row.get("state") in {"settled", "unresolved"} else "physical_owner_unsettled"
    if not task_id:
        return "owner_unavailable"
    reads = TaskOwnershipRead(root)
    task = reads.load(task_id)
    if task.get("status") not in SETTLED_STATUSES:
        return "owner_unsettled"
    if queue.INITIALIZED and Path(queue.DRIVE_ROOT).resolve() == root:
        live = queue.task_has_live_ownership(task_id, ownership=reads)
    else:
        # A standalone command has no supervisor maps. Read the selected
        # root's current snapshot; missing/stale observations retain custody.
        live = (task_has_live_queue_ownership(root, task_id)
                or task_has_live_review_operation(root, task_id, result_loader=reads.load))
        from ouroboros.utils import read_json_dict

        direct_path = root / "state" / "direct_roots.json"
        if direct_path.exists():
            direct = read_json_dict(direct_path)
            if (not isinstance(direct, dict) or direct.get("incomplete")
                    or not isinstance(direct.get("roots"), list)
                    or any(not isinstance(item, dict) or item.get("task_id") == task_id
                           for item in direct["roots"])):
                return "owner_unsettled"
        # The direct fragment is positive-only, not an interlock: absence
        # cannot prove all live stacks ended, nor prevent later admission.
    if live or not reads.unchanged((task_id,)):
        return "owner_unsettled"
    if post_task_synthesis_is_open((task.get("root_phase_checkpoint") or {}).get("post_task_synthesis")):
        return "post_task_work_open"
    return None


def reconcile_selected(root: Path, attempt_ids: list[str], *, fetch: bool = False, apply: bool = False) -> list[dict]:
    """Addressed inspect → retained source → optional GET → optional CAS application."""
    from ouroboros import openrouter_cost as cost, usage_store

    root = root.resolve()
    outcomes = []
    for attempt_id in dict.fromkeys(attempt_ids):
        outcome = {"attempt_id": attempt_id}
        try:
            if not (root / usage_store.STORE_REL).is_file():
                outcomes.append({**outcome, "status": "store_unavailable"})
                continue
            # Even if another process exports/removes the store after the
            # existence check, inspection must never initialize/import history.
            with usage_store.hold(root, write=False, migrate=False) as txn:
                row = txn.attempt(attempt_id)
            if row is None:
                outcomes.append({**outcome, "status": "attempt_not_found"})
                continue
            outcome.update(state=row["state"], cost_usd=row.get("cost_usd"),
                           cost_final=row.get("cost_final"), revision=row["revision"])
            if row.get("provider") != "openrouter" or row.get("kind", "attempt") != "attempt":
                outcomes.append({**outcome, "status": "ineligible"})
                continue
            receipt = cost.read_retained_receipt(root, row)
            outcome["receipt_available"] = receipt is not None
            if not fetch and not apply:
                outcomes.append({**outcome, "status": "inspect", "binding_available": cost.generation_binding(row) is not None,
                                 "receipt_cost_usd": receipt.get("cost_usd") if receipt else None})
                continue
            blocked = _custody(root, row)
            if blocked:
                outcomes.append({**outcome, "status": blocked})
                continue
            if row["state"] not in {"dispatched", "unresolved", "settled"}:
                outcomes.append({**outcome, "status": "ineligible"})
                continue
            if receipt is None and fetch:
                if row.get("cost_final") is True:
                    outcomes.append({**outcome, "status": "already_final"})
                    continue
                receipt = cost.fetch_generation_receipt(root, row, _target())
            if receipt is None:
                outcomes.append({**outcome, "status": "receipt_unavailable"})
                continue
            if receipt.get("status") != "price":
                outcomes.append({**outcome, **{key: receipt[key] for key in ("status", "status_code", "retry_after")
                                             if key in receipt}})
                continue
            outcome.update(receipt_available=True, receipt_cost_usd=receipt["cost_usd"])
            if apply:
                # Reobserve owner custody after network I/O; monetary races are
                # decided separately under the accounting owner's row revision.
                blocked = _custody(root, row)
                result = ({"status": blocked} if blocked else cost.apply_retained_receipt(
                    root, attempt_id, receipt, expected_revision=row["revision"]))
                outcome["status"] = result["status"]
                if result.get("reason"):
                    outcome["reason"] = result["reason"]
                if result.get("row") is not None:
                    outcome["previous_cost_usd"] = outcome["cost_usd"]
                    outcome["previous_state"] = outcome["state"]
                    outcome.update({key: result["row"].get(key) for key in ("state", "cost_usd", "cost_final", "revision")})
            else:
                outcome["status"] = "retained"
            outcomes.append(outcome)
        except Exception as exc:
            # Do not print provider bodies, keys, or exception strings containing them.
            outcomes.append({**outcome, "status": "error", "error_type": type(exc).__name__})
    return outcomes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", action="append", required=True, help="exact physical attempt; repeatable")
    parser.add_argument("--data-root", type=Path, help="existing data root; defaults to OUROBOROS_DATA_DIR")
    parser.add_argument("--fetch", action="store_true", help="permit one bounded generation GET per selected ID")
    parser.add_argument("--apply", action="store_true", help="apply retained exact receipts through accounting")
    args = parser.parse_args(argv)
    from ouroboros import config

    outcomes = reconcile_selected(args.data_root or config.DATA_DIR, args.attempt_id, fetch=args.fetch, apply=args.apply)
    print(json.dumps({"fetch": args.fetch, "apply": args.apply, "outcomes": outcomes}, sort_keys=True))
    return 0 if all(row["status"] in {"inspect", "retained", "applied", "duplicate", "already_final"}
                    for row in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
