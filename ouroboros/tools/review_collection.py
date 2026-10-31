"""Collect an existing paid review by identity, before consulting mutable inputs."""
from __future__ import annotations

import concurrent.futures
import contextvars
import copy
import pathlib


def reference_for(record, attempt=None):
    reference = {"record_id": record["record_id"], "task_id": record["task_id"],
                 "surface": "commit" if attempt and attempt.tool_name == "commit_reviewed" else (record["surface"] if record["surface"] in {"result", "skill"} else "change")}
    if attempt is not None:
        reference.update({key: getattr(attempt, key) for key in
                          ("repo_key", "tool_name", "attempt", "pre_review_fingerprint")})
        reference["review_record_id"] = record["record_id"]
    return reference


def resolve_reference(ctx, reference, *, commit=False):
    """Identity lookup only. It grants neither a new dispatch nor a landing."""
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.review_state import load_state, make_repo_key
    from ouroboros.tools.review_change_custody import review_state_root

    if not isinstance(reference, dict):
        raise ValueError("review_reference must be the object returned by the original review")
    record_id = str(reference.get("record_id") or reference.get("review_record_id") or "")
    record = load_record(ledger_root(ctx), record_id) if record_id else None
    if not record or record.get("task_id") != str(ctx.task_id):
        raise ValueError("review_reference does not name this author's review record")
    root = pathlib.Path(ctx.drive_root) if commit else review_state_root(ctx)
    attempt = next((row for row in reversed(load_state(root).attempts)
                    if row.review_record_id == record_id and row.task_id == str(ctx.task_id)), None)
    if attempt is None or not attempt.paid or (attempt.raw_stripped and record.get("state") != "settled"):
        raise ValueError("the original paid attempt or its recorded roster is unavailable")
    if (attempt.tool_name == "commit_reviewed") != commit:
        raise ValueError("collect this reference with its original review tool")
    if commit and attempt.repo_key != make_repo_key(pathlib.Path(ctx.repo_dir)):
        raise ValueError("the reference belongs to another repository")
    expected = reference_for(record, attempt)
    if any(key not in expected or expected[key] != value for key, value in reference.items()):
        raise ValueError("review_reference identity differs from the recorded paid attempt")
    return root, attempt, record


def collect_change(ctx, reference):
    """Join original operations at $0; never freeze, test, apply or consume a cycle."""
    from ouroboros import review_pause
    from ouroboros.review_custody import prepare_frozen_review_reconciliation
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.tools.review_change import review_result
    from ouroboros.tools.review_change_custody import recorded_change_panel, collect_recorded_commit_actor, settle_collected_attempt

    root, attempt, record = resolve_reference(ctx, reference)
    if (record.get("brief") or {}).get("dispute_input") is None:
        from ouroboros.review_history import _gap
        from ouroboros.review_ledger import revise_record

        inputs = copy.deepcopy(attempt.review_input) or {"previous_records": [], "gaps": [
            _gap("paid wave has no saved dispute-input binding", record_id=record["record_id"])]}
        def retain_input(current):
            if (current.get("brief") or {}).get("dispute_input") is None:
                current["brief"] = {**(current.get("brief") or {}), "dispute_input": inputs}
            return current
        record = revise_record(root, record["record_id"], retain_input) or record
    ctx._left_review_wait = copy.deepcopy(record.get("waiting") or {})
    from ouroboros.review_waiting import select_helpers
    select_helpers(ctx)  # Synchronous collectors also restore the saved selection.
    if record.get("state") == "settled":
        ctx._left_review_wait = None
        return review_result(record, reused=True)
    collector = copy.copy(ctx)
    collector._pending_review_attempt = attempt
    collector._current_review_retry_key = attempt.review_retry_key
    collector._review_detached_operations = []
    collector._review_reconcile_only = True
    prepare_frozen_review_reconciliation(collector, attempt)
    panel = recorded_change_panel(collector, attempt)
    seats = []
    with review_pause.receptive_author(review_pause.receptive_binding(ctx)):
        collector._review_wait_selection = copy.deepcopy(ctx._review_wait_selection)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(panel.seats))) as pool:
            futures = [(seat, pool.submit(contextvars.copy_context().run, collect_recorded_commit_actor,
                       collector, "multi_model_review", seat.slot.slot_id, attempt.review_retry_key))
                       for seat in panel.seats]
            for seat, future in futures:
                raw = future.result()
                seats.append({**raw, "slot_id": seat.slot.slot_id, "parts": list(seat.parts),
                              "model": raw.get("model") or raw.get("model_id"),
                              "raw_text": raw.get("raw_text") or raw.get("text") or ""})
    coordinates = {"record_id": record["record_id"], "drive_root": str(ctx.drive_root),
                   "canonical_root": str(ledger_root(ctx)), "retry_key": attempt.review_retry_key}
    settled = review_pause._settle_left_record(coordinates, seats, final=False)
    if settled is None:
        raise ValueError("the original review record could not be durably collected")
    settle_collected_attempt(root, attempt, seats, settled)
    ctx._review_detached_operations = collector._review_detached_operations
    wait = review_pause.watch_left_review(ctx, surface=reference_for(record)["surface"],
        retry_key=attempt.review_retry_key, record_id=record["record_id"],
        reissue="review_change(review_reference=the original reference)")
    if wait is None:
        ctx._left_review_wait = None
    return review_result(load_record(ledger_root(ctx), record["record_id"]) or settled, reused=True)


def prepare_commit_collection(ctx, reference, *, commit_message="", mutations=()):
    """Separate from Advisory author continuation: resume the paid gate itself."""
    _root, attempt, record = resolve_reference(ctx, reference, commit=True)
    if any(mutations) or (commit_message and commit_message != attempt.commit_message):
        raise ValueError("a paid reference cannot change its commit message, brief or panel")
    if attempt.status == "succeeded":
        raise ValueError("this paid attempt already committed; collection cannot commit it again")
    ctx._collect_commit_source = attempt
    ctx._review_resume_pending = True
    ctx._pending_review_attempt = attempt
    ctx._current_review_attempt_number = attempt.attempt
    ctx._current_review_record_id = record["record_id"]
    ctx._left_review_wait = copy.deepcopy(record.get("waiting") or {})
    return attempt


def bind_commit_collection(ctx, fingerprint, retry_key):
    source = getattr(ctx, "_collect_commit_source", None)
    if source is None:
        return None, retry_key
    if fingerprint != source.pre_review_fingerprint:
        return {"status": "blocked", "message": "REVIEW_BINDING_BLOCKED: current staged bytes differ from the paid reference.",
                "block_reason": "review_binding"}, retry_key
    ctx._current_review_retry_key = source.review_retry_key
    ctx._current_review_rebuttal_sha256 = source.rebuttal_sha256
    return None, source.review_retry_key


def publish_commit_delivery(ctx, text):
    from ouroboros.review_waiting import publish_delivery
    from ouroboros.review_ledger import ledger_root, load_record
    record_id = str(getattr(ctx, "_current_review_record_id", "") or "")
    record = load_record(ledger_root(ctx), record_id) if record_id else None
    return publish_delivery(ctx, text, record) if record else text
