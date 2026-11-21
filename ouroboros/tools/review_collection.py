"""Collect an existing paid review by identity, before consulting mutable inputs. An author continuation's
reference names the subject that review covered the same way: its paths stay the default scope (#2680)."""
from __future__ import annotations

import concurrent.futures
import contextvars
import copy
import json
import os
import pathlib

SHOWN_OUTCOMES = "shown_outcomes"  # the delivery a commit reference answers (``require_shown``), never its identity


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
    from ouroboros.review_state import _attempt_review_roster_rows, load_state, make_repo_key
    from ouroboros.tools.review_change_custody import review_state_root

    if not isinstance(reference, dict):
        raise ValueError("review_reference must be the object returned by the original review")
    record_id = str(reference.get("record_id") or reference.get("review_record_id") or "")
    record = load_record(ledger_root(ctx), record_id) if record_id else None
    if not record or not (record.get("task_id") == str(ctx.task_id) or (commit and _continued_author(ctx, record))):
        raise ValueError("review_reference does not name this author's review record")
    root = pathlib.Path(ctx.drive_root) if commit else review_state_root(ctx)
    # A commit reference names its own attempt: a later call that collected the same settled record at $0 under
    # its own row (``settle_completed_round``) leaves the original reference resolving to its original attempt.
    named = reference.get("attempt") if commit and type(reference.get("attempt")) is int else None
    attempt = next((row for row in reversed(load_state(root).attempts) if row.review_record_id == record_id
                    and row.task_id == record["task_id"] and named in (None, row.attempt)), None)
    # A wave reserves its roster before any seat sends and is paid only at its first send (#2410): the
    # reserved operations are the custody this reference collects, whether or not one was sent yet.
    reserved = attempt is not None and any(row.get("operation_id") for row in _attempt_review_roster_rows(attempt))
    if not (attempt is not None and (attempt.paid or reserved)) or (
            attempt.raw_stripped and record.get("state") != "settled"):
        if record.get("state") == "settled" and (record.get("verdict") or {}).get("aggregate") == "NOT_DISPATCHED":
            refusal = record.get("dispatch_refusal") if isinstance(record.get("dispatch_refusal"), dict) else {}
            raise ValueError("this review was never dispatched (NOT_DISPATCHED"
                             + (f", {refusal['kind']}" if refusal.get("kind") else "")
                             + "): no reviewer was sent or paid, so there is no answer to collect")
        raise ValueError("the original paid attempt or its recorded roster is unavailable")
    if (attempt.tool_name == "commit_reviewed") != commit:
        raise ValueError("collect this reference with its original review tool")
    if commit and attempt.repo_key != make_repo_key(pathlib.Path(ctx.repo_dir)):
        raise ValueError("the reference belongs to another repository")
    expected = reference_for(record, attempt)
    if any(key not in expected or expected[key] != value for key, value in reference.items() if key != SHOWN_OUTCOMES):
        raise ValueError("review_reference identity differs from the recorded paid attempt")
    return root, attempt, record


def _continued_author(ctx, record):
    return continued_review_author(ctx, record.get("task_id"))


def continued_review_author(ctx, task_id):
    """Whether this task may read and decide on a commit review that ``task_id`` received: its own, or one a
    predecessor received when the owner's recorded Continue claims lead from that task to this one
    (``owner_continue.recorded_continuation``). Room, root, sibling or parent ancestry grants nothing (#2901)."""
    own, task_id = str(getattr(ctx, "task_id", "") or ""), str(task_id or "")
    if not own or not task_id or task_id == own:
        return bool(own) and task_id == own
    from ouroboros.owner_continue import recorded_continuation
    from ouroboros.review_ledger import ledger_root

    return recorded_continuation(ledger_root(ctx), task_id, own)


def attempt_roster(root, attempt):
    """The paid attempt's recorded seats, and each seat's compare-and-swap (``Roster.transition``) under its state lock."""
    from types import SimpleNamespace
    from ouroboros.review_continuation import ContinuationRefused, Roster
    from ouroboros.review_custody import prepare_frozen_review_reconciliation
    from ouroboros.review_state import ReviewStateMutation, _utc_now, load_state, update_state

    holder = SimpleNamespace()
    prepare_frozen_review_reconciliation(holder, attempt)
    rows = {slot: (surface, row) for surface, seats in holder._review_frozen_rows.items() for slot, row in seats.items()}

    def seat_of(state, seat_id):
        current = state.latest_attempt_for(repo_key=attempt.repo_key, tool_name=attempt.tool_name,
                                           task_id=attempt.task_id, attempt=int(attempt.attempt or 0))
        for seats in ((current.triad_raw_results or []), ((current.scope_raw_result or {}).get("raw_results") or [])) if current else ():
            index = next((i for i, row in enumerate(seats) if str(row.get("slot_id") or row.get("slot") or "") == seat_id), None)
            if index is not None:
                return current, seats, index
        return current, None, None

    def carried(seat_id):
        _current, seats, index = seat_of(load_state(pathlib.Path(root)), seat_id)
        return dict(seats[index]) if seats is not None else {}

    def transition(seat_id, decision, build):
        after = {}

        def mutate(state):
            current, seats, index = seat_of(state, seat_id)
            if current is None or seats is None:
                raise ContinuationRefused("roster_unavailable", "the paid attempt row is no longer in the review state")
            operation = str(seats[index].get("operation_id") or "")
            if operation != decision["operation_id"]:
                if operation != decision["supersedes"]:
                    raise ContinuationRefused("roster_changed", f"seat {seat_id}'s attempt row no longer names {decision['supersedes']}")
                seats[index] = build(seats[index])
                current.late_result_pending = bool(current.late_result_pending or seats[index].get("late_result_pending"))
                current.updated_ts = _utc_now()
            after.clear()
            after.update(seats[index])
            return ReviewStateMutation(None, changed=operation != decision["operation_id"])
        update_state(pathlib.Path(root), mutate)
        return dict(after)
    return Roster(task_id=str(attempt.task_id), retry_key=str(attempt.review_retry_key), rows=rows, transition=transition,
                  carried=carried, root_task_id=str(attempt.root_task_id or ""),
                  review_contract=str(attempt.review_contract_fingerprint or ""))


def continue_or_repair(ctx, root, attempt, record, selections):
    """An explicit choice (``selections``), else the repair of an already recorded one; ``None`` facts when
    neither applies. Returns the record, the attempt row read again, and what this call decided or projected."""
    from ouroboros.delegate_custody import custody_root
    from ouroboros.review_continuation import arm, current_decisions, project_decisions
    from ouroboros.review_ledger import ledger_root
    from ouroboros.review_state import load_state

    if selections is None and not current_decisions(record):
        return record, attempt, None
    roster = attempt_roster(root, attempt)
    if selections is not None:
        outcome = arm(ctx, record, roster, selections, drive=ledger_root(ctx), custody=custody_root(ctx))
        record, facts = outcome["record"], {"continued": outcome["continued"]}
    else:
        projected = project_decisions(ctx, custody_root(ctx), record, roster, drive=ledger_root(ctx))
        record, facts = projected["record"], {"repaired": projected["written"], "window_spent": projected["spent"],
                                              "not_sent": projected["unsent"]}
    attempt = load_state(pathlib.Path(root)).latest_attempt_for(
        repo_key=attempt.repo_key, tool_name=attempt.tool_name, task_id=attempt.task_id, attempt=int(attempt.attempt or 0)) or attempt
    return record, attempt, facts


def repair_round(ctx, root, *, repo_key, tool_name, retry_key):
    """Before a call of a round reads that round's cached roster (an identical reissue, the result delivery's
    collection, an identical commit): a choice recorded on the round's newest attempt is projected first, so it
    is rejoined and sent under its own token rather than met by a competing wave. ``None`` when none is owed."""
    from ouroboros.review_continuation import current_decisions
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.review_state import load_state

    attempts = [row for row in load_state(pathlib.Path(root)).filter_attempts(
        repo_key=repo_key, tool_name=tool_name, task_id=str(getattr(ctx, "task_id", "") or ""))
        if retry_key and row.review_retry_key == retry_key and row.review_record_id]
    attempt = max(attempts, key=lambda row: int(row.attempt or 0), default=None)
    record = load_record(ledger_root(ctx), attempt.review_record_id) if attempt is not None else None
    if not record or not current_decisions(record):
        return None
    return continue_or_repair(ctx, root, attempt, record, None)


def collect_change(ctx, reference, continue_seats=None):
    """Join original operations at $0; never freeze, test, apply or consume a cycle. ``continue_seats`` (or a
    recorded continuation this collection finds unsent) sends the chosen successors first, in the same wave."""
    root, attempt, record = resolve_reference(ctx, reference)
    return collect_wave(ctx, root, attempt, record, continue_seats,
                        reissue="review_change(review_reference=the original reference)")


def _row_deferred(exc):
    """The ``attempt_row`` fact of a collection whose attempt-row write the busy shared review-state lock refused (#2779)."""
    return {"state": "deferred", "reason": "review_state_busy", "detail": str(exc), "next": (
        "the collected answers are durable on the review record above; once the lock is free, collecting the same "
        "reference again at $0 (or deciding on a commit reference) reconciles the attempt row from its original "
        "operations' durable outcomes")}


def collect_wave(ctx, root, attempt, record, continue_seats=None, *, reissue, read_only=False):
    """One recorded wave's collection: join its original operations at $0 and settle what they returned onto its record
    and attempt row (``review_change``'s reference, and a commit reference whose subject the index no longer holds).
    ``read_only`` projects and sends no recorded continuation. A record already settled is read, never sent, after the
    attempt row an earlier writer left open beside it is reconciled from its operations' durable outcomes
    (``reconcile_collected_attempt``). A busy shared review-state lock leaves the row open and says so
    (``attempt_row``, #2779)."""
    from ouroboros import review_pause
    from ouroboros.review_custody import prepare_frozen_review_reconciliation
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.review_state_records import ReviewStateLockError
    from ouroboros.tools.review_change import review_result
    from ouroboros.tools.review_change_custody import (
        collect_recorded_commit_actor, reconcile_collected_attempt, recorded_change_panel, settle_collected_attempt,
    )

    if (record.get("brief") or {}).get("dispute_input") is None:
        from ouroboros.review_history import _gap
        from ouroboros.review_ledger import revise_record

        inputs = copy.deepcopy(attempt.review_input) or {"previous_records": [], "gaps": [
            _gap("paid wave has no saved dispute-input binding", record_id=record["record_id"])]}
        def retain_input(current):
            if (current.get("brief") or {}).get("dispute_input") is None:
                current["brief"] = {**(current.get("brief") or {}), "dispute_input": inputs}
            return current
        record = revise_record(ledger_root(ctx), record["record_id"], retain_input) or record
    record, attempt, continuation = (record, attempt, None) if read_only else continue_or_repair(
        ctx, root, attempt, record, continue_seats)
    ctx._left_review_wait = copy.deepcopy(record.get("waiting") or {})
    from ouroboros.review_waiting import select_helpers
    select_helpers(ctx)  # Synchronous collectors also restore the saved selection.
    if record.get("state") == "settled":  # a rejoined successor that already settled is read, never sent
        ctx._left_review_wait = None
        result = {**review_result(record, reused=True), **({"continuation": continuation} if continuation else {})}
        try:
            reconcile_collected_attempt(root, ledger_root(ctx), attempt, record)
        except ReviewStateLockError as exc:
            result["attempt_row"] = _row_deferred(exc)
        return result
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
    row_deferred = None
    try:
        settle_collected_attempt(root, attempt, seats, settled)
    except ReviewStateLockError as exc:  # the record already holds every collected answer; nothing is resent
        row_deferred = _row_deferred(exc)
    ctx._review_detached_operations = collector._review_detached_operations
    surface = reference_for(record, attempt)["surface"]
    wait = review_pause.watch_left_review(ctx, surface=surface, retry_key=attempt.review_retry_key,
        attempt=int(attempt.attempt or 0) if surface == "commit" else 0, record_id=record["record_id"], reissue=reissue)
    if wait is None:
        ctx._left_review_wait = None
    result = review_result(load_record(ledger_root(ctx), record["record_id"]) or settled, reused=not continuation)
    if row_deferred:
        result["attempt_row"] = row_deferred
    return {**result, "continuation": continuation} if continuation else result


def commit_offer(ctx, record_id):
    """A commit outcome without the handback (an owner's stop, a refusal after the wave): the exact
    reference and offer of its stopped session seats, or ``""``. Choosing one is the same commit tool's own
    collection of that paid gate (``prepare_commit_collection``), never an automatic send."""
    import json

    from ouroboros.review_continuation import offers_for
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.review_state import load_state

    offers = offers_for(ctx, record_id)
    attempt = next((row for row in reversed(load_state(pathlib.Path(ctx.drive_root)).attempts)
                    if row.review_record_id == record_id and row.task_id == str(ctx.task_id)), None) if offers else None
    if attempt is None:
        return ""
    return ("\nA reviewer in continuable_seats stopped with work its engine can continue: the same commit tool with "
            "review_reference and continue_seats (no author_disposition) continues it in the same wave over its frozen "
            "subject, inside this paid gate when the reviewed bytes are staged, else read-only.\n" + json.dumps({"review_reference": reference_for(load_record(ledger_root(ctx), record_id), attempt),
                                           "continuable_seats": offers}, ensure_ascii=False, default=str))


def state_busy_refusal(exc):
    """A collection the shared review-state lock refused before it settled the wave (#2779): recoverable, never
    a missing result. Returns the tool text."""
    return (f"⚠️ REVIEW_STATE_BUSY: {exc}. The shared review state stayed locked, so this collection stopped before it "
            "settled its wave; nothing was staged or committed. A start this call carried, a continuation it chose "
            "included, may already have been sent under its recorded key; every answer and choice recorded for this wave "
            "stays on its review record, and the same review_reference collects them at $0, replaying no start under a "
            "new key.")


def collect_commit_reference(ctx, reference, **request):
    """The commit tools' collection by reference (``prepare_commit_collection``): the attempt this call resumes, or
    its whole outcome before anything is staged, a refused reference (``TOOL_ARG_ERROR``), a busy shared review state
    (``REVIEW_STATE_BUSY``) or an outcome read back (``settled_read``, ``read_only_collection``)."""
    from ouroboros.review_continuation import ContinuationRefused
    from ouroboros.review_state_records import ReviewStateLockError
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    try:
        return prepare_commit_collection(ctx, reference, **request)
    except ContinuationRefused as exc:  # an explicit choice of the old subject's seats, refused or recorded unsent
        return _publish_tool_result(ctx, ToolResult(status="blocked", code="LEGACY_BLOCKED", text=_continuation_block(exc)["message"]
                                                    + "\n" + json.dumps(exc.payload(), ensure_ascii=False, default=str)))
    except ValueError as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=str(exc)))
    except ReviewStateLockError as exc:
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="LEGACY_UNAVAILABLE", text=state_busy_refusal(exc)))


_PAID_FIELDS = ("review_rebuttal", "goal", "scope", "reviewers", "reason", "preflight_reviewer", "since_review",
                "author_disposition")  # the order of ``commit_reviewed``'s ``mutations``


def prepare_commit_collection(ctx, reference, *, commit_message="", mutations=(), continue_seats=None):
    """Separate from the author continuation: resume the paid gate itself (``continue_seats``: and its
    chosen stopped seats, armed only after the staged bytes match, ``bind_commit_collection``). Only this task's
    collection of the exact staged bytes it reviewed resumes the gate; any other collection reads the outcome
    (``settled_read``) or joins the wave read-only (``read_only_collection``), before anything is staged."""
    from ouroboros.review_continuation import parse_selections
    from ouroboros.tools import git as git_mod

    root, attempt, record = resolve_reference(ctx, reference, commit=True)
    changed = [name for name, value in zip(_PAID_FIELDS, mutations) if value]
    changed += ["commit_message"] if commit_message and commit_message != attempt.commit_message else []
    if changed:  # the fields named (06.11), so the next call can omit exactly them
        raise ValueError(f"a paid reference cannot change its commit message, brief, panel or author disposition "
                         f"(changed: {', '.join(changed)}); omit them to collect the paid review as recorded")
    ctx._collect_commit_continue_seats = None if continue_seats is None else (parse_selections(continue_seats) and continue_seats)
    if attempt.status == "succeeded":
        raise ValueError("this paid attempt already committed; collection cannot commit it again")
    own = attempt.task_id == str(ctx.task_id)
    staged = git_mod._fingerprint_staged_diff(pathlib.Path(ctx.repo_dir))
    exact = own and staged.get("fingerprint") == attempt.pre_review_fingerprint
    if continue_seats is None and (read := settled_read(ctx, attempt, record, exact=exact, staged=staged)):
        return read
    if attempt.raw_stripped:  # a compacted row keeps no roster to rejoin, and custody is never rebuilt from it
        raise ValueError("the returned review outcome is missing or no longer has its exact evidence")
    if not exact:  # never this gate's landing: an explicit continue_seats continues the OLD frozen subject's seats
        return read_only_collection(ctx, root, attempt, record, staged, continue_seats)
    ctx._collect_commit_source = attempt
    ctx._review_resume_pending = True
    ctx._pending_review_attempt = attempt
    ctx._current_review_attempt_number = attempt.attempt
    ctx._current_review_record_id = record["record_id"]
    ctx._left_review_wait = copy.deepcopy(record.get("waiting") or {})
    return attempt


def _read_only_why(ctx, attempt, record, staged):
    """Why a commit collection returns read-only, in one clause: a predecessor's review, or how the index now differs
    from the reviewed subject (its tree and parent, else the rest of the staged binding)."""
    if attempt.task_id != str(ctx.task_id):
        return (f"it is the review a recorded Continue predecessor ({attempt.task_id}) received, which this task reads "
                "and never resumes")
    if staged.get("fingerprint") != attempt.pre_review_fingerprint:
        subject, binding = record.get("subject") or {}, staged.get("binding") or {}
        reviewed = (str(subject.get("tree_sha") or ""), str(subject.get("base") or ""))
        current = (str(binding.get("tree_sha") or ""), str((binding.get("parents") or [""])[0]))
        if not staged.get("ok"):
            detail = f"the staged binding is unreadable ({staged.get('reason') or 'no detail'})"
        elif all(reviewed) and reviewed == current:
            detail = ("the staged tree and parent are the reviewed ones, so its MERGE_HEAD, staged VERSION, release tag "
                      "or patch identity differs")
        else:
            detail = (f"staged tree {current[0][:12]} on {current[1][:12]} against reviewed tree {reviewed[0][:12]} "
                      f"on {reviewed[1][:12]}")
        return f"the index no longer holds the bytes it reviewed ({detail})"
    return "its attempt row was compacted out of the recent review history, so its custody is not rebuilt"


def _read_only_text(ctx, attempt, record, staged, result, lead):
    import json

    from ouroboros.review_waiting import publish_delivery, shown_outcomes

    payload = {**result, "review_reference": {**reference_for(record, attempt),
                                              SHOWN_OUTCOMES: shown_outcomes(record, result.get("rows") or [])}}
    running = payload.get("state") != "settled"
    sent = ("only the continuation you chose was sent, in the original wave over its frozen subject"
            if payload.get("continuation") else "no new reviewer was sent")
    text = (f"{lead}: {_read_only_why(ctx, attempt, record, staged)}, so nothing was staged, committed, tagged or "
            f"pushed, {sent}, and the files and index are unchanged. "
            + ("Its reviewers are still running: the same review_reference collects them again at $0. " if running else "")
            + "To land the current candidate, continue with this review_reference and an author_disposition, " + ("stage the reviewed bytes again to land this review through its own gate, "
                              if attempt.task_id == str(ctx.task_id) and not attempt.raw_stripped else "")
            + "or ask for a new review.\n" + json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return publish_delivery(ctx, text, record, rendered_rows=payload["rows"])


def settled_read(ctx, attempt, record, *, exact, staged):
    """Reading a settled commit review needs none of its old bytes back and rebuilds no custody: when its bound record
    settled its counted seats (``review_change_custody.settled_record``) and the collection is not this task's of the
    exact staged bytes it reviewed (``exact``: the index changed, or the review is a recorded Continue predecessor's) or
    its attempt row was compacted out of the recent history (no rows left to rejoin), the collection returns that whole
    recorded outcome before anything is staged, rendered as every review record is (``review_change.review_result``:
    each seat's answer and findings) and published with its delivery receipt, the files and index as they are. Nothing
    is collected, committed or inherited, and the record keeps its verdict: a PASS stays the reviewed bytes'
    own. ``""`` when it does not apply."""
    from ouroboros.review_ledger import ledger_root
    from ouroboros.tools.review_change import review_result
    from ouroboros.tools.review_change_custody import settled_record

    settled = settled_record(ledger_root(ctx), attempt, record)
    if settled is None or (exact and not attempt.raw_stripped):
        return ""
    return _read_only_text(ctx, attempt, settled, staged, review_result(settled, reused=True),
                           "Review outcome already settled, returned read-only")


def read_only_collection(ctx, root, attempt, record, staged, continue_seats=None):
    """A commit wave collected by a call that cannot resume its gate (#2797, #2901): this task's wave whose subject the
    index no longer holds joins its original operations over their own frozen subject at $0 (``collect_wave``,
    ``read_only``: no recorded continuation is projected); a recorded Continue predecessor's wave, whose dispatch
    authority stays its predecessor's, is collected as a wave its author left (``attached_collection``). An explicit
    ``continue_seats`` (either author) is the one sending act: its successors are new generation of the same wave under
    its recorded window and today's limits (``review_continuation.arm``), collected the same way. The outcome returns
    before anything is staged; the gate's live-index checkout is never entered, so the retained subject its seats read
    is never replaced by today's index, and no landing is inherited."""
    if attempt.task_id == str(ctx.task_id) or continue_seats is not None:
        result = collect_wave(ctx, root, attempt, record, continue_seats,
                              reissue="commit_reviewed(review_reference=the original reference)", read_only=continue_seats is None)
    else:
        result = attached_collection(ctx, root, attempt, record)
    from ouroboros.review_ledger import ledger_root, load_record

    current = load_record(ledger_root(ctx), record["record_id"]) or record
    return _read_only_text(ctx, attempt, current, staged, {**result, "rows": result.get("rows") or []},
                           "Review outcome collected read-only")


def attached_collection(ctx, root, attempt, record):
    """A recorded Continue predecessor's commit wave read by its successor (#2901), with no dispatch authority over it:
    its original operations are collected at $0 as a wave its author left (``review_pause._left_seats``): a retained
    outcome, a bound run read attach-only to its terminal state; a live worker or an unbound start is left as it is,
    never joined, replayed or sent again. What they settled lands on the record and, closed seats only, on the
    predecessor's attempt row (``_settle_left_record``, ``_settle_left_attempt``); a busy shared review-state lock
    leaves that row open and says so (``attempt_row``). ``review_result`` of the record as it then stands."""
    from ouroboros import review_pause
    from ouroboros.review_ledger import ledger_root
    from ouroboros.review_state_records import ReviewStateLockError
    from ouroboros.tools.review_change import review_result

    left = {"surface": "commit", "task_id": str(attempt.task_id), "record_id": str(record["record_id"]),
            "retry_key": str(attempt.review_retry_key or ""), "attempt": int(attempt.attempt or 0),
            "drive_root": str(root), "canonical_root": str(ledger_root(ctx))}
    wave, seats, gap = review_pause._left_seats(left)
    settled = None if gap else review_pause._settle_left_record(left, seats, final=False)
    result = review_result(settled or record, reused=True)
    try:
        review_pause._settle_left_attempt(left, wave, seats, settled)
    except ReviewStateLockError as exc:
        result["attempt_row"] = _row_deferred(exc)
    return result


def reviewed_subject(ctx, source):
    """What the review an author continuation names covered: its record's base and tree, read once and bound to the
    reference's own fingerprint, and their concrete paths in this repository (``diff-tree --no-renames -z``: rename
    sides apart, names literal). ``{"error": ...}`` when that composition cannot be recovered."""
    from ouroboros.mutation_attribution import _run_git
    from ouroboros.review_ledger import ledger_root, load_record

    record_id = str(getattr(source, "review_record_id", "") or "")
    try:
        record = load_record(ledger_root(ctx), record_id) if record_id else None
    except ValueError as exc:
        return {"error": f"its review record is unreadable ({exc})"}
    subject = (record or {}).get("subject") or {}
    base, tree = str(subject.get("base") or ""), str(subject.get("tree_sha") or "")
    if record is None:
        return {"error": "the review record it names is missing" if record_id else "this older review has no review record"}
    if (record.get("fingerprints") or {}).get("binding") != source.pre_review_fingerprint or not (base and tree):
        return {"error": "its review record does not bind the reviewed subject"}
    try:
        names = _run_git(pathlib.Path(ctx.repo_dir), "diff-tree", "-r", "--no-renames", "--name-only", "-z", base, tree)
    except (OSError, RuntimeError) as exc:
        return {"error": f"the reviewed trees are not readable in this repository ({exc})"}
    paths = [name for name in names.split("\0") if name]
    return {"base": base, "tree": tree, "paths": paths} if paths else {"error": "its review record names no changed path"}


def _directories_without_entry(repo, names):
    """Of ``names``, those whose path is a directory Git stages as no entry of its own. Git's evidence decides, never
    the directory's shape: a gitlink indexed at the name is one entry (``git add`` keeps or moves it), and so is a
    repository Git resolves there (``rev-parse --resolve-git-dir``) while nothing is indexed below the name; a
    ``.git`` marker alone proves nothing. The files any other such directory holds are other paths."""
    from ouroboros.mutation_attribution import _run_git

    found = [name for name in names if (repo / name).is_dir() and not (repo / name).is_symlink()]
    if not found:
        return []
    rows = [row.split("\t", 1) for row in _run_git(
        repo, "--literal-pathspecs", "ls-files", "--stage", "-z", "--", *found).split("\0") if "\t" in row]
    gitlinks = {path for entry, path in rows if entry.startswith("160000 ")}
    below = {name for name in found for _entry, path in rows if path.startswith(f"{name}/")}

    def repository(name):
        try:
            return os.path.lexists(repo / name / ".git") and bool(_run_git(repo, "rev-parse", "--resolve-git-dir",
                                                                           f"{name}/.git").strip())
        except RuntimeError:  # not a gitdir: a marker, an empty directory or a gitfile that resolves nowhere
            return False
    return [name for name in found if name not in gitlinks and (name in below or not repository(name))]


def _unchanged_now(repo, names):
    """Of ``names``, those that differ from HEAD neither in the index nor on disk: tracked and clean, or absent from
    both (a directory Git stages as no entry there is none, ``_directories_without_entry``). An ignored file on disk
    still differs, so staging answers for it as for any explicit selection."""
    from ouroboros.mutation_attribution import _git_status_paths, _run_git

    changed, tracked = set(_git_status_paths(repo)), set(_run_git(repo, "ls-files", "-z").split("\0"))
    gone = set(_directories_without_entry(repo, [name for name in names if name not in changed and name not in tracked]))
    return [name for name in names if name not in changed
            and (name in tracked or not os.path.lexists(repo / name) or name in gone)]


def replaced_by_directories(ctx, paths):
    """Of the paths about to be staged, the reviewed entries an author continuation selected with paths omitted
    (``author_scope``) whose names now hold a directory Git stages as no entry (``_directories_without_entry``):
    staging removes each exact entry and adds nothing the directory holds, paths its review never named. An explicit
    selection keeps Git's meaning of a directory."""
    selected = set((getattr(ctx, "_author_commit_scope", None) or {}).get("selected") or ())
    return _directories_without_entry(pathlib.Path(ctx.repo_dir), [path for path in paths if path in selected])


def author_scope(ctx, *, paths_omitted):
    """Read the subject an author continuation's reference reviewed once (``ctx._author_commit_scope``). With paths
    omitted, its paths are the selection, never every attributed candidate (#2680): those that no longer differ from
    HEAD are named and left out, and nothing else is added. Before anything is staged, a composition that cannot be
    recovered asks for explicit paths, and one with nothing left to land says so; ``None`` lets the call continue."""
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    scope = ctx._author_commit_scope = reviewed_subject(ctx, ctx._author_commit_source)
    if not paths_omitted:
        return None
    if not scope.get("error"):
        try:
            scope["unchanged"] = _unchanged_now(pathlib.Path(ctx.repo_dir), scope["paths"])
            scope["selected"] = sorted(set(scope["paths"]) - set(scope["unchanged"]))
        except (OSError, RuntimeError) as exc:
            scope["error"] = f"which of its paths still differ from HEAD is unreadable ({exc})"
    if scope.get("error"):
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            "ERROR: REVIEWED_PATHS_UNAVAILABLE: omitted paths mean the paths this review_reference reviewed, and "
            f"{scope['error']}. Name the paths to land in paths: an explicit selection continues without a new "
            "review. Nothing was staged or committed.")))
    if not scope["selected"]:
        return (f"⚠️ GIT_NO_CHANGES: no reviewed path differs from HEAD now ({', '.join(scope['unchanged'])}); "
                "nothing was staged or committed, and no other path was taken in their place.")
    return None


def _listed(names, limit=20):
    return ", ".join(names[:limit]) + (f" and {len(names) - limit} more" if len(names) > limit else "")


def disclose_subject_delta(ctx, pre_fingerprint):
    """Compare, once, the staged subject an author continuation is about to land with the subject its reference
    reviewed, and publish the comparison before tests and Git: a progress note now, the same text in the result, and
    the returned fact for the author's decision record. The note names the paths that differ in bytes or
    composition; the four trees keep everything else recoverable."""
    from ouroboros.mutation_attribution import _run_git

    scope = getattr(ctx, "_author_commit_scope", None) or reviewed_subject(ctx, ctx._author_commit_source)
    binding = pre_fingerprint.get("binding") or {}
    staged, head = str(binding.get("tree_sha") or ""), str((binding.get("parents") or [""])[0])
    base, tree = str(scope.get("base") or ""), str(scope.get("tree") or "")
    fact = {"relation": "unknown", "reviewed_base": base, "reviewed_tree": tree, "staged_base": head, "staged_tree": staged}
    error = str(scope.get("error") or "")
    if not error and (staged, head) == (tree, base):
        fact["relation"], text = "same", f"staged tree {staged[:12]} is exactly the reviewed tree."
    elif not error:
        def paths(a, b):
            return {name for name in _run_git(pathlib.Path(ctx.repo_dir), "diff-tree", "-r", "--no-renames",
                                              "--name-only", "-z", a, b).split("\0") if name}
        try:
            reviewed, current, changed = set(scope["paths"]), paths(head, staged), paths(tree, staged)
        except (OSError, RuntimeError) as exc:
            error = f"the trees are not comparable here ({exc})"
        else:
            unchanged = set(scope.get("unchanged") or ())
            clauses = [f"{label}: {_listed(sorted(names))}" for label, names in (
                ("changed since review", changed & reviewed & current), ("not in the reviewed paths", current - reviewed),
                ("reviewed but no longer different from HEAD", (reviewed - current) & unchanged),
                ("reviewed but left out", reviewed - current - unchanged)) if names]
            clauses += [f"HEAD moved from {base[:12]} to {head[:12]}"] if head != base else []
            fact["relation"] = "different"
            text = f"staged tree {staged[:12]} differs from the reviewed tree {tree[:12]}: " + "; ".join(clauses) + "."
    if error:
        text = f"the reviewed tree is unavailable ({error}), so staged tree {staged[:12]} cannot be compared with it."
    fact["text"] = text = "Author continuation: " + text
    advisory = getattr(ctx, "_review_advisory", None)
    if isinstance(advisory, list):
        advisory.append(text)
    try:
        ctx.emit_progress_fn(text)
    except Exception:
        pass
    return fact


def collected_open_peer(ctx, commit_message, goal, scope, review_rebuttal, paths):
    """The open frozen attempt this call collects exactly, known before any effect: the index already
    holds its subject under this intent (``commit_gate.exact_open_peer``) and staging changes nothing
    (each path to stage, the retained ones when omitted, equals its working file; the release sync
    writes nothing). Only that collection keeps a retained composition; a distinct intent, or newer
    bytes under the same intent, is a new subject under the ordinary selection rules."""
    from ouroboros.commit_admission import _version_sync_due
    from ouroboros.tools import git as git_mod
    from ouroboros.tools.commit_gate import exact_open_peer
    from ouroboros.tools.git_vcs_ops import _scope_changes
    from ouroboros.tools.release_sync import sync_release_metadata

    if not getattr(ctx, "_review_open_peers", None):
        return None
    repo = pathlib.Path(ctx.repo_dir)
    try:
        fingerprint = git_mod._fingerprint_staged_diff(repo).get("fingerprint", "")
        peer = exact_open_peer(ctx, commit_message, goal=goal, scope=scope, review_rebuttal=review_rebuttal,
                               fingerprint=fingerprint) if fingerprint else None
        if peer is None:
            return None
        targets = ([git_mod.safe_relpath(path) for path in paths if str(path).strip()] if paths is not None
                   else sorted(git_mod._staged_index(repo)))
        moved = bool(targets) and any(state[1] != " " for state, _path in _scope_changes(
            repo, [f":(literal){name}" for name in targets]))
        synced = _version_sync_due(repo, targets) and bool(sync_release_metadata(str(repo), write=False))
    except Exception:
        return None
    return None if moved or synced else peer


def _selection_changed(ctx, retry_key):
    """A pre-lock peer only permits its exact final subject; restaging cannot lend that permission to a new wave."""
    peer = getattr(ctx, "_review_collected_peer", None)
    selection = getattr(ctx, "_commit_attribution", None) or {}
    unnamed = sorted(set(selection.get("uncertain") or []) - set(selection.get("explicit_uncertain") or []))
    return unnamed if peer is not None and retry_key != peer.review_retry_key else []


def _continuation_block(exc):
    """A commit call's typed outcome for a continuation refused before its choice, or recorded and unsent."""
    from ouroboros.review_continuation import ContinuationRefused

    typed = isinstance(exc, ContinuationRefused)
    unsent = typed and bool(exc.recorded)
    return {"status": "blocked", "block_reason": "review_continuation_unsent" if unsent else "review_continuation_refused",
            "continuation": exc.payload() if typed else {"reason": "record_unreadable"},
            "message": f"REVIEW_CONTINUATION_{'UNSENT' if unsent else 'REFUSED'}: {exc}. "
                       + (exc.consequence() if typed else "Nothing was decided or sent; the review is unchanged.")
                       + " No commit was created."}


def decided_round_attempts(ctx, repo_key):
    """This task's settled commit attempts (an additional seat may still run) whose record holds a recorded
    continuation no projection carried yet."""
    from ouroboros.review_continuation import current_decisions
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.review_state import _attempt_has_active_review_custody, attempt_counted_seats_closed, load_state
    from ouroboros.tools.commit_gate import _current_review_tool_name

    found = []
    for attempt in load_state(pathlib.Path(ctx.drive_root)).filter_attempts(
            repo_key=repo_key, tool_name=_current_review_tool_name(ctx), task_id=str(getattr(ctx, "task_id", "") or "")):
        if attempt.review_record_id and attempt.review_retry_key and (
                not _attempt_has_active_review_custody(attempt) or attempt_counted_seats_closed(attempt)):
            try:
                record = load_record(ledger_root(ctx), attempt.review_record_id)
            except ValueError:
                continue
            if record and current_decisions(record):
                found.append(attempt)
    return found


def recorded_round_choice(ctx, retry_key):
    """A plain commit call of a round whose recorded continuation a crash left unprojected: that round's own
    attempt with the choice projected (``repair_round``), for the gate to rejoin as its open attempt, never a
    new paid wave; ``None`` when nothing is owed. The retry key binds the staged bytes and intent, a changed
    selection projects nothing, and the rejoined gate keeps its tests and Stop/deadline checks before Git. A
    projection refused now is held for ``bind_commit_collection``'s typed block."""
    from ouroboros.review_continuation import ContinuationRefused
    from ouroboros.review_state import _attempt_has_active_review_custody, make_repo_key
    from ouroboros.tools.commit_gate import _current_review_tool_name

    ctx._review_round_refusal = None
    if getattr(ctx, "_collect_commit_source", None) is not None or _selection_changed(ctx, retry_key):
        return None  # a reference binds its own source below; a changed selection is blocked below
    try:
        repaired = repair_round(ctx, ctx.drive_root, repo_key=make_repo_key(pathlib.Path(ctx.repo_dir)),
                                tool_name=_current_review_tool_name(ctx), retry_key=retry_key)
    except ContinuationRefused as exc:
        ctx._review_round_refusal = _continuation_block(exc)
        return None
    if repaired is None:
        return None
    _record, attempt, ctx._review_continuation = repaired
    return attempt if _attempt_has_active_review_custody(attempt) else None


def bind_commit_collection(ctx, fingerprint, retry_key):
    unnamed = _selection_changed(ctx, retry_key)
    if unnamed:
        return {"status": "blocked", "block_reason": "review_binding", "message": (
            "⚠️ REVIEW_SELECTION_CHANGED_BLOCKED: the prepared index no longer matches the retained review, "
            f"and this call did not explicitly select its uncertain paths: {', '.join(unnamed)}. "
            "No new review was dispatched or commit created. The prepared index and files remain; inspect "
            "the changed composition and name the paths you intend to review.")}, retry_key
    refused, ctx._review_round_refusal = getattr(ctx, "_review_round_refusal", None), None
    if refused:  # this round's recorded choice could not be projected now (``recorded_round_choice``)
        return refused, retry_key
    source = getattr(ctx, "_collect_commit_source", None)
    if source is None:
        return None, retry_key
    if fingerprint != source.pre_review_fingerprint:
        return {"status": "blocked", "message": (
            "REVIEW_BINDING_BLOCKED: current staged bytes differ from the paid reference, so this paid gate cannot "
            "resume. Call again with the same review_reference: it reads the original wave back read-only, and with "
            "continue_seats continues its stopped seats over their frozen subject; nothing was staged or committed."),
                "block_reason": "review_binding"}, retry_key
    from ouroboros.review_ledger import ledger_root, load_record

    try:  # the exact subject is bound: only now may a chosen stopped seat continue (tests still run before Git)
        record, attempt, ctx._review_continuation = continue_or_repair(
            ctx, ctx.drive_root, source, load_record(ledger_root(ctx), source.review_record_id) or {},
            getattr(ctx, "_collect_commit_continue_seats", None))
    except ValueError as exc:  # a typed refusal (or a recorded choice nothing sent yet), or an unreadable record
        return _continuation_block(exc), retry_key
    ctx._collect_commit_source = ctx._pending_review_attempt = source = attempt
    ctx._current_review_retry_key = source.review_retry_key
    ctx._current_review_rebuttal_sha256 = source.rebuttal_sha256
    return None, source.review_retry_key


def publish_commit_delivery(ctx, text):
    from ouroboros.review_waiting import publish_delivery
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.tools.tool_result import append_published_text
    if notes := getattr(ctx, "_commit_call_notes", None):  # fields this call set aside or filled (06.11)
        text, ctx._commit_call_notes = append_published_text(ctx, text, "\nNote: " + " ".join(notes)), []
    record_id = str(getattr(ctx, "_current_review_record_id", "") or "")
    record = load_record(ledger_root(ctx), record_id) if record_id else None
    return publish_delivery(ctx, text, record) if record else text


def note_commit_call(ctx, text):
    """One sentence the commit call's result ends with: a field this call set aside or filled (06.11)."""
    notes = getattr(ctx, "_commit_call_notes", None)
    if isinstance(notes, list) and text not in notes:
        notes.append(text)


class ContinuationWaits(ValueError):
    """A returned commit outcome whose reviewers are still running: the author's decision waits for them."""


class ContinuationUnread(ValueError):
    """The review record holds an outcome the decision's reference does not name (#2902)."""


def _seat_outcome(seat):
    """What a seat row shows, comparable between an attempt row and its ledger record (``review_ledger._apply_raw``)."""
    from ouroboros.review_ledger import ANSWERED_STATUSES

    outcome = seat.get("outcome") or {}
    return (str(seat.get("operation_id") or ""), str(outcome.get("outcome_id") or seat.get("outcome_id") or ""),
            str(outcome.get("status") or seat.get("status") or "") in ANSWERED_STATUSES)


def shown_field(ctx, row):
    """The reference field of a handback rendering attempt ``row``: its record's outcomes that the row itself shows, seat
    by seat, under the same operation and outcome. A writer settles the record before the row, so a seat the row shows
    running, or closed with an earlier outcome (an answer after a failure, a continued seat's successor), is not named.
    Empty when the row is bound to no readable record."""
    from ouroboros.review_ledger import ledger_root, load_record, seat_open
    from ouroboros.review_state import _attempt_review_roster_rows
    from ouroboros.review_waiting import shown_outcomes

    try:
        record = load_record(ledger_root(ctx), row.review_record_id) if row.review_record_id else None
    except (OSError, ValueError):
        record = None
    if not record:
        return {}
    roster, seats = _attempt_review_roster_rows(row), [seat for seat in record.get("rows") or [] if isinstance(seat, dict)]

    def slot(raw):
        return str(raw.get("slot_id") or raw.get("slot") or "")

    def carried(index, seat):  # joined as ``review_ledger.build_rows`` joins raw rows: by slot, else by position
        raw = next((item for item in roster if slot(item) == str(seat.get("seat_id") or "")), None)
        raw = roster[index] if raw is None and len(roster) == len(seats) and not slot(roster[index]) else raw
        return raw is not None and not seat_open(raw) and (
            not raw.get("late_result_pending") or raw.get("operation_state") == "custody_lost") and _seat_outcome(raw) == _seat_outcome(seat)
    return {SHOWN_OUTCOMES: shown_outcomes(record, [seat for index, seat in enumerate(seats) if carried(index, seat)])}


def require_shown(ctx, reference, source):
    """``source`` when the decision's ``reference`` names every outcome its record holds now, else
    ``ContinuationUnread``: a host-stored answer is no answer the author received (#2902), and the free collection
    returns it under a new reference. An older reference proves no delivery, except of a record with nothing to read; a
    review with no record cannot have gained an outcome. Cyber Pro's judgment stands, with the gap stated."""
    from ouroboros.review_ledger import VERDICT_NOT_DISPATCHED, ledger_root, load_record
    from ouroboros.review_waiting import outcome_states, shown_outcomes
    from ouroboros.tools.review_helpers import cyber_review_authority

    ctx._author_commit_unread = False
    record_id = str(reference.get("review_record_id") or getattr(source, "review_record_id", "") or "")
    record = load_record(ledger_root(ctx), record_id) if record_id else None
    if record is None or reference.get(SHOWN_OUTCOMES) == shown_outcomes(record) or (
            SHOWN_OUTCOMES not in reference and record.get("state") == "settled"
            and (not outcome_states(record) or (record.get("verdict") or {}).get("aggregate") == VERDICT_NOT_DISPATCHED)):
        return source
    if not cyber_review_authority():
        raise ContinuationUnread("its review record now holds an outcome this review_reference did not return")
    ctx._author_commit_unread = True  # the landing check states it beside the decision's evidence
    text = "Cyber Pro: the review record holds an outcome this review_reference did not return; this decision did not read it."
    advisory = getattr(ctx, "_review_advisory", None)
    if isinstance(advisory, list) and text not in advisory:
        advisory.append(text)
    return source


def _state_unread_evidence(ctx, source):
    """The Cyber Pro decision's evidence, on its row and record: what it holds was not all returned to this author."""
    from ouroboros.review_doctrine import EVIDENCE_KIND
    from ouroboros.review_ledger import ledger_root, note_author_decision

    author = getattr(ctx, "_author_commit_record", None)
    snapshot = author.get(EVIDENCE_KIND) if isinstance(author, dict) else None
    if getattr(ctx, "_author_commit_unread", False) and isinstance(snapshot, dict) and not snapshot.get("unread_outcomes"):
        snapshot["unread_outcomes"] = True
        if source.review_record_id:
            note_author_decision(ledger_root(ctx), source.review_record_id, author)


def commit_landing_refusal(ctx, commit_message, started_at, managed_tx):
    """The check at the last moment before ``git commit`` (tests done, no review-state lock held, Pause undisturbed): an
    outcome settled since the decision was validated sends it back. Later arrivals are concurrent with the commit."""
    import time

    from ouroboros.tools import git as git_mod
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    source = getattr(ctx, "_author_commit_source", None)
    if source is None:
        return ""
    try:
        require_shown(ctx, ctx._author_commit_reference, source)
        _state_unread_evidence(ctx, source)
        return ""
    except (OSError, ValueError) as exc:
        text = continuation_refusal(exc, effect="No commit was created and the files are preserved.")
    try:  # the refusal is recorded and returned even when its cleanup fails
        if managed_tx:  # a managed merge keeps its whole index, retryable
            from supervisor.update_merge import restore_assisted_resolution_after_commit_error
            if restore_assisted_resolution_after_commit_error(managed_tx) is False:
                text += " The managed update's resolution phase was not restored; recover it before retrying."
        else:
            git_mod._unstage_candidate(ctx)
    except Exception as cleanup:  # noqa: BLE001 - named in the refusal, never silent
        text += f" Restoring the index after this refusal failed ({type(cleanup).__name__}: {cleanup}); it may still hold the candidate."
    git_mod._record_commit_attempt(ctx, commit_message, "blocked", block_reason="review_outcome_unread", block_details=text,
                                   duration_sec=time.time() - started_at, phase="preflight")
    return _publish_tool_result(ctx, ToolResult(status="blocked", code="LEGACY_BLOCKED", text=text))


def continuation_refusal(exc: Exception, *, effect: str = "Nothing was staged or committed.") -> str:
    """A refused author continuation names the next call its cause leaves (06.11)."""
    if isinstance(exc, ContinuationWaits):
        step = ("collect that outcome (its review_reference, no author_disposition) once its reviewers answer, "
                "then decide.")
    elif isinstance(exc, ContinuationUnread):
        step = ("collect it at $0 (the same commit tool, this review_reference, no author_disposition), read it, then "
                "decide with the review_reference that call returns.")
    else:
        step = ("decide with the review_reference an outcome you read returned (the last commit_reviewed call names "
                "its own), or call without author_disposition to review the current bytes.")
    return f"⚠️ REVIEW_AUTHOR_CONTINUATION_INVALID: {exc}. {effect} Next: {step}"


def default_commit_continuation(ctx, kwargs) -> str:
    """A decision without ``review_reference``: "" with the decision set aside, so this call asks for the review
    when none of this copy was returned yet; else a refusal that names the exact reference of the latest returned
    outcome. The reference a decision quotes from an outcome it read is what shows that outcome reached the mind: a
    default would let a decision written beside its review in one message land before that review was shown, on a
    live call or a fallback alike (owner 11.10, review 06.11)."""
    if (kwargs.get("author_disposition") is None or kwargs.get("review_reference") is not None
            or kwargs.get("continue_seats") is not None):
        return ""
    import json
    from ouroboros.review_state import _load_state_unlocked, make_repo_key

    repo_key, task_id = make_repo_key(pathlib.Path(ctx.repo_dir)), str(getattr(ctx, "task_id", "") or "")
    try:
        state = _load_state_unlocked(pathlib.Path(ctx.drive_root), strict_attempt_authority=True)
        returned = [attempt for attempt in state.filter_attempts(repo_key=repo_key, task_id=task_id)
                    if attempt.tool_name in {"commit_reviewed", "vcs_commit_reviewed"}
                    and attempt.phase in {"review_only", "late_wait"} and attempt.pre_review_fingerprint]
    except (OSError, ValueError) as exc:
        return (f"⚠️ REVIEW_AUTHOR_CONTINUATION_INVALID: this copy's review state is unreadable ({exc}). Nothing "
                "was staged or committed. Next: name review_reference, or call without author_disposition.")
    if not returned:
        kwargs["author_disposition"] = None
        note_commit_call(ctx, "author_disposition was set aside: no review of this copy was returned yet, so this call "
                         "asks for it; decide after its outcome.")
        return ""
    latest = returned[-1]
    named = {"surface": "commit", "repo_key": repo_key, "task_id": task_id, "tool_name": latest.tool_name,
             "attempt": latest.attempt, "pre_review_fingerprint": latest.pre_review_fingerprint,
             **({"review_record_id": record_id} if (record_id := str(getattr(latest, "review_record_id", "") or "")) else {})}
    return ("⚠️ REVIEW_AUTHOR_CONTINUATION_INVALID: author_disposition names no review_reference, and a decision "
            "answers an outcome it has read. Nothing was staged or committed. Next: decide with the review_reference "
            f"that outcome (attempt {latest.attempt}, the latest returned for this copy) returned; without it, collect "
            f"that outcome at $0 with review_reference={json.dumps(named)} and no author_disposition.")


def set_aside_wrong_step_fields(ctx, continuation, wake_on):
    """A continuation's fields of another step are set aside with a note, never a refusal (06.11): it keeps the
    saved wake selection, and it dispatches no critic for ``since_review`` to brief. Returns the wake to use."""
    if continuation and wake_on is not None:
        wake_on = None
        note_commit_call(ctx, "wake_on was set aside: it shapes a new ask, and a continuation keeps the saved selection.")
    if continuation and getattr(ctx, "_review_since", ""):
        ctx._review_since = ""
        note_commit_call(ctx, "since_review was set aside: it shapes a new wave's brief, and a continuation "
                         "dispatches no critic.")
    return wake_on


def without_broken_subagent_check(ctx, disposition):
    """A malformed optional ``subagent_check`` is a recorded gap, never the refusal of the decision it rides with."""
    from ouroboros.review_doctrine import parse_subagent_check

    if not isinstance(disposition, dict) or disposition.get("subagent_check") is None:
        return disposition
    try:
        parse_subagent_check(disposition["subagent_check"])
    except ValueError as exc:
        note_commit_call(ctx, f"subagent_check was set aside ({exc}); the decision records no helper check.")
        return {key: value for key, value in disposition.items() if key != "subagent_check"}
    return disposition
