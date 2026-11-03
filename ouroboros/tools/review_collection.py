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
    from ouroboros.review_state import _attempt_review_roster_rows, load_state, make_repo_key
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
    # A wave reserves its roster before any seat sends and is paid only at its first send (#2410): the
    # reserved operations are the custody this reference collects, whether or not one was sent yet.
    reserved = attempt is not None and any(row.get("operation_id") for row in _attempt_review_roster_rows(attempt))
    if not (attempt is not None and (attempt.paid or reserved)) or (
            attempt.raw_stripped and record.get("state") != "settled"):
        raise ValueError("the original paid attempt or its recorded roster is unavailable")
    if (attempt.tool_name == "commit_reviewed") != commit:
        raise ValueError("collect this reference with its original review tool")
    if commit and attempt.repo_key != make_repo_key(pathlib.Path(ctx.repo_dir)):
        raise ValueError("the reference belongs to another repository")
    expected = reference_for(record, attempt)
    if any(key not in expected or expected[key] != value for key, value in reference.items()):
        raise ValueError("review_reference identity differs from the recorded paid attempt")
    return root, attempt, record


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
    record, attempt, continuation = continue_or_repair(ctx, root, attempt, record, continue_seats)
    ctx._left_review_wait = copy.deepcopy(record.get("waiting") or {})
    from ouroboros.review_waiting import select_helpers
    select_helpers(ctx)  # Synchronous collectors also restore the saved selection.
    if record.get("state") == "settled":  # a rejoined successor that already settled is read, never sent
        ctx._left_review_wait = None
        return {**review_result(record, reused=True), **({"continuation": continuation} if continuation else {})}
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
    result = review_result(load_record(ledger_root(ctx), record["record_id"]) or settled, reused=not continuation)
    return {**result, "continuation": continuation} if continuation else result


def commit_offer(ctx, record_id):
    """A commit outcome without the Advisory handback (Blocking's block, a refusal after the wave): the exact
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
    return ("\nA reviewer in continuable_seats stopped with work its engine can continue: with the reviewed bytes "
            "staged again, the same commit tool with review_reference and continue_seats (no author_disposition) "
            "continues it in this paid gate.\n" + json.dumps({"review_reference": reference_for(load_record(ledger_root(ctx), record_id), attempt),
                                           "continuable_seats": offers}, ensure_ascii=False, default=str))


def prepare_commit_collection(ctx, reference, *, commit_message="", mutations=(), continue_seats=None):
    """Separate from Advisory author continuation: resume the paid gate itself (``continue_seats``: and its
    chosen stopped seats, armed only after the staged bytes match, ``bind_commit_collection``)."""
    from ouroboros.review_continuation import parse_selections

    _root, attempt, record = resolve_reference(ctx, reference, commit=True)
    if any(mutations) or (commit_message and commit_message != attempt.commit_message):
        raise ValueError("a paid reference cannot change its commit message, brief, panel or author disposition")
    ctx._collect_commit_continue_seats = None if continue_seats is None else (parse_selections(continue_seats) and continue_seats)
    if attempt.status == "succeeded":
        raise ValueError("this paid attempt already committed; collection cannot commit it again")
    ctx._collect_commit_source = attempt
    ctx._review_resume_pending = True
    ctx._pending_review_attempt = attempt
    ctx._current_review_attempt_number = attempt.attempt
    ctx._current_review_record_id = record["record_id"]
    ctx._left_review_wait = copy.deepcopy(record.get("waiting") or {})
    return attempt


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
    """This task's settled commit attempts whose record holds a recorded continuation no projection carried yet."""
    from ouroboros.review_continuation import current_decisions
    from ouroboros.review_ledger import ledger_root, load_record
    from ouroboros.review_state import _attempt_has_active_review_custody, load_state
    from ouroboros.tools.commit_gate import _current_review_tool_name

    found = []
    for attempt in load_state(pathlib.Path(ctx.drive_root)).filter_attempts(
            repo_key=repo_key, tool_name=_current_review_tool_name(ctx), task_id=str(getattr(ctx, "task_id", "") or "")):
        if attempt.review_record_id and attempt.review_retry_key and not _attempt_has_active_review_custody(attempt):
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
        return {"status": "blocked", "message": "REVIEW_BINDING_BLOCKED: current staged bytes differ from the paid reference.",
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
    record_id = str(getattr(ctx, "_current_review_record_id", "") or "")
    record = load_record(ledger_root(ctx), record_id) if record_id else None
    return publish_delivery(ctx, text, record) if record else text
