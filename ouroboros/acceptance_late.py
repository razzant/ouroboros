"""Purpose-bound admission of a frozen delivered acceptance debt.

Execution, accounting, recovery and publication remain with the existing review
owners. This adapter never resumes an author or rebuilds historical evidence.

Authority: a late review needs an owner chat, quiz or mailbox ``owner_source``
newer than the frozen answer; the target may sit in any Project. ``amend_cap``
only records an absolute original-root cap (no panel, no claim); ``review`` runs
the full triad over the frozen terminal root even with review mode off.
Bindings: Stop, Panic, review prohibitions, Pauses/budget fences, calendar
deadlines, original-root and live global caps with in-flight holds. Hurry and
finalize-now block only unstarted automatic work, not a later explicit request;
an automatic panel also needs calendar headroom above the acceptance floor. An
already paid or unknown panel is collected, never replayed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def owner_historical_acceptance(ctx: Any, request: dict) -> dict:
    """Resolve NEW owner authority; only an explicit review enters the shared operation."""
    from ouroboros.acceptance_history import prepare_owner_historical_review

    prepared = prepare_owner_historical_review(ctx, request)
    if prepared['status'] != 'prepared':
        return prepared
    if prepared['action'] == 'amend_cap':
        # Money only: no panel, preparation operation, purpose, paid claim or
        # reservation. A later explicit review spends the amended original cap.
        return {**prepared, 'status': 'amended', 'reason': 'original_root_cap_amended', 'review_requested': False}
    return {**prepared, **run_historical_acceptance(ctx, task_id=prepared['task_id'],
                debt_id=prepared['debt_id'], prepared=prepared)}


def historical_operation_controls(root: Any, purpose: dict, *, admission: bool = False, unstarted: bool = False,
                                  pending_panel: dict | None = None) -> list[str]:
    from ouroboros.acceptance_history import historical_review_controls, read_acceptance_history
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now
    from ouroboros.task_results import load_task_result

    try:
        row = load_task_result(root, purpose['task_id'], strict=True) or {}
        debt = row.get('acceptance_debt') or {}
        if (purpose.get('schema_version') != 1 or debt.get('debt_id') != purpose.get('debt_id')
                or debt.get('source_ref') != purpose.get('source_ref')
                or debt.get('delivery') != purpose.get('delivery')
                or debt.get('accounting_root_task_id') != purpose.get('accounting_root_task_id')):
            return ['historical_subject_changed']
        frozen = read_acceptance_history(root, row['task_id'], debt)
        blocked = historical_review_controls(root, row, debt, caller_task_id=purpose.get('caller_task_id', ''),
            automatic=(admission or unstarted) and purpose['automatic'], historical_contract=frozen.get('task_contract') or {},
            check_paid=admission, check_pause=admission or unstarted, pending_panel=pending_panel)
        deadline = purpose.get('caller_deadline_at')
        if deadline and (parse_deadline_ts(deadline) is None or parse_deadline_ts(deadline) <= utc_now()):
            blocked.append('caller_deadline')
        return blocked
    except Exception:
        return ['historical_control_authority_unavailable']


def owner_paused_only(root: Any, purpose: dict, blocked: list) -> bool:
    """Only the accounting root's closed owner Pause blocks this work: unsent, it waits for Resume (D10)."""
    from ouroboros.owner_pause import fence_closed, read_fence

    if not blocked or any(item != 'root_budget_fence' and not str(item).startswith('pause:') for item in blocked):
        return False
    try:
        return fence_closed(read_fence(root, purpose['accounting_root_task_id']))
    except Exception:
        return False


def _receipt(root: Path, debt: dict) -> dict | None:
    from ouroboros.acceptance_history import historical_receipt_matches
    from ouroboros.artifacts import read_actor_source_bytes
    from supervisor.terminal_delivery import terminal_answer_receipts
    import json

    for receipt in terminal_answer_receipts(root, debt['task_id'])['delivered']:
        delivery = debt['delivery']
        if (receipt.get('basis') != 'send_handler_returned' or not receipt.get('source_ref')
                or any(receipt.get(key) != delivery.get(key) for key in ('task_id', 'delivery_id', 'text_sha256'))):
            continue
        captured = json.loads(read_actor_source_bytes(root, debt['task_id'], receipt['source_ref']))
        routing = captured.get('routing') or {}
        addressed = (historical_receipt_matches(delivery, receipt) or (
            routing.get('basis') == 'terminal_sender_bound_project'
            and routing.get('intended_chat_id') == delivery['chat_id']
            and routing.get('routed_chat_id') == receipt['chat_id']))
        if (addressed and all(captured.get(key) == receipt.get(key) for key in ('task_id', 'delivery_id', 'chat_id'))
                and captured.get('basis') == 'send_handler_returned'
                and hashlib.sha256(captured['text'].encode()).hexdigest() == delivery['text_sha256']):
            return receipt
    return None


def historical_preclaim_refusal(ctx: Any) -> str:
    """Recheck the purpose at the existing strict cycle-claim boundary."""
    from ouroboros.task_results import load_task_result

    purpose = ctx.historical_purpose
    # The paid stamp captures its preparer's owner. Physical workers carry a
    # narrow wait context, not the preparation thread's operation ContextVar.
    operation = getattr(ctx, 'historical_operation', None)
    pending_panel = None
    if (operation is not None and not operation.closed and operation.checkpointed
            and operation.result_root == Path(ctx.drive_root) and operation.historical_purpose == purpose):
        row = load_task_result(ctx.drive_root, purpose['task_id'], strict=True) or {}
        pointer = (row.get('review_operations') or {}).get(operation.owner_id) or {}
        if (pointer.get('state') in {'retained', 'dispatched'} and pointer.get('source_ref')
                and pointer.get('controller') == operation.identity and pointer.get('retry_key') == operation.retry_key
                and pointer.get('task_attempt') == purpose['task_attempt']
                and pointer.get('intent_ref') == operation.intent_ref):
            pending_panel = {'binding_hash': ctx.review_binding['binding_hash'],
                             'operations': pointer.get('operations') or {}}
    blocked = historical_operation_controls(ctx.drive_root, purpose, admission=True, pending_panel=pending_panel)
    if blocked:
        return blocked[0]
    debt = load_task_result(ctx.drive_root, purpose['task_id'], strict=True)['acceptance_debt']
    receipt = _receipt(Path(ctx.drive_root), debt)
    return '' if receipt and all(receipt.get(key) == value for key, value in
        purpose['confirmed_delivery'].items()) else 'exact_delivery_unconfirmed'


def _collect_existing(ctx: Any, row: dict, retry_key: str, *, collect: bool = True) -> dict:
    # Automatic notifications run on the sender's event drain. Existing duty
    # belongs to ordinary operation maintenance, including ready/unknown work
    # and the race where another entry retained the pointer during handoff.
    if not collect:
        return {'status': 'deferred', 'reason': 'existing_operation_collection_owned', 'dispatched': False}
    from ouroboros.acceptance_settlement import settle_acceptance_operation

    pointers = [(owner, item) for owner, item in (row.get('review_operations') or {}).items()
                if item.get('retry_key') == retry_key]
    intents = [item for _owner, item in pointers if not item.get('source_ref')]
    if intents and not any(item.get('source_ref') for _owner, item in pointers):
        latest = intents[-1]
        return latest.get('preparation_outcome') or {
            'status': 'preparing' if latest.get('state') == 'preparing' else 'unknown',
            'reason': 'existing_historical_preparation', 'dispatched': None}
    states = [settle_acceptance_operation(ctx, task_id=row['task_id'], retry_key=retry_key, result=row,
                checkpoint={**item, 'owner_id': owner}, controller=item.get('controller')) for owner, item in pointers]
    if not pointers:
        states = [settle_acceptance_operation(ctx, task_id=row['task_id'], retry_key=retry_key, result=row)]
    return {'status': 'collected' if any(s in {'published', 'announced', 'settled'} for s in states) else 'unknown',
            'reason': 'existing_paid_operation', 'dispatched': False, 'collection': states}


def run_historical_acceptance(ctx: Any, *, task_id: str, debt_id: str,
                              prepared: dict | None = None, automatic: bool = False) -> dict:
    """Hand identity to the existing operation BEFORE expensive preparation."""
    from ouroboros.review_operation import prepare_historical_operation
    from ouroboros.task_results import load_task_result, load_task_acceptance_review_state
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.tools.review import _owner_deadline_at

    root = canonical_data_root(ctx)
    row = load_task_result(root, task_id, strict=True) or {}
    debt = row.get('acceptance_debt') or {}
    refused = lambda reason: {'status': 'owed', 'reason': reason, 'dispatched': False,
                              'task_id': task_id, 'debt_id': debt_id}
    if not debt_id or debt.get('debt_id') != debt_id:
        return refused('historical_debt_not_found')
    if not automatic and (not prepared or prepared.get('status') != 'prepared' or not prepared.get('owner_source_ref')):
        return refused('explicit_owner_request_required')
    paid = hashlib.sha256(debt_id.encode()).hexdigest()
    retry_key = 'task_acceptance:' + paid
    claims = load_task_acceptance_review_state(root, debt['accounting_root_task_id'], require_root_result=True)['claims_by_binding']
    existing = any(p.get('retry_key') == retry_key and (automatic or p.get('state') != 'preparation_refused')
                   for p in (row.get('review_operations') or {}).values())
    if existing or any(claim.get('paid_identity') == paid for claim in claims.values()):
        return _collect_existing(ctx, row, retry_key, collect=not automatic)
    receipt = _receipt(root, debt)
    if not receipt:
        return refused('exact_delivery_unconfirmed')
    purpose = {key: debt[key] for key in ('task_id', 'task_attempt', 'debt_id', 'source_ref', 'delivery', 'accounting_root_task_id')}
    purpose.update(confirmed_delivery={key: receipt[key] for key in ('task_id', 'delivery_id', 'chat_id', 'text_sha256', 'source_ref')},
                   schema_version=1, automatic=automatic, caller_task_id='' if automatic else str(ctx.task_id),
                   caller_deadline_at='' if automatic else _owner_deadline_at(ctx),
                   owner_source_ref=None if automatic else prepared['owner_source_ref'])
    # Fast owner exclusions avoid creating automatic preparation duty for mode
    # off. Full frozen/current controls and money are checked by the owner worker.
    if automatic:
        from ouroboros.config import get_task_review_mode
        if get_task_review_mode() == 'off':
            return refused('review_mode_off')
    try:
        return prepare_historical_operation(root=root, purpose=purpose, event_queue=getattr(ctx, 'event_queue', None),
            background=automatic, work=lambda operation: _run_historical_acceptance(
                ctx, task_id=task_id, debt_id=debt_id, prepared=prepared, automatic=automatic, operation=operation))
    except Exception:
        latest = load_task_result(root, task_id, strict=True) or {}
        if any(p.get('retry_key') == retry_key for p in (latest.get('review_operations') or {}).values()):
            return _collect_existing(ctx, latest, retry_key, collect=not automatic)
        return refused('historical_preparation_handoff_failed')


def handoff_delivered_acceptance(ctx: Any, task_id: str, delivery_id: str) -> None:
    """Join canonical publication and exact receipt, whichever becomes ready last."""
    from ouroboros.headless import terminal_task_files_ready
    from ouroboros.task_results import load_task_result
    from supervisor.terminal_delivery import _TERMINAL_ANSWER_ID

    match = _TERMINAL_ANSWER_ID.fullmatch(delivery_id)
    if not match or match['task'] != task_id:
        return
    row = load_task_result(ctx.DRIVE_ROOT, task_id, strict=True) or {}
    debt = row.get('acceptance_debt') or {}
    if (debt.get('delivery') or {}).get('delivery_id') != delivery_id:
        return
    running = getattr(ctx, 'RUNNING', {}) or {}
    task = (running.get(task_id) or {}).get('task') or {}
    if not terminal_task_files_ready(ctx.DRIVE_ROOT, {**row, **task, 'id': task_id}, row):
        return
    owner = SimpleNamespace(task_id=task_id, drive_root=ctx.DRIVE_ROOT,
        task_metadata={'budget_drive_root': str(ctx.DRIVE_ROOT)}, event_queue=getattr(ctx, 'event_queue', None))
    run_historical_acceptance(owner, task_id=task_id, debt_id=debt.get('debt_id', ''), automatic=True)


def _historical_writer_live(root: Path, task_id: str, operation_id: str) -> bool:
    """Use this venue's authority; worker imports of supervisor maps prove nothing."""
    from ouroboros.post_task_checkpoint import post_task_synthesis_is_terminal
    from ouroboros.review_operation import task_has_live_review_operation
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import task_has_live_queue_ownership, queue_snapshot_observation
    from ouroboros.utils import in_worker_process, read_json_dict
    from supervisor import queue as task_queue
    from supervisor.direct_roots import FRAGMENT_NAME

    try:
        if (not in_worker_process() and task_queue.INITIALIZED
                and Path(task_queue.DRIVE_ROOT).resolve() == root.resolve()):
            return task_queue.task_has_live_ownership(task_id, ignore_review_operation=operation_id)
        if task_has_live_review_operation(root, task_id, exclude_owner_id=operation_id):
            return True
        row = load_task_result(root, task_id, strict=True) or {}
        direct = row.get('_is_direct_chat')
        if direct is not True and task_has_live_queue_ownership(root, task_id):
            return True
        if direct is False:
            return False
        # Direct turns do not enter the pooled snapshot. Their existing fragment
        # omits postwork, so absence also needs its terminal phase checkpoint.
        fragment = read_json_dict(root / FRAGMENT_NAME) or {}
        roots = fragment.get('roots')
        if (not queue_snapshot_observation(fragment)['fresh'] or fragment.get('incomplete') is not False
                or not isinstance(roots, list) or any(not isinstance(item, dict) for item in roots)):
            return True
        return (any(item.get('task_id') == task_id for item in roots)
                or not post_task_synthesis_is_terminal((row.get('root_phase_checkpoint') or {}).get('post_task_synthesis')))
    except Exception:
        return True


def resume_paused_acceptance_preparations(root: Any, task_id: str, fence: dict) -> int:
    """Rebind only Pause-attested unsent intents; caller holds the exact root launch lock.

    Live preparers already wait for this fence. Dead ones reuse their owner/debt
    identity with a compare-and-set before handoff. Paid/unknown work stays with
    its collector, and unreadable sources never authorize another preparation.
    """
    from ouroboros import review_operation as operations
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import load_task_acceptance_review_state

    restart = []
    for owner, subject, entry in operations.paused_acceptance_preparations(root, task_id, fence['fence_id']):
        pause = entry['preparation_pause']
        if (entry.get('state') != operations.OPERATION_PREPARING or entry.get('source_ref')
                or pause.get('intent_ref') != entry.get('intent_ref')
                or pause.get('controller') != entry.get('controller')
                or pause.get('generation') != fence.get('generation')):
            raise ValueError('late_preparation_authority_unknown')
        with operations._LOCK:
            live = operations._LIVE.get(owner)
        controller = operations.controller_state(entry.get('controller'))
        if (live is not None and not live.closed) or controller == 'alive':
            continue
        if controller != 'dead' and entry.get('controller') != operations.controller_identity():
            raise ValueError('late_preparation_controller_unknown')
        intent = json.loads(read_actor_source_bytes(root, subject, entry['intent_ref']))
        purpose = intent.get('purpose') or {}
        if (intent.get('kind') != 'historical_acceptance_preparation_intent' or intent.get('owner_id') != owner
                or intent.get('controller') != entry.get('controller') or purpose.get('task_id') != subject
                or purpose.get('task_attempt') != entry.get('task_attempt')
                or purpose.get('accounting_root_task_id') != task_id
                or entry.get('retry_key') != 'task_acceptance:' + hashlib.sha256(purpose['debt_id'].encode()).hexdigest()):
            raise ValueError('late_preparation_source_mismatch')
        claims = load_task_acceptance_review_state(root, task_id, require_root_result=True)['claims_by_binding']
        if any(entry['retry_key'] == 'task_acceptance:' + str(claim.get('paid_identity') or '')
               for claim in claims.values()):
            raise ValueError('late_preparation_dispatch_unknown')
        restart.append((owner, subject, entry, purpose))
    from supervisor.workers import get_event_q
    for owner, subject, entry, purpose in restart:
        ctx = SimpleNamespace(task_id=purpose.get('caller_task_id') or subject, task_attempt=purpose['task_attempt'],
            drive_root=root, budget_drive_root=root, task_metadata={'root_task_id': task_id},
            event_queue=get_event_q(), pending_events=[])
        prepared = {'status': 'prepared', 'owner_source_ref': purpose.get('owner_source_ref')}
        operations.prepare_historical_operation(root=root, purpose=purpose, event_queue=ctx.event_queue,
            background=True, resume_entry={'owner_id': owner, 'entry': entry},
            work=lambda operation, ctx=ctx, subject=subject, purpose=purpose, prepared=prepared:
                _run_historical_acceptance(ctx, task_id=subject, debt_id=purpose['debt_id'], prepared=prepared,
                                          automatic=purpose['automatic'], operation=operation))
    return len(restart)


def _run_historical_acceptance(ctx: Any, *, task_id: str, debt_id: str,
                              prepared: dict | None, automatic: bool, operation: Any) -> dict:
    """Run one late panel or collect its existing identity, without author work."""
    from ouroboros.acceptance_history import effective_original_root_cap, read_acceptance_history
    from ouroboros.acceptance_settlement import remember_settlement_trace, settle_acceptance_operation
    from ouroboros.acceptance_retrieving import acceptance_retrieving_work_order, retain_review_source
    from ouroboros.config import adaptive_quorum
    from ouroboros.loop_acceptance_review import _ACCEPTANCE_REVIEW_CHECKLIST
    from ouroboros.model_wait import task_model_wait_scope
    from ouroboros.review_dispatch import bind_task_acceptance_paid_dispatch, task_acceptance_preclaim_refusal
    from ouroboros.review_evidence_sections import acceptance_packet_budget_chars
    from ouroboros.review_projection import build_review_binding, publish_acceptance_checkpoint
    from ouroboros.review_source_closure import retain_review_request_sources
    from ouroboros.review_substrate import ReviewRequest, run_review_request, _messages_char_count, _request_messages
    from ouroboros.reviewer_slot_config import triad_delivery_slots
    from ouroboros.task_results import load_task_result, load_task_acceptance_review_state, resolve_task_lineage
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.usage_accounting import UsageScope, usage_scope, review_wave_admission

    receipt_verified = False
    refused = lambda reason: {'status': 'owed', 'reason': reason, 'dispatched': False,
                              'task_id': task_id, 'debt_id': debt_id,
                              'delivery_status': 'delivered' if receipt_verified else 'unconfirmed'}
    try:
        root = canonical_data_root(ctx)
        row = load_task_result(root, task_id, strict=True) or {}
        debt = row.get('acceptance_debt') or {}
        if not debt_id or debt.get('debt_id') != debt_id:
            return refused('historical_debt_not_found')
        frozen = read_acceptance_history(root, task_id, debt)
        if not automatic and (not prepared or prepared.get('status') != 'prepared'
                              or not prepared.get('owner_source_ref')):
            return refused('explicit_owner_request_required')
        paid = hashlib.sha256(debt_id.encode()).hexdigest()
        retry_key = 'task_acceptance:' + paid
        accounting_id = debt['accounting_root_task_id']
        usage_ctx = SimpleNamespace(task_id=task_id, task_attempt=debt['task_attempt'], drive_root=root,
            budget_drive_root=root, task_metadata={'root_task_id': accounting_id, 'budget_drive_root': str(root)},
            event_queue=getattr(ctx, 'event_queue', None), pending_events=[])
        claims = load_task_acceptance_review_state(root, accounting_id, require_root_result=True)['claims_by_binding']
        if (any(claim.get('paid_identity') == paid for claim in claims.values())
                or any(p.get('retry_key') == retry_key and p.get('state') != 'preparation_refused'
                       for owner, p in (row.get('review_operations') or {}).items() if owner != operation.owner_id)):
            return _collect_existing(usage_ctx, row, retry_key)
        purpose = operation.historical_purpose
        blocked = historical_operation_controls(root, purpose, admission=True)
        if blocked and not owner_paused_only(root, purpose, blocked):
            return refused(blocked[0])
        if not _receipt(root, debt):
            return refused('exact_delivery_unconfirmed')
        receipt_verified = True
        while True:
            # The existing operation holds this preparation drain until the original
            # terminal writer releases custody — and, automatic and still unsent,
            # through an owner Pause of that root until its Resume — without a scheduler.
            live = _historical_writer_live(root, task_id, operation.owner_id)
            blocked = [] if live else historical_operation_controls(root, purpose, admission=True)
            if not live and not owner_paused_only(root, purpose, blocked):
                break
            if live and not automatic:
                return refused('historical_writer_still_live')
            if operation.control():
                return refused('historical_preparation_cancelled')
            from ouroboros.budget_pause import _HOLD_POLL_SEC
            time.sleep(_HOLD_POLL_SEC)  # existing owner-hold observation cadence
        if blocked:
            return refused(blocked[0])
        lineage = resolve_task_lineage(task_id, metadata=row.get('metadata'), **{
            key: row.get(key) for key in ('root_task_id', 'parent_task_id', 'delegation_role', 'original_task_id', 'timeout_retry_from')})
        if not lineage['is_root_task'] or lineage['root_task_id'] != accounting_id:
            return refused('historical_root_lineage_unavailable')
        usage_ctx.task_metadata.update(lineage)
        cap = effective_original_root_cap(debt, load_task_result(root, accounting_id, strict=True))
        if cap['state'] not in {'finite', 'unlimited'} or (cap['state'] == 'finite' and (
                type(cap.get('usd')) not in (int, float) or not math.isfinite(cap['usd']) or cap['usd'] < 0)):
            return refused('original_root_cap_unknown')
        slots = triad_delivery_slots(role_hint='task acceptance')
        if not slots:
            return refused('no_review_slots')
        binding = {**build_review_binding(candidate=frozen['answer'], evidence={'historical_source': debt['source_ref']},
                                          fence_token_or_state=debt_id), 'paid_identity': paid}
        from ouroboros.deadline_utils import parse_deadline_ts

        deadlines = [value for value in ((frozen.get('task_contract') or {}).get('deadline_at'),
                     purpose['caller_deadline_at']) if value]
        request = ReviewRequest(surface='task_acceptance', task_id=task_id, task_attempt=debt['task_attempt'],
            subject=frozen['answer'], goal='Assess the frozen delivered answer against its captured owner requirements.',
            checklist=_ACCEPTANCE_REVIEW_CHECKLIST, evidence={'historical_subject': frozen}, retry_key=retry_key,
            deadline_at=min(deadlines, key=parse_deadline_ts) if deadlines else '',
            drain_deadline=time.monotonic(), policy={'historical_acceptance': purpose, 'original_root_cap': cap,
                'hardness': 'advisory_visible', 'min_successful_slots': adaptive_quorum(len(slots)),
                'fail_closed_on_errors': True, 'classify_outcome_tier': True,
                'slot_input_caps': acceptance_packet_budget_chars(slots).slot_input_caps})
        retain_review_request_sources(request, source_root=root, custody_root=root, historical_debt=debt)
        # Historical readers get the retained view as their workspace, never the
        # author's current checkout or a reconstructed artifact manifest.
        request.session_root = request.policy['native_data_root']
        acceptance_retrieving_work_order(request, [slot for slot in slots if slot.retrieves],
            session_root=request.session_root, data_root=Path(request.policy['native_data_root']))
        for slot in slots:
            if slot.retrieves:
                retain_review_source(request, slot.slot_id, root)
        from ouroboros.usage_admission import task_billing_fields, effective_billing_fields
        billing = task_billing_fields({'id': task_id}, accounting_id, cap.get('usd'), root)
        billing = effective_billing_fields(root, accounting_id, billing)
        scope = UsageScope(drive_root=root, task_id=task_id, root_task_id=accounting_id,
            source='historical_acceptance', **{**billing,
                'root_limit_source': str(cap.get('source') or 'original_admission')})
        admission = SimpleNamespace(tools=SimpleNamespace(_ctx=usage_ctx), task_id=task_id, drive_root=root,
            review_binding=binding, historical_purpose=purpose, historical_operation=operation,
            purpose='' if automatic else 'owner_historical_acceptance')
        with usage_scope(scope):
            paid_slots = [slot for slot in slots if str(getattr(slot.route, 'value', slot.route)) != 'agent_session']
            if paid_slots:
                wave = review_wave_admission(root, root_task_id=accounting_id, task_id=task_id,
                    root_limit_usd=scope.root_limit_usd, root_limit_source=scope.root_limit_source,
                    models=[slot.model for slot in paid_slots],
                    prompt_chars=[_messages_char_count(_request_messages(request, slot)) for slot in paid_slots],
                    categories='task_acceptance_review', slot_ids=[slot.slot_id for slot in paid_slots],
                    processing_preferences=[slot.processing_preference for slot in paid_slots])
                if not wave.get('fits'):
                    return refused('review_wave_budget_insufficient')
            refusal = task_acceptance_preclaim_refusal(admission)
            if refusal is not None:
                return refused(refusal.degraded_reasons[0])
            task = {'id': task_id, '_attempt': debt['task_attempt'], 'chat_id': purpose['confirmed_delivery']['chat_id'],
                    'metadata': usage_ctx.task_metadata, 'task_contract': frozen.get('task_contract') or {}}
            with task_model_wait_scope(task=task, drive_root=root, event_queue=usage_ctx.event_queue,
                    worker_slot_held=False, owner_control=lambda: 'cancelled' if historical_operation_controls(root, purpose) else None):
                with bind_task_acceptance_paid_dispatch(admission):
                    result = run_review_request(request, slots=slots, drive_root=root, usage_ctx=usage_ctx)
        if not result.actors or all(actor.get('operation_state') == 'not_dispatched' for actor in result.actors):
            return refused('; '.join(result.degraded_reasons) or 'no_physical_dispatch')
        run = {**dataclasses.asdict(result), **binding, 'authority': 'host_root',
               'accounting_root_task_id': accounting_id, 'task_attempt': debt['task_attempt'], 'completion_impact': 'supplement_only'}
        trace = {'review_runs': [run]}
        remember_settlement_trace(usage_ctx, trace, run)
        publish_acceptance_checkpoint(usage_ctx, trace, task_id=task_id, drive_root=root,
                                      chat_id=purpose['confirmed_delivery']['chat_id'], partial_trace=True)
        status = settle_acceptance_operation(usage_ctx, retry_key=retry_key, task_id=task_id, result=row)
        return {'status': status, 'task_id': task_id, 'debt_id': debt_id, 'paid_identity': paid,
                'panel_id': run['panel_id'], 'dispatched': True if any(a.get('status') == 'ok' for a in result.actors) else None,
                'delivery_status': 'delivered', 'source_ref': run.get('applied_source_ref')}
    except Exception as exc:
        # A failed handoff is never an LLM retry. Durable claims/pointers, if
        # already written, remain exclusively with the existing collection owner.
        result = refused(f'historical_review_unavailable:{type(exc).__name__}:{exc}')
        if 'retry_key' in locals():
            latest = load_task_result(root, task_id) or {}
            if any(p.get('retry_key') == retry_key and p.get('source_ref') for p in (latest.get('review_operations') or {}).values()):
                result.update(status='unknown', dispatched=None)
        return result


def supplement_chat(root: Any, event: dict) -> int | None:
    """A canonical historical supplement keeps its proven original recipient."""
    from ouroboros.task_results import load_task_result
    from supervisor.terminal_delivery import pending_deliveries

    row = load_task_result(root, str(event.get('task_id') or ''), strict=True) or {}
    evidence = (event.get('progress_meta') or {}).get('late_evidence') or {}
    for panel in (row.get('review_projection') or {}).get('panels') or []:
        late = panel.get('late_settlement') or {}
        delivery = late.get('historical_delivery') or {}
        if (delivery and panel.get('panel_id') == evidence.get('panel_id')
                and late.get('note') == event.get('text')
                and event.get('delivery_id') == 'acceptance-late:' + str((late.get('reviewed_subject') or {}).get('retry_key'))):
            if panel.get('applied_source_ref') == evidence.get('source_ref'):
                return int(delivery['chat_id'])
            # Concurrent collection can publish a newer source for this same
            # settlement. The outbox retains its FIRST notice and exact source;
            # that durable event still belongs to the proven historical room.
            if any(owed.get('chat_id') == delivery.get('chat_id') and all(
                    owed.get(key) == event.get(key) for key in
                    ('task_id', 'delivery_id', 'chat_id', 'text', 'system_type', 'progress_meta'))
                    for owed in pending_deliveries(root)):
                return int(delivery['chat_id'])
    return None
