// Producer census: loop_model_call/round_limits/nudges/budget, task_pacing,
// transcript_prefix, loop_forced_finalization and agent budget-pause events.
export const CHECKPOINT_LABELS = {
    context_fit_low_retry: 'Context rebuilt in Low mode',
    context_fit_route_rebound: 'Context fit rebound the route',
    context_fit_physical_mode: 'Context starts in a smaller mode (window too small)',
    context_view: 'Context inspected',
    context_reclaim_manual: 'Context reclaimed on request',
    context_reclaim_automatic: 'Context reclaimed automatically',
    prompt_prefix_break: 'Prompt prefix rebuilt',
    cost_budget_milestone: 'Cost budget milestone',
    cost_budget_wrapup: 'Cost budget wrap-up',
    time_budget_milestone: 'Time budget milestone',
    intrinsic_pacing: 'Pacing check',
    nanny_economics_reminder: 'Economics reminder',
    nanny_finalization_nudge: 'Finalization nudge',
    budget_scope_paused: 'Budget scope paused',
    services_stopped: 'Background services stopped',
    services_kept: 'Background services kept',
    forced_candidate_drift: 'Forced candidate drifted',
};

export function taskCheckpointLabel(evt = {}) {
    const kind = String(evt.checkpoint_kind || '');
    if (kind === 'usage_lock_wait') return evt.phase === 'ended' ? 'Accounting wait ended' : 'Waiting for accounting access';
    if (!kind && Number(evt.checkpoint_number) > 0) {
        return `Checkpoint ${evt.checkpoint_number} — periodic self-check`;
    }
    return Object.hasOwn(CHECKPOINT_LABELS, kind) ? CHECKPOINT_LABELS[kind]
        : kind ? `Checkpoint · ${kind}` : 'Task checkpoint';
}

export function checkpointHasProgressRow(evt = {}) {
    // These exact producer paths already emit a visible host progress note.
    return (!evt.checkpoint_kind && Number(evt.checkpoint_number) > 0)
        || (evt.checkpoint_kind === 'context_reclaim_manual'
            && ['checkpoint_failed', 'summarizer_failed', 'binding_mismatch'].includes(evt.status));
}
