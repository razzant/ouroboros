// Producer census: loop_model_call/round_limits/nudges/budget, task_pacing,
// transcript_prefix, loop_forced_finalization, owner_pause and agent budget-pause events.
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
    mcp_browser_bridges_stopped: 'Browser bridge cleanup',
    services_kept: 'Background services kept',
    service_finalization_error: 'Background service cleanup unconfirmed',
    forced_candidate_drift: 'Forced candidate drifted',
};

const WAIT_LABELS = {
    usage_lock_wait: ['Waiting for accounting access', 'Accounting wait ended'],
    control_store_wait: ['Waiting for control storage', 'Control storage wait ended'],
};

// What a cost crossing measured (the producer's typed `budget_axis`). A row without one comes
// from a historical producer, whose notice was advice: it keeps its kind's label below.
const BUDGET_BOUNDS = {
    in_task_cost_ceiling: 'Cost ceiling',
    whole_work_allowance: 'Whole-work cost allowance',
    hard_tree_cap: 'Task tree cost cap',
    shared_wallet: 'Shared wallet',
};
const usd = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);

// The share of the bound spent when the ~80% notice fired, from the numbers the notice itself
// rendered (`deciding_spend_usd` of `base_usd`; the wallet against the headroom it started with).
function spentPercent(evt, wallet) {
    const whole = usd(wallet ? evt.wallet_baseline_usd : evt.base_usd);
    const left = usd(evt.wallet_remaining_known_usd);
    const spent = wallet ? (whole !== null && left !== null ? whole - left : null) : usd(evt.deciding_spend_usd);
    return whole > 0 && spent !== null && spent >= 0 ? Math.round(100 * spent / whole) : null;
}

/** A threshold notice is a fact, so its row states the fact: the bound and the share that
 *  remained or was spent (`''` when the payload does not carry them). */
function budgetCrossingLabel(kind, evt) {
    const milestone = String(evt.milestone || '');
    if (kind === 'time_budget_milestone') return milestone ? `Time budget: ${milestone} remaining` : '';
    const bound = BUDGET_BOUNDS[evt.budget_axis];
    if (!bound) return '';
    const wallet = evt.budget_axis === 'shared_wallet';
    const against = wallet ? ' of starting headroom' : '';
    if (kind === 'cost_budget_milestone') return milestone ? `${bound}: ${milestone}${against} remaining` : '';
    if (kind !== 'cost_budget_wrapup') return '';
    const spent = spentPercent(evt, wallet);
    if (spent === null) return `${bound}: spend threshold crossed`;
    // An own-calls-only reading is a lower bound against the tree, as the notice itself says.
    const floor = evt.spend_basis === 'own_fallback_tree_unknown' ? 'at least ' : '';
    return `${bound}: ${floor}${spent}%${against} spent`;
}

export function taskCheckpointLabel(evt = {}) {
    const kind = String(evt.checkpoint_kind || '');
    if (kind === 'mcp_browser_bridges_stopped' && Array.isArray(evt.bridges) && evt.bridges.length) {
        return evt.bridges.every((bridge) => bridge?.closure === 'confirmed')
            ? 'Browser bridge process closed' : 'Browser bridge cleanup unconfirmed';
    }
    if (Object.hasOwn(WAIT_LABELS, kind)) return WAIT_LABELS[kind][evt.phase === 'ended' ? 1 : 0];
    if (!kind && Number(evt.checkpoint_number) > 0) {
        return `Checkpoint ${evt.checkpoint_number} — periodic self-check`;
    }
    return budgetCrossingLabel(kind, evt) || (Object.hasOwn(CHECKPOINT_LABELS, kind) ? CHECKPOINT_LABELS[kind]
        : kind ? `Checkpoint · ${kind}` : 'Task checkpoint');
}

/** What tells one checkpoint row from another of its kind. A threshold crossing is the bound
 *  it measured and the threshold it crossed (its payload carries no round, so a round could
 *  not tell 50% from 25%); a wait is its episode and phase; any other is its number or round. */
export function taskCheckpointIdentity(evt = {}) {
    if (Object.hasOwn(WAIT_LABELS, evt.checkpoint_kind)) return `${evt.episode_id || evt.ts || ''}:${evt.phase}`;
    if (evt.milestone || evt.budget_axis) return `${evt.budget_axis || ''}:${evt.milestone || ''}`;
    return evt.checkpoint_number || evt.round || '';
}

export function checkpointHasProgressRow(evt = {}) {
    // These exact producer paths already emit a visible host progress note.
    return (!evt.checkpoint_kind && Number(evt.checkpoint_number) > 0)
        || (evt.checkpoint_kind === 'context_reclaim_manual'
            && ['checkpoint_failed', 'summarizer_failed', 'binding_mismatch'].includes(evt.status));
}
