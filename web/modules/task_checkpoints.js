import { causeOf } from './context_observation.js';
import { fmt } from './i18n.js';
import { statusMarkHtml } from './status_mark.js';

// Producer census: loop_model_call/round_limits/nudges/budget, task_pacing,
// transcript_prefix, loop_forced_finalization, owner_pause and agent budget-pause events.
export const CHECKPOINT_LABELS = {
    context_fit_low_retry: 'Books shown as chapter navigation during context-size recovery',
    context_fit_refused: 'Context-size check found no room for results',
    context_fit_route_rebound: 'Context fit rebound the route',
    context_fit_physical_mode: 'Context starts in a smaller mode',
    context_view: 'Context inspected',
    context_reclaim_emergency: 'Context shrunk during context-size recovery (no model)',
    context_memory_coarsened: 'Memory shown coarser during context-size recovery',
    context_view_renewed: 'Working view renewed: same task, fresh window',
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
    service_finalization_error: 'Background service cleanup unconfirmed',
    forced_candidate_drift: 'Forced candidate drifted',
};

const WAIT_LABELS = {
    usage_lock_wait: ['Waiting for accounting access', 'Accounting wait ended', 'Accounting waits'],
    control_store_wait: ['Waiting for control storage', 'Control storage wait ended', 'Control storage waits'],
    // The Safety Supervisor's check of one tool call; an open one also shows on the closed card's face (chat.js).
    safety_check_wait: ['Checking safety', 'Safety check ended', 'Safety checks'],
};

function waitSeconds(s) {
    if (s < 10) return `${s.toFixed(1)} s`;
    const whole = Math.round(s);
    return whole < 60 ? `${whole} s` : `${Math.floor(whole / 60)} min${whole % 60 ? ` ${whole % 60} s` : ''}`;
}

/** A host wait (accounting, control storage, a Safety check) is how the host ran, not the turn's work. Its episode
 *  facts, or null for any other checkpoint; the producer still writes both phases of every episode.
 *  Python twin: ``ouroboros/checkpoint_waits.py`` (cases in ``web/tests/fixtures/wait_fold_parity.json``). */
export function waitEpisode(evt = {}) {
    if (!Object.hasOwn(WAIT_LABELS, evt.checkpoint_kind) || !evt.episode_id) return null;
    const elapsed = evt.elapsed_sec;
    return { kind: evt.checkpoint_kind, id: String(evt.episode_id), ended: evt.phase === 'ended',
        elapsed: typeof elapsed === 'number' && Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : null };
}

/** One line for a kind's episodes (id -> {ended, elapsed}): live while one is open, then how many
 *  ended and the longest measured one. */
export function waitFoldLine(kind, episodes) {
    const closed = [...episodes.values()].filter((episode) => episode.ended);
    const [waiting, ended, plural] = WAIT_LABELS[kind];
    const known = closed.map((episode) => episode.elapsed).filter((elapsed) => elapsed !== null);
    const longest = known.length ? waitSeconds(Math.max(...known)) : '';
    const done = closed.length === 1 ? [ended, longest].filter(Boolean).join(' · ')
        : closed.length ? `${plural} ×${closed.length}${longest ? ` · longest ${longest}` : ''}` : '';
    return [closed.length < episodes.size ? waiting : '', done].filter(Boolean).join(' · ');
}

/** All of a block's episodes of one kind are ONE receipt row (it never makes a block on its own):
 *  live while an episode is open, then the count and the longest closed one, as measured. Live
 *  frames and history replay land in the same row; an episode seen twice counts once. ``earliest``
 *  (replay) moves the row to an older episode a later-loaded page brings, where live placed it. */
export function foldWaitRow(record, summary, { ts = '', sourceTs = '', earliest = false } = {}) {
    const { kind, id, ended, elapsed } = summary.waitEpisode;
    const folds = record.waitFolds || (record.waitFolds = {});
    const episodes = folds[kind] || (folds[kind] = new Map());
    const seen = episodes.get(id);
    episodes.set(id, { ended: Boolean(seen?.ended) || ended, elapsed: ended ? elapsed : seen?.elapsed ?? null });
    const headline = waitFoldLine(kind, episodes);
    const phase = [...episodes.values()].every((episode) => episode.ended) ? 'result' : 'thinking';
    syncWaitFace(record);
    const index = record.items.findIndex((item) => item.dedupeKey === summary.dedupeKey);
    if (index !== -1) {
        const item = record.items[index];
        const moved = earliest && Date.parse(sourceTs) < Date.parse(item.sourceTs);
        if (item.headline === headline && item.phase === phase && !moved) return { changed: false, index, moved };
        Object.assign(item, { headline, fullHeadline: headline, phase }, moved ? { ts, sourceTs } : {});
        return { changed: true, index, moved };
    }
    record.items.push({ phase, headline, fullHeadline: headline, body: '', fullBody: '', fullRef: '', truncated: false,
        evidenceRef: null, receipt: true, ts, sourceTs, count: 1, dedupeKey: summary.dedupeKey,
        lineKey: `wait-${String(summary.dedupeKey).replace(/[^A-Za-z0-9_-]/g, '-')}` });
    return { changed: true, index: -1, moved: false };
}

// Only a Safety check names its wait on a closed card's face: the tool the card shows is waiting on it (owner,
// Track09 A, 9 Nov 2026). Accounting and storage waits stay in Details.
const FACE_WAITS = ['safety_check_wait'];

/** The quiet line under a live card's current action while one of its calls' Safety checks is open: the still mark of
 *  a wait (DESIGN "Subscription waits": no animation or counter), gone once every episode ended or the card finished.
 *  Concurrent checks are one line; the folded Details row keeps each check and the longest. Returns whether it changed. */
export function syncWaitFace(record) {
    const anchor = record?.activityEl;
    if (!anchor?.ownerDocument) return false;
    const open = !record.finished && FACE_WAITS.some((kind) =>
        [...(record.waitFolds?.[kind]?.values?.() || [])].some((episode) => !episode.ended));
    let el = record.waitFaceEl;
    if (!el && !open) return false;
    if (!el) {
        el = record.waitFaceEl = anchor.ownerDocument.createElement('div');
        el.className = 'chat-live-hostwait';
        el.dataset.liveHostwait = '';
        el.setAttribute('role', 'status');
        anchor.after(el);
    }
    const html = open ? `${statusMarkHtml('', { shape: 'still', hue: 'ink' })}<span>${WAIT_LABELS.safety_check_wait[0]}</span>` : '';
    if (el.innerHTML === html && el.hidden === !open) return false;
    el.innerHTML = html;
    el.hidden = !open;
    return true;
}

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
    if (Object.hasOwn(WAIT_LABELS, kind)) return WAIT_LABELS[kind][evt.phase === 'ended' ? 1 : 0];
    if (!kind && Number(evt.checkpoint_number) > 0) {
        return `Checkpoint ${evt.checkpoint_number} — periodic self-check`;
    }
    if (kind === 'context_reclaim_emergency' && evt.cause === 'output_exhaustion') {
        return 'Source-address recovery after empty length-limited replies (no model)';
    }
    if (kind === 'context_fit_physical_mode') {  // the start's recorded reason, in the Context capsule's words
        const why = causeOf({ preferred_at_send: evt.preferred_mode, applied: evt.effective_mode, cause: evt.cause },
            evt.preferred_mode);
        if (why && evt.cause === 'after_start') return why;  // a route change after the start: not how it started
        const label = fmt(CHECKPOINT_LABELS[kind]);  // in the language of the capsule's sentence beside it
        return why ? `${label}. ${why}` : label;
    }
    const folded = Number(evt.reclaimed_tokens);
    if (kind === 'prompt_prefix_break' && evt.reasoning_retired) {
        const turns = evt.reasoning_retired.turns;
        return `Earlier reasoning not replayed (${turns} turn${turns === 1 ? '' : 's'})`;
    }
    if (kind === 'context_view' && evt.status === 'applied' && folded > 0) {
        return `Ouroboros folded its context (−${folded.toLocaleString('en-US')} tokens)`;
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
