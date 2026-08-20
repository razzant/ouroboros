// The chat-activity vocabulary shared by chat.js and dependency-free node tests:
// the in-flight direct/ephemeral/managed turn status reducer, the local-echo
// journal reconciliation, the reconnect banner text, and the /api/state snapshot
// hydration. The live-card presentation projections it also published live with
// their domain owners; the names stay reachable here for their historical
// importers.
export {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    clearStickyCardState,
    isTerminalTaskPhase,
    liveLineRowToggleKey,
    projectCollapsedActivity,
} from './chat_card_state.js';
export {
    headerBudgetPresentation,
    mergeStickyCostMeta,
    taskCostMeta,
    taskCostProjection,
} from './costs.js';
export { rawTimestampEpoch } from './utils.js';

// ---------------------------------------------------------------------------
// In-flight chat activity status (owner decisions 1A-5A; managed continuity).
// ---------------------------------------------------------------------------

// Snapshot-authoritative activity kinds: only kinds the server's activity
// snapshot actually enumerates may be deleted by /api/state hydration.
// "managed_task" rows come from the supervisor queue (PENDING/RUNNING roots),
// so their absence from a snapshot is authoritative conclusion. Typing frames
// without a kind stamp (legacy frames, subagents) stay exempt — they are
// concluded by their own final/summary frames, as before.
const SNAPSHOT_AUTHORITATIVE_KINDS = new Set(['direct_chat', 'ephemeral_decision', 'managed_task']);

/**
 * Single status reducer for the chat header (owner decisions 2A/5A; managed
 * activities added by the project-continuity contract). Priority: disconnected
 * > background live card (Working...) > admitted managed work (Working...) >
 * server-confirmed direct/ephemeral turns (Thinking...) > local pending
 * submissions (Sending...) > queue-admitted but unstarted managed work
 * (Queued...) > terminal attention > idle. A queued task ranks below
 * Sending... because an unacknowledged local submission is the more actionable
 * state. Pure over its inputs for dependency-free node tests.
 */
export function computeDerivedChatStatus({
    isConnected = true,
    hasActiveLiveCard = false,
    activeDirectCount = 0,
    activeManagedCount = 0,
    queuedManagedCount = 0,
    pendingSubmissionsCount = 0,
    lastTerminalAttention = false,
} = {}) {
    if (!isConnected) {
        return { kind: 'offline', text: 'Reconnecting...', showDots: false };
    }
    if (hasActiveLiveCard) {
        return { kind: 'thinking', text: 'Working...', showDots: false };
    }
    if (activeManagedCount > 0) {
        return { kind: 'thinking', text: 'Working...', showDots: true };
    }
    if (activeDirectCount > 0) {
        return { kind: 'thinking', text: 'Thinking...', showDots: true };
    }
    if (pendingSubmissionsCount > 0) {
        return { kind: 'thinking', text: 'Sending...', showDots: true };
    }
    if (queuedManagedCount > 0) {
        return { kind: 'thinking', text: 'Queued...', showDots: true };
    }
    if (lastTerminalAttention) {
        return { kind: 'error', text: 'Attention', showDots: false };
    }
    return { kind: 'online', text: 'Online', showDots: false };
}

/**
 * Local-echo continuity: split the bounded journal of locally-sent owner rows
 * against ONE fetched history response. Entries whose client_message_id
 * appears in the response are CONFIRMED durable (server history is the
 * authority; the local copy retires). The rest are UNCONFIRMED and must
 * survive a full feed rebuild: a stale history snapshot — fetched before the
 * send was logged — has no authority to erase a message the owner just sent.
 * Pure over its inputs for dependency-free node tests.
 */


/**
 * Local-echo continuity: split the bounded journal of locally-sent owner rows
 * against ONE fetched history response. Entries whose client_message_id
 * appears in the response are CONFIRMED durable (server history is the
 * authority; the local copy retires). The rest are UNCONFIRMED and must
 * survive a full feed rebuild: a stale history snapshot — fetched before the
 * send was logged — has no authority to erase a message the owner just sent.
 * Pure over its inputs for dependency-free node tests.
 */
export function partitionLocalEchoJournal(journal, serverClientMessageIds) {
    const confirmed = [];
    const unconfirmed = [];
    const entries = journal instanceof Map ? journal.values() : (journal || []);
    for (const entry of entries) {
        const cmid = String(entry?.clientMessageId || '');
        if (!cmid) continue;
        if (serverClientMessageIds && serverClientMessageIds.has(cmid)) confirmed.push(entry);
        else unconfirmed.push(entry);
    }
    return { confirmed, unconfirmed };
}

// ---------------------------------------------------------------------------
// Pure message-presentation helpers (moved verbatim from chat.js — that
// module sits at its byte ceiling).
// ---------------------------------------------------------------------------

/** Dedupe key for one rendered chat row; client_message_id wins when present. */


export function reconnectBannerText(reason = '') {
    if (reason === 'sha-change') return '♻️ Restart complete';
    if (reason) return '♻️ Reconnected';
    return '';
}

/** {short, full} presentation of a message timestamp, or null when unreadable. */


/**
 * Reconcile the client's active-activity map against one /api/state snapshot
 * (owner decision 1A). The snapshot is authoritative ONLY over kinds it
 * enumerates (direct/ephemeral registry turns and queue-listed managed roots)
 * that existed before it was requested; it must never delete (a) an activity
 * registered by a WS typing frame AFTER the request started (the barrier), or
 * (b) a kind-less typing entry (legacy frames, subagents), which no snapshot
 * source tracks.
 *
 * `concludedIds` (Set/Map with .has) is the client-side conclusion ledger: a
 * turn already concluded by its keyed final must never be re-inserted by a
 * snapshot that was captured while it still ran (activity ids are unique task
 * ids and never restart, so conclusion is final). Without this, a one-shot
 * hydration (project panels) could resurrect a finished turn indefinitely.
 */
export function computeHydratedDirectActivities(existingMap, turnsList, chatId, snapshotBarrierMs = Infinity, concludedIds = null) {
    const nextMap = new Map(existingMap || []);
    if (!Array.isArray(turnsList)) return nextMap;
    const currentChatTurns = turnsList.filter((t) => Number(t?.chat_id ?? 1) === chatId);
    const activeIdsInSnapshot = new Set();
    for (const turn of currentChatTurns) {
        const aid = String(turn?.activity_id || '').trim();
        if (!aid) continue;
        if (concludedIds && concludedIds.has(aid)) continue;
        activeIdsInSnapshot.add(aid);
        const existing = nextMap.get(aid) || {};
        nextMap.set(aid, {
            activityId: aid,
            kind: turn.kind || 'direct_chat',
            phase: turn.phase || 'thinking',
            clientMessageId: turn.client_message_id || existing.clientMessageId || '',
            // Strictly CLIENT-clock "first observed" time: the snapshot's
            // server-clock started_at must never enter the barrier comparison
            // below (clock skew would let finished activities linger).
            startedAt: existing.startedAt || Date.now(),
        });
    }
    for (const [aid, entry] of nextMap.entries()) {
        if (activeIdsInSnapshot.has(aid)) continue;
        // Deletion authority is scoped to snapshot-enumerated kinds: a
        // kind-less typing entry is invisible to every snapshot source and is
        // concluded by its own final/summary frame instead.
        if (!SNAPSHOT_AUTHORITATIVE_KINDS.has(String(entry?.kind || ''))) continue;
        const startedAt = Number(entry?.startedAt) || 0;
        if (startedAt >= snapshotBarrierMs) continue;
        nextMap.delete(aid);
    }
    return nextMap;
}
