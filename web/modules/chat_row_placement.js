// Where a task-keyed chat row belongs, read from its declared facts the same way live
// and on replay (DESIGN "Unread"). A row's task_id names a task; by itself
// it says neither who speaks, nor whether the row stands alone, nor whether it ends that task.
import { cardRowEvidenceRef, isNonTerminalMediaHistoryRow, positiveTaskTerminalFact } from './chat_activity.js';
import { normalizeLogTs } from './log_events.js';
import { hostRowDisplayText, hostSentenceText } from './host_notice_text.js';
import { mergeHistoricalTimelineItem } from './chat_history_replay.js';
import { insertTimelineNode, updateLiveTimelineItem } from './chat_render_batch.js';

const WORK_REPLY_SLOT = ':scope > .chat-live-details > [data-live-work-replies]';
const taskKey = value => String(value || '').trim();
const CARD_ROW_PHASES = new Map([['timeline', 'warn'], ['reviews', 'result']]);

export function placedCardRow(msg) {
    const placement = taskKey(msg?.card_row);
    const phase = CARD_ROW_PHASES.get(placement);
    const taskId = taskKey(msg?.task_id);
    return phase && taskId ? { placement, phase, taskId } : null;
}

export function cardRowSummary(msg, phase, rawTs = '') {
    const text = String(msg.text ?? msg.content ?? '');
    const lines = hostRowDisplayText(msg, msg.role === 'system' && msg.system_type === 'task_pause_notice' ? hostSentenceText(text) : text).split('\n');
    const rowId = String(msg.card_row_id || '').trim() || `${String(msg.system_type || '').trim()}|${rawTs}`;
    return {
        phase, headline: lines[0].trim(), body: lines.slice(1).join('\n').trim(), dedupeKey: `cardrow|${rowId}`,
        cardRowRevision: msg.card_row_revision, evidenceRef: cardRowEvidenceRef(msg),
        replyAddress: msg.role === 'system' ? msg.row_address || '' : '',
        cardReview: msg?.late_evidence?.kind === 'merge_review' ? msg.late_evidence : null,
    };
}

export function mergePlacedCardRow(record, msg, { phase }, rawTs) {
    const summary = cardRowSummary(msg, phase, rawTs);
    const ts = normalizeLogTs(rawTs);
    // Replay keeps source/order/page identity.
    if (msg.history_id) return mergeHistoricalTimelineItem(record, summary, msg, ts);
    const { timelineUpdate } = updateLiveTimelineItem(record, summary, {
        ts, rawTs, syntheticKey: summary.dedupeKey, headline: summary.headline, inPlaceByKey: true,
    });
    return !['none', 'duplicate-skip'].includes(timelineUpdate);
}

// The card may move, but its reply is the feed's same chronological node, never a copy.
export function restoreWorkReplies(root, feed = root.parentElement) {
    if (!feed) return;
    const typing = feed.lastElementChild?.classList?.contains('typing-bubble') ? feed.lastElementChild : null;
    for (const slot of root.querySelectorAll('[data-live-work-replies]')) {
        for (const bubble of Array.from(slot.children)) insertTimelineNode(feed, bubble, typing);
        slot.hidden = true;
    }
}

// project_dialogue.ORIGIN_ADDRESSED_NOTICE_TYPES: a routing refusal names the task it
// refused, never its speaker; it stays the issuing chat's own row.
export const ORIGIN_ADDRESSED_NOTICE_TYPES = new Set(['task_not_started', 'task_start_unconfirmed', 'steer_not_delivered', 'relay_reply']);

/**
 * Where a known child's non-progress, unplaced row goes: a routing notice stays a row
 * of the conversation (''), the child's own words (send_user_message) are a line of
 * its card ('line'), anything else is its card's answer ('result').
 */
export function childRowPlacement(row) {
    const type = String(row?.system_type || '');
    return ORIGIN_ADDRESSED_NOTICE_TYPES.has(type) ? '' : type === 'proactive_message' ? 'line' : 'result';
}

/**
 * Replay conclusion: a typed terminal fact OR a plain untyped final (replay has no
 * later task_done frame, so the bare final is the task's last word; marked rows —
 * system_type/msg_type — still never conclude, nor does a delivered photo or video).
 */
export function replayConcludesTask(row) {
    return (row.role === 'assistant' || row.role === 'system')
        && (positiveTaskTerminalFact(row) || (!row.system_type && !row.msg_type && row.task_phase !== 'unfinished'))
        && !isNonTerminalMediaHistoryRow(row);
}

export const replyAwareRows = feed => [...feed.children, ...feed.querySelectorAll('[data-reply-work]')];

// A recent reply may be mounted before the old page holding its card.
export function reconcileWorkReplies(feed, cards) {
    let changed = false;
    for (const bubble of feed.querySelectorAll('[data-reply-work]')) {
        changed = moveWorkReply(cards.get(bubble.dataset.replyWork), bubble) || changed;
    }
    return changed;
}

export function placeWorkReply(feed, cards, row) {
    const work = row.reply_to?.work_task_id;
    if (!work || !row.client_message_id) return false;
    const bubble = Array.from(feed.querySelectorAll('.chat-bubble.user[data-client-message-id]'))
        .find(node => node.dataset.clientMessageId === row.client_message_id);
    if (!bubble) return false;
    bubble.dataset.replyWork = work;
    return moveWorkReply(cards.get(work), bubble);
}

// Move the canonical bubble, with its quote and receipt, only if this room has the card.
function moveWorkReply(record, bubble) {
    const root = record?.root;
    const slot = root?.isConnected && !root.closest('[data-portal="1"]') && root.querySelector(WORK_REPLY_SLOT);
    if (!slot || bubble.parentElement === slot) return false;
    slot.hidden = false;
    insertTimelineNode(slot, bubble);
    return true;
}
