// Where a task-keyed chat row belongs, read from its declared facts the same way live
// and on replay (DESIGN "Unread"). A row's task_id names a task; by itself
// it says neither who speaks, nor whether the row stands alone, nor whether it ends that task.
import { isNonTerminalMediaHistoryRow, positiveTaskTerminalFact } from './chat_activity.js';
import { insertTimelineNode } from './chat_render_batch.js';

const replyOrigins = new WeakMap();
const WORK_REPLY_SLOT = ':scope > .chat-live-details > [data-live-work-replies]';

export function releaseWorkReplyOrigin(node) {
    replyOrigins.get(node)?.remove();
    replyOrigins.delete(node);
}

// Card lifetime is independent of saved owner rows. Restore the actual node (not a copy)
// before the card loses its visible slot, at the reply's original position in the feed.
export function restoreWorkReplies(root) {
    if (!root) return;
    for (const slot of root.querySelectorAll('[data-live-work-replies]')) {
        for (const bubble of Array.from(slot.children)) {
            const origin = replyOrigins.get(bubble);
            if (origin?.isConnected) origin.replaceWith(bubble);
            else root.after(bubble);
            replyOrigins.delete(bubble);
        }
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
    if (!replyOrigins.has(bubble)) {
        const origin = document.createComment('work reply');
        bubble.before(origin);
        replyOrigins.set(bubble, origin);
    }
    slot.hidden = false;
    insertTimelineNode(slot, bubble);
    return true;
}
