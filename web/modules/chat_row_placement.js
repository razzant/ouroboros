// Where a task-keyed chat row belongs, read from its declared facts the same way live
// and on replay (DESIGN "Unread"). A row's task_id names a task; by itself
// it says neither who speaks, nor whether the row stands alone, nor whether it ends that task.
import { cardRowSummary as summarizeCardRow, isNonTerminalMediaHistoryRow, positiveTaskTerminalFact } from './chat_activity.js';
import { normalizeLogTs } from './log_events.js';
import { clearAnchoredCardRows, detachAnchoredCardRow, exactRowAnchor, itemMatchesExactAnchor,
    mergeHistoricalTimelineItem, mountedLineAnchor, sortBySource } from './chat_history_replay.js';
import { updateLiveTimelineItem } from './chat_render_batch.js';

const taskKey = value => String(value || '').trim();
// Host placement: custody warns; settled reviews show results.
const CARD_ROW_PHASES = new Map([['timeline', 'warn'], ['reviews', 'result']]);

export function placedCardRow(msg) {
    const placement = taskKey(msg?.card_row);
    const phase = CARD_ROW_PHASES.get(placement);
    const taskId = taskKey(msg?.task_id);
    return phase && taskId ? { placement, phase, taskId } : null;
}

// A work's own saved line (a root's or a child's) is quoted by its own address, never its card's.
export const lineReplyFields = (msg) => (msg?.row_address && !msg.lifecycle
    && msg.role !== 'system' && msg.system_type !== 'subagent_terminal_notice'
    ? { replyAddress: msg.row_address, replyRole: 'assistant' } : {});

export function cardRowSummary(msg, phase, rawTs = '') {
    const system = msg?.role === 'system' && msg?.narration !== true
        && !msg?.lifecycle_virtual && !msg?.origin_projected;
    return { ...summarizeCardRow(msg, phase, rawTs),
        replyAddress: system ? msg.row_address || '' : '',
        replyRole: system ? 'system' : (msg?.role && msg.role !== 'system' ? String(msg.role) : 'unauthored') };
}

export function mergePlacedCardRow(record, msg, { phase }, rawTs, { anchor = null, anchoredRecords = null } = {}) {
    const summary = cardRowSummary(msg, phase, rawTs);
    const ts = normalizeLogTs(rawTs);
    // Replay keeps source/order/page identity.
    if (msg.history_id) return mergeHistoricalTimelineItem(record, summary, msg, ts, { anchor, anchoredRecords });
    const existing = anchor ? record.items.find((it) => it.dedupeKey === summary.dedupeKey) : null;
    const stale = existing && Number.isSafeInteger(existing.cardRowRevision)
        && (!Number.isSafeInteger(summary.cardRowRevision) || summary.cardRowRevision <= existing.cardRowRevision);
    const detached = Boolean(existing && !stale
        && (!summary.replyAddress || summary.replyAddress !== existing.replyAddress)
        && itemMatchesExactAnchor(existing, anchor)
        && detachAnchoredCardRow(existing, summary.dedupeKey));
    if (detached) anchoredRecords?.add(record);
    const { timelineUpdate } = updateLiveTimelineItem(record, summary, {
        ts, rawTs, syntheticKey: summary.dedupeKey, headline: summary.headline, inPlaceByKey: true,
    });
    if (detached) sortBySource(record);
    return !['none', 'duplicate-skip'].includes(timelineUpdate);
}

// One room's exact work-card line and the temporary revisions it keeps mounted. The room owns
// page changes and viewport writes; placed-row replay owns which older lines those changes retire.
export function createPlacedCardAnchors({ repaint, updateCount, withStableViewport }) {
    let anchor = null;
    const anchoredRecords = new Set();

    function drop(renderNow = false, keepAddress = '') {
        for (const { record, removed } of clearAnchoredCardRows(anchoredRecords, keepAddress)) {
            record._timelineDirty = true;
            if (record.isSubagent && record.root?.dataset?.expanded !== '1' && record.timelineEl) {
                const gone = new Set(removed.map((item) => item.lineKey));
                for (const child of Array.from(record.timelineEl.children)) {
                    if (gone.has(child.dataset?.liveLineKey)) child.remove();
                }
            } else if (renderNow) repaint(record);
            updateCount(record);
        }
    }

    return {
        forget(record) { anchoredRecords.delete(record); },
        merge(record, msg, placed, rawTs) { return mergePlacedCardRow(record, msg, placed, rawTs, { anchor, anchoredRecords }); },
        beginPage(direction, target) {
            if (!['recent', 'latest', 'around', 'revisit'].includes(direction)) return;
            anchor = direction === 'around' ? exactRowAnchor(target) : null;
            drop();
        },
        pin(address, node) {
            anchor = mountedLineAnchor(address, node);
            drop(true, anchor?.row_address);
        },
        toPresent() {
            anchor = null;
            if (anchoredRecords.size) withStableViewport(() => { drop(true); return true; });
        },
    };
}

// project_dialogue.ORIGIN_ADDRESSED_NOTICE_TYPES: a routing refusal names the task it
// refused, never its speaker; it stays the issuing chat's own row.
export const ORIGIN_ADDRESSED_NOTICE_TYPES = new Set(['task_not_started', 'task_start_unconfirmed', 'steer_not_delivered', 'relay_reply']);

export function referencesTaskOnly(row) {
    return ORIGIN_ADDRESSED_NOTICE_TYPES.has(row.system_type)
        || ['project_question_pointer', 'quiz_answer', 'quiz_closed'].includes(row.system_type);
}

/**
 * Where a known child's non-progress, unplaced row goes: a routing notice stays a row
 * of the conversation (''), the child's own words (send_user_message) are a line of
 * its card ('line'), anything else is its card's answer ('result').
 */
export function childRowPlacement(row) {
    const type = String(row?.system_type || '');
    return ORIGIN_ADDRESSED_NOTICE_TYPES.has(type) ? '' : type === 'proactive_message' ? 'line' : 'result';
}

/** A hidden room can show only what the ordinary timeline would show as its own assistant text. Shared appearances,
 * cards, connector/external attribution and attachment notices wait for the full room projection. */
export function isWindowText(row, knownChild) {
    return row?.role === 'assistant' && ['', 'proactive_message'].includes(row.system_type || '')
        && !row.is_progress && !row.card_row && !row.msg_type && !row.narration && !row.relay && !row.origin_projected
        && !row.text_placeholder && !row.attachments?.length && !row.presence_contact
        && !row.parent_task_id && !row.subagent_task_id && row.delegation_role !== 'subagent'
        && (!row.task_id || row.task_id === row.root_task_id && !knownChild(row.task_id));
}

/**
 * Replay conclusion: a typed terminal fact OR a plain untyped final (replay has no
 * later task_done frame, so the bare final is the task's last word; marked rows —
 * system_type/msg_type — still never conclude, nor does a delivered photo or video).
 */
export function replayConcludesTask(row) {
    return (row.role === 'assistant' || row.role === 'system')
        && !referencesTaskOnly(row)
        && (positiveTaskTerminalFact(row) || (!row.system_type && !row.msg_type && row.task_phase !== 'unfinished'))
        && !isNonTerminalMediaHistoryRow(row);
}
