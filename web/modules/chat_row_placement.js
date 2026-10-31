// Where a task-keyed chat row belongs, read from its declared facts the same way live
// and on replay (DESIGN "Unread"). A row's task_id names a task; by itself
// it says neither who speaks, nor whether the row stands alone, nor whether it ends that task.
import { isNonTerminalMediaHistoryRow, positiveTaskTerminalFact } from './chat_activity.js';

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
