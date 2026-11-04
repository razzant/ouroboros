// The settled task card's one seam: its Files row (result_files.js) and its
// Continue action (task_continue.js) follow the same settled task detail.

import { syncResultFilesItem } from './result_files.js';
import { positiveTaskTerminalFact } from './chat_activity.js';
import { isTerminalTaskDetail } from './log_events.js';
import { syncContinueAction } from './task_continue.js';

/** The finished root's closing line, shared by its card, room row and portal. */
export function closingLine(record, title = record?.titleEl?.textContent || record?.suggestedName || '') {
    if (!record?.finished) return '';
    if (record.causeHeld) return record.collapsedActivity || '';
    const own = !record.isSubagent && record.wakeSummary || '';
    return own !== title ? own : '';
}

/** @returns {boolean} whether the card's timeline items changed (the Files row) */
export function syncSettledItems(record, detail, options) {
    syncContinueAction(record, detail, options);  // a late offer read writes through the caller's viewport seam
    return syncResultFilesItem(record, detail);
}

/** A terminal detail or a new attempt replaces the previous sentence, including with absence. */
export function syncClosingSentence(record, detail) {
    const attempt = detail?.task_attempt;
    const reset = isTerminalTaskDetail(detail) || positiveTaskTerminalFact(detail)
        || (Number.isSafeInteger(attempt) && attempt !== record?.summaryAttempt);
    const sentence = detail && 'wake_summary' in detail ? detail.wake_summary : reset ? '' : record?.wakeSummary;
    const words = record && record.wakeSummary !== sentence;
    if (record && Number.isSafeInteger(attempt)) record.summaryAttempt = attempt;
    if (words) {
        if (record.collapsedActivity === record.wakeSummary) record.collapsedActivity = '';
        record.wakeSummary = sentence;
    }
    return Boolean(words);
}
