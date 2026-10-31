/** Historical rows enrich the existing timeline without acquiring live authority. */
import { foldWaitRow } from './task_checkpoints.js';

export function compareHistoryPosition(left, right) {
    if (!left || !right) return 0;
    const stream = String(left.source || '').localeCompare(String(right.source || ''));
    return stream || Number(left.offset) - Number(right.offset);
}

export function historyRowIds(row) {
    return [row.history_id, ...(row.review_group?.attempts || []).map(attempt => attempt.history_id)].filter(Boolean);
}

export function stampHistoryNode(node, id, position) {
    if (!node || !id) return;
    node.dataset.historyId = id;
    if (position) {
        node.dataset.historySource = position.source;
        node.dataset.historyOffset = String(position.offset);
    }
}

export function mergeHistoricalTimelineItem(record, summary, row, ts) {
    if (summary.visible === false || !(summary.headline || summary.body)) return false;
    const identity = String(row?.history_id || '');
    if (!identity) return false;
    if (summary.waitEpisode) {  // the same one row per wait kind as live
        const { changed, index, moved } = foldWaitRow(record, summary, { ts, sourceTs: row.ts || '', earliest: true });
        if (index === -1 || moved) sortBySource(record);
        return changed;
    }
    // A child's lifecycle is one evolving status, just as it is live. Its
    // authored progress keeps every source record, even when text and time match;
    // so does each delegated observation (`activity`), projected per seq at render.
    // The task's outcome (`task_done|`) is one line too, as live: a read inside the
    // post-task window projects it still finalizing, which is that line Working,
    // never a line of work, and a finished card takes nothing from it (live: closedPhase).
    const outcome = String(summary.dedupeKey || '').startsWith('task_done|');
    if (outcome && !summary.terminal && record.finished) return false;
    const evolving = summary.terminal || outcome || String(summary.dedupeKey || '').startsWith('subagent-lifecycle:')
        || String(summary.dedupeKey || '').startsWith('cardrow|');
    const activity = summary.activity ? { activity: summary.activity } : {};
    const key = evolving ? summary.dedupeKey : `history:${identity}`;
    // Speech keeps its live occurrence alias beside the physical history identity:
    // history may precede live delivery, or adopt a line before it is redelivered.
    const occurrence = String(summary.dedupeKey || '').startsWith('subagent-line:')
        ? { occurrenceKey: summary.dedupeKey } : {};
    const source = { sourceHistoryId: identity, sourceHistoryRevision: summary.cardRowRevision,
        historyPosition: row.history_position };
    let item = record.items.find((entry) => entry.dedupeKey === key);
    if (!item && !evolving) {
        item = record.items.find((entry) => !entry.historyId
            && entry.dedupeKey === summary.dedupeKey && entry.sourceTs === row.ts);
    }
    if (item && evolving) {
        // Source revision advances independently of live content: an older
        // fallback can yield to its canonical current row without rolling back
        // content, while stale sources cannot redirect an already newer locator.
        const existingPosition = item.historyPosition;
        const adoptedSource = !item.sourceHistoryId || Number.isSafeInteger(summary.cardRowRevision)
            && (!Number.isSafeInteger(item.sourceHistoryRevision) || summary.cardRowRevision > item.sourceHistoryRevision);
        if (adoptedSource) Object.assign(item, source);
        const incomingTime = Date.parse(row.ts), existingTime = Date.parse(item.sourceTs || '');
        if (Number.isSafeInteger(item.cardRowRevision)) {
            if (!Number.isSafeInteger(summary.cardRowRevision) || summary.cardRowRevision <= item.cardRowRevision) return adoptedSource;
        } else if (!Number.isSafeInteger(summary.cardRowRevision) && (incomingTime < existingTime || incomingTime === existingTime
                && compareHistoryPosition(row.history_position, existingPosition) < 0)) return adoptedSource;
        const update = { headline: summary.headline || item.headline,
            cardRowRevision: summary.cardRowRevision,
            fullHeadline: summary.fullHeadline || summary.headline || item.fullHeadline,
            body: summary.body || '', fullBody: summary.fullBody || summary.body || '',
            phase: summary.phase || item.phase, sourceTs: row.ts || item.sourceTs,
            ts, ...source };
        if (Object.entries(update).every(([key, value]) => key === 'historyPosition'
            ? JSON.stringify(item[key]) === JSON.stringify(value) : item[key] === value)) return adoptedSource;
        Object.assign(item, update);
        return true;
    }
    if (item?.historyId === identity) return false;
    if (item) {
        item.historyId = identity;
        item.historyPosition = row.history_position;
        item.dedupeKey = key;
        Object.assign(item, occurrence);
        item.count = 1;
        if (summary.evidenceRef && !item.evidenceRef) item.evidenceRef = summary.evidenceRef;
    } else {
        record.items.push({
            cardRowRevision: summary.cardRowRevision,
            phase: summary.phase || 'working', headline: summary.headline || 'Update',
            fullHeadline: summary.fullHeadline || summary.headline || 'Update',
            body: summary.body || '', fullBody: summary.fullBody || summary.body || '',
            fullRef: summary.fullRef || '', truncated: summary.truncated || false,
            evidenceRef: summary.evidenceRef || null,
            ts: ts || '', sourceTs: row.ts || '', count: 1, dedupeKey: key, ...activity, ...occurrence,
            ...(evolving ? source : { historyId: identity, historyPosition: row.history_position }),
            lineKey: evolving && !String(key).startsWith('cardrow|') ? `terminal-${String(key).replace(/[^A-Za-z0-9_-]/g, '-')}`
                : `history-${identity.replace(/[^A-Za-z0-9_-]/g, '-')}`,
        });
    }
    sortBySource(record);
    return true;
}

function sortBySource(record) {
    record.items.sort((a, b) => {
        const first = Date.parse(a.sourceTs || ''), second = Date.parse(b.sourceTs || '');
        return (Number.isFinite(first) && Number.isFinite(second) ? first - second : 0)
            || compareHistoryPosition(a.historyPosition, b.historyPosition);
    });
}

/** Each mounted node a released id removes, as [id, node]: the rows it renders
 * and the evolving lines that only locate their page by it (never a row stamp). */
export const historyStamps = root => [['[data-history-id]', 'historyId'], ['[data-source-history-id]', 'sourceHistoryId']]
    .flatMap(([selector, key]) => Array.from(root.querySelectorAll(selector), node => [node.dataset[key], node]));

/** A page can leave the rendered window only outside reading, focus and selection. */
export function historyNodeIsProtected(node, viewport, selection = globalThis.getSelection?.()) {
    if (!node?.isConnected) return false;
    const active = node.ownerDocument?.activeElement;
    if (active && node.contains(active)) return true;
    if (selection && !selection.isCollapsed) {
        for (let index = 0; index < selection.rangeCount; index += 1) {
            try { if (selection.getRangeAt(index).intersectsNode(node)) return true; } catch {}
        }
    }
    return historyNodeOnScreen(node, viewport);
}

/** A rendered node intersecting the feed's viewport (or a band of it); a collapsed one has no boxes. A row asleep
 * (chat_row_sleep.js) shows nothing outside its own box, which stays laid out while what it holds is not: off the
 * screen, it answers for everything in it without laying that out. */
export function historyNodeOnScreen(node, viewport, bounds = viewport.getBoundingClientRect()) {
    if (!node?.isConnected) return false;
    const row = node.closest?.('[data-laid]');
    if (row && row !== node && !historyNodeOnScreen(row, viewport, bounds)) return false;
    const rect = node.getBoundingClientRect();
    return Boolean(node.getClientRects().length && rect.bottom > bounds.top && rect.top < bounds.bottom);
}
