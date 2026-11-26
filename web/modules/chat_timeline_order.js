// Timeline placement keys are shared by the canonical feed and the saved text preview.
// This leaf must not import UI or network modules: connection bootstrap reads the preview.

/** Convert a raw source timestamp to sortable epoch milliseconds. */
export function rawTimestampEpoch(raw) {
    if (raw == null || raw === '') return NaN;
    const epoch = typeof raw === 'number' ? raw : Date.parse(String(raw));
    return Number.isFinite(epoch) ? epoch : NaN;
}

export function timelineTime(raw) {
    const ts = rawTimestampEpoch(raw);
    const tsTail = typeof raw === 'string'
        ? (/:\d{2}\.\d{3}(\d+)(?:Z|[+-]\d{2}:?\d{2})?$/i.exec(raw.trim())?.[1] || '').replace(/0+$/, '') : '';
    return { ts, tsTail };
}

// Keep data-ts in milliseconds for elapsed time and labels. Normalized decimal tails compare lexically;
// a missing tail (including an older cached timestamp) means no precision past the stored millisecond.
const compareTail = (a = '', b = '') => a === b ? 0 : a > b ? 1 : -1;
function compareTimelineTime(a, b) {
    if (a?.ts == null || a.ts === '' || b?.ts == null || b.ts === '') return NaN;
    const delta = Number(a.ts) - Number(b.ts);
    return delta === 0 ? compareTail(a.tsTail, b.tsTail) : delta;
}

export function compareHistoryPosition(left, right) {
    if (!left || !right) return 0;
    const stream = String(left.source || '').localeCompare(String(right.source || ''));
    return stream || Number(left.offset) - Number(right.offset);
}

const position = row => row.historySource ? { source: row.historySource, offset: row.historyOffset } : null;

/** Untimed rows do not compare; exact ties with no comparable position retain arrival order. */
export function compareTimelineOrder(a, b) {
    const time = compareTimelineTime(a, b);
    return time === 0 ? compareHistoryPosition(position(a), position(b)) : time;
}

// Anchor mode only moves a node's effective timestamp earlier so replay cannot teleport it downward.
export function stampNodeTimestamp(node, raw, { anchor = false } = {}) {
    if (!node) return false;
    const { ts, tsTail } = timelineTime(raw);
    if (!Number.isFinite(ts)) return false;
    const order = anchor ? compareTimelineTime({ ts, tsTail }, node.dataset) : NaN;
    if (order >= 0) return false;
    if (node.dataset.ts !== String(ts)) node.dataset.ts = String(ts);
    if (tsTail) node.dataset.tsTail = tsTail;
    else delete node.dataset.tsTail;
    return order < 0;
}
