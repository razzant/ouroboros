// The work's review-gate field (ouroboros/review_overrides.py): how many times the review gate let this work go on
// where Blocking enforcement would have stopped it, and the latest reason. One compact phrase on the card's meta line
// and one Details item, both replaced in place by the task detail or the live `review_advisory_override` event: never
// a row per decision, and never a claim that a change was committed, merged or runs.

/** Twin of `review_overrides.card_line`: the count and the latest reason, or '' without a decision. */
export function reviewGateLine(fact) {
    const count = fact?.count;
    if (!Number.isSafeInteger(count) || count < 1) return '';
    return `Review: ${count}× continued where Blocking would have stopped · latest: ${reasonWords(fact)}`;
}

const reasonWords = (fact) => String(fact?.latest?.reason || 'unknown').replaceAll('_', ' ');

/** @returns {boolean} whether the card's field or Details item changed; an older count never replaces a newer one. */
export function syncReviewGateItem(record, detail) {
    const fact = detail?.review_overrides;
    const line = reviewGateLine(fact);
    if (!record || !line || fact.count <= (record.reviewGateCount || 0)) return false;
    record.reviewGateCount = fact.count;
    record.reviewGateLine = line;
    const latest = fact.latest || {};
    const by = latest.task_id && latest.task_id !== record.groupId ? ` (task ${latest.task_id})` : '';
    const body = `${fact.count === 1 ? 'Once' : `${fact.count} times`} the review gate let this work continue where `
        + 'Blocking enforcement would have stopped it: it did not prohibit continuation, which says nothing about '
        + `what was committed or runs.\nLatest${by}: ${reasonWords(fact)}${latest.text ? ` — ${latest.text}` : ''}`;
    const key = `review-gate|${record.groupId}`;
    const item = {
        phase: 'warn', headline: 'Review gate did not prohibit continuation',
        fullHeadline: 'Review gate did not prohibit continuation', body, fullBody: body, receipt: true, count: 1,
        ts: '', sourceTs: String(latest.at || ''), dedupeKey: key,
        lineKey: `review-gate-${String(record.groupId).replace(/[^A-Za-z0-9_-]/g, '-')}`,
    };
    const current = record.items.find((entry) => entry.dedupeKey === key);
    if (current) Object.assign(current, item);
    else record.items.push(item);
    return true;
}
