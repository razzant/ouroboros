// The work's review-gate field (ouroboros/review_overrides.py): how many times a review of this work returned FAIL or
// no verdict, how often its author explicitly went on over one and how many commits landed over one, each part only
// when recorded (an absent part is unknown, never zero). One compact phrase on the card's meta line and one Details
// item, both replaced in place by the task detail or the live `review_advisory_override` event (which carries the
// outcome count; the author's continuation and the landing show at the next read): never a row per decision, and
// never a claim that a change was merged or runs.

const QUOTED = 80;  // review_overrides.QUOTED: the head of an author's reason the compact line quotes

/** Twin of `review_overrides._head`: at most `limit` characters, a cut head ending in an ellipsis that says so. */
const headOf = (text, limit) => {
    const points = Array.from(String(text || ''));
    return points.length <= limit ? points.join('') : `${points.slice(0, limit - 1).join('').replace(/[ \t\n\r\f\v]+$/, '')}…`;
};
const counted = (part) => (Number.isSafeInteger(part?.count) && part.count > 0 ? part.count : 0);
const reasonWords = (fact) => String(fact?.latest?.reason || 'unknown').replaceAll('_', ' ');
/** Twin of `review_overrides._decision`: the author's disposition and the head of its reason. */
const decisionWords = (latest, limit = QUOTED) => {
    const why = headOf(latest?.rationale, limit);
    return [String(latest?.disposition || ''), why ? `"${why}"` : ''].filter(Boolean).join(' — ');
};

/** Twin of `review_overrides.card_line`: each recorded part, or '' without one; a part the work's own outcome count
 * does not precede names what it went on over itself. */
export function reviewGateLine(fact) {
    const continued = fact?.continued, landed = counted(fact?.landed), said = decisionWords(continued?.latest);
    const parts = counted(fact) ? [`FAIL or no verdict ${fact.count}× (latest: ${reasonWords(fact)})`] : [];
    const over = parts.length ? '' : String(continued?.latest?.reason || '').replaceAll('_', ' ');
    if (counted(continued)) {
        parts.push(`author continued ${continued.count}×${over ? ` over ${over}` : ''}${said ? ` (latest: ${said})` : ''}`);
    }
    if (landed) parts.push(`landed over ${parts.length ? 'it' : 'a FAIL or no verdict'}: ${landed} commit${landed === 1 ? '' : 's'}`);
    return parts.length ? `Review: ${parts.join(' · ')}` : '';
}

/** @returns {boolean} whether the card's field or Details item changed; an older read never replaces a newer one. */
export function syncReviewGateItem(record, detail) {
    const fact = detail?.review_overrides;
    const line = reviewGateLine(fact);
    // Each part only grows, so the sum of its counts orders two reads of one work's fact.
    const total = counted(fact) + counted(fact?.continued) + counted(fact?.landed);
    if (!record || !line || total <= (record.reviewGateTotal || 0)) return false;
    record.reviewGateTotal = total;
    record.reviewGateLine = line;
    const latest = fact.latest || {}, continued = fact.continued || {}, landed = fact.landed || {};
    const by = latest.task_id && latest.task_id !== record.groupId ? ` (task ${latest.task_id})` : '';
    const times = (count) => (count === 1 ? 'once' : `${count} times`);
    const body = [
        counted(fact) ? `A review of this work returned FAIL or no verdict ${times(fact.count)}.\nLatest${by}: `
            + `${reasonWords(fact)}${latest.text ? ` — ${latest.text}` : ''}` : '',
        counted(continued) ? `The author explicitly continued ${times(continued.count)}`
            + (!counted(fact) && continued.latest?.reason ? ` over ${String(continued.latest.reason).replaceAll('_', ' ')}` : '')
            + `; latest: ${decisionWords(continued.latest, Infinity) || 'not recorded'}` : '',
        counted(landed) ? `Landed over ${counted(fact) || counted(continued) ? 'it' : 'a FAIL or no verdict'}: `
            + `${landed.count} commit${landed.count === 1 ? '' : 's'}; latest: `
            + (String(landed.latest?.sha || '').slice(0, 12) || 'not recorded') : '',
    ].filter(Boolean).join('\n');
    const at = [latest.at, continued.latest?.at, landed.latest?.at].map((value) => String(value || '')).sort().pop();
    const key = `review-gate|${record.groupId}`;
    const item = {
        phase: 'warn', headline: 'Review returned FAIL or no verdict',
        fullHeadline: 'Review returned FAIL or no verdict', body, fullBody: body, receipt: true, count: 1,
        ts: '', sourceTs: at, dedupeKey: key,
        lineKey: `review-gate-${String(record.groupId).replace(/[^A-Za-z0-9_-]/g, '-')}`,
    };
    const current = record.items.find((entry) => entry.dedupeKey === key);
    if (current) Object.assign(current, item);
    else record.items.push(item);
    return true;
}
