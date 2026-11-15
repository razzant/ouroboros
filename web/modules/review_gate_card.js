// The work's review-gate field (ouroboros/review_overrides.py): how many times a review of this work returned FAIL or
// no verdict, how often its author explicitly went on over one, how many commits landed over one and which skills it
// left on although no review passed their files, each part only when recorded (an absent part is unknown, never
// zero). One compact phrase on the card's meta line and one Details item, both replaced in place by the task detail
// or the live `review_advisory_override` event (which carries the outcome count; the other parts show at the next
// read): never a row per decision, and never a claim that a change was merged or runs or that a skill is on now.

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

// Twins of `review_overrides._SKILL_OVER` and `_SKILL_EITHER`: what a skill was left on over, in the line's words.
const SKILL_OVER = { review_problems: 'over review problems', no_review_verdict: 'with no review verdict' };
const SKILL_EITHER = 'over review problems or no verdict';
// How it came to be left on, for Details: the mind's own switch, the end of its review, or its author's acceptance.
const SKILL_WHY = { review_problems: 'its review found problems', no_review_verdict: 'no review passed its current files' };
const SKILL_DOOR = {
    turned_on: (name, why) => `Ouroboros turned on skill ${name} although ${why}`,
    after_review: (name, why) => `Skill ${name} was left on although ${why}`,
    author_finish: (name, why) => `Skill ${name} stayed on because Ouroboros accepted it as its author, although ${why}`,
};
const skillName = (skills) => `"${skills?.latest?.skill || 'unknown'}"`;

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
    const skills = counted(fact?.skills);
    if (skills) {
        parts.push(skills === 1 ? `skill ${skillName(fact.skills)} left on ${SKILL_OVER[fact.skills.latest?.reason] || SKILL_EITHER}`
            : `${skills} skills left on ${SKILL_EITHER} (latest: ${skillName(fact.skills)})`);
    }
    return parts.length ? `Review: ${parts.join(' · ')}` : '';
}

/** @returns {boolean} whether the card's field or Details item changed; an older read never replaces a newer one. */
export function syncReviewGateItem(record, detail) {
    const fact = detail?.review_overrides;
    const line = reviewGateLine(fact);
    // Each part's count only grows and each latest carries its time (a skill left on again keeps its count and moves
    // its latest): the sum of counts, then the newest of those times, orders two reads of one work's fact.
    const total = counted(fact) + counted(fact?.continued) + counted(fact?.landed) + counted(fact?.skills);
    const at = [fact?.latest?.at, fact?.continued?.latest?.at, fact?.landed?.latest?.at, fact?.skills?.latest?.at]
        .map((value) => String(value || '')).sort().pop();
    const seen = record?.reviewGateTotal || 0;
    if (!record || !line || total < seen || (total === seen && at <= (record.reviewGateAt || ''))) return false;
    record.reviewGateTotal = total;
    record.reviewGateAt = at;
    record.reviewGateLine = line;
    const latest = fact.latest || {}, continued = fact.continued || {}, landed = fact.landed || {}, skills = fact.skills || {};
    const reviewed = counted(fact) || counted(continued) || counted(landed);
    const door = SKILL_DOOR[skills.latest?.door] || SKILL_DOOR.after_review;
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
        counted(skills) ? (skills.count === 1 ? '' : `${skills.count} skills were left on ${SKILL_EITHER}. Latest: `)
            + door(skillName(skills), SKILL_WHY[skills.latest?.reason] || 'no review passed its current files')
            + (skills.latest?.task_id && skills.latest.task_id !== record.groupId ? ` (task ${skills.latest.task_id})` : '') + '.' : '',
    ].filter(Boolean).join('\n');
    const headline = reviewed ? 'Review returned FAIL or no verdict' : 'Skill left on without a passed review';
    const key = `review-gate|${record.groupId}`;
    const item = {
        phase: 'warn', headline, fullHeadline: headline, body, fullBody: body, bodyFormat: 'text', receipt: true, count: 1,
        ts: '', sourceTs: at, dedupeKey: key,
        lineKey: `review-gate-${String(record.groupId).replace(/[^A-Za-z0-9_-]/g, '-')}`,
    };
    const current = record.items.find((entry) => entry.dedupeKey === key);
    if (current) Object.assign(current, item);
    else record.items.push(item);
    return true;
}
