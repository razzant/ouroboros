// Where a root's money and work went, in its card's Details: the task detail's read-time `cost_breakdown`
// (GET /api/tasks/{id}, roots only) gives two partitions of one subtree, by purpose ("Spent on") and by the
// route that served the rows ("Ran on"). Each line words its money exactly as the card does
// (`costPresentationMeta` over the slice's own carrier) and adds the volume the ledger recorded: model calls,
// delegated sessions, tokens. Nothing is summed here; a part the ledger did not record is left out, never a zero.

import { API_PROVIDER_LABELS, harnessPresentation } from './harness_presentation.js';
import { costPresentationMeta } from './chat_activity.js';
import { SURFACE_LABELS } from './review_opinions.js';
import { escapeHtmlText, formatCompactNumber } from './utils.js';

// A task's own work and its helpers, the commit gate's review waves (`review_price.REVIEW_USAGE_CATEGORY`) and
// the older advisory category; another review surface's category (`<surface>_review`) takes the word its
// Reviews door uses; the rest reads as its own key.
const PURPOSE_WORDS = Object.freeze({
    task: 'Work', subagent: 'Helpers', multi_model_review_review: 'Code review', advisory_review: 'Early look',
});

const plural = (n, word) => `${n.toLocaleString('en-US')} ${word}${n === 1 ? '' : 's'}`;
const mechanical = (key) => {
    const words = String(key || '').replace(/[_-]+/g, ' ').replace(/\breview review$/, 'review').trim();
    return words ? words[0].toUpperCase() + words.slice(1) : 'Other';
};

/** A usage category in the owner's words. */
export const purposeWord = (key) => PURPOSE_WORDS[key] || SURFACE_LABELS[String(key || '').replace(/_review$/, '')]
    || mechanical(key);

/** A route (the ledger's provider) by its product name. */
export function routeWord(key) {
    if (API_PROVIDER_LABELS[key]) return API_PROVIDER_LABELS[key];
    const harness = harnessPresentation(key);
    return harness.known ? harness.label : mechanical(key);
}

function tokensText(tokens) {
    if (!Number.isFinite(tokens) || tokens <= 0) return '';
    return tokens >= 1e9 ? `${(tokens / 1e9).toFixed(1)}B tokens` : `${formatCompactNumber(tokens)} tokens`;
}

/** One slice as `{word} · {money} · {volume}`; parts the ledger did not record are absent. */
export function costSliceLine(slice, word) {
    const calls = Number(slice?.model_calls) || 0;
    const sessions = Number(slice?.sessions) || 0;
    const money = slice?.cost_presentation ? (costPresentationMeta(slice.cost_presentation) || [])[0] : '';
    return [word, money, calls ? plural(calls, 'model call') : '', sessions ? plural(sessions, 'session') : '',
        tokensText(Number(slice?.tokens))].filter(Boolean).join(' · ');
}

/** The Details rows `[label, html]` for a root's breakdown view; none without one. */
export function costBreakdownRows(view) {
    if (!view || typeof view !== 'object') return [];
    const rows = (slices, word) => (Array.isArray(slices) ? slices : [])
        .map((slice) => escapeHtmlText(costSliceLine(slice, word(slice?.key)))).join('<br>');
    return [['Spent on', rows(view.by_purpose, purposeWord)], ['Ran on', rows(view.by_route, routeWord)]];
}

/** Keep a ROOT card's breakdown rows from the task detail it fetched (`record.costBreakdownRows`, which the
 *  card's Details append). Only a detail carrying the view changes them: an event, summary or replayed record
 *  without one changes nothing, and rows only grow with the ledger. True when the rows changed. */
export function syncCostBreakdown(record, detail) {
    const view = detail?.cost_breakdown;
    if (!record || record.isSubagent || !view || typeof view !== 'object') return false;
    const rows = costBreakdownRows(view);
    if (JSON.stringify(rows) === JSON.stringify(record.costBreakdownRows ?? [])) return false;
    record.costBreakdownRows = rows;
    return true;
}
