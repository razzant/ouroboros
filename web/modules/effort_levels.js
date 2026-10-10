// The reasoning-effort vocabulary of the web UI: the 8-tier runtime scale (a mirror of
// `ouroboros/config.py` EFFORT_SCALE, pinned by tests/test_available_subagents_ui_static.py),
// the 7 owner tiers the range control offers (`minimal` is a runtime tier the owner never
// sets as a standing level), the owner's range as the server reads it, and the English
// label of a tier read through the catalog seam so a translated install hears the same word
// the strip shows.
import { fmt, fmtInto, tr } from './i18n.js';

export const EFFORT_SCALE = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'];
export const EFFORT_LABELS = Object.freeze({
    none: 'None', minimal: 'Minimal', low: 'Low', medium: 'Medium', high: 'High', xhigh: 'X-High', max: 'Max', ultra: 'Ultra',
});
// The owner-facing subset: `minimal` is a valid runtime tier (bench adapters and a child's
// own switch_model use it) but is deliberately not offered as an owner level — sub-`low`
// thinking is a per-call tactical choice, not a standing configuration.
export const EFFORT_OPTIONS = [
    { value: 'none', label: 'None' }, { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' }, { value: 'high', label: 'High' },
    { value: 'xhigh', label: 'X-High' }, { value: 'max', label: 'Max' }, { value: 'ultra', label: 'Ultra' },
];
export const OWNER_TIERS = EFFORT_OPTIONS.map((option) => option.value);
// The range the server reads when nothing is stored (config: OUROBOROS_EFFORT_MIN/TASK/MAX).
export const EFFORT_RANGE_DEFAULT = Object.freeze({ min: 'low', recommended: 'medium', max: 'high' });

export function effortTier(value) {
    const tier = String(value || '').trim().toLowerCase();
    return EFFORT_SCALE.includes(tier) ? tier : '';
}

/** The English word for a tier; an unknown value shows as it is. */
export function effortLabel(tier) {
    return EFFORT_LABELS[effortTier(tier)] || String(tier || '');
}

/** The tier's word through the catalog seam: a card or an aria text is never overlay-translated. */
export function effortText(tier) {
    const known = effortTier(tier);
    return known ? tr(`effort.tier.${known}`, EFFORT_LABELS[known]) : String(tier || '');
}

export const effortRank = (tier) => EFFORT_SCALE.indexOf(effortTier(tier));

/** The owner-tier index a stored tier is SHOWN at: `minimal` sits at Low (the nearest owner
 *  tier above it); an unknown value is -1. */
export function ownerLevelIndex(tier) {
    const known = effortTier(tier);
    if (!known) return -1;
    const exact = OWNER_TIERS.indexOf(known);
    if (exact >= 0) return exact;
    const rank = effortRank(known);
    return OWNER_TIERS.findIndex((owner) => effortRank(owner) >= rank);
}

/**
 * The server's tolerant read of the three stored keys: an unknown value is that key's
 * default; the minimum is lowered to the recommended level and the maximum raised to it,
 * so the triple always orders min ≤ recommended ≤ max.
 */
export function normalizeEffortRange(range = {}) {
    const pick = (key) => effortTier(range?.[key]) || EFFORT_RANGE_DEFAULT[key];
    const recommended = pick('recommended');
    const lower = (a, b) => (effortRank(a) <= effortRank(b) ? a : b);
    const higher = (a, b) => (effortRank(a) >= effortRank(b) ? a : b);
    return { min: lower(pick('min'), recommended), recommended, max: higher(pick('max'), recommended) };
}

/** The same read over a settings document (`GET /api/settings`), by the three flat keys. */
export function effortRangeFromSettings(settings = {}) {
    return normalizeEffortRange({
        min: settings?.OUROBOROS_EFFORT_MIN, recommended: settings?.OUROBOROS_EFFORT_TASK, max: settings?.OUROBOROS_EFFORT_MAX,
    });
}

export function sameEffortRange(a, b) {
    return ['min', 'recommended', 'max'].every((key) => effortTier(a?.[key]) === effortTier(b?.[key]));
}

/** "Low · High · High" — the triple as one short phrase. */
export function effortRangePhrase(range) {
    const r = normalizeEffortRange(range);
    return [r.min, r.recommended, r.max].map(effortText).join(' · ');
}

/** The Behavior tab's one read-only line about the range and where it is edited. */
export const EFFORT_RANGE_SUMMARY_TEMPLATE = 'Effort range: {min} · {recommended} · {max} (minimum · recommended · maximum). Change it with the round Effort button next to Swarm in the chat.';
const summaryParams = (settings) => {
    const r = effortRangeFromSettings(settings);
    return { min: effortText(r.min), recommended: effortText(r.recommended), max: effortText(r.max) };
};
export function effortRangeSummaryText(settings) {
    return fmt(EFFORT_RANGE_SUMMARY_TEMPLATE, summaryParams(settings));
}
export function paintEffortRangeSummary(element, settings) {
    if (element) fmtInto(element, EFFORT_RANGE_SUMMARY_TEMPLATE, summaryParams(settings));
}
