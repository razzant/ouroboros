/**
 * A processing tier beside its model (docs/DESIGN.md "Composer owner controls", "A work's model on its card"): the
 * strip at the foot of the Model sheet, its words, and what a route says about each tier. The stored vocabulary is the
 * settings' own (`standard`, `fast`, `economy`); the owner reads `Flex` for `economy`. `''` is Default: no tier of the
 * scope's own, so it takes what it would take without one (Main's own tier, else the default one). An explicit
 * Standard is a choice like the others, never the same as Default.
 *
 * Support is the catalog's word, never a guess: `listed` when the route's transport advertises the tier and its
 * accounts list it for the model, `unlisted` when either one's list leaves it out, `unknown` when nothing says. A
 * choice is never refused here: an unlisted or unconfirmed tier may be served as Standard, and the strip says so
 * before the press; the work's card says what its calls were actually served as.
 */
import { fmt, tr } from './i18n.js';
import { escapeHtmlAttr as esc } from './ui_primitives.js';
import { routeSource, shortName } from './chat_model_routes.js';

/** The tiers in the strip's order: cheaper and slower first. */
export const TIERS = Object.freeze(['economy', 'standard', 'fast', 'ultrafast']);
const WORDS = Object.freeze({ economy: 'Flex', standard: 'Standard', fast: 'Fast', ultrafast: 'Ultrafast' });
/** A tier in the owner's word; '' (Default) and anything unknown read as Default. */
export const tierWord = (tier) => (WORDS[tier] ? tr(`processing.tier.${tier}`, WORDS[tier]) : tr('processing.default', 'Default'));
/** Whether a tier is one the closed capsule and the card name: only one that is not Standard. */
export const tierNamed = (tier) => tier === 'fast' || tier === 'economy' || tier === 'ultrafast';
// What each tier means, said once (its price is its endpoint's, never the catalog's Standard price).
const MEANS = Object.freeze({
    economy: ['processing.means.economy', 'Lower price, slower; a provider may be out of capacity for it.'],
    standard: ['processing.means.standard', 'The ordinary service at the catalog price.'],
    fast: ['processing.means.fast', 'Faster replies at a higher price.'],
    ultrafast: ['processing.means.ultrafast', 'The fastest replies at the highest price, where a source offers it.'],
});

/**
 * Whether `route`'s source and model list `tier`: `listed`, `unlisted` or `unknown` (Standard is every route's
 * ordinary service). `index` is the catalog index (chat_model_routes.js).
 */
export function tierSupport(index, route, tier) {
    if (!route || !tier || tier === 'standard') return 'listed';
    const transport = index?.source?.(routeSource(route))?.transport;
    const modes = index?.row?.(route)?.modes;
    if ((Array.isArray(transport) && !transport.includes(tier)) || (modes && !modes.has(tier))) return 'unlisted';
    // Ultrafast needs both words: the transport's normalized tier and the model's; a native mode id is neither.
    if (tier === 'ultrafast') return Array.isArray(transport) && modes?.has(tier) ? 'listed' : 'unknown';
    return modes?.has(tier) && (!Array.isArray(transport) || transport.includes(tier)) ? 'listed' : 'unknown';
}

/** Whether the strip offers Ultrafast here: never for Main (its tier is a Settings value, which never holds it),
 *  else where the route lists it, or where it is already this scope's choice (so it can be seen and left). */
export const offersUltrafast = ({ main = false, support = 'unknown', choice = '' } = {}) => !main
    && (support === 'listed' || choice === 'ultrafast');

/**
 * The tier a scope's new calls take: its own `choice`, else Main's own tier for what follows Main (`follows`), else the
 * default one. `tiers` is `/api/state`'s `processing_tiers` (`{ main, default }`).
 */
export function resolvedTier({ choice = '', follows = false, tiers = null } = {}) {
    return choice || (follows ? tiers?.main || tiers?.default : tiers?.default) || '';
}

/**
 * One sentence before a press: what `tier` means and what this route says of it. An unlisted or unconfirmed tier
 * may be served as Standard; that is said here, before the choice.
 */
export function tierExplanation(tier, { support = 'unknown', route = null, sourceLabel = '' } = {}) {
    const shown = tier || 'standard';
    const means = tr(...MEANS[shown]);
    if (shown === 'standard' || support === 'listed') return means;
    const model = route ? shortName(route) : '';
    if (support === 'unlisted') {
        return `${means} ${fmt('{source} does not list {tier} for {model}: its calls go as Standard.',
            { source: sourceLabel || tr('processing.this_source', 'This source'), tier: tierWord(shown), model })}`;
    }
    return `${means} ${fmt('Not confirmed for {model} here: a provider may serve it as Standard.', { model })}`;
}

/**
 * What the newest call was served as, beside what was chosen (the card's fact and its sheet): '' when nothing says,
 * or when it was served as asked. `last` is the work's `processing.last` receipt.
 */
export function servedWords(last) {
    if (!last || typeof last !== 'object') return '';
    const asked = last.requested || '';
    const served = last.observed && last.observed !== 'unknown' ? last.observed : '';
    if (!asked && !served) return '';
    if (asked && served && asked !== served) return fmt('{asked} asked · served {served}', { asked: tierWord(asked), served: tierWord(served) });
    // Standard sent with no tier is the ordinary service: only another tier left unsent is news.
    if (tierNamed(asked) && !last.submitted && last.reason === 'processing_not_submitted') {
        return fmt('{asked} asked · sent as Standard', { asked: tierWord(asked) });
    }
    // Sent as another tier and served as not reported (the host's Standard continuation of a refused tier): the
    // receipt still knows what was sent.
    if (tierNamed(asked) && !served && last.submitted && last.submitted !== asked) {
        return fmt('{asked} asked · sent as {submitted}', { asked: tierWord(asked), submitted: tierWord(last.submitted) });
    }
    if (asked && !served && tierNamed(asked)) return fmt('{asked} asked · not confirmed', { asked: tierWord(asked) });
    return '';
}

// The providers' own words that fold into each tier (_usage_response.processing_receipt); any other is said as it is.
const NATIVE_FOLD = Object.freeze({ default: 'standard', standard: 'standard', priority: 'fast', fast: 'fast',
    flex: 'economy', economy: 'economy', ultrafast: 'ultrafast' });

/** The receipt in full, for the sheet's state line: what was asked, sent and seen. */
export function receiptWords(last) {
    if (!last || typeof last !== 'object') return '';
    const parts = [];
    if (last.requested) parts.push(fmt('asked {tier}', { tier: tierWord(last.requested) }));
    parts.push(last.submitted ? fmt('sent {tier}', { tier: tierWord(last.submitted) }) : tr('processing.sent.none', 'sent with no tier'));
    // The provider's own word where it is not the folded one (an older Codex reports `ultrafast` as Fast).
    const native = Array.isArray(last.observedNative) ? last.observedNative.filter((word) => typeof word === 'string') : [];
    const nativeSaid = native.some((word) => NATIVE_FOLD[word] !== last.observed) ? ` (${native.join(', ')})` : '';
    parts.push(last.observed && last.observed !== 'unknown' ? `${fmt('served {tier}', { tier: tierWord(last.observed) })}${nativeSaid}`
        : tr('processing.served.unknown', 'served as not reported'));
    return parts.join(' · ');
}

/**
 * The strip: Flex · Standard · Fast as one track (a choice among a few short words), the chosen one filled; while
 * the scope has no tier of its own the tier it takes wears a ring instead, and `Default` stands beside them only
 * while there is a choice to return from. Under it what the scope is (`scope`) and the tier under the mouse or
 * focus, else the chosen one, explained for this route, in a box as tall as its tallest explanation (a hidden copy of
 * each holds it), so a preview never moves the track under the pointer. A finger previews nothing: its tap chooses.
 * A press chooses at once (`onChoose(tier)`, '' for Default); arrows move among the tiers without choosing.
 */
export function createTierStrip({ doc = globalThis.document, id = 'tier', onChoose = () => {} } = {}) {
    const el = doc.createElement('div');
    el.className = 'model-tier';
    // The shared strip (ui.css .ui-segment-group): one track, each tier as wide as its word, only the chosen one filled.
    el.innerHTML = `<div class="model-tier-row"><div class="ui-segment-group model-tier-track" role="radiogroup" aria-label="${esc(tr('processing.label', 'Processing tier'))}">`
        + TIERS.map((tier) => `<button type="button" class="ui-segment model-tier-seg" role="radio" aria-checked="false" tabindex="-1" data-tier="${tier}">${esc(tierWord(tier))}</button>`).join('')
        + `</div><button type="button" class="btn btn-sm btn-default model-tier-default" data-tier="" hidden>${esc(tr('processing.default', 'Default'))}</button></div>`
        + `<p class="model-tier-scope" id="${esc(id)}-scope" data-i18n-skip></p><div class="model-tier-meansbox">`
        + `<p class="model-tier-means" id="${esc(id)}-means" data-i18n-skip aria-live="polite"></p>`
        + '<div class="model-tier-sizer" data-i18n-skip aria-hidden="true"></div></div>';
    const segs = [...el.querySelectorAll('.model-tier-seg')];
    const reset = el.querySelector('.model-tier-default');
    const scopeLine = el.querySelector('.model-tier-scope');
    const meansLine = el.querySelector('.model-tier-means');
    const sizer = el.querySelector('.model-tier-sizer');
    let sized = '';
    el.querySelector('.model-tier-track').setAttribute('aria-describedby', `${id}-scope ${id}-means`);
    let view = { choice: '', resolved: '', scope: '', explain: () => '', busy: false };
    let preview = '';

    const sayMeans = () => {
        const words = view.explain(preview || view.choice || view.resolved);
        if (meansLine.textContent !== words) meansLine.textContent = words;
    };
    function paint(next) {
        view = { ...view, ...next };
        const chosen = view.choice;
        for (const seg of segs) seg.hidden = seg.dataset.tier === 'ultrafast' && !view.ultrafast;
        const shown = segs.filter((seg) => !seg.hidden);
        const focusable = shown.find((seg) => seg.dataset.tier === (chosen || view.resolved || 'standard')) || shown[1] || shown[0];
        // A save in flight is said, never by disabling: a disabled button would drop the keyboard's focus.
        const busy = String(Boolean(view.busy));
        for (const seg of segs) {
            const tier = seg.dataset.tier;
            seg.setAttribute('aria-checked', String(tier === chosen));
            seg.dataset.resolved = String(!chosen && tier === (view.resolved || 'standard'));
            seg.dataset.support = view.supportOf?.(tier) || 'listed';
            seg.tabIndex = seg === focusable ? 0 : -1;
            seg.setAttribute('aria-disabled', busy);
        }
        // Default pressed by the keyboard leaves with its choice: the focus goes to the tier the scope takes now.
        if (!chosen && !reset.hidden && reset.ownerDocument?.activeElement === reset) focusable?.focus({ preventScroll: true });
        reset.hidden = !chosen;
        reset.setAttribute('aria-disabled', busy);
        if (scopeLine.textContent !== view.scope) scopeLine.textContent = view.scope;
        // Every explanation a shown tier may say, each in a hidden copy stacked in the box's one cell.
        const all = shown.map((seg) => view.explain(seg.dataset.tier));
        const key = JSON.stringify(all);
        if (key !== sized) {
            sized = key;
            sizer.replaceChildren(...all.map((words) => {
                const line = doc.createElement('span');
                line.textContent = words;
                return line;
            }));
        }
        sayMeans();
    }
    const onClick = (event) => {
        const made = event.target?.closest?.('[data-tier]');
        if (!made || made.getAttribute('aria-disabled') === 'true' || view.busy) return;
        const tier = made.dataset.tier;
        if (tier === view.choice) return;
        onChoose(tier);
    };
    const onKey = (event) => {
        const shown = segs.filter((seg) => !seg.hidden);  // a hidden tier is never a stop
        const at = shown.indexOf(event.target?.closest?.('.model-tier-seg'));
        if (at < 0) return;
        const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
        const to = step ? Math.min(shown.length - 1, Math.max(0, at + step))
            : event.key === 'Home' ? 0 : event.key === 'End' ? shown.length - 1 : -1;
        if (to < 0) return;
        event.preventDefault?.();
        segs.forEach((seg) => { seg.tabIndex = seg === shown[to] ? 0 : -1; });
        shown[to].focus({ preventScroll: true });
    };
    // The mouse or focus on a tier says, before any press, what it means here (in the same box, so nothing moves).
    const onOver = (event) => {
        if (event.type === 'pointerover' && event.pointerType && event.pointerType !== 'mouse') return;
        const next = event.target?.closest?.('.model-tier-seg')?.dataset?.tier || '';
        if (next !== preview) { preview = next; sayMeans(); }
    };
    const onOut = () => { if (preview) { preview = ''; sayMeans(); } };
    el.addEventListener('click', onClick);
    el.addEventListener('keydown', onKey);
    el.addEventListener('pointerover', onOver);
    el.addEventListener('focusin', onOver);
    el.addEventListener('pointerleave', onOut);
    el.addEventListener('focusout', onOut);
    return {
        el, paint, segs, reset,
        destroy() {
            el.removeEventListener('click', onClick);
            el.removeEventListener('keydown', onKey);
            el.removeEventListener('pointerover', onOver);
            el.removeEventListener('focusin', onOver);
            el.removeEventListener('pointerleave', onOut);
            el.removeEventListener('focusout', onOut);
        },
    };
}
