/**
 * A processing tier beside its model (docs/DESIGN.md "Composer owner controls", "A work's model on its card"): the
 * strip at the foot of the Model sheet, its words, and what a route says about each tier. The stored vocabulary is the
 * settings' own (`standard`, `fast`, `economy`, `ultrafast`); the owner reads `Flex` for `economy`. `''` is Default: no
 * tier of the scope's own, so it takes what it would take without one (Main's own tier, else the default one). An
 * explicit Standard is a choice like the others, never the same as Default.
 *
 * Support is the catalog's word for one route as it is bound (its transport, its model, and the account it is pinned
 * to or Auto), never a guess and never another route's: `listed` when everything that would serve it lists the tier,
 * `some` when only some of the source's accounts are known to, `unlisted` when its transport cannot carry it or a
 * key model's whole endpoint list leaves it out, and `unknown` when nothing says (a list not read, empty or shipped,
 * an account in a pause, an account's list without the tier, or a pin that is not known is no refusal and no
 * confirmation). The strip offers the tiers that are not proven absent (Ultrafast
 * where something lists it or its transport is known to carry it), says what an uncertain one may be served as
 * before the press, and the work's card says what its calls were actually asked, sent and served as.
 */
import { openConfirmDialog } from './confirm_dialog.js';
import { fmt, tr, translateString } from './i18n.js';
import { PRICE_UNKNOWN, priceWords } from './route_editor_primitives.js';
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
    economy: ['processing.means.economy', 'Requests slower processing at a discount when capacity is available.'],
    standard: ['processing.means.request_standard', 'Request Standard processing.'],
    fast: ['processing.means.fast', 'Requests faster processing. It may cost more.'],
    ultrafast: ['processing.means.ultrafast', 'Requests the fastest processing this source offers. It may cost more.'],
});

// One subscription route among its source's accounts: the pinned one's own word, or under Auto all of theirs. An
// account's list is its positive word only (the engine's lists are not exhaustive entitlements): a tier it leaves
// out is not refused there, so accounts never prove a tier absent; only a transport's ceiling does. A pin that is
// not known (`null`: the settings were not read) is no account's word at all, and never read as Auto. Under Auto a
// tier is every account's only when the source's whole pool was read and each account that may serve the model
// lists it; one nothing is known of stays in that count.
function accountsSay({ accounts, complete }, tier, account) {
    if (account == null) return 'unknown';
    if (account) return accounts.find((entry) => entry.id === account)?.modes?.has(tier) ? 'listed' : 'unknown';
    const serving = accounts.filter((entry) => !entry.absent);  // a live whole list without the model serves none of its calls
    const yes = serving.filter((entry) => entry.modes?.has(tier)).length;
    return !yes ? 'unknown' : complete && yes === serving.length ? 'listed' : 'some';
}
// A key's route by its model's own endpoints (`detail`, model_catalog_store.js): a list read whole may leave a tier out.
const endpointsSay = (detail, tier) => (detail?.state !== 'ok' ? 'unknown'
    : detail.modes.has(tier) ? 'listed' : detail.complete ? 'unlisted' : 'unknown');

/**
 * Whether `route` lists `tier` where it is bound: `listed`, `some`, `unlisted` or `unknown` (Standard is every
 * route's ordinary service). `index` is the catalog index (chat_model_routes.js); `account` the subscription account
 * the route is pinned to ('' is Auto: every account of its source; null: not known); `detail` a key route's own
 * endpoints.
 */
export function tierSupport(index, route, tier, { account = '', detail = null } = {}) {
    if (!route || !tier || tier === 'standard') return 'listed';
    const said = index?.tiers?.(route);
    if (!said) return 'unknown';
    if (said.ceiling && !said.ceiling.includes(tier)) return 'unlisted';
    const declared = said.lane === 'subscription' ? accountsSay(said, tier, account) : endpointsSay(detail, tier);
    // Ultrafast needs both words: the transport's normalized tier and the model's; a native mode id is neither.
    return tier === 'ultrafast' && !said.ceiling && declared !== 'unlisted' ? 'unknown' : declared;
}

/** Whether `route`'s transport is known to carry `tier`: its ceiling names it. Nothing said is not a yes. */
export const tierCarried = (index, route, tier) => Boolean(index?.tiers?.(route)?.ceiling?.includes(tier));

/** A subscription sheet names its binding even when no account has published tiers for this model. */
export const pinDecides = (index, route) => index?.tiers?.(route)?.lane === 'subscription';

/**
 * What a question about a route was asked for, as one string: the route, the account it is pinned to ('' Auto, null
 * not known) and, where this read names them, the engine store and that account's fingerprint. An answer given for
 * one binding is never used for another.
 */
export function tierBinding(index, route, account = '', general = null) {
    const pin = route && routeSource(route).startsWith('subscription:') ? account : '';
    const evidence = [index, general].filter((read) => pin && read?.storeId && (!index?.storeId || index.storeId === read.storeId))
        .map((read) => [read.storeId, read.tiers?.(route)?.accounts.find((entry) => entry.id === pin)?.fingerprint])
        .find(([, fingerprint]) => fingerprint);
    return JSON.stringify([route?.model || '', Boolean(route?.use_local), pin ?? null, ...(evidence || ['', ''])]);
}

/**
 * The tiers the strip shows, in its order: Standard, every tier not proven absent, and whatever the scope chose or
 * takes now, so a wish the route does not serve stays in view and can be left. Ultrafast is new only where something
 * lists it, or where the route's transport is known to carry it (`carries`): there it is an attempt nothing confirms,
 * and is worded so; a transport that says nothing offers no Ultrafast. No list of model names decides any of it.
 */
export function canChooseTier(tier, { supportOf = () => 'unknown', carries = () => false } = {}) {
    if (!tier || tier === 'standard') return true;
    const support = supportOf(tier);
    return support !== 'unlisted' && (tier !== 'ultrafast' || support === 'listed' || support === 'some' || carries(tier));
}

/** Hide Standard only when every inherited value already means Standard here and no other tier is available or
 * unconfirmed. A whole endpoint list still says nothing about what omission asks the provider to do. */
export function redundantStandard(choice, inherited, { supportOf, ordinaryDefault } = {}) {
    return !choice && inherited.every((tier) => tier === 'standard' || (tier === '' && ordinaryDefault))
        && TIERS.every((tier) => tier === 'standard' || supportOf?.(tier) === 'unlisted');
}

export function offeredTiers(view = {}) {
    const { choice = '', resolved = '' } = view;
    return redundantStandard(choice, [resolved], view) ? []
        : TIERS.filter((tier) => tier === choice || tier === resolved || canChooseTier(tier, view));
}

/**
 * The tier a scope's new calls take: its own `choice`, else Main's own tier for what follows Main (`follows`), else the
 * default one. `tiers` is `/api/state`'s `processing_tiers` (`{ main, default }`).
 */
export function resolvedTier({ choice = '', follows = false, tiers = null } = {}) {
    return choice || (follows ? tiers?.main || tiers?.default : tiers?.default) || '';
}

/**
 * Before a press: what `tier` means and what this route says of it. A tier some accounts list, an unlisted or an
 * unconfirmed one may be served as Standard; that is said here, before the choice, with `price` when one is told.
 */
export function tierExplanation(tier, { support = 'unknown', route = null, sourceLabel = '', price = '', unsent = false } = {}) {
    const shown = tier;
    const openRouter = route && routeSource(route).split('@')[0] === 'openrouter';
    if (!shown || shown === 'standard') {
        const means = shown ? tr(...MEANS.standard) : tr('processing.means.omit', 'No tier requested.');
        const native = !shown && openRouter ? tr('processing.omit.openrouter', 'OpenRouter may apply a workspace default or another tier.') : '';
        return [means, native, price].filter(Boolean).join(' ');
    }
    const means = tr(...MEANS[shown]);
    const words = { source: sourceLabel || tr('processing.this_source', 'This source'), tier: tierWord(shown), model: route ? shortName(route) : '' };
    const here = support === 'listed' ? ''
        : support === 'some' ? fmt('Only some {source} accounts are known to list {tier} for {model}: on the others a call may go as Standard.', words)
            // A transport that cannot carry the tier never sends it (a local model, a forwarding-less adapter).
            : support === 'unlisted' && unsent ? fmt('{source} does not send {tier}.', words)
            : support === 'unlisted' ? fmt('{source} does not list {tier} for {model}: its calls may go as Standard.', words)
                : fmt('Not confirmed for {model} here: a provider may serve it as Standard.', { model: words.model });
    return [means, here, price].filter(Boolean).join(' ');
}

/**
 * What `tier` is listed at on a key's route, from its model's own endpoints (`detail.endpoints`, USD per 1M tokens),
 * and only when that list is whole and every endpoint of the tier states a price: the one price they share, or their
 * range by provider. '' otherwise: a list that is not whole, or an endpoint without a price, prices nothing. It is
 * what the catalog lists, said so; never the model's Standard price, one endpoint's among several, a multiple of
 * anything, or what a call will be charged.
 */
export function tierPriceWords(detail, tier) {
    const serving = tier && detail?.state === 'ok' && detail.complete ? detail.endpoints.filter((endpoint) => endpoint.mode === tier) : [];
    if (!serving.length || serving.some((endpoint) => !priceWords(endpoint.price))) return '';
    const stated = new Set(serving.map((endpoint) => priceWords(endpoint.price, { unit: true })));
    if (stated.size === 1) return fmt('Listed at {price}.', { price: [...stated][0] });
    const figures = serving.flatMap((endpoint) => [endpoint.price, ...(Array.isArray(endpoint.price.tiers) ? endpoint.price.tiers : [])]);
    const edge = (key, pick) => pick(...figures.map((figure) => figure[key]));
    const range = priceWords({ input: edge('input', Math.min), output: edge('output', Math.min),
        tiers: [{ input: edge('input', Math.max), output: edge('output', Math.max) }] }, { unit: true });
    return range ? fmt('Listed at {price}, depending on the provider.', { price: range }) : '';
}

/**
 * One bound route's word on every tier, for the strip and for Settings: `supportOf(tier)`, `carries(tier)` (its
 * transport is known to carry it), `priceOf(tier)` (a key's tier says what its endpoints list or
 * that its price is unknown; a subscription names none) and `explain(tier)`. `account`: '' Auto, null not known.
 */
export function routeTiers({ index = null, route = null, account = '', detail = null, sourceLabel = '' } = {}) {
    const { lane, ceiling } = index?.tiers?.(route) || {};
    const supportOf = (tier) => tierSupport(index, route, tier, { account, detail });
    const priceOf = (tier) => (lane === 'api'
        ? tierPriceWords(detail, tier) || `${translateString(PRICE_UNKNOWN)}.` : '');
    const accountLabel = lane !== 'subscription' ? '' : account == null ? tr('processing.account.unknown', 'Account not known')
        : account ? fmt('Pinned account {account}', { account }) : tr('processing.account.auto', 'Account: Auto');
    // The host's own transport (a key's, the local model's) decides what is sent; an engine's list does not.
    const unsent = (tier) => lane !== 'subscription' && Boolean(ceiling) && !ceiling.includes(tier);
    // An ordinary-only host adapter has no tier field; a subscription or a model's endpoint list is not that proof.
    const ordinaryDefault = lane !== 'subscription' && ceiling?.length === 1 && ceiling[0] === 'standard';
    return { supportOf, priceOf, accountLabel, ordinaryDefault, unsent, carries: (tier) => tierCarried(index, route, tier),
        explain: (tier) => tierExplanation(tier, { support: supportOf(tier), route, sourceLabel, price: priceOf(tier), unsent: unsent(tier) }) };
}

/**
 * The options of a role's Processing select in Settings, in `base`'s order: each tier of `base` its bound route
 * offers (`said`: its `routeTiers`, null before a route is known; `canChooseTier`) and the stored one whatever the
 * route says, an uncertain one worded so: `Fast · some accounts`, `Fast · not confirmed`, `Fast · not listed`, and one
 * its host transport cannot carry `Fast · not sent`.
 */
export function tierChoices(base, said = null, selected = '') {
    const supportOf = said?.supportOf || (() => 'unknown');
    const notes = { some: tr('processing.option.some', 'some accounts'), unknown: tr('processing.option.unknown', 'not confirmed'),
        unlisted: tr('processing.option.unlisted', 'not listed') };
    return [...base.filter((tier) => tier === selected || canChooseTier(tier, said || {})),
        ...(TIERS.includes(selected) && !base.includes(selected) ? [selected] : [])].map((tier) => {
        const note = said && tier !== 'standard' ? (said.unsent?.(tier) ? tr('processing.option.unsent', 'not sent') : notes[supportOf(tier)]) : '';
        return { value: tier, label: note ? `${tierWord(tier)} · ${note}` : tierWord(tier),
            disabled: !base.includes(tier) || !canChooseTier(tier, said || {}) };
    });
}

/** What Main's Ultrafast can reach, regardless of which models are running now. */
export const MAIN_ULTRAFAST_SCOPE = 'Main requests this tier on its next calls. Running and new work that follows Main can also request it, even on a fallback model. A tier set for the work or fallback takes precedence. New subagents inherit the work’s tier unless they have one of their own.';
export const MAIN_ULTRAFAST_ROLES = 'Direct reviews using Main and Safety falling back to Main can also inherit its tier when they have no tier of their own.';
export const PIN_OBSERVATION = 'Ultrafast will follow this named profile. The catalog shows its last observed account. A sign-in since that read may already have replaced it.';
export const MAIN_ULTRAFAST_ROLLBACK = 'An older Ouroboros cannot read the settings file with Ultrafast saved for Main. Before rolling back, choose Standard or Default for Main.';

/**
 * Before selecting Ultrafast for Main, the shared dialog names the route it is asked
 * for, what it can reach, what that route says of Ultrafast (`explain`, its price in it when `priceSaid`;
 * else that the cost is not known here), and what an older version does with the setting. Resolves whether the
 * owner confirmed; a cancel writes nothing.
 */
export async function confirmMainUltrafast({ model = '', source = '', explain = '', priceSaid = false, pinned = false,
    dialog = openConfirmDialog, doc = globalThis.document } = {}) {
    return Boolean(await dialog({
        title: 'Ultrafast for Main',
        body: MAIN_ULTRAFAST_ROLLBACK,
        confirmLabel: 'Use Ultrafast',
        content: { label: 'Ultrafast for Main', mount(container) {
            const line = (text, literal = false) => {
                const made = doc.createElement('p');
                made.className = 'chat-model-sheet-fact';
                made.textContent = text;
                if (literal) made.dataset.i18nSkip = '';  // composed through tr()/fmt(): it names a model and its source
                return made;
            };
            container.append(...[model ? line([model, source].filter(Boolean).join(' · '), true) : null,
                line(MAIN_ULTRAFAST_SCOPE), line(MAIN_ULTRAFAST_ROLES),
                pinned ? line(PIN_OBSERVATION) : null,
                line(explain || tr(...MEANS.ultrafast), true),
                priceSaid ? null : line('What Ultrafast costs on this route is not known here.')].filter(Boolean));
        } },
    }));
}

/**
 * What the newest call came to, beside what was chosen (the card's fact and its sheet); '' when nothing says, or when
 * it was sent and served as asked. `last` is the work's `processing.last` receipt (`_usage_response.processing_receipt`).
 * Three things read apart: a tier sent as another one (a native setting of the route, or the host's Standard
 * continuation: `sent as Standard`) or not sent at all (`sent with no tier`, never called Standard); a tier the
 * provider served otherwise (`served Fast`); and a named tier nothing echoed (`not confirmed`).
 */
export function servedWords(last) {
    if (!last || typeof last !== 'object' || !last.requested) return '';
    const asked = last.requested;
    const sent = last.submitted || '';
    const served = TIERS.includes(last.observed) ? last.observed : '';
    const how = sent && sent !== asked ? fmt('sent as {submitted}', { submitted: tierWord(sent) })
        : !sent && TIERS.includes(asked) ? tr('processing.sent.none', 'sent with no tier') : '';
    // Echoes of several tiers are mixed, a provider word no tier folds into is said as it came: neither is a tier.
    const came = last.observed === 'mixed' ? tr('processing.served.mixed', 'served mixed tiers')
        : served && served !== (sent || asked) ? fmt('served {tier}', { tier: tierWord(served) })
            : !served && nativeWords(last).length ? fmt('provider said {words}', { words: nativeWords(last).join(', ') })
                : !served && !how && tierNamed(asked) ? tr('processing.not_confirmed', 'not confirmed') : '';
    return how || came ? [fmt('{tier} asked', { tier: tierWord(asked) }), how, came].filter(Boolean).join(' · ') : '';
}

// The providers' own words that fold into each tier (_usage_response.processing_receipt); any other is said as it is.
const NATIVE_FOLD = Object.freeze({ default: 'standard', standard: 'standard', priority: 'fast', fast: 'fast',
    flex: 'economy', economy: 'economy', ultrafast: 'ultrafast' });
/** The provider's own words of a call that fold into no tier (an opaque `auto` among them), as they came. */
function nativeWords(last) {
    const native = Array.isArray(last?.observedNative) ? last.observedNative.filter((word) => typeof word === 'string') : [];
    return TIERS.includes(last?.observed) || last?.observed === 'mixed' ? [] : native.filter((word) => !NATIVE_FOLD[word]);
}

/** The receipt in full, for the sheet's state line: what was asked, sent and seen. A call nothing reported a tier
 *  for is said so, whether the provider answered without one or with none at all: neither is Standard. */
export function receiptWords(last) {
    if (!last || typeof last !== 'object') return '';
    const parts = [];
    if (last.requested) parts.push(fmt('asked {tier}', { tier: tierWord(last.requested) }));
    parts.push(last.submitted ? fmt('sent {tier}', { tier: tierWord(last.submitted) }) : tr('processing.sent.none', 'sent with no tier'));
    // The provider's own word where it is not the folded one (an older Codex reports `ultrafast` as Fast), or where
    // it is no tier this page knows: said as it came, never read as a tier.
    const native = Array.isArray(last.observedNative) ? last.observedNative.filter((word) => typeof word === 'string') : [];
    const nativeSaid = native.some((word) => NATIVE_FOLD[word] !== last.observed) ? ` (${native.join(', ')})` : '';
    // Several echoes are mixed; a word no tier folds into (`auto`) is what the provider reported, never "not
    // reported"; only a call with no such field at all is that.
    parts.push(`${TIERS.includes(last.observed) ? fmt('served {tier}', { tier: tierWord(last.observed) })
        : last.observed === 'mixed' ? tr('processing.served.mixed', 'served mixed tiers')
            : native.length ? tr('processing.served.native', 'served as the provider reported')
                : tr('processing.served.unknown', 'served as not reported')}${nativeSaid}`);
    return parts.join(' · ');
}

/**
 * The strip: the tiers this route offers (`offeredTiers`) as one track (a choice among a few short words), the chosen
 * one filled; while the scope has no tier of its own the tier it takes wears a ring instead, and `Default` stands
 * beside them only while there is a choice to return from. Each tier wears what the route says of it (`data-support`);
 * with none to offer the strip is not there. Under it what the scope is (`scope`) and the tier under the mouse or
 * focus, else the chosen one, explained for this route, in a box as tall as its tallest explanation (a hidden copy of
 * each holds it), so a preview never moves the track under the pointer. A finger previews nothing: its tap chooses.
 * A press chooses at once (`onChoose(tier)`, '' for Default); arrows move among the tiers without choosing.
 */
export function createTierStrip({ doc = globalThis.document, id = 'tier', onChoose = () => {}, onFocusLost = () => {} } = {}) {
    const el = doc.createElement('div');
    el.className = 'model-tier';
    // The shared strip (ui.css .ui-segment-group): one track, each tier as wide as its word, only the chosen one filled.
    el.innerHTML = `<div class="model-tier-row"><div class="ui-segment-group model-tier-track" role="radiogroup" aria-label="${esc(tr('processing.label', 'Processing tier'))}">`
        + TIERS.map((tier) => `<button type="button" class="ui-segment model-tier-seg" role="radio" aria-checked="false" tabindex="-1" data-tier="${tier}">${esc(tierWord(tier))}</button>`).join('')
        + `</div><button type="button" class="btn btn-sm btn-default model-tier-default" data-tier="" hidden>${esc(tr('processing.default', 'Default'))}</button></div>`
        + `<p class="model-tier-scope" id="${esc(id)}-scope" data-i18n-skip></p><div class="model-tier-meansbox">`
        + `<p class="model-tier-means" id="${esc(id)}-means" data-i18n-skip aria-live="polite"></p>`
        + '<div class="model-tier-sizer" data-i18n-skip aria-hidden="true"></div></div>';
    const row = el.querySelector('.model-tier-row');
    const segs = [...el.querySelectorAll('.model-tier-seg')];
    const reset = el.querySelector('.model-tier-default');
    const scopeLine = el.querySelector('.model-tier-scope');
    const meansLine = el.querySelector('.model-tier-means');
    const sizer = el.querySelector('.model-tier-sizer');
    let sized = '';
    el.querySelector('.model-tier-track').setAttribute('aria-describedby', `${id}-scope ${id}-means`);
    let view = { choice: '', resolved: '', scope: '', explain: () => '', busy: false };
    let preview = '';

    const explain = (tier) => [view.accountLabel, view.explain(tier)].filter(Boolean).join(' · ');
    const sayMeans = () => {
        const words = explain(preview || view.choice || view.resolved);
        if (meansLine.textContent !== words) meansLine.textContent = words;
    };
    function paint(next) {
        view = { ...view, ...next };
        const chosen = view.choice;
        const offered = offeredTiers(view);
        const shown = segs.filter((seg) => offered.includes(seg.dataset.tier));
        const focusable = shown.find((seg) => seg.dataset.tier === (chosen || view.resolved || 'standard')) || shown[1] || shown[0];
        const active = doc.activeElement;
        const leaving = (!chosen && active === reset) || (segs.includes(active) && !shown.includes(active));
        if (!offered.includes(preview)) preview = '';
        // Move focus before hiding its control, including when Default makes the entire strip redundant.
        for (const seg of shown) seg.hidden = false;
        if (leaving) {
            if (focusable) focusable.focus({ preventScroll: true });
            else onFocusLost();
        }
        row.hidden = !offered.length;
        // A save in flight is said, never by disabling: a disabled button would drop the keyboard's focus.
        const busy = String(Boolean(view.busy));
        el.setAttribute('aria-busy', busy);
        for (const seg of segs) {
            const tier = seg.dataset.tier;
            seg.setAttribute('aria-checked', String(tier === chosen));
            seg.dataset.resolved = String(!chosen && tier === view.resolved);
            seg.dataset.support = view.supportOf?.(tier) || (tier === 'standard' ? 'listed' : 'unknown');
            seg.tabIndex = seg === focusable ? 0 : -1;
            seg.setAttribute('aria-disabled', String(view.busy || !canChooseTier(tier, view)));
            seg.hidden = !shown.includes(seg);
        }
        reset.hidden = !chosen;
        reset.setAttribute('aria-disabled', busy);
        if (scopeLine.textContent !== view.scope) scopeLine.textContent = view.scope;
        // Every explanation a shown tier may say, each in a hidden copy stacked in the box's one cell.
        const all = [...new Set([...shown.map((seg) => seg.dataset.tier), chosen || view.resolved])].map(explain);
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
        if (!made || made.hidden || made.getAttribute('aria-disabled') === 'true' || view.busy) return;
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
