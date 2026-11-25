/**
 * A work's model on its card (docs/DESIGN.md "A work's model on its card"): the first fact of the card's meta line,
 * said from the server's published fact (CLIENT_CONTRACT §5: `works[].lead`, and `active_chat_activities[].lead` for a
 * direct turn). The line itself is rebuilt as one string by chat_activity.js, so the fact stands in a node of its
 * own before it: the rest of the line moves into a wrapper the card's record then writes to, a repaint of the line
 * never touches the fact, and an open popover keeps its anchor. A card whose work publishes no such fact (an older
 * record, a child, a portal) keeps the line it has today.
 *
 * Where the server offers the work's `model` control the fact is a button. It opens the Model sheet the composer's
 * capsule opens (model_sheet.js): its Main row is the one way to follow Main, and its facts say so; while the work runs
 * on a fallback its three exits stand above the list: Return, Wait and Keep; at its foot this work's processing tier,
 * from its next call. Each press is one `POST /api/tasks/{id}/model` (CLIENT_CONTRACT §5) whose answer says when it
 * applies; a tier press is its own act and moves nothing else. A refusal is said in the sheet in the owner's words,
 * and one whose decision the server saved (a 503) offers the same request again. Without the control the fact is plain
 * text: a control is the server's decision and is never guessed here. Beside the model the fact names the work's tier
 * when it is not Standard, and what its newest call came to when that differs (`works[].processing`). The sheet's
 * strip uses the same next route as the chosen row and endpoint read: Main's bound account for a follower, Auto
 * for an explicit lead, otherwise no account assumed.
 */
import { taskModel } from './api_client.js';
import { renderLiveCardMeta } from './chat_activity.js';
import { clampWithToggle } from './chat_reply_quote.js';
import { observeLayout } from './capsule.js';
import { fmt, tr, translateString } from './i18n.js';
import { isCoarsePointer, isStackLayout } from './layout.js';
import { modelCatalog } from './model_catalog_store.js';
import { closeModelSheet, openModelSheet } from './model_sheet.js';
import { sourceMark } from './model_source_mark.js';
import { receiptWords, resolvedTier, routeTiers, servedWords, tierNamed, tierWord } from './processing_choice.js';
import { modelWaitReasonLabel, modelWaitRoleLabel } from './model_wait.js';
import { controlIcon } from './page_icons.js';
import { showToast } from './toast.js';
import { bindPopoverPosition } from './ui_interactions.js';
import { describeRoute, normalizeRoute, routeKey, sameRoute, shortName } from './chat_model_routes.js';

// When a change of a work's model takes effect, in the answer's own code (`applied`).
export const APPLIED_WORDS = Object.freeze({
    next_step: 'Switches at the next step; the running call finishes.',
    at_start: 'Applies when it starts.',
});
// Why a work is on its model, by the fact's `basis`; any other basis is a role the work follows.
const BASIS_WORDS = Object.freeze({
    main: ['model.basis.main', 'follows Main'],
    room: ['model.basis.room', 'this room\'s model'],
    owner: ['model.basis.owner', 'you chose it'],
    owner_words: ['model.basis.owner_words', 'you asked for it'],
    owner_press: ['model.basis.owner_press', 'you pressed this option'],
    mind: ['model.basis.mind', 'Ouroboros switched to it'],
    place: ['model.basis.place', 'its place\'s model'],
    circle: ['model.basis.circle', 'its circle\'s model'],
});
// A fallback's cause is the failed call's code (lead_fact.adopt_fallback). The codes whose condition the wait card
// already names are said in its words (model_wait.js); any other code is shown as the server gave it.
const CAUSE_WAIT_REASONS = Object.freeze({ subscription_window_exhausted: 'quota', quota_exhausted: 'quota',
    auth_error: 'auth', auth_required: 'auth', connection_unavailable: 'connection_unavailable' });
// The two causes the fact's own words already say: a fallback with no failure code, and the mind's own switch.
const CAUSES_SAID = new Set(['configured fallback', 'switch_model']);
// A refused act by the server's `reason_code`. The last three are 503s: the decision is saved and only its delivery
// failed, so the same request may be sent again.
const REFUSAL_WORDS = Object.freeze({
    task_model_invalid: 'This model was not accepted.',
    request_id_required: 'The request was incomplete. Try again.',
    task_not_live: 'This work has already ended.',
    presence_model_unsupported: 'A place\'s model is set in its place sheet.',
    not_a_root_task: 'A subagent keeps the model and tier it started with; a change on its work\'s card reaches only that work\'s next steps and its new subagents.',
    model_control_unavailable: 'This work has no saved record yet, so its model cannot be changed.',
    cancel_pending: 'This work is stopping; its model stays as it is.',
    model_action_state: 'This work is not running on a fallback now.',
    model_recents_pending: 'Your choice is saved, but the list of recent models could not be updated yet.',
    model_queue_publish_failed: 'Your choice is saved, but the queue could not be updated yet.',
    model_wake_failed: 'Your choice is saved, but the work could not be told yet.',
});
// How many recent models the sheet offers under Main and the pinned ones.
const RECENT_ROWS = 3;
// These admitted leads use Auto. A wait-card move is not such a lead, even though it says `owner`.
const EXPLICIT_LEADS = new Set(['room', 'owner', 'owner_words', 'owner_press', 'place', 'circle']);
let serial = 0;
let openPopover = null;  // one sheet in the window: opening another closes it

/** Why a work left its primary, for the popover's first line: '' when the fact's own words already say it. */
export function causeWords(cause) {
    const code = String(cause || '').trim();
    if (!code || CAUSES_SAID.has(code)) return '';
    const reason = CAUSE_WAIT_REASONS[code];
    return reason ? translateString(modelWaitReasonLabel(reason)) : code;
}

/**
 * A refused act in the owner's words: `{ words, retry }`. `retry` is true for a 503, whose decision the server saved:
 * the same request id may be sent again. A code this page does not know keeps the server's own words.
 */
export function refusalWords(error) {
    const code = String(error?.body?.reason_code || '');
    const known = REFUSAL_WORDS[code];
    return { words: known ? tr(`model.refused.${code}`, known) : String(error?.message || error || ''),
        retry: Number(error?.status) === 503 };
}
const nodeOf = (doc, tag, className, text) => {
    const made = doc.createElement(tag);
    made.className = className;
    if (text !== undefined) made.textContent = text;
    return made;
};
const write = (node, text) => { if (node.textContent !== text) node.textContent = text; };

/** Why a work is on its model, in the product's short words; '' for a fact without a basis. */
export function basisWords(basis) {
    const known = BASIS_WORDS[basis];
    if (known) return tr(known[0], known[1]);
    if (basis === 'fallback') return tr('model.basis.fallback', 'a fallback');
    return basis ? fmt('follows {role}', { role: modelWaitRoleLabel(String(basis)) }) : '';
}

/**
 * One published fact in the words the card and the Main sheet say. `since` is the fact's `since` already in the
 * viewer's clock ('' when it has none). Null for a fact that names no model.
 *   name     what the fact shows: the source and the model, or the amber line of a fallback; `parts` is the same
 *            line as the fact lays it out (what leads, the model, what trails), joined by its real separators
 *   why      the words after it: why this model, or what a fallback left; '' for a work that follows Main
 *   pending  the route the next step uses when it differs, with when the change lands
 */
export function leadView(lead, { index = null, since = '', queued = false } = {}) {
    const said = describeRoute(index, lead);
    if (!said) return null;
    const basis = String(lead.basis || '');
    const fallback = basis === 'fallback';
    const primary = normalizeRoute(lead.primary);
    const pending = normalizeRoute(lead.pending);
    // A newly accepted primary can be published before the acting call changes (CLIENT_CONTRACT §5).
    const awaitingPrimary = sameRoute(primary, pending) && (!sameRoute(pending, said.route)
        || Boolean(lead.pending.basis && lead.pending.basis !== basis));
    // What the acting route left: Main for a work that followed it, else the model that was the work's own.
    const leftMain = primary && lead.primary.basis === 'main';
    const leftName = primary ? (leftMain ? 'Main' : shortName(primary)) : '';
    const left = primary && !awaitingPrimary ? (leftMain ? fmt('left Main ({model})', { model: shortName(primary) })
        : fmt('left {model}', { model: leftName })) : '';
    const origin = tr('model.fallback.settings', 'fallback from Settings');
    // CLIENT_CONTRACT §5: on the primary the cite rides the lead; away from it, the primary it names.
    const quote = primary ? lead.primary.cited : lead.cited;
    const cited = typeof quote?.text === 'string' && quote.text.length ? { text: quote.text, clipped: quote.clipped === true } : null;
    const parts = [fallback ? origin : said.sourceLabel, said.short, fallback && since ? fmt('since {time}', { time: since }) : ''];
    return {
        said, basis, fallback, primary, awaitingPrimary, leftMain, leftName, since, parts, cited,
        name: parts.filter(Boolean).join(' · '),
        why: fallback ? left : [basis === 'main' ? '' : basisWords(basis), left].filter(Boolean).join(' · '),
        pending: pending && !sameRoute(pending, said.route)
            ? fmt(queued ? '→ {model} when it starts' : '→ {model} at the next step', { model: shortName(pending) }) : '',
        cause: causeWords(lead.cause),
    };
}

/* --- the fact's node: built once, written in place, taken away with its work's fact --- */

function buildFact(record, deps) {
    const meta = record.metaEl;
    const doc = meta.ownerDocument || globalThis.document;
    const part = nodeOf(doc, 'span', 'chat-live-meta-part card-model');
    part.dataset.cardModel = '';
    const why = nodeOf(doc, 'span', 'chat-live-meta-text card-model-why');
    why.dataset.i18nSkip = '';  // said through tr()/fmt() here: it may name a model
    const whyWords = nodeOf(doc, 'span', 'card-model-words');
    why.append(whyWords);
    const pending = nodeOf(doc, 'span', 'chat-live-meta-text card-model-pending');
    pending.dataset.i18nSkip = '';
    const pendingWords = nodeOf(doc, 'span', 'card-model-words');
    pending.append(pendingWords);
    // The work's tier when it is not Standard, and what its newest call was served as when that differs.
    const tier = nodeOf(doc, 'span', 'chat-live-meta-text card-model-tier');
    tier.dataset.i18nSkip = '';
    const tierWords = nodeOf(doc, 'span', 'card-model-words');
    tier.append(tierWords);
    const sep = nodeOf(doc, 'span', 'chat-live-meta-sep');
    sep.innerHTML = '&nbsp;· ';
    // The rest of the line, as chat_activity.js writes it: its parts stay items of the line itself (style.css).
    const rest = nodeOf(doc, 'span', 'card-model-rest');
    rest.append(...[...meta.children]);
    meta.append(part, rest);
    const state = { meta, part, fact: null, factGroup: null, mark: null, name: null, why, whyWords, pending,
        pendingWords, tier, tierWords, sep, rest, doc, deps, view: null, input: null, markKey: null, pressable: null, watch: null };
    record.cardModel = state;
    record.metaEl = rest;
    // The separator is real text (a copied line keeps it), so it is shown only while something follows the fact.
    const Observer = doc.defaultView?.MutationObserver || globalThis.MutationObserver;
    if (typeof Observer === 'function') {
        state.watch = new Observer(() => syncSeparator(state));
        state.watch.observe(rest, { childList: true });
    }
    return state;
}

function syncSeparator(state) {
    const alone = !state.rest.children.length;
    if (state.sep.hidden !== alone) state.sep.hidden = alone;
    // Like chat_activity's other meta parts, the dot stays with the last fact, never on a line of its own.
    const last = !state.tier.hidden ? state.tier : !state.pending.hidden ? state.pending : !state.why.hidden ? state.why : state.factGroup;
    if (state.sep.parentNode !== last) last.append(state.sep);
}

// The fact as a button (the server offers the control) or as text (it does not); swapped only when that changes.
function shapeFact(record, state, pressable) {
    if (state.pressable === pressable) return;
    if (openPopover?.record === record) openPopover.close({ focus: false });  // its anchor goes with the control
    state.pressable = pressable;
    const { doc } = state;
    const fact = nodeOf(doc, pressable ? 'button' : 'span', pressable ? 'card-model-fact ui-hit-area' : 'card-model-fact');
    if (pressable) {
        fact.setAttribute('type', 'button');
        fact.setAttribute('aria-haspopup', 'dialog');
        fact.setAttribute('aria-expanded', 'false');
        fact.addEventListener('click', (event) => {
            event.stopPropagation?.();  // the card's own press opens its Details; this one opens the list
            if (openPopover?.record === record) openPopover.close({ focus: false });
            else openCardModel(record);
        });
    }
    const mark = nodeOf(doc, 'span', 'card-model-mark');
    mark.setAttribute('aria-hidden', 'true');
    // What leads (the source, or where a fallback came from), the model, what trails: three spans of one line, so
    // the lead gives way before the model's name on a narrow card (style.css). Their separators are real text.
    const name = nodeOf(doc, 'span', 'card-model-name');
    name.dataset.i18nSkip = '';
    name.append(nodeOf(doc, 'span', 'card-model-lead'), nodeOf(doc, 'span', 'card-model-id'), nodeOf(doc, 'span', 'card-model-tail'));
    fact.append(mark, name);
    if (pressable) {
        const caret = nodeOf(doc, 'span', 'card-model-caret');
        caret.setAttribute('aria-hidden', 'true');
        caret.innerHTML = controlIcon('chevron', 11);
        fact.append(caret);
    }
    const factGroup = nodeOf(doc, 'span', 'card-model-fact-group');
    factGroup.append(fact);
    state.fact?.remove();
    Object.assign(state, { fact, factGroup, mark, name, markKey: null });
    state.part.replaceChildren(factGroup, state.why, state.pending, state.tier);
}

// A subscription's mark is drawn from the catalog's `credentialHarness` (model_source_mark.js), which only a read
// brings: a fact that needs it asks for that read once, and every such mounted fact repaints when the catalog
// changes, as the composer's chip does (composer_model.js). No timer: the store says when it changed.
const catalogFacts = new WeakMap();  // store -> the records whose painted fact follows it
const catalogAsked = new WeakSet();  // the stores this module asked for their first read

function followCatalog(record, store) {
    let records = catalogFacts.get(store);
    if (!records) {
        records = new Set();
        catalogFacts.set(store, records);
        let queued = false;
        store.subscribe?.(() => {
            if (queued) return;
            queued = true;
            queueMicrotask(() => {
                queued = false;
                for (const each of [...records]) {
                    const state = each.cardModel;
                    if (!state || each.root?.isConnected === false) { records.delete(each); continue; }
                    paintCardModel(each, state.view);
                }
            });
        });
    }
    records.add(record);
    // Once per store: a failed read is the store's to say and to retry, never a loop of this one.
    const status = store.status?.();
    if (!catalogAsked.has(store) && status && !status.read && !status.reading) {
        catalogAsked.add(store);
        void store.read?.();
    }
}

function removeFact(record) {
    const state = record.cardModel;
    if (!state) return false;
    if (openPopover?.record === record) openPopover.close({ focus: false });
    state.watch?.disconnect();
    state.part.remove();
    state.meta.append(...[...state.rest.children]);
    state.rest.remove();
    record.metaEl = state.meta;
    record.cardModel = null;
    record.lead = null;
    catalogFacts.get(state.deps.store)?.delete(record);
    state.deps.rerender(record);  // the line names the model itself again
    return true;
}

/**
 * The server's call resolver includes authored fallback tiers. Older bodies without that fact retain the
 * primary's resolution: own choice, then Main's while following Main, else the default.
 */
export function workTier(view) {
    if (typeof view?.processing?.resolved === 'string') return view.processing.resolved;
    const own = String(view?.processing?.choice || '');
    const follows = view?.said?.basis === 'main' || Boolean(view?.said?.fallback && view?.said?.leftMain);
    return resolvedTier({ choice: own, follows, tiers: view?.tiers || null });
}

// The chooser acts on the next call; the card's fact and last receipt stay on the acting/observed route.
function nextChoice({ lead, said, main }) {
    const pending = normalizeRoute(lead.pending);
    const basis = pending ? lead.pending.basis : said.basis;
    const follows = basis === 'main';
    return { route: pending || (follows && main) || said.said.route, follows, pending,
        explicit: EXPLICIT_LEADS.has(basis) && Boolean(pending || lead.cause !== 'wait_card') };
}

/**
 * Paints one card's model fact. `view` is `{ lead, target, main, queued, since, processing, tiers }`: the published
 * fact, the task id the server's `model` control acts on ('' when it offers none), Main's route, whether the work has
 * not started, the fact's `since` in the viewer's clock, the work's tier fact (`works[].processing`) and the tiers
 * Main and a lead take (`processing_tiers`); null (or a fact without a model) takes the fact away. Returns whether a
 * fact stands on the card.
 */
export function paintCardModel(record, view, deps = {}) {
    if (!record?.metaEl) return false;
    const lead = normalizeRoute(view?.lead) ? view.lead : null;
    if (!lead) { removeFact(record); return false; }
    const fresh = !record.cardModel;
    const state = record.cardModel || buildFact(record, { store: modelCatalog, act: taskModel, toast: showToast,
        rerender: renderLiveCardMeta, coarse: isCoarsePointer, stack: isStackLayout,
        place: (popup, anchor) => bindPopoverPosition(popup, { anchor, align: 'start' }), ...deps });
    const said = leadView(lead, { index: state.deps.store.index(), since: view.since || '', queued: Boolean(view.queued) });
    // A repaint of the view a card already holds brings no new tiers, so it adopts none; only a view from the server does.
    const replay = view === state.view;
    state.view = { ...view, lead, target: String(view.target || ''), main: normalizeRoute(view.main), said };
    if (view.tiers && !replay) state.deps.store.adoptTiers?.(view.tiers);
    record.lead = lead;
    shapeFact(record, state, Boolean(state.view.target));
    // The mark itself is the identity: what it is drawn from (a subscription's `credentialHarness`) can arrive
    // after the words it stands beside, and a key of hand-picked fields would hide that change.
    const mark = sourceMark(said.said);
    if (mark !== state.markKey) { state.markKey = mark; state.mark.innerHTML = mark; }
    if (said.said.lane === 'subscription') followCatalog(record, state.deps.store);
    const [leads, id, tail] = state.name.children;
    // A separator rides at the end of the span before it, so a wrapped line never starts with one.
    write(leads, `${said.parts[0]}\u00a0·`);
    write(id, `\u00a0${said.parts[1]}${said.parts[2] ? '\u00a0·\u00a0' : ''}`);
    write(tail, said.parts[2]);
    write(state.whyWords, said.why);
    write(state.pendingWords, said.pending);
    state.why.hidden = !said.why;
    state.pending.hidden = !said.pending;
    const tier = workTier(state.view);
    const tierText = [tierNamed(tier) ? tierWord(tier) : '', servedWords(state.view.processing?.last)].filter(Boolean).join(' · ');
    // After the words of why or what comes next, the tier follows a real separator (a copied line keeps it).
    write(state.tierWords, tierText && (said.why || said.pending) ? `·\u00a0${tierText}` : tierText);
    state.tier.hidden = !tierText;
    // The receipt in full (asked, sent, served, a provider's own words) is the card's Details line, not the meta line's.
    const receipt = receiptWords(state.view.processing?.last);
    const detail = receipt ? fmt('{tier} · last call: {receipt}', { tier: state.view.processing?.choice || !tier ? tierWord(tier)
        : fmt('Default · {tier}', { tier: tierWord(tier) }), receipt }) : '';
    const detailChanged = (record.processingDetail || '') !== detail;
    record.processingDetail = detail;
    // Whose model this is rides the node as its look: the default is quiet, a choice stronger, a fallback amber.
    state.part.dataset.basis = said.fallback ? 'fallback' : said.basis === 'main' ? 'main' : 'own';
    state.fact.classList.toggle('model-fallback', said.fallback);
    if (state.view.target) {
        state.fact.setAttribute('aria-label', [fmt('This work\'s model: {model}, {source}', { model: said.said.short, source: said.said.sourceLabel }),
            said.fallback ? said.name : basisWords(said.basis)].filter(Boolean).join(', '));
    }
    syncSeparator(state);
    if (fresh || detailChanged) state.deps.rerender(record);  // the fact names the model now; Details say its last call
    if (openPopover?.record === record) openPopover.sync();
    return true;
}

const sameFact = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
const tierChoice = (processing) => [String(processing?.choice || ''), String(processing?.source || '')];

/**
 * An act's answer (`lead` for a model act, `processing` for a tier act) painted onto the card as it stands now, never
 * over the view the act was pressed in (`before`): a body that arrived meanwhile keeps its model, receipt and control.
 * An answer for a control the card no longer offers (withdrawn, or another work's) is dropped, and each part of it
 * stands only where no body has said that part again since the press. Returns whether the card took it.
 */
function adoptAnswer(record, state, before, { lead = null, processing = null } = {}) {
    const now = state.view;
    if (record.cardModel !== state || !now?.target || now.target !== before.target) return false;
    const next = { ...now };
    let took = false;
    if (normalizeRoute(lead) && sameFact(now.lead, before.lead)) {
        Object.assign(next, { lead, since: lead.since === now.lead.since ? now.since : '' });
        if (next.processing) {
            next.processing = { ...next.processing };
            delete next.processing.resolved; // the prior acting route's resolution is no longer this route's
        }
        took = true;
    }
    const sameResolution = sameFact(now.lead, before.lead) && now.processing?.resolved === before.processing?.resolved;
    if (processing && typeof processing === 'object' && sameFact(tierChoice(now.processing), tierChoice(before.processing))
        && (processing.choice || sameResolution)) {
        // The receipt is the newest body's own: an answer says the work's tier, never what a call ran as.
        const [choice, source] = tierChoice(processing);
        next.processing = { choice, ...(source ? { source } : {}), ...(now.processing?.last ? { last: now.processing.last } : {}) };
        // A concurrent body owns its resolved route; an act's resolution belongs only to the route it saw.
        if (sameResolution && typeof processing.resolved === 'string') next.processing.resolved = processing.resolved;
        took = true;
    }
    if (took) paintCardModel(record, next);
    return took;
}

/* --- the popover: this work's acts, then the capsule's list of models --- */

/** Opens the Model sheet of a card whose fact is pressable; null when the server offers no control for it. */
export function openCardModel(record) {
    const state = record?.cardModel;
    if (!state?.view?.target) return null;
    openPopover?.close({ focus: false });
    const { doc, deps } = state;
    const { store } = deps;
    // Above the field: whose model this is, a refusal said where it was pressed, and a fallback's own acts.
    const head = nodeOf(doc, 'div', 'card-model-head');
    const title = nodeOf(doc, 'div', 'card-model-pop-title');
    title.dataset.i18nSkip = '';
    const acts = nodeOf(doc, 'div', 'card-model-acts');
    // `Retry` stands beside a refusal only when the server saved the decision.
    const refusal = nodeOf(doc, 'p', 'card-model-refusal');
    refusal.setAttribute('role', 'status');
    refusal.dataset.i18nSkip = '';
    refusal.hidden = true;
    const refusalText = nodeOf(doc, 'span', 'card-model-refusal-words');
    const retry = nodeOf(doc, 'button', 'btn btn-sm btn-default card-model-retry', 'Retry');
    retry.setAttribute('type', 'button');
    refusal.append(refusalText, retry);
    // What this work's newest call was served as, beside its tier (the whole receipt is in Details): above the field,
    // its line held while a receipt may come, so one arriving with the sheet open never moves the list or the strip.
    const served = nodeOf(doc, 'p', 'card-model-served');
    served.dataset.i18nSkip = '';
    served.hidden = true;
    head.append(title, served, refusal, acts);
    let busy = false;
    let closed = false;
    let actsKey = null;
    let headKey = null;
    let unwatchHead = () => {};
    let sheet = null;
    let again = null;  // the refused request a 503 may send again, under its own id
    let unsubscribe = () => {};
    let detailAsked = null;  // the routed value whose own endpoints this sheet asked for
    const view = () => state.view;
    const next = () => nextChoice(view());
    function readTierDetail() {
        const { route } = next();
        const asking = `${routeKey(route)}|${store.detailEpoch?.() ?? ''}`;  // again after another connection's drop
        if (asking === detailAsked) return;
        detailAsked = asking;
        void store.readDetail?.(route);
    }
    const follows = () => next().follows;

    async function run(action, route = null, requestId = globalThis.crypto?.randomUUID?.() || `card-model-${Date.now()}-${++serial}`,
        tier = undefined) {
        if (busy || closed) return;
        busy = true;
        again = null;
        refusal.hidden = true;
        sheet?.el.setAttribute('aria-busy', 'true');
        sheet?.paint();
        const before = view();
        try {
            const reply = await deps.act(before.target, requestId, { action,
                ...(route ? { model: route.model, use_local: route.use_local } : {}),
                ...(action === 'processing' ? { processing: tier } : {}) });
            const kept = before.said.said.short;
            if (action === 'processing') {
                const words = tier ? fmt('{tier} for this work.', { tier: tierWord(tier) }) : tr('model.tier.default_work', 'Default tier for this work.');
                deps.toast(`${words} ${APPLIED_WORDS[reply?.applied] || ''}`.trim(), 'ok');
                // The answer's own tier stands on the card until the next state body says it again.
                adoptAnswer(record, state, before, { processing: reply?.processing });
                return;
            }
            deps.toast(action === 'wait' ? fmt('Waits for {model}: no paid calls until it answers.', { model: before.said.leftName || kept })
                : action === 'keep' ? fmt('Stays on {model}: your choice from now on.', { model: kept })
                    : APPLIED_WORDS[reply?.applied] || 'The model was changed.', 'ok');
            // The answer's own fact stands on the card until the next state body says it again.
            adoptAnswer(record, state, before, { lead: reply?.lead });
            void store.readRecents();  // the server wrote this act into the recent models
            close({ focus: true });
        } catch (error) {
            const said = refusalWords(error);
            if (closed) deps.toast(said.words, 'error');  // its sheet went while the answer was on its way
            else {
                again = said.retry ? { action, route, requestId, tier } : null;
                write(refusalText, said.words);
                retry.hidden = !again;
                refusal.hidden = false;
                sheet?.reposition();
            }
        } finally {
            busy = false;
            if (!closed) { sheet?.el.removeAttribute('aria-busy'); sheet?.paint(); }
        }
    }
    // Main's row follows Main; any other row is that route, Main's model now included (a model row is always itself).
    // The route still serving a pending switch is a choice again: it cancels the switch before the next step.
    const pick = (route) => {
        const choice = next();
        if (!route) return choice.follows ? close({ focus: true }) : run('follow');
        if (!choice.follows && !view().said.fallback && !choice.pending && sameRoute(route, choice.route)) return close({ focus: true });
        return run('set', route);
    };

    function paintHead() {
        const now = view();
        const said = now.said;
        let heading = said.fallback
            ? [said.awaitingPrimary ? [said.name, said.pending].filter(Boolean).join(' · ')
                : said.leftName ? fmt(said.since ? 'Left {model} at {time}' : 'Left {model}', { model: said.leftName, time: said.since })
                    : said.name, said.cause].filter(Boolean).join(' · ')
            : [tr('model.popover.title', 'Model for this work'), said.why || basisWords(said.basis),
                said.pending].filter(Boolean).join(' · ');
        if (said.cited && said.primary) heading += ` · ${basisWords(now.lead.primary.basis)}`;
        const nextHead = JSON.stringify([heading, said.cited]);
        if (nextHead !== headKey) {
            headKey = nextHead;
            unwatchHead();
            unwatchHead = () => {};
            const words = nodeOf(doc, 'span', 'card-model-pop-words', heading);
            title.replaceChildren(words);
            if (said.cited) {
                const line = nodeOf(doc, 'p', 'card-model-pop-line');
                const quote = nodeOf(doc, 'span', 'card-model-cited', said.cited.text);
                quote.dataset.i18nAuthored = '';
                line.append(words, nodeOf(doc, 'span', '', ' “'), quote, nodeOf(doc, 'span', '', '”'));
                title.replaceChildren(line);
                if (said.cited.clipped) title.append(nodeOf(doc, 'span', 'card-model-cited-note',
                    tr('model.cited.excerpt', 'Excerpt; the original quotation is longer.')));
                // The published words stay intact, including whitespace and case. Only CSS clips them.
                const cut = () => line.scrollHeight > line.clientHeight + 1;
                const toggle = clampWithToggle(line, title, {
                    fit: (_node, _text, _lines, onFit) => onFit(cut()),
                    onDomWrite: (write) => {
                        const changed = write();
                        if (line.dataset.expanded !== '1') line.scrollTop = 0;
                        sheet?.reposition();
                        return changed;
                    },
                });
                unwatchHead = observeLayout(doc.defaultView, [line], () => {
                    if (line.dataset.expanded !== '1') toggle.hidden = !cut();
                    sheet?.reposition();
                });
            }
        }
        const last = servedWords(now.processing?.last);
        write(served, last);
        // Reserved (empty, one line high) while the work runs on a named tier: its receipt may arrive with the sheet open.
        served.hidden = !last && !tierNamed(workTier(now));
        const mainName = now.main ? shortName(now.main) : '';
        // A fallback's three exits; any other work is moved from the list alone (its Main row follows Main).
        const offered = said.fallback ? [
            ['return', fmt('Return to {model}', { model: said.leftName || mainName }),
                said.leftMain && mainName ? fmt('{model} · at the next step', { model: mainName }) : tr('model.act.return.sub', 'at the next step')],
            ['wait', fmt('Wait for {model}', { model: said.leftName || mainName }), tr('model.act.wait.sub', 'no paid calls until it answers')],
            ['keep', fmt('Keep on {model}', { model: said.said.short }), tr('model.act.keep.sub', 'your choice from now on')],
        ] : [];
        const key = JSON.stringify(offered);
        if (key === actsKey) return;
        actsKey = key;
        acts.replaceChildren(...offered.map(([action, words, sub]) => {
            const made = nodeOf(doc, 'button', 'card-model-act ui-hit-area');
            made.setAttribute('type', 'button');
            made.dataset.act = action;
            made.dataset.i18nSkip = '';
            made.append(nodeOf(doc, 'span', 'card-model-act-label', words), nodeOf(doc, 'span', 'card-model-act-sub', sub));
            return made;
        }));
        acts.hidden = !offered.length;
    }
    // The strip's words: this work's own tier from its next call (`Applies when it starts.` for a queued one).
    const tierView = () => {
        const now = view();
        const { route, follows, explicit, pending } = next();
        const own = String(now.processing?.choice || '');
        const tier = !pending || typeof now.processing?.resolved === 'string' ? workTier(now)
            : (follows || explicit) ? resolvedTier({ choice: own, follows, tiers: now.tiers || store.tiers?.() }) : own;
        const stated = own || !tier ? tierWord(own || tier) : fmt('Default · {tier}', { tier: tierWord(tier) });
        // The tier this work runs on, its own or Default's (Main's included), reaches its new subagents without one.
        const scope = now.queued ? tr('processing.scope.work_queued_helpers', 'for this work when it starts, and its new subagents without a tier of their own')
            : tr('processing.scope.work_helpers', 'for this work from its next call and its new subagents without a tier of their own; the running call and started subagents keep theirs');
        const account = follows ? store.mainAccount?.(route) ?? null : explicit ? '' : null;
        const index = store.index();
        const here = routeTiers({ index, route, account,
            detail: store.detail?.(route) || null, sourceLabel: describeRoute(index, route).sourceLabel });
        return { choice: own, resolved: tier, busy, scope: `${stated} · ${scope}`, ...here };
    };

    const onActs = (event) => {
        const made = event.target?.closest?.('.card-model-act');
        if (made) run(made.dataset.act);
    };
    const onRetry = () => { if (again) run(again.action, again.route, again.requestId, again.tier); };
    function close({ focus = false } = {}) {
        if (closed) return;
        closed = true;
        acts.removeEventListener('click', onActs);
        retry.removeEventListener('click', onRetry);
        unsubscribe();
        unwatchHead();
        sheet?.close({ focus: false });
        if (openPopover === handle) openPopover = null;
        if (state.pressable) state.fact.setAttribute('aria-expanded', 'false');
        if (focus && state.pressable && state.fact.isConnected) state.fact.focus({ preventScroll: true });
    }
    const handle = {
        record, close,
        get popup() { return sheet?.el || null; },
        // The work's fact changed under the open sheet: its acts and its chosen row follow, or it closes with the control.
        sync() {
            if (closed) return;
            if (!state.view?.target || record.cardModel !== state) { close({ focus: false }); return; }
            readTierDetail();
            paintHead();
            sheet?.paint();
        },
    };
    paintHead();
    sheet = openModelSheet({
        anchor: state.fact, label: 'Model for this work', head, doc, place: deps.place,
        stack: deps.stack || (() => false), coarse: deps.coarse, returnFocus: state.fact,
        list: {
            store, main: () => view().main, follows,
            current: () => next().route,
            recents: () => store.recents().slice(0, RECENT_ROWS),
            mainRow: 'follow',
            tierOf: (_route, { main = false } = {}) => String(view().processing?.choice || '')
                || resolvedTier({ follows: main, tiers: view().tiers || store.tiers?.() || null }),
            // The Main row is the way to follow Main: its facts say what a press does, or that the work already does.
            mainFacts: () => (follows() ? tr('model.row.follows_main', 'this work follows Main') : tr('model.row.will_follow_main', 'this work will follow Main')),
            onRoute: pick,
            onPin: (route, pinned) => store.setPinned(route, pinned).catch((error) => deps.toast(error?.message || 'Could not change the pinned models.', 'error')),
        },
        tier: { onChoose: (tier) => run('processing', null, undefined, tier), view: tierView,
            mainRoute: () => { const { follows, route } = next(); return follows ? route : null; } },
        onClose: () => { if (!closed) close({ focus: false }); },
    });
    void store.read();
    void store.readRecents();
    unsubscribe = store.subscribe(() => { if (!closed) { readTierDetail(); paintHead(); sheet?.paint(); } });
    readTierDetail();
    acts.addEventListener('click', onActs);
    retry.addEventListener('click', onRetry);
    state.fact.setAttribute('aria-expanded', 'true');
    openPopover = handle;
    return handle;
}

/** Closes the open sheet, if any (a room that leaves the screen takes its sheet with it). */
export function closeCardModel() {
    openPopover?.close({ focus: false });
    closeModelSheet();
}
