/**
 * A work's model on its card (docs/DESIGN.md "A work's model on its card"): the first fact of the card's meta line,
 * said from the server's published fact (CLIENT_CONTRACT §5: `works[].lead`, and `active_chat_activities[].lead` for a
 * direct turn). The line itself is rebuilt as one string by chat_activity.js, so the fact stands in a node of its
 * own before it: the rest of the line moves into a wrapper the card's record then writes to, a repaint of the line
 * never touches the fact, and an open popover keeps its anchor. A card whose work publishes no such fact (an older
 * record, a child, a portal) keeps the line it has today.
 *
 * Where the server offers the work's `model` control the fact is a button. It opens the list of models the
 * composer's Model capsule uses (model_list.js): its Main row is the one way to follow Main, as `Main` is in the
 * capsule's strip, and its facts say so; while the work runs on a fallback its three exits stand above the list:
 * Return, Wait and Keep. Each press is one `POST /api/tasks/{id}/model` (CLIENT_CONTRACT §5) whose answer says when it
 * applies; a refusal is said in the popover in the owner's words, and one whose decision the server saved (a 503)
 * offers the same request again. Without the control the fact is plain text: a control is the server's decision and
 * is never guessed here.
 */
import { taskModel } from './api_client.js';
import { renderLiveCardMeta } from './chat_activity.js';
import { clampWithToggle } from './chat_reply_quote.js';
import { observeLayout } from './capsule.js';
import { fmt, tr, translateString } from './i18n.js';
import { isCoarsePointer } from './layout.js';
import { modelCatalog } from './model_catalog_store.js';
import { createModelList, modelSearchField } from './model_list.js';
import { sourceMark } from './model_source_mark.js';
import { modelWaitReasonLabel, modelWaitRoleLabel } from './model_wait.js';
import { controlIcon } from './page_icons.js';
import { showToast } from './toast.js';
import { bindPopoverPosition } from './ui_interactions.js';
import { describeRoute, normalizeRoute, sameRoute, shortName } from './chat_model_routes.js';

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
    not_a_root_task: 'A helper runs on the model its work gave it; change the model on the work\'s own card.',
    model_control_unavailable: 'This work has no saved record yet, so its model cannot be changed.',
    cancel_pending: 'This work is stopping; its model stays as it is.',
    model_action_state: 'This work is not running on a fallback now.',
    model_recents_pending: 'Your choice is saved, but the list of recent models could not be updated yet.',
    model_queue_publish_failed: 'Your choice is saved, but the queue could not be updated yet.',
    model_wake_failed: 'Your choice is saved, but the work could not be told yet.',
});
// How many recent models the popover offers under Main.
const RECENT_ROWS = 3;
let serial = 0;
let openPopover = null;  // one popover in the window: opening another closes it

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
    const awaitingPrimary = Boolean(primary && pending && sameRoute(primary, pending) && !sameRoute(pending, said.route));
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
    const sep = nodeOf(doc, 'span', 'chat-live-meta-sep');
    sep.innerHTML = '&nbsp;· ';
    // The rest of the line, as chat_activity.js writes it: its parts stay items of the line itself (style.css).
    const rest = nodeOf(doc, 'span', 'card-model-rest');
    rest.append(...[...meta.children]);
    meta.append(part, rest);
    const state = { meta, part, fact: null, factGroup: null, mark: null, name: null, why, whyWords, pending,
        pendingWords, sep, rest, doc, deps, view: null, input: null, markKey: null, pressable: null, watch: null };
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
    const last = !state.pending.hidden ? state.pending : !state.why.hidden ? state.why : state.factGroup;
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
    state.part.replaceChildren(factGroup, state.why, state.pending);
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
    state.deps.rerender(record);  // the line names the model itself again
    return true;
}

/**
 * Paints one card's model fact. `view` is `{ lead, target, main, queued, since }`: the published fact, the task id
 * the server's `model` control acts on ('' when it offers none), Main's route, whether the work has not started,
 * and the fact's `since` in the viewer's clock; null (or a fact without a model) takes the fact away. Returns
 * whether a fact stands on the card.
 */
export function paintCardModel(record, view, deps = {}) {
    if (!record?.metaEl) return false;
    const lead = normalizeRoute(view?.lead) ? view.lead : null;
    if (!lead) { removeFact(record); return false; }
    const fresh = !record.cardModel;
    const state = record.cardModel || buildFact(record, { store: modelCatalog, act: taskModel, toast: showToast,
        rerender: renderLiveCardMeta, coarse: isCoarsePointer,
        place: (popup, anchor) => bindPopoverPosition(popup, { anchor, align: 'start' }), ...deps });
    const said = leadView(lead, { index: state.deps.store.index(), since: view.since || '', queued: Boolean(view.queued) });
    state.view = { ...view, lead, target: String(view.target || ''), main: normalizeRoute(view.main), said };
    record.lead = lead;
    shapeFact(record, state, Boolean(state.view.target));
    const mark = `${said.said.lane}|${said.said.harness}|${said.said.kind}|${said.said.sourceLabel}`;
    if (mark !== state.markKey) { state.markKey = mark; state.mark.innerHTML = sourceMark(said.said); }
    const [leads, id, tail] = state.name.children;
    // A separator rides at the end of the span before it, so a wrapped line never starts with one.
    write(leads, `${said.parts[0]}\u00a0·`);
    write(id, `\u00a0${said.parts[1]}${said.parts[2] ? '\u00a0·\u00a0' : ''}`);
    write(tail, said.parts[2]);
    write(state.whyWords, said.why);
    write(state.pendingWords, said.pending);
    state.why.hidden = !said.why;
    state.pending.hidden = !said.pending;
    // Whose model this is rides the node as its look: the default is quiet, a choice stronger, a fallback amber.
    state.part.dataset.basis = said.fallback ? 'fallback' : said.basis === 'main' ? 'main' : 'own';
    state.fact.classList.toggle('model-fallback', said.fallback);
    if (state.view.target) {
        state.fact.setAttribute('aria-label', [fmt('This work\'s model: {model}, {source}', { model: said.said.short, source: said.said.sourceLabel }),
            said.fallback ? said.name : basisWords(said.basis)].filter(Boolean).join(', '));
    }
    syncSeparator(state);
    if (fresh) state.deps.rerender(record);  // the fact names the model now: the line no longer does
    if (openPopover?.record === record) openPopover.sync();
    return true;
}

/* --- the popover: this work's acts, then the capsule's list of models --- */

/** Opens the model popover of a card whose fact is pressable; null when the server offers no control for it. */
export function openCardModel(record) {
    const state = record?.cardModel;
    if (!state?.view?.target) return null;
    openPopover?.close({ focus: false });
    const { doc, deps } = state;
    const { store } = deps;
    const id = `card-model-${++serial}`;
    const popup = nodeOf(doc, 'div', 'ui-popup card-model-pop');
    popup.setAttribute('role', 'dialog');
    popup.setAttribute('aria-label', 'Model for this work');
    const title = nodeOf(doc, 'div', 'card-model-pop-title');
    title.dataset.i18nSkip = '';
    const acts = nodeOf(doc, 'div', 'card-model-acts');
    // A refused act, said where it was pressed; `Retry` stands beside it only when the server saved the decision.
    const refusal = nodeOf(doc, 'p', 'card-model-refusal');
    refusal.setAttribute('role', 'status');
    refusal.dataset.i18nSkip = '';
    refusal.hidden = true;
    const refusalText = nodeOf(doc, 'span', 'card-model-refusal-words');
    const retry = nodeOf(doc, 'button', 'btn btn-sm btn-default card-model-retry', 'Retry');
    retry.setAttribute('type', 'button');
    refusal.append(refusalText, retry);
    const { label, input } = modelSearchField(doc, `${id}-list`);
    const list = nodeOf(doc, 'div', 'card-model-list');
    list.id = `${id}-list`;
    list.setAttribute('role', 'listbox');
    list.setAttribute('aria-label', 'Models');
    popup.append(title, refusal, acts, label, list);
    let busy = false;
    let closed = false;
    let actsKey = null;
    let headKey = null;
    let unwatchHead = () => {};
    let position = null;
    let again = null;  // the refused request a 503 may send again, under its own id
    let unsubscribe = () => {};
    const view = () => state.view;

    async function run(action, route = null, requestId = globalThis.crypto?.randomUUID?.() || `card-model-${Date.now()}-${++serial}`) {
        if (busy || closed) return;
        busy = true;
        again = null;
        refusal.hidden = true;
        popup.setAttribute('aria-busy', 'true');
        const before = view();
        try {
            const reply = await deps.act(before.target, requestId,
                { action, ...(route ? { model: route.model, use_local: route.use_local } : {}) });
            const kept = before.said.said.short;
            deps.toast(action === 'wait' ? fmt('Waits for {model}: no paid calls until it answers.', { model: before.said.leftName || kept })
                : action === 'keep' ? fmt('Stays on {model}: your choice from now on.', { model: kept })
                    : APPLIED_WORDS[reply?.applied] || 'The model was changed.', 'ok');
            // The answer's own fact stands on the card until the next state body says it again.
            if (record.cardModel === state && normalizeRoute(reply?.lead)) {
                paintCardModel(record, { ...before, lead: reply.lead, since: reply.lead.since === before.lead.since ? before.since : '' });
            }
            void store.readRecents();  // the server wrote this act into the recent models
            close({ focus: true });
        } catch (error) {
            const said = refusalWords(error);
            if (closed) deps.toast(said.words, 'error');  // its popover went while the answer was on its way
            else {
                again = said.retry ? { action, route, requestId } : null;
                write(refusalText, said.words);
                retry.hidden = !again;
                refusal.hidden = false;
                position?.reposition();
            }
        } finally {
            busy = false;
            if (!closed) popup.removeAttribute('aria-busy');
        }
    }
    // A route Main is on follows Main (as its row in the capsule does); any other becomes this work's own.
    const pick = (route) => {
        const now = view();
        const follows = now.said.basis === 'main' && !now.lead.pending;
        if (sameRoute(route, now.main)) return follows ? close({ focus: true }) : run('follow');
        return run('set', route);
    };

    function paintHead() {
        const now = view();
        const said = now.said;
        let heading = said.fallback
            ? [said.awaitingPrimary ? `${said.name} · ${said.pending}`
                : said.leftName ? fmt(said.since ? 'Left {model} at {time}' : 'Left {model}', { model: said.leftName, time: said.since })
                    : said.name, said.cause].filter(Boolean).join(' · ')
            : [tr('model.popover.title', 'Model for this work'), said.why || basisWords(said.basis),
                said.awaitingPrimary ? said.pending : ''].filter(Boolean).join(' · ');
        if (said.cited && said.primary) heading += ` · ${basisWords('owner_words')}`;
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
                        position?.reposition();
                        return changed;
                    },
                });
                unwatchHead = observeLayout(doc.defaultView, [line], () => {
                    if (line.dataset.expanded !== '1') toggle.hidden = !cut();
                    position?.reposition();
                });
            }
        }
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
    const rows = createModelList({
        el: list, input, id, store, main: () => view().main,
        // The row that is this work's now: Main's while it follows Main, else its own.
        current: () => (view().said.basis === 'main' ? view().main : view().said.said.route),
        recents: () => store.recents().filter((route) => !sameRoute(route, view().main)).slice(0, RECENT_ROWS),
        // The Main row is the way to follow Main: its facts say what a press does, or that the work already does.
        mainFacts: () => (view().said.basis === 'main' && !view().lead.pending
            ? tr('model.row.follows_main', 'this work follows Main') : tr('model.row.will_follow_main', 'this work will follow Main')),
        rule: APPLIED_WORDS[view().queued ? 'at_start' : 'next_step'],
        onRoute: pick,
        onLeave: ({ focus }) => { if (focus) close({ focus: true }); },
        onPaint: () => position?.reposition(),
    });

    const onActs = (event) => {
        const made = event.target?.closest?.('.card-model-act');
        if (made) run(made.dataset.act);
    };
    const onRetry = () => { if (again) run(again.action, again.route, again.requestId); };
    const onKey = (event) => {
        if (event.key !== 'Escape' || event.defaultPrevented) return;
        event.preventDefault?.();
        event.stopPropagation?.();
        close({ focus: true });
    };
    // Escape closes it from wherever focus is: under a finger nothing in the popover holds it. Inside the popover
    // the field's own Escape comes first (it clears what was typed).
    const onPageKey = (event) => { if (!popup.contains(event.target)) onKey(event); };
    const onOutside = (event) => {
        if (!popup.contains(event.target) && !state.fact?.contains(event.target)) close({ focus: false });
    };
    // Focus that leaves the popover for the page closes it; a press inside that lands on nothing focusable does not.
    const onFocusOut = (event) => {
        const to = event.relatedTarget;
        if (to && !popup.contains(to) && !state.fact?.contains(to)) close({ focus: false });
    };
    function close({ focus = false } = {}) {
        if (closed) return;
        closed = true;
        doc.removeEventListener('pointerdown', onOutside, true);
        doc.removeEventListener('keydown', onPageKey, true);
        popup.removeEventListener('keydown', onKey);
        popup.removeEventListener('focusout', onFocusOut);
        acts.removeEventListener('click', onActs);
        retry.removeEventListener('click', onRetry);
        unsubscribe();
        unwatchHead();
        position?.destroy();
        rows.destroy();
        popup.remove();
        if (openPopover === handle) openPopover = null;
        if (state.pressable) state.fact.setAttribute('aria-expanded', 'false');
        if (focus && state.pressable && state.fact.isConnected) state.fact.focus({ preventScroll: true });
    }
    const handle = {
        record, popup, close,
        // The work's fact changed under the open popover: its acts and its chosen row follow, or it closes with the control.
        sync() {
            if (closed) return;
            if (!state.view?.target || record.cardModel !== state) { close({ focus: false }); return; }
            paintHead();
            rows.paint();
        },
    };
    doc.body.append(popup);
    paintHead();
    void store.read();
    void store.readRecents();
    rows.paint();
    position = deps.place(popup, state.fact);
    unsubscribe = store.subscribe(() => { if (!closed) { paintHead(); rows.paint(); } });
    doc.addEventListener('pointerdown', onOutside, true);
    doc.addEventListener('keydown', onPageKey, true);
    popup.addEventListener('keydown', onKey);
    popup.addEventListener('focusout', onFocusOut);
    acts.addEventListener('click', onActs);
    retry.addEventListener('click', onRetry);
    state.fact.setAttribute('aria-expanded', 'true');
    openPopover = handle;
    // A finger's press shows the acts and the list; the keyboard waits for a press on the field itself.
    if (!deps.coarse()) input.focus({ preventScroll: true });
    return handle;
}

/** Closes the open popover, if any (a room that leaves the screen takes its popover with it). */
export function closeCardModel() {
    openPopover?.close({ focus: false });
}
