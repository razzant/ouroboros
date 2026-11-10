/**
 * The Model capsule (docs/DESIGN.md "Composer owner controls"): the first capsule of the composer's row, before
 * Subagents, Context and Effort, and one of their family. A Project room's capsule holds the model new work in that
 * room starts on and saves it for the room; until the room chooses it follows Main and says so: the name word is
 * `Main`, the value Main's model. In the Main chat the capsule is Main itself, and a pick there asks first (owner
 * 10.11, «1. A», «6. A»). Closed it always shows the room's choice, never the model of a running work (owner 11.11);
 * open, its hint adds one fact line when works here run on other models.
 *
 * It opens inline into its strip on the capsule machine (capsule.js): `Main · the two most recent models · Search`
 * (one recent in the stack layout). Search turns the strip into a field on the composer's line; its suggestions float
 * above the row like the hint and never move the composer (owner 10.11, «9. A»). They list only the sources that can
 * lead a work, one row per (source, model) (chat_model_routes.js); the catalog suggests and never admits, so any id
 * can still be typed. A room's pick saves at once through `POST /api/owner/chat-model`; Send waits for it.
 */
import { apiClient } from './api_client.js';
import { createCapsule } from './capsule.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { fmt, tr, translateString } from './i18n.js';
import { haptic } from './haptics.js';
import { isCoarsePointer, isStackLayout, onLayoutChange } from './layout.js';
import { modelCatalog } from './model_catalog_store.js';
import { MAIN_FOLLOW_NOTE } from './model_roles.js';
import { sourceMark } from './model_source_mark.js';
import { bindPopoverPosition } from './ui_interactions.js';
import { escapeHtmlAttr as esc } from './ui_primitives.js';
import {
    MAIN_CHOICE, describeRoute, normalizeChoice, normalizeRoute, routeKey, sameRoute, searchModels, shortName,
    sourceUnlisted, typedRoute,
} from './chat_model_routes.js';

// When a pick takes effect, the hint's rule line: for a room, and for Main itself.
export const MODEL_RULE = 'Applies to new work here; running work keeps its model.';
export const MAIN_RULE = 'New work starts on it everywhere, unless a room has its own model.';
const MAIN_KEY = 'main';
// How many suggestions one list shows; the rest wait for another letter.
const LIST_ROWS = 60;
let serial = 0;

const SEARCH_ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"'
    + ' stroke-linejoin="round" aria-hidden="true" focusable="false"><circle cx="11" cy="11" r="7"></circle><path d="m20 20-3.5-3.5"></path></svg>';

// Words this module composes into lines the translation overlay leaves alone (they name models and sources): each
// is asked for by its own code, the composed ones by their template. Words standing alone in the page are the overlay's.
const payWords = (said) => tr(`model.pay.${said.lane}`, said.pay);

/** A route's facts in the owner's words: its source, how that source is paid, its window when one is known. */
export function routeFacts(said) {
    return [said.sourceLabel, payWords(said), said.window ? fmt('{window} window', { window: said.window }) : '']
        .filter(Boolean).join(' · ');
}

/**
 * The models of this room's live works that differ from what the capsule shows, by their short names: the hint's one
 * fact line (CONTRACT §5 `works[].lead`). A work without the fact says nothing, so a server that does not publish it
 * yet shows no line.
 */
export function modelsRunningElsewhere(works, chatId, shown) {
    if (!Array.isArray(works) || chatId == null) return [];
    const names = [];
    for (const work of works) {
        const lead = normalizeRoute(work?.lead);
        if (!lead || !work.live || String(work.room || 0) !== String(chatId) || sameRoute(lead, shown)) continue;
        const name = shortName(lead);
        if (!names.includes(name)) names.push(name);
    }
    return names;
}

/**
 * The confirming sheet of a pick in the Main chat, on the shared dialog: what Main becomes and what it was, the new
 * route's facts, what running work does, and in its one sentence what else the change resets. The lists of works and
 * rooms that are not on Main come to its content region with a later change; nothing stands in for them here.
 * Resolves true when the owner confirms.
 */
export function confirmMainChange({ next, previous, dialog = openConfirmDialog, doc = globalThis.document }) {
    const resets = previous?.lane === 'subscription' && next.source !== previous.source
        ? 'If Main\'s account is pinned it becomes Auto, and a context window set by hand for Main is cleared.'
        : 'A context window set by hand for Main is cleared.';
    return dialog({
        title: 'Change Main',
        body: resets,
        confirmLabel: 'Change Main',
        content: { label: 'Main model', mount(container) {
            const row = doc.createElement('p');
            row.className = 'chat-model-sheet-row';
            row.innerHTML = `${sourceMark(next)}<b data-i18n-skip>${esc(`${next.sourceLabel} · ${next.id}`)}</b>`
                + (previous ? `<span data-i18n-skip>${esc(fmt('was {model}', { model: previous.short }))}</span>` : '');
            const facts = doc.createElement('p');
            facts.className = 'chat-model-sheet-fact';
            facts.dataset.i18nSkip = '';
            facts.textContent = `${routeFacts(next)}.`;
            const follow = doc.createElement('p');
            follow.className = 'chat-model-sheet-fact';
            follow.textContent = `${MAIN_RULE} ${MAIN_FOLLOW_NOTE}`;
            container.append(row, facts, follow);
        } },
    });
}

/**
 * The capsule in its room's composer row. `save(body)` is the owner write (`apiClient.ownerChatModel`), `openSettings()`
 * Settings › Accounts, `askMain({next, previous})` the Main chat's confirming sheet, `store` the window's catalog store,
 * `placeList(list)` the suggestions' place above the row.
 * Returns `{ el, capsule, syncState(data), hasPendingSave, pendingSave, pick, startSearch, destroy }`.
 */
export function createModelControl({
    el, row, projectId = '', chatId = projectId ? null : 1, save = apiClient.ownerChatModel, openSettings = () => {},
    showToast = () => {}, refreshState = () => {}, onLayout, onSettled, askMain = confirmMainChange, store = modelCatalog,
    isStack = () => isStackLayout(), isCoarse = () => isCoarsePointer(),
    // Anchored to the row, as the hint is to its capsule: the composer under it never moves.
    placeList = (list) => bindPopoverPosition(list, { anchor: row, align: 'start' }),
    watchLayout = (handler) => (typeof win?.addEventListener === 'function' ? onLayoutChange(handler, { w: win }) : () => {}),
    win = globalThis.window, doc = globalThis.document,
}) {
    const isMain = !projectId;
    const id = `chat-model-${++serial}`;
    el.classList.add('chat-capsule');
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', 'Model');
    el.dataset.known = 'false';
    const node = (tag, className, attrs = {}) => {
        const made = doc.createElement(tag);
        made.className = className;
        for (const [key, text] of Object.entries(attrs)) made.setAttribute(key, text);
        return made;
    };
    // The head's words are written by render(): the mark, whose model this is, and the model.
    el.innerHTML = '<button type="button" class="chat-capsule-head ui-hit-area chat-model-head" aria-expanded="false"><span class="chat-capsule-glyph" aria-hidden="true"></span><span class="chat-capsule-name"></span><span class="chat-capsule-value" data-i18n-skip></span></button><div class="chat-capsule-body"></div>';
    const head = el.querySelector('.chat-model-head');
    const glyph = el.querySelector('.chat-capsule-glyph');
    const name = el.querySelector('.chat-capsule-name');
    const value = el.querySelector('.chat-capsule-value');
    const strip = node('div', 'chat-context-strip chat-model-strip', { role: 'radiogroup', 'aria-label': 'Model' });
    strip.dataset.scroll = 'none';
    el.querySelector('.chat-capsule-body').append(strip);

    // `main` is what Main is now (`/api/state.main_route`), `choice` the room's own (`projects[].model_choice`); the
    // Main chat's capsule is Main, so it has no choice of its own.
    const state = { main: null, choice: { ...MAIN_CHOICE }, works: null, saving: null, lastOk: true, destroyed: false };
    const search = { on: false, query: '', only: '', active: -1, rows: [], list: null, input: null, position: null };
    const segs = [];          // the strip's values as the machine's fit measures them: Main, the recents, Search
    const routes = new Map(); // a segment's key → its route
    let segsKey = null, markKey = null, nameWords = null, control = null;

    const shown = () => (isMain || state.choice.kind === 'main' ? state.main : normalizeRoute(state.choice));
    const follows = () => isMain || state.choice.kind === 'main';
    const describe = (route) => (route ? describeRoute(store.index(), route) : null);
    // The strip's models after Main: the room's own, then the owner's recent ones, none twice. Main's model is the
    // Main segment; only a room that chose that same model for itself keeps it as its own.
    function quick() {
        const own = isMain ? null : normalizeRoute(state.choice);
        const out = own ? [own] : [];
        for (const route of store.recents()) {
            if (!sameRoute(route, state.main) && !out.some((seen) => sameRoute(seen, route))) out.push(route);
        }
        return out.slice(0, isStack() ? 1 : 2);
    }

    /* --- the strip: Main · recents · Search, the chosen one under the family's glass thumb --- */
    function seg(key, words, { model = false } = {}) {
        const made = node('button', 'chat-seg ui-hit-area', { type: 'button', role: 'radio', tabindex: '-1', 'aria-checked': 'false' });
        made.dataset.mode = key;
        const text = node('span', 'chat-seg-text', model ? { 'data-i18n-skip': '' } : {});
        text.textContent = words;
        made.append(text);
        return made;
    }
    function buildStrip() {
        const mainWords = isMain ? (state.main ? shortName(state.main) : 'Main') : 'Main';
        const models = quick();
        const key = JSON.stringify([mainWords, models.map(routeKey)]);
        if (key === segsKey && !search.on && segs.length) return false;
        segsKey = key;
        routes.clear();
        routes.set(MAIN_KEY, state.main);
        const made = [seg(MAIN_KEY, mainWords, { model: isMain && Boolean(state.main) })];
        for (const route of models) {
            routes.set(routeKey(route), route);
            made.push(seg(routeKey(route), shortName(route), { model: true }));
        }
        const find = node('button', 'chat-seg ui-hit-area chat-model-find', { type: 'button', tabindex: '-1',
            'aria-label': 'Search all models' });
        find.innerHTML = `<span class="chat-seg-text">${SEARCH_ICON}<span class="chat-model-find-word">Search</span></span>`;
        const thumb = node('span', 'chat-context-thumb', { 'aria-hidden': 'true' });
        strip.replaceChildren(thumb, ...made, find);
        segs.length = 0;
        segs.push(...made, find);
        return true;
    }
    const chosenKey = () => (follows() ? MAIN_KEY : routeKey(shown()));
    const chosenSeg = () => segs.find((made) => made.dataset.mode === chosenKey()) || segs[0];

    function render() {
        const open = Boolean(control?.isOpen());
        const said = describe(shown());
        el.dataset.known = String(Boolean(said));
        el.dataset.own = String(Boolean(said) && !follows());
        const mark = said ? `${said.lane}|${said.harness}|${said.kind}|${said.sourceLabel}` : '';
        if (mark !== markKey) { markKey = mark; glyph.innerHTML = sourceMark(said); }
        // Open, the name says what the strip sets; closed, whose model this is: Main's, or its own source's.
        const sourceNamed = !open && Boolean(said) && !follows();
        const words = open ? (isMain ? 'Main model' : 'New work here') : sourceNamed ? said.sourceLabel : 'Main';
        if (words !== nameWords) {  // the overlay may have translated the word it was given
            nameWords = words;
            if (sourceNamed) name.dataset.i18nSkip = ''; else delete name.dataset.i18nSkip;  // a source's name is not chrome
            name.textContent = words;
        }
        const model = said ? said.short : '—';
        if (value.textContent !== model) value.textContent = model;
        head.setAttribute('aria-label', !said ? 'Model for new work: not read yet'
            : isMain ? fmt('Main model: {model}, {source}.', { model: said.short, source: said.sourceLabel })
                : follows() ? fmt('Model for new work here: {model}, {source}. Follows Main.', { model: said.short, source: said.sourceLabel })
                    : fmt('Model for new work here: {model}, {source}.', { model: said.short, source: said.sourceLabel }));
        if (!search.on) {
            const rebuilt = buildStrip();
            const key = chosenKey();
            const stop = chosenSeg();
            for (const made of segs) {
                if (made.dataset.mode !== undefined) made.setAttribute('aria-checked', String(made.dataset.mode === key));
                made.tabIndex = open && made === stop ? 0 : -1;  // one tab stop; arrows move within
            }
            const on = segs.find((made) => made.dataset.mode === key);
            strip.style.setProperty('--thumb-left', `${Number(on?.offsetLeft) || 0}px`);
            strip.style.setProperty('--thumb-width', `${Number(on?.offsetWidth) || 0}px`);
            if (rebuilt && open) control.relayout();
        }
        // A saved API source this catalog read did not list: the settings document says why it cannot send.
        if (said && store.status().read && !store.status().failed && sourceUnlisted(store.index(), shown())) void store.readConnections();
        control?.sayHint();
        if (search.on) paintList();
    }

    // The chosen route, or the one a mouse rests on; under it, the models this room's works run on when they differ.
    const say = (preview) => {
        const rule = isMain ? MAIN_RULE : MODEL_RULE;
        const previewed = preview && routes.has(preview);
        const said = describe(previewed ? routes.get(preview) : shown());
        if (!said) return { text: 'Main\'s model has not been read yet.', rule };
        const ofMain = previewed ? preview === MAIN_KEY : follows();
        const elsewhere = modelsRunningElsewhere(state.works, chatId, shown());
        return {
            word: ofMain ? translateString('Main') : said.short,
            text: `${ofMain ? `${said.short} · ` : ''}${routeFacts(said)}.`,
            fact: elsewhere.length ? { word: tr('model.running_here', 'Running here'),
                text: fmt('on other models: {models}', { models: elsewhere.join(', ') }) } : null,
            rule,
        };
    };

    /* --- saving: a room's pick posts at once; Main's asks first, then writes Main itself --- */
    function write(body, accept) {
        if (state.saving || state.destroyed) return state.saving;
        el.dataset.disabled = 'true';
        state.saving = (async () => {
            try {
                const reply = await save(projectId ? { ...body, project_id: projectId } : body);
                if (state.destroyed) return false;
                if ((reply?.project_id || '') !== projectId) throw new Error('The answer belongs to another room.');
                if (reply?.ok !== true) throw new Error('The server did not confirm the model.');
                accept(reply);
                state.lastOk = true;
            } catch (error) {
                if (state.destroyed) return false;
                state.lastOk = false;
                showToast(error?.message || 'Could not change the model.', 'error');
            } finally {
                state.saving = null;
                if (!state.destroyed) {
                    el.dataset.disabled = 'false';
                    render();
                    refreshState(true);
                    void store.readRecents();  // the server wrote this act into the recent models
                }
            }
            return state.lastOk;
        })();
        return state.saving;
    }
    function saveRoom(choice) {
        return write(choice, (reply) => {
            const saved = reply.model_choice;
            if (saved?.kind !== 'main' && !normalizeRoute(saved)) throw new Error('The server did not confirm the model.');
            state.choice = normalizeChoice(saved);
        });
    }
    async function changeMain(route) {
        const next = describe(route);
        if (!(await askMain({ next, previous: describe(state.main), doc })) || state.destroyed) return false;
        return write({ kind: 'model', model: route.model, use_local: route.use_local }, (reply) => {
            const written = normalizeRoute(reply.main_route);
            if (!written) throw new Error('The server did not confirm the model.');
            state.main = written;
            // What else the write changed, from its own answer: a dropped pin and a cleared manual window.
            const also = [reply.account === 'auto' ? 'Main\'s account is Auto now.' : '',
                reply.window_reset === true ? 'Main\'s context window set by hand was cleared.' : ''].filter(Boolean);
            if (also.length) showToast(also.join(' '), 'info');
        });
    }
    /** One pick, from the strip or the suggestions: null follows Main again; a route becomes the room's own. */
    function pick(route) {
        if (state.saving || state.destroyed) return undefined;
        if (isMain) {
            if (!route || sameRoute(route, state.main)) return undefined;
            control.setOpen(false);
            return changeMain(route);
        }
        if (route ? sameRoute(route, normalizeRoute(state.choice)) : state.choice.kind === 'main') return undefined;
        haptic('tick');  // another model, chosen by a press (haptics.js)
        return saveRoom(route ? { kind: 'model', model: route.model, use_local: route.use_local } : { kind: 'main' });
    }

    /* --- Search: the strip becomes a field in the row; its suggestions float above the row --- */
    function startSearch({ focus = !isCoarse() } = {}) {
        if (search.on || state.destroyed) return;
        Object.assign(search, { on: true, query: '', only: '', active: -1 });
        el.dataset.search = 'true';
        row.dataset.ownerSearch = 'true';
        const label = node('label', 'chat-model-search');
        label.innerHTML = SEARCH_ICON;
        const input = node('input', 'chat-model-query', { type: 'text', role: 'combobox', autocomplete: 'off',
            autocorrect: 'off', autocapitalize: 'off', spellcheck: 'false', 'aria-autocomplete': 'list', 'aria-expanded': 'true',
            'aria-controls': `${id}-list`, 'aria-label': 'Search models' });
        input.dataset.i18nSkip = '';  // its placeholder may name a source: written through the catalog below
        label.append(input);
        strip.replaceChildren(label);
        segs.length = 0;
        const list = node('div', 'ui-popup chat-model-list', { role: 'listbox', 'aria-label': 'Models' });
        list.id = `${id}-list`;
        doc.body.append(list);
        Object.assign(search, { input, list });
        input.addEventListener('input', onQuery);
        input.addEventListener('keydown', onSearchKey);
        list.addEventListener('mousedown', keepFocus);
        list.addEventListener('click', onListClick);
        void store.read();
        void store.readRecents();
        paintList();
        search.position = placeList(list);
        control.relayout();
        // A finger's press on Search shows the field and the list; the keyboard waits for a press on the field itself.
        if (focus) input.focus({ preventScroll: true });
        onLayout?.();
    }
    function endSearch({ focus = false } = {}) {
        if (!search.on) return;
        search.position?.destroy();
        search.list?.remove();
        Object.assign(search, { on: false, query: '', only: '', active: -1, rows: [], list: null, input: null, position: null });
        delete el.dataset.search;
        delete row.dataset.ownerSearch;
        segsKey = null;
        if (state.destroyed) return;
        render();
        if (control.isOpen()) control.relayout();
        if (focus) segs.at(-1)?.focus({ preventScroll: true });
    }
    const keepFocus = (event) => event.preventDefault();  // a press on a suggestion leaves the caret in the field

    const rowMarkup = (said, { tag = '', selected = false, extra = '' } = {}) => `<div class="chat-model-row" role="option"`
        + ` data-row="${esc(said.key)}" aria-selected="${selected}">${sourceMark(said)}`
        + `<span class="chat-model-row-id" data-i18n-skip>${esc(said.id)}</span>`
        + (tag ? `<span class="chat-model-row-tag">${esc(tag)}</span>` : '')
        + `<span class="chat-model-row-facts" data-i18n-skip>${esc([said.window, said.sourceLabel, payWords(said)].filter(Boolean).join(' · '))}</span>${extra}</div>`;

    function paintList() {
        const { list, input } = search;
        if (!list) return;
        const index = store.index();
        const status = store.status();
        const current = shown();
        const picks = new Map();  // a row's key → what a press on it does
        const parts = [];
        const add = (route, options = {}) => {
            const said = describe(route);
            picks.set(said.key, { route });
            const unavailable = said.available === false ? '<span class="chat-model-row-tag">Unavailable</span>' : '';
            parts.push(rowMarkup(said, { selected: sameRoute(route, current), ...options, extra: unavailable }));
        };
        const note = (words) => parts.push(`<p class="chat-model-note" role="presentation">${esc(words)}</p>`);
        const section = (words) => parts.push(`<p class="chat-model-section" role="presentation">${esc(words)}</p>`);
        const others = index.rows.filter((entry) => !sameRoute(entry.route, state.main));
        if (!search.query && !search.only) {
            if (state.main) add(state.main, { tag: 'Main' });
            const recent = isMain ? store.recents().filter((route) => !sameRoute(route, state.main)) : quick();
            if (recent.length) { section('Recent'); recent.forEach((route) => add(route)); }
            if (status.reading && !status.read) note('Reading models…');
            else if (!others.length && !index.sources.some((entry) => entry.problems.length)) {
                // Nothing but Main can lead: say so and offer the way to where a source is added.
                note(status.failed ? 'The model catalog could not be read.' : 'Only Main\'s model can lead work here.');
                picks.set('accounts', { settings: true });
                parts.push(`<div class="chat-model-row chat-model-door" role="option" data-row="accounts" aria-selected="false">`
                    + '<span class="chat-model-row-id">Add a subscription or a key in Settings › Accounts</span></div>');
            } else if (index.sources.length === 1) {
                // One source: its models at once, no source step.
                listRows(searchModels(index, '', index.sources[0].id), add, section);
            } else {
                section('Sources');
                for (const entry of index.sources) {
                    picks.set(`source:${entry.id}`, { only: entry.id });
                    const facts = [entry.count === 1 ? fmt('{n} model', { n: 1 }) : entry.count ? fmt('{n} models', { n: entry.count }) : '', payWords(entry),
                        entry.problems.length ? tr('model.list_not_read', 'list not read') : ''].filter(Boolean).join(' · ');
                    parts.push(`<div class="chat-model-row chat-model-source" role="option" data-row="${esc(`source:${entry.id}`)}" aria-selected="false">`
                        + `${sourceMark(entry)}<span class="chat-model-row-id" data-i18n-skip>${esc(entry.label)}</span>`
                        + `<span class="chat-model-row-facts" data-i18n-skip>${esc(facts)}</span></div>`);
                }
            }
        } else {
            if (search.only && index.sources.length > 1) {
                picks.set('source:', { only: '' });
                parts.push(`<div class="chat-model-row chat-model-source" role="option" data-row="source:" aria-selected="false">`
                    + `${SEARCH_ICON}<span class="chat-model-row-id">All sources</span>`
                    + `<span class="chat-model-row-facts" data-i18n-skip>${esc(index.source(search.only)?.label || '')}</span></div>`);
            }
            const found = searchModels(index, search.query, search.only);
            listRows(found, add, section);
            // Suggestions assist typing and are not an allowlist: what was typed can be used as it is.
            const typed = search.query ? typedRoute(index, search.query, search.only) : null;
            if (typed && !found.some((entry) => sameRoute(entry.route, typed))) {
                const said = describe(typed);
                picks.set(`typed:${said.key}`, { route: typed });
                parts.push(`<div class="chat-model-row chat-model-typed" role="option" data-row="${esc(`typed:${said.key}`)}" aria-selected="false">`
                    + `${sourceMark(said)}<span class="chat-model-row-id" data-i18n-skip>${esc(fmt('Use “{model}” as typed', { model: search.query }))}</span>`
                    + `<span class="chat-model-row-facts" data-i18n-skip>${esc(said.sourceLabel)}</span></div>`);
            } else if (search.query && !typed && !found.length) {
                note('No model matches. Choose a source to use a model as typed.');
            }
            const gaps = (search.only ? [index.source(search.only)] : index.sources).filter((entry) => entry?.problems.length);
            if (status.failed) note('The model catalog could not be read; earlier suggestions are kept.');
            else if (gaps.length) note(fmt('Could not read every list: {sources}.', { sources: gaps.map((entry) => entry.label).join(', ') }));
        }
        list.innerHTML = `<div class="chat-model-rows">${parts.join('')}</div>`
            + `<p class="chat-model-rule" role="presentation">${esc(isMain ? MAIN_RULE : MODEL_RULE)}</p>`;
        search.rows = [...list.querySelectorAll('.chat-model-row')];
        search.rows.forEach((made, at) => { made.id = `${id}-o${at}`; });
        search.picks = picks;
        setActive(Math.min(search.active, search.rows.length - 1));
        if (input) {
            input.placeholder = search.only ? fmt('{source}: type a model', { source: index.source(search.only)?.label || '' })
                : tr('model.search.placeholder', 'Model or source');
        }
        search.position?.reposition();
    }
    function listRows(found, add, section) {
        found.slice(0, LIST_ROWS).forEach((entry) => add(entry.route, { tag: sameRoute(entry.route, state.main) ? 'Main' : '' }));
        if (found.length > LIST_ROWS) section(fmt('{n} more · keep typing', { n: found.length - LIST_ROWS }));
    }
    function setActive(at) {
        search.active = at;
        search.rows.forEach((made, order) => { made.dataset.active = String(order === at); });
        const active = search.rows[at];
        if (active) search.input?.setAttribute('aria-activedescendant', active.id);
        else search.input?.removeAttribute('aria-activedescendant');
        active?.scrollIntoView?.({ block: 'nearest' });
    }
    function take(made) {
        const act = search.picks?.get(made?.dataset?.row);
        if (!act) return;
        if (act.settings) { control.setOpen(false); openSettings(); return; }
        if (act.only !== undefined) {
            Object.assign(search, { only: act.only, query: '', active: -1 });
            if (search.input) { search.input.value = ''; search.input.focus({ preventScroll: true }); }
            paintList();
            return;
        }
        if (!isMain) control.setOpen(false);  // the closed capsule says what was set
        // Main's own row follows Main; every other row is that route.
        pick(sameRoute(act.route, state.main) ? null : act.route);
    }
    const onQuery = () => {
        search.query = search.input.value.trim();
        search.active = search.query ? 0 : -1;
        paintList();
    };
    const onListClick = (event) => take(event.target?.closest?.('.chat-model-row'));
    const onSearchKey = (event) => {
        if (event.isComposing) return;
        const count = search.rows.length;
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            if (count) setActive(event.key === 'ArrowDown' ? (search.active + 1) % count : search.active <= 0 ? count - 1 : search.active - 1);
        } else if (event.key === 'Enter') {
            // The active row, else the first model when something was typed or a source chosen.
            const first = search.query || search.only ? search.rows.find((made) => !made.classList.contains('chat-model-source')) : null;
            const target = search.rows[search.active] || first;
            if (target) take(target);
        } else if (event.key === 'Escape') {
            // Escape clears what was typed, then leaves the search for the strip; the capsule's own Escape closes it.
            if (search.query || search.only) {
                Object.assign(search, { query: '', only: '', active: -1 });
                search.input.value = '';
                paintList();
            } else endSearch({ focus: true });
        } else if (event.key === 'Tab') {
            endSearch();
            return;
        } else return;
        event.preventDefault();
        event.stopPropagation?.();
    };

    /* --- the strip's presses and keys --- */
    const onStripClick = (event) => {
        const made = event.target?.closest?.('.chat-seg');
        if (!made || search.on) return undefined;
        if (made.classList.contains('chat-model-find')) { startSearch(); return undefined; }
        return pick(made.dataset.mode === MAIN_KEY ? null : routes.get(made.dataset.mode));
    };
    // Arrows, Home and End move among the values; a model is chosen by its own press, never by passing over it.
    const onStripKey = (event) => {
        const at = segs.indexOf(event.target?.closest?.('.chat-seg'));
        if (at < 0) return;
        const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
        const to = step ? Math.min(segs.length - 1, Math.max(0, at + step))
            : event.key === 'Home' ? 0 : event.key === 'End' ? segs.length - 1 : -1;
        if (to < 0) return;
        event.preventDefault?.();
        segs[to].focus({ preventScroll: true });
        control.reveal(Number(segs[to].offsetLeft), Number(segs[to].offsetLeft) + Number(segs[to].offsetWidth));
    };
    const revealChosen = () => {
        const made = chosenSeg();
        if (made) control.reveal(Number(made.offsetLeft), Number(made.offsetLeft) + Number(made.offsetWidth));
    };

    control = createCapsule({
        el, row, kind: 'model', strip, segs, render, focusTarget: chosenSeg, say, onLayout, onSettled, win, doc,
        // The suggestions stand outside the capsule's element and are its own; a search in progress holds the strip.
        parts: () => [search.list], busy: () => search.on,
        onClosed: () => endSearch(),
        afterOpen: () => {
            void store.read();
            void store.readRecents();
            control.relayout();  // the open name differs from the closed one: the fit measures it as it stands
            revealChosen();
        },
        afterOverflow: revealChosen,
    });
    // The hint's first line and its fact name models: authored words, not chrome.
    for (const selector of ['.ui-hint-word', '.ui-hint-text']) {
        for (const part of el.querySelectorAll(selector)) part.dataset.i18nSkip = '';
    }
    strip.addEventListener('click', onStripClick);
    strip.addEventListener('keydown', onStripKey);
    const unsubscribe = store.subscribe(() => { if (!state.destroyed) render(); });
    const unwatchLayout = watchLayout(() => { if (!state.destroyed && !search.on) render(); });
    render();

    return {
        el,
        capsule: control,
        /** One `/api/state` body (or a room's own read): what Main is now, this room's choice, and its works' models. A
         *  save in flight answers for itself; a body without a field keeps what was last read. */
        syncState(data) {
            if (state.destroyed || !data) return;
            if (Array.isArray(data.works)) state.works = data.works;
            if (!state.saving) {
                const main = normalizeRoute(data.main_route);
                if (main) state.main = main;
                const room = projectId ? data.projects?.find?.((project) => project?.id === projectId) : null;
                if (room) state.choice = normalizeChoice(room.model_choice);
            }
            render();
        },
        hasPendingSave: () => Boolean(state.saving),
        /** Resolves once the save settled: true when it was accepted. */
        pendingSave: () => state.saving || Promise.resolve(state.lastOk),
        /** The capsule's one save path: a route, or null to follow Main again. */
        pick,
        startSearch,
        destroy() {
            state.destroyed = true;
            endSearch();
            unsubscribe();
            unwatchLayout?.();
            control.destroy();
            strip.removeEventListener('click', onStripClick);
            strip.removeEventListener('keydown', onStripKey);
        },
    };
}
