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
import { createModelList, modelSearchField, routeFacts, SEARCH_ICON } from './model_list.js';
import { haptic } from './haptics.js';
import { isCoarsePointer, isStackLayout, onLayoutChange } from './layout.js';
import { modelCatalog } from './model_catalog_store.js';
import { MAIN_FOLLOW_NOTE } from './model_roles.js';
import { sourceMark } from './model_source_mark.js';
import { bindPopoverPosition } from './ui_interactions.js';
import { escapeHtmlAttr as esc } from './ui_primitives.js';
import {
    MAIN_CHOICE, describeRoute, normalizeChoice, normalizeRoute, routeKey, sameRoute, shortName, sourceUnlisted,
} from './chat_model_routes.js';

// When a pick takes effect, the hint's rule line: for a room, and for Main itself.
export const MODEL_RULE = 'Applies to new work here; running work keeps its model.';
export const MAIN_RULE = 'New work starts on it everywhere, unless a room has its own model.';
const MAIN_KEY = 'main';
let serial = 0;

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
    const search = { on: false, rows: null, list: null, position: null };  // rows: the list of models while Search is on
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
        if (search.on) search.rows.paint();
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

    /* --- Search: the strip becomes a field in the row; its suggestions float above the row (model_list.js) --- */
    function startSearch({ focus = !isCoarse() } = {}) {
        if (search.on || state.destroyed) return;
        search.on = true;
        el.dataset.search = 'true';
        row.dataset.ownerSearch = 'true';
        const { label, input } = modelSearchField(doc, `${id}-list`);
        strip.replaceChildren(label);
        segs.length = 0;
        const list = node('div', 'ui-popup chat-model-list', { role: 'listbox', 'aria-label': 'Models' });
        list.id = `${id}-list`;
        doc.body.append(list);
        search.list = list;
        search.rows = createModelList({
            el: list, input, id, store, main: () => state.main, current: shown, rule: isMain ? MAIN_RULE : MODEL_RULE,
            recents: () => (isMain ? store.recents().filter((route) => !sameRoute(route, state.main)) : quick()),
            // Main's own row follows Main; every other row is that route. A room's closed capsule then says what was set.
            onRoute: (route) => { if (!isMain) control.setOpen(false); pick(sameRoute(route, state.main) ? null : route); },
            onSettings: () => { control.setOpen(false); openSettings(); },
            // Escape with nothing typed leaves the search for the strip; the capsule's own Escape closes it.
            onLeave: endSearch,
            onPaint: () => search.position?.reposition(),
        });
        void store.read();
        void store.readRecents();
        search.rows.paint();
        search.position = placeList(list);
        control.relayout();
        // A finger's press on Search shows the field and the list; the keyboard waits for a press on the field itself.
        if (focus) input.focus({ preventScroll: true });
        onLayout?.();
    }
    function endSearch({ focus = false } = {}) {
        if (!search.on) return;
        search.position?.destroy();
        search.rows?.destroy();
        search.list?.remove();
        Object.assign(search, { on: false, rows: null, list: null, position: null });
        delete el.dataset.search;
        delete row.dataset.ownerSearch;
        segsKey = null;
        if (state.destroyed) return;
        render();
        if (control.isOpen()) control.relayout();
        if (focus) segs.at(-1)?.focus({ preventScroll: true });
    }

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
