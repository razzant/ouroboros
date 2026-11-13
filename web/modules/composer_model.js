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
import { apiClient, taskModel } from './api_client.js';
import { createCapsule } from './capsule.js';
import { APPLIED_WORDS, basisWords, leadView, refusalWords } from './card_model.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { fmt, fmtInto, tr, translateString } from './i18n.js';
import { createModelList, modelSearchField, routeFacts, SEARCH_ICON } from './model_list.js';
import { haptic } from './haptics.js';
import { isCoarsePointer, isStackLayout, onLayoutChange } from './layout.js';
import { modelCatalog } from './model_catalog_store.js';
import { MAIN_FOLLOW_NOTE } from './model_roles.js';
import { sourceMark } from './model_source_mark.js';
import { bindPopoverPosition } from './ui_interactions.js';
import { escapeHtmlAttr as esc } from './ui_primitives.js';
import { clock, workControls } from './work_status.js';
import {
    MAIN_CHOICE, describeRoute, normalizeChoice, normalizeRoute, routeKey, sameRoute, shortName, sourceUnlisted,
} from './chat_model_routes.js';

// When a pick takes effect, the hint's rule line: for a room, and for Main itself.
export const MODEL_RULE = 'Applies to new work here; running work keeps its model.';
export const MAIN_RULE = 'New work starts on it everywhere, unless a room has its own model.';
const MAIN_KEY = 'main';
// How long Main's sheet waits for a state read taken for it before it lists what the page already has.
const FRESH_STATE_MS = 700;
let serial = 0;

/**
 * The models of this room's live works that differ from what the capsule shows, by their short names: the hint's one
 * fact line (CLIENT_CONTRACT §5 `works[].lead`). A work without the fact says nothing, so a server that does not publish it
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
 * What a change of Main does not move, from the state the page already has (CLIENT_CONTRACT §5 `works[].lead`, §9
 * `projects[].model_choice`): how many running works follow Main, the works that are not on Main, and the rooms that
 * start new work on their own model. `followers` is null when the census is incomplete: an unknown count is never
 * said as zero. A work on a fallback is one that is not on Main, whatever it left, and never one that will switch.
 * A row's `id` is the task or the room its tick acts on; a work the server offers no model control for has none.
 */
export function mainSheetLists({ works = null, complete = false, projects = [], index = null } = {}) {
    const rooms = (Array.isArray(projects) ? projects : []).filter((project) => project?.id);
    const roomOf = (work) => (work.project_id ? rooms.find((project) => project.id === work.project_id)?.name || work.project_id
        : tr('model.sheet.main_chat', 'Main Chat'));
    const off = [];
    let followers = 0;
    for (const work of Array.isArray(works) ? works : []) {
        if (!work || work.confirmed === false) continue;
        const said = leadView(work.lead, { index, since: clock(work.lead?.since) });
        if (!said) continue;
        if (said.basis === 'main') { if (work.activity !== 'queued') followers += 1; continue; }
        off.push({ kind: 'work', id: workControls(work).model, room: roomOf(work),
            name: String(work.name || '') || tr('model.sheet.unnamed', 'Unnamed work'),
            model: `${said.said.sourceLabel} · ${said.said.short}`, fallback: said.fallback,
            why: said.fallback ? (said.since ? fmt('fallback since {time}', { time: said.since }) : basisWords('fallback'))
                : said.basis === 'room' ? tr('model.basis.room.elsewhere', 'its room\'s model') : basisWords(said.basis) });
    }
    const own = rooms.filter((project) => normalizeChoice(project.model_choice).kind === 'model' && !project.closed
        && !['deleting', 'tombstoned'].includes(project.lifecycle)).map((project) => {
        const said = describeRoute(index, project.model_choice);
        return { kind: 'room', id: project.id, name: String(project.name || project.id), model: `${said.sourceLabel} · ${said.short}` };
    });
    return { followers: complete && Array.isArray(works) ? followers : null, works: off, rooms: own };
}

// One row of the sheet's lists: a name over its facts and, where it can be moved, its own tick (a label, so the
// whole row presses it). After the change the same row says its result instead.
function offRow(doc, entry, { result = null } = {}) {
    const made = (tag, className, text) => {
        const node = doc.createElement(tag);
        node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    };
    const row = made(entry.id && !result ? 'label' : 'div', 'chat-model-off');
    const name = made('span', 'chat-model-off-name', entry.name);
    name.dataset.i18nAuthored = '';
    const meta = made('span', 'chat-model-off-meta');
    meta.dataset.i18nSkip = '';
    if (result) {
        row.dataset.result = result.ok ? 'moved' : 'refused';
        meta.append(made('span', '', result.words));
        row.append(name, meta);
        return { row, input: null };
    }
    meta.append(made('span', '', entry.why ? `${entry.model} · ` : entry.model));
    if (entry.why) meta.append(made('span', entry.fallback ? 'model-fallback' : '', entry.why));
    if (entry.room) meta.append(made('span', '', ` · ${entry.room}`));
    let input = null;
    if (entry.id) {
        input = made('input', 'ui-checkbox');
        input.setAttribute('type', 'checkbox');
        input.dataset.move = `${entry.kind}:${entry.id}`;
        input.checked = false;  // nothing is ticked until the owner ticks it
    }
    row.append(...[name, input, meta].filter(Boolean));
    return { row, input };
}
function offList(doc, rows) {
    const list = doc.createElement('ul');
    list.className = 'chat-model-off-list';
    for (const { row } of rows) {
        const item = doc.createElement('li');
        item.append(row);
        list.append(item);
    }
    return list;
}

/**
 * The confirming sheet of a pick in the Main chat, on the shared dialog: what Main becomes and what it was, the new
 * route's facts, what running work does, how many running works follow Main, and in its one sentence what else the
 * change resets. Under them, what is not on Main now (`census`: the page's last state, see mainSheetLists): the works
 * on other models and the rooms with their own, each with a tick to move it too, none ticked beforehand.
 * Resolves false when the owner cancels, else `{ moves }`: the ticked rows, each to be moved by its own act.
 */
export async function confirmMainChange({ next, previous, census = null, dialog = openConfirmDialog, doc = globalThis.document }) {
    const resets = previous?.lane === 'subscription' && next.source !== previous.source
        ? 'If Main\'s account is pinned it becomes Auto, and a context window set by hand for Main is cleared.'
        : 'A context window set by hand for Main is cleared.';
    const lists = mainSheetLists(census || {});
    const ticks = [];
    const confirmed = await dialog({
        title: 'Change Main',
        body: resets,
        confirmLabel: 'Change Main',
        content: { label: 'Main model', mount(container, sheet) {
            const line = (text) => {
                const made = doc.createElement('p');
                made.className = 'chat-model-sheet-fact';
                if (text !== undefined) made.textContent = text;
                return made;
            };
            const row = doc.createElement('p');
            row.className = 'chat-model-sheet-row';
            row.innerHTML = `${sourceMark(next)}<b data-i18n-skip>${esc(next.id)}</b>`
                + (previous ? `<span data-i18n-skip>${esc(fmt('was {model}', { model: previous.short }))}</span>` : '');
            const facts = line(`${routeFacts(next)}.`);
            facts.dataset.i18nSkip = '';
            const count = line();
            count.dataset.mainFollowers = lists.followers === null ? 'unknown' : String(lists.followers);
            if (lists.followers === null) count.textContent = 'How many running works follow Main is unknown right now: the list of running work is incomplete.';
            else if (lists.followers === 0) count.textContent = 'No running work follows Main now.';
            else if (lists.followers === 1) count.textContent = '1 running work follows Main now.';
            else fmtInto(count, '{n} running works follow Main now.', { n: lists.followers });
            container.append(row, facts, line(`${MAIN_RULE} ${MAIN_FOLLOW_NOTE}`), count);
            const sub = (text) => { const made = line(text); made.className = 'chat-model-sheet-sub'; return made; };
            const section = (entries, heading) => {
                if (!entries.length) return;
                const rows = entries.map((entry) => ({ entry, ...offRow(doc, entry) }));
                ticks.push(...rows.filter((made) => made.input));
                container.append(heading, offList(doc, rows));
            };
            section(lists.works, sub(lists.works.some((entry) => entry.id) ? 'Not on Main now · tick to move too' : 'Not on Main now'));
            const roomsHead = sub();
            fmtInto(roomsHead, 'Rooms that start new work on their own model · {n}', { n: lists.rooms.length });
            section(lists.rooms, roomsHead);
            if (!ticks.length) return undefined;
            container.append(line('A ticked work follows Main from its next step; a ticked room starts new work on Main.'));
            // The confirming button says how many more it moves, as the ticks change.
            const onTick = () => {
                const n = ticks.filter((made) => made.input.checked).length;
                if (n) sheet?.setConfirmLabel('Change Main and move {n} more', { n });
                else sheet?.setConfirmLabel('Change Main');
            };
            container.addEventListener('change', onTick);
            return () => container.removeEventListener('change', onTick);
        } },
    });
    return confirmed ? { moves: ticks.filter((made) => made.input.checked).map((made) => made.entry) } : false;
}

/**
 * What the ticked rows did, each in its own words, on the shared dialog: every tick was one act after Main was
 * written, so every row has its own result (a work's answer says when it switches; a refusal says the server's words).
 */
export function showMainMoves({ results, dialog = openConfirmDialog, doc = globalThis.document }) {
    const moved = results.filter((result) => result.ok).length;
    return dialog({
        title: 'Main changed',
        alert: true,
        body: moved === results.length ? (moved === 1 ? '1 more was moved to Main.' : fmt('{n} more were moved to Main.', { n: moved }))
            : fmt('{moved} of {n} were moved to Main.', { moved, n: results.length }),
        content: { label: 'What was moved', mount(container) {
            container.append(offList(doc, results.map((result) => offRow(doc, result, { result }))));
        } },
    });
}

/**
 * The capsule in its room's composer row. `save(body)` is the owner write (`apiClient.ownerChatModel`), `openSettings()`
 * Settings › Accounts and keys, `askMain({next, previous, census})` the Main chat's confirming sheet, `moveWork(taskId)` one
 * work's return to Main and `sayMoves({results})` the sheet of what its ticks did, `store` the window's catalog store,
 * `placeList(list)` the suggestions' place above the row.
 * Returns `{ el, capsule, syncState(data), hasPendingSave, pendingSave, pick, startSearch, destroy }`.
 */
export function createModelControl({
    el, row, projectId = '', chatId = projectId ? null : 1, save = apiClient.ownerChatModel, openSettings = () => {},
    showToast = () => {}, refreshState = () => {}, onLayout, onSettled, askMain = confirmMainChange, store = modelCatalog,
    moveWork = (taskId) => taskModel(taskId, globalThis.crypto?.randomUUID?.() || `main-sheet-${Date.now()}-${++serial}`, { action: 'follow' }),
    sayMoves = showMainMoves,
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

    // `main` is what Main is now (the state body's `main_route` field), `choice` the room's own (`projects[].model_choice`); the
    // Main chat's capsule is Main, so it has no choice of its own.
    // `works`, `complete` and `projects` are the last state body's census, for the hint's fact line and Main's sheet.
    const state = { main: null, choice: { ...MAIN_CHOICE }, works: null, complete: false, projects: [], saving: null,
        lastOk: true, destroyed: false };
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
        // The settings document, read only when it changes a word: why a saved API source this catalog did not list
        // cannot send, or Main's processing tier where the catalog states prices (Fast and Economy price otherwise).
        if (said && store.status().read && !store.status().failed
            && (sourceUnlisted(store.index(), shown()) || store.index().priced)) void store.readConnections();
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
    // One act per tick, after Main itself was written (there is no batch endpoint): a work follows Main again, a
    // room's own model is cleared. Each answers for itself, and a refusal of one stops none of the others.
    async function moveToMain(moves) {
        const results = [];
        for (const move of moves) {
            try {
                if (move.kind === 'work') {
                    const reply = await moveWork(move.id);
                    results.push({ ...move, ok: true, words: APPLIED_WORDS[reply?.applied] || tr('model.sheet.follows', 'Follows Main now.') });
                } else {
                    const reply = await save({ project_id: move.id, kind: 'main' });
                    if (reply?.ok !== true || reply.model_choice?.kind !== 'main') throw new Error('The server did not confirm it.');
                    results.push({ ...move, ok: true, words: tr('model.sheet.room_follows', 'New work here starts on Main.') });
                }
            } catch (error) {
                results.push({ ...move, ok: false, words: fmt('Not moved: {reason}', { reason: refusalWords(error).words }) });
            }
        }
        if (state.destroyed) return;
        refreshState(true);
        await sayMoves({ results, doc });
    }
    async function changeMain(route) {
        const next = describe(route);
        // The sheet lists the works and rooms of the page's last state: read it once more first, so an act made a
        // moment ago (a card's model, another room's capsule) is in it. A slow read does not hold the sheet back.
        await Promise.race([Promise.resolve().then(() => refreshState(true)).catch(() => {}),
            new Promise((resolve) => { setTimeout(resolve, FRESH_STATE_MS); })]);
        if (state.destroyed) return false;
        const answer = await askMain({ next, previous: describe(state.main), doc,
            census: { works: state.works, complete: state.complete, projects: state.projects, index: store.index() } });
        if (!answer || state.destroyed) return false;
        const written = await writeMain(route);
        if (written && answer.moves?.length && !state.destroyed) await moveToMain(answer.moves);
        return written;
    }
    function writeMain(route) {
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
            if (Array.isArray(data.works)) {
                state.works = data.works;
                state.complete = data.works_complete === true && data.supervisor_ready === true;
            }
            if (Array.isArray(data.projects)) state.projects = data.projects;
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
