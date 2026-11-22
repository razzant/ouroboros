/**
 * The Model capsule (docs/DESIGN.md "Composer owner controls"): the first capsule of the composer's row, before
 * Subagents, Context and Effort, and one of their family closed. A Project room's capsule holds the model new work in
 * that room starts on and saves it for the room; until the room chooses it follows Main and says so: the name word is
 * `Main`, the value Main's model. Changing Main's model or selecting Ultrafast for Main asks for confirmation.
 * Closed, the capsule shows the room's model and its non-Standard tier, never the model of a running work.
 *
 * A press opens the Model sheet (model_sheet.js; never hover): search, Main, the pinned and recent
 * models, all sources, and at its foot the room's processing tier. The sheet lists only the sources that can lead a
 * work, one row per (source, model) (chat_model_routes.js); the catalog suggests and never admits, so any id can still
 * be typed. A room's pick saves at once through `POST /api/owner/chat-model`, its tier through
 * `POST /api/owner/chat-processing`; Send waits for both. The strip offers what the scope's own route says where it
 * is bound (processing_choice.js): what follows Main runs on Main's account, its pin or Auto, which the settings say
 * (until they do nothing of an account is confirmed: an unread pin is never read as Auto); a room's own model on
 * Auto across its source's accounts.
 */
import { apiClient, taskModel } from './api_client.js';
import { APPLIED_WORDS, basisWords, leadView, refusalWords } from './card_model.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { fmt, fmtInto, tr, translateString } from './i18n.js';
import { routeFacts } from './model_list.js';
import { haptic } from './haptics.js';
import { isCoarsePointer, isStackLayout } from './layout.js';
import { CONNECTION_CONDITION, modelCatalog, revisionIn } from './model_catalog_store.js';
import { MAIN_FOLLOW_NOTE } from './model_roles.js';
import { openModelSheet } from './model_sheet.js';
import { sourceMark } from './model_source_mark.js';
import { MAIN_ULTRAFAST_SCOPE, MAIN_ULTRAFAST_ROLES, MAIN_ULTRAFAST_ROLLBACK, PIN_OBSERVATION, canChooseTier, confirmMainUltrafast, resolvedTier, routeTiers, tierBinding, tierNamed, tierWord } from './processing_choice.js';
import { PROCESSING_PREFERENCE_KEY } from './route_editor_primitives.js';
import { bindPopoverPosition } from './ui_interactions.js';
import { escapeHtmlAttr as esc } from './ui_primitives.js';
import { clock, workControls } from './work_status.js';
import {
    MAIN_CHOICE, describeRoute, normalizeChoice, normalizeRoute, routeSource, sameRoute, shortName, sourceUnlisted,
} from './chat_model_routes.js';

// When a pick of Main takes effect, its confirming sheet's rule line.
export const MAIN_RULE = 'New work starts on it everywhere, unless a room has its own model.';
/** The sheet's own fact about what follows Main, said where Main is changed (owner 2026-11-15, #3271): a fact, never
 *  a forecast of cost. The catalog's cache-write price follows it when the catalog states one. */
export const MAIN_CACHE_NOTE = 'Whatever follows Main (running works, child works, a consciousness wake) may write most of its start to the new model\'s cache again at its next call.';
// How long Main's sheet waits for a state read taken for it before it lists what the page already has.
const FRESH_STATE_MS = 700;
const MAIN_TIER_CHANGED = 'Main changed. Check the current model and account, then choose its tier again.';
const MAIN_MODEL_CHANGED = 'Main changed on another device. Check its model and tier, then choose again: nothing was changed.';
const SOURCE_CHANGED = 'The key {model} goes through changed on another device. Its tiers were read again: check them, then choose again.';
let serial = 0;

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
export async function confirmMainChange({ next, previous, census = null, tierNote = '', dialog = openConfirmDialog, doc = globalThis.document }) {
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
            const cache = line(next.cacheWrite ? `${translateString(MAIN_CACHE_NOTE)} ${next.cacheWrite}` : MAIN_CACHE_NOTE);
            if (next.cacheWrite) cache.dataset.i18nFmt = 'main cache note';  // composed: the overlay leaves it alone
            // Main's own tier goes with it: what the new route says of that tier, said before the write.
            const tier = tierNote ? line(tierNote) : null;
            if (tier) tier.dataset.i18nSkip = '';
            container.append(...[row, facts, line(`${MAIN_RULE} ${MAIN_FOLLOW_NOTE}`), cache, tier, count].filter(Boolean));
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
 * The capsule in its room's composer row. `save(body)` is the owner write (`apiClient.ownerChatModel`),
 * `saveTier(body)` the room's or Main's own processing tier (`apiClient.ownerChatProcessing`), `openSettings()`
 * Settings › Accounts and keys, `askMain({next, previous, census, tierNote})` the Main chat's confirming sheet,
 * `askTier(facts)` the question before Main takes Ultrafast (processing_choice.js), `moveWork(taskId)`
 * one work's return to Main and `sayMoves({results})` the sheet of what its ticks did, `store` the window's catalog
 * store, `place(sheet, anchor)` the sheet's place above the row.
 * Returns `{ el, capsule, syncState(data), hasPendingSave, pendingSave, pick, chooseTier, open, destroy }`.
 */
export function createModelControl({
    el, row, projectId = '', chatId = projectId ? null : 1, save = apiClient.ownerChatModel,
    saveTier = apiClient.ownerChatProcessing, openSettings = () => {},
    showToast = () => {}, refreshState = () => {}, onLayout, onSettled, askMain = confirmMainChange,
    askTier = confirmMainUltrafast, store = modelCatalog,
    moveWork = (taskId) => taskModel(taskId, globalThis.crypto?.randomUUID?.() || `main-sheet-${Date.now()}-${++serial}`, { action: 'follow' }),
    sayMoves = showMainMoves,
    isStack = () => isStackLayout(), isCoarse = () => isCoarsePointer(),
    // Anchored to the row, as the hint is to its capsule: the composer under it never moves.
    place = (sheet) => bindPopoverPosition(sheet, { anchor: row, align: 'start' }),
    win = globalThis.window, doc = globalThis.document,
}) {
    const isMain = !projectId;
    el.classList.add('chat-capsule');
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', 'Model');
    el.dataset.known = 'false';
    // The head's words are written by render(): the mark, whose model this is, the model, and its tier when not Standard.
    el.innerHTML = '<button type="button" class="chat-capsule-head ui-hit-area chat-model-head" aria-expanded="false" aria-haspopup="dialog"><span class="chat-capsule-glyph" aria-hidden="true"></span><span class="chat-capsule-name"></span><span class="chat-capsule-value" data-i18n-skip></span><span class="chat-capsule-tier" data-i18n-skip hidden></span></button>';
    const head = el.querySelector('.chat-model-head');
    const glyph = el.querySelector('.chat-capsule-glyph');
    const name = el.querySelector('.chat-capsule-name');
    const value = el.querySelector('.chat-capsule-value');
    const tierWordEl = el.querySelector('.chat-capsule-tier');

    // `main` is what Main is now (the state body's `main_route` field), `choice` the room's own (`projects[].model_choice`),
    // `tier` its own processing tier (`projects[].processing_choice`, '' = Default); the Main chat's capsule is Main, so
    // its tier is Main's own (`processing_tiers.main`). `works`, `complete` and `projects` are the last state body's
    // census, for the sheet's fact lines and Main's confirming sheet.
    const state = { main: null, choice: { ...MAIN_CHOICE }, tier: '', works: null, complete: false, projects: [],
        saving: null, tierSaving: null, lastOk: true, destroyed: false };
    let sheet = null, markKey = null, nameWords = null;
    let detailAsked = null;  // the routed value whose own endpoints the open sheet asked for

    const shown = () => (isMain || state.choice.kind === 'main' ? state.main : normalizeRoute(state.choice));
    const follows = () => isMain || state.choice.kind === 'main';
    const tiers = () => store.tiers?.() || null;
    // The tier this scope's new work takes: its own, else Main's for what follows Main, else the default one.
    const resolved = () => (isMain ? resolvedTier({ choice: tiers()?.main || '', tiers: tiers() })
        : resolvedTier({ choice: state.tier, follows: follows(), tiers: tiers() }));
    // The tier a pick would run on here, for the catalog's price beside it (a catalog price is Standard's).
    const tierOf = (route, { main = false } = {}) => (isMain ? resolved()
        : state.tier || resolvedTier({ follows: main, tiers: tiers() }));
    const describe = (route) => (route ? describeRoute(store.index(), route, { tier: resolved() }) : null);
    // The account `route` runs on here: Main's own for what follows Main (its pin, '' for Auto, null until the
    // settings say which: then no account's word is taken), Auto for a room's own model (CLIENT_CONTRACT: a carried
    // model has no pinned account).
    const accountOf = (route) => (follows() ? store.mainAccount?.(route) ?? null : '');
    const tiersOf = (route, account = accountOf(route)) => routeTiers({ index: store.index(), route, account,
        detail: store.detail?.(route) || null, sourceLabel: describe(route)?.sourceLabel });
    // The open sheet asks once for its route's own endpoints, and again when the route under it changes.
    function syncDetail() {
        const route = shown();
        // Asked again when the held answers were dropped (a connection changed under the same routed value).
        const asking = `${route?.model}|${store.detailEpoch?.() ?? ''}`;
        if (!sheet || !route || asking === detailAsked) return;
        detailAsked = asking;
        void store.readDetail?.(route);
    }

    function render() {
        const said = describe(shown());
        el.dataset.known = String(Boolean(said));
        el.dataset.own = String(Boolean(said) && !follows());
        // The mark itself is the identity: what it is drawn from (a subscription's `credentialHarness`) can arrive
        // after the words it stands beside, and a key of hand-picked fields would hide that change.
        const mark = sourceMark(said);
        if (mark !== markKey) { markKey = mark; glyph.innerHTML = mark; }
        // Whose model this is: Main's, or its own source's.
        const sourceNamed = Boolean(said) && !follows();
        const words = sourceNamed ? said.sourceLabel : 'Main';
        if (words !== nameWords) {  // the overlay may have translated the word it was given
            nameWords = words;
            if (sourceNamed) name.dataset.i18nSkip = ''; else delete name.dataset.i18nSkip;  // a source's name is not chrome
            name.textContent = words;
        }
        const model = said ? said.short : '—';
        if (value.textContent !== model) value.textContent = model;
        // A tier other than Standard is said beside the model, closed too (owner 12.11: the word only when not Standard).
        const tier = resolved();
        const tierText = tierNamed(tier) ? tierWord(tier) : '';
        if (tierWordEl.textContent !== tierText) tierWordEl.textContent = tierText;
        tierWordEl.hidden = !tierText;
        el.dataset.tier = tier || 'default';
        const tierSaid = tierText ? `, ${tierText}` : '';
        head.setAttribute('aria-label', !said ? 'Model for new work: not read yet'
            : isMain ? fmt('Main model: {model}, {source}.', { model: `${said.short}${tierSaid}`, source: said.sourceLabel })
                : follows() ? fmt('Model for new work here: {model}, {source}. Follows Main.', { model: `${said.short}${tierSaid}`, source: said.sourceLabel })
                    : fmt('Model for new work here: {model}, {source}.', { model: `${said.short}${tierSaid}`, source: said.sourceLabel }));
        // The settings document, read only when it changes a word: why a saved API source this catalog did not list
        // cannot send, or in Main the cache tier a catalog's cache-write figure is said for (Main's confirming sheet).
        if (said && store.status().read && !store.status().failed
            && (sourceUnlisted(store.index(), shown()) || (isMain && store.index().cacheFigures))) void store.readConnections();
        syncDetail();
        sheet?.paint();
    }

    /* --- the sheet's words: what the tier strip applies to, and what each tier means on this route --- */
    function scopeWords() {
        const tier = resolved();
        const own = isMain ? tiers()?.main : state.tier;
        const state_ = own || !tier ? tierWord(tier)
            : fmt(!isMain && follows() ? 'Default · {tier} as Main' : 'Default · {tier}', { tier: tierWord(tier) });
        // The tier a work runs on, its own or Default's (Main's included), reaches its new subagents without one
        // (owner 13.11).
        if (!isMain) return `${state_} · ${tr('processing.scope.room_helpers', 'for new work here and its new subagents without a tier of their own; running work keeps its own')}`;
        // The model census describes the acting route, not where its tier came from.
        return `${state_} · ${tr('processing.scope.main_inheritance', 'for Main and work that inherits its tier, even on a fallback model; new subagents without their own tier inherit it too')}`;
    }
    const tierView = () => {
        const here = tiersOf(shown());
        return {
            choice: isMain ? tiers()?.main || '' : state.tier,
            resolved: resolved(),
            busy: Boolean(state.tierSaving),
            scope: scopeWords(),
            ...here,
        };
    };
    // Main's Ultrafast asks first, for Main as it is bound at the press (its route, its account, and the engine
    // store and account fingerprint the catalog names): what it reaches, what that route says of it and what it is
    // listed at where that is told. A Main that moved under the question was not asked about: nothing is written.
    async function confirmUltrafast() {
        const fromSheet = sheet;
        const bound = () => tierBinding(store.index(), state.main, accountOf(state.main));
        const route = state.main;
        const pin = accountOf(route);
        const tier = tiers()?.main;
        const shownSource = store.revisionOf?.(route) ?? null;  // the source whose tiers the sheet showed
        let read;
        try { read = await store.readMainTierSettings(); } catch (error) {
            if (!state.destroyed) showToast(error?.message || MAIN_TIER_CHANGED, 'error');
            return null;
        }
        if (state.destroyed || sheet !== fromSheet) return null;
        // The read is what the question stands on: its source, its pin, its tiers. A pin this page had not read yet
        // (null) is learnt, never a change; one it had read and that now reads otherwise is one.
        store.adoptSettings?.(read.settings);
        const source = revisionIn(read.settings, route);
        const asked = bound();
        const stillShown = () => bound() === asked && tiers()?.main === tier && canChooseTier('ultrafast', tiersOf(state.main))
            && (store.revisionOf?.(route) ?? null) === source;
        if (!sameRoute(state.main, route) || !sameRoute(read.route, route) || (pin !== null && read.account !== pin) || read.tier !== tier) {
            showToast(MAIN_TIER_CHANGED, 'info');
            return null;
        }
        if (source === null) {  // an API connection the read gives no revision for: nothing to bind the yes to
            showToast('Could not verify the key Main uses. Reload, then choose again.', 'error');
            return null;
        }
        if (shownSource && shownSource !== source) {  // another source than the one shown: its own word first
            await store.readDetail?.(route);
            if (!state.destroyed) showToast(fmt(SOURCE_CHANGED, { model: describe(route)?.short || '' }), 'info');
            return null;
        }
        await store.readDetail?.(route);  // this source's endpoints, never another's: the price below is theirs
        // The question stands on what was read before this wait (route, pin, tier, source); a sheet dismissed
        // meanwhile was not a press anyone still makes: nothing opens.
        if (state.destroyed || sheet !== fromSheet) return null;
        if (!stillShown()) {  // the same Main, now known whole: its account does not offer Ultrafast
            showToast(fmt('Ultrafast is not offered for {model} here. {words}', { model: describe(route)?.short || '',
                words: tiersOf(route).explain('ultrafast') }), 'info');
            return null;
        }
        const said = describe(read.route);
        const here = tiersOf(read.route, read.account);
        const pinned = said?.lane === 'subscription' && Boolean(read.account);
        closeSheet({ focus: true });  // the sheet stays usable during the read; the question opens alone
        const yes = await askTier({ model: said?.short || '', source: [said?.sourceLabel, here.accountLabel].filter(Boolean).join(' · '), pinned,
            explain: here.explain('ultrafast'), priceSaid: Boolean(here.priceOf('ultrafast')), doc });
        if (!yes || state.destroyed) return null;
        if (stillShown()) return { pinned, condition: { ...read.binding, ultrafast: asked, ...(source ? { [CONNECTION_CONDITION]: source } : {}) } };
        showToast('Main changed while you were asked: Ultrafast was not set.', 'info');
        return null;
    }

    /* --- saving: a room's pick posts at once; Main's asks first, then writes Main itself --- */
    function readChangedMain(pinned) {
        // A pinned confirmation can be refused by the account observation even when none of the settings changed.
        const settings = store.readConnections({ fresh: true });
        return pinned ? Promise.all([settings, store.read({ force: true })]) : settings;
    }
    function write(body, accept, pinned = false) {
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
                if (error?.body?.code === 'settings_changed_elsewhere') {
                    showToast(MAIN_MODEL_CHANGED, 'info');
                    await readChangedMain(pinned);
                } else showToast(error?.message || 'Could not change the model.', 'error');
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
        const shownSource = store.revisionOf?.(route) ?? null;  // the picked route's source as this sheet showed it
        // The sheet lists the works and rooms of the page's last state: read it once more first, so an act made a
        // moment ago (a card's model, another room's capsule) is in it. A slow read does not hold the sheet back.
        await Promise.race([Promise.resolve().then(() => refreshState(true)).catch(() => {}),
            new Promise((resolve) => { setTimeout(resolve, FRESH_STATE_MS); })]);
        if (state.destroyed) return false;
        // Main as stored now, and the tokens that bind this change to it: the tier said below is the one the new model
        // keeps, a tier chosen elsewhere a moment ago included, and a move elsewhere before the write is refused (409).
        let read;
        try { read = await store.readMainTierSettings(); } catch (error) {
            if (!state.destroyed) showToast(error?.message || MAIN_MODEL_CHANGED, 'error');
            return false;
        }
        if (state.destroyed) return false;
        if (!sameRoute(read.route, state.main)) {
            showToast(MAIN_MODEL_CHANGED, 'info');
            refreshState(true);
            return false;
        }
        store.adoptSettings?.(read.settings);  // Main's tier and pin, and the sources, as this read has them
        // Main's own tier goes to the new route with it: a pinned account stays only within its source.
        const kept = read.tier;
        // An Ultrafast carried onto an API connection stands on that connection as shown: the read must name it, and a
        // source other than the one the sheet showed is read again before anything is asked.
        const source = kept === 'ultrafast' ? revisionIn(read.settings, route) : '';
        if (source === null) {
            showToast(fmt('Could not verify the key {model} goes through. Reload, then choose again.', { model: shortName(route) }), 'error');
            return false;
        }
        const effective = kept || String(read.settings[PROCESSING_PREFERENCE_KEY] || '');
        if (tierNamed(effective)) await store.readDetail?.(route);
        if (state.destroyed) return false;
        if (!sameRoute(read.route, state.main)) { showToast(MAIN_MODEL_CHANGED, 'info'); return false; }
        const connectionMoved = () => source && (store.revisionOf?.(route) ?? null) !== source;
        if ((source && shownSource && shownSource !== source) || connectionMoved()) {
            showToast(fmt(SOURCE_CHANGED, { model: shortName(route) }), 'info');
            return false;
        }
        // Standard's catalog amount and cache write do not describe a premium tier, even when its price is unknown.
        const next = describeRoute(store.index(), route, { tier: effective });
        const previous = describe(read.route);
        const pin = next.lane === 'subscription' && next.source === routeSource(read.route) ? read.account : '';
        const confirmedTier = () => tierBinding(store.index(), route, pin);
        const asked = kept === 'ultrafast' ? confirmedTier() : '';
        const tiers = tiersOf(route, pin);
        const tierNote = tierNamed(effective) ? fmt('Main keeps its tier, {tier}. {words}', { tier: tierWord(effective),
            words: [tiers.accountLabel, tiers.explain(effective),
                asked && !tiers.priceOf(effective) && translateString('What Ultrafast costs on this route is not known here.'),
                ...(asked ? [MAIN_ULTRAFAST_SCOPE, MAIN_ULTRAFAST_ROLES, MAIN_ULTRAFAST_ROLLBACK, pin && PIN_OBSERVATION]
                    .filter(Boolean).map((words) => translateString(words)) : [])].filter(Boolean).join(' ') }) : '';
        const answer = await askMain({ next, previous, doc, tierNote,
            census: { works: state.works, complete: state.complete, projects: state.projects, index: store.index() } });
        if (!answer || state.destroyed) return false;
        // A source that moved under the question was not the one asked about: nothing is written.
        if ((asked && asked !== confirmedTier()) || connectionMoved()) {
            showToast(fmt(SOURCE_CHANGED, { model: next.short }), 'info');
            return false;
        }
        const written = await writeMain(route, { ...read.binding, ...(asked ? { ultrafast: asked } : {}), ...(source ? { [CONNECTION_CONDITION]: source } : {}) }, Boolean(asked && pin));
        if (written && answer.moves?.length && !state.destroyed) await moveToMain(answer.moves);
        return written;
    }
    function writeMain(route, binding, pinned) {
        return write({ kind: 'model', model: route.model, use_local: route.use_local, if_unchanged: binding }, (reply) => {
            const written = normalizeRoute(reply.main_route);
            if (!written) throw new Error('The server did not confirm the model.');
            state.main = written;
            // What else the write changed, from its own answer: a dropped pin and a cleared manual window.
            if (reply.account === 'auto') store.adoptMainAccount?.('', written);
            const also = [reply.account === 'auto' ? 'Main\'s account is Auto now.' : '',
                reply.window_reset === true ? 'Main\'s context window set by hand was cleared.' : ''].filter(Boolean);
            if (also.length) showToast(also.join(' '), 'info');
        }, pinned);
    }
    /** One pick from the sheet: null follows Main again (its own row); a route becomes the room's own, even Main's
     *  model now (a model row is always that route). In the Main chat a pick changes Main itself. */
    function pick(route) {
        if (state.saving || state.destroyed) return undefined;
        if (isMain) {
            closeSheet();
            if (!route || sameRoute(route, state.main)) return undefined;  // Main's own row: nothing changes
            return changeMain(route);
        }
        if (route ? state.choice.kind === 'model' && sameRoute(route, normalizeRoute(state.choice)) : state.choice.kind === 'main') return undefined;
        haptic('tick');  // another model, chosen by a press (haptics.js)
        return saveRoom(route ? { kind: 'model', model: route.model, use_local: route.use_local } : { kind: 'main' });
    }
    /** The scope's own tier ('' = Default): the room's for its new work, or Main's own. It moves nothing else. */
    function chooseTier(tier) {
        if (state.tierSaving || state.destroyed) return state.tierSaving;
        let settle = () => {};
        state.tierSaving = new Promise((resolve) => { settle = resolve; });
        sheet?.paint();  // the strip shows the save in flight before it is sent
        const saving = state.tierSaving;
        void (async () => {
            let ok = false;
            let pinned = false;
            try {
                const body = projectId ? { processing: tier, project_id: projectId } : { processing: tier };
                if (isMain && tier === 'ultrafast') {
                    const confirmation = await confirmUltrafast();
                    if (!confirmation || state.destroyed) return;
                    body.if_unchanged = confirmation.condition;
                    pinned = confirmation.pinned;
                }
                const reply = await saveTier(body);
                if (state.destroyed) return false;
                if ((reply?.project_id || '') !== projectId || reply?.ok !== true) throw new Error('The server did not confirm the tier.');
                if (isMain) store.adoptTiers?.({ main: reply.main_processing, default: reply.processing_default });
                else state.tier = String(reply.processing_choice || '');
                haptic('tick');
                ok = true;
            } catch (error) {
                if (state.destroyed) return;
                if (isMain && tier === 'ultrafast' && error?.body?.code === 'settings_changed_elsewhere') {
                    showToast(MAIN_TIER_CHANGED, 'info');
                    await readChangedMain(pinned);
                } else showToast(error?.message || 'Could not change the processing tier.', 'error');
            } finally {
                state.tierSaving = null;
                if (!state.destroyed) { render(); refreshState(true); }
                settle(ok);
            }
        })();
        return saving;
    }
    async function setPinned(route, pinned) {
        try {
            await store.setPinned(route, pinned);
        } catch (error) {
            if (!state.destroyed) showToast(error?.message || 'Could not change the pinned models.', 'error');
        }
    }

    /* --- the sheet: opened by a press, never hover; one capsule or sheet open at a time --- */
    function openSheet() {
        if (sheet || state.destroyed) return;
        control.peers.forEach((peer) => peer.snapClosed());
        head.setAttribute('aria-expanded', 'true');
        el.dataset.sheetOpen = 'true';
        row.dataset.ownerSheet = 'true';
        sheet = openModelSheet({
            anchor: row, label: isMain ? 'Main model' : 'Model for new work here', doc, place,
            stack: isStack, coarse: isCoarse, returnFocus: head,
            list: {
                store, main: () => state.main, current: shown, follows: () => !isMain && follows(),
                mainRow: isMain ? 'main' : 'follow', tierOf,
                // The room's own model, then the owner's recent ones: a room keeps its choice in view.
                recents: () => [...(isMain ? [] : [normalizeRoute(state.choice)].filter(Boolean)), ...store.recents()],
                mainFacts: () => {
                    const said = describe(state.main);
                    if (!said) return '';
                    const tier = resolvedTier({ follows: true, tiers: tiers() });
                    return [tr('model.row.follows', 'Follows Main'), said.short, tierNamed(tier) ? tierWord(tier) : ''].filter(Boolean).join(' · ');
                },
                onRoute: (route) => { if (!isMain) closeSheet({ focus: true }); pick(route); },
                onPin: setPinned,
                onSettings: () => { closeSheet(); openSettings(); },
            },
            tier: { onChoose: chooseTier, view: tierView, mainRoute: () => (follows() ? shown() : null) },
            onClose: () => {
                sheet = null;
                head.setAttribute('aria-expanded', 'false');
                delete el.dataset.sheetOpen;
                delete row.dataset.ownerSheet;
                onLayout?.();
                onSettled?.();
            },
        });
        void store.read();
        void store.readRecents();
        detailAsked = null;
        render();  // the open sheet's own reads: its route's endpoints, and Main's account where it decides a tier
        onLayout?.();
    }
    function closeSheet({ focus = false } = {}) {
        sheet?.close({ focus });
    }
    const onHead = () => { if (sheet) closeSheet(); else openSheet(); };
    head.addEventListener('click', onHead);

    // The composer's row treats it as one of its capsules: it fits closed like them, and opening one closes the others.
    const control = {
        el, kind: 'model', hoverCapable: false, peers: [],
        isOpen: () => Boolean(sheet), isPinned: () => Boolean(sheet), isSettled: () => true,
        measureName() {}, sayHint() {}, relayout() { sheet?.reposition(); },
        setOpen(open) { if (open) openSheet(); else closeSheet(); },
        snapClosed() { closeSheet(); },
        destroy() {},
    };
    const unsubscribe = store.subscribe(() => { if (!state.destroyed) render(); });
    render();

    return {
        el,
        capsule: control,
        /** One `/api/state` body (or a room's own read): what Main is now, this room's choice and tier, and its works'
         *  models. A save in flight answers for itself; a body without a field keeps what was last read. */
        syncState(data) {
            if (state.destroyed || !data) return;
            if (Array.isArray(data.works)) {
                state.works = data.works;
                state.complete = data.works_complete === true && data.supervisor_ready === true;
            }
            if (Array.isArray(data.projects)) state.projects = data.projects;
            if (data.processing_tiers) store.adoptTiers?.(data.processing_tiers);
            if (!state.saving) {
                const main = normalizeRoute(data.main_route);
                if (main) state.main = main;
                const room = projectId ? data.projects?.find?.((project) => project?.id === projectId) : null;
                if (room) state.choice = normalizeChoice(room.model_choice);
                if (room && !state.tierSaving) state.tier = String(room.processing_choice || '');
            }
            render();
        },
        hasPendingSave: () => Boolean(state.saving || state.tierSaving),
        /** Resolves once the saves settled: true unless a model save in flight was refused (a refused tier keeps the
         *  old one and says so; an earlier refusal already said its own). */
        pendingSave: async () => {
            const model = state.saving;
            await state.tierSaving;
            return model ? model : true;
        },
        /** The capsule's one save path: a route, or null to follow Main again. */
        pick,
        chooseTier,
        open: openSheet,
        destroy() {
            state.destroyed = true;
            closeSheet();
            unsubscribe();
            head.removeEventListener('click', onHead);
        },
    };
}
