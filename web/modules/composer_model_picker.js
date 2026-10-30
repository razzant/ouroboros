// The composer's chat model (DESIGN "Composer owner controls"): a quiet trigger, first on the capsule row, names what
// NEW work in this room starts on and opens one sheet: a search, Favorites / Subscription / API, a source rail, the
// routes with their stars, and Main and Fallback in its foot. A choice posts at once (POST /api/owner/chat-model); the
// trigger says Saving until the room's readback answers, a refusal returns it to the saved choice and says why, and
// Send waits for the save (composer.js). Running work keeps the model it started on. The data is chat_model_routes.js;
// the catalog is read when a sheet opens (and then only when stale or failed), never polled.
import { apiClient, apiFetch } from './api_client.js';
import { fmt } from './i18n.js';
import { escapeHtmlAttr as esc } from './ui_primitives.js';
import { bindDialogFocus, bindPopoverPosition } from './ui_interactions.js';
import { HARNESS_MARKS, harnessPresentation } from './harness_presentation.js';
import { MODEL_CATALOG_TIMEOUT_MS, mergeModelCatalog } from './settings_catalog.js';
import {
    LANES, LANE_LABELS, LOCAL_SOURCE, MAIN_REFERENCE, buildRouteModel, customRoute, homeOf, normalizeChoice,
    rememberedNames, resolveChoice, rowChoice, sameChoice, searchRoutes,
} from './chat_model_routes.js';

const CATALOG_FRESH_MS = 5 * 60 * 1000;
// The catalog's display names, kept on this device so a reloaded trigger names its model before the next read. A
// name only labels: it never lists a route, and a read replaces it.
const NAMES_KEY = 'ouro_chat_model_names';
const ALL = 'all';
let serial = 0;

// Brand marks are monochrome and inherit currentColor. OpenAI's is the harness SSOT's (harness_presentation.js,
// the same knot as web/providers/openai.svg); Anthropic's and OpenRouter's are Simple Icons' (CC0; OpenRouter's from
// simple-icons 16.36.0, sourced from openrouter.ai, the split arrows of web/providers/openrouter.ico); the
// OpenAI-compatible mark is the stroke of web/providers/openai-compatible.svg without its tile. A source without a
// mark of its own shows its initial, never an invented logo.
const MARKS = Object.freeze({
    openai: { viewBox: HARNESS_MARKS.codex.viewBox, body: `<path fill="currentColor" d="${HARNESS_MARKS.codex.path}"/>` },
    anthropic: { viewBox: '0 0 24 24', body: '<path fill="currentColor" d="M17.3041 3.541h-3.6718l6.696 16.918H24Zm-10.6082 0L0 20.459h3.7442l1.3693-3.5527h7.0052l1.3693 3.5528h3.7442L10.5363 3.5409Zm-.3712 10.2232 2.2914-5.9456 2.2914 5.9456Z"/>' },
    openrouter: { viewBox: '0 0 24 24', body: '<path fill="currentColor" d="M16.778 1.844v1.919q-.569-.026-1.138-.032-.708-.008-1.415.037c-1.93.126-4.023.728-6.149 2.237-2.911 2.066-2.731 1.95-4.14 2.75-.396.223-1.342.574-2.185.798-.841.225-1.753.333-1.751.333v4.229s.768.108 1.61.333c.842.224 1.789.575 2.185.799 1.41.798 1.228.683 4.14 2.75 2.126 1.509 4.22 2.11 6.148 2.236.88.058 1.716.041 2.555.005v1.918l7.222-4.168-7.222-4.17v2.176c-.86.038-1.611.065-2.278.021-1.364-.09-2.417-.357-3.979-1.465-2.244-1.593-2.866-2.027-3.68-2.508.889-.518 1.449-.906 3.822-2.59 1.56-1.109 2.614-1.377 3.978-1.466.667-.044 1.418-.017 2.278.02v2.176L24 6.014Z"/>' },
    'openai-compatible': { viewBox: '0 0 64 64', body: '<path d="M20 18h24a10 10 0 0 1 0 20H32a8 8 0 0 0 0 16h12M18 46h14" fill="none" stroke="currentColor" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round"/>' },
    [ALL]: { viewBox: '0 0 24 24', body: '<g fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><rect x="4" y="4" width="6.5" height="6.5" rx="1.6"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1.6"/><rect x="4" y="13.5" width="6.5" height="6.5" rx="1.6"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.6"/></g>' },
});
const svg = (viewBox, body, cls = 'chat-model-mark') => `<svg class="${cls}" viewBox="${viewBox}" aria-hidden="true" focusable="false">${body}</svg>`;
const GLYPHS = {
    search: svg('0 0 24 24', '<circle cx="11" cy="11" r="6.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="m16 16 4 4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>', 'chat-model-glyph'),
    chevron: svg('0 0 24 24', '<path d="m7 10 5 5 5-5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>', 'chat-model-glyph'),
    star: svg('0 0 24 24', '<path d="m12 3.5 2.6 5.4 5.9.8-4.3 4.1 1 5.9L12 17l-5.2 2.7 1-5.9L3.5 9.7l5.9-.8z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>', 'chat-model-glyph'),
    check: svg('0 0 24 24', '<path d="m5 12.5 4.5 4.5L19 7" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>', 'chat-model-check'),
};

/** A source's mark: its harness's (subscriptions), its provider's, else its initial. */
export function sourceMark(source) {
    if (!source) return '<span class="chat-model-mark chat-model-initial" aria-hidden="true">·</span>';
    if (source.lane === 'subscription') {
        const mark = harnessPresentation(source.harness);
        return mark.generic ? initial(source.label) : svg(mark.viewBox, `<path fill="currentColor" d="${mark.path}"/>`);
    }
    const mark = MARKS[source.id === ALL ? ALL : source.kind];
    return mark ? svg(mark.viewBox, mark.body) : initial(source.label);
}
const initial = (label) => `<span class="chat-model-mark chat-model-initial" aria-hidden="true">${esc((String(label || '?').trim()[0] || '?').toUpperCase())}</span>`;

// One catalog and one star list for every composer in the window: a second room's sheet reads what the first one read.
const storedNames = () => { try { return JSON.parse(globalThis.localStorage?.getItem(NAMES_KEY) || '{}') || {}; } catch { return {}; } };
const shared = { catalog: null, readAt: 0, failed: false, reading: null, favorites: null, favoritesRead: null,
    favoritesConfirmed: [], favoriteChanges: [], favoritesSave: Promise.resolve(), listeners: new Set(), names: storedNames() };
const notify = () => shared.listeners.forEach((listener) => listener());
function adoptCatalog(data) {
    shared.catalog = mergeModelCatalog(shared.catalog || {}, data);
    shared.names = rememberedNames(shared.names, data);
    try { globalThis.localStorage?.setItem(NAMES_KEY, JSON.stringify(shared.names)); } catch { /* names stay in memory */ }
}

function readCatalog({ force = false } = {}) {
    if (shared.reading) return shared.reading;
    if (!force && shared.catalog && !shared.failed && Date.now() - shared.readAt < CATALOG_FRESH_MS) return Promise.resolve();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), MODEL_CATALOG_TIMEOUT_MS);
    shared.reading = (async () => {
        try {
            const response = await apiFetch('/api/model-catalog', { cache: 'no-store', signal: controller.signal });
            const data = await response.json();
            if (!response.ok || !Array.isArray(data?.items)) throw new Error(data?.error || `HTTP ${response.status}`);
            adoptCatalog(data);
            shared.failed = false;
        } catch (error) {
            // A failed read keeps what an earlier one listed, and says so.
            shared.catalog = mergeModelCatalog(shared.catalog || {}, { read_state: 'transport',
                errors: [{ provider_id: 'catalog', error: error?.name === 'AbortError' ? 'timed out' : String(error?.message || error) }] });
            shared.failed = true;
        } finally {
            clearTimeout(timer);
            shared.readAt = Date.now();
            shared.reading = null;
            notify();
        }
    })();
    notify();
    return shared.reading;
}

if (typeof document !== 'undefined') {
    // Settings › Models refreshing its catalog refreshes ours too, without another read.
    document.addEventListener('settings-model-catalog:updated', (event) => {
        if (!event?.detail || !Array.isArray(event.detail.items)) return;
        adoptCatalog(event.detail);
        shared.readAt = Date.now();
        shared.failed = false;
        notify();
    });
}

function readFavorites() {
    shared.favoritesRead ||= apiClient.uiPreferences()
        .then((prefs) => { shared.favorites = Array.isArray(prefs?.model_favorites) ? prefs.model_favorites : [];
            shared.favoritesConfirmed = shared.favorites; })
        .catch(() => { shared.favoritesRead = null; })  // unread, never shown as an empty list
        .finally(notify);
    return shared.favoritesRead;
}

/** Star or unstar one route; saves run in order and a refusal restores the list it replaced. An unread list is
 *  never replaced by a guess (the stars wait for it). */
function toggleFavorite(route, showToast) {
    if (shared.favorites === null) return;
    const change = { route, wanted: !shared.favorites.includes(route) };
    const apply = (list, op) => op.wanted ? [...new Set([...list, op.route])] : list.filter((value) => value !== op.route);
    shared.favoriteChanges.push(change);
    shared.favorites = apply(shared.favorites, change);
    notify();
    shared.favoritesSave = shared.favoritesSave.then(async () => {
        try {
            const saved = await apiClient.saveUiPreferences({ model_favorites: apply(shared.favoritesConfirmed, change) });
            if (!Array.isArray(saved?.model_favorites)) throw new Error('The server did not confirm favorites.');
            shared.favoritesConfirmed = saved.model_favorites;
        } catch (error) {
            showToast?.(`Could not save favorites: ${error?.message || error}`, 'error');
        } finally {
            shared.favoriteChanges.shift();
            shared.favorites = shared.favoriteChanges.reduce(apply, shared.favoritesConfirmed);
            notify();
        }
    });
}

/**
 * @param {object} deps
 * @param {HTMLElement} deps.el  the row's `.chat-model-picker` slot (composer.js markup)
 * @param {string} [deps.projectId]  the Project room; empty for Main
 * @param {(choice: object) => Promise<object>} [deps.save]  `apiClient.ownerChatModel`
 * @param {(message: string, tone?: string) => void} [deps.showToast]
 * @param {() => void} [deps.onLayout]  the trigger's words changed width
 */
export function createComposerModelPicker({ el, projectId = '', save = apiClient.ownerChatModel, showToast = () => {},
    onLayout = () => {}, win = globalThis.window, doc = globalThis.document }) {
    const id = `chat-model-${++serial}`;
    const state = { saved: null, shown: null, routes: null, saving: null, lastOk: true, destroyed: false,
        open: false, lane: 'subscription', source: { subscription: '', api: ALL }, query: '', active: 0 };
    const node = (tag, className, attrs = {}) => {
        const made = doc.createElement(tag);
        made.className = className;
        for (const [key, value] of Object.entries(attrs)) made.setAttribute(key, value);
        return made;
    };
    // The trigger: the source's mark, the model (a value the row's fit may cut), the scope word and the chevron.
    const trigger = node('button', 'chat-model-trigger ui-hit-area', { type: 'button', 'aria-haspopup': 'dialog',
        'aria-expanded': 'false', 'aria-controls': `${id}-sheet` });
    const icon = node('span', 'chat-model-icon');
    const name = node('span', 'chat-model-name', { 'data-i18n-skip': '' });
    const scope = node('span', 'chat-model-scope');
    scope.hidden = true;
    const chevron = node('span', 'chat-model-chevron', { 'aria-hidden': 'true' });
    chevron.innerHTML = GLYPHS.chevron;
    trigger.append(icon, name, scope, chevron);
    const said = node('span', 'chat-model-said', { role: 'status', 'aria-live': 'polite' });
    el.replaceChildren(trigger, said);
    let sheet = null;
    let scrim = null;
    let releaseFocus = null;
    let position = null;
    let model = buildRouteModel();
    let flat = [];
    let painted = false;

    const rebuild = () => {
        model = buildRouteModel({ catalog: shared.catalog || {}, routes: state.routes || {}, choice: state.shown || MAIN_REFERENCE,
            favorites: shared.favorites || [], names: shared.names });
        return resolveChoice(model, state.shown || MAIN_REFERENCE, state.routes || {});
    };
    const sourceLine = (entry) => `${entry.sourceLabel} · ${LANE_LABELS[entry.lane === 'subscription' ? 'subscription' : 'api']}`;

    function renderTrigger() {
        const resolved = rebuild();
        const current = resolved.current;
        const known = Boolean(state.shown && current);
        el.dataset.known = String(known);
        el.dataset.moved = String(known && resolved.choice.kind !== 'main');
        el.dataset.saving = String(Boolean(state.saving));
        icon.innerHTML = sourceMark(current ? model.source(current.source)
            || { lane: current.lane, label: current.sourceLabel, harness: current.harness, kind: current.kind } : null);
        const words = known ? current.name : 'Main';
        if (name.textContent !== words) {
            name.textContent = words;
            if (painted) onLayout();  // a later width change re-measures the dock; the first paint is the composer's own
        }
        scope.textContent = state.saving ? 'Saving' : 'this chat';
        scope.hidden = !state.saving && !(known && resolved.choice.kind !== 'main');
        const via = current ? sourceLine(current) : '';
        const whose = resolved.choice.kind === 'main' ? 'Follows Main.' : resolved.unconfigured
            ? 'Fallback is not set, so new work starts on Main.' : resolved.choice.kind === 'fallback'
                ? 'Follows Fallback.' : 'Chosen for this chat.';
        trigger.setAttribute('aria-label', known
            ? fmt('Model for new work in this chat: {name}, {via}. {whose}', { name: current.name, via, whose })
            : 'Model for new work in this chat: not read yet');
        trigger.title = known ? `${current.name} · ${via}` : 'Model for new work in this chat';
    }

    /* --- the sheet ----------------------------------------------------------------------------------------- */
    function sheetMarkup() {
        return `<div class="chat-model-search">${GLYPHS.search}<input type="text" class="chat-model-query" role="combobox"`
            + ` aria-expanded="true" aria-controls="${id}-list" aria-autocomplete="list" placeholder="Search models"`
            + ' autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" aria-label="Search models">'
            + '<kbd class="chat-model-key">esc</kbd></div>'
            + `<div class="chat-model-tabs" role="tablist" aria-label="Model access">${LANES.map((lane) =>
                `<button type="button" role="tab" class="chat-model-tab" id="${id}-tab-${lane}" data-lane="${lane}">${LANE_LABELS[lane]}</button>`).join('')}</div>`
            + `<div class="chat-model-body"><div class="chat-model-rail" role="group" aria-label="Source"></div>`
            + `<div class="chat-model-list" id="${id}-list" role="listbox" aria-label="Models"></div></div>`
            + '<div class="chat-model-foot"></div>';
    }

    function railMarkup(lane) {
        const sources = model.laneSources(lane);
        const choices = (lane === 'api' && sources.length > 1 ? [{ id: ALL, label: 'All API sources', lane }] : []).concat(sources);
        return choices.map((entry) => `<button type="button" class="chat-model-source ui-hit-area" data-source="${esc(entry.id)}"`
            + ` aria-pressed="${state.source[lane] === entry.id}" aria-label="${esc(entry.label)}" title="${esc(entry.label)}">`
            + `${sourceMark(entry)}</button>`).join('');
    }

    function optionMarkup(entry, index, { twoLine, resolved }) {
        const own = model.source(entry.source);
        const selected = resolved.choice.kind === 'model' && resolved.choice.model === entry.route && !resolved.choice.use_local;
        const tags = [resolved.main && !resolved.main.local && resolved.main.route === entry.route ? '<span class="chat-model-tag" data-tag="main">Main</span>' : '',
            resolved.fallback && !resolved.fallback.local && resolved.fallback.route === entry.route ? '<span class="chat-model-tag">Fallback</span>' : ''].join('');
        const note = !entry.listed ? (shared.catalog ? 'not listed' : '') : entry.shipped ? 'shipped list' : '';
        const sub = twoLine ? `<span class="chat-model-sub">${sourceMark(own)}<span data-i18n-skip>${esc(own?.label || '')}</span> · ${LANE_LABELS[entry.lane]}${note ? ` · ${note}` : ''}</span>`
            : note ? `<span class="chat-model-sub">${note}</span>` : '';
        const starred = model.starred.has(entry.route);
        const unread = shared.favorites === null;
        const label = `${entry.name}, ${own?.label || ''} · ${LANE_LABELS[entry.lane]}${note ? `, ${note}` : ''}`;
        return `<div class="chat-model-option${twoLine || note ? ' chat-model-two' : ''}" role="option" id="${id}-o${index}" data-index="${index}"`
            + ` aria-selected="${selected}" data-active="${index === state.active}" aria-label="${esc(label)}">`
            + `<span class="chat-model-text"><span class="chat-model-row-name" data-i18n-skip>${esc(entry.name)}</span>${sub}</span>${tags}`
            + `<button type="button" class="chat-model-star ui-hit-area" data-star="${esc(entry.route)}" aria-pressed="${starred}"${unread ? ' aria-disabled="true"' : ''}`
            + ` aria-label="${esc(fmt('Favorite {name} via {source}', { name: entry.name, source: own?.label || '' }))}">${GLYPHS.star}</button>`
            + `${GLYPHS.check}</div>`;
    }

    function sourceNote(source, empty) {
        const parts = [];
        if (shared.reading && !shared.catalog) parts.push('Reading models…');
        else if (source?.problems?.length) parts.push(`Could not read every list: ${source.problems.slice(0, 3).join('; ')}.`);
        else if (shared.failed) parts.push('The model catalog could not be read; earlier suggestions are kept.');
        if (empty && !shared.reading) parts.push(source?.lane === 'subscription' ? 'No models listed for this source.' : 'No models listed.');
        return parts.length ? `<div class="chat-model-note" role="presentation">${esc(parts.join(' '))}</div>` : '';
    }

    function catalogNote(lane = '') {
        const affected = model.sources.filter((source) => (!lane || source.lane === lane) && source.problems.length);
        const global = model.problems.filter((problem) => !lane || !problem.lane || problem.lane === lane);
        const problems = [...global.map((problem) => problem.text), ...affected.map((source) => source.label)];
        return sourceNote(problems.length ? { problems } : null, false);
    }

    function listItems(resolved) {
        const query = state.query.trim();
        const out = [];
        let rows = [];
        if (query) {
            rows = searchRoutes(model, query);
            out.push({ html: catalogNote() });
            out.push({ head: rows.length ? fmt('{n} results in all sources', { n: rows.length }) : '' });
            if (!rows.length) out.push({ html: `<div class="chat-model-empty" role="presentation"><b data-i18n-skip>${esc(fmt('No model matches “{query}”', { query }))}</b>Searched Favorites, Subscription and API.</div>` });
            rows.forEach((entry) => out.push({ entry, twoLine: true }));
            const typedSource = state.lane === 'favorites' ? '' : state.source[state.lane];
            const route = typedSource && typedSource !== ALL ? customRoute(typedSource, query) : '';
            if (route && !model.row(route)) {
                out.push({ custom: route, label: fmt('Use “{model}” via {source}', { model: query, source: model.source(typedSource)?.label || typedSource }) });
            }
            return out;
        }
        if (state.lane === 'favorites') {
            rows = model.favorites;
            if (shared.favorites === null) out.push({ html: `<div class="chat-model-empty" role="presentation">${shared.favoritesRead ? 'Reading favorites…' : 'Favorites could not be read.'}</div>` });
            else if (!rows.length) out.push({ html: '<div class="chat-model-empty" role="presentation"><b>No favorites yet</b>Star a model in Subscription or API.<button type="button" class="chat-model-go" data-go="subscription">Browse Subscription</button></div>' });
            else out.push({ head: fmt('Favorites · {n}', { n: rows.length }) });
            rows.forEach((entry) => out.push({ entry, twoLine: true }));
            return out;
        }
        const chosen = state.source[state.lane];
        if (chosen === ALL) {
            out.push({ head: 'All API sources' });
            rows = model.laneRows('api');
            out.push({ html: catalogNote('api') });
            if (!rows.length) out.push({ html: sourceNote(null, true) });
            rows.forEach((entry) => out.push({ entry, twoLine: true }));
            return out;
        }
        const source = model.source(chosen);
        if (!source) {
            const gap = catalogNote(state.lane);
            if (gap) { out.push({ html: gap }); return out; }
            out.push({ html: `<div class="chat-model-empty" role="presentation">${state.lane === 'subscription'
                ? (shared.catalog ? '<b>No subscription sources</b>Connect one in Settings › Accounts.' : 'Reading models…')
                : (shared.catalog ? '<b>No API sources answered</b>Add a key in Settings › Accounts.' : 'Reading models…')}</div>` });
            return out;
        }
        out.push({ head: `${sourceMark(source)}<span data-i18n-skip>${esc(source.label)}</span> · ${LANE_LABELS[source.lane]}` });
        rows = model.sourceRows(chosen);
        const global = model.problems.filter((problem) => !problem.lane || problem.lane === state.lane);
        const note = (global.length ? sourceNote({ problems: global.map((problem) => problem.text) }, false) : '')
            + sourceNote(source, !rows.length);
        if (note) out.push({ html: note });
        rows.forEach((entry) => out.push({ entry, twoLine: false }));
        if (source.id === LOCAL_SOURCE && resolved.current?.local) {
            out.push({ html: `<div class="chat-model-note" role="presentation">${esc(resolved.current.name)} runs on the local lane; change it in Settings › Models.</div>` });
        }
        return out;
    }

    // What held the focus inside a repainted part, by identity, so a star, a source or a Use keeps it.
    function focusKey() {
        const focused = doc.activeElement;
        if (!sheet?.contains(focused) || focused.classList?.contains('chat-model-query') || focused === sheet) return null;
        for (const name of ['star', 'source', 'use', 'go']) if (focused.dataset?.[name] !== undefined) return [name, focused.dataset[name]];
        return null;
    }

    function renderSheet({ keepActive = false } = {}) {
        if (!sheet) return;
        const resolved = rebuild();
        const held = focusKey();
        const searching = Boolean(state.query.trim());
        for (const tab of sheet.querySelectorAll('.chat-model-tab')) {
            const on = !searching && tab.dataset.lane === state.lane;
            tab.setAttribute('aria-selected', String(on));
            tab.tabIndex = tab.dataset.lane === state.lane ? 0 : -1;
        }
        const body = sheet.querySelector('.chat-model-body');
        const rail = !searching && state.lane !== 'favorites';
        body.dataset.rail = rail ? 'on' : 'off';
        if (rail) {
            const lane = state.lane;
            const ids = model.laneSources(lane).map((entry) => entry.id);
            if (!(state.source[lane] === ALL && lane === 'api') && !ids.includes(state.source[lane])) state.source[lane] = ids[0] || (lane === 'api' ? ALL : '');
            if (lane === 'api' && ids.length <= 1 && state.source.api === ALL && ids[0]) state.source.api = ids[0];
            sheet.querySelector('.chat-model-rail').innerHTML = railMarkup(lane);
        }
        const items = listItems(resolved);
        flat = items.filter((item) => item.entry || item.custom);
        if (!keepActive) {
            const at = flat.findIndex((item) => item.entry && resolved.choice.kind === 'model' && item.entry.route === resolved.choice.model);
            state.active = Math.max(0, at);
        }
        state.active = Math.max(0, Math.min(state.active, flat.length - 1));
        let index = -1;
        const list = sheet.querySelector('.chat-model-list');
        list.innerHTML = items.map((item) => {
            if (item.head !== undefined) return item.head ? `<div class="chat-model-head" role="presentation">${item.head}</div>` : '';
            if (item.html !== undefined) return item.html;
            index += 1;
            if (item.custom) {
                return `<div class="chat-model-option chat-model-custom" role="option" id="${id}-o${index}" data-index="${index}" aria-selected="false"`
                    + ` data-active="${index === state.active}"><span class="chat-model-text"><span class="chat-model-row-name">${esc(item.label)}</span>`
                    + '<span class="chat-model-sub">A model ID this source may serve; nothing checks it first.</span></span></div>';
            }
            return optionMarkup(item.entry, index, { twoLine: item.twoLine, resolved });
        }).join('');
        const active = list.querySelector(`#${id}-o${state.active}`);
        sheet.querySelector('.chat-model-query').setAttribute('aria-activedescendant', active ? active.id : '');
        if (!keepActive) active?.scrollIntoView?.({ block: 'nearest' });
        sheet.querySelector('.chat-model-foot').innerHTML = footMarkup(resolved);
        if (held) {
            const again = [...sheet.querySelectorAll(`[data-${held[0]}]`)].find((node) => node.dataset[held[0]] === held[1]);
            (again || sheet.querySelector('.chat-model-query')).focus({ preventScroll: true });
        }
        position?.reposition();
    }

    function footMarkup(resolved) {
        const ref = (kind, entry) => {
            const inUse = resolved.choice.kind === kind && !(kind === 'fallback' && !entry);
            const own = entry ? model.source(entry.source) || { lane: entry.lane, label: entry.sourceLabel, harness: entry.harness, kind: entry.kind } : null;
            const label = kind === 'main' ? 'Main' : 'Fallback';
            const body = entry ? `${sourceMark(own)}<b data-i18n-skip>${esc(entry.name)}</b><span class="chat-model-ref-source" data-i18n-skip>${esc(entry.sourceLabel)}</span>`
                : '<span class="chat-model-ref-source">Not set in Settings</span>';
            const action = !entry ? '' : inUse ? '<span class="chat-model-in-use">In use</span>'
                : `<button type="button" class="chat-model-use" data-use="${kind}" aria-label="${esc(fmt('Use {role}: {name} via {source}', { role: label, name: entry.name, source: entry.sourceLabel }))}">Use</button>`;
            return `<div class="chat-model-ref" data-ref="${kind}"><span class="chat-model-ref-key">${label}</span><span class="chat-model-ref-route">${body}</span>${action}</div>`;
        };
        return ref('main', resolved.main) + ref('fallback', resolved.fallback)
            + '<p class="chat-model-hint">Applies to new work in this chat; running work keeps its model. Main and Fallback stay as set.</p>';
    }

    function setActive(index, { scroll = false } = {}) {
        state.active = index;
        for (const option of sheet.querySelectorAll('.chat-model-option[data-active="true"]')) option.dataset.active = 'false';
        const option = sheet.querySelector(`#${id}-o${index}`);
        if (!option) return;
        option.dataset.active = 'true';
        sheet.querySelector('.chat-model-query').setAttribute('aria-activedescendant', option.id);
        if (scroll) option.scrollIntoView({ block: 'nearest' });
    }

    function open() {
        if (state.open || state.destroyed) return;
        state.open = true;
        void readCatalog();
        if (shared.favorites === null) void readFavorites();
        const resolved = rebuild();
        const home = homeOf(model, resolved);
        state.lane = home.lane;
        if (home.source) state.source[home.lane] = home.source;
        state.query = '';
        sheet = doc.createElement('section');
        sheet.className = 'ui-popup chat-model-sheet';
        sheet.id = `${id}-sheet`;
        sheet.setAttribute('role', 'dialog');
        sheet.setAttribute('aria-modal', 'true');
        sheet.setAttribute('aria-label', projectId ? 'Model for this Project chat' : 'Model for Main chat');
        sheet.innerHTML = sheetMarkup();
        scrim = doc.createElement('div');
        scrim.className = 'chat-model-scrim';
        doc.body.append(scrim, sheet);
        el.dataset.open = 'true';
        trigger.setAttribute('aria-expanded', 'true');
        renderSheet();
        position = bindPopoverPosition(sheet, { anchor: trigger, align: 'start' });
        // A phone keeps its keyboard down until the owner asks to search: the sheet itself takes the focus.
        const coarse = typeof win?.matchMedia === 'function' && win.matchMedia('(pointer: coarse)').matches;
        releaseFocus = bindDialogFocus(sheet, { initialFocus: coarse ? sheet : sheet.querySelector('.chat-model-query'),
            returnFocus: trigger, preventScroll: true });
        sheet.addEventListener('keydown', onKey);
        sheet.addEventListener('click', onClick);
        sheet.addEventListener('pointermove', onPointer);
        sheet.querySelector('.chat-model-query').addEventListener('input', onQuery);
        scrim.addEventListener('click', () => close());
        doc.addEventListener('pointerdown', onOutside, true);
    }

    function close({ restoreFocus = true } = {}) {
        if (!state.open) return;
        state.open = false;
        doc.removeEventListener('pointerdown', onOutside, true);
        position?.destroy();
        position = null;
        releaseFocus?.({ restoreFocus });
        releaseFocus = null;
        sheet?.remove();
        scrim?.remove();
        sheet = scrim = null;
        el.dataset.open = 'false';
        trigger.setAttribute('aria-expanded', 'false');
    }

    const onOutside = (event) => {
        if (sheet?.contains(event.target) || trigger.contains(event.target) || scrim?.contains(event.target)) return;
        close({ restoreFocus: false });
    };
    const onPointer = (event) => {
        const option = event.target?.closest?.('.chat-model-option');
        if (option && Number(option.dataset.index) !== state.active) setActive(Number(option.dataset.index));
    };
    const onQuery = (event) => {
        state.query = event.target.value;
        renderSheet();
    };
    function goLane(lane, { focusTab = false } = {}) {
        state.lane = lane;
        state.query = '';
        sheet.querySelector('.chat-model-query').value = '';
        renderSheet();
        if (focusTab) sheet.querySelector(`#${id}-tab-${lane}`)?.focus();
    }
    function activate(index) {
        const item = flat[index];
        if (!item) return;
        choose(item.custom ? { kind: 'model', model: item.custom, use_local: false } : rowChoice(item.entry));
    }
    function onClick(event) {
        const target = event.target;
        const tab = target.closest?.('.chat-model-tab');
        if (tab) { goLane(tab.dataset.lane); return; }
        const go = target.closest?.('[data-go]');
        if (go) { goLane(go.dataset.go); return; }
        const source = target.closest?.('.chat-model-source');
        if (source) {
            state.source[state.lane] = source.dataset.source;
            renderSheet();
            return;
        }
        const star = target.closest?.('.chat-model-star');
        if (star) {
            event.stopPropagation();
            toggleFavorite(star.dataset.star, showToast);
            return;
        }
        const use = target.closest?.('.chat-model-use');
        if (use) { choose({ kind: use.dataset.use }); return; }
        const option = target.closest?.('.chat-model-option');
        if (option) activate(Number(option.dataset.index));
    }
    function onKey(event) {
        if (event.isComposing) return;
        const query = sheet.querySelector('.chat-model-query');
        if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            if (state.query) { state.query = ''; query.value = ''; renderSheet(); } else close();
        } else if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && !event.target.closest?.('.chat-model-tabs')) {
            if (!flat.length) return;
            event.preventDefault();
            setActive((state.active + (event.key === 'ArrowDown' ? 1 : -1) + flat.length) % flat.length, { scroll: true });
        } else if (event.key === 'Enter' && event.target === query) {
            event.preventDefault();
            activate(state.active);
        } else if ((event.key === 'ArrowLeft' || event.key === 'ArrowRight') && event.target.closest?.('.chat-model-tabs')) {
            event.preventDefault();
            const at = LANES.indexOf(state.lane);
            goLane(LANES[(at + (event.key === 'ArrowRight' ? 1 : LANES.length - 1)) % LANES.length], { focusTab: true });
        }
    }

    /* --- saving ---------------------------------------------------------------------------------------------- */
    function choose(choice) {
        close();
        const next = normalizeChoice(choice);
        if (state.saving || sameChoice(next, state.saved || MAIN_REFERENCE)) return state.saving;
        const before = state.saved;
        state.shown = next;
        state.saving = (async () => {
            try {
                const reply = await save({ ...next, ...(projectId ? { project_id: projectId } : {}) });
                if ((reply?.project_id || '') !== projectId) throw new Error('The answer belongs to another room.');
                if (reply?.ok !== true || !['main', 'fallback', 'model'].includes(reply?.model_choice?.kind)
                    || (reply.model_choice.kind === 'model' && !reply.model_choice.model)) {
                    throw new Error('The server did not confirm the saved model choice.');
                }
                if (state.destroyed) return false;
                state.saved = state.shown = normalizeChoice(reply?.model_choice);
                state.lastOk = true;
                const resolved = rebuild();
                said.textContent = resolved.current ? fmt('New work in this chat starts on {name}.', { name: resolved.current.name }) : '';
            } catch (error) {
                if (state.destroyed) return false;
                state.lastOk = false;
                state.shown = before;
                showToast(`Could not change this chat's model: ${error?.message || error}`, 'error');
            } finally {
                state.saving = null;
                if (!state.destroyed) renderTrigger();
            }
            return state.lastOk;
        })();
        renderTrigger();
        return state.saving;
    }

    const onShared = () => {
        if (state.destroyed) return;
        renderTrigger();
        if (state.open) renderSheet({ keepActive: true });
    };
    shared.listeners.add(onShared);
    trigger.addEventListener('click', () => (state.open ? close() : open()));
    // The owner reaching for the trigger reads the catalog the sheet will show (once; a fresh read is reused).
    const prefetch = () => { void readCatalog(); };
    trigger.addEventListener('pointerenter', prefetch, { once: true });
    trigger.addEventListener('focus', prefetch, { once: true });
    renderTrigger();
    painted = true;

    return {
        el,
        trigger,
        /** The trigger's model name, a value the row's fit may cut (composer_owner_controls.js). */
        nameEl: () => name,
        isOpen: () => state.open,
        /** The room's readback: Main's state, or the named Project's row; the references come with /api/state. */
        syncState(data) {
            if (state.destroyed || !data) return;
            if (data.model_routes && typeof data.model_routes === 'object') state.routes = data.model_routes;
            const row = projectId ? data.projects?.find?.((project) => project?.id === projectId) : data;
            const known = projectId ? Boolean(row) : 'model_choice' in data;
            if (known && !state.saving) {
                state.saved = state.shown = normalizeChoice(row?.model_choice);
            }
            renderTrigger();
            if (state.open) renderSheet({ keepActive: true });
        },
        hasPendingSave: () => Boolean(state.saving),
        /** Resolves once the save settled: true when it was accepted. */
        pendingSave: () => state.saving || Promise.resolve(state.lastOk),
        /** The sheet's one save path (a row, a typed route, a reference's Use): `{kind, model?, use_local?}`. */
        choose,
        open,
        close,
        destroy() {
            state.destroyed = true;
            close({ restoreFocus: false });
            shared.listeners.delete(onShared);
            el.replaceChildren();
        },
    };
}
