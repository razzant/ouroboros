// Skills → Browse (UX §2, boards B2, B2c, B5, B5b, B14): every connected
// source's catalog as one board of cards in sort order. Search filters the
// loaded catalogs while typing; Enter also asks the registry (ClawHub), whose
// hits join the board. Each card says the consequence of its one action before
// it is pressed. The list never waits for GitHub: it is the host's cache, and a
// source that could not be read says so in the status line without blanking
// the others. Install returns at once; the card follows the skill's real state.
import { applyMasonry } from './masonry.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { skillSourcesApi } from './api_client.js';
import { installedSkillModel } from './skill_card_renderer.js';
import { openSkillDetailSheet } from './skill_detail_sheet.js';
import {
    buttonHtml, catalogWhoHtml, chipHtml, connectsServices, connectsText, countsAsYoursText, factsText, officialMarkHtml, plural, relativeAge,
    skillIconHtml, skipReviewButtonHtml, statusHtml, trustSentence, versionText,
    skillTitle,
} from './skill_presentation.js';

const HIDDEN_KEY = 'ouroboros.skills.recommendedHidden';
// The Connectors chip is derived from each entry's `connects` fact (an installed skill's place declaration, or
// the Hub entry's), never a typed tag; a catalog category of the same id joins it.
export const CONNECTORS = 'connectors';
export const isConnector = (item) => connectsServices(item).length > 0 || (item?.category || []).includes(CONNECTORS);
const CLAMP_CHARS = 600;

function readHiddenShelf() {
    try { return String(globalThis.localStorage?.getItem(HIDDEN_KEY) || ''); } catch { return ''; }
}

function writeHiddenShelf(value) {
    try { globalThis.localStorage?.setItem(HIDDEN_KEY, value); } catch { /* the shelf just comes back */ }
}

const votesOf = (item) => {
    if (typeof item?.signals?.votes === 'number') return item.signals.votes;
    if (typeof item?.registry_stats?.stars === 'number') return item.registry_stats.stars;
    return -1;
};
const timeOf = (item) => Date.parse(item?.updated_at || '') || 0;
const byName = (a, b) => String(a.name).localeCompare(String(b.name), 'en', { sensitivity: 'base' });

/** Sort a list of catalog items: Popular (votes, unknown last), Name, Recently updated (unknown last). */
export function sortCatalogItems(items, sort, query = '') {
    const list = [...items];
    if (sort === 'name') return list.sort(byName);
    if (sort === 'updated') return list.sort((a, b) => (timeOf(b) - timeOf(a)) || byName(a, b));
    if (sort === 'best' && query) {
        const q = query.toLowerCase();
        const rank = (item) => (String(item.name).toLowerCase() === q ? 0 : String(item.name).toLowerCase().startsWith(q) ? 1 : 2);
        return list.sort((a, b) => rank(a) - rank(b) || votesOf(b) - votesOf(a) || byName(a, b));
    }
    return list.sort((a, b) => votesOf(b) - votesOf(a) || byName(a, b));
}

/** Local text filter over the loaded catalogs (name, description, author, category labels). */
export function matchesQuery(item, query, categoryLabels = new Map()) {
    const q = String(query || '').trim().toLowerCase();
    if (!q) return true;
    const cats = (item.category || []).map((id) => categoryLabels.get(id) || id);
    return [item.name, item.description, item.owner, item.source_name, ...cats]
        .some((text) => String(text || '').toLowerCase().includes(q));
}

/** Category counts over the whole loaded catalog («All 52 · Messaging 7 · …»). */
export function categoryCounts(items, categories) {
    const counts = new Map(categories.map((category) => [category.id, 0]));
    for (const item of items) {
        for (const id of new Set(item.category || [])) {
            if (counts.has(id)) counts.set(id, counts.get(id) + 1);
        }
    }
    return counts;
}

/** The recommended skills the owner does not have yet (the «Recommended to start» shelf). */
export function recommendedMissing(items, installedByName) {
    return items.filter((item) => item.recommended && !item.withdrawn && !installedByName.has(item.name));
}

/**
 * The «Recommended to start» shelf: only for an install that predates the starter
 * skills (the host's starter marker says `upgrade`; a new install chose its ticks
 * in the wizard), on the unfiltered board, once the installed listing is known.
 */
export function shelfItems(catalog, installedByName, { query = '', source = 'all', category = 'all', officialOnly = false } = {}) {
    if (catalog?.starter?.state !== 'upgrade' || !installedByName || query || source !== 'all' || category !== 'all' || officialOnly) return [];
    return recommendedMissing(Array.isArray(catalog.items) ? catalog.items : [], installedByName);
}

function installedFrom(row) {
    return row?.install?.source?.id || '';
}

/**
 * One Browse card's action row: a consequence sentence and one action, from the
 * catalog item, its source, the installed listing (`listed`: it has been read)
 * and the installs started here.
 */
export function browseCardState(item, { source = null, installed = null, listed = true, pending = null, failure = '', reviewing = false } = {}) {
    const id = escapeHtml(item.id);
    const act = (label, action, tone = 'default', extra = '') => buttonHtml(label, { tone, attrs: `data-browse-action="${action}" data-item="${id}"${extra ? ` ${extra}` : ''}` });
    if (pending && !installed?.lifecycle_virtual) {
        if (pending.phase === 'reviewing') {
            const age = relativeAge(pending.startedAt) || 'just now';
            return {
                status: statusHtml(`Reviewing · started ${escapeHtml(age)}`, 'muted', { work: true }),
                action: skipReviewButtonHtml(`data-browse-action="skip" data-item="${id}"`),
                note: 'Installed. Ouroboros reads it before it runs or gets your keys; you can leave this page.',
            };
        }
        return { status: statusHtml('Installing…', 'muted', { work: true }), action: act('Install', 'install', 'primary').replace('<button', '<button disabled') };
    }
    if (failure) {
        return {
            status: statusHtml(`Couldn’t install ${escapeHtml(item.name)} from ${escapeHtml(item.source_name || 'its source')}`, 'danger'),
            action: act('Try again', 'install'),
            note: `${escapeHtml(failure)} Nothing was installed.`,
        };
    }
    if (!listed && item.installed) {
        // The catalog's own join, until the installed listing (which carries every action) has been read.
        return item.installed.state === 'name_taken'
            ? { status: statusHtml(`You have another “${escapeHtml(item.name)}” (from ${escapeHtml(item.installed.source_name || 'another source')})`, 'warn'), action: '' }
            : { status: statusHtml(`Installed · ${escapeHtml(versionText(item.installed.version))}`, 'muted'), action: '' };
    }
    if (installed && !installed.lifecycle_virtual) {
        const fromHere = !installedFrom(installed) || installedFrom(installed) === item.source_id;
        if (!fromHere) {
            const other = installed.install?.source?.name || 'another source';
            return { status: statusHtml(`You have another “${escapeHtml(item.name)}” (from ${escapeHtml(other)})`, 'warn'), action: act('Compare…', 'compare') };
        }
        const model = installedSkillModel(installed, { reviewing });
        const v = escapeHtml(versionText(installed.version));
        const update = installed.install?.update_available;
        const retryUpdate = update && installed.install?.update_rejected;
        if (model.modified && !retryUpdate) return { status: statusHtml(`Installed · modified here`, 'muted'), action: act('Details', 'open', 'ghost') };
        if (installed.lifecycle_pending) return { status: statusHtml(escapeHtml(installed.lifecycle_kind === 'update' ? 'Updating…' : 'Working…'), 'muted', { work: true }), action: '' };
        if (/Reviewing/.test(model.chip)) {
            return { status: statusHtml('Reviewing · you can skip', 'muted', { work: true }), action: skipReviewButtonHtml(`data-browse-action="skip" data-item="${id}"`) };
        }
        if (model.needs && !retryUpdate) return { status: statusHtml(`Installed ${v} · needs you`, 'warn'), action: act('Open in Installed', 'installed') };
        if (update) {
            const same = update.version && update.version === installed.install?.release?.version;
            return { status: statusHtml(same ? `Installed ${v} · new files for the same version` : `Installed ${v} · ${escapeHtml(versionText(update.version))} available`, 'muted'), action: act('Update', 'update') };
        }
        if (installed.enabled) return { status: statusHtml(`On · ${v}`, 'ok'), action: model.widgets ? act('Open widgets', 'widgets', 'ghost') : act('Details', 'open', 'ghost') };
        return { status: statusHtml(`Off · ${v}`, 'muted'), action: act('Turn on', 'turn_on') };
    }
    if (item.withdrawn) return { status: statusHtml(`Withdrawn by ${escapeHtml(item.source_name || 'its source')}`, 'muted'), action: '' };
    return { status: statusHtml(escapeHtml(trustSentence(source)), 'muted'), action: act('Install', 'install', 'primary') };
}

/** One Browse card: the shared head, the facts, and the consequence row. */
export function renderBrowseCard(item, { source, state, categoryLabels = new Map(), showCategory = true } = {}) {
    const id = escapeHtml(item.id);
    const long = String(item.description || '').length > CLAMP_CHARS;
    const marks = [
        item.source_id === 'builtin' ? chipHtml('Built in', 'neutral') : '',
        item.official ? officialMarkHtml({ sourceId: item.source_id }) : '',
    ].join('');
    const facts = factsText(item);
    const connects = connectsText(item);
    const yours = countsAsYoursText(item);
    const cats = [connects ? chipHtml(connects, 'neutral') : '', ...(showCategory ? (item.category || [])
        .filter((cat) => cat !== CONNECTORS || !connects).map((cat) => chipHtml(categoryLabels.get(cat) || cat, 'neutral')) : [])].join('');
    const icon = connects ? `<span class="skills-card-icon" aria-hidden="true">${skillIconHtml(item.icon, 18)}</span>` : '';
    return `<article class="marketplace-card skills-head-card" data-item="${id}" data-source="${escapeHtml(item.source_id)}">
        <div class="skills-head-title">${icon}<button type="button" class="skills-card-name" data-browse-action="open" data-item="${id}"><span data-i18n-authored>${escapeHtml(skillTitle(item))}</span><span class="skills-card-name-more" aria-hidden="true">›</span></button>${marks}</div>
        <p class="skills-card-purpose"${long ? ' data-clamped' : ''} data-i18n-authored>${escapeHtml(item.description || '')}</p>
        ${long ? `<button type="button" class="skills-read-more" data-browse-action="open" data-item="${id}">Read more ›</button>` : ''}
        <div class="skills-card-meta-lines">
            <div class="skills-card-who"><span data-i18n-authored>${escapeHtml(item.name)}</span> · ${catalogWhoHtml(item)}</div>
            ${facts ? `<div class="skills-card-facts">${escapeHtml(facts)}</div>` : ''}
            ${yours ? `<div class="skills-card-facts">${escapeHtml(yours)}</div>` : ''}
            ${cats ? `<div class="skills-card-cats">${cats}</div>` : ''}
        </div>
        <div class="settings-action-row skills-card-action-row">${state.status}<div class="skills-card-actions-end">${state.action || ''}</div></div>
        ${state.note ? `<p class="skills-card-note">${state.note}</p>` : ''}
    </article>`;
}

function groupHeadHtml(title, { copy = '', actions = '', count = '' } = {}) {
    return `<div class="skills-group-head"><div class="skills-group-title"><h3>${escapeHtml(title)}${count !== '' ? ` <span class="skills-group-count">${escapeHtml(String(count))}</span>` : ''}</h3>${copy ? `<p>${copy}</p>` : ''}</div>${actions ? `<div class="skills-group-actions">${actions}</div>` : ''}</div>`;
}

/**
 * Mount the Browse view. `host` = {
 *   installedByName(): Map|null, refreshInstalled(): Promise, sources(force): Promise<sources[]>,
 *   catalog({refresh}): Promise<catalog>, runInstalledAction(name, action): Promise,
 *   confirmFirstStart(sourceId): Promise, compare(item, installedRow): Promise<'replace'|'keep'|null>,
 *   toast(msg, tone), showInstalled(name) }.
 */
export function createSkillBrowse({ pane, chrome, host, api = skillSourcesApi }) {
    const state = {
        query: '', source: 'all', sort: 'popular', officialOnly: false, category: 'all',
        catalog: null, catalogError: '', sources: [], search: null, searching: false, searchError: '',
        pending: new Map(), failures: new Map(), sheet: null,
    };
    let destroyed = false;
    // A catalog read and a registry search each keep only their own newest answer.
    let loadGeneration = 0;
    let searchGeneration = 0;
    const disposers = [];
    chrome.innerHTML = `<div class="skills-browse-controls">
            <input type="search" class="ui-control skills-search" data-browse-search placeholder="Search skills" aria-label="Search skills" autocomplete="off" enterkeyhint="search">
            <label class="ui-field-inline skills-browse-filter"><span>Source</span><select class="ui-control" data-browse-source aria-label="Source"><option value="all">All sources</option></select></label>
            <label class="ui-field-inline skills-browse-filter"><span>Sort</span><select class="ui-control" data-browse-sort aria-label="Sort"></select></label>
            <label class="skills-official-toggle"><span class="skills-switch"><input type="checkbox" role="switch" data-browse-official aria-checked="false"><span class="skills-switch-track" aria-hidden="true"><span class="skills-switch-thumb"></span></span></span><span>Official only</span></label>
        </div>`;
    pane.innerHTML = '<div class="skills-browse"><div class="skills-category-strip" data-browse-categories role="group" aria-label="Categories"></div><div class="ui-status skills-browse-status" data-browse-status role="status" aria-live="polite"></div><div data-browse-board></div></div>';
    const searchInput = chrome.querySelector('[data-browse-search]');
    const sourceSelect = chrome.querySelector('[data-browse-source]');
    const sortSelect = chrome.querySelector('[data-browse-sort]');
    const officialToggle = chrome.querySelector('[data-browse-official]');
    const statusEl = pane.querySelector('[data-browse-status]');
    const board = pane.querySelector('[data-browse-board]');
    const categoriesEl = pane.querySelector('[data-browse-categories]');

    const sourceById = () => new Map(state.sources.map((source) => [source.id, source]));
    const categoryLabels = () => new Map([...(state.catalog?.categories || []).map((cat) => [cat.id, cat.label || cat.id]), [CONNECTORS, 'Connectors']]);
    const catalogItems = () => (Array.isArray(state.catalog?.items) ? state.catalog.items : []);
    const allKnownItems = () => {
        const seen = new Map(catalogItems().map((item) => [item.id, item]));
        for (const item of state.search?.items || []) if (!seen.has(item.id)) seen.set(item.id, item);
        return seen;
    };

    function paintSortOptions() {
        // «Recently updated» needs the entries' dates; a catalog without them offers no such sort.
        const dated = [...allKnownItems().values()].some((item) => Date.parse(item.updated_at || ''));
        const options = [...(state.query ? [['best', 'Best match']] : []), ['popular', 'Popular'], ['name', 'Name'],
            ...(dated ? [['updated', 'Recently updated']] : [])];
        if (!options.some(([value]) => value === state.sort)) state.sort = options[0][0];
        sortSelect.innerHTML = options.map(([value, label]) => `<option value="${value}"${value === state.sort ? ' selected' : ''}>${escapeHtml(label)}</option>`).join('');
    }

    function paintSourceOptions() {
        const counts = new Map();
        for (const item of catalogItems()) counts.set(item.source_id, (counts.get(item.source_id) || 0) + 1);
        const options = [['all', 'All sources'], ...state.sources.map((source) => [source.id,
            source.kind === 'registry' ? `${source.name} (search)` : `${source.name} (${counts.get(source.id) || 0})`])];
        if (!options.some(([value]) => value === state.source)) state.source = 'all';
        sourceSelect.innerHTML = options.map(([value, label]) => `<option value="${escapeHtml(value)}"${value === state.source ? ' selected' : ''}>${escapeHtml(label)}</option>`).join('');
    }

    function paintCategories() {
        const categories = (Array.isArray(state.catalog?.categories) ? state.catalog.categories : []).filter((cat) => cat.id !== CONNECTORS);
        const counts = categoryCounts(catalogItems(), categories);
        const connectors = catalogItems().filter(isConnector).length;
        const chips = [['all', 'All', catalogItems().length], ...(connectors ? [[CONNECTORS, 'Connectors', connectors]] : []),
            ...categories.filter((cat) => counts.get(cat.id)).map((cat) => [cat.id, cat.label || cat.id, counts.get(cat.id)])];
        if (!chips.some(([id]) => id === state.category)) state.category = 'all';
        categoriesEl.hidden = chips.length < 2;
        categoriesEl.innerHTML = chips.map(([id, label, count]) => `<button type="button" class="filter-chip skills-category${id === state.category ? ' active' : ''}" aria-pressed="${id === state.category ? 'true' : 'false'}" data-browse-category="${escapeHtml(id)}">${escapeHtml(label)}<span class="skills-category-count">${count}</span></button>`).join('');
    }

    function visibleItems() {
        const labels = categoryLabels();
        const map = new Map();
        for (const item of catalogItems()) if (matchesQuery(item, state.query, labels)) map.set(item.id, item);
        if (state.query && state.search) for (const item of state.search.items || []) if (!map.has(item.id)) map.set(item.id, item);
        return [...map.values()].filter((item) => (state.source === 'all' || item.source_id === state.source)
            && (!state.officialOnly || item.official)
            && (state.category === 'all' || (state.category === CONNECTORS ? isConnector(item) : (item.category || []).includes(state.category))));
    }

    // The registry a search also asks (ClawHub by default); none connected, none asked.
    const registry = () => state.sources.find((source) => source.kind === 'registry'
        && (state.source === 'all' || state.source === source.id)) || null;

    function statusLine(count) {
        const parts = [];
        const sourceNames = new Set(visibleItems().map((item) => item.source_id));
        const asked = escapeHtml(registry()?.name || '');
        if (!state.catalog && !state.catalogError) return 'Reading the catalogs…';
        if (state.catalogError && !state.catalog) return `Couldn’t read the catalogs: ${escapeHtml(state.catalogError)}. Refresh to try again.`;
        if (state.catalogError) parts.push(`Couldn’t refresh the catalogs: ${escapeHtml(state.catalogError)}. Showing the earlier list`);
        if (state.query) {
            parts.push(`${plural(count, 'skill')} match “${escapeHtml(state.query)}”`);
            if (state.searching) parts.push(`asking ${asked}…`);
            else if (state.searchError) parts.push(`${asked} couldn’t be searched: ${escapeHtml(state.searchError)}`);
            else if (state.search?.registry_error) parts.push(`couldn’t search ${escapeHtml(state.search.registry_error)}`);
            else if (state.search) parts.push(`${asked} answered`);
            else if (asked) parts.push(`press Enter to search ${asked} too`);
        } else if (state.category !== 'all') {
            parts.push(`${plural(count, 'skill')} in ${escapeHtml(categoryLabels().get(state.category) || state.category)}`);
        } else {
            parts.push(`${plural(count, 'skill')} from ${plural(sourceNames.size, 'source')}`);
            if (asked) parts.push(`${asked} joins the list when you search`);
        }
        // A failed fetch keeps the last list (state ready, the failure in `error`); the line says both.
        for (const source of state.catalog?.sources || []) {
            if (source.state === 'error' || (source.state === 'ready' && source.error)) {
                const age = source.state === 'ready' ? relativeAge(source.last_read_at) : '';
                parts.push(`${escapeHtml(source.name)} couldn’t be read just now (${escapeHtml(source.error || 'no answer')})${age ? `. Showing its skills from ${escapeHtml(age)}` : ''}`);
            } else if (source.state === 'reading') parts.push(`reading ${escapeHtml(source.name)}…`);
        }
        return parts.join(' · ');
    }

    function stateFor(item) {
        const listing = host.installedByName();
        const pending = state.pending.get(item.id) || null;
        return browseCardState(item, {
            source: sourceById().get(item.source_id) || null,
            installed: listing?.get(item.name) || null, listed: Boolean(listing),
            pending, failure: state.failures.get(item.id) || '',
        });
    }

    function gridHtml(items, key, showCategory) {
        const labels = categoryLabels();
        return `<div class="skills-board" data-browse-grid="${key}">${items.map((item) => renderBrowseCard(item, {
            source: sourceById().get(item.source_id), state: stateFor(item), categoryLabels: labels, showCategory,
        })).join('')}</div>`;
    }

    function layoutGrids() {
        for (const grid of board.querySelectorAll('[data-browse-grid]')) {
            grid.setAttribute('data-masonry', '');
            const order = Array.from(grid.querySelectorAll(':scope > [data-item]')).map((node) => node.dataset.item);
            disposers.push(applyMasonry(grid, { itemSelector: ':scope > .marketplace-card', gap: 12, minColumnWidth: 380,
                keyOf: (node) => node.dataset.item || '', order }));
        }
    }

    function render() {
        if (destroyed) return;
        paintSourceOptions();
        paintSortOptions();
        paintCategories();
        officialToggle.checked = state.officialOnly;
        officialToggle.setAttribute('aria-checked', state.officialOnly ? 'true' : 'false');
        while (disposers.length) disposers.pop()?.();
        const items = sortCatalogItems(visibleItems(), state.sort, state.query);
        statusEl.innerHTML = statusLine(items.length);
        statusEl.dataset.tone = state.catalogError && !state.catalog ? 'danger' : 'muted';
        const official = state.sources.find((source) => source.kind === 'hub' && source.part === 'official');
        const shelf = shelfItems(state.catalog, host.installedByName(), state);
        // Shown once: Hide dismisses the shelf on this device for good (SPEC §8.2).
        const showShelf = shelf.length > 0 && !readHiddenShelf();
        let html = '';
        if (showShelf) {
            const copy = official?.trust === 'trusted'
                ? `Picked by ${escapeHtml(official.name)}. They install without review — you trust ${escapeHtml(official.name)}.`
                : `Picked by ${escapeHtml(official?.name || 'Ouroboros')}. Ouroboros reviews each one before it runs.`;
            html += groupHeadHtml('Recommended to start', { copy, actions: `${buttonHtml(`Install all ${shelf.length}`, { tone: 'primary', attrs: 'data-browse-action="install_all"' })}${buttonHtml('Hide', { tone: 'ghost', attrs: 'data-browse-action="hide_shelf"' })}` });
            html += gridHtml(sortCatalogItems(shelf, 'popular'), 'recommended', state.category === 'all');
            const shown = new Set(shelf.map((item) => item.id));
            html += groupHeadHtml('All skills');
            html += gridHtml(items.filter((item) => !shown.has(item.id)), 'all', state.category === 'all');
        } else if (items.length) {
            html += gridHtml(items, 'all', state.category === 'all');
        } else if (state.catalog) {
            html += `<p class="skills-empty">${state.sources.length ? (state.query ? 'No skill matches this search.' : 'Nothing to show with these filters.') : 'No sources connected, so there is nothing to browse. Skills made here still work.'}</p>`;
        }
        board.innerHTML = html;
        layoutGrids();
    }

    /** Repaint only the action rows (installed listing changed); scroll position and grids stay. */
    function repaintStates() {
        if (destroyed) return;
        const known = allKnownItems();
        for (const card of board.querySelectorAll('[data-item].marketplace-card')) {
            const item = known.get(card.dataset.item);
            if (!item) continue;
            const installed = host.installedByName()?.get(item.name);
            const pending = state.pending.get(item.id);
            if (pending && installed && !installed.lifecycle_virtual && (pending.phase === 'installing'
                || installed.enabled || installed.review_status !== 'pending' || installed.skill_review?.current?.lifecycle_status)) {
                state.pending.delete(item.id);
            }
            const next = stateFor(item);
            if (state.sheet && !state.sheet.handle.closed && state.sheet.item.id === item.id) state.sheet.handle.setActions(sheetActions(item));
            const row = card.querySelector('.skills-card-action-row');
            const html = `${next.status}<div class="skills-card-actions-end">${next.action || ''}</div>`;
            if (row && row.innerHTML !== html) row.innerHTML = html;
            let note = card.querySelector(':scope > .skills-card-note');
            if (next.note && !note) {
                note = document.createElement('p');
                note.className = 'skills-card-note';
                card.appendChild(note);
            }
            if (note) {
                if (next.note) note.innerHTML = next.note;
                else note.remove();
            }
        }
    }

    async function load({ refresh = false } = {}) {
        const current = ++loadGeneration;
        const [catalog, sources] = await Promise.allSettled([host.catalog({ refresh }), host.sources(refresh)]);
        if (destroyed || current !== loadGeneration) return;
        if (catalog.status === 'fulfilled') { state.catalog = catalog.value; state.catalogError = ''; }
        else state.catalogError = catalog.reason?.message || String(catalog.reason);
        if (sources.status === 'fulfilled') state.sources = Array.isArray(sources.value) ? sources.value : [];
        render();
    }

    async function searchRegistry() {
        const query = state.query;
        if (!query || !registry()) return;
        const current = ++searchGeneration;
        state.searching = true;
        state.searchError = '';
        render();
        try {
            const result = await api.search({ q: query, source: state.source === 'all' ? '' : state.source });
            if (destroyed || current !== searchGeneration || query !== state.query) return;
            state.search = result && typeof result === 'object' ? result : { items: [] };
        } catch (err) {
            if (destroyed || current !== searchGeneration) return;
            state.searchError = err?.message || String(err);
        } finally {
            if (!destroyed && current === searchGeneration) { state.searching = false; render(); }
        }
    }

    async function install(item, { replace = false } = {}) {
        state.failures.delete(item.id);
        state.pending.set(item.id, { phase: 'installing', startedAt: Date.now() });
        repaintStates();
        try {
            const result = await api.install(item.id, { turnOnWhenReady: true, replace });
            if (result?.already_installed) {
                state.pending.delete(item.id);
                host.toast(`${item.name} is already installed from ${item.source_name || 'this source'}.`, 'muted');
            } else {
                state.pending.set(item.id, { phase: result?.basis === 'review' ? 'reviewing' : 'installing', startedAt: Date.now() });
                host.toast(result?.basis === 'review'
                    ? `${item.name} is installing. Ouroboros reviews it before it runs.`
                    : `${item.name} is installing. It turns on as soon as it is ready.`, 'ok');
            }
        } catch (err) {
            state.pending.delete(item.id);
            // A typed refusal about what is already installed is said once; the card then shows
            // Compare… or Update from the refreshed listing. Anything else failed this install.
            const code = err?.body?.code || '';
            if (['name_taken', 'name_collision', 'update_instead'].includes(code)) host.toast(`${item.name}: ${err.message}`, 'warn');
            else state.failures.set(item.id, err?.message || String(err));
        }
        repaintStates();
        await host.refreshInstalled();
        repaintStates();
    }

    // The sheet's head mirrors the card's one action; «Details» would open what is already open.
    const sheetActions = (item) => {
        const html = stateFor(item).action || '';
        return /data-browse-action="open"/.test(html) ? '' : html.replaceAll('data-browse-action=', 'data-sheet-action=');
    };

    function openSheet(item, opener) {
        state.sheet?.handle.close();
        const handle = openSkillDetailSheet({
            item, installed: host.installedByName()?.get(item.name) || null,
            source: sourceById().get(item.source_id) || null,
            actionsHtml: sheetActions(item),
            // A registry hit has no entry to read until it is installed; its card is all there is.
            readDetail: sourceById().get(item.source_id)?.kind === 'registry' || item.registry_slug ? null
                : ({ signal } = {}) => api.item(item.source_id, item.name, { signal }),
            onAction: (action, button) => act(action, button),
            returnFocus: opener,
        });
        state.sheet = { item, handle };
    }

    async function act(action, node) {
        const item = allKnownItems().get(node?.dataset?.item || '');
        if (action === 'hide_shelf') { writeHiddenShelf('1'); render(); return; }
        if (action === 'install_all') {
            const missing = shelfItems(state.catalog, host.installedByName(), state);
            await host.confirmFirstStart?.();
            for (const entry of missing) await install(entry);
            return;
        }
        if (!item) return;
        const installed = host.installedByName()?.get(item.name) || null;
        if (action === 'open') { openSheet(item, node); return; }
        if (action === 'install') { await host.confirmFirstStart?.(item.source_id); await install(item); return; }
        if (action === 'compare') {
            // Replace is one host operation: the other copy goes only when this one has landed.
            if (await host.compare(item, installed) === 'replace') await install(item, { replace: true });
            return;
        }
        if (action === 'installed') { host.showInstalled(item.name); return; }
        const map = { skip: 'attest_review', update: 'update', turn_on: 'turn_on', widgets: 'open_widgets' };
        if (map[action]) {
            await host.runInstalledAction(item.name, map[action]);
            repaintStates();
        }
    }

    const onPaneClick = (event) => {
        const category = event.target.closest('[data-browse-category]');
        if (category) { state.category = category.dataset.browseCategory; render(); return; }
        const target = event.target.closest('[data-browse-action]');
        if (target && !target.disabled) act(target.dataset.browseAction, target);
    };
    let typingTimer = 0;
    const onInput = () => {
        clearTimeout(typingTimer);
        typingTimer = setTimeout(() => {
            const query = searchInput.value.trim();
            if (query === state.query) return;
            state.query = query;
            state.search = null;
            state.searchError = '';
            state.sort = query ? 'best' : 'popular';
            render();
        }, 120);
    };
    const onSearchKey = (event) => {
        if (event.key !== 'Enter' || event.isComposing) return;
        event.preventDefault();
        clearTimeout(typingTimer);
        const query = searchInput.value.trim();
        if (query !== state.query) state.sort = query ? 'best' : 'popular';
        state.query = query;
        state.search = null;
        render();
        searchRegistry();
    };
    const onSource = () => { state.source = sourceSelect.value; state.search = null; render(); if (state.query) searchRegistry(); };
    const onSort = () => { state.sort = sortSelect.value; render(); };
    const onOfficial = () => { state.officialOnly = officialToggle.checked; render(); };
    pane.addEventListener('click', onPaneClick);
    searchInput.addEventListener('input', onInput);
    searchInput.addEventListener('keydown', onSearchKey);
    sourceSelect.addEventListener('change', onSource);
    sortSelect.addEventListener('change', onSort);
    officialToggle.addEventListener('change', onOfficial);

    // The newest read of the catalogs and sources: a search waits for it to know which registry to ask.
    let loaded = Promise.resolve();
    return {
        show: (options) => (loaded = load(options)),
        refresh: () => (loaded = load({ refresh: true })),
        repaintStates,
        async searchFor(name) {
            searchInput.value = name;
            state.query = name;
            state.search = null;
            state.sort = 'best';
            render();
            await loaded;
            searchRegistry();
        },
        itemFor(sourceId, name) { return catalogItems().find((item) => item.source_id === sourceId && item.name === name) || null; },
        /** Show one category, Connectors included (the place Connect flow's way here). */
        async showCategory(id) { await loaded; state.category = id; render(); },
        catalogItems,
        destroy() {
            destroyed = true;
            state.sheet?.handle.close();
            clearTimeout(typingTimer);
            while (disposers.length) disposers.pop()?.();
            pane.removeEventListener('click', onPaneClick);
            searchInput.removeEventListener('input', onInput);
            searchInput.removeEventListener('keydown', onSearchKey);
            sourceSelect.removeEventListener('change', onSource);
            sortSelect.removeEventListener('change', onSort);
            officialToggle.removeEventListener('change', onOfficial);
        },
    };
}
