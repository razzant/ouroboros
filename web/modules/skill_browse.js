// Extensions › Browse (UX §2, boards B2, B2c, B5, B5b, B14): every connected
// source's catalog as one board of cards in sort order. Search filters the
// loaded catalogs while typing; Enter also asks the registry (ClawHub), whose
// hits join the board. A registry chosen alone with nothing typed shows its own
// page, in its own order, with Show more; in All sources it only joins a search,
// so opening Browse never waits for it. Each card says the consequence of its one action before
// it is pressed. The list never waits for GitHub: it is the host's cache, and a
// source that could not be read says so in the status line without blanking
// the others. Install returns at once; the card follows the skill's real state.
import { applyMasonry } from './masonry.js';
import { bindScrollFade } from './scroll_fade.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { skillSourcesApi } from './api_client.js';
import { installedSkillModel } from './skill_card_renderer.js';
import { openSkillDetailSheet } from './skill_detail_sheet.js';
import {
    buttonHtml, catalogWhoHtml, chipHtml, connectsServices, connectsText, countsAsYoursText, factsText, nameMatch, officialMarkHtml, plural,
    relativeAge, skillIconHtml, skillNames, skillOfferChanged, skipReviewButtonHtml, statusHtml, trustSentence, versionText,
    skillTitle,
} from './skill_presentation.js';

const HIDDEN_KEY = 'ouroboros.skills.recommendedHidden';
// The Connectors chip is derived from each entry's `connects` fact (an installed skill's place declaration, or
// the Hub entry's), never a typed tag; a catalog category of the same id joins it.
export const CONNECTORS = 'connectors';
export const isConnector = (item) => connectsServices(item).length > 0 || (item?.category || []).includes(CONNECTORS);
const READ_MORE_CHARS = 600;

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
        // The card whose printed title or id is what was typed comes first, then one that starts with it.
        const rank = (item) => Math.min(nameMatch(query, skillNames(item)), 2);
        return list.sort((a, b) => rank(a) - rank(b) || votesOf(b) - votesOf(a) || byName(a, b));
    }
    return list.sort((a, b) => votesOf(b) - votesOf(a) || byName(a, b));
}

/** Local text filter over the loaded catalogs: the card's names as typed (title or id, space, `_` and `-` alike),
 *  else the words as typed in its name, description, author, source or category labels. */
export function matchesQuery(item, query, categoryLabels = new Map()) {
    const q = String(query || '').trim().toLowerCase();
    if (!q) return true;
    const cats = (item.category || []).map((id) => categoryLabels.get(id) || id);
    return nameMatch(q, skillNames(item)) < 3 || [item.name, item.description, item.owner, item.source_name, ...cats]
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
    // Install all is for vacant names, not an occupied name or one whose install is still in flight.
    return items.filter((item) => item.recommended && !item.withdrawn
        && !installedByName.has(item.name) && !resolveInstalled(item, installedByName).installed);
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

const hasSourceKey = (row) => row != null && Object.hasOwn(row, 'source_key');
const sameSource = (a, b) => typeof a?.source_key === 'string' && a.source_key !== '' && a.source_key === b?.source_key;

export function installedHere(item, row) {
    if (!row || row.lifecycle_virtual || row.identity_collision) return false;
    if (hasSourceKey(item) || hasSourceKey(row)) {
        const local = row.source === 'user_repo' || row.install?.source?.kind === 'local';
        return sameSource(item, row) && (item.registry_slug ? row.registry_slug === item.registry_slug
            : row.name === item.name || (local && Boolean(row.catalog_id && row.catalog_id === item.id)));
    }
    if (item.registry_slug) {
        const source = row.install?.source;
        return source?.kind === 'registry' && source.id === item.source_id && row.provenance?.slug === item.registry_slug;
    }
    // Only an old host uses the detail address to relate the installed copy to a catalog entry.
    return row.catalog_id !== undefined ? row.catalog_id === item.id
        : row.name === item.name && (!row.install || row.install.source?.id === item.source_id);
}

/** Registry leaf names are not local install identities. Git can still offer a named occupant for comparison. */
function resolveInstalled(item, listing) {
    const occupant = listing?.get(item.name);
    if (installedHere(item, occupant)) return { installed: occupant };
    if (item.registry_slug || item.source_key || !hasSourceKey(item)) {
        for (const row of listing?.values() || []) if (installedHere(item, row)) return { installed: row };
    }
    // Equal bytes with unknown installed origin allow a jump to that card, not an installed action or a replacement.
    if (occupant && originUnknown(occupant) && item.digest && item.digest === occupant.content_hash) return { sameFiles: occupant };
    // So does a copy of unknown origin the listing itself ties to this entry (its detail address, a registry's full
    // slug): Install or Compare could only fail on it or act on an origin nobody can name.
    const tied = [...(listing?.values() || [])].find((row) => originUnknown(row)
        && ((row.catalog_id && row.catalog_id === item.id) || (item.registry_slug && row.registry_slug === item.registry_slug && row.install?.source?.id === item.source_id)));
    if (tied) return { tied };
    return { other: !item.registry_slug && !occupant?.lifecycle_virtual ? occupant : null };
}

/** An installed copy on a host that names origins, whose own origin it cannot name. */
function originUnknown(row) {
    return hasSourceKey(row) && !row.source_key && !row.lifecycle_virtual && !row.identity_collision;
}

export function offeredUpdate(item, installed) {
    const update = installed?.install?.update_available;
    return update?.digest && item.digest === update.digest ? update : null;
}

/**
 * How a new install of an entry goes, as its catalog read it: `trust` is its source's setting then. The host reads the
 * source again at Install and installs by what it finds, so the sentences built on this say «while» (trustSentence).
 * Whether the owner confirmed that setting is said only by a catalog row that is the same source with the same
 * setting: a registry hit comes from another answer than `catalog.sources`, and an id may name another source by then.
 */
export function installPolicy(item, catalogSources = []) {
    const row = (catalogSources || []).find((each) => each.id === item.source_id && each.name === item.source_name && each.trust === item.trust);
    return { trust: item.trust, name: item.source_name, confirmed: row?.confirmed };
}

/**
 * One Browse card's action row: a consequence sentence and one action, from the
 * catalog item, how it installs (`policy`), the installed listing (`listed`: it has
 * been read) and the installs started here.
 */
export function browseCardState(item, { policy = installPolicy(item), installed = null, listed = true, pending = null, failure = '', reviewing = false, live = {}, pageModules = {} } = {}) {
    return cardState(item, { policy, listed, pending, failure, reviewing, live, pageModules,
        ...resolveInstalled(item, installed ? new Map([[installed.name, installed]]) : null) });
}

function cardState(item, { policy, installed, other, sameFiles, tied, listed, pending, failure, reviewing, live, pageModules }) {
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
    if (!listed && item.installed && !(item.registry_slug && item.installed.state === 'name_taken')) {
        // The catalog's own join, until the installed listing (which carries every action) has been read.
        return item.installed.state === 'name_taken'
            ? { status: statusHtml(`You have another “${escapeHtml(item.name)}” (from ${escapeHtml(item.installed.source_name || 'another source')})`, 'warn'), action: '' }
            : { status: statusHtml(`Installed · ${escapeHtml(versionText(item.installed.version))}`, 'muted'), action: '' };
    }
    if (other) {
        const source = other.install?.source?.name || 'another source';
        return { status: statusHtml(`You have another “${escapeHtml(item.name)}” (from ${escapeHtml(source)})`, 'warn'), action: act('Compare…', 'compare') };
    }
    if (sameFiles || tied) {
        return { status: statusHtml(sameFiles ? 'Same files installed · source unknown' : 'Installed · source unknown', 'muted'),
            action: act('Open in Installed', 'installed') };
    }
    if (installed) {
        const model = installedSkillModel(installed, { reviewing, live, pageModules });
        const v = escapeHtml(versionText(installed.version));
        const update = offeredUpdate(item, installed);
        const retryUpdate = update && installed.install?.update_rejected;
        if (model.modified && !retryUpdate && !model.needs) return { status: statusHtml(`Installed · modified here`, 'muted'), action: act('Details', 'open', 'ghost') };
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
    return { status: statusHtml(escapeHtml(trustSentence(policy)), 'muted'), action: act('Install', 'install', 'primary') };
}

/** One Browse card: the shared head, the facts, and the consequence row. */
export function renderBrowseCard(item, { state, categoryLabels = new Map(), showCategory = true } = {}) {
    const id = escapeHtml(item.id);
    const long = String(item.description || '').length > READ_MORE_CHARS;
    const marks = [
        item.source_id === 'builtin' ? chipHtml('Built in', 'neutral') : '',
        item.official ? officialMarkHtml() : '',  // explained from this entry (`officialSource`), never a source looked up by id
    ].join('');
    const facts = factsText(item);
    const connects = connectsText(item);
    const yours = countsAsYoursText(item);
    const cats = [connects ? chipHtml(connects, 'neutral') : '', ...(showCategory ? (item.category || [])
        .filter((cat) => cat !== CONNECTORS || !connects).map((cat) => chipHtml(categoryLabels.get(cat) || cat, 'neutral')) : [])].join('');
    const icon = connects ? `<span class="skills-card-icon" aria-hidden="true">${skillIconHtml(item.icon, 18)}</span>` : '';
    return `<article class="ui-card marketplace-card skills-head-card" data-item="${id}" data-source="${escapeHtml(item.source_id)}">
        <div class="skills-head-title">${icon}<button type="button" class="skills-card-name ui-hit-area" data-browse-action="open" data-item="${id}"><span data-i18n-authored>${escapeHtml(skillTitle(item))}</span><span class="skills-card-name-more" aria-hidden="true">›</span></button>${marks}</div>
        <p class="skills-card-purpose"${long ? ' data-browse-purpose-preview' : ''} data-i18n-authored>${escapeHtml(item.description || '')}</p>
        ${long ? `<button type="button" class="skills-read-more ui-hit-area" data-browse-action="open" data-item="${id}">Read more ›</button>` : ''}
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
 *   installedByName(): Map|null, live(): object, refreshInstalled(): Promise, sources(options): Promise<sources[]>,
 *   sourceSnapshot(): { rows } (the shared source read: it names the filters and the registry; what a card says of
 *   an install comes from its catalog entry, so a list that settles later changes no card, sheet or dialog),
 *   catalog({refresh}): Promise<catalog>, runInstalledAction(name, action, options): Promise,
 *   confirmFirstStart(sourceId): Promise, compare(item, installedRow): Promise<'replace'|'keep'|null>,
 *   toast(msg, tone), showInstalled(name, opener) }.
 */
export function createSkillBrowse({ pane, chrome, host, api = skillSourcesApi }) {
    const state = {
        query: '', source: 'all', sort: 'popular', officialOnly: false, category: 'all',
        catalog: null, catalogError: '', search: null, searching: false, searchError: '',
        browse: null,  // one registry's own pages: {sort, items, cursor, loading, more, error}; dropped with its source
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
            <label class="skills-official-toggle"><span class="ui-switch"><input type="checkbox" role="switch" data-browse-official aria-checked="false"><span class="ui-switch-track" aria-hidden="true"><span class="ui-switch-thumb"></span></span></span><span>Official only</span></label>
        </div>`;
    pane.innerHTML = '<div class="skills-browse"><div class="skills-category-strip scroll-fade-x" data-browse-categories role="group" aria-label="Categories" data-scroll-x></div><div class="ui-status skills-browse-status" data-browse-status role="status" aria-live="polite"></div><div data-browse-board></div><div class="skills-browse-more" data-browse-more hidden><button type="button" class="btn btn-ghost btn-sm" data-browse-action="browse_more">Show more</button></div></div>';
    const searchInput = chrome.querySelector('[data-browse-search]');
    const sourceSelect = chrome.querySelector('[data-browse-source]');
    const sortSelect = chrome.querySelector('[data-browse-sort]');
    const officialToggle = chrome.querySelector('[data-browse-official]');
    const statusEl = pane.querySelector('[data-browse-status]');
    const board = pane.querySelector('[data-browse-board]');
    const categoriesEl = pane.querySelector('[data-browse-categories]');
    const moreEl = pane.querySelector('[data-browse-more]');
    const moreButton = moreEl.querySelector('button');
    if (categoriesEl) bindScrollFade(categoriesEl, { axis: 'x' });  // a phone's one scrolling line fades where chips hide

    const sourceRows = () => host.sourceSnapshot().rows || [];
    const policyOf = (item) => installPolicy(item, state.catalog?.sources);
    const categoryLabels = () => new Map([...(state.catalog?.categories || []).map((cat) => [cat.id, cat.label || cat.id]), [CONNECTORS, 'Connectors']]);
    const catalogItems = () => (Array.isArray(state.catalog?.items) ? state.catalog.items : []);
    const allKnownItems = () => {
        const seen = new Map(catalogItems().map((item) => [item.id, item]));
        for (const item of [...(state.search?.items || []), ...(state.browse?.items || [])]) if (!seen.has(item.id)) seen.set(item.id, item);
        return seen;
    };
    // The row each card on the board was drawn from: a card opens and reports the row it shows, so a listing row and
    // a search row that share an id but differ in title never redirect each other's card.
    let drawnById = new Map();
    function sourcePresent(item) {
        const { rows, error, problem } = host.sourceSnapshot();
        return Array.isArray(rows) && !error && !problem ? rows.some((row) => row.id === item.source_id) : null;
    }
    function staleSource(item, fromSheet = false) {
        const shown = fromSheet && state.sheet?.item === item ? state.sheet : null;
        if (shown?.handle.offerChanged || (shown?.sourceSeen && sourcePresent(item) === false)) return true;
        const current = drawnById.get(item.id) || allKnownItems().get(item.id);
        if (current && current !== item && (skillOfferChanged(item, current)
            || ((hasSourceKey(item) || hasSourceKey(current)) && !sameSource(item, current)))) return true;
        const row = shown?.installedName ? host.installedByName()?.get(shown.installedName) : null;
        return row && (hasSourceKey(item) || hasSourceKey(row)) && !installedHere(item, row);
    }
    // The registry shown on its own: chosen by name, with nothing typed.
    const browsedRegistry = () => (state.query ? null : sourceRows().find((source) => source.kind === 'registry' && source.id === state.source) || null);

    function paintSortOptions() {
        // «Recently updated» needs the entries' dates; a catalog without them offers no such sort.
        const dated = [...allKnownItems().values()].some((item) => Date.parse(item.updated_at || ''));
        // A registry's own pages come in the orders it can give them.
        const options = browsedRegistry() ? [['popular', 'Popular'], ['updated', 'Recently updated']]
            : [...(state.query ? [['best', 'Best match']] : []), ['popular', 'Popular'], ['name', 'Name'],
                ...(dated ? [['updated', 'Recently updated']] : [])];
        if (!options.some(([value]) => value === state.sort)) state.sort = options[0][0];
        sortSelect.innerHTML = options.map(([value, label]) => `<option value="${value}"${value === state.sort ? ' selected' : ''}>${escapeHtml(label)}</option>`).join('');
    }

    function paintSourceOptions() {
        const counts = new Map();
        for (const item of catalogItems()) counts.set(item.source_id, (counts.get(item.source_id) || 0) + 1);
        const options = [['all', 'All sources'], ...sourceRows().map((source) => [source.id,
            source.kind === 'registry' ? source.name : `${source.name} (${counts.get(source.id) || 0})`])];
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
        categoriesEl.hidden = chips.length < 2 || Boolean(browsedRegistry());  // the catalogs' categories say nothing about a registry's page
        categoriesEl.innerHTML = chips.map(([id, label, count]) => `<button type="button" class="filter-chip skills-category${id === state.category ? ' active' : ''}" aria-pressed="${id === state.category ? 'true' : 'false'}" data-browse-category="${escapeHtml(id)}">${escapeHtml(label)}<span class="skills-category-count">${count}</span></button>`).join('');
    }

    function visibleItems() {
        const labels = categoryLabels();
        const map = new Map();
        for (const item of catalogItems()) if (matchesQuery(item, state.query, labels)) map.set(item.id, item);
        if (state.query && state.search) for (const item of state.search.items || []) if (!map.has(item.id)) map.set(item.id, item);
        if (browsedRegistry()) for (const item of state.browse?.items || []) map.set(item.id, item);
        return [...map.values()].filter((item) => (state.source === 'all' || item.source_id === state.source)
            && (!state.officialOnly || item.official)
            && (browsedRegistry() || state.category === 'all' || (state.category === CONNECTORS ? isConnector(item) : (item.category || []).includes(state.category))));
    }

    // The registry a search also asks (ClawHub by default); none connected, none asked.
    const registry = () => sourceRows().find((source) => source.kind === 'registry'
        && (state.source === 'all' || state.source === source.id)) || null;

    function browseStatus(count) {
        const name = escapeHtml(browsedRegistry().name);
        const { items = [], loading = true, more = false, error = '' } = state.browse || {};
        const order = state.sort === 'updated' ? 'recently updated' : 'most downloaded';
        if (error && !more) return `Couldn’t read ${escapeHtml(error)}`;
        if (loading && !more) return `Asking ${name} for its ${order} skills…`;
        if (!count) return items.length ? `No ${name} skill is marked Official` : `${name} answered with no skills`;
        const parts = [`${plural(count, 'skill')} from ${name}, ${order} first`];
        if (error) parts.push(`couldn’t read more from ${escapeHtml(error)}`);
        else if (loading) parts.push('reading more…');
        else parts.push(`type and press Enter to search all of ${name}`);
        return parts.join(' · ');
    }

    function statusLine(count) {
        if (browsedRegistry()) return browseStatus(count);
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

    function stateFor(item, listing = host.installedByName()) {
        const pageModules = host.pageModuleFacts?.() || {};
        const pending = state.pending.get(item.id) || null;
        return cardState(item, {
            policy: policyOf(item),
            ...resolveInstalled(item, listing), listed: Boolean(listing),
            pending, failure: state.failures.get(item.id) || '',
            live: host.live?.(), pageModules,
        });
    }

    function gridHtml(items, key, showCategory) {
        const labels = categoryLabels();
        const listing = host.installedByName();
        return `<div class="skills-board" data-browse-grid="${key}">${items.map((item) => renderBrowseCard(item, {
            state: stateFor(item, listing), categoryLabels: labels, showCategory,
        })).join('')}</div>`;
    }

    function layoutGrids(onLayout = null) {
        for (const grid of board.querySelectorAll('[data-browse-grid]')) {
            grid.setAttribute('data-masonry', '');
            const order = Array.from(grid.querySelectorAll(':scope > [data-item]')).map((node) => node.dataset.item);
            const previews = Array.from(grid.querySelectorAll('[data-browse-purpose-preview]'));
            disposers.push(applyMasonry(grid, { itemSelector: ':scope > .marketplace-card', gap: 12, minColumnWidth: 380,
                keyOf: (node) => node.dataset.item || '', order, onLayout: () => {
                    // Measure after masonry has placed the cards at their current widths. The separate button can
                    // still lead to more material in Details even if the catalog description fits here in full.
                    const clipped = previews.map((node) => node.scrollHeight > node.clientHeight + 1);
                    previews.forEach((node, idx) => node.toggleAttribute('data-fold-clipped', clipped[idx]));
                    onLayout?.();
                } }));
        }
    }

    /** A rebuilt board keeps each grid's height and each card's place until the next layout measures them,
     *  so the scroller never collapses under the reader while a page is appended. */
    function keepGeometry() {
        const names = ['--masonry-w', '--masonry-x', '--masonry-y'];
        const heights = new Map();
        const places = new Map();
        for (const grid of board.querySelectorAll('[data-browse-grid]')) {
            heights.set(grid.dataset.browseGrid, grid.style.getPropertyValue('--masonry-h'));
            for (const card of grid.querySelectorAll(':scope > [data-item]')) {
                places.set(`${grid.dataset.browseGrid}|${card.dataset.item}`, names.map((name) => [name, card.style.getPropertyValue(name)]));
            }
        }
        return () => {
            for (const grid of board.querySelectorAll('[data-browse-grid]')) {
                const height = heights.get(grid.dataset.browseGrid);
                if (height) grid.style.setProperty('--masonry-h', height);
                for (const card of grid.querySelectorAll(':scope > [data-item]')) {
                    for (const [name, value] of places.get(`${grid.dataset.browseGrid}|${card.dataset.item}`) || []) {
                        if (value) card.style.setProperty(name, value);
                    }
                }
            }
        };
    }

    // The name, Read more and the consequence row can all open one item; keep the pressed door, not the first action.
    function currentControl(node) {
        const item = node.closest('.marketplace-card')?.dataset.item;
        const inActionRow = Boolean(node.closest('.skills-card-action-row'));
        const selector = node.matches('.skills-card-name') ? '.skills-card-name'
            : node.matches('.skills-read-more') ? '.skills-read-more' : `[data-browse-action="${CSS.escape(node.dataset.browseAction)}"]`;
        return () => {
            if (board.contains(node)) return node;
            const card = item ? board.querySelector(`article[data-item="${CSS.escape(item)}"]`) : board;
            return (inActionRow ? card?.querySelector('.skills-card-action-row') : card)?.querySelector(selector);
        };
    }

    function keepFocus() {
        const focused = document.activeElement?.closest('[data-browse-action]');
        if (!focused || !board.contains(focused)) return () => {};
        const current = currentControl(focused);
        return () => {
            const target = current();
            if (target && document.activeElement !== target) target.focus({ preventScroll: true });
        };
    }

    function paintMore(visible = !moreEl.hidden) {
        moreEl.hidden = !visible;
        const reading = Boolean(state.browse?.loading && state.browse?.more);
        moreButton.textContent = reading ? 'Reading…' : 'Show more';
        moreButton.setAttribute('aria-busy', reading ? 'true' : 'false');  // stays enabled: a press mid-read is ignored and the keyboard keeps its place
    }

    function shelfCopy(items) {
        if (new Set(items.map((item) => item.source_id)).size !== 1) return 'Picked by Ouroboros · Each card says how it installs';
        const hub = (state.catalog?.sources || []).find((row) => row.id === items[0].source_id && row.kind === 'hub');
        return `Picked by ${hub?.name || 'Ouroboros'} · ${trustSentence(policyOf(items[0]))}`;
    }

    function render() {
        if (destroyed) return;
        paintSourceOptions();
        paintSortOptions();
        paintCategories();
        officialToggle.checked = state.officialOnly;
        officialToggle.setAttribute('aria-checked', state.officialOnly ? 'true' : 'false');
        while (disposers.length) disposers.pop()?.();
        const browsing = browsedRegistry();
        const items = browsing ? visibleItems() : sortCatalogItems(visibleItems(), state.sort, state.query);  // a registry's page keeps its order
        statusEl.innerHTML = statusLine(items.length);
        statusEl.dataset.tone = (state.catalogError && !state.catalog) || (browsing && state.browse?.error) ? 'danger' : 'muted';
        const shelf = shelfItems(state.catalog, host.installedByName(), state);
        // Shown once: Hide dismisses the shelf on this device for good (SPEC §8.2).
        const showShelf = shelf.length > 0 && !readHiddenShelf();
        drawnById = new Map(items.map((item) => [item.id, item]));  // the shelf's cards are among them: it stands on the unfiltered board
        let html = '';
        if (showShelf) {
            const copy = escapeHtml(shelfCopy(shelf));
            html += groupHeadHtml('Recommended to start', { copy, actions: `${buttonHtml(`Install all ${shelf.length}`, { tone: 'primary', attrs: 'data-browse-action="install_all"' })}${buttonHtml('Hide', { tone: 'ghost', attrs: 'data-browse-action="hide_shelf"' })}` });
            html += gridHtml(sortCatalogItems(shelf, 'popular'), 'recommended', state.category === 'all');
            const shown = new Set(shelf.map((item) => item.id));
            html += groupHeadHtml('All skills');
            html += gridHtml(items.filter((item) => !shown.has(item.id)), 'all', state.category === 'all');
        } else if (items.length) {
            html += gridHtml(items, 'all', state.category === 'all');
        } else if (browsing) {
            html += browseEmptyHtml(browsing);
        } else if (state.catalog) {
            html += `<p class="skills-empty">${sourceRows().length ? (state.query ? 'No skill matches this search.' : 'Nothing to show with these filters.') : 'No sources connected, so there is nothing to browse. Skills made here still work.'}</p>`;
        }
        const scroller = board.closest('.skills-scroll');
        let held = scroller?.scrollTop || 0;
        const restoreGeometry = keepGeometry();
        const restoreFocus = keepFocus();
        board.innerHTML = html;
        restoreGeometry();
        restoreFocus();
        // Until the new cards are measured a frame later, a Show more that went away can shorten the board;
        // the first layout gives the reader back the place the browser clamped, once.
        layoutGrids(held ? () => { if (held && scroller.scrollTop < held) scroller.scrollTop = held; held = 0; } : null);
        paintMore(Boolean(browsing && items.length && state.browse?.cursor));
        repaintSheet();
    }

    function browseEmptyHtml(source) {
        const { items = [], loading = true, error = '' } = state.browse || {};
        if (loading) return '';
        if (error) return `<p class="skills-empty">Nothing was loaded from ${escapeHtml(source.name)}.</p>${buttonHtml('Try again', { attrs: 'data-browse-action="browse_retry"' })}`;
        return items.length ? `<p class="skills-empty">${escapeHtml(source.name)}’s skills have no Official mark here, so Official only hides them.</p>` : '';
    }

    /** Repaint only the action rows (installed listing changed); scroll position and grids stay. */
    function repaintStates() {
        if (destroyed) return;
        const listing = host.installedByName();
        for (const card of board.querySelectorAll('[data-item].marketplace-card')) {
            const item = drawnById.get(card.dataset.item);
            if (!item) continue;
            const { installed } = resolveInstalled(item, listing);
            const pending = state.pending.get(item.id);
            if (pending && installed && !installed.lifecycle_virtual && (pending.phase === 'installing'
                || installed.enabled || installed.review_status !== 'pending' || installed.skill_review?.current?.lifecycle_status)) {
                state.pending.delete(item.id);
            }
            const next = stateFor(item, listing);
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
        // An open sheet's head mirrors the card's one action for the entry it shows, never another found by its id.
        repaintSheet();
    }

    // A refresh or another source leaves the registry's earlier answer, and any still on its way, describing a list
    // no longer shown: it is dropped while the query and filters stay, and Enter asks again.
    function dropSearch() {
        searchGeneration += 1;
        state.search = null;
        state.searching = false;
        state.searchError = '';
    }

    async function load({ refresh = false } = {}) {
        const current = ++loadGeneration;
        if (refresh) dropSearch();
        const [catalog] = await Promise.allSettled([host.catalog({ refresh }), host.sources({ replace: refresh })]);
        if (destroyed || current !== loadGeneration) return;
        if (catalog.status === 'fulfilled') { state.catalog = catalog.value; state.catalogError = ''; }
        else state.catalogError = catalog.reason?.message || String(catalog.reason);
        render();
        if (refresh && browsedRegistry()) browseRegistry();
        else ensureBrowse();
    }

    async function searchRegistry() {
        const query = state.query;
        const source = registry();
        if (!query || !source) return;
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

    /** The chosen registry's first page, or with `more` its next one appended. A failed page keeps what is shown.
     *  Resolves true when this page is the one now shown, false when another page, order or source replaced it. */
    async function browseRegistry({ more = false } = {}) {
        const source = browsedRegistry();
        if (!source) return false;
        const kept = more ? state.browse : null;
        const reading = { sort: state.sort, items: kept?.items || [], cursor: kept?.cursor || '', loading: true, more, error: '' };
        state.browse = reading;
        if (more) paintMore();  // the shown cards stay as they are while the next page is read
        else render();
        let answer = null;
        let error = '';
        try {
            answer = await api.search({ q: '', source: source.id, sort: reading.sort, cursor: reading.cursor });
            error = String(answer?.registry_error || '');
        } catch (err) {
            error = `${source.name}: ${err?.message || err}`;
        }
        if (destroyed || state.browse !== reading) return false;  // another page, order or source was asked for since
        const items = new Map(reading.items.map((item) => [item.id, item]));
        for (const item of error ? [] : answer?.items || []) items.set(item.id, item);
        state.browse = { ...reading, items: [...items.values()], cursor: error ? reading.cursor : String(answer?.next_cursor || ''), loading: false, error };
        render();
        return true;
    }

    /** Ask only when this order of the chosen registry is not already shown or being read. */
    function ensureBrowse() {
        if (browsedRegistry() && state.browse?.sort !== state.sort) browseRegistry();
    }

    /** A page read in the background may move the keyboard only when it is the page that was asked for and
     *  the focus is still where the action left it: nowhere, or on one of `from`. */
    function mayHandFocus(shown, ...from) {
        const current = document.activeElement;
        return shown && !destroyed && (!current || current === document.body || from.includes(current));
    }

    /** Show more: the keyboard lands on the first new card, where the button was. */
    async function showMore() {
        if (!state.browse?.cursor || state.browse.loading) return;
        const from = state.browse.items.length;
        const hadFocus = document.activeElement === moreButton;
        const shown = await browseRegistry({ more: true });
        if (!hadFocus || !mayHandFocus(shown, moreButton)) return;
        const target = board.querySelectorAll('.marketplace-card')[from]?.querySelector('[data-browse-action="open"]');
        if (!target) return;
        target.focus({ preventScroll: true });
        // After the masonry places the new cards: bring the focused one into view only if it is not already.
        requestAnimationFrame(() => { if (!destroyed && document.activeElement === target) target.scrollIntoView({ block: 'nearest' }); });
    }

    async function retryBrowse() {
        if (!mayHandFocus(await browseRegistry())) return;
        board.querySelector('[data-browse-action="browse_retry"], .marketplace-card [data-browse-action="open"]')?.focus({ preventScroll: true });
    }

    async function install(item, { replace = false } = {}) {
        state.failures.delete(item.id);
        state.pending.set(item.id, { phase: 'installing', startedAt: Date.now() });
        repaintStates();
        try {
            const result = await api.install(item.id, {
                turnOnWhenReady: true, replace,
                ...(item.source_key ? { expectedSourceKey: item.source_key } : {}),
                ...(item.digest ? { expectedDigest: item.digest } : {}),
                ...(item.registry_slug && item.version ? { version: item.version } : {}),
            });
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
            if (code === 'candidate_changed') {
                if (state.sheet?.item === item) state.sheet.handle.rejectOffer();
                host.toast('This offer changed. Refresh Browse and try again.', 'warn');
            } else if (['name_taken', 'name_collision', 'update_instead'].includes(code)) host.toast(`${item.name}: ${err.message}`, 'warn');
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

    function repaintSheet() {
        const shown = state.sheet;
        if (!shown || shown.handle.closed) return;
        if (sourcePresent(shown.item) === true) shown.sourceSeen = true;
        if (staleSource(shown.item, true)) shown.handle.rejectOffer();
        else shown.handle.setActions(sheetActions(shown.item));
    }

    function openSheet(item, opener) {
        state.sheet?.handle.close();
        const currentOpener = currentControl(opener);
        const { installed } = resolveInstalled(item, host.installedByName());
        const handle = openSkillDetailSheet({
            item, installed,
            policy: policyOf(item),
            actionsHtml: sheetActions(item),
            // A registry hit has no entry to read until it is installed; its card is all there is.
            readDetail: sourceRows().find((row) => row.id === item.source_id)?.kind === 'registry' || item.registry_slug ? null
                : ({ signal } = {}) => api.item(item.source_id, item.name, { signal }),
            onAction: (action, button) => act(action, button, currentOpener, item),
            returnFocus: currentOpener,
        });
        state.sheet = { item, handle, installedName: installed?.name, sourceSeen: sourcePresent(item) === true };
        repaintSheet();
    }

    // A sheet acts on the entry it shows (`shown`); a card on the entry it was drawn from.
    async function act(action, node, sheetOpener = null, shown = null) {
        const item = shown || drawnById.get(node?.dataset?.item || '');
        if (action === 'hide_shelf') { writeHiddenShelf('1'); render(); return; }
        if (action === 'browse_more') { await showMore(); return; }
        if (action === 'browse_retry') { await retryBrowse(); return; }
        if (action === 'install_all') {
            const missing = shelfItems(state.catalog, host.installedByName(), state);
            await host.confirmFirstStart?.();
            for (const entry of missing) if (!staleSource(entry)) await install(entry);
            return;
        }
        if (!item) return;
        if (action === 'open') { openSheet(item, node); return; }
        const stale = () => {
            const changed = staleSource(item, shown !== null);
            if (changed && shown && state.sheet?.item === item) state.sheet.handle.rejectOffer();
            return changed;
        };
        if (stale()) return;
        const resolve = () => resolveInstalled(item, host.installedByName());
        const { installed, other, sameFiles, tied } = resolve();
        if (action === 'install') {
            if (installed || other || sameFiles || tied) return;
            await host.confirmFirstStart?.(item.source_id, () => !stale());
            const now = resolve();
            if (!stale() && !now.installed && !now.other && !now.sameFiles && !now.tied) await install(item);
            return;
        }
        if (action === 'compare') {
            // Replace is one host operation: the other copy goes only when this one has landed.
            if (other && await host.compare(item, other) === 'replace' && !stale() && resolve().other === other) {
                await install(item, { replace: true });
            }
            return;
        }
        if (action === 'installed' || action === 'widgets') {
            const opened = installed || (action === 'installed' ? sameFiles || tied : null);
            if (!opened) return;
            // A departed sheet has no focus target; its opening control on Browse is the return.
            const opener = sheetOpener ? sheetOpener() : node;
            if (sheetOpener) state.sheet?.handle.close();
            if (action === 'installed') await host.showInstalled(opened.name, opener);
            else await host.runInstalledAction(opened.name, 'open_widgets', sheetOpener ? { opener } : undefined);
            return;
        }
        if (!installed) return;  // a stale control must not act on a different installed copy
        const map = { skip: 'attest_review', update: 'update', turn_on: 'turn_on' };
        if (map[action]) {
            if (action === 'update' && !offeredUpdate(item, installed)) return;
            await host.runInstalledAction(installed.name, map[action], { browseItem: item });
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
            ensureBrowse();
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
        ensureBrowse();
    };
    const onSource = () => { state.source = sourceSelect.value; dropSearch(); state.browse = null; render(); if (state.query) searchRegistry(); else ensureBrowse(); };
    const onSort = () => { state.sort = sortSelect.value; render(); ensureBrowse(); };
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
        repaintSheet,
        /** What the Official mark of a shown entry may say of its source: the entry's own facts, or null. */
        officialSource(id) {
            const item = drawnById.get(id);
            return item ? { ...policyOf(item), url: item.repo_url } : null;
        },
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
