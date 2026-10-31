// Skills → Sources (UX §6, boards B3, B4, B9): where skills come from and how
// much each place is trusted. Rows are equivalent (no row is heavier for being
// first or official); the trust value on the right is the setting the owner
// chose once, the health sentence in the meta line is live evidence — two axes,
// never merged. Changing a row's trust never rewrites what is already installed.
import { bindMenu } from './ui_interactions.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { skillSourcesApi } from './api_client.js';
import {
    openAddSourceDialog, openFirstStartDialog, openRemoveSourceDialog, openRenameSourceDialog, openTrustDialog,
} from './skill_dialogs.js';
import {
    buttonHtml, chipHtml, displayAddress, officialMarkHtml, plural, relativeAge, sourceAboutText,
    statusHtml, trayHtml, trustValueText,
} from './skill_presentation.js';

/**
 * Health of one source as a status sentence (dot + words): the host's read state
 * (ready | reading | error | search) with the age of the list it has.
 */
export function sourceHealthHtml(source, health) {
    if (source.kind === 'builtin') return '';  // it ships with the app: nothing is read
    if (!health?.state) return source.kind === 'registry' ? statusHtml('searched when you search', 'muted') : statusHtml('not read yet', 'muted');
    const age = relativeAge(health.last_read_at);
    if (health.state === 'reading' || (health.refreshing && health.state !== 'error')) {
        return statusHtml(age ? `reading… (list from ${escapeHtml(age)})` : 'reading…', 'muted', { work: true });
    }
    if (health.state === 'error' || health.error) {
        const shown = health.state === 'ready' && age ? ` — showing the list from ${escapeHtml(age)}` : '';
        return `${statusHtml(`couldn’t be read: ${escapeHtml(health.error || 'no answer')}${shown}`, 'warn')} ${buttonHtml('Try again', { tone: 'ghost', attrs: `data-source-action="refresh" data-source-id="${escapeHtml(source.id)}"` })}`;
    }
    if (source.kind === 'registry' || health.state === 'search') return statusHtml('searched when you search', 'muted');
    return statusHtml(age ? `read ${escapeHtml(age)}` : 'read', 'muted');
}

/** Hub rows still waiting for the owner's one-time confirmation of their suggested trust. */
export const unconfirmedHubRows = (sources = []) => sources.filter((source) => source.kind === 'hub' && source.confirmed === false);

/**
 * One source row: name and marks, trust value and ⋮, about and meta (DESIGN §6 row anatomy).
 * `health` is the catalog read's state of this source; without it the row's own state is used.
 */
export function renderSourceRow(source, { items = [], health = null } = {}) {
    const id = escapeHtml(source.id);
    const marks = [
        source.kind === 'hub' && source.part === 'official' ? officialMarkHtml() : '',
        source.kind === 'local' ? chipHtml('Folder', 'neutral') : '',
        // A row a build's settings supply (B9) — only when the host marks it so.
        source.managed ? chipHtml('Set by your organization', 'neutral') : '',
    ].join('');
    const meta = [];
    const state = health || source;
    const address = displayAddress(source.url || source.path || '');
    if (address) meta.push(escapeHtml(address));
    // A count is a fact of a list that was read; before that it is unknown, never 0.
    if (source.kind !== 'registry' && (state.state === 'ready' || items.length)) {
        const authors = new Set(items.map((item) => item.owner).filter(Boolean)).size;
        meta.push(`${plural(items.length, 'skill')}${source.kind === 'hub' && source.part !== 'official' && authors > 1 ? ` by ${plural(authors, 'author')}` : ''}`);
    }
    const healthHtml = sourceHealthHtml(source, state);
    if (healthHtml) meta.push(healthHtml);
    const suggested = source.confirmed === false;
    return `<div class="skills-source-row" data-source-row="${id}">
        <div class="skills-source-main"><strong>${escapeHtml(source.name)}</strong>${marks}</div>
        <div class="skills-source-actions">
            <button type="button" class="skills-trust-value${suggested ? ' is-suggested' : ''}" data-source-action="trust" data-source-id="${id}" aria-label="${escapeHtml(`${source.name}: ${trustValueText(source)}${suggested ? ' (suggested)' : ''}. Change`)}">${escapeHtml(trustValueText(source))}${suggested ? ' (suggested)' : ''} ›</button>
            <div class="skills-card-menu"><button type="button" class="skills-card-menu-trigger" aria-label="More actions for ${escapeHtml(source.name)}" aria-haspopup="menu" aria-expanded="false" data-source-menu="${id}">⋮</button></div>
        </div>
        <div class="skills-source-meta">
            <div class="skills-source-about">${escapeHtml(sourceAboutText(source))}</div>
            <div class="skills-source-facts">${meta.join(' · ')}</div>
        </div>
    </div>`;
}

/**
 * Mount the Sources view. `host` = { sources(force), sourcesProblem(): string, catalog({refresh}), installedSkills(): rows|null,
 *   reloadAll(): Promise, toast(msg, tone), showMadeHere(), openBehaviorSettings() }.
 */
export function createSkillSources({ pane, host, api = skillSourcesApi }) {
    let sources = [];
    let catalog = null;
    let error = '';
    let destroyed = false;
    let generation = 0;
    let activeMenu = null;

    const itemsBySource = () => {
        const map = new Map();
        for (const item of Array.isArray(catalog?.items) ? catalog.items : []) {
            if (!map.has(item.source_id)) map.set(item.source_id, []);
            map.get(item.source_id).push(item);
        }
        return map;
    };
    const healthById = () => new Map((catalog?.sources || []).map((health) => [health.id, health]));
    const installedFrom = (id) => (host.installedSkills() || []).filter((row) => row?.install?.source?.id === id).map((row) => row.name);

    function render() {
        if (destroyed) return;
        const items = itemsBySource();
        const health = healthById();
        const madeHere = (host.installedSkills() || []).filter((row) => !row.install && ['external', 'self_authored'].includes(String(row.source || ''))).length;
        const pendingRows = unconfirmedHubRows(sources);
        pane.innerHTML = `
            <section class="skills-card skills-sources-card">
                <div class="skills-sources-head">
                    <div class="skills-sources-id"><h3>Sources</h3>${statusHtml(error ? `couldn’t be read: ${escapeHtml(error)}` : `${plural(sources.length, 'source')} connected`, error ? 'danger' : 'muted')}</div>
                    ${buttonHtml('Add source', { attrs: 'data-source-action="add"' })}
                </div>
                <p class="skills-sources-copy">Skills come from these sources. For each one you chose once whether its skills install without review or are reviewed first.</p>
                ${host.sourcesProblem?.() ? trayHtml({ sentence: `Your saved list of sources couldn’t be used (${escapeHtml(host.sourcesProblem())}), so the default sources are shown.` }) : ''}
                ${pendingRows.length ? trayHtml({ sentence: 'Confirm how skills from the Ouroboros hub install. Suggested choices are shown until you do.',
                    actions: buttonHtml('Confirm…', { tone: 'primary', attrs: 'data-source-action="first_start"' }) }) : ''}
                <div class="skills-source-rows">${sources.map((source) => renderSourceRow(source, { items: items.get(source.id) || [], health: health.get(source.id) })).join('')}</div>
            </section>
            <div class="skills-sources-foot">
                <p>Skills made on this computer${host.installedSkills() ? ` (${madeHere})` : ''} don’t come from a source. <button type="button" class="skills-link-quiet" data-source-action="made_here">Publish one to share it ›</button></p>
                <p>Automatic keys for skills that may run: <button type="button" class="skills-link-quiet" data-source-action="behavior">Settings → Behavior ›</button></p>
            </div>`;
    }

    async function load({ refresh = false } = {}) {
        const current = ++generation;
        const [list, read] = await Promise.allSettled([host.sources(true), host.catalog({ refresh })]);
        if (destroyed || current !== generation) return;
        if (list.status === 'fulfilled') { sources = Array.isArray(list.value) ? list.value : []; error = ''; }
        else error = list.reason?.message || String(list.reason);
        if (read.status === 'fulfilled') catalog = read.value;
        render();
    }

    async function afterChange(message) {
        await host.reloadAll();
        await load();
        if (message) host.toast(message, 'ok');
    }

    async function confirmFirstStart() {
        if (!sources.length || !catalog) await load();
        const rows = unconfirmedHubRows(sources);
        if (!rows.length) return;
        const items = itemsBySource();
        const choice = await openFirstStartDialog(rows.map((row) => ({
            id: row.id, name: row.name, trust: row.trust,
            markHtml: row.part === 'official' ? officialMarkHtml({ control: false }) : '',
            about: row.part === 'official' ? `${plural((items.get(row.id) || []).length, 'skill')} marked Official by the maintainer`
                : `${plural((items.get(row.id) || []).length, 'skill')} shared by other people`,
        })));
        if (!choice) return;
        for (const [id, trust] of Object.entries(choice)) await api.update(id, { trust, confirmed: true });
        await afterChange('Saved. You can change it any time here.');
    }

    function openMenu(trigger) {
        activeMenu?.close();
        const id = trigger.dataset.sourceMenu;
        const source = sources.find((row) => row.id === id);
        if (!source) return;
        const menu = document.createElement('div');
        menu.className = 'ui-popup skills-card-menu-dialog';
        menu.setAttribute('role', 'menu');
        menu.innerHTML = [
            `<button type="button" role="menuitem" class="skills-menu-item" data-menu-action="refresh">Refresh</button>`,
            `<button type="button" role="menuitem" class="skills-menu-item" data-menu-action="rename">Rename…</button>`,
            source.kind === 'builtin' ? '' : `<button type="button" role="menuitem" class="skills-menu-item" data-menu-action="remove">Remove…</button>`,
        ].join('');
        document.body.appendChild(menu);
        trigger.setAttribute('aria-expanded', 'true');
        const binding = bindMenu(menu, { anchor: trigger, onClose: () => { menu.remove(); trigger.setAttribute('aria-expanded', 'false'); if (activeMenu?.menu === menu) activeMenu = null; } });
        activeMenu = { menu, close: () => binding.close() };
        menu.addEventListener('click', (event) => {
            const item = event.target.closest('[data-menu-action]');
            if (!item) return;
            binding.close({ restoreFocus: true });
            run(item.dataset.menuAction, source).catch((err) => host.toast(`${source.name}: ${err?.message || err}`, 'danger'));
        });
    }

    async function run(action, source) {
        if (action === 'refresh') {
            await api.refresh(source.id);
            host.toast(`Reading ${source.name}…`, 'muted');
            await afterChange();
        } else if (action === 'rename') {
            const result = await openRenameSourceDialog(source);
            const name = String(result?.value || '').trim();
            if (result?.confirmed && name && name !== source.name) {
                await api.update(source.id, { name });
                await afterChange(`Renamed to ${name}.`);
            }
        } else if (action === 'remove') {
            if (await openRemoveSourceDialog(source, installedFrom(source.id)) === true) {
                await api.remove(source.id);
                await afterChange(`${source.name} removed.`);
            }
        } else if (action === 'trust') {
            const trust = await openTrustDialog(source, { installedCount: installedFrom(source.id).length });
            if (trust && (trust !== source.trust || source.confirmed === false)) {
                await api.update(source.id, { trust, confirmed: true });
                await afterChange(trust === 'trusted' ? `Skills from ${source.name} now install without review.` : `Skills from ${source.name} are now reviewed first.`);
            }
        }
    }

    const onClick = async (event) => {
        const menuTrigger = event.target.closest('[data-source-menu]');
        if (menuTrigger) { openMenu(menuTrigger); return; }
        const target = event.target.closest('[data-source-action]');
        if (!target || target.disabled) return;
        const action = target.dataset.sourceAction;
        const source = sources.find((row) => row.id === target.dataset.sourceId);
        try {
            if (action === 'add') {
                const added = await openAddSourceDialog({ api });
                if (added) await afterChange(`${added.name || 'The source'} added. Its skills are in Browse.`);
            } else if (action === 'first_start') await confirmFirstStart();
            else if (action === 'made_here') host.showMadeHere();
            else if (action === 'behavior') host.openBehaviorSettings();
            else if (source) await run(action, source);
        } catch (err) {
            host.toast(err?.message || String(err), 'danger');
        }
    };
    pane.addEventListener('click', onClick);
    return {
        show: (options) => load(options),
        refresh: () => load({ refresh: true }),
        confirmFirstStart,
        destroy() {
            destroyed = true;
            activeMenu?.close();
            pane.removeEventListener('click', onClick);
        },
    };
}
