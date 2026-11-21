// Extensions › Sources (UX §6, boards B3, B4, B9): where skills come from and how
// much each place is trusted. Rows are equivalent (no row is heavier for being
// first or official); the trust value on the right is the setting the owner
// chose once, the health sentence in the meta line is live evidence — two axes,
// never merged. Changing a row's trust never rewrites what is already installed.
import { bindMenu } from './ui_interactions.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { skillSourcesApi } from './api_client.js';
import { fmt } from './i18n.js';
import {
    openAddSourceDialog, openFirstStartDialog, openRemoveSourceDialog, openRenameSourceDialog, openTrustDialog, sourceProblemsHtml,
} from './skill_dialogs.js';
import {
    buttonHtml, chipHtml, officialMarkHtml, plural, relativeAge, sourceAboutText, sourceAddressText,
    statusHtml, trayHtml, trustValueText,
} from './skill_presentation.js';

/**
 * Health of one source as a status sentence (dot + words): the host's read state
 * (ready | reading | error | search) with the age of the list it has.
 */
export function sourceHealthHtml(source, health) {
    if (source.kind === 'builtin') return '';  // it ships with the app: nothing is read
    if (!health?.state) return source.kind === 'registry' ? statusHtml('read when chosen or searched', 'muted') : statusHtml('not read yet', 'muted');
    const age = relativeAge(health.last_read_at);
    if (health.state === 'reading' || (health.refreshing && health.state !== 'error')) {
        return statusHtml(age ? `reading… (list from ${escapeHtml(age)})` : 'reading…', 'muted', { work: true });
    }
    if (health.state === 'error' || health.error) {
        const shown = health.state === 'ready' && age ? ` — showing the list from ${escapeHtml(age)}` : '';
        return `${statusHtml(`couldn’t be read: ${escapeHtml(health.error || 'no answer')}${shown}`, 'warn')} ${buttonHtml('Try again', { tone: 'ghost', attrs: `data-source-action="refresh" data-source-id="${escapeHtml(source.id)}"` })}`;
    }
    if (source.kind === 'registry' || health.state === 'search') return statusHtml('read when chosen or searched', 'muted');
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
        source.kind === 'hub' && source.part === 'official' ? officialMarkHtml({ sourceId: source.id }) : '',  // this row's own mark
        source.kind === 'local' ? chipHtml('Folder', 'neutral') : '',
        // A row a build's settings supply (B9) — only when the host marks it so.
        source.managed ? chipHtml('Set by your organization', 'neutral') : '',
    ].join('');
    const meta = [];
    const state = health || source;
    const address = sourceAddressText(source);
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
            <button type="button" class="skills-trust-value${suggested ? ' is-suggested' : ''} ui-hit-area" data-source-action="trust" data-source-id="${id}" aria-label="${escapeHtml(`${source.name}: ${trustValueText(source)}${suggested ? ' (suggested)' : ''}. Change`)}">${escapeHtml(trustValueText(source))}${suggested ? ' (suggested)' : ''} ›</button>
            <div class="skills-card-menu"><button type="button" class="btn btn-ghost btn-sm skills-card-menu-trigger" aria-label="More actions for ${escapeHtml(source.name)}" aria-haspopup="menu" aria-expanded="false" data-source-menu="${id}">⋮</button></div>
        </div>
        <div class="skills-source-meta">
            <div class="skills-source-about">${escapeHtml(sourceAboutText(source))}</div>
            <div class="skills-source-facts">${meta.join(' · ')}</div>
            ${sourceProblemsHtml(Array.isArray(state.problems) ? state.problems : [])}
        </div>
    </div>`;
}

/**
 * Mount the Sources view. `host` = { sources(options), sourceSnapshot(): { rows, error, problem }, sourceSuggestions(),
 *   catalog({refresh}), installedSkills(): rows|null, reloadAll(): Promise, toast(msg, tone), showMadeHere(),
 *   openSettings(path, options) }.
 */
export function createSkillSources({ pane, host, api = skillSourcesApi }) {
    let catalog = null;
    let destroyed = false;
    let generation = 0;
    let activeMenu = null;
    const sourceRows = () => host.sourceSnapshot().rows || [];

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

    function paintMadeHereCopy() {
        const copy = pane.querySelector('[data-source-made-copy]');
        if (!copy) return;
        const madeHere = host.installedSkills()?.filter((row) => !row.install && ['external', 'self_authored'].includes(String(row.source || ''))).length;
        const words = madeHere === undefined ? fmt('Skills made on this computer don’t come from a source.')
            : fmt('Skills made on this computer ({n}) don’t come from a source.', { n: madeHere });
        if (copy.textContent !== words) copy.textContent = words;
    }

    function render() {
        if (destroyed) return;
        const { rows, error, problem } = host.sourceSnapshot();
        const sources = rows || [];
        const summary = error ? statusHtml(`couldn’t be read: ${escapeHtml(error.message || error)}`, 'danger')
            : rows ? statusHtml(`${plural(sources.length, 'source')} connected`, 'muted') : '';
        // Keep the footer's navigation controls mounted: a jump's return may hold one during a Sources read.
        if (!pane.querySelector('.skills-sources-card')) pane.innerHTML = `
            <section class="ui-card skills-card skills-sources-card"></section>
            <div class="skills-sources-foot">
                <p><span data-source-made-copy data-i18n-skip></span> <button type="button" class="skills-link-quiet ui-hit-area" data-source-action="made_here">Publish one to share it ›</button></p>
                <p>Automatic keys for skills that may run: <button type="button" class="skills-link-quiet ui-hit-area" data-source-action="auto_grant">Permissions and checks ›</button></p>
                <p>The skills folder on this computer and the ClawHub address: <button type="button" class="skills-link-quiet ui-hit-area" data-source-action="skill_sources">Skill sources ›</button></p>
            </div>`;
        const card = pane.querySelector('.skills-sources-card');
        const active = document.activeElement;
        const focused = card.contains(active) ? active?.closest('[data-source-action], [data-source-menu]') : null;
        const items = itemsBySource();
        const health = healthById();
        paintMadeHereCopy();
        const pendingRows = unconfirmedHubRows(sources);
        card.innerHTML = `
                <div class="skills-sources-head">
                    <div class="skills-sources-id"><h3>Sources</h3>${summary}</div>
                    ${buttonHtml('Add source', { attrs: 'data-source-action="add"' })}
                </div>
                <p class="skills-sources-copy">Skills come from these sources. For each one you chose once whether its skills install without review or are reviewed first.</p>
                ${problem ? trayHtml({ sentence: `Your saved list of sources couldn’t be used (${escapeHtml(problem)}), so the default sources are shown.` }) : ''}
                ${pendingRows.length ? trayHtml({ sentence: 'Confirm how skills from the Ouroboros hub install. Suggested choices are shown until you do.',
                    actions: buttonHtml('Confirm…', { tone: 'primary', attrs: 'data-source-action="first_start"' }) }) : ''}
                <div class="skills-source-rows">${sources.map((source) => renderSourceRow(source, { items: items.get(source.id) || [], health: health.get(source.id) })).join('')}</div>`;
        // A row drawn again keeps focus on its current control, not an old one.
        if (focused) {
            const target = [...pane.querySelectorAll('[data-source-action], [data-source-menu]')].find((node) =>
                ['sourceAction', 'sourceId', 'sourceMenu'].every((key) => node.dataset[key] === focused.dataset[key]));
            (target || pane.querySelector('[data-source-action="add"]'))?.focus({ preventScroll: true });
        }
    }

    async function readCatalog({ refresh = false } = {}) {
        const current = ++generation;
        const read = await host.catalog({ refresh }).catch(() => null);
        if (destroyed || current !== generation || !read || read === catalog) return;
        catalog = read;
        render();
    }

    async function load({ refresh = false } = {}) {
        render();
        await Promise.all([host.sources({ fresh: true, replace: refresh }).catch(() => null), readCatalog({ refresh })]);
    }

    async function afterChange(message) {
        await host.reloadAll();
        await readCatalog();
        if (message) host.toast(message, 'ok');
    }

    async function confirmFirstStart(canOpen = () => true) {
        if (!catalog) await readCatalog();
        const rows = unconfirmedHubRows(sourceRows());
        if (!rows.length || destroyed || !canOpen()) return;
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
        const source = sourceRows().find((row) => row.id === id);
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
        const source = sourceRows().find((row) => row.id === target.dataset.sourceId);
        try {
            if (action === 'add') {
                const added = await openAddSourceDialog({ api, suggestions: host.sourceSuggestions?.() || [] });
                if (added) await afterChange(`${added.name || 'The source'} added. Its skills are in Browse.`);
            } else if (action === 'first_start') await confirmFirstStart();
            else if (action === 'made_here') await host.showMadeHere();
            else if (action === 'auto_grant') host.openSettings('permissions', { row: 'OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS', opener: target });
            else if (action === 'skill_sources') host.openSettings('skill-sources', { opener: target });
            else if (source) await run(action, source);
        } catch (err) {
            host.toast(err?.message || String(err), 'danger');
        }
    };
    pane.addEventListener('click', onClick);
    window.addEventListener('ouro:language-changed', paintMadeHereCopy);
    return {
        show: (options) => load(options),
        refresh: () => load({ refresh: true }),
        repaint: render,
        confirmFirstStart,
        destroy() {
            destroyed = true;
            activeMenu?.close();
            pane.removeEventListener('click', onClick);
            window.removeEventListener('ouro:language-changed', paintMadeHereCopy);
        },
    };
}
