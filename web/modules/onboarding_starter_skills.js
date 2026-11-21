// The first-run wizard's Skills step, between Consciousness and Review limits (owner 2026-11-21: «показать что есть
// много крутых скилов анимированным списком, но юзер может поснимать галочки»; it replaces the Summary row his 4A of
// 2026-10-17 chose). The server offers it only where completion honors the ticks (`stepOrder`: this data root's
// starter list never ran). One categorized list of cards, built-ins first, each card its human title, one line of
// what it gives and the prerequisite in words; the maintainer's recommended skills are ticked. The cards arrive in
// one short sequence on entering the step, never on a repaint (DESIGN §7). How the hub's skills install is one quiet
// line under the list. Nothing installs here: the one atomic Finish carries the ticked ids and the trust choices,
// and the host installs them in the background after start. Summary shows one row back to the step. A list that
// cannot be read leaves the host's defaults in place and says so.
import { skillSourcesApi } from './api_client.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { officialMarkHtml } from './skill_presentation.js';

const BUILT_IN_TITLE = 'Built in — comes with the app';
const ARRIVE_LAST = 12;  // cards from the twelfth on arrive together, so the whole entrance stays under a second
const FOCUS_KEYS = ['starterItem', 'starterAll', 'starterTrustToggle', 'starterTrust'];

/** Built-in skills lead as their own group; the rest keep the host's category order. */
export function normalizeStarterGroups(groups = []) {
    const list = Array.isArray(groups) ? groups : [];
    if (list.some((group) => group?.category === 'builtin')) {
        return list.map((group) => ({ ...group, label: group.category === 'builtin' ? (group.label || BUILT_IN_TITLE) : (group.label || group.category), items: group.items || [] }))
            .filter((group) => group.items.length);
    }
    const builtin = [];
    const rest = list.map((group) => {
        const items = [];
        for (const item of group?.items || []) (item.builtin ? builtin : items).push(item);
        return { ...group, label: group?.label || group?.category || '', items };
    }).filter((group) => group.items.length);
    return builtin.length ? [{ category: 'builtin', label: BUILT_IN_TITLE, items: builtin }, ...rest] : rest;
}

const keyWords = (key) => String(key).toLowerCase().split('_').filter(Boolean)
    .map((word, index) => (word === 'api' ? 'API' : index ? word : word[0].toUpperCase() + word.slice(1))).join(' ');

/** The prerequisite in words: the catalog's own (`needs`; "" says it has none), else the first setting the skill
 * reads, named by the setup contract's label or in words, and how many more it reads. */
export function needsText(item, keyLabels = {}) {
    if (typeof item?.needs === 'string') return item.needs.trim();
    const keys = (Array.isArray(item?.needs_keys) ? item.needs_keys : [])
        .map((key) => (typeof key === 'string' ? key : key?.label || key?.name)).filter(Boolean);
    const more = keys.length - 1;
    return keys.length ? `Uses your ${keyLabels[keys[0]] || keyWords(keys[0])}${more ? ` and ${more} more setting${more === 1 ? '' : 's'}` : ''}` : '';
}

const quote = (value) => String(value).replace(/["\\]/g, '\\$&');
const dataAttr = (key) => `data-${key.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}`;

/** The step and its Summary row. `offered`: the bootstrap's `stepOrder` has the step (else it renders and sends
 * nothing and reads nothing); `open` returns to the step from Summary. `read()` defaults to GET
 * /api/onboarding/starter-skills. */
export function createStarterSkillsStep({ offered = true, open = () => {}, keyLabels = {}, read = () => skillSourcesApi.starterSkills(), setTimer = setTimeout, clearTimer = clearTimeout, now = Date.now } = {}) {
    const state = { status: 'loading', groups: [], ticked: new Set(), trust: {}, rows: [], sources: [], trustOpen: false, error: '', ticksChanged: false, untickNew: false };
    let host = null;
    let view = '';  // what the mounted node shows: the step's list, Summary's row, or nothing
    let painted = '';
    let arrived = false;  // this entry of the step has shown its cards once
    let destroyed = false;
    let timer = null;
    let deadline = null;
    const items = () => state.groups.flatMap((group) => group.items);
    const tickedItems = () => items().filter((item) => state.ticked.has(item.id));
    const hubs = () => state.sources.filter((source) => source.kind === 'hub' || state.rows.some((row) => row.id === source.id));
    const reading = () => hubs().some((source) => source.state === 'reading');
    const useDefaults = () => !state.ticksChanged && hubs().some((source) => source.state !== 'ready');

    async function refresh() {
        if (destroyed || (deadline !== null && now() > deadline)) return;
        try {
            const data = await read();
            if (destroyed) return;
            const known = new Set(items().map((item) => item.id));
            const groups = new Map(state.groups.map((group) => [group.category, group]));
            for (const group of normalizeStarterGroups(data?.groups)) {
                const merged = new Map((groups.get(group.category)?.items || []).map((item) => [item.id, item]));
                for (const item of group.items) {
                    merged.set(item.id, item);
                    if (!known.has(item.id) && item.recommended && !state.untickNew) state.ticked.add(item.id);
                }
                groups.set(group.category, { ...group, items: [...merged.values()] });
            }
            state.groups = [...groups.values()];
            state.rows = Array.isArray(data?.hub_rows) ? data.hub_rows.filter((row) => row?.id) : [];
            state.sources = Array.isArray(data?.sources) ? data.sources : [];
            for (const row of state.rows) state.trust[row.id] ??= row.trust === 'trusted' ? 'trusted' : 'review';
            state.status = items().length || state.rows.length ? 'ready' : 'empty';
            state.error = '';
        } catch (error) {
            if (destroyed) return;
            if (state.status === 'loading') state.status = 'error';
            state.error = error?.message || String(error);
        }
        if (destroyed) return;
        paint();
        if (deadline !== null && reading() && now() + 3000 <= deadline) timer = setTimer(refresh, 3000);
    }
    const loading = offered ? Promise.resolve().then(refresh) : Promise.resolve();

    function pendingText() {
        return [reading() ? 'Reading the Ouroboros hub…' : '',
            useDefaults() ? 'If you finish before it is ready, Ouroboros installs the default starter list once it can read the hub.' : '',
            state.error && state.status === 'ready' ? `The list couldn’t be refreshed (${state.error}).` : ''].filter(Boolean).join(' ');
    }
    const unreadText = () => `The list couldn’t be read (${state.error}). Ouroboros installs the recommended skills after it starts; you can change them on the Skills page.`;
    const countText = () => `${tickedItems().length} of ${items().length} ticked`;

    function rowHtml() {
        const count = tickedItems().length;
        const value = { loading: 'Reading the list…', error: 'The recommended set', empty: 'None offered' }[state.status]
            || (count ? `${count} will install` : 'None will install');
        const sub = state.status === 'error' ? unreadText() : pendingText();
        return `<div class="summary-kv skills-starter-row"><strong>Skills</strong><span class="skills-starter-summary">
            <span class="skills-starter-value">${escapeHtml(value)} · <button type="button" class="skills-starter-change" data-starter-change>Change</button></span>
            ${sub ? `<span class="skills-starter-sub" role="status">${escapeHtml(sub)}</span>` : ''}</span></div>`;
    }

    function cardHtml(item, order) {
        const on = state.ticked.has(item.id);
        const marks = (item.builtin ? '<span class="ui-chip skills-chip" data-tone="neutral">Built in</span>' : '')
            + (item.official ? officialMarkHtml({ control: false }) : '');
        const title = String(item.title || '').trim();
        const pitch = String(item.pitch || '').trim();
        const needs = needsText(item, keyLabels);
        // The catalog's own words translate (the interface layer); a name, a skill author's text and a line composed
        // from setting names stay as they are.
        const authored = (own) => (own ? '' : ' data-i18n-authored');
        return `<label class="wizard-choice skills-starter-card${on ? ' active' : ''}" data-starter-card="${escapeHtml(item.id)}" style="--arrive: ${order}">
            <span class="skills-starter-head"><input type="checkbox" class="ui-checkbox" data-starter-item="${escapeHtml(item.id)}"${on ? ' checked' : ''}><strong${authored(title)}>${escapeHtml(title || item.name)}</strong><span class="skills-starter-marks">${marks}</span></span>
            <span class="skills-starter-pitch"${authored(pitch)}>${escapeHtml(pitch || item.description || '')}</span>
            ${needs ? `<span class="skills-starter-needs"${authored(typeof item.needs === 'string')}>${escapeHtml(needs)}</span>` : ''}
        </label>`;
    }

    function trustHtml() {
        if (!state.rows.length) return '';
        const official = state.rows.find((row) => row.part === 'official') || state.rows[0];
        const lead = state.trust[official.id] === 'trusted'
            ? `They install without review — you trust ${official.name || 'Ouroboros'}. Skills that need a key wait for it.`
            : 'Ouroboros reviews each one before it runs. Skills that need a key wait for it.';
        const rows = state.rows.map((row) => `<span class="skills-starter-trust-row"><strong>${escapeHtml(row.name)}</strong>
            ${[['trusted', 'Without review'], ['review', 'Review each']].map(([value, label]) => `<button type="button" class="selection-pill${state.trust[row.id] === value ? ' active' : ''}" aria-pressed="${state.trust[row.id] === value ? 'true' : 'false'}" data-starter-trust="${escapeHtml(row.id)}" data-trust="${value}">${label}</button>`).join('')}</span>`).join('');
        return `<div class="skills-starter-trust"><p class="skills-starter-sub">${escapeHtml(lead)} <button type="button" class="skills-starter-change" data-starter-trust-toggle aria-expanded="${state.trustOpen}">${state.trustOpen ? 'Hide' : 'Change'}</button></p>
            ${state.trustOpen ? `${rows}<span class="skills-starter-sub">Suggested choices apply until you choose.</span>` : ''}</div>`;
    }

    function stepHtml(arrive) {
        if (state.status !== 'ready') {
            const words = { loading: 'Reading the list…', error: unreadText(), empty: 'No skills are offered here. You can add skills any time on the Skills page.' }[state.status];
            return `<p class="skills-starter-sub" role="status">${escapeHtml(words)}</p>`;
        }
        let order = 0;
        const pending = pendingText();
        // What is still being read stands under the list, where the hub's skills will join it: its going moves no card.
        return `<div class="skills-starter-list"${arrive ? ' data-arrive' : ''}>
            ${items().length ? `<div class="skills-starter-bar"><span class="skills-starter-count" role="status">${countText()}</span><span class="skills-starter-bulk"><button type="button" data-starter-all="1">Tick all</button><span aria-hidden="true">·</span><button type="button" data-starter-all="0">Untick all</button></span></div>` : ''}
            ${state.groups.map((group) => `<section class="skills-starter-group"><h3 style="--arrive: ${Math.min(order, ARRIVE_LAST)}">${escapeHtml(group.label)}<span>${group.items.length}</span></h3>
                <div class="wizard-choice-grid skills-starter-cards">${group.items.map((item) => cardHtml(item, Math.min(order++, ARRIVE_LAST))).join('')}</div></section>`).join('')}
            ${pending ? `<p class="skills-starter-sub" role="status">${escapeHtml(pending)}</p>` : ''}
            ${trustHtml()}
        </div>`;
    }

    // The reading place a repaint keeps (DESIGN principle 6): the focused control and the first card in view.
    function readingPlace() {
        if (typeof host.querySelectorAll !== 'function') return null;
        const active = host.ownerDocument?.activeElement;
        const key = active && host.contains(active) ? FOCUS_KEYS.find((name) => active.dataset?.[name] !== undefined) : '';
        const focus = key ? `[${dataAttr(key)}="${quote(active.dataset[key])}"]${key === 'starterTrust' ? `[data-trust="${quote(active.dataset.trust)}"]` : ''}` : '';
        const card = [...host.querySelectorAll('[data-starter-card]')].find((node) => node.getBoundingClientRect().bottom > 0);
        const anchor = card && { id: card.dataset.starterCard, top: card.getBoundingClientRect().top };
        return () => {
            if (focus) host.querySelector(focus)?.focus({ preventScroll: true });
            const again = anchor && host.querySelector(`[data-starter-card="${quote(anchor.id)}"]`);
            const moved = again ? again.getBoundingClientRect().top - anchor.top : 0;
            if (moved) host.ownerDocument?.defaultView?.scrollBy(0, moved);
        };
    }

    function paint() {
        if (!host?.isConnected) return;
        const html = view === 'step' ? stepHtml(false) : view === 'row' ? rowHtml() : '';
        if (html === painted) return;  // a read that changed nothing rewrites nothing
        const arrive = view === 'step' && !arrived && state.status === 'ready' && items().length > 0;
        const restore = painted && view === 'step' ? readingPlace() : null;
        host.innerHTML = arrive ? stepHtml(true) : html;
        painted = html;
        arrived ||= arrive;
        restore?.();
    }

    function onClick(event) {
        if (event.target.closest('[data-starter-change]')) { open(); return; }
        if (event.target.closest('[data-starter-trust-toggle]')) { state.trustOpen = !state.trustOpen; paint(); return; }
        const bulk = event.target.closest('[data-starter-all]');
        if (bulk) {
            state.ticksChanged = true;
            state.untickNew = bulk.dataset.starterAll === '0';
            state.ticked = bulk.dataset.starterAll === '1' ? new Set(items().map((item) => item.id)) : new Set();
            paint();
            return;
        }
        const trust = event.target.closest('[data-starter-trust]');
        if (trust) { state.trust[trust.dataset.starterTrust] = trust.dataset.trust; paint(); }
    }

    function onChange(event) {
        const box = event.target.closest('[data-starter-item]');
        if (!box) return;
        state.ticksChanged = true;
        if (box.checked) state.ticked.add(box.dataset.starterItem);
        else state.ticked.delete(box.dataset.starterItem);
        box.closest('.skills-starter-card')?.classList.toggle('active', box.checked);
        const count = host.querySelector('.skills-starter-count');
        if (count) count.textContent = countText();
    }

    return {
        loading,
        offered: () => offered,
        /** The step's header (the contract's title and copy) over the node its list paints into. */
        html: ({ title = 'Skills', copy = '' } = {}) => `
            <div class="step-header"><div>
                <h2 class="step-title">${escapeHtml(title)}</h2>
                ${copy ? `<p class="step-copy">${escapeHtml(copy)}</p>` : ''}
            </div></div>
            <div class="skills-starter-step" data-starter-skills="step"></div>`,
        /** Mount into the node a wizard render gives (`data-starter-skills`: "step" or Summary's "row"); the ticks
         * survive re-renders, and the cards arrive once each time the step is entered. */
        mount(node) {
            if (host === node) return;
            host?.removeEventListener('click', onClick);
            host?.removeEventListener('change', onChange);
            host = offered && node ? node : null;
            const next = !host ? '' : host.dataset?.starterSkills === 'step' ? 'step' : 'row';
            if (next === 'step' && view !== 'step') arrived = false;
            view = next;
            painted = '';
            if (!host) return;
            if (deadline === null) {
                deadline = now() + 180_000;
                if (state.status !== 'loading' && reading()) timer = setTimer(refresh, 3000);
            }
            host.addEventListener('click', onClick);
            host.addEventListener('change', onChange);
            paint();
        },
        /** What the one atomic Finish carries; nothing when the step is not offered or the list could not be read. */
        payload() {
            if (!offered || state.status !== 'ready') return {};
            return {
                ...(useDefaults() ? {} : { starter_skills: tickedItems().map((item) => item.id) }),
                skill_source_trust: { ...state.trust },
            };
        },
        snapshot: () => ({ ...state, ticked: [...state.ticked], view, arrived }),
        destroy() {
            destroyed = true;
            clearTimer(timer);
            host?.removeEventListener('click', onClick);
            host?.removeEventListener('change', onChange);
            host = null;
        },
    };
}
