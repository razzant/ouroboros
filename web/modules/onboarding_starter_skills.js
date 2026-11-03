// The first-run wizard's «Starter skills» row (Summary step; board B13). One
// summary row «20 · 6 ticked · Change ›» that opens in place into one scrollable
// panel of categorized cards (two columns, one on a phone) with the maintainer's
// recommended skills ticked, plus how skills from the two public hub rows
// install, the suggested choices selected. Nothing installs here: the one atomic
// Finish carries the ticked ids and the trust choices, and the host installs
// them in the background after start. A list that cannot be read leaves the
// host's defaults in place and says so.
import { skillSourcesApi } from './api_client.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { officialMarkHtml } from './skill_presentation.js';
import { bindScrollFade } from './scroll_fade.js';

const BUILT_IN_TITLE = 'Built in — comes with the app';

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

function needsLine(item) {
    const keys = (Array.isArray(item.needs_keys) ? item.needs_keys : [])
        .map((key) => (typeof key === 'string' ? key : key?.label || key?.name)).filter(Boolean);
    return keys.length ? `Needs: ${keys.join(', ')}` : '';
}

/** The wizard row and its panel. `read()` defaults to GET /api/onboarding/starter-skills. */
export function createStarterSkillsRow({ read = () => skillSourcesApi.starterSkills(), setTimer = setTimeout, clearTimer = clearTimeout, now = Date.now } = {}) {
    const state = { status: 'loading', groups: [], ticked: new Set(), trust: {}, rows: [], sources: [], expanded: false, error: '', ticksChanged: false, untickNew: false };
    let host = null;
    let disposeFade = null;
    let destroyed = false;
    let timer = null;
    let deadline = null;
    const items = () => state.groups.flatMap((group) => group.items);
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
    const loading = Promise.resolve().then(refresh);

    function summaryValue() {
        if (state.status === 'loading') return '<span class="skills-starter-value">Reading the list…</span>';
        if (state.status === 'error') return `<span class="skills-starter-value">The default set</span><span class="skills-starter-sub">The list couldn’t be read (${escapeHtml(state.error)}). Ouroboros installs the recommended skills after it starts; you can change them on the Skills page.</span>`;
        if (state.status === 'empty') return '<span class="skills-starter-value">None offered</span>';
        const all = items();
        const ticked = all.filter((item) => state.ticked.has(item.id));
        const toggle = `<button type="button" class="skills-starter-change" data-starter-toggle aria-expanded="${state.expanded ? 'true' : 'false'}">${state.expanded ? 'Hide ▾' : 'Change ›'}</button>`;
        const sub = state.expanded || useDefaults() ? '' : `<span class="skills-starter-sub">${ticked.length ? `Ticked: ${escapeHtml(ticked.map((item) => item.name).join(', '))}.` : 'Nothing is ticked: no skills install now.'}${all.length > ticked.length ? ` ${all.length - ticked.length} more you can tick.` : ''}</span>`;
        const pending = [reading() ? 'Reading the Ouroboros hub…' : '',
            useDefaults() ? 'If you finish before it is ready, Ouroboros installs the default starter list once it can read the hub.' : '',
            state.error ? `The list couldn’t be refreshed (${state.error}).` : ''].filter(Boolean).join(' ');
        return `<span class="skills-starter-value">${all.length} · ${ticked.length} ticked · ${toggle}</span>${sub}${pending ? `<span class="skills-starter-sub" role="status">${escapeHtml(pending)}</span>` : ''}`;
    }

    function cardHtml(item) {
        const on = state.ticked.has(item.id);
        const mark = (item.builtin ? '<span class="ui-chip skills-chip" data-tone="neutral">Built in</span>' : '')
            + (item.official ? officialMarkHtml({ control: false }) : '');
        const needs = needsLine(item);
        return `<label class="wizard-choice skills-starter-card${on ? ' active' : ''}">
            <span class="skills-starter-head"><input type="checkbox" class="ui-checkbox" data-starter-item="${escapeHtml(item.id)}"${on ? ' checked' : ''}><strong>${escapeHtml(item.name)}</strong><span class="skills-starter-marks">${mark}</span></span>
            <span class="skills-starter-desc" data-i18n-authored>${escapeHtml(item.description || '')}</span>
            ${needs ? `<span class="skills-starter-needs">${escapeHtml(needs)}</span>` : ''}
        </label>`;
    }

    function panelHtml() {
        const official = state.rows.find((row) => row.part === 'official') || state.rows[0];
        const officialTrusted = official ? state.trust[official.id] === 'trusted' : true;
        const lead = officialTrusted
            ? `They install without review — you trust ${escapeHtml(official?.name || 'Ouroboros')}. Skills that need a key wait for it.`
            : 'Ouroboros reviews each one before it runs. Skills that need a key wait for it.';
        const trustRows = state.rows.map((row) => `<span class="skills-starter-trust-row"><strong>${escapeHtml(row.name)}</strong>
            ${[['trusted', 'Without review'], ['review', 'Review each']].map(([value, label]) => `<button type="button" class="selection-pill${state.trust[row.id] === value ? ' active' : ''}" aria-pressed="${state.trust[row.id] === value ? 'true' : 'false'}" data-starter-trust="${escapeHtml(row.id)}" data-trust="${value}">${label}</button>`).join('')}</span>`).join('');
        return `<div class="skills-starter-panel" data-starter-panel>
            ${items().length ? `<div class="skills-starter-bar"><p>${lead}</p><span class="skills-starter-bulk"><button type="button" data-starter-all="1">Tick all</button><span aria-hidden="true">·</span><button type="button" data-starter-all="0">Untick all</button></span></div>
            <div class="skills-starter-scroll" tabindex="0" aria-label="Starter skills" data-starter-scroll>${state.groups.map((group) => `
                <section class="skills-starter-group"><h4>${escapeHtml(group.label)}<span>${group.items.length}</span></h4>
                    <div class="wizard-choice-grid skills-starter-cards">${group.items.map(cardHtml).join('')}</div></section>`).join('')}</div>` : ''}
            ${trustRows ? `<div class="skills-starter-trust"><span class="skills-starter-sub">How skills from the Ouroboros hub install</span>${trustRows}
                <span class="skills-starter-sub">Suggested choices apply until you choose. You can add or remove skills any time on the Skills page.</span></div>` : ''}
        </div>`;
    }

    function paint() {
        if (!host?.isConnected) return;
        const scroll = Number(host.querySelector('[data-starter-scroll]')?.scrollTop) || 0;
        disposeFade?.();
        disposeFade = null;
        host.innerHTML = `<div class="summary-kv skills-starter-row"><strong>Starter skills</strong><span class="skills-starter-summary">${summaryValue()}</span></div>${state.expanded && state.status === 'ready' ? panelHtml() : ''}`;
        const scroller = host.querySelector('[data-starter-scroll]');
        if (scroller?.nodeType === 1) {
            scroller.scrollTop = scroll;
            if (typeof ResizeObserver === 'function') disposeFade = bindScrollFade(scroller);
        }
    }

    function onClick(event) {
        if (event.target.closest('[data-starter-toggle]')) { state.expanded = !state.expanded; paint(); return; }
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
        const summary = host.querySelector('.skills-starter-summary');
        if (summary) summary.innerHTML = summaryValue();
    }

    return {
        loading,
        /** Mount into the wizard's placeholder after each wizard render; the state survives re-renders. */
        mount(node) {
            if (host === node) return;
            disposeFade?.();
            disposeFade = null;
            host?.removeEventListener('click', onClick);
            host?.removeEventListener('change', onChange);
            host = node || null;
            if (!host) return;
            if (deadline === null) {
                deadline = now() + 180_000;
                if (state.status !== 'loading' && reading()) timer = setTimer(refresh, 3000);
            }
            host.addEventListener('click', onClick);
            host.addEventListener('change', onChange);
            paint();
        },
        /** What the one atomic Finish carries; nothing when the list could not be read. */
        payload() {
            if (state.status !== 'ready') return {};
            return {
                ...(useDefaults() ? {} : { starter_skills: items().filter((item) => state.ticked.has(item.id)).map((item) => item.id) }),
                skill_source_trust: { ...state.trust },
            };
        },
        snapshot: () => ({ ...state, ticked: [...state.ticked] }),
        destroy() {
            destroyed = true;
            clearTimer(timer);
            disposeFade?.();
            host?.removeEventListener('click', onClick);
            host?.removeEventListener('change', onChange);
            host = null;
        },
    };
}
