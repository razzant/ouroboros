// The first-run wizard's Skills step, between Consciousness and Review limits (owner 2026-11-21: «показать что есть
// много крутых скилов анимированным списком, но юзер может поснимать галочки»; it replaces the Summary row his 4A of
// 2026-10-17 chose). The server offers it only where completion honors the ticks (`stepOrder`: this data root's
// starter list never ran). One categorized list, built-ins first, one row per skill (his 21.11 «слишком крупные
// карточки»: rows, not boxes): its box, its human title and, muted on one line with it, what it gives and, in a few
// words, what it lacks («· Gemini key»; the full sentence is the row's accessible text); the maintainer's recommended
// ones are ticked. A group says its name only, and categories of one skill close the list together as «More». Where
// the skills come from and how the hub's install is one small sentence under the list, its choices behind it. Inside a
// group what works at once leads and what needs a key comes last. The rows arrive in one short sequence on entering
// the step, both columns at one pace, never on a repaint (DESIGN §7). A skill that declares a connection card (`connection`) stands first, as that card
// (onboarding_connection_card.js). Nothing installs here: the one atomic Finish carries the ticked ids, the trust
// choices and a card's values, and the host installs them in the background after start. Summary shows one row back
// to the step. A list that cannot be read leaves the host's defaults in place and says so. Where the step is not
// offered (a returning install), Summary's row says skills change in Extensions › Skills and keeps the hub's trust
// sentence with its choices, which Finish still applies.
import { skillSourcesApi } from './api_client.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { fmt } from './i18n.js';
import { createConnectionCard, normalizeConnection } from './onboarding_connection_card.js';

const BUILT_IN_TITLE = 'Built in';
const MORE = { category: 'more', label: 'More' };  // where categories of a single skill close the list together
const ARRIVE_LAST = 12;  // a column's twelfth place and later arrive together: the whole entrance stays within 700 ms
const FOCUS_KEYS = ['starterItem', 'starterAll', 'starterTrustToggle', 'starterTrust', 'connectionToggle', 'connectionToken',
    'connectionCheck', 'connectionOption'];

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

// How a hub row's skills are named in the sentence under the list and behind it (Extensions › Sources lists the rows
// themselves): in the sentence the Official row also says where they come from.
const SOURCE_WORDS = { official: [(row) => `Official skills from the ${row.name} hub`, () => 'Official skills'],
    community: [() => 'skills from the community', () => 'Community skills'] };
const sourceWords = (row, inLine) => SOURCE_WORDS[row.part]?.[inLine ? 0 : 1](row) || `${inLine ? 'skills from ' : ''}${row.name}`;
const capital = (text) => text.charAt(0).toUpperCase() + text.slice(1);

const quote = (value) => String(value).replace(/["\\]/g, '\\$&');
// Where a row sits in the page's layout, whatever an entrance's transform does to it at this moment.
const layoutTop = (node) => { let top = 0; for (let at = node; at; at = at.offsetParent) top += at.offsetTop; return top; };
const dataAttr = (key) => `data-${key.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}`;

/** The step and its Summary row. `offered`: the bootstrap's `stepOrder` has the step (else only Summary's row shows,
 * with where skills change and the hub's trust choices, and only those ride Finish); `open` returns to the step from
 * Summary. `read()` defaults to GET
 * /api/onboarding/starter-skills; `card` passes the connection card its check, randomness and encoder (tests). */
export function createStarterSkillsStep({ offered = true, open = () => {}, keyLabels = {}, read = () => skillSourcesApi.starterSkills(), setTimer = setTimeout, clearTimer = clearTimeout, now = Date.now, card = {} } = {}) {
    const state = { status: 'loading', groups: [], ticked: new Set(), trust: {}, rows: [], sources: [], more: 0, trustOpen: false, error: '', ticksChanged: false, untickNew: false };
    let host = null;
    let view = '';  // what the mounted node shows: the step's list, Summary's row, or nothing
    let painted = '';
    let arrived = false;  // this entry of the step has shown its rows once
    let destroyed = false;
    let timer = null;
    let deadline = null;
    let connection = null;  // the first item's declared connection card, made once and kept until Finish
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
            state.more = Number.isInteger(data?.catalog_more) && data.catalog_more > 0 ? data.catalog_more : 0;
            for (const row of state.rows) state.trust[row.id] ??= row.trust === 'trusted' ? 'trusted' : 'review';
            state.status = items().length || state.rows.length ? 'ready' : 'empty';
            state.error = '';
            const declared = !connection && items().find((item) => normalizeConnection(item.connection));
            if (declared) {
                connection = createConnectionCard(declared, normalizeConnection(declared.connection), { ...card, repaint: () => paint(), edited: () => { touched = true; },
                    ticked: () => state.ticked.has(declared.id), tick: () => { state.ticksChanged = true; state.ticked.add(declared.id); } });
            }
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

    function pendingText() {
        return [reading() ? 'Reading the Ouroboros hub…' : '',
            useDefaults() ? 'If you finish before it is ready, Ouroboros installs the default starter list once it can read the hub.' : '',
            state.error && state.status === 'ready' ? `The list couldn’t be refreshed (${state.error}).` : ''].filter(Boolean).join(' ');
    }
    const unreadText = () => `The list couldn’t be read (${state.error}). Ouroboros installs the recommended skills after it starts. Manage them in Extensions › Skills.`;
    // Counts are composed words (`fmt`): the span keeps its mark, so the overlay never keys a number of its own.
    const COUNT = '{n} of {total} ticked';
    const countText = () => fmt(COUNT, { n: tickedItems().length, total: items().length });

    function rowHtml() {
        if (!offered) {  // no ticks Finish would ignore: where skills change, and the trust choices Finish still applies
            return `<div class="summary-kv skills-starter-row"><strong>Skills</strong><span class="skills-starter-summary">
                <span class="skills-starter-value">Add or remove them in Extensions › Skills</span></span></div>${state.status === 'ready' ? trustHtml() : ''}`;
        }
        const count = tickedItems().length;
        const fixed = { loading: 'Reading the list…', error: 'The recommended set', empty: 'None offered' }[state.status]
            || (count ? '' : 'None will install');
        const value = fixed ? escapeHtml(fixed) : `<span data-i18n-fmt="{n} will install">${escapeHtml(fmt('{n} will install', { n: count }))}</span>`;
        const sub = state.status === 'error' ? unreadText() : pendingText();
        const card = connection?.summary();
        return `<div class="summary-kv skills-starter-row"><strong>Skills</strong><span class="skills-starter-summary">
            <span class="skills-starter-value">${value} · <button type="button" class="skills-starter-change" data-starter-change>Change</button></span>
            ${sub ? `<span class="skills-starter-sub" role="status">${escapeHtml(sub)}</span>` : ''}</span></div>${card ? `
            <div class="summary-kv skills-starter-row"><strong>${escapeHtml(card[0])}</strong><span class="skills-starter-summary">
            <span class="skills-starter-value"><span${card[2] ? ' data-i18n-authored' : ''}>${escapeHtml(card[1])}</span> · <button type="button" class="skills-starter-change" data-starter-change>Change</button></span></span></div>` : ''}`;
    }

    // Inside a group: recommended skills that work at once, then the rest that do, then those that need something first.
    const rank = (item) => (needsText(item, keyLabels) ? 2 : 0) + (item.recommended ? 0 : 1);
    const label = (item) => String(item.title || item.name || '');
    const shown = (group) => group.items.filter((item) => item !== connection?.item && item.id !== connection?.item.id)
        .sort((a, b) => rank(a) - rank(b) || label(a).localeCompare(label(b), 'en'));

    // One row: the box, the title and, muted on one line with it (wrapping whole under it where the line is short), what
    // it gives and, quieter, what it lacks in a few words: the catalog's `mark`, else «needs setup»; the whole sentence
    // stays the row's accessible text.
    function skillHtml(item, order) {
        const on = state.ticked.has(item.id);
        const title = String(item.title || '').trim();
        const pitch = String(item.pitch || '').trim();
        const needs = needsText(item, keyLabels);
        const mark = needs ? String(item.mark || '').trim() : '';
        // The catalog's own words translate (the interface layer); a name, a skill author's text and a line composed
        // from setting names stay as they are.
        const authored = (own) => (own ? '' : ' data-i18n-authored');
        return `<label class="skills-starter-skill" data-starter-skill="${escapeHtml(item.id)}" style="--arrive: ${order}">
            <input type="checkbox" class="ui-checkbox" data-starter-item="${escapeHtml(item.id)}"${on ? ' checked' : ''}>
            <span class="skills-starter-words"><strong${authored(title)}>${escapeHtml(title || item.name)}</strong>
                <span class="skills-starter-pitch"${authored(pitch)}>${escapeHtml(pitch || item.description || '')}</span>${needs ? `
                <span class="skills-starter-needs" aria-hidden="true">· ${escapeHtml(mark || 'needs setup')}</span><span class="skills-starter-sr"${authored(typeof item.needs === 'string')}>${escapeHtml(needs)}</span>` : ''}</span>
        </label>`;
    }

    // The list's groups: each category with its rows; categories of one skill close it together as «More».
    function listed() {
        const groups = state.groups.map((group) => [group, shown(group)]).filter(([, rows]) => rows.length);
        const single = groups.filter(([, rows]) => rows.length === 1).flatMap(([, rows]) => rows);
        if (!single.length) return groups;
        return [...groups.filter(([, rows]) => rows.length > 1),
            [MORE, single.sort((a, b) => rank(a) - rank(b) || label(a).localeCompare(label(b), 'en'))]];
    }

    // Where the hub's skills come from and how they install, in plain words and said once, under the list: e.g.
    // «Official skills from the Ouroboros hub install without a review; skills from the community are reviewed first»;
    // the two choices per row (Settings' words) wait behind it.
    function trustHtml() {
        if (!state.rows.length) return '';
        const say = (trusted, verb) => {
            const names = state.rows.filter((row) => (state.trust[row.id] === 'trusted') === trusted).map((row) => sourceWords(row, true));
            return names.length ? `${names.join(' and ')} ${verb}` : '';
        };
        const line = capital([say(true, 'install without a review'), say(false, 'are reviewed first')].filter(Boolean).join('; ')) + '.';
        const rows = state.rows.map((row) => `<span class="skills-starter-trust-row"><strong>${escapeHtml(sourceWords(row, false))}</strong>
            ${[['trusted', 'Install without review'], ['review', 'Review each skill']].map(([value, choice]) => `<button type="button" class="selection-pill${state.trust[row.id] === value ? ' active' : ''}" aria-pressed="${state.trust[row.id] === value ? 'true' : 'false'}" data-starter-trust="${escapeHtml(row.id)}" data-trust="${value}">${choice}</button>`).join('')}</span>`).join('');
        return `<div class="skills-starter-trust"><p class="skills-starter-sub">${escapeHtml(line)} <button type="button" class="skills-starter-change" data-starter-trust-toggle aria-expanded="${state.trustOpen}">${state.trustOpen ? 'Hide' : 'Change'}</button></p>
            ${state.trustOpen ? `${rows}<span class="skills-starter-sub">Suggested choices apply until you choose.</span>` : ''}</div>`;
    }

    function stepHtml(arrive) {
        if (state.status !== 'ready') {
            const words = { loading: 'Reading the list…', error: unreadText(), empty: 'No skills are offered here. You can add skills any time in Extensions › Browse.' }[state.status];
            return `<p class="skills-starter-sub" role="status">${escapeHtml(words)}</p>`;
        }
        let order = 0;
        const next = () => Math.min(order++, ARRIVE_LAST);
        const pending = pendingText();
        // A declared connection card leads; a group is its name over its rows. Under the list, one muted line counts
        // the catalog's other skills (owner 21.11, 4A), none while it is unread; what is still being read stands
        // there too, where the hub's skills will join the list: its going moves no row.
        const groups = listed();
        const more = state.more === 1 ? '{n} more skill in the catalog, after start.' : '{n} more skills in the catalog, after start.';
        return `<div class="skills-starter-list"${arrive ? ' data-arrive' : ''}>
            ${connection ? connection.html(next()) : ''}
            ${items().length ? `<div class="skills-starter-bar"><span class="skills-starter-count" role="status" data-i18n-fmt="${COUNT}">${escapeHtml(countText())}</span><span class="skills-starter-bulk"><button type="button" data-starter-all="1">Tick all</button><span aria-hidden="true">·</span><button type="button" data-starter-all="0">Untick all</button></span></div>` : ''}
            <div class="skills-starter-groups">${groups.map(([group, rows]) => `<section class="skills-starter-group"><h3 style="--arrive: ${Math.min(order, ARRIVE_LAST)}">${escapeHtml(group.label)}</h3>
                ${rows.map((item) => skillHtml(item, next())).join('')}</section>`).join('')}</div>
            ${state.more ? `<p class="skills-starter-sub" data-starter-more data-i18n-fmt="${more}">${escapeHtml(fmt(more, { n: state.more }))}</p>` : ''}
            ${pending ? `<p class="skills-starter-sub" role="status">${escapeHtml(pending)}</p>` : ''}
            ${trustHtml()}
        </div>`;
    }

    // The reading place a repaint keeps (DESIGN principle 6): the focused control and the first row in view.
    function readingPlace() {
        if (typeof host.querySelectorAll !== 'function') return null;
        const active = host.ownerDocument?.activeElement;
        const key = active && host.contains(active) ? FOCUS_KEYS.find((name) => active.dataset?.[name] !== undefined) : '';
        const focus = key ? `[${dataAttr(key)}="${quote(active.dataset[key])}"]${key === 'starterTrust' ? `[data-trust="${quote(active.dataset.trust)}"]` : ''}` : '';
        const caret = focus && typeof active.selectionStart === 'number' ? [active.selectionStart, active.selectionEnd] : null;
        const row = [...host.querySelectorAll('[data-starter-skill]')].find((node) => node.getBoundingClientRect().bottom > 0);
        const anchor = row && { id: row.dataset.starterSkill, top: layoutTop(row) };
        return () => {
            const field = focus ? host.querySelector(focus) : null;
            field?.focus({ preventScroll: true });
            if (caret && field?.setSelectionRange) field.setSelectionRange(...caret);  // typing goes on where it was
            const again = anchor && host.querySelector(`[data-starter-skill="${quote(anchor.id)}"]`);
            const moved = again ? layoutTop(again) - anchor.top : 0;
            if (moved) host.ownerDocument?.defaultView?.scrollBy(0, moved);
        };
    }

    // Chromium reports a focused field's unreported edit (`change`) from inside the rewrite that removes it; a repaint
    // the card's handler asks for then would rewrite the list under that rewrite, which throws. It follows instead.
    let painting = false;
    let touched = false;  // the card changed its markup in place since the last write: the next paint writes anew
    function paint() {
        if (!host?.isConnected) return;
        if (painting) { queueMicrotask(paint); return; }
        const html = view === 'step' ? stepHtml(false) : view === 'row' ? rowHtml() : '';
        if (html === painted && !touched) return;  // a read that changed nothing rewrites nothing
        const arrive = view === 'step' && !arrived && state.status === 'ready' && items().length > 0;
        const restore = painted && view === 'step' ? readingPlace() : null;
        painting = true;
        try { host.innerHTML = arrive ? stepHtml(true) : html; } finally { painting = false; }
        painted = html;
        touched = false;
        connection?.painted();  // a QR the page now shows is one the person could scan: only then may its code ride
        arrived ||= arrive;
        if (arrive) paceColumns();
        restore?.();
    }

    // Both columns build at one pace: a group's head and rows take their place within their own column (the card and
    // anything above the columns come first), set before the first frame so the entrance starts with them.
    function paceColumns() {
        if (typeof host.querySelectorAll !== 'function') return;
        const places = new Map();
        for (const node of host.querySelectorAll('.skills-starter-group h3, .skills-starter-group .skills-starter-skill')) {
            const column = Math.round(node.getBoundingClientRect().left);
            const place = places.get(column) || 0;  // each column's head starts with the card, at once
            places.set(column, place + 1);
            node.style.setProperty('--arrive', String(Math.min(place, ARRIVE_LAST)));
        }
    }

    function onClick(event) {
        if (event.target.closest('[data-starter-change]')) { open(); return; }
        if (connection?.click(event.target)) return;
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
        if (connection?.change(event.target)) return;
        const box = event.target.closest('[data-starter-item]');
        if (!box) return;
        state.ticksChanged = true;
        if (box.checked) state.ticked.add(box.dataset.starterItem);
        else state.ticked.delete(box.dataset.starterItem);
        const count = host.querySelector('.skills-starter-count');
        if (count) count.textContent = countText();
    }
    const onInput = (event) => connection?.input(event);
    const onKeydown = (event) => connection?.keydown(event);
    const listen = (node, on) => [['click', onClick], ['change', onChange], ['input', onInput], ['keydown', onKeydown]]
        .forEach(([type, fn]) => node?.[on ? 'addEventListener' : 'removeEventListener'](type, fn));

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
         * survive re-renders, and the rows arrive once each time the step is entered. */
        mount(node) {
            if (host === node) return;
            listen(host, false);
            host = node && (offered || node.dataset?.starterSkills === 'row') ? node : null;
            const next = !host ? '' : host.dataset?.starterSkills === 'step' ? 'step' : 'row';
            if (next === 'step' && view !== 'step') arrived = false;
            view = next;
            painted = '';
            if (!host) return;
            if (deadline === null) {
                deadline = now() + 180_000;
                if (state.status !== 'loading' && reading()) timer = setTimer(refresh, 3000);
            }
            listen(host, true);
            paint();
        },
        /** What the one atomic Finish carries; nothing when the list could not be read, and only the trust choices when
         * the step is not offered. A connection card's values ride as `connections`, apart from the settings (DESIGN §7). */
        payload() {
            if (state.status !== 'ready') return {};
            if (!offered) return { skill_source_trust: { ...state.trust } };
            const connections = connection?.payload();
            return {
                ...(useDefaults() ? {} : { starter_skills: tickedItems().map((item) => item.id) }),
                skill_source_trust: { ...state.trust },
                ...(connections ? { connections } : {}),
            };
        },
        snapshot: () => ({ ...state, ticked: [...state.ticked], view, arrived, connection: connection?.snapshot() || null }),
        destroy() {
            destroyed = true;
            clearTimer(timer);
            listen(host, false);
            host = null;
        },
    };
}
