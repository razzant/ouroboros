// Dashboard › Costs, "By project and day": one room's spend on each day of a span (`GET /api/cost-breakdown?view=
// project_days&project_days=<selector>&days=7|30`, that view alone, so a choice never reads the page's totals again;
// shape: management_contract_types.js `CostProjectDaysResponse`). The rooms are the live catalog's projects
// (`GET /api/projects?all=1`, closed ones too), Main, and the records outside both (another room such as a place, or
// none known), each a selector of its own (`project:<id>`, `main`, `unassigned`). A deleted project's records are no
// other room's: they stay in the totals above, and its name never comes back here. The view covers [since, until), UTC,
// from the oldest day's midnight to the moment it was asked; its rows are the days with records, newest first, and
// every day of the span is a row here, a day without records said so. Only records with a known date are in it: older ones stay in the totals above. Days and the
// span share the ledger's amount and price basis: an unpriced day is unknown, a measured zero is $0, and both can stand
// beside calls still open, unreconciled or unpriced. Only presentation rounds the amounts. Every word said from a view
// is composed here, one phrase per fact (`fmt`/`tx`/`plural`), so the overlay leaves those lines alone and a language
// change says the accepted view again without asking for it (`repaint`).
import { apiClient } from './api_client.js';
import { currentLanguage, fmt, tx } from './i18n.js';
import { formatUsd, number, plural } from './storage_format.js';
import { escapeHtmlAttr } from './ui_primitives.js';

export const PROJECT_DAY_SPANS = [[7, '7 days'], [30, '30 days']];
const DAY_MS = 86400000;
const MAIN = 'main';
const UNASSIGNED = 'unassigned';

const money = (value) => (value === null || value === undefined || value === '' || !Number.isFinite(Number(value)) ? null : Number(value));
const count = (value) => Math.max(0, Math.trunc(Number(value) || 0));
const utcDay = (ms) => new Date(ms).toISOString().slice(0, 10);

// Why a view could not be had, in the owner's words (the view's `error_code`).
const UNAVAILABLE_WORDS = Object.freeze({
    ledger_unavailable: 'The usage ledger could not be read, so this room’s spending is unknown.',
    project_source_unavailable: 'Room assignments could not be read, so this room’s spending is unknown.',
    unknown_selector: 'This room could not be found in the project registry.',
    unknown_days: 'This span cannot be counted.',
});

/** Main, catalog projects in order, and records outside both. A project's label is its unmodified authored name. */
export function projectDayRooms(projects = []) {
    const rows = (Array.isArray(projects) ? projects : []).filter((row) => row?.id && row.lifecycle !== 'tombstoned').map((row) => {
        return { value: `project:${row.id}`, label: String(row.name || ''), group: 'project', closed: Boolean(row.closed) };
    });
    return [{ value: MAIN, label: 'Main', group: '' }, ...rows, { value: UNASSIGNED, label: 'Other or unknown room', group: '' }];
}

/** One amount's words over a row (or a span's sums) of the view: `{settled, text, detail, confirmed, estimated, open}`;
 *  `settled` null when no record of it has a price. The view tells a measured $0 from none (null), so a zero stays
 *  $0.00 beside unpriced records or calls still open. An estimate is said as one and nothing more: a recorded estimate
 *  is promised no exact price later. Only calls with no outcome (`open_rows`) are said to be open; a call that ended
 *  unreconciled (`unresolved_rows`, an answer may have arrived) is said so, never as one still running. */
export function dayWords(row, { incomplete = false } = {}) {
    const settled = money(row?.settled_usd);
    const confirmed = money(row?.confirmed_usd) || 0;
    const estimated = money(row?.estimated_usd) || 0;
    const unpriced = count(row?.unknown_unmetered);
    const open = count(row?.open_rows);
    const unreconciled = count(row?.unresolved_rows);
    const sessions = count(row?.subscription_sessions);
    const amount = settled === null ? '' : formatUsd(settled);
    const text = settled === null ? tx('Unknown') : (unpriced || incomplete) ? fmt('Tracked: {amount}', { amount }) : amount;
    const basis = row?.price_basis === 'mixed' ? fmt('{amount} of it estimated', { amount: formatUsd(estimated) })
        : row?.price_basis === 'estimated' ? tx('estimated') : '';
    const notes = [basis,
        open ? plural(open, '{n} call still open', '{count} calls still open') : '',
        unreconciled ? plural(unreconciled, '{n} call not reconciled', '{count} calls not reconciled') : '',
        unpriced ? fmt('{count} without a full price', { n: unpriced, count: number(unpriced) }) : '',
        sessions ? plural(sessions, '{n} subscription session', '{count} subscription sessions') : '',
        incomplete ? tx('ledger incomplete') : ''].filter(Boolean);
    return { settled, text, detail: notes.join(' · '), confirmed, estimated, open: open > 0 };
}

/** The UTC calendar days of [since, until), newest first: an `until` at midnight ends the day before it. */
export function spanDays(since, until) {
    const start = Date.parse(String(since || ''));
    const end = Date.parse(String(until || ''));
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return [];
    const first = utcDay(start);
    const last = utcDay(end - 1);
    const days = [];
    for (let at = Date.parse(`${first}T00:00:00Z`); days.length < 400; at += DAY_MS) {
        const day = utcDay(at);
        if (day > last) break;
        days.push(day);
    }
    return days.reverse();
}

const UTC_DAY_FORMAT = Object.freeze({ weekday: 'short', month: 'short', day: 'numeric', timeZone: 'UTC' });

/** One UTC day formatter per render, bound to the active BCP-47 UI language (region included) and falling back to
 *  English for an empty or unknown tag — never the browser's own locale. */
function dayFormatter() {
    const lang = String(currentLanguage() || '').trim();
    let locale = 'en-US';
    if (lang && typeof Intl !== 'undefined' && typeof Intl.DateTimeFormat === 'function') {
        try {
            const [matched] = Intl.DateTimeFormat.supportedLocalesOf([lang]);
            if (matched) locale = matched;
        } catch { /* fall back to English */ }
    }
    if (typeof Intl !== 'undefined' && typeof Intl.DateTimeFormat === 'function') {
        try {
            const formatter = new Intl.DateTimeFormat(locale, UTC_DAY_FORMAT);
            return (day) => formatter.format(new Date(`${day}T00:00:00Z`));
        } catch { /* fall back below */ }
    }
    return (day) => new Date(`${day}T00:00:00Z`).toLocaleDateString('en-US', UTC_DAY_FORMAT);
}

/** The card's words for one view: `{state, total, days, note}`, `state` one of available, empty, unavailable. */
export function projectDaysPresentation(view) {
    if (!view || view.available !== true || !Array.isArray(view.rows)) {
        const why = UNAVAILABLE_WORDS[view?.error_code] || 'This room’s spending could not be read.';
        return { state: 'unavailable', total: null, days: [], note: tx(why) };
    }
    const unassigned = view.selector !== UNASSIGNED ? ''
        : ` ${tx('These records belong to neither Main nor a project: another room, such as a place, or none known.')}`
        + ` ${tx('A deleted project’s records are in the totals above, not in this room.')}`;
    const degraded = view.integrity_degraded === true;
    const degradedNote = degraded
        ? ` ${tx('Some earlier ledger rows were quarantined, so this view may be incomplete.')}`
        : '';
    const scope = `${tx('UTC days · only records with a known date.')}${unassigned}${degradedNote}`;
    if (!view.rows.length) {
        const span = Number(view.days) || 0;
        const none = fmt(
            degraded ? 'No available records with a known date in the last {count} UTC days.'
                : 'No records with a known date in the last {count} UTC days.',
            { n: span, count: number(span) },
        );
        return { state: 'empty', total: null, days: [], note: `${none}${unassigned}${degradedNote}` };
    }
    const byDay = new Map(view.rows.map((row) => [String(row?.day || ''), row]));
    const end = Date.parse(String(view.until || ''));
    // The day `until` falls in is still being counted, unless `until` is that day's midnight (then it is not in the span).
    const counting = Number.isFinite(end) && end % DAY_MS !== 0 ? utcDay(end) : '';
    const all = [...new Set([...spanDays(view.since, view.until), ...[...byDay.keys()].filter((day) => /^\d{4}-\d{2}-\d{2}$/.test(day))])]
        .sort().reverse();
    const formatDay = dayFormatter();
    const days = all.map((day) => {
        const row = byDay.get(day);
        return { day, label: formatDay(day), counting: day === counting, quiet: !row, words: row ? dayWords(row) : null };
    });
    const peak = Math.max(0, ...days.map((item) => item.words?.settled || 0));
    for (const item of days) {
        item.confirmedShare = peak > 0 && item.words ? (item.words.confirmed / peak) * 100 : 0;
        item.estimatedShare = peak > 0 && item.words ? (item.words.estimated / peak) * 100 : 0;
    }
    const total = dayWords(view.total, { incomplete: degraded });  // the ledger adds before any presentation rounding
    const calls = count(view.total?.physical_calls);
    return {
        state: 'available',
        total: { ...total, calls: calls ? plural(calls, '{n} model call', '{count} model calls') : '' },
        days,
        note: scope,
    };
}

function optionHtml(room) {
    // A project's own name is authored (never translated); one absent from the list keeps none and claims no cause.
    const authored = room.group === 'project' && !room.unlisted;
    const name = room.unlisted ? tx('Project not in list') : authored && !room.label ? tx('Unnamed project') : room.label;
    const label = room.closed ? fmt('{name} (closed)', { name }) : name;
    return `<option value="${escapeHtmlAttr(room.value)}"${authored ? ' data-i18n-authored' : ''}>${escapeHtmlAttr(label)}</option>`;
}

function roomOptions(rooms) {
    const projects = rooms.filter((room) => room.group === 'project');
    const own = rooms.filter((room) => room.group !== 'project');
    // The overlay translates no `label` attribute: the group's word is translated where it is written.
    return [optionHtml(own[0]), projects.length ? `<optgroup label="${escapeHtmlAttr(tx('Projects'))}">${projects.map(optionHtml).join('')}</optgroup>` : '',
        ...own.slice(1).map(optionHtml)].join('');
}

const dayHtml = (item) => `<li class="costs-room-day"${item.quiet ? ' data-quiet' : ''}${item.counting ? ' data-counting' : ''}>
        <span class="costs-room-date">${escapeHtmlAttr(item.label)}${item.counting ? ` <span class="costs-room-sofar">${escapeHtmlAttr(tx('so far'))}</span>` : ''}</span>
        ${item.quiet ? `<span class="costs-room-none">${escapeHtmlAttr(tx('No records'))}</span>` : `${item.words.settled === null
        ? '<span class="costs-room-gap" aria-hidden="true"></span>' : `<span class="costs-room-bar" aria-hidden="true">${item.confirmedShare > 0
        ? `<i data-part="confirmed" style="--share:${item.confirmedShare.toFixed(2)}%"></i>` : ''}${item.estimatedShare > 0
        ? `<i data-part="estimated" style="--share:${item.estimatedShare.toFixed(2)}%"></i>` : ''}</span>`}
        <span class="costs-room-amount">${escapeHtmlAttr(item.words.text)}${item.words.detail ? `<span class="costs-room-note">${escapeHtmlAttr(item.words.detail)}</span>` : ''}</span>`}
    </li>`;

/** One room's view: the answer's `project_days`, a refusal included (the view alone answers 200); none when unread. */
async function readProjectDays(selector, days, signal) {
    try {
        return (await apiClient.costProjectDays(selector, days, { signal }))?.project_days ?? null;
    } catch (error) {
        if (error?.name === 'AbortError' || signal?.aborted) throw error;
        return null;
    }
}

const readRooms = async (signal) => (await apiClient.projectList({ signal }))?.projects;

/** The catalog and the selected day range read independently. A selection invalidates only its own figures;
 *  leaving discards both pending reads. A failed catalog read keeps the last good list and the selected room.
 *  `repaint` says the rooms and the accepted view again (a language change), asking nothing. */
export function mountCostProjectDays(host, { load = readProjectDays, rooms: listRooms = readRooms, days = 7 } = {}) {
    let listed = projectDayRooms();
    let choicesHtml = roomOptions(listed);
    host.innerHTML = `<section class="ui-card costs-rooms" aria-labelledby="costs-rooms-title">
        <h3 class="dashboard-card-title" id="costs-rooms-title">By project and day</h3>
        <p class="costs-table-note">What one room’s records cost on each day: Main, a project, or the records outside both.</p>
        <div class="costs-rooms-controls">
            <label class="costs-rooms-pick"><span class="costs-rooms-pick-label">Room</span>
                <select class="ui-control" data-cost-room>${choicesHtml}</select></label>
            <div class="costs-period-choice" role="group" aria-label="Days">${PROJECT_DAY_SPANS.map(([key, label]) =>
        `<button type="button" class="ui-segment" data-cost-room-span="${key}" aria-pressed="false">${label}</button>`).join('')}</div>
        </div>
        <p class="ui-status costs-rooms-status" role="status" aria-live="polite"></p>
        <p class="costs-rooms-total" data-i18n-fmt="" hidden></p>
        <ol class="costs-rooms-days" aria-label="Days" data-i18n-fmt=""></ol>
        <p class="costs-copy costs-rooms-note" data-i18n-fmt=""></p>
        <p class="costs-copy" data-cost-room-catalog-note role="status" hidden></p>
    </section>`;
    const select = host.querySelector('[data-cost-room]');
    const status = host.querySelector('.costs-rooms-status');
    const totalEl = host.querySelector('.costs-rooms-total');
    const daysEl = host.querySelector('.costs-rooms-days');
    const noteEl = host.querySelector('.costs-rooms-note');
    const catalogNoteEl = host.querySelector('[data-cost-room-catalog-note]');
    let span = Number(days) || 7;  // a week first: a month of mostly quiet days is the owner's explicit choice
    let chosen = MAIN;
    let pendingDays = null;
    let pendingRooms = null;
    let shown;  // the accepted view the figures say (null: a failed read); undefined while a choice's figures are cleared

    function paintRooms() {
        // Absence from a refreshed list does not prove deletion (rows can be omitted or unreadable): keep the choice
        // under a neutral label so its days stay its own without bringing back its earlier authored name.
        const unlisted = listed.some((room) => room.value === chosen) ? null : { value: chosen, label: '', group: 'project', unlisted: true };
        const html = roomOptions(unlisted ? [...listed.slice(0, -1), unlisted, listed.at(-1)] : listed);
        if (choicesHtml !== html) select.innerHTML = choicesHtml = html;
        select.value = chosen;
    }

    async function refreshRooms(ifIdle) {
        if (ifIdle && pendingRooms) return;
        pendingRooms?.abort();
        const control = new AbortController();
        pendingRooms = control;
        let projects;
        try { projects = await listRooms(control.signal); } catch { projects = null; }
        if (pendingRooms !== control) return;
        pendingRooms = null;
        const available = Array.isArray(projects);
        if (available) {
            listed = projectDayRooms(projects);
            paintRooms();
        }
        catalogNoteEl.hidden = available;
        catalogNoteEl.textContent = available ? '' : 'The project list could not be read. Refresh to try again.';
    }

    function paint() {
        if (shown === undefined) return;
        const words = projectDaysPresentation(shown);
        totalEl.hidden = !words.total;
        totalEl.innerHTML = words.total ? `<b>${escapeHtmlAttr(words.total.text)}</b>${[words.total.detail, words.total.calls].filter(Boolean)
            .map((part) => ` <span class="costs-room-note">${escapeHtmlAttr(part)}</span>`).join('')}` : '';
        daysEl.innerHTML = words.days.map(dayHtml).join('');
        noteEl.textContent = words.note;
    }

    async function refreshDays(ifIdle) {
        if (ifIdle && pendingDays) return;
        host.querySelectorAll('[data-cost-room-span]').forEach((button) => {
            const active = Number(button.dataset.costRoomSpan) === span;
            button.classList.toggle('active', active);
            button.setAttribute('aria-pressed', String(active));
        });
        status.textContent = 'Counting…';
        pendingDays?.abort();
        const control = new AbortController();
        pendingDays = control;
        let view;
        try { view = await load(chosen, span, control.signal); } catch { view = null; }
        if (pendingDays !== control) return;
        pendingDays = null;
        shown = view;
        paint();
        status.textContent = '';
    }

    function reload({ ifIdle = false } = {}) {
        return Promise.all([refreshRooms(ifIdle), refreshDays(ifIdle)]);
    }

    function selectionChanged() {
        shown = undefined;
        totalEl.hidden = true;
        totalEl.innerHTML = '';
        daysEl.innerHTML = '';
        noteEl.textContent = '';
        void refreshDays(false);
    }

    host.addEventListener('click', (event) => {
        const button = event.target.closest?.('[data-cost-room-span]');
        if (!button || Number(button.dataset.costRoomSpan) === span) return;
        span = Number(button.dataset.costRoomSpan);
        selectionChanged();
    });
    select.addEventListener('change', () => {
        if (chosen === select.value) return;
        chosen = select.value;
        selectionChanged();
    });
    function leave() {
        const cut = pendingDays !== null || pendingRooms !== null;
        pendingDays?.abort();
        pendingRooms?.abort();
        pendingDays = pendingRooms = null;
        status.textContent = '';
        return cut;
    }
    function repaint() {
        paintRooms();
        paint();
    }
    return { reload, leave, repaint };
}
