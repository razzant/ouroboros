import { formatUsd, number } from './storage_format.js';
import { apiFetch } from './api_client.js';
import { escapeHtmlAttr } from './ui_primitives.js';
import { bindSettingRows, readSettings, settingsRow } from './settings_controls.js';
import { rowsOf } from './settings_rows.js';
import { statusMarkHtml } from './status_mark.js';
import { API_PROVIDER_LABELS, harnessPresentation } from './harness_presentation.js';
import { readConsciousness, readConsciousnessFromState } from './consciousness_api.js';
import { allowanceHold, allowanceText } from './consciousness_words.js';
import { openTaskReference, taskReferenceHtml } from './task_reference.js';
import { mountCostPeriod } from './cost_period.js';
import { READ_OK, accountRows, facetReadState } from './claudexor_status_store.js';
import { humanizeResetAt, quotaSummary } from './harness_accounts.js';
import { controlIcon } from './page_icons.js';
import { fmt, translateString } from './i18n.js';
import { parseModelSource } from './route_editor_primitives.js';

// The limits card is rows of settings_rows.js (their home is this tab), drawn and written by settings_controls.js: each
// saves itself and says what happened. The consciousness allowance has its one editor here (owner, 10 Nov 2026: moved
// from Settings with the spend it counts); its 0 is a real choice (it may not spend).
const LIMITS_PAGE = 'dashboard:costs';
const ALLOWANCE_KEY = 'OUROBOROS_CONSCIOUSNESS_DAILY_USD';
// What else describes a limit's field: the note of how its limit counts, and for the allowance its spend and state.
const LIMIT_NOTES = Object.freeze({
    TOTAL_BUDGET: 'costs-budget-help', OUROBOROS_PER_TASK_COST_USD: 'costs-budget-help',
    [ALLOWANCE_KEY]: 'costs-allowance-spend costs-allowance-state costs-allowance-help',
});
const limitRows = (fold) => rowsOf(LIMITS_PAGE, { fold }).map((row) => settingsRow(row, { describedBy: ({ id }) => LIMIT_NOTES[id] || '' })).join('');
// A breakdown table shows its largest rows; the rest wait behind its "Show all (N)" (the list's one length control).
const TABLE_ROWS = 8;
// Each table says what it groups by (owner, Track09 A, 9 Nov 2026). The ledger's `by_api_key` groups by provider; a
// model's row is its total over every provider; the role table reads the roles in today's Settings, not the call's.
export const TABLES = [
    ['cost-by-model', 'By model', 'Model', 'by_model', 'Each model\u2019s total over every provider and connection.'],
    ['cost-by-key', 'By provider', 'Provider', 'by_api_key', ''],
    ['cost-by-model-cat', 'By current role', 'Role', 'by_model_category', 'Calls matched to the roles set in Settings now, not when each call ran.'],
    ['cost-by-task-cat', 'By task category', 'Category', 'by_task_category', ''],
];
const ROLE_WORDS = Object.freeze({ main: 'Main', heavy: 'Heavy', light: 'Light', fallback: 'Fallback', other: 'No current role' });
const providerWord = (key) => API_PROVIDER_LABELS[key] || (harnessPresentation(key).known ? harnessPresentation(key).label : key);
const NAME_WORDS = Object.freeze({ 'cost-by-key': providerWord, 'cost-by-model-cat': (key) => ROLE_WORDS[key] || key });
/** A breakdown row's name as the owner reads it: a provider's or a current role's word, else the recorded key. */
export const breakdownRowName = (tableId, key) => (NAME_WORDS[tableId] ? NAME_WORDS[tableId](key) : key);

// The setup contract is authoritative once /api/settings answers. Keep a useful
// non-empty explanation during the request and on a failed read so aria-describedby
// never points at an empty node; the fallback is deliberately shorter than the
// canonical note and is replaced whenever the contract is available.
const BUDGET_NOTE_FALLBACK =
    'Budget limits use known spend: Total Budget applies globally, while Per-task Cost Cap applies to one root task tree. '
    + 'Unpriced calls remain unknown and are not counted as $0; reservations and unresolved charges are not counted as spending.';

function optionalFiniteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

/** One breakdown cell in the task card's words: `Cost pending` while a call of it is still open and nothing is
 *  priced, `Cost not priced` once its calls settled without a price (subscription work among them: no price will
 *  arrive), `Tracked:` before a priced subtotal beside unpriced calls. The face is that amount alone, an open bucket
 *  marked; the whole sentence (`(pending)` after an amount more may join, the count of calls without a price — a
 *  subscription row may have no physical call) is the cell's `note`. Never an unpriced or open zero as free, never a
 *  positive amount as $0.00. `{ text, open, note }`. */
export function costBucketPresentation(info) {
    const cost = optionalFiniteNumber(info?.cost);
    if (cost === null) return { text: 'unknown', open: false, detail: '' };
    const unknown = Math.trunc(optionalFiniteNumber(info?.unknown_unmetered) || 0);
    const nonFinal = optionalFiniteNumber(info?.non_final_rows) || 0;
    // Rows that can still settle; an older payload without the count has its non-final rows past the unpriced ones.
    const open = (optionalFiniteNumber(info?.accounting_open_rows) ?? nonFinal - unknown) > 0;
    const word = cost === 0 && (open || unknown > 0);
    const text = word ? (open ? 'Cost pending' : 'Cost not priced') : `${unknown > 0 ? 'Tracked: ' : ''}${formatUsd(cost)}`;
    // What the amount leaves out is said on the face under it, never in a tooltip alone (DESIGN principle 7): more may
    // still join an open amount, and the calls that ended without a price are counted.
    const detail = [open && !word ? 'pending' : '', unknown > 0 ? `${number(unknown)} without a price` : '']
        .filter(Boolean).join(' · ');
    return { text, open, detail };
}

/** Pure cost-dashboard projection: null/unavailable never renders as $0.
 * The limit is set against KNOWN spend (settled: confirmed + estimated), the number that refuses new paid calls
 * (#1487), so the headline is known spend of the limit and what is left of it; reservations and unresolved upper
 * bounds are open holds, said beside it and never added in. The accounting facts are the record's (Details). */
export function costDashboardPresentation(data) {
    if (!data) return { state: 'loading' };
    const accounting = data.accounting || {};
    if (accounting.available === false) return { state: 'unavailable' };
    const known = optionalFiniteNumber(accounting.settled_usd);
    const confirmed = optionalFiniteNumber(accounting.confirmed_usd);
    const estimated = optionalFiniteNumber(accounting.estimated_usd);
    const reserved = optionalFiniteNumber(accounting.reserved_usd);
    const unresolved = optionalFiniteNumber(accounting.unresolved_upper_bound_usd);
    const unknown = optionalFiniteNumber(accounting.unknown_unmetered);
    const calls = optionalFiniteNumber(data.total_calls);
    if ([known, confirmed, estimated, reserved, unresolved, unknown, calls].some(value => value === null)) {
        return { state: 'unavailable' };
    }
    const rawLimit = optionalFiniteNumber(accounting.limit_usd);
    const limit = rawLimit !== null && rawLimit > 0 ? rawLimit : 0;
    // The admission rule's own room above known spend when the server says it; else the same subtraction.
    const remaining = limit > 0 ? (optionalFiniteNumber(accounting.remaining_known_usd) ?? Math.max(0, limit - known)) : null;
    const reached = limit > 0 && known >= limit;
    // A flag without its cause is not reconstructible. `cost_final: false` holds with every
    // dollar bucket at $0.00 and `unknown` at 0 — an ESTIMATED zero, or a dispatched row
    // whose reservation is exactly zero — so the count of open rows rides beside the flag
    // it explains rather than in a new tile nobody correlates. Open rows are the ledger's
    // `accounting_open_rows`; `non_final_rows` also counts calls that will never get a price
    // (subscription work), which "open" would misname. Absent on an older payload, the
    // older count stands in; with neither this says only what it knows.
    const openRows = optionalFiniteNumber(accounting.accounting_open_rows ?? accounting.non_final_rows);
    const openCause = openRows !== null && openRows > 0 ? ` (${number(Math.trunc(openRows))} open)` : '';
    return {
        state: 'available',
        figure: limit > 0 ? `${formatUsd(known)} of ${formatUsd(limit)}` : formatUsd(known),
        left: reached ? 'Limit reached: new paid work is refused' : limit > 0 ? `${formatUsd(remaining)} left` : 'No limit set',
        // The producer's words for the unresolved upper bound (cost_projection.uncounted_money_text): a worst case.
        open: unresolved > 0 ? `up to ${formatUsd(unresolved)} for calls without a known price` : '',
        reached,
        share: limit > 0 ? Math.min(100, Math.max(0, (known / limit) * 100)) : null,
        facts: [
            ['Final', accounting.cost_final === true ? 'Yes' : `No${openCause}`],
            ['Confirmed', formatUsd(confirmed)],
            ['Estimated', formatUsd(estimated)],
            ['Reserved', formatUsd(reserved)],
            ['Unresolved upper bound', formatUsd(unresolved)],
            ['Calls without a price', number(Math.trunc(unknown))],
            ['Physical attempts', number(Math.trunc(calls))],
        ],
    };
}

/** `21:41 GMT+3` (a date first when not today), the whole minute by which room is back after ``iso``, the instant after
 *  which it returns: rounded up past it, so the words are never early. The zone is named: the host sentences of the
 *  Consciousness tab keep the server's clock, this one the viewer's. '' for an unreadable or past instant. */
export function roomTimeText(iso, { now = Date.now(), timeZone } = {}) {
    const after = Date.parse(String(iso || ''));
    if (!Number.isFinite(after) || after < now) return '';
    const at = new Date(Math.floor(after / 60000) * 60000 + 60000);
    const zone = timeZone ? { timeZone } : {};
    const day = (date) => date.toLocaleDateString('en-CA', zone);  // YYYY-MM-DD in that zone
    const date = day(at) === day(new Date(now)) ? '' : `${at.toLocaleDateString([], { month: 'short', day: 'numeric', ...zone })}, `;
    return date + at.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false, timeZoneName: 'short', ...zone });
}

const NEW_STARTS = 'new wake-ups, replies in places and its own tasks';

/** The consciousness allowance under its field, from the consciousness snapshot (consciousness_api.js): the spend line
 *  is the clock's own window (`allowanceText`, the Shade's and the Dashboard's words), never a sum of the tables;
 *  `detail` what it leaves out (pending calls, calls without a price); `lines` at most three: the window (its limit
 *  reached, with a conditional time when room returns), a held wake (its proven cause, with its card's door) and the
 *  model its wake-ups think with when the host could read it, each its fact (`text`, in the line's tone) and what
 *  follows from it (`note`, quieter). Off, a $0 limit and an unreadable ledger say so instead. No time is guessed: a
 *  withheld or stale forecast says so. */
export function allowancePresentation(snapshot, { now = Date.now(), timeZone } = {}) {
    if (!snapshot) return { spend: 'What it spent in the last 24 hours could not be read.', detail: '', lines: [] };
    const spend = allowanceText(snapshot) || 'What it spent in the last 24 hours is unknown.';
    const detail = [snapshot.allowanceOpenRows > 0 ? 'pending' : '',
        snapshot.unknownUnmetered > 0 ? `${number(snapshot.unknownUnmetered)} without a price` : ''].filter(Boolean).join(' · ');
    const lines = [];
    if (snapshot.enabled === false) {
        lines.push({ text: 'Consciousness is off: it starts nothing that counts here until it is started again.', tone: 'muted' });
        return { spend, detail, lines };
    }
    const exhausted = snapshot.allowanceStatus ? snapshot.allowanceStatus === 'exhausted' : snapshot.status === 'allowance_exhausted';
    if (snapshot.dailyUsd === 0) {
        lines.push({ text: `At $0 consciousness may not spend: ${NEW_STARTS} do not start.`, tone: 'muted' });
    } else if (exhausted) {
        const time = roomTimeText(snapshot.allowanceRoomAfter, { now, timeZone });
        lines.push({ text: `Consciousness limit reached: ${NEW_STARTS} wait.`, tone: 'warn',
            note: time ? `If its charges stay as they are, there is room again by ${time}.` : 'When there is room again cannot be said now.' });
    } else if (snapshot.allowanceStatus === 'allowance_unknown' || snapshot.status === 'allowance_unknown') {
        lines.push({ text: 'The usage ledger could not be read, so whether consciousness has room is unknown.', tone: 'muted' });
    }
    const wake = snapshot.liveWakeTaskId;
    if (allowanceHold(snapshot)) {
        const facts = [snapshot.holdCeilingUsd !== null ? formatUsd(snapshot.holdCeilingUsd) : '',
            snapshot.holdCause === 'allowance_and_cap' ? 'the per-task cap too' : ''].filter(Boolean);
        lines.push({ wake, tone: 'warn', text: `The consciousness wake-up is paused at the allowance it had left when it started`
            + `${facts.length ? ` (${facts.join(', ')})` : ''}.`,
        note: 'Resume keeps that limit, so it pauses again while its known spend is at it; room returning or a higher '
            + 'limit does not raise it. Stop now on its card ends it and frees the clock for its next check.' });
    } else if (snapshot.liveWakeHeld && snapshot.holdCause === 'per_task_cap') {
        lines.push({ wake, tone: 'muted', text: 'The consciousness wake-up is paused at the per-task cost cap it started with; its card shows why.' });
    } else if (snapshot.liveWakeHeld && snapshot.holdCause === 'launch_ceiling') {
        lines.push({ wake, tone: 'muted', text: 'The consciousness wake-up is paused at the spending limit it started with; its card shows why.' });
    } else if (snapshot.liveWakeHeld && snapshot.holdCause === 'owner') {
        lines.push({ wake, tone: 'muted', text: 'The consciousness wake-up is on an explicit pause; its card shows who paused it and how to continue.' });
    }
    const route = snapshot.modelRoute;
    if (route) {  // the words of the Consciousness tab's row and the Models page: "Main model", "Local model"
        const name = route.local ? 'Local model' : parseModelSource(route.value).model || route.value;
        lines.push({ tone: 'muted', text: `Thinks with ${name}${route.layer === 'main' ? ' (Main model)' : ''}.` });
    }
    return { spend, detail, lines };
}

// What the allowance counts, in this card's own note: the shared note of the setup contract speaks of the total budget
// and the per-task cap alone (settings_setup_contract._BUDGET_NOTE).
const ALLOWANCE_NOTE = 'Known spend of its wake-ups, its replies in places and the work they start, over the last 24 hours: '
    + 'confirmed and estimated prices count, calls without a price add nothing to it, and calls still open may add more. '
    + `When it is reached, ${NEW_STARTS} wait until older spending leaves the 24 hours. A wake-up may spend at most `
    + 'what was left of this limit when it started (and no more than the per-task cap); there it pauses. A higher limit '
    + 'does not raise what it was given and Resume keeps it, so a wake-up whose known spend is still at it pauses again. '
    + 'Stop now on its card ends it, and the clock\'s next check starts a new wake-up if there is room. At $0 '
    + 'consciousness may not spend.';

/** The Subscriptions block's doors, from the passive quota read (`GET /api/claudexor/status?view=quota`: the roster and
 *  the windows the engine last observed; it starts and refreshes nothing). One per kind of account, in counted facts
 *  and never an average of several accounts: how many are in rotation, how many of them have room, how many are at
 *  their limit and when the first of those returns, how many were not checked. An account has room only when its
 *  window says a share used and nothing else (harness_accounts.js quotaSummary); any other answer, "availability not
 *  proven" included, is not checked. Then Accounts and keys, which also says when the roster could not be read.
 *  `{ path, name, meta, value, tone, harness }`, `path` a Settings path. */
export function subscriptionDoors(payload, { transportError = '', nowMs = Date.now() } = {}) {
    const accountsRead = facetReadState(payload, 'accounts', { transportError });
    const quotaRead = facetReadState(payload, 'quota', { transportError });
    const accounts = accountsRead === READ_OK ? accountRows(payload) : [];
    const count = (n, words) => (n ? `${n} ${words}` : '');
    const doors = [...new Set(accounts.map((row) => row.harness))].map((harness) => {
        const own = accounts.filter((row) => row.harness === harness);
        const rotation = own.filter((row) => row.enabled);
        const windows = rotation.map((row) => quotaSummary(payload?.quota || [], harness, row.profile_id,
            { quotaRead, nowMs, absences: payload?.quota_absences || [] }));
        const spent = windows.filter((window) => window.exhausted);
        const open = windows.filter((window) => !window.exhausted && /^\d+% used( · resets [^·]+)?$/.test(window.label)).length;
        const checked = spent.length + open;
        const returns = spent.map((window) => window.resetsAt).filter((at) => Number.isFinite(Date.parse(at))).sort((a, b) => Date.parse(a) - Date.parse(b))[0];
        const value = !rotation.length ? 'None in rotation' : quotaRead !== READ_OK ? 'Limits not checked' : !checked ? 'Usage not checked'
            : rotation.length === 1 ? (open ? 'Has room' : 'At its limit') : `${open} of ${rotation.length} have room`;
        return { path: `accounts/${harness}`, name: harnessPresentation(harness).label, harness, value, tone: checked && !open ? 'warn' : '',
            meta: [count(rotation.length, 'in rotation'), rotation.length > 1 ? count(spent.length, 'at the limit') : '',
                returns ? `next reset ${humanizeResetAt(returns, nowMs)}` : '', checked ? count(rotation.length - checked, 'not checked') : '',
                count(own.length - rotation.length, 'out of rotation')].filter(Boolean).join(' · ') };
    });
    const meta = accountsRead !== READ_OK ? 'The accounts could not be read now.'
        : doors.length ? 'Sign in, add an account or take one out of rotation.' : 'No subscription account yet.';
    return [...doors, { path: 'accounts', name: 'Accounts and keys', meta, value: '', tone: '', harness: '' }];
}
function subscriptionDoorHtml({ path, name, meta, value, tone, harness }) {
    const mark = harness ? harnessPresentation(harness) : null;
    const glyph = mark ? `<svg viewBox="${mark.viewBox}" width="1em" height="1em" fill="currentColor" focusable="false"><path d="${mark.path}"></path></svg>` : controlIcon('globe', 16);
    return `<button type="button" class="ui-row settings-row" data-kind="door" data-lead data-costs-account="${escapeHtmlAttr(path)}">`
        + `<span class="settings-head-glyph dashboard-settings-glyph" aria-hidden="true">${glyph}</span>`
        + `<span class="ui-row-title">${escapeHtmlAttr(name)}</span><span class="ui-row-meta"><span>${escapeHtmlAttr(meta)}</span></span><span class="ui-row-actions">`
        + `${value ? `<span class="settings-row-value"${tone ? ` data-tone="${tone}"` : ''}>${escapeHtmlAttr(value)}</span>` : ''}`
        + '<span class="settings-row-chevron" aria-hidden="true">›</span></span></button>';
}

const tableCardHtml = ([id, title, column, , note]) => `<section class="ui-card costs-table-card" aria-labelledby="${id}-title">
        <h3 class="dashboard-card-title" id="${id}-title">${title}</h3>${note ? `<p class="costs-table-note">${note}</p>` : ''}
        <table class="cost-table" id="${id}"><thead><tr><th>${column}</th><th class="cost-cell-right">Calls</th><th class="cost-cell-right">Cost</th></tr></thead><tbody></tbody></table>
        <button type="button" class="btn btn-ghost btn-sm ui-list-more" data-costs-more hidden></button>
    </section>`;

export function initCosts({ state, mount, openDashboardTab = null, openSettings = null }) {
    // The tab's one root and scroller (style.css "The Dashboard tab body"): the known spend of the limit leads, the
    // budget it is set against follows as the tab's action row, then where the money went.
    const page = document.createElement('div');
    page.id = 'page-costs';
    page.className = 'dashboard-body scroll-fade-y';  // the page binds every .scroll-fade-y scroller once (app.js)
    page.innerHTML = `
        <section class="ui-card" aria-label="Known spend">
            <div class="dashboard-hero">
                <p class="dashboard-hero-kicker">Known spend</p>
                <p class="dashboard-hero-figure" id="cost-known-limit">Loading…</p>
                <div class="costs-spend-bar" id="cost-spend-bar" hidden><i></i></div>
                <p class="dashboard-hero-meta ui-status" id="cost-left"></p>
                <div class="dashboard-hero-actions"><button type="button" class="btn btn-default btn-sm" id="btn-refresh-costs">Refresh</button></div>
            </div>
            <details class="ui-disclosure costs-details" id="cost-details" hidden>
                <summary>Details</summary>
                <dl class="dashboard-facts" id="cost-facts"></dl>
                <p class="costs-copy">The unresolved upper bound is calls in flight, or ended without a price, at their reserved ceiling: a worst case, not spending.</p>
            </details>
        </section>
        <section class="ui-card costs-budget" aria-labelledby="costs-budget-title">
            <h3 class="dashboard-card-title" id="costs-budget-title">Budget limits</h3>
            <div class="costs-limits dashboard-settings">${limitRows(false)}</div>
            <p class="ui-field-help costs-gaps" id="costs-gaps" hidden></p>
            <div class="costs-allowance">
                <p class="ui-field-help costs-allowance-spend" id="costs-allowance-spend">Reading…</p>
            </div>
            <div class="costs-allowance-state" id="costs-allowance-state" aria-live="polite"></div>
            <details class="ui-disclosure">
                <summary>How the total and per-task limits count</summary>
                <p class="costs-copy" id="costs-budget-help" role="note">${BUDGET_NOTE_FALLBACK}</p>
            </details>
            <details class="ui-disclosure">
                <summary>How the consciousness limit counts</summary>
                <p class="costs-copy" id="costs-allowance-help" role="note">${ALLOWANCE_NOTE}</p>
            </details>
            <details class="ui-disclosure">
                <summary>More limits</summary>
                <div class="costs-limits dashboard-settings">${limitRows(true)}</div>
            </details>
        </section>
        <section class="ui-card dashboard-settings costs-subscriptions" aria-labelledby="costs-subscriptions-title">
            <h3 class="dashboard-card-title" id="costs-subscriptions-title">Subscriptions</h3>
            <p class="dashboard-settings-lead">Quotas of your accounts. They are not dollars, and no limit above counts them.</p>
            <div data-costs-subscriptions><p class="dashboard-settings-lead">Reading…</p></div>
        </section>
        <div id="costs-period"></div>
        <div class="costs-tables">${TABLES.map(tableCardHtml).join('')}</div>
    `;
    mount.appendChild(page);
    const periodCard = mountCostPeriod(page.querySelector('#costs-period'));  // its own read: the period governs that card alone
    const budgetCard = page.querySelector('.costs-budget');
    // The rows are drawn once and never redrawn: a refresh of the tab repaints the spend, the tables and the allowance's
    // facts around them, and `limits.paint()` only writes a fresh value into a field the owner is not in.
    const limits = bindSettingRows(budgetCard);
    const expanded = new Set();
    let lastBreakdown = null;
    let allowanceRead = 0;
    let allowanceExpiry = null;
    let subscriptionsRead = 0;

    function renderBreakdownTable(tableId, data, emptyLabel = 'No data') {
        const table = page.querySelector('#' + tableId);
        const tbody = table.querySelector('tbody');
        const more = table.parentElement.querySelector('[data-costs-more]');
        tbody.innerHTML = '';
        const cell = (className, text) => {
            const td = document.createElement('td');
            td.className = className;
            td.textContent = text;
            return td;
        };
        const rows = Object.entries(data || {});
        for (const [name, info] of expanded.has(tableId) ? rows : rows.slice(0, TABLE_ROWS)) {
            const tr = document.createElement('tr');
            const shown = breakdownRowName(tableId, name);
            const nameCell = cell('cost-cell-name', shown);
            nameCell.setAttribute('title', shown);
            const cost = costBucketPresentation(info);
            const costCell = cell('cost-cell-right', '');
            costCell.innerHTML = `${escapeHtmlAttr(cost.text)}${cost.open ? statusMarkHtml('cost-open', { shape: 'ring', hue: 'ink' }) : ''}`
                + (cost.detail ? `<span class="cost-cell-note">${escapeHtmlAttr(cost.detail)}</span>` : '');
            tr.append(nameCell, cell('cost-cell-right', number(info?.calls)), costCell);
            tbody.appendChild(tr);
        }
        if (rows.length === 0) {
            const tr = document.createElement('tr');
            const td = cell('cost-empty-cell', emptyLabel);
            td.setAttribute('colspan', '3');
            tr.appendChild(td);
            tbody.appendChild(tr);
        }
        more.hidden = rows.length <= TABLE_ROWS;
        more.textContent = expanded.has(tableId) ? 'Show fewer' : `Show all (${number(rows.length)})`;
        more.dataset.costsMore = tableId;
    }

    function renderTables(data, emptyLabel) {
        TABLES.forEach(([id, , , key]) => renderBreakdownTable(id, data?.[key] || {}, emptyLabel));
    }

    function renderLead(view) {
        const bar = page.querySelector('#cost-spend-bar');
        const details = page.querySelector('#cost-details');
        const facts = page.querySelector('#cost-facts');
        const left = page.querySelector('#cost-left');
        const available = view.state === 'available';
        page.querySelector('#cost-known-limit').textContent = available ? view.figure
            : view.state === 'loading' ? 'Loading…' : 'Unavailable';
        left.textContent = available ? [view.left, view.open].filter(Boolean).join(' · ') : '';
        if (available && view.reached) left.dataset.tone = 'warn'; else delete left.dataset.tone;
        bar.hidden = !available || view.share === null;
        bar.dataset.reached = available && view.reached ? '1' : '0';
        bar.setAttribute('style', `--spend-share: ${available && view.share !== null ? view.share.toFixed(1) : 0}%`);
        details.hidden = !available;
        facts.innerHTML = available ? view.facts.map(([label, value]) => `<dt>${escapeHtmlAttr(label)}</dt><dd>${escapeHtmlAttr(value)}</dd>`).join('') : '';
    }

    // The allowance's facts under its field: the consciousness snapshot read when the tab is shown, refreshed and
    // saved, from `/api/consciousness` (its window is current), else `/api/state`; no polling of its own. A room time
    // it gave is read once more when it passes, so an old time never stands as current.
    function renderAllowance(snapshot) {
        clearTimeout(allowanceExpiry);
        const wait = Date.parse(String(snapshot?.allowanceRoomAfter || '')) - Date.now();
        if (wait > 0 && wait < 2 ** 31 - 1000) {
            allowanceExpiry = setTimeout(() => { if (page.isConnected !== false) loadAllowance(); }, wait + 1000);
            allowanceExpiry?.unref?.();  // a test runner's clock is never kept alive by it
        }
        const view = allowancePresentation(snapshot);
        const door = '<button type="button" class="consciousness-inline-link ui-hit-area" data-costs-consciousness>Consciousness ›</button>';
        // Each separator keeps to the word before it (a no-break space), so a wrapped line never starts with one.
        page.querySelector('#costs-allowance-spend').innerHTML = `${escapeHtmlAttr(view.spend)}`
            + `${view.detail ? `\u00a0· <span class="cost-cell-note">${escapeHtmlAttr(view.detail)}</span>` : ''}\u00a0· ${door}`;
        page.querySelector('#costs-allowance-state').innerHTML = view.lines.map((line) => `<p class="ui-status costs-allowance-line" `
            + `data-tone="${line.tone}">${escapeHtmlAttr(line.text)}`
            + `${line.note ? ` <span class="costs-allowance-note">${escapeHtmlAttr(line.note)}</span>` : ''}`
            + `${line.wake ? ` ${taskReferenceHtml(line.wake)}` : ''}</p>`).join('');
    }

    async function loadAllowance() {
        const read = ++allowanceRead;
        let snapshot = null;
        try { snapshot = await readConsciousness(); } catch {
            try { snapshot = await readConsciousnessFromState(); } catch { snapshot = null; }
        }
        if (read === allowanceRead) renderAllowance(snapshot);
    }

    async function loadCosts() {
        renderLead({ state: 'loading' });
        try {
            const resp = await apiFetch('/api/cost-breakdown');
            const d = await resp.json();
            const presentation = costDashboardPresentation(d);
            if (!resp.ok || presentation.state !== 'available') throw new Error('accounting unavailable');
            lastBreakdown = d;
            renderLead(presentation);
            renderTables(d);
        } catch {
            lastBreakdown = null;
            renderLead({ state: 'unavailable' });
            renderTables(null, 'Unavailable');
        }
    }

    // Read when the tab is shown and on Refresh, never on a save elsewhere: no answer of it changes with a setting.
    async function loadSubscriptions() {
        const read = ++subscriptionsRead;
        let payload = null;
        let transportError = '';
        try {
            const resp = await apiFetch('/api/claudexor/status?view=quota', { cache: 'no-store' });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            payload = await resp.json();
        } catch (error) { transportError = String(error?.message || error || 'unreachable'); }
        if (read !== subscriptionsRead) return;
        page.querySelector('[data-costs-subscriptions]').innerHTML = subscriptionDoors(payload, { transportError }).map(subscriptionDoorHtml).join('');
    }

    // What the limits cannot count, one line under them (pricing.known_spend_gaps). The words are composed here,
    // so the overlay leaves them alone and a language change says them again.
    let spendGaps = [];
    function paintGaps() {
        const line = page.querySelector('#costs-gaps');
        if (!line) return;
        line.textContent = spendGaps.length ? fmt('Not counted: {roles} (price unknown).',
            { roles: spendGaps.map((name) => translateString(name)).join(', ') }) : '';
        line.hidden = !spendGaps.length;
        if (spendGaps.length) line.dataset.i18nFmt = 'spend gaps'; else delete line.dataset.i18nFmt;
    }

    async function loadLimits() {
        try {
            const settings = await readSettings();
            const fields = settings?._meta?.setup_contract?.budgetFields || [];
            const note = fields.map((field) => String(field.note || '').trim()).find(Boolean) || '';
            const noteEl = page.querySelector('#costs-budget-help');
            if (noteEl && note) noteEl.textContent = note;
            spendGaps = Array.isArray(settings?._meta?.known_spend_gaps) ? settings._meta.known_spend_gaps.map(String) : [];
            paintGaps();
        } catch { /* the rows keep the last values they read */ }
        limits.paint();
    }

    page.querySelector('#btn-refresh-costs').addEventListener('click', () => { loadCosts(); loadAllowance(); loadSubscriptions(); periodCard.reload(); });
    // Each door opens its Settings page, which leads back here.
    page.querySelector('.costs-subscriptions').addEventListener('click', (event) => {
        const path = event.target.closest?.('[data-costs-account]')?.dataset.costsAccount;
        if (path !== undefined) void openSettings?.(path, { back: { label: 'Costs', go: () => openDashboardTab?.('costs') } });
    });
    // The allowance's doors: a held wake's card in Main, and the Consciousness tab.
    budgetCard.addEventListener('click', (event) => {
        if (openTaskReference(event)) return;
        if (event.target.closest?.('[data-costs-consciousness]')) void openDashboardTab?.('consciousness');
    });
    page.querySelector('.costs-tables').addEventListener('click', (event) => {
        const more = event.target.closest?.('[data-costs-more]');
        if (!more) return;
        const id = more.dataset.costsMore;
        if (expanded.has(id)) expanded.delete(id); else expanded.add(id);
        const key = TABLES.find(([tableId]) => tableId === id)?.[3];
        renderBreakdownTable(id, lastBreakdown?.[key] || {});
    });

    function refreshCostsPanel() {
        loadCosts();
        loadLimits();
        loadAllowance();
    }

    // A result the owner saw ends when the tab or the Dashboard is left.
    window.addEventListener('ouro:dashboard-subtab-shown', (event) => {
        if (event.detail?.tab !== 'costs') limits.leave();
        else if (state.activePage === 'dashboard') { refreshCostsPanel(); loadSubscriptions(); periodCard.reload(); }  // the period is read when the tab is looked at
    });
    window.addEventListener('ouro:page-shown', (event) => { if (event.detail?.page !== 'dashboard') limits.leave(); });
    window.addEventListener('ouro:language-changed', paintGaps);
    window.addEventListener('ouro:settings-updated', (event) => {
        const { source, page: home, row } = event.detail || {};
        if (source !== 'settings-row' || home !== LIMITS_PAGE) { refreshCostsPanel(); return; }
        // One of this card's own rows was saved: read again only what that limit moves.
        if (row === 'TOTAL_BUDGET') loadCosts();
        if (row === ALLOWANCE_KEY) loadAllowance();
    });
}
