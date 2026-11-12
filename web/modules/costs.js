import { formatUsd, number } from './storage_format.js';
import { APP_SETTINGS_ORIGIN, apiFetch, jsonPost } from './api_client.js';
import { clearResultOnEdit, errorWords, escapeHtmlAttr, setResult } from './ui_primitives.js';
import { statusMarkHtml } from './status_mark.js';
import { API_PROVIDER_LABELS, harnessPresentation } from './harness_presentation.js';
import { readConsciousness, readConsciousnessFromState } from './consciousness_api.js';
import { allowanceHold, allowanceText } from './consciousness_words.js';
import { openTaskReference, taskReferenceHtml } from './task_reference.js';
import { mountCostPeriod } from './cost_period.js';

const COST_BUDGET_INPUTS = {
    TOTAL_BUDGET: 's-budget',
    OUROBOROS_PER_TASK_COST_USD: 's-per-task-cost',
};
// The three limits of Budget. The consciousness allowance has its one editor here (owner, 10 Nov 2026: moved from
// Settings with the spend it counts); its 0 is a real choice (it may not spend), so it is no setup-contract budget
// field, whose minimum is 0.01. Only an edited limit is posted: a Costs tab left open never writes back an old value.
const ALLOWANCE_KEY = 'OUROBOROS_CONSCIOUSNESS_DAILY_USD';
export const BUDGET_LIMITS = [
    { id: 's-budget', key: 'TOTAL_BUDGET', min: 0.01, invalid: 'Budget values must be at least 0.01.' },
    { id: 's-per-task-cost', key: 'OUROBOROS_PER_TASK_COST_USD', min: 0.01, invalid: 'Budget values must be at least 0.01.' },
    { id: 's-consciousness-allowance', key: ALLOWANCE_KEY, min: 0, invalid: 'The consciousness limit must be a number, 0 or more.' },
];
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

/** One limit's draft against the value it was loaded with (`data-loaded`): `{ value, edited, valid }`. An empty field
 *  is no number (never 0); an untouched one, or one typed back to its loaded number, is not edited. */
export function limitDraft(raw, loaded, min) {
    const text = String(raw ?? '').trim();
    const base = String(loaded ?? '').trim();
    const value = text === '' ? NaN : Number(text);
    const edited = text === '' ? base !== '' : !(base !== '' && Number(base) === value);
    return { value, edited, valid: Number.isFinite(value) && value >= min };
}

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

// The consciousness clock reads the allowance at its next check, never at a task's start; a wake already paused at the
// allowance it started with keeps that number (consciousness_view.held_sentence).
const ALLOWANCE_SAVED = 'The next consciousness check uses it; a paused wake keeps its launch limit.';
const ALLOWANCE_SAVED_BESIDE = 'The next consciousness check uses the new consciousness limit; a paused wake keeps its launch limit.';

/** What one budget save says and whether that saying may end by itself: `{ text, tone, transient }`. `sent` names the
 *  edited limits the save posted: the server files the allowance under the next task, which it is not, so its own
 *  sentence stands for it and the other limits keep their words. */
export function budgetSavePresentation(data = {}, { sent = null } = {}) {
    let text;
    const keys = sent ? new Set(sent) : null;
    const allowance = Boolean(keys?.has(ALLOWANCE_KEY));
    const others = !keys || [...keys].some((key) => key !== ALLOWANCE_KEY);
    // The per-task cap is the one other limit applying on the next task; Total budget takes effect at once.
    const nextTask = data.next_task_changed && (!allowance || keys.has('OUROBOROS_PER_TASK_COST_USD'));
    // `restart_required` is the server's whole restart state, so with nothing changed it is a setting saved earlier.
    if (data.no_changes) text = data.restart_required ? 'No changes. Restart still required.' : 'No changes.';
    else if (allowance && !others && !data.restart_required) text = `Saved. ${ALLOWANCE_SAVED}`;
    else {
        if (data.restart_required) text = 'Saved. Restart required.';
        else if (data.immediate_changed && nextTask) {
            text = 'Saved. Some changes took effect immediately; others apply on the next task.';
        } else if (data.immediate_changed) text = 'Saved. Took effect immediately.';
        else text = 'Saved. Applies on the next task.';
        if (allowance) text += ` ${others ? ALLOWANCE_SAVED_BESIDE : ALLOWANCE_SAVED}`;
    }
    const warned = Boolean(data.warnings?.length);
    if (warned) text += ' Warnings: ' + data.warnings.join(' | ');
    const standing = warned || Boolean(data.restart_required);
    return { text, tone: standing ? 'warn' : 'ok', transient: !standing };
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
 *  `detail` what it leaves out (pending calls, calls without a price); `lines` at most two: the window (its limit
 *  reached, with a conditional time when room returns) and a held wake (its proven cause, with its card's door), each
 *  its fact (`text`, in the line's tone) and what follows from it (`note`, quieter). Off, a $0 limit and an
 *  unreadable ledger say so instead. No time is guessed: a withheld or stale forecast says so. */
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

const tableCardHtml = ([id, title, column, , note]) => `<section class="ui-card costs-table-card" aria-labelledby="${id}-title">
        <h3 class="dashboard-card-title" id="${id}-title">${title}</h3>${note ? `<p class="costs-table-note">${note}</p>` : ''}
        <table class="cost-table" id="${id}"><thead><tr><th>${column}</th><th class="cost-cell-right">Calls</th><th class="cost-cell-right">Cost</th></tr></thead><tbody></tbody></table>
        <button type="button" class="btn btn-ghost btn-sm ui-list-more" data-costs-more hidden></button>
    </section>`;

export function initCosts({ state, mount, openDashboardTab = null }) {
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
            <h3 class="dashboard-card-title" id="costs-budget-title">Budget</h3>
            <div class="costs-budget-fields">
                <div class="form-field ui-field">
                    <label for="s-budget">Total budget ($)</label>
                    <input id="s-budget" class="ui-control" type="number" value="200" aria-describedby="costs-total-note costs-budget-help">
                    <div class="ui-field-help" id="costs-total-note">Paid work stops when known spend reaches it.</div>
                </div>
                <div class="form-field ui-field">
                    <label for="s-per-task-cost">Per-task cost cap ($)</label>
                    <input id="s-per-task-cost" class="ui-control" type="number" value="50" aria-describedby="costs-per-task-note costs-budget-help">
                    <div class="ui-field-help" id="costs-per-task-note">A task and its helpers stop when their known spend reaches it.</div>
                </div>
                <div class="form-field ui-field costs-allowance">
                    <label for="s-consciousness-allowance">Consciousness per 24 hours ($)</label>
                    <input id="s-consciousness-allowance" class="ui-control" type="number" min="0" step="any" inputmode="decimal"
                        aria-describedby="costs-allowance-note costs-allowance-spend costs-allowance-state costs-allowance-help">
                    <div class="ui-field-help" id="costs-allowance-note">Its wake-ups, replies in places and the work they start, over a rolling 24 hours.</div>
                    <p class="ui-field-help costs-allowance-spend" id="costs-allowance-spend">Reading…</p>
                </div>
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
            <div class="ui-action-row">
                <button type="button" class="btn btn-save" id="btn-save-budget">Save</button>
                <p class="ui-result" id="budget-save-status" role="status" aria-live="polite"></p>
            </div>
        </section>
        <div id="costs-period"></div>
        <div class="costs-tables">${TABLES.map(tableCardHtml).join('')}</div>
    `;
    mount.appendChild(page);
    const periodCard = mountCostPeriod(page.querySelector('#costs-period'));  // its own read: the period governs that card alone
    const budgetCard = page.querySelector('.costs-budget');
    const statusEl = page.querySelector('#budget-save-status');
    // An edit of either field starts a new subject: the last save's words go (DESIGN "Status and chips").
    clearResultOnEdit(budgetCard, statusEl);
    const expanded = new Set();
    let lastBreakdown = null;
    let allowanceRead = 0;
    let allowanceExpiry = null;
    let budgetRead = 0;
    budgetCard.addEventListener('input', (event) => {
        const limit = BUDGET_LIMITS.find(({ id }) => id === event.target.id);
        // While a save of the field is on its way, an edit is measured against what that save sends.
        if (limit) event.target.dataset.edited = String(limitDraft(event.target.value,
            event.target.dataset.saving ?? event.target.dataset.loaded, limit.min).edited);
    });

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

    async function loadBudget() {
        const read = ++budgetRead;
        try {
            const resp = await apiFetch('/api/settings', { cache: 'no-store' });
            const s = await resp.json().catch(() => ({}));
            if (!resp.ok || read !== budgetRead) return;
            const fields = s?._meta?.setup_contract?.budgetFields || [];
            const note = fields.map((field) => String(field.note || '').trim()).find(Boolean) || '';
            const noteEl = document.getElementById('costs-budget-help');
            if (noteEl && note) {
                noteEl.textContent = note;
                noteEl.hidden = false;
            }
            fields.forEach((field) => {
                const input = document.getElementById(COST_BUDGET_INPUTS[field.settingKey]);
                if (!input) return;
                input.min = field.min || '0.01';
                input.step = field.step || 'any';
                if (field.default != null && input.dataset.edited !== 'true' && !String(input.value || '').trim()) {
                    input.value = field.default;
                }
            });
            BUDGET_LIMITS.forEach(({ id, key }) => {
                const input = document.getElementById(id);
                // A tab return or a Settings notification may finish reading while the owner is typing.
                // Keep that draft; update only fields still showing their last loaded/saved value.
                if (input.dataset.edited !== 'true' && input.dataset.saving === undefined && s[key] != null) {
                    input.value = String(s[key]);
                    input.dataset.loaded = String(input.value);
                }
            });
        } catch {}
    }

    page.querySelector('#btn-refresh-costs').addEventListener('click', () => { loadCosts(); loadAllowance(); periodCard.reload(); });
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

    // The result of a save sits right after Save and keeps its own lifetime (ui_primitives.js setResult): a plain
    // acknowledgement ends by itself, a restart still required, a warning or a failure stays until the next save or
    // an edit of the fields replaces it.
    page.querySelector('#btn-save-budget').addEventListener('click', async (event) => {
        const drafts = BUDGET_LIMITS.map((limit) => {
            const input = document.getElementById(limit.id);
            const draft = limitDraft(input.value, input.dataset.loaded, limit.min);
            return { ...limit, ...draft, edited: draft.edited && input.dataset.edited === 'true' };
        });
        const invalid = drafts.find((draft) => draft.edited && !draft.valid);
        if (invalid) {
            setResult(statusEl, { text: invalid.invalid, tone: 'danger' });
            return;
        }
        // Only what was edited travels; nothing edited still asks the server, which says so (and any restart owed).
        const sending = drafts.filter((draft) => draft.edited);
        const body = Object.fromEntries(sending.map((draft) => [draft.key, draft.value]));
        sending.forEach((draft) => { document.getElementById(draft.id).dataset.saving = String(draft.value); });
        const saveButton = event.currentTarget;
        saveButton.disabled = true;
        saveButton.setAttribute('aria-busy', 'true');
        setResult(statusEl, { text: 'Saving…', tone: 'muted' });
        try {
            const data = await jsonPost('/api/settings', body, { headers: APP_SETTINGS_ORIGIN });
            ++budgetRead;  // a read started before this save must not put the old value back
            const saved = budgetSavePresentation(data || {}, { sent: Object.keys(body) });
            setResult(statusEl, { text: saved.text, tone: saved.tone, ends: saved.transient });
            sending.forEach((draft) => { document.getElementById(draft.id).dataset.loaded = String(draft.value); });
            if (ALLOWANCE_KEY in body) loadAllowance();
            window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { reason: 'budget saved', source: 'costs' } }));
        } catch (error) {
            const words = errorWords(error);
            setResult(statusEl, { text: `Not saved. ${words.text}`, tone: words.tone, details: words.details });
        } finally {
            // What was typed while it travelled stays a draft against what is now saved (or, failed, still loaded).
            sending.forEach((draft) => {
                const input = document.getElementById(draft.id);
                delete input.dataset.saving;
                input.dataset.edited = String(limitDraft(input.value, input.dataset.loaded, draft.min).edited);
            });
            saveButton.disabled = false;
            saveButton.removeAttribute('aria-busy');
        }
    });

    function refreshCostsPanel() {
        loadCosts();
        loadBudget();
        loadAllowance();
    }

    window.addEventListener('ouro:dashboard-subtab-shown', (event) => {
        if (event.detail?.tab !== 'costs' || state.activePage !== 'dashboard') return;
        refreshCostsPanel();
        periodCard.reload();  // read when the tab is looked at, never at app start or on a settings save
    });
    window.addEventListener('ouro:settings-updated', (event) => {
        if (event.detail?.source === 'costs') return;
        refreshCostsPanel();
    });
}
