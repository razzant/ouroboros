import { formatUsd, number } from './storage_format.js';
import { APP_SETTINGS_ORIGIN, apiFetch, jsonPost } from './api_client.js';
import { clearResultOnEdit, errorWords, escapeHtmlAttr, setResult } from './ui_primitives.js';
import { statusMarkHtml } from './status_mark.js';

const COST_BUDGET_INPUTS = {
    TOTAL_BUDGET: 's-budget',
    OUROBOROS_PER_TASK_COST_USD: 's-per-task-cost',
};
// A breakdown table shows its largest rows; the rest wait behind its "Show all (N)" (the list's one length control).
const TABLE_ROWS = 8;
const TABLES = [
    ['cost-by-model', 'By model', 'Model', 'by_model'],
    ['cost-by-key', 'By API key', 'Key', 'by_api_key'],
    ['cost-by-model-cat', 'By model category', 'Category', 'by_model_category'],
    ['cost-by-task-cat', 'By task category', 'Category', 'by_task_category'],
];

function readPositiveBudget(id) {
    const input = document.getElementById(id);
    const raw = String(input?.value || '').trim();
    const value = Number(raw);
    const min = Number(input?.min || 0.01);
    return Number.isFinite(value) && value >= min ? value : null;
}

function optionalFiniteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

/** One breakdown cell: the amount alone, an open bucket marked as not final with its cause in words; an open zero is
 *  never free and an absent amount never $0.00. `{ text, open, note }`. */
export function costBucketPresentation(info) {
    const cost = optionalFiniteNumber(info?.cost);
    if (cost === null) return { text: 'unknown', open: false, note: '' };
    const unknown = Math.trunc(optionalFiniteNumber(info?.unknown_unmetered) || 0);
    const nonFinalRows = optionalFiniteNumber(info?.non_final_rows) || 0;
    const open = info?.cost_final === false || nonFinalRows > 0 || unknown > 0;
    if (!open) return { text: formatUsd(cost), open: false, note: '' };
    const unpriced = unknown > 0 ? `; ${number(unknown)} ${unknown === 1 ? 'call has' : 'calls have'} no price yet` : '';
    return { text: cost === 0 ? 'pending' : formatUsd(cost), open: true, note: `Not final${unpriced}.` };
}

/** What one budget save says and whether that saying may end by itself: `{ text, tone, transient }`. */
export function budgetSavePresentation(data = {}) {
    let text;
    // `restart_required` is the server's whole restart state, so with nothing changed it is a setting saved earlier.
    if (data.no_changes) text = data.restart_required ? 'No changes. Restart still required.' : 'No changes.';
    else if (data.restart_required) text = 'Saved. Restart required.';
    else if (data.immediate_changed && data.next_task_changed) {
        text = 'Saved. Some changes took effect immediately; others apply on the next task.';
    } else if (data.immediate_changed) text = 'Saved. Took effect immediately.';
    else text = 'Saved. Applies on the next task.';
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
    // it explains rather than in a new tile nobody correlates. Absent on an older payload,
    // and then this says only what it knows instead of inventing a zero.
    const nonFinal = optionalFiniteNumber(accounting.non_final_rows);
    const openCause = nonFinal !== null && nonFinal > 0 ? ` (${number(Math.trunc(nonFinal))} open)` : '';
    return {
        state: 'available',
        figure: limit > 0 ? `${formatUsd(known)} of ${formatUsd(limit)}` : formatUsd(known),
        left: reached ? 'Limit reached: new paid work is refused' : limit > 0 ? `${formatUsd(remaining)} left` : 'No limit set',
        open: unresolved > 0 ? `up to ${formatUsd(unresolved)} more is still being priced` : '',
        reached,
        share: limit > 0 ? Math.min(100, Math.max(0, (known / limit) * 100)) : null,
        facts: [
            ['Final', accounting.cost_final === true ? 'Yes' : `Pending${openCause}`],
            ['Confirmed', formatUsd(confirmed)],
            ['Estimated', formatUsd(estimated)],
            ['Reserved', formatUsd(reserved)],
            ['Unresolved upper bound', formatUsd(unresolved)],
            ['Calls without a price', number(Math.trunc(unknown))],
            ['Physical attempts', number(Math.trunc(calls))],
        ],
    };
}

const tableCardHtml = ([id, title, column]) => `<section class="ui-card costs-table-card" aria-labelledby="${id}-title">
        <h3 class="costs-card-title" id="${id}-title">${title}</h3>
        <table class="cost-table" id="${id}"><thead><tr><th>${column}</th><th class="cost-cell-right">Calls</th><th class="cost-cell-right">Cost</th></tr></thead><tbody></tbody></table>
        <button type="button" class="btn btn-ghost btn-sm ui-list-more" data-costs-more hidden></button>
    </section>`;

export function initCosts({ state, mount }) {
    // The tab's one root and scroller (style.css "The Dashboard tab body"): the known spend of the limit leads, the
    // budget it is set against follows as the tab's action row, then where the money went.
    const page = document.createElement('div');
    page.id = 'page-costs';
    page.className = 'dashboard-body';
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
                <dl class="costs-facts" id="cost-facts"></dl>
            </details>
        </section>
        <section class="ui-card costs-budget" aria-labelledby="costs-budget-title">
            <h3 class="costs-card-title" id="costs-budget-title">Budget</h3>
            <div class="costs-budget-fields">
                <div class="form-field ui-field">
                    <label for="s-budget">Total budget ($)</label>
                    <input id="s-budget" class="ui-control" type="number" value="200" aria-describedby="costs-budget-help">
                    <div class="ui-field-help" id="costs-budget-help">Paid work stops when known spend reaches it.</div>
                </div>
                <div class="form-field ui-field">
                    <label for="s-per-task-cost">Per-task cap ($)</label>
                    <input id="s-per-task-cost" class="ui-control" type="number" value="50" aria-describedby="costs-per-task-note">
                    <div class="ui-field-help" id="costs-per-task-note">A task and its helpers stop when their known spend reaches it.</div>
                </div>
            </div>
            <details class="ui-disclosure">
                <summary>How the limits count</summary>
                <p class="costs-copy">Known spend is the confirmed and estimated cost. The cap covers a task's whole tree. Calls already in flight settle normally, and raising the cap does not resume paused work.</p>
            </details>
            <div class="ui-action-row">
                <button type="button" class="btn btn-save" id="btn-save-budget">Save</button>
                <p class="ui-result" id="budget-save-status" role="status" aria-live="polite"></p>
            </div>
        </section>
        <div class="costs-tables">${TABLES.map(tableCardHtml).join('')}</div>
    `;
    mount.appendChild(page);
    const budgetCard = page.querySelector('.costs-budget');
    const statusEl = page.querySelector('#budget-save-status');
    // An edit of either field starts a new subject: the last save's words go (DESIGN "Status and chips").
    clearResultOnEdit(budgetCard, statusEl);
    const expanded = new Set();
    let lastBreakdown = null;

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
            const nameCell = cell('cost-cell-name', name);
            nameCell.setAttribute('title', name);
            const cost = costBucketPresentation(info);
            const costCell = cell('cost-cell-right', cost.text);
            if (cost.open) {
                costCell.innerHTML = `${escapeHtmlAttr(cost.text)}${statusMarkHtml('cost-open', { shape: 'ring', hue: 'ink' })}`;
                costCell.setAttribute('title', cost.note);
                costCell.setAttribute('aria-label', `${cost.text}. ${cost.note}`);
            }
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
        try {
            const resp = await apiFetch('/api/settings', { cache: 'no-store' });
            const s = await resp.json().catch(() => ({}));
            const fields = s?._meta?.setup_contract?.budgetFields || [];
            fields.forEach((field) => {
                const input = document.getElementById(COST_BUDGET_INPUTS[field.settingKey]);
                if (!input) return;
                input.min = field.min || '0.01';
                input.step = field.step || 'any';
                if (field.default != null && !String(input.value || '').trim()) {
                    input.value = field.default;
                }
            });
            if (s.TOTAL_BUDGET != null) document.getElementById('s-budget').value = s.TOTAL_BUDGET;
            if (s.OUROBOROS_PER_TASK_COST_USD != null) document.getElementById('s-per-task-cost').value = s.OUROBOROS_PER_TASK_COST_USD;
        } catch {}
    }

    page.querySelector('#btn-refresh-costs').addEventListener('click', loadCosts);
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
        const budget = readPositiveBudget('s-budget');
        const perTask = readPositiveBudget('s-per-task-cost');
        if (budget === null || perTask === null) {
            setResult(statusEl, { text: 'Budget values must be at least 0.01.', tone: 'danger' });
            return;
        }
        const saveButton = event.currentTarget;
        saveButton.disabled = true;
        saveButton.setAttribute('aria-busy', 'true');
        setResult(statusEl, { text: 'Saving…', tone: 'muted' });
        try {
            const data = await jsonPost('/api/settings', { TOTAL_BUDGET: budget, OUROBOROS_PER_TASK_COST_USD: perTask },
                { headers: APP_SETTINGS_ORIGIN });
            const saved = budgetSavePresentation(data || {});
            setResult(statusEl, { text: saved.text, tone: saved.tone, ends: saved.transient });
            window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { reason: 'budget saved', source: 'costs' } }));
        } catch (error) {
            const words = errorWords(error);
            setResult(statusEl, { text: `Not saved. ${words.text}`, tone: words.tone, details: words.details });
        } finally {
            saveButton.disabled = false;
            saveButton.removeAttribute('aria-busy');
        }
    });

    function refreshCostsPanel() {
        loadCosts();
        loadBudget();
    }

    window.addEventListener('ouro:dashboard-subtab-shown', (event) => {
        if (event.detail?.tab === 'costs' && state.activePage === 'dashboard') refreshCostsPanel();
    });
    window.addEventListener('ouro:settings-updated', (event) => {
        if (event.detail?.source === 'costs') return;
        refreshCostsPanel();
    });
}
