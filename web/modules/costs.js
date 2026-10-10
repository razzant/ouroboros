import { formatUsd2 } from './utils.js';
import { apiFetch } from './api_client.js';

const COST_BUDGET_INPUTS = {
    TOTAL_BUDGET: 's-budget',
    OUROBOROS_PER_TASK_COST_USD: 's-per-task-cost',
};

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

/** Render one legacy breakdown bucket without turning an open zero into free. */
export function costBucketPresentation(info) {
    const cost = optionalFiniteNumber(info?.cost);
    if (cost === null) return 'unknown';
    const unknown = optionalFiniteNumber(info?.unknown_unmetered) || 0;
    const nonFinalRows = optionalFiniteNumber(info?.non_final_rows) || 0;
    const pending = info?.cost_final === false || nonFinalRows > 0 || unknown > 0;
    if (!pending) return formatUsd2(cost);
    const unknownLabel = unknown > 0 ? `unmetered=${Math.trunc(unknown)}` : '';
    const details = [unknownLabel].filter(Boolean).join(', ');
    if (cost === 0) return details ? `cost pending (${details})` : 'cost pending';
    return details ? `${formatUsd2(cost)} (pending, ${details})` : `${formatUsd2(cost)} (pending)`;
}

/** Pure cost-dashboard projection: null/unavailable never renders as $0.
 * The limit is set against KNOWN spend (settled: confirmed + estimated), the number
 * that refuses new paid calls (#1487); reservations and unresolved upper bounds are
 * shown beside it as open holds, never added in. */
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
    const models = Object.entries(data.by_model || {});
    // A flag without its cause is not reconstructible. `cost_final: false` holds with every
    // dollar bucket at $0.00 and `unknown` at 0 — an ESTIMATED zero, or a dispatched row
    // whose reservation is exactly zero — so the count of open rows rides beside the flag
    // it explains rather than in a new tile nobody correlates. Absent on an older payload,
    // and then this says only what it knows instead of inventing a zero.
    const nonFinal = optionalFiniteNumber(accounting.non_final_rows);
    const openCause = nonFinal !== null && nonFinal > 0 ? ` (${Math.trunc(nonFinal)} open)` : '';
    return {
        state: 'available',
        knownLimit: `${formatUsd2(known)} / ${limit > 0 ? formatUsd2(limit) : '∞'}`,
        confirmed: formatUsd2(confirmed),
        estimated: formatUsd2(estimated),
        reserved: formatUsd2(reserved),
        unresolved: formatUsd2(unresolved),
        unknown: String(Math.trunc(unknown)),
        final: accounting.cost_final === true ? 'Yes' : `Pending${openCause}`,
        calls: String(Math.trunc(calls)),
        topModel: models.length > 0 ? models[0][0] : '-',
    };
}

export function initCosts({ state, mount }) {
    const page = document.createElement('div');
    page.id = 'page-costs';
    page.className = 'settings-embedded-content settings-costs-panel';
    page.innerHTML = `
        <div class="costs-scroll">
            <div class="costs-budget-card">
                <div class="costs-budget-head">
                    <h3 class="costs-budget-title">Budget</h3>
                    <button class="btn btn-default btn-sm costs-budget-refresh" id="btn-refresh-costs">Refresh</button>
                </div>
                <div class="costs-budget-fields">
                    <div class="form-field ui-field">
                        <label for="s-budget">Total Budget ($)</label>
                        <input id="s-budget" class="ui-control" type="number" value="200">
                    </div>
                    <div class="form-field ui-field">
                        <label for="s-per-task-cost">Per-task Cost Cap ($)</label>
                        <input id="s-per-task-cost" class="ui-control" type="number" value="50" aria-describedby="costs-per-task-note">
                        <div class="settings-inline-note ui-field-help" id="costs-per-task-note">Hard dispatch cap for the whole root task tree, reached when its known spend (confirmed and estimated) gets there. In-flight calls settle normally; increasing the cap does not auto-resume paused work.</div>
                    </div>
                </div>
                <button class="btn btn-save costs-budget-save" id="btn-save-budget">Save Budget</button>
                <div id="budget-save-status" class="settings-inline-status"></div>
            </div>
            <div class="costs-stats-grid">
                <div class="stat-card"><div class="label">Known spend / Limit</div><div class="value" id="cost-known-limit">Loading…</div></div>
                <div class="stat-card"><div class="label">Confirmed</div><div class="value" id="cost-confirmed">—</div></div>
                <div class="stat-card"><div class="label">Estimated</div><div class="value" id="cost-estimated">—</div></div>
                <div class="stat-card"><div class="label">Reserved</div><div class="value" id="cost-reserved">—</div></div>
                <div class="stat-card"><div class="label">Unresolved upper bound</div><div class="value" id="cost-unresolved">—</div></div>
                <div class="stat-card"><div class="label">Unknown / unmetered</div><div class="value" id="cost-unknown">—</div></div>
                <div class="stat-card"><div class="label">Cost final</div><div class="value" id="cost-final">Loading…</div></div>
                <div class="stat-card"><div class="label">Physical attempts</div><div class="value" id="cost-calls">—</div></div>
                <div class="stat-card"><div class="label">Top Model</div><div class="value cost-top-model" id="cost-top-model">-</div></div>
            </div>
            <div class="costs-tables-grid">
                <div>
                    <h3 class="costs-table-label">By Model</h3>
                    <table class="cost-table" id="cost-by-model"><thead><tr><th>Model</th><th>Calls</th><th>Cost</th><th></th></tr></thead><tbody></tbody></table>
                </div>
                <div>
                    <h3 class="costs-table-label">By API Key</h3>
                    <table class="cost-table" id="cost-by-key"><thead><tr><th>Key</th><th>Calls</th><th>Cost</th><th></th></tr></thead><tbody></tbody></table>
                </div>
                <div>
                    <h3 class="costs-table-label">By Model Category</h3>
                    <table class="cost-table" id="cost-by-model-cat"><thead><tr><th>Category</th><th>Calls</th><th>Cost</th><th></th></tr></thead><tbody></tbody></table>
                </div>
                <div>
                    <h3 class="costs-table-label">By Task Category</h3>
                    <table class="cost-table" id="cost-by-task-cat"><thead><tr><th>Category</th><th>Calls</th><th>Cost</th><th></th></tr></thead><tbody></tbody></table>
                </div>
            </div>
        </div>
    `;
    mount.appendChild(page);

    function renderBreakdownTable(tableId, data, totalCost, emptyLabel = 'No data') {
        const tbody = document.querySelector('#' + tableId + ' tbody');
        tbody.innerHTML = '';
        const cell = (className, text, attrs = {}) => {
            const td = document.createElement('td');
            td.className = className;
            td.textContent = text;
            Object.entries(attrs).forEach(([key, value]) => td.setAttribute(key, value));
            return td;
        };
        for (const [name, info] of Object.entries(data)) {
            const pct = totalCost > 0 ? (info.cost / totalCost * 100) : 0;
            const tr = document.createElement('tr');
            const bar = document.createElement('progress');
            bar.className = 'cost-bar';
            bar.max = 100;
            bar.value = Math.min(100, pct);
            const tdBar = document.createElement('td');
            tdBar.className = 'cost-bar-cell';
            tdBar.appendChild(bar);
            tr.append(
                cell('cost-cell-name', name, { title: name }),
                cell('cost-cell-right', info.calls),
                cell('cost-cell-right', costBucketPresentation(info)),
                tdBar,
            );
            tbody.appendChild(tr);
        }
        if (Object.keys(data).length === 0) {
            const tr = document.createElement('tr');
            tr.appendChild(cell('cost-empty-cell', emptyLabel, { colspan: '4' }));
            tbody.appendChild(tr);
        }
    }

    async function loadCosts() {
        const renderLoading = () => {
            document.getElementById('cost-known-limit').textContent = 'Loading…';
            ['cost-confirmed', 'cost-estimated', 'cost-reserved', 'cost-unresolved', 'cost-unknown',
                'cost-calls', 'cost-top-model'].forEach((id) => {
                document.getElementById(id).textContent = '—';
            });
            document.getElementById('cost-final').textContent = 'Loading…';
        };
        const renderUnavailable = () => {
            ['cost-known-limit', 'cost-confirmed', 'cost-estimated', 'cost-reserved', 'cost-unresolved',
                'cost-unknown', 'cost-calls', 'cost-top-model'].forEach((id) => {
                document.getElementById(id).textContent = id === 'cost-known-limit' ? 'Unavailable' : '—';
            });
            document.getElementById('cost-final').textContent = 'Unavailable';
            ['cost-by-model', 'cost-by-key', 'cost-by-model-cat', 'cost-by-task-cat']
                .forEach((id) => renderBreakdownTable(id, {}, 0, 'Unavailable'));
        };
        renderLoading();
        try {
            const resp = await apiFetch('/api/cost-breakdown');
            const d = await resp.json();
            const presentation = costDashboardPresentation(d);
            if (!resp.ok || presentation.state !== 'available') throw new Error('accounting unavailable');
            document.getElementById('cost-known-limit').textContent = presentation.knownLimit;
            document.getElementById('cost-confirmed').textContent = presentation.confirmed;
            document.getElementById('cost-estimated').textContent = presentation.estimated;
            document.getElementById('cost-reserved').textContent = presentation.reserved;
            document.getElementById('cost-unresolved').textContent = presentation.unresolved;
            document.getElementById('cost-unknown').textContent = presentation.unknown;
            document.getElementById('cost-final').textContent = presentation.final;
            document.getElementById('cost-calls').textContent = presentation.calls;
            document.getElementById('cost-top-model').textContent = presentation.topModel;
            renderBreakdownTable('cost-by-model', d.by_model || {}, d.total_cost);
            renderBreakdownTable('cost-by-key', d.by_api_key || {}, d.total_cost);
            renderBreakdownTable('cost-by-model-cat', d.by_model_category || {}, d.total_cost);
            renderBreakdownTable('cost-by-task-cat', d.by_task_category || {}, d.total_cost);
        } catch { renderUnavailable(); }
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

    document.getElementById('btn-refresh-costs').addEventListener('click', loadCosts);

    document.getElementById('btn-save-budget').addEventListener('click', async (event) => {
        const statusEl = document.getElementById('budget-save-status');
        const budget = readPositiveBudget('s-budget');
        const perTask = readPositiveBudget('s-per-task-cost');
        if (budget === null || perTask === null) {
            statusEl.textContent = 'Budget values must be at least 0.01.';
            return;
        }
        const saveButton = event.currentTarget;
        saveButton.disabled = true;
        saveButton.setAttribute('aria-busy', 'true');
        statusEl.textContent = 'Saving…';
        statusEl.dataset.tone = 'muted';
        try {
            const resp = await apiFetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ TOTAL_BUDGET: budget, OUROBOROS_PER_TASK_COST_USD: perTask }),
            });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
            let msg;
            if (data.no_changes) {
                msg = 'No changes.';
            } else if (data.restart_required) {
                msg = 'Saved. Restart required.';
            } else if (data.immediate_changed && data.next_task_changed) {
                msg = 'Saved. Some changes took effect immediately; others apply on the next task.';
            } else if (data.immediate_changed) {
                msg = 'Saved. Took effect immediately.';
            } else {
                msg = 'Saved. Applies on the next task.';
            }
            if (data.warnings && data.warnings.length) msg += ' ⚠️ ' + data.warnings.join(' | ');
            statusEl.textContent = msg;
            statusEl.dataset.tone = (data.warnings?.length || data.restart_required) ? 'warn' : 'ok';
            window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { reason: 'budget saved', source: 'costs' } }));
        } catch (e) {
            statusEl.textContent = 'Error: ' + e.message;
            statusEl.dataset.tone = 'error';
        } finally {
            saveButton.disabled = false;
            saveButton.removeAttribute('aria-busy');
        }
        setTimeout(() => { statusEl.textContent = ''; }, 4000);
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
