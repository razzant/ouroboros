import {
    accountedUpperBound,
    accountedUpperBoundWithChildren,
    formatUsd2,
    formatUsdWhole,
    rawTimestampEpoch,
} from './utils.js';
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

/** Pure presentation projection used by the header and dependency-free tests. */
export function headerBudgetPresentation(data) {
    if (!data || data.accounting_loading === true) {
        return { state: 'loading', label: 'Loading…', fillPct: 0 };
    }
    if (data?.accounting?.available === false) {
        return { state: 'unavailable', label: 'Unavailable', fillPct: 0 };
    }
    // Older state shapes did not carry accounting.available.  Keep accepting
    // them when they contain a real numeric projection, but never coerce null
    // (ledger failure in the new shape) into a convincing $0.
    const spent = optionalFiniteNumber(data.spent_usd);
    if (spent === null) {
        return { state: 'unavailable', label: 'Unavailable', fillPct: 0 };
    }
    const rawLimit = optionalFiniteNumber(data.budget_limit);
    const limit = rawLimit !== null && rawLimit > 0 ? rawLimit : 0;
    const label = typeof data.budget_text === 'string' && data.budget_text.trim()
        ? data.budget_text
        : `${formatUsdWhole(spent)} / ${limit > 0 ? formatUsdWhole(limit) : '∞'}`;
    return {
        state: 'available',
        label,
        fillPct: limit > 0 ? Math.min(100, Math.max(0, (spent / limit) * 100)) : 0,
    };
}

/**
 * Render task money without conflating unknown/non-final values with a final
 * zero.  The returned strings are card metadata, not another cost authority.
 */
export function taskCostMeta(payload = {}) {
    const has = (key) => Object.prototype.hasOwnProperty.call(payload, key);
    // Task-scope accounting evidence only (v6.82 P1): a bare `cost_usd` is NOT
    // enough — llm_round_finished carries a per-round delta under that key, and
    // rendering it as task cost lied on the card. Subagent progress_meta and
    // task_done/task_cost_finalized frames carry cost_accounting_status /
    // cost_final alongside cost_usd, so honest task-scope frames still qualify.
    const hasAccountingEvidence = [
        'cost_accounting_status', 'cost_final',
        'cost_usd_with_children', 'cost_with_children_partial',
        'accounted_upper_bound_usd', 'accounted_upper_bound_usd_with_children',
        'reserved_usd', 'unresolved_upper_bound_usd', 'unknown_unmetered',
    ].some(has);
    if (!hasAccountingEvidence) return [];
    if (payload.cost_accounting_status === 'unavailable') return ['cost unavailable'];

    // C2/F12: ONE precedence resolver, shared with the Python seams and with
    // log_events — the deprecated alias wins a diverged pair, so the read side
    // and the write side never pick opposite winners for the same record.
    const own = accountedUpperBound(payload);
    const finalKnown = payload.cost_final === true;
    const pendingKnown = payload.cost_final === false
        || payload.cost_with_children_partial === true
        || payload.cost_accounting_status === 'available' && !has('cost_final');
    const meta = [];
    if (own === null) {
        meta.push('cost pending');
    } else if (finalKnown || pendingKnown || own !== 0) {
        meta.push(`cost=$${own.toFixed(2)}${pendingKnown && !finalKnown ? ' (pending)' : ''}`);
    }

    const subtree = accountedUpperBoundWithChildren(payload);
    if (subtree !== null && (
        own === null || subtree !== own || payload.cost_with_children_partial === true
    )) {
        const partial = payload.cost_with_children_partial === true || !finalKnown;
        meta.push(`subtree=$${subtree.toFixed(2)}${partial ? ' (pending)' : ''}`);
    }
    const reserved = optionalFiniteNumber(payload.reserved_usd);
    if (reserved !== null && reserved > 0) meta.push(`reserved=$${reserved.toFixed(2)}`);
    const unresolved = optionalFiniteNumber(payload.unresolved_upper_bound_usd);
    if (unresolved !== null && unresolved > 0) meta.push(`unresolved≤$${unresolved.toFixed(2)}`);
    const unknown = optionalFiniteNumber(payload.unknown_unmetered);
    if (unknown !== null && unknown > 0) meta.push(`unmetered=${Math.trunc(unknown)}`);
    return meta;
}

/**
 * Project one frame's task-scope cost evidence into the sticky structured form
 * `{meta, ts, final}` (v6.82 P1). Returns null when the frame carries NO
 * task-scope accounting evidence (e.g. an llm_round_finished per-round delta)
 * — such frames must never touch a card's cost.
 */
export function taskCostProjection(payload = {}, rawTs = '') {
    const meta = taskCostMeta(payload);
    if (!meta.length) return null;
    const unavailable = payload.cost_accounting_status === 'unavailable';
    return {
        meta,
        ts: rawTimestampEpoch(rawTs),
        // Only a SETTLED ledger value is final. "unavailable" is an honest
        // unknown, not a settled truth: marking it final let one transient
        // ledger-read failure outrank every later real reading.
        final: payload.cost_final === true,
        unavailable,
    };
}

/**
 * Sticky per-card cost precedence (v6.82 P1). Rank unavailable < pending < final:
 * an honest reading always outranks an unknown (one transient ledger-read failure
 * must not pin the card for the whole run) and a settled value outranks both.
 * Among equal rank the newer raw source timestamp wins, so an older history replay
 * can never overwrite newer evidence; frames without evidence (null `next`) keep
 * the previous projection, so an unavailable snapshot is still sticky.
 */
export function mergeStickyCostMeta(previous, next) {
    if (!next || !Array.isArray(next.meta) || !next.meta.length) return previous || null;
    if (!previous || !Array.isArray(previous.meta) || !previous.meta.length) return next;
    // Rank: unavailable < pending < final. An `unavailable` snapshot is sticky (a
    // costless frame must not erase it) but must NOT outrank a later HONEST reading:
    // one transient ledger-read failure would otherwise pin the card to "cost
    // unavailable" for the rest of the run.
    const rank = (p) => (p.final ? 2 : (p.unavailable ? 0 : 1));
    const prevRank = rank(previous);
    const nextRank = rank(next);
    if (prevRank !== nextRank) return nextRank > prevRank ? next : previous;
    const prevTs = Number(previous.ts);
    const nextTs = Number(next.ts);
    if (Number.isFinite(prevTs) && Number.isFinite(nextTs) && nextTs < prevTs) return previous;
    // A frame whose source timestamp is unreadable must not defeat a
    // timestamped previous value of equal finality.
    if (Number.isFinite(prevTs) && !Number.isFinite(nextTs)) return previous;
    return next;
}

/** Pure cost-dashboard projection: null/unavailable never renders as $0. */
export function costDashboardPresentation(data) {
    if (!data) return { state: 'loading' };
    const accounting = data.accounting || {};
    if (accounting.available === false) return { state: 'unavailable' };
    const accounted = optionalFiniteNumber(accounting.accounted_usd);
    const confirmed = optionalFiniteNumber(accounting.confirmed_usd);
    const reserved = optionalFiniteNumber(accounting.reserved_usd);
    const unresolved = optionalFiniteNumber(accounting.unresolved_upper_bound_usd);
    const unknown = optionalFiniteNumber(accounting.unknown_unmetered);
    const calls = optionalFiniteNumber(data.total_calls);
    if ([accounted, confirmed, reserved, unresolved, unknown, calls].some(value => value === null)) {
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
        accountedLimit: `${formatUsd2(accounted)} / ${limit > 0 ? formatUsd2(limit) : '∞'}`,
        confirmed: formatUsd2(confirmed),
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
                    <div class="form-field">
                        <label>Total Budget ($)</label>
                        <input id="s-budget" type="number" value="200">
                    </div>
                    <div class="form-field">
                        <label>Per-task Cost Cap ($)</label>
                        <input id="s-per-task-cost" type="number" value="50">
                        <div class="settings-inline-note">Hard dispatch cap for the whole root task tree. In-flight calls settle normally; increasing the cap does not auto-resume paused work.</div>
                    </div>
                </div>
                <button class="btn btn-save costs-budget-save" id="btn-save-budget">Save Budget</button>
                <div id="budget-save-status" class="settings-inline-status"></div>
            </div>
            <div class="costs-stats-grid">
                <div class="stat-card"><div class="label">Accounted / Limit</div><div class="value" id="cost-accounted-limit">Loading…</div></div>
                <div class="stat-card"><div class="label">Confirmed</div><div class="value" id="cost-confirmed">—</div></div>
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
                cell('cost-cell-right', formatUsd2(info.cost)),
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
            document.getElementById('cost-accounted-limit').textContent = 'Loading…';
            ['cost-confirmed', 'cost-reserved', 'cost-unresolved', 'cost-unknown',
                'cost-calls', 'cost-top-model'].forEach((id) => {
                document.getElementById(id).textContent = '—';
            });
            document.getElementById('cost-final').textContent = 'Loading…';
        };
        const renderUnavailable = () => {
            ['cost-accounted-limit', 'cost-confirmed', 'cost-reserved', 'cost-unresolved',
                'cost-unknown', 'cost-calls', 'cost-top-model'].forEach((id) => {
                document.getElementById(id).textContent = id === 'cost-accounted-limit' ? 'Unavailable' : '—';
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
            document.getElementById('cost-accounted-limit').textContent = presentation.accountedLimit;
            document.getElementById('cost-confirmed').textContent = presentation.confirmed;
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

    document.getElementById('btn-save-budget').addEventListener('click', async () => {
        const statusEl = document.getElementById('budget-save-status');
        const budget = readPositiveBudget('s-budget');
        const perTask = readPositiveBudget('s-per-task-cost');
        if (budget === null || perTask === null) {
            statusEl.textContent = 'Budget values must be at least 0.01.';
            return;
        }
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
            window.dispatchEvent(new CustomEvent('ouro:settings-updated', { detail: { reason: 'budget saved', source: 'costs' } }));
        } catch (e) {
            statusEl.textContent = 'Error: ' + e.message;
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

// Presentation of one live frame's task-scope cost evidence: a `replace` frame
// (task_done/task_cost_finalized) drops the summarizer's own meta strings, and a
// summarizer-built `cost=` string is dropped unconditionally — money renders ONLY
// from the card's sticky projection, never from a bare per-call number.
export function withTaskCostMeta(summary, payload, { replace = false, rawTs = '' } = {}) {
    const projection = taskCostProjection(payload, rawTs);
    // `replace` frames (task_done/task_cost_finalized) never keep the
    // summarizer's own meta strings. Cost renders ONLY from the card's sticky
    // record.costMeta (applyLiveCardState); summarizer-built `cost=` strings
    // are dropped UNCONDITIONALLY — a frame without task-scope accounting
    // evidence must show no money at all, not a bare per-call number.
    const base = replace ? { ...summary, meta: [] } : summary;
    const out = projection ? { ...base, costProjection: projection } : { ...base };
    if (Array.isArray(out.meta) && out.meta.length) {
        out.meta = out.meta.filter((entry) => !String(entry || '').startsWith('cost='));
    }
    return out;
}
