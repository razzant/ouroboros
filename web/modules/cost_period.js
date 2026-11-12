// Dashboard › Costs, "By period": what the model calls started in a period cost, by token class, and what each
// cache write followed (`GET /api/cost-breakdown?period=…`, ouroboros/gateway/cost_period.py). The parts are each
// call's RECORDED cost shared out by the model's price list, never invoice lines; what could not be shared is named
// beside them. Four fixed hues name the four classes (ui.css `--chart-series-*`): none of them is a verdict, and the
// card raises no signal. The page's totals and tables above stay all-time; the period governs this card alone.
import { formatUsd, number } from './storage_format.js';
import { apiFetch } from './api_client.js';
import { escapeHtmlAttr } from './ui_primitives.js';

export const COST_PERIODS = [['24h', '24 hours'], ['7d', '7 days'], ['30d', '30 days']];
const CLASS_WORDS = [['cache_write', 'Cache writes'], ['cache_read', 'Cache reads'], ['output', 'Output'], ['uncached_input', 'Uncached input']];
// What a cache write followed, in the words of the attempt's recorded fact (transcript_prefix.prefix_class).
const WRITE_WORDS = Object.freeze({
    first_request: 'First request of a task',
    entered_model: 'A running task switched to this model',
    tools_changed: 'Tool descriptions changed',
    system_changed: 'A part of the request start changed',
    history_rewritten: 'Earlier history was rewritten',
    paused: 'Paused longer than the cache lifetime',
    returned_to_model: 'A task came back to this model',
    append: 'Next step of the same request',
    not_recorded: 'Not recorded',
});
// The fact's own detail beside the words: the model a task came from, the block that changed, the seams that rewrote.
const writeDetail = (name, detail) => (!detail ? '' : (name === 'entered_model' || name === 'returned_to_model')
    ? ` after ${detail}` : ` (${String(detail).replaceAll('_', ' ')})`);

/** Tokens at the precision a glance needs: 1.2B, 345M, 12.4M, 630k, 812. */
export function compactTokens(value) {
    const n = Number(value) || 0;
    const scaled = (unit, word) => `${(n / unit).toFixed(n / unit >= 100 ? 0 : 1).replace(/\.0$/, '')}${word}`;
    if (n >= 1e9) return scaled(1e9, 'B');
    if (n >= 1e6) return scaled(1e6, 'M');
    return n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(Math.round(n));
}

const percent = (share) => (share > 0 && share < 0.005 ? '<1%' : `${Math.round(share * 100)}%`);

/** The card's words for one period view: models by cost, each with its parts, and the cache writes by what they followed. */
export function costPeriodPresentation(view) {
    if (!view || view.available !== true) {
        return { state: 'unavailable', models: [], writes: [], note: 'This period could not be read. The totals above are unaffected.' };
    }
    const models = (view.models || []).filter((row) => row.calls || row.cost || row.open_rows).map((row) => {
        const split = row.by_class ? CLASS_WORDS.map(([key, label]) => ({ key, label, usd: Number(row.by_class[key]) || 0 })) : null;
        const shared = split ? split.reduce((sum, part) => sum + part.usd, 0) : 0;
        const notes = [];  // one line under the parts: what the amounts leave out, then the price ratio
        if (!split && Number(row.cost) > 0) notes.push('Not split by class: no price list for this model is at hand.');
        else if (Number(row.unshared_cost) >= 0.005) notes.push(`${formatUsd(row.unshared_cost)} is not split (no price list, token counts or named cache lifetime for those calls).`);
        if (row.open_rows) notes.push(`${number(row.open_rows)} ${row.open_rows === 1 ? 'call is' : 'calls are'} still open.`);
        if (row.unpriced_rows) notes.push(`${number(row.unpriced_rows)} ended without a price.`);
        if (row.write_price_in_reads) notes.push(`One written token costs ${number(row.write_price_in_reads)} read tokens here.`);
        return {
            name: row.model,
            sum: `${formatUsd(row.cost)} · ${number(row.calls)} ${row.calls === 1 ? 'call' : 'calls'}`,
            parts: CLASS_WORDS.map(([key, label], index) => {
                const usd = split ? split[index].usd : null;
                const share = shared > 0 && usd !== null ? usd / shared : 0;
                const tokens = `${compactTokens(row.tokens?.[key])} tokens`;
                return { key, label, share, text: usd === null ? tokens : `${formatUsd(usd)} · ${percent(share)} · ${tokens}` };
            }),
            split: Boolean(split && shared > 0),
            notes,
        };
    });
    const writes = (view.cache_writes || []).map((row) => ({
        label: `${WRITE_WORDS[row.class] || row.class}${writeDetail(row.class, row.detail)}`,
        calls: number(row.calls),
        tokens: compactTokens(row.write_tokens),
        cost: row.unshared_calls >= row.calls ? '—' : formatUsd(row.write_cost),
    }));
    const observed = String(view.tariff_observed_at || '').slice(0, 10);
    return {
        state: models.length ? 'available' : 'empty',
        models,
        writes,
        note: models.length
            ? `Each call’s recorded cost is shared among the four classes by the model’s price list${observed ? ` (as recorded on ${observed})` : ''}: `
                + 'a share of the recorded cost, not the provider’s invoice lines. A period holds the calls started in it.'
            : 'No model calls started in this period.',
    };
}

const modelHtml = (model) => `<article class="costs-period-model">
        <div class="costs-period-head"><span class="costs-period-name">${escapeHtmlAttr(model.name)}</span><span class="costs-period-sum">${escapeHtmlAttr(model.sum)}</span></div>
        ${model.split ? `<div class="costs-period-bar" role="img" aria-label="${escapeHtmlAttr(model.parts.map((part) => `${part.label} ${percent(part.share)}`).join(', '))}">${
        model.parts.filter((part) => part.share > 0).map((part) => `<i data-cost-class="${part.key}" style="--share:${(part.share * 100).toFixed(2)}"></i>`).join('')}</div>` : ''}
        <dl class="costs-period-parts">${model.parts.map((part) => `<div><dt><i data-cost-class="${part.key}"></i>${part.label}</dt><dd>${escapeHtmlAttr(part.text)}</dd></div>`).join('')}</dl>
        ${model.notes.length ? `<p class="costs-copy">${escapeHtmlAttr(model.notes.join(' '))}</p>` : ''}
    </article>`;

const writesHtml = (writes) => (writes.length ? `<h4 class="costs-period-subtitle">Cache writes by what they followed</h4>
        <table class="cost-table costs-period-writes"><thead><tr><th>Before the write</th><th class="cost-cell-right">Calls</th><th class="cost-cell-right">Written</th><th class="cost-cell-right">Share of cost</th></tr></thead>
        <tbody>${writes.map((row) => `<tr><td>${escapeHtmlAttr(row.label)}</td><td class="cost-cell-right">${row.calls}</td><td class="cost-cell-right">${row.tokens}</td><td class="cost-cell-right">${row.cost}</td></tr>`).join('')}</tbody></table>` : '');

async function fetchPeriod(period) {
    const response = await apiFetch(`/api/cost-breakdown?period=${encodeURIComponent(period)}`);
    return (await response.json()).period;
}

/** Mount the card into `host`; returns `{ reload }`. Nothing is read until the page calls `reload` (the tab shown,
 *  Refresh) or the owner picks a period. A later choice supersedes an earlier read. */
export function mountCostPeriod(host, { load = fetchPeriod, period = '24h' } = {}) {
    host.innerHTML = `<section class="ui-card costs-period" aria-labelledby="costs-period-title">
        <h3 class="dashboard-card-title" id="costs-period-title">By period</h3>
        <p class="costs-table-note">Where the money of a period went: cache writes, cache reads, output and uncached input, per model.</p>
        <div class="costs-period-choice" role="group" aria-label="Period">${COST_PERIODS.map(([key, label]) =>
        `<button type="button" class="ui-segment" data-cost-period="${key}" aria-pressed="false">${label}</button>`).join('')}</div>
        <p class="ui-status costs-period-status" role="status" aria-live="polite"></p>
        <div class="costs-period-models"></div>
        <div class="costs-period-writes-host"></div>
        <p class="costs-copy costs-period-note"></p>
    </section>`;
    const status = host.querySelector('.costs-period-status');
    const modelsEl = host.querySelector('.costs-period-models');
    const writesEl = host.querySelector('.costs-period-writes-host');
    const noteEl = host.querySelector('.costs-period-note');
    let chosen = period;
    let read = 0;
    async function reload() {
        const mine = ++read;
        host.querySelectorAll('[data-cost-period]').forEach((button) => {
            const active = button.dataset.costPeriod === chosen;
            button.classList.toggle('active', active);
            button.setAttribute('aria-pressed', String(active));
        });
        status.textContent = 'Counting…';
        let view = null;
        try { view = await load(chosen); } catch { view = null; }
        if (mine !== read) return;  // a later choice owns the card
        const shown = costPeriodPresentation(view);
        status.textContent = '';
        modelsEl.innerHTML = shown.models.map(modelHtml).join('');
        writesEl.innerHTML = writesHtml(shown.writes);
        noteEl.textContent = shown.note;
    }
    host.addEventListener('click', (event) => {
        const button = event.target.closest('[data-cost-period]');
        if (!button || button.dataset.costPeriod === chosen) return;
        chosen = button.dataset.costPeriod;
        reload();
    });
    return { reload };
}
