// Pure projection, markup, and minimal in-place patching for the Settings › Agents list view.
// One projection drives both full list markup and discovery-only patching so the cached HTML
// never claims untouched chrome changed, surviving controls/disclosures keep their elements and
// focus, filtered rows stay in roster order, and inactive zero-count filter chips stay omitted.

import { harnessIdentityMarkup, identityPartsHtml, META_PART_SEPARATOR } from './harness_presentation.js';
import { DEFAULT_API_PROVIDER, connectionParts, routeModelFields } from './route_editor_primitives.js';
import { modelTariffText, rowIdentity, rowStatus } from './subagent_status_primitives.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export const FILTERS = [
    ['all', 'All'],
    ['reviewers', 'Reviewers'],
    ['off', 'Off'],
    ['unavailable', 'Unavailable'],
];

export function inFilter(row, state, key) {
    if (key === 'all') return true;
    if (key === 'reviewers') return row.review_eligible === true;
    if (key === 'off') return row.enabled === false;
    return rowStatus(row, state).tone === 'warn';
}

const chip = (text, tone = 'neutral') =>
    `<span class="ui-chip settings-chip agent-chip" data-tone="${tone}">${escapeHtml(text)}</span>`;

/**
 * A list head's published rate (docs/DESIGN.md §6): an OpenRouter key's full figures or `free`,
 * from that key's own catalog while the key is still configured and its last read succeeded (a
 * failed one keeps the old rows for the chooser, never as the price now); nothing for a condition,
 * an unknown price, another tier or a row that pays no per-token rate. Never the measured Review
 * cost, which is the agent page's. The list head shows it in place.
 */
export function listRate(row, state) {
    const { provider } = routeModelFields(row?.route || {});
    const catalog = state.rowCatalog?.(row);
    if (connectionParts(provider).kind !== DEFAULT_API_PROVIDER
        || !(state.providers || []).some((entry) => entry.id === provider)
        || catalog?.read_state !== 'ok') return '';
    return modelTariffText(row, catalog, state.processingPreference, { rate: true });
}

/**
 * A list head's facts a read can change: its name and its marks, the mark of its source, and the
 * identity line with the rate. The list draws them and patches them from here, so the two never
 * disagree.
 */
export function listHeadFacts(row, state, name) {
    const identity = rowIdentity(row, state);
    const rate = listRate(row, state);
    const status = rowStatus(row, state);
    const rateHtml = rate
        ? `<span class="harness-identity-sep" data-i18n-skip>${escapeHtml(META_PART_SEPARATOR)}</span>`
            + `<span data-subagent-rate data-i18n-fmt="model price">${escapeHtml(rate)}</span>`
        : '';
    return {
        name,
        title: `${escapeHtml(name)}${row.review_eligible === true ? chip('Reviewer') : ''}`
            + `${status.tone === 'warn' ? chip(status.label, 'warn') : ''}`,
        mark: harnessIdentityMarkup(identity.harnessId, {
            label: identity.label,
            channel: identity.channel,
            className: 'available-subagent-route-identity',
        }),
        line: `${identityPartsHtml(identity.parts)}${rateHtml}`,
    };
}

export function renderListFactsHtml({ readProblem = '', diagnostics = '' } = {}) {
    return `<div class="available-subagents-source" data-subagents-read-problem ${readProblem ? '' : 'hidden'}>${escapeHtml(readProblem)}</div>
            <div data-subagents-diagnostics class="available-subagents-diagnostics" ${diagnostics ? '' : 'hidden'}>${escapeHtml(diagnostics)}</div>`;
}

export function renderListAboveHtml(above = {}) {
    return `<div class="available-subagents-source" data-review-pool-stays ${above.stays ? '' : 'hidden'}>${escapeHtml(above.stays || '')}</div>
            <div class="available-subagents-diagnostics" data-review-pool-note ${above.note ? '' : 'hidden'}>${escapeHtml(above.note || '')}</div>
            <details class="model-role-details ui-disclosure" data-review-pool-history ${above.history ? '' : 'hidden'}><summary>Reviewer origin</summary>
                <div class="available-subagents-source" data-review-pool-history-text>${escapeHtml(above.history || '')}</div></details>
            ${renderListFactsHtml(above)}`;
}

// Lines a row edit can toggle (the empty pool, the validation summary) sit below the rows:
// a conditional line never moves the control that caused it (docs/DESIGN.md §6).
export function renderListBelowHtml(below = {}) {
    return `<div class="available-subagents-review-empty" data-review-pool-empty data-tone="warn" ${below.empty ? '' : 'hidden'}>
                <span data-review-pool-empty-text>${escapeHtml(below.empty || '')}</span>
                <label class="local-toggle ui-field-inline" data-review-pool-confirm ${below.confirm ? '' : 'hidden'}><input class="ui-checkbox" type="checkbox" data-review-pool-allow-empty ${below.allowEmpty ? 'checked' : ''}> Save without reviewers</label>
            </div>
            <div class="available-subagents-review-copies" data-review-pool-copies data-tone="warn" ${below.copies ? '' : 'hidden'}>
                <span data-review-pool-copies-text>${escapeHtml(below.copies || '')}</span>
                <button type="button" class="btn btn-default" data-review-pool-keep-one>Keep one per model</button>
            </div>
            <div data-subagents-validation class="available-subagents-diagnostics" data-tone="error" ${below.validation ? '' : 'hidden'}>${escapeHtml(below.validation || '')}</div>`;
}

function filterChipHtml(filter) {
    return `<button type="button" class="filter-chip${filter.active ? ' active' : ''}" data-agents-filter="${escapeHtml(filter.key)}" aria-pressed="${filter.active}">${escapeHtml(`${filter.label} ${filter.count}`)}</button>`;
}

function listHeadHtml(item, renderRowActions) {
    const { facts, use } = item;
    return `<div class="ui-row settings-head" data-subagent-row="${escapeHtml(item.key)}" data-entry-id="${escapeHtml(item.id)}"><span class="settings-head-glyph" aria-hidden="true" data-i18n-skip>${facts.mark}</span>
            <button type="button" class="settings-head-open ui-hit-area" data-subagent-field="open">
                <span class="ui-row-title">${facts.title}</span>
                <span class="ui-row-meta" data-subagent-identity>${facts.line}</span>
                ${use ? `<span class="ui-row-meta agents-head-use" data-i18n-authored>${escapeHtml(use)}</span>` : ''}
            </button>
            <div class="ui-row-actions">${renderRowActions(item)}</div></div>`;
}

/** One canonical desired projection of the Settings › Agents list view. */
export function projectAgentsList(state, {
    readProblem = '', diagnostics = [], pool = {}, errors = [], atLimit = false,
    said = (text) => text, agentName, unloadedHtml = '', footerText = '',
} = {}) {
    const items = state.setting?.items || [];
    const loaded = Boolean(state.loaded);
    const filters = FILTERS
        .map(([key, label]) => ({
            key,
            label,
            count: items.filter((row) => inFilter(row, state, key)).length,
            active: key === state.filter,
        }))
        .filter((f) => f.key === 'all' || f.active || f.count > 0);
    const rows = loaded
        ? items
            .filter((row) => inFilter(row, state, state.filter))
            .map((row) => ({
                key: row._uiKey || row.subagent_id,
                id: row.subagent_id,
                enabled: row.enabled !== false,
                use: String(row.recommended_use || '').trim(),
                facts: listHeadFacts(row, state, agentName(row, state)),
            }))
        : [];
    return {
        loaded,
        title: `${items.length} agent${items.length === 1 ? '' : 's'}`,
        addHeld: Boolean(atLimit),
        automatic: state.source === 'undecided' || state.automatic === true,
        filters,
        rows,
        emptyText: items.length
            ? 'No agent here.'
            : 'No agents yet, so the main model is the one agent. Add one to choose others.',
        unloadedHtml,
        above: {
            stays: pool.stays || '',
            note: pool.note || '',
            history: pool.history || '',
            readProblem: readProblem || '',
            diagnostics: diagnostics.join(' · '),
        },
        below: {
            empty: pool.empty || '',
            confirm: Boolean(pool.confirm),
            allowEmpty: Boolean(state.allowEmptyReviewPool),
            copies: pool.copies ? said(pool.copies) : '',
            validation: errors[0] || '',
        },
        footerText,
    };
}

export function renderAgentsListHtml(projection, { renderRowActions }) {
    const heads = projection.rows.map((item) => listHeadHtml(item, renderRowActions)).join('');
    const empty = `<p class="available-subagents-empty">${escapeHtml(projection.emptyText)}</p>`;
    return `<section class="ui-card settings-card">
                <div class="settings-card-head"><h3 class="settings-card-title agents-card-title">${escapeHtml(projection.title)}</h3>
                    <span class="ui-result" data-entry-result role="status" aria-live="polite"></span>
                    <div class="ui-section-actions"><span class="ui-hint-host" data-full-hint><button type="button" class="btn btn-default btn-sm" data-subagent-add ${projection.loaded ? '' : 'disabled'}${projection.addHeld ? ' aria-disabled="true"' : ''}>Add agent</button></span></div></div>
                <div class="agents-filters">${projection.filters.map(filterChipHtml).join('')}</div>
                ${projection.automatic ? '<p class="available-subagents-source" data-agents-automatic>Automatic: this list follows your Main model and accounts. Your first change here saves it as yours.</p>' : ''}
                ${renderListAboveHtml(projection.above)}
                <div class="settings-rows">${!projection.loaded ? projection.unloadedHtml : heads || empty}</div>
                ${renderListBelowHtml(projection.below)}
                <p class="available-subagents-source">${escapeHtml(projection.footerText)}</p>
            </section>`;
}

/**
 * Patch every discovery-mutable region of the mounted Settings › Agents list in place, or return
 * false so the caller falls back to a full paint whenever unpatched structure or chrome changed.
 * Unpatched structure is compared against the previous painted canonical projection (never
 * i18n-translated DOM text or live input properties), and surviving elements keep translated
 * text when their canonical source field did not change.
 */
export function patchAgentsList(container, projection, {
    previous = null, doc, renderRowActions, onSelectFilter, bindNewRows,
} = {}) {
    if (!projection.loaded || !previous?.loaded) return false;
    const filterRow = container.querySelector('.agents-filters');
    const list = container.querySelector('.settings-rows');
    const addBtn = container.querySelector('[data-subagent-add]');
    if (!filterRow || !list || !addBtn || addBtn.disabled) return false;
    if (previous.title !== projection.title) return false;
    if (previous.addHeld !== projection.addHeld) return false;
    if (previous.automatic !== projection.automatic) return false;
    if (previous.below.allowEmpty !== projection.below.allowEmpty) return false;

    const prevRows = new Map(previous.rows.map((item) => [item.key, item]));
    const existingRows = new Map([...list.children]
        .filter((node) => node.dataset?.subagentRow)
        .map((node) => [node.dataset.subagentRow, node]));
    for (const item of projection.rows) {
        if (!existingRows.has(item.key)) continue;
        const prev = prevRows.get(item.key);
        if (!prev || prev.id !== item.id || prev.use !== item.use || prev.enabled !== item.enabled) {
            return false;
        }
    }

    // Reconcile filter chips in canonical order: omit inactive zero-count chips, preserve survivors.
    const prevFilters = new Map(previous.filters.map((filter) => [filter.key, filter]));
    const wantedFilters = new Set(projection.filters.map((f) => f.key));
    for (const child of [...filterRow.children]) {
        if (!wantedFilters.has(child.dataset?.agentsFilter)) child.remove();
    }
    let filterCursor = filterRow.firstElementChild;
    for (const filter of projection.filters) {
        let btn = filterRow.querySelector(`[data-agents-filter="${filter.key}"]`);
        if (!btn) {
            const tpl = doc.createElement('template');
            tpl.innerHTML = filterChipHtml(filter);
            btn = tpl.content.firstElementChild;
            btn.addEventListener('click', () => onSelectFilter(filter.key));
            filterRow.insertBefore(btn, filterCursor || null);
        } else {
            if (btn !== filterCursor) filterRow.insertBefore(btn, filterCursor || null);
            else filterCursor = btn.nextElementSibling;
            const prevFilter = prevFilters.get(filter.key);
            const text = `${filter.label} ${filter.count}`;
            if (!prevFilter || `${prevFilter.label} ${prevFilter.count}` !== text) btn.textContent = text;
            if (!prevFilter || prevFilter.active !== filter.active) {
                btn.classList.toggle('active', filter.active);
                btn.setAttribute('aria-pressed', String(filter.active));
            }
        }
    }

    // Patch above/below chrome in place so Keep one per model and open Reviewer origin stay intact.
    const patchLine = (selector, text, prevText, visible = Boolean(text), prevVisible = Boolean(prevText)) => {
        const node = container.querySelector(selector);
        if (!node) return;
        if (text !== null && text !== prevText) node.textContent = text;
        if (visible !== prevVisible) node.hidden = !visible;
    };
    patchLine('[data-review-pool-stays]', projection.above.stays, previous.above.stays);
    patchLine('[data-review-pool-note]', projection.above.note, previous.above.note);
    patchLine('[data-review-pool-history]', null, null, Boolean(projection.above.history), Boolean(previous.above.history));
    patchLine('[data-review-pool-history-text]', projection.above.history, previous.above.history, true, true);
    patchLine('[data-subagents-read-problem]', projection.above.readProblem, previous.above.readProblem);
    patchLine('[data-subagents-diagnostics]', projection.above.diagnostics, previous.above.diagnostics);
    patchLine('[data-review-pool-empty]', null, null, Boolean(projection.below.empty), Boolean(previous.below.empty));
    patchLine('[data-review-pool-empty-text]', projection.below.empty, previous.below.empty, true, true);
    patchLine('[data-review-pool-confirm]', null, null, projection.below.confirm, previous.below.confirm);
    patchLine('[data-review-pool-copies]', null, null, Boolean(projection.below.copies), Boolean(previous.below.copies));
    patchLine('[data-review-pool-copies-text]', projection.below.copies, previous.below.copies, true, true);
    patchLine('[data-subagents-validation]', projection.below.validation, previous.below.validation);

    // Reconcile visible rows in exact roster order while preserving surviving rows and their switches.
    if (!projection.rows.length) {
        const empty = list.firstElementChild;
        if (list.children.length === 1 && empty?.classList?.contains('available-subagents-empty')) {
            if (previous.rows.length || previous.emptyText !== projection.emptyText) {
                empty.textContent = projection.emptyText;
            }
        } else {
            list.innerHTML = `<p class="available-subagents-empty">${escapeHtml(projection.emptyText)}</p>`;
        }
    } else {
        const wantedRows = new Set(projection.rows.map((item) => item.key));
        for (const child of [...list.children]) {
            if (!wantedRows.has(child.dataset?.subagentRow)) child.remove();
        }
        let rowCursor = list.firstElementChild;
        for (const item of projection.rows) {
            let node = existingRows.get(item.key);
            if (!node) {
                const tpl = doc.createElement('template');
                tpl.innerHTML = listHeadHtml(item, renderRowActions);
                node = tpl.content.firstElementChild;
                list.insertBefore(node, rowCursor || null);
            } else {
                const prev = prevRows.get(item.key);
                if (node !== rowCursor) list.insertBefore(node, rowCursor || null);
                else rowCursor = node.nextElementSibling;
                for (const [selector, field] of [
                    ['.settings-head-glyph', 'mark'],
                    ['[data-subagent-identity]', 'line'],
                    ['.ui-row-title', 'title'],
                ]) {
                    if (!prev || prev.facts[field] !== item.facts[field]) {
                        const live = node.querySelector(selector);
                        if (live) live.innerHTML = item.facts[field];
                    }
                }
                if (!prev || prev.facts.name !== item.facts.name) {
                    const sw = node.querySelector('[data-entry-switch]');
                    if (sw) sw.setAttribute('aria-label', `Available: ${item.facts.name}`);
                }
            }
        }
        bindNewRows(list);
    }
    return true;
}
