import { renderPageHeader } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';
import { applyMasonry } from './masonry.js';
import {
    apiClient,
    apiFetch,
    cleanExtensionRoute,
    extensionRoutePath,
    extensionRoutePrefix,
} from './api_client.js';
import {
    chartColorAlpha,
    escapeHtmlAttr as escapeHtml,
    readThemeTokens,
    renderMarkdownSafe,
} from './utils.js';
import {
    collectSafeFieldValues,
    downloadViaHostBridge,
    normalizeTone,
    renderSafeField,
} from './ui_helpers.js';

function pageTemplate() {
    return `
        <section class="page app-page-glass" id="page-widgets">
            ${renderPageHeader({
                title: 'Widgets',
                icon: PAGE_ICONS.widgets,
                description: 'Reviewed extension UI surfaces live here, separate from the skill catalogue.',
                actionsHtml: '<button id="widgets-refresh" class="btn btn-default btn-sm">Refresh</button>',
            })}
            <div class="widgets-scroll scroll-fade-y">
                <div id="widgets-list" class="widgets-list"></div>
            </div>
        </section>
    `;
}

function renderShell(host, tabs) {
    if (!tabs.length) {
        host.innerHTML = '<div class="muted">No live widgets yet. Review and enable an extension that registers a UI tab.</div>';
        return;
    }
    host.innerHTML = tabs.map((tab) => {
        // Avoid leaking internal "skill:tab_id"; show skill only as needed.
        const title = tab.title || tab.tab_id || tab.skill;
        const subtitle = tab.skill && tab.skill !== title
            ? `<span class="widgets-card-source">from ${escapeHtml(tab.skill)}</span>`
            : '';
        const span = Number(tab.span || tab.grid_span || 1);
        const spanClass = span >= 2 ? ' widgets-card-span-2' : '';
        return `
        <article class="widgets-card${spanClass}" data-widget-key="${escapeHtml(tab.key || `${tab.skill}:${tab.tab_id}`)}">
            <div class="widgets-card-head">
                <div class="widgets-card-title">
                    <strong>${escapeHtml(title)}</strong>
                    ${subtitle}
                </div>
                <button class="widgets-card-drag" type="button" data-widget-reorder-handle title="Move widget: drag or use arrow keys" aria-label="Move widget: drag or use arrow keys">↕</button>
            </div>
            <div class="widgets-card-body" data-widget-mount></div>
        </article>
        `;
    }).join('');
}

function widgetKey(tab) {
    return tab.key || `${tab.skill}:${tab.tab_id}`;
}

function normalizeWidgetOrder(value) {
    if (!Array.isArray(value)) return [];
    const seen = new Set();
    return value
        .map((item) => String(item || '').trim())
        .filter((item) => {
            if (!item || seen.has(item)) return false;
            seen.add(item);
            return true;
        });
}

function sortTabsByWidgetOrder(tabs, order) {
    const rank = new Map(normalizeWidgetOrder(order).map((key, idx) => [key, idx]));
    return tabs.map((tab, originalIndex) => ({ tab, originalIndex })).sort((a, b) => {
        const aRank = rank.has(widgetKey(a.tab)) ? rank.get(widgetKey(a.tab)) : Number.MAX_SAFE_INTEGER;
        const bRank = rank.has(widgetKey(b.tab)) ? rank.get(widgetKey(b.tab)) : Number.MAX_SAFE_INTEGER;
        if (aRank !== bRank) return aRank - bRank;
        return a.originalIndex - b.originalIndex;
    }).map((item) => item.tab);
}

function currentWidgetOrderFromDom(list) {
    return Array.from(list.querySelectorAll('[data-widget-key]'))
        .map((card) => card.dataset.widgetKey || '')
        .filter(Boolean);
}

function bindWidgetCardReorder(list, onOrderChange) {
    if (!list) return;
    let draggedKey = '';
    const clearDragState = () => {
        list.querySelectorAll('.widgets-card.dragging, .widgets-card.drag-over').forEach((card) => {
            card.classList.remove('dragging', 'drag-over');
        });
        draggedKey = '';
    };
    const finishReorder = () => {
        applyMasonry(list);
        onOrderChange(currentWidgetOrderFromDom(list));
    };
    list.querySelectorAll('[data-widget-reorder-handle]').forEach((handle) => {
        const card = handle.closest('[data-widget-key]');
        if (!card) return;
        handle.setAttribute('draggable', 'true');
        handle.addEventListener('dragstart', (event) => {
            draggedKey = card.dataset.widgetKey || '';
            if (!draggedKey) return;
            card.classList.add('dragging');
            if (event.dataTransfer) {
                event.dataTransfer.effectAllowed = 'move';
                event.dataTransfer.setData('text/plain', draggedKey);
            }
        });
        handle.addEventListener('dragend', clearDragState);
        handle.addEventListener('keydown', (event) => {
            let moved = false;
            if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') {
                const previous = card.previousElementSibling;
                if (previous?.classList.contains('widgets-card')) {
                    previous.before(card);
                    moved = true;
                }
            } else if (event.key === 'ArrowDown' || event.key === 'ArrowRight') {
                const next = card.nextElementSibling;
                if (next?.classList.contains('widgets-card')) {
                    next.after(card);
                    moved = true;
                }
            } else if (event.key === 'Home') {
                const first = list.querySelector('.widgets-card');
                if (first && first !== card) {
                    first.before(card);
                    moved = true;
                }
            } else if (event.key === 'End') {
                const cards = list.querySelectorAll('.widgets-card');
                const last = cards[cards.length - 1];
                if (last && last !== card) {
                    last.after(card);
                    moved = true;
                }
            }
            if (!moved) return;
            event.preventDefault();
            clearDragState();
            finishReorder();
            handle.focus();
        });
    });
    list.querySelectorAll('.widgets-card').forEach((card) => {
        card.addEventListener('dragover', (event) => {
            if (!draggedKey || card.dataset.widgetKey === draggedKey) return;
            event.preventDefault();
            card.classList.add('drag-over');
            if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        });
        card.addEventListener('dragleave', () => card.classList.remove('drag-over'));
        card.addEventListener('drop', (event) => {
            if (!draggedKey || card.dataset.widgetKey === draggedKey) return;
            event.preventDefault();
            const dragged = list.querySelector(`[data-widget-key="${CSS.escape(draggedKey)}"]`);
            if (!dragged) return;
            const cards = Array.from(list.querySelectorAll('.widgets-card'));
            const draggedIdx = cards.indexOf(dragged);
            const targetIdx = cards.indexOf(card);
            if (draggedIdx < 0 || targetIdx < 0) return;
            if (draggedIdx < targetIdx) card.after(dragged);
            else card.before(dragged);
            clearDragState();
            finishReorder();
        });
    });
}

function getPath(root, path, fallback = '') {
    if (!path) return root ?? fallback;
    let current = root;
    for (const part of String(path).split('.').filter(Boolean)) {
        if (current == null || typeof current !== 'object') return fallback;
        current = current[part];
    }
    return current ?? fallback;
}

function safeMediaSrc(tab, spec, state, effectiveTarget = '') {
    const route = spec.route || spec.api_route || '';
    if (route) {
        const params = new URLSearchParams();
        for (const [key, value] of Object.entries(spec.query || {})) {
            params.set(key, String(value ?? ''));
        }
        return extensionRoutePath(tab.skill, route, params);
    }
    const value = getPath(state[effectiveTarget || spec.target || 'result'], spec.path || '', spec.src || '');
    const text = String(value || '').trim();
    if (/^data:(image\/(?:png|jpeg|jpg|gif|webp)|audio\/(?:mpeg|wav|ogg)|video\/(?:mp4|webm|ogg));base64,[A-Za-z0-9+/=]+$/i.test(text)) {
        return text;
    }
    if (text.startsWith('/api/extensions/')) {
        try {
            const parsed = new URL(text, window.location.origin);
            const expectedPrefix = extensionRoutePrefix(tab.skill);
            if (parsed.origin === window.location.origin && parsed.pathname.startsWith(expectedPrefix)) {
                return parsed.pathname + parsed.search;
            }
        } catch {
            return '';
        }
    }
    return '';
}

function routePrefixToMediaSpec(routePrefix, value, itemType = 'image') {
    const text = String(value || '').trim();
    const prefix = String(routePrefix || '').trim();
    if (!prefix || !text) return { type: itemType, src: text };
    const [route, queryKey = 'path'] = prefix.split('?', 2);
    const key = queryKey.endsWith('=') ? queryKey.slice(0, -1) : queryKey;
    return {
        type: itemType,
        route,
        query: { [key || 'path']: text },
    };
}

function filenameFromWidgetUrl(url, fallback = 'download') {
    try {
        const parsed = new URL(url, window.location.origin);
        for (const key of ['filename', 'image_id', 'clip_id']) {
            const value = parsed.searchParams.get(key);
            if (value) return value.split('/').pop() || fallback;
        }
        const base = parsed.pathname.split('/').filter(Boolean).pop();
        return base || fallback;
    } catch {
        return fallback;
    }
}

function componentIdentity(component, treePath) {
    const explicitId = String(component?.id || '').trim();
    return explicitId ? `id:${explicitId}` : `path:${treePath}`;
}

function indexComponentTree(components) {
    const entries = [];
    const byKey = new Map();
    const visit = (component, path) => {
        if (!component || typeof component !== 'object') return;
        const key = componentIdentity(component, path);
        if (byKey.has(key)) throw new Error(`duplicate declarative widget component identity: ${key}`);
        const entry = { component, key, path };
        byKey.set(key, entry);
        entries.push(entry);
        if (String(component.type || '') === 'group') {
            (Array.isArray(component.components) ? component.components : []).forEach((child, idx) => {
                visit(child, `${path}.components.${idx}`);
            });
        }
        if (String(component.type || '') === 'tabs') {
            (Array.isArray(component.tabs) ? component.tabs : []).forEach((item, tabIdx) => {
                (Array.isArray(item?.components) ? item.components : []).forEach((child, idx) => {
                    visit(child, `${path}.tabs.${tabIdx}.components.${idx}`);
                });
            });
        }
        if (String(component.type || '') === 'subscription') {
            (Array.isArray(component.render) ? component.render : []).forEach((child, idx) => {
                visit(child, `${path}.render.${idx}`);
            });
        }
    };
    (Array.isArray(components) ? components : []).forEach((component, idx) => visit(component, `components.${idx}`));
    return { entries, byKey };
}

function safeTableHref(value) {
    const raw = String(value || '').trim();
    if (!raw) return '';
    try {
        const parsed = new URL(raw, window.location.origin);
        return ['http:', 'https:'].includes(parsed.protocol) ? parsed.href : '';
    } catch {
        return '';
    }
}

function formatNumber(value, precision) {
    if (value === null || value === undefined || value === '') return '—';
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '—';
    const parsedPrecision = Number(precision);
    if (precision === undefined || precision === null || precision === '' || !Number.isFinite(parsedPrecision)) {
        return numeric.toLocaleString(undefined, { maximumFractionDigits: 12 });
    }
    const digits = Math.max(0, Math.min(12, parsedPrecision));
    return numeric.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function renderTableCell(row, column) {
    const presentation = String(column.presentation || column.format || 'plain');
    const raw = getPath(row, column.path, '');
    if (presentation === 'number') {
        const rendered = formatNumber(raw, column.precision);
        return `${escapeHtml(rendered)}${rendered !== '—' && column.unit ? ` ${escapeHtml(column.unit)}` : ''}`;
    }
    if (presentation === 'status') {
        const label = raw && typeof raw === 'object' ? (raw.label ?? raw.value ?? raw.status ?? '') : raw;
        const toneValue = raw && typeof raw === 'object' ? raw.tone : getPath(row, column.tone_path || '', 'muted');
        const tone = normalizeTone(toneValue);
        return `<span class="widget-table-status" data-tone="${escapeHtml(tone)}">${escapeHtml(label || '—')}</span>`;
    }
    if (presentation === 'link') {
        const rawHref = getPath(row, column.href_path || column.path, '');
        const href = safeTableHref(rawHref);
        const label = column.label_path ? getPath(row, column.label_path, rawHref) : raw;
        if (!href) return escapeHtml(label || rawHref || '—');
        return `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label || href)}</a>`;
    }
    return escapeHtml(raw ?? '');
}

// Widget series colours. Chart.js paints on a canvas and cannot resolve CSS
// variables, so the palette is RESOLVED from the design tokens at chart-BUILD
// time through the shared `readThemeTokens` seam (web/modules/utils.js). No hue
// is copied here, so nothing can drift out of step with web/style.css :root.
// Every name in these lists must be DEFINED in `:root` — aliases resolve
// through the seam fine, an UNDEFINED token resolves to `''` and drops a series.
const CHART_SERIES_TOKENS = ['--accent-light', '--user', '--green', '--amber'];
const CHART_GRID_TOKEN = '--surface-border-soft';
const CHART_FILL_ALPHA = 0.22;

/** `[borderColor, backgroundColor]` per series, from already-resolved tokens. */
function chartPalette(series) {
    return series.map((color) => [color, chartColorAlpha(color, CHART_FILL_ALPHA)]);
}

export function finiteChartValue(value) {
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (typeof value !== 'string' || !value.trim()) return null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}

function chartConfig(component, data) {
    const type = ['line', 'bar'].includes(component.chart_type) ? component.chart_type : 'line';
    const labels = component.labels || getPath(data, component.labels_path || 'labels', []);
    const datasets = component.datasets || getPath(data, component.datasets_path || 'datasets', []);
    const unit = String(component.unit || '');
    // ONE `getComputedStyle` round-trip per chart build for every token this
    // chart needs — grid first, then the series ramp, in list order.
    const [gridColor, ...series] = readThemeTokens([CHART_GRID_TOKEN, ...CHART_SERIES_TOKENS]);
    const palette = chartPalette(series);
    return {
        type,
        data: {
            labels: Array.isArray(labels) ? labels.map((item) => String(item ?? '')) : [],
            datasets: Array.isArray(datasets) ? datasets.map((dataset, idx) => {
                const [borderColor, backgroundColor] = palette[idx % palette.length];
                return {
                    label: String(dataset?.label ?? 'Series'),
                    data: Array.isArray(dataset?.data) ? dataset.data.map(finiteChartValue) : [],
                    borderColor,
                    backgroundColor,
                    spanGaps: false,
                };
            }) : [],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            spanGaps: false,
            plugins: { legend: { display: true } },
            scales: {
                x: { grid: { color: gridColor } },
                y: {
                    grid: { color: gridColor },
                    title: { display: Boolean(unit), text: unit },
                },
            },
        },
    };
}

function renderChartDataTable(config, label, expanded) {
    const labels = config.data.labels || [];
    const datasets = config.data.datasets || [];
    const rows = labels.map((item, idx) => `<tr><th scope="row">${escapeHtml(item)}</th>${datasets.map((dataset) => `<td data-label="${escapeHtml(dataset.label)}">${escapeHtml(dataset.data[idx] ?? '—')}</td>`).join('')}</tr>`).join('');
    return `<details class="widget-chart-data"${expanded ? ' open' : ''}><summary>View ${escapeHtml(label)} data</summary><div class="widget-table-wrap"><table class="widget-table"><thead><tr><th>Label</th>${datasets.map((dataset) => `<th>${escapeHtml(dataset.label)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div></details>`;
}

function renderComponent(tab, component, view, treePath, inheritedTarget = '') {
    const { state, status, componentState, formValues, pendingActions, visibleKeys } = view;
    const type = String(component.type || '');
    const target = component.target || inheritedTarget || 'result';
    const data = state[target] ?? {};
    const key = componentIdentity(component, treePath);
    if (component.condition_key && !getPath(data, component.condition_key, false)) {
        return '';
    }
    visibleKeys.add(key);
    if (type === 'form') {
        const busy = status[target] === 'loading' || pendingActions.has(key);
        const disabled = Boolean(component.disabled) || busy;
        const columns = Math.max(1, Math.min(4, Number(component.columns) || 1));
        const fields = (Array.isArray(component.fields) ? component.fields : []).map((field) => renderSafeField(
            field,
            formValues[key] || {},
            { disabled, maxSpan: columns, spanClassPrefix: 'widget-field-span-' },
        )).join('');
        const label = busy ? (component.busy_label || 'Working…') : (component.submit_label || 'Submit');
        const heading = component.title || component.label || '';
        return `<form class="widget-form" data-widget-form="${escapeHtml(key)}" aria-busy="${busy ? 'true' : 'false'}"${heading ? ` aria-label="${escapeHtml(heading)}"` : ''}>${heading ? `<h4>${escapeHtml(heading)}</h4>` : ''}<div class="widget-form-fields widget-grid-cols-${columns}">${fields}</div><button class="btn btn-primary" type="submit"${disabled ? ' disabled' : ''}>${escapeHtml(label)}</button></form>`;
    }
    if (type === 'action') {
        const busy = status[target] === 'loading' || pendingActions.has(key);
        const label = busy ? (component.busy_label || 'Working…') : (component.label || 'Run');
        return `<button type="button" class="btn btn-default" data-widget-action="${escapeHtml(key)}"${component.disabled || busy ? ' disabled' : ''}>${escapeHtml(label)}</button>`;
    }
    if (type === 'poll') {
        const busy = status[target] === 'loading';
        return `<button type="button" class="btn btn-default" data-widget-poll="${escapeHtml(key)}"${busy ? ' disabled' : ''}>${escapeHtml(busy ? (component.busy_label || 'Polling…') : (component.label || 'Start polling'))}</button>`;
    }
    if (type === 'group') {
        const layout = ['stack', 'grid', 'cluster'].includes(component.layout) ? component.layout : 'stack';
        const columns = Math.max(1, Math.min(4, Number(component.columns) || 2));
        const passiveTarget = inheritedTarget ? target : '';
        const children = (Array.isArray(component.components) ? component.components : []).map((child, idx) => (
            renderComponent(tab, child, view, `${treePath}.components.${idx}`, passiveTarget)
        )).join('');
        const heading = component.title ? `<h4>${escapeHtml(component.title)}</h4>` : '';
        const description = component.description ? `<p>${escapeHtml(component.description)}</p>` : '';
        const columnClass = layout === 'grid' ? ` widget-grid-cols-${columns}` : '';
        return `<section class="widget-group widget-group-${layout}${columnClass}">${heading}${description}<div class="widget-group-components">${children}</div></section>`;
    }
    if (type === 'metric') {
        const raw = Object.prototype.hasOwnProperty.call(component, 'value')
            ? component.value
            : getPath(data, component.path || '', undefined);
        const text = typeof raw === 'string' ? raw.trim() : '';
        const numericValue = text ? Number(text) : Number.NaN;
        const numericText = Number.isFinite(numericValue);
        const nonFiniteText = Boolean(text) && (
            (!Number.isNaN(numericValue) && !Number.isFinite(numericValue))
            || ['nan', 'inf', '+inf', '-inf'].includes(text.toLowerCase())
        );
        const structured = raw !== null && typeof raw === 'object';
        const missing = raw === undefined
            || raw === null
            || (typeof raw === 'string' && !text)
            || (typeof raw === 'number' && !Number.isFinite(raw))
            || nonFiniteText
            || structured;
        const rendered = missing
            ? '—'
            : (typeof raw === 'number' || numericText ? formatNumber(raw, component.precision) : String(raw));
        const tone = normalizeTone(component.tone);
        return `<div class="widget-metric" data-tone="${escapeHtml(tone)}"><span>${escapeHtml(component.label || '')}</span><strong>${escapeHtml(rendered)}${!missing && component.unit ? ` <small>${escapeHtml(component.unit)}</small>` : ''}</strong></div>`;
    }
    if (type === 'callout') {
        const value = Object.prototype.hasOwnProperty.call(component, 'text')
            ? component.text
            : getPath(data, component.path || '', '');
        return `<div class="widget-callout" data-tone="${escapeHtml(normalizeTone(component.tone, 'info'))}">${escapeHtml(value ?? '')}</div>`;
    }
    if (type === 'status') {
        const current = status[target] || 'idle';
        const label = component[current]
            || (current === 'refreshing' ? component.loading : '')
            || current;
        return `<div class="widget-status" data-state="${escapeHtml(current)}">${escapeHtml(label)}</div>`;
    }
    if (type === 'kv') {
        const fields = component.fields || [];
        const rows = fields.map((field) => {
            const label = escapeHtml(field.label || field.path || '');
            const value = getPath(data, field.path, '—');
            return `<div class="widget-kv-row"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`;
        }).join('');
        return `<div class="widget-kv">${rows || '<div class="muted">No data.</div>'}</div>`;
    }
    if (type === 'key_value') {
        const rows = getPath(data, component.items_key || component.path || '', []);
        if (!Array.isArray(rows) || !rows.length) return '';
        return `<div class="widget-kv">${rows.map((row) => `<div class="widget-kv-row"><span>${escapeHtml(row?.key || row?.label || '')}</span><strong>${escapeHtml(row?.value ?? '')}</strong></div>`).join('')}</div>`;
    }
    if (type === 'table') {
        const rows = getPath(data, component.path || '', []);
        const cols = component.columns || [];
        if (!Array.isArray(rows)) return '<div class="muted">No rows.</div>';
        return `<div class="widget-table-wrap"><table class="widget-table"><thead><tr>${cols.map((c) => `<th>${escapeHtml(c.label || c.path || '')}</th>`).join('')}</tr></thead><tbody>${rows.map((row) => `<tr>${cols.map((c) => `<td data-label="${escapeHtml(c.label || c.path || '')}">${renderTableCell(row, c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
    }
    if (type === 'markdown') {
        const value = component.text ?? getPath(data, component.path || '', '');
        return `<div class="widget-markdown ui-rich-content">${renderMarkdownSafe(value)}</div>`;
    }
    if (type === 'json') {
        const value = component.path ? getPath(data, component.path, {}) : data;
        return `<details class="widget-json"><summary>${escapeHtml(component.label || 'JSON')}</summary><pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre></details>`;
    }
    if (type === 'code') {
        const value = component.text ?? getPath(data, component.path || '', '');
        const label = component.label ? `<div class="widget-code-label">${escapeHtml(component.label)}</div>` : '';
        return `<div class="widget-code">${label}<pre><code>${escapeHtml(value)}</code></pre></div>`;
    }
    if (type === 'chart') {
        const config = chartConfig(component, component.path ? getPath(data, component.path, {}) : data);
        const label = String(component.aria_label || component.label || component.title || 'Chart');
        const chartAvailable = typeof Chart !== 'undefined';
        const canvas = chartAvailable ? `<div class="widget-chart-canvas"><canvas role="img" aria-label="${escapeHtml(label)}" data-widget-chart-key="${escapeHtml(key)}" data-widget-chart-config="${escapeHtml(JSON.stringify(config))}"></canvas></div>` : '';
        return `<div class="widget-chart${chartAvailable ? '' : ' widget-chart-fallback'}">${canvas}${renderChartDataTable(config, label, !chartAvailable)}</div>`;
    }
    if (type === 'tabs') {
        const tabs = Array.isArray(component.tabs) ? component.tabs : [];
        const stateKey = `tab:${key}`;
        const active = Math.max(0, Math.min(Number(componentState[stateKey] || 0), Math.max(tabs.length - 1, 0)));
        const buttons = tabs.map((item, idx) => (
            `<button type="button" class="widget-tab-btn ${idx === active ? 'active' : ''}" data-widget-tab-key="${escapeHtml(stateKey)}" data-widget-tab-idx="${idx}" aria-selected="${idx === active ? 'true' : 'false'}">${escapeHtml(item.label || `Tab ${idx + 1}`)}</button>`
        )).join('');
        const activeTab = tabs[active] || {};
        const passiveTarget = inheritedTarget ? target : '';
        const body = (activeTab.components || [])
            .map((child, idx) => renderComponent(tab, child, view, `${treePath}.tabs.${active}.components.${idx}`, passiveTarget))
            .join('');
        return `<div class="widget-tabs"><div class="widget-tab-list">${buttons}</div><div class="widget-tab-body">${body || '<div class="muted">No content.</div>'}</div></div>`;
    }
    if (type === 'stream') {
        const current = status[target] || 'idle';
        return `<div class="widget-stream" data-state="${escapeHtml(current)}">${escapeHtml(component[current] || component.label || current)}</div>`;
    }
    if (['image', 'audio', 'video', 'file'].includes(type)) {
        const src = safeMediaSrc(tab, component, state, target);
        const label = escapeHtml(component.label || component.alt || type);
        if (!src) return `<div class="muted">${label}: no safe media source.</div>`;
        if (type === 'image') return `<figure class="widget-media"><img src="${escapeHtml(src)}" alt="${escapeHtml(component.alt || label)}"><figcaption>${label}</figcaption></figure>`;
        if (type === 'audio') return `<div class="widget-media"><div>${label}</div><audio controls src="${escapeHtml(src)}"></audio></div>`;
        if (type === 'video') return `<div class="widget-media"><div>${label}</div><video controls src="${escapeHtml(src)}"></video></div>`;
        const filename = escapeHtml(component.filename || filenameFromWidgetUrl(src, label || 'download'));
        return `<button class="btn btn-default widget-download" type="button" data-widget-download-url="${escapeHtml(src)}" data-widget-download-filename="${filename}">${label}</button>`;
    }
    if (type === 'gallery') {
        let items = component.items || getPath(data, component.path || component.items_key || '', []);
        if (!Array.isArray(items)) return '<div class="muted">No media items.</div>';
        if (component.items_key && component.route_prefix) {
            items = items.map((item) => routePrefixToMediaSpec(
                component.route_prefix,
                typeof item === 'object' ? (item.path || item.src || item.url || '') : item,
                component.item_type || 'image',
            ));
        }
        const passiveTarget = inheritedTarget ? target : '';
        return `<div class="widget-gallery">${items.map((item, idx) => renderComponent(tab, { ...item, type: item.type || 'image' }, view, `${treePath}.gallery.${idx}`, passiveTarget)).join('')}</div>`;
    }
    if (type === 'progress') {
        const value = Number(getPath(data, component.path || component.value_key || 'progress', 0));
        const bounded = Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0;
        const label = component.label_key ? getPath(data, component.label_key, '') : '';
        return `<div class="widget-progress"><progress max="100" value="${bounded}"></progress><span>${bounded}%${label ? ` · ${escapeHtml(label)}` : ''}</span></div>`;
    }
    // Host-owned map renderer; no skill-supplied JS reaches the SPA origin.
    if (type === 'map') {
        const markers = Array.isArray(component.markers) ? component.markers : [];
        const list = markers.length
            ? `<ul class="widget-map-list">${markers.map((m) => `<li><strong>${escapeHtml(m.label || `${m.lat}, ${m.lon}`)}</strong>${m.popup ? ` — ${escapeHtml(m.popup)}` : ''}</li>`).join('')}</ul>`
            : '<div class="muted">No map markers.</div>';
        return `<div class="widget-map" data-widget-map-config="${escapeHtml(JSON.stringify({ tiles_url: component.tiles_url, markers }))}">${list}</div>`;
    }
    if (type === 'calendar') {
        const items = Array.isArray(component.items) ? component.items : (Array.isArray(getPath(data, component.path || '', [])) ? getPath(data, component.path || '', []) : []);
        if (!items.length) return '<div class="muted">No calendar entries.</div>';
        const rows = items.map((item) => `<li class="widget-calendar-row"><strong>${escapeHtml(item.label || '—')}</strong>${item.start ? ` <span class="muted">${escapeHtml(item.start)}${item.end ? ' → ' + escapeHtml(item.end) : ''}</span>` : ''}${item.row ? ` <em>${escapeHtml(item.row)}</em>` : ''}</li>`).join('');
        return `<div class="widget-calendar"><ul class="widget-calendar-list">${rows}</ul></div>`;
    }
    if (type === 'kanban') {
        const columns = Array.isArray(component.columns) ? component.columns : [];
        if (!columns.length) return '<div class="muted">Kanban has no columns.</div>';
        const rawMoveRoute = component.on_move?.route || '';
        const moveRoute = cleanExtensionRoute(rawMoveRoute) ? rawMoveRoute : '';
        const cardsByCol = new Map();
        for (const col of columns) cardsByCol.set(col.id || col.label, []);
        const cardsList = Array.isArray(component.cards) ? component.cards : (Array.isArray(getPath(data, component.path || '', [])) ? getPath(data, component.path || '', []) : []);
        for (const card of cardsList) {
            const colKey = card.column || card.col || columns[0]?.id || columns[0]?.label;
            if (!cardsByCol.has(colKey)) cardsByCol.set(colKey, []);
            cardsByCol.get(colKey).push(card);
        }
        const colHtml = columns.map((col) => {
            const colKey = col.id || col.label;
            const cards = cardsByCol.get(colKey) || [];
            const busy = status[target] === 'loading' || pendingActions.has(`kanban:${key}`);
            const empty = cards.length
                ? ''
                : '<div class="widget-kanban-empty">No cards</div>';
            return `<div class="widget-kanban-col${cards.length ? '' : ' is-empty'}" data-widget-kanban-col="${escapeHtml(colKey)}">
                <div class="widget-kanban-col-head"><strong>${escapeHtml(col.label || colKey)}</strong></div>
                ${cards.map((c, idx) => {
                    const cardId = c.id || `${colKey}-${idx}`;
                    const move = moveRoute ? `<label class="widget-kanban-move"><span>Move to</span><select data-widget-kanban-move data-widget-kanban-card-id="${escapeHtml(cardId)}"${busy ? ' disabled' : ''}>${columns.map((option) => {
                        const optionKey = option.id || option.label;
                        return `<option value="${escapeHtml(optionKey)}"${String(optionKey) === String(colKey) ? ' selected' : ''}>${escapeHtml(option.label || optionKey)}</option>`;
                    }).join('')}</select></label>` : '';
                    return `<div class="widget-kanban-card" draggable="${busy ? 'false' : 'true'}" data-widget-kanban-card="${escapeHtml(cardId)}"><span>${escapeHtml(c.label || c.title || '—')}</span>${move}</div>`;
                }).join('')}${empty}
            </div>`;
        }).join('');
        const error = status[target] === 'error' ? `<div class="widget-status" data-state="error">${escapeHtml(state[target]?.error || 'Move failed.')}</div>` : '';
        return `<div class="widget-kanban" data-widget-kanban-key="${escapeHtml(key)}" data-widget-kanban-route="${escapeHtml(moveRoute || '')}" aria-busy="${status[target] === 'loading' ? 'true' : 'false'}">${colHtml}${error}</div>`;
    }
    if (type === 'subscription') {
        const children = Array.isArray(component.render) ? component.render : [];
        if (!children.length) return '';
        return `<div class="widget-subscription-render">${children.map((child, idx) => {
            if (!child || typeof child !== 'object') return '';
            return renderComponent(tab, child, view, `${treePath}.render.${idx}`, target);
        }).join('')}</div>`;
    }
    return '';
}

const widgetDisposers = new Map();
const widgetMessageHandlers = new Set();
const widgetSessionState = new Map();
let widgetsWsBridgeBound = false;

function boundedNumber(value, fallback, min, max) {
    const parsed = Number(value);
    const safe = Number.isFinite(parsed) ? parsed : fallback;
    return Math.max(min, Math.min(safe, max));
}

async function callWidgetRoute(tab, spec, values, signal) {
    const method = String(spec.method || 'GET').toUpperCase();
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(values || {})) {
        params.set(key, String(value ?? ''));
    }
    const noBody = method === 'GET' || method === 'HEAD';
    const url = extensionRoutePath(tab.skill, spec.route || spec.api_route, noBody ? params : null);
    if (!url) throw new Error('invalid widget route');
    const init = noBody
        ? { method, signal }
        : {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(values || {}),
            signal,
        };
    const resp = await apiFetch(url, init);
    const contentType = resp.headers.get('content-type') || '';
    const data = contentType.includes('application/json')
        ? await resp.json().catch(() => ({}))
        : { text: await resp.text() };
    if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
}

async function mountDeclarativeWidget(mount, tab, render) {
    const components = Array.isArray(render.components) ? render.components : [];
    const { entries: componentEntries, byKey: componentByKey } = indexComponentTree(components);
    const componentSpec = (key) => componentByKey.get(key)?.component || null;
    const persistenceKey = tab.key || `${tab.skill}:${tab.tab_id}`;
    const saved = widgetSessionState.get(persistenceKey) || {};
    const state = { ...(saved.state || {}) };
    const status = { ...(saved.status || {}) };
    const formValues = { ...(saved.formValues || {}) };
    const componentState = { ...(saved.componentState || {}) };
    const timers = new Set();
    const controllers = new Set();
    const chartInstances = new Map();
    const chartShapes = new Map();
    const eventSources = new Map();
    const activePolls = new Set();
    const activeJobs = new Set();
    const autoStarted = new Set();
    const messageHandlers = new Set();
    const subscribed = new Set();
    const pendingActions = new Set();
    let disposed = false;

    const downloadWidgetFile = async (url, filename) => {
        const resolvedUrl = new URL(url, window.location.origin);
        const expectedPrefix = extensionRoutePrefix(tab.skill);
        if (resolvedUrl.origin !== window.location.origin || !resolvedUrl.pathname.startsWith(expectedPrefix)) {
            throw new Error('download URL is outside this widget extension');
        }
        const safeName = filenameFromWidgetUrl(resolvedUrl.toString(), filename || 'download');
        await downloadViaHostBridge(resolvedUrl.pathname + resolvedUrl.search, safeName, { fetchOptions: { credentials: 'include' } });
    };

    const schedule = (fn, delay) => {
        if (disposed) return null;
        const timer = setTimeout(() => {
            timers.delete(timer);
            fn();
        }, delay);
        timers.add(timer);
        return timer;
    };
    const dispose = () => {
        widgetSessionState.set(persistenceKey, {
            state: { ...state },
            status: { ...status },
            formValues: { ...formValues },
            componentState: { ...componentState },
        });
        disposed = true;
        controllers.forEach((controller) => controller.abort());
        controllers.clear();
        chartInstances.forEach((chart) => chart.destroy());
        chartInstances.clear();
        chartShapes.clear();
        eventSources.forEach((source) => source.close());
        eventSources.clear();
        timers.forEach((timer) => clearTimeout(timer));
        timers.clear();
        activePolls.clear();
        activeJobs.clear();
        pendingActions.clear();
        messageHandlers.forEach((handler) => widgetMessageHandlers.delete(handler));
        messageHandlers.clear();
        subscribed.clear();
    };
    const callRoute = async (spec, values) => {
        if (disposed) throw new Error('widget disposed');
        const controller = new AbortController();
        controllers.add(controller);
        try {
            return await callWidgetRoute(tab, spec, values, controller.signal);
        } finally {
            controllers.delete(controller);
        }
    };
    const rememberFormValues = () => {
        mount.querySelectorAll('[data-widget-form]').forEach((form) => {
            const key = form.dataset.widgetForm || '';
            const spec = componentSpec(key);
            if (!spec) return;
            formValues[key] = collectSafeFieldValues(form, spec.fields || [], { includePasswords: false });
        });
    };
    const startPoll = (key) => {
        if (disposed || activePolls.has(key)) return;
        const spec = componentSpec(key);
        if (!spec) return;
        const target = spec.target || 'result';
        const maxTicks = boundedNumber(spec.max_ticks, 20, 1, 100);
        const intervalMs = boundedNumber(spec.interval_ms, 2000, 1000, 30000);
        let ticks = 0;
        activePolls.add(key);
        const poll = async () => {
            if (disposed) return;
            ticks += 1;
            // SWR (v6.71.0): 'loading' only when there is nothing to show yet;
            // a background refetch keeps the content and shows a thin indicator.
            const hadData = state[target] !== undefined;
            status[target] = hadData ? 'refreshing' : 'loading';
            renderAll();
            try {
                state[target] = await callRoute(spec, {});
                if (disposed) return;
                status[target] = 'success';
            } catch (err) {
                // SWR: a failed background refetch keeps the stale content —
                // only the status reports the error; a failed FIRST load still
                // surfaces the error payload (there is nothing else to show).
                if (!hadData) state[target] = { error: err.message || String(err) };
                status[target] = 'error';
            }
            const stopValue = getPath(state[target], spec.stop_path || '', undefined);
            if (ticks < maxTicks && String(stopValue) !== String(spec.stop_value ?? 'done')) {
                schedule(poll, intervalMs);
            } else {
                activePolls.delete(key);
            }
            renderAll();
        };
        poll();
    };
    // A job's progress is fed by two writers — the status poll and the WS
    // subscription. Keep the percent monotonic per job so a stale poll tick or an
    // out-of-order WS event can never move the bar backward. Resets when the job id
    // changes. The percent key is whatever the progress component(s) read
    // (`value_key`), so this works regardless of the skill's field name.
    const progressValueKeys = (() => {
        const keys = [];
        for (const { component: c } of componentEntries) {
            if (String(c?.type || '') !== 'progress') continue;
            const k = String(c.path || c.value_key || 'progress');
            if (k && !k.includes('.') && !keys.includes(k)) keys.push(k);
        }
        return keys.length ? keys : ['progress_pct', 'progress'];
    })();
    const clampMonotonicProgress = (target, jobId, nextObj) => {
        if (!nextObj || typeof nextObj !== 'object') return nextObj;
        let pctKey = '';
        let pct;
        for (const k of progressValueKeys) {
            if (typeof nextObj[k] === 'number' && Number.isFinite(nextObj[k])) { pct = nextObj[k]; pctKey = k; break; }
        }
        if (pctKey === '') return nextObj;
        const stateKey = `progress-clamp:${target}`;
        const prev = componentState[stateKey];
        if (prev && prev.jobId === jobId && prev.pct > pct) {
            nextObj[pctKey] = prev.pct;
            return nextObj;
        }
        componentState[stateKey] = { jobId, pct: nextObj[pctKey] };
        return nextObj;
    };

    const startJobPoll = (key, jobId) => {
        if (disposed || !jobId || activeJobs.has(key)) return;
        const spec = componentSpec(key);
        if (!spec) return;
        const target = spec.target || 'result';
        const statusRoute = spec.status_route || spec.job_status_route || 'status';
        const intervalMs = boundedNumber(spec.interval_ms, 2000, 1000, 30000);
        const maxTicks = boundedNumber(spec.max_ticks, 240, 1, 1000);
        let ticks = 0;
        activeJobs.add(key);
        componentState[`job:${key}`] = { job_id: jobId, status_route: statusRoute };
        const pollJob = async () => {
            if (disposed) return;
            ticks += 1;
            try {
                const data = await callRoute({ route: statusRoute, method: 'GET' }, { job_id: jobId });
                if (disposed) return;
                const currentStatus = String(data.status || data.state || '').toLowerCase();
                if (currentStatus === 'done' || currentStatus === 'succeeded' || currentStatus === 'success') {
                    state[target] = data.result && typeof data.result === 'object' ? data.result : data;
                    status[target] = 'success';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    renderAll();
                    return;
                }
                if (currentStatus === 'error' || currentStatus === 'failed') {
                    state[target] = { error: data.error || 'job failed' };
                    status[target] = 'error';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    renderAll();
                    return;
                }
                // Merge the whole flat status payload so the renderer's value_key
                // (e.g. `progress_pct`) is surfaced — cherry-picking `data.progress`
                // dropped the percent and broke the poll fallback when WS hiccuped.
                state[target] = clampMonotonicProgress(target, jobId, {
                    ...(state[target] || {}),
                    ...data,
                    job_id: jobId,
                });
                status[target] = 'loading';
                renderAll();
                if (ticks < maxTicks) {
                    schedule(pollJob, intervalMs);
                } else {
                    state[target] = { error: 'job timed out waiting for result' };
                    status[target] = 'error';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    renderAll();
                }
            } catch (err) {
                state[target] = { error: err.message || String(err) };
                status[target] = 'error';
                delete componentState[`job:${key}`];
                activeJobs.delete(key);
                renderAll();
            }
        };
        pollJob();
    };
    const renderAll = () => {
        if (disposed) return;
        rememberFormValues();
        widgetSessionState.set(persistenceKey, {
            state: { ...state },
            status: { ...status },
            formValues: { ...formValues },
            componentState: { ...componentState },
        });
        // In-place chart updates (v6.71.0): keep live canvases so a data-only
        // re-render updates chart.data instead of destroy/recreate flicker.
        const liveChartCanvases = new Map();
        mount.querySelectorAll('canvas[data-widget-chart-key]').forEach((liveCanvas) => {
            liveChartCanvases.set(liveCanvas.dataset.widgetChartKey || '', liveCanvas);
        });
        const visibleKeys = new Set();
        const view = { state, status, componentState, formValues, pendingActions, visibleKeys };
        mount.innerHTML = components.map((component, idx) => renderComponent(tab, component, view, `components.${idx}`)).join('');
        mount.querySelectorAll('[data-widget-form]').forEach((form) => {
            form.addEventListener('submit', async (event) => {
                event.preventDefault();
                const key = form.dataset.widgetForm || '';
                const spec = componentSpec(key);
                if (!spec || spec.disabled || pendingActions.has(key)) return;
                const target = spec.target || 'result';
                const values = collectSafeFieldValues(form, spec.fields || []);
                pendingActions.add(key);
                status[target] = 'loading';
                renderAll();
                try {
                    const data = await callRoute(spec, values);
                    if (disposed) return;
                    if (spec.job === true || spec.mode === 'job') {
                        const jobId = data.job_id || data.id;
                        if (!jobId) throw new Error('job response missing job_id');
                        state[target] = { job_id: jobId, message: data.message || 'Job started.' };
                        status[target] = 'loading';
                        startJobPoll(key, jobId);
                    } else {
                        state[target] = data;
                        status[target] = 'success';
                    }
                } catch (err) {
                    state[target] = { error: err.message || String(err) };
                    status[target] = 'error';
                } finally {
                    pendingActions.delete(key);
                    renderAll();
                }
            });
        });
        mount.querySelectorAll('[data-widget-action]').forEach((button) => {
            button.addEventListener('click', async () => {
                const key = button.dataset.widgetAction || '';
                const spec = componentSpec(key);
                if (!spec || spec.disabled || pendingActions.has(key)) return;
                const target = spec.target || 'result';
                pendingActions.add(key);
                status[target] = 'loading';
                renderAll();
                try {
                    const data = await callRoute(spec, spec.body || {});
                    if (disposed) return;
                    if (spec.job === true || spec.mode === 'job') {
                        const jobId = data.job_id || data.id;
                        if (!jobId) throw new Error('job response missing job_id');
                        state[target] = { job_id: jobId, message: data.message || 'Job started.' };
                        status[target] = 'loading';
                        startJobPoll(key, jobId);
                    } else {
                        state[target] = data;
                        status[target] = 'success';
                    }
                } catch (err) {
                    state[target] = { error: err.message || String(err) };
                    status[target] = 'error';
                } finally {
                    pendingActions.delete(key);
                    renderAll();
                }
            });
        });
        mount.querySelectorAll('[data-widget-poll]').forEach((button) => {
            button.addEventListener('click', () => startPoll(button.dataset.widgetPoll || ''));
        });
        mount.querySelectorAll('[data-widget-tab-key]').forEach((button) => {
            button.addEventListener('click', () => {
                componentState[button.dataset.widgetTabKey] = Number(button.dataset.widgetTabIdx || 0);
                renderAll();
            });
        });
        mount.querySelectorAll('[data-widget-download-url]').forEach((button) => {
            button.addEventListener('click', async (event) => {
                event.preventDefault();
                button.disabled = true;
                try {
                    await downloadWidgetFile(button.dataset.widgetDownloadUrl || '', button.dataset.widgetDownloadFilename || 'download');
                } catch (err) {
                    state.download = { error: err.message || String(err) };
                    status.download = 'error';
                    renderAll();
                } finally {
                    button.disabled = false;
                }
            });
        });
        mount.querySelectorAll('[data-widget-kanban-key]').forEach((board) => {
            const key = board.dataset.widgetKanbanKey || '';
            const spec = componentSpec(key);
            if (!spec) return;
            const target = spec.target || 'result';
            const pendingKey = `kanban:${key}`;
            let draggedCardId = '';
            const moveCard = async (cardId, columnId) => {
                const route = board.dataset.widgetKanbanRoute || '';
                if (!route || !cardId || !columnId || pendingActions.has(pendingKey)) return;
                pendingActions.add(pendingKey);
                status[target] = 'loading';
                renderAll();
                try {
                    state[target] = await callRoute(
                        { route, method: spec.on_move?.method || 'POST' },
                        { card_id: cardId, column_id: columnId },
                    );
                    if (disposed) return;
                    status[target] = 'success';
                } catch (err) {
                    state[target] = { error: err.message || String(err) };
                    status[target] = 'error';
                } finally {
                    pendingActions.delete(pendingKey);
                    renderAll();
                }
            };
            board.querySelectorAll('[data-widget-kanban-card]').forEach((card) => {
                card.addEventListener('dragstart', (event) => {
                    if (pendingActions.has(pendingKey)) {
                        event.preventDefault();
                        return;
                    }
                    draggedCardId = card.dataset.widgetKanbanCard || '';
                    if (event.dataTransfer) {
                        event.dataTransfer.effectAllowed = 'move';
                        event.dataTransfer.setData('text/plain', draggedCardId);
                    }
                });
            });
            board.querySelectorAll('[data-widget-kanban-col]').forEach((column) => {
                column.addEventListener('dragover', (event) => {
                    if (!board.dataset.widgetKanbanRoute) return;
                    event.preventDefault();
                    if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
                });
                column.addEventListener('drop', (event) => {
                    if (!board.dataset.widgetKanbanRoute) return;
                    event.preventDefault();
                    const cardId = event.dataTransfer?.getData('text/plain') || draggedCardId;
                    const columnId = column.dataset.widgetKanbanCol || '';
                    moveCard(cardId, columnId);
                });
            });
            board.querySelectorAll('[data-widget-kanban-move]').forEach((select) => {
                select.addEventListener('change', () => {
                    moveCard(select.dataset.widgetKanbanCardId || '', select.value || '');
                });
            });
        });
        const mountedChartKeys = new Set();
        mount.querySelectorAll('[data-widget-chart-config]').forEach((canvas) => {
            if (typeof Chart === 'undefined') return;
            const chartKey = canvas.dataset.widgetChartKey || '';
            mountedChartKeys.add(chartKey);
            try {
                const config = JSON.parse(canvas.dataset.widgetChartConfig || '{}');
                const existing = chartInstances.get(chartKey);
                const liveCanvas = liveChartCanvases.get(chartKey);
                const shape = JSON.stringify({ type: config.type, options: config.options || {} });
                const liveWrap = liveCanvas ? liveCanvas.parentElement : null;
                const newWrap = canvas.parentElement;
                if (existing && liveCanvas && existing.canvas === liveCanvas && chartShapes.get(chartKey) === shape
                        && liveWrap && newWrap && liveWrap.classList.contains('widget-chart-canvas')
                        && newWrap.classList.contains('widget-chart-canvas')) {
                    // Same chart shape: adopt the live WRAPPER (Chart.js observes
                    // the canvas's parent for responsive resize — adopting the bare
                    // canvas would leave the ResizeObserver on a detached node) and
                    // update data in place — no destroy/recreate flicker on poll
                    // ticks. Mirror the fresh config attributes so the live DOM
                    // never lies about the rendered chart.
                    newWrap.replaceWith(liveWrap);
                    liveCanvas.dataset.widgetChartConfig = canvas.dataset.widgetChartConfig;
                    liveCanvas.setAttribute('aria-label', canvas.getAttribute('aria-label') || '');
                    existing.data = config.data;
                    existing.update();
                    return;
                }
                if (existing) existing.destroy();
                chartInstances.set(chartKey, new Chart(canvas, config));
                chartShapes.set(chartKey, shape);
            } catch (err) {
                console.warn('widgets: chart render failed', err);
            }
        });
        chartInstances.forEach((chart, chartKey) => {
            if (!mountedChartKeys.has(chartKey)) {
                chart.destroy();
                chartInstances.delete(chartKey);
                chartShapes.delete(chartKey);
            }
        });
        visibleKeys.forEach((key) => {
            const component = componentSpec(key);
            if (!component) return;
            if (String(component.type || '') === 'poll' && component.auto_start === true && !autoStarted.has(key)) {
                autoStarted.add(key);
                queueMicrotask(() => startPoll(key));
            }
            if (component.job === true || component.mode === 'job') {
                const savedJob = componentState[`job:${key}`];
                const jobId = savedJob && savedJob.job_id;
                if (jobId && status[component.target || 'result'] === 'loading') {
                    queueMicrotask(() => startJobPoll(key, jobId));
                }
            }
            if (String(component.type || '') !== 'stream' || eventSources.has(key)) return;
            const url = extensionRoutePath(tab.skill, component.route || component.api_route, new URLSearchParams());
            if (!url || typeof EventSource === 'undefined') return;
            const target = component.target || 'result';
            const source = new EventSource(url);
            eventSources.set(key, source);
            status[target] = 'loading';
            source.onmessage = (event) => {
                if (disposed) return;
                try {
                    state[target] = JSON.parse(event.data);
                } catch {
                    state[target] = { text: event.data || '' };
                }
                status[target] = 'success';
                renderAll();
            };
            source.onerror = () => {
                if (disposed) return;
                status[target] = 'error';
                renderAll();
            };
        });
        visibleKeys.forEach((key) => {
            const component = componentSpec(key);
            if (!component || String(component.type || '') !== 'subscription' || subscribed.has(key)) return;
            const event = String(component.event || component.message_type || '').trim();
            const prefix = String(tab.ws_prefix || '').trim();
            if (!event || !prefix) return;
            const expectedType = `${prefix}${event}`;
            const target = component.target || 'result';
            const handler = (msg) => {
                if (disposed || msg?.type !== expectedType) return;
                const data = msg.data || {};
                // Same monotonic guard as the poll writer: an out-of-order WS event
                // must not rewind the bar.
                state[target] = clampMonotonicProgress(target, data.job_id || '', { ...data });
                status[target] = 'success';
                renderAll();
            };
            subscribed.add(key);
            messageHandlers.add(handler);
            widgetMessageHandlers.add(handler);
        });
    };
    renderAll();
    return dispose;
}

async function mountTab(card, tab) {
    const mount = card.querySelector('[data-widget-mount]');
    const render = tab.render || {};
    if (!mount) return;
    if (render.kind === 'iframe' && render.route) {
        const src = extensionRoutePath(tab.skill, render.route);
        if (!src) throw new Error('invalid widget iframe route');
        mount.innerHTML = `<iframe class="widgets-frame" sandbox="" src="${src}"></iframe>`;
        return;
    }
    if (render.kind === 'declarative') {
        return mountDeclarativeWidget(mount, tab, render);
    }
    if (render.kind === 'module' && render.entry) {
        // Reviewed JS runs in an opaque iframe; parent fetch bridge only allows
        // this skill's extension route prefix, preserving route IO without cookies.
        const entryName = String(render.entry).replace(/[^A-Za-z0-9._-]/g, '');
        const entryUrl = `${extensionRoutePrefix(tab.skill)}module/${encodeURIComponent(entryName)}`;
        const resp = await apiFetch(entryUrl, { cache: 'no-store' });
        const moduleSource = await resp.text();
        if (!resp.ok) {
            mount.innerHTML = `<div class="skills-load-error">module load failed: ${escapeHtml(moduleSource || `HTTP ${resp.status}`)}</div>`;
            return;
        }
        const expectedPrefix = extensionRoutePrefix(tab.skill);
        const nonce = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        const csp = [
            "default-src 'none'",
            "script-src 'unsafe-inline'",
            "style-src 'unsafe-inline'",
            "img-src data:",
        ].join('; ');
        const escapeScript = (value) => String(value || '')
            .replace(/<\/script/gi, '<\\/script')
            .replace(/<!--/g, '<\\!--');
        const bridge = `
            (() => {
                const nonce = ${JSON.stringify(nonce)};
                let seq = 0;
                const pending = new Map();
                window.addEventListener('message', (event) => {
                    const msg = event.data || {};
                    if (msg.type !== 'ouro-widget-fetch-result' || msg.nonce !== nonce) return;
                    const item = pending.get(msg.id);
                    if (!item) return;
                    pending.delete(msg.id);
                    if (msg.error) {
                        item.reject(new Error(msg.error));
                        return;
                    }
                    item.resolve(new Response(msg.body || '', {
                        status: msg.status || 200,
                        headers: msg.headers || {},
                    }));
                });
                window.fetch = (url, init = {}) => {
                    const id = ++seq;
                    return new Promise((resolve, reject) => {
                        pending.set(id, { resolve, reject });
                        window.parent.postMessage({
                            type: 'ouro-widget-fetch',
                            nonce,
                            id,
                            url: String(url || ''),
                            init: {
                                method: init.method || 'GET',
                                headers: init.headers || {},
                                body: init.body || null,
                            },
                        }, '*');
                    });
                };
                window.OuroborosWidget = { fetch: window.fetch };
            })();
        `;
        const srcdoc = `<!doctype html><html><head><meta http-equiv="Content-Security-Policy" content="${csp}"></head><body><div id="root"></div><script>${bridge}</script><script>${escapeScript(moduleSource)}</script></body></html>`;
        mount.innerHTML = `<iframe class="widgets-frame" sandbox="allow-scripts" srcdoc="${escapeHtml(srcdoc)}"></iframe>`;
        const iframe = mount.querySelector('iframe');
        const onMessage = async (event) => {
            if (event.source !== iframe.contentWindow) return;
            const msg = event.data || {};
            if (msg.type !== 'ouro-widget-fetch' || msg.nonce !== nonce) return;
            try {
                const parsed = new URL(String(msg.url || ''), window.location.origin);
                if (parsed.origin !== window.location.origin || !parsed.pathname.startsWith(expectedPrefix)) {
                    throw new Error('module widget fetch outside extension route prefix');
                }
                const r = await apiFetch(parsed.pathname + parsed.search, {
                    method: String(msg.init?.method || 'GET').toUpperCase(),
                    headers: msg.init?.headers || {},
                    body: msg.init?.body || undefined,
                    credentials: 'same-origin',
                });
                const body = await r.text();
                iframe.contentWindow?.postMessage({
                    type: 'ouro-widget-fetch-result',
                    nonce,
                    id: msg.id,
                    status: r.status,
                    headers: { 'content-type': r.headers.get('content-type') || '' },
                    body,
                }, '*');
            } catch (err) {
                iframe.contentWindow?.postMessage({
                    type: 'ouro-widget-fetch-result',
                    nonce,
                    id: msg.id,
                    error: err.message || String(err),
                }, '*');
            }
        };
        window.addEventListener('message', onMessage);
        return () => window.removeEventListener('message', onMessage);
    }
    mount.innerHTML = `<div class="muted">Widget render kind <code>${escapeHtml(render.kind || 'unknown')}</code> is not supported yet.</div>`;
    return null;
}

function disposeMountedWidgets() {
    widgetDisposers.forEach((dispose) => {
        try {
            dispose();
        } catch (err) {
            console.warn('widgets: dispose failed', err);
        }
    });
    widgetDisposers.clear();
}

async function mountTrackedTab(card, tab) {
    const key = tab.key || `${tab.skill}:${tab.tab_id}`;
    const existing = widgetDisposers.get(key);
    if (existing) {
        existing();
        widgetDisposers.delete(key);
    }
    const dispose = await mountTab(card, tab);
    if (typeof dispose === 'function') {
        widgetDisposers.set(key, dispose);
        return;
    }
}

export function initWidgets(ctx = {}) {
    const page = document.createElement('div');
    page.innerHTML = pageTemplate();
    document.getElementById('content').appendChild(page.firstElementChild);
    const list = document.getElementById('widgets-list');
    const refreshBtn = document.getElementById('widgets-refresh');
    let renderGeneration = 0;
    let widgetsVisible = false;
    let widgetsMounted = false;
    // Last good payload keeps revisits and slow refreshes from blanking the page.
    let lastTabs = null;
    let uiPreferences = { widget_order: [], nested_subagents_expanded: false };
    if (ctx.ws && !widgetsWsBridgeBound) {
        widgetsWsBridgeBound = true;
        ctx.ws.on('message', (msg) => {
            widgetMessageHandlers.forEach((handler) => handler(msg));
        });
    }

    async function render(force = false) {
        const generation = ++renderGeneration;
        widgetsVisible = true;
        if (widgetsMounted && !force) return;
        refreshBtn.disabled = true;
        refreshBtn.classList.add('is-loading');
        disposeMountedWidgets();
        if (lastTabs) {
            renderShell(list, lastTabs);
            bindWidgetCardReorder(list, persistWidgetOrder);
            applyMasonry(list);
        } else {
            list.innerHTML = '<div class="muted">Loading widgets…</div>';
        }
        try {
            const [data, prefs] = await Promise.all([
                apiClient.extensions(),
                apiClient.uiPreferences().catch(() => null),
            ]);
            if (!widgetsVisible || generation !== renderGeneration) return;
            if (prefs) {
                uiPreferences = {
                    widget_order: normalizeWidgetOrder(prefs.widget_order),
                    nested_subagents_expanded: prefs.nested_subagents_expanded === true,
                };
            }
            const tabs = sortTabsByWidgetOrder(
                Array.isArray(data.live?.ui_tabs) ? data.live.ui_tabs : [],
                uiPreferences.widget_order,
            );
            lastTabs = tabs;
            renderShell(list, tabs);
            bindWidgetCardReorder(list, persistWidgetOrder);
            applyMasonry(list);
            widgetsMounted = true;
            for (const tab of tabs) {
                if (!widgetsVisible || generation !== renderGeneration) return;
                const key = widgetKey(tab);
                const card = list.querySelector(`[data-widget-key="${CSS.escape(key)}"]`);
                if (!card) continue;
                try {
                    await mountTrackedTab(card, tab);
                    applyMasonry(list);
                } catch (err) {
                    const mount = card.querySelector('[data-widget-mount]');
                    if (mount) mount.innerHTML = `<div class="skills-load-error">widget failed: ${escapeHtml(err.message || err)}</div>`;
                    applyMasonry(list);
                }
            }
            applyMasonry(list);
        } catch (err) {
            if (!widgetsVisible || generation !== renderGeneration) return;
            // Preserve cached widgets on transient fetch errors.
            if (!lastTabs) {
                list.innerHTML = `<div class="skills-load-error">Failed to load widgets: ${escapeHtml(err.message || err)}</div>`;
            }
            widgetsMounted = false;
        } finally {
            if (widgetsVisible && generation === renderGeneration) {
                refreshBtn.disabled = false;
                refreshBtn.classList.remove('is-loading');
            }
        }
    }

    function persistWidgetOrder(order) {
        const normalized = normalizeWidgetOrder(order);
        uiPreferences = { ...uiPreferences, widget_order: normalized };
        if (lastTabs) {
            lastTabs = sortTabsByWidgetOrder(lastTabs, normalized);
        }
        apiClient.saveUiPreferences({ widget_order: normalized }).catch((err) => {
            console.warn('Failed to save widget order', err);
        });
    }

    refreshBtn.addEventListener('click', () => render(true));
    window.addEventListener('ouro:page-shown', (event) => {
        if (event.detail?.page === 'widgets') {
            render();
        } else {
            // Hide stops stale paints; next render reuses lastTabs for instant repaint.
            widgetsVisible = false;
            widgetsMounted = false;
            disposeMountedWidgets();
        }
    });
}
