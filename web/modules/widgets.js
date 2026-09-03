import { renderPageHeader } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';
import { applyMasonry } from './masonry.js';
import {
    classifyWidgetJobStatus,
    isRetryableWidgetError,
    readWidgetJobStatus,
    boundedNumber,
    withWidgetRequestTimeout,
} from './widget_job.js';
import { chartConfig, formatNumber, getPath, renderChartDataTable, renderTableCell } from './widget_chart.js';
import { mountModuleWidget, mountRouteIframeWidget } from './widget_module.js';
import { planWidgetListPatch, widgetKey, widgetTabsSignature } from './widget_list.js';
import {
    bindWidgetCardMenus,
    effectiveStartMode,
    isFramedWidget,
    isRetainedWidget,
    renderWidgetCardControls,
    renderWidgetFacade,
    syncWidgetCardControls,
    WIDGET_START_MODES,
    withWidgetStartMode,
} from './widget_card.js';
import { bindWidgetCardReorder, normalizeWidgetOrder, sortTabsByWidgetOrder } from './widget_reorder.js';
import {
    apiClient,
    apiFetch,
    cleanExtensionRoute,
    extensionRoutePath,
    extensionRoutePrefix,
} from './api_client.js';
import {
    escapeHtmlAttr as escapeHtml,
    renderMarkdownSafe,
} from './utils.js';
import {
    collectSafeFieldValues,
    downloadViaHostBridge,
    normalizeTone,
    renderSafeField,
} from './ui_helpers.js';

export {
    WIDGET_FRAME_BORDER_RESERVE,
    WIDGET_FRAME_DEFAULT_HEIGHT,
    WIDGET_FRAME_MAX_HEIGHT,
} from './widget_module.js';

function pageTemplate() {
    return `
        <section class="page app-page-glass" id="page-widgets">
            ${renderPageHeader({
                title: 'Widgets',
                icon: PAGE_ICONS.widgets,
                description: 'Reviewed extension UI surfaces live here, separate from the skill catalogue.',
            })}
            <div class="widgets-scroll scroll-fade-y">
                <div id="widgets-list" class="widgets-list"></div>
            </div>
        </section>
    `;
}

function renderCardHtml(tab) {
    // Avoid leaking internal "skill:tab_id"; show skill only as needed.
    const title = tab.title || tab.tab_id || tab.skill;
    const subtitle = tab.skill && tab.skill !== title
        ? `<span class="widgets-card-source">from ${escapeHtml(tab.skill)}</span>`
        : '';
    const span = Number(tab.span || tab.grid_span || 1);
    const spanClass = span >= 2 ? ' widgets-card-span-2' : '';
    return `
        <article class="widgets-card${spanClass}" data-widget-key="${escapeHtml(widgetKey(tab))}">
            <div class="widgets-card-head">
                <div class="widgets-card-title">
                    <strong>${escapeHtml(title)}</strong>
                    ${subtitle}
                </div>
                <div class="widgets-card-controls">
                    ${renderWidgetCardControls(tab)}
                    <button class="widgets-card-drag" type="button" data-widget-reorder-handle title="Move widget: drag or use arrow keys" aria-label="Move widget: drag or use arrow keys">↕</button>
                </div>
            </div>
            <div class="widgets-card-body" data-widget-mount></div>
        </article>
        `;
}

// Full paint — only for a list that has no cards yet.
function renderShell(host, tabs) {
    if (!tabs.length) {
        host.innerHTML = '<div class="muted">No live widgets yet. Review and enable an extension that registers a UI tab.</div>';
        return;
    }
    host.innerHTML = tabs.map(renderCardHtml).join('');
}

function createCardElement(tab) {
    const template = document.createElement('template');
    template.innerHTML = renderCardHtml(tab).trim();
    return template.content.firstElementChild;
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
// key → settle promise of an ordered stop still waiting for the child's
// acknowledgement (its frame is still in the card until then).
const widgetDisposing = new Map();
// key → the mount in flight for it, `{card, isCurrent, promise}`: one per key;
// a second request for the same card joins it (`mountTrackedTab`).
const widgetMounting = new Map();
const widgetMountControllers = new Set();
const widgetMessageHandlers = new Set();
const widgetSessionState = new Map();
// Owner pressed Stop on this card in this page session: re-entering Widgets
// shows its facade instead of auto-starting it again. Start and a launch-policy
// change to Auto / Keep running clear it. Never persisted, so a window reload
// forgets it.
const stoppedByOwner = new Set();
let widgetsWsBridgeBound = false;

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
    try {
        const resp = await apiFetch(url, init);
        const contentType = resp.headers.get('content-type') || '';
        const data = contentType.includes('application/json')
            ? await resp.json().catch(() => ({}))
            : { text: await resp.text() };
        if (!resp.ok || data?.error || data === null) {
            const error = new Error(data?.error || `HTTP ${resp.status}`);
            error.status = resp.status;
            error.retryable = resp.status === 408 || resp.status === 429 || (resp.status >= 500 && resp.status <= 599);
            throw error;
        }
        return data;
    } catch (error) {
        if (error?.name === 'TypeError' && typeof error.retryable !== 'boolean') error.retryable = true;
        throw error;
    }
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
            return await withWidgetRequestTimeout(
                (signal) => callWidgetRoute(tab, spec, values, signal),
                controller,
            );
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
                if (!data || typeof data !== 'object' || Array.isArray(data)) {
                    state[target] = { error: 'invalid job status response' };
                    status[target] = 'error';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    renderAll();
                    return;
                }
                const currentStatus = readWidgetJobStatus(data);
                const statusKind = classifyWidgetJobStatus(currentStatus);
                if (statusKind === 'success') {
                    state[target] = data.result && typeof data.result === 'object' ? data.result : data;
                    status[target] = 'success';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    renderAll();
                    return;
                }
                if (statusKind === 'failure') {
                    state[target] = { error: data.error || 'job failed' };
                    status[target] = 'error';
                    delete componentState[`job:${key}`];
                    activeJobs.delete(key);
                    renderAll();
                    return;
                }
                if (statusKind === 'invalid') {
                    state[target] = { error: 'invalid job status response' };
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
                if (disposed) return;
                if (isRetryableWidgetError(err) && ticks < maxTicks) {
                    // Keep the durable job id and any useful progress while a
                    // transient transport/server failure is retried.
                    status[target] = 'loading';
                    renderAll();
                    schedule(pollJob, intervalMs);
                    return;
                }
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

async function mountTab(card, tab, mountSignal = null) {
    const mount = card.querySelector('[data-widget-mount]');
    const render = tab.render || {};
    if (!mount) return;
    if (render.kind === 'iframe' && render.route) {
        return mountRouteIframeWidget(mount, tab, render);
    }
    if (render.kind === 'declarative') {
        return mountDeclarativeWidget(mount, tab, render);
    }
    if (render.kind === 'module' && render.entry) {
        return mountModuleWidget(mount, tab, render, mountSignal, widgetMessageHandlers);
    }
    mount.innerHTML = `<div class="muted">Widget render kind <code>${escapeHtml(render.kind || 'unknown')}</code> is not supported yet.</div>`;
    return null;
}

// A framed disposer settles asynchronously (dispose → child ack, ≤ 1 s); track
// it by key so a remount of the same key waits for it — never two frames per key.
function trackSettling(key, result) {
    if (!result || typeof result.then !== 'function') return null;
    const settling = result
        .catch((err) => console.warn('widgets: dispose failed', err))
        .finally(() => {
            if (widgetDisposing.get(key) === settling) widgetDisposing.delete(key);
        });
    widgetDisposing.set(key, settling);
    return settling;
}

// Returns the settle promise of an ordered stop, or null when nothing is left to
// wait for (declarative and route-iframe disposers finish synchronously).
function disposeWidgetByKey(key) {
    const dispose = widgetDisposers.get(key);
    if (!dispose) return widgetDisposing.get(key) || null;
    widgetDisposers.delete(key);
    try {
        return trackSettling(key, dispose());
    } catch (err) {
        console.warn('widgets: dispose failed', err);
        return null;
    }
}

// Issue the stop for every mounted card except the keys `keep` answers true for
// (the frames the owner keeps running while Widgets is hidden; a caller that
// passes nothing stops them too); resolves once every ordered stop in flight
// has settled (they run in parallel, each bounded by the ack timeout).
function disposeMountedWidgets(keep = null) {
    widgetMountControllers.forEach((controller) => controller.abort());
    widgetMountControllers.clear();
    Array.from(widgetDisposers.keys()).forEach((key) => {
        if (!keep?.(key)) disposeWidgetByKey(key);
    });
    return Promise.allSettled(Array.from(widgetDisposing.values()));
}

// A card whose skill left the live list: its stop is still the ordered one, so
// the node (and the frame in it) stays until the acknowledgement or the timeout,
// marked so the keyed patch treats it as already gone.
function liveCardFor(list, key) {
    return list.querySelector(`[data-widget-key="${CSS.escape(key)}"]:not([data-widget-removed])`);
}

// A card whose frame is stopping in order stays in the document, marked and in
// the `stopping` state, until the acknowledgement or the timeout — both the
// vanished card and the changed card (a revision or render change while its
// frame runs) leave this way; the frame is never torn down in the same turn as
// the dispose message. The keyed patch treats a marked card as already gone.
function retireCard(card, settling) {
    card.setAttribute('data-widget-removed', '');
    syncWidgetCardControls(card, 'stopping');
    settling.then(() => card.remove());
}

// Keyed patch over the existing <article> nodes: vanished cards go (their mounted
// work disposed first, their session state evicted), cards whose own entry
// changed are replaced — a running one retires first while its fresh card is
// inserted beside it and mounts once the stop settled (`mountTrackedTab` waits
// on the same settle promise) — new cards are appended, every other card keeps
// its DOM node. No node ever moves: the visible order is the masonry key order,
// and a moved <iframe> would reload. The list's masonry relayouts on the mutation.
function patchWidgetCards(list, previousTabs, nextTabs) {
    const plan = planWidgetListPatch(previousTabs, nextTabs);
    for (const key of plan.removed) {
        const card = liveCardFor(list, key);
        const settling = disposeWidgetByKey(key);
        widgetSessionState.delete(key);
        stoppedByOwner.delete(key);
        if (!card) continue;
        if (settling) retireCard(card, settling);
        else card.remove();
    }
    for (const tab of nextTabs) {
        const key = widgetKey(tab);
        const card = liveCardFor(list, key);
        if (card && !plan.changed.includes(key)) continue;
        const settling = disposeWidgetByKey(key);
        const fresh = createCardElement(tab);
        if (!card) list.append(fresh);
        else if (!settling) card.replaceWith(fresh);
        else {
            card.after(fresh);
            retireCard(card, settling);
        }
    }
}

// One mount in flight per key: a second request for the same card while it is
// `starting` (the policy menu's Auto / Keep running) joins the mount under way
// instead of racing it for the body and the disposer. A request for another
// card node (the fresh card of a changed entry) or from a later page generation
// (leave → return inside the ack window) never joins a mount that bails as
// stale: it waits for that mount to finish, then runs its own `mountTabOnce`,
// which re-checks `isCurrent()` and the node's connection before it mounts.
function mountTrackedTab(card, tab, isCurrent = () => true) {
    const key = widgetKey(tab);
    const inFlight = widgetMounting.get(key);
    if (inFlight && inFlight.card === card && inFlight.isCurrent()) return inFlight.promise;
    const started = inFlight
        ? inFlight.promise.catch(() => {}).then(() => mountTabOnce(card, tab, key, isCurrent))
        : mountTabOnce(card, tab, key, isCurrent);
    const mounting = started.finally(() => {
        if (widgetMounting.get(key)?.promise === mounting) widgetMounting.delete(key);
    });
    widgetMounting.set(key, { card, isCurrent, promise: mounting });
    return mounting;
}

async function mountTabOnce(card, tab, key, isCurrent) {
    // One frame per key: a stop still awaiting its acknowledgement finishes first.
    const settling = disposeWidgetByKey(key);
    if (settling) await settling;
    if (!isCurrent() || !card.isConnected) return;
    const mountController = new AbortController();
    widgetMountControllers.add(mountController);
    try {
        const dispose = await mountTab(card, tab, mountController.signal);
        if (typeof dispose === 'function') {
            // Stale (page left, list rebuilt, card replaced meanwhile): stop it in
            // order instead of registering a frame nobody can reach.
            if (!isCurrent() || !card.isConnected) {
                trackSettling(key, dispose());
                return;
            }
            widgetDisposers.set(key, dispose);
        }
    } finally {
        widgetMountControllers.delete(mountController);
    }
}

export function initWidgets(ctx = {}) {
    const page = document.createElement('div');
    page.innerHTML = pageTemplate();
    document.getElementById('content').appendChild(page.firstElementChild);
    const list = document.getElementById('widgets-list');
    let renderGeneration = 0;
    let widgetsVisible = false;
    let widgetsMounted = false;
    // Last good payload keeps revisits and slow refreshes from blanking the page;
    // its order-independent signature decides whether a fetched list touches the DOM.
    let lastTabs = null;
    let lastSignature = '';
    // Generation of the list sync in flight (0 = none). A reconcile trigger landing
    // mid-sync marks the list dirty and the running sync loops once more.
    let activeSync = 0;
    let listDirty = false;
    let uiPreferences = { widget_order: [], widget_start_mode: {}, nested_subagents_expanded: false };
    if (ctx.ws && !widgetsWsBridgeBound) {
        widgetsWsBridgeBound = true;
        ctx.ws.on('message', (msg) => {
            widgetMessageHandlers.forEach((handler) => handler(msg));
        });
        // List reconcile triggers besides page entry: a skill lifecycle change, and
        // every (re)connect — events may have been missed while offline, or the
        // server restarted with the same SHA (no SPA reload then). No polling.
        ctx.ws.on('extension_lifecycle', reconcileWidgetList);
        ctx.ws.on('open', reconcileWidgetList);
    }

    const hasCards = () => Boolean(list.querySelector('[data-widget-key]:not([data-widget-removed])'));
    const isCurrentFor = (generation) => () => widgetsVisible && generation === renderGeneration;
    const tabByKey = (key) => (lastTabs || []).find((tab) => widgetKey(tab) === key) || null;
    // A framed card the owner keeps running (effective policy `retain`) holds
    // its frame while Widgets is hidden; every other mounted card is stopped.
    const retainsWhileHidden = (key) => isRetainedWidget(tabByKey(key), uiPreferences);
    const keptRunning = () => Array.from(widgetDisposers.keys()).filter(retainsWhileHidden);
    // The complete visible key order (`lastTabs` already carries `widget_order`):
    // masonry packs the cards in it; no DOM node is ever moved for it.
    const currentWidgetOrder = () => (lastTabs || []).map(widgetKey);
    const relayout = () => applyMasonry(list, { order: currentWidgetOrder() });

    function paintShell(tabs) {
        renderShell(list, tabs);
        bindWidgetCardReorder(list, currentWidgetOrder, persistWidgetOrder);
        relayout();
    }

    // Page entry: the shell is on screen before the first await. Leaving
    // disposed the mounted work but kept the cards, so an entry reuses them and
    // only mounts into them again. A window reload is the only hard reset there
    // is; nothing in the page rebuilds every card behind the owner's back.
    async function render() {
        const generation = ++renderGeneration;
        widgetsVisible = true;
        if (widgetsMounted) return;
        if (!lastTabs) {
            list.innerHTML = '<div class="muted">Loading widgets…</div>';
        } else if (!hasCards()) {
            paintShell(lastTabs);
        } else {
            relayout();
        }
        await syncWidgets(generation);
    }

    // WS-driven reconcile: visible → sync now (or mark dirty while one runs);
    // hidden → dirty flag, consumed by the next entry's unconditional fetch —
    // plus the force-stop of kept-running frames whose skill left the list,
    // whether or not a sync was still in flight when the owner left.
    function reconcileWidgetList() {
        if (widgetsVisible && !activeSync) {
            syncWidgets(renderGeneration);
            return;
        }
        listDirty = true;
        if (!widgetsVisible) stopVanishedRetainedWidgets();
    }

    // Lifecycle event while Widgets is hidden and frames are kept running: a
    // skill that left the live list (disable / unload / delete) must not keep
    // its frame alive until the next visit. Stop those frames now, in order;
    // the card itself goes with the next entry's sync (the list is dirty).
    async function stopVanishedRetainedWidgets() {
        const kept = keptRunning();
        if (!kept.length) return;
        let data;
        try {
            data = await apiClient.widgets();
        } catch {
            return;
        }
        if (widgetsVisible) return;
        const live = new Set((Array.isArray(data.ui_tabs) ? data.ui_tabs : []).map(widgetKey));
        kept.filter((key) => !live.has(key)).forEach(disposeWidgetByKey);
    }

    // One list sync: GET /api/widgets (+ preferences), compare signatures, patch
    // cards by key, then mount every card without a live mount. Repeats while a
    // trigger marked the list dirty mid-flight.
    async function syncWidgets(generation) {
        const isCurrent = isCurrentFor(generation);
        activeSync = generation;
        try {
            do {
                listDirty = false;
                const [data, prefs] = await Promise.all([
                    apiClient.widgets(),
                    apiClient.uiPreferences().catch(() => null),
                ]);
                if (!isCurrent()) return;
                if (prefs) {
                    uiPreferences = {
                        widget_order: normalizeWidgetOrder(prefs.widget_order),
                        widget_start_mode: prefs.widget_start_mode && typeof prefs.widget_start_mode === 'object'
                            ? prefs.widget_start_mode
                            : {},
                        nested_subagents_expanded: prefs.nested_subagents_expanded === true,
                    };
                }
                const tabs = sortTabsByWidgetOrder(
                    Array.isArray(data.ui_tabs) ? data.ui_tabs : [],
                    uiPreferences.widget_order,
                );
                const signature = widgetTabsSignature(tabs);
                if (hasCards() && tabs.length) {
                    // Same signature: no card node is added, removed, replaced or
                    // moved (the sync still reconciles controls and layout below).
                    if (signature !== lastSignature) patchWidgetCards(list, lastTabs, tabs);
                } else {
                    // Rebuilding the shell destroys frames, so the ordered stops
                    // still in flight get their acknowledgement window first.
                    await disposeMountedWidgets();
                    if (!isCurrent()) return;
                    // Same eviction the keyed patch performs for a vanished card,
                    // for the transition the patch never sees (the last card
                    // leaving, or the first list arriving): a key that is gone must
                    // not leave its declarative session state or the owner's
                    // page-session Stop behind, or re-enabling the skill would
                    // restore values the owner never re-entered and keep a card
                    // suppressed. It runs after disposal, because a declarative
                    // disposer writes that snapshot on its way out.
                    const live = new Set(tabs.map(widgetKey));
                    for (const key of (lastTabs || []).map(widgetKey)) {
                        if (live.has(key)) continue;
                        widgetSessionState.delete(key);
                        stoppedByOwner.delete(key);
                    }
                    renderShell(list, tabs);
                }
                lastTabs = tabs;
                lastSignature = signature;
                bindWidgetCardReorder(list, currentWidgetOrder, persistWidgetOrder);
                relayout();
                widgetsMounted = true;
                for (const tab of tabs) {
                    if (!isCurrent()) return;
                    const key = widgetKey(tab);
                    const card = liveCardFor(list, key);
                    if (!card) continue;
                    if (widgetDisposers.has(key)) {
                        if (isFramedWidget(tab)) syncWidgetCardControls(card, 'running', effectiveStartMode(tab, uiPreferences));
                        continue;
                    }
                    if (isFramedWidget(tab) && !startsOnShow(tab)) await settleStopped(card, tab);
                    else await startWidget(card, tab, isCurrent);
                    relayout();
                }
                relayout();
            } while (listDirty && isCurrent());
        } catch (err) {
            if (!isCurrent()) return;
            // Preserve cached widgets on transient fetch errors.
            if (!lastTabs) {
                list.innerHTML = `<div class="skills-load-error">Failed to load widgets: ${escapeHtml(err.message || err)}</div>`;
            }
            widgetsMounted = false;
        } finally {
            if (activeSync === generation) activeSync = 0;
        }
    }

    // A reorder (handle drag / keys) hands over the next key order: remember it,
    // re-sort the last good list, relayout in place, persist. No node moves.
    function persistWidgetOrder(order) {
        const normalized = normalizeWidgetOrder(order);
        uiPreferences = { ...uiPreferences, widget_order: normalized };
        if (lastTabs) {
            lastTabs = sortTabsByWidgetOrder(lastTabs, normalized);
        }
        relayout();
        apiClient.saveUiPreferences({ widget_order: normalized }).catch((err) => {
            console.warn('Failed to save widget order', err);
        });
    }

    // Framed card, effective policy `auto` or `retain` (both start on show; only
    // `retain` survives leaving), and the owner has not stopped it this session.
    const startsOnShow = (tab) => effectiveStartMode(tab, uiPreferences) !== 'manual'
        && !stoppedByOwner.has(widgetKey(tab));

    // Mount one card and keep its head controls truthful: the facade stands in
    // while it starts (idempotent — never over a settling frame), a failed mount
    // leaves the error in the body and Start available for a retry, and a card
    // the owner left meanwhile ends stopped with its facade, never a stale Stop.
    async function startWidget(card, tab, isCurrent) {
        const mount = card.querySelector('[data-widget-mount]');
        if (isFramedWidget(tab)) renderWidgetFacade(mount, tab);
        syncWidgetCardControls(card, 'starting', effectiveStartMode(tab, uiPreferences));
        try {
            await mountTrackedTab(card, tab, isCurrent);
        } catch (err) {
            if (isCurrent() && mount) mount.innerHTML = `<div class="skills-load-error">widget failed: ${escapeHtml(err.message || err)}</div>`;
        }
        if (isCurrent()) syncWidgetCardControls(card, widgetDisposers.has(widgetKey(tab)) ? 'running' : 'stopped', effectiveStartMode(tab, uiPreferences));
        else if (isFramedWidget(tab)) await settleStopped(card, tab);
    }

    // A framed card that is not to run now: let an ordered stop still in flight
    // finish (its frame is still in the card), and let a mount under way for
    // the key (another card node, or a stale one about to bail) decide the body
    // first, then show the facade — unless the key ended up running. The card
    // never ends with an empty body behind a mount that bailed.
    async function settleStopped(card, tab) {
        const key = widgetKey(tab);
        const mode = effectiveStartMode(tab, uiPreferences);
        if (widgetDisposing.has(key)) {
            syncWidgetCardControls(card, 'stopping', mode);
            await widgetDisposing.get(key);
        }
        if (widgetMounting.has(key)) await widgetMounting.get(key).promise.catch(() => {});
        if (widgetDisposers.has(key) || !card.isConnected) return;
        renderWidgetFacade(card.querySelector('[data-widget-mount]'), tab);
        syncWidgetCardControls(card, 'stopped', mode);
    }

    async function stopWidgetByOwner(card, tab) {
        stoppedByOwner.add(widgetKey(tab));
        syncWidgetCardControls(card, 'stopping', effectiveStartMode(tab, uiPreferences));
        await disposeWidgetByKey(widgetKey(tab));
        await settleStopped(card, tab);
        relayout();
    }

    async function startWidgetByOwner(card, tab) {
        stoppedByOwner.delete(widgetKey(tab));
        await startWidget(card, tab, isCurrentFor(renderGeneration));
        relayout();
    }

    // Launch-policy writes are a whole-map replace through the preferences API:
    // read the stored map first (a stale in-memory copy never drops another
    // card's choice), merge, POST. One at a time — two quick changes chain
    // behind each other, so the second reads the first's map and loses nothing.
    let startModeWrites = Promise.resolve();
    async function persistWidgetStartMode(key, mode) {
        let current = uiPreferences.widget_start_mode;
        try {
            const stored = await apiClient.uiPreferences();
            if (stored?.widget_start_mode && typeof stored.widget_start_mode === 'object') current = stored.widget_start_mode;
        } catch { /* fall back to the in-memory map */ }
        const next = withWidgetStartMode(current, key, mode);
        uiPreferences = { ...uiPreferences, widget_start_mode: next };
        await apiClient.saveUiPreferences({ widget_start_mode: next }).catch((err) => {
            console.warn('Failed to save widget start mode', err);
        });
    }

    // Launch-policy change from the card menu, applied once its write landed —
    // Auto / Keep running start a stopped card now, Manual changes nothing until Start.
    async function setWidgetStartMode(key, mode) {
        const tab = tabByKey(key);
        if (!tab || !WIDGET_START_MODES.includes(mode)) return;
        const write = startModeWrites.then(() => persistWidgetStartMode(key, mode));
        startModeWrites = write.catch(() => {});
        await write;
        const card = liveCardFor(list, key);
        if (!card) return;
        syncWidgetCardControls(card, widgetDisposers.has(key) ? 'running' : 'stopped', mode);
        if (mode !== 'manual' && !widgetDisposers.has(key) && widgetsVisible) await startWidgetByOwner(card, tab);
    }

    bindWidgetCardMenus(list, setWidgetStartMode);
    list.addEventListener('click', (event) => {
        const power = event.target.closest('[data-widget-power]');
        if (!power || power.disabled) return;
        const card = power.closest('[data-widget-key]');
        const tab = tabByKey(card?.dataset.widgetKey || '');
        if (!card || !tab) return;
        if (widgetDisposers.has(widgetKey(tab))) stopWidgetByOwner(card, tab);
        else startWidgetByOwner(card, tab);
    });

    window.addEventListener('ouro:page-shown', (event) => {
        if (event.detail?.page === 'widgets') {
            render();
        } else {
            // Leaving disposes the mounted work — except the frames the owner
            // keeps running, which stay mounted in the hidden page — and stops
            // stale paints; the cards stay in the DOM so the next entry mounts
            // into them instead of rebuilding. A stopped framed card ends with
            // its facade and Start once its stop settled, so a hidden or
            // returned card never shows a stale Stop / Running over an empty body.
            widgetsVisible = false;
            widgetsMounted = false;
            const stopping = Array.from(widgetDisposers.keys()).filter((key) => !retainsWhileHidden(key));
            disposeMountedWidgets(retainsWhileHidden);
            for (const key of stopping) {
                const card = liveCardFor(list, key);
                const tab = tabByKey(key);
                if (card && tab && isFramedWidget(tab)) settleStopped(card, tab);
            }
        }
    });
}
