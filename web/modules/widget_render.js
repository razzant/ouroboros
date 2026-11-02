// The declarative widget renderer (docs/WIDGET_HOST_PROTOCOL.md): one host-owned component declaration becomes
// markup, never skill-supplied script. Every component root carries its stable identity (`data-widget-component`:
// the author's `id`, else its tree path), which the mounting host reconciles and dispatches by. `ctx` says where the
// markup lives, so the same renderer serves the Widgets page and any other host place:
//   ctx.routePath(route, params) — the same-origin URL of the owning skill's route, '' where there is none;
//   ctx.routePrefix — that skill's route prefix; '' accepts no extension URL from data.
// `view` is the mount's live state: `{ state, status, componentState, formValues, pendingActions, visibleKeys }`
// (a rendered component adds its key to `visibleKeys`). Every host that mounts the markup reconciles it in place with
// `patchWidgetChildren` (a data update never detaches a retained control) and keeps its charts with `syncWidgetCharts`.
import { cleanExtensionRoute } from './api_client.js';
import { chartConfig, formatNumber, getPath, renderChartDataTable, renderTableCell } from './widget_chart.js';
import { applyChartTheme, paintChartSeries } from './theme_palette.js';
import { escapeHtmlAttr as escapeHtml, renderMarkdownSafe } from './utils.js';
import { normalizeTone, renderSafeField } from './ui_helpers.js';

function safeMediaSrc(ctx, spec, state, effectiveTarget = '') {
    const route = spec.route || spec.api_route || '';
    if (route) {
        const params = new URLSearchParams();
        for (const [key, value] of Object.entries(spec.query || {})) {
            params.set(key, String(value ?? ''));
        }
        return ctx.routePath(route, params);
    }
    const value = getPath(state[effectiveTarget || spec.target || 'result'], spec.path || '', spec.src || '');
    const text = String(value || '').trim();
    if (/^data:(image\/(?:png|jpeg|jpg|gif|webp)|audio\/(?:mpeg|wav|ogg)|video\/(?:mp4|webm|ogg));base64,[A-Za-z0-9+/=]+$/i.test(text)) {
        return text;
    }
    if (text.startsWith('/api/extensions/')) {
        try {
            const parsed = new URL(text, window.location.origin);
            const expectedPrefix = ctx.routePrefix || '';
            if (expectedPrefix && parsed.origin === window.location.origin && parsed.pathname.startsWith(expectedPrefix)) {
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

export function filenameFromWidgetUrl(url, fallback = 'download') {
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

export function componentIdentity(component, treePath) {
    const explicitId = String(component?.id || '').trim();
    return explicitId ? `id:${explicitId}` : `path:${treePath}`;
}

export function indexComponentTree(components) {
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

export function renderComponent(ctx, component, view, treePath, inheritedTarget = '') {
    const markup = renderComponentBody(ctx, component, view, treePath, inheritedTarget);
    // The renderer owns these tags; carry its existing identity onto each root
    // without adding a wrapper that would change group/grid composition.
    return markup.replace(/^(\s*)<([a-z][a-z0-9-]*)/, `$1<$2 data-widget-component="${escapeHtml(componentIdentity(component, treePath))}"`);
}

function renderActionFeedback(componentState, key) {
    const feedback = componentState[`feedback:${key}`];
    if (!feedback) return '';
    return `<div class="widget-status ui-status" data-widget-feedback="${escapeHtml(key)}" data-state="${escapeHtml(feedback.status)}" data-tone="${normalizeTone(feedback.status)}" role="status" aria-live="polite">${escapeHtml(feedback.message)}</div>`;
}

function renderComponentBody(ctx, component, view, treePath, inheritedTarget = '') {
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
        const busy = pendingActions.has(key) || Boolean(componentState[`job:${key}`]);
        const disabled = Boolean(component.disabled) || busy;
        const columns = Math.max(1, Math.min(4, Number(component.columns) || 1));
        const fields = (Array.isArray(component.fields) ? component.fields : []).map((field) => renderSafeField(
            field,
            formValues[key] || {},
            { disabled, maxSpan: columns, spanClassPrefix: 'widget-field-span-' },
        )).join('');
        const label = busy ? (component.busy_label || 'Working…') : (component.submit_label || 'Submit');
        const heading = component.title || component.label || '';
        return `<form class="widget-form" data-widget-form="${escapeHtml(key)}" aria-busy="${busy ? 'true' : 'false'}"${heading ? ` aria-label="${escapeHtml(heading)}"` : ''}>${heading ? `<h4>${escapeHtml(heading)}</h4>` : ''}<div class="widget-form-fields widget-grid-cols-${columns}">${fields}</div><button class="btn btn-primary" type="submit"${disabled ? ' disabled' : ''}>${escapeHtml(label)}</button>${renderActionFeedback(componentState, key)}</form>`;
    }
    if (type === 'action') {
        const busy = pendingActions.has(key) || Boolean(componentState[`job:${key}`]);
        const label = busy ? (component.busy_label || 'Working…') : (component.label || 'Run');
        return `<div class="widget-action"><button type="button" class="btn btn-default" data-widget-action="${escapeHtml(key)}"${component.disabled || busy ? ' disabled' : ''}>${escapeHtml(label)}</button>${renderActionFeedback(componentState, key)}</div>`;
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
            renderComponent(ctx, child, view, `${treePath}.components.${idx}`, passiveTarget)
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
            .map((child, idx) => renderComponent(ctx, child, view, `${treePath}.tabs.${active}.components.${idx}`, passiveTarget))
            .join('');
        return `<div class="widget-tabs"><div class="widget-tab-list">${buttons}</div><div class="widget-tab-body">${body || '<div class="muted">No content.</div>'}</div></div>`;
    }
    if (type === 'stream') {
        const current = status[target] || 'idle';
        return `<div class="widget-stream" data-state="${escapeHtml(current)}">${escapeHtml(component[current] || component.label || current)}</div>`;
    }
    if (['image', 'audio', 'video', 'file'].includes(type)) {
        const src = safeMediaSrc(ctx, component, state, target);
        const label = escapeHtml(component.label || component.alt || type);
        if (!src) return `<div class="muted">${label}: no safe media source.</div>`;
        if (type === 'image') return `<figure class="widget-media"><img src="${escapeHtml(src)}" alt="${escapeHtml(component.alt || label)}"><figcaption>${label}</figcaption></figure>`;
        if (type === 'audio') return `<div class="widget-media"><div>${label}</div><audio controls src="${escapeHtml(src)}"></audio></div>`;
        if (type === 'video') return `<div class="widget-media"><div>${label}</div><video controls src="${escapeHtml(src)}"></video></div>`;
        const filename = escapeHtml(component.filename || filenameFromWidgetUrl(src, label || 'download'));
        return `<div class="widget-file"><button class="btn btn-default widget-download" type="button" data-widget-download-url="${escapeHtml(src)}" data-widget-download-filename="${filename}"${pendingActions.has(`download:${key}`) ? ' disabled' : ''}>${label}</button>${renderActionFeedback(componentState, key)}</div>`;
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
        return `<div class="widget-gallery">${items.map((item, idx) => renderComponent(ctx, { ...item, type: item.type || 'image' }, view, `${treePath}.gallery.${idx}`, passiveTarget)).join('')}</div>`;
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
            const busy = pendingActions.has(`kanban:${key}`);
            const empty = cards.length
                ? ''
                : '<div class="widget-kanban-empty">No cards</div>';
            return `<div class="widget-kanban-col${cards.length ? '' : ' is-empty'}" data-widget-kanban-col="${escapeHtml(colKey)}">
                <div class="widget-kanban-col-head"><strong>${escapeHtml(col.label || colKey)}</strong></div>
                ${cards.map((c, idx) => {
                    const cardId = c.id || `${colKey}-${idx}`;
                    const move = moveRoute ? `<label class="widget-kanban-move"><span>Move to</span><select class="ui-control" aria-label="${escapeHtml(`Move to column for ${c.label || c.title || cardId}`)}" data-widget-kanban-move data-widget-kanban-card-id="${escapeHtml(cardId)}"${busy ? ' disabled' : ''}>${columns.map((option) => {
                        const optionKey = option.id || option.label;
                        return `<option value="${escapeHtml(optionKey)}"${String(optionKey) === String(colKey) ? ' selected' : ''}>${escapeHtml(option.label || optionKey)}</option>`;
                    }).join('')}</select></label>` : '';
                    return `<div class="widget-kanban-card" draggable="${busy ? 'false' : 'true'}" data-widget-kanban-card="${escapeHtml(cardId)}"><span>${escapeHtml(c.label || c.title || '—')}</span>${move}</div>`;
                }).join('')}${empty}
            </div>`;
        }).join('');
        return `<div class="widget-kanban" data-widget-kanban-key="${escapeHtml(key)}" data-widget-kanban-route="${escapeHtml(moveRoute || '')}" aria-busy="${pendingActions.has(`kanban:${key}`) ? 'true' : 'false'}">${colHtml}${renderActionFeedback(componentState, key)}</div>`;
    }
    if (type === 'subscription') {
        const children = Array.isArray(component.render) ? component.render : [];
        if (!children.length) return '';
        return `<div class="widget-subscription-render">${children.map((child, idx) => {
            if (!child || typeof child !== 'object') return '';
            return renderComponent(ctx, child, view, `${treePath}.render.${idx}`, target);
        }).join('')}</div>`;
    }
    return '';
}

// Data updates patch the mounted declaration, never detach a retained control
// or its ancestors. Component ids/tree paths already own identity; field and
// kanban keys distinguish editable siblings inside those component roots.
function widgetNodeKey(node) {
    const data = node.dataset || {};
    if (data.widgetComponent) return `component:${data.widgetComponent}`;
    if (data.widgetFeedback) return `feedback:${data.widgetFeedback}`;
    if (data.widgetTabKey) return `tab:${data.widgetTabKey}:${data.widgetTabIdx}`;
    if (data.widgetKanbanCard) return `card:${data.widgetKanbanCard}`;
    if (data.widgetKanbanCol) return `column:${data.widgetKanbanCol}`;
    return node.name ? `field:${node.name}` : '';
}

export function patchWidgetChildren(current, desired) {
    const available = Array.from(current.childNodes);
    const used = new Set();
    const pools = new Map();
    const matchKey = (node) => `${node.nodeType}:${node.nodeName}:${widgetNodeKey(node)}`;
    for (const node of available) {
        const key = matchKey(node);
        if (!pools.has(key)) pools.set(key, { nodes: [], next: 0 });
        pools.get(key).nodes.push(node);
    }
    const pairs = Array.from(desired.childNodes).map((next) => {
        const pool = pools.get(matchKey(next));
        const live = pool?.nodes[pool.next++];
        if (live) used.add(live);
        return [live, next];
    });
    available.filter((node) => !used.has(node)).forEach((node) => node.remove());
    let anchor = current.firstChild;
    for (const [live, next] of pairs) {
        const node = live || next.cloneNode(true);
        if (node !== anchor) current.insertBefore(node, anchor);
        if (live) patchWidgetNode(live, next);
        anchor = node.nextSibling;
    }
}

function patchWidgetNode(current, desired) {
    if (current.nodeType !== 1) {
        if (current.nodeValue !== desired.nodeValue) current.nodeValue = desired.nodeValue;
        return;
    }
    if (current.classList.contains('widget-chart-canvas')) {
        // Chart.js owns its canvas and responsive parent. Only the declaration
        // attributes change here; renderAll updates the existing chart below.
        const canvas = current.querySelector('canvas');
        const next = desired.querySelector('canvas');
        if (canvas && next) {
            canvas.dataset.widgetChartConfig = next.dataset.widgetChartConfig;
            canvas.setAttribute('aria-label', next.getAttribute('aria-label') || '');
            return;
        }
    }
    const field = current.matches('input, textarea, select') && current.closest('[data-widget-form]');
    const preserve = (name) => (field && ['value', 'checked', 'selected'].includes(name))
        || (current.tagName === 'DETAILS' && name === 'open');
    for (const attr of Array.from(current.attributes)) {
        if (!preserve(attr.name) && !desired.hasAttribute(attr.name)) current.removeAttribute(attr.name);
    }
    for (const attr of Array.from(desired.attributes)) {
        if (!preserve(attr.name) && current.getAttribute(attr.name) !== attr.value) current.setAttribute(attr.name, attr.value);
    }
    // The declaration is fixed for this mount. Do not rewrite editable values,
    // textarea children or select options during a composition/native popup.
    // Passwords remain only in their live control, outside saved formValues.
    if (!field) patchWidgetChildren(current, desired);
}

// The mounted charts follow the patched declarations: a canvas whose type and options are unchanged keeps its
// Chart.js instance and takes the new data; a changed one is rebuilt; a canvas that left the mount takes its chart
// with it. `instances` and `shapes` are the mount's own maps (key → chart, key → its type/options shape).
export function syncWidgetCharts(mount, instances, shapes) {
    const mountedChartKeys = new Set();
    mount.querySelectorAll('[data-widget-chart-config]').forEach((canvas) => {
        if (typeof Chart === 'undefined') return;
        const chartKey = canvas.dataset.widgetChartKey || '';
        mountedChartKeys.add(chartKey);
        try {
            const config = JSON.parse(canvas.dataset.widgetChartConfig || '{}');
            const existing = instances.get(chartKey);
            const shape = JSON.stringify({ type: config.type, options: config.options || {} });
            // widget_chart.js is DOM-free by contract, so its serialized config
            // carries neutral colours; the live theme is layered on here, where
            // the document is readable: new data and a new chart land in it.
            paintChartSeries(config.data?.datasets);
            if (existing && existing.canvas === canvas && shapes.get(chartKey) === shape) {
                existing.data = config.data;
                existing.update();
                return;
            }
            if (existing) existing.destroy();
            const chart = new Chart(canvas, config);
            applyChartTheme(chart);
            instances.set(chartKey, chart);
            shapes.set(chartKey, shape);
        } catch (err) {
            console.warn('widgets: chart render failed', err);
        }
    });
    instances.forEach((chart, chartKey) => {
        if (!mountedChartKeys.has(chartKey)) {
            chart.destroy();
            instances.delete(chartKey);
            shapes.delete(chartKey);
        }
    });
}
