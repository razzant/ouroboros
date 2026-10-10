/* Widgets list seam (no DOM): the per-card and whole-list change signatures the
   page compares after every `GET /api/widgets`, the keyed patch plan it applies
   to the existing <article> nodes when the list changed, and the list READ
   itself — its shared deadline, its sibling-abort policy and the controller
   lifecycle navigation and disposal cancel through.
   Card order is deliberately NOT part of the list signature — `widget_order`
   is a separate, cheap fact the page applies through the masonry key order,
   never by moving or rebuilding nodes. */

import { WIDGET_REQUEST_TIMEOUT_MS, withWidgetRequestTimeout } from './widget_job.js';

export function widgetKey(tab) {
    return tab.key || `${tab.skill}:${tab.tab_id}`;
}

// JSON with sorted object keys, so two snapshots of one declaration compare
// equal regardless of the serializer's key order.
function stableStringify(value) {
    if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
    if (value && typeof value === 'object') {
        const body = Object.keys(value).sort()
            .map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`)
            .join(',');
        return `{${body}}`;
    }
    return JSON.stringify(value) ?? 'null';
}

/** Everything a card's mount consumes, plus the owning skill's `revision`. */
export function widgetCardSignature(tab) {
    return stableStringify({
        key: widgetKey(tab),
        title: tab.title ?? '',
        icon: tab.icon ?? '',
        span: Number(tab.span || tab.grid_span || 1),
        ws_prefix: tab.ws_prefix ?? '',
        render: tab.render ?? null,
        revision: tab.revision ?? '',
    });
}

/** Order-independent signature of the whole card list. */
export function widgetTabsSignature(tabs) {
    return (Array.isArray(tabs) ? tabs : []).map(widgetCardSignature).sort().join('\n');
}

/** Keyed diff: cards to add, cards to remove, cards whose own entry changed. */
export function planWidgetListPatch(previousTabs, nextTabs) {
    const before = new Map((previousTabs || []).map((tab) => [widgetKey(tab), widgetCardSignature(tab)]));
    const after = new Set((nextTabs || []).map(widgetKey));
    const added = [];
    const changed = [];
    for (const tab of nextTabs || []) {
        const key = widgetKey(tab);
        if (!before.has(key)) added.push(key);
        else if (before.get(key) !== widgetCardSignature(tab)) changed.push(key);
    }
    return { added, changed, removed: [...before.keys()].filter((key) => !after.has(key)) };
}

/**
 * One Widgets-list read: the cards and the owner's card preferences under ONE
 * abort controller and ONE deadline that spans headers AND body of both.
 *
 * Failure asymmetry is deliberate: the cards ARE the page, so a timeout or a
 * list failure aborts its preferences sibling and surfaces; a preferences
 * failure degrades to `null` and the last known order is kept. The deadline
 * still covers a preferences-only stall, because `Promise.all` cannot settle
 * until both sides do.
 *
 * Exported so the deadline is testable against a fake clock as the product
 * composes it, rather than against a copy of it.
 *
 * @returns {Promise<[import('./api_types.js').WidgetsResponse, any]>}
 */
/**
 * The cards list is authoritative for what may be STOPPED (a kept-running frame
 * whose skill left the list), so a reply without an `ui_tabs` array is refused
 * rather than read as "no cards": a truncated or foreign body must never
 * dispose retained work.
 */
export function assertWidgetsList(data) {
    if (!data || typeof data !== 'object' || !Array.isArray(data.ui_tabs)) {
        const error = new Error('malformed widgets list response: ui_tabs is not an array');
        error.code = 'WIDGET_LIST_MALFORMED';
        throw error;
    }
    return data;
}

export function requestWidgetListPayload(client, controller, timeoutMs = WIDGET_REQUEST_TIMEOUT_MS) {
    return withWidgetRequestTimeout((signal) => Promise.all([
        client.widgets({ signal }).then(assertWidgetsList).catch((error) => {
            controller.abort();
            throw error;
        }),
        client.uiPreferences({ signal }).catch(() => null),
    ]), controller, timeoutMs);
}

/** The cards alone, under the same deadline — the kept-running reconcile. */
export function requestWidgetCards(client, controller, timeoutMs = WIDGET_REQUEST_TIMEOUT_MS) {
    return withWidgetRequestTimeout((signal) => client.widgets({ signal }).then(assertWidgetsList), controller, timeoutMs);
}

/**
 * The Widgets page's list-request lifecycle: one controller per read, tracked
 * so navigation, disposal or page hide can cancel whatever is in flight.
 *
 * Owning it here keeps the page module from repeating the same
 * create/track/finally/abort-all shape at four call sites, and makes the
 * cancellation contract testable on its own.
 */
export function widgetListRequests() {
    const controllers = new Set();
    return {
        async run(task) {
            const controller = new AbortController();
            controllers.add(controller);
            try {
                return await task(controller);
            } finally {
                controllers.delete(controller);
            }
        },
        abortAll() {
            controllers.forEach((controller) => controller.abort());
            controllers.clear();
        },
        get size() { return controllers.size; },
    };
}
