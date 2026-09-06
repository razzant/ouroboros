/* Pure Widgets list helpers (no DOM): the per-card and whole-list change
   signatures the page compares after every `GET /api/widgets`, and the keyed
   patch plan it applies to the existing <article> nodes when the list changed.
   Card order is deliberately NOT part of the list signature — `widget_order`
   is a separate, cheap fact the page applies through the masonry key order,
   never by moving or rebuilding nodes. */

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
