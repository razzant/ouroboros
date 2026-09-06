/* Absolute-position masonry for the Widgets list. `layout()` measures the
   container and its items, plans the columns (`planMasonryLayout`, pure) and
   writes the plan back ONLY as narrow custom properties — `--masonry-w/-x/-y`
   on each item, `--masonry-h` on the container — which one static rule set in
   web/style.css turns into width / transform / height. The visual order is the
   caller's explicit key order (`options.order`), never the DOM order: a reorder
   relayouts without moving a node, so a running <iframe> in a card is never
   reloaded by it. `applyMasonry` returns an idempotent disposer. */

const bound = new WeakMap();

function shortestColumn(columns) {
    let index = 0;
    for (let i = 1; i < columns.length; i += 1) {
        if (columns[i] < columns[index]) index = i;
    }
    return index;
}

function bestPair(columns) {
    if (columns.length < 2) return 0;
    let index = 0;
    let best = Math.max(columns[0], columns[1]);
    for (let i = 1; i < columns.length - 1; i += 1) {
        const candidate = Math.max(columns[i], columns[i + 1]);
        if (candidate < best) {
            best = candidate;
            index = i;
        }
    }
    return index;
}

export function planMasonryLayout(width, itemSpecs, options = {}) {
    const gap = Number(options.gap ?? 14);
    const minColumnWidth = Number(options.minColumnWidth ?? 280);
    const denseMinColumnWidth = Number(options.denseMinColumnWidth ?? 240);
    const spans = itemSpecs.map((item) => Number(item.span) >= 2 ? 2 : 1);
    const desiredColumns = spans.reduce((total, span) => total + span, 0);
    const availableColumns = Math.max(1, Math.floor((width + gap) / (minColumnWidth + gap)));
    let count = Math.min(desiredColumns, availableColumns);

    // Multiple wide cards cannot pack usefully into three tracks: every pair
    // overlaps an occupied track and leaves a tall visual void. When four
    // still-legible tracks fit, let wide cards sit side by side. One-wide and
    // narrow layouts retain the ordinary minimum width.
    const wideCount = spans.filter((span) => span === 2).length;
    const denseAvailableColumns = Math.max(
        1,
        Math.floor((width + gap) / (denseMinColumnWidth + gap)),
    );
    if (wideCount >= 2 && denseAvailableColumns >= 4) {
        count = Math.min(desiredColumns, denseAvailableColumns);
    }

    // On the common four-track desktop layout, one narrow card between
    // multiple wide cards otherwise occupies only half of the right lane.
    // That leaves a persistent visual hole and needlessly squeezes long
    // readouts. Treat the lone narrow span as a responsive width hint and
    // give it the same readable lane width as its wide neighbours.
    const effectiveSpans = [...spans];
    const narrowIndexes = spans
        .map((span, index) => span === 1 ? index : -1)
        .filter((index) => index >= 0);
    if (count === 4 && wideCount >= 2 && narrowIndexes.length === 1) {
        effectiveSpans[narrowIndexes[0]] = 2;
    }

    const columnWidth = Math.floor((width - gap * (count - 1)) / count);
    const heights = Array(count).fill(0);
    const placements = itemSpecs.map((item, index) => {
        const span = effectiveSpans[index] === 2 && count > 1 ? 2 : 1;
        const column = span === 2 ? bestPair(heights) : shortestColumn(heights);
        const top = span === 2
            ? Math.max(heights[column], heights[column + 1])
            : heights[column];
        const left = column * (columnWidth + gap);
        const itemWidth = span * columnWidth + (span - 1) * gap;
        const bottom = top + Math.max(0, Number(item.height) || 0) + gap;
        for (let i = column; i < column + span; i += 1) heights[i] = bottom;
        return { span, column, top, left, width: itemWidth };
    });
    return {
        columnCount: count,
        columnWidth,
        height: Math.max(0, Math.max(...heights, 0) - gap),
        placements,
    };
}

// Items in the caller's key order; keys the order does not name keep their
// relative DOM order after the named ones (the `sortTabsByWidgetOrder` rule).
function orderedItems(container, config) {
    const rank = new Map(config.order.map((key, index) => [key, index]));
    return Array.from(container.querySelectorAll(config.itemSelector))
        .map((item, index) => {
            const key = config.keyOf(item);
            return { item, index, rank: rank.has(key) ? rank.get(key) : Number.MAX_SAFE_INTEGER };
        })
        .sort((a, b) => a.rank - b.rank || a.index - b.index)
        .map((entry) => entry.item);
}

function layout(container, config) {
    const items = orderedItems(container, config);
    if (!items.length) {
        container.style.removeProperty('--masonry-h');
        return;
    }
    const width = container.clientWidth;
    if (!width) return;
    const spanClass = config.spanClass || 'widgets-card-span-2';
    const itemSpecs = items.map((item) => ({
        span: item.classList.contains(spanClass) ? 2 : 1,
        height: item.offsetHeight,
    }));
    const plan = planMasonryLayout(width, itemSpecs, config);
    items.forEach((item, idx) => {
        const placement = plan.placements[idx];
        item.style.setProperty('--masonry-w', `${placement.width}px`);
        item.style.setProperty('--masonry-x', `${placement.left}px`);
        item.style.setProperty('--masonry-y', `${placement.top}px`);
    });
    container.style.setProperty('--masonry-h', `${plan.height}px`);
}

/**
 * Bind (once per container) and schedule a layout. A later call with
 * `options.order` replaces the key order and relayouts; every call returns the
 * same idempotent disposer, which disconnects the three observers, cancels a
 * pending frame and forgets the container.
 */
export function applyMasonry(container, options = {}) {
    if (!container) return () => {};
    const existing = bound.get(container);
    if (existing) {
        if (Array.isArray(options.order)) existing.config.order = options.order.slice();
        existing.run();
        return existing.dispose;
    }
    const config = {
        itemSelector: options.itemSelector || '.widgets-card',
        gap: options.gap ?? 14,
        minColumnWidth: options.minColumnWidth ?? 280,
        spanClass: options.spanClass || 'widgets-card-span-2',
        keyOf: options.keyOf || ((item) => item.dataset.widgetKey || ''),
        order: Array.isArray(options.order) ? options.order.slice() : [],
    };
    // One layout per frame however many triggers land before it.
    let frame = 0;
    const run = () => {
        if (frame) cancelAnimationFrame(frame);
        frame = requestAnimationFrame(() => {
            frame = 0;
            layout(container, config);
        });
    };
    const observedItems = new Set();
    const itemResizeObserver = new ResizeObserver(run);
    const observeItems = () => {
        Array.from(observedItems).forEach((item) => {
            if (container.contains(item)) return;
            itemResizeObserver.unobserve(item);
            observedItems.delete(item);
        });
        container.querySelectorAll(config.itemSelector).forEach((item) => {
            if (observedItems.has(item)) return;
            observedItems.add(item);
            itemResizeObserver.observe(item);
        });
    };
    const resizeObserver = new ResizeObserver(run);
    resizeObserver.observe(container);
    const mutationObserver = new MutationObserver(() => {
        observeItems();
        run();
    });
    mutationObserver.observe(container, { childList: true, subtree: true });
    const entry = { config, run };
    entry.dispose = () => {
        if (bound.get(container) !== entry) return;
        bound.delete(container);
        if (frame) cancelAnimationFrame(frame);
        frame = 0;
        resizeObserver.disconnect();
        itemResizeObserver.disconnect();
        mutationObserver.disconnect();
        observedItems.clear();
    };
    bound.set(container, entry);
    observeItems();
    run();
    return entry.dispose;
}
