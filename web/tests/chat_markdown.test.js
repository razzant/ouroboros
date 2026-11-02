import assert from 'node:assert/strict';
import test from 'node:test';

import {
    bindMarkdownTables,
    chatMarkdownUrl,
    destroyChatMarkdown,
    enhanceChatMarkdown,
    parseChartConfig,
    prepareMarkdownSource,
    renderChatMarkdown,
} from '../modules/chat_markdown.js';

test('prepareMarkdownSource neutralizes only raw HTML and leaves entities to the parser', () => {
    // `<` becomes a reference the text does not contain; `&` and `>` stay the
    // author's, so marked reads entities in prose and keeps code literal.
    const cases = [
        ['<tag>&', '&#60;tag>&'],
        ['&lt;tag&gt;&amp;', '&lt;tag&gt;&amp;'],
        ['> quote with a <b> tag', '> quote with a &#60;b> tag'],
        ['`<b>&lt;x&gt;&`', '`&#60;b>&lt;x&gt;&`'],
        // An autolink to an address a link may carry opens no HTML and keeps its `<`.
        ['<https://e.com/a> <MAILTO:a@e.com> <https://e.com/a b> <std::x> <javascript:x>',
            '<https://e.com/a> <MAILTO:a@e.com> &#60;https://e.com/a b> &#60;std::x> &#60;javascript:x>'],
        // An escaped `<` takes its backslash into its own stand-in; backslashes pair as marked pairs them.
        ['\\<b> \\\\<i> \\\\\\<u> `\\<c>` \\a \\<https://e.com/a>',
            '&#060;b> \\\\&#60;i> \\\\&#060;u> `&#060;c>` \\a \\<https://e.com/a>'],
    ];
    for (const [source, expected] of cases) {
        assert.deepEqual(prepareMarkdownSource(source), {
            source: expected, lessThan: '&#60;', escapedLessThan: '&#060;',
        }, source);
    }
    // An author who writes a reference itself keeps it: each stand-in moves on.
    assert.deepEqual(prepareMarkdownSource('&#60; and &#060; < \\<'), {
        source: '&#60; and &#060; &#0060; &#00060;', lessThan: '&#0060;', escapedLessThan: '&#00060;',
    });
    assert.deepEqual(prepareMarkdownSource('&#060; < \\<'), {
        source: '&#060; &#60; &#0060;', lessThan: '&#60;', escapedLessThan: '&#0060;',
    });
});

test('renderChatMarkdown restores 11th+ display-math blocks without sentinel prefix collisions', () => {
    const priorDocument = globalThis.document;
    const priorMarked = globalThis.marked;
    const priorPurify = globalThis.DOMPurify;
    globalThis.document = {
        createElement: (tagName) => {
            assert.equal(tagName, 'template');
            return {
                content: { querySelectorAll: () => [] },
                get innerHTML() { return this.value; },
                set innerHTML(value) { this.value = String(value); },
            };
        },
    };
    globalThis.marked = {
        Marked: class {
            parse(source) {
                return source.split(/\n{2,}/)
                    .filter(Boolean)
                    .map((block) => `<div class="math-display">${block}</div>`)
                    .join('');
            }
        },
    };
    globalThis.DOMPurify = { sanitize: (html) => html };

    const assertRestored = (html, dollarBlocks, bracketBlocks) => {
        for (let index = 0; index < 12; index += 1) {
            const equation = `eq_${index} = ${index}^2`;
            assert.equal(html.split(equation).length - 1, 1, equation);
        }
        assert.doesNotMatch(html, /OUROBOROSLATEX|DISPLAY/);
        assert.doesNotMatch(html, /(?:\$\$|\\\])\d/);
        assert.equal(html.match(/\$\$[\s\S]*?\$\$/g)?.length || 0, dollarBlocks);
        assert.equal(html.match(/\\\[[\s\S]*?\\\]/g)?.length || 0, bracketBlocks);
        assert.equal(html.match(/class="math-display"/g)?.length || 0, 12);
    };

    try {
        const dollarSource = Array.from(
            { length: 12 },
            (_, index) => `$$eq_${index} = ${index}^2$$`,
        ).join('\n\n');
        assertRestored(renderChatMarkdown(dollarSource), 12, 0);

        const mixedSource = [
            ...Array.from(
                { length: 10 },
                (_, index) => `$$eq_${index} = ${index}^2$$`,
            ),
            '\\\[\neq_10 = 10^2\n\\\]',
            '\\\[\neq_11 = 11^2\n\\\]',
        ].join('\n\n');
        assertRestored(renderChatMarkdown(mixedSource), 10, 2);
    } finally {
        globalThis.document = priorDocument;
        globalThis.marked = priorMarked;
        globalThis.DOMPurify = priorPurify;
    }
});

test('chatMarkdownUrl accepts external schemes and the exact file route', () => {
    const cases = [
        ['https://example.com/report', 'https://example.com/report'],
        ['http://example.com/report', 'http://example.com/report'],
        ['mailto:owner@example.com', 'mailto:owner@example.com'],
        ['/api/files/download?path=reports%2Fq1.pdf', '/api/files/download?path=reports%2Fq1.pdf'],
        ['/api/files/download?path=reports%2Ffinal+copy.pdf', '/api/files/download?path=reports%2Ffinal+copy.pdf'],
    ];
    for (const [source, expected] of cases) assert.equal(chatMarkdownUrl(source), expected, source);
});

test('chatMarkdownUrl rejects unsafe or non-exact relative URLs', () => {
    const rejected = [
        '',
        'javascript:alert(1)',
        'data:text/html,bad',
        'report.pdf',
        '//example.com/report',
        '/api/other?path=report.pdf',
        '/api/files/download',
        '/api/files/download?path=',
        '/api/files/download?path=../secret.txt',
        '/api/files/download?path=%2Fetc%2Fpasswd',
        '/api/files/download?path=report.pdf&extra=1',
        '/api/files/download?path=report.pdf#fragment',
    ];
    for (const source of rejected) assert.equal(chatMarkdownUrl(source), '', source);
});

test('parseChartConfig accepts exactly the supported chart types', () => {
    const types = ['bar', 'line', 'pie', 'doughnut', 'polarArea', 'radar', 'scatter', 'bubble'];
    for (const type of types) {
        const result = parseChartConfig(JSON.stringify({
            type,
            data: { labels: ['A'], datasets: [{ label: 'Series', data: [1] }] },
        }));
        assert.equal(result.type, type);
    }
    assert.throws(
        () => parseChartConfig('{"type":"area","data":{"datasets":[]}}'),
        /unsupported chart type/,
    );
});

test('parseChartConfig enforces dataset, point, and label caps', () => {
    const config = (datasets, labels = []) => JSON.stringify({ type: 'line', data: { datasets, labels } });
    const dataset = { data: Array(500).fill(1) };
    assert.equal(parseChartConfig(config(Array(24).fill(dataset), Array(500).fill('x'))).data.datasets.length, 24);
    assert.throws(() => parseChartConfig(config(Array(25).fill(dataset))), /too many chart datasets/);
    assert.throws(() => parseChartConfig(config([{ data: Array(501).fill(1) }])), /too many chart points/);
    assert.throws(() => parseChartConfig(config([{ data: [] }], Array(501).fill('x'))), /too many chart labels/);
    assert.throws(() => parseChartConfig(config([{ label: 'missing data' }])), /data array/);
});

test('parseChartConfig applies forced options after user options', () => {
    const parsed = parseChartConfig(JSON.stringify({
        type: 'bar',
        data: { datasets: [{ data: [1, 2] }] },
        options: { responsive: false, maintainAspectRatio: true, animation: false },
    }));
    assert.deepEqual(parsed.options, {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
    });
    assert.throws(
        () => parseChartConfig('{"type":"bar","data":{"datasets":[]},}'),
        SyntaxError,
    );
});

test('parseChartConfig exposes only allowlisted chart and dataset keys', () => {
    const parsed = parseChartConfig(JSON.stringify({
        type: 'line',
        plugins: { arbitrary: { nested: true } },
        bogus: 'drop me',
        data: {
            labels: ['A'],
            extra: { nested: true },
            datasets: [{
                data: [1],
                label: 'Series',
                backgroundColor: '#123',
                borderColor: '#456',
                borderWidth: 2,
                fill: false,
                tension: 0.25,
                extra: { nested: true },
            }],
        },
        options: { animation: false },
    }));
    assert.deepEqual(parsed, {
        type: 'line',
        data: {
            labels: ['A'],
            datasets: [{
                data: [1],
                label: 'Series',
                backgroundColor: '#123',
                borderColor: '#456',
                borderWidth: 2,
                fill: false,
                tension: 0.25,
            }],
        },
        options: { animation: false, responsive: true, maintainAspectRatio: false },
    });
});

test('chart success and error writes use the supplied viewport boundary', () => {
    const prior = { document: globalThis.document, Chart: globalThis.Chart };
    const node = (text) => ({
        textContent: text,
        dataset: {},
        classList: { values: new Set(), add(value) { this.values.add(value); } },
        replaceChildren(...children) { this.children = children; },
    });
    const valid = node('{"type":"bar","data":{"datasets":[{"data":[1]}]}}');
    const invalid = node('not json');
    const root = {
        isConnected: true,
        matches: () => false,
        querySelectorAll: (selector) => selector === '.md-chart' ? [valid, invalid] : [],
        setAttribute() {}, removeAttribute() {}, addEventListener() {}, removeEventListener() {},
    };
    let destroyed = false;
    globalThis.document = { createElement: () => ({}) };
    globalThis.Chart = class {
        destroy() { destroyed = true; }
    };
    let writes = 0;
    try {
        const dispose = enhanceChatMarkdown(root, {
            onDomWrite: (mutate) => { writes += 1; return mutate(); },
        });
        assert.equal(writes, 3); // shared sync enhancement + success + error
        assert.equal(valid.dataset.processed, 'true');
        assert.equal(valid.children.length, 1);
        assert.equal(invalid.dataset.processed, 'true');
        assert.ok(invalid.classList.values.has('md-chart-error'));
        dispose();
        assert.equal(destroyed, true);
    } finally {
        globalThis.document = prior.document;
        globalThis.Chart = prior.Chart;
    }
});

test('a table is a focusable region with faded edges only while it scrolls, and the disposer releases it', () => {
    const prior = { ResizeObserver: globalThis.ResizeObserver, document: globalThis.document };
    const observed = [];
    let callback = null;
    let disconnected = false;
    globalThis.ResizeObserver = class {
        constructor(fn) { callback = fn; }
        observe(node) { observed.push(node); }
        disconnect() { disconnected = true; }
    };
    const attributes = new Map();
    const table = {};
    const wrap = {
        isConnected: true, scrollWidth: 900, clientWidth: 300, scrollLeft: 0, firstElementChild: table,
        classList: { contains: (name) => name === 'md-table-wrap' },
        hasAttribute: (name) => attributes.has(name),
        setAttribute: (name, value) => attributes.set(name, String(value)),
        removeAttribute: (name) => attributes.delete(name),
        toggleAttribute: (name, force) => { if (force) attributes.set(name, ''); else attributes.delete(name); return force; },
        get tabIndex() { return Number(attributes.get('tabindex') ?? -1); },
        set tabIndex(value) { attributes.set('tabindex', String(value)); },
    };
    table.closest = () => wrap;
    wrap.closest = (selector) => selector === '.md-table-wrap' ? wrap : null;
    const listeners = new Map();
    const root = {
        isConnected: true, matches: () => false,
        querySelector: (selector) => selector === '.md-table-wrap' ? wrap : null,
        querySelectorAll: (selector) => selector === '.md-table-wrap' ? [wrap] : [],
        setAttribute() {}, removeAttribute() {},
        addEventListener: (type, fn, capture) => listeners.set(`${type}:${Boolean(capture)}`, fn),
        removeEventListener: (type, fn, capture) => {
            if (listeners.get(`${type}:${Boolean(capture)}`) === fn) listeners.delete(`${type}:${Boolean(capture)}`);
        },
    };
    globalThis.document = { createElement: () => ({}) };
    try {
        const dispose = enhanceChatMarkdown(root);
        assert.deepEqual(observed, [wrap, table], 'the wrapper and its table are observed');
        callback([{ target: wrap }]);
        assert.deepEqual(Object.fromEntries(attributes), {
            'data-scroll-end': '', role: 'region', 'aria-label': 'Scrollable table', tabindex: '0',
        });
        wrap.scrollLeft = 600;
        listeners.get('scroll:true')({ target: wrap });
        assert.equal(attributes.has('data-scroll-start'), true);
        assert.equal(attributes.has('data-scroll-end'), false, 'scrolled to the end, nothing hides there');
        wrap.scrollWidth = 300;
        wrap.scrollLeft = 0;
        callback([{ target: table }]);
        assert.deepEqual(Object.fromEntries(attributes), {}, 'a table that fits is plain content again');
        dispose();
        assert.equal(disconnected, true);
        assert.equal(listeners.has('scroll:true'), false);
    } finally {
        globalThis.ResizeObserver = prior.ResizeObserver;
        globalThis.document = prior.document;
    }
});

function fakeTableWrap(name) {
    const attributes = new Map();
    const table = { name: `${name} table` };
    const wrap = {
        name, nodeType: 1, isConnected: true, scrollWidth: 900, clientWidth: 300, scrollLeft: 0,
        firstElementChild: table, attributes,
        matches: (selector) => selector === '.md-table-wrap',
        querySelectorAll: () => [],
        classList: { contains: (className) => className === 'md-table-wrap' },
        hasAttribute: (key) => attributes.has(key),
        setAttribute: (key, value) => attributes.set(key, String(value)),
        removeAttribute: (key) => attributes.delete(key),
        toggleAttribute: (key, force) => { if (force) attributes.set(key, ''); else attributes.delete(key); return force; },
        get tabIndex() { return Number(attributes.get('tabindex') ?? -1); },
        set tabIndex(value) { attributes.set('tabindex', String(value)); },
    };
    table.closest = () => wrap;
    wrap.closest = (selector) => selector === '.md-table-wrap' ? wrap : null;
    return wrap;
}

test('a compact surface binds only its tables, follows later writes, and an ancestor destroy releases it', () => {
    const prior = { ResizeObserver: globalThis.ResizeObserver, MutationObserver: globalThis.MutationObserver };
    const observed = new Set();
    let resized = null;
    let written = null;
    let watching = null;
    globalThis.ResizeObserver = class {
        constructor(fn) { resized = fn; }
        observe(node) { observed.add(node); }
        unobserve(node) { observed.delete(node); }
        disconnect() { observed.clear(); }
    };
    globalThis.MutationObserver = class {
        constructor(fn) { written = fn; }
        observe(node, options) { watching = { node, options }; }
        disconnect() { watching = null; }
    };
    const first = fakeTableWrap('first');
    const later = fakeTableWrap('later');
    const inside = new Set([first]);
    // A timeline row that arrives later carrying a table; a text node carries none.
    const row = { nodeType: 1, matches: () => false, querySelectorAll: (selector) => selector === '.md-table-wrap' ? [later] : [] };
    const text = { nodeType: 3 };
    const markers = new Map();
    const listeners = new Map();
    const root = {
        nodeType: 1,
        querySelectorAll: (selector) => selector === '.md-table-wrap' ? [...inside] : [],
        contains: (node) => inside.has(node),
        setAttribute: (key, value) => markers.set(key, value),
        removeAttribute: (key) => markers.delete(key),
        addEventListener: (type, fn, capture) => listeners.set(`${type}:${Boolean(capture)}`, fn),
        removeEventListener: (type, fn, capture) => {
            if (listeners.get(`${type}:${Boolean(capture)}`) === fn) listeners.delete(`${type}:${Boolean(capture)}`);
        },
    };
    const page = {
        querySelectorAll: (selector) => selector === '[data-md-tables]' && markers.has('data-md-tables') ? [root] : [],
    };
    const names = () => [...observed].map((node) => node.name).sort();
    try {
        const dispose = bindMarkdownTables(root);
        assert.equal(bindMarkdownTables(root), dispose, 'binding a bound root again returns its disposer');
        assert.deepEqual(names(), ['first', 'first table']);
        assert.deepEqual(watching, { node: root, options: {
            childList: true, subtree: true, attributeFilter: ['tabindex', 'data-scroll-start', 'data-scroll-end'],
        } });
        assert.equal(markers.has('data-md-tables'), true);
        assert.equal(listeners.has('scroll:true'), true);

        inside.add(later);
        written([{ addedNodes: [row, text], removedNodes: [] }]);
        assert.deepEqual(names(), ['first', 'first table', 'later', 'later table'], 'a table written later is observed');
        resized([{ target: later.firstElementChild }]);
        assert.deepEqual(Object.fromEntries(later.attributes), {
            'data-scroll-end': '', role: 'region', 'aria-label': 'Scrollable table', tabindex: '0',
        }, 'an overflowing compact table is a keyboard region with its hidden side faded');
        // Late registration and keyed attribute writes must respect the containing preview.
        let folded = true;
        later.closest = (selector) => selector === '.md-table-wrap' ? later : folded ? {} : null;
        resized([{ target: later.firstElementChild }]);
        assert.equal(later.tabIndex, -1);
        later.tabIndex = 0;
        written([{ type: 'attributes', target: later, attributeName: 'tabindex' }]);
        assert.equal(later.tabIndex, -1, 'a later attribute write cannot reopen a folded table');
        folded = false;
        resized([{ target: later.firstElementChild }]);
        assert.equal(later.tabIndex, 0, 'an ordinary visible table keeps keyboard scrolling');
        later.scrollLeft = 600;
        // A host's own tests may call every scroll listener with no event.
        assert.doesNotThrow(() => listeners.get('scroll:true')());
        listeners.get('scroll:true')({ target: later });
        assert.deepEqual([later.attributes.has('data-scroll-start'), later.attributes.has('data-scroll-end')], [true, false]);
        // A keyed row patch keeps the wrapper but copies fresh markup's attributes:
        // nothing resizes, and the removal itself re-marks the region.
        later.attributes.clear();
        written([{ type: 'attributes', target: later, attributeName: 'tabindex', addedNodes: [], removedNodes: [] }]);
        assert.deepEqual(Object.fromEntries(later.attributes), {
            'data-scroll-start': '', role: 'region', 'aria-label': 'Scrollable table', tabindex: '0',
        });

        // Unfolding may remove tabindex before the table also stops overflowing.
        later.removeAttribute('tabindex');
        later.scrollWidth = later.clientWidth;
        written([{ type: 'attributes', target: later, attributeName: 'tabindex' }]);
        assert.equal(later.hasAttribute('role'), false);
        assert.equal(later.hasAttribute('aria-label'), false);
        assert.equal(later.hasAttribute('tabindex'), false);

        // A replaced table is released; one moved within the root stays observed.
        inside.delete(first);
        written([{ addedNodes: [], removedNodes: [first] }, { addedNodes: [row], removedNodes: [row] }]);
        assert.deepEqual(names(), ['later', 'later table']);

        destroyChatMarkdown(page);
        assert.equal(observed.size, 0);
        assert.equal(watching, null);
        assert.equal(listeners.has('scroll:true'), false);
        assert.equal(markers.has('data-md-tables'), false);
        dispose();
        const again = bindMarkdownTables(root);
        assert.notEqual(again, dispose, 'a released root binds afresh');
        again();
        assert.equal(markers.has('data-md-tables'), false);
    } finally {
        globalThis.ResizeObserver = prior.ResizeObserver;
        globalThis.MutationObserver = prior.MutationObserver;
    }
});

test('a resize batch measures every table first, then marks every one: a mark never lays out the next table', () => {
    const prior = globalThis.ResizeObserver;
    let resized = null;
    globalThis.ResizeObserver = class {
        constructor(fn) { resized = fn; }
        observe() {}
        unobserve() {}
        disconnect() {}
    };
    const steps = [];
    const wraps = ['one', 'two'].map((name) => {
        const wrap = fakeTableWrap(name);
        let width = wrap.scrollWidth;
        Object.defineProperty(wrap, 'scrollWidth', { get: () => { steps.push(`read ${name}`); return width; } });
        const toggle = wrap.toggleAttribute;
        wrap.toggleAttribute = (key, force) => { steps.push(`mark ${name}`); return toggle(key, force); };
        return wrap;
    });
    const root = {
        nodeType: 1, querySelectorAll: (selector) => selector === '.md-table-wrap' ? wraps : [],
        setAttribute() {}, removeAttribute() {}, addEventListener() {}, removeEventListener() {},
    };
    try {
        const dispose = bindMarkdownTables(root);
        resized(wraps.flatMap((wrap) => [{ target: wrap }, { target: wrap.firstElementChild }]));
        assert.deepEqual(steps, ['read one', 'read two', 'mark one', 'mark one', 'mark two', 'mark two'],
            'each table once, read before any write');
        assert.equal(wraps.every((wrap) => wrap.attributes.has('data-scroll-end')), true);
        dispose();
    } finally {
        globalThis.ResizeObserver = prior;
    }
});

test('a chat instance binds its own column, so compact tables need no caller of their own', async () => {
    const { installDom, restoreDom } = await import('./chat_dom_fixture.js');
    const { createChatInstance } = await import('../modules/chat.js');
    const { prior, mount } = installDom();
    let instance;
    try {
        instance = createChatInstance({
            ws: { on: () => () => {}, isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
            stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }),
                gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
            chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true,
        });
        const column = globalThis.document.byId.get('chat-messages');
        assert.equal(column.attributes.has('data-md-tables'), true, 'createChatInstance bound the column');
        assert.equal(bindMarkdownTables(column), bindMarkdownTables(column), 'one binding per column');
        // chat.js's own scroll listeners (its reading gestures last) keep their order; the table binding registers after them.
        assert.deepEqual(column.listeners.get('scroll').slice(-2).map(fn => fn.name), ['noteBoxScroll', 'onScroll']);
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('without ResizeObserver the table binding is inert', () => {
    const prior = globalThis.ResizeObserver;
    delete globalThis.ResizeObserver;
    try {
        const touched = [];
        const root = new Proxy({}, { get: (_, key) => { touched.push(key); return undefined; } });
        const dispose = bindMarkdownTables(root);
        assert.equal(typeof dispose, 'function');
        dispose();
        assert.deepEqual(touched, []);
    } finally {
        if (prior) globalThis.ResizeObserver = prior;
    }
});
