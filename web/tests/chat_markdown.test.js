import assert from 'node:assert/strict';
import test from 'node:test';

import {
    bindCompactMarkdown,
    bindRenderedMarkdown,
    chatMarkdownUrl,
    destroyChatMarkdown,
    enhanceChatMarkdown,
    parseChartConfig,
    prepareMarkdownSource,
    renderChatMarkdown,
    linkify,
} from '../modules/chat_markdown.js';
import { restoreFocusWithoutReveal, syncFoldStops } from '../modules/scroll_fade.js';

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

// The owner's text and the addresses in it that link, as [shown words, href].
const OWNER_TEXT_LINKS = [
    // A sentence's punctuation and an unopened closing bracket end an address; a bracket it opened stays in it.
    ['See https://en.wikipedia.org/wiki/Foo_(bar)). Then https://e.com/a, http://[::1]:8000/b; ok…',
        [['https://en.wikipedia.org/wiki/Foo_(bar)', 'https://en.wikipedia.org/wiki/Foo_(bar)'],
            ['https://e.com/a', 'https://e.com/a'], ['http://[::1]:8000/b', 'http://[::1]:8000/b']]],
    // Guillemets and quotes are not part of the address; Cyrillic stays shown as written, the href is its parsed form.
    ['«https://пример.рф/путь?q=1&lang=ru#раздел» "https://e.com/q" \'https://e.com/s\' “https://e.com/t”',
        [['https://пример.рф/путь?q=1&lang=ru#раздел',
            'https://xn--e1afmkfd.xn--p1ai/%D0%BF%D1%83%D1%82%D1%8C?q=1&lang=ru#%D1%80%D0%B0%D0%B7%D0%B4%D0%B5%D0%BB'],
        ['https://e.com/q', 'https://e.com/q'], ['https://e.com/s', 'https://e.com/s'], ['https://e.com/t', 'https://e.com/t']]],
    // Markdown and backticks stay literal text; an address inside them still links.
    ['`https://e.com/code` **https://e.com/bold** [x](https://e.com/md)',
        [['https://e.com/code', 'https://e.com/code'], ['https://e.com/bold**', 'https://e.com/bold**'],
            ['https://e.com/md', 'https://e.com/md']]],
    // URL data is not a Markdown delimiter: standalone suffixes survive, including query tokens.
    ['https://e.com/download?token=AbC_ https://e.com/~ https://e.com/?q=topic*',
        [['https://e.com/download?token=AbC_', 'https://e.com/download?token=AbC_'],
            ['https://e.com/~', 'https://e.com/~'], ['https://e.com/?q=topic*', 'https://e.com/?q=topic*']]],
    // HTML stays text: its javascript: attribute is never a destination, only the visible address is.
    ['<a href="javascript:alert(1)">https://e.com/x</a><img src=x onerror=alert(1)>', [['https://e.com/x', 'https://e.com/x']]],
    // Only an explicit http(s) address with a host links: no other scheme, bare host or address glued to a word.
    ['javascript:alert(1) data:text/html,x mailto:a@e.com www.e.com xhttps://e.com https:// https:///x https://\\x', []],
    // An upper-case scheme links; an invisible format character ends an address rather than hiding inside it.
    ['HTTPS://E.COM/Up and https://e.com/a‮b​c', [['HTTPS://E.COM/Up', 'https://e.com/Up'], ['https://e.com/a', 'https://e.com/a']]],
    // Blanks, line breaks and entities stay the author's.
    ['a & b\n\t<c>  https://e.com/x?y=1&z=2\r\n&amp;', [['https://e.com/x?y=1&z=2', 'https://e.com/x?y=1&z=2']]],
];

test('linkify keeps the owner text literal and links only its explicit http(s) addresses', () => {
    const anchor = /<a class="md-link" href="([^"]*)" target="_blank" rel="noopener noreferrer">([^<]*)<\/a>/g;
    const unescape = (value) => value.replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&#39;', "'")
        .replaceAll('&quot;', '"').replaceAll('&#96;', '`').replaceAll('&amp;', '&');
    for (const [source, expected] of OWNER_TEXT_LINKS) {
        const html = linkify(source);
        assert.deepEqual([...html.matchAll(anchor)].map(([, href, shown]) => [unescape(shown), unescape(href)]), expected, source);
        // No markup but those links, and the words read back exactly as written.
        const text = html.replace(anchor, '$2');
        assert.equal(text.includes('<'), false, source);
        assert.equal(unescape(text), source);
    }
    assert.equal(linkify(null), '');
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
                parsing: { yAxisKey: 'walk.distance' },
                extra: { nested: true },
            }],
        },
        options: { animation: false, parsing: { yAxisKey: 'walk.total' } },
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
        options: { animation: false, parsing: { yAxisKey: 'walk.total' }, responsive: true, maintainAspectRatio: false },
    });
});

test('a selected code fence keeps its plain text only while that selection lives, then resumes highlighting through the viewport writer', async () => {
    const prior = { document: globalThis.document, hljs: globalThis.hljs };
    let held = true;
    // writeUnselected defers on the node's own document; the selection lives there too.
    const doc = new EventTarget();
    doc.getSelection = () => ({ isCollapsed: !held, rangeCount: 1,
        getRangeAt: () => ({ intersectsNode: (node) => node === selectedCode }) });
    const fence = (text) => {
        const code = {
            isConnected: true, textContent: text, classList: new Set(['language-js']),
            get innerHTML() { return this._html ?? this.textContent; },
            set innerHTML(value) { this._html = String(value); },
        };
        code.ownerDocument = doc;
        return code;
    };
    const selectedCode = fence('const selected = true;');
    const plainCode = fence('const plain = true;');
    const root = {
        isConnected: true, ownerDocument: doc, matches: () => false,
        querySelector: (selector) => (selector === '.md-code-block' ? selectedCode : null),
        querySelectorAll: (selector) => (selector === '.md-code-block pre > code' ? [selectedCode, plainCode] : []),
        setAttribute() {}, removeAttribute() {}, addEventListener() {}, removeEventListener() {},
    };
    globalThis.document = { createElement: () => ({}) };
    globalThis.hljs = {
        getLanguage: () => true,
        highlight: (source, { language }) => ({ value: `<span data-lang="${language}">${source}</span>` }),
    };
    let writes = 0;
    try {
        const dispose = enhanceChatMarkdown(root, { onDomWrite: (write) => { writes += 1; return write(); } });
        // The unselected fence paints at once through the same writer.
        assert.equal(plainCode.innerHTML, '<span data-lang="js">const plain = true;</span>');
        assert.ok(plainCode.classList.has('hljs'));
        // The selected fence keeps the reader's exact bytes untouched.
        assert.equal(selectedCode.innerHTML, 'const selected = true;');
        assert.equal(selectedCode.classList.has('hljs'), false);
        assert.equal(writes, 1);

        held = false;
        doc.dispatchEvent(new Event('selectionchange'));
        // One selectionchange paints the deferred fence and releases its deferral.
        assert.equal(selectedCode.innerHTML, '<span data-lang="js">const selected = true;</span>');
        assert.ok(selectedCode.classList.has('hljs'));
        assert.equal(writes, 2);
        assert.equal(selectedCode.textContent, 'const selected = true;', 'the highlighted text still reads as written');
        dispose();
    } finally {
        globalThis.document = prior.document;
        globalThis.hljs = prior.hljs;
    }
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
        assert.equal(writes, 2); // only the two actual writes (success and error); no no-op sync enhancement
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
        getAttribute: (name) => (attributes.has(name) ? attributes.get(name) : null),
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

function fakeTableWrap(name, selector = '.md-table-wrap') {
    const attributes = new Map();
    const child = { name: `${name} ${selector === '.md-table-wrap' ? 'table' : 'code'}` };
    const wrap = {
        name, nodeType: 1, isConnected: true, scrollWidth: 900, clientWidth: 300, scrollLeft: 0,
        firstElementChild: child, attributes,
        matches: (name) => name === selector,
        querySelectorAll: () => [],
        classList: { contains: (className) => `.${className}` === selector },
        hasAttribute: (key) => attributes.has(key),
        getAttribute: (key) => (attributes.has(key) ? attributes.get(key) : null),
        setAttribute: (key, value) => attributes.set(key, String(value)),
        removeAttribute: (key) => attributes.delete(key),
        toggleAttribute: (key, force) => { if (force) attributes.set(key, ''); else attributes.delete(key); return force; },
        get tabIndex() { return Number(attributes.get('tabindex') ?? -1); },
        set tabIndex(value) { attributes.set('tabindex', String(value)); },
    };
    child.closest = (name) => name === selector ? wrap : null;
    wrap.closest = child.closest;
    return wrap;
}

test('a compact surface binds tables and compact code, follows later writes, and an ancestor destroy releases it', () => {
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
    const later = fakeTableWrap('later', '.md-compact-code');
    const inside = new Set([first]);
    // A late compact timeline row; the rich renderer's code keeps its own binding.
    const row = { nodeType: 1, matches: () => false, querySelectorAll: (selector) => selector === '.md-compact-code' ? [later] : [] };
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
        querySelectorAll: (selector) => selector === '[data-md-compact]' && markers.has('data-md-compact') ? [root] : [],
    };
    const names = () => [...observed].map((node) => node.name).sort();
    try {
        const dispose = bindCompactMarkdown(root);
        assert.equal(bindCompactMarkdown(root), dispose, 'binding a bound root again returns its disposer');
        assert.deepEqual(names(), ['first', 'first table']);
        assert.deepEqual(watching, { node: root, options: {
            childList: true, subtree: true, attributeFilter: ['tabindex', 'data-scroll-start', 'data-scroll-end', 'data-scroll-rtl'],
        } });
        assert.equal(markers.has('data-md-compact'), true);
        assert.equal(listeners.has('scroll:true'), true);

        inside.add(later);
        written([{ addedNodes: [row, text], removedNodes: [] }]);
        assert.deepEqual(names(), ['first', 'first table', 'later', 'later code'], 'code written later is observed');
        resized([{ target: later.firstElementChild }]);
        assert.deepEqual(Object.fromEntries(later.attributes), {
            'data-scroll-end': '', role: 'region', 'aria-label': 'Scrollable code', tabindex: '0',
        }, 'overflowing compact code is a keyboard region with its hidden side faded');
        // Late registration and keyed attribute writes must respect the containing preview.
        let folded = true;
        later.closest = (selector) => selector === '.md-compact-code' ? later : folded ? {} : null;
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
            'data-scroll-start': '', role: 'region', 'aria-label': 'Scrollable code', tabindex: '0',
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
        assert.deepEqual(names(), ['later', 'later code']);

        destroyChatMarkdown(page);
        assert.equal(observed.size, 0);
        assert.equal(watching, null);
        assert.equal(listeners.has('scroll:true'), false);
        assert.equal(markers.has('data-md-compact'), false);
        dispose();
        const again = bindCompactMarkdown(root);
        assert.notEqual(again, dispose, 'a released root binds afresh');
        again();
        assert.equal(markers.has('data-md-compact'), false);
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
        const dispose = bindCompactMarkdown(root);
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
        assert.equal(column.attributes.has('data-md-compact'), true, 'createChatInstance bound the column');
        assert.equal(bindCompactMarkdown(column), bindCompactMarkdown(column), 'one binding per column');
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
        const dispose = bindCompactMarkdown(root);
        assert.equal(typeof dispose, 'function');
        dispose();
        assert.deepEqual(touched, []);
    } finally {
        if (prior) globalThis.ResizeObserver = prior;
    }
});

function fakeRichScroller(classes, { width = 900, client = 300, folded = false, dir = '' } = {}) {
    const attributes = new Map();
    const classSet = new Set(classes.split(/\s+/).filter(Boolean));
    const child = { name: `${classes}-child`, closest: () => el };
    const el = {
        ownerDocument: { defaultView: { getComputedStyle: () => ({ direction: dir || 'ltr', overflowX: 'auto' }) } },
        nodeType: 1, isConnected: true, scrollWidth: width, clientWidth: client, scrollLeft: 0, dir,
        firstElementChild: child, attributes,
        classList: { contains: (name) => classSet.has(name), add: (name) => classSet.add(name) },
        // A keyed patch reuses the very node as another wrapper: the same object, a new class.
        reclass: (next) => { classSet.clear(); next.split(/\s+/).filter(Boolean).forEach((name) => classSet.add(name)); },
        matches: (sel) => sel.startsWith('.') && classSet.has(sel.slice(1)),
        querySelectorAll: () => [],
        closest: (sel) => (sel === '.project-answer.is-folded' ? (folded ? {} : null)
            : sel === '[dir]' ? (dir ? { getAttribute: () => dir } : null)
            : (sel.startsWith('.') && classSet.has(sel.slice(1)) ? el : null)),
        hasAttribute: (k) => attributes.has(k),
        getAttribute: (k) => (attributes.has(k) ? attributes.get(k) : null),
        setAttribute: (k, v) => attributes.set(k, String(v)),
        removeAttribute: (k) => attributes.delete(k),
        toggleAttribute: (k, force) => { if (force) attributes.set(k, ''); else attributes.delete(k); return force; },
        get tabIndex() { return Number(attributes.get('tabindex') ?? -1); },
        set tabIndex(v) { attributes.set('tabindex', String(v)); },
    };
    return el;
}

test('rich horizontal scrollers allocate nothing on plain prose, batch via deferred ResizeObserver, support RTL, preserve consumer attributes on short geometry, and follow async replacements', () => {
    const prior = { ResizeObserver: globalThis.ResizeObserver, MutationObserver: globalThis.MutationObserver, document: globalThis.document };
    let resizeCount = 0;
    let resized = null;
    let written = null;
    let disconnected = 0;
    const observed = new Set();
    globalThis.ResizeObserver = class {
        constructor(fn) { resizeCount += 1; resized = fn; }
        observe(node) { observed.add(node); }
        unobserve(node) { observed.delete(node); }
        disconnect() { disconnected += 1; observed.clear(); }
    };
    globalThis.MutationObserver = class {
        constructor(fn) { written = fn; }
        observe() {}
        disconnect() {}
    };
    globalThis.document = { createElement: () => ({ className: '', textContent: '' }) };
    try {
        // Plain prose row: zero ResizeObservers, zero scroll listeners.
        const plainListeners = new Map();
        const plainRoot = {
            isConnected: true, matches: () => false, querySelector: () => null, querySelectorAll: () => [],
            setAttribute() {}, removeAttribute() {},
            addEventListener: (type, fn, cap) => plainListeners.set(`${type}:${Boolean(cap)}`, fn),
            removeEventListener: (type, fn, cap) => plainListeners.delete(`${type}:${Boolean(cap)}`),
        };
        const disposePlain = enhanceChatMarkdown(plainRoot);
        assert.equal(resizeCount, 0, 'plain historic rows instantiate no ResizeObserver');
        assert.equal(plainListeners.has('scroll:true'), false, 'plain rows add no scroll listener');
        disposePlain();

        // Rich row with overflowing code, short code bearing consumer/fold attributes, RTL table, wide formula, and folded code.
        const wideCode = fakeRichScroller('scroll-fade-x', { width: 900, client: 300 });
        const shortConsumer = fakeRichScroller('scroll-fade-x', { width: 200, client: 300 });
        shortConsumer.setAttribute('tabindex', '-1');
        shortConsumer.setAttribute('data-fold-stop', '');
        const rtlTable = fakeRichScroller('md-table-wrap scroll-fade-x', { width: 900, client: 300, dir: 'rtl' });
        const wideMath = fakeRichScroller('katex-display scroll-fade-x', { width: 700, client: 300 });
        const foldedCode = fakeRichScroller('scroll-fade-x', { width: 900, client: 300, folded: true });
        const items = new Set([wideCode, shortConsumer, rtlTable, wideMath, foldedCode]);
        const listeners = new Map();
        const richRoot = {
            nodeType: 1, isConnected: true, matches: () => false,
            contains: (node) => items.has(node),
            querySelector: (sel) => [...items].find((el) => el.matches(sel)) || null,
            querySelectorAll: (sel) => [...items].filter((el) => el.matches(sel)),
            setAttribute() {}, removeAttribute() {},
            addEventListener: (type, fn, cap) => listeners.set(`${type}:${Boolean(cap)}`, fn),
            removeEventListener: (type, fn, cap) => {
                if (listeners.get(`${type}:${Boolean(cap)}`) === fn) listeners.delete(`${type}:${Boolean(cap)}`);
            },
        };
        const disposeRich = enhanceChatMarkdown(richRoot);
        assert.equal(resizeCount, 1, 'all rich scrollers and tables in a root share one ResizeObserver');
        assert.equal(wideCode.attributes.size, 0, 'no synchronous layout read before ResizeObserver reports');

        resized([...items].map((target) => ({ target })));
        assert.deepEqual(Object.fromEntries(wideCode.attributes), {
            'data-scroll-end': '', role: 'region', 'aria-label': 'Scrollable code', tabindex: '0',
        });
        assert.deepEqual(Object.fromEntries(shortConsumer.attributes), {
            tabindex: '-1', 'data-fold-stop': '',
        }, 'short geometry never strips consumer or fold tabindex=-1');
        assert.deepEqual(Object.fromEntries(rtlTable.attributes), {
            'data-scroll-end': '', 'data-scroll-rtl': '', role: 'region', 'aria-label': 'Scrollable table', tabindex: '0',
        });
        assert.equal(wideMath.attributes.get('aria-label'), 'Scrollable formula');
        assert.equal(foldedCode.attributes.get('tabindex'), '-1', 'folded Project mirror never gains a tab stop');

        // RTL travel uses negative scrollLeft toward inline-end and restores data-scroll-rtl across keyed attribute wipes.
        rtlTable.scrollLeft = -600;
        listeners.get('scroll:true')({ target: rtlTable });
        assert.deepEqual(Object.fromEntries(rtlTable.attributes), {
            'data-scroll-start': '', 'data-scroll-rtl': '', role: 'region', 'aria-label': 'Scrollable table', tabindex: '0',
        });
        rtlTable.attributes.clear();
        written([{ type: 'attributes', target: rtlTable, attributeName: 'data-scroll-rtl' }]);
        assert.equal(rtlTable.attributes.has('data-scroll-rtl'), true);
        assert.equal(rtlTable.attributes.has('data-scroll-start'), true);

        // Async diagram replacement observes only the newly added scroller subtree.
        const wideDiagram = fakeRichScroller('md-mermaid-scroll scroll-fade-x', { width: 800, client: 300 });
        items.add(wideDiagram);
        written([{ type: 'childList', target: richRoot, addedNodes: [wideDiagram], removedNodes: [] }]);
        resized([{ target: wideDiagram }]);
        assert.equal(wideDiagram.attributes.get('aria-label'), 'Scrollable diagram');

        // Shrinking removes only attributes granted by the binder itself.
        wideCode.scrollWidth = 300;
        resized([{ target: wideCode }]);
        assert.deepEqual(Object.fromEntries(wideCode.attributes), {});

        disposeRich();
        assert.equal(disconnected, 1);
        assert.equal(observed.size, 0);
        assert.equal(listeners.has('scroll:true'), false);
    } finally {
        globalThis.ResizeObserver = prior.ResizeObserver;
        globalThis.MutationObserver = prior.MutationObserver;
        globalThis.document = prior.document;
    }
});

test('bindRenderedMarkdown binds code and tables on render-only surfaces without touching plain prose and releases via destroyChatMarkdown', () => {
    const prior = globalThis.ResizeObserver;
    let observers = 0;
    let resized = null;
    let disconnected = 0;
    globalThis.ResizeObserver = class {
        constructor(fn) { observers += 1; resized = fn; }
        observe() {}
        unobserve() {}
        disconnect() { disconnected += 1; }
    };
    try {
        const plainAttrs = new Map();
        const plain = {
            matches: () => false, querySelector: () => null, querySelectorAll: () => [],
            setAttribute: (k, v) => plainAttrs.set(k, v), removeAttribute: (k) => plainAttrs.delete(k),
        };
        bindRenderedMarkdown(plain);
        assert.equal(observers, 0);
        assert.equal(plainAttrs.size, 0);

        const code = fakeRichScroller('scroll-fade-x', { width: 800, client: 300 });
        const attrs = new Map();
        const host = {
            nodeType: 1, isConnected: true, matches: () => false,
            querySelector: (s) => (code.matches(s) ? code : null),
            querySelectorAll: (s) => (code.matches(s) ? [code] : []),
            setAttribute: (k, v) => attrs.set(k, v),
            removeAttribute: (k) => attrs.delete(k),
            addEventListener() {}, removeEventListener() {},
        };
        const parent = {
            querySelectorAll: (s) => (s === '[data-md-rendered]' && attrs.has('data-md-rendered') ? [host] : []),
        };
        bindRenderedMarkdown(host);
        assert.equal(attrs.has('data-md-rendered'), true);
        resized([{ target: code }]);
        assert.equal(code.attributes.get('role'), 'region');
        destroyChatMarkdown(parent);
        assert.equal(attrs.has('data-md-rendered'), false);
        assert.equal(disconnected, 1);
    } finally {
        globalThis.ResizeObserver = prior;
    }
});

test('horizontal scrollers preserve fold stops, overflow loss during hold, and authored tabindex across run orders', () => {
    const prior = { ResizeObserver: globalThis.ResizeObserver, MutationObserver: globalThis.MutationObserver, document: globalThis.document };
    let resized = null, written = null;
    globalThis.ResizeObserver = class { constructor(fn) { resized = fn; } observe() {} unobserve() {} disconnect() {} };
    globalThis.MutationObserver = class { constructor(fn) { written = fn; } observe() {} disconnect() {} };
    globalThis.document = { createElement: () => ({ className: '', textContent: '' }) };
    try {
        const overflowFirst = fakeRichScroller('md-table-wrap scroll-fade-x', { width: 900, client: 300 });
        const foldFirst = fakeRichScroller('md-table-wrap scroll-fade-x', { width: 900, client: 300 });
        const lostInHold = fakeRichScroller('scroll-fade-x', { width: 200, client: 300 });
        const authored = fakeRichScroller('scroll-fade-x', { width: 900, client: 300 });
        authored.setAttribute('tabindex', '2');
        const authoredNeg = fakeRichScroller('scroll-fade-x', { width: 900, client: 300 });
        authoredNeg.setAttribute('tabindex', '-1');
        const lateProject = fakeRichScroller('scroll-fade-x', { width: 900, client: 300, folded: true });
        const items = [overflowFirst, foldFirst, lostInHold, authored, authoredNeg, lateProject];
        const root = {
            nodeType: 1, isConnected: true, matches: () => false,
            contains: (node) => items.includes(node),
            querySelector: (sel) => items.find((el) => el.matches(sel)) || null,
            querySelectorAll: (sel) => (sel === '[data-fold-stop]'
                ? items.filter((el) => el.hasAttribute('data-fold-stop'))
                : sel.includes(',') ? [...items] : items.filter((el) => el.matches(sel))),
            setAttribute() {}, removeAttribute() {}, addEventListener() {}, removeEventListener() {},
        };
        const dispose = enhanceChatMarkdown(root);

        // Overflow before fold + late folded Project answer entering the same marked hold immediately.
        resized([{ target: overflowFirst }, { target: lateProject }]);
        assert.equal(overflowFirst.getAttribute('tabindex'), '0');
        assert.equal(lateProject.hasAttribute('data-fold-stop'), true);
        assert.equal(lateProject.getAttribute('tabindex'), '-1');

        // Updates-style fold stops every target; subsequent attribute/resize events cannot clobber tabindex=-1.
        syncFoldStops(root, true);
        written([{ type: 'attributes', target: overflowFirst, attributeName: 'tabindex' }]);
        resized([{ target: overflowFirst }, { target: foldFirst }, { target: authored }, { target: authoredNeg }]);
        for (const el of items) {
            assert.equal(el.hasAttribute('data-fold-stop'), true);
            assert.equal(el.getAttribute('tabindex'), '-1');
        }
        assert.equal(foldFirst.getAttribute('role'), 'region');

        // Overflow gained then lost during hold, plus authored scrollers losing overflow during hold.
        lostInHold.scrollWidth = 900;
        resized([{ target: lostInHold }]);
        lostInHold.scrollWidth = 200;
        authored.scrollWidth = 200;
        authoredNeg.scrollWidth = 200;
        resized([{ target: lostInHold }, { target: authored }, { target: authoredNeg }]);
        assert.equal(lostInHold.hasAttribute('role'), false);
        assert.equal(lostInHold.getAttribute('tabindex'), '-1');

        // Release restores each scroller's exact underlying tabindex (absent vs -1 vs 0 vs authored 2).
        syncFoldStops(root, false);
        assert.deepEqual(items.map((el) => el.hasAttribute('data-fold-stop')), items.map(() => false));
        assert.equal(overflowFirst.getAttribute('tabindex'), '0');
        assert.equal(foldFirst.getAttribute('tabindex'), '0');
        assert.equal(lateProject.getAttribute('tabindex'), '0');
        assert.deepEqual(Object.fromEntries(lostInHold.attributes), {});
        assert.deepEqual(Object.fromEntries(authored.attributes), { tabindex: '2' });
        assert.deepEqual(Object.fromEntries(authoredNeg.attributes), { tabindex: '-1' });
        dispose();
    } finally {
        globalThis.ResizeObserver = prior.ResizeObserver;
        globalThis.MutationObserver = prior.MutationObserver;
        globalThis.document = prior.document;
    }
});

test('a keyed scroller that changes shape is reconciled, and one that stops matching releases only its own state', () => {
    const prior = { ResizeObserver: globalThis.ResizeObserver, MutationObserver: globalThis.MutationObserver };
    let resized = null, written = null;
    const observed = new Set();
    globalThis.ResizeObserver = class {
        constructor(fn) { resized = fn; }
        observe(node) { observed.add(node); }
        unobserve(node) { observed.delete(node); }
        disconnect() { observed.clear(); }
    };
    globalThis.MutationObserver = class {
        constructor(fn) { written = fn; }
        observe() {}
        disconnect() {}
    };
    const code = fakeRichScroller('md-compact-code scroll-fade-x', { width: 900, client: 300 });
    const wrap = fakeTableWrap('patched');
    const items = new Set([code, wrap]);
    const root = {
        nodeType: 1, matches: () => false,
        contains: (node) => items.has(node),
        querySelector: (sel) => [...items].find((el) => el.matches(sel)) || null,
        querySelectorAll: (sel) => [...items].filter((el) => el.matches(sel)),
        setAttribute() {}, removeAttribute() {}, addEventListener() {}, removeEventListener() {},
    };
    try {
        const dispose = bindCompactMarkdown(root);
        resized([{ target: code }]);
        assert.equal(code.attributes.get('role'), 'region');
        // The keyed patch reuses an outer div: its class becomes a table wrap while its children are replaced.
        written([{ type: 'childList', target: wrap, addedNodes: [wrap.firstElementChild], removedNodes: [] }]);
        assert.equal(observed.has(wrap), true, 'the reused outer div is observed once it matches a scroller selector');
        assert.equal(observed.has(wrap.firstElementChild), true);
        resized([{ target: wrap }]);
        assert.deepEqual(Object.fromEntries(wrap.attributes), {
            'data-scroll-end': '', role: 'region', 'aria-label': 'Scrollable table', tabindex: '0',
        });
        // The old code scroller no longer matches: its owned region and edge marks go, its registration goes.
        code.matches = () => false;
        written([{ type: 'childList', target: code, addedNodes: [], removedNodes: [code.firstElementChild] }]);
        assert.equal(observed.has(code), false);
        assert.deepEqual(Object.fromEntries(code.attributes), {});
        dispose();
        assert.equal(observed.size, 0);
    } finally {
        globalThis.ResizeObserver = prior.ResizeObserver;
        globalThis.MutationObserver = prior.MutationObserver;
    }
});

test('one node reused as another wrapper keeps the region name it wrote current, and never takes or drops a caller\'s', () => {
    const prior = { ResizeObserver: globalThis.ResizeObserver, MutationObserver: globalThis.MutationObserver };
    let resized = null, written = null, watching = null;
    globalThis.ResizeObserver = class { constructor(fn) { resized = fn; } observe() {} unobserve() {} disconnect() {} };
    globalThis.MutationObserver = class {
        constructor(fn) { written = fn; }
        observe(node, options) { watching = options; }
        disconnect() {}
    };
    const scroller = (classes, authored = {}) => {
        const el = fakeRichScroller(classes, { width: 900, client: 300 });
        for (const [name, value] of Object.entries(authored)) el.setAttribute(name, value);
        return el;
    };
    const ours = scroller('md-mermaid-scroll scroll-fade-x');
    const authored = scroller('md-mermaid-scroll scroll-fade-x', { 'aria-label': 'Plans', role: 'figure', tabindex: '3' });
    const taken = scroller('md-mermaid-scroll scroll-fade-x');
    const items = [ours, authored, taken];
    const root = {
        nodeType: 1, isConnected: true, matches: () => false, contains: (node) => items.includes(node),
        querySelector: (sel) => items.find((el) => el.matches(sel)) || null,
        querySelectorAll: (sel) => items.filter((el) => el.matches(sel)),
        setAttribute() {}, removeAttribute() {}, addEventListener() {}, removeEventListener() {},
    };
    const region = (el) => Object.fromEntries([...el.attributes].filter(([name]) => !name.startsWith('data-scroll')));
    const reuseAsTable = (el) => {
        el.reclass('md-table-wrap scroll-fade-x');
        written([{ type: 'childList', target: el, addedNodes: [el.firstElementChild], removedNodes: [] }]);
    };
    try {
        const dispose = bindRenderedMarkdown(root);
        resized(items.map((target) => ({ target })));
        assert.deepEqual(region(ours), { role: 'region', 'aria-label': 'Scrollable diagram', tabindex: '0' });
        assert.deepEqual(region(authored), { 'aria-label': 'Plans', role: 'figure', tabindex: '3' }, 'a caller\'s name, role and tab stop are never ours');
        // The patch keeps the node and what was written on it: a table now, still wider than its box.
        reuseAsTable(ours); reuseAsTable(authored);
        resized([{ target: ours }, { target: authored }]);
        assert.deepEqual(region(ours), { role: 'region', 'aria-label': 'Scrollable table', tabindex: '0' }, 'the name follows the node: the same object');
        assert.deepEqual(region(authored), { 'aria-label': 'Plans', role: 'figure', tabindex: '3' });
        // A name a caller sets on a node that held ours is theirs from then on, on this pass and at release.
        taken.setAttribute('aria-label', 'Revenue by plan');
        reuseAsTable(taken);
        resized([{ target: taken }]);
        assert.equal(taken.getAttribute('aria-label'), 'Revenue by plan');
        // A patch that wipes the region attributes (the keyed attribute copy) still gets the node's present name back.
        ours.attributes.clear();
        written([{ type: 'attributes', target: ours, attributeName: 'tabindex' }]);
        assert.equal(ours.getAttribute('aria-label'), 'Scrollable table');
        // Ordinary again: only what was written goes, and the observed writes never include the name or the role.
        for (const el of items) el.scrollWidth = el.clientWidth;
        resized(items.map((target) => ({ target })));
        assert.deepEqual(region(ours), {});
        assert.deepEqual(region(authored), { 'aria-label': 'Plans', role: 'figure', tabindex: '3' });
        assert.deepEqual(region(taken), { 'aria-label': 'Revenue by plan' });
        assert.deepEqual(watching.attributeFilter, ['tabindex', 'data-scroll-start', 'data-scroll-end', 'data-scroll-rtl']);
        dispose();
    } finally {
        globalThis.ResizeObserver = prior.ResizeObserver;
        globalThis.MutationObserver = prior.MutationObserver;
    }
});

test('markdown horizontal scrollers reveal a keyboard-focused descendant through one root delegation', () => {
    const prior = { ResizeObserver: globalThis.ResizeObserver, MutationObserver: globalThis.MutationObserver, document: globalThis.document };
    let resized = null;
    globalThis.ResizeObserver = class {
        constructor(fn) { resized = fn; }
        observe() {}
        unobserve() {}
        disconnect() {}
    };
    globalThis.MutationObserver = class { constructor() {} observe() {} disconnect() {} };
    globalThis.document = { createElement: () => ({}) };
    const wrap = fakeRichScroller('md-table-wrap scroll-fade-x', { width: 900, client: 300 });
    wrap.ownerDocument.defaultView.getComputedStyle = () => ({
        direction: 'ltr', overflowX: 'auto',
        getPropertyValue: (name) => name === '--scroll-fade-inline-edge' ? '24px' : '',
    });
    wrap.getBoundingClientRect = () => ({ left: 0, right: 300, top: 0, bottom: 100 });
    let keyboard = false;
    const link = {
        matches: (sel) => sel === ':focus-visible' && keyboard,
        getBoundingClientRect: () => ({ left: 280, right: 295, top: 10, bottom: 30 }),
        closest: (sel) => sel === '.md-table-wrap' ? wrap : null,
        focus: () => listeners.get('focusin:false')({ target: link }),
    };
    const listeners = new Map();
    const root = {
        nodeType: 1, isConnected: true, matches: () => false,
        contains: (node) => node === wrap || node === link,
        querySelector: (sel) => sel === '.md-table-wrap' ? wrap : null,
        querySelectorAll: (sel) => sel === '.md-table-wrap' ? [wrap] : [],
        setAttribute() {}, removeAttribute() {},
        addEventListener: (type, fn, capture) => listeners.set(`${type}:${Boolean(capture)}`, fn),
        removeEventListener: (type, fn, capture) => {
            if (listeners.get(`${type}:${Boolean(capture)}`) === fn) listeners.delete(`${type}:${Boolean(capture)}`);
        },
    };
    try {
        const dispose = enhanceChatMarkdown(root);
        resized([{ target: wrap }]);
        const focusin = listeners.get('focusin:false');
        assert.equal(typeof focusin, 'function', 'one delegated focus listener serves the root’s scrollers');
        focusin({ target: link });
        assert.equal(wrap.scrollLeft, 0, 'pointer focus does not move the pressed row');
        keyboard = true;
        focusin({ target: link });
        assert.equal(wrap.scrollLeft, 19, 'Tab brings the link clear of the 24px end fade');
        wrap.scrollLeft = 0;
        restoreFocusWithoutReveal(link);
        assert.equal(wrap.scrollLeft, 0, 'reading-place restoration keeps the scroller where the reader left it');
        focusin({ target: wrap });
        assert.equal(wrap.scrollLeft, 0, 'the region itself is not a descendant row to reveal');
        dispose();
        assert.equal(listeners.has('focusin:false'), false);
    } finally {
        globalThis.ResizeObserver = prior.ResizeObserver;
        globalThis.MutationObserver = prior.MutationObserver;
        globalThis.document = prior.document;
    }
});
