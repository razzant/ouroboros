import assert from 'node:assert/strict';
import test from 'node:test';

import {
    chatMarkdownUrl,
    enhanceChatMarkdown,
    parseChartConfig,
    prepareMarkdownSource,
    renderChatMarkdown,
} from '../modules/chat_markdown.js';

test('prepareMarkdownSource decodes one entity layer and escapes raw text', () => {
    const cases = [
        ['<tag>&', '&lt;tag&gt;&amp;'],
        ['&lt;tag&gt;&amp;', '&lt;tag&gt;&amp;'],
        ['&amp;lt;tag&amp;gt;&amp;amp;', '&amp;lt;tag&amp;gt;&amp;amp;'],
        ['Fish & chips &amp; tea', 'Fish &amp; chips &amp; tea'],
        ['`<b>&lt;x&gt;&`', '`&lt;b&gt;&lt;x&gt;&amp;`'],
    ];
    for (const [source, expected] of cases) {
        assert.equal(prepareMarkdownSource(source), expected, source);
    }
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
