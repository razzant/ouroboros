// Share's File seams (message_share_file.js, message_share.js, message_actions.js): a card's components as a file shows
// them (every tab a labelled section, every control that would press the card disabled, the card itself untouched), a
// file's name from its title, a card read once from its own body with an honest refusal where no file can carry it,
// nothing made for a sheet that closed meanwhile, and Share offered on a living card row the host recorded. Pages
// themselves are drawn in a real engine: tests/test_message_share_files_browser.py.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createMessageActions, messageRow, recordCardSource } from '../modules/message_actions.js';
import { SHARE, cardFile, registerMessageShare } from '../modules/message_share.js';
import { SHARE_APP_FILE_CSP, SHARE_FILE_CSP, drawingUnderWay, drawnAnswer, portableComponents, selectorParts,
    shareFileName } from '../modules/message_share_file.js';
import { renderChartDataTable } from '../modules/widget_chart.js';

test('a card’s components as a file shows them: each tab a labelled section, every press disabled, the card untouched', () => {
    const components = [
        { type: 'tabs', id: 'plan', target: 'result', tabs: [
            { label: 'Overview', components: [{ type: 'metric', label: 'Total', value: 70 }] },
            { components: [{ type: 'form', id: 'book', submit: { id: 'book' }, fields: [{ name: 'guests' }] }] },
        ] },
        { type: 'action', label: 'Confirm', submit: { id: 'confirm', once: true } },
        { type: 'poll', label: 'Refresh' },
        { type: 'kanban', columns: [{ id: 'todo' }], on_move: { route: '/move' } },
        { type: 'group', components: [{ type: 'action', label: 'Inner' }] },
        { type: 'subscription', render: [{ type: 'form', fields: [] }] },
    ];
    const before = JSON.stringify(components);
    const [tabs, action, poll, kanban, group, subscription] = portableComponents(components);
    assert.deepEqual(tabs, { id: 'plan', target: 'result', type: 'group', layout: 'stack', components: [
        { type: 'group', layout: 'stack', title: 'Overview', components: [{ type: 'metric', label: 'Total', value: 70 }] },
        { type: 'group', layout: 'stack', title: 'Tab 2', components: [
            { type: 'form', id: 'book', submit: { id: 'book' }, fields: [{ name: 'guests' }], disabled: true }] },
    ] });
    assert.equal(action.disabled, true);
    assert.equal(poll.disabled, true);
    assert.equal('on_move' in kanban, false, 'no move between columns leaves');
    assert.equal(group.components[0].disabled, true);
    assert.equal(subscription.render[0].disabled, true);
    assert.equal(JSON.stringify(components), before, 'the card’s own declaration is never edited');
    assert.deepEqual(portableComponents(null), []);
});

test('a file is named by its title, without what a file system refuses', () => {
    assert.equal(shareFileName('Day by the river'), 'Day by the river.html');
    assert.equal(shareFileName('Route budget', 'ouroboros-answer'), 'Route budget.html');
    assert.equal(shareFileName('Stay by the river: the hills are behind…', 'ouroboros-answer'),
        'Stay by the river the hills are behind.html', 'a cut title leaves its ellipsis out of the name');
    assert.equal(shareFileName('', 'ouroboros-answer'), 'ouroboros-answer.html');
    assert.equal(shareFileName('../plans/a:b?'), 'plans a b.html');
    assert.equal(shareFileName('   '), 'ouroboros-card.html');
    assert.equal(shareFileName(''), 'ouroboros-card.html');
    assert.equal(shareFileName('День у реки'), 'День у реки.html');
    assert.equal(Array.from(shareFileName('ю'.repeat(200))).length, 80 + '.html'.length);
});

test('a selector list splits at its own commas only, and a file’s policy loads nothing', () => {
    assert.deepEqual(selectorParts('a, b:is(c, d) > e,[x=","] , :not(.f,.g)'), ['a', 'b:is(c, d) > e', '[x=","]', ':not(.f,.g)']);
    assert.deepEqual(selectorParts(''), []);
    assert.match(SHARE_FILE_CSP, /^default-src 'none';/);
    assert.match(SHARE_FILE_CSP, /form-action 'none'/);
    assert.doesNotMatch(SHARE_FILE_CSP, /script-src|connect-src|https?:/);
    assert.match(SHARE_APP_FILE_CSP, /script-src 'unsafe-inline' blob: data:$/);
});

test('a chart’s data table keeps every value: points by series, x, y and r; a pair as its range; labels as before', () => {
    const rows = (html) => [...html.matchAll(/<tr><th scope="row">(.*?)<\/th>(.*?)<\/tr>/g)]
        .map(([, head, cells]) => [head, ...[...cells.matchAll(/<td[^>]*>(.*?)<\/td>/g)].map(([, cell]) => cell)]);
    const scatter = renderChartDataTable({ type: 'scatter', data: { datasets: [{ label: 'Walk', data: [{ x: 1, y: 3 }, { x: 2.5, y: 7 }] }] } }, 'chart', false);
    assert.match(scatter, /<thead><tr><th>Series<\/th><th>x<\/th><th>y<\/th><\/tr><\/thead>/);
    assert.deepEqual(rows(scatter), [['Walk', '1', '3'], ['Walk', '2.5', '7']]);
    const bubble = renderChartDataTable({ type: 'bubble', data: { datasets: [{ label: 'Stops', data: [{ x: 1, y: 2, r: 5 }, [3, 4]] }] } }, 'chart', false);
    assert.deepEqual(rows(bubble), [['Stops', '1', '2', '5'], ['Stops', '3', '4', '—']]);
    const tuple = renderChartDataTable({ type: 'bubble', data: { labels: ['Waypoint Alpha'], datasets: [{ label: 'Stops', data: [[1, 2, 5]] }] } }, 'chart', false);
    assert.match(tuple, /<thead><tr><th>Series<\/th><th>Label<\/th><th>x<\/th><th>y<\/th><th>r<\/th><\/tr><\/thead>/);
    assert.deepEqual(rows(tuple), [['Stops', 'Waypoint Alpha', '1', '2', '5']], 'a bubble’s [x, y, r] keeps its radius and its label');
    const line = renderChartDataTable({ type: 'line', data: { datasets: [{ label: 'Path', data: [[1, 2], 4] }] } }, 'chart', false);
    assert.deepEqual(rows(line), [['Path', '1', '2'], ['Path', '—', '4']], 'values without labels still leave');
    const floating = renderChartDataTable({ type: 'bar', data: { labels: ['Mon'], datasets: [{ label: 'Range', data: [[1, 5]] }] } }, 'chart', false);
    assert.deepEqual(rows(floating), [['Mon', '1 – 5']]);
    assert.equal(renderChartDataTable({ data: { labels: ['Museum', 'Cafe'], datasets: [{ label: 'Euro', data: [24, null] }] } }, 'Budget', true),
        '<details class="widget-chart-data" open><summary>View Budget data</summary><div class="widget-table-wrap"><table class="widget-table">'
        + '<thead><tr><th>Label</th><th>Euro</th></tr></thead><tbody><tr><th scope="row">Museum</th><td data-label="Euro">24</td></tr>'
        + '<tr><th scope="row">Cafe</th><td data-label="Euro">—</td></tr></tbody></table></div></details>', 'labelled numbers read as before');
    assert.doesNotMatch(renderChartDataTable({ data: { labels: [], datasets: [] } }, 'chart', false), /Series/, 'an empty chart keeps its labelled head');
});

// A drawn message's diagram and chart nodes as chat_markdown.js leaves them, read by what the File waits for.
const node = ({ svg = false, canvas = null, processed = '' } = {}) => ({
    dataset: processed ? { processed } : {},
    querySelector: (selector) => (selector === 'svg' ? (svg ? {} : null) : selector === 'canvas' ? canvas : null),
});
const content = (diagrams, charts) => ({
    querySelectorAll: (selector) => (selector === '.md-mermaid' ? diagrams : selector === '.md-chart' ? charts : []),
});

test('a File waits for the room’s own drawing: a diagram until its SVG, a chart until drawn and done animating', () => {
    const canvas = {};
    const running = new Set();
    const Chart = { getChart: (node) => (node === canvas ? { canvas } : null), animator: { running: (chart) => running.has(chart.canvas) } };
    assert.deepEqual(drawingUnderWay(content([node()], [node()]), { Chart }), ['diagram', 'chart']);
    assert.deepEqual(drawingUnderWay(content([node({ svg: true })], [node({ processed: 'true', canvas })]), { Chart }), []);
    running.add(canvas);
    assert.deepEqual(drawingUnderWay(content([], [node({ processed: 'true', canvas })]), { Chart }), ['chart'], 'still animating');
    assert.deepEqual(drawingUnderWay(content([], [node({ processed: 'true' })]), { Chart }), [], 'a refused chart is drawn as it will be');
    assert.deepEqual(drawingUnderWay(null), []);
});

test('the wait ends when the drawing is done, at its bound, or when the sheet is gone, and says what is still drawing', async () => {
    let looks = 0;
    assert.deepEqual(await drawnAnswer({}, { step: 1, underWay: () => (++looks < 3 ? ['diagram'] : []) }), []);
    assert.equal(looks, 3);
    assert.deepEqual(await drawnAnswer({}, { step: 1, wait: 15, underWay: () => ['chart'] }), ['chart'], 'the bound passes');
    let open = true;
    looks = 0;
    const gone = await drawnAnswer({}, { step: 1, alive: () => open, underWay: () => { looks += 1; open = looks < 2; return ['diagram']; } });
    assert.deepEqual(gone, ['diagram']);
    assert.equal(looks, 2, 'a closed sheet stops looking');
});

function pages(calls) {
    return {
        componentsPage: async (body, options) => { calls.push(['components', body.card_id, options.material]); return { html: '<p>c</p>', title: body.title }; },
        appPage: (body, html, options) => { calls.push(['app', body.card_id, html, options.material]); return { html: `<p>${html}</p>`, title: body.title }; },
    };
}

test('a card is read once from its own body: components and an app become their page, a skill widget never does', async () => {
    const calls = [];
    const reads = [];
    const bodies = {
        trip: { card_id: 'trip', kind: 'components', title: 'Trip plan' },
        river: { card_id: 'river', kind: 'app', title: 'Day by the river', app: { url: '/api/cards/river/app', sha256: 'aa' } },
        bare: { card_id: 'bare', kind: 'app', title: 'No app' },
        weather: { card_id: 'weather', kind: 'widget', title: 'Weather', widget: { skill: 'weather', tab_id: 'now' } },
        odd: { card_id: 'odd', kind: 'poem', title: 'Odd' },
    };
    const readCard = async (id, revision) => { reads.push([id, revision]); return bodies[id]; };
    const readApp = async (url, sha) => { reads.push([url, sha]); return 'APP'; };
    const node = { isConnected: true };
    const made = (id, extra = {}) => cardFile({ id, revision: 3, node, ...extra }, { readCard, readApp, pages: pages(calls) });
    assert.deepEqual(await made('trip'), { html: '<p>c</p>', title: 'Trip plan', kind: 'components', name: 'Trip plan.html', scripted: false });
    assert.deepEqual(await made('river'), { html: '<p>APP</p>', title: 'Day by the river', kind: 'app', name: 'Day by the river.html', scripted: true });
    assert.deepEqual(await made('bare'), { refused: 'unsupported' });
    assert.deepEqual(await made('weather'), { refused: 'widget' });
    assert.deepEqual(await made('odd'), { refused: 'unsupported' });
    assert.deepEqual(reads, [['trip', 3], ['river', 3], ['/api/cards/river/app', 'aa'], ['bare', 3], ['weather', 3], ['odd', 3]]);
    assert.deepEqual(calls, [['components', 'trip', node], ['app', 'river', 'APP', node]], 'the room’s colours come from the card’s own node');
    calls.length = 0;
    await cardFile({ id: 'trip', node: { isConnected: false } }, { readCard, readApp, pages: pages(calls) });
    assert.deepEqual(calls, [['components', 'trip', null]], 'a node no longer in the room lends no colours');
});

test('a card that cannot be read says why, and a sheet closed meanwhile has nothing made for it', async () => {
    const calls = [];
    const gone = Object.assign(new Error('card not found'), { status: 404 });
    assert.deepEqual(await cardFile({ id: 'x' }, { readCard: async () => { throw gone; }, pages: pages(calls) }), { refused: 'gone', reason: 'card not found' });
    assert.deepEqual(await cardFile({ id: 'x' }, { readCard: async () => { throw new Error('HTTP 503'); }, pages: pages(calls) }),
        { refused: 'unreadable', reason: 'HTTP 503' });
    const app = { card_id: 'river', kind: 'app', app: { url: '/u', sha256: 's' } };
    assert.deepEqual(await cardFile({ id: 'river' }, { readCard: async () => app, readApp: async () => { throw new Error('HTTP 404'); }, pages: pages(calls) }),
        { refused: 'unreadable', reason: 'HTTP 404' });
    let open = true;
    const closing = cardFile({ id: 'trip' }, { readCard: async () => { open = false; return { kind: 'components' }; }, alive: () => open, pages: pages(calls) });
    assert.deepEqual(await closing, { refused: 'closed' });
    open = true;
    const late = cardFile({ id: 'river' }, { readCard: async () => app, readApp: async () => { open = false; return 'APP'; }, alive: () => open, pages: pages(calls) });
    assert.deepEqual(await late, { refused: 'closed' });
    assert.deepEqual(calls, [], 'nothing is drawn for a sheet that is gone');
});

function bubble(role, address = 'row:1@2026-11-10T10:00:00Z#cccccccccccc') {
    const classes = new Set(['chat-bubble', role]);
    return { dataset: { rowAddress: address, rowWords: 'Trip plan words' }, classList: { contains: (name) => classes.has(name) },
        querySelector: () => null };
}

test('Share applies to a living card row the host recorded, while its card is still that row’s', () => {
    const registry = createMessageActions();
    const win = { addEventListener() {}, removeEventListener() {} };
    const off = registerMessageShare({ registry, win, doc: {} });
    const card = bubble('assistant');
    let current = { id: 'trip', revision: 3, title: 'Trip plan', words: 'Trip plan words', node: null };
    let held = null;
    recordCardSource(card, () => current, (close) => { held = close; return () => { held = null; }; });
    const row = messageRow(card);
    assert.equal(row.answerWords, '', 'a card has no answer words: its words are its own');
    assert.deepEqual(row.card, current);
    assert.equal(typeof row.holdTransient, 'function');
    assert.deepEqual(registry.actionsFor(row).map((action) => action.key), [SHARE]);
    current = null;  // released or drawn again: no longer this row's card
    assert.equal(messageRow(card).card, null);
    assert.deepEqual(registry.actionsFor(messageRow(card)), []);
    recordCardSource(card, () => { throw new Error('gone'); });
    assert.equal(messageRow(card).card, null, 'a description that fails is no card');
    assert.equal(held, null, 'nothing is held before a press');
    off();
});
