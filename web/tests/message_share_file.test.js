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
import { parseChartConfig } from '../modules/chat_markdown.js';
import { renderChartDataTable, renderDrawnChartDataTable } from '../modules/widget_chart.js';
// The bundled Chart.js the room draws with (it sets `globalThis.Chart`): mounted below on a canvas that paints nothing.
import '../chart.umd.min.js';

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

const drawnRows = (html) => [...html.matchAll(/<tr><th scope="row">(.*?)<\/th>(.*?)<\/tr>/g)]
    .map(([, head, cells]) => [head, ...[...cells.matchAll(/<td[^>]*>(.*?)<\/td>/g)].map(([, cell]) => cell)]);

// The chart as an answer's Markdown draws it: the production sanitizer's configuration on the bundled Chart.js, mounted on a
// canvas that paints nothing, so Share reads the real scales and controllers. A configuration that cannot draw does not
// mount; each case below is a chart a reader of the answer sees.
// `painted` (an array) collects every text the chart writes on the canvas: what the room's chart shows, apart from any table.
const canvasOf = (painted = null) => {
    const canvas = { width: 600, height: 300, style: {}, addEventListener() {}, removeEventListener() {},
        getBoundingClientRect: () => ({ width: 600, height: 300 }), getContext: () => canvas.context };
    canvas.context = new Proxy({ canvas, measureText: () => ({ width: 8 }), getLineDash: () => [],
        createLinearGradient: () => ({ addColorStop() {} }), ...(painted && { fillText: (text) => painted.push(String(text)) }) }, {
        get: (target, key) => (key in target ? target[key] : () => {}),
        set: (target, key, value) => { target[key] = value; return true; },
    });
    return canvas;
};
const markdownChart = (t, source, painted = null) => {
    const config = parseChartConfig(JSON.stringify(source));
    const chart = new globalThis.Chart(canvasOf(painted), { ...config, options: { ...config.options, responsive: false, animation: false } });
    t.after(() => chart.destroy());
    return chart;
};
const drawnTable = (t, source) => renderDrawnChartDataTable(markdownChart(t, source), 'chart');
// What the chart drew, series by series: each point's category (the label at its parsed index) and its value or interval.
const drawnFacts = (chart) => chart.data.datasets.map((dataset, index) => {
    const { controller, iScale, vScale } = chart.getDatasetMeta(index);
    return dataset.data.map((_, point) => {
        const parsed = controller.getParsed(point);
        return [iScale.getLabels()[parsed[iScale.axis]] ?? null,
            parsed._custom ? `${parsed._custom.start} – ${parsed._custom.end}` : parsed[vScale.axis]];
    });
});
const tableHead = (html) => [...html.match(/<thead><tr>(.*?)<\/tr><\/thead>/)[1].matchAll(/<th>(.*?)<\/th>/g)].map(([, cell]) => cell);
const bars = (options, data, datasets = [{ label: 'Hours', data: [[8, 12], [13, 17]] }]) => ({ type: 'bar', options, data: { ...data, datasets } });
const lines = (options, data, datasets) => ({ type: 'line', options, data: { ...data, datasets } });

test('Share names a drawn floating bar by the category its index scale holds, on either axis', (t) => {
    const days = () => ['Monday', 'Tuesday'];
    const drawn = {
        'a vertical axis names its categories': bars({ scales: { x: { type: 'category', labels: days() } } }),
        'a horizontal axis names its categories': bars({ indexAxis: 'y', scales: { y: { type: 'category', labels: days() } } }),
        'the data names them': bars({}, { labels: days() }),
        'the data names them on a horizontal axis': bars({ indexAxis: 'y' }, { labels: days() }),
    };
    for (const [name, config] of Object.entries(drawn)) {
        const html = drawnTable(t, config);
        assert.deepEqual(tableHead(html), ['Label', 'Hours'], name);
        assert.deepEqual(drawnRows(html), [['Monday', '8 – 12'], ['Tuesday', '13 – 17']], name);
    }
});

test('Share keeps the categories the axis drew where the data names others, for every series', (t) => {
    const html = drawnTable(t, bars({ scales: { x: { type: 'category', labels: ['Monday', 'Tuesday'] } } }, { labels: ['A', 'B'] },
        [{ label: 'Shift', data: [[8, 12], [13, 17]] }, { label: 'Break', data: [[12, 13], [10, 11]] }]));
    assert.deepEqual(tableHead(html), ['Label', 'Shift', 'Break']);
    assert.deepEqual(drawnRows(html), [['Monday', '8 – 12', '12 – 13'], ['Tuesday', '13 – 17', '10 – 11']]);
});

test('Share exports an interval as the room parsed it, its endpoints in the order given, a missing end a gap', (t) => {
    const html = drawnTable(t, bars({ scales: { x: { type: 'category', labels: ['Mon', 'Tue', 'Wed', 'Thu'] } } }, {},
        [{ label: 'Hours', data: [['8', '12'], [null, 17], [15, 9]] }]));
    assert.deepEqual(drawnRows(html), [['Mon', '8 – 12'], ['Tue', '— – 17'], ['Wed', '15 – 9'], ['Thu', '—']],
        'text is read as a number, a missing end is a gap, a category without a value stays a row');
});

test('Share names a drawn bar or line of plain values by the categories its axis holds, zero and gap included', (t) => {
    const axis = (labels) => ({ scales: { x: { type: 'category', labels } } });
    const euro = drawnTable(t, bars(axis(['Museum', 'Cafe', 'Tram']), {}, [{ label: 'Euro', data: [24, 0, null] }]));
    assert.deepEqual(tableHead(euro), ['Label', 'Euro']);
    assert.deepEqual(drawnRows(euro), [['Museum', '24'], ['Cafe', '0'], ['Tram', '—']]);
    const visits = drawnTable(t, { type: 'line', options: axis(['Mon', 'Tue']), data: { datasets: [{ label: 'Visits', data: [3, 5] }] } });
    assert.deepEqual(drawnRows(visits), [['Mon', '3'], ['Tue', '5']]);
});

test('Share keeps a floating bar of records as its interval at the category each record names', (t) => {
    const vertical = drawnTable(t, bars({}, {}, [{ label: 'Hours', data: [{ x: 'Monday', y: [8, 12] }, { x: 'Tuesday', y: [13, 17] }] }]));
    assert.deepEqual(tableHead(vertical), ['Series', 'Label', 'x', 'y']);
    assert.deepEqual(drawnRows(vertical), [['Hours', 'Monday', 'Monday', '8 – 12'], ['Hours', 'Tuesday', 'Tuesday', '13 – 17']]);
    const horizontal = drawnTable(t, bars({ indexAxis: 'y' }, {}, [{ label: 'Hours', data: [{ y: 'Monday', x: [8, 12] }, { y: 'Tuesday', x: [13, 17] }] }]));
    assert.deepEqual(drawnRows(horizontal), [['Hours', 'Monday', '8 – 12', 'Monday'], ['Hours', 'Tuesday', '13 – 17', 'Tuesday']]);
});

test('Share keeps ranges by place beside records of the other order at the category and endpoints each was drawn at, on either axis', (t) => {
    // Chart.js draws the ranges at Monday, Tuesday by their place and the records at Tuesday, Monday by what each names.
    const drawn = { Ranges: [['Monday', '8 – 12'], ['Tuesday', '13 – 17']], Records: [['Tuesday', '21 – 23'], ['Monday', '1 – 3']] };
    for (const [index, value] of [['x', 'y'], ['y', 'x']]) {
        for (const order of [['Ranges', 'Records'], ['Records', 'Ranges']]) {
            const where = `${index} axis, ${order.join(' then ')}`;
            const series = { Ranges: { label: 'Ranges', data: [[8, 12], [13, 17]] },
                Records: { label: 'Records', data: [{ [index]: 'Tuesday', [value]: [21, 23] }, { [index]: 'Monday', [value]: [1, 3] }] } };
            const chart = markdownChart(t, { type: 'bar', options: { indexAxis: index, scales: { [index]: { type: 'category', labels: ['Monday', 'Tuesday'] } } },
                data: { datasets: order.map((label) => series[label]) } });
            assert.deepEqual(drawnFacts(chart), order.map((label) => drawn[label]), `${where}: as Chart.js drew them`);
            const html = renderDrawnChartDataTable(chart, 'chart');
            const row = (label, category, range) => (index === 'x' ? [label, category, range] : [label, range, category]);
            assert.deepEqual(tableHead(html), ['Series', 'x', 'y'], `${where}: no label by place, each row says its own category`);
            assert.deepEqual(drawnRows(html), order.flatMap((label) => drawn[label].map(([category, range]) => row(label, category, range))),
                `${where}: the category on the index axis, the interval on the value axis`);
        }
    }
});

test('Share gives no row of mixed series its category by place, whichever labels the chart names, and a gap stays a row', (t) => {
    const chart = markdownChart(t, { type: 'bar', data: { labels: ['Monday', 'Tuesday'], datasets: [
        { label: 'Values', data: [24, null] }, { label: 'Records', data: [{ x: 'Tuesday', y: 5 }, { x: 'Monday', y: 7 }] }] } });
    assert.deepEqual(drawnFacts(chart), [[['Monday', 24], ['Tuesday', null]], [['Tuesday', 5], ['Monday', 7]]]);
    const html = renderDrawnChartDataTable(chart, 'chart');
    assert.deepEqual(tableHead(html), ['Series', 'x', 'y']);
    assert.deepEqual(drawnRows(html), [['Values', 'Monday', '24'], ['Values', 'Tuesday', '—'], ['Records', 'Tuesday', '5'], ['Records', 'Monday', '7']]);
});

test('Share keeps a line’s pairs as x and y where its axis names categories: a pair is a point, never the label of its place', (t) => {
    const chart = markdownChart(t, { type: 'line', options: { scales: { x: { type: 'category', labels: ['Monday', 'Tuesday'] } } },
        data: { datasets: [{ label: 'Visits', data: [['Tuesday', 21], ['Monday', 7]] }] } });
    assert.deepEqual(drawnFacts(chart), [[['Tuesday', 21], ['Monday', 7]]], 'drawn at Tuesday, then Monday');
    const html = renderDrawnChartDataTable(chart, 'chart');
    assert.deepEqual(tableHead(html), ['Series', 'x', 'y']);
    assert.deepEqual(drawnRows(html), [['Visits', 'Tuesday', '21'], ['Visits', 'Monday', '7']]);
});

test('Share names a line’s records and pairs by the category Chart.js drew each at, never by the label of its place', (t) => {
    const days = ['Monday', 'Tuesday'];
    const records = { label: 'Visits', data: [{ x: 'Tuesday', y: 21 }, { x: 'Monday', y: 7 }] };
    const cases = [
        ['records, the data names the days', lines({}, { labels: days }, [records]), true],
        ['pairs, the data names the days', lines({}, { labels: days }, [{ label: 'Visits', data: [['Tuesday', 21], ['Monday', 7]] }]), true],
        ['records, the axis names the days', lines({ scales: { x: { type: 'category', labels: days } } }, {}, [records]), true],
        ['records on a horizontal axis', lines({ indexAxis: 'y' }, { labels: days }, [{ label: 'Visits', data: [{ y: 'Tuesday', x: 21 }, { y: 'Monday', x: 7 }] }]), false],
    ];
    for (const [name, config, vertical] of cases) {
        const chart = markdownChart(t, config);
        assert.deepEqual(drawnFacts(chart), [[['Tuesday', 21], ['Monday', 7]]], `${name}: drawn at Tuesday, then Monday`);
        const html = renderDrawnChartDataTable(chart, 'chart');
        assert.deepEqual(tableHead(html), ['Series', 'x', 'y'], `${name}: the days in place name no row`);
        assert.deepEqual(drawnRows(html), vertical ? [['Visits', 'Tuesday', '21'], ['Visits', 'Monday', '7']] : [['Visits', '21', 'Tuesday'], ['Visits', '7', 'Monday']],
            `${name}: the category on the index axis, the value on the other`);
    }
});

test('Share keeps the values of a mixed line at the category Chart.js drew them, beside records, whichever labels name the days', (t) => {
    const days = ['Monday', 'Tuesday'];
    const sources = { 'the data names the days': [{}, { labels: days }], 'the axis names the days': [{ scales: { x: { type: 'category', labels: days } } }, {}] };
    const records = {
        'records away from their place': [[['Tuesday', 5], ['Monday', 7]], [['Records', 'Tuesday', '5'], ['Records', 'Monday', '7']]],
        'records at their place': [[['Monday', 5], ['Tuesday', 6]], [['Records', 'Monday', '5'], ['Records', 'Tuesday', '6']]],
    };
    for (const [source, [options, data]] of Object.entries(sources)) {
        for (const [order, [points, rows]] of Object.entries(records)) {
            const chart = markdownChart(t, lines(options, data, [{ label: 'Values', data: [24, null] },
                { label: 'Records', data: points.map(([x, y]) => ({ x, y })) }]));
            const where = `${source}, ${order}`;
            assert.deepEqual(drawnFacts(chart), [[['Monday', 24], ['Tuesday', null]], points], `${where}: as Chart.js drew them`);
            const html = renderDrawnChartDataTable(chart, 'chart');
            assert.deepEqual(tableHead(html), ['Series', 'x', 'y'], `${where}: each row says its own category`);
            assert.deepEqual(drawnRows(html), [['Values', 'Monday', '24'], ['Values', 'Tuesday', '—'], ...rows], where);
        }
    }
});

test('Share keeps the label of a place where every point lies at it, as the axis holds it, and names the category each point was drawn at where the labels cannot', (t) => {
    const days = ['Monday', 'Tuesday'];
    const visits = [{ label: 'Visits', data: [{ x: 'Monday', y: 7 }, { x: 'Tuesday', y: 21 }] }];
    const inPlace = drawnTable(t, lines({}, { labels: days }, visits));
    assert.deepEqual(tableHead(inPlace), ['Series', 'Label', 'x', 'y'], 'records at their own place keep the label by place');
    assert.deepEqual(drawnRows(inPlace), [['Visits', 'Monday', 'Monday', '7'], ['Visits', 'Tuesday', 'Tuesday', '21']]);
    // The axis names the days and the data other labels: the column is what the axis drew, never the data's.
    const onAxis = markdownChart(t, lines({ scales: { x: { type: 'category', labels: days } } }, { labels: ['A', 'B'] }, visits));
    assert.deepEqual(drawnFacts(onAxis), [[['Monday', 7], ['Tuesday', 21]]], 'drawn at the axis’s days');
    const onAxisTable = renderDrawnChartDataTable(onAxis, 'chart');
    assert.deepEqual(tableHead(onAxisTable), ['Series', 'Label', 'x', 'y']);
    assert.deepEqual(drawnRows(onAxisTable), [['Visits', 'Monday', 'Monday', '7'], ['Visits', 'Tuesday', 'Tuesday', '21']]);
    // The axis alone names them: no column, as before.
    const axisOnly = drawnTable(t, lines({ scales: { x: { type: 'category', labels: days } } }, {}, visits));
    assert.deepEqual(tableHead(axisOnly), ['Series', 'x', 'y']);
    assert.deepEqual(drawnRows(axisOnly), [['Visits', 'Monday', '7'], ['Visits', 'Tuesday', '21']]);
    // Chart.js finds one category in two records at Monday, so the place of the second holds none.
    const twice = drawnTable(t, lines({}, {}, [{ label: 'Visits', data: [{ x: 'Monday', y: 1 }, { x: 'Monday', y: 2 }] }]));
    assert.deepEqual(tableHead(twice), ['Series', 'x', 'y']);
    assert.deepEqual(drawnRows(twice), [['Visits', 'Monday', '1'], ['Visits', 'Monday', '2']]);
    // Floating bars of records, the data naming the days in the other order.
    const hours = drawnTable(t, bars({}, { labels: days }, [{ label: 'Hours', data: [{ x: 'Tuesday', y: [13, 17] }, { x: 'Monday', y: [8, 12] }] }]));
    assert.deepEqual(tableHead(hours), ['Series', 'x', 'y']);
    assert.deepEqual(drawnRows(hours), [['Hours', 'Tuesday', '13 – 17'], ['Hours', 'Monday', '8 – 12']]);
});

test('Share keeps a bubble’s radius where its category axis draws a record away from the label of its place', (t) => {
    const chart = markdownChart(t, { type: 'bubble', options: { scales: { x: { type: 'category' } } },
        data: { labels: ['Mon', 'Tue'], datasets: [{ label: 'Stops', data: [{ x: 'Tue', y: 3, r: 5 }, { x: 'Mon', y: 4, r: 6 }] }] } });
    const { controller, data, iScale } = chart.getDatasetMeta(0);
    assert.deepEqual([0, 1].map((point) => iScale.getLabels()[controller.getParsed(point).x]), ['Tue', 'Mon'], 'drawn at Tue, then Mon');
    assert.deepEqual(data.map((point) => point.options.radius), [5, 6], 'with the radius each names');
    const html = renderDrawnChartDataTable(chart, 'chart');
    assert.deepEqual(tableHead(html), ['Series', 'x', 'y', 'r']);
    assert.deepEqual(drawnRows(html), [['Stops', 'Tue', '3', '5'], ['Stops', 'Mon', '4', '6']]);
});

// A category scale on the value axis too (a level axis: Low, High): Chart.js turns what it is given into a place on that
// scale and writes the label there. Share names each point by the labels its mounted scales hold at the places the chart drew
// it, on both axes, never by the index of a place or by the number it was given (Chart.js reads a number on such a scale as the
// place of its own point, whatever the number says).
const levels = (extra = {}) => ({ scales: { x: { type: 'category', labels: ['Mon', 'Tue'] }, y: { type: 'category', labels: ['Low', 'High'] }, ...extra } });
// What each point was drawn at, as its mounted scales name it: its category on the index axis, its category on the value axis.
const drawnLevels = (chart) => chart.data.datasets.map((dataset, index) => {
    const { controller, iScale, vScale } = chart.getDatasetMeta(index);
    return dataset.data.map((_, point) => {
        const parsed = controller.getParsed(point);
        return [iScale.getLabels()[parsed[iScale.axis]] ?? null, vScale.getLabels()[parsed[vScale.axis]] ?? null];
    });
});

test('Share names a point by the categories both of its mounted scales drew it at, a level axis included', (t) => {
    const records = [{ x: 'Tue', y: 'High' }, { x: 'Mon', y: 'Low' }];
    const cases = {
        'records line': [{ type: 'line', options: levels(), data: { datasets: [{ label: 'Level', data: records }] } }, ['Series', 'x', 'y']],
        'records bar': [{ type: 'bar', options: levels(), data: { datasets: [{ label: 'Level', data: records }] } }, ['Series', 'x', 'y']],
        'pairs line': [{ type: 'line', options: levels(), data: { datasets: [{ label: 'Level', data: [['Tue', 'High'], ['Mon', 'Low']] }] } }, ['Series', 'x', 'y']],
        'records scatter': [{ type: 'scatter', options: levels(), data: { datasets: [{ label: 'Level', data: records }] } }, ['Series', 'x', 'y']],
    };
    for (const [name, [config, head]] of Object.entries(cases)) {
        const painted = [];
        const chart = markdownChart(t, config, painted);
        assert.deepEqual(drawnLevels(chart), [[['Tue', 'High'], ['Mon', 'Low']]], `${name}: drawn at Tue/High, then Mon/Low`);
        assert.deepEqual(['Low', 'High', 'Mon', 'Tue'].filter((word) => painted.includes(word)), ['Low', 'High', 'Mon', 'Tue'], `${name}: the chart wrote every label`);
        const html = renderDrawnChartDataTable(chart, 'chart');
        assert.deepEqual([tableHead(html), drawnRows(html)], [head, [['Level', 'Tue', 'High'], ['Level', 'Mon', 'Low']]], name);
    }
    // The index axis may be the vertical one: the value axis is then the horizontal.
    const horizontal = markdownChart(t, { type: 'bar', options: { indexAxis: 'y', scales: { y: { type: 'category', labels: ['Mon', 'Tue'] },
        x: { type: 'category', labels: ['Low', 'High'] } } }, data: { datasets: [{ label: 'Level', data: [{ y: 'Tue', x: 'High' }, { y: 'Mon', x: 'Low' }] }] } });
    assert.deepEqual(drawnLevels(horizontal), [[['Tue', 'High'], ['Mon', 'Low']]]);
    assert.deepEqual(drawnRows(renderDrawnChartDataTable(horizontal, 'chart')), [['Level', 'High', 'Tue'], ['Level', 'Low', 'Mon']], 'x is the value, y the index');
});

test('Share keeps a gap, a radius and the other series of a chart whose value axis names levels', (t) => {
    const gap = markdownChart(t, { type: 'line', options: levels({ x: { type: 'category', labels: ['Mon', 'Tue', 'Wed'] } }),
        data: { datasets: [{ label: 'Level', data: [{ x: 'Tue', y: 'High' }, { x: 'Mon', y: null }, { x: 'Wed', y: 'Low' }] }] } });
    assert.deepEqual(drawnLevels(gap), [[['Tue', 'High'], ['Mon', null], ['Wed', 'Low']]]);
    assert.deepEqual(drawnRows(renderDrawnChartDataTable(gap, 'chart')), [['Level', 'Tue', 'High'], ['Level', 'Mon', '—'], ['Level', 'Wed', 'Low']],
        'a point with no level is a row with no level, never an index');
    const bubble = markdownChart(t, { type: 'bubble', options: levels(), data: { datasets: [{ label: 'Level', data: [{ x: 'Tue', y: 'High', r: 5 }, { x: 'Mon', y: 'Low', r: 7 }] }] } });
    assert.deepEqual(drawnLevels(bubble), [[['Tue', 'High'], ['Mon', 'Low']]]);
    const html = renderDrawnChartDataTable(bubble, 'chart');
    assert.deepEqual([tableHead(html), drawnRows(html)], [['Series', 'x', 'y', 'r'], [['Level', 'Tue', 'High', '5'], ['Level', 'Mon', 'Low', '7']]], 'the radius stays beside the two categories');
    // Plain values beside records: every series is a row of the place it was drawn at.
    const mixed = markdownChart(t, { type: 'bar', options: { scales: { y: { type: 'category', labels: ['Low', 'High'] } } },
        data: { labels: ['Mon', 'Tue'], datasets: [{ label: 'A', data: ['High', 'Low'] }, { label: 'B', data: [{ x: 'Tue', y: 'High' }, { x: 'Mon', y: 'Low' }] }] } });
    assert.deepEqual(drawnLevels(mixed), [[['Mon', 'High'], ['Tue', 'Low']], [['Tue', 'High'], ['Mon', 'Low']]]);
    assert.deepEqual(drawnRows(renderDrawnChartDataTable(mixed, 'chart')),
        [['A', 'Mon', 'High'], ['A', 'Tue', 'Low'], ['B', 'Tue', 'High'], ['B', 'Mon', 'Low']]);
});

test('Share names a level axis’s points by the place Chart.js drew them at, where it was given numbers, not by the numbers', (t) => {
    // A number on a category scale is no label: Chart.js places the point of its own position (here Low, then High).
    const plain = markdownChart(t, { type: 'line', options: { scales: { y: { type: 'category', labels: ['Low', 'High'] } } },
        data: { labels: ['Mon', 'Tue'], datasets: [{ label: 'Level', data: [1, 0] }] } });
    assert.deepEqual(drawnLevels(plain), [[['Mon', 'Low'], ['Tue', 'High']]]);
    assert.deepEqual(drawnRows(renderDrawnChartDataTable(plain, 'chart')), [['Mon', 'Low'], ['Tue', 'High']]);
    const records = markdownChart(t, { type: 'line', options: levels(), data: { datasets: [{ label: 'Level', data: [{ x: 'Mon', y: 1 }, { x: 'Tue', y: 0 }] }] } });
    assert.deepEqual(drawnLevels(records), [[['Mon', 'Low'], ['Tue', 'High']]]);
    assert.deepEqual(drawnRows(renderDrawnChartDataTable(records, 'chart')), [['Level', 'Mon', 'Low'], ['Level', 'Tue', 'High']]);
    // A value axis that is linear keeps the numbers it was given, and so does an interval.
    const linear = markdownChart(t, { type: 'line', options: { scales: { x: { type: 'category', labels: ['Mon', 'Tue'] } } },
        data: { datasets: [{ label: 'Visits', data: [{ x: 'Tue', y: 21 }, { x: 'Mon', y: 7 }] }] } });
    assert.deepEqual(drawnRows(renderDrawnChartDataTable(linear, 'chart')), [['Visits', 'Tue', '21'], ['Visits', 'Mon', '7']]);
});

// A radar's radial scale holds the label of each spoke (its own `labels`, else the data's) and writes it round the chart;
// a polar area has the same scale and writes none of them (its names are the data's, in the legend).
test('Share names a drawn radar by the spoke labels its radial scale holds and writes, never by the data’s, keeping its values and the chart as they were', (t) => {
    const config = { type: 'radar', options: { scales: { r: { labels: ['North', 'West', 'East'] } } },
        data: { labels: ['A', 'B', 'C'], datasets: [{ label: 'Wind', data: [3, 5, 2] }, { label: 'Gust', data: [4, 0, null] }] } };
    const painted = [];
    const chart = markdownChart(t, config, painted);
    const written = (words) => words.filter((word) => painted.includes(word));
    const { iScale } = chart.getDatasetMeta(0);
    assert.deepEqual([iScale.type, iScale.getLabels()], ['radialLinear', ['North', 'West', 'East']], 'the mounted scale holds the spokes');
    assert.deepEqual([written(['North', 'West', 'East']).length, written(['A', 'B', 'C'])], [3, []], 'and the chart wrote them, none of the data’s');
    const before = JSON.stringify([chart.config.data, chart.config.options]);
    const html = renderDrawnChartDataTable(chart, 'chart');
    assert.deepEqual(tableHead(html), ['Label', 'Wind', 'Gust']);
    assert.deepEqual(drawnRows(html), [['North', '3', '4'], ['West', '5', '0'], ['East', '2', '—']], 'each spoke under its own label, its values as drawn');
    assert.equal(JSON.stringify([chart.config.data, chart.config.options]), before, 'the chart’s own configuration is not edited');
    assert.deepEqual(chart.data.labels, ['A', 'B', 'C']);
});

test('Share names a radar’s spokes by its scale where the data names none, for records and for a spoke the data leaves empty', (t) => {
    const scale = (labels) => ({ scales: { r: { labels } } });
    const wind = (data, labels) => ({ ...(labels && { labels }), datasets: [{ label: 'Wind', data }] });
    const cases = {
        'the scale names them and the data none': [{ type: 'radar', options: scale(['North', 'West', 'East']), data: wind([3, 5, 2]) },
            [['North', '3'], ['West', '5'], ['East', '2']]],
        'records at their spokes': [{ type: 'radar', options: scale(['North', 'West', 'East']), data: wind([{ r: 3 }, { r: 5 }, { r: 2 }], ['A', 'B', 'C']) },
            [['North', '3'], ['West', '5'], ['East', '2']]],
        'a spoke the data leaves empty': [{ type: 'radar', options: scale(['North', 'West', 'East', 'South']), data: wind([3, 5, 2], ['A', 'B', 'C']) },
            [['North', '3'], ['West', '5'], ['East', '2'], ['South', '—']]],
    };
    for (const [name, [config, rows]] of Object.entries(cases)) {
        const html = drawnTable(t, config);
        assert.deepEqual([tableHead(html), drawnRows(html)], [['Label', 'Wind'], rows], name);
    }
});

test('Share keeps the radar and polar-area views whose labels the data names, or nobody does, as the room drew them', (t) => {
    const wind = (data, labels) => ({ ...(labels && { labels }), datasets: [{ label: 'Wind', data }] });
    const cases = {
        'a radar whose data names the spokes': [{ type: 'radar', data: { labels: ['Speed', 'Power', 'Range'], datasets: [{ label: 'Craft', data: [7, 3, 9] }] } },
            ['Label', 'Craft'], [['Speed', '7'], ['Power', '3'], ['Range', '9']]],
        'a radar with no labels anywhere': [{ type: 'radar', data: wind([3, 5, 2]) }, ['Position', 'Wind'], [['1', '3'], ['2', '5'], ['3', '2']]],
        // The scale holds `labels` here too, but writes none: the arcs are named in the legend, from the data.
        'a polar area whose scale holds labels it does not write': [{ type: 'polarArea', options: { scales: { r: { labels: ['North', 'West', 'East'] } } },
            data: wind([3, 5, 2], ['A', 'B', 'C']) }, ['Label', 'Wind'], [['A', '3'], ['B', '5'], ['C', '2']]],
    };
    for (const [name, [config, head, rows]] of Object.entries(cases)) {
        const html = drawnTable(t, config);
        assert.deepEqual([tableHead(html), drawnRows(html)], [head, rows], name);
    }
});

test('Share still exports points, radii, radial positions, zeros and pairs exactly as the room drew them', (t) => {
    const euro = [24, 0, null, 7].map((value) => ({ budget: { euro: value } }));
    // Each chart as its answer's Markdown draws it (a series keeps only its data and look, never its own parsing).
    const charts = [
        [{ type: 'scatter', data: { datasets: [{ label: 'Walk', data: [{ x: 1, y: 3 }, { x: 2.5, y: 7 }] }] } },
            ['Series', 'x', 'y'], [['Walk', '1', '3'], ['Walk', '2.5', '7']]],
        [{ type: 'pie', options: { parsing: { key: 'budget.euro' } }, data: { labels: ['Museum', 'Park', 'Cafe'], datasets: [{ label: 'Cost', data: euro }] } },
            ['Label', 'Cost'], [['Museum', '24'], ['Park', '0'], ['Cafe', '—'], ['—', '7']]],
        [{ type: 'doughnut', data: { labels: [], datasets: [{ label: 'Euro', data: [24, 0, 7] }] } },
            ['Position', 'Euro'], [['1', '24'], ['2', '0'], ['3', '7']]],
        [{ type: 'bubble', options: { elements: { point: { radius: [4, 0, 8] } } }, data: { datasets: [{ label: 'Steps', data: [[2, 5], [2, 6], [2, 7, 3]] }] } },
            ['Series', 'x', 'y', 'r'], [['Steps', '2', '5', '4'], ['Steps', '2', '6', '0'], ['Steps', '2', '7', '3']]],
        [{ type: 'bubble', options: { elements: { point: { radius: [2, 6, 0] } } }, data: { labels: [7, 8, 0], datasets: [{ label: 'Kiosks', data: [10, 20, 0] }] } },
            ['Series', 'Label', 'x', 'y', 'r'], [['Kiosks', '7', '7', '10', '2'], ['Kiosks', '8', '8', '20', '6'], ['Kiosks', '0', '0', '0', '0']]],
        [{ type: 'bubble', options: { indexAxis: 'y', scales: { y: { type: 'category' } } }, data: { labels: ['Mon', 'Tue'], datasets: [{ label: 'Visits', data: [3, 5] }] } },
            ['Series', 'Label', 'x', 'y', 'r'], [['Visits', 'Mon', '3', 'Mon', '3'], ['Visits', 'Tue', '5', 'Tue', '3']]],
        [{ type: 'line', data: { labels: ['Start', 'End'], datasets: [{ label: 'Route', data: [[0, 4], [1, 8]] }] } },
            ['Series', 'Label', 'x', 'y'], [['Route', 'Start', '0', '4'], ['Route', 'End', '1', '8']]],
        [{ type: 'bar', data: { labels: ['Start', 'End'], datasets: [{ label: 'Route', data: [[0, 4], [1, 8]] }] } },
            ['Label', 'Route'], [['Start', '0 – 4'], ['End', '1 – 8']]],
        [{ type: 'bar', data: { labels: ['Museum', 'Cafe', 'Tram'], datasets: [{ label: 'Euro', data: [24, 36, 10] }] } },
            ['Label', 'Euro'], [['Museum', '24'], ['Cafe', '36'], ['Tram', '10']]],
    ];
    for (const [config, head, rows] of charts) {
        const html = drawnTable(t, config);
        assert.deepEqual([tableHead(html), drawnRows(html)], [head, rows], `${config.type} ${JSON.stringify(config.data.datasets[0].data)}`);
    }
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
