import test from 'node:test';
import assert from 'node:assert/strict';

import {
    CHART_SERIES, THEME_TRANSITION, applyChartTheme, chartChrome, chartSeries, onThemeChange, paintChartSeries,
} from '../modules/theme_palette.js';
import { chartConfig } from '../modules/widget_chart.js';

/* Chart.js is not loaded here: the module only reads `options`/`scales` and calls
   `update`, so a shape-accurate stand-in is enough — and it lets the test watch
   for the one thing that must NEVER happen, a rebuild. */
function fakeChart({ options = {}, scales = {} } = {}) {
    const chart = { options, scales, updates: [] };
    chart.update = (mode) => chart.updates.push(mode);
    return chart;
}

function withTokens(text, grid, run, series = []) {
    const priorDocument = globalThis.document;
    const priorStyle = globalThis.getComputedStyle;
    const tokens = { '--chart-text': text, '--chart-grid': grid };
    series.forEach((value, idx) => { tokens[`--chart-series-${idx + 1}`] = value; });
    globalThis.document = { documentElement: {} };
    globalThis.getComputedStyle = () => ({
        getPropertyValue: (key) => tokens[key] ?? '',
    });
    try { run(); } finally {
        globalThis.document = priorDocument;
        globalThis.getComputedStyle = priorStyle;
    }
}

test('chrome comes from the tokens, and a client without CSSOM still gets ink', () => {
    withTokens(' #4a515d ', ' rgba(0,0,0,.14) ', () => {
        assert.deepEqual(chartChrome(), { text: '#4a515d', grid: 'rgba(0,0,0,.14)' });
    });
    // No document at all (module loaded outside a page) must not throw.
    assert.equal(typeof chartChrome().text, 'string');
    withTokens('', '', () => {
        assert.equal(chartChrome().text, '#a8b0bd', 'an unset token falls back, not to empty');
    });
});

test('a repaint re-tints chrome and axes in place, keeping the data', () => {
    const x = { options: {} };
    const y = { options: { ticks: { stepSize: 5 } } };
    const chart = fakeChart({
        options: { responsive: true },
        scales: { x, y },
    });
    chart.data = { datasets: [{ data: [1, 2, 3] }] };
    withTokens('#4a515d', '#dddddd', () => applyChartTheme(chart));
    assert.equal(chart.options.color, '#4a515d');
    assert.equal(chart.options.borderColor, '#dddddd');
    assert.equal(chart.options.plugins.legend.labels.color, '#4a515d');
    assert.equal(chart.options.plugins.title.color, '#4a515d');
    assert.equal(y.options.ticks.color, '#4a515d');
    assert.equal(y.options.ticks.stepSize, 5, 'an unrelated scale option is untouched');
    assert.deepEqual([x.options.grid.color, x.options.grid.tickColor], ['#dddddd', '#dddddd']);
    // 'none' is the whole point: the chart is re-tinted, not re-entered.
    assert.deepEqual(chart.updates, ['none']);
    assert.deepEqual(chart.data.datasets[0].data, [1, 2, 3]);
});

test('colours the author actually chose survive a theme switch', () => {
    const authored = { color: '#ff0000', plugins: { legend: { display: false } }, scales: { y: {} } };
    const scale = { options: { ticks: { color: '#00ff00' } } };
    const chart = fakeChart({
        options: { color: '#ff0000', plugins: { legend: { display: false } } },
        scales: { y: scale },
    });
    withTokens('#4a515d', '#dddddd', () => applyChartTheme(chart, authored));
    assert.equal(chart.options.color, '#ff0000');
    assert.equal(chart.options.plugins.legend.display, false);
    assert.equal(chart.options.plugins.legend.labels, undefined, 'declaring plugins claims the group');
    assert.equal(scale.options.ticks.color, '#00ff00');
    // borderColor was NOT authored, so it still follows the theme: the claim is
    // per group, not all-or-nothing.
    assert.equal(chart.options.borderColor, '#dddddd');
});

test('a chart destroyed mid-switch is not an error, and a scale-less type is skipped', () => {
    const doughnut = fakeChart({ options: {}, scales: {} });
    withTokens('#fff', '#000', () => applyChartTheme(doughnut));
    assert.deepEqual(doughnut.updates, ['none']);
    const dead = fakeChart();
    dead.update = () => { throw new Error('Cannot read properties of null'); };
    withTokens('#fff', '#000', () => applyChartTheme(dead));
    // Nothing to assert but the absence of a throw: a view unmounting while the
    // theme changes is ordinary, not a failure the owner should ever see.
    applyChartTheme(null);
    applyChartTheme({});
});

// The series the owner approved for Light (#474); the dark ones are CHART_SERIES.
const LIGHT = ['#c0284c', '#2c68d0', '#1e9c76', '#bd6a00'];
const fill = (hex) => `rgba(${[16, 8, 0].map((shift) => (parseInt(hex.slice(1), 16) >> shift) & 255).join(', ')}, 0.22)`;

test('a host chart declares its default series in the approved order, a fifth taking the first again', () => {
    assert.deepEqual(CHART_SERIES, ['#ee6780', '#60a5fa', '#3fdca3', '#fbbf24']);
    const datasets = Array.from({ length: 5 }, (_, idx) => ({ label: `S${idx}`, data: [idx], borderColor: '#000000' }));
    const bar = chartConfig({ chart_type: 'bar', labels: ['a'], datasets }, {});
    assert.deepEqual(bar.data.datasets.map((d) => [d.ouroSeries, d.borderColor, d.backgroundColor, d.borderWidth]),
        [0, 1, 2, 3, 0].map((series, idx) => [idx, CHART_SERIES[series], fill(CHART_SERIES[series]), 2]),
        'built without a document, in the dark values; an author colour in the declaration is not a host option');
    const line = chartConfig({ labels: ['a'], datasets: datasets.slice(0, 1) }, {});
    assert.equal(line.data.datasets[0].borderWidth, undefined, 'only a bar takes the series edge');
    assert.deepEqual(line.options.transitions, { [THEME_TRANSITION]: { animation: { duration: 0 } } });
});

test('a theme paints only the series the host marked and says whether a colour moved', () => {
    const marked = [{ ouroSeries: 0, borderColor: CHART_SERIES[0] }, { ouroSeries: 4, borderColor: CHART_SERIES[0] }];
    const author = { borderColor: '#123456', backgroundColor: '#654321' };
    const odd = { ouroSeries: -1, borderColor: '#abcdef' };
    withTokens('#4a515d', '#dddddd', () => {
        assert.equal(paintChartSeries([...marked, author, odd, null]), true);
        assert.equal(paintChartSeries(marked), false, 'painted already: nothing moves');
    }, LIGHT);
    assert.deepEqual(marked.map((d) => [d.borderColor, d.backgroundColor]), [[LIGHT[0], fill(LIGHT[0])], [LIGHT[0], fill(LIGHT[0])]]);
    assert.deepEqual(author, { borderColor: '#123456', backgroundColor: '#654321' });
    assert.equal(odd.borderColor, '#abcdef');
    assert.equal(paintChartSeries(undefined), false);
    // A token that is not #rrggbb, or none at all, keeps the dark value of its place.
    withTokens('#fff', '#000', () => {
        assert.deepEqual(chartSeries(), [LIGHT[0], CHART_SERIES[1], CHART_SERIES[2], '#1E9C76']);
    }, [` ${LIGHT[0]} `, 'rgb(1, 2, 3)', '', '#1E9C76']);
    assert.deepEqual(chartSeries(), CHART_SERIES, 'no document: the dark values');
});

test('a host series that moved lands through the zero-length theme transition; otherwise none', () => {
    const chart = fakeChart({ options: {}, scales: {} });
    chart.data = { datasets: [{ ouroSeries: 0, borderColor: CHART_SERIES[0], backgroundColor: fill(CHART_SERIES[0]) }] };
    withTokens('#4a515d', '#dddddd', () => applyChartTheme(chart), LIGHT);
    withTokens('#4a515d', '#dddddd', () => applyChartTheme(chart), LIGHT);
    // update('none') never refreshes the colours Chart.js resolved into points and bars.
    assert.deepEqual(chart.updates, [THEME_TRANSITION, 'none']);
    assert.equal(chart.data.datasets[0].borderColor, LIGHT[0]);
});

test('the subscription hands back an unsubscribe that really detaches', () => {
    const prior = globalThis.window;
    const listeners = new Map();
    globalThis.window = {
        addEventListener: (key, fn) => listeners.set(key, fn),
        removeEventListener: (key, fn) => { if (listeners.get(key) === fn) listeners.delete(key); },
    };
    try {
        let seen = 0;
        const off = onThemeChange(() => { seen += 1; });
        listeners.get('ouro:theme-changed')();
        assert.equal(seen, 1);
        off();
        assert.equal(listeners.size, 0, 'a disposed view must not keep re-tinting dead charts');
    } finally { globalThis.window = prior; }
});
