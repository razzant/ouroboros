import assert from 'node:assert/strict';
import test from 'node:test';

import { evolutionChartTheme } from '../modules/evolution.js';
import { chartColorAlpha } from '../modules/utils.js';

// `chartColorAlpha` is the ONE place a design token becomes a canvas fill, and it
// is fed whatever `getComputedStyle().getPropertyValue()` hands back — which is
// browser-dependent. Chrome normalizes to `rgb(r, g, b)`, but a modern engine may
// return the space-separated `rgb(r g b / a)` form, and an author may have written
// the token as `#abc`. A miss here does not throw: it produces `rgba(NaN, …)`,
// which Chart.js silently drops, so the fill just disappears. Hence a unit test
// rather than a visual check.

test('hex tokens expand to rgba, both 3- and 6-digit', () => {
    assert.equal(chartColorAlpha('#22c55e', 0.22), 'rgba(34, 197, 94, 0.22)');
    assert.equal(chartColorAlpha('#abc', 0.5), 'rgba(170, 187, 204, 0.5)');
    // Case and surrounding whitespace come from CSS, not from the caller.
    assert.equal(chartColorAlpha('  #F59E0B  ', 0.13), 'rgba(245, 158, 11, 0.13)');
});

test('legacy comma rgb()/rgba() keeps its channels and takes the new alpha', () => {
    assert.equal(chartColorAlpha('rgb(240, 122, 134)', 0.22), 'rgba(240, 122, 134, 0.22)');
    // The INCOMING alpha is replaced, never multiplied: a token that already
    // carries 0.55 must still fill at exactly the requested 0.22.
    assert.equal(
        chartColorAlpha('rgba(255, 255, 255, 0.55)', 0.22),
        'rgba(255, 255, 255, 0.22)',
    );
});

test('modern space-separated rgb(r g b / a) parses — the regression this pins', () => {
    // The previous splitter was /[,/]/, so `110 150 210` stayed ONE token: r
    // became "110 150 210" and g/b became undefined, yielding
    // `rgba(110 150 210, undefined, undefined, 0.22)`.
    assert.equal(chartColorAlpha('rgb(110 150 210 / 0.55)', 0.22), 'rgba(110, 150, 210, 0.22)');
    assert.equal(chartColorAlpha('rgb(34 197 94)', 0.22), 'rgba(34, 197, 94, 0.22)');
    assert.equal(chartColorAlpha('rgba(239 68 68 / 15%)', 0.3), 'rgba(239, 68, 68, 0.3)');
});

test('anything not three plain numeric channels falls back to the colour UNCHANGED', () => {
    // A visible fully-opaque fill beats an invisible rgba(NaN, …) Chart.js drops.
    for (const input of [
        'rgb(50% 20% 30%)',            // percentage channels
        'rgb(none 20 30)',             // CSS Color 4 `none`
        'color-mix(in srgb, red, blue)',
        'var(--some-token)',           // defensive: the seam resolves aliases, so
                                       // a literal var() never actually arrives
        'transparent',
        'nonsense',
        '#12',                         // malformed hex
    ]) {
        assert.equal(chartColorAlpha(input, 0.22), input, `expected passthrough for ${input}`);
    }
});

test('empty and non-string input degrade to an empty string, never "rgba(NaN"', () => {
    for (const input of ['', '   ', null, undefined, 0, false, {}]) {
        const out = chartColorAlpha(input, 0.22);
        assert.equal(typeof out, 'string');
        assert.ok(!out.includes('NaN'), `NaN leaked for ${String(input)}`);
    }
    assert.equal(chartColorAlpha('', 0.22), '');
    assert.equal(chartColorAlpha(null, 0.22), '');
});

test('the evolution theme resolves through the live token seam, in ramp order', () => {
    // A stub `:root` proves the theme reads tokens at CALL time and keeps the
    // series ramp positional — the dataset order in evolution.js indexes into it.
    const tokens = {
        '--accent-light': '#f07a86',
        '--user': '#6e96d2',
        '--green': '#22c55e',
        '--amber': '#f59e0b',
        '--project': '#2dd4bf',
        '--accent': '#c93545',
        '--text-muted': 'rgba(255, 255, 255, 0.35)',
        '--text-secondary': 'rgba(255, 255, 255, 0.55)',
        '--divider': 'rgba(255, 255, 255, 0.07)',
        '--surface-border-soft': 'rgba(255, 255, 255, 0.06)',
        '--bg-elevated': '#1a1a1d',
        '--surface-border': 'rgba(255, 255, 255, 0.08)',
        '--text-primary': '#e7e7ea',
        '--font-mono': 'ui-monospace, Menlo, monospace',
    };
    const root = {};
    globalThis.getComputedStyle = () => ({
        getPropertyValue: (name) => tokens[name] ?? '',
    });
    try {
        const theme = evolutionChartTheme(root);
        assert.equal(theme.series.length, 6);
        assert.deepEqual(theme.series, [
            '#f07a86', '#6e96d2', '#22c55e', '#f59e0b', '#2dd4bf', '#c93545',
        ]);
        // Six DISTINGUISHABLE lanes, not shades of one hue.
        assert.equal(new Set(theme.series).size, 6);
        // Every scalar slot is filled — a renamed token would surface as ''.
        for (const key of ['axis', 'axisTitle', 'grid', 'surface', 'border', 'strong', 'mono']) {
            assert.ok(theme[key], `theme.${key} resolved empty`);
        }
        assert.equal(theme.mono, 'ui-monospace, Menlo, monospace');
    } finally {
        delete globalThis.getComputedStyle;
    }
});
