/* Canvas-side half of the appearance tokens.

   Chart.js paints into a canvas, so it cannot inherit a CSS variable the way the
   DOM and Mermaid (via themeVariables) do: the chrome colours have to be copied
   into each live instance. theme.js owns the choice and announces a real repaint
   with `ouro:theme-changed`; this module is the single place that translates the
   current tokens into Chart options, so a mounted chart follows the palette
   without being rebuilt and without its data or author config being re-read.
   A chart the host draws (widget_chart.js) takes its default series from here
   too; an author's chart keeps the series colours its author chose. */

const CHART_FALLBACK = { text: '#a8b0bd', grid: 'rgba(255, 255, 255, 0.10)' };

// The default series in the owner's order (#474): rose, blue, green, amber. These are the dark
// values, which the DOM-free config and a page without the tokens use; ui.css holds both themes.
export const CHART_SERIES = Object.freeze(['#ee6780', '#60a5fa', '#3fdca3', '#fbbf24']);
const SERIES_TOKENS = ['--chart-series-1', '--chart-series-2', '--chart-series-3', '--chart-series-4'];
// The chart mode a host series repaint lands through; widget_chart.js gives it no duration.
export const THEME_TRANSITION = 'theme';

/** Current chart chrome from the CSS tokens on <html>. */
export function chartChrome() {
    if (typeof getComputedStyle !== 'function' || typeof document === 'undefined') return { ...CHART_FALLBACK };
    const style = getComputedStyle(document.documentElement);
    return {
        text: style.getPropertyValue('--chart-text').trim() || CHART_FALLBACK.text,
        grid: style.getPropertyValue('--chart-grid').trim() || CHART_FALLBACK.grid,
    };
}

/** Current default series from the tokens; a token that is not `#rrggbb` keeps its dark value. */
export function chartSeries() {
    if (typeof getComputedStyle !== 'function' || typeof document === 'undefined') return [...CHART_SERIES];
    const style = getComputedStyle(document.documentElement);
    return SERIES_TOKENS.map((token, idx) => {
        const value = style.getPropertyValue(token).trim();
        return /^#[0-9a-f]{6}$/i.test(value) ? value : CHART_SERIES[idx];
    });
}

/** Series `index` (a fifth takes the first again): its line or bar edge, and the same colour at 22 % inside. */
export function seriesColours(index, palette = CHART_SERIES) {
    const borderColor = palette[index % palette.length];
    const n = parseInt(borderColor.slice(1), 16);
    return { borderColor, backgroundColor: `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, 0.22)` };
}

/**
 * Give the host's series (the datasets widget_chart.js marks with `ouroSeries`) the current
 * theme's colours, before a chart is built or takes new data and when the theme moves. An
 * author's dataset never carries the mark (chat_markdown.js keeps only its own keys). Returns
 * whether any colour changed.
 */
export function paintChartSeries(datasets, palette = chartSeries()) {
    let moved = false;
    for (const dataset of Array.isArray(datasets) ? datasets : []) {
        const index = dataset?.ouroSeries;
        if (!Number.isInteger(index) || index < 0) continue;
        const colours = seriesColours(index, palette);
        if (dataset.borderColor === colours.borderColor && dataset.backgroundColor === colours.backgroundColor) continue;
        Object.assign(dataset, colours);
        moved = true;
    }
    return moved;
}

const branch = (parent, key) => {
    const existing = parent[key];
    if (existing && typeof existing === 'object') return existing;
    parent[key] = {};
    return parent[key];
};

/**
 * Repaint one mounted Chart instance in the current theme, keeping its data.
 *
 * `authored` is the caller's own untouched option object. Chart.js resolves
 * every scale option against its defaults, so "did the author pick this colour?"
 * cannot be read back off the instance; the caller, which still holds what it
 * was given, decides instead. Declaring `scales` or `plugins` claims that whole
 * group — a coarse rule, but one that never silently discards author colours.
 */
export function applyChartTheme(chart, authored = {}) {
    const options = chart?.options;
    if (!options) return;
    const { text, grid } = chartChrome();
    if (authored.color === undefined) options.color = text;
    if (authored.borderColor === undefined) options.borderColor = grid;
    if (authored.plugins === undefined) {
        const plugins = branch(options, 'plugins');
        branch(branch(plugins, 'legend'), 'labels').color = text;
        branch(plugins, 'title').color = text;
    }
    if (authored.scales === undefined) {
        // chart.scales holds only the scales this chart really built, so a
        // doughnut or any scale-less type is skipped instead of being handed
        // x/y axes it never asked for.
        for (const scale of Object.values(chart.scales || {})) {
            const scaleOptions = scale?.options;
            if (!scaleOptions) continue;
            branch(scaleOptions, 'ticks').color = text;
            const gridOptions = branch(scaleOptions, 'grid');
            gridOptions.color = grid;
            gridOptions.tickColor = grid;
            branch(scaleOptions, 'title').color = text;
        }
    }
    // 'none': re-tint without replaying entry animations on an already-shown chart. It never
    // refreshes the colours Chart.js resolved into points and bars (one object it shares per
    // dataset), so a host series that moved lands through the zero-length theme transition.
    const mode = paintChartSeries(chart.data?.datasets) ? THEME_TRANSITION : 'none';
    try { chart.update(mode); } catch { /* a chart destroyed mid-switch is not an error */ }
}

/** Subscribe to real theme repaints. Returns the unsubscribe the caller must own. */
export function onThemeChange(handler) {
    if (typeof window === 'undefined') return () => {};
    window.addEventListener('ouro:theme-changed', handler);
    return () => window.removeEventListener('ouro:theme-changed', handler);
}
