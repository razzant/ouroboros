/* Declarative `chart` and `table` value presentation, pure and DOM-free: the
   Chart.js config built from the declaration plus the target data, the
   accessible data table every chart carries, the finite-value coercion, the
   table cell renderer with its number formatter and http(s)-only link guard,
   and the dotted-path reader (`getPath`) the whole declarative renderer in
   widgets.js shares with them. Moved out of widgets.js unchanged (widgets
   lifecycle phase 3; the table cell helpers in the cycle-A fix round). */

import { THEME_TRANSITION, seriesColours } from './theme_palette.js';
import { normalizeTone } from './ui_helpers.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

export function getPath(root, path, fallback = '') {
    if (!path) return root ?? fallback;
    let current = root;
    for (const part of String(path).split('.').filter(Boolean)) {
        if (current == null || typeof current !== 'object') return fallback;
        current = current[part];
    }
    return current ?? fallback;
}

export function finiteChartValue(value) {
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (typeof value !== 'string' || !value.trim()) return null;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}

export function chartConfig(component, data) {
    const type = ['line', 'bar'].includes(component.chart_type) ? component.chart_type : 'line';
    const labels = component.labels || getPath(data, component.labels_path || 'labels', []);
    const datasets = component.datasets || getPath(data, component.datasets_path || 'datasets', []);
    const unit = String(component.unit || '');
    return {
        type,
        data: {
            labels: Array.isArray(labels) ? labels.map((item) => String(item ?? '')) : [],
            // The host's default series in their dark values; `ouroSeries` lets theme_palette.js
            // repaint each in the mounted chart's theme. A bar carries its colour's edge.
            datasets: Array.isArray(datasets) ? datasets.map((dataset, idx) => ({
                label: String(dataset?.label ?? 'Series'),
                data: Array.isArray(dataset?.data) ? dataset.data.map(finiteChartValue) : [],
                ...seriesColours(idx),
                ...(type === 'bar' ? { borderWidth: 2 } : {}),
                ouroSeries: idx,
                spanGaps: false,
            })) : [],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            spanGaps: false,
            // A theme switch repaints the series at once, as the rest of the page does.
            transitions: { [THEME_TRANSITION]: { animation: { duration: 0 } } },
            plugins: { legend: { display: true } },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.06)' } },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    title: { display: Boolean(unit), text: unit },
                },
            },
        },
    };
}

// A value a chart carries: a number at its label, a point (`{x, y}`, a bubble's `r`), or a tuple: `[x, y]` or a
// bubble's `[x, y, r]` where the chart plots points, a floating bar's `[from, to]` at its label.
const isChartPoint = (value) => Boolean(value) && typeof value === 'object' && !Array.isArray(value);
const chartRange = (value) => (Array.isArray(value) ? value.map((part) => part ?? '—').join(' – ') : value);
const RADIAL_VALUE_KEYS = { pie: 'value', doughnut: 'value', radar: 'r', polarArea: 'r' };

// What Chart.js reads by the place of a label: a number or text and, in a bar, a floating pair. A record, and the pair of
// any other series, carries coordinates of its own, so no place proves its category.
const readByPlace = (seriesType, raw) => !isChartPoint(raw) && (seriesType === 'bar' || !Array.isArray(raw));

/** What a mounted category scale draws a datum at: the label it holds at the datum's parsed place, which is the one the
 *  owner reads on that axis, never the number the datum was given (Chart.js reads a number on a category scale as the
 *  place of its own point) or the index of a name. A gap, a place no label names and a scale that names no categories
 *  keep `cell`. */
function drawnAt(scale, parsed, cell) {
    const label = scale?.type === 'category' ? scale.getLabels()[parsed?.[scale.axis]] : undefined;
    return label === undefined ? cell : label;
}

/** One point as the chart holds it: the category its parsed index names and its value, a bar's floating interval, each on
 *  its own axis (a horizontal chart's index axis is `y`; a value axis that names categories says the label it drew), with a
 *  bubble's radius beside them. */
function drawnPoint({ controller, iScale, vScale, type }, point) {
    const parsed = controller.getParsed(point) || {};
    const interval = type === 'bar' && parsed._custom;
    return {
        [iScale.axis]: drawnAt(iScale, parsed),
        [vScale.axis]: interval ? chartRange([interval.start, interval.end]) : drawnAt(vScale, parsed, parsed[vScale.axis]),
        ...(type === 'bubble' && { r: controller.resolveDataElementOptions(point).radius }),
    };
}

/** A cell read from the data, as the chart drew it where the series' value axis names categories: that axis says the label
 *  its scale holds at the datum's place (`drawnAt`). The index axis (a label by place, or the coordinate as the data gave it),
 *  the other axes, a floating bar's interval and a gap stay as they were, and a series whose value axis names none is
 *  returned as it came. */
function onValueCategories({ controller, vScale, type }, point, cell) {
    if (cell === undefined || vScale?.type !== 'category') return cell;
    const parsed = controller.getParsed(point);
    if (!parsed || (type === 'bar' && parsed._custom)) return cell;
    if (!isChartPoint(cell)) return drawnAt(vScale, parsed, cell);
    return vScale.axis in cell ? { ...cell, [vScale.axis]: drawnAt(vScale, parsed, cell[vScale.axis]) } : cell;
}

/** Read the same object keys as the drawn Chart.js series, before its scales turn category names into indexes or
 *  coerce missing values to numbers. The mounted controller resolves parsing and Chart.js reads each path.
 *  BubbleController's normal radius (Chart.js 4.4.8) uses parsed `_custom` or the point option, zero included:
 *  it covers `[x, y]` and a bare value, and does not mistake a raw `r` for parsed data when parsing is off.
 *  Chart.js places a bubble's bare value on its value axis at the label its index scale holds for that position.
 *  A bar's pair is its floating interval, never two coordinates, whichever layout the table takes.
 *  A label stands for the place of its row where every point lies at its place (a value by place does; a record, and the
 *  pair of a line, lies where it names): then it is the one the category index scale holds (its own `labels`, else the
 *  data's), a column that records and pairs show only where the data names labels. Where a point lies away from its
 *  place, or a value by place lies beside points that name their own (a gap is no value), no place proves a category:
 *  each point is a row of the category Chart.js drew it at and its value, interval or radius, the category on the index
 *  axis, and no label by place stands beside it. */
export function renderDrawnChartDataTable(chart, label) {
    const { type, data } = chart.config;
    const { resolveObjectKey } = chart.constructor.helpers;
    const metas = data.datasets.map((_, index) => chart.getDatasetMeta(index));
    const byPlace = (index, raw) => metas[index].controller.options.parsing !== false && readByPlace(metas[index].type, raw);
    // A Markdown series names no axis, so one category index scale serves them all.
    const scale = metas[0]?.iScale;
    const onCategories = scale?.type === 'category' && metas.every(({ iScale }) => iScale === scale);
    // A radar's radial scale holds the label of each spoke and writes it there, a datum at the place of its spoke whatever
    // it is; a polar area's scale writes none (its names are the data's, in the legend), so the controller decides.
    const onSpokes = metas.length > 0 && metas.every(({ type: seriesType, iScale }) => seriesType === 'radar' && iScale === scale);
    const placed = data.datasets.every((dataset, index) => dataset.data.every((raw) => byPlace(index, raw)));
    // A value by place beside points that name their own category, or a point whose parsed index is not its place.
    const carried = onCategories && !placed && data.datasets.some((dataset, index) => dataset.data.some((raw, point) =>
        raw != null && (byPlace(index, raw) || metas[index].controller.getParsed(point)?.[scale.axis] !== point)));
    const datasets = data.datasets.map((dataset, index) => {
        const { controller, type: seriesType, iScale, vScale } = metas[index];
        const parsing = controller.options.parsing;
        const radialKey = RADIAL_VALUE_KEYS[seriesType];
        const bubble = seriesType === 'bubble';
        const bar = seriesType === 'bar';
        const drawnRadius = (point) => controller.resolveDataElementOptions(point).radius;
        const read = (raw, point) => {
            if (parsing !== false && Array.isArray(raw)) {
                // Cartesian controllers read coordinates; a bar reads the same pair as a floating interval.
                if (bubble) return { x: raw[0], y: raw[1], r: drawnRadius(point) };
                if (seriesType === 'line' || seriesType === 'scatter') return { x: raw[0], y: raw[1] };
                if (bar) return chartRange(raw);
            }
            if (bubble && parsing !== false && !isChartPoint(raw)) {
                return { [iScale.axis]: iScale.getLabels()[point], [vScale.axis]: raw, r: drawnRadius(point) };
            }
            if (!isChartPoint(raw)) return raw;
            if (parsing === false) return bubble ? { x: raw.x, y: raw.y, r: drawnRadius(point) } : raw;
            if (radialKey) return resolveObjectKey(raw, parsing?.key ?? radialKey);
            const coordinate = (key) => (bar ? chartRange(resolveObjectKey(raw, key)) : resolveObjectKey(raw, key));
            return {
                x: coordinate(parsing?.xAxisKey ?? 'x'),
                y: coordinate(parsing?.yAxisKey ?? 'y'),
                r: bubble ? drawnRadius(point) : raw.r,
            };
        };
        const values = dataset.data.map((raw, point) => (carried ? drawnPoint(metas[index], point) : onValueCategories(metas[index], point, read(raw, point))));
        return { label: dataset.label ?? '', data: values };
    });
    const labels = carried ? [] : onSpokes || (onCategories && (placed || data.labels?.length)) ? scale.getLabels() : data.labels;
    return renderChartDataTable({ type, data: { labels, datasets } }, label, false);
}

/** The accessible table keeps rows by label (by position for unnamed radial values) or by Cartesian point:
 *  its series, label if present, x, y and radius. */
export function renderChartDataTable(config, label, expanded) {
    const labels = config.data.labels || [];
    const datasets = config.data.datasets || [];
    const unlabelledRadial = !labels.length && Object.hasOwn(RADIAL_VALUE_KEYS, config.type);
    const values = (dataset) => (Array.isArray(dataset?.data) ? dataset.data : []);
    const cell = (name, value) => `<td data-label="${escapeHtml(name)}">${escapeHtml(value ?? '—')}</td>`;
    let head;
    let rows;
    if (['scatter', 'bubble'].includes(config.type) || datasets.some((dataset) => values(dataset).some(isChartPoint))
        || (!labels.length && !unlabelledRadial && datasets.some((dataset) => values(dataset).length))) {
        const points = datasets.flatMap((dataset) => values(dataset).map((value, idx) => {
            const point = isChartPoint(value) ? value : Array.isArray(value) ? { x: value[0], y: value[1], r: value[2] } : { y: value };
            return { series: dataset.label, label: labels[idx], x: point.x, y: point.y, r: point.r };
        }));
        const labelled = labels.length > 0;
        const radius = points.some((point) => point.r !== undefined);
        head = `<th>Series</th>${labelled ? '<th>Label</th>' : ''}<th>x</th><th>y</th>${radius ? '<th>r</th>' : ''}`;
        rows = points.map((point) => `<tr><th scope="row">${escapeHtml(point.series ?? '—')}</th>${labelled ? cell('Label', point.label) : ''}${cell('x', point.x)}${cell('y', point.y)}${radius ? cell('r', point.r) : ''}</tr>`).join('');
    } else {
        head = `<th>${unlabelledRadial ? 'Position' : 'Label'}</th>${datasets.map((dataset) => `<th>${escapeHtml(dataset.label)}</th>`).join('')}`;
        const count = datasets.reduce((length, dataset) => Math.max(length, values(dataset).length), labels.length);
        rows = Array.from({ length: count }, (_, idx) => `<tr><th scope="row">${escapeHtml(labels[idx] ?? (unlabelledRadial ? idx + 1 : '—'))}</th>${datasets.map((dataset) => cell(dataset.label, chartRange(values(dataset)[idx]))).join('')}</tr>`).join('');
    }
    return `<details class="widget-chart-data"${expanded ? ' open' : ''}><summary>View ${escapeHtml(label)} data</summary><div class="widget-table-wrap"><table class="widget-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div></details>`;
}

function safeTableHref(value) {
    const raw = String(value || '').trim();
    if (!raw) return '';
    try {
        const parsed = new URL(raw, window.location.origin);
        return ['http:', 'https:'].includes(parsed.protocol) ? parsed.href : '';
    } catch {
        return '';
    }
}

export function formatNumber(value, precision) {
    if (value === null || value === undefined || value === '') return '—';
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '—';
    const parsedPrecision = Number(precision);
    if (precision === undefined || precision === null || precision === '' || !Number.isFinite(parsedPrecision)) {
        return numeric.toLocaleString(undefined, { maximumFractionDigits: 12 });
    }
    const digits = Math.max(0, Math.min(12, parsedPrecision));
    return numeric.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function renderTableCell(row, column) {
    const presentation = String(column.presentation || column.format || 'plain');
    const raw = getPath(row, column.path, '');
    if (presentation === 'number') {
        const rendered = formatNumber(raw, column.precision);
        return `${escapeHtml(rendered)}${rendered !== '—' && column.unit ? ` ${escapeHtml(column.unit)}` : ''}`;
    }
    if (presentation === 'status') {
        const label = raw && typeof raw === 'object' ? (raw.label ?? raw.value ?? raw.status ?? '') : raw;
        const toneValue = raw && typeof raw === 'object' ? raw.tone : getPath(row, column.tone_path || '', 'muted');
        const tone = normalizeTone(toneValue);
        return `<span class="widget-table-status" data-tone="${escapeHtml(tone)}">${escapeHtml(label || '—')}</span>`;
    }
    if (presentation === 'link') {
        const rawHref = getPath(row, column.href_path || column.path, '');
        const href = safeTableHref(rawHref);
        const label = column.label_path ? getPath(row, column.label_path, rawHref) : raw;
        if (!href) return escapeHtml(label || rawHref || '—');
        return `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label || href)}</a>`;
    }
    return escapeHtml(raw ?? '');
}
