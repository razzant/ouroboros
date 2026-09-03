/* Declarative `chart` and `table` value presentation, pure and DOM-free: the
   Chart.js config built from the declaration plus the target data, the
   accessible data table every chart carries, the finite-value coercion, the
   table cell renderer with its number formatter and http(s)-only link guard,
   and the dotted-path reader (`getPath`) the whole declarative renderer in
   widgets.js shares with them. Moved out of widgets.js unchanged (widgets
   lifecycle phase 3; the table cell helpers in the cycle-A fix round). */

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

const CHART_PALETTE = [
    ['#e85d6f', 'rgba(232, 93, 111, 0.22)'],
    ['#60a5fa', 'rgba(96, 165, 250, 0.22)'],
    ['#34d399', 'rgba(52, 211, 153, 0.22)'],
    ['#fbbf24', 'rgba(251, 191, 36, 0.22)'],
];

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
            datasets: Array.isArray(datasets) ? datasets.map((dataset, idx) => {
                const [borderColor, backgroundColor] = CHART_PALETTE[idx % CHART_PALETTE.length];
                return {
                    label: String(dataset?.label ?? 'Series'),
                    data: Array.isArray(dataset?.data) ? dataset.data.map(finiteChartValue) : [],
                    borderColor,
                    backgroundColor,
                    spanGaps: false,
                };
            }) : [],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            spanGaps: false,
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

export function renderChartDataTable(config, label, expanded) {
    const labels = config.data.labels || [];
    const datasets = config.data.datasets || [];
    const rows = labels.map((item, idx) => `<tr><th scope="row">${escapeHtml(item)}</th>${datasets.map((dataset) => `<td data-label="${escapeHtml(dataset.label)}">${escapeHtml(dataset.data[idx] ?? '—')}</td>`).join('')}</tr>`).join('');
    return `<details class="widget-chart-data"${expanded ? ' open' : ''}><summary>View ${escapeHtml(label)} data</summary><div class="widget-table-wrap"><table class="widget-table"><thead><tr><th>Label</th>${datasets.map((dataset) => `<th>${escapeHtml(dataset.label)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div></details>`;
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
