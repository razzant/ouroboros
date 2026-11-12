/**
 * Dashboard → Evolution: how Ouroboros changed, on one time line (docs/DESIGN.md §5; ARCHITECTURE §3 "Dashboard").
 * Three independent reads. `/api/state` paints the campaign object (Start and Pause through `/api/command`, consciousness
 * as one status sentence); it is read through the page's one state clock when there is one, whose every answer (its
 * periodic read included) reaches this page too, so a command the runtime applies late shows up without a poll of its
 * own. The cycles' settings cards read `/api/settings` (dashboard.js). `/api/evolution-data` answers at once with the
 * saved version history and says whether a newer copy is being measured, which this module polls only while that runs.
 * One measure is drawn at a time. A milestone (a MAJOR or MINOR release, or a version its release author named) is a dot
 * on the line, labelled where there is room; every version is a tick under it, and the chosen version's own release row
 * reads in full beneath. Identity and Scratchpad belong to this installation, not to a release: they are separate
 * measures in the characters their journals record, and a date with no record is a gap, never zero.
 */
import { formatUsd2 } from './utils.js';
import { apiFetch, readState } from './api_client.js';
import { readStateSnapshot } from './state_snapshots.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { chooseAndSendReview } from './review_command.js';
import { tr } from './i18n.js';
import { showToast } from './toast.js';
import { statusMarkHtml } from './status_mark.js';
import { formatBytes, number, whenLabel } from './storage_format.js';
import { escapeHtmlAttr as esc } from './ui_primitives.js';
import { CODECS, GUARDED_WRITERS, WRITERS, commitSetting } from './settings_controls.js';
import { rowById } from './settings_rows.js';
import { mountTabSettings } from './dashboard.js';

// ---- The campaign and consciousness --------------------------------------------------------------------------------

// The statuses that wait for the owner: a first conversation, or the owner's chat.
const NEEDS_OWNER = ['waiting_for_first_conversation', 'waiting_for_owner_chat'];

/**
 * A runtime status's tone on the shared chip and status sentence (docs/DESIGN.md "Status and chips"): ok while it runs
 * or rests ready; warn (amber) only while it needs the owner (a first conversation, the owner's chat, or a spending
 * limit that stopped it); error for a failure; neutral for everything that goes on by itself or is off.
 */
export function runtimeTone(status) {
    if (['running', 'queued', 'idle_ready', 'thinking', 'sleeping', 'due'].includes(status)) return 'ok';
    if ([...NEEDS_OWNER, 'budget_blocked', 'budget_stopped'].includes(status)) return 'warn';
    if (['paused_failures', 'error_backoff', 'allowance_exhausted', 'allowance_unknown', 'wake_failed', 'wake_rejected'].includes(status)) return 'error';
    return 'neutral';
}

// A status's short word: a null control is never "off", and the boot window (consciousness_view) says "starting".
const TRANSLATED = { due: 'due', wake_paused: 'paused', wake_outcome_unknown: 'unknown' };
const SHORT_WORDS = { state_unknown: 'unknown', unknown: 'unknown', thinking: 'thinking', sleeping: 'sleeping',
    ...Object.fromEntries(NEEDS_OWNER.map((status) => [status, 'needs owner'])), allowance_exhausted: 'allowance',
    allowance_unknown: 'allowance', wake_failed: 'failed', wake_rejected: 'refused', running: 'running', queued: 'queued',
    idle_ready: 'idle', waiting_for_idle: 'waiting', waiting_for_restart_verify: 'restart verify', paused: 'paused',
    paused_failures: 'paused', accounting_unavailable: 'paused', budget_blocked: 'budget', budget_stopped: 'budget',
    error_backoff: 'retrying', stopped: 'stopped', starting: 'starting' };
function shortStatusLabel(status, fallback = 'off') {
    if (Object.hasOwn(TRANSLATED, status)) return tr(`consciousness.${status}`, TRANSLATED[status]);
    return Object.hasOwn(SHORT_WORDS, status) ? SHORT_WORDS[status] : fallback;
}

const capital = (text) => (text ? text[0].toUpperCase() + text.slice(1) : '');
const count = (n, one) => `${n} ${one}${Number(n) === 1 ? '' : 's'}`;

/** What the campaign object says about one `/api/state` answer (pure; `null` before the first answer). */
export function campaignView(runtime, now = new Date()) {
    if (!runtime) return null;
    const evolution = runtime.evolution_state || {};
    const campaign = evolution.campaign || {};
    const consciousness = runtime.bg_consciousness_state || {};
    const status = evolution.status || (runtime.evolution_enabled === null ? 'state_unknown'
        : runtime.evolution_enabled ? 'idle_ready' : 'disabled');
    const enabled = typeof evolution.enabled === 'boolean' ? evolution.enabled
        : typeof runtime.evolution_enabled === 'boolean' ? runtime.evolution_enabled : null;
    const bgStatus = consciousness.status || (runtime.bg_consciousness_enabled === null ? 'unknown'
        : runtime.bg_consciousness_enabled ? 'sleeping' : 'disabled');
    // Evolution is self-modification work, so it is hard-blocked in light mode.
    const lightMode = String(runtime.runtime_mode || '') === 'light';
    const off = status === 'disabled';
    const budget = evolution.budget_remaining_usd;
    const facts = off ? [] : [
        evolution.cycle ? `Cycle ${evolution.cycle}` : '',
        campaign.cycles_done ? count(campaign.cycles_done, 'attempt') : '',
        campaign.absorbed_cycles_done ? `${campaign.absorbed_cycles_done} absorbed` : '',
        budget !== null && budget !== undefined && Number.isFinite(Number(budget)) ? `Budget left ${formatUsd2(budget)}` : '',
        evolution.consecutive_failures ? count(evolution.consecutive_failures, 'failure') : '',
        evolution.last_task_at ? `Last cycle ${whenLabel(evolution.last_task_at, now)}` : '',
    ].filter(Boolean);
    const wake = bgStatus === 'sleeping' && consciousness.next_wake_at ? whenLabel(consciousness.next_wake_at, now) : '';
    return {
        status, enabled, lightMode,
        chip: { text: capital(shortStatusLabel(status, 'off')), tone: runtimeTone(status) },
        note: off ? (lightMode ? 'Campaigns need runtime mode Advanced or Pro; this install runs Light.'
            : 'Improvement cycles run only during a campaign.') : String(evolution.detail || ''),
        lines: [campaign.objective ? `Objective: ${campaign.objective}` : '', campaign.progress_notes ? `Progress: ${campaign.progress_notes}` : '',
            facts.join(' · ')].filter(Boolean),
        // One pointer to the Consciousness tab, which says the rest (its detail and allowance are that tab's facts):
        // the neutral tone unless it works right now, or something there needs the owner or failed.
        consciousness: {
            text: `Consciousness: ${shortStatusLabel(bgStatus, 'off')}${wake ? ` until ${wake.replace(/^today at /, '')}` : ''}`,
            tone: runtimeTone(bgStatus) !== 'ok' || ['thinking', 'running'].includes(bgStatus) ? runtimeTone(bgStatus) : 'neutral',
        },
    };
}

/**
 * Ask for the evolution-campaign objective (pure decision helper, node-tested with an injected dialog). window.prompt is
 * dead on the macOS desktop shell (pywebview/WKWebView answers null silently), which made Start campaign a one-click
 * PAID launch with a hidden-empty objective and no way to cancel. Semantics (owner-approved Б1-8): cancel/Escape/backdrop
 * = do NOT start; confirmed-empty = start with the default autonomous objective downstream (evolution_lifecycle.py);
 * confirmed-text = start with that objective. `running`: the same `/evolve on` gives a running campaign the typed
 * objective in place, and an empty one keeps the campaign's own.
 */
export async function promptCampaignObjective({ dialogImpl = openConfirmDialog, running = false } = {}) {
    const answer = await dialogImpl(running ? {
        title: 'New campaign objective', input: true, confirmLabel: 'Set objective',
        body: 'The running campaign continues with this objective. Leave it empty to keep the current one.',
    } : {
        title: 'Start evolution campaign', input: true, confirmLabel: 'Start campaign',
        body: 'Evolution campaign objective (optional). Leave empty to let Ouroboros pick its own autonomous objective.',
    });
    if (!answer?.confirmed) return { confirmed: false, objective: '', entered: '' };
    // `entered` keeps the owner's words byte for byte; `objective` is the command's trimmed form.
    return { confirmed: true, objective: String(answer.value || '').trim(), entered: String(answer.value || '') };
}

// ---- The history: measures, words and numbers ----------------------------------------------------------------------

const DAY = 86_400_000;
const MINUS = '−';
export const MEASURES = [
    { key: 'code_lines', label: 'Code', what: 'Python lines in every tracked .py file, tests included' },
    { key: 'bible_bytes', label: 'Constitution', what: 'the size of BIBLE.md' },
    { key: 'system_bytes', label: 'System prompt', what: 'the size of prompts/SYSTEM.md' },
    { key: 'identity', label: 'Identity', install: true,
        what: 'characters of identity.md, as its journal records them at every write' },
    { key: 'scratchpad', label: 'Scratchpad', install: true,
        what: 'characters of the scratchpad’s block contents (block headers not counted), as its journal records them at every append' },
];
const MEASURE = Object.fromEntries(MEASURES.map((measure) => [measure.key, measure]));
const KIND_WORDS = { feat: 'Feature', fix: 'Fix', chore: 'Maintenance', refactor: 'Refactor', docs: 'Docs', test: 'Tests',
    tests: 'Tests', perf: 'Performance', ci: 'Build', build: 'Build', release: 'Release', rc: 'Pre-release', 'pre-release': 'Pre-release',
    constitution: 'Constitution', repo: 'Repository' };

const formatter = (options, format = new Intl.DateTimeFormat('en-GB', options)) => (time) => format.format(new Date(time));
const fullDate = formatter({ day: 'numeric', month: 'short', year: 'numeric' });
const dayMonth = formatter({ day: 'numeric', month: 'short' });
const monthShort = formatter({ month: 'short' });
const finite = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);

const SIZES = new Set(['bible_bytes', 'system_bytes']);

/** One measure's value as words, in the Storage tab's formats (storage_format.js): counts grouped, sizes in decimal
 *  units; unknown is said, never zero. */
export function formatMeasure(key, value) {
    if (value === null || value === undefined) return MEASURE[key]?.install ? 'Not recorded' : 'Not measured';
    if (SIZES.has(key)) return formatBytes(value);
    return `${number(value)} ${key === 'code_lines' ? 'lines' : 'characters'}`;
}

/** The change from one value to the next in the measure's own unit (lines as a percentage). */
export function formatDelta(key, value, previous) {
    if (value === null || previous === null || value === undefined || previous === undefined) return '';
    const sign = (d) => (d > 0 ? '+' : MINUS);
    if (key === 'code_lines') {
        if (!previous) return '';
        const pct = ((value - previous) / previous) * 100;
        if (Math.abs(pct) < 0.05) return 'no change';
        return `${sign(pct)}${Math.abs(pct) < 10 ? Math.abs(pct).toFixed(1) : Math.round(Math.abs(pct))}%`;
    }
    if (value === previous) return 'no change';
    const d = value - previous;
    return `${sign(d)}${SIZES.has(key) ? formatBytes(Math.abs(d)) : `${number(Math.abs(d))} characters`}`;
}

/** The versions of one `/api/evolution-data` answer with their numbers and the installation's journals (pure). */
export function historyModel(data) {
    const install = data?.install || {};
    const versions = (Array.isArray(data?.versions) ? data.versions : []).map((version, index) => ({
        ...version,
        t: Date.parse(version.date),
        release: version.release || {},
        values: {
            code_lines: finite(version.code_lines), bible_bytes: finite(version.bible_bytes), system_bytes: finite(version.system_bytes),
            identity: finite(install.identity?.at_versions?.[index]), scratchpad: finite(install.scratchpad?.at_versions?.[index]),
        },
    })).filter((version) => Number.isFinite(version.t));
    versions.forEach((version, index) => { version.i = index; });
    const observed = {};
    for (const key of ['identity', 'scratchpad']) {
        const latest = install[key]?.latest;
        const t = latest ? Date.parse(latest.at) : NaN;
        observed[key] = Number.isFinite(t) && finite(latest.chars) !== null ? { t, v: latest.chars, at: latest.at } : null;
    }
    return {
        versions, milestones: versions.filter((version) => version.milestone), latest: versions.at(-1) || null,
        observed, firstAt: { identity: install.identity?.first_at || '', scratchpad: install.scratchpad?.first_at || '' },
        savedAt: data?.saved_at || '', refresh: data?.refresh || { state: 'idle' },
    };
}

/**
 * What a version says about itself: the author's short title, else its row heading, else its number. `authored` says
 * the title is the release author's words, which the interface language never translates (docs/DESIGN.md §4).
 */
export function versionWords(version) {
    const release = version.release || {};
    const heading = capital(String(release.heading || '').trim().replace(/\.$/, ''));
    const base = String(release.kind || '').replace(/\(.*$/, '').replace(/!$/, '').toLowerCase();
    const kind = version.level === 'pre' ? 'Pre-release'
        : KIND_WORDS[base] || (/^v\d+$/.test(base) || version.level === 'major' ? 'Major release' : 'Version');
    return { title: release.title || heading || `Version ${version.version}`, lead: release.title ? heading : '',
        body: String(release.body || ''), kind, authored: Boolean(release.title || heading) };
}

// Where a release author's words are shown (title, lead, body, a label, a tooltip line), the overlay keeps out.
const AUTHORED = ' data-i18n-authored';
const authoredIf = (words) => (words.authored ? AUTHORED : '');

function sourceLabel(version) {
    const also = version.aliases?.length ? ` (also ${version.aliases.join(', ')})` : '';
    const source = version.release?.source;
    if (source === 'readme') return `README Version History at ${version.tag}${also}`;
    if (source === 'tag') return `tag annotation of ${version.tag}${also}`;
    return `no release note recorded for ${version.tag}${also}`;
}

// ---- The time line ------------------------------------------------------------------------------------------------

/**
 * A time axis over `times` for `width` px. When one quiet stretch stands out — over a third of the whole span, over two
 * weeks and three times the next longest (the first days' burst of tags, then months without one) — the axis breaks
 * there: each side keeps its own time scale, a short side keeps a readable width, and the gap is drawn as a gap.
 */
export function timeAxis(times, width) {
    const sorted = times.filter(Number.isFinite).sort((a, b) => a - b);
    if (!sorted.length || width <= 0) return null;
    const first = sorted[0], last = sorted.at(-1);
    let at = -1, gap = 0, next = 0;
    for (let i = 1; i < sorted.length; i += 1) {
        const step = sorted[i] - sorted[i - 1];
        if (step > gap) { next = gap; gap = step; at = i; } else if (step > next) next = step;
    }
    const segment = (a, b, x0, x1) => {
        const pad = Math.max((b - a) * 0.02, DAY / 2);
        return { a, b, t0: a - pad, t1: b + pad, x0, x1 };
    };
    if (at < 0 || gap < Math.max((last - first) / 3, 14 * DAY, next * 3)) {
        const only = segment(first, last, 0, width);
        return { segments: [only], gap: null, x: (t) => ((t - only.t0) / (only.t1 - only.t0)) * width };
    }
    const gapPx = width < 420 ? 40 : 54;
    const room = width - gapPx;
    const floor = Math.max(56, Math.min(110, width * 0.13));
    const left = segment(first, sorted[at - 1], 0, 0), right = segment(sorted[at], last, 0, width);
    const share = (left.t1 - left.t0) / ((left.t1 - left.t0) + (right.t1 - right.t0));
    left.x1 = Math.min(Math.max(room * share, floor), room - floor);
    right.x0 = left.x1 + gapPx;
    const x = (t) => {
        const side = t <= left.t1 ? left : t >= right.t0 ? right : null;
        return side ? side.x0 + ((t - side.t0) / (side.t1 - side.t0)) * (side.x1 - side.x0) : left.x1 + gapPx / 2;
    };
    return { segments: [left, right], gap: [left.x1, right.x0], x };
}

function axisLabels(axis) {
    const labels = [];
    for (const side of axis.segments) {
        if ((side.b - side.a) / DAY < 20) {
            const same = monthShort(side.a) === monthShort(side.b);
            const range = dayMonth(side.a) === dayMonth(side.b) ? dayMonth(side.a)
                : same ? `${new Date(side.a).getDate()}–${dayMonth(side.b)}` : `${dayMonth(side.a)} – ${dayMonth(side.b)}`;
            labels.push({ x: (side.x0 + side.x1) / 2, text: side.x1 - side.x0 < 70 ? monthShort(side.a) : range, anchor: 'middle' });
            continue;
        }
        const start = new Date(side.t0);
        const many = side.t1 - side.t0 > 300 * DAY;
        for (let month = new Date(start.getFullYear(), start.getMonth() + 1, 1); month.getTime() < side.t1;
            month = new Date(month.getFullYear(), month.getMonth() + 1, 1)) {
            const text = many && month.getMonth() === 0 ? `${monthShort(month)} ${month.getFullYear()}` : monthShort(month);
            labels.push({ x: axis.x(month.getTime()), text, anchor: 'start', tick: true });
        }
    }
    let lastX = -Infinity;
    return labels.filter((label) => {
        if (label.x - lastX < 34) return false;
        lastX = label.x;
        return true;
    });
}

function niceMax(value) {
    if (!(value > 0)) return 1;
    const e = 10 ** Math.floor(Math.log10(value)), f = value / e;
    return [1, 1.5, 2, 3, 4, 5, 6, 8, 10].find((step) => f <= step) * e;
}

const CHART = { height: 190, top: 22, base: 140, endPad: 78 };

/** The chosen measure's points: every version, plus the installation's newest journal record when it is later. */
function seriesOf(model, key) {
    const points = model.versions.map((version) => ({ t: version.t, v: version.values[key], version }));
    const observed = model.observed[key];
    if (observed && (!model.latest || observed.t > model.latest.t)) points.push({ t: observed.t, v: observed.v, version: null });
    return points;
}

export function chartSvg(model, view, width) {
    const { height, top, base, endPad } = CHART;
    const key = view.key;
    const series = seriesOf(model, key);
    // One axis for every measure (all versions and the installation's newest records), so switching never moves a tick.
    const times = [...model.versions.map((version) => version.t), ...Object.values(model.observed).filter(Boolean).map((o) => o.t)];
    const axis = timeAxis(times, width - endPad);
    const known = series.filter((point) => point.v !== null);
    const max = niceMax(Math.max(0, ...known.map((point) => point.v)));
    const y = (v) => base - (v / max) * (base - top);
    const gapAt = axis.gap ? axis.segments[0].t1 : null;
    const runs = [];
    let run = [];
    for (const point of series) {
        const crosses = gapAt !== null && run.length > 0 && run.at(-1).point.t <= gapAt && point.t > gapAt;
        if (point.v === null || crosses) { if (run.length) runs.push(run); run = []; }
        if (point.v !== null) run.push({ x: axis.x(point.t), y: y(point.v), point });
    }
    if (run.length) runs.push(run);
    const fixed = (n) => n.toFixed(1);
    const line = runs.map((r) => r.map((q, i) => `${i ? 'L' : 'M'}${fixed(q.x)},${fixed(q.y)}`).join('')).join('');
    const area = runs.filter((r) => r.length > 1).map((r) => `M${fixed(r[0].x)},${base}${r.map((q) => `L${fixed(q.x)},${fixed(q.y)}`).join('')}L${fixed(r.at(-1).x)},${base}Z`).join('');
    // Every drawn stroke a milestone label must stay off: the line, the bridge over the gap, the chosen version's rule.
    const segments = runs.flatMap((r) => r.slice(1).map((q, i) => [r[i], q]));
    let bridge = '';
    if (gapAt !== null) {
        const before = [...known].reverse().find((point) => point.t <= gapAt), after = known.find((point) => point.t > gapAt);
        if (before && after) {
            const [a, b] = [before, after].map((point) => ({ x: axis.x(point.t), y: y(point.v) }));
            bridge = `M${fixed(a.x)},${fixed(a.y)}L${fixed(b.x)},${fixed(b.y)}`;
            segments.push([a, b]);
        }
    }
    const right = width - endPad + 8;
    const tickText = (v) => (SIZES.has(key) ? formatBytes(v) : number(v));
    const axisWidth = Math.max(AXIS_LABELS, ...[0.5, 1].map((f) => tickText(max * f).length * 6.6 + 6));
    const grid = [0.5, 1].map((f) => `<line class="evo-ch-grid" x1="0" x2="${right}" y1="${y(max * f)}" y2="${y(max * f)}"/>`
        + (known.length ? `<text x="0" y="${y(max * f) - 5}">${esc(tickText(max * f))}</text>` : '')).join('');
    const rug = model.versions.map((version) => {
        const x = fixed(axis.x(version.t));
        return `<line class="evo-ch-rug${version.milestone ? ' is-milestone' : ''}" x1="${x}" x2="${x}" y1="${base + 6}" y2="${base + (version.level === 'pre' ? 10 : 13)}"/>`;
    }).join('');
    const labels = axisLabels(axis).map((label) => `${label.tick ? `<line class="evo-ch-grid" x1="${fixed(label.x)}" x2="${fixed(label.x)}" y1="${base}" y2="${base + 4}"/>` : ''}`
        + `<text x="${fixed(label.anchor === 'start' ? label.x + 4 : label.x)}" y="${base + 30}" text-anchor="${label.anchor}">${esc(label.text)}</text>`).join('');
    const gapMarks = axis.gap ? `<path class="evo-ch-break" d="M${axis.gap[0] + 8},${base + 5}l6,-10M${axis.gap[1] - 14},${base + 5}l6,-10"/>`
        + (axis.gap[1] - axis.gap[0] >= 50 ? `<text x="${(axis.gap[0] + axis.gap[1]) / 2}" y="${base + 30}" text-anchor="middle">no releases</text>` : '') : '';
    const pins = model.milestones.filter((version) => version.values[key] !== null)
        .map((version) => ({ version, x: axis.x(version.t), y: y(version.values[key]) }));
    const chosen = view.version;
    const selected = chosen ? { x: axis.x(chosen.t), y: chosen.values[key] !== null ? y(chosen.values[key]) : base, hollow: chosen.values[key] === null } : null;
    const end = known.at(-1);
    const endText = end ? formatMeasure(key, end.v) : '';
    // What labels may not cover: the axis values, the value at the line's end, then the chosen version's own label.
    const taken = [0.5, 1].map((f) => ({ x0: 0, x1: axisWidth, y0: y(max * f) - 18, y1: y(max * f) }));
    if (end) taken.push({ x0: axis.x(end.t) + 6, x1: axis.x(end.t) + 14 + endText.length * 6.9, y0: y(end.v) - 9, y1: y(end.v) + 6 });
    // The chart names versions by their numbers (the card under it says what each one is): the chosen version, and
    // the major releases where there is room. A sliced sentence over the line said less than its number.
    const chosenText = selected ? chosenLabel(versionNumber(chosen), selected, width - endPad, [...runs.flat(), ...pins], taken, axisWidth) : '';
    if (selected) segments.push([selected, { x: selected.x, y: base + 13 }]);
    const named = milestoneLabels(pins.filter((pin) => pin.version !== chosen && pin.version.level === 'major'), taken, width - endPad,
        { segments, words: versionNumber });
    const svg = `${grid}<line class="evo-ch-grid" x1="0" x2="${right}" y1="${base}" y2="${base}"/>
        <path class="evo-ch-area" d="${area}"/>${bridge ? `<path class="evo-ch-bridge" d="${bridge}"/>` : ''}<path class="evo-ch-line" d="${line}"/>
        ${rug}${labels}${gapMarks}
        ${selected ? `<line class="evo-ch-chosen-line" x1="${fixed(selected.x)}" x2="${fixed(selected.x)}" y1="${fixed(selected.y)}" y2="${base + 13}"/>` : ''}
        ${pins.map((pin) => `<circle class="evo-ch-pin" cx="${fixed(pin.x)}" cy="${fixed(pin.y)}" r="${pin.version.level === 'minor' && !pin.version.release?.title ? 3 : 4}"/>`).join('')}
        ${named.map((label) => `<text class="evo-ch-mlabel"${authoredIf(label)} x="${fixed(label.x)}" y="${fixed(label.y)}" text-anchor="${label.anchor}">${esc(label.text)}</text>`).join('')}
        ${selected ? `<circle class="evo-ch-chosen${selected.hollow ? ' is-hollow' : ''}" cx="${fixed(selected.x)}" cy="${fixed(selected.y)}" r="5.5"/>` : ''}
        ${chosenText}
        ${end ? `<text class="evo-ch-strong" x="${fixed(axis.x(end.t) + 10)}" y="${fixed(y(end.v) + 4)}">${esc(endText)}</text>` : ''}`;
    return { svg, axis, y, pins, labels: named, height };
}

const overlaps = (a, b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

/** Whether a straight stroke from a to b passes through the box (y is linear in x, so its ends over the box bound it). */
function crosses(box, [a, b]) {
    const lo = Math.max(box.x0, Math.min(a.x, b.x)), hi = Math.min(box.x1, Math.max(a.x, b.x));
    if (lo > hi) return false;
    const at = (x) => a.y + ((b.y - a.y) * (x - a.x)) / (b.x - a.x);
    const ys = a.x === b.x ? [a.y, b.y] : [at(lo), at(hi)];
    return Math.min(...ys) < box.y1 && box.y0 < Math.max(...ys);
}

/**
 * Milestone labels by density (owner 21.10: releases and authored milestones, labels by density): every milestone keeps
 * its dot; a label is drawn right or else left of its dot, only where it touches no other label, the end value, the axis
 * values or a drawn stroke (so a rising line takes its labels on the left). Versions their authors named come first, then
 * MAJOR, then MINOR releases, newest first within each, so the narrowest chart still names the newest changes; a wider
 * one names more. Nothing is selected or hidden by this: every version stays reachable.
 */
export function milestoneLabels(pins, taken, right, { charWidth = 6.6, maxChars = 26, segments = [], words: say = versionWords } = {}) {
    const rank = (version) => (version.release?.title ? 0 : version.level === 'major' ? 1 : 2);
    const order = [...pins].sort((a, b) => rank(a.version) - rank(b.version) || b.version.t - a.version.t || b.version.i - a.version.i);
    const placed = [];
    for (const pin of order) {
        const { title: words, authored } = say(pin.version);
        const text = words.length > maxChars ? `${words.slice(0, maxChars - 1).trimEnd()}…` : words;
        const width = text.length * charWidth;
        const spot = [true, false].map((toRight) => {
            const x = toRight ? pin.x + 7 : pin.x - 7;
            return { toRight, x, box: { x0: (toRight ? x : x - width) - 3, x1: (toRight ? x + width : x) + 3, y0: pin.y - 21, y1: pin.y - 4 } };
        }).find(({ toRight, box }) => (toRight ? box.x1 - 3 <= right : box.x0 >= 0) && box.y0 >= 0
            && !taken.some((other) => overlaps(other, box)) && !segments.some((segment) => crosses(box, segment)));
        if (!spot) continue;
        taken.push(spot.box);
        placed.push({ text, x: spot.x, y: pin.y - 8, anchor: spot.toRight ? 'start' : 'end', version: pin.version, authored });
    }
    return placed;
}

// A version's own number, never authored words.
const versionNumber = (version) => ({ title: String(version?.version || ''), authored: false });

// The chosen version names itself beside its dot, above every mark the label passes over and clear of the axis labels.
const AXIS_LABELS = 48;
function chosenLabel(said, at, right, marks, taken, axisWidth = AXIS_LABELS) {
    const words = said.title;
    const toRight = at.x < right * 0.42;
    const room = toRight ? right - at.x - 10 : at.x - 10 - axisWidth;
    const max = Math.max(8, Math.floor(room / 6.9));
    const text = words.length > max ? `${words.slice(0, max - 1).trimEnd()}…` : words;
    const x0 = toRight ? at.x + 10 : at.x - 10 - text.length * 6.9, x1 = x0 + text.length * 6.9;
    const topY = marks.filter((mark) => mark.x >= x0 - 4 && mark.x <= x1 + 4).reduce((m, mark) => Math.min(m, mark.y), at.y);
    const y = Math.max(13, topY - 11);
    taken.push({ x0: x0 - 3, x1: x1 + 3, y0: y - 13, y1: y + 4 });
    return `<text class="evo-ch-label"${authoredIf(said)} x="${(toRight ? at.x + 10 : at.x - 10).toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${toRight ? 'start' : 'end'}">${esc(text)}</text>`;
}

/** The tooltip of one version: its value in the measure, its change, number, date and words (pure). */
export function tipHtml(model, version, key) {
    const value = version.values[key];
    const previous = version.i > 0 ? model.versions[version.i - 1].values[key] : null;
    const change = formatDelta(key, value, previous);
    const words = versionWords(version);
    return `<b>${esc(formatMeasure(key, value))}${change && change !== 'no change' ? ` <span class="evo-tip-delta">${esc(change)}</span>` : ''}</b>`
        + `<span>${esc(version.version)} · ${esc(fullDate(version.t))}</span><span class="evo-tip-head"${authoredIf(words)}>${esc(words.title)}</span>`;
}

const ICON_BACK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m15 18-6-6 6-6"/></svg>';
const ICON_NEXT = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>';

function neighbourMilestone(model, version, step) {
    const list = model.milestones;
    return step < 0 ? [...list].reverse().find((m) => m.t < version.t || (m.t === version.t && m.i < version.i)) || null
        : list.find((m) => m.t > version.t || (m.t === version.t && m.i > version.i)) || null;
}

/** The chosen version beneath the chart: its number, words, numbers and source (pure; every text escaped). */
export function detailHtml(model, version, full) {
    const words = versionWords(version);
    const previous = version.i > 0 ? model.versions[version.i - 1] : null;
    const facts = ['code_lines', 'bible_bytes', 'system_bytes'].map((key) => {
        const value = version.values[key];
        const change = previous ? formatDelta(key, value, previous.values[key]) : '';
        return `<div class="evo-fact"><span class="evo-meta">${MEASURE[key].label}</span>`
            + `<span class="evo-fact-value">${esc(formatMeasure(key, value))}${change ? `<span class="evo-fact-delta">${esc(change)}</span>` : ''}</span></div>`;
    }).join('');
    const back = model.versions[version.i - 1], next = model.versions[version.i + 1];
    return `<div class="evo-detail-top"><span class="evo-detail-version">${esc(version.version)}</span>`
        + `<span class="evo-meta">${esc(fullDate(version.t))} · ${esc(words.kind)}${version.milestone ? ' · Milestone' : ''}</span>`
        + `<span class="evo-detail-nav"><button type="button" class="btn btn-ghost btn-sm evo-quiet" data-step="-1" aria-label="Previous version"${back ? '' : ' disabled'}>${ICON_BACK}</button>`
        + `<button type="button" class="btn btn-ghost btn-sm evo-quiet" data-step="1" aria-label="Next version"${next ? '' : ' disabled'}>${ICON_NEXT}</button></span></div>`
        + `<h4 class="evo-detail-title"${authoredIf(words)}>${esc(words.title)}</h4>`
        + (words.lead ? `<p class="evo-detail-lead"${AUTHORED}>${esc(words.lead)}</p>` : '')
        + (words.body ? `<p class="evo-detail-body${full ? ' is-full' : ''}" data-body${AUTHORED}>${esc(words.body)}</p>` : '')
        + `<div class="evo-facts" aria-label="Measured at this version${previous ? `, change since ${esc(previous.version)}` : ''}">${facts}</div>`
        + `<div class="evo-detail-source"><span class="evo-meta">${previous ? `Change since ${esc(previous.version)} · ` : ''}Source: ${esc(sourceLabel(version))}</span>`
        + (words.body ? `<button type="button" class="evo-link ui-hit-area" data-full hidden>${full ? 'Show less' : 'Show more'}</button>` : '') + '</div>';
}

function caption(model, key) {
    const measure = MEASURE[key];
    if (!measure.install) {
        return `${measure.label}: ${measure.what}. Dots are milestones (major and minor releases, and any version its author `
            + `named), labelled where there is room; the ticks under the line are all ${model.versions.length} versions.`;
    }
    const since = model.firstAt[key] ? ` Recorded since ${fullDate(Date.parse(model.firstAt[key]))};` : ' Nothing recorded yet;';
    return `This installation · ${measure.label}: ${measure.what}, at each version’s date.${since} dates without a record are gaps.`;
}

// ---- When a cycle may follow your tasks: one row over two keys (settings_rows.js) ----------------------------------
const CADENCE = /^(off|llm|every_n:[1-9]\d*)$/;
const isOn = (value) => /^(1|true|yes|on)$/i.test(String(value ?? '').trim());
/** The choice a stored pair [enabled, cadence] stands for: off, llm (after each task) or every_n. */
export const whenMode = ([on, cadence] = []) => (!isOn(on) || cadence === 'off' ? 'off' : String(cadence).startsWith('every_n:') ? 'every_n' : 'llm');
// A malformed stored cadence reads as llm, the server's own reading (config.get_post_task_evolution_cadence).
CODECS.evolutionWhen = {
    read: (settings, row, cadence = String(settings?.OUROBOROS_POST_TASK_EVOLUTION_CADENCE ?? 'llm').trim().toLowerCase()) =>
        (settings ? [isOn(settings.OUROBOROS_POST_TASK_EVOLUTION), CADENCE.test(cadence) ? cadence : 'llm'] : undefined),
    body: ([on, cadence]) => ({ OUROBOROS_POST_TASK_EVOLUTION: Boolean(on), OUROBOROS_POST_TASK_EVOLUTION_CADENCE: cadence }),
};
const everyWords = (value, n = Number(String(value?.[1]).slice(8))) => (n === 1 ? 'every task' : `every ${n} tasks`);
export const whenName = (pair) => ({ off: 'Off', llm: 'After each task', every_n: capital(everyWords(pair)) }[whenMode(pair)]);
// Checks the count, then writes as the generic writer does. "was" names the old pair, which rides on the name for Undo.
WRITERS.evolutionWhen = async (row, value, options, undo = value?.pair) => {
    if (!undo && !CADENCE.test(String(value?.[1]))) return { state: 'refused', effect: '', withheld: '', message: 'Enter a whole number of tasks, 1 or more.', code: 'invalid_value' };
    const outcome = await WRITERS.generic(row, undo ? [isOn(undo[0]), undo[1]] : value, options);
    return outcome.old === undefined ? outcome : { ...outcome, old: Object.assign([whenName(outcome.old)], { pair: outcome.old }) };
};
GUARDED_WRITERS.add('evolutionWhen');
/** The choice in its own sentence (dashboard.js `words`): "every N" says only when it asks itself (post_task_evolution.py). */
export function whenWords(settings, value = CODECS.evolutionWhen.read(settings)) {
    const decides = 'it decides whether one improvement is worth a cycle.';
    return { OUROBOROS_POST_TASK_EVOLUTION: { caption: !value ? '' : { off: 'It starts no cycle by itself.',
        llm: `After each task ${decides}`, every_n: `After ${everyWords(value)}, chats included, ${decides}` }[whenMode(value)] } };
}

// ---- The page ------------------------------------------------------------------------------------------------------

function element(tag, className, attrs = {}, text = '') {
    const node = document.createElement(tag);
    if (className) node.className = className;
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
    if (text) node.textContent = text;
    return node;
}
const press = (buttons, chosen) => buttons.forEach((button) => button.setAttribute('aria-pressed', String(button.classList.toggle('active', chosen(button)))));

/**
 * Mount the page. `stateSnapshots` is app.js's sequencer of every `/api/state` read on the page; app.js hands each of
 * its answers (and each failed read, as null) to the returned `hydrateStateSnapshot`.
 */
export function initEvolution({ ws, state, mount, stateSnapshots = null, dialog = openConfirmDialog, openDashboardTab = null }) {
    // One Dashboard tab body is the tab's root and scroller; its three objects are its cards.
    const page = element('div', 'dashboard-body scroll-fade-y', { id: 'page-evolution' });

    // The campaign object: built node by node, so its controls keep identity and focus across repaints.
    const campaign = element('section', 'evo-campaign ui-card', { 'aria-label': 'Evolution campaign' });
    const main = element('div', 'evo-campaign-main');
    const head = element('div', 'evo-campaign-line');
    const name = element('h3', 'dashboard-card-title', {}, 'Evolution campaign');
    const chip = element('span', 'ui-chip evo-campaign-chip', { id: 'evo-campaign-chip', 'data-tone': 'neutral' });
    chip.hidden = true;
    head.append(name, chip);
    const note = element('p', 'evo-meta', { id: 'evo-campaign-note' }, 'Reading the runtime state…');
    const lines = element('div', 'evo-campaign-lines');
    // A failed read never passes an earlier answer off as current: it says how old the shown state is.
    const freshness = element('p', 'ui-status', { id: 'evo-campaign-freshness', 'data-tone': 'warn' });
    freshness.hidden = true;
    main.append(head, note, lines, freshness);
    const side = element('div', 'evo-campaign-side');
    // Consciousness once, as a quiet pointer to its own tab (U071): its detail and allowance are that tab's.
    // Its name is its visible state and the › that opens the tab (M16: a label over it would hide the state).
    const mind = element('button', 'btn btn-ghost btn-sm ui-status evo-consciousness',
        { id: 'evo-consciousness', type: 'button', 'data-tone': 'neutral' });
    mind.hidden = true;
    mind.addEventListener('click', () => openDashboardTab?.('consciousness'));
    side.append(mind);
    const actions = element('div', 'evo-campaign-actions');
    const startBtn = element('button', 'btn btn-primary', { id: 'evo-start', type: 'button' }, 'Start campaign…');
    // Pause keeps its place: offered while a campaign runs (or may), present and inactive while there is nothing to
    // pause, so a state change never moves its neighbours (DESIGN principle 6).
    const stopBtn = element('button', 'btn btn-default', { id: 'evo-stop', type: 'button' }, 'Pause');
    // A deep self-review beside the campaign (owner 25.10, 5A): the one reviewer picker `/review` uses.
    const reviewBtn = element('button', 'btn btn-default', { id: 'evo-review', type: 'button' }, 'Run review…');
    actions.append(startBtn, stopBtn, reviewBtn);
    campaign.append(main, side, actions);

    const story = element('section', 'evo-story ui-card', { 'aria-labelledby': 'evo-story-title' });
    const storyHead = element('div', 'evo-story-head');
    const storyWords = element('div', 'evo-story-words');
    const storyTitle = element('h3', 'dashboard-card-title', { id: 'evo-story-title' }, 'How Ouroboros changed');
    const statusLine = element('p', 'evo-story-status', { 'aria-live': 'polite' });
    storyWords.append(storyTitle, statusLine);
    const measures = element('div', 'evo-measures', { role: 'group', 'aria-label': 'Measure' });
    measures.hidden = true;
    MEASURES.forEach((measure, index) => {
        if (index === 3) measures.append(element('span', 'evo-measures-sep', { 'aria-hidden': 'true' }));
        const button = element('button', `ui-segment evo-measure${index === 0 ? ' active' : ''}`,
            { type: 'button', 'data-measure': measure.key, 'aria-pressed': String(index === 0) });
        if (measure.install) button.setAttribute('aria-label', `${measure.label}, this installation`);
        button.textContent = measure.label;
        measures.append(button);
    });
    storyHead.append(storyWords, measures);
    // The measures are one single choice (the shared segments): pressed choices of one view, not tabs (no panels,
    // nothing to select by arrows). On a narrow card they wrap onto a second line, so none hides past the edge (U125).
    const body = element('div', 'evo-story-body');
    story.append(storyHead, body);
    // The cycles' settings (settings_rows.js), drawn once: no repaint touches them. The direction is a card of its own.
    const cycles = element('div', 'dashboard-settings-cards', { 'data-evo-settings': '' });
    page.append(campaign, cycles, story);
    mount.appendChild(page);
    const settings = mountTabSettings(cycles, 'evolution', { state, words: whenWords, onPaint: paintWhen,
        cards: [{ label: 'Cycles after your tasks' }, { label: 'What to work on', solo: true, rows: ['OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE'] }],
        custom: (row) => `<div class="ui-segment-group" role="group" aria-labelledby="setting-${row.id}-name">${row.control.options.map((option) =>
            `<button type="button" class="ui-segment" data-setting-control data-evo-when="${option.value}" aria-pressed="false">${option.label}</button>`).join('')}</div>`
            + '<span class="settings-row-field" data-evo-every hidden><span class="settings-row-unit">N</span><input class="ui-control" data-evo-n value="3" inputmode="numeric" autocomplete="off" aria-label="Every how many tasks"></span>' });
    cycles.querySelector('#setting-OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE')?.setAttribute('placeholder', 'For example: faster start, fewer repeated questions');
    function paintWhen(read) {
        const value = CODECS.evolutionWhen.read(read), count = cycles.querySelector('[data-evo-n]');
        if (!value || !count) return;
        press(cycles.querySelectorAll('[data-evo-when]'), (button) => button.dataset.evoWhen === whenMode(value));
        count.parentElement.hidden = whenMode(value) !== 'every_n';
        if (whenMode(value) === 'every_n' && document.activeElement !== count) count.value = value[1].slice(8);
    }
    // A choice is saved when it is pressed; the count when its field is left or on Enter.
    const onWhen = (event) => {
        const pressed = event.target.closest?.('[data-evo-when]');
        const mode = event.type === 'click' ? pressed && !pressed.classList.contains('active') && pressed.dataset.evoWhen
            : event.target.matches?.('[data-evo-n]') && 'every_n';
        if (mode) void commitSetting(rowById('OUROBOROS_POST_TASK_EVOLUTION'), mode === 'off' ? [false, 'llm']
            : [true, mode === 'llm' ? 'llm' : `every_n:${cycles.querySelector('[data-evo-n]').value.trim()}`]);
    };
    for (const type of ['click', 'change']) cycles.addEventListener(type, onWhen);

    const view = { key: 'code_lines', version: null, full: false };
    let model = null;
    let history = null;        // the last answer of /api/evolution-data
    let historyError = '';     // the last request's own failure, '' after an answer
    let runtime = null;
    let runtimeAt = '';        // when the shown runtime answer arrived
    let runtimeError = false;  // the latest read failed: until a new answer the state is not known to be current
    let geometry = null;
    let disposed = false;
    let commandPending = false;
    // A Start the gateway queued (or may have queued) that no state answer has shown applied yet: /api/command only
    // queues it, so the answer read right after can still say off. Pause stays offered until the campaign is seen on,
    // a Pause is queued after it, or a read taken START_WINDOW_MS after the Start still shows it off: the runtime
    // refused it (its reason is in chat), and the chip says the state again with Start offered. The time it was sent.
    const START_WINDOW_MS = 10000;
    let startUnconfirmed = 0;
    let historySeq = 0, runtimeSeq = 0, pollTimer = 0, startTimer = 0;
    let resizeObserver = null;
    let painted = '';          // what the chart and the detail show: an equal answer repaints only the status line

    const visible = () => !document.hidden && state.activePage === 'dashboard' && state.dashboardActiveSubtab === 'evolution';

    // ---- Campaign ----
    function paintCampaign() {
        const facts = campaignView(runtime);
        lines.replaceChildren();
        freshness.hidden = !(facts && runtimeError);
        freshness.textContent = facts && runtimeError
            ? `Couldn’t read the runtime state again; this is the answer from ${whenLabel(runtimeAt)}.` : '';
        // Off offers Start; on, Pause and a new objective (the same command re-targets a running campaign in place); an
        // unknown control both Start and Pause: the owner may need either. So does off while a Start is unconfirmed,
        // and then the chip says the Start, never "Off" beside an offered Pause (U072); Start stays pressable, since a
        // Start that never applies must stay retryable.
        const starting = Boolean(facts) && startUnconfirmed > 0 && facts.enabled !== true;
        const pausable = Boolean(facts) && (facts.enabled !== false || startUnconfirmed > 0);
        if (!facts) {
            chip.hidden = true;
            note.textContent = runtimeError ? 'The runtime state could not be read. It is read again when this tab opens.' : 'Reading the runtime state…';
            mind.hidden = true;
        } else {
            chip.hidden = false;
            chip.textContent = starting ? 'Starting…' : facts.chip.text;
            chip.setAttribute('data-tone', starting ? 'neutral' : facts.chip.tone);
            note.textContent = facts.note;
            for (const text of facts.lines) {
                const line = element('p', 'evo-meta');
                line.textContent = text;
                lines.append(line);
            }
            mind.hidden = false;
            mind.textContent = `${facts.consciousness.text} ›`;
            mind.setAttribute('data-tone', facts.consciousness.tone);
            startBtn.textContent = facts.enabled === true ? 'New objective…' : 'Start campaign…';
        }
        const lightMode = Boolean(facts?.lightMode);
        // A paid start waits for a current answer: not while the state is loading or its latest read failed.
        startBtn.disabled = lightMode || commandPending || !facts || runtimeError;
        stopBtn.disabled = commandPending || !pausable;
        startBtn.title = lightMode ? "Evolution campaigns need runtime mode 'advanced' or 'pro' (currently 'light')." : '';
    }

    // One state answer, or null for a failed read: an earlier answer stays on screen, marked with its age, until a
    // later one replaces it.
    function hydrateStateSnapshot(answer) {
        if (disposed) return;
        if (answer && typeof answer === 'object') {
            runtime = answer;
            runtimeAt = new Date().toISOString();
            runtimeError = false;
            if (campaignView(runtime).enabled === true || Date.now() - startUnconfirmed >= START_WINDOW_MS) startUnconfirmed = 0;
        } else runtimeError = true;
        paintCampaign();
    }

    async function loadRuntime() {
        if (disposed) return;
        // Through the page's clock, an older answer never lands over a newer one; a read that joined another's comes
        // back null, and that read's answer reaches hydrateStateSnapshot from app.js.
        const request = stateSnapshots ? await stateSnapshots.gate(true) : ++runtimeSeq;
        if (!request) return;
        if (stateSnapshots) {
            await readStateSnapshot(stateSnapshots, request);
            return;
        }
        let answer = null;
        try {
            const { response, state: body } = await readState();
            if (!response.ok || body === undefined) throw new Error(`State API error ${response.status}`);
            answer = body;
        } catch (error) {
            console.error('Evolution runtime state error:', error);
        }
        if (request === runtimeSeq) hydrateStateSnapshot(answer);
    }

    // One command at a time: Start and Pause stay disabled until the gateway answers (the objective dialog is modal, so
    // it needs no extra gate). The command's own outcome arrives in chat: a refusal here means only that it was never sent.
    async function sendCommand(cmd, button, control) {
        commandPending = true;
        paintCampaign();
        const start = cmd.startsWith('/evolve on');
        let refused = false;
        try {
            const resp = await apiFetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cmd, ui_control: control }),
            });
            if (!resp.ok) {
                refused = true;
                const answer = await resp.json().catch(() => ({}));
                throw new Error(answer?.error || `HTTP ${resp.status}`);
            }
            startUnconfirmed = start ? Date.now() : 0;  // queued: a Start waits to be seen applied; a queued Pause supersedes it
        } catch (error) {
            // A refused command was never queued; without an answer the gateway may still have queued a Start.
            if (start && !refused) startUnconfirmed = Date.now();
            showToast(`Could not send ${cmd.split(' ', 2).join(' ')}: ${error?.message || error}`, 'error');
        } finally {
            commandPending = false;
            paintCampaign();
            // Disabling the pressed button dropped keyboard focus to the page; give it back.
            if (document.activeElement === document.body && !button.disabled && !button.hidden) button.focus({ preventScroll: true });
        }
        loadRuntime();
        // The read that ends an unapplied Start's window, if nothing reads the state again before then.
        clearTimeout(startTimer);
        if (startUnconfirmed) {
            startTimer = setTimeout(loadRuntime, START_WINDOW_MS);
            startTimer?.unref?.();
        }
    }
    startBtn.addEventListener('click', async () => {
        if (commandPending) return;
        const running = campaignView(runtime)?.enabled === true;
        const asked = await promptCampaignObjective({ dialogImpl: dialog, running });
        // Cancel = no campaign start (semantic fix, owner-approved Б1-8: the
        // old `prompt() || ''` started a paid campaign even on Cancel/null).
        // A running campaign asked for nothing new keeps its objective: nothing to send.
        if (!asked.confirmed || commandPending || (running && !asked.objective)) return;
        // The press composes the command; only the objective my human typed is theirs (#2778).
        await sendCommand(`/evolve on${asked.objective ? ` ${asked.objective}` : ''}`, startBtn,
            { name: 'evolution_start', ...(asked.objective ? { owner_text: asked.entered } : {}) });
    });
    stopBtn.addEventListener('click', () => {
        if (!commandPending) sendCommand('/evolve off', stopBtn, { name: 'evolution_stop' });
    });
    reviewBtn.addEventListener('click', () => chooseAndSendReview({ openConfirmDialog: dialog, ws }));

    // ---- History ----
    function statusHtml() {
        const refresh = model?.refresh || history?.refresh || {};
        const parts = [];
        if (model?.versions.length) parts.push(`<span class="evo-meta">${esc(`${model.versions.length} versions since ${fullDate(model.versions[0].t)}`)}</span>`);
        if (historyError) {
            parts.push(`<span class="ui-status" data-tone="warn">${esc(model ? `Couldn’t reach the server; this is the copy saved ${whenLabel(model.savedAt)}` : 'Couldn’t reach the server')}</span>`);
        } else if (refresh.state === 'running') {
            parts.push(`<span class="evo-progress">${statusMarkHtml('', { shape: 'moving', hue: 'ink' })}<span>${model?.versions.length ? 'Checking for new versions…' : 'Measuring every version…'}</span></span>`);
        } else if (refresh.state === 'failed') {
            parts.push(`<span class="ui-status" data-tone="warn">Couldn’t check for new versions</span>`);
        } else if (model?.savedAt) {
            parts.push(`<span class="evo-meta">${esc(`Saved ${whenLabel(model.savedAt)}`)}</span>`);
        }
        // Versions whose commit could not be read keep "Not measured" and are said here; Try again re-measures just them.
        const partial = !historyError && refresh.state !== 'running' && refresh.unmeasured > 0;
        if (partial) {
            const reason = String(refresh.unmeasured_error || '');
            parts.push(`<span class="ui-status" data-tone="warn">${esc(`${count(refresh.unmeasured, 'version')} couldn’t be measured${reason ? `: ${reason}` : ''}`)}</span>`);
        }
        if (historyError || refresh.state === 'failed' || partial) parts.push('<button type="button" class="evo-link ui-hit-area" data-retry>Try again</button>');
        return parts.join('<span class="evo-meta" aria-hidden="true">·</span>');
    }

    function paintHistory() {
        statusLine.innerHTML = statusHtml();
        statusLine.querySelector('[data-retry]')?.addEventListener('click', () => loadHistory({ force: !historyError }));
        const refresh = history?.refresh || {};
        const shown = model?.versions.length ? `${model.savedAt}|${JSON.stringify(history?.install || {})}` : '';
        if (shown && shown === painted && body.querySelector('[data-chart]')) return;
        painted = shown;
        if (model?.versions.length) {
            measures.hidden = false;
            if (!body.querySelector('[data-chart]')) {
                body.innerHTML = `<figure class="evo-chart" data-chart><svg data-svg role="img"></svg>
                    <div class="evo-chart-hit" data-hit tabindex="0" role="slider" aria-label="Versions on the time line. Arrow keys move one version; Page Up and Page Down move between milestones."></div>
                    <div class="evo-tip" data-tip hidden></div></figure>
                    <details class="ui-disclosure evo-method"><summary>How this is measured</summary><p class="evo-meta evo-caption" data-caption></p></details>
                    <article class="evo-detail" data-detail aria-live="polite"></article>`;
                wireChart();
            }
            view.version ||= model.latest;
            draw();
            paintDetail();
            return;
        }
        measures.hidden = true;
        resizeObserver?.disconnect();
        geometry = null;
        let words;
        if (!history && !historyError) words = '<p class="evo-empty-line">Reading the saved history…</p>';
        else if (refresh.state === 'running') words = '<p class="evo-empty-line">This is the first measurement on this install: every release tag is read from Git once, and later visits show the saved copy at once.</p>';
        else if (historyError || refresh.state === 'failed') {
            const reason = historyError || refresh.error || '';
            words = `<p class="evo-empty-line">The version history didn’t load, and there is no saved copy on this install yet. The campaign controls above don’t depend on it.</p>`
                + (reason ? `<p class="evo-meta">${esc(reason)}</p>` : '');
        } else words = '<p class="evo-empty-line">No versions yet: this repository has no release tags.</p>';
        body.innerHTML = `<div class="evo-empty">${words}</div>`;
    }

    function draw() {
        const figure = body.querySelector('[data-chart]');
        const width = figure?.clientWidth || 0;
        if (!figure || !width || !model?.versions.length) return;
        const chart = chartSvg(model, view, width);
        const svg = figure.querySelector('[data-svg]');
        svg.setAttribute('viewBox', `0 0 ${width} ${chart.height}`);
        svg.setAttribute('height', String(chart.height));
        svg.setAttribute('aria-label', `${MEASURE[view.key].label} at each version, ${fullDate(model.versions[0].t)} to ${fullDate(model.latest.t)}`);
        svg.innerHTML = chart.svg;
        geometry = chart;
        const hit = figure.querySelector('[data-hit]');
        const words = versionWords(view.version);
        hit.setAttribute('aria-valuemin', '1');
        hit.setAttribute('aria-valuemax', String(model.versions.length));
        hit.setAttribute('aria-valuenow', String(view.version.i + 1));
        hit.setAttribute('aria-valuetext', `${view.version.version}, ${fullDate(view.version.t)}: ${words.title}`);
        body.querySelector('[data-caption]').textContent = caption(model, view.key);
    }

    // "Show more" only while the excerpt is actually cut, or while it is open; a narrower window can start cutting it.
    function syncShowMore() {
        const host = body.querySelector('[data-detail]');
        const more = host?.querySelector('[data-full]');
        const text = host?.querySelector('[data-body]');
        if (more && text) more.hidden = !view.full && text.scrollHeight <= text.clientHeight + 1;
    }

    function paintDetail() {
        const host = body.querySelector('[data-detail]');
        if (!host || !view.version) return;
        // The repaint replaces the controls: a keyboard user keeps focus on the same one (its sibling if it is now off).
        const focused = host.contains(document.activeElement) ? document.activeElement : null;
        const again = focused?.dataset?.step ? `[data-step="${focused.dataset.step}"]` : focused?.hasAttribute?.('data-full') ? '[data-full]' : '';
        host.innerHTML = detailHtml(model, view.version, view.full);
        host.querySelectorAll('[data-step]').forEach((button) => button.addEventListener('click', () => choose(model.versions[view.version.i + Number(button.dataset.step)])));
        host.querySelector('[data-full]')?.addEventListener('click', () => { view.full = !view.full; paintDetail(); });
        syncShowMore();
        if (again) {
            const target = [host.querySelector(again), ...host.querySelectorAll('[data-step]')].find((node) => node && !node.disabled && !node.hidden);
            (target || body.querySelector('[data-hit]'))?.focus({ preventScroll: true });
        }
    }

    function choose(version) {
        if (!version) return;
        view.version = model.versions[version.i] || version;
        view.full = false;
        draw();
        paintDetail();
    }

    // Prefer nearby milestone dots. Overlapping dates remain reachable through the version controls and keyboard.
    function nearest(px) {
        const pin = geometry.pins.reduce((best, q) => (!best || Math.abs(q.x - px) < Math.abs(best.x - px) ? q : best), null);
        if (pin && Math.abs(pin.x - px) <= 3) return pin.version;
        let best = null, distance = Infinity;
        for (const version of model.versions) {
            const d = Math.abs(geometry.axis.x(version.t) - px);
            if (d < distance) { distance = d; best = version; }
        }
        return best;
    }

    function wireChart() {
        const figure = body.querySelector('[data-chart]');
        if (!figure) return;
        const hit = figure.querySelector('[data-hit]'), tip = figure.querySelector('[data-tip]'), svg = figure.querySelector('[data-svg]');
        const pointerX = (event) => event.clientX - hit.getBoundingClientRect().left;
        hit.addEventListener('pointermove', (event) => {
            if (!geometry) return;
            const version = nearest(pointerX(event));
            const x = geometry.axis.x(version.t);
            const value = version.values[view.key];
            tip.hidden = false;
            tip.innerHTML = tipHtml(model, version, view.key);
            const y = value !== null ? geometry.y(value) : CHART.base;
            let left = x + 14;
            if (left + tip.offsetWidth > figure.clientWidth) left = x - 14 - tip.offsetWidth;
            tip.style.left = `${Math.max(0, left)}px`;
            tip.style.top = `${y - tip.offsetHeight - 12 < 0 ? y + 14 : y - tip.offsetHeight - 12}px`;
            let cross = svg.querySelector('.evo-ch-cross');
            if (!cross) {
                cross = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                cross.setAttribute('class', 'evo-ch-cross');
                svg.prepend(cross);
            }
            for (const [name, value_] of [['x1', x], ['x2', x], ['y1', CHART.top - 6], ['y2', CHART.base]]) cross.setAttribute(name, String(value_));
        });
        hit.addEventListener('pointerleave', () => { tip.hidden = true; svg.querySelector('.evo-ch-cross')?.remove(); });
        hit.addEventListener('click', (event) => { if (geometry) choose(nearest(pointerX(event))); });
        hit.addEventListener('keydown', (event) => {
            const at = view.version.i, last = model.versions.length - 1;
            const target = { ArrowRight: at + 1, ArrowUp: at + 1, ArrowLeft: at - 1, ArrowDown: at - 1, Home: 0, End: last }[event.key];
            if (target !== undefined) choose(model.versions[Math.min(last, Math.max(0, target))]);
            else if (event.key === 'PageUp' || event.key === 'PageDown') choose(neighbourMilestone(model, view.version, event.key === 'PageDown' ? 1 : -1));
            else return;
            event.preventDefault();
        });
        resizeObserver?.disconnect();
        if (typeof ResizeObserver === 'function') {
            resizeObserver = new ResizeObserver(() => { draw(); syncShowMore(); });
            resizeObserver.observe(figure);
        }
    }

    measures.addEventListener('click', (event) => {
        const button = event.target.closest?.('[data-measure]');
        if (!button) return;
        view.key = button.dataset.measure;
        press(measures.querySelectorAll('[data-measure]'), (b) => b === button);
        // A tooltip still under the resting pointer read the previous measure's value: it goes until the pointer moves.
        const tip = body.querySelector('[data-tip]');
        if (tip) tip.hidden = true;
        body.querySelector('[data-svg] .evo-ch-cross')?.remove();
        draw();
    });

    function schedulePoll() {
        clearTimeout(pollTimer);
        pollTimer = 0;
        // Only while a refresh runs and the tab is on screen; the next visit asks again anyway.
        if (!disposed && history?.refresh?.state === 'running' && visible()) pollTimer = setTimeout(() => loadHistory(), 2000);
    }

    async function loadHistory({ force = false } = {}) {
        if (disposed) return;
        const seq = ++historySeq;
        try {
            const response = await apiFetch(`/api/evolution-data${force ? '?force=1' : ''}`, { cache: 'no-store' });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const answer = await response.json();
            if (disposed || seq !== historySeq) return;
            history = answer;
            historyError = '';
            model = historyModel(answer);
            // The chosen version survives a new answer by identity (its tag and commit), else the newest is chosen.
            view.version = model.versions.find((version) => version.tag === view.version?.tag && version.commit === view.version?.commit) || null;
        } catch (error) {
            if (disposed || seq !== historySeq) return;
            console.error('Evolution history error:', error);
            historyError = String(error?.message || error);  // the saved copy on screen stays
        }
        paintHistory();
        schedulePoll();
    }

    function loadAll() {
        if (!visible()) return;
        loadRuntime();
        loadHistory();
    }

    paintCampaign();
    paintHistory();
    const unsubscribeOpen = ws.on('open', loadAll);
    const onPageShown = (event) => { if (event?.detail?.page === 'dashboard') loadAll(); };
    const onSubtabShown = (event) => { if (event?.detail?.tab === 'evolution') loadAll(); };
    const onVisibilityChange = () => { if (!document.hidden) loadAll(); else clearTimeout(pollTimer); };
    window.addEventListener('ouro:page-shown', onPageShown);
    window.addEventListener('ouro:dashboard-subtab-shown', onSubtabShown);
    document.addEventListener('visibilitychange', onVisibilityChange);
    return {
        hydrateStateSnapshot,
        dispose() {
            if (disposed) return;
            disposed = true;
            historySeq += 1;
            runtimeSeq += 1;
            clearTimeout(pollTimer);
            clearTimeout(startTimer);
            resizeObserver?.disconnect();
            settings.dispose();
            unsubscribeOpen();
            window.removeEventListener('ouro:page-shown', onPageShown);
            window.removeEventListener('ouro:dashboard-subtab-shown', onSubtabShown);
            document.removeEventListener('visibilitychange', onVisibilityChange);
        },
    };
}
