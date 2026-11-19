// The one set of number, money, size, date and duration formats of the Dashboard tabs (Storage's gauge and
// free-space tray, Costs, Consciousness, the Updates record): decimal units, a fixed English date and time
// inside the sentence (never the browser's locale, which mixed languages in one line), counts and money
// grouped like the rest of the app; and the one "Largest parts" list a place and a deleted project's folder
// both open.

import { fmt } from './i18n.js';
import { statusMarkHtml } from './status_mark.js';
import { escapeHtmlAttr, escapeHtmlText, formatUsdKnown } from './utils.js';

/** Counts in one fixed format, like the file counts of result_files.js. */
export const number = (n) => Number(n || 0).toLocaleString('en-US');

/** Money grouped like a count ($29,438.20) under the app's one amount rule (utils.js `formatUsdKnown`): four places
 *  under a cent and `<$0.0001` below that, so a real cost never reads $0.00; an absent amount is the dash. */
export function formatUsd(value) {
    const amount = Number(value);
    return Number.isFinite(amount) && amount >= 999.995  // a cent below $1,000 rounds into the grouped form
        ? `$${amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : formatUsdKnown(value);
}

/** Money that stands beside a limit (the limit cards of Costs, the consciousness allowance, the Costs rows of Settings'
 *  and Dashboard's lists, Main's budget), one rule for every such place. The precision follows the LIMIT: a limit in
 *  whole dollars (or none) reads in whole dollars, the spend (ROLE 'spent', also an upper bound) rounded UP and what
 *  is left (ROLE 'left') rounded DOWN, so the two add up to the limit and the room is never overstated; a limit set
 *  with cents keeps cents for the limit, the spend and what is left, rounded the same ways. A positive amount that
 *  would round to nothing is said in words (`less than $1`, with cents `less than $0.01`). An amount that is not
 *  known is the dash, never $0. Thousands are grouped ($42,088). A task's or a call's cost is not this: `formatUsd`. */
export function formatLimitUsd(value, limit, role = 'spent') {
    if (value === null || value === undefined || value === '') return '—';
    const amount = Number(value);
    if (!Number.isFinite(amount)) return '—';
    const cap = Number(limit);
    const unit = Number.isFinite(cap) && cap > 0 && !Number.isInteger(Math.round(cap * 100) / 100) ? 0.01 : 1;
    const steps = amount / unit;
    // A hair of float error (42087.99999…) is not a dollar of spend; `|| 0` keeps a rounded zero from reading -0.
    const rounded = ((role === 'left' ? Math.floor(steps + 1e-6) : role === 'limit' ? Math.round(steps) : Math.ceil(steps - 1e-6)) * unit) || 0;
    if (amount > 0 && (rounded === 0 || (role === 'spent' && amount < unit))) return unit === 1 ? 'less than $1' : 'less than $0.01';
    const digits = unit === 1 ? 0 : 2;
    return `$${rounded.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
}

export const plural = (n, one, many) => (Number(n) === 1 ? fmt(one, { n }) : fmt(many, { n, count: number(n) }));

// Work that runs now (a count, a listing, a cleanup): the shared moving mark in the line's ink, beside the
// sentence that says what is working.
export const DOTS = statusMarkHtml('', { shape: 'moving', hue: 'ink' });

export const isSize = (value) => typeof value === 'number' && Number.isFinite(value);

/** Decimal units, one decimal below 10 of a unit (1.0 MB, 1.4 MB, 12 MB, 4.0 TB) so neighbouring rows stay
 *  distinguishable. Null or non-numeric stays empty: unknown is never zero. */
export function formatBytes(bytes) {
    if (!isSize(bytes)) return '';
    if (bytes < 1e3) return plural(bytes, '{n} byte', '{count} bytes');
    // The unit is chosen after rounding, so 999.96 GB reads 1.0 TB rather than 1000 GB.
    const [unit, size] = bytes >= 999.5e9 ? ['TB', 1e12] : bytes >= 999.5e6 ? ['GB', 1e9] : bytes >= 999.5e3 ? ['MB', 1e6] : ['KB', 1e3];
    const value = bytes / size;
    return `${value < 9.95 ? value.toFixed(1) : Math.round(value)} ${unit}`;
}

export const signedBytes = (bytes) => `${bytes < 0 ? '−' : '+'}${formatBytes(Math.abs(bytes))}`;

/** A row's "Largest parts": each part of a known size, then the rest when there is one. KEY and OPEN are the
 *  disclosure's and its summary's `data-key`, which keep it open and focused on the same row when the list is drawn again. */
export function largestPartsHtml(parts, rest, key, open) {
    const known = Array.isArray(parts) ? parts.filter((part) => isSize(part?.bytes)) : [];
    if (!known.length) return '';
    const items = known.map((part) => `<li><span data-i18n-authored>${escapeHtmlText(part.name)}</span><span data-i18n-fmt="storage.size">${escapeHtmlText(formatBytes(part.bytes))}</span></li>`);
    if (rest?.count > 0) items.push(`<li class="storage-parts-rest" data-i18n-fmt="storage.rest">${escapeHtmlText(fmt('and {n} more, {size}', { n: number(rest.count), size: formatBytes(rest.bytes) }))}</li>`);
    return `<details class="activity-history storage-parts" data-key="${escapeHtmlAttr(key)}">
            <summary data-key="${escapeHtmlAttr(open)}">Largest parts</summary>
            <ul class="storage-parts-list">${items.join('')}</ul>
        </details>`;
}

/** "today at 18:40", "16 Oct at 18:10": one fixed format inside the English sentence. */
export function whenLabel(iso, now = new Date()) {
    const date = new Date(iso);
    if (!iso || Number.isNaN(date.getTime())) return '';
    const time = date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
    if (date.toDateString() === now.toDateString()) return fmt('today at {time}', { time });
    return fmt('{date} at {time}', { date: date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }), time });
}

export function durationLabel(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value < 0) return '';
    if (value < 60) return fmt('{seconds} s', { seconds: value < 10 ? value.toFixed(1) : String(Math.round(value)) });
    const minutes = Math.round(value / 60);
    if (minutes < 60) return fmt('{minutes} min', { minutes });
    return fmt('{hours} h {minutes} min', { hours: Math.floor(minutes / 60), minutes: minutes % 60 });
}
