// The one set of number, size, date and duration formats of the Storage tab (the gauge and the
// free-space tray): decimal units, a fixed English date and time inside the sentence (never the
// browser's locale, which mixed languages in one line), counts like the rest of the app.

import { fmt } from './i18n.js';
import { statusMarkHtml } from './status_mark.js';

/** Counts in one fixed format, like the file counts of result_files.js. */
export const number = (n) => Number(n || 0).toLocaleString('en-US');

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
