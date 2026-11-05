// The one set of number, size, date and duration formats of the Storage tab (the gauge and the
// free-space tray): decimal units, a fixed English date and time inside the sentence (never the
// browser's locale, which mixed languages in one line), counts like the rest of the app; and the one
// "Largest parts" list a place and a deleted project's folder both open.

import { fmt } from './i18n.js';
import { statusMarkHtml } from './status_mark.js';
import { escapeHtmlAttr, escapeHtmlText } from './utils.js';

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

// A retirer pass's typed holds (the engine's `GcHold` classes and the server retirers' own reason words) in
// the owner's words; a class this client lacks words for is shown by its name.
const HOLD_WORDS = {
    task_live: 'task running', liveness_unknown: 'task state unknown', task_not_terminal: 'task not finished',
    parent_not_terminal: 'parent task not finished', owner_not_terminal: 'owner task not finished',
    not_proven_over: 'task not proven over', owner_processes_live: 'process still running', owner_paused: 'task paused',
    run_open: 'run still open', seat_open: 'reviewer seat still open', seat_busy: 'reviewer seat busy', custody_held: 'held by a coding run',
    custody_unreadable: 'custody unreadable', capture_unproven: 'work not verified saved',
    capture_failed: 'capture failed', child_result_unadopted: 'result not copied yet', child_refs_pending: 'result not copied yet',
    call_inventory_pending: 'call records not copied yet', call_inventory_changed: 'call records moved',
    unread_mailbox_uncustodied: 'unread messages not carried', verification_receipts_uncustodied: 'receipts not copied yet',
    raw_evidence_uncustodied: 'evidence not copied yet', inputs_uncustodied: 'inputs not copied yet',
    canonical_not_settled: 'task record not settled', post_work_open: 'work after the task open', artifacts_not_terminal: 'files still being saved',
    task_result_unreadable: 'task record unreadable', record_unreadable: 'record unreadable', record_busy: 'record busy',
    store_busy: 'record store busy', custody_busy: 'custody busy', mailbox_busy: 'mailbox busy', owner_keep: 'kept on your request',
    producer_retained: 'marked as evidence', removal_failed: 'removal failed', generation_closed: 'stopped by a restart',
    missing_partition: 'engine partition missing', unbound_active_registration: 'engine registration active',
    recovery_pending: 'engine recovery pending', namespace_unreadable: 'engine namespace unreadable',
    anchor_references_unreadable: 'engine references unreadable', references_changed: 'engine references changed',
    native_home_veto: 'vendor home vetoed', cleanup_suppressed: 'engine cleanup suppressed',
};
const holdWords = (hold) => fmt(HOLD_WORDS[hold.class] || String(hold.class || '').replace(/_/g, ' '));

/** A pass's holds in one phrase: "holding 3 since …: 2 task running, 1 kept on your request". */
export function holdsPhrase(fact, now) {
    const holds = Array.isArray(fact.holds) ? fact.holds.filter((hold) => hold && hold.count > 0) : [];
    const held = holds.length ? holds.reduce((sum, hold) => sum + hold.count, 0) : (fact.held > 0 ? fact.held : 0);
    if (!(held > 0)) return '';
    const head = fact.held_since ? fmt('holding {n} since {when}', { n: number(held), when: whenLabel(fact.held_since, now) })
        : fmt('holding {n}', { n: number(held) });
    return holds.length ? `${head}: ${holds.map((hold) => `${number(hold.count)} ${holdWords(hold)}`).join(', ')}` : head;
}
