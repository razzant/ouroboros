// Dashboard -> Storage: the disk's free space first, with the one ordinary action beside it, then the last
// finished count of everything Ouroboros owns (the data folder with the engine home, working copies, project
// folders, Deliverables, the installed program), its date and the growth since the count before it.
// `POST /api/storage/measure` starts one complete count in a server child; `GET /api/storage` is its passive
// projection, polled only while counting, listing or freeing (including trash deletion, not an entry the
// maintenance child reported stuck) AND this tab is on screen. The bar is drawn only for a finished count (the
// owner's choice «Полоса и группы», 2026-10-05): five group shares that sum to the headline; a running, stopped
// or failed count never replaces the saved one, so the bar and the list stay while a new count runs. Each group
// folds under its head, its places, sizes and removers one press away. Unknown is never zero: a place that could
// not be listed has no size. The first card holds the flow of Free temporary space (storage_free.js): a pure
// preview, the ordinary pool read whole by its real ids, one Remove, progress and the result in place. Finished
// copies not verified saved are a separate, deliberate form: their own list read whole, each id chosen by the owner
// (nothing chosen to begin with), the shared confirm dialog as the acknowledgement of the whole copy's loss.

import { apiClient, isStoreMigrating } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { fmt } from './i18n.js';
import { showToast } from './toast.js';
import { escapeHtmlAttr, escapeHtmlText } from './utils.js';
import { DOTS, durationLabel, formatBytes, isSize, number, plural, signedBytes as signed, whenLabel } from './storage_format.js';
import { EMPTY_LIST, MANUAL_ROWS, OUTDATES_PLAN, acceptPage, attemptSentences, confirmCopy, doorOffered, excludedCount, lostNotice, manualConfirmCopy, manualFacts, manualKnown, manualOf, notStarted, resultSentences, savedGroups, trayHtml, trayState } from './storage_free.js';

export { durationLabel, formatBytes, whenLabel };

const POLL_MS = 1000;
const POOL_PAGE = 500;  // the contract's largest page: the pool is read whole before anything can be sent
const LIST_PAGE = 100;
const MANUAL_PAGE = 50;  // a copy's entry lists its fifty largest unsaved paths: smaller pages than the other lists
const LIST_RESTARTS = 3;  // a list whose version keeps moving stops and says so; the owner may try again
const text = escapeHtmlText;
const attr = escapeHtmlAttr;

/** The five groups of the bar and the list, in display order (server `place.group`). */
export const STORAGE_GROUPS = [
    { id: 'records', label: 'Model call records' },
    { id: 'engine', label: 'Agent engine' },
    { id: 'work', label: 'Work and history' },
    { id: 'temporary', label: 'Task work folders and copies' },
    { id: 'other', label: 'App state and the rest' },
];

// Server place id -> the words this screen says about it. `parts` offers the "Largest parts" disclosure,
// `split` the engine's three parts (`projects` and `profiles` read by name: the server lists them whatever
// their rank). A place this client does not know is shown by its id.
const PLACE_WORDS = {
    model_calls: { label: 'Model call records', why: 'The exact prompts and answers behind every decision. Kept; this screen cannot delete them.', parts: true },
    engine: { label: 'Agent engine', why: 'Kept by the engine. Not a cache.', split: true },
    task_results: { label: 'Task results and files', why: 'Kept with each task.' },
    archives: { label: 'Archives', why: 'Kept. Rotated logs and backups.' },
    memory_skills: { label: 'Memory, notes and skills', why: 'Kept until you delete that project or skill.' },
    project_folders: { label: 'Project folders', why: 'Your project files. Not deleted here.' },
    deliverables: { label: 'Deliverables', why: 'Files Ouroboros made for you. Not deleted here.' },
    work_folders: { label: 'Task work folders', why: 'Removed by the app’s own cleanup {days} days after a task finishes.' },
    working_copies: { label: 'Working copies', why: 'Copies of code that helpers work in. Removed {days} days after they were made, once their task has finished, their work is saved and you have not asked to keep them.' },
    app_state: { label: 'App state', why: 'The task database, settings, caches and working state.', parts: true },
    data_other: { label: 'Everything else in the data folder', why: 'Logs, attachments and anything not listed above.', parts: true },
    program: { label: 'Ouroboros program', why: 'The installed app and its history. Updated by the app.' },
};

// ---------------------------------------------------------------------------
// Pure presentation
// ---------------------------------------------------------------------------

const words = (place) => PLACE_WORDS[place?.id] || { label: String(place?.id || ''), why: '' };

// Who removes or shrinks what a place holds, and how its last pass ended (server `removers`, read fresh
// on every answer): one plain line; unknown, unavailable and unsupported are said as such, never as zero.
const JOB_WORDS = {
    runtime_retirement: 'Retirement',
    custody_trash: 'Deleting moved folders',
    evidence_packing: 'Evidence packing',
    record_compression: 'Compression',
};
const JOB_STATES = {
    done: '{job}: done',
    pending: '{job}: in progress',
    running: '{job}: running',
    failed: '{job}: failed',
    never: '{job}: not run yet',
    unsupported: '{job}: not supported on this system',
    unavailable: '{job}: unknown, its record could not be read',
};

function engineLine(fact, now) {
    const head = fmt('Removed by engine retention');
    if (fact.state === 'none') return `${head} · ${fmt('no pass yet')}`;
    if (fact.state === 'unsupported') return `${head} · ${fmt('last pass unknown: this engine does not report it')}`;
    if (!['completed', 'held', 'truncated', 'failed'].includes(fact.state)) return `${head} · ${fmt('last pass unknown: the engine is not answering')}`;
    const when = whenLabel(fact.at, now);
    // The state says how the pass ended; a hold is said even when it held no folder (a namespace or an anchor set).
    const parts = [head, fact.state === 'failed' ? fmt('last pass {when} failed', { when })
        : fact.state === 'truncated' ? fmt('last pass {when} stopped early', { when })
            : fact.state === 'held' && !(fact.held > 0) ? fmt('last pass {when} held', { when }) : fmt('last pass {when}', { when })];
    if (isSize(fact.freed_bytes)) parts.push(fmt('{size} freed', { size: formatBytes(fact.freed_bytes) }));
    if (fact.held > 0) {
        parts.push(fact.held_since ? fmt('holding {n} folders since {when}', { n: number(fact.held), when: whenLabel(fact.held_since, now) })
            : fmt('holding {n} folders', { n: number(fact.held) }));
    }
    if (fact.errors > 0) parts.push(plural(fact.errors, '{n} error', '{count} errors'));
    return parts.join(' · ');
}

function jobPhrase(fact, now) {
    const job = fmt(JOB_WORDS[fact.id] || String(fact.id));
    if (fact.last_error) {
        return fmt('{job}: stopped by an error ({error}, {when})', { job, error: fact.last_error.class, when: whenLabel(fact.last_error.at, now) });
    }
    if (fact.id === 'record_compression' && fact.state !== 'unsupported' && fact.compressed > 0) {
        // This place's own plane (task evidence, or reviewer seat homes), never one total under two places.
        const files = plural(fact.compressed, '{n} file compressed', '{count} files compressed');
        const line = isSize(fact.saved_bytes) ? fmt('{job}: {files}, {size} saved', { job, files, size: formatBytes(fact.saved_bytes) })
            : fmt('{job}: {files}', { job, files });
        return fact.state === 'pending' ? fmt('{line}, in progress', { line }) : line;
    }
    return fmt(JOB_STATES[fact.state] || '{job}: {state}', { job, state: String(fact.state || '') });
}

/** One place's removers as one line ('' when nothing removes it). */
export function removerLine(removers, now = new Date()) {
    const facts = Array.isArray(removers) ? removers.filter((fact) => fact && fact.id) : [];
    return facts.map((fact) => (fact.id === 'engine' ? engineLine(fact, now) : jobPhrase(fact, now))).join(' · ');
}

function engineSplit(place) {
    const part = (name) => place.parts.find((row) => row.name === name)?.bytes || 0;
    const history = part('projects');
    const accounts = part('profiles');
    return [
        fmt('Run history {size}', { size: formatBytes(history) }),
        fmt('Account data {size}', { size: formatBytes(accounts) }),
        fmt('Engine files {size}', { size: formatBytes(Math.max(0, (place.bytes || 0) - history - accounts)) }),
    ].join(' · ');
}

/** The count's places as rows in server order (absent places omitted), each with its words, growth, its
 *  removers' line (`removers`: the server's per-place facts) and the parts its disclosure lists. `days` is
 *  the server's retention for the two temporary rows. */
export function placeRows(count, days = 7, removers = {}, now = new Date()) {
    return (Array.isArray(count?.places) ? count.places : []).filter((place) => place && place.present !== false).map((place) => {
        const said = words(place);
        const parts = Array.isArray(place.parts) ? place.parts : [];
        return {
            id: String(place.id),
            group: STORAGE_GROUPS.some((group) => group.id === place.group) ? place.group : 'other',
            label: fmt(said.label),
            why: said.why ? fmt(said.why, { days }) : '',
            bytes: isSize(place.bytes) ? place.bytes : null,
            unreadable: isSize(place.unreadable) ? place.unreadable : 0,
            growth: isSize(place.growth_bytes) ? place.growth_bytes : null,
            caveat: place.growth_flag === 'unreadable',
            undecided: typeof place.undecided === 'string' ? place.undecided : '',
            otherDisk: place.same_volume === false,
            split: said.split && parts.length ? engineSplit({ ...place, parts }) : '',
            remover: removerLine(removers?.[place.id], now),
            parts: said.parts ? parts.filter((row) => isSize(row?.bytes)) : [],
            rest: said.parts && place.rest && place.rest.count > 0 ? place.rest : null,
        };
    });
}

/** The groups with members, in display order: bytes summed over the known members (unknown when none is
 *  known: never zero), growth summed only when every sized member has one. */
export function groupRows(rows) {
    return STORAGE_GROUPS.map((group) => {
        const members = rows.filter((row) => row.group === group.id);
        const sized = members.filter((row) => row.bytes !== null);
        return {
            id: group.id,
            label: fmt(group.label),
            members,
            bytes: sized.length ? sized.reduce((sum, row) => sum + row.bytes, 0) : null,
            growth: sized.length && sized.every((row) => row.growth !== null) ? sized.reduce((sum, row) => sum + row.growth, 0) : null,
            caveat: members.some((row) => row.caveat),
        };
    }).filter((group) => group.members.length);
}

/** One segment per group with bytes, its share of the finished headline (the shares sum to 100). */
export function barSegments(count, days = 7) {
    const groups = groupRows(placeRows(count, days)).filter((group) => group.bytes > 0);
    const total = groups.reduce((sum, group) => sum + group.bytes, 0);
    if (!(total > 0)) return [];
    return groups.map((group) => ({ group: group.id, bytes: group.bytes, share: Number(((group.bytes / total) * 100).toFixed(2)) }));
}

/** 'loading' before the first answer; 'counting' while a count runs; 'counted' with a saved count;
 *  'failed' after a failure with nothing saved; 'never' before any count. */
export function storageState(response) {
    if (!response) return 'loading';
    if (response.counting) return 'counting';
    if (response.count) return 'counted';
    return response.error && !response.error.stopped ? 'failed' : 'never';
}

function factsLine(count, now) {
    const parts = [fmt('Last count: {when}', { when: whenLabel(count.finished_at, now) })];
    const took = durationLabel(count.duration_seconds);
    if (took) parts.push(fmt('took {duration}', { duration: took }));
    if (isSize(count.growth_bytes) && count.growth_since) {
        const when = whenLabel(count.growth_since, now);
        const growth = count.growth_bytes === 0 ? fmt('no change since {when}', { when })
            : fmt(count.growth_bytes > 0 ? '+{size} since {when}' : '−{size} since {when}', { size: formatBytes(Math.abs(count.growth_bytes)), when });
        parts.push(count.growth_flag === 'unreadable' ? `${growth} ${fmt('(some items could not be read)')}` : growth);
    }
    return parts.join(' · ');
}

function progressHtml(counting) {
    const sentence = fmt('Measuring {place} · {current} · {size} so far · {done} of {total} places. You can leave this tab.', {
        place: words({ id: counting.place }).label ? fmt(words({ id: counting.place }).label) : '',
        current: counting.current || '',
        size: formatBytes(counting.bytes || 0),
        done: Math.min(Number(counting.places_done || 0) + 1, Number(counting.places_total || 0)),
        total: number(counting.places_total),
    });
    return `<p class="storage-progress" data-i18n-fmt="storage.progress">${DOTS}<span>${text(sentence)}</span></p>`;
}

/** The places already finished in a first count, under the progress line; nothing is summed into a headline. */
function firstCountHtml(counting) {
    const done = (Array.isArray(counting.done) ? counting.done : []).filter((row) => isSize(row?.bytes) && PLACE_WORDS[row.name]);
    if (!done.length) return '';
    return `<ul class="storage-first-count">${done.map((row) => `<li><span>${text(fmt(PLACE_WORDS[row.name].label))}</span>
        <span data-i18n-fmt="storage.size">${text(formatBytes(row.bytes))}</span></li>`).join('')}</ul>`;
}

function statusSentence(error, count, now) {
    const when = count ? whenLabel(count.finished_at, now) : '';
    const reason = String(error.message || '').replace(/\.$/, '');  // the server's reason is a sentence
    if (error.stopped) return count ? fmt('Stopped before it finished. Showing the count from {when}.', { when }) : fmt('Stopped before it finished.');
    return count ? fmt('This count stopped: {reason}. Showing the count from {when}.', { reason, when })
        : fmt('This count stopped: {reason}.', { reason });
}

function statusHtml(error, count, now) {
    return `<p class="ui-status storage-status" data-tone="warn" data-i18n-fmt="storage.error">${text(statusSentence(error, count, now))}</p>`;
}

/** One stable sentence per phase for the live region: a screen reader hears a count start, end, stop or
 *  fail, and the tray's phases, never the per-second figures of the progress line. */
export function announceSentence(response, { readFailed = false, now = new Date(), tray = null } = {}) {
    const parts = [];
    if (readFailed) parts.push(fmt('Storage status could not be read.'));
    if (response) {
        const { count, counting, error } = response;
        if (counting) parts.push(fmt('Measuring. You can leave this tab.'));
        else if (error) parts.push(statusSentence(error, count, now));
        else if (count) parts.push(fmt('Ouroboros uses {size}.', { size: formatBytes(count.bytes) }));
        else parts.push(fmt('Not counted yet.'));
        const free = freeOf(response);
        const state = trayState(free, tray?.dismissed);
        parts.push(...attemptSentences(tray));  // what happened to the owner's attempt, before the phase it stands beside
        if (state === 'listing') parts.push(fmt('Checking what can go. Nothing is removed.'));
        else if (state === 'ready' && free.pool?.count > 0) parts.push(fmt('{items} ready to remove.', { items: plural(free.pool.count, '{n} item', '{count} items') }));
        else if (state === 'running') parts.push(fmt('Removing. You can leave this tab.'));
        else if (state === 'done') {
            const { problems, lead } = resultSentences(free);
            parts.push(...problems, lead);
        } else if (state === 'failed') parts.push(fmt('Temporary space could not be listed. Nothing was removed.'));
    }
    return parts.filter(Boolean).join(' ');
}

/** The first card: the disk's free space as last read, what else that reading says, and a read that failed. */
function summaryHtml(response, { readFailed, now }) {
    const kicker = '<div class="storage-kicker">Free on this disk</div>';
    if (!response) {
        return `${kicker}<div class="storage-total is-unknown">${readFailed ? 'Could not be read' : 'Loading…'}</div>`;
    }
    const { count, volume } = response;
    const known = isSize(volume?.free_bytes);
    const facts = [
        isSize(volume?.total_bytes) ? fmt('of {total}', { total: formatBytes(volume.total_bytes) }) : '',
        known && whenLabel(volume?.observed_at, now) ? fmt('read {when}', { when: whenLabel(volume.observed_at, now) }) : '',
        count ? fmt('Ouroboros uses {size}', { size: formatBytes(count.bytes) }) : fmt('Ouroboros not counted yet'),
    ].filter(Boolean).join(' · ');
    const lines = [`<div class="storage-meta" data-i18n-fmt="storage.free">${text(facts)}</div>`];
    if (readFailed) lines.push('<p class="ui-status storage-status" data-tone="warn">Storage status could not be read. Showing the last answer.</p>');
    return `${kicker}<div class="storage-total${known ? '' : ' is-unknown'}" data-i18n-fmt="storage.total">${text(known ? formatBytes(volume.free_bytes) : fmt('Could not be read'))}</div>${lines.join('')}`;
}

/** The second card's head: how the saved count stands, or why there is none yet. */
function countStatusHtml(response, now) {
    if (!response) return '';
    const { count, counting, error } = response;
    const lines = [];
    if (count) {
        const unreadable = count.unreadable > 0 ? ` · ${plural(count.unreadable, '{n} item could not be read', '{count} items could not be read')}` : '';
        lines.push(`<div class="storage-meta" data-i18n-fmt="storage.facts">${text(factsLine(count, now) + unreadable)}</div>`);
    } else if (error && !error.stopped && !counting) {
        lines.push(`<div class="storage-meta" data-i18n-fmt="storage.total">${text(fmt('Could not be counted'))}</div>`);
    } else {
        lines.push('<div class="storage-meta">Not counted yet</div>');
    }
    if (counting) {
        lines.push(progressHtml(counting));
        if (!count) lines.push(firstCountHtml(counting));
    } else if (error) {
        lines.push(statusHtml(error, count, now));
    } else if (!count) {
        lines.push('<div class="storage-meta">Measuring reads file sizes only and changes nothing. It can take a while on a large data folder.</div>');
    }
    return lines.join('');
}

function sizeCells(row) {
    const size = row.bytes === null ? '' : formatBytes(row.bytes);
    const growth = row.growth === null || row.growth === 0 ? '' : signed(row.growth);
    return `<span class="storage-place-size${size ? '' : ' is-unknown'}" data-i18n-fmt="storage.size">${text(size)}</span>
        <span class="storage-place-delta" data-i18n-fmt="storage.growth">${text(growth)}</span>`;
}

/** The reason, then what qualifies the row: unreadable (or a folder not chosen yet, with the server's
 *  sentence naming the setting), on another disk, a growth some items escaped. */
function whyLine(row) {
    const notes = [row.why];
    if (row.bytes === null) notes.push(row.undecided ? fmt('Not counted: {reason}', { reason: row.undecided }) : fmt('Could not be read.'));
    if (row.otherDisk) notes.push(fmt('on another disk'));
    if (row.caveat && row.growth) notes.push(fmt('(some items could not be read)'));
    return `<span class="storage-place-why" data-i18n-fmt="storage.why">${text(notes.filter(Boolean).join(' · '))}</span>`;
}

function partsHtml(row) {
    if (!row.parts.length) return '';
    const items = row.parts.map((part) => `<li><span data-i18n-authored>${text(part.name)}</span><span data-i18n-fmt="storage.size">${text(formatBytes(part.bytes))}</span></li>`);
    if (row.rest) items.push(`<li class="storage-parts-rest" data-i18n-fmt="storage.rest">${text(fmt('and {n} more, {size}', { n: number(row.rest.count), size: formatBytes(row.rest.bytes) }))}</li>`);
    return `<details class="activity-history storage-parts" data-key="parts-${attr(row.id)}">
            <summary data-key="parts-${attr(row.id)}-open">Largest parts</summary>
            <ul class="storage-parts-list">${items.join('')}</ul>
        </details>`;
}

const removerHtml = (row) => (row.remover
    ? `<span class="storage-place-remover" data-i18n-fmt="storage.remover">${text(row.remover)}</span>` : '');

function memberHtml(row) {
    return `<li class="storage-place is-member" data-storage-place="${attr(row.id)}">
            <span class="storage-place-label">${text(row.label)}</span>${sizeCells(row)}
            ${whyLine(row)}
            ${row.split ? `<span class="storage-place-split" data-i18n-fmt="storage.split">${text(row.split)}</span>` : ''}
            ${removerHtml(row)}${partsHtml(row)}
        </li>`;
}

/** What one group holds, in a line, before anyone opens it. */
const GROUP_SAID = {
    records: 'The exact prompts and answers behind every decision. Kept.',
    engine: 'Run history, accounts and engine files. Kept by the engine.',
    work: 'Your projects, results, files, memory and archives. Kept.',
    temporary: 'Copies and work folders of tasks. Cleared by the app’s own cleanup.',
    other: 'The installed app, its state and logs.',
};

/** A group folds under its head and one line; opening it shows the places, sizes and removers in full. */
function groupHtml(group) {
    const single = group.members.length === 1 ? group.members[0] : null;
    // A group of one place carries that place's words under its heading; a larger group lists its places.
    const body = single
        ? `<div class="storage-group-body">${whyLine(single)}
            ${single.split ? `<span class="storage-place-split" data-i18n-fmt="storage.split">${text(single.split)}</span>` : ''}
            ${removerHtml(single)}${partsHtml(single)}</div>`
        : `<ul class="storage-group-places">${group.members.map(memberHtml).join('')}</ul>`;
    return `<li class="storage-group" data-storage-group="${attr(group.id)}"${single ? ` data-storage-place="${attr(single.id)}"` : ''}>
        <details class="storage-group-fold" data-key="group-${attr(group.id)}">
            <summary data-key="group-${attr(group.id)}-open"><span class="storage-group-head"><span class="storage-key" data-group="${attr(group.id)}"></span>
                <span class="storage-group-label">${text(group.label)}</span>${sizeCells(group)}</span>
                <span class="storage-group-said">${text(fmt(GROUP_SAID[group.id] || ''))}</span></summary>
            <div class="storage-group-open">${body}</div>
        </details></li>`;
}

function placesHtml(response, now) {
    const count = response?.count;
    if (!count) return '';
    const days = response.retention_days ?? 7;
    // The bar pictures the last finished count only; its label is fixed chrome, the list carries the names
    // and sizes (the language overlay translates an aria-label by its English text).
    const segments = barSegments(count, days).map((segment) => `<span class="storage-bar-seg" data-group="${attr(segment.group)}" data-share="${segment.share}"></span>`);
    const bar = segments.length ? `<div class="storage-bar" role="img" aria-label="Share of each group; sizes are listed below">${segments.join('')}</div>` : '';
    return `${bar}<ul class="storage-groups">${groupRows(placeRows(count, days, response.removers, now)).map(groupHtml).join('')}</ul>`;
}

export const freeOf = (response) => response?.free || { state: 'idle', groups: [], run: null, error: null, done: 0, total: 0, pool: null };

/** The whole view for one response: markup per region, the measure action's label and busy flag, whether
 *  Stop and the door are offered, and the flow for the owner's state (see `trayHtml`). */
export function renderStorage(response, { pending = false, readFailed = false, now = new Date(), tray = null } = {}) {
    const state = storageState(response);
    const counting = state === 'counting' || (pending && !response?.count);
    const busy = Boolean(pending || response?.counting);
    const label = counting ? 'Measuring…' : response?.count ? 'Measure again' : state === 'failed' ? 'Try again' : 'Measure now';
    return {
        state,
        summary: summaryHtml(response, { readFailed, now }),
        auto: response ? fmt('Leftovers of finished one-off work clear by themselves after {days} days (Settings › Advanced › Cleanup), a project task’s scratch and copies too. Project folders stay.', { days: response.retention_days ?? 7 }) : '',
        status: countStatusHtml(response, now),
        announce: announceSentence(response, { readFailed, now, tray }),
        places: placesHtml(response, now),
        tray: response ? trayHtml(freeOf(response), tray || { checked: new Set(), dismissed: false }, response.retention_days ?? 7, { volume: response.volume, now }) : '',
        door: Boolean(response) && doorOffered(trayState(freeOf(response), tray?.dismissed)),
        measure: { label, busy },
        stop: Boolean(response?.counting),
    };
}

// ---------------------------------------------------------------------------
// The Dashboard tab
// ---------------------------------------------------------------------------

/** Replace a region only when its markup changed, keeping open disclosures and the focused control by key. */
function patch(node, html, painted) {
    if (painted.get(node) === html) return;
    painted.set(node, html);
    const open = Array.from(node.querySelectorAll('details[open][data-key]'), (element) => element.dataset.key);
    const focused = node.contains(document.activeElement) ? document.activeElement?.dataset?.key : '';
    node.innerHTML = html;
    open.forEach((key) => node.querySelector(`details[data-key="${key}"]`)?.setAttribute('open', ''));
    if (focused) node.querySelector(`[data-key="${focused}"]`)?.focus({ preventScroll: true });
    // A bar share is a measured runtime value: one narrowly named custom property.
    node.querySelectorAll('[data-share]').forEach((element) => element.style?.setProperty('--storage-share', element.dataset.share));
}

export function initStorage({ state, mount } = {}) {
    if (!mount) return null;
    const page = document.createElement('div');
    page.id = 'page-storage';
    page.className = 'settings-embedded-content';
    page.innerHTML = `
        <div class="costs-scroll">
            <div class="storage-root">
                <section class="storage-summary ui-card" aria-label="Storage summary">
                    <p class="storage-announce" data-storage-live role="status" aria-live="polite"></p>
                    <div class="storage-summary-head">
                        <div class="storage-summary-text" data-storage-summary></div>
                        <div class="storage-actions storage-summary-actions">
                            <button type="button" class="btn btn-default btn-sm" data-free-preview data-key="data-free-preview" data-storage-door hidden>Free temporary space…</button>
                        </div>
                    </div>
                    <p class="storage-auto" data-storage-auto data-i18n-fmt="storage.auto"></p>
                    <div class="storage-tray" data-storage-tray></div>
                </section>
                <section class="storage-panel ui-card" aria-label="Where it is" data-storage-panel hidden>
                    <div class="storage-tray-head">
                        <h3 class="storage-card-title">Where it is</h3>
                        <div class="storage-actions storage-panel-actions">
                            <button type="button" class="btn btn-ghost btn-sm storage-refresh" data-storage-action="measure">Measure now</button>
                            <button type="button" class="btn btn-default btn-sm" data-storage-action="stop" hidden>Stop</button>
                        </div>
                    </div>
                    <div data-storage-status></div>
                    <div data-storage-places></div>
                </section>
            </div>
        </div>
    `;
    mount.appendChild(page);
    const regions = {
        summary: page.querySelector('[data-storage-summary]'),
        status: page.querySelector('[data-storage-status]'),
        places: page.querySelector('[data-storage-places]'),
        tray: page.querySelector('[data-storage-tray]'),
    };
    // The one live region, never replaced: it carries one sentence per phase (announceSentence), so a
    // screen reader hears phase changes and not the figures the progress line repaints every second.
    const live = page.querySelector('[data-storage-live]');
    const panel = page.querySelector('[data-storage-panel]');
    // The owner's flow state, for one plan: the lists read so far, the items chosen (every one, once all are read),
    // the saved groups checked (none to begin with). A dismissed list or result hides without asking the server.
    // `sending`: a Remove, Free or Discard is out. `postUnknown`: the action ('Remove', 'Free' or 'Discard') whose answer was lost, so this plan
    // is never sent again; `confirming`
    // while no read has answered since. `outdated`: 'refused' or 'read' when the server no longer holds this plan.
    // `notice`: what happened to the owner's attempt under a plan the server has since replaced (attemptSentences).
    // `manualOpen`: the copies-by-hand screen; `manualChosen`: the copy ids the owner chose there, none to begin with.
    const freshFlow = (plan = null) => ({ plan, chosen: new Set(), checked: new Set(), dismissed: false, saved: false, sending: false, outdated: '',
        postUnknown: false, confirming: false, notice: '', poolShown: LIST_PAGE, pool: EMPTY_LIST(), excluded: EMPTY_LIST(), outcomes: EMPTY_LIST(),
        manual: EMPTY_LIST(), manualOpen: false, manualChosen: new Set(), manualShown: MANUAL_ROWS });
    let ui = freshFlow();
    // The measure node is never replaced, and a running count marks it aria-disabled rather than disabled,
    // so keyboard focus stays on it through a count and every repaint.
    const measureButton = page.querySelector('[data-storage-action="measure"]');
    const stopButton = page.querySelector('[data-storage-action="stop"]');
    const door = page.querySelector('[data-storage-door]');
    const auto = page.querySelector('[data-storage-auto]');
    const scroller = page.querySelector('.costs-scroll');
    const painted = new Map();
    let response = null;
    let readFailed = false;
    let pending = false;  // any POST out: a second one is dropped until it answers
    let measuring = false;  // the measure POST out: the one that reads "Measuring…" before the server does
    let disposed = false;
    let generation = 0;
    let controller = null;
    let timer = 0;

    const onScreen = () => !document.hidden && state?.activePage === 'dashboard' && state?.dashboardActiveSubtab === 'storage';

    /** A new plan is a new flow: whatever was read or chosen under the last one means nothing now. What happened to the
     *  owner's attempt on it is still said beside the new facts, never merged into them. */
    function syncPlan() {
        const { plan = null, run = null } = freeOf(response);
        if (plan !== ui.plan) {
            abortLists();
            const { dismissed } = ui;
            const notice = ui.postUnknown ? lostNotice(ui.postUnknown) : ui.outdated || ui.notice;
            ui = freshFlow(plan);
            ui.dismissed = plan === null ? dismissed : false;
            ui.notice = notice;
        }
        // A run of this plan is the answer a lost request could not give. One that began after this list was
        // dismissed has a result to show: only Done dismisses that.
        if (run && !ui.runSeen) Object.assign(ui, { runSeen: true, dismissed: false, postUnknown: false, confirming: false });
    }

    function paint() {
        syncPlan();
        const view = renderStorage(response, { pending: measuring, readFailed, tray: ui });
        const top = scroller.scrollTop;
        const flowHeld = regions.tray.contains(document.activeElement) || document.activeElement === door;
        patch(regions.summary, view.summary, painted);
        if (live.textContent !== view.announce) live.textContent = view.announce;
        if (auto.textContent !== view.auto) auto.textContent = view.auto;
        patch(regions.status, view.status, painted);
        patch(regions.places, view.places, painted);
        patch(regions.tray, view.tray, painted);
        panel.hidden = !response;
        door.hidden = !view.door;
        door.parentElement.hidden = !view.door;  // the empty row takes no room on a phone
        door.setAttribute('aria-disabled', pending ? 'true' : 'false');
        if (measureButton.textContent !== view.measure.label) measureButton.textContent = view.measure.label;
        measureButton.setAttribute('aria-disabled', view.measure.busy ? 'true' : 'false');
        if (view.measure.busy) measureButton.setAttribute('aria-busy', 'true');
        else measureButton.removeAttribute('aria-busy');
        stopButton.hidden = !view.stop;
        // The control that held focus can leave with its phase: focus follows the flow instead of falling to the body.
        if (flowHeld && !regions.tray.contains(document.activeElement) && (document.activeElement !== door || door.hidden)) refocusTray();
        if (scroller.scrollTop !== top) scroller.scrollTop = top;
    }

    function abortLists() {
        for (const list of [ui.pool, ui.excluded, ui.manual, ui.outcomes]) {
            list.abort?.abort();
            list.abort = null;
            list.loading = false;
            list.run = (list.run || 0) + 1;
        }
    }

    function stopPolling() {
        clearTimeout(timer);
        timer = 0;
    }

    // A passive GET is only useful while this panel is the visible tab. Abort it as soon as the owner
    // leaves the panel so a hidden read cannot repaint a later view, and advance the generation first.
    function cancelRead() {
        generation += 1;
        controller?.abort();
        controller = null;
        stopPolling();
        abortLists();
    }

    const working = () => Boolean(response?.counting) || ['listing', 'running'].includes(freeOf(response).state)
        || freeOf(response).run?.deleting > 0 || ui.confirming;

    function schedule() {
        stopPolling();
        if (!disposed && working() && onScreen()) timer = setTimeout(refresh, POLL_MS);
    }

    /** A read that names the count it holds gets no `count`: the held one stays. */
    function apply(next) {
        response = { ...next, count: 'count' in next ? next.count : (response?.count ?? null) };
        readFailed = false;
        // A read answered; it cannot say a lost request will never land, so the plan stays unsendable (syncPlan).
        ui.confirming = false;
        paint();
        schedule();
        syncLists();
    }

    async function refresh() {
        // While a POST is out its own answer is the next truth: a read now could answer from before it and be dropped
        // or overtake it. act() reads once it is answered when the tab moved meanwhile.
        if (disposed || pending) return;
        stopPolling();
        controller?.abort();
        const mine = ++generation;
        const abort = new AbortController();
        controller = abort;
        try {
            const next = await apiClient.storage({ signal: abort.signal, count: response?.count?.finished_at || '' });
            if (!disposed && mine === generation) apply(next);
        } catch {
            if (disposed || mine !== generation || abort.signal.aborted) return;
            readFailed = true;
            paint();
            schedule();
            syncLists();  // the held plan's lists read on; a plan the server lost answers so itself
        } finally {
            if (controller === abort) controller = null;
        }
    }

    /** A flow action replaces the control it came from; keyboard focus moves to the flow's next control
     *  (Stop while it works, Remove on the review, the door at rest) instead of falling to the body. */
    function refocusTray() {
        if (disposed || regions.tray.contains(document.activeElement)) return;
        const next = ['stop', 'go', 'done', 'preview', 'saved-back', 'manual-back', 'cancel'].map((name) => regions.tray.querySelector(`[data-free-${name}]`)).find(Boolean);
        if (next) next.focus({ preventScroll: true });
        else if (!door.hidden) door.focus({ preventScroll: true });
    }

    /** The owner's own press replaced the flow: bring its start into view. A poll never moves the page. */
    function revealFlow() {
        const edge = regions.tray.getBoundingClientRect?.();
        const box = scroller.getBoundingClientRect?.();
        if (edge && box && (edge.top < box.top || edge.bottom > box.bottom)) scroller.scrollTop += edge.top - box.top - 12;
    }

    // ---------------------------------------------------------------------------
    // The lists: read only while this tab is on screen, one reader per list, never a timer of their own.
    // ---------------------------------------------------------------------------

    const LISTS = { pool: { size: POOL_PAGE }, excluded: { size: LIST_PAGE }, manual: { size: MANUAL_PAGE }, outcomes: { size: LIST_PAGE } };

    /** Read `view` until it holds `want` items or all of them. A version that moved restarts from the first page,
     *  so nothing is silently missed or shown twice; a refusal or failure stops and says so, with no retry loop. */
    async function readList(view) {
        const list = ui[view];
        const plan = ui.plan;
        if (disposed || list.loading || !plan || !onScreen()) return;
        const mine = (list.run || 0) + 1;
        const abort = new AbortController();
        Object.assign(list, { run: mine, abort, loading: true, error: '' });
        paint();
        const restart = () => {
            Object.assign(list, { items: [], ids: new Set(), total: null, version: '', next: 0, complete: false, restarts: list.restarts + 1 });
            if (view === 'manual') ui.manualChosen = new Set();  // a list read again is chosen from again
        };
        try {
            while (!list.complete && list.items.length < list.want) {
                let page = null;
                try {
                    page = await apiClient.storageFreeItems({ plan, view, cursor: list.next, limit: LISTS[view].size, version: list.version, signal: abort.signal });
                } catch (error) {
                    if (error?.body?.reason_code !== 'view_changed') throw error;
                    if (list.run !== mine) return;
                    if (list.restarts >= LIST_RESTARTS) { list.error = 'changing'; return; }
                    restart();
                    paint();
                    continue;
                }
                if (disposed || list.run !== mine) return;
                const expected = view === 'pool' ? freeOf(response).pool?.count ?? null : view === 'manual' ? manualOf(freeOf(response))?.count ?? null : null;
                const problem = acceptPage(list, page, { plan, view, expected });
                if (problem === 'moved' && list.restarts < LIST_RESTARTS) restart();
                else if (problem) { list.error = problem; return; }
                if (view === 'pool' && list.complete) ui.chosen = new Set(list.items.map((item) => item.id));
                paint();
            }
            list.restarts = 0;
        } catch (error) {
            if (disposed || list.run !== mine || abort.signal.aborted) return;
            if (error?.body?.reason_code === 'preview_outdated') {
                ui.outdated = 'read';
                if (onScreen()) void refresh();
            } else list.error = error?.body?.reason_code || 'read';
        } finally {
            if (list.run === mine) {
                list.loading = false;
                list.abort = null;
                if (!disposed) {
                    paint();
                    syncLists();  // the run may have moved while this read was out
                }
            }
        }
    }

    /** A fingerprint of what the run reports: the outcomes are read again only when it moved. */
    const runMark = (run) => JSON.stringify([run?.removed, run?.deleting, run?.stuck, run?.failed, run?.not_started, run?.kept]);

    /** What the current phase needs read: the whole pool at a ready plan, an open list when it has more to show. */
    function syncLists() {
        if (disposed || !onScreen()) return;
        const free = freeOf(response);
        const state = trayState(free, ui.dismissed);
        if (state === 'ready' && free.pool?.count > 0 && !ui.outdated) {
            ui.pool.want = Infinity;
            if (!ui.pool.complete && !ui.pool.error) void readList('pool');
        }
        // The copies by hand are read whole once their screen is open, never earlier and never on a plan the server refused.
        if (state === 'ready' && ui.manualOpen && manualOf(free) && !ui.outdated) {
            ui.manual.want = Infinity;
            if (!ui.manual.complete && !ui.manual.error) void readList('manual');
        }
        // A plan the server no longer holds is not read for: its answer is known, and a failing status read must not loop.
        if (state === 'ready' && !ui.outdated && ui.excluded.want > ui.excluded.items.length && !ui.excluded.complete && !ui.excluded.error) void readList('excluded');
        if (state === 'done' && ui.outcomes.want > 0 && free.run?.outcomes_ready) {
            const mark = runMark(free.run);
            if (mark !== ui.outcomes.mark && !ui.outcomes.loading) {
                // The trash settled: the same number of rows is read again from the first page, under its new version.
                const { want } = ui.outcomes;
                ui.outcomes = { ...EMPTY_LIST(), want, mark };
            }
            if (!ui.outcomes.complete && ui.outcomes.items.length < ui.outcomes.want && !ui.outcomes.error) void readList('outcomes');
        }
    }

    /** A disclosure opened: its list starts being read, one page's worth. */
    function wantList(view) {
        if (ui[view].want <= 0) ui[view].want = LIST_PAGE;
        syncLists();
    }

    function showMore(view) {
        ui[view].want = ui[view].items.length + LIST_PAGE;
        syncLists();
    }

    function retryList(view) {
        // A failed request resumes where it stopped; an answer that could not be trusted starts the list over.
        const untrusted = ['incomplete', 'moved', 'changing'].includes(ui[view].error);
        Object.assign(ui[view], { error: '', restarts: 0 }, untrusted ? { items: [], ids: new Set(), total: null, version: '', next: 0, complete: false } : {});
        if (view === 'manual' && untrusted) ui.manualChosen = new Set();
        paint();
        syncLists();
    }

    /** One POST whose answer is the same envelope as the GET; `measure` marks the measure POST, `removal` a Remove, Free
     *  or Discard ('Remove' | 'Free' | 'Discard'), whose flow is held (`sending`) until it is answered. */
    async function act(request, failure, { measure = false, removal = false } = {}) {
        if (disposed || pending) return;
        pending = true;
        measuring = measure;
        const flow = ui;
        if (removal) flow.sending = true;
        stopPolling();
        controller?.abort();
        const mine = ++generation;
        paint();
        let next = null;
        let failed = null;
        try {
            next = await request();
        } catch (error) {
            failed = error || new Error(fmt(failure));
        }
        pending = false;
        measuring = false;
        flow.sending = false;
        if (disposed) return;
        if (failed) {
            // No HTTP status (also an OK answer whose body could not be read), or a 5xx: the request may have reached the server, and a 5xx can follow a started run (the
            // answer failed after the start). A 4xx and the migration gate answered before any handler ran, so they are
            // definite. An unknown one is never sent again; a read says what happened, and the toast claims neither way
            // (`failure` says nothing was removed; the browser's own text says nothing).
            const unknown = !failed.status || failed.status >= 500 && !isStoreMigrating(failed);
            const lost = Boolean(removal && unknown);
            if (lost) Object.assign(flow, { postUnknown: removal, confirming: true });
            showToast(lost ? fmt(`The answer to ${removal === 'Free' || removal === 'Discard' ? removal : 'Remove'} was lost; nothing is sent again.`)
                : failed.message || fmt(failure), 'warn');
            paint();
            if ((mine === generation || removal) && onScreen()) void refresh();
            else schedule();
            return;
        }
        // A hide moved the generation while the POST was out: a hidden tab reads nothing until it is shown again
        // (onSubtab/onPage), and a shown one reads now rather than trust an answer it stopped waiting for.
        if (mine === generation) apply(next);
        else {
            paint();
            if (onScreen()) void refresh();
        }
    }

    function measure() {
        if (response?.counting || measureButton.getAttribute('aria-disabled') === 'true') return Promise.resolve();
        return act(() => apiClient.measureStorage(), 'Storage could not be measured.', { measure: true });
    }

    const stopCount = () => act(() => apiClient.cancelStorageMeasure(), 'The count could not be stopped.');
    const stopFree = () => act(() => apiClient.cancelStorageFree(), 'Freeing could not be stopped.').then(refocusTray);

    /** The answer's new plan opens the flow; until then the dismissed one stays dismissed. */
    function preview() {
        // The owner's new start: a refusal's sentence has been read. A lost answer stays unknown and is still said.
        Object.assign(ui, { notice: '', outdated: '' });
        return act(() => apiClient.previewStorageFree(), 'Temporary space could not be listed. Nothing was removed.').then(refocusTray);
    }

    /** One POST for this plan; a refusal says why and a stale list stops offering Remove. Nothing is ever resent. */
    async function postRemoval(send, failure, action) {
        ui.notice = '';  // the owner's next send: an earlier attempt's sentence has been read
        await act(async () => {
            const next = await send();
            if (next?.started === false) {
                showToast(notStarted(next.reason), 'warn');
                if (OUTDATES_PLAN.has(next.reason)) ui.outdated = 'refused';
            }
            return next;
        }, failure, { removal: action });
        refocusTray();
        revealFlow();
    }

    /** The review was the one confirmation: its Remove sends the real ids of exactly the items chosen. */
    async function removeChosen() {
        const free = freeOf(response);
        const items = ui.pool.items.filter((item) => ui.chosen.has(item.id)).map((item) => item.id);
        if (trayState(free, ui.dismissed) !== 'ready' || pending || ui.outdated || ui.postUnknown) return;
        if (free.plan !== ui.plan || !ui.pool.complete || !items.length) return;
        await postRemoval(() => apiClient.freeStorageItems(free.plan, items), 'Nothing was removed.', 'Remove');
    }

    /** The saved groups: the shared confirm dialog, then one POST naming the plan and the checked groups. */
    async function freeSavedGroups() {
        const free = freeOf(response);
        const groups = savedGroups(free).filter((group) => ui.checked.has(group.id)).map((group) => group.id);
        const sendable = () => trayState(freeOf(response), ui.dismissed) === 'ready' && !pending && !ui.outdated && !ui.postUnknown;
        if (!sendable() || !groups.length) return;
        const confirmed = await openConfirmDialog(confirmCopy(free, ui.checked));
        // The dialog was open while the server could answer: the plan must still be this one and still sendable.
        if (!confirmed || disposed || freeOf(response).plan !== free.plan || !sendable()) return;
        await postRemoval(() => apiClient.freeStorage(free.plan, groups), 'Nothing was removed.', 'Free');
    }

    /** The copies chosen by hand: the shared confirm dialog states the whole copy's loss, and only its confirmation sends
     *  one POST of the chosen ids with the acknowledgement. Nothing is sent from a list not read whole, for an id not
     *  chosen, or from a refused plan; the acknowledgement alone never stands in for a choice. */
    async function discardChosen() {
        const free = freeOf(response);
        const sendable = () => trayState(freeOf(response), ui.dismissed) === 'ready' && !pending && !ui.outdated && !ui.postUnknown
            && ui.manualOpen && ui.manual.complete && !ui.manual.error;
        const facts = manualFacts(ui.manual, ui.manualChosen);
        const ids = facts.rows.map((item) => item.id);
        if (!sendable() || free.plan !== ui.plan || !ids.length) return;
        const confirmed = await openConfirmDialog(manualConfirmCopy(ui.manual, ui.manualChosen));
        // The dialog was open while the server could answer: the plan must still be this one and still sendable.
        if (!confirmed || disposed || freeOf(response).plan !== free.plan || !sendable()) return;
        await postRemoval(() => apiClient.freeStorageManual(free.plan, ids, true), 'Nothing was removed.', 'Discard');
    }

    // A Remove, Free or Discard that is out or whose answer was lost holds the flow: nothing hides or re-chooses it.
    const held = () => ui.sending || ui.postUnknown;

    const dismiss = () => {
        if (held()) return;
        abortLists();
        ui.notice = '';
        ui.dismissed = true;
        ui.saved = false;
        ui.manualOpen = false;
        paint();
        door.focus({ preventScroll: true });
    };

    const onClick = (event) => {
        if (event.currentTarget === measureButton) void measure();
        else if (event.currentTarget === stopButton) void stopCount();
    };
    const onTrayClick = (event) => {
        const target = event.target?.closest?.('button');
        if (!target || target.disabled || target.getAttribute('aria-disabled') === 'true') return;
        const has = (name) => target.hasAttribute(name);
        if (has('data-free-preview')) void preview();
        else if (has('data-free-cancel') || has('data-free-done')) dismiss();
        else if (has('data-free-go')) void removeChosen();
        else if (has('data-free-stop')) void stopFree();
        else if (has('data-free-saved-go')) void freeSavedGroups();
        else if (has('data-free-manual-go')) void discardChosen();
        else if (has('data-free-retry-pool')) retryList('pool');
        else if (has('data-free-retry-excluded')) retryList('excluded');
        else if (has('data-free-retry-manual')) retryList('manual');
        else if (has('data-free-retry-outcomes')) retryList('outcomes');
        else if (has('data-free-more-excluded')) showMore('excluded');
        else if (has('data-free-more-outcomes')) showMore('outcomes');
        else {
            if (held() && !has('data-free-more-picks') && !has('data-free-more-manual')) return;
            if (has('data-free-saved')) ui.saved = true;
            else if (has('data-free-saved-back')) ui.saved = false;
            else if (has('data-free-manual-open')) ui.manualOpen = true;
            else if (has('data-free-manual-back')) ui.manualOpen = false;
            else if (has('data-free-manual-none')) ui.manualChosen = new Set();
            else if (has('data-free-more-manual')) ui.manualShown += MANUAL_ROWS;
            else if (has('data-free-all')) ui.chosen = new Set(ui.pool.items.map((item) => item.id));
            else if (has('data-free-none')) ui.chosen = new Set();
            else if (has('data-free-more-picks')) ui.poolShown += LIST_PAGE;
            else return;
            paint();
            if (has('data-free-manual-open')) syncLists();  // its list starts being read, whole
            if (has('data-free-saved') || has('data-free-saved-back') || has('data-free-manual-open') || has('data-free-manual-back')) {
                refocusTray();
                revealFlow();
            }
        }
    };
    const onTrayChange = (event) => {
        const box = event.target?.closest?.('[data-free-group], [data-free-pick], [data-free-manual-pick]');
        if (!box) return;
        if (held()) {
            paint();  // the box shows the choice that was sent again
            return;
        }
        if (box.hasAttribute('data-free-group')) {
            if (box.checked) ui.checked.add(box.dataset.freeGroup);
            else ui.checked.delete(box.dataset.freeGroup);
        } else if (box.hasAttribute('data-free-manual-pick')) {
            // One copy, one choice: only a row the server offered by hand and described can be chosen.
            const copy = ui.manual.complete ? ui.manual.items[Number(box.dataset.freeManualPick)] : null;
            if (!copy || !manualKnown(copy)) {
                paint();
                return;
            }
            if (box.checked) ui.manualChosen.add(copy.id);
            else ui.manualChosen.delete(copy.id);
        } else {
            const item = ui.pool.items[Number(box.dataset.freePick)];
            if (!item) return;
            if (box.checked) ui.chosen.add(item.id);
            else ui.chosen.delete(item.id);
        }
        paint();
    };
    // `toggle` does not bubble: a capture listener sees each disclosure of the flow as it opens.
    const onTrayToggle = (event) => {
        if (!event.target?.hasAttribute?.('open')) return;
        const which = event.target.dataset?.freeDisclosure;
        if (which === 'left' && excludedCount(freeOf(response).pool) > 0) wantList('excluded');
        else if (which === 'each') wantList('outcomes');
    };
    const onSubtab = (event) => {
        if (event?.detail?.tab === 'storage' && onScreen()) void refresh();
        else cancelRead();
    };
    const onPage = () => {
        if (onScreen()) void refresh();
        else cancelRead();
    };
    measureButton.addEventListener('click', onClick);
    stopButton.addEventListener('click', onClick);
    door.addEventListener('click', onTrayClick);
    regions.tray.addEventListener('click', onTrayClick);
    regions.tray.addEventListener('change', onTrayChange);
    regions.tray.addEventListener('toggle', onTrayToggle, true);
    window.addEventListener('ouro:dashboard-subtab-shown', onSubtab);
    window.addEventListener('ouro:page-shown', onPage);
    document.addEventListener('visibilitychange', onPage);

    function destroy() {
        if (disposed) return;
        disposed = true;
        cancelRead();
        measureButton.removeEventListener('click', onClick);
        stopButton.removeEventListener('click', onClick);
        door.removeEventListener('click', onTrayClick);
        regions.tray.removeEventListener('click', onTrayClick);
        regions.tray.removeEventListener('change', onTrayChange);
        regions.tray.removeEventListener('toggle', onTrayToggle, true);
        window.removeEventListener('ouro:dashboard-subtab-shown', onSubtab);
        window.removeEventListener('ouro:page-shown', onPage);
        document.removeEventListener('visibilitychange', onPage);
    }

    paint();
    if (onScreen()) void refresh();
    return { refresh, measure, destroy };
}
