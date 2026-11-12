// Dashboard -> Storage -> Free temporary space: the owner's one ordinary action and what stands behind it.
// The summary card's door asks the server for a pure preview; the review names the ordinary pool's exact
// composition with its size as observed, "Show all and choose" narrows it, "Not included, and why" holds what the
// pool leaves out, and one Remove sends the real ids of the chosen items. The saved groups stay reachable on purpose
// from "Not included": a checklist of whole groups, nothing checked, their loss sentences and the shared confirm
// dialog. So do the finished copies not verified saved: a list read whole, each copy a choice of its own, nothing
// chosen to begin with, what would be lost said per copy, and the shared confirm dialog as the one acknowledgement
// that the whole copy goes. Progress and the result stay in place, errors first; the result holds the door back until Done. Pure
// rendering over the server's `free` envelope (`ouroboros.storage_free.StorageFree`) and the items already read;
// the controller in storage.js owns the requests.

import { fmt } from './i18n.js';
import { escapeHtmlAttr, escapeHtmlText } from './utils.js';
import { DOTS, formatBytes, isSize, number, plural, whenLabel } from './storage_format.js';

const text = escapeHtmlText;
const attr = escapeHtmlAttr;

/** The groups the server offers, with their words: what is lost, what stays, the unit. A group that holds saved
 *  work (`saved`) is offered only in the saved-groups checklist, never in the ordinary pool, and starts unchecked:
 *  freeing it loses that work for good. `stays` says in one line why the ordinary button leaves it alone. */
export const FREE_GROUPS = {
    work_folders: {
        label: 'Finished tasks’ work folders',
        loss: 'Scratch files and helper logs of finished tasks. Results, files, messages and the exact model history are already saved.',
        unit: ['{n} folder', '{count} folders'],
    },
    working_copies: {
        label: 'Working copies of finished work',
        loss: 'Ready-to-use copies of code. Their results and commits are kept.',
        unit: ['{n} copy', '{count} copies'],
    },
    review_evidence: {
        label: 'Reviewers’ saved files of finished tasks',
        loss: 'What reviewers of finished tasks set aside or changed in their copies: test-run records (often with whole environments and databases), logs, screenshots. Their verdicts stay in the review results.',
        unit: ['{n} reviewer folder', '{count} reviewer folders'],
        stays: 'Test records, logs and screenshots behind review verdicts. Saved work.',
        saved: true,
    },
    review_checkouts: {
        label: 'Review copies no task owns',
        loss: 'Copies of reviewed code from rounds no recorded task owns. A review that needs one makes it again; changes a reviewer left in one are not kept.',
        unit: ['{n} copy', '{count} copies'],
        stays: 'No recorded task owns them, so nobody knows whether a reviewer left changes in them.',
        saved: true,
    },
    lost_copies: {
        label: 'Helper copies no record names',
        loss: 'Copies of code whose record is gone. They may hold changes that were never moved into the project.',
        unit: ['{n} copy', '{count} copies'],
        stays: 'Their record is gone, so nobody knows whether they hold changes never moved into a project.',
        saved: true,
    },
    service_folders: {
        label: 'Leftover environments and test-run records',
        loss: 'Environments of copies that are already gone, and records of finished test runs made inside copies.',
        unit: ['{n} folder', '{count} folders'],
        stays: 'Saved test-run records are mixed in with them, so the whole group stays.',
        saved: true,
    },
    result_pins: {
        label: 'Git marks of results never applied',
        loss: 'Git marks of helper results their finished task never applied or rejected. The saved change stays in the task results and can still be applied as a plain patch; the space returns at Git’s next cleanup.',
        unit: ['{n} result', '{count} results'],
        stays: 'Helper results nobody has applied or rejected yet.',
        saved: true,
    },
};

/** The server's typed kept reasons in the owner's words. */
export const KEPT_REASONS = {
    results_pending: 'their results are not copied yet',
    in_use: 'still in use',
    busy: 'busy right now',
    unreadable_record: 'their task record could not be read',
    restart: 'stopped by a restart',
    capture_unproven: 'unique work not verified saved',
    unproven_excluded_inputs: 'files only a patch rule calls disposable',
    retained_evidence: 'marked as evidence, or could not be read through',
    undeclared_environment: 'no recorded owner',
    owner_keep: 'you asked to keep it',
    changed: 'changed after the check',
    advanced_group: 'saved work: only its saved group can remove it',
    other: 'for another reason',
};
const ADVANCED = 'advanced_group';  // left out of the pool only because a saved group offers it

/** The kind of one item in the owner's words; `environment` is a whole temporary working environment. */
const KINDS = {
    work_folder: { label: 'Work folder', unit: ['{n} work folder', '{count} work folders'] },
    working_copy: { label: 'Working copy', unit: ['{n} working copy', '{count} working copies'] },
    environment: { label: 'Temporary environment', unit: ['{n} temporary environment', '{count} temporary environments'] },
    scratch: { label: 'Session scratch', unit: ['{n} session scratch folder', '{count} session scratch folders'] },
    other: { label: 'Other item', unit: ['{n} other item', '{count} other items'] },
};
const kindOf = (item) => KINDS[item?.kind] || KINDS.other;

/** The outcome of one item, in the server's order of importance (`view=outcomes` sends them in this order). */
const OUTCOMES = {
    failed: () => fmt('Could not be deleted.'),
    stuck: () => fmt('Not deleted yet; the app retries at its next maintenance run.'),
    // A copy the owner chose by hand whose work could not be saved first stays whole: its reason is that, not the
    // pool's "not verified saved".
    kept: (item) => (item.availability === 'manual_discard' && item.reason === 'capture_unproven'
        ? fmt('Kept: its work could not be saved first, so nothing was removed.')
        : fmt('Kept: {reason}.', { reason: reasonWords(item.reason, item.cause) })),
    not_started: () => fmt('Not started: left as it was.'),
    deleting: () => fmt('Moved aside; still being deleted.'),
    removed: () => fmt('Removed.'),
};

const NOT_STARTED = {
    preview_outdated: 'The list changed; check again.',
    busy: 'Something else is still running.',
    unknown_group: 'The list changed; check again.',
    unknown_item: 'The list changed; check again.',
    not_offered: 'The list changed; check again.',
    nothing_selected: 'Nothing was selected.',
    loss_not_accepted: 'The loss was not confirmed.',
};

/** The refusals that mean this plan's list can no longer be sent: the review offers Check again, not Remove. */
export const OUTDATES_PLAN = new Set(['preview_outdated', 'unknown_group', 'unknown_item', 'not_offered']);

/** The one map from a retirer's exact code (a Free item's `cause`, a pass's hold `class` on the Storage page) to the
 *  owner's words: one code reads the same in the Free tray and on a place's remover line. A code that is itself a
 *  reason group reads that group's words (`KEPT_REASONS`). */
export const CAUSES = {
    task_live: 'the task has a live owner', liveness_unknown: 'whether the task is still in use could not be established',
    owner_paused: 'the task is paused', owner_processes_live: 'a process of this task is still running',
    child_result_unadopted: 'the child result is not adopted yet', child_refs_pending: 'some result files are not copied yet',
    call_inventory_pending: 'model call records are not saved yet', call_inventory_changed: 'model call records changed during cleanup',
    inputs_uncustodied: 'input files are not saved yet', unread_mailbox_uncustodied: 'messages are not saved yet',
    verification_receipts_uncustodied: 'verification records are not saved yet', canonical_changed: 'the task record changed during cleanup',
    drive_occupant_changed: 'the folder’s task owners changed during cleanup', canonical_not_settled: 'the task is not finished',
    post_work_open: 'post-task work is still open', custody_held: 'the result is still held by its run',
    custody_busy: 'result publication is busy', mailbox_busy: 'the mailbox is busy', store_busy: 'the task store is busy',
    review_custody_active: 'the review still has an active run', seat_checking: 'a reviewer still uses this copy',
    review_environment_held: 'a retained review copy still needs this environment',
    operation_live: 'a review operation is still running', task_not_terminal: 'the task is not finished',
    review_state_unreadable: 'the review state could not be read', marker_unreadable: 'the copy ownership record could not be read',
    raw_evidence_uncustodied: 'raw evidence is not saved yet', artifact_source_missing: 'the source of a saved artifact could not be found',
    undeclared_environment: 'the environment has no recorded owner',
    task_result_unreadable: 'the task record could not be read', unread_mailbox_unreadable: 'the mailbox could not be read',
    parent_not_terminal: 'the parent task is not finished', owner_not_terminal: 'the owner task is not finished',
    not_proven_over: 'the task is not proven over (running, paused or with a live process)',
    run_open: 'a run of the task has not finished', seat_open: 'a reviewer seat still holds it', seat_busy: 'the reviewer seat is busy',
    copy_registered: 'the copy is still registered', copy_present: 'the copy still exists',
    artifacts_not_terminal: 'the task’s files are still being saved', child_drive_mismatch: 'the task record names another folder',
    artifact_source_mismatch: 'a saved artifact differs from its source', not_own_drive: 'the folder is not its task’s own',
    admission_rollback_evidence: 'the folder holds evidence of a start', record_busy: 'the record is busy',
    record_unreadable: 'the record could not be read', custody_unreadable: 'the run custody could not be read',
    capture_incomplete: 'the capture of its work is incomplete', capture_failed: 'its capture failed',
    retained_process_environment: 'a process marked this environment to keep', producer_retained: 'its producer marked it to keep',
    nothing_unsaved: 'nothing in it is unsaved any more', nested_repository: 'a repository inside it cannot be saved as a patch',
    commits_unprotected: 'its commits are not held by a branch', generation_closed: 'stopped by a restart',
    removal_failed: 'its removal failed', younger_than_retention: 'it is still inside the retention days',
    source_recovery_pending: 'startup recovery is not confirmed complete: old task folders and service logs are kept',
    closed_set_unrecorded: 'closed log files waiting for the next app restart (no earlier run of the app saw them closed)',
    prior_workers_unconfirmed: 'closed log files waiting for the next app restart (the previous run’s background processes are not confirmed stopped)',
    closed_set_changed: 'closed log files waiting for the next app restart (their log’s archive changed since the last run saw it)',
};
/** A code's words from the one map, `''` when it has none. */
const causeWords = (code) => CAUSES[code] || KEPT_REASONS[code] || '';
const reasonWords = (reason, cause = '') => cause ? fmt(causeWords(cause) || '{reason} ({cause})', { reason: fmt(KEPT_REASONS[reason] || KEPT_REASONS.other), cause }) : fmt(KEPT_REASONS[reason] || KEPT_REASONS.other);

/** A retirer pass's holds in one phrase, in the one map's words: "holding 3 since …: 2 the task has a live owner,
 *  1 you asked to keep it"; a class without words is shown by its name. */
export function holdsPhrase(fact, now) {
    const holds = Array.isArray(fact.holds) ? fact.holds.filter((hold) => hold && hold.count > 0) : [];
    // A hold whose count is unknown (`null`: what it keeps was never listed) is said by its reason alone, never a number.
    const unknown = Array.isArray(fact.holds) ? fact.holds.filter((hold) => hold && hold.class && hold.count === null) : [];
    const held = holds.length ? holds.reduce((sum, hold) => sum + hold.count, 0) : (fact.held > 0 ? fact.held : 0);
    const words = (hold) => (causeWords(hold.class) ? fmt(causeWords(hold.class)) : String(hold.class || '').replace(/_/g, ' '));
    const parts = unknown.map((hold) => (hold.since ? fmt('held since {when}: {reason}', { when: whenLabel(hold.since, now), reason: words(hold) })
        : fmt('held: {reason}', { reason: words(hold) })));
    if (held > 0) {
        const head = fact.held_since ? fmt('holding {n} since {when}', { n: number(held), when: whenLabel(fact.held_since, now) })
            : fmt('holding {n}', { n: number(held) });
        parts.unshift(holds.length ? `${head}: ${holds.map((hold) => `${number(hold.count)} ${words(hold)}`).join(', ')}` : head);
    }
    return parts.join(' · ');
}

/** What happened to the owner's own attempt, kept beside the server's current facts: `notice` outlives a plan the
 *  server replaced, `outdated` and `postUnknown` belong to the plan still shown. A lost answer is never read as nothing
 *  removed: the request may still reach the server, so that list is not offered again. */
const NOTICES = {
    refused: 'Nothing was removed: the list you reviewed had changed.',
    read: 'The list you were reviewing changed.',
    lost: 'The answer to Remove was lost and the list has changed since; whether it removed anything is not known here. Nothing is sent again.',
    lostFree: 'The answer to Free was lost and the list has changed since; whether it removed anything is not known here. Nothing is sent again.',
    lostDiscard: 'The answer to Discard was lost and the list has changed since; whether it removed anything is not known here. Nothing is sent again.',
};
/** The action whose answer was lost: 'Free' (saved groups), 'Discard' (copies by hand), else 'Remove' (the pool). */
const lostAction = (postUnknown) => (postUnknown === 'Free' || postUnknown === 'Discard' ? postUnknown : 'Remove');
export const lostNotice = (postUnknown) => (postUnknown === 'Free' ? 'lostFree' : postUnknown === 'Discard' ? 'lostDiscard' : 'lost');
export function attemptSentences(ui) {
    if (!ui || ui.dismissed) return [];
    const action = lostAction(ui.postUnknown);
    const lost = ui.confirming ? `The answer to ${action} was lost. Reading what the server says now; nothing is sent again.`
        : `The answer to ${action} was lost. It may still start, so nothing is sent again from this list. Check again for a new one.`;
    return [
        NOTICES[ui.notice] ? fmt(NOTICES[ui.notice]) : '',
        // Only the server's refusal of this send proves it removed nothing; a refused read says the list changed.
        ui.outdated === 'refused' ? fmt('The list changed; check again. Nothing was removed.') : ui.outdated ? fmt('The list changed; check again.') : '',
        ui.postUnknown ? fmt(lost) : '',
    ].filter(Boolean);
}
const attemptHtml = (ui) => attemptSentences(ui)
    .map((sentence) => `<p class="ui-status storage-status" data-tone="warn" data-i18n-fmt="storage.free.attempt">${text(sentence)}</p>`).join('');
/** A Remove or Free is out, or its answer was lost: the list takes no new choice until the server says what happened. */
const heldFlow = (ui) => Boolean(ui?.sending || ui?.postUnknown);
const inert = (held) => (held ? ' aria-disabled="true"' : '');

export const groupsOf = (free) => (Array.isArray(free?.groups) ? free.groups : []);
const words = (group) => FREE_GROUPS[group?.id] || { label: String(group?.id || ''), loss: '', unit: ['{n} folder', '{count} folders'], saved: true };
/** The groups only the saved-groups checklist offers: the ordinary pool covers the other two, and a group this
 *  client has no words for is treated as saved. */
export const savedGroups = (free) => groupsOf(free).filter((group) => group.count > 0 && words(group).saved);
const groupBytes = (group) => (isSize(group?.bytes) ? group.bytes : 0);
const chosen = (free, checked = null) => groupsOf(free).filter((group) => !checked || checked.has(group.id));
/** The sum of the sizes known for the (checked) groups: unknown is never zero, so see `sumComplete`. */
export const freeSum = (free, checked = null) => {
    let measured = false;
    const sum = chosen(free, checked).reduce((total, group) => {
        if (!isSize(group?.bytes)) return total;
        measured = true;
        return total + group.bytes;
    }, 0);
    return measured ? sum : null;
};
/** Every (checked) group with folders was sized whole: the sum is the whole truth. */
export const sumComplete = (free, checked = null) => chosen(free, checked)
    .every((group) => !group.count || (isSize(group.bytes) && group.bytes_complete !== false));
export const unitOf = (group, n) => plural(n, ...words(group).unit);
const itemsWord = (n) => plural(n, '{n} item', '{count} items');
export const excludedCount = (pool) => (Array.isArray(pool?.excluded) ? pool.excluded : []).reduce((sum, row) => sum + (row.count > 0 ? row.count : 0), 0);
/** The finished copies the owner may discard by hand: the pool's own fact, `null` when there are none. They are never in
 *  `pool.count`. */
export const manualOf = (free) => (free?.pool?.manual?.count > 0 ? free.pool.manual : null);
/** How many of an excluded row's items can be discarded by hand: that row's own `manual_count`, never the global one. */
const manualShare = (row) => (row.manual_count > 0 ? Math.min(row.manual_count, row.count) : 0);
/** The rows of what the pool leaves out that nothing can remove now: each row's count without its own manual share, and
 *  never a saved-group row (a saved group offers those). */
const blockedRows = (pool) => (Array.isArray(pool?.excluded) ? pool.excluded : [])
    .filter((row) => row.count > 0 && row.reason !== ADVANCED)
    .map((row) => ({ ...row, count: row.count - manualShare(row) })).filter((row) => row.count > 0);
/** Rows of the manual list drawn at a time; a choice never depends on what is drawn. */
export const MANUAL_ROWS = 25;
/** A row of `view=manual` the owner may choose: the server offers it by hand and says what would be lost. Anything
 *  else is shown but never chosen, so no id the server did not offer is ever sent. */
export const manualKnown = (item) => item?.availability === 'manual_discard' && (item.manual?.producer_retained
    ? typeof item.manual.loss === 'string' && item.manual.loss.length > 0
    : Number.isInteger(item.manual?.not_proven_saved_count) && isSize(item.manual.not_proven_saved_bytes));

/** What the by-hand form holds: finished copies not verified saved, session scratch its run marked to keep
 *  (`manual.producer_retained`), or both; each reads its own nouns, never a scratch called a copy. */
const holdsOf = (copies, scratch) => (scratch > 0 ? (copies > 0 ? 'both' : 'scratch') : 'copies');
const BY_HAND = {
    copies: { choose: 'Choose copies…', note: 'Only if you choose; each copy is deleted whole.', head: 'Copies not verified saved',
        title: 'Discard these copies?', show: 'Show the copies', kept: 'Task results, history and commits on a branch that stays are kept.' },
    scratch: { choose: 'Choose scratch…', note: 'Only if you choose; its run marked it to keep.', head: 'Session scratch marked to keep',
        title: 'Discard the marked scratch?', show: 'Show the scratch', kept: 'Task results and history are kept.' },
    both: { choose: 'Choose…', note: 'Only if you choose; each is deleted whole.', head: 'Copies not verified saved, and marked scratch',
        title: 'Discard these copies and marked scratch?', show: 'Show what goes', kept: 'Task results, history and commits on a branch that stays are kept.' },
};
const copiesLine = (n) => plural(n, '{n} finished copy not verified saved', '{count} finished copies not verified saved');
const scratchLine = (n) => plural(n, '{n} session scratch its run marked to keep', '{count} session scratch folders their runs marked to keep');
const retainedOf = (item) => Boolean(item?.manual?.producer_retained);

/** 'rest' | 'listing' | 'ready' | 'empty' | 'running' | 'done' | 'failed' (a dismissed list or result reads 'rest').
 *  A ready plan is 'ready' while it holds anything to say: the pool, what it leaves out, or a saved group. */
export function trayState(free, dismissed = false) {
    const state = free?.state || 'idle';
    if (state === 'ready') {
        if (dismissed) return 'rest';
        const pool = free.pool;
        return pool?.count > 0 || excludedCount(pool) > 0 || manualOf(free) || groupsOf(free).some((group) => group.count > 0) ? 'ready' : 'empty';
    }
    if (state === 'stopped' || state === 'done' || (state === 'failed' && free.run)) return free.run && !dismissed ? 'done' : 'rest';
    return state === 'listing' || state === 'running' || state === 'failed' ? state : 'rest';
}

// ---------------------------------------------------------------------------
// A list the server pages: every item read, in order, or say what is missing
// ---------------------------------------------------------------------------

export const EMPTY_LIST = () => ({ items: [], total: null, version: '', next: 0, complete: false, loading: false, error: '', counts: {}, want: 0, restarts: 0, ids: new Set() });

/** Take one page into `list`, or leave the list as it was and name why the page cannot be trusted: 'moved' when the
 *  version moved without the server's 409, 'incomplete' for an answer that would leave an item missing or counted
 *  twice. The caller never sends a selection or claims a whole list from a list that is not `complete`. */
export function acceptPage(list, page, { plan, view, expected = null } = {}) {
    if (page?.plan !== plan || page?.view !== view || !Array.isArray(page.items) || typeof page.total !== 'number') return 'incomplete';
    if (list.version && page.version !== list.version) return 'moved';
    if ((list.total !== null && page.total !== list.total) || (expected !== null && page.total !== expected)) return 'incomplete';
    const ids = new Set(list.ids);
    for (const item of page.items) {
        if (!item || typeof item.id !== 'string' || !item.id || ids.has(item.id)) return 'incomplete';
        ids.add(item.id);
    }
    const next = page.next_cursor ?? null;
    const held = list.items.length + page.items.length;
    if (next === null ? held !== page.total : !Number.isInteger(next) || next <= list.next || !page.items.length) return 'incomplete';
    list.items.push(...page.items);
    list.ids = ids;
    list.total = page.total;
    list.version = page.version || '';
    list.counts = page.counts && typeof page.counts === 'object' ? page.counts : {};
    list.next = next;
    list.complete = next === null;
    return '';
}

// ---------------------------------------------------------------------------
// One item and its size
// ---------------------------------------------------------------------------

/** An item's or a group's size: unknown or incomplete is said, never zero. */
function sizeWords(entry) {
    if (!isSize(entry?.bytes)) return fmt('size unknown');
    return entry.bytes_complete === false ? fmt('{size} (some could not be read)', { size: formatBytes(entry.bytes) }) : formatBytes(entry.bytes);
}

const sizeCell = (entry) => `<span class="storage-free-size${isSize(entry?.bytes) && entry.bytes_complete !== false ? '' : ' is-qualified'}" data-i18n-fmt="storage.size">${text(sizeWords(entry))}</span>`;

/** When the app's own pass may reach the item: its earliest eligibility, never a promised deletion. */
function eligibleWords(item, now) {
    if (item.eligible_after === null || item.eligible_after === undefined) return fmt('the app does not clear this by itself');
    const when = whenLabel(item.eligible_after, now);
    return when ? fmt('the app may clear this by itself from {when}', { when }) : '';
}

function pickRow(item, index, chosenIds, now, held = false) {
    const note = [fmt(kindOf(item).label), eligibleWords(item, now)].filter(Boolean).join(' · ');
    return `<li class="storage-free-group" data-free-item>
        <input type="checkbox" class="ui-checkbox" id="storage-pick-${index}" data-free-pick="${index}" data-key="pick-${index}"${chosenIds.has(item.id) ? ' checked' : ''}${held ? ' disabled' : ''}>
        <label class="storage-free-label" for="storage-pick-${index}" data-i18n-authored>${text(item.name)}</label>
        ${sizeCell(item)}
        <p class="storage-free-why" data-i18n-fmt="storage.free.item">${text(note)}</p>
        <p class="storage-free-why storage-free-path" data-i18n-authored>${text(item.path)}</p>
    </li>`;
}

/** An item that stays out of the pool: not sized by the server, so a missing size says nothing at all. */
function excludedRow(item) {
    return `<li class="storage-free-group" data-free-excluded="${attr(item.reason || 'other')}"${item.availability ? ` data-availability="${attr(item.availability)}"` : ''}>
        <span class="storage-free-label" data-i18n-authored>${text(item.name)}</span>
        ${isSize(item.bytes) ? sizeCell(item) : ''}
        <p class="storage-free-why" data-i18n-fmt="storage.free.item">${text(`${fmt(kindOf(item).label)} · ${reasonWords(item.reason, item.cause)}${item.availability === 'manual_discard' ? ` · ${fmt('can be discarded by hand')}` : ''}`)}</p>
        <p class="storage-free-why storage-free-path" data-i18n-authored>${text(item.path)}</p>
    </li>`;
}

function outcomeRow(item) {
    const said = OUTCOMES[item.outcome] || (() => fmt('Outcome unknown.'));
    return `<li class="storage-free-group" data-outcome="${attr(item.outcome || '')}">
        <span class="storage-free-label" data-i18n-authored>${text(item.name)}</span>
        ${sizeCell(item)}
        <p class="storage-free-why" data-i18n-fmt="storage.free.item">${text(`${fmt(kindOf(item).label)} · ${said(item)}`)}</p>
    </li>`;
}

// `data` is the button's one data attribute (`data-free-go`); it doubles as the focus key. A label
// composed with `fmt` names its key so the language overlay leaves the rendered text alone.
const button = (label, data, tone = 'btn-default', key = '', extra = '') => `<button type="button" class="btn ${tone} btn-sm" ${data} data-key="${attr(data)}"${key ? ` data-i18n-fmt="${attr(key)}"` : ''}${extra}>${text(label)}</button>`;

function disclosure(key, summary, body, summaryKey = 'storage.free.more') {
    return `<details class="activity-history storage-largest storage-free-more" data-key="${attr(key)}" data-free-disclosure="${attr(key)}">
        <summary data-key="${attr(key)}-open" data-i18n-fmt="${attr(summaryKey)}">${text(summary)}</summary>${body}</details>`;
}

// ---------------------------------------------------------------------------
// The review: the pool, what it leaves out, one Remove
// ---------------------------------------------------------------------------

/** What is chosen now, once every id is read; until then only the pool's own facts. `bytes` sums the known sizes. */
function chosenFacts(free, ui) {
    const list = ui.pool;
    if (!list?.complete) return { count: free.pool?.count || 0, bytes: free.pool?.bytes ?? null, whole: free.pool?.bytes_complete !== false && isSize(free.pool?.bytes), kinds: free.pool?.kinds || {}, read: false };
    const rows = list.items.filter((item) => ui.chosen?.has(item.id));
    const kinds = {};
    for (const row of rows) kinds[KINDS[row.kind] ? row.kind : 'other'] = (kinds[KINDS[row.kind] ? row.kind : 'other'] || 0) + 1;
    const known = rows.filter((row) => isSize(row.bytes));
    return {
        count: rows.length,
        bytes: known.length ? known.reduce((sum, row) => sum + row.bytes, 0) : null,
        whole: rows.every((row) => isSize(row.bytes) && row.bytes_complete !== false),
        kinds,
        read: true,
    };
}

function listStatus(list, { reading, problem, retry }) {
    if (list.error) return `<p class="ui-status storage-status" data-tone="warn" data-i18n-fmt="storage.free.list">${text(problem)}</p><div class="storage-free-picks">${button('Try again', retry)}</div>`;
    if (list.loading || (!list.complete && list.want > list.items.length)) return `<p class="storage-progress" data-i18n-fmt="storage.free.list">${DOTS}<span>${text(reading)}</span></p>`;
    return '';
}

function poolStatus(free, ui) {
    return listStatus(ui.pool, {
        reading: fmt('Reading the list · {done} of {total}', { done: number(ui.pool.items.length), total: number(free.pool?.count) }),
        problem: fmt('The whole list could not be read, so nothing can be removed from here yet.'),
        retry: 'data-free-retry-pool',
    });
}

function showAllHtml(free, ui, now) {
    const list = ui.pool;
    const total = free.pool?.count || 0;
    if (!list.complete || !total) return '';
    const shown = list.items.slice(0, ui.poolShown || 100);
    const count = list.items.filter((item) => ui.chosen?.has(item.id)).length;
    const more = list.items.length - shown.length;
    // Rows not drawn yet keep their choice, and Remove sends every chosen id: the line says how many of them go.
    const hiddenChosen = list.items.slice(shown.length).filter((item) => ui.chosen?.has(item.id)).length;
    const hidden = { shown: number(shown.length), total: number(list.items.length), hidden: number(more) };
    const held = heldFlow(ui);
    const hiddenLine = hiddenChosen
        ? fmt('Showing {shown} of {total}. The {hidden} not shown keep their choice: {chosen} chosen, and Remove takes those too.', { ...hidden, chosen: number(hiddenChosen) })
        : fmt('Showing {shown} of {total}. The {hidden} not shown keep their choice: none chosen.', hidden);
    const body = `<p class="storage-meta">Results, messages and model history of these tasks are already saved. Each item is removed whole, with whatever was installed in it.</p><div class="storage-free-picks">
                <p class="storage-meta" data-i18n-fmt="storage.free.picked">${text(fmt('{chosen} of {total} chosen', { chosen: number(count), total: number(total) }))}</p>
                ${button(fmt('Choose all {total}', { total: number(total) }), 'data-free-all', 'btn-ghost', 'storage.free.all', inert(held))}
                ${button('Choose none', 'data-free-none', 'btn-ghost', '', inert(held))}
            </div>
            <ul class="storage-free-groups storage-free-list">${shown.map((item, index) => pickRow(item, index, ui.chosen || new Set(), now, held)).join('')}</ul>
            ${more > 0 ? `<p class="storage-meta" data-i18n-fmt="storage.free.shown">${text(hiddenLine)}</p>
                ${button(fmt('Show {n} more', { n: number(Math.min(more, 100)) }), 'data-free-more-picks', 'btn-ghost', 'storage.free.more.picks')}` : ''}`;
    return disclosure('included', fmt('Review the {total} items', { total: number(total) }), body);
}

function excludedHtml(free, ui) {
    const list = ui.excluded || EMPTY_LIST();
    const reasons = (Array.isArray(free.pool?.excluded) ? free.pool.excluded : []).filter((row) => row.count > 0);
    const saved = savedGroups(free);
    const more = list.total !== null && list.items.length < list.total;
    // A saved-group item is not blocked: the saved groups below count and offer it. Nor is a copy offered by hand: each
    // reason row gives up only its own `manual_count`.
    const blocked = blockedRows(free.pool);
    const manual = manualOf(free);
    const decisionRow = (label, entry, action, note) => {
        const qualified = !isSize(entry?.bytes) || entry.bytes_complete === false;
        return `<li class="storage-decision-row"><span class="storage-decision-label" data-i18n-fmt="storage.free.decision">${text(label)}<span class="storage-meta">${text(note)}</span></span><span class="storage-decision-size${qualified ? ' is-qualified' : ''}" data-i18n-fmt="storage.size">${text(sizeWords(entry))}</span>${action}</li>`;
    };
    const copiesEntry = manual && { bytes: manual.bytes, bytes_complete: manual.bytes_complete };
    const scratch = manual ? Math.min(manual.retained > 0 ? manual.retained : 0, manual.count) : 0;
    const said = BY_HAND[holdsOf(manual ? manual.count - scratch : 0, scratch)];
    const label = manual ? [manual.count > scratch ? copiesLine(manual.count - scratch) : '', scratch ? scratchLine(scratch) : ''].filter(Boolean).join(' · ') : '';
    const copies = manual ? decisionRow(label, copiesEntry, button(said.choose, 'data-free-manual-open', 'btn-default', '', inert(heldFlow(ui))), fmt(said.note)) : '';
    const groupsEntry = saved.length && { bytes: freeSum(free, new Set(saved.map((g) => g.id))), bytes_complete: sumComplete(free, new Set(saved.map((g) => g.id))) };
    const groups = saved.length ? decisionRow(fmt('Saved work in {groups}', { groups: plural(saved.length, '{n} group', '{count} groups') }), groupsEntry, button('Choose groups…', 'data-free-saved', 'btn-default', '', inert(heldFlow(ui))), fmt('Only if you choose; a whole group is deleted permanently.')) : '';
    const each = reasons.length ? `<h4 class="storage-free-sub">Each item left out</h4>
        ${list.items.length ? `<ul class="storage-free-groups storage-free-list is-single">${list.items.map(excludedRow).join('')}</ul>` : ''}
        ${listStatus(list, { reading: fmt('Reading the list…'), problem: fmt('The list of what stays could not be read in full.'), retry: 'data-free-retry-excluded' })}
        ${more && !list.loading && !list.error ? button(fmt('Show {n} more', { n: number(Math.min(list.total - list.items.length, 100)) }), 'data-free-more-excluded', 'btn-ghost', 'storage.free.more.excluded') : ''}` : '';
    const kept = blocked.length ? disclosure('left', fmt('Kept · {reasons}', { reasons: blocked.map((row) => `${number(row.count)} ${reasonWords(row.reason)}`).join(' · ') }), each) : reasons.length ? disclosure('left', fmt('Each item left out ({total})', { total: number(excludedCount(free.pool)) }), each) : '';
    return copies || groups || kept ? `<div class="storage-decisions"><h4 class="storage-free-sub">Your decision, and what is not included</h4><ul class="storage-decision-rows">${copies}${groups}</ul>${kept}</div>` : '';

}

function reviewActions(free, ui, facts) {
    // A lost answer leaves one way on: a new list, under which the lost request can no longer start.
    if (ui.postUnknown) {
        return `<div class="storage-actions">${ui.confirming ? button('Checking what happened…', 'data-free-go', 'btn-primary', '', inert(true)) : button('Check again', 'data-free-preview')}</div>`;
    }
    const cancel = button(facts.total > 0 ? 'Cancel' : 'Close', 'data-free-cancel', 'btn-default', '', inert(ui.sending));
    if (ui.outdated) return `<div class="storage-actions">${cancel}${button('Check again', 'data-free-preview')}</div>`;
    if (!facts.total) return `<div class="storage-actions">${cancel}</div>`;
    if (ui.pool?.error) return `<div class="storage-actions">${cancel}</div>`;
    if (!ui.pool?.complete) return `<div class="storage-actions">${cancel}${button('Reading the list…', 'data-free-go', 'btn-primary', '', ' aria-disabled="true"')}</div>`;
    if (ui.sending) return `<div class="storage-actions">${cancel}${button('Sending…', 'data-free-go', 'btn-primary', '', inert(true))}</div>`;
    const go = facts.count ? button(plural(facts.count, 'Remove {n} selected item', 'Remove {count} selected items'), 'data-free-go', 'btn-primary', 'storage.free.go')
        : '<span class="storage-meta">Nothing selected</span>';
    return `<div class="storage-actions">${cancel}${go}</div>`;
}

const figure = (label, value, meta = '', quiet = false) => `<div class="storage-cleanup-figure${quiet ? ' is-quiet' : ''}"><span class="storage-kicker">${text(fmt(label))}</span><span class="storage-cleanup-amount" data-i18n-fmt="storage.size">${text(value)}</span>${meta ? `<span class="storage-meta" data-i18n-fmt="storage.free.figure">${text(meta)}</span>` : ''}</div>`;

function reviewHtml(free, ui, now) {
    const facts = { ...chosenFacts(free, ui), total: free.pool?.count || 0 };
    const checked = free.finished_at ? fmt('Checked {when}', { when: whenLabel(free.finished_at, now) }) : '';
    const counted = !free.pool ? fmt('Not available') : !facts.total ? fmt('Nothing right now') : sizeWords(free.pool);
    const selected = !ui.pool?.complete && facts.total ? fmt('Reading the list…') : !facts.count ? fmt('Nothing') : sizeWords({ bytes: facts.bytes, bytes_complete: facts.whole });
    return `<div class="storage-meta" data-i18n-fmt="storage.free.checked">${text(checked)}</div>${attemptHtml(ui)}${!free.pool ? '<p class="ui-status storage-status" data-tone="warn">The ordinary list is not available, so nothing can be removed from here. Check again.</p>' : ''}
        <div class="storage-cleanup-figures">${figure('Can be removed', counted, itemsWord(facts.total))}
            ${facts.total ? figure('Selected', selected, fmt('{chosen} of {total} items', { chosen: number(facts.count), total: number(facts.total) }), true) : ''}
            ${facts.total || ui.postUnknown || ui.outdated ? reviewActions(free, ui, facts) : `<div class="storage-actions">${button('Check again', 'data-free-preview')}</div>`}</div>
        ${ui.pool?.complete || ui.outdated || !facts.total ? '' : poolStatus(free, ui)}
        ${showAllHtml(free, ui, now)}${excludedHtml(free, ui)}
        <p class="storage-meta storage-cleanup-foot">Sizes are what the files take on disk; free space can grow by less.</p>`;
}

// ---------------------------------------------------------------------------
// The saved groups: reached on purpose, nothing checked, the shared confirm dialog
// ---------------------------------------------------------------------------

function groupWhy(group, days) {
    const said = words(group);
    return [
        said.loss ? fmt(said.loss) : '',
        group.inside_wait ? fmt('{n} of {total} folders finished less than {days} days ago.', { n: number(group.inside_wait), total: number(group.count), days }) : '',
        group.in_use ? plural(group.in_use, '{n} stays; each item names its reason below.', '{count} stay; each item names its reason below.') : '',
        group.unreadable ? plural(group.unreadable, '{n} stays: its task record could not be read.', '{count} stay: their task record could not be read.') : '',
    ].filter(Boolean).join(' ');
}

function checklistHtml(free, checked, days, held = false) {
    const groups = savedGroups(free);
    return `<ul class="storage-free-groups">${groups.map((group) => `<li class="storage-free-group">
            <input type="checkbox" class="ui-checkbox" id="storage-free-${attr(group.id)}" data-free-group="${attr(group.id)}" data-key="check-${attr(group.id)}"${checked.has(group.id) ? ' checked' : ''}${held ? ' disabled' : ''}>
            <label class="storage-free-label" for="storage-free-${attr(group.id)}">${text(fmt(words(group).label))}</label>
            ${sizeCell(group)}
            <p class="storage-free-why" data-i18n-fmt="storage.free.why">${text(groupWhy(group, days))}</p>
        </li>`).join('')}</ul>`;
}

function largestHtml(free) {
    const groups = savedGroups(free);
    const items = groups
        .flatMap((group) => (Array.isArray(group.items) ? group.items : []).map((item) => ({ ...item, group: fmt(words(group).label) })))
        .filter((item) => isSize(item.bytes))
        .sort((a, b) => b.bytes - a.bytes);
    if (!items.length) return '';
    const total = groups.reduce((sum, group) => sum + (group.count || 0), 0);
    return `<details class="activity-history storage-largest" data-key="largest">
            <summary data-key="largest-open" data-i18n-fmt="storage.free.largest">${text(fmt('Largest items ({shown} of {total})', { shown: number(items.length), total: number(total) }))}</summary>
            <p class="storage-meta">Freeing removes every checked group, not only these.</p>
            <ul class="storage-largest-list">${items.map((item) => `<li class="storage-largest-item">
                <span class="storage-largest-name" data-i18n-authored>${text(item.name)}</span>
                <span class="storage-largest-size" data-i18n-fmt="storage.size">${text(formatBytes(item.bytes))}</span>
                <span class="storage-largest-group">${text(item.group)}</span>
            </li>`).join('')}</ul>
        </details>`;
}

/** The danger capsule's label: the sum when every checked folder was sized, else the honest rest. */
export function goLabel(free, checked) {
    const sum = freeSum(free, checked);
    if (sumComplete(free, checked)) return fmt('Free {size}', { size: formatBytes(sum || 0) });
    return sum !== null ? fmt('Free {size} or more', { size: formatBytes(sum) }) : fmt('Free, size unknown');
}

function savedHtml(free, ui, days) {
    const checked = ui.checked || new Set();
    const selected = savedGroups(free).some((group) => checked.has(group.id));
    // A refused plan or a lost answer offers Check again here too: no destructive capsule stands on a list the server
    // no longer holds or may still be running.
    const go = ui.postUnknown ? (ui.confirming ? '<span class="storage-meta">Checking what happened…</span>' : button('Check again', 'data-free-preview'))
        : ui.outdated ? button('Check again', 'data-free-preview')
            : ui.sending ? button('Sending…', 'data-free-saved-go', 'btn-danger', '', inert(true))
                : selected ? button(goLabel(free, checked), 'data-free-saved-go', 'btn-danger', 'storage.free.saved.go')
                    : '<span class="storage-meta">Nothing selected</span>';
    const back = ui.postUnknown ? '' : button('Back', 'data-free-saved-back', 'btn-default', '', inert(ui.sending));
    return `${headHtml('Saved groups')}
        ${attemptHtml(ui)}
        <p class="storage-tray-text">These hold saved work, so nothing here starts checked. Each says what goes.</p>
        ${checklistHtml(free, checked, days, heldFlow(ui))}
        ${largestHtml(free)}
        <p class="storage-meta">Gone for good. Sizes are what the listed folders occupy; the space freed can be smaller.</p>
        <div class="storage-actions">${back}${go}</div>`;
}

// ---------------------------------------------------------------------------
// Finished copies not verified saved: each a choice of its own, nothing chosen, the whole copy goes
// ---------------------------------------------------------------------------

/** A size as counted, never promised: unknown is said, an incomplete sum says so. */
function countedWords(bytes, whole) {
    if (!isSize(bytes)) return fmt('size unknown');
    return whole ? fmt('{size} counted', { size: formatBytes(bytes) }) : fmt('{size} counted, some sizes unknown', { size: formatBytes(bytes) });
}

/** What discarding one copy loses, in words: the files not verified saved and what stays. */
function unsavedWords(manual) {
    const stays = manual.keeps_commits ? ` ${fmt('Its branch and the commits on it stay.')}` : '';
    if (!(manual.not_proven_saved_count > 0)) return `${fmt('No file in it is unsaved.')}${stays}`;
    const files = plural(manual.not_proven_saved_count, '{n} file', '{count} files');
    return `${fmt('{files} not verified saved · {size}.', { files, size: formatBytes(manual.not_proven_saved_bytes) })}${stays}`;
}

/** The largest files not verified saved of one copy (at most fifty, paths relative to the copy) and how many are not listed. */
function unsavedPaths(manual, index) {
    const paths = (Array.isArray(manual.not_proven_saved) ? manual.not_proven_saved : []).filter((row) => row && typeof row.path === 'string');
    if (!paths.length) return '';
    const summary = manual.omitted_count > 0
        ? fmt('Show the {shown} largest of {total} files', { shown: number(paths.length), total: number(manual.not_proven_saved_count) })
        : plural(paths.length, 'Show the file', 'Show the {count} files');
    const more = manual.omitted_count > 0
        ? `<p class="storage-meta" data-i18n-fmt="storage.free.manual.omitted">${text(plural(manual.omitted_count, '{n} more is not listed.', '{count} more are not listed.'))}</p>` : '';
    return `<details class="activity-history storage-largest storage-free-paths" data-key="manual-paths-${index}">
        <summary data-key="manual-paths-${index}-open" data-i18n-fmt="storage.free.manual.paths">${text(summary)}</summary>
        <ul class="storage-largest-list">${paths.map((row) => `<li class="storage-largest-item">
            <span class="storage-largest-name storage-free-path" data-i18n-authored>${text(row.path)}</span>
            <span class="storage-largest-size" data-i18n-fmt="storage.size">${text(isSize(row.bytes) ? formatBytes(row.bytes) : fmt('size unknown'))}</span>
        </li>`).join('')}</ul>${more}</details>`;
}

function manualRow(item, index, picked, held) {
    const choosable = manualKnown(item);
    const scratch = retainedOf(item);
    const what = choosable ? scratch
        ? `${fmt('Discarding deletes it and what it holds for good.')} ${fmt('Its marker says: {words}', { words: item.manual.loss })}`
        : unsavedWords(item.manual) : fmt('What it would lose could not be read, so it cannot be chosen here.');
    const where = scratch && Array.isArray(item.manual.paths) && item.manual.paths.length ? item.manual.paths.join(' · ') : item.path;
    return `<li class="storage-free-group" data-free-manual-item="${attr(item.id)}">
        <input type="checkbox" class="ui-checkbox" id="storage-manual-${index}" data-free-manual-pick="${index}" data-key="manual-${index}"${picked.has(item.id) && choosable ? ' checked' : ''}${held || !choosable ? ' disabled' : ''}>
        <label class="storage-free-label" for="storage-manual-${index}" data-i18n-authored>${text(item.name)}</label>
        ${sizeCell(item)}
        <p class="storage-free-why" data-i18n-fmt="storage.free.item">${text(`${scratch ? fmt('Session scratch its run marked to keep') : fmt(kindOf(item).label)} · ${fmt('the whole folder goes')}`)}</p>
        <p class="storage-free-why storage-free-path" data-i18n-authored>${text(where)}</p>
        <p class="storage-free-why storage-free-risk" data-i18n-fmt="storage.free.manual.risk">${text(what)}</p>
        ${choosable ? unsavedPaths(item.manual, index) : ''}
    </li>`;
}

/** What is chosen by hand, over the whole list once it is read (rows not drawn count too): copies, folder size and the
 *  files not verified saved inside them. */
export function manualFacts(list, picked) {
    const rows = (list?.complete ? list.items : []).filter((item) => picked?.has(item.id) && manualKnown(item));
    const sized = rows.filter((row) => isSize(row.bytes));
    return {
        rows,
        count: rows.length,
        bytes: sized.length ? sized.reduce((sum, row) => sum + row.bytes, 0) : null,
        whole: rows.every((row) => isSize(row.bytes) && row.bytes_complete !== false),
        unsavedCount: rows.reduce((sum, row) => sum + row.manual.not_proven_saved_count, 0),
        unsavedBytes: rows.reduce((sum, row) => sum + row.manual.not_proven_saved_bytes, 0),
        scratch: rows.filter(retainedOf).length,
    };
}

const copiesWord = (n) => plural(n, '{n} copy', '{count} copies');
const discardLabel = (facts) => (facts.count && facts.scratch === facts.count ? plural(facts.count, 'Discard {n} scratch folder', 'Discard {count} scratch folders')
    : facts.scratch ? plural(facts.count, 'Discard {n} item', 'Discard {count} items') : fmt('Discard {copies}', { copies: copiesWord(facts.count) }));

function manualHtml(free, ui) {
    const list = ui.manual || EMPTY_LIST();
    const picked = ui.manualChosen || new Set();
    const total = free.pool?.manual?.count || 0;
    const facts = manualFacts(list, picked);
    const held = heldFlow(ui);
    const shown = list.items.slice(0, ui.manualShown || MANUAL_ROWS);
    const more = list.items.length - shown.length;
    const hiddenChosen = list.items.slice(shown.length).filter((item) => picked.has(item.id) && manualKnown(item)).length;
    const sum = facts.count ? `${plural(facts.count, '{n} chosen', '{count} chosen')} · ${countedWords(facts.bytes, facts.whole)}` : '';
    const status = listStatus(list, {
        reading: fmt('Reading the list · {done} of {total}', { done: number(list.items.length), total: number(total) }),
        problem: fmt('The whole list could not be read, so nothing can be discarded from here yet.'),
        retry: 'data-free-retry-manual',
    });
    // The list is offered whole: a choice is made only on a list read to its last copy, never on a part of it.
    const body = list.complete ? `<div class="storage-free-picks">
            <p class="storage-meta" data-i18n-fmt="storage.free.picked">${text(fmt('{chosen} of {total} chosen', { chosen: number(facts.count), total: number(list.items.length) }))}</p>
            ${facts.count ? button('Choose none', 'data-free-manual-none', 'btn-ghost', '', inert(held)) : ''}
        </div>
        <ul class="storage-free-groups storage-free-list">${shown.map((item, index) => manualRow(item, index, picked, held)).join('')}</ul>
        ${more > 0 ? `<p class="storage-meta" data-i18n-fmt="storage.free.shown">${text(hiddenChosen
        ? fmt('Showing {shown} of {total}. The {hidden} not shown keep their choice: {chosen} chosen, and Discard takes those too.', { shown: number(shown.length), total: number(list.items.length), hidden: number(more), chosen: number(hiddenChosen) })
        : fmt('Showing {shown} of {total}. The {hidden} not shown keep their choice: none chosen.', { shown: number(shown.length), total: number(list.items.length), hidden: number(more) }))}</p>
            ${button(fmt('Show {n} more', { n: number(Math.min(more, MANUAL_ROWS)) }), 'data-free-more-manual', 'btn-ghost', 'storage.free.more.manual')}` : ''}` : '';
    const go = ui.postUnknown ? (ui.confirming ? '<span class="storage-meta">Checking what happened…</span>' : button('Check again', 'data-free-preview'))
        : ui.outdated ? button('Check again', 'data-free-preview')
            : ui.sending ? button('Sending…', 'data-free-manual-go', 'btn-danger', '', inert(true))
                : list.error ? ''
                    : !list.complete ? button('Reading the list…', 'data-free-manual-go', 'btn-default', '', ' aria-disabled="true"')
                        : facts.count ? button(discardLabel(facts), 'data-free-manual-go', 'btn-danger', 'storage.free.manual.go')
                            : '<span class="storage-meta">Nothing selected</span>';
    const back = ui.postUnknown ? '' : button('Back', 'data-free-manual-back', 'btn-default', '', inert(ui.sending));
    const scratch = list.complete ? list.items.filter(retainedOf).length : Math.min(free.pool?.manual?.retained || 0, total);
    const holds = holdsOf((list.complete ? list.items.length : total) - scratch, scratch);
    const copyText = holds === 'scratch' ? '' : `<p class="storage-tray-text">Finished work copies the app’s own cleanup leaves alone, because some files in them are not verified saved. Nothing here starts chosen.</p>
        <p class="ui-status storage-status" data-tone="warn">Discarding a copy deletes its whole folder, not only the files listed. What is not verified saved can be lost for good. Task results, history and commits on a branch that stays are kept.</p>`;
    const scratchText = holds === 'copies' ? '' : `<p class="storage-tray-text">Session scratch its run marked as evidence to keep; the app keeps it until you discard it. Nothing here starts chosen.</p>
        <p class="ui-status storage-status" data-tone="warn">Discarding marked scratch deletes it and the evidence in it for good. Task results and history are kept.</p>`;
    return `${headHtml(BY_HAND[holds].head, sum)}
        ${attemptHtml(ui)}
        ${copyText}${scratchText}
        ${ui.outdated || ui.postUnknown ? '' : status}${ui.outdated || ui.postUnknown ? '' : body}
        <p class="storage-meta">Gone for good. Sizes are what the whole folders occupy; the space freed can be smaller.</p>
        <div class="storage-actions">${back}${go}</div>`;
}

/** What the confirm dialog shows for the copies chosen by hand: the loss in words, one detail row per copy. Pressing its
 *  confirm is the owner's acknowledgement that each whole copy goes with whatever is not verified saved in it. */
export function manualConfirmCopy(list, picked) {
    const facts = manualFacts(list, picked);
    const copies = facts.count - facts.scratch;
    const said = BY_HAND[holdsOf(copies, facts.scratch)];
    const goes = [copies ? plural(copies, 'This deletes {n} copy whole.', 'This deletes {count} copies whole.') : '',
        copies && facts.unsavedCount > 0
            ? fmt('{files} in them are not verified saved ({size}) and can be lost for good.', { files: plural(facts.unsavedCount, '{n} file', '{count} files'), size: formatBytes(facts.unsavedBytes) })
            : copies ? fmt('No file in them is unsaved.') : '',
        facts.scratch ? plural(facts.scratch, 'This deletes {n} scratch folder its run marked to keep, and the evidence in it.', 'This deletes {count} scratch folders their runs marked to keep, and the evidence in them.') : ''];
    return {
        title: fmt(said.title),
        body: `${goes.filter(Boolean).join(' ')} ${fmt(said.kept)}`,
        details: {
            summary: fmt(said.show),
            rows: facts.rows.map((item) => ({ label: item.name, value: retainedOf(item)
                ? `${sizeWords(item)} · ${fmt('marked to keep: {words}', { words: item.manual.loss })}` : `${sizeWords(item)} · ${item.manual.not_proven_saved_count > 0
                    ? fmt('{files} not verified saved', { files: plural(item.manual.not_proven_saved_count, '{n} file', '{count} files') }) : fmt('no file unsaved')}` })),
        },
        confirmLabel: discardLabel(facts),
        danger: true,
    };
}

// ---------------------------------------------------------------------------
// Progress and the result
// ---------------------------------------------------------------------------

const deletingSentence = (run) => (run?.deleting ? plural(run.deleting, '{n} is still being deleted.', '{count} are still being deleted.') : '');
const stuckSentence = (run) => (run?.stuck ? plural(run.stuck, '{n} could not be deleted yet; the app retries at its next maintenance run.', '{count} could not be deleted yet; the app retries at its next maintenance run.') : '');
const failedSentence = (run) => (run?.failed ? plural(run.failed, '{n} could not be deleted.', '{count} could not be deleted.') : '');
const notStartedSentence = (run) => (run?.not_started ? plural(run.not_started, '{n} was not started and stays as it was.', '{count} were not started and stay as they were.') : '');

function runningHtml(free) {
    const { run } = free;
    const sentence = fmt('Removing · {done} of {total} · {size} of their counted size gone so far. You can leave this tab.', {
        done: number(run?.done), total: number(run?.selected || free.total), size: formatBytes(run?.removed_bytes || 0),
    });
    return `<p class="storage-progress" data-i18n-fmt="storage.free.running">${DOTS}<span>${text([sentence, deletingSentence(run)].filter(Boolean).join(' '))}</span></p>
        <div class="storage-actions">${button('Stop', 'data-free-stop')}</div>`;
}

function listingHtml(free) {
    // While it lists, the pool's sizes are still filling in: no size yet is said by saying none, never as zero.
    const size = free.pool ? free.pool.bytes : freeSum(free);
    const counted = { done: number(free.done), total: number(free.total) };
    const sentence = isSize(size) && size > 0
        ? fmt('Checking for files to remove · {done} of {total} folders · {size} so far. Nothing is removed.', { ...counted, size: formatBytes(size) })
        : fmt('Checking for files to remove · {done} of {total} folders. Nothing is removed.', counted);
    return `<p class="storage-progress" data-i18n-fmt="storage.free.listing">${DOTS}<span>${text(sentence)}</span></p>
        <div class="storage-actions">${button('Stop', 'data-free-stop')}</div>`;
}

/** The result in the owner's words, what went wrong first: `{problems, lead, rest}`. The size is what the removed
 *  items were counted at and free space is the disk's own reading: two facts, neither promised by the other. */
export function resultSentences(free, volume = null, now = new Date()) {
    const run = free?.run;
    if (!run) return { problems: [], lead: '', rest: [] };
    const removed = itemsWord(run.removed || 0);
    const lead = run.stopped ? fmt('Removed {items}, then stopped. The rest was left as it was.', { items: removed }) : fmt('Removed {items}.', { items: removed });
    const kept = (run.kept || []).filter((row) => row.count > 0).map((row) => (
        plural(row.count, '{n} was kept: {reason}.', '{count} were kept: {reason}.').replace('{reason}', row.causes?.length ? row.causes.map((entry) => `${number(entry.count)} ${reasonWords(row.reason, entry.cause)}`).join(' · ') : reasonWords(row.reason))));
    const free_ = isSize(volume?.free_bytes)
        ? fmt('Free now: {size}, read {when}.', { size: formatBytes(volume.free_bytes), when: whenLabel(volume.observed_at, now) }) : fmt('Free space could not be read.');
    return {
        problems: [
            run.error ? fmt('Freeing stopped: {reason}.', { reason: String(run.error).replace(/\.$/, '') }) : '',
            failedSentence(run),
            stuckSentence(run),
        ].filter(Boolean),
        lead,
        rest: [
            isSize(run.removed_bytes) ? fmt('Their counted size was {size}.', { size: formatBytes(run.removed_bytes) }) : '',
            free_,
            deletingSentence(run),
            ...kept,
            notStartedSentence(run),
            fmt('The next count will show the new total.'),
        ].filter(Boolean),
    };
}

/** Every sentence of the result in reading order. */
export const doneSentences = (free, volume = null, now = new Date()) => {
    const { problems, lead, rest } = resultSentences(free, volume, now);
    return [...problems, lead, ...rest].filter(Boolean);
};

function outcomesHtml(free, ui) {
    const { run } = free;
    if (!run?.outcomes_ready) return '<p class="storage-meta">Each item’s result is not available for this run; the counts above stand.</p>';
    const list = ui.outcomes || EMPTY_LIST();
    const more = list.total !== null && list.items.length < list.total;
    return disclosure('each', fmt('Each item ({total})', { total: number(list.total ?? run.selected ?? 0) }), `
        ${list.items.length ? `<ul class="storage-free-groups storage-free-list is-single">${list.items.map(outcomeRow).join('')}</ul>` : ''}
        ${listStatus(list, { reading: fmt('Reading each item’s result…'), problem: fmt('Each item’s result could not be read in full.'), retry: 'data-free-retry-outcomes' })}
        ${more && !list.loading && !list.error ? button(fmt('Show {n} more', { n: number(Math.min(list.total - list.items.length, 100)) }), 'data-free-more-outcomes', 'btn-ghost', 'storage.free.more.outcomes') : ''}`);
}

function resultHtml(free, ui, volume, now) {
    const { problems, lead, rest } = resultSentences(free, volume, now);
    return `${problems.length ? `<p class="ui-status storage-status" data-tone="warn" data-i18n-fmt="storage.free.problems">${text(problems.join(' '))}</p>` : ''}
        <p class="storage-tray-text" data-i18n-fmt="storage.free.done">${text(lead)}</p>
        <p class="storage-meta" data-i18n-fmt="storage.free.done.rest">${text(rest.join(' '))}</p>
        ${outcomesHtml(free, ui)}
        <div class="storage-actions">${button('Done', 'data-free-done')}</div>`;
}

function headHtml(title, sum = '') {
    return `<div class="storage-tray-head"><h3 class="storage-card-title">${text(title)}</h3>${sum
        ? `<span class="storage-tray-sum" data-i18n-fmt="storage.free.sum">${text(sum)}</span>` : ''}</div>`;
}

/** The whole flow for one `free` envelope, empty at rest unless a notice stands. `ui` is the owner's state: `{dismissed,
 *  chosen: Set of ids, checked: Set of group ids, pool, excluded, manual, outcomes (the lists read), poolShown, saved,
 *  manualOpen, manualChosen (Set of copy ids, empty to begin with), manualShown, sending, outdated, postUnknown, confirming,
 *  notice}` (see `attemptSentences`). */
export function trayHtml(free, ui, days = 7, { volume = null, now = new Date() } = {}) {
    const state = trayState(free, ui?.dismissed);
    const held = { pool: EMPTY_LIST(), ...ui };
    if (state === 'ready') {
        if (held.manualOpen && manualOf(free)) return manualHtml(free, held);
        return held.saved && savedGroups(free).length ? savedHtml(free, held, days) : reviewHtml(free, held, now);
    }
    // Every other phase is the server's; a notice of the owner's last attempt stands before it.
    const notice = attemptHtml(held);
    if (state === 'listing') return notice + listingHtml(free);
    // Not knowing a size yet, or having none to offer, is a sentence beside the door, never a figure (U070).
    if (state === 'empty') return `${notice}<p class="storage-tray-text">${text(fmt('No files of finished work are waiting.'))}</p>`;
    if (state === 'running') return notice + runningHtml(free);
    if (state === 'done') return notice + resultHtml(free, held, volume, now);
    if (state === 'failed') {
        const reason = String(free.error || '').replace(/\.$/, '');  // the server's reason is a sentence
        return `${notice}<p class="ui-status storage-status" data-tone="warn" data-i18n-fmt="storage.free.failed">${text(fmt('Temporary space could not be listed: {reason}. Nothing was removed.', { reason }))}</p>
            <div class="storage-actions">${button('Try again', 'data-free-preview')}</div>`;
    }
    return `${notice}<p class="storage-tray-text">${text(fmt(free.run ? 'Not checked since the last removal.' : 'See what finished work can be removed.'))}</p>`;
}

/** The summary card's door stands alone at rest and beside an empty answer; every other phase holds it back,
 *  so no second start competes with the review, the work or the result before Done. */
export const doorOffered = (state) => state === 'rest' || state === 'empty';

/** The sentence for a run the server refused (`started: false`). */
export const notStarted = (reason) => fmt('Nothing was removed. {reason}', { reason: fmt(NOT_STARTED[reason] || 'Try again.') });

/** What the confirm dialog shows for the checked saved groups: the body and one detail row per group. */
export function confirmCopy(free, checked) {
    const groups = groupsOf(free).filter((group) => checked.has(group.id));
    const total = freeSum(free, checked);
    const size = total === null ? fmt('size unknown') : formatBytes(total);
    const removes = sumComplete(free, checked) ? fmt('This removes {size} for good.', { size })
        : fmt('This removes the checked folders for good; some of their sizes could not be read.');
    const saved = groups.filter((group) => words(group).saved).map((group) => fmt(words(group).label));
    const kept = saved.length ? fmt('Saved work goes too: {groups}.', { groups: saved.join('; ') })
        : fmt('Results, messages and the exact model history are kept. Continuing one of these tasks later starts without its scratch files.');
    return {
        title: fmt('Free temporary space?'),
        body: `${removes} ${kept}`,
        details: {
            summary: fmt('Show details'),
            rows: groups.map((group) => ({ label: fmt(words(group).label), value: `${sizeWords(group)} · ${unitOf(group, group.count)}` })),
        },
        confirmLabel: goLabel(free, checked),
        danger: true,
    };
}
