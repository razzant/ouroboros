/**
 * Thread branch / merge / checkout / lifecycle actions, as small pure helpers.
 *
 * This module owns the DECISIONS behind those owner gestures — what a menu may
 * offer for a given thread, what each answer means, and the exact words shown
 * when something refuses. It deliberately renders no DOM: the thread menus live
 * in `project_threads.js` (phase T1) and this file is the seam they call, so the
 * two phases could be built in parallel and joined without either one guessing
 * at the other's internals.
 *
 * Three rules the helpers exist to keep:
 *
 *   1. A thread's LOCATION is derived, never stored (A7). Every helper reads it
 *      from `location.where` — no caller keeps a boolean about it.
 *   2. A refusal is SHOWN, never smoothed over (A9/A10). `describeOutcome` turns
 *      every typed reason into the sentence the server already wrote;
 *      `conflicts`/`dirty_files`/`inspection` ride along as evidence rather than
 *      being flattened into "something went wrong", and so do the two fields that
 *      make a refusal ANSWERABLE — `acknowledgeable` (the flag that re-sends the
 *      call) and `decision` (T2's typed `git_init_required` offer, whose yes is
 *      `apiClient.projectInitGit`). A refusal a menu cannot answer is a dead end
 *      wearing a sentence.
 *   3. Removing a checkout is always a separate, confirmed act (A10). There is
 *      no path here that removes one as a side effect of anything else, and the
 *      acknowledgement is a distinct second call the owner has to reach.
 */

import { apiClient } from './api_client.js';

/** A thread works in the project folder unless a worktree exists for it (A7). */
export function isBranched(location) {
    return String(location?.where || '') === 'worktree';
}

/**
 * Which actions a menu may offer for one thread, and which are disabled.
 *
 * Returned in menu order with a stable `id` per action, an owner-facing `label`,
 * and — when it cannot be run — a `disabledReason` sentence rather than a silent
 * omission. A missing item teaches nothing; a greyed one with a reason teaches
 * what to do first.
 *
 * `deleting` and `tombstoned` are NOT the same state, and collapsing them into
 * one `terminal` flag left the owner with no way out of the first. A thread stuck
 * in `deleting` — the deliberate end state of `fail_thread_deletion`, which
 * `thread_is_visible` keeps on screen precisely so the owner can act on it — had
 * EVERY row disabled, while the server accepts a delete retry idempotently and
 * the checkout-removal route still works. Its committed work was unreachable both
 * ways and the only escape was a server restart. So `deleting` disables
 * everything EXCEPT `delete`, relabelled "Retry delete" and carrying
 * `thread.delete_error` as the reason it is on offer — that field is normalised
 * onto every row and nothing read it. `tombstoned` really is terminal.
 *
 * @param {{id?: number, lifecycle?: string, delete_error?: string}} thread
 * @param {{where?: string, branch?: string}} location
 */
export function threadActions(thread, location) {
    const isMain = Number(thread?.id ?? 0) === 0;
    const lifecycle = String(thread?.lifecycle || 'active');
    const branched = isBranched(location);
    const deleting = lifecycle === 'deleting';
    const fenced = deleting || lifecycle === 'tombstoned';
    // Thread #0 mirrors the PROJECT's lifecycle, so a deleting project makes it
    // read `deleting` too — and the delete route refuses thread #0 by name.
    // Offering it a retry would be a button that can only ever fail.
    const offerRetry = deleting && !isMain;
    // Thread #0 IS the project. Offering it a lifecycle of its own would promise
    // an operation the server refuses by name, so it is disabled with the reason.
    const projectItself = isMain ? 'This thread is the project itself.' : '';
    // A thread on its way out overrides every other explanation: "already works
    // in its own branch" is true but useless when the thread is being deleted.
    const goingAway = deleting ? 'This thread is being deleted.' : 'This thread is deleted.';
    const row = (id, label, allowed, reason) => ({
        id,
        label,
        available: allowed && !fenced,
        disabledReason: fenced ? goingAway : (allowed ? '' : reason),
    });
    const noCheckout = 'This thread works in the project folder.';
    const deleteError = String(thread?.delete_error || '').trim();
    const retry = {
        id: 'delete',
        label: 'Retry delete',
        available: true,
        // Not a disabled reason — the reason this row is OFFERED. A deletion that
        // did not quiesce stays fenced on purpose, and the owner has to be able to
        // see why before deciding to ask again.
        disabledReason: '',
        reason: deleteError || 'This thread\'s deletion did not finish.',
    };
    return [
        row('branch_off', 'Branch off…', !branched, 'This thread already works in its own branch.'),
        row('merge_back', 'Merge back', branched, noCheckout),
        row('show_changes', 'Show changes', branched, noCheckout),
        row('remove_worktree', 'Remove checkout…', branched, 'This thread has no checkout.'),
        row(
            lifecycle === 'archived' ? 'restore' : 'archive',
            lifecycle === 'archived' ? 'Restore' : 'Archive',
            !isMain,
            projectItself,
        ),
        offerRetry ? retry : row('delete', 'Delete…', !isMain, projectItself),
    ];
}

/**
 * The owner-facing reading of any branch/merge/remove/lifecycle answer.
 *
 * `tone` drives a CSS token only. `text` is the server's own sentence wherever it
 * wrote one — the copy for a refusal belongs beside the rule that produced it,
 * and re-authoring it here is how two surfaces end up explaining the same
 * refusal differently. `evidence` is the list the owner needs to act: the
 * conflicting paths, the files blocking a merge, what a removal would destroy.
 *
 * Two fields ride the refusal that a menu cannot do without, and both were being
 * dropped here:
 *
 *   - `acknowledgeable` — the server saying this refusal HAS an owner-answerable
 *     flag. Without it nothing could render the second call, so `checkout_dirty`
 *     read as a dead end when it is a question. DELETE uses the same field for
 *     `checkout_holds_rebuildable_files` (a checkout whose only contents are
 *     ignored or untracked files); its `checkout_holds_work` sibling deliberately
 *     does not set it, because that one is a wall with a named way around it.
 *   - `decision` — T2's typed `git_init_required` OFFER. `apiClient.projectInitGit`
 *     exists and is the yes to it; a menu that never sees the object cannot
 *     offer the yes.
 */
export function describeOutcome(outcome) {
    const reason = String(outcome?.reason || '');
    const server = String(outcome?.message || '').trim();
    if (outcome?.ok) {
        return { tone: 'ok', text: server || successText(outcome), evidence: [] };
    }
    const evidence = []
        .concat(Array.isArray(outcome?.conflicts) ? outcome.conflicts : [])
        .concat(Array.isArray(outcome?.dirty_files) ? outcome.dirty_files : [])
        .concat(Array.isArray(outcome?.inspection?.dirty_files) ? outcome.inspection.dirty_files : []);
    return {
        tone: reason === 'merge_conflict' ? 'conflict' : 'blocked',
        // A refusal with no sentence at all is still named, never rendered blank.
        text: server || `This could not be done (${reason || 'unknown reason'}).`,
        evidence,
        reason,
        // A refusal the owner can answer, and the object that answers it.
        acknowledgeable: Boolean(outcome?.acknowledgeable),
        decision: (outcome && typeof outcome.decision === 'object') ? outcome.decision : null,
    };
}

/** The one sentence for a success the server did not narrate itself. */
export function successText(outcome) {
    if (outcome?.merged === false) return 'Nothing new to merge — the folder already has this work.';
    if (outcome?.merged) {
        // A10 is stated at the moment it matters: the checkout is still there.
        const base = 'Merged into the project folder. The thread keeps its checkout until you remove it.';
        const behind = Array.isArray(outcome.checkout_left_behind)
            ? outcome.checkout_left_behind.filter(Boolean)
            : [];
        // Acknowledging that work stays behind is not the same as forgetting it
        // did, so the SUCCESS says it again rather than leaving the owner to
        // rediscover it in a folder they have stopped looking at.
        if (!behind.length) return base;
        // The listing is bounded; how much stayed behind is not. Counting the
        // slice here would have disagreed with the refusal that preceded it.
        const declared = Number(outcome.dirty_files_total);
        const n = Number.isFinite(declared) ? Math.max(behind.length, declared) : behind.length;
        return `${base} ${n} uncommitted change${n === 1 ? '' : 's'} stayed in the checkout — a merge brings commits only.`;
    }
    if (outcome?.removed) {
        // T3R-5's disclosure has to REACH the owner: a branch that survived is
        // exactly what the next branch-off refuses on, and meeting that later as
        // a bare "branch already exists" is what this sentence prevents.
        const branch = String(outcome.branch || '').trim();
        if (outcome.branch_removed) {
            return branch
                ? `Checkout removed, and its branch ${branch} with it — this thread can branch off again.`
                : 'Checkout removed.';
        }
        const why = String(outcome.branch_kept_reason || '').trim();
        if (!why) return 'Checkout removed.';
        return `Checkout removed. Its branch ${branch || ''} was kept — ${why}. Branching off again will refuse until it is gone.`
            .replace(/\s{2,}/g, ' ');
    }
    if (outcome?.branch) return `Branched off into ${outcome.branch}.`;
    return 'Done.';
}

/**
 * What a snapshot base actually did, for the receipt after a branch-off.
 *
 * Returns '' when no snapshot was involved. TWO disclosures, and they are
 * opposite facts that used to be told as one sentence:
 *
 *   - `skipped_sensitive` — credential-shaped files git did NOT already track.
 *     They were kept out of the commit and are still in the folder, still
 *     untracked. An owner who is not told will believe their `.env` came along.
 *   - `tracked_sensitive` — credential-shaped files git ALREADY tracks. They were
 *     snapshotted like every other tracked file, because unstaging a tracked path
 *     stages a DELETION on the owner's branch and protects nothing: its contents
 *     are in history already. "Still untracked" was only ever true of these
 *     because the snapshot had just untracked them, which is the bug it described.
 */
export function snapshotReceipt(outcome) {
    const snapshot = outcome?.snapshot_commit;
    if (!snapshot) return '';
    const list = (key) => (Array.isArray(snapshot[key]) ? snapshot[key].filter(Boolean) : []);
    const skipped = list('skipped_sensitive');
    const tracked = list('tracked_sensitive');
    const parts = [
        snapshot.created
            ? 'Your uncommitted changes were committed first, so the branch starts from exactly what was there.'
            : 'The folder had no uncommitted changes, so nothing new was committed.',
    ];
    if (skipped.length) {
        parts.push(`Left out of that commit (still in your folder, still untracked): ${skipped.join(', ')}.`);
    }
    if (tracked.length) {
        parts.push(`Already tracked by git, so committed with everything else: ${tracked.join(', ')}.`);
    }
    return parts.join(' ');
}

/**
 * The confirmation an owner must see BEFORE a checkout is removed (A10).
 *
 * Returns `{needsAcknowledgement, text, evidence}`. When the inspection could
 * not be read, this treats it as unsafe and says so — "cannot tell" must never
 * be rendered as "nothing to lose".
 *
 * `dirty_files` is a BOUNDED listing and `dirty_files_total` is how many there
 * are; this prompt is the last thing between the owner and an irreversible
 * removal, so it states the total and discloses that the list is shorter.
 * `dirty_files.length` was the same false magnitude the server refusal used to
 * state — 800 modified files rendered as 200.
 */
export function removalPrompt(inspection) {
    const dirty = Array.isArray(inspection?.dirty_files) ? inspection.dirty_files.filter(Boolean) : [];
    const declared = Number(inspection?.dirty_files_total);
    const total = Number.isFinite(declared) ? Math.max(dirty.length, declared) : dirty.length;
    const commits = Number(inspection?.unmerged_commits || 0);
    const error = String(inspection?.error || '').trim();
    if (error) {
        return {
            needsAcknowledgement: true,
            text: 'This checkout could not be read, so what removing it would destroy is unknown.',
            evidence: [error],
        };
    }
    if (!total && !commits) {
        return {
            needsAcknowledgement: false,
            text: 'This checkout has no unmerged work. Removing it deletes only the folder.',
            evidence: [],
        };
    }
    const parts = [];
    if (commits) parts.push(`${commits} commit${commits === 1 ? '' : 's'} the project folder never received`);
    if (total) parts.push(`${total} uncommitted file change${total === 1 ? '' : 's'}`);
    let omitted = '';
    if (total > dirty.length) {
        omitted = dirty.length
            ? ` Only the first ${dirty.length} of those files are listed here.`
            : ' None of those files are listed here.';
    }
    return {
        needsAcknowledgement: true,
        text: `Removing this checkout deletes ${parts.join(' and ')}.${omitted}`,
        evidence: dirty,
    };
}

/**
 * The honest queue sentence (A14), or '' when nothing would wait.
 *
 * The copy itself is the SERVER's — one sentence, one place — so the UI can
 * never soften "queued behind" into "rejected" or the reverse.
 */
export function queueNoticeText(notice) {
    return notice?.queued ? String(notice.message || '') : '';
}

/** Does the queue notice offer branching as the way to run in parallel? */
export function queueNoticeOffersBranching(notice) {
    return Boolean(notice?.queued) && String(notice?.remedy || '') === 'branch_off';
}

// ---------------------------------------------------------------------------
// Thin call wrappers. They exist so the menu never hand-rolls a fetch and never
// has to remember which routes take a body.
// ---------------------------------------------------------------------------

/**
 * Open a branched thread's checkout in the Changes screen (A13).
 *
 * Dispatches the same kind of window event the task inspector already uses to
 * open a task diff, so a thread menu needs no reference to the Changes
 * controller. Returns false when the thread has no checkout to show, which is a
 * STATE the caller should explain, not an error to swallow.
 */
export function openThreadChanges({ projectId, threadId, label = '', branch = '', filePath = '' } = {}) {
    if (!projectId || threadId === undefined || threadId === null || threadId === '') return false;
    window.dispatchEvent(new CustomEvent('ouro:open-thread-changes', {
        detail: { projectId: String(projectId), threadId: String(threadId), label, branch, filePath },
    }));
    return true;
}

/**
 * A typed refusal is DATA, not an exception.
 *
 * Every thread route answers a refusal with the shared envelope
 * `{ok:false, reason, message, acknowledgeable?, decision?, inspection?}` under a
 * 400/404/409 — and `api_client.fetchJson` turns EVERY non-2xx into a throw
 * carrying that envelope on `error.body`. Unwrapped, the whole design above is
 * unreachable in a browser: `describeOutcome` never sees the refusal,
 * `acknowledgeable` never renders its second call, T2's `decision` never renders
 * its offer, and a menu can only report that "something went wrong". Neither
 * stream could see this — T3 wrote the envelopes with no menu calling them, and
 * T1's menu only ever called routes that answer 200 or genuinely fail.
 *
 * A body that is NOT one of these envelopes (a transport failure, a 500, an HTML
 * error page) still throws: those are not answers the owner can act on, and
 * swallowing them into `{ok:false}` would dress an outage up as a refusal.
 */
async function typedAnswer(call) {
    try {
        return await call();
    } catch (error) {
        const body = error?.body;
        const refusal = body && typeof body === 'object'
            && (typeof body.reason === 'string' || body.ok === false);
        if (refusal) return body;
        throw error;
    }
}

export const threadOps = {
    bases: (projectId, threadId) => typedAnswer(() => apiClient.threadBranchBases(projectId, threadId)),
    branchOff: (projectId, threadId, baseRef) => typedAnswer(
        () => apiClient.threadBranchOff(projectId, threadId, baseRef),
    ),
    /**
     * Merge a thread's branch home. `acknowledged` is the owner's answer to the
     * `checkout_dirty` refusal — the SAME shape `removeWorktree` already has, and
     * deliberately a separate argument so no caller passes it by accident. It had
     * no producer at all, which made the server's only escape from that refusal
     * unreachable: one stray `build.log` in a checkout and merge-back was over.
     */
    mergeBack: (projectId, threadId, acknowledged = false) => typedAnswer(
        () => apiClient.threadMergeBack(projectId, threadId, acknowledged),
    ),
    inspectWorktree: (projectId, threadId) => typedAnswer(
        () => apiClient.threadWorktree(projectId, threadId),
    ),
    /**
     * Remove a checkout. `acknowledged` is the owner's answer to `removalPrompt`
     * and is the ONLY way past unmerged work — deliberately a separate argument
     * so no caller can pass it by accident.
     */
    removeWorktree: (projectId, threadId, acknowledged = false) => typedAnswer(
        () => apiClient.threadWorktreeRemove(projectId, threadId, acknowledged),
    ),
    archive: (projectId, threadId) => typedAnswer(() => apiClient.threadArchive(projectId, threadId)),
    restore: (projectId, threadId) => typedAnswer(() => apiClient.threadRestore(projectId, threadId)),
    /**
     * Delete a thread, checkout and all. `acknowledged` is the owner's answer to
     * `checkout_holds_rebuildable_files` — a checkout whose only contents are
     * ignored or untracked files — and is the SAME separate-argument shape
     * `removeWorktree` and `mergeBack` use, so nobody passes it by accident.
     *
     * It is not an override: `checkout_holds_work` (unmerged commits, changes to
     * tracked files, an unreadable checkout) refuses whatever this says, and the
     * route out of that one is `removeWorktree` or a merge back.
     */
    delete: (projectId, threadId, acknowledged = false) => typedAnswer(
        () => apiClient.threadDelete(projectId, threadId, acknowledged),
    ),
};
