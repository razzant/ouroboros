/**
 * T4 — the wiring between T1's thread menu and T3's server escapes.
 *
 * T1 built the menu before T3's helpers existed; T3 built typed refusals with no
 * client. Everything covered here is a refusal the owner could SEE but could not
 * ANSWER, or a state the sidebar could not tell apart from another one. The seam
 * under test is deliberately injectable (`ask`, `ops`, `openMenu`) because the
 * decisions live in these functions and the DOM dialog does not run under
 * `node --test`: without the seam the only covered part of a branch/merge/delete
 * gesture would be the part that decides nothing.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    openArchivedThreadsMenu,
    runThreadAction,
    threadActionItemsHtml,
    withEvidence,
} from '../modules/project_threads.js';

const PROJECT = { id: 'alpha', name: 'Alpha' };
const BRANCHED = { where: 'worktree', branch: 'thread/alpha__3', path: '/w/alpha__3' };
const IN_FOLDER = { where: 'project_folder' };

/** An `ask` that answers a scripted queue and records what it was asked. */
function scriptedAsk(answers) {
    const seen = [];
    const queue = [...answers];
    const fn = async (options) => {
        seen.push(options);
        const next = queue.shift();
        return next === undefined ? false : next;
    };
    fn.seen = seen;
    return fn;
}

// ---------------------------------------------------------------------------
// The menu rows
// ---------------------------------------------------------------------------

test('a branched thread offers merge/changes/remove and refuses to re-branch', () => {
    const html = threadActionItemsHtml({ id: 3, lifecycle: 'active' }, BRANCHED);

    assert.match(html, /data-prm="merge_back"(?![^>]*disabled)/);
    assert.match(html, /data-prm="show_changes"(?![^>]*disabled)/);
    assert.match(html, /data-prm="remove_worktree"(?![^>]*disabled)/);
    // Disabled WITH its reason, never omitted: a missing row teaches nothing.
    assert.match(html, /data-prm="branch_off"[^>]*disabled/);
    assert.match(html, /already works in its own branch/);
});

test('a thread stuck in `deleting` offers exactly ONE action, and says why', () => {
    // `fail_thread_deletion`'s deliberate end state, which `thread_is_visible`
    // keeps on screen precisely so the owner can act on it. Every row disabled
    // left its committed work unreachable with a server restart as the only way
    // out; the server accepts the retry idempotently.
    const html = threadActionItemsHtml(
        { id: 3, lifecycle: 'deleting', delete_error: 'a task would not cancel' },
        IN_FOLDER,
    );

    const enabled = [...html.matchAll(/data-prm="([a-z_]+)"(?![^>]*disabled)/g)].map((m) => m[1]);
    assert.deepEqual(enabled, ['delete']);
    assert.match(html, /Retry delete/);
    // `thread.delete_error` is normalised onto every row and nothing read it: it
    // is the reason the row is OFFERED, not a reason it is disabled.
    assert.match(html, /title="a task would not cancel"/);
});

test('a checkout read that FAILED disables what depends on it rather than guessing', () => {
    const html = threadActionItemsHtml({ id: 3, lifecycle: 'active' }, IN_FOLDER, 'registry unreadable');

    for (const id of ['branch_off', 'merge_back', 'show_changes', 'remove_worktree']) {
        assert.match(html, new RegExp(`data-prm="${id}"[^>]*disabled`), id);
    }
    assert.match(html, /checkout could not be read: registry unreadable/);
    // Archive and delete do not depend on the checkout, so they stay reachable.
    assert.match(html, /data-prm="archive"(?![^>]*disabled)/);
});

test('withEvidence counts what it omits instead of trimming silently', () => {
    assert.equal(withEvidence('Blocked.', []), 'Blocked.');
    const many = Array.from({ length: 11 }, (_, i) => `f${i}`);
    const text = withEvidence('Blocked.', many);
    assert.match(text, /f0; f1/);
    assert.match(text, /\(and 3 more\)/);
});

// ---------------------------------------------------------------------------
// Merge back: the `checkout_dirty` retry (T3's escape had no producer)
// ---------------------------------------------------------------------------

test('merge back RENDERS the checkout_dirty retry and re-sends with the flag', async () => {
    const calls = [];
    const ops = {
        mergeBack: async (pid, tid, acknowledged) => {
            calls.push(acknowledged);
            return acknowledged
                ? { ok: true, merged: true, checkout_left_behind: ['M build.log'] }
                : {
                    ok: false,
                    reason: 'checkout_dirty',
                    message: 'The checkout still holds uncommitted changes.',
                    acknowledgeable: true,
                    dirty_files: [' M build.log'],
                };
        },
    };
    const ask = scriptedAsk([true, false]);

    const described = await runThreadAction('merge_back', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
    });

    assert.deepEqual(calls, [false, true]);
    // The confirmation NAMES what stays behind — acknowledging is not forgetting.
    assert.match(ask.seen[0].body, /build\.log/);
    assert.equal(ask.seen[0].confirmLabel, 'Merge anyway');
    assert.equal(described.tone, 'ok');
});

test('declining the retry sends NOTHING and does not replay the sentence', async () => {
    const calls = [];
    const ops = {
        mergeBack: async (pid, tid, acknowledged) => {
            calls.push(acknowledged);
            return {
                ok: false, reason: 'checkout_dirty', acknowledgeable: true,
                message: 'The checkout still holds uncommitted changes.',
            };
        },
    };
    const ask = scriptedAsk([false, false]);

    await runThreadAction('merge_back', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
    });

    assert.deepEqual(calls, [false]);
    // I14: the owner has just READ this exact sentence and answered no. Following
    // it with the identical text as an alert is the dialog answering itself — the
    // last thing they saw is the question, and nothing follows it.
    assert.equal(ask.seen.length, 1);
    assert.equal(ask.seen.at(-1).confirmLabel, 'Merge anyway');
    assert.notEqual(ask.seen.at(-1).alert, true);
});

test('merge_abort_failed gets its OWN banner, not a generic refusal', async () => {
    // The one state that blocks everything else in that folder: the merge could
    // neither finish nor be undone. A conflict left the folder byte-for-byte as it
    // was; this did not, and reading the same red sentence for both hides exactly
    // the difference the owner has to act on.
    const ops = {
        mergeBack: async () => ({
            ok: false,
            reason: 'merge_abort_failed',
            message: 'The merge could not be completed or undone.',
            folder_left_mid_merge: true,
            working_dir: '/w/alpha',
            abort_detail: 'git merge --abort exited 128',
        }),
    };
    const ask = scriptedAsk([]);

    await runThreadAction('merge_back', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
    });

    const shown = ask.seen.at(-1);
    assert.equal(shown.title, 'The project folder is mid-merge');
    assert.match(shown.body, /\/w\/alpha/);
    assert.match(shown.body, /git merge --abort/);
    assert.match(shown.body, /nothing else can run in it/);
});

// ---------------------------------------------------------------------------
// git_init_required: a refusal whose answer is a different call entirely
// ---------------------------------------------------------------------------

test('branch off renders the init_git OFFER and retries after the owner says yes', async () => {
    let initialised = 0;
    const bases = [];
    const ops = {
        bases: async () => {
            bases.push(initialised);
            return initialised
                ? { ok: true, current_branch: 'main', bases: [{ ref: 'main', label: 'main' }] }
                : {
                    ok: false,
                    reason: 'git_init_required',
                    message: 'This folder is not tracked by git.',
                    decision: {
                        decision: 'git_init_required',
                        offer: 'init_git',
                        enables: ['diff', 'rollback', 'branching'],
                        workspace_root: '/w/alpha',
                    },
                };
        },
        branchOff: async () => ({ ok: true, branch: 'thread/alpha__3' }),
    };
    const apiClient = { projectInitGit: async () => { initialised += 1; return { ok: true }; } };
    // yes to the offer, then "1" for the base, then the success alert.
    const ask = scriptedAsk([true, { confirmed: true, value: '1' }, false]);

    await runThreadAction('branch_off', PROJECT, { id: 3, name: 'T' }, {
        ops, apiClient, ask, onChanged: () => {},
    });

    assert.equal(initialised, 1);
    assert.deepEqual(bases, [0, 1], 'the base list is asked again once the folder is tracked');
    assert.match(ask.seen[0].title, /Start tracking/);
    assert.match(ask.seen[0].body, /diff, rollback, branching/);
});

test('a branch-off receipt discloses BOTH sensitive-file facts', async () => {
    // `tracked_sensitive` had no surface beyond the receipt object: files git
    // ALREADY tracks were snapshotted like any other tracked file, and an owner
    // told only about the skipped ones would believe their `.env` was left out
    // when the opposite happened.
    const ops = {
        bases: async () => ({ ok: true, current_branch: 'main', bases: [{ ref: 'main', label: 'main' }] }),
        branchOff: async () => ({
            ok: true,
            branch: 'thread/alpha__3',
            snapshot_commit: {
                created: true,
                skipped_sensitive: ['.env.local'],
                tracked_sensitive: ['config/secrets.yml'],
            },
        }),
    };
    const ask = scriptedAsk([{ confirmed: true, value: '1' }, false]);

    await runThreadAction('branch_off', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
    });

    const receipt = ask.seen.at(-1).body;
    assert.match(receipt, /still in your folder, still untracked\): \.env\.local/);
    assert.match(receipt, /Already tracked by git, so committed with everything else: config\/secrets\.yml/);
});

test('the A14 queue sentence rides the base list, in the server\'s own words', async () => {
    const ops = {
        bases: async () => ({
            ok: true,
            current_branch: 'main',
            bases: [{ ref: 'main', label: 'main' }],
            queue_notice: { queued: true, message: 'Another task is working in this folder right now.', remedy: 'branch_off' },
        }),
        branchOff: async () => ({ ok: true, branch: 'thread/alpha__3' }),
    };
    const ask = scriptedAsk([{ confirmed: false, value: '' }]);

    await runThreadAction('branch_off', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
    });

    assert.match(ask.seen[0].body, /Another task is working in this folder right now\./);
});

// ---------------------------------------------------------------------------
// Delete: two steps for rebuildable dirt, a wall for work
// ---------------------------------------------------------------------------

test('deleting over rebuildable files is TWO steps, and the copy names them', async () => {
    const calls = [];
    const ops = {
        delete: async (pid, tid, acknowledged) => {
            calls.push(acknowledged);
            return acknowledged ? { ok: true, worktree_removed: true } : {
                ok: false,
                reason: 'checkout_holds_rebuildable_files',
                acknowledgeable: true,
                message: 'This checkout holds 1 file git was told to ignore.',
                inspection: { dirty_files: ['!! node_modules/'] },
            };
        },
    };
    // yes to the delete prompt, yes to the acknowledgement.
    const ask = scriptedAsk([true, true]);

    const described = await runThreadAction('delete', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
    });

    assert.deepEqual(calls, [false, true]);
    assert.match(ask.seen[1].body, /git was told to ignore/);
    assert.match(ask.seen[1].body, /node_modules/);
    assert.equal(described.tone, 'ok');
});

test('deleting over real work is a WALL, with no acknowledgement offered', async () => {
    const calls = [];
    const ops = {
        delete: async (pid, tid, acknowledged) => {
            calls.push(acknowledged);
            return {
                ok: false,
                reason: 'checkout_holds_work',
                // Deliberately NOT acknowledgeable: the way past this is a merge
                // back or an acknowledged removal, not a louder yes here.
                acknowledgeable: false,
                message: 'This checkout holds 2 commits that exist nowhere else.',
                inspection: { dirty_files: [' M src/app.py'] },
            };
        },
    };
    const ask = scriptedAsk([true, true]);

    await runThreadAction('delete', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
    });

    assert.deepEqual(calls, [false], 'a wall is never re-sent with a flag');
    assert.match(ask.seen.at(-1).body, /commits that exist nowhere else/);
    assert.equal(ask.seen.at(-1).alert, true);
});

test('Retry delete says what went wrong before asking again', async () => {
    const ops = { delete: async () => ({ ok: true }) };
    const ask = scriptedAsk([true]);

    await runThreadAction('delete', PROJECT, {
        id: 3, name: 'T', lifecycle: 'deleting', delete_error: 'a task would not cancel',
    }, { ops, ask, onChanged: () => {} });

    assert.equal(ask.seen[0].title, 'Retry delete');
    assert.match(ask.seen[0].body, /a task would not cancel/);
});

test('the ordinary delete prompt states the two things it does NOT do', async () => {
    const ops = { delete: async () => ({ ok: true }) };
    const ask = scriptedAsk([true]);

    await runThreadAction('delete', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
    });

    // The id is reserved forever and the journal rows physically remain. Saying
    // "deleted" without either would be a promise of erasure nothing delivers.
    assert.match(ask.seen[0].body, /reserved forever/);
    assert.match(ask.seen[0].body, /journal rows physically remain/);
});

// ---------------------------------------------------------------------------
// Remove checkout (A10) — the inspection is shown BEFORE anything is removed
// ---------------------------------------------------------------------------

test('removing a checkout shows the inspection first and passes the acknowledgement', async () => {
    const calls = [];
    const ops = {
        removeWorktree: async (pid, tid, acknowledged) => {
            calls.push(acknowledged);
            return { ok: true, removed: true, branch: 'thread/alpha__3', branch_removed: false, branch_kept_reason: 'the checkout held unmerged work, so its branch keeps the commits' };
        },
    };
    const ask = scriptedAsk([true, false]);

    await runThreadAction('remove_worktree', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
        inspection: { dirty_files: [' M a.txt'], unmerged_commits: 2 },
    });

    assert.deepEqual(calls, [true]);
    assert.match(ask.seen[0].body, /2 commits the project folder never received/);
    // The surviving branch is exactly what the next branch-off refuses on, so the
    // success discloses it rather than leaving it to be met later as a bare error.
    assert.match(ask.seen.at(-1).body, /Branching off again will refuse/);
});

test('cancelling the removal prompt removes nothing', async () => {
    const calls = [];
    const ops = { removeWorktree: async (...args) => { calls.push(args); return { ok: true }; } };

    await runThreadAction('remove_worktree', PROJECT, { id: 3, name: 'T' }, {
        ops, ask: scriptedAsk([false]), onChanged: () => {},
        inspection: { dirty_files: [], unmerged_commits: 0 },
    });

    assert.deepEqual(calls, []);
});

// ---------------------------------------------------------------------------
// Archived threads — the surface that makes `restore` reachable at all
// ---------------------------------------------------------------------------

test('the archived list asks for archived threads and restores the chosen one', async () => {
    const asked = [];
    const apiClient = {
        projectsList: async (includeArchived) => {
            asked.push(includeArchived);
            return {
                projects: [{
                    id: 'alpha',
                    threads: [
                        { id: 0, name: 'Alpha', lifecycle: 'active' },
                        { id: 4, name: 'Old spike', lifecycle: 'archived' },
                    ],
                }],
            };
        },
    };
    const restored = [];
    const ops = { restore: async (pid, tid) => { restored.push([pid, tid]); return { ok: true }; } };
    let menu = null;
    const rows = await openArchivedThreadsMenu(PROJECT, {
        apiClient, ops, onChanged: () => {},
        openMenu: (options) => { menu = options; },
        ask: scriptedAsk([]),
    });

    // `/api/state` hides archived threads, so this is the ONLY call that can see
    // them — without the flag, `restore` is unreachable by construction.
    assert.deepEqual(asked, [true]);
    assert.equal(rows.length, 1);
    assert.match(menu.itemsHtml, /Old spike/);
    await menu.onSelect('restore:4');
    assert.deepEqual(restored, [['alpha', '4']]);
});

test('a project with nothing archived says so rather than opening an empty menu', async () => {
    const apiClient = { projectsList: async () => ({ projects: [{ id: 'alpha', threads: [] }] }) };
    const ask = scriptedAsk([]);
    let opened = false;

    const rows = await openArchivedThreadsMenu(PROJECT, {
        apiClient, ask, openMenu: () => { opened = true; }, onChanged: () => {},
    });

    assert.deepEqual(rows, []);
    assert.equal(opened, false);
    assert.match(ask.seen[0].body, /No archived threads/);
});

test('an unreadable archived list is disclosed, never rendered as "none"', async () => {
    const apiClient = { projectsList: async () => { throw new Error('offline'); } };
    const ask = scriptedAsk([]);

    await openArchivedThreadsMenu(PROJECT, {
        apiClient, ask, openMenu: () => {}, onChanged: () => {},
    });

    assert.match(ask.seen[0].body, /could not be read: .*offline/);
});

// ---------------------------------------------------------------------------
// The sidebar row: three lifecycles, painted apart
// ---------------------------------------------------------------------------

test('a deleting thread is painted apart from an archived one, and from neither', async () => {
    const { threadRowPresentation } = await import('../modules/project_threads.js');

    const active = threadRowPresentation({ id: 3, name: 'Spike', lifecycle: 'active' });
    assert.equal(active.modifier, '');
    assert.equal(active.state, '');
    assert.equal(active.title, 'Spike');
    assert.equal(active.draggable, true);
    assert.equal(active.showsUnread, true);

    const deleting = threadRowPresentation({
        id: 3, name: 'Spike', lifecycle: 'deleting', delete_error: 'a task would not cancel',
    });
    assert.equal(deleting.modifier, ' is-deleting');
    assert.equal(deleting.state, 'Deleting…');
    assert.match(deleting.title, /a task would not cancel/);
    // Not draggable and no unread dot: an order about to vanish is not an order
    // worth persisting, and a dot invites a click into a room being torn down.
    assert.equal(deleting.draggable, false);
    assert.equal(deleting.showsUnread, false);

    const archived = threadRowPresentation({ id: 3, name: 'Spike', lifecycle: 'archived' });
    assert.equal(archived.modifier, ' is-archived');
    assert.equal(archived.state, 'Archived');
    // An archived thread only reaches the sidebar while a task is LIVE in it, so
    // it stays draggable and still shows what that task is producing (X10).
    assert.equal(archived.draggable, true);
    assert.equal(archived.showsUnread, true);
    assert.match(archived.title, /a task is running in it/);

    assert.notEqual(deleting.modifier, archived.modifier);
    assert.notEqual(deleting.state, archived.state);
});

test('an unreadable checkout is not rendered as "works in the project folder"', async () => {
    const { readThreadCheckout } = await import('../modules/project_threads.js');

    // A typed refusal now arrives as a VALUE, so `ok === false` has to be read
    // here too: a location we did not learn must never be painted as one we did.
    const refused = await readThreadCheckout('alpha', 9, {
        ops: { inspectWorktree: async () => ({ ok: false, reason: 'unknown_thread', message: 'unknown thread: 9' }) },
    });
    assert.equal(refused.locationError, 'unknown thread: 9');

    const thrown = await readThreadCheckout('alpha', 3, {
        ops: { inspectWorktree: async () => { throw new TypeError('Failed to fetch'); } },
    });
    assert.match(thrown.locationError, /Failed to fetch/);

    const ok = await readThreadCheckout('alpha', 3, {
        ops: { inspectWorktree: async () => ({ ok: true, location: { where: 'worktree', branch: 'thread/x' }, inspection: { dirty_files: [] } }) },
    });
    assert.equal(ok.locationError, '');
    assert.equal(ok.location.where, 'worktree');
});

test('archiving a thread with a live task SAYS why it is still on screen', async () => {
    // X10 decided archive does not refuse while a task runs and the thread stays
    // visible until it is terminal. The server answers `visible_until_terminal`
    // "so the surface can say which of the two just happened" — and nothing said
    // it, so the owner archived a thread, watched it stay put, and could not tell
    // a deliberate rule from an instruction that had not landed.
    const ops = { archive: async () => ({ ok: true, visible_until_terminal: true }) };
    const ask = scriptedAsk([]);

    await runThreadAction('archive', PROJECT, { id: 3, name: 'Spike' }, {
        ops, ask, onChanged: () => {},
    });

    assert.equal(ask.seen.length, 1);
    assert.match(ask.seen[0].body, /stays on screen until the task running in it finishes/);

    // ...and an ordinary archive stays silent: it did exactly what it looks like.
    const quiet = scriptedAsk([]);
    await runThreadAction('archive', PROJECT, { id: 3, name: 'Spike' }, {
        ops: { archive: async () => ({ ok: true }) }, ask: quiet, onChanged: () => {},
    });
    assert.equal(quiet.seen.length, 0);
});

test('the project row menu takes thread #0\'s CHECKOUT rows only', () => {
    // The project row is thread #0's row, and its menu already carries the
    // project's own Rename…/Delete project…. Emitting thread #0's disabled
    // Archive/Delete… beside them would put two delete-shaped rows in one menu
    // meaning entirely different things.
    const html = threadActionItemsHtml(
        { id: 0, lifecycle: 'active' }, IN_FOLDER, '',
        ['branch_off', 'merge_back', 'show_changes', 'remove_worktree'],
    );

    assert.match(html, /data-prm="branch_off"(?![^>]*disabled)/);
    assert.match(html, /data-prm="merge_back"[^>]*disabled/);
    assert.equal(/data-prm="archive"/.test(html), false);
    assert.equal(/data-prm="delete"/.test(html), false);
    // ...and without the filter, thread #0 still gets those rows disabled with
    // the reason, which is what every OTHER surface should show.
    const all = threadActionItemsHtml({ id: 0, lifecycle: 'active' }, IN_FOLDER);
    assert.match(all, /data-prm="archive"[^>]*disabled/);
    assert.match(all, /the project itself/);
});

// ---------------------------------------------------------------------------
// P2 — restoring an archived thread cannot become an unhandled rejection
// ---------------------------------------------------------------------------

/** The archived-menu harness: capture `onSelect`, watch what the owner is told. */
function archivedHarness(restoreImpl) {
    let onSelect = null;
    const asked = [];
    const changed = [];
    const apiClient = {
        projectsList: async () => ({
            projects: [{
                id: 'alpha',
                threads: [{ id: 4, name: 'Old idea', lifecycle: 'archived' }],
            }],
        }),
    };
    return {
        asked,
        changed,
        run: async () => {
            await openArchivedThreadsMenu(PROJECT, {
                apiClient,
                anchorEl: null,
                onChanged: (x) => changed.push(x),
                openMenu: (options) => { onSelect = options.onSelect; },
                ask: async (options) => { asked.push(options); return false; },
                ops: { restore: restoreImpl },
            });
            return onSelect;
        },
    };
}

test('a 500 / HTML error page during restore is announced, not thrown past the menu', async () => {
    // The FINDING is narrower than filed and the narrower half is real: a TYPED
    // refusal was already handled (`typedAnswer` unwraps a 409 envelope to a VALUE,
    // so describeOutcome + announce fired). A non-envelope failure re-throws, and
    // this was the one unguarded `await ops.*` in the module — so the rejection
    // escaped `onSelect` into `project_create.js`'s async click listener, which has
    // no try/catch: an unhandled rejection, no owner-facing error, no refresh, and a
    // stale archived row still clickable.
    const boom = Object.assign(new Error('Internal Server Error'), {
        status: 500, body: '<html>500</html>',
    });
    const harness = archivedHarness(async () => { throw boom; });
    const onSelect = await harness.run();

    await onSelect('restore:4');  // must NOT reject

    assert.equal(harness.asked.length, 1, 'the owner is told');
    assert.match(harness.asked[0].body, /restore did not finish/i);
    assert.match(harness.asked[0].body, /Internal Server Error/);
    assert.equal(harness.changed.length, 1, 'and the sidebar is re-read');
    assert.equal(harness.changed[0].authoritative, true);
});

test('a TYPED restore refusal keeps announcing exactly as it did', async () => {
    const harness = archivedHarness(async () => ({
        ok: false, reason: 'thread_busy', message: 'A task is still running.',
    }));
    const onSelect = await harness.run();

    await onSelect('restore:4');

    assert.equal(harness.changed.length, 1);
    assert.equal(harness.asked.length, 1);
    assert.match(harness.asked[0].body, /A task is still running/);
});

test('a successful restore stays silent and refreshes once', async () => {
    const calls = [];
    const harness = archivedHarness(async (pid, tid) => {
        calls.push([pid, tid]);
        return { ok: true };
    });
    const onSelect = await harness.run();

    await onSelect('restore:4');

    // The id the route receives is unchanged by the rerouting through
    // `runThreadAction`: the string the menu carried.
    assert.deepEqual(calls, [['alpha', '4']]);
    assert.equal(harness.changed.length, 1);
    assert.equal(harness.asked.length, 0, 'success is not announced');
});
