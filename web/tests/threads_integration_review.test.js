/**
 * Integration-review findings on the client half of the merged threads tree.
 *
 * Every one of these is a gesture the owner could reach and could not finish, or
 * a consent that was sent for something other than what they answered. The seams
 * (`ask`, `ops`, `openMenu`) are injected for the same reason the sibling file
 * gives: the decisions live in these functions and the DOM dialog does not run
 * under `node --test`.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    openArchivedThreadsMenu,
    runThreadAction,
} from '../modules/project_threads.js';
import { fetchJson } from '../modules/api_client.js';

const PROJECT = { id: 'alpha', name: 'Alpha' };

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

const BASES = {
    ok: true,
    current_branch: 'main',
    bases: [{ ref: 'main', kind: 'branch', label: 'main (current)' }],
    snapshot: { ref: '@snapshot', kind: 'snapshot', label: 'Exactly as it is now', dirty: false },
};

// ---------------------------------------------------------------------------
// I8 — branching thread #0 has to say what it is doing
// ---------------------------------------------------------------------------

test('branching the PROJECT\'s own chat says so before asking for a base', async () => {
    // The four checkout rows sit on the PROJECT row menu, next to Rename… and
    // Delete project…, and for thread #0 `thread.name` IS the project name. So the
    // owner read "Base for Alpha's own checkout" under a menu titled "Actions for
    // Alpha" and could not tell they were moving the project's CHAT rather than the
    // project. The behaviour is coherent — siblings keep the project folder and
    // this branch merges back into it — the row just could not describe it.
    const ops = { bases: async () => BASES, branchOff: async () => ({ ok: true, branch: 'thread/alpha__0' }) };
    const ask = scriptedAsk([{ confirmed: false, value: '' }]);

    await runThreadAction('branch_off', PROJECT, { id: 0, name: 'Alpha' }, {
        ops, ask, onChanged: () => {},
    });

    const body = ask.seen[0].body;
    assert.match(body, /This is Alpha's own chat\./);
    assert.match(body, /gives THAT chat its own copy of the folder/);
    assert.match(body, /other threads keep working in the folder itself/);
    assert.match(body, /merges back into it/);
});

test('an ordinary thread gets no such preface', async () => {
    const ops = { bases: async () => BASES, branchOff: async () => ({ ok: true }) };
    const ask = scriptedAsk([{ confirmed: false, value: '' }]);

    await runThreadAction('branch_off', PROJECT, { id: 3, name: 'Side quest' }, {
        ops, ask, onChanged: () => {},
    });

    assert.doesNotMatch(ask.seen[0].body, /own chat/);
    assert.match(ask.seen[0].body, /Base for Side quest's own checkout/);
});

// ---------------------------------------------------------------------------
// I9 — the removal's own refusal is answerable in the same gesture
// ---------------------------------------------------------------------------

test('a checkout that went dirty since the menu opened can still be confirmed', async () => {
    // The prompt is a PRE-FLIGHT read: the inspection captured when the menu was
    // OPENED. An agent writing a log in that window is the normal case, so the call
    // went out with `acknowledged: false`, the server refused, and the gesture had
    // no way to answer its own "or confirm you want it gone" — recovery was closing
    // and reopening the menu, which nothing disclosed.
    const calls = [];
    const ops = {
        removeWorktree: async (pid, tid, acknowledged) => {
            calls.push(acknowledged);
            return acknowledged ? { ok: true, removed: true, branch_removed: true, branch: 'thread/alpha__3' } : {
                ok: false,
                reason: 'unmerged_work',
                acknowledgeable: true,
                message: 'This checkout still holds 1 uncommitted file change. Removing it deletes that work. Merge it back first, or confirm you want it gone.',
                inspection: { dirty_files: [' M agent.log'] },
            };
        },
    };
    // yes to the pre-flight prompt (clean), then yes to the server's refusal.
    const ask = scriptedAsk([true, true, false]);

    const described = await runThreadAction('remove_worktree', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
        inspection: { dirty_files: [], unmerged_commits: 0 },   // the stale pre-flight
    });

    assert.deepEqual(calls, [false, true]);
    assert.match(ask.seen[0].body, /no unmerged work/);          // the stale read
    assert.match(ask.seen[1].body, /agent\.log/);                // the live refusal
    assert.equal(ask.seen[1].confirmLabel, 'Remove anyway');
    assert.equal(described.tone, 'ok');
});

test('declining the removal retry sends nothing and does not replay the sentence', async () => {
    const calls = [];
    const ops = {
        removeWorktree: async (pid, tid, acknowledged) => {
            calls.push(acknowledged);
            return {
                ok: false, reason: 'unmerged_work', acknowledgeable: true,
                message: 'This checkout still holds work.',
            };
        },
    };
    const ask = scriptedAsk([true, false]);

    await runThreadAction('remove_worktree', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {}, inspection: {},
    });

    assert.deepEqual(calls, [false]);
    assert.equal(ask.seen.length, 2);
    // The second dialog is the ACKNOWLEDGEMENT the server's refusal offered — not
    // an alert replaying it (I9 makes it exist at all; I14 keeps it the last word).
    assert.equal(ask.seen[1].confirmLabel, 'Remove anyway');
    assert.notEqual(ask.seen[1].alert, true);
});

// ---------------------------------------------------------------------------
// I10 — one consent must never be sent as a different one
// ---------------------------------------------------------------------------

test('answering the git_init OFFER does not send acknowledge_checkout_dirty', async () => {
    // `withAcknowledgement` could not tell WHICH refusal was answered: ANY answered
    // refusal was followed by `run(true)`. So "yes, start tracking this folder"
    // travelled as "yes, merge anyway even though the checkout holds uncommitted
    // work" — a sentence the owner never saw and never agreed to.
    const calls = [];
    let initialised = 0;
    const ops = {
        mergeBack: async (pid, tid, acknowledged) => {
            calls.push(acknowledged);
            return initialised
                ? { ok: true, merged: true }
                : {
                    ok: false,
                    reason: 'git_init_required',
                    message: 'This folder is not tracked by git.',
                    decision: {
                        decision: 'git_init_required', offer: 'init_git',
                        enables: ['diff', 'rollback', 'branching'], workspace_root: '/w/alpha',
                    },
                };
        },
    };
    const apiClient = { projectInitGit: async () => { initialised += 1; return { ok: true }; } };
    const ask = scriptedAsk([true, false]);

    await runThreadAction('merge_back', PROJECT, { id: 3, name: 'T' }, {
        ops, apiClient, ask, onChanged: () => {},
    });

    assert.equal(initialised, 1);
    assert.deepEqual(calls, [false, false], 'the retry after an init_git yes is the PLAIN call');
    assert.match(ask.seen[0].title, /Start tracking/);
});

test('acknowledging the refusal ITSELF still sends the flag', async () => {
    // The other half: narrowing the consent must not disarm the escape it exists
    // for.
    const calls = [];
    const ops = {
        mergeBack: async (pid, tid, acknowledged) => {
            calls.push(acknowledged);
            return acknowledged ? { ok: true, merged: true } : {
                ok: false, reason: 'checkout_dirty', acknowledgeable: true,
                message: 'The checkout holds uncommitted changes.',
            };
        },
    };
    const ask = scriptedAsk([true, false]);

    await runThreadAction('merge_back', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
    });

    assert.deepEqual(calls, [false, true]);
});

// ---------------------------------------------------------------------------
// I13 — an archived thread's row says why Restore is the only action
// ---------------------------------------------------------------------------

test('the archived-threads menu discloses that a thread must be restored to act on it', async () => {
    // `begin_thread_deletion` accepts an archived thread by design and
    // `_live_thread_refusal` permits branch/merge on one, but this menu emits only
    // `restore:` — so a branched-then-archived thread's checkout was two
    // undisclosed steps from any A10 surface.
    let itemsHtml = '';
    const apiClient = {
        projectsList: async () => ({
            projects: [{ id: 'alpha', threads: [{ id: 4, name: 'Old idea', lifecycle: 'archived' }] }],
        }),
    };
    const openMenu = (options) => { itemsHtml = options.itemsHtml; };

    const rows = await openArchivedThreadsMenu(PROJECT, {
        apiClient, anchorEl: null, onChanged: () => {}, openMenu, ask: async () => false,
    });

    assert.equal(rows.length, 1);
    assert.match(itemsHtml, /data-prm="restore:4"/);
    // The guarantee is that the row DISCLOSES restore-first — an archived thread
    // otherwise looks as though it simply has no merge back, no changes and no way
    // to reach its checkout. Asserted on the substance, not on one phrasing: the
    // first wording read "restore it to act on it", a tautology beside the word
    // "Restore" that explained nothing, caught by looking at the rendered menu.
    assert.match(itemsHtml, /reachable again once it is active/);
    assert.match(itemsHtml, /checkout/);
});

// ---------------------------------------------------------------------------
// I16 — a 2xx whose body cannot be parsed is not an answer
// ---------------------------------------------------------------------------

test('an unparseable 200 THROWS instead of answering {error}', async () => {
    // It returned `{error: 'non-json response (HTTP 200)'}` as a payload, so
    // `threadOps.bases` handed that object back, `listed.ok` was false, and
    // branch-off rendered an EMPTY base offer and asked the owner to type one.
    const original = globalThis.fetch;
    globalThis.fetch = async () => ({
        ok: true,
        status: 200,
        json: async () => { throw new SyntaxError('Unexpected token <'); },
    });
    try {
        await assert.rejects(
            () => fetchJson('/api/anything'),
            (error) => {
                assert.match(String(error.message), /non-json response \(HTTP 200\)/);
                assert.equal(error.status, 200);
                return true;
            },
        );
    } finally {
        globalThis.fetch = original;
    }
});

test('a parseable 200 is still returned unchanged', async () => {
    const original = globalThis.fetch;
    globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({ ok: true, bases: [] }) });
    try {
        assert.deepEqual(await fetchJson('/api/anything'), { ok: true, bases: [] });
    } finally {
        globalThis.fetch = original;
    }
});

test('branch off never renders an empty base offer for an unreadable answer', async () => {
    // The consequence I16 actually produced, pinned end to end through the REAL
    // `threadOps` -> `apiClient` -> `fetchJson` chain rather than a stubbed throw:
    // an unparseable 200 became `{error: …}`, `listed.ok` was undefined so the
    // refusal branch did not fire, `bases` was [] — and the owner was shown an
    // empty numbered list and asked to type a base.
    const original = globalThis.fetch;
    globalThis.fetch = async () => ({
        ok: true,
        status: 200,
        json: async () => { throw new SyntaxError('Unexpected token <'); },
    });
    const ask = scriptedAsk([{ confirmed: false, value: '' }]);
    try {
        await runThreadAction('branch_off', PROJECT, { id: 3, name: 'T' }, {
            ask, onChanged: () => {},
        });
    } finally {
        globalThis.fetch = original;
    }

    assert.equal(ask.seen.length, 1);
    assert.equal(ask.seen[0].title, 'That did not finish');
    assert.match(ask.seen[0].body, /non-json response/);
    assert.notEqual(ask.seen[0].input, true, 'the owner was asked to type a base for a list we never got');
});

// ---------------------------------------------------------------------------
// I14 — a declined question is not read back to the owner
// ---------------------------------------------------------------------------

test('declining the delete acknowledgement leaves the question as the last word', async () => {
    const calls = [];
    const ops = {
        delete: async (pid, tid, acknowledged) => {
            calls.push(acknowledged);
            return {
                ok: false, reason: 'checkout_holds_rebuildable_files', acknowledgeable: true,
                message: 'This checkout holds 1 file git was told to ignore.',
            };
        },
    };
    // yes to "Delete thread", then no to the acknowledgement.
    const ask = scriptedAsk([true, false]);

    await runThreadAction('delete', PROJECT, { id: 3, name: 'T' }, {
        ops, ask, onChanged: () => {},
    });

    assert.deepEqual(calls, [false]);
    assert.equal(ask.seen.length, 2);
    assert.notEqual(ask.seen.at(-1).alert, true);
});
