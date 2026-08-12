import assert from 'node:assert/strict';
import test from 'node:test';

import {
    SOURCE_TASK,
    SOURCE_THREAD,
    diffBanners,
    diffSummaryMeta,
    requestEditsParts,
    requestEditsThreadPrefix,
    threadCheckoutRefusal,
} from '../modules/changes.js';
import { makeTextPart } from '../modules/composer_parts.js';

const NOT_BRANCHED = {
    status: 'blocked',
    source: 'thread_checkout',
    blockers: ['thread_not_branched'],
    patch: '',
};

test('the two source modes are named constants, not string literals at call sites', () => {
    assert.equal(SOURCE_TASK, 'task');
    assert.equal(SOURCE_THREAD, 'thread');
});

test('a thread that works in the project folder is a STATE, never an alarm', () => {
    const refusal = threadCheckoutRefusal(NOT_BRANCHED);

    assert.equal(refusal.reason, 'thread_not_branched');
    assert.match(refusal.text, /works in the project folder/);
    // Not "no trustworthy diff": nothing failed, and the sentence says where the
    // thread's work actually lives instead of implying something is broken.
    const banners = diffBanners(NOT_BRANCHED);
    assert.equal(banners.length, 1);
    assert.equal(banners[0].tone, 'neutral');
    assert.doesNotMatch(banners[0].text, /trustworthy/);
    assert.equal(banners[0].detail, undefined);
    assert.equal(diffSummaryMeta(NOT_BRANCHED, null), 'works in the project folder');
});

test('a checkout that vanished says the branch still holds the commits', () => {
    const diff = { status: 'blocked', source: 'thread_checkout', blockers: ['checkout_missing'] };

    const refusal = threadCheckoutRefusal(diff);

    assert.equal(refusal.reason, 'checkout_missing');
    assert.match(refusal.text, /branch still holds/);
    assert.equal(diffBanners(diff)[0].tone, 'neutral');
});

test('a REAL thread-checkout failure is still disclosed as one', () => {
    const diff = {
        status: 'blocked',
        source: 'thread_checkout',
        blockers: ['baseline_diff_failed'],
    };

    assert.equal(threadCheckoutRefusal(diff), null);
    const banner = diffBanners(diff)[0];
    assert.equal(banner.tone, 'blocked');
    assert.equal(banner.detail, 'baseline_diff_failed');
});

test('a task diff keeps its own wording — the thread copy never leaks into it', () => {
    const diff = { status: 'blocked', source: 'mutation_baseline', blockers: ['base_commit_unknown'] };

    assert.equal(threadCheckoutRefusal(diff), null);
    assert.equal(diffSummaryMeta(diff, null), 'diff unavailable');
});

test('notes on a thread checkout are not called "attribution"', () => {
    // Nothing was attributed to a task window here: the whole tree is in scope,
    // so naming the attribution mechanism would name something not involved.
    const thread = diffBanners({
        status: 'ready', source: 'thread_checkout', blockers: ['untracked_patch_unavailable'],
    });
    assert.equal(thread.at(-1).text, 'Notes on this checkout');

    const task = diffBanners({
        status: 'ready', source: 'mutation_baseline', blockers: ['untracked_patch_unavailable'],
    });
    assert.equal(task.at(-1).text, 'Attribution notes');
});

test('HEAD drift stays a task-projection sentence, not a thread one', () => {
    // For a checkout whose whole purpose is to move ahead of its base, "HEAD
    // differs from the task baseline" would be noise about the normal state.
    const rows = diffBanners({ status: 'ready', source: 'thread_checkout', head_advanced: true });
    assert.deepEqual(rows, []);
});

test('the edit handoff names the BRANCH for a thread, not a task id that does not exist', () => {
    const parts = requestEditsParts(null, [makeTextPart('rename this')], {
        sourceMode: SOURCE_THREAD,
        threadRef: { projectId: 'racer', threadId: 2, label: 'Side quest', branch: 'thread/racer__2' },
    });

    // `normalizeParts` merges adjacent text, so the prefix and the comment are
    // one part — what matters is that the prefix leads and names the branch.
    assert.ok(parts[0].text.startsWith('Re the "Side quest" thread\'s branch thread/racer__2: '));
    assert.ok(parts[0].text.endsWith('rename this'));
    assert.doesNotMatch(parts[0].text, /Re task/);
    // Without a branch yet, it still refuses to invent a task reference.
    assert.equal(
        requestEditsThreadPrefix({ threadId: 2, label: 'Side quest' }),
        'Re the "Side quest" thread\'s own checkout: ',
    );
});

test('task mode still prepends the task line unchanged', () => {
    const parts = requestEditsParts({ task_id: 't1', title: 'Fix the parser' }, [makeTextPart('again')]);

    assert.ok(parts[0].text.startsWith('Re task t1 ("Fix the parser"): '));
});

test('a blocked CHECKOUT diff is not described as a task (T3R2-L4)', () => {
    // The neighbouring notes label already keys on `source`. This sentence did
    // not, so a thread-checkout diff blocked for anything other than the two
    // named states sent the owner looking for a task a worktree does not have.
    const [row] = diffBanners({
        status: 'blocked', source: 'thread_checkout', blockers: ['git_unavailable'],
    });
    assert.equal(row.tone, 'blocked');
    assert.equal(row.text, 'No trustworthy diff can be shown for this checkout.');

    // A task diff is still a task diff.
    const [taskRow] = diffBanners({
        status: 'blocked', source: 'mutation_baseline', blockers: ['git_unavailable'],
    });
    assert.equal(taskRow.text, 'No trustworthy diff can be shown for this task.');
});
