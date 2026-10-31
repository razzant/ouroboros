// Owner Batch4: the Continue press keeps ONE action nonce per task across
// retries and reloads, reads the server's offer, and points a stale card to
// its accepted successor instead of offering a second Continue.

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import { continueNonce, continueOfferView, continueTaskAction, syncContinueAction } from '../modules/task_continue.js';
import { taskReasonDetail, taskReasonPhrase } from '../modules/log_events.js';
import { ElementStub, installDom, restoreDom } from './chat_dom_fixture.js';

function memoryStorage() {
    const map = new Map();
    return { getItem: (k) => (map.has(k) ? map.get(k) : null), setItem: (k, v) => map.set(k, String(v)), map };
}

test('the action nonce is created once per task and survives a reload (same storage)', async () => {
    const storage = memoryStorage();
    const first = continueNonce('task-1', storage);
    assert.match(first, /^[A-Za-z0-9_-]{8,128}$/);
    assert.equal(continueNonce('task-1', storage), first, 'a retry or a reload reuses the SAME nonce');
    assert.notEqual(continueNonce('task-2', storage), first, 'another task is another action');
    const reloaded = await import('../modules/task_continue.js?reload-test');
    assert.equal(reloaded.continueNonce('task-1', storage), first);
});

test('lost answers retain the page action when storage reads or writes throw', async () => {
    for (const failingMethod of ['getItem', 'setItem']) {
        const storage = memoryStorage();
        storage[failingMethod] = () => { throw new Error('storage unavailable'); };
        const seen = [];
        const request = async (_id, nonce) => {
            seen.push(nonce);
            if (seen.length === 1) throw new Error('answer lost');
            return { ok: true, successor_task_id: 'accepted-root' };  // the server's success body
        };
        const id = `storage-failure-${failingMethod}`;
        assert.equal(await continueTaskAction(id, { request, storage, toast() {} }), '');
        assert.equal(await continueTaskAction(id, { request, storage, toast() {} }), 'accepted-root');
        assert.equal(seen[0], seen[1]);
        const recoveredStorage = memoryStorage();
        assert.equal(continueNonce(id, recoveredStorage), seen[0]);
        assert.equal(recoveredStorage.getItem(`ouro_continue_nonce:${id}`), seen[0]);
    }
});

test('a throwing localStorage accessor still permits a stable page action', () => {
    const prior = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
    try {
        Object.defineProperty(globalThis, 'localStorage', { configurable: true,
            get() { throw new Error('access denied'); } });
        const first = continueNonce('accessor-failure');
        assert.equal(continueNonce('accessor-failure'), first);
    } finally {
        if (prior) Object.defineProperty(globalThis, 'localStorage', prior);
        else delete globalThis.localStorage;
    }
});

test('the card offers Continue only as the server says, and otherwise points to the successor', () => {
    assert.deepEqual(continueOfferView({ continuation_offer: { eligible: true, cause: 'owner_restart' } }),
        { kind: 'offer', cause: 'owner_restart' });
    assert.deepEqual(continueOfferView({ continuation_offer: { eligible: false, refusal: 'stopped_by_owner' } }),
        { kind: 'none' });
    assert.deepEqual(continueOfferView({ continuation_offer: { eligible: false, successor_task_id: 's-1' } }),
        { kind: 'successor', successorId: 's-1' });
    assert.deepEqual(continueOfferView({}), { kind: 'none' });
});

test('a lost answer is retried under the SAME nonce; an already accepted press names its successor', async () => {
    const storage = memoryStorage();
    const seen = [];
    const toasts = [];
    const toast = (text, kind) => toasts.push([text, kind]);
    let attempt = 0;
    const request = async (id, nonce) => {
        seen.push([id, nonce]);
        attempt += 1;
        if (attempt === 1) throw Object.assign(new Error('network timeout'), { body: {} });
        return { ok: true, successor_task_id: 'root-1-cabc', held: attempt === 2 };
    };
    assert.equal(await continueTaskAction('root-1', { request, storage, toast }), '');
    assert.match(toasts[0][0], /^Continue not confirmed: network timeout/);
    assert.equal(await continueTaskAction('root-1', { request, storage, toast }), 'root-1-cabc');
    assert.equal(seen[0][1], seen[1][1], 'the retry carries the same action nonce');
    // Held without named blockers: the fact, never a promise that earlier work is merely settling (#1721). It starts
    // by itself once that work settles: no Resume is offered for it.
    assert.equal(toasts[1][0], "Continue accepted; it starts by itself when the earlier attempt's work finishes: "
        + 'earlier work is not proven settled.');
    assert.doesNotMatch(toasts[1][0], /Resume/);
    assert.ok(!toasts[1][0].includes('root-1-cabc'), 'compact acknowledgement hides opaque identity');
    const refused = async () => { throw Object.assign(new Error('continue refused: already_continued'), {
        body: { reason_code: 'already_continued', successor_task_id: 'root-1-cabc' } }); };
    assert.equal(await continueTaskAction('root-9', { request: refused, storage, toast }), 'root-1-cabc');
});

test('chat reaches the Continue action through the one settled-card seam', () => {
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    assert.match(chat, /import \{ syncSettledItems \} from '\.\/settled_card\.js';/);
    const seam = readFileSync(new URL('../modules/settled_card.js', import.meta.url), 'utf8');
    assert.match(seam, /syncContinueAction\(record, detail, options\)/);
    assert.match(seam, /return syncResultFilesItem\(record, detail\)/);
});

function settledRootCard(taskId, { isSubagent = false } = {}) {
    const root = new ElementStub('div', globalThis.document);
    root.isConnected = true;
    root.classList.add('chat-live-card');
    const deep = (node, key) => {
        for (const child of node.children) {
            if (Object.hasOwn(child.dataset, key)) return child;
            const hit = deep(child, key);
            if (hit) return hit;
        }
        return null;
    };
    root.querySelector = (selector) => {
        const key = { '[data-continue-task]': 'continueTask', '[data-continue-refusal]': 'continueRefusal' }[selector];
        return key ? deep(root, key) : null;
    };
    return { root, groupId: taskId, isSubagent };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

test('a Continue read late, after its card was drawn, is written through the caller\'s viewport seam', async () => {
    // An older page lands, then each unclean root's full detail arrives and adds its action row. Written outside the
    // reading owner's transaction, every row above the passage pushed it down (WebKit has no native scroll anchoring).
    const { prior } = installDom();
    try {
        const card = settledRootCard('late-root');
        const seam = [];
        const write = (fn) => { seam.push('open'); const result = fn(); seam.push('close'); return result; };
        const read = async () => ({ status: 'cancelled', continuation_offer: { eligible: true, cause: 'snapshot_restore' } });
        syncContinueAction(card, { type: 'task_done', status: 'cancelled' }, { read, write });
        assert.deepEqual(seam, [], 'nothing is written before the detail arrives');
        await flush();
        assert.deepEqual(seam, ['open', 'close'], 'the late write is one transaction of the caller\'s seam');
        assert.equal(card.root.querySelector('[data-continue-task]')?.textContent, 'Continue');
    } finally { restoreDom(prior); }
});

test('a canonical hard-bound refusal remains one readable fact beside the existing actions', async () => {
    const { prior } = installDom();
    try {
        const card = settledRootCard('expired-root');
        const detail = { status: 'failed', reason_code: 'provider_unavailable',
            continuation_offer: { eligible: false, refusal: 'hard_limit_reached', cause: 'deadline' } };
        const terminalReason = taskReasonDetail(detail);
        assert.match(terminalReason, /usable model answer/);
        assert.deepEqual(continueOfferView(detail), { kind: 'refusal', cause: 'deadline' });
        syncContinueAction(card, detail);
        const note = card.root.querySelector('[data-continue-refusal]');
        assert.ok(note, 'an API-only reason and a hidden button are insufficient');
        assert.equal(note.textContent, `Continue unavailable. ${taskReasonPhrase('deadline')}`);
        assert.equal(note.className, 'ui-status');
        assert.equal(note.dataset.tone, 'muted', 'the shared primitive normalizes neutral to its muted alias');
        assert.equal(note.tagName, 'SPAN');
        assert.equal(note.getAttribute('aria-live'), '');
        assert.equal(note.onclick, undefined);
        assert.equal(card.root.querySelector('[data-continue-task]'), null);
        syncContinueAction(card, detail);
        syncContinueAction(card, { type: 'task_metrics_event' });
        assert.equal(card.root.querySelector('[data-continue-refusal]'), note);
        assert.equal(note.parentElement.children.length, 1);
        assert.equal(taskReasonDetail(detail), terminalReason, 'action availability does not rewrite task outcome');

        syncContinueAction(card, { continuation_offer: { eligible: false,
            refusal: 'hard_limit_reached', cause: 'budget_exhausted' } });
        assert.equal(card.root.querySelector('[data-continue-refusal]'), note);
        assert.equal(note.textContent, `Continue unavailable. ${taskReasonPhrase('budget_exhausted')}`);
        const reloaded = settledRootCard('expired-root');
        syncContinueAction(reloaded, detail);
        assert.equal(reloaded.root.querySelector('[data-continue-refusal]').textContent,
            `Continue unavailable. ${taskReasonPhrase('deadline')}`);

        for (const offer of [{ eligible: true, cause: 'provider_unavailable' },
            { eligible: false, state: 'bound', action_nonce: 'bound-nonce-123', successor_task_id: 'next' },
            { eligible: false, successor_task_id: 'next' }]) {
            syncContinueAction(card, detail);
            syncContinueAction(card, { continuation_offer: offer });
            assert.equal(card.root.querySelector('[data-continue-refusal]'), null);
            assert.ok(card.root.querySelector('[data-continue-task]'));
        }
        assert.equal(card.root.querySelector('[data-continue-task]').textContent, 'Continued');
    } finally { restoreDom(prior); }
});

test('missing offers, other refusals and converted Main pointers gain no hard-bound claim', () => {
    const { prior } = installDom();
    try {
        for (const offer of [undefined, null, {}, { eligible: false, refusal: 'stopped_by_owner' },
            { eligible: false, refusal: 'author_finished' }, { eligible: false, refusal: 'predecessor_live' }]) {
            const card = settledRootCard('without-hard-bound');
            syncContinueAction(card, offer === undefined ? {} : { continuation_offer: offer });
            assert.equal(card.root.querySelector('[data-continue-refusal]'), null);
            assert.equal(card.root.querySelector('[data-continue-task]'), null);
        }
        const converted = settledRootCard('converted-root');
        converted.root.dataset.projectCreated = '1';
        syncContinueAction(converted, { continuation_offer: {
            eligible: false, refusal: 'hard_limit_reached', cause: 'deadline' } });
        assert.equal(converted.root.children.length, 0, 'Main keeps its Project pointer, without a second action row');
    } finally { restoreDom(prior); }
});

test('an event row never erases the offer; a root that ended without an answer reads its detail once', async () => {
    const { prior } = installDom();
    try {
        const reads = [];
        const read = async (id) => {
            reads.push(id);
            return { status: 'cancelled', continuation_offer: { eligible: true, cause: 'snapshot_restore' } };
        };
        const card = settledRootCard('crashed-root');
        // A replayed terminal event carries no offer: it asks the full detail once.
        assert.equal(syncContinueAction(card, { type: 'task_done', status: 'cancelled' }, { read }), false);
        await flush();
        assert.deepEqual(reads, ['crashed-root']);
        const button = card.root.querySelector('[data-continue-task]');
        assert.equal(button?.textContent, 'Continue');
        // Later event rows neither re-read nor remove the shown action.
        syncContinueAction(card, { type: 'task_eval', status: 'cancelled' }, { read });
        syncContinueAction(card, { type: 'task_metrics_event' }, { read });
        await flush();
        assert.deepEqual(reads, ['crashed-root']);
        assert.equal(card.root.querySelector('[data-continue-task]'), button);
        // The full detail still decides: the accepted successor replaces the offer.
        syncContinueAction(card, { continuation_offer: { eligible: false, successor_task_id: 'crashed-root-c1' } });
        assert.equal(button.textContent, 'Continued');
        assert.equal(button.dataset.continueSuccessor, 'crashed-root-c1');
        assert.equal(button.disabled, true);

        // A finished answer, a live row and a child card never read the detail.
        for (const [record, row] of [[settledRootCard('done-root'), { type: 'task_done', status: 'completed' }],
            [settledRootCard('live-root'), { type: 'task_done', status: 'running' }],
            [settledRootCard('child', { isSubagent: true }), { type: 'task_done', status: 'failed' }]]) {
            syncContinueAction(record, row, { read });
        }
        await flush();
        assert.deepEqual(reads, ['crashed-root']);
    } finally {
        restoreDom(prior);
    }
});

test('a child card never shows Continue, even when a replayed row carries its parent\'s offer', () => {
    const { prior } = installDom();
    try {
        // A rejected subagent's notice rides under the parent's task id and can carry the parent's offer.
        const row = { is_progress: true, task_terminal_status: 'cancelled',
            continuation_offer: { eligible: true, cause: 'snapshot_restore' } };
        const parent = settledRootCard('parent-probe');
        syncContinueAction(parent, row);
        assert.equal(parent.root.querySelector('[data-continue-task]')?.textContent, 'Continue', 'the root keeps its offer');
        const child = settledRootCard('child-probe', { isSubagent: true });
        syncContinueAction(child, row);
        assert.equal(child.root.querySelector('[data-continue-task]'), null);
        syncContinueAction(child, { ...row, continuation_offer: { eligible: false, refusal: 'not_a_root_task' } });
        assert.equal(child.root.querySelector('[data-continue-refusal]'), null, 'no refusal line on a child either');
    } finally {
        restoreDom(prior);
    }
});

for (const failure of ['reject', 'null', 'missing-offer', 'null-offer']) {
test(`an unusable detail read (${failure}) lets the next terminal row retry without polling`, async () => {
    const { prior } = installDom();
    try {
        let calls = 0;
        const read = async () => {
            calls += 1;
            if (calls === 1) {
                if (failure === 'reject') throw new Error('offline');
                return failure === 'null' ? null : failure === 'null-offer' ? { continuation_offer: null } : {};
            }
            return { status: 'failed', continuation_offer: { eligible: true, cause: 'provider_unavailable' } };
        };
        const card = settledRootCard('flaky-root');
        syncContinueAction(card, { type: 'task_done', status: 'failed' }, { read });
        await flush();
        await flush();
        assert.equal(calls, 1, 'a failed read does not start a retry loop');
        assert.equal(card.root.querySelector('[data-continue-task]'), null);
        syncContinueAction(card, { type: 'task_done', status: 'failed' }, { read });
        await flush();
        assert.equal(calls, 2);
        assert.equal(card.root.querySelector('[data-continue-task]')?.textContent, 'Continue');
        syncContinueAction(card, { type: 'task_eval', status: 'failed' }, { read });
        await flush();
        assert.equal(calls, 2, 'usable detail is cached for later terminal rows');
    } finally {
        restoreDom(prior);
    }
});
}

test('production 503/null detail recovers on the next terminal row with one read in flight', async () => {
    let reads = 0;
    let finishRead;
    const { prior } = installDom(async () => {
        reads += 1;
        if (reads === 1) return { ok: false, status: 503 };
        return new Promise((resolve) => { finishRead = () => resolve({ ok: true,
            json: async () => ({ continuation_offer: { eligible: false, refusal: 'stopped_by_owner' } }) }); });
    });
    try {
        const card = settledRootCard('production-flaky-root');
        const terminal = { type: 'task_done', status: 'failed' };
        syncContinueAction(card, terminal);
        await flush();
        assert.equal(reads, 1);
        syncContinueAction(card, terminal);
        syncContinueAction(card, terminal);
        await flush();
        assert.equal(reads, 2, 'the production null reader can retry, but not concurrently');
        finishRead();
        await flush();
        syncContinueAction(card, terminal);
        await flush();
        assert.equal(reads, 2, 'a valid ineligible offer is also a completed read');
        assert.equal(card.root.querySelector('[data-continue-task]'), null);
    } finally { restoreDom(prior); }
});

test('a replayed history row states the offer: shown after a reload without opening, never read', async () => {
    const { prior } = installDom();
    try {
        const reads = [];
        const read = async (id) => { reads.push(id); return null; };
        // The history projection carries the host's offer on the settled root's rows.
        const eligible = settledRootCard('restarted-root');
        syncContinueAction(eligible, { system_type: 'task_summary', task_terminal_status: 'cancelled',
            continuation_offer: { eligible: true, cause: 'owner_restart' } }, { read });
        assert.equal(eligible.root.querySelector('[data-continue-task]')?.textContent, 'Continue');
        const claimed = settledRootCard('continued-root');
        syncContinueAction(claimed, { is_progress: true, task_terminal_status: 'cancelled',
            continuation_offer: { eligible: false, refusal: 'already_continued', successor_task_id: 'continued-root-c9' } },
        { read });
        const pointer = claimed.root.querySelector('[data-continue-task]');
        assert.equal(pointer?.textContent, 'Continued');
        assert.equal(pointer?.dataset.continueSuccessor, 'continued-root-c9');
        assert.equal(pointer?.disabled, true);
        // An ineligible root's rows say so too: no button and no detail read.
        const stopped = settledRootCard('stopped-root');
        syncContinueAction(stopped, { system_type: 'task_summary', task_terminal_status: 'cancelled',
            continuation_offer: { eligible: false, refusal: 'stopped_by_owner' } }, { read });
        assert.equal(stopped.root.querySelector('[data-continue-task]'), null);
        await flush();
        assert.deepEqual(reads, []);
    } finally {
        restoreDom(prior);
    }
});

test('a live best-effort completion reads its detail once (a technical limit may have ended it)', async () => {
    const { prior } = installDom();
    try {
        const reads = [];
        const read = async (id) => {
            reads.push(id);
            return { status: 'completed', continuation_offer: { eligible: true, cause: 'round_limit' } };
        };
        const limited = settledRootCard('limited-root');
        syncContinueAction(limited, { type: 'task_done', status: 'completed', reason_code: 'round_limit',
            outcome_axes: { execution: { status: 'best_effort' } } }, { read });
        const clean = settledRootCard('clean-root');
        syncContinueAction(clean, { type: 'task_done', status: 'completed',
            outcome_axes: { execution: { status: 'ok' } } }, { read });
        await flush();
        assert.deepEqual(reads, ['limited-root'], 'a clean completion never reads');
        assert.equal(limited.root.querySelector('[data-continue-task]')?.textContent, 'Continue');
    } finally {
        restoreDom(prior);
    }
});

test('chat replays settled rows through the one seam, not a card-opening detail read', () => {
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    const summary = chat.slice(chat.indexOf('function appendTaskSummaryToLiveCard'),
        chat.indexOf('function renderLiveCardMeta'));
    assert.match(summary, /syncSettledItems\(record, msg, \{ write: withStableViewport \}\);/);
    // A task detail can start the late Continue read (an unclean root): its write goes through the same seam.
    const details = chat.slice(chat.indexOf('function noteTaskDetails'), chat.indexOf('function noteTaskDetails') + 300);
    assert.match(details, /syncSettledItems\(record, detail, \{ write: withStableViewport \}\);/);
});


test('bound history restores the server nonce after reload; its ID is not admission', async () => {
    const reloaded = await import('../modules/task_continue.js?bound-reload');
    const storage = memoryStorage();
    const offer = { state: 'bound', eligible: false, successor_task_id: 'bound-successor', action_nonce: 'retained-nonce-123' };
    assert.deepEqual(reloaded.continueOfferView({ continuation_offer: offer }),
        { kind: 'retry', actionNonce: offer.action_nonce });
    assert.deepEqual(reloaded.continueOfferView({ continuation_offer: { ...offer, action_nonce: '' } }), { kind: 'none' });
    const seen = [];
    const request = async (_id, nonce) => { seen.push(nonce); return { ok: true, successor_task_id: 'bound-successor' }; };
    assert.equal(await reloaded.continueTaskAction('bound-root', { storage, request, toast() {}, actionNonce: offer.action_nonce }), 'bound-successor');
    assert.deepEqual(seen, [offer.action_nonce]);
    assert.equal(reloaded.continueNonce('bound-root', storage), offer.action_nonce);
});

test('a resolved malformed 200 is unconfirmed: no success toast, the same nonce retries', async () => {
    // Real continueTask -> fetchJson: a truncated HTTP-200 body resolves (not rejects) as {error}.
    const priorFetch = globalThis.fetch;
    const storage = memoryStorage();
    const toasts = [];
    const toast = (text, kind) => toasts.push([text, kind]);
    const nonces = [];
    const replies = [
        { ok: true, status: 200, json: async () => { throw new SyntaxError('Unexpected end of JSON input'); } },
        { ok: true, status: 200, json: async () => ({ ok: true, held: false }) },
        { ok: true, status: 200, json: async () => ({ ok: true, successor_task_id: 'malformed-root-c1' }) },
    ];
    globalThis.fetch = async (_url, init) => { nonces.push(JSON.parse(init.body).action_nonce); return replies.shift(); };
    try {
        assert.equal(await continueTaskAction('malformed-root', { storage, toast }), '');
        assert.deepEqual(toasts, [['Continue not confirmed: non-json response (HTTP 200)', 'error']]);
        assert.equal(storage.getItem('ouro_continue_nonce:malformed-root'), nonces[0], 'the unconfirmed action keeps its nonce');
        assert.equal(await continueTaskAction('malformed-root', { storage, toast }), '', 'ok without a successor is not an acknowledgement');
        assert.deepEqual(toasts[1], ['Continue not confirmed: no acknowledgement', 'error']);
        assert.equal(await continueTaskAction('malformed-root', { storage, toast }), 'malformed-root-c1');
        assert.deepEqual(toasts[2], ['Continue accepted.', 'ok']);
        assert.deepEqual(nonces, [nonces[0], nonces[0], nonces[0]], 'every retry is the same admission');
    } finally {
        globalThis.fetch = priorFetch;
    }
});

test('a held Continue toasts the blockers its census observed, not a generic wait (#1721)', async () => {
    const toasts = [];
    const blockers = [{ kind: 'delegated_run', task_id: 'pred-1', run_id: '', invocation_id: 'inv-1',
        state: 'stop_unknown', stop_outcome: 'pending_invocation_unbound', source: '' }];
    const request = async () => ({ ok: true, held: true, successor_task_id: 'pred-1-s1', blockers });
    assert.equal(await continueTaskAction('held-facts', { request, storage: memoryStorage(), toast: (t, k) => toasts.push([t, k]) }),
        'pred-1-s1');
    assert.deepEqual(toasts, [["Continue accepted; it starts by itself when the earlier attempt's work finishes: "
        + 'a delegated start has an unknown outcome (no run was recorded).', 'ok']]);
});

test('an accepted Continue names the records its census did not wait for in its own warning, never as another task\'s', async () => {
    // The real request through fetch: the server's admission answer with its typed gap (membership_gaps).
    const priorFetch = globalThis.fetch;
    const toasts = [];
    const toast = (...args) => toasts.push(args);
    const gap = { rows: [{ task_id: 'stray-1', reason: 'body_unreadable' }], count: 1 };
    const replies = [
        { ok: true, successor_task_id: 'gap-root-c1', held: false, membership_gaps: gap },
        { ok: true, successor_task_id: 'gap-root-c1', held: false, replay: true, membership_gaps: gap },
        { ok: true, successor_task_id: 'plain-root-c1', held: false },
    ];
    globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => replies.shift() });
    try {
        assert.equal(await continueTaskAction('gap-root', { storage: memoryStorage(), toast }), 'gap-root-c1');
        assert.deepEqual(toasts.shift(), ['Continue accepted.', 'ok'], 'the action\'s own words are unchanged');
        const [words, tone, options] = toasts.shift();
        assert.equal(words, 'Continue did not wait for 1 saved task record that could not be read and that nothing '
            + 'ties to this task (stray-1); whether it belongs here is unknown.');
        assert.equal(tone, 'warn');
        assert.deepEqual(options, { ttl: 0, dismissLabel: 'Dismiss notice' }, 'kept until the owner dismisses it');
        assert.doesNotMatch(words, /another task|other task|belongs to/);
        // A replay of the same admission answers the same gap.
        assert.equal(await continueTaskAction('gap-root', { storage: memoryStorage(), toast }), 'gap-root-c1');
        assert.equal(toasts.length, 2);
        assert.equal(toasts[1][0], words);
        toasts.length = 0;
        assert.equal(await continueTaskAction('plain-root', { storage: memoryStorage(), toast }), 'plain-root-c1');
        assert.deepEqual(toasts, [['Continue accepted.', 'ok']], 'no gap, no second notice');
    } finally {
        globalThis.fetch = priorFetch;
    }
});

test('a held Continue keeps its observed blockers and adds a counted, bounded gap', async () => {
    const toasts = [];
    const rows = Array.from({ length: 20 }, (_, i) => ({ task_id: `r${i}`, reason: 'body_unreadable' }));
    const request = async () => ({ ok: true, held: true, successor_task_id: 'held-gap-c1',
        blockers: [{ kind: 'running_member', task_id: 'kid' }], membership_gaps: { rows, count: 23 } });
    assert.equal(await continueTaskAction('held-gap', { request, storage: memoryStorage(), toast: (...a) => toasts.push(a) }),
        'held-gap-c1');
    assert.equal(toasts[0][0], "Continue accepted; it starts by itself when the earlier attempt's work finishes: "
        + 'task kid is still running.');
    assert.match(toasts[1][0], /^Continue did not wait for 23 saved task records that could not be read and that nothing ties to this task \(r0, r1, .*, r19, …\); whether they belong here is unknown\.$/);
});
