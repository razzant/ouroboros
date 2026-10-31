// An onboarding save whose answer did not arrive is read back by the attempt id the wizard made before
// sending (#682). These pin the client half: which answers count as unknown, and what each read-back
// answer lets the wizard do. Only the attempt's own receipt completes; nothing here re-sends.
import assert from 'node:assert/strict';
import test from 'node:test';

import {
    COMPLETION_REPLY_LOST_CODE,
    SAVE_NOT_CONFIRMED_TEXT,
    SAVE_STORED_TEXT,
    completionAttemptId,
    completionAttemptOutcome,
    completionFailureNotice,
    postCompletion,
    readCompletionAnswer,
    readCompletionAttempt,
} from '../modules/onboarding_agents_step.js';

const RECEIPT = { ok: true, status: 'saved', runtime_mode: 'pro', restart_required: true, preset: { applied: false } };
const reply = (status, body, { parses = true } = {}) => ({
    ok: status >= 200 && status < 300, status,
    json: async () => { if (!parses) throw new SyntaxError('Unexpected token <'); return body; },
});

test('each attempt id is fresh, made before sending, and fits the server\'s attempt_id rule', () => {
    const ids = new Set(Array.from({ length: 64 }, () => completionAttemptId()));
    assert.equal(ids.size, 64);
    for (const id of ids) assert.match(id, /^[0-9a-f]{32}$/);
    // It needs only getRandomValues, which a plain-HTTP LAN page has too (randomUUID needs a secure context).
    const fixed = completionAttemptId({ getRandomValues: (bytes) => bytes.fill(171) });
    assert.equal(fixed, 'ab'.repeat(16));
});

test('postCompletion sends the attempt and reads a reply that never arrived, or was not the endpoint\'s, as unknown', async () => {
    const calls = [];
    const ok = await postCompletion('/api/onboarding/complete', { attempt_id: 'a'.repeat(32) }, async (url, init) => {
        calls.push({ url, init });
        return reply(200, RECEIPT);
    });
    assert.deepEqual(ok, { receipt: RECEIPT });
    assert.equal(calls[0].init.method, 'POST');
    assert.equal(JSON.parse(calls[0].init.body).attempt_id, 'a'.repeat(32));

    const lost = await postCompletion('/x', {}, async () => { throw new TypeError('Failed to fetch'); });
    assert.equal(lost.failure.saved, null);
    assert.equal(lost.failure.code, COMPLETION_REPLY_LOST_CODE);
    assert.equal(completionFailureNotice(lost.failure).saveUnknown, true);

    for (const status of [200, 502]) {   // a proxy's HTML page, success-coded or not, is not the endpoint's envelope
        const html = await postCompletion('/x', {}, async () => reply(status, null, { parses: false }));
        assert.equal(html.failure.saved, null, `HTTP ${status}`);
        assert.equal(completionFailureNotice(html.failure).saveUnknown, true, `HTTP ${status}`);
    }
    // The endpoint's own typed answers keep their meaning.
    for (const saved of [undefined, null, 'false', 0, {}]) {
        const proxy = await postCompletion('/x', {}, async () => reply(502, { error: 'upstream unavailable', saved }));
        assert.equal(proxy.failure.saved, null);
        assert.equal(completionFailureNotice(proxy.failure).saveUnknown, true);
    }
    const refused = await postCompletion('/x', {}, async () => reply(409, { error: 'changed', code: 'onboarding_state_changed', saved: false }));
    assert.equal(refused.failure.saved, false);
    assert.equal(readCompletionAnswer({ status: 500, ok: false, parsed: true, data: { saved: true } }).failure.saved, true);
});

test('only the attempt\'s own receipt completes; stored, refused and unconfirmed each keep the draft', () => {
    assert.deepEqual(completionAttemptOutcome({ saved: true, receipt: RECEIPT }), { state: 'completed', receipt: RECEIPT });
    // A receipt without the envelope the wizard needs is no completion: the save is stored, finishing unconfirmed.
    assert.deepEqual(completionAttemptOutcome({ saved: true, receipt: { ok: true } }), { state: 'stored', text: SAVE_STORED_TEXT });
    assert.deepEqual(completionAttemptOutcome({ saved: true }), { state: 'stored', text: SAVE_STORED_TEXT });
    // A receipt only counts beside saved: true (the server's comparison with the settings document).
    assert.deepEqual(completionAttemptOutcome({ saved: null, receipt: RECEIPT }), { state: 'unconfirmed', text: SAVE_NOT_CONFIRMED_TEXT });
    assert.deepEqual(completionAttemptOutcome({ saved: null }), { state: 'unconfirmed', text: SAVE_NOT_CONFIRMED_TEXT });
    assert.deepEqual(completionAttemptOutcome(null), { state: 'unconfirmed', text: SAVE_NOT_CONFIRMED_TEXT });

    const refused = completionAttemptOutcome({ saved: false, error: 'nothing was written', code: 'onboarding_state_changed' });
    assert.equal(refused.state, 'failed');
    const refusal = completionFailureNotice(refused.failure);
    assert.deepEqual([refusal.saved, refusal.saveUnknown, refusal.canSkip, refusal.text], [false, false, false, 'nothing was written']);

    const afterCommit = completionAttemptOutcome({ saved: true, error: 'Settings were saved to disk, but the hot-reload step failed', post_commit_failed: 'hot-reload' });
    const notice = completionFailureNotice(afterCommit.failure);
    assert.equal(notice.saved, true);
    assert.match(notice.text, /WERE written — the step that failed afterwards was hot-reload/);
});

test('readCompletionAttempt asks about exactly one id, and an unreadable answer claims nothing', async () => {
    const asked = [];
    const outcome = await readCompletionAttempt('id_with-dash_0123', async (url, init) => {
        asked.push({ url, init });
        return reply(200, { saved: true, receipt: RECEIPT });
    });
    assert.equal(outcome.state, 'completed');
    assert.equal(asked[0].url, '/api/onboarding/complete?attempt=id_with-dash_0123');
    assert.equal(asked[0].init.method, undefined, 'a read, never a second POST');
    assert.equal(asked[0].init.cache, 'no-store');

    const offline = await readCompletionAttempt('a'.repeat(32), async () => { throw new TypeError('Failed to fetch'); });
    assert.deepEqual(offline, { state: 'unreadable', text: 'Could not check the save status: Failed to fetch. Try again in a moment.' });
    const older = await readCompletionAttempt('a'.repeat(32), async () => reply(405, null, { parses: false }));
    assert.deepEqual(older, { state: 'unreadable', text: 'Could not check the save status: HTTP 405. Try again in a moment.' });
    const malformed = await readCompletionAttempt('a'.repeat(32), async () => reply(400, { error: 'attempt is required' }));
    assert.equal(malformed.state, 'unreadable');
    assert.match(malformed.text, /attempt is required/);
});
