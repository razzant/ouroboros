import assert from 'node:assert/strict';
import test from 'node:test';

import { apiClient, fetchJson, signInUrl } from '../modules/api_client.js';

function respond(status, body) {
    const prior = globalThis.fetch;
    globalThis.fetch = async () => ({
        ok: status >= 200 && status < 300,
        status,
        json: async () => body,
    });
    return () => { globalThis.fetch = prior; };
}

test('a structured error body becomes a sentence, never [object Object]', async () => {
    const restore = respond(400, { error: { code: 'skill_not_finalized', message: 'Skill review is not finalized.' } });
    try {
        await assert.rejects(() => fetchJson('/api/x'), (error) => {
            assert.equal(error.message, 'Skill review is not finalized.');
            assert.equal(error.status, 400);
            return true;
        });
    } finally { restore(); }
});

test('a structured error with only a code still names the code', async () => {
    const restore = respond(409, { error: { code: 'conflict' } });
    try {
        await assert.rejects(() => fetchJson('/api/x'), /conflict/);
    } finally { restore(); }
});

test('a structured error with neither message nor code is serialized, not swallowed', async () => {
    const restore = respond(500, { error: { fields: ['a'] } });
    try {
        await assert.rejects(() => fetchJson('/api/x'), (error) => {
            assert.equal(error.message, '{"fields":["a"]}');
            assert.equal(error.message.includes('[object Object]'), false);
            return true;
        });
    } finally { restore(); }
});

test('a plain string error and a bare status are unchanged', async () => {
    let restore = respond(422, { error: 'Path escapes file browser root.' });
    try {
        await assert.rejects(() => fetchJson('/api/x'), /Path escapes file browser root\./);
    } finally { restore(); }
    restore = respond(503, {});
    try {
        await assert.rejects(() => fetchJson('/api/x'), /HTTP 503/);
    } finally { restore(); }
});

test('rejectOkFalse still reads the structured error of a 200 body', async () => {
    const restore = respond(200, { ok: false, error: { message: 'Nothing to restore.' } });
    try {
        await assert.rejects(
            () => fetchJson('/api/x', {}, { rejectOkFalse: true }),
            /Nothing to restore\./,
        );
    } finally { restore(); }
});

// Only the gate's 401 on the readiness and state reads sends a page here; the
// return keeps the page address and drops only the reload bookkeeping.
test('the sign-in return keeps the page address', () => {
    assert.equal(signInUrl('http://h/'), '/auth/login?next=%2F');
    assert.equal(signInUrl('http://h/ui/?a=1&_ouro_refresh=5#chat'), '/auth/login?next=%2Fui%2F%3Fa%3D1%23chat');
    assert.equal(signInUrl('https://h:8443/?_ouro_reason=ui-source&_ouro_refresh=1'), '/auth/login?next=%2F');
});

test('an unreadable OK body keeps its placeholder for ordinary callers and is an error only where asked', async () => {
    const prior = globalThis.fetch;
    globalThis.fetch = async () => ({ ok: true, status: 202, json: async () => { throw new TypeError('network error'); } });
    try {
        assert.deepEqual(await fetchJson('/api/x'), { error: 'non-json response (HTTP 202)' });
        await assert.rejects(() => fetchJson('/api/x', {}, { requireBody: true }), (error) => {
            assert.equal(error.status, undefined, 'no HTTP status: the caller reads it as an unknown outcome');
            assert.equal(error.unreadable, true);
            assert.equal(error.responseStatus, 202);
            return true;
        });
        // All nine Storage methods opt in; a method added later without it stays the placeholder reader.
        const storageCalls = [() => apiClient.storage(), () => apiClient.measureStorage(),
            () => apiClient.cancelStorageMeasure(), () => apiClient.previewStorageFree(),
            () => apiClient.freeStorage('p', ['g']), () => apiClient.freeStorageItems('p', ['a']),
            () => apiClient.freeStorageManual('p', ['a'], true),
            () => apiClient.storageFreeItems({ plan: 'p', view: 'pool' }), () => apiClient.cancelStorageFree()];
        assert.equal(storageCalls.length, Object.keys(apiClient).filter((name) => /^storage|Storage/.test(name)).length);
        for (const call of storageCalls) {
            await assert.rejects(call, (error) => error.unreadable === true && error.status === undefined && error.message === '');
        }
    } finally { globalThis.fetch = prior; }
});


for (const failure of [new SyntaxError('truncated JSON'), new TypeError('body stream failed')]) {
    test(`both update status readers reject an unreadable HTTP 200 body: ${failure.message}`, async () => {
        const prior = globalThis.fetch;
        globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => { throw failure; } });
        try {
            for (const read of [apiClient.updateStatus, apiClient.updateCheck]) {
                await assert.rejects(read, (error) => error.unreadable === true
                    && error.responseStatus === 200 && error.status === undefined);
            }
            assert.deepEqual(await fetchJson('/api/other'), { error: 'non-json response (HTTP 200)' });
        } finally { globalThis.fetch = prior; }
    });
}

test('updateStatus keeps readable JSON and typed HTTP failures', async () => {
    const body = { managed: true, available: true, letter: { text: 'Saved description.' } };
    let restore = respond(200, body);
    try { assert.deepEqual(await apiClient.updateStatus(), body); } finally { restore(); }
    restore = respond(503, { error: 'temporarily unavailable' });
    try {
        await assert.rejects(() => apiClient.updateStatus(), (error) => error.status === 503
            && error.body.error === 'temporarily unavailable');
    } finally { restore(); }
});

test('status and description doors reject JSON without an update status or a usable outcome', async () => {
    for (const body of [{}, null, [], { error: 'lost status' }, { managed: true }]) {
        const restore = respond(200, body);
        try {
            for (const read of [apiClient.updateStatus, apiClient.updateCheck, apiClient.updateDescribeLatest,
                () => apiClient.updateDescription('write')]) {
                await assert.rejects(read, error => error.unreadable && error.status === undefined);
            }
        } finally { restore(); }
    }
    for (const body of [{ status: 'queued' }, { status: 'writing', task_id: ' ' }, { status: 'current', update: {} }]) {
        const restore = respond(200, body);
        try { await assert.rejects(apiClient.updateDescribeLatest, error => error.unreadable); }
        finally { restore(); }
    }
    const restore = respond(200, { status: 'queued', task_id: 'writer', update: {} });
    try {
        assert.deepEqual(await apiClient.updateDescribeLatest(), { status: 'queued', task_id: 'writer', update: null });
    } finally { restore(); }
});

// The named Write/Rewrite door may already have admitted its writer when the answer arrives unreadable.
// That is an unknown outcome with no HTTP status, never a reply: the press says so instead of returning
// silently, and it never claims a refusal the server did not state.
test('the named description door rejects an unreadable HTTP 200 body and keeps readable answers', async () => {
    const prior = globalThis.fetch;
    globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError('truncated JSON'); } });
    try {
        await assert.rejects(() => apiClient.updateDescription('write', { base_sha: 'a', target_sha: 'b' }), (error) => {
            assert.equal(error.unreadable, true);
            assert.equal(error.responseStatus, 200);
            assert.equal(error.status, undefined, 'no HTTP status: the press reads it as an unknown outcome');
            assert.equal(error.message, '');
            return true;
        });
        // A reader that did not ask for a readable body still gets the placeholder: the change is this door's alone.
        assert.deepEqual(await fetchJson('/api/other'), { error: 'non-json response (HTTP 200)' });
    } finally { globalThis.fetch = prior; }
    const answer = { status: 'queued', task_id: 't1', key: { base_sha: 'a', target_sha: 'b' } };
    let restore = respond(200, answer);
    try { assert.deepEqual(await apiClient.updateDescription('rewrite', { base_sha: 'a', target_sha: 'b' }), answer); }
    finally { restore(); }
    restore = respond(409, { error: 'A description is already being written.', task_id: 't2' });
    try {
        await assert.rejects(() => apiClient.updateDescription('write', { base_sha: 'a', target_sha: 'b' }), (error) => {
            assert.equal(error.status, 409);
            assert.equal(error.body.task_id, 't2');
            return true;
        });
    } finally { restore(); }
});
