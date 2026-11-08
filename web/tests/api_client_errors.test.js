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
