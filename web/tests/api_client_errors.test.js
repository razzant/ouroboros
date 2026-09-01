import assert from 'node:assert/strict';
import test from 'node:test';

import { fetchJson } from '../modules/api_client.js';

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
