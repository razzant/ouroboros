import assert from 'node:assert/strict';
import test from 'node:test';

import { apiConnections } from '../modules/route_editor_primitives.js';
import { moreProvidersOpen } from '../modules/settings_connections.js';

// The "More providers" fold (Cloud.ru, MiniMax, DeepSeek, Z.ai, GigaChat) opens by itself only for a key
// set up inside it: a stored credential, or a key added in one of its cards.
const FOLDED = ['cloudru', 'minimax', 'deepseek', 'zai', 'gigachat'];
const opens = (settings) => moreProvidersOpen(apiConnections(settings), FOLDED);

test('defaults-only values keep the More providers section closed', () => {
    assert.equal(opens({}), false);
    // Base URLs / scope / TLS carry shipped defaults and are not credentials: an unconfigured install stays closed.
    assert.equal(opens({
        CLOUDRU_FOUNDATION_MODELS_BASE_URL: 'https://foundation-models.api.cloud.ru/v1',
        GIGACHAT_SCOPE: 'GIGACHAT_API_PERS', GIGACHAT_BASE_URL: 'https://api.giga.chat/v1', GIGACHAT_VERIFY_SSL_CERTS: 'true',
        MINIMAX_REGION: 'global_en', ZAI_PLAN: 'payg',
    }), false);
    assert.equal(opens({ CLOUDRU_FOUNDATION_MODELS_API_KEY: '   ' }), false);
    // A key outside the fold never opens it.
    assert.equal(opens({ OPENROUTER_API_KEY: 'sk-or-v1...' }), false);
});

test('each usable credential path opens the section', () => {
    for (const settings of [
        { CLOUDRU_FOUNDATION_MODELS_API_KEY: 'ck-123' }, { MINIMAX_API_KEY: 'mm-123' }, { DEEPSEEK_API_KEY: 'sk-ds' },
        { ZAI_API_KEY: 'zai' }, { GIGACHAT_CREDENTIALS: 'base64pair' },
        // Masked secrets from the server ("***set***" / prefixed) are stored credentials.
        { CLOUDRU_FOUNDATION_MODELS_API_KEY: '***set***' },
        { GIGACHAT_USER: 'alice', GIGACHAT_PASSWORD: 'pw' },
    ]) assert.equal(opens(settings), true, JSON.stringify(settings));
});

test('an incomplete GigaChat basic-auth half does not open the section', () => {
    assert.equal(opens({ GIGACHAT_USER: 'alice' }), false);
    assert.equal(opens({ GIGACHAT_PASSWORD: 'pw' }), false);
});

test('the server-listed connections decide, and an added key opens the fold even before its key is stored', () => {
    const listed = (connections) => opens({ _meta: { connections } });
    assert.equal(listed([{ kind: 'deepseek', name: '', id: 'deepseek', builtin: true, enabled: true, configured: false }]), false);
    assert.equal(listed([{ kind: 'deepseek', name: '', id: 'deepseek', builtin: true, enabled: false, configured: true }]), true,
        'a turned-off key is still set up there');
    assert.equal(listed([{ kind: 'zai', name: 'coding', id: 'zai@coding', builtin: false, enabled: true, configured: false }]), true);
    assert.equal(listed([{ kind: 'openrouter', name: 'credits', id: 'openrouter@credits', builtin: false, enabled: true, configured: true }]), false);
});
