import assert from 'node:assert/strict';
import test from 'node:test';

import { moreProvidersCredentialConfigured } from '../modules/settings.js';

test('defaults-only values keep the More providers section closed', () => {
    assert.equal(moreProvidersCredentialConfigured({}), false);
    // Base URLs / scope / TLS carry shipped defaults and are deliberately not
    // part of the predicate's inputs — an unconfigured install stays closed.
    assert.equal(moreProvidersCredentialConfigured({
        cloudruKey: '', minimaxKey: '', deepseekKey: '', gigachatCredentials: '', gigachatUser: '', gigachatPassword: '',
    }), false);
    assert.equal(moreProvidersCredentialConfigured({ cloudruKey: '   ' }), false);
});

test('each usable credential path opens the section', () => {
    assert.equal(moreProvidersCredentialConfigured({ cloudruKey: 'ck-123' }), true);
    assert.equal(moreProvidersCredentialConfigured({ minimaxKey: 'mm-123' }), true);
    assert.equal(moreProvidersCredentialConfigured({ deepseekKey: 'sk-ds' }), true);
    assert.equal(moreProvidersCredentialConfigured({ gigachatCredentials: 'base64pair' }), true);
    // Masked secrets from the server ("***set***" / prefixed) are non-empty
    // strings and count as configured.
    assert.equal(moreProvidersCredentialConfigured({ cloudruKey: '***set***' }), true);
    assert.equal(moreProvidersCredentialConfigured({
        gigachatUser: 'alice', gigachatPassword: 'pw',
    }), true);
});

test('an incomplete GigaChat basic-auth half does not open the section', () => {
    assert.equal(moreProvidersCredentialConfigured({ gigachatUser: 'alice' }), false);
    assert.equal(moreProvidersCredentialConfigured({ gigachatPassword: 'pw' }), false);
});
