import assert from 'node:assert/strict';
import test from 'node:test';
import { completionRequest, mountProviderReturn } from '../modules/provider_return.js';

const state = (skill = 'calendar', extra = {}) => Buffer.from(JSON.stringify({ v: 1, skill, nonce: 'nonce', ...extra })).toString('base64url');
const href = (value = state(), query = 'code=synthetic') => `https://example.test/instance/onboarding/static/provider-return.html?${query}&state=${value}`;

function page() {
    const nodes = Object.fromEntries(['status', 'detail', 'account', 'home'].map((name) => [name, {
        textContent: '', hidden: true, dataset: {}, setAttribute(key, value) { this[key] = value; },
    }]));
    return { nodes, root: { querySelector(selector) { return nodes[selector.match(/data-return-(\w+)/)[1]]; } } };
}

test('the exact return constructs one same-origin completion, retaining deployment prefix and canonical skill names', () => {
    for (const skill of ['calendar', 'mail.personal', 'почта_1']) {
        const request = completionRequest(href(state(skill)) + '&target=https://other.test/collect');
        assert.equal(request.url, `https://example.test/instance/api/extensions/${encodeURIComponent(skill)}/oauth/complete`);
        assert.equal(request.home, 'https://example.test/instance/');
        assert.equal(request.init.redirect, 'error');
        assert.equal(request.init.credentials, 'same-origin');
        assert.deepEqual(JSON.parse(request.init.body), { state: state(skill), code: 'synthetic' });
    }
    assert.deepEqual(JSON.parse(completionRequest(href(state(), 'error=access_denied&error_description=No')).init.body),
        { state: state(), error: 'access_denied', error_description: 'No' });
});

test('unrelated destinations, malformed envelopes and ambiguous results never become a request', () => {
    for (const skill of ['../settings', '.', 'calendar/complete', 'https://other.test', 'a%2fb']) {
        assert.throws(() => completionRequest(href(state(skill))));
    }
    for (const value of [state('calendar', { v: true }), state('calendar', { nonce: '' }), 'not-json']) {
        assert.throws(() => completionRequest(href(value)));
    }
    assert.throws(() => completionRequest(href(state(), 'error=denied&code=code')));
    assert.throws(() => completionRequest(href().replace('provider-return.html', 'other.html')));
});

test('actual startup sends once and paints the confirmed account without exposing tokens', async () => {
    const { root, nodes } = page();
    const calls = [], history = { replaceState(...args) { calls.push(['history', ...args]); } };
    await mountProviderReturn(root, { href: href(), history, lifecycle: null, send: async (url, init) => {
        calls.push(['post', url, init]);
        return { ok: true, json: async () => ({ ok: true, email: 'owner@example.test', access_token: 'must-not-render' }) };
    } });
    assert.equal(calls[1][0], 'post');
    assert.equal(calls[0][3], 'https://example.test/instance/onboarding/static/provider-return.html');
    assert.equal(nodes.status.textContent, 'Account connected');
    assert.equal(nodes.account.textContent, 'owner@example.test');
    assert.equal(nodes.account.hidden, false);
    assert.equal(nodes.home.href, 'https://example.test/instance/');
    assert.doesNotMatch(Object.values(nodes).map((node) => node.textContent).join(' '), /must-not-render/);
});

test('denied consent reaches the skill, and unconfirmed outcomes never claim connected or replay', async () => {
    for (const failure of ['denied', 'network', 'not-json']) {
        const { root, nodes } = page();
        let count = 0;
        await mountProviderReturn(root, { href: href(state(), 'error=access_denied'), history: null, lifecycle: null,
            send: async (_url, init) => {
                count += 1;
                assert.equal(JSON.parse(init.body).error, 'access_denied');
                if (failure === 'network') throw new Error('network');
                if (failure === 'not-json') return { ok: true, json: async () => { throw new Error('html'); } };
                return { ok: false, json: async () => ({ ok: false, error: 'Permission was not granted.' }) };
            } });
        assert.equal(count, 1);
        assert.equal(nodes.status.textContent, 'Could not finish connecting');
        assert.equal(nodes.account.hidden, true);
        if (failure === 'denied') assert.equal(nodes.detail.textContent, 'Permission was not granted.');
    }
});

test('history restrictions do not prevent a valid connection, and page exit disposes its request', async () => {
    const { root, nodes } = page();
    const events = new Map();
    const lifecycle = { addEventListener(key, fn) { events.set(key, fn); }, removeEventListener(key) { events.delete(key); } };
    await mountProviderReturn(root, { href: href(), lifecycle, history: { replaceState() { throw new Error('disabled'); } },
        send: async (_url, init) => {
            events.get('pagehide')();
            assert.equal(init.signal.aborted, true);
            throw new Error('closed');
        } });
    assert.equal(nodes.status.textContent, '');
    assert.equal(events.size, 0);
});
