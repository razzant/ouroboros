import assert from 'node:assert/strict';
import test from 'node:test';
import { completionRequest, mountProviderReturn } from '../modules/provider_return.js';

const state = (skill = 'calendar', extra = {}) => Buffer.from(JSON.stringify({ v: 1, skill, nonce: 'nonce', ...extra })).toString('base64url');
const href = (value = state(), query = 'code=synthetic') => `https://example.test/instance/onboarding/static/provider-return.html?${query}&state=${value}`;

function page() {
    const nodes = Object.fromEntries(['status', 'detail', 'account', 'home', 'next', 'again'].map((name) => [name, {
        textContent: '', hidden: true, dataset: {}, setAttribute(key, value) { this[key] = value; },
        addEventListener(type, handler) { this.handlers = { ...this.handlers, [type]: handler }; },
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

const mcpState = (server = 'superset', extra = {}) => Buffer.from(JSON.stringify({ v: 1, mcp: server, nonce: 'nonce', ...extra })).toString('base64url');

test('a remote MCP server return completes on the core sign-in route, and every return forwards the issuer it names', () => {
    const iss = 'iss=https%3A%2F%2Fauth.example.test';
    const request = completionRequest(href(mcpState(), `code=synthetic&${iss}`));
    assert.equal(request.url, 'https://example.test/instance/api/mcp/sign-in');
    assert.deepEqual(JSON.parse(request.init.body),
        { action: 'complete', state: mcpState(), code: 'synthetic', iss: 'https://auth.example.test' });
    assert.deepEqual(JSON.parse(completionRequest(href(state(), `code=synthetic&${iss}`)).init.body),
        { state: state(), code: 'synthetic', iss: 'https://auth.example.test' });
    assert.deepEqual(JSON.parse(completionRequest(href(mcpState(), 'error=access_denied')).init.body),
        { action: 'complete', state: mcpState(), error: 'access_denied', error_description: '' });
    for (const value of [mcpState('../settings'), mcpState('superset', { skill: 'calendar' }), mcpState('')]) {
        assert.throws(() => completionRequest(href(value)));
    }
});

const signInState = (extra = {}) => Buffer.from(JSON.stringify({ v: 1, sign_in: 'calendar', svc: 'google', nonce: 'nonce', ...extra })).toString('base64url');

test('a skill\'s declared-service return completes on the core sign-in route; one recipient and its service only', () => {
    const request = completionRequest(href(signInState(), 'code=synthetic&iss=https%3A%2F%2Fissuer.test'));
    assert.equal(request.url, 'https://example.test/instance/api/sign-in');
    assert.deepEqual(JSON.parse(request.init.body),
        { action: 'complete', state: signInState(), code: 'synthetic', iss: 'https://issuer.test' });
    for (const value of [signInState({ svc: undefined }), signInState({ svc: '../x' }), signInState({ skill: 'calendar' }),
        signInState({ mcp: 'docs' })]) {
        assert.throws(() => completionRequest(href(value)));
    }
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

test('a declared-service return names the confirmed account, including when access is partial', async () => {
    for (const [state, reason, scopes] of [['signed_in', '', []], ['sign_in_required', 'insufficient_scope', ['records.read']]]) {
        const { root, nodes } = page();
        const partial = state === 'sign_in_required';
        await mountProviderReturn(root, { href: href(signInState()), history: null, lifecycle: null, send: async () => ({ ok: true,
            json: async () => ({ ok: true, label: 'Fixture service', state, reason, attempt_id: 'fixture-attempt',
                account: { key: 'fixture-account', label: 'work@example.test' }, missing_required: scopes }),
        }) });
        assert.equal(nodes.account.textContent, 'work@example.test');
        assert.equal(nodes.account.hidden, false);
        assert.equal(nodes.status.dataset.tone, partial ? 'warn' : 'ok');
        assert.equal(nodes.status.textContent === 'Account connected', !partial);
    }
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
        const check = 'Return to Ouroboros and check the account before starting sign-in again.';
        if (failure === 'denied') {
            assert.equal(nodes.detail.textContent, 'Permission was not granted.');
            assert.deepEqual([nodes.next.textContent, nodes.next.hidden], [check, false], 'the host message never replaces the next step');
        } else {
            assert.deepEqual([nodes.detail.textContent, nodes.next.hidden], [check, true], 'without a message the next step is the detail itself');
        }
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

test('a saved provider answer is finished again only by the owner pressing Finish, from memory and as the same return', async () => {
    const { root, nodes } = page();
    const sent = [];
    const replies = [
        { ok: false, code: 'SIGN_IN_REQUIRED', reason: 'answer_saved', attempt_id: 'fixture-attempt', error: 'kept' },
        { ok: true, label: 'Fixture service', state: 'signed_in', attempt_id: 'fixture-attempt', account: { key: 'fixture-account', label: 'work@example.test' } },
    ];
    const history = { replaceState() {} };
    await mountProviderReturn(root, { href: href(signInState()), history, lifecycle: null, send: async (_url, init) => {
        sent.push(init.body);
        return { ok: true, json: async () => replies.shift() };
    } });
    assert.equal(sent.length, 1, 'no automatic second send');
    assert.equal(nodes.again.hidden, false);
    assert.equal(nodes.next.hidden, true, 'the press itself is the next step');
    assert.match(nodes.detail.textContent, /the one-time code isn’t sent to the provider a second time/);
    assert.equal(nodes.account.hidden, true);
    await nodes.again.handlers.click();
    assert.equal(sent.length, 2);
    assert.equal(sent[1], sent[0], 'the same return, from memory');
    assert.equal(nodes.status.textContent, 'Account connected');
    assert.equal(nodes.account.textContent, 'work@example.test');
    assert.equal(nodes.again.hidden, true);
    const bare = page();
    await mountProviderReturn(bare.root, { href: href(signInState()), history, lifecycle: null,
        send: async () => ({ ok: true, json: async () => ({ ok: false, reason: 'answer_saved', error: 'kept' }) }) });
    assert.equal(bare.nodes.again.hidden, true, 'without the host receipt of an attempt there is nothing to finish');
});

test('consent repair keeps the three supported return destinations and their instance prefix', () => {
    const service = Buffer.from(`{"v":1.0,"sign_in":"календарь_${'界'.repeat(54)}",` +
        '"svc":"calendar","nonce":"nonce"}').toString('base64url');
    for (const [value, route, action] of [
        [state('почта_1'), '/api/extensions/%D0%BF%D0%BE%D1%87%D1%82%D0%B0_1/oauth/complete', undefined],
        [mcpState('work_docs'), '/api/mcp/sign-in', 'complete'],
        [service, '/api/sign-in', 'complete'],
    ]) {
        const request = completionRequest(href(value, 'code=synthetic&iss=https%3A%2F%2Fissuer.test'));
        assert.equal(request.url, `https://example.test/instance${route}`);
        assert.deepEqual(JSON.parse(request.init.body), { state: value, code: 'synthetic', iss: 'https://issuer.test',
            ...(action && { action }) });
    }
});
