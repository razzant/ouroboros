import assert from 'node:assert/strict';
import test from 'node:test';
import { completionRequest, mountProviderReturn } from '../modules/provider_return.js';

const state = (skill = 'calendar', extra = {}) => Buffer.from(JSON.stringify({ v: 1, skill, nonce: 'nonce', ...extra })).toString('base64url');
const href = (value = state(), query = 'code=synthetic') => `https://example.test/instance/onboarding/static/provider-return.html?${query}&state=${value}`;

function page() {
    const nodes = Object.fromEntries(['status', 'detail', 'account', 'home', 'next', 'again', 'technical', 'said'].map((name) => [name, {
        textContent: '', hidden: true, dataset: {}, setAttribute(key, value) { this[key] = value; }, removeAttribute(key) { delete this[key]; },
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

test('an MCP return stopped before anything changed at the provider is sent again only on the owner\'s press, from memory', async () => {
    // The host's answer as published: HTTP 200 with ok false (docs/CLIENT_CONTRACT.md "MCP servers").
    const stopped = { ok: false, server: 'bank', code: 'MCP_UNAVAILABLE', reason: 'not_sent',
        error: 'Sign-in stopped before anything changed at the provider: the Extra certificate bundle can\'t be used. Fix the file under Extra certificate bundle in Settings → This computer → Advanced, then try again.',
        detail: 'OUROBOROS_EXTRA_CA_BUNDLE is not readable: /Users/synthetic-owner/bank-root.pem ([Errno 2] No such file or directory)' };
    const { root, nodes } = page();
    const sent = [];
    const replies = [stopped, stopped, { ok: true, server: 'bank', scope: 'mcp' }];
    let release = null;
    let held = null;
    const hosted = [];
    await mountProviderReturn(root, { href: href(mcpState('bank')), history: { replaceState() {} }, lifecycle: null,
        translateHost: (text) => { hosted.push(text); return text; },
        send: async (_url, init) => {
            sent.push(init.body);
            const body = replies.shift();
            if (held) await held;
            return { ok: true, json: async () => body };
        } });
    assert.equal(sent.length, 1, 'no automatic second send');
    assert.equal(nodes.status.textContent, 'Sign-in not finished yet');
    assert.equal(nodes.detail.textContent, stopped.error, 'the host\'s fixed words, translated as a host sentence');
    assert.deepEqual(hosted, [stopped.error]);
    assert.equal(nodes.detail['data-i18n-authored'], undefined);
    assert.ok(!nodes.detail.textContent.includes('synthetic-owner'), 'the path stays out of the words');
    assert.deepEqual([nodes.technical.hidden, nodes.said.textContent], [false, stopped.detail], 'behind Details, never translated');
    assert.deepEqual([nodes.next.hidden, nodes.again.hidden], [false, false]);
    assert.match(nodes.next.textContent, /Keep this tab open.*press Finish sign-in here/);
    // The repair opens another tab of this Ouroboros; this tab, with the return in its memory, stays.
    assert.deepEqual([nodes.home.href, nodes.home.target, nodes.home.rel, nodes.home.textContent],
        ['https://example.test/instance/#settings', '_blank', 'noopener', 'Open Settings in a new tab']);

    held = new Promise((resolve) => { release = resolve; });
    const pressed = nodes.again.handlers.click();
    await Promise.resolve();
    assert.deepEqual([nodes.status.textContent, nodes.status.dataset.tone, nodes.detail.textContent, nodes.next.hidden, nodes.again.hidden],
        ['Completing sign-in…', 'muted', 'Waiting for the connector to confirm this account.', true, true], 'in flight: the earlier refusal is gone');
    assert.equal(nodes.technical.hidden, true);
    assert.equal(nodes.detail['data-i18n-authored'], undefined);
    assert.deepEqual([nodes.home.href, nodes.home.target, nodes.home.textContent],
        ['https://example.test/instance/', undefined, 'Return to Ouroboros']);
    release();
    await pressed;
    held = null;
    assert.equal(sent.length, 2);
    assert.equal(nodes.again.hidden, false, 'still stopped: Finish stays');
    await nodes.again.handlers.click();
    assert.deepEqual([sent.length, sent[1], sent[2]], [3, sent[0], sent[0]], 'the same return, from memory');
    assert.equal(nodes.status.textContent, 'Account connected');
    assert.equal(nodes.account.textContent, 'bank');
    assert.equal(nodes.again.hidden, true);
    assert.equal(nodes.detail['data-i18n-authored'], undefined, 'the confirmation is the page\'s own words');
    assert.deepEqual([nodes.home.href, nodes.home.target], ['https://example.test/instance/', undefined]);

    for (const [value, reply] of [[state(), stopped], [signInState(), stopped],
        [mcpState('bank'), { ...stopped, reason: 'token_rejected' }], [mcpState('bank'), { ...stopped, error: undefined }]]) {
        const other = page();
        let count = 0;
        await mountProviderReturn(other.root, { href: href(value), history: null, lifecycle: null,
            send: async () => { count += 1; return { ok: true, json: async () => reply }; } });
        assert.equal(count, 1);
        assert.equal(other.nodes.again.hidden, true, 'only a typed MCP no-send with its words is offered again');
        assert.equal(other.nodes.status.textContent, 'Could not finish connecting');
        assert.equal(other.nodes.home.target, undefined, 'any other failure keeps the way home in this tab');
    }
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
