import assert from 'node:assert/strict';
import test from 'node:test';
import { apiClient } from '../modules/api_client.js';
import { bindSecretReveal, resetSecretReveals } from '../modules/settings_secrets.js';

class Node {
    constructor(doc) {
        this.ownerDocument = doc;
        this.dataset = {};
        this.attributes = new Map();
        this.listeners = new Map();
        this.children = [];
        this.isConnected = true;
        this.textContent = '';
        this.hidden = false;
    }
    set innerHTML(_value) { throw new Error('Secret disclosure must not parse HTML'); }
    setAttribute(key, value) { this.attributes.set(key, String(value)); }
    getAttribute(key) { return this.attributes.get(key) ?? null; }
    removeAttribute(key) { this.attributes.delete(key); }
    addEventListener(type, handler) {
        if (!this.listeners.has(type)) this.listeners.set(type, []);
        this.listeners.get(type).push(handler);
    }
    async emit(type) { await Promise.all((this.listeners.get(type) || []).map((handler) => handler())); }
    append(...nodes) { this.children.push(...nodes); }
    after(...nodes) { this.following = nodes; }
}

function fixture({ value = 'sk-ant-a...', applied = value, savedSelector = () => ({ key: 'ANTHROPIC_API_KEY' }), ...options } = {}) {
    const doc = { createElement: () => new Node(doc) };
    const row = new Node(doc);
    const label = new Node(doc);
    const input = new Node(doc);
    input.id = 'test-secret';
    input.value = value;
    input.type = 'password';
    input.dataset.appliedValue = applied;
    input.labels = [label];
    input.closest = () => row;
    const button = new Node(doc);
    const root = { querySelectorAll: () => [input] };
    const controller = bindSecretReveal(input, button, { savedSelector, ...options });
    const [source, status] = row.children;
    const [preview] = input.following;
    return { input, button, root, label, source, preview, status, controller,
        snapshot: () => JSON.stringify({ value: input.value, type: input.type, dataset: input.dataset }) };
}

const longValue = 'sk-ant-api03-' + 'A1b2'.repeat(35) + '-distinct-tail<&"\nsecond line';

test('selected saved reveal is complete plain text and leaves the draft byte-identical', async (t) => {
    const selectors = [];
    t.mock.method(apiClient, 'revealSettingsSecret', async (selector) => {
        selectors.push(selector);
        return { value: longValue };
    });
    const f = fixture();
    const before = f.snapshot();
    await f.button.emit('click');
    assert.deepEqual(selectors, [{ key: 'ANTHROPIC_API_KEY' }]);
    assert.equal(f.preview.textContent, longValue);
    assert.equal(f.preview.hidden, false);
    assert.equal(f.input.hidden, true, 'the value stands in the masked input\'s own place');
    assert.equal(f.snapshot(), before);
    assert.equal(f.button.textContent, 'Hide');
    assert.equal(f.button.getAttribute('aria-expanded'), 'true');
    assert.equal(f.button.getAttribute('aria-controls'), f.preview.id);
    assert.equal(f.preview.getAttribute('aria-labelledby'), f.label.id);
    assert.equal(f.preview.tabIndex, 0);
    await f.button.emit('click');
    assert.equal(f.preview.textContent, '');
    assert.equal(f.preview.hidden, true);
    assert.equal(f.input.hidden, false);
    assert.equal(f.button.getAttribute('aria-expanded'), 'false');
    assert.equal(f.snapshot(), before);
    assert.equal(selectors.length, 1);
});

test('new, changed, empty and cleared drafts use the same full preview without reading an old value', async (t) => {
    const read = t.mock.method(apiClient, 'revealSettingsSecret', async () => { throw new Error('unexpected read'); });
    for (const [value, applied, clear] of [[longValue, '', false], [longValue, 'sk-ant-a...', false], ['', 'sk-ant-a...', true], ['', '', false]]) {
        const f = fixture({ value, applied });
        if (clear) f.input.dataset.forceClear = '1';
        const before = f.snapshot();
        await f.button.emit('click');
        assert.equal(f.preview.textContent, value);
        assert.equal(f.preview.hidden, false);
        await f.button.emit('click');
        assert.equal(f.snapshot(), before);
    }
    assert.equal(read.mock.callCount(), 0);
});

test('a renamed field reads its saved identity and names that source without copying it into the draft', async (t) => {
    const identity = new Node();
    identity.value = 'RENAMED_KEY';
    const selectors = [];
    t.mock.method(apiClient, 'revealSettingsSecret', async (selector) => { selectors.push(selector); return { value: longValue }; });
    const f = fixture({ savedSelector: () => ({ key: 'ORIGINAL_KEY' }),
        savedLabel: () => 'Saved value for ORIGINAL_KEY', identityInputs: [identity] });
    const before = f.snapshot();
    await f.button.emit('click');
    assert.deepEqual(selectors, [{ key: 'ORIGINAL_KEY' }]);
    assert.equal(f.source.textContent, 'Saved value for ORIGINAL_KEY');
    assert.equal(f.preview.getAttribute('aria-describedby'), f.source.id);
    assert.equal(f.snapshot(), before);
    identity.value = 'ANOTHER_KEY';
    await identity.emit('input');
    assert.equal(f.preview.textContent, '');
    assert.equal(f.source.textContent, '');
});

for (const action of ['hide', 'type', 'clear', 'identity', 'reload', 'detach']) {
    test(`a late saved response cannot restore disclosure after ${action}`, async (t) => {
        let resolve;
        const identity = new Node();
        identity.value = 'saved-server';
        t.mock.method(apiClient, 'revealSettingsSecret', () => new Promise((done) => { resolve = done; }));
        const f = fixture({ savedSelector: () => ({ mcp_server_id: 'saved-server' }), identityInputs: [identity] });
        const pending = f.button.emit('click');
        assert.equal(f.button.getAttribute('aria-busy'), 'true');
        assert.equal(f.input.hidden, false, 'a pending read keeps the masked input in place');
        if (action === 'hide') await f.button.emit('click');
        if (action === 'type' || action === 'clear') {
            f.input.value = action === 'type' ? 'newly typed value' : '';
            await f.input.emit('input');
        }
        if (action === 'identity') { identity.value = 'another-server'; await identity.emit('input'); }
        if (action === 'reload') resetSecretReveals(f.root);
        if (action === 'detach') f.input.isConnected = false;
        resolve({ value: longValue });
        await pending;
        assert.equal(f.preview.textContent, '');
        assert.equal(f.preview.hidden, true);
        assert.equal(f.input.hidden, false);
        assert.notEqual(f.input.value, longValue);
        if (action !== 'detach') {
            assert.equal(f.button.textContent, 'Show');
            assert.equal(f.status.textContent, '');
        }
    });
}

test('a superseded response cannot replace the newly requested value', async (t) => {
    const replies = [];
    t.mock.method(apiClient, 'revealSettingsSecret', () => new Promise((done) => replies.push(done)));
    const f = fixture();
    const first = f.button.emit('click');
    await f.button.emit('click');
    const second = f.button.emit('click');
    replies[1]({ value: 'new saved value' });
    await second;
    replies[0]({ value: 'stale saved value' });
    await first;
    assert.equal(f.preview.textContent, 'new saved value');
    assert.equal(f.button.textContent, 'Hide');
});

test('failure and malformed success remain retryable, never a partial Show or a changed credential', async (t) => {
    const outcomes = [new Error('Synthetic read failure'), { error: 'not a value' }, { value: longValue }];
    t.mock.method(apiClient, 'revealSettingsSecret', async () => {
        const next = outcomes.shift();
        if (next instanceof Error) throw next;
        return next;
    });
    const f = fixture();
    const before = f.snapshot();
    for (let attempt = 0; attempt < 2; attempt += 1) {
        await f.button.emit('click');
        assert.equal(f.preview.textContent, '');
        assert.equal(f.input.hidden, false);
        assert.equal(f.button.textContent, 'Show');
        assert.equal(f.status.hidden, false);
        assert.match(f.status.textContent, /Could not show this value/);
        assert.equal(f.button.getAttribute('aria-busy'), null);
        assert.equal(f.snapshot(), before);
    }
    await f.button.emit('click');
    assert.equal(f.preview.textContent, longValue);
    assert.equal(f.status.hidden, true);
});

test('gateway client sends only the selected identifier to the read endpoint', async (t) => {
    const requests = [];
    t.mock.method(globalThis, 'fetch', async (url, init) => {
        requests.push({ url, method: init.method, body: JSON.parse(init.body) });
        return { ok: true, json: async () => ({ value: longValue }) };
    });
    for (const selector of [{ key: 'CUSTOM_KEY' }, { mcp_server_id: 'saved-server' }]) {
        assert.deepEqual(await apiClient.revealSettingsSecret(selector), { value: longValue });
        assert.deepEqual(requests.at(-1), { url: '/api/settings/secret', method: 'POST', body: selector });
    }
});
