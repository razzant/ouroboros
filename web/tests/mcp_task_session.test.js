import assert from 'node:assert/strict';
import test from 'node:test';
import { applyMcpSettings, collectMcpSettings, initMcpSettings } from '../modules/mcp_settings.js';

/** Exercise the real settings renderer and handlers; real layout is checked in the browser lane. */
function mount(t, server) {
    const previous = { document: globalThis.document, fetch: globalThis.fetch };
    t.after(() => Object.assign(globalThis, previous));
    const view = { changes: 0, requests: [], markup: '', field: null, note: null };
    const card = {
        dataset: { mcpIndex: '0' },
        querySelectorAll: (selector) => selector === '[data-mcp-field]' && view.field ? [view.field] : [],
        querySelector: (selector) => ({
            '[data-mcp-lifetime]': view.note,
            '[data-mcp-field="task_session"]': view.field,
        })[selector] || null,
    };
    const host = {
        get innerHTML() { return view.markup; },
        set innerHTML(value) {
            view.markup = value;
            const tag = value.match(/<input\b[^>]*data-mcp-field="task_session"[^>]*>/)?.[0];
            view.note = { textContent: '', hidden: false };
            view.field = tag ? {
                dataset: { mcpField: 'task_session' }, checked: /\bchecked\b/.test(tag),
                disabled: /\bdisabled\b/.test(tag), listeners: {},
                addEventListener(type, handler) { this.listeners[type] = handler; },
                emit(type) { this.listeners[type](); },
            } : null;
        },
        querySelectorAll: (selector) => selector === '[data-mcp-card]' ? [card] : [],
    };
    globalThis.document = {
        getElementById: (id) => id === 'mcp-servers-list' ? host : null,
        querySelector: () => card,
    };
    globalThis.fetch = async (url, options) => {
        view.requests.push({ url, options });
        return { ok: false };
    };
    initMcpSettings({ onChange: () => { view.changes += 1; } });
    applyMcpSettings({ MCP_ENABLED: true, MCP_SERVERS: [server] });
    view.saved = () => collectMcpSettings().MCP_SERVERS[0];
    return view;
}

const local = () => ({ id: 'files', enabled: true, transport: 'stdio', command: 'node',
    args: ['server.mjs'], env_from_settings: { TOKEN: 'CUSTOM_KEY' }, allowed_tools: ['read'] });

test('an existing stdio server opts in and out through its draft without saving or remounting', (t) => {
    const server = local();
    const view = mount(t, server);
    const field = view.field;
    assert.equal(field.checked, false);
    assert.equal(field.disabled, false);
    assert.equal(Object.hasOwn(view.saved(), 'task_session'), false);
    assert.match(view.markup, /aria-describedby="mcp-0-lifetime"/);
    assert.match(view.markup, /Off: each tool call starts a new server process/);
    for (const [enabled, event] of [[true, 'input'], [false, 'change']]) {
        field.checked = enabled;
        field.emit(event);
        assert.equal(view.saved().task_session, enabled);
        assert.equal(view.field, field); // the focused control keeps its node
        assert.match(view.note.textContent, enabled ? /calls share one running server process/ : /each tool call/);
        assert.deepEqual(view.saved().env_from_settings, server.env_from_settings);
        assert.deepEqual(view.saved().allowed_tools, server.allowed_tools);
    }
    assert.equal(view.changes, 2);
    assert.ok(view.requests.every(({ url }) => url === '/api/mcp/status')); // the section's Save is the only writer
});

test('an imported browser policy keeps its required lifetime and explains the disabled control', (t) => {
    const view = mount(t, { ...local(), task_session: 'TRUE', browser_policy: 'yes' });
    assert.equal(view.field.checked, true);
    assert.equal(view.field.disabled, true);
    assert.match(view.markup, /requires a task session; use Import to change that policy explicitly/);
    for (const event of ['input', 'change']) {
        view.field.checked = false;
        view.field.emit(event); // a stale or synthetic event cannot drop the separately selected policy
        assert.equal(view.field.checked, true);
        assert.equal(view.saved().task_session, 'TRUE');
        assert.equal(view.saved().browser_policy, 'yes');
    }
    assert.equal(view.changes, 0);
});

test('a remote transport never offers the stdio-only lifetime switch', (t) => {
    const view = mount(t, { id: 'remote', transport: 'sse', url: 'https://example.test/mcp' });
    assert.equal(view.field, null);
    assert.equal(Object.hasOwn(view.saved(), 'task_session'), false);
    assert.ok(!view.markup.includes('data-mcp-lifetime'));
});
