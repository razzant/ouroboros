import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { apiClient } from '../modules/api_client.js';
import {
    HEADER_PLACEHOLDER as MASK, bindHeaderFields, canonicalServerId, credentialPayload, headerDraftRows, headerImportFacts,
    headerKeepsSavedValues, keptHeaderNames, renderHeaderFields, serverIdentity, validateHeaderFields,
} from '../modules/mcp_header_fields.js';
import { endpointChange } from '../modules/mcp_import_dialog.js';

class El {
    constructor(doc, value = '') {
        this.ownerDocument = doc;
        this.dataset = {};
        this.attributes = new Map();
        this.listeners = new Map();
        this.children = [];
        this.isConnected = true;
        this.textContent = '';
        this.hidden = false;
        this.value = value;
    }
    setAttribute(key, value) { this.attributes.set(key, String(value)); }
    getAttribute(key) { return this.attributes.get(key) ?? null; }
    removeAttribute(key) { this.attributes.delete(key); }
    addEventListener(type, handler) {
        if (!this.listeners.has(type)) this.listeners.set(type, []);
        this.listeners.get(type).push(handler);
    }
    async emit(type) { for (const handler of [...(this.listeners.get(type) || [])]) await handler(); }
    append(...nodes) { this.children.push(...nodes); }
    after(...nodes) { this.following = nodes; }
    focus() { this.ownerDocument.focused = this; }
    scrollIntoView() {}
}

/** Mount one card's header rows the way renderHeaderFields lays them out; rerendering remounts. */
function mount(server, savedIdentity, { idValue = server.id } = {}) {
    const doc = { createElement: () => new El(doc) };
    const view = { server, doc, idInput: new El(doc, idValue), rows: [] };
    const build = () => {
        view.addButton = new El(doc);  // a rerender replaces the card's controls
        const rows = headerDraftRows(server).rows.map(([, name, value], number) => {
            const node = new El(doc);
            node.dataset.mcpHeaderRow = String(number);
            const fields = {
                name: new El(doc, name), value: new El(doc, value), toggle: new El(doc), clear: new El(doc),
                remove: new El(doc), secretRow: new El(doc), label: new El(doc),
            };
            fields.value.id = `v-${number}`;
            fields.value.labels = [fields.label];
            fields.value.closest = () => fields.secretRow;
            node.querySelector = (selector) => ({
                '[data-mcp-header-field="name"]': fields.name, '[data-mcp-header-field="value"]': fields.value,
                '[data-mcp-header-toggle]': fields.toggle, '[data-mcp-header-clear]': fields.clear,
                '[data-mcp-header-remove]': fields.remove,
            })[selector] ?? null;
            return { node, ...fields };
        });
        const card = {
            dataset: { mcpIndex: '0' },
            querySelectorAll: (selector) => (selector === '[data-mcp-header-row]' ? rows.map((row) => row.node) : []),
            querySelector: (selector) => {
                const match = /^\[data-mcp-header-row="(\d+)"\]$/.exec(selector);
                if (match) return rows[Number(match[1])]?.node ?? null;
                return { '[data-mcp-field="id"]': view.idInput, '[data-mcp-header-add]': view.addButton }[selector] ?? null;
            },
        };
        view.rows = rows;
        view.card = card;
        bindHeaderFields(card, { server, savedIdentity, onChange: () => {}, render: build });
    };
    globalThis.document = { querySelector: (selector) => (selector === '[data-mcp-index="0"]' ? view.card : null) };
    build();
    view.type = async (row, field, value) => { view.rows[row][field].value = value; await view.rows[row][field].emit('input'); };
    view.errors = () => validateHeaderFields(view.card, server, savedIdentity).map(({ message }) => message);
    return view;
}

const saved = () => ({
    id: 'docs', url: 'https://docs.example/mcp', auth_header: 'X-Legacy', auth_token: 'legacy-t...',
    headers: { 'X-A': MASK, 'X-B': MASK, 'X-Empty': '' },
});

test('an untouched card sends nothing new: the legacy pair and every mask stay as loaded', () => {
    const server = saved();
    const view = mount(server, 'docs');
    assert.deepEqual(view.rows.map((row) => [row.name.value, row.value.value]),
        [['X-Legacy', 'legacy-t...'], ['X-A', MASK], ['X-B', MASK], ['X-Empty', '']]);
    assert.deepEqual(credentialPayload(server), {});
    assert.deepEqual(view.errors(), []);
    assert.equal(headerKeepsSavedValues(server), true);
    assert.deepEqual(keptHeaderNames(server), ['X-Legacy', 'X-A', 'X-B']);
    assert.deepEqual(headerImportFacts(server), { header_names: ['X-Legacy', 'X-A', 'X-B', 'X-Empty'], legacy_header: 'X-Legacy' });
});

test('editing another row keeps the untouched legacy row in its own fields with its prefix mask', async () => {
    const server = saved();
    const view = mount(server, 'docs');
    await view.type(2, 'value', 'synthetic-new-b');
    assert.deepEqual(credentialPayload(server), {
        auth_header: 'X-Legacy', auth_token: 'legacy-t...',
        headers: { 'X-A': MASK, 'X-B': 'synthetic-new-b', 'X-Empty': '' },
    });
    assert.deepEqual(view.errors(), []);
});

test('a deliberately edited legacy row becomes a literal header and clears the legacy pair', async () => {
    const server = saved();
    const view = mount(server, 'docs');
    await view.type(0, 'value', 'Bearer synthetic-rotated');
    assert.deepEqual(credentialPayload(server), {
        auth_header: 'Authorization', auth_token: '',
        headers: { 'X-Legacy': 'Bearer synthetic-rotated', 'X-A': MASK, 'X-B': MASK, 'X-Empty': '' },
    });
    await view.rows[0].remove.emit('click');
    assert.deepEqual(credentialPayload(server), {
        auth_header: 'Authorization', auth_token: '', headers: { 'X-A': MASK, 'X-B': MASK, 'X-Empty': '' },
    });
});

test('a case-only rename keeps the saved value; another name asks for the value again', async () => {
    const server = saved();
    const view = mount(server, 'docs');
    await view.type(0, 'name', 'x-legacy');
    await view.type(1, 'name', 'x-a');
    assert.deepEqual(view.errors(), []);
    assert.deepEqual(credentialPayload(server), {
        auth_header: 'x-legacy', auth_token: 'legacy-t...', headers: { 'x-a': MASK, 'X-B': MASK, 'X-Empty': '' },
    });
    await view.type(0, 'name', 'X-Other');
    assert.match(view.errors()[0], /belongs to header X-Legacy/);
});

test('saved A and B: remove B, rename masked A to B — Save and Test refuse; Show labels the value as A', async (t) => {
    const reads = [];
    t.mock.method(apiClient, 'revealSettingsSecret', async (selector) => { reads.push(selector); return { value: 'alpha' }; });
    const server = { id: 'docs', url: 'https://docs.example/mcp', headers: { A: MASK, B: MASK } };
    const view = mount(server, 'docs');
    await view.rows[1].remove.emit('click');
    await view.type(0, 'name', 'B');
    assert.deepEqual(view.errors(), ['The saved value belongs to header A. Restore that name, or enter a new value.']);
    await view.rows[0].toggle.emit('click');
    const [source] = view.rows[0].secretRow.children;
    const [preview] = view.rows[0].value.following;
    assert.deepEqual(reads, [{ mcp_server_id: 'docs', header_name: 'A' }]);
    assert.equal(preview.textContent, 'alpha');
    assert.equal(source.textContent, 'Saved value for A');
    assert.equal(view.rows[0].value.value, MASK, 'Show never changes the draft');
});

test('Remove hands the caret to the header now in its place, else to Add', async () => {
    const view = mount({ id: 'docs', url: 'https://docs.example/mcp', headers: { A: MASK, B: MASK } }, 'docs');
    await view.rows[0].remove.emit('click');
    assert.equal(view.doc.focused, view.rows[0].name);
    assert.equal(view.rows[0].name.value, 'B');
    await view.rows[0].remove.emit('click');
    assert.equal(view.doc.focused, view.addButton);
});

test('a renamed server keeps no masked value, legacy token included', async () => {
    const server = saved();
    const view = mount(server, 'docs');
    server.id = 'renamed';
    const errors = view.errors();
    assert.equal(errors.length, 3);
    assert.ok(errors.every((message) => message.includes('belongs to server docs')));
});

test('server identity follows the backend canonical_server_id table exactly', () => {
    const table = JSON.parse(fs.readFileSync(new URL('./fixtures/mcp_server_identity_parity.json', import.meta.url), 'utf8'));
    for (const { raw, canonical } of table.canonical) assert.equal(canonicalServerId(raw), canonical, JSON.stringify(raw));
    for (const { entry, identity } of table.entries) {
        assert.equal(canonicalServerId(serverIdentity(entry)), identity, JSON.stringify(entry));
    }
});

test('a spelling of the same canonical server keeps every masked value; another server does not', async (t) => {
    const reads = [];
    t.mock.method(apiClient, 'revealSettingsSecret', async (selector) => { reads.push(selector); return { value: 'alpha' }; });
    for (const spelling of ['DOCS', ' Docs ', 'docs!', 'Docs']) {
        const server = saved();
        const view = mount(server, 'docs');
        server.id = spelling;
        assert.deepEqual(view.errors(), [], spelling);
        assert.equal(headerKeepsSavedValues(server), true);
        await view.rows[1].toggle.emit('click');
        const [source] = view.rows[1].secretRow.children;
        assert.equal(source.textContent, '', 'the same server needs no saved-identity label');
    }
    assert.deepEqual(reads.at(-1), { mcp_server_id: 'docs', header_name: 'X-A' });
    const named = { name: 'My Docs', url: 'https://d.example/mcp', headers: { 'X-A': MASK } };
    const view = mount(named, 'My Docs');
    named.name = 'my-docs';
    assert.deepEqual(view.errors(), []);
    named.id = 'other_docs';  // the explicit ID now decides, and it is another server
    assert.match(view.errors()[0], /belongs to server My Docs/);
});

test('re-typing the mask is a value edit to dirty tracking and import freshness', async () => {
    const server = saved();
    const view = mount(server, 'docs');
    const untouched = JSON.stringify(headerDraftRows(server));
    await view.type(1, 'value', 'synthetic-other');
    await view.type(1, 'value', MASK);
    assert.equal(view.rows[1].value.value, MASK);
    assert.notEqual(JSON.stringify(headerDraftRows(server)), untouched);
    assert.deepEqual(headerDraftRows(server).rows[1], ['map', 'X-A', MASK, true]);
    assert.deepEqual(view.errors(), ['***set*** is reserved for keeping a saved value. Enter the real value.']);
});

test('editing a saved legacy mask away and back refuses it instead of storing mask text', async () => {
    for (const mask of ['legacy-t...', '***']) {
        const server = { id: 'docs', transport: 'streamable_http', auth_header: 'X-Legacy', auth_token: mask };
        const view = mount(server, 'docs');
        assert.deepEqual(view.errors(), []);
        assert.deepEqual(credentialPayload(server), {});
        await view.type(0, 'value', 'synthetic-other');
        await view.type(0, 'value', mask);
        assert.match(view.errors()[0] || '', /saved token.*mask/);
        await view.type(0, 'value', 'synthetic-replacement');
        assert.deepEqual(view.errors(), []);
        assert.deepEqual(credentialPayload(server), { auth_header: 'Authorization', auth_token: '',
            headers: { 'X-Legacy': 'synthetic-replacement' } });
    }
});

test('protocol-header help states what each transport actually sets', () => {
    const http = renderHeaderFields({ id: 'h', transport: 'streamable_http', headers: {} }, 0);
    const sse = renderHeaderFields({ id: 's', transport: 'sse', headers: {} }, 1);
    assert.match(http, /Streamable HTTP sets Accept and Content-Type on every request/);
    assert.match(sse, /SSE sets Accept and Cache-Control on its event stream; message requests send every row as written, so a Content-Type row replaces application\/json/);
    assert.doesNotMatch(sse, /Streamable HTTP/);
});

test('a typed keep marker is a field error, even after the loaded mask was edited away and back', async () => {
    const server = saved();
    const view = mount(server, 'docs');
    await view.type(1, 'value', 'synthetic-other');
    await view.type(1, 'value', MASK);
    assert.deepEqual(view.errors(), ['***set*** is reserved for keeping a saved value. Enter the real value.']);
    assert.equal(view.rows[1].value.dataset.appliedValue, '', 'a typed marker is shown as typed, never read as saved');
});

test('exact duplicate empty rows stay visible to dirty tracking and fail validation', async () => {
    const server = { id: 'dup', url: 'https://d.example/mcp', headers: { 'X-Empty': '' } };
    const view = mount(server, 'dup');
    const before = JSON.stringify(headerDraftRows(server));
    await view.addButton.emit('click');
    await view.type(1, 'name', 'X-Empty');
    // The saved map would collapse the pair into one key; the raw rows keep both.
    assert.deepEqual(credentialPayload(server).headers, { 'X-Empty': '' });
    assert.deepEqual(headerDraftRows(server).rows.map(([, name, value]) => [name, value]), [['X-Empty', ''], ['X-Empty', '']]);
    assert.notEqual(JSON.stringify(headerDraftRows(server)), before);
    assert.deepEqual(view.errors(), ['This name repeats header X-Empty; keep one.']);
});

test('address changes are disclosed as origins only', () => {
    assert.deepEqual(endpointChange('https://***@a.example/mcp?key=1', 'https://b.example:8443/other?key=2'),
        { from: 'https://a.example', to: 'https://b.example:8443', sameOrigin: false });
    assert.deepEqual(endpointChange('https://a.example/v1', 'https://a.example/v2'),
        { from: 'https://a.example', to: 'https://a.example', sameOrigin: true });
    assert.equal(endpointChange('https://a.example/v1', 'https://a.example/v1'), null);
    assert.equal(endpointChange('https://a.example/v1', 'not an address'), null);
    assert.deepEqual(endpointChange('http://a.example/mcp', 'https://a.example/mcp'),
        { from: 'http://a.example', to: 'https://a.example', sameOrigin: false });
});
