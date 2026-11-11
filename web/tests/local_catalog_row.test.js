// The catalog's Local row (ouroboros/gateway/models.py `_local_model_entry`) is a model with no route: `use_local`
// reaches it, never a spelling a setting could store. So only the composer's Model capsule, which saves
// `{model, use_local}`, may offer it. Every other chooser stores what it suggests as a route string: the Models
// role chooser as Settings, the wait picker and the first-run wizard each configure it (all three are
// createModelRolesEditor fed through adoptCatalog), the Available-subagents model field, and the place sheet's and
// the circle editor's Model field. Each is given a catalog that holds the row and must offer nothing for it, while
// still offering its own source's models.
import assert from 'node:assert/strict';
import test from 'node:test';
import { buildModelIndex, searchModels } from '../modules/chat_model_routes.js';
import { createModelCatalogStore } from '../modules/model_catalog_store.js';
import { createModelRolesEditor } from '../modules/model_roles.js';
import { modelSuggestions } from '../modules/place_sheet.js';
import { routeModelInputHtml, routeModelSuggestions } from '../modules/route_editor_primitives.js';
import { mountPair } from './composer_controls_fixture.js';

// The row exactly as the server writes it for LOCAL_MODEL_SOURCE=~/models/qwen3.gguf.
const LOCAL = { provider_id: 'local', provider: 'Local', source: 'Local', id: 'qwen3.gguf', name: 'qwen3.gguf',
    value: 'local::qwen3.gguf', label: 'Local · qwen3.gguf', use_local: true, context_window: 32768 };
const codex = (model) => ({ provider_id: 'claudexor', provider: 'Codex', source: 'Claudexor', source_id: 'codex', id: model, name: model,
    value: `claudexor::codex=${model}`, label: `Claudexor · ${model}`, credential_profile_id: 'home' });
const CATALOG = {
    model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }],
    items: [
        codex('gpt-6-astra'),
        { provider_id: 'openrouter', provider: 'Z.ai (GLM)', source: 'OpenRouter', id: 'z-ai/glm-5.3', name: 'GLM 5.3', value: 'z-ai/glm-5.3', label: 'OpenRouter · GLM 5.3' },
        { provider_id: 'openai', provider: 'OpenAI', source: 'OpenAI', id: 'gpt-6', name: 'gpt-6', value: 'openai::gpt-6', label: 'OpenAI · gpt-6' },
        { provider_id: 'openai-compatible', provider: 'OpenAI Compatible', source: 'OpenAI Compatible', id: 'served', name: 'served', value: 'openai-compatible::served', label: 'OpenAI Compatible · served' },
        LOCAL,
    ],
    errors: [],
};
const isLocal = (text) => /qwen3|local::|\bLocal\b/.test(text);

/* The editor rendered into a host that keeps each role row's real chooser: the editor binds the field and fills it
   through model_chooser.js as in a page, and focusing the field shows what it offers. */
function rolesEditor({ settings, slots, options = {}, adopt = (editor, catalog) => editor.adoptCatalog(catalog) }) {
    const inert = () => new Proxy(function () {}, {
        get: (target, key) => (key === Symbol.iterator ? function* () {} : key === Symbol.toPrimitive ? () => ''
            : key === 'then' ? undefined : key in target ? target[key] : inert()),
        set: (target, key, value) => { target[key] = value; return true; },
        apply: () => inert(),
    });
    const win = { addEventListener() {}, removeEventListener() {} };
    const lists = new Map();
    const fields = new Map();  // role row id → its chooser's field
    const doc = { defaultView: win, addEventListener() {}, removeEventListener() {},
        body: { appendChild(node) { node.parentElement = doc.body; } },
        getElementById: (id) => (id === 'roles' ? host : lists.get(id) || inert()) };
    const field = (rowId, listId) => {
        const attrs = new Map([['aria-controls', listId]]);
        const listeners = new Map();
        // The list and its non-option footer share the chooser's portalled popup.
        const footer = { hidden: true, innerHTML: '', contains: () => false, addEventListener() {}, removeEventListener() {} };
        const owner = { querySelector: (selector) => selector === '[data-model-footer]' ? footer : null,
            appendChild(node) { node.parentElement = owner; } };
        const popup = { ownerDocument: doc, parentElement: owner, hidden: true, isConnected: false,
            addEventListener() {}, removeEventListener() {}, contains: (node) => node === list || node === footer };
        const list = { id: listId, ownerDocument: doc, parentElement: popup, innerHTML: '', closest: () => popup,
            setAttribute() {}, querySelectorAll: () => [], addEventListener() {}, removeEventListener() {}, contains: () => false };
        lists.set(listId, list);
        return { rowId, list, listeners, ownerDocument: doc, value: '', disabled: false, dataset: {}, closest: () => null,
            getAttribute: (key) => attrs.get(key) ?? null, setAttribute: (key, value) => attrs.set(key, value),
            removeAttribute: (key) => attrs.delete(key), addEventListener: (type, fn) => listeners.set(type, fn),
            removeEventListener() {}, focus() {}, dispatchEvent() {} };
    };
    const host = new Proxy(function () {}, {
        get(target, key) {
            if (key === 'querySelectorAll') return (selector) => (selector === '[data-model-chooser]' ? [...fields.values()] : []);
            if (key === 'querySelector') return (selector) => {
                const rowId = /^\[data-model-role="([^"]+)"\]$/.exec(selector)?.[1];
                if (!rowId) return selector === '[data-global-processing]' ? null : inert();
                const row = inert();
                return new Proxy(row, { get: (_row, name) => (name === 'querySelector'
                    ? (inner) => (inner === '[data-model-role-model]' ? fields.get(rowId) : inert()) : row[name]) });
            };
            return key in target ? target[key] : inert();
        },
        set(target, key, value) {
            if (key === 'innerHTML') {
                // A new paint replaces the rows: each row's field is the one its markup names.
                fields.clear();
                for (const [, rowId, rest] of String(value).matchAll(/data-model-role="([^"]+)"([\s\S]*?)(?=data-model-role="|$)/g)) {
                    const listId = /aria-controls="([^"]+)"/.exec(rest)?.[1];
                    if (listId) fields.set(rowId, field(rowId, listId));
                }
            }
            target[key] = value;
            return true;
        },
    });
    const editor = createModelRolesEditor({ hostId: 'roles', doc: () => doc, store: { snapshot: {}, accountsKnown: false },
        // The source-scoped read answers as the server does: that subscription's own models, nothing else.
        catalogFetch: async () => ({ items: CATALOG.items.filter((item) => item.source_id === 'codex'), model_sources: CATALOG.model_sources }),
        ...options });
    editor.load(settings, { providerProfiles: {}, modelSlots: slots });
    adopt(editor, CATALOG);
    return {
        editor,
        /** What each role's chooser offers once its field is focused: `[value, label]` per option. */
        async offered() {
            await new Promise((resolve) => setImmediate(resolve));
            const out = {};
            for (const [rowId, made] of fields) {
                made.listeners.get('focus')();
                out[rowId] = [...made.list.innerHTML.matchAll(/data-model-value="([^"]*)" aria-selected="false">([^<]*)</g)].map(([, value, label]) => [value, label]);
            }
            editor.destroy();
            return out;
        },
    };
}
const slot = (name, key, extra = {}) => ({ slot: name, settingKey: key, inputId: `settings-${name}`, label: `${name} Model`, ...extra });
const KEYS = { OPENROUTER_API_KEY: 'k', OPENAI_API_KEY: 'k', OPENAI_COMPATIBLE_BASE_URL: 'http://127.0.0.1:1/v1' };
const SURFACES = {
    // Settings › Models: every role on its own source, a fallback list, a role left to inherit.
    'Settings': () => rolesEditor({ options: { mainNote: true },
        settings: { ...KEYS, OUROBOROS_MODEL: 'z-ai/glm-5.3', OUROBOROS_MODEL_LIGHT: 'openai::gpt-6', OUROBOROS_MODEL_CONSCIOUSNESS: 'claudexor::codex=gpt-6-astra',
            OUROBOROS_MODEL_FALLBACKS: 'openai-compatible::served, z-ai/glm-5.3', OUROBOROS_MODEL_CODE: '', USE_LOCAL_MAIN: true },
        slots: [slot('main', 'OUROBOROS_MODEL', { settingsToggleId: 'local-main' }), slot('light', 'OUROBOROS_MODEL_LIGHT'),
            slot('consciousness', 'OUROBOROS_MODEL_CONSCIOUSNESS'), slot('code', 'OUROBOROS_MODEL_CODE'), slot('fallback', 'OUROBOROS_MODEL_FALLBACKS')] }),
    // The wait card's picker: one slot keyed `model`, no context details (model_wait.js openPicker).
    'the wait picker': () => rolesEditor({ options: { showContext: false }, settings: { ...KEYS, model: 'z-ai/glm-5.3', OUROBOROS_MODEL_ACCOUNTS: { main: '' } },
        slots: [{ slot: 'main', settingKey: 'model', inputId: 'wait-model', settingsToggleId: 'wait-local', label: 'Main Model' }] }),
    // The first-run wizard: the same editor fed its merged catalog with the unsaved compatible server's own list.
    'the wizard': () => rolesEditor({ settings: { ...KEYS, OUROBOROS_MODEL: 'openai-compatible::served', OUROBOROS_MODEL_LIGHT: 'z-ai/glm-5.3' },
        slots: [slot('main', 'OUROBOROS_MODEL', { settingsToggleId: 'local-main' }), slot('light', 'OUROBOROS_MODEL_LIGHT')],
        adopt: (editor, catalog) => editor.adoptCatalog(catalog, { connectionCatalogs: { 'openai-compatible': catalog.items.filter((item) => item.provider_id === 'openai-compatible') } }) }),
};

for (const [surface, make] of Object.entries(SURFACES)) {
    test(`${surface}: a catalog that holds the Local row offers each role its own source's models and nothing for the row`, async () => {
        const offered = await make().offered();
        const rows = Object.entries(offered);
        assert.ok(rows.length, 'the role rows have their choosers');
        // The working direction: the catalog did arrive, each source lists its own.
        assert.ok(rows.some(([, options]) => options.some(([value]) => value === 'z-ai/glm-5.3')), 'OpenRouter\'s model is offered');
        for (const [row, options] of rows) {
            assert.deepEqual(options.filter(([value, label]) => isLocal(value) || isLocal(label)), [], `${surface} · ${row}`);
        }
    });
}

test('Settings: each source\'s role is offered exactly that source\'s catalog', async () => {
    const offered = await SURFACES.Settings().offered();
    assert.deepEqual(offered.main, [['z-ai/glm-5.3', 'GLM 5.3']]);
    assert.deepEqual(offered.light, [['gpt-6', 'gpt-6']]);
    assert.deepEqual(offered.consciousness, [['gpt-6-astra', 'gpt-6-astra']]);
    assert.deepEqual(offered.code, [], 'a role left to inherit has no source of its own to list');
    assert.deepEqual(Object.entries(offered).filter(([row]) => row.startsWith('fallback')).map(([, options]) => options),
        [[['served', 'served']], [['z-ai/glm-5.3', 'GLM 5.3']]]);
});

test('Available subagents: an API or a subscription row\'s model field offers its source\'s models, never the Local row', () => {
    for (const target of ['z-ai/glm-5.3', 'openai::gpt-6', 'openai-compatible::served', 'claudexor::codex=gpt-6-astra', '', 'deepseek::']) {
        const route = { kind: 'api_model', target_id: target };
        const html = routeModelInputHtml('', route, CATALOG.items, 'list');
        assert.ok(!isLocal(html.replace(/placeholder="[^"]*"/, '')), target || 'an empty OpenRouter row');
        assert.deepEqual(routeModelSuggestions(route, CATALOG.items).filter(isLocal), []);
    }
    assert.deepEqual(routeModelSuggestions({ kind: 'api_model', target_id: '' }, CATALOG.items), ['z-ai/glm-5.3']);
});

test('a place\'s and a circle\'s Model field stores a route string: their suggestions leave the Local row out', () => {
    const offered = modelSuggestions(CATALOG);
    assert.equal(offered.length, CATALOG.items.length - 1);
    assert.ok(!offered.some((item) => item.use_local || isLocal(item.value)));
    assert.throws(() => modelSuggestions({ read_state: 'failed', items: [] }));
});

test('the Model capsule alone offers it, as the artifact with use_local and never as the label', async () => {
    const index = buildModelIndex({ catalog: CATALOG });
    assert.deepEqual(index.sources.map((source) => source.id), ['subscription:codex', 'openrouter', 'openai', 'openai-compatible', 'local']);
    assert.deepEqual(searchModels(index, 'qwen').map((row) => row.route), [{ model: 'qwen3.gguf', use_local: true }]);
    // And through the capsule itself: Search, the Local source, the row, the save.
    const store = createModelCatalogStore({ readCatalog: async () => CATALOG, readPreferences: async () => ({}) });
    const m = mountPair({ projectId: 'p7', chatId: 7, model: { store } });
    m.controls.syncState({ main_route: { model: 'claudexor::codex=gpt-6-astra', use_local: false }, projects: [{ id: 'p7', model_choice: { kind: 'main' } }] });
    await m.press(m.modelHead);
    await new Promise((resolve) => setImmediate(resolve));
    m.controls.model.startSearch();
    const list = m.doc.body.children.find((node) => node.classList.contains('chat-model-list'));
    const row = (key) => list.querySelectorAll('.chat-model-row').find((node) => node.dataset.row === key);
    assert.match(list.innerHTML, /data-row="source:local"[^>]*>.*?Local<\/span><span class="chat-model-row-facts" data-i18n-skip>1 model · on this computer</);
    await m.fire(list, 'click', { target: { closest: () => row('source:local') } });
    assert.match(list.innerHTML, /data-row="local\|qwen3\.gguf"[^>]*>.*?qwen3\.gguf<\/span><span class="chat-model-row-facts" data-i18n-skip>33K · Local · on this computer</);
    await m.fire(list, 'click', { target: { closest: () => row('local|qwen3.gguf') } });
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(m.posts.at(-1), ['/api/owner/chat-model', { kind: 'model', model: 'qwen3.gguf', use_local: true, project_id: 'p7' }]);
    m.controls.destroy();
});
