// The catalog's published price beside the model wherever the owner chooses one, as one short fact (owner, 12.11:
// compact, no disclaimers). The server keeps OpenRouter's price on its rows (ouroboros/gateway/models.py `price`);
// route_editor_primitives.js says it in words, the Settings chooser shows it beside the name (model_chooser.js
// `meta`), the line under a role says it per 1M tokens (model_roles.js) and the composer's list says it where
// "paid by key" stood (model_list.js). A row without a price reads as before; a price never filters a model.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { buildModelIndex, describeRoute } from '../modules/chat_model_routes.js';
import { bindModelChooser, matchingModelOptions, modelChooserHtml, modelOptions, updateModelChooserOptions } from '../modules/model_chooser.js';
import { createModelList } from '../modules/model_list.js';
import { createModelRolesEditor } from '../modules/model_roles.js';
import { applyPayload, CODE_PREFIX, setMissTransport } from '../modules/i18n.js';
import { catalogModelOptions, priceWords, routeModelInputHtml } from '../modules/route_editor_primitives.js';
import { ElementStub } from './chat_dom_fixture.js';
import { appendHtml, domDouble } from './dom_double.js';

setMissTransport(() => Promise.resolve(null));

const OPUS = { provider_id: 'openrouter', id: 'anthropic/claude-opus-5.5', value: 'anthropic/claude-opus-5.5',
    name: 'Anthropic: Claude Opus 5.5', price: { input: 4, output: 20 } };

test('a catalog price reads as one short fact, and as nothing where the catalog states none', () => {
    assert.equal(priceWords({ input: 4, output: 20 }), '$4 in / $20 out');
    assert.equal(priceWords({ input: 4, output: 20 }, { unit: true }), '$4 in / $20 out per 1M tokens');
    assert.equal(priceWords({ input: 0.075, output: 0.3 }), '$0.075 in / $0.30 out', 'the published figure, in cents at least');
    // A price input length changes says each figure's range; a figure no tier changes reads once.
    const tiered = { input: 2, output: 10, tiers: [{ min_prompt_tokens: 200000, input: 4, output: 15 }] };
    assert.equal(priceWords(tiered), '$2–4 in / $10–15 out');
    assert.equal(priceWords(tiered, { unit: true }), '$2–4 in / $10–15 out per 1M tokens');
    assert.equal(priceWords({ input: 0.1, output: 0.5, tiers: [{ min_prompt_tokens: 1, input: 0.5, output: 2.5 }] }),
        '$0.10–0.50 in / $0.50–2.50 out');
    assert.equal(priceWords({ input: 2, output: 10, tiers: [{ min_prompt_tokens: 128000, input: 2, output: 12 }] }), '$2 in / $10–12 out');
    // A free model is a price too, said once.
    assert.equal(priceWords({ input: 0, output: 0 }), 'free');
    assert.equal(priceWords({ input: 0, output: 0 }, { unit: true }), 'free');
    assert.equal(priceWords({ input: 0, output: 0, tiers: [{ min_prompt_tokens: 1000, input: 1, output: 2 }] }), '$0–1 in / $0–2 out');
    for (const none of [undefined, null, {}, { input: 4 }, { output: 20 }, { input: '4', output: '20' }, { input: -1, output: 20 },
        { input: Number.NaN, output: 1 }, { input: Number.POSITIVE_INFINITY, output: 1 },
        // A tier that cannot be read leaves the whole price unsaid, never the base with part of a range.
        { input: 2, output: 10, tiers: [{ min_prompt_tokens: 1, input: -1, output: 15 }] },
        { input: 2, output: 10, tiers: [{ min_prompt_tokens: 1, output: 15 }] },
        // A charge four decimals would print as zero is never shown as $0 or free.
        { input: 0.00001, output: 1 }, { input: 0.00001, output: 0.00001 }]) {
        assert.equal(priceWords(none), '', JSON.stringify(none));
        assert.equal(priceWords(none, { unit: true }), '', JSON.stringify(none));
    }
});

test('the price words go through the translation seam: one template, never one entry per price', () => {
    applyPayload({ language: 'ru', english: false, revision: 31, entries: {
        '{input} in / {output} out': { text: '{input} вход / {output} выход' },
        '{input} in / {output} out per 1M tokens': { text: '{input} вход / {output} выход за 1 млн токенов' },
        [CODE_PREFIX + 'model.price.free']: { text: 'бесплатно' } } });
    try {
        assert.equal(priceWords({ input: 4, output: 20 }), '$4 вход / $20 выход');
        assert.equal(priceWords({ input: 2, output: 10, tiers: [{ min_prompt_tokens: 1, input: 4, output: 15 }] }, { unit: true }),
            '$2–4 вход / $10–15 выход за 1 млн токенов');
        assert.equal(priceWords({ input: 0, output: 0 }), 'бесплатно');
    } finally {
        applyPayload({ language: '', english: true, revision: 32, entries: {} });
    }
    assert.equal(priceWords({ input: 4, output: 20 }), '$4 in / $20 out');
});

test('a suggestion carries the price apart from its label, which stays the name with one account and with eighteen', () => {
    const { price: _price, ...unpriced } = OPUS;
    const [option] = catalogModelOptions([OPUS]);
    assert.deepEqual(option, { value: OPUS.value, label: OPUS.name, meta: '$4 in / $20 out' });
    assert.deepEqual(catalogModelOptions([unpriced]), [{ value: OPUS.value, label: OPUS.name }]);
    const eighteen = Array.from({ length: 18 }, (_, index) => ({ ...OPUS, credential_profile_id: `acct-${index + 1}` }));
    assert.deepEqual(catalogModelOptions(eighteen), [option]);
    // Rows of one value that disagree, or one that states none, leave the price unsaid rather than pick one.
    assert.deepEqual(catalogModelOptions([OPUS, { ...OPUS, price: { input: 5, output: 25 } }]), [{ value: OPUS.value, label: OPUS.name }]);
    assert.deepEqual(catalogModelOptions([OPUS, unpriced]), [{ value: OPUS.value, label: OPUS.name }]);
});

test('the Settings chooser shows the price beside the name, never in it: not in the search, not in a repaint', () => {
    const items = [{ value: 'vendor/x', label: 'Model X', meta: '$4 in / $20 out' }, { value: 'vendor/y', label: 'Model 4o' }];
    const priced = (meta) => new RegExp(`aria-selected="false"><span class="ui-model-option-name">Model X</span> `
        + `<span class="ui-model-option-meta" data-i18n-skip>${meta.replace(/[$/]/g, '\\$&')}</span></div>`);
    const html = modelChooserHtml('aria-label="Main"', '', 'models', items);
    assert.match(html, priced('$4 in / $20 out'));
    assert.match(html, /aria-selected="false">Model 4o<\/div>/, 'an option without a price keeps its markup');
    assert.match(html, /class="ui-model-option ui-model-option-priced" role="option" id="models-0"/, 'a priced option says so by class');
    assert.match(html, /class="ui-model-option" role="option" id="models-1"/);
    assert.deepEqual(modelOptions(items), items);
    assert.deepEqual(matchingModelOptions(items, '4').map((item) => item.value), ['vendor/y']);

    // The bound chooser reads its options back from its own markup, as a page does.
    const { doc } = domDouble();
    appendHtml(doc.body, html);
    const input = doc.body.querySelector('[data-model-chooser]');
    const list = doc.getElementById('models');
    const dispose = bindModelChooser(input);
    const typed = (text) => { input.value = text; input.emit('input'); return list.innerHTML; };
    assert.doesNotMatch(typed('4'), /vendor\/x/, 'a figure of the price finds no model');
    assert.match(typed('4'), /data-model-value="vendor\/y"/);
    assert.match(typed('model'), priced('$4 in / $20 out'));
    // A catalog refresh repaints from new markup: its price, the same name.
    const desired = appendHtml(doc.createElement('div'), modelChooserHtml('aria-label="Main"', '', 'models',
        [{ ...items[0], meta: '$5 in / $25 out' }, items[1]]));
    updateModelChooserOptions(doc.body, desired);
    assert.match(list.innerHTML, priced('$5 in / $25 out'));
    assert.match(list.innerHTML, /aria-selected="false">Model 4o<\/div>/);
    assert.doesNotMatch(typed('25'), /vendor\/x/);
    dispose();
});

test('under a role: an OpenRouter model says its price per 1M tokens; every other row keeps its words', async () => {
    const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url), 'utf8'));
    const { doc } = domDouble();
    const host = doc.createElement('div');
    host.id = 'roles';
    doc.body.append(host);
    Object.defineProperty(host, 'innerHTML', { set(html) { host.replaceChildren(); appendHtml(host, html); }, get: () => '' });
    const sources = [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }];
    const codex = { provider_id: 'claudexor', source_id: 'codex', id: 'gpt-6-astra', name: 'gpt-6-astra', value: 'claudexor::codex=gpt-6-astra' };
    const editor = createModelRolesEditor({ hostId: 'roles', doc, showContext: false, store: { snapshot: {}, accountsKnown: false },
        catalogFetch: async () => ({ items: [codex], model_sources: sources }) });
    editor.load({ OPENROUTER_API_KEY: 'k', OPENAI_API_KEY: 'k', OUROBOROS_MODEL: OPUS.value,
        OUROBOROS_MODEL_LIGHT: codex.value, OUROBOROS_MODEL_CONSCIOUSNESS: 'x-ai/grok-4.7',
        OUROBOROS_MODEL_FALLBACKS: `openai::gpt-6, openrouter@gone::${OPUS.value}` }, contract);
    editor.adoptCatalog({ model_sources: sources, items: [OPUS, codex,
        { provider_id: 'openrouter', id: 'x-ai/grok-4.7', value: 'x-ai/grok-4.7', name: 'xAI: Grok 4.7' },
        // OpenRouter lists the very id the direct key below names, with a price; the key's own row states none.
        { provider_id: 'openrouter', id: 'gpt-6', value: 'gpt-6', name: 'gpt-6', price: { input: 1.25, output: 10 } },
        { provider_id: 'openai', id: 'gpt-6', value: 'openai::gpt-6', name: 'gpt-6' },
        { ...OPUS, provider_id: 'openrouter@gone', value: `openrouter@gone::${OPUS.value}` }] });
    await new Promise((resolve) => setImmediate(resolve));
    const said = (group) => host.querySelectorAll(`[data-model-role-group="${group}"] [data-model-role-status]`)
        .map((node) => node.textContent);
    assert.deepEqual(said('main'), ['$4 in / $20 out per 1M tokens']);
    assert.deepEqual(said('light'), ['Uses your subscription. No API key required.']);
    assert.deepEqual(said('vision'), ['Uses Main.']);
    assert.deepEqual(said('safety'), ['Uses Light.']);
    assert.deepEqual(said('consciousness'), ['Price unknown'], 'an OpenRouter model whose catalog states no price');
    const [direct, gone] = said('fallback');
    assert.equal(direct, 'Price unknown', 'a direct key names no price of its own, though OpenRouter prices the same id');
    assert.match(gone, /Accounts/, 'a key that cannot send says so first');
    assert.doesNotMatch(gone, /\$/);
    editor.destroy();
});

test('in the composer list a priced model says its price where "paid by key" stood; other rows read as before', () => {
    const index = buildModelIndex({ catalog: { items: [OPUS,
        { provider_id: 'openrouter', id: 'openai/gpt-6', value: 'openai/gpt-6', price: { input: 1.25, output: 10 } },
        { provider_id: 'openrouter', id: 'z-ai/glm-5.3', value: 'z-ai/glm-5.3' },
        { provider_id: 'openai', id: 'gpt-6', value: 'openai::gpt-6' }] } });
    assert.equal(describeRoute(index, { model: OPUS.value }).price, '$4 in / $20 out');
    assert.equal(describeRoute(index, { model: 'openai::gpt-6' }).price, '', 'a direct key states no price');
    assert.equal(describeRoute(index, { model: 'z-ai/glm-5.3' }).facts, 'OpenRouter · paid by key');
    const markup = (query) => {
        const [el, input] = [new ElementStub('div'), new ElementStub('input')];
        const list = createModelList({ el, input, id: 'models', store: { index: () => index, status: () => ({ read: true }) },
            main: () => ({ model: OPUS.value, use_local: false }), current: () => null, recents: () => [], onRoute() {} });
        list.paint();
        input.value = query;
        for (const listener of input.listeners.get('input') || []) listener();
        list.destroy();
        // Each row's facts; the "use as typed" row a query adds names its source only.
        return Object.fromEntries([...el.innerHTML.matchAll(/data-row="([^"]+)"[^>]*>[\s\S]*?<span class="chat-model-row-facts" data-i18n-skip>([^<]*)</g)]
            .map(([, row, facts]) => [row, facts]).filter(([row]) => !row.startsWith('typed:')));
    };
    assert.deepEqual(markup('p'), {
        [`route|${OPUS.value}`]: 'OpenRouter · $4 in / $20 out',
        'route|openai/gpt-6': 'OpenRouter · $1.25 in / $10 out',
        'route|z-ai/glm-5.3': 'OpenRouter · paid by key',
        'route|openai::gpt-6': 'OpenAI · paid by key',
    });
    // Main's own row and the source rows: the price stands with a model, a source still says how it is paid.
    assert.deepEqual(markup(''), {
        [`route|${OPUS.value}`]: 'OpenRouter · $4 in / $20 out',
        'source:openrouter': '3 models · paid by key',
        'source:openai': '1 model · paid by key',
    });
});


test('one rule decides where a processing tier prices the call instead of the catalog', async () => {
    const { tierPriced } = await import('../modules/route_editor_primitives.js');
    assert.deepEqual(['', 'standard', 'fast', 'economy', undefined].map(tierPriced), [false, false, true, true, false]);
});

test('the Agents chooser of an OpenRouter row shows the price too; a direct key reads as before', () => {
    const items = [OPUS, { provider_id: 'openai', id: 'gpt-6', value: 'openai::gpt-6', name: 'gpt-6' }];
    const priced = routeModelInputHtml('aria-label="Model"', { kind: 'api_model', target_id: OPUS.value }, items, 'agent-models');
    assert.match(priced, /<span class="ui-model-option-meta" data-i18n-skip>\$4 in \/ \$20 out<\/span>/);
    const direct = routeModelInputHtml('aria-label="Model"', { kind: 'api_model', target_id: 'openai::gpt-6' }, items, 'agent-models');
    assert.doesNotMatch(direct, /ui-model-option-meta/);
    assert.match(direct, /aria-selected="false">gpt-6<\/div>/);
});

test('in the composer list two catalog rows of one model that disagree on its price state none', () => {
    const index = buildModelIndex({ catalog: { items: [OPUS, { ...OPUS, credential_profile_id: 'second', price: { input: 5, output: 25 } }] } });
    assert.equal(describeRoute(index, { model: OPUS.value }).price, '');
    const agreeing = buildModelIndex({ catalog: { items: [OPUS, { ...OPUS, credential_profile_id: 'second' }] } });
    assert.equal(describeRoute(agreeing, { model: OPUS.value }).price, '$4 in / $20 out');
});


// The Settings role editor over one OpenRouter Main, as a page builds it; `settings` adds processing preferences.
async function mainRole(settings = {}) {
    const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url), 'utf8'));
    const { doc } = domDouble();
    const host = doc.createElement('div');
    host.id = 'roles';
    doc.body.append(host);
    Object.defineProperty(host, 'innerHTML', { set(html) { host.replaceChildren(); appendHtml(host, html); }, get: () => '' });
    const store = { snapshot: {}, accountsKnown: false, subscribe: () => () => {}, refresh: async () => {}, read: async () => {} };
    const editor = createModelRolesEditor({ hostId: 'roles', doc, showContext: false, store,
        catalogFetch: async () => ({ items: [], model_sources: [] }) });
    editor.load({ OPENROUTER_API_KEY: 'k', OUROBOROS_MODEL: OPUS.value, ...settings }, contract);
    editor.adoptCatalog({ model_sources: [], items: [OPUS] });
    await new Promise((resolve) => setImmediate(resolve));
    const status = host.querySelector('[data-model-role-group="main"] [data-model-role-status]');
    const input = host.querySelector('[data-model-role-group="main"] [data-model-role-model]');
    input.emit('click');  // opens the chooser without changing the role: it paints its suggestions
    const suggestions = doc.getElementById(input.getAttribute('aria-controls')).innerHTML;
    return { editor, status, suggestions, input, host };
}

test('a role on Fast or Economy names no catalog price: its tier is priced by its own endpoint, so the price is unknown', async () => {
    for (const settings of [{}, { OUROBOROS_PROCESSING_PREFERENCE: 'standard' }]) {
        const { editor, status, suggestions, input } = await mainRole(settings);
        assert.equal(status.textContent, '$4 in / $20 out per 1M tokens', JSON.stringify(settings));
        assert.equal(status.dataset.i18nFmt, 'model price', 'composed words: the overlay leaves them alone');
        assert.match(suggestions, /ui-model-option-meta/);
        input.value = '';
        input.emit('input');  // the role loses its model: the line is a sentence again, which the overlay reads
        assert.equal(status.textContent, 'Choose a model to continue.');
        assert.equal(status.dataset.i18nFmt, undefined);
        editor.destroy();
    }
    for (const settings of [{ OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"fast"}' }, { OUROBOROS_PROCESSING_PREFERENCE: 'economy' },
        { OUROBOROS_PROCESSING_PREFERENCE: 'fast', OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"light":"standard"}' }]) {
        const { editor, status, suggestions } = await mainRole(settings);
        assert.equal(status.textContent, 'Price unknown', JSON.stringify(settings));
        assert.equal(status.dataset.i18nFmt, undefined, 'no composed price: the overlay reads the line again');
        assert.match(suggestions, /data-model-value="anthropic\/claude-opus-5\.5"/, 'the model is still offered');
        assert.doesNotMatch(suggestions, /ui-model-option-meta/);
        editor.destroy();
    }
});

test('a fallback with no tier of its own names a price only while Main and the default both pay Standard', async () => {
    const line = async (settings) => {
        const { editor, host } = await mainRole({ OUROBOROS_MODEL_FALLBACKS: OPUS.value, ...settings });
        const text = host.querySelector('[data-model-role-group="fallback"] [data-model-role-status]').textContent;
        editor.destroy();
        return text;
    };
    assert.equal(await line({}), '$4 in / $20 out per 1M tokens');
    assert.equal(await line({ OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"fast"}' }), 'Price unknown', "Main's calls through it pay Fast");
    assert.equal(await line({ OUROBOROS_PROCESSING_PREFERENCE: 'economy' }), 'Price unknown', 'a lead on the default pays Economy');
    assert.equal(await line({ OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"fast","fallback":["standard"]}' }),
        '$4 in / $20 out per 1M tokens', 'its own tier is the one paid');
});

test('in the composer list the price follows the language of each paint, not of the catalog read', () => {
    const index = buildModelIndex({ catalog: { items: [OPUS] } });
    assert.equal(describeRoute(index, { model: OPUS.value }).price, '$4 in / $20 out');
    applyPayload({ language: 'ru', english: false, revision: 41, entries: { '{input} in / {output} out': { text: '{input} вход / {output} выход' } } });
    try {
        assert.equal(describeRoute(index, { model: OPUS.value }).price, '$4 вход / $20 выход', 'the same index, painted after a language change');
    } finally {
        applyPayload({ language: '', english: true, revision: 42, entries: {} });
    }
});

test('a circle or a place names the price beside a model only while its lead pays the catalog one', async () => {
    const { modelSuggestions } = await import('../modules/place_sheet.js');
    const { createModelCatalogStore } = await import('../modules/model_catalog_store.js');
    const direct = { provider_id: 'openai', id: 'gpt-6', value: 'openai::gpt-6', name: 'gpt-6' };
    const store = createModelCatalogStore({ readCatalog: async () => ({ items: [] }), readPreferences: async () => ({}),
        readSettings: async () => ({}) });
    const said = () => modelSuggestions({ items: [OPUS, direct, { id: 'local', use_local: true }] }, { tierPriced: store.tierPriced('lead') });
    assert.equal(said()[0].meta, undefined, 'before the settings document is read the tier is unknown');
    // A lead pays the default tier: Main's own tier changes nothing here.
    for (const settings of [{}, { OUROBOROS_PROCESSING_PREFERENCE: 'standard', OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"fast"}' }]) {
        store.adoptSettings(settings);
        const [priced, plain] = said();
        assert.equal(priced.meta, '$4 in / $20 out', JSON.stringify(settings));
        assert.equal(plain.meta, undefined, 'a model without a catalog price reads as before');
        assert.match(modelChooserHtml('aria-label="Model"', '', 'room-models', [priced, plain]), /ui-model-option-meta" data-i18n-skip>\$4 in \/ \$20 out</);
    }
    for (const settings of [{ OUROBOROS_PROCESSING_PREFERENCE: 'fast' },
        { OUROBOROS_PROCESSING_PREFERENCE: 'economy', OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"standard"}' }]) {
        store.adoptSettings(settings);
        const [offered] = said();
        assert.equal(offered.meta, undefined, JSON.stringify(settings));
        assert.equal(offered.value, OPUS.value, 'the model is still offered');
    }
});

test('the capsule strip and the Change Main sheet say the price where "paid by key" stood', async () => {
    const { routeFacts } = await import('../modules/model_list.js');
    const index = buildModelIndex({ catalog: { items: [OPUS, { provider_id: 'openrouter', id: 'z-ai/glm-5.3', value: 'z-ai/glm-5.3' }] } });
    const said = describeRoute(index, { model: OPUS.value });
    assert.equal(routeFacts(said), 'OpenRouter · $4 in / $20 out');
    assert.equal(said.facts, 'OpenRouter · $4 in / $20 out');
    assert.equal(routeFacts(describeRoute(index, { model: 'z-ai/glm-5.3' })), 'OpenRouter · paid by key');
});

test('a stale account pin left from a subscription never hides an OpenRouter role price', async () => {
    const { editor, status } = await mainRole({ OUROBOROS_MODEL_ACCOUNTS: '{"main":"old-codex-account"}' });
    assert.equal(status.textContent, '$4 in / $20 out per 1M tokens');
    editor.destroy();
});

test('a language change says the price under a role again, in the new language', async () => {
    const previous = globalThis.window;
    globalThis.window = new EventTarget();
    try {
        const { editor, host } = await mainRole();
        editor.mount();  // paints the editor anew
        await new Promise((resolve) => setImmediate(resolve));
        const line = () => host.querySelector('[data-model-role-group="main"] [data-model-role-status]').textContent;
        assert.equal(line(), '$4 in / $20 out per 1M tokens');
        applyPayload({ language: 'ru', english: false, revision: 51, entries: {
            '{input} in / {output} out per 1M tokens': { text: '{input} вход / {output} выход за 1 млн токенов' } } });
        globalThis.window.dispatchEvent(new Event('ouro:language-changed'));
        assert.equal(line(), '$4 вход / $20 выход за 1 млн токенов');
        editor.destroy();
        applyPayload({ language: '', english: true, revision: 53, entries: {} });
        globalThis.window.dispatchEvent(new Event('ouro:language-changed'));
        assert.equal(line(), '$4 вход / $20 выход за 1 млн токенов', 'a destroyed editor no longer listens');
    } finally {
        applyPayload({ language: '', english: true, revision: 52, entries: {} });
        globalThis.window = previous;
    }
});

test('the Agents editor paints again for another language: its prices are composed words', async () => {
    const { availableSubagentsRenderSignature } = await import('../modules/subagents_settings.js');
    const state = { loaded: true, setting: { items: [] } };
    const english = availableSubagentsRenderSignature(state, 0);
    applyPayload({ language: 'ru', english: false, revision: 61, entries: {} });
    try {
        assert.notEqual(availableSubagentsRenderSignature(state, 0), english);
    } finally {
        applyPayload({ language: '', english: true, revision: 62, entries: {} });
    }
    assert.equal(availableSubagentsRenderSignature(state, 0), english);
});

test('while Main or the default tier is on Fast or Economy the composer list names no catalog price; on Standard it does', async () => {
    assert.equal(describeRoute(buildModelIndex({ catalog: { items: [OPUS] }, tierPriced: true }), { model: OPUS.value }).price, '');
    const { createModelCatalogStore } = await import('../modules/model_catalog_store.js');
    const shown = async (settings) => {
        const store = createModelCatalogStore({ readCatalog: async () => ({ items: [OPUS] }), readPreferences: async () => ({}),
            readSettings: async () => settings });
        await store.read();
        await store.readConnections();
        return describeRoute(store.index(), { model: OPUS.value }).price;
    };
    assert.equal(await shown({}), '$4 in / $20 out');
    assert.equal(await shown({ OUROBOROS_PROCESSING_PREFERENCE: 'standard' }), '$4 in / $20 out');
    assert.equal(await shown({ OUROBOROS_PROCESSING_PREFERENCE: 'fast' }), '');
    assert.equal(await shown({ OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"economy"}' }), '');
    // The list serves Main and its rooms alike: a room's new work, a lead, would pay the default tier's price.
    assert.equal(await shown({ OUROBOROS_PROCESSING_PREFERENCE: 'fast', OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: 'standard' } }), '');
});

test('a role on a model stored with its local label says it runs locally, never Price unknown', async () => {
    const { editor, host } = await mainRole({ OUROBOROS_MODEL_VISION: 'qwen2.5-vl (local)' });
    assert.equal(host.querySelector('[data-model-role-group="vision"] [data-model-role-status]').textContent, 'Uses the local runtime.');
    editor.destroy();
});
