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
import { cacheWriteWords, catalogModelOptions, modelPriceWords, priceWords, routeModelInputHtml, withoutPrice } from '../modules/route_editor_primitives.js';
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
    const { doc } = domDouble({ parseHtml: true });
    appendHtml(doc.body, html);
    const input = doc.body.querySelector('[data-model-chooser]');
    const list = doc.getElementById('models');
    const dispose = bindModelChooser(input);
    const typed = (text) => { input.value = text; input.emit('input'); };
    const option = (id) => list.querySelector(`[data-model-value="${id}"]`);
    typed('4');
    assert.equal(option('vendor/x'), null, 'a figure of the price finds no model');
    assert.equal(option('vendor/y').textContent, 'Model 4o');
    typed('model');
    assert.equal(option('vendor/x').querySelector('.ui-model-option-name').textContent, 'Model X');
    assert.equal(option('vendor/x').querySelector('.ui-model-option-meta').textContent, '$4 in / $20 out');
    // A catalog refresh repaints from new markup: its price, the same name.
    const desired = appendHtml(doc.createElement('div'), modelChooserHtml('aria-label="Main"', '', 'models',
        [{ ...items[0], meta: '$5 in / $25 out' }, items[1]]));
    updateModelChooserOptions(doc.body, desired);
    assert.equal(option('vendor/x').querySelector('.ui-model-option-name').textContent, 'Model X');
    assert.equal(option('vendor/x').querySelector('.ui-model-option-meta').textContent, '$5 in / $25 out');
    assert.equal(option('vendor/y').textContent, 'Model 4o');
    assert.equal(option('vendor/y').querySelector('.ui-model-option-meta'), null);
    typed('25');
    assert.equal(option('vendor/x'), null);
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
        catalogFetch: async () => ({ items: [{ ...codex, max_context_window: 400000 }], model_sources: sources }) });
    editor.load({ OPENROUTER_API_KEY: 'k', OPENAI_API_KEY: 'k', OUROBOROS_MODEL: OPUS.value,
        OUROBOROS_MODEL_LIGHT: codex.value, OUROBOROS_MODEL_CONSCIOUSNESS: 'x-ai/grok-4.7',
        OUROBOROS_MODEL_FALLBACKS: `openai::gpt-6, openrouter@gone::${OPUS.value}` }, contract);
    editor.adoptCatalog({ model_sources: sources, items: [OPUS, codex,
        { provider_id: 'openrouter', id: 'x-ai/grok-4.7', value: 'x-ai/grok-4.7', name: 'xAI: Grok 4.7', context_window: 2000000 },
        // OpenRouter lists the very id the direct key below names, with a price; the key's own row states none.
        { provider_id: 'openrouter', id: 'gpt-6', value: 'gpt-6', name: 'gpt-6', price: { input: 1.25, output: 10 } },
        { provider_id: 'openai', id: 'gpt-6', value: 'openai::gpt-6', name: 'gpt-6' },
        { ...OPUS, provider_id: 'openrouter@gone', value: `openrouter@gone::${OPUS.value}` }] });
    await new Promise((resolve) => setImmediate(resolve));
    const said = (group) => host.querySelectorAll(`[data-model-role-group="${group}"] [data-model-role-status]`)
        .map((node) => node.textContent);
    assert.deepEqual(said('main'), ['$4 in / $20 out per 1M tokens']);
    assert.deepEqual(said('light'), ['Uses your subscription. No API key required. 400K window.'], 'the window its account\'s catalog states');
    assert.deepEqual(said('vision'), ['Uses Main.']);
    assert.deepEqual(said('safety'), ['Uses Light.']);
    assert.deepEqual(said('consciousness'), ['Price unknown · 2M window'],
        'an OpenRouter model whose catalog states no price; the window its catalog publishes stands beside it');
    // A window is said beside how the model is paid, never invented: the catalog's, in the model list's words.
    editor.adoptCatalog({ model_sources: sources, items: [{ ...OPUS, context_window: 1000000 }, codex] });
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(said('main'), ['$4 in / $20 out per 1M tokens · 1M window']);
    assert.deepEqual(said('safety'), ['Uses Light.'], 'an empty slot names what it follows, no window of its own');
    const [direct, gone] = said('fallback');
    assert.equal(direct, 'Price unknown', 'a direct key names no price of its own, though OpenRouter prices the same id');
    assert.match(gone, /Accounts/, 'a key that cannot send says so first');
    assert.doesNotMatch(gone, /\$/);
    editor.adoptSources({ read_state: 'transport', errors: [{ error: 'Engine offline' }], model_sources: sources });
    assert.deepEqual(said('main'), ['Subscription sources could not be read. $4 in / $20 out per 1M tokens · 1M window']);
    editor.adoptSources({ read_state: 'ok', errors: [], model_sources: sources });
    assert.deepEqual(said('main'), ['$4 in / $20 out per 1M tokens · 1M window']);
    editor.destroy();
});

test('a DeepSeek row its published entry prices only for text off-peak in USD says that condition, never a figure; Vision says Price unknown', async () => {
    assert.equal(modelPriceWords({ price_condition: 'off_peak_usd' }), 'Estimated for text, off-peak · USD only');
    assert.equal(modelPriceWords(withoutPrice({ price_condition: 'off_peak_usd' }, { media: true })), '', 'a role sending images: none');
    assert.equal(modelPriceWords({ price: { input: 4, output: 20 }, price_condition: 'off_peak_usd' }, { unit: true }),
        '$4 in / $20 out per 1M tokens', 'a published figure is said as before');
    assert.equal(modelPriceWords({ price_condition: 'surge' }), '', 'a condition this client does not know is no price');
    assert.equal(modelPriceWords(null), '');
    const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url), 'utf8'));
    const { doc } = domDouble();
    const host = doc.createElement('div');
    host.id = 'roles';
    doc.body.append(host);
    Object.defineProperty(host, 'innerHTML', { set(html) { host.replaceChildren(); appendHtml(host, html); }, get: () => '' });
    const editor = createModelRolesEditor({ hostId: 'roles', doc, showContext: false, store: { snapshot: {}, accountsKnown: false },
        catalogFetch: async () => ({ items: [], model_sources: [] }) });
    editor.load({ DEEPSEEK_API_KEY: 'k', OUROBOROS_MODEL: 'deepseek::deepseek-v4-pro', OUROBOROS_MODEL_LIGHT: 'deepseek::deepseek-chat',
        OUROBOROS_MODEL_VISION: 'deepseek::deepseek-v4-pro' }, contract);
    editor.adoptCatalog({ model_sources: [], items: [
        { provider_id: 'deepseek', id: 'deepseek-v4-pro', value: 'deepseek::deepseek-v4-pro', name: 'deepseek-v4-pro',
            price_condition: 'off_peak_usd', context_window: 1000000 },
        { provider_id: 'deepseek', id: 'deepseek-chat', value: 'deepseek::deepseek-chat', name: 'deepseek-chat' }] });
    await new Promise((resolve) => setImmediate(resolve));
    const said = (group) => host.querySelectorAll(`[data-model-role-group="${group}"] [data-model-role-status]`)
        .map((node) => node.textContent);
    assert.deepEqual(said('main'), ['Estimated for text, off-peak · USD only · 1M window']);
    assert.deepEqual(said('vision'), ['Price unknown · 1M window'], 'its calls carry images the text tariff never prices');
    assert.deepEqual(said('light'), ['Price unknown'], 'a retired entry stays unknown');
    editor.destroy();
});

test('every list of models says a conditional price the role line says, and drops it where the price is not the one paid', async () => {
    const pro = { provider_id: 'deepseek', id: 'deepseek-v4-pro', value: 'deepseek::deepseek-v4-pro', name: 'deepseek-v4-pro',
        price_condition: 'off_peak_usd' };
    const said = 'Estimated for text, off-peak · USD only';
    // Settings' suggestions (Models, Agents, the wizard); a Fast/Flex row and Vision drop the condition with the price.
    assert.equal(catalogModelOptions([pro])[0].meta, said);
    assert.equal(catalogModelOptions([withoutPrice(pro, { tiered: true })])[0].meta, undefined);
    assert.equal(catalogModelOptions([withoutPrice(pro, { media: true })])[0].meta, undefined);
    assert.deepEqual(withoutPrice({ ...OPUS }, { media: true }), OPUS, 'figures stay for Vision; only a text condition goes');
    assert.deepEqual(withoutPrice(OPUS, { tiered: true }), { ...OPUS, price: null, price_condition: '' });
    assert.equal(catalogModelOptions([pro, { ...pro, price_condition: '' }])[0].meta, undefined, 'records that disagree: none');
    // The composer's capsule and the work card's list (buildModelIndex → describeRoute).
    const index = (options = {}) => buildModelIndex({ catalog: { items: [pro] }, ...options });
    assert.equal(describeRoute(index(), { model: pro.value }).price, said);
    for (const tier of ['fast', 'economy', 'ultrafast']) {
        assert.equal(describeRoute(index(), { model: pro.value }, { tier }).price, '', `a context on ${tier}`);
    }
    assert.equal(describeRoute(index(), { model: pro.value }, { tier: 'standard' }).price, said, 'Standard pays it');
    const split = buildModelIndex({ catalog: { items: [pro, { ...pro, price_condition: 'surge' }] } });
    assert.equal(describeRoute(split, { model: pro.value }).price, '', 'a row whose records disagree states no condition');
    // A place's or a circle's Model field.
    const { modelSuggestions } = await import('../modules/place_sheet.js');
    assert.equal(modelSuggestions({ items: [pro] }, { tierPriced: false })[0].meta, said);
    assert.equal(modelSuggestions({ items: [pro] }, { tierPriced: true })[0].meta, undefined);
});

test('in the composer list a priced model says its price where "paid by key" stood; other rows read as before', () => {
    const index = buildModelIndex({ catalog: { items: [OPUS,
        { provider_id: 'openrouter', id: 'openai/gpt-6', value: 'openai/gpt-6', price: { input: 1.25, output: 10 } },
        { provider_id: 'openrouter', id: 'z-ai/glm-5.3', value: 'z-ai/glm-5.3' },
        { provider_id: 'openai', id: 'gpt-6', value: 'openai::gpt-6' }] } });
    assert.equal(describeRoute(index, { model: OPUS.value }).price, '$4 in / $20 out');
    assert.equal(describeRoute(index, { model: 'openai::gpt-6' }).price, '', 'a direct key states no price');
    assert.equal(describeRoute(index, { model: 'z-ai/glm-5.3' }).facts, 'OpenRouter · paid by key');
    const markup = (query, { mainRow = 'follow', open = false } = {}) => {
        const [el, input] = [new ElementStub('div'), new ElementStub('input')];
        const list = createModelList({ el, input, id: 'models', store: { index: () => index, status: () => ({ read: true }), pins: () => [], isPinned: () => false },
            main: () => ({ model: OPUS.value, use_local: false }), current: () => null, recents: () => [], mainRow, onRoute() {} });
        list.paint();
        // A press on `All sources` opens the fold: each source is a row of its own.
        if (open) for (const listener of el.listeners.get('click') || []) listener({ target: { closest: (selector) => (selector === '.chat-model-row' ? { dataset: { row: 'sources' } } : null) } });
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
    // The price stands with a model (Main's own row in the Main chat); a room's Main row says what it follows; the sources
    // fold under one row and, opened, each still says how it is paid.
    assert.deepEqual(markup('', { mainRow: 'main' }), { main: 'OpenRouter · $4 in / $20 out', sources: 'OpenRouter, OpenAI' });
    assert.deepEqual(markup('', { open: true }), {
        main: 'Follows Main · claude-opus-5.5',
        sources: 'OpenRouter, OpenAI',
        'source:openrouter': '3 models · paid by key',
        'source:openai': '1 model · paid by key',
    });
});


test('one rule decides where a processing tier prices the call instead of the catalog', async () => {
    const { tierPriced } = await import('../modules/route_editor_primitives.js');
    assert.deepEqual(['', 'standard', 'fast', 'economy', 'ultrafast', undefined].map(tierPriced), [false, false, true, true, true, false]);
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
    const { doc } = domDouble({ parseHtml: true });
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
    const suggestions = doc.getElementById(input.getAttribute('aria-controls'));
    return { editor, status, suggestions, input, host };
}

test('a role on Fast or Economy names no catalog price: its tier is priced by its own endpoint, so the price is unknown', async () => {
    for (const settings of [{}, { OUROBOROS_PROCESSING_PREFERENCE: 'standard' }]) {
        const { editor, status, suggestions, input } = await mainRole(settings);
        assert.equal(status.textContent, '$4 in / $20 out per 1M tokens', JSON.stringify(settings));
        assert.equal(status.dataset.i18nFmt, 'model price', 'composed words: the overlay leaves them alone');
        assert.equal(suggestions.querySelector('.ui-model-option-meta').textContent, '$4 in / $20 out');
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
        assert.ok([...suggestions.querySelectorAll('[data-model-value]')].some(row => row.dataset.modelValue === OPUS.value), 'the model is still offered');
        assert.equal(suggestions.querySelector('.ui-model-option-meta'), null);
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
    // The sheet's cache-write price (composer_model.test.js) is the catalog's own figure for the install's cache tier
    // (`price.cache_write`, chosen and stamped with its tier by the server): said as stated, never derived from the
    // input price, and not said where the catalog states none, where the figure would print as zero, where the settings
    // name another tier than the figure's (or none), or where another processing tier prices the call.
    assert.equal(said.cacheWrite, '');
    const cached = { ...OPUS, price: { ...OPUS.price, cache_write: 6.4, cache_write_ttl: '1h' } };
    const sheet = (options, tier = '') => describeRoute(buildModelIndex({ catalog: { items: [cached] }, ...options }), { model: OPUS.value },
        { tier }).cacheWrite;
    assert.equal(sheet({ cacheTier: '1h' }), '', 'without a requested tier, the workspace may apply another rate');
    assert.equal(sheet({ cacheTier: '1h' }, 'standard'), 'Cache write: $6.40 per 1M tokens.');
    assert.deepEqual([sheet({ cacheTier: '5m' }, 'standard'), sheet({ cacheTier: 'default' }, 'standard'), sheet({}, 'standard'), sheet({ cacheTier: '1h' }, 'fast'),
        sheet({ cacheTier: '1h' }, 'economy'), sheet({ cacheTier: '1h' }, 'ultrafast')], ['', '', '', '', '', '']);
    for (const none of [undefined, {}, { cache_write: 0, cache_write_ttl: '1h' }, { cache_write: 0.00001, cache_write_ttl: '1h' },
        { cache_write: -1, cache_write_ttl: '1h' }, { cache_write: '6.4', cache_write_ttl: '1h' }, { cache_write: 6.4 }]) {
        assert.equal(cacheWriteWords(none, '1h'), '', JSON.stringify(none));
    }
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

test('the Agents editor paints again for another language, and for its dictionary arriving: its prices are composed words', async () => {
    const { availableSubagentsRenderSignature } = await import('../modules/subagents_settings.js');
    const state = { loaded: true, setting: { items: [] } };
    const paint = () => availableSubagentsRenderSignature(state, 0);
    applyPayload({ language: '', english: true, revision: 60, entries: {} });
    const english = paint();
    try {
        applyPayload({ language: 'ru', english: false, revision: 60, entries: {} });
        const bare = paint();
        assert.notEqual(bare, english, 'another language');
        // The same language, its dictionary read again (a later revision): the composed words may read differently.
        applyPayload({ language: 'ru', english: false, revision: 61, entries: {
            '{input} in / {output} out per 1M tokens': { text: '{input} вход / {output} выход за 1 млн токенов' } } });
        assert.notEqual(paint(), bare, 'the dictionary arrived');
    } finally {
        applyPayload({ language: '', english: true, revision: 60, entries: {} });
    }
    assert.equal(paint(), english, 'the same language and dictionary: nothing to paint again');
});

test('where a row\'s context runs on Fast, Ultrafast or Economy the composer list names no catalog price; on Standard it does', async () => {
    // The catalog's price is Standard's: the context's tier decides (model_list.js asks it per row, `tierOf`).
    const index = buildModelIndex({ catalog: { items: [OPUS] } });
    const price = (tier) => describeRoute(index, { model: OPUS.value }, { tier }).price;
    assert.deepEqual(['', 'standard', 'fast', 'ultrafast', 'economy'].map(price), ['Catalog rate · $4 in / $20 out', '$4 in / $20 out', '', '', '']);
    const { createModelCatalogStore } = await import('../modules/model_catalog_store.js');
    const read = async (settings) => {
        const store = createModelCatalogStore({ readCatalog: async () => ({ items: [OPUS] }), readPreferences: async () => ({}),
            readSettings: async () => settings });
        await store.read();
        await store.readConnections();
        return { kept: describeRoute(store.index(), { model: OPUS.value }).price, main: store.tierPriced('main'), lead: store.tierPriced('lead') };
    };
    // The window's index keeps the price whatever the tiers; the tiers say where a role pays another tier's.
    assert.deepEqual(await read({}), { kept: '$4 in / $20 out', main: false, lead: false });
    assert.deepEqual(await read({ OUROBOROS_PROCESSING_PREFERENCE: 'standard' }), { kept: '$4 in / $20 out', main: false, lead: false });
    assert.deepEqual(await read({ OUROBOROS_PROCESSING_PREFERENCE: 'fast' }), { kept: '$4 in / $20 out', main: true, lead: true });
    assert.deepEqual(await read({ OUROBOROS_MODEL_PROCESSING_PREFERENCES: '{"main":"economy"}' }), { kept: '$4 in / $20 out', main: true, lead: false });
    // A room's new work, a lead, pays the default tier's price whatever Main's own one is.
    assert.deepEqual(await read({ OUROBOROS_PROCESSING_PREFERENCE: 'fast', OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: 'standard' } }),
        { kept: '$4 in / $20 out', main: false, lead: true });
});

test('a local role says the local model\'s window, never the window of a cloud id its slot still holds', async () => {
    const wide = { ...OPUS, context_window: 1000000 };
    const local = { provider_id: 'local', id: 'qwen.gguf', value: 'qwen.gguf (local)', name: 'qwen.gguf', use_local: true, context_window: 64000 };
    const said = async (settings, items) => {
        const { editor, host } = await mainRole(settings);
        editor.adoptCatalog({ model_sources: [], items });
        await new Promise((resolve) => setImmediate(resolve));
        const text = host.querySelector('[data-model-role-group="main"] [data-model-role-status]').textContent;
        editor.destroy();
        return text;
    };
    assert.equal(await said({ USE_LOCAL_MAIN: true }, [wide, local]), 'Uses the local runtime. 64K window.');
    assert.equal(await said({ USE_LOCAL_MAIN: true }, [wide]), 'Uses the local runtime.', 'no local row: no window is claimed');
    assert.equal(await said({}, [wide, local]), '$4 in / $20 out per 1M tokens · 1M window', 'the same slot on its cloud route');
});

test('the window a route is sized with is said before the one its catalog publishes', async () => {
    const { apiClient } = await import('../modules/api_client.js');
    const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url), 'utf8'));
    const { doc } = domDouble();
    const host = doc.createElement('div');
    host.id = 'roles';
    doc.body.append(host);
    Object.defineProperty(host, 'innerHTML', { set(html) { host.replaceChildren(); appendHtml(host, html); }, get: () => '' });
    const preview = apiClient.responseLimitPreview;
    apiClient.responseLimitPreview = async () => ({ route: { model: OPUS.value }, window_ack: false,
        response_limit: { route_fp: 'fp', max_output_tokens: 0, source: 'auto' }, window: { tokens: 400000, source: 'provider_metadata' } });
    try {
        const editor = createModelRolesEditor({ hostId: 'roles', doc, showContext: true,
            store: { snapshot: {}, accountsKnown: false, subscribe: () => () => {}, refresh: async () => {}, read: async () => {} },
            catalogFetch: async () => ({ items: [], model_sources: [] }) });
        editor.load({ OPENROUTER_API_KEY: 'k', OUROBOROS_MODEL: OPUS.value }, contract);
        editor.adoptCatalog({ model_sources: [], items: [{ ...OPUS, context_window: 1000000 }] });
        for (let turn = 0; turn < 6; turn += 1) await new Promise((resolve) => setImmediate(resolve));
        assert.equal(host.querySelector('[data-model-role-group="main"] [data-model-role-status]').textContent,
            '$4 in / $20 out per 1M tokens · 400K window');
        editor.destroy();
    } finally {
        apiClient.responseLimitPreview = preview;
    }
});

test('a role on a model stored with its local label says it runs locally, never Price unknown', async () => {
    const { editor, host } = await mainRole({ OUROBOROS_MODEL_VISION: 'qwen2.5-vl (local)' });
    assert.equal(host.querySelector('[data-model-role-group="vision"] [data-model-role-status]').textContent, 'Uses the local runtime.');
    editor.destroy();
});
