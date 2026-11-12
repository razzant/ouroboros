// The catalog's published price beside the model wherever the owner chooses one, as one short fact (owner, 12.11:
// compact, no disclaimers). The server keeps OpenRouter's price on its rows (ouroboros/gateway/models.py `price`);
// route_editor_primitives.js says it in words, the Settings chooser shows it beside the name (model_chooser.js
// `meta`) and the line under a role says it per 1M tokens (model_roles.js). A row without a price reads as before;
// a price never filters a model.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { bindModelChooser, matchingModelOptions, modelChooserHtml, modelOptions, updateModelChooserOptions } from '../modules/model_chooser.js';
import { createModelRolesEditor } from '../modules/model_roles.js';
import { catalogModelOptions, priceWords } from '../modules/route_editor_primitives.js';
import { appendHtml, domDouble } from './dom_double.js';

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
        { input: Number.NaN, output: 1 }, { input: Number.POSITIVE_INFINITY, output: 1 }]) {
        assert.equal(priceWords(none), '', JSON.stringify(none));
        assert.equal(priceWords(none, { unit: true }), '', JSON.stringify(none));
    }
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
    assert.deepEqual(said('consciousness'), [''], 'an OpenRouter model whose catalog states no price reads as before');
    const [direct, gone] = said('fallback');
    assert.equal(direct, '', 'a direct key says nothing new, though OpenRouter prices the same id');
    assert.match(gone, /Accounts/, 'a key that cannot send says so first');
    assert.doesNotMatch(gone, /\$/);
    editor.destroy();
});
