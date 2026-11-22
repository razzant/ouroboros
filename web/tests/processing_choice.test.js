// A processing tier beside its model (processing_choice.js; DESIGN "Composer owner controls", "A work's model on its
// card"): the stored words (`economy`, `standard`, `fast`, `ultrafast`; '' is Default) in the owner's (Flex, Standard,
// Fast, Ultrafast), what the catalog says one bound route serves (its transport, its model, its account or Auto, a
// key model's own endpoints), which tiers the strip then offers, which tier a scope takes without a choice of its own,
// what a choice is and what a call came to, the question before Main takes Ultrafast, and the strip at the foot of the
// Model sheet. The saves are composer_model.test.js and card_model.test.js; the strip's markup is built into nodes here
// (dom_double.js `parseHtml`).
import assert from 'node:assert/strict';
import test from 'node:test';
import { buildModelIndex } from '../modules/chat_model_routes.js';
import {
    MAIN_ULTRAFAST_ROLLBACK, MAIN_ULTRAFAST_SCOPE, MAIN_ULTRAFAST_ROLES, TIERS, confirmMainUltrafast, createTierStrip, offeredTiers, pinDecides,
    receiptWords, resolvedTier, routeTiers, servedWords, tierBinding, tierCarried, tierChoices, tierExplanation, tierNamed,
    tierPriceWords, tierSupport, tierWord,
} from '../modules/processing_choice.js';
import { domDouble } from './dom_double.js';

const route = (model) => ({ model, use_local: false });
const ASTRA = route('claudexor::codex=gpt-6-astra');
const SOL = route('claudexor::codex=gpt-6.1-sol');
const OLD = route('claudexor::codex=gpt-5-old');
const GLM = route('z-ai/glm-5.3');
const codex = (id, modes, profile = 'home') => ({ provider_id: 'claudexor', provider: 'Codex', source_id: 'codex', id, name: id,
    value: `claudexor::codex=${id}`, credential_profile_id: profile, ...(modes ? { processing: { modes } } : {}) });
// One account of the engine's own envelope: what it lists per model (null: the account was not read).
const account = (id, models, extra = {}) => ({ credentialProfileId: id, availability: 'available', problem: null,
    catalog: models && { source: 'codex', credentialProfileId: id, provenance: 'live',
        models: Object.entries(models).map(([model, modes]) => ({ id: model, ...(modes ? { processing: { modes } } : {}) })), ...extra } });
const CEILING = ['standard', 'fast', 'ultrafast'];
// A Codex catalog as one read gives it: the transport's ceiling, the accounts' envelopes, and the list's rows.
const codexCatalog = (accounts, { ceiling = CEILING, partial = false } = {}) => ({
    model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex', ...(ceiling ? { processingPreferences: ceiling } : {}) }],
    ...(accounts ? { account_catalogs: [{ source: 'codex', partial, accounts }] } : {}),
    items: [codex('gpt-6-astra'), codex('gpt-6.1-sol'), codex('gpt-5-old'),
        { provider_id: 'openrouter', id: GLM.model, name: GLM.model, value: GLM.model }],
});
const support = (index, seen, binding) => TIERS.map((tier) => tierSupport(index, seen, tier, binding));

test('the tiers in the strip\'s order and in the owner\'s words; every tier but Standard is named beside a model', () => {
    assert.deepEqual(TIERS, ['economy', 'standard', 'fast', 'ultrafast'], 'cheaper and slower first');
    assert.ok(Object.isFrozen(TIERS));
    assert.deepEqual(['economy', 'standard', 'fast', 'ultrafast', '', 'priority', undefined].map(tierWord),
        ['Flex', 'Standard', 'Fast', 'Ultrafast', 'Default', 'Default', 'Default'], 'anything unknown reads as Default');
    assert.deepEqual(['economy', 'standard', 'fast', 'ultrafast', '', 'priority'].map(tierNamed), [true, false, true, true, false, false]);
});

test('support is one bound route\'s: its transport\'s ceiling, and the account it is pinned to or all of them under Auto', () => {
    const index = buildModelIndex({ catalog: codexCatalog([
        account('work', { 'gpt-6-astra': ['standard', 'fast', 'ultrafast'], 'gpt-6.1-sol': ['standard', 'fast'] }),
        account('home', { 'gpt-6-astra': ['standard', 'fast'], 'gpt-6.1-sol': ['standard', 'fast'], 'gpt-5-old': ['standard'] }),
    ]) });
    // In the strip's order: Flex, Standard, Fast, Ultrafast. Flex is beyond the transport: proven absent for every binding.
    assert.deepEqual(support(index, ASTRA, { account: 'work' }), ['unlisted', 'listed', 'listed', 'listed'], 'the pinned account\'s own list');
    assert.deepEqual(support(index, ASTRA, { account: 'home' }), ['unlisted', 'listed', 'listed', 'unknown'],
        'another pin of the same model: an account\'s list without a tier is no refusal, and never the other account\'s word');
    assert.deepEqual(support(index, ASTRA), ['unlisted', 'listed', 'listed', 'some'], 'Auto: Fast on every account, Ultrafast on some');
    assert.deepEqual(support(index, ASTRA, { account: '' }), support(index, ASTRA), 'no pin is Auto');
    assert.deepEqual(support(index, SOL), ['unlisted', 'listed', 'listed', 'unknown'], 'no account lists Ultrafast for it: unconfirmed, never refused');
    assert.deepEqual(support(index, OLD), ['unlisted', 'listed', 'unknown', 'unknown'],
        'the account that does not list the model serves none of its calls; the one that does lists only Standard');
    assert.deepEqual(support(index, ASTRA, { account: 'gone' }), ['unlisted', 'listed', 'unknown', 'unknown'], 'a pin this read does not know');
    // A pin that is not known (the settings were not read, or their read failed) is no account's word: never Auto,
    // even where the one account read lists everything.
    const one = buildModelIndex({ catalog: codexCatalog([account('work', { 'gpt-6-astra': CEILING })]) });
    assert.deepEqual(support(one, ASTRA), ['unlisted', 'listed', 'listed', 'listed'], 'Auto over the whole pool of one');
    assert.deepEqual(support(one, ASTRA, { account: null }), ['unlisted', 'listed', 'unknown', 'unknown'], 'an unread pin confirms nothing');
    assert.deepEqual(support(one, ASTRA, { account: 'removed' }), ['unlisted', 'listed', 'unknown', 'unknown'],
        'a saved pin to an account that is gone is not confirmed by the one that is left');
    assert.deepEqual([pinDecides(one, ASTRA), pinDecides(one, SOL), pinDecides(one, GLM), pinDecides(null, ASTRA)], [true, true, false, false],
        'a subscription sheet names the pin even when no account lists the model');
    // A transport that does not advertise a tier leaves it out, whatever an account lists.
    const narrow = buildModelIndex({ catalog: codexCatalog([account('work', { 'gpt-6-astra': CEILING })], { ceiling: ['standard', 'fast'] }) });
    assert.deepEqual(support(narrow, ASTRA), ['unlisted', 'listed', 'listed', 'unlisted']);
    // No transport list: the accounts decide Flex and Fast; Ultrafast alone needs the transport's own word too (13.11).
    const unsaid = buildModelIndex({ catalog: codexCatalog([account('work', { 'gpt-6-astra': ['economy', ...CEILING] })], { ceiling: null }) });
    assert.deepEqual(support(unsaid, ASTRA), ['listed', 'listed', 'listed', 'unknown']);
    assert.deepEqual(support(unsaid, GLM), ['unknown', 'listed', 'unknown', 'unknown'], 'a key\'s row says nothing of tiers by itself');
    assert.deepEqual(support(unsaid, route('vendor/unlisted')), ['unknown', 'listed', 'unknown', 'unknown'], 'a route the catalog does not list');
    assert.equal(tierSupport(index, null, 'fast'), 'listed', 'no route: nothing to warn about');
    assert.equal(tierSupport(index, SOL, ''), 'listed', 'Default is no tier of its own');
    assert.equal(tierSupport(null, SOL, 'fast'), 'unknown', 'no catalog read yet');
});

test('under Auto a tier is every account\'s only when every account was read: an error, a pause, an empty or a shipped list is a gap, never a refusal', () => {
    const lists = { 'gpt-6-astra': CEILING };
    const fast = (accounts, options) => tierSupport(buildModelIndex({ catalog: codexCatalog(accounts, options) }), ASTRA, 'fast');
    assert.equal(fast([account('work', lists), account('home', lists)]), 'listed', 'declared by all');
    assert.equal(fast([account('work', lists), account('home', { 'gpt-6-astra': ['standard'] })]), 'some', 'declared by some');
    assert.equal(fast([account('work', lists), account('home', null)]), 'some', 'one account was not read: no "all" from the rest');
    assert.equal(fast([account('work', lists), { ...account('home', null), availability: 'unavailable', problem: { code: 'subscription_window_exhausted' } }]),
        'some', 'an account in a pause was not asked');
    assert.equal(fast([account('work', lists), account('home', {})]), 'some', 'an empty list says nothing');
    assert.equal(fast([account('work', lists), account('home', lists, { provenance: 'bundled' })]), 'some', 'a list a program ships is no account\'s word');
    assert.equal(fast([account('work', lists), account('home', lists)], { partial: true }), 'some', 'a partial envelope is not every account');
    assert.equal(fast([account('work', lists), account('other', { 'gpt-5-old': ['standard'] })]), 'listed',
        'an account whose live list, read whole, lacks the model serves none of its calls');
    // A shipped or partial list that lacks the model is no such proof: that account stays in the count, unknown.
    assert.equal(fast([account('work', lists), account('other', { 'gpt-5-old': ['standard'] }, { provenance: 'bundled' })]), 'some');
    assert.equal(fast([account('work', lists), account('other', { 'gpt-5-old': ['standard'] }, { partial: true })]), 'some');
    // An account marked not eligible for a model's tiers lists nothing for it.
    assert.equal(fast([account('work', lists), { ...account('home', lists), catalog: { source: 'codex', credentialProfileId: 'home', provenance: 'live',
        models: [{ id: 'gpt-6-astra', processing: { modes: CEILING, eligible: false } }] } }]), 'some');
    // An empty transport list is no list: never "only the ordinary service".
    assert.equal(fast([account('work', lists)], { ceiling: [] }), 'listed');
    assert.equal(tierSupport(buildModelIndex({ catalog: codexCatalog([], { ceiling: [] }) }), ASTRA, 'economy'), 'unknown');
    assert.equal(fast([account('work', { 'gpt-6-astra': ['standard'] }), account('home', { 'gpt-6-astra': ['standard'] })]), 'unknown',
        'no account lists it: unconfirmed, since an account\'s list is not its whole entitlement');
    assert.equal(fast([account('work', null), account('home', null)]), 'unknown', 'nothing read');
    assert.equal(fast([]), 'unknown', 'no account at all');
    // An older engine sends no envelopes: its rows name accounts, never all of them.
    const legacy = buildModelIndex({ catalog: { ...codexCatalog(null), items: [codex('gpt-6-astra', CEILING, 'work'), codex('gpt-6-astra', CEILING, 'home')] } });
    assert.deepEqual([tierSupport(legacy, ASTRA, 'fast'), tierSupport(legacy, ASTRA, 'fast', { account: 'work' })], ['some', 'listed']);
});

/* A key model's own endpoints, as the catalog store keeps one `?processing_model=` answer (model_catalog_store.js). */
const detail = (modes, { complete = true, endpoints = [], state = 'ok' } = {}) => ({ state, modes: new Set(modes), complete, endpoints });
const KEY_CATALOG = { items: [{ provider_id: 'openrouter', id: GLM.model, name: GLM.model, value: GLM.model, processing_transport: ['standard', 'fast', 'economy', 'ultrafast'] },
    { provider_id: 'openrouter', id: 'moonshotai/kimi-k3', name: 'moonshotai/kimi-k3', value: 'moonshotai/kimi-k3' },
    { provider_id: 'openai', id: 'gpt-6', name: 'gpt-6', value: 'openai::gpt-6', processing_transport: ['standard'] }] };

test('a key\'s route is told by its model\'s own endpoints: a whole list may leave a tier out, a missing or failed one tells nothing', () => {
    const index = buildModelIndex({ catalog: KEY_CATALOG });
    const KIMI = route('moonshotai/kimi-k3');
    assert.deepEqual(support(index, GLM), ['unknown', 'listed', 'unknown', 'unknown'], 'not read yet: nothing is told, never Standard only');
    assert.deepEqual(support(index, GLM, { detail: detail([], { state: 'reading' }) }), ['unknown', 'listed', 'unknown', 'unknown']);
    assert.deepEqual(support(index, GLM, { detail: detail([], { state: 'failed' }) }), ['unknown', 'listed', 'unknown', 'unknown'], 'a failed read is no refusal');
    assert.deepEqual(support(index, GLM, { detail: detail(['standard', 'fast', 'ultrafast']) }), ['unlisted', 'listed', 'listed', 'listed'],
        'the whole list leaves Flex out');
    assert.deepEqual(support(index, GLM, { detail: detail(['standard', 'fast'], { complete: false }) }), ['unknown', 'listed', 'listed', 'unknown'],
        'a list that is not whole proves no absence');
    assert.deepEqual(support(index, KIMI, { detail: detail(['standard', 'fast', 'ultrafast']) }), ['unlisted', 'listed', 'listed', 'unknown'],
        'Ultrafast without the transport\'s own word stays unconfirmed');
    assert.deepEqual(support(index, route('openai::gpt-6')), ['unlisted', 'listed', 'unlisted', 'unlisted'], 'a transport that carries only Standard');
    assert.deepEqual(support(index, GLM, { account: 'work', detail: detail(['standard']) }), ['unlisted', 'listed', 'unlisted', 'unlisted'],
        'a key has no accounts: a pin changes nothing');
});

test('the strip offers what is not proven absent, Ultrafast only where something lists it, and always what the scope chose or takes', () => {
    const of = (answers) => (tier) => answers[tier] || 'unknown';
    assert.deepEqual(offeredTiers({ supportOf: of({}) }), ['economy', 'standard', 'fast'], 'nothing known: the unconfirmed tiers stand, Ultrafast does not');
    assert.deepEqual(offeredTiers({ supportOf: of({ economy: 'unlisted', fast: 'listed', ultrafast: 'some' }) }), ['standard', 'fast', 'ultrafast']);
    const none = of({ economy: 'unlisted', fast: 'unlisted', ultrafast: 'unlisted' });
    assert.deepEqual(offeredTiers({ supportOf: none, ordinaryDefault: true }), [], 'omission already uses the only tier');
    assert.deepEqual(offeredTiers({ supportOf: none }), ['standard'], 'only what is listed is known, not what omission does');
    assert.deepEqual(offeredTiers({ supportOf: none, choice: 'standard' }), ['standard'], 'an explicit Standard is a choice to see and leave');
    assert.deepEqual(offeredTiers({ supportOf: none, choice: 'fast' }), ['standard', 'fast'], 'a wish this route does not serve stays in view');
    assert.deepEqual(offeredTiers({ supportOf: none, resolved: 'ultrafast' }), ['standard', 'ultrafast'], 'and so does an inherited one');
    assert.deepEqual(offeredTiers({ supportOf: none, resolved: 'standard' }), [], 'Standard by Default on such a route: nothing to choose');
    assert.deepEqual(offeredTiers({ supportOf: of({}), choice: 'ultrafast' }), ['economy', 'standard', 'fast', 'ultrafast']);
    // A transport known to carry Ultrafast offers it as an attempt nothing confirms; one proven absent is never new.
    assert.deepEqual(offeredTiers({ supportOf: of({}), carries: (tier) => tier === 'ultrafast' }), ['economy', 'standard', 'fast', 'ultrafast']);
    assert.deepEqual(offeredTiers({ supportOf: of({ ultrafast: 'unlisted' }), carries: () => true }), ['economy', 'standard', 'fast']);
    assert.deepEqual(offeredTiers(), ['economy', 'standard', 'fast'], 'an unknown transport cannot offer a new Ultra');
});

test('a scope without a tier of its own takes Main\'s while it follows Main, else the default one; Standard is a choice', () => {
    const tiers = { main: 'fast', default: 'economy' };
    assert.equal(resolvedTier({ choice: 'standard', follows: true, tiers }), 'standard', 'an explicit Standard is never Default');
    assert.equal(resolvedTier({ choice: '', follows: true, tiers }), 'fast');
    assert.equal(resolvedTier({ choice: '', follows: false, tiers }), 'economy');
    assert.equal(resolvedTier({ follows: true, tiers: { main: '', default: 'economy' } }), 'economy', 'Main without its own tier takes the default');
    assert.equal(resolvedTier({ follows: true, tiers: { main: 'ultrafast', default: 'economy' } }), 'ultrafast', 'Main\'s own Ultrafast is inherited like any tier');
    assert.equal(resolvedTier({ follows: true, tiers: null }), '', 'nothing read: nothing claimed');
    assert.equal(resolvedTier(), '');
});

test('before a press the strip says what a tier means here, and that an uncertain one may go as Standard', () => {
    assert.equal(tierExplanation('standard', { support: 'unlisted', route: ASTRA }), 'Request Standard processing.');
    assert.equal(tierExplanation('', { route: ASTRA }), 'No tier requested.');
    assert.equal(tierExplanation('fast', { support: 'listed', route: ASTRA }), 'Requests faster processing. It may cost more.');
    assert.equal(tierExplanation('fast', { support: 'some', route: ASTRA, sourceLabel: 'Codex' }),
        'Requests faster processing. It may cost more. Only some Codex accounts are known to list Fast for gpt-6-astra: on the others a call may go as Standard.',
        'available on a part, and no count of accounts is said');
    assert.equal(tierExplanation('economy', { support: 'unlisted', route: ASTRA, sourceLabel: 'Codex' }),
        'Requests slower processing at a discount when capacity is available. Codex does not list Flex for gpt-6-astra: its calls may go as Standard.',
        'said as what may happen, never as what a call will be served as');
    assert.equal(tierExplanation('fast', { support: 'unlisted', route: SOL }),
        'Requests faster processing. It may cost more. This source does not list Fast for gpt-6.1-sol: its calls may go as Standard.');
    assert.equal(tierExplanation('fast', { support: 'unknown', route: OLD }),
        'Requests faster processing. It may cost more. Not confirmed for gpt-5-old here: a provider may serve it as Standard.');
    assert.equal(tierExplanation('ultrafast', { support: 'listed', route: ASTRA }), 'Requests the fastest processing this source offers. It may cost more.');
    assert.equal(tierExplanation('fast', { support: 'listed', route: GLM, price: 'Listed at $8 in / $40 out per 1M tokens.' }),
        'Requests faster processing. It may cost more. Listed at $8 in / $40 out per 1M tokens.', 'what it is listed at follows where that is told');
});

test('a tier is listed at its own endpoints\' price, only from a whole priced list: never Standard\'s, one of several, a multiple or a promise', () => {
    const fast = (price, tag = 'a') => ({ mode: 'fast', tag, price });
    const said = (endpoints, tier = 'fast', options = {}) => tierPriceWords(detail(['standard', 'fast'], { endpoints, ...options }), tier);
    const standard = { mode: 'standard', tag: 's', price: { input: 4, output: 20 } };
    assert.equal(said([standard, fast({ input: 8, output: 40 })]), 'Listed at $8 in / $40 out per 1M tokens.', 'what the catalog lists, said so');
    assert.equal(said([standard, fast({ input: 8, output: 40 }), fast({ input: 8, output: 40 }, 'b')]), 'Listed at $8 in / $40 out per 1M tokens.',
        'every provider the same');
    assert.equal(said([standard, fast({ input: 8, output: 40 }), fast({ input: 9, output: 45 }, 'b')]),
        'Listed at $8–9 in / $40–45 out per 1M tokens, depending on the provider.', 'several prices: their range, never one of them as the price');
    assert.equal(said([standard, fast({ input: 8, output: 40 }), fast(null, 'b')]), '', 'a provider that states none: no price of the others is quoted');
    assert.equal(said([standard, fast(null)]), '', 'a tier listed without a price is still a tier, and has no price here');
    assert.equal(said([standard]), '', 'never the model\'s Standard price');
    assert.equal(said([standard, fast({ input: 8, output: 40 })], 'fast', { complete: false }), '',
        'a list that is not whole: the one endpoint it holds is not the tier\'s price');
    assert.equal(said([standard, fast({ input: 8, output: 40, tiers: [{ min_prompt_tokens: 200000, input: 16, output: 60 }] })]),
        'Listed at $8–16 in / $40–60 out per 1M tokens.', 'a price that changes with the input says its range');
    assert.equal(said([standard, fast({ input: 8, output: 40, tiers: [{ min_prompt_tokens: 200000, input: 16, output: 60 }] }), fast({ input: 6, output: 30 }, 'b')]),
        'Listed at $6–16 in / $30–60 out per 1M tokens, depending on the provider.');
    assert.equal(said([standard, fast({ input: 8, output: 40 })], ''), '', 'omitting a tier does not select Standard endpoints');
    assert.equal(tierPriceWords(null, 'fast'), '');
    assert.equal(tierPriceWords(detail(['fast'], { state: 'failed', endpoints: [fast({ input: 8, output: 40 })] }), 'fast'), '', 'a failed read prices nothing');
    // One bound route's words: a key's tier says what it is listed at or that its price is unknown; a subscription names none.
    const index = buildModelIndex({ catalog: { ...KEY_CATALOG, ...codexCatalog([account('work', { 'gpt-6-astra': CEILING })]),
        items: [...KEY_CATALOG.items, codex('gpt-6-astra')] } });
    const key = routeTiers({ index, route: GLM, sourceLabel: 'OpenRouter',
        detail: detail(['standard', 'fast', 'ultrafast'], { endpoints: [standard, fast({ input: 8, output: 40 }), { mode: 'ultrafast', tag: 'u', price: null }] }) });
    assert.deepEqual(TIERS.map(key.priceOf), ['Price unknown.', 'Listed at $4 in / $20 out per 1M tokens.', 'Listed at $8 in / $40 out per 1M tokens.', 'Price unknown.'], 'only the selected tier\'s own published endpoint prices');
    assert.equal(key.explain('standard'), 'Request Standard processing. Listed at $4 in / $20 out per 1M tokens.');
    // A private base keeps the OpenRouter route kind but publishes no native tier transport.
    const privateRoute = route('openrouter@private::fixture/model');
    const privateIndex = buildModelIndex({ catalog: { items: [
        { provider_id: 'openrouter', id: 'fixture/model', value: privateRoute.model },
    ] } });
    assert.equal(routeTiers({ index: privateIndex, route: privateRoute }).explain('standard'), 'Request Standard processing. Price unknown.');
    assert.equal(key.explain(''), 'No tier requested. OpenRouter may apply a workspace default or another tier. Price unknown.');
    assert.equal(key.explain('ultrafast'), 'Requests the fastest processing this source offers. It may cost more. Price unknown.');
    assert.equal(routeTiers({ index, route: GLM }).explain('fast'),
        'Requests faster processing. It may cost more. Not confirmed for glm-5.3 here: a provider may serve it as Standard. Price unknown.', 'nothing read yet');
    const subscription = routeTiers({ index, route: ASTRA, account: 'work', sourceLabel: 'Codex' });
    assert.deepEqual([TIERS.map(subscription.priceOf), subscription.explain('fast')], [['', '', '', ''], 'Requests faster processing. It may cost more.']);
});

test('a new Ultrafast is offered on a transport known to carry it, as an attempt nothing confirms; an unknown or non-carrying transport offers none', () => {
    // Three direct keys, no per-model or per-account word on any of them: one whose transport carries every tier (a
    // Responses connection), one whose transport says nothing, one whose transport carries no Ultrafast (Chat Completions).
    const key = (source, transport) => ({ provider_id: source.split('@')[0], id: 'gpt-6', name: 'gpt-6', value: `${source}::gpt-6`,
        ...(transport ? { processing_transport: transport } : {}) });
    const index = buildModelIndex({ catalog: { items: [key('openai', ['standard', 'fast', 'economy', 'ultrafast']), key('deepseek'),
        key('openai@chat', ['standard', 'fast', 'economy'])] } });
    const [RESPONSES, UNSAID, CHAT] = [route('openai::gpt-6'), route('deepseek::gpt-6'), route('openai@chat::gpt-6')];
    const said = (seen) => routeTiers({ index, route: seen, sourceLabel: 'OpenAI' });
    const strip = (seen, view = {}) => offeredTiers({ ...said(seen), ...view });
    assert.deepEqual(support(index, RESPONSES), ['unknown', 'listed', 'unknown', 'unknown'], 'nothing confirms any tier for this model or key');
    assert.deepEqual([tierCarried(index, RESPONSES, 'ultrafast'), tierCarried(index, UNSAID, 'ultrafast'), tierCarried(index, CHAT, 'ultrafast'),
        tierCarried(null, RESPONSES, 'ultrafast')], [true, false, false, false]);
    assert.deepEqual(strip(RESPONSES), ['economy', 'standard', 'fast', 'ultrafast'], 'the transport is known to carry it: Ultrafast can be tried');
    assert.equal(said(RESPONSES).explain('ultrafast'),
        'Requests the fastest processing this source offers. It may cost more. Not confirmed for gpt-6 here: a provider may serve it as Standard. Price unknown.',
        'and is said to be unconfirmed, with no price it was not told');
    assert.deepEqual(tierChoices(['standard', 'fast', 'economy', 'ultrafast'], said(RESPONSES)).at(-1), { value: 'ultrafast', label: 'Ultrafast · not confirmed', disabled: false },
        'Settings offers the same unconfirmed attempt as the Main sheet');
    assert.deepEqual(strip(UNSAID), ['economy', 'standard', 'fast'], 'a transport that says nothing offers no new Ultrafast');
    assert.deepEqual([support(index, CHAT), strip(CHAT)], [['unknown', 'listed', 'unknown', 'unlisted'], ['economy', 'standard', 'fast']],
        'a transport that does not carry it: proven absent');
    // A saved or inherited Ultrafast stays in view on any of them, to be seen and left; no model name decides anything.
    for (const seen of [UNSAID, CHAT]) {
        assert.deepEqual(strip(seen, { choice: 'ultrafast' }), ['economy', 'standard', 'fast', 'ultrafast']);
        assert.deepEqual(strip(seen, { resolved: 'ultrafast' }), ['economy', 'standard', 'fast', 'ultrafast']);
    }
    assert.deepEqual(strip(route('openai::any-name-at-all')), ['economy', 'standard', 'fast'], 'a model the catalog does not list has no transport word of its own');
});

test('a question is asked for one binding: the route, its account, and the store and account fingerprint the read names', () => {
    const read = (store, fingerprint) => buildModelIndex({ catalog: { store_id: store, ...codexCatalog([
        { ...account('work', { 'gpt-6-astra': CEILING }), catalog: { source: 'codex', credentialProfileId: 'work', accountFingerprint: fingerprint,
            provenance: 'live', models: [{ id: 'gpt-6-astra', processing: { modes: CEILING } }] } }]) } });
    const asked = tierBinding(read('store-a', 'fp-1'), ASTRA, 'work');
    assert.equal(tierBinding(read('', ''), ASTRA, 'work', read('store-a', 'fp-1')), asked,
        'a scoped read that omits identity retains the known store and fingerprint');
    assert.equal(tierBinding(read('store-a', 'fp-1'), ASTRA, 'work'), asked, 'the same read again is the same binding');
    for (const [other, why] of [[tierBinding(read('store-b', 'fp-1'), ASTRA, 'work'), 'another engine store'],
        [tierBinding(read('store-a', 'fp-2'), ASTRA, 'work'), 'another account under the same name'],
        [tierBinding(read('store-a', 'fp-1'), ASTRA, ''), 'Auto'], [tierBinding(read('store-a', 'fp-1'), ASTRA, null), 'a pin that is not known'],
        [tierBinding(read('store-a', 'fp-1'), SOL, 'work'), 'another model']]) assert.notEqual(other, asked, why);
    assert.notEqual(tierBinding(null, ASTRA, ''), tierBinding(null, ASTRA, null), 'Auto and an unknown pin are told apart with no catalog at all');
});

test('a role\'s Processing in Settings offers the same tiers for its route, an uncertain one worded so, the stored one always', () => {
    const base = ['standard', 'fast', 'economy'];
    const said = (answers) => ({ supportOf: (tier) => answers[tier] || 'unknown' });
    const labels = (...args) => tierChoices(...args).map((choice) => `${choice.value}=${choice.label}`);
    assert.deepEqual(labels(base), ['standard=Standard', 'fast=Fast', 'economy=Flex'], 'before a route is known: the roles\' tiers as they are');
    assert.deepEqual(labels([...base, 'ultrafast']), ['standard=Standard', 'fast=Fast', 'economy=Flex'], 'Ultrafast is offered only where something lists it');
    assert.deepEqual(labels([...base, 'ultrafast'], said({ fast: 'listed', economy: 'unlisted', ultrafast: 'some' })),
        ['standard=Standard', 'fast=Fast', 'ultrafast=Ultrafast · some accounts']);
    assert.deepEqual(labels(base, said({ fast: 'unknown', economy: 'unlisted' }), 'economy'),
        ['standard=Standard', 'fast=Fast · not confirmed', 'economy=Flex · not listed'], 'a stored tier the route does not serve is kept, and said');
    assert.deepEqual(labels(base, said({ fast: 'listed', ultrafast: 'listed' })), ['standard=Standard', 'fast=Fast', 'economy=Flex · not confirmed'],
        'a role that may not hold Ultrafast is not offered it');
    assert.deepEqual(labels(base, null, 'ultrafast'), ['standard=Standard', 'fast=Fast', 'economy=Flex', 'ultrafast=Ultrafast'], 'one already stored can be seen and left');
});

test('Main\'s Ultrafast asks first: the route, what it reaches, what the route says with its price or that it is unknown, and the way back', async () => {
    const { doc } = domDouble({ parseHtml: true });
    const asked = [];
    const answer = { value: true };
    const dialog = async (request) => { asked.push(request); return answer.value; };
    const lines = (request) => { const box = doc.createElement('div'); request.content.mount(box); return box.children.map((line) => [line.textContent, 'i18nSkip' in line.dataset]); };
    assert.equal(await confirmMainUltrafast({ model: 'gpt-6-astra', source: 'Codex', explain: 'The fastest replies. Only some Codex accounts…', dialog, doc }), true);
    assert.deepEqual([asked[0].title, asked[0].confirmLabel, asked[0].body], ['Ultrafast for Main', 'Use Ultrafast', MAIN_ULTRAFAST_ROLLBACK]);
    assert.match(MAIN_ULTRAFAST_ROLLBACK, /older Ouroboros cannot read the settings file.*choose Standard or Default/);
    assert.deepEqual(lines(asked[0]), [['gpt-6-astra · Codex', true], [MAIN_ULTRAFAST_SCOPE, false], [MAIN_ULTRAFAST_ROLES, false],
        ['The fastest replies. Only some Codex accounts…', true], ['What Ultrafast costs on this route is not known here.', false]],
    'no price told: that is said, never a figure');
    assert.match(lines(asked[0])[1][0], /Running and new work.*even on a fallback model.*tier set for the work or fallback takes precedence/);
    await confirmMainUltrafast({ model: 'glm-5.3', source: 'OpenRouter', priceSaid: true,
        explain: 'The fastest replies. $30 in / $150 out per 1M tokens.', dialog, doc });
    assert.deepEqual(lines(asked[1]).map(([text]) => text), ['glm-5.3 · OpenRouter', MAIN_ULTRAFAST_SCOPE, MAIN_ULTRAFAST_ROLES,
        'The fastest replies. $30 in / $150 out per 1M tokens.'], 'the route\'s own price is in its words, and both routes use the same scope disclosure');
    answer.value = false;
    assert.equal(await confirmMainUltrafast({ dialog, doc }), false, 'a cancel is a no');
    answer.value = undefined;
    assert.equal(await confirmMainUltrafast({ dialog, doc }), false);
});

test('the card says what the newest call came to only where it differs from what was asked, each way in its own words', () => {
    assert.equal(servedWords(null), '');
    assert.equal(servedWords('fast'), '');
    assert.equal(servedWords({}), '', 'nothing asked, nothing seen');
    assert.equal(servedWords({ submitted: 'standard', observed: 'standard' }), '', 'nothing asked: nothing to differ from');
    assert.equal(servedWords({ requested: 'fast', submitted: 'fast', observed: 'fast' }), '', 'served as asked');
    // The provider served another tier than the one sent (`provider_mode_changed`).
    assert.equal(servedWords({ requested: 'fast', submitted: 'fast', observed: 'standard' }), 'Fast asked · served Standard');
    assert.equal(servedWords({ requested: 'economy', submitted: 'economy', observed: 'standard' }), 'Flex asked · served Standard');
    assert.equal(servedWords({ requested: 'ultrafast', submitted: 'ultrafast', submittedNative: 'ultrafast', observed: 'fast', observedNative: ['priority'],
        reason: 'provider_mode_changed', source: 'provider_response' }), 'Ultrafast asked · served Fast');
    // Nothing echoed the tier (`observed: 'unknown'`): not confirmed, and never called Standard.
    assert.equal(servedWords({ requested: 'ultrafast', submitted: 'ultrafast', observed: 'unknown', observedNative: [], source: 'host_request' }),
        'Ultrafast asked · not confirmed');
    assert.equal(servedWords({ requested: 'fast', submitted: 'fast', observed: 'unknown' }), 'Fast asked · not confirmed');
    assert.equal(servedWords({ requested: 'standard', submitted: 'standard', observed: 'unknown' }), '', 'an unreported Standard is no news');
    // The route could not carry the tier: none was sent, which is not a Standard that was sent.
    assert.equal(servedWords({ requested: 'fast', submitted: null, submittedNative: null, observed: 'unknown', reason: 'processing_not_submitted' }),
        'Fast asked · sent with no tier');
    assert.equal(servedWords({ requested: 'fast', submitted: null, observed: 'standard', reason: 'processing_not_submitted' }),
        'Fast asked · sent with no tier · served Standard', 'and what a provider then said it served');
    assert.equal(servedWords({ requested: 'standard', submitted: null, observed: 'unknown', reason: 'processing_not_submitted' }), 'Standard asked · sent with no tier',
        'the route omitted the requested Standard tier');
    // Sent as another tier than asked: a native setting of the route, or the host's Standard continuation of a Fast or
    // Flex refused before generating (owner 13.11). The receipt keeps what the call asked, so the deviation is said.
    for (const [tier, word] of [['fast', 'Fast'], ['economy', 'Flex'], ['ultrafast', 'Ultrafast']]) {
        const override = { requested: tier, submitted: 'standard', submittedNative: 'default', observed: 'standard',
            observedNative: ['default'], reason: 'submitted_mode_differs' };
        assert.equal(servedWords(override), `${word} asked · sent as Standard`, 'never read as the provider\'s own change');
        assert.equal(receiptWords(override), `asked ${word} · sent Standard · served Standard`);
        // The same where nothing echoed it: what was sent is still known.
        assert.equal(servedWords({ requested: tier, submitted: 'standard', observed: 'unknown', reason: 'submitted_mode_differs' }), `${word} asked · sent as Standard`);
    }
    assert.equal(servedWords({ requested: 'ultrafast', submitted: 'fast', observed: 'standard' }), 'Ultrafast asked · sent as Fast · served Standard',
        'sent as one tier and served as a third');
});

test('the receipt in full says what was asked, sent and served, and a provider\'s own word that is not the tier\'s', () => {
    assert.equal(receiptWords(undefined), '');
    assert.equal(receiptWords({ requested: 'fast', submitted: 'fast', observed: 'fast', observedNative: ['priority'] }),
        'asked Fast · sent Fast · served Fast', 'priority is Fast in the provider\'s words: nothing more to say');
    assert.equal(receiptWords({ requested: 'ultrafast', submitted: 'ultrafast', observed: 'fast', observedNative: ['priority'], reason: 'provider_mode_changed' }),
        'asked Ultrafast · sent Ultrafast · served Fast');
    assert.equal(receiptWords({ requested: 'fast', submitted: 'fast', observed: 'fast', observedNative: ['ultrafast'] }),
        'asked Fast · sent Fast · served Fast (ultrafast)', 'an older Codex says Ultrafast where it served Fast');
    assert.equal(receiptWords({ requested: 'ultrafast', submitted: 'ultrafast', observed: 'ultrafast', observedNative: ['ultrafast'] }),
        'asked Ultrafast · sent Ultrafast · served Ultrafast');
    assert.equal(receiptWords({ requested: 'economy', submitted: 'economy', observed: 'standard', observedNative: ['default', 7, 'flex'] }),
        'asked Flex · sent Flex · served Standard (default, flex)', 'a word that does not fold into what was seen is said with the others');
    assert.equal(receiptWords({ requested: 'fast', submitted: null, observed: 'unknown', reason: 'processing_not_submitted' }),
        'asked Fast · sent with no tier · served as not reported');
    assert.equal(receiptWords({ submitted: 'standard', observed: 'standard' }), 'sent Standard · served Standard', 'nothing asked: no asked part');
    // No tier reported, whether the provider answered with an explicit null or without the field: neither is Standard.
    for (const source of ['provider_response', 'host_request']) {
        const silent = { requested: 'fast', submitted: 'fast', submittedNative: 'priority', observed: 'unknown', observedNative: [], source };
        assert.deepEqual([receiptWords(silent), servedWords(silent)], ['asked Fast · sent Fast · served as not reported', 'Fast asked · not confirmed'], source);
    }
    // A raw echo that is no tier this page knows (OpenAI's opaque `auto` among them) is said as it came, and is not
    // read as one: the provider reported something, which a call without the field did not.
    for (const word of ['turbo', 'auto']) {
        const strange = { requested: 'fast', submitted: 'fast', observed: 'unknown', observedNative: [word], source: 'provider_response' };
        assert.deepEqual([receiptWords(strange), servedWords(strange)],
            [`asked Fast · sent Fast · served as the provider reported (${word})`, `Fast asked · provider said ${word}`], word);
    }
    // Echoes of several tiers in one call are mixed: neither of them, and never Default.
    const mixed = { requested: 'ultrafast', submitted: 'ultrafast', observed: 'mixed', observedNative: ['ultrafast', 'priority'], source: 'provider_response' };
    assert.deepEqual([receiptWords(mixed), servedWords(mixed)],
        ['asked Ultrafast · sent Ultrafast · served mixed tiers (ultrafast, priority)', 'Ultrafast asked · served mixed tiers']);
    assert.equal(servedWords({ requested: 'fast', submitted: null, observed: 'mixed', observedNative: ['default', 'priority'] }),
        'Fast asked · sent with no tier · served mixed tiers');
});

test('a tier its host transport cannot carry is said not sent; an endpoint list that leaves it out, or an engine\'s list, keeps its words', () => {
    const index = buildModelIndex({ catalog: { model_sources: [], items: [
        { provider_id: 'openai', provider: 'OpenAI', id: 'gpt-6', name: 'gpt-6', value: 'openai::gpt-6', processing_transport: ['standard', 'economy', 'fast'] },
        { provider_id: 'local', provider: 'Local', id: 'qwen3', name: 'qwen3', value: 'qwen3', use_local: true, processing_transport: ['standard'] },
        { provider_id: 'openai', provider: 'OpenAI', id: 'gpt-old', name: 'gpt-old', value: 'openai::gpt-old' }] } });
    const chat = routeTiers({ index, route: route('openai::gpt-6'), sourceLabel: 'OpenAI' });
    assert.equal(chat.explain('ultrafast'), 'Requests the fastest processing this source offers. It may cost more. OpenAI does not send Ultrafast. Price unknown.');
    const local = routeTiers({ index, route: { model: 'anything-loaded', use_local: true }, sourceLabel: 'Local' });
    assert.deepEqual(['economy', 'fast', 'ultrafast'].map((tier) => local.supportOf(tier)), ['unlisted', 'unlisted', 'unlisted'],
        'the local adapter\'s word holds for whatever model it loads');
    assert.match(local.explain('fast'), /Local does not send Fast\.$/);
    assert.deepEqual(tierChoices(['standard', 'fast', 'economy'], local, 'fast').map((choice) => [choice.value, choice.label]),
        [['standard', 'Standard'], ['fast', 'Fast · not sent']], 'a local role keeps its inherited Fast in view, said as not sent');
    // A connection that says nothing of its transport (an older core) proves nothing absent.
    const legacy = routeTiers({ index, route: route('openai::gpt-old') });
    assert.deepEqual(['economy', 'fast'].map((tier) => legacy.supportOf(tier)), ['unknown', 'unknown']);
});

/* The strip on a page, painted as its sheet paints it, with every choice it hands back. */
const MEANS = { economy: 'Requests slower processing at a discount when capacity is available.',
    standard: 'Request Standard processing.', fast: 'Requests faster processing. It may cost more.' };
// What a route usually says: Flex and Fast listed, nothing of Ultrafast; `EVERY` lists them all.
const NO_ULTRAFAST = (tier) => (tier === 'ultrafast' ? 'unknown' : 'listed');
const EVERY = () => 'listed';
function mountStrip(view = {}, onFocusLost) {
    const { doc } = domDouble({ parseHtml: true });
    const chosen = [];
    const strip = createTierStrip({ doc, id: 'sheet-1-tier', onChoose: (tier) => chosen.push(tier), onFocusLost });
    doc.body.append(strip.el);
    strip.paint({ choice: '', resolved: '', scope: 'Default · for new work here', supportOf: NO_ULTRAFAST,
        explain: (tier) => tierExplanation(tier, { support: 'listed' }), ...view });
    const seg = (tier) => strip.segs.find((made) => made.dataset.tier === tier);
    return {
        doc, strip, chosen, seg,
        checked: () => strip.segs.filter((made) => made.getAttribute('aria-checked') === 'true').map((made) => made.dataset.tier),
        ring: () => strip.segs.filter((made) => made.dataset.resolved === 'true').map((made) => made.dataset.tier),
        stop: () => strip.segs.filter((made) => made.tabIndex === 0).map((made) => made.dataset.tier),
        scope: () => strip.el.querySelector('.model-tier-scope').textContent,
        means: () => strip.el.querySelector('.model-tier-means').textContent,
    };
}

test('the strip is one track of the tiers and a Default beside it, with its scope and its meaning under them', () => {
    const s = mountStrip();
    const track = s.strip.el.querySelector('.model-tier-track');
    assert.ok(track.classList.contains('ui-segment-group'), 'the shared segment strip');
    assert.deepEqual([track.getAttribute('role'), track.getAttribute('aria-label'), track.getAttribute('aria-describedby')],
        ['radiogroup', 'Processing tier', 'sheet-1-tier-scope sheet-1-tier-means']);
    assert.deepEqual(s.strip.segs.map((made) => [made.dataset.tier, made.textContent, made.getAttribute('role'), made.className]), [
        ['economy', 'Flex', 'radio', 'ui-segment model-tier-seg'], ['standard', 'Standard', 'radio', 'ui-segment model-tier-seg'],
        ['fast', 'Fast', 'radio', 'ui-segment model-tier-seg'], ['ultrafast', 'Ultrafast', 'radio', 'ui-segment model-tier-seg']]);
    assert.deepEqual([s.strip.reset.dataset.tier, s.strip.reset.textContent, s.strip.reset.hidden], ['', 'Default', true],
        'Default stands only while there is a choice to return from');
    assert.equal(s.strip.el.querySelector('.model-tier-means').getAttribute('aria-live'), 'polite');
    assert.deepEqual([s.scope(), s.means()], ['Default · for new work here', 'No tier requested.'],
        'without a choice or inherited tier, the provider decides');
});

test('a choice is filled; without one the tier the scope takes wears the ring; an explicit Standard is a choice too', () => {
    const s = mountStrip({ resolved: 'fast', supportOf: (tier) => (tier === 'economy' ? 'unlisted' : 'listed') });
    assert.deepEqual([s.checked(), s.ring(), s.stop(), s.strip.reset.hidden, s.means()], [[], ['fast'], ['fast'], true, MEANS.fast]);
    assert.deepEqual(s.strip.segs.map((made) => made.dataset.support), ['unlisted', 'listed', 'listed', 'listed']);
    s.strip.paint({ choice: 'economy', resolved: 'economy', scope: 'Flex · for new work here' });
    assert.deepEqual([s.checked(), s.ring(), s.stop(), s.strip.reset.hidden, s.scope(), s.means()],
        [['economy'], [], ['economy'], false, 'Flex · for new work here', MEANS.economy]);
    s.strip.paint({ choice: 'standard', resolved: 'standard' });
    assert.deepEqual([s.checked(), s.ring(), s.strip.reset.hidden], [['standard'], [], false], 'Standard chosen is not Default');
    s.strip.paint({ choice: '', resolved: '' });
    assert.deepEqual([s.checked(), s.ring(), s.stop(), s.strip.reset.hidden], [[], [], ['standard'], true]);
});

test('a press chooses at once, Default included; the chosen tier and a busy strip choose nothing', () => {
    const s = mountStrip({ choice: 'economy', resolved: 'economy' });
    s.seg('fast').click();
    s.seg('economy').click();
    s.strip.reset.click();
    assert.deepEqual(s.chosen, ['fast', ''], 'the chosen tier again is no act');
    s.strip.paint({ busy: true });
    // A save in flight is said (aria-disabled) and its presses do nothing; a disabled button would drop the focus.
    assert.ok(s.strip.segs.every((made) => made.getAttribute('aria-disabled') === 'true' && !made.disabled)
        && s.strip.reset.getAttribute('aria-disabled') === 'true', 'a save in flight takes no press and keeps the focus');
    s.strip.el.emit('click', { target: s.seg('standard') });
    assert.deepEqual(s.chosen, ['fast', ''], 'even a press that reaches the strip');
    s.strip.paint({ busy: false });
    assert.equal(s.seg('standard').getAttribute('aria-disabled'), 'false');
    s.seg('standard').click();
    assert.deepEqual(s.chosen, ['fast', '', 'standard']);
});

test('the pointer or the focus on a tier says what it means before a press; arrows move among tiers without choosing', () => {
    const s = mountStrip({ resolved: 'standard', supportOf: EVERY });
    s.seg('fast').emit('pointerover');
    assert.equal(s.means(), MEANS.fast);
    s.strip.el.emit('pointerleave');
    assert.equal(s.means(), MEANS.standard, 'the tier taken again');
    s.seg('economy').emit('focusin');
    assert.equal(s.means(), MEANS.economy);
    s.seg('economy').emit('focusout');
    assert.equal(s.means(), MEANS.standard);
    const right = s.seg('standard').emit('keydown', { key: 'ArrowRight' });
    assert.deepEqual([right.defaultPrevented, s.doc.activeElement, s.stop(), s.chosen], [true, s.seg('fast'), ['fast'], []]);
    s.seg('fast').emit('keydown', { key: 'ArrowDown' });
    assert.equal(s.doc.activeElement, s.seg('ultrafast'));
    s.seg('ultrafast').emit('keydown', { key: 'ArrowRight' });
    assert.equal(s.doc.activeElement, s.seg('ultrafast'), 'the last tier is the last');
    s.seg('ultrafast').emit('keydown', { key: 'Home' });
    assert.deepEqual([s.doc.activeElement, s.stop()], [s.seg('economy'), ['economy']]);
    s.seg('economy').emit('keydown', { key: 'ArrowLeft' });
    assert.equal(s.doc.activeElement, s.seg('economy'), 'the first is the first');
    s.seg('economy').emit('keydown', { key: 'End' });
    assert.equal(s.doc.activeElement, s.seg('ultrafast'));
    s.seg('ultrafast').emit('keydown', { key: 'ArrowUp' });
    assert.equal(s.doc.activeElement, s.seg('fast'));
    assert.equal(s.seg('fast').emit('keydown', { key: 'Enter' }).defaultPrevented, undefined, 'Enter is the button\'s own press');
    assert.deepEqual(s.chosen, [], 'moving chose nothing');
    s.strip.destroy();
    assert.equal(s.strip.el.listenerCount(), 0, 'destroy takes its listeners');
});

test('a finger previews nothing, its tap chooses; the meaning\'s box is as tall as its tallest explanation', () => {
    const s = mountStrip({ resolved: 'standard' });
    s.seg('fast').emit('pointerover', { pointerType: 'touch' });
    assert.equal(s.means(), MEANS.standard, 'a finger on a tier changes no line under it before its tap lands');
    s.seg('fast').emit('pointerover', { pointerType: 'mouse' });
    assert.equal(s.means(), MEANS.fast, 'the mouse previews');
    s.strip.el.emit('pointerleave', { pointerType: 'mouse' });
    // Every explanation a shown tier may say, each in a hidden copy in the box's one cell: a preview never grows it.
    const box = s.strip.el.querySelector('.model-tier-meansbox');
    const sizer = box.querySelector('.model-tier-sizer');
    assert.deepEqual([box.children.map((node) => node.className), sizer.getAttribute('aria-hidden')],
        [['model-tier-means', 'model-tier-sizer'], 'true']);
    assert.deepEqual(sizer.children.map((line) => line.textContent), [MEANS.economy, MEANS.standard, MEANS.fast]);
    const kept = sizer.children[0];
    s.strip.paint({ scope: 'Fast · for new work here' });
    assert.ok(sizer.children[0] === kept, 'the same explanations are not written again');
    s.strip.paint({ supportOf: EVERY });
    assert.equal(sizer.children.length, 4, 'an offered Ultrafast has its copy too');
});

test('Default pressed by the keyboard leaves with its choice and hands the focus to the tier the scope takes', () => {
    const s = mountStrip({ choice: 'fast', resolved: 'fast' });
    s.strip.reset.focus();
    s.strip.reset.click();
    assert.deepEqual(s.chosen, ['']);
    s.strip.paint({ choice: '', resolved: 'standard' });  // its save answered
    assert.deepEqual([s.strip.reset.hidden, s.doc.activeElement], [true, s.seg('standard')], 'never lost to the page');
});

test('the strip shows the tiers its route offers, each wearing what the route says; with none to offer it is not there', () => {
    const shown = (s) => TIERS.filter((tier) => !s.seg(tier).hidden);
    const usual = mountStrip();
    assert.deepEqual([usual.strip.el.hidden, shown(usual)], [false, ['economy', 'standard', 'fast']], 'Ultrafast nothing lists is not offered');
    // Some accounts list Ultrafast, nothing confirms Fast, the transport cannot carry Flex.
    const says = { economy: 'unlisted', fast: 'unknown', ultrafast: 'some' };
    const mixed = mountStrip({ supportOf: (tier) => says[tier] || 'listed' });
    assert.deepEqual([shown(mixed), mixed.strip.segs.map((made) => made.dataset.support)],
        [['standard', 'fast', 'ultrafast'], ['unlisted', 'listed', 'unknown', 'some']], 'a tier proven absent leaves; an uncertain one stays, marked');
    // The scope's own Flex, or the one it inherits, stays in view where the route does not list it, and can be left.
    mixed.strip.paint({ choice: 'economy', resolved: 'economy' });
    assert.deepEqual([shown(mixed), mixed.checked(), mixed.strip.reset.hidden, mixed.seg('economy').dataset.support],
        [['economy', 'standard', 'fast', 'ultrafast'], ['economy'], false, 'unlisted']);
    mixed.strip.paint({ choice: '', resolved: 'economy' });
    assert.deepEqual([shown(mixed), mixed.ring(), mixed.stop()], [['economy', 'standard', 'fast', 'ultrafast'], ['economy'], ['economy']],
        'Default is not Standard: the inherited tier wears the ring');
    // A route proven to have only its ordinary service: no strip, until something was chosen for it.
    const only = mountStrip({ ordinaryDefault: true, supportOf: (tier) => (tier === 'standard' ? 'listed' : 'unlisted') });
    assert.deepEqual([only.strip.el.querySelector('.model-tier-row').hidden, shown(only)], [true, []]);
    only.strip.paint({ resolved: 'standard' });
    assert.equal(only.strip.el.querySelector('.model-tier-row').hidden, true, 'Standard by Default there is nothing to choose');
    only.strip.paint({ choice: 'standard', resolved: 'standard' });
    assert.deepEqual([only.strip.el.querySelector('.model-tier-row').hidden, shown(only), only.checked(), only.strip.reset.hidden], [false, ['standard'], ['standard'], false],
        'an explicit Standard is a choice: it stands with Default beside it');
    only.strip.paint({ choice: '', resolved: 'fast' });
    assert.deepEqual([only.strip.el.querySelector('.model-tier-row').hidden, shown(only), only.ring()], [false, ['standard', 'fast'], ['fast']], 'an inherited Fast is said, and can be overridden');
    // The catalog arriving under the keyboard: a tier it proves absent hands its focus to the tier taken now.
    const moving = mountStrip({ resolved: 'standard' });
    moving.seg('fast').focus();
    moving.strip.paint({ supportOf: (tier) => (tier === 'fast' ? 'unlisted' : NO_ULTRAFAST(tier)) });
    assert.deepEqual([shown(moving), moving.doc.activeElement], [['economy', 'standard'], moving.seg('standard')]);
});

test('reset gives focus back to the sheet if it hides every tier, not to a hidden control', () => {
    let moved = 0;
    const s = mountStrip({ choice: 'standard', resolved: 'standard', ordinaryDefault: true,
        supportOf: (tier) => tier === 'standard' ? 'listed' : 'unlisted' }, () => { moved += 1; });
    s.strip.reset.focus();
    s.strip.reset.click();
    s.strip.paint({ choice: '', resolved: '' });
    assert.deepEqual([s.chosen, s.stop(), moved, s.strip.reset.hidden], [[''], [], 1, true]);
});

test('unavailable Ultra stays visible for an own or inherited choice without authoring it again', () => {
    for (const support of ['unknown', 'unlisted']) {
        for (const choice of ['', 'ultrafast']) {
            const s = mountStrip({ choice, resolved: 'ultrafast', supportOf: (tier) => tier === 'ultrafast' ? support : 'listed' });
            const ultra = s.seg('ultrafast');
            assert.equal(ultra.hidden, false);
            assert.equal(ultra.getAttribute('aria-disabled'), 'true');
            ultra.click();
            assert.deepEqual(s.chosen, []);
            (choice ? s.strip.reset : s.seg('standard')).click();
            assert.deepEqual(s.chosen, [choice ? '' : 'standard']);
        }
    }
});

test('a hidden Ultrafast is never a keyboard stop: End and arrows stop at Fast', () => {
    const s = mountStrip({ resolved: 'standard' });
    s.seg('standard').emit('keydown', { key: 'End' });
    assert.equal(s.doc.activeElement, s.seg('fast'));
    s.seg('fast').emit('keydown', { key: 'ArrowRight' });
    assert.equal(s.doc.activeElement, s.seg('fast'), 'Fast is the last shown tier');
});
