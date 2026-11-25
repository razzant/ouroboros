import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { MAIN_FOLLOW_NOTE, composeModelSource, createModelRolesEditor, inheritTargetOf, modelContextNote,
    modelSourceGroups, parseModelSource, sourceChoice, sourceFromChoice } from '../modules/model_roles.js';
import { configuredApiProviders, routeChoiceGroups } from '../modules/route_editor_primitives.js';
import { settingsTitle } from '../modules/settings_home.js';

const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url), 'utf8'));

test('a model role names its destination without repeating the source section, while the parent stays Models', () => {
    assert.equal(settingsTitle('models'), 'Models');
    for (const [slot, name] of [['main', 'Main model'], ['light', 'Light model'],
        ['consciousness', 'Consciousness model'], ['websearch', 'Web search model'], ['fallback/2', 'Fallback model 2']]) {
        assert.equal(settingsTitle(`models/role/${slot}`), name);
    }
    assert.equal(settingsTitle('models'), 'Models');
});

test('source/model spelling is reversible and never embeds the account', () => {
    for (const value of ['google/a', 'openai::b', 'claudexor::codex=gpt-test', 'custom::x=y']) {
        const { source, model } = parseModelSource(value);
        assert.equal(composeModelSource(source, model), value);
    }
    assert.equal(composeModelSource('subscription:codex', ''), '');
    assert.equal(composeModelSource('openrouter', 'openai::owner-model'), 'openai::owner-model');
});

test('Models shows only model-capable sources and only providers with a stored key', () => {
    const providers = configuredApiProviders({ OPENAI_API_KEY: '***set***' }, contract.providerProfiles);
    const groups = modelSourceGroups({ sources: [{ id: 'codex', label: 'Codex' }], providers,
        providerProfiles: contract.providerProfiles });
    assert.deepEqual(groups[0].options.map((row) => row.value), ['subscription:codex']);
    assert.deepEqual(groups[1].options.map((row) => row.value), ['api:openai', '']);
    assert.equal(groups[1].options[0].label, 'OpenAI');
    // Every other provider is a key the owner has not added; the tail says where.
    assert.equal(groups[1].options.at(-1).disabled, true);
    assert.match(groups[1].options.at(-1).label, /Accounts/);
    // A model role cannot be delivered by an agent session, so that group is absent.
    assert.deepEqual(groups.map((group) => group.label), ['Subscriptions · models', 'API keys']);
    const saved = modelSourceGroups({ current: 'subscription:future' });
    assert.equal(saved[0].options[0].value, 'subscription:future');
    assert.match(saved[0].options[0].label, /not checked/);
});

test('Models and the route editors draw the same groups, in the same order, from one builder', () => {
    const args = { sources: [{ id: 'codex', label: 'Codex' }],
        providers: configuredApiProviders({ OPENROUTER_API_KEY: 'k' }), catalogKnown: true, accountsKnown: true };
    const shared = routeChoiceGroups({ modelSources: args.sources, providers: args.providers,
        catalogKnown: true, accountsKnown: true, includeSessions: false });
    assert.deepEqual(modelSourceGroups(args), shared);
    // "Uses Main" is the only entry a model role adds on top of the shared list.
    const inherited = modelSourceGroups({ ...args, current: 'inherit' });
    assert.deepEqual(inherited[0].options, [{ value: 'inherit', label: 'Uses Main' }]);
    assert.deepEqual(inherited.slice(1), shared);
});

test('a saved source whose key was removed stays selectable and says the key is gone', () => {
    const groups = modelSourceGroups({ current: 'anthropic',
        providers: configuredApiProviders({ OPENROUTER_API_KEY: 'k' }) });
    const api = groups.find((group) => group.label === 'API keys').options;
    assert.deepEqual(api.map((row) => row.value), ['api:openrouter', 'api:anthropic', '']);
    assert.equal(api[1].label, 'Anthropic (no key)');
    assert.ok(!api[1].disabled, 'the owner can still keep their own assignment');
});

test('the select vocabulary is shared while the stored spelling is unchanged', () => {
    for (const [source, choice] of [['openrouter', 'api:openrouter'], ['openai', 'api:openai'],
        ['subscription:codex', 'subscription:codex'], ['inherit', 'inherit']]) {
        assert.equal(sourceChoice(source), choice);
        assert.equal(sourceFromChoice(choice), source);
        assert.ok(!choice.includes('::'), choice);
    }
});

test('Models distinguishes an unread Accounts facet from a confirmed empty account set', () => {
    for (const facts of [{}, { catalogKnown: true }, { accountsKnown: true }]) {
        const groups = modelSourceGroups(facts);
        assert.doesNotMatch(groups[0].options[0].label, /connect one/i);
    }
    const noAccounts = modelSourceGroups({ catalogKnown: true, accountsKnown: true, hasConfiguredAccounts: false });
    assert.match(noAccounts[0].options[0].label, /connect one in Accounts/i);
    const empty = modelSourceGroups({ catalogKnown: true, accountsKnown: true, hasConfiguredAccounts: true });
    assert.match(empty[0].options[0].label, /No model sources listed/);
    assert.match(empty[0].options[0].label, /press Read again in Models/);
    assert.doesNotMatch(empty[0].options[0].label, /connect one in Accounts/i);
    const saved = modelSourceGroups({ current: 'subscription:owner-source' });
    assert.equal(saved[0].options[0].value, 'subscription:owner-source');
});

test('role pins and context survive a no-edit save, including identical model names', () => {
    const editor = createModelRolesEditor({ hostId: 'test', doc: () => null });
    const settings = {
        OUROBOROS_MODEL: 'claudexor::codex=gpt-test',
        OUROBOROS_MODEL_LIGHT: 'claudexor::codex=gpt-test',
        OUROBOROS_MODEL_FALLBACKS: 'claudexor::codex=gpt-second, openai::model, openai/gpt-5.6-terra',
        OUROBOROS_MODEL_ACCOUNTS: { main: 'personal', light: 'work', vision: 'work', fallback: ['reserve', '', ''], websearch: 'saved' },
        OUROBOROS_MODEL_CONTEXT_WINDOWS: { main: 1000000, fallback: [872000, 0, 0], deep_review: 250000 },
    };
    editor.load(settings, contract);
    const after = editor.collect();
    for (const [key, value] of Object.entries(settings)) assert.deepEqual(after[key], value, key);
    editor.adoptCatalog({ model_sources: [{ id: 'codex', label: 'Codex' }], items: [] });
    assert.deepEqual(editor.collect(), after, 'new discovery never rewrites the assignment');
    assert.equal(editor.validate(), '');
    editor.destroy();
});

test('image generation has no settings default or chat model role', () => {
    const editor = createModelRolesEditor({ hostId: 'test', doc: () => null });
    editor.load({ OUROBOROS_MODEL: 'openai::model', OUROBOROS_MODEL_IMAGE: 'legacy' }, contract);
    const saved = editor.collect();
    assert.ok(!('OUROBOROS_MODEL_IMAGE' in saved));
    assert.ok(!contract.modelSlots.some((slot) => slot.slot === 'image'));
    assert.equal(editor.validate(), '');
    editor.destroy();
});

test('an unloaded editor authors nothing and an API-only save does not create role maps', () => {
    const editor = createModelRolesEditor({ hostId: 'test', doc: () => null });
    assert.deepEqual(editor.collect(), {});
    editor.load({ OUROBOROS_MODEL: 'openai::model' }, contract);
    const setting = editor.collect();
    assert.equal(setting.OUROBOROS_MODEL, 'openai::model');
    assert.ok(!('OUROBOROS_MODEL_ACCOUNTS' in setting));
    assert.ok(!('OUROBOROS_MODEL_CONTEXT_WINDOWS' in setting));
    editor.destroy();
});

test('context note uses the server projection, never catalogue arithmetic', () => {
    // The projection's typed fields become owner words; its own sentence (for Main) is not copied.
    const window = { tokens: 872000, kind: 'total', source: 'provider_metadata', observed_at: '2026-11-05T16:33:00Z',
        reserve: 65536, usable: 806464, text: 'window 872,000 total: server metadata 2026-11-05 16:33; usable input 806,464; reply reserve 65,536' };
    assert.equal(modelContextNote({ window }), 'Window 872k tokens (from the server) · 806k usable for input');
    assert.equal(modelContextNote({ window: { ...window, source: 'user_setting', usable: undefined,
        contradiction: { source: 'provider_metadata', tokens: 272000, kind: 'total' } } }),
    'Window 872k tokens (set in Settings) · 806k usable for input · the server says 272k');
    assert.equal(modelContextNote({ window: { ...window, retired_ack: { window_tokens: 1000000, ts: '2026-11-01T10:00:00Z' } } }),
        'Window 872k tokens (from the server) · 806k usable for input · old confirmation 1M not used');
    assert.doesNotMatch(modelContextNote({ window }), /2026|65,536|reserve|metadata/);
    assert.equal(modelContextNote({ window: { tokens: 1050000, source: 'user_setting', usable: 984464 } }),
        'Window 1.1M tokens (set in Settings) · 984k usable for input');
    assert.match(modelContextNote({ context_window: 272000, max_context_window: 872000 }), /not known/);
    assert.match(modelContextNote(null), /not known/);
});

test('a window a refusal set says how its number was known, and claims no basis the projection does not name', () => {
    // A size refusal's input limit in window_projection's shape; the numbers are made up, no provider was asked.
    const window = { tokens: 883899, kind: 'input', source: 'refusal', observed_at: '2026-11-15T12:00:00+00:00',
        reserve: 65536, usable: 883899, admitted: 883899, contradiction: null, retired_ack: {} };
    const note = (basis) => modelContextNote({ window: { ...window, basis } });
    const said = (words) => `Window 884k tokens (${words}) · 884k usable for input`;
    assert.equal(note('provider_stated'), said('stated by the provider in a refusal'));
    // A local instance counted the request Ouroboros refused before sending: no server stated or withheld a number.
    assert.equal(note('exact_local_count'), said('derived from an exact local count before sending'));
    // Derived by Ouroboros from a numberless refusal: an anchored estimate or a confirmed declared total.
    assert.equal(note('anchored_estimate'), said('derived from a refusal that stated no number'));
    assert.equal(note('unrecorded'), said('from an earlier refusal, basis not recorded'));
    // A server from before bases were kept sends none; an empty, unknown or inherited name is never guessed at.
    assert.equal(modelContextNote({ window }), said('from a refusal, basis unknown'));
    for (const basis of ['', null, 'sent_estimate', 'toString', '__proto__']) assert.equal(note(basis), said('from a refusal, basis unknown'));
    for (const basis of ['provider_stated', 'exact_local_count', 'anchored_estimate', 'unrecorded', undefined]) {
        assert.doesNotMatch(note(basis), /2026|65,536|reserve|\(refusal\)|_/);  // stamps and codes stay in the record
    }
    // A basis is a refusal's alone: every other source keeps its words, whatever a projection carries beside it.
    for (const [source, words] of [['provider_metadata', 'from the server'], ['user_setting', 'set in Settings'],
        ['owner_ack', 'your confirmation'], ['local_health', 'the local serving window'], ['accepted', 'accepted']]) {
        for (const basis of [undefined, '', 'provider_stated']) {
            assert.equal(modelContextNote({ window: { ...window, source, basis } }), said(words), `${source} ${basis}`);
        }
    }
});

test('the same role sheets are loaded by both actual UI hosts', () => {
    for (const file of ['index.html', 'onboarding_template.html']) {
        const html = readFileSync(new URL(`../${file}`, import.meta.url), 'utf8');
        // The wizard loads the core's own tree whatever main interface is selected.
        const sheet = `${file === 'index.html' ? '' : '/onboarding'}/static/model_roles.css`;
        assert.ok(html.includes(`href="${sheet}"`), `${file} loads ${sheet}`);
        // The review lanes editor is retired: reviewers are rows of the subagent catalog.
        assert.ok(!html.includes('reviewer_slots.css'), `${file} still loads the retired review lanes sheet`);
    }
});

test('role disclosures wear the one shared disclosure look on their own line', () => {
    const css = readFileSync(new URL('../model_roles.css', import.meta.url), 'utf8');
    const block = (selector) => css.match(new RegExp(`${selector}\\s*\\{([^}]*)\\}`))?.[1] ?? '';
    // The caret and the quiet summary are ui.css details.ui-disclosure's; the role sheet draws no marker of its own.
    assert.doesNotMatch(css, /summary::before|summary::-webkit-details-marker/);
    assert.match(readFileSync(new URL('../modules/model_roles.js', import.meta.url), 'utf8'), /<details class="model-role-details ui-disclosure">/);
    // The disclosure owns a full row under the status text instead of sharing its baseline.
    assert.match(block('\\.model-role-notes'), /flex-wrap:\s*wrap/);
    assert.match(block('\\.model-role-notes\\s*>\\s*\\.model-role-details'), /flex-basis:\s*100%/);
});

test('an empty role names the slot it uses: Main, and the Light model for the Safety Supervisor', () => {
    assert.equal(inheritTargetOf({ slot: 'vision' }), 'main');
    assert.equal(inheritTargetOf({ slot: 'light' }), 'main');
    assert.equal(inheritTargetOf({ slot: 'safety' }), 'light');
    assert.equal(inheritTargetOf({ slot: 'future', inherits: 'consciousness' }), 'consciousness', 'a contract row may name its own');
    assert.deepEqual(modelSourceGroups({ current: 'inherit', inheritLabel: 'Uses Light' })[0].options,
        [{ value: 'inherit', label: 'Uses Light' }]);
    const source = readFileSync(new URL('../modules/model_roles.js', import.meta.url), 'utf8');
    assert.match(source, /`Empty uses the \$\{inheritName\(row\)\} model`/);
    assert.match(source, /`Uses \$\{inheritName\(row\)\}\.`/);
});

/** An element stub that accepts every property, call and query (the house idiom: no jsdom). */
function inertNode() {
    return new Proxy(function () {}, {
        get: (target, key) => (key === Symbol.iterator ? function* () {} : key === Symbol.toPrimitive ? () => ''
            : key === 'then' ? undefined : key in target ? target[key] : inertNode()),
        set: (target, key, value) => { target[key] = value; return true; },
        apply: () => inertNode(),
    });
}

test('Settings says under Main, before Save, what a new Main does to running work; no other host does', () => {
    // Its fact, not its words (DESIGN principle 7): work that follows Main switches at its next call, and work with
    // its own model is not said to switch. The note forecasts no tokens or cost (DESIGN §7).
    assert.match(MAIN_FOLLOW_NOTE, /\bfollow(?:s|ing)? Main\b/);
    assert.match(MAIN_FOLLOW_NOTE, /\bnext (?:call|request|step|turn)\b/);
    assert.match(MAIN_FOLLOW_NOTE, /\bswitch(?:es)?\b|\buses this model\b/i);
    assert.doesNotMatch(MAIN_FOLLOW_NOTE, /\b(?:not|never|no longer)\b|n't/i);
    assert.doesNotMatch(MAIN_FOLLOW_NOTE, /\bown model\b[^.]*\bswitch/i);
    assert.doesNotMatch(MAIN_FOLLOW_NOTE, /token|cost|\$|\d/i);
    const rendered = (mainNote) => {
        const host = inertNode();
        const editor = createModelRolesEditor({ hostId: 'roles', doc: { getElementById: () => host }, showContext: false, mainNote });
        editor.load({ OUROBOROS_MODEL: 'openai::main', OUROBOROS_MODEL_FALLBACKS: 'openai::reserve' }, contract);
        editor.destroy();
        return String(host.innerHTML);
    };
    const html = rendered(true);
    const main = html.slice(html.indexOf('data-model-role-group="main"'), html.indexOf('data-model-role-group="light"'));
    assert.ok(main.includes(`<p class="model-role-copy" id="roles-main-note" data-main-follow-note>${MAIN_FOLLOW_NOTE}</p>`));
    assert.match(main, /data-model-role-model aria-label="Main Model" aria-describedby="roles-main-note"/);
    assert.equal(html.split('data-main-follow-note').length, 2, 'one note, under Main and not under Fallback');
    assert.ok(!rendered(false).includes('data-main-follow-note'), 'the wizard and the wait card never show it');
});

test('each role says what it serves once, visibly under its name, never as a hover title', () => {
    const rendered = (slots) => {
        const host = inertNode();
        const editor = createModelRolesEditor({ hostId: 'roles', doc: { getElementById: () => host }, showContext: false });
        editor.load({ OUROBOROS_MODEL: 'openai::main' }, { ...contract, ...(slots ? { modelSlots: slots } : {}) });
        editor.destroy();
        return String(host.innerHTML);
    };
    const html = rendered();
    const group = (slot) => html.slice(html.indexOf(`data-model-role-group="${slot}"`)).split('</section>')[0];
    const light = contract.modelSlots.find((slot) => slot.slot === 'light');
    assert.match(light.note, /Memory consolidation/);
    assert.ok(group('light').includes(`<p class="model-role-copy ui-field-help" data-model-role-note>${light.note}</p>`));
    assert.doesNotMatch(html, /<h4 title=/, 'touch and keyboard readers never meet a hover-only caption');
    for (const slot of contract.modelSlots) assert.equal(group(slot.slot).split('data-model-role-note').length, 2, `${slot.slot}: one note`);
    assert.match(group('fallback'), /data-model-role-note>Tried in this order/);
    // A host whose role carries no note (the wait card's picker) draws none.
    assert.ok(!rendered([{ slot: 'main', settingKey: 'model', inputId: 'wait-model', label: 'Model' }]).includes('data-model-role-note'));
});

test('a kept fallback preserves its unfinished field while source choices follow a new settings read', async (t) => {
    const { Stub, world, settle } = await import('./helpers/settings_dom.js');
    const { env } = world(t);
    const host = new Stub(); host.id = 'roles'; env.document.body.append(host);
    const editor = createModelRolesEditor({ hostId: 'roles', doc: env.document,
        only: { slot: 'fallback', index: 2 }, store: { snapshot: {}, accountsKnown: true } });
    const settings = { OUROBOROS_MODEL: 'owner/main', OPENROUTER_API_KEY: 'key',
        OUROBOROS_MODEL_FALLBACKS: 'owner/first, owner/second, owner/third',
        OUROBOROS_MODEL_CONTEXT_WINDOWS: { fallback: [0, 0, 0] } };
    try {
        editor.load(settings, contract);
        const field = host.querySelector('[data-model-role-context]');
        const details = host.querySelector('.model-role-details'); details.open = true;
        field.value = ''; field.validity = { badInput: true };  // the native number field holding "1e"
        await field.fire('input', { target: field });
        const choices = () => host.querySelector('[data-model-role-source]').querySelectorAll('option').map(node => node.getAttribute('value'));
        assert.ok(!choices().includes('api:openai'));
        editor.load({ ...settings, OPENAI_API_KEY: 'new key', OUROBOROS_MODEL: 'owner/new-main' }, contract, { keep: true });
        assert.equal(host.querySelector('[data-model-role-context]'), field, 'the browser keeps the unfinished field itself');
        assert.equal(host.querySelector('.model-role-details'), details);
        assert.equal(details.open, true);
        assert.equal(field.validity.badInput, true);
        assert.equal(editor.entry().value, 'owner/third');
        assert.match(editor.entry().errors[0], /context window must be a positive whole number/);
        assert.ok(choices().includes('api:openai'), 'the newly added account still reaches the kept form');
        assert.equal(editor.collect().OUROBOROS_MODEL, 'owner/new-main');
        // Discarding the draft still takes the stored field anew, including a new numeric value.
        editor.load({ ...settings, OUROBOROS_MODEL_CONTEXT_WINDOWS: { fallback: [0, 0, 32768] } }, contract);
        assert.notEqual(host.querySelector('[data-model-role-context]'), field);
        assert.equal(editor.entry().context, 32768);
        assert.deepEqual(editor.entry().errors, []);
        await settle();
    } finally { editor.destroy(); }
});


test('a clean fallback refresh takes a new saved context window', async (t) => {
    const { Stub, world, settle } = await import('./helpers/settings_dom.js');
    const { env } = world(t);
    const host = new Stub(); host.id = 'roles'; env.document.body.append(host);
    const editor = createModelRolesEditor({ hostId: 'roles', doc: env.document,
        only: { slot: 'fallback', index: 0 }, store: { snapshot: {}, accountsKnown: true } });
    const settings = { OUROBOROS_MODEL_FALLBACKS: 'owner/first', OPENROUTER_API_KEY: 'key' };
    try {
        editor.load(settings, contract);
        editor.load({ ...settings, OUROBOROS_MODEL_CONTEXT_WINDOWS: { fallback: [32768] } }, contract);
        assert.equal(editor.entry().context, 32768);
        assert.deepEqual(editor.entry().errors, []);
        await settle();
    } finally { editor.destroy(); }
});

/* --- a role's Processing follows its own route as it is bound (DESIGN "Composer owner controls", the tier strip) --- */

// Codex with two accounts as one read gives them: `home` lists Ultrafast for gpt-6-astra, `work` does not; the
// transport carries no Flex.
const CODEX_TIERS = (() => {
    const account = (id, modes) => ({ credentialProfileId: id, availability: 'available', problem: null,
        catalog: { source: 'codex', credentialProfileId: id, provenance: 'live', models: [{ id: 'gpt-6-astra', processing: { modes } }] } });
    return { errors: [], model_sources: [{ id: 'codex', label: 'Codex', credentialHarness: 'codex', processingPreferences: ['standard', 'fast', 'ultrafast'] }],
        account_catalogs: [{ source: 'codex', partial: false, accounts: [account('home', ['standard', 'fast', 'ultrafast']), account('work', ['standard', 'fast'])] }],
        items: ['home', 'work'].map((profile) => ({ provider_id: 'claudexor', provider: 'Codex', source_id: 'codex', id: 'gpt-6-astra', name: 'gpt-6-astra',
            value: 'claudexor::codex=gpt-6-astra', credential_profile_id: profile })) };
})();

/* One role alone on its page: its scoped catalog read answered by `fetched.value` (an Error: the read fails), its
   question and its endpoint reads recorded. `answer.value` is what the question is answered, or a function run while
   it stands. */
async function rolePage(t, slot, settings, { answer = { value: true }, fetched = { value: CODEX_TIERS }, detail = () => null, readDetail = () => {} } = {}) {
    const { Stub, world, settle } = await import('./helpers/settings_dom.js');
    const { env } = world(t);
    const host = new Stub(); host.id = 'roles'; env.document.body.append(host);
    const seen = { questions: [], endpoints: [], changes: 0 };
    const editor = createModelRolesEditor({ hostId: 'roles', doc: env.document, only: { slot, index: 0 }, store: { snapshot: {}, accountsKnown: true },
        catalogFetch: async () => { if (fetched.value instanceof Error) throw fetched.value; return fetched.value; }, onChange: () => { seen.changes += 1; },
        tierStore: { detail, readDetail: async (route) => { seen.endpoints.push(route.model); return readDetail(route); }, subscribe: () => () => {} },
        askUltrafast: async (facts) => { seen.questions.push(facts); return typeof answer.value === 'function' ? answer.value() : answer.value; } });
    t.after(() => editor.destroy());
    editor.load(settings, contract);
    await settle();
    const select = () => host.querySelector('[data-model-role-processing]');
    const put = (selector, value) => { const node = host.querySelector(selector); node.value = value; return node.fire('change', { target: node }); };
    const set = async (selector, value) => { await put(selector, value); await settle(); };
    return { editor, host, seen, answer, fetched, select, settle, put,
        field: () => host.querySelector('[data-processing-field]'),
        options: () => select().querySelectorAll('option').map((node) => [node.getAttribute('value'), node.textContent]),
        note: () => host.querySelector('[data-processing-capability]').textContent,
        account: (id) => set('[data-model-role-account]', id), tier: (value) => set('[data-model-role-processing]', value) };
}
const MAIN_ON_CODEX = { OUROBOROS_MODEL: 'claudexor::codex=gpt-6-astra', OUROBOROS_MODEL_ACCOUNTS: { main: 'work' } };

test('the role page hides redundant Standard processing without hiding a reset or an inherited override', async (t) => {
    const settings = { OUROBOROS_MODEL: 'zai::example' };
    const page = await rolePage(t, 'main', settings);
    assert.equal(page.field().hidden, false, 'not read is not Standard-only');
    page.editor.adoptCatalog({ items: [{ id: 'example', value: settings.OUROBOROS_MODEL, processing_transport: ['standard'] }] });
    assert.equal(page.field().hidden, true);
    assert.match(page.note(), /Standard is the only listed tier/);
    assert.equal(page.editor.entry().processing, '', 'catalog information did not create a draft override');

    for (const tier of ['standard', 'fast']) {
        page.editor.load({ ...settings, OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: tier } }, contract);
        assert.equal(page.field().hidden, false, `a saved ${tier} can still be cleared`);
        assert.equal(page.editor.entry().processing, tier);
        await page.tier('');
        assert.equal(page.editor.entry().processing, '');
        assert.equal(page.field().hidden, true);
    }
    page.editor.load({ ...settings, OUROBOROS_PROCESSING_PREFERENCE: 'fast' }, contract);
    assert.equal(page.field().hidden, false, 'inherited Fast remains visible and overridable');
    assert.match(page.host.querySelector('[data-processing-summary]').textContent, /Fast/);
    assert.ok(page.options().some(([tier]) => tier === 'standard'));
    await page.tier('standard');
    assert.equal(page.editor.entry().processing, 'standard');
});

test('a Standard-only endpoint list does not erase the OpenRouter workspace override', async (t) => {
    const settings = { OUROBOROS_MODEL: 'vendor/example' };
    const page = await rolePage(t, 'main', settings, { detail: () => ({ state: 'ok', complete: true, modes: new Set(['standard']), endpoints: [] }) });
    page.editor.adoptCatalog({ items: [{ id: settings.OUROBOROS_MODEL, value: settings.OUROBOROS_MODEL, provider_id: 'openrouter',
        processing_transport: ['standard', 'fast', 'economy', 'ultrafast'] }] });
    assert.equal(page.field().hidden, false);
    assert.match(page.note(), /OpenRouter may apply a workspace default/);
    assert.deepEqual(page.options().map(([tier]) => tier), ['', 'standard']);
    page.editor.load({ ...settings, OUROBOROS_PROCESSING_PREFERENCE: 'standard' }, contract);
    assert.equal(page.field().hidden, true, 'inheriting explicit Standard already requests the same mode');
});

test('a role\'s page offers the tiers its own route lists for its own account; another account of the same model is another word', async (t) => {
    const page = await rolePage(t, 'main', MAIN_ON_CODEX);
    assert.deepEqual(page.options(), [['', 'Use global setting'], ['standard', 'Standard'], ['fast', 'Fast'], ['ultrafast', 'Ultrafast · not confirmed']],
        'pinned to the account that does not list Ultrafast: never confirmed on the other account\'s word (its transport carries it, so Main may try it); Flex is beyond the transport');
    assert.equal(page.note(), '', 'the ordinary service needs no note');
    assert.deepEqual(page.seen.endpoints, ['claudexor::codex=gpt-6-astra'], 'the page\'s own model is asked for once, however often it repaints');
    await page.account('home');
    assert.deepEqual(page.options().slice(1), [['standard', 'Standard'], ['fast', 'Fast'], ['ultrafast', 'Ultrafast']]);
    await page.account('');
    assert.deepEqual(page.options().at(-1), ['ultrafast', 'Ultrafast · some accounts'], 'Auto: said for a part of the accounts, with no count of them');
    await page.tier('fast');
    assert.deepEqual([page.note(), page.editor.entry().processing, page.seen.questions], ['Requests faster processing. It may cost more.', 'fast', []]);
});

test('a subscription role is told by its own account\'s read alone: a failed one, or one kept from another engine store, confirms nothing', async (t) => {
    const fetched = { value: new Error('offline') };
    const page = await rolePage(t, 'main', { ...MAIN_ON_CODEX, OUROBOROS_MODEL_ACCOUNTS: { main: 'home' } }, { fetched });
    // The general catalog lists everything for this account; the role's own read failed, so it stands in for nothing.
    page.editor.adoptCatalog({ ...CODEX_TIERS, store_id: 'store-a' });
    await page.settle();
    assert.deepEqual(page.options().slice(1), [['standard', 'Standard'], ['fast', 'Fast · not confirmed'], ['ultrafast', 'Ultrafast · not confirmed']],
        'the transport still excludes Flex and carries Ultra, but no account confirms it');
    // Its own read answers, from the same store: the account's word stands.
    fetched.value = { ...CODEX_TIERS, store_id: 'store-a' };
    page.editor.adoptCatalog({ ...CODEX_TIERS, store_id: 'store-a' });
    await page.settle();
    assert.deepEqual(page.options().slice(1), [['standard', 'Standard'], ['fast', 'Fast'], ['ultrafast', 'Ultrafast']]);
    // The engine is another store now, and the role's read still answers from the old one: yesterday's account is not today's.
    page.editor.adoptCatalog({ ...CODEX_TIERS, store_id: 'store-b' });
    await page.settle();
    assert.deepEqual(page.options().slice(1), [['standard', 'Standard'], ['fast', 'Fast · not confirmed'], ['ultrafast', 'Ultrafast · not confirmed']]);
    fetched.value = { ...CODEX_TIERS, store_id: 'store-b' };
    page.editor.adoptCatalog({ ...CODEX_TIERS, store_id: 'store-b' });
    await page.settle();
    assert.deepEqual(page.options().at(-1), ['ultrafast', 'Ultrafast'], 'read again from the store that serves now');
});

test('Main\'s draft takes Ultrafast only after its question, for the route it has; a cancel leaves the draft and a changed account asks again at Save', async (t) => {
    const page = await rolePage(t, 'main', { ...MAIN_ON_CODEX, OUROBOROS_MODEL_ACCOUNTS: { main: 'home' } }, { answer: { value: false } });
    const read = page.editor.entry();
    await page.tier('ultrafast');
    const { doc: _doc, ...facts } = page.seen.questions[0];
    assert.deepEqual(facts, { model: 'gpt-6-astra', source: 'Codex · Pinned account home', pinned: true, priceSaid: false,
        explain: 'Requests the fastest processing this source offers. It may cost more.' });
    assert.deepEqual([page.editor.entry().processing, page.select().value, page.seen.changes], ['', '', 0], 'a cancel leaves the draft and its select as they were');
    page.answer.value = true;
    await page.tier('ultrafast');
    assert.deepEqual([page.seen.questions.length, page.editor.entry().processing, page.editor.entry().errors, page.editor.ultrafastUnasked(read)],
        [2, 'ultrafast', [], false], 'asked at the choice: its Save asks nothing more');
    // The account changes after the question: the yes was for another binding, so Save asks for this one.
    await page.account('work');
    assert.equal(page.editor.ultrafastUnasked(read), true);
    page.answer.value = false;
    assert.equal(await page.editor.confirmUltrafast(), false);
    assert.match(page.seen.questions.at(-1).explain, /Not confirmed for gpt-6-astra here: a provider may serve it as Standard\.$/);
    assert.equal(page.editor.ultrafastUnasked(read), true, 'a no confirms nothing');
    page.answer.value = true;
    assert.deepEqual([await page.editor.confirmUltrafast(), page.editor.ultrafastUnasked(read)], [true, false]);
    // The account moves while the question stands, back to one confirmed earlier: the answer is for the binding it
    // was asked for, which the row no longer has, so it confirms nothing, whatever was confirmed before.
    await page.account('home');
    assert.equal(await page.editor.confirmUltrafast(), true);
    page.answer.value = () => { void page.put('[data-model-role-account]', 'work'); return true; };
    assert.equal(await page.editor.confirmUltrafast(), false, 'asked for home, answered when the row is on work');
    await page.settle();
    page.answer.value = () => { void page.put('[data-model-role-account]', 'home'); return true; };
    assert.equal(await page.editor.confirmUltrafast(), false, 'asked for work: an earlier yes for home does not stand in');
    await page.settle();
    page.answer.value = async () => {
        await page.account('work');
        await page.account('home');
        return true;
    };
    assert.equal(await page.editor.confirmUltrafast(), false, 'a return to the same pin cannot revive a dialog left open across edits');
    assert.equal(page.editor.ultrafastUnasked(read), true);
    // A stored Ultrafast left as it was read never asks; it is offered whatever the route says, so it can be left.
    page.editor.load({ ...MAIN_ON_CODEX, OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: 'ultrafast' } }, contract);
    await page.settle();
    const stored = page.editor.entry();
    assert.deepEqual([stored.processing, page.editor.ultrafastUnasked(stored), page.editor.ultrafastUnasked(null)], ['ultrafast', false, true]);
    assert.deepEqual(page.options().at(-1), ['ultrafast', 'Ultrafast · not confirmed']);
});

test('Main confirmation retains the general fingerprint when the scoped catalog omits it', async (t) => {
    const page = await rolePage(t, 'main', MAIN_ON_CODEX);
    const read = page.editor.entry();
    const adopt = async (fingerprint) => {
        const catalog = structuredClone(CODEX_TIERS);
        catalog.store_id = 'same-store';
        catalog.account_catalogs[0].accounts.find((entry) => entry.credentialProfileId === 'work').catalog.accountFingerprint = fingerprint;
        page.editor.adoptCatalog(catalog);
        await page.settle();
    };
    await adopt('first-account');
    await page.tier('ultrafast');
    assert.equal(page.editor.entry().processing, 'ultrafast');
    assert.equal(page.editor.ultrafastUnasked(read), false);
    await adopt('replaced-account');
    assert.equal(page.editor.ultrafastUnasked(read), true);
});

test('after a refused pinned Ultrafast the page asks again, and the yes stands on the store and fingerprint its reread names', async (t) => {
    const read = (store, login) => {
        const catalog = structuredClone(CODEX_TIERS);
        catalog.store_id = store;
        for (const account of catalog.account_catalogs[0].accounts) account.catalog.accountFingerprint = `${login}-${account.credentialProfileId}`;
        return catalog;
    };
    const fetched = { value: read('store-a', 'first') };
    const page = await rolePage(t, 'main', { ...MAIN_ON_CODEX, OUROBOROS_MODEL_ACCOUNTS: { main: 'home' } }, { fetched });
    const asRead = page.editor.entry();
    page.editor.adoptCatalog(read('store-a', 'first'));
    await page.settle();
    await page.tier('ultrafast');
    const refused = page.editor.entry();
    assert.deepEqual(JSON.parse(refused.tierBinding).slice(2), ['home', 'store-a', 'first-home']);
    // Its Save answered 409 for the pinned account: the page retires what it held of that scope and reads again
    // (mountRole's save). The account signed in anew under another engine store meanwhile.
    fetched.value = read('store-b', 'second');
    page.editor.invalidateAccountCatalog(refused);
    assert.equal(page.options().at(-1)[1], 'Ultrafast · not confirmed', 'nothing held of the scope confirms it meanwhile');
    page.editor.adoptCatalog(read('store-b', 'second'));
    await page.settle();
    assert.equal(page.editor.ultrafastUnasked(asRead), true, 'the yes was for the identity the server refused');
    assert.equal(await page.editor.confirmUltrafast(), true);
    assert.deepEqual(JSON.parse(page.editor.entry().tierBinding).slice(2), ['home', 'store-b', 'second-home']);
    assert.deepEqual([page.editor.ultrafastUnasked(asRead), page.seen.questions.length], [false, 2]);
});

test('Settings confirms a key model after its priced detail answers, not after the picker moves away', async (t) => {
    const first = 'fixture/first', second = 'fixture/second';
    const reads = new Map();
    const replies = new Map();
    const page = await rolePage(t, 'main', { OUROBOROS_MODEL: first }, {
        detail: (route) => replies.get(route.model),
        readDetail: (route) => reads.get(route.model)?.promise,
    });
    const block = (model) => { let done; const promise = new Promise((resolve) => { done = resolve; }); reads.set(model, { promise }); return done; };
    const firstDone = block(first);
    const secondDone = block(second);
    page.editor.adoptCatalog({ api_sources: [{ id: 'openrouter', label: 'Fixture key', processing_transport: ['standard', 'ultrafast'] }],
        items: [first, second].map((value) => ({ id: value, value })) });
    const awaitingPrice = page.put('[data-model-role-processing]', 'ultrafast');
    await page.settle();
    assert.equal(page.seen.questions.length, 0, 'the already requested model detail is still out');
    replies.set(first, { state: 'ok', complete: true, modes: new Set(['standard', 'ultrafast']),
        endpoints: [{ mode: 'ultrafast', price: { input: 19, output: 23 } }] });
    firstDone();
    await awaitingPrice;
    assert.equal(page.seen.questions.length, 1);
    assert.match(page.seen.questions[0].explain, /Listed at \$19 in \/ \$23 out per 1M tokens/);
    assert.equal(page.editor.entry().processing, 'ultrafast');

    const field = page.host.querySelector('[data-model-role-model]');
    field.disabled = true;  // no layout in this stub: a disabled field never opens the chooser's popup, the editor still reads the input
    field.value = second;
    await field.fire('input', { target: field });
    const awaitingSecond = page.editor.confirmUltrafast(true);
    field.value = first;
    await field.fire('input', { target: field });
    secondDone();
    assert.equal(await awaitingSecond, false);
    assert.equal(page.seen.questions.length, 1, 'the late reply cannot open a question under either model');
});

test('no other role may hold Ultrafast, and the whole list of roles keeps each role\'s tiers as they are', async (t) => {
    const light = await rolePage(t, 'light', { ...MAIN_ON_CODEX, OUROBOROS_MODEL_PROCESSING_PREFERENCES: { light: 'ultrafast' } });
    assert.match(light.editor.entry().errors[0], /choose Standard, Fast, Flex or inherited processing/);
    await light.tier('fast');
    assert.deepEqual([light.editor.entry().errors, light.seen.questions], [[], []], 'another role\'s tier asks nothing');
    const editor = createModelRolesEditor({ hostId: 'none', doc: () => null });
    editor.load({ ...MAIN_ON_CODEX, OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: 'ultrafast', light: 'ultrafast' } }, contract);
    assert.match(editor.validate(), /^Light/, 'Main\'s stored Ultrafast is valid wherever the roles are edited');
    editor.destroy();
});

test('an empty Safety Supervisor\'s page names the route it runs on as the runtime resolves it: Light, else Main, with account, tier and local', async (t) => {
    const viaMain = await rolePage(t, 'safety', { ...MAIN_ON_CODEX, OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: 'fast' } });
    const placeholder = (page) => page.host.querySelector('[data-model-role-model]').getAttribute('placeholder');
    assert.equal(placeholder(viaMain), 'Empty uses the Light model: gpt-6-astra · account work · Fast');
    // Light's box over a cloud Main puts Main's model on the local model, for the Supervisor too.
    const local = await rolePage(t, 'safety', { ...MAIN_ON_CODEX, USE_LOCAL_LIGHT: true });
    assert.match(placeholder(local), /^Empty uses the Light model: gpt-6-astra · local/);
    // Light with its own model ends the chain there: Main's tier is not the Supervisor's.
    const viaLight = await rolePage(t, 'safety', { ...MAIN_ON_CODEX, OUROBOROS_MODEL_LIGHT: 'openai::gpt-light',
        OUROBOROS_MODEL_PROCESSING_PREFERENCES: { main: 'fast' } });
    assert.equal(placeholder(viaLight), 'Empty uses the Light model: gpt-light');
});
