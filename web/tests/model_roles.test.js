import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { MAIN_FOLLOW_NOTE, composeModelSource, createModelRolesEditor, inheritTargetOf, modelContextNote,
    modelSourceGroups, parseModelSource, sourceChoice, sourceFromChoice } from '../modules/model_roles.js';
import { configuredApiProviders, routeChoiceGroups } from '../modules/route_editor_primitives.js';

const { contract } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url)));

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
