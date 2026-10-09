import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
    ROUTE_KIND_AGENT_SESSION,
    ROUTE_KIND_API_MODEL,
    compoundSessionEffort,
    configuredApiProviders,
    compoundSessionEffortConflict,
    normalizeRouteSpec,
    serializeRouteSpec,
} from '../modules/route_editor_primitives.js';
import {
    ALLOW_EMPTY_REVIEW_POOL,
    MAX_AVAILABLE_SUBAGENTS,
    REVIEW_POOL_DEFAULT_EFFORT,
    availableSubagentsHasExplicitDraft,
    availableSubagentRowMarkup,
    availableSubagentsLoadValue,
    availableSubagentsRenderSignature,
    availableSubagentsSavePayload,
    buildAvailableSubagentsSetting,
    createAvailableSubagentsEditor,
    generatedPreviewCanReplace,
    lastReviewRunText,
    parseAvailableSubagentsSetting,
    renderSubagentsSection,
    reviewCostText,
    reviewPoolErrors,
    reviewPoolRows,
    subagentSettingsFingerprint,
    validateAvailableSubagentsSetting,
} from '../modules/subagents_settings.js';
import { reviewTwinAllowed, rowMeta, rowStatus, sessionRouteVerdict } from '../modules/subagent_status_primitives.js';
import { revealNewRow } from '../modules/ui_helpers.js';

const CONTRACT_FIXTURE = JSON.parse(fs.readFileSync(
    new URL('./fixtures/available_subagents_contract.json', import.meta.url),
    'utf8',
));

function apiRow(overrides = {}) {
    return {
        subagent_id: 'api_scout',
        recommended_use: 'Fast independent research and verification.',
        route: { kind: ROUTE_KIND_API_MODEL, target_id: 'openai/gpt-5.6-luna' },
        effort: 'high',
        ...overrides,
    };
}

function sessionRow(overrides = {}) {
    return {
        subagent_id: 'codex_builder',
        access: 'full',
        recommended_use: 'Implementation in a real workspace.',
        route: {
            kind: ROUTE_KIND_AGENT_SESSION,
            target_id: 'codex=gpt-5.6-sol-high',
            credential_profile_id: 'koshak',
        },
        ...overrides,
    };
}

function setting(items = [apiRow(), sessionRow()]) {
    return { enabled: true, items };
}

// An edited catalog is judged for its review pool too; this row keeps that
// pool non-empty on its own engine, so the twin rules under test stay alone.
function poolReviewer(overrides = {}) {
    return apiRow({
        subagent_id: 'pool_reviewer', recommended_use: 'Reviews changes.',
        route: { kind: ROUTE_KIND_API_MODEL, target_id: 'anthropic/claude-opus-5' },
        review_eligible: true, ...overrides,
    });
}

const EMPTY_POOL_ERROR = 'No row is marked Reviewer. Mark at least one row, or tick “Save without reviewers”.';

test('canonical parser accepts object or JSON and refuses unknown saved fields', () => {
    const objectResult = parseAvailableSubagentsSetting(setting());
    assert.equal(objectResult.error, '');
    assert.deepEqual(objectResult.setting, setting());

    const textResult = parseAvailableSubagentsSetting(JSON.stringify(setting([apiRow()])));
    assert.equal(textResult.setting.items[0].subagent_id, 'api_scout');

    // Legacy `name` parses and is DROPPED (retired field, 1=A) — the next
    // serialize omits the key, which is the whole migration.
    const legacyNamed = parseAvailableSubagentsSetting(setting([
        apiRow({ name: 'Fast scout' }),
    ]));
    assert.equal(legacyNamed.error, '');
    assert.equal('name' in legacyNamed.setting.items[0], false);

    const unknown = parseAvailableSubagentsSetting({ ...setting(), surprise: true });
    assert.equal(unknown.setting, null);
    assert.match(unknown.error, /unknown field: surprise/);

    const rowUnknown = parseAvailableSubagentsSetting(setting([{ ...apiRow(), role: 'scout' }]));
    assert.equal(rowUnknown.setting, null);
    assert.match(rowUnknown.error, /unknown field: role/);

    const routeUnknown = parseAvailableSubagentsSetting(setting([
        apiRow({ route: { ...apiRow().route, base_url: 'https://example.test' } }),
    ]));
    assert.equal(routeUnknown.setting, null);
    assert.match(routeUnknown.error, /route has unknown field: base_url/);

    const badKind = parseAvailableSubagentsSetting(setting([
        apiRow({ route: { kind: 'api_chat', target_id: 'openai/gpt-5.6-luna' } }),
    ]));
    assert.equal(badKind.setting, null);
    assert.match(badKind.error, /unsupported route kind/);

    const apiPin = parseAvailableSubagentsSetting(setting([
        apiRow({ route: {
            kind: 'api_model', target_id: 'openai/gpt-5.6-luna',
            credential_profile_id: 'must-not-ride-api',
        } }),
    ]));
    assert.equal(apiPin.setting, null);
    assert.match(apiPin.error, /account pin on an API route/);
});

test('the shared strict contract fixture has the same accept/reject boundary in the UI', () => {
    for (const fixture of CONTRACT_FIXTURE.valid) {
        const parsed = parseAvailableSubagentsSetting(fixture.value);
        assert.ok(parsed.setting, `${fixture.name}: ${parsed.error}`);
    }
    for (const fixture of CONTRACT_FIXTURE.invalid) {
        const parsed = parseAvailableSubagentsSetting(fixture.value);
        assert.equal(parsed.setting, null, fixture.name);
        assert.ok(parsed.error, fixture.name);
    }

    assert.equal(compoundSessionEffort('agy=gemini-3.7-flash-high-fast'), 'high');
    assert.equal(
        compoundSessionEffortConflict('cursor=gpt-5.6-sol-high-fast', 'medium'),
        'high',
    );
});

test('an unloaded or malformed view cannot replace the owner setting', () => {
    assert.deepEqual(availableSubagentsSavePayload({ loaded: false, setting: setting() }), {});
    assert.deepEqual(availableSubagentsSavePayload({
        loaded: false,
        parseError: 'invalid JSON',
        setting: setting(),
    }), {});
    assert.deepEqual(availableSubagentsSavePayload({ loaded: true, setting: setting([apiRow()]) }), {
        OUROBOROS_SUBAGENTS: setting([apiRow()]),
    });
});

test('only an omitted draft may stay out of an unrelated Settings save', () => {
    const editor = createAvailableSubagentsEditor({
        doc: null,
        win: null,
        allowUnloadedOmission: true,
    });
    editor.load(undefined, { source: 'undecided', allowOmission: true });
    assert.deepEqual(editor.validate(), []);
    assert.deepEqual(editor.collect(), {});

    editor.load({ enabled: true, items: [{ ...apiRow(), recommended_use: 7 }] }, {
        source: 'configured',
        allowOmission: false,
    });
    assert.match(editor.validate()[0], /recommended use must be a string/);
    assert.deepEqual(editor.collect(), {});

    assert.equal(availableSubagentsHasExplicitDraft({}), false);
    assert.equal(availableSubagentsHasExplicitDraft({
        OUROBOROS_SUBAGENTS: '',
        _meta: { available_subagents: { candidate: null } },
    }), false);
    assert.equal(availableSubagentsHasExplicitDraft({
        _meta: { available_subagents: { candidate: { enabled: true, items: [] } } },
    }), true);
    assert.equal(availableSubagentsHasExplicitDraft({
        OUROBOROS_SUBAGENTS: '{malformed owner bytes',
    }), true);
});

test('object and serialized settings compare as the same new-child-task intent', () => {
    assert.equal(
        subagentSettingsFingerprint(setting([apiRow()])),
        subagentSettingsFingerprint(JSON.stringify(setting([apiRow()]))),
    );
});

test('loaded saved rows remain collectible when live status is unavailable', () => {
    const store = {
        error: 'agent service offline',
        snapshot: null,
        facet: () => 'transport_error',
        subscribe: () => () => {},
        refresh: async () => {},
    };
    const editor = createAvailableSubagentsEditor({ store, doc: null, win: null });
    editor.load(setting([sessionRow()]), { source: 'configured' });
    assert.deepEqual(editor.collect(), {
        OUROBOROS_SUBAGENTS: setting([sessionRow()]),
    });
});

test('validation protects stable unique IDs, route shape, effort and the row limit', () => {
    assert.deepEqual(validateAvailableSubagentsSetting(setting()), []);
    assert.match(validateAvailableSubagentsSetting(setting([
        apiRow(), apiRow({ name: 'duplicate' }),
    ])).join(' '), /repeats stable ID/);
    assert.match(validateAvailableSubagentsSetting(setting([
        apiRow({ subagent_id: 'bad id' }),
    ])).join(' '), /stable ID/);
    assert.match(validateAvailableSubagentsSetting(setting([
        apiRow({ route: { kind: 'api_chat', target_id: 'x' } }),
    ])).join(' '), /API model or Agent session/);
    assert.match(validateAvailableSubagentsSetting(setting([
        apiRow({ effort: 'enormous' }),
    ])).join(' '), /unsupported reasoning effort/);
    assert.deepEqual(validateAvailableSubagentsSetting(setting([
        apiRow({ effort: 'ultra' }),
    ])), []);
    assert.deepEqual(validateAvailableSubagentsSetting(setting([
        apiRow({ subagent_id: 'owner.scout' }),
    ])), []);
    assert.match(validateAvailableSubagentsSetting(setting([
        sessionRow({ route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'codex=' } }),
    ])).join(' '), /harness=model/);
    // The ceiling is the catalog's (MAX_CONFIGURED_SUBAGENTS, 26 since the review
    // pool joined the catalog); the message quotes the module constant.
    const tooMany = Array.from({ length: MAX_AVAILABLE_SUBAGENTS + 1 }, (_, index) =>
        apiRow({ subagent_id: `actor_${index}` }));
    assert.match(validateAvailableSubagentsSetting(setting(tooMany)).join(' '),
        new RegExp(`at most ${MAX_AVAILABLE_SUBAGENTS}\\b`));
});

test('the catalog holds 26 rows: they load and save, and a 27th is refused on read and on save', () => {
    assert.equal(MAX_AVAILABLE_SUBAGENTS, 26);
    const rows = (length) => Array.from({ length }, (_, index) => apiRow({
        subagent_id: `actor_${index}`, route: { kind: ROUTE_KIND_API_MODEL, target_id: `openai/model-${index}` },
    }));
    const full = parseAvailableSubagentsSetting(setting(rows(26)));
    assert.equal(full.error, '');
    assert.equal(full.setting.items.length, 26);
    assert.deepEqual(validateAvailableSubagentsSetting(full.setting, { uniqueEngines: true }), []);
    assert.match(parseAvailableSubagentsSetting(setting(rows(27))).error, /more than 26 rows/);
    assert.deepEqual(validateAvailableSubagentsSetting(setting(rows(27))), ['Available subagents supports at most 26 rows.']);
    // A full catalog offers neither Add nor Duplicate a 27th row.
    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(full.setting);
    dom.toolbar.add.emit('click', '');
    dom.row(25).querySelector('[data-subagent-duplicate]').emit('click');
    assert.equal(editor.setting.items.length, 26);
    editor.destroy();
});

test('Settings loads the backend migration candidate when no new setting is materialized', () => {
    const candidate = setting([apiRow()]);
    assert.deepEqual(availableSubagentsLoadValue({
        OUROBOROS_SUBAGENTS: '',
        _meta: { available_subagents: { source: 'undecided', candidate } },
    }), candidate);
    const configured = JSON.stringify(setting([sessionRow()]));
    assert.equal(availableSubagentsLoadValue({
        OUROBOROS_SUBAGENTS: configured,
        _meta: { available_subagents: { candidate } },
    }), configured);
});

test('a clean undecided Settings draft is enriched from connected status through preview', async () => {
    const requests = [];
    const store = {
        error: '',
        snapshot: {
            harnesses: [{ id: 'codex', status: 'ok', models: [{ id: 'gpt-5.6-sol-high' }] }],
            profiles: {
                harnessAccounts: [],
                profiles: [{
                    profile: { harness_id: 'codex', profile_id: 'owner', enabled: true },
                    status: { verification: 'passed' },
                }],
            },
        },
        facet: () => 'ok',
        subscribe: () => () => {},
        refresh: async () => {},
    };
    const editor = createAvailableSubagentsEditor({
        store,
        doc: null,
        win: null,
        previewGenerated: async (request) => {
            requests.push(request);
            return {
                available_subagents: setting([sessionRow()]),
                source: 'onboarding_default',
                diagnostics: [],
            };
        },
    });
    editor.load(setting([apiRow()]), { source: 'undecided' });

    await editor.reloadStatus();

    assert.deepEqual(requests, [{ subscriptionsConnected: true }]);
    assert.equal(editor.setting.items[0].subagent_id, 'codex_builder');
    assert.equal(editor.dirty, false);
});

test('Settings status reload does not await preview and late preview obeys the whole-draft gate', async () => {
    const releases = [];
    let outerDraftClean = true;
    let applied = 0;
    const store = {
        error: '',
        snapshot: { profiles: { harnessAccounts: [], profiles: [] }, harnesses: [] },
        facet: () => 'ok',
        subscribe: () => () => {},
        refresh: async () => {},
    };
    const editor = createAvailableSubagentsEditor({
        store,
        doc: null,
        win: null,
        isOuterDraftClean: () => outerDraftClean,
        onGeneratedApply: () => { applied += 1; },
        previewGenerated: () => new Promise((resolve) => { releases.push(resolve); }),
    });
    editor.load(setting([apiRow()]), { source: 'undecided' });

    await editor.reloadStatus();
    assert.equal(releases.length, 1, 'preview starts in the background');
    assert.equal(editor.setting.items[0].subagent_id, 'api_scout');

    outerDraftClean = false;
    releases.shift()({
        available_subagents: setting([sessionRow()]),
        source: 'onboarding_default',
        diagnostics: [],
    });
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(editor.setting.items[0].subagent_id, 'api_scout');
    assert.equal(applied, 0);

    outerDraftClean = true;
    const refresh = editor.refreshGeneratedPreview({ force: true });
    assert.equal(releases.length, 1);
    releases.shift()({
        available_subagents: setting([sessionRow()]),
        source: 'onboarding_default',
        diagnostics: [],
    });
    assert.equal(await refresh, true);
    assert.equal(editor.setting.items[0].subagent_id, 'codex_builder');
    assert.equal(applied, 1);
});

test('a late generated preview cannot replace a newly loaded configured document', async () => {
    let releasePreview;
    const store = {
        error: '',
        snapshot: { profiles: { harnessAccounts: [], profiles: [] }, harnesses: [] },
        facet: () => 'ok',
        subscribe: () => () => {},
        refresh: async () => {},
    };
    const editor = createAvailableSubagentsEditor({
        store,
        doc: null,
        win: null,
        previewGenerated: () => new Promise((resolve) => { releasePreview = resolve; }),
    });
    editor.load(setting([apiRow()]), { source: 'undecided' });
    const pending = editor.reloadStatus();
    while (!releasePreview) await new Promise((resolve) => setImmediate(resolve));
    editor.load(setting([sessionRow()]), { source: 'configured' });
    releasePreview({
        available_subagents: setting([apiRow({ subagent_id: 'stale' })]),
        source: 'onboarding_default',
        diagnostics: [],
    });
    await pending;

    assert.equal(editor.setting.items[0].subagent_id, 'codex_builder');
});

test('save never fabricates or carries a name — the field is retired (1=A)', () => {
    const built = buildAvailableSubagentsSetting(setting([
        apiRow({ subagent_id: 'fast_research', name: 'Legacy Label', recommended_use: '  owner text  ' }),
    ]));
    assert.equal(built.items[0].subagent_id, 'fast_research');
    assert.equal('name' in built.items[0], false);
    assert.equal(built.items[0].recommended_use, '  owner text  ');
});

test('the editor shows a numbered row and only one owner-authored prose field', () => {
    const html = availableSubagentRowMarkup(apiRow(), {
        catalogKnown: false,
        accountsKnown: false,
        quotaKnown: false,
        statusError: '',
        snapshot: null,
    }, 2);

    assert.match(html, /class="available-subagent-heading"[^>]*>Subagent 3</);
    assert.match(html, />Description\s*<textarea\b[^>]*data-subagent-field="recommended_use"/);
    assert.equal((html.match(/<textarea\b/g) || []).length, 1);
    assert.doesNotMatch(html, /data-subagent-field="(?:id|name)"/);
    assert.doesNotMatch(html, />Stable ID<|<label>Name/);
    assert.match(html, /data-subagent-field="model"/);
    assert.match(html, /aria-labelledby="available-subagent-api_scout-heading"/);
    assert.match(html, /aria-label="Duplicate Subagent 3"/);
});

test('API and session rows render different controls; account belongs only to session', () => {
    const state = {
        catalogKnown: true,
        accountsKnown: true,
        statusError: '',
        snapshot: {
            harnesses: [{
                id: 'codex', display_name: 'Codex', status: 'ok',
                models: [{ id: 'gpt-5.6-sol-high' }],
            }],
            profiles: {
                harnessAccounts: [],
                profiles: [{
                    profile: { harness_id: 'codex', profile_id: 'koshak', enabled: true },
                    status: { verification: 'passed' },
                }],
            },
        },
    };
    const apiHtml = availableSubagentRowMarkup(apiRow(), state);
    assert.match(apiHtml, /aria-label="API model for Subagent 1"/);
    assert.doesNotMatch(apiHtml, /data-subagent-field="account"/);

    const sessionHtml = availableSubagentRowMarkup(sessionRow(), state);
    assert.match(sessionHtml, /aria-label="Agent session model for Subagent 1"/);
    assert.match(sessionHtml, /data-subagent-field="account"/);
    assert.match(sessionHtml, /Account: koshak \(pinned\)/);
});

test('saved unavailable session route and account remain selectable', () => {
    const state = {
        catalogKnown: true,
        accountsKnown: true,
        statusError: '',
        snapshot: { harnesses: [], profiles: { harnessAccounts: [], profiles: [] } },
    };
    const html = availableSubagentRowMarkup(sessionRow(), state);
    assert.match(html, /codex \(not in discovery\)/);
    assert.match(html, /gpt-5.6-sol-high \(not in discovery\)/);
    assert.match(html, /Account: koshak \(not in discovery\)/);
    assert.match(html, /currently unavailable/);
});

test('session status never calls a missing model or failed pin available', () => {
    const state = {
        catalogKnown: true,
        accountsKnown: true,
        quotaKnown: true,
        statusError: '',
        snapshot: {
            harnesses: [{
                id: 'codex', status: 'ok', enabled: true,
                models: [{ id: 'different-model' }],
            }],
            profiles: { harnessAccounts: [], profiles: [{
                profile: { harness_id: 'codex', profile_id: 'koshak', enabled: true },
                status: { verification: 'failed' },
            }] },
            quota: [],
        },
    };
    const missingModel = availableSubagentRowMarkup(sessionRow(), state);
    assert.match(missingModel, /selected model gpt-5\.6-sol-high currently unavailable/);
    assert.doesNotMatch(missingModel, /available now/);

    state.snapshot.harnesses[0].models = [{ id: 'gpt-5.6-sol-high' }];
    const failedPin = availableSubagentRowMarkup(sessionRow(), state);
    assert.match(failedPin, /pinned account koshak currently unavailable/);
    assert.doesNotMatch(failedPin, /available now/);
});

test('session status uses model-scoped quota and keeps missing quota as not proven', () => {
    const state = {
        catalogKnown: true,
        accountsKnown: true,
        quotaKnown: false,
        statusError: '',
        snapshot: {
            harnesses: [{
                id: 'codex', status: 'ok', enabled: true,
                models: [{ id: 'gpt-5.6-sol-high' }],
            }],
            profiles: { harnessAccounts: [], profiles: [{
                profile: { harness_id: 'codex', profile_id: 'koshak', enabled: true },
                status: { verification: 'passed' },
            }] },
            quota: [],
        },
    };
    assert.match(availableSubagentRowMarkup(sessionRow(), state), /quota not checked/);
    assert.doesNotMatch(availableSubagentRowMarkup(sessionRow(), state), /available now/);

    state.quotaKnown = true;
    state.snapshot.quota = [{
        subject: { harness: 'codex', subject_id: 'koshak' },
        freshness: 'fresh',
        constraints: [{
            applies_to_models: ['gpt-5.6-sol'], used_ratio: 1, resets_at: '2099-01-01T00:00:00Z',
        }],
    }];
    const exhausted = availableSubagentRowMarkup(sessionRow(), state);
    assert.match(exhausted, /pinned account koshak limit reached/);
    assert.doesNotMatch(exhausted, /available now/);

    state.snapshot.quota[0].constraints[0].applies_to_models = ['other-model'];
    assert.match(availableSubagentRowMarkup(sessionRow(), state), /available now/);
});

test('session render signature follows account-pool routing verdict changes', () => {
    const state = {
        loaded: true, parseError: '', setting: setting([apiRow(), sessionRow({
            route: {
                kind: ROUTE_KIND_AGENT_SESSION,
                target_id: 'codex=gpt-5.6-sol-high',
                credential_profile_id: '',
            },
        })]), baseline: 'saved',
        source: 'configured', diagnostics: [], statusError: '', catalogKnown: true,
        accountsKnown: true, quotaKnown: true, apiModels: [],
        snapshot: {
            harnesses: [{
                id: 'codex', status: 'ok', enabled: true,
                models: [{ id: 'gpt-5.6-sol-high' }],
            }],
            profiles: {
                profiles: [{
                    profile: { harness_id: 'codex', profile_id: 'p1', enabled: true },
                    status: { verification: 'passed' },
                }],
                harnessAccounts: [],
                accountPools: [{ harness_id: 'codex', next_up: { kind: 'profile', profile_id: 'p1' } }],
            },
            quota: [],
        },
    };
    const available = availableSubagentsRenderSignature(state);
    state.snapshot.profiles.accountPools[0].next_up = { kind: 'none' };
    assert.notEqual(availableSubagentsRenderSignature(state), available);
});

test('session render signature expires a cooldown without a changed payload', () => {
    const cooldownUntil = Date.parse('2030-01-01T00:00:00Z');
    const state = {
        loaded: true, parseError: '', setting: setting(), baseline: 'saved',
        source: 'configured', diagnostics: [], statusError: '', catalogKnown: true,
        accountsKnown: true, quotaKnown: true, apiModels: [],
        snapshot: {
            harnesses: [{
                id: 'codex', status: 'ok', enabled: true,
                models: [{ id: 'gpt-5.6-sol-high' }],
            }],
            profiles: { profiles: [{
                profile: { harness_id: 'codex', profile_id: 'koshak', enabled: true },
                status: { verification: 'passed' },
            }] },
            quota: [{
                subject: { harness: 'codex', subject_id: 'koshak' },
                freshness: 'fresh', constraints: [{
                    cooldown_until: '2030-01-01T00:00:00Z', applies_to_models: ['gpt-5.6-sol'],
                }],
            }],
        },
    };
    const cooling = availableSubagentsRenderSignature(state, cooldownUntil - 1);
    const healed = availableSubagentsRenderSignature(state, cooldownUntil + 1);
    assert.notEqual(healed, cooling);
});

test('last actual execution uses the one typed receipt and only its exact actor id', () => {
    const state = {
        catalogKnown: false,
        accountsKnown: false,
        quotaKnown: false,
        statusError: '',
        snapshot: {
            harnesses: [], profiles: {}, quota: [],
            subagent_last_delegation: {
                selected_subagent_id: 'codex_builder',
                route: 'codex',
                requested_model: 'gpt-5.6-sol-high',
                applied_model: 'GPT-5.6 Sol High',
                requested_profile: 'koshak',
                applied_profile: 'koshak',
                run_id: 'run-1',
                ts: new Date().toISOString(),
            },
        },
    };
    const matching = availableSubagentRowMarkup(sessionRow(), state);
    assert.match(matching, /Last actual run: codex session/);
    assert.match(matching, /GPT-5\.6 Sol High/);
    assert.match(matching, /account koshak/);

    const other = availableSubagentRowMarkup(sessionRow({ subagent_id: 'other' }), state);
    assert.doesNotMatch(other, /Last actual run:/);

    state.snapshot.subagent_last_delegation.applied_model = '';
    state.snapshot.subagent_last_delegation.applied_profile = '';
    const oldReceipt = availableSubagentRowMarkup(sessionRow(), state);
    assert.match(oldReceipt, /Last actual run: codex session · model not disclosed/);
    assert.doesNotMatch(oldReceipt, /Last actual run:[^<]*gpt-5\.6-sol-high/);
    assert.doesNotMatch(oldReceipt, /Last actual run:[^<]*account koshak/);
});

test('preview replaces only a clean generated baseline', () => {
    assert.equal(generatedPreviewCanReplace({ dirty: false, parsedSetting: setting() }), true);
    assert.equal(generatedPreviewCanReplace({ dirty: true, parsedSetting: setting() }), false);
    assert.equal(generatedPreviewCanReplace({
        dirty: false, outerDraftClean: false, parsedSetting: setting(),
    }), false);
    assert.equal(generatedPreviewCanReplace({ dirty: false, parsedSetting: null }), false);

    const editor = createAvailableSubagentsEditor({ doc: null, win: null });
    editor.load(setting([apiRow()]), { source: 'onboarding_default' });
    const result = editor.applyGeneratedPreview({
        available_subagents: setting([sessionRow()]),
        source: 'onboarding_default',
        diagnostics: [],
    });
    assert.equal(result.applied, true);
    assert.equal(editor.setting.items[0].subagent_id, 'codex_builder');
});

test('explicit owner preview becomes an unsaved draft and survives later generated previews', () => {
    const changes = [], dirty = [];
    const editor = createAvailableSubagentsEditor({ doc: null, win: null,
        onChange: (value) => changes.push(value), onDirtyChange: (value) => dirty.push(value) });
    const original = setting([apiRow()]);
    editor.load(original, { source: 'onboarding_default' });
    const recovered = setting([apiRow(), apiRow({ subagent_id: 'main-reviewer',
        route: { kind: 'api_model', target_id: 'claudexor::codex=main' } })]);
    assert.equal(editor.applyOwnerPreview({ available_subagents: recovered }).applied, true);
    assert.equal(editor.dirty, true);
    assert.equal(dirty.at(-1), true);
    assert.deepEqual(changes.at(-1), recovered);
    assert.equal(editor.applyGeneratedPreview({ available_subagents: original }).applied, false);
    assert.deepEqual(editor.setting, recovered);
    assert.equal(editor.applyOwnerPreview({ available_subagents: 'broken' }).applied, false);
    assert.deepEqual(editor.setting, recovered, 'invalid replacement does not erase the authored draft');
    editor.destroy();
});

test('dated API failures stay informational and bind to the exact execution choices', () => {
    const row = apiRow({ processing_preference: 'standard' });
    const state = { snapshot: { subagent_last_delegation: { latest_by_subagent: {
        api_scout: { selected_subagent_id: 'api_scout', route: 'api_model',
            requested_model: row.route.target_id, applied_model: '', outcome: 'failed',
            failure_code: 'quota_exhausted', ts: '2026-09-18T12:00:00Z', occurred_at: '2026-09-18T12:00:00Z',
            identity: { ...row.route, credential_profile_id: '', effort: 'high', processing_preference: 'standard' } },
    } } } };
    const meta = rowMeta(row, state, []);
    assert.equal(meta.tone, '');
    assert.match(meta.text, /Last run: API model.*failed \(quota_exhausted\).*2026-09-18/);
    assert.equal(rowMeta({ ...row, recommended_use: 'Changed description' }, state, []).text, meta.text);
    for (const changed of [
        { ...row, effort: 'low' },
        { ...row, processing_preference: 'flex' },
        { ...row, route: { ...row.route, target_id: 'another-model' } },
        { ...row, route: { ...row.route, credential_profile_id: 'another-account' } },
    ]) assert.match(rowMeta(changed, state, []).text, /Earlier settings:/);
    const oldStatus = rowStatus(row, state);
    delete state.snapshot.subagent_last_delegation;
    assert.deepEqual(rowStatus(row, state), oldStatus, 'history never changes live admission/status');
});

test('a typed preview refusal stays typed and cannot become an empty fictional draft', () => {
    const editor = createAvailableSubagentsEditor({ doc: null, win: null });
    editor.setPreviewFailure({
        message: 'preview refused',
        body: {
            code: 'subagent_preview_unavailable',
            diagnostics: { errors: [{ code: 'catalog_unread', message: 'Model catalog was not read.' }] },
        },
    });
    assert.equal(editor.loaded, false);
    assert.match(editor.parseError, /subagent_preview_unavailable: preview refused/);
    assert.match(editor.parseError, /catalog_unread: Model catalog was not read/);
    assert.deepEqual(editor.collect(), {});
});

test('shared route primitive preserves each semantic owner account spelling', () => {
    const normalizedReviewer = normalizeRouteSpec({
        kind: 'agent_session', target_id: 'codex=gpt-5.6-sol-high', profile_id: 'review-account',
    });
    assert.equal(normalizedReviewer.credential_pin, 'review-account');
    assert.deepEqual(serializeRouteSpec(normalizedReviewer, {
        apiKind: 'api_chat', credentialField: 'profile_id',
    }), {
        kind: 'agent_session',
        target_id: 'codex=gpt-5.6-sol-high',
        profile_id: 'review-account',
    });
    assert.deepEqual(serializeRouteSpec(sessionRow().route, {
        apiKind: ROUTE_KIND_API_MODEL,
        credentialField: 'credential_profile_id',
    }), sessionRow().route);
});

test('Settings section keeps global task-authority controls beside the actor list', () => {
    const html = renderSubagentsSection();
    assert.match(html, /<h3>Available subagents<\/h3>/);
    assert.match(html, /id="available-subagents-editor"/);
    assert.match(html, /id="s-allow-mutative-subagents"/);
    assert.match(html, /id="s-active-subagents"/);
    assert.match(html, /id="s-subagent-depth"/);
    assert.match(html, /id="s-subagent-worktree-root"/);
    assert.match(html, /id="s-subagent-projects-root"/);
    assert.doesNotMatch(html, /chooses one by its stable ID/);
    assert.match(html, /Rows marked Reviewer form the review pool/);
    assert.doesNotMatch(html, /review lane|triad|scope review/i);
});

test('revealNewRow scrolls the shortest distance and focuses the named field without a second scroll', () => {
    // docs/DESIGN.md "List editors": a freshly added entry is scrolled into
    // view without animation and takes the caret. Both arguments are the
    // caller's; a stub or detached node without the DOM methods is tolerated.
    const calls = [];
    const row = { scrollIntoView: (opts) => calls.push(['scroll', opts]) };
    const field = { focus: (opts) => calls.push(['focus', opts]) };
    revealNewRow(row, field);
    assert.deepEqual(calls, [
        ['scroll', { block: 'nearest' }],
        ['focus', { preventScroll: true }],
    ]);
    assert.doesNotThrow(() => revealNewRow({}, null));
    assert.doesNotThrow(() => revealNewRow(null, {}));
});

const QUIET_STATE = Object.freeze({
    snapshot: null, catalogKnown: false, accountsKnown: false, quotaKnown: false,
    dirty: false, baseline: 'saved', saveAttempted: false,
});

test('the card head carries the ordinal, the route mark, a two-word status and the actions', () => {
    // docs/DESIGN.md §6 row anatomy on the compact card: one primary thing (the
    // ordinal), the harness mark, a dot + short words for the two status axes
    // (intent · availability, full sentences in the title), actions docked right.
    const html = availableSubagentRowMarkup(sessionRow(), QUIET_STATE, 2);
    const head = html.slice(html.indexOf('available-subagent-head'), html.indexOf('available-subagent-purpose'));
    assert.match(head, /class="available-subagent-heading"[^>]*>Subagent 3</);
    assert.match(head, /available-subagent-route-identity-wrap/);
    assert.match(head, /class="settings-inline-status" data-subagent-status data-tone="neutral" title="Saved intent · Agent session · live availability not checked">Saved · Not checked</);
    assert.match(head, /data-subagent-duplicate/);
    assert.match(head, /data-subagent-remove/);
    assert.match(html, /<textarea\b[^>]*data-subagent-field="recommended_use" rows="1"/);
    assert.equal((html.match(/<textarea/g) || []).length, 1);
    // A routed row with no run evidence carries no meta band at all.
    assert.match(html, /data-subagent-meta[^>]*hidden/);
    assert.doesNotMatch(html, /data-invalid/);
    // An API model's availability is only known when a child starts: the
    // second word says that instead of repeating the route mark beside it.
    const api = availableSubagentRowMarkup(apiRow(), { ...QUIET_STATE, dirty: true }, 0);
    assert.match(api, /data-tone="neutral" title="Draft intent · OpenRouter API model · availability is checked when a child starts">Draft · Checked at start</);
});

test('a fresh row invites instead of erroring until the owner tries to save', () => {
    const fresh = {
        subagent_id: 'subagent_new', recommended_use: '',
        route: { kind: ROUTE_KIND_API_MODEL, target_id: '' },
    };
    const before = availableSubagentRowMarkup(fresh, QUIET_STATE, 3);
    assert.doesNotMatch(before, /data-invalid/);
    assert.doesNotMatch(before, /data-tone="error"/);
    assert.match(before, /data-subagent-meta[^>]*>Choose how this subagent runs: an API model or an agent session\.</);

    // A save attempt judges the rows that existed then (`_uiAttempted`) …
    const judged = availableSubagentRowMarkup({ ...fresh, _uiAttempted: true }, { ...QUIET_STATE, saveAttempted: true }, 3);
    assert.match(judged, /<article[^>]*data-invalid/);
    assert.match(judged, /data-subagent-meta data-tone="error"[^>]*>Subagent 4 needs a model or agent-session route\.</);
    // … while an entry added AFTER that attempt is an invitation again.
    const later = availableSubagentRowMarkup(fresh, { ...QUIET_STATE, saveAttempted: true }, 4);
    assert.doesNotMatch(later, /data-invalid/);
    assert.match(later, /data-subagent-meta[^>]*>Choose how this subagent runs/);
});

test('validate() stays pure and names rows the way the cards do', () => {
    const editor = createAvailableSubagentsEditor({ doc: null, win: null });
    editor.load(setting([apiRow()]), { source: 'configured' });
    assert.deepEqual(editor.validate(), []);
    // The Save button reports the attempt; the validator itself changes nothing
    // and a host-less editor (node tests, detached panel) tolerates the note.
    assert.doesNotThrow(() => editor.noteSaveAttempt());
    assert.deepEqual(editor.validate(), []);
    assert.deepEqual(editor.collect(), { OUROBOROS_SUBAGENTS: setting([apiRow()]) });

    const unrouted = validateAvailableSubagentsSetting(setting([
        apiRow({ route: { kind: ROUTE_KIND_API_MODEL, target_id: '' } }),
    ]));
    assert.deepEqual(unrouted, ['Subagent 1 needs a model or agent-session route.']);
    const errors = validateAvailableSubagentsSetting(setting([apiRow(), apiRow()]));
    assert.match(errors[0], /^Subagent 2 repeats stable ID/);
    assert.doesNotMatch(errors.join(' '), /\bRow \d/);
});

test('sessionRouteVerdict decides label, tone and sentence together', () => {
    const unchecked = sessionRouteVerdict(sessionRow(), { catalogKnown: false, accountsKnown: false });
    assert.deepEqual(unchecked, {
        label: 'Not checked', tone: 'neutral', text: 'Agent session · live availability not checked',
    });
    const gone = { catalogKnown: true, accountsKnown: true, quotaKnown: true, snapshot: { harnesses: [] } };
    const missing = sessionRouteVerdict(sessionRow(), gone);
    assert.deepEqual(missing, { label: 'Unavailable', tone: 'warn', text: 'codex · currently unavailable' });
});

test('the verdict reads a reviewer row pin, spelled profile_id, not only the roster spelling', () => {
    // Review-lane rows serialize their account pin as `profile_id`; roster rows
    // use `credential_profile_id`. Reading one spelling judged every pinned
    // reviewer row as unpinned — a worse falsehood than saying nothing.
    const state = {
        catalogKnown: true, accountsKnown: true, quotaKnown: true, statusError: '',
        snapshot: {
            harnesses: [{ id: 'codex', status: 'ok', enabled: true, models: [{ id: 'gpt-5.6-sol-high' }] }],
            profiles: { harnessAccounts: [], profiles: [{
                profile: { harness_id: 'codex', profile_id: 'koshak', enabled: true },
                status: { verification: 'failed' },
            }] },
            quota: [],
        },
    };
    const reviewerRow = { route: {
        kind: ROUTE_KIND_AGENT_SESSION, target_id: 'codex=gpt-5.6-sol-high', profile_id: 'koshak',
    } };
    const live = sessionRouteVerdict(reviewerRow, state);
    assert.equal(live.label, 'Unavailable');
    assert.match(live.text, /pinned account koshak currently unavailable/);
});

function pinnedAccountState(verification) {
    return {
        ...QUIET_STATE, catalogKnown: true, accountsKnown: true, quotaKnown: true, statusError: '',
        snapshot: {
            harnesses: [{ id: 'codex', status: 'ok', enabled: true, models: [{ id: 'gpt-5.6-sol-high' }] }],
            profiles: { harnessAccounts: [], profiles: [{
                profile: { harness_id: 'codex', profile_id: 'koshak', enabled: true }, status: { verification },
            }] },
            quota: [{ subject: { harness: 'codex', subject_id: 'koshak' }, freshness: 'fresh', constraints: [] }],
        },
    };
}
const PINNED_UNAVAILABLE = 'Saved intent · codex · pinned account koshak currently unavailable';

test('a row that will not run says why in a visible line and keeps its title', () => {
    // A phone, the Telegram mini app or a touch screen has no hover: the title alone hid the reason.
    const blocked = availableSubagentRowMarkup(sessionRow(), pinnedAccountState('failed'), 0);
    assert.match(blocked, new RegExp(`data-subagent-status data-tone="warn" title="${PINNED_UNAVAILABLE}">Saved · Unavailable<`));
    assert.match(blocked, new RegExp(
        `<div class="available-subagent-status-reason" data-subagent-status-reason>${PINNED_UNAVAILABLE}</div>`));
    // A row that runs, one not checked yet and an API row checked when a child starts carry no reason line.
    for (const html of [
        availableSubagentRowMarkup(sessionRow(), pinnedAccountState('passed'), 0),
        availableSubagentRowMarkup(sessionRow(), QUIET_STATE, 0),
        availableSubagentRowMarkup(apiRow(), pinnedAccountState('failed'), 0),
    ]) assert.match(html, /<div class="available-subagent-status-reason" data-subagent-status-reason hidden><\/div>/);
});

test('the status reason line follows a live status change in place', async () => {
    const store = {
        error: '', snapshot: pinnedAccountState('failed').snapshot,
        facet: () => 'ok', subscribe: () => () => {}, refresh: async () => {},
    };
    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ store, doc: dom.doc, win: null });
    editor.load(setting([sessionRow()]));
    await editor.reloadStatus();
    assert.equal(dom.row(0).status.textContent, 'Saved · Unavailable');
    assert.deepEqual({ ...dom.row(0).reason }, { textContent: PINNED_UNAVAILABLE, hidden: false });

    store.snapshot = pinnedAccountState('passed').snapshot;
    await editor.reloadStatus();
    assert.equal(dom.row(0).status.textContent, 'Saved · Available');
    assert.deepEqual({ ...dom.row(0).reason }, { textContent: '', hidden: true });
});

test('an unpinned verdict intersects the usable accounts with the accounts carrying the model', () => {
    // The live defect this pins: `gpt-5.4` was listed only by `gptopro6`, whose
    // login is not verified, while a sibling account passed — "some account
    // works" and "some account has this model" were both true of DIFFERENT
    // accounts and the row still read Available.
    const snapshot = (models) => ({
        harnesses: [{ id: 'codex', status: 'ok', enabled: true, models }],
        profiles: { harnessAccounts: [], profiles: [
            { profile: { harness_id: 'codex', profile_id: 'gptopro6', enabled: true }, status: { verification: '' } },
            { profile: { harness_id: 'codex', profile_id: 'koshak', enabled: true }, status: { verification: 'passed' } },
        ] },
        quota: [{ subject: { harness: 'codex', subject_id: 'koshak' }, freshness: 'fresh', constraints: [] }],
    });
    const facets = { catalogKnown: true, accountsKnown: true, quotaKnown: true, statusError: '' };
    const row = { route: { kind: ROUTE_KIND_AGENT_SESSION, target_id: 'codex=gpt-5.4', profile_id: '' } };

    const orphaned = sessionRouteVerdict(row, {
        ...facets, snapshot: snapshot([{ id: 'gpt-5.4', credential_profile_id: 'gptopro6' }]),
    });
    assert.deepEqual(orphaned, {
        label: 'No account', tone: 'warn', text: 'codex · no usable account currently carries gpt-5.4',
    });
    // The verified account carries it: the verdict is the one it always was.
    const carried = sessionRouteVerdict(row, {
        ...facets, snapshot: snapshot([{ id: 'gpt-5.4', credential_profile_id: 'koshak' }]),
    });
    assert.equal(carried.label, 'Available');
    // A legacy engine stamps no account on its catalog entries, so it proves no
    // absence: the older, weaker rule stands rather than a new accusation.
    const legacy = sessionRouteVerdict(row, { ...facets, snapshot: snapshot([{ id: 'gpt-5.4' }]) });
    assert.equal(legacy.label, carried.label);
    assert.doesNotMatch(legacy.text, /carries/);
    // No usable account at all keeps today's sentence, unqualified by a model.
    const none = snapshot([{ id: 'gpt-5.4', credential_profile_id: 'gptopro6' }]);
    none.profiles.profiles[1].status.verification = '';
    assert.equal(sessionRouteVerdict(row, { ...facets, snapshot: none }).text,
        'codex · no usable account currently');
});

test('the head dot takes the worse of the two status axes', () => {
    // docs/ARCHITECTURE.md §3: intent · availability, one dot whose tone is the
    // worse of the two — an unsaved draft is never shown as green success even
    // when its session is available now, and a saved API row stays neutral
    // because an API model is only checked when a child starts.
    const live = {
        catalogKnown: true, accountsKnown: true, quotaKnown: true, statusError: '',
        dirty: false, baseline: 'saved', saveAttempted: false,
        snapshot: {
            harnesses: [{ id: 'codex', status: 'ok', enabled: true, models: [{ id: 'gpt-5.6-sol-high' }] }],
            profiles: { harnessAccounts: [], profiles: [{
                profile: { harness_id: 'codex', profile_id: 'koshak', enabled: true },
                status: { verification: 'passed' },
            }] },
            quota: [{ subject: { harness: 'codex', subject_id: 'koshak' }, freshness: 'fresh', constraints: [] }],
        },
    };
    assert.match(availableSubagentRowMarkup(sessionRow(), live, 0),
        /data-tone="ok" title="Saved intent · codex · available now[^"]*">Saved · Available</);
    assert.match(availableSubagentRowMarkup(sessionRow(), { ...live, dirty: true }, 0),
        /data-tone="neutral" title="Draft intent · codex · available now[^"]*">Draft · Available</);
    assert.match(availableSubagentRowMarkup(sessionRow(), { ...live, baseline: 'generated' }, 0),
        /data-tone="neutral"[^>]*>Generated · Available</);
    assert.match(availableSubagentRowMarkup(apiRow(), live, 0), /data-tone="neutral"[^>]*>Saved · Checked at start</);
});

test('actor access defaults to full and round-trips an explicit lower choice', () => {
    const defaults = setting([apiRow(), sessionRow()]);
    const explicit = setting([apiRow(), sessionRow({ access: 'workspace_write' })]);
    assert.deepEqual(parseAvailableSubagentsSetting(explicit).setting, explicit);
    assert.notEqual(subagentSettingsFingerprint(explicit), subagentSettingsFingerprint(defaults));
    const full = setting([sessionRow({ access: 'full' })]);
    const parsed = parseAvailableSubagentsSetting(JSON.stringify(full));
    assert.equal(parsed.error, '');
    assert.deepEqual(buildAvailableSubagentsSetting(parsed.setting), full);
    assert.equal(subagentSettingsFingerprint(full), subagentSettingsFingerprint(setting([sessionRow()])));
    for (const access of [null, '', 'FULL', ' full ', false, 'readonly', 'inherit_native']) {
        assert.equal(parseAvailableSubagentsSetting(setting([sessionRow({ access })])).setting, null);
    }
    assert.equal(parseAvailableSubagentsSetting(setting([apiRow({ access: 'full' })])).setting, null);
    assert.match(validateAvailableSubagentsSetting(setting([apiRow({ access: 'full' })]))[0], /Agent session/);
    assert.equal(parseAvailableSubagentsSetting(setting([sessionRow({ route: {
        ...sessionRow().route, access: 'full',
    } })])).setting, null, 'access belongs to the actor, never RouteSpec');
});

test('session access uses a named native select with a readable capability explanation', () => {
    const html = availableSubagentRowMarkup(sessionRow({ access: 'full' }), QUIET_STATE);
    assert.match(html, /<select class="ui-control"[^>]*data-subagent-field="access"/);
    assert.match(html, /value="full" selected>Full system access/);
    assert.match(html, /Full system access \(default\)/);
    assert.match(html, /Full system access can reach outside the working folder/);
    assert.match(html, /The selected agent must support it/);
    assert.doesNotMatch(availableSubagentRowMarkup(apiRow(), QUIET_STATE), /data-subagent-field="access"/);
    const row = sessionRow({ effort: 'high', processing_preference: 'standard' });
    const receipt = { selected_subagent_id: row.subagent_id, route: 'codex', applied_model: 'observed',
        outcome: 'succeeded', identity: { ...row.route, access: 'full', effort: 'high', processing_preference: 'standard' } };
    const history = { ...QUIET_STATE, snapshot: { subagent_last_delegation: receipt } };
    assert.match(rowMeta(row, history, []).text, /Last run:/);
    assert.match(rowMeta({ ...row, access: 'workspace_write' }, history, []).text, /Earlier settings:/);
    const both = availableSubagentRowMarkup(row, history);
    assert.match(both, /data-subagent-field="access"/);
    assert.match(both, /data-subagent-meta data-run-history/);
});

// A small event surface for the real editor binder. Only the controls this
// test operates are parsed; rendered geometry/chooser behavior is browser QA.
function accessEditorDom() {
    const field = (name) => ({
        dataset: { subagentField: name }, attributes: {}, listeners: {},
        addEventListener(type, handler) { this.listeners[type] = handler; },
        setAttribute(name, value) { this.attributes[name] = value; },
        emit(type, value) { this.listeners[type]({ target: { value } }); },
        toggle(checked) { this.listeners.change({ target: { checked } }); },
    });
    let rows = [];
    // The toolbar controls the real binder wires beside the rows, and the
    // review-pool lines the painter fills in place.
    const toolbar = { add: field('add'), listEnabled: field('listEnabled'), allowEmpty: field('allowEmpty') };
    const pool = Object.fromEntries(['count', 'stays', 'empty', 'empty-text', 'confirm', 'note']
        .map((name) => [name, { textContent: '', hidden: false }]));
    const container = {
        scrollTop: 0,
        toolbar,
        set innerHTML(html) {
            rows = [...html.matchAll(/<article[^>]*data-subagent-row="([^"]+)"[^>]*>([\s\S]*?)<\/article>/g)].map((match) => {
                const fields = new Map([...match[2].matchAll(/data-subagent-field="([^"]+)"/g)]
                    .map((entry) => [entry[1], field(entry[1])]));
                const duplicate = field('duplicate');
                const meta = { dataset: {}, toggleAttribute() {}, textContent: '' };
                const facts = { textContent: '' };
                const notes = { textContent: '', hidden: true };
                const status = { dataset: {}, textContent: '', title: '' };
                const reason = { textContent: '', hidden: true };
                return {
                    dataset: { subagentRow: match[1] }, toggleAttribute() {}, meta, facts, notes, status, reason, html: match[2],
                    querySelector(selector) {
                        if (selector === '[data-subagent-duplicate]') return duplicate;
                        if (selector === '[data-subagent-meta]') return meta;
                        if (selector === '[data-subagent-review-facts]') return facts;
                        if (selector === '[data-subagent-review-notes]') return notes;
                        if (selector === '[data-subagent-status]') return status;
                        if (selector === '[data-subagent-status-reason]') return reason;
                        return fields.get(selector.match(/data-subagent-field="([^"]+)"/)?.[1]) || null;
                    },
                    querySelectorAll: (selector) => selector === '[data-subagent-field]' ? [...fields.values()] : [],
                };
            });
        },
        querySelector(selector) {
            if (selector === '[data-subagent-add]') return toolbar.add;
            if (selector === '[data-subagents-enabled]') return toolbar.listEnabled;
            if (selector === '[data-review-pool-allow-empty]') return toolbar.allowEmpty;
            const line = selector.match(/^\[data-review-pool-([a-z-]+)\]$/)?.[1];
            if (line) return pool[line] || null;
            const key = selector.match(/data-subagent-row="([^"]+)"/)?.[1];
            return rows.find((row) => row.dataset.subagentRow === key) || null;
        },
        querySelectorAll: (selector) => selector === '[data-subagent-row]' ? rows : [],
    };
    return {
        doc: { getElementById: () => container }, toolbar, pool,
        row: (index = 0) => rows[index],
    };
}

test('access edit saves and clones the lower choice, resets for API and restores full', () => {
    const dom = accessEditorDom();
    const changes = [];
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null, onChange: (value) => changes.push(value) });
    editor.load(setting([sessionRow(), poolReviewer()]));
    const control = (name, index = 0) => dom.row(index).querySelector(`[data-subagent-field="${name}"]`);
    control('access').emit('change', 'workspace_write');
    assert.equal(editor.dirty, true);
    assert.equal(editor.collect().OUROBOROS_SUBAGENTS.items[0].access, 'workspace_write');
    assert.equal(changes.at(-1).items[0].route.access, undefined);
    assert.match(control('access').attributes['aria-describedby'], /-access-help/);
    dom.row().querySelector('[data-subagent-duplicate]').emit('click');
    assert.equal(editor.setting.items.length, 3);
    assert.equal(editor.setting.items[1].access, 'workspace_write');
    assert.notEqual(editor.setting.items[0].subagent_id, editor.setting.items[1].subagent_id);
    control('route', 1).emit('change', 'api');
    assert.equal(editor.setting.items[1].access, undefined);
    assert.equal(control('access', 1), null);
    control('model', 1).emit('input', 'openai/gpt-5.6-luna');
    assert.deepEqual(editor.validate(), []);
    control('access').emit('change', 'full');
    assert.equal(editor.collect().OUROBOROS_SUBAGENTS.items[0].access, 'full');
    editor.destroy();
});

test('Duplicate is born a judged draft that names its twin until one engine field changes', () => {
    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(setting([sessionRow({ subagent_id: 'fast-scout', effort: 'high' }), poolReviewer()]));
    assert.deepEqual(editor.validate(), []);
    const control = (name, index) => dom.row(index).querySelector(`[data-subagent-field="${name}"]`);

    dom.row().querySelector('[data-subagent-duplicate]').emit('click');
    // The hidden key is neutral: an inherited `<source>_copy` label would rot with the route.
    assert.match(editor.setting.items[1].subagent_id, /^subagent_[a-z0-9]+$/);
    // The card names its twin BEFORE any Save click; the source row stays clean.
    assert.match(dom.row(1).meta.textContent, /^Subagent 2 runs the same engine as Subagent 1 — change its model/);
    assert.doesNotMatch(dom.row(0).meta.textContent, /same engine/);
    assert.deepEqual(editor.validate().filter((text) => /same engine/.test(text)).length, 1);

    // The description is not part of the engine; one engine field is.
    control('recommended_use', 1).emit('input', 'Other words, same engine.');
    assert.match(editor.validate()[0], /same engine as Subagent 1/);
    control('effort', 1).emit('change', 'low');
    assert.deepEqual(editor.validate(), []);
    assert.doesNotMatch(dom.row(1).meta.textContent, /same engine/);
    editor.destroy();
});

test('twins saved earlier are hinted, never Save-blocking, until the roster is edited', () => {
    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(setting([apiRow({ subagent_id: 'one' }), apiRow({ subagent_id: 'two' }), sessionRow(), poolReviewer()]));
    // Untouched: an unrelated Settings save must go through, so nothing blocks...
    assert.deepEqual(editor.validate(), []);
    editor.noteSaveAttempt();
    assert.deepEqual(editor.validate(), []);
    // ...but the later twin says what it is, in a neutral tone.
    assert.match(dom.row(1).meta.textContent, /^Runs the same engine as Subagent 1 — change one of them/);
    assert.equal(dom.row(1).meta.dataset.tone, undefined);
    assert.doesNotMatch(dom.row(0).meta.textContent, /same engine/);
    // Any roster edit - here another row's words - makes the save judge the whole roster.
    dom.row(2).querySelector('[data-subagent-field="recommended_use"]').emit('input', 'New words.');
    assert.deepEqual(editor.validate(), ['Subagent 2 runs the same engine as Subagent 1 — change its model, effort, access, account or processing, mark both as Reviewer for a repeated review, or remove it.']);
    editor.noteSaveAttempt();
    assert.equal(dom.row(1).meta.dataset.tone, 'error');
    editor.destroy();
});

test('the row switch is not an engine facet: switching a twin off keeps the twin, so that edit is judged', () => {
    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(setting([apiRow({ subagent_id: 'one' }), apiRow({ subagent_id: 'two' }), poolReviewer()]));
    assert.deepEqual(editor.validate(), [], 'twins saved earlier never block an untouched roster');
    dom.row(1).querySelector('[data-subagent-field="enabled"]').toggle(false);
    assert.match(editor.validate()[0], /^Subagent 2 runs the same engine as Subagent 1/);
    // A switched-off row with its own engine is an ordinary seat, and an off twin is hinted like any twin.
    dom.row(1).querySelector('[data-subagent-field="effort"]').emit('change', 'low');
    assert.deepEqual(editor.validate(), []);
    const parked = { setting: setting([apiRow({ subagent_id: 'one' }), apiRow({ subagent_id: 'two', enabled: false })]) };
    assert.match(rowMeta(parked.setting.items[1], { ...QUIET_STATE, ...parked }, []).text, /^Runs the same engine as Subagent 1/);
    editor.destroy();
});

test('engine uniqueness is a SAVE rule: a roster saved with twins still loads, and empty drafts are not twins', () => {
    const twins = setting([apiRow({ subagent_id: 'one' }), apiRow({ subagent_id: 'two', recommended_use: 'x' })]);
    const parsed = parseAvailableSubagentsSetting(twins);
    assert.equal(parsed.error, '', 'an existing install never turns invalid on read');
    assert.deepEqual(validateAvailableSubagentsSetting(parsed.setting), []);
    assert.deepEqual(validateAvailableSubagentsSetting(parsed.setting, { uniqueEngines: true }),
        ['Subagent 2 runs the same engine as Subagent 1 — change its model, effort, access, account or processing, mark both as Reviewer for a repeated review, or remove it.']);
    // The engine is judged under the processing it inherits: an unset row IS a fast row under a global fast.
    const inherits = setting([apiRow({ subagent_id: 'one' }), apiRow({ subagent_id: 'two', processing_preference: 'fast' })]);
    assert.deepEqual(validateAvailableSubagentsSetting(inherits, { uniqueEngines: true }), []);
    assert.match(validateAvailableSubagentsSetting(inherits, { uniqueEngines: true, processingPreference: 'fast' })[0], /same engine as Subagent 1/);
    // Two freshly added rows have no engine yet: each asks for a route, neither is called a twin.
    const blank = { recommended_use: '', route: { kind: ROUTE_KIND_API_MODEL, target_id: '' } };
    const drafts = validateAvailableSubagentsSetting(
        setting([{ ...blank, subagent_id: 'a' }, { ...blank, subagent_id: 'b' }]), { uniqueEngines: true });
    assert.equal(drafts.length, 2);
    assert.ok(drafts.every((text) => /needs a model or agent-session route/.test(text)));
});

// ---------------------------------------------------------------------------
// The source is CHOSEN, never spelled (docs/DESIGN.md §7): the roster card
// offers the same grouped picker the review lanes and Models do, scoped to the
// providers this install actually has a credential for.
// ---------------------------------------------------------------------------

test('the roster picker offers only credentialed providers and keeps a saved keyless one', () => {
    const state = {
        ...QUIET_STATE,
        providers: configuredApiProviders({ OPENROUTER_API_KEY: 'k', OPENAI_API_KEY: '***set***' }),
        providerProfiles: { openai: { label: 'OpenAI' } },
    };
    const html = availableSubagentRowMarkup(apiRow({ route: { kind: ROUTE_KIND_API_MODEL, target_id: 'openai::gpt-x' } }), state, 0);
    assert.match(html, /<optgroup label="API keys">/);
    assert.match(html, /<option value="api:openai" selected>OpenAI<\/option>/);
    assert.match(html, /<option value="api:openrouter">OpenRouter<\/option>/);
    assert.match(html, /<option value="" disabled>Add a key in Accounts for more<\/option>/);
    assert.doesNotMatch(html, /value="api:anthropic"/, 'a provider with no key is not offered');
    // The model field holds the model ALONE; the editor composes the prefix.
    assert.match(html, /data-subagent-field="model"[^>]*value="gpt-x"/);
    // The chip names the provider, and the exact stored id rides the meta line.
    assert.match(html, />API · OpenAI<\/span>/);
    assert.match(html, /data-subagent-meta[^>]*>stored as openai::gpt-x</);
    assert.match(html, /title="Saved intent · OpenAI API model · availability is checked when a child starts"/);

    // A saved provider whose key is gone stays selectable and says why.
    const keyless = availableSubagentRowMarkup(
        apiRow({ route: { kind: ROUTE_KIND_API_MODEL, target_id: 'anthropic::claude-opus-5' } }), state, 0);
    assert.match(keyless, /<option value="api:anthropic" selected>Anthropic \(no key\)<\/option>/);
    assert.match(keyless, />API · Anthropic<\/span>/);
});

test('the roster row status and meta name the source, never a bare channel', () => {
    const base = { ...QUIET_STATE, providerProfiles: { openai: { label: 'OpenAI' } } };
    const row = apiRow({ route: { kind: ROUTE_KIND_API_MODEL, target_id: 'openai::gpt-x' } });
    assert.equal(rowStatus(row, base).text,
        'Saved intent · OpenAI API model · availability is checked when a child starts');
    assert.deepEqual(rowMeta(row, base, []), { text: 'stored as openai::gpt-x', tone: '' });
    // An empty provider draft is still an invitation, not a stored id.
    assert.match(rowMeta(apiRow({ route: { kind: ROUTE_KIND_API_MODEL, target_id: 'openai::' } }), base, []).text, /Choose how this subagent runs/);
    // A subscription row answers to its source, not to an API provider.
    const subscription = apiRow({ route: { kind: ROUTE_KIND_API_MODEL, target_id: 'claudexor::codex-models=gpt' } });
    assert.match(rowStatus(subscription, base).text, /Subscription model/);
    assert.deepEqual(rowMeta(subscription, base, []),
        { text: 'stored as claudexor::codex-models=gpt', tone: '' });
    // A session already spells harness and model in its own controls.
    assert.deepEqual(rowMeta(sessionRow(), base, []), { text: '', tone: '' });
    // An error and the fresh-row invitation still outrank the disclosure.
    assert.deepEqual(rowMeta({ ...row, _uiAttempted: true }, base, ['Subagent 1 needs a model or agent-session route.']),
        { text: 'Subagent 1 needs a model or agent-session route.', tone: 'error' });
    assert.match(rowMeta(apiRow({ route: { kind: ROUTE_KIND_API_MODEL, target_id: '' } }), base, []).text,
        /^Choose how this subagent runs/);
});

test('the editor derives its provider list from the settings document it is given', () => {
    const editor = createAvailableSubagentsEditor({ doc: null, win: null });
    assert.doesNotThrow(() => editor.setSourceContext({
        settings: { OPENAI_API_KEY: '***set***' }, providerProfiles: { openai: { label: 'OpenAI' } },
    }));
    // The signature moves with the provider list, so a key typed in Accounts
    // repaints these rows instead of leaving a stale picker on screen.
    const state = { ...QUIET_STATE, setting: setting([apiRow()]), providers: [], providerProfiles: {} };
    assert.notEqual(
        availableSubagentsRenderSignature(state),
        availableSubagentsRenderSignature({ ...state, providers: configuredApiProviders({ OPENAI_API_KEY: 'k' }) }),
    );
});

// ---------------------------------------------------------------------------
// The per-row owner switch (docs/DESIGN.md "List editors"): a native checkbox
// leading the card head, saved through the section's common Save like every
// other field. Owner-disabled is its own axis — distinct from the list-level
// Enabled and from live availability — and never dims or locks the card.
// ---------------------------------------------------------------------------

test('the card head leads with a native enable checkbox that never dims the row', () => {
    const html = availableSubagentRowMarkup(sessionRow(), QUIET_STATE, 1);
    const head = html.slice(html.indexOf('available-subagent-head'), html.indexOf('available-subagent-purpose'));
    // BEFORE the title, using the shared primitive, with its own hit target.
    assert.ok(head.indexOf('data-subagent-field="enabled"') < head.indexOf('available-subagent-heading'));
    assert.match(head, /<label class="available-subagent-enable"[^>]*title="[^"]+"><input class="ui-checkbox" type="checkbox"/);
    assert.match(head, /data-subagent-field="enabled" aria-label="Subagent 2 enabled for new work" checked>/);
    // A switched-off row keeps every control editable: only the box clears.
    const off = availableSubagentRowMarkup(sessionRow({ enabled: false }), QUIET_STATE, 1);
    assert.match(off, /data-subagent-field="enabled" aria-label="Subagent 2 enabled for new work"><\/label>/);
    assert.doesNotMatch(off, /data-subagent-field="(model|effort|access|recommended_use)"[^>]*\sdisabled/);
    // The status chip still reports intent × availability only; the switch is
    // not smuggled into the live-availability sentence.
    assert.equal(rowStatus(sessionRow({ enabled: false }), QUIET_STATE).text,
        rowStatus(sessionRow(), QUIET_STATE).text);
});

test('the row switch is a draft the common Save writes, never an instant save', () => {
    const dom = accessEditorDom();
    const changes = [];
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null, onChange: (value) => changes.push(value) });
    editor.load(setting([apiRow(), sessionRow()]));
    const control = (name, index = 0) => dom.row(index).querySelector(`[data-subagent-field="${name}"]`);
    assert.equal(editor.dirty, false);

    control('enabled').toggle(false);
    assert.equal(editor.dirty, true);
    // Held in the draft only: `collect()` is the one writer, and the payload
    // carries `false` explicitly while the untouched sibling stays omitted.
    const payload = editor.collect().OUROBOROS_SUBAGENTS;
    assert.equal(payload.items[0].enabled, false);
    assert.equal('enabled' in payload.items[1], false);
    assert.equal(changes.at(-1).items[0].enabled, false);

    // A pending unrelated edit survives the same draft; both ride ONE save.
    control('recommended_use', 1).emit('input', 'Pending prose the owner is still writing.');
    assert.deepEqual(editor.collect().OUROBOROS_SUBAGENTS.items.map((row) => row.enabled),
        [false, undefined]);
    assert.equal(editor.collect().OUROBOROS_SUBAGENTS.items[1].recommended_use,
        'Pending prose the owner is still writing.');

    // off -> save -> reload -> on: the saved bytes round-trip, and switching the
    // row back on returns the roster to its original canonical form.
    const saved = editor.collect().OUROBOROS_SUBAGENTS;
    editor.load(saved);
    assert.equal(editor.dirty, false);
    assert.equal(editor.setting.items[0].enabled, false);
    dom.row(0).querySelector('[data-subagent-field="enabled"]').toggle(true);
    assert.equal('enabled' in editor.collect().OUROBOROS_SUBAGENTS.items[0], false);
    editor.destroy();
});

test('duplicate carries the row switch and the two enabled axes stay independent', () => {
    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(setting([apiRow({ enabled: false })]));
    assert.equal(editor.setting.enabled, true, 'a disabled ROW does not disable the list');

    dom.row(0).querySelector('[data-subagent-duplicate]').emit('click');
    assert.equal(editor.setting.items.length, 2);
    assert.equal(editor.setting.items[1].enabled, false, 'the copy preserves the owner choice');
    assert.notEqual(editor.setting.items[0].subagent_id, editor.setting.items[1].subagent_id);
    // A freshly ADDED row is enabled: the invitation is never born switched off.
    dom.toolbar.add.emit('click', '');
    assert.equal(editor.setting.items.length, 3);
    assert.equal('enabled' in editor.setting.items[2], false);

    // The list-level switch writes only itself, leaving every row switch alone.
    dom.toolbar.listEnabled.toggle(false);
    const payload = editor.collect().OUROBOROS_SUBAGENTS;
    assert.equal(payload.enabled, false);
    assert.deepEqual(payload.items.map((row) => row.enabled), [false, false, undefined]);
    dom.toolbar.listEnabled.toggle(true);
    assert.equal(editor.collect().OUROBOROS_SUBAGENTS.enabled, true);
    assert.equal(editor.collect().OUROBOROS_SUBAGENTS.items[0].enabled, false);
    editor.destroy();
});

// ---------------------------------------------------------------------------
// The review pool: a row marked Reviewer reviews. The mark, its price or
// non-money equivalent, the repeat caption, the last run and the empty-pool
// refusal are visible text on the card and the toolbar, never only a tooltip.
// ---------------------------------------------------------------------------

test('review fields parse strictly and only their non-default values are written', () => {
    const marked = apiRow({ review_eligible: true, delivery: 'packet', minted_from: 'review_lane' });
    assert.deepEqual(parseAvailableSubagentsSetting(setting([marked])).setting, setting([marked]));
    const defaults = parseAvailableSubagentsSetting(setting([apiRow({ review_eligible: false, delivery: 'native' })]));
    assert.deepEqual(defaults.setting, setting([apiRow()]), 'an untouched row keeps its exact bytes');
    for (const [row, error] of [
        [apiRow({ review_eligible: 'yes' }), /^row 1 reviewer mark must be true or false$/],
        [apiRow({ review_eligible: true, delivery: 'tools' }), /^row 1 delivery must be native or packet$/],
        [sessionRow({ review_eligible: true, delivery: 'packet' }), /^row 1 delivery requires an API model$/],
        [apiRow({ minted_from: 'elsewhere' }), /^row 1 has an unknown origin$/],
        [apiRow({ review_eligible: true, coupling_focus: true }), /^row 1 has unknown field: coupling_focus$/],
    ]) {
        const parsed = parseAvailableSubagentsSetting(setting([row]));
        assert.equal(parsed.setting, null);
        assert.match(parsed.error, error);
    }
});

test('every card carries the Reviewer box with its price beside it; a marked API row adds its delivery', () => {
    const plain = availableSubagentRowMarkup(apiRow(), QUIET_STATE, 0);
    assert.match(plain, /<label class="available-subagent-reviewer"><input class="ui-checkbox" type="checkbox" data-subagent-field="review_eligible" aria-label="Subagent 1 reviews"> Reviewer<\/label>/);
    assert.match(plain, /data-subagent-review-facts>price appears after saving</);
    assert.doesNotMatch(plain, /data-subagent-field="delivery"/);

    const marked = availableSubagentRowMarkup(apiRow({ review_eligible: true }), QUIET_STATE, 0);
    assert.match(marked, /aria-label="Subagent 1 reviews" checked> Reviewer/);
    assert.match(marked, /data-subagent-review-facts>In the review pool · price appears after saving</);
    assert.match(marked, /<select class="ui-control" data-subagent-field="delivery" aria-label="Review delivery for Subagent 1">/);
    assert.match(marked, /<option value="native" selected>Reads the work itself<\/option>/);
    assert.match(availableSubagentRowMarkup(apiRow({ review_eligible: true, delivery: 'packet' }), QUIET_STATE, 0),
        /<option value="packet" selected>Packet — for models without tool calling<\/option>/);

    // Effort left to the route reviews at the pool default, and the card says so.
    assert.match(availableSubagentRowMarkup(apiRow({ review_eligible: true, effort: '' }), QUIET_STATE, 0),
        new RegExp(`data-subagent-review-facts>In the review pool · price appears after saving · reviews at ${REVIEW_POOL_DEFAULT_EFFORT} effort<`));
    assert.equal(REVIEW_POOL_DEFAULT_EFFORT, 'high');
    // A session is spoken as a seat and time, and has no packet delivery.
    const session = availableSubagentRowMarkup(sessionRow({ review_eligible: true }), QUIET_STATE, 0);
    assert.match(session, /data-subagent-review-facts>In the review pool · uses a session seat and time · reviews at high effort</);
    assert.doesNotMatch(session, /data-subagent-field="delivery"/);
    assert.match(availableSubagentRowMarkup(apiRow({ review_eligible: true, enabled: false }), QUIET_STATE, 0),
        /data-subagent-review-facts>Switched off, so not in the review pool · /);
});

test('a price is the route tariff or plainly unknown, and a seat is spoken as time, never as $0', () => {
    const api = apiRow();
    assert.equal(reviewCostText(sessionRow()), 'uses a session seat and time');
    assert.equal(reviewCostText(apiRow({ route: { kind: ROUTE_KIND_API_MODEL, target_id: 'claudexor::codex-models=gpt' } })),
        'uses a session seat and time');
    assert.equal(reviewCostText(api), 'price appears after saving');
    assert.equal(reviewCostText(api, { usd_per_review: null, basis: 'unknown' }), 'cost unknown');
    assert.equal(reviewCostText(api, { usd_per_review: 0.5, basis: 'unknown' }), 'cost unknown');
    assert.equal(reviewCostText(api, { usd_per_review: null, basis: 'route_tariff' }), 'cost unknown');
    assert.equal(reviewCostText(api, { usd_per_review: 0, basis: 'route_tariff' }), 'no API cost per review');
    // One full call of the row, never a cap on a whole review: a reading reviewer makes several.
    assert.equal(reviewCostText(api, { usd_per_review: 1.234, basis: 'route_tariff' }),
        '≈$1.23 per full call (route tariff); a reading reviewer makes several');
    assert.equal(reviewCostText(apiRow({ delivery: 'packet' }), { usd_per_review: 0.0042, basis: 'route_tariff' }),
        '≈$0.0042 per full call (route tariff)');
});

test('a minted row names its origin, and a copy keeps the mark but never the origin', () => {
    assert.match(availableSubagentRowMarkup(apiRow({ review_eligible: true, minted_from: 'review_lane' }), QUIET_STATE, 0),
        /<span class="available-subagent-minted" data-subagent-minted>From a former review lane<\/span>/);
    assert.match(availableSubagentRowMarkup(apiRow({ review_eligible: true, minted_from: 'factory_default' }), QUIET_STATE, 0),
        /data-subagent-minted>Factory reviewer</);
    assert.doesNotMatch(availableSubagentRowMarkup(apiRow({ review_eligible: true }), QUIET_STATE, 0), /data-subagent-minted/);

    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(setting([apiRow({ review_eligible: true, minted_from: 'review_lane' })]));
    dom.row(0).querySelector('[data-subagent-duplicate]').emit('click');
    assert.equal(editor.setting.items[1].minted_from, undefined);
    assert.equal(editor.setting.items[1].review_eligible, true);
    // Two marked rows on one engine are a repeated review, not a copy slip.
    assert.deepEqual(editor.validate(), []);
    assert.equal(dom.row(1).notes.textContent, 'Repeat of Subagent 1: another independent run of the same model, not a different reviewer.');
    assert.equal(dom.row(1).notes.hidden, false);
    assert.equal(dom.row(0).notes.hidden, true);
    editor.destroy();
});

test('two marked rows on one engine are a repeat; a twin only one of which reviews is still a copy slip', () => {
    const twins = (first, second) => validateAvailableSubagentsSetting(setting([
        apiRow({ subagent_id: 'one', ...first }), apiRow({ subagent_id: 'two', ...second }),
    ]), { uniqueEngines: true });
    assert.deepEqual(twins({ review_eligible: true }, { review_eligible: true }), []);
    assert.deepEqual(twins({ review_eligible: true, minted_from: 'review_lane' }, {}), [],
        'a review row minted beside the owner delegation row is no slip');
    assert.match(twins({ review_eligible: true }, {})[0], /^Subagent 2 runs the same engine as Subagent 1 — .* mark both as Reviewer for a repeated review, or remove it\.$/);
    assert.match(twins({ review_eligible: true, minted_from: 'factory_default' }, { minted_from: 'review_lane' })[0], /same engine as Subagent 1/);
    assert.equal(reviewTwinAllowed({ review_eligible: true }, { review_eligible: true }), true);
    assert.equal(reviewTwinAllowed({}, {}), false);

    const repeat = setting([apiRow({ subagent_id: 'one', review_eligible: true }), apiRow({ subagent_id: 'two', review_eligible: true })]);
    const state = { ...QUIET_STATE, setting: repeat };
    assert.doesNotMatch(rowMeta(repeat.items[1], state, []).text, /same engine/);
    assert.match(availableSubagentRowMarkup(repeat.items[1], state, 1),
        /data-subagent-review-notes data-run-history>Repeat of Subagent 1: another independent run of the same model, not a different reviewer\.</);
    const slip = setting([apiRow({ subagent_id: 'one', review_eligible: true }), apiRow({ subagent_id: 'two' })]);
    const slipState = { ...QUIET_STATE, setting: slip };
    assert.match(rowMeta(slip.items[1], slipState, []).text, /^Runs the same engine as Subagent 1/);
    assert.doesNotMatch(availableSubagentRowMarkup(slip.items[1], slipState, 1), /Repeat of/);
});

test('Last run as … names what actually ran and the review record it wrote', () => {
    assert.equal(lastReviewRunText(null), '');
    assert.equal(lastReviewRunText({}), '');
    assert.equal(lastReviewRunText({ observed_model: 'openai/gpt-5.6-sol', record_id: 'rev_42', status: 'ok' }),
        'Last run as openai/gpt-5.6-sol (record rev_42)');
    assert.equal(lastReviewRunText({ effective: { route: 'agent_session:codex', model: 'gpt-5.6-sol-high', credential_profile_id: 'koshak' } }),
        'Last run as codex session · gpt-5.6-sol-high · account koshak');
});

test('review-pool facts price saved rows by their loaded route; an edited route waits for its save', () => {
    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(setting([apiRow({ review_eligible: true }), sessionRow()]));
    editor.setReviewPool({
        row_costs: {
            api_scout: { usd_per_review: 0.42, basis: 'route_tariff' },
            codex_builder: { usd_per_review: null, basis: 'subscription_seat' },
        },
        pool: [{ subagent_id: 'api_scout', last_execution: { observed_model: 'openai/gpt-5.6-luna', record_id: 'rev_7' } }],
        last_executions: { codex_builder: { effective: { route: 'agent_session:codex', model: 'gpt-5.6-sol-high' } } },
    });
    assert.equal(dom.row(0).facts.textContent,
        'In the review pool · ≈$0.42 per full call (route tariff); a reading reviewer makes several');
    assert.equal(dom.row(0).notes.textContent, 'Last run as openai/gpt-5.6-luna (record rev_7)');
    assert.equal(dom.row(1).facts.textContent, 'uses a session seat and time');
    assert.equal(dom.row(1).notes.textContent, 'Last run as codex session · gpt-5.6-sol-high');
    assert.equal(dom.pool.count.textContent, 'Reviewers: 1');

    // The saved price belongs to the saved route, not to the edited one.
    dom.row(0).querySelector('[data-subagent-field="model"]').emit('input', 'openai/gpt-5.6-sol');
    assert.equal(dom.row(0).facts.textContent, 'In the review pool · price appears after saving');

    // A failed read, or a pool error, prices nothing and says why.
    editor.load(setting([apiRow({ review_eligible: true })]));
    editor.setReviewPool({ load_error: 'Review pool facts could not be read: HTTP 503' });
    assert.equal(dom.row(0).facts.textContent, 'In the review pool · cost unknown');
    assert.equal(dom.pool.note.textContent, 'Review pool facts could not be read: HTTP 503');
    assert.equal(dom.pool.note.hidden, false);
    editor.setReviewPool({ config_error: 'row 2 route is malformed', pool: [], row_costs: {} });
    assert.equal(dom.pool.note.textContent, 'The saved review pool has an error: row 2 route is malformed');
    assert.equal(dom.row(0).facts.textContent, 'In the review pool · cost unknown');
    editor.setReviewPool({ row_costs: {}, pool: [], migration: { snapshot: 'state/review_lanes_snapshot.json', reported: false } });
    assert.equal(dom.pool.note.textContent,
        'Rows marked “From a former review lane” were converted from your review lanes; snapshot state/review_lanes_snapshot.json keeps their previous value.');
    editor.destroy();
});

test('the pool note says what the migration did to the owner\'s review settings and when no pool row has credentials', () => {
    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(setting([apiRow({ review_eligible: true })]));
    const snapshot = 'state/review_migrations/20261008T101010Z-slots-to-pool.json';
    const receipt = (fields) => ({ snapshot, reported: true, trigger: 'lanes_key', error: '', source: 'document', ...fields });
    const pool = [{ subagent_id: 'api_scout', route: { target_id: 'openai/gpt-5.6-sol' }, cost: { basis: 'per_review' } }];

    editor.setReviewPool({ row_costs: {}, pool, migration: receipt({ outcome: 'converted' }) });
    assert.equal(dom.pool.note.textContent,
        `Rows marked “From a former review lane” were converted from your review lanes; snapshot ${snapshot} keeps their previous value.`);
    editor.setReviewPool({ row_costs: {}, pool, migration: receipt({ outcome: 'factory', trigger: 'never_configured' }) });
    assert.equal(dom.pool.note.textContent,
        `This install had no review settings, so factory reviewers were set up (rows marked “Factory reviewer”); snapshot ${snapshot}.`);
    editor.setReviewPool({ row_costs: {}, pool, migration: receipt({ outcome: 'factory', trigger: 'never_configured', source: 'environment' }) });
    assert.match(dom.pool.note.textContent, /factory reviewers were set up .* The catalog from the environment runs instead of these rows\.$/);
    editor.setReviewPool({ row_costs: {}, pool, migration: receipt({ outcome: 'error', error: 'OUROBOROS_REVIEWER_SLOTS: row 2 has no model', source: 'error' }) });
    assert.equal(dom.pool.note.textContent,
        `Review migration failed: OUROBOROS_REVIEWER_SLOTS: row 2 has no model; your lanes were kept; snapshot ${snapshot}. Mark reviewers here and save to finish.`);
    editor.setReviewPool({ row_costs: {}, pool, migration: receipt({ outcome: 'error', error: 'bad lanes', snapshot: '', source: 'error' }) });
    assert.equal(dom.pool.note.textContent,
        'Review migration failed: bad lanes; your lanes were kept; no snapshot was written. Mark reviewers here and save to finish.');
    // A receipt of an earlier document the owner has since re-saved says nothing.
    editor.setReviewPool({ row_costs: {}, pool, migration: receipt({ outcome: 'converted', source: 'history' }) });
    assert.equal(dom.pool.note.hidden, true);

    // VD3-08: every pool row without credentials is the loud fact; some rows is a shorter one.
    editor.setReviewPool({ row_costs: {}, pool, migration: null, pool_without_credentials: ['api_scout'] });
    assert.equal(dom.pool.note.textContent,
        'No pool row has credentials: none of the 1 reviewer model has an API key in this install, so every review will fail until a key is added or a reviewer with one is marked.');
    const two = [...pool, { subagent_id: 'critic', route: { target_id: 'anthropic/claude-fable-5' }, cost: { basis: 'per_review' } }];
    editor.setReviewPool({ row_costs: {}, pool: two, migration: null, pool_without_credentials: ['critic'] });
    assert.equal(dom.pool.note.textContent, 'No credentials in this install for critic: those seats cannot answer.');
    // Both facts stand side by side; a pool error still comes first.
    editor.setReviewPool({ row_costs: {}, pool, migration: receipt({ outcome: 'converted' }), pool_without_credentials: ['api_scout'] });
    assert.match(dom.pool.note.textContent, /^Rows marked .* No pool row has credentials: /);
    editor.destroy();
});

test('an edited catalog with rows but no Reviewer is refused until the owner saves without reviewers', () => {
    assert.deepEqual(reviewPoolErrors(setting([apiRow()]), { judged: true }), [EMPTY_POOL_ERROR]);
    assert.deepEqual(reviewPoolErrors(setting([apiRow()]), { judged: false }), [], 'an untouched saved catalog is not judged');
    assert.deepEqual(reviewPoolErrors(setting([]), { judged: true }), [], 'no rows leaves nothing to mark');
    assert.deepEqual(reviewPoolErrors(setting([apiRow()]), { judged: true, allowEmpty: true }), []);
    // A pool of switched-off reviewers reviews nothing either: the server refuses it the same way.
    assert.deepEqual(reviewPoolErrors(setting([apiRow({ review_eligible: true, enabled: false })]), { judged: true }),
        ['Every row marked Reviewer is switched off. Switch one on, or tick “Save without reviewers”.']);
    assert.deepEqual(reviewPoolErrors(setting([
        apiRow({ review_eligible: true, enabled: false }), sessionRow({ review_eligible: true }),
    ]), { judged: true }), [], 'one reviewer switched on is a pool');
    assert.deepEqual(reviewPoolRows(setting([
        apiRow({ review_eligible: true }), sessionRow({ review_eligible: true, enabled: false }), apiRow({ subagent_id: 'plain' }),
    ])).map((row) => row.subagent_id), ['api_scout']);

    const dom = accessEditorDom();
    const editor = createAvailableSubagentsEditor({ doc: dom.doc, win: null });
    editor.load(setting([apiRow(), sessionRow()]));
    assert.deepEqual(editor.validate(), [], 'loading is not saving');
    assert.equal(dom.pool.count.textContent, 'Reviewers: 0');
    assert.equal(dom.pool.empty.hidden, false);
    assert.equal(dom.pool['empty-text'].textContent, 'No row is marked Reviewer, so reviews will not run and will report “not performed”.');
    assert.equal(dom.pool.confirm.hidden, false);
    assert.equal(dom.pool.stays.hidden, true);

    dom.row(1).querySelector('[data-subagent-field="recommended_use"]').emit('input', 'Edited.');
    assert.deepEqual(editor.validate(), [EMPTY_POOL_ERROR]);
    assert.equal(editor.allowEmptyReviewPool, false);
    assert.equal(ALLOW_EMPTY_REVIEW_POOL in editor.collect(), false);

    dom.toolbar.allowEmpty.toggle(true);
    assert.deepEqual(editor.validate(), []);
    assert.equal(editor.allowEmptyReviewPool, true);
    assert.equal(editor.collect()[ALLOW_EMPTY_REVIEW_POOL], true);

    // A marked row makes the confirmation moot: the flag never rides a non-empty pool.
    dom.row(0).querySelector('[data-subagent-field="review_eligible"]').toggle(true);
    assert.equal(editor.collect().OUROBOROS_SUBAGENTS.items[0].review_eligible, true);
    assert.equal(ALLOW_EMPTY_REVIEW_POOL in editor.collect(), false);
    assert.deepEqual(editor.validate(), []);
    assert.equal(dom.pool.count.textContent, 'Reviewers: 1');
    assert.equal(dom.pool.empty.hidden, true);
    assert.equal(dom.pool.stays.textContent, 'Off stops delegation only: review stays on for rows marked Reviewer.');
    assert.equal(dom.pool.stays.hidden, false);
    assert.ok(dom.row(0).querySelector('[data-subagent-field="delivery"]'), 'the mark repaints the card with its delivery');
    dom.row(0).querySelector('[data-subagent-field="delivery"]').emit('change', 'packet');
    assert.equal(editor.collect().OUROBOROS_SUBAGENTS.items[0].delivery, 'packet');
    dom.row(0).querySelector('[data-subagent-field="delivery"]').emit('change', 'native');
    assert.equal('delivery' in editor.collect().OUROBOROS_SUBAGENTS.items[0], false);

    // A reload forgets the confirmation: it answered one save.
    editor.load(setting([apiRow(), sessionRow()]));
    assert.equal(editor.allowEmptyReviewPool, false);

    // Switching the only reviewer off empties the pool: the same confirmation is asked.
    editor.load(setting([apiRow({ review_eligible: true }), sessionRow()]));
    assert.equal(dom.pool.confirm.hidden, true);
    dom.row(0).querySelector('[data-subagent-field="enabled"]').toggle(false);
    assert.equal(dom.pool['empty-text'].textContent,
        'Every row marked Reviewer is switched off, so reviews will not run and will report “not performed”.');
    assert.equal(dom.pool.confirm.hidden, false);
    assert.deepEqual(editor.validate(), ['Every row marked Reviewer is switched off. Switch one on, or tick “Save without reviewers”.']);
    dom.toolbar.allowEmpty.toggle(true);
    assert.equal(editor.collect()[ALLOW_EMPTY_REVIEW_POOL], true);
    editor.destroy();
});
