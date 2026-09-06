import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
    ROUTE_KIND_AGENT_SESSION,
    ROUTE_KIND_API_MODEL,
    compoundSessionEffort,
    compoundSessionEffortConflict,
    normalizeRouteSpec,
    serializeRouteSpec,
} from '../modules/route_editor_primitives.js';
import {
    MAX_AVAILABLE_SUBAGENTS,
    availableSubagentsHasExplicitDraft,
    availableSubagentRowMarkup,
    availableSubagentsLoadValue,
    availableSubagentsRenderSignature,
    availableSubagentsSavePayload,
    buildAvailableSubagentsSetting,
    createAvailableSubagentsEditor,
    generatedPreviewCanReplace,
    parseAvailableSubagentsSetting,
    renderSubagentsSection,
    subagentSettingsFingerprint,
    validateAvailableSubagentsSetting,
} from '../modules/subagents_settings.js';
import { buildReviewerSlotsSetting } from '../modules/reviewer_slots.js';
import { sessionRouteVerdict } from '../modules/subagent_status_primitives.js';
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

test('validation protects stable unique IDs, route shape, effort and ten-row limit', () => {
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
    const tooMany = Array.from({ length: MAX_AVAILABLE_SUBAGENTS + 1 }, (_, index) =>
        apiRow({ subagent_id: `actor_${index}` }));
    assert.match(validateAvailableSubagentsSetting(setting(tooMany)).join(' '), /at most 10/);
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
    assert.match(html, />Description\s*<textarea data-subagent-field="recommended_use"/);
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

test('reviewer structured bytes keep api_chat and profile_id after extraction', () => {
    const reviewer = buildReviewerSlotsSetting({
        triad: [{
            slot_id: 'triad_1',
            route: {
                kind: 'agent_session',
                target_id: 'codex=gpt-5.6-sol-high',
                profile_id: 'koshak',
            },
            effort: 'high',
        }],
        scope: [{
            slot_id: 'scope_1',
            route: { kind: 'api_chat', target_id: 'openai/gpt-5.6-sol' },
        }],
        advisory: { enabled: true, route: { kind: 'api_chat', target_id: '' }, effort: 'low' },
    });
    const parsed = JSON.parse(reviewer);
    assert.equal(parsed.triad[0].route.profile_id, 'koshak');
    assert.equal(parsed.triad[0].route.credential_profile_id, undefined);
    assert.equal(parsed.scope[0].route.kind, 'api_chat');
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
    assert.match(html, /<textarea data-subagent-field="recommended_use" rows="1"/);
    assert.equal((html.match(/<textarea/g) || []).length, 1);
    // A routed row with no run evidence carries no meta band at all.
    assert.match(html, /data-subagent-meta[^>]*hidden/);
    assert.doesNotMatch(html, /data-invalid/);
    // An API model's availability is only known when a child starts: the
    // second word says that instead of repeating the route mark beside it.
    const api = availableSubagentRowMarkup(apiRow(), { ...QUIET_STATE, dirty: true }, 0);
    assert.match(api, /data-tone="neutral" title="Draft intent · API model · availability is checked when a child starts">Draft · Checked at start</);
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
