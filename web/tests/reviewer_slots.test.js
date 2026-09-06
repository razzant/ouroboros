import assert from 'node:assert/strict';
import test from 'node:test';

import { createClaudexorStatusStore } from '../modules/claudexor_status_store.js';
import { serviceBannerLine } from '../modules/harness_accounts.js';
import { renderSettingsPage } from '../modules/settings_ui.js';

import {
    API_ROUTE_CHOICE,
    CATEGORIES,
    ROUTE_KIND_API,
    ROUTE_KIND_SESSION,
    advisoryRouteTransition,
    buildReviewerSlotsSetting,
    capabilityBadge,
    composeSessionTarget,
    decodeRouteChoice,
    deepReviewDeliveryNote,
    deepReviewMetaNotes,
    describeLastExecution,
    describeSubagentReference,
    encodeRouteChoice,
    harnessModelsKnown,
    mintSlotId,
    modelsGapNote,
    pinnedAccountWarning,
    profileOptionsFor,
    renderReviewerSlotsSection,
    reviewerRouteIdentityMarkup,
    routeChoiceGroups,
    sessionModelOptions,
    splitSessionTarget,
    subagentOptionsFor,
    SUBAGENT_CHOICE_PREFIX,
    advisoryReferenceTransition,
    encodeReviewerChoice,
    reviewerChoiceGroups,
} from '../modules/reviewer_slots.js';

test('reviewer routes reuse the shared visible harness identity without claiming execution', () => {
    const harnesses = {
        claude: { id: 'claude', display_name: 'Claude Code Max' },
    };
    const known = reviewerRouteIdentityMarkup({
        kind: ROUTE_KIND_SESSION,
        target_id: 'claude=claude-fable-5',
    }, harnesses, { catalogKnown: true });
    assert.match(known, /data-harness-identity="claude"/);
    assert.match(known, /Claude Code Max/);
    assert.match(known, /fill="currentColor"/);
    assert.doesNotMatch(known, /executed|status|available/i);

    const unread = reviewerRouteIdentityMarkup({
        kind: ROUTE_KIND_SESSION,
        target_id: 'claude=claude-fable-5',
    }, harnesses, { catalogKnown: false });
    assert.match(unread, /Claude Code/);
    assert.doesNotMatch(unread, /Claude Code Max/);

    const api = reviewerRouteIdentityMarkup({ kind: ROUTE_KIND_API, target_id: 'openai/gpt' });
    assert.match(api, /data-presentation-kind="channel"/);
    assert.match(api, />API</);
});


test('the category table is the one driver of the multi-row editor and matches its markup', () => {
    // ONE table replaced three `group === 'scope' ? … : …` ternaries (lookup,
    // add, remove): a new multi-row category is a table entry, and every id
    // the table names must exist in the static section it paints into.
    const markup = renderReviewerSlotsSection();
    assert.deepEqual(Object.keys(CATEGORIES), ['triad', 'scope']);
    for (const [group, cat] of Object.entries(CATEGORIES)) {
        assert.equal(cat.stateKey, group);
        assert.equal(cat.limitKey, group);
        for (const id of [cat.rowsId, cat.limitId, cat.addId]) {
            assert.match(markup, new RegExp(`id="${id}"`), `${group}: ${id} missing from the section markup`);
        }
        assert.ok(cat.idPrefix && cat.surfaceDefault && cat.empty, `${group}: incomplete table entry`);
    }
    assert.equal(CATEGORIES.scope.surfaceDefault, 'scope review effort');
    assert.equal(CATEGORIES.triad.surfaceDefault, 'review effort');
});

test('the deep self-review row rides the composed setting on the shared vocabulary (R6/R7)', () => {
    const base = {
        triad: [{ slot_id: 't1', route: { kind: ROUTE_KIND_API, target_id: 'openai/x' }, effort: '' }],
        scope: [{ slot_id: 's1', route: { kind: ROUTE_KIND_API, target_id: 'openai/y' }, effort: '' }],
        advisory: { enabled: true, route: { kind: ROUTE_KIND_API, target_id: '' }, effort: 'low' },
    };
    // Legacy callers that know nothing about the singleton keep their bytes:
    // no `deep_review` key is ever invented.
    assert.equal('deep_review' in JSON.parse(buildReviewerSlotsSetting(base)), false);

    // A direct api row: the packed review. '' effort is OMITTED (the
    // Behavior-tab deep effort keeps deciding); the synthesized label and the
    // fixed identity never reach the saved bytes.
    const api = JSON.parse(buildReviewerSlotsSetting({
        ...base,
        deepReview: { route: { kind: ROUTE_KIND_API, target_id: 'openai/gpt-5.6-sol-pro' }, effort: '',
                      subagent_id: '', synthesizedFrom: 'OUROBOROS_MODEL_DEEP_SELF_REVIEW' },
    }));
    assert.deepEqual(api.deep_review, { route: { kind: 'api_chat', target_id: 'openai/gpt-5.6-sol-pro' } });
    assert.doesNotMatch(JSON.stringify(api), /synthesized|slot_id":"deep|enabled":true,"route":\{"kind":"api_chat","target_id":"openai\/gpt-5\.6-sol-pro/);

    // A session row keeps its pin and an explicit effort.
    const session = JSON.parse(buildReviewerSlotsSetting({
        ...base,
        deepReview: { route: { kind: ROUTE_KIND_SESSION, target_id: 'codex=gpt-5.6-sol', profile_id: 'koshak' }, effort: 'xhigh', subagent_id: '' },
    }));
    assert.deepEqual(session.deep_review, {
        route: { kind: ROUTE_KIND_SESSION, target_id: 'codex=gpt-5.6-sol', profile_id: 'koshak' }, effort: 'xhigh',
    });

    // A configured-subagent reference carries no route knobs (decision 5A).
    const ref = JSON.parse(buildReviewerSlotsSetting({
        ...base,
        deepReview: { subagent_id: 'deep-critic', route: { kind: ROUTE_KIND_API, target_id: 'stash' }, effort: '' },
    }));
    assert.deepEqual(ref.deep_review, { subagent_id: 'deep-critic' });
});

test('the deep self-review block says the ONE difference from the advisory where the owner picks', () => {
    const markup = renderReviewerSlotsSection();
    assert.match(markup, /<h4[^>]*>Deep self-review<\/h4>/);
    assert.match(markup, /id="reviewer-deep-review-row"/);
    // API model = one packed review here; the advisory's API model = inspection episode.
    assert.match(markup, /receives ONE packed review/);
    assert.match(markup, /unlike the advisory, whose API model runs an inspection episode/);
    assert.match(markup, /native\s+inspection episode with host-observed reads/);
    assert.match(markup, /reads not host-observed/);
    assert.match(markup, /memory whitelist reaches the reviewer\s+inline byte-exact/);
    assert.match(markup, /outranks the Behavior-tab deep\s+self-review effort/);

    const roster = [
        { subagent_id: 'api-critic', route: { kind: 'api_model', target_id: 'openai/gpt-5.6-terra' } },
        { subagent_id: 'sess', route: { kind: ROUTE_KIND_SESSION, target_id: 'codex=gpt-5.6-sol' } },
    ];
    assert.match(deepReviewDeliveryNote({ route: { kind: ROUTE_KIND_API, target_id: 'openai/x' } }), /One packed review/);
    assert.match(deepReviewDeliveryNote({ route: { kind: ROUTE_KIND_API, target_id: 'openai/x' } }), /inspection episode instead/);
    assert.match(deepReviewDeliveryNote({ subagent_id: 'api-critic' }, { roster }), /Native inspection episode/);
    assert.match(deepReviewDeliveryNote({ subagent_id: 'api-critic' }, { roster }), /host-observed/);
    assert.match(deepReviewDeliveryNote({ subagent_id: 'api-critic' }, { roster }), /memory whitelist reaches it inline byte-exact/);
    assert.match(deepReviewDeliveryNote({ subagent_id: 'sess' }, { roster }), /Agent session .* not host-observed/);
    const direct = deepReviewDeliveryNote({ route: { kind: ROUTE_KIND_SESSION, target_id: 'codex' } }, { harnesses: { codex: { status: 'ok' } } });
    assert.match(direct, /agent session — retrieves context with its own tools · route ok — reads not host-observed/);
    // Absence claims follow provenance, as everywhere in this editor.
    assert.match(deepReviewDeliveryNote({ subagent_id: 'gone' }, { roster }), /none exists with this ID/);
    assert.match(deepReviewDeliveryNote({ subagent_id: 'gone' }, { roster: [], rosterKnown: false }), /could not be read/);
});

test('the Models tab no longer authors the deep self-review model (R7)', () => {
    // The row lives in Review lanes; the key survives only as the backend's
    // invisible migration source, so no Settings control writes it.
    const page = renderSettingsPage();
    assert.doesNotMatch(page, /s-deep-self-review-model|Deep Self-Review Model/);
    assert.match(page, /id="s-websearch-model"/, 'the sibling field in Other Model Slots stays');
    assert.match(page, /id="s-effort-deep-self-review"/, 'the Behavior-tab surface effort stays (the row effort outranks it)');
});

test('the standing note states the POLICY, never the current routing', () => {
    // The section's inline note is STATIC markup: it renders identically for an
    // owner with three subscriptions and for one with none, whose every row is
    // an API model. It used to open "Commit and scope review run on
    // subscriptions and never fall back to API spend", so the second owner read
    // Settings and believed their commit reviews were spending subscription
    // windows and would wait for capacity — while in fact they were spending
    // API budget on the next commit (BIBLE P1). The only conditional sentence
    // lives server-side: `reviewer_slot_config.reviewer_slot_save_check` returns
    // the one-time R12 disclosure when a save first gives the triad a retrieving
    // row; a static copy cannot carry that condition, so this note must state
    // the rule instead of the situation.
    const markup = renderReviewerSlotsSection();

    // The section ships with NO rows — they are painted from the saved setting
    // at runtime — so this is exactly the markup an all-API owner sees.
    assert.doesNotMatch(markup, /data-route-kind|session:/,
        'the static section must not ship a pre-rendered row');

    assert.match(markup, /Rows routed to a subscription never fall back to API spend/);
    assert.match(markup, /waits for capacity/);
    // The unconditional claim, in the shapes it could come back as.
    assert.doesNotMatch(markup, /review runs? on subscriptions/i);
    assert.doesNotMatch(markup, /reviews? run on your subscription/i);
    assert.match(markup, /skill\s+review and task acceptance all follow their configured rows/i);
    // Owner R2 (2026-09-01): task acceptance follows the triad rows on their own
    // delivery — the former "remains API-only" pin must not come back in any spelling.
    assert.match(markup, /task acceptance runs\s+the triad rows on their own delivery/i);
    assert.doesNotMatch(markup, /API-only|shipped defaults when none remain/i);
});


test('each group carries its own Add action in its head, above the rows it adds to', () => {
    // docs/DESIGN.md "List editors": a group's add action lives in its head,
    // never in a footer toolbar under the rows, and the new row lands at the
    // group's end, revealed by `revealNewRow`.
    const markup = renderReviewerSlotsSection();
    for (const [head, rows, button] of [
        ['Triad slots', 'reviewer-triad-rows', 'btn-add-triad-slot'],
        ['Scope slots', 'reviewer-scope-rows', 'btn-add-scope-slot'],
    ]) {
        const headingAt = markup.indexOf(`class="reviewer-slots-heading">${head}`);
        assert.ok(headingAt > 0, `${head} heading exists`);
        const headOpen = markup.lastIndexOf('<div class="reviewer-slots-head">', headingAt);
        const headClose = markup.indexOf('</div>', headingAt);
        assert.ok(headOpen > 0 && headOpen < headingAt, `${head} heading sits inside a group head`);
        assert.match(markup.slice(headOpen, headClose), new RegExp(`id="${button}"`),
            `${button} lives in the ${head} head`);
        assert.ok(headClose < markup.indexOf(`id="${rows}"`), `${head} head precedes its rows`);
    }
    assert.doesNotMatch(markup, /settings-toolbar/, 'no footer Add toolbar remains');
});

test('a saved account pin survives a discovery list that no longer contains it', () => {
    // The select's value must EXIST as an option or the browser silently selects the
    // first one — "automatic rotation" — so a row pinned to one account redrew as
    // unpinned whenever the daemon was down or that account was signed out. Nothing
    // looked wrong, and saving the panel made the widening real.
    const discovered = profileOptionsFor(['koshak', 'valentine'], 'koshak');
    assert.deepEqual(discovered.map((o) => o.value), ['', 'koshak', 'valentine']);

    const undiscovered = profileOptionsFor(['valentine'], 'koshak');
    assert.deepEqual(undiscovered.map((o) => o.value), ['', 'valentine', 'koshak']);
    assert.match(undiscovered[2].label, /not in discovery/);

    // Discovery empty entirely (daemon down) is the SAME case, not a special one.
    assert.deepEqual(profileOptionsFor([], 'koshak').map((o) => o.value), ['', 'koshak']);
    // No pin: nothing invented, and the rotation entry stays the only default.
    assert.deepEqual(profileOptionsFor([], '').map((o) => o.value), ['']);
    assert.deepEqual(profileOptionsFor(null, '').map((o) => o.value), ['']);
});

test('a disabled account is offered with a "(disabled)" label, still selectable', () => {
    // The index carries {id, enabled} entries; the shared builder says the
    // fact instead of offering a disabled account bare. The option stays
    // SELECTABLE: the engine's typed refusal is the authority on a pin it
    // will not serve (D-U6) — this is honesty, not a client-side gate.
    const options = profileOptionsFor([
        { id: 'koshak', enabled: true },
        { id: 'retired', enabled: false },
    ], '');
    assert.deepEqual(options.map((o) => o.value), ['', 'koshak', 'retired']);
    assert.equal(options[1].label, 'Account: koshak (pinned)');
    assert.equal(options[2].label, 'Account: retired (pinned) (disabled)');
    assert.ok(options.every((o) => !o.disabled), 'every account stays selectable');
});

test('the provider shown for a delegated row is the harness name, never Claudexor', () => {
    const groups = routeChoiceGroups({
        harnesses: [{ id: 'codex', display_name: 'Codex CLI', status: 'ok', enabled: true }],
    });
    const flat = JSON.stringify(groups);
    assert.ok(flat.includes('Codex CLI'));
    assert.ok(!/claudexor/i.test(flat), 'the aggregator brand must not appear as a provider');
    // The route select carries ROUTES only (finding #6): one API entry, one
    // entry per harness — never the flat model catalog. Both groups labeled.
    assert.equal(groups[0].label, 'API');
    assert.deepEqual(groups[0].options, [{ value: API_ROUTE_CHOICE, label: 'API model' }]);
    assert.equal(groups[1].options[0].value, 'session:codex');
});

test('a saved session route survives a discovery list that no longer contains its harness', () => {
    // Same rule as profileOptionsFor: the select's value must EXIST as an
    // option or the browser silently redraws the row as the first choice.
    const groups = routeChoiceGroups({ harnesses: [{ id: 'codex' }], currentChoice: 'session:claude' });
    const session = groups[1].options;
    assert.deepEqual(session.map((o) => o.value), ['session:codex', 'session:claude']);
    assert.match(session[1].label, /not in discovery/);
    // A choice discovery DOES list gains no duplicate.
    const listed = routeChoiceGroups({ harnesses: [{ id: 'codex' }], currentChoice: 'session:codex' });
    assert.deepEqual(listed[1].options.map((o) => o.value), ['session:codex']);
});

test('no :: syntax anywhere in encoded choices or composed targets', () => {
    const groups = routeChoiceGroups({
        harnesses: [{ id: 'codex' }, { id: 'claude' }],
    });
    for (const group of groups) {
        for (const option of group.options) {
            assert.ok(!option.value.includes('::'), option.value);
        }
    }
    assert.equal(composeSessionTarget('codex', 'gpt-5.6-sol'), 'codex=gpt-5.6-sol');
    assert.equal(composeSessionTarget('codex', ''), 'codex');
});

test('route choice round-trips through encode/decode', () => {
    // The API choice no longer carries the model id — the free-text input
    // does — so encode collapses every api row to the ONE api option, and a
    // fresh row (target '') displays exactly that option, not the first
    // catalog model (finding #6c).
    const apiRow = { route: { kind: ROUTE_KIND_API, target_id: 'openai/gpt-5.6-luna' } };
    assert.equal(encodeRouteChoice(apiRow), API_ROUTE_CHOICE);
    assert.equal(encodeRouteChoice({ route: { kind: ROUTE_KIND_API, target_id: '' } }), API_ROUTE_CHOICE);
    assert.deepEqual(decodeRouteChoice(encodeRouteChoice(apiRow)), { kind: ROUTE_KIND_API });
    const sessionRow = { route: { kind: ROUTE_KIND_SESSION, target_id: 'codex=gpt-5.6-sol' } };
    assert.deepEqual(decodeRouteChoice(encodeRouteChoice(sessionRow)),
        { kind: ROUTE_KIND_SESSION, harness: 'codex' });
    assert.deepEqual(splitSessionTarget('codex=gpt-5.6-sol'),
        { harness: 'codex', model: 'gpt-5.6-sol' });
});

test('advisory route switching never wipes a stored target (finding #7c)', () => {
    // Saved: api with an explicit target. Flip to a session and back: the
    // saved api target is restored, not written to ''.
    const savedApi = { kind: ROUTE_KIND_API, target_id: 'anthropic/claude-opus-5' };
    let memory = { api: { ...savedApi }, session: null };
    const toSession = advisoryRouteTransition(savedApi, { kind: ROUTE_KIND_SESSION, harness: 'codex' }, memory);
    assert.deepEqual(toSession.route, { kind: ROUTE_KIND_SESSION, target_id: 'codex' });
    const back = advisoryRouteTransition(toSession.route, { kind: ROUTE_KIND_API }, toSession.memory);
    assert.deepEqual(back.route, savedApi);

    // Saved: session with a model spec. Kind round-trip restores the FULL
    // spec (harness=model), not the bare harness.
    const savedSession = { kind: ROUTE_KIND_SESSION, target_id: 'claude=claude-opus-5' };
    memory = { api: null, session: { ...savedSession } };
    const toApi = advisoryRouteTransition(savedSession, { kind: ROUTE_KIND_API }, memory);
    assert.deepEqual(toApi.route, { kind: ROUTE_KIND_API, target_id: '' });
    const restored = advisoryRouteTransition(toApi.route, { kind: ROUTE_KIND_SESSION, harness: 'claude' }, toApi.memory);
    assert.deepEqual(restored.route, savedSession);

    // Re-selecting the CURRENT kind/harness is a no-op, never a reset.
    const noop = advisoryRouteTransition(savedSession, { kind: ROUTE_KIND_SESSION, harness: 'claude' }, memory);
    assert.deepEqual(noop.route, savedSession);
});

test('the composed setting carries stable ids, per-row routes/efforts and the optional pin', () => {
    const setting = JSON.parse(buildReviewerSlotsSetting({
        triad: [
            { slot_id: 't_api', route: { kind: ROUTE_KIND_API, target_id: 'openai/gpt-5.6-luna' }, effort: 'high' },
            { slot_id: 't_sess', route: { kind: ROUTE_KIND_SESSION, target_id: 'codex=gpt-5.6-sol', profile_id: 'koshak' }, effort: '' },
        ],
        scope: [{ slot_id: 's_1', route: { kind: ROUTE_KIND_API, target_id: 'openai/gpt-5.6-terra' }, effort: 'xhigh' }],
        advisory: { enabled: false, route: { kind: ROUTE_KIND_SESSION, target_id: 'codex' }, effort: 'low' },
    }));
    assert.deepEqual(setting.triad[0], {
        slot_id: 't_api',
        route: { kind: 'api_chat', target_id: 'openai/gpt-5.6-luna' },
        effort: 'high',
    });
    // '' effort means "surface default" and is OMITTED, never written as ''.
    assert.equal('effort' in setting.triad[1], false);
    // The optional manual pin (Q2-в) rides only when set; rotation is default.
    assert.equal(setting.triad[1].route.profile_id, 'koshak');
    assert.equal('profile_id' in setting.scope[0].route, false);
    assert.equal(setting.advisory.enabled, false);
    assert.equal(setting.advisory.effort, 'low');
});

test('an advisory session preserves route-default effort instead of inventing low', () => {
    const setting = JSON.parse(buildReviewerSlotsSetting({
        triad: [{ slot_id: 't1', route: { kind: ROUTE_KIND_API, target_id: 'm/t' } }],
        scope: [{ slot_id: 's1', route: { kind: ROUTE_KIND_API, target_id: 'm/s' } }],
        advisory: {
            enabled: true,
            route: { kind: ROUTE_KIND_SESSION, target_id: 'claude=claude-fable-5' },
            effort: '',
        },
    }));
    assert.equal(setting.advisory.effort, '');

    const api = JSON.parse(buildReviewerSlotsSetting({
        triad: setting.triad,
        scope: setting.scope,
        advisory: { enabled: true, route: { kind: ROUTE_KIND_API, target_id: '' }, effort: '' },
    }));
    assert.equal(api.advisory.route.kind, ROUTE_KIND_API);
    assert.equal(api.advisory.effort, 'low');
});

test('minted slot ids are prefixed, unique, and never an array index', () => {
    const taken = ['triad_abc123'];
    const minted = mintSlotId('triad', taken);
    assert.match(minted, /^triad_[a-z0-9]{4,}$/);
    assert.ok(!taken.includes(minted));
    assert.notEqual(mintSlotId('scope', []), mintSlotId('scope', []));
});

test('the runs-as line is the capability_delta projection, compact and honest', () => {
    const line = describeLastExecution({
        ts: '2026-08-03T10:00:00Z',
        effective: { route: 'agent_session:codex', model: 'gpt-5.6-sol', effort: 'xhigh',
                     verdict_method: 'light_model_extraction' },
        capability_delta: [{ reason: 'extraction_instead_of_schema' }],
    });
    assert.ok(line.includes('codex session'));
    assert.ok(line.includes('gpt-5.6-sol'));
    assert.ok(line.includes('verdict via light model extraction'));
    assert.ok(line.includes('1 capability delta disclosed'));
    // The timestamp is humanized for the visible line; the raw ISO instant
    // stays recoverable in the row tooltip, never inline.
    assert.ok(!line.includes('2026-08-03T10:00:00Z'));
    assert.equal(describeLastExecution(null), '');
});

test('capability badges display facts and never configure', () => {
    const sessionRow = { route: { kind: ROUTE_KIND_SESSION, target_id: 'codex' } };
    assert.ok(capabilityBadge(sessionRow, { codex: { status: 'ok' } }).includes('route ok'));
    assert.ok(capabilityBadge(sessionRow, {}).includes('not discovered'));
    const apiRow = { route: { kind: ROUTE_KIND_API, target_id: 'openai/gpt-5.6-luna' } };
    assert.equal(capabilityBadge(apiRow, {}), 'API delivery');
});

test('the session model-options fragment guards a saved model discovery no longer lists', () => {
    // ONE fragment for every session model select (triad, scope, advisory,
    // Subagents): "Engine default model" first, discovery next, and a saved
    // model the daemon cannot see right now keeps an option — otherwise the
    // browser silently redraws the select as "Engine default" and the next
    // Save really erases the pin (REG:1190 #7 class).
    const listed = sessionModelOptions({ models: [{ id: 'sonnet' }, { id: 'claude-opus-5' }] }, 'sonnet');
    assert.deepEqual(listed.map((o) => o.value), ['', 'sonnet', 'claude-opus-5']);
    assert.equal(listed[0].label, 'Engine default model');

    const unlisted = sessionModelOptions({ models: [{ id: 'claude-opus-5' }] }, 'sonnet');
    assert.deepEqual(unlisted.map((o) => o.value), ['', 'claude-opus-5', 'sonnet']);
    assert.match(unlisted[2].label, /not in discovery/);

    // Daemon down (no harness at all) is the SAME case, not a special one.
    assert.deepEqual(sessionModelOptions(null, 'sonnet').map((o) => o.value), ['', 'sonnet']);
    assert.deepEqual(sessionModelOptions(null, '').map((o) => o.value), ['']);
});

test('a per-harness model-read failure is not a discovery: the not-in-discovery claim is withdrawn', () => {
    // Discovery is TWO reads. The endpoint answers `models: []` with a typed
    // `models_error` for one harness while the daemon stays globally running
    // — so `catalogKnown` is true and the empty list is NOT authoritative for
    // this harness. Reading it as one labelled the saved model
    // "gpt-saved (not in discovery)": a successful discovery cited as proof of
    // absence, for a discovery that never happened.
    const refused = { id: 'codex', models: [], models_error: 'models_probe_failed' };
    assert.equal(harnessModelsKnown(refused, true), false, 'the catalog read says nothing about this list');
    assert.equal(harnessModelsKnown({ id: 'codex', models: [{ id: 'x' }] }, true), true);
    assert.equal(harnessModelsKnown({ id: 'codex', models: [{ id: 'x' }] }, false), false,
        'an unread catalog still withdraws the claim');

    const options = sessionModelOptions(refused, 'gpt-saved', { catalogKnown: true });
    assert.deepEqual(options.map((o) => o.value), ['', 'gpt-saved'], 'the saved pin keeps its option');
    assert.equal(options[1].label, 'gpt-saved (not checked)',
        'and trades the absence verdict for the honest "not checked"');
    assert.doesNotMatch(JSON.stringify(options), /not in discovery/);

    // …and the gap is SAID, not left as a silently short list.
    assert.match(modelsGapNote(refused, true), /model list could not be read/);
    assert.equal(modelsGapNote({ id: 'codex', models: [{ id: 'x' }] }, true), '');
    // With the catalog itself unread the section note already explains it; a
    // second sentence for the same silence would be noise.
    assert.equal(modelsGapNote(refused, false), '');

    // A harness whose list really WAS read keeps the honest accusation.
    const read = { id: 'codex', models: [{ id: 'gpt-5.6-sol' }] };
    assert.match(sessionModelOptions(read, 'gpt-saved', { catalogKnown: true })[2].label,
        /not in discovery/);
});

test('an advisory session model composes into the target and survives save-load', () => {
    // Compose exactly as the data-advisory-model handler does…
    const target = composeSessionTarget('codex', 'gpt-5.6-luna');
    const setting = JSON.parse(buildReviewerSlotsSetting({
        triad: [{ slot_id: 't1', route: { kind: ROUTE_KIND_API, target_id: 'openai/x' }, effort: '' }],
        scope: [{ slot_id: 's1', route: { kind: ROUTE_KIND_API, target_id: 'openai/y' }, effort: '' }],
        advisory: { enabled: true, route: { kind: ROUTE_KIND_SESSION, target_id: target, profile_id: 'koshak' }, effort: 'low' },
    }));
    // …and the saved value round-trips model AND profile pin intact.
    assert.equal(setting.advisory.route.kind, ROUTE_KIND_SESSION);
    assert.equal(setting.advisory.route.target_id, 'codex=gpt-5.6-luna');
    assert.equal(setting.advisory.route.profile_id, 'koshak');
    assert.ok(!setting.advisory.route.target_id.includes('::'));
    assert.deepEqual(splitSessionTarget(setting.advisory.route.target_id),
        { harness: 'codex', model: 'gpt-5.6-luna' });
});

test('switching the advisory harness resets the model to the bare harness', () => {
    // The model belongs to the harness it was picked for (same rule as the
    // triad rows and the Subagents tail): A→B lands on B's engine default,
    // spelled as the bare harness — never A's model carried across.
    const saved = { kind: ROUTE_KIND_SESSION, target_id: 'claude=claude-opus-5' };
    const out = advisoryRouteTransition(saved, { kind: ROUTE_KIND_SESSION, harness: 'codex' },
        { api: null, session: { ...saved } });
    assert.deepEqual(out.route, { kind: ROUTE_KIND_SESSION, target_id: 'codex' });
});

test('the advisory api route writes the shared api_chat kind, never the retired api alias', () => {
    // Successor pin for the retired Claude-SDK 'api' kind: the advisory row is
    // on the SHARED vocabulary now. A stale 'api' kind in a loaded draft still
    // normalizes to api_chat on save, the routed target rides verbatim, and no
    // profile pin is emitted on the api kind.
    const setting = JSON.parse(buildReviewerSlotsSetting({
        triad: [{ slot_id: 't1', route: { kind: ROUTE_KIND_API, target_id: 'openai/x' }, effort: '' }],
        scope: [{ slot_id: 's1', route: { kind: ROUTE_KIND_API, target_id: 'openai/y' }, effort: '' }],
        advisory: { enabled: true, route: { kind: 'api', target_id: 'anthropic/claude-sonnet-5', profile_id: 'koshak' }, effort: 'low' },
    }));
    assert.equal(setting.advisory.route.kind, ROUTE_KIND_API);
    assert.equal(setting.advisory.route.target_id, 'anthropic/claude-sonnet-5');
    assert.equal('profile_id' in setting.advisory.route, false);
    // The retired spelling never appears anywhere in the composed bytes.
    assert.doesNotMatch(JSON.stringify(setting), /"kind":"api"/);
});

test('a configured-subagent reference serializes without route knobs (decision 5A)', () => {
    // The stored forms are mutually exclusive: a reference never duplicates
    // route/model/account knobs (the roster row is their SSOT), and an empty
    // explicit effort is OMITTED so the roster row's own effort keeps deciding.
    const setting = JSON.parse(buildReviewerSlotsSetting({
        triad: [
            { slot_id: 't_ref', subagent_id: 'deep-reviewer', route: { kind: ROUTE_KIND_API, target_id: 'stash/kept' }, effort: 'high' },
            { slot_id: 't_ref2', subagent_id: 'fast-reviewer', effort: '' },
        ],
        scope: [{ slot_id: 's_1', route: { kind: ROUTE_KIND_API, target_id: 'openai/y' }, effort: '' }],
        advisory: { enabled: true, subagent_id: 'deep-reviewer', route: { kind: ROUTE_KIND_API, target_id: 'stash' }, effort: '' },
    }));
    assert.deepEqual(setting.triad[0], { slot_id: 't_ref', subagent_id: 'deep-reviewer', effort: 'high' });
    assert.deepEqual(setting.triad[1], { slot_id: 't_ref2', subagent_id: 'fast-reviewer' });
    assert.deepEqual(setting.advisory, { enabled: true, subagent_id: 'deep-reviewer' });
    // A direct row in the same panel keeps its inline route untouched.
    assert.equal(setting.scope[0].route.kind, ROUTE_KIND_API);
});

test('the roster select survives a saved reference the roster no longer lists', () => {
    // Same rule as profileOptionsFor: the select's value must EXIST as an
    // option or the browser silently redraws the row as the first roster entry
    // and the next Save really rewires the reviewer. The absence claim follows
    // provenance: only a roster that was READ may say "not in the roster".
    const roster = [
        { subagent_id: 'deep', recommended_use: 'Long reasoning over big diffs',
          route: { kind: 'api_model', target_id: 'openai/gpt-5.6-sol' }, effort: 'high' },
        { subagent_id: 'fast', recommended_use: '',
          route: { kind: 'agent_session', target_id: 'cursor=grok-4.6' } },
    ];
    const listed = subagentOptionsFor(roster, 'deep');
    assert.deepEqual(listed.map((o) => o.value), ['deep', 'fast']);
    // 2=A label contract: FACTS lead (channel first), description is a caption.
    assert.equal(listed[0].label, '#deep · API · openai/gpt-5.6-sol · high — Long reasoning over big diffs');
    assert.equal(listed[1].label, '#fast · cursor · grok-4.6');

    const missing = subagentOptionsFor(roster, 'gone');
    assert.deepEqual(missing.map((o) => o.value), ['deep', 'fast', 'gone']);
    assert.match(missing[2].label, /not in the roster/);
    // An unreadable roster licenses no absence claim.
    assert.match(subagentOptionsFor([], 'gone', { rosterKnown: false })[0].label, /not checked/);
    assert.doesNotMatch(subagentOptionsFor([], 'gone', { rosterKnown: false })[0].label, /not in the roster/);
});

test('the one flat reviewer picker leads with roster references, then the inline channels (1=B)', () => {
    const roster = [
        { subagent_id: 'deep', recommended_use: 'Long reasoning',
          route: { kind: 'api_model', target_id: 'openai/gpt-5.6-sol' }, effort: 'high' },
    ];
    const harnesses = [{ id: 'claude', display_name: 'Claude Code' }];

    // A reference row: its select value is the prefixed id, the roster group
    // leads, and the stashed inline route adds NO undiscovered session entry.
    const refRow = { subagent_id: 'deep', route: { kind: ROUTE_KIND_SESSION, target_id: 'gone=old' } };
    assert.equal(encodeReviewerChoice(refRow), `${SUBAGENT_CHOICE_PREFIX}deep`);
    const refGroups = reviewerChoiceGroups({ roster, row: refRow, harnesses });
    assert.equal(refGroups[0].label, 'Available subagents');
    assert.deepEqual(refGroups[0].options.map((o) => o.value), [`${SUBAGENT_CHOICE_PREFIX}deep`]);
    assert.match(refGroups[0].options[0].label, /^#deep · API/);
    assert.deepEqual(refGroups.slice(1).map((g) => g.label), ['API', 'Agents — subscriptions']);
    assert.ok(!refGroups.slice(1).flatMap((g) => g.options).some((o) => o.value === 'session:gone'));

    // An inline row: its own choice threads down so an undiscovered harness
    // keeps its option (the survive-the-save rule).
    const inlineRow = { subagent_id: '', route: { kind: ROUTE_KIND_SESSION, target_id: 'gone=old' } };
    assert.equal(encodeReviewerChoice(inlineRow), 'session:gone');
    const inlineGroups = reviewerChoiceGroups({ roster, row: inlineRow, harnesses });
    assert.ok(inlineGroups.at(-1).options.some((o) => o.value === 'session:gone'));

    // A saved reference missing from the roster survives as a prefixed option;
    // an empty roster contributes no group at all.
    const missingGroups = reviewerChoiceGroups({ roster, row: { subagent_id: 'gone' }, harnesses });
    assert.ok(missingGroups[0].options.some(
        (o) => o.value === `${SUBAGENT_CHOICE_PREFIX}gone` && /not in the roster/.test(o.label)));
    const emptyGroups = reviewerChoiceGroups({ roster: [], row: { subagent_id: '' }, harnesses });
    assert.notEqual(emptyGroups[0].label, 'Available subagents');

    // The advisory api label rides through.
    const advisory = reviewerChoiceGroups({ roster: [], row: { subagent_id: '' }, harnesses, apiLabel: 'API model (inspection episode)' });
    assert.equal(advisory[0].options[0].label, 'API model (inspection episode)');
});

test('an advisory reference switch keeps an explicit effort override; crossing from inline clears it', () => {
    // Sol delta-review finding: the merged picker unconditionally cleared the
    // advisory effort on ANY reference pick, silently erasing a saved
    // {subagent_id, effort: 'xhigh'} override on a reference-to-reference
    // switch. Inline branches still shed their effort ('' = the roster row's
    // own effort decides).
    assert.deepEqual(
        advisoryReferenceTransition({ subagent_id: 'deep', effort: 'xhigh' }, 'fast'),
        { subagent_id: 'fast', effort: 'xhigh' });
    assert.deepEqual(
        advisoryReferenceTransition({ subagent_id: '', effort: 'low', route: { kind: ROUTE_KIND_API } }, 'deep'),
        { subagent_id: 'deep', effort: '' });
    assert.deepEqual(
        advisoryReferenceTransition({ subagent_id: '', effort: 'high', route: { kind: ROUTE_KIND_SESSION, target_id: 'codex=x' } }, 'deep'),
        { subagent_id: 'deep', effort: '' });
});

test('picker captions strip directional marks and never split a surrogate pair', () => {
    const marked = subagentOptionsFor([{
        subagent_id: 'row',
        recommended_use: '\u200Efast\u200F \u061Ccheap\u202Eevil',
        route: { kind: 'api_model', target_id: 'openai/gpt-5.6-luna' },
    }], '')[0].label;
    assert.equal(marked, '#row · API · openai/gpt-5.6-luna — fast cheap' + 'evil');

    const emoji = '\u{1F9EA}'.repeat(60); // 60 code points, 120 UTF-16 units
    const long = subagentOptionsFor([{
        subagent_id: 'row',
        recommended_use: emoji,
        route: { kind: 'api_model', target_id: 'x' },
    }], '')[0].label;
    const caption = long.split(' — ')[1];
    assert.equal(caption, '\u{1F9EA}'.repeat(45) + '…');
});

test('the derived disclosure reports the roster row facts read-only, with honest absence', () => {
    const roster = [
        {
            subagent_id: 'sess',
            name: 'Session Reviewer',
            route: { kind: ROUTE_KIND_SESSION, target_id: 'codex=gpt-5.6-sol', credential_profile_id: 'koshak' },
            effort: 'xhigh',
        },
        { subagent_id: 'apirow', name: 'Api Reviewer', route: { kind: 'api_model', target_id: 'openai/gpt-5.6-luna' } },
    ];
    const session = describeSubagentReference('sess', roster);
    assert.ok(session.includes('codex session'));
    assert.ok(session.includes('gpt-5.6-sol'));
    assert.ok(session.includes('account koshak'));
    assert.ok(session.includes('effort xhigh'));
    assert.match(session, /roster row/);

    const api = describeSubagentReference('apirow', roster);
    assert.ok(api.includes('API model openai/gpt-5.6-luna'));

    // A missing roster row is said plainly — kept, refusing, never rerouted —
    // and an UNREAD roster never claims the row does not exist.
    assert.match(describeSubagentReference('gone', roster), /no roster row with this ID exists/);
    assert.match(describeSubagentReference('gone', roster), /refuse rather than reroute/);
    assert.match(describeSubagentReference('gone', [], { rosterKnown: false }), /roster could not be read/);
});

test('the runs-as line shows APPLIED account/access and honest absence for an undisclosed model', () => {
    const applied = describeLastExecution({
        effective: { route: 'agent_session:codex', model: 'gpt-5.6-sol',
                     profile_id: 'koshak', access: 'readonly', effort: 'xhigh' },
    });
    assert.ok(applied.includes('account koshak'));
    assert.ok(applied.includes('access readonly'));
    // Old telemetry: a session with NO resolved model says so — the requested
    // model never masquerades as the applied one.
    const bare = describeLastExecution({ effective: { route: 'agent_session:codex' } });
    assert.ok(bare.includes('model not disclosed'));
    assert.ok(!bare.includes('account'));
    // An api row keeps its sent-model-is-applied-model reading with no noise.
    const api = describeLastExecution({ effective: { route: 'api_chat', model: 'openai/x' } });
    assert.ok(api.includes('openai/x') && !api.includes('not disclosed'));
});

// ---------------------------------------------------------------------------
// Phase 2: a row may only be labeled "(not in discovery)" after a SUCCESSFUL
// discovery. With the daemon stopped (or the endpoint unreachable) the backend
// answers `harnesses: []` by construction, and this page used to stamp that
// label onto every saved row while explaining nothing — the exact screen the
// owner reported (2026-08-08). The saved option itself must survive either
// way: that guard is what stops the next Save from erasing the pin.
// ---------------------------------------------------------------------------

test('an unread facet never accuses a saved row of being undiscovered', () => {
    // Same empty discovery, two different worlds.
    const discoveredMiss = routeChoiceGroups({ harnesses: [], currentChoice: 'session:codex' });
    assert.match(discoveredMiss[1].options[0].label, /not in discovery/);

    const cannotAsk = routeChoiceGroups({ harnesses: [], currentChoice: 'session:codex', catalogKnown: false });
    assert.equal(cannotAsk[1].options[0].value, 'session:codex', 'the saved option SURVIVES');
    assert.equal(cannotAsk[1].options[0].label, 'codex (not checked)',
        'and is labelled unchecked, never undiscovered');
    assert.doesNotMatch(JSON.stringify(cannotAsk), /not in discovery/);
    // The empty-group placeholder stops promising a sign-in that would not help.
    const emptyGroup = routeChoiceGroups({ harnesses: [], catalogKnown: false })[1].options[0];
    assert.doesNotMatch(emptyGroup.label, /sign in under Providers/);
});

test('an unread facet is labelled "not checked", never "not in discovery"', () => {
    // Rebased provenance pin (invariant 5): the suffix is the FACET's own
    // verdict — the account pin answers to the ACCOUNTS facet, the model and
    // route to the CATALOG facet. Hardcoding "not in discovery" into any suffix
    // helper stays green through every behavioural test that passes the
    // authoritative default, so this walks the unread worlds explicitly.
    const pins = profileOptionsFor(['valentine'], 'koshak', { accountsKnown: false });
    const pin = pins.find((o) => o.value === 'koshak');
    assert.ok(pin, 'an unread account store dropped the saved pin');
    assert.match(pin.label, /not checked/);
    assert.doesNotMatch(pin.label, /not in discovery/);
    // A facet that WAS read keeps the honest accusation.
    assert.match(profileOptionsFor(['valentine'], 'koshak', { accountsKnown: true })
        .find((o) => o.value === 'koshak').label, /not in discovery/);

    const models = sessionModelOptions({ models: [{ id: 'other' }] }, 'gpt-5.6-sol', { catalogKnown: false });
    const savedModel = models.find((o) => o.value === 'gpt-5.6-sol');
    assert.ok(savedModel, 'an unread catalog dropped the saved model');
    assert.match(savedModel.label, /not checked/);
    assert.doesNotMatch(savedModel.label, /not in discovery/);

    // A refused per-harness model read (catalog itself fine) is the SAME
    // unread world for this one list: `models_error` withdraws the claim.
    const refused = sessionModelOptions({ models: [], models_error: 'daemon_unreachable' },
        'gpt-5.6-sol', { catalogKnown: true });
    assert.match(refused.find((o) => o.value === 'gpt-5.6-sol').label, /not checked/);

    // The saved ROUTE is not called undiscovered before the catalog was read.
    const route = routeChoiceGroups({ harnesses: [], currentChoice: 'session:codex', catalogKnown: false });
    assert.match(route[1].options[0].label, /not checked/);
    assert.doesNotMatch(route[1].options[0].label, /not in discovery/);

    // Emptiness states ABSENCE only when the catalog was actually read.
    const emptyRead = routeChoiceGroups({ harnesses: [], catalogKnown: true })[1].options[0];
    assert.match(emptyRead.label, /None available/);
    const emptyUnread = routeChoiceGroups({ harnesses: [], catalogKnown: false })[1].options[0];
    assert.doesNotMatch(emptyUnread.label, /None available/);
});

test('the model and account pins survive a daemon-down save without the undiscovered label', () => {
    const models = sessionModelOptions({ models: [] }, 'gpt-5.6-sol', { catalogKnown: false });
    assert.deepEqual(models.map((o) => o.value), ['', 'gpt-5.6-sol'], 'the pin keeps its option');
    assert.equal(models[1].label, 'gpt-5.6-sol (not checked)');
    assert.match(sessionModelOptions({ models: [] }, 'gpt-5.6-sol')[1].label, /not in discovery/);

    const pins = profileOptionsFor([], 'koshak', { accountsKnown: false });
    assert.deepEqual(pins.map((o) => o.value), ['', 'koshak']);
    assert.match(pins[1].label, /not checked/);
    assert.doesNotMatch(pins[1].label, /not in discovery/);
    assert.match(profileOptionsFor([], 'koshak')[1].label, /not in discovery/);

    // …and the pin still reaches the save payload unchanged, which is the
    // whole point of keeping the option (a Save with the daemon down must not
    // silently widen which account a reviewer may spend).
    const row = { slot_id: 'triad_a', route: { kind: ROUTE_KIND_SESSION, target_id: 'codex=gpt-5.6-sol', profile_id: 'koshak' } };
    const saved = JSON.parse(buildReviewerSlotsSetting({ triad: [row], scope: [], advisory: {} }));
    assert.deepEqual(saved.triad[0].route, { kind: ROUTE_KIND_SESSION, target_id: 'codex=gpt-5.6-sol', profile_id: 'koshak' });
});

test('the delivery badge does not claim "route not discovered" when nobody could be asked', () => {
    const row = { route: { kind: ROUTE_KIND_SESSION, target_id: 'codex' } };
    assert.match(capabilityBadge(row, {}), /route not discovered/);
    assert.doesNotMatch(capabilityBadge(row, {}, { catalogKnown: false }), /not discovered/);
    assert.match(capabilityBadge(row, {}, { catalogKnown: false }), /agent session/);
});

test('facets are independent: an unread ACCOUNT store does not silence the CATALOG verdict', () => {
    // The concrete mislabel a single global verdict produces: the harness
    // catalog was read fine and genuinely no longer lists `claude`, while the
    // credential-profile read never happened. The route option must keep its
    // earned "(not in discovery)" and the account pin must NOT be given one.
    const groups = routeChoiceGroups({
        harnesses: [{ id: 'codex' }], currentChoice: 'session:claude', catalogKnown: true,
    });
    assert.match(groups[1].options.at(-1).label, /not in discovery/);

    const pins = profileOptionsFor([], 'koshak', { accountsKnown: false });
    assert.doesNotMatch(pins[1].label, /not in discovery/);
    assert.deepEqual(pins.map((o) => o.value), ['', 'koshak']);
});

test('neither facet gap is dropped: the tab banner names it and the section claims nothing', async () => {
    // This section renders two facets — the route/model lists from the catalog,
    // the account pins from the credential profiles — and a gap in EITHER used
    // to vanish. The sentence now belongs to the tab's ONE service banner (the
    // sections moved onto the Agents tab, and three scattered service notes
    // became one); what this section owes is the other half of the same rule —
    // making no claim the unread facet never licensed.
    const store = (reads) => createClaudexorStatusStore({
        fetchImpl: async () => ({
            ok: true,
            status: 200,
            json: async () => ({ daemon: { state: 'running' }, harnesses: [], profiles: {}, quota: [], reads }),
        }),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
    });
    const pins = (accountsKnown) => profileOptionsFor(['koshak'], 'gone', { accountsKnown });
    const pinnedRows = {
        triad: [{ slot_id: 't1', route: { kind: ROUTE_KIND_SESSION, target_id: 'cursor:x', profile_id: 'gone' } }],
        profilesByHarness: { cursor: ['koshak'] },
    };

    const accountsDied = store({ catalog: 'ok', accounts: 'failed', quota: 'ok' });
    await accountsDied.refresh();
    assert.match(serviceBannerLine(accountsDied).text, /Your agent accounts could not be read/,
        'the banner NAMES the gap this section would otherwise have dropped');
    // …and the rows say nothing they did not earn: a pin is not "gone" because
    // nobody could ask, and the missing-account warning stays silent.
    assert.doesNotMatch(pins(accountsDied.accountsKnown)[1].label, /not in discovery/);
    assert.equal(pinnedAccountWarning({ ...pinnedRows, accountsKnown: accountsDied.accountsKnown }), '');
    accountsDied.dispose();

    const catalogDied = store({ catalog: 'failed', accounts: 'ok', quota: 'ok' });
    await catalogDied.refresh();
    const catalogLine = serviceBannerLine(catalogDied);
    assert.match(catalogLine.text, /Your agents could not be read/);
    assert.doesNotMatch(catalogLine.text, /agent accounts could not be read/, 'a healthy facet is not accused');
    // …and a read that merely did not land never claims nobody asked.
    assert.doesNotMatch(catalogLine.text, /was not asked/);
    // The route select points at that one sentence instead of writing a second.
    const empty = routeChoiceGroups({ harnesses: [], catalogKnown: catalogDied.catalogKnown })[1].options[0];
    assert.match(empty.label, /see the service banner above/);
    // With the accounts read, the SAME pin is now genuinely missing and says so.
    assert.match(pinnedAccountWarning({ ...pinnedRows, accountsKnown: catalogDied.accountsKnown }),
        /pinned to an account the agent service no longer lists/);
    catalogDied.dispose();

    // Both facets in the SAME state: the account pins are STILL named. They
    // used to be dropped for matching the catalog's enum, so the note spoke
    // only of agents while the pins sat on screen unexplained — equal state is
    // not equal subject.
    const both = store({ catalog: 'not_read', accounts: 'not_read', quota: 'not_read' });
    await both.refresh();
    const bothLine = serviceBannerLine(both);
    assert.match(bothLine.text, /daemon was not asked/);
    assert.equal(bothLine.text.match(/could not be read/g), null, 'one sentence covers both');
    both.dispose();

    const healthy = store({ catalog: 'ok', accounts: 'ok', quota: 'ok' });
    await healthy.refresh();
    assert.doesNotMatch(serviceBannerLine(healthy).text, /could not be read/,
        'nothing to say when everything was read');
    healthy.dispose();
});

test('an untouched deep self-review placeholder is omitted from the save; an edited or saved row rides it (items 1/10)', () => {
    const base = {
        triad: [{ slot_id: 't1', route: { kind: ROUTE_KIND_API, target_id: 'openai/x' }, effort: '' }],
        scope: [{ slot_id: 's1', route: { kind: ROUTE_KIND_API, target_id: 'openai/y' }, effort: '' }],
        advisory: { enabled: true, route: { kind: ROUTE_KIND_API, target_id: '' }, effort: 'low' },
    };
    // Synthesized (shown from the key) but untouched: OMITTED — the runtime
    // synthesizes the identical row and nothing is written behind the owner.
    const untouched = { route: { kind: ROUTE_KIND_API, target_id: 'openai/from-key' }, effort: '', subagent_id: '',
                        synthesizedFrom: 'OUROBOROS_MODEL_DEEP_SELF_REVIEW', materialized: false };
    assert.equal('deep_review' in JSON.parse(buildReviewerSlotsSetting({ ...base, deepReview: untouched })), false);
    // An empty placeholder (older server answered no row) is omitted too — a
    // triad/scope edit is never blocked by the singleton.
    const empty = { route: { kind: ROUTE_KIND_API, target_id: '' }, effort: '', subagent_id: '', synthesizedFrom: '', materialized: false };
    assert.equal('deep_review' in JSON.parse(buildReviewerSlotsSetting({ ...base, deepReview: empty })), false);
    // Edited (materialized) or loaded as SAVED: emitted verbatim.
    const edited = { ...untouched, materialized: true };
    assert.deepEqual(JSON.parse(buildReviewerSlotsSetting({ ...base, deepReview: edited })).deep_review,
        { route: { kind: 'api_chat', target_id: 'openai/from-key' } });
    // A blanked model box on an edited row IS emitted (the server's typed 400
    // is the refusal; the row itself says so before the save).
    const blanked = { ...edited, route: { kind: ROUTE_KIND_API, target_id: '' } };
    assert.deepEqual(JSON.parse(buildReviewerSlotsSetting({ ...base, deepReview: blanked })).deep_review,
        { route: { kind: 'api_chat', target_id: '' } });
    assert.match(deepReviewMetaNotes(blanked).join(' '), /Model id required — an empty model id is refused at save/);
    assert.match(deepReviewMetaNotes(untouched).join(' '), /Not saved as a row yet — shown from OUROBOROS_MODEL_DEEP_SELF_REVIEW/);
    assert.match(deepReviewMetaNotes(untouched).join(' '), /an untouched row is not written/);
    assert.deepEqual(deepReviewMetaNotes(edited), []);
    assert.deepEqual(deepReviewMetaNotes({ ...blanked, subagent_id: 'deep' }), []);
    assert.deepEqual(deepReviewMetaNotes({ ...blanked, route: { kind: ROUTE_KIND_SESSION, target_id: 'codex' } }), []);
});

test('the missing-account warning walks the deep self-review row too (items 2/21)', () => {
    const deepReview = { route: { kind: ROUTE_KIND_SESSION, target_id: 'codex=gpt-5.6-sol', profile_id: 'gone' }, effort: '', subagent_id: '' };
    const warning = pinnedAccountWarning({ deepReview, profilesByHarness: { codex: ['koshak'] }, accountsKnown: true });
    assert.match(warning, /A review row is\s+pinned to an account the agent service no longer lists \(codex · gone\)/);
    // A present account, a reference row and an api row raise nothing.
    assert.equal(pinnedAccountWarning({ deepReview, profilesByHarness: { codex: ['gone'] }, accountsKnown: true }), '');
    assert.equal(pinnedAccountWarning({ deepReview: { subagent_id: 'deep' }, profilesByHarness: {}, accountsKnown: true }), '');
    assert.equal(pinnedAccountWarning({ deepReview: { route: { kind: ROUTE_KIND_API, target_id: 'openai/x' } }, profilesByHarness: {}, accountsKnown: true }), '');
    // An unread accounts facet licenses no claim.
    assert.equal(pinnedAccountWarning({ deepReview, profilesByHarness: {}, accountsKnown: false }), '');
});
