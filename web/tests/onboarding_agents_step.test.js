// The first-run Agents step (phase 3C). What is asserted here is what the
// owner is PROMISED: the ladder's honesty, the rotation artwork's inertness,
// what zero / one / several connected accounts declare to the completion
// endpoint, and that the step holds nothing open after it is disposed.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createClaudexorStatusStore, familyLabel } from '../modules/claudexor_status_store.js';
import {
    AGENT_FAMILIES,
    LADDER_FOOTNOTE,
    MALFORMED_RECEIPT_CODE,
    VALUE_LADDER,
    agentsOutcomeText,
    agentsStepHtml,
    completionFailureNotice,
    connectedHarnesses,
    createAgentsStep,
    familyListHtml,
    familyStatusText,
    ladderHtml,
    onboardingSettingsDraft,
    readCompletionAnswer,
    rotationDiagramSvg,
    subscriptionDeclaration,
    subagentPreviewStatusSignature,
} from '../modules/onboarding_agents_step.js';

const json = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body });
const flush = async () => { for (let i = 0; i < 40; i += 1) await Promise.resolve(); };

function snapshotWith(harnesses) {
    // Shaped like the producer's own answer. `quota` is UNCONDITIONAL there —
    // `_status_payload` sets daemon/harnesses/profiles/quota before it reaches
    // the daemon at all — and the shared store requires all four before it will
    // derive a facet from a 2xx body (a 200 carrying an unrelated object used to
    // sail through as an authoritative empty world). A fixture missing one of
    // them is not a legacy wire; it is a body the real endpoint never sends.
    return {
        daemon: { state: 'running' },
        reads: { catalog: 'ok', accounts: 'ok', quota: 'ok' },
        harnesses: [{ id: 'claude' }, { id: 'codex' }, { id: 'cursor' }, { id: 'agy' }],
        profiles: {
            harnessAccounts: harnesses.map((harness) => ({
                harness_id: harness, native_login_detected: true,
            })),
            profiles: [],
        },
        quota: [],
    };
}

// ---------------------------------------------------------------------------
// The ladder.
// ---------------------------------------------------------------------------

test('the ladder is three rungs and states the launch gate honestly', () => {
    assert.equal(VALUE_LADDER.length, 3);

    const [runs, better, best] = VALUE_LADDER;
    // Rung 1: the access step ALREADY satisfied the requirement.
    assert.match(runs.title, /API key/i);
    assert.match(runs.body, /Ouroboros runs/i);
    // Rung 2: the benefit and the D-1 limit in the same breath — a plan moves
    // delegated work and configured review rows, and CANNOT run the main agent.
    assert.match(better.body, /delegated subagents/i);
    assert.match(better.body, /commit, plan, and skill review and task acceptance/i);
    assert.match(better.body, /main\s+agent keeps using the API key or local model/i);
    assert.match(better.body, /a plan cannot run it/i);
    // Rung 3: rotation, in the owner's own terms.
    assert.match(best.body, /rotate/i);
    assert.match(best.body, /window is spent/i);

    // No rung may imply a subscription is what starts Ouroboros.
    for (const rung of VALUE_LADDER) {
        assert.doesNotMatch(rung.body, /(subscription|plan) (alone )?(is enough|starts Ouroboros)/i);
    }
});

test('the footnote refuses both easy lies: "free", and "every reviewer moves"', () => {
    assert.match(LADDER_FOOTNOTE, /not free/i);
    assert.match(LADDER_FOOTNOTE, /already\s+pay for/i);
    // Owner R2 (2026-09-01): task acceptance follows the triad rows too — the
    // footnote states the RULE (what is routed moves), never "everything moves".
    assert.match(LADDER_FOOTNOTE, /task acceptance each follow their configured\s+triad row/i);
    assert.match(LADDER_FOOTNOTE, /acceptance panel on the subscription/i);
    // R12: the migration disclosure carries the measured numbers, not adjectives.
    assert.match(LADDER_FOOTNOTE, /about 12 s and\s+\$0\.07 per model row per task/i);
    assert.match(LADDER_FOOTNOTE, /minutes of your window per task/i);
    assert.doesNotMatch(LADDER_FOOTNOTE, /stays on the API|API-only/i);
    assert.doesNotMatch(LADDER_FOOTNOTE, /all reviewers|every reviewer/i);
});

test('the step renders the ladder, one row per family, and the editable actor host', () => {
    const html = agentsStepHtml();
    const rows = familyListHtml(snapshotWith([]));

    for (const rung of VALUE_LADDER) assert.ok(html.includes(rung.title), rung.title);
    for (const family of AGENT_FAMILIES) {
        const label = familyLabel(family.harness, null, { catalogKnown: false });
        assert.ok(rows.includes(label), label);
        assert.ok(rows.includes(`data-agent-connect="${family.harness}"`), family.harness);
        assert.ok(rows.includes(`data-harness-identity="${family.harness}"`), family.harness);
    }
    assert.ok(html.includes('id="agents-login-host"'));
    assert.ok(html.includes('id="agents-outcome"'));
    assert.ok(html.includes('id="onboarding-available-subagents"'));
    // Continue stays non-blocking on this step; final completion validates the
    // canonical draft once it has been previewed.
    assert.doesNotMatch(html + rows, /\srequired(?:\s|>)/);
});

// ---------------------------------------------------------------------------
// The rotation artwork.
// ---------------------------------------------------------------------------

test('the rotation diagram is inert artwork: no script, no animation, aria-hidden', () => {
    const svg = rotationDiagramSvg();

    assert.match(svg, /aria-hidden="true"/);
    assert.match(svg, /focusable="false"/);
    assert.match(svg, /role="presentation"/);
    assert.doesNotMatch(svg, /<script|<foreignObject|<animate|<set\b/i);
    // No event handlers and no external references of any kind.
    assert.doesNotMatch(svg, /\son[a-z]+=/i);
    assert.doesNotMatch(svg, /https?:\/\//);
    // Colour and size come from CSS classes so the figure inherits the theme;
    // the only url() is the local arrow marker.
    assert.doesNotMatch(svg, /\sfill="(?!none)/);
    assert.doesNotMatch(svg, /\sstroke="/);
    assert.doesNotMatch(svg, /font-size=/);

    // The three things it must draw for the loop to read at a glance.
    assert.match(svg, /API key or local model/);
    assert.match(svg, /runs the main agent/);
    assert.match(svg, /Agent plans/);
    assert.match(svg, /one window spent/);
    assert.match(svg, /the next takes over/);
});

test('the ladder text survives on its own — the artwork carries no unique fact', () => {
    const html = ladderHtml();
    for (const rung of VALUE_LADDER) assert.ok(html.includes(rung.title), rung.title);
    // Everything the figure says is also in the prose beside it, which is what
    // the short-viewport rule keeps when it drops the figure.
    assert.match(html, /rotate/i);
    assert.match(html, /window is spent/i);
});

// ---------------------------------------------------------------------------
// Zero / one / several connected accounts.
// ---------------------------------------------------------------------------

test('nothing connected: no declaration, and the outcome says so plainly', () => {
    const snapshot = snapshotWith([]);
    assert.deepEqual(connectedHarnesses(snapshot), []);
    assert.deepEqual(subscriptionDeclaration({ connected: [] }), {
        subscriptionsConnected: false, skipSubscriptionPresets: false,
    });
    const text = agentsOutcomeText([]);
    assert.match(text, /No agent account connected/i);
    assert.match(text, /Settings → Agents/);
});

test('one connected account declares the preset request and promises nothing certain', () => {
    const snapshot = snapshotWith(['claude']);
    assert.deepEqual(connectedHarnesses(snapshot), ['claude']);
    assert.deepEqual(subscriptionDeclaration({ connected: ['claude'] }), {
        subscriptionsConnected: true, skipSubscriptionPresets: false,
    });

    const text = agentsOutcomeText(['claude']);
    assert.match(text, /Claude Code is connected/);
    assert.match(text, /commit, scope, advisory, plan, skill review, and task acceptance/);
    assert.match(text, /Available subagents/);
    // Conditional by construction: the compiler may still refuse a seat.
    assert.match(text, /will try to/);
    assert.match(text, /nothing is changed/);
    assert.doesNotMatch(text, /guarantee|always/i);
});

test('an Antigravity-only setup promises task actors but no reviewer migration', () => {
    const snapshot = snapshotWith(['agy']);
    assert.deepEqual(connectedHarnesses(snapshot), ['agy']);
    const text = agentsOutcomeText(['agy'], { snapshot });
    assert.match(text, /Antigravity is connected/);
    assert.match(text, /subscription-backed choices to Available subagents/);
    assert.match(text, /task-only/);
    assert.match(text, /does not change reviewer routes/);
    assert.doesNotMatch(text, /move commit review|scope pass|advisory pre-review/);
});

test('a mixed reviewer-capable and task-only setup names each capability separately', () => {
    const snapshot = snapshotWith(['codex', 'agy']);
    const text = agentsOutcomeText(['codex', 'agy'], { snapshot });
    assert.match(text, /Codex and Antigravity are connected/);
    assert.match(text, /Codex can also move commit, scope, advisory, plan, skill review, and task acceptance/);
    assert.match(text, /Antigravity is task-only/);
    assert.doesNotMatch(text, /Antigravity can also move commit review/);
});

test('several accounts are named in family order and the rows say they rotate', () => {
    const snapshot = snapshotWith(['cursor', 'claude']);
    assert.deepEqual(connectedHarnesses(snapshot), ['claude', 'cursor']);
    assert.match(agentsOutcomeText(['claude', 'cursor']), /Claude Code and Cursor are connected/);

    // Two accounts in ONE family is the rotation case the owner asked about.
    const twoInOne = snapshotWith([]);
    twoInOne.profiles.profiles = [
        { profile: { harness_id: 'codex', profile_id: 'a', enabled: true }, status: { verification: 'passed' } },
        { profile: { harness_id: 'codex', profile_id: 'b', enabled: true }, status: { verification: 'passed' } },
    ];
    assert.deepEqual(familyStatusText(twoInOne, 'codex'), {
        tone: 'ok', text: '2 accounts connected · they rotate',
    });
    assert.deepEqual(familyStatusText(twoInOne, 'claude'), { tone: 'muted', text: 'Not connected' });
});

test('an all-disabled family is not connected — the aggregate reads the enabled fact', () => {
    // The owner switched every codex account OFF: rotation never takes them,
    // so the wizard must not declare the family connected under a header that
    // says the opposite (the same rule the Subagents section applies).
    const allOff = snapshotWith([]);
    allOff.profiles.profiles = [
        { profile: { harness_id: 'codex', profile_id: 'a', enabled: false }, status: { verification: 'passed' } },
        { profile: { harness_id: 'codex', profile_id: 'b', enabled: false }, status: { verification: 'passed' } },
    ];
    assert.deepEqual(connectedHarnesses(allOff), []);
    assert.deepEqual(familyStatusText(allOff, 'codex'), { tone: 'muted', text: 'Not connected' });

    // One row re-enabled: enabled+passed still connects and counts — and the
    // count excludes the disabled sibling instead of over-promising the pool.
    const oneOn = snapshotWith([]);
    oneOn.profiles.profiles = [
        { profile: { harness_id: 'codex', profile_id: 'a', enabled: false }, status: { verification: 'passed' } },
        { profile: { harness_id: 'codex', profile_id: 'b', enabled: true }, status: { verification: 'passed' } },
    ];
    assert.deepEqual(connectedHarnesses(oneOn), ['codex']);
    assert.deepEqual(familyStatusText(oneOn, 'codex'), { tone: 'ok', text: 'Connected' });

    // Absent stays connected: the fail-open default, exactly like the panel.
    const legacy = snapshotWith(['claude']);
    assert.deepEqual(connectedHarnesses(legacy), ['claude']);
});

test('a family the engine renames is spoken in the engine words, never as a raw id', () => {
    // The step used to keep its OWN map of three families and fall through to
    // the harness id, while the Agents tab preferred the engine's display_name.
    // Two authorities is how an owner ends up reading "claude" in a sentence.
    // Both now go through the store's `familyLabel`, so a renamed family — or a
    // fourth one the engine adds — reaches this text spelled properly.
    const renamed = snapshotWith(['claude']);
    renamed.harnesses = [{ id: 'claude', display_name: 'Claude Code Max' },
                         { id: 'codex' }, { id: 'cursor' }];
    const text = agentsOutcomeText(['claude'], { snapshot: renamed, catalogKnown: true });
    assert.match(text, /Claude Code Max is connected/);
    assert.doesNotMatch(text, /\bclaude\b/);

    // A family with no product name of its own is still never printed raw...
    const fourth = snapshotWith([]);
    fourth.harnesses = [{ id: 'gemini_cli', display_name: 'Gemini CLI' }];
    assert.match(agentsOutcomeText(['gemini_cli'], { snapshot: fourth, catalogKnown: true }),
                 /Gemini CLI is connected/);

    // A retained snapshot after a failed catalog read is useful for controls,
    // but its daemon label is no longer fresh evidence.
    assert.match(agentsOutcomeText(['claude'], { snapshot: renamed }), /Claude Code is connected/);
    assert.doesNotMatch(agentsOutcomeText(['claude'], { snapshot: renamed }), /Claude Code Max/);

    // ...and with no payload at all the bootstrap product names still apply,
    // which is exactly what every surface printed before the two merged.
    assert.match(agentsOutcomeText(['claude', 'cursor']), /Claude Code and Cursor are connected/);
});

test('an unread account facet claims nothing — a gap is not a zero', () => {
    const rows = familyListHtml(snapshotWith(['claude']), { accountsKnown: false });
    assert.ok(rows.includes('Not checked'));
    assert.doesNotMatch(rows, /Not connected/);
    assert.match(agentsOutcomeText([], { accountsKnown: false }), /could not be checked/i);
});

test('the owner skip produces a declaration that asks for NO preset', () => {
    assert.deepEqual(subscriptionDeclaration({ connected: ['claude', 'codex'], skipPresets: true }), {
        subscriptionsConnected: true, skipSubscriptionPresets: true,
    });
    const text = agentsOutcomeText(['claude'], { skipPresets: true });
    assert.match(text, /skip the automatic subscription preset/i);
    assert.match(text, /saved exactly as you edit it/i);
    assert.match(text, /Available subagents draft/i);
    assert.doesNotMatch(text, /subagents stay on your API access/i);
});

test('preview and completion can share one open provider/local/model draft', () => {
    const draft = onboardingSettingsDraft({
        state: {
            openaiKey: ' key ', totalBudget: '25', reviewEnforcement: 'blocking',
            localSource: '', localRoutingMode: 'cloud', mainModel: 'openai/gpt-5.6-sol',
            runtimeMode: 'advanced',
        },
        providerFields: [{ settingKey: 'OPENAI_API_KEY', stateKey: 'openaiKey' }],
        budgetFields: [{ settingKey: 'OUROBOROS_TOTAL_BUDGET_USD', stateKey: 'totalBudget' }],
        modelSlots: [{ settingKey: 'OUROBOROS_MODEL', stateKey: 'mainModel' }],
        trim: (value) => String(value || '').trim(),
    });
    assert.equal(draft.OPENAI_API_KEY, 'key');
    assert.equal(draft.OUROBOROS_TOTAL_BUDGET_USD, 25);
    assert.equal(draft.OUROBOROS_MODEL, 'openai/gpt-5.6-sol');
    assert.equal(draft.OUROBOROS_REVIEW_ENFORCEMENT, 'blocking');
    assert.equal(draft.OUROBOROS_RUNTIME_MODE, 'advanced');
    assert.equal('OUROBOROS_MODEL_HEAVY' in draft, false);
});

test('a late status settle invalidates preview on model/account facts, not timestamps', () => {
    const base = snapshotWith(['codex']);
    base.harnesses = [{ id: 'codex', models: [{ id: 'model-a' }] }];
    const view = { reads: { catalog: 'ok', accounts: 'ok' } };
    const first = subagentPreviewStatusSignature(view, base);

    const modelChanged = structuredClone(base);
    modelChanged.harnesses[0].models = [{ id: 'model-b' }];
    assert.notEqual(subagentPreviewStatusSignature(view, modelChanged), first);

    const timestampOnly = structuredClone(base);
    timestampOnly.profiles.harnessAccounts[0].last_verified_at = '2099-01-01T00:00:00Z';
    assert.equal(subagentPreviewStatusSignature(view, timestampOnly), first);
});

// ---------------------------------------------------------------------------
// A typed completion failure.
// ---------------------------------------------------------------------------

test('a typed refusal keeps its real reason and offers the escape it was given', () => {
    const error = new Error('The agent accounts were connected, but their models could not be verified right now, so nothing was saved.');
    error.code = 'daemon_unavailable';
    error.detail = 'The agent engine is unreachable (connect_failed: boom)';
    error.canSkip = true;

    const notice = completionFailureNotice(error);
    assert.equal(notice.code, 'daemon_unavailable');
    assert.equal(notice.canSkip, true);
    // BOTH halves reach the owner: the constant sentence AND the engine's own.
    assert.match(notice.text, /could not be verified/);
    assert.match(notice.text, /connect_failed: boom/);
});

test('an untyped failure is not dressed up as a skippable preset problem', () => {
    const notice = completionFailureNotice(new Error('HTTP 500'));
    assert.equal(notice.canSkip, false);
    assert.equal(notice.saved, false);
    assert.equal(notice.text, 'HTTP 500');
});

test('a 503 settings_save_timeout is read as UNKNOWN, never as "nothing was saved"', () => {
    // The shared bounded writer answers `saved: null` when the save body outlives
    // its bound: the bytes may already be on disk. Collapsing null to false used
    // to offer the skip (a second write) over a transaction that may have landed.
    const read = readCompletionAnswer({
        status: 503, ok: false, parsed: true,
        data: { error: 'the settings save is still running in the server', code: 'settings_save_timeout', saved: null, can_skip: true },
    });
    assert.equal(read.failure.saved, null);
    const notice = completionFailureNotice(read.failure);
    assert.equal(notice.saved, null);
    assert.equal(notice.saveUnknown, true);
    assert.equal(notice.canSkip, false);
    assert.match(notice.text, /still running in the server/);
    assert.match(notice.text, /unknown/);
    assert.match(notice.text, /Check status/);
    assert.doesNotMatch(notice.text, /WERE written/);
    // The two boolean states are untouched by the third.
    assert.equal(readCompletionAnswer({ status: 503, ok: false, parsed: true, data: { saved: false } }).failure.saved, false);
    assert.equal(readCompletionAnswer({ status: 500, ok: false, parsed: true, data: { saved: true } }).failure.saved, true);
});

test('a failure AFTER the bytes reached disk never claims nothing was saved', () => {
    // The endpoint distinguishes a refusal (nothing persisted) from a failure
    // in a post-commit stage. Reporting the second as "nothing was saved" would
    // repeat, one layer up, the exact dishonesty the atomic write removed — and
    // would send the owner back to re-enter settings that already exist.
    const error = new Error('Onboarding completion failed.');
    error.saved = true;
    error.stage = 'supervisor_start';
    error.canSkip = true;

    const notice = completionFailureNotice(error);
    assert.equal(notice.saved, true);
    assert.match(notice.text, /settings WERE written/i);
    assert.match(notice.text, /supervisor_start/);
    assert.doesNotMatch(notice.text, /nothing was saved/i);
    // And the escape hatch is withdrawn: with bytes on disk, "finish without
    // agent defaults" would be a SECOND write, not an alternative to the first.
    assert.equal(notice.canSkip, false);

    // The `stage` above was hand-built, so it proved the PROSE and not the
    // reader. This is the envelope `post_commit_failure_response` really sends
    // — the field is named `post_commit_failed`, and the reader used to look
    // for `stage`, so a genuine post-commit failure reached the owner with the
    // one word identifying the failed step silently blanked.
    const real = readCompletionAnswer({
        status: 500,
        ok: false,
        parsed: true,
        data: {
            error: 'Settings were saved to disk, but the supervisor start step failed afterwards: RuntimeError: boom',
            status: 'saved_with_post_commit_error',
            saved: true,
            post_commit_failed: 'supervisor start',
        },
    });
    assert.equal(real.failure.stage, 'supervisor start');
    assert.equal(real.failure.saved, true);
    assert.match(completionFailureNotice(real.failure).text, /supervisor start/);
});

// ---------------------------------------------------------------------------
// Reading the completion answer.
// ---------------------------------------------------------------------------

test('a 2xx without the success envelope is a failure, not a completion', () => {
    // Everything downstream reads this body: the saved runtime mode and whether
    // it needs a restart. A shape-blind `ok` announced a finished setup while
    // silently discarding both — and an unparseable body used to become `{}`,
    // which is truthy.
    const bad = [
        { status: 200, ok: true, parsed: false, data: null },                  // HTML / empty
        { status: 200, ok: true, parsed: true, data: {} },                     // no envelope
        { status: 200, ok: true, parsed: true, data: { ok: false } },          // explicit failure
        { status: 200, ok: true, parsed: true, data: { ok: true } },           // no receipt fields
        { status: 200, ok: true, parsed: true, data: { ok: true, runtime_mode: 'pro' } },
        { status: 200, ok: true, parsed: true, data: { ok: true, restart_required: true } },
    ];
    for (const answer of bad) {
        const read = readCompletionAnswer(answer);
        assert.ok(read.failure, JSON.stringify(answer));
        assert.equal(read.failure.code, MALFORMED_RECEIPT_CODE);
        assert.equal(read.failure.canSkip, false);
        assert.match(read.failure.message, /not confirmed/i);
    }

    const good = readCompletionAnswer({
        status: 200, ok: true, parsed: true,
        data: { ok: true, status: 'saved', runtime_mode: 'pro', restart_required: true, preset: {} },
    });
    assert.ok(good.receipt);
    assert.equal(good.receipt.restart_required, true);
    assert.equal(good.receipt.runtime_mode, 'pro');
});

test('a typed refusal keeps every field the wizard renders', () => {
    const read = readCompletionAnswer({
        status: 503, ok: false, parsed: true,
        data: {
            error: 'models could not be verified', code: 'daemon_unavailable',
            detail: 'engine unreachable', can_skip: true, saved: false,
        },
    });
    assert.deepEqual(read.failure, {
        message: 'models could not be verified', status: 503, code: 'daemon_unavailable',
        detail: 'engine unreachable', canSkip: true, saved: false, stage: '',
    });
});

// ---------------------------------------------------------------------------
// The controller: it reads the SHARED store, and releases everything.
// ---------------------------------------------------------------------------

function fakeDom() {
    const documentListeners = [];
    const windowListeners = [];
    const nodes = new Map();
    const make = (id) => {
        const node = {
            id,
            innerHTML: '',
            textContent: '',
            hidden: false,
            dataset: {},
            contains: () => false,
            querySelector: () => null,
            querySelectorAll: (selector) => (
                node.id === 'agents-family-list' && selector === '[data-agent-connect]'
                    ? node.buttons
                    : []
            ),
            buttons: [],
        };
        return node;
    };
    for (const id of [
        'agents-family-list', 'agents-status-note', 'agents-outcome',
        'agents-login-host', 'onboarding-available-subagents',
    ]) {
        nodes.set(id, make(id));
    }
    const defaultView = {
        addEventListener: (type, fn) => windowListeners.push([type, fn]),
        removeEventListener: (type, fn) => {
            const idx = windowListeners.findIndex(([t, f]) => t === type && f === fn);
            if (idx >= 0) windowListeners.splice(idx, 1);
        },
    };
    return {
        nodes,
        documentListeners,
        windowListeners,
        doc: {
            hidden: false,
            activeElement: null,
            defaultView,
            getElementById: (id) => nodes.get(id) || null,
            addEventListener: (type, fn) => documentListeners.push([type, fn]),
            removeEventListener: (type, fn) => {
                const idx = documentListeners.findIndex(([t, f]) => t === type && f === fn);
                if (idx >= 0) documentListeners.splice(idx, 1);
            },
        },
    };
}

test('the step reads the shared store — it never fetches the status endpoint itself', async () => {
    const urls = [];
    const store = createClaudexorStatusStore({
        fetchImpl: async (url) => { urls.push(url); return json(200, snapshotWith(['codex'])); },
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const seen = [];
    const step = createAgentsStep({ doc: dom.doc, store, onChange: (c) => seen.push(c) });

    step.mount();
    await flush();

    // ONE read, through the store's own endpoint — no second reader.
    assert.deepEqual(urls, ['/api/claudexor/status']);
    assert.deepEqual(step.connected, ['codex']);
    assert.deepEqual(seen, [['codex']]);
    assert.deepEqual(step.declaration(), {
        subscriptionsConnected: true, skipSubscriptionPresets: false,
    });
    assert.ok(dom.nodes.get('agents-family-list').innerHTML.includes('Codex'));
    assert.match(dom.nodes.get('agents-outcome').textContent, /Codex is connected/);

    assert.equal(await step.dispose(), 'released');
    step.detach();
    assert.equal(store.subscriberCount, 0);
    assert.equal(dom.documentListeners.length, 0, 'pagehide must never bind to Document');
    assert.equal(dom.windowListeners.length, 0, 'the step must leave no Window listener behind');
    store.dispose();
});

test('the generated onboarding draft shows a credentialed one-harness API scout and stays editable', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, snapshotWith(['codex'])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const previews = [];
    const step = createAgentsStep({
        doc: dom.doc,
        store,
        previewPayload: () => ({ OPENAI_API_KEY: 'MASKED', OUROBOROS_MODEL: 'openai/gpt-5.6-sol' }),
        previewTransport: async (payload) => {
            previews.push(payload);
            return {
                source: 'onboarding_default',
                diagnostics: [],
                available_subagents: {
                    enabled: true,
                    items: [
                        {
                            subagent_id: 'api_fast_scout',
                            name: 'Fast API scout',
                            recommended_use: 'Fast independent research before implementation.',
                            route: { kind: 'api_model', target_id: 'openai/gpt-5.6-luna' },
                            effort: 'high',
                        },
                        {
                            subagent_id: 'codex_builder',
                            name: 'Codex builder',
                            recommended_use: 'Workspace implementation.',
                            route: { kind: 'agent_session', target_id: 'codex=gpt-5.6-sol-high' },
                        },
                    ],
                },
            };
        },
    });
    step.mount();
    await flush();
    await flush();

    assert.equal(previews.length >= 1, true);
    assert.equal(previews.at(-1).subscriptionsConnected, true);
    assert.equal(step.availableSubagents.items[0].subagent_id, 'api_fast_scout');
    assert.match(dom.nodes.get('onboarding-available-subagents').innerHTML, /Subagent 1/);
    assert.match(dom.nodes.get('onboarding-available-subagents').innerHTML, /Fast independent research before implementation/);
    assert.match(dom.nodes.get('onboarding-available-subagents').innerHTML, /Subagent 2/);
    assert.match(dom.nodes.get('onboarding-available-subagents').innerHTML, /Workspace implementation/);
    assert.doesNotMatch(dom.nodes.get('onboarding-available-subagents').innerHTML, /data-subagent-field="(?:id|name)"/);
    assert.match(dom.nodes.get('onboarding-available-subagents').innerHTML, /Generated draft/);
    assert.deepEqual(step.validateSubagents(), []);

    step.detach();
    store.dispose();
});

test('a model change regenerates a clean onboarding draft and invalidates the older receipt', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, snapshotWith([])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    let model = 'openai/model-a';
    const step = createAgentsStep({
        doc: dom.doc,
        store,
        previewPayload: () => ({ OUROBOROS_MODEL: model }),
        previewTransport: async (payload) => ({
            source: 'onboarding_default',
            diagnostics: [],
            available_subagents: {
                enabled: true,
                items: [
                    {
                        subagent_id: 'main-builder',
                        name: 'Main builder',
                        recommended_use: 'Use the current main model.',
                        route: { kind: 'api_model', target_id: payload.OUROBOROS_MODEL },
                    },
                ],
            },
        }),
    });
    step.mount();
    await flush();
    assert.equal(step.availableSubagents.items[0].route.target_id, 'openai/model-a');
    assert.equal(step.generatedPreviewReady, true);

    model = 'openai/model-b';
    step.invalidateGeneratedPreview();
    assert.equal(step.generatedPreviewReady, false);
    assert.equal(await step.refreshSubagentsPreview({ force: true }), true);
    assert.equal(step.availableSubagents.items[0].route.target_id, 'openai/model-b');
    assert.equal(step.generatedPreviewReady, true);

    step.detach();
    store.dispose();
});

test('Connect starts the login through the shared card controller', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const calls = [];
    const fetchImpl = async (url, init) => {
        calls.push([String(url), init?.method || 'GET']);
        if (String(url).startsWith('/api/claudexor/login') && (init?.method || 'GET') === 'DELETE') {
            return json(200, { job: { state: 'cancelled' } });
        }
        if (String(url).startsWith('/api/claudexor/login')) {
            return json(200, { job_id: 'j1', job: { state: 'running' } });
        }
        return json(200, snapshotWith([]));
    };
    const store = createClaudexorStatusStore({
        fetchImpl,
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const list = dom.nodes.get('agents-family-list');
    const handlers = [];
    list.buttons = [{
        getAttribute: () => 'claude',
        addEventListener: (_type, fn) => handlers.push(fn),
    }];

    const step = createAgentsStep({ doc: dom.doc, store, fetchImpl });
    step.mount();
    await flush();

    assert.ok(handlers.length >= 1, 'every family row wires its own Connect');
    handlers[handlers.length - 1]();
    await flush();

    assert.ok(calls.some(([url, method]) => url === '/api/claudexor/login' && method === 'POST'));
    // The login card renders into the step's own host, never a second surface.
    assert.match(dom.nodes.get('agents-login-host').innerHTML, /harness-login-card/);
    await step.dispose();
    step.detach();
    store.dispose();
});

test('unknown sign-in cleanup stays retryable until explicit local detach', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let cancels = 0;
    const fetchImpl = async (url, init) => {
        const u = String(url);
        const method = init?.method || 'GET';
        if (u.startsWith('/api/claudexor/login') && method === 'POST') {
            return json(200, { job_id: 'j1', job: { state: 'running' } });
        }
        if (u.startsWith('/api/claudexor/login') && method === 'DELETE') {
            cancels += 1;
            return json(503, { error: 'daemon unreachable' });   // never proven gone
        }
        if (u.startsWith('/api/claudexor/login')) return json(200, { job: { state: 'running' } });
        return json(200, snapshotWith([]));
    };
    const store = createClaudexorStatusStore({
        fetchImpl,
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const handlers = [];
    dom.nodes.get('agents-family-list').buttons = [{
        getAttribute: () => 'claude',
        addEventListener: (_type, fn) => handlers.push(fn),
    }];

    const step = createAgentsStep({ doc: dom.doc, store, fetchImpl });
    step.mount();
    await flush();
    handlers[handlers.length - 1]();
    await flush();

    assert.equal(await step.dispose(), 'unknown');
    assert.ok(cancels >= 1, 'the disposer must actually attempt the cancel');

    // Unknown transport remains retryable against the same attached job.
    const before = cancels;
    assert.equal(await step.dispose(), 'unknown');
    assert.ok(cancels > before, 'unknown cleanup must stay retryable');

    step.detach();
    const detachedAt = cancels;
    assert.equal(await step.dispose(), 'unknown', 'detach must not fabricate release proof');
    assert.equal(cancels, detachedAt, 'local detach initiates no further cancel');

    store.dispose();
});

test('terminal-unconfirmed cleanup is retained without repeating cancel', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    let cancels = 0;
    const fetchImpl = async (url, init) => {
        const u = String(url);
        const method = init?.method || 'GET';
        if (u === '/api/claudexor/login' && method === 'POST') {
            return json(200, { job_id: 'j1', job: { state: 'running' } });
        }
        if (u.startsWith('/api/claudexor/login') && method === 'DELETE') {
            cancels += 1;
            return json(200, {
                job: { state: 'failed', outcome: { reason: 'termination_unconfirmed' } },
            });
        }
        if (u.startsWith('/api/claudexor/login')) return json(200, { job: { state: 'running' } });
        return json(200, snapshotWith([]));
    };
    const store = createClaudexorStatusStore({
        fetchImpl,
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const handlers = [];
    dom.nodes.get('agents-family-list').buttons = [{
        getAttribute: () => 'claude',
        addEventListener: (_type, fn) => handlers.push(fn),
    }];
    const step = createAgentsStep({ doc: dom.doc, store, fetchImpl });
    step.mount();
    await flush();
    handlers[handlers.length - 1]();
    await flush();

    assert.equal(await step.dispose(), 'retained');
    assert.equal(cancels, 1);
    assert.equal(await step.dispose(), 'retained');
    assert.equal(cancels, 1, 'known retained custody must not repeat a pointless cancel');
    step.detach();

    store.dispose();
});

test('pagehide binds to Window, preserves bfcache, and detaches late work synchronously', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const calls = [];
    let resolveCreate;
    const createResponse = new Promise((resolve) => { resolveCreate = resolve; });
    const fetchImpl = async (url, init) => {
        const method = init?.method || 'GET';
        calls.push([String(url), method]);
        if (String(url) === '/api/claudexor/login' && method === 'POST') return createResponse;
        if (String(url).startsWith('/api/claudexor/login')) {
            return json(200, { job: { state: 'running' } });
        }
        return json(200, snapshotWith([]));
    };
    const store = createClaudexorStatusStore({
        fetchImpl,
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const handlers = [];
    dom.nodes.get('agents-family-list').buttons = [{
        getAttribute: () => 'claude',
        addEventListener: (_type, fn) => handlers.push(fn),
    }];
    const step = createAgentsStep({ doc: dom.doc, store, fetchImpl });
    step.mount();
    await flush();

    assert.equal(dom.documentListeners.some(([event]) => event === 'pagehide'), false,
        'Document is not the pagehide target');
    assert.equal(dom.windowListeners.length, 3,
        'pagehide plus the shared status surface activation hooks are bounded');
    const [type, onPageHide] = dom.windowListeners[0];
    assert.equal(type, 'pagehide');
    handlers[handlers.length - 1]();
    await flush();

    onPageHide({ persisted: true });
    assert.equal(dom.windowListeners.length, 3, 'bfcache keeps the live step mounted');

    const lifecycleBefore = calls.filter(([url]) => url.startsWith('/api/claudexor/login')).length;
    onPageHide({ persisted: false });
    assert.equal(dom.nodes.get('agents-login-host').innerHTML, '', 'detach clears the card synchronously');
    assert.equal(dom.windowListeners.length, 0, 'detach removes the exact captured Window listener');
    assert.equal(
        calls.filter(([url]) => url.startsWith('/api/claudexor/login')).length,
        lifecycleBefore,
        'departure initiates no create, cancel, or reconcile request',
    );
    handlers[handlers.length - 1]();
    await flush();
    assert.equal(
        calls.filter(([url]) => url.startsWith('/api/claudexor/login')).length,
        lifecycleBefore,
        'an old Connect handler cannot recreate login work after detach',
    );

    resolveCreate(json(200, { job_id: 'j1', job: { state: 'running' } }));
    await flush();
    assert.equal(dom.nodes.get('agents-login-host').innerHTML, '', 'late create cannot repaint after detach');
    assert.equal(
        calls.filter(([url, method]) => url.startsWith('/api/claudexor/login') && method === 'GET').length,
        0,
        'late create cannot arm polling after detach',
    );
    store.dispose();
});

test('the skip choice is reflected in the outcome the owner reads before finishing', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, snapshotWith(['claude'])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const step = createAgentsStep({ doc: dom.doc, store });
    step.mount();
    await flush();

    await step.setSkipPresets(true);
    assert.match(dom.nodes.get('agents-outcome').textContent, /skip the automatic subscription preset/i);
    assert.deepEqual(step.declaration(), {
        subscriptionsConnected: true, skipSubscriptionPresets: true,
    });
    step.detach();
    store.dispose();
});

test('the skip choice refreshes a failed subscription preview before completion', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, snapshotWith(['claude'])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const previews = [];
    const step = createAgentsStep({
        doc: dom.doc,
        store,
        previewTransport: async (payload) => {
            previews.push(payload);
            if (!payload.skipSubscriptionPresets) throw new Error('subscription preview unavailable');
            return {
                source: 'api_default',
                diagnostics: [],
                available_subagents: {
                    enabled: true,
                    items: [{
                        subagent_id: 'api-scout',
                        name: 'API scout',
                        recommended_use: 'Use when subscription presets are skipped.',
                        route: { kind: 'api_model', target_id: 'openai/gpt-5.6-luna' },
                    }],
                },
            };
        },
    });
    step.mount();
    await flush();
    assert.equal(step.generatedPreviewReady, false);

    assert.equal(await step.setSkipPresets(true), true);
    assert.equal(previews.at(-1).skipSubscriptionPresets, true);
    assert.equal(step.generatedPreviewReady, true);
    assert.equal(step.availableSubagents.items[0].subagent_id, 'api-scout');

    step.detach();
    store.dispose();
});
