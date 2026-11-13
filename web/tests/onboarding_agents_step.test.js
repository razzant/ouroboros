// The first-run Agents step (phase 3C). What is asserted here is what the
// owner is PROMISED: the ladder's honesty, the rotation artwork's inertness,
// what zero / one / several connected accounts declare to the completion
// endpoint, and that the step holds nothing open after it is disposed.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { createClaudexorStatusStore, familyLabel } from '../modules/claudexor_status_store.js';
import { compatibleEndpoint, configuredApiProviders } from '../modules/route_editor_primitives.js';
import {
    AGENT_FAMILIES,
    LADDER_FOOTNOTE,
    MALFORMED_RECEIPT_CODE,
    VALUE_LADDER,
    accessRowState,
    agentsOnlyAccess,
    agentsOutcomeText,
    agentsStepHtml,
    completionFailureNotice,
    connectedHarnesses,
    createAgentsStep,
    familyListHtml,
    familyStatusText,
    ladderHtml,
    mainSourceReason,
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
        // The unified roster: a connected family is a verified named row (the
        // engine's migrated default login), with `enabled` absent (fail-open).
        profiles: {
            profiles: harnesses.map((harness) => ({
                profile: { harness_id: harness, profile_id: `${harness}-default` },
                status: { verification: 'passed', verification_source: 'vendor' },
            })),
            harnessAccounts: [],
            accountPools: [],
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
    assert.match(runs.title, /Codex subscription/i);
    assert.match(runs.body, /models and agents without an API key/i);
    assert.match(better.body, /API keys and local models work on their own/i);
    assert.match(better.body, /Claude Code, Cursor, and Antigravity remain agent connections/i);
    // Rung 3: rotation, in the owner's own terms.
    assert.match(best.body, /rotate/i);
    assert.match(best.body, /window is spent/i);

    // Access is subscription-backed, not a promise of unlimited free service.
    for (const rung of VALUE_LADDER) {
        assert.doesNotMatch(rung.body, /guaranteed free|unlimited/i);
    }
});

test('the footnote refuses both easy lies: "free", and "every reviewer moves"', () => {
    assert.match(LADDER_FOOTNOTE, /not free/i);
    assert.match(LADDER_FOOTNOTE, /already\s+pay for/i);
    // Owner R2 (2026-09-01): a task's result review runs on the review pool too — the
    // footnote states the RULE (what is routed moves), never "everything moves".
    assert.match(LADDER_FOOTNOTE, /result reviews all take their reviewers from the review pool\s+\(the rows marked Reviewer\)/i);
    assert.doesNotMatch(LADDER_FOOTNOTE, /triad/i);
    assert.match(LADDER_FOOTNOTE, /puts each review on the subscription/i);
    // R12: the disclosure points at measured prices, never adjectives or a stale constant; the cost
    // follows the reviewers asked, never every pool row.
    assert.match(LADDER_FOOTNOTE, /Settings → Agents shows what one review by each API row actually cost/i);
    assert.match(LADDER_FOOTNOTE, /a wave costs the reviewers it asks/i);
    assert.doesNotMatch(LADDER_FOOTNOTE, /\$\d/);
    assert.doesNotMatch(LADDER_FOOTNOTE, /per model row|all run on the review pool/i);
    assert.match(LADDER_FOOTNOTE, /minutes of your window per review/i);
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

test('nothing connected: no declaration, and no outcome sentence repeating the rows and the reason beside Continue', () => {
    const snapshot = snapshotWith([]);
    assert.deepEqual(connectedHarnesses(snapshot), []);
    assert.deepEqual(subscriptionDeclaration({ connected: [] }), {
        subscriptionsConnected: false, skipSubscriptionPresets: false,
    });
    assert.equal(agentsOutcomeText([]), '');
});

test('agent-only subscriptions are told what they run and what Main needs; an unread or serving source is not', () => {
    const codex = [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }];
    const blank = mainSourceReason([], codex);
    assert.match(blank, /^Connect Codex, enter an API key/);

    // The engine named its model sources and none of the connected accounts is one: a fact, with the way forward.
    for (const [connected, names] of [[['claude'], /^Claude Code can run agents; /], [['claude', 'cursor'], /^Claude Code and Cursor can run agents; /],
        [['claude', 'cursor', 'agy'], /^Claude Code, Cursor and Antigravity can run agents; /]]) {
        assert.equal(agentsOnlyAccess(connected, codex), true);
        const reason = mainSourceReason(connected, codex, { snapshot: snapshotWith(connected) });
        assert.match(reason, names);
        assert.match(reason, /Ouroboros itself still needs a model\. Connect Codex, add an API key, or choose a local model\./);
        assert.doesNotMatch(reason, /Retry|discovery|confirm/i, 'nothing here can be retried');
    }
    // The sources in the sentence are the engine's own names for them, and several are alternatives.
    const two = [{ id: 'managed', label: 'Managed models', credentialHarness: 'claude' }, { id: 'codex', label: 'Codex', credentialHarness: 'codex' }];
    assert.match(mainSourceReason(['cursor'], two), /Connect Managed models or Codex,/);

    // Unknown sources (an unread list) and a connected account that IS a source keep the retryable reason.
    for (const [connected, sources] of [[['claude'], []], [['claude'], undefined], [['codex'], codex], [['claude', 'codex'], codex]]) {
        assert.equal(agentsOnlyAccess(connected, sources), false);
        assert.match(mainSourceReason(connected, sources), /^Could not confirm that a connected account can run .*Retry model discovery,/);
    }
    assert.equal(agentsOnlyAccess([], codex), false, 'nothing connected is not an agents-only install');
});

test('the access row never offers what Continue refuses, and calls the API section Optional only when it is', () => {
    const codex = [{ id: 'codex', label: 'Codex', credentialHarness: 'codex' }];
    const row = (facts) => accessRowState({ sources: codex, ...facts });
    // A preview that failed while Continue is blocked asks for a Retry alone; with Continue open it offers both.
    assert.match(row({ connected: ['claude'], previewError: 'Down.', continueBlocked: true }).note, /prepared: Down\. Retry\.$/);
    assert.match(row({ connected: ['codex'], previewError: 'Down.' }).note, /Retry, or Continue to choose models yourself\.$/);
    // Agents-only accounts: no badge; nothing connected, an unread or a failed list: one of the ways forward the
    // reason names, so Optional; a known model account: Optional; entered access: Configured.
    assert.equal(row({ connected: [] }).badge, 'Optional');
    assert.equal(row({ connected: ['claude'] }).badge, '');
    assert.equal(row({ connected: ['claude'], sources: [], read: 'not_read' }).badge, 'Optional');
    assert.equal(row({ connected: ['codex'], sources: [], read: 'failed' }).badge, 'Optional');
    assert.equal(row({ connected: ['codex'] }).badge, 'Optional');
    assert.equal(row({ connected: ['claude'], configured: true }).badge, 'Configured');
    // Retry: hidden with nothing connected and after a healthy agents-only read; a failed preview brings it back.
    assert.equal(row({ connected: [] }).retryHidden, true);
    assert.equal(row({ connected: ['claude'] }).retryHidden, true);
    assert.equal(row({ connected: ['claude'], previewError: 'Down.' }).retryHidden, false);
    assert.equal(row({ connected: ['codex'] }).retryHidden, false, 'a healthy read can still be read again');
    // The read state in a person's words, never its internal name ("failed", "transport").
    for (const [read, words] of [['failed', 'could not be read'], ['transport', 'could not be read'],
        ['partial', 'is incomplete'], ['stale', 'is out of date'], ['not_read', 'is not available yet']]) {
        assert.equal(row({ connected: ['codex'], sources: [], read, readErrors: 'Offline.' }).note,
            `The model list ${words}. Offline. Your choices are kept.`, read);
    }
    assert.equal(row({ connected: ['codex'], checking: true }).note, 'Checking model sources…');
});

test('the limits warning stands under the rows in every state, before any connection', () => {
    const html = agentsStepHtml({ compact: true, showRoster: false });
    const [rows, warning, login] = ['id="agents-family-list"', 'Subscription limits still apply.', 'id="agents-login-host"'].map((part) => html.indexOf(part));
    assert.ok(rows >= 0 && rows < warning && warning < login, 'rows, then the warning, then any sign-in');
    assert.match(html, /does not enable paid credits or change your provider's spending limits\./);
    assert.doesNotMatch(html, /agent-limits-note"[^>]*\bhidden\b/);
});

test('one connected account declares the preset request and promises nothing certain', () => {
    const snapshot = snapshotWith(['claude']);
    assert.deepEqual(connectedHarnesses(snapshot), ['claude']);
    assert.deepEqual(subscriptionDeclaration({ connected: ['claude'] }), {
        subscriptionsConnected: true, skipSubscriptionPresets: false,
    });

    const text = agentsOutcomeText(['claude']);
    assert.match(text, /Signed in to Claude Code/);
    assert.match(text, /can also move commit, plan, skill and result review\./);
    assert.match(text, /Available subagents/);
    // Conditional by construction: the compiler may still refuse a seat.
    assert.match(text, /will try to/);
    // What Main still needs is said once, beside Continue, and only while true: this paragraph cannot know of a key.
    assert.doesNotMatch(text, /Main still needs|still needs a model/);
    assert.doesNotMatch(text, /guarantee|always/i);
});

test('an Antigravity-only setup promises task actors but no reviewer migration', () => {
    const snapshot = snapshotWith(['agy']);
    assert.deepEqual(connectedHarnesses(snapshot), ['agy']);
    const text = agentsOutcomeText(['agy'], { snapshot });
    assert.match(text, /Signed in to Antigravity/);
    assert.match(text, /subscription-backed choices to Available subagents/);
    assert.match(text, /task-only/);
    assert.match(text, /does not change reviewer routes/);
    assert.doesNotMatch(text, /move commit review|scope pass|advisory pre-review/);
});

test('a mixed reviewer-capable and task-only setup names each capability separately', () => {
    const snapshot = snapshotWith(['codex', 'agy']);
    const text = agentsOutcomeText(['codex', 'agy'], { snapshot });
    assert.match(text, /Signed in to Codex and Antigravity/);
    assert.match(text, /Codex can also move commit, plan, skill and result review\./);
    assert.match(text, /Antigravity is task-only/);
    assert.doesNotMatch(text, /Antigravity can also move commit review/);
});

test('several accounts are named in family order and the rows count their logins', () => {
    const snapshot = snapshotWith(['cursor', 'claude']);
    assert.deepEqual(connectedHarnesses(snapshot), ['claude', 'cursor']);
    assert.match(agentsOutcomeText(['claude', 'cursor']), /Signed in to Claude Code and Cursor/);

    // Two accounts in ONE family is the rotation case the owner asked about.
    const twoInOne = snapshotWith([]);
    twoInOne.profiles.profiles = [
        { profile: { harness_id: 'codex', profile_id: 'a', enabled: true }, status: { verification: 'passed' } },
        { profile: { harness_id: 'codex', profile_id: 'b', enabled: true }, status: { verification: 'passed' } },
    ];
    assert.deepEqual(familyStatusText(twoInOne, 'codex'), {
        tone: 'ok', text: '2 accounts signed in',
    });
    assert.deepEqual(familyStatusText(twoInOne, 'claude'), { tone: 'muted', text: 'No account connected' });
});

test('family sign-in counts stay independent of enabled-only preset admission', () => {
    const allOff = snapshotWith([]);
    allOff.profiles.profiles = [
        { profile: { harness_id: 'codex', profile_id: 'a', enabled: false }, status: { verification: 'passed' } },
        { profile: { harness_id: 'codex', profile_id: 'b', enabled: false }, status: { verification: 'passed' } },
    ];
    assert.deepEqual(connectedHarnesses(allOff), []);
    assert.deepEqual(familyStatusText(allOff, 'codex'), { tone: 'muted', text: '2 accounts signed in · all disabled' });
    assert.deepEqual(subscriptionDeclaration({ connected: connectedHarnesses(allOff) }), {
        subscriptionsConnected: false, skipSubscriptionPresets: false,
    });
    assert.match(familyListHtml(allOff), /2 accounts signed in · all disabled/);

    // Re-enabling changes admission, while both passed logins remain counted.
    const oneOn = snapshotWith([]);
    oneOn.profiles.profiles = [
        { profile: { harness_id: 'codex', profile_id: 'a', enabled: false }, status: { verification: 'passed' } },
        { profile: { harness_id: 'codex', profile_id: 'b', enabled: true }, status: { verification: 'passed' } },
    ];
    assert.deepEqual(connectedHarnesses(oneOn), ['codex']);
    assert.deepEqual(familyStatusText(oneOn, 'codex'), { tone: 'ok', text: '2 accounts signed in' });
    assert.match(familyListHtml(oneOn), /2 accounts signed in/);
    const unknown = snapshotWith([]);
    unknown.profiles.profiles = [{ profile: { harness_id: 'codex', profile_id: 'a', enabled: true },
        status: { verification: 'not_run', availability: 'unknown' } }];
    assert.deepEqual(familyStatusText(unknown, 'codex'), { tone: 'muted', text: '1 account · not verified' });

    // Absent stays connected: the fail-open default, exactly like the panel.
    const absent = snapshotWith(['claude']);
    assert.deepEqual(connectedHarnesses(absent), ['claude']);
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
    assert.match(text, /Signed in to Claude Code Max/);
    assert.doesNotMatch(text, /\bclaude\b/);

    // A family with no product name of its own is still never printed raw...
    const fourth = snapshotWith([]);
    fourth.harnesses = [{ id: 'gemini_cli', display_name: 'Gemini CLI' }];
    assert.match(agentsOutcomeText(['gemini_cli'], { snapshot: fourth, catalogKnown: true }),
                 /Signed in to Gemini CLI/);

    // A retained snapshot after a failed catalog read is useful for controls,
    // but its daemon label is no longer fresh evidence.
    assert.match(agentsOutcomeText(['claude'], { snapshot: renamed }), /Signed in to Claude Code/);
    assert.doesNotMatch(agentsOutcomeText(['claude'], { snapshot: renamed }), /Claude Code Max/);

    // ...and with no payload at all the bootstrap product names still apply,
    // which is exactly what every surface printed before the two merged.
    assert.match(agentsOutcomeText(['claude', 'cursor']), /Signed in to Claude Code and Cursor/);
});

test('an unread account facet claims nothing — a gap is not a zero', () => {
    // The known read renders both claims, so the negative below can fail.
    const known = familyListHtml(snapshotWith(['claude']));
    assert.match(known, /No account connected/);
    assert.match(known, /Signed in/);
    const rows = familyListHtml(snapshotWith(['claude']), { accountsKnown: false });
    assert.ok(rows.includes('Not checked'));
    assert.doesNotMatch(rows, /No account connected|Signed in/);
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
            openaiKey: ' key ', totalBudget: '25', reviewEnforcement: 'blocking',  // a stale draft field writes nothing
            localSource: '', localRoutingMode: 'cloud', mainModel: 'openai/gpt-5.6-sol',
            runtimeMode: 'advanced',
            connections: [{ kind: 'openai-compatible', name: 'local', base_url: 'http://127.0.0.1:41001/v1' }],
        },
        providerFields: [{ settingKey: 'OPENAI_API_KEY', stateKey: 'openaiKey' }],
        budgetFields: [{ settingKey: 'OUROBOROS_TOTAL_BUDGET_USD', stateKey: 'totalBudget' }],
        modelSlots: [{ settingKey: 'OUROBOROS_MODEL', stateKey: 'mainModel' }],
        trim: (value) => String(value || '').trim(),
    });
    assert.equal(draft.OPENAI_API_KEY, 'key');
    assert.deepEqual(draft.OUROBOROS_CONNECTIONS, [{ kind: 'openai-compatible', name: 'local', base_url: 'http://127.0.0.1:41001/v1' }]);
    assert.equal(draft.OUROBOROS_TOTAL_BUDGET_USD, 25);
    assert.equal(draft.OUROBOROS_MODEL, 'openai/gpt-5.6-sol');
    assert.equal('OUROBOROS_REVIEW_ENFORCEMENT' in draft, false);
    assert.equal(draft.OUROBOROS_RUNTIME_MODE, 'advanced');
    assert.equal('OUROBOROS_MODEL_HEAVY' in draft, false);
});

test('the draft reaches the compatible pair the runtime reads, following each edit', () => {
    // The real contract's fields, drafted exactly as the wizard drafts a bootstrap's initial state.
    const { contract, initialState } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url)));
    const legacy = 'http://127.0.0.1:8080/v1';
    const facts = (state) => {
        const draft = onboardingSettingsDraft({ state: { ...initialState, ...state }, providerFields: contract.providerFields,
            budgetFields: contract.budgetFields, modelSlots: contract.modelSlots });
        return { sources: configuredApiProviders(draft).map((entry) => entry.id), endpoint: compatibleEndpoint(draft),
            legacyKey: draft.OPENAI_BASE_URL };
    };
    // A new install drafts no legacy key at all.
    assert.deepEqual(facts({}), { sources: [], endpoint: { baseUrl: '', apiKey: '', apiKeySetting: 'OPENAI_API_KEY', legacy: true }, legacyKey: undefined });
    // The saved legacy pair: a masked OpenAI key now reaches the legacy address, and official OpenAI stays a choice.
    const saved = { legacyCompatibleBaseUrl: legacy, openaiKey: '***set***' };
    assert.deepEqual(facts(saved), { sources: ['openai', 'openai-compatible'],
        endpoint: { baseUrl: legacy, apiKey: '***set***', apiKeySetting: 'OPENAI_API_KEY', legacy: true }, legacyKey: legacy });
    // Clearing the key leaves an address without its key: no compatible source, nothing to send.
    assert.deepEqual(facts({ ...saved, openaiKey: '' }).sources, []);
    // A dedicated address, keyless or keyed, takes the built-in source; the legacy key stays home.
    assert.deepEqual(facts({ ...saved, compatibleBaseUrl: 'http://127.0.0.1:9090/v1' }).endpoint,
        { baseUrl: 'http://127.0.0.1:9090/v1', apiKey: '', apiKeySetting: 'OPENAI_COMPATIBLE_API_KEY', legacy: false });
    assert.deepEqual(facts({ ...saved, compatibleBaseUrl: 'http://127.0.0.1:9090/v1', compatibleApiKey: '***set***' }).endpoint,
        { baseUrl: 'http://127.0.0.1:9090/v1', apiKey: '***set***', apiKeySetting: 'OPENAI_COMPATIBLE_API_KEY', legacy: false });
    // Both pairs saved, even at one address: clearing the dedicated address selects the legacy pair and names
    // its key setting, so a kept key is read from that pair; restoring the address selects the dedicated one again.
    const both = { ...saved, compatibleBaseUrl: legacy, compatibleApiKey: '***set***' };
    assert.deepEqual(facts(both).endpoint,
        { baseUrl: legacy, apiKey: '***set***', apiKeySetting: 'OPENAI_COMPATIBLE_API_KEY', legacy: false });
    assert.deepEqual(facts({ ...both, compatibleBaseUrl: '' }).endpoint,
        { baseUrl: legacy, apiKey: '***set***', apiKeySetting: 'OPENAI_API_KEY', legacy: true });
    // Official OpenAI alone is that source only; a named server is its own source beside the legacy one.
    assert.deepEqual(facts({ openaiKey: '***set***' }).sources, ['openai']);
    assert.deepEqual(facts({ ...saved, connections: [{ kind: 'openai-compatible', name: 'lab', base_url: 'http://127.0.0.1:7070/v1' }] }).sources,
        ['openai', 'openai-compatible', 'openai-compatible@lab']);
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
    timestampOnly.profiles.profiles[0].status.last_verified_at = '2099-01-01T00:00:00Z';
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
    assert.match(dom.nodes.get('agents-outcome').textContent, /Signed in to Codex/);

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
                            review_eligible: true,
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
    assert.match(dom.nodes.get('onboarding-available-subagents').innerHTML, /aria-label="Subagent 1 reviews" checked/);
    assert.match(dom.nodes.get('onboarding-available-subagents').innerHTML, /Reviewers: 1/);
    assert.deepEqual(step.validateSubagents(), []);

    step.detach();
    store.dispose();
});

test('a generated draft whose rows carry no Reviewer mark is refused, never finished as an empty pool', async () => {
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, snapshotWith([])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const step = createAgentsStep({
        doc: dom.doc,
        store,
        previewTransport: async () => ({
            source: 'onboarding_default',
            diagnostics: [],
            available_subagents: { enabled: true, items: [{
                subagent_id: 'api_scout', recommended_use: 'Research.',
                route: { kind: 'api_model', target_id: 'openai/gpt-5.6-luna' },
            }] },
        }),
    });
    try {
        step.mount();
        await flush();
        await flush();
        assert.deepEqual(step.validateSubagents(), [
            'No row is marked Reviewer. Mark at least one row, or tick “Save without reviewers”.',
        ]);
        assert.equal(step.allowEmptyReviewPool, false);
        assert.match(dom.nodes.get('onboarding-available-subagents').innerHTML,
            /No row is marked Reviewer, so the main model reviews alone: one model, no second opinion\./);
    } finally {
        step.detach();
        store.dispose();
    }
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
    assert.equal(dom.windowListeners.length, 4,
        'pagehide, the shared status surface activation hooks and the Agents list language listener are bounded');
    assert.ok(dom.windowListeners.some(([type]) => type === 'ouro:language-changed'), 'the Agents list says its prices again on a language change');
    const [type, onPageHide] = dom.windowListeners[0];
    assert.equal(type, 'pagehide');
    handlers[handlers.length - 1]();
    await flush();

    onPageHide({ persisted: true });
    assert.equal(dom.windowListeners.length, 4, 'bfcache keeps the live step mounted');

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

test('preview status exposes the typed cause and explicit recovery replacement lasts one request', async () => {
    const dom = fakeDom();
    const calls = [];
    const notices = [];
    let fail = true;
    const response = { source: 'api_default', diagnostics: [], available_subagents: { enabled: true, items: [] } };
    const step = createAgentsStep({
        doc: dom.doc,
        previewPayload: () => ({ OUROBOROS_MODEL: 'claudexor::codex=owner-main' }),
        previewTransport: async () => {
            if (fail) throw Object.assign(new Error('Generic failure; see detail below.'), {
                body: { code: 'models_unavailable', detail: 'The selected account returned no models.', can_skip: true },
            });
            return response;
        },
        onPreviewStatus: () => notices.push({ pending: step.previewPending, error: step.previewError }),
        onSetupPreview: (_, options) => calls.push(options),
    });
    assert.equal(await step.refreshSubagentsPreview(), false);
    assert.deepEqual(step.previewFailure, { code: 'models_unavailable', canSkip: true,
        detail: 'The selected account returned no models.' });
    assert.deepEqual(notices.at(-1), { pending: false, error: 'The selected account returned no models.' });
    fail = false;
    assert.equal(await step.setSkipPresets(true, { replaceReviewers: true }), true);
    assert.equal(await step.refreshSubagentsPreview({ force: true }), true);
    assert.deepEqual(calls, [{ replaceReviewers: true }, { replaceReviewers: false }]);
    step.detach();
});

test('explicit reviewer recovery sends the visible clean roster and adopts the returned owner draft', async () => {
    const dom = fakeDom(), requests = [];
    const actor = { subagent_id: 'original', recommended_use: 'Inspect the task.',
        route: { kind: 'api_model', target_id: 'claudexor::codex=original' }, effort: 'high' };
    const recoveryActor = { ...actor, subagent_id: 'main-reviewer',
        route: { kind: 'api_model', target_id: 'claudexor::codex=main' } };
    const generated = { enabled: true, items: [actor] };
    const recovered = { enabled: true, items: [actor, recoveryActor] };
    const step = createAgentsStep({ doc: dom.doc,
        previewTransport: async (payload) => {
            requests.push(payload);
            return { source: 'onboarding_default', available_subagents:
                requests.length === 2 || requests.length === 4 ? recovered : generated };
        } });
    assert.equal(await step.refreshSubagentsPreview(), true);
    assert.equal(await step.setSkipPresets(true, { replaceReviewers: true }), true);
    assert.deepEqual(requests[1].OUROBOROS_SUBAGENTS, generated, 'clean visible actors are part of explicit recovery');
    assert.deepEqual(step.availableSubagents, recovered);
    assert.equal(await step.refreshSubagentsPreview({ force: true }), true);
    assert.deepEqual(step.availableSubagents, recovered, 'ordinary generation cannot remove the owner recovery actor');
    assert.equal(await step.setSkipPresets(true, { replaceReviewers: true }), true);
    assert.deepEqual(requests[3].OUROBOROS_SUBAGENTS, recovered, 'reprepare includes the already authored roster');
    step.detach();
});

test('a late preview settlement cannot notify or apply after the wizard detaches', async () => {
    const dom = fakeDom();
    let resolvePreview;
    let notifications = 0;
    let applied = 0;
    const step = createAgentsStep({ doc: dom.doc,
        previewTransport: () => new Promise((resolve) => { resolvePreview = resolve; }),
        onPreviewStatus: () => { notifications += 1; }, onSetupPreview: () => { applied += 1; } });
    const pending = step.refreshSubagentsPreview();
    assert.equal(notifications, 1);
    step.detach();
    resolvePreview({ available_subagents: { enabled: true, items: [] } });
    assert.equal(await pending, false);
    assert.equal(notifications, 1);
    assert.equal(applied, 0);
});

test('the wizard roster offers the providers whose keys the owner has typed so far', async () => {
    // docs/DESIGN.md §7: a source is CHOSEN. The wizard's keys are typed on the
    // Accounts step, so the provider list is derived from the CURRENT draft on
    // every entry into a step that shows these rows — never once at
    // construction, when no key exists yet.
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, snapshotWith([])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const draft = { OUROBOROS_MODEL: 'openai/gpt-5.6-sol' };
    const step = createAgentsStep({
        doc: dom.doc,
        store,
        providerProfiles: { openai: { label: 'OpenAI' } },
        previewPayload: () => ({ ...draft }),
        previewTransport: async () => ({
            source: 'onboarding_default',
            diagnostics: [],
            available_subagents: { enabled: true, items: [{
                subagent_id: 'api_scout', recommended_use: 'Research.',
                route: { kind: 'api_model', target_id: 'openai/gpt-5.6-luna' },
            }] },
        }),
    });

    try {
        step.mount();
        await flush();
        await flush();
        const html = () => dom.nodes.get('onboarding-available-subagents').innerHTML;
        // No key typed yet: the group offers nothing but the pointer to Accounts,
        // and the row's own OpenRouter spelling is rescued rather than swapped.
        assert.match(html(), /<option value="" disabled>Add a key in Accounts for more<\/option>/);
        assert.doesNotMatch(html(), /value="api:openai"/);
        assert.match(html(), /value="api:openrouter" selected>OpenRouter \(no key\)</);

        // The owner types an OpenAI key on Accounts and comes back: the provider
        // is offered now, under the setup contract's name for it.
        draft.OPENAI_API_KEY = 'sk-not-a-secret';
        step.mount();
        await flush();
        assert.match(html(), /<option value="api:openai">OpenAI<\/option>/);
        assert.doesNotMatch(html(), /value="api:anthropic"/);
    } finally {
        // A failed assertion must not leave the store polling: the runner would
        // wait for that timer instead of reporting the failure.
        step.detach();
        store.dispose();
    }
});
