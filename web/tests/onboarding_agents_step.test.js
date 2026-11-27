// The first-run Agents step (phase 3C). What is asserted here is what the
// owner is PROMISED: the ladder's honesty, the rotation artwork's inertness,
// what zero / one / several connected accounts declare to the completion
// endpoint, and that the step holds nothing open after it is disposed.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { createClaudexorStatusStore, familyLabel } from '../modules/claudexor_status_store.js';
import { compatibleEndpoint, configuredApiProviders } from '../modules/route_editor_primitives.js';
import { PROVIDER_CARDS } from '../modules/settings_provider_cards.js';
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

test('a family whose program the running engine reads as absent says so, and its one press installs and connects', () => {
    const absent = { installable: true, failed: false, byHand: false };
    const row = (html, harness) => html.slice(html.indexOf(`data-agent-family="${harness}"`)).split('data-agent-family=')[1];
    const rows = familyListHtml(snapshotWith([]), { programMissing: (harness) => (harness === 'agy' ? absent : null) });
    assert.match(row(rows, 'agy'), /Not installed where Claudexor runs/);
    assert.match(row(rows, 'agy'), /data-agent-connect="agy">\s*Install &amp; connect\s*</);
    assert.match(row(rows, 'codex'), /No account connected[\s\S]*data-agent-connect="codex">\s*Connect\s*</, 'a program not read as absent keeps Connect');
    // Where the engine cannot install it here, the row offers no press: its program line says why and offers Check again.
    const unable = familyListHtml(snapshotWith([]), { programMissing: () => ({ installable: false, failed: false, byHand: true }) });
    assert.doesNotMatch(unable, /data-agent-connect=/);
    // A listed account keeps the row's words and its Add another.
    const listed = familyListHtml(snapshotWith(['agy']), { programMissing: () => absent });
    assert.match(row(listed, 'agy'), /Signed in[\s\S]*data-agent-connect="agy">\s*Add another/);
    assert.doesNotMatch(row(listed, 'agy'), /Not installed/);
});

test('a program the engine reads absent and cannot install leaves its family no press, whatever the roster says', () => {
    const unable = { installable: false, failed: false, byHand: true };
    const able = { installable: true, failed: false, byHand: false };
    const row = (html, harness) => html.slice(html.indexOf(`data-agent-family="${harness}"`)).split('data-agent-family=')[1];
    const agy = (snapshot, missing, options = {}) => familyListHtml(snapshot, { ...options,
        programMissing: (harness) => (harness === 'agy' ? missing : null),
        programLine: (harness) => (harness === 'agy' ? '<span data-check-again>Check again</span>' : ''),
    });
    // Roster unread: no press at all — the ordinary unknown-roster Connect would install the program the engine has just
    // refused to install. The row keeps its own words (its status is the roster's gap, not a claim) and its line keeps the
    // reason and Check again; a family whose program was not read has no missing fact and keeps its press.
    const unread = agy(snapshotWith([]), unable, { accountsKnown: false });
    assert.match(row(unread, 'agy'), /Not checked/);
    assert.match(row(unread, 'agy'), /data-check-again/, 'the line keeps the reason and Check again');
    assert.doesNotMatch(row(unread, 'agy'), /data-agent-connect/);
    assert.match(row(unread, 'codex'), /data-agent-connect="codex">\s*Connect\s*</, 'an unread program is no missing fact');
    // Signed in already: Add another would need that same program, so it is gone too; its account stays listed.
    const listed = agy(snapshotWith(['agy']), unable);
    assert.match(row(listed, 'agy'), /Signed in/);
    assert.doesNotMatch(row(listed, 'agy'), /data-agent-connect/);
    // The engine still offering an install — a failed one included — keeps both presses: only a refused install removes them.
    assert.match(row(agy(snapshotWith([]), able, { accountsKnown: false }), 'agy'), /data-agent-connect="agy">\s*Connect\s*</);
    assert.match(row(agy(snapshotWith(['agy']), able), 'agy'), /data-agent-connect="agy">\s*Add another/);
});

// ---------------------------------------------------------------------------
// Which saved account the one press may start from (W2: stale, disabled,
// several — the account journey).
// ---------------------------------------------------------------------------

const savedRow = (harness, profileId, { enabled = true, verification = 'not_run' } = {}) => ({
    profile: { harness_id: harness, profile_id: profileId, enabled, credential_kind: 'config_dir_login' },
    status: { verification },
});
const roster = (rows) => ({ profiles: { profiles: rows, harnessAccounts: [], accountPools: [] } });
const codexRow = (html) => html.slice(html.indexOf('data-agent-family="codex"')).split('data-agent-family=')[1];

test('an unread roster never carries a remembered profile into the press (the stale account)', () => {
    const snapshot = roster([savedRow('codex', 'fixture-saved')]);
    // A remembered row of a roster this step has not read is no one's choice: the press would sign
    // in an account the engine may no longer hold.
    const stale = familyListHtml(snapshot, { accountsKnown: false });
    assert.doesNotMatch(stale, /data-agent-profile="fixture-saved"/);
    // A current read keeps the single named default — the positive control.
    assert.match(familyListHtml(snapshot, { accountsKnown: true }), /data-agent-profile="fixture-saved"/);
});

test('several saved accounts become an explicit choice, and the press guesses no profile', () => {
    const snapshot = roster([savedRow('codex', 'work'), savedRow('codex', 'personal')]);
    const html = familyListHtml(snapshot);
    assert.match(codexRow(html), /data-agent-connect="codex">\s*Connect\s*</, 'the press stays');
    assert.match(codexRow(html), /<select[^>]*data-agent-account/, 'an explicit choice names the account');
    assert.match(codexRow(html), /<option value="work">/);
    assert.match(codexRow(html), /<option value="personal">/);
    assert.doesNotMatch(codexRow(html), /data-agent-profile=/, 'no account is guessed onto the press');
    // While the program is missing, the same family offers no press: the program line installs, and
    // Connect would start an unnamed login that creates a second account.
    const absent = { installable: true, failed: false, byHand: false };
    const installs = [];
    const withMissing = familyListHtml(snapshot, { programMissing: () => absent,
        programLine: (harness, { install }) => { installs.push({ harness, install }); return ''; } });
    assert.doesNotMatch(codexRow(withMissing), /data-agent-connect=/);
    assert.doesNotMatch(codexRow(withMissing), /data-agent-account/);
    assert.deepEqual(installs.filter((entry) => entry.harness === 'codex'), [{ harness: 'codex', install: true }],
        'the program line installs, only');
});

test('a switched-off account is listed disabled, never reused, and Add another stays the way in', () => {
    const snapshot = roster([savedRow('codex', 'old', { enabled: false, verification: 'passed' })]);
    // Program installed: the off account stays visible as a disabled option — never pickable, never
    // re-enabled — and with nothing enabled to choose, the row's press is the ordinary Add another,
    // the separate explicit step to a NEW account (§7); it is not gated on the disabled pick.
    const html = familyListHtml(snapshot);
    assert.doesNotMatch(codexRow(html), /data-agent-profile=/);
    assert.match(codexRow(html), /<option value="old" disabled>/);
    assert.doesNotMatch(codexRow(html), /<option value="old">/, 'the off account is not offered as usable');
    assert.match(codexRow(html), /data-agent-connect="codex" data-agent-new>\s*Add another\s*</);
    assert.doesNotMatch(codexRow(html), /disabled data-agent-connect/);
    // Program missing and installable: no Install & connect carries it, and the program line
    // installs only — the press neither signs the disabled account back in nor creates a profile.
    const absent = { installable: true, failed: false, byHand: false };
    const installs = [];
    const withMissing = familyListHtml(snapshot, { programMissing: () => absent,
        programLine: (harness, { install }) => { installs.push({ harness, install }); return ''; } });
    assert.doesNotMatch(codexRow(withMissing), /data-agent-connect=/);
    assert.doesNotMatch(codexRow(withMissing), /data-agent-profile=/);
    assert.deepEqual(installs.filter((entry) => entry.harness === 'codex'), [{ harness: 'codex', install: true }],
        'a disabled single account gets the install, never the sign-in');
    // A remembered pick the roster can no longer serve ENABLED is dropped, never substituted: the
    // placeholder returns and the press waits, while the off account keeps its disabled listing.
    const mixed = roster([savedRow('codex', 'live'), savedRow('codex', 'old', { enabled: false, verification: 'passed' })]);
    const dropped = codexRow(familyListHtml(mixed, { chosen: { codex: 'old' } }));
    assert.match(dropped, /<option value="" disabled selected>Account…<\/option>/);
    assert.match(dropped, /<option value="old" disabled>/);
    assert.match(dropped, /disabled data-agent-connect="codex">/, 'an off pick leaves the press waiting');
    const stands = codexRow(familyListHtml(mixed, { chosen: { codex: 'live' } }));
    assert.match(stands, /<option value="live" selected>/);
    assert.doesNotMatch(stands, /disabled data-agent-connect/);
    // W5: a sole survivor is never the silent substitute for a pick this roster dropped. The owner chose
    // `away`; only `home` remains — the press guesses no profile, and Add another stays the named way in.
    const shrank = roster([savedRow('codex', 'home')]);
    const substituted = codexRow(familyListHtml(shrank, { chosen: { codex: 'away' } }));
    assert.doesNotMatch(substituted, /data-agent-profile=/, 'the sole survivor is not remembered into the press');
    assert.match(substituted, /<option value="" disabled selected>Account…<\/option>/,
        'the row asks for the answer it does not have');
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

test('the draft carries the built-in compatible key\'s consent beside its address; a consent is no access', () => {
    const { contract, initialState } = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url)));
    const field = contract.providerFields.find((each) => each.settingKey === 'OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED');
    assert.deepEqual([field.stateKey, field.group, field.inputType, initialState[field.stateKey]], ['compatibleContinueUnconfirmed', 'more', 'text', ''],
        'a client that maps the contract\'s fields to the save body sends it, empty until the owner gives one');
    assert.equal(field.label, PROVIDER_CARDS.find((card) => card.kind === 'openai-compatible').consent.label, 'the box and the contract say it in one sentence');
    const draftOf = (state) => onboardingSettingsDraft({ state: { ...initialState, ...state }, providerFields: contract.providerFields,
        budgetFields: contract.budgetFields, modelSlots: contract.modelSlots });
    assert.equal(draftOf({}).OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED, '');
    assert.equal(draftOf({ compatibleContinueUnconfirmed: ' http://lab.lan/v1 ' }).OPENAI_COMPATIBLE_CONTINUE_UNCONFIRMED, 'http://lab.lan/v1');
    assert.deepEqual(configuredApiProviders(draftOf({ compatibleContinueUnconfirmed: 'http://lab.lan/v1' })), [], 'an address\'s consent configures no provider');
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

// W5 (2026-11-27): the family row's own Install press starts a program install that can outlive the
// click; a Connect on the same family meanwhile started a SECOND job server-side and one of the two
// was orphaned. The fence lives at the ONE sign-in boundary every press of the row passes, so Connect
// AND Add another meet it while the install stands; the refusal card keeps the reason and its own
// recovery (that card is the way back). A program the engine reads INSTALLED, an old indeterminate
// update included, leaves the ordinary sign-in alone.
test('a standing install fences every sign-in press of that family; an installed program signs in', async () => {
    const posts = [];
    let installed = false;   // the engine's own read of the program: false = still missing
    let pendingIntent = false;   // the row's own Install press, unresolved
    let newAccount = false;   // the press the fake list currently wires: the ordinary Connect
    let settled = false;   // the SAME operation observed terminal: the install finished
    const fetchImpl = async (url, init) => {
        if (String(url) === '/api/claudexor/login' && (init?.method || 'GET') === 'POST') {
            posts.push(JSON.parse(init.body || '{}'));
            return json(200, { job_id: 'fence', job: { state: 'running' } });
        }
        if (String(url).startsWith('/api/claudexor/login')) return json(200, { job: { state: 'running' } });
        return json(200, snapshotWith([]));
    };
    const store = createClaudexorStatusStore({
        fetchImpl,
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const list = dom.nodes.get('agents-family-list');
    const clicks = [];
    const press = { dataset: {}, getAttribute: () => 'claude',
        hasAttribute: (name) => name === 'data-agent-new' && newAccount,
        closest: () => ({ getAttribute: () => 'claude', querySelector: () => null }),
        addEventListener: (_type, fn) => clicks.push(fn) };
    list.buttons = [press];
    const step = createAgentsStep({ doc: dom.doc, store, fetchImpl,
        maintenanceApi: {
            harnessMaintenanceInventory: async () => ({ harnesses: [{ harness: 'claude',
                selection: installed ? { path: '/vendor/claude', version: '1.0' } : null,
                newest: { status: 'available', version: '1.0', observedAt: 'now', source: 'registry' },
                updateAvailable: installed ? 'no' : 'cannot_tell',
                actions: { install: { available: true, reason: null, commands: [] } },
                operation: settled ? { id: 'op-1', harness: 'claude', state: 'succeeded' }
                    : (pendingIntent ? { id: 'op-1', harness: 'claude', state: 'running' } : null) }] }),
        } });
    try {
        await store.refresh();
        step.mount();
        await flush();
        assert.equal(posts.length, 0, 'mounting starts nothing');
        // Missing program + the row's own Install standing: the fence answers before any start, so no
        // POST goes out. The refusal CARD is the way back — it keeps the reason and its own recovery,
        // never an empty host. A remount (what the wizard does around a state change) re-reads the
        // rows, so the fence observes it.
        pendingIntent = true;
        await store.refresh();
        step.mount();
        await flush();
        clicks.at(-1)();
        await flush();
        assert.equal(posts.length, 0, 'a standing install fences the family\'s Connect');
        assert.match(dom.nodes.get('agents-login-host').innerHTML,
            /Claude Code is still being installed\. Connect once that install finishes\./,
            'the fence says why and what to do, on the card that owns the recovery');
        // Add another passes the SAME boundary: an unnamed new account would orphan one of the two jobs
        // just as the ordinary Connect would, so it is fenced too while the install stands.
        newAccount = true;
        clicks.at(-1)();
        await flush();
        assert.equal(posts.length, 0, 'Add another is fenced by the same standing install');
        assert.match(dom.nodes.get('agents-login-host').innerHTML, /still being installed/);
        // The install settles and the engine READS the program installed: the ordinary sign-in is left
        // alone. The terminal observation of the SAME operation is what releases the shared record — a
        // later `operation: null` would leave this session's own receipt standing for the next test.
        await step.dispose();
        step.detach();
        installed = true;
        pendingIntent = false;
        settled = true;
        newAccount = false;
        clicks.length = 0;
        await store.refresh();
        const fresh = createAgentsStep({ doc: dom.doc, store, fetchImpl,
            maintenanceApi: {
                harnessMaintenanceInventory: async () => ({ harnesses: [{ harness: 'claude',
                    selection: { path: '/vendor/claude', version: '1.0' },
                    newest: { status: 'available', version: '1.0', observedAt: 'now', source: 'registry' },
                    updateAvailable: 'no',
                    actions: { install: { available: true, reason: null, commands: [] } },
                    operation: { id: 'op-1', harness: 'claude', state: 'succeeded' } }] }),
            } });
        try {
            fresh.mount();
            await flush();
            clicks.at(-1)();
            await flush();
            assert.equal(posts.length, 1, 'an installed program leaves the ordinary sign-in alone');
        } finally {
            await fresh.dispose();
            fresh.detach();
        }
    } finally {
        await step.dispose();
        step.detach();
        store.dispose();
    }
});

test('each family row carries its program line, read only while the engine runs and again after a sign-in settles', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const asked = [];
    const program = (harness) => ({ harness, selection: { path: `/vendor/${harness}`, version: '0.50.0' },
        newest: { status: 'available', version: '0.60.0', observedAt: 'now', source: 'registry' }, updateAvailable: 'yes',
        actions: { update: { available: true, reason: null } }, operation: null });
    const maintenanceApi = { harnessMaintenanceInventory: async (query) => {
        asked.push(query);
        return { harnesses: [program('codex')].filter((row) => !query.harness || row.harness === query.harness) };
    } };
    let daemon = 'stale';
    const urls = [];
    const fetchImpl = async (url, init) => {
        urls.push(String(url));
        if (String(url).startsWith('/api/claudexor/login')) {
            return json(200, { job_id: 'j1', job: { state: 'failed', outcome: { reason: 'vendor_program_failed' } } });
        }
        // `stale` with every facet `ok` is a RUNNING engine by daemonAnswered (the partial-answer
        // case its comment documents), so the stopped phase here carries the stopped engine's own
        // honest `not_read` marks — the wire shape the real endpoint answers for a daemon nobody
        // has woken. The wake the owner's press triggers then answers `ok`, the ordinary running phase.
        const reads = daemon === 'stale'
            ? { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } : undefined;
        return json(200, { ...snapshotWith([]), daemon: { state: daemon }, ...(reads ? { reads } : {}) });
    };
    const store = createClaudexorStatusStore({
        fetchImpl, doc: { hidden: false, addEventListener() {}, removeEventListener() {} }, pollMs: 5000,
    });
    const dom = fakeDom();
    const list = dom.nodes.get('agents-family-list');
    const connect = [];
    list.buttons = [{ getAttribute: () => 'codex', addEventListener: (_type, fn) => connect.push(fn) }];
    const step = createAgentsStep({ doc: dom.doc, store, fetchImpl, maintenanceApi });
    step.mount();
    await flush();
    assert.deepEqual(asked, [], 'a stopped engine is not asked');
    assert.doesNotMatch(list.innerHTML, /harness-model-program|Not installed|Install &amp; connect/);
    // Said once: Connect installs a missing program first; the owner's press, which says it starts the daemon, lists them.
    assert.match(list.innerHTML, /data-programs-unknown><span>If a program is missing, Connect tries to install it first\.<\/span><button[^>]*data-program-action="find">Show programs to install \(prepares and starts the agent daemon\)</);
    assert.ok(!urls.includes('/api/claudexor/wake'), 'showing the step starts nothing');

    daemon = 'running';
    await store.refresh();
    await flush();
    assert.deepEqual(asked.map((query) => [query.harness, query.checkLatest]), [['', false], ['codex', true]]);
    assert.doesNotMatch(list.innerHTML, /data-programs-unknown/, 'running and read, the programs speak for themselves');
    const codexRow = list.innerHTML.slice(list.innerHTML.indexOf('data-agent-family="codex"'),
        list.innerHTML.indexOf('data-agent-family="claude"'));
    assert.match(codexRow, /<span>Program 0\.50\.0\u00a0· <span class="harness-model-latest">0\.60\.0 available<\/span><\/span>/);
    assert.match(codexRow, /data-program-action="latest"[^>]*>Update</);
    assert.equal((list.innerHTML.match(/harness-model-program/g) || []).length, 1, 'only the family with something to act on');

    connect.at(-1)();
    for (let tick = 0; tick < 5; tick += 1) { t.mock.timers.tick(2000); await flush(); }
    assert.ok(asked.length >= 4, 'a settled sign-in, failed here, asks the engine again');

    const count = asked.length;
    await step.dispose();
    step.detach();
    await store.refresh();
    await flush();
    assert.equal(asked.length, count, 'the disposed step reads nothing more');
    store.dispose();
});

test('a family whose program the running engine lacks installs and connects from its row; each answer reads it again', async (t) => {
    const asked = [];
    let agy = { harness: 'agy', selection: { path: null, version: null }, newest: { status: 'not_read', source: null }, releaseNotesUrl: 'https://vendor.invalid/agy/releases',
        updateAvailable: 'cannot_tell', actions: { install: { available: true, reason: null, commands: [] } }, operation: null };
    const opencode = { ...agy, harness: 'opencode' };  // a program of no subscription family: installed from its own line
    const maintenanceApi = { harnessMaintenanceInventory: async (query) => {
        asked.push(query);
        return { harnesses: [agy, opencode].filter((row) => !query.harness || row.harness === query.harness) };
    } };
    const posts = [];
    const fetchImpl = async (url, init = {}) => {
        if (url === '/api/claudexor/login' && init.method === 'POST') {
            posts.push(JSON.parse(init.body));
            // The engine's own words for a failed installer on the way to the sign-in (Connect installs first).
            return json(503, { error: 'The installer exited with 1.', code: 'installer_failed', context: { operationId: 'op-9' } });
        }
        return json(200, { ...snapshotWith([]), daemon: { state: 'running' } });
    };
    const store = createClaudexorStatusStore({ fetchImpl, doc: { hidden: false, addEventListener() {}, removeEventListener() {} }, pollMs: 5000 });
    const dom = fakeDom();
    const list = dom.nodes.get('agents-family-list');
    const connect = [];
    list.buttons = [{ getAttribute: () => 'agy', addEventListener: (_type, fn) => connect.push(fn) }];
    const step = createAgentsStep({ doc: dom.doc, store, fetchImpl, maintenanceApi });
    t.after(async () => { await step.dispose(); step.detach(); store.dispose(); });  // a failed assertion leaves no poll behind
    step.mount();
    await store.refresh();
    await flush();
    const row = () => list.innerHTML.slice(list.innerHTML.indexOf('data-agent-family="agy"'));
    assert.match(row(), /Not installed where Claudexor runs/);
    assert.match(row(), /data-agent-connect="agy">\s*Install &amp; connect/);
    assert.doesNotMatch(row().split('agent-other-programs')[0], /harness-model-program/, 'nothing to say beside the row\'s own press');
    // The engine's other programs: one compact line, each its own Install, no sign-in or model promised.
    const more = list.innerHTML.slice(list.innerHTML.indexOf('agent-other-programs'));
    assert.match(more, /<span>More programs:<\/span>/);
    assert.match(more, /data-program="opencode">\s*<span>OpenCode · Not installed<\/span><button[^>]*data-program-action="install">Install</);
    assert.doesNotMatch(more, /data-program="agy"/, 'a subscription family installs from its own row');

    const before = asked.length;
    agy = { ...agy, operation: { id: 'op-9', harness: 'agy', state: 'failed', termination: 'confirmed', outcome: 'unknown', mutation: 'none',
        commands: [{ binary: 'bash', args: [], env: {}, script: { url: 'https://vendor.invalid/agy/install.sh', size: 9, sha256: 'x' } }] } };
    connect.at(-1)();
    await flush();
    assert.deepEqual(posts, [{ harness: 'agy', login_flow: 'device_auth' }]);
    assert.ok(asked.slice(before).some((query) => query.fresh), 'the answer reads the program again');
    assert.match(row(), /Install failed/);
    assert.doesNotMatch(row(), /install\.sh|vendor\.invalid\/agy\/releases/, 'neither what the engine ran nor a release-notes page');
    assert.match(row(), /data-program-action="inspect">Check again</);
});

test('a missing program is a family row\'s own press only where no second account follows', () => {
    const missing = { installable: true, failed: false, byHand: false };
    const row = (html, harness) => html.slice(html.indexOf(`data-agent-family="${harness}"`)).split('data-agent-family=')[1];
    const lines = [];
    const html = (snapshot, accountsKnown = true) => familyListHtml(snapshot, { accountsKnown,
        programMissing: (harness) => (harness === 'agy' ? missing : null),
        programLine: (harness, options) => { lines.push([harness, options]); return ''; } });
    const named = (ids, verification = 'failed') => ({ ...snapshotWith([]), profiles: { profiles: ids.map((id) => ({
        profile: { harness_id: 'agy', profile_id: id }, status: { verification, verification_source: 'vendor' } })) } });
    // Read empty: Install & connect, a first sign-in; the line does not install it twice.
    let out = html(snapshotWith([]));
    assert.match(row(out, 'agy'), /Not installed where Claudexor runs[\s\S]*data-agent-connect="agy">\s*Install &amp; connect/);
    assert.doesNotMatch(row(out, 'agy'), /data-agent-profile/);
    assert.deepEqual(lines.find(([harness]) => harness === 'agy')[1], { install: false });
    // Its one account, saved before an install that failed: the press signs that one in again, no second is made.
    lines.length = 0;
    out = html(named(['work']));
    assert.match(row(out, 'agy'), /data-agent-connect="agy" data-agent-profile="work">\s*Install &amp; connect/);
    // Several read, none signed in: the ordinary press would sign in a default profile and add an account beside
    // them, and no account is guessed — the row offers no press; the line installs alone, Settings signs one in.
    lines.length = 0;
    out = html(named(['work', 'home']));
    assert.doesNotMatch(row(out, 'agy'), /data-agent-connect/);
    assert.deepEqual(lines.find(([harness]) => harness === 'agy')[1], { install: true });
    // Accounts unread: the press stays the engine's ordinary first flow and the line installs alone.
    lines.length = 0;
    out = html(snapshotWith([]), false);
    assert.match(row(out, 'agy'), /data-agent-connect="agy">\s*Connect\s*</);
    assert.doesNotMatch(row(out, 'agy'), /data-agent-profile|Not installed/);
    assert.deepEqual(lines.find(([harness]) => harness === 'agy')[1], { install: true });
    // Its one account with the program present: Connect signs that one in again; with two, the saved
    // accounts become an explicit choice — the press starts exactly the account the reader names,
    // and none is guessed onto it (W2: no dead row for several unsigned saved accounts).
    const present = (snapshot) => familyListHtml(snapshot, { accountsKnown: true, programMissing: () => null });
    assert.match(row(present(named(['work'])), 'agy'), /data-agent-connect="agy" data-agent-profile="work">\s*Connect\s*</);
    const two = row(present(named(['work', 'home'])), 'agy');
    assert.match(two, /data-agent-connect="agy">\s*Connect\s*/, 'the press stays for an explicit choice');
    assert.match(two, /<select[^>]*data-agent-account/);
    assert.match(two, /<option value="work">/);
    assert.match(two, /<option value="home">/);
    assert.doesNotMatch(two, /data-agent-profile=/);
    // Connected already: Add another, and the line installs the program alone.
    lines.length = 0;
    out = html(named(['work'], 'passed'));
    assert.match(row(out, 'agy'), /Add another/);
    assert.deepEqual(lines.find(([harness]) => harness === 'agy')[1], { install: true });
});

test('Antigravity on a fictitious engine: name, saved account, failed install, retried on the same account', async (t) => {
    const asked = [];
    let saved = [], installs = false;
    const absent = { harness: 'agy', selection: { path: null, version: null }, newest: { status: 'not_read', source: null },
        updateAvailable: 'cannot_tell', actions: { install: { available: true, reason: null, commands: [] } }, operation: null };
    let program = absent;
    const maintenanceApi = { harnessMaintenanceInventory: async (query) => { asked.push(query); return { harnesses: [program] }; } };
    const posts = [];
    const fetchImpl = async (url, init = {}) => {
        if (url === '/api/claudexor/login' && init.method === 'POST') {
            const body = JSON.parse(init.body);
            posts.push(body);
            if (!body.profile_id) return json(400, { error: 'Antigravity keeps each account under its own name.',
                code: 'credential_profile_required', required_actions: ['add_named_account'] });
            if (!saved.includes(body.profile_id)) saved = [...saved, body.profile_id];  // saved unverified before install and sign-in
            if (!installs) return json(503, { error: 'The installer exited with 1.', code: 'installer_failed', context: { operationId: 'op-9' } });
            program = { ...absent, selection: { path: '/vendor/agy', version: '1.11.3' }, updateAvailable: 'no' };
            return json(200, { job_id: 'j-agy', job: { state: 'failed', harness: 'agy', outcome: { reason: 'vendor_program_failed' } } });
        }
        const snapshot = snapshotWith(['codex']);
        snapshot.profiles.profiles.push(...saved.map((id) => ({ profile: { harness_id: 'agy', profile_id: id },
            status: { verification: 'failed', verification_source: 'vendor' } })));
        return json(200, snapshot);
    };
    const store = createClaudexorStatusStore({ fetchImpl, doc: { hidden: false, addEventListener() {}, removeEventListener() {} }, pollMs: 5000 });
    const dom = fakeDom();
    const list = dom.nodes.get('agents-family-list');
    const presses = [];
    // The fake list hands back one Connect, carrying the profile the rebuilt markup names for it.
    list.querySelectorAll = (selector) => (selector === '[data-agent-connect]' ? [{ getAttribute: () => 'agy',
        dataset: { agentProfile: (list.innerHTML.match(/data-agent-connect="agy" data-agent-profile="([^"]+)"/) || [])[1] },
        addEventListener: (_type, fn) => presses.push(fn) }] : []);
    const step = createAgentsStep({ doc: dom.doc, store, fetchImpl, maintenanceApi });
    t.after(async () => { await step.dispose(); step.detach(); store.dispose(); });
    step.mount();
    await store.refresh();
    await flush();
    assert.match(list.innerHTML, /data-agent-connect="agy">\s*Install &amp; connect/);
    presses.at(-1)();
    await flush();
    assert.deepEqual(posts, [{ harness: 'agy', login_flow: 'device_auth' }]);
    const host = dom.nodes.get('agents-login-host');
    assert.match(host.innerHTML, /data-profile-name-entry/, 'the engine\'s own next step: a name');
    assert.doesNotMatch(host.innerHTML, /Installing|it is installed first/, 'nothing was said installed for a sign-in that only asked a name');
    // The named sign-in (typed in the card; Settings' test drives it through the DOM): the engine saved the name, then its
    // installer failed. The wizard reads the engine again; the owner's codex account stays listed.
    posts.length = 0;
    saved = ['work'];
    await store.refresh();
    await flush();
    assert.match(list.innerHTML, /data-agent-connect="agy" data-agent-profile="work">\s*Install &amp; connect/, 'retried on the saved name');
    assert.match(list.innerHTML, /data-agent-family="codex"[\s\S]*Add another/, 'the earlier account keeps its place');
    presses.at(-1)();
    await flush();
    assert.deepEqual(posts, [{ harness: 'agy', login_flow: 'device_auth', profile_id: 'work' }], 'no second account is made');
    assert.ok(asked.some((query) => query.harness === 'agy' && query.fresh), 'the answer reads the program again');
    // Installed this time, the sign-in itself failed. Back on the step, the ordinary row signs that same account in.
    installs = true;
    presses.at(-1)();
    await flush();
    step.mount();
    await store.refresh();
    await flush();
    assert.match(list.innerHTML, /data-agent-connect="agy" data-agent-profile="work">\s*Connect\s*</);
    posts.length = 0;
    presses.at(-1)();
    await flush();
    assert.deepEqual(posts, [{ harness: 'agy', login_flow: 'device_auth', profile_id: 'work' }]);
    assert.deepEqual(saved, ['work'], 'one account, however many attempts');
});

test('the Models step mounts the same step without family rows, and asks the engine nothing about programs', async () => {
    const asked = [];
    const maintenanceApi = { harnessMaintenanceInventory: async (query) => { asked.push(query); return { harnesses: [] }; } };
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, snapshotWith(['codex'])),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} }, pollMs: 5000,
    });
    const dom = fakeDom();
    const rows = dom.nodes.get('agents-family-list');
    dom.nodes.delete('agents-family-list');  // the Models step: the editor, no Accounts rows
    const step = createAgentsStep({ doc: dom.doc, store, maintenanceApi });
    try {
        step.mount();
        await store.refresh();
        await flush();
        assert.deepEqual(asked, [], 'no program line here, so no inventory or newest-release read');
        dom.nodes.set('agents-family-list', rows);  // back on Accounts: the remount reads for its rows
        step.mount();
        await flush();
        assert.deepEqual(asked.map((query) => [query.harness, query.checkLatest]), [['', false]]);
    } finally { step.detach(); store.dispose(); }
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

// W3 (2026-11-26): the first-run Codex row did not keep the account the owner picked. Widening the list replaces
// its markup wholesale, and the <select> carried no memory of the answer, so an unrelated program row's own press
// (More programs -> Install) made a chosen `studio-b` silently become the first option again.
test('a rebuilt multiple-account row keeps the pick and never falls back to the first account', () => {
    const two = roster([savedRow('codex', 'studio-a'), savedRow('codex', 'studio-b')]);
    // A fresh row starts on nothing at all, and its press cannot act until an account is named.
    const fresh = codexRow(familyListHtml(two, { chosen: {} }));
    assert.match(fresh, /<option value="" disabled selected>Account…<\/option>/);
    assert.match(fresh, /disabled data-agent-connect="codex">\s*Connect\s*</);
    // The remembered answer is the row's own choice: it stands selected, and the press is open.
    const kept = codexRow(familyListHtml(two, { chosen: { codex: 'studio-b' } }));
    assert.match(kept, /<option value="studio-b" selected>\s*studio-b<\/option>/);
    assert.match(kept, /<option value="studio-a">/);
    assert.doesNotMatch(kept, /Choose an account…|disabled data-agent-connect/);
    // An account the roster no longer holds is never substituted by whichever one is now first.
    const swapped = codexRow(familyListHtml(roster([savedRow('codex', 'studio-a'), savedRow('codex', 'studio-c')]),
        { chosen: { codex: 'studio-b' } }));
    assert.match(swapped, /<option value="" disabled selected>Account…<\/option>/);
    assert.doesNotMatch(swapped, /studio-b/);
    // One saved account is still the ordinary named default: nothing to choose, and no placeholder either.
    const one = codexRow(familyListHtml(roster([savedRow('codex', 'studio-a')]), { chosen: { codex: 'studio-a' } }));
    assert.match(one, /data-agent-profile="studio-a">/);
    assert.doesNotMatch(one, /data-agent-account/);
});

test('the repaint after an unrelated row keeps the pick the owner made in the standing row', async () => {
    const snapshot = snapshotWith([]);
    snapshot.profiles = roster([savedRow('codex', 'studio-a'), savedRow('codex', 'studio-b')]).profiles;
    const store = createClaudexorStatusStore({
        fetchImpl: async () => json(200, snapshot),
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} },
        pollMs: 5000,
    });
    const dom = fakeDom();
    const list = dom.nodes.get('agents-family-list');
    // What the account rows standing right now hold: nothing chosen yet.
    let standing = [];
    list.querySelectorAll = (selector) => (selector === '[data-agent-account]' ? standing : []);
    const step = createAgentsStep({ doc: dom.doc, store });
    try {
        await store.refresh();
        step.paint();
        assert.match(codexRow(list.innerHTML), /Account…/);
        assert.match(codexRow(list.innerHTML), /disabled data-agent-connect="codex">/);
        // The owner answers the row (studio-b); then an unrelated row repaints the same list.
        standing = [{ value: 'studio-b', closest: () => ({ getAttribute: () => 'codex' }), addEventListener: () => {} }];
        step.paint();
        assert.match(codexRow(list.innerHTML), /<option value="studio-b" selected>/, 'the repaint keeps the pick');
        assert.doesNotMatch(codexRow(list.innerHTML), /Account…|disabled data-agent-connect/);
    } finally {
        step.detach();
        store.dispose();
    }
});

// W4 (2026-11-26): the pick was remembered only when a paint happened to read the standing row, so an
// explicit later answer was lost the moment the wizard re-rendered the step's DOM (the Accounts step's
// own local-preset select does exactly that): the rebuilt row named the account an EARLIER paint had
// saved, and its press signed that one in. The answer is now remembered the moment it is given.
test('two consecutive picks survive a rebuild that cannot read the standing row, and the press signs the latest', async () => {
    const snapshot = snapshotWith([]);
    snapshot.profiles = roster([savedRow('codex', 'studio-a'), savedRow('codex', 'studio-b')]).profiles;
    const posts = [];
    const fetchImpl = async (url, init) => {
        if (String(url) === '/api/claudexor/login' && init?.method === 'POST') {
            posts.push(JSON.parse(init.body));
            return json(200, { job_id: 'w4', job: { state: 'running' } });
        }
        if (String(url).startsWith('/api/claudexor/login')) return json(200, { job: { state: 'running' } });
        return json(200, snapshot);
    };
    const store = createClaudexorStatusStore({ fetchImpl,
        doc: { hidden: false, addEventListener() {}, removeEventListener() {} }, pollMs: 5000 });
    const dom = fakeDom();
    const list = dom.nodes.get('agents-family-list');
    const clicks = [];
    const change = [];
    const row = { getAttribute: () => 'codex', querySelector: (selector) => (selector === '[data-agent-account]' ? select : press) };
    const press = { disabled: true, dataset: {}, getAttribute: () => 'codex', hasAttribute: () => false,
        closest: () => row, addEventListener: (_type, fn) => clicks.push(fn) };
    const select = { value: '', closest: () => row, addEventListener: (_type, fn) => change.push(fn) };
    // The standing row the owner answers in; a wizard step re-render leaves none of it behind.
    let standing = [select];
    list.querySelectorAll = (selector) => (selector === '[data-agent-account]' ? standing
        : selector === '[data-agent-connect]' ? [press] : []);
    const step = createAgentsStep({ doc: dom.doc, store, fetchImpl });
    try {
        await store.refresh();
        step.paint();
        assert.match(codexRow(list.innerHTML), /<option value="" disabled selected>Account…<\/option>/,
            'a fresh row starts on the placeholder');
        // The owner answers studio-b; the store settles and the row repaints around that answer.
        select.value = 'studio-b';
        change.at(-1)();
        step.paint();
        assert.match(codexRow(list.innerHTML), /<option value="studio-b" selected>/);
        // Then the owner picks studio-a, and the wizard re-renders the step's DOM for an unrelated
        // change: the standing row is gone, so the rebuild cannot read it — only the remembered answer.
        select.value = 'studio-a';
        change.at(-1)();
        standing = [];
        step.paint();
        const rebuilt = codexRow(list.innerHTML);
        assert.match(rebuilt, /<option value="studio-a" selected>/, 'the rebuilt row names the exact latest answer');
        assert.doesNotMatch(rebuilt, /<option value="studio-b" selected>/);
        // The fresh <select> the browser built from that markup selects that account; its press
        // signs exactly it in — not the name an earlier paint happened to save.
        select.value = /<option value="([^"]+)" selected>/.exec(rebuilt)[1];
        clicks.at(-1)();
        await flush();
        assert.equal(posts.at(-1)?.profile_id, 'studio-a', 'the press dispatches the exact latest selection');
    } finally {
        step.detach();
        store.dispose();
    }
});
