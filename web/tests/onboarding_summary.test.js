// The first-run Summary (owner 2026-11-23, after the side-by-side: both screens as proposed): the choices that shape
// the first hour lead, one row each in the rail's order, and every assignment Start saves waits behind one disclosure.
// The rows are built from the real setup contract and the subscription fixture the wizard walks use; the wizard's own
// walks (onboarding_wizard_render.test.js) prove the same rows reach the page and the completion request.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { createConsciousnessStep } from '../modules/onboarding_consciousness.js';
import { onboardingSettingsDraft } from '../modules/onboarding_agents_step.js';
import { createSummaryStep } from '../modules/onboarding_summary.js';

const BOOTSTRAP = JSON.parse(readFileSync(new URL('./fixtures/onboarding_bootstrap.json', import.meta.url), 'utf8'));
const SUBSCRIPTION = JSON.parse(readFileSync(new URL('./fixtures/subscription_setup.json', import.meta.url), 'utf8'));
const SOURCES = SUBSCRIPTION.catalog.model_sources;
const CODEX = 'claudexor::codex=gpt-test';
const ACCESS = Object.fromEntries(BOOTSTRAP.contract.runtimeModes.map((mode) => [mode.value, mode.label]));
const PACKET = { enabled: true, items: [{ subagent_id: 'packet', review_eligible: true, delivery: 'packet',
    route: { kind: 'api_model', target_id: 'openai/gpt-5.6-sol' } }] };

/** A Summary over a draft: by default a fresh Codex-only setup (no API key), as Review & start reaches it. */
function summaryOf(draft = {}, read = {}, { offer = BOOTSTRAP.consciousnessOffer, local = [] } = {}) {
    const state = { agentsConnected: ['codex'], runtimeMode: 'advanced', modelAccounts: {}, modelContextWindows: {},
        mainModel: CODEX, lightModel: CODEX, totalBudget: 10, perTaskCostUsd: 5, ...draft };
    const reads = () => ({ api: false, subscription: true, local: false, profile: 'openrouter', connected: SOURCES, sources: SOURCES,
        draft: onboardingSettingsDraft({ state, providerFields: BOOTSTRAP.contract.providerFields, modelSlots: BOOTSTRAP.contract.modelSlots }),
        catalog: SUBSCRIPTION.preview.available_subagents, agents: { snapshot: null, catalogKnown: false }, mainBinding: 'main-now',
        ...read });
    const step = createSummaryStep({ contract: BOOTSTRAP.contract, state, consciousness: createConsciousnessStep(offer),
        localSlot: (slot) => local.includes(slot), accessLabel: (mode) => ACCESS[mode] || 'Advanced',
        keyWords: (key) => (key === 'openaiKey' ? ['Responses'] : []), reads });
    return { step, state };
}

/** One part's rows as `[label, value]`, from the host that holds them; `null` when the page has no such part. */
function rows(html, part) {
    const body = html.match(new RegExp(`data-summary-part="${part}">((?:<div class="summary-kv">[\\s\\S]*?</div>)*)</div>`))?.[1];
    return body === undefined ? null : [...body.matchAll(/<strong>([^<]*)<\/strong><span>([^<]*)<\/span>/g)].map(([, label, value]) => [label, value]);
}
const labels = (list) => list.map(([label]) => label);

test('the choices lead in the rail\'s order, then the Skills row and the limits; every assignment waits behind one closed disclosure', () => {
    const html = summaryOf().step.html();
    assert.deepEqual(rows(html, 'choices'), [['Connected access', 'Codex subscription · no API key'], ['Main model', 'Codex · gpt-test'],
        ['Access level', 'Advanced'], ['Consciousness', 'On · Also on its own']]);
    assert.deepEqual(rows(html, 'limits'), [['When quota runs out', 'Waits for the reset, then continues']]);
    const assignments = rows(html, 'assignments');
    assert.deepEqual(labels(assignments), ['Main', 'Light', 'Safety Supervisor', 'Vision', 'Consciousness model', 'Fallback', 'Agents', 'Reviewers'],
        'every role Start saves, then the agents and the reviewers');
    assert.deepEqual(assignments[0], ['Main', 'Codex · gpt-test · Auto rotation · Processing: Route default (inherited)'], 'Main whole');
    assert.equal(assignments[2][1], 'Uses Light · Codex · gpt-test · Auto rotation · Processing: Route default (inherited)');
    assert.equal(assignments[7][1], 'codex=gpt-test · Effort: Auto · Processing: Route default (inherited)');
    // The parts stand in that order in one card: the Skills row (the starter list's own) between the choices and the
    // limits, the disclosure last, closed, with a summary a keyboard reaches and a screen reader names.
    assert.match(html, /<div class="summary-card">\s*<div class="summary-card-rows" data-summary-part="choices">[\s\S]*<\/div>\s*<div data-starter-skills="row"><\/div>\s*<div class="summary-card-rows" data-summary-part="limits">[\s\S]*<details class="ui-disclosure summary-assignments" data-summary-assignments>\s*<summary>Models, agents and reviewers<\/summary>/);
    assert.match(html, /<\/details>\s*<\/div>[\s\S]*?<div class="form-section" data-i18n-settings/, 'then the Language control');
});

test('the step says Start begins the first conversation only where a first state is still to come', () => {
    const first = summaryOf().step.html();
    const lead = BOOTSTRAP.consciousnessOffer.words.summaryLead;
    assert.match(lead, /begins your first conversation/);
    assert.ok(first.includes(`<p class="step-copy">${lead.replace(/'/g, '&#39;')}</p>`), first.match(/<p class="step-copy">[^<]*/)?.[0]);
    const summaryStep = BOOTSTRAP.contract.steps.find((step) => step.id === 'summary');
    for (const offer of [{ ...BOOTSTRAP.consciousnessOffer, offered: false }, null]) {
        const later = summaryOf({}, {}, { offer }).step.html();  // a state exists (or an older host): no promise, no row
        assert.ok(later.includes(`<p class="step-copy">${summaryStep.copy}</p>`));
        assert.doesNotMatch(later, /first conversation/);
        assert.equal(labels(rows(later, 'choices')).includes('Consciousness'), false);
    }
});

test('Main among the choices names its source, its model and a pinned account; its whole row keeps the rest', () => {
    const pinned = summaryOf({ modelAccounts: { main: 'work' }, modelContextWindows: { main: 200000 } }).step.html();
    assert.deepEqual(rows(pinned, 'choices')[1], ['Main model', 'Codex · gpt-test · Account: work']);
    assert.deepEqual(rows(pinned, 'assignments')[0],
        ['Main', 'Codex · gpt-test · Account: work · Processing: Route default (inherited) · 200,000 tokens, set by you']);
    // An API Main, with its key: the connection rows say the key, the limits say the money.
    const api = summaryOf({ mainModel: 'google/gemini-3.8-flash', lightModel: '', openaiKey: 'sk-test-not-a-secret', openrouterKey: 'sk-or-test' },
        { api: true, subscription: false, profile: 'openrouter' }).step.html();
    assert.deepEqual(rows(api, 'choices').slice(0, 4), [['Connected access', 'OpenRouter'], ['OpenAI', 'configured · Responses'],
        ['OpenRouter', 'configured'], ['Main model', 'OpenRouter · google/gemini-3.8-flash']]);
    assert.deepEqual(rows(api, 'limits'), [['Total API budget', '$10.00'], ['Per-task API cap', '$5.00']]);
    // A blank budget is said as unknown, never $0.00.
    assert.deepEqual(rows(summaryOf({ totalBudget: '' }, { api: true }).step.html(), 'limits')[0], ['Total API budget', '—']);
});

test('a repository model beside a subscription names the GGUF, not the cloud id, and does not use its quota', () => {
    const { step } = summaryOf({ localSource: 'fixture-labs/Local-GGUF', localFilename: 'quant/repository-test.gguf',
        localRoutingMode: 'all', mainModel: CODEX, lightModel: '', fallbackModel: CODEX },
        { local: true }, { local: ['main', 'light', 'consciousness', 'fallback'] });
    const html = step.html();
    assert.deepEqual(rows(html, 'choices').slice(0, 3), [['Connected access', 'Codex subscription · no API key'],
        ['Local source', 'fixture-labs/Local-GGUF / quant/repository-test.gguf'], ['Main model', 'Local · repository-test.gguf']]);
    const assignments = Object.fromEntries(rows(html, 'assignments'));
    assert.equal(assignments['Local routing'], 'All active models local');
    assert.match(assignments.Fallback, /^Local · repository-test\.gguf · Processing:/);
    assert.deepEqual(rows(html, 'limits'), [], 'no subscription serves a role and no API is set: no limit row');
    assert.match(html, /data-summary-part="limits"><\/div>/, 'an empty part is empty (its CSS takes no gap)');
});

test('every local or inherited role ignores the leftover repository filename for direct paths', () => {
    for (const source of ['/fixtures/example.gguf', '~/fixtures/example.gguf', 'W:/fixtures/example.gguf',
        String.raw`W:\fixtures\example.gguf`, String.raw`\\fixture-host\share\example.gguf`]) {
        const html = summaryOf({ agentsConnected: [], localSource: source, localFilename: 'quant/old-repo.gguf',
            localRoutingMode: 'all', mainModel: '', lightModel: '' }, { local: true, subscription: false, profile: 'local' },
        { local: ['main', 'light', 'consciousness', 'fallback'] }).step.html();
        assert.deepEqual(rows(html, 'choices').slice(0, 3), [['Connected access', 'Local-first'],
            ['Local source', source], ['Main model', 'Local · example.gguf']]);
        const assignments = Object.fromEntries(rows(html, 'assignments'));
        assert.match(assignments.Main, /^Local · example\.gguf/);
        for (const role of ['Light', 'Vision', 'Consciousness model']) assert.match(assignments[role], /^Uses Main · Local · example\.gguf/);
        assert.match(assignments['Safety Supervisor'], /^Uses Light · Local · example\.gguf/);
        assert.equal(assignments.Fallback, 'None');
        assert.doesNotMatch(html, /old-repo\.gguf/);
    }
});

test('the built-in OpenAI key names its legacy destination even when the compatible endpoint has its own address', () => {
    const legacy = 'https://legacy.example.invalid/v1';
    for (const compatibleBaseUrl of ['', 'https://dedicated.example.invalid/v1']) {
        const html = summaryOf({ openaiKey: '[fixture-key]', compatibleApiKey: '[fixture-compatible-key]',
            legacyCompatibleBaseUrl: legacy, compatibleBaseUrl }).step.html();
        assert.equal(Object.fromEntries(rows(html, 'choices')).OpenAI, `configured for the legacy address ${legacy} · Responses`);
        assert.doesNotMatch(html, /dedicated\.example|\[fixture-(?:compatible-)?key\]/);
    }
});

test('Safety inherits account, processing and a user-set window only as far as its effective model chain', () => {
    const draft = { lightModel: '', safetyModel: '', processingPreference: 'standard',
        modelAccounts: { main: 'fixture-studio' }, modelContextWindows: { main: 200000 },
        modelProcessingPreferences: { main: 'fast' } };
    const assignments = (state) => Object.fromEntries(rows(summaryOf(state).step.html(), 'assignments'));
    const throughMain = assignments(draft);
    assert.match(throughMain['Safety Supervisor'], /^Uses Light · Codex · gpt-test · Account: fixture-studio · Processing: Fast .* · 200,000 tokens, set by you$/);
    assert.match(throughMain.Light, /Auto rotation · Processing: Standard/, 'Light keeps its own account and tier scope');
    assert.match(assignments({ ...draft, lightModel: CODEX })['Safety Supervisor'],
        /^Uses Light · Codex · gpt-test · Auto rotation · Processing: Standard \(from Models\)$/, 'an explicit Light cuts off Main');
    assert.match(assignments({ ...draft, modelAccounts: { ...draft.modelAccounts, safety: 'checker' } })['Safety Supervisor'],
        /Account: checker · Processing: Fast/, 'a Safety pin wins without changing the inherited tier');
});

test('the assignments open by themselves when a note starts asking to check them, and otherwise stay as the owner left them', () => {
    const owner = { open: true, isConnected: true, listener: null, addEventListener(type, fn) { if (type === 'toggle') this.listener = fn; } };
    const root = { querySelector: (selector) => (selector === '[data-summary-assignments]' ? owner : null) };
    // A packet-only pool: its warning asks to check the reviewers, so they open once, the warning under the card.
    const packet = summaryOf({}, { catalog: PACKET }).step;
    let html = packet.html();
    assert.match(html, /data-summary-assignments open>/);
    assert.match(html, /<\/details>\s*<\/div>\s*<div class="wizard-inline-note" data-tone="warn" data-review-pool-warning>Every reviewer is a Packet row/);
    packet.bind(root);
    owner.open = false;
    owner.listener();  // the owner closes them
    html = packet.html();
    assert.doesNotMatch(html, /data-summary-assignments open/, 'the same note asking again does not reopen what the owner closed');
    assert.match(html, /data-review-pool-warning/, 'the warning itself stays');
    // A recovery the owner pressed: the note says what happened and asks to check; the assignments open with it.
    const { step, state } = summaryOf();
    assert.doesNotMatch(step.html(), /data-summary-assignments open|data-recovery-note>/);
    Object.assign(state, { recoveryPrepared: true, recoveryMain: 'main-now', skipSubscriptionPresets: true });
    html = step.html();
    assert.match(html, /data-summary-assignments open>/);
    assert.match(html, /<div class="wizard-inline-note" data-recovery-note>Automatic subscription presets were skipped\. Reviewers were assigned to Main\. Check the assignments/);
    assert.match(html, /data-recovery-note>[\s\S]*<div class="form-section" data-i18n-settings/, 'before the Language control');
    assert.match(rows(html, 'assignments').at(-2)[1], /automatic subscription preset skipped$/);
    state.recoveryMain = 'main-before';  // Main changed after the recovery
    assert.match(step.html(), /Main changed; reviewers keep the assignments shown above\./);
});

test('late reads paint the live warning and its rows without re-opening a manually closed disclosure', () => {
    const read = {}, { step, state } = summaryOf({}, read);
    step.html();
    const details = { open: false }, warning = { hidden: true, textContent: '' };
    const hosts = Object.fromEntries(['choices', 'limits', 'assignments'].map((part) => [part, { innerHTML: '' }]));
    const nodes = { '[data-summary-assignments]': details, '[data-review-pool-warning]': warning,
        ...Object.fromEntries(Object.entries(hosts).map(([part, host]) => [`[data-summary-part="${part}"]`, host])) };
    const root = { querySelector: (selector) => nodes[selector] };
    read.catalog = PACKET;
    state.fallbackModel = 'openai/gpt-5.6-luna';
    step.refresh(root);
    assert.equal(details.open, true);
    assert.equal(warning.hidden, false);
    assert.match(warning.textContent, /Every reviewer is a Packet row/);
    assert.match(hosts.assignments.innerHTML, /openai\/gpt-5\.6-sol/);
    assert.match(hosts.limits.innerHTML, /When quota runs out<\/strong><span>Tries the Fallback, then waits for the reset/);
    assert.match(hosts.assignments.innerHTML, /<strong>Fallback<\/strong><span>OpenRouter · openai\/gpt-5\.6-luna · Processing/);
    assert.match(hosts.choices.innerHTML, /<strong>Main model<\/strong><span>Codex · gpt-test<\/span>/);
    details.open = false;
    step.refresh(root);
    assert.equal(details.open, false);
    assert.equal(warning.hidden, false);
    delete read.catalog;
    step.refresh(root);
    assert.equal(warning.hidden, true);
    assert.equal(warning.textContent, '');
    assert.equal(details.open, false);
    details.open = true;
    step.refresh(root);
    assert.equal(details.open, true, 'a fresh read never closes a manually opened disclosure');
});
