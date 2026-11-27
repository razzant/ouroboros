// The setup's Consciousness step (owner decision 2026-11-20): offered by the server only before a first state exists,
// on by default. Its position rides the completion request as `consciousness`; no command is sent (the browser
// witness, tests/test_onboarding_consciousness_browser.py, records every request of the ordinary path).
import assert from 'node:assert/strict';
import test from 'node:test';

import { createConsciousnessStep, limitPhrases, normalizeOffer, stepHtml } from '../modules/onboarding_consciousness.js';

const text = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
// The host's words (consciousness_view.first_run_words): the step shows what it is given.
const WORDS = { gloss: 'When it acts.', off: 'Only when you write', on: 'Also on its own', lead: 'What should it do while you are away?',
    offAway: 'Away, it works on what you asked.', onAway: 'Away, it also wakes on its own.', meaning: 'Off: only when asked. On: also on its own.',
    subscriptionCost: 'A subscription pays with its quota.', door: 'Change it later in Dashboard.', summaryLead: 'Start also begins your first conversation.' };
const DEFAULTS = { offered: true, enabled: null, autonomy: 'act', dailyUsd: 20, maxTasks: 2, words: WORDS };
const AUTONOMY = 'Autonomy: Acts. Also acts on its own: starts work, runs commands, speaks and acts in places. It does not change itself.';
const LIMITS = `While it's on ${AUTONOMY} Up to $20.00 a day in recorded spending. Subscription quota is separate. At most 2 of its own tasks at once.`;

/** A root whose two choices record clicks, as the wizard's bound step sees them. */
function boundStep(offer) {
    const step = createConsciousnessStep(offer);
    const buttons = ['off', 'on'].map((value) => {
        const attrs = { 'data-consciousness-choice': value }; const listeners = [];
        return { attrs, listeners, classList: { toggle(name, on) { attrs[`class:${name}`] = on; } },
            getAttribute: (key) => attrs[key], setAttribute(key, v) { attrs[key] = v; }, addEventListener(_type, fn) { listeners.push(fn); },
            click() { listeners.forEach((fn) => fn()); } };
    });
    step.bind({ querySelectorAll: (selector) => (selector === '[data-consciousness-choice]' ? buttons : []) });
    return { step, off: buttons[0], on: buttons[1] };
}

test('an older bootstrap offers nothing and sends nothing; values keep unknown as unknown', () => {
    assert.equal(normalizeOffer(undefined).offered, false);
    const step = createConsciousnessStep(undefined);
    assert.deepEqual(step.payload(), {});
    assert.deepEqual(step.summaryRows(), []);
    assert.equal(step.summaryLead(), '', 'no first state is known to come: Summary promises no first conversation');
    const facts = normalizeOffer({ offered: true, autonomy: 'act', dailyUsd: '', maxTasks: 'x' });
    assert.deepEqual([facts.dailyUsd, facts.maxTasks], [null, null]);
});

test('no state yet: the switch starts on, its field says true, and Summary reads it', () => {
    const { step, off, on } = boundStep(DEFAULTS);
    assert.equal(step.on(), true);
    assert.deepEqual(step.payload(), { consciousness: true });
    assert.deepEqual(step.summaryRows(), [['Consciousness', 'On · Also on its own']]);
    assert.equal(step.summaryLead(), WORDS.summaryLead, 'a first state is to come: Start also begins the first conversation');
    off.click();  // the owner turns it off: the field says false (which changes nothing on the server)
    assert.deepEqual([step.on(), off.attrs['aria-pressed'], on.attrs['aria-pressed']], [false, 'true', 'false']);
    assert.deepEqual(step.payload(), { consciousness: false });
    assert.deepEqual(step.summaryRows(), [['Consciousness', 'Off · Only when you write']]);
    assert.equal(step.summaryLead(), WORDS.summaryLead, 'Start begins it in either position');
    on.click();
    assert.deepEqual(step.payload(), { consciousness: true });
});

test('a step the server does not offer (a state exists, whatever it says) shows and sends nothing', () => {
    for (const enabled of [false, true, null]) {
        const step = createConsciousnessStep({ ...DEFAULTS, offered: false, enabled });
        assert.deepEqual([step.payload(), step.summaryRows(), step.summaryLead()], [{}, [], ''], String(enabled));
    }
});

// Owner 2026-11-23 (the side-by-side, both screens as proposed): the step asks one question and each position says,
// under its name, what it does while the owner is away; what holds while it is on is said once under them.
test('the step: its question leads, each position says what it does while you are away, then what holds while it is on', () => {
    const facts = normalizeOffer(DEFAULTS);
    const html = stepHtml(facts, true);
    assert.match(html, /class="wizard-choice active" data-consciousness-choice="on" aria-pressed="true"/);
    assert.match(html, /class="wizard-choice " data-consciousness-choice="off" aria-pressed="false"/);
    assert.match(html, /<p class="step-copy">What should it do while you are away\?<\/p>/, 'the question leads at body size');
    assert.match(html, /data-consciousness-choice="off"[^>]*>\s*<span class="tone">Off<\/span><h3>Only when you write<\/h3><p>Away, it works on what you asked\.<\/p>\s*<\/button>/,
        'Off says its consequence inside its own choice');
    assert.match(html, /data-consciousness-choice="on"[^>]*>\s*<span class="tone">On<\/span><h3>Also on its own<\/h3><p>Away, it also wakes on its own\.<\/p>\s*<\/button>/,
        'On says its consequence inside its own choice');
    assert.match(html, /role="group" aria-label="When it acts\."/);
    assert.doesNotMatch(html, /only when asked/, 'the whole paragraph is not said beside its parts');
    assert.equal(text(html), `Consciousness What should it do while you are away? Off Only when you write Away, it works on what you asked. `
        + `On Also on its own Away, it also wakes on its own. ${LIMITS} Change it later in Dashboard.`);
    // A host that names no position's consequence (one built before them) keeps its one paragraph as the lead.
    const older = normalizeOffer({ ...DEFAULTS, words: { ...WORDS, lead: '', offAway: '', onAway: '' } });
    const kept = stepHtml(older, true);
    assert.match(kept, /<p class="step-copy">Off: only when asked\. On: also on its own\.<\/p>/);
    assert.match(kept, /<h3>Only when you write<\/h3>\s*<\/button>/, 'no empty line inside a choice');
});

test('a subscription alone names its quota once, beside the allowance; a $0.00 allowance says it does not wake', () => {
    const facts = normalizeOffer(DEFAULTS);
    const paid = text(stepHtml(facts, true, { subscriptionOnly: true }));
    assert.match(paid, /While it's on Autonomy: .* Up to \$20\.00 a day in recorded spending\. A subscription pays with its quota\. At most 2/,
        'the money and the quota said together, once');
    assert.equal(paid.match(/quota/g).length, 1, 'the limits leave the quota to the subscription sentence');
    const html = stepHtml(facts, true, { subscriptionOnly: true });
    assert.match(html, /<p class="consciousness-step-note">Up to \$20\.00 a day in recorded spending\. A subscription pays with its quota\.<\/p>/,
        'one line holds both');
    // No wakes, nothing of the quota they would spend.
    const idle = text(stepHtml(normalizeOffer({ ...DEFAULTS, dailyUsd: 0 }), true, { subscriptionOnly: true }));
    assert.match(idle, /Daily allowance \$0\.00: it does not wake on its own\. At most 2/);
    assert.doesNotMatch(idle, /quota/);
    assert.deepEqual(limitPhrases(normalizeOffer({ ...DEFAULTS, autonomy: 'observe', dailyUsd: 5, maxTasks: 0 })), [
        'Autonomy: Advises. On its own it studies, keeps its memory and tells you; it changes nothing outside.',
        'Up to $5.00 a day in recorded spending. Subscription quota is separate.', 'It starts no tasks of its own.']);
    assert.deepEqual(limitPhrases(normalizeOffer({ ...DEFAULTS, autonomy: 'full', dailyUsd: 0, maxTasks: 1 }), { subscriptionOnly: true }).slice(1), [
        'Daily allowance $0.00: it does not wake on its own.', 'At most 1 of its own tasks at once.']);
    assert.deepEqual(limitPhrases(normalizeOffer({ offered: true })), [
        'Autonomy unknown.', 'Daily allowance unknown.', 'Its own-task limit is unknown.']);
});
