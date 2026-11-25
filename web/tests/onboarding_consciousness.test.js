// The setup's Consciousness step (owner decision 2026-11-20): offered by the server only before a first state exists,
// on by default. Its position rides the completion request as `consciousness`; no command is sent (the browser
// witness, tests/test_onboarding_consciousness_browser.py, records every request of the ordinary path).
import assert from 'node:assert/strict';
import test from 'node:test';

import { createConsciousnessStep, limitPhrases, normalizeOffer, stepHtml } from '../modules/onboarding_consciousness.js';

const text = (html) => html.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
// The host's words (consciousness_view.first_run_words): the step shows what it is given.
const WORDS = { gloss: 'When it acts.', off: 'Only when you write', on: 'Also on its own', meaning: 'Off: only when asked. On: also on its own.',
    subscriptionCost: 'A subscription pays with its quota.', door: 'Change it later in Dashboard.' };
const DEFAULTS = { offered: true, enabled: null, autonomy: 'act', dailyUsd: 20, maxTasks: 2, words: WORDS };
const LIMITS = 'Autonomy: Acts. Also acts on its own: starts work, runs commands, speaks and acts in places. It does not change itself. '
    + 'Up to $20.00 a day in recorded spending. Subscription quota is separate. At most 2 of its own tasks at once.';

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
    const facts = normalizeOffer({ offered: true, autonomy: 'act', dailyUsd: '', maxTasks: 'x' });
    assert.deepEqual([facts.dailyUsd, facts.maxTasks], [null, null]);
});

test('no state yet: the switch starts on, its field says true, and Summary reads it', () => {
    const { step, off, on } = boundStep(DEFAULTS);
    assert.equal(step.on(), true);
    assert.deepEqual(step.payload(), { consciousness: true });
    assert.deepEqual(step.summaryRows(), [['Consciousness', 'On · Also on its own']]);
    off.click();  // the owner turns it off: the field says false (which changes nothing on the server)
    assert.deepEqual([step.on(), off.attrs['aria-pressed'], on.attrs['aria-pressed']], [false, 'true', 'false']);
    assert.deepEqual(step.payload(), { consciousness: false });
    assert.deepEqual(step.summaryRows(), [['Consciousness', 'Off · Only when you write']]);
    on.click();
    assert.deepEqual(step.payload(), { consciousness: true });
});

test('a step the server does not offer (a state exists, whatever it says) shows and sends nothing', () => {
    for (const enabled of [false, true, null]) {
        const step = createConsciousnessStep({ ...DEFAULTS, offered: false, enabled });
        assert.deepEqual([step.payload(), step.summaryRows()], [{}, []], String(enabled));
    }
});

test('the step: title, what the positions mean as its lead, the two positions, the limits and the door', () => {
    const facts = normalizeOffer(DEFAULTS);
    const html = stepHtml(facts, true);
    assert.match(html, /class="wizard-choice active" data-consciousness-choice="on" aria-pressed="true"/);
    assert.match(html, /class="wizard-choice " data-consciousness-choice="off" aria-pressed="false"/);
    assert.match(html, /<p class="step-copy">Off: only when asked\. On: also on its own\.<\/p>/, 'the meaning leads at body size');
    assert.match(html, /role="group" aria-label="When it acts\."/);
    assert.equal(text(html), `Consciousness Off: only when asked. On: also on its own. Off Only when you write On Also on its own ${LIMITS} Change it later in Dashboard.`);
});

test('a subscription alone names its quota once; a $0.00 allowance says it does not wake', () => {
    const facts = normalizeOffer(DEFAULTS);
    const paid = text(stepHtml(facts, true, { subscriptionOnly: true }));
    assert.match(paid, /Also on its own A subscription pays with its quota\. Autonomy/);
    assert.equal(paid.match(/quota/g).length, 1, 'the limits leave the quota to the subscription sentence');
    assert.match(paid, /Up to \$20\.00 a day in recorded spending\. At most 2/);
    assert.deepEqual(limitPhrases(normalizeOffer({ ...DEFAULTS, autonomy: 'observe', dailyUsd: 5, maxTasks: 0 })), [
        'Autonomy: Advises. On its own it studies, keeps its memory and tells you; it changes nothing outside.',
        'Up to $5.00 a day in recorded spending. Subscription quota is separate.', 'It starts no tasks of its own.']);
    assert.deepEqual(limitPhrases(normalizeOffer({ ...DEFAULTS, autonomy: 'full', dailyUsd: 0, maxTasks: 1 }), { subscriptionOnly: true }).slice(1), [
        'Daily allowance $0.00: it does not wake on its own.', 'At most 1 of its own tasks at once.']);
    assert.deepEqual(limitPhrases(normalizeOffer({ offered: true })), [
        'Autonomy unknown.', 'Daily allowance unknown.', 'Its own-task limit is unknown.']);
});
