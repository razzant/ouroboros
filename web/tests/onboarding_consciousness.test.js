// The setup's Consciousness step (owner decision 2026-11-20): offered by the server, on by default with no state yet,
// off when a state says off, On already when it says on. Its position rides the completion request as `consciousness`;
// the page sends no command (the builtin page's only consciousness write is that field).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
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
    const facts = normalizeOffer({ offered: true, enabled: 'yes', autonomy: 'act', dailyUsd: '', maxTasks: 'x' });
    assert.deepEqual([facts.enabled, facts.dailyUsd, facts.maxTasks], [null, null, null]);
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

test('a state that says off shows off and stays off unless the owner turns it; one already on sends nothing', () => {
    const { step, on } = boundStep({ ...DEFAULTS, enabled: false });
    assert.deepEqual([step.on(), step.payload()], [false, { consciousness: false }]);
    on.click();
    assert.deepEqual(step.payload(), { consciousness: true });
    const already = createConsciousnessStep({ ...DEFAULTS, enabled: true });
    assert.deepEqual(already.payload(), {}, 'the wizard never turns it off, and an on needs no start');
    assert.deepEqual(already.summaryRows(), [['Consciousness', 'On already']]);
    assert.doesNotMatch(already.html(), /data-consciousness-choice/);
    assert.match(text(already.html()), /On already Also on its own/);
    assert.deepEqual(createConsciousnessStep({ ...DEFAULTS, offered: false }).payload(), {});
});

test('the step: title, the two positions, what they mean, the limits and the door; a subscription alone names its quota', () => {
    const facts = normalizeOffer(DEFAULTS);
    const html = stepHtml(facts, true);
    assert.match(html, /class="wizard-choice active" data-consciousness-choice="on" aria-pressed="true"/);
    assert.match(html, /class="wizard-choice " data-consciousness-choice="off" aria-pressed="false"/);
    assert.equal(text(html), `Consciousness When it acts. Off Only when you write On Also on its own Off: only when asked. On: also on its own. ${LIMITS} Change it later in Dashboard.`);
    assert.match(text(stepHtml(facts, true, { subscriptionOnly: true })), /also on its own\. A subscription pays with its quota\. Autonomy/);
    assert.deepEqual(limitPhrases(normalizeOffer({ ...DEFAULTS, autonomy: 'observe', dailyUsd: 5, maxTasks: 0 })), [
        'Autonomy: Advises. On its own it studies, keeps its memory and tells you; it changes nothing outside.',
        'Up to $5.00 a day in recorded spending. Subscription quota is separate.', 'It starts no tasks of its own.']);
    assert.deepEqual(limitPhrases(normalizeOffer({ ...DEFAULTS, autonomy: 'full', dailyUsd: 0, maxTasks: 1 })).slice(1), [
        'Daily allowance $0.00: it may not spend.', 'At most 1 of its own tasks at once.']);
    assert.deepEqual(limitPhrases(normalizeOffer({ offered: true })), [
        'Autonomy unknown.', 'Daily allowance unknown.', 'Its own-task limit is unknown.']);
});

test('the builtin page composes no consciousness command', () => {
    for (const file of ['../modules/onboarding_consciousness.js', '../modules/onboarding_wizard.js']) {
        const source = readFileSync(new URL(file, import.meta.url), 'utf8');
        assert.doesNotMatch(source, /\/bg|\/api\/command/, file);
    }
});
