// The "+" menu's contributed and host items (composer_plus.js): one section per skill in the owner's arrangement,
// "Add dictation…" only once the voice controls offered their chooser and while no engine is chosen, "Make a button
// with Ouroboros…" starting a request in the draft, and every item doing its verb on this room's draft.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createContributionsStore } from '../modules/contributions.js';
import { bindComposerPlus, MAKE_A_BUTTON_TEXT, setDictationOffer } from '../modules/composer_plus.js';

const item = (key, extra = {}) => {
    const [skill, id] = key.split(':');
    return { key, skill, skill_label: skill.toUpperCase(), id, label: id, title: '', icon: '', order: 0, revision: 'r1',
        render: { kind: 'declarative', url: '' }, placement: 'composer.plus',
        action: { verb: 'draft.insert', text: `${id} text` }, ...extra };
};

async function fixture(payload) {
    let current = payload;
    const store = createContributionsStore({ client: { uiContributions: async () => current } });
    await store.refresh();
    let text = '';
    const sections = [];
    const composer = {
        plusMenu: { setSections: (next) => sections.push(next) },
        getDraft: () => text,
        insertAtCursor: (words) => { text += words; },
        setDraft: (value) => { text = value; },
    };
    const runs = [];
    const dispose = bindComposerPlus({ composer, store, run: async (contribution, ctx) => {
        runs.push([contribution.key, ctx.draft.text()]);
        return true;
    } });
    const latest = () => sections.at(-1);
    return { store, composer, runs, dispose, latest, sections, draft: () => text,
        async publish(next) { current = next; await store.refresh(); } };
}

const labels = (sections) => sections.map((section) => [section.key, section.label || '',
    section.items.map((entry) => entry.label)]);

test('skills\' items come in one section each, in the owner\'s order, before the host\'s own items', async () => {
    const f = await fixture({
        contributions: [item('templates:simpler', { icon: 'sparkle' }), item('lists:add'), item('templates:shorter')],
        preferences: { contributions: { hidden: [], order: ['lists:add'] } },
    });
    try {
        assert.deepEqual(labels(f.latest()), [
            ['skill:lists', 'LISTS', ['add']],
            ['skill:templates', 'TEMPLATES', ['simpler', 'shorter']],
            ['host', 'Ouroboros', ['Make a button with Ouroboros…']],
        ]);
        assert.equal(f.latest()[1].items[0].icon, 'sparkle', 'a skill names its glyph from the host set');
        f.latest()[1].items[1].onSelect();
        await Promise.resolve();
        assert.deepEqual(f.runs, [['templates:shorter', '']], 'an item does its own verb on this room\'s draft');
        await f.publish({ contributions: [item('lists:add')] });
        assert.deepEqual(labels(f.latest()).map(([key]) => key), ['skill:lists', 'host'], 'a skill that left takes its section');
    } finally { f.dispose(); }
});

test('Make a button with Ouroboros… starts the request in the draft and sends nothing', async () => {
    const f = await fixture({ contributions: [] });
    try {
        const make = f.latest().at(-1).items.at(-1);
        assert.equal(make.key, 'make-button');
        make.onSelect();
        assert.equal(f.draft(), MAKE_A_BUTTON_TEXT);
        assert.equal(MAKE_A_BUTTON_TEXT, 'Make me a button in the + menu that ');
    } finally { f.dispose(); }
});

test('Add dictation… appears once the voice controls offer their chooser, and only while no engine is chosen', async () => {
    const f = await fixture({ contributions: [], voice: { stt: { key: '', available: false, reason: 'not_selected' } } });
    const hostLabels = () => f.latest().at(-1).items.map((entry) => entry.key);
    let unset = null;
    try {
        assert.deepEqual(hostLabels(), ['make-button'], 'absent until the voice controls offer a chooser');
        const opened = [];
        unset = setDictationOffer((ctx) => opened.push(ctx.composer === f.composer && typeof ctx.draft.insert));
        assert.deepEqual(hostLabels(), ['dictation', 'make-button']);
        assert.equal(f.latest().at(-1).items[0].label, 'Add dictation…');
        f.latest().at(-1).items[0].onSelect();
        assert.deepEqual(opened, ['function'], 'the chooser opens for this room');
        await f.publish({ contributions: [], voice: { stt: { key: 'whisper:stt', available: true, reason: '' } } });
        assert.deepEqual(hostLabels(), ['make-button'], 'a chosen engine needs no offer');
        unset();
        const custom = setDictationOffer({ open() {}, shown: () => true });
        assert.deepEqual(hostLabels(), ['dictation', 'make-button'], 'the voice controls may decide when it shows');
        custom();
        assert.deepEqual(hostLabels(), ['make-button']);
        unset = null;
    } finally {
        unset?.();
        f.dispose();
    }
    const count = f.sections.length;
    setDictationOffer(() => {})();
    assert.equal(f.sections.length, count, 'a disposed room no longer follows the offer');
});
