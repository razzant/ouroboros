// Contributed chips (composer_chips.js): the value a source answers as its chip shows it, the page's one reader of
// sources (its own refresh of 15 seconds at least, only while a room shows chips on a visible page, backing off after
// failures, never a hidden chip), and one room's chips: words, tone and description, a press doing the chip's action
// on this room's draft, and the chips that do not fit folding into one "+N" whose list holds them.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createContributionsStore } from '../modules/contributions.js';
import { bindComposerChips, chipFace, chipValue, createChipFeed } from '../modules/composer_chips.js';
import { installDomDouble } from './dom_double.js';

const chip = (key, extra = {}) => {
    const [skill, id] = key.split(':');
    return { key, skill, skill_label: skill, id, label: skill.charAt(0).toUpperCase() + skill.slice(1), title: '', icon: '',
        order: 0, revision: 'r1', render: { kind: 'declarative', url: '' }, placement: 'composer.chips',
        source: { url: `/api/extensions/${skill}/chip`, refresh_s: 60 }, action: { verb: 'open.tab' }, ...extra };
};

async function storeOf(payload) {
    const box = { payload };
    const store = createContributionsStore({ client: { uiContributions: async () => box.payload } });
    await store.refresh();
    return { store, async publish(next) { box.payload = next; await store.refresh(); } };
}

const flush = async () => { for (let i = 0; i < 5; i += 1) await Promise.resolve(); };

test('a source\'s answer as its chip shows it: 24 characters, a known tone, the rest described', () => {
    assert.deepEqual(chipValue({ text: ' 38% left ', tone: 'warn', title: 'Codex weekly window' }),
        { text: '38% left', full: '38% left', tone: 'warn', title: 'Codex weekly window' });
    const long = chipValue({ text: 'Twenty-five characters!!!', tone: 'loud' });
    assert.equal(Array.from(long.text).length, 24);
    assert.equal(long.text.endsWith('…'), true);
    assert.equal(long.tone, '', 'an unknown tone is no tone');
    assert.equal(Array.from(chipValue({ text: '🙂'.repeat(30) }).text).length, 24, 'characters, not code units');
    assert.deepEqual(chipValue(null), { text: '', full: '', tone: '', title: '' });
    const item = chip('quota:chip', { title: 'Usage of the accounts' });
    assert.deepEqual(chipFace({ item, value: long, error: '' }).title, 'Twenty-five characters!!!', 'the cut words describe it');
    assert.deepEqual(chipFace({ item, value: null, error: '' }), { text: '', tone: '', title: 'Usage of the accounts' });
    assert.deepEqual(chipFace({ item, value: long, error: 'HTTP 502' }),
        { text: '—', tone: 'muted', title: 'Quota could not be read: HTTP 502' });
});

test('the feed reads only while a room shows chips on a visible page, on each chip\'s own refresh of 15 s at least', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 1_000_000 });
    const { store, publish } = await storeOf({ contributions: [
        chip('quota:chip', { source: { url: '/q', refresh_s: 5 } }),
        chip('usage:chip', { source: { url: '/u', refresh_s: 60 } }),
        chip('hidden:chip', { source: { url: '/h', refresh_s: 15 } }),
    ], preferences: { contributions: { hidden: ['hidden:chip'], order: [] } } });
    const doc = Object.assign(new EventTarget(), { visibilityState: 'visible' });
    const reads = [];
    const feed = createChipFeed({ store, doc, now: () => Date.now(),
        read: async (url) => { reads.push(url); return { text: url }; } });
    let changes = 0;
    feed.onChange(() => { changes += 1; });
    t.mock.timers.tick(60_000);
    await flush();
    assert.deepEqual(reads, [], 'no room shows chips: nothing is read');
    const release = feed.watch();
    t.mock.timers.tick(0);
    await flush();
    assert.deepEqual(reads.sort(), ['/q', '/u'], 'shown now: each read at once; a hidden chip never');
    assert.deepEqual(feed.chips().map((entry) => entry.value.text), ['/q', '/u']);
    assert.ok(changes >= 2);
    reads.length = 0;
    t.mock.timers.tick(14_999);
    await flush();
    assert.deepEqual(reads, [], 'a 5 s refresh waits 15 s');
    t.mock.timers.tick(1);
    await flush();
    assert.deepEqual(reads, ['/q']);
    doc.visibilityState = 'hidden';
    doc.dispatchEvent(new Event('visibilitychange'));
    t.mock.timers.tick(120_000);
    await flush();
    assert.deepEqual(reads, ['/q'], 'a hidden page reads nothing');
    doc.visibilityState = 'visible';
    doc.dispatchEvent(new Event('visibilitychange'));
    t.mock.timers.tick(0);
    await flush();
    assert.deepEqual(reads.sort(), ['/q', '/q', '/u'], 'back on screen, the overdue chips read at once');
    release();
    reads.length = 0;
    t.mock.timers.tick(600_000);
    await flush();
    assert.deepEqual(reads, [], 'released: no room shows chips any more');
    await publish({ contributions: [chip('usage:chip', { source: { url: '/u2', refresh_s: 60 } })] });
    assert.deepEqual(feed.chips().map((entry) => [entry.item.key, entry.value]), [['usage:chip', null]],
        'a changed source starts over; a chip that left is gone');
    feed.destroy();
});

test('a failing source backs off doubling its refresh, shows a muted dash with the reason, and recovers', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 2_000_000 });
    const { store } = await storeOf({ contributions: [chip('quota:chip', { source: { url: '/q', refresh_s: 15 } })] });
    let fail = true;
    const at = [];
    const feed = createChipFeed({ store, doc: Object.assign(new EventTarget(), { visibilityState: 'visible' }),
        now: () => Date.now(), read: async () => {
            at.push(Date.now());
            if (fail) throw new Error('HTTP 502');
            return { text: '38%', tone: 'ok' };
        } });
    // The next read comes exactly `seconds` later, not a millisecond sooner.
    const readsAfter = async (seconds) => {
        const before = at.length;
        t.mock.timers.tick(seconds * 1000 - 1);
        await flush();
        assert.equal(at.length, before, `nothing before ${seconds} s`);
        t.mock.timers.tick(1);
        await flush();
        assert.equal(at.length, before + 1, `one read at ${seconds} s`);
    };
    feed.watch();
    t.mock.timers.tick(0);
    await flush();
    assert.equal(at.length, 1);
    for (const seconds of [30, 60, 120, 240, 240]) await readsAfter(seconds);  // doubling up to sixteen refreshes
    assert.deepEqual(chipFace(feed.chips()[0]), { text: '—', tone: 'muted', title: 'Quota could not be read: HTTP 502' });
    fail = false;
    await readsAfter(240);
    assert.deepEqual(chipFace(feed.chips()[0]), { text: '38%', tone: 'ok', title: '' }, 'one good read clears it');
    await readsAfter(15);  // and the refresh is its own again
    feed.destroy();
});

function roomFixture({ room: initialRoom = 300, widths = {} } = {}) {
    let room = initialRoom;
    const env = installDomDouble();
    const { doc, el } = env;
    const slot = el('span', { className: 'chat-composer-chips', attrs: { 'data-composer-chips': '' } });
    slot.hidden = true;
    doc.body.append(el('div', { className: 'chat-toolbar-row' }, slot));
    const width = (node) => (node.hidden ? 0 : widths[node.dataset.chipKey] ?? (node.classList.contains('chat-chip-more') ? 30 : 80));
    for (const prop of ['offsetWidth']) {
        Object.defineProperty(Object.getPrototypeOf(slot), prop, { configurable: true, get() { return width(this); } });
    }
    Object.defineProperty(slot, 'clientWidth', { get: () => (slot.hidden ? 0 : room) });
    Object.defineProperty(slot, 'scrollWidth', { get: () => slot.children.reduce((sum, child) => sum + width(child), 0) });
    let text = '';
    const composer = { chipsSlot: slot, getDraft: () => text, insertAtCursor: (words) => { text += words; }, setDraft: (v) => { text = v; } };
    const lists = [];
    const menu = ({ button }) => {
        const list = { button, sections: [], destroyed: false, setSections(next) { this.sections = next; }, destroy() { this.destroyed = true; } };
        lists.push(list);
        return list;
    };
    return { ...env, slot, composer, menu, lists, setRoom: (next) => { room = next; },
        chipsShown: () => slot.children.filter((node) => !node.hidden
        && !node.classList.contains('chat-chip-more')).map((node) => node.dataset.chipKey),
    more: () => slot.children.find((node) => node.classList.contains('chat-chip-more')),
    restore() { delete Object.getPrototypeOf(slot).offsetWidth; env.restore(); } };
}

function fakeFeed(entries) {
    const listeners = new Set();
    let watching = 0;
    return { entries, chips: () => entries, onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },
        watch() { watching += 1; return () => { watching -= 1; }; }, watching: () => watching,
        emit() { for (const fn of [...listeners]) fn(); } };
}

test('a room shows its chips after the capsules: the name, the value in its tone, the description; a press acts here', () => {
    const f = roomFixture();
    try {
        const feed = fakeFeed([
            { item: chip('quota:chip', { title: 'Accounts usage' }), value: chipValue({ text: '38%', tone: 'warn' }), error: '' },
            { item: chip('usage:chip'), value: null, error: 'HTTP 502' },
        ]);
        const runs = [];
        let visible = null;
        const dispose = bindComposerChips({ composer: f.composer, menu: f.menu, feed,
            run: async (item, ctx) => { runs.push([item.key, typeof ctx.draft.insert]); return true; },
            observe: (node, onChange) => { visible = onChange; return () => { visible = null; }; }, resize: () => () => {} });
        assert.equal(f.slot.hidden, false);
        assert.deepEqual(f.chipsShown(), ['quota:chip', 'usage:chip']);
        const [quota, usage] = f.slot.children;
        assert.deepEqual(quota.children.map((part) => part.textContent), ['Quota', '', '38%']);
        assert.equal(quota.dataset.tone, 'warn');
        assert.equal(quota.title, 'Accounts usage');
        assert.equal(quota.getAttribute('aria-label'), 'Quota 38%', 'two words to a screen reader, not "Quota38%"');
        assert.equal(quota.children[1].getAttribute('aria-hidden'), 'true', 'the dot is the tone, the words say it');
        assert.deepEqual([usage.children[2].textContent, usage.dataset.tone, usage.title],
            ['—', 'muted', 'Usage could not be read: HTTP 502']);
        quota.click();
        assert.deepEqual(runs, [['quota:chip', 'function']]);
        assert.equal(feed.watching(), 0, 'not read before the room is on screen');
        visible(true);
        assert.equal(feed.watching(), 1);
        visible(false);
        assert.equal(feed.watching(), 0, 'off screen, the room stops asking');
        dispose();
        assert.equal(f.slot.children.length, 0);
        assert.equal(f.slot.hidden, true);
        assert.equal(f.lists[0].destroyed, true);
    } finally { f.restore(); }
});

test('chips that do not fit fold into one "+N" whose list holds them; room again unfolds them', () => {
    const f = roomFixture({ room: 200 });
    try {
        const entries = ['a', 'b', 'c', 'd', 'e', 'f'].map((name) => ({ item: chip(`${name}:chip`),
            value: chipValue({ text: `${name}1` }), error: '' }));
        const feed = fakeFeed(entries);
        const runs = [];
        const dispose = bindComposerChips({ composer: f.composer, menu: f.menu, feed,
            run: async (item) => { runs.push(item.key); return true; }, observe: () => () => {}, resize: () => () => {} });
        // 200 px: "+N" (30) and two 80 px chips fit; four fold.
        assert.deepEqual(f.chipsShown(), ['a:chip', 'b:chip']);
        assert.equal(f.more().hidden, false);
        assert.equal(f.more().textContent, '+4');
        assert.equal(f.more().getAttribute('aria-label'), '4 more');
        assert.equal(f.more().title, 'C, D, E, F', 'its description names what it holds');
        assert.equal(f.slot.dataset.folded, '1', 'folded, the slot takes the room the capsules leave');
        const listed = f.lists[0].sections[0].items;
        assert.deepEqual(listed.map((entry) => entry.label), ['C c1', 'D d1', 'E e1', 'F f1']);
        listed[1].onSelect();
        assert.deepEqual(runs, ['d:chip'], 'a folded chip acts from the list');
        entries.splice(2);
        feed.emit();
        assert.deepEqual(f.chipsShown(), ['a:chip', 'b:chip']);
        assert.equal(f.more().hidden, true, 'everything fits again: no "+N"');
        assert.equal(f.slot.dataset.folded, undefined);
        dispose();
    } finally { f.restore(); }
});

test('with no room even for "+N", the chips are listed in "+" under their skill; room again brings them back', async () => {
    const f = roomFixture({ room: 2 });  // a phone's row: ☰ and the host capsules leave 2 px
    try {
        const entries = ['a', 'b'].map((name) => ({ item: { ...chip(`${name}:chip`), skill: name, skill_label: name.toUpperCase() },
            value: chipValue({ text: `${name}1` }), error: '' }));
        const feed = fakeFeed(entries);
        const runs = [];
        const plus = { sections: [], setSections(next) { this.sections = next; } };
        f.composer.plusMenu = plus;
        const { bindComposerPlus } = await import('../modules/composer_plus.js');
        const store = { byPlacement: () => [], voice: () => ({ key: 'x' }), onChange: () => () => {} };
        const unbindPlus = bindComposerPlus({ composer: f.composer, store });
        const dispose = bindComposerChips({ composer: f.composer, menu: f.menu, feed,
            run: async (item) => { runs.push(item.key); return true; }, observe: () => () => {}, resize: () => () => {} });
        const chipSections = () => plus.sections.filter((section) => section.key.startsWith('skill:'))
            .map((section) => [section.label, section.items.map((item) => item.label)]);
        assert.deepEqual(f.chipsShown(), []);
        assert.equal(f.more().hidden, true, 'a clipped "+N" would hide them: none is drawn');
        assert.equal(f.slot.dataset.roomless, '1');
        assert.deepEqual(chipSections(), [['A', ['A a1']], ['B', ['B b1']]]);
        plus.sections.find((section) => section.key === 'skill:b').items[0].onSelect();
        assert.deepEqual(runs, ['b:chip'], 'a chip listed in "+" acts here');
        entries.splice(1);
        feed.emit();
        assert.deepEqual(chipSections(), [['A', ['A a1']]]);
        f.setRoom(300);  // a wider row (a rotated phone)
        feed.emit();
        assert.deepEqual(f.chipsShown(), ['a:chip'], 'room again: the chip is back in the row');
        assert.equal(f.slot.dataset.roomless, undefined);
        assert.deepEqual(chipSections(), [], 'and no longer in "+"');
        dispose();
        assert.deepEqual(chipSections(), [], 'an unbound room leaves nothing in "+"');
        unbindPlus();
    } finally { f.restore(); }
});

test('a room that shows no chips leaves the slot as it found it', () => {
    const f = roomFixture();
    try {
        f.slot.hidden = false;  // another control filled it
        const dispose = bindComposerChips({ composer: f.composer, menu: f.menu, feed: fakeFeed([]),
            observe: () => () => {}, resize: () => () => {} });
        assert.equal(f.slot.hidden, false);
        dispose();
        assert.equal(f.slot.hidden, false);
    } finally { f.restore(); }
});
