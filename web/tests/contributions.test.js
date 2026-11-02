// The contributions store (contributions.js): one read at a time and the last good snapshot kept, the selectors per
// placement and provider kind with the owner's arrangement and the page's fault book, the read triggers, and the
// verbs a placed control performs (none of which sends anything as the owner).
import assert from 'node:assert/strict';
import test from 'node:test';
import { createContributionsStore, draftPort, runContributionAction } from '../modules/contributions.js';

const item = (key, extra = {}) => {
    const [skill, id] = key.split(':');
    return { key, skill, skill_label: skill.toUpperCase(), id, label: id, title: '', icon: '', order: 0, revision: 'r1',
        render: { kind: 'declarative', url: '' }, ...extra };
};

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function deferred() {
    let resolve, reject;
    const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
    return { promise, resolve, reject };
}

function client(queue) {
    const calls = [];
    return { calls, uiContributions: () => { calls.push(1); const next = queue.shift(); return next instanceof Error
        ? Promise.reject(next) : (next?.promise || Promise.resolve(next)); } };
}

test('overlapping triggers share one read and run it once more; a failed read keeps the last good snapshot', async () => {
    const first = deferred();
    const reads = client([first, { contributions: [item('notes:save', { placement: 'message.action' })] },
        new Error('offline')]);
    const store = createContributionsStore({ client: reads });
    let changes = 0;
    store.onChange(() => { changes += 1; });
    const a = store.refresh();
    const b = store.refresh();
    const c = store.refresh();
    assert.equal(a, b, 'a trigger during a read joins it');
    assert.equal(b, c);
    first.resolve({ contributions: [] });
    assert.equal(await a, true);
    assert.equal(reads.calls.length, 2, 'the triggers that landed mid-read cost exactly one more read');
    assert.deepEqual(store.byPlacement('message.action').map((row) => row.key), ['notes:save']);
    assert.equal(changes, 1, 'one change notice per settled read');
    assert.equal(await store.refresh(), false);
    assert.deepEqual(store.byPlacement('message.action').map((row) => row.key), ['notes:save'], 'the last good snapshot stays');
    assert.match(store.status().error, /offline/);
    assert.equal(changes, 1, 'a failed read changes nothing');
});

test('an unchanged projection notifies nobody; the first read always does, even when empty', async () => {
    const store = createContributionsStore({ client: client([{}, {}]) });
    let changes = 0;
    store.onChange(() => { changes += 1; });
    assert.equal(await store.refresh(), true);
    assert.equal(store.status().loaded, true);
    assert.equal(await store.refresh(), false);
    assert.equal(changes, 1);
});

test('placements follow the owner order, leave out hidden and faulted items; providers keep an off one with its reason', async () => {
    const host = item('voice:tts', { provides: 'tts', runtime: 'client', render: { kind: 'host', url: '/api/extensions/voice/ui/r1/ui/tts.js' } });
    const store = createContributionsStore({ client: client([{
        contributions: [item('a:one', { placement: 'composer.plus' }), item('b:two', { placement: 'composer.plus' }),
            item('c:three', { placement: 'composer.plus' }), item('d:four', { placement: 'composer.chips' }), host,
            item('w:stt', { provides: 'stt', runtime: 'server' })],
        offers: { stt: [{ skill: 'browser_speech', id: 'stt', enabled: false }] },
        voice: { stt: { key: 'w:stt', available: true, reason: '' } },
        preferences: { contributions: { hidden: ['b:two'], order: ['c:three'] } },
    }]) });
    await store.refresh();
    assert.deepEqual(store.byPlacement('composer.plus').map((row) => row.key), ['c:three', 'a:one'],
        'the owner order first, hidden left out, the rest in projection order');
    assert.deepEqual(store.byPlacement('composer.chips').map((row) => row.key), ['d:four']);
    assert.deepEqual(store.hostEntries().map((row) => row.key), ['voice:tts']);
    assert.deepEqual(store.providers('stt').map((row) => row.key), ['w:stt']);
    assert.deepEqual(store.offers('stt').map((row) => row.skill), ['browser_speech']);
    assert.deepEqual(store.offers('tts'), []);
    assert.equal(store.voice('stt').key, 'w:stt');
    store.markFault(store.byKey('a:one'), 'it threw');
    store.markFault(host, 'no voices');
    assert.deepEqual(store.byPlacement('composer.plus').map((row) => row.key), ['c:three'], 'an off control is not drawn');
    assert.deepEqual(store.providers('tts').map((row) => row.key), ['voice:tts'], 'an off provider is still listed');
    assert.equal(store.fault('voice:tts'), 'no voices', 'with the reason it is off');
    assert.deepEqual(store.faults().map((row) => [row.key, row.reason, row.current]),
        [['a:one', 'it threw', true], ['voice:tts', 'no voices', true]]);
});

test('a fault belongs to the revision that failed: the skill\'s next revision is back on', async () => {
    const store = createContributionsStore({ client: client([
        { contributions: [item('a:one', { placement: 'composer.plus' })] },
        { contributions: [item('a:one', { placement: 'composer.plus' })] },
        { contributions: [item('a:one', { placement: 'composer.plus', revision: 'r2' })] },
    ]) });
    await store.refresh();
    store.markFault(store.byKey('a:one'), 'it threw');
    await store.refresh();
    assert.equal(store.fault('a:one'), 'it threw', 'a re-read of the same revision keeps it off');
    assert.deepEqual(store.byPlacement('composer.plus'), []);
    await store.refresh();
    assert.equal(store.fault('a:one'), '');
    assert.deepEqual(store.byPlacement('composer.plus').map((row) => row.key), ['a:one']);
    assert.equal(store.faults()[0].current, false, 'the record stays for Settings and Skills');
});

test('attach reads at boot, on every socket open, on extension_lifecycle, on a finished lifecycle job and on the page\'s skill lifecycle', async () => {
    const reads = client([{}, {}, {}, {}, {}, {}]);
    const store = createContributionsStore({ client: reads });
    const handlers = new Map();
    const ws = { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); } };
    const target = new EventTarget();
    const detach = store.attach({ ws, target });
    await tick();
    assert.equal(reads.calls.length, 1, 'the boot read');
    handlers.get('open')();
    await tick();
    handlers.get('extension_lifecycle')({ type: 'extension_lifecycle', skill: 'x' });
    await tick();
    target.dispatchEvent(new Event('ouro:skill-lifecycle'));
    await tick();
    assert.equal(reads.calls.length, 4, 'one read per trigger');
    // An install from Browse fires no page event: the end of its lifecycle job (a progress row) is the trigger.
    const row = (phase, extra = {}) => ({ type: 'chat', role: 'system', is_progress: true, system_type: 'skill_lifecycle',
        lifecycle: { kind: 'install', target: 'browser_speech', phase }, ...extra });
    handlers.get('chat')(row('running'));
    handlers.get('chat')({ type: 'chat', role: 'assistant', content: 'completed' });
    await tick();
    assert.equal(reads.calls.length, 4, 'a running job and an ordinary message read nothing');
    handlers.get('chat')(row('completed'));
    await tick();
    assert.equal(reads.calls.length, 5, 'a finished job reads the projection again');
    detach();
    assert.equal(handlers.size, 0, 'the socket subscriptions are released');
    target.dispatchEvent(new Event('ouro:skill-lifecycle'));
    assert.equal(reads.calls.length, 5, 'and the page event no longer reads');
});

test('onLifecycle hears each lifecycle end, changed projection or not; never a subscription, the boot read or a socket open', async () => {
    const reads = client([{}, {}, {}, {}, {}, {}, {}]);
    const store = createContributionsStore({ client: reads });
    const handlers = new Map();
    const ws = { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); } };
    const target = new EventTarget();
    let changes = 0, ends = 0;
    store.onChange(() => { changes += 1; });
    const off = store.onLifecycle(() => { ends += 1; });
    store.attach({ ws, target });
    await tick();
    handlers.get('open')();
    await tick();
    assert.equal(ends, 0, 'subscribing, the boot read and a socket open are no lifecycle end');
    const row = (phase) => ({ type: 'chat', role: 'system', is_progress: true, system_type: 'skill_lifecycle',
        lifecycle: { kind: 'update', target: 'slack-bridge', phase } });
    handlers.get('chat')(row('running'));
    handlers.get('chat')({ type: 'chat', role: 'assistant', content: 'completed' });
    assert.equal(ends, 0, 'a running job and an ordinary message are none either');
    // An off connector's update: the projection leaves the skill out, so nothing changed for onChange.
    handlers.get('chat')(row('completed'));
    await tick();
    assert.equal(ends, 1);
    assert.equal(changes, 1, 'only the boot read changed the projection');
    handlers.get('extension_lifecycle')({ type: 'extension_lifecycle', skill: 'x' });
    target.dispatchEvent(new Event('ouro:skill-lifecycle'));
    assert.equal(ends, 3, 'every trigger the store reads on, except a socket open');
    off();
    handlers.get('chat')(row('failed'));
    assert.equal(ends, 3, 'the disposer stops it');
    store.onLifecycle(() => { ends += 1; });
    store.destroy();
    assert.equal(handlers.size, 0);
    assert.equal(ends, 3, 'destroy releases the socket and its listeners');
});

function room(initial = '') {
    let text = initial;
    const composer = {
        getDraft: () => text,
        insertAtCursor(words) { text += words; },
        setDraft(value) { text = value; },
    };
    return { composer, get text() { return text; }, set text(value) { text = value; } };
}

test('draft.insert puts words into the visible room\'s draft and never sends; a room without a composer says so', async () => {
    const notices = [];
    const notice = (text, tone) => notices.push([text, tone]);
    const target = room('Hi. ');
    const insert = item('t:simpler', { placement: 'composer.plus', action: { verb: 'draft.insert', text: 'Explain simpler.' } });
    assert.equal(await runContributionAction(insert, { draft: draftPort(() => target), notice }), true);
    assert.equal(target.text, 'Hi. Explain simpler.');
    assert.deepEqual(notices, []);
    assert.equal(await runContributionAction(insert, { draft: draftPort(() => null), notice }), false);
    assert.deepEqual(notices, [['There is no message field here to put this in.', 'warn']]);
});

test('notice shows its words; open.tab opens the skill\'s widgets; an unknown verb says so', async () => {
    const notices = [];
    const opened = [];
    const notice = (text, tone) => notices.push([text, tone]);
    assert.equal(await runContributionAction(item('t:hint', { action: { verb: 'notice', text: 'Saved lists live on Widgets.' } }),
        { notice }), true);
    assert.equal(await runContributionAction(item('quota:chip', { action: { verb: 'open.tab' } }),
        { notice, openTab: (skill) => opened.push(skill) }), true);
    assert.equal(await runContributionAction(item('t:new', { label: 'Later', action: { verb: 'teleport' } }), { notice }), false);
    assert.deepEqual(opened, ['quota']);
    assert.deepEqual(notices, [['Saved lists live on Widgets.', 'info'],
        ['Later asks for an action this version does not know.', 'warn']]);
});

test('skill.request posts the message or the draft to its own route and shows the skill\'s answer', async () => {
    const posted = [];
    const notices = [];
    const notice = (text, tone) => notices.push([text, tone]);
    const post = async (url, payload) => { posted.push([url, payload]); return posted.length === 1 ? { notice: 'Saved to Groceries' } : {}; };
    const save = item('lists:save', { label: 'Save to list', placement: 'message.action',
        action: { verb: 'skill.request', route: 'add', with: 'message', then: 'keep_draft', url: '/api/extensions/lists/add' } });
    const row = { address: 'row:1@t#a', role: 'assistant', text: 'Milk, eggs' };
    assert.equal(await runContributionAction(save, { row, post, notice }), true);
    const ask = item('lists:ask', { label: 'Ask lists', action: { verb: 'skill.request', with: 'none', url: '/api/extensions/lists/ask' } });
    assert.equal(await runContributionAction(ask, { post, notice }), true);
    assert.deepEqual(posted, [['/api/extensions/lists/add', { text: 'Milk, eggs', row_id: 'row:1@t#a' }],
        ['/api/extensions/lists/ask', {}]]);
    assert.deepEqual(notices, [['Saved to Groceries', 'ok'], ['Ask lists: done.', 'ok']]);
});

test('clear_draft clears only after a 2xx and only the words it sent; a failure keeps the draft and says why', async () => {
    const notices = [];
    const notice = (text, tone) => notices.push([text, tone]);
    const add = item('lists:add', { label: 'Add to list', placement: 'composer.plus',
        action: { verb: 'skill.request', with: 'draft', then: 'clear_draft', url: '/api/extensions/lists/add' } });
    const target = room('Buy milk');
    const sent = [];
    assert.equal(await runContributionAction(add, { draft: draftPort(() => target), notice,
        post: async (_url, payload) => { sent.push(payload.text); return { notice: 'Added' }; } }), true);
    assert.deepEqual(sent, ['Buy milk']);
    assert.equal(target.text, '', 'the sent draft is cleared after the 2xx');
    target.text = 'Buy bread';
    assert.equal(await runContributionAction(add, { draft: draftPort(() => target), notice,
        post: async () => { target.text = 'Buy bread and jam'; return {}; } }), true);
    assert.equal(target.text, 'Buy bread and jam', 'words typed while it ran are not lost');
    assert.equal(await runContributionAction(add, { draft: draftPort(() => target), notice,
        post: async () => { const error = new Error('skill is not running'); error.status = 409; throw error; } }), false);
    assert.equal(target.text, 'Buy bread and jam', 'a failure keeps the draft');
    assert.deepEqual(notices.at(-1), ['Add to list did not work: skill is not running', 'danger']);
});
