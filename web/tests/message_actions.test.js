// The message-action registry (message_actions.js): registration order and replacement, the row an action sees, the
// controls it draws between a bubble's Reply and Copy (redrawn when the registry or an action's own state moves), and
// skills' `message.action` contributions joining it on Ouroboros's saved replies only.
import assert from 'node:assert/strict';
import test from 'node:test';
import { bindBubbleActions, bridgeContributedActions, createMessageActions, messageRow, presentAction }
    from '../modules/message_actions.js';
import { installDomDouble } from './dom_double.js';

const noop = () => {};

test('actions come in order, then registration; a key replaces; an unregister removes only its own entry', () => {
    const registry = createMessageActions();
    const notices = [];
    registry.onChange((key) => notices.push(key));
    const offA = registry.register({ key: 'a', label: 'A', run: noop });
    registry.register({ key: 'first', label: 'First', order: -1, run: noop });
    registry.register({ key: 'never', label: 'Never', applies: () => { throw new Error('boom'); }, run: noop });
    registry.register({ key: 'users', label: 'Users', applies: (row) => row.role === 'user', run: noop });
    const row = { role: 'assistant' };
    assert.deepEqual(registry.actionsFor(row).map((action) => action.key), ['first', 'a'],
        'a throwing applies() offers nothing; applies() picks the rows');
    const offA2 = registry.register({ key: 'a', label: 'A again', run: noop });
    assert.equal(registry.actionsFor(row).find((action) => action.key === 'a').label, 'A again', 'the same key replaces');
    offA();
    assert.ok(registry.actionsFor(row).some((action) => action.key === 'a'), 'a stale unregister leaves the replacement');
    offA2();
    assert.deepEqual(registry.actionsFor(row).map((action) => action.key), ['first']);
    assert.throws(() => registry.register({ key: '', run: noop }), TypeError);
    assert.throws(() => registry.register({ key: 'x' }), TypeError);
    notices.length = 0;
    registry.batch(() => {
        registry.register({ key: 'b1', label: 'B1', run: noop });
        registry.register({ key: 'b2', label: 'B2', run: noop });
    });
    assert.deepEqual(notices, [''], 'a batch tells its listeners once');
});

test('the row an action sees: a saved bubble\'s address, role, task and plain words; an unsaved one is no row', () => {
    const env = installDomDouble();
    try {
        const { el } = env;
        const saved = el('div', { className: 'chat-bubble assistant', dataset: { rowAddress: 'row:1@t#a', taskId: 't-1' } },
            el('div', { className: 'sender', text: 'Ouroboros' }), el('div', { className: 'message', text: '  Milk and eggs ' }));
        const row = messageRow(saved);
        assert.deepEqual({ ...row, content: undefined }, { address: 'row:1@t#a', role: 'assistant', taskId: 't-1',
            text: 'Milk and eggs', content: undefined });
        assert.equal(row.content, saved.children[1]);
        assert.equal(messageRow(el('div', { className: 'chat-bubble user' })), null);
        assert.deepEqual(presentAction({ label: 'Read aloud', icon: 'volume', state: () => ({ label: 'Stop', icon: 'stop', pressed: true }) }, row),
            { label: 'Stop', icon: 'stop', title: 'Stop', pressed: true });
        assert.deepEqual(presentAction({ label: 'Save', title: 'Save this answer', state: () => { throw new Error('x'); } }, row),
            { label: 'Save', icon: '', title: 'Save this answer', pressed: false }, 'a throwing state() keeps the action\'s own face');
    } finally { env.restore(); }
});

function bubbleFixture() {
    const env = installDomDouble();
    const { doc, el } = env;
    const feed = el('div', { className: 'chat-messages' });
    doc.body.append(feed);
    const bubble = (role, address) => {
        const reply = el('button', { className: 'chat-message-reply' });
        const copy = el('button', { className: 'chat-message-copy' });
        const node = el('div', { className: `chat-bubble ${role} has-reply`, dataset: { rowAddress: address } },
            el('div', { className: 'message', text: `${role} words` }), el('div', { className: 'chat-message-actions' }, reply, copy));
        feed.append(node);
        return node;
    };
    const strip = (node) => node.children.find((child) => child.classList.contains('chat-message-actions'));
    // Each control's role (its first class); the shared finger target (ui-hit-area) is asserted on its own.
    const classes = (node) => strip(node).children.map((child) => child.className.split(' ')[0]);
    return { ...env, feed, bubble, strip, classes };
}

test('a bubble draws its actions between Reply and Copy, redraws them when the registry moves, and keeps its time clear', () => {
    const f = bubbleFixture();
    try {
        const registry = createMessageActions();
        const writes = [];
        const bound = bindBubbleActions({ feed: f.feed, registry, onDomWrite: (write) => { writes.push(1); return write(); } });
        const answer = f.bubble('assistant', 'row:1@a');
        const own = f.bubble('user', 'row:1@u');
        assert.equal(bound.sync(answer), false, 'nothing registered: the row stays Reply, Copy');
        assert.deepEqual(f.classes(answer), ['chat-message-reply', 'chat-message-copy']);
        const ran = [];
        let speaking = false;
        const off = registry.register({ key: 'read_aloud', label: 'Read aloud', icon: 'volume',
            applies: (row) => row.role === 'assistant',
            state: () => (speaking ? { label: 'Stop', icon: 'stop', pressed: true } : null),
            run: (row, ctx) => ran.push([row.address, ctx.anchor.dataset.messageAction]) });
        assert.equal(writes.length, 1, 'a registry change redraws inside the room\'s viewport seam');
        assert.deepEqual(f.classes(answer), ['chat-message-reply', 'chat-message-action', 'chat-message-copy']);
        assert.ok(f.strip(answer).children[1].className.split(' ').includes('ui-hit-area'),
            'under a finger the action takes the same hit area as Reply and Copy');
        assert.deepEqual(f.classes(own), ['chat-message-reply', 'chat-message-copy'], 'an action draws only where it applies');
        const control = f.strip(answer).children[1];
        assert.equal(control.getAttribute('aria-label'), 'Read aloud', 'an icon control has its name');
        assert.equal(control.title, 'Read aloud');
        assert.match(control.innerHTML, /^<svg /, 'a monochrome stroke glyph');
        assert.equal(answer.css.get('--chat-message-extra-actions'), '1', 'the time keeps clear of one more control');
        control.click();
        assert.deepEqual(ran, [['row:1@a', 'read_aloud']]);
        speaking = true;
        registry.changed('read_aloud');
        const pressed = f.strip(answer).children[1];
        assert.equal(pressed.getAttribute('aria-label'), 'Stop', 'the face follows the action\'s own state');
        assert.equal(pressed.getAttribute('aria-pressed'), 'true');
        off();
        assert.deepEqual(f.classes(answer), ['chat-message-reply', 'chat-message-copy']);
        assert.equal(answer.css.get('--chat-message-extra-actions'), '', 'the gutter returns to Reply and Copy');
        bound.destroy();
        registry.register({ key: 'late', label: 'Late', run: noop });
        assert.deepEqual(f.classes(answer), ['chat-message-reply', 'chat-message-copy'], 'a destroyed room draws nothing more');
    } finally { f.restore(); }
});

test('an action that throws or rejects says so in one line naming it', async () => {
    const f = bubbleFixture();
    try {
        const registry = createMessageActions();
        const notices = [];
        const bound = bindBubbleActions({ feed: f.feed, registry, notice: (text, tone) => notices.push([text, tone]) });
        registry.register({ key: 'boom', label: 'Boom', run: () => { throw new Error('no voice'); } });
        registry.register({ key: 'later', label: 'Later', run: async () => { throw new Error('offline'); } });
        const answer = f.bubble('assistant', 'row:1@a');
        assert.equal(bound.sync(answer), true);
        f.strip(answer).children[1].click();
        f.strip(answer).children[2].click();
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.deepEqual(notices, [['Boom did not work: no voice', 'danger'], ['Later did not work: offline', 'danger']]);
    } finally { f.restore(); }
});

test('skills\' message actions join after the host\'s own, on Ouroboros\'s saved replies, and follow the store', () => {
    const registry = createMessageActions();
    registry.register({ key: 'read_aloud', label: 'Read aloud', run: noop });
    let items = [
        { key: 'lists:save', skill: 'lists', label: 'Save to list', title: 'Keep this answer in a list', icon: 'list',
            action: { verb: 'skill.request', with: 'message', url: '/api/extensions/lists/save' } },
        { key: 'notes:note', skill: 'notes', label: 'Note', icon: '', action: { verb: 'notice', text: 'Noted' } },
    ];
    let listener = null;
    const store = { byPlacement: (name) => (name === 'message.action' ? items : []),
        onChange: (fn) => { listener = fn; return () => { listener = null; }; } };
    const runs = [];
    const draft = { text: () => '' };
    let changes = 0;
    registry.onChange(() => { changes += 1; });
    const dispose = bridgeContributedActions({ store, registry, draft,
        run: (item, ctx) => runs.push([item.key, ctx.row.address, ctx.draft === draft, typeof ctx.notice]) });
    assert.equal(changes, 1, 'the bridge registers in one batch');
    const answer = { role: 'assistant', address: 'row:1@a' };
    const actions = registry.actionsFor(answer);
    assert.deepEqual(actions.map((action) => [action.key, action.label, action.icon, action.title]), [
        ['read_aloud', 'Read aloud', undefined, undefined],
        ['contribution:lists:save', 'Save to list', 'list', 'Keep this answer in a list'],
        ['contribution:notes:note', 'Note', '', ''],
    ]);
    assert.deepEqual(registry.actionsFor({ role: 'user', address: 'row:1@u' }).map((action) => action.key), ['read_aloud'],
        'a skill\'s action is for Ouroboros\'s replies, not the owner\'s own rows');
    actions[1].run(answer, { notice: noop });
    assert.deepEqual(runs, [['lists:save', 'row:1@a', true, 'function']]);
    listener();
    assert.equal(changes, 1, 'an unchanged store changes nothing');
    items = [items[1]];
    listener();
    assert.deepEqual(registry.actionsFor(answer).map((action) => action.key), ['read_aloud', 'contribution:notes:note']);
    dispose();
    assert.deepEqual(registry.actionsFor(answer).map((action) => action.key), ['read_aloud']);
    assert.equal(listener, null, 'the store subscription is released');
});
