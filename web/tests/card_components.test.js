// Component cards (card_components.js): a living card's tree drawn in a CARD context, presses to the card's
// decisions. A press is the owner's act: it posts `card:<card_id>:<submit.id>` once per press, a failed or owed press
// is retried with the SAME request id, a `once` control locks after it counted, and local components never post.
import assert from 'node:assert/strict';
import test from 'node:test';

import { createCardPresses, mountCardComponents, mountPinnedCard } from '../modules/card_components.js';
import { installCardDom } from './helpers/card_dom.js';

function recorder(...answers) {
    const sent = [];
    const post = async (body) => { sent.push(body); return answers.shift() || { status: 200, body: { ok: true, delivery: 'delivered' } }; };
    let n = 0;
    return { sent, post, newId: () => `req-${++n}` };
}

const tick = () => new Promise((resolve) => setImmediate(resolve));

test('a press posts the card decision; each delivered press is a new submission with a new request id', async () => {
    const r = recorder();
    const presses = createCardPresses({ cardId: 'card_0123456789abcdef', post: r.post, newId: r.newId });
    assert.deepEqual(await presses.press('approve'), { status: 'success', message: 'Sent ✓' });
    assert.deepEqual(await presses.press('approve'), { status: 'success', message: 'Sent ✓' });
    assert.deepEqual(r.sent, [
        { request_id: 'req-1', decision_id: 'card:card_0123456789abcdef:approve' },
        { request_id: 'req-2', decision_id: 'card:card_0123456789abcdef:approve' },
    ]);
});

test('a press that got no answer or failed on the server is retried with the same request id; new values start anew', async () => {
    const r = recorder({ status: 0, body: { error: 'Failed to fetch' } }, { status: 503, body: { error: 'disk full' } });
    const presses = createCardPresses({ cardId: 'card_a', post: r.post, newId: r.newId });
    const first = await presses.press('save', { values: { name: 'Ann' } });
    assert.equal(first.status, 'error');
    assert.match(first.message, /^Not sent: Failed to fetch\. Press again to retry\.$/);
    assert.match((await presses.press('save', { values: { name: 'Ann' } })).message, /disk full/);
    assert.equal((await presses.press('save', { values: { name: 'Ann' } })).message, 'Sent ✓');
    assert.deepEqual(r.sent.map((body) => body.request_id), ['req-1', 'req-1', 'req-1'], 'one submission, retried');
    assert.deepEqual(r.sent[0].values, { name: 'Ann' });

    const changed = recorder({ status: 0, body: {} });
    const other = createCardPresses({ cardId: 'card_a', post: changed.post, newId: changed.newId });
    await other.press('save', { values: { name: 'Ann' } });
    await other.press('save', { values: { name: 'Bob' } });
    assert.deepEqual(changed.sent.map((body) => body.request_id), ['req-1', 'req-2'], 'changed values are a new press');
});

test('an owed delivery (202) keeps the request id for the next press; a refusal (400) does not', async () => {
    const r = recorder({ status: 202, body: { ok: true, delivery: 'owed' } }, { status: 200, body: { ok: true } },
        { status: 400, body: { error: 'values name fields the form does not have: x', reason_code: 'values_invalid' } });
    const presses = createCardPresses({ cardId: 'card_b', post: r.post, newId: r.newId });
    assert.deepEqual(await presses.press('go'), { status: 'warning', message: 'Saved, not delivered yet. Press again to retry.' });
    assert.equal((await presses.press('go')).message, 'Sent ✓');
    assert.equal((await presses.press('go')).message, 'Not sent: values name fields the form does not have: x');
    await presses.press('go');
    assert.deepEqual(r.sent.map((body) => body.request_id), ['req-1', 'req-1', 'req-2', 'req-3']);
});

test('a once control locks after its press counted, here or in the card\'s recorded presses; never a second post', async () => {
    const r = recorder();
    const presses = createCardPresses({ cardId: 'card_c', post: r.post, newId: r.newId });
    await presses.press('pick', { once: true });
    assert.equal(presses.locked('pick'), true);
    assert.equal((await presses.press('pick', { once: true })).message, 'Sent ✓');
    assert.equal(r.sent.length, 1);

    const late = recorder({ status: 409, body: { ok: false, reason_code: 'card_action_once', winner: { request_id: 'x', delivery: 'delivered' } } });
    const elsewhere = createCardPresses({ cardId: 'card_c', post: late.post, newId: late.newId });
    assert.equal((await elsewhere.press('pick', { once: true })).message, 'Already sent ✓');
    assert.equal(elsewhere.locked('pick'), true);

    const replayed = createCardPresses({ cardId: 'card_c', post: r.post, newId: r.newId });
    replayed.adopt([{ action_id: 'pick', label: 'Pick', at: 't', delivery: 'forwarded' },
        { action_id: 'more', label: 'More', at: 't', delivery: 'delivered' }], new Set(['pick']));
    assert.equal(replayed.locked('pick'), true);
    assert.equal(replayed.locked('more'), false, 'only a once control locks from history');
    // A control the replaced card no longer has locks with the host's word.
    const gone = recorder({ status: 404, body: { reason_code: 'action_not_found' } });
    const replaced = createCardPresses({ cardId: 'card_c', post: gone.post, newId: gone.newId });
    assert.equal((await replaced.press('old')).message, 'This control is no longer on the card.');
    assert.equal(replaced.locked('old'), true);
});

const CARD = {
    card_id: 'card_d',
    revision: 1,
    components: [
        { type: 'markdown', text: 'Pick a plan' },
        { type: 'tabs', tabs: [{ label: 'Monthly', components: [{ type: 'metric', label: 'Price', value: 9 }] },
            { label: 'Yearly', components: [{ type: 'metric', label: 'Price', value: 90 }] }] },
        { type: 'action', id: 'buy', label: 'Buy', submit: { id: 'buy', once: true } },
        { type: 'form', id: 'note', title: 'Note', submit_label: 'Send note', fields: [{ name: 'text', label: 'Text' }], submit: { id: 'note' } },
        { type: 'table', path: 'rows', columns: [{ label: 'Name', path: 'name' }] },
    ],
    data: { rows: [{ name: 'Basic' }] },
};

test('the card draws in place: data fills path fields, local tabs never post, presses post with their values', async () => {
    const dom = installCardDom();
    try {
        const r = recorder();
        const mount = document.createElement('div');
        const presses = createCardPresses({ cardId: CARD.card_id, post: r.post, newId: r.newId });
        const mounted = mountCardComponents(mount, CARD, { presses });
        assert.match(mount.innerHTML, /Pick a plan/);
        assert.match(mount.innerHTML, /<td data-label="Name">Basic<\/td>/, 'data is what path fields read');
        assert.match(mount.textContent, /Price9/);
        const markdown = mount.querySelector('.widget-markdown');
        mount.querySelectorAll('[data-widget-tab-key]')[1].click();
        assert.match(mount.textContent, /Price90/, 'the tab switched');
        assert.equal(mount.querySelector('.widget-markdown'), markdown, 'kept in place: the keyed patch, not a repaint');
        assert.equal(r.sent.length, 0, 'a tab is local');

        mount.querySelector('[data-widget-action]').click();
        assert.match(mount.querySelector('[data-widget-action]').textContent, /Sending…/);
        assert.equal(mount.querySelector('[data-widget-action]').disabled, true);
        await tick();
        assert.deepEqual(r.sent, [{ request_id: 'req-1', decision_id: 'card:card_d:buy', revision: 1 }]);
        assert.equal(mount.querySelector('[data-widget-action]').disabled, true, 'a once control locks');
        assert.match(mount.querySelector('[data-widget-feedback]').textContent, /Sent ✓/);

        const form = mount.querySelector('[data-widget-form]');
        form.elements = { namedItem: (name) => (name === 'text' ? { value: 'Call me' } : null) };
        form.dispatch('submit');
        await tick();
        assert.deepEqual(r.sent[1], { request_id: 'req-2', decision_id: 'card:card_d:note', revision: 1, values: { text: 'Call me' } });
        mounted.dispose();
        mount.querySelector('[data-widget-action]').click();
        assert.equal(r.sent.length, 2, 'a disposed card listens to nothing');
    } finally {
        dom.restore();
    }
});

test('a card read again shows its recorded presses; a kept card on Widgets uses the same host', async () => {
    const dom = installCardDom();
    try {
        const mount = document.createElement('div');
        const mounted = mountCardComponents(mount, { ...CARD, submissions_summary: [] }, { presses: createCardPresses({ cardId: 'card_d', post: async () => ({ status: 200, body: {} }) }) });
        assert.equal(mount.querySelector('[data-widget-action]').disabled, false);
        mounted.update({ ...CARD, submissions_summary: [{ action_id: 'buy', label: 'Buy', at: 't', delivery: 'delivered' },
            { action_id: 'note', label: 'Send note', at: 't', delivery: 'owed' }] });
        assert.equal(mount.querySelector('[data-widget-action]').disabled, true);
        const feedback = mount.querySelectorAll('[data-widget-feedback]').map((node) => node.textContent);
        assert.deepEqual(feedback, ['Sent ✓', 'Saved, not delivered yet. Press again to retry.']);

        const sent = [];
        const prior = globalThis.fetch;
        globalThis.fetch = async (url, init) => { sent.push([url, JSON.parse(init.body)]); return { status: 200, json: async () => ({ ok: true }) }; };
        try {
            const board = document.createElement('div');
            const dispose = mountPinnedCard(board, { key: '_cards:card_d', skill: '_cards', tab_id: 'card_d', pinned_card: true,
                render: { kind: 'declarative', components: CARD.components, data: CARD.data } });
            assert.match(board.innerHTML, /Basic/);
            board.querySelector('[data-widget-action]').click();
            await tick();
            assert.equal(sent[0][0], '/api/decisions');
            assert.equal(sent[0][1].decision_id, 'card:card_d:buy');
            dispose();
        } finally {
            globalThis.fetch = prior;
        }
    } finally {
        dom.restore();
    }
});

test('a press that waits for its paused task says so until that task reads it, live and after a reload', async () => {
    const dom = installCardDom();
    try {
        const waiting = { ok: true, delivery: 'delivered', reason_code: 'awaiting_recipient_read' };
        const r = recorder({ status: 200, body: waiting }, { status: 200, body: { ok: true, delivery: 'delivered' } });
        const mount = document.createElement('div');
        const mounted = mountCardComponents(mount, CARD, { presses: createCardPresses({ cardId: CARD.card_id, post: r.post, newId: r.newId }) });
        mount.querySelector('[data-widget-action]').click();
        await tick();
        assert.equal(mount.querySelector('[data-widget-feedback]').textContent, 'Saved · the task reads it when it continues');
        assert.equal(mount.querySelector('[data-widget-action]').disabled, true, 'a once control still locks');
        mounted.update({ ...CARD, submissions_summary: [{ action_id: 'buy', label: 'Buy', at: 't', delivery: 'delivered', request_id: 'req-1' }] });
        assert.equal(mount.querySelector('[data-widget-feedback]').textContent, 'Sent ✓', 'its read ends the wait, live');

        const reread = document.createElement('div');
        const again = mountCardComponents(reread, { ...CARD, submissions_summary: [{ action_id: 'note', label: 'Send note', at: 't',
            delivery: 'delivered', request_id: 'req-9', reason_code: 'awaiting_recipient_read' }] },
        { presses: createCardPresses({ cardId: CARD.card_id, post: r.post, newId: r.newId }) });
        assert.deepEqual(reread.querySelectorAll('[data-widget-feedback]').map((node) => node.textContent), ['Saved · the task reads it when it continues']);
        const form = reread.querySelector('[data-widget-form]');
        form.elements = { namedItem: (name) => (name === 'text' ? { value: 'Later' } : null) };
        form.dispatch('submit');
        await tick();
        assert.deepEqual(reread.querySelectorAll('[data-widget-feedback]').map((node) => node.textContent), ['Sent ✓'],
            'a press to a task that reads its mail does not wait');
        again.dispose();

        // A kept copy on Widgets hears no card_state: it says Sent ✓ rather than a wait it could never see end.
        const prior = globalThis.fetch;
        globalThis.fetch = async () => ({ status: 200, json: async () => waiting });
        try {
            const board = document.createElement('div');
            const dispose = mountPinnedCard(board, { key: '_cards:card_d', skill: '_cards', tab_id: 'card_d', pinned_card: true,
                render: { kind: 'declarative', components: CARD.components, data: CARD.data } });
            board.querySelector('[data-widget-action]').click();
            await tick();
            assert.equal(board.querySelector('[data-widget-feedback]').textContent, 'Sent ✓');
            dispose();
        } finally {
            globalThis.fetch = prior;
        }
    } finally {
        dom.restore();
    }
});

test('an owed once press is retried under its recorded id, after a reload too; once delivered it locks', async () => {
    const r = recorder({ status: 202, body: { ok: true, delivery: 'owed' } }, { status: 200, body: { ok: true, delivery: 'forwarded' } });
    const presses = createCardPresses({ cardId: 'card_e', post: r.post, newId: r.newId });
    assert.equal((await presses.press('pick', { once: true })).message, 'Saved, not delivered yet. Press again to retry.');
    // The live card_state that follows a 202 records the press as owed: that must not lock its retry.
    presses.adopt([{ action_id: 'pick', label: 'Pick', at: 't', delivery: 'owed', request_id: 'req-1' }], new Set(['pick']));
    assert.equal(presses.locked('pick'), false, 'owed: the press can still be delivered');
    assert.equal((await presses.press('pick', { once: true })).message, 'Sent ✓');
    assert.deepEqual(r.sent.map((body) => body.request_id), ['req-1', 'req-1'], 'the retry is the same press');
    assert.equal(presses.locked('pick'), true);

    // A reload starts a new ledger: the recorded owed press is retried under its own id, whatever the form holds now.
    const again = recorder();
    const reloaded = createCardPresses({ cardId: 'card_e', post: again.post, newId: again.newId });
    reloaded.adopt([{ action_id: 'pick', label: 'Pick', at: 't', delivery: 'owed', request_id: 'saved-7' }], new Set(['pick']));
    assert.equal(reloaded.locked('pick'), false);
    await reloaded.press('pick', { once: true, values: { note: 'changed' } });
    assert.equal(again.sent[0].request_id, 'saved-7');
    const counted = createCardPresses({ cardId: 'card_e', post: again.post, newId: again.newId });
    counted.adopt([{ action_id: 'pick', label: 'Pick', at: 't', delivery: 'forwarded', request_id: 'saved-7' }], new Set(['pick']));
    assert.equal(counted.locked('pick'), true, 'a delivered once press locks from history');
    await counted.press('pick', { once: true });
    assert.equal(again.sent.length, 1, 'a counted once control never posts again');
});

test('a press names the revision it was drawn from; a changed card is read again, a kept copy says it is out of date', async () => {
    const dom = installCardDom();
    const settle = async () => { for (let i = 0; i < 6; i += 1) await tick(); };
    try {
        const r = recorder({ status: 409, body: { ok: false, reason_code: 'card_stale', revision: 3 } });
        const mount = document.createElement('div');
        const stale = [];
        mountCardComponents(mount, { ...CARD, revision: 2 }, { presses: createCardPresses({ cardId: 'card_d', post: r.post, newId: r.newId }),
            onStale: (revision) => stale.push(revision) });
        mount.querySelector('[data-widget-action]').click();
        await settle();
        assert.equal(r.sent[0].revision, 2);
        assert.deepEqual(stale, [3], 'the feed reads the newer card');
        assert.match(mount.querySelector('[data-widget-feedback]').textContent, /changed since it was drawn/);
        assert.equal(mount.querySelector('[data-widget-action]').disabled, false, 'a stale press locks nothing');

        const sent = [];
        const prior = globalThis.fetch;
        globalThis.fetch = async (url, init) => {
            sent.push(JSON.parse(init.body));
            return { status: 409, json: async () => ({ ok: false, reason_code: 'card_stale', revision: 5 }) };
        };
        try {
            const board = document.createElement('div');
            mountPinnedCard(board, { key: '_cards:card_d', skill: '_cards', tab_id: 'card_d', pinned_card: true, revision: '4',
                render: { kind: 'declarative', components: CARD.components, data: CARD.data } });
            board.querySelector('[data-widget-action]').click();
            await settle();
            assert.equal(sent[0].revision, 4, 'a kept card presses the revision it was kept at');
            assert.match(board.querySelector('[data-widget-feedback]').textContent, /kept copy is out of date/);
        } finally {
            globalThis.fetch = prior;
        }
    } finally {
        dom.restore();
    }
});

test('a once press that lost to another press still owed retries that one; a delivered winner locks', async () => {
    const owedWinner = { status: 409, body: { ok: false, reason_code: 'card_action_once', winner: { request_id: 'first', delivery: 'owed' } } };
    const r = recorder(owedWinner, { status: 200, body: { ok: true, delivery: 'forwarded', duplicate: true } });
    const presses = createCardPresses({ cardId: 'card_f', post: r.post, newId: r.newId });
    const feedback = await presses.press('pick', { once: true, revision: 2 });
    assert.equal(feedback.message, 'Saved, not delivered yet. Press again to retry.', 'never "Already sent" while the winner is owed');
    assert.equal(presses.locked('pick'), false);
    assert.equal((await presses.press('pick', { once: true, revision: 2 })).message, 'Sent ✓');
    assert.deepEqual(r.sent.map((body) => body.request_id), ['req-1', 'first'], 'the second press retries the winner');
    assert.equal(presses.locked('pick'), true);
});

test('a once control has one press: after an unanswered press, an edited form retries that press', async () => {
    const r = recorder({ status: 0, body: { error: 'Failed to fetch' } }, { status: 200, body: { ok: true } });
    const presses = createCardPresses({ cardId: 'card_g', post: r.post, newId: r.newId });
    await presses.press('dates', { once: true, values: { from: 'Mon' } });
    await presses.press('dates', { once: true, values: { from: 'Tue' } });
    assert.deepEqual(r.sent.map((body) => body.request_id), ['req-1', 'req-1'], 'never a second press of a once control');

    const other = recorder({ status: 0, body: {} });
    const ordinary = createCardPresses({ cardId: 'card_g', post: other.post, newId: other.newId });
    await ordinary.press('note', { values: { text: 'a' } });
    await ordinary.press('note', { values: { text: 'b' } });
    assert.deepEqual(other.sent.map((body) => body.request_id), ['req-1', 'req-2'], 'an ordinary form with new values is a new press');
});
