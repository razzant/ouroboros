import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../app.js', import.meta.url), 'utf8');
const door = source.slice(source.indexOf('async function revealQuotedSource('), source.indexOf('\nfunction freshProjectRow'));
function harness() {
    const calls = [], notices = [];
    const context = {
        projectNavigationGeneration: 0, state: { activePage: 'chat' }, canShow: true,
        page: { anchor: { state: 'found', task_id: '214a31b6', history_id: 'chat:42' } },
        window: { addEventListener: (name, fn) => { assert.equal(name, 'ouro:open-task'); context.listener = fn; } },
        showPage: async name => { calls.push(name); return context.canShow; },
        apiClient: { chatHistory: async ({ chatId, around }) => { calls.push(['read', chatId, around]); return context.page; } },
        drawn: new Set(),
        mainChat: { drawsSource: address => context.drawn.has(address), revealSource: async (address, current, page) => {
            assert.equal(page, context.page, 'reuse the pre-read page');
            calls.push(['reveal', address, current()]); return { state: 'found' };
        } },
        closeProjectPanel: () => calls.push('main'),
        showToast: text => notices.push(text), tr: (_, english) => english,
    };
    vm.createContext(context); vm.runInContext(door, context);
    return { context, calls, notices, press: (id = '214a31b6', chatId = 1) => context.listener({ detail: { task_id: id, chat_id: chatId } }) };
}

test('the Dashboard door lands through the real task anchor and keeps the pre-read page', async () => {
    const { context, calls, notices, press } = harness();
    await press();
    assert.deepEqual(calls, ['chat', ['read', 1, 'task:214a31b6'], 'main', ['reveal', 'task:214a31b6', true]]);
    assert.deepEqual(notices, []);
    calls.length = 0;
    await press('focus', 1099511627777);
    assert.deepEqual(calls, []);
    context.canShow = false;
    await press();
    assert.deepEqual(calls, ['chat']);
});

test('a card Main already draws is the destination without a read; Main lands on it itself', async () => {
    const { context, calls, notices, press } = harness();
    context.drawn.add('task:214a31b6');
    context.mainChat.revealSource = async (address, current, page) => {
        calls.push(['reveal', address, current(), page]); return { state: 'found' };
    };
    await press();
    assert.deepEqual(calls, ['chat', 'main', ['reveal', 'task:214a31b6', true, null]]);
    assert.deepEqual(notices, []);
});

test('missing, mismatched and failed task reads never land and explain unavailability', async () => {
    for (const anchor of [{ state: 'absent' }, { state: 'found', task_id: 'another' },
        { state: 'found', row_address: 'task:214a31b6', chat_id: 1 }, null]) {
        const { context, calls, notices, press } = harness();
        context.page = { anchor };
        await press();
        assert.equal(calls.some(call => call[0] === 'reveal'), false);
        assert.deepEqual(notices, ['This task is unavailable.']);
    }
    const { context, notices, press } = harness();
    context.apiClient.chatHistory = async () => { throw new Error('offline'); };
    await press();
    assert.deepEqual(notices, ['This task is unavailable.']);
});

test('row quotes still require both the exact row address and chat identity', async () => {
    for (const anchor of [
        { state: 'found', row_address: 'row:chat:42', chat_id: 1 },
        { state: 'found', row_address: 'row:chat:43', chat_id: 1 },
        { state: 'found', row_address: 'row:chat:42', chat_id: 2 },
        { state: 'found', task_id: '214a31b6', history_id: 'chat:42' },
    ]) {
        const { context, calls } = harness();
        context.page = { anchor };
        const result = await context.revealQuotedSource('row:chat:42', 1, () => true);
        const matches = anchor.row_address === 'row:chat:42' && anchor.chat_id === 1;
        assert.equal(result.state, matches ? 'found' : 'unknown');
        assert.equal(calls.some(call => call[0] === 'reveal'), matches);
    }
});

test('navigation that changes during the pre-read cancels landing', async () => {
    const { context, calls, notices, press } = harness();
    context.apiClient.chatHistory = async () => { context.projectNavigationGeneration++; return context.page; };
    await press();
    assert.deepEqual(calls, ['chat']);
    assert.deepEqual(notices, []);
});
