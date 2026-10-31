import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { fullText } from '../modules/text_fit.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
const fixture = JSON.parse(readFileSync(new URL('./fixtures/consciousness_contract.json', import.meta.url), 'utf8'));

async function replay(history, check) {
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? structuredClone(history) : {} }));
    const handlers = new Map();
    const ws = { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} };
    let instance, generation = 0;
    const stateSnapshots = { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} };
    try {
        instance = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 1, idPrefix: 'chat', mountEl: mount, asPanel: true });
        await instance.refreshHistory({ revision: 1 });
        await new Promise(resolve => setTimeout(resolve, 0));
        await check(document.byId.get('chat-messages'), handlers, instance);
    } finally { instance?.destroy(); restoreDom(prior); }
}
const card = (feed, id) => walkCard(feed, id);
const text = (node) => [node.textContent, node.innerHTML, ...(node.children || []).map(text)].join(' ');

test('terminal-only producer history keeps quiet truth without a progress row', async () => {
    for (const [index, history] of fixture.terminal_only_history.entries()) {
        assert.ok(history.messages.every(row => !row.is_progress));
        await replay(history, feed => {
            const root = card(feed, '214a31b6');
            assert.ok(root);
            assert.equal(text(root).includes('Done · no message'), index === 0);
        });
    }
});

test('checkpoint-only history applies the admission title, with no fabricated title when absent', async () => {
    for (const named of [true, false]) {
        const history = structuredClone(fixture.running_history);
        assert.ok(history.messages.length > 0 && history.messages.every(row => row.is_progress));
        if (!named) for (const row of history.messages) delete row.suggested_name;
        await replay(history, feed => {
            const root = card(feed, 'w-running');
            assert.ok(root);
            assert.equal(text(root).includes('Own time'), named);
        });
    }
});

test('the authored summary is the first Details fact on replay, absent on older wakes', async () => {
    for (const messages of [fixture.history.messages, [...fixture.history.messages].reverse()]) await replay({ ...fixture.history, messages }, feed => {
        const named = card(feed, 'w-summary'), old = card(feed, '214a31b6');
        const summary = fixture.consciousness.recent_wakes.find(row => row.task_id === 'w-summary').summary;
        const findFacts = (node) => Object.hasOwn(node?.dataset || {}, 'liveFacts') ? node : (node?.children || []).map(findFacts).find(Boolean);
        const facts = findFacts(named);
        assert.ok(facts, 'the actual card Details facts');
        assert.match(facts.innerHTML, /^<div class="chat-live-fact"><dt>In its words<\/dt>/);
        assert.ok(facts.innerHTML.includes(`<span data-i18n-authored>${summary}</span>`));
        assert.doesNotMatch(text(old), /In its words/);
    });
});

test('live terminal frames add and clear the authored summary through card facts', async () => {
    await replay(fixture.running_history, (feed, handlers) => {
        const send = (summary) => handlers.get('log')({ chat_id: 1, data: { type: 'task_done',
            task_id: 'w-running', status: 'completed', wake_summary: summary, quiet_effective: true,
            ts: '2026-10-24T08:00:00+00:00' } });
        send('Kept my own words.');
        assert.match(card(feed, 'w-running').querySelector('[data-live-facts]').innerHTML, /In its words/);
        assert.match(card(feed, 'w-running').querySelector('[data-live-facts]').innerHTML, /data-i18n-authored>Kept my own words/);
        send(null);
        assert.doesNotMatch(card(feed, 'w-running').querySelector('[data-live-facts]').innerHTML, /In its words/);
    });
});

// A self-started run says what it did: a schedule-born root keeps the schedule's name as its title and
// shows its sentence as the line under it; a wake's sentence is its title. Live and replay meet in one place.
const SENTENCE = 'Checked the disk: 30 GiB free, 30 less than last night; the cause is still unknown.';
const AT = '2026-10-28T06:39:00+00:00';
const checkpoint = (task_id, extra = {}) => {
    const row = { ...structuredClone(fixture.running_history.messages[0]), task_id, ts: '2026-10-28T06:30:00+00:00', ...extra };
    delete row.suggested_name; if (!extra.initiator) delete row.initiator;
    return row;
};
const terminal = (task_id, words = {}) => ({ role: 'system', system_type: 'task_summary', summary_kind: 'terminal_root_projection',
    task_id, chat_id: 1, ts: AT, text: 'Done · no message', status: 'completed', outcome_phase: 'done', outcome_final: true,
    outcome_authority: 'canonical_task_result_after_finalization', wake_silent: true, quiet_effective: true, ...words });
const evidence = (task_id, words = {}) => ({ role: 'system', system_type: 'task_summary', summary_kind: 'terminal_root_projection',
    task_id, chat_id: 1, ts: AT, text: '', is_progress: false, wake_silent: true, quiet_effective: true,
    historical_terminal: { status: 'completed', phase: 'done', ts: AT, wake_silent: true, quiet_effective: true }, ...words });
const named = { suggested_name: 'Daily check', schedule_name: 'Daily check' };
const find = (node, test) => (test(node) ? node : (node?.children || []).map((child) => find(child, test)).find(Boolean));
const view = (feed, id) => {
    const root = card(feed, id);
    return { title: find(root, (n) => n.classList?.contains('chat-live-title'))?.textContent || '',
        line: fullText(find(root, (n) => Object.hasOwn(n?.dataset || {}, 'liveActivity'))),
        facts: find(root, (n) => Object.hasOwn(n?.dataset || {}, 'liveFacts'))?.innerHTML || '' };
};

test('replay: a quiet schedule run is titled by its schedule and says what it did under that title', async () => {
    for (const row of [terminal('s1', { ...named, wake_summary: SENTENCE }), evidence('s1', { ...named, wake_summary: SENTENCE })]) {
        await replay({ messages: [checkpoint('s1'), row], window: { complete: true } }, (feed) => {
            const seen = view(feed, 's1');
            assert.equal(seen.title, 'Daily check', 'the authored name from the terminal row or its evidence');
            assert.equal(seen.line, SENTENCE, 'the sentence is the line under the title');
            assert.match(seen.facts, /^<div class="chat-live-fact"><dt>In its words<\/dt><dd><span data-i18n-authored>Checked the disk/);
            assert.match(seen.facts, /<dt>Started by<\/dt><dd>Schedule · <span data-i18n-authored>Daily check<\/span><\/dd>/);
        });
    }
    // Without the sentence nothing is invented; without the name the ended card stays Task activity.
    await replay({ messages: [checkpoint('s2'), terminal('s2', named)], window: { complete: true } }, (feed) => {
        const seen = view(feed, 's2');
        assert.equal(seen.title, 'Daily check');
        assert.notEqual(seen.line, SENTENCE);
        assert.doesNotMatch(seen.facts, /In its words/);
        assert.match(seen.facts, /<dt>Started by<\/dt><dd>Schedule · <span data-i18n-authored>Daily check<\/span><\/dd>/);
    });
    await replay({ messages: [checkpoint('s3'), evidence('s3', { schedule_name: 'Daily check', wake_summary: SENTENCE })], window: { complete: true } }, (feed) => {
        assert.equal(view(feed, 's3').title, 'Task activity');
        assert.equal(view(feed, 's3').line, SENTENCE);
    });
});

// A wake's narration as the live path sends it and history replays it: the words after the mark are its line.
const narration = (task_id, words, extra = {}) => ({ task_id, chat_id: 1, ts: '2026-10-28T06:30:00+00:00', role: 'assistant',
    is_progress: true, narration: true, text: `💬 ${words}`, content: `💬 ${words}`, ...extra });

test('replay and live: a wake\'s sentence is its title and its last narration stays its line; origin rows name their fact or stay absent', async () => {
    const wake = narration('w1', 'Reading the rooms', { initiator: 'consciousness', suggested_name: 'Own time' });
    await replay({ messages: [wake, terminal('w1', { suggested_name: SENTENCE, wake_summary: SENTENCE, initiator: 'consciousness' })], window: { complete: true } }, (feed) => {
        const seen = view(feed, 'w1');
        assert.equal(seen.title, SENTENCE);
        assert.equal(seen.line, 'Reading the rooms', 'the title already says the sentence; the narration keeps the line');
        assert.match(seen.facts, /<dt>Started by<\/dt><dd>Consciousness<\/dd>/);
    });
    await replay({ messages: [], window: { complete: true } }, (feed, handlers) => {
        handlers.get('chat')(wake);
        handlers.get('log')({ chat_id: 1, data: { type: 'task_done', task_id: 'w1', status: 'completed', wake_summary: SENTENCE,
            wake_silent: true, quiet_effective: true, initiator: 'consciousness', ts: AT } });
        handlers.get('task_named')({ task_id: 'w1', suggested_name: SENTENCE });  // _wake_finished names the card after task_done
        const seen = view(feed, 'w1');
        assert.equal(seen.title, SENTENCE);
        assert.equal(seen.line, 'Reading the rooms');
    });
    await replay({ messages: [checkpoint('o1'), terminal('o1', { suggested_name: 'Owner work', quiet_effective: false, wake_silent: false })], window: { complete: true } }, (feed) => {
        assert.doesNotMatch(view(feed, 'o1').facts, /Started by/);
    });
});

test('a host cause that holds the line is not hidden by the sentence', async () => {
    const stopped = terminal('c1', { ...named, wake_summary: SENTENCE, status: 'cancelled', outcome_phase: 'cancelled',
        reason_code: 'author_stop', quiet_effective: false, text: 'Stopped' });
    await replay({ messages: [checkpoint('c1'), stopped], window: { complete: true } }, (feed) => {
        const seen = view(feed, 'c1');
        assert.equal(seen.title, 'Daily check');
        assert.match(seen.line, /^Ouroboros stopped with unfinished work/);
        assert.match(seen.facts, /In its words/);
    });
});

test('live: the terminal frame of a quiet schedule run names the card and puts its sentence under the name', async () => {
    await replay({ messages: [], window: { complete: true } }, (feed, handlers) => {
        handlers.get('chat')({ chat_id: 1, ts: '2026-10-28T06:30:00+00:00', task_id: 'l1', role: 'assistant', is_progress: true,
            text: '💬 Reading the disk', content: '💬 Reading the disk', suggested_name: 'Daily check' });
        assert.equal(view(feed, 'l1').title, 'Daily check');
        handlers.get('chat')(terminal('l1', { ...named, wake_summary: SENTENCE }));
        const seen = view(feed, 'l1');
        assert.equal(seen.title, 'Daily check');
        assert.equal(seen.line, SENTENCE, 'the run\'s own words replace its last narration once it ends');
        assert.match(seen.facts, /<dt>Started by<\/dt><dd>Schedule · <span data-i18n-authored>Daily check<\/span><\/dd>/);
    });
});

test('replay: a page whose terminal row lies beyond it settles the card from its last progress row\'s terminal truth', async () => {
    // The history producer stamps the retained result's terminal truth on the window's last progress row.
    const stamped = { ...checkpoint('p1'), task_terminal_status: 'completed', outcome_phase: 'done', outcome_final: true,
        outcome_axes: { lifecycle: { status: 'completed' }, execution: { status: 'ok' } }, ...named, wake_summary: SENTENCE };
    delete stamped.task_phase;
    await replay({ messages: [stamped], window: { complete: true } }, (feed) => {
        const seen = view(feed, 'p1');
        assert.equal(seen.title, 'Daily check');
        assert.equal(seen.line, SENTENCE);
        assert.match(seen.facts, /^<div class="chat-live-fact"><dt>In its words<\/dt>/);
        assert.match(seen.facts, /<dt>Started by<\/dt><dd>Schedule · <span data-i18n-authored>Daily check<\/span><\/dd>/);
    });
});

test('live: the supervisor\'s task_done names the origin without waiting for a history read', async () => {
    await replay({ messages: [], window: { complete: true } }, (feed, handlers) => {
        handlers.get('chat')(narration('l2', 'Reading the disk', { suggested_name: 'Daily check' }));
        assert.doesNotMatch(view(feed, 'l2').facts, /Started by/);
        handlers.get('log')({ chat_id: 1, data: { type: 'task_done', task_id: 'l2', status: 'completed', wake_summary: SENTENCE,
            schedule_name: 'Daily check', wake_silent: true, quiet_effective: true, ts: AT } });
        const seen = view(feed, 'l2');
        assert.equal(seen.title, 'Daily check');
        assert.equal(seen.line, SENTENCE);
        assert.match(seen.facts, /<dt>Started by<\/dt><dd>Schedule · <span data-i18n-authored>Daily check<\/span><\/dd>/);
    });
});
