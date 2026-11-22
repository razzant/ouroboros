import assert from 'node:assert/strict';
import test from 'node:test';

import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

// A review reports the stored ending it read. Older records may lack an attempt number.
// Drive the real log handler: words, chip, counters and episode must all follow one ending.

const TASK = 'root-card';
const T1 = '2026-11-18T10:00:00.000000+00:00';  // attempt 1's ending, delayed in transit
const T2 = '2026-11-18T11:00:00.000000+00:00';  // attempt 2's ending, published on time
const R3 = '2026-11-18T12:00:00.000000+00:00';  // the late PASS: the row revision it read

const AXES = {
    lifecycle: { status: 'completed' },
    execution: { status: 'ok', reason_code: 'final_message' },
};

const doneFrame = (attempt, ts, words, metrics, extra = {}) => ({
    type: 'task_done', task_id: TASK, task_attempt: attempt, status: 'completed', outcome_final: true,
    reason_code: 'final_message', outcome_axes: AXES, ts, wake_summary: words, ...metrics, ...extra,
});

// The quiet flag makes the repaint observable: the late PASS reads the stored row as a quiet
// finish, so an accepted outcome repaints the chip to 'Done · no message' (taskPresentation).
const latePass = (revision, attempt) => ({
    type: 'review_late_result', task_id: TASK, surface: 'result', record_id: 'rec-late',
    subject_sha: 'sha-late', status: 'settled', aggregate: 'PASS', ts: '2026-11-18T12:05:00.000000+00:00',
    current_outcome: {
        ...(attempt === undefined ? {} : { task_attempt: attempt }),
        task_id: TASK, status: 'completed', task_terminal_status: 'completed', reason_code: 'final_message',
        quiet_effective: true, _result_text_present: true, revision,
        outcome_axes: { ...AXES, objective: { status: 'pass', source: 'result_review', review_status: 'pass' },
            review: { status: 'pass', source: 'result_review' }, artifacts: { status: 'not_applicable' } },
    },
});

const SECOND = 'Second attempt words.';
const FIRST = 'First attempt words.';

async function withChat(run) {
    let history = { messages: [] }, revision = 0;
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? history : { active_direct_turns: [] } }));
    const handlers = new Map();
    let instance;
    try {
        instance = createChatInstance({
            ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
                isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
            stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }),
                gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
            chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true,
        });
        const messages = globalThis.document.byId.get('chat-messages');
        const settle = () => new Promise(resolve => setImmediate(resolve));
        const log = async (data) => { handlers.get('log')({ chat_id: 2, data }); await settle(); };
        // Only a host-attested progress frame on the chat channel grants Stop authority
        // (updateLiveCardFromProgressMessage, `grantCancelAuthority`).
        const chat = async (msg) => { handlers.get('chat')({ ...msg, chat_id: 2 }); await settle(); };
        const page = (rows, next = null) => ({ messages: rows, page_cursor: 'newest', next_cursor: next,
            has_more: Boolean(next), window: { complete: !next, truncated_by: next ? ['quota'] : [] } });
        const read = async (rows, next = null) => {
            history = page(rows, next);
            const receipt = await instance.refreshHistory({ revision: ++revision });
            assert.equal(receipt.painted, true);
        };
        const older = async (rows) => {
            history = { ...page(rows), page_cursor: 'older' };
            const button = messages.querySelector('.chat-load-older-btn');
            assert.ok(button && !button.hidden);
            for (const click of button.listeners.get('click')) await click({ target: button });
        };
        await run(log, chat, (id = TASK) => walkCard(messages, id), { read, older });
    } finally { instance?.destroy(); restoreDom(prior); }
}

// Every fact of the settled card the delayed frame could displace, read from the DOM nodes.
const facts = (card) => ({
    finished: card.dataset.finished,
    phase: card.querySelector('[data-live-phase]').textContent,
    title: card.querySelector('[data-live-title]').textContent,
    activity: card.querySelector('[data-live-activity]').textContent,
    meta: card.querySelector('[data-live-meta]').innerHTML,
    details: card.querySelector('[data-live-facts]').innerHTML,
    episodes: card.querySelector('[data-live-episodes]').innerHTML,
    anchor: card.dataset.ts,
});

const markup = (card) => {
    const nodes = (node) => [node, ...(node.children || []).flatMap(nodes)];
    return nodes(card).map(node => `${node.innerHTML || ''}|${node.textContent || ''}`).join('\n');
};

test('a late PASS owns the finished card: the delayed older task_done changes nothing', async () => {
    await withChat(async (log, _chat, card) => {
        await log(doneFrame(2, T2, SECOND, { rounds: 4, tool_calls: 5, duration_sec: 37 }));
        const settled = card();
        assert.ok(settled, 'the current attempt settles its card');
        assert.equal(settled.dataset.finished, '1');
        assert.equal(settled.querySelector('[data-live-phase]').textContent, 'Done');
        assert.ok(markup(settled).includes(SECOND), 'its closing sentence is on the card');

        await log(latePass(R3));
        const reviewed = card();
        const afterPass = facts(reviewed);
        assert.equal(afterPass.finished, '1');
        assert.equal(afterPass.phase, 'Done · no message', 'the late PASS repaints the current outcome');
        assert.ok(markup(reviewed).includes(SECOND), 'the repainted card keeps the current sentence');

        // Attempt 1's ending arrives now: its own words, inflated counters, an older anchor.
        await log(doneFrame(1, T1, FIRST, { rounds: 9, tool_calls: 12, duration_sec: 98 }));
        const after = facts(card());
        assert.deepEqual(after, afterPass,
            'the older ending leaves outcome, sentence, title, chip, metrics, episodes and anchor as the review left them');
        assert.ok(!markup(card()).includes(FIRST), 'no older sentence anywhere on the card');
        assert.match(after.meta, /37s.*5 tool calls/);
        assert.match(after.details, /<dt>Rounds<\/dt><dd>4 · usable solve responses<\/dd>/);
    });
});

test('the same pair in the other order: the older ending is refused first, the late PASS still lands', async () => {
    await withChat(async (log, _chat, card) => {
        await log(doneFrame(2, T2, SECOND, { rounds: 4, tool_calls: 5 }));
        // The delayed older ending arrives before the review resolves: attempt 1 < attempt 2.
        await log(doneFrame(1, T1, FIRST, { rounds: 9, tool_calls: 12 }));
        const beforePass = card();
        assert.ok(markup(beforePass).includes(SECOND), 'the card still shows the current attempt');
        assert.ok(!markup(beforePass).includes(FIRST), 'the older sentence never lands');
        const before = facts(beforePass);
        assert.match(before.details, /<dt>Rounds<\/dt><dd>4 · usable solve responses<\/dd>/);
        assert.match(before.meta, /5 tool calls/);
        assert.equal(before.phase, 'Done');

        // The refusal must not wedge the card: the genuine late PASS is still accepted.
        await log(latePass(R3));
        const afterPass = facts(card());
        assert.equal(afterPass.finished, '1');
        assert.equal(afterPass.phase, 'Done · no message', 'the late PASS repaints after the refusal');
        assert.ok(markup(card()).includes(SECOND), 'the current sentence survives the whole sequence');
    });
});

test('a review for an older attempt never repaints the current card, and its own review cannot pin the next attempt', async () => {
    await withChat(async (log, _chat, card) => {
        await log(doneFrame(2, T2, SECOND, { rounds: 4, tool_calls: 5 }));
        const before = facts(card());
        await log(latePass(R3, 1));  // later timestamp, but not this attempt
        assert.deepEqual(facts(card()), before);
        await log(latePass(R3, 2));
        assert.equal(facts(card()).phase, 'Done · no message');
        await log(doneFrame(3, '2026-11-18T13:00:00Z', 'Third attempt finished.', { rounds: 2, tool_calls: 3 }));
        assert.equal(facts(card()).phase, 'Done');
        assert.ok(markup(card()).includes('Third attempt finished.'));
        assert.match(facts(card()).details, /<dt>Rounds<\/dt><dd>2 · usable solve responses<\/dd>/);
        assert.match(facts(card()).meta, /3 tool calls/);
    });
});

test('control: the legitimate current attempt still shows its Stop, then folds its own ending', async () => {
    await withChat(async (log, chat, card) => {
        await chat({ task_id: TASK, role: 'assistant', is_progress: true, narration: true, cancelable: true,
            content: 'Checking the invariant', text: 'Checking the invariant', ts: T1 });
        const working = card();
        assert.ok(working, 'the running work has its card');
        assert.equal(working.dataset.finished, '0');
        assert.ok(working.querySelector('[data-cancel-run]'), 'the host-attested Stop stays on the card');

        // The host's two live carriers: numbered task_metrics_event and task_done with a bound execution_owner.
        await log({ type: 'task_metrics_event', task_id: TASK, task_attempt: 1, ts: T2,
            rounds: 3, tool_calls: 7, duration_sec: 37 });
        await log(doneFrame(undefined, T2, 'The work finished cleanly.', {}, {
            execution_owner: { kind: 'pooled', task_id: TASK, task_attempt: 1 },
        }));
        const done = facts(card());
        assert.equal(done.finished, '1');
        assert.equal(done.phase, 'Done');
        assert.match(done.details, /<dt>Rounds<\/dt><dd>3 · usable solve responses<\/dd>/);
        assert.match(done.meta, /37s.*7 tool calls/);
        assert.ok(markup(card()).includes('The work finished cleanly.'), 'its closing sentence shows');
    });
});

test('physical attempt zero keeps its live counters in either delivery order', async () => {
    const metrics = { type: 'task_metrics_event', task_id: TASK, task_attempt: 0, ts: T2,
        rounds: 3, tool_calls: 7, duration_sec: 37 };
    const done = doneFrame(undefined, T2, FIRST, {}, {
        _is_direct_chat: true, execution_owner: { kind: 'direct', task_id: TASK, task_attempt: 0 },
    });
    for (const order of [[metrics, done], [done, metrics]]) {
        await withChat(async (log, chat, card) => {
            // Direct chat's real start carrier can omit the ordinal; it is not an attempt-1 witness.
            await log({ type: 'task_started', task_id: TASK, chat_id: 2, task_type: 'task',
                task_text: 'Check the inventory.', direct_chat: true, ts: T1, activity_emitted_at: T1 });
            await chat({ task_id: TASK, role: 'assistant', is_progress: true, narration: true, cancelable: true,
                content: 'Checking', text: 'Checking', ts: T1 });
            for (const frame of order) await log(frame);
            const settled = facts(card());
            assert.equal(settled.finished, '1');
            assert.equal(settled.phase, 'Done');
            assert.match(settled.details, /<dt>Rounds<\/dt><dd>3 · usable solve responses<\/dd>/);
            assert.match(settled.meta, /37s.*7 tool calls/);
        });
    }
});

const saved = (event, id, extra = {}) => ({ ...event, role: 'system', system_type: 'task_summary',
    text: '', history_id: id, ...extra });

test('Load older keeps the selected result, duration and counts, including with a legacy carrier', async () => {
    await withChat(async (_log, _chat, card, { read, older }) => {
        const second = saved(doneFrame(2, T2, SECOND, { rounds: 4, tool_calls: 5, duration_sec: 37 }), 'current');
        await read([second], 'older');
        const mounted = card(), before = facts(mounted);
        assert.match(before.meta, /37s.*5 tool calls/);
        assert.match(before.details, /<dt>Rounds<\/dt><dd>4 · usable solve responses<\/dd>/);
        assert.ok(markup(mounted).includes(SECOND));
        await older([
            saved(doneFrame(1, T1, FIRST, { rounds: 9, tool_calls: 12, duration_sec: 98 }), 'old'),
            saved(doneFrame(undefined, T1, 'Legacy words.', { rounds: 13, tool_calls: 19, duration_sec: 108 }), 'legacy'),
        ]);
        assert.equal(card(), mounted);
        assert.deepEqual(facts(card()), before);
        assert.doesNotMatch(markup(card()), /First attempt words|Legacy words/);
    });
});

test('a paged old row carries the current outcome\'s own rounds into Details once', async () => {
    await withChat(async (_log, _chat, card, { read }) => {
        const current = { ...latePass(R3, 2).current_outcome, rounds: 17, wake_summary: SECOND };
        const old = { ...saved(doneFrame(1, T1, FIRST, { rounds: 4, tool_calls: 3 }), 'old'),
            current_outcome: current };
        const narration = { task_id: TASK, role: 'assistant', is_progress: true, narration: true,
            text: 'Inspecting the current attempt', history_id: 'current-progress', task_attempt: 2, ts: T2 };
        await read([narration, old]);
        await read([narration, old]);
        const shown = facts(card());
        assert.match(shown.details, /<dt>Rounds<\/dt><dd>17 · usable solve responses<\/dd>/);
        assert.doesNotMatch(shown.details, /4 · usable solve responses/);
        assert.ok(markup(card()).includes(SECOND));
    });
});

test('Continue history shows the selected attempt’s interval and never its replaced cause', async () => {
    const next = 'successor';
    const firstClock = '2026-11-18T09:34:00Z', secondClock = '2026-11-18T10:51:00Z';
    const clock = ts => [new Date(ts).getHours(), new Date(ts).getMinutes()].map(n => String(n).padStart(2, '0')).join(':');
    const end = (attempt, started_at, occurred_at) => ({ source: 'executor_terminal', occurred_at, attempt: { task_attempt: attempt, started_at } });
    const first = doneFrame(0, T1, FIRST, { tool_calls: 12 }, { status: 'failed', reason_code: 'earlier_fault_code',
        outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'failed' } }, terminal_time: end(0, firstClock, T1) });
    const current = doneFrame(1, T2, SECOND, { tool_calls: 5 }, { suggested_name: 'Read the catalog',
        terminal_time: end(1, secondClock, T2), continued_by: { state: 'admitted', successor_task_id: next } });
    for (const order of [[first, current], [current, first]]) {
        await withChat(async (log, _chat, card) => {
            for (const frame of order) await log(frame);
            await log(doneFrame(1, R3, 'Continue finished.', { tool_calls: 2 }, { task_id: next, suggested_name: 'Continue: Read the catalog',
                terminal_time: end(1, '2026-11-18T11:25:00Z', R3) }));
            assert.equal(card().hidden, true);
            const successor = card(next);
            const history = successor.querySelector(`[data-episode-task="${TASK}"]`);
            assert.match(history.firstElementChild.textContent, new RegExp(`${clock(secondClock)}–${clock(T2)}.*Done.*${SECOND}`));
            assert.doesNotMatch(history.firstElementChild.textContent, /First attempt words|reason has no description/);
            assert.equal(history.querySelector('.chat-episode-codes').hidden, true);
            assert.doesNotMatch(facts(successor).episodes, new RegExp(clock(firstClock)));
        });
    }
});

test('a cold-opened still-running retry keeps the open turn\'s counters; the frozen ending owns none', async () => {
    const stopped = saved({ type: 'task_done', task_id: TASK, status: 'failed', reason_code: 'tool_error',
        ts: T1, rounds: 4, tool_calls: 3,
        outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'infra_failed' } } },
        'old:1', { task_attempt: 1, task_phase: 'unfinished', outcome_final: false });
    // The gateway attaches the task's whole invocation journal to the page's carrier row, and a summary
    // on the page is that carrier: its observations stay the task's, its frozen scalars do not.
    const journalled = { ...stopped, history_id: 'old:2',
        tool_evidence: { legacy: { calls: 9 }, coverage: { matched: 9, shown: 9 } } };
    const narration = { role: 'assistant', task_id: TASK, ts: T2, history_id: 'progress:8', is_progress: true,
        narration: true, text: 'Counting the east bed', task_phase: 'unfinished', outcome_final: false };
    // A task whose stored result is still open has every row marked `unfinished`; its own carrier — the
    // latest progress row — still folds the current attempt's stamped count (the result's tool_calls).
    const carrier = { ...narration, history_id: 'progress:9', text: 'Counting it again',
        task_attempt: 2, tool_calls: 7 };
    for (const [name, rows, ownCount] of [
        ['the open attempt\'s live carrier folds its own count', [stopped, carrier], /7 tool calls/],
        ['the frozen ending\'s journal remains the task\'s; its scalars own nothing', [journalled, narration], /9 tool calls/],
    ]) {
        await withChat(async (_log, _chat, card, { read }) => {
            await read(rows);
            const open = facts(card());
            assert.notEqual(open.finished, '1', name);
            assert.doesNotMatch(open.details, /4 · usable solve responses/,
                `${name}: the earlier attempt's rounds do not own the open turn's card`);
            assert.doesNotMatch(open.meta, /3 tool calls/,
                `${name}: the earlier attempt's calls do not own it either`);
            assert.match(open.meta, ownCount, `${name}: the card keeps its live counters`);
        });
    }
});

test('a selected ending recorded zero replaces the page summary it outranks', async () => {
    // A preliminary publication of the selected attempt, and the older row the gateway joins the
    // current outcome to; _separate_current_outcome carries the recorded rounds of that outcome.
    const preliminary = () => saved(doneFrame(2, T2, SECOND, { rounds: 4, tool_calls: 3 }), 'preliminary');
    const joined = () => ({ ...saved(doneFrame(1, T1, FIRST, { rounds: 9, tool_calls: 3 }), 'old'),
        current_outcome: { ...latePass(R3, 2).current_outcome, rounds: 0, wake_summary: SECOND } });
    for (const order of ['preliminary, then joined', 'joined, then preliminary']) {
        await withChat(async (_log, _chat, card, { read }) => {
            await read(order === 'preliminary, then joined' ? [preliminary(), joined()] : [joined(), preliminary()]);
            const shown = facts(card());
            assert.match(shown.details, /<dt>Rounds<\/dt><dd>0 · usable solve responses<\/dd>/, order);
            assert.doesNotMatch(shown.details, /4 · usable solve responses/, order);
            assert.doesNotMatch(shown.details, /9 · usable solve responses/, order);
        });
    }
});

test('a frozen row of the open attempt still states its recorded zero', async () => {
    // The gateway marks every row of a task whose stored result is still open as unfinished. The open
    // attempt row is frozen then, and its recorded zero is still the count of that attempt.
    const narration = { role: 'assistant', task_id: TASK, ts: T2, history_id: 'progress:8', is_progress: true,
        narration: true, text: 'Counting the east bed', task_phase: 'unfinished', outcome_final: false,
        task_attempt: 2, tool_calls: 7 };
    const frozen = saved({ type: 'task_done', task_id: TASK, status: 'failed', reason_code: 'tool_error',
        ts: T1, rounds: 0, tool_calls: 3,
        outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'infra_failed' } } }, 'facts:2',
        { task_attempt: 2, task_phase: 'unfinished', outcome_final: false });
    await withChat(async (_log, _chat, card, { read }) => {
        await read([narration, frozen]);
        const open = facts(card());
        assert.notEqual(open.finished, '1');
        assert.match(open.details, /<dt>Rounds<\/dt><dd>0 · usable solve responses<\/dd>/);
        assert.match(open.meta, /7 tool calls/);
    });
});
