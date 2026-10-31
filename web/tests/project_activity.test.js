// The sidebar's status marks read each room in the server's work projection (`rooms`): its shape (the server's
// `mark`), whether something waits for the owner (its turn), and its word with the line of its first work. The
// rooms here are the server's own output (fixtures/work_projection_cases.json).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
    IDLE_SUMMARY,
    MARK_RANK,
    buildProjectActivityIndex,
    markWords,
    reconcileProjectRooms,
    summarizeRoom,
    waitingModel,
} from '../modules/project_activity.js';
import { MARK_SHAPES } from '../modules/status_mark.js';
import { clock } from '../modules/work_status.js';
import { SERVER_CASES } from './work_fixture.js';

// One server body whose case room is re-keyed to a Project chat id.
function body(name, chatId = 42, extra = {}) {
    const served = structuredClone(SERVER_CASES[name]);
    const rooms = Object.fromEntries(Object.values(served.rooms).map((room) => [String(chatId), room]));
    return { supervisor_ready: true, ...served, rooms, ...extra };
}
const summary = (name) => {
    const served = SERVER_CASES[name];
    return summarizeRoom(Object.values(served.rooms)[0], served.works);
};
// The owner's press as the fixture's fences record it, in the viewer's clock.
const PRESSED = clock('2026-10-17T15:00:00Z');

test('the shape is the server room\'s, passed through: moving work, a still queue or wait, a pause, a ring', () => {
    assert.equal(summary('held_continue').mark, 'moving');
    assert.equal(summary('queued_and_pausing').mark, 'moving', 'a Pause still settling keeps the motion');
    assert.deepEqual([summary('paused_by_owner').mark, summary('paused_by_mind').mark], ['pause', 'pause']);
    assert.deepEqual([summary('sleeping').mark, summary('model_access').mark], ['still', 'still'], 'waits that end by themselves');
    assert.equal(summary('unknown').mark, 'ring');
    // A paused work whose review still finishes: the room moves while the review runs.
    assert.equal(summary('paused_review_finishing').mark, 'moving');
    assert.equal(summary('unconfirmed_run').mark, 'ring');
    const queued = summarizeRoom({ activity: 'queued', mark: 'still', turn: [], amber: false, works: [] }, []);
    assert.deepEqual([queued.mark, queued.owner, queued.label], ['still', false, 'Queued']);
    assert.equal(summarizeRoom({ activity: 'idle', mark: 'none' }), IDLE_SUMMARY);
    assert.equal(summarizeRoom(null), IDLE_SUMMARY);
    // A room that is not at rest with a shape outside the grammar (an older or newer server) is unconfirmed, never
    // something it is not.
    for (const mark of ['dot', 'sparkle', undefined]) {
        assert.equal(summarizeRoom({ activity: 'waiting', mark, turn: [], amber: false, works: [] }).mark, 'ring', String(mark));
    }
    for (const name of Object.keys(SERVER_CASES)) assert.ok(MARK_SHAPES.includes(summary(name).mark), name);
});

test('the owner\'s move is the room\'s turn alone: questions and owner actions, never a Pause by the owner or Ouroboros', () => {
    assert.equal(summary('direct_question').owner, true);
    assert.equal(summary('spending_limit').owner, true);
    assert.equal(summary('model_access').owner, true);
    assert.equal(summary('paused_by_owner').owner, false);
    assert.equal(summary('paused_by_mind').owner, false);
    assert.equal(summary('held_continue').owner, false);
    // A turn the server lists in a room it could not confirm is still the owner's move, and its words say it.
    const unsure = summarizeRoom({ activity: 'paused', mark: 'ring', confirmed: false, amber: true, works: [],
        turn: [{ kind: 'action', task_id: 't' }] }, []);
    assert.deepEqual([unsure.mark, unsure.owner, unsure.label], ['ring', true, 'Paused · Something here waits for you']);
});

test('the label is the room word and its first work\'s line: the episode\'s reason, then what still runs beside it', () => {
    assert.equal(summary('held_continue').label, 'Working · Continue starts when 3 helpers of the earlier attempt finish');
    assert.equal(summary('paused_by_owner').label, `Paused · You paused it · ${PRESSED}`);
    assert.equal(summary('paused_by_mind').label, 'Paused · Ouroboros paused it: pause before the update you asked for');
    assert.equal(summary('paused_unattributed').label, 'Paused', 'a Pause that declared nobody names nobody');
    assert.equal(summary('sleeping').label, 'Waiting · Sleeping until it is woken');
    assert.equal(summary('direct_question').label, 'Waiting for your answer', 'the word already says the question');
    assert.equal(summary('queued_and_pausing').label, `Pausing… · You paused it · ${PRESSED}`);
    assert.equal(summary('unknown').label, 'Activity unconfirmed');
    assert.equal(summary('paused_review_finishing').label, `Paused · You paused it · ${PRESSED} · Review work finishing`);
    assert.equal(summary('ended_with_runs').label, 'Working · 2 external runs still working');
    // Unconfirmed, and the line says what: a ring is no reason to hide it.
    assert.equal(summary('unconfirmed_run').label, 'Activity unconfirmed · 1 external run may still be running');
});

test('the accessible name says what waits for the owner whenever the word does not: amber is never colour alone', () => {
    assert.equal(summary('spending_limit').label, 'Paused · Spending limit reached · Something here waits for you');
    assert.equal(summary('model_access').label,
        'Waiting · The model needs you: sign in or restore access · Something here waits for you');
    // A question in one work while another works: the room's word is Working, and the question is said beside it.
    const room = { activity: 'working', transition: 'finalizing', call: '', mark: 'moving', amber: true, works: ['f', 'q'],
        turn: [{ kind: 'question', task_id: 'q', quiz_id: 'q1' }] };
    assert.equal(summarizeRoom(room, []).label, 'Finalizing… · A question waits for your answer');
    // Nothing waits for the owner, or the word already says the question: nothing is added.
    assert.equal(summarizeRoom({ ...room, amber: false, turn: [] }, []).label, 'Finalizing…');
    assert.equal(summary('direct_question').label, 'Waiting for your answer');
});

test('a complete ready body clears absent rooms; a partial, unready or unavailable one keeps them unconfirmed', () => {
    const first = reconcileProjectRooms(new Map(), body('held_continue'));
    assert.equal(first.complete, true);
    assert.equal(first.rooms.get('42').mark, 'moving');
    for (const data of [
        { ...body('held_continue'), works: [], rooms: {}, works_complete: false },
        { ...body('held_continue'), works: [], rooms: {}, supervisor_ready: false },
        { ...body('held_continue'), works: [], rooms: {}, works_complete: 1 },
        { works_complete: true, supervisor_ready: true, rooms: {} },  // no works array
        {}, undefined,
    ]) {
        const next = reconcileProjectRooms(first.rooms, data);
        assert.equal(next.complete, false, JSON.stringify(data));
        assert.deepEqual(next.rooms.get('42'), { mark: 'ring', owner: false, word: 'Activity unconfirmed', note: '',
            label: 'Activity unconfirmed' }, 'kept, unconfirmed: a ring, never the owner\'s move it cannot see');
    }
    const empty = reconcileProjectRooms(first.rooms, { supervisor_ready: true, works: [], rooms: {}, works_complete: true });
    assert.equal(empty.rooms.size, 0);
});

test('a partial body marks only its omissions unknown; a room it lists describes itself; inputs stay untouched', () => {
    const initial = reconcileProjectRooms(new Map(), body('held_continue', 42));
    const partialBody = { ...body('paused_by_owner', 43), works_complete: false };
    const partial = reconcileProjectRooms(initial.rooms, partialBody);
    assert.equal(partial.rooms.get('42').mark, 'ring');
    assert.deepEqual([partial.rooms.get('43').mark, partial.rooms.get('43').label], ['pause', `Paused · You paused it · ${PRESSED}`]);
    assert.equal(initial.rooms.get('42').mark, 'moving', 'the previous map is not mutated');
    const restored = reconcileProjectRooms(partial.rooms, body('held_continue', 42));
    assert.equal(restored.rooms.get('42').mark, 'moving');
    assert.equal(restored.rooms.has('43'), false, 'a complete body proves the paused room idle once it is not listed');
});

test('the index reads Project rooms only; the header\'s one mark is the strongest shape and the owner\'s move', () => {
    const rooms = new Map([
        ['42', summary('paused_by_owner')], ['43', summary('spending_limit')], ['1', summary('held_continue')],
    ]);
    const index = buildProjectActivityIndex(rooms, [42, 43, 44]);
    assert.deepEqual([...index.byChat.keys()], ['42', '43']);
    assert.deepEqual([index.aggregate.mark, index.aggregate.owner, index.aggregate.label],
        ['pause', true, 'Paused · Something here waits for you'], 'Main\'s own room never moves the Projects header');
    const moving = buildProjectActivityIndex(new Map([['42', summary('held_continue')], ['43', summary('unknown')]]), ['42', '43']);
    assert.deepEqual([moving.aggregate.mark, moving.aggregate.owner, moving.aggregate.label],
        ['moving', false, 'Working · Activity unconfirmed']);
    // The strongest shape wins whatever the order: moving, still, a pause, a ring.
    const shapes = { moving: 'held_continue', still: 'sleeping', pause: 'paused_by_owner', ring: 'unknown' };
    assert.deepEqual(MARK_RANK, ['moving', 'still', 'pause', 'ring', 'none']);
    for (const [index, strongest] of MARK_RANK.slice(0, 4).entries()) {
        const present = MARK_RANK.slice(index, 4).reverse();  // this shape and every weaker one, weakest first
        const set = new Map(present.map((shape, i) => [String(50 + i), summary(shapes[shape])]));
        assert.equal(buildProjectActivityIndex(set, [...set.keys()]).aggregate.mark, strongest, present.join(', '));
    }
    assert.equal(buildProjectActivityIndex(new Map(), [42]).aggregate, IDLE_SUMMARY);
    assert.equal(buildProjectActivityIndex().aggregate, IDLE_SUMMARY);
});

test('a mark\'s words: its place\'s label and new messages, in one phrase', () => {
    assert.equal(markWords(summary('paused_by_owner')), `Paused · You paused it · ${PRESSED}`);
    assert.equal(markWords(summary('paused_by_owner'), true), `Paused · You paused it · ${PRESSED} · new messages`);
    assert.equal(markWords(IDLE_SUMMARY, true), 'new messages', 'a room at rest with new messages says only that');
    assert.equal(markWords(IDLE_SUMMARY), '');
    assert.equal(markWords(), '');
});

test('model waits use the existing current-attempt rule (Activity reads it)', () => {
    const row = { phase: 'working', task_attempt: 2,
        model_waits: { previous: { wait_id: 'previous', revision: 1, task_attempt: 1, state: 'waiting', reason: 'quota',
            requires_owner_action: true } } };
    assert.deepEqual(waitingModel(row), { waiting: false, ownerAction: false });
    row.model_waits.current = { wait_id: 'current', revision: 1, task_attempt: 2, state: 'waiting', reason: 'auth' };
    assert.deepEqual(waitingModel(row), { waiting: true, ownerAction: false });
    // Whether it needs the owner is the host's own fact on the admitted wait, never its reason's words.
    row.model_waits.current.requires_owner_action = true;
    assert.deepEqual(waitingModel(row), { waiting: true, ownerAction: true });
    row.model_waits.current.state = 'resolved';
    assert.deepEqual(waitingModel(row), { waiting: false, ownerAction: false });
    // A malformed wait row (no id, revision or reason) is dropped exactly as the chat card drops it.
    assert.deepEqual(waitingModel({ task_attempt: 1, model_waits: { w: { state: 'waiting', task_attempt: 1,
        requires_owner_action: true } } }), { waiting: false, ownerAction: false });
});
