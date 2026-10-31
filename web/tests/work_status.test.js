// One work, one status in the browser (work_status.js) over the server's real projection: the cases in
// fixtures/work_projection_cases.json are what gateway/work_projection.works_snapshot builds for them
// (tests/test_work_projection_web_fixture.py keeps the two in step). The client says those facts in
// short words, keeps what an incomplete body omitted, and lays each work's controls on the shared doors.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { applyPayload, CODE_PREFIX, currentPayload, setMissTransport } from '../modules/i18n.js';
import {
    ACTION_FINALIZE, ACTION_HURRY, ACTION_PAUSE, ACTION_RESUME, ACTION_STOP_EARLIER, ACTION_STOP_NOW,
    TASK_CONTROL_LABELS, requestStopAll,
} from '../modules/task_control_menu.js';
import { censusTaskPhase, desiredLiveCardPhase, pausePhaseLabel, taskActivityPresentation } from '../modules/task_phase_chip.js';
import { markView } from '../modules/status_mark.js';
import { summarizeRoom } from '../modules/project_activity.js';
import {
    OUTCOME_TONES, activityWord, applyCardWorkViews, cardControls, cardMenuActions, cardWorkView, clock, emptyWorkState,
    memberState, ownStopPress, reasonText, reconcileWorkState, resumeTargets, roomStatusView, workControls, workLine,
    workMenuActions, workQuestion, workRowView,
} from '../modules/work_status.js';
import { SERVER_REASONS, room, serverMark, work as builtWork } from './work_fixture.js';

setMissTransport(() => Promise.resolve(null));
const CASES = JSON.parse(readFileSync(new URL('./fixtures/work_projection_cases.json', import.meta.url), 'utf8')).cases;
// One /api/state body for a case, as the server sends it (supervisor ready unless said otherwise).
const body = (name, extra = {}) => ({ supervisor_ready: true, ...structuredClone(CASES[name]), ...extra });
const work = (name, index = 0) => structuredClone(CASES[name].works[index]);
const stateOf = (name, chatId = 42, extra = {}) => reconcileWorkState(emptyWorkState(), body(name, extra), chatId);
// The owner's press as the fixture's fences record it, in the viewer's clock (the press time, never a restamped one).
const PRESSED = clock('2026-10-17T15:00:00Z');

test('the room says one short word for every activity, transition and exact call', () => {
    const words = {
        held_continue: 'Working', working_call: 'Sending', paused_by_owner: 'Paused', spending_limit: 'Paused',
        model_access: 'Waiting', sleeping: 'Waiting', queued_and_pausing: 'Pausing…', finalizing_failed: 'Finalizing…',
        unknown: 'Activity unconfirmed', ended_with_runs: 'Working', paused_review_finishing: 'Paused',
        unconfirmed_run: 'Activity unconfirmed', paused_unattributed: 'Paused', pausing_with_queued_helper: 'Pausing…',
        two_works_one_pausing: 'Working', working_with_review: 'Working', paused_stop_unknown: 'Paused',
    };
    for (const [name, word] of Object.entries(words)) assert.equal(roomStatusView(stateOf(name)).text, word, name);
    // A waiting room whose turn is a question says so in its word, in the amber ink beside the owner's dot; no note
    // repeats it.
    const asked = roomStatusView(stateOf('direct_question', 1));
    assert.deepEqual([asked.text, asked.tone, asked.owner, asked.amber], ['Waiting for your answer', 'wait', true, '']);
    for (const [call, word] of [['preparing', 'Preparing'], ['thinking', 'Thinking'], ['working', 'Working'], ['', 'Working']]) {
        assert.equal(activityWord({ activity: 'working', transition: 'none', call }), word, call);
    }
    assert.equal(activityWord({ activity: 'queued' }), 'Queued');
    assert.equal(activityWord({ activity: 'idle' }), 'Online');
    assert.equal(activityWord({ activity: 'something new' }), 'Activity unconfirmed');
});

test('the mark is the server\'s shape; whose move it is stands beside it, said in words where the word does not', () => {
    const view = (name, chatId = 42) => roomStatusView(stateOf(name, chatId));
    const facts = (name, chatId) => ['mark', 'tone', 'owner', 'amber'].map((key) => view(name, chatId)[key]);
    assert.deepEqual(facts('held_continue'), ['moving', 'live', false, '']);
    assert.deepEqual(facts('paused_by_owner'), ['pause', 'held', false, '']);
    assert.deepEqual(facts('paused_by_mind'), ['pause', 'held', false, ''], "Ouroboros's own Pause is not the owner's move");
    // An action only the owner takes: the word keeps its neutral ink, and the control's name says what waits.
    assert.deepEqual(facts('spending_limit'), ['pause', 'held', true, 'action']);
    assert.deepEqual(facts('model_access'), ['still', 'held', true, 'action']);
    assert.deepEqual(facts('sleeping'), ['still', 'held', false, ''], 'a wait that ends by itself holds still dots');
    assert.deepEqual(facts('direct_question', 1), ['still', 'wait', true, ''], 'its own word says the question');
    assert.deepEqual(facts('unknown'), ['ring', 'unknown', false, '']);
    // The room moves while anything in it runs: a paused work's finishing review moves its room, under the word Paused.
    assert.deepEqual([view('paused_review_finishing').text, ...facts('paused_review_finishing')],
        ['Paused', 'moving', 'held', false, '']);
    assert.deepEqual([view('unconfirmed_run').text, ...facts('unconfirmed_run')], ['Activity unconfirmed', 'ring', 'unknown', false, '']);
    const pausing = view('queued_and_pausing');
    assert.deepEqual([pausing.text, pausing.tone, pausing.mark], ['Pausing…', 'held', 'moving'],
        'a Pause still settling: neutral ink, the server\'s moving mark');
    const queued = roomStatusView({ room: room('queued') });
    assert.deepEqual([queued.text, queued.mark, queued.tone], ['Queued', 'still', 'queued']);
    // A working room with a question elsewhere keeps its word and its shape; the owner's dot takes the mark's place.
    const mixed = roomStatusView({ room: room('working', { turn: [{ kind: 'question', task_id: 'x', quiz_id: 'q' }] }) });
    assert.deepEqual([mixed.text, mixed.tone, mixed.mark, mixed.owner, mixed.amber], ['Working', 'live', 'moving', true, 'question']);
    assert.deepEqual(markView(mixed), { shape: 'dot', hue: 'owner' });
    assert.deepEqual(markView(view('held_continue')), { shape: 'moving', hue: 'ink' });
    // Every room the server rolls up keeps the server's own mark: nothing on the client chooses another shape.
    for (const [name, { rooms }] of Object.entries(CASES)) {
        for (const [chat, served] of Object.entries(rooms || {})) assert.equal(view(name, Number(chat)).mark, served.mark, name);
    }
});

test('only the client facts overlay the room, each with its mark: socket closed, own send, start, own Stop press', () => {
    const working = stateOf('held_continue');
    assert.deepEqual(roomStatusView(working, { connected: false }),
        { kind: 'offline', text: 'Reconnecting…', tone: 'offline', mark: 'ring', owner: false, amber: '' });
    assert.equal(roomStatusView(stateOf('spending_limit'), { connected: false }).owner, false,
        'a closed socket proves nothing current, the owner\'s move included');
    const idle = emptyWorkState();
    assert.deepEqual(roomStatusView(idle),
        { kind: 'online', text: 'Online', tone: 'online', mark: 'none', owner: false, amber: '' });
    assert.deepEqual(roomStatusView(idle, { ready: false }),
        { kind: 'starting', text: 'Starting…', tone: 'starting', mark: 'none', owner: false, amber: '', title: '' });
    // The owner's own unconfirmed send outranks a quiet room, never work the room already does; it moves.
    const said = (state, overlay) => ['text', 'mark', 'owner'].map((key) => roomStatusView(state, overlay)[key]);
    assert.deepEqual(said(idle, { sending: true }), ['Sending', 'moving', false]);
    assert.deepEqual(said(stateOf('paused_by_owner'), { sending: true }), ['Sending', 'moving', false]);
    assert.deepEqual(said(stateOf('spending_limit'), { sending: true }), ['Sending', 'moving', true],
        'what waits for the owner still waits while a message goes out');
    assert.equal(roomStatusView(working, { sending: true }).text, 'Working');
    // The owner's own Stop press on every working work says Cancelling… or Finalizing… until the server catches up.
    const records = new Map([['c', { cancelPendingPolicy: 'immediate' }]]);
    assert.equal(ownStopPress(working, records), 'immediate');
    assert.deepEqual(said(working, { stopping: 'immediate' }), ['Cancelling…', 'moving', false]);
    assert.deepEqual(said(working, { stopping: 'finalize' }), ['Finalizing…', 'moving', false]);
    assert.equal(ownStopPress(working, new Map()), '');
    assert.equal(ownStopPress(stateOf('paused_by_owner'), new Map([['o', { cancelPendingPolicy: 'immediate' }]])), '',
        'a press on work that does not work changes no word');
});

// A direct turn before its first model request: every surface says Preparing with the start step the turn stands
// at (the bar keeps it apart from its word, to drop first); once its Main call started it says the word its census
// move gives, never Sending or Thinking before the census sees the request go out.
test('a turn still preparing says its start step beside Preparing on the bar and on every other surface', () => {
    const state = stateOf('direct_preparing', 1);
    const bar = roomStatusView(state);
    assert.deepEqual([bar.text, bar.step, bar.tone, bar.mark], ['Preparing', 'Building context', 'live', 'moving']);
    assert.equal(summarizeRoom(state.room, state.works).label, 'Preparing · Building context');
    const row = workRowView(state.works[0]);
    assert.deepEqual([row.word, row.secondary], ['Preparing', 'Building context']);
    const card = cardWorkView(state.works[0]);
    assert.deepEqual([card.text, card.secondary], ['Preparing', 'Building context']);
    assert.equal(roomStatusView(state, { stopping: 'immediate' }).text, 'Cancelling…', 'the owner\'s own Stop says only its word');
    const called = stateOf('direct_requested', 1);
    assert.deepEqual([roomStatusView(called).text, workRowView(called.works[0]).secondary], ['Working', '']);
    assert.ok(!('step' in roomStatusView(called)) && !('step' in roomStatusView(stateOf('working_call'))),
        'a working room says no start step');
    // Only a preparing room says a line in the bar: a working room's reason stays beside its work.
    assert.equal(workLine(stateOf('held_continue').works[0]), 'Continue starts when 3 helpers of the earlier attempt finish');
    assert.equal(roomStatusView(stateOf('held_continue')).text, 'Working');
});

test('reasons read their code through the catalog; counts stay parameters; an unknown code reads its sentence', () => {
    const held = work('held_continue');
    assert.equal(reasonText(held.because[0]), 'Continue starts when 3 helpers of the earlier attempt finish');
    assert.equal(reasonText({ code: 'brand_new_reason', text: 'A sentence no client knows yet' }), 'A sentence no client knows yet');
    assert.equal(reasonText({ code: '', text: 'raw' }), 'raw');
    assert.equal(reasonText(null), '');
    // The episode's own sentence, as the server says it: who paused, and Ouroboros in its own words.
    assert.equal(reasonText(work('paused_by_owner').because[0]), 'You paused it');
    assert.equal(reasonText(work('paused_by_mind').because[0]), 'Ouroboros paused it: pause before the update you asked for');
    assert.equal(reasonText(work('spending_limit').because[0]), 'Spending limit reached');
    // A Pause that declared nobody says nothing, never its code.
    assert.equal(reasonText(work('paused_unattributed').because[0]), '');
    assert.equal(reasonText({ code: 'paused_unattributed' }), 'paused_unattributed', 'a reason without its sentence reads its code');
    const unconfirmed = work('unconfirmed_run').tails[0];
    assert.equal(reasonText(unconfirmed), '1 external run may still be running');
    assert.equal(reasonText({ ...unconfirmed, count: 3, text: '3 external runs may still be running' }), '3 external runs may still be running');
    const prior = currentPayload();
    try {
        applyPayload({ language: 'ru', english: false, entries: {
            'Continue starts when {count} helpers of the earlier attempt finish':
                { text: 'Продолжение начнётся, когда закончат {count} помощника прошлой попытки' },
            '{count} external runs may still be running': { text: 'Возможно, ещё работают внешние запуски: {count}' },
            'Ouroboros: {reason}': { text: 'Уроборос: {reason}' },
            [CODE_PREFIX + 'work.because.earlier_attempt_finishing']: { text: 'Продолжение начнётся, когда закончит помощник' },
            [CODE_PREFIX + 'work.because.model_access']: { text: 'Модели нужен вход' },
            [CODE_PREFIX + 'work.because.paused_by_owner']: { text: 'Вы поставили на паузу' },
            [CODE_PREFIX + 'work.because.paused_by_mind']: { text: 'Поставлено на паузу Уроборосом' },
        } });
        // The count is a parameter of the translated template, never a number baked into a translation.
        assert.equal(reasonText(held.because[0]), 'Продолжение начнётся, когда закончат 3 помощника прошлой попытки');
        assert.equal(reasonText({ ...held.because[0], count: 18, text: 'Continue starts when 18 helpers of the earlier attempt finish' }),
            'Продолжение начнётся, когда закончат 18 помощника прошлой попытки');
        assert.equal(reasonText({ ...held.because[0], count: 1 }), 'Продолжение начнётся, когда закончит помощник');
        assert.equal(reasonText({ ...unconfirmed, count: 2, text: '2 external runs may still be running' }), 'Возможно, ещё работают внешние запуски: 2');
        assert.equal(reasonText(work('model_access').because[0]), 'Модели нужен вход');
        assert.equal(reasonText(work('paused_by_owner').because[0]), 'Вы поставили на паузу');
        // Ouroboros's sentence is its own prose: only the frame is translated, never its words, never by its code.
        assert.equal(reasonText(work('paused_by_mind').because[0]), 'Поставлено на паузу Уроборосом: pause before the update you asked for');
        assert.equal(reasonText({ code: 'paused_by_mind', text: 'Ouroboros paused it' }), 'Поставлено на паузу Уроборосом');
        assert.equal(reasonText(work('paused_unattributed').because[0]), '');
        assert.equal(reasonText({ code: 'brand_new_reason', text: 'Unknown sentence' }), 'Unknown sentence');
    } finally { applyPayload(prior); }
});

test('every reason the server can send reads in the server\'s own sentence; a counted one in its template', () => {
    assert.ok(Object.keys(SERVER_REASONS).length > 10);
    for (const [code, forms] of Object.entries(SERVER_REASONS)) {
        for (const [count, text] of Object.entries(forms)) {
            const reason = count ? { code, text, count: Number(count) } : { code, text };
            assert.equal(reasonText(reason), text, `${code} ${count}`);
        }
    }
    // A counted code this client does not know keeps the server's sentence: no translation bakes its number.
    assert.equal(reasonText({ code: 'brand_new_count', count: 2, text: '2 brand new things' }), '2 brand new things');
});

test('the line beside a word never says what the word already says, in any language', () => {
    const asked = work('direct_question');
    assert.deepEqual([workRowView(asked).word, workRowView(asked).secondary], ['Waiting for your answer', '']);
    const prior = currentPayload();
    try {
        applyPayload({ language: 'ru', english: false, entries: {
            [CODE_PREFIX + 'task.chip.waiting_for_answer']: { text: 'Ждёт вашего ответа' } } });
        // The word is translated and the server's sentence is not: still nothing beside the word.
        assert.deepEqual([workRowView(asked).word, workRowView(asked).secondary], ['Ждёт вашего ответа', '']);
        // Another reason beside the same word still shows.
        assert.equal(workRowView(work('model_access')).secondary, 'The model needs you: sign in or restore access');
    } finally { applyPayload(prior); }
});

test('the owner\'s own Stop press on a work\'s card says itself on the row at once; work that does not work keeps its word', () => {
    const live = work('working_call');
    assert.deepEqual(['word', 'secondary', 'tone', 'mark'].map((key) => workRowView(live, { stopping: 'finalize' })[key]),
        ['Finalizing…', '', 'live', 'moving']);
    assert.equal(workRowView(live, { stopping: 'immediate' }).word, 'Cancelling…');
    assert.equal(workRowView(live).word, 'Sending');
    assert.equal(workRowView(work('paused_by_owner'), { stopping: 'immediate' }).word, 'Paused');
});

test('a sleep whose chosen helper still awaits its own Resume says so beside Waiting, with no new door or turn', () => {
    const state = stateOf('sleeping_on_paused_helper');
    const row = workRowView(state.works[0]);
    assert.deepEqual(row, { word: 'Waiting', tone: 'held', mark: 'still', owner: false, outcome: '', focus: '',
        secondary: 'Sleeping until it is woken · 1 helper it waits for is paused until Ouroboros resumes it' });
    // The sleeping root keeps exactly its own controls; the helper's Resume stays Ouroboros's selection.
    const grouped = workControls(state.works[0]);
    assert.deepEqual(grouped, { stop: ['sw'], stopEarlier: [], pause: 'sw', resume: '', hurry: 'sw', wrapUp: 'sw' });
    assert.deepEqual([roomStatusView(state).owner, roomStatusView(state).amber], [false, '']);
    // A translation keeps the count; the helper's own Activity sentence is unchanged.
    assert.equal(reasonText({ code: 'awaited_helpers_mind', count: 3, text: 'x' }),
        '3 helpers it waits for are paused until Ouroboros resumes them');
    assert.equal(memberState(state, 'sw-c').because[0].code, 'awaiting_selection');
});

test('a reusable slot gets no control by its id; a real target keeps its own', () => {
    const slot = { controls: [{ action: 'stop', target: 'active' }, { action: 'pause', target: 'active' },
        { action: 'stop', target: 't-1' }] };
    assert.deepEqual(workControls(slot), { stop: ['t-1'], stopEarlier: [], pause: '', resume: '', hurry: '', wrapUp: '' });
});

test('a held Continue with three helpers of its failed earlier attempt: one working row with the reason, Stop and Stop earlier attempt, no Resume', () => {
    const state = stateOf('held_continue');
    assert.equal(state.works.length, 1, 'one work, one row: the Continue and its earlier helpers are inside it');
    const row = workRowView(state.works[0]);
    assert.deepEqual(row, { word: 'Working', secondary: 'Continue starts when 3 helpers of the earlier attempt finish',
        tone: 'live', mark: 'moving', owner: false, outcome: '', focus: '' });
    const grouped = workControls(state.works[0]);
    assert.deepEqual(grouped, { stop: ['c', 'p'], stopEarlier: ['p'], pause: '', resume: '', hurry: '', wrapUp: '' });
    assert.deepEqual(workMenuActions(grouped), [ACTION_STOP_EARLIER, ACTION_STOP_NOW]);
    assert.equal(TASK_CONTROL_LABELS[ACTION_STOP_EARLIER], 'Stop earlier attempt');
    // The held Continue's own card says the work's word and reason, and offers no Resume.
    const card = cardControls(state, 'c');
    assert.deepEqual([card.resume, card.menu, Boolean(card.work)], [false, true, true]);
    assert.deepEqual(cardWorkView(state.works[0]), { phase: 'working', text: 'Working',
        secondary: 'Continue starts when 3 helpers of the earlier attempt finish', classes: 'working',
        tone: 'live', waiting: false });
    // A count says itself the same from one helper to eighteen; the structure does not change.
    const many = { ...state.works[0], because: [{ code: 'earlier_attempt_finishing', count: 18,
        text: 'Continue starts when 18 helpers of the earlier attempt finish' }] };
    assert.deepEqual(Object.keys(workRowView(many)), Object.keys(row));
    assert.equal(workRowView(many).tone, row.tone);
});

test('rows: their word, the line beside it, tone, mark and the owner\'s move from the work alone', () => {
    const row = (name, index = 0) => workRowView(work(name, index));
    const quiet = { owner: false, outcome: '', focus: '' };
    assert.deepEqual(row('paused_by_owner'), { word: 'Paused', secondary: `You paused it · ${PRESSED}`, tone: 'held',
        mark: 'pause', ...quiet });
    assert.deepEqual([row('paused_by_mind').word, row('paused_by_mind').secondary],
        ['Paused', 'Ouroboros paused it: pause before the update you asked for']);
    assert.deepEqual([row('paused_unattributed').word, row('paused_unattributed').secondary], ['Paused', ''],
        'a Pause that declared nobody names nobody');
    assert.deepEqual(row('spending_limit'), { word: 'Paused', secondary: 'Spending limit reached', tone: 'held', mark: 'pause',
        ...quiet, owner: true });
    assert.deepEqual(row('model_access'), { word: 'Waiting', secondary: 'The model needs you: sign in or restore access',
        tone: 'held', mark: 'still', ...quiet, owner: true });
    // The word already says the owner's question: the reason does not repeat it; the owner's dot stands beside it.
    assert.deepEqual(row('direct_question'), { word: 'Waiting for your answer', secondary: '', tone: 'wait', mark: 'still',
        ...quiet, owner: true });
    assert.deepEqual(markView(row('direct_question')), { shape: 'dot', hue: 'owner' });
    // A Pause still settling: the owner's own Pause keeps the neutral ink while the work still moves.
    assert.deepEqual(['word', 'secondary', 'tone', 'mark'].map((key) => row('queued_and_pausing', 0)[key]),
        ['Pausing…', `You paused it · ${PRESSED}`, 'held', 'moving']);
    assert.deepEqual(['word', 'tone', 'mark'].map((key) => row('queued_and_pausing', 1)[key]), ['Queued', 'queued', 'still']);
    assert.deepEqual(['tone', 'mark'].map((key) => row('working_call')[key]), ['live', 'moving']);
    assert.deepEqual(['word', 'tone', 'mark'].map((key) => row('sleeping')[key]), ['Waiting', 'held', 'still']);
    // An outcome stays beside Finalizing…, its dot in the outcome's tone; unknown stays unknown, a ring.
    assert.deepEqual(row('finalizing_failed'), { word: 'Failed', secondary: 'Finalizing…', tone: 'failed', mark: 'moving',
        ...quiet, outcome: 'error' });
    assert.deepEqual(markView(row('finalizing_failed')), { shape: 'dot', hue: 'failed' });
    assert.deepEqual(['word', 'tone', 'mark'].map((key) => row('unknown')[key]), ['Activity unconfirmed', 'unknown', 'ring']);
    assert.deepEqual([row('ended_with_runs').word, row('ended_with_runs').secondary], ['Working', '2 external runs still working']);
    assert.deepEqual([row('helper_after_parent').word, row('helper_after_parent').secondary], ['Working', '1 helper still working']);
    // A paused work whose review still finishes stays Paused, its bars on its row; the review is the line's second fact.
    assert.deepEqual(['word', 'secondary', 'tone', 'mark'].map((key) => row('paused_review_finishing')[key]),
        ['Paused', `You paused it · ${PRESSED} · Review work finishing`, 'held', 'pause']);
    // A run nobody can confirm stays unconfirmed: a ring, and the line says what is not confirmed.
    assert.deepEqual(['word', 'secondary', 'tone', 'mark'].map((key) => row('unconfirmed_run')[key]),
        ['Activity unconfirmed', '1 external run may still be running', 'unknown', 'ring']);
    assert.deepEqual(workQuestion(work('direct_question')), { taskId: 'd', quizId: 'q1' });
    assert.equal(workQuestion(work('model_access')), null, 'an access wait is not an answer wait');
});

test('a known outcome beside Finalizing… is a dot in its tone: Done with warnings succeeds, Stopped is neutral', () => {
    const said = (outcome) => {
        const view = workRowView({ ...work('finalizing_failed'), outcome });
        return [view.word, view.secondary, view.tone, view.outcome, markView(view)];
    };
    assert.deepEqual(said('completed'), ['Done', 'Finalizing…', 'done', 'done', { shape: 'dot', hue: 'ok' }]);
    assert.deepEqual(said('rejected_duplicate'), ['Done with warnings', 'Finalizing…', 'done', 'warn', { shape: 'dot', hue: 'ok' }]);
    assert.deepEqual(said('cancelled'), ['Stopped', 'Finalizing…', 'stopped', 'cancelled', { shape: 'dot', hue: 'neutral' }]);
    assert.deepEqual(said('failed'), ['Failed', 'Finalizing…', 'failed', 'error', { shape: 'dot', hue: 'failed' }]);
    assert.deepEqual(OUTCOME_TONES, { done: 'done', warn: 'done', cancelled: 'stopped', error: 'failed', timeout: 'failed',
        lifecycle_error: 'failed' });
});

test('whatever waits for the owner takes a row\'s mark place beside any word; the word keeps its own ink', () => {
    const question = [{ kind: 'question', task_id: 'w', quiz_id: 'q' }];
    const asking = workRowView({ ...work('working_call'), turn: question });
    assert.deepEqual([asking.word, asking.tone, asking.mark, asking.owner], ['Sending', 'live', 'moving', true]);
    assert.deepEqual(markView(asking), { shape: 'dot', hue: 'owner' });
    // The owner's own Stop press says itself and keeps the owner's dot; an outcome yields to it too.
    const stopping = workRowView({ ...work('working_call'), turn: question }, { stopping: 'finalize' });
    assert.deepEqual([stopping.word, stopping.mark, markView(stopping)], ['Finalizing…', 'moving', { shape: 'dot', hue: 'owner' }]);
    assert.deepEqual(markView(workRowView({ ...work('finalizing_failed'), turn: [{ kind: 'action' }] })),
        { shape: 'dot', hue: 'owner' });
    // Nothing for the owner: the work's shape in the word's ink.
    assert.deepEqual(markView(workRowView(work('working_call'))), { shape: 'moving', hue: 'ink' });
    assert.deepEqual(markView(workRowView(work('paused_by_owner'))), { shape: 'pause', hue: 'ink' });
});

test('the test builders name the marks the server names', () => {
    const unconfirmedTail = (item) => item.tails.some((tail) => tail.state === 'unknown');
    for (const { works, rooms } of Object.values(CASES)) {
        // A work's own mark follows its live episode (a run nobody can confirm is its line, never its mark); once the
        // episode ended, whatever still runs beside it.
        for (const item of works) {
            assert.equal(serverMark(item.activity, item.confirmed, { live: !item.episode.live && item.live,
                unknown: unconfirmedTail(item) && !item.live && !item.episode.live }), item.mark, item.work_id);
        }
        // A room moves while anything in it runs.
        for (const served of Object.values(rooms || {})) {
            const members = works.filter((item) => served.works.includes(item.work_id));
            assert.equal(serverMark(served.activity, served.confirmed, { live: members.some((item) => item.live),
                unknown: members.some(unconfirmedTail) }), served.mark, served.activity);
        }
    }
    assert.equal(serverMark('paused', true, { live: true }), 'moving');
    assert.equal(serverMark('paused', true), 'pause');
    for (const activity of ['working', 'queued', 'waiting', 'paused', 'unknown']) {
        assert.equal(room(activity).mark, serverMark(activity, activity !== 'unknown'), activity);
    }
    assert.equal(builtWork({ activity_id: 'x' }, { activity: 'paused' }).mark, 'pause');
    assert.equal(builtWork({ activity_id: 'x' }, { activity: 'queued', confirmed: false }).mark, 'ring');
    assert.equal(builtWork({ activity_id: 'x' }, { confirmed: false }).mark, 'moving', 'work observed running moves');
});

test('the focus line is the work\'s own update_focus text under its state, and none without one', () => {
    const focused = { ...work('working_call'), focus: { text: '  Release 7.7 · code merged · waiting for 7 signed files ', at: 1 } };
    assert.equal(workRowView(focused).focus, 'Release 7.7 · code merged · waiting for 7 signed files');
    assert.equal(workRowView(work('working_call')).focus, '');
    assert.equal(workRowView({ ...work('working_call'), focus: { text: '   ', at: 1 } }).focus, '');
});

test('controls group by action with exact targets; Resume is a button, a pending Stop leaves its escalation', () => {
    const running = workControls(work('working_call'));
    assert.deepEqual(workMenuActions(running), [ACTION_FINALIZE, ACTION_HURRY, ACTION_PAUSE, ACTION_STOP_NOW]);
    assert.deepEqual(workMenuActions(running, { cancelPending: true }), [ACTION_STOP_NOW]);
    const paused = workControls(work('paused_by_owner'));
    assert.equal(paused.resume, 'o');
    assert.deepEqual(workMenuActions(paused), [ACTION_STOP_NOW], 'Resume stands as its own button');
    assert.deepEqual(workMenuActions(paused, { resume: true }), [ACTION_RESUME, ACTION_STOP_NOW]);
    assert.deepEqual(workMenuActions(workControls(work('direct_question'))), [ACTION_PAUSE, ACTION_STOP_NOW],
        'a direct turn waiting for its answer can be paused and stopped, as its doors take it');
    assert.deepEqual(workControls({ controls: [{ action: 'teleport', target: 'x' }, { action: 'stop', target: '' }] }),
        { stop: [], stopEarlier: [], pause: '', resume: '', hurry: '', wrapUp: '' }, 'unknown or targetless actions are not offered');
    assert.deepEqual([...resumeTargets(body('paused_by_owner'))], ['o']);
    assert.deepEqual([...resumeTargets(body('held_continue'))], [], 'a held Continue is never resumed by the owner');
});

test('a card menu reads its work, else its own task controls without Resume', () => {
    const state = stateOf('paused_by_owner');
    assert.deepEqual(cardMenuActions(state, { groupId: 'o' }), [ACTION_STOP_NOW]);
    assert.equal(cardControls(state, 'o').resume, true);
    assert.deepEqual(cardMenuActions(state, { groupId: 'loose' }, { phase: 'budget_paused' }), [ACTION_STOP_NOW]);
    assert.deepEqual(cardMenuActions(state, { groupId: 'loose' }, { phase: 'working' }),
        [ACTION_FINALIZE, ACTION_HURRY, ACTION_PAUSE, ACTION_STOP_NOW]);
    assert.equal(cardControls(state, 'loose').resume, false);
});

test('a released answered tree selects helpers individually, never through its work Resume', () => {
    const held = { ...work('helper_after_parent'), controls: [
        { action: 'stop', target: 'p', scope: 'cascade' },
        { action: 'resume', target: 'child-a', scope: 'single' },
        { action: 'resume', target: 'child-b', scope: 'single' },
    ] };
    assert.equal(workControls(held).resume, '', 'a work press must not choose the first helper');
    assert.deepEqual([...resumeTargets({ works: [held] })], ['child-a', 'child-b']);
});

test('a nonexecuting pause offers Resume and says the unresolved model request beside Paused', () => {
    const paused = work('paused_model_handoff');
    assert.equal(workRowView(paused).word, 'Paused');
    assert.equal(workControls(paused).resume, 'mh');
    assert.equal(paused.live, false);
    assert.match(workLine(paused), /a model request has no recorded result/);
});

test('Activity reads each member from the server, so only an executing helper says Pausing', () => {
    const data = body('pausing_with_queued_helper');
    assert.equal(workRowView(memberState(data, 'pc-new')).word, 'Paused');
    assert.equal(workRowView(memberState(data, 'pc-run')).word, 'Pausing…');
    assert.equal(memberState(data, 'absent'), null);
    const sleeping = memberState(body('sleeping'), work('sleeping').episode.task_id);
    assert.equal(workRowView(sleeping).word, 'Waiting');
    assert.equal(workLine(sleeping), 'Sleeping until it is woken');
});

test('a parent that finished while its helper works keeps the Stop its work offers; an ended work offering nothing gives none', () => {
    const helper = stateOf('helper_after_parent');
    assert.equal(work('helper_after_parent').episode.live, false, 'the server says the parent\'s episode ended');
    const card = cardControls(helper, 'p');
    assert.deepEqual([card.menu, card.resume, card.work?.work_id], [true, false, 'p']);
    assert.deepEqual(cardMenuActions(helper, { groupId: 'p' }), [ACTION_PAUSE, ACTION_STOP_NOW]);
    // An ended work that offers nothing (the server may also offer its open runs' Stop) gives none:
    // the card keeps its own controls.
    const ended = stateOf('ended_with_runs', 42, { works: [{ ...work('ended_with_runs'), controls: [] }] });
    const runs = cardControls(ended, 'r');
    assert.deepEqual([runs.menu, runs.resume, runs.work], [false, false, null]);
});

test('the grouped Stop reaches every target in order and says one combined result', async () => {
    const asked = [], toasts = [];
    const ok = await requestStopAll(['c', 'p', 'c', ''], ACTION_STOP_NOW, {
        stopOne: async (target, action) => { asked.push([target, action]); },
        toast: (...args) => toasts.push(args) });
    assert.equal(ok, true);
    assert.deepEqual(asked, [['c', ACTION_STOP_NOW], ['p', ACTION_STOP_NOW]]);
    assert.deepEqual(toasts, []);
    const failing = await requestStopAll(['c', 'p'], ACTION_STOP_NOW, {
        stopOne: async (target) => { if (target === 'p') throw Object.assign(new Error('busy'), { status: 409 }); },
        toast: (...args) => toasts.push(args) });
    assert.equal(failing, false);
    assert.deepEqual(toasts, [['Stop failed for 1 of 2 parts of this work: busy', 'error']]);
    toasts.length = 0;
    // A target that already ended is settled, not a failure.
    assert.equal(await requestStopAll(['gone'], ACTION_STOP_NOW, {
        stopOne: async () => { throw Object.assign(new Error('missing'), { status: 404 }); },
        toast: (...args) => toasts.push(args) }), true);
    assert.deepEqual(toasts, []);
});

test('an incomplete body keeps what it omitted, unconfirmed and without controls; only a complete one clears', () => {
    const seen = stateOf('held_continue');
    const partial = reconcileWorkState(seen, { supervisor_ready: true, works: [], rooms: {}, works_complete: false }, 42);
    assert.equal(partial.complete, false);
    assert.deepEqual(partial.works.map((item) => [item.work_id, item.activity, item.confirmed, item.controls.length, item.mark]),
        [['p', 'unknown', false, 0, 'ring']]);
    assert.deepEqual(['word', 'tone', 'mark'].map((key) => workRowView(partial.works[0])[key]),
        ['Activity unconfirmed', 'unknown', 'ring'], 'what an incomplete body omitted is a ring, never moving work');
    assert.deepEqual([partial.room.activity, partial.room.mark, roomStatusView(partial).text], ['unknown', 'ring', 'Activity unconfirmed']);
    // What still ran beside an omitted work is no longer confirmed either: its line goes with its controls.
    const runs = stateOf('ended_with_runs');
    assert.equal(workRowView(runs.works[0]).secondary, '2 external runs still working');
    const omitted = reconcileWorkState(runs, { supervisor_ready: true, works: [], rooms: {}, works_complete: false }, 42);
    assert.deepEqual([omitted.works[0].tails, omitted.works[0].live, workRowView(omitted.works[0]).secondary], [[], false, '']);
    // Not ready, or read over a closed socket: no absence is proved either.
    assert.equal(reconcileWorkState(seen, { ...body('held_continue'), works: [], rooms: {}, supervisor_ready: false }, 42).works.length, 1);
    assert.equal(reconcileWorkState(seen, { ...body('held_continue'), works: [], rooms: {} }, 42, false).works.length, 1);
    const cleared = reconcileWorkState(partial, { supervisor_ready: true, works: [], rooms: {}, works_complete: true }, 42);
    assert.deepEqual([cleared.works.length, cleared.room.activity, roomStatusView(cleared).text], [0, 'idle', 'Online']);
    // A failed read or a body without the projection changes nothing.
    assert.equal(reconcileWorkState(seen, null, 42), seen);
    assert.equal(reconcileWorkState(seen, { active_chat_activities: [] }, 42), seen);
    // Nothing seen before and nothing listed: the room is quiet, never invented work.
    assert.equal(roomStatusView(reconcileWorkState(emptyWorkState(), { works: [], works_complete: false }, 42)).text, 'Online');
});

test('a room reads only its own works; every room feeds the card index and the member ids', () => {
    const merged = { supervisor_ready: true, works_complete: true,
        works: [...CASES.held_continue.works, ...CASES.direct_question.works],
        rooms: { ...CASES.held_continue.rooms, ...CASES.direct_question.rooms },
        active_chat_activities: [{ activity_id: 'c', task_attempt: 2 }] };
    const main = reconcileWorkState(emptyWorkState(), merged, 1);
    assert.deepEqual(main.works.map((item) => item.work_id), ['d']);
    assert.equal(main.room.activity, 'waiting');
    // A Main portal's work runs in its Project: its card still finds that work.
    assert.equal(main.episodes.get('c').work_id, 'p');
    assert.deepEqual([...main.members].sort(), ['c', 'd', 'h0', 'h1', 'h2']);
    assert.equal(main.attempts.get('c'), 2);
    assert.equal(reconcileWorkState(emptyWorkState(), merged, 42).works[0].work_id, 'p');
});

test('a work card says its work\'s word alone, its line beside it; plain running, finalizing and unconfirmed work keep their own', () => {
    const view = (name, index = 0) => cardWorkView(work(name, index));
    const said = (name, index = 0) => [view(name, index).text, view(name, index).secondary];
    assert.deepEqual([...said('paused_by_owner'), view('paused_by_owner').classes], ['Paused', `You paused it · ${PRESSED}`, 'paused']);
    assert.deepEqual(said('paused_by_mind'), ['Paused', 'Ouroboros paused it: pause before the update you asked for']);
    assert.deepEqual(said('paused_unattributed'), ['Paused', '']);
    assert.deepEqual([...said('spending_limit'), view('spending_limit').classes], ['Paused', 'Spending limit reached', 'warn']);
    assert.deepEqual(said('sleeping'), ['Waiting', 'Sleeping until it is woken']);
    assert.deepEqual(said('model_access'), ['Waiting', 'The model needs you: sign in or restore access']);
    assert.deepEqual([...said('queued_and_pausing', 0), view('queued_and_pausing', 0).tone],
        ['Pausing…', `You paused it · ${PRESSED}`, 'neutral']);
    assert.deepEqual(said('paused_review_finishing'), ['Paused', `You paused it · ${PRESSED} · Review work finishing`]);
    assert.equal(view('working_call'), null, 'its own model-call words already say Sending');
    assert.equal(view('finalizing_failed'), null, 'its outcome beside Finalizing… is its own');
    assert.equal(view('unknown'), null);
    assert.equal(view('unconfirmed_run'), null, 'unconfirmed work keeps the card\'s own words');
    assert.equal(view('ended_with_runs'), null, 'an ended episode keeps its own outcome');
    // Plain running work says its own word too once something runs beside it: the line names what.
    const helper = { kind: 'helper', state: 'live', code: 'helpers_running', text: '1 helper still working', count: 1, ids: ['h'] };
    const busy = cardWorkView({ ...work('working_call'), tails: [helper] });
    assert.deepEqual([busy.text, busy.secondary, busy.tone, busy.waiting], ['Sending', '1 helper still working', 'live', false]);
    // Its exact model call still explains that word in the card's details, never another word.
    const sending = { groupId: 'w', censusPhase: 'working', modelActivity: { phase: 'awaiting_response' }, workView: busy,
        workLook: { mark: 'moving', owner: false } };
    assert.equal(desiredLiveCardPhase(sending).detail, 'Sending request / waiting for response');
    assert.equal(desiredLiveCardPhase({ ...sending, modelActivity: { phase: 'queued' } }).detail, undefined);
    // The chip presenter takes the view, but a disconnected card still says Activity unconfirmed.
    const workView = view('paused_by_owner');
    assert.deepEqual(['text', 'secondary', 'className'].map((key) => taskActivityPresentation({ phase: 'budget_paused', workView })[key]),
        ['Paused', `You paused it · ${PRESSED}`, 'chat-live-phase paused']);
    assert.equal(taskActivityPresentation({ phase: 'unknown', workView }).text, 'Activity unconfirmed');
    const records = new Map([['o', { groupId: 'o' }], ['child', { groupId: 'o', isSubagent: true }]]);
    assert.equal(applyCardWorkViews(records, stateOf('paused_by_owner')), true);
    assert.equal(records.get('o').workView.text, 'Paused');
    assert.deepEqual(records.get('o').workLook, { mark: 'pause', owner: false }, 'the work\'s own shape, nobody\'s move');
    assert.equal(records.get('child').workView, undefined, 'a helper card keeps its own presenter');
    assert.equal(applyCardWorkViews(records, stateOf('paused_by_owner')), false, 'an equal view is no change');
});

// The work's look reaches its card apart from the word: a running parent whose helper waits for the owner keeps
// its own live word with the owner's amber dot (the server takes the turn from every member), and the look survives
// the card's outcome, its record adapter and its painter (docs/DESIGN.md "Status marks").
test('a work card keeps its work\'s look beside its own words: the owner\'s dot reaches a running parent', () => {
    const parent = structuredClone(CASES.working_call.works[0]);
    assert.equal(cardWorkView(parent), null, 'plain running work keeps the card\'s own words');
    parent.turn = [{ kind: 'question', task_id: 'helper', quiz_id: 'q1' }];
    const body = { supervisor_ready: true, works_complete: true, works: [parent], rooms: {} };
    const state = reconcileWorkState(emptyWorkState(), body, 42);
    const record = { groupId: parent.episode.task_id };
    assert.equal(applyCardWorkViews(new Map([['w', record]]), state), true);
    assert.deepEqual([record.workView, record.workLook], [null, { mark: 'moving', owner: true }]);
    const view = desiredLiveCardPhase(record);
    assert.deepEqual([view.text, view.tone, view.owner, view.mark], ['Working', 'live', true, 'moving']);
    // The outcome owns the chip while its work finalizes; the owner's move stays its dot.
    const late = desiredLiveCardPhase({ ...record, finalizingHold: true, observedOutcome: 'done' });
    assert.deepEqual([late.text, late.tone, late.owner, late.secondary], ['Done', 'ok', true, 'Finalizing…']);
    // The work's own word that is the owner's move takes the amber body too.
    const limit = structuredClone(CASES.spending_limit.works[0]);
    const waits = { ...record, workView: cardWorkView(limit), workLook: { mark: limit.mark, owner: limit.turn.length > 0 } };
    assert.deepEqual(['text', 'secondary', 'tone', 'owner'].map((key) => desiredLiveCardPhase(waits)[key]),
        ['Paused', 'Spending limit reached', 'owner', true]);
    // A turn the body lists is a fact even when another member could not be confirmed: the dot stays.
    const partial = { ...parent, confirmed: false };
    const read = reconcileWorkState(emptyWorkState(), { ...body, works_complete: false, works: [partial] }, 42);
    applyCardWorkViews(new Map([['w', record]]), read);
    assert.deepEqual([record.workView, record.workLook], [null, { mark: 'moving', owner: true }]);
    assert.deepEqual([desiredLiveCardPhase(record).text, desiredLiveCardPhase(record).owner], ['Working', true]);
    // A work an incomplete body omitted proves nothing: no look, so the card's own words and shape stand.
    const kept = reconcileWorkState(state, { supervisor_ready: true, works_complete: false, works: [], rooms: {} }, 42);
    applyCardWorkViews(new Map([['w', record]]), kept);
    assert.deepEqual([record.workView, record.workLook], [null, null]);
    assert.equal(desiredLiveCardPhase(record).owner, false);
});

test('a census pause says its word alone; only the reviews a settled Pause lets finish stand beside it', () => {
    for (const cause of ['budget', 'restart', 'authority', 'owner', 'future-cause', '']) {
        assert.equal(pausePhaseLabel('budget_paused', cause), 'Paused', cause);
        assert.equal(pausePhaseLabel('budget_pausing', cause), 'Pausing…', cause);
    }
    assert.equal(pausePhaseLabel('budget_paused', 'sleep'), 'Waiting', 'a cold sleep waits: it goes on by itself');
    const census = (row) => censusTaskPhase({ activity_id: 't', ...row });
    const finishing = census({ phase: 'budget_paused', pause_cause: 'owner', finishing_reviews: true });
    assert.deepEqual([finishing.text, finishing.secondary], ['Paused', 'Review work finishing']);
    assert.equal(census({ phase: 'budget_paused', pause_cause: 'owner' }).secondary, undefined);
    assert.equal(census({ phase: 'budget_pausing', pause_cause: 'owner', finishing_reviews: true }).secondary, undefined,
        'only a settled Pause names the reviews it lets finish');
});

// Every work the server can project (each case of the shared fixture): the chip and the row say the work's word alone,
// never joined with a reason, and the line beside it carries the episode's own sentence and every tail's.
test('on every surface the word is the work\'s word alone and the line carries its reason and what still runs', () => {
    for (const [name, { works, rooms }] of Object.entries(CASES)) {
        for (const item of works) {
            const word = activityWord(item);
            const line = workLine(item);
            const reason = reasonText(item.because[0]);
            if (reason && reason !== word) assert.ok(line.startsWith(reason), `${name}: the episode's sentence leads the line`);
            // Helpers and external runs still working side by side are one sentence; every other tail says its own.
            const both = ['helpers_running', 'delegated_running'].every((code) => item.tails.some((tail) => tail.code === code));
            for (const tail of item.tails) {
                if (!(both && ['helpers_running', 'delegated_running'].includes(tail.code))) {
                    assert.ok(line.includes(reasonText(tail)), `${name}: ${tail.code} on the line`);
                }
            }
            if (both) assert.match(line, / and \d+ external runs? still working/, `${name}: said once`);
            const card = cardWorkView(item);
            if (card) assert.deepEqual([card.text, card.secondary], [word, line], `${name}: the card's chip and line`);
            const row = workRowView(item);
            if (!row.outcome) assert.deepEqual([row.word, row.secondary], [word, line], `${name}: the Shade row`);
            for (const said of [card?.text, row.word]) assert.ok(!said || !said.includes(' · '), `${name}: a word alone`);
        }
        for (const served of Object.values(rooms || {})) {
            // The room's line is that of the first work saying the room's own word, never another work's reason.
            const word = activityWord(served);
            const speaker = served.works.map((id) => works.find((item) => item.work_id === id))
                .find((item) => item && activityWord(item) === word);
            const sidebar = summarizeRoom(served, works);
            assert.equal(sidebar.label, [word, speaker ? workLine(speaker, word) : '', sidebar.note]
                .filter(Boolean).join(' · '), `${name}: the sidebar's name`);
        }
    }
});

// The owner's screenshot of 20.10: a Pause the owner pressed has settled while the review it launched still finishes.
test('a paused work whose review still finishes says Paused with its bars; the room moves while the review runs', () => {
    const state = stateOf('paused_review_finishing');
    const records = new Map([['rv', { groupId: 'rv' }]]);
    applyCardWorkViews(records, state);
    const card = desiredLiveCardPhase(records.get('rv'));
    assert.deepEqual([card.text, card.secondary, card.mark, card.tone, card.owner],
        ['Paused', `You paused it · ${PRESSED} · Review work finishing`, 'pause', 'neutral', false]);
    assert.equal(cardControls(state, 'rv').resume, true, 'its Resume stays beside it');
    const row = workRowView(state.works[0]);
    assert.deepEqual([row.word, row.secondary, row.mark], ['Paused', `You paused it · ${PRESSED} · Review work finishing`, 'pause']);
    const bar = roomStatusView(state);
    assert.deepEqual([bar.text, bar.mark], ['Paused', 'moving']);
    assert.equal(summarizeRoom(state.room, state.works).label, `Paused · You paused it · ${PRESSED} · Review work finishing`);
    // The review ends: the same work reads its plain Pause, the room stands still on its bars.
    const settled = structuredClone(CASES.paused_by_owner);
    const ended = reconcileWorkState(emptyWorkState(), { supervisor_ready: true, ...settled }, 42);
    assert.deepEqual([workRowView(ended.works[0]).secondary, roomStatusView(ended).mark], [`You paused it · ${PRESSED}`, 'pause']);
});

test('a work whose only remainder is a run nobody can confirm reads unconfirmed with its ring, and keeps its Stop', () => {
    const state = stateOf('unconfirmed_run');
    const row = workRowView(state.works[0]);
    assert.deepEqual([row.word, row.secondary, row.tone, row.mark], ['Activity unconfirmed', '1 external run may still be running', 'unknown', 'ring']);
    assert.deepEqual(workControls(state.works[0]).stop, ['x']);
    assert.equal(summarizeRoom(state.room, state.works).label, 'Activity unconfirmed · 1 external run may still be running');
});

test('a room never lends a newer work\'s Pause reason to the Working of the work that runs', () => {
    const state = stateOf('two_works_one_pausing');
    assert.deepEqual(state.room.works, ['wa', 'wb'], 'the newer, pausing work is listed first');
    assert.equal(summarizeRoom(state.room, state.works).label, 'Working');
    assert.equal(workLine(state.works.find((item) => item.work_id === 'wa')), `You paused it · ${PRESSED}`,
        'its own row still says its reason');
});

test('helpers and external runs still working side by side are said once; either alone keeps its sentence', () => {
    const tails = (helpers, runs) => [
        ...(helpers ? [{ code: 'helpers_running', kind: 'helper', state: 'live', count: helpers, text: 'h' }] : []),
        ...(runs ? [{ code: 'delegated_running', kind: 'run', state: 'live', count: runs, text: 'r' }] : []),
        { code: 'external_runs_unconfirmed', kind: 'run', state: 'unknown', count: 1, text: 'u' },
    ];
    const line = (helpers, runs) => workLine({ activity: 'working', transition: 'none', because: [], tails: tails(helpers, runs) });
    assert.equal(line(3, 2), '3 helpers and 2 external runs still working · 1 external run may still be running');
    assert.equal(line(1, 1), '1 helper and 1 external run still working · 1 external run may still be running');
    assert.equal(line(2, 0), '2 helpers still working · 1 external run may still be running');
    assert.equal(line(0, 1), '1 external run still working · 1 external run may still be running');
});

test('the line\'s clock: the time today, the date as well on another day; seconds or an ISO instant; nothing without one', () => {
    const time = (date) => date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const now = new Date();
    assert.equal(clock(now.getTime() / 1000), time(now));
    assert.equal(clock(now.toISOString()), time(now));
    const earlier = new Date(now.getTime() - 3 * 86400000);
    assert.notEqual(clock(earlier.toISOString()), time(earlier), 'another day is never read as today');
    assert.ok(clock(earlier.toISOString()).includes(time(earlier)));
    assert.equal(clock(earlier.getTime() / 1000), clock(earlier.toISOString()));
    for (const missing of ['', null, undefined, 0, 'not a time']) assert.equal(clock(missing), '', String(missing));
    // The owner's press is said with its time; no other reason gets one.
    assert.equal(workLine(work('paused_by_owner')), `You paused it · ${clock(work('paused_by_owner').because[0].since)}`);
    assert.equal(workLine(work('spending_limit')), 'Spending limit reached');
});
