import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { carryOpenQuestions, questionMark, questionPresentation, questionWaits, vouchedBy, waitFacts }
    from '../modules/question_presentation.js';

// Each fixture row is the pointer row the Python producer emits for that case
// (tests/test_project_question_pointer.py pins the emission); the browser must read the
// same status out of it.
const cases = JSON.parse(readFileSync(new URL('./fixtures/question_presentation_parity.json', import.meta.url)));
for (const row of cases) test(`question status parity: ${row.case}`, () => {
    assert.deepEqual(questionPresentation(row.row), { status: row.status });
});

test('whether a question still waits on the owner is one rule on both sides', () => {
    // The pointer rows are a Project's; the server reads the same rows by main_decisions.question_waits.
    for (const row of cases) assert.equal(questionWaits({ ...row.row, project_id: 'p' }), row.waits, row.case);
    // An undelivered answer is the owner's to remove in its Project only: Main's own question has no such row.
    const lost = cases.find((row) => row.row.question_delivery?.reason_code === 'late_answer_delivery_unknown' && row.waits).row;
    assert.equal(questionWaits(lost), false);
});

test('the head mark beside each word: amber exactly while Main\'s pill counts the question', () => {
    const hue = (row) => questionMark(row).hue;
    // Every row the Python producer emits reads one dot, in the grammar's hues: amber exactly where the pill counts it
    // (the pointer rows are a Project's), whatever its words say; the words themselves do not change.
    for (const { case: name, row, status, waits } of cases) {
        const project = { ...row, project_id: 'p' };
        assert.equal(questionMark(project).shape, 'dot', name);
        assert.equal(hue(project) === 'owner', waits, name);
        assert.equal(questionPresentation(project).status, status, name);
        assert.ok(['owner', 'ok', 'neutral', 'ink'].includes(hue(project)), name);
    }
    // Whichever field carries the state, every unanswered question is amber: its task waits, went on or ended.
    for (const row of [{ state: 'open', wait_for_answer: true }, { state: 'open', owner_wait_state: 'waiting' },
        { state: 'open' }, { state: 'open', wait_for_answer: true, wait_ended_at: 'x' },
        { quiz_state: 'open', owner_wait_state: 'resumed' }, { state: 'expired_terminal', wait_for_answer: true }]) {
        assert.equal(hue(row), 'owner', JSON.stringify(row));
    }
    assert.deepEqual(['Unanswered', 'Unanswered · task continued', 'Unanswered · task finished'],
        [{ state: 'open' }, { state: 'open', owner_wait_state: 'resumed' }, { state: 'expired_terminal' }]
            .map((row) => questionPresentation(row).status), 'the words stay as they are');
    // What the pill does not count is not amber.
    for (const row of [{}, { state: 'mystery' }]) assert.equal(hue(row), 'neutral', JSON.stringify(row));
    assert.equal(hue({ quiz_state: 'answered' }), 'ok');
    // A replaced or closed question wears the head's disabled ink (the swap DESIGN "Quiz card" forbids).
    for (const state of ['superseded', 'closed']) assert.equal(hue({ quiz_state: state }), 'ink', state);
    // An owed delivery keeps its lifecycle's dot beside its own word.
    assert.equal(hue({ quiz_state: 'answered', question_delivery: { state: 'owed' } }), 'ok');
    assert.equal(hue({ quiz_state: 'closed', question_delivery: { state: 'owed' } }), 'ink');
    // A Project's late answer whose delivery became unknown waits for the owner to stop counting it (Remove from
    // pending); Main's own the pill does not count, and its card offers nothing to press: not amber.
    const unknown = { quiz_state: 'answered', question_delivery: { state: 'owed', reason_code: 'late_answer_delivery_unknown' } };
    assert.equal(hue({ ...unknown, project_id: 'p' }), 'owner');
    assert.equal(hue(unknown), 'ok');
});

test('waiting needs positive evidence and a closed bound ends it', () => {
    assert.deepEqual(waitFacts({ wait_for_answer: true }), { waiting: true, resumed: false });
    assert.deepEqual(waitFacts({ wait_for_answer: true, owner_wait_state: 'resumed' }), { waiting: false, resumed: true });
    assert.deepEqual(waitFacts({ wait_for_answer: true, wait_ended_at: 'x' }), { waiting: false, resumed: true });
    assert.deepEqual(waitFacts({ owner_wait_state: 'waiting' }), { waiting: true, resumed: false });
    assert.deepEqual(waitFacts({}), { waiting: false, resumed: false });
});

test('a question whose task is a paused member of its work says that its answer waits for Resume', () => {
    const waiting = { state: 'open', owner_wait_state: 'waiting' };
    const saved = { quiz_state: 'answered', question_delivery: { state: 'owed', reason_code: 'awaiting_recipient_read' } };
    // Facts, not form (DESIGN principle 7): the status still says whether the answer is awaited or saved and names
    // Resume; that the work is paused is its own card's line. How the words are joined is review's.
    const [awaited, kept] = [/\bwaiting\b/i, /\bsaved\b/i];
    for (const [row, state, other] of [[waiting, awaited, kept], [saved, kept, awaited]]) {
        const plain = questionPresentation(row).status;
        const paused = questionPresentation(row, { paused: true }).status;
        assert.match(plain, state, plain);
        assert.match(paused, state, paused);
        assert.doesNotMatch(paused, other, paused);
        assert.match(paused, /\bResume\b/, paused);
        assert.doesNotMatch(paused, /\bpaused\b/i, paused);
        // After its Resume (or never paused) the plain words return; a queued or held recipient keeps them too.
        assert.doesNotMatch(plain, /\bResume\b/);
        assert.equal(questionPresentation(row, { paused: false }).status, plain);
    }
    assert.equal(questionPresentation(waiting).status, 'Waiting for your answer');
    // The fact changes no other lifecycle word and never the head's mark.
    for (const row of [{ state: 'open' }, { quiz_state: 'answered' }, { quiz_state: 'closed' }, { state: 'expired_terminal' },
        { quiz_state: 'answered', question_delivery: { state: 'owed', reason_code: 'late_answer_delivery_unknown' } }]) {
        assert.equal(questionPresentation(row, { paused: true }).status, questionPresentation(row).status, JSON.stringify(row));
    }
    assert.deepEqual(questionMark(waiting), { shape: 'dot', hue: 'owner' });
});

test('a read vouches for every question when complete, for what its gaps do not name when not, for nothing without gaps', () => {
    assert.equal(vouchedBy({ current_questions_complete: true })('t', 'q'), true);
    const named = vouchedBy({ current_questions_complete: false, current_questions_gaps: [{ task_id: 't', quiz_id: 'q' }] });
    assert.deepEqual([named('t', 'q'), named('t', 'other'), named('u', 'q')], [false, true, true]);
    const task = vouchedBy({ current_questions_gaps: [{ task_id: 't' }] });
    assert.deepEqual([task('t', 'q'), task('t', 'other'), task('u', 'q')], [false, false, true]);
    for (const data of [null, {}, { current_questions_gaps: [] }, { current_questions_gaps: [{ reason: 'index' }] }]) {
        assert.equal(vouchedBy(data)('u', 'q'), false, JSON.stringify(data));
    }
});

test('a read that does not vouch against an open question keeps it in its room; one that does removes it', () => {
    const item = { kind: 'open_question', task_id: 't', quiz_id: 'q' };
    const idle = (turn) => ({ activity: 'idle', transition: 'none', call: '', turn, amber: turn.length > 0, mark: 'none',
        confirmed: true, works: [] });
    const prior = { rooms: { 42: idle([item]) } };
    const read = (extra) => ({ supervisor_ready: true, works: [], works_complete: true, rooms: {}, ...extra });
    const gap = read({ current_questions_complete: false, current_questions_gaps: [{ task_id: 't', quiz_id: 'q' }] });
    assert.deepEqual(carryOpenQuestions(prior, gap).rooms['42'], idle([item]), 'a gap over it: still the owner\'s turn');
    // A gap that names another question, or a complete read, vouches against it: it is gone.
    const other = read({ current_questions_gaps: [{ task_id: 'u' }] });
    assert.equal(carryOpenQuestions(prior, other), other);
    const complete = read({ current_questions_complete: true });
    assert.equal(carryOpenQuestions(prior, complete), complete);
    // Never invented: nothing earlier, or a body without the projection, is returned as it is.
    assert.equal(carryOpenQuestions(null, gap), gap);
    assert.equal(carryOpenQuestions(prior, null), null);
    const census = { current_questions_gaps: [{ task_id: 't' }] };
    assert.equal(carryOpenQuestions(prior, census), census);
    // A room the body lists keeps its own word and gains the item once; a work of the room that held it holds it again.
    const held = { ...item, work_id: 'w' };
    const working = { activity: 'working', transition: 'none', call: '', turn: [], amber: false, mark: 'moving',
        confirmed: true, works: ['w'] };
    const listed = read({ current_questions_gaps: [{ task_id: 't' }], works: [{ work_id: 'w', room: 42, turn: [] }],
        rooms: { 42: working } });
    const kept = carryOpenQuestions({ rooms: { 42: { ...working, turn: [held], amber: true } } }, listed);
    assert.deepEqual([kept.rooms['42'].activity, kept.rooms['42'].amber, kept.rooms['42'].turn], ['working', true, [held]]);
    assert.deepEqual(kept.works[0].turn, [item]);
    assert.equal(carryOpenQuestions({ rooms: { 42: kept.rooms['42'] } }, { ...listed, rooms: kept.rooms }).rooms['42'].turn.length, 1,
        'an item the room already lists is said once');
    // Its work no longer listed in the room: the item stays the room's own.
    const gone = carryOpenQuestions({ rooms: { 42: { ...working, turn: [held], amber: true } } }, gap);
    assert.deepEqual(gone.rooms['42'].turn, [item]);
});

test('a partial read keeps a known question whatever its item was; a bare wait is not carried; a vouching read removes it', () => {
    const idle = (turn) => ({ activity: 'idle', transition: 'none', call: '', turn, amber: turn.length > 0, mark: 'none',
        confirmed: true, works: [] });
    const gap = { supervisor_ready: true, works: [], works_complete: true, rooms: {},
        current_questions_gaps: [{ task_id: 't' }] };
    const open = { kind: 'open_question', task_id: 't', quiz_id: 'q' };
    // The work stood on its exact question and was stopped unanswered: the next body lists no work and cannot read
    // the question. It stays the room's turn, as the open question it still is, never as a wait nobody holds.
    for (const known of [open, { kind: 'question', task_id: 't', quiz_id: 'q', work_id: 'w' }]) {
        const carried = carryOpenQuestions({ rooms: { 42: idle([known]) } }, gap);
        assert.deepEqual(carried.rooms['42'].turn, [open], known.kind);
        assert.equal(carried.rooms['42'].amber, true, known.kind);
        // A read that vouches against it removes it.
        const complete = { ...gap, current_questions_complete: true };
        assert.equal(carryOpenQuestions({ rooms: { 42: idle([known]) } }, complete), complete, known.kind);
    }
    // A wait that never named its quiz has no proved address: it is not carried.
    assert.equal(carryOpenQuestions({ rooms: { 42: idle([{ kind: 'question', task_id: 't' }]) } }, gap), gap);
    // A bare wait the body still lists never stands for the exact question it does not name.
    const bare = { ...gap, rooms: { 42: idle([{ kind: 'question', task_id: 't' }]) } };
    assert.deepEqual(carryOpenQuestions({ rooms: { 42: idle([open]) } }, bare).rooms['42'].turn,
        [{ kind: 'question', task_id: 't' }, open]);
});
