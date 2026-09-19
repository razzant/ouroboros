import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { questionPresentation, waitFacts } from '../modules/question_presentation.js';

// Each fixture row is the pointer row the Python producer emits for that case
// (tests/test_project_question_pointer.py pins the emission); the browser must read the
// same status out of it.
const cases = JSON.parse(readFileSync(new URL('./fixtures/question_presentation_parity.json', import.meta.url)));
for (const row of cases) test(`question status parity: ${row.case}`, () => {
    assert.deepEqual(questionPresentation(row.row), { status: row.status });
});

test('waiting needs positive evidence and a closed bound ends it', () => {
    assert.deepEqual(waitFacts({ wait_for_answer: true }), { waiting: true, resumed: false });
    assert.deepEqual(waitFacts({ wait_for_answer: true, owner_wait_state: 'resumed' }), { waiting: false, resumed: true });
    assert.deepEqual(waitFacts({ wait_for_answer: true, wait_ended_at: 'x' }), { waiting: false, resumed: true });
    assert.deepEqual(waitFacts({ owner_wait_state: 'waiting' }), { waiting: true, resumed: false });
    assert.deepEqual(waitFacts({}), { waiting: false, resumed: false });
});
