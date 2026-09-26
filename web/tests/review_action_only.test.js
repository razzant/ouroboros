import test from 'node:test';
import assert from 'node:assert/strict';
import { taskAcceptanceGroupFromTaskDetail } from '../modules/review_presentation.js';

test('action-only task finish remains visible without an invented stance', () => {
    const group = taskAcceptanceGroupFromTaskDetail({
        task_id: 'root',
        review_projection: { panels: [{ panel_id: 'p1', surface: 'task_acceptance', aggregate_signal: 'FAIL' }] },
        review_status: { acceptance_decision: {
            status: 'finalized_unaccepted', reason: 'author_finish',
            author_disposition: { action: 'finish', disposition: '', rationale: 'Result retained despite criticism.',
                subject_hash: 'current', reviewer_signal: 'FAIL', source: 'author' },
        } },
    });
    assert.match(group.authorDecisionText, /Author finish/);
    assert.match(group.authorDecisionText, /Result retained despite criticism/);
    assert.doesNotMatch(group.authorDecisionText, /partial|accepted|: FAIL/);
    assert.equal(group.attempts[0].verdict, 'FAIL');
});
