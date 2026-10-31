// stdin is the unmodified production GET /api/tasks response from the Python consumer test.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fixture, turn } from './chat_decision_fixture.js';
const { detail, project, refuse, unknown } = JSON.parse(readFileSync(0, 'utf8'));
let reads = 0, current = unknown || detail;
const fx = fixture({ isMain: true, rawDetail: true, fetchDetail: async () => { reads++; return current; } });
const tid = detail.task_id;
try {
    const card = fx.decision.buildQuizCard({ ...detail.owner_quiz.q, task_id: tid });
    const input = card.querySelector('.chat-quiz-comment');
    input.value = 'Retain this draft';
    fx.decision.setConnected(false);
    assert.equal(input.disabled, true);
    fx.decision.setConnected(true);
    assert.equal(input.disabled, true);
    if (unknown) {
        fx.decision.syncCurrentQuestions([], false);
        await turn(); await turn();
        assert.equal(reads, 1, 'the unknown detail was actually fetched');
        assert.equal(input.disabled, true, 'unreadable room authority cannot enable');
        current = detail;
        fx.decision.syncCurrentQuestions([], false);
        await turn(); await turn();
        assert.equal(reads, 2, 'fresh proof was fetched in the same epoch');
        assert.equal(input.disabled, false, 'next census recovers transient unknown in the same epoch');
    }
    const answer = await fx.decision.readQuestion(tid, 'q', project);
    if (refuse) {
        assert.equal(answer, null, 'foreign or missing room proof must refuse');
        assert.equal(input.disabled, true);
    } else {
        assert.ok(answer, 'real task detail must prove the current question room');
        assert.equal(card.querySelector('.chat-quiz-comment'), input);
        assert.equal(input.value, 'Retain this draft');
        assert.equal(input.disabled, false);
        const before = reads;
        fx.decision.syncCurrentQuestions([], false);
        await turn();
        assert.equal(reads, before, 'a restored native card needs no repeated reads');
    }
    assert.equal(fx.calls.length, 0);
} finally { fx.decision.destroy(); fx.restore(); }
