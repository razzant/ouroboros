// Unmodified production frames/history/detail consumed by the shipped question reader.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fixture } from './chat_decision_fixture.js';

const input = JSON.parse(readFileSync(0, 'utf8'));
if (input.reask) checkReask(input.reask);
else await checkDetail(input);

function checkReask({ first, ended, fresh, history }) {
    const markOf = card => {
        const mark = card.querySelector('.chat-quiz-mark');
        return [mark.dataset.shape, mark.dataset.hue, mark.hidden];
    };
    const assertCard = (card, waiting) => {
        assert.equal(card.dataset.state, 'open');
        assert.equal(Boolean(card.querySelector('.chat-quiz-wait')), waiting);
        assert.equal(Boolean(card.querySelector('.chat-quiz-wait-ended')), !waiting);
        assert.deepEqual(markOf(card), ['dot', waiting ? 'owner' : 'neutral', false]);
        assert.equal(card.querySelector('.chat-quiz-comment').disabled, false);
    };
    const live = fixture({ isMain: true });
    try {
        const original = live.decision.buildQuizCard(first);
        assertCard(original, true);
        assert.equal(live.decision.applyQuizStateFrame(original, ended), true);
        assertCard(original, false);
        const next = live.decision.buildQuizCard(fresh);
        assert.notEqual(next, original, 'a new request_id publishes its own waiting question');
        assertCard(next, true);
        assertCard(original, false);
        assert.equal(live.calls.length, 0);
    } finally { live.decision.destroy(); live.restore(); }
    const replay = fixture({ isMain: true });
    try {
        const questions = history.messages.filter(row => row.quiz);
        assert.equal(questions.length, 2);
        for (const row of questions) {
            assertCard(replay.decision.buildQuizCard(row), row.quiz.quiz_id === fresh.quiz_id);
        }
        assert.equal(replay.calls.length, 0, 'replay leaves both questions answerable without submitting');
    } finally { replay.decision.destroy(); replay.restore(); }
}

async function checkDetail({ detail, quizId, sourceUrl }) {
    const fx = fixture({ isMain: true, rawDetail: true, fetchDetail: async () => detail });
    try {
        const card = fx.decision.buildQuizCard({ ...detail.owner_quiz[quizId], task_id: detail.task_id });
        const input = card.querySelector('.chat-quiz-comment');
        input.value = 'Research only; no promise.';
        fx.decision.setConnected(false);
        assert.equal(input.disabled, true);
        fx.decision.setConnected(true);
        const answer = await fx.decision.readQuestion(detail.task_id, quizId, '');
        assert.ok(answer, 'Main detail proves this private question even with a Project binding');
        assert.equal(input.disabled, false);
        assert.equal(input.value, 'Research only; no promise.');
        assert.equal(card.querySelector('.chat-quiz-question').textContent, 'May I take on the requested work?');
        assert.equal(detail.owner_quiz[quizId].presence_contact.url, sourceUrl);
        assert.equal(fx.calls.length, 0, 'reading/reconnecting must not submit an answer');
    } finally { fx.decision.destroy(); fx.restore(); }
}
