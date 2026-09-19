// Read-only presentation of one owner question: the quiz card header, in the Project and in its
// Main mirror alike. Task liveness and answerability are separate facts: a status
// leads with one word that answers "is there an unanswered question for me?" and
// keeps the lifecycle context after it. The Python pointer fallback
// (ouroboros/project_dialogue.py::QUESTION_STATUS) emits the same words; the shared
// fixture web/tests/fixtures/question_presentation_parity.json pins both sides on the
// rows Python actually emits.
const STATUS = {
    waiting: 'Waiting for your answer',
    open: 'Unanswered · an answer is still accepted',
    resumed: 'Unanswered · the task continued; an answer is still accepted',
    expired_terminal: 'Unanswered · the task finished; a late answer is accepted as your message',
    answered: 'You answered',
    superseded: 'Replaced by a newer question',
    unknown: 'Status unavailable',
};
// The closed lifecycle set, and the states that still take an answer (only a settled one —
// answered/superseded — turns a card into a pure record). One JS home for both lists.
export const QUIZ_LIFECYCLE = ['open', 'answered', 'expired_terminal', 'superseded'];
export const ANSWERABLE_QUIZ_STATES = ['open', 'expired_terminal'];

// Waiting needs positive evidence: the task's live wait record, or the original
// required flag before any record exists. A resumed record or a closed bound ends it.
export function waitFacts(row = {}) {
    const resumed = row.owner_wait_state === 'resumed' || Boolean(row.wait_ended_at);
    const waiting = !resumed && (row.owner_wait_state === 'waiting'
        || (!row.owner_wait_state && row.wait_for_answer === true));
    return { waiting, resumed };
}

export function questionPresentation(row = {}) {
    const state = row.quiz_state || row.state || 'unknown';
    const { waiting, resumed } = waitFacts(row);
    const key = !QUIZ_LIFECYCLE.includes(state) ? 'unknown' : state !== 'open' ? state
        : waiting ? 'waiting' : resumed ? 'resumed' : 'open';
    return { status: STATUS[key] };
}
