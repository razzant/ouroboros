import { tr } from './i18n.js';
import { MAX_QUIZ_OPTIONS, MAX_DECISION_COMMENT } from './api_types.js';
import { bindEnterSubmit } from './ui_interactions.js';

export const WAIT_FIELDS = ['wait_for_answer', 'wait_ended_at', 'owner_wait_state', 'owner_wait_resume_reason'];

// Read-only presentation of one owner question: the quiz card header, in the Project and in its
// Main mirror alike. Task liveness and answerability are separate facts: a status
// leads with one word that answers "is there an unanswered question for me?" and
// keeps the lifecycle context after it. The Python pointer fallback
// (ouroboros/project_dialogue.py::QUESTION_STATUS) emits the same words; the shared
// fixture web/tests/fixtures/question_presentation_parity.json pins both sides on the
// rows Python actually emits.
const STATUS = {
    waiting: 'Waiting for your answer',
    open: 'Unanswered',
    resumed: 'Unanswered · task continued',
    expired_terminal: 'Unanswered · task finished',
    answered: 'You answered',
    superseded: 'Replaced by a newer question',
    closed: 'Question closed',
    unknown: 'Status unavailable',
};
// The closed lifecycle set, and the states that still take an answer (only a settled one —
// answered/superseded — turns a card into a pure record). One JS home for both lists.
export const QUIZ_LIFECYCLE = ['open', 'answered', 'expired_terminal', 'superseded', 'closed'];
export const ANSWERABLE_QUIZ_STATES = ['open', 'expired_terminal'];

// Waiting needs positive evidence: the task's live wait record, or the original
// required flag before any record exists. A resumed record or a closed bound ends it.
export function waitFacts(row = {}) {
    const resumed = row.owner_wait_state === 'resumed' || Boolean(row.wait_ended_at);
    const waiting = !resumed && (row.owner_wait_state === 'waiting'
        || (!row.owner_wait_state && row.wait_for_answer === true));
    return { waiting, resumed };
}

const deliveryOwed = row => ['answered', 'closed'].includes(row.quiz_state || row.state)
    && row.question_delivery?.state === 'owed';
// An owed delivery keeps a settled card current in Main; an unknown the owner removed keeps its
// honest status but no longer counts as pending.
export const questionDeliveryPending = row => deliveryOwed(row) && row.question_delivery?.dismissed !== true;
export const unknownDeliveryRemovable = row => questionDeliveryPending(row)
    && (row.quiz_state || row.state) === 'answered' && row.question_delivery?.reason_code === 'late_answer_delivery_unknown';

// The lifecycle code of one row: a settled state names itself; an open one says whether the task waits.
function lifecycleKey(row) {
    const state = row.quiz_state || row.state || 'unknown';
    const { waiting, resumed } = waitFacts(row);
    return !QUIZ_LIFECYCLE.includes(state) ? 'unknown' : state !== 'open' ? state
        : waiting ? 'waiting' : resumed ? 'resumed' : 'open';
}

export function questionPresentation(row = {}) {
    if (deliveryOwed(row)) {
        if ((row.quiz_state || row.state) !== 'answered') {
            return { status: tr('question.status.closure_pending', 'Question closed · update pending') };
        }
        // An accepted late answer whose dispatch witness ended with its process: no retry knows more.
        // An answer in its task's mailbox waits for that task to read it, also through a pause.
        const reason = row.question_delivery.reason_code;
        return { status: reason === 'late_answer_delivery_unknown'
            ? tr('question.status.answer_unknown', 'Answer recorded · delivery unknown')
            : reason === 'awaiting_recipient_read'
                ? tr('question.status.answer_saved', 'Answer saved · the task reads it when it continues')
                : tr('question.status.answer_pending', 'Answer recorded · delivery pending') };
    }
    const key = lifecycleKey(row);
    // The English words stay the source (and the Python twin's); the install language reads
    // them by the lifecycle code (web/modules/i18n.js).
    return { status: tr(`question.status.${key}`, STATUS[key]) };
}

// The dot beside that word (DESIGN "Quiz card"): the owner's amber only while the task positively waits for the
// answer, ok once it is answered, the head's own disabled ink for a replaced or closed question, and neutral for
// every other lifecycle (unanswered, the task continued or finished, status unavailable).
const MARK_HUES = { waiting: 'owner', answered: 'ok', superseded: 'ink', closed: 'ink' };

/** The question head's mark as a status-mark view ({ shape, hue }, web/modules/status_mark.js). */
export function questionMark(row = {}) {
    // An answer whose late delivery became unknown waits for the owner to stop counting it (Remove from pending).
    if (unknownDeliveryRemovable(row)) return { shape: 'dot', hue: 'owner' };
    return { shape: 'dot', hue: MARK_HUES[lifecycleKey(row)] || 'neutral' };
}

export function normalizeQuiz(msg) {
    const nested = msg && typeof msg.quiz === 'object' && msg.quiz ? msg.quiz : null;
    const src = nested || msg || {};
    // Strict per-card validation: ONE corrupt option refuses THIS card
    // (buildQuizCard -> null), never the whole history hydration pass.
    // Filtering instead would silently shift option_index against the
    // producer's original list — a wrong answer, not a degraded card.
    const raw = Array.isArray(src.options) ? src.options : [];
    const normalized = raw.map((option, index) => (typeof option === 'string'
        ? { label: option, ...(src.option_details?.[index] ? { detail: src.option_details[index] } : {}),
            ...(src.recommended_index === index ? { recommended: true } : {}) } : option));
    const corrupt = normalized.some(
        (option) => !option || typeof option !== 'object' || !String(option.label || '').trim());
    const optionsKnown = Array.isArray(src.options) && !corrupt && normalized.length <= MAX_QUIZ_OPTIONS;
    return {
        quizId: String(src.quiz_id || ''),
        question: String((nested ? msg.text : src.question) || ''),
        options: optionsKnown ? normalized : [],
        optionsKnown,
        stake: String(src.stake || ''),
        assumption: String(src.assumption || ''),
        closedReason: String(src.closed_reason || ''),
        materials: Array.isArray(src.materials) ? src.materials : [],
        // The wait facts the header and the signature line read (waitFacts): the
        // original required flag, the closed bound, and the task's wait record when
        // history or a detail read attached it.
        waitRow: Object.fromEntries(WAIT_FIELDS.filter((key) => Object.hasOwn(src, key)).map((key) => [key, src[key]])),
        waitForAnswer: src.wait_for_answer === true,
        state: String(src.state || 'open'),
        taskId: String(msg.task_id || ''),
        ts: msg.ts || null,
        answerFields: Object.fromEntries(['answered_index', 'comment', 'question_delivery'].filter((key) => Object.hasOwn(src, key))
            .map((key) => [key, src[key]])),
        answeredIndex: Number.isInteger(src.answered_index) ? src.answered_index : null,
        // The owner's verbatim words on a settled card (history replay
        // merges them from the projection). With no answeredIndex they
        // ARE the answer, not a remark beside one.
        comment: String(src.comment || ''),
        detailsUnavailable: src.option_details === undefined && raw.every((option) => typeof option === 'string'),
    };
}

export function appendRecommendedBadge(button) {
    // The asker's recommendation (the "A" option) is a badge on that option, every surface alike.
    if (button.querySelector('.chat-quiz-option-recommended')) return;
    const badge = document.createElement('span');
    badge.className = 'chat-quiz-option-recommended';
    badge.textContent = tr('quiz.recommended', 'recommended');
    button.append(badge);
}

const CHOSEN_PARTS = ['chat-quiz-option-chosen-name', 'chat-quiz-option-chosen-cue'];

/**
 * The owner's recorded answer names itself (docs/DESIGN.md "Quiz card"): after the option's label, `Your choice` joins
 * its accessible name in every theme, and a decorative copy shows where the theme draws it (ui.css
 * --chosen-answer-cue-display). Only the canonical selection carries them: a pending choice, a comment-only answer or
 * another option never does. Idempotent; true when it wrote.
 */
export function paintChosenAnswer(button, chosen) {
    const parts = CHOSEN_PARTS.map((name) => button.querySelector(`.${name}`));
    if (chosen && parts.every(Boolean)) return false;
    if (!chosen && !parts.some(Boolean)) return false;
    parts.forEach((part) => part?.remove());
    if (!chosen) return true;
    const words = tr('quiz.your_choice', 'Your choice');
    const [spoken, shown] = CHOSEN_PARTS.map((name) => {
        const part = document.createElement('span');
        part.className = name;
        part.textContent = words;
        return part;
    });
    shown.setAttribute('aria-hidden', 'true');
    const label = button.querySelector('.chat-quiz-option-label');
    if (label) { label.after(shown); label.after(spoken); } else button.append(spoken, shown);
    return true;
}


export function renderOwnerAnswer(card, comment) {
    // The owner's own words are a SECOND primary line under the question:
    // with no chosen option they are the entire answer, and beside a
    // chosen one they qualify it.
    let line = card.querySelector('.chat-quiz-answer');
    if (!comment) {
        if (!line) return false;
        line.remove();
        return true;
    }
    const text = `Owner's answer: ${comment}`;
    if (line) {
        if (line.textContent === text) return false;
        line.textContent = text;
        return true;
    }
    line = document.createElement('div');
    line.className = 'chat-quiz-answer';
    line.textContent = text;
    const assumption = card.querySelector('.chat-quiz-assumption');
    if (assumption) assumption.before(line);
    else card.append(line);
    return true;
}

const CLOSE_GLYPH = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>';

// The same answer and explicit no-answer closure controls serve Main and Project. Closing is one
// quiet × at the end of the card's head (a tall card's foot can be screens away), one tap with no
// reason to type; it waits while an answer is typed, so a mis-tap never discards the owner's words.
// The answer is one frame holding its words and its Send, as the composer holds its own.
export function appendQuestionReplyControls(card, quiz, { submit, close, showToast, onDomWrite }) {
    const box = document.createElement('div');
    box.className = 'chat-quiz-comment-box';
    const field = document.createElement('textarea');
    field.className = 'chat-quiz-comment'; field.rows = 2; field.maxLength = MAX_DECISION_COMMENT;
    // The placeholder is an example, never the field's only name (DESIGN §3).
    field.setAttribute('aria-label', tr('quiz.answer_label', 'Your answer'));
    field.placeholder = tr('quiz.answer_placeholder', 'Your answer or comment…');
    const send = document.createElement('button');
    send.type = 'button'; send.className = 'chat-quiz-send ui-hit-area';
    send.textContent = tr('quiz.send_answer', 'Send my answer'); send.disabled = true;
    const closeButton = document.createElement('button');
    closeButton.type = 'button'; closeButton.className = 'chat-quiz-close ui-hit-area';
    const closeLabel = tr('quiz.close_question', 'Close question without answering');
    closeButton.setAttribute('aria-label', closeLabel); closeButton.title = closeLabel;
    closeButton.innerHTML = CLOSE_GLYPH;
    field.addEventListener('input', () => onDomWrite(() => {
        const answerable = ANSWERABLE_QUIZ_STATES.includes(card.dataset.state);
        send.disabled = !answerable || !String(field.value || '').trim() || field.value.length > MAX_DECISION_COMMENT;
        closeButton.disabled = !answerable || Boolean(String(field.value || '').trim());
        return true;
    }));
    send.addEventListener('click', () => {
        if (!ANSWERABLE_QUIZ_STATES.includes(card.dataset.state) || !String(field.value || '').trim()) return;
        if (field.value.length > MAX_DECISION_COMMENT) {
            showToast(`Keep the answer under ${MAX_DECISION_COMMENT} characters — it is delivered word for word.`, 'error'); return;
        }
        submit(String(field.value));
    });
    closeButton.addEventListener('click', () => {
        if (closeButton.disabled || !ANSWERABLE_QUIZ_STATES.includes(card.dataset.state)) return;
        close('');
    });
    box.append(field, send); bindEnterSubmit(field, send);
    (card.querySelector('.chat-quiz-head-end') || card.querySelector('.chat-quiz-head') || box).append(closeButton);
    card.append(box);
    return field;
}
