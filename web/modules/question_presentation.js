import { tr } from './i18n.js';
import { MAX_QUIZ_OPTIONS, MAX_DECISION_COMMENT } from './api_types.js';
import { bindEnterSubmit } from './ui_interactions.js';
import { secretRequestOf } from './secret_hitl.js';

export const WAIT_FIELDS = ['wait_for_answer', 'wait_ended_at', 'owner_wait_state', 'owner_wait_resume_reason'];

// Read-only presentation of one owner question: the quiz card header, in the Project and in its
// Main mirror alike. Task liveness and answerability are separate facts: a status
// leads with one word that answers "is there an unanswered question for me?" and
// keeps the lifecycle context after it. The Python pointer fallback
// (ouroboros/project_dialogue.py::QUESTION_STATUS) emits the same words; the shared
// fixture web/tests/fixtures/question_presentation_parity.json pins both sides on the
// rows Python actually emits. Whether the question's task is a paused member of its work is the
// browser's own work projection (`paused`, work_status.memberPaused), so only this side says it.
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
// Whether a current question still waits on the owner: unanswered (its task may wait, go on or have ended), or a
// Project's answer a restart left undelivered, which only the owner can remove. Main's pill counts by this rule; the
// server's rooms follow its twin (ouroboros/gateway/main_decisions.py question_waits; the parity fixture pins both).
export const questionWaits = row => ANSWERABLE_QUIZ_STATES.includes(row.quiz_state)
    || (Boolean(row.project_id) && unknownDeliveryRemovable(row));

// What a census read vouches for: complete, everything; incomplete, every question its gaps do
// not name (a gap names a task, or a task and its quiz). A gap naming no task (index, registry or
// bindings unavailable), an unexplained incompleteness or a failed read vouches for nothing.
// Main's pill and the rooms' turn keep a question by this one scope (main_waiting.js, carryOpenQuestions).
export function vouchedBy(data) {
    if (data?.current_questions_complete === true) return () => true;
    const gaps = Array.isArray(data?.current_questions_gaps) ? data.current_questions_gaps : [];
    if (!gaps.length || gaps.some(gap => !gap?.task_id)) return () => false;
    return (taskId, quizId) => !gaps.some(gap => String(gap.task_id) === String(taskId)
        && (!gap.quiz_id || String(gap.quiz_id) === String(quizId)));
}

const listed = value => (Array.isArray(value) ? value : []);
// One question's exact address: a bare wait without a quiz never stands for an addressed question (as on the server).
const sameAddress = (one, other) => String(one?.task_id) === String(other?.task_id)
    && String(one?.quiz_id || '') === String(other?.quiz_id || '');

/**
 * One state body with the `open_question` items an earlier body (`prior`) had in its rooms and this one dropped
 * without vouching against them (`vouchedBy`): the server recomputes each room's turn from the questions it could
 * read, so a read with a gap would turn a room's amber off for one read while Main's pill keeps the question. A kept
 * item stays in its room (with its work when that work is listed there, else as the room's own) and the room stays
 * the owner's turn; a read never invents an item and a read that vouches against one, a complete one included,
 * removes it. A room kept so keeps its feed row too: the row `prior.projects` held for it stays after this body's
 * `projects` when they no longer hold it (the server kept it only while it could read the question), unless the
 * body's `project_chat_ids` no longer name the room. A deleted Project's tombstone stays named there, so its row can
 * come back for as long as reads cannot vouch against its question. A body without the
 * work projection is returned as it is.
 */
export function carryOpenQuestions(prior, data) {
    if (!data || typeof data !== 'object' || !Array.isArray(data.works) || !data.rooms || typeof data.rooms !== 'object') {
        return data;
    }
    const vouched = vouchedBy(data);
    const complete = data.works_complete === true && data.supervisor_ready === true;
    const rooms = { ...data.rooms };
    const kept = new Set();
    let works = data.works;
    for (const [chat, before] of Object.entries(prior?.rooms || {})) {
        for (const known of listed(before?.turn)) {
            // A known question keeps its address whatever its item was: a wait this body no longer confirms is carried
            // as the open question it still is, never as a wait nobody holds; a bare wait has no proved address.
            const exact = known?.kind === 'question' && known.quiz_id;
            if (!(known?.kind === 'open_question' || exact) || !known.task_id || vouched(known.task_id, known.quiz_id)) continue;
            const item = { ...known, kind: 'open_question' };
            const room = rooms[chat] && typeof rooms[chat] === 'object' ? rooms[chat]
                : { activity: complete ? 'idle' : 'unknown', transition: 'none', call: '', turn: [], amber: false,
                    mark: complete ? 'none' : 'ring', confirmed: complete, works: [] };
            if (listed(room.turn).some(entry => ['question', 'open_question'].includes(entry?.kind) && sameAddress(entry, item))) continue;
            const { work_id: workId, ...own } = item;
            const holder = workId && listed(room.works).includes(workId) ? workId : '';
            if (holder) works = works.map(work => (work?.work_id === holder && String(work.room ?? 0) === chat
                ? { ...work, turn: [...listed(work.turn), own] } : work));
            rooms[chat] = { ...room, turn: [...listed(room.turn), holder ? item : own], amber: true };
            kept.add(String(chat));
        }
    }
    let projects = data.projects;
    if (Array.isArray(projects) && kept.size) {
        const named = Array.isArray(data.project_chat_ids) ? new Set(data.project_chat_ids.map(String)) : null;
        const shown = new Set(projects.map(row => String(row?.chat_id)));
        const rows = listed(prior?.projects).filter(row => kept.has(String(row?.chat_id))
            && !shown.has(String(row?.chat_id)) && (!named || named.has(String(row?.chat_id))));
        if (rows.length) projects = [...projects, ...rows];
    }
    return works === data.works && projects === data.projects
        && Object.keys(rooms).every(chat => rooms[chat] === data.rooms[chat])
        ? data : { ...data, rooms, works, ...(projects === data.projects ? {} : { projects }) };
}

// The lifecycle code of one row: a settled state names itself; an open one says whether the task waits.
function lifecycleKey(row) {
    const state = row.quiz_state || row.state || 'unknown';
    const { waiting, resumed } = waitFacts(row);
    return !QUIZ_LIFECYCLE.includes(state) ? 'unknown' : state !== 'open' ? state
        : waiting ? 'waiting' : resumed ? 'resumed' : 'open';
}

// A paused task reads its mail only after its Resume, which nobody presses for the owner (owner 07.10): an answer it
// waits for, or one saved in its mailbox, says so while its work is paused, in one clause after its lead word (owner
// 04.11, 1A: why it is paused is the work's own line). A queued or held one keeps the plain words.
const PAUSED_STATUS = {
    waiting: 'Waiting for your answer · read after Resume',
    answer_saved: 'Answer saved · read after Resume',
};

export function questionPresentation(row = {}, { paused = false } = {}) {
    if (deliveryOwed(row)) {
        if ((row.quiz_state || row.state) !== 'answered') {
            return { status: tr('question.status.closure_pending', 'Question closed · update pending') };
        }
        // An accepted late answer whose dispatch witness ended with its process: no retry knows more.
        // An answer in its task's mailbox waits for that task to read it, also through a pause.
        const reason = row.question_delivery.reason_code;
        return { status: reason === 'late_answer_delivery_unknown'
            ? tr('question.status.answer_unknown', 'Answer recorded · delivery unknown')
            : reason === 'awaiting_recipient_read' && paused
                ? tr('question.status.answer_saved_paused', PAUSED_STATUS.answer_saved)
                : reason === 'awaiting_recipient_read'
                    ? tr('question.status.answer_saved', 'Answer saved · the task reads it when it continues')
                    : tr('question.status.answer_pending', 'Answer recorded · delivery pending') };
    }
    const key = lifecycleKey(row);
    if (paused && key === 'waiting') return { status: tr('question.status.waiting_paused', PAUSED_STATUS.waiting) };
    // The English words stay the source (and the Python twin's); the install language reads
    // them by the lifecycle code (web/modules/i18n.js).
    return { status: tr(`question.status.${key}`, STATUS[key]) };
}

// The dot beside that word (DESIGN "Quiz card"): the owner's amber exactly while Main's pill counts the question
// (`questionWaits`: unanswered, whether its task waits, went on or ended, or an answer whose delivery is unknown), ok
// once it is answered, the head's own disabled ink for a replaced or closed question, and neutral for any other
// lifecycle (status unavailable). The words beside it stay the lifecycle's own.
const MARK_HUES = { answered: 'ok', superseded: 'ink', closed: 'ink' };

/** The question head's mark as a status-mark view ({ shape, hue }, web/modules/status_mark.js). */
export function questionMark(row = {}) {
    // A card carries its state as `quiz_state` or as `state`: the pill's rule reads the one field. Amber exactly while
    // the pill counts it: a Project's answer whose delivery is unknown (Remove from pending) is, Main's own is not.
    if (questionWaits({ ...row, quiz_state: row.quiz_state || row.state })) return { shape: 'dot', hue: 'owner' };
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
        answerFields: Object.fromEntries(['answered_index', 'comment', 'question_delivery', 'secret_outcome', 'secret_save']
            .filter((key) => Object.hasOwn(src, key))
            .map((key) => [key, src[key]])),
        answeredIndex: Number.isInteger(src.answered_index) ? src.answered_index : null,
        // The owner's verbatim words on a settled card (history replay
        // merges them from the projection). With no answeredIndex they
        // ARE the answer, not a remark beside one.
        comment: String(src.comment || ''),
        // An integration-secret request (showSecretHitl): its protected form replaces the free answer.
        secretRequest: secretRequestOf(src),
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
// The wrapper keeps the answer lifecycle together, while the textarea owns the visible field and Send follows below it.
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
