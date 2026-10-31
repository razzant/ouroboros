// Quiz-local draft and one immutable, explicitly submitted attempt. This is not an
// outbox: restoration and recovery only GET; only a new owner press can POST. Drafts
// save as they are typed and an emptied field releases them: no separate control
// clears them (docs/DESIGN.md "Quiz card").
import { MAX_DECISION_COMMENT } from './api_types.js';
import { ANSWERABLE_QUIZ_STATES } from './question_presentation.js';
import { tr } from './i18n.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { attachChatInputSizing } from './chat_input_sizing.js';

const PREFIX = 'ouro_quiz_reply_v1:';
const LOCAL_EVENT = 'ouro:quiz-reply';
const WAIT_MS = 15000;
const keyOf = quiz => `${PREFIX}${JSON.stringify([quiz.taskId, quiz.quizId])}`;
const attemptKey = quiz => `${keyOf(quiz)}:attempt`;
const terminal = state => ['answered', 'closed', 'superseded'].includes(state);
const nonce = () => globalThis.crypto?.randomUUID?.() || `q-${Date.now()}-${Math.random().toString(36).slice(2)}`;

// A stalled transport/body/detail read must release the UI, not declare failure.
// Its late positive answer is still consumed by the caller's original promise.
function bounded(promise, ms, view) {
    let timer, cancel;
    const timeout = new Promise(resolve => {
        cancel = () => { clearTimeout(timer); resolve(null); };
        timer = setTimeout(cancel, ms);
        view.waits.add(cancel);
    });
    return Promise.race([promise, timeout]).finally(() => { clearTimeout(timer); view.waits.delete(cancel); });
}

function validRecord(value, quiz) {
    if (!value || value.v !== 1 || typeof value.draft !== 'string' || typeof value.reason !== 'string'
        || value.draft.length > MAX_DECISION_COMMENT || value.reason.length > MAX_DECISION_COMMENT) return false;
    if (!value.attempt) return true;
    const body = value.attempt;
    return typeof body.request_id === 'string' && body.request_id.length > 0
        && body.decision_id === `quiz:${quiz.taskId}:${quiz.quizId}`
        && (!Object.hasOwn(body, 'option_index') || (Number.isInteger(body.option_index) && body.option_index >= 0
            && body.option_index < quiz.options.length))
        && !Object.hasOwn(body, 'action')
        && (!Object.hasOwn(body, 'comment') || (typeof body.comment === 'string' && body.comment.length <= MAX_DECISION_COMMENT))
        && (!Object.hasOwn(body, 'reason') || (typeof body.reason === 'string' && body.reason.length <= MAX_DECISION_COMMENT));
}

export function createQuizSubmission({ apiFetch, readQuestion, accept, canUse, onDomWrite, showToast = () => {},
    paintHead = () => false, storage = () => globalThis.localStorage, waitMs = WAIT_MS }) {
    const views = new Map();
    let disposed = false;
    const alive = view => !disposed && views.get(view.card) === view;
    function load(quiz) {
        try {
            const store = storage();
            const raw = store?.getItem(attemptKey(quiz)) || store?.getItem(keyOf(quiz));
            if (!raw) return { record: { v: 1, draft: '', reason: '', attempt: null }, storageFailed: false };
            const record = JSON.parse(raw);
            if (!validRecord(record, quiz)) throw new Error('Invalid local quiz reply');
            return { record, storageFailed: false };
        } catch { return { record: { v: 1, draft: '', reason: '', attempt: null }, storageFailed: true }; }
    }
    function adopt(view, stored) {
        const changed = JSON.stringify(view.record.attempt) !== JSON.stringify(stored.record.attempt);
        Object.assign(view, stored);
        if (changed || (view.record.attempt && !view.phase)) {
            view.note = ''; view.phase = view.record.attempt ? 'unknown' : '';
        }
        if (!view.record.attempt) for (const [selector, value] of [
            ['.chat-quiz-comment', view.record.draft], ['.chat-quiz-close-reason', view.record.reason],
        ]) {
            const field = view.card.querySelector(selector); if (field) field.value = value;
        }
        view.sizing?.sync();
        return changed;
    }
    // Draft edits have their own key: typing before a queued storage event cannot
    // overwrite immutable submitted words. Phase publication never writes storage.
    function save(view, operation = 'phase', expectedAttempt = null) {
        if (operation !== 'phase') try {
            const store = storage();
            if (!store) throw new Error('Local storage unavailable');
            if (operation === 'attempt') store.setItem(attemptKey(view.quiz), JSON.stringify(view.record));
            if (operation === 'settled' || operation === 'forget') {
                store.removeItem(attemptKey(view.quiz)); store.removeItem(keyOf(view.quiz));
            } else if (operation === 'draft' || operation === 'refused') {
                const draft = { ...view.record, attempt: null };
                if (!draft.draft && !draft.reason) store.removeItem(keyOf(view.quiz));
                else store.setItem(keyOf(view.quiz), JSON.stringify(draft));
                if (operation === 'refused') {
                    const raw = store.getItem(attemptKey(view.quiz));
                    if (raw && JSON.stringify(JSON.parse(raw).attempt) === expectedAttempt) store.removeItem(attemptKey(view.quiz));
                }
            }
            view.storageFailed = false;
        } catch { view.storageFailed = true; }
        const publication = JSON.stringify([view.record, view.phase]);
        const changed = view.lastPublished !== publication;
        view.lastPublished = publication;
        if (changed && typeof CustomEvent !== 'undefined') globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:quiz-reply',
            { detail: { key: keyOf(view.quiz), record: view.record, sender: view, phase: view.phase } }));
    }
    function paint(view) {
        if (!alive(view)) return;
        onDomWrite(() => {
            const { card, record } = view;
            const attempt = record.attempt;
            const held = Boolean(attempt) && !terminal(card.dataset.state);
            card.dataset.submission = held ? view.phase : '';
            paintHead(card);  // the head's mark follows the answer in flight
            card.setAttribute('aria-busy', held && ['sending', 'checking'].includes(view.phase) ? 'true' : 'false');
            const field = card.querySelector('.chat-quiz-comment');
            const reason = card.querySelector('.chat-quiz-close-reason');
            if (field) {
                field.readOnly = held;
                if (held && field.value !== record.draft) { field.value = record.draft; view.sizing?.sync(); }
            }
            const send = card.querySelector('.chat-quiz-send');
            if (send && field) send.disabled = !ANSWERABLE_QUIZ_STATES.includes(card.dataset.state) || !String(field.value || '').trim() || String(field.value).length > MAX_DECISION_COMMENT;
            const close = card.querySelector('.chat-quiz-close');
            if (close) close.disabled = held || !ANSWERABLE_QUIZ_STATES.includes(card.dataset.state) || Boolean(String(field?.value || '').trim());
            if (reason) { reason.readOnly = held; if (held && reason.value !== record.reason) reason.value = record.reason; }
            for (const selector of ['.chat-quiz-option', '.chat-quiz-send', '.chat-quiz-close', '.chat-quiz-close-submit', '.chat-quiz-dismiss'])
                card.querySelectorAll(selector).forEach(button => button.setAttribute('aria-disabled', held ? 'true' : 'false'));
            card.querySelectorAll('.chat-quiz-option').forEach((button, index) =>
                { const chosen = held && attempt.option_index === index;
                    if (button.classList.contains('pending-choice') !== chosen) button.classList.toggle('pending-choice', chosen); });
            const status = card.querySelector('.chat-quiz-status-text');
            if (held && status) status.textContent = view.phase === 'sending'
                ? tr('quiz.sending', 'Sending…') : tr('quiz.unconfirmed', 'Not confirmed');
            view.line.textContent = held ? (view.phase === 'sending'
                ? tr('quiz.sending_note', 'Your choice is being sent. Waiting for confirmation.')
                : view.phase === 'checking' ? tr('quiz.checking_note', 'Checking the recorded question…')
                    : view.note || tr('quiz.unconfirmed_note', 'Confirmation unavailable. Your original answer is kept; nothing is resent automatically.')) : view.note || '';
            view.warning.textContent = view.storageFailed && !terminal(card.dataset.state)
                ? tr('quiz.local_unavailable', 'Local saving is unavailable. Keep this window open or copy your words before closing.') : '';
            view.check.hidden = view.retry.hidden = view.forget.hidden = !held || view.phase !== 'unknown';
            view.check.disabled = view.retry.disabled = !canUse(card) || ['sending', 'checking'].includes(view.phase);
            view.feedback.hidden = !view.line.textContent && !view.warning.textContent && view.check.hidden;
            return true;
        });
    }
    function settle(view, frame) {
        if (!terminal(frame?.state)) return false;
        view.record = { v: 1, draft: '', reason: '', attempt: null };
        view.phase = ''; view.note = '';
        save(view, 'settled');
        accept(view.card, frame);
        return true;
    }
    async function reconcile(view) {
        if (!alive(view) || !canUse(view.card)) return null;
        const attempt = JSON.stringify(view.record.attempt), observation = view.observation;
        view.phase = 'checking'; paint(view);
        const reading = Promise.resolve().then(() => readQuestion(view.quiz)).then(frame => {
            if (!alive(view)) return null;
            if (settle(view, frame)) return frame;
            return frame;
        }).catch(() => null);
        const frame = await bounded(reading, waitMs, view);
        if (!alive(view) || !view.record.attempt || JSON.stringify(view.record.attempt) !== attempt
            || view.observation !== observation) return frame;
        view.phase = 'unknown';
        view.note = frame && ANSWERABLE_QUIZ_STATES.includes(frame.state)
            ? tr('quiz.still_open', 'The question is still unanswered. You can explicitly retry the same answer.')
            : tr('quiz.read_unavailable', 'Question status could not be confirmed. Your original answer is kept.');
        paint(view);
        return frame;
    }
    function confirmation(view, body, response) {
        if (!body || !terminal(body.state)) return false;
        if (body.state === 'answered') {
            const index = Number.isInteger(body.answered_index) ? body.answered_index : null;
            if ((index !== null && (index < 0 || index >= view.quiz.options.length))
                || (index === null && !(typeof body.comment === 'string' && body.comment.trim()))) return false;
            // 503 explicitly describes a durable answer with owed delivery; 409 names
            // the winner. A success response without ok:true is not confirmation.
            if (!(response?.ok && body.ok === true) && response?.status !== 409
                && !(response?.status === 503 && body.question_delivery?.state === 'owed')) return false;
        } else if (!(response?.status === 409 || (response?.ok && body.ok === true))) return false;
        const settled = settle(view, body);
        if (settled && response?.status === 409) showToast(body.state === 'answered' ? 'Already answered.' : 'This question is no longer open.', 'error');
        if (settled && body.answered_after_terminal === true) showToast(body.forwarded === true
            ? 'Answer recorded. The task had finished, so it was delivered to its chat as your message.'
            : body.question_delivery?.state === 'owed' ? 'Answer recorded; delivery to the chat is pending.'
                : 'Answer recorded. The task had finished; nothing is waiting on it.', 'info');
        return settled;
    }
    async function dispatch(view) {
        if (!alive(view) || !canUse(view.card) || !view.record.attempt) return;
        const stored = load(view.quiz);
        // An empty read after our failed write is not proof the press vanished.
        // A positively stored replacement still wins over the memory-only copy.
        if (!stored.storageFailed && (!view.storageFailed || stored.record.attempt)) {
            if (adopt(view, stored)) { paint(view); return; }
        }
        // Persist before I/O: a refusal proves only this physical send. Once a
        // retry exists (or an older record has no dispatch facts), no negative
        // reply can discharge the earlier uncertainty, even after reload.
        const dispatch = { id: nonce(), sole: view.record.dispatch === null };
        view.record = { ...view.record, dispatch };
        view.observation += 1;
        view.phase = 'sending'; view.note = ''; save(view, 'attempt'); paint(view);
        const attempt = view.record.attempt;
        const body = JSON.stringify(attempt);
        const ownsAttempt = () => JSON.stringify(view.record.attempt) === body;
        const controller = new AbortController(); view.controllers.add(controller);
        const sending = (async () => {
            const response = await apiFetch('/api/decisions', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body, signal: controller.signal,
            });
            let reply = null;
            try { reply = await response?.json?.(); } catch { /* unknown body, never a refusal */ }
            if (!alive(view) || terminal(view.card.dataset.state)) return true;
            if (confirmation(view, reply, response)) return true;
            // Only definite validation/auth refusals permit a changed new attempt.
            // 409 without lifecycle and every 5xx may have followed a saved answer.
            if ([400, 401, 403, 404, 422].includes(response?.status) && ownsAttempt()) {
                if (!dispatch.sole || view.storageFailed || view.record.dispatch?.id !== dispatch.id) return false;
                const current = load(view.quiz);
                if (current.storageFailed) { view.storageFailed = true; paint(view); return false; }
                adopt(view, current);
                if (!ownsAttempt() || view.record.dispatch?.id !== dispatch.id
                    || view.record.dispatch?.sole !== true) { paint(view); return false; }
                view.record = { ...view.record, attempt: null }; view.phase = '';
                view.note = tr('quiz.refused', 'Answer not accepted. Your draft is kept; check the question before trying again.');
                save(view, 'refused', body); accept(view.card, null); paint(view);
                showToast(`Could not record the answer (${response.status}).`, 'error'); return true;
            }
            return false;
        })().catch(() => false).finally(() => view.controllers.delete(controller));
        const confirmed = await bounded(sending, waitMs, view);
        if (!alive(view) || !ownsAttempt() || terminal(view.card.dataset.state)
            || view.record.dispatch?.id !== dispatch.id) return;
        if (!confirmed) {
            view.phase = 'unknown'; save(view); paint(view);
            showToast('Answer not confirmed. Your original answer is kept.', 'info');
            await reconcile(view);
        }
    }
    function attach(card, quiz) {
        if (views.has(card)) return;
        const view = { card, quiz, ...load(quiz), phase: 'unknown', observation: 0, note: '', waits: new Set(), controllers: new Set() };
        const feedback = document.createElement('div'); feedback.className = 'chat-quiz-feedback';
        const line = document.createElement('div'); line.className = 'chat-quiz-submission-note'; line.setAttribute('role', 'status');
        const warning = document.createElement('div'); warning.className = 'chat-quiz-storage-note';
        const check = document.createElement('button'); check.type = 'button'; check.className = 'chat-quiz-check btn btn-xs btn-default';
        check.textContent = tr('quiz.check_status', 'Check status');
        const retry = document.createElement('button'); retry.type = 'button'; retry.className = 'chat-quiz-retry btn btn-xs btn-default';
        retry.textContent = tr('quiz.retry_same', 'Retry same answer');
        // Forget deletes only the local copy: a quiet control at the row's start, the two server actions at its end.
        const forget = document.createElement('button'); forget.type = 'button'; forget.className = 'chat-quiz-forget btn btn-xs btn-ghost';
        forget.textContent = tr('quiz.forget_unconfirmed', 'Forget unconfirmed local answer');
        Object.assign(view, { feedback, line, warning, check, retry, forget });
        views.set(card, view); feedback.append(line, warning, forget, check, retry);
        // The notes and recovery speak of the answer, so they follow it rather than the card's foot.
        const box = card.querySelector('.chat-quiz-comment-box');
        if (box) box.after(feedback); else card.append(feedback);
        // The answer is a message field: it grows with its words to one cap, then scrolls (DESIGN §3).
        const answer = card.querySelector('.chat-quiz-comment');
        view.sizing = answer ? attachChatInputSizing(answer) : null;
        const edited = () => {
            view.sizing?.sync();
            const stored = load(quiz);
            if (stored.record.attempt) {
                const changed = adopt(view, stored); paint(view);
                if (changed) void reconcile(view);
                return;
            }
            if (view.record.attempt) { paint(view); return; }
            view.record = { v: 1, draft: card.querySelector('.chat-quiz-comment')?.value || '',
                reason: card.querySelector('.chat-quiz-close-reason')?.value || '', attempt: null };
            // Emptying the unsent words also releases a refusal about them; a frozen attempt never gets here.
            if (!view.record.draft && !view.record.reason) view.note = '';
            save(view, 'draft'); paint(view);
        };
        card.querySelector('.chat-quiz-comment')?.addEventListener('input', edited);
        card.querySelector('.chat-quiz-close-reason')?.addEventListener('input', edited);
        check.addEventListener('click', () => { if (!check.disabled) void reconcile(view); });
        retry.addEventListener('click', async () => {
            if (retry.disabled) return;
            const attempt = view.record.attempt;
            const frame = await reconcile(view);
            if (alive(view) && view.record.attempt === attempt && ANSWERABLE_QUIZ_STATES.includes(frame?.state)
                && canUse(card)) await dispatch(view);
        });
        forget.addEventListener('click', async () => {
            if (forget.hidden || !view.record.attempt) return;
            const attempt = JSON.stringify(view.record.attempt);
            const confirmed = await openConfirmDialog({ title: tr('quiz.forget_title', 'Forget this local answer?'),
                body: tr('quiz.forget_warning', 'This deletes only the local unconfirmed copy. It does not cancel or retract an answer the server may already have saved. A later confirmation can still show that answer.'),
                confirmLabel: tr('quiz.forget_confirm', 'Forget local copy') });
            if (confirmed !== true || !alive(view) || JSON.stringify(view.record.attempt) !== attempt) return;
            // Storage events are queued: another window may already have replaced
            // the selected answer while this dialog was open. Never delete it.
            const current = load(quiz);
            if (current.storageFailed) { view.storageFailed = true; paint(view); return; }
            if (JSON.stringify(current.record.attempt) !== attempt) {
                adopt(view, current); paint(view);
                return;
            }
            view.record = { v: 1, draft: '', reason: '', attempt: null }; view.phase = ''; view.note = '';
            const field = card.querySelector('.chat-quiz-comment'), reason = card.querySelector('.chat-quiz-close-reason');
            if (field) field.value = ''; if (reason) reason.value = '';
            view.sizing?.sync();
            save(view, 'forget'); accept(card, null); paint(view);
        });
        const field = card.querySelector('.chat-quiz-comment');
        const reason = card.querySelector('.chat-quiz-close-reason');
        if (field) field.value = view.record.draft;
        if (reason) reason.value = view.record.reason;
        view.sizing?.sync();
        if (terminal(card.dataset.state)) settled(card);
        else { paint(view); if (view.record.attempt) void reconcile(view); }
    }
    function settled(card) {
        const view = views.get(card);
        if (!view) return;
        if (terminal(card.dataset.state)) {
            view.record = { v: 1, draft: '', reason: '', attempt: null }; view.phase = ''; view.note = ''; save(view, 'settled');
            // A settled card has dropped its answer field; its sizing observers go with it.
            view.sizing?.dispose(); view.sizing = null;
        }
        paint(view);
    }
    function submit(card, quiz, index, text) {
        const view = views.get(card);
        if (!view || !canUse(card) || card.dataset.pending === '1' || view.record.attempt || !ANSWERABLE_QUIZ_STATES.includes(card.dataset.state)) return;
        const stored = load(quiz);
        if (stored.record.attempt) { adopt(view, stored); paint(view); void reconcile(view); return; }
        if (String(text).length > MAX_DECISION_COMMENT) return;
        // Read-only fields retain verbatim bytes. A new deliberate submission gets
        // a new ID; the recovery button alone can reuse this frozen body.
        view.record = { ...view.record,
            dispatch: null,
            draft: card.querySelector('.chat-quiz-comment')?.value || '',
            reason: card.querySelector('.chat-quiz-close-reason')?.value || '',
            attempt: { request_id: nonce(),
                decision_id: `quiz:${quiz.taskId}:${quiz.quizId}`,
                ...(Number.isInteger(index) ? { option_index: index } : {}),
                ...(text ? { comment: text } : {}) } };
        save(view, 'attempt');
        void dispatch(view); // synchronous pending paint precedes the first await/network call
    }
    function localChange(event) {
        for (const view of views.values()) {
            const hadAttempt = Boolean(view.record.attempt);
            if (event.type === LOCAL_EVENT) {
                if (event.detail?.sender === view || event.detail?.key !== keyOf(view.quiz)) continue;
                adopt(view, { record: event.detail.record }); view.phase = event.detail.phase;
                view.lastPublished = JSON.stringify([view.record, view.phase]);
            } else {
                if (![keyOf(view.quiz), attemptKey(view.quiz)].includes(event.key)) continue;
                const stored = load(view.quiz);
                // As in dispatch, an empty/draft-only read cannot retire a press
                // whose latest persistence failed. Canonical settlement and an
                // explicit Forget still resolve it; a stored attempt can replace it.
                if (!stored.storageFailed && (!view.storageFailed || stored.record.attempt)) adopt(view, stored);
            }
            paint(view);
            if ((hadAttempt && !view.record.attempt) || (event.type === 'storage' && view.record.attempt && view.phase === 'unknown')) void reconcile(view);
        }
    }
    function release(card) {
        const view = views.get(card);
        if (!view) return;
        views.delete(card);
        if (view.record.attempt && !terminal(card.dataset.state) && ['sending', 'checking'].includes(view.phase)) {
            view.phase = 'unknown'; save(view);
        }
        view.waits.forEach(cancel => cancel()); view.controllers.forEach(controller => controller.abort());
        view.sizing?.dispose(); view.sizing = null;
    }
    globalThis.window?.addEventListener?.(LOCAL_EVENT, localChange);
    globalThis.window?.addEventListener?.('storage', localChange);
    return { attach, submit, settled, paint: card => { const view = views.get(card); if (view) paint(view); },
        release,
        destroy() {
            disposed = true; [...views.keys()].forEach(release);
            globalThis.window?.removeEventListener?.(LOCAL_EVENT, localChange);
            globalThis.window?.removeEventListener?.('storage', localChange);
        } };
}
