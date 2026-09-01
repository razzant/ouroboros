// Owner decision cards: the typed quiz card (question + option buttons +
// stake + assumption) and the routing picker (#198) — one decision-card
// family, one answer contract (POST /api/decisions). The quiz card is
// fire-and-continue UI: the asking task keeps working under the stated
// assumption, so it must read correctly both as "you can redirect me" (open)
// and as a record of what happened (answered / expired). The routing picker
// settles into the plain routing ack line once its dispatch is confirmed.
import { MAX_DECISION_COMMENT, MAX_QUIZ_OPTIONS } from './api_types.js';
import { renderRoutingAnnotation, routingOptionLabel } from './chat_activity.js';

const QUIZ_STATUS_TEXT = {
    open: 'Awaiting answer',
    answered: 'Answered',
    expired_terminal: 'Task finished — question expired',
    superseded: 'Superseded by a retry',
};

// Neutral, factual statuses (owner decision 15~A): the card never scolds the
// router — it states what the click does and what happened.
const ROUTING_STATUS_TEXT = {
    open: 'Choose a destination',
    pending: 'Routing…',
    answered: 'Routed',
    superseded: 'Superseded by a newer attempt',
};
const ROUTING_TOP_OPTIONS = 8;

export function createChatDecision({
    apiFetch,
    frameNode,
    renderMarkdown,
    enhanceMarkdown,
    showToast,
    onDomWrite = (mutate) => mutate(),
}) {
    function normalizeQuiz(msg) {
        const nested = msg && typeof msg.quiz === 'object' && msg.quiz ? msg.quiz : null;
        const src = nested || msg || {};
        // Strict per-card validation: ONE corrupt option refuses THIS card
        // (buildQuizCard -> null), never the whole history hydration pass.
        // Filtering instead would silently shift option_index against the
        // producer's original list — a wrong answer, not a degraded card.
        const raw = Array.isArray(src.options) ? src.options : [];
        const normalized = raw.map((option) => (typeof option === 'string' ? { label: option } : option));
        const corrupt = normalized.some(
            (option) => !option || typeof option !== 'object' || !String(option.label || '').trim());
        const options = corrupt ? [] : normalized.slice(0, MAX_QUIZ_OPTIONS);
        return {
            quizId: String(src.quiz_id || ''),
            question: String((nested ? msg.text : src.question) || ''),
            options,
            stake: String(src.stake || ''),
            assumption: String(src.assumption || ''),
            state: String(src.state || 'open'),
            taskId: String(msg.task_id || ''),
            ts: msg.ts || null,
            answeredIndex: Number.isInteger(src.answered_index) ? src.answered_index : null,
            // The owner's verbatim words on a settled card (history replay
            // merges them from the projection). With no answeredIndex they
            // ARE the answer, not a remark beside one.
            comment: String(src.comment || ''),
        };
    }

    function statusText(state) {
        // Unknown states read as settled, never as an open invitation.
        return QUIZ_STATUS_TEXT[state] || 'Closed';
    }

    async function submitAnswer(card, quiz, index, comment) {
        if (card.dataset.pending === '1') return;
        card.dataset.pending = '1';
        const text = String(comment || '');
        // STABLE per-card idempotency key: a retry after a transient failure
        // must replay the SAME request, or the server-side first-wins latch
        // reads the retry as a competing second answer.
        if (!card.dataset.requestId) {
            card.dataset.requestId = (crypto.randomUUID && crypto.randomUUID()) || `q-${Date.now()}`;
        }
        try {
            const res = await apiFetch('/api/decisions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    request_id: card.dataset.requestId,
                    decision_id: `quiz:${quiz.taskId}:${quiz.quizId}`,
                    // Omitted, never null: no option means the owner took none
                    // of them and the comment carries the whole answer.
                    ...(Number.isInteger(index) ? { option_index: index } : {}),
                    ...(text ? { comment: text } : {}),
                }),
            });
            let body = null;
            try { body = res && res.json ? await res.json() : null; } catch (parseErr) { body = null; }
            if (res && res.ok) {
                // The confirmation is the display truth: a same-request_id
                // retry may have carried a different payload, and the server
                // answers with what was actually RECORDED — index absent for a
                // free answer, comment as stored. Never render this attempt's
                // own click over it.
                const answered = body && Number.isInteger(body.answered_index)
                    ? body.answered_index
                    : (body && body.duplicate ? null : (Number.isInteger(index) ? index : null));
                const recorded = body && typeof body.comment === 'string' ? body.comment : text;
                if (recorded) card.dataset.ownerComment = recorded;
                else delete card.dataset.ownerComment;
                setCardState(card, 'answered', answered);
                return;
            }
            const status = res ? res.status : 0;
            if (status === 409 && body && body.state) {
                // The refusal body carries the card's TRUE lifecycle state —
                // an already-answered quiz settles as answered (with the
                // winning option when known), never as a false expiry.
                const answered = Number.isInteger(body.answered_index) ? body.answered_index : null;
                // The 409 loser learns the WINNING answer, comment included —
                // the local draft must not survive as the displayed record.
                if (typeof body.comment === 'string' && body.comment) card.dataset.ownerComment = body.comment;
                else delete card.dataset.ownerComment;
                setCardState(card, body.state, answered);
                showToast(body.state === 'answered'
                    ? 'Already answered.' : 'This question is no longer open.', 'error');
                return;
            }
            if (status === 409 && card.dataset.state === 'open') {
                setCardState(card, 'expired_terminal', null);
                showToast('This question is no longer open.', 'error');
                return;
            }
            showToast(`Could not record the answer (${status || 'network error'}).`, 'error');
        } catch (err) {
            showToast('Could not record the answer (network error).', 'error');
        } finally {
            delete card.dataset.pending;
        }
    }

    function renderOwnerAnswer(card, comment) {
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

    function setCardState(card, state, answeredIndex) {
        if (!card) return false;
        return onDomWrite(() => {
            let changed = card.dataset.state !== state;
            if (changed) card.dataset.state = state;
            if (state !== 'open') {
                // A settled card takes no more input: the draft field goes,
                // and what the owner actually said takes its place.
                const box = card.querySelector('.chat-quiz-comment-box');
                if (box) { box.remove(); changed = true; }
                if (renderOwnerAnswer(card, String(card.dataset.ownerComment || ''))) changed = true;
            }
            const status = card.querySelector('.chat-quiz-status-text');
            const nextStatus = statusText(state);
            if (status && status.textContent !== nextStatus) {
                status.textContent = nextStatus;
                changed = true;
            }
            const buttons = card.querySelectorAll('.chat-quiz-option');
            buttons.forEach((btn, i) => {
                const disabled = state !== 'open';
                const chosen = answeredIndex !== null && i === answeredIndex;
                if (btn.disabled !== disabled) {
                    btn.disabled = disabled;
                    changed = true;
                }
                if (btn.classList.contains('chosen') !== chosen) {
                    btn.classList.toggle('chosen', chosen);
                    changed = true;
                }
            });
            return changed;
        });
    }

    function buildQuizCard(msg) {
        const quiz = normalizeQuiz(msg);
        if (!quiz.quizId || !quiz.taskId || !quiz.question || quiz.options.length < 2) return null;

        const card = document.createElement('div');
        card.className = 'chat-quiz-card';
        card.dataset.quizId = quiz.quizId;

        const head = document.createElement('div');
        head.className = 'chat-quiz-head';
        const chip = document.createElement('span');
        chip.className = 'chat-quiz-chip';
        chip.textContent = 'Question';
        const status = document.createElement('span');
        status.className = 'chat-quiz-status';
        const dot = document.createElement('span');
        dot.className = 'chat-quiz-dot';
        const statusLabel = document.createElement('span');
        statusLabel.className = 'chat-quiz-status-text';
        status.append(dot, statusLabel);
        head.append(chip, status);
        card.append(head);

        // DRY with the chat surface (owner requirement): question and stake go
        // through the SAME sanitizing markdown pipeline as assistant bubbles,
        // so chat rendering improvements reach the card automatically.
        const question = document.createElement('div');
        question.className = 'chat-quiz-question';
        if (renderMarkdown) question.innerHTML = renderMarkdown(quiz.question);
        else question.textContent = quiz.question;
        card.append(question);

        if (quiz.stake) {
            const stake = document.createElement('div');
            stake.className = 'chat-quiz-stake';
            if (renderMarkdown) stake.innerHTML = renderMarkdown(`At stake: ${quiz.stake}`);
            else stake.textContent = `At stake: ${quiz.stake}`;
            card.append(stake);
        }

        let commentField = null;
        // The raw field value is the answer (VERBATIM to the model); the
        // trimmed view only decides whether there IS one.
        const commentText = () => String((commentField && commentField.value) || '');
        const commentPresent = () => commentText().trim().length > 0;

        const optionsBox = document.createElement('div');
        optionsBox.className = 'chat-quiz-options';
        quiz.options.forEach((option, index) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'chat-quiz-option';
            const label = document.createElement('span');
            label.className = 'chat-quiz-option-label';
            label.textContent = String(option.label || '');
            btn.append(label);
            const detailText = String(option.detail || '');
            if (detailText) {
                const detail = document.createElement('span');
                detail.className = 'chat-quiz-option-detail';
                detail.textContent = detailText;
                btn.append(detail);
            }
            btn.addEventListener('click', () => {
                if (card.dataset.state !== 'open') return;
                // A typed remark rides WITH the click: the owner picked this
                // option and said why, one answer, one request.
                submitAnswer(card, quiz, index, commentText());
            });
            optionsBox.append(btn);
        });
        card.append(optionsBox);

        // Free answer: none of the options may fit, and the owner must not be
        // forced to pick the least wrong one. Always visible while the card is
        // open (no disclosure to discover), removed once it settles.
        if (quiz.state === 'open') {
            const box = document.createElement('div');
            box.className = 'chat-quiz-comment-box';
            commentField = document.createElement('textarea');
            commentField.className = 'chat-quiz-comment';
            commentField.rows = 2;
            commentField.maxLength = MAX_DECISION_COMMENT;
            commentField.placeholder = 'Your answer or comment…';
            const send = document.createElement('button');
            send.type = 'button';
            send.className = 'chat-quiz-send';
            send.textContent = 'Send my answer';
            send.disabled = true;
            const syncSend = () => {
                const text = commentText();
                const enabled = commentPresent() && text.length <= MAX_DECISION_COMMENT;
                if (send.disabled === !enabled) return;
                send.disabled = !enabled;
            };
            commentField.addEventListener('input', () => onDomWrite(() => { syncSend(); return true; }));
            send.addEventListener('click', () => {
                if (card.dataset.state !== 'open') return;
                const text = commentText();
                if (!commentPresent()) return;
                if (text.length > MAX_DECISION_COMMENT) {
                    // The ingress refuses it rather than truncating the
                    // owner's words — say so here instead of sending.
                    showToast(`Keep the answer under ${MAX_DECISION_COMMENT} characters — `
                        + 'it is delivered word for word.', 'error');
                    return;
                }
                submitAnswer(card, quiz, null, text);
            });
            box.append(commentField, send);
            card.append(box);
        }

        // The signature line: what the agent keeps doing while the owner has
        // not answered — and, once the card settles, the record of the path
        // it took by default.
        if (quiz.assumption) {
            const assumption = document.createElement('div');
            assumption.className = 'chat-quiz-assumption';
            assumption.textContent = `Continuing meanwhile: ${quiz.assumption}`;
            card.append(assumption);
        }

        if (quiz.comment) card.dataset.ownerComment = quiz.comment;
        setCardState(card, quiz.state, quiz.answeredIndex);
        const framed = frameNode(msg, card);
        if (enhanceMarkdown && renderMarkdown) enhanceMarkdown(card);
        return framed;
    }

    function setRoutingCardState(card, state, chosenIndex) {
        if (!card) return false;
        return onDomWrite(() => {
            let changed = card.dataset.state !== state;
            if (changed) card.dataset.state = state;
            const status = card.querySelector('.chat-quiz-status-text');
            const nextStatus = ROUTING_STATUS_TEXT[state] || 'Closed';
            if (status && status.textContent !== nextStatus) {
                status.textContent = nextStatus;
                changed = true;
            }
            card.querySelectorAll('.chat-quiz-option').forEach((btn, i) => {
                const disabled = state !== 'open';
                const chosen = chosenIndex !== null && i === chosenIndex;
                if (btn.disabled !== disabled) {
                    btn.disabled = disabled;
                    changed = true;
                }
                if (btn.classList.contains('chosen') !== chosen) {
                    btn.classList.toggle('chosen', chosen);
                    changed = true;
                }
            });
            return changed;
        });
    }

    async function submitRouting(card, cmid, token, index) {
        if (card.dataset.pending === '1') return;
        card.dataset.pending = '1';
        // Same idempotency discipline as the quiz card: ONE stable id per
        // card, replayed on retry, so the server latch never reads a retry
        // as a competing second click.
        if (!card.dataset.requestId) {
            card.dataset.requestId = (crypto.randomUUID && crypto.randomUUID()) || `r-${Date.now()}`;
        }
        try {
            const res = await apiFetch('/api/decisions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    request_id: card.dataset.requestId,
                    decision_id: `routing:${cmid}:${token}`,
                    option_index: index,
                }),
            });
            let body = null;
            try { body = res && res.json ? await res.json() : null; } catch (parseErr) { body = null; }
            if (res && res.ok) {
                const answered = body && Number.isInteger(body.answered_index) ? body.answered_index : index;
                setRoutingCardState(card, 'answered', answered);
                return;
            }
            const status = res ? res.status : 0;
            if (status === 409 && body && body.state) {
                // Honest settlement: the body carries the TRUE state (another
                // click won, or a newer routing attempt superseded this card).
                setRoutingCardState(card,
                    body.state === 'open' ? 'open' : body.state,
                    Number.isInteger(body.answered_index) ? body.answered_index : null);
                showToast(body.state === 'open'
                    ? `Not routed: ${body.reason || 'the destination refused this message'} — pick again.`
                    : body.state === 'pending'
                        ? 'Another choice is already being routed.'
                        : 'This message was already routed.', 'error');
                return;
            }
            showToast(`Could not route the message (${status || 'network error'}) — try again.`, 'error');
        } catch (err) {
            showToast('Could not route the message (network error) — try again.', 'error');
        } finally {
            delete card.dataset.pending;
        }
    }

    function buildRoutingCard(cmid, token, options) {
        const card = document.createElement('div');
        card.className = 'chat-quiz-card chat-routing-card';
        card.dataset.routingToken = token;

        const head = document.createElement('div');
        head.className = 'chat-quiz-head';
        const chip = document.createElement('span');
        chip.className = 'chat-quiz-chip';
        chip.textContent = 'Route';
        const status = document.createElement('span');
        status.className = 'chat-quiz-status';
        const dot = document.createElement('span');
        dot.className = 'chat-quiz-dot';
        const statusLabel = document.createElement('span');
        statusLabel.className = 'chat-quiz-status-text';
        status.append(dot, statusLabel);
        head.append(chip, status);
        card.append(head);

        const optionsBox = document.createElement('div');
        optionsBox.className = 'chat-quiz-options';
        const overflow = options.length > ROUTING_TOP_OPTIONS;
        options.forEach((option, index) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'chat-quiz-option';
            if (overflow && index >= ROUTING_TOP_OPTIONS) btn.hidden = true;
            const label = document.createElement('span');
            label.className = 'chat-quiz-option-label';
            label.textContent = routingOptionLabel(option) || `Option ${index + 1}`;
            btn.append(label);
            btn.addEventListener('click', () => {
                if (card.dataset.state !== 'open') return;
                submitRouting(card, cmid, token, index);
            });
            optionsBox.append(btn);
        });
        card.append(optionsBox);
        if (overflow) {
            const more = document.createElement('button');
            more.type = 'button';
            more.className = 'chat-quiz-more';
            more.textContent = `Show all ${options.length}`;
            more.addEventListener('click', () => onDomWrite(() => {
                optionsBox.querySelectorAll('.chat-quiz-option')
                    .forEach((btn) => { btn.hidden = false; });
                more.remove();
                return true;
            }));
            card.append(more);
        }
        setRoutingCardState(card, 'open', null);
        return card;
    }

    function renderRoutingDecision(bubble, annotation) {
        // ONE entry point for a user bubble's routing surface: an actionable
        // refusal renders the picker card; every other annotation state
        // settles back into the plain text ack line.
        if (!bubble) return false;
        return onDomWrite(() => {
            const cmid = String(bubble.dataset.clientMessageId || '');
            const status = String((annotation && annotation.status) || '');
            const token = String((annotation && annotation.routing_token) || '');
            const options = Array.isArray(annotation && annotation.options) ? annotation.options : [];
            const actionable = status === 'needs_manual_target' && cmid && token
                && options.length > 0 && options.every((o) => o && typeof o === 'object');
            if (!actionable) {
                const card = bubble.querySelector('.chat-routing-card');
                card?.remove();
                return renderRoutingAnnotation(bubble, annotation) || Boolean(card);
            }
            const annotationChanged = bubble.querySelector('.msg-routing-annotation')
                ? renderRoutingAnnotation(bubble, null) : false;
            let card = bubble.querySelector('.chat-routing-card');
            if (card && card.dataset.routingToken === token) return annotationChanged;
            card?.remove();
            card = buildRoutingCard(cmid, token, options);
            const time = bubble.querySelector('.msg-time');
            if (time) time.before(card);
            else bubble.append(card);
            bubble.dataset.chatAnnotationStatus = status;
            return true;
        });
    }

    function applyQuizStateFrame(rootNode, frame) {
        // Live lifecycle update for an already-rendered card (WS "quiz_state").
        // The card is found by identity, never appended: state changes must
        // not create a second card (the quiz frame dedupe is id+ts keyed).
        const quizId = String(frame && frame.quiz_id || '');
        if (!quizId || !rootNode) return false;
        const card = rootNode.querySelector(`.chat-quiz-card[data-quiz-id="${CSS.escape(quizId)}"]`);
        if (!card) return false;
        const index = Number.isInteger(frame.answered_index) ? frame.answered_index : null;
        return setCardState(card, String(frame.state || ''), index);
    }

    return { buildQuizCard, setCardState, applyQuizStateFrame, renderRoutingDecision };
}
