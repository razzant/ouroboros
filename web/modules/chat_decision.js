// Owner decision cards: the typed quiz card (question + option buttons +
// stake + assumption) and the routing picker (#198) — one decision-card
// family, one answer contract (POST /api/decisions). Optional questions let
// the task keep working under an assumption; required questions wait for an
// answer. Both read as a record after settlement. The routing picker
// settles into the plain routing ack line once its dispatch is confirmed.
import { MAX_DECISION_COMMENT, MAX_QUIZ_OPTIONS } from './api_types.js';
import { renderRoutingAnnotation, routingOptionLabel } from './chat_activity.js';
import { renderProjectChip } from './ui_helpers.js';

import { ANSWERABLE_QUIZ_STATES, QUIZ_LIFECYCLE, questionPresentation, waitFacts } from './question_presentation.js';

const WAIT_FIELDS = ['wait_for_answer', 'wait_ended_at', 'owner_wait_state', 'owner_wait_resume_reason'];
// What one observation of a question carries: its identity, lifecycle, recorded answer and wait facts.
const LIFECYCLE_FIELDS = ['task_id', 'quiz_id', 'state', 'answered_index', 'comment', ...WAIT_FIELDS];
const lifecycleOf = (row) => Object.fromEntries(LIFECYCLE_FIELDS.filter((field) => Object.hasOwn(row, field))
    .map((field) => [field, row[field]]));
// How many questions one chat instance remembers; least recently touched first.
const OBSERVATION_LIMIT = 2000;
// The signature line after a bounded wait closed says the same thing the host notice
// does (DESIGN "Quiz card"): the default path the task took, and that silence was not
// read as consent. The card stays answerable either way.
const waitEndedText = (assumption) => (assumption
    ? `The wait ended; the task continued under its assumption (${assumption}) — you can still answer.`
    : 'The wait ended without an answer; the task continued and did not take silence as consent — you can still answer.');

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
    fetchDetail = null,
    onDomWrite = (mutate) => mutate(),
    isMain = false,
    chatId = 1,
    insertMessageNode = null,
    // Main's own node retirement (viewport-stable; media, decision views and markdown released).
    removeMessageNode = null,
    focusAfterRemoval = null,
}) {
    const observations = new Map();
    const quizViews = new Map();
    const mirrors = new Map();
    const detailReads = new Map();
    const questionKey = (taskId, quizId) => JSON.stringify([String(taskId || ''), String(quizId || '')]);
    let disposed = false;
    let questionNavigation = 0;
    // The memory is bounded: the question touched least recently goes first. Once one has gone,
    // a question absent from it is no longer proven new to this tab (buildQuestionPointer).
    let forgotten = false;
    // One lifecycle observation per question, merged from every source (history rows,
    // the live quiz_state frame, a detail read). Lifecycle only moves forward: a settled
    // state never reopens, an answer is never downgraded to expiry, an unknown row
    // keeps what is known. Wait facts have one extra rule: once a LIVE frame closed the
    // wait, a snapshot that still says "waiting" (an older history row, a detail read
    // begun before the frame) cannot reopen it — the live half is the newer fact.
    function observe(frame, live = false) {
        frame = { ...frame, state: frame.state || frame.quiz_state };
        delete frame.quiz_state;
        const key = questionKey(frame.task_id, frame.quiz_id);
        const previous = observations.get(key);
        if (!frame.task_id || !frame.quiz_id) return frame;
        if (!QUIZ_LIFECYCLE.includes(frame.state)) return { ...frame, ...previous };
        if (previous && previous.state !== 'open' && frame.state === 'open') return { ...frame, ...previous };
        if (previous?.state === 'answered' && frame.state === 'expired_terminal') return { ...frame, ...previous };
        const next = { ...previous };
        for (const field of LIFECYCLE_FIELDS)
            if (Object.hasOwn(frame, field)) next[field] = frame[field];
        if (!live && previous?.live_wait && frame.state === 'open')
            for (const field of WAIT_FIELDS) {
                if (Object.hasOwn(previous, field)) next[field] = previous[field]; else delete next[field];
            }
        if (live && WAIT_FIELDS.some((field) => Object.hasOwn(frame, field))) next.live_wait = true;
        observations.delete(key);
        observations.set(key, next);
        if (observations.size > OBSERVATION_LIMIT) {
            observations.delete(observations.keys().next().value);
            forgotten = true;
        }
        return { ...frame, ...next };
    }

    // Navigation single-flights per task. A revalidation is its own read, begun after the copy it
    // validates mounted: it never joins a navigation read or a sibling copy's earlier validation,
    // either of which may predate an answer this tab missed. The copy's pending flag shares it
    // among repeated deliveries (buildQuestionPointer).
    async function readQuestion(taskId, quizId, projectId, { fresh = false } = {}) {
        if (!fetchDetail || disposed) return null;
        const read = () => Promise.resolve().then(() => disposed ? null : fetchDetail(taskId));
        if (!fresh && !detailReads.has(taskId)) {
            const promise = read().finally(() => { if (detailReads.get(taskId) === promise) detailReads.delete(taskId); });
            detailReads.set(taskId, promise);
        }
        const detail = await (fresh ? read() : detailReads.get(taskId));
        const block = detail?.owner_quiz?.[quizId];
        if (disposed || String(detail?.task_id || detail?.id || '') !== String(taskId)
            || (projectId && String(detail?.project_id || '') !== String(projectId))
            || !block || String(block.quiz_id || '') !== String(quizId)
            || !QUIZ_LIFECYCLE.includes(block.state)) return null;
        const wait = detail.owner_wait?.quiz_id === quizId ? detail.owner_wait
            : detail.owner_wait?.quiz_id && (block.wait_for_answer === true || block.wait_ended_at) ? { state: 'resumed' } : null;
        const source = { ...block, task_id: taskId, project_id: detail.project_id, ts: block.asked_at,
            ...(wait ? { owner_wait_state: wait.state || '', owner_wait_resume_reason: wait.resume_reason || '' } : {}) };
        return { ...source, ...observe(source) };
    }

    async function revealQuestion(taskId, quizId, projectId, chatId, appendQuiz, isVisible, beforeReveal = () => {}) {
        const navigation = ++questionNavigation;
        const current = () => !disposed && isVisible() && navigation === questionNavigation;
        if (!projectId || !taskId || !quizId || !current()) return false;
        let card = quizViews.get(questionKey(taskId, quizId));
        if (!card) {
            try {
                const question = await readQuestion(taskId, quizId, projectId);
                if (!current()) return false;
                if (!question) { showToast('Question unavailable.', 'error'); return false; }
                onDomWrite(() => appendQuiz({ ...question, chat_id: chatId, type: 'quiz' }));
                card = quizViews.get(questionKey(taskId, quizId));
            } catch {
                if (current()) showToast('Question unavailable.', 'error');
                return false;
            }
        }
        if (!current() || !card) return false;
        // An explicit target supersedes any pending restoration of the room's
        // earlier scroll position; the chat instance owns that restoration.
        beforeReveal();
        card.scrollIntoView?.({ block: 'center', behavior: 'auto' });
        (card.querySelector('.chat-quiz-comment') || card.querySelector('.chat-quiz-question'))?.focus?.({ preventScroll: true });
        return true;
    }

    // The Main mirror of a Project question (DESIGN "Project question mirror") is the Project's
    // own form — this module's buildQuizCard with the question, option details, recommendation,
    // stake, assumption, status and own-answer field — plus one chip that opens the question in
    // its Project. History, the live delivery and the activity census carry that form in the
    // pointer row (project_dialogue.project_question_pointer), so an unchanged row writes
    // nothing. The first confirmed answer from any source — a press here, the Project form or
    // another device (quiz_state), a history or census snapshot — shows the recorded result for
    // MIRROR_SETTLE_MS and then removes only this Main copy: one countdown, never restarted. A
    // question already answered never enters Main, and lifecycle only moves forward, so a stale
    // open snapshot cannot bring a removed copy back — while the bounded observation memory
    // holds the answer, and after it let that go, through one canonical read (revalidateMirror).
    // No new store, reader or poller.
    const MIRROR_SETTLE_MS = 5000;
    // Display fields a narrower delivery (the census, a lifecycle frame) may lack: an empty value
    // there never blanks what a complete row already carried.
    const MIRROR_FIELDS = ['question', 'options', 'option_details', 'stake', 'project_name', 'assumption', 'recommended_index'];
    const MIRROR_SIGNATURE = ['quiz_state', ...MIRROR_FIELDS, 'answered_index', 'comment', ...WAIT_FIELDS];
    const openQuestion = (row) => window.dispatchEvent(new CustomEvent('ouro:open-project', { detail: {
        project: { id: row.project_id, name: row.project_name, chat_id: row.project_chat_id },
        task_id: row.task_id, quiz_id: row.quiz_id,
    } }));
    // The pointer row in the shape of the Project's quiz row, so one normalizer reads both.
    const mirrorQuiz = (row) => ({ ...row, type: 'quiz', role: 'assistant', state: row.quiz_state });
    // An empty recorded comment and no comment are the same fact.
    const mirrorSignature = (row) => JSON.stringify(MIRROR_SIGNATURE.map((field) => (row[field] === '' ? null : row[field] ?? null)));
    // What the copy can offer: the form needs the question and its options; only a known open or
    // finished question takes an answer. Either may arrive later than the first delivery.
    const mirrorShape = (row) => {
        const complete = Boolean(row.question) && (row.options?.length || 0) >= 2;
        return { complete, answerable: complete && ANSWERABLE_QUIZ_STATES.includes(row.quiz_state) };
    };

    // A copy on screen is its question's lifecycle memory too: an observation the bounded memory
    // let go is taken back from the copy, so a stale snapshot still cannot move it backwards.
    function rememberMirror(view) {
        if (QUIZ_LIFECYCLE.includes(view.row.quiz_state) && !observations.has(view.key))
            observe({ ...lifecycleOf(view.row), state: view.row.quiz_state });
    }

    function mirrorRow(view, frame, live = false) {
        rememberMirror(view);
        const state = frame.state || frame.quiz_state;
        // After eviction, neither repeated snapshots nor a stale detail observation
        // can authorize this form. Only its fresh canonical read clears the gate;
        // a positive answer can still settle the navigation-only copy normally.
        const current = view.needsValidation && state !== 'answered'
            ? { ...frame, state: 'unknown' } : observe({ ...frame, state }, live);
        for (const field of MIRROR_FIELDS)
            if (field in current && (current[field] == null || current[field] === '' || current[field]?.length === 0)) delete current[field];
        view.row = { ...view.row, ...current, quiz_state: current.state };
        return view.row;
    }

    function mirrorChip(view) {
        const name = view.row.project_name || 'Project';
        view.chip = renderProjectChip({ name, status: '↗', className: 'chat-quiz-project', onClick: () => openQuestion(view.row) });
        view.chip.title = `Open this question in ${name}`;
        view.chip.querySelector('.chat-live-project-status')?.setAttribute('aria-hidden', 'true');
        return view.chip;
    }

    function mountMirror(view) {
        const bubble = buildQuizCard(mirrorQuiz(view.row), view);
        if (!bubble) return null;
        bubble.classList.add('project-question');
        view.bubble = bubble;
        view.shape = mirrorShape(view.row);
        view.signature = mirrorSignature(view.row);
        return bubble;
    }

    function updateMirror(view, frame, live = false) {
        mirrorRow(view, frame, live);
        const signature = mirrorSignature(view.row);
        if (view.signature === signature) return false;
        view.signature = signature;
        return onDomWrite(() => {
            if (disposed || mirrors.get(view.key) !== view) return false;
            const shape = mirrorShape(view.row);
            if ((shape.complete && !view.shape.complete) || (shape.answerable && !view.shape.answerable)) {
                // The form arrived, or a question of unknown state proved answerable: the whole
                // card mounts in place of the partial one.
                const { bubble, card } = view;
                const focused = card.contains?.(document.activeElement);
                view.disposeMarkdown?.();
                if (quizViews.get(view.key) === card) quizViews.delete(view.key);
                const next = mountMirror(view);
                if (!next) return false;
                bubble.replaceWith(next);
                if (focused) view.chip.focus?.({ preventScroll: true });
                return true;
            }
            const name = view.row.project_name || 'Project';
            const label = view.chip.querySelector('.chat-live-project-name');
            if (label && label.textContent !== name) { label.textContent = name; view.chip.title = `Open this question in ${name}`; }
            buildQuizCard(mirrorQuiz(view.row), view);
            return true;
        });
    }

    function settleMirror(view) {
        if (view.settling || mirrors.get(view.key) !== view) return;
        view.settling = true;
        view.timer = setTimeout(() => removeMirror(view), MIRROR_SETTLE_MS);
    }

    function releaseMirror(view) {
        clearTimeout(view.timer);
        view.timer = null;
        view.disposeMarkdown?.();
        view.disposeMarkdown = null;
        if (mirrors.get(view.key) === view) mirrors.delete(view.key);
        if (quizViews.get(view.key) === view.card) quizViews.delete(view.key);
    }

    function removeMirror(view) {
        if (disposed || mirrors.get(view.key) !== view) return;
        const { bubble } = view;
        const active = document.activeElement;
        const focused = bubble.contains?.(active);
        // Focus that was inside moves on to the next Main question below; the composer takes a
        // keyboard owner's focus only, so a touch owner never gets a keyboard it did not ask for.
        const others = new Map([...mirrors.values()].filter((other) => other !== view).map((other) => [other.bubble, other]));
        let next = bubble.nextElementSibling;
        while (next && !others.has(next)) next = next.nextElementSibling;
        const target = others.get(next)?.card.querySelector('.chat-quiz-question');
        let keyboard = focused;
        try { keyboard = focused && active.matches?.(':focus-visible') !== false; } catch { /* an engine without the selector */ }
        releaseMirror(view);
        if (removeMessageNode) removeMessageNode(bubble);
        else onDomWrite(() => { bubble.remove(); return true; });
        if (target && focused) target.focus?.({ preventScroll: true });
        else if (keyboard) focusAfterRemoval?.();
    }

    function buildQuestionPointer(msg) {
        if (disposed || !msg.task_id || !msg.quiz_id || !msg.project_id || !msg.project_chat_id) return null;
        const key = questionKey(msg.task_id, msg.quiz_id);
        const prior = mirrors.get(key);
        if (prior) {
            updateMirror(prior, msg);
            if (prior.needsValidation && prior.revalidationFailed && !prior.revalidationPending)
                revalidateMirror(prior);
            return null;
        }
        // An answered pointer never enters Main. Once memory has evicted anything,
        // every unmounted non-answered question needs canonical confirmation: an
        // old in-flight read can have re-seeded its stale open observation meanwhile.
        if (msg.quiz_state === 'answered' || observations.get(key)?.state === 'answered') {
            observe(msg); return null;
        }
        const view = { key, row: {}, observedAt: Date.now(), timer: null, needsValidation: forgotten,
            revalidationPending: false, revalidationFailed: false };
        mirrorRow(view, msg);
        const bubble = mountMirror(view);
        if (bubble) {
            mirrors.set(key, view);
            if (view.needsValidation) revalidateMirror(view);
        }
        return bubble;
    }

    // Mount the safe unknown copy synchronously so history owns its normal node
    // retirement and its Project chip stays accessible on a failed/foreign read.
    // No late insertion, retry poller or tombstones: only this still-owned view can
    // acquire the canonical form. Repeated stale pointers cannot unlock it.
    function revalidateMirror(view) {
        if (disposed || mirrors.get(view.key) !== view) return;
        const { task_id: taskId, quiz_id: quizId, project_id: projectId } = view.row;
        // Distinguish canonical answers merged by observe() from live confirmation during the read.
        const answeredBeforeRead = view.row.quiz_state === 'answered'
            || observations.get(view.key)?.state === 'answered';
        view.revalidationPending = true;
        view.revalidationFailed = false;
        readQuestion(taskId, quizId, projectId, { fresh: true }).catch(() => null).then((question) => {
            if (disposed || mirrors.get(view.key) !== view) return;
            view.revalidationPending = false;
            if (!question) { view.revalidationFailed = true; return; }
            // Removal is its own viewport transaction, the one that keeps the leaving node out of
            // the scroll anchor (removeMessageNode); nested in another write it would anchor there.
            if (question.state === 'answered' && !answeredBeforeRead
                && view.row.quiz_state !== 'answered') { removeMirror(view); return; }
            view.needsValidation = false;
            view.revalidationFailed = false;
            updateMirror(view, question);
        });
    }

    function appendQuestionPointer(msg) {
        if (!isMain || !insertMessageNode) return false;
        return onDomWrite(() => {
            const bubble = buildQuestionPointer(msg);
            return bubble ? insertMessageNode(bubble) !== false : false;
        });
    }

    function appendActivityQuestion(msg, requestedAt = Infinity) {
        // The census positively names the task's single wait. Mere absence proves
        // nothing. A read begun before a card arrived cannot end that newer wait.
        if (!isMain || !msg?.task_id || !msg.quiz_id
            || !['waiting', 'resumed'].includes(msg.owner_wait_state)) return false;
        // A quiz is published before its wait record. Only strictly earlier asked_at
        // proves an older wait; equal/missing stamps (including millisecond truncation)
        // cannot close a newer card. Request time alone proves no question ordering.
        const namedAt = Date.parse(msg.ts ?? '');
        return onDomWrite(() => {
            let changed = false;
            for (const view of [...mirrors.values()]) {
                const viewAt = Date.parse(view.row.ts ?? '');
                if (view.row.task_id !== msg.task_id || view.row.quiz_id === msg.quiz_id
                    || view.row.project_id !== msg.project_id || view.observedAt > requestedAt
                    || !Number.isFinite(namedAt) || !Number.isFinite(viewAt) || viewAt >= namedAt
                    || view.row.quiz_state !== 'open' || !waitFacts(view.row).waiting) continue;
                changed = updateMirror(view, { task_id: msg.task_id, quiz_id: view.row.quiz_id,
                    state: 'open', owner_wait_state: 'resumed' }, true) || changed;
            }
            return appendQuestionPointer(msg) || changed;
        });
    }

    function normalizeQuiz(msg) {
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
        const options = corrupt ? [] : normalized.slice(0, MAX_QUIZ_OPTIONS);
        return {
            quizId: String(src.quiz_id || ''),
            question: String((nested ? msg.text : src.question) || ''),
            options,
            stake: String(src.stake || ''),
            assumption: String(src.assumption || ''),
            // The wait facts the header and the signature line read (waitFacts): the
            // original required flag, the closed bound, and the task's wait record when
            // history or a detail read attached it.
            waitRow: Object.fromEntries(WAIT_FIELDS.filter((key) => Object.hasOwn(src, key)).map((key) => [key, src[key]])),
            waitForAnswer: src.wait_for_answer === true,
            state: String(src.state || 'open'),
            taskId: String(msg.task_id || ''),
            ts: msg.ts || null,
            answerFields: Object.fromEntries(['answered_index', 'comment'].filter((key) => Object.hasOwn(src, key))
                .map((key) => [key, src[key]])),
            answeredIndex: Number.isInteger(src.answered_index) ? src.answered_index : null,
            // The owner's verbatim words on a settled card (history replay
            // merges them from the projection). With no answeredIndex they
            // ARE the answer, not a remark beside one.
            comment: String(src.comment || ''),
            detailsUnavailable: src.option_details === undefined && raw.every((option) => typeof option === 'string'),
        };
    }

    function appendRecommendedBadge(button) {
        // The asker's recommendation (the "A" option) is a badge on that option, every surface alike.
        if (button.querySelector('.chat-quiz-option-recommended')) return;
        const badge = document.createElement('span');
        badge.className = 'chat-quiz-option-recommended';
        badge.textContent = 'recommended';
        button.append(badge);
    }


    async function submitAnswer(card, quiz, index, comment, settle = setCardState) {
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
                const answered = Number.isInteger(body?.answered_index) ? body.answered_index : null;
                const recorded = typeof body?.comment === 'string' ? body.comment : '';
                if (body?.ok !== true || body.state !== 'answered'
                    || (answered !== null && (answered < 0 || answered >= quiz.options.length))
                    || (answered === null && !recorded.trim())) {
                    showToast('Answer confirmation unavailable. Check the question before retrying.', 'error');
                    return;
                }
                if (recorded) card.dataset.ownerComment = recorded;
                else delete card.dataset.ownerComment;
                settle(card, 'answered', answered);
                // A late answer is recorded like any other; where it went is the
                // host's fact (`forwarded`), so the card says so instead of implying
                // the finished task will act on it.
                if (body.answered_after_terminal === true) {
                    showToast(body.forwarded === true
                        ? 'Answer recorded. The task had finished, so it was delivered to its chat as your message.'
                        : 'Answer recorded. The task had finished; nothing is waiting on it.', 'info');
                }
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
                settle(card, body.state, answered);
                showToast(body.state === 'answered'
                    ? 'Already answered.' : 'This question is no longer open.', 'error');
                return;
            }
            // A bodyless 409 no longer invents an expiry: an expired card is
            // still answerable, so the only honest thing to report is that
            // this attempt did not land. The card keeps its state.
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
        const mirror = mirrors.get(questionKey(card.dataset.taskId, card.dataset.quizId));
        const frame = { task_id: card.dataset.taskId, quiz_id: card.dataset.quizId,
            state, answered_index: answeredIndex, comment: card.dataset.ownerComment || '' };
        const current = state === 'unknown' && (mirror?.needsValidation || !mirror) ? frame : observe(frame);
        state = current.state;
        answeredIndex = Number.isInteger(current.answered_index) ? current.answered_index : null;
        if (current.comment) card.dataset.ownerComment = current.comment;
        else if (Object.hasOwn(current, 'comment')) delete card.dataset.ownerComment;
        const mirrored = mirror?.card === card;
        if (mirrored) {
            mirror.row = { ...mirror.row, ...current, quiz_state: state };
            mirror.signature = mirrorSignature(mirror.row);
        }
        const focused = mirrored && card.contains?.(document.activeElement);
        const answerable = ANSWERABLE_QUIZ_STATES.includes(state);
        const written = onDomWrite(() => {
            let changed = card.dataset.state !== state;
            if (changed) card.dataset.state = state;
            if (!answerable) {
                // A settled card takes no more input: the draft field goes,
                // and what the owner actually said takes its place.
                const box = card.querySelector('.chat-quiz-comment-box');
                if (box) { box.remove(); changed = true; }
                if (renderOwnerAnswer(card, state === 'answered' ? String(card.dataset.ownerComment || '') : '')) changed = true;
            }
            if (state !== 'open') {
                // Nothing is waiting on the owner any more — the task moved on
                // or finished — even while the card still accepts an answer.
                const waiting = card.querySelector('.chat-quiz-wait');
                if (waiting) { waiting.remove(); changed = true; }
                const ended = card.querySelector('.chat-quiz-wait-ended');
                if (ended) { ended.remove(); changed = true; }
            }
            const status = card.querySelector('.chat-quiz-status-text');
            const nextStatus = questionPresentation(current).status;
            if (status && status.textContent !== nextStatus) {
                status.textContent = nextStatus;
                changed = true;
            }
            const buttons = card.querySelectorAll('.chat-quiz-option');
            buttons.forEach((btn, i) => {
                const disabled = !answerable;
                const chosen = state === 'answered' && answeredIndex !== null && i === answeredIndex;
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
        if (mirrored) {
            // Settling disables or removes the control the owner used: focus stays in this copy
            // for the moment it still shows the result.
            const active = document.activeElement;
            if (focused && (!card.contains?.(active) || active?.disabled)) card.querySelector('.chat-quiz-question')?.focus?.({ preventScroll: true });
            if (state === 'answered') settleMirror(mirror);
        }
        return written;
    }

    function buildQuizCard(msg, mirror = null) {
        const quiz = normalizeQuiz(msg);
        // A Main mirror keeps its way to the Project even while its row cannot carry the whole
        // form yet: then it shows what is known and takes no answer (never a guessed one).
        const complete = Boolean(quiz.question) && quiz.options.length >= 2;
        if (!quiz.quizId || !quiz.taskId || !(complete || mirror)) return null;
        const key = questionKey(quiz.taskId, quiz.quizId);
        const frame = { task_id: quiz.taskId, quiz_id: quiz.quizId, state: quiz.state,
            ...quiz.waitRow, ...quiz.answerFields };
        const current = mirror?.needsValidation && quiz.state !== 'answered'
            ? { ...frame, state: 'unknown' } : observe(frame);
        quiz.state = current.state;
        quiz.answeredIndex = Number.isInteger(current.answered_index) ? current.answered_index : null;
        quiz.comment = current.comment || '';
        const wait = waitFacts(current);
        const existing = quizViews.get(key);
        if (existing) {
            if (quiz.comment) existing.dataset.ownerComment = quiz.comment;
            else if (Object.hasOwn(current, 'comment')) delete existing.dataset.ownerComment;
            if (!quiz.detailsUnavailable) {
                existing.querySelectorAll('.chat-quiz-option').forEach((button, index) => {
                    const detail = quiz.options[index]?.detail;
                    if (detail && !button.querySelector('.chat-quiz-option-detail')) {
                        const line = document.createElement('span');
                        line.className = 'chat-quiz-option-detail'; line.textContent = detail; button.append(line);
                    }
                    if (quiz.options[index]?.recommended === true) appendRecommendedBadge(button);
                });
                existing.querySelector('.chat-quiz-details-unavailable')?.remove();
            }
            if (!wait.waiting) {
                // A missed timeout frame, or a wait the owner resumed by ordinary input:
                // reconciliation projects the closed wait too.
                const waiting = existing.querySelector('.chat-quiz-wait');
                if (waiting) { waiting.textContent = waitEndedText(existing.dataset.assumption || '');
                    waiting.classList.remove('chat-quiz-wait'); waiting.classList.add('chat-quiz-wait-ended'); }
            }
            setCardState(existing, quiz.state, quiz.answeredIndex);
            return null;
        }

        const card = document.createElement('div');
        card.className = mirror ? 'chat-quiz-card project-question-card' : 'chat-quiz-card';
        card.dataset.quizId = quiz.quizId;
        card.dataset.taskId = quiz.taskId;
        if (quiz.assumption) card.dataset.assumption = quiz.assumption;
        quizViews.set(key, card);
        // The copy owns its card before the card's first settlement below: a form that arrives
        // already answered still starts the copy's countdown (setCardState -> settleMirror).
        if (mirror) mirror.card = card;

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
        // The mirror's one addition to the Project form: the chip that opens it in its Project.
        head.append(chip, ...(mirror ? [mirrorChip(mirror)] : []), status);
        card.append(head);

        // DRY with the chat surface (owner requirement): question and stake go
        // through the SAME sanitizing markdown pipeline as assistant bubbles,
        // so chat rendering improvements reach the card automatically.
        const question = document.createElement('div');
        question.className = 'chat-quiz-question';
        question.tabIndex = -1;
        const questionText = quiz.question || 'Open the original question for its text.';
        if (renderMarkdown) question.innerHTML = renderMarkdown(questionText);
        else question.textContent = questionText;
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
        (complete ? quiz.options : []).forEach((option, index) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'chat-quiz-option';
            const label = document.createElement('span');
            label.className = 'chat-quiz-option-label';
            label.textContent = String(option.label || '');
            btn.append(label);
            if (option.recommended === true) appendRecommendedBadge(btn);
            const detailText = String(option.detail || '');
            if (detailText) {
                const detail = document.createElement('span');
                detail.className = 'chat-quiz-option-detail';
                detail.textContent = detailText;
                btn.append(detail);
            }
            btn.addEventListener('click', () => {
                if (!ANSWERABLE_QUIZ_STATES.includes(card.dataset.state)) return;
                // A typed remark rides WITH the click: the owner picked this
                // option and said why, one answer, one request.
                submitAnswer(card, quiz, index, commentText());
            });
            optionsBox.append(btn);
        });
        if (complete) card.append(optionsBox);
        if (complete && quiz.detailsUnavailable) {
            const note = document.createElement('div');
            note.className = 'chat-quiz-stake chat-quiz-details-unavailable';
            note.textContent = 'Option details were not retained for this older question.';
            card.append(note);
        }

        // Free answer: none of the options may fit, and the owner must not be
        // forced to pick the least wrong one. Always visible while the card
        // still takes an answer (no disclosure to discover), removed once it
        // settles — a finished task's card is still answerable.
        if (complete && ANSWERABLE_QUIZ_STATES.includes(quiz.state)) {
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
                if (!ANSWERABLE_QUIZ_STATES.includes(card.dataset.state)) return;
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
        // A replayed row keeps only the closed bound once its required flag was dropped.
        const waitEnded = (quiz.waitForAnswer || Boolean(quiz.waitRow.wait_ended_at)) && !wait.waiting;
        if (quiz.assumption || quiz.waitForAnswer || waitEnded) {
            const assumption = document.createElement('div');
            assumption.className = 'chat-quiz-assumption';
            if (wait.waiting) assumption.classList.add('chat-quiz-wait');
            else if (waitEnded) assumption.classList.add('chat-quiz-wait-ended');
            assumption.textContent = wait.waiting
                ? 'Waiting for your answer; Stop and the task deadline still apply.'
                : (waitEnded ? waitEndedText(quiz.assumption) : `Continuing meanwhile: ${quiz.assumption}`);
            card.append(assumption);
        }

        if (quiz.comment) card.dataset.ownerComment = quiz.comment;
        setCardState(card, quiz.state, quiz.answeredIndex);
        const framed = frameNode(msg, card);
        const disposeMarkdown = enhanceMarkdown && renderMarkdown ? enhanceMarkdown(card) : null;
        if (mirror) mirror.disposeMarkdown = disposeMarkdown;
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
                    ? `Not routed: ${body.cause || body.reason || 'the destination refused this message'} — pick again.`
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
                return renderRoutingAnnotation(bubble, annotation, chatId) || Boolean(card);
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
        const taskId = String(frame && frame.task_id || '');
        if (!quizId || !taskId || !rootNode) return false;
        // The production timeout frame says only `wait_for_answer:false`: that IS the
        // resumed wait, and as a live fact it outranks any snapshot that still waits.
        if (frame.wait_for_answer === false && frame.state === 'open')
            frame = { ...frame, owner_wait_state: 'resumed' };
        const key = questionKey(taskId, quizId);
        const mirror = mirrors.get(key);
        if (mirror) rememberMirror(mirror);
        frame = observe(frame, true);
        // Observed once above as live; a Main mirror repaints from the merged observation. Only
        // the lifecycle rides along: the frame's own send time is not when the question was asked.
        if (mirror) return updateMirror(mirror, Object.fromEntries(LIFECYCLE_FIELDS
            .filter((field) => Object.hasOwn(frame, field)).map((field) => [field, frame[field]])));
        const card = quizViews.get(key);
        if (!card) return false;
        const index = Number.isInteger(frame.answered_index) ? frame.answered_index : null;
        // The owner's recorded free-text answer rides the frame (#471) so the
        // live card shows `Owner's answer:` exactly as the replayed card does.
        // Set only when present, never cleared by its absence: a later
        // lifecycle frame (expired/superseded) carries no comment.
        const comment = String(frame.comment || '');
        if (comment) card.dataset.ownerComment = comment;
        else if (Object.hasOwn(frame, 'comment')) delete card.dataset.ownerComment;
        let waitChanged = false;
        if (frame.wait_for_answer === false) {
            // The bounded wait closed and the task resumed: the card stays open and
            // answerable, but it no longer says the task is waiting.
            const waiting = card.querySelector('.chat-quiz-wait');
            if (waiting) {
                waiting.textContent = waitEndedText(card.dataset.assumption || '');
                waiting.classList.remove('chat-quiz-wait');
                waiting.classList.add('chat-quiz-wait-ended');
                waitChanged = true;
            }
        }
        return setCardState(card, String(frame.state || ''), index) || waitChanged;
    }

    // A released node takes its views with it: a mirror's countdown and rendered markdown go too.
    function releaseViews(root) {
        for (const [key, card] of quizViews) if (root.contains(card)) quizViews.delete(key);
        for (const view of [...mirrors.values()]) if (root.contains(view.bubble)) releaseMirror(view);
    }

    return { buildQuizCard, buildQuestionPointer, appendQuestionPointer, appendActivityQuestion, readQuestion, revealQuestion, setCardState, applyQuizStateFrame, renderRoutingDecision,
        releaseViews,
        destroy() {
            disposed = true;
            for (const view of [...mirrors.values()]) releaseMirror(view);
            observations.clear(); quizViews.clear(); detailReads.clear();
        },
    };
}
