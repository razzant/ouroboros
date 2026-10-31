// Typed quiz and routing picker (#198): one family, POST /api/decisions contract.
// Optional questions let work continue under an assumption; required questions
// wait. Both become records after settlement; confirmed routing becomes an ack.
import { createQuizSubmission } from './quiz_submission.js';
import { MAX_DECISION_COMMENT, MAX_QUIZ_OPTIONS } from './api_types.js';
import { renderRoutingAnnotation, routingOptionLabel } from './routing_receipt.js';
import { paintAttachmentOutcome } from './chat_attachments.js';
import { nameProjectReference, projectReference } from './project_reference.js';
import { ANSWERABLE_QUIZ_STATES, QUIZ_LIFECYCLE, questionMark, questionPresentation, waitFacts, WAIT_FIELDS,
    normalizeQuiz, appendRecommendedBadge, renderOwnerAnswer, questionDeliveryPending, unknownDeliveryRemovable,
    appendQuestionReplyControls, paintChosenAnswer } from './question_presentation.js';
import { fmt, tr, tx } from './i18n.js';
import { syncQuestionMaterials } from './result_files.js';
import { createStatusMark, paintStatusMark } from './status_mark.js';

// A question head's mark: its lifecycle's dot (question_presentation.questionMark), neutral while an answer is in
// flight (Sending…, checking it, Not confirmed: the owner has moved). The one painter of it, whoever changed the card.
const HEAD_ROWS = new WeakMap();
function paintQuestionHead(card, row = HEAD_ROWS.get(card)) {
    if (!card) return false;
    if (row) HEAD_ROWS.set(card, row);
    const inFlight = ['sending', 'unknown', 'checking'].includes(card.dataset.submission);
    return paintStatusMark(card.querySelector('.chat-quiz-mark'),
        inFlight ? { shape: 'dot', hue: 'neutral' } : questionMark(row || {}));
}

// What one observation of a question carries: its identity, lifecycle, recorded answer and wait facts.
const LIFECYCLE_FIELDS = ['task_id', 'quiz_id', 'state', 'answered_index', 'comment', 'closed_reason', 'closed_at', 'closed_actor', 'closed_request_id', 'question_delivery', ...WAIT_FIELDS];
const lifecycleOf = (row) => Object.fromEntries(LIFECYCLE_FIELDS.filter((field) => Object.hasOwn(row, field))
    .map((field) => [field, row[field]]));
// How many questions one chat instance remembers; least recently touched first.
const OBSERVATION_LIMIT = 2000;
// The signature line after a bounded wait closed says the same thing the host notice
// does (DESIGN "Quiz card"): the default path the task took, and that silence was not
// read as consent. The card stays answerable either way.
const waitEndedText = (assumption) => (assumption
    ? fmt('The wait ended; the task continued under its assumption ({assumption}) — you can still answer.', { assumption })
    : tr('quiz.wait_ended_no_answer', 'The wait ended without an answer; the task continued and did not take silence as consent — you can still answer.'));

// Neutral, factual statuses (owner decision 15~A): the card never scolds the
// router — it states what the click does and what happened.
const ROUTING_STATUS_TEXT = {
    open: 'Choose a destination',
    pending: 'Routing…',
    answered: 'Routed',
    superseded: 'Superseded by a newer attempt',
};
// Its head's dot, the question's family: the owner's move while it waits for a destination, ok once routed, the head's
// disabled ink once superseded, neutral while the chosen route is sent.
const ROUTING_MARK_HUES = { open: 'owner', answered: 'ok', superseded: 'ink' };
const ROUTING_TOP_OPTIONS = 8;

export function createChatDecision({
    apiFetch,
    frameNode,
    renderMaterial = null,
    releaseMaterials = () => {},
    mountMarkdown,
    enhanceMarkdown,
    showToast,
    fetchDetail = null,
    onDomWrite = (mutate) => mutate(),
    isMain = false,
    chatId = 1,
    projectId = '',
    insertMessageNode = null,
    // Main's own node retirement (viewport-stable; media, decision views and markdown released).
    removeMessageNode = null,
    focusAfterRemoval = null,
    // Main's waiting navigator recounts what is on screen when a question copy comes or goes.
    onQuestionsChanged = () => {},
}) {
    const observations = new Map();
    const quizViews = new Map();
    const mirrors = new Map();
    const detailReads = new Map();
    const questionKey = (taskId, quizId) => JSON.stringify([String(taskId || ''), String(quizId || '')]);
    let disposed = false;
    let questionNavigation = 0;
    let connected = true, epoch = 0;  // the socket's state and how many closes voided reads (revalidateMirror)
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
        if (previous?.state === 'closed' && ['open', 'expired_terminal'].includes(frame.state)) return { ...frame, ...previous };
        const next = { ...previous };
        for (const field of LIFECYCLE_FIELDS)
            if (Object.hasOwn(frame, field)) next[field] = frame[field];
        if (!live && previous?.live_wait && frame.state === 'open')
            for (const field of WAIT_FIELDS) {
                if (Object.hasOwn(previous, field)) next[field] = previous[field]; else delete next[field];
            }
        if ((previous?.question_delivery?.state === 'delivered' || previous?.question_delivery?.dismissed === true)
            && next.question_delivery?.kind === previous.question_delivery.kind
            && next.question_delivery?.state !== 'delivered') next.question_delivery = previous.question_delivery;
        if (live && WAIT_FIELDS.some((field) => Object.hasOwn(frame, field))) next.live_wait = true;
        observations.delete(key);
        observations.set(key, next);
        if (observations.size > OBSERVATION_LIMIT) {
            observations.delete(observations.keys().next().value);
            forgotten = true;
        }
        return { ...frame, ...next };
    }

    // What this tab has seen of a question's lifecycle: Main's waiting pill drops one that settled
    // here at once instead of waiting for the next census.
    function observedQuestion(taskId, quizId) {
        const known = observations.get(questionKey(taskId, quizId));
        return known ? { state: known.state, question_delivery: known.question_delivery } : null;
    }

    // The detail projects the shared room resolver for this exact quiz. Stored task context
    // and publication/reply addresses do not prove where its question is displayed now.
    const questionInRoom = (room, projectId) => room?.state === 'known' && Number.isInteger(room.chat_id)
        && (projectId ? room.project_id === String(projectId)
            : isMain && room.project_id === '' && room.chat_id === Number(chatId));

    // Navigation single-flights per task. A revalidation is its own read, begun after the copy it
    // validates mounted: it never joins a navigation read or a sibling copy's earlier validation,
    // either of which may predate an answer this tab missed. The copy's pending flag shares it
    // among repeated deliveries (buildQuestionPointer).
    async function readQuestion(taskId, quizId, projectId, { fresh = false } = {}) {
        if (!fetchDetail || disposed || (isMain && !connected)) return null;
        const begun = epoch;
        const read = () => Promise.resolve().then(() => disposed ? null : fetchDetail(taskId));
        if (!fresh && !detailReads.has(taskId)) {
            const promise = read().finally(() => { if (detailReads.get(taskId) === promise) detailReads.delete(taskId); });
            detailReads.set(taskId, promise);
        }
        const detail = await (fresh ? read() : detailReads.get(taskId));
        const block = detail?.owner_quiz?.[quizId];
        const room = detail?.question_rooms?.[quizId];
        if (disposed || (isMain && (!connected || begun !== epoch)) || String(detail?.task_id || detail?.id || '') !== String(taskId)
            || !questionInRoom(room, projectId)
            || !block || String(block.quiz_id || '') !== String(quizId)
            || !QUIZ_LIFECYCLE.includes(block.state)) return null;
        const wait = detail.owner_wait?.quiz_id === quizId ? detail.owner_wait
            : detail.owner_wait?.quiz_id && (block.wait_for_answer === true || block.wait_ended_at) ? { state: 'resumed' } : null;
        // The detail carries the canonical record; the projection's shape names the owner's removal.
        const delivery = block.delivery && { ...block.delivery, ...(block.delivery.dismissed_at ? { dismissed: true } : {}) };
        const source = { ...block, ...(delivery ? { question_delivery: delivery } : {}), task_id: taskId, project_id: room.project_id, ts: block.asked_at,
            ...(wait ? { owner_wait_state: wait.state || '', owner_wait_resume_reason: wait.resume_reason || '' } : {}) };
        const question = { ...source, ...observe(source) };
        const card = quizViews.get(questionKey(taskId, quizId));
        if (isMain && card && !mirrors.has(questionKey(taskId, quizId))) {
            delete card.dataset.authorityStale;
            buildQuizCard(question);
        }
        return question;
    }

    // `focus: 'question'` (the room's work list) lands on the question itself, never its answer field,
    // so going to a question summons no keyboard and invites no typing before it is read.
    async function revealQuestion(taskId, quizId, projectId, chatId, appendQuiz, isVisible, beginReveal = () => {},
        didReveal = () => {}, { focus = 'answer' } = {}) {
        const begun = epoch;
        const navigation = ++questionNavigation; // one naming no question still voids older reveals
        let ownsPosition = () => true;
        const current = () => !disposed && (!isMain || (connected && begun === epoch)) && isVisible() && navigation === questionNavigation && ownsPosition();
        // Main's own question names no Project: readQuestion proves its room instead.
        if (!(projectId || isMain) || !taskId || !quizId || !current()) return false;
        // Navigation supersedes the room bookmark before I/O, even if the
        // question is unavailable. A later wheel/latest intent can supersede us.
        ownsPosition = beginReveal() || ownsPosition;
        let card = quizViews.get(questionKey(taskId, quizId));
        if (!card || card.dataset.authorityStale === '1') {
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
        card.scrollIntoView?.({ block: 'center', behavior: 'auto' });
        ((focus === 'question' ? null : card.querySelector('.chat-quiz-comment'))
            || card.querySelector('.chat-quiz-question'))?.focus?.({ preventScroll: true });
        didReveal();
        return true;
    }

    // The Main mirror of a Project question (DESIGN "Project question mirror") is the Project's
    // own form — this module's buildQuizCard with the question, option details, recommendation,
    // stake, assumption, status and own-answer field — plus one chip that opens the question in
    // its Project. History, the live delivery and the activity census carry that form in the
    // pointer row (project_dialogue.project_question_pointer), so an unchanged row writes
    // nothing. The first confirmed answer from any source — a press here, the Project form or
    // another device (quiz_state), a history or census snapshot — shows the recorded result for
    // MIRROR_SETTLE_MS and then removes only this Main copy after recorded delivery is complete.
    // A census that stops naming a copy proves no outcome: every unsettled copy but a replaced record
    // stays the same node until its own canonical read decides (revalidateMirror).
    // Undischarged answers/closures stay available; lifecycle only moves forward, so a stale
    // open snapshot cannot bring a removed copy back — while the bounded observation memory
    // holds the answer, and after it let that go, through one canonical read (revalidateMirror).
    // No new store, reader or poller.
    const MIRROR_SETTLE_MS = 5000;
    // Display fields a narrower delivery (the census, a lifecycle frame) may lack: an empty value
    // there never blanks what a complete row already carried.
    const MIRROR_FIELDS = ['question', 'options', 'option_details', 'stake', 'project_name', 'assumption', 'recommended_index', 'materials'];
    const MIRROR_SIGNATURE = ['quiz_state', ...MIRROR_FIELDS, 'answered_index', 'comment', 'closed_reason', 'question_delivery', ...WAIT_FIELDS];
    // The pointer row in the shape of the Project's quiz row, so one normalizer reads both.
    const mirrorQuiz = (row) => ({ ...row, type: 'quiz', role: 'assistant', state: row.quiz_state });
    // An empty recorded comment and no comment are the same fact.
    const mirrorSignature = (row) => JSON.stringify(MIRROR_SIGNATURE.map((field) => (row[field] === '' ? null : row[field] ?? null)));
    // What the copy can offer: the form needs the question and its options; only a known open or
    // finished question takes an answer. Either may arrive later than the first delivery.
    const mirrorShape = (row) => {
        const complete = Boolean(row.question) && Array.isArray(row.options) && row.options.length <= MAX_QUIZ_OPTIONS;
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
        const current = view.needsValidation && !['answered', 'closed', 'superseded'].includes(state)
            ? { ...frame, state: 'unknown' } : observe({ ...frame, state }, live);
        for (const field of MIRROR_FIELDS)
            if (field in current && (current[field] == null || current[field] === ''
                || (current[field]?.length === 0
                    && (field !== 'options' || view.row.options?.length > 0)))) delete current[field];
        view.row = { ...view.row, ...current, quiz_state: current.state };
        return view.row;
    }

    const mirrorProject = (row) => ({ id: row.project_id, name: row.project_name, chat_id: row.project_chat_id });
    function mirrorChip(view) {
        view.chip = projectReference(mirrorProject(view.row), { taskId: view.row.task_id, quizId: view.row.quiz_id });
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
        if (live) view.observedAt = Date.now();
        mirrorRow(view, frame, live);
        const signature = mirrorSignature(view.row);
        if (view.signature === signature) return false;
        view.signature = signature;
        return onDomWrite(() => {
            if (disposed || mirrors.get(view.key) !== view) return false;
            const shape = mirrorShape(view.row);
            if ((shape.complete && !view.shape.complete) || (shape.answerable && !view.shape.answerable && !view.card.querySelector('.chat-quiz-comment'))) {
                // The form arrived, or a question of unknown state proved answerable: the whole
                // card mounts in place of the partial one.
                const { bubble, card } = view;
                const focused = card.contains?.(document.activeElement);
                view.disposeMarkdown?.();
                releaseMaterials(bubble);
                submissions.release(card);
                if (quizViews.get(view.key) === card) quizViews.delete(view.key);
                const next = mountMirror(view);
                if (!next) return false;
                bubble.replaceWith(next);
                if (focused) view.chip.focus?.({ preventScroll: true });
                return true;
            }
            nameProjectReference(view.chip, mirrorProject(view.row));
            buildQuizCard(mirrorQuiz(view.row), view);
            return true;
        });
    }

    function settleMirror(view) {
        if (questionDeliveryPending(view.row)) {
            clearTimeout(view.timer); view.timer = null; view.settling = false; return;
        }
        if (view.settling || mirrors.get(view.key) !== view) return;
        view.settling = true;
        view.timer = setTimeout(() => removeMirror(view), MIRROR_SETTLE_MS);
    }

    function releaseMirror(view) {
        clearTimeout(view.timer);
        view.timer = null;
        view.disposeMarkdown?.();
        if (view.card) submissions.release(view.card);
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
        onQuestionsChanged();
        if (target && focused) target.focus?.({ preventScroll: true });
        else if (keyboard) focusAfterRemoval?.();
    }

    function buildQuestionPointer(msg, canonical = false, stale = false) {
        if (disposed || !msg.task_id || !msg.quiz_id || !msg.project_id || !msg.project_chat_id) return null;
        const key = questionKey(msg.task_id, msg.quiz_id);
        const prior = mirrors.get(key);
        if (prior) {
            if (canonical) { prior.needsValidation = false; prior.offered = true; delete prior.card.dataset.authorityStale; }
            updateMirror(prior, msg);
            if (prior.needsValidation && prior.revalidationFailed && !prior.revalidationPending)
                revalidateMirror(prior);
            return null;
        }
        // A settled (answered or closed) pointer never enters Main unless its delivery is still
        // owed. Once memory has evicted anything, every unmounted unsettled question needs
        // canonical confirmation: an old in-flight read can have re-seeded its stale open
        // observation meanwhile.
        // Only a receipt for this same delivery settles it: a publication receipt says nothing of a later answer.
        const known = observations.get(key)?.question_delivery;
        const owed = questionDeliveryPending(msg) && !(known?.kind === msg.question_delivery?.kind
            && (known.state === 'delivered' || known.dismissed === true));
        if (!owed && (['answered', 'closed'].includes(msg.quiz_state) || ['answered', 'closed'].includes(observations.get(key)?.state))) {
            observe(msg); return null;
        }
        const needsValidation = stale || (!canonical && forgotten);
        // `offered`: the owner has had this copy's live form. One mounted withheld has not, until proven.
        const view = { key, row: {}, observedAt: Date.now(), timer: null, needsValidation, offered: !needsValidation,
            revalidationPending: false, revalidationFailed: false };
        mirrorRow(view, msg);
        const bubble = mountMirror(view);
        if (bubble) {
            mirrors.set(key, view);
            if (view.needsValidation) revalidateMirror(view);
            if (['answered', 'closed'].includes(view.row.quiz_state)) settleMirror(view);
        }
        return bubble;
    }

    // Mount the safe unknown copy synchronously so history owns its normal node
    // retirement and its Project chip stays accessible on a failed/foreign read.
    // No late insertion, retry poller or tombstones: only this still-owned view can
    // acquire the canonical form. Repeated stale pointers cannot unlock it, nor can a closed
    // socket: a close withdraws every form (setConnected) and loses frames sent after any read
    // taken before the reopen. So nothing reads while closed, a read begun before a close applies
    // nothing even after the reopen, and only the census or a later read restores the form.
    function revalidateMirror(view) {
        if (disposed || mirrors.get(view.key) !== view) return;
        if (!connected) { view.revalidationFailed = true; return; }  // the first pointer after the reopen reads
        const { task_id: taskId, quiz_id: quizId, project_id: projectId } = view.row;
        // Distinguish canonical answers merged by observe() from live confirmation during the read.
        const answeredBeforeRead = view.row.quiz_state === 'answered'
            || observations.get(view.key)?.state === 'answered';
        const begun = epoch;
        view.revalidationPending = true;
        view.revalidationFailed = false;
        readQuestion(taskId, quizId, projectId, { fresh: true }).catch(() => null).then((question) => {
            if (disposed || mirrors.get(view.key) !== view || begun !== epoch) return;
            view.revalidationPending = false;
            if (!question) {
                view.revalidationFailed = true;
                // A live open copy left unproven keeps its words, not its controls (DESIGN: Status unavailable).
                // An answer or closure confirmed meanwhile, or still owed, keeps its record and interval.
                if (!view.needsValidation && !view.settling && ANSWERABLE_QUIZ_STATES.includes(view.row.quiz_state)) {
                    view.needsValidation = true;
                    updateMirror(view, { task_id: view.row.task_id, quiz_id: view.row.quiz_id, state: 'unknown' });
                }
                return;
            }
            // Removal is its own viewport transaction, the one that keeps the leaving node out of
            // the scroll anchor (removeMessageNode); nested in another write it would anchor there.
            // Only a copy never offered leaves at once; an offered one shows its outcome for the interval.
            if (question.state === 'answered' && !questionDeliveryPending(question) && !answeredBeforeRead
                && view.row.quiz_state !== 'answered' && !view.offered) { removeMirror(view); return; }
            view.needsValidation = false;
            view.offered = true;
            delete view.card.dataset.authorityStale;
            view.revalidationFailed = false;
            updateMirror(view, question);
        });
    }

    // A Project question's Main copy takes its own place in the feed, at the time it was asked.
    // `canonical`: the row is the current census's; `stale`: an older read's, so the copy mounts
    // withheld and takes its form only from its own canonical read.
    function appendQuestionPointer(msg, { canonical = false, stale = false } = {}) {
        if (!isMain || !insertMessageNode) return false;
        return onDomWrite(() => {
            const bubble = buildQuestionPointer(msg, canonical, stale);
            if (!bubble) return false;
            const inserted = insertMessageNode(bubble) !== false;
            onQuestionsChanged();
            return inserted;
        });
    }

    // The canonical current questions (`/api/state`) never mount copies: history and the live
    // frame place questions in the feed. The census only confirms or settles copies already
    // mounted, and reconfirms cards a socket close made stale. `complete` is a boolean or the
    // read's `(taskId, quizId) => vouched` predicate: an unnamed copy it vouches for is no longer
    // current, which says nothing of its outcome, so it waits for its own read (revalidateMirror).
    function syncCurrentQuestions(rows, complete = false, requestedAt = Infinity) {
        if (!isMain || !Array.isArray(rows)) return;
        const vouched = typeof complete === 'function' ? complete : () => complete === true;
        onDomWrite(() => {
            for (const [key, card] of quizViews) {
                if (!connected || mirrors.has(key) || card.dataset.authorityStale !== '1' || card.dataset.validationPending === String(epoch)) continue;
                const known = observations.get(key);
                if (known && !ANSWERABLE_QUIZ_STATES.includes(known.state) && known.state !== 'unknown'
                    && !questionDeliveryPending(known)) continue; // settled native history has no authority to restore
                const begun = String(epoch);
                card.dataset.validationPending = begun;
                readQuestion(card.dataset.taskId, card.dataset.quizId, '', { fresh: true })
                    .catch(() => null).finally(() => {
                        if (card.dataset.validationPending === begun) delete card.dataset.validationPending;
                    });
            }
            const named = new Set();
            for (const row of rows) {
                const key = questionKey(row.task_id, row.quiz_id);
                named.add(key);
                if (mirrors.has(key)) buildQuestionPointer(row, true);
            }
            for (const view of [...mirrors.values()]) {
                if (named.has(view.key) || view.observedAt > requestedAt || view.settling) continue;
                if (vouched(view.row.task_id, view.row.quiz_id)) {
                    // One exact read at a time (void across a socket close) paints what became of the
                    // copy through the settle interval; a missing, failed, foreign or owed read keeps it,
                    // and the next such census reads again: a copy no read can prove costs one read per
                    // census while it stays shown. Only a replaced record leaves with absence.
                    if (view.row.quiz_state === 'superseded') removeMirror(view);
                    else if (!view.revalidationPending) revalidateMirror(view);
                } else {
                    view.needsValidation = true;
                    view.revalidationFailed = true;
                    updateMirror(view, { task_id: view.row.task_id, quiz_id: view.row.quiz_id, state: 'unknown' });
                }
            }
            onQuestionsChanged();
            return true;
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

    const submissions = createQuizSubmission({ apiFetch, onDomWrite, showToast, paintHead: paintQuestionHead,
        canUse: card => !disposed && (!isMain || (connected && card.dataset.authorityStale !== '1')),
        readQuestion: quiz => readQuestion(quiz.taskId, quiz.quizId, quiz.projectId, { fresh: true }),
        accept(card, body) {
            if (body) {
                const current = observe({ ...body, task_id: card.dataset.taskId, quiz_id: card.dataset.quizId });
                if (Object.hasOwn(current, 'comment')) card.dataset.ownerComment = current.comment || '';
                setCardState(card, current.state, current.answered_index);
            } else setCardState(card, card.dataset.state, null);
        },
    });
    function submitAnswer(card, quiz, index, comment, _settle = setCardState, action = 'answer') {
        if (action !== 'answer') {
            if (card.dataset.submission) return;
            return submitOtherQuizAction(card, quiz, index, comment, _settle, action);
        }
        submissions.submit(card, quiz, index, String(comment || ''));
    }

    async function submitOtherQuizAction(card, quiz, index, comment, settle = setCardState, action = 'answer') {
        if (card.dataset.pending === '1' || (isMain && (!connected || card.dataset.authorityStale === '1'))) return;
        card.dataset.pending = '1';
        const text = String(comment || '');
        // STABLE per-card idempotency key: a retry after a transient failure
        // must replay the SAME request, or the server-side first-wins latch
        // reads the retry as a competing second answer.
        if (!card.dataset.requestId || card.dataset.requestAction !== action) {
            card.dataset.requestId = (crypto.randomUUID && crypto.randomUUID()) || `q-${Date.now()}`;
            card.dataset.requestAction = action;
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
                    ...(action === 'close' ? { action, reason: text } : action === 'dismiss' ? { action }
                        : text ? { comment: text } : {}),
                }),
            });
            let body = null;
            try { body = res && res.json ? await res.json() : null; } catch (parseErr) { body = null; }
            if (res && res.ok) {
                if (action === 'dismiss') {
                    if (body?.ok !== true || body.state !== 'answered' || body.question_delivery?.dismissed !== true) {
                        showToast('Removal confirmation unavailable. Check the question before retrying.', 'error'); return;
                    }
                    observe({ ...body, task_id: quiz.taskId, quiz_id: quiz.quizId });
                    settle(card, 'answered', Number.isInteger(body.answered_index) ? body.answered_index : null); return;
                }
                if (action === 'close') {
                    if (body?.ok !== true || body.state !== 'closed' || !String(body.closed_reason || '').trim()) {
                        showToast('Closure confirmation unavailable. Check the question before retrying.', 'error'); return;
                    }
                    observe({ ...body, task_id: quiz.taskId, quiz_id: quiz.quizId });
                    settle(card, 'closed', null); return;
                }
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
                observe({ ...body, task_id: quiz.taskId, quiz_id: quiz.quizId });
                settle(card, body.state, answered);
                showToast(action === 'dismiss' ? 'This answer no longer waits on an unknown delivery.'
                    : body.state === 'answered' ? 'Already answered.' : 'This question is no longer open.', 'error');
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

    function setCardState(card, state, answeredIndex) {
        if (!card) return false;
        const mirror = mirrors.get(questionKey(card.dataset.taskId, card.dataset.quizId));
        const frame = { task_id: card.dataset.taskId, quiz_id: card.dataset.quizId,
            state, answered_index: answeredIndex, comment: card.dataset.ownerComment || '' };
        const current = state === 'unknown' && (mirror?.needsValidation || !mirror) ? frame : observe(frame);
        state = isMain && (!connected || card.dataset.authorityStale === '1')
            && ANSWERABLE_QUIZ_STATES.includes(current.state) ? 'unknown' : current.state;
        answeredIndex = Number.isInteger(current.answered_index) ? current.answered_index : null;
        if (current.comment) card.dataset.ownerComment = current.comment;
        else if (Object.hasOwn(current, 'comment')) delete card.dataset.ownerComment;
        const mirrored = mirror?.card === card;
        if (mirrored) {
            mirror.row = { ...mirror.row, ...current, quiz_state: state };
            mirror.signature = mirrorSignature(mirror.row);
        }
        const focused = card.contains?.(document.activeElement);
        const answerable = ANSWERABLE_QUIZ_STATES.includes(state);
        const written = onDomWrite(() => {
            let changed = card.dataset.state !== state;
            if (changed) card.dataset.state = state;
            if (!answerable && state !== 'unknown') {
                // A settled card takes no more input: the draft field goes,
                // and what the owner actually said takes its place.
                const box = card.querySelector('.chat-quiz-comment-box');
                if (box) { box.remove(); changed = true; }
                if (renderOwnerAnswer(card, state === 'answered' ? String(card.dataset.ownerComment || '') : '')) changed = true;
            }
            const comment = card.querySelector('.chat-quiz-comment');
            const send = card.querySelector('.chat-quiz-send');
            const closeButton = card.querySelector('.chat-quiz-close');
            if (closeButton && !answerable && state !== 'unknown') { closeButton.remove(); changed = true; }
            else if (closeButton) closeButton.disabled = !answerable || Boolean(String(comment?.value || '').trim());
            if (comment) comment.disabled = !answerable;
            if (send && comment) send.disabled = !answerable || !String(comment.value || '').trim()
                || String(comment.value || '').length > MAX_DECISION_COMMENT;
            const reason = current.closed_reason || card.dataset.closedReason || '';
            if (state === 'closed' && reason) {
                let line = card.querySelector('.chat-quiz-closed-reason');
                if (!line) { line = document.createElement('div'); line.className = 'chat-quiz-closed-reason msg-provenance'; card.append(line); }
                if (line.textContent !== reason) { line.textContent = reason; changed = true; }
            }
            if (state !== 'open' && state !== 'unknown') {
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
            if (paintQuestionHead(card, current)) changed = true;
            // An accepted late answer whose delivery became unknown: Main's copy lets the owner
            // stop counting it as pending. Nothing is resent; the answer and its record stay.
            const removable = card.classList.contains('project-question-card') && unknownDeliveryRemovable({ ...current, state });
            let remove = card.querySelector('.chat-quiz-dismiss');
            if (removable && !remove) {
                remove = document.createElement('button');
                remove.type = 'button';
                remove.className = 'chat-quiz-dismiss btn btn-xs btn-default';
                remove.textContent = tr('quiz.remove_unknown', 'Remove from pending');
                remove.addEventListener('click', () => submitAnswer(card,
                    { taskId: card.dataset.taskId, quizId: card.dataset.quizId, options: [] }, null, '', setCardState, 'dismiss'));
                card.append(remove);
                changed = true;
            } else if (!removable && remove) {
                remove.remove();
                changed = true;
            }
            if (remove) remove.disabled = isMain && (!connected || card.dataset.authorityStale === '1');
            const buttons = card.querySelectorAll('.chat-quiz-option');
            buttons.forEach((btn, i) => {
                const chosen = state === 'answered' && answeredIndex !== null && i === answeredIndex;
                if (btn.disabled !== !answerable) {
                    btn.disabled = !answerable;
                    changed = true;
                }
                if (btn.classList.contains('chosen') !== chosen) {
                    btn.classList.toggle('chosen', chosen);
                    changed = true;
                }
                if (paintChosenAnswer(btn, chosen)) changed = true;
            });
            return changed;
        });
        submissions.settled(card);
        // Settling disables or removes the control the owner used (an option, the ×): focus stays in
        // this card, on its question, rather than falling to the page.
        const active = document.activeElement;
        if (focused && (!card.contains?.(active) || active?.disabled)) card.querySelector('.chat-quiz-question')?.focus?.({ preventScroll: true });
        if (mirrored && (state === 'answered' || state === 'closed')) settleMirror(mirror);
        if (written && isMain && !answerable && state !== 'unknown') onQuestionsChanged();
        return written;
    }

    function buildQuizCard(msg, mirror = null) {
        const quiz = normalizeQuiz(msg);
        quiz.projectId = String(msg.project_id || mirror?.row.project_id || projectId || '');
        // A Main mirror keeps its way to the Project even while its row cannot carry the whole
        // form yet: then it shows what is known and takes no answer (never a guessed one).
        const complete = Boolean(quiz.question) && quiz.optionsKnown;
        if (!quiz.quizId || !quiz.taskId || !(complete || mirror)) return null;
        const key = questionKey(quiz.taskId, quiz.quizId);
        const frame = { task_id: quiz.taskId, quiz_id: quiz.quizId, state: quiz.state,
            ...quiz.waitRow, ...quiz.answerFields, ...(quiz.closedReason ? { closed_reason: quiz.closedReason } : {}) };
        const current = mirror?.needsValidation && !['answered', 'closed', 'superseded'].includes(quiz.state)
            ? { ...frame, state: 'unknown' } : observe(frame);
        quiz.state = current.state;
        quiz.answeredIndex = Number.isInteger(current.answered_index) ? current.answered_index : null;
        quiz.comment = current.comment || '';
        const wait = waitFacts(current);
        const existing = quizViews.get(key);
        if (existing) {
            if (quiz.comment) existing.dataset.ownerComment = quiz.comment;
            else if (Object.hasOwn(current, 'comment')) delete existing.dataset.ownerComment;
            if (quiz.closedReason) existing.dataset.closedReason = quiz.closedReason;
            syncQuestionMaterials(existing, quiz.materials, renderMaterial);
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
        if (isMain && !connected) card.dataset.authorityStale = '1';
        if (quiz.assumption) card.dataset.assumption = quiz.assumption;
        if (quiz.closedReason) card.dataset.closedReason = quiz.closedReason;
        quizViews.set(key, card);
        // The copy owns its card before the card's first settlement below: a form that arrives
        // already answered still starts the copy's countdown (setCardState -> settleMirror).
        if (mirror) mirror.card = card;

        const head = document.createElement('div');
        head.className = 'chat-quiz-head';
        const chip = document.createElement('span');
        chip.className = 'chat-quiz-chip';
        chip.textContent = tr('quiz.chip_question', 'Question');
        const status = document.createElement('span');
        status.className = 'chat-quiz-status';
        const statusLabel = document.createElement('span');
        statusLabel.className = 'chat-quiz-status-text';
        status.append(createStatusMark(document, 'chat-quiz-mark'), statusLabel);
        // One row in every room: the status describes the question, so it sits beside the chip, and the card's
        // quiet Close alone ends the row. Where the question comes from is the bubble's voice line (frameNode).
        const headEnd = document.createElement('span');
        headEnd.className = 'chat-quiz-head-end';
        head.append(chip, status, headEnd);
        card.append(head);

        // DRY with the chat surface (owner requirement): question and stake go
        // through the SAME sanitizing markdown pipeline as assistant bubbles,
        // so chat rendering improvements reach the card automatically.
        const question = document.createElement('div');
        question.className = 'chat-quiz-question';
        question.tabIndex = -1;
        const questionText = quiz.question || 'Open the original question for its text.';
        if (mountMarkdown) mountMarkdown(question, questionText);
        else question.textContent = questionText;
        card.append(question);
        syncQuestionMaterials(card, quiz.materials, renderMaterial);

        if (quiz.stake) {
            const stake = document.createElement('div');
            stake.className = 'chat-quiz-stake';
            const stakeText = `${tr('quiz.at_stake', 'At stake:')} ${quiz.stake}`;
            if (mountMarkdown) mountMarkdown(stake, stakeText);
            else stake.textContent = stakeText;
            card.append(stake);
        }

        let commentField = null;
        // The raw field value is the answer (VERBATIM to the model); the
        // trimmed view only decides whether there IS one.
        const commentText = () => String((commentField && commentField.value) || '');

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
        if (complete && quiz.options.length) card.append(optionsBox);
        if (complete && quiz.options.length && quiz.detailsUnavailable) {
            const note = document.createElement('div');
            note.className = 'chat-quiz-stake chat-quiz-details-unavailable';
            note.textContent = tr('quiz.details_not_retained', 'Option details were not retained for this older question.');
            card.append(note);
        }

        if (complete && ANSWERABLE_QUIZ_STATES.includes(quiz.state))
            commentField = appendQuestionReplyControls(card, quiz, { showToast, onDomWrite,
                submit: text => submitAnswer(card, quiz, null, text),
                close: reason => submitAnswer(card, quiz, null, reason, setCardState, 'close') });

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
                ? tr('quiz.waiting_clause', 'Waiting for your answer; Stop and the task deadline still apply.')
                : (waitEnded ? waitEndedText(quiz.assumption) : fmt('Continuing meanwhile: {assumption}', { assumption: quiz.assumption }));
            card.append(assumption);
        }

        if (quiz.comment) card.dataset.ownerComment = quiz.comment;
        setCardState(card, quiz.state, quiz.answeredIndex);
        if (complete) submissions.attach(card, quiz);
        // The bubble says who asks and when: a Main copy's Project reference joins its voice line beside
        // `Ouroboros` (the mirror's one addition to the Project form), and its time line is the time alone.
        const framed = frameNode(msg, card, { voice: mirror ? [mirrorChip(mirror)] : [] });
        const disposeMarkdown = enhanceMarkdown && mountMarkdown ? enhanceMarkdown(card) : null;
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
            const view = { shape: 'dot', hue: ROUTING_MARK_HUES[state] || 'neutral' };
            if (paintStatusMark(card.querySelector('.chat-quiz-mark'), view)) changed = true;
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
                if (paintChosenAnswer(btn, chosen && state === 'answered')) changed = true;
            });
            return changed;
        });
    }

    async function submitRouting(card, cmid, token, index) {
        if (card.dataset.pending === '1' || (isMain && (!connected || card.dataset.authorityStale === '1'))) return;
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
                const cause = tx(body.cause || body.reason || tr('routing.refused_generic', 'the destination refused this message'));
                showToast(body.state === 'open' ? fmt('Not routed: {cause} — pick again.', { cause })
                    : body.state === 'pending' ? tr('routing.another_choice_in_flight', 'Another choice is already being routed.')
                        : tr('routing.already_routed', 'This message was already routed.'), 'error');
                return;
            }
            showToast(fmt('Could not route the message ({status}) — try again.', { status: status || tr('routing.network_error', 'network error') }), 'error');
        } catch (err) {
            showToast(fmt('Could not route the message ({status}) — try again.', { status: tr('routing.network_error', 'network error') }), 'error');
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
        chip.textContent = tr('routing.chip_route', 'Route');
        const status = document.createElement('span');
        status.className = 'chat-quiz-status';
        const statusLabel = document.createElement('span');
        statusLabel.className = 'chat-quiz-status-text';
        status.append(createStatusMark(document, 'chat-quiz-mark'), statusLabel);
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
            label.textContent = routingOptionLabel(option) || fmt('Option {n}', { n: index + 1 });
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
            more.textContent = fmt('Show all {n}', { n: options.length });
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
        // ONE entry point for a user bubble's routing surface: the picker card for an actionable refusal, else the
        // plain ack line; a file the host could not attach keeps its own never-folded note beside either.
        if (!bubble) return false;
        return onDomWrite(() => {
            const outcome = paintAttachmentOutcome(bubble, annotation);
            const cmid = String(bubble.dataset.clientMessageId || '');
            const status = String((annotation && annotation.status) || '');
            const token = String((annotation && annotation.routing_token) || '');
            const options = Array.isArray(annotation && annotation.options) ? annotation.options : [];
            const actionable = status === 'needs_manual_target' && cmid && token
                && options.length > 0 && options.every((o) => o && typeof o === 'object');
            if (!actionable) {
                const card = bubble.querySelector('.chat-routing-card');
                card?.remove();
                return renderRoutingAnnotation(bubble, annotation, chatId) || Boolean(card) || outcome;
            }
            const annotationChanged = bubble.querySelector('.msg-routing-annotation')
                ? renderRoutingAnnotation(bubble, null) : false;
            let card = bubble.querySelector('.chat-routing-card');
            if (card && card.dataset.routingToken === token) return annotationChanged || outcome;
            card?.remove();
            card = buildRoutingCard(cmid, token, options);
            const time = bubble.querySelector('.msg-time');
            if (time) time.before(card); else bubble.append(card);
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
            .filter((field) => Object.hasOwn(frame, field)).map((field) => [field, frame[field]])), true);
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
        for (const [key, card] of quizViews) if (root.contains(card)) { submissions.release(card); quizViews.delete(key); }
        for (const view of [...mirrors.values()]) if (root.contains(view.bubble)) releaseMirror(view);
    }

    return { buildQuizCard, buildQuestionPointer, appendQuestionPointer, appendActivityQuestion, syncCurrentQuestions, observedQuestion, readQuestion, revealQuestion, setCardState, applyQuizStateFrame, renderRoutingDecision,
        releaseViews,
        setConnected(value) {
            connected = value;
            if (value || !isMain) return;  // reconnection alone is no census
            epoch += 1;
            detailReads.clear();
            for (const card of quizViews.values()) {
                card.dataset.authorityStale = '1';
                setCardState(card, card.dataset.state, null);
            }
            // every read in flight is void; its copy may read anew after the reopen
            for (const view of mirrors.values()) view.revalidationPending = false;
            syncCurrentQuestions([], false);
        },
        destroy() {
            disposed = true;
            submissions.destroy();
            for (const view of [...mirrors.values()]) releaseMirror(view);
            observations.clear(); quizViews.clear(); detailReads.clear();
        },
    };
}
