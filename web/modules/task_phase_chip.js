import { isTerminalTaskDetail, taskPresentation, taskTerminalSummary } from './log_events.js';
import { fmt, tr } from './i18n.js';
import { waitFacts } from './question_presentation.js';
import { strokeIcon } from './page_icons.js';
import { createStatusMark, markView, paintStatusMark } from './status_mark.js';

// The live card's own words. Its root is one the overlay never enters (a transcript), so every
// label is read through the translation seam here, at the producer.
export const liveCardLabel = {
    turnIntoProject: () => tr('task.card.turn_into_project', 'Turn into project'),
    creatingProject: () => tr('task.card.creating_project', 'Creating project…'),
};
export function liveCardCountBits(notes, children) {
    const bits = [];
    if (notes >= 2) bits.push(fmt('{n} notes', { n: notes }));
    if (children) bits.push(fmt(children === 1 ? '{n} child' : '{n} children', { n: children }));
    return bits;
}
const CHIP = {
    finalizing: () => tr('task.chip.finalizing', 'Finalizing…'),
    cancelling: () => tr('task.chip.cancelling', 'Cancelling…'),
    paused: () => tr('task.chip.paused', 'Paused'),
    pausing: () => tr('task.chip.pausing', 'Pausing…'),
    waiting: () => tr('task.chip.waiting', 'Waiting'),
    unconfirmed: () => tr('task.chip.activity_unconfirmed', 'Activity unconfirmed'),
    waitingAccess: () => tr('task.chip.waiting_for_access', 'Waiting for access'),
    working: () => tr('task.chip.working', 'Working'),
    preparing: () => tr('task.chip.preparing', 'Preparing'),
    sending: () => tr('task.chip.sending', 'Sending'),
    rejoining: () => tr('task.chip.rejoining', 'Rejoining'),
    thinking: () => tr('task.chip.thinking', 'Thinking'),
    queued: () => tr('task.chip.queued', 'Queued'),
    ownerWait: () => tr('task.chip.waiting_for_answer', 'Waiting for your answer'),
    ownerHurry: () => tr('task.chip.owner_hurry_applied', 'Owner hurry applied'),
    background: () => tr('task.fact.in_background', 'in background'),
};

/** One chip word by its name (the room and the work rows say the cards' own words). */
export const chipWord = (name) => CHIP[name]?.() || '';

// A census pause says its word alone: who or what holds the work is its reason, said beside the word on the work's
// line (work_status.workLine); the census's `owner` cause cannot even tell the owner's Pause from Ouroboros's. A cold
// sleep continues by itself when it is woken: it waits, it is not paused.
export function pausePhaseLabel(phase, cause = '') {
    return phase === 'budget_pausing' ? CHIP.pausing() : cause === 'sleep' ? CHIP.waiting() : CHIP.paused();
}

// Reviewers a settled Pause lets finish run on separately and visibly: said beside its word, in the work line's own
// words. The census names tasks and model sends alike, so it names no count.
const reviewsFinishing = () => tr('work.because.review_work_finishing', 'Review work finishing');

// What a census pause cause says on the card's line, in the work's own sentences (gateway/work_status.py REASON_TEXT).
// The census's `owner` cause cannot tell the owner's Pause from Ouroboros's, so it says nothing there.
const CAUSE_SENTENCES = {
    budget: () => tr('work.because.spending_limit', 'Spending limit reached'),
    restart: () => tr('work.because.after_restart', 'Held after a restart'),
    authority: () => tr('work.because.technical_pause', 'After a technical problem'),
    sleep: () => tr('work.because.sleeping', 'Sleeping until it is woken'),
};

export function activityWaitPhase(activity = {}) {
    const question = activity.owner_wait ?? activity.required_question;
    if (question?.owner_wait_state === 'resumed' || question?.wait_ended_at
        || ['answered', 'expired_terminal', 'superseded', 'closed'].includes(question?.quiz_state || question?.state)) return '';
    if (waitFacts(question || {}).waiting) return 'owner_wait';
    if (activity.required_question_unavailable || question) return 'unknown';
    return '';
}

// What each census phase does, as a status mark (docs/DESIGN.md "Status marks"): the census's own vocabulary
// (gateway/state.py `active_chat_activities`), which the server reads into a work's shape in
// gateway/work_status.classify_member; tests/test_status_marks_census.py holds the two together. A card without a
// projected work says its shape from here; a work's own word comes with the server's `mark`.
export const CENSUS_MARKS = Object.freeze({
    preparing: 'moving', working: 'moving', thinking: 'moving', finalizing: 'moving', budget_pausing: 'moving',
    queued: 'still', budget_paused: 'pause', unknown: 'ring',
});

/** A known outcome's tone, the same on a chip and a Logs headline: Done and Done with warnings succeed (the word says
 *  the warnings), Stopped was someone's decision, the rest failed; '' for anything that is not an outcome. */
export function outcomeTone(outcome) {
    const view = markView({ outcome });
    return view.shape === 'dot' ? view.hue : '';
}

// A chip's look beside its words: `mark` its shape, `tone` its body (the live word crimson, the owner's move amber, an
// outcome in its tone, anything else neutral), `owner` whether the owner's amber dot replaces the mark.
function phaseView(phase, text, classes = phase, { mark = 'none', tone = 'neutral', owner = false, waiting = false } = {}) {
    return { phase, text, className: `chat-live-phase ${classes}`, mark, tone, owner, waiting };
}

// The live word: crimson, in its phase's shape (a model call that works is `LIVE`).
const live = (phase) => ({ mark: CENSUS_MARKS[phase], tone: 'live' });
const LIVE = { mark: 'moving', tone: 'live' };

// The owner's move on a chip whose word does not say it: the amber dot replaces the mark, a quiet word takes the amber
// body and a live word keeps its own. Nothing is proven about unconfirmed work, so its ring stays.
function withOwner(view, owner) {
    if (!owner || view.mark === 'ring') return view;
    return { ...view, owner: true, tone: view.tone === 'neutral' ? 'owner' : view.tone };
}

function outcomeView(outcome, quietEffective = false) {
    const presentation = taskPresentation(outcome || 'done', quietEffective);
    return { ...phaseView(presentation.phase, presentation.headline, presentation.phase,
        { tone: outcomeTone(presentation.phase) || 'neutral' }), outcome: presentation.phase };
}

// Facts from the active call, not provider reasoning or task lifecycle. The
// transport sends no prompt, model/account details or inferred queue position.
// The chip names the call's state in one short word that keeps its meaning (Sending is
// not Working); ``detail`` keeps the exact phase and admission blockers for the card's details.
export function modelActivityPresentation(activity) {
    const labels = {
        preparing: [CHIP.preparing, 'Preparing request'], queued: [CHIP.queued, 'Request accepted · waiting for admission'],
        dispatching: [CHIP.sending, 'Preparing request dispatch'],
        awaiting_response: [CHIP.sending, 'Sending request / waiting for response'],
        response_ready: [CHIP.working, 'Response available'], response_received: [CHIP.working, 'Response received · processing'],
        rejoining: [CHIP.rejoining, 'Rejoining the same model operation'], unknown: [CHIP.unconfirmed, 'Model operation unconfirmed'],
    };
    const phase = activity?.phase;
    if (!Object.hasOwn(labels, phase)) return null;
    const waiting = ['queued', 'rejoining', 'unknown'].includes(phase);
    const reasons = { global_limit: 'shared capacity limit', class_limit: 'model capacity limit',
        thread_busy: 'conversation busy', stopping: 'engine stopping' };
    const blockers = phase === 'queued' && Array.isArray(activity.blockers) ? activity.blockers : [];
    const detail = [tr(`task.model_activity.${phase}`, labels[phase][1]), ...blockers
        .filter(row => Object.hasOwn(reasons, row?.kind))
        .map(row => tr(`task.model_activity.${row.kind}`, reasons[row.kind]))].join(' · ');
    // A call admitted or rejoining goes on by itself (still); an unconfirmed one is a ring.
    const look = waiting ? { mark: phase === 'unknown' ? 'ring' : 'still', waiting } : LIVE;
    return { ...phaseView('working', labels[phase][0](), waiting ? 'working waiting' : 'working', look), detail, operation: phase };
}

// One pure presentation over task facts, shared by cards, receipts and project
// summaries. Outcome and ongoing finalization are independent; a Failed label
// never moves merely because the task still has post-task work. The look rides with
// the words: a census phase's shape from CENSUS_MARKS, a work's own word in the server's
// shape (`work`, the look work_status.applyCardWorkViews keeps), and the owner's move only
// from proved facts: the chip's own question, a Resume only the owner gives
// (`resumeAction`), an admitted wait that needs the owner (`modelOwnerAction`), a member
// of its work (`work.owner`) or the owner's System message about this task (`ownerMove`).
export function taskActivityPresentation({ phase = 'working', outcome = '', ended = false,
    finalizing = false, ownerWait = '', modelWaiting = false, modelOwnerAction = false, resumeAction = false,
    ownerMove = false, hold = '', pauseCause = '', stopPolicy = '', finishingReviews = false, modelActivity = null,
    workView = null, work = null, quietEffective = false } = {}) {
    if (ended) return outcomeView(outcome, quietEffective);
    const theirs = Boolean(ownerMove || work?.owner);
    if (stopPolicy) {
        const moving = ['working', 'thinking', 'finalizing'].includes(phase) && !modelWaiting && !ownerWait && !hold;
        return withOwner(phaseView('working', stopPolicy === 'finalize' ? CHIP.finalizing() : CHIP.cancelling(),
            'working cancelling', { mark: moving ? 'moving' : 'still', tone: 'live', waiting: !moving }), theirs);
    }
    let current;
    const modelPhase = modelActivityPresentation(modelActivity);
    if (phase === 'unknown') current = phaseView('unknown', CHIP.unconfirmed(), 'warn', { mark: CENSUS_MARKS.unknown });
    // The card of a work's current episode says the work's word and its line (work_status.cardWorkView).
    else if (workView) current = { ...phaseView(workView.phase, workView.text, workView.classes, { mark: work?.mark,
        tone: workView.tone, waiting: workView.waiting }), ...(workView.secondary ? { secondary: workView.secondary } : {}),
        // The exact call still explains the work's word in the card's details where both say the same word.
        ...(modelPhase?.detail && modelPhase.text === workView.text ? { detail: modelPhase.detail } : {}) };
    else if (['budget_paused', 'budget_pausing'].includes(phase)) {
        // The chip says the word; what holds it, a hold and the reviews a settled Pause lets finish are its quieter
        // second fact.
        const beside = [CAUSE_SENTENCES[pauseCause]?.() || '', hold,
            phase === 'budget_paused' && finishingReviews ? reviewsFinishing() : ''].filter(Boolean);
        // A cold sleep wakes by itself: it waits, it is not paused.
        const mark = phase === 'budget_paused' && pauseCause === 'sleep' ? 'still' : CENSUS_MARKS[phase];
        current = withOwner(phaseView(phase === 'budget_paused' ? 'paused' : 'working', pausePhaseLabel(phase, pauseCause),
            phase === 'budget_paused' ? 'warn' : 'working waiting', { mark, waiting: true }), resumeAction);
        if (beside.length) current.secondary = beside.join(' · ');
    } else if (ownerWait === 'owner_wait') {
        current = phaseView('waiting', CHIP.ownerWait(), 'warn', { mark: 'still', tone: 'owner', owner: true, waiting: true });
    } else if (modelWaiting) {
        current = withOwner(phaseView('working', CHIP.waitingAccess(), 'working waiting', { mark: 'still', waiting: true }),
            modelOwnerAction);
    } else if (hold) current = phaseView('working', hold, 'warn', { mark: 'still', waiting: true });
    else if (ownerWait === 'unknown') current = phaseView('unknown', CHIP.unconfirmed(), 'warn', { mark: CENSUS_MARKS.unknown });
    else if (phase === 'queued') current = phaseView('queued', CHIP.queued(), 'warn', { mark: CENSUS_MARKS.queued });
    else if (finalizing || phase === 'finalizing') current = phaseView('working', CHIP.finalizing(), 'working finalizing', live('finalizing'));
    else if (phase === 'preparing') current = modelActivityPresentation({ phase: 'preparing' });
    else if (modelPhase) current = modelPhase;
    else if (phase === 'thinking') current = phaseView('working', CHIP.thinking(), 'working', live('thinking'));
    else if (phase === 'working') current = phaseView('working', CHIP.working(), 'working', live('working'));
    else current = phaseView('unknown', CHIP.unconfirmed(), 'warn', { mark: CENSUS_MARKS.unknown });
    current = withOwner(current, theirs);
    // The outcome owns the chip; the work still running is its quieter second fact, and the owner's move keeps its dot.
    if (finalizing && outcome) return { ...outcomeView(outcome), secondary: [current.text, current.secondary]
        .filter(Boolean).join(' · '), waiting: current.waiting, owner: current.owner };
    return current;
}

// Outcome knowledge and finalization are shared by the census presenter and
// the full-card adapter. Neither copy infers outcome from its rendered phase.
export function censusTaskFacts(activity, detail = null) {
    const facts = { ...detail, ...activity };
    const observedFinalizing = activity?.phase === 'finalizing';
    const summary = taskTerminalSummary({ ...facts, ...(observedFinalizing ? { task_phase: 'finalizing' } : {}) });
    // Positive finalizing supplies an open checkpoint for this display read.
    // The existing predicate still owns exceptions such as immediate Cancelled.
    const ended = isTerminalTaskDetail(observedFinalizing
        ? { ...facts, root_phase_checkpoint: { post_task_synthesis: 'running' } } : facts);
    const finalizing = observedFinalizing || Boolean(!summary.terminal && summary.observedOutcome
        && facts.root_phase_checkpoint?.post_task_synthesis);
    return { outcome: summary.observedOutcome || (ended ? summary.phase : ''), ended, finalizing,
        ...(typeof facts.quiet_effective === 'boolean' ? { quietEffective: facts.quiet_effective } : {}) };
}

// Current census facts supersede cached detail; a missing/disconnected census
// cannot prove ongoing work. The model-wait owner supplies its validated facts:
// whether an admitted wait is current and whether it needs the owner.
export function censusTaskPhase(activity, detail = null, connected = true, { waiting = false, ownerAction = false } = {}) {
    return taskActivityPresentation({
        ...censusTaskFacts(activity, detail),
        phase: connected && activity && !activity._activityUnconfirmed ? activity.phase || 'unknown' : 'unknown',
        ownerWait: activityWaitPhase(activity || {}), modelWaiting: waiting, modelOwnerAction: waiting && ownerAction,
        resumeAction: activity?.can_resume === true && activity?.pause_cause !== 'owner',
        hold: activity?.project_admission_hold?.label || '', pauseCause: activity?.pause_cause || '',
        finishingReviews: Boolean(activity?.finishing_reviews),
        modelActivity: activity?.model_activity,
    });
}

function recordPhase(record = {}, terminalPhase = 'done') {
    // The card's one parked word also represents an optional question gap;
    // its observed census phase distinguishes that from unknown activity.
    const ownerWait = record.parkedPhase === 'owner_wait' ? 'owner_wait'
        : record.parkedPhase === 'unknown' && record.censusPhase && record.censusPhase !== 'unknown' ? 'unknown' : '';
    return taskActivityPresentation({
        phase: ownerWait ? record.censusPhase || 'working'
            : record.parkedPhase || record.censusPhase || 'working',
        outcome: record.finished ? terminalPhase || 'done' : record.observedOutcome,
        ended: Boolean(record.finished), finalizing: Boolean(record.finalizingHold),
        quietEffective: record.quietEffective ?? record.historicalTerminal?.quiet_effective,
        ownerWait,
        modelWaiting: Boolean(record.modelWaiting), modelOwnerAction: Boolean(record.modelWaiting && record.modelOwnerAction),
        // The census's explicit Resume (main_waiting's rule): the owner's own Pause is not the owner's move.
        resumeAction: record.canResume === true && record.pauseCause !== 'owner',
        ownerMove: Boolean(record.ownerMove), hold: record.projectHold,
        pauseCause: record.pauseCause, stopPolicy: record.cancelPendingPolicy,
        finishingReviews: Boolean(record.finishingReviews),
        modelActivity: record.modelActivity, workView: record.workView, work: record.workLook,
    });
}

// The private chat record is an adapter, never the shared task-fact contract.
export function desiredLiveCardPhase(record = {}, terminalPhase = 'done') {
    const view = recordPhase(record, terminalPhase);
    // A detached work (FINAL_DESIGN D4) that is plainly running says so beside the chip: a fact the
    // chip's own words do not carry. Every other state keeps the chip's established wording.
    const secondary = view.secondary || (record.detached && !record.finished && view.text === CHIP.working()
        ? CHIP.background() : '');
    return { phase: view.phase, text: view.text, className: view.className, ...chipLook(view),
        ...(secondary ? { secondary } : {}), ...(view.detail ? { detail: view.detail } : {}) };
}

// The look a chip's view carries to its painter (and to a work-list row that says a card's words).
function chipLook(view) {
    return { mark: view.mark, tone: view.tone, owner: Boolean(view.owner), ...(view.outcome ? { outcome: view.outcome } : {}) };
}

/**
 * Keep an unfinished card's chip on its task's census phase (`/api/state`
 * `active_chat_activities`): paused/pausing/unknown park it until a positive
 * phase releases it. true when the chip changed.
 */
export function syncParkedPhase(record, phase = '', activity = {}) {
    if (!record || record.finished) return false;
    const facts = censusTaskFacts({ ...activity, phase: activity.phase || phase });
    if (facts.outcome) record.observedOutcome = facts.outcome;
    if (facts.finalizing) record.finalizingHold = true;
    if (activity._activityUnconfirmed) phase = 'unknown';
    const wait = activityWaitPhase(activity);
    const observed = ['budget_paused', 'budget_pausing', 'unknown'].includes(phase) ? phase
        : wait || phase;
    const parked = ['budget_paused', 'budget_pausing', 'unknown', 'queued', 'owner_wait'].includes(observed) ? observed : '';
    const cause = ['budget_paused', 'budget_pausing'].includes(parked) ? String(activity.pause_cause || '') : '';
    record.parkedPhase = parked;
    record.pauseCause = cause;
    record.censusPhase = phase;
    record.canResume = activity.can_resume === true;  // a Resume only the owner gives, unless their own Pause holds it
    record.modelActivity = activity.model_activity || null;
    record.finishingReviews = parked === 'budget_paused' && Boolean(activity.finishing_reviews);
    record.detached = activity.detached === true;  // a detached work's census row (CONTRACTS K7/K8), in any room
    const desired = desiredLiveCardPhase(record);
    return setLiveCardPhase(record, desired.phase, desired.text, desired.className, desired.secondary);
}

// A replayed final may preserve only an already-terminal phase. Ordinary DOM
// progress is presentation state, not terminal outcome truth.
export function replayTerminalPhase(record) {
    return (record?.finished ? record?.phaseEl?.dataset?.phase : '') || 'done';
}

// Preserve the authoritative unfinished phase fact across an optimistic owner
// stop. DOM text alone is insufficient: a failed request must also restore an
// open post-task finalization hold so later progress cannot repaint Working.
export function captureLiveCardPhaseState(record = {}) {
    return {
        phase: String(record?.phaseEl?.dataset?.phase || 'working'),
        finalizingHold: Boolean(record?.finalizingHold),
        observedOutcome: String(record?.observedOutcome || ''),
    };
}

export function restoreLiveCardPhaseState(record, snapshot) {
    if (!record || !snapshot || record.finished) return null;
    record.cancelPendingPolicy = '';
    record.finalizingHold = Boolean(snapshot.finalizingHold);
    record.observedOutcome = String(snapshot.observedOutcome || '');
    return desiredLiveCardPhase(record, snapshot.phase || 'working');
}

const CHIP_TONES = ['live', 'owner', 'ok', 'neutral', 'failed'];

// The chip's anatomy: the shared status mark first, then its words as one text node. Both are made on the chip's
// first paint (a template writes only the chip itself) and kept by every later one.
function chipParts(phaseEl) {
    const doc = phaseEl.ownerDocument || globalThis.document;
    let mark = phaseEl.firstElementChild;
    if (!mark?.classList?.contains('status-mark')) mark = phaseEl.insertBefore(createStatusMark(doc), phaseEl.firstChild || null);
    let words = phaseEl.lastChild;
    if (words?.nodeType !== 3) words = phaseEl.appendChild(doc.createTextNode(''));
    return { mark, words };
}

// One DOM writer for both compact and full task cards. The look is a supplied
// fact (docs/DESIGN.md "Status marks"), never a consequence of a CSS class or of
// the element's ancestry: the body tone as `data-chip-tone`, the mark through the
// shared painter in the chip's own ink.
export function paintTaskPhase(phaseEl, view, secondaryEl = null, isSubagent = false) {
    if (!phaseEl) return false;
    const activePhase = String(view.phase || 'working');
    const activeText = String(view.text || taskPresentation(activePhase).headline);
    const activeClassName = view.className || `chat-live-phase ${activePhase}`;
    const tone = CHIP_TONES.includes(view.tone) ? view.tone : 'neutral';
    const secondaryText = String(view.secondary || '');
    // The owner's move is said in words too, unless the word already says it: the mark is decorative.
    const said = `${activeText} ${secondaryText}`.includes(CHIP.ownerWait());
    const ownerNote = view.owner && !said ? tr('work.amber.action', 'Something here waits for you') : '';
    const activeLabel = `${isSubagent ? 'Subagent' : 'Task'} status: ${activeText}`
        + (secondaryText ? `, ${secondaryText}` : '') + (ownerNote ? ` · ${ownerNote}` : '');
    const secondaryChanged = paintPhaseSecondary(secondaryEl, secondaryText, phaseEl.hidden);
    const { mark, words } = chipParts(phaseEl);
    const markChanged = paintStatusMark(mark, markView({ mark: view.mark, owner: view.owner, outcome: view.outcome }));
    const changed = phaseEl.dataset.phase !== activePhase
        || phaseEl.dataset.chipTone !== tone
        || phaseEl.className !== activeClassName
        || words.data !== activeText;
    if (phaseEl.dataset.phase !== activePhase) phaseEl.dataset.phase = activePhase;
    if (phaseEl.dataset.chipTone !== tone) phaseEl.dataset.chipTone = tone;
    if (phaseEl.className !== activeClassName) phaseEl.className = activeClassName;
    // Do not make a polite live region re-announce identical routine progress.
    if (words.data !== activeText) words.data = activeText;
    if (phaseEl.getAttribute('role') !== 'status') phaseEl.setAttribute('role', 'status');
    if (phaseEl.getAttribute('aria-live') !== 'polite') phaseEl.setAttribute('aria-live', 'polite');
    if (phaseEl.getAttribute('aria-atomic') !== 'true') phaseEl.setAttribute('aria-atomic', 'true');
    if (phaseEl.getAttribute('aria-label') !== activeLabel) phaseEl.setAttribute('aria-label', activeLabel);
    return changed || markChanged || secondaryChanged;
}

// HQ1: an applied owner hurry is one more fact beside the phase chip, drawn as the
// house fast-forward vector and named for assistive technology; the root attribute
// stays the card's record of it. Returns whether the card changed.
export function markOwnerHurry(record) {
    const root = record?.root;
    if (!root || root.getAttribute('data-owner-hurry') === '1') return false;
    root.setAttribute('data-owner-hurry', '1');
    const mark = root.querySelector('[data-live-hurry]');
    if (mark) {
        mark.innerHTML = strokeIcon('<polygon points="13 19 22 12 13 5 13 19"></polygon><polygon points="2 19 11 12 2 5 2 19"></polygon>', 14);
        mark.setAttribute('aria-label', CHIP.ownerHurry());
        mark.title = CHIP.ownerHurry();
        mark.hidden = false;
    }
    return true;
}

// A phase painted without its record's view speaking for it (a reused card's first word, a review anchor let go):
// an outcome in its tone, anything else as the census path draws it.
function plainView(phase) {
    return outcomeTone(phase) ? outcomeView(phase) : taskActivityPresentation({ phase });
}

// Preserve the established setter while adapting the card's private facts to
// the same look used by compact cards and the room: the record's own view when it
// speaks for the painted phase.
export function setLiveCardPhase(record, phase = 'working', text = '', className = '', secondary = '') {
    if (!record?.phaseEl) return false;
    const view = recordPhase(record, phase);
    const changed = paintTaskPhase(record.phaseEl, {
        phase, text, className, secondary, ...chipLook(view.phase === phase ? view : plainView(phase)),
    }, phaseSecondaryElement(record), record.isSubagent);
    // The exact phase rides with the chip word it explains, never beside another one.
    const detail = view.detail && view.text === text ? fmt('Model call: {detail}', { detail: view.detail }) : '';
    const detailChanged = paintLiveDetail(record, 'livePhaseDetail', detail);
    return changed || detailChanged;
}

// One line of a card's OWN details block (a nested child card carries its own): what the
// expanded card says about its chip or amount, or its list of labelled facts.
export function ownLiveDetail(record, key) {
    const own = node => Boolean(node?.dataset && key in node.dataset);
    for (const node of record?.root?.children || []) {
        if (own(node)) return node;
        if (node.classList?.contains('chat-live-details')) {
            const line = [...(node.children || [])].find(own);
            if (line) return line;
        }
    }
    return null;
}

// Paint one details line (shown with the expanded card, which its summary toggles by pointer,
// touch and keyboard); compact rows and the sidebar keep the chip word or amount alone.
export function paintLiveDetail(record, key, text = '') {
    const el = ownLiveDetail(record, key);
    if (!el || (el.textContent === text && el.hidden === !text)) return false;
    el.textContent = text;
    el.hidden = !text;
    return true;
}

function phaseSecondaryElement(record) {
    if (record.phaseSecondaryEl === undefined) {
        record.phaseSecondaryEl = record.root?.querySelector?.('[data-live-phase-secondary]') || null;
    }
    return record.phaseSecondaryEl;
}

// The secondary chip is another fact beside the outcome: quiet meta text that never
// moves. The primary chip's accessible name states both so the pair is read as one status.
function paintPhaseSecondary(el, text, primaryHidden) {
    if (!el) return false;
    const hidden = !text || Boolean(primaryHidden);
    const changed = el.textContent !== text || el.hidden !== hidden;
    if (el.textContent !== text) el.textContent = text;
    if (el.hidden !== hidden) el.hidden = hidden;
    return changed;
}

// Shared inert anatomy; the caller retains the reason (review or missing
// historical outcome), and no runtime lifecycle status is invented.
export function setInertCardPresentation(record, enabled) {
    if (!record?.phaseEl) return;
    record.phaseEl.hidden = enabled;
    const view = recordPhase(record);
    const phase = record.phaseEl.dataset.phase || view.phase;
    paintTaskPhase(record.phaseEl, { phase, secondary: view.secondary,
        text: record.phaseEl.textContent || view.text,
        className: record.phaseEl.className || view.className,
        ...chipLook(view.phase === phase ? view : plainView(phase)) },
        phaseSecondaryElement(record), record.isSubagent);
    if (record.root?.dataset) record.root.dataset.inert = enabled ? '1' : '0';
}

// Census/queue reads carry the host's hold fact; {} clears it after recovery.
// Undefined preserves the recorded fact across unrelated presentation writes.
export function setHistoricalUnavailable(record, enabled, held) {
    const label = typeof held === 'string' ? held : held?.label || '';
    const detail = held?.detail || '';
    const holdChanged = Boolean(record) && held !== undefined
        && ((record.projectHold || '') !== label || (record.projectHoldDetail || '') !== detail);
    if (holdChanged) Object.assign(record, { projectHold: label, projectHoldDetail: detail });
    if (!record || (!holdChanged && Boolean(record.historicalUnavailable) === enabled && !record.historicalUnconfirmed)) return false;
    record.historicalUnavailable = enabled;
    record.historicalUnconfirmed = false;
    setInertCardPresentation(record, enabled || Boolean(record.reviewAnchor));
    if (!enabled && !record.reviewAnchor) {
        const desired = desiredLiveCardPhase(record);
        setLiveCardPhase(record, desired.phase, desired.text, desired.className, desired.secondary);
    }
    return true;
}

export function setHistoricalUnconfirmed(record) {
    if (!record || record.finished || record.reviewAnchor || record.historicalUnavailable
            || record.historicalUnconfirmed) return false;
    record.historicalUnconfirmed = true;
    setInertCardPresentation(record, true);
    return true;
}
