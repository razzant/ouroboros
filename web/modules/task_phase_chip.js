import { isTerminalTaskDetail, taskPresentation, taskTerminalSummary } from './log_events.js';
import { fmt, tr } from './i18n.js';
import { waitFacts } from './question_presentation.js';

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
    unconfirmed: () => tr('task.chip.activity_unconfirmed', 'Activity unconfirmed'),
    waitingAccess: () => tr('task.chip.waiting_for_access', 'Waiting for access'),
    working: () => tr('task.chip.working', 'Working'),
    thinking: () => tr('task.chip.thinking', 'Thinking'),
    queued: () => tr('task.chip.queued', 'Queued'),
    ownerWait: () => tr('task.chip.waiting_for_answer', 'Waiting for your answer'),
};

export function pausePhaseLabel(phase, cause = '') {
    const label = phase === 'budget_pausing' ? CHIP.pausing() : CHIP.paused();
    const reasons = { budget: ['budget', 'budget limit'], owner: ['owner', 'owner pause'],
        restart: ['restart', 'after restart'], sleep: ['sleep', 'sleep'] };
    const reason = reasons[cause];
    return reason ? fmt('{state} · {reason}', { state: label,
        reason: tr(`task.pause_cause.${reason[0]}`, reason[1]) }) : label;
}

export function activityWaitPhase(activity = {}) {
    const question = activity.owner_wait ?? activity.required_question;
    if (question?.owner_wait_state === 'resumed' || question?.wait_ended_at
        || ['answered', 'expired_terminal', 'superseded'].includes(question?.quiz_state || question?.state)) return '';
    if (waitFacts(question || {}).waiting) return 'owner_wait';
    if (activity.required_question_unavailable || question) return 'unknown';
    return '';
}

function phaseView(phase, text, classes = phase, motion = false, waiting = false) {
    return { phase, text, className: `chat-live-phase ${classes}`, motion, waiting };
}

function outcomeView(outcome) {
    const presentation = taskPresentation(outcome || 'done');
    return phaseView(presentation.phase, presentation.headline);
}

// One pure presentation over task facts, shared by cards, receipts and project
// summaries. Outcome and ongoing finalization are independent; a Failed label
// never animates merely because the task still has post-task work.
export function taskActivityPresentation({ phase = 'working', outcome = '', ended = false,
    finalizing = false, ownerWait = '', modelWaiting = false, hold = '', pauseCause = '', stopPolicy = '' } = {}) {
    if (ended) return outcomeView(outcome);
    if (stopPolicy) {
        const motion = ['working', 'thinking', 'finalizing'].includes(phase) && !modelWaiting && !ownerWait && !hold;
        return phaseView('working', stopPolicy === 'finalize' ? CHIP.finalizing() : CHIP.cancelling(),
            'working cancelling', motion, !motion);
    }
    let current;
    if (phase === 'unknown') current = phaseView('unknown', CHIP.unconfirmed(), 'warn');
    else if (['budget_paused', 'budget_pausing'].includes(phase)) {
        const label = [pausePhaseLabel(phase, pauseCause), hold].filter(Boolean).join(' · ');
        current = phaseView(phase === 'budget_paused' ? 'paused' : 'working', label,
            phase === 'budget_paused' ? 'warn' : 'working waiting', false, true);
    } else if (ownerWait === 'owner_wait') current = phaseView('waiting', CHIP.ownerWait(), 'warn', false, true);
    else if (modelWaiting) current = phaseView('working', CHIP.waitingAccess(), 'working waiting', false, true);
    else if (hold) current = phaseView('working', hold, 'warn', false, true);
    else if (ownerWait === 'unknown') current = phaseView('unknown', CHIP.unconfirmed(), 'warn');
    else if (phase === 'queued') current = phaseView('queued', CHIP.queued(), 'warn');
    else if (finalizing || phase === 'finalizing') current = phaseView('working', CHIP.finalizing(), 'working finalizing', true);
    else if (phase === 'thinking') current = phaseView('working', CHIP.thinking(), 'working', true);
    else if (phase === 'working') current = phaseView('working', CHIP.working(), 'working', true);
    else current = phaseView('unknown', CHIP.unconfirmed(), 'warn');
    if (finalizing && outcome) return { ...outcomeView(outcome), secondary: current.text,
        secondaryMotion: current.motion, waiting: current.waiting };
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
    return { outcome: summary.observedOutcome || (ended ? summary.phase : ''), ended, finalizing };
}

// Current census facts supersede cached detail; a missing/disconnected census
// cannot prove ongoing work. The model-wait owner supplies its validated fact.
export function censusTaskPhase(activity, detail = null, connected = true, modelWaiting = false) {
    return taskActivityPresentation({
        ...censusTaskFacts(activity, detail),
        phase: connected && activity && !activity._activityUnconfirmed ? activity.phase || 'unknown' : 'unknown',
        ownerWait: activityWaitPhase(activity || {}), modelWaiting,
        hold: activity?.project_admission_hold?.label || '', pauseCause: activity?.pause_cause || '',
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
        ownerWait,
        modelWaiting: Boolean(record.modelWaiting), hold: record.projectHold,
        pauseCause: record.pauseCause, stopPolicy: record.cancelPendingPolicy,
    });
}

// The private chat record is an adapter, never the shared task-fact contract.
export function desiredLiveCardPhase(record = {}, terminalPhase = 'done') {
    const view = recordPhase(record, terminalPhase);
    return { phase: view.phase, text: view.text, className: view.className,
        ...(view.secondary ? { secondary: view.secondary } : {}) };
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

// One DOM writer for both compact and full task cards. Motion is a supplied
// fact, never a consequence of a CSS class or of the element's ancestry.
export function paintTaskPhase(phaseEl, view, secondaryEl = null, isSubagent = false) {
    if (!phaseEl) return false;
    const activePhase = String(view.phase || 'working');
    const activeText = String(view.text || taskPresentation(activePhase).headline);
    const activeClassName = view.className || `chat-live-phase ${activePhase}`;
    const secondaryText = String(view.secondary || '');
    const motion = view.motion && !phaseEl.hidden ? '1' : '0';
    const activeLabel = `${isSubagent ? 'Subagent' : 'Task'} status: ${activeText}`
        + (secondaryText ? `, ${secondaryText}` : '');
    const secondaryChanged = paintPhaseSecondary(secondaryEl, secondaryText, phaseEl.hidden, view.secondaryMotion);
    const changed = phaseEl.dataset.phase !== activePhase
        || phaseEl.dataset.motion !== motion
        || phaseEl.className !== activeClassName
        || phaseEl.textContent !== activeText;
    if (phaseEl.dataset.phase !== activePhase) phaseEl.dataset.phase = activePhase;
    if (phaseEl.dataset.motion !== motion) phaseEl.dataset.motion = motion;
    if (phaseEl.className !== activeClassName) phaseEl.className = activeClassName;
    // Do not make a polite live region re-announce identical routine progress.
    if (phaseEl.textContent !== activeText) phaseEl.textContent = activeText;
    if (phaseEl.getAttribute('role') !== 'status') phaseEl.setAttribute('role', 'status');
    if (phaseEl.getAttribute('aria-live') !== 'polite') phaseEl.setAttribute('aria-live', 'polite');
    if (phaseEl.getAttribute('aria-atomic') !== 'true') phaseEl.setAttribute('aria-atomic', 'true');
    if (phaseEl.getAttribute('aria-label') !== activeLabel) phaseEl.setAttribute('aria-label', activeLabel);
    return changed || secondaryChanged;
}

// Preserve the established setter while adapting the card's private facts to
// the same motion projection used by compact cards and the sidebar.
export function setLiveCardPhase(record, phase = 'working', text = '', className = '', secondary = '') {
    if (!record?.phaseEl) return false;
    const view = recordPhase(record, phase);
    const active = !record.reviewAnchor && !record.historicalUnavailable && !record.historicalUnconfirmed;
    const changed = paintTaskPhase(record.phaseEl, {
        phase, text, className, secondary,
        motion: active && view.phase === phase && view.motion,
        secondaryMotion: active && view.secondaryMotion,
    }, phaseSecondaryElement(record), record.isSubagent);
    return setLiveCardTypingVisible(record, !record.finished) || changed;
}

function phaseSecondaryElement(record) {
    if (record.phaseSecondaryEl === undefined) {
        record.phaseSecondaryEl = record.root?.querySelector?.('[data-live-phase-secondary]') || null;
    }
    return record.phaseSecondaryEl;
}

// The secondary chip is another fact beside the outcome. The primary chip's
// accessible name states both so the pair is read as one status.
function paintPhaseSecondary(el, text, primaryHidden, motion) {
    if (!el) return false;
    const hidden = !text || Boolean(primaryHidden);
    const moving = motion && !hidden ? '1' : '0';
    const changed = el.textContent !== text || el.hidden !== hidden || Boolean(el.dataset && el.dataset.motion !== moving);
    if (el.textContent !== text) el.textContent = text;
    if (el.hidden !== hidden) el.hidden = hidden;
    if (el.dataset && el.dataset.motion !== moving) el.dataset.motion = moving;
    return changed;
}

// Phase and activity share this one animation writer. A subscription wait
// remains unfinished without pretending the paused role is doing computation.
export function setLiveCardTypingVisible(record, visible) {
    if (!record?.inlineTypingEl) return false;
    const view = recordPhase(record);
    const display = visible && (view.motion || view.secondaryMotion)
        && !record.reviewAnchor && !record.historicalUnavailable && !record.historicalUnconfirmed ? '' : 'none';
    if (record.inlineTypingEl.style.display === display) return false;
    record.inlineTypingEl.style.display = display;
    return Boolean(record.inlineTypingEl.isConnected);
}

// Shared inert anatomy; the caller retains the reason (review or missing
// historical outcome), and no runtime lifecycle status is invented.
export function setInertCardPresentation(record, enabled) {
    if (!record?.phaseEl) return;
    record.phaseEl.hidden = enabled;
    const view = recordPhase(record);
    paintTaskPhase(record.phaseEl, { ...view,
        phase: record.phaseEl.dataset.phase || view.phase,
        text: record.phaseEl.textContent || view.text,
        className: record.phaseEl.className || view.className },
        phaseSecondaryElement(record), record.isSubagent);
    if (record.root?.dataset) record.root.dataset.inert = enabled ? '1' : '0';
    setLiveCardTypingVisible(record, !enabled && !record.finished);
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
