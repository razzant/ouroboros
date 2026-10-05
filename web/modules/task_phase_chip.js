import { taskPresentation } from './log_events.js';
import { fmt, tr } from './i18n.js';

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
        || ['answered', 'expired_terminal', 'superseded'].includes(question?.quiz_state)) return '';
    if (question?.owner_wait_state === 'waiting') return 'owner_wait';
    if (activity.required_question_unavailable || question) return 'unknown';
    return '';
}

// Pure desired-chip projection. Terminal truth wins; while unfinished, an
// owner stop/finalization hold stays sticky across ordinary progress frames.
export function desiredLiveCardPhase(record = {}, terminalPhase = 'done') {
    if (record.finished) {
        const presentation = taskPresentation(terminalPhase || 'done');
        return {
            phase: presentation.phase,
            text: presentation.headline,
            className: `chat-live-phase ${presentation.phase}`,
        };
    }
    if (record.cancelPendingPolicy) {
        return {
            phase: 'working',
            text: record.cancelPendingPolicy === 'finalize' ? CHIP.finalizing() : CHIP.cancelling(),
            className: 'chat-live-phase working cancelling',
        };
    }
    if (record.finalizingHold) {
        // #1110: when the outcome is already observed, it OWNS the chip and the
        // hold states itself beside it. A card whose task had failed used to read
        // only "Finalizing…", so the failure had to be smuggled into the title.
        // D10: the owner's Pause of that late work is the same second fact.
        const observed = String(record.observedOutcome || '');
        const lateKind = { budget_paused: 'paused', budget_pausing: 'pausing' }[record.parkedPhase] || 'finalizing';
        const late = lateKind === 'finalizing' ? CHIP.finalizing()
            : pausePhaseLabel(record.parkedPhase, record.pauseCause);
        if (observed) {
            const presentation = taskPresentation(observed);
            return {
                phase: presentation.phase,
                text: presentation.headline,
                className: `chat-live-phase ${presentation.phase}`,
                secondary: late,
            };
        }
        if (lateKind === 'finalizing') return {
            phase: 'working',
            text: late,
            className: 'chat-live-phase working finalizing',
        };
    }
    // Owner Batch4: a paused task (owner Pause, budget pause, Restart hold) is
    // not working, and neither is one still settling its Pause.
    if (record.parkedPhase === 'unknown') return { phase: 'unknown', text: CHIP.unconfirmed(), className: 'chat-live-phase warn' };
    if (record.parkedPhase === 'budget_paused') return { phase: 'paused', text: pausePhaseLabel(record.parkedPhase, record.pauseCause), className: 'chat-live-phase warn' };
    if (record.parkedPhase === 'budget_pausing') return {
        phase: 'working', text: pausePhaseLabel(record.parkedPhase, record.pauseCause), className: 'chat-live-phase working waiting',
    };
    if (record.parkedPhase === 'owner_wait') return { phase: 'waiting', text: CHIP.ownerWait(), className: 'chat-live-phase warn' };
    if (record.modelWaiting) return {
        phase: 'working', text: CHIP.waitingAccess(), className: 'chat-live-phase working waiting',
    };
    // A census Project/scope verification hold: an unfinished, static amber wait.
    if (record.projectHold) return { phase: 'working', text: record.projectHold, className: 'chat-live-phase warn' };
    if (record.parkedPhase === 'queued') return { phase: 'queued', text: CHIP.queued(), className: 'chat-live-phase warn' };
    return { phase: 'working', text: CHIP.working(), className: 'chat-live-phase working' };
}

/**
 * Keep an unfinished card's chip on its task's census phase (`/api/state`
 * `active_chat_activities`): paused/pausing/unknown park it until a positive
 * phase releases it. true when the chip changed.
 */
export function syncParkedPhase(record, phase = '', activity = {}) {
    const observed = ['budget_paused', 'budget_pausing', 'unknown'].includes(phase) ? phase
        : activityWaitPhase(activity) || phase;
    const parked = ['budget_paused', 'budget_pausing', 'unknown', 'queued', 'owner_wait'].includes(observed) ? observed : '';
    const cause = ['budget_paused', 'budget_pausing'].includes(parked) ? String(activity.pause_cause || '') : '';
    if (!record || record.finished || (record.parkedPhase || '') === parked && (record.pauseCause || '') === cause) return false;
    record.parkedPhase = parked;
    record.pauseCause = cause;
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

// One writer for the stable factual task/subagent phase chip. Technical
// nonterminal diagnostics stay in the card timeline/details.
export function setLiveCardPhase(record, phase = 'working', text = '', className = '', secondary = '') {
    if (!record?.phaseEl) return false;
    const activePhase = String(phase || 'working');
    const activeText = String(text || taskPresentation(activePhase).headline);
    const activeClassName = className || `chat-live-phase ${activePhase}`;
    const secondaryText = String(secondary || '');
    const activeLabel = `${record.isSubagent ? 'Subagent' : 'Task'} status: ${activeText}`
        + (secondaryText ? `, ${secondaryText}` : '');
    const secondaryChanged = setLiveCardPhaseSecondary(record, secondaryText);
    const phaseEl = record.phaseEl;
    const changed = phaseEl.dataset.phase !== activePhase
        || phaseEl.className !== activeClassName
        || phaseEl.textContent !== activeText;
    if (phaseEl.dataset.phase !== activePhase) phaseEl.dataset.phase = activePhase;
    if (phaseEl.className !== activeClassName) phaseEl.className = activeClassName;
    // Do not make a polite live region re-announce identical routine progress.
    if (phaseEl.textContent !== activeText) phaseEl.textContent = activeText;
    if (phaseEl.getAttribute('role') !== 'status') phaseEl.setAttribute('role', 'status');
    if (phaseEl.getAttribute('aria-live') !== 'polite') phaseEl.setAttribute('aria-live', 'polite');
    if (phaseEl.getAttribute('aria-atomic') !== 'true') phaseEl.setAttribute('aria-atomic', 'true');
    if (phaseEl.getAttribute('aria-label') !== activeLabel) phaseEl.setAttribute('aria-label', activeLabel);
    return setLiveCardTypingVisible(record, !record.finished) || changed || secondaryChanged;
}

// The secondary chip is a SEPARATE fact beside the outcome, never a second
// status word: only the finalization hold writes it, and the primary chip's
// accessible name states both so the pair is read as one status.
export function setLiveCardPhaseSecondary(record, text = '') {
    if (!record) return false;
    if (record.phaseSecondaryEl === undefined) {
        record.phaseSecondaryEl = record.root?.querySelector?.('[data-live-phase-secondary]') || null;
    }
    const el = record.phaseSecondaryEl;
    if (!el) return false;
    const next = String(text || '');
    const hidden = !next || Boolean(record.phaseEl?.hidden);
    if (el.textContent === next && el.hidden === hidden) return false;
    el.textContent = next;
    el.hidden = hidden;
    return Boolean(el.isConnected);
}

// Phase and activity share this one animation writer. A subscription wait
// remains unfinished without pretending the paused role is doing computation.
export function setLiveCardTypingVisible(record, visible) {
    if (!record?.inlineTypingEl) return false;
    const display = visible && !record.modelWaiting && !record.projectHold && !['budget_paused', 'unknown', 'queued', 'owner_wait'].includes(record.parkedPhase)
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
    setLiveCardPhaseSecondary(record, enabled ? '' : desiredLiveCardPhase(record).secondary);
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
