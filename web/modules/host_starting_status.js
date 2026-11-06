// The host's start as the Main and Project headers read it from /api/state: Starting… until the
// body states readiness, and the boot phases that say which part of the start runs or failed.
import { tr } from './i18n.js';

// В9: one /api/state body's `supervisor_ready`, null when it states nothing. Only
// true ends Starting…; a `supervisor_error` is not readiness and is not read here.
export function supervisorReady(data) {
    return typeof data?.supervisor_ready === 'boolean' ? data.supervisor_ready : null;
}

// The host's boot phases (the `boot` object of /api/state, the same one /api/health carries), null
// when the body states none: `phase`, `dialogue_ready`, `tasks_ready`, `steps`, `error`.
export function bootPhase(data) {
    const boot = data?.boot;
    return boot && typeof boot === 'object' && typeof boot.phase === 'string' ? boot : null;
}

// The Starting… header's words for a boot phase: the chat accepts messages while the
// background tasks are still being restored, or the start failed. A failed start is its own
// state (U001): the failure's tone, so the room's capsule draws the red dot and stays
// pressable; its cause is the step a short bar drops before the word, and rides the title for
// the Shade's Details. Nothing known about the phase keeps the plain Starting….
export const START_FAILED_LEAD = 'Ouroboros could not finish starting.';
export function startingStatus(boot) {
    const failedStep = Array.isArray(boot?.steps) ? boot.steps.find((step) => step?.state === 'failed') : null;
    const title = String(boot?.error || (failedStep ? `${failedStep.name}: ${failedStep.error || ''}` : '') || '');
    if (boot?.phase === 'failed') return { kind: 'failed', text: 'Start failed', tone: 'failed', step: title, title };
    if (boot?.dialogue_ready === true) {
        return { kind: 'starting', text: 'Chat ready, restoring background tasks', title };
    }
    return { kind: 'starting', text: 'Starting…', title };
}

/** The cause of a failed start a room's list says (U003), null when the start did not fail: the bar's own failure
 * (`badge`, the status setStatus writes), or, where the room's work words outrank the start's word in the bar, the same
 * producer over the latest snapshot's boot, so the list never says "starting" for a start that failed. */
export function startFailureCause({ badge, ready, latest }) {
    if (badge?.dataset?.tone === 'failed') return String(badge.title || '');
    const view = !ready && startingStatus(bootPhase(latest));
    return view?.kind === 'failed' ? view.title : null;
}

/**
 * The first line of a room's Shade while its start failed (U001/U003): the capsule and the Shade read the same state
 * (the badge setStatus writes from startingStatus), so the Shade says the failure, never "starting", with Restart
 * beside it and the cause behind Details. `restart` runs the shared confirmation. `paint(cause)`: null hides it.
 */
export function createStartFailure(doc, { restart }) {
    const make = (tag, className, text = '') => Object.assign(doc.createElement(tag), { className, textContent: text });
    const el = make('div', 'rw-failure');
    const row = make('div', 'ui-action-row');
    const button = make('button', 'btn btn-default btn-sm', tr('room_work.failed.restart', 'Restart'));
    button.type = 'button';
    button.dataset.primary = '';
    button.addEventListener('click', () => restart?.());
    row.append(make('p', 'rw-failure-lead', tr('room_work.failed.lead', START_FAILED_LEAD)), button);
    const details = make('details', 'ui-disclosure');
    const cause = make('pre', 'ui-result-details');
    details.append(make('summary', '', tr('room_work.failed.details', 'Details')), cause);
    el.append(row, details);
    el.hidden = true;
    return {
        el,
        paint(text) {
            el.hidden = text === null;
            if (cause.textContent !== (text || '')) cause.textContent = text || '';
            details.hidden = !text;
        },
    };
}

// One chat instance's view of the start: the readiness and phases the last snapshot that stated
// them said, both forgotten on a disconnect (the reconnect's snapshot proves them again).
// `status()` is the header reducer's input; `ready()` answers the room's readiness question.
export function createHostStart() {
    let ready = false;
    let boot = null;
    return {
        read(data) {
            ready = supervisorReady(data) ?? ready;
            boot = bootPhase(data) ?? boot;
        },
        forget() {
            ready = false;
            boot = null;
        },
        ready: () => ready,
        status: () => ({ supervisorStarting: !ready, bootPhase: boot }),
    };
}
