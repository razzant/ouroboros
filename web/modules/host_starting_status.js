// The host's start as the Main and Project headers read it from /api/state: Starting… until the
// body states readiness, and the boot phases that say which part of the start runs or failed.
import { fmt, tr } from './i18n.js';

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

// The start's words for a boot phase, one producer for the room bars, their lists and Settings. `text` is the bar's
// word, Starting… until the chat accepts messages and Chat ready from then on; `step` the first named start step that
// still runs (the supervisor's startup sub-steps and the coordinator's long ones, in their order), said beside the word
// and dropped before it on a short bar, as a preparing turn's step is (work_status.js); `title` every running step and
// a fail-soft step's error, an extra for a pointer; `detail` the same in sentences where no hover is needed (the room's
// list, Settings): whether a message sent now is saved, what runs, what failed. A failed start is its own state (U001):
// the failure's tone, so the room's capsule draws the red dot and stays pressable; its cause is the step, the title and
// the Details of the room's list and of Settings. Nothing known about the phase keeps the plain Starting….
export const START_FAILED_LEAD = 'Ouroboros could not finish starting.';
const START_STEPS = [
    ['skills', 'preparing skills'],
    ['extensions', 'loading extensions'],
    ['startup:event_queue', 'preparing task transport'],
    ['startup:bootstrap', 'checking program files'],
    ['startup:workers_init', 'preparing executors'],
    ['startup:queue_restore', 'restoring the queue'],
    ['startup:old_workers_stop', 'settling previous executors'],
    ['startup:worker_spawn', 'starting executors'],
    ['startup:delegate_pre_adopt', 'restoring delegated work'],
    ['startup:custody', 'checking saved processes'],
    ['startup:task_recovery', 'restoring saved work'],
    ['startup:prune', 'releasing finished work folders'],
    ['startup:resume', 'resuming saved work'],
];
export function startingStatus(boot) {
    const steps = Array.isArray(boot?.steps) ? boot.steps : [];
    const failedStep = steps.find((step) => step?.state === 'failed');
    const failure = String(boot?.error || (failedStep ? `${failedStep.name}: ${failedStep.error || ''}` : '') || '');
    if (boot?.phase === 'failed') return { kind: 'failed', text: 'Start failed', tone: 'failed', step: failure, title: failure, detail: failure };
    const labels = START_STEPS.filter(([name]) => steps.some((step) => step?.name === name && step.state === 'running'))
        .map(([name, label]) => tr(`host_start.step.${name}`, label));
    const title = [failure, ...labels].filter(Boolean).join(' · ');
    const chat = boot?.dialogue_ready === true;
    const detail = !boot ? '' : [
        chat ? tr('host_start.saved', 'Messages are saved and answered once the start finishes.')
            : tr('host_start.unsaved', 'The chat opens once the data store is ready; a message sent before that is not saved.'),
        labels.length ? fmt('Now: {steps}.', { steps: labels.join(', ') }) : '',
        failure ? fmt('A start step failed and the start goes on: {failure}', { failure }) : '',
    ].filter(Boolean).join(' ');
    const step = labels[0] || (chat ? tr('host_start.step.finishing', 'finishing startup') : '');
    return { kind: 'starting', text: chat ? 'Chat ready' : 'Starting…', step, title, detail };
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
// them said, both forgotten on a disconnect and again at the reconnect, since a snapshot polled
// between them may be the server's from before a restart (the reconnect's snapshot proves them again).
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
