/**
 * The held own wake as its own block (docs/DESIGN.md "Dashboard › Consciousness"; owner 27.11.2026, «Блок с кнопками»):
 * one tray under the switch's header, in the chip's Shade and on Dashboard › Consciousness alike, about the one own wake
 * the clock waits for. The header keeps the switch (owner 15.11, 5A); this block keeps the wake: its head names it (Own
 * time, its start in the viewer's clock, its status mark and word), the host's sentence says what holds it and what
 * follows, and its controls are exactly what the server's controls for that work allow, on the shared task controls:
 * Resume (`createTaskResumeButton`), Stop now (the shared stop request after its shared confirmation) and Close for a
 * passed deadline (`createTaskCloseButton`). `Costs ›` stands while the next step is a limit edited in Costs and leads
 * when the total budget holds the wake (Resume then stays pressable, no longer the primary: the server refuses honestly
 * while nothing is left); the door to its card in Main is the task door (task_reference.js). Nothing here polls: each
 * surface paints it from the facts it already reads, and every press re-checks those facts as they are then.
 */
import { tr } from './i18n.js';
import { clockText, costsMove, heldWake, heldWakeState } from './consciousness_words.js';
import { createStatusMark, paintStatusMark } from './status_mark.js';
import { openTaskReference, taskReferenceHtml } from './task_reference.js';
import { ACTION_STOP_NOW, createTaskCloseButton, createTaskResumeButton, openTaskControlMenu, requestStopAll, resumeEligible,
    taskControlBusy, taskControlTriggerLabel } from './task_control_menu.js';
import { workCloseOffer, workControls, workOfEpisode } from './work_status.js';

// The Costs row each limit's door opens; the limit a wake started with is not one row, so its door opens the tab.
const COSTS_ROWS = Object.freeze({ allowance: 'OUROBOROS_CONSCIOUSNESS_DAILY_USD',
    allowance_and_cap: 'OUROBOROS_CONSCIOUSNESS_DAILY_USD', total_budget: 'TOTAL_BUDGET' });
const SEP = ' · ';
let sequence = 0;

/**
 * The block's view: null unless its own wake is held while consciousness is on. `snapshot` is a normalized
 * consciousness state (consciousness_api.js); `data` an `/api/state` body whose work projection carries that wake's
 * controls (`workOfEpisode`): without it the block offers only its doors.
 */
export function heldWakeView(snapshot, data = null) {
    const taskId = heldWake(snapshot);
    if (!taskId) return null;
    const work = workOfEpisode(data, taskId);
    const controls = work ? workControls(work) : null;
    const cause = snapshot.holdCause;
    return {
        taskId, ...heldWakeState(snapshot), time: clockText(snapshot.liveWakeStartedAt),
        text: snapshot.detail ? tr(snapshot.detail, snapshot.detail) : '',
        note: snapshot.detailNote ? tr(snapshot.detailNote, snapshot.detailNote) : '',
        costs: costsMove(snapshot), costsRow: COSTS_ROWS[cause] || '', costsLeads: cause === 'total_budget',
        resume: controls?.resume || '', stop: controls?.stop || [], close: work ? workCloseOffer(work, taskId) : '',
        unboundStart: work?.unbound_start === true,
    };
}

/**
 * Build the block in `doc`. `current()` is the view as the surface's latest facts say it now (what every press
 * re-checks), `connected()` whether the socket is up, `refresh()` the reads that reconcile a press (its promise keeps a
 * Resume pending until they settle), `openCosts(row)` opens Costs at that limit's row, `leave()` runs before the task
 * door opens its card (the Shade closes). `paint(view)` draws `heldWakeView`'s answer; null hides the block.
 */
export function createHeldWake(doc, { current = () => null, connected = () => true, refresh = () => {},
    openCosts = () => {}, leave = () => {} } = {}) {
    const element = (tag, className = '', text = '') => {
        const node = doc.createElement(tag);
        if (className) node.className = className;
        if (text) node.textContent = text;
        return node;
    };
    const root = element('section', 'cx-held');
    root.hidden = true;
    const head = element('p', 'cx-held-head');
    const name = element('span', 'cx-held-name');
    name.id = `cx-held-name-${(sequence += 1)}`;
    const status = element('span', 'cx-held-status');
    const mark = createStatusMark(doc, 'cx-held-mark');
    const word = element('span', 'cx-held-word');
    status.append(mark, word);
    head.append(name, status);
    const text = element('p', 'cx-held-text');
    const note = element('p', 'cx-held-note');
    const actions = element('div', 'cx-held-actions');
    const costs = element('button', 'btn btn-sm btn-default ui-hit-area cx-held-costs', 'Costs ›');
    costs.type = 'button';
    const stop = element('button', 'btn btn-sm btn-default ui-hit-area cx-held-stop', taskControlTriggerLabel([ACTION_STOP_NOW]));
    stop.type = 'button';
    stop.dataset.taskControlDirect = '1';  // its one action, after the shared confirmation (openTaskControlMenu)
    const door = element('span', 'cx-held-door');
    root.append(head, text, note, actions);
    root.setAttribute('aria-labelledby', name.id);

    let shown = null;
    let resume = null;
    let close = null;
    const stopKey = (view) => (view?.stop || []).join('\u0000');

    costs.addEventListener('click', (event) => {
        event.stopPropagation?.();
        if (shown?.costs) openCosts(shown.costsRow);
    });
    stop.addEventListener('click', (event) => {
        event.stopPropagation?.();
        const targets = shown?.stop || [];
        if (!targets.length || !connected()) return;
        const key = stopKey(shown);
        openTaskControlMenu(stop, { actions: [ACTION_STOP_NOW], busy: targets.some(taskControlBusy),
            canAct: () => connected() && stopKey(current()) === key && !targets.some(taskControlBusy),
            onAction: () => requestStopAll(targets, ACTION_STOP_NOW), afterAction: () => refresh() });
    });
    // The task door: a press opens its card in Main (task_reference.js raises it once; a surface's own door handler
    // above this block does not raise it again).
    door.addEventListener('click', (event) => {
        if (!openTaskReference(event)) return;
        event.stopPropagation?.();
        leave();
    });

    function placeActions(view) {
        const wanted = [view.costs && view.costsLeads && costs, resume, close, view.stop.length && stop,
            view.costs && !view.costsLeads && costs, door].filter(Boolean);
        // Only a changed order moves a node: a control that keeps its place keeps the reader's focus.
        if (wanted.length !== actions.children.length || wanted.some((node, index) => actions.children[index] !== node)) {
            actions.replaceChildren(...wanted);
        }
    }

    function paint(view) {
        shown = view;
        root.hidden = !view;
        if (!view) return;
        const named = [tr('task.title.own_time', 'Own time'), view.time].filter(Boolean).join(SEP);
        if (name.textContent !== named) name.textContent = named;
        if (word.textContent !== view.word) word.textContent = view.word;
        paintStatusMark(mark, view.mark);
        if (text.textContent !== view.text) text.textContent = view.text;
        if (note.textContent !== view.note) note.textContent = view.note;
        text.hidden = !view.text;
        note.hidden = !view.note;
        const online = connected();
        if (resume && resume.target !== view.resume) { resume.remove(); resume = null; }
        if (view.resume && !resume) {
            const target = view.resume;
            // The shared button re-checks the work's control at press time; its target never changes.
            resume = createTaskResumeButton(target, { canResume: () => connected() && current()?.resume === target,
                onSettled: () => refresh() });
            resume.target = target;
        }
        if (resume) {
            // While the total budget holds it, raising that budget leads; Resume stays a neutral, pressable capsule.
            resume.classList.toggle('btn-primary', !view.costsLeads);
            resume.classList.toggle('btn-default', view.costsLeads);
            resume.disabled = !resumeEligible(resume.target, () => connected() && current()?.resume === resume.target);
        }
        if (close && close.offer !== view.close) { close.remove(); close = null; }
        if (view.close && !close) {
            const offer = view.close;
            close = createTaskCloseButton(view.taskId, { nameId: name.id, onSettled: () => refresh(),
                tokenNow: () => (connected() && current()?.close === offer ? offer : '') });
            close.offer = offer;
        }
        if (close) close.disabled = !online || taskControlBusy(view.taskId);
        stop.disabled = !online || view.stop.some(taskControlBusy);
        if (view.unboundStart) stop.dataset.unboundStart = '1'; else delete stop.dataset.unboundStart;
        costs.classList.toggle('btn-primary', view.costsLeads);
        costs.classList.toggle('btn-default', !view.costsLeads);
        if (door.dataset.taskId !== view.taskId) {
            door.innerHTML = taskReferenceHtml(view.taskId);
            door.dataset.taskId = view.taskId;
        }
        placeActions(view);
    }

    return { element: root, paint, destroy() { shown = null; root.remove(); } };
}
