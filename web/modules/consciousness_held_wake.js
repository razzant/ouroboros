/**
 * The held own wake as its own block (docs/DESIGN.md "Dashboard › Consciousness"; owner 27.11.2026, «Блок с кнопками»):
 * one tray under the switch's header, in the chip's Shade and on Dashboard › Consciousness alike, about the one own wake
 * the clock waits for. The header keeps the switch (owner 15.11, 5A); this block keeps the wake: its head names it (Own
 * time, its start in the viewer's clock, its status mark and word), the host's sentence says what holds it and what
 * follows, a client line adds what the same state says now (Resume's estimated room at an allowance; whether the total
 * budget has room), and its controls are exactly what the server's controls for that work allow, on the shared task
 * controls: Resume (`createTaskResumeButton`), Stop now (the shared stop request after its shared confirmation) and
 * Close for a passed deadline (`createTaskCloseButton`). `Costs ›` stands while the next step is a limit edited in
 * Costs and leads while the total budget holds the wake with nothing left (Resume then stays pressable, no longer the
 * primary: the server refuses honestly); a passed deadline says so in the shared word alone, beside Close.
 * The door to its card in Main is the task door (task_reference.js). Nothing here polls: each surface paints it from the
 * facts it already reads; the last confirmed controls stay, disabled, while the newest state read failed or the socket
 * is down, and every press re-checks those facts as they are then.
 */
import { tr } from './i18n.js';
import { allowanceHold, clockText, costsMove, heldWake, heldWakeState, totalBudgetNow, wakeResumeRoomWords } from './consciousness_words.js';
import { createStatusMark, paintStatusMark } from './status_mark.js';
import { openTaskReference, taskReferenceHtml } from './task_reference.js';
import { ACTION_STOP_NOW, createTaskCloseButton, createTaskResumeButton, deadlineExpiredWord, openTaskControlMenu,
    requestStopAll, resumeEligible, taskControlBusy, taskControlTriggerLabel } from './task_control_menu.js';
import { deadlineExpiredIn, workCloseOffer, workControls, workOfEpisode } from './work_status.js';

// The Costs row each limit's door opens; the limit a wake started with is not one row, so its door opens the tab.
const COSTS_ROWS = Object.freeze({ allowance: 'OUROBOROS_CONSCIOUSNESS_DAILY_USD',
    allowance_and_cap: 'OUROBOROS_CONSCIOUSNESS_DAILY_USD', total_budget: 'TOTAL_BUDGET' });
const SEP = ' · ';
let sequence = 0;

/**
 * The block's view: null unless its own wake is held while consciousness is on. `snapshot` is a normalized
 * consciousness state (consciousness_api.js); `data` an `/api/state` body whose work projection carries that wake's
 * controls (`workOfEpisode`) and whose accounting says the total budget now: without it the block offers only its
 * doors. A work past its hard deadline (its Close offer, or the server's own expired fact) says the shared expired word
 * and drops the host's sentences, the continuation advice and the Costs door.
 */
export function heldWakeView(snapshot, data = null) {
    const taskId = heldWake(snapshot);
    if (!taskId) return null;
    const work = workOfEpisode(data, taskId);
    const controls = work ? workControls(work) : null;
    const close = work ? workCloseOffer(work, taskId) : '';
    const expired = Boolean(close) || deadlineExpiredIn(data, taskId);
    const cause = snapshot.holdCause;
    const budget = !expired && cause === 'total_budget' ? totalBudgetNow(data?.accounting) : { room: null, words: '' };
    const room = !expired && allowanceHold(snapshot) ? wakeResumeRoomWords(snapshot) : '';
    const costs = !expired && costsMove(snapshot);
    return {
        taskId, ...(expired ? { mark: { shape: work?.mark || 'pause', hue: 'ink' }, word: deadlineExpiredWord() } : heldWakeState(snapshot)),
        time: clockText(snapshot.liveWakeStartedAt), expired,
        // Past its deadline the host's sentence (sleep's "then goes on by itself") would promise what Close ends.
        text: !expired && snapshot.detail ? tr(snapshot.detail, snapshot.detail) : '',
        note: !expired && snapshot.detailNote ? tr(snapshot.detailNote, snapshot.detailNote) : '',
        advice: room || budget.words,
        costs, costsRow: COSTS_ROWS[cause] || '', costsLeads: costs && cause === 'total_budget' && budget.room !== true,
        resume: controls?.resume || '', stop: controls?.stop || [], close, unboundStart: work?.unbound_start === true,
    };
}

/**
 * Build the block in `doc`. `current()` is the view as the surface's latest facts say it now (what every press
 * re-checks), `connected()` whether the socket is up, `confirmed()` whether the newest state read stands (a failed or
 * retired read leaves the last controls in place, disabled), `refresh()` the reads that reconcile a press (its promise
 * keeps a Resume pending until they settle), `openCosts(row)` opens Costs at that limit's row, `leave()` runs before the
 * task door opens its card (the Shade closes), `confirmStop` is Stop now's confirmation (the shared dialog unless given). `paint(view)` draws `heldWakeView`'s answer; null hides the block. A press
 * of one control disables its siblings until it settles; a press started on another surface is seen at the next state
 * read only (no event says it), so every press re-checks.
 */
export function createHeldWake(doc, { current = () => null, connected = () => true, confirmed = () => true,
    refresh = () => {}, openCosts = () => {}, leave = () => {}, confirmStop = undefined } = {}) {
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
    const advice = element('p', 'cx-held-advice');
    const actions = element('div', 'cx-held-actions');
    const costs = element('button', 'btn btn-sm btn-default ui-hit-area cx-held-costs', 'Costs ›');
    costs.type = 'button';
    const stop = element('button', 'btn btn-sm btn-default ui-hit-area cx-held-stop', taskControlTriggerLabel([ACTION_STOP_NOW]));
    stop.type = 'button';
    stop.dataset.taskControlDirect = '1';  // its one action, after the shared confirmation (openTaskControlMenu)
    const door = element('span', 'cx-held-door');
    root.append(head, text, note, advice, actions);
    root.setAttribute('aria-labelledby', name.id);

    let shown = null;
    let resume = null;
    let close = null;
    let pending = 0;  // presses of this block still settling: their siblings wait
    const stopKey = (view) => (view?.stop || []).join('\u0000');
    // A press may act only on standing authority: the socket up, the newest state read confirmed, no press of this
    // block settling, and the control still offered by the facts as they are now.
    const live = () => connected() && confirmed() && pending === 0;
    const repaint = () => { if (shown) paint(shown); };
    const begin = () => { pending += 1; repaint(); };
    // The shared controls let go of their task's lease only after this returns (task_control_menu.js): the siblings it
    // disabled (Stop now beside Close) are painted again once it is let go.
    const settled = async () => {
        try { await refresh(); } finally { pending = Math.max(0, pending - 1); repaint(); setTimeout(repaint, 0); }
    };

    costs.addEventListener('click', (event) => {
        event.stopPropagation?.();
        if (shown?.costs) openCosts(shown.costsRow);
    });
    stop.addEventListener('click', (event) => {
        event.stopPropagation?.();
        const targets = shown?.stop || [];
        if (!targets.length || !live()) return;
        const key = stopKey(shown);
        openTaskControlMenu(stop, { actions: [ACTION_STOP_NOW], busy: targets.some(taskControlBusy), askConfirm: confirmStop,
            canAct: () => live() && stopKey(current()) === key && !targets.some(taskControlBusy),
            onAction: () => { begin(); return requestStopAll(targets, ACTION_STOP_NOW); }, afterAction: settled });
    });
    // The task door: a press opens its card in Main (task_reference.js raises it once; a surface's own door handler
    // above this block does not raise it again).
    door.addEventListener('click', (event) => {
        if (!openTaskReference(event)) return;
        event.stopPropagation?.();
        leave();
    });

    function placeActions(view) {
        const wanted = [view.costsLeads && costs, resume, close, view.stop.length && stop,
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
        for (const [node, words] of [[text, view.text], [note, view.note], [advice, view.advice]]) {
            if (node.textContent !== words) node.textContent = words;
            node.hidden = !words;
        }
        const usable = live();
        if (resume && resume.target !== view.resume) { resume.remove(); resume = null; }
        if (view.resume && !resume) {
            const target = view.resume;
            // The shared button re-checks the work's control at press time; its target never changes.
            const button = createTaskResumeButton(target, { canResume: () => live() && current()?.resume === target,
                onSettled: settled });
            // Its own handler runs first and, when it sends, disables itself before waiting: then this press settles.
            button.addEventListener('click', () => { if (button.disabled) begin(); });
            button.target = target;
            resume = button;
        }
        if (resume) {
            // While the total budget holds it with nothing left, raising that budget leads; Resume stays a neutral,
            // pressable capsule.
            resume.classList.toggle('btn-primary', !view.costsLeads);
            resume.classList.toggle('btn-default', view.costsLeads);
            resume.disabled = !usable || !resumeEligible(resume.target, () => current()?.resume === resume.target);
        }
        if (close && close.offer !== view.close) { close.remove(); close = null; }
        if (view.close && !close) {
            const offer = view.close;
            const button = createTaskCloseButton(view.taskId, { nameId: name.id, onSettled: settled,
                tokenNow: () => (live() && current()?.close === offer ? offer : '') });
            button.addEventListener('click', () => { if (button.disabled) begin(); });
            button.offer = offer;
            close = button;
        }
        if (close) close.disabled = !usable || taskControlBusy(view.taskId);
        stop.disabled = !usable || view.stop.some(taskControlBusy);
        if (view.unboundStart) stop.dataset.unboundStart = '1'; else delete stop.dataset.unboundStart;
        costs.classList.toggle('btn-primary', view.costsLeads);
        costs.classList.toggle('btn-default', !view.costsLeads);
        if (door.dataset.taskId !== view.taskId) {
            door.innerHTML = taskReferenceHtml(view.taskId);
            door.dataset.taskId = view.taskId;
        }
        placeActions(view);
    }

    /** The enabled controls in their order: where a keyboard open of the Shade lands first. */
    const focusable = () => (root.hidden ? [] : [...actions.children]
        .map((node) => (node === door ? node.querySelector('button') : node))
        .filter((node) => node && !node.hidden && !node.disabled));

    return { element: root, paint, focusable, destroy() { shown = null; root.remove(); } };
}
