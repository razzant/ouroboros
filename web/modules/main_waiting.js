// What waits on the owner in Main has no section of its own (DESIGN "Project question
// mirror"): questions keep their place in the feed, and a task that cannot continue without
// the owner is one System message there with the task's own controls. One amber pill among the
// composer's pills reaches whichever of them is off screen. It owns no polling or saved state:
// `/api/state` names the canonical questions and actions, the feed holds their cards.
import { createModelWaitController } from './model_wait.js';
import { projectReference, nameProjectReference } from './project_reference.js';
import { createTaskResumeButton, openTaskControlMenu, requestStop, hurryTaskAction,
    resumeTaskAction, ACTION_RESUME, ACTION_HURRY, TASK_CONTROL_TRIGGER_LABEL } from './task_control_menu.js';
import { syncParkedPhase } from './task_phase_chip.js';
import { unknownDeliveryRemovable } from './question_presentation.js';
import { createSystemMessageActions } from './ui_helpers.js';
import { showToast } from './toast.js';
import { fmt, tx } from './i18n.js';

const GLYPH = '<svg class="chat-question-nav-glyph" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>';
const ARROW = '<svg class="chat-question-nav-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>';
// An unanswered question stays current after its author ends (until answered or closed). A late
// answer whose delivery a restart left unknown waits on the owner too: only they can remove it
// from pending, and its card offers that.
const UNANSWERED = ['open', 'expired_terminal'];
const SETTLED = ['answered', 'closed', 'superseded'];
const waitsAnswer = row => UNANSWERED.includes(row.quiz_state);
const waitsOwner = row => waitsAnswer(row) || (Boolean(row.project_id) && unknownDeliveryRemovable(row));
const rowKey = row => `${row.task_id}:${row.quiz_id}`;
// What a census read vouches for: complete, everything; incomplete, every question its gaps do
// not name (a gap names a task, or a task and its quiz). A gap naming no task (index, registry or
// bindings unavailable), an unexplained incompleteness or a failed read vouches for nothing.
function vouchedBy(data) {
    if (data?.current_questions_complete === true) return () => true;
    const gaps = Array.isArray(data?.current_questions_gaps) ? data.current_questions_gaps : [];
    if (!gaps.length || gaps.some(gap => !gap?.task_id)) return () => false;
    return (taskId, quizId) => !gaps.some(gap => String(gap.task_id) === String(taskId)
        && (!gap.quiz_id || String(gap.quiz_id) === String(quizId)));
}
const REASONS = { budget: 'Paused at the spending limit. Change the limit if needed, then Resume.',
    restart: 'Needs Resume after the restart.', resume: 'Needs an explicit Resume.',
    model_access: 'Needs you to restore model access.' };

// `reading` returns the chat's reading position (chat_reading_position.js), read when used like the
// question owner: a landing claims it, as any other navigation does, so the feed stops following the live edge.
export function createMainWaiting({ page, messagesDiv, toolbar = page?.querySelector('.chat-toolbar-row'), insertMessageNode,
    stampNodeTimestamp, openSettings, refresh = () => {}, mutate = fn => fn(), decision = () => null,
    revealQuestion = () => {}, reading = () => null }) {
    // The question owner (chat_decision.js), read when used: chat.js creates it after this navigator.
    const owner = () => decision() || {};
    const pill = document.createElement('button');
    pill.type = 'button';
    pill.className = 'chat-question-nav';
    pill.hidden = true;
    pill.innerHTML = `${GLYPH}<span class="chat-question-nav-count"></span><span class="chat-question-nav-word"></span>${ARROW}`;
    toolbar?.append(pill);
    const records = new Map();
    let questions = [];  // the canonical rows that wait on the owner
    // Rows the latest census named while the socket was open: only these mount with their form
    // live. One kept from an older read (a gap, a failed read, a reconnect) mounts withheld until
    // its own canonical read confirms it.
    let fresh = new Set();
    let connected = true;
    let cursor = '';     // the last target shown: the next press continues after it
    const waits = createModelWaitController({ getRecord: id => records.get(id), openSettings,
        onDomWrite: mutate, includeWait: (id, row) => Boolean(records.get(id)?.row.model_waits?.[row.wait_id]) });

    // ---- owner-required actions: one System message per task, the task's own controls
    function recordFor(row) {
        const id = String(row.task_id || row.activity_id || '');
        let record = records.get(id);
        if (record) return record;
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble system owner-action';
        bubble.dataset.ownerAction = id;
        bubble.tabIndex = -1;  // the pill lands keyboard focus on the message, beside its controls
        const sender = document.createElement('div');
        sender.className = 'sender';
        sender.textContent = 'System';
        const root = document.createElement('section');
        root.className = 'owner-action-body';
        root.dataset.taskId = id;
        const heading = document.createElement('div');
        heading.className = 'model-wait-heading';
        const title = document.createElement('strong');
        const phaseEl = document.createElement('span');
        const reason = document.createElement('div');
        reason.className = 'owner-action-reason';
        const controls = createSystemMessageActions();
        const timelineEl = document.createElement('div');
        timelineEl.hidden = true;
        heading.append(title, phaseEl);
        root.append(heading, reason, controls, timelineEl);
        bubble.append(sender, root);
        const reference = row.project_id ? projectReference({ id: row.project_id, name: row.project_name }, { taskId: id }) : null;
        if (reference) controls.append(reference);
        // The message stands only while the census says the task needs the owner, so its chip is the owner's move.
        record = { groupId: id, bubble, root, phaseEl, title, reason, controls, timelineEl, reference,
            phaseSecondaryEl: null, row, available: true, finished: false, ownerMove: true };
        records.set(id, record);
        // The time it is first seen: a blocker is a current fact, so it stands at the live edge.
        stampNodeTimestamp(bubble, new Date().toISOString());
        insertMessageNode(bubble);
        return record;
    }

    const canResume = record => connected && record.available && records.get(record.groupId) === record
        && record.row.can_resume === true && record.row.pause_cause !== 'owner';

    function paint(record, row) {
        record.row = row;
        record.title.textContent = row.task_name || row.title || row.project_name || 'Task';
        record.reason.textContent = tx(REASONS[row.attention_reason] || row.reason || row.project_admission_hold?.label || '');
        if (record.reference) nameProjectReference(record.reference, { id: row.project_id, name: row.project_name });
        syncParkedPhase(record, connected ? row.phase : 'unknown', row);
        if (canResume(record) && !record.resume) {
            record.resume = createTaskResumeButton(record.groupId, { canResume: () => canResume(record), onSettled: refresh });
            record.controls.append(record.resume);
        }
        if (record.resume) record.resume.disabled = !canResume(record);
        if (row.cancelable === true && !record.stop) {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'btn btn-xs btn-default ui-hit-area';
            button.textContent = tx(TASK_CONTROL_TRIGGER_LABEL);
            button.addEventListener('click', () => {
                if (!connected || !record.available || record.row.cancelable !== true) return;
                openTaskControlMenu(button, { budgetPaused: record.row.phase === 'budget_paused',
                    // Resume rides the menu where the server allows it and no Resume button of its own stands
                    // beside it (an interrupted owner Resume of a member).
                    resume: record.row.can_resume === true && !canResume(record), onAction: async action => {
                        if (!connected || !record.available || record.row.cancelable !== true) return;
                        try {
                            if (action === ACTION_RESUME) await resumeTaskAction(record.groupId);
                            else if (action === ACTION_HURRY) await hurryTaskAction(record.groupId);
                            else await requestStop(record.groupId, action);
                            await refresh();
                        } catch (error) { showToast(`Task control not confirmed: ${error.message || error}`, 'error'); }
                    } });
            });
            record.stop = button;
            record.controls.append(button);
        }
        if (record.stop) record.stop.disabled = !connected || !record.available || row.cancelable !== true;
        waits.observe(record.groupId, row);
        waits.refresh(record.groupId);
        const waitHost = record.root.querySelector('.model-waits');
        if (waitHost) waitHost.inert = !connected || !record.available;
    }

    // ---- the pill: the oldest waiting target first, then onward, cycling
    // What this tab already saw settle (an answer, a close, a removal from pending) leaves the
    // count at once; the next census only confirms it.
    function stillWaits(row) {
        const seen = owner().observedQuestion?.(row.task_id, row.quiz_id);
        if (!seen) return true;
        // An answer seen here whose delivery a restart left unknown still waits on the owner.
        if (waitsAnswer(row)) return !SETTLED.includes(seen.state)
            || (Boolean(row.project_id) && unknownDeliveryRemovable({ ...row, quiz_state: seen.state, question_delivery: seen.question_delivery }));
        // Only a receipt for this same delivery settles it: the question's publication receipt
        // says nothing about the answer that came later.
        const known = seen.question_delivery;
        return !(known?.kind === row.question_delivery?.kind && (known.dismissed === true || known.state === 'delivered'));
    }
    const identity = t => (t.kind === 'question' ? `q:${t.row.task_id}:${t.row.quiz_id}` : `a:${t.id}`);
    // Question rows carry ISO times, feed nodes epoch milliseconds: both order as milliseconds.
    function targets() {
        const list = questions.filter(stillWaits).map(row => ({ kind: 'question', row, at: Date.parse(row.ts || '') || 0 }));
        for (const [id, record] of records) list.push({ kind: 'action', id, row: record.row, at: Number(record.bubble.dataset.ts) || 0 });
        return list.sort((a, b) => a.at - b.at || identity(a).localeCompare(identity(b)));
    }
    function nodeOf(target) {
        if (target.kind === 'action') return records.get(target.id)?.bubble || null;
        const card = messagesDiv.querySelector(`.chat-quiz-card[data-task-id="${CSS.escape(String(target.row.task_id))}"][data-quiz-id="${CSS.escape(String(target.row.quiz_id))}"]`);
        return card?.closest('.chat-bubble') || card || null;
    }
    // The reading area between the floating header and the composer.
    function view() {
        const box = messagesDiv.getBoundingClientRect();
        const header = page.querySelector('.chat-page-header')?.getBoundingClientRect();
        const composer = toolbar?.getBoundingClientRect();
        return { top: Math.max(box.top, header ? header.bottom : box.top),
            bottom: Math.min(box.bottom, composer && composer.height ? composer.top : box.bottom) };
    }
    function onScreen(node) {
        if (!node?.isConnected) return false;
        const rect = node.getBoundingClientRect(), area = view();
        return rect.bottom > area.top + 24 && rect.top < area.bottom - 24;
    }
    // Where a target stands, or will stand once it mounts at its own time (the feed orders by `data-ts`).
    function placeOf(target) {
        const node = nodeOf(target);
        if (node) return node.getBoundingClientRect().top;
        const later = [...messagesDiv.children].find(el => Number(el.dataset.ts) > target.at);
        return later ? later.getBoundingClientRect().top
            : messagesDiv.getBoundingClientRect().top + messagesDiv.scrollHeight - messagesDiv.scrollTop;
    }
    function nextTarget() {
        const list = targets();
        const away = list.filter(target => !onScreen(nodeOf(target)));
        if (!away.length) return null;
        const at = list.findIndex(target => identity(target) === cursor);
        return away.find(target => list.indexOf(target) > at) || away[0];
    }

    function update() {
        const list = targets();
        const target = list.length ? nextTarget() : null;
        pill.hidden = !target;
        if (!target) return;
        const asked = list.filter(t => t.kind === 'question' && waitsAnswer(t.row)).length;
        const acts = list.length - asked;
        const word = acts === 0 ? (asked === 1 ? 'question' : 'questions')
            : asked === 0 ? (acts === 1 ? 'action' : 'actions') : 'waiting';
        pill.querySelector('.chat-question-nav-count').textContent = String(list.length);
        pill.querySelector('.chat-question-nav-word').textContent = ` ${word}`;
        pill.querySelector('.chat-question-nav-arrow')?.classList.toggle('down', placeOf(target) >= view().bottom);
        const label = acts === 0 ? fmt('{n} unanswered: show the next question', { n: asked })
            : fmt('{n} waiting for you: show the next one', { n: list.length });
        pill.setAttribute('aria-label', label);
        pill.title = label;
    }

    // The card's head lands just under the chat header; the question text takes focus (never the
    // answer field), so going to a question summons no phone keyboard.
    function land(node) {
        const position = reading();
        if (position) { position.claim(); position.stick = false; }
        messagesDiv.scrollTop += node.getBoundingClientRect().top - view().top - 12;
        (node.querySelector('.chat-quiz-question') || node).focus?.({ preventScroll: true });
    }
    pill.addEventListener('click', async () => {
        const target = nextTarget();
        if (!target) return;
        cursor = identity(target);
        let node = nodeOf(target);
        if (!node && target.kind === 'question') {
            // Outside the loaded history: a Project question's copy mounts at its own date; Main's
            // own question opens through its canonical read.
            if (target.row.project_id) {
                owner().appendQuestionPointer?.(target.row, fresh.has(rowKey(target.row)) ? { canonical: true } : { stale: true });
                node = nodeOf(target);
            }
            else { await revealQuestion(target.row.task_id, target.row.quiz_id, { focus: 'question' }); node = nodeOf(target); }
        }
        if (node) land(node);
        requestAnimationFrame(update);
    });
    let queued = false;
    messagesDiv.addEventListener('scroll', () => {
        if (queued) return;
        queued = true;
        requestAnimationFrame(() => { queued = false; update(); });
    }, { passive: true });

    return {
        refresh: update,
        // One state snapshot feeds both halves: the decision owner's canonical questions first, then
        // task controls. `null` is a failed read: nothing current is known, so both become incomplete.
        snapshot(data, requestedAt = Infinity) {
            // HTTP can answer while the socket is down, but frames sent after that body are lost until
            // the reopen, so it restores nothing: only a census the reopen sequences (app.js retires
            // reads begun before it) brings back questions and controls; the disconnect kept the rest.
            if (!connected) return;
            // An install can carry gaps for good (an old question whose room no import proved), so
            // incompleteness counts only for what the gaps name, never the whole list.
            const vouched = vouchedBy(data);
            owner().syncCurrentQuestions?.(data ? data.current_questions : [], vouched, requestedAt);
            const rows = (Array.isArray(data?.current_questions) ? data.current_questions : []).filter(waitsOwner);
            // A partial read keeps what it cannot see: never a count that silently drops to zero.
            fresh = new Set(rows.map(rowKey));
            questions = [...rows, ...questions.filter(row => !fresh.has(rowKey(row)) && !vouched(row.task_id, row.quiz_id))];
            const actions = Array.isArray(data?.main_attention) ? data.main_attention : [];
            const actionsComplete = data?.main_attention_complete === true;
            mutate(() => {
                const current = new Set();
                for (const row of actions) {
                    const id = String(row.task_id || row.activity_id || '');
                    if (!id) continue;
                    current.add(id);
                    const record = recordFor(row);
                    record.available = true;
                    paint(record, row);
                }
                for (const [id, record] of records) if (!current.has(id)) {
                    if (actionsComplete) { waits.forget(id); record.bubble.remove(); records.delete(id); }
                    else { record.available = false; paint(record, record.row); }
                }
                return true;
            });
            update();
        },
        setConnected(value) {
            connected = value;
            if (!value) {
                fresh = new Set();  // what the closed socket last confirmed waits for the next census
                for (const record of records.values()) record.available = false;
            }
            mutate(() => { for (const record of records.values()) paint(record, record.row); return true; });
            owner().setConnected?.(value);  // the question owner withdraws its forms and voids its reads in flight
        },
        destroy() {
            waits.destroy();
            for (const record of records.values()) record.bubble.remove();
            records.clear();
            pill.remove();
        },
    };
}
