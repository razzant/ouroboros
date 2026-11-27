// What waits on the owner in Main has no section of its own (DESIGN "Project question
// mirror"): questions keep their place in the feed, and a task that cannot continue without
// the owner is one System message there with the task's own controls. One amber pill among the
// composer's pills reaches whichever of them is off screen. It owns no polling or saved state:
// `/api/state` names the canonical questions and actions, the feed holds their cards. The work Panic
// stopped is one System message more, the owner's choice of what to resume (panic_recovery.js); a work
// whose card offers only that Resume keeps no second card here — the message's row is its card, with its
// Stop…, and the message the pill's one count — while a card for anything else, or a work the list cannot
// name, stays the owner's door.
import { createModelWaitController } from './model_wait.js';
import { createPanicRecovery, panicCards } from './panic_recovery.js';
import { readBand } from './project_read_state.js';
import { projectReference, nameProjectReference } from './project_reference.js';
import { closeExpiredOffer, createTaskCloseButton, createTaskResumeButton, openTaskControlMenu, requestStop, hurryTaskAction, resumeEligible, resumePending,
    resumeTaskAction, ACTION_RESUME, ACTION_HURRY, TASK_CONTROL_TRIGGER_LABEL } from './task_control_menu.js';
import { syncParkedPhase } from './task_phase_chip.js';
import { indexDeadlineExpired, memberState, reasonText, resumeAttemptWords, workLine, workOfEpisode } from './work_status.js';
import { ANSWERABLE_QUIZ_STATES, questionWaits, unknownDeliveryRemovable, vouchedBy } from './question_presentation.js';
import { createSystemMessageActions } from './ui_helpers.js';
import { showToast } from './toast.js';
import { fmt, tx } from './i18n.js';
import { normalizeConsciousness } from './consciousness_api.js';
import { allowanceHold, heldWake, wakeResumeRoomWords } from './consciousness_words.js';

const GLYPH = '<svg class="chat-question-nav-glyph" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>';
const ARROW = '<svg class="chat-question-nav-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>';
// An unanswered question stays current after its author ends (until answered or closed). A late
// answer whose delivery a restart left unknown waits on the owner too: only they can remove it
// from pending, and its card offers that.
const SETTLED = ['answered', 'closed', 'superseded'];
const waitsAnswer = row => ANSWERABLE_QUIZ_STATES.includes(row.quiz_state);
const waitsOwner = questionWaits;  // the one rule the rooms' amber follows too (question_presentation.js)
const rowKey = row => `${row.task_id}:${row.quiz_id}`;
let nameSeq = 0;
const REASONS = { budget: 'Paused at the spending limit. Change the limit if needed, then Resume.',
    restart: 'Needs Resume after the restart.', resume: 'Needs an explicit Resume.',
    model_access: 'Needs you to restore model access.' };
// The spending limit named, where the same body proves which one held its own wake.
const TOTAL_BUDGET_REASON = 'Paused: the total budget was spent. Raise Total budget in Costs, then Resume.';

// The wake the same body names as held: at its daily share, what its Resume may add (in Costs' words), said beside
// that Resume on its card; by the total budget, that budget as its reason. Any other task, a wake held for another cause
// and a host that does not say add nothing.
function heldWakeRoom(data) {
    const bg = data?.bg_consciousness_state;
    if (!bg || typeof bg !== 'object') return null;
    const wake = normalizeConsciousness(bg, { enabled: data.bg_consciousness_enabled });
    const words = allowanceHold(wake) ? wakeResumeRoomWords(wake) : '';
    const reason = heldWake(wake) && wake.holdCause === 'total_budget' ? TOTAL_BUDGET_REASON : '';
    return (words || reason) && wake.liveWakeTaskId ? { taskId: wake.liveWakeTaskId, words, reason } : null;
}

// `reading` returns the chat's reading position (chat_reading_position.js), read when used like the
// question owner: a landing claims it, as any other navigation does, so the feed stops following the live edge.
export function createMainWaiting({ page, messagesDiv, toolbar = page?.querySelector('.chat-toolbar-row'), insertMessageNode,
    stampNodeTimestamp, openSettings, refresh = () => {}, mutate = fn => fn(), decision = () => null,
    revealQuestion = () => {}, reading = () => null, noticeVisible }) {
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
    // A work's `Stop…` where it waits on the owner: its card here, or its row in the Panic choice, whose box is its
    // Resume (so that menu offers none). Every effect re-checks `allowed` at press time.
    function stopControl(id, { row, allowed, resume = () => false }) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-sm btn-default ui-hit-area';
        button.textContent = tx(TASK_CONTROL_TRIGGER_LABEL);
        button.addEventListener('click', () => {
            if (!allowed()) return;
            openTaskControlMenu(button, { budgetPaused: row()?.phase === 'budget_paused', resume: resume(),
                onAction: async action => {
                    if (!allowed()) return;
                    try {
                        // A Resume keeps its task pending until this re-read has settled (the request owner holds it).
                        if (action === ACTION_RESUME) { await resumeTaskAction(id, { settle: refresh }); return; }
                        if (action === ACTION_HURRY) await hurryTaskAction(id);
                        else await requestStop(id, action);
                        await refresh();
                    } catch (error) { showToast(`Task control not confirmed: ${error.message || error}`, 'error'); }
                } });
        });
        return button;
    }
    const panic = createPanicRecovery({ insertMessageNode, stampNodeTimestamp, mutate, refresh, onChange: () => update(),
        stopControl, ...(noticeVisible ? { isVisible: noticeVisible } : {}) });

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
        // The message stands only while the census says the task needs the owner, so its chip is the owner's move.
        record = { groupId: id, bubble, root, phaseEl, title, reason, controls, timelineEl, reference: null,
            phaseSecondaryEl: null, row, available: true, finished: false, ownerMove: true };
        records.set(id, record);
        // The time it is first seen: a blocker is a current fact, so it stands at the live edge.
        stampNodeTimestamp(bubble, new Date().toISOString());
        insertMessageNode(bubble);
        return record;
    }

    const canResume = record => connected && record.available && records.get(record.groupId) === record
        && record.row.can_resume === true && record.row.pause_cause !== 'owner' && !record.expired;

    function paint(record, row) {
        if (record.reference && record.row.project_id !== row.project_id) {
            record.reference.remove();
            record.reference = null;
        }
        record.row = row;
        const close = closeExpiredOffer(row.close_expired, record.groupId);
        // The status (deadline passed) and the offer (Close) are separate facts; either way nothing waits on the owner.
        const expired = Boolean(close) || row.deadline_expired === true || row.attention_reason === 'deadline_expired';
        record.expired = expired;
        // A distinct request on the same task (a sign-in or model access it waits for) still needs the owner.
        record.ownerMove = !expired || Object.keys(row.model_waits || {}).length > 0 || row.attention_reason === 'model_access';
        if (record.close && record.close.offer !== close) { record.close.remove(); record.close = null; }
        if (close && !record.close) {
            record.title.id ||= `owner-action-name-${nameSeq += 1}`;
            record.close = createTaskCloseButton(record.groupId, { onSettled: refresh, nameId: record.title.id,
                tokenNow: () => connected && record.available && closeExpiredOffer(record.row.close_expired, record.groupId) === close ? close : '',
            });
            record.close.offer = close;
            record.controls.append(record.close);
        }
        if (record.close) record.close.disabled = !connected || !record.available;
        record.title.textContent = row.suggested_name || row.task_name || row.title || row.project_name || tx('Task');
        // What it needs, then its latest refused Resume as dated history in the door's words (never a second message).
        const attempt = resumeAttemptWords(row.last_resume);
        // A named stop explains the button. A plain Resume needs no second instruction.
        // Past its deadline the pause keeps its dated author (the work line's own words), never a wait for Resume.
        const needs = expired && row.attention_reason !== 'model_access' ? row.expiredLine || ''
            : row.attention_reason === 'resume' && row.can_resume === true && row.pause_cause !== 'owner' ? reasonText(row.resumeReason)
                : row.attention_reason === 'budget' && row.wakeReason ? tx(row.wakeReason)
                    : tx(REASONS[row.attention_reason] || row.reason || row.project_admission_hold?.label || '');
        const head = needs && attempt && !/[.!?…]$/.test(needs) ? `${needs}.` : needs;
        // Only a wake that can still Resume has a prospective allowance.
        const room = !expired && row.wakeRoom ? `${row.wakeRoom}.` : '';
        record.reason.textContent = [head, room, attempt && (/[.!?…]$/.test(attempt) ? attempt : `${attempt}.`)].filter(Boolean).join(' ');
        record.reason.hidden = !record.reason.textContent;
        if (row.project_id) {
            const project = { id: row.project_id, name: row.project_name };
            if (record.reference) nameProjectReference(record.reference, project);
            else {
                record.reference = projectReference(project, { taskId: record.groupId, beside: record.title });
                record.controls.prepend(record.reference);
            }
        }
        // A row no census vouches for now (kept from an earlier read, or the socket is down) has no confirmed
        // place: its Project reference leads to no room until a census names it again.
        if (record.reference) record.reference.disabled = !record.available;
        syncParkedPhase(record, connected ? row.phase : 'unknown', row);
        if (expired && record.resume) { record.resume.remove(); record.resume = null; }
        if (canResume(record) && !record.resume && !expired) {
            record.resume = createTaskResumeButton(record.groupId, { canResume: () => canResume(record), onSettled: refresh });
            record.controls.append(record.resume);
        }
        if (record.resume) {
            record.resume.hidden = row.can_resume !== true || row.pause_cause === 'owner';
            // A repaint reads the task's one pending fact with the census's authority, never the census alone: a read
            // taken before a press still names the task resumable until the read that follows the press says otherwise.
            record.resume.disabled = !resumeEligible(record.groupId, () => canResume(record));
        }
        // A Stop an earlier read granted leaves with that grant once the deadline has passed; it never lingers disabled.
        if (expired && row.cancelable !== true && record.stop) { record.stop.remove(); record.stop = null; }
        if (row.cancelable === true && !record.stop) {
            record.stop = stopControl(record.groupId, { row: () => record.row,
                allowed: () => connected && record.available && record.row.cancelable === true,
                // Resume rides the menu where the server allows it and no Resume button of its own stands beside it
                // (an interrupted owner Resume of a member), and nothing of the task is pending — a request of its
                // own, or the read that reconciles a Resume it just answered.
                resume: () => record.row.can_resume === true && !record.expired && !canResume(record) && !resumePending(record.groupId) });
            record.controls.append(record.stop);
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
    const identity = t => (t.kind === 'question' ? `q:${t.row.task_id}:${t.row.quiz_id}` : t.kind === 'panic' ? 'p:' : `a:${t.id}`);
    // Question rows carry ISO times, feed nodes epoch milliseconds: both order as milliseconds.
    function targets() {
        const list = questions.filter(stillWaits).map(row => ({ kind: 'question', row, at: Date.parse(row.ts || '') || 0 }));
        // A card that only offers Close (its deadline passed) needs nothing from the owner: it is never counted as waiting.
        for (const [id, record] of records) if (record.ownerMove !== false) list.push({ kind: 'action', id, row: record.row, at: Number(record.bubble.dataset.ts) || 0 });
        const stopped = panic.target();  // the Panic choice is one action, however many works it names
        if (stopped) list.push({ kind: 'panic', id: '', row: {}, at: stopped.at });
        return list.sort((a, b) => a.at - b.at || identity(a).localeCompare(identity(b)));
    }
    function nodeOf(target) {
        if (target.kind === 'panic') return panic.node();
        if (target.kind === 'action') return records.get(target.id)?.bubble || null;
        const card = messagesDiv.querySelector(`.chat-quiz-card[data-task-id="${CSS.escape(String(target.row.task_id))}"][data-quiz-id="${CSS.escape(String(target.row.quiz_id))}"]`);
        return card?.closest('.chat-bubble') || card || null;
    }
    // The reading area between the floating header (with its frosted edge) and the composer.
    function view() {
        return readBand(messagesDiv, page.querySelector('.chat-page-header'), toolbar);
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
    // answer field), so going to a question summons no phone keyboard. An action's landing focuses its
    // explicit Resume where one stands (owner 18.11, A: the pill leads to the System row's own door);
    // focusing never sends the operation.
    function land(node, door = null) {
        const position = reading();
        if (position) { position.claim(); position.stick = false; }
        messagesDiv.scrollTop += node.getBoundingClientRect().top - view().top - 12;
        (door || node.querySelector('.chat-quiz-question') || node).focus?.({ preventScroll: true });
    }
    // The action target's own Resume button while a press of it would act and can be made now: the owner's
    // current authority (`resumeEligible`) AND a shown, enabled button (a press keeps its own disabled until its
    // census read settles, and a browser focuses no disabled button); otherwise the message takes the landing.
    function resumeDoor(target) {
        // The Panic choice's next step is choosing: its landing focuses the first box that can be chosen.
        if (target.kind === 'panic') return panic.door();
        const record = target.kind === 'action' ? records.get(target.id) : null;
        const door = record?.resume;
        return door && !door.hidden && !door.disabled && resumeEligible(record.groupId, () => canResume(record)) ? door : null;
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
        if (node) land(node, resumeDoor(target));
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
            // The same census names every Panic-held work in both projections (state.py `_panic_recovery`; main_decisions
            // offers its Resume): its card here would be a second card beside the choice's one message and a second pill
            // count of one decision. Only a card that offers just that Resume, of a row the message can act on (its task
            // and the stop it names), is covered (`panicCards`; the row carries its Stop…). A card for anything else (a
            // spending limit, a restart, model access) stays its own; a list that cannot name a work (absent,
            // unreadable, or a row without its hold) leaves its card as the owner's only door here.
            const panicCovered = panicCards(data);
            const wake = heldWakeRoom(data);
            const expiredIds = indexDeadlineExpired(data);
            mutate(() => {
                const current = new Set();
                for (const action of actions) {
                    const id = String(action.task_id || action.activity_id || '');
                    if (!id || panicCovered.has(id)) continue;
                    current.add(id);
                    // The same body's work projection says the task's latest refused Resume (`last_resume`).
                    const member = memberState(data, id);
                    const work = workOfEpisode(data, id);
                    const source = member || work;
                    const late = expiredIds.has(id);
                    const row = { ...action, deadline_expired: action.deadline_expired === true || late,
                        // Omit `last_resume` from `expiredLine`: `paint` appends the full dated refusal words once.
                        ...(late && source ? { expiredLine: workLine({ ...source, last_resume: undefined }) } : {}),
                        last_resume: member?.last_resume || work?.last_resume || action.last_resume,
                        resumeReason: !action.project_id ? member?.because?.[0] : null,
                        wakeRoom: wake?.taskId === id ? wake.words : '', wakeReason: wake?.taskId === id ? wake.reason : '' };
                    const record = recordFor(row);
                    record.available = true;
                    paint(record, row);
                }
                for (const [id, record] of records) if (!current.has(id)) {
                    // The choice's message names this work: its card leaves on this read alone, whatever this read
                    // says of the rest of the attention list.
                    if (panicCovered.has(id)) { waits.forget(id); record.bubble.remove(); records.delete(id); continue; }
                    if (actionsComplete) { waits.forget(id); record.bubble.remove(); records.delete(id); }
                    else { record.available = false; paint(record, record.row); }
                }
                return true;
            });
            panic.snapshot(data, requestedAt);
            update();
        },
        setConnected(value) {
            connected = value;
            if (!value) {
                fresh = new Set();  // what the closed socket last confirmed waits for the next census
                for (const record of records.values()) record.available = false;
            }
            mutate(() => { for (const record of records.values()) paint(record, record.row); return true; });
            panic.setConnected(value);
            owner().setConnected?.(value);  // the question owner withdraws its forms and voids its reads in flight
        },
        // Main's Panic notice keeps the room seam's rules (chat.js passes it to the history controls).
        notice: panic.notice,
        destroy() {
            panic.destroy();
            waits.destroy();
            for (const record of records.values()) record.bubble.remove();
            records.clear();
            pill.remove();
        },
    };
}
