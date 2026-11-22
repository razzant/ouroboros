/**
 * The one door from a page to a task's card (docs/DESIGN.md "References and actions").
 *
 * Every place that says "this is that Main task — show me its card" gets its door here, and only this
 * module raises `ouro:open-task`. Main is the default; a run's observed Project room is explicit.
 * app.js lands on the card using around-history when needed, or reports unavailable. The door is inline
 * (`.consciousness-inline-link`, with `.ui-hit-area` for a finger's target).
 */
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { tr } from './i18n.js';

const MAIN_CHAT_ID = 1;

/** The door for an identified task and room, or no door. */
export function taskReferenceHtml(taskId, chatId = MAIN_CHAT_ID) {
    const room = Number(chatId);
    if (!taskId || !Number.isSafeInteger(room) || room < MAIN_CHAT_ID) return '';
    if (room !== MAIN_CHAT_ID) return `<button type="button" class="consciousness-inline-link ui-hit-area" data-open-task="${escapeHtml(taskId)}" data-task-chat="${room}">${escapeHtml(tr('schedule.open_run', 'Open run'))} ↗</button>`;
    return `<button type="button" class="consciousness-inline-link ui-hit-area" data-open-task="${escapeHtml(taskId)}">${escapeHtml(tr('consciousness.open_main', 'Open in Main'))} ↗</button>`;
}

/** A click handler for a region holding doors: a press on one opens its task. True when it was a door's. */
export function openTaskReference(event) {
    const door = event?.target?.closest?.('[data-open-task]');
    if (!door) return false;
    const chatId = Number(door.dataset.taskChat ?? MAIN_CHAT_ID);
    if (Number.isSafeInteger(chatId) && chatId >= MAIN_CHAT_ID) window.dispatchEvent(new CustomEvent('ouro:open-task', { detail: { task_id: door.dataset.openTask, chat_id: chatId } }));
    return true;
}
