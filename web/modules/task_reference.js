/**
 * The one door from a page to a Main task's card (docs/DESIGN.md "References and actions").
 *
 * Every place that says "this is that Main task — show me its card" gets its door here, and only this
 * module raises `ouro:open-task`. app.js answers it by landing on the card Main draws, else by locating it
 * through around-history; a card neither finds is reported as unavailable, never guessed. Main only: a
 * task of another room has no door here. The door is words inside its sentence's own lines
 * (`.consciousness-inline-link`, with `.ui-hit-area` for a finger's target).
 */
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { tr } from './i18n.js';

const MAIN_CHAT_ID = 1;

/** The door's markup for one Main task, or '' without a task. */
export function taskReferenceHtml(taskId) {
    if (!taskId) return '';
    return `<button type="button" class="consciousness-inline-link ui-hit-area" data-open-task="${escapeHtml(taskId)}">${escapeHtml(tr('consciousness.open_main', 'Open in Main'))} ↗</button>`;
}

/** A click handler for a region holding doors: a press on one opens its task. True when it was a door's. */
export function openTaskReference(event) {
    const door = event?.target?.closest?.('[data-open-task]');
    if (!door) return false;
    window.dispatchEvent(new CustomEvent('ouro:open-task', { detail: { task_id: door.dataset.openTask, chat_id: MAIN_CHAT_ID } }));
    return true;
}
