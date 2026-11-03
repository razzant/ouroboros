/**
 * The one way the UI points at a Project (docs/DESIGN.md "References and actions").
 *
 * Every place that says "this belongs to that Project — take me there" gets its
 * control here, and only this module raises `ouro:open-project`. Callers choose a
 * LAYOUT, one of two forms; they never write a label, a glyph or a class name,
 * because that freedom is how one intent came to be drawn several ways while each
 * call still "reused the shared chip".
 *
 *   inline  the capsule wherever content points at a Project: under a message or a
 *           System row, at the end of a routing receipt, in a mirrored question's voice line
 *   footer  the capsule that leads a work portal's action row (a task card bound to a Project)
 */
import { PAGE_ICONS } from './page_icons.js';
import { tr } from './i18n.js';

// Two forms only: anything else a caller names is the inline capsule. The arrow alone says "opens". The capsule
// keeps a finger's target wherever its row makes it shorter (a question's 28px voice line, a portal's action row):
// `.ui-hit-area`.
const LAYOUT_CLASS = { inline: 'chat-quiz-project ui-hit-area', footer: 'chat-live-bound-pointer ui-hit-area' };

// What a press opens, per mounted reference: a name that arrives later reaches the payload too.
const targets = new WeakMap();
// The title a reference stands beside (a work portal's own title), per mounted reference.
const besides = new WeakMap();

// A row that was never named carries its minted id as the name. Only that shape is a non-name:
// an owner's own `blog` has the id `blog` too, and it IS the name.
const MINTED_ID = /^proj_[0-9a-f]+$/;
function displayName(project) {
    const name = String(project?.name || '').trim();
    return name && !(name === String(project?.id || '') && MINTED_ID.test(name)) ? name : tr('project.reference.generic', 'Project');
}

// Beside a title that already says the Project's name, the capsule shows `Project` rather than that name a second
// time (owner 26.10, B: a work portal named like its Project); a differently named Project keeps its name there.
const words = (text) => String(text || '').replace(/\s+/g, ' ').trim().toLocaleLowerCase();
function shownName(node, name) {
    const beside = words(besides.get(node)?.textContent);
    return beside && beside === words(name) ? tr('project.reference.generic', 'Project') : name;
}

/** One name for the tooltip and the accessible name, so they cannot disagree; the visible text is that name, or
 *  `Project` beside a title that already says it. Without `project` it names the reference again from its own
 *  target: what the owner of the title beside it calls after renaming that title. */
export function nameProjectReference(node, project = targets.get(node)) {
    const name = displayName(project);
    const target = targets.get(node);
    if (target && project?.name) target.name = project.name;
    const label = node.querySelector('.chat-live-project-name');
    const shown = shownName(node, name);
    if (label && label.textContent !== shown) label.textContent = shown;
    const spoken = node.dataset.opensQuestion ? `Open this question in ${name}` : `Open project ${name}`;
    node.title = spoken;
    node.setAttribute('aria-label', spoken);
    return node;
}

/** `beside` is the title element the reference stands next to, if any; the reference holds no watch on it. */
export function projectReference(project, { layout = 'inline', taskId = '', quizId = '', beside = null } = {}) {
    const target = { ...(project || {}) };
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `chat-live-project-card-btn ${LAYOUT_CLASS[layout] || LAYOUT_CLASS.inline}`;
    btn.dataset.intent = 'open-project';
    if (quizId) btn.dataset.opensQuestion = '1';
    const icon = document.createElement('span');
    icon.className = 'chat-live-project-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.innerHTML = PAGE_ICONS.projects;
    const nameEl = document.createElement('span');
    nameEl.className = 'chat-live-project-name';
    const statusEl = document.createElement('span');
    statusEl.className = 'chat-live-project-status';
    statusEl.setAttribute('aria-hidden', 'true');
    statusEl.textContent = '↗';
    btn.append(icon, nameEl, statusEl);
    targets.set(btn, target);
    if (beside) besides.set(btn, beside);
    nameProjectReference(btn, target);
    // A press with a pointer (a click's detail counts it; a key's click has none) opens the room quietly.
    btn.addEventListener('click', (event) => window.dispatchEvent(new CustomEvent('ouro:open-project', {
        detail: { project: { ...target }, task_id: String(taskId || ''), quiz_id: String(quizId || ''),
            pointer: event.detail > 0 },
    })));
    return btn;
}
