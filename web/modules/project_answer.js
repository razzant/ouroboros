/**
 * Project lifecycle rows in Main (docs/DESIGN.md "Project completion mirror").
 *
 * A Project root that ended with Ouroboros's own final answer reaches Main as an
 * ordinary Ouroboros message, folded when it is long. Every other lifecycle row (a
 * start, an ending with no answer, a row written before the answer rode the row)
 * keeps its System text; an ending with no answer whose row carries the host's typed
 * view reads compactly from it. Either way the row ends with the same
 * Project reference: the voice of a row never chooses how the UI points at its Project.
 */
import { batch, measure } from './layout_batch.js';
import { createSystemMessageActions } from './ui_helpers.js';
import { tr } from './i18n.js';
import { reasonCodesNote, taskPresentation } from './log_events.js';
import { projectReference } from './project_reference.js';
import { outcomeTone } from './task_phase_chip.js';
import { terminalTimeNote } from './utils.js';

export const PROJECT_ROW_TYPES = new Set(['project_started', 'project_handoff', 'project_completion_summary']);

// Host-stamped Project lifecycle rows (§3 Main rows), live and replayed alike:
// a mirrored final answer is an ordinary assistant message, the rest System text.
export function appendProjectRow(addMessage, msg, text, opts) {
    const a = msg.completion_answer;
    return addMessage(a || text, a ? 'assistant' : 'system', !!(a || msg.markdown), msg.ts || null, false, {
        ...opts, systemType: msg.system_type, narration: msg.narration, projectId: msg.project_id || '',
        projectName: msg.project_name || '', taskName: msg.task_name || '', handoffId: msg.handoff_id || '', terminalTime: msg.terminal_time || null,
        taskId: msg.task_id || opts?.taskId || '', completionLimitation: msg.completion_reason || '', deliveryId: msg.delivery_id || '',
        completionNotice: msg.completion_notice || null,
    });
}

// The fade says "there is more": compare with the seven-line footprint, never the extended
// painted box. Otherwise answers between seven and nine lines alternate on every refresh.
// A row mounted while Chat is not the visible page has no layout box (clientHeight 0): it is left alone and
// measured when Chat is shown; a resize re-measures, because the width decides the height. Answers refreshed in one
// batch (layout_batch.js: the reading owner's write, a frame's re-measure) are measured together, then marked together,
// then their heights read together before any margin is written: one layout serves each step for all of them.
export function refreshProjectAnswerFold(bubble) {
    if (!bubble?.classList.contains('project-answer')) return;
    measure(() => measureFold(bubble), (seen) => {
        if (!seen) return;
        const { message, fold, folded } = seen;
        bubble.classList.toggle('is-folded', folded);
        syncFold(bubble, message, folded);
        // Compensate only the actual extension: a 180px answer borrows 12px, not the full 48px tail.
        measure(() => (folded ? Math.max(0, parseFloat(getComputedStyle(message).height) - fold) : 0),
            (tail) => message.style.setProperty('--project-answer-compensation', `${-tail}px`));
    });
}
function measureFold(bubble) {
    const message = bubble.querySelector('.message');
    if (!message || !message.clientHeight) return null;
    const style = getComputedStyle(message);
    const fold = parseFloat(style.lineHeight) * parseFloat(style.getPropertyValue('--project-answer-lines'));
    return { message, fold, folded: message.scrollHeight > fold + 1 };
}

// Folded text offers no keyboard stops (pointer clicks still reach visible links): the Project
// reference leads to the complete answer. The table enhancer respects the fold even
// when it registers later; removing its tabindex lets it restore normal keyboard use.
// Browsers also tab into overflow boxes without an explicit tabindex.
const FOCUSABLE = 'a[href], button, input, select, textarea, summary, [tabindex], pre, .katex-display, .md-mermaid, .md-chart';
function syncFold(bubble, message, collapsed) {
    for (const node of message.querySelectorAll(collapsed ? FOCUSABLE : '[data-fold-stop], .md-table-wrap[tabindex="-1"]')) {
        node.toggleAttribute('data-fold-stop', collapsed);
        if (collapsed) node.setAttribute('tabindex', '-1');
        else node.removeAttribute('tabindex');
    }
}

// One set of window listeners for the app's lifetime, however many rows exist: a row holds
// only listeners on its own nodes, so a released row needs no disposer here.
let watching = false;
let queued = false;
function afterLayout(work) {
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(work);
}
// A feed whose width is still moving lays out only the rows near the screen (chat_reading_position.js marks it
// [data-resizing]); measuring a row it skips would lay that row out again, on every frame of the move. Its rows are
// measured once the width holds. They are measured a frame after the change: a new width wakes the rows that sleep
// in the observers of the frame it is laid out in, after that frame's animation callbacks.
function refreshFoldsSoon() {
    if (queued) return;
    queued = true;
    afterLayout(() => afterLayout(() => {
        queued = false;
        // A sleeping row ([data-laid], chat_row_sleep.js) keeps the fold it was measured with at this width: a row that
        // changes wakes, a new width wakes every row, and a feed that is not shown puts none to sleep. Measuring it
        // would lay it out alone. The rest are measured together, then marked together: one layout serves them all.
        const due = [...document.querySelectorAll('.chat-bubble.project-answer')]
            .filter((bubble) => !bubble.closest('[data-resizing], [data-laid]'));
        batch(() => due.forEach(refreshProjectAnswerFold));
    }));
}
function watchFolds() {
    if (watching || typeof window === 'undefined') return;
    watching = true;
    window.addEventListener('ouro:page-shown', refreshFoldsSoon);
    window.addEventListener('resize', refreshFoldsSoon);
    window.addEventListener('ouro:feed-width-settled', refreshFoldsSoon);
}

export function decorateProjectRow(bubble, { role = 'system', projectId = '', projectName = '', taskName = '', terminalTime, addedAt,
    systemType = '', completion = systemType === 'project_completion_summary', completionNotice = null,
    completionLimitation = '', limitation = completionLimitation, fold = role === 'assistant' } = {}) {
    if (!bubble || !projectId) return null;
    if (fold) {
        bubble.classList.add('project-answer');  // measured when its markdown is written (chat.js enhanceMountedMarkdown)
        watchFolds();
    }
    const actions = createSystemMessageActions(projectReference({ id: projectId, name: projectName }));
    const message = bubble.querySelector('.message');
    message?.after(actions);
    if (completion) {
        const notice = endingNotice(completionNotice);
        if (role === 'system' && message && notice && taskName && outcomeTone(notice.phase)) {
            paintEnding(message, notice, taskName === projectName ? '' : taskName);
        } else if (limitation && role === 'assistant') {
            const shown = notice?.cause || limitation;
            const warning = bubble.ownerDocument.createElement('div');
            warning.className = 'msg-provenance project-answer-limitation';
            warning.textContent = shown;
            warning.title = shown;
            bubble.insertBefore(warning, actions);
            const details = recordedDetails(bubble.ownerDocument, notice?.codes);
            if (details) bubble.insertBefore(details, actions);
        }
        paintTerminalTime(bubble, terminalTimeNote(terminalTime, addedAt));
    }
    return actions;
}

// The host's typed view of a Project ending. Older rows without cause keep their retained text, never parsed into fields.
function endingNotice(notice) {
    if (!notice || typeof notice !== 'object' || typeof notice.cause !== 'string') return null;
    return {
        phase: String(notice.phase || ''), cause: notice.cause,
        salvage: typeof notice.salvage === 'string' ? notice.salvage : '',
        codes: Array.isArray(notice.undescribed) ? notice.undescribed.map(String).filter(Boolean) : [],
    };
}

// The status sentence's tones (ui.css `.ui-status`) for the chips' outcome hues (status_mark.js).
const ENDING_TONES = { ok: 'ok', failed: 'error' };

// An ending with no authored answer (docs/DESIGN.md "Project completion mirror"): the work's name and true outcome,
// its cause, the recorded details and the labelled preserved output, in the System voice; the Project is named once,
// by the reference under the row. Display only: Copy, the row's keys and history keep its canonical text.
function paintEnding(message, notice, title) {
    const doc = message.ownerDocument;
    const node = (tag, className, text) => {
        const el = doc.createElement(tag);
        el.className = className;
        el.textContent = text;
        return el;
    };
    const head = node('div', 'project-ending-head', '');
    if (title) head.append(node('span', 'project-ending-title', title), doc.createTextNode(' '));
    const outcome = node('span', 'ui-status', taskPresentation(notice.phase).headline);
    outcome.dataset.tone = ENDING_TONES[outcomeTone(notice.phase)] || 'neutral';
    head.append(outcome);
    const parts = [head, notice.cause && node('p', 'project-ending-cause', notice.cause), recordedDetails(doc, notice.codes),
        notice.salvage && node('p', 'project-ending-salvage', notice.salvage)].filter(Boolean);
    message.classList.add('project-ending');
    // Line breaks between the parts keep the row's words readable where only its text is read (a Reply's snippet).
    message.replaceChildren(...parts.flatMap((part, index) => (index ? [doc.createTextNode('\n'), part] : [part])));
}

// The cause line says a typed code has no sentence yet; its exact code opens here by pointer, touch and keyboard
// (a native disclosure), and stays on the record.
function recordedDetails(doc, codes) {
    const note = reasonCodesNote(codes);
    if (!note) return null;
    const details = doc.createElement('details');
    details.className = 'project-ending-details';
    const summary = doc.createElement('summary');
    summary.textContent = tr('project.ending.recorded_details', 'Recorded details');
    const line = doc.createElement('p');
    line.textContent = note;
    details.append(summary, line);
    return details;
}

// One compact time line (owner 17.10, Q7 A): the row's own time also says when its task ended, both facts only
// when their minutes differ; the full instants are its tooltip and what a screen reader hears.
function paintTerminalTime(bubble, { short, full }) {
    const doc = bubble.ownerDocument;
    const time = bubble.querySelector('.msg-time') || bubble.appendChild(doc.createElement('div'));
    time.className = 'msg-time';
    time.title = full;
    const shown = doc.createElement('span');
    shown.setAttribute('aria-hidden', 'true');
    shown.textContent = short;
    const spoken = doc.createElement('span');
    spoken.className = 'msg-time-full';
    spoken.textContent = full;
    time.replaceChildren(shown, spoken);
}

/** Kept beside the authored body; adoption never replaces the selected text. */
export function syncSavedProjectContext(bubble, projected, originId = '') {
    if (bubble && originId) bubble.dataset.originId = originId;
    if (!bubble || (projected && bubble.dataset.historyId)) return false;
    const prior = bubble.querySelector('.saved-project-context');
    if (!projected) {
        if (!prior) return false;
        prior.remove(); delete bubble.dataset.originProjected;
        return true;
    }
    if (prior) return false;
    const note = bubble.ownerDocument.createElement('div');
    note.className = 'msg-provenance saved-project-context';
    note.textContent = tr('project.saved_context_note', 'Saved project context — original request retained here; surrounding history may still need loading.');
    bubble.insertBefore(note, bubble.querySelector('.message')?.nextSibling || null);
    bubble.dataset.originProjected = '1';
    return true;
}
