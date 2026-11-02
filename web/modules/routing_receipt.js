// The owner message's routing receipt (docs/DESIGN.md "Chat authorship and System rows": a fact about the owner's
// message is the typed annotation on that message): its words from the typed annotation the host stamps, and the
// one line under the message — what happened to his words, then the Project reference when they went to another
// room. The picker card (chat_decision.js) shares the option labels.
import { projectReference } from './project_reference.js';
import { replySnippet } from './chat_reply_quote.js';
import { ellipsize } from './text_fit.js';
import { fmt, tr, tx } from './i18n.js';

/** Human label for ONE manual-routing option row (shared with the picker card). */
export function routingOptionLabel(option) {
    if (!option || typeof option !== 'object') return '';
    if (option.label) return String(option.label);
    if (option.action === 'new_task_in_project') {
        return fmt('New task in {name}', { name: String(option.project_name || tr('routing.generic_project', 'Project')) });
    }
    if (option.title || option.project_name) {
        return String(option.title || option.project_name);
    }
    return option.project_id && !option.task_id ? tr('routing.generic_project', 'Project') : tr('routing.generic_task', 'Task');
}

/** Human text for a typed routing annotation ('' hides the line). */
export function routingAnnotationText(annotation) {
    if (!annotation || typeof annotation !== 'object') return '';
    // A refused act carries the host's own owner-facing sentence (`cause`);
    // it outranks the status matrix below. Absent on scheduled/delivered/
    // pending rows and on the picker frame, so those labels are unchanged.
    const cause = String(annotation.cause || '').trim();
    if (cause) return tx(cause);
    const action = String(annotation.action || '');
    const status = String(annotation.status || '');
    const target = String(annotation.target || '');
    const targetLabel = String(annotation.target_label || '')
        || (target ? (action === 'project_route' ? tr('routing.generic_project', 'Project') : tr('routing.generic_task', 'Task')) : '');
    if (status === 'pending') return tr('routing.pending', 'Choosing the right destination…');
    if (status === 'needs_manual_target') {
        const optionLabels = (Array.isArray(annotation.options) ? annotation.options : [])
            .map(routingOptionLabel)
            .filter(Boolean);
        if (optionLabels.length) return `${tr('routing.choose_target', 'Choose a target')} · ${optionLabels.join(' / ')}`;
        // No options and (by the guard above) no cause: a receipt written
        // before the host sentence existed, or by a producer that bypasses
        // `_emit_routing_receipt`. Nothing can be chosen on such a row, so it
        // must not invite a choice.
        const notRouted = tr('routing.not_routed', 'Not routed');
        return targetLabel ? `${notRouted} · ${targetLabel}` : notRouted;
    }
    if (status === 'project_unavailable') return tr('routing.project_unavailable', 'Project is unavailable');
    const labels = {
        mailbox_delivery: 'Message saved for task',  // a recorded mailbox write, never a read receipt
        steer_task: 'Steered task',
        promote_chat_to_task: 'Started task',
        route_to_project: 'Routed to project',
        project_route: 'Project routing',
    };
    const label = labels[action] ? tr(`routing.action.${action}`, labels[action])
        : (status.replaceAll('_', ' ') || action.replaceAll('_', ' '));
    return targetLabel && label ? `${label} · ${targetLabel}` : label;
}

/** A reconnect drops the transient `pending` receipts; the next fact writes the line again. */
export function clearTransientRoutingAnnotations(messagesDiv = globalThis.document?.querySelector?.('#chat-messages')) {
    if (!messagesDiv) return false;
    let changed = false;
    for (const note of messagesDiv.querySelectorAll(
        '.msg-routing-annotation[data-annotation-status="pending"]',
    )) {
        const bubble = note.closest('.chat-bubble');
        if (bubble) delete bubble.dataset.chatAnnotationStatus;
        note.remove();
        changed = true;
    }
    return changed;
}

export function renderRoutingAnnotation(bubble, annotation, chatId = 1) {
    if (!bubble) return false;
    const text = routingAnnotationText(annotation);
    let note = bubble.querySelector('.msg-routing-annotation');
    if (!text) {
        const hasStatus = bubble.dataset.chatAnnotationStatus !== undefined;
        if (!note && !hasStatus) return false;
        note?.remove();
        bubble.querySelector('.msg-routing-actions')?.remove();
        if (hasStatus) delete bubble.dataset.chatAnnotationStatus;
        return true;
    }
    const status = String(annotation.status || '');
    const target = annotation.target || (Array.isArray(annotation.target_task_ids) ? annotation.target_task_ids[0] : '') || '';
    const destinationKey = `${annotation.project_id || ''}|${annotation.project_chat_id || ''}|${target}`;
    const changed = !note || note.dataset.annotationText !== text
        || note.dataset.destinationKey !== destinationKey
        || note.dataset.annotationStatus !== status
        || bubble.dataset.chatAnnotationStatus !== status;
    if (!note) {
        note = document.createElement('div');
        note.className = 'msg-routing-annotation';
        const time = bubble.querySelector('.msg-time');
        if (time) time.before(note);
        else bubble.append(note);
    }
    if (!changed) return false;
    bubble.querySelector('.msg-routing-actions')?.remove();
    // The words went to another room: a named room chat, else (an annotation without one) any Project seen from Main.
    // The work reading them (`read`) keeps the same destination: the receipt only grows stronger.
    const room = Number(annotation.project_chat_id) || 0;
    const destination = annotation.project_id && (room > 0 ? room !== Number(chatId) : Number(chatId) === 1)
        && ['scheduled', 'delivered', 'read'].includes(status);
    note.dataset.annotationText = text;
    note.dataset.annotationAction = String(annotation.action || '');
    note.dataset.destinationKey = destinationKey;
    if (note.dataset.annotationStatus !== status) note.dataset.annotationStatus = status;
    if (bubble.dataset.chatAnnotationStatus !== status) bubble.dataset.chatAnnotationStatus = status;
    note.classList.toggle('has-reference', Boolean(destination));
    if (destination) {
        // The receipt names the room by the Project reference, on the receipt's own line: what happened to the owner's
        // words, then where (never the internal act name, never a second padded row).
        note.replaceChildren(Object.assign(document.createElement('span'), { textContent: routingLeadText(annotation) }),
            projectReference({ id: annotation.project_id, chat_id: room || undefined, name: annotation.project_name },
                { taskId: target }));
    } else {
        note.textContent = text;
    }
    return changed;
}

// The words before the Project reference on an owner message's receipt: what happened to them. Words passed on to
// a live work name it by its title when the host names it (`target_titles`).
const ROUTING_LEAD = {
    promote_chat_to_task: ['routing.lead.started', 'Started as a task in'],
    route_to_project: ['routing.lead.routed', 'Routed to'],
    project_route: ['routing.lead.routed', 'Routed to'],
};
const PASSED_ON = new Set(['steer_task', 'tell_task', 'mailbox_delivery']);
// A work's name on the line, within a deliberate display budget (80 characters; presentation only, never a rule for
// what a name may be): its first readable line cut on a word, any cut said by the ellipsis. The stored name stays
// whole; an older host stored a nameless work's whole text here, so only its opening words show.
const SHOWN_NAME_CHARS = 80;
function shownWorkName(title) {
    const shown = replySnippet(title, SHOWN_NAME_CHARS);
    return shown && !shown.endsWith('…') && /\n\s*\S/.test(String(title ?? '').trim()) ? ellipsize(shown) : shown;
}
function routingLeadText(annotation) {
    const action = String(annotation.action || '');
    if (PASSED_ON.has(action)) {
        const titles = (Array.isArray(annotation.target_titles) ? annotation.target_titles : [])
            .map(shownWorkName).filter(Boolean);
        return titles.length ? fmt('Passed on to {works} in', { works: titles.map((title) => `«${title}»`).join(', ') })
            : tr('routing.lead.passed_on', 'Passed on to the work in');
    }
    const [code, english] = ROUTING_LEAD[action] || ['routing.lead.sent', 'Sent to'];
    return tr(code, english);
}
