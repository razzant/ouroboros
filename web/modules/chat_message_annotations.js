// Routing acknowledgements and delivery marks on a chat instance's own user
// bubbles. A routing ack is a compact sidecar keyed by client_message_id: it
// updates the existing owner message in place — a one-line note above its
// timestamp — and never adds a synthetic assistant bubble. The transcript
// column and the pending-bubble registry are handed over explicitly, so a Main
// chat and a Project panel annotate only their own messages.
export function createMessageAnnotations({ messagesDiv, pendingUserBubbles, localEchoJournal = null }) {
    function routingAnnotationText(annotation) {
        if (!annotation || typeof annotation !== 'object') return '';
        const action = String(annotation.action || '');
        const status = String(annotation.status || '');
        const target = String(annotation.target || '');
        if (status === 'pending') return 'Choosing the right destination…';
        if (status === 'needs_manual_target') {
            const optionLabels = (Array.isArray(annotation.options) ? annotation.options : [])
                .map(option => {
                    if (!option || typeof option !== 'object') return '';
                    if (option.label) return String(option.label);
                    if (option.action === 'new_task_in_project') {
                        return `New task in ${String(option.project_name || 'Project')}`;
                    }
                    return String(option.title || option.task_id || option.project_name || option.project_id || '');
                })
                .filter(Boolean);
            if (optionLabels.length) return `Choose a target · ${optionLabels.join(' / ')}`;
            return target ? `Choose a target · ${target}` : 'Choose a target';
        }
        if (status === 'project_unavailable') return 'Project is unavailable';
        const labels = {
            mailbox_delivery: 'Delivered to task',
            steer_task: 'Steered task',
            promote_chat_to_task: 'Started task',
            route_to_project: 'Routed to project',
            project_route: 'Project routing',
        };
        const label = labels[action] || status.replaceAll('_', ' ') || action.replaceAll('_', ' ');
        return target && label ? `${label} · ${target}` : label;
    }

    function renderRoutingAnnotation(bubble, annotation) {
        if (!bubble) return false;
        const text = routingAnnotationText(annotation);
        let note = bubble.querySelector('.msg-routing-annotation');
        if (!text) {
            note?.remove();
            delete bubble.dataset.chatAnnotationStatus;
            return false;
        }
        if (!note) {
            note = document.createElement('div');
            note.className = 'msg-routing-annotation';
            const time = bubble.querySelector('.msg-time');
            if (time) time.before(note);
            else bubble.append(note);
        }
        const status = String(annotation.status || '');
        note.textContent = text;
        note.dataset.annotationStatus = status;
        bubble.dataset.chatAnnotationStatus = status;
        return true;
    }

    function updateMessageAnnotation(clientMessageId, annotation) {
        const messageId = String(clientMessageId || '');
        if (!messageId) return false;
        // The journal copy carries the ack, so a re-render restores it too.
        const journalEntry = localEchoJournal?.get(messageId);
        if (journalEntry) journalEntry.annotation = annotation || null;
        const bubble = Array.from(messagesDiv.querySelectorAll('.chat-bubble.user[data-client-message-id]'))
            .find((candidate) => candidate.dataset.clientMessageId === messageId);
        return renderRoutingAnnotation(bubble, annotation);
    }

    function clearTransientRoutingAnnotations() {
        for (const note of messagesDiv.querySelectorAll(
            '.msg-routing-annotation[data-annotation-status="pending"]',
        )) {
            const bubble = note.closest('.chat-bubble');
            if (bubble) delete bubble.dataset.chatAnnotationStatus;
            note.remove();
        }
    }

    function markPendingDelivered(clientMessageId) {
        const bubble = pendingUserBubbles.get(clientMessageId || '');
        if (!bubble) return;
        bubble.classList.remove('pending');
        bubble.querySelector('.msg-pending')?.remove();
        pendingUserBubbles.delete(clientMessageId);
    }

    return {
        routingAnnotationText,
        renderRoutingAnnotation,
        updateMessageAnnotation,
        clearTransientRoutingAnnotations,
        markPendingDelivered,
    };
}
