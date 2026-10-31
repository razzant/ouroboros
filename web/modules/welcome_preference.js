import { feedIsEmpty } from './chat_render_batch.js';

export const DEFAULT_WELCOME_TEXT = 'Ouroboros has awakened';

export function welcomeText(value) {
    if (value?.mode === 'hidden') return null;
    if (value?.mode === 'custom') return typeof value.text === 'string' && value.text.trim()
        ? value.text : null;
    return value?.mode === 'default' ? DEFAULT_WELCOME_TEXT : null;
}

// Main's empty state. Its copy is the hidden `welcome` UI preference (no Settings
// control: docs/DESIGN.md "Chat authorship and System rows"), read when Main connects.
// Only a successful recent read whose own window reports complete coverage can
// confirm emptiness; every read in flight, failed or partial withdraws it, and any
// insertion into the feed decides again. Chrome (the typing indicator, the restart
// notice) is not content: the history loading state counts the same nodes.
export function mountEmptyChatWelcome(messages) {
    const doc = messages.ownerDocument;
    let preference = null;
    let confirmedEmpty = false;
    let node = null;
    const render = () => {
        const copy = welcomeText(preference);
        if (!confirmedEmpty || !copy || !feedIsEmpty(messages)) {
            node?.remove();
            node = null;
            return;
        }
        if (!node) {
            node = doc.createElement('div');
            node.className = 'chat-empty-welcome';
            node.dataset.welcomeState = 'ready';
            node.innerHTML = '<span class="chat-empty-welcome-label">Welcome</span>';
            node.appendChild(doc.createElement('p'));
            messages.insertBefore(node, messages.querySelector('.typing-bubble'));
        }
        node.lastElementChild.textContent = copy;  // owner copy is text, never markup
    };
    const observer = typeof MutationObserver === 'function' ? new MutationObserver(render) : null;
    observer?.observe(messages, { childList: true });
    return {
        // Called only with a successful read: before one nothing shows, and a failed
        // re-read keeps the choice already observed.
        setPreference(value) { preference = value || null; render(); },
        // An earlier confirmation cannot vouch for a read still in flight.
        historyPending() { confirmedEmpty = false; render(); },
        historyRead(complete) { confirmedEmpty = complete === true; render(); },
        dispose() { observer?.disconnect(); },
    };
}
