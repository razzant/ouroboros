// The composer row of ONE chat instance and the viewport reserve it governs:
// the one-shot Swarm arm, the send-busy presentation, the auto-growing textarea,
// and the scroll affordances whose geometry depends on the composer's rendered
// height (the CSS reserve that keeps the absolute composer off the messages, the
// jump-to-newest button, and the pin-to-tail write). Every element handle and the
// two viewport predicates are handed over explicitly, so a Main chat and a
// Project panel size and scroll only their own column.
export function createComposer({
    page,
    input,
    inputArea,
    pageHeader,
    messagesDiv,
    sendBtn,
    sendGroup,
    swarmBtn,
    scrollBottomBtn,
    isInstanceVisible,
    isNearBottom,
    scrollToBottomAfterLayout,
}) {
    function resizeChatInput({ preserveStickiness = false } = {}) {
        const caretAtEnd = input.selectionEnd >= input.value.length - 1;
        const previousScrollTop = input.scrollTop;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        input.scrollTop = caretAtEnd ? input.scrollHeight : previousScrollTop;
        updateMessagesPadding({ preserveStickiness });
    }

    function swarmArmed() {
        return swarmBtn?.dataset.armed === 'true';
    }
    function setSwarm(armed) {
        if (swarmBtn) swarmBtn.dataset.armed = armed ? 'true' : 'false';
    }

    function setSendBusy(busy, label = '') {
        sendGroup.dataset.busy = busy ? '1' : '0';
        sendBtn.disabled = busy;
        if (busy) {
            sendBtn.textContent = label || 'Sending';
            sendBtn.title = label || 'Sending';
        } else {
            sendBtn.textContent = 'Send';
            sendBtn.title = 'Send message';
        }
    }

    // Dynamic CSS reserve keeps the absolute composer from covering messages.
    function scrollToBottom() {
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    // Round glass "jump to newest" affordance — shown only when the user has
    // scrolled up away from the bottom, for both the main chat and panels.
    function updateScrollButton() {
        if (!scrollBottomBtn) return;
        scrollBottomBtn.classList.toggle('visible', isInstanceVisible() && !isNearBottom());
    }

    function updateMessagesPadding(options = {}) {
        const preserveStickiness = options.preserveStickiness !== false;
        const shouldStick = preserveStickiness && isNearBottom();
        if (pageHeader && messagesDiv) {
            // The main header wraps to two rows on narrow viewports. Reserve its
            // REAL rendered height so scrollTop=0 never hides the first message
            // behind the absolute overlay; project panels have no overlay header.
            const headerReserve = Math.max(56, Math.ceil(pageHeader.offsetHeight || 0));
            page.style.setProperty('--chat-header-reserve', `${headerReserve}px`);
        }
        if (inputArea && messagesDiv) {
            const reserve = Math.max(92, Math.ceil(inputArea.offsetHeight || 0) + 16);
            // Set on the instance page root so it cascades to #chat-messages
            // (padding) AND the sibling scroll-to-bottom button (bottom offset).
            page.style.setProperty('--chat-input-reserve', `${reserve}px`);
        }
        if (shouldStick) scrollToBottomAfterLayout();
        updateScrollButton();
    }

    return {
        resizeChatInput,
        swarmArmed,
        setSwarm,
        setSendBusy,
        scrollToBottom,
        updateScrollButton,
        updateMessagesPadding,
    };
}
