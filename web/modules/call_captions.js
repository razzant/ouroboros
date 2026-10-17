/** Live captions are ordinary bubbles of the room while they are spoken: the owner's in the
 * owner's blue, Ouroboros's in its blush. Engines send partial captions as additions and the final
 * one whole. The room's own row for an utterance replaces its bubble: the owner's row
 * (`client_message_id` `call:<call>:<item>`) and Ouroboros's spoken narration (`spoken.item_id`),
 * which the chat paints before this listener runs (it subscribed first), so the words never show
 * twice. A caption whose row this page missed (a dropped socket) leaves on the reconnect, whose
 * history read brings the row; a row that came first keeps its late caption out.
 */
// How close to the newest message still counts as reading it: the feed then follows the words.
const FOLLOW_SLACK_PX = 48;
const KEPT_CALLS = 4;

export function createCallCaptions({ feed, ws = null, doc = feed?.ownerDocument } = {}) {
    const live = new Map();     // `call␀item` → { node, message, text }
    const settled = new Map();  // call id → item ids the room already holds
    const key = (call, item) => `${call}\u0000${item}`;

    function remove(id) {
        const entry = live.get(id);
        if (!entry) return;
        entry.node.remove();
        live.delete(id);
    }

    function bubble(speaker) {
        const node = doc.createElement('div');
        node.className = `chat-bubble ${speaker === 'owner' ? 'user' : 'assistant'} call-caption`;
        const sender = doc.createElement('div');
        sender.className = 'sender';
        sender.textContent = speaker === 'owner' ? 'You' : 'Ouroboros';
        const message = doc.createElement('div');
        message.className = 'message';
        node.append(sender, message);
        return { node, message, speaker, text: '' };
    }

    function update(msg) {
        const call = String(msg?.call_id || '');
        const item = String(msg?.item_id || '');
        if (!feed || !call || !item || settled.get(call)?.has(item)) return;
        const id = key(call, item);
        const text = String(msg.text || '');
        if (msg.final && !text.trim()) { remove(id); return; }
        let entry = live.get(id);
        if (!entry && !text) return;  // an item named before its words: nothing to show yet
        const following = feed.scrollHeight - feed.scrollTop - feed.clientHeight <= FOLLOW_SLACK_PX;
        if (!entry) {
            // A new utterance of the same speaker means the earlier one was cut short.
            for (const other of live.values()) if (other.speaker === msg.speaker) other.node.dataset.callCaption = 'final';
            entry = bubble(msg.speaker);
            live.set(id, entry);
        }
        entry.text = msg.final ? text : entry.text + text;
        entry.message.textContent = entry.text;
        entry.node.dataset.callCaption = msg.final ? 'final' : 'streaming';
        if (!entry.node.isConnected) {
            const typing = [...feed.children].find(child => child.classList?.contains('typing-bubble'));
            if (typing) feed.insertBefore(entry.node, typing); else feed.append(entry.node);
        }
        if (following) feed.scrollTop = feed.scrollHeight;
    }

    /** A room row for a call utterance retires its caption bubble. */
    function settle(frame) {
        let call = '';
        let item = '';
        const clientId = String(frame?.client_message_id || '');
        if (frame?.role === 'user' && clientId.startsWith('call:')) {
            const rest = clientId.slice(5);
            const colon = rest.indexOf(':');
            if (colon > 0) { call = rest.slice(0, colon); item = rest.slice(colon + 1); }
        } else if (frame?.spoken && typeof frame.spoken === 'object') {
            call = String(frame.spoken.call_id || '');
            item = String(frame.spoken.item_id || '');
        }
        if (!call || !item) return;
        if (!settled.has(call)) {
            settled.set(call, new Set());
            while (settled.size > KEPT_CALLS) settled.delete(settled.keys().next().value);
        }
        settled.get(call).add(item);
        remove(key(call, item));
    }

    /** The call ended: what is still streaming is as complete as it will get. */
    function finish() {
        for (const entry of live.values()) entry.node.dataset.callCaption = 'final';
    }

    const offChat = ws?.on?.('chat', settle);
    const offOpen = ws?.on?.('open', (info) => {
        if (!info?.previouslyConnected) return;
        for (const [id, entry] of live) if (entry.node.dataset.callCaption === 'final') remove(id);
    });
    return {
        update,
        settle,
        finish,
        count: () => live.size,
        destroy() {
            offChat?.();
            offOpen?.();
            for (const id of [...live.keys()]) remove(id);
            settled.clear();
        },
    };
}
