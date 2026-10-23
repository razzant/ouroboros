/** One call is one card in the room (the owner's choice for how a call reads in the chat): the call's
 * own rows fold into it as lines — the owner's spoken words (`client_message_id` `call:<call>:<item>`),
 * Ouroboros's spoken narration of the same task, the call's work card and its closing notice — and
 * the live captions stream at its end. The transcript is kept, never the call's focus: one's own
 * words running past while one speaks pull attention from the conversation, so a card is one folded
 * line, live or ended, until the owner opens it. The rows stay the room's and stay mounted, only
 * hidden (memory, replies, history and Copy keep reading them); a link Ouroboros sends during the
 * call stays an ordinary message.
 *
 * The opening echo names its call before a task exists; the owner's spoken row later names both.
 * A task is a call once that owner row is seen. A caption that arrives before that row opens the
 * card at the room's end, and the row adopts it. Engines send partial captions as additions and the
 * final one whole; a room row for an utterance retires its caption (the chat paints the row first:
 * it subscribed first), so the words never show twice.
 */
const CALL_PREFIX = 'call:';
const NARRATION = 'model_narration';
// How close to the newest message still counts as reading it: the feed then follows the words.
const FOLLOW_SLACK_PX = 48;
const KEPT_CALLS = 4;

/** What a mounted room node is to a call: `{task, kind, call?}` or null for any other row. */
export function callMemberOf(node, callTasks) {
    const list = node?.classList;
    if (!list || !(list.contains('chat-bubble') || list.contains('chat-live-card'))) return null;
    if (list.contains('call-caption')) return null;
    const task = String(node.dataset?.taskId || '');
    const clientId = String(node.dataset?.clientMessageId || '');
    const opening = /^call:([^:]+):open$/.exec(clientId);
    if (opening && list.contains('system') && node.dataset.systemType === 'call_open') {
        return { task, kind: 'opening', call: opening[1] };
    }
    if (task && list.contains('user') && clientId.startsWith(CALL_PREFIX)) {
        return { task, kind: 'owner', call: clientId.slice(CALL_PREFIX.length).split(':')[0] };
    }
    if (!task || !callTasks.has(task)) return null;
    if (list.contains('chat-live-card')) return { task, kind: 'work' };
    if (list.contains('assistant') && node.dataset.systemType === NARRATION) return { task, kind: 'ouroboros' };
    if (list.contains('system')) return { task, kind: 'notice' };
    return null;
}

/** "3 min 30 s", "45 s": how long the call's lines span. */
export function callSpan(ms) {
    const seconds = Math.max(0, Math.round(Number(ms || 0) / 1000));
    const minutes = Math.floor(seconds / 60);
    if (!minutes) return `${seconds} s`;
    return seconds % 60 ? `${minutes} min ${seconds % 60} s` : `${minutes} min`;
}

function spokenText(node) {
    const message = node.querySelector?.('.message');
    return String((message || node).textContent || '').trim().replace(/^💬\s*/u, '');
}

export function createCallCards({ feed, ws = null, doc = feed?.ownerDocument, live = () => '', sign = null } = {}) {
    const cards = new Map();        // task id, or `call:<id>` before its task is known → card
    const callTasks = new Map();    // task id → call id
    const captions = new Map();     // `call␀item` → caption line
    const settled = new Map();      // call id → item ids the room already holds
    const key = (call, item) => `${call}\u0000${item}`;
    let observer = null;
    let applying = false;

    const following = () => !feed || feed.scrollHeight - feed.scrollTop - feed.clientHeight <= FOLLOW_SLACK_PX;
    const follow = (was) => { if (was && feed) feed.scrollTop = feed.scrollHeight; };
    /** A reader of older messages keeps the row at the top of the view while rows above it fold. */
    function readingAnchor() {
        const node = [...feed.children].find(child => !child.classList?.contains('call-folded')
            && !child.classList?.contains('call-card') && child.offsetTop + child.offsetHeight > feed.scrollTop);
        return node ? { node, top: node.offsetTop } : null;
    }

    function element(tag, className, text) {
        const node = doc.createElement(tag);
        if (className) node.className = className;
        if (text != null) node.textContent = text;
        return node;
    }

    function createCard(id) {
        const node = element('div', 'call-card');
        const head = element('button', 'call-card-head');
        head.type = 'button';
        const facts = element('span', 'call-card-facts');
        // The call's own small sign (the visual's, as on Talk) leads the card.
        head.append(sign?.() || element('span', 'call-entry-mark'), element('span', 'call-card-title', 'Call'), facts);
        const lines = element('div', 'call-card-lines');
        node.append(head, lines);
        const card = { id, task: '', call: '', node, head, facts, lines, open: null, count: 0, span: 0 };
        head.addEventListener?.('click', () => {
            const was = following();
            card.open = !isOpen(card);
            paint(card);
            follow(was && card.open);
        });
        cards.set(id, card);
        return card;
    }

    const isLive = card => Boolean(card.call) && live() === card.call;
    const isOpen = card => card.open === true;

    function paint(card) {
        const liveNow = isLive(card);
        card.node.dataset.callCard = liveNow ? 'live' : 'ended';
        card.node.dataset.callOpen = isOpen(card) ? '1' : '0';
        card.head.setAttribute?.('aria-expanded', isOpen(card) ? 'true' : 'false');
        const lines = `${card.count} ${card.count === 1 ? 'line' : 'lines'}`;
        card.facts.textContent = liveNow ? `Live · ${lines}` : (card.span ? `${callSpan(card.span)} · ${lines}` : lines);
    }

    function line(speaker, text) {
        const node = element('div', `call-line ${speaker}`);
        if (speaker !== 'notice') node.append(element('span', 'call-line-who', speaker === 'owner' ? 'You' : 'Ouroboros'));
        node.append(element('span', 'call-line-text', text));
        return node;
    }

    /** The card a call's caption streams into: its task's card, or one opened at the room's end. */
    function cardForCall(call) {
        for (const card of cards.values()) if (card.call === call) return card;
        const card = createCard(`${CALL_PREFIX}${call}`);
        card.call = call;
        const typing = [...feed.children].find(child => child.classList?.contains('typing-bubble'));
        applying = true;
        try { if (typing) feed.insertBefore(card.node, typing); else feed.append(card.node); } finally { applying = false; }
        paint(card);
        return card;
    }

    function cardForTask(task, call) {
        let card = cards.get(task);
        if (card) return card;
        const pending = cards.get(`${CALL_PREFIX}${call}`);
        if (pending) {  // the caption's card is adopted by its task
            cards.delete(pending.id);
            pending.id = task;
            cards.set(task, pending);
            card = pending;
        } else {
            card = createCard(task);
        }
        card.task = task;
        card.call = call || card.call;
        return card;
    }

    /** Fold every call row mounted in the room into its call's card. */
    function scan() {
        if (!feed) return;
        const was = following();
        const anchor = was ? null : readingAnchor();
        const nodes = [...feed.children];
        for (const node of nodes) {
            const member = callMemberOf(node, callTasks);
            if (member?.call && member.task) callTasks.set(member.task, member.call);
        }
        const groups = new Map();
        for (const node of nodes) {
            const member = callMemberOf(node, callTasks);
            if (!member) continue;
            const call = member.call || callTasks.get(member.task);
            if (!groups.has(call)) groups.set(call, []);
            groups.get(call).push({ node, ...member });
        }
        applying = true;
        try {
            for (const [call, members] of groups) {
                const task = members.find(member => member.task)?.task;
                fold(task ? cardForTask(task, call) : cardForCall(call), members);
            }
        } finally {
            applying = false;
        }
        if (anchor?.node.isConnected) feed.scrollTop += anchor.node.offsetTop - anchor.top;
        follow(was);
    }

    function fold(card, members) {
        if (members[0].node.previousElementSibling !== card.node) feed.insertBefore(card.node, members[0].node);
        const spoken = [];
        let first = NaN;
        let last = NaN;
        const notices = [];
        for (const member of members) {
            member.node.classList.add('call-folded');
            if (member.kind === 'work' || member.kind === 'opening') continue;
            const text = spokenText(member.node);
            if (!text) continue;
            // The span runs from the first words to the call's end, its closing notice.
            const ts = Number(member.node.dataset.ts);
            if (Number.isFinite(ts)) last = Math.max(Number.isFinite(last) ? last : ts, ts);
            // The closing notice says how the call ended; it reads under the words, not as one of them.
            if (member.kind === 'notice') { notices.push(line('notice', text)); continue; }
            spoken.push(line(member.kind, text));
            if (Number.isFinite(ts)) first = Number.isFinite(first) ? Math.min(first, ts) : ts;
        }
        const streaming = [...captions.values()].filter(entry => entry.call === card.call).map(entry => entry.node);
        card.lines.replaceChildren(...spoken, ...streaming, ...notices);
        card.count = spoken.length;
        card.span = Number.isFinite(first) && Number.isFinite(last) ? last - first : 0;
        if (Number.isFinite(first)) card.node.dataset.ts = String(first);
        paint(card);
    }

    function dropCaption(id) {
        const entry = captions.get(id);
        if (!entry) return;
        entry.node.remove();
        captions.delete(id);
    }

    /** A caption of the live call: a line at its card's end, streaming until the room's row comes. */
    function update(msg) {
        const call = String(msg?.call_id || '');
        const item = String(msg?.item_id || '');
        if (!feed || !call || !item || settled.get(call)?.has(item)) return;
        const id = key(call, item);
        const text = String(msg.text || '');
        if (msg.final && !text.trim()) { dropCaption(id); return; }
        let entry = captions.get(id);
        if (!entry && !text) return;  // an item named before its words: nothing to show yet
        const was = following();
        const card = cardForCall(call);
        if (!entry) {
            // A new utterance of the same speaker means the earlier one was cut short.
            for (const other of captions.values()) if (other.speaker === msg.speaker) other.node.dataset.callCaption = 'final';
            entry = { call, speaker: msg.speaker, text: '', node: line(msg.speaker === 'owner' ? 'owner' : 'ouroboros', '') };
            entry.node.classList.add('call-caption');
            captions.set(id, entry);
        }
        entry.text = (msg.final || msg.replace) ? text : entry.text + text;
        entry.node.lastChild.textContent = entry.text;
        entry.node.dataset.callCaption = msg.final ? 'final' : 'streaming';
        if (entry.node.parentNode !== card.lines) card.lines.append(entry.node);
        paint(card);
        follow(was);
    }

    /** A room row for a call utterance retires its caption line. */
    function settle(frame) {
        let call = '';
        let item = '';
        const clientId = String(frame?.client_message_id || '');
        if (frame?.role === 'user' && clientId.startsWith(CALL_PREFIX)) {
            const rest = clientId.slice(CALL_PREFIX.length);
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
        dropCaption(key(call, item));
    }

    /** The call ended: what is still streaming is as complete as it will get, and its card folds. */
    function finish() {
        for (const entry of captions.values()) entry.node.dataset.callCaption = 'final';
        for (const card of cards.values()) paint(card);
    }

    const offChat = ws?.on?.('chat', settle);
    const offOpen = ws?.on?.('open', (info) => {
        if (!info?.previouslyConnected) return;
        for (const [id, entry] of captions) if (entry.node.dataset.callCaption === 'final') dropCaption(id);
    });
    const Observer = doc?.defaultView?.MutationObserver;
    if (feed && Observer) {
        let queued = false;
        observer = new Observer(() => {
            if (applying || queued) return;
            queued = true;
            queueMicrotask(() => { queued = false; scan(); });
        });
        observer.observe(feed, { childList: true });
    }
    scan();
    return {
        update,
        settle,
        finish,
        scan,
        count: () => captions.size,
        destroy() {
            offChat?.();
            offOpen?.();
            observer?.disconnect();
            for (const id of [...captions.keys()]) dropCaption(id);
            for (const card of cards.values()) card.node.remove();
            for (const node of feed ? [...feed.children] : []) node.classList?.remove('call-folded');
            cards.clear();
            settled.clear();
        },
    };
}
