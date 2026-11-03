// Replies of one chat instance (docs/DESIGN.md "Replies"): the Reply control beside Copy in a bubble's
// control row (the registered message actions stand between them, message_actions.js), the quote above the
// replying bubble's text, the strip in the composer's field while the owner is replying, the jump from a quote to the
// message it quotes, and the owner's words a relay carries into a work's room (one Ouroboros bubble: his words in
// a quote panel above the note). A reply is a reference the host resolves: this module sends only the quoted
// row's address and never decides who reads the words.
import { tr } from './i18n.js';
import { showToast } from './toast.js';
import { isStoreMigrating } from './api_client.js';
import { authorOf, createRelayQuote, createReplyQuote, normalizeReplyRef, replySnippet } from './chat_reply_quote.js';
import { createRelayDecks } from './chat_relay_deck.js';
import { fullText } from './text_fit.js';
import { bindBubbleActions } from './message_actions.js';
import { revealTimelineNode } from './chat_render_batch.js';
import { afterMotion } from './motion.js';
import { readBand } from './project_read_state.js';

export const REPLY_ICON = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor"'
    + ' stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    + '<path d="M6 11.5 2.5 8 6 4.5"></path><path d="M13.5 12v-1A3 3 0 0 0 10.5 8h-8"></path></svg>';
const TARGET_MS = 1600;
const HEADER_CLEARANCE = 72;
const ROLE_NAMES = ['user', 'assistant', 'system'];
const ROWS = '.chat-bubble, .chat-live-card';

/** The row a host-derived quote would repeat: the visible row right above `node` (its own work card is part of it). */
function rowAbove(node) {
    let row = node?.previousElementSibling;
    while (row && (row.hidden || !row.matches(ROWS) || (row.matches('.chat-live-card') && row.dataset.taskId
        && row.dataset.taskId === node.dataset.taskId))) row = row.previousElementSibling;
    return row;
}

/** Why a quote did not land, as `[code, words]` from the answer or the host's typed reason. Answers and a moved
 * work card (its task names no room) stay so; an unreadable source may stay so too. Only a read that may answer
 * differently asks to try again. */
function failureText({ state, task_id: taskId, error }) {
    if (state === 'absent') return ['absent', 'The quoted message was not found in saved history.'];
    if (state === 'ambiguous') return ['ambiguous', 'This reference matches more than one saved message.'];
    if (state === 'unavailable') return ['unavailable', 'The quoted message cannot be shown in an available room.'];
    if (state === 'moved' && taskId) return ['moved', 'This work is in another room.'];
    if (isStoreMigrating(error)) return ['migrating', 'Saved history is being upgraded. Try again shortly.'];
    if (error?.body?.reason_code === 'history_source_unavailable') {
        return ['source_unavailable', 'Saved history that may hold the quoted message could not be read.'];
    }
    return ['unknown', 'Could not load the quoted message. Try again.'];
}

/**
 * `feed` holds the bubbles (each stamped `data-row-address`), `input` is the composer field (the strip goes before its
 * row), `key` names the sessionStorage slot of a staged reply. `reading()` is the room's reading-position owner,
 * resolved when a quote is pressed; `onDomWrite` is the instance's stable-viewport seam for a clamp's toggle and the
 * control rows' redraws; `actions` is the message-action registry (the page's own when omitted).
 * `history()` supplies exact-node revelation through this room's pager; `navigate(address, chatId, current)`
 * lets app switch to the host's destination only while this claim remains current. It returns a state or null
 * for cancellation. Neither the quote's stored chat label nor its task chooses the destination.
 */
export function createChatReplies({ chatId, feed, input, composerArea = null, key = '', reading, history = () => null,
    navigate = null, present = () => true, onDomWrite = (write) => write(), actions = undefined }) {
    // The strip is the field's first line: inside the field, right before its text row (composer.js); a bare field (a
    // test fixture) takes it itself.
    const textRow = input.closest('.chat-text-row') || input;
    let storage = null;
    try { storage = sessionStorage; } catch { /* storage blocked: a staged reply lives for this page only */ }
    const storageKey = key;
    let staged = null;
    let strip = null;
    let targetTimer = 0;
    const bubbleActions = bindBubbleActions({ feed, registry: actions, onDomWrite });
    // How much of the feed's top the room's header covers: a placed row lands under it, not behind it.
    const headerClearance = () => {
        const header = feed.closest('.project-panel')?.querySelector('.project-panel-bar')
            || feed.closest('.page, .chat-instance-panel')?.querySelector('.chat-page-header, .chat-panel-statusbar');
        return Math.max(HEADER_CLEARANCE, (header?.getBoundingClientRect().bottom || 0) - feed.getBoundingClientRect().top + 12);
    };
    /** One explicit placement of a row under the header, as a quote's landing is: it claims the reading intent, so the
     * reader's own later gesture supersedes it. `follow` instead returns a reader who was following the newest message
     * to that message. */
    function place(target, { follow = false } = {}) {
        const position = reading();
        if (position.claim) position.claim(); else position.cancel();
        if (follow) {
            position.stick = true;
            position.followAfterLayout?.();
            return;
        }
        position.stick = false;
        revealTimelineNode(target, feed, headerClearance());
        position.scroll();
        position.remember(true);
    }
    // Notes between works: a relay row's bubble is dressed as a note card and grouped into stacks (chat_relay_deck.js);
    // the room decides how the address reads (Main is chat 1) and lends a fanned column its reading owner and placement.
    // What of the feed a reader sees below: the composer is drawn over the feed's foot.
    const band = () => readBand(feed, null, composerArea);
    const decks = createRelayDecks({ feed, chatId, onDomWrite, reading, clearance: headerClearance, band, place });
    let destroyed = false;
    // `held`: the reader's own live-edge intent while a reveal is pending (a reveal sets stick false itself).
    let revealGeneration = 0, revealing = 0, held = null;

    const remember = () => {
        if (!storage || !storageKey) return;
        try {
            if (staged) storage.setItem(storageKey, JSON.stringify(staged));
            else storage.removeItem(storageKey);
        } catch { /* a refused write leaves the strip for this page only */ }
    };

    const rowOf = (address) => (address ? [...feed.querySelectorAll('[data-row-address]')]
        .find((node) => node.dataset.rowAddress === address) : null) || null;

    function markTarget(target) {
        clearTimeout(targetTimer);
        feed.querySelectorAll('.is-quote-target').forEach((old) => old.classList.remove('is-quote-target'));
        void target.offsetWidth;
        target.classList.add('is-quote-target');
        if (target.tabIndex < 0) target.tabIndex = -1;
        target.focus({ preventScroll: true });
        targetTimer = setTimeout(() => target.classList.remove('is-quote-target'), TARGET_MS);
    }

    /** One claim covers the loaded fastpath, bounded read and app-owned room switch. Never sends a message.
     * A reveal that moved nothing (an answer, a failed read, a room it could not open) gives a reader who followed
     * the live edge that edge back, with what arrived meanwhile, while the claim is still theirs: a gesture, a
     * newer reveal or leaving the room decided instead. A landing or a page drawn around it (`left`) keeps its view. */
    async function reveal(address, isCurrent = () => true, prepared = null) {
        const position = reading();
        const following = held?.claimed() ? held.following : position.stick;
        const claimed = position.claim ? position.claim() : (position.cancel(), () => true);
        const wanted = () => !destroyed && present() && claimed() && isCurrent();
        const generation = revealing = ++revealGeneration;
        const own = held = { following, claimed };
        position.stick = false;
        const result = await seek(address, position, wanted, prepared)
            .catch(error => (wanted() ? { state: 'unknown', error } : null));
        if (generation === revealing) revealing = 0;
        if (held === own) held = null;
        if (following && generation === revealGeneration && wanted() && result?.state !== 'found' && !result?.left) {
            position.stick = true;
            position.followAfterLayout?.();
        }
        return result;
    }

    const drawn = (address) => history()?.exact(address) || (!history() && rowOf(address));
    /** Whether a reveal of `address` lands here without a read: its row is drawn and is not a portal, whose work
     * lives in another room. */
    const draws = (address) => {
        const node = drawn(address);
        return Boolean(node) && node.dataset.portal !== '1';
    };

    async function seek(address, position, wanted, prepared) {
        let target = !prepared && drawn(address);
        if (!target) {
            const result = await history()?.reference(address, wanted, prepared);
            if (!wanted() || !result) return null;
            if (result.state === 'moved' && Number.isInteger(result.chat_id) && navigate) {
                return navigate(address, result.chat_id, wanted);
            }
            if (!result.target) return result;
            target = result.target;
        }
        if (!wanted()) return null;
        target = decks.surface(target) || target;  // a tucked note's deck opens, so the row it lands on shows
        revealTimelineNode(target, feed, headerClearance());
        position.scroll();
        // This explicit placement measured the current column. An ordinary scroll
        // keeps the old-width anchor until reflow; a newly opened room must replace it.
        position.remember(true);
        markTarget(target);
        return { state: 'found' };
    }

    async function jump(ref) {
        const result = await reveal(ref.address);
        if (!result || result.state === 'found') return;
        const [code, words] = failureText(result);
        showToast(tr(`chat.reply.${code}`, words), 'info');
    }

    // The strip opens and closes like the capsule row's fold (style.css): its height runs from 0 while its words fade,
    // so the row above it rises with it instead of jumping; a reply staged over another one swaps in place. The room it
    // takes at rest (its box and margins, --reply-room) is where the composer's reserve goes at once (chat.js). Staging
    // focuses the field and lays the composer out, a long first frame the first time: the strip is held shut through it
    // (`data-hold`) and opens from the frame after, so no long frame shows it half open.
    function run(node, to) {
        const style = typeof getComputedStyle === 'function' ? getComputedStyle(node) : null;
        const margins = style ? (parseFloat(style.marginTop) || 0) + (parseFloat(style.marginBottom) || 0) : 0;
        node.style.setProperty('--reply-h', `${node.offsetHeight}px`);
        node.style.setProperty('--reply-room', `${node.offsetHeight + margins}px`);
        if (to === 'open' && typeof requestAnimationFrame === 'function') {
            node.dataset.hold = '';
            node.dataset.motion = 'open';
            return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))).then(() => {
                if (!('hold' in node.dataset)) return 'cancel';
                delete node.dataset.hold;
                return afterMotion(node, { property: 'max-height' }).done;
            });
        }
        delete node.dataset.hold;
        node.dataset.motion = to === 'open' ? 'shut' : 'open';
        void node.offsetHeight;  // the motion starts from this height
        node.dataset.motion = to;
        return afterMotion(node, { property: 'max-height' }).done;
    }
    function paint(animate = true) {
        const before = strip;
        strip = staged ? createReplyQuote(staged, { composer: true, onJump: jump, onCancel: () => cancel() }) : null;
        if (strip) {
            const main = strip.querySelector('.chat-quote-main');
            main.id = `${input.id}-reply`;
            // The field announces what it answers whenever it takes focus.
            input.setAttribute('aria-describedby', main.id);
            if (before?.isConnected && before.dataset.motion !== 'shut') { before.replaceWith(strip); return; }
            textRow.before(strip);
            const opened = strip;
            // A reply the room kept comes with the room; one staged now opens.
            if (animate) void run(opened, 'open').then(() => { if (opened.dataset.motion === 'open') delete opened.dataset.motion; });
        } else input.removeAttribute('aria-describedby');
        if (before?.isConnected && before.dataset.motion !== 'shut') {
            before.inert = true;  // leaving, its controls take no press
            void run(before, 'shut').then(() => before.remove());
        }
    }

    /**
     * Reply to one row. A still-open question card answers in its own field (the existing answer door, never a
     * second path); anything else stages the strip, the composer field's first line.
     */
    function start(raw) {
        const ref = normalizeReplyRef(raw);
        if (!ref?.address) return;
        const door = rowOf(ref.address)?.querySelector('.chat-quiz-comment:not(:disabled):not([readonly])');
        if (door) { door.focus(); return; }
        staged = ref;
        paint();
        remember();
        input.focus({ preventScroll: true });
    }

    /** Drops the strip and keeps whatever the owner already typed. */
    function cancel() {
        if (!staged) return;
        staged = null;
        paint();
        remember();
        input.focus({ preventScroll: true });
    }

    /** After a send: the next message answers nothing. */
    function clear() {
        if (!staged) return;
        staged = null;
        paint();
        remember();
    }

    function onKeydown(event) {
        if (event.key !== 'Escape' || !staged || event.isComposing) return;
        event.preventDefault();
        cancel();
    }
    // One listener on the field (the text row's parent): Esc from the words and from the strip's own buttons lands here.
    const keyHost = textRow.parentElement || input;
    keyHost?.addEventListener('keydown', onKeydown);

    /** What a Reply on this bubble quotes (its own address, author, words and time), or null without an address. */
    function bubbleRef(bubble) {
        const address = bubble?.dataset?.rowAddress || '';
        if (!address) return null;
        const role = ROLE_NAMES.find((name) => bubble.classList.contains(name)) || '';
        const words = bubble.querySelector(':scope > .message')?.textContent
            || bubble.querySelector('.chat-quiz-question')?.textContent || bubble.dataset.rowWords || '';
        const sender = bubble.querySelector(':scope > .sender, :scope > .chat-bubble-voice > .sender');
        return { address, chat_id: chatId, role, author_label: sender?.textContent || '',
            kind: bubble.querySelector('.chat-quiz-card') ? 'question' : 'message', snippet: replySnippet(words),
            ts: bubble.dataset.ts ? new Date(Number(bubble.dataset.ts)).toISOString() : '', task_id: bubble.dataset.taskId || '' };
    }

    /** The Reply button, placed with the bubble's Copy in one control row. Needs a row address: an unsaved echo has none. */
    function mountControl(bubble) {
        if (!bubbleRef(bubble)) return null;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'chat-message-reply ui-hit-area';
        button.innerHTML = REPLY_ICON;
        button.title = tr('chat.reply.reply', 'Reply');
        button.setAttribute('aria-label', tr('chat.reply.reply', 'Reply'));
        button.dataset.rowReply = '1';
        button.addEventListener('click', () => start(bubbleRef(bubble)));
        const copy = bubble.querySelector(':scope > .chat-message-copy');
        const row = document.createElement('div');
        row.className = 'chat-message-actions';
        if (copy) copy.replaceWith(row); else bubble.append(row);
        row.append(button);
        if (copy) row.append(copy);
        bubble.classList.add('has-reply');
        bubbleActions.sync(bubble);
        return button;
    }

    try {
        const saved = storage && storageKey ? storage.getItem(storageKey) : '';
        if (saved) { staged = normalizeReplyRef(JSON.parse(saved)); paint(false); }
    } catch { staged = null; }

    /**
     * What a reply to a work card quotes, or null while the card offers none: the work itself (its task id is its
     * durable address; for a Continue card, the current episode's), named by its title, with its latest line or,
     * settled, its chip's words.
     */
    function cardRef(record) {
        if (!record?.replyable) return null;
        const line = fullText(record.activityEl).trim();
        const chip = record.phaseEl?.textContent?.trim() || '';
        return {
            address: `task:${record.groupId}`, chat_id: chatId, role: 'assistant', kind: 'work_card',
            author_label: record.titleEl?.textContent?.trim() || authorOf('assistant'),
            snippet: replySnippet(line || chip), ts: record.latestActivityRaw || '', task_id: record.groupId,
        };
    }

    /**
     * A root work card is answered from its row menu (docs/DESIGN.md "Replies"; owner 17.10, Q9 B): its action row
     * holds only commands over the work, so it draws no Reply of its own. `wanted` is false for anything that is not
     * a root work card. The card becomes a quote's jump target by the same address.
     */
    function syncCard(record, wanted) {
        if (!record?.root) return;
        record.replyable = Boolean(wanted);
        const address = `task:${record.groupId}`;
        if (wanted && record.root.dataset.rowAddress !== address) record.root.dataset.rowAddress = address;
    }

    /**
     * A quote the host derived for an Ouroboros row (the owner message its turn answers) repeats nothing when that
     * message is the row right above it; it shows once another message or card came between (a late answer, a
     * return from a Project). History fills in above, so a landing row re-reads itself and the row below it.
     */
    function syncDerivedQuote(bubble) {
        const quote = bubble?.querySelector?.(':scope > .chat-quote[data-derived="1"]');
        if (!quote) return false;
        const hidden = rowAbove(bubble)?.dataset.rowAddress === quote.dataset.address;
        if (quote.hidden === hidden) return false;
        quote.hidden = hidden;
        return true;
    }

    function syncDerivedQuotes(around = null) {
        decks.schedule();  // a landed row may join, split or end a run of notes
        if (!around) {
            return [...feed.querySelectorAll('.chat-bubble > .chat-quote[data-derived="1"]')]
                .reduce((changed, quote) => syncDerivedQuote(quote.parentElement) || changed, false);
        }
        // The first bubble below (past cards, which may be that bubble's own work) may now repeat or stop repeating.
        let below = around.nextElementSibling;
        while (below && (below.hidden || !below.matches('.chat-bubble'))) below = below.nextElementSibling;
        return Boolean(syncDerivedQuote(around) | syncDerivedQuote(below));
    }

    return {
        /**
         * One call per text bubble, after Copy is mounted. `row` is the history row or live frame the bubble was built
         * from: its `reply_to` becomes the quote above the text (the owner's own reference, or on an Ouroboros row the
         * host-derived one), an Ouroboros row's `relay` the owner's words in a relay panel, and its `row_address` makes
         * the bubble a jump target with a Reply control (author, role and task come from the bubble itself).
         */
        decorate(bubble, row) {
            const address = String(row?.row_address || '');
            if (address) bubble.dataset.rowAddress = address;
            // Work speech another room also shows: its read is shared (read_ahead.js).
            if (row?.room_revisions && typeof row.room_revisions === 'object') bubble.dataset.roomRevisions = JSON.stringify(row.room_revisions);
            const sender = bubble.querySelector('.sender');
            const role = ROLE_NAMES.find((name) => bubble.classList.contains(name));
            const relay = role === 'assistant' ? createRelayQuote(row?.relay, { onDomWrite, card: bubble.dataset.systemType === 'relay_entry' }) : null;
            const quote = relay || createReplyQuote(row?.reply_to, { onJump: jump });
            if (quote && !relay && role !== 'user') {
                quote.dataset.derived = '1';
                quote.dataset.address = normalizeReplyRef(row.reply_to)?.address || '';
            }
            if (quote) sender?.after(quote);
            mountControl(bubble);
            if (row?.relay && bubble.dataset.systemType === 'relay_entry') decks.adopt(bubble, row, Boolean(relay));
        },
        /** A row built outside addMessage (media, files, links, a question card): its address makes it a reply target. */
        stamp(bubble, row) {
            const address = String(row?.row_address || '');
            if (bubble && address && bubble.dataset.rowAddress !== address) bubble.dataset.rowAddress = address;
        },
        /** The host's saved row for the owner's own echo: the bubble takes its address and, with it, its Reply. */
        adopt(row) {
            const cmid = String(row?.client_message_id || '');
            const address = String(row?.row_address || '');
            if (row?.role !== 'user' || !cmid || !address) return false;
            const bubble = [...feed.querySelectorAll('.chat-bubble.user[data-client-message-id]')]
                .find((node) => node.dataset.clientMessageId === cmid);
            if (!bubble || bubble.dataset.rowAddress === address) return false;
            bubble.dataset.rowAddress = address;
            if (!bubble.querySelector(':scope > .chat-message-actions > .chat-message-reply')) mountControl(bubble);
            syncDerivedQuotes(bubble);
            return true;
        },
        /** What the row menu's Reply on this bubble quotes, or null. */
        bubbleRef,
        /** What the row menu's Reply on this root work card quotes, or null. */
        cardRef,
        start,
        syncCard,
        syncDerivedQuotes,
        reveal,
        draws,
        get revealing() { return revealing !== 0; },
        /**
         * "Send as a new message" under a delivery doubt: the owner's words join this composer's draft (sent, they are
         * a new message with a fresh id); nothing is sent here.
         */
        compose(words) {
            const draft = input.value.trimEnd();
            input.value = draft ? `${draft}\n\n${words}` : words;
            input.dispatchEvent?.(new Event('input', { bubbles: true }));  // the composer's own sizing and draft keeping
            input.focus({ preventScroll: true });
        },
        /** The field of the chat frame: only the address travels, the host makes the snippet. */
        frameField: () => (staged ? { reply_to: staged.address } : {}),
        /** The local echo of a reply, shaped like a row so it draws the same quote (it has no row address yet). */
        echoRow: () => (staged ? { reply_to: staged } : null),
        clear,
        destroy() {
            destroyed = true;
            clearTimeout(targetTimer);
            bubbleActions.destroy();
            decks.destroy();
            keyHost?.removeEventListener('keydown', onKeydown);
            strip?.remove();
            strip = null;
            staged = null;
        },
    };
}
