// The quote panel (docs/DESIGN.md "Replies"): the one component that shows another message inside a bubble.
// A reply shows one ellipsized line and is a button that jumps to the quoted message; with a cancel control it
// is the strip in the composer's field while the owner replies. A relay shows the owner's complete words (they ARE
// the instruction) clamped to six lines, with a real `Show all` control in its head; inside a relay note card it
// clamps to three lines and the card's `more`/`less` (chat_relay_deck.js) owns its expansion. The bar and the author
// line wear the QUOTED author's ink; the bubble around keeps its own voice. Every string is set as text: the
// quoted words are data, never markup.
import { fmt, tr } from './i18n.js';
import { formatMsgTime } from './chat_activity.js';
import { attachmentViews } from './chat_attachments.js';
import { ellipsize, fitText } from './text_fit.js';

const ROLES = ['user', 'assistant', 'system'];
// The referenced row's kind as the host names it (CONTRACTS K5); anything else reads as a message.
const KINDS = ['message', 'work_card', 'question', 'completion', 'system'];
const DEFAULT_AUTHOR = {
    user: () => tr('chat.sender.you', 'You'),
    assistant: () => tr('chat.sender.ouroboros', 'Ouroboros'),
    system: () => tr('chat.sender.system', 'System'),
};
const CANCEL_ICON = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor"'
    + ' stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M4 4l8 8"></path><path d="M12 4l-8 8"></path></svg>';

const oneSpace = (value) => String(value ?? '').replace(/\s+/g, ' ').trim();
export const authorOf = (role) => DEFAULT_AUTHOR[role]?.() || '';

function node(tag, className, content) {
    const el = document.createElement(tag);
    el.className = className;
    if (content !== undefined) el.textContent = content;
    return el;
}

/** The first readable line of a message, markup marks dropped and cut at a word: what a reply quote shows. */
export function replySnippet(text, limit = 160) {
    const line = String(text ?? '').split('\n').map((part) => oneSpace(part.replace(/^[#>\s]+/, '').replace(/[*`]+/g, ''))).find(Boolean) || '';
    if (line.length <= limit) return line;
    const cut = line.slice(0, limit);
    return ellipsize(cut.slice(0, Math.max(cut.lastIndexOf(' '), limit >> 1)));
}

/** The host's typed `reply_to` as a safe reference, or null when it carries nothing to show. */
export function normalizeReplyRef(raw) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
    const role = ROLES.includes(raw.role) ? raw.role : '';
    const chatId = Number(raw.chat_id);
    const ref = {
        address: oneSpace(raw.address),
        chat_id: Number.isInteger(chatId) ? chatId : 1,
        role,
        author_label: oneSpace(raw.author_label) || authorOf(role),
        kind: KINDS.includes(raw.kind) ? raw.kind : 'message',
        snippet: oneSpace(raw.snippet),
        // The acceptance-time lookup was unresolved; a click retries the current source state.
        unresolved: raw.unresolved === true,
        ts: oneSpace(raw.ts),
        task_id: oneSpace(raw.task_id),
        // The quoted message's first photo, as its own attachment view (the same safe upload URL the
        // bubble shows): a reply to a photo carries its thumbnail, never an arbitrary address.
        media: attachmentViews([raw.attachment]).find((view) => view.kind === 'image' && view.available) || null,
    };
    return ref.address || ref.snippet ? ref : null;
}

/** One quote of a relay (one of the owner's messages passed into a work) as a safe value, or null. */
function normalizeRelayQuote(raw) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
    const text = String(raw.text ?? '').trim();
    // His attachments travel with his words (a manifest per file); the panel names them, it does not re-serve them.
    const files = (Array.isArray(raw.attachments) ? raw.attachments : [])
        .map((item) => oneSpace(item?.name || item?.display_name || item?.filename || item?.label)).filter(Boolean);
    // A photo or file sent without a caption is still his message: the panel names the files, never a placeholder.
    if (!text && !files.length) return null;
    const source = raw.source && typeof raw.source === 'object' ? raw.source : {};
    const chatId = Number(source.chat_id);
    return { text, files, where: oneSpace(source.label), ts: oneSpace(source.ts),
        // The host's retained reference to the owner's original message (its row address), so the words a relay
        // carries can be replied to without minting a second owner message.
        address: oneSpace(source.address), chatId: Number.isInteger(chatId) ? chatId : 1 };
}

/** The host's typed `relay` (the owner's words passed into a work): every quote it carries, or null. */
export function normalizeRelay(raw) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
    // `quotes` holds every message the turn passed on; an older stored envelope has only its first, in the singular fields.
    const quotes = (Array.isArray(raw.quotes) && raw.quotes.length ? raw.quotes : [raw]).map(normalizeRelayQuote).filter(Boolean);
    return quotes.length ? { quotes, relayId: oneSpace(raw.relay_id) } : null;
}

/**
 * The panel for one reply reference. `composer: true` is the field's first line: a lead word
 * (`Replying to`) and a cancel button beside the jump button; otherwise it is the quote inside a bubble.
 * Returns null for a reference with nothing to show.
 */
export function createReplyQuote(raw, { onJump = null, onCancel = null, composer = false, formatTime = formatMsgTime } = {}) {
    const ref = normalizeReplyRef(raw);
    if (!ref) return null;
    const author = ref.author_label || tr('chat.reply.unknown_author', 'Message');
    const when = formatTime(ref.ts)?.short || '';
    const root = node('div', composer ? 'chat-quote chat-quote--composer' : 'chat-quote');
    if (composer) root.setAttribute('role', 'group');
    root.dataset.author = ref.role || 'neutral';
    root.dataset.kind = ref.kind;
    root.dataset.mode = composer ? 'composer' : 'reply';

    const main = node('button', 'chat-quote-main');
    main.type = 'button';
    if (ref.media) {
        // A photo's thumbnail leads the panel (decorative: the panel's name already says what is quoted).
        const thumb = node('img', 'chat-quote-thumb');
        thumb.src = ref.media.url;
        thumb.alt = '';
        thumb.loading = 'lazy';
        thumb.decoding = 'async';
        main.classList.add('has-thumb');
        main.append(thumb);
    }
    const head = node('span', 'chat-quote-head');
    if (composer) head.append(node('span', 'chat-quote-lead', tr('chat.reply.replying_to', 'Replying to')));
    head.append(node('span', 'chat-quote-author', author));
    if (when) head.append(node('span', 'chat-quote-when', `· ${when}`));
    // A message with no words (a photo, a file) has no snippet: the panel names what it holds instead of staying empty.
    const snippet = ref.snippet || (ref.unresolved ? tr('chat.reply.unresolved_retry', 'Open quoted message')
        : ref.media ? tr('chat.reply.photo', 'Photo') : tr('chat.reply.attachment', 'Attachment'));
    // One line, cut on a word with a clean ending (text_fit.js), never a CSS ellipsis after a full stop.
    main.append(head, fitText(node('span', 'chat-quote-text'), snippet, 1));
    const lead = composer ? tr('chat.reply.replying_to', 'Replying to') : tr('chat.reply.reply_to', 'Reply to');
    main.setAttribute('aria-label', `${lead} ${author}${when ? `, ${when}` : ''}: ${snippet}`);
    main.addEventListener('click', () => onJump?.(ref));
    root.append(main);

    if (composer) {
        const cancel = node('button', 'chat-quote-cancel');
        cancel.type = 'button';
        cancel.innerHTML = CANCEL_ICON;
        cancel.title = tr('chat.reply.cancel', 'Cancel reply');
        cancel.setAttribute('aria-label', tr('chat.reply.cancel', 'Cancel reply'));
        cancel.addEventListener('click', () => onCancel?.(ref));
        root.append(cancel);
    }
    return root;
}

// ---- The clamp of a long quotation (a relay panel's text) ----
let nextClampId = 0;
const CLAMP_LINES = 6;
/** His words on a note card (chat_relay_deck.js): two lines until the note is opened, which the deck fits them to. */
export const RELAY_CARD_LINES = 2;

/**
 * Clamp `text` to six lines (cut on a word, never `.…`; text_fit.js) and add a `Show all` / `Show less` toggle
 * into `host`, a row that already exists (the relay panel's head), so no
 * control floats on a row of its own. The toggle shows only while the clamp hides text or the text is open.
 */
export function clampWithToggle(text, host, { onDomWrite = (write) => write() } = {}) {
    if (!text || !host) return null;
    const words = text.textContent;
    text.classList.add('chat-clamp');
    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'btn btn-default btn-sm chat-clamp-toggle';
    toggle.hidden = true;
    text.id ||= `chat-clamp-${++nextClampId}`;
    toggle.setAttribute('aria-controls', text.id);
    toggle.setAttribute('aria-expanded', 'false');
    toggle.textContent = tr('chat.clamp.show_all', 'Show all');
    const onFit = (cut) => { if (text.dataset.expanded !== '1') toggle.hidden = !cut; };
    toggle.addEventListener('click', () => onDomWrite(() => {
        const open = text.dataset.expanded !== '1';
        if (open) text.dataset.expanded = '1'; else delete text.dataset.expanded;
        toggle.setAttribute('aria-expanded', String(open));
        toggle.textContent = open ? tr('chat.clamp.show_less', 'Show less') : tr('chat.clamp.show_all', 'Show all');
        toggle.hidden = false;
        fitText(text, words, open ? 0 : CLAMP_LINES, onFit);
        return true;
    }));
    host.append(toggle);
    fitText(text, words, CLAMP_LINES, onFit);
    return toggle;
}

/** One quoted message of a relay: who wrote it, where and when, his words (clamped) and the files that came with them. */
function createRelayPanel(quote, relayId, { onDomWrite, formatTime, card }) {
    const author = authorOf('user');
    const place = [quote.where, formatTime(quote.ts)?.short || ''].filter(Boolean);
    const root = node('div', 'chat-quote');
    root.dataset.author = 'user';
    root.dataset.mode = 'relay';
    if (relayId) root.dataset.relayId = relayId;
    root.setAttribute('role', 'group');
    root.setAttribute('aria-label', [tr('chat.quote.label', 'Quoted message'), author, ...place].join(', '));
    const head = node('div', 'chat-quote-head');
    head.append(node('span', 'chat-quote-author', author));
    if (place.length) head.append(node('span', 'chat-quote-when', `· ${place.join(' · ')}`));
    const text = quote.text ? node('div', 'chat-quote-text', quote.text) : null;
    root.append(head, ...(text ? [text] : []));
    if (quote.files.length) root.append(node('div', 'chat-quote-files', fmt('Attached: {names}', { names: quote.files.join(', ') })));
    if (text) {
        if (card) {
            text.classList.add('chat-clamp');
            fitText(text, quote.text, RELAY_CARD_LINES);
        } else clampWithToggle(text, head, { onDomWrite });
    }
    return root;
}

/**
 * A relay's panel: the owner's complete words, labelled with where and when he wrote them (`You · Main · 00:37`),
 * clamped to six lines with `Show all` in the head, or two lines when a note card owns their expansion.
 * The words stay selectable text (not inside a button). A relay
 * that passes several of his messages shows one such panel per message, in the host's order, inside one block.
 */
export function createRelayQuote(raw, { onDomWrite, formatTime = formatMsgTime, card = false } = {}) {
    const relay = normalizeRelay(raw);
    if (!relay) return null;
    const panels = relay.quotes.map((quote) => createRelayPanel(quote, relay.relayId, { onDomWrite, formatTime, card }));
    if (panels.length === 1) return panels[0];
    const group = node('div', 'chat-relay-quotes');
    group.append(...panels);
    return group;
}
