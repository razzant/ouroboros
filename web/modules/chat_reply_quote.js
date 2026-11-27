// The quote panel (docs/DESIGN.md "Replies"): the one component that shows another message inside a bubble.
// A reply shows one ellipsized line and has a button that jumps to the quoted message; with a cancel control it
// is the strip in the composer's field while the owner replies. A relay shows the owner's complete words (they ARE
// the instruction) clamped to six lines, with a real `Show all` control in its head; inside a relay note card it
// clamps to three lines and the card's `more`/`less` (chat_relay_deck.js) owns its expansion. The bar and the author
// line wear the QUOTED author's ink; the bubble around keeps its own voice. Every string is set as text: the
// quoted words are data, never markup.
import { currentLanguage, dictionaryRevision, fmt, tr, tx } from './i18n.js';
import { formatMsgTime, selectionInside } from './chat_activity.js';
import { attachmentViews } from './chat_attachments.js';
import { validExcerpt } from './chat_reply_excerpt.js';
import { ellipsize, fitText } from './text_fit.js';
import { bindScrollFade, replaceScrollFade } from './scroll_fade.js';

const ROLES = ['user', 'assistant', 'system'];
// The referenced row's kind as the host names it (CONTRACTS K5); anything else reads as a message.
const KINDS = ['message', 'work_card', 'question', 'completion', 'system'];
const DEFAULT_AUTHOR = {
    user: () => tr('chat.sender.you', 'You'),
    assistant: () => tr('chat.sender.ouroboros', 'Ouroboros'),
    system: () => tr('chat.sender.system', 'System'),
};
const ROOM_LABEL_KINDS = ['main', 'project'];
const DEFAULT_ROOM_LABEL = {
    main: () => tx('Main'),
    project: () => tr('project.reference.generic', 'Project'),
};
const CANCEL_ICON = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor"'
    + ' stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M4 4l8 8"></path><path d="M12 4l-8 8"></path></svg>';

const oneSpace = (value) => String(value ?? '').replace(/\s+/g, ' ').trim();
const QUOTE_SIGNATURES = new WeakMap();
// The ref exactly as the host gave it for a derived question quote, so a later in-place re-render
// (a dictionary change repaints only its tr/tx chrome) rebuilds this same quote: the author's label,
// the room's own name and the snippet are the host's data and never re-translate.
const QUOTE_REFS = new WeakMap();
export const authorOf = (role) => DEFAULT_AUTHOR[role]?.() || '';
/** The host ref a derived question quote was built from, or null for any other quote. */
export const derivedQuoteRaw = (node) => QUOTE_REFS.get(node) || null;
const roomLabelOf = (ref) => (ref?.room_label_kind ? DEFAULT_ROOM_LABEL[ref.room_label_kind]() : ref?.room || '');

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
    const roomKind = ROOM_LABEL_KINDS.includes(raw.room_label_kind) ? raw.room_label_kind : '';
    const ref = {
        address: oneSpace(raw.address),
        chat_id: Number.isInteger(chatId) ? chatId : 1,
        role,
        ...(raw.quote_role === 'user' ? { quote_role: 'user' } : {}),
        author_label: oneSpace(raw.author_label) || authorOf(role),
        kind: KINDS.includes(raw.kind) ? raw.kind : 'message',
        snippet: oneSpace(raw.snippet),
        ...(validExcerpt(raw.excerpt) ? { excerpt: { text: raw.excerpt.text, start_utf16: raw.excerpt.start_utf16 } } : {}),
        // The acceptance-time lookup was unresolved; a click retries the current source state.
        unresolved: raw.unresolved === true,
        // The room holding the quoted row, named by the host only where another room shows the quote: a press leaves.
        // The normalized form is `room`, so a ref that already passed here (a stored staged reply restored, an echo
        // row, a signature re-read) keeps its room instead of silently losing it.
        room: oneSpace(raw.room_label ?? raw.room),
        ...(roomKind ? { room_label_kind: roomKind } : {}),
        ts: oneSpace(raw.ts),
        task_id: oneSpace(raw.task_id),
        // The quoted message's first photo, as its own attachment view (the same safe upload URL the
        // bubble shows): a reply to a photo carries its thumbnail, never an arbitrary address.
        // The normalized form is `media`, so an already-normalized ref keeps its thumbnail through the same re-reads.
        media: attachmentViews([raw.attachment ?? raw.media]).find((view) => view.kind === 'image' && view.available) || null,
        // Set by the host only on an automatically derived question quote for a late answer (never on an explicit Reply).
        ...(raw.derived_question === true ? { derived_question: true } : {}),
    };
    return ref.address || ref.snippet ? ref : null;
}

/** Canonical comparison key for a normalized reply reference, or '' when it carries nothing to show. */
export function replyQuoteSignature(raw) {
    const ref = normalizeReplyRef(raw);
    if (!ref) return '';
    return JSON.stringify([
        ref.address, ref.chat_id, ref.role, ref.quote_role || '', ref.author_label, ref.kind,
        ref.snippet, ref.excerpt?.text || '', ref.excerpt?.start_utf16 ?? -1, ref.unresolved,
        ref.room, ref.room_label_kind || '', currentLanguage(), dictionaryRevision(), ref.ts, ref.task_id,
        ref.media?.url || '', Boolean(ref.derived_question),
    ]);
}

/** The signature recorded for a derived question quote built by `createReplyQuote`, or '' for any other quote. */
export const quoteSignatureOf = (node) => QUOTE_SIGNATURES.get(node) || '';

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
export function createReplyQuote(raw, { onJump = null, onCancel = null, composer = false, onDomWrite, formatTime = formatMsgTime } = {}) {
    const ref = normalizeReplyRef(raw);
    if (!ref) return null;
    const author = ref.author_label || tr('chat.reply.unknown_author', 'Message');
    const when = formatTime(ref.ts)?.short || '';
    const roomLabel = roomLabelOf(ref);
    const root = node('div', composer ? 'chat-quote chat-quote--composer' : 'chat-quote');
    if (composer) root.setAttribute('role', 'group');
    root.dataset.author = ref.quote_role || ref.role || 'neutral';
    root.dataset.kind = ref.kind;
    if (ref.excerpt) { root.dataset.excerpt = '1'; root.dataset.address = ref.address; }
    if (!composer && ref.derived_question) {
        root.dataset.derivedQuestion = '1';
        if (roomLabel) root.dataset.away = '1';
        // Only this quote is reconciled in place (chat_reply.js syncDerivedQuestion); other quotes keep no signature.
        QUOTE_SIGNATURES.set(root, replyQuoteSignature(ref));
        QUOTE_REFS.set(root, raw);
    }
    root.dataset.mode = composer ? 'composer' : 'reply';

    const main = node('button', 'chat-quote-main');
    main.type = 'button';
    const excerptBody = ref.excerpt ? node('div', 'chat-quote-excerpt-body') : null;
    const body = excerptBody || main;
    const jump = () => onJump?.(ref);
    // Mixed chrome and source words: the chrome below uses tr; the DOM overlay must not walk the text or aria-label.
    main.dataset.i18nAuthored = '';
    if (excerptBody) excerptBody.dataset.i18nAuthored = '';
    if (ref.media) {
        // A photo's thumbnail leads the panel (decorative: the panel's name already says what is quoted).
        const thumb = node('img', 'chat-quote-thumb');
        thumb.src = ref.media.url;
        thumb.alt = '';
        thumb.loading = 'lazy';
        thumb.decoding = 'async';
        body.classList.add('has-thumb');
        body.append(thumb);
        if (excerptBody) thumb.addEventListener('click', jump);
    }
    const head = node('span', 'chat-quote-head');
    const selectionLead = ref.excerpt ? tr('chat.reply.sender_selection', 'Selected by sender · Source:') : '';
    if (selectionLead) head.append(node('span', 'chat-quote-lead', selectionLead));
    else if (composer) head.append(node('span', 'chat-quote-lead', tr('chat.reply.replying_to', 'Replying to')));
    head.append(node('span', 'chat-quote-author', author));
    // A row in another room names that room, and the arrow says a press goes there (DESIGN "References and actions").
    if (roomLabel) head.append(node('span', 'chat-quote-room', `· ${roomLabel}`));
    if (when) head.append(node('span', 'chat-quote-when', `· ${when}`));
    if (roomLabel) {
        const away = node('span', 'chat-quote-away', '↗');
        away.setAttribute('aria-hidden', 'true');
        head.append(away);
    }
    // A message with no words (a photo, a file) has no snippet: the panel names what it holds instead of staying empty.
    const snippet = ref.excerpt?.text || ref.snippet || (ref.unresolved ? tr('chat.reply.unresolved_retry', 'Open quoted message')
        : ref.media ? tr('chat.reply.photo', 'Photo') : tr('chat.reply.attachment', 'Attachment'));
    // One line, cut on a word with a clean ending (text_fit.js), never a CSS ellipsis after a full stop.
    const words = node('span', 'chat-quote-text', snippet);
    main.append(head);
    if (excerptBody) {
        // The jump button and the potentially focusable scroller are siblings. A pointer can still press the words;
        // a drag that selected words is a selection, not a jump (chat_activity.js selectionInside).
        excerptBody.append(main, words);
        words.addEventListener('click', (event) => {
            if (event.detail && selectionInside(words)) return;
            jump();
        });
    } else main.append(words);
    if (!ref.excerpt) fitText(words, snippet, 1);
    const lead = selectionLead || (composer ? tr('chat.reply.replying_to', 'Replying to') : tr('chat.reply.reply_to', 'Reply to'));
    // The name says in words what the room and the arrow say in pixels.
    const away = roomLabel ? `, ${fmt('opens in {room}', { room: roomLabel })}` : '';
    main.setAttribute('aria-label', `${lead} ${author}${away}${when ? `, ${when}` : ''}: ${snippet}`);
    main.addEventListener('click', jump);
    root.append(body);
    if (excerptBody) clampWithToggle(words, root, { onDomWrite, lines: 1,
        scrollRegion: tr('chat.reply.quotation_region', 'Selected quotation') });

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
// Selected excerpts and a work's cited prompt scroll while open. Relay prose instead grows in flow.
export function releaseClampFade(text) {
    replaceScrollFade(text);
    text?.classList?.remove('scroll-fade-y');
}
export const releaseQuoteFades = root => root?.querySelectorAll?.('.chat-clamp.scroll-fade-y').forEach(releaseClampFade);
/** His words on a note card (chat_relay_deck.js): two lines until the note is opened, which the deck fits them to. */
export const RELAY_CARD_LINES = 2;

/**
 * Clamp `text` to six lines (cut on a word, never `.…`; text_fit.js) and add a `Show all` / `Show less` toggle
 * into `host`, a row that already exists (the relay panel's head), so no
 * control floats on a row of its own. The toggle shows only while the clamp hides text or the text is open.
 * `fit` may measure a CSS-only clamp when authored child nodes must remain intact.
 */
export function clampWithToggle(text, host, { onDomWrite = (write) => write(), fit = fitText, lines = CLAMP_LINES, scrollRegion = '' } = {}) {
    if (!text || !host) return null;
    const document = text.ownerDocument || globalThis.document;
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
        fit(text, words, open ? 0 : lines, onFit);
        if (open && scrollRegion) {
            text.classList.add('scroll-fade-y');
            bindScrollFade(text, { region: scrollRegion });
        } else if (scrollRegion) releaseClampFade(text);
        return true;
    }));
    host.append(toggle);
    fit(text, words, lines, onFit);
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
