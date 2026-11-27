// One text row of a room's feed drawn as its bubble (docs/DESIGN.md "Chat authorship and System rows", "Replies"):
// the sender line; the words in their author's voice (the owner's literal with each explicit address linked,
// Ouroboros's sanitized Markdown, System text escaped unless `markdown: true`, a Skill Review its own disclosure, a
// known host prefix hidden at display only); its attachments, record link, pending note and time; and the doors a
// saved row offers: Copy with the exact words shown, Share's source (an Ouroboros message's own words,
// message_actions.js), the reply quote and Reply (chat_reply.js), the Project reference of another room's answer
// (project_answer.js). Its caller decides whether a row is new, places the bubble and enhances its Markdown once it
// is mounted. A host notice says its one line, Copy keeps that line, and its raw details open from Details.
import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { destroyChatMarkdown, linkify, renderChatMarkdown } from './chat_markdown.js';
import { hostRowDisplayText, hostSentenceText, noticeLine } from './host_notice_text.js';
import { tr } from './i18n.js';
import { rawDetails } from './ui_primitives.js';
import { attachmentCaption, attachmentViews } from './chat_attachments.js';
import { renderSkillReviewDisclosure, wireSkillReviewDisclosure } from './skill_review_card.js';
import { evidenceLinkHtml, formatMsgTime, senderLabel, stampNodeTimestamp } from './chat_activity.js';
import { adoptsRetainedCopy, captionRevisionKey, rowAddressStem, stampHistoryNode } from './chat_history_replay.js';
import { nextRevisionOrder } from './chat_timeline_order.js';
import { decorateContactSource } from './chat_media.js';
import { recordAnswerWords } from './message_actions.js';
import { PROJECT_ROW_TYPES, decorateProjectRow, syncSavedProjectContext } from './project_answer.js';
import { markSwarmLabel } from './swarm_presentation.js';

/**
 * What a text row's `.message` shows, the one path its bubble (below) and the room's search (chat_search_words.js)
 * read: a host sentence in the install's language, a known host prefix hidden, the owner's literal words without the
 * composer's file-name tail, Ouroboros's (or a Markdown System row's) sanitized Markdown, other System text escaped,
 * a Skill Review's disclosure. `attachments` are the owner's attachment views. Answers `{ rendered, rich, shown,
 * skillReview, unparsed }`: the `.message` HTML, whether it is rich Markdown (enhanced once mounted), the words Copy
 * keeps, whether the row is a Skill Review and whether Markdown was left unparsed (no parser: its marks show).
 */
export function bubbleWords(text, role, { markdown = false, systemType = '', source = '', attachments = [], narration,
    textPlaceholder = false, skillReview: review = null, notice = null } = {}) {
    const stored = text;
    if (role === 'system' && ['task_pause_notice', 'legacy_memory_notice'].includes(systemType)) text = hostSentenceText(text);
    // A notice's line is what it says and what Copy keeps; other rows: render-only, Copy keeps `shown`.
    text = noticeLine({ role, narration, notice }) || text;
    const painted = hostRowDisplayText({ text: stored, role, system_type: systemType, narration, notice }, text);
    const skillReview = role === 'system' && systemType === 'skill_review' && narration !== true;
    const rich = role !== 'user' && !skillReview && (role !== 'system' || markdown === true);
    // Only the web composer writes the file-name tail; a Telegram/skill caption is kept word for word.
    const shown = attachments.length ? attachmentCaption(text, attachments,
        { placeholder: textPlaceholder === true, composed: source === 'web' }) : text;
    let unparsed = false;  // Markdown the page could not parse shows its marks: other coordinates than a parsed page's
    const rendered = role === 'user'
        ? linkify(shown)
        : skillReview
            ? renderSkillReviewDisclosure(text, review || null)
            : role === 'system' && markdown !== true
                ? escapeHtml(painted)
                : renderChatMarkdown(painted, { onPlain: () => { unparsed = true; } });
    return { rendered, rich, shown, skillReview, unparsed };
}

/**
 * One room's text bubbles. `chatMedia` is the room's media owner (chat_media.js: a row's attachments, its Copy and the
 * room's hold on what a row's action opens), `replies` its replies (chat_reply.js) and `feed` its column of rows;
 * `isMain` and `ownProjectId` name the room; `onDomWrite` is its stable-viewport seam. The returned
 * `draw(text, role, { markdown, ts, isProgress, meta, legacyKey, opts })` takes a row as its caller normalized it
 * (`meta`: the identity fields that also deduplicate it; `legacyKey`: its key without a history id) and answers
 * `{ bubble, rich }`: the bubble, built but not yet placed, and whether its words are rich Markdown.
 */
export function createTextBubbles({ chatMedia, replies, feed, isMain = false, ownProjectId = '',
    onDomWrite = (write) => write(), enhanceMarkdown = () => {} }) {
    const earlyPresentation = new WeakMap();
    // Per-node provenance of the latest words drawn/revised onto each bubble (WeakMap: GC'd with removed DOM nodes).
    const revisionState = new WeakMap();
    const isHistoryWrite = (opts) => Boolean(opts?.historyId || Number.isFinite(opts?.readOrder) || Number.isFinite(opts?.requestedAt));
    function recordRevision(node, opts, live) {
        if (live) {
            revisionState.set(node, { order: nextRevisionOrder(), at: Date.now(), live: true });
            return;
        }
        const order = Number.isFinite(opts?.readOrder) ? opts.readOrder : null;
        const at = Number.isFinite(opts?.requestedAt) ? opts.requestedAt : null;
        if (order === null && at === null) return;
        const current = revisionState.get(node);
        if (current && order !== null && current.order !== null && order < current.order) return;
        revisionState.set(node, { order, at, live: false });
    }
    function isStaleHistoryWrite(node, opts) {
        if (!isHistoryWrite(opts)) return false;
        const current = revisionState.get(node);
        if (!current) return false;
        const order = Number.isFinite(opts?.readOrder) ? opts.readOrder : null;
        if (order !== null && current.order !== null) return order < current.order;
        const at = Number.isFinite(opts?.requestedAt) ? opts.requestedAt : null;
        if (at !== null && current.at !== null) return current.live ? current.at >= at : at < current.at;
        return false;
    }
    function draw(text, role, { markdown = false, ts, isProgress = false, meta, legacyKey = '', opts = {}, existing = null }) {
        const pending = !!opts.pending;
        const ephemeral = !!opts.ephemeral;
        const { source, systemType, projectId } = meta;
        const attachments = role === 'user' ? attachmentViews(opts.attachments) : [];
        const bubble = existing || document.createElement('div');
        if (!existing) bubble.className = `chat-bubble ${role}` + (isProgress ? ' progress' : '');
        if (pending) bubble.classList.add('pending');
        if (ephemeral) bubble.dataset.ephemeral = '1';
        for (const key of ['clientMessageId', 'systemType', 'senderSessionId', 'taskId', 'deliveryId', 'projectId']) {
            if (meta[key]) bubble.dataset[key] = meta[key];
        }
        if (role === 'user' && Number.isInteger(meta.ownerChatId)) bubble.dataset.ownerChatId = String(meta.ownerChatId);
        // A Project's retained copy names its original's room (history's `chat_id`), the room that room's receipts for
        // it name (chat_history_replay.statedOwnerRoom). A copy from before copies named it states none.
        if (role === 'user' && opts.originProjected && Number.isInteger(opts.reply?.chat_id)) {
            bubble.dataset.sourceChatId = String(opts.reply.chat_id);
            if (opts.reply.source_row_address) bubble.dataset.sourceRowAddress = opts.reply.source_row_address;
        }
        const captionKey = captionRevisionKey(opts.reply, role, { ...meta, ts });
        if (captionKey) bubble.dataset.captionKey = captionKey;
        if (legacyKey) bubble.dataset.messageKey = legacyKey;
        stampHistoryNode(bubble, opts.historyId, opts.historyPosition);

        const notice = role === 'system' ? opts.reply?.notice : null;
        // A notice's phrase alone, for a reader that sets its step under it (the call card); the bubble says the line.
        if (typeof notice?.phrase === 'string' && notice.phrase && opts.narration !== true) bubble.dataset.noticePhrase = notice.phrase;
        else delete bubble.dataset.noticePhrase;
        const { rendered, rich: richMarkdown, shown, skillReview, unparsed } = bubbleWords(text, role, { markdown, systemType, source,
            attachments, narration: opts.narration, textPlaceholder: opts.textPlaceholder, skillReview: opts.skillReview, notice });
        if (skillReview) bubble.classList.add('chat-skill-review');
        // A passage's coordinates are the parsed words': an unparsed message offers none, nor lands a saved one.
        const authored = ['user', 'assistant'].includes(role) && !isProgress && opts.textPlaceholder !== true && Boolean(shown)
            && !unparsed && !opts.wordsOnly;
        const timeFmt = formatMsgTime(ts);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const pendingHtml = pending ? `<div class="msg-pending">Queued until reconnect</div>` : '';
        // Preserve the card's source link in a standalone row.
        const evidenceHtml = evidenceLinkHtml(opts.evidenceRef);
        const footer = `${evidenceHtml}${pendingHtml}${timeHtml}`;
        const label = senderLabel(role, isProgress, systemType, meta);
        const body = existing?.querySelector('.message');
        if (!body) {
            bubble.innerHTML = `
                <div class="sender">${escapeHtml(label)}</div>
                <div class="message${richMarkdown ? ' ui-rich-content' : ''}"${authored ? ' data-reply-body' : ''}>${rendered}</div>
                ${footer}
            `;
        } else {
            // Overflow can decorate mounted markup. Compare the serializer's two outputs, not its output with live DOM.
            if (earlyPresentation.get(bubble) !== rendered) body.innerHTML = rendered;
            earlyPresentation.delete(bubble);
            body.classList.toggle('ui-rich-content', richMarkdown);
            const sender = bubble.querySelector('.sender');
            if (sender && sender.textContent !== label) sender.textContent = label;
            if (authored) body.setAttribute('data-reply-body', '');
            else body.removeAttribute('data-reply-body');
            const time = bubble.querySelector('.msg-time');
            if (evidenceHtml || pending || time?.getAttribute('title') !== timeFmt?.full) {
                time?.remove();
                const tail = document.createElement('div');
                tail.innerHTML = footer;
                bubble.append(...tail.children);
            }
        }
        stampNodeTimestamp(bubble, ts);
        if (opts.wordsOnly) {
            earlyPresentation.set(bubble, rendered);
            bubble.dataset.windowWords = '';
            return { bubble, rich: false };
        }
        delete bubble.dataset.windowWords;
        if (notice?.details && opts.narration !== true) bubble.querySelector('.message')
            .after(rawDetails(document, notice.details, tr('chat.notice.details', 'Details')));
        decorateContactSource(bubble, opts.presenceContact);
        if (attachments.length) chatMedia.mountAttachments(bubble, attachments, shown);
        if (!isProgress && shown) {
            chatMedia.attachCopyControl(bubble, String(shown));
            // What Share passes on: Ouroboros's own message, Copy's exact words (a note between works is not one).
            if (role === 'assistant' && systemType !== 'relay_entry' && !String(systemType).startsWith('peer_letter') && opts.textPlaceholder !== true) {
                recordAnswerWords(bubble, String(shown), (close) => chatMedia.holdTransient(bubble, close));
            }
        }
        if (!isProgress) { replies.decorate(bubble, opts.reply); markSwarmLabel(feed, opts.reply, bubble); }
        // Legacy start/transfer rows without task or Project stay plain text with Copy; an answer tied to
        // another room's work ends with that Project reference, unfolded.
        const projectRow = PROJECT_ROW_TYPES.has(systemType);
        if (projectRow ? !(isMain && ['project_handoff', 'project_started'].includes(systemType))
            : role === 'assistant' && projectId !== ownProjectId) decorateProjectRow(bubble, { ...meta, ...opts, role,
            fold: projectRow && role === 'assistant', addedAt: ts });
        syncSavedProjectContext(bubble, opts.originProjected, opts.originId);
        if (skillReview) wireSkillReviewDisclosure(bubble, { onDomWrite, own: dispose => chatMedia.own(dispose, bubble) });
        if (isHistoryWrite(opts)) recordRevision(bubble, opts, false);
        else if (!pending && !ephemeral && (captionKey || opts.reply?.row_address)) recordRevision(bubble, opts, true);
        return { bubble, rich: richMarkdown };
    }
    // Revise an already-mounted live-call caption (or same-stem revised row) in place without replacing its DOM node
    // or re-rendering unchanged ordinary history rows.
    draw.revise = function revise(node, text, role, { markdown = false, isProgress = false, meta = {}, legacyKey = '', opts = {} } = {}) {
        if (!node || opts.originProjected || node.dataset.originProjected) return false;
        const captionKey = captionRevisionKey(opts.reply, role, meta);
        const prevAddress = node.dataset.rowAddress || '';
        const nextAddress = String(opts.reply?.row_address || '');
        const sameStemRevised = Boolean(prevAddress && nextAddress && prevAddress !== nextAddress
            && rowAddressStem(prevAddress) && rowAddressStem(prevAddress) === rowAddressStem(nextAddress));
        const isCaption = Boolean(captionKey || node.dataset.captionKey);
        if (!isCaption && !sameStemRevised) return false;
        const fromHistory = isHistoryWrite(opts);
        if (prevAddress && nextAddress && prevAddress === nextAddress) {
            if (fromHistory && !isStaleHistoryWrite(node, opts)) recordRevision(node, opts, false);
            return false;
        }
        if (fromHistory && isStaleHistoryWrite(node, opts)) return false;
        const body = [...(node.children || [])].find((child) => child.classList?.contains('message')) || node.querySelector?.('.message');
        if (!body) return false;
        const { source, systemType } = meta;
        const attachments = role === 'user' ? attachmentViews(opts.attachments) : [];
        const notice = role === 'system' ? opts.reply?.notice : null;
        const { rendered, rich: richMarkdown, shown, unparsed } = bubbleWords(text, role, { markdown, systemType, source,
            attachments, narration: opts.narration, textPlaceholder: opts.textPlaceholder, skillReview: opts.skillReview, notice });
        if (!sameStemRevised && !nextAddress && body.innerHTML === rendered) return false;

        destroyChatMarkdown(node);
        body.innerHTML = rendered;
        body.classList.toggle('ui-rich-content', richMarkdown);
        const authored = ['user', 'assistant'].includes(role) && !isProgress && opts.textPlaceholder !== true && Boolean(shown)
            && !unparsed && !opts.wordsOnly;
        if (authored) body.setAttribute('data-reply-body', '');
        else body.removeAttribute('data-reply-body');
        if (captionKey) node.dataset.captionKey = captionKey;
        if (legacyKey) node.dataset.messageKey = legacyKey;

        const doc = node.ownerDocument || globalThis.document;
        const actionsRow = [...(node.children || [])].find((child) => child.classList?.contains('chat-message-actions')) || null;
        const oldCopy = [...((actionsRow || node).children || [])].find((child) => child.classList?.contains('chat-message-copy')) || null;
        const hadCopyFocus = Boolean(oldCopy && doc?.activeElement === oldCopy);
        const copyBefore = oldCopy && actionsRow && oldCopy.parentNode === actionsRow
            ? (oldCopy.nextSibling ?? oldCopy.nextElementSibling) : null;
        if (oldCopy) { chatMedia.release?.(oldCopy); oldCopy.remove(); }
        if (!isProgress && shown) {
            const copy = chatMedia.attachCopyControl(node, String(shown));
            if (copy && actionsRow && copy.parentNode !== actionsRow) {
                if (copyBefore && copyBefore.parentNode === actionsRow) actionsRow.insertBefore(copy, copyBefore);
                else actionsRow.appendChild(copy);
            }
            if (hadCopyFocus) copy?.focus?.({ preventScroll: true });
            if (role === 'assistant' && systemType !== 'relay_entry' && !String(systemType).startsWith('peer_letter') && opts.textPlaceholder !== true) {
                recordAnswerWords(node, String(shown), (close) => chatMedia.holdTransient(node, close));
            }
        }
        if (!isProgress) {
            replies.stamp?.(node, opts.reply);
            replies.syncDerivedQuotes?.(node);
        }
        if (richMarkdown) enhanceMarkdown(node);
        // call_card.js observes `feed` with `{ childList: true }`: re-inserting a folded node at its exact index
        // keeps its DOM identity while queueing a childList mutation so the folded call card refreshes.
        if (node.classList?.contains('call-folded') && node.parentNode === feed) {
            feed.insertBefore(node, node.nextSibling ?? node.nextElementSibling);
        }
        recordRevision(node, opts, !fromHistory);
        return true;
    };
    draw.reconcile = function reconcile(node, text, role, { markdown = false, isProgress = false, meta = {}, legacyKey = '', opts = {}, adopt = false } = {}) {
        const revised = draw.revise(node, text, role, { markdown, isProgress, meta, legacyKey, opts });
        const adopted = adopt && adoptsRetainedCopy(node, opts) && replies.adopt(opts.reply, node);
        const quoted = replies.syncDerivedQuestion(node, opts.reply, { authoritative: true });
        const origin = syncSavedProjectContext(node, opts.originProjected, opts.originId);
        return revised || adopted || quoted || origin;
    };
    return draw;
}

/** Reconcile early words against saved history without tearing away an interaction outside a bounded page. */
export function createEarlyTextBubbles(feed, draw, insert, release, write = fn => fn(), shown = () => {},
    kept = () => [], retired = () => {}) {
    const doc = feed.ownerDocument;
    const nodes = new Map();
    const omitted = new Set();
    let invalidated = false;
    function remove(address, node) {
        nodes.delete(address); omitted.delete(address);
        release(node); retired(address);
    }
    function held(node) {
        if (node.contains(doc.activeElement)) return true;
        const selection = doc.getSelection?.();
        const ranges = !selection?.isCollapsed ? Array.from({ length: selection?.rangeCount || 0 }, (_, i) => selection.getRangeAt(i)) : [];
        if (ranges.some(range => range.intersectsNode?.(node))) return true;
        return (kept() || []).some(([start, from, end, to]) => {
            if (!start.isConnected || !end.isConnected) return false;
            try {
                const range = doc.createRange();
                range.setStart(start, from); range.setEnd(end, to); return range.intersectsNode(node);
            }
            catch { return false; }
        });
    }
    function clear(rows = null, scope = '') {
        if (!nodes.size) return false;
        let changed = false;
        const named = new Set(rows?.map(row => row.row_address));
        for (const [address, node] of nodes) {
            if (!rows || scope === 'complete' || named.has(address)) { remove(address, node); changed = true; }
            else if (scope) omitted.add(address);
        }
        return releaseOmitted() || changed;
    }
    function releaseOmitted() {
        if (doc.hidden) return false;
        const due = [...omitted].filter(address => !held(nodes.get(address)));
        for (const address of due) remove(address, nodes.get(address));
        return due.length > 0;
    }
    const forget = () => { invalidated = doc.hidden; if (!invalidated && nodes.size) write(clear); };
    const onRelease = () => { if (omitted.size && !doc.hidden) write(releaseOmitted); };
    const onBlur = () => { if (omitted.size) queueMicrotask(onRelease); };  // focus is still moving
    const events = ['selectionchange', 'ouro:retained-selection-released'];
    for (const event of events) doc.addEventListener(event, onRelease);
    feed.addEventListener('focusout', onBlur);
    return {
        get: address => nodes.get(address),
        delete(address) { nodes.delete(address); omitted.delete(address); retired(address); },
        show(rows) {
            if (invalidated) forget();
            if (rows === null) return forget();
            onRelease();
            if (doc.hidden || !rows.length) return;
            const mounted = new Set([...feed.querySelectorAll('.chat-bubble.assistant[data-row-address]')]
                .map(node => node.dataset.rowAddress));
            const pending = rows.filter(({ address }) => !mounted.has(address) && !nodes.has(address));
            if (pending.length) write(() => {
                for (const { address, text, markdown, ts, initiator } of pending) {
                    const { bubble } = draw(text, 'assistant', { markdown, ts, meta: { initiator }, opts: { wordsOnly: true } });
                    nodes.set(address, bubble);
                    if (insert(bubble, { forceStick: true })) shown(address);
                }
                return true;
            });
        },
        clear,
        forget,
        destroy() {
            for (const event of events) doc.removeEventListener(event, onRelease);
            feed.removeEventListener('focusout', onBlur);
            nodes.clear(); omitted.clear();
        },
    };
}
