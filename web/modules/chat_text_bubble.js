// One text row of a room's feed drawn as its bubble (docs/DESIGN.md "Chat authorship and System rows", "Replies"):
// the sender line; the words in their author's voice (the owner's literal with each explicit address linked,
// Ouroboros's sanitized Markdown, System text escaped unless `markdown: true`, a Skill Review its own disclosure, a
// known host prefix hidden at display only); its attachments, record link, pending note and time; and the doors a
// saved row offers: Copy with the exact words shown, Share's source (an Ouroboros message's own words,
// message_actions.js), the reply quote and Reply (chat_reply.js), the Project reference of another room's answer
// (project_answer.js). Its caller decides whether a row is new, places the bubble and enhances its Markdown once it
// is mounted.
import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { linkify, renderChatMarkdown } from './chat_markdown.js';
import { hostRowDisplayText, hostSentenceText } from './host_notice_text.js';
import { attachmentCaption, attachmentViews } from './chat_attachments.js';
import { renderSkillReviewDisclosure, wireSkillReviewDisclosure } from './skill_review_card.js';
import { evidenceLinkHtml, formatMsgTime, senderLabel, stampNodeTimestamp } from './chat_activity.js';
import { stampHistoryNode } from './chat_history_replay.js';
import { decorateContactSource } from './chat_media.js';
import { recordAnswerWords } from './message_actions.js';
import { PROJECT_ROW_TYPES, decorateProjectRow, syncSavedProjectContext } from './project_answer.js';
import { markSwarmLabel } from './swarm_presentation.js';

/**
 * One room's text bubbles. `chatMedia` is the room's media owner (chat_media.js: a row's attachments, its Copy and the
 * room's hold on what a row's action opens), `replies` its replies (chat_reply.js) and `feed` its column of rows;
 * `isMain` and `ownProjectId` name the room; `onDomWrite` is its stable-viewport seam. The returned
 * `draw(text, role, { markdown, ts, isProgress, meta, legacyKey, opts })` takes a row as its caller normalized it
 * (`meta`: the identity fields that also deduplicate it; `legacyKey`: its key without a history id) and answers
 * `{ bubble, rich }`: the bubble, built but not yet placed, and whether its words are rich Markdown.
 */
export function createTextBubbles({ chatMedia, replies, feed, isMain = false, ownProjectId = '',
    onDomWrite = (write) => write() }) {
    return function draw(text, role, { markdown = false, ts, isProgress = false, meta, legacyKey = '', opts = {} }) {
        const pending = !!opts.pending;
        const ephemeral = !!opts.ephemeral;
        const { source, systemType, projectId } = meta;
        const attachments = role === 'user' ? attachmentViews(opts.attachments) : [];
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}` + (isProgress ? ' progress' : '');
        if (pending) bubble.classList.add('pending');
        if (ephemeral) bubble.dataset.ephemeral = '1';
        for (const key of ['clientMessageId', 'systemType', 'senderSessionId', 'taskId', 'deliveryId', 'projectId']) {
            if (meta[key]) bubble.dataset[key] = meta[key];
        }
        if (legacyKey) bubble.dataset.messageKey = legacyKey;
        stampHistoryNode(bubble, opts.historyId, opts.historyPosition);

        const stored = text;
        if (role === 'system' && ['task_pause_notice', 'legacy_memory_notice'].includes(systemType)) text = hostSentenceText(text);
        // Render-only: Copy below keeps `shown`, the pre-cleanup sentence.
        const painted = hostRowDisplayText({ text: stored, role, system_type: systemType, narration: opts.narration }, text);
        const skillReview = role === 'system' && systemType === 'skill_review' && opts.narration !== true;
        if (skillReview) bubble.classList.add('chat-skill-review');
        const richMarkdown = role !== 'user' && !skillReview && (role !== 'system' || markdown === true);
        // Only the web composer writes the file-name tail; a Telegram/skill caption is kept word for word.
        const shown = attachments.length ? attachmentCaption(text, attachments,
            { placeholder: opts.textPlaceholder === true, composed: source === 'web' }) : text;
        const rendered = role === 'user'
            ? linkify(shown)
            : skillReview
                ? renderSkillReviewDisclosure(text, opts.skillReview || null)
                : role === 'system' && markdown !== true
                    ? escapeHtml(painted)
                    : renderChatMarkdown(painted);
        const timeFmt = formatMsgTime(ts);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const pendingHtml = pending ? `<div class="msg-pending">Queued until reconnect</div>` : '';
        // Preserve the card's source link in a standalone row.
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(senderLabel(role, isProgress, systemType, meta))}</div>
            <div class="message${richMarkdown ? ' ui-rich-content' : ''}">${rendered}</div>
            ${evidenceLinkHtml(opts.evidenceRef)}
            ${pendingHtml}
            ${timeHtml}
        `;
        decorateContactSource(bubble, opts.presenceContact);
        if (attachments.length) chatMedia.mountAttachments(bubble, attachments, shown);
        if (!isProgress && shown) {
            chatMedia.attachCopyControl(bubble, String(shown));
            // What Share passes on: Ouroboros's own message, Copy's exact words (a note between works is not one).
            if (role === 'assistant' && systemType !== 'relay_entry' && opts.textPlaceholder !== true) {
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
        wireSkillReviewDisclosure(bubble, { onDomWrite });
        stampNodeTimestamp(bubble, ts);
        return { bubble, rich: richMarkdown };
    };
}
