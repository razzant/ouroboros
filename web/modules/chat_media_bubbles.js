// Photo and video WS frames rendered as media bubbles. Ownership transfer
// from chat.js (v7 W3 wave D): the anonymous onWs('photo'|'video') handler
// bodies move here as the named handlePhotoFrame/handleVideoFrame members of
// the createMediaBubbles instance factory, with their captured helpers lifted
// to explicit factory parameters of the same names; chat.js keeps only the
// two onWs registrations.

import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';

export function createMediaBubbles({
    isMyThread,
    hideTypingIndicatorOnly,
    syncChatStatus,
    getSenderLabel,
    formatMsgTime,
    stampNodeTimestamp,
    insertMessageNode,
    incrementUnreadIfNeeded,
}) {
    function handlePhotoFrame(msg) {
        if (!isMyThread(msg)) return;
        // Media frames carry no activity identity: hide the dots row for the
        // incoming bubble but leave the authoritative active set intact (4A) —
        // syncChatStatus re-derives the header from live state.
        hideTypingIndicatorOnly();
        syncChatStatus();
        const role = msg.role === 'user' ? 'user' : 'assistant';
        const sender = role === 'user'
            ? getSenderLabel('user', false, '', {
                source: msg.source || '',
                senderLabel: msg.sender_label || '',
                senderSessionId: msg.sender_session_id || '',
            })
            : 'Ouroboros';
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}`;
        const rawTs = msg.ts || new Date().toISOString();
        const timeFmt = formatMsgTime(rawTs);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const captionHtml = msg.caption ? `<div class="message">${escapeHtml(msg.caption)}</div>` : '';
        const mime = /^image\/[a-z0-9.+-]+$/i.test(String(msg.mime || '')) ? String(msg.mime) : 'image/png';
        const imageBase64 = /^[A-Za-z0-9+/=\s]+$/.test(String(msg.image_base64 || ''))
            ? String(msg.image_base64 || '').replace(/\s+/g, '')
            : '';
        const imageUrl = imageBase64 ? `data:${mime};base64,${imageBase64}` : '';
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(sender)}</div>
            ${captionHtml}
            <div class="message"><img class="chat-photo" src="${escapeHtmlAttr(imageUrl)}" alt="Photo attachment"></div>
            ${timeHtml}
        `;
        const img = bubble.querySelector('.chat-photo');
        if (img && imageUrl) {
            img.addEventListener('click', () => window.open(imageUrl, '_blank'));
        }
        stampNodeTimestamp(bubble, rawTs);
        insertMessageNode(bubble);
        incrementUnreadIfNeeded(msg);
    }

    function handleVideoFrame(msg) {
        if (!isMyThread(msg)) return;
        hideTypingIndicatorOnly();
        syncChatStatus();
        const role = msg.role === 'user' ? 'user' : 'assistant';
        const sender = role === 'user'
            ? getSenderLabel('user', false, '', {
                source: msg.source || '',
                senderLabel: msg.sender_label || '',
                senderSessionId: msg.sender_session_id || '',
            })
            : 'Ouroboros';
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}`;
        const rawTs = msg.ts || new Date().toISOString();
        const timeFmt = formatMsgTime(rawTs);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const captionHtml = msg.caption ? `<div class="message">${escapeHtml(msg.caption)}</div>` : '';
        const mime = /^video\/[a-z0-9.+-]+$/i.test(String(msg.mime || '')) ? String(msg.mime) : 'video/mp4';
        const videoBase64 = /^[A-Za-z0-9+/=\s]+$/.test(String(msg.video_base64 || ''))
            ? String(msg.video_base64 || '').replace(/\s+/g, '')
            : '';
        const videoUrl = videoBase64 ? `data:${mime};base64,${videoBase64}` : '';
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(sender)}</div>
            ${captionHtml}
            <div class="message"><video class="chat-video" src="${escapeHtmlAttr(videoUrl)}" controls></video></div>
            ${timeHtml}
        `;
        stampNodeTimestamp(bubble, rawTs);
        insertMessageNode(bubble);
        incrementUnreadIfNeeded(msg);
    }

    return {
        handlePhotoFrame,
        handleVideoFrame,
    };
}
