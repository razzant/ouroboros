import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { showToast } from './toast.js';
import { downloadViaHostBridge, openViaHostBridge } from './ui_helpers.js';

// Delivered-document bubbles for ONE chat instance: the builder that turns a
// send_document payload into a file row with open/download affordances, the
// dedup key that lets the live socket frame and its history replay describe the
// same file, and the insert that combines the two. The instance hands over its
// dedup window and the identity/insert helpers it owns, so the bubble lands in
// the right transcript with the right sender and a sortable timestamp.
export function createDocumentBubbles({
    seenMessageKeys,
    getSenderLabel,
    formatMsgTime,
    stampNodeTimestamp,
    rememberMessageKey,
    insertMessageNode,
}) {
    // Shared document-bubble builder for both live WS frames and history replay.
    // Download priority: a durable server download_url routed through
    // downloadViaHostBridge (desktop host-bridge saves to Downloads instead of
    // navigating the WKWebView fullscreen; browser falls back to fetch+blob),
    // else an in-memory base64 blob (live-only), else a disabled label.
    function buildDocumentBubble(msg) {
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
        const mime = /^[A-Za-z0-9!#$&^_.+-]+\/[A-Za-z0-9!#$&^_.+-]+$/.test(String(msg.mime || ''))
            ? String(msg.mime)
            : 'application/octet-stream';
        const fileBase64 = /^[A-Za-z0-9+/=\s]+$/.test(String(msg.file_base64 || ''))
            ? String(msg.file_base64 || '').replace(/\s+/g, '')
            : '';
        const downloadUrl = /^\/api\/files\/download\?/.test(String(msg.download_url || ''))
            ? String(msg.download_url)
            : '';
        const filename = String(msg.filename || 'file').replace(/[\r\n]+/g, ' ').slice(0, 200);
        const canDownload = Boolean(downloadUrl || fileBase64);
        // Body click = open in default OS app (external window); a separate ↓
        // button saves to ~/Downloads. Both degrade to a base64 blob when only
        // the live payload is present (no durable server URL to hand the bridge).
        const openHtml = canDownload
            ? `<button type="button" class="chat-file" data-open="1">📎 ${escapeHtml(filename)}</button>`
            : `<span class="chat-file chat-file-empty">📎 ${escapeHtml(filename)}</span>`;
        const downloadHtml = canDownload
            ? `<button type="button" class="chat-file-download" data-download="1" title="Download" aria-label="Download">↓</button>`
            : '';
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(sender)}</div>
            ${captionHtml}
            <div class="message"><div class="chat-file-row">${openHtml}${downloadHtml}</div></div>
            ${timeHtml}
        `;
        const saveBlobFallback = () => {
            const bytes = Uint8Array.from(atob(fileBase64), (c) => c.charCodeAt(0));
            const blobUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
            const tmp = document.createElement('a');
            Object.assign(tmp, { href: blobUrl, download: filename, rel: 'noopener' });
            document.body.appendChild(tmp);
            tmp.click();
            tmp.remove();
            setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
        };
        const openBtn = bubble.querySelector('.chat-file[data-open]');
        if (openBtn && canDownload) {
            openBtn.addEventListener('click', async () => {
                try {
                    if (downloadUrl) {
                        await openViaHostBridge(downloadUrl, filename);
                        return;
                    }
                    saveBlobFallback();
                } catch (err) {
                    showToast(`Could not open file: ${err && err.message ? err.message : err}`, 'error');
                }
            });
        }
        const dlBtn = bubble.querySelector('.chat-file-download[data-download]');
        if (dlBtn && canDownload) {
            dlBtn.addEventListener('click', async () => {
                try {
                    if (downloadUrl) {
                        await downloadViaHostBridge(downloadUrl, filename);
                        return;
                    }
                    saveBlobFallback();
                } catch (err) {
                    showToast(`Could not download file: ${err && err.message ? err.message : err}`, 'error');
                }
            });
        }
        stampNodeTimestamp(bubble, rawTs);
        return bubble;
    }

    // Dedup key shared by the live WS insert and history replay of the SAME
    // document (send_document uses one ts for both the frame and the persisted
    // row), so a routine background sync (rebuildAll=false, bubbles not cleared)
    // does not re-insert an already-rendered file bubble.
    function documentMessageKey(msg) {
        return [
            'document',
            String(msg.ts || ''),
            String(msg.download_url || ''),
            String(msg.filename || ''),
            String(msg.caption || ''),
        ].join('|');
    }

    function appendDocumentBubble(msg) {
        const key = documentMessageKey(msg);
        if (key && seenMessageKeys.has(key)) return false;
        rememberMessageKey(key);
        insertMessageNode(buildDocumentBubble(msg));
        return true;
    }

    return { buildDocumentBubble, documentMessageKey, appendDocumentBubble };
}
