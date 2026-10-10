import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { tr } from './i18n.js';
import { showToast } from './toast.js';
import { downloadViaHostBridge, normalizeTone, openViaHostBridge } from './ui_helpers.js';
import { MAX_LINK_ACTIONS } from './api_types.js';
import { apiFetch, taskArtifactDownloadUrl } from './api_client.js';
import { bindMenu } from './ui_interactions.js';
import { stampHistoryNode } from './chat_history_replay.js';
import { isFileDrag } from './chat_activity.js';
import { buildAttachmentBlock } from './chat_attachments.js';
import { createDocumentReader, documentReaderKind } from './document_reader.js';

const MIME_RE = /^[A-Za-z0-9!#$&^_.+-]+\/[A-Za-z0-9!#$&^_.+-]+$/;
const BASE64_RE = /^[A-Za-z0-9+/=\s]+$/;
const FILE_URL_RE = /^\/api\/files\/download\?/;
const SPEEDS = [1, 1.5, 2, 0.5];

function cleanMime(value, fallback) {
    const mime = String(value || '');
    return MIME_RE.test(mime) ? mime : fallback;
}

function cleanBase64(value) {
    const raw = String(value || '');
    return raw && BASE64_RE.test(raw) ? raw.replace(/\s+/g, '') : '';
}

function base64Bytes(value) {
    if (!value) return 0;
    const padding = value.endsWith('==') ? 2 : value.endsWith('=') ? 1 : 0;
    return Math.max(0, Math.floor(value.length * 3 / 4) - padding);
}

function humanSize(value) {
    if (value === null || value === undefined || value === '') return '';
    const bytes = Number(value);
    if (!Number.isFinite(bytes) || bytes < 0) return '';
    if (bytes < 1024) return `${bytes} B`;
    const units = ['KB', 'MB', 'GB'];
    let size = bytes;
    let unit = -1;
    do {
        size /= 1024;
        unit += 1;
    } while (size >= 1024 && unit < units.length - 1);
    return `${size >= 10 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`;
}

function fileExtension(filename) {
    const match = String(filename || '').match(/\.([^.\s]+)$/);
    return match ? match[1].slice(0, 10).toUpperCase() : 'FILE';
}

function formatDuration(seconds) {
    const total = Number.isFinite(Number(seconds)) ? Math.max(0, Math.floor(Number(seconds))) : 0;
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    return hours
        ? `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
        : `${minutes}:${String(secs).padStart(2, '0')}`;
}

// Toast dedupe for task incident frames (moved from chat.js, byte-ratchet
// extraction; the module-level set is deliberately shared across chat
// instances so a Main and a Project panel never double-toast one incident).
const shownIncidentToastKeys = new Set();

export function showTaskIncidentToast(msg) {
    const incident = String(msg?.task_incident || '').trim();
    if (!incident) return null;
    const key = String(msg?.toast_once || `${msg?.task_id || ''}:${incident}`).trim();
    if (!key || shownIncidentToastKeys.has(key)) return null;
    shownIncidentToastKeys.add(key);
    if (shownIncidentToastKeys.size > 500) {
        const oldest = shownIncidentToastKeys.values().next().value;
        shownIncidentToastKeys.delete(oldest);
    }
    // The incident's valence rides the frame (#628): a recovery is good news,
    // a wait is a warning, an exhaustion or a cancellation fault is the alarm.
    // No tone on the frame (older producers, cancellation_fault) keeps the
    // alarm tone; the text is never parsed for it.
    return showToast(String(msg?.content || msg?.text || incident), normalizeTone(msg?.toast_tone || 'error', 'error'));
}

// Best-effort teardown of temporary uploads after a failed send; lives with
// the attachment/media domain so chat.js stays within its byte ratchet.
export async function cleanupUploadedAttachments(uploaded) {
    const filenames = uploaded
        .map((item) => item.filename)
        .filter(Boolean);
    if (!filenames.length) return;
    const results = await Promise.allSettled(filenames.map(async (filename) => {
        const resp = await apiFetch('/api/chat/upload', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename }),
        });
        if (!resp.ok) throw new Error(`DELETE ${filename} failed with HTTP ${resp.status}`);
    }));
    const failed = results.filter((result) => result.status === 'rejected');
    if (failed.length) {
        console.warn('Failed to clean up uploaded chat attachments after send failure', failed);
    }
}

export function safeHttpUrl(value) {
    // safeExternalHrefAttr returns escaped markup; chat needs a raw URL before its own attribute escaping.
    const raw = String(value || '').trim();
    if (!/^https?:\/\//i.test(raw)) return '';
    try {
        const parsed = new URL(raw);
        return (parsed.protocol === 'http:' || parsed.protocol === 'https:') && parsed.host
            ? parsed.href
            : '';
    } catch {
        return '';
    }
}

export function bindComposerFileTargets({ page, inputArea, input, stagePendingFiles }) {
    // The two implicit routes a file takes into the composer beside the
    // paperclip: an image pasted into the textarea, and a file dropped
    // anywhere on the page. Both end in the caller's local stager, so
    // nothing uploads before Send, and the drop target is the whole page
    // while only the input area shows the active state.
    input.addEventListener('paste', (e) => {
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        const pastedImages = [];
        for (let i = 0; i < items.length; i += 1) {
            const item = items[i];
            if (item && item.kind === 'file' && typeof item.type === 'string' && item.type.startsWith('image/')) {
                const blob = item.getAsFile();
                if (!blob) continue;
                const ext = (item.type.split('/')[1] || 'png').split(';')[0].trim() || 'png';
                const ts = Date.now() + i;
                const safeBlob = blob instanceof File
                    ? new File([blob], `clipboard-${ts}.${ext}`, { type: blob.type })
                    : new File([blob], `clipboard-${ts}.${ext}`, { type: item.type });
                pastedImages.push(safeBlob);
            }
        }
        if (!pastedImages.length) return;
        e.preventDefault();
        stagePendingFiles(pastedImages);
    });

    let fileDragDepth = 0;
    function setFileDragActive(active) {
        inputArea.classList.toggle('drag-active', Boolean(active));
    }
    page.addEventListener('dragenter', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        fileDragDepth += 1;
        setFileDragActive(true);
    });
    page.addEventListener('dragover', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
        setFileDragActive(true);
    });
    page.addEventListener('dragleave', (event) => {
        if (!isFileDrag(event)) return;
        fileDragDepth = Math.max(0, fileDragDepth - 1);
        if (fileDragDepth === 0) setFileDragActive(false);
    });
    page.addEventListener('drop', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        fileDragDepth = 0;
        setFileDragActive(false);
        stagePendingFiles(event.dataTransfer?.files || []);
    });
}

export function createChatMedia({
    chatSessionId,
    durableChatMediaUrl,
    formatMsgTime,
    insertMessageNode,
    senderLabel,
    stampNodeTimestamp,
    onDomWrite = (mutate) => mutate(),
}) {
    const disposers = new Set();
    const resourceOwners = new Map();
    const timers = new Set();
    const timerOwners = new Map();
    const objectUrls = new Set();
    const groupingWrappers = new Set();
    const playerRegistry = new Set();
    const photoGroups = new Map();
    const fileGroups = new Map();
    let fileDialog = null;
    let dialogFile = null;
    let dialogOwner = null;  // the card's item (buildGallery may move it to an earlier group): releasing it or its holder closes the dialog
    let destroyed = false;

    // A disposer owned like a listener (a late answer's end too): its owner's release, reset or destroy runs it.
    function own(dispose, owner) { disposers.add(dispose); resourceOwners.set(dispose, owner); return dispose; }

    function listen(target, type, handler, options, owner = target) {
        if (!target) return () => {};
        target.addEventListener(type, handler, options);
        return own(() => target.removeEventListener(type, handler, options), owner);
    }

    function later(handler, delay, owner = null) {
        const timer = setTimeout(() => {
            timers.delete(timer);
            timerOwners.delete(timer);
            if (owner && !owner.isConnected) return;
            handler();
        }, delay);
        timers.add(timer);
        if (owner) timerOwners.set(timer, owner);
        return timer;
    }

    function messageSender(msg, role) {
        return role === 'user'
            ? senderLabel('user', false, '', {
                source: msg.source || '',
                senderLabel: msg.sender_label || '',
                senderSessionId: msg.sender_session_id || '',
            }, chatSessionId)
            : 'Ouroboros';
    }

    function bubbleFrame(msg, bodyHtml) {
        const role = msg.role === 'user' ? 'user' : 'assistant';
        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}`;
        const rawTs = msg.ts || new Date().toISOString();
        const timeFmt = formatMsgTime(rawTs);
        const timeHtml = timeFmt
            ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>`
            : '';
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(messageSender(msg, role))}</div>
            ${bodyHtml}
            ${timeHtml}
        `;
        stampNodeTimestamp(bubble, rawTs);
        if (msg.history_id) bubble.dataset.historyId = msg.history_id;
        if (msg.history_position) {
            bubble.dataset.historySource = msg.history_position.source;
            bubble.dataset.historyOffset = String(msg.history_position.offset);
        }
        return bubble;
    }

    function bubbleFrameNode(msg, node) {
        const bubble = bubbleFrame(msg, '');
        const time = bubble.querySelector('.msg-time');
        if (time) bubble.insertBefore(node, time); else bubble.append(node);
        return bubble;
    }

    function mediaSource(msg, type, mime) {
        const value = type === 'photo' ? msg.image_base64 : msg.video_base64;
        const base64 = cleanBase64(value);
        const durable = durableChatMediaUrl(msg.download_url);
        return base64 ? `data:${mime};base64,${base64}` : durable;
    }

    // Compat address for the same media bytes: packaged desktop launchers gate
    // their file bridge to a URL allowlist that predates the task-artifact
    // route, so the bridge is handed this form when the server offered one.
    // The browser keeps the canonical URL — this is an extra address, not a
    // replacement, and it is validated exactly like the document form.
    function compatMediaUrl(value) {
        const raw = String(value || '');
        return FILE_URL_RE.test(raw) ? raw : '';
    }

    // Bridge-facing view of one delivered media item. `durable` is the
    // canonical same-origin URL used for fetches; `bridge` is what a host-bridge
    // call should be given.
    function mediaSourceRef(msg, source) {
        // The displayed source may be a data: URI (live delivery) while the
        // frame still carries durable addresses; the bridge can only be handed
        // a URL, so those addresses are read from the frame, not the display.
        const isData = source.startsWith('data:');
        const canonical = isData ? durableChatMediaUrl(msg?.download_url) : source;
        return {
            base64: isData ? source.split(',')[1] || '' : '',
            durable: canonical,
            bridge: canonical ? (compatMediaUrl(msg?.download_url_compat) || canonical) : '',
        };
    }

    function fileSource(msg, mime) {
        const base64 = cleanBase64(msg.file_base64);
        const raw = String(msg.download_url || '');
        let canonical = '';
        try { canonical = taskArtifactDownloadUrl(msg.task_id, decodeURIComponent(raw.split('/').at(-1))); } catch {}
        const durable = raw && (raw === canonical || FILE_URL_RE.test(raw)) ? raw : '';
        return {
            base64,
            durable,
            // The reader takes only the delivered copy: inline bytes or the immutable artifact route.
            reader: base64 ? { base64 } : durable && durable === canonical ? { url: durable } : null,
            bridge: compatMediaUrl(msg?.download_url_compat) || durable,
            src: base64 ? `data:${mime};base64,${base64}` : durable,
        };
    }

    async function sourceBlob(source, mime) {
        if (source.base64) {
            const bytes = Uint8Array.from(atob(source.base64), (char) => char.charCodeAt(0));
            return new Blob([bytes], { type: mime });
        }
        if (!source.durable) throw new Error('File data is unavailable');
        const response = await apiFetch(source.durable);
        if (!response.ok) throw new Error(`Download failed (${response.status})`);
        return response.blob();
    }

    function downloadBlob(blob, filename) {
        const blobUrl = URL.createObjectURL(blob);
        objectUrls.add(blobUrl);
        const anchor = document.createElement('a');
        Object.assign(anchor, { href: blobUrl, download: filename, rel: 'noopener' });
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        later(() => {
            URL.revokeObjectURL(blobUrl);
            objectUrls.delete(blobUrl);
        }, 1000);
    }

    async function downloadSource(source, filename, mime) {
        if (source.durable) {
            await downloadViaHostBridge(source.bridge || source.durable, filename, { browserUrl: source.durable, streaming: true });
            return;
        }
        downloadBlob(await sourceBlob(source, mime), filename);
    }

    // The card's Open and Download, shared by the file dialog and the document reader.
    const fileActions = {
        async open(file) {
            try {
                await openViaHostBridge(file.source.bridge || file.source.durable, file.filename, { browserUrl: file.source.durable });
                return true;
            } catch (error) {
                showToast(`Could not open file: ${error?.message || error}`, 'error');
                return false;
            }
        },
        async download(file) {
            try {
                await downloadSource(file.source, file.filename, file.mime);
                return true;
            } catch (error) {
                showToast(`Could not download file: ${error?.message || error}`, 'error');
                return false;
            }
        },
    };
    const reader = createDocumentReader({ actions: fileActions, formatSize: humanSize });

    function ensureFileDialog() {
        if (fileDialog) return fileDialog;
        fileDialog = document.createElement('dialog');
        fileDialog.className = 'chat-file-dialog';
        fileDialog.innerHTML = `
            <form method="dialog" class="chat-file-dialog-panel">
                <div class="chat-file-dialog-title" data-i18n-authored></div>
                <div class="chat-file-dialog-actions">
                    <button type="button" class="btn btn-default" data-file-action="open">Open</button>
                    <button type="button" class="btn btn-default" data-file-action="download">Download</button>
                    <button type="button" class="btn btn-default" data-file-action="close">Close</button>
                </div>
            </form>`;
        document.body.appendChild(fileDialog);
        listen(fileDialog.querySelector('[data-file-action="close"]'), 'click', closeFileDialog);
        listen(fileDialog, 'cancel', closeFileDialog);
        listen(fileDialog.querySelector('[data-file-action="open"]'), 'click', async () => {
            if (dialogFile?.source.durable && await fileActions.open(dialogFile)) closeFileDialog();
        });
        listen(fileDialog.querySelector('[data-file-action="download"]'), 'click', async () => {
            if (dialogFile && await fileActions.download(dialogFile)) closeFileDialog();
        });
        return fileDialog;
    }

    function closeFileDialog() {
        dialogFile = null;
        dialogOwner = null;
        if (!fileDialog) return;
        if (typeof fileDialog.close === 'function') fileDialog.close();
        else fileDialog.removeAttribute('open');
    }

    function openFileDialog(file, owner = null) {
        const dialog = ensureFileDialog();
        dialogFile = file;
        dialogOwner = owner;
        dialog.querySelector('.chat-file-dialog-title').textContent = file.filename;
        const open = dialog.querySelector('[data-file-action="open"]');
        open.hidden = !file.source.durable;
        if (typeof dialog.showModal === 'function') dialog.showModal();
        else dialog.setAttribute('open', '');
    }

    function wirePlayer(root, media, { audio = false, source, filename, mime, fullscreenTarget = root }) {
        if (!root || !media) return;
        const play = root.querySelector('[data-media-action="play"]');
        const progress = root.querySelector('.chat-media-progress');
        const time = root.querySelector('.chat-media-time');
        const speed = root.querySelector('[data-media-action="speed"]');
        const rates = root.querySelector('.chat-media-rate-menu');
        const repeat = root.querySelector('[data-media-action="repeat"]');
        const mute = root.querySelector('[data-media-action="mute"]');
        const fullscreen = root.querySelector('[data-media-action="fullscreen"]');
        const update = () => {
            const duration = Number.isFinite(media.duration) ? media.duration : 0;
            const current = Number.isFinite(media.currentTime) ? media.currentTime : 0;
            const ratio = duration > 0 ? Math.min(100, current / duration * 100) : 0;
            progress.value = String(ratio);
            progress.style?.setProperty?.('--media-progress', `${ratio}%`);
            time.textContent = `${formatDuration(current)} / ${formatDuration(duration)}`;
        };
        const syncPlay = () => {
            play.textContent = media.paused ? '▶' : '❚❚';
            play.setAttribute('aria-label', media.paused ? tr('media.play', 'Play') : tr('media.pause', 'Pause'));
        };
        const toggle = async () => {
            try {
                if (media.paused) await media.play();
                else media.pause();
                syncPlay();
            } catch (error) {
                showToast(`Could not play media: ${error?.message || error}`, 'error');
            }
        };
        listen(play, 'click', toggle);
        if (!audio) listen(media, 'click', toggle);
        listen(media, 'play', syncPlay);
        listen(media, 'pause', syncPlay);
        listen(media, 'timeupdate', update);
        listen(media, 'durationchange', update);
        listen(media, 'loadedmetadata', update);
        listen(progress, 'input', () => {
            if (Number.isFinite(media.duration) && media.duration > 0) {
                media.currentTime = Number(progress.value) / 100 * media.duration;
                update();
            }
        });
        if (speed) listen(speed, 'click', () => {
            const current = Number(media.playbackRate || 1);
            const index = SPEEDS.findIndex((rate) => rate === current);
            const next = SPEEDS[(index + 1) % SPEEDS.length];
            media.playbackRate = next;
            speed.textContent = `×${next}`;
            if (rates) rates.value = String(next);
        });
        if (rates) listen(rates, 'change', () => {
            media.playbackRate = Number(rates.value) || 1;
            speed.textContent = `×${media.playbackRate}`;
        });
        if (repeat) listen(repeat, 'click', async () => {
            media.currentTime = 0;
            try { await media.play(); } catch {}
            syncPlay();
        });
        if (mute) listen(mute, 'click', () => {
            media.muted = !media.muted;
            mute.textContent = media.muted ? '🔇' : '🔊';
            mute.setAttribute('aria-label', media.muted ? tr('media.unmute', 'Unmute') : tr('media.mute', 'Mute'));
        });
        if (fullscreen) listen(fullscreen, 'click', async () => {
            try {
                const request = fullscreenTarget.requestFullscreen || fullscreenTarget.webkitRequestFullscreen;
                if (!request) throw new Error('Fullscreen is not supported');
                await request.call(fullscreenTarget);
            } catch (error) {
                showToast(`Could not enter fullscreen: ${error?.message || error}`, 'error');
            }
        });
        const download = root.querySelector('[data-media-action="download"]');
        if (download) listen(download, 'click', async () => {
            try {
                await downloadSource(source, filename, mime);
            } catch (error) {
                showToast(`Could not download media: ${error?.message || error}`, 'error');
            }
        });
        playerRegistry.add(media);
        update();
        syncPlay();
    }

    function playerHtml({ audio = false, src, title = '' }) {
        const media = audio
            ? `<audio preload="metadata" src="${escapeHtmlAttr(src)}"></audio>`
            : `<video preload="metadata" src="${escapeHtmlAttr(src)}" playsinline></video>`;
        const videoControls = audio ? '' : `
            <button type="button" data-media-action="speed" aria-label="${escapeHtmlAttr(tr('media.cycle_speed', 'Cycle playback speed'))}">×1</button>
            <select class="chat-media-rate-menu" aria-label="${escapeHtmlAttr(tr('media.playback_speed', 'Playback speed'))}">
                ${SPEEDS.map((rate) => `<option value="${rate}">×${rate}</option>`).join('')}
            </select>
            <button type="button" data-media-action="repeat" aria-label="${escapeHtmlAttr(tr('media.repeat', 'Repeat'))}">↻</button>
            <button type="button" data-media-action="mute" aria-label="${escapeHtmlAttr(tr('media.mute', 'Mute'))}">🔊</button>
            <button type="button" data-media-action="fullscreen" aria-label="${escapeHtmlAttr(tr('media.enter_fullscreen', 'Enter fullscreen'))}">⛶</button>`;
        return `<div class="chat-media-player${audio ? ' is-audio' : ''}">
            ${title ? `<div class="chat-media-title">${escapeHtml(title)}</div>` : ''}
            <div class="chat-media-stage">${media}</div>
            <div class="chat-media-controls">
                <button type="button" data-media-action="play" aria-label="${escapeHtmlAttr(tr('media.play', 'Play'))}">▶</button>
                <input class="chat-media-progress" type="range" min="0" max="100" value="0" step="0.1" aria-label="${escapeHtmlAttr(tr('media.progress', 'Media progress'))}">
                <span class="chat-media-time">0:00 / 0:00</span>
                ${videoControls}
                <button type="button" data-media-action="download" aria-label="${escapeHtmlAttr(tr('media.download_media', 'Download media'))}">↓</button>
            </div>
        </div>`;
    }

    function photoActionsHtml() {
        return `<details class="chat-photo-actions">
            <summary aria-label="${escapeHtmlAttr(tr('media.photo_actions', 'Photo actions'))}" aria-haspopup="menu" aria-expanded="false">•••</summary>
            <div class="chat-photo-menu ui-popup" role="menu" aria-label="${escapeHtmlAttr(tr('media.photo_actions', 'Photo actions'))}" hidden>
                <button type="button" role="menuitem" data-photo-action="open">${escapeHtml(tr('media.open_in_new_tab', 'Open in new tab'))}</button>
                <button type="button" role="menuitem" data-photo-action="download">${escapeHtml(tr('media.download', 'Download'))}</button>
                <button type="button" role="menuitem" data-photo-action="copy">${escapeHtml(tr('media.copy_to_clipboard', 'Copy to clipboard'))}</button>
            </div>
        </details>`;
    }

    function wirePhotoActions(item, source, sourceRef, filename, mime) {
        const action = (name) => item.querySelector(`[data-photo-action="${name}"]`);
        const details = item.querySelector('.chat-photo-actions');
        const trigger = details.querySelector('summary');
        const menu = details.querySelector('.chat-photo-menu');
        let binding = null;
        listen(trigger, 'click', (event) => {
            event.preventDefault();
            if (binding) return binding.close({ restoreFocus: true });
            document.body.appendChild(menu);
            menu.hidden = false;
            details.open = true;
            trigger.setAttribute('aria-expanded', 'true');
            binding = bindMenu(menu, { anchor: trigger, onClose: () => {
                binding = null;
                menu.hidden = true;
                details.open = false;
                trigger.setAttribute('aria-expanded', 'false');
                details.appendChild(menu);
            } });
        });
        listen(menu, 'click', (event) => {
            if (event.target.closest('[data-photo-action]')) binding?.close({ restoreFocus: true });
        }, { capture: true }, item);
        const disposeMenu = () => { binding?.destroy(); menu.remove(); };
        disposers.add(disposeMenu);
        resourceOwners.set(disposeMenu, item);

        // A durable photo rides the host-bridge helper with BOTH addresses
        // (bridge form for the launcher gate, canonical for browsers); a data:
        // display keeps window.open, whose shell interceptor saves the bytes.
        const openPhoto = () => {
            if (sourceRef.durable) {
                openViaHostBridge(sourceRef.bridge || sourceRef.durable, filename, { browserUrl: sourceRef.durable })
                    .catch((error) => showToast(`Could not open image: ${error?.message || error}`, 'error'));
                return;
            }
            window.open(source, '_blank', 'noopener');
        };
        listen(item.querySelector('.chat-photo'), 'click', openPhoto);
        listen(action('open'), 'click', openPhoto, undefined, item);
        listen(action('download'), 'click', async () => {
            try { await downloadSource(sourceRef, filename, mime); }
            catch (error) { showToast(`Could not download image: ${error?.message || error}`, 'error'); }
        }, undefined, item);
        listen(action('copy'), 'click', async () => {
            try {
                if (!navigator.clipboard?.write || typeof ClipboardItem !== 'function') {
                    throw new Error('copying images is not available here. Use Open in new tab or Download.');
                }
                const blob = await sourceBlob(sourceRef, mime);
                await navigator.clipboard.write([new ClipboardItem({ [blob.type || mime]: blob })]);
                showToast('Image copied.', 'ok');
            } catch (error) {
                showToast(`Could not copy image: ${error?.message || error}`, 'error');
            }
        }, undefined, item);
    }

    function buildMediaBubble(msg) {
        if (destroyed) return null;
        const type = msg.msg_type || msg.type;
        if (type !== 'photo' && type !== 'video') return null;
        const fallback = type === 'photo' ? 'image/png' : 'video/mp4';
        const mimePattern = type === 'photo' ? /^image\/[a-z0-9.+-]+$/i : /^video\/[a-z0-9.+-]+$/i;
        const mime = mimePattern.test(String(msg.mime || '')) ? String(msg.mime) : fallback;
        const source = mediaSource(msg, type, mime);
        if (!source) return null;
        const caption = String(msg.caption || '');
        if (type === 'video') {
            const extension = (mime.split('/')[1]?.split('+')[0] || '').slice(0, 10) || 'mp4';
            const sourceRef = mediaSourceRef(msg, source);
            const body = `${caption ? `<div class="message">${escapeHtml(caption)}</div>` : ''}
                <div class="message">${playerHtml({ src: source })}</div>`;
            const bubble = bubbleFrame(msg, body);
            const player = bubble.querySelector('.chat-media-player');
            wirePlayer(bubble, bubble.querySelector('video'), {
                source: sourceRef, filename: `video.${extension}`, mime, fullscreenTarget: player,
            });
            return bubble;
        }
        const filename = `image.${mime.split('/')[1]?.split('+')[0] || 'png'}`;
        const body = `<div class="message"><div class="chat-gallery-grid">
            <figure class="chat-gallery-item">
                <img class="chat-photo" src="${escapeHtmlAttr(source)}" alt="Photo attachment">
                ${photoActionsHtml()}
                ${caption ? `<figcaption>${escapeHtml(caption)}</figcaption>` : ''}
            </figure>
        </div></div>`;
        const bubble = bubbleFrame(msg, body);
        wirePhotoActions(bubble, source, mediaSourceRef(msg, source), filename, mime);
        return bubble;
    }

    function buildDocumentBubble(msg) {
        if (destroyed) return null;
        const mime = cleanMime(msg.mime, 'application/octet-stream');
        const source = fileSource(msg, mime);
        // The type is read from the whole name; only its display is cut.
        const fullName = String(msg.filename || 'file').replace(/[\r\n]+/g, ' ');
        const filename = fullName.slice(0, 200);
        const caption = String(msg.caption || '');
        const explicitSize = msg.size_bytes !== null && msg.size_bytes !== undefined
            && msg.size_bytes !== '' && Number.isFinite(Number(msg.size_bytes));
        const size = explicitSize ? Number(msg.size_bytes)
            : source.base64 ? base64Bytes(source.base64) : null;
        const meta = [fileExtension(fullName), humanSize(size)].filter(Boolean).join(' · ');
        const readable = source.reader ? documentReaderKind(fullName, mime) : '';
        const more = readable
            ? `<span class="chat-file-more is-read">${escapeHtml(tr('media.read', 'Read'))}</span>`
            : '<span class="chat-file-more" aria-hidden="true">•••</span>';
        let content;
        if (mime.startsWith('audio/') && source.src) {
            content = playerHtml({ audio: true, src: source.src, title: filename });
        } else {
            const preview = mime.startsWith('image/') && source.src
                ? `<img class="chat-file-thumb" src="${escapeHtmlAttr(source.src)}" alt="">`
                : '<span class="chat-file-glyph" aria-hidden="true">▤</span>';
            content = `<button type="button" class="chat-file-card" aria-haspopup="dialog" ${source.src ? '' : 'disabled'}>
                ${preview}
                <span class="chat-file-copy">
                    <span class="chat-file-name">${escapeHtml(filename)}</span>
                    <span class="chat-file-meta">${escapeHtml(meta)}</span>
                </span>
                ${more}
            </button>`;
        }
        const body = `<div class="message"><div class="chat-file-grid">
            <div class="chat-file-item">${content}${caption ? `<div class="chat-file-caption">${escapeHtml(caption)}</div>` : ''}</div>
        </div></div>`;
        const bubble = bubbleFrame(msg, body);
        if (mime.startsWith('audio/') && source.src) {
            wirePlayer(bubble, bubble.querySelector('audio'), {
                audio: true, source, filename, mime,
            });
        } else {
            const card = bubble.querySelector('.chat-file-card');
            // The item, not the bubble: a grouped file moves into an earlier bubble's grid.
            const item = bubble.querySelector('.chat-file-item');
            const file = { source, filename, mime, meta, kind: readable, reader: source.reader, size, canOpen: Boolean(source.durable) };
            if (card && source.src) listen(card, 'click', (event) => (readable
                ? reader.open(file, { owner: item, returnFocus: card, pointer: event?.detail > 0 })
                : openFileDialog(file, item)));
        }
        return bubble;
    }

    function buildLinksMessage(msg) {
        if (destroyed) return null;
        const actions = Array.isArray(msg.actions) ? msg.actions : [];
        const rows = actions.map((item) => {
            const url = safeHttpUrl(item?.url);
            const label = String(item?.label || '').trim().slice(0, 120);
            return url && label
                ? `<a class="chat-link-button" href="${escapeHtmlAttr(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}<span aria-hidden="true">↗</span></a>`
                : '';
        }).filter(Boolean).slice(0, MAX_LINK_ACTIONS);
        if (!rows.length) return null;
        const title = String(msg.title || '').trim();
        return bubbleFrame(msg, `<div class="message chat-links-message">
            ${title ? `<div class="chat-links-title">${escapeHtml(title)}</div>` : ''}
            <div class="chat-links-list">${rows.join('')}</div>
        </div>`);
    }

    function isFeedTailWrapper(wrapper) {
        // Timeline contract (ARCHITECTURE: messages ordered by raw numeric
        // timestamps): media may join an existing gallery only while its
        // wrapper is still the LAST message node of its feed — only typing
        // indicators may trail it. Once any other message lands below the
        // wrapper, appending there would teleport newer media above older
        // messages, so the same key starts a fresh group instead. Works for
        // the mounted live feed and for the rebuild-replay holding fragment
        // (chat_render_batch parents collected nodes in arrival order).
        const parent = wrapper?.parentNode;
        if (!parent) return false;
        const siblings = Array.from(parent.children || []);
        const index = siblings.indexOf(wrapper);
        if (index < 0) return false;
        for (let i = index + 1; i < siblings.length; i += 1) {
            if (!siblings[i]?.classList?.contains?.('typing-bubble')) return false;
        }
        return true;
    }

    function buildGallery(kind, msg, bubble) {
        if (!bubble) return false;
        const role = msg.role === 'user' ? 'user' : 'assistant';
        const taskId = String(msg.task_id || '');
        const map = kind === 'photos' ? photoGroups : fileGroups;
        const selector = kind === 'photos' ? '.chat-gallery-item' : '.chat-file-item';
        const gridSelector = kind === 'photos' ? '.chat-gallery-grid' : '.chat-file-grid';
        const ownItem = bubble.querySelector(selector);
        if (ownItem && bubble.dataset.messageKey) ownItem.dataset.messageKey = bubble.dataset.messageKey;
        if (ownItem && msg.history_id) {
            ownItem.dataset.historyId = msg.history_id;
            stampNodeTimestamp(ownItem, msg.ts);
            delete bubble.dataset.historyId;
        }
        if (!taskId) {
            insertMessageNode(bubble);
            return true;
        }
        const key = `${role}:${kind}:${taskId}`;
        const existing = map.get(key);
        const sourceAfter = !msg.history_id || Number(existing?.dataset.ts) < Date.parse(msg.ts || '')
            || Number(existing?.dataset.ts) === Date.parse(msg.ts || '')
                && Number(existing?.dataset.historyOffset) <= Number(msg.history_position?.offset);
        if (existing && isFeedTailWrapper(existing) && sourceAfter) {
            const item = bubble.querySelector(selector);
            const grid = existing.querySelector(gridSelector);
            if (!item || !grid) return false;
            grid.appendChild(item);
            existing.classList.add('is-multiple');
            if (!existing.querySelector('.chat-group-title')) {
                const title = document.createElement('div');
                title.className = 'chat-group-title';
                title.textContent = kind === 'photos' ? tr('media.multiple_images', 'Multiple images') : tr('media.multiple_files', 'Multiple files');
                grid.before(title);
            }
            stampNodeTimestamp(existing, msg.ts || '', { anchor: true });
            return true;
        }
        // First bubble for this key, or adjacency broken by an intervening
        // message: start a new group under the same key. The map keeps the
        // LATEST wrapper so the next contiguous item groups with this one.
        bubble.dataset.mediaGroup = key;
        map.set(key, bubble);
        groupingWrappers.add(bubble);
        insertMessageNode(bubble);
        return true;
    }

    // The owner's attachments sit above the caption inside the bubble addMessage is
    // building (DESIGN "Chat attachments"), from these same atoms and disposers; an
    // empty caption leaves no empty text row (the node stays, hidden, as the anchor
    // other bubble decorations are placed against).
    function mountAttachments(bubble, views, caption) {
        const block = destroyed ? null : buildAttachmentBlock({
            listen, own, photoActionsHtml, wirePhotoActions, playerHtml, wirePlayer, openFileDialog,
            humanSize, fileExtension, release, onDomWrite,
        }, views);
        const message = block && bubble?.querySelector('.message');
        if (!message) return false;
        message.before(block);
        message.hidden = !caption;
        bubble.classList.add('has-attachments');
        // Several photos or a player take the media width; one photo or a card shrinks to fit.
        bubble.classList.toggle('has-wide-media', Boolean(block.querySelector('.is-multiple')
            || block.querySelector('.chat-attachment-player')));
        return true;
    }

    // D12: the standard "two squares" copy icon, always visible on the bubble.
    // Explicit closing tags (no self-closing) keep lightweight DOM stubs happy.
    const COPY_ICON_SVG = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none"'
        + ' stroke="currentColor" stroke-width="1.5" stroke-linecap="round"'
        + ' stroke-linejoin="round" aria-hidden="true">'
        + '<rect x="5.5" y="5.5" width="8" height="8" rx="1.5"></rect>'
        + '<path d="M10.5 2.5h-6a2 2 0 0 0-2 2v6"></path></svg>';

    function attachCopyControl(bubble, rawText) {
        if (!bubble || !String(rawText || '')) return null;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'chat-message-copy';
        button.innerHTML = COPY_ICON_SVG;
        button.title = tr('code.copy', 'Copy');
        button.setAttribute('aria-label', tr('media.copy_message', 'Copy message'));
        const writeFallback = () => {
            const area = document.createElement('textarea');
            area.className = 'chat-copy-fallback';
            area.value = String(rawText);
            area.setAttribute('readonly', '');
            document.body.appendChild(area);
            try {
                area.select();
                const copied = typeof document.execCommand === 'function'
                    && document.execCommand('copy') === true;
                if (!copied) throw new Error('Clipboard access is unavailable');
            } finally {
                area.remove();
            }
        };
        listen(button, 'click', async () => {
            let ok = true;
            try {
                if (navigator.clipboard?.writeText) {
                    try { await navigator.clipboard.writeText(String(rawText)); }
                    catch { writeFallback(); }
                } else writeFallback();
            } catch {
                ok = false;
            }
            button.textContent = ok ? '✓' : '✗';
            button.title = ok ? tr('media.message_copied', 'Message copied') : tr('media.copy_failed', 'Copy failed');
            button.setAttribute('aria-label', ok ? tr('media.message_copied', 'Message copied') : tr('media.copy_failed', 'Copy failed'));
            later(() => {
                button.innerHTML = COPY_ICON_SVG;
                button.title = tr('code.copy', 'Copy');
                button.setAttribute('aria-label', tr('media.copy_message', 'Copy message'));
            }, 1500, button);
        });
        // The bubble class reserves a timestamp gutter under the icon (style.css).
        bubble.classList.add('has-copy');
        bubble.appendChild(button);
        return button;
    }

    function reset() {
        for (const dispose of disposers) {
            try { dispose(); } catch {}
        }
        disposers.clear();
        resourceOwners.clear();
        for (const timer of timers) clearTimeout(timer);
        timers.clear();
        timerOwners.clear();
        for (const player of playerRegistry) {
            try {
                player.pause();
                player.removeAttribute('src');
                player.load?.();
            } catch {}
        }
        playerRegistry.clear();
        for (const url of objectUrls) {
            try { URL.revokeObjectURL(url); } catch {}
        }
        objectUrls.clear();
        for (const wrapper of groupingWrappers) {
            try { wrapper.remove(); } catch {}
        }
        groupingWrappers.clear();
        photoGroups.clear();
        fileGroups.clear();
        closeTransient();
        if (fileDialog) {
            try { fileDialog.remove(); } catch {}
            fileDialog = null;
        }
    }

    // The chat left the screen: the reader or file dialog a card opened closes, nothing else.
    function closeTransient() {
        reader.close({ restoreFocus: false });
        closeFileDialog();
    }

    // Release only the evicted page's subtree; other media and copy controls
    // keep their listeners, playback, focus and outstanding user actions.
    function release(root) {
        reader.release(root);
        const owns = (node) => node === root || root?.contains?.(node);
        if (dialogOwner && owns(dialogOwner)) closeFileDialog();  // no actions on a released message's file
        for (const [dispose, owner] of resourceOwners) if (owns(owner)) {
            try { dispose(); } catch {}
            resourceOwners.delete(dispose);
            disposers.delete(dispose);
        }
        for (const [timer, owner] of timerOwners) if (owns(owner)) {
            clearTimeout(timer); timers.delete(timer); timerOwners.delete(timer);
        }
        for (const player of playerRegistry) if (owns(player)) {
            try { player.pause(); player.removeAttribute('src'); player.load?.(); } catch {}
            playerRegistry.delete(player);
        }
        for (const wrapper of groupingWrappers) if (owns(wrapper)) {
            groupingWrappers.delete(wrapper);
            for (const map of [photoGroups, fileGroups]) {
                for (const [key, value] of map) if (value === wrapper) map.delete(key);
            }
        }
    }

    function destroy() {
        if (destroyed) return;
        reset();
        reader.destroy();
        destroyed = true;
    }

    function wireDeliveries({
        onWs,
        isMyThread,
        hideTypingIndicatorOnly,
        syncChatStatus,
        incrementUnreadIfNeeded,
        seenMessageKeys,
        rememberMessageKey,
        chatMediaMessageKey,
        documentMessageKey,
        buildQuizCard,
        applyQuizStateFrame,
        messagesRoot,
        deliverContentMutation = (mutate) => mutate(),
    }) {
        const keysFor = (msg, legacy) => ({ legacy, exact: msg.history_id ? `history:${msg.history_id}` : legacy });
        function alreadyRendered(msg, keys) {
            if (seenMessageKeys.has(keys.exact)) return true;
            const existing = Array.from(messagesRoot().querySelectorAll('[data-message-key]'))
                .sort((a, b) => Number(b.matches('.chat-gallery-item, .chat-file-item'))
                    - Number(a.matches('.chat-gallery-item, .chat-file-item')))
                .find(node => msg.history_id && node.dataset.historyId === msg.history_id
                    || node.dataset.messageKey === keys.legacy && (!msg.history_id || !node.dataset.historyId));
            if (!existing) return false;
            if (msg.history_id) stampHistoryNode(existing, msg.history_id, msg.history_position);
            rememberMessageKey(keys.exact);
            return true;
        }
        function rememberBubble(bubble, keys) {
            bubble.dataset.messageKey = keys.legacy;
            rememberMessageKey(keys.exact);
            rememberMessageKey(keys.legacy);
        }
        function appendMediaBubble(msg) {
            const keys = keysFor(msg, chatMediaMessageKey(msg));
            if (alreadyRendered(msg, keys)) return false;
            const bubble = buildMediaBubble(msg);
            if (!bubble) return false;
            rememberBubble(bubble, keys);
            if ((msg.msg_type || msg.type) === 'photo') return buildGallery('photos', msg, bubble);
            return insertMessageNode(bubble) !== false;
        }
        function appendDocumentBubble(msg) {
            const keys = keysFor(msg, documentMessageKey(msg));
            if (alreadyRendered(msg, keys)) return false;
            const bubble = buildDocumentBubble(msg);
            if (!bubble) return false;
            rememberBubble(bubble, keys);
            return buildGallery('files', msg, bubble);
        }
        function appendLinksMessage(msg) {
            const actions = Array.isArray(msg.actions) ? msg.actions.slice(0, MAX_LINK_ACTIONS) : [];
            const key = `links:${msg.task_id || ''}:${msg.ts || ''}:${JSON.stringify(actions)}:${msg.title || ''}`;
            const keys = keysFor(msg, key);
            if (alreadyRendered(msg, keys)) return false;
            const bubble = buildLinksMessage(msg);
            if (!bubble) return false;
            rememberBubble(bubble, keys);
            return insertMessageNode(bubble) !== false;
        }
        function appendQuizMessage(msg) {
            const quizId = String((msg.quiz && msg.quiz.quiz_id) || msg.quiz_id || '');
            const keys = keysFor(msg, `quiz:${msg.task_id || ''}:${quizId}`);
            // The decision controller updates the existing keyed card, keeping
            // drafts/focus when targeted detail and ordinary history overlap. A
            // card first shown live gains the row history names, as media does.
            const bubble = buildQuizCard ? buildQuizCard(msg) : null;
            if (!bubble) { alreadyRendered(msg, keys); return false; }
            rememberBubble(bubble, keys);
            return insertMessageNode(bubble) !== false;
        }
        // Media frames carry no activity identity: hide the dots row but keep
        // the authoritative active set intact; sync derives the header from it.
        const deliver = (append) => (msg) => {
            if (!isMyThread(msg)) return;
            hideTypingIndicatorOnly();
            syncChatStatus();
            if (deliverContentMutation(() => append(msg))) incrementUnreadIfNeeded(msg);
        };
        for (const type of ['photo', 'video']) onWs(type, deliver(appendMediaBubble));
        onWs('document', deliver(appendDocumentBubble));
        onWs('links', deliver(appendLinksMessage));
        onWs('quiz', deliver(appendQuizMessage));
        if (applyQuizStateFrame && messagesRoot) {
            // Lifecycle updates address an EXISTING card by quiz_id — no
            // thread routing, no unread bump, never a new bubble.
            onWs('quiz_state', (msg) => deliverContentMutation(
                () => applyQuizStateFrame(messagesRoot(), msg),
            ));
        }
        return { appendMediaBubble, appendDocumentBubble, appendLinksMessage, appendQuizMessage };
    }

    return {
        buildMediaBubble,
        buildDocumentBubble,
        buildLinksMessage,
        buildGallery,
        bubbleFrameNode,
        attachCopyControl,
        mountAttachments,
        wireDeliveries,
        reset,
        release,
        closeTransient,
        destroy,
    };
}
