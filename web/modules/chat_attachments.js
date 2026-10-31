import { isPanicFrame } from './panic.js';
// Owner attachments inside the owner's own chat bubble (docs/DESIGN.md "Chat
// attachments"): one message, attachments above the caption, whole images
// (one at its own ratio, several in the shared contain grid), the existing
// players for video/audio and the existing file cards with always-visible
// actions. Everything renders from chat_media's atoms through the instance's
// own disposers; this module adds composition, never a second renderer.
//
// The views come from the server (chat_uploads.attachment_view): the upload
// response for the sender's own bubble, the echo for other tabs, history for
// replay — the same dict everywhere. Nothing here builds a file URL.
import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { tr } from './i18n.js';
import { apiClient, apiFetch } from './api_client.js';
import { documentReaderKind } from './document_reader.js';

// At most this many `[Attached file: …]` lines ride the message text.
export const ATTACHMENT_PREVIEW_COUNT = 25;
const KINDS = new Set(['image', 'video', 'audio', 'file']);
const UPLOAD_URL_RE = /^\/api\/files\/download\?upload=[0-9a-f]{32}_[^&#\s]+$/;
const FILE_GLYPH = '<span class="chat-file-glyph" aria-hidden="true">▤</span>';

/** The generated text tail naming attachments, exactly as the composer sends it. */
export function attachmentTail(names) {
    const list = Array.from(names || [], (name) => String(name || ''));
    return list.slice(0, ATTACHMENT_PREVIEW_COUNT).map((name) => `[Attached file: ${name}]`)
        .concat(list.length > ATTACHMENT_PREVIEW_COUNT ? [`[${list.length - ATTACHMENT_PREVIEW_COUNT} more attached files]`] : [])
        .join('\n');
}

/**
 * The words a composer frame carries: the owner's text, then the generated tail
 * naming the files for the model. Only the host's exact `/restart` stays exact, as
 * the host matches it (ws.py, server_control): it is still the Restart command,
 * its files ride the same accepted row. Any other words — `/restart …` too — are a
 * message and get the tail.
 */
export function composerText(text, names) {
    const words = String(text ?? '');
    if (words.trim().toLowerCase() === '/restart') return words;
    return words + (words ? '\n\n' : '') + attachmentTail(names);
}

/**
 * The words to show under the attachments. Display only: the canonical text and
 * what the model reads stay as stored. A tail is hidden only on a row the web
 * composer wrote (`composed`: its `source` is `web`, the one producer of that
 * tail) and only when it is EXACTLY the one these attachments generate (names in
 * order); the whole text only when the row marks it as the host's
 * (`text_placeholder`: the sender sent no words, e.g. "(image attached)"). The
 * same words typed by the owner, or a Telegram/skill caption no composer wrote,
 * are the owner's and show as is.
 */
export function attachmentCaption(text, views, { placeholder = false, composed = false } = {}) {
    const raw = String(text ?? '');
    if (!views?.length) return raw;
    if (placeholder === true) return '';
    if (composed !== true) return raw;
    const tail = attachmentTail(views.map((view) => view.name));
    if (raw === tail) return '';
    return raw.endsWith(`\n\n${tail}`) ? raw.slice(0, raw.length - tail.length - 2) : raw;
}

/** Server views kept to their closed shape; an unexpected URL makes a view unavailable. */
export function attachmentViews(value) {
    if (!Array.isArray(value)) return [];
    // Every attachment renders: ATTACHMENT_PREVIEW_COUNT bounds only the text tail.
    return value.filter((view) => view && typeof view === 'object').map((view) => {
        const url = String(view.url || '');
        const available = view.available === true && UPLOAD_URL_RE.test(url);
        const size = Number.isFinite(Number(view.size)) && view.size !== null && view.size !== '' ? Number(view.size) : null;
        return {
            // Bounded per code point, as the server labels it (chat_uploads._label): a server
            // name passes whole, so the composer's tail still matches, and no pair is split.
            name: Array.from(String(view.name || 'attachment').replace(/[\r\n]+/g, ' ')).slice(0, 200).join(''),
            kind: available && KINDS.has(view.kind) ? view.kind : 'file',
            mime: String(view.mime || ''),
            size,
            available,
            url: available ? url : '',
        };
    });
}

/**
 * A file the host could not attach to the task, named directly under the owner's own words from the
 * routing receipt's manifest (DESIGN "Chat attachments"). Its own node, not the receipt line, so the
 * Main fold of a landed receipt never hides it; a clean delivery paints nothing. Upload availability
 * is a separate fact: the file's own tile keeps Open/Download. Returns whether it changed.
 */
export function paintAttachmentOutcome(bubble, annotation) {
    if (!bubble) return false;
    const rows = Array.isArray(annotation?.attachment_manifest) ? annotation.attachment_manifest : [];
    const names = rows.filter((row) => row && typeof row === 'object' && row.status === 'rejected')
        .map((row) => String(row.label || `#${Number(row.ordinal || 0) + 1}`));
    const text = names.length ? `${tr('chat.attachment_not_attached', 'Not attached to the task')}: ${names.join(', ')}` : '';
    const child = (name) => [...bubble.children].find((node) => node.classList?.contains(name));
    let note = child('msg-attachment-outcome');
    if (!text) {
        note?.remove();
        return Boolean(note);
    }
    if (note?.textContent === text) return false;
    if (!note) {
        note = document.createElement('div');
        note.className = 'msg-attachment-outcome';
        // One place whatever lands later (saved mark, receipt line, time): right under the words.
        const next = child('message')?.nextElementSibling;
        if (next) next.before(note);
        else bubble.append(note);
    }
    note.textContent = text;
    return true;
}

const metaText = (view, atoms, note = '') => [atoms.fileExtension(view.name), atoms.humanSize(view.size), note]
    .filter(Boolean).join(' · ');

// The owner's own Markdown or text reads in the reader from its upload; any other file, an image
// the host did not prove (served only as a download) included, keeps the file dialog.
function readable(view) {
    const kind = view.available ? documentReaderKind(view.name, view.mime) : '';
    return kind === 'markdown' || kind === 'text' ? kind : '';
}

function cardHtml(view, atoms, note = view.available ? '' : tr('media.attachment_unavailable', 'Unavailable')) {
    const meta = metaText(view, atoms, note);
    const kind = note ? '' : readable(view);
    return `<div class="chat-file-item${view.available ? '' : ' is-unavailable'}">
        <button type="button" class="btn btn-default chat-file-card" ${view.available ? 'aria-haspopup="dialog"' : 'disabled aria-disabled="true"'}
            aria-label="${escapeHtmlAttr(`${view.name}${meta ? ` — ${meta}` : ''}`)}">
            ${FILE_GLYPH}
            <span class="chat-file-copy">
                <span class="chat-file-name" title="${escapeHtmlAttr(view.name)}">${escapeHtml(view.name)}</span>
                <span class="chat-file-meta">${escapeHtml(meta)}</span>
            </span>
            ${kind ? atoms.readLabelHtml(kind) : '<span class="chat-file-more" aria-hidden="true">•••</span>'}
        </button>
    </div>`;
}

function sourceOf(view) {
    // The upload route is both the canonical address and the one every desktop
    // launcher's file bridge already allows: no compat twin is needed.
    return { base64: '', durable: view.url, bridge: view.url, src: view.url };
}

/**
 * The attachment block for one owner bubble: images, then players, then files.
 * `atoms` are chat_media's own builders and lifecycle (listen/players/menus/dialog),
 * so the instance's release/reset/destroy dispose everything mounted here.
 */
export function buildAttachmentBlock(atoms, rawViews) {
    const views = attachmentViews(rawViews);
    if (!views.length) return null;
    const images = views.filter((view) => view.available && view.kind === 'image');
    const players = views.filter((view) => view.available && (view.kind === 'video' || view.kind === 'audio'));
    const files = views.filter((view) => !view.available || view.kind === 'file');
    const block = document.createElement('div');
    block.className = 'chat-attachments';
    block.innerHTML = [
        images.length ? `<div class="chat-gallery-grid${images.length > 1 ? ' is-multiple' : ''}">${images.map((view) => `
            <figure class="chat-gallery-item">
                ${atoms.photoHtml(view.url, view.name, ' loading="lazy" decoding="async"')}
            </figure>`).join('')}</div>` : '',
        players.map((view) => `<div class="chat-attachment-player">${atoms.playerHtml({
            audio: view.kind === 'audio', src: view.url, title: view.name })}</div>`).join(''),
        files.length ? `<div class="chat-file-grid">${files.map((view) => cardHtml(view, atoms)).join('')}</div>` : '',
    ].join('');
    const figures = block.querySelectorAll('.chat-gallery-item');
    images.forEach((view, index) => {
        const item = figures[index];
        if (!item) return;
        atoms.wirePhotoActions(item, view.url, sourceOf(view), view.name, view.mime, { meta: metaText(view, atoms) });
        atoms.listen(item.querySelector('.chat-photo'), 'error', () => degrade(atoms, block, item, view), { once: true }, item);
    });
    const playerRoots = block.querySelectorAll('.chat-attachment-player');
    players.forEach((view, index) => {
        const root = playerRoots[index];
        const media = root?.querySelector(view.kind === 'audio' ? 'audio' : 'video');
        if (!root || !media) return;
        atoms.wirePlayer(root, media, { audio: view.kind === 'audio', source: sourceOf(view), filename: view.name,
            mime: view.mime, fullscreenTarget: root.querySelector('.chat-media-player') || root });
        atoms.listen(media, 'error', () => degrade(atoms, block, root, view), { once: true }, root);
    });
    const cards = block.querySelectorAll('.chat-file-item');
    files.forEach((view, index) => wireCard(atoms, cards[index], view));
    return block;
}

function wireCard(atoms, item, view) {
    const card = item?.querySelector('.chat-file-card');
    if (!card || !view.available) return;
    const kind = card.querySelector('.chat-file-more.is-read') ? readable(view) : '';
    const file = { source: sourceOf(view), filename: view.name, mime: view.mime || 'application/octet-stream', kind,
        meta: metaText(view, atoms), reader: kind ? { url: view.url } : null, size: view.size, canOpen: true };
    atoms.listen(card, 'click', (event) => (kind ? atoms.openReader(file, item, card, event)
        : atoms.openFileDialog(file, item)), undefined, item);
}

// A preview the engine cannot show (HEIC in Chromium, an unsupported codec) stays an
// honest card with Open/Download; a file that is gone becomes the inert card. The HEAD
// answer is late, so its end is owned by `node` like the node's listeners: a message the feed
// released (a removed subtree keeps its parentNode) or the instance's reset/destroy ends it.
async function degrade(atoms, block, node, view) {
    let live = true;
    atoms.own(() => { live = false; }, node);
    let gone = false;
    try { gone = (await apiFetch(view.url, { method: 'HEAD' })).status === 404; } catch { /* unknown: keep actions */ }
    if (!live) return;
    const shown = { ...view, available: !gone };
    atoms.onDomWrite(() => {
        if (!live || !node.parentNode) return false;
        let grid = block.querySelector('.chat-file-grid');
        if (!grid) {
            grid = document.createElement('div');
            grid.className = 'chat-file-grid';
            block.appendChild(grid);
        }
        const holder = document.createElement('div');
        holder.innerHTML = cardHtml(shown, atoms, gone ? tr('media.attachment_unavailable', 'Unavailable')
            : tr('media.preview_unavailable', 'Preview unavailable'));
        const item = holder.querySelector('.chat-file-item');
        const gallery = node.closest?.('.chat-gallery-grid');
        atoms.release(node);
        node.remove();
        grid.appendChild(item);
        wireCard(atoms, item, shown);
        if (gallery) {
            const left = gallery.querySelectorAll('.chat-gallery-item').length;
            gallery.classList.toggle('is-multiple', left > 1);
            if (!left) gallery.remove();
        }
        return true;
    });
}

/**
 * The composer's staged attachments: badges with a thumbnail for each image,
 * upload on Send, and the one owner of their object URLs — created once when a
 * file is staged, revoked when it is removed, sent or the composer is destroyed,
 * and kept across a failed send so the owner can retry.
 */
export function createComposerAttachments({ preview, attachBtn, fileInput, input, onLayout, showToast }) {
    let pending = [];
    let uploading = false;

    const thumbFor = (file) => (String(file?.type || '').startsWith('image/') && typeof URL?.createObjectURL === 'function'
        ? URL.createObjectURL(file) : '');
    const revoke = (item) => {
        if (item.thumb) try { URL.revokeObjectURL(item.thumb); } catch { /* already released */ }
        item.thumb = '';
    };

    function render() {
        preview.classList.toggle('visible', pending.length > 0);
        preview.innerHTML = pending.map((item) => `
            <span class="attach-badge" data-attachment-id="${escapeHtmlAttr(item.id)}">
                ${item.thumb ? `<img class="attach-thumb" src="${escapeHtmlAttr(item.thumb)}" alt="">`
                    : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>'}
                <span class="attach-name" title="${escapeHtmlAttr(item.display_name)}">${escapeHtml(item.display_name)}</span>
                <button class="attach-remove" type="button" title="Remove" aria-label="Remove attachment ${escapeHtmlAttr(item.display_name)}" data-attachment-remove="${escapeHtmlAttr(item.id)}" ${uploading ? 'disabled aria-disabled="true"' : ''}>×</button>
            </span>
        `).join('');
        preview.querySelectorAll('[data-attachment-remove]').forEach((button) => {
            button.addEventListener('click', () => {
                if (uploading) return;
                const removeId = button.dataset.attachmentRemove || '';
                pending = pending.filter((item) => item.id !== removeId || (revoke(item), false));
                render();
            });
        });
        onLayout();
    }

    function stage(files) {
        const incoming = Array.from(files || []).filter(Boolean);
        if (!incoming.length) return;
        if (uploading) {
            showToast('Wait for the current upload to finish before changing attachments.', 'error');
            return;
        }
        pending = pending.concat(incoming.map((file) => ({
            id: (globalThis.crypto && typeof crypto.randomUUID === 'function')
                ? crypto.randomUUID()
                : `attachment-${Date.now()}-${Math.random().toString(16).slice(2)}`,
            file,
            display_name: file.name || 'upload',
            thumb: thumbFor(file),
        })));
        render();
    }

    function setUploading(flag) {
        uploading = flag;
        attachBtn.disabled = flag;
        attachBtn.classList.toggle('uploading', flag);
        fileInput.disabled = flag;
        // Read-only, not disabled: a phone keeps its keyboard and an IME its composition.
        input.readOnly = flag;
        render();
    }

    /**
     * Upload every staged file in order; `connected()` is checked before each one.
     * Resolves with `{filename, display_name, mime, view}` per file. On failure the
     * uploads already made are handed back on the error (`error.uploaded`) for
     * cleanup, and the staged files (with their thumbnails) stay for a retry.
     */
    async function upload(connected) {
        const uploaded = [];
        setUploading(true);
        try {
            for (const item of [...pending]) {
                if (!connected()) throw new Error('Connection closed during upload. Reconnect and try again.');
                const data = await apiClient.uploadChatAttachment(item.file);
                uploaded.push({
                    filename: data.filename || '',
                    display_name: data.display_name || item.display_name,
                    mime: data.mime || item.file?.type || '',
                    view: data.view || null,
                });
            }
            if (!connected()) throw new Error('Connection closed after upload. Reconnect and try again.');
            return uploaded;
        } catch (error) {
            error.uploaded = uploaded;
            throw error;
        } finally {
            setUploading(false);
        }
    }

    function clear() {
        pending.forEach(revoke);
        pending = [];
        render();
    }

    return {
        stage,
        upload,
        clear,
        destroy() { pending.forEach(revoke); pending = []; },
        get count() { return pending.length; },
        get busy() { return uploading; },
    };
}

// What a reload keeps of a sent frame: its words, identity, routing and the refs of uploads the
// host already stores — never bytes, never the send-time client_surface (an honest gap on resend).
const KEPT_FRAME_KEYS = ['type', 'content', 'client_message_id', 'sender_session_id', 'chat_id', 'project_id', 'reply_to'];

function keptFrame(frame) {
    if (isPanicFrame(frame)) return null;
    if (frame?.type !== 'chat' || typeof frame.content !== 'string' || !frame.client_message_id) return null;
    return { ...Object.fromEntries(KEPT_FRAME_KEYS.filter((key) => key in frame).map((key) => [key, frame[key]])),
        attachments: Array.from(Array.isArray(frame.attachments) ? frame.attachments : [], (item) => ({
            filename: String(item?.filename || ''), display_name: String(item?.display_name || ''), mime: String(item?.mime || '') })) };
}

/**
 * Where an owner bubble's delivery note stands (docs/DESIGN.md "Chat attachments"). The plain saved fact (its
 * `[data-ingress-saved]` note, dispatched or still pending, with nothing to act on) is a quiet tick before the message's
 * time: saved, never read or started; its words stay its text for assistive technology and a pointer's title. A note
 * with words to read or actions (the delivery doubt, Send again, Send as new) is a row above the time.
 */
export function placeIngressNote(bubble, note) {
    if (!bubble || !note) return;
    const time = bubble.querySelector('.msg-time');
    // Only the plain saved states are quiet; every other state (a delivery doubt, a legacy `undispatched`) keeps its words.
    const quiet = Boolean(time) && ['', 'pending'].includes(note.dataset.ingressSaved)
        && !note.querySelector('[data-unconfirmed-action]');
    if (quiet) note.title = note.textContent; else note.removeAttribute('title');
    if (quiet && note.parentNode !== time) time.insertBefore(note, time.lastChild);
    else if (!quiet && (note.parentNode !== bubble || note.nextElementSibling !== time)) bubble.insertBefore(note, time);
}

/**
 * Owner messages the host has not yet confirmed saving: every send, words or attachments (a bare slash
 * command is a command, not a message). A local `send()` is not acceptance; the echo or history row
 * marked `ingress_accepted` is (`settle`, beside `markIngressSaved`). Until then the sent frame (its
 * `reply_to` included) is kept under its own client_message_id, with no count bound: each one is an
 * unsaved message of the owner's. A frame still in the socket's offline queue (`queued`) is that
 * queue's until it is written (`flushed`); a close offers nothing for it meanwhile. This tab's `storage` (sessionStorage) keeps its words, id, routing and
 * upload refs (`keptFrame`, with the bubble's views and time) across a reload and a
 * room's teardown; a storage refusal is shown and the in-memory copy still retries, and
 * a kept copy the next instance cannot read is shown too, never silently dropped. A
 * bubble the feed released or rebuilt is no proof of saving, so its frame stays. If the
 * socket closes, or the host answers with its failure notice, first — or the first
 * history read after a reload does not show the message saved (`reconcile`) — every
 * such bubble says so and offers "Send again": the SAME frame and id, which the host
 * rejoins if it saved the message after all (never a second message) or accepts now,
 * and "Discard", which forgets only this tab's copy. A saved row marked
 * `ingress_pending` is the running host's, between its append and its dispatch
 * (or echoed before a deferred dispatch): the frame waits for the next fact. A saved row
 * with neither dispatch fact came from a host process that has since ended (a restart came
 * between): whether it reached the agent is unknown and Send again would only rejoin
 * it, so its bubble says "Saved; delivery not confirmed" and the frame is dropped; the one action it keeps is "Send as a
 * new message", which only prepares the composer with the owner's words (`compose`) and sends nothing. Bubble
 * labels (markIngressSaved): `Input saved` on `ingress_dispatched` and `ingress_pending`.
 * `count` feeds the chat
 * instance's `hasPendingWork`, which keeps a Project room holding one hidden instead of
 * destroying it. Nothing here resends by itself or deletes an upload: those stay on the host.
 */
export function createUnconfirmedSends({ send, root, onDomWrite, showToast, compose = null, storage = null, storageKey = '' }) {
    const frames = new Map();  // id -> { frame, views, ts, restored, queued }
    const bubbleOf = (id) => [...(root()?.querySelectorAll('.chat-bubble.user[data-client-message-id]') || [])]
        .find((node) => node.dataset.clientMessageId === id) || null;
    let unreadable = false;  // a kept copy this tab cannot read restores nothing, and says so
    try {
        const kept = storage && storageKey ? storage.getItem(storageKey) : null;
        for (const entry of kept === null ? [] : JSON.parse(kept)) {
            const frame = keptFrame(entry?.frame);
            if (frame) frames.set(String(frame.client_message_id), { frame, views: attachmentViews(entry.views),
                ts: String(entry.ts || ''), restored: true });
            else unreadable = true;
        }
    } catch { unreadable = true; }
    // The tab's own storage, not an action of the owner's that failed: said, but not felt (toast.js, haptics.js).
    if (unreadable) showToast(tr('chat.send_unconfirmed_unread',
        'This tab could not read an unsaved message it kept for a reload, so it cannot be offered again.'), 'error',
        { haptic: false });

    function persist() {
        if (!storage || !storageKey) return;
        try {
            if (!frames.size) storage.removeItem(storageKey);
            else storage.setItem(storageKey, JSON.stringify([...frames.values()]
                .map(({ frame, views, ts }) => ({ frame: keptFrame(frame), views, ts }))));
        } catch {
            showToast(tr('chat.send_unconfirmed_unkept',
                'This tab could not keep the unsaved message for a reload. Send again still works until then.'), 'error',
                { haptic: false });
        }
    }

    function forget(id) {
        if (frames.delete(id)) persist();
    }

    function retry(id, note) {
        const entry = frames.get(id);
        if (!entry) return;
        if (send(entry.frame, { queue: false })?.status !== 'sent') {
            showToast(tr('chat.send_again_offline', 'Still offline. Reconnect and send again.'), 'error');
            return;
        }
        // The doubt waits for the next fact; a pending saved row keeps its note, without actions.
        onDomWrite(() => (note.dataset.ingressUnconfirmed !== undefined ? note.remove() : clearActions(note, bubbleOf(id)), true));
    }

    // Without its actions a pending saved note is the quiet tick again (placeIngressNote).
    function clearActions(note, bubble = null) {
        for (const button of note.querySelectorAll('[data-unconfirmed-action]')) button.remove();
        placeIngressNote(bubble, note);
    }

    function action(label, kind, onClick) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-default btn-sm';
        button.dataset.unconfirmedAction = kind;
        button.textContent = label;
        button.addEventListener('click', onClick);
        return button;
    }

    // The doubt on its bubble, or a pending saved note after a disconnect, gains the two actions.
    function mark(id, bubble) {
        const saved = bubble.querySelector('[data-ingress-saved]');
        let note = saved || bubble.querySelector('[data-ingress-unconfirmed]');
        if (!frames.has(id) || saved?.dataset.ingressSaved === '' || note?.querySelector('[data-unconfirmed-action]')) return false;
        if (!note) {
            note = document.createElement('div');
            note.className = 'msg-pending';
            note.dataset.ingressUnconfirmed = '';
            note.textContent = `${tr('chat.send_unconfirmed', 'Not confirmed as saved.')} `;
            bubble.insertBefore(note, bubble.querySelector('.msg-time'));
        }
        note.append(action(tr('chat.send_again', 'Send again'), 'retry', () => retry(id, note)),
            action(tr('chat.send_discard', 'Discard'), 'discard', () => { forget(id); onDomWrite(() => (clearActions(note, bubble), true)); }));
        placeIngressNote(bubble, note);  // a pending saved note with actions is a row again
        return true;
    }

    // Saved by a host process that has since ended: what it did with the row is unknown, and nothing replays it.
    // The one honest next step is a NEW message: its action prepares the composer with the owner's own words and sends
    // nothing; the saved row stays as it is, and the owner reads, edits and sends (or does not) from the composer.
    function doubtDelivery(bubble) {
        let note = bubble.querySelector('[data-ingress-saved]');
        if (!note) {
            note = Object.assign(document.createElement('div'), { className: 'msg-pending' });
            bubble.insertBefore(note, bubble.querySelector('.msg-time'));
        }
        bubble.querySelector('[data-ingress-unconfirmed]')?.remove();
        clearActions(note);
        note.dataset.ingressSaved = 'unconfirmed';
        note.textContent = tr('chat.saved_delivery_unconfirmed', 'Saved; delivery not confirmed.');
        if (compose) {
            const words = () => bubble.querySelector('.message')?.textContent || '';
            // Short visible words so the sentence and its action share one row on a phone; the name stays whole.
            const asNew = action(tr('chat.send_as_new_short', 'Send as new'), 'send_as_new', () => compose(words()));
            asNew.setAttribute('aria-label', tr('chat.send_as_new', 'Send as a new message'));
            note.append(asNew);
        }
        placeIngressNote(bubble, note);
        return true;
    }

    // Every still-held message (or only `ids`) whose bubble is in the feed: a plain saved mark ends it; anything else offers the actions.
    function unsettle(ids = [...frames.keys()]) {
        for (const id of ids) {
            const bubble = bubbleOf(id);
            if (!bubble) continue;  // not in the feed now: still unsaved, the frame stays
            if (bubble.querySelector('[data-ingress-saved]')?.dataset.ingressSaved === '') forget(id);
            else if (!frames.get(id).queued) onDomWrite(() => mark(id, bubble));  // the socket's own queue still holds a queued one
        }
    }

    return {
        track(frame, { views = [], ts = '', queued = false } = {}) {
            if (isPanicFrame(frame)) return;
            const id = String(frame?.client_message_id || '');
            if (!id) return;
            frames.set(id, { frame, views: attachmentViews(views), ts, queued });
            persist();
        },
        /** The socket's offline queue wrote (or dropped) a queued frame: from now on it is this tab's to offer again. */
        flushed(id) {
            const entry = frames.get(String(id || ''));
            if (entry) entry.queued = false;
        },
        /** The host's saved row for an id (its echo or history) ends a doubt once dispatch is confirmed.
         *  `ingress_pending` keeps the frame waiting for that fact. A saved row of a host process that
         *  has since ended (neither dispatch fact) ends it in a delivery doubt. */
        settle(row) {
            const id = String(row?.client_message_id || '');
            if (row?.role !== 'user' || row.ingress_accepted !== true || !frames.has(id)) return;
            const bubble = bubbleOf(id);
            if (row.ingress_dispatched === true) {
                forget(id);
            } else if (row.ingress_pending === true) {
                // The running host took it and still owes dispatch: the frame waits for its confirmation,
                // and a reload's first read that shows it so has answered for the kept copy (no doubt from `reconcile`).
                frames.get(id).restored = false;
            } else if (bubble) {  // without its bubble the frame waits, so the doubt is not lost unseen
                forget(id);
                onDomWrite(() => doubtDelivery(bubble));
            }
        },
        /** The socket closed or a frame was refused: every still-unsaved sent message in the feed says so. */
        unsettle,
        /** After a reload's first history read (or its failure): a kept message it did not settle (nor show
         *  `ingress_pending`) shows its bubble again from the kept words and views if the read did not, then its doubt.
         *  A message this page sent while that read was pending is not in doubt: the read predates it. */
        reconcile(show) {
            const kept = [...frames].filter(([, entry]) => entry.restored);
            for (const [id, entry] of kept) {
                if (!bubbleOf(id)) show(entry);
                entry.restored = false;
            }
            unsettle(kept.map(([id]) => id));
        },
        get count() { return frames.size; },
        /** Room teardown: this instance forgets; the tab's kept copies stay for the room's next instance. */
        release() { frames.clear(); },
    };
}
