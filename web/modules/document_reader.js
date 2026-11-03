/**
 * Read-only reader for a Markdown or plain-text document in chat, and the
 * in-chat view of an image (DESIGN "Document reading"). A document is read only
 * from the exact copy its message carries — the live frame's inline bytes, the
 * immutable task-artifact route or the owner's own upload — never a current
 * file path, and at most DOCUMENT_READER_MAX_BYTES of it. An image shows the
 * very source its tile displays, decoded by the engine in a detached <img>,
 * never another address.
 *
 * The body is streamed to that byte bound; a read or an image load stops after
 * the read timeout, and a transport failure offers Retry. Every close path
 * (`close`, `chat_media.release`, `closeTransient`, `destroy`) aborts the
 * in-flight read or load and disposes Markdown enhancements, so a late answer
 * or image event paints nothing.
 */

import { apiFetch } from './api_client.js';
import { fmtInto } from './i18n.js';
import { bindDialogFocus } from './ui_interactions.js';
import { destroyChatMarkdown, enhanceChatMarkdown, mountChatMarkdown } from './chat_markdown.js';

// A long report is far below this. Decoding and rendering it run on the main
// thread and are not measured on slow devices; larger files show this prefix as
// an explicit partial and keep Download for the whole.
export const DOCUMENT_READER_MAX_BYTES = 1024 * 1024;
// One bound for the whole read, answer and body. A read that outlasts it stops
// and says so; only the owner's Retry starts another.
export const DOCUMENT_READER_TIMEOUT_MS = 30000;
// A refusal's body is read only this far, for its reason code.
const ERROR_BODY_MAX_BYTES = 4096;

const MARKDOWN_EXTENSIONS = new Set(['md', 'markdown', 'mdown', 'mkd']);
const TEXT_EXTENSIONS = new Set(['txt', 'text']);
// Raster formats the engines decode; SVG stays a file (Open/Download), never a document view.
const IMAGE_TYPES = { png: 'png', jpg: 'jpeg', jpeg: 'jpeg', gif: 'gif', webp: 'webp', avif: 'avif' };

/**
 * `markdown`, `text`, `image`, or '' when the reader does not open this file.
 * The extension decides when there is one, so `page.html` sent as `text/plain`
 * is not read; an extensionless file falls back to its MIME type.
 */
export function documentReaderKind(filename, mime) {
    const extension = String(filename || '').match(/\.([^.\s/\\]+)$/)?.[1]?.toLowerCase() || '';
    if (MARKDOWN_EXTENSIONS.has(extension)) return 'markdown';
    if (TEXT_EXTENSIONS.has(extension)) return 'text';
    if (Object.hasOwn(IMAGE_TYPES, extension)) return 'image';
    if (extension) return '';
    const type = String(mime || '').split(';')[0].trim().toLowerCase();
    if (Object.values(IMAGE_TYPES).some((name) => type === `image/${name}`)) return 'image';
    return type === 'text/markdown' ? 'markdown' : type === 'text/plain' ? 'text' : '';
}

export class DocumentReadError extends Error {
    constructor(code, { status = 0, reason = '' } = {}) {
        super(reason || code);
        this.name = 'DocumentReadError';
        this.code = code;
        this.status = status;
        this.reason = reason;
    }
}

function inlineBytes(base64, limit) {
    const padding = base64.endsWith('==') ? 2 : base64.endsWith('=') ? 1 : 0;
    const total = Math.max(0, Math.floor(base64.length * 3 / 4) - padding);
    // Whole base64 quanta only: the cut decodes on its own without the rest.
    const binary = atob(total > limit ? base64.slice(0, Math.ceil(limit / 3) * 4) : base64);
    const bytes = new Uint8Array(Math.min(binary.length, limit));
    for (let index = 0; index < bytes.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return { bytes, total, partial: total > bytes.length };
}

const aborted = (signal, error) => signal?.aborted || error?.name === 'AbortError';

/**
 * At most `limit` bytes of a response body, and `more` when it went on. Only
 * that prefix is held — an overflowing chunk is copied down to what fits — and
 * the stream is cancelled however the read ends. A body that cannot be
 * streamed is refused rather than read whole.
 */
async function readPrefix(response, limit, signal) {
    const reader = response.body?.getReader?.();
    if (!reader) {
        if (response.body?.cancel) response.body.cancel().catch(() => {});
        throw new DocumentReadError('unsupported');
    }
    const chunks = [];
    let kept = 0;
    let more = false;
    let finished = false;
    // An abort ends a pending chunk read even when the body does not follow the signal.
    const stop = () => { reader.cancel().catch(() => {}); };
    signal?.addEventListener('abort', stop, { once: true });
    try {
        for (;;) {
            if (signal?.aborted) throw new DOMException('The read was stopped.', 'AbortError');
            const { done, value } = await reader.read();
            if (signal?.aborted) throw new DOMException('The read was stopped.', 'AbortError');
            if (done) { finished = true; break; }
            const room = limit - kept;
            if (value.length > room) {
                if (room > 0) chunks.push(value.slice(0, room));
                kept += Math.max(room, 0);
                more = true;
                break;
            }
            chunks.push(value);
            kept += value.length;
        }
    } catch (error) {
        if (aborted(signal, error)) throw error;
        throw new DocumentReadError('incomplete', { status: response.status });
    } finally {
        signal?.removeEventListener('abort', stop);
        if (!finished) reader.cancel().catch(() => {});
    }
    const bytes = new Uint8Array(kept);
    let offset = 0;
    for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.length;
    }
    return { bytes, more };
}

async function errorReason(response, signal) {
    try {
        const { bytes } = await readPrefix(response, ERROR_BODY_MAX_BYTES, signal);
        return String(JSON.parse(new TextDecoder().decode(bytes))?.reason_code || '');
    } catch (error) {
        if (aborted(signal, error)) throw error;
        return '';
    }
}

async function fetchDocument(url, { signal, limit, knownSize, fetchImpl }) {
    const size = Number(knownSize);
    // A known small file is fetched whole: a Range on an empty file answers 416.
    const ranged = !(knownSize !== null && knownSize !== '' && Number.isFinite(size) && size >= 0 && size <= limit);
    let response;
    try {
        response = await fetchImpl(url, { signal, headers: ranged ? { Range: `bytes=0-${limit - 1}` } : {} });
    } catch (error) {
        if (aborted(signal, error)) throw error;
        throw new DocumentReadError('network', { reason: String(error?.message || error) });
    }
    // Only the exact empty-file answer is an empty document; any other 416 is a refusal.
    if (response.status === 416 && /^bytes \*\/0$/.test(response.headers.get('content-range') || '')) {
        response.body?.cancel?.().catch(() => {});
        return { bytes: new Uint8Array(0), total: 0, partial: false };
    }
    if (response.status !== 200 && response.status !== 206) {
        throw new DocumentReadError('http', { status: response.status, reason: await errorReason(response, signal) });
    }
    const range = (response.headers.get('content-range') || '').match(/^bytes (\d+)-(\d+)\/(\d+)$/);
    if (response.status === 206 && (!range || Number(range[1]) !== 0)) {
        response.body?.cancel?.().catch(() => {});
        throw new DocumentReadError('http', { status: 206, reason: 'unexpected_range' });
    }
    // A content-length counts encoded bytes once a content-encoding applies.
    const length = response.headers.has('content-length') && !response.headers.has('content-encoding')
        ? Number(response.headers.get('content-length')) : NaN;
    const total = range ? Number(range[3]) : Number.isFinite(length) ? length : null;
    const expected = range ? Number(range[2]) + 1 : total;
    const { bytes, more } = await readPrefix(response, limit, signal);
    // A body that stops before its own declared length is a broken transfer,
    // not a deliberate prefix: it is never presented as the document.
    if (!more && expected !== null && bytes.length < Math.min(expected, limit)) {
        throw new DocumentReadError('incomplete', { status: response.status });
    }
    return { bytes, total, partial: more || (total !== null && total > bytes.length) };
}

/**
 * Bounded read of one delivered copy: `{bytes, total, partial}`. `total` is the
 * file's full size when the source states it, else null. A refused or
 * unavailable copy throws DocumentReadError with the route's status and
 * reason code; there is no fallback to another address of the file. The read
 * ends with `signal` and, as `timeout`, after `timeoutMs`.
 */
export async function readDocumentBytes(source, {
    signal, limit = DOCUMENT_READER_MAX_BYTES, knownSize = null, timeoutMs = DOCUMENT_READER_TIMEOUT_MS,
    fetchImpl = apiFetch,
} = {}) {
    if (source?.base64) return inlineBytes(source.base64, limit);
    if (!source?.url) throw new DocumentReadError('unavailable');
    const controller = new AbortController();
    const stop = () => controller.abort();
    if (signal?.aborted) stop();
    else signal?.addEventListener('abort', stop, { once: true });
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; stop(); }, timeoutMs);
    try {
        return await fetchDocument(source.url, { signal: controller.signal, limit, knownSize, fetchImpl });
    } catch (error) {
        if (timedOut && !signal?.aborted) throw new DocumentReadError('timeout');
        throw error;
    } finally {
        clearTimeout(timer);
        signal?.removeEventListener('abort', stop);
    }
}

/**
 * The image at `src`, loaded and decoded by the engine in a detached <img>,
 * which the promise resolves with. It rejects DocumentReadError `image` when the
 * engine cannot load or show it and `timeout` after `timeoutMs`; with `signal`
 * it stops loading (its source is dropped) and rejects as an abort. However it
 * ends, its listeners and timer are gone, so a late event does nothing.
 */
export function loadReaderImage(src, { signal, alt = '', doc = globalThis.document, timeoutMs = DOCUMENT_READER_TIMEOUT_MS } = {}) {
    return new Promise((resolve, reject) => {
        const image = doc.createElement('img');
        image.className = 'document-reader-image';
        image.setAttribute('alt', alt);
        let timer = null;
        const finish = (error) => {
            clearTimeout(timer);
            image.removeEventListener('load', onLoad);
            image.removeEventListener('error', onError);
            signal?.removeEventListener('abort', onAbort);
            if (!error) { resolve(image); return; }
            image.removeAttribute('src');
            reject(error);
        };
        const onLoad = () => finish(null);
        const onError = () => finish(new DocumentReadError('image'));
        const onAbort = () => finish(new DOMException('The load was stopped.', 'AbortError'));
        if (signal?.aborted) { onAbort(); return; }
        image.addEventListener('load', onLoad);
        image.addEventListener('error', onError);
        signal?.addEventListener('abort', onAbort, { once: true });
        timer = setTimeout(() => finish(new DocumentReadError('timeout')), timeoutMs);
        image.setAttribute('src', src);
    });
}

/**
 * UTF-8 text of the read bytes. A partial prefix may end inside a character:
 * that trailing sequence is withheld, not called malformed. Bytes that are not
 * UTF-8 show as U+FFFD with `malformed`; a NUL byte marks the file `binary`.
 */
export function decodeDocumentText(bytes, { partial = false } = {}) {
    if (bytes.includes(0)) return { text: '', binary: true, malformed: false };
    try {
        return { text: new TextDecoder('utf-8', { fatal: true }).decode(bytes, { stream: partial }), binary: false, malformed: false };
    } catch {
        return { text: new TextDecoder('utf-8').decode(bytes, { stream: partial }), binary: false, malformed: true };
    }
}

const FAILURES = {
    review_source_changed: 'The selected saved answer changed or is unavailable. This view has not switched to another answer.',
    review_source_unavailable: 'The selected saved review text is missing, unreadable, or failed its integrity check. Readable neighboring reviews remain available.',
    review_source_invalid: 'This review has no valid saved source selector.',
    artifact_unverified: 'This delivered copy is missing or did not pass its integrity check, so it is not shown. Ask Ouroboros to send the file again.',
    artifact_identity_changed: 'This file changed after it was delivered, so the delivered copy cannot be shown. Ask Ouroboros to send the file again.',
    artifact_unavailable: 'This device cannot read the delivered copy right now.',
    upload_unavailable: 'This attachment is no longer stored, so it cannot be shown. Attach the file again to share it.',
    upload_unreadable: 'Ouroboros cannot read this attachment right now.',
};

// `[template, params, retriable]`: what the owner reads and whether Retry can help.
function failure(error, file) {
    if (file.kind === 'image') {
        if (error?.code === 'timeout') {
            return ['The image did not arrive within {seconds} seconds, so loading stopped.',
                { seconds: Math.round(DOCUMENT_READER_TIMEOUT_MS / 1000) }, true];
        }
        // Inline bytes the engine refused stay refused; an address may answer next time.
        return String(file.src).startsWith('data:')
            ? ['This app cannot show the image. Download it to open it in another app.', {}, false]
            : ['Could not load the image. Retry, or download it to open it in another app.', {}, true];
    }
    if (error?.code === 'unavailable') return ['This message carries no readable copy of the file.', {}, false];
    if (error?.code === 'incomplete') return ['The transfer ended before the whole document arrived.', {}, true];
    if (error?.code === 'timeout') {
        return ['The document did not arrive within {seconds} seconds, so loading stopped.',
            { seconds: Math.round(DOCUMENT_READER_TIMEOUT_MS / 1000) }, true];
    }
    if (error?.code === 'unsupported') return ['This app cannot read the document here. Download it to open it in another app.', {}, false];
    if (error?.code !== 'http') return ['Could not reach Ouroboros to load the document.', {}, true];
    const gone = error.status === 404 || error.reason === 'artifact_identity_changed';
    if (FAILURES[error.reason]) return [FAILURES[error.reason], {}, !gone];
    if (error.status === 404) return ['The delivered copy is no longer available. Ask Ouroboros to send the file again.', {}, false];
    return ['Could not load the document (HTTP {status}).', { status: error.status }, true];
}

let readerCount = 0;
// An image takes the same two view buttons: the whole image fitted, never enlarged, or its own pixels, scrolled.
const IMAGE_VIEWS = [['fit', 'Fit'], ['actual', 'Actual size']];

const DIALOG_HTML = `
    <div class="document-reader-panel">
        <header class="document-reader-head has-lead">
            <div class="document-reader-lead">
                <div class="document-reader-identity" data-i18n-authored>
                    <h2 class="document-reader-title" tabindex="0"></h2>
                    <div class="document-reader-meta"></div>
                </div>
                <div class="document-reader-actions">
                    <div class="document-reader-views" role="group" aria-label="View" hidden>
                        <button type="button" class="btn btn-default" data-reader-view="formatted" aria-pressed="true">Formatted</button>
                        <button type="button" class="btn btn-default" data-reader-view="source" aria-pressed="false">Source</button>
                    </div>
                    <div class="document-reader-file">
                        <button type="button" class="btn btn-default" data-reader-action="copy" hidden>Copy</button>
                        <button type="button" class="btn btn-default" data-reader-action="open">Open</button>
                        <button type="button" class="btn btn-default" data-reader-action="download">Download</button>
                    </div>
                </div>
            </div>
            <button type="button" class="btn btn-default document-reader-close" data-reader-action="close" aria-label="Close document">Close</button>
        </header>
        <div class="document-reader-notice" role="note" hidden></div>
        <div class="document-reader-body" tabindex="0" role="region" aria-label="Document text" aria-busy="true"></div>
    </div>`;

/**
 * One reader per chat instance. `open(file, {owner, returnFocus, pointer})`
 * shows a modal over the conversation with the reading region focused, so keys
 * scroll it at once; opened by a pointer, its focus ring waits for the first key.
 * A document `file` carries its message's copy as `reader`; an `image` one, the
 * `src` its tile shows (and `title` when its header names more than the file).
 * `release(root)` closes it when the message that opened it leaves; `destroy()`
 * with the chat. Closing aborts the read, disposes Markdown enhancements and
 * returns focus to the card when it is still there, without scrolling, so the
 * conversation keeps its place. `actions.open/download` are the card's existing
 * file actions, and `actions.copy` the photo menu's, offered for an image.
 */
export function createDocumentReader({ actions = {}, formatSize = (bytes) => `${bytes} B`, doc = globalThis.document } = {}) {
    let session = null;
    let destroyed = false;

    function setNotice(current, lines) {
        const notice = current.dialog.querySelector('.document-reader-notice');
        notice.replaceChildren(...lines.map(([key, params]) => {
            const line = doc.createElement('p');
            fmtInto(line, key, params);
            return line;
        }));
        notice.hidden = !lines.length;
    }

    function showStatus(current, key, params = {}, { retry = false, tone = 'info' } = {}) {
        const body = current.body;
        destroyChatMarkdown(body);
        body.classList.remove('is-image', 'is-actual');
        const status = doc.createElement('div');
        status.className = `document-reader-status is-${tone}`;
        status.setAttribute('role', tone === 'error' ? 'alert' : 'status');
        const text = doc.createElement('p');
        fmtInto(text, key, params);
        status.append(text);
        if (retry) {
            const button = doc.createElement('button');
            button.type = 'button';
            button.className = 'btn btn-default';
            button.dataset.readerAction = 'retry';
            button.textContent = 'Retry';
            status.append(button);
        }
        body.replaceChildren(status);
        body.setAttribute('aria-busy', String(tone === 'loading'));
    }

    function showContent(current) {
        const body = current.body;
        destroyChatMarkdown(body);
        const formatted = current.kind === 'markdown' && current.view === 'formatted';
        for (const button of current.dialog.querySelectorAll('[data-reader-view]')) {
            button.setAttribute('aria-pressed', String(button.dataset.readerView === current.view));
        }
        let content;
        if (current.kind === 'image') {
            content = current.image;
            body.classList.add('is-image');
            body.classList.toggle('is-actual', current.view === 'actual');
            body.scrollLeft = 0;
        } else if (formatted) {
            content = doc.createElement('div');
            content.className = 'document-reader-markdown';
            // Document semantics: a single newline inside a paragraph is a soft break.
            mountChatMarkdown(content, current.text, { softBreaks: true });
        } else {
            content = doc.createElement('pre');
            content.className = 'document-reader-source';
            content.textContent = current.text;
        }
        // The document's own words, attributes included, are never interface text.
        content.setAttribute('data-i18n-authored', '');
        body.replaceChildren(content);
        body.setAttribute('aria-busy', 'false');
        body.scrollTop = 0;
        if (formatted) enhanceChatMarkdown(content);
    }

    async function load(current) {
        current.controller?.abort();
        const controller = new AbortController();
        current.controller = controller;
        current.loaded = false;
        const views = current.dialog.querySelector('.document-reader-views');
        views.hidden = true;
        setNotice(current, []);
        const image = current.kind === 'image';
        showStatus(current, image ? 'Loading the image…' : 'Loading the document…', {}, { tone: 'loading' });
        let result;
        try {
            result = image
                ? await loadReaderImage(current.file.src, { signal: controller.signal, alt: current.file.title || current.file.filename, doc })
                : await readDocumentBytes(current.file.reader, { signal: controller.signal, knownSize: current.file.size });
        } catch (error) {
            if (session !== current || controller.signal.aborted) return;
            const [key, params, retry] = failure(error, current.file);
            showStatus(current, key, params, { retry, tone: 'error' });
            return;
        }
        if (session !== current || controller.signal.aborted) return;
        if (image) {
            current.image = result;
            current.loaded = true;
            views.hidden = false;
            showContent(current);
            return;
        }
        const decoded = decodeDocumentText(result.bytes, { partial: result.partial });
        if (decoded.binary) {
            showStatus(current, 'This file does not look like text, so it is not shown here. Download it to open it in another app.', {}, { tone: 'error' });
            return;
        }
        const notices = [];
        if (result.partial) {
            notices.push(result.total
                ? ['Showing the first {shown} of {total}. Download the file to read all of it.',
                    { shown: formatSize(result.bytes.length), total: formatSize(result.total) }]
                : ['Showing the first {shown}. Download the file to read all of it.', { shown: formatSize(result.bytes.length) }]);
        }
        if (decoded.malformed) notices.push(['Some bytes are not valid UTF-8 text and show as �.', {}]);
        setNotice(current, notices);
        current.text = decoded.text;
        current.loaded = true;
        views.hidden = current.kind !== 'markdown';
        showContent(current);
    }

    function close({ restoreFocus = true } = {}) {
        const current = session;
        if (!current) return;
        session = null;
        current.controller?.abort();
        destroyChatMarkdown(current.body);
        for (const dispose of current.disposers) {
            try { dispose(); } catch {}
        }
        // Closed before focus returns: the card behind a modal dialog is inert.
        if (typeof current.dialog.close === 'function' && current.dialog.open) current.dialog.close();
        else current.dialog.removeAttribute('open');
        current.disposeFocus?.({ restoreFocus });
        current.dialog.remove();
    }

    function open(file, { owner = null, returnFocus = doc.activeElement, pointer = false } = {}) {
        if (destroyed || !file?.kind || !(file.kind === 'image' ? file.src : file.reader)) return false;
        close({ restoreFocus: false });
        const dialog = doc.createElement('dialog');
        dialog.className = 'document-reader';
        dialog.setAttribute('aria-modal', 'true');
        dialog.innerHTML = DIALOG_HTML;
        const title = dialog.querySelector('.document-reader-title');
        readerCount += 1;
        title.id = `document-reader-title-${readerCount}`;
        dialog.setAttribute('aria-labelledby', title.id);
        title.textContent = file.title || file.filename;
        dialog.querySelector('.document-reader-meta').textContent = file.meta || '';
        dialog.querySelector('[data-reader-action="copy"]').hidden = file.kind !== 'image' || !actions.copy;
        dialog.querySelector('[data-reader-action="open"]').hidden = !actions.open || !file.canOpen;
        dialog.querySelector('[data-reader-action="download"]').hidden = !actions.download;
        const current = {
            dialog, file, owner, kind: file.kind, view: 'formatted', text: '', image: null, loaded: false, controller: null,
            disposers: [], body: dialog.querySelector('.document-reader-body'), disposeFocus: null,
        };
        if (file.kind === 'image') {
            dialog.querySelectorAll('[data-reader-view]').forEach((button, index) => {
                [button.dataset.readerView, button.textContent] = IMAGE_VIEWS[index];
            });
            current.view = 'fit';
            current.body.setAttribute('aria-label', 'Image');
        }
        const listen = (target, type, handler) => {
            target.addEventListener(type, handler);
            current.disposers.push(() => target.removeEventListener(type, handler));
        };
        listen(dialog, 'click', (event) => {
            if (session !== current) return;
            // The panel fills the dialog box, so the dialog itself is only hit through its backdrop.
            if (event.target === dialog) { close(); return; }
            const action = event.target.closest?.('[data-reader-action]')?.dataset.readerAction;
            const view = event.target.closest?.('[data-reader-view]')?.dataset.readerView;
            if (action === 'close') close();
            else if (action === 'retry') {
                // The status that held Retry is replaced: the reading region keeps the focus.
                void load(current);
                current.body.focus();
            }
            else if (action === 'open' || action === 'download' || action === 'copy') void actions[action]?.(file);
            else if (view && view !== current.view && current.loaded) {
                current.view = view;
                showContent(current);
            }
        });
        listen(dialog, 'cancel', (event) => { event.preventDefault(); close(); });
        current.body.classList.toggle('is-quiet-focus', pointer);
        listen(dialog, 'keydown', () => current.body.classList.remove('is-quiet-focus'));
        listen(dialog, 'close', () => { if (session === current) close({ restoreFocus: false }); });
        doc.body.appendChild(dialog);
        session = current;
        // The engine refocuses what was focused at showModal when the dialog closes; WebKit does
        // not focus a pressed button, so the opener takes focus first, in place.
        if (returnFocus && returnFocus !== doc.activeElement) returnFocus.focus?.({ preventScroll: true });
        if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
        current.disposeFocus = bindDialogFocus(dialog, {
            initialFocus: current.body, returnFocus, onEscape: () => close(), preventScroll: true,
        });
        void load(current);
        return true;
    }

    function release(root) {
        const owner = session?.owner;
        if (owner && (owner === root || root?.contains?.(owner))) close({ restoreFocus: false });
    }

    function destroy() {
        close({ restoreFocus: false });
        destroyed = true;
    }

    return { open, close, release, destroy, isOpen: () => session !== null };
}
