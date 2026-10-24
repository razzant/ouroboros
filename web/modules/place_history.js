// Read-only captured conversation reader. No composer, task admission or transport effect.
import { apiClient } from './api_client.js';
import { readPlaceRooms } from './consciousness_api.js';
import { bindDialogFocus } from './ui_interactions.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { mountChatMarkdown, destroyChatMarkdown } from './chat_markdown.js';

/** A returned window describes its own coverage, never the whole remote service. */
export function capturedWindowNote(page) {
    if (!page?.window) return 'Coverage was not reported for this captured window.';
    return page.window.complete === true ? 'Captured local conversation; not a complete remote-provider archive.'
        : 'This captured window has incomplete coverage. Earlier or unavailable rows may be missing.';
}

export function historySpeaker(row) {
    return String(row.sender_label || row.sender_session_id
        || (row.role === 'assistant' ? 'Ouroboros' : row.role === 'system' ? 'System' : 'Participant'));
}

/** Open only rooms supplied by the selected Place's canonical index. */
export function openPlaceHistory(place, { returnFocus = document.activeElement, readRooms = readPlaceRooms,
    readHistory = apiClient.chatHistory } = {}) {
    const dialog = document.createElement('dialog');
    dialog.className = 'document-reader place-sheet place-history';
    dialog.setAttribute('aria-labelledby', 'place-history-title');
    dialog.innerHTML = `<div class="document-reader-panel">
        <div class="document-reader-head"><div class="document-reader-identity">
            <h2 class="document-reader-title" id="place-history-title">Conversations · <span data-i18n-authored>${escapeHtml(place.label)}</span></h2>
            <div class="document-reader-meta">Read-only · captured local history</div></div>
            <button type="button" class="btn btn-default document-reader-close" data-history-close>Close</button></div>
        <div class="document-reader-body place-sheet-body" tabindex="-1" data-history-body>
            <div data-history-rooms></div>
            <section class="place-sheet-section" data-history-conversation hidden>
                <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-history-back>All conversations</button></div>
                <h3 data-history-address data-i18n-authored></h3><p class="place-sheet-note" data-history-coverage></p>
                <div data-history-messages></div>
                <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-history-older>Load older</button></div>
            </section>
            <p class="place-sheet-status ui-status" data-history-status role="status"></p>
            <button type="button" class="btn btn-default btn-sm" data-history-retry hidden>Try again</button>
        </div></div>`;
    const body = dialog.querySelector('[data-history-body]');
    const find = (selector) => dialog.querySelector(selector);
    let closed = false;
    let controller = null;
    let generation = 0;
    let rooms = [];
    let selected = null;
    let cursor = null;
    let busy = false;
    let retry = null;
    let disposeFocus = null;
    const identities = new Set();
    const markdown = new Set();
    const clearMessages = () => {
        markdown.forEach((node) => destroyChatMarkdown(node)); markdown.clear(); identities.clear();
        find('[data-history-messages]').replaceChildren();
    };
    const close = () => {
        if (closed) return;
        closed = true; generation += 1; controller?.abort(); clearMessages();
        if (dialog.open && typeof dialog.close === 'function') dialog.close();
        disposeFocus?.(); dialog.remove();
    };
    function start() {
        controller?.abort(); controller = new AbortController(); generation += 1;
        busy = true; retry = null; find('[data-history-retry]').hidden = true;
        find('[data-history-older]').disabled = true;
        return { version: generation, signal: controller.signal };
    }
    function fail(error, operation) {
        find('[data-history-status]').textContent = error?.message || String(error);
        find('[data-history-status]').dataset.tone = 'error';
        retry = operation; find('[data-history-retry]').hidden = false;
    }
    async function loadRooms() {
        selected = null; cursor = null; clearMessages();
        find('[data-history-conversation]').hidden = true; find('[data-history-rooms]').hidden = false;
        const request = start(); find('[data-history-status]').textContent = 'Reading captured conversations…';
        try {
            const value = await readRooms(place.id, { signal: request.signal });
            if (closed || request.version !== generation) return;
            if (!Array.isArray(value) || value.some((room) => !Number.isSafeInteger(room?.room_chat_id)
                || room.room_chat_id <= 0 || typeof room.address !== 'string')) throw new Error('Captured room index is invalid.');
            rooms = value;
            find('[data-history-rooms]').innerHTML = `<section class="place-sheet-section"><p class="place-sheet-note">Select a captured conversation.</p><div class="place-sheet-actions">${rooms.map((room, index) =>
                `<button type="button" class="btn btn-default btn-sm" data-history-room="${index}"><span data-i18n-authored>${escapeHtml(room.address)}</span>${room.unanswered_count === null ? ' · work count unknown' : ''}</button>`).join('')}</div></section>`;
            find('[data-history-status]').textContent = rooms.length ? '' : 'No conversations have been captured for this Place yet.';
        } catch (error) { if (!closed && request.version === generation) fail(error, loadRooms); }
        finally { if (!closed && request.version === generation) busy = false; }
    }
    async function loadPage() {
        if (closed || !selected || busy) return;
        const room = selected; const askedCursor = cursor; const request = start();
        find('[data-history-status]').textContent = 'Reading captured history…';
        try {
            const page = await readHistory({ chatId: room.room_chat_id, cursor: askedCursor, signal: request.signal });
            if (closed || request.version !== generation) return;
            if (!Array.isArray(page?.messages) || page.messages.some((row) => typeof row?.text !== 'string')
                || (page.has_more === true && (typeof page.next_cursor !== 'string' || !page.next_cursor || page.next_cursor === askedCursor))) {
                throw new Error('Captured history page is invalid; no page was skipped.');
            }
            const fragment = document.createDocumentFragment();
            for (const row of page.messages) {
                const id = row.row_address || row.history_id;
                if (id && identities.has(id)) continue;
                if (id) identities.add(id);
                const article = document.createElement('article'); article.className = 'place-sheet-section place-history-message';
                const meta = document.createElement('p'); meta.className = 'document-reader-meta'; meta.dataset.i18nAuthored = '';
                meta.textContent = [historySpeaker(row), row.ts, row.system_type].filter(Boolean).join(' · ');
                const text = document.createElement('div'); text.className = 'chat-markdown'; text.dataset.i18nAuthored = '';
                article.append(meta, text);
                if (row.markdown) { mountChatMarkdown(text, row.text, { softBreaks: true }); markdown.add(text); }
                else { text.classList.add('place-history-plain'); text.textContent = row.text; }
                const attachments = Array.isArray(row.attachments) ? row.attachments : [];
                if (attachments.length || row.msg_type) {
                    const note = document.createElement('p'); note.className = 'place-sheet-note'; note.dataset.i18nAuthored = '';
                    note.textContent = attachments.length ? `Retained attachments: ${attachments.map((entry) => entry.name || entry.label || 'attachment').join(', ')}`
                        : `Retained ${row.msg_type}; this reader shows its stored text.`;
                    article.append(note);
                }
                fragment.append(article);
            }
            const host = find('[data-history-messages]');
            const hadRows = host.childElementCount > 0; const beforeHeight = body.scrollHeight; const beforeTop = body.scrollTop;
            host.prepend(fragment);
            if (hadRows) body.scrollTop = beforeTop + body.scrollHeight - beforeHeight;
            cursor = page.next_cursor || null;
            find('[data-history-coverage]').textContent = capturedWindowNote(page);
            find('[data-history-status]').textContent = host.childElementCount ? '' : 'No stored messages in this returned window.';
            find('[data-history-older]').hidden = page.has_more !== true;
            find('[data-history-older]').disabled = false;
        } catch (error) { if (!closed && request.version === generation) fail(error, loadPage); }
        finally { if (!closed && request.version === generation) busy = false; }
    }
    dialog.addEventListener('cancel', (event) => { event.preventDefault(); close(); });
    dialog.addEventListener('click', (event) => {
        if (event.target === dialog || event.target.closest('[data-history-close]')) { close(); return; }
        if (event.target.closest('[data-history-retry]')) { if (!busy) void retry?.(); return; }
        if (event.target.closest('[data-history-back]')) { void loadRooms(); return; }
        if (event.target.closest('[data-history-older]')) { void loadPage(); return; }
        const button = event.target.closest('[data-history-room]');
        const room = button && rooms[Number(button.dataset.historyRoom)];
        if (!room) return;
        controller?.abort(); generation += 1; busy = false; selected = room; cursor = null; clearMessages();
        find('[data-history-rooms]').hidden = true; find('[data-history-conversation]').hidden = false;
        find('[data-history-address]').textContent = room.address;
        find('[data-history-coverage]').textContent = '';
        void loadPage();
    });
    document.body.append(dialog);
    if (typeof dialog.showModal === 'function') dialog.showModal(); else dialog.setAttribute('open', '');
    disposeFocus = bindDialogFocus(dialog, { initialFocus: body, returnFocus, onEscape: close });
    void loadRooms();
    return { close, get closed() { return closed; } };
}
