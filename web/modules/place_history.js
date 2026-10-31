// Captured conversation content mounted inside the Place sheet. The sheet owns navigation and drafts.
import { apiClient } from './api_client.js';
import { escapeHtmlAttr as escapeHtml } from './ui_primitives.js';
import { mountChatMarkdown, destroyChatMarkdown } from './chat_markdown.js';
import { tr } from './i18n.js';
import { sendsAsHtml, signalLabels, heardFactWord } from './consciousness_words.js';

export const INDEX_NOTE = 'The list finds conversations with saved incoming Place events. Older records and conversations where it only wrote may not be listed.';
const LOCAL_NOTE = 'Only records saved by Ouroboros are shown. Some events may never have reached it. Attention and delivery appear only when recorded; a saved reply alone does not confirm delivery.';
// Imported history spans years: a time outside the current year names its year.
const timeText = (value, now = new Date()) => {
    const date = new Date(value);
    return Number.isFinite(date.getTime()) ? date.toLocaleString([], { ...(date.getFullYear() !== now.getFullYear()
        ? { year: 'numeric' } : {}), month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
};

export function windowIssues(page) {
    const reasons = page?.window?.truncated_by || [];
    const gaps = Object.values(page?.coverage?.spans || {}).flatMap((span) => span?.gaps || []);
    const importing = reasons.some((reason) => reason.endsWith('_importing')) || gaps.includes('importing');
    const unavailable = gaps.some((gap) => gap !== 'importing')
        || reasons.some((reason) => /^(chat|progress)_/.test(reason) && !reason.endsWith('_importing'));
    const bounded = reasons.some((reason) => !['page', 'quota'].includes(reason) && !/^(chat|progress)_/.test(reason));
    const unknown = !page?.window || !Array.isArray(page.window.truncated_by)
        || (!page.window.complete && !reasons.length);
    return { importing, unavailable, bounded, unknown, retry: importing || unavailable || bounded || unknown };
}

/** Each returned range owns its uncertainty. A normal older page is not a gap. The pages read what
 * was saved when the newest of them was read: their start is that record's, never later imports'. */
export function capturedWindowNote(page) {
    const issues = windowIssues(page);
    return [issues.importing ? 'Older records are still being imported.' : '',
        issues.unavailable ? 'Some records in this returned range are unavailable.' : '',
        issues.bounded ? 'This returned range was limited.' : '',
        issues.unknown ? 'Coverage was not fully reported for this returned range.' : '',
        !issues.retry ? (page.has_more ? 'Earlier saved records are available.' : 'Start of the record saved when this was read. Refresh history to include anything saved since. The service may hold earlier messages.') : '',
    ].filter(Boolean).join(' ');
}

export function historySpeaker(row) {
    if (row.role === 'assistant') return 'Ouroboros';
    if (row.role === 'system') return 'System';
    return String(row.sender_label || row.sender_session_id || 'Participant');
}

export function historyFactWords(row) {
    const facts = row.place_history || {};
    const attention = facts.attention;
    const heard = { paused: 'Heard while Paused', off: 'Heard while consciousness was off',
        no_wake_match: 'Heard; no wake matched', current_author: 'Passed to the current author', disconnected: 'Heard after disconnection' };
    return [attention?.decision === 'focus_started' ? 'Focus started' : attention?.decision === 'heard'
        ? heard[attention.reason] || 'Heard' : '',
    { authored: 'Delivery not confirmed', accepted: 'Accepted by service; delivery not confirmed',
        delivered: 'Delivered', failed: 'Delivery failed', uncertain: 'Delivery outcome unknown' }[facts.delivery_state] || ''].filter(Boolean);
}

export function placeRoomsHtml({ rooms = null, coverage = null, error = '' } = {}) {
    const state = error ? `<p role="alert" data-i18n-authored>${escapeHtml(error)}</p>` : rooms === null
        ? '<p class="place-sheet-note" role="status">Reading captured conversations…</p>' : rooms.length
            ? `<ul class="consciousness-place-rows place-rooms">${rooms.map((room) => `<li><button type="button" class="consciousness-place-row place-room-row" data-place-room="${room.room_chat_id}">
                <span class="consciousness-place-name" data-i18n-authored>${escapeHtml(room.label || room.address)}</span>
                <span class="consciousness-place-facts">${!room.label ? '<span>Name not recorded</span> · ' : ''}<time data-i18n-authored>${escapeHtml(timeText(room.last_at))}</time>${room.unanswered_count === null ? ' · Focus count unavailable' : room.unanswered_count > 0 ? ` · ${room.unanswered_count} open ${room.unanswered_count === 1 ? 'focus' : 'focuses'}` : ''}</span>
                ${room.thread_id ? `<span class="place-sheet-note">Thread: <span data-i18n-authored>${escapeHtml(room.thread_id)}</span></span>` : ''}
            </button></li>`).join('')}</ul>`
            : '<p>No saved conversation found in this index.</p>';
    return `${state}${coverage?.importing ? '<p class="place-sheet-note" role="status">Older records are still being imported; this list may be incomplete.</p>' : ''}
        ${coverage?.gaps ? '<p class="place-sheet-note" role="status">Some saved records are unavailable; this list may be incomplete.</p>' : ''}
        <p class="place-sheet-note">${INDEX_NOTE}</p>${error || coverage?.importing || coverage?.gaps
            ? '<button type="button" class="btn btn-default btn-sm" data-place-rooms-retry>Try again</button>' : ''}`;
}

export function placeHistoryState(place) {
    return { silent: 'Silent now: it still hears and wakes here; what it would say or change here is held.',
        paused: 'Paused now: it does not wake or act here; what the connector still reports is saved.',
        active: 'Active now: it hears, wakes, speaks and acts here.' }[place.state] || 'The current Place state is unavailable.';
}

/** A row's shown time: the provider send time its arrival recorded, else when it was saved. The title keeps
 * both; an import's saved time is its receipt, since its `ts` is the send time. */
export function shownTime(row) {
    const facts = row.place_history || {};
    const saved = facts.historical_import?.received_at || row.ts || '';
    return facts.sent_at ? { at: facts.sent_at, title: `Sent ${facts.sent_at} · saved ${saved}` } : { at: row.ts || '', title: row.ts || '' };
}

function updateHeardLabels(host, place) {
    const labels = { ...signalLabels(place.facts), ...place.labels };
    for (const node of host.querySelectorAll('[data-heard-fact]')) {
        const fact = node.dataset.heardFact, declared = labels[fact];
        const word = heardFactWord(fact, labels);
        if (node.textContent !== word) node.textContent = word;
        if (declared) node.setAttribute('data-i18n-authored', ''); else node.removeAttribute('data-i18n-authored');
    }
}

function messageNode(row, markdown, place) {
    const voice = row.role === 'user' ? (row.place_history.owner_confirmed ? 'owner' : 'participant') : row.role;
    const facts = row.place_history;
    const heard = facts.attention?.decision === 'heard' ? (facts.facts || []).filter((fact) => heardFactWord(fact)).map((fact) =>
        `<span class="place-convo-fact" data-heard-fact="${escapeHtml(fact)}"></span>`).join('') : '';
    const sender = facts.sends_as && sendsAsHtml({ ...facts.sends_as, owner: facts.sends_as_owner });
    const article = document.createElement('article'); article.className = `place-convo-row is-${voice}`;
    const meta = document.createElement('div'); meta.className = 'place-convo-meta';
    const time = shownTime(row);
    meta.innerHTML = `<span class="place-convo-who"${row.role === 'user' ? ' data-i18n-authored' : ''}>${escapeHtml(historySpeaker(row))}</span>${voice === 'owner' ? '<span>(you)</span>' : ''}
        <time data-i18n-authored datetime="${escapeHtml(time.at)}" title="${escapeHtml(time.title)}">${escapeHtml(timeText(time.at))}</time>
        ${historyFactWords(row).map((fact) => `<span class="place-convo-fact">${escapeHtml(fact)}</span>`).join('')}${heard}${sender ? `<span class="place-convo-fact">${sender}</span>` : ''}`;
    updateHeardLabels(meta, place);
    const text = document.createElement('div'); text.className = 'chat-markdown place-convo-text'; text.dataset.i18nAuthored = '';
    if (row.role === 'assistant' || row.markdown) { mountChatMarkdown(text, row.text, { softBreaks: true }); markdown.add(text); }
    else { text.classList.add('place-history-plain'); text.textContent = row.text; }
    article.append(meta, text);
    if (facts.held === true && facts.checked === false && typeof facts.held_text === 'string') {
        const label = document.createElement('p'); label.className = 'place-sheet-note';
        label.textContent = tr('places.history.held_unchecked', 'held, unchecked');
        const held = document.createElement('div'); held.className = 'chat-markdown place-convo-text'; held.dataset.i18nAuthored = '';
        mountChatMarkdown(held, facts.held_text, { softBreaks: true }); markdown.add(held);
        article.append(label, held);
    }
    const attachments = Array.isArray(row.attachments) ? row.attachments : [];
    if (attachments.length || row.msg_type) {
        const note = document.createElement('p'); note.className = 'place-sheet-note';
        note.innerHTML = attachments.length ? `Retained attachments: <span data-i18n-authored>${escapeHtml(attachments.map((entry) => entry.name || entry.label || 'attachment').join(', '))}</span>`
            : 'This reader shows the stored text of the retained attachment.';
        article.append(note);
    }
    return article;
}

/** One selected room; Back/Close dispose requests and Markdown before the sheet changes view. */
export function mountPlaceHistory(host, place, room, { onBack, readHistory = apiClient.chatHistory } = {}) {
    host.innerHTML = `<section class="place-sheet-section place-convo">
        <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-history-back>Back to Place</button></div>
        <p class="place-silent-card" role="note" data-history-state></p>
        <p class="place-sheet-note">${LOCAL_NOTE}</p>
        <div data-history-coverage></div>
        <p class="place-sheet-status ui-status" data-history-status role="status" data-i18n-authored></p>
        <div class="place-sheet-actions"><button type="button" class="btn btn-default btn-sm" data-history-retry hidden>Try again</button>
        <button type="button" class="btn btn-default btn-sm" data-history-older hidden>Load older</button>
        <button type="button" class="btn btn-default btn-sm" data-history-refresh>Refresh history</button></div>
        <div class="place-convo-rows" data-history-messages></div>
        <p class="place-sheet-note">Reading sends nothing, marks nothing read and starts no work.</p>
    </section>`;
    const find = (selector) => host.querySelector(selector);
    let closed = false, busy = false, generation = 0, controller = null, retry = null;
    const pages = [], markdown = new Set();
    const clear = () => { markdown.forEach(destroyChatMarkdown); markdown.clear(); find('[data-history-messages]').replaceChildren(); };
    const paint = () => {
        clear();
        const seen = new Set();
        for (const { page } of pages) for (const row of page.messages) {
            if (!row.place_history || row.is_progress || !['user', 'assistant', 'system'].includes(row.role)) continue;
            const id = row.row_address || row.history_id;
            if (id && seen.has(id)) continue;
            if (id) seen.add(id);
            find('[data-history-messages]').append(messageNode(row, markdown, place));
        }
        find('[data-history-coverage]').innerHTML = pages.map(({ page }, index) => `<div class="place-convo-edge">
            <p class="place-sheet-note">${escapeHtml(capturedWindowNote(page))}</p>${windowIssues(page).retry
                ? `<button type="button" class="btn btn-default btn-sm" data-history-range="${index}">Retry this range</button>` : ''}</div>`)
            .filter((_, index) => index === 0 || windowIssues(pages[index].page).retry).join('');
        find('[data-history-older]').hidden = pages[0]?.page.has_more !== true;
        find('[data-history-status]').textContent = find('[data-history-messages]').childElementCount ? '' : 'No stored messages in these returned ranges.';
    };
    // `replacing`: a loaded range's index, or a newest page that replaces every range once it arrives:
    // 'chain' (Refresh history) or 'changed' (the shown chain's view changed).
    async function loadPage(askedCursor = null, replacing = null) {
        if (closed || busy) return;
        controller?.abort(); controller = new AbortController(); const version = ++generation;
        const focused = host.contains(document.activeElement) ? document.activeElement : null;
        busy = true; retry = null; find('[data-history-retry]').hidden = true;
        find('[data-history-status]').textContent = 'Reading captured history…';
        find('[data-history-status]').dataset.tone = 'muted';
        find('[data-history-older]').disabled = true; find('[data-history-refresh]').disabled = true;
        let cursor = askedCursor, target = replacing;
        try {
            let page;
            try {
                page = await readHistory({ chatId: room.room_chat_id, cursor, signal: controller.signal });
            } catch (error) {
                // A changed view (such as the room's first imported history) refuses every cursor of the shown
                // chain, so the newest page is read afresh; the shown ranges stay until it arrives.
                if (cursor === null || error?.body?.reason_code !== 'history_view_changed' || closed || version !== generation) throw error;
                cursor = null; target = 'changed';
                page = await readHistory({ chatId: room.room_chat_id, cursor, signal: controller.signal });
            }
            if (closed || version !== generation) return;
            if (!Array.isArray(page?.messages) || page.messages.some((row) => typeof row?.text !== 'string')
                || (page.has_more === true && (typeof page.next_cursor !== 'string' || !page.next_cursor || page.next_cursor === cursor))) {
                throw new Error('Captured history page is invalid; no page was skipped.');
            }
            const scroller = host.closest('[data-place-body]') || host;
            const beforeHeight = scroller.scrollHeight, beforeTop = scroller.scrollTop;
            const hadRows = pages.length > 0;
            const saved = { page, askedCursor: page.page_cursor || cursor };
            if (target === 'chain' || target === 'changed') pages.splice(0, pages.length, saved);
            else if (target !== null) pages[target] = saved; else pages.unshift(saved);
            paint();
            if (target === 'changed') find('[data-history-status]').textContent = 'Saved history changed, so it was read again from the newest messages.';
            if (hadRows && target === null) scroller.scrollTop = beforeTop + scroller.scrollHeight - beforeHeight;
        } catch (error) {
            if (closed || version !== generation) return;
            find('[data-history-status]').textContent = error?.message || String(error);
            find('[data-history-status]').dataset.tone = 'error';
            retry = () => loadPage(cursor, target); find('[data-history-retry]').hidden = false;
        } finally {
            if (!closed && version === generation) {
                busy = false; find('[data-history-older]').disabled = false; find('[data-history-refresh]').disabled = false;
                // A control disabled while it read, or gone with its range, dropped the focus: give it back
                // within the reader, never to the page. Focus moved elsewhere meanwhile stays there.
                const active = document.activeElement;
                if (focused && (!active || active === document.body || (host.contains(active) && active.hidden))) {
                    const older = find('[data-history-older]');
                    (focused.isConnected && !focused.hidden ? focused : older.hidden ? find('[data-history-refresh]') : older)
                        .focus({ preventScroll: true });
                }
            }
        }
    }
    const click = (event) => {
        if (event.target.closest('[data-history-back]')) { onBack(); return; }
        if (event.target.closest('[data-history-retry]')) { void retry?.(); return; }
        if (event.target.closest('[data-history-refresh]')) { void loadPage(null, 'chain'); return; }
        const range = event.target.closest('[data-history-range]');
        if (range) { const index = Number(range.dataset.historyRange); void loadPage(pages[index].askedCursor, index); return; }
        if (event.target.closest('[data-history-older]')) void loadPage(pages[0]?.page.next_cursor);
    };
    host.addEventListener('click', click);
    const updatePlace = (value) => {
        place = value;
        find('[data-history-state]').textContent = placeHistoryState(place);
        updateHeardLabels(host, place);
    };
    updatePlace(place);
    void loadPage();
    return { updatePlace, close() {
        if (closed) return;
        closed = true; generation += 1; controller?.abort(); clear(); host.removeEventListener('click', click); host.replaceChildren();
    } };
}
