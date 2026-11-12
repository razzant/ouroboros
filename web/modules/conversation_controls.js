/** A loaded conversation feature mounts in the ordinary room's composer lifetime. */
import { getContributions } from './contributions.js';

let hosts = null;
let openSettings = () => {};
let roomOptions = () => ({});
const rooms = new Set();
export function startConversations({ hosts: runtime, openSettings: settings, roomOptions: options }) {
    hosts = runtime;
    openSettings = settings;
    roomOptions = options || (() => ({}));
    runtime.onProvidersChange(() => { for (const room of rooms) room.sync(); });
    for (const room of rooms) room.sync();
}
// The send button's face before a conversation feature has mounted: the face the feature last asked this device to
// show on an empty field (Talk with its sign), painted with the composer's first frame, so a load never turns Send into
// Talk and the feature's own paint changes no pixel. The feature takes the face over when it mounts (`claim`) and keeps
// this device's answer as its last good one until its own configuration read answers; its answers update the memory
// (`remember`). Before the mount a press cannot start a call, so the face is announced as not ready (aria-disabled)
// while it looks the same. It gives way to Send when the owner fills the composer first or no single live conversation
// feature is declared, and quietly when anything else paints the button.
const ENTRY_KEY = 'ouro:conversation-entry';
function readEntry() {
    try {
        const face = JSON.parse(globalThis.localStorage?.getItem(ENTRY_KEY) || 'null');
        return typeof face?.html === 'string' && face.html ? face : null;
    } catch { return null; }
}
function writeEntry(face) {
    try {
        if (face?.html) globalThis.localStorage?.setItem(ENTRY_KEY, JSON.stringify({ html: face.html, title: face.title || '' }));
        else globalThis.localStorage?.removeItem(ENTRY_KEY);
    } catch { /* storage blocked */ }
}

function reserveEntry(page, store) {
    const send = page?.querySelector?.('.chat-send-inline');
    const field = page?.querySelector?.('textarea');
    const preview = page?.querySelector?.('.chat-attachment-preview');
    const filled = () => Boolean(field?.value?.trim()) || Boolean(preview?.querySelector?.('.attach-badge'));
    const face = send && !filled() ? readEntry() : null;
    let reserved = Boolean(face);
    const end = () => { reserved = false; delete send.dataset.entryReserved; send.removeAttribute('aria-disabled'); };
    const release = () => {
        if (!reserved) return;
        end();
        send.textContent = 'Send';
        send.title = 'Send message';
    };
    if (reserved) {
        send.innerHTML = face.html;
        send.title = face.title;
        send.dataset.entryReserved = '1';
        send.setAttribute('aria-disabled', 'true');
    }
    // Typing, a recalled message, dictation or a staged file fill the composer (the chat's own history keys run first).
    const check = () => { if (filled()) release(); };
    const watching = reserved && typeof MutationObserver === 'function';
    const painted = watching ? new MutationObserver(() => { if (reserved) end(); }) : null;
    const staged = watching && preview ? new MutationObserver(check) : null;
    painted?.observe(send, { childList: true });
    staged?.observe(preview, { childList: true });
    const typed = reserved ? ['input', 'change', 'keydown'] : [];
    for (const type of typed) field?.addEventListener?.(type, check);
    const settle = () => {
        if (!reserved || !store.status().loaded) return;
        // A room mounts exactly one live feature; with none (or an ambiguous pair) nothing will take the face over.
        const live = store.providers('conversation').filter((item) => item.runtime === 'client' && !store.fault(item.key));
        if (live.length !== 1) { writeEntry(null); release(); }
    };
    const unwatch = reserved ? store.onChange(settle) : null;
    settle();
    return {
        claim() { const shown = reserved; if (shown) end(); return shown; },
        release,
        remember(next) { writeEntry(next); if (!next?.html) release(); },
        dispose() {
            painted?.disconnect(); staged?.disconnect();
            for (const type of typed) field?.removeEventListener?.(type, check);
            unwatch?.();
        },
    };
}

export function bindConversationControls({ composer, page, chatId, ws, contributions = getContributions() }) {
    let active = null, key = null;
    const priorPending = composer.hasPendingWork;
    composer.hasPendingWork = () => Boolean(active?.retains?.()) || priorPending();
    const entry = reserveEntry(page, contributions);
    const room = { sync() {
        const available = hosts?.clientProviders('conversation') || [];
        // Ambiguous feature selection never substitutes another active feature.
        const provider = available.length === 1 ? available[0] : null;
        if (provider === key) return;
        active?.destroy(); active = null; key = provider;
        if (typeof provider?.impl.mountRoom === 'function')
            active = provider.impl.mountRoom({ page, chatId, ws, composer, openSettings, rememberEntry: entry.remember, claimEntry: entry.claim,
                onScreen: () => page.getClientRects().length > 0, ...roomOptions({ page, chatId }) });
        // A feature takes the face over while it mounts; one that did not (an older one) paints its own, from Send.
        if (provider) entry.release();
    } };
    rooms.add(room); room.sync();
    return () => { rooms.delete(room); active?.destroy(); entry.dispose(); composer.hasPendingWork = priorPending; };
}

// The room facts a call surface takes from the shell. A directly opened Project need not be in the
// bounded sidebar census, so an unknown chat id names the active project panel.
export function projectRoomOptions({ rows, nav, state, title, freshProjectRow, openProjectPanel, showPage, putAway }) {
    return ({ page, chatId }) => {
        const project = chatId === 1 ? null : rows().find(row => Number(row.chat_id) === chatId)
            || { id: nav.activeProjectId, chat_id: chatId, name: title.textContent };
        const current = () => project ? nav.activeProjectId === project.id : !nav.activeProjectId;
        return {
            label: () => project ? freshProjectRow(project).name || project.id : 'Main',
            onScreen: () => state.activePage === 'chat' && !page.hidden && current(),
            reveal: () => project ? openProjectPanel({ id: project.id, name: project.name }, { openOnly: true }) : showPage('chat'),
            // A call that ended in a Project room not on screen lets the room go away as any room left does.
            onSettled: () => queueMicrotask(() => { if (project && !current()) putAway(project.id); }),
        };
    };
}
