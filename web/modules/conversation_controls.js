/** A loaded conversation feature mounts in the ordinary room's composer lifetime. */
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
export function bindConversationControls({ composer, page, chatId, ws }) {
    let active = null, key = null;
    const priorPending = composer.hasPendingWork;
    composer.hasPendingWork = () => Boolean(active?.retains?.()) || priorPending();
    const room = { sync() {
        const available = hosts?.clientProviders('conversation') || [];
        // Ambiguous feature selection never substitutes another active feature.
        const entry = available.length === 1 ? available[0] : null;
        if (entry === key) return;
        active?.destroy(); active = null; key = entry;
        if (typeof entry?.impl.mountRoom === 'function')
            active = entry.impl.mountRoom({ page, chatId, ws, composer, openSettings,
                onScreen: () => page.getClientRects().length > 0, ...roomOptions({ page, chatId }) });
    } };
    rooms.add(room); room.sync();
    return () => { rooms.delete(room); active?.destroy(); composer.hasPendingWork = priorPending; };
}

// The room facts a call surface takes from the shell. A directly opened Project need not be in the
// bounded sidebar census, so an unknown chat id names the active project panel.
export function projectRoomOptions({ rows, nav, state, title, freshProjectRow, openProjectPanel, showPage, destroyProjectInstance }) {
    return ({ page, chatId }) => {
        const project = chatId === 1 ? null : rows().find(row => Number(row.chat_id) === chatId)
            || { id: nav.activeProjectId, chat_id: chatId, name: title.textContent };
        const current = () => project ? nav.activeProjectId === project.id : !nav.activeProjectId;
        return {
            label: () => project ? freshProjectRow(project).name || project.id : 'Main',
            onScreen: () => state.activePage === 'chat' && !page.hidden && current(),
            reveal: () => project ? openProjectPanel(freshProjectRow(project), { openOnly: true }) : showPage('chat'),
            onSettled: () => queueMicrotask(() => { if (project && !current()) destroyProjectInstance(project.id); }),
        };
    };
}
