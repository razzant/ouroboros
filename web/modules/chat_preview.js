/** Same-origin saved text for a warm opening, never history coverage or work authority.
 * One bounded browser record holds the most recently read rooms. Only successful
 * newest history reads replace it; live echoes, cursors, controls and credentials
 * never enter it. A new origin or refused storage simply opens cold.
 */
export const PREVIEW_KEY = 'ouroboros.chat-preview.v1';
const MAX_CHARS = 200000, MAX_ROOMS = 8, MAX_ROWS = 80;

export function clearChatPreviews(storage) {
    try { (storage || globalThis.localStorage)?.removeItem(PREVIEW_KEY); } catch { /* storage unavailable */ }
}

export function createChatPreview({ feed, room, scope = '', mutate, storage = () => globalThis.localStorage }) {
    const key = JSON.stringify([scope, room]);
    let root = null, disposed = false;
    const read = () => {
        try {
            const raw = storage()?.getItem(PREVIEW_KEY);
            if (!raw || raw.length > MAX_CHARS) return [];
            const data = JSON.parse(raw);
            return data?.v === 1 && Array.isArray(data.rooms) ? data.rooms : [];
        } catch { return []; }
    };
    const remove = () => { root?.remove(); root = null; };
    // A selected passage remains readable/copyable until the selection is released.
    // It stays labelled as a preview, not another current transcript.
    let retiring = false;
    const selected = () => {
        const selection = feed.ownerDocument.getSelection?.();
        if (!root || !selection || selection.isCollapsed) return false;
        for (let i = 0; i < selection.rangeCount; i++) {
            try { if (selection.getRangeAt(i).intersectsNode(root)) return true; } catch { /* detached range */ }
        }
        return false;
    };
    const retire = () => {
        if (!root) return;
        retiring = true;
        if (!selected()) mutate(() => { remove(); return true; });
        else root.firstElementChild.textContent = 'Saved text preview · retained until selection is released';
    };
    const onSelection = () => { if (retiring && !selected()) retire(); };
    feed.ownerDocument.addEventListener('selectionchange', onSelection);
    return {
        show() {
            if (disposed || root) return false;
            const rows = read().find(entry => entry?.key === key)?.rows;
            if (!Array.isArray(rows) || rows.length > MAX_ROWS || !rows.length) return false;
            if (rows.some(row => !['user', 'assistant', 'system'].includes(row?.role)
                || typeof row.text !== 'string' || typeof row.id !== 'string')) return false;
            const doc = feed.ownerDocument;
            root = doc.createElement('div');
            root.className = 'chat-saved-preview';
            root.dataset.savedPreview = 'true';
            const note = doc.createElement('div');
            note.className = 'chat-load-older-note';
            note.textContent = 'Saved text preview · updating';
            root.append(note);
            for (const row of rows) {
                const bubble = doc.createElement('div');
                bubble.className = `chat-bubble ${row.role}`;
                // Identity locates a passage after the canonical read; it grants no
                // delivered/read fact (the preview owns no pager or receipt state).
                bubble.dataset.previewHistoryId = row.id;
                const name = doc.createElement('div');
                name.className = 'chat-sender';
                name.textContent = row.role === 'user' ? 'You' : row.role === 'assistant' ? 'Ouroboros' : 'System';
                const text = doc.createElement('div');
                text.className = 'chat-text';
                text.textContent = row.text;
                bubble.append(name, text);
                root.append(bubble);
            }
            feed.prepend(root);
            return true;
        },
        failed() { if (root) root.firstElementChild.textContent = 'Saved text preview · not current'; },
        save(data) {
            if (disposed || data?.reason_code || !Array.isArray(data?.messages) || !data.page_cursor) return;
            const rows = data.messages.filter(row => row.history_id && typeof row.text === 'string' && row.text
                && ['user', 'assistant', 'system'].includes(row.role) && !row.is_progress
                && !['task_evidence', 'task_summary', 'quiz', 'card'].includes(row.system_type))
                .slice(-MAX_ROWS).map(row => ({ id: row.history_id, role: row.role, text: row.text }));
            let rooms = [{ key, rows }, ...read().filter(entry => entry?.key !== key)].slice(0, MAX_ROOMS);
            // Drop oldest rooms, then oldest rows: never cut a person's text.
            const encode = () => JSON.stringify({ v: 1, rooms });
            while (encode().length > MAX_CHARS && rooms.length > 1) rooms.pop();
            while (encode().length > MAX_CHARS && rows.length) rows.shift();
            try { storage()?.setItem(PREVIEW_KEY, encode()); } catch { /* online chat still works */ }
        },
        retire,
        destroy() {
            disposed = true;
            feed.ownerDocument.removeEventListener('selectionchange', onSelection);
            remove();
        },
    };
}
