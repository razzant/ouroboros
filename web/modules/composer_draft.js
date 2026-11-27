// The cold selector names the last ordinary write. Displacement drops its old body, not the
// word-free proof that a tab's revision is still unsent. Send/clear remove that exact proof and
// body, never the selector. Displaced unsent drafts of closed tabs can leave a proof without words.
// Open fields are independent; tab-only text is not reconciled across windows or storage failures.
import { isPanicFrame } from './panic.js';

const tabOnly = text => isPanicFrame({ type: 'chat', content: text });
const read = (storage, key) => { try { return storage().getItem(key); } catch { return undefined; } };
const remove = (storage, key) => { try { storage().removeItem(key); } catch {} };
const write = (storage, key, value) => {
    try { storage().setItem(key, value); return true; } catch { return false; }
};
const nonce = () => globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;

export function createSavedDraft({ storeKey, durable,
    local = () => globalThis.localStorage, session = () => globalThis.sessionStorage }) {
    const legacy = storeKey('ouro_chat_draft');
    const head = storeKey('ouro_chat_text_v1');
    const body = id => `${head}:edit:${id}`;
    const live = id => `${head}:live:${id}`;
    let revision = null;
    let ownRevision = false;

    function tabRecord() {
        try { return JSON.parse(read(session, head)); } catch { return null; }
    }

    function release(id) {
        if (id) {
            remove(local, live(id));
            remove(local, body(id));
            if (tabRecord()?.id === id) remove(session, head);
        }
        if (revision === id) {
            revision = null;
            ownRevision = false;
        }
    }

    function clearTab(expected) {
        if (expected !== undefined && read(session, legacy) !== expected) return;
        remove(session, legacy);
        remove(session, head);
    }

    function restore({ ordinary = true } = {}) {
        const oldText = read(session, legacy) ?? null;
        if (!durable || (oldText && tabOnly(oldText))) return oldText;
        const selected = read(local, head);
        const cold = selected && read(local, body(selected));
        if (oldText) {
            // The original buffer remains if publication fails. An unreadable store is not empty.
            if (ordinary && cold === null) publish(oldText);
            return oldText;
        }
        const tab = tabRecord();
        if (typeof tab?.text === 'string' && tab.text && !tabOnly(tab.text)) {
            if (tab.id === null) return tab.text;
            if (typeof tab.id === 'string') {
                const liveState = read(local, live(tab.id));
                if (liveState === '1') {
                    revision = tab.id;
                    ownRevision = true;
                    return tab.text;
                }
                if (liveState === null) remove(session, head);
            }
        }
        if (ordinary && cold && read(local, live(selected)) === '1' && !tabOnly(cold)) {
            revision = selected;
            ownRevision = false;
            return cold;
        }
        return null;
    }

    function keepInTab(text) {
        revision = null;
        ownRevision = false;
        if (write(session, head, JSON.stringify({ id: null, text }))) remove(session, legacy);
    }

    function fork(text) {
        // A real quote/file change on borrowed words is a separate tab composition, not a cold write.
        if (revision && !ownRevision && text && !tabOnly(text)) keepInTab(text);
    }

    function save(text) {
        if (!text) {
            release(revision);
            clearTab();
            return;
        }
        if (!durable || tabOnly(text)) {
            if (write(session, legacy, text)) {
                // A tab-only replacement can supersede this tab's edit, not the cold draft it read.
                if (ownRevision) release(revision);
                revision = null;
                remove(session, head);
            }
            return;
        }
        if (!publish(text)) keepInTab(text);
    }

    function publish(text) {
        const previous = read(local, head), id = nonce();
        if (!write(local, body(id), text)) return false;
        if (!write(local, live(id), '1') || !write(local, head, id)) {
            remove(local, body(id));
            remove(local, live(id));
            return false;
        }
        // Replace this tab's own edit. Another tab's exact revision has merely been displaced.
        if (ownRevision) release(revision);
        if (previous) remove(local, body(previous));
        revision = id;
        ownRevision = true;
        remove(session, legacy);
        if (!write(session, head, JSON.stringify({ id, text }))) remove(session, head);
        return true;
    }

    return { restore, save, fork, clearTab, release, revision: () => revision };
}
