// The Main and Project composer (docs/DESIGN.md "Controls and editable choices", "Composer owner controls",
// "Chat attachments"): its markup inside the room's input dock, the message field (its growth, Enter, the input
// recall and the room's draft), the room's Model, Subagents, Context and Effort capsules with their readback, the "+"
// menu (Attach file first, then the sections later controls set), the voice and chips slots, the staged attachments with
// their paste and drop routes, and the send path that turns the draft into the owner's frame. Skills' "+" items and
// chips follow the live contributions (composer_plus.js, composer_chips.js). The room (chat.js) mounts the markup,
// namespaces its ids and shows in its feed what was sent.
import { bindComposerFileTargets, cleanupUploadedAttachments } from './chat_media.js';
import { composerText, createComposerAttachments } from './chat_attachments.js';
import { loadChatInputHistory, saveChatInputHistory, sendPanic } from './chat_activity.js';
import { roomStorage, desktopRoomStorage, roomOpening, whenRoomOpens } from './desktop_room_session.js';
import { isPanicFrame } from './panic.js';
import { clientSurfaceField } from './client_surface.js';
import { apiClient, apiFetch } from './api_client.js';
import { bindEnterSubmit, bindMenu } from './ui_interactions.js';
import { controlIcon, fluentIcon } from './page_icons.js';
import { renderMobileNavToggle } from './page_header.js';
import { createComposerOwnerControls } from './composer_owner_controls.js';
import { attachChatInputSizing } from './chat_input_sizing.js';
import { showToast } from './toast.js';
import { bindComposerPlus } from './composer_plus.js';
import { bindComposerChips } from './composer_chips.js';
import { bindVoiceControls } from './voice_controls.js';
import { bindConversationControls } from './conversation_controls.js';
import { draftPlace, flySent } from './send_flight.js';

const CHAT_DRAFT_KEY = 'ouro_chat_draft';
const CHAT_INPUT_HISTORY_KEY = 'ouro_chat_input_history';
// The host icons a "+" item may name (`icon`); other names draw from the control glyphs a skill may name.
const PLUS_ICONS = {
    attach: fluentIcon('attach', 16, 'regular'),
    search: fluentIcon('search', 16, 'regular'),
};

/** The composer inside the room's `.chat-input-wrap`; chat.js mounts it and namespaces its `chat-*` ids in a Project.
 * The field (`.chat-field`) is one object: a staged reply's strip (chat_reply.js puts it before the text row) and the
 * text row with "+", the words, the mic slot and Send. */
export function composerMarkup() {
    return `
                <div id="chat-attachment-preview" class="chat-attachment-preview"></div>
                <div class="chat-toolbar-row">
                    ${renderMobileNavToggle({ inComposer: true })}
                    <div class="chat-model-mode" id="chat-model-mode"></div>
                    <div class="chat-subagents-mode" id="chat-subagents-mode"></div>
                    <div class="chat-context-mode" id="chat-context-mode" data-context-mode="max"></div>
                    <span class="chat-composer-chips" data-composer-chips hidden></span>
                </div>
                <div class="chat-field">
                    <div class="chat-text-row">
                        <button class="chat-plus-btn ui-hit-area" id="chat-plus" type="button" data-composer-plus aria-label="More actions" aria-haspopup="menu" aria-expanded="false" title="More actions">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 5v14"/><path d="M5 12h14"/></svg>
                        </button>
                        <input type="file" id="chat-file-input" class="chat-file-input-hidden" accept="*/*" multiple>
                        <textarea id="chat-input" placeholder="Message Ouroboros..." rows="1" autocorrect="on" autocapitalize="sentences" spellcheck="true"></textarea>
                        <span class="chat-voice-slot" data-composer-voice hidden></span>
                        <div class="chat-send-group">
                            <button class="chat-send-inline ui-hit-area" id="chat-send" title="Send message">Send</button>
                        </div>
                    </div>
                </div>`;
}

const keepFocus = (event) => event.preventDefault();

/**
 * The composer's "+" menu (docs/DESIGN.md "Composer owner controls"): `hostItems` first (Attach file), then the
 * sections later controls set, each under its label. Items are `{key, label, title, icon, disabled, onSelect}`.
 * The shared menu sheet outside every clipping parent: Enter or Space on "+" opens it with focus on the first item,
 * arrows move, Esc closes and returns focus. A pointer (a tap, a click) from elsewhere opens it and chooses from it
 * without taking the focus, so a phone typing in the field keeps its keyboard and the composer stays under the menu.
 */
export function createPlusMenu({ button, hostItems = [] }) {
    const doc = button.ownerDocument;
    let sections = [];
    let current = null;  // { menu, binding }

    function item(spec) {
        const node = doc.createElement('button');
        node.type = 'button';
        node.className = 'chat-plus-item';
        node.setAttribute('role', 'menuitem');
        node.dataset.plusItem = String(spec.key || '');
        if (spec.title) node.title = String(spec.title);
        node.disabled = Boolean(spec.disabled);
        if (spec.icon) {
            const icon = doc.createElement('span');
            icon.className = 'chat-plus-icon';
            icon.setAttribute('aria-hidden', 'true');
            icon.innerHTML = PLUS_ICONS[spec.icon] || controlIcon(spec.icon, 16);
            node.append(icon);
        }
        const label = doc.createElement('span');
        label.textContent = String(spec.label || '');
        node.append(label);
        node.addEventListener('mousedown', keepFocus);
        node.addEventListener('click', (event) => {
            event.stopPropagation();
            close();
            spec.onSelect?.();
        });
        return node;
    }

    function group(section) {
        const node = doc.createElement('div');
        node.className = 'chat-plus-group';
        node.setAttribute('role', 'group');
        node.dataset.plusSection = String(section.key || '');
        if (section.label) {
            node.setAttribute('aria-label', String(section.label));
            const heading = doc.createElement('div');
            heading.className = 'chat-plus-label';
            heading.setAttribute('aria-hidden', 'true');
            heading.textContent = String(section.label);
            node.append(heading);
        }
        node.append(...section.items.filter(Boolean).map(item));
        return node;
    }

    function open({ focus = true } = {}) {
        if (current) return;
        const menu = doc.createElement('div');
        menu.className = 'ui-popup chat-plus-menu';
        menu.setAttribute('role', 'menu');
        menu.setAttribute('aria-label', button.getAttribute('aria-label') || 'More actions');
        menu.append(...hostItems.map(item),
            ...sections.filter((section) => section?.items?.length).map(group));
        doc.body.append(menu);
        button.setAttribute('aria-expanded', 'true');
        // Opened by a pointer, focus stays in the field: Esc still closes the menu, and only the menu.
        const onKey = (event) => {
            if (event.key !== 'Escape' || menu.contains(doc.activeElement)) return;
            event.preventDefault();
            event.stopPropagation();
            binding.close({ reason: 'escape' });
        };
        const binding = bindMenu(menu, { anchor: button, align: 'start', initialFocus: focus, onClose: () => {
            doc.removeEventListener('keydown', onKey, true);
            menu.remove();
            button.setAttribute('aria-expanded', 'false');
            if (current?.menu === menu) current = null;
        } });
        doc.addEventListener('keydown', onKey, true);
        current = { menu, binding };
    }

    function close() {
        current?.binding.close({ restoreFocus: true, reason: 'action' });
    }

    button.addEventListener('mousedown', keepFocus);
    // A keyboard press (Enter, Space) is a click without a pointer count; it, or a press on a focused "+", moves
    // focus into the menu.
    button.addEventListener('click', (event) => (current ? close()
        : open({ focus: !event.detail || doc.activeElement === button })));

    return {
        /** Replace the sections after the host's items: `[{key, label, items: [{key, label, title, icon, disabled,
         *  onSelect}]}]`. The next opening shows them; an open menu keeps the list it opened with. */
        setSections(next) {
            sections = Array.isArray(next) ? next : [];
        },
        open,
        close,
        destroy: close,
    };
}

/**
 * Wires one room's composer once chat.js has mounted `composerMarkup()` and namespaced its ids. `byId(suffix)`
 * resolves a namespaced id; `page` is the room (the drop target and the capsules' docking box); `ws` the shared
 * socket; `replies` the room's reply controller, whose reference rides the frame; `storeKey(base)` the room's
 * sessionStorage key. `refreshState()` re-reads the room's settings after a capsule saves, `openSettings(tab)` opens a
 * Settings tab (the Subagents capsule's way to Agents, the Model capsule's to Accounts), `onLayout()` re-measures
 * the room's dock, `onSent({ text, frame, result, views, reply })` shows an accepted send in the feed (`reply` is the echo
 * of the reply the click captured, its staged reference itself: the room consumes that one, never a newer staging) and
 * returns the row it added (the sent draft rises into it, send_flight.js), and `readingBack()`
 * says whether the owner's own gesture took the room off its newest row (the phone fold reads it: composer_fold.js).
 */
export function createComposer({ page, byId, ws, replies, chatSessionId, getSessionId = () => chatSessionId, isMain, chatId, projectId, storeKey,
    refreshState, openSettings = null, onLayout, onSent, readingBack }) {
    const input = byId('input');
    const storage = roomStorage();
    const sendBtn = byId('send');
    const plusBtn = byId('plus');
    const fileInput = byId('file-input');
    let destroyed = false, cognitionRead = false;
    const session = desktopRoomStorage();
    const editable = () => input.dataset.desktopRoomOpening !== '1';
    let releaseInput = () => {};
    if (session?.inputLocked()) {
        const hint = input.placeholder;
        input.dataset.desktopRoomOpening = '1';
        input.placeholder = 'Restoring the room… Choose Main or a Project to type.';
        input.readOnly = sendBtn.disabled = plusBtn.disabled = fileInput.disabled = true;
        releaseInput = session.onInputReady(() => {
            delete input.dataset.desktopRoomOpening;
            input.placeholder = hint;
            input.readOnly = sendBtn.disabled = plusBtn.disabled = fileInput.disabled = false;
        });
    }

    // Staged files, their thumbnails and object URLs, and the upload on Send ("+" waits while it runs).
    const attachments = createComposerAttachments({ preview: byId('attachment-preview'), attachBtn: plusBtn, fileInput,
        input, showToast, fileCountStore: session, fileCountKey: storeKey('ouro_chat_files'), onLayout: () => requestAnimationFrame(() => onLayout()) });
    const stageFiles = files => { if (editable()) attachments.stage(files); };

    // Search this chat is the pointer's way to the room's search (Ctrl+F or Cmd+F on a keyboard; chat_search.js).
    const plusMenu = createPlusMenu({ button: plusBtn, hostItems: [
        { key: 'attach', label: 'Attach file', icon: 'attach', onSelect: () => fileInput.click() },
        ...(replies?.search ? [{ key: 'search', label: 'Search this chat', icon: 'search', onSelect: () => replies.search.open() }] : []),
    ] });

    // Local-only staging avoids orphan uploads and fast-send races.
    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files || []);
        fileInput.value = '';
        stageFiles(files);
    });

    bindComposerFileTargets({ page, inputArea: byId('input-area'), input, stagePendingFiles: stageFiles });

    const inputHistory = [];
    let inputHistoryIndex = inputHistory.length;
    let inputDraft = '';
    let inputHistorySeededFromServer = false; // set true only after a successful server-side recall seed
    // The Send click in flight, followed through its saves and uploads (`sendMessage`): the words it read and whether
    // the owner has since rewritten them rather than only added words after them. An edit that began inside them, or
    // replaced the whole field (a selection typed over, a recalled history entry, a replaced draft), rewrites them for
    // good, as does a field that no longer begins with them: whatever the field then holds, even words that share the
    // click's beginning or are its very words, is the owner's new draft. The lineage decides, never the prefix alone.
    // Typing, correcting or deleting only after the click's words never counts. `from` is where the edit began.
    let followed = null;
    function rewriting(from = 0) {
        if (followed && (from < followed.raw.length || !input.value.startsWith(followed.raw))) followed.rewritten = true;
    }

    function rememberInput(text) {
        if (!text) return;
        if (inputHistory[inputHistory.length - 1] !== text) inputHistory.push(text);
        saveChatInputHistory(storage, CHAT_INPUT_HISTORY_KEY, inputHistory);
        inputHistoryIndex = inputHistory.length;
        inputDraft = '';
    }

    // The first successful history read seeds the recall with the room's own sent words.
    function seedInputHistory(messages) {
        if (inputHistorySeededFromServer) return;
        const serverTexts = [];
        for (const msg of messages) {
            if (msg.role !== 'user') continue;
            let text = (msg.text || '').trim();
            if (text) serverTexts.push(text);
        }
        const deduped = [...new Set([...serverTexts, ...inputHistory].reverse())].reverse();
        inputHistory.length = 0;
        inputHistory.push(...deduped.slice(-50));
        saveChatInputHistory(storage, CHAT_INPUT_HISTORY_KEY, inputHistory);
        inputHistoryIndex = inputHistory.length;
        inputHistorySeededFromServer = true;
    }

    // Text height is capped by chat_input_sizing.js.
    const inputSizing = attachChatInputSizing(input);
    function resizeChatInput() {
        inputSizing.sync();
        onLayout();
    }

    function restoreInputHistory(step) {
        if (!inputHistory.length) return;
        const edge = step < 0 ? 0 : input.value.length;
        if (input.selectionStart !== edge || input.selectionEnd !== edge) return;
        const index = step < 0 ? Math.max(0, inputHistoryIndex - 1) : Math.min(inputHistory.length, inputHistoryIndex + 1);
        const draft = step < 0 && inputHistoryIndex === inputHistory.length ? input.value : inputDraft;
        const entry = index === inputHistory.length ? draft : (inputHistory[index] || '');
        // An arrow at the end of the recall (ArrowDown on the newest draft) moves to no other entry and changes no
        // word: the field is as the owner left it, and a Send still out keeps consuming the words it read.
        if (index !== inputHistoryIndex || entry !== input.value) {
            inputDraft = draft;
            inputHistoryIndex = index;
            input.value = entry;
            rewriting();  // a recalled entry replaces the whole field
            resizeChatInput();
            saveInputDraft();
        }
        const cursor = input.value.length;
        input.setSelectionRange(cursor, cursor);
    }

    async function sendMessage() {
        // One click owns the field as it stands now: the words it sends, the reply it answers and the files it
        // uploads are read here, once. A settings save or an upload that follows never re-reads them, so an edit
        // made while they run stays for the next message instead of riding this one or being lost.
        const sentRaw = input.value;
        const text = sentRaw.trim();
        if (isPanicFrame({ type: 'chat', content: text })) {
            // Emergency input precedes settings saves, uploads and their disabled Send state.
            const frame = { type: 'chat', content: text, ...clientSurfaceField() };
            sendPanic({ ws, frame, onAccepted: () => {
                // Only the accepted command text is consumed. Replies, files and
                // any newer edit belong to the next message.
                if (input.value !== sentRaw) return;
                if (!destroyed) {
                    input.value = inputDraft = '';
                    inputHistoryIndex = inputHistory.length;
                    resizeChatInput();
                }
                try {
                    if (storage.getItem(storeKey(CHAT_DRAFT_KEY)) === sentRaw) clearInputDraft();
                } catch {}
            } });
            return;
        }
        if (!editable() || roomOpening()) return;
        if (sendBtn.disabled) return;  // guard against Enter re-entry during async upload
        const click = { raw: sentRaw, text, rewritten: false, reply: replies.frameField(),
            echo: replies.echoRow(), files: attachments.snapshot() };
        followed = click;
        try { await deliver(click); } finally { if (followed === click) followed = null; }
    }

    async function deliver(click) {
        let { text } = click;
        if (ownerControls.hasPendingSave()) {  // the room's settings land before the next message
            setSendBusy(true, 'Saving');
            try { if (!(await ownerControls.pendingSave())) return; } catch { return; } finally { setSendBusy(false); }
        }
        // The click's files the owner still has staged: one taken out while the settings saved is not sent, and one
        // staged since is the next message's.
        const files = attachments.carried(click.files);
        const hasAttachments = files.length > 0;
        let uploadedAttachments = [];
        if (!text && !hasAttachments) return;
        if (hasAttachments) {
            // Upload at Send; offline queuing orphans files.
            if (ws.ws?.readyState !== WebSocket.OPEN) {
                showToast('Cannot attach file while offline. Reconnect and try again.', 'error');
                return;
            }
            setSendBusy(true, files.length > 1 ? 'Uploading files' : 'Uploading');
            try {
                uploadedAttachments = await attachments.upload(() => ws.ws?.readyState === WebSocket.OPEN, files);
            } catch (e) {
                await cleanupUploadedAttachments(e.uploaded || []);
                showToast('Upload error: ' + e.message, 'error');
                return;  // staged files and their previews remain so the user can retry
            } finally {
                setSendBusy(false);
            }
            // Model sees file names; /restart stays exact, bubbles use server views.
            text = composerText(text, uploadedAttachments.map((item) => item.view?.name || item.display_name));
        }
        if (!text) return;
        const frame = {
            type: 'chat',
            content: text,
            sender_session_id: getSessionId(),
            ...(isMain ? {} : { chat_id: chatId }),
            ...(projectId ? { project_id: projectId } : {}),
            // Structured metadata rides the frame: model staging and native image blocks.
            ...(hasAttachments ? { attachments: uploadedAttachments.map(({ filename, display_name, mime }) => ({ filename, display_name, mime })) } : {}),
            ...click.reply,
            ...clientSurfaceField(),
        };
        const result = ws.send(frame, hasAttachments ? { queue: false } : undefined);
        if (hasAttachments && result?.status !== 'sent') {
            await cleanupUploadedAttachments(uploadedAttachments);
            showToast('Connection lost before send. Reconnect and try again.', 'error');
            return;
        }
        // Where the draft stands, read before the field empties: the sent bubble rises from here (send_flight.js).
        const draft = draftPlace(input.closest('.chat-field'), input);
        attachments.consume(click.files);
        rememberInput(text);
        keepRest(click);
        flySent({ from: draft, row: onSent({ text, frame, result, views: uploadedAttachments.map((item) => item.view), reply: click.echo }) });
    }

    // Consume only what this click sent. Words added at the end while it saved or uploaded stay for the next message;
    // a rewritten field stays whole, even when it shares the click's beginning or equals its words: the message went
    // out as the click read it. The room's kept draft follows only while it still is this field's own copy: a
    // newer one (another window's, a reopened room's) is not this click's to clear or overwrite.
    function keepRest(click) {
        const now = input.value;
        const rest = !click.rewritten && now.startsWith(click.raw) ? now.slice(click.raw.length) : now;
        if (!destroyed) { input.value = inputDraft = rest; resizeChatInput(); }
        try {
            if ((storage.getItem(storeKey(CHAT_DRAFT_KEY)) ?? '') !== now) return;
            if (rest) storage.setItem(storeKey(CHAT_DRAFT_KEY), rest); else clearInputDraft();
        } catch {}
    }

    // Send mode lives on DOM so CSS and click/Enter share one source.
    const sendGroup = page.querySelector('.chat-send-group');

    function setSendBusy(busy, label = '') {
        sendGroup.dataset.busy = busy ? '1' : '0';
        sendBtn.disabled = busy;
        if (busy) {
            sendBtn.textContent = label || 'Sending';
            sendBtn.title = label || 'Sending';
        } else {
            sendBtn.textContent = 'Send';
            sendBtn.title = 'Send message';
        }
    }

    // Each composer owns its room's Model, Subagents, Context and Effort settings.
    const ownerControls = createComposerOwnerControls({ row: page.querySelector('.chat-toolbar-row'), box: page, byId, apiFetch, showToast,
        projectId, chatId, saveEffortRange: apiClient.ownerEffortRange,
        saveSubagentsMode: apiClient.ownerSubagentsMode, saveChatModel: apiClient.ownerChatModel,
        openSettings: (tab) => openSettings?.(tab), refreshState, onLayout, readingBack });

    // Rooms outside the bounded sidebar still read their settings.
    async function refreshProjectCognition() {
        if (destroyed || cognitionRead) return;
        cognitionRead = true;
        try {
            const room = await apiFetch(`/api/projects?chat_id=${encodeURIComponent(chatId)}`, { cache: 'no-store' });
            if (room.ok && !destroyed) ownerControls.syncState(await room.json());
        } catch { /* Preserve the last room value, never substitute Main. */ }
        finally { cognitionRead = false; }
    }

    const submit = () => {
        const work = sendMessage();
        desktopRoomStorage()?.trackSend(work);
        return work;
    };
    sendBtn.addEventListener('click', submit);
    // The send path admits Panic even while an ordinary upload/save disables Send.
    bindEnterSubmit(input, sendBtn, submit);
    input.addEventListener('keydown', (e) => {
        if (!editable()) return;
        if (e.key === 'ArrowUp' && !e.shiftKey) {
            restoreInputHistory(-1);
        } else if (e.key === 'ArrowDown' && !e.shiftKey) {
            restoreInputHistory(1);
        }
    });

    // Per-thread drafts survive room closure; Send clears them.
    function saveInputDraft() {
        if (roomOpening()) return;
        try {
            if (input.value) storage.setItem(storeKey(CHAT_DRAFT_KEY), input.value);
            else storage.removeItem(storeKey(CHAT_DRAFT_KEY));
        } catch {}
    }

    function clearInputDraft() {
        try { storage.removeItem(storeKey(CHAT_DRAFT_KEY)); } catch {}
    }

    whenRoomOpens(() => {
        if (destroyed) return;
        inputHistory.push(...loadChatInputHistory(storage, CHAT_INPUT_HISTORY_KEY));
        inputHistoryIndex = inputHistory.length;
        try {
            const saved = storage.getItem(storeKey(CHAT_DRAFT_KEY)) || '';
            // Desktop restores this room's slots before unlocking ordinary input.
            if (saved) { input.value = saved + input.value; inputDraft = input.value; resizeChatInput(); }
            if (input.value && input.value !== saved) saveInputDraft();
            // The restored draft is a field change like any programmatic one: watchers that did not
            // type see it too, so Send's reserved Talk face yields for the room's kept words.
            if (saved) input.dispatchEvent?.(new Event('input', { bubbles: true }));
        } catch {}
    });

    // Typing, and every change this module makes on another's behalf, keeps the recall's draft, the height and the
    // room's draft current.
    function onInput() {
        if (!editable()) return;
        rewriting(Infinity);
        if (inputHistoryIndex === inputHistory.length) inputDraft = input.value;
        resizeChatInput();
        saveInputDraft();
    }
    input.addEventListener('input', onInput);
    // An edit begins where the selection does, even when what replaces the words begins with the same ones.
    input.addEventListener('beforeinput', () => rewriting(Math.min(input.selectionStart ?? 0, input.selectionEnd ?? 0)));

    const composer = {
        input,
        // `setSections(sections)` adds items after Attach file; see createPlusMenu.
        plusMenu,
        // Empty and hidden until a control mounts into them: the dictation mic between the field and Send, and
        // contributed chips after the room's capsules.
        voiceSlot: page.querySelector('[data-composer-voice]'),
        chipsSlot: page.querySelector('[data-composer-chips]'),
        focus: () => input.focus({ preventScroll: true }),
        getDraft: () => input.value,
        /** Replace the draft (it is kept for the room like typed words). */
        setDraft(text) {
            if (!editable()) return false;
            input.value = String(text ?? '');
            rewriting();
            onInput();
            input.dispatchEvent?.(new Event('change', { bubbles: true }));
        },
        /** Insert words at the caret, replacing a selection, and leave the caret after them in the focused field. */
        insertAtCursor(text) {
            if (!editable()) return false;
            const words = String(text ?? '');
            const start = input.selectionStart ?? input.value.length;
            const end = input.selectionEnd ?? start;
            input.value = input.value.slice(0, start) + words + input.value.slice(end);
            input.focus({ preventScroll: true });
            input.setSelectionRange(start + words.length, start + words.length);
            rewriting(start);
            onInput();
            input.dispatchEvent?.(new Event('change', { bubbles: true }));
        },
        // After a send, once the feed shows it: the emptied field's height and the room's dock.
        resize: resizeChatInput,
        seedInputHistory,
        // The room's Model, Subagents, Context and Effort readback, from Main's state or the named Project.
        syncState(data) {
            ownerControls.syncState(data);
            if (projectId && !data?.projects?.some(project => project.id === projectId && project.cognition)) void refreshProjectCognition();
        },
        // A brief composer brighten when a task leaves the main chat for a project — a calm "you're free to start
        // something else" signal (P3). The CSS animation owns the duration; the class leaves with it, no timer.
        signalFreed() {
            const field = page.querySelector('.chat-field');
            field?.classList.add('chat-freed');
            field?.addEventListener('animationend', function done(event) {
                if (event.target !== field) return;  // a child's animation is not this cue
                field.classList.remove('chat-freed');
                field.removeEventListener('animationend', done);
            });
        },
        // Unsendable client-side state: staged File objects or an upload in flight.
        hasPendingWork: () => attachments.count > 0 || attachments.busy,
        destroy() {
            destroyed = true;
            releaseInput();
            for (const dispose of contributed) dispose();
            plusMenu.destroy();
            attachments.destroy();
            ownerControls.destroy();
            inputSizing.dispose();
        },
    };
    const contributed = [bindComposerPlus({ composer }), bindComposerChips({ composer, menu: createPlusMenu }),
        bindVoiceControls({ composer }), bindConversationControls({ composer, page, chatId, ws: ws.socket || ws })];  // a call keeps its frames in a room away
    return composer;
}
