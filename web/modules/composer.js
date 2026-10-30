// The Main and Project composer (docs/DESIGN.md "Controls and editable choices", "Composer owner controls",
// "Chat attachments"): its markup inside the room's input dock, the message field (its growth, Enter, the input
// recall and the room's draft), the room's chat model (composer_model_picker.js) and its Subagents, Context and
// Effort capsules with their readback, the "+" menu
// (Attach file first, then the sections later controls set), the voice and chips slots, the staged attachments with
// their paste and drop routes, and the send path that turns the draft into the owner's frame. Skills' "+" items and
// chips follow the live contributions (composer_plus.js, composer_chips.js). The room (chat.js) mounts the markup,
// namespaces its ids and shows in its feed what was sent.
import { bindComposerFileTargets, cleanupUploadedAttachments } from './chat_media.js';
import { composerText, createComposerAttachments } from './chat_attachments.js';
import { loadChatInputHistory, saveChatInputHistory, sendPanic } from './chat_activity.js';
import { isPanicFrame } from './panic.js';
import { clientSurfaceField } from './client_surface.js';
import { apiClient, apiFetch } from './api_client.js';
import { bindEnterSubmit, bindMenu } from './ui_interactions.js';
import { controlIcon, strokeIcon } from './page_icons.js';
import { renderMobileNavToggle } from './page_header.js';
import { createComposerOwnerControls } from './composer_owner_controls.js';
import { createComposerModelPicker } from './composer_model_picker.js';
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
    attach: strokeIcon('<path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>', 16),
};

/** The composer inside the room's `.chat-input-wrap`; chat.js mounts it and namespaces its `chat-*` ids in a Project.
 * The field (`.chat-field`) is one object: a staged reply's strip (chat_reply.js puts it before the text row) and the
 * text row with "+", the words, the mic slot and Send. */
export function composerMarkup() {
    return `
                <div id="chat-attachment-preview" class="chat-attachment-preview"></div>
                <div class="chat-toolbar-row">
                    ${renderMobileNavToggle({ inComposer: true })}
                    <div class="chat-model-picker" id="chat-model-picker"></div>
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
                        <textarea id="chat-input" placeholder="Message Ouroboros..." rows="1" autocorrect="off" autocapitalize="off" spellcheck="false"></textarea>
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
 * Settings tab (the Subagents capsule's way to Agents), `onLayout()` re-measures
 * the room's dock, `onSent({ text, frame, result, views })` shows an accepted send in the feed and returns the row it
 * added (the sent draft rises into it, send_flight.js), and `readingBack()`
 * says whether the owner's own gesture took the room off its newest row (the phone fold reads it: composer_fold.js).
 */
export function createComposer({ page, byId, ws, replies, chatSessionId, isMain, chatId, projectId, storeKey,
    refreshState, openSettings = null, onLayout, onSent, readingBack }) {
    const input = byId('input');
    const sendBtn = byId('send');
    const plusBtn = byId('plus');
    const fileInput = byId('file-input');
    let destroyed = false, cognitionRead = false;

    // Staged files, their thumbnails and object URLs, and the upload on Send ("+" waits while it runs).
    const attachments = createComposerAttachments({ preview: byId('attachment-preview'), attachBtn: plusBtn, fileInput,
        input, showToast, onLayout: () => requestAnimationFrame(() => onLayout()) });

    const plusMenu = createPlusMenu({ button: plusBtn, hostItems: [
        { key: 'attach', label: 'Attach file', icon: 'attach', onSelect: () => fileInput.click() },
    ] });

    // Local-only staging avoids orphan uploads and fast-send races.
    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files || []);
        fileInput.value = '';
        attachments.stage(files);
    });

    bindComposerFileTargets({ page, inputArea: byId('input-area'), input, stagePendingFiles: attachments.stage });

    const inputHistory = loadChatInputHistory(sessionStorage, CHAT_INPUT_HISTORY_KEY);
    let inputHistoryIndex = inputHistory.length;
    let inputDraft = '';
    let inputHistorySeededFromServer = false; // set true only after a successful server-side recall seed

    function rememberInput(text) {
        if (!text) return;
        if (inputHistory[inputHistory.length - 1] !== text) inputHistory.push(text);
        saveChatInputHistory(sessionStorage, CHAT_INPUT_HISTORY_KEY, inputHistory);
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
        saveChatInputHistory(sessionStorage, CHAT_INPUT_HISTORY_KEY, inputHistory);
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
        if (step < 0) {
            if (input.selectionStart !== 0 || input.selectionEnd !== 0) return;
            if (inputHistoryIndex === inputHistory.length) inputDraft = input.value;
            inputHistoryIndex = Math.max(0, inputHistoryIndex - 1);
            input.value = inputHistory[inputHistoryIndex] || '';
        } else {
            if (input.selectionStart !== input.value.length || input.selectionEnd !== input.value.length) return;
            inputHistoryIndex = Math.min(inputHistory.length, inputHistoryIndex + 1);
            input.value = inputHistoryIndex === inputHistory.length ? inputDraft : (inputHistory[inputHistoryIndex] || '');
        }
        resizeChatInput();
        const cursor = input.value.length;
        input.setSelectionRange(cursor, cursor);
    }

    async function sendMessage() {
        let text = input.value.trim();
        if (isPanicFrame({ type: 'chat', content: text })) {
            // Emergency input precedes settings saves, uploads and their disabled Send state.
            const frame = { type: 'chat', content: text, ...clientSurfaceField() };
            const sentDraft = input.value;
            sendPanic({ ws, frame, onAccepted: () => {
                // Only the accepted command text is consumed. Replies, files and
                // any newer edit belong to the next message.
                if (input.value !== sentDraft) return;
                if (!destroyed) {
                    input.value = inputDraft = '';
                    inputHistoryIndex = inputHistory.length;
                    resizeChatInput();
                }
                try {
                    if (sessionStorage.getItem(storeKey(CHAT_DRAFT_KEY)) === sentDraft) clearInputDraft();
                } catch {}
            } });
            return;
        }
        if (sendBtn.disabled) return;  // guard against Enter re-entry during async upload
        if (ownerControls.hasPendingSave() || modelPicker?.hasPendingSave()) {  // the room's settings land first
            setSendBusy(true, 'Saving');
            try {
                const saved = await Promise.all([ownerControls.pendingSave(), modelPicker?.hasPendingSave() ? modelPicker.pendingSave() : true]);
                if (saved.some((ok) => ok === false)) return;  // a refused save keeps the draft, said by its control
            } finally { setSendBusy(false); }
        }
        text = input.value.trim();
        const hasAttachments = attachments.count > 0;
        let uploadedAttachments = [];
        if (!text && !hasAttachments) return;
        if (hasAttachments) {
            // Upload at Send; offline queuing orphans files.
            if (ws.ws?.readyState !== WebSocket.OPEN) {
                showToast('Cannot attach file while offline. Reconnect and try again.', 'error');
                return;
            }
            setSendBusy(true, attachments.count > 1 ? 'Uploading files' : 'Uploading');
            try {
                uploadedAttachments = await attachments.upload(() => ws.ws?.readyState === WebSocket.OPEN);
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
            sender_session_id: chatSessionId,
            ...(isMain ? {} : { chat_id: chatId }),
            ...(projectId ? { project_id: projectId } : {}),
            // Structured metadata rides the frame: model staging and native image blocks.
            ...(hasAttachments ? { attachments: uploadedAttachments.map(({ filename, display_name, mime }) => ({ filename, display_name, mime })) } : {}),
            ...replies.frameField(),
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
        if (hasAttachments) attachments.clear();
        rememberInput(text);
        input.value = '';
        clearInputDraft();
        flySent({ from: draft, row: onSent({ text, frame, result, views: uploadedAttachments.map((item) => item.view) }) });
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

    // Each composer owns its room's chat model and its Subagents, Context and Effort settings.
    const pickerEl = byId('model-picker');
    const modelPicker = pickerEl ? createComposerModelPicker({ el: pickerEl, projectId, save: apiClient.ownerChatModel,
        showToast, onLayout }) : null;
    const ownerControls = createComposerOwnerControls({ row: page.querySelector('.chat-toolbar-row'), box: page, byId, apiFetch, showToast,
        projectId, saveEffortRange: apiClient.ownerEffortRange,
        saveSubagentsMode: apiClient.ownerSubagentsMode,
        fitValues: () => (modelPicker ? [modelPicker.nameEl()] : []),
        openSettings: () => openSettings?.('agents'), refreshState, onLayout, readingBack });

    // Rooms outside the bounded sidebar still read their settings.
    async function refreshProjectCognition() {
        if (destroyed || cognitionRead) return;
        cognitionRead = true;
        try {
            const room = await apiFetch(`/api/projects?chat_id=${encodeURIComponent(chatId)}`, { cache: 'no-store' });
            if (room.ok && !destroyed) {
                const data = await room.json();
                ownerControls.syncState(data);
                modelPicker?.syncState(data);
            }
        } catch { /* Preserve the last room value, never substitute Main. */ }
        finally { cognitionRead = false; }
    }

    sendBtn.addEventListener('click', () => sendMessage());
    // The send path admits Panic even while an ordinary upload/save disables Send.
    bindEnterSubmit(input, sendBtn, () => sendMessage());
    input.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowUp' && !e.shiftKey) {
            restoreInputHistory(-1);
        } else if (e.key === 'ArrowDown' && !e.shiftKey) {
            restoreInputHistory(1);
        }
    });

    // Per-thread drafts survive room closure; Send clears them.
    function saveInputDraft() {
        try {
            if (input.value) sessionStorage.setItem(storeKey(CHAT_DRAFT_KEY), input.value);
            else sessionStorage.removeItem(storeKey(CHAT_DRAFT_KEY));
        } catch {}
    }

    function clearInputDraft() {
        try { sessionStorage.removeItem(storeKey(CHAT_DRAFT_KEY)); } catch {}
    }

    try {
        const savedDraft = sessionStorage.getItem(storeKey(CHAT_DRAFT_KEY)) || '';
        if (savedDraft && !input.value) {
            input.value = savedDraft;
            inputDraft = savedDraft;
            resizeChatInput();
        }
    } catch {}

    // Typing, and every change this module makes on another's behalf, keeps the recall's draft, the height and the
    // room's draft current.
    function onInput() {
        if (inputHistoryIndex === inputHistory.length) inputDraft = input.value;
        resizeChatInput();
        saveInputDraft();
    }
    input.addEventListener('input', onInput);

    const composer = {
        input,
        // `setSections(sections)` adds items after Attach file; see createPlusMenu.
        plusMenu,
        // The room's chat model (composer_model_picker.js): its readback, its one save path and its pending save.
        modelPicker,
        // Empty and hidden until a control mounts into them: the dictation mic between the field and Send, and
        // contributed chips after the room's capsules.
        voiceSlot: page.querySelector('[data-composer-voice]'),
        chipsSlot: page.querySelector('[data-composer-chips]'),
        focus: () => input.focus({ preventScroll: true }),
        getDraft: () => input.value,
        /** Replace the draft (it is kept for the room like typed words). */
        setDraft(text) {
            input.value = String(text ?? '');
            onInput();
            input.dispatchEvent?.(new Event('change', { bubbles: true }));
        },
        /** Insert words at the caret, replacing a selection, and leave the caret after them in the focused field. */
        insertAtCursor(text) {
            const words = String(text ?? '');
            const start = input.selectionStart ?? input.value.length;
            const end = input.selectionEnd ?? start;
            input.value = input.value.slice(0, start) + words + input.value.slice(end);
            input.focus({ preventScroll: true });
            input.setSelectionRange(start + words.length, start + words.length);
            onInput();
            input.dispatchEvent?.(new Event('change', { bubbles: true }));
        },
        // After a send, once the feed shows it: the emptied field's height and the room's dock.
        resize: resizeChatInput,
        seedInputHistory,
        // The room's chat model, Subagents, Context and Effort readback, from Main's state or the named Project.
        syncState(data) {
            ownerControls.syncState(data);
            modelPicker?.syncState(data);
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
            for (const dispose of contributed) dispose();
            plusMenu.destroy();
            attachments.destroy();
            modelPicker?.destroy();
            ownerControls.destroy();
            inputSizing.dispose();
        },
    };
    const contributed = [bindComposerPlus({ composer }), bindComposerChips({ composer, menu: createPlusMenu }),
        bindVoiceControls({ composer }), bindConversationControls({ composer, page, chatId, ws })];
    return composer;
}
