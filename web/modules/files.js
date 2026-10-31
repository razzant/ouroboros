import { bindPageHeader, renderPageHeader } from './page_header.js';
import { SHELL_ICONS } from './page_icons.js';
import { bindScrollFade } from './scroll_fade.js';
import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { fetchJson, jsonPost } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { downloadViaHostBridge, setInlineStatus } from './ui_helpers.js';
import { bindMenu } from './ui_interactions.js';

function formatFileSize(size) {
    const num = Number(size);
    if (!Number.isFinite(num) || num < 0) return '';
    if (num < 1024) return `${num} B`;
    if (num < 1024 * 1024) return `${(num / 1024).toFixed(1)} KB`;
    return `${(num / (1024 * 1024)).toFixed(1)} MB`;
}

function iconForEntry(entry) {
    return entry.type === 'dir' ? '▸' : '•';
}

function defaultDirectoryMeta() {
    return 'Browse folders, preview/edit text files, upload, download, copy, and move files here. This is a file manager, not a chat attachment picker.';
}

function defaultDirectoryContent() {
    return 'Open a folder or file from the left panel to browse, preview, or edit its contents.';
}

export function initFiles({ state: appState, setBeforePageLeave, layers = null } = {}) {
    const page = document.createElement('div');
    page.id = 'page-files';
    page.className = 'page app-page-glass';
    page.innerHTML = `
        ${renderPageHeader({
            page: 'files',
            actionsHtml: '<button class="btn btn-default" id="files-refresh">Refresh</button>',
        })}
        <div class="files-layout" data-files-view="list">
            <section class="files-sidebar">
                <div class="files-toolbar">
                    <input id="files-search" class="ui-control" name="files-filter" type="text" aria-label="Filter current folder" placeholder="Filter current folder...">
                </div>
                <div class="files-browser-header">
                    <div id="files-breadcrumb" class="files-breadcrumb"></div>
                    <div class="files-browser-actions">
                        <button class="btn btn-default" id="files-paste" title="Paste copied or moved item" hidden>Paste</button>
                        <button class="btn btn-default" id="files-new-file" title="Create file">+ File</button>
                        <button class="btn btn-default" id="files-new-dir" title="Create directory">+ Dir</button>
                    </div>
                </div>
                <div id="files-list" class="files-list scroll-fade-y" tabindex="0" aria-label="Files in current folder"></div>
            </section>
            <section class="files-preview">
                <div class="files-preview-back">
                    <button type="button" class="btn btn-default files-back" data-files-back>Back to folder</button>
                </div>
                <div class="files-preview-header">
                    <div>
                        <div id="files-preview-path" class="files-preview-path">Files</div>
                        <div id="files-preview-meta" class="files-preview-meta">${defaultDirectoryMeta()}</div>
                    </div>
                    <div class="files-preview-actions">
                        <button class="btn btn-default" id="files-download" hidden>Download</button>
                        <button class="btn btn-default" id="files-open-external" hidden>Open externally</button>
                        <button class="btn btn-primary" id="files-save" hidden disabled>Save</button>
                    </div>
                </div>
                <div id="files-preview-status" class="ui-status files-preview-status" role="status" aria-live="polite" hidden></div>
                <div id="files-preview-content" class="files-preview-content scroll-fade-y">${defaultDirectoryContent()}</div>
            </section>
            <div class="files-drop-overlay" aria-hidden="true">
                <div class="files-drop-card">Drop files to upload into the current folder</div>
            </div>
            <div id="files-context-menu" class="files-context-menu ui-popup" role="menu" aria-label="File actions" hidden>
                <button type="button" role="menuitem" class="files-context-item" data-action="download">Download</button>
                <button type="button" role="menuitem" class="files-context-item" data-action="copy">Copy</button>
                <button type="button" role="menuitem" class="files-context-item" data-action="move">Move</button>
                <button type="button" role="menuitem" class="files-context-item" data-action="paste">Paste Here</button>
                <button type="button" role="menuitem" class="files-context-item files-context-item-danger" data-action="delete">Delete</button>
            </div>
        </div>
    `;
    document.getElementById('content').appendChild(page);

    const layoutEl = page.querySelector('.files-layout');
    const listEl = page.querySelector('#files-list');
    const breadcrumbEl = page.querySelector('#files-breadcrumb');
    const previewPathEl = page.querySelector('#files-preview-path');
    const previewMetaEl = page.querySelector('#files-preview-meta');
    const previewContentEl = page.querySelector('#files-preview-content');
    const previewStatusEl = page.querySelector('#files-preview-status');
    const contextMenuEl = page.querySelector('#files-context-menu');
    let contextMenuBinding = null;
    let contextAnchor = null;
    const saveBtn = page.querySelector('#files-save');
    const downloadBtn = page.querySelector('#files-download');
    const openExternalBtn = page.querySelector('#files-open-external');
    const pasteBtn = page.querySelector('#files-paste');
    const newFileBtn = page.querySelector('#files-new-file');
    const newDirBtn = page.querySelector('#files-new-dir');
    const searchEl = page.querySelector('#files-search');
    const refreshBtn = page.querySelector('#files-refresh');
    const sidebarEl = page.querySelector('.files-sidebar');
    const previewEl = page.querySelector('.files-preview');
    const backBtn = page.querySelector('[data-files-back]');
    backBtn.insertAdjacentHTML('afterbegin', `<span aria-hidden="true">${SHELL_ICONS.back}</span>`);
    // On a desk the folder's location starts the bar and its own actions join Refresh at the end; in the stack
    // layout they stay over the list they act on (docs/DESIGN.md "Page header"). In the bar a long location scrolls
    // inside it: it rests at its end, the folder you are in, after each path and whenever the bar gives it another
    // width (the page shown, the layout switched, the window resized), and a folder the keyboard reaches comes
    // wholly into view, where an engine would stop at partly. Only the strip scrolls. Files lives as long as the app
    // (app.js builds it once and keeps no handle; its window and document listeners below have no destroy either),
    // so these bindings live as long and keep no disposer.
    bindPageHeader(page.querySelector('.app-page-header'), {
        lead: breadcrumbEl, actions: page.querySelector('.files-browser-actions') });
    const revealFolder = (crumb) => {
        if (!crumb || !breadcrumbEl.clientWidth) return;
        const view = breadcrumbEl.getBoundingClientRect();
        const box = crumb.getBoundingClientRect();
        const inset = Math.max(0, Math.min(parseFloat(getComputedStyle(breadcrumbEl).scrollPaddingInlineStart) || 0,
            (view.width - box.width) / 2));
        if (box.left < view.left + inset) breadcrumbEl.scrollLeft -= view.left + inset - box.left;
        else if (box.right > view.right - inset) breadcrumbEl.scrollLeft += box.right - view.right + inset;
    };
    const revealCurrentFolder = () => revealFolder(breadcrumbEl.lastElementChild);
    bindScrollFade(breadcrumbEl, { axis: 'x' });
    if (typeof ResizeObserver === 'function') new ResizeObserver(revealCurrentFolder).observe(breadcrumbEl);
    breadcrumbEl.addEventListener('focusin', (event) => revealFolder(event.target.closest?.('.files-crumb')));

    const state = {
        path: '.',
        parentPath: '.',
        entries: [],
        selectedPath: '',
        selectedType: '',
        filter: '',
        rootPath: '',
        dragDepth: 0,
        contextPath: '',
        editorPath: '',
        editorOriginalFilename: '',
        editorOriginal: '',
        editorValue: '',
        editorDirty: false,
        editorWritable: false,
        editorIsNew: false,
        editorFilename: '',
        editorSaving: false,
        clipboard: null,
        contextEntryType: '',
        contextDestinationPath: '.',
    };
    let viewRequest = 0;
    let draftRevision = 0;

    // A page too narrow for both (style.css, the `files` container) shows the folder's list first and the chosen file
    // once one is chosen, never both: `data-files-view` says which. Back to folder returns to the list and keeps an
    // unsaved draft, which the file's row (or a new file's own row) reopens and which leaving the file still asks
    // about. Side by side the view changes nothing on screen; focus moves there only off a control a resize hid.
    // The file shown alone is one layer of the shell's stack (layers.js, DESIGN "Layers and dismissal"), so Escape,
    // the system Back and Telegram's Back close it exactly as Back to folder does; beside the list it is no layer.
    const FILE_LAYER = 'files-file';
    let listScroll = 0;
    let unsavedMark = '';
    const panesShown = () => [sidebarEl, previewEl].filter((pane) => pane.getClientRects().length > 0).length;
    const singlePane = () => panesShown() === 1;
    const unsavedPath = () => (state.editorDirty && !state.editorIsNew ? state.editorPath : '');

    // On the list alone, the folder's status (a failed read, a copy waiting for Paste, an upload) stands over the list
    // it is about; beside the file it stays under the file's header.
    function placeStatus() {
        const home = singlePane() && layoutEl.dataset.filesView !== 'file' ? listEl : previewContentEl;
        if (previewStatusEl.nextElementSibling !== home) home.before(previewStatusEl);
    }

    // What the person last focused inside Files: by the time a resize is observed an engine may already have moved
    // focus from a control the new layout hid to the page's body.
    let lastFocus = null;
    layoutEl.addEventListener('focusin', (event) => { lastFocus = event.target; });

    function showPane(view, focusTarget = null) {
        const focused = document.activeElement;
        const active = focused === document.body && lastFocus?.isConnected ? lastFocus : focused;
        const changed = layoutEl.dataset.filesView !== view;
        layoutEl.dataset.filesView = view;
        placeStatus();
        const single = singlePane();
        syncLayer(single && view === 'file');
        if (single && changed && view === 'list') listEl.scrollTop = listScroll;
        // Focus never stays on what just left the screen (the pane put away, or Back once both panes fit): it goes
        // to Back on a file shown alone, otherwise to the file's row.
        const lost = layoutEl.contains(active) && !active.getClientRects().length;
        if (!(single && focusTarget) && !lost) return;
        const row = [...listEl.querySelectorAll('.files-entry.selected')].find((node) => node.getClientRects().length);
        const target = (single && focusTarget) || (single && view === 'file' ? backBtn : row || listEl);
        target.focus({ preventScroll: true });
    }

    // The row of the file whose edits are not saved says so while only the list is in sight (style.css).
    function markUnsaved() {
        const path = unsavedPath();
        if (path === unsavedMark) return;
        unsavedMark = path;
        listEl.querySelectorAll('.files-entry[data-path]').forEach((row) => {
            row.toggleAttribute('data-unsaved', Boolean(path) && row.dataset.path === path);
        });
    }

    function updateEditorActions() {
        const visible = state.editorWritable && state.selectedType === 'file';
        const canSave = visible && (
            state.editorIsNew
                ? Boolean(state.editorFilename.trim())
                : state.selectedPath === state.editorPath
        ) && (state.editorDirty || state.editorIsNew);
        saveBtn.hidden = !visible;
        saveBtn.disabled = !canSave || state.editorSaving;
        saveBtn.textContent = state.editorSaving ? 'Saving…' : 'Save';
        const fileSelected = state.selectedType === 'file' && Boolean(state.selectedPath);
        downloadBtn.hidden = !fileSelected;
        openExternalBtn.hidden = !fileSelected;
        markUnsaved();
    }

    function resetEditorState() {
        state.editorPath = '';
        state.editorOriginalFilename = '';
        state.editorOriginal = '';
        state.editorValue = '';
        state.editorDirty = false;
        state.editorWritable = false;
        state.editorIsNew = false;
        state.editorFilename = '';
        updateEditorActions();
    }

    function updateClipboardActions() {
        pasteBtn.hidden = !state.clipboard;
        pasteBtn.disabled = !state.clipboard;
        pasteBtn.textContent = state.clipboard
            ? `Paste ${state.clipboard.mode === 'move' ? 'Move' : 'Copy'}`
            : 'Paste';
    }

    function setPreview({ path, meta, content, html, node }) {
        showStatus('');
        previewPathEl.textContent = path || 'Select a file';
        previewMetaEl.textContent = meta || '';
        if (node) {
            previewContentEl.replaceChildren(node);
            return;
        }
        if (typeof html === 'string') {
            previewContentEl.innerHTML = html;
            return;
        }
        previewContentEl.textContent = content || '';
    }

    function renderEditor(content, options = {}) {
        const wrapper = document.createElement('div');
        wrapper.className = 'files-editor-shell';

        if (options.isNew) {
            const nameInput = document.createElement('input');
            nameInput.className = 'files-editor-name ui-control';
            nameInput.type = 'text';
            nameInput.name = 'filename';
            nameInput.setAttribute('aria-label', 'File name');
            nameInput.placeholder = 'new-file.txt';
            nameInput.value = state.editorFilename || '';
            nameInput.autocomplete = 'off';
            nameInput.spellcheck = false;
            nameInput.addEventListener('input', () => {
                draftRevision += 1;
                state.editorFilename = nameInput.value;
                state.editorDirty = state.editorValue !== state.editorOriginal || state.editorFilename !== state.editorOriginalFilename;
                showStatus('');
                updateEditorActions();
            });
            wrapper.appendChild(nameInput);
        }

        const textarea = document.createElement('textarea');
        textarea.className = 'files-editor ui-control ui-control-code';
        textarea.name = 'file-content';
        textarea.setAttribute('aria-label', 'File contents');
        textarea.setAttribute('aria-describedby', 'files-preview-path files-preview-meta');
        textarea.value = content || '';
        textarea.spellcheck = false;
        textarea.placeholder = options.isNew ? 'Start typing file contents...' : '';
        textarea.addEventListener('input', () => {
            draftRevision += 1;
            state.editorValue = textarea.value;
            state.editorDirty = state.editorValue !== state.editorOriginal || state.editorFilename !== state.editorOriginalFilename;
            showStatus('');
            updateEditorActions();
        });
        wrapper.appendChild(textarea);
        return wrapper;
    }

    async function showModal({ title, message, input = false, initialValue = '', confirmLabel = 'OK', cancelLabel = 'Cancel' }) {
        const result = await openConfirmDialog({
            title,
            body: message,
            input,
            initialValue,
            confirmLabel,
            cancelLabel,
            danger: /delete|discard/i.test(confirmLabel),
        });
        return typeof result === 'boolean' ? { confirmed: result, value: '' } : result;
    }

    async function canLeaveEditor() {
        if (!state.editorDirty) return true;
        const result = await showModal({
            title: 'Discard Changes?',
            message: 'You have unsaved edits in the current file. Leave without saving?',
            confirmLabel: 'Discard',
            cancelLabel: 'Stay',
        });
        return Boolean(result?.confirmed);
    }

    function showContextMenu(x, y, path, type, destinationPath = '.', anchor = listEl) {
        hideContextMenu();
        state.contextPath = path || '';
        state.contextEntryType = type || '';
        state.contextDestinationPath = destinationPath || '.';
        const downloadItem = contextMenuEl.querySelector('[data-action="download"]');
        const pasteItem = contextMenuEl.querySelector('[data-action="paste"]');
        const deleteItem = contextMenuEl.querySelector('[data-action="delete"]');
        if (downloadItem) {
            downloadItem.hidden = type !== 'file';
        }
        if (pasteItem) {
            pasteItem.hidden = !state.clipboard || (type === 'file');
        }
        if (deleteItem) {
            deleteItem.hidden = !path;
        }
        contextMenuEl.querySelectorAll('[data-action="copy"], [data-action="move"]').forEach((item) => {
            item.hidden = !path;
        });
        contextAnchor = anchor;
        contextAnchor.setAttribute('aria-expanded', 'true');
        contextAnchor.setAttribute('aria-haspopup', 'menu');
        document.body.appendChild(contextMenuEl);
        contextMenuEl.hidden = false;
        contextMenuBinding = bindMenu(contextMenuEl, {
            anchor,
            point: Number.isFinite(x) && Number.isFinite(y) ? { x, y } : undefined,
            onClose: resetContextMenu,
        });
    }

    function resetContextMenu() {
        state.contextPath = '';
        state.contextEntryType = '';
        state.contextDestinationPath = '.';
        contextAnchor?.setAttribute('aria-expanded', 'false');
        contextAnchor = null;
        contextMenuBinding = null;
        contextMenuEl.hidden = true;
        layoutEl.appendChild(contextMenuEl);
    }

    function hideContextMenu(options) {
        contextMenuBinding?.close(options);
    }

    function filteredEntries() {
        const needle = state.filter.trim().toLowerCase();
        if (!needle) return state.entries;
        return state.entries.filter((entry) => entry.name.toLowerCase().includes(needle));
    }

    function renderBreadcrumb(items) {
        breadcrumbEl.innerHTML = '';
        items.forEach((item, idx) => {
            const btn = document.createElement('button');
            btn.className = 'files-crumb';
            btn.textContent = item.name;
            btn.addEventListener('click', () => {
                loadDirectory(item.path).catch(showError);
            });
            breadcrumbEl.appendChild(btn);
            if (idx < items.length - 1) {
                const sep = document.createElement('span');
                sep.className = 'files-crumb-sep';
                sep.textContent = '/';
                breadcrumbEl.appendChild(sep);
            }
        });
        revealCurrentFolder();
    }

    // A new file's unsaved draft has no row in the folder yet: on the list alone it gets one, at the top, to reopen it.
    function draftRow() {
        const name = state.editorFilename.trim();
        const label = name ? `<span class="files-entry-name">${escapeHtml(name)}</span>`
            : '<span class="files-entry-untitled">New file</span>';
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'files-entry is-draft selected';
        button.innerHTML = `
            <span class="files-entry-icon">•</span>
            ${label}
            <span class="files-entry-meta"><span class="files-entry-unsaved">unsaved</span></span>
        `;
        button.addEventListener('click', () => showPane('file', backBtn));
        return button;
    }

    function renderList() {
        listEl.innerHTML = '';
        if (state.editorIsNew) listEl.appendChild(draftRow());
        unsavedMark = unsavedPath();
        const entries = filteredEntries();
        const listEntries = [];
        if (state.parentPath && state.path !== '.') {
            listEntries.push({
                name: '..',
                path: state.parentPath,
                type: 'dir',
                isParentLink: true,
            });
        }
        listEntries.push(...entries);

        if (!listEntries.length) {
            const empty = document.createElement('div');
            empty.className = 'files-empty';
            empty.textContent = state.filter ? 'No matches in this folder.' : 'Folder is empty.';
            listEl.appendChild(empty);
            return;
        }

        listEntries.forEach((entry) => {
            const button = document.createElement('button');
            const selected = state.selectedPath === entry.path;
            button.type = 'button';
            if (!entry.isParentLink) {
                button.setAttribute('aria-haspopup', 'menu');
                button.setAttribute('aria-expanded', 'false');
            }
            button.className = `files-entry ${entry.isParentLink ? 'parent-link' : ''} ${selected ? 'selected' : ''}`;
            if (!entry.isParentLink) button.dataset.path = entry.path;
            button.toggleAttribute('data-unsaved', Boolean(unsavedMark) && entry.path === unsavedMark);
            const meta = entry.isParentLink ? 'up' : (entry.type === 'file' ? formatFileSize(entry.size) : 'open');
            const unsaved = entry.type === 'file' ? '<span class="files-entry-unsaved">unsaved</span>' : '';
            button.innerHTML = `
                <span class="files-entry-icon">${iconForEntry(entry)}</span>
                <span class="files-entry-name">${escapeHtml(entry.name)}</span>
                <span class="files-entry-meta"><span class="files-entry-size">${meta}</span>${unsaved}</span>
            `;
            button.addEventListener('contextmenu', (event) => {
                if (entry.isParentLink) return;
                event.preventDefault();
                showContextMenu(
                    event.clientX,
                    event.clientY,
                    entry.path,
                    entry.type,
                    entry.type === 'dir' ? entry.path : state.path || '.',
                    button,
                );
            });
            button.addEventListener('keydown', (event) => {
                if (entry.isParentLink || (event.key !== 'ContextMenu' && !(event.shiftKey && event.key === 'F10'))) return;
                event.preventDefault();
                showContextMenu(undefined, undefined, entry.path, entry.type, entry.type === 'dir' ? entry.path : state.path, button);
            });
            button.addEventListener('click', () => {
                hideContextMenu();
                if (entry.type === 'dir') {
                    loadDirectory(entry.path).catch(showError);
                } else {
                    loadFile(entry.path).catch(showError);
                }
            });
            listEl.appendChild(button);
        });
    }

    // The list's place survives the file shown in front of it: a list out of sight forgets its scroll.
    listEl.addEventListener('scroll', () => {
        if (listEl.getClientRects().length) listScroll = listEl.scrollTop;
    }, { passive: true });
    // Back to folder, and the layer's own close (Escape, the system Back, Telegram's Back): the list again, the draft
    // kept for its row, focus on that row.
    function closeFile() {
        renderList();
        showPane('list', listEl.querySelector('.files-entry.selected') || listEl);
    }
    backBtn.addEventListener('click', closeFile);

    // Focus stays this module's (showPane), so the layer neither takes it nor hands it back.
    function syncLayer(open) {
        if (!layers) return;
        if (open) layers.push({ id: FILE_LAYER, el: previewEl, page: true, modal: false, focus: false, opener: null, onClose: closeFile });
        else layers.remove(FILE_LAYER);
    }

    // A width that changes how many panes fit (a turned phone, a resized window) keeps focus in sight and the status
    // beside what it is about. The watch lives while Files is the page shown (app.js `showPage`, `ouro:page-shown`):
    // put away, the page drops its watch and its layer; shown again, the first observation re-reads its panes.
    let panes = 0;
    const panesWatch = typeof ResizeObserver === 'function' ? new ResizeObserver(() => {
        const shown = panesShown();
        if (shown === panes) return;
        panes = shown;
        if (shown) showPane(layoutEl.dataset.filesView);
        else syncLayer(false);
    }) : null;
    function watchPanes(shown) {
        if (shown) {
            panesWatch?.observe(layoutEl);
            return;
        }
        panesWatch?.disconnect();
        panes = 0;
        syncLayer(false);
    }
    window.addEventListener('ouro:page-shown', (event) => watchPanes(event?.detail?.page === 'files'));
    if (appState?.activePage === 'files') watchPanes(true);

    function showStatus(text, tone = 'muted') {
        setInlineStatus(previewStatusEl, text, tone);
        previewStatusEl.hidden = !text;
    }

    function showError(err) {
        showStatus(`Request failed: ${err instanceof Error ? err.message : String(err)}`, 'error');
    }

    async function loadDirectory(path = '.', options = {}) {
        const request = ++viewRequest;
        const preserveEditor = options.skipEditorReset || (!options.useBackendDefault && path === state.path);
        if (!preserveEditor && !options.skipLeaveCheck && !(await canLeaveEditor())) return;
        if (request !== viewRequest) return;
        const revision = draftRevision;
        hideContextMenu();
        const params = new URLSearchParams();
        if (options.useBackendDefault !== true) {
            params.set('path', path);
        }
        const query = params.toString();
        const data = await fetchJson(`/api/files/list${query ? `?${query}` : ''}`);
        if (request !== viewRequest) return;
        if (!preserveEditor && revision !== draftRevision) {
            showStatus('New edits kept. Choose the folder again when ready to leave.');
            return;
        }

        if (!options.keepStatus) showStatus('');
        if (!preserveEditor) {
            state.selectedPath = '';
            state.selectedType = 'dir';
            resetEditorState();
            showPane('list');
        }
        state.rootPath = data.root_path || state.rootPath;
        state.path = data.path || '.';
        state.parentPath = data.parent_path || '.';
        state.entries = Array.isArray(data.entries) ? data.entries : [];
        renderBreadcrumb(Array.isArray(data.breadcrumb) ? data.breadcrumb : []);
        renderList();

        if (!preserveEditor || (!state.selectedPath && !state.editorWritable)) {
            setPreview({
                path: data.display_path || state.rootPath || 'Files',
                meta: data.truncated ? 'Directory listing truncated.' : defaultDirectoryMeta(),
                content: defaultDirectoryContent(),
            });
        }
    }

    async function loadFile(path) {
        // The file already in the editor, its draft as it was (on the list alone, its row reopens it).
        if (state.selectedPath === path && state.editorPath === path) {
            showPane('file', backBtn);
            return;
        }
        const request = ++viewRequest;
        if (!(await canLeaveEditor()) || request !== viewRequest) return;
        const revision = draftRevision;
        hideContextMenu();
        const params = new URLSearchParams({ path });
        const data = await fetchJson(`/api/files/read?${params.toString()}`);
        if (request !== viewRequest) return;
        if (revision !== draftRevision) {
            showStatus('New edits kept. Choose the file again when ready to leave.');
            return;
        }

        state.selectedPath = data.path || path;
        state.selectedType = 'file';
        renderList();
        // Every kind of file is shown the same way: alone on a narrow page, with Back to folder focused.
        if (data.is_image && data.content_url) {
            resetEditorState();
            setPreview({
                path: data.display_path || state.rootPath || 'Files',
                meta: `${formatFileSize(data.size)} • ${data.media_type || 'image'}`,
                html: `<img class="files-preview-image" src="${escapeHtmlAttr(data.content_url)}" alt="${escapeHtmlAttr(data.name || data.path || 'image')}">`,
            });
        } else if (data.is_pdf && data.content_url) {
            resetEditorState();
            const safeUrl = escapeHtmlAttr(data.content_url);
            setPreview({
                path: data.display_path || state.rootPath || 'Files',
                meta: `${formatFileSize(data.size)} • PDF preview`,
                html: `<iframe class="files-preview-frame" sandbox="allow-same-origin" src="${safeUrl}" title="${escapeHtmlAttr(data.name || 'PDF preview')}"></iframe>`,
            });
        } else if (!data.is_text) {
            resetEditorState();
            setPreview({
                path: data.display_path || state.rootPath || 'Files',
                meta: `${formatFileSize(data.size)} • binary or unsupported preview`,
                content: 'Binary or non-text file preview is not available in the UI yet.',
            });
        } else {
            const editable = !data.truncated;
            state.editorPath = state.selectedPath;
            state.editorOriginalFilename = data.name || '';
            state.editorOriginal = data.content || '';
            state.editorValue = data.content || '';
            state.editorDirty = false;
            state.editorWritable = editable;
            state.editorIsNew = false;
            state.editorFilename = data.name || '';
            updateEditorActions();
            setPreview({
                path: data.display_path || state.rootPath || 'Files',
                meta: editable
                    ? `${formatFileSize(data.size)} • editable`
                    : `${formatFileSize(data.size)} • preview truncated • read-only`,
                node: editable ? renderEditor(data.content || '', { isNew: false }) : document.createTextNode(data.content || ''),
            });
        }
        showPane('file', backBtn);
    }

    function filenameFromPath(path) {
        return String(path || '').split('/').filter(Boolean).pop() || 'download';
    }

    async function downloadFile(path, { openExternal = false } = {}) {
        if (!path) return;
        const params = new URLSearchParams({ path });
        const url = `/api/files/download?${params.toString()}`;
        const filename = filenameFromPath(path);
        const result = await downloadViaHostBridge(url, filename, { openExternal });
        if (result.native) {
            showStatus(openExternal ? `Opened ${filename} externally.` : `${filename} saved to ${result.path || 'Downloads'}.`, 'success');
            return;
        }
    }

    async function createDirectory() {
        const result = await showModal({
            title: 'Create Directory',
            message: 'Enter a name for the new directory in the current folder.',
            input: true,
            confirmLabel: 'Create',
            cancelLabel: 'Cancel',
        });
        const name = (result?.value || '').trim();
        if (!result?.confirmed || !name) return;
        await jsonPost('/api/files/mkdir', { path: state.path || '.', name });
        await loadDirectory(state.path || '.');
        showStatus(`Directory ${name} created.`, 'success');
    }

    async function finishFileOperation(view, { directory, path = '', type = 'dir', message }) {
        const currentView = view.request === viewRequest;
        const replacePreview = currentView && view.revision === draftRevision;
        if (replacePreview) {
            resetEditorState();
            state.selectedPath = path;
            state.selectedType = type;
            updateEditorActions();
            setPreview({ path: path || state.rootPath, meta: message, content: '' });
            // What changed is in the folder: its list says so, with the message over it on a narrow page.
            showPane('list');
        }
        showStatus(message, 'success');
        // A completed write remains real, but it cannot take over newer work.
        // Refreshing the original folder must not cancel a newer navigation.
        if (!currentView || (!replacePreview && directory !== state.path)) return;
        try {
            await loadDirectory(directory, { skipLeaveCheck: true, skipEditorReset: true, keepStatus: true });
            showStatus(message, 'success');
        } catch (err) {
            showStatus(`${message} Folder refresh failed: ${err instanceof Error ? err.message : String(err)}`, 'error');
        }
    }

    async function pasteClipboard(destinationPath = state.path || '.') {
        const clipboard = state.clipboard;
        if (!clipboard) return;
        if (!(await canLeaveEditor())) return;
        const view = { request: viewRequest, revision: draftRevision };

        const data = await jsonPost('/api/files/transfer', {
            source_path: clipboard.path,
            destination_dir: destinationPath || '.',
            mode: clipboard.mode,
        });

        if (state.clipboard === clipboard) state.clipboard = null;
        updateClipboardActions();
        await finishFileOperation(view, {
            directory: destinationPath || '.', path: data.path || '', type: data.type || '',
            message: `${clipboard.mode === 'move' ? 'Moved' : 'Copied'} ${data.display_path || data.path || clipboard.name}.`,
        });
    }

    async function deleteSelectedEntry(path = state.selectedPath) {
        if (!path) return;
        const entry = state.entries.find((item) => item.path === path);
        if (!entry) return;
        if (!(await canLeaveEditor())) return;

        const result = await showModal({
            title: `Delete ${entry.type === 'dir' ? 'Directory' : 'File'}?`,
            message: entry.type === 'dir'
                ? `Delete "${entry.name}" and all its contents? This cannot be undone.`
                : `Delete "${entry.name}"? This cannot be undone.`,
            confirmLabel: 'Delete',
            cancelLabel: 'Cancel',
        });
        if (!result?.confirmed) return;
        const view = { request: viewRequest, revision: draftRevision };
        const directory = state.path || '.';

        await jsonPost('/api/files/delete', { path });

        await finishFileOperation(view, {
            directory, message: `${entry.type === 'dir' ? 'Directory' : 'File'} deleted: ${entry.name}.`,
        });
    }

    async function uploadFiles(fileList) {
        const files = Array.from(fileList || []);
        if (!files.length) return;
        if (!(await canLeaveEditor())) return;
        const uploadPath = state.path || '.';
        const view = { request: viewRequest, revision: draftRevision };
        let uploaded;

        for (const file of files) {
            const form = new FormData();
            form.set('path', uploadPath);
            form.set('file', file);

            showStatus(`Uploading ${file.name} into ${uploadPath}…`);

            uploaded = await fetchJson('/api/files/upload', {
                method: 'POST',
                body: form,
            });
        }

        await finishFileOperation(view, {
            directory: uploadPath, path: uploaded.path || '', type: 'file',
            message: `Upload complete: ${files.map(file => file.name).join(', ')} in ${uploadPath}.`,
        });
    }

    async function saveCurrentFile() {
        if (!state.editorWritable || state.editorSaving || (!state.editorDirty && !state.editorIsNew)) return;
        const relName = state.editorFilename.trim();
        if (state.editorIsNew && !relName) return;
        const savePath = state.editorIsNew
            ? (state.path && state.path !== '.' ? `${state.path}/${relName}` : relName)
            : state.editorPath;
        if (!savePath) return;
        const editor = previewContentEl.querySelector('.files-editor');
        const nameInput = previewContentEl.querySelector('.files-editor-name');
        const content = state.editorValue;
        state.editorSaving = true;
        if (nameInput) nameInput.disabled = true;
        updateEditorActions();
        showStatus('Saving…');
        try {
            const data = await jsonPost('/api/files/write', {
                path: savePath,
                content,
                create: state.editorIsNew,
            });
            if (editor !== previewContentEl.querySelector('.files-editor')) return;
            state.selectedPath = data.path || savePath;
            state.selectedType = 'file';
            state.editorPath = state.selectedPath;
            state.editorFilename = data.name || relName;
            state.editorOriginalFilename = state.editorFilename;
            state.editorIsNew = false;
            state.editorOriginal = content;
            state.editorDirty = state.editorValue !== content;
            if (nameInput === document.activeElement) editor?.focus();
            nameInput?.remove();
            previewPathEl.textContent = data.display_path || state.editorPath;
            previewMetaEl.textContent = `${formatFileSize(data.size)} • editable`;
            showStatus(state.editorDirty ? 'Saved. Newer edits are still unsaved.' : 'Saved.', 'success');
        } finally {
            state.editorSaving = false;
            if (nameInput) nameInput.disabled = false;
            updateEditorActions();
        }
        try {
            await loadDirectory(state.path || '.', { skipEditorReset: true, keepStatus: true });
        } catch (err) {
            if (editor === previewContentEl.querySelector('.files-editor')) {
                showStatus(`Saved. Folder refresh failed: ${err instanceof Error ? err.message : String(err)}`, 'error');
            }
        }
    }

    function createNewFile(options = {}) {
        if (state.editorDirty && !options.force) return;
        hideContextMenu();
        viewRequest += 1;
        state.selectedPath = '';
        state.selectedType = 'file';
        state.editorPath = '';
        state.editorOriginalFilename = '';
        state.editorOriginal = '';
        state.editorValue = '';
        state.editorDirty = true;
        state.editorWritable = true;
        state.editorIsNew = true;
        state.editorFilename = '';
        renderList();
        updateEditorActions();
        setPreview({
            path: state.path && state.path !== '.'
                ? `${state.rootPath || 'Files'}/${state.path}`
                : (state.rootPath || 'Files'),
            meta: 'New file • editable',
            node: renderEditor('', { isNew: true }),
        });
        showPane('file', previewContentEl.querySelector('.files-editor-name'));
    }

    searchEl.addEventListener('input', () => {
        state.filter = searchEl.value || '';
        renderList();
    });

    newFileBtn.addEventListener('click', async () => {
        if (!(await canLeaveEditor())) return;
        createNewFile({ force: true });
    });

    newDirBtn.addEventListener('click', () => {
        createDirectory().catch(showError);
    });

    pasteBtn.addEventListener('click', () => {
        pasteClipboard().catch(showError);
    });

    saveBtn.addEventListener('click', () => {
        saveCurrentFile().catch(showError);
    });
    downloadBtn.addEventListener('click', () => {
        downloadFile(state.selectedPath).catch(showError);
    });
    openExternalBtn.addEventListener('click', () => {
        downloadFile(state.selectedPath, { openExternal: true }).catch(showError);
    });

    layoutEl.addEventListener('dragenter', (event) => {
        event.preventDefault();
        state.dragDepth += 1;
        layoutEl.classList.add('drag-active');
    });

    layoutEl.addEventListener('dragover', (event) => {
        event.preventDefault();
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = 'copy';
        }
    });

    layoutEl.addEventListener('dragleave', (event) => {
        event.preventDefault();
        state.dragDepth = Math.max(0, state.dragDepth - 1);
        if (state.dragDepth === 0) {
            layoutEl.classList.remove('drag-active');
        }
    });

    layoutEl.addEventListener('drop', async (event) => {
        event.preventDefault();
        state.dragDepth = 0;
        layoutEl.classList.remove('drag-active');
        hideContextMenu();
        try {
            await uploadFiles(event.dataTransfer && event.dataTransfer.files);
        } catch (err) {
            showError(err);
        }
    });

    contextMenuEl.addEventListener('click', (event) => {
        const action = event.target.closest('[data-action]')?.dataset.action;
        if (!action) return;
        const path = state.contextPath;
        const destinationPath = state.contextDestinationPath;
        hideContextMenu({ restoreFocus: true });
        if (action === 'download') {
            downloadFile(path).catch(showError);
        } else if (action === 'copy' || action === 'move') {
            const entry = state.entries.find((item) => item.path === path);
            if (entry) {
                state.clipboard = {
                    mode: action,
                    path: entry.path,
                    name: entry.name,
                    type: entry.type,
                };
                updateClipboardActions();
                showStatus(`${action === 'move' ? 'Move' : 'Copy'} ready: ${entry.name} will be ${action === 'move' ? 'moved' : 'copied'} into the next folder where you press Paste.`);
            }
        } else if (action === 'paste') {
            pasteClipboard(destinationPath).catch(showError);
        } else if (action === 'delete') {
            deleteSelectedEntry(path).catch(showError);
        }
    });

    listEl.addEventListener('contextmenu', (event) => {
        if (event.target === listEl && state.clipboard) {
            event.preventDefault();
            showContextMenu(event.clientX, event.clientY, '', 'dir', state.path || '.');
        }
    });

    listEl.addEventListener('keydown', (event) => {
        if (event.target !== listEl || !state.clipboard || (event.key !== 'ContextMenu' && !(event.shiftKey && event.key === 'F10'))) return;
        event.preventDefault();
        showContextMenu(undefined, undefined, '', 'dir', state.path || '.');
    });

    refreshBtn.addEventListener('click', () => {
        loadDirectory(state.path || '.').catch(showError);
    });

    window.addEventListener('beforeunload', (event) => {
        if (!state.editorDirty) return;
        event.preventDefault();
        event.returnValue = '';
    });

    document.addEventListener('keydown', (event) => {
        const active = document.activeElement;
        const inEditor = active && (
            active.classList?.contains('files-editor') ||
            active.classList?.contains('files-editor-name')
        );
        const dialogOpen = Boolean(document.querySelector('[aria-modal="true"]'));
        if (!page.classList.contains('active')) return;
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
            if (!inEditor || dialogOpen) return;
            event.preventDefault();
            if (event.repeat) return;
            saveCurrentFile().catch(showError);
            return;
        }
        if (event.key === 'Delete') {
            if (inEditor || active === searchEl || dialogOpen || contextMenuBinding) return;
            event.preventDefault();
            deleteSelectedEntry().catch(showError);
        }
    });

    if (typeof setBeforePageLeave === 'function') {
        setBeforePageLeave(async ({ from }) => {
            if (from !== 'files') return true;
            hideContextMenu();
            return canLeaveEditor();
        });
    }
    if (appState) {
        appState.filesState = state;
    }

    updateClipboardActions();
    loadDirectory('.', { useBackendDefault: true }).catch(showError);
}
