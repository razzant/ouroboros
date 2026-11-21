// New Project dialog + project row actions (v6.59.0, Phase 3).
//
// One modal, four sources (quiz-approved UX): file-less | genesis (fresh managed
// folder) | attach an existing folder (server-side directory browser — works in
// web/Docker where no native picker exists; optional init_git attach-snapshot)
// | clone a git URL (server-side, typed auth_required). The dialog carries the
// honest trust line: attaching gives the agent write+shell in that folder
// (notification model — attaching IS the grant; no second confirmation).
// "No folder" is sent as the owner's recorded choice (`no_folder`): the project keeps none until one is chosen in
// its Working folder… dialog (project_folder.js), whose folder browser this dialog shares.
import { openConfirmDialog } from './confirm_dialog.js';
import { mountFolderBrowser, openProjectFolderDialog } from './project_folder.js';
import { createStatusMark, paintStatusMark } from './status_mark.js';
import { errorWords, setResult } from './ui_primitives.js';
import { bindDialogFocus, bindMenu } from './ui_interactions.js';

// Mirrored from the frozen backend PROJECT_NAME_MAX contract.
const PROJECT_NAME_MAX = 80;
// A working folder's short name in the choice, and the line said under the choice for the chosen one only (U025).
const SOURCES = [
    ['fileless', 'No folder', 'A chat and research project with no folder of its own: tasks work in their own temporary area.'],
    ['genesis', 'New folder', 'A new folder with a fresh Git repository, under the projects folder.'],
    ['attach', 'Existing folder…', 'A folder on this computer; Git is optional.'],
    ['clone', 'Clone from Git…', 'A copy of a repository, cloned from its Git URL.'],
];
// A failure's owner words: a typed refusal says what to do; anything else is errorWords' (the raw text in Details).
const failureWords = (e) => (e?.body?.error_code === 'auth_required'
    ? { text: 'This repository needs credentials. Set up Git access on this computer, then try again.', tone: 'danger',
        details: errorWords(e).details }
    : errorWords(e));

export function openNewProjectDialog({ apiClient, onCreated }) {
    const maxNameLength = PROJECT_NAME_MAX;
    return new Promise((resolve) => {
        const backdrop = document.createElement('div');
        backdrop.className = 'marketplace-modal-backdrop new-project-backdrop';
        backdrop.innerHTML = `
            <div class="marketplace-modal new-project-dialog" role="dialog" aria-modal="true" aria-labelledby="new-project-title">
                <div class="marketplace-modal-head">
                    <h3 id="new-project-title">New project</h3>
                </div>
                <div class="marketplace-modal-body">
                    <label class="new-project-field ui-field">
                        <span>Name</span>
                        <input class="files-modal-input ui-control" data-np-name name="project-name" type="text" placeholder="My project" maxlength="${maxNameLength}">
                    </label>
                    <fieldset class="new-project-sources new-project-field">
                        <legend>Working folder</legend>
                        <div class="new-project-source-choices">${SOURCES.map(([value, label], index) => `
                            <label><input class="ui-checkbox" type="radio" name="np-source" value="${value}"${index ? '' : ' checked'}> ${label}</label>`).join('')}
                        </div>
                        <p class="ui-field-help" data-np-help>${SOURCES[0][2]}</p>
                    </fieldset>
                    <div data-np-attach hidden>
                        <div data-np-browser></div>
                        <label class="local-toggle ui-field ui-field-inline" title="Runs git init + an 'attach snapshot' commit of the current state with a local identity. Never done without this checkbox.">
                            <input class="ui-checkbox" type="checkbox" name="project-init-git" data-np-initgit>
                            Initialize Git and commit current files
                        </label>
                    </div>
                    <div data-np-clone hidden>
                        <label class="new-project-field ui-field">
                            <span>Git URL</span>
                            <input class="files-modal-input ui-control" data-np-giturl name="project-git-url" type="text" placeholder="https://github.com/user/repo.git or git@github.com:user/repo.git">
                        </label>
                    </div>
                    <p class="new-project-trust-note" data-np-trust hidden>
                        The agent gets <strong>read, write and shell</strong> in this folder when working on this project's tasks.
                    </p>
                    <div class="new-project-error ui-result" data-np-error role="alert"></div>
                </div>
                <div class="marketplace-modal-actions">
                    <button type="button" class="btn btn-default" data-np-cancel>Cancel</button>
                    <button type="button" class="btn" data-primary data-np-create>Create project</button>
                </div>
            </div>
        `;
        const q = (sel) => backdrop.querySelector(sel);
        let settled = false;
        let disposeFocus = () => {};

        const setError = (result) => setResult(q('[data-np-error]'), typeof result === 'string' ? { text: result, tone: 'danger' } : result || {});
        const browser = mountFolderBrowser(q('[data-np-browser]'), {
            apiClient,
            onSelect: () => { q('[data-np-trust]').hidden = false; },
            onError: (e) => setError(failureWords(e)),
        });

        function syncSource() {
            const source = backdrop.querySelector('input[name="np-source"]:checked')?.value || 'fileless';
            q('[data-np-help]').textContent = SOURCES.find(([value]) => value === source)?.[2] || '';
            q('[data-np-attach]').hidden = source !== 'attach';
            q('[data-np-clone]').hidden = source !== 'clone';
            q('[data-np-trust]').hidden = !(
                (source === 'attach' && browser.selected()) || source === 'clone' || source === 'genesis'
            );
            if (source === 'attach' && !browser.current()) browser.open('');
        }
        backdrop.querySelectorAll('input[name="np-source"]').forEach((radio) => {
            radio.addEventListener('change', syncSource);
        });

        const finish = (value) => {
            if (settled) return;
            settled = true;
            browser.dispose();
            disposeFocus();
            backdrop.remove();
            resolve(value);
        };

        async function create() {
            if (settled || q('[data-np-create]').disabled) return;
            setError('');
            const name = (q('[data-np-name]').value || '').trim();
            const source = backdrop.querySelector('input[name="np-source"]:checked')?.value || 'fileless';
            if (!name) { setError('Give the project a name.'); return; }
            const payload = { name };
            if (source === 'fileless') payload.no_folder = true;  // the choice the dialog shows, never a default behind it
            if (source === 'genesis') payload.with_workspace = true;
            if (source === 'attach') {
                if (!browser.selected()) { setError('Pick a folder to attach (Select this folder).'); return; }
                payload.path = browser.selected();
                payload.init_git = Boolean(q('[data-np-initgit]').checked);
            }
            if (source === 'clone') {
                const url = (q('[data-np-giturl]').value || '').trim();
                if (!url) { setError('Enter the Git URL to clone.'); return; }
                payload.git_url = url;
            }
            const btn = q('[data-np-create]');
            btn.disabled = true;
            btn.textContent = source === 'clone' ? 'Cloning…' : 'Creating…';
            try {
                const data = await apiClient.projectCreate(payload);
                if (data?.error) throw Object.assign(new Error(data.error), { body: data });
                finish(data?.project || null);
                onCreated?.(data?.project || null);
            } catch (e) {
                if (settled) return;
                setError(failureWords(e));
                btn.disabled = false;
                btn.textContent = 'Create project';
            }
        }
        backdrop.addEventListener('click', (event) => {
            if (event.target === backdrop || event.target.closest('[data-np-cancel]')) finish(null);
            else if (event.target.closest('[data-np-create]')) create();
        });
        document.body.appendChild(backdrop);
        disposeFocus = bindDialogFocus(q('[role="dialog"]'), {
            initialFocus: q('[data-np-name]'),
            onEscape: () => finish(null),
        });
    });
}

let closeOpenProjectRowMenu = null;
let repositionOpenProjectRowMenu = null;

export function repositionProjectRowMenu() {
    repositionOpenProjectRowMenu?.();
}

const shown = (node) => Boolean(node?.isConnected && node.getClientRects().length);

// The menu's first line: the row's mark painted with the row's view and what it says in words (owner 19.10, 5A), so the
// row's state reads in words where a finger opened the menu. Not an item: the arrows and the first focus pass it by,
// and the menu's name says the same words. The words come translated (project_activity.js), so the interface
// language's overlay leaves the line alone.
function statusLine({ view, word = '', text = '' }) {
    const line = document.createElement('div');
    line.className = 'project-row-menu-status';
    line.setAttribute('role', 'none');
    line.dataset.i18nSkip = '';
    const mark = createStatusMark(document, 'project-row-menu-mark');
    paintStatusMark(mark, view);
    const said = text.charAt(0).toLocaleUpperCase() + text.slice(1);
    const lead = document.createElement('span');
    lead.className = 'project-row-menu-word';
    lead.textContent = word && said.startsWith(word) ? word : said;
    line.append(mark, lead);
    if (lead.textContent !== said) line.append(document.createTextNode(said.slice(lead.textContent.length)));
    return line;
}

/** A row's Pin or Unpin in words, for the state it shows: this menu's item and the row's own glyph (projects_nav.js)
 *  say it alike, and each asks for the state its words name. A pin still being saved reads as such. */
export const pinWords = (pinned, pending = false) => (pending ? (pinned ? 'Unpinning…' : 'Pinning…')
    : (pinned ? 'Unpin' : 'Pin'));

// Accessible row menu: Pin/Unpin, Rename, Working folder, Delete. `project` is a sidebar read model;
// callbacks refresh the authoritative registry projection. `onPin` receives the state
// the label named; a pin still being saved reads as such and cannot be pressed again.
// The menu opens under `anchorEl` (the row's ⋯, or the focused row) or at a pointer's `point` (a right-click or a
// long press on the row), whose focus returns to `anchorEl`. Only `menuButton`, the ⋯ that opened it, says it has a
// popup. `status` ({ view, word, text }) is the row's mark and its words, the menu's first line. When the anchor is
// gone or hidden (a delete, an archive, a find that no longer lists it), the menu closes and the keyboard goes to
// `fallbackFocus`.
// Returns true: the menu is open.
export function openProjectRowMenu(project, {
    apiClient, anchorEl, point = null, menuButton = null, status = null, fallbackFocus = null,
    onChanged, onPin, isPinPending = () => false,
}) {
    const pinPending = isPinPending();
    const maxNameLength = PROJECT_NAME_MAX;
    const pinned = project.pinned === true;
    closeOpenProjectRowMenu?.();
    const menu = document.createElement('div');
    menu.className = 'project-row-menu ui-popup';
    menu.setAttribute('role', 'menu');
    const name = project.name || project.id;
    menu.setAttribute('aria-label', `Actions for ${name}${status?.text ? ` · ${status.text}` : ''}`);
    menu.innerHTML = `
        <button type="button" role="menuitem" data-prm="pin"${pinPending ? ' aria-disabled="true"' : ''}>${
            pinWords(pinned, pinPending)}</button>
        <button type="button" role="menuitem" data-prm="rename">Rename…</button>
        <button type="button" role="menuitem" data-prm="folder">Working folder…</button>
        <button type="button" role="menuitem" class="danger" data-prm="delete">Delete project…</button>
    `;
    if (status?.text) menu.prepend(statusLine(status));
    // After a dialog the keyboard returns to the row, or to the fallback when the row has gone or hidden meanwhile.
    const focusBack = () => (shown(anchorEl) ? anchorEl : fallbackFocus)?.focus();
    let binding;
    const close = (options) => binding?.close(options);
    menu.addEventListener('click', async (event) => {
        const item = event.target.closest('[data-prm]');
        const action = item?.dataset?.prm;
        if (!action || item.getAttribute('aria-disabled') === 'true') return;
        close({ restoreFocus: true });
        if (action === 'pin') onPin?.(!pinned);
        else if (action === 'rename') {
            const res = await openConfirmDialog({
                title: 'Rename project',
                body: `New name for “${name}”:`,
                input: true,
                initialValue: name,
                confirmLabel: 'Rename',
            });
            const newName = res?.confirmed ? String(res.value || '').trim() : '';
            if (newName.length > maxNameLength) {
                await openConfirmDialog({
                    title: 'Rename project',
                    body: `Project names are limited to ${maxNameLength} characters.`,
                    alert: true,
                });
            } else if (newName && newName !== project.name) {
                try { await apiClient.projectUpdate(project.id, newName); onChanged?.(); }
                catch (e) {
                    await openConfirmDialog({
                        title: 'Rename failed',
                        body: errorWords(e).text, details: { summary: 'Details', rows: [{ label: 'Error', value: errorWords(e).details }] },
                        alert: true,
                    });
                }
            }
            focusBack();
        } else if (action === 'folder') {
            if (await openProjectFolderDialog({ apiClient, project })) onChanged?.();
            focusBack();
        } else if (action === 'delete') {
            const ok = await openConfirmDialog({
                title: 'Delete project',
                body: `Delete “${name}”? Running work will be cancelled. The Project will be removed from the active UI; its id, chat history, task bindings, memory, and working folder are preserved.`,
                confirmLabel: 'Delete',
                danger: true,
            });
            if (ok === true) {
                onChanged?.({ projectId: project.id, lifecycle: 'deleting', optimistic: true });
                // Its answer is the registry's own word: the Project was fenced `deleting` before the reply came.
                let fenced = false;
                try { await apiClient.projectDelete(project.id); fenced = true; }
                catch (e) {
                    await openConfirmDialog({
                        title: 'Delete did not finish',
                        body: errorWords(e).text, details: { summary: 'Details', rows: [{ label: 'Error', value: errorWords(e).details }] },
                        alert: true,
                    });
                }
                finally { onChanged?.({ authoritative: true, ...(fenced && { projectId: project.id, lifecycle: 'deleting' }) }); }
            }
            focusBack();
        }
    });
    document.body.appendChild(menu);
    menuButton?.setAttribute('aria-haspopup', 'menu');
    menuButton?.setAttribute('aria-expanded', 'true');
    binding = bindMenu(menu, {
        anchor: anchorEl,
        point,
        onClose: () => {
            menu.remove();
            menuButton?.setAttribute('aria-expanded', 'false');
            if (closeOpenProjectRowMenu === close) {
                closeOpenProjectRowMenu = repositionOpenProjectRowMenu = null;
            }
        },
    });
    closeOpenProjectRowMenu = close;
    repositionOpenProjectRowMenu = () => {
        if (!shown(anchorEl)) close({ restoreFocus: true, focusTarget: fallbackFocus || anchorEl, reason: 'hidden' });
        // A menu opened during a save must not retain its disabled snapshot afterwards.
        else if (pinPending && !isPinPending()) close({ restoreFocus: true });
        else binding.reposition();
    };
    return true;
}
