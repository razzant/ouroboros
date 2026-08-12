// New Project dialog + project row actions (v6.59.0, Phase 3).
//
// One modal, four sources (quiz-approved UX): file-less | genesis (fresh managed
// folder) | attach an existing folder (server-side directory browser — works in
// web/Docker where no native picker exists; optional init_git attach-snapshot)
// | clone a git URL (server-side, typed auth_required). The dialog carries the
// honest trust line: attaching gives the agent write+shell in that folder
// (notification model — attaching IS the grant; no second confirmation).
import { openConfirmDialog } from './confirm_dialog.js';

// Mirrored from the frozen backend PROJECT_NAME_MAX contract.
const PROJECT_NAME_MAX = 80;

export function openNewProjectDialog({ apiClient, onCreated }) {
    const maxNameLength = PROJECT_NAME_MAX;
    return new Promise((resolve) => {
        const backdrop = document.createElement('div');
        backdrop.className = 'marketplace-modal-backdrop new-project-backdrop';
        backdrop.innerHTML = `
            <div class="marketplace-modal new-project-dialog" role="dialog" aria-modal="true" aria-labelledby="new-project-title">
                <div class="marketplace-modal-head">
                    <h3 id="new-project-title">New Project</h3>
                    <button type="button" class="btn btn-default btn-sm" data-np-cancel aria-label="Close">Close</button>
                </div>
                <div class="marketplace-modal-body">
                    <label class="new-project-field">
                        <span>Name</span>
                        <input class="files-modal-input" data-np-name type="text" placeholder="My project" maxlength="${maxNameLength}">
                    </label>
                    <fieldset class="new-project-sources">
                        <legend>Working folder</legend>
                        <label><input type="radio" name="np-source" value="fileless" checked> None — chat/research project (no folder)</label>
                        <label><input type="radio" name="np-source" value="genesis"> New managed folder (fresh git repo under the projects root)</label>
                        <label><input type="radio" name="np-source" value="attach"> Attach an existing folder…</label>
                        <label><input type="radio" name="np-source" value="clone"> Clone a git URL…</label>
                    </fieldset>
                    <div class="new-project-source-detail" data-np-attach hidden>
                        <div class="new-project-browser">
                            <div class="new-project-browser-path" data-np-path></div>
                            <div class="new-project-browser-list" data-np-dirs></div>
                        </div>
                        <label class="local-toggle" title="Runs git init + an 'attach snapshot' commit of the current state with a local identity. Never done without this checkbox.">
                            <input type="checkbox" data-np-initgit>
                            Initialize git here if missing (attach-snapshot commit)
                        </label>
                    </div>
                    <div class="new-project-source-detail" data-np-clone hidden>
                        <label class="new-project-field">
                            <span>Git URL</span>
                            <input class="files-modal-input" data-np-giturl type="text" placeholder="https://github.com/user/repo.git or git@github.com:user/repo.git">
                        </label>
                    </div>
                    <p class="new-project-trust-note" data-np-trust hidden>
                        The agent gets <strong>read, write and shell</strong> in this folder when working on this project's tasks.
                    </p>
                    <p class="new-project-error" data-np-error hidden></p>
                </div>
                <div class="marketplace-modal-actions">
                    <button type="button" class="btn btn-default" data-np-cancel>Cancel</button>
                    <button type="button" class="btn btn-primary" data-np-create>Create project</button>
                </div>
            </div>
        `;
        const q = (sel) => backdrop.querySelector(sel);
        let selectedDir = '';
        let browsePath = '';

        const setError = (text) => {
            const el = q('[data-np-error]');
            el.textContent = text || '';
            el.hidden = !text;
        };

        async function renderDirs(path) {
            const listEl = q('[data-np-dirs]');
            const pathEl = q('[data-np-path]');
            listEl.textContent = 'Loading…';
            try {
                const data = await apiClient.fsDirs(path);
                browsePath = data.path;
                pathEl.textContent = '';
                const cur = document.createElement('strong');
                cur.textContent = data.path;
                pathEl.appendChild(cur);
                const select = document.createElement('button');
                select.type = 'button';
                select.className = 'btn btn-sm ' + (selectedDir === data.path ? 'btn-primary' : 'btn-default');
                select.textContent = selectedDir === data.path ? 'Selected ✓' : 'Select this folder';
                select.addEventListener('click', () => {
                    selectedDir = data.path;
                    q('[data-np-trust]').hidden = false;
                    renderDirs(data.path);
                });
                pathEl.appendChild(select);
                listEl.textContent = '';
                if (data.parent) {
                    const up = document.createElement('button');
                    up.type = 'button';
                    up.className = 'new-project-dir-row new-project-dir-up';
                    up.textContent = '.. (up)';
                    up.addEventListener('click', () => renderDirs(data.parent));
                    listEl.appendChild(up);
                }
                for (const dir of data.dirs || []) {
                    const row = document.createElement('button');
                    row.type = 'button';
                    row.className = 'new-project-dir-row';
                    row.textContent = `${dir.name}${dir.is_git ? '  ⎇' : ''}`;
                    row.title = dir.path + (dir.is_git ? ' (git repository)' : '');
                    row.addEventListener('click', () => renderDirs(dir.path));
                    listEl.appendChild(row);
                }
                if (!(data.dirs || []).length) {
                    const empty = document.createElement('div');
                    empty.className = 'new-project-dir-empty';
                    empty.textContent = '(no subfolders)';
                    listEl.appendChild(empty);
                }
                if (data.truncated) {
                    const more = document.createElement('div');
                    more.className = 'new-project-dir-empty';
                    more.textContent = '(more folders exist — showing the first 500)';
                    listEl.appendChild(more);
                }
            } catch (e) {
                listEl.textContent = `Cannot browse: ${e?.message || e}`;
            }
        }

        function syncSource() {
            const source = backdrop.querySelector('input[name="np-source"]:checked')?.value || 'fileless';
            q('[data-np-attach]').hidden = source !== 'attach';
            q('[data-np-clone]').hidden = source !== 'clone';
            q('[data-np-trust]').hidden = !(
                (source === 'attach' && selectedDir) || source === 'clone' || source === 'genesis'
            );
            if (source === 'attach' && !browsePath) renderDirs('');
        }
        backdrop.querySelectorAll('input[name="np-source"]').forEach((radio) => {
            radio.addEventListener('change', syncSource);
        });

        let settled = false;
        const finish = (value) => {
            if (settled) return;
            settled = true;
            document.removeEventListener('keydown', onKey);
            backdrop.remove();
            resolve(value);
        };
        const onKey = (event) => {
            if (event.key === 'Escape') finish(null);
        };
        document.addEventListener('keydown', onKey);

        async function create() {
            setError('');
            const name = (q('[data-np-name]').value || '').trim();
            const source = backdrop.querySelector('input[name="np-source"]:checked')?.value || 'fileless';
            if (!name) { setError('Give the project a name.'); return; }
            const payload = { name };
            if (source === 'genesis') payload.with_workspace = true;
            if (source === 'attach') {
                if (!selectedDir) { setError('Pick a folder to attach (Select this folder).'); return; }
                payload.path = selectedDir;
                payload.init_git = Boolean(q('[data-np-initgit]').checked);
            }
            if (source === 'clone') {
                const url = (q('[data-np-giturl]').value || '').trim();
                if (!url) { setError('Enter the git URL to clone.'); return; }
                payload.git_url = url;
            }
            const btn = q('[data-np-create]');
            btn.disabled = true;
            btn.textContent = source === 'clone' ? 'Cloning…' : 'Creating…';
            try {
                const data = await apiClient.projectCreate(payload);
                if (data?.error) throw new Error(data.error);
                finish(data?.project || null);
                onCreated?.(data?.project || null);
            } catch (e) {
                const detail = e?.body?.error || e?.message || String(e);
                const code = e?.body?.error_code || '';
                setError(code === 'auth_required'
                    ? `This repository needs credentials (private repo). Set up git access on this machine, then retry. Detail: ${detail}`
                    : detail);
                btn.disabled = false;
                btn.textContent = 'Create project';
            }
        }
        backdrop.addEventListener('click', (event) => {
            if (event.target === backdrop || event.target.closest('[data-np-cancel]')) finish(null);
            else if (event.target.closest('[data-np-create]')) create();
        });
        document.body.appendChild(backdrop);
        q('[data-np-name]').focus();
    });
}

let closeOpenProjectRowMenu = null;

/**
 * The ONE row-menu shell: markup contract (`role=menu` + `[role=menuitem]`),
 * keyboard model (Escape / ArrowUp / ArrowDown / Home / End), click-outside
 * dismissal, focus restoration and viewport-safe placement. Project rows and
 * project THREAD rows (`modules/project_threads.js`) both mount through it, so
 * there is one accessible menu behaviour to keep correct rather than two that
 * drift. `itemsHtml` supplies the `[data-prm]` items; `onSelect(action)` runs
 * after the menu has closed.
 *
 * @returns {{ close: (opts?: {restoreFocus?: boolean}) => void }}
 */
export function openRowMenu({ anchorEl, ariaLabel, itemsHtml, onSelect }) {
    closeOpenProjectRowMenu?.();
    const menu = document.createElement('div');
    menu.className = 'project-row-menu';
    menu.setAttribute('role', 'menu');
    menu.setAttribute('aria-label', ariaLabel);
    menu.innerHTML = itemsHtml;
    const rect = anchorEl.getBoundingClientRect();
    const close = ({ restoreFocus = false } = {}) => {
        menu.remove();
        document.removeEventListener('click', onDoc, true);
        document.removeEventListener('keydown', onKey, true);
        if (closeOpenProjectRowMenu === close) closeOpenProjectRowMenu = null;
        if (restoreFocus && anchorEl.isConnected) anchorEl.focus();
    };
    const onDoc = (event) => { if (!menu.contains(event.target)) close(); };
    const onKey = (event) => {
        const items = Array.from(menu.querySelectorAll('[role="menuitem"]'));
        const index = items.indexOf(document.activeElement);
        if (event.key === 'Escape') {
            event.preventDefault();
            close({ restoreFocus: true });
        } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            const delta = event.key === 'ArrowDown' ? 1 : -1;
            items[(index + delta + items.length) % items.length]?.focus();
        } else if (event.key === 'Home' || event.key === 'End') {
            event.preventDefault();
            items[event.key === 'Home' ? 0 : items.length - 1]?.focus();
        }
    };
    closeOpenProjectRowMenu = close;
    document.addEventListener('click', onDoc, true);
    document.addEventListener('keydown', onKey, true);
    menu.addEventListener('click', async (event) => {
        const action = event.target.closest('[data-prm]')?.dataset?.prm;
        if (!action) return;
        close();
        await onSelect?.(action);
    });
    document.body.appendChild(menu);
    const menuRect = menu.getBoundingClientRect();
    const margin = 8;
    const top = Math.min(
        Math.max(margin, rect.bottom + 4),
        Math.max(margin, window.innerHeight - menuRect.height - margin),
    );
    const left = Math.min(
        Math.max(margin, rect.right - menuRect.width),
        Math.max(margin, window.innerWidth - menuRect.width - margin),
    );
    menu.style.setProperty('--prm-top', `${Math.round(top)}px`);
    menu.style.setProperty('--prm-left', `${Math.round(left)}px`);
    menu.querySelector('[role="menuitem"]')?.focus();
    return { close };
}

// Accessible row menu: Rename/Fork/Delete. `project` is a sidebar read model;
// callbacks refresh the authoritative registry projection. The project row IS
// thread #0's row (T1), so Fork here forks the project's main thread — the one
// thing the global Main chat cannot do (A3).
export async function openProjectRowMenu(project, {
    apiClient, anchorEl, onChanged, extraItemsHtml = '', onExtraSelect = null,
} = {}) {
    const maxNameLength = PROJECT_NAME_MAX;
    // `extraItemsHtml`/`onExtraSelect` let a caller append project-level rows this
    // module does not own — today the archived-thread list (T4), which belongs to
    // the thread vocabulary in `project_threads.js`. Passed IN rather than
    // imported, because that module already imports `openRowMenu` from here and an
    // import back would close a cycle. `onExtraSelect` returning true means "this
    // was mine"; anything else falls through to the rows below.
    openRowMenu({
        anchorEl,
        ariaLabel: `Actions for ${project.name || project.id}`,
        itemsHtml: `
        <button type="button" role="menuitem" data-prm="rename">Rename…</button>
        <button type="button" role="menuitem" data-prm="fork">Fork thread</button>
        ${extraItemsHtml}
        <button type="button" role="menuitem" class="danger" data-prm="delete">Delete project…</button>
    `,
        onSelect: async (action) => {
            if (onExtraSelect && (await onExtraSelect(action)) === true) return;
            await runProjectRowAction(action, project, {
                apiClient, anchorEl, onChanged, maxNameLength,
            });
        },
    });
}

async function runProjectRowAction(action, project, { apiClient, anchorEl, onChanged, maxNameLength }) {
    if (action === 'rename') {
        const res = await openConfirmDialog({
            title: 'Rename project',
            body: `New name for “${project.name || project.id}”:`,
            input: true,
            initialValue: project.name || project.id,
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
                    body: `Rename failed: ${e?.body?.error || e?.message || e}`,
                    alert: true,
                });
                // A failure is a reason to RE-READ, not to stop: the commonest one
                // is a 404 for a project another tab just deleted, so the row we
                // are painting no longer exists. Without this it survives the alert
                // and stays clickable until the next poll tick.
                onChanged?.({ authoritative: true });
            }
        }
        if (anchorEl.isConnected) anchorEl.focus();
    } else if (action === 'delete') {
        const ok = await openConfirmDialog({
            title: 'Delete project',
            // The last sentence is the one that was missing (I1): a project's
            // threads can each own a git CHECKOUT and a `thread/<name>` branch,
            // and a tombstoned project leaves no surface that could reach them —
            // so they go with it. Saying "your working folder is preserved" and
            // nothing else described a deletion that also destroyed N folders.
            body: `Delete “${project.name || project.id}”? Running work will be cancelled. The Project will be removed from the active UI; its id, chat history, task bindings, memory, and working folder are preserved. Any checkouts its threads branched off — and their thread/… branches — are removed with it; if one still holds work the project folder never received, the delete stops and says which thread.`,
            confirmLabel: 'Delete',
            danger: true,
        });
        if (ok === true) {
            onChanged?.({ projectId: project.id, lifecycle: 'deleting', optimistic: true });
            try { await apiClient.projectDelete(project.id); }
            catch (e) {
                await openConfirmDialog({
                    title: 'Delete did not finish',
                    body: `Delete did not finish: ${e?.body?.error || e?.message || e}`,
                    alert: true,
                });
            }
            finally { onChanged?.({ authoritative: true }); }
        }
        if (anchorEl.isConnected) anchorEl.focus();
    } else if (action === 'fork') {
        // Thread #0 of a project — NOT the global Main chat, which has no
        // forkable spelling. The fork stores a cursor into this thread's
        // rows; nothing is copied and the source is untouched (A3).
        try {
            const payload = await apiClient.projectThreadFork(project.id, 0);
            onChanged?.({ authoritative: true, thread: payload?.thread || null });
        } catch (e) {
            await openConfirmDialog({
                title: 'Fork failed',
                body: `Fork failed: ${e?.body?.error || e?.message || e}`,
                alert: true,
            });
            onChanged?.({ authoritative: true });  // same reason as rename
        }
        if (anchorEl.isConnected) anchorEl.focus();
    }
}
