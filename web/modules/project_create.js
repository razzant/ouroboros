// New Project dialog + project row actions (v6.59.0, Phase 3).
//
// One modal, four sources (quiz-approved UX): file-less | genesis (fresh managed
// folder) | attach an existing folder (server-side directory browser — works in
// web/Docker where no native picker exists; optional init_git attach-snapshot)
// | clone a git URL (server-side, typed auth_required). The dialog carries the
// honest trust line: attaching gives the agent write+shell in that folder
// (notification model — attaching IS the grant; no second confirmation).
import { escapeHtmlAttr as escapeHtml } from './utils.js';
import { openConfirmDialog } from './confirm_dialog.js';

export function openNewProjectDialog({ apiClient, onCreated }) {
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
                        <input class="files-modal-input" data-np-name type="text" placeholder="My project" maxlength="80">
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

// Row menu: rename / hide / delete. `project` is a sidebar row; callbacks refresh.
export async function openProjectRowMenu(project, { apiClient, anchorEl, onChanged, onHide }) {
    document.querySelectorAll('.project-row-menu').forEach((el) => el.remove());
    const menu = document.createElement('div');
    menu.className = 'project-row-menu';
    menu.innerHTML = `
        <button type="button" data-prm="rename">Rename…</button>
        <button type="button" data-prm="hide">Hide from sidebar</button>
        <button type="button" class="danger" data-prm="delete">Delete project…</button>
    `;
    const rect = anchorEl.getBoundingClientRect();
    menu.style.setProperty('--prm-top', `${Math.round(rect.bottom + 4)}px`);
    menu.style.setProperty('--prm-left', `${Math.round(rect.left)}px`);
    const close = () => { menu.remove(); document.removeEventListener('click', onDoc, true); };
    const onDoc = (event) => { if (!menu.contains(event.target)) close(); };
    document.addEventListener('click', onDoc, true);
    menu.addEventListener('click', async (event) => {
        const action = event.target.closest('[data-prm]')?.dataset?.prm;
        if (!action) return;
        close();
        if (action === 'rename') {
            const res = await openConfirmDialog({
                title: 'Rename project',
                body: `New name for “${project.name || project.id}”:`,
                input: true,
                initialValue: project.name || project.id,
                confirmLabel: 'Rename',
            });
            const newName = res?.confirmed ? String(res.value || '').trim() : '';
            if (newName && newName !== project.name) {
                try { await apiClient.projectUpdate(project.id, newName); onChanged?.(); }
                catch (e) { alert(`Rename failed: ${e?.body?.error || e?.message || e}`); }
            }
        } else if (action === 'hide') {
            onHide?.(project.id);
        } else if (action === 'delete') {
            const ok = await openConfirmDialog({
                title: 'Delete project',
                body: `Delete “${escapeHtml(project.name || project.id)}” from Ouroboros? The chat history entry is unregistered and task bindings are removed. The working folder and its files are NOT touched.`,
                confirmLabel: 'Delete',
                danger: true,
            });
            if (ok === true) {
                try { await apiClient.projectDelete(project.id); onChanged?.(); }
                catch (e) { alert(`Delete failed: ${e?.body?.error || e?.message || e}`); }
            }
        }
    });
    document.body.appendChild(menu);
}
