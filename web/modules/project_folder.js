/**
 * A Project's working folder: the folder browser the New Project dialog shares, and the Project menu's
 * Working folder… dialog (docs/DESIGN.md "Controls and editable choices").
 *
 * The dialog shows what one read of `GET /api/projects/{id}/folder` gives: the registered folder, what one look at it
 * found, and the changes that state offers. A folder in place is only shown. Where the project has no usable default
 * (its folder is missing or unusable, no choice was made yet, or No folder is recorded) the owner records No folder or
 * attaches an existing folder; nothing is created, copied or deleted here. Before saving, the dialog says what the
 * change means for the project's work: work already running or waiting keeps its folder, and only work moved into
 * the project that has not restarted since takes the new default at its next start. A change sends the read's
 * `expected` back, so a choice that moved meanwhile is read again, never overwritten.
 */
import { fmt } from './i18n.js';
import { errorWords, setInlineStatus, setResult } from './ui_primitives.js';
import { bindDialogFocus } from './ui_interactions.js';
import { showToast } from './toast.js';

// What one look found, as a status sentence (ui.css .ui-status): its tone and its words.
const STATES = {
    usable: ['ok', 'This folder is in place. New tasks in this project start there.'],
    missing: ['danger', 'This folder is no longer on this computer.'],
    unusable: ['danger', 'This folder cannot be used as a working folder.'],
    unset: ['muted', 'No folder chosen yet. The first task started in this project would create one.'],
    none: ['muted', 'No folder. Tasks in this project work in their own temporary area.'],
};
// The two changes, named as the New Project dialog names the same two sources, each with the line said under it.
const CHOICES = {
    clear: ['No folder', 'Tasks work in their own temporary area. Nothing creates a folder for this project until you choose one.'],
    choose: ['Existing folder…', 'A folder on this computer; Git is optional.'],
};

/** The server-side folder browser (it works in web and Docker, where no native picker exists), in `host`: the open
 *  folder with Select this folder, its subfolders, and under them the line that says what is selected. `onSelect`
 *  hears a selection and `onError` a failed read; `open(path)` reads a folder ('' is home), `current()` answers the
 *  folder last read ('' before one was, so a failed first read can be asked again), `selected()` the selection ('' for
 *  none), and `dispose()` drops answers still under way. */
export function mountFolderBrowser(host, { apiClient, onSelect, onError }) {
    host.innerHTML = `
        <div class="new-project-browser">
            <div class="new-project-browser-path" data-np-path></div>
            <div class="new-project-browser-list" data-np-dirs></div>
        </div>
        <p class="new-project-selected-target ui-field-help" data-np-selected role="status" aria-live="polite">No folder selected.</p>`;
    const [pathEl, listEl, selectedEl] = ['[data-np-path]', '[data-np-dirs]', '[data-np-selected]'].map((sel) => host.querySelector(sel));
    let selectedDir = '';
    let currentDir = '';
    let request = 0;
    let disposed = false;
    async function open(path) {
        const mine = ++request;
        listEl.textContent = 'Loading…';
        try {
            const data = await apiClient.fsDirs(path);
            if (disposed || mine !== request) return;
            currentDir = data.path;
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
                selectedEl.textContent = `Selected folder: ${selectedDir}`;
                select.classList.replace('btn-default', 'btn-primary');
                select.textContent = 'Selected ✓';
                onSelect?.(selectedDir);
            });
            pathEl.appendChild(select);
            listEl.textContent = '';
            if (data.parent) {
                const up = document.createElement('button');
                up.type = 'button';
                up.className = 'new-project-dir-row new-project-dir-up';
                up.textContent = '.. (up)';
                up.addEventListener('click', () => open(data.parent));
                listEl.appendChild(up);
            }
            for (const dir of data.dirs || []) {
                const row = document.createElement('button');
                row.type = 'button';
                row.className = 'new-project-dir-row';
                row.textContent = `${dir.name}${dir.is_git ? '  ⎇' : ''}`;
                row.title = dir.path + (dir.is_git ? ' (git repository)' : '');
                row.addEventListener('click', () => open(dir.path));
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
            if (disposed || mine !== request) return;
            listEl.textContent = '';
            onError?.(e);
        }
    }
    return { open, current: () => currentDir, selected: () => selectedDir, dispose() { disposed = true; } };
}

/** What saving `action` means for the project's work, one sentence a line: what is kept, what moves and where. */
export function folderEffects(view, action, selected = '') {
    const work = view.work || {};
    const lines = [];
    if (action === 'clear' && view.folder) {
        lines.push(fmt('Only the link to this folder is removed. Files, history and results stay where they are.'));
    }
    if (!work.observed) {
        lines.push(fmt('Running and waiting tasks could not be read, so it is unknown which of them keep their folder.'));
        return lines;
    }
    if (work.kept) {
        lines.push(fmt(work.kept === 1 ? '{n} task already running or waiting keeps its folder. It is not moved.'
            : '{n} tasks already running or waiting keep their folders. They are not moved.', { n: work.kept }));
    }
    // Moved work takes the project's default at its next start. Work on Ouroboros's own repository is said apart: a
    // chosen folder would take it from there, and No folder keeps it.
    for (const own of [true, false]) {
        const names = work.moved.filter((item) => Boolean(item.system_repo) === own)
            .map((item) => item.name || item.task_id).join(', ');
        if (!names) continue;
        const folder = selected || fmt('the folder you choose');
        if (own) {
            lines.push(action === 'clear'
                ? fmt("Moved here and not restarted yet: {names}. The next start stays on Ouroboros's own repository.", { names })
                : fmt("Moved here and not restarted yet: {names}. The next start would work in {folder}, not on Ouroboros's own repository where it ran. No folder keeps it there.", { names, folder }));
        } else {
            lines.push(action === 'clear'
                ? fmt('Moved here and not restarted yet: {names}. The next start gets no folder from this project and works in its own temporary area, even if it already ran elsewhere.', { names })
                : fmt('Moved here and not restarted yet: {names}. The next start works in {folder}, even if it already ran elsewhere.', { names, folder }));
        }
    }
    const more = work.moved_total - work.moved.length;
    if (more > 0) {
        lines.push(fmt(more === 1 ? '{n} more moved task is not listed; the same holds for it.'
            : '{n} more moved tasks are not listed; the same holds for them.', { n: more }));
    }
    return lines;
}

/** Open the dialog for `project`. True confirms a change; false means dismissed, not proof that nothing changed. */
export function openProjectFolderDialog({ apiClient, project }) {
    return new Promise((resolve) => {
        const backdrop = document.createElement('div');
        backdrop.className = 'marketplace-modal-backdrop new-project-backdrop';
        backdrop.innerHTML = `
            <div class="marketplace-modal new-project-dialog project-folder-dialog" role="dialog" aria-modal="true" aria-labelledby="project-folder-title">
                <div class="marketplace-modal-head">
                    <h3 id="project-folder-title">Working folder</h3>
                </div>
                <div class="marketplace-modal-body">
                    <div class="new-project-field">
                        <div class="ui-field-help" data-pf-name data-i18n-skip></div>
                        <div class="project-folder-path" data-pf-path data-i18n-skip hidden></div>
                        <div class="ui-status" data-pf-state role="status">Reading the folder…</div>
                        <div class="ui-field-help" data-pf-chosen data-i18n-skip hidden></div>
                        <details class="ui-disclosure" data-pf-reason hidden>
                            <summary>Details</summary>
                            <pre class="ui-result-details" data-i18n-skip></pre>
                        </details>
                    </div>
                    <fieldset class="new-project-sources new-project-field" data-pf-choice hidden>
                        <legend>Change to</legend>
                        <div class="new-project-source-choices" data-pf-options></div>
                        <div class="ui-field-help" data-pf-help hidden></div>
                    </fieldset>
                    <div data-pf-browser hidden></div>
                    <p class="new-project-trust-note" data-pf-trust hidden>
                        The agent gets <strong>read, write and shell</strong> in this folder when working on this project's tasks.
                    </p>
                    <div class="new-project-field project-folder-effects" data-pf-effects data-i18n-skip hidden></div>
                    <div class="new-project-error ui-result" data-pf-error role="alert"></div>
                </div>
                <div class="marketplace-modal-actions">
                    <button type="button" class="btn btn-default" data-pf-cancel>Close</button>
                    <button type="button" class="btn" data-primary data-pf-save hidden disabled>Save</button>
                </div>
            </div>
        `;
        const q = (sel) => backdrop.querySelector(sel);
        q('[data-pf-name]').textContent = project.name || project.id;
        let view = null;      // the last read: what Save compares and what the effects are told from
        let action = '';      // '' until a change is chosen
        let settled = false;
        let saving = false;
        let disposeFocus = () => {};
        const setError = (result) => setResult(q('[data-pf-error]'), result || {});
        const browser = mountFolderBrowser(q('[data-pf-browser]'), {
            apiClient,
            onSelect: () => { setError(null); sync(); },
            onError: (e) => setError(errorWords(e)),
        });

        function sync() {
            const choosing = action === 'choose';
            const help = q('[data-pf-help]');
            help.textContent = CHOICES[action]?.[1] || '';
            help.hidden = !help.textContent;
            q('[data-pf-browser]').hidden = !choosing;
            q('[data-pf-trust]').hidden = !(choosing && browser.selected());
            if (choosing && !browser.current()) browser.open('');  // home, the first time Existing folder… shows
            const lines = action && view ? folderEffects(view, action, choosing ? browser.selected() : '') : [];
            const effects = q('[data-pf-effects]');
            effects.replaceChildren(...lines.map((line) => {
                const row = document.createElement('div');
                row.className = 'ui-field-help';
                row.textContent = line;
                return row;
            }));
            effects.hidden = !lines.length;
            const save = q('[data-pf-save]');
            // The button says what pressing it does: unlink the registered folder, keep none, or use the selected one.
            save.textContent = saving ? 'Saving…' : choosing ? 'Use this folder'
                : action === 'clear' ? (view?.folder ? 'Remove folder' : 'Keep no folder') : 'Save';
            save.disabled = saving || !view || !(action === 'clear' || (choosing && browser.selected()));
            q('[data-pf-choice]').disabled = saving || !view;
            q('[data-pf-browser]').inert = saving || !view;
            q('[data-pf-cancel]').textContent = saving ? 'Close' : (view?.actions || []).length ? 'Cancel' : 'Close';
        }

        function paint(next) {
            view = next;
            const [tone, words] = STATES[view.state] || ['muted', ''];
            const path = q('[data-pf-path]');
            path.textContent = view.folder;
            path.hidden = !view.folder;
            setInlineStatus(q('[data-pf-state]'), words, tone);
            const chosen = q('[data-pf-chosen]');
            const at = view.no_folder?.at ? new Date(view.no_folder.at) : null;
            chosen.hidden = !at || Number.isNaN(at.getTime());
            if (!chosen.hidden) {
                chosen.textContent = fmt(view.no_folder.by === 'owner' ? 'Chosen by you on {date}.' : 'Chosen by Ouroboros on {date}.',
                    { date: at.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) });
            }
            const reason = q('[data-pf-reason]');
            reason.hidden = !view.reason;
            reason.querySelector('pre').textContent = view.reason || '';
            const offered = (view.actions || []).filter((name) => CHOICES[name]);
            // One offered change is the dialog's one act (a folder for a project that keeps none): it stands chosen.
            action = offered.length === 1 ? offered[0] : offered.includes(action) ? action : '';
            const options = q('[data-pf-options]');
            options.replaceChildren(...offered.map((name) => {
                const label = document.createElement('label');
                const radio = document.createElement('input');
                radio.type = 'radio';
                radio.className = 'ui-checkbox';
                radio.name = 'pf-action';
                radio.value = name;
                radio.checked = name === action;
                radio.addEventListener('change', () => { action = name; setError(null); sync(); });
                label.append(radio, ` ${CHOICES[name][0]}`);
                return label;
            }));
            q('[data-pf-choice]').hidden = !offered.length;
            q('[data-pf-save]').hidden = !offered.length;
            const cancel = q('[data-pf-cancel]');
            cancel.textContent = offered.length ? 'Cancel' : 'Close';
            sync();
            // A control that held the keyboard may be gone now (Save, once nothing is offered): the dialog keeps it.
            if (!q('[role="dialog"]').contains(document.activeElement)) cancel.focus();
        }

        async function load() {
            view = null;  // A failed read never leaves an old expected word available for another Save.
            sync();
            try {
                const data = await apiClient.projectFolder(project.id);
                if (!settled) paint(data.folder);
            } catch (e) {
                if (settled) return;
                setInlineStatus(q('[data-pf-state]'), '', 'muted');
                setError(errorWords(e));
            }
        }

        const finish = (changed) => {
            if (settled) return;
            if (saving && !changed) {
                showToast(fmt('Save has not been confirmed. Closing does not cancel it. Reopen Working folder to read its current state.'),
                    'warn', { ttl: 0 });
            }
            settled = true;
            browser.dispose();
            disposeFocus();
            backdrop.remove();
            resolve(changed);
        };

        async function save() {
            const button = q('[data-pf-save]');
            if (settled || saving || button.disabled || !view) return;
            saving = true;
            sync();
            q('[role="dialog"]').focus();  // a disabled button hears no key
            setError({ text: 'Saving… You can close this dialog; closing does not cancel the change.', tone: 'info' });
            try {
                await apiClient.projectFolderChange(project.id, {
                    action, expected: view.expected, ...(action === 'choose' ? { path: browser.selected() } : {}),
                });
                finish(true);
            } catch (e) {
                if (settled) return;  // A late failure cannot paint or refocus a dialog the owner left.
                const status = Number(e?.status);
                if (status >= 400 && status < 500) saving = false;  // A refusal is known, even while its fresh read waits.
                if (status === 409) {
                    // The choice moved since it was read (another window, Ouroboros): show it as it is now.
                    await load();
                    if (!settled) setError({ text: view
                        ? 'The folder changed meanwhile. This is how it is now.'
                        : 'The folder changed meanwhile, but could not be read. Close and reopen to check it.',
                    tone: 'warn', details: errorWords(e).details });
                } else if (!(status >= 400 && status < 500)) {
                    // A lost reply says nothing about the write. Read before offering another explicit Save;
                    // no automatic resend, even if this read still shows the previous choice.
                    await load();
                    if (!settled) setError({ text: view
                        ? 'Save was not confirmed. This is the folder now; the earlier change may still finish.'
                        : 'Save was not confirmed and the folder could not be read. Close and reopen to check it.',
                    tone: 'warn', details: errorWords(e).details });
                } else {
                    setError(errorWords(e));
                }
            } finally {
                saving = false;
                if (!settled) sync();
            }
        }

        backdrop.addEventListener('click', (event) => {
            if (event.target === backdrop || event.target.closest('[data-pf-cancel]')) finish(false);
            else if (event.target.closest('[data-pf-save]')) save();
        });
        document.body.appendChild(backdrop);
        disposeFocus = bindDialogFocus(q('[role="dialog"]'), {
            initialFocus: q('[data-pf-cancel]'),
            onEscape: () => finish(false),
        });
        load();
    });
}
