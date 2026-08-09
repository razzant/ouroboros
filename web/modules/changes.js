/**
 * Changes screen: review one task's diff and ask for edits.
 *
 * Left rail = the recent task list (newest first) with a project badge; middle =
 * the file list of the selected task, derived from the patch bytes by
 * `patch_parse.js`; main pane = the unified or split renderer over the same
 * parsed hunks; bottom dock = an ordered composer-parts field whose "Request
 * edits" hands the parts to the chat controller.
 *
 * Two honesty rules shape this module:
 *   1. The file list, per-file status and +/- counts come from the SAME patch the
 *      renderer shows — never a second server-side stat source that could
 *      disagree with what the owner is reading.
 *   2. A diff the server could not produce is disclosed, never smoothed over: a
 *      `pending` task says its artifacts are not finalized, a `blocked` one names
 *      its typed blockers, and drift says HEAD moved — no page ever implies "no
 *      changes" when the truth is "we could not tell".
 *
 * There is deliberately NO approve action here (owner-locked scope): the only
 * action is asking the agent for edits.
 *
 * ⌘L capture: the dock is a `[data-capture-dock]` composer-parts field, and this
 * page owns the capture for `page === 'changes'`. The CANCELABLE
 * `ouro:capture-selection` event is the ONE seam — nothing outside this module
 * holds a handle into it. The global handler names the active page in that event
 * and this page consuming it (`preventDefault()`) is what licenses suppressing
 * the browser default. A diff selection becomes a chip whose range is the
 * NEW-side line numbers found anywhere in the selection, because those are the
 * lines that exist in the file the agent will edit; a selection with no new-side
 * number at all is quoted as text instead (see `diffChipDecision`).
 */

import { renderPageHeader } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';
import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { showToast } from './toast.js';
import { apiClient } from './api_client.js';
import { createComposerParts, makeChipPart, makeTextPart, normalizeParts } from './composer_parts.js';
import { fileStatusLetter, parsePatch, splitRows, unifiedRows } from './patch_parse.js';

// The rail is a REVIEW surface, not a task log: 30 rows is what fits without
// turning the newest-first list into a scroll hunt (the inspector and Chat own
// the long history).
const TASK_RAIL_LIMIT = 30;

/** The one drift sentence (owner-locked wording). */
export const HEAD_DRIFT_NOTICE = 'HEAD differs from the task baseline; showing the '
    + 'current projection for paths attributed during the task window';

// ---------------------------------------------------------------------------
// Pure presentation helpers (node-tested)
// ---------------------------------------------------------------------------

/**
 * Resolve a task's project badge.
 *
 * `task.project_id` is the task's OWN stored scope and wins. A task that was
 * bound to a project after the fact (the "turn into project" path) carries the
 * binding only in `/api/state.task_bindings`, so that is the fallback. A project
 * whose name is not in the (capped) sidebar summary still shows its id rather
 * than nothing — an unnamed badge is better than a silently unscoped task.
 *
 * @returns {{projectId: string, label: string}|null}
 */
export function taskProjectBadge(task, { taskBindings = {}, projects = [] } = {}) {
    const taskId = String(task?.task_id || task?.id || '');
    const own = String(task?.project_id || '').trim();
    const bound = String((taskBindings && taskBindings[taskId] && taskBindings[taskId].project_id) || '').trim();
    const projectId = own || bound;
    if (!projectId) return null;
    const row = (Array.isArray(projects) ? projects : []).find((p) => p && p.id === projectId);
    return { projectId, label: String((row && row.name) || projectId) };
}

/** Short human title for a task row / the "Re task …" line. */
export function taskShortTitle(task) {
    const candidates = [task?.title, task?.objective, task?.description, task?.text];
    for (const value of candidates) {
        const text = String(value || '').trim().replace(/\s+/g, ' ');
        if (text) return text.length > 72 ? `${text.slice(0, 71)}…` : text;
    }
    return String(task?.task_id || task?.id || 'task');
}

/** The one neutral sentence for a task that simply has no recorded baseline. */
export const NO_BASELINE_NOTICE = 'No diff baseline was recorded for this task';

/**
 * Is this `blocked` answer just "there was never a baseline to diff against"?
 *
 * `baseline_missing` (and a blocked answer carrying no blockers at all) is an
 * ABSENCE of evidence, not a failure to trust: most tasks that never touched the
 * repo land here. Dressing it up as "no trustworthy diff" teaches the owner to
 * read alarm into the ordinary case, so it gets its own neutral wording.
 */
export function diffLacksBaselineOnly(diff) {
    if (String(diff?.status || '') !== 'blocked') return false;
    const blockers = Array.isArray(diff?.blockers) ? diff.blockers.filter(Boolean) : [];
    return blockers.length === 0 || (blockers.length === 1 && blockers[0] === 'baseline_missing');
}

/**
 * The rail/header meta line for one loaded diff: `N files · +A −R`.
 * A diff that is not `ready` reports its lifecycle instead of a fake `0 files`.
 */
export function diffSummaryMeta(diff, parsed) {
    const status = String(diff?.status || '');
    if (status === 'pending') return 'waiting for the task to finalize its changes';
    if (status === 'blocked') return diffLacksBaselineOnly(diff) ? 'no diff baseline recorded' : 'diff unavailable';
    if (status === 'empty') return 'no changes';
    const files = parsed?.files?.length || 0;
    return `${files} file${files === 1 ? '' : 's'} · +${parsed?.added || 0} −${parsed?.removed || 0}`;
}

/**
 * Banner rows for one diff response: pending / blocked / drift.
 * Returns [] when there is nothing to disclose. Tone drives the CSS token only.
 */
export function diffBanners(diff) {
    const rows = [];
    const status = String(diff?.status || '');
    const blockers = Array.isArray(diff?.blockers) ? diff.blockers.filter(Boolean) : [];
    if (status === 'pending') {
        rows.push({
            tone: 'pending',
            text: 'This task has not finalized its changes yet. The diff appears once its '
                + 'artifacts are written.',
        });
    }
    if (status === 'blocked') {
        // A missing baseline is an absence, not a broken read: it gets the neutral
        // sentence and no blocker code, because `baseline_missing` is not an
        // owner-actionable fault.
        rows.push(diffLacksBaselineOnly(diff)
            ? { tone: 'neutral', text: NO_BASELINE_NOTICE }
            : {
                tone: 'blocked',
                text: 'No trustworthy diff can be shown for this task.',
                detail: blockers.join(', '),
            });
    }
    // Drift is a LIVE-projection fact: it means the repo's HEAD moved away from the
    // baseline the patch is taken against. A workspace task's patch is durable
    // bytes captured at its own base, so `head_advanced` there would describe a
    // repo this patch does not depend on — the sentence would be a non-sequitur.
    if (diff?.head_advanced && String(diff?.source || '') === 'mutation_baseline') {
        rows.push({ tone: 'drift', text: HEAD_DRIFT_NOTICE });
    }
    if (status !== 'blocked' && blockers.length) {
        rows.push({ tone: 'evidence', text: 'Attribution notes', detail: blockers.join(', ') });
    }
    return rows;
}

/**
 * Selected diff rows -> the chip a capture will build. PURE, because the rule is
 * a DECISION about what the bytes may CLAIM, not a DOM fact.
 *
 * Rules, in order:
 *   • no selection at all is the whole-file chip: `[context: <path>]`, no range,
 *     no bytes — the agent opens the file itself;
 *   • a selection is CLAMPED to the hunk its START row belongs to. Two rows in
 *     hunks hundreds of lines apart would otherwise be named `L18-L401`: a
 *     384-line claim for a 4-line selection. The bytes are clamped with the
 *     range (a chip must carry exactly what it names), and `clampedRows` reports
 *     how many rows were dropped so the caller can disclose it.
 *   • the range is min/max over every NEW-side line number present ANYWHERE in
 *     the clamped selection — not the boundary rows. Those numbers are the lines
 *     that exist in the file the agent will edit, and a selection that merely
 *     STARTS or ENDS on a `-` row still has a TRUE new-side span; the deleted
 *     lines ride verbatim (prefixes included) inside the fenced content, so the
 *     agent sees exactly what was removed around the lines the range names.
 *     Reading the range off the boundaries instead threw the whole selection
 *     away whenever an owner highlighted from a deletion, which is the common
 *     shape of "this removal is wrong".
 *   • a selection with NO new-side number anywhere is a PURE deletion: nothing
 *     in the new file can name it, so `lineStart`/`lineEnd` stay null and the
 *     content is still returned. The caller quotes it as TEXT
 *     (`deletionQuoteText`) rather than letting the codec's range-only-bytes
 *     rule silently degrade it to a whole-file marker that lost the selection.
 *
 * @param {{path?: string, rows?: Array<{newNumber?: *, text?: string,
 *          hunkIndex?: *}>}} input
 *        `rows` are the selected diff lines in document order, `text` verbatim,
 *        `hunkIndex` the render-order index of the hunk each row came from
 *        (omitted by callers that do not track hunks — then nothing is clamped).
 */
export function diffChipDecision({ path = '', rows = [] } = {}) {
    const all = (Array.isArray(rows) ? rows : []).filter(Boolean);
    if (!all.length) return { path, lineStart: null, lineEnd: null, content: null, clampedRows: 0 };
    const hunkOf = (row) => {
        const num = Number(row?.hunkIndex);
        return Number.isInteger(num) && num >= 0 ? num : null;
    };
    // The anchor is the START row's hunk. A caller that tracks no hunks at all
    // clamps nothing — dropping every row would be worse than a wide range.
    const anchor = hunkOf(all[0]);
    const selected = anchor === null ? all : all.filter((row) => hunkOf(row) === anchor);
    const clampedRows = all.length - selected.length;
    const content = selected.map((row) => String(row.text == null ? '' : row.text)).join('\n');
    const numbers = selected
        .map((row) => Number(row?.newNumber))
        .filter((num) => Number.isInteger(num) && num > 0);
    if (!numbers.length) return { path, lineStart: null, lineEnd: null, content, clampedRows };
    return {
        path,
        lineStart: Math.min(...numbers),
        lineEnd: Math.max(...numbers),
        content,
        clampedRows,
    };
}

/**
 * A pure-deletion selection as a path-bearing TEXT quote, or '' when there is
 * nothing to quote.
 *
 * The bytes are exactly what the owner highlighted; what does not exist is a
 * new-side line number to name them by. A text part keeps BOTH the bytes and the
 * path, so the agent can locate the removal by content — strictly more than the
 * bare whole-file marker the codec's range-only-bytes rule would have left.
 *
 * Round-trip safety is structural, not hopeful: every line here is a `-` diff
 * line (a context or added line would have carried a new-side number), so no
 * line can be read back as a `[context: …]` marker or as a bare ``` fence, and
 * `parseContent(serializeParts(...))` returns the same single text part.
 */
export function deletionQuoteText({ path = '', content = '' } = {}) {
    const label = String(path == null ? '' : path).trim();
    const body = typeof content === 'string' ? content : '';
    if (!label || !body) return '';
    return `${label} — deleted lines (no new-side line numbers):\n${body}`;
}

/** The plain-text task line prepended to a "Request edits" handoff. */
export function requestEditsPrefix(task) {
    const taskId = String(task?.task_id || task?.id || '');
    return `Re task ${taskId} ("${taskShortTitle(task)}"): `;
}

/**
 * Ordered parts for the handoff: the task line, then everything in the dock.
 * The dock's own order is preserved — a chip/comment interleaving is the message.
 */
export function requestEditsParts(task, dockParts) {
    const prefix = makeTextPart(requestEditsPrefix(task));
    return normalizeParts([prefix, ...(Array.isArray(dockParts) ? dockParts : [])].filter(Boolean));
}

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

function renderShell() {
    return `
        ${renderPageHeader({
            title: 'Changes',
            icon: PAGE_ICONS.changes || '',
            actionsHtml: '<button class="btn btn-default" data-changes-refresh>Refresh</button>',
        })}
        <div class="changes-layout">
            <aside class="changes-rail">
                <div class="changes-rail-head">
                    <div class="changes-rail-title">Tasks</div>
                    <div class="changes-rail-meta" data-changes-rail-meta></div>
                </div>
                <div class="changes-task-list scroll-fade-y" data-changes-task-list></div>
                <div class="changes-file-head">
                    <div class="changes-rail-title">Files</div>
                    <div class="changes-rail-meta" data-changes-file-meta></div>
                </div>
                <div class="changes-file-list scroll-fade-y" data-changes-file-list></div>
            </aside>
            <section class="changes-main">
                <div class="changes-main-head">
                    <div class="changes-path" data-changes-path></div>
                    <div class="changes-counts" data-changes-counts></div>
                    <div class="ui-segment-group changes-mode" role="group" aria-label="Diff view">
                        <button type="button" class="ui-segment active" data-changes-mode="unified">Unified</button>
                        <button type="button" class="ui-segment" data-changes-mode="split">Split</button>
                    </div>
                </div>
                <div class="changes-banners" data-changes-banners></div>
                <div class="changes-diff scroll-fade-y" data-changes-diff></div>
                <button type="button" class="changes-selection-btn" data-changes-capture-selection hidden>
                    Add selection <kbd class="changes-kbd">⌘L</kbd>
                </button>
                <form class="changes-dock" data-changes-dock>
                    <div class="changes-dock-field" data-changes-dock-field data-capture-dock>
                        <input
                            type="text"
                            class="changes-dock-input"
                            data-changes-dock-input
                            placeholder="⌘L adds lines from the diff, type comments between · Enter sends"
                            aria-label="Request edits message"
                        >
                    </div>
                    <button type="submit" class="btn btn-primary" data-changes-request>Request edits</button>
                </form>
            </section>
        </div>
    `;
}

function statusClass(letter) {
    return { A: 'is-added', D: 'is-deleted', R: 'is-renamed' }[letter] || 'is-modified';
}

export function initChanges(ctx = {}) {
    const { showPage, subscribeState, getChatController } = ctx;
    const page = document.getElementById('page-changes');
    if (!page) return null;
    page.classList.add('app-page-glass');
    page.innerHTML = renderShell();

    const railMetaEl = page.querySelector('[data-changes-rail-meta]');
    const taskListEl = page.querySelector('[data-changes-task-list]');
    const fileMetaEl = page.querySelector('[data-changes-file-meta]');
    const fileListEl = page.querySelector('[data-changes-file-list]');
    const pathEl = page.querySelector('[data-changes-path]');
    const countsEl = page.querySelector('[data-changes-counts]');
    const bannersEl = page.querySelector('[data-changes-banners]');
    const diffEl = page.querySelector('[data-changes-diff]');
    const dockForm = page.querySelector('[data-changes-dock]');
    const dockField = page.querySelector('[data-changes-dock-field]');
    const dockInput = page.querySelector('[data-changes-dock-input]');
    const selectionBtn = page.querySelector('[data-changes-capture-selection]');

    const view = {
        tasks: [],
        taskId: '',
        task: null,
        diff: null,
        parsed: { files: [], added: 0, removed: 0 },
        filePath: '',
        mode: 'unified',
        error: '',
        loading: false,
        // The one /api/state snapshot (project names + post-hoc task bindings).
        projects: [],
        taskBindings: {},
        // Capture state. `captureRows` is the CURRENT render's line list — one entry
        // per canonical diff line, in document order — and every selectable text
        // cell carries its index in `data-diff-row`. Reading the bytes from here
        // rather than from the DOM keeps the captured content verbatim: the rendered
        // cell may also hold the `⏎̸` no-newline marker, which is presentation.
        captureRows: [],
        // The last non-empty selection span, for the two callers allowed to read a
        // collapsed selection (the sticky button, and ⌘L pressed inside the dock —
        // focusing the dock is what collapsed it).
        lastSelection: null,
    };

    const dock = createComposerParts({ container: dockField, input: dockInput });

    function paintTaskRail() {
        railMetaEl.textContent = view.tasks.length
            ? `${view.tasks.length} recent`
            : (view.error ? 'unavailable' : 'no tasks yet');
        taskListEl.textContent = '';
        if (view.error) {
            const row = document.createElement('div');
            row.className = 'changes-empty';
            row.textContent = view.error;
            taskListEl.appendChild(row);
            return;
        }
        if (!view.tasks.length) {
            const row = document.createElement('div');
            row.className = 'changes-empty';
            row.textContent = 'No tasks have run yet.';
            taskListEl.appendChild(row);
            return;
        }
        for (const task of view.tasks) {
            const taskId = String(task.task_id || task.id || '');
            const badge = taskProjectBadge(task, {
                taskBindings: view.taskBindings, projects: view.projects,
            });
            const button = document.createElement('button');
            button.type = 'button';
            button.className = `changes-task-row${taskId === view.taskId ? ' active' : ''}`;
            button.dataset.changesTask = taskId;
            button.title = taskShortTitle(task);
            button.innerHTML = `
                <span class="changes-task-title">${escapeHtml(taskShortTitle(task))}</span>
                <span class="changes-task-meta">
                    <span class="changes-task-id">${escapeHtml(taskId)}</span>
                    ${badge ? `<span class="changes-task-project" title="${escapeHtmlAttr(badge.projectId)}">${escapeHtml(badge.label)}</span>` : ''}
                    <span class="changes-task-status">${escapeHtml(String(task.status || ''))}</span>
                </span>
            `;
            taskListEl.appendChild(button);
        }
    }

    function paintFileList() {
        fileMetaEl.textContent = view.diff ? diffSummaryMeta(view.diff, view.parsed) : '';
        fileListEl.textContent = '';
        if (!view.taskId) {
            const row = document.createElement('div');
            row.className = 'changes-empty';
            row.textContent = 'Pick a task to review its changes.';
            fileListEl.appendChild(row);
            return;
        }
        if (view.loading) {
            const row = document.createElement('div');
            row.className = 'changes-empty';
            row.textContent = 'Loading diff…';
            fileListEl.appendChild(row);
            return;
        }
        if (!view.parsed.files.length) {
            const row = document.createElement('div');
            row.className = 'changes-empty';
            row.textContent = diffSummaryMeta(view.diff, view.parsed);
            fileListEl.appendChild(row);
            return;
        }
        for (const file of view.parsed.files) {
            const letter = fileStatusLetter(file);
            const button = document.createElement('button');
            button.type = 'button';
            button.className = `changes-file-row${file.path === view.filePath ? ' active' : ''}`;
            button.dataset.changesFile = file.path;
            button.title = file.renamed ? `${file.oldPath} → ${file.path}` : file.path;
            button.innerHTML = `
                <span class="changes-file-status ${statusClass(letter)}">${escapeHtml(letter)}</span>
                <span class="changes-file-path">${escapeHtml(file.path)}</span>
                <span class="changes-file-counts">${
                    file.binary
                        ? '<span class="changes-file-binary">bin</span>'
                        : `<span class="changes-add">+${file.added}</span><span class="changes-del">−${file.removed}</span>`
                }</span>
            `;
            fileListEl.appendChild(button);
        }
    }

    function paintBanners() {
        bannersEl.textContent = '';
        for (const banner of view.diff ? diffBanners(view.diff) : []) {
            const row = document.createElement('div');
            row.className = 'changes-banner';
            row.dataset.tone = banner.tone;
            const text = document.createElement('span');
            text.className = 'changes-banner-text';
            text.textContent = banner.text;
            row.appendChild(text);
            if (banner.detail) {
                const detail = document.createElement('code');
                detail.className = 'changes-banner-detail';
                detail.textContent = banner.detail;
                row.appendChild(detail);
            }
            bannersEl.appendChild(row);
        }
    }

    function activeFile() {
        return view.parsed.files.find((file) => file.path === view.filePath) || null;
    }

    function paintDiff() {
        const file = activeFile();
        // A repaint replaces every row element, so any remembered selection span
        // points at rows that no longer exist. Both are rebuilt from this render.
        view.captureRows = [];
        view.lastSelection = null;
        if (selectionBtn) selectionBtn.hidden = true;
        // With no file to name (empty / pending / blocked diff) the header keeps the
        // task's identity instead of going blank.
        pathEl.textContent = file
            ? file.path
            : (view.taskId ? taskShortTitle(view.task || { task_id: view.taskId }) : 'Changes');
        countsEl.textContent = '';
        if (file && !file.binary) {
            const add = document.createElement('span');
            add.className = 'changes-add';
            add.textContent = `+${file.added}`;
            const del = document.createElement('span');
            del.className = 'changes-del';
            del.textContent = `−${file.removed}`;
            countsEl.append(add, del);
        }
        diffEl.textContent = '';
        diffEl.dataset.mode = view.mode;
        if (!file) {
            const empty = document.createElement('div');
            empty.className = 'changes-empty changes-diff-empty';
            empty.textContent = view.taskId
                ? diffSummaryMeta(view.diff, view.parsed)
                : 'Select a task on the left to see what it changed.';
            diffEl.appendChild(empty);
            return;
        }
        if (file.binary || !file.hunks.length) {
            const note = document.createElement('div');
            note.className = 'changes-empty changes-diff-empty';
            note.textContent = file.notes.length
                ? file.notes.join(' · ')
                : 'No textual hunks for this entry.';
            diffEl.appendChild(note);
            return;
        }
        diffEl.appendChild(view.mode === 'split' ? renderSplit(file) : renderUnified(file));
    }

    function lineCell(className, text) {
        const cell = document.createElement('div');
        cell.className = className;
        cell.textContent = text;
        return cell;
    }

    /**
     * Register one canonical diff line for capture and return its index.
     * `text` is the VERBATIM diff line (its `+`/`-`/space prefix included, no
     * presentation markers), `newNumber` is its new-side number if it has one, and
     * `hunkIndex` is the render-order index of the hunk it came from — hunk headers
     * are not capture rows, so a selection could otherwise not tell that two of its
     * rows sit hundreds of lines apart (`diffChipDecision` clamps on it).
     */
    function pushCaptureRow(newNumber, text, hunkIndex) {
        view.captureRows.push({ newNumber, text, hunkIndex });
        return view.captureRows.length - 1;
    }

    function renderUnified(file) {
        const grid = document.createElement('div');
        grid.className = 'changes-unified';
        let hunkIndex = -1;
        for (const row of unifiedRows(file)) {
            if (row.kind === 'hunk') {
                hunkIndex += 1;
                const header = document.createElement('div');
                header.className = 'changes-hunk';
                header.textContent = row.text;
                grid.appendChild(header);
                continue;
            }
            const line = document.createElement('div');
            line.className = `changes-row is-${row.kind}`;
            if (row.newNumber) line.dataset.newLine = row.newNumber;
            const text = lineCell('changes-text', row.noNewline ? `${row.text} ⏎̸` : row.text);
            text.dataset.diffRow = String(pushCaptureRow(row.newNumber, row.text, hunkIndex));
            line.append(
                lineCell('changes-num', row.oldNumber),
                lineCell('changes-num', row.newNumber),
                text,
            );
            grid.appendChild(line);
        }
        return grid;
    }

    function renderSplit(file) {
        const grid = document.createElement('div');
        grid.className = 'changes-split';
        let hunkIndex = -1;
        for (const row of splitRows(file)) {
            if (row.kind === 'hunk') {
                hunkIndex += 1;
                const header = document.createElement('div');
                header.className = 'changes-hunk';
                header.textContent = row.text;
                grid.appendChild(header);
                continue;
            }
            const line = document.createElement('div');
            line.className = 'changes-row is-split';
            // A context row shows ONE canonical line on both sides, so both cells
            // share a single capture index and a selection over it captures the line
            // once. A change row shows two different lines, so each side is its own.
            const ctx = row.kind === 'ctx';
            const leftIndex = row.left
                ? pushCaptureRow(ctx ? row.right?.number : '', `${ctx ? ' ' : '-'}${row.left.text}`, hunkIndex)
                : null;
            const rightIndex = !row.right
                ? null
                : (ctx && leftIndex !== null
                    ? leftIndex
                    : pushCaptureRow(row.right.number, `+${row.right.text}`, hunkIndex));
            const side = (cell, kind, index) => {
                const num = lineCell('changes-num', cell ? cell.number : '');
                const text = lineCell(`changes-text is-${cell ? cell.kind : 'none'}`, cell ? cell.text : '');
                if (!cell) {
                    num.classList.add('is-none');
                    text.classList.add('is-empty-counterpart');
                }
                num.classList.add(`is-${cell ? cell.kind : 'none'}`);
                num.dataset.side = kind;
                if (index !== null) text.dataset.diffRow = String(index);
                if (cell && kind === 'new') text.dataset.newLine = cell.number;
                return [num, text];
            };
            line.append(...side(row.left, 'old', leftIndex), ...side(row.right, 'new', rightIndex));
            grid.appendChild(line);
        }
        return grid;
    }

    function paintAll() {
        paintTaskRail();
        paintFileList();
        paintBanners();
        paintDiff();
    }

    async function loadTasks() {
        try {
            const data = await apiClient.tasks(TASK_RAIL_LIMIT);
            view.tasks = Array.isArray(data?.tasks) ? data.tasks : [];
            view.error = '';
        } catch (err) {
            view.tasks = [];
            view.error = `Task list unavailable: ${err?.message || 'request failed'}`;
        }
        paintTaskRail();
    }

    async function selectTask(taskId, { filePath = '' } = {}) {
        const id = String(taskId || '');
        if (!id) return;
        view.taskId = id;
        view.task = view.tasks.find((task) => String(task.task_id || task.id || '') === id) || { task_id: id };
        view.diff = null;
        view.parsed = { files: [], added: 0, removed: 0 };
        view.filePath = '';
        view.loading = true;
        paintAll();
        let diff = null;
        try {
            diff = await apiClient.taskDiff(id);
        } catch (err) {
            diff = {
                status: 'blocked',
                source: '',
                blockers: [err?.message || 'request failed'],
                patch: '',
            };
        }
        if (view.taskId !== id) return;  // a newer selection won
        view.loading = false;
        view.diff = diff;
        view.parsed = parsePatch(diff?.patch || '');
        const wanted = view.parsed.files.find((file) => file.path === filePath);
        view.filePath = (wanted || view.parsed.files[0] || {}).path || '';
        paintAll();
    }

    // -----------------------------------------------------------------------
    // ⌘L capture (DOM layer over `diffChipDecision`)
    // -----------------------------------------------------------------------

    /**
     * Does a Range boundary sit BEFORE its cell's first character?
     *
     * An ELEMENT boundary carries a CHILD INDEX, not a character offset, so the
     * two scales must never be compared. The question is answered positionally
     * instead: a probe range from the cell's start to the boundary that stringifies
     * to '' means nothing lies before it.
     */
    function boundaryAtCellStart(cell, container, offset) {
        try {
            const probe = document.createRange();
            probe.setStart(cell, 0);
            probe.setEnd(container, offset);
            return probe.toString() === '';
        } catch {
            return false;
        }
    }

    /** One Range boundary -> `{index, atStart}` of its diff line, or null. */
    function resolveDiffBoundary(container, offset) {
        const element = container instanceof Element ? container : (container?.parentElement || null);
        if (!element || !diffEl.contains(element)) return null;
        const cell = element.closest('[data-diff-row]');
        if (!cell || !diffEl.contains(cell)) return null;
        const index = Number(cell.dataset.diffRow);
        if (!Number.isInteger(index) || index < 0 || index >= view.captureRows.length) return null;
        return { index, atStart: boundaryAtCellStart(cell, container, offset) };
    }

    /**
     * The current selection as an inclusive `captureRows` span, or null.
     *
     * BOTH boundaries must land on diff lines of THIS view: a selection that starts
     * in the diff and ends in the dock, the file rail, or a hunk header names no
     * range rather than a wrong one. An END boundary sitting before its row's first
     * character (dragging just past a line break) excludes that row, so the capture
     * is never silently one line wider than what the owner highlighted.
     */
    function readDiffSelection() {
        const selection = typeof window !== 'undefined' && window.getSelection ? window.getSelection() : null;
        if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return null;
        const range = selection.getRangeAt(0);
        const start = resolveDiffBoundary(range.startContainer, range.startOffset);
        const end = resolveDiffBoundary(range.endContainer, range.endOffset);
        if (!start || !end) return null;
        const [first, last] = start.index <= end.index ? [start, end] : [end, start];
        let endIndex = last.index;
        if (endIndex > first.index && last.atStart) endIndex -= 1;
        return { startIndex: first.index, endIndex };
    }

    /**
     * Keep the sticky button in step with the live selection, and remember the
     * span: the button and a dock-focused ⌘L are the only callers allowed to read a
     * selection the browser has since collapsed.
     */
    function syncSelectionButton() {
        const span = readDiffSelection();
        if (span) view.lastSelection = span;
        if (selectionBtn) selectionBtn.hidden = !span;
    }

    function clearWindowSelection() {
        const selection = typeof window !== 'undefined' && window.getSelection ? window.getSelection() : null;
        if (selection && typeof selection.removeAllRanges === 'function') selection.removeAllRanges();
        view.lastSelection = null;
        if (selectionBtn) selectionBtn.hidden = true;
    }

    /**
     * Append a TEXT part to the dock, preserving dock order.
     *
     * `createComposerParts` exposes a chip append only, so the typed draft is
     * committed first (exactly what `addChip` does) and the part is appended
     * through `setParts`. Used for the one capture the marker codec has no chip
     * form for: a pure-deletion selection.
     */
    function addDockText(text) {
        const part = makeTextPart(text);
        if (!part) return false;
        dock.setParts([...dock.commitDraft(), part]);
        dock.focus();
        return true;
    }

    /**
     * Append a context chip for what the owner is looking at to the DOCK (never
     * straight to chat): the dock is where they see exactly what will be sent.
     * With no selection this is the whole active file.
     */
    function capture({ selectionOnly = false } = {}) {
        const file = activeFile();
        if (!file) {
            if (!selectionOnly) {
                showToast('Open a task with changes first — there is nothing to add to chat context yet.', 'warn');
            }
            return false;
        }
        const active = document.activeElement;
        const focusInDock = Boolean(active instanceof Element && active.closest('[data-capture-dock]'));
        const span = readDiffSelection() || (selectionOnly || focusInDock ? view.lastSelection : null);
        if (!span && selectionOnly) return false;
        const rows = span ? view.captureRows.slice(span.startIndex, span.endIndex + 1) : [];
        const decision = diffChipDecision({ path: file.path, rows });
        if (rows.length && decision.lineStart === null) {
            // A PURE-deletion selection: no new-side number anywhere, so no range
            // could honestly name it and the codec would drop the bytes with the
            // range. The lines are still the owner's ask, so they go into the dock
            // as a path-bearing text QUOTE instead of a whole-file chip that lost
            // them. Rare by construction (a single context or added row in the
            // selection gives a real range) — which matters, because a text part
            // has no remove button of its own.
            if (!addDockText(deletionQuoteText(decision))) {
                showToast(`Nothing captured: "${file.path}" has no lines to quote here.`, 'warn');
                return false;
            }
            showToast('Deleted lines have no line number in the new file — added them to the '
                + 'dock as a quoted excerpt instead of a chip.', 'info');
            clearWindowSelection();
            return true;
        }
        const chip = makeChipPart(decision);
        if (!chip) {
            showToast(`Nothing captured: "${file.path}" cannot be written as a context reference.`, 'warn');
            return false;
        }
        if (decision.clampedRows) {
            // The range and the bytes were clamped to the first hunk together, so
            // the chip is accurate — but the owner highlighted more than it carries.
            showToast(`Selection crossed a hunk boundary — captured the first hunk only `
                + `(${decision.clampedRows} more selected line${decision.clampedRows === 1 ? '' : 's'} left out).`, 'warn');
        }
        dock.addChip(chip);  // commits any typed draft first, then focuses the dock
        clearWindowSelection();
        return true;
    }

    async function requestEdits() {
        const controller = typeof getChatController === 'function' ? getChatController() : null;
        if (!controller || typeof controller.sendParts !== 'function') return;
        const dockParts = dock.commitDraft();
        // The task line alone says nothing the owner asked for: an empty dock is a
        // no-op that puts the cursor back in the field instead of sending prose.
        const asked = dockParts.some(
            (part) => part.type === 'chip' || String(part.text || '').trim(),
        );
        if (!asked) {
            dock.focus();
            return;
        }
        const parts = requestEditsParts(view.task || { task_id: view.taskId }, dockParts);
        const sent = await controller.sendParts(parts);
        // The draft is the owner's only copy until the handoff succeeded.
        if (sent === false) return;
        dock.clear();
        if (typeof showPage === 'function') await showPage('chat');
    }

    page.addEventListener('click', (event) => {
        const taskRow = event.target.closest('[data-changes-task]');
        if (taskRow && page.contains(taskRow)) {
            selectTask(taskRow.dataset.changesTask);
            return;
        }
        const fileRow = event.target.closest('[data-changes-file]');
        if (fileRow && page.contains(fileRow)) {
            view.filePath = fileRow.dataset.changesFile;
            paintFileList();
            paintDiff();
            return;
        }
        const mode = event.target.closest('[data-changes-mode]');
        if (mode && page.contains(mode)) {
            view.mode = mode.dataset.changesMode === 'split' ? 'split' : 'unified';
            page.querySelectorAll('[data-changes-mode]').forEach((button) => {
                button.classList.toggle('active', button.dataset.changesMode === view.mode);
            });
            paintDiff();
            return;
        }
        if (event.target.closest('[data-changes-refresh]')) {
            event.preventDefault();
            loadTasks().then(() => (view.taskId ? selectTask(view.taskId, { filePath: view.filePath }) : null));
        }
    });

    dockForm.addEventListener('submit', (event) => {
        event.preventDefault();
        requestEdits();
    });

    if (selectionBtn) {
        // The button is the GUARANTEED capture path (⌘L is best-effort: some
        // browsers keep it for the address bar), so it must not destroy the
        // selection it is about to read — mousedown default = focus change =
        // collapsed selection.
        selectionBtn.addEventListener('mousedown', (event) => event.preventDefault());
        selectionBtn.addEventListener('click', () => { capture({ selectionOnly: true }); });
    }
    diffEl.addEventListener('mouseup', () => syncSelectionButton());
    document.addEventListener('selectionchange', () => syncSelectionButton());

    // The global ⌘L handler (app.js `[anchor:phase-C]`) knows nothing about this
    // module: it names the active page in one event and the owning page listens.
    // The event is CANCELABLE, and calling preventDefault here is how this page
    // says "I consumed the keystroke" — only then does the global handler suppress
    // the browser default. Staying silent leaves ⌘L to the browser, which is more
    // honest than swallowing the address-bar shortcut to do nothing.
    //
    // Which is why the cancel is gated on the capture ACTUALLY happening.
    // `capture()` is synchronous and returns whether a chip (or a deletion quote)
    // reached the dock; on the "open a task with changes first" path nothing was
    // consumed, so ⌘L falls through instead of being eaten to no effect.
    window.addEventListener('ouro:capture-selection', (event) => {
        if (event.detail?.page !== 'changes') return;
        if (!page.classList.contains('active')) return;
        if (capture()) event.preventDefault();
    });

    if (typeof subscribeState === 'function') {
        // Badge inputs only. Repainting the rail on every poll would reset its
        // scroll position and hover state a few times a minute for nothing, so the
        // paint is gated on the badge data actually changing.
        let knownBadgeJson = '';
        subscribeState((data) => {
            view.projects = Array.isArray(data?.projects) ? data.projects : view.projects;
            view.taskBindings = (data && data.task_bindings) || view.taskBindings;
            const json = JSON.stringify([
                view.projects.map((project) => [project?.id, project?.name]),
                Object.entries(view.taskBindings).map(([id, binding]) => [id, binding?.project_id]),
            ]);
            if (json === knownBadgeJson) return;
            knownBadgeJson = json;
            paintTaskRail();
        });
    }

    // The rail is refreshed when the page is entered: a task that finished while
    // the owner was elsewhere must be reviewable without a reload.
    window.addEventListener('ouro:page-shown', (event) => {
        if (event?.detail?.page !== 'changes') return;
        loadTasks();
        if (view.taskId) selectTask(view.taskId, { filePath: view.filePath });
    });

    // Opening a specific task (inspector → "open full diff") lands here.
    window.addEventListener('ouro:open-changes', async (event) => {
        const taskId = String(event?.detail?.taskId || '');
        if (!taskId) return;
        if (typeof showPage === 'function') await showPage('changes');
        if (!view.tasks.length) await loadTasks();
        await selectTask(taskId, { filePath: String(event?.detail?.filePath || '') });
    });

    paintAll();
    loadTasks();

    return {
        page,
        dock,
        selectTask,
        refresh: loadTasks,
        /** Capture the current diff selection (or the whole file) into the dock. */
        capture: (options) => capture(options || {}),
        /** Test/inspection seam: the current view snapshot. */
        snapshot: () => ({ ...view }),
    };
}
