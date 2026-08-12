/**
 * Project threads (T1): the sidebar thread list, the owner's manual order, the
 * per-thread unread arithmetic, the thread row menu, and the CENTRE stage a
 * thread's chat mounts into.
 *
 * Why this is its own module (R8): `web/app.js` is under a hard 1600-line module
 * gate and is already the navigation/state-machine owner. Thread UI lands here;
 * app.js keeps only the instance lifecycle (single live instance, scroll stash,
 * paint ACK) it already owned and calls into these seams.
 *
 * WHY THE CENTRE, not the right panel: a project chat used to open as a right
 * split panel, which on a phone became a full-screen overlay stacked ON TOP of
 * the content area — two competing full-screen surfaces with two close
 * affordances. A thread is a conversation, i.e. the primary thing you look at,
 * so it takes over the centre exactly like Main Chat does. The right panel is
 * now the task inspector's alone.
 *
 * The pure half (ordering, unread aggregation, cursor normalization) is exported
 * separately from the DOM half and is covered by `web/tests/project_threads.test.js`.
 */

import { openConfirmDialog } from './confirm_dialog.js';
import { openRowMenu } from './project_create.js';
import { renderMobileNavToggle } from './page_header.js';
import {
    describeOutcome,
    openThreadChanges,
    removalPrompt,
    snapshotReceipt,
    threadActions,
    threadOps,
} from './project_thread_actions.js';

/** Thread #0 IS the project's original chat — mirrors `contracts/chat_id_policy.py`. */
export const MAIN_THREAD_ID = 0;
/** Mirrored from the frozen backend THREAD_NAME_MAX contract. */
export const THREAD_NAME_MAX = 80;

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

// Drag rows are the ITEM wrappers, never their inner row buttons.
const THREAD_ITEM_SELECTOR = '.nav-thread-item[data-thread-id]';
const PROJECT_ITEM_SELECTOR = '.nav-project-item[data-project-id]';

/** The instance/stash/cursor key for one thread. Project ids never contain '#'. */
export function threadKey(projectId, threadId) {
    return `${String(projectId || '')}#${Number(threadId) || 0}`;
}

/**
 * Browser mirror of `gateway/ui_preferences.py::_normalize_seen_revision`.
 *
 * The read cursor is NESTED per thread (`{project: {thread: revision}}`). A FLAT
 * `{project: revision}` entry is what every pre-T1 runtime stored and is the one
 * compatibility spelling of thread #0's cursor, so it normalizes to
 * `{project: {"0": revision}}` — the same rule the server applies, kept here so
 * a browser that read preferences written by an older server agrees with it
 * instead of treating every project as unread.
 *
 * @param {Object|null|undefined} raw
 * @returns {Object.<string, Object.<string, number>>}
 */
export function normalizeSeenRevision(raw) {
    const out = {};
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out;
    for (const [pid, value] of Object.entries(raw)) {
        const key = String(pid || '').trim();
        if (!key) continue;
        if (value && typeof value === 'object' && !Array.isArray(value)) {
            const lane = {};
            for (const [tid, revision] of Object.entries(value)) {
                const thread = Number(tid);
                if (!Number.isFinite(thread)) continue;
                lane[String(Math.trunc(thread))] = Math.max(0, Number(revision) || 0);
            }
            out[key] = lane;
        } else {
            out[key] = { [String(MAIN_THREAD_ID)]: Math.max(0, Number(value) || 0) };
        }
    }
    return out;
}

/** The acknowledged revision of one thread (0 when never acknowledged). */
export function seenRevisionFor(cursor, projectId, threadId) {
    const lane = (cursor || {})[String(projectId || '')];
    if (!lane) return 0;
    return Math.max(0, Number(lane[String(Number(threadId) || 0)]) || 0);
}

/** Record an acknowledgement in the local mirror, monotonically. */
export function rememberSeenRevision(cursor, projectId, threadId, revision) {
    const store = cursor || {};
    const pid = String(projectId || '');
    const tid = String(Number(threadId) || 0);
    const lane = store[pid] || (store[pid] = {});
    lane[tid] = Math.max(Number(lane[tid]) || 0, Math.max(0, Number(revision) || 0));
    return store;
}

/**
 * The canonical thread rows of a sidebar project entry, thread #0 first.
 *
 * `/api/state` already ships `ProjectEntry.threads` from the server's canonical
 * projection. The fallback synthesizes thread #0 from the project's own
 * `chat_id`/`name`/`visible_revision` so a browser talking to a server that
 * predates the projection still renders exactly one (correct) thread rather
 * than an empty list.
 */
export function projectThreadRows(project) {
    const rows = Array.isArray(project?.threads) ? project.threads : null;
    if (rows && rows.length) {
        return rows
            .filter((thread) => thread && Number.isFinite(Number(thread.id)))
            .map((thread) => ({
                ...thread,
                id: Number(thread.id) || 0,
                chat_id: Number(thread.chat_id) || 0,
                name: String(thread.name || '') || `Thread ${Number(thread.id) || 0}`,
                visible_revision: Math.max(0, Number(thread.visible_revision) || 0),
            }));
    }
    return [{
        id: MAIN_THREAD_ID,
        chat_id: Number(project?.chat_id) || 0,
        name: String(project?.name || project?.id || ''),
        visible_revision: Math.max(0, Number(project?.visible_revision) || 0),
    }];
}

/** Every thread EXCEPT #0 — the ones the sidebar lists under the project row. */
export function extraThreadRows(project) {
    return projectThreadRows(project).filter((thread) => thread.id !== MAIN_THREAD_ID);
}

/** A thread is unread when its OWN revision exceeds its OWN acknowledged cursor. */
export function isThreadUnread(thread, cursor, projectId) {
    return Math.max(0, Number(thread?.visible_revision) || 0)
        > seenRevisionFor(cursor, projectId, thread?.id);
}

/**
 * How many of a project's threads are unread — the GROUP total behind the
 * `#nav-projects-count` pill. The project row owns no unread number of its own;
 * it is a grouping over its threads, so a sibling thread's message can never
 * mark the project's main thread read.
 *
 * This is deliberately NOT what the project row's dot reads — see
 * `isMainThreadUnread`.
 */
export function unreadThreadCount(project, cursor) {
    if (String(project?.lifecycle || 'active') !== 'active') return 0;
    const pid = String(project?.id || '');
    return projectThreadRows(project).filter((thread) => isThreadUnread(thread, cursor, pid)).length;
}

/**
 * Is the project's OWN chat (thread #0) unread? THIS is the project row's dot.
 *
 * The project row is simultaneously thread #0's row and the group's header, and
 * a dot can only mean one of those. It means thread #0: clicking the row opens
 * thread #0, so a dot the click cannot clear would be an indicator with no way
 * out, and an aggregate dot would light a SECOND dot for a message whose own
 * thread row is already lit right below it. The group's unread is not lost —
 * it is the `#nav-projects-count` pill, which counts unread THREADS and stays
 * visible while the list is collapsed.
 */
export function isMainThreadUnread(project, cursor) {
    if (String(project?.lifecycle || 'active') !== 'active') return false;
    const rows = projectThreadRows(project);
    const main = rows.find((thread) => Number(thread.id) === MAIN_THREAD_ID);
    return Boolean(main) && isThreadUnread(main, cursor, String(project?.id || ''));
}

/**
 * Apply an owner's manual order (D3) as an explicit PREFIX: everything the owner
 * has placed keeps that order, everything else falls in behind it in the
 * caller's default order. A stale id in the stored order is ignored, so
 * deleting a project or thread never scrambles the rest of the list.
 *
 * @param {Array} rows          default-ordered rows
 * @param {string[]} manual     owner-ordered ids
 * @param {(row:any)=>string} idOf
 */
export function applyManualOrder(rows, manual, idOf) {
    const list = Array.isArray(rows) ? [...rows] : [];
    const order = Array.isArray(manual) ? manual.map(String) : [];
    if (!order.length) return list;
    const rank = new Map(order.map((id, index) => [id, index]));
    const placed = [];
    const rest = [];
    for (const row of list) {
        (rank.has(idOf(row)) ? placed : rest).push(row);
    }
    placed.sort((a, b) => rank.get(idOf(a)) - rank.get(idOf(b)));
    return [...placed, ...rest];
}

/**
 * D3 project order: newest on top by default (recency), owner's manual order
 * first. Deliberately NO unread hoist — "no clever logic" is the owner's rule,
 * and a list that reshuffles itself when a message arrives is exactly the
 * jumping-target problem drag-and-drop ordering exists to end.
 */
export function orderProjectRows(rows, manualOrder) {
    const recency = (p) => String(p?.last_active_at || p?.updated_at || p?.created_at || '');
    const byRecency = [...(rows || [])].sort((a, b) => recency(b).localeCompare(recency(a)));
    return applyManualOrder(byRecency, manualOrder, (row) => String(row.id));
}

/**
 * D3 thread order within a project: a NEW thread on top (ids are monotonic, so
 * descending id is "newest first" without reading a clock), owner's manual
 * order first.
 */
export function orderThreadRows(threads, manualOrder) {
    const byNewest = [...(threads || [])].sort((a, b) => (Number(b.id) || 0) - (Number(a.id) || 0));
    return applyManualOrder(byNewest, manualOrder, (row) => String(row.id));
}

/**
 * The new explicit order after dropping `draggedId` onto `targetId`.
 * Returns the FULL id list so what is persisted is what is displayed — a
 * partial prefix would let the default order re-sort the tail under the owner.
 *
 * @param {string[]} ids       currently displayed ids, in display order
 * @param {string} draggedId
 * @param {string} targetId
 * @param {boolean} placeAfter drop below the target rather than above it
 */
export function reorderIds(ids, draggedId, targetId, placeAfter = false) {
    const list = (ids || []).map(String);
    const dragged = String(draggedId);
    const target = String(targetId);
    if (dragged === target || !list.includes(dragged) || !list.includes(target)) return list;
    const without = list.filter((id) => id !== dragged);
    const at = without.indexOf(target);
    without.splice(at + (placeAfter ? 1 : 0), 0, dragged);
    return without;
}

// ---------------------------------------------------------------------------
// Sidebar: the thread list under a project row
// ---------------------------------------------------------------------------

/**
 * How one thread row PAINTS, by lifecycle. Pure, so the rule is testable.
 *
 * THREE states reach the sidebar, not two, and they mean different things:
 *
 *   - `active` — an ordinary thread.
 *   - `deleting` — the deliberate end state of a deletion that would not quiesce
 *     (`fail_thread_deletion`). The projection keeps it on screen precisely so the
 *     owner can retry it, and the menu offers exactly that one action. An unread
 *     dot here would invite a click into a room that is being torn down.
 *   - `archived` — hidden everywhere EXCEPT while a task is still live in it
 *     (X10: hiding a room that is still emitting output leaves the owner watching
 *     nothing while work continues).
 *
 * Painting all three alike told the owner nothing about which of their
 * instructions had landed; painting `deleting` and `archived` the SAME told them
 * the wrong one.
 */
export function threadRowPresentation(thread) {
    const lifecycle = String(thread?.lifecycle || 'active');
    const deleting = lifecycle === 'deleting';
    const archived = lifecycle === 'archived';
    const name = String(thread?.name || '');
    const error = String(thread?.delete_error || '').trim();
    let title = name;
    if (deleting) title = `${name} — Deleting…${error ? ` ${error}` : ''}`;
    else if (archived) title = `${name} — Archived; still shown because a task is running in it`;
    return {
        lifecycle,
        modifier: `${deleting ? ' is-deleting' : ''}${archived ? ' is-archived' : ''}`,
        state: deleting ? 'Deleting…' : (archived ? 'Archived' : ''),
        title,
        // A thread on its way out is not a thread to open, and the drag order of a
        // row that is about to disappear is not an order worth persisting.
        draggable: !deleting,
        showsUnread: !deleting,
    };
}

/**
 * Build the sibling container listing a project's extra threads.
 *
 * SIBLING, not a child of the project row: the pinned markup contract keeps the
 * project row a single `<button>` with one trailing action slot
 * (`item.append(btn, trailing)`), and interactive UI must never be nested inside
 * a button. Returns `null` when the project has no extra threads, so a project
 * that never used threads renders exactly the sidebar it always did.
 *
 * @returns {HTMLElement|null}
 */
export function renderThreadList(project, {
    cursor = {},
    manualOrder = [],
    activeThreadKey = '',
    onOpen,
    onMenu,
    onReorder,
} = {}) {
    const pid = String(project?.id || '');
    const threads = orderThreadRows(extraThreadRows(project), manualOrder);
    if (!threads.length) return null;
    const list = document.createElement('div');
    list.className = 'nav-thread-list';
    list.dataset.threadsFor = pid;
    list.setAttribute('role', 'group');
    list.setAttribute('aria-label', `Threads in ${project.name || pid}`);

    // The ITEM wrapper only. Both the wrapper and its inner row button carry
    // `data-thread-id` (the button's is what the active-state sync and tests
    // select on), so a bare `[data-thread-id]` would report every id twice —
    // committing a duplicated order — and would attach the drag feedback classes
    // to the button, which the `.nav-thread-item.*` rules do not style.
    const displayedIds = () => Array.from(list.querySelectorAll(THREAD_ITEM_SELECTOR))
        .map((el) => el.dataset.threadId);

    for (const thread of threads) {
        const key = threadKey(pid, thread.id);
        const paint = threadRowPresentation(thread);
        const item = document.createElement('div');
        item.className = `nav-thread-item${paint.modifier}`;
        item.dataset.threadKey = key;
        item.dataset.threadId = String(thread.id);
        item.dataset.threadLifecycle = paint.lifecycle;
        item.draggable = paint.draggable;

        const row = document.createElement('button');
        row.type = 'button';
        row.className = 'nav-row nav-thread-row';
        // NOT `data-project-id`: that attribute is the project-row active-state
        // selector, and reusing it here would light up every thread of the
        // active project as if it were the open one.
        row.dataset.threadProjectId = pid;
        row.dataset.threadId = String(thread.id);
        row.dataset.threadKey = key;
        row.title = paint.title;
        // The rule this presentation states ("a thread on its way out is not a
        // thread to open") was implemented for `draggable` and `showsUnread` and
        // NOT for the click, so a tombstoning room stayed openable, the admission
        // fence answered it, and the chat annotated it as an unavailable PROJECT.
        // The project row does this correctly with `btn.disabled`; so does this one
        // now, off the same derived flag (I12).
        row.disabled = !paint.draggable;
        const label = document.createElement('span');
        label.className = 'nav-row-label nav-thread-label';
        label.textContent = thread.name;
        row.appendChild(label);
        if (paint.state) {
            // The state, in words, on the row itself: a greyed row with no reason
            // teaches nothing, and this one is the reason the menu offers what it
            // offers.
            const state = document.createElement('span');
            state.className = 'nav-thread-state';
            state.textContent = paint.state;
            row.appendChild(state);
        }
        if (paint.showsUnread && isThreadUnread(thread, cursor, pid)) {
            const dot = document.createElement('span');
            dot.className = 'nav-unread-dot';
            dot.title = 'New activity';
            row.appendChild(dot);
            row.classList.add('has-unread');
        }
        if (key === activeThreadKey) {
            row.classList.add('active');
            row.setAttribute('aria-current', 'page');
        }
        row.addEventListener('click', () => onOpen?.(project, thread));

        const kebab = document.createElement('button');
        kebab.type = 'button';
        kebab.className = 'nav-project-kebab nav-thread-kebab';
        kebab.textContent = '⋯';
        kebab.title = 'Thread actions';
        kebab.setAttribute('aria-label', `Actions for thread ${thread.name}`);
        kebab.addEventListener('click', (event) => {
            event.stopPropagation();
            onMenu?.(project, thread, kebab);
        });

        item.append(row, kebab);
        list.appendChild(item);
    }

    attachReorder(list, THREAD_ITEM_SELECTOR, (ids) => onReorder?.(pid, ids), displayedIds);
    return list;
}

/**
 * Pointer drag-and-drop reordering over a container of rows (D3).
 *
 * Uses the native HTML drag events rather than a pointer-move reimplementation:
 * the browser then owns the drag image, the escape-to-cancel behaviour and the
 * accessibility semantics. `onCommit(ids)` receives the FULL new id order and is
 * called once, on drop.
 */
function attachReorder(container, rowSelector, onCommit, displayedIds) {
    let draggedId = '';
    container.addEventListener('dragstart', (event) => {
        const row = event.target.closest(rowSelector);
        if (!row) return;
        draggedId = row.dataset.threadId || row.dataset.projectId || '';
        row.classList.add('is-dragging');
        try { event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', draggedId); } catch {}
    });
    container.addEventListener('dragend', () => {
        draggedId = '';
        container.querySelectorAll('.is-dragging, .drop-before, .drop-after')
            .forEach((el) => el.classList.remove('is-dragging', 'drop-before', 'drop-after'));
    });
    container.addEventListener('dragover', (event) => {
        const row = event.target.closest(rowSelector);
        if (!row || !draggedId) return;
        event.preventDefault();
        const box = row.getBoundingClientRect();
        const after = (event.clientY - box.top) > box.height / 2;
        container.querySelectorAll('.drop-before, .drop-after')
            .forEach((el) => el.classList.remove('drop-before', 'drop-after'));
        row.classList.add(after ? 'drop-after' : 'drop-before');
    });
    container.addEventListener('drop', (event) => {
        const row = event.target.closest(rowSelector);
        if (!row || !draggedId) return;
        event.preventDefault();
        const box = row.getBoundingClientRect();
        const after = (event.clientY - box.top) > box.height / 2;
        const targetId = row.dataset.threadId || row.dataset.projectId || '';
        const next = reorderIds(displayedIds(), draggedId, targetId, after);
        draggedId = '';
        onCommit?.(next);
    });
}

/** Drag-and-drop ordering for the PROJECT rows themselves (same D3 surface). */
export function attachProjectReorder(listEl, onCommit) {
    attachReorder(
        listEl,
        PROJECT_ITEM_SELECTOR,
        onCommit,
        () => Array.from(listEl.querySelectorAll(PROJECT_ITEM_SELECTOR))
            .map((el) => el.dataset.projectId),
    );
}

// ---------------------------------------------------------------------------
// Thread row menu: Rename… / Fork
// ---------------------------------------------------------------------------

/**
 * Per-thread actions, mounted through the SAME accessible row-menu shell the
 * project row uses (`project_create.js::openRowMenu`) so the keyboard model and
 * viewport-safe placement have exactly one implementation.
 *
 * Rename validates against the mirrored 80-char backend contract before the
 * request, so the owner gets the limit explained rather than a 400.
 */
/**
 * Where a thread WORKS, and what removing its checkout would destroy — one read.
 *
 * A thread's location is derived, never stored (A7), so a menu has to ASK before
 * it can know whether there is a checkout to merge, show or remove. The same
 * answer carries the inspection the removal prompt needs, so this is one request,
 * not two. A failed read is DISCLOSED rather than guessed: reporting "works in
 * the project folder" for a registry we could not read would offer Branch off…
 * for a thread that already has a checkout, and provisioning refuses that by name
 * a second later.
 */
export async function readThreadCheckout(projectId, threadId, { ops = threadOps } = {}) {
    const unknown = (why) => ({
        location: { where: 'project_folder' },
        inspection: {},
        locationError: String(why),
    });
    try {
        const seen = await ops.inspectWorktree(projectId, threadId);
        // A typed refusal now arrives as a VALUE (`threadOps` unwraps it), so
        // `ok === false` has to be read here or an unknown thread would report
        // "works in the project folder" — a location we did not learn, rendered
        // as one we did.
        if (seen && seen.ok === false) return unknown(seen.message || seen.reason || 'unknown');
        return {
            location: seen?.location || { where: 'project_folder' },
            inspection: seen?.inspection || {},
            locationError: '',
        };
    } catch (e) {
        return unknown(e?.body?.error || e?.message || e);
    }
}

export async function openThreadRowMenu(project, thread, { apiClient, anchorEl, onChanged }) {
    const { location, inspection, locationError } = await readThreadCheckout(project.id, thread.id);
    openRowMenu({
        anchorEl,
        ariaLabel: `Actions for thread ${thread.name}`,
        itemsHtml: `
            <button type="button" role="menuitem" data-prm="rename">Rename…</button>
            <button type="button" role="menuitem" data-prm="fork">Fork</button>
            ${threadActionItemsHtml(thread, location, locationError)}
        `,
        onSelect: async (action) => {
            if (action && action !== 'rename' && action !== 'fork') {
                await runThreadAction(action, project, thread, {
                    apiClient, onChanged, location, inspection,
                });
                if (anchorEl.isConnected) anchorEl.focus();
                return;
            }
            if (action === 'rename') {
                const res = await openConfirmDialog({
                    title: 'Rename thread',
                    body: `New name for “${thread.name}”:`,
                    input: true,
                    initialValue: thread.name,
                    confirmLabel: 'Rename',
                });
                const newName = res?.confirmed ? String(res.value || '').trim() : '';
                if (newName.length > THREAD_NAME_MAX) {
                    await openConfirmDialog({
                        title: 'Rename thread',
                        body: `Thread names are limited to ${THREAD_NAME_MAX} characters.`,
                        alert: true,
                    });
                } else if (newName && newName !== thread.name) {
                    try {
                        await apiClient.projectThreadUpdate(project.id, thread.id, newName);
                        onChanged?.({ authoritative: true });
                    } catch (e) {
                        await openConfirmDialog({
                            title: 'Rename failed',
                            body: `Rename failed: ${e?.body?.error || e?.message || e}`,
                            alert: true,
                        });
                        // A failure is a reason to RE-READ, not to stop. The
                        // commonest one is a 404 for a thread another tab just
                        // deleted, i.e. the sidebar is painting a row the server no
                        // longer has; without this the stale row survives the alert
                        // and stays clickable until the next poll tick.
                        onChanged?.({ authoritative: true });
                    }
                }
            } else if (action === 'fork') {
                // A cursor into this thread's rows, not a copy: the source thread
                // is untouched and the fork keeps resolving the shared past even
                // if the source is later archived or deleted (A3/A3a).
                try {
                    const payload = await apiClient.projectThreadFork(project.id, thread.id);
                    onChanged?.({ authoritative: true, thread: payload?.thread || null });
                } catch (e) {
                    await openConfirmDialog({
                        title: 'Fork failed',
                        body: `Fork failed: ${e?.body?.error || e?.message || e}`,
                        alert: true,
                    });
                    onChanged?.({ authoritative: true });  // same reason as rename
                }
            }
            if (anchorEl.isConnected) anchorEl.focus();
        },
    });
}

// ---------------------------------------------------------------------------
// Branch / merge / checkout / lifecycle: the menu half of the T3 seam
// ---------------------------------------------------------------------------

/** Menu-safe text. The row-menu shell takes an HTML string, so this is required. */
function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/** How many evidence lines a one-paragraph dialog shows before it says "and N more". */
const EVIDENCE_CAP = 8;

/**
 * A sentence with its evidence attached, never flattened away.
 *
 * `openConfirmDialog` renders ONE escaped paragraph, so the paths ride inline.
 * They are capped and the omission is COUNTED rather than silently dropped —
 * "and 30 more" is information; a truncated list that looks complete is not.
 */
export function withEvidence(text, evidence) {
    const list = (evidence || []).filter(Boolean).map(String);
    if (!list.length) return String(text || '');
    const shown = list.slice(0, EVIDENCE_CAP);
    const more = list.length - shown.length;
    return `${text} — ${shown.join('; ')}${more ? ` (and ${more} more)` : ''}`;
}

/**
 * The action rows for one thread, as menu HTML.
 *
 * Order and availability come from `threadActions` (the T3 seam) so the rule
 * "what may this thread do right now" has ONE definition. An unavailable row is
 * rendered DISABLED with its reason as the tooltip rather than omitted: a missing
 * item teaches nothing, a greyed one teaches what to do first. The `Retry delete`
 * row is the exception that carries a `reason` while being AVAILABLE — that is
 * `thread.delete_error`, i.e. why the row is on offer at all.
 */
export function threadActionItemsHtml(thread, location, locationError = '', only = null) {
    const wanted = Array.isArray(only) ? new Set(only) : null;
    // `only` exists for ONE caller: the project row menu, which is thread #0's row
    // and already carries the project's own Rename…/Delete project…. Showing
    // thread #0's disabled Archive/Delete… beside them would put two delete-shaped
    // rows in one menu, one of which means something entirely different.
    const rows = threadActions(thread, location).filter((row) => !wanted || wanted.has(row.id));
    const items = rows.map((row) => {
        // A checkout read that FAILED is not "no checkout": disable the rows that
        // depend on the answer and say why, instead of offering an action that
        // will refuse a second later for a reason the owner cannot connect to it.
        const dependsOnCheckout = ['branch_off', 'merge_back', 'show_changes', 'remove_worktree']
            .includes(row.id);
        const blocked = Boolean(locationError) && dependsOnCheckout;
        const available = row.available && !blocked;
        const title = blocked
            ? `This thread's checkout could not be read: ${locationError}`
            : (row.disabledReason || row.reason || '');
        return `<button type="button" role="menuitem" data-prm="${escapeHtml(row.id)}"${
            available ? '' : ' disabled'
        }${row.id === 'delete' ? ' class="danger"' : ''}${
            title ? ` title="${escapeHtml(title)}"` : ''
        }>${escapeHtml(row.label)}</button>`;
    });
    return items.join('\n');
}

/**
 * Show one owner-facing outcome, with its evidence.
 *
 * `ask` is injected all the way down this file rather than imported at each call
 * site. It defaults to `openConfirmDialog`, and it is what makes these decisions
 * testable at all: the DOM dialog cannot run under `node --test`, so without the
 * seam the only covered part of a branch/merge/delete gesture would be the part
 * that decides nothing. It is NOT called `confirm`: that is the banned native
 * dialog's name, and the deterministic gate over `web/modules` cannot tell a
 * parameter from the global — nor should a reader have to.
 */
async function announce(ask, title, described) {
    await ask({
        title,
        body: withEvidence(described.text, described.evidence),
        alert: true,
    });
}

/**
 * A refusal the owner can ANSWER, answered — and it says WHICH one they answered.
 *
 * Two typed follow-ups, and both were unreachable before this existed:
 *
 *   - `acknowledgeable` — the server saying this refusal has a second call. The
 *     owner is shown what stays behind, by name, and the retry passes the flag.
 *   - `decision` (`git_init_required`) — T2's OFFER. `apiClient.projectInitGit`
 *     is the yes; a menu that never renders it leaves a folder the owner is
 *     perfectly happy to track blocking every file operation in the project.
 *
 * Returning a bare `true` for either laundered one consent into a different one:
 * ANY answered refusal was retried with `run(true)`, so answering the
 * `git_init_required` offer ("yes, start tracking this folder") was sent as
 * `acknowledge_checkout_dirty: true` — a flag the owner never saw a sentence for
 * and never said "Merge anyway" to (I10). `'decision'` means the offer was taken
 * and the plain call should simply be retried; `'acknowledged'` means the owner
 * acknowledged THIS refusal and its flag is theirs to set.
 *
 * `'declined'` and `false` are different facts too: the owner SAW a question and
 * said no, versus this refusal had nothing to ask. Only the first means the
 * sentence has already been read, which is what stops it being replayed as an
 * alert one line later (I14).
 *
 * @returns {'decision'|'acknowledged'|'declined'|false}
 */
async function answerRefusal(described, { title, confirmLabel, project, apiClient, ask }) {
    const decision = described.decision;
    if (decision && String(decision.offer || '') === 'init_git') {
        const ok = await ask({
            title: 'Start tracking this folder?',
            body: `${described.text} Starting git here enables ${
                (decision.enables || ['diff', 'rollback', 'branching']).join(', ')
            }. Nothing is committed by Ouroboros without you.`,
            confirmLabel: 'Start tracking',
        });
        if (ok !== true) return 'declined';
        try {
            await apiClient.projectInitGit(project.id);
            return 'decision';
        } catch (e) {
            await ask({
                title: 'Could not start tracking',
                body: `Could not start tracking that folder: ${e?.body?.error || e?.message || e}`,
                alert: true,
            });
            // The owner said yes and it failed; they have just read why. Their
            // consent was not withdrawn, so this is not a decline.
            return 'declined';
        }
    }
    if (!described.acknowledgeable) return false;
    const ok = await ask({
        title,
        body: withEvidence(described.text, described.evidence),
        confirmLabel,
        danger: true,
    });
    return ok === true ? 'acknowledged' : 'declined';
}

/**
 * One call plus its owner-answerable retry. `run(acknowledged)` does the work.
 *
 * The retry passes `true` ONLY when the owner acknowledged THIS refusal. An
 * answered `git_init_required` offer is a different consent about a different
 * thing, so it re-runs the plain call (I10). `declined` is reported so the caller
 * can stay quiet: replaying the identical sentence as an alert after the owner has
 * just said no is the dialog answering itself (I14).
 */
async function withAcknowledgement(run, { title, confirmLabel, project, apiClient, ask }) {
    let outcome = await run(false);
    let described = describeOutcome(outcome);
    if (!outcome?.ok) {
        const answered = await answerRefusal(described, {
            title, confirmLabel, project, apiClient, ask,
        });
        if (answered === false || answered === 'declined') {
            return { outcome, described, retried: false, declined: answered === 'declined' };
        }
        outcome = await run(answered === 'acknowledged');
        described = describeOutcome(outcome);
        return { outcome, described, retried: true, declined: false };
    }
    return { outcome, described, retried: false, declined: false };
}

/**
 * Perform one thread action. The menu decides WHAT may be offered
 * (`threadActions`); this decides what each answer means.
 *
 * Every refusal is shown with its evidence; every refusal that carries an
 * owner-answerable flag is offered its second call in the same gesture, because a
 * refusal a menu cannot answer is a dead end wearing a sentence.
 */
export async function runThreadAction(action, project, thread, {
    apiClient, onChanged, location = null, inspection = null, ask = openConfirmDialog,
    ops = threadOps,
} = {}) {
    const pid = project.id;
    const tid = thread.id;
    const refresh = () => onChanged?.({ authoritative: true });
    try {
        if (action === 'branch_off') return await branchOff(project, thread, { apiClient, refresh, ask, ops });
        if (action === 'merge_back') return await mergeBack(project, thread, { apiClient, refresh, ask, ops });
        if (action === 'show_changes') {
            const shown = openThreadChanges({
                projectId: pid, threadId: tid, label: thread.name,
                branch: String(location?.branch || ''),
            });
            if (!shown) {
                await ask({
                    title: 'Show changes',
                    body: 'This thread works in the project folder, so it has no checkout of its own to diff.',
                    alert: true,
                });
            }
            return shown;
        }
        if (action === 'remove_worktree') {
            return await removeCheckout(project, thread, { apiClient, refresh, inspection, ask, ops });
        }
        if (action === 'archive' || action === 'restore') {
            const outcome = action === 'archive' ? await ops.archive(pid, tid) : await ops.restore(pid, tid);
            const described = describeOutcome(outcome);
            refresh();
            if (described.tone !== 'ok') {
                await announce(ask, action === 'archive' ? 'Archive' : 'Restore', described);
            } else if (outcome?.visible_until_terminal) {
                // X10's decision, said out loud. The server answers this flag "so
                // the surface can say which of the two just happened" and nothing
                // said it: the owner archived a thread, watched it stay exactly
                // where it was, and had no way to tell a deliberate rule from an
                // instruction that did not land.
                await ask({
                    title: 'Archived',
                    body: `“${thread.name}” is archived, and stays on screen until the task running in it finishes — hiding a room that is still producing output would leave you watching nothing while the work continues.`,
                    alert: true,
                });
            }
            return described;
        }
        if (action === 'delete') return await deleteThread(project, thread, { apiClient, refresh, ask, ops });
    } catch (e) {
        await ask({
            title: 'That did not finish',
            body: `${action.replace(/_/g, ' ')} did not finish: ${e?.body?.error || e?.message || e}`,
            alert: true,
        });
        refresh();
    }
    return null;
}

/** BRANCH OFF (A7/A8): choose a base, then disclose exactly what the snapshot did. */
async function branchOff(project, thread, { apiClient, refresh, ask, ops }) {
    const listed = await ops.bases(project.id, thread.id);
    if (listed && listed.ok === false) {
        const described = describeOutcome(listed);
        // A folder-less or untracked project answers HERE, before any base list
        // exists, and `git_init_required` is answerable — so the offer is made at
        // the moment the owner asked to branch, not two refusals later.
        const answered = await answerRefusal(described, {
            title: 'Branch off', confirmLabel: 'Continue', project, apiClient, ask,
        });
        if (answered === 'decision' || answered === 'acknowledged') {
            return branchOff(project, thread, { apiClient, refresh, ask, ops });
        }
        // A question the owner has just answered "no" to is not re-read to them
        // as an alert (I14); a refusal nothing could ask about still is.
        if (answered !== 'declined') await announce(ask, 'Branch off', described);
        return described;
    }
    // A14, at the one moment it is both true and actionable.
    const notice = listed?.queue_notice;
    const bases = Array.isArray(listed?.bases) ? listed.bases : [];
    const snapshot = listed?.snapshot;
    const offer = [...bases, ...(snapshot ? [snapshot] : [])];
    const labels = offer.map((base, index) => `${index + 1}. ${base.label || base.ref}`).join('  ');
    // I8: for thread #0 `thread.name` IS the project name, and this dialog is
    // reached from a menu titled "Actions for <project>" alongside Rename…/Delete
    // project…. "Base for Alpha's own checkout" then reads as an operation on the
    // project, when what it moves is the project's CHAT. Branching thread #0 is
    // coherent — its siblings keep the project folder and this branch merges back
    // into it — but the row could not say so.
    const isProjectChat = Number(thread.id) === MAIN_THREAD_ID;
    const preface = isProjectChat
        ? `This is ${project.name || project.id}'s own chat. Branching it gives THAT chat its own copy of the folder; the project's other threads keep working in the folder itself, and this branch merges back into it. `
        : '';
    const res = await ask({
        title: 'Branch off',
        body: `${preface}${notice?.queued ? `${notice.message} ` : ''}Base for ${thread.name}'s own checkout — type a number, a branch, a tag or a commit. ${labels}`,
        input: true,
        initialValue: String(listed?.current_branch || ''),
        confirmLabel: 'Branch off',
    });
    if (!res?.confirmed) return null;
    const typed = String(res.value || '').trim();
    if (!typed) return null;
    // The numbered list is an OFFER, not a restriction (A8): anything else the
    // owner types goes to the server as a commit-ish and is resolved there.
    const picked = /^\d+$/.test(typed) ? offer[Number(typed) - 1] : null;
    const baseRef = picked ? String(picked.ref || '') : typed;
    const { outcome, described, declined } = await withAcknowledgement(
        () => ops.branchOff(project.id, thread.id, baseRef),
        { title: 'Branch off', confirmLabel: 'Continue', project, apiClient, ask },
    );
    refresh();
    if (declined) return described;
    // The snapshot receipt is the only surface `tracked_sensitive` has: which
    // credential-shaped files were LEFT OUT of the commit (still untracked, still
    // in the folder) and which were already tracked and therefore committed with
    // everything else. Two opposite facts, and the owner needs both.
    const receipt = snapshotReceipt(outcome);
    await announce(ask, 'Branch off', {
        ...described,
        text: [described.text, receipt].filter(Boolean).join(' '),
    });
    return described;
}

/** MERGE BACK (A9), including the `checkout_dirty` retry the server offers. */
async function mergeBack(project, thread, { apiClient, refresh, ask, ops }) {
    const { outcome, described, declined } = await withAcknowledgement(
        (acknowledged) => ops.mergeBack(project.id, thread.id, acknowledged),
        {
            title: 'Merge back',
            confirmLabel: 'Merge anyway',
            project,
            apiClient,
            ask,
        },
    );
    refresh();
    if (!outcome?.ok && String(outcome?.reason || '') === 'merge_abort_failed') {
        // The one state that blocks everything else in that folder: the merge
        // could neither finish NOR be undone, so the project folder is stopped
        // part-way through it. Rendering that as one more red sentence next to
        // "conflicts" would hide the difference that matters — a conflict left the
        // folder byte-for-byte as it was, and this did not.
        await ask({
            title: 'The project folder is mid-merge',
            body: withEvidence(
                `${described.text} The folder ${outcome.working_dir || ''} is stopped part-way through a merge, so nothing else can run in it until you finish or abort it there yourself (git merge --abort).`.replace(/\s{2,}/g, ' '),
                [outcome.abort_detail, ...(described.evidence || [])],
            ),
            alert: true,
        });
        return described;
    }
    // The owner was OFFERED this refusal and said no. Announcing it now replays
    // the identical sentence they just dismissed (I14).
    if (!declined) await announce(ask, 'Merge back', described);
    return described;
}

/**
 * Remove a checkout (A10): the inspection is SHOWN before anything is removed.
 *
 * The prompt above is a PRE-FLIGHT read — the inspection captured when the menu
 * was opened — and the checkout can go dirty in that window, which for a thread an
 * agent is working in is the normal case. It then went out with
 * `acknowledged: false`, the server refused `unmerged_work`, and the gesture had
 * no way to answer its own "or confirm you want it gone": recovery was closing and
 * reopening the menu, which nothing disclosed. So the call rides
 * `withAcknowledgement` exactly like merge-back and delete, and the server's
 * refusal — now declaring `acknowledgeable` — is answered in the same gesture (I9).
 */
async function removeCheckout(project, thread, { apiClient, refresh, inspection, ask, ops }) {
    const prompt = removalPrompt(inspection);
    const ok = await ask({
        title: 'Remove checkout',
        body: withEvidence(prompt.text, prompt.evidence),
        confirmLabel: 'Remove checkout',
        danger: prompt.needsAcknowledgement,
    });
    if (ok !== true) return null;
    const { described, declined } = await withAcknowledgement(
        (acknowledged) => ops.removeWorktree(
            project.id, thread.id, prompt.needsAcknowledgement || acknowledged,
        ),
        {
            title: 'Remove checkout',
            confirmLabel: 'Remove anyway',
            project,
            apiClient,
            ask,
        },
    );
    refresh();
    if (!declined) await announce(ask, 'Remove checkout', described);
    return described;
}

/**
 * Delete a thread, checkout and all (D4) — two steps for rebuildable dirt, a wall
 * for work.
 *
 * `checkout_holds_rebuildable_files` is a QUESTION: the checkout's only contents
 * are files git was told to ignore or was never told about, so nothing the
 * repository would not still have is at stake. `checkout_holds_work` is a WALL:
 * commits that exist nowhere else, edits to tracked files, or an inspection that
 * could not be taken — and the way past it is a merge back or an acknowledged
 * removal, not a louder yes here. The server names which one it is; this only has
 * to stop conflating them.
 */
async function deleteThread(project, thread, { apiClient, refresh, ask, ops }) {
    const retrying = String(thread.lifecycle || 'active') === 'deleting';
    const ok = await ask({
        title: retrying ? 'Retry delete' : 'Delete thread',
        body: retrying
            ? `Deleting “${thread.name}” did not finish${thread.delete_error ? `: ${thread.delete_error}` : ''}. Ask again?`
            : `Delete “${thread.name}”? Its id and chat id are reserved forever and its journal rows physically remain — they are not erased. A checkout it owns is removed with it.`,
        confirmLabel: retrying ? 'Retry delete' : 'Delete',
        danger: true,
    });
    if (ok !== true) return null;
    const { described, declined } = await withAcknowledgement(
        (acknowledged) => ops.delete(project.id, thread.id, acknowledged),
        {
            title: 'Delete thread',
            confirmLabel: 'Delete anyway',
            project,
            apiClient,
            ask,
        },
    );
    refresh();
    if (described.tone !== 'ok' && !declined) await announce(ask, 'Delete thread', described);
    return described;
}

/**
 * ARCHIVED threads, and the way back (D4).
 *
 * The sidebar paints `/api/state`, whose projection FILTERS archived threads, so
 * without this affordance an archived thread was on no surface the owner could
 * reach and `restore` could not be invoked at all — archive was a one-way trip
 * with a documented inverse nobody could press. `include_archived` exists on
 * `/api/projects` precisely for a surface that can show them; this is that
 * surface, deliberately built out of the SAME row-menu vocabulary rather than a
 * new screen (P7) — and, since the restore itself rides `runThreadAction`, out of
 * the same failure handling as every other thread gesture.
 */
export async function openArchivedThreadsMenu(project, {
    apiClient, anchorEl, onChanged, ask = openConfirmDialog, openMenu = openRowMenu,
    ops = threadOps,
} = {}) {
    let rows = [];
    let error = '';
    try {
        const listed = await apiClient.projectsList(true);
        const entry = (listed?.projects || []).find((row) => String(row.id) === String(project.id));
        rows = (entry?.threads || []).filter((t) => String(t.lifecycle || '') === 'archived');
    } catch (e) {
        error = String(e?.body?.error || e?.message || e);
    }
    if (error || !rows.length) {
        await ask({
            title: 'Archived threads',
            body: error
                ? `The archived threads could not be read: ${error}`
                : `No archived threads in “${project.name || project.id}”.`,
            alert: true,
        });
        if (anchorEl?.isConnected) anchorEl.focus();
        return rows;
    }
    openMenu({
        anchorEl,
        ariaLabel: `Archived threads in ${project.name || project.id}`,
        // Restore is the ONLY action here, deliberately — but the row said nothing
        // about that, so an archived thread looked as though it simply had no
        // delete, no merge back and no way to reach its checkout, even though the
        // server accepts all three on an archived thread. A branched-then-archived
        // thread's checkout was two undisclosed steps from any A10 surface (I13).
        itemsHtml: rows.map((row) => (
            `<button type="button" role="menuitem" data-prm="restore:${escapeHtml(row.id)}" title="Restore this thread. Merging it back, its changes and its checkout become reachable again once it is active.">${escapeHtml(row.name)}</button>`
        )).join('\n'),
        // Routed through `runThreadAction` rather than awaiting `ops.restore` here,
        // because this was the ONE unguarded `await ops.*` left in the module. A
        // TYPED refusal was already handled — `typedAnswer` unwraps a 409 envelope
        // to a VALUE, so `describeOutcome` + `announce` fired — but a 500, an HTML
        // error page or a transport error RE-THROWS: the rejection escaped
        // `onSelect`, and `project_create.js`'s `menu.addEventListener('click',
        // async …)` has no try/catch, so it became an unhandled rejection with no
        // owner-facing error and no refresh, leaving a stale archived row clickable.
        // `runThreadAction` already owns exactly this shape: the catch, the
        // announce, the authoritative refresh on BOTH paths, and the refocus.
        onSelect: async (action) => {
            const id = String(action || '').startsWith('restore:') ? action.slice(8) : '';
            if (!id) return;
            const row = rows.find((thread) => String(thread.id) === id) || { name: id };
            // The id stays the STRING the menu carried — that is what the route
            // takes and what this gesture has always sent.
            await runThreadAction('restore', project, { ...row, id }, {
                apiClient, onChanged, ask, ops,
            });
            if (anchorEl?.isConnected) anchorEl.focus();
        },
    });
    return rows;
}

// ---------------------------------------------------------------------------
// The CENTRE stage
// ---------------------------------------------------------------------------

/**
 * Create the `#page-thread` centre page: an in-flow header bar (mobile nav
 * toggle, project/thread title, thread menu, close) above the mount point a
 * chat instance attaches to.
 *
 * ONE page element hosts every thread rather than one page per thread: `showPage`
 * keys on a stable page name, and the single-live-instance policy means at most
 * one instance is mounted here anyway.
 */
export function createThreadStage({ content, onClose, onMenu }) {
    const page = document.createElement('section');
    page.id = 'page-thread';
    page.className = 'page thread-stage';
    page.innerHTML = `
        <div class="thread-stage-bar project-panel-bar">
            <div class="app-page-leading">${renderMobileNavToggle()}</div>
            <div class="thread-stage-heading">
                <span class="thread-stage-project" id="thread-stage-project"></span>
                <h2 class="thread-stage-title project-panel-title app-page-title" id="thread-stage-title"></h2>
            </div>
            <button type="button" class="nav-project-kebab thread-stage-menu" id="thread-stage-menu" title="Thread actions" aria-label="Thread actions">⋯</button>
            <button type="button" class="project-panel-close thread-stage-close" id="thread-stage-close" title="Close thread" aria-label="Close thread">×</button>
        </div>
        <div class="thread-stage-body" id="thread-stage-body"></div>
    `;
    content.appendChild(page);
    const titleEl = page.querySelector('#thread-stage-title');
    const projectEl = page.querySelector('#thread-stage-project');
    const menuBtn = page.querySelector('#thread-stage-menu');
    page.querySelector('#thread-stage-close')?.addEventListener('click', () => onClose?.());
    menuBtn?.addEventListener('click', () => onMenu?.(menuBtn));
    return {
        page,
        body: page.querySelector('#thread-stage-body'),
        menuAnchor: menuBtn,
        setTitle(project, thread) {
            projectEl.textContent = project?.name || project?.id || '';
            titleEl.textContent = thread?.name || projectEl.textContent;
            // Thread #0 IS the project: showing the project name twice would read
            // as two different rooms with the same name.
            projectEl.hidden = Number(thread?.id) === MAIN_THREAD_ID;
        },
    };
}
