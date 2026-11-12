/**
 * The sidebar's Projects list (docs/DESIGN.md "Shell"): its keyed rows, their marks and names, the collapse, and the
 * way to every Project older than the recent feed.
 *
 * `/api/state` carries a compact feed, the pins and the Projects with the newest messages; app.js decides what it says
 * (unread, the open room, the read cursors) and hands its rows over with that read's request order. Older
 * Projects are reached by name in Find a project or by Show more, from the registry's complete list
 * (`GET /api/projects?all=1`): read as a find or a browse starts and again, while one is open, when the registry
 * announces a change (`refresh`) or, for a browse, when a room's newest message moved (`messagesRevision`, also beyond
 * the feed); never on a timer. Nothing here acknowledges a read. A row opens through app.js: one
 * the feed carries at once, as it shows; one older than the feed names its id and app.js reads its room first. The
 * unread app.js shows for the Projects is what this list shows: the feed's rooms, hidden or not, and the older rooms
 * an open find or browse lists, never one only a held list names (`unreadRows`; `onShown` and `onListRead` say when it
 * may have changed). Whether rooms lie beyond the feed before that list is read comes from the state's complete set of
 * Project rooms (`reserved`), never from a guessed cap.
 *
 * A read is as fresh as its request. Requests are ordered by the state reads' own generation (state_snapshots.js):
 * a feed carries the generation it was read under, and a list read or a confirmed change takes the next one, so two
 * reads in one millisecond keep their order. Once a complete list is held it keeps each Project's newest observed
 * row with its request order: a feed row replaces an older one and stays after it leaves the feed, a complete
 * list drops what it omits unless something newer named it, and a feed older than that list adds nothing back (a
 * capped feed proves no absence). A failed read keeps the list, says so with its time and offers to try again; while
 * a read is unanswered or failed, nothing is said to be missing. The list owns its order, at rest and in a browse
 * alike: pins, then the newest message (`last_message_at`; unread is a mark, never a rank); a browse keeps every row
 * it showed. A repaint away from the list's top keeps the first row in view whose place did not change where it was,
 * and the keyboard on its row wherever it moved; a find keeps its rows where they stand while words are in the
 * field. Collapsing the Projects ends the find or the browse.
 */
import { fmt, fmtInto, tr } from './i18n.js';
import { SHELL_ICONS, controlIcon } from './page_icons.js';
import { IDLE_SUMMARY, markWords } from './project_activity.js';
import { pinWords, repositionProjectRowMenu } from './project_create.js';
import { createStatusMark, markView, paintStatusMark } from './status_mark.js';

const PAGE = 50;  // Show more adds a page; Show all projects (after a find with no match) opens the rest
export const PROJECT_LIST_TIMEOUT_MS = 30000;
const ELIGIBLE = new Set(['active', 'deleting']);
const listable = (row) => Boolean(row?.id) && ELIGIBLE.has(String(row.lifecycle || 'active'));
const isDeleting = (row) => String(row?.lifecycle || 'active') === 'deleting';

/** A name and a query compare as the text a person reads: compatibility forms, accents and case folded, every run of
 *  space one. Each word of the query must be inside the name, in any order; nothing beyond the letters is guessed. */
function foldName(text) {
    return String(text ?? '').normalize('NFKD').replace(/\p{M}+/gu, '').toLowerCase().replace(/\s+/g, ' ').trim();
}

/** A mark's words go into its place's name (its tooltip and accessible name): the hue is never the only way it is
 *  told. The Projects header's toggle (app.js) and every row here name themselves this way. */
export function setActivityOwnerName(owner, baseName, words) {
    if (!owner) return;
    const name = words ? `${baseName} · ${words}` : baseName;
    owner.setAttribute('aria-label', name);
    owner.title = name;
}

// The server's order (projects_summary): the newest message, else the activity of a store still dating its history.
const recency = (row) => String(row.last_message_at || row.last_active_at || row.updated_at || row.created_at || '');
// The registry's own order (project_closure.select_projects): that time, then id, newest first.
const newestFirst = (a, b) => {
    const [x, y] = [recency(a), recency(b)];
    if (x !== y) return x < y ? 1 : -1;
    return String(a.id) < String(b.id) ? 1 : String(a.id) > String(b.id) ? -1 : 0;
};
// The list's order: pins first, then the server's (equal times, empty ones too, by id, newest first).
export const listOrder = (a, b) => ((a.pinned === true) !== (b.pinned === true) ? (a.pinned === true ? -1 : 1)
    : newestFirst(a, b));
// A row keeps its place while this key does.
const placeKey = (row) => `${row?.pinned === true}|${row ? recency(row) : ''}`;

export function createProjectsNav({ section, header, list, toggle, feed, feedOrder, nextOrder, reserved, listableCount, activeId,
    expanded, isUnread, readList, onListRead, onShown, open, openMenu, pin, isPinPending, messagesRevision = () => null }) {
    // `browse`: the ids a browse shows (sorted with the list) and the rows Show more still owes; `found`: the ids a
    // find shows (null: the owner's new words, laid out anew) and how many rows besides the pins it may show.
    const state = { query: '', browse: null, found: null, budget: PAGE, resume: null, collapsed: null, handoff: null };
    // Once a complete list was read: each Project's newest observed row and its read's request order (`at`).
    let held = null;       // { rows: Map id → { row, at }, at: the list's order, time: its clock time, folded, revision }
    let reading = null;    // the read in flight: { controller }
    let failure = '';      // why the last read failed; '' once one answered
    let listed = new Set();  // the rooms an open find or browse showed at the last paint; none at rest
    let activity = { byChat: new Map() };  // the census index the last patch painted from
    let pressedPin = null;  // the drawn Pin a pointer's last press in the list began on; null after any other press
    let destroyed = false;
    const folded = new Map();  // name → its folded form

    const feedRows = () => (Array.isArray(feed()) ? feed() : []);
    const searching = () => state.query.trim() !== '';
    const beyondFeed = () => searching() || Boolean(state.browse);
    const failed = () => Boolean(failure) && !reading;
    // A feed row carries its unread from app.js; a listed row's is read from the shared cursors the same way.
    const unreadOf = (row) => (typeof row._unread === 'boolean' ? row._unread : !isDeleting(row) && isUnread(row));
    const foldOf = (row) => {
        const name = String(row.name || row.id || '');
        if (!folded.has(name)) folded.set(name, foldName(name));
        return folded.get(name);
    };
    const readTime = () => new Date(held.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const plain = (row) => { const { _unread, ...copy } = row; return copy; };  // unread is read from the cursors

    // A feed folds in once: its rows replace older observations; a room no row names is added only when the feed was
    // requested after the last complete list (a room created since), never one that list omitted.
    function observeFeed() {
        const at = feedOrder();
        if (!held) return;
        const fresh = at > held.folded;
        if (fresh) held.folded = at;
        for (const row of feedRows()) {
            const seen = held.rows.get(row.id);
            if (fresh && (seen ? seen.at < at : at > held.at)) {
                held.rows.set(row.id, { row: plain(row), at });
            } else if (seen && row._unread && Number(row.visible_revision) > Number(seen.row.visible_revision)) {
                // A state response requested before a complete-list read can land after it. Its feed row still carries
                // the authoritative newer unread revision; merge that fact without letting stale feed names or order
                // resurrect a room the list has already removed.
                held.rows.set(row.id, { ...seen, row: { ...seen.row, visible_revision: row.visible_revision } });
            }
        }
    }
    function observeList(rows, at, time, revision) {
        const next = new Map();
        for (const row of rows) {
            const seen = held?.rows.get(row.id);
            next.set(row.id, seen && seen.at > at ? seen : { row, at });
        }
        for (const [id, seen] of held?.rows || []) if (!next.has(id) && seen.at > at) next.set(id, seen);
        held = { rows: next, at, time, folded: held?.folded || 0, revision };
    }
    // Every Project's newest row: the held list's, or the feed's before one is read.
    function facts() {
        if (!held) return new Map(feedRows().map((row) => [row.id, row]));
        observeFeed();
        return new Map([...held.rows].map(([id, seen]) => [id, seen.row]));
    }
    // What this sidebar shows unread: the feed's rooms, hidden or not (collapsed, the header speaks for them), and the
    // rooms an open find or browse lists. A room only the held list names is shown nowhere here, so it counts nowhere.
    const unreadRows = () => {
        const rows = facts();
        return [...new Set([...feedRows().map((row) => row.id), ...listed])].map((id) => rows.get(id))
            .filter((row) => listable(row) && unreadOf(row));
    };
    // The resting list: the feed's rooms, each as its newest observation tells it, in the list's order.
    const resting = (rows) => feedRows().map((row) => rows.get(row.id)).filter(listable).sort(listOrder);
    // Pins lead any order: a row pinned meanwhile ends the pins, one unpinned starts the rest.
    const pinsFirst = (ids, rows) => [...ids.filter((id) => rows.get(id).pinned === true),
        ...ids.filter((id) => rows.get(id).pinned !== true)];
    // What a press hands app.js: a row the feed carries opens as it shows, at once; an older row may be as old as the
    // list it came from, so it names only its id and name and its room is read first.
    const pressed = (row) => (feedRows().some(({ id }) => id === row.id) ? row : { id: row.id, name: row.name });
    // Before the complete list is read: does the state know a Project room the feed does not show?
    const roomsBeyond = () => {
        const total = listableCount?.();
        if (Number.isSafeInteger(total) && total >= 0) {
            return total > feedRows().filter(listable).length;
        }
        const shown = new Set(feedRows().map((row) => Number(row.chat_id)));
        return [...(reserved() || [])].some((chat) => !shown.has(Number(chat)));
    };

    // ---- the find field, between the header and the list (A1: always there while the list is open) -------------
    const find = document.createElement('div');
    find.className = 'nav-projects-find';
    find.setAttribute('role', 'search');
    const glyph = document.createElement('span');
    glyph.className = 'nav-find-glyph';
    glyph.setAttribute('aria-hidden', 'true');
    glyph.innerHTML = SHELL_ICONS.search;
    const input = document.createElement('input');
    input.type = 'search';
    input.className = 'ui-control nav-find-input';
    input.placeholder = 'Find a project';
    input.setAttribute('aria-label', 'Find a project by name');
    input.setAttribute('aria-controls', list.id);
    input.autocomplete = 'off';
    input.spellcheck = false;
    input.setAttribute('enterkeyhint', 'search');
    const clear = document.createElement('button');
    clear.type = 'button';
    clear.className = 'nav-projects-add nav-find-clear';
    clear.innerHTML = SHELL_ICONS.close;
    clear.title = 'Clear search';
    clear.setAttribute('aria-label', 'Clear search');
    clear.hidden = true;
    const status = document.createElement('span');
    status.className = 'nav-find-status';
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    status.dataset.i18nSkip = '';
    find.append(glyph, input, clear, status);
    header.after(find);

    // ---- the list's own commands: Project-row geometry in the meta ink, so none reads as one more Project --------
    const command = (className) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `nav-projects-more${className ? ` ${className}` : ''}`;
        return button;
    };
    const more = command('');
    const showAll = command('');
    const clearCommand = command('nav-projects-clear');
    const retry = command('');
    const note = document.createElement('p');
    note.className = 'nav-projects-note';
    const labelled = (button, key, params = {}) => {
        button.replaceChildren();
        const words = document.createElement('span');
        fmtInto(words, key, params);
        if (button === more) {
            button.insertAdjacentHTML('afterbegin', controlIcon('chevron', 18));
            button.firstElementChild.setAttribute('aria-hidden', 'true');
        }
        button.append(words);
        return button;
    };
    const said = (text) => { note.textContent = text; note.dataset.i18nSkip = ''; return note; };
    const busy = (button, key) => { labelled(button, key); button.setAttribute('aria-busy', 'true'); return button; };
    const moreOf = (left) => labelled(more, 'Show {n} more', { n: Math.min(PAGE, left) });

    // ---- what the list shows ------------------------------------------------------------------------------------
    function view() {
        const rows = facts();
        const rest = resting(rows);
        if (state.collapsed) return { shown: rest, after: [] };
        const all = [...rows.values()].filter(listable);
        if (searching()) return found(rows, all);
        if (state.browse) return browsed(rows, rest, all);
        const ids = new Set(rest.map((row) => row.id));
        const left = held ? all.filter((row) => !ids.has(row.id)).length : 0;
        return { shown: rest, after: left ? [moreOf(left)] : !held && roomsBeyond() ? [labelled(more, 'Show more')] : [] };
    }

    // A browse shows every room it has shown, sorted with the list: the feed's rooms (a new one too), the rows Show more
    // brought and, once the complete list is read, the next ones it owes; a room the registry no longer lists goes.
    function browsed(rows, rest, all) {
        const shown = new Set([...state.browse.shown, ...rest.map((row) => row.id)].filter((id) => listable(rows.get(id))));
        if (held && state.browse.owed) {
            for (const row of all.filter((item) => !shown.has(item.id)).sort(newestFirst).slice(0, state.browse.owed)) {
                shown.add(row.id);
            }
            state.browse.owed = 0;
        }
        state.browse.shown = shown;
        const after = [];
        if (!held) {
            after.push(...(failed() ? [said(fmt('Could not load more projects: {error}', { error: failure })),
                labelled(retry, 'Try again')] : [busy(more, 'Loading projects…')]));
        } else {
            if (failed()) {
                after.push(said(fmt('Could not read all projects again: {error}. Older rows are as read at {time}.',
                    { error: failure, time: readTime() })), labelled(retry, 'Try again'));
            }
            const left = all.length - shown.size;
            if (left > 0) after.push(moreOf(left));
        }
        return { shown: [...shown].map((id) => rows.get(id)).sort(listOrder), after,
            waiting: Boolean(state.browse.owed) && !failed() };
    }

    // A find keeps the rows it shows where they stand while their names still match; a match it does not show yet
    // (the complete list arriving, a renamed room) joins at the end while the find's page has room.
    function found(rows, all) {
        const words = foldName(state.query).split(' ').filter(Boolean);
        const matches = (row) => listable(row) && words.every((word) => foldOf(row).includes(word));
        const hits = all.filter(matches).sort(listOrder);
        const order = (state.found || []).filter((id) => matches(rows.get(id)));
        const known = new Set(order);
        let others = order.filter((id) => rows.get(id).pinned !== true).length;
        for (const row of hits) {
            if (known.has(row.id) || (row.pinned !== true && others >= state.budget)) continue;
            order.push(row.id);
            if (row.pinned !== true) others += 1;
        }
        state.found = pinsFirst(order, rows);
        const after = [];
        let spoken;
        if (failed()) {
            spoken = fmt(!held ? 'Could not search all projects: {error}'
                : order.length ? 'Could not search all projects again: {error}. These matches are from the list read at {time}.'
                    : 'Could not search all projects again: {error}.', { error: failure, time: held ? readTime() : '' });
            after.push(said(spoken), labelled(retry, 'Try again'));
        } else if (!held || (reading && !hits.length)) {
            spoken = fmt('Searching all projects…');  // unanswered: nothing is said to be missing
            after.push(said(spoken));
            // Read again: both ways on stay where the last answer put them, the keyboard with them.
            if (held) after.push(labelled(clearCommand, 'Clear search'), labelled(showAll, 'Show all projects'));
        } else if (!hits.length) {
            // The names hold no such words: said of the names, never of the Projects, with both ways on.
            spoken = fmt('No project names contain “{query}”.', { query: state.query.trim() });
            after.push(said(spoken), labelled(clearCommand, 'Clear search'), labelled(showAll, 'Show all projects'));
        } else {
            spoken = fmt(hits.length === 1 ? '{n} project found' : '{n} projects found', { n: hits.length });
        }
        if (held && hits.length > order.length) after.push(moreOf(hits.length - order.length));
        if (status.textContent !== spoken) status.textContent = spoken;
        return { shown: state.found.map((id) => rows.get(id)), after };
    }

    // ---- painting -----------------------------------------------------------------------------------------------
    function itemFor(project, keyed) {
        const deleting = isDeleting(project);
        let item = keyed.get(project.id);
        keyed.delete(project.id);
        if (item && item.classList.contains('is-deleting') !== deleting) { item.remove(); item = null; }
        if (!item) {
            item = document.createElement('div');
            item.dataset.navProjectId = project.id;
            item.className = `nav-project-item${deleting ? ' is-deleting' : ''}`;
            const btn = document.createElement('button');
            btn.className = 'nav-row nav-project-row';
            btn.type = 'button';
            btn.dataset.projectId = project.id;
            const label = document.createElement('span');
            label.className = 'nav-row-label';
            btn.append(label, createStatusMark(document, 'nav-mark'));
            item.appendChild(btn);
            if (deleting) {
                const deletingStatus = document.createElement('span');
                deletingStatus.className = 'nav-project-deleting-status';
                deletingStatus.textContent = 'Deleting…';
                item.appendChild(deletingStatus);
            } else {
                // Pressing a row opens its Project; pressing the open one again keeps it open (like every reference).
                const openRow = (pointer) => open(pressed(item.project), { pointer });
                btn.addEventListener('click', (event) => openRow(event.detail > 0));
                // Sibling actions, never nested buttons: Pin over the row's icon column, ⋯ after its name.
                // Semantic activation of the focused glyph or a single press begun on it asks for its painted state.
                // On a hybrid screen a finger can begin on the row, then hover draws the glyph under its click:
                // that click opens the row. A double click's second press may hit a moved neighbour; it pins nothing.
                const pinControl = document.createElement('button');
                pinControl.type = 'button';
                pinControl.className = 'nav-project-pin';
                pinControl.addEventListener('click', (event) => {
                    if (event.detail === 0 || (event.detail === 1
                        && (pressedPin === pinControl || pinControl.matches(':focus-visible')))) {
                        pin(item.project, pinControl.dataset.action === 'pin');
                    } else if (pressedPin !== pinControl) openRow(true);
                });
                item.prepend(pinControl);
                const kebab = document.createElement('button');
                kebab.type = 'button';
                kebab.className = 'nav-project-kebab';
                kebab.innerHTML = SHELL_ICONS.more;  // the shell's one ⋯ (page_icons.js)
                kebab.title = tr('project.kebab_title', 'Project actions');
                kebab.addEventListener('click', (event) => {
                    event.stopPropagation();
                    openMenu(item, { anchor: kebab });
                });
                item.appendChild(kebab);
            }
        }
        item.project = project;
        item.paintedPlaceKey = placeKey(project);  // the shared project may change before the next repaint
        item.toggleAttribute('data-pinned', project.pinned === true);
        const btn = item.querySelector('.nav-project-row');
        btn.disabled = deleting;
        const label = btn.querySelector('.nav-row-label');
        if (label.textContent !== (project.name || project.id)) label.textContent = project.name || project.id;
        const active = project.id === activeId();
        btn.classList.toggle('active', active);
        if (active) btn.setAttribute('aria-current', 'page');
        else btn.removeAttribute('aria-current');
        const pinControl = item.querySelector('.nav-project-pin');
        if (pinControl) {
            const action = project.pinned === true ? 'unpin' : 'pin';
            if (pinControl.dataset.action !== action) {
                pinControl.dataset.action = action;
                pinControl.innerHTML = SHELL_ICONS[action];  // the pin, or the slashed pin of Unpin (page_icons.js)
            }
            const pending = isPinPending(project.id);
            const words = pinWords(action === 'unpin', pending);
            if (pending) pinControl.setAttribute('aria-disabled', 'true');
            else pinControl.removeAttribute('aria-disabled');
            pinControl.title = fmt(words);
            pinControl.setAttribute('aria-label', fmt(`${words} {name}`, { name: project.name || project.id }));
        }
        const kebab = item.querySelector('.nav-project-kebab');
        if (kebab) kebab.setAttribute('aria-label', fmt('Actions for {name}', { name: project.name || project.id }));
        else item.querySelector('.nav-project-deleting-status').title = project.delete_error || 'Cancellation and cleanup are in progress';
        return item;
    }

    const firstInView = () => {
        const top = Math.max(list.getBoundingClientRect().top, 0);
        return [...list.children].find((el) => el.dataset.navProjectId && el.getBoundingClientRect().bottom > top) || null;
    };
    // The rows in view and their tops, measured before a repaint moves anything; none while the list's first row is
    // in view, so what a new message brings to the top shows there.
    function rowsInView() {
        const top = Math.max(list.getBoundingClientRect().top, 0);
        const rows = [...list.children].filter((el) => el.dataset.navProjectId && !el.hidden);
        if (!rows.length || rows[0].getBoundingClientRect().top >= top - 0.5) return [];
        const bottom = Math.min(list.getBoundingClientRect().bottom, window.innerHeight);
        const seen = [];
        for (const el of rows) {
            const box = el.getBoundingClientRect();
            if (box.top >= bottom) break;
            if (box.bottom > top) seen.push([el, box.top]);
        }
        return seen;
    }
    // Scroll whatever scrolls (the list, or the whole column in a short window) by what the anchor moved.
    function keepOnScreen(anchor, top) {
        let moved = anchor.getBoundingClientRect().top - top;
        for (let box = list; box && Math.abs(moved) >= 0.5; box = box.parentElement) {
            if (box.scrollHeight <= box.clientHeight) continue;
            const before = box.scrollTop;
            box.scrollTop = before + moved;
            moved -= box.scrollTop - before;
        }
    }

    // Keyed: a row keeps its node (and so its focus, its menu's anchor and its mark's phase) however the list moves.
    function paintRows() {
        if (destroyed) return;
        const collapsed = !expanded();
        if (collapsed && state.collapsed === false) {  // a collapse ends the find and the browse with the list
            // What has the keyboard hides or goes with them, only a pinned row stays: the toggle takes it first, also
            // from a press that let go of it on the way (a pointer focuses no button in WebKit).
            const at = document.activeElement;
            const letGo = beyondFeed() && (!at || at === document.body);
            if (letGo || find.contains(at) || (list.contains(at) && !at.closest('[data-pinned]'))) {
                toggle?.focus({ preventScroll: true });
            }
            Object.assign(state, { query: '', browse: null, found: null, handoff: null, resume: null });
            input.value = '';
            clear.hidden = true;
        }
        state.collapsed = collapsed;
        more.removeAttribute('aria-busy');
        const beyond = beyondFeed() && !collapsed;
        const measured = rowsInView();
        const { shown, after, waiting } = view();
        // The place being read is held by the first row in view whose place in the order holds (its pin and time):
        // a row a new message moves never takes the view with it (style.css turns the engine's anchoring off).
        const next = new Map(shown.map((row) => [row.id, row]));
        const [anchor, anchorTop] = measured.find(([el]) => next.has(el.dataset.navProjectId)
            && el.paintedPlaceKey === placeKey(next.get(el.dataset.navProjectId))) || [];
        // The rooms an open find or browse lists count toward the unread app.js shows, so a change of them is said.
        const showing = new Set(beyond ? shown.map((row) => row.id) : []);
        const relisted = showing.size !== listed.size || [...showing].some((id) => !listed.has(id));
        listed = showing;
        const keyed = new Map([...list.children].filter((el) => el.dataset.navProjectId)
            .map((el) => [el.dataset.navProjectId, el]));
        const focused = document.activeElement;
        const nodes = [...shown.map((project) => itemFor(project, keyed)), ...after];
        nodes.forEach((node, index) => {
            if (list.children[index] !== node) list.insertBefore(node, list.children[index] || null);
        });
        const keep = new Set(nodes);
        for (const el of [...list.children]) if (!keep.has(el)) el.remove();
        list.toggleAttribute('data-finding', searching() && !collapsed);
        if (!searching() && status.textContent) status.textContent = '';
        syncCollapse();  // visibility and the pin group's gap belong to the geometry we restore
        if (anchor?.isConnected) keepOnScreen(anchor, anchorTop);
        // Moving an existing node can blur it even though its identity survives.
        if (focused?.isConnected && list.contains(focused) && focused !== document.activeElement) focused.focus({ preventScroll: true });
        if (state.handoff && !waiting) handOff(state.handoff, nodes);
        repositionProjectRowMenu();  // anchoring may have scrolled its row after visibility settled
        if (relisted) onShown();  // app.js says its number, header and ☰ again, marks in place: no read, no render
        // A browse reads the complete list again once a room's newest message moved since that list was asked for,
        // beyond the feed too; a read in flight or a failed one waits (its answer, or Try again, repaints).
        if (state.browse && !collapsed && held && !reading && !failure && messagesRevision() !== held.revision) load();
    }

    // A key's Show more or Show all hands the keyboard on once, when its rows have settled (read, or the read
    // failed): to the first row it brought; when none came and its command is gone, to Try again or the field.
    // Once the owner has moved the keyboard elsewhere, it stays there.
    function handOff({ before, from }, nodes) {
        state.handoff = null;
        const at = document.activeElement;
        if (at && at !== document.body && at !== from) return;
        const first = nodes.find((node) => node.dataset?.navProjectId && !before.has(node.dataset.navProjectId));
        if (first) first.querySelector('.nav-project-row')?.focus();
        else if (!from.isConnected) (retry.isConnected ? retry : input).focus();
    }

    /** The marks and names of the shown rows, in place: focus, menus and keyboard navigation stay. */
    function patch(activityIndex = activity) {
        activity = activityIndex || activity;
        for (const item of list.children) {
            const row = item.project;
            if (!row || !item.dataset.navProjectId) continue;
            const button = item.querySelector('.nav-project-row');
            const summary = activity.byChat?.get(String(row.chat_id)) || IDLE_SUMMARY;
            const deleting = isDeleting(row);
            const unreadRow = unreadOf(row) && !deleting;
            button.toggleAttribute('data-unread', unreadRow);
            // A deleting row draws only `Deleting…`; its census fact stays in its name.
            paintStatusMark(button.querySelector('.nav-mark'),
                deleting ? markView() : markView({ mark: summary.mark, owner: summary.owner, unread: unreadRow }));
            // The pin is said in the row's full name (its tooltip and accessible name); at rest nothing draws it.
            const name = `${row.name || row.id}${row.pinned === true ? ' · Pinned' : ''}`;
            setActivityOwnerName(button, deleting ? `${name} — Deleting…${row.delete_error ? ` ${row.delete_error}` : ''}` : name,
                markWords(summary, unreadRow));
        }
    }

    // Collapse is all or nothing (owner 19.10, 2A): collapsed, the Projects keep only the pinned ones, and the
    // header's mark and unread count speak for the rest (docs/DESIGN.md "Shell"); expanded, every row shows. A row
    // that hides gives the keyboard to the Projects toggle first, and an open menu of a hidden row closes.
    function syncCollapse() {
        if (destroyed) return;
        const collapsed = !expanded();
        if (collapsed !== state.collapsed) { render(); return; }
        section?.toggleAttribute('data-collapsed', collapsed);
        find.hidden = collapsed;
        let shown = 0;
        let pinsShown = false;
        let afterPins = null;
        for (const item of list.children) {
            if (!item.classList.contains('nav-project-item')) continue;
            const keep = !collapsed || item.hasAttribute('data-pinned');
            if (!keep && item.contains(document.activeElement)) toggle?.focus({ preventScroll: true });
            item.hidden = !keep;
            if (keep) shown += 1;
            // Pins lead the list; air before the first shown row after them is the group's one boundary, no caption.
            if (keep && item.hasAttribute('data-pinned')) pinsShown = true;
            else if (keep && pinsShown) afterPins ||= item;
        }
        for (const item of list.children) item.toggleAttribute('data-after-pins', item === afterPins);
        list.hidden = collapsed && shown === 0;
        // An open row menu follows its anchor once row visibility has settled, or closes with its hidden row.
        repositionProjectRowMenu();
    }

    function render() {
        if (destroyed) return;
        paintRows();
        patch();
    }

    // ---- the registry's complete list ---------------------------------------------------------------------------
    function load() {
        if (reading || destroyed) return;
        const controller = new AbortController();
        let timedOut = false;
        const timeout = setTimeout(() => {
            timedOut = true;
            controller.abort();
        }, PROJECT_LIST_TIMEOUT_MS);
        const at = nextOrder();  // after every state read requested so far, before every later one
        const time = Date.now();  // only for saying when the list shown was read
        const revision = messagesRevision();  // the rooms' message times this read answers at least
        reading = { controller };
        Promise.resolve().then(() => readList({ signal: controller.signal })).then((data) => {
            if (!Array.isArray(data?.projects)) throw new Error(fmt('the answer could not be read'));
            if (reading?.controller !== controller) return;
            observeList(data.projects.filter(listable), at, time, revision);
            failure = '';
        }).catch((err) => {
            // The last complete list stays and the failure is said beside it; without one, never "no match".
            if (reading?.controller === controller && (timedOut || err?.name !== 'AbortError')) {
                failure = timedOut ? fmt('Project list did not arrive within {seconds} seconds',
                    { seconds: Math.round(PROJECT_LIST_TIMEOUT_MS / 1000) }) : err?.message || 'unknown error';
            }
        }).finally(() => {
            clearTimeout(timeout);
            if (reading?.controller !== controller) return;
            reading = null;
            render();
            onListRead();
        });
    }

    /** The registry changed (or may have, across a reconnect): a find or browse that goes on reads its list again. A
     *  list kept after both ended counts no unread and is read again only as the next one starts. */
    function refresh() {
        if (destroyed || !beyondFeed()) return;
        reading?.controller.abort();
        reading = null;
        load();
        // No-match is no longer said while this is unanswered; rows and focus stay. A failure says nothing is missing,
        // so it stands with its Try again until this read answers.
        if (!failure) render();
    }

    function setQuery(value, { typed = false } = {}) {
        const was = searching();
        const now = value.trim() !== '';
        // Measure before changing the field or its clear button: WebKit can reveal a focused field while that
        // layout changes, losing the whole-column browsing position before we had a chance to keep it.
        if (now && !was) {
            const row = firstInView();
            state.resume = row && { id: row.dataset.navProjectId, top: row.getBoundingClientRect().top };
        }
        state.query = value;
        if (!typed && input.value !== value) input.value = value;
        clear.hidden = !value;
        if (now && !was && !state.browse) load();  // a find reads the complete list as it starts (a browse has)
        Object.assign(state, { found: null, budget: PAGE, handoff: null });
        render();
        let restored = null;
        if (now && !was) list.scrollTop = 0;  // the matches from their first (a column's scroll keeps the field)
        else if (was && !now && state.resume) {
            restored = [...list.children].find((el) => el.dataset.navProjectId === state.resume.id) || null;
            if (restored) keepOnScreen(restored, state.resume.top);
            state.resume = null;
        }
        return restored;
    }

    // After Clear or Escape the reader is back where the find began. The keyboard stays in the field while the field
    // shows (in a short window the column may have carried it off, or under the sticky head), else it goes to that
    // row, or the first one in view; neither moves the column again. Typing that empties the field is just typing.
    function settleAfterClear(restored) {
        const box = input.getBoundingClientRect();
        const shown = find.contains(document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2));
        const row = (restored?.isConnected ? restored : firstInView())?.querySelector('.nav-project-row');
        (shown || !row ? input : row).focus({ preventScroll: true });
    }

    // A browse starts from the resting rows as they stand; Show all projects owes every other room.
    const browseFrom = (owed) => ({ shown: new Set(resting(facts()).map((row) => row.id)), owed });

    // ---- the owner's keys and presses ---------------------------------------------------------------------------
    const rowStops = () => [...list.querySelectorAll(':scope > .nav-project-item:not([hidden]) > .nav-project-row:not(:disabled), :scope > .nav-projects-more')];
    const onInput = () => setQuery(input.value, { typed: true });
    const onFieldKey = (event) => {
        if (event.isComposing) return;
        if (event.key === 'ArrowDown') {
            const first = rowStops()[0];
            if (first) { event.preventDefault(); first.focus(); }
        } else if (event.key === 'Enter' && searching()) {
            const first = list.querySelector(':scope > .nav-project-item:not([hidden]) > .nav-project-row:not(:disabled)');
            event.preventDefault();
            if (first) first.click();
        } else if (event.key === 'Escape' && input.value) {
            event.preventDefault();  // the field's own Escape: the drawer or the room stays (layers.js)
            settleAfterClear(setQuery(''));
        }
    };
    const onListKey = (event) => {
        if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
        const stops = rowStops();
        const at = stops.indexOf(document.activeElement);
        if (at < 0) return;
        event.preventDefault();
        const next = stops[at + (event.key === 'ArrowDown' ? 1 : -1)];
        if (next) next.focus();
        else if (event.key === 'ArrowUp' && !find.hidden) input.focus();
    };
    const onMore = (event) => {
        if (more.getAttribute('aria-busy') === 'true') return;
        state.handoff = event.detail === 0
            ? { before: new Set([...list.children].map((el) => el.dataset.navProjectId).filter(Boolean)), from: more } : null;
        if (searching()) state.budget += PAGE;
        else if (state.browse) state.browse.owed += PAGE;
        else {
            state.browse = browseFrom(PAGE);
            load();  // a browse reads the complete list as it starts
        }
        render();
    };
    const onClearCommand = () => settleAfterClear(setQuery(''));
    const onShowAll = (event) => {
        state.browse = browseFrom(Number.POSITIVE_INFINITY);
        const restored = setQuery('');
        if (event.detail !== 0) { settleAfterClear(restored); return; }
        // A key: the first room beyond the recent ones takes the keyboard (the command itself is gone).
        state.handoff = { before: new Set(feedRows().map((row) => row.id)), from: showAll };
        render();
    };
    const onRetry = (event) => {
        state.handoff = event.detail === 0
            ? { before: new Set([...list.children].map((el) => el.dataset.navProjectId).filter(Boolean)), from: retry } : null;
        load();
        render();
    };
    const onListPress = (event) => { pressedPin = event.target.closest?.('.nav-project-pin') || null; };
    const bindings = [[input, 'input', onInput], [input, 'keydown', onFieldKey], [clear, 'click', onClearCommand],
        [list, 'keydown', onListKey], [list, 'pointerdown', onListPress], [more, 'click', onMore],
        [clearCommand, 'click', onClearCommand], [showAll, 'click', onShowAll], [retry, 'click', onRetry]];
    for (const [node, type, handler] of bindings) node.addEventListener(type, handler);

    return {
        /** The feed changed (app.js renderProjectsNav, a cursor move, a pin): repaint the rows from it. */
        paint: paintRows,
        patch,
        syncCollapse,
        refresh,
        /** A change the server just confirmed for a Project (a pin, a deletion begun): newest for that row, so a read
         *  requested before it cannot undo it. Before a complete list is held, the feed row app.js changed speaks. */
        noteChange(id, fields) {
            const seen = held?.rows.get(id);
            if (seen) held.rows.set(id, { row: { ...seen.row, ...fields }, at: nextOrder() });
        },
        /** Every Project this list shows unread, the feed's hidden or not and those an open find or browse lists: what
         *  the header's number, ☰ and the re-read of the shared cursors go by. */
        unreadRows,
        /** The row shown for an id as its freshest read tells it, with its unread said; null when no read names it. */
        row(id) {
            const row = facts().get(id);
            return row ? { ...row, _unread: unreadOf(row) } : null;
        },
        destroy() {
            destroyed = true;
            reading?.controller.abort();
            reading = null;
            for (const [node, type, handler] of bindings) node.removeEventListener(type, handler);
            find.remove();
        },
    };
}
