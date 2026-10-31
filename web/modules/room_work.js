/**
 * The room's work status in its bar (DESIGN "Room work status"). The bar's one status control, right after
 * the room's title, paints the word, tone and mark the chat's producer gives it (the room in the server's work
 * projection, work_status.roomStatusView; status_mark.js draws the mark) and opens this room's work as one floating
 * list under the bar's capsules, in Main and a Project alike. Presentation only, over the chat instance's own owners;
 * it keeps no task state, phase vocabulary, poller, pager, question index or permission:
 *   membership  this room's works (`works()`, the instance's reconciled projection: one row per work, helpers and
 *               Continue episodes inside it), the owner's unconfirmed sends (`pendingSubmissions`); a Project with
 *               nothing else keeps its latest represented root card
 *   words       work_status.workRowView: the work's word, its line beside it and its mark
 *   controls    the work's own controls (work_status.openWorkControls), offered only while the current projection
 *               confirms them; a loaded card stops through its own card path
 *   question    the instance's revealQuestion (chat_decision.js); it navigates and never answers
 *   reading     the instance's reading intent and timeline anchors (`Back to reading`); an unloaded row reads
 *               the page around its task through the instance's pager (`around`), Back its own page (`revisit`)
 */
import { fetchTaskDetail, fetchTaskDetailStrict } from './api_client.js';
import { bindContentButton } from './chat_activity.js';
import { compareHistoryPosition } from './chat_history_replay.js';
import { createHistoryReveal } from './chat_history_reveal.js';
import { fmt, tr } from './i18n.js';
import { taskCancelPending } from './log_events.js';
import { desiredLiveCardPhase, replayTerminalPhase } from './task_phase_chip.js';
import { createStatusMark, markView, paintStatusMark } from './status_mark.js';
import { closeTaskControlMenu, createTaskResumeButton, requestStop, taskControlBusy } from './task_control_menu.js';
import {
    OUTCOME_TONES, amberNote, clock, openWorkControls, workControls, workMenuActions, workQuestion, workRowView,
} from './work_status.js';
import { showToast } from './toast.js';
import { motionMs, reducedMotion } from './motion.js';

// The closed list rises back as it fades (room_work.css rw-unfold, reversed).
const SHADE_FOLDS = [{ opacity: 1, transform: 'none' }, { opacity: 0, transform: 'translateY(-6px)' }];
const NAME_CHARS = 160;
const REPLY_CHARS = 60;
const GROUP_RANK = { needs: 0, progress: 1, waiting: 2, unconfirmed: 3, latest: 4 };
const clean = (value) => String(value ?? '').replace(/\s+/g, ' ').trim();
const listOf = (value) => (Array.isArray(value) ? value : []);
const clip = (text, max) => (text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text);
let sequence = 0;
// A status sentence's own dot speaks only at rest: the room at rest says Online beside its green dot.
const AT_REST = Object.freeze({ shape: 'dot', hue: 'ok' });

/** The latest represented root card: the newest unfinished one, else the newest of all. */
export function latestRepresentedRoot(records = []) {
    const roots = records.filter((record) => !record.isSubagent && record.root?.isConnected)
        .sort((a, b) => Number(a.root.dataset?.ts || 0) - Number(b.root.dataset?.ts || 0)
            || compareHistoryPosition(a.historyPosition, b.historyPosition));
    return roots.filter((record) => !record.finished).at(-1) || roots.at(-1) || null;
}

/** A card's own words as a row's view (a settled row, the Project's latest root): a known outcome keeps its tone and
 *  its dot; any other state of a card is unconfirmed here (a ring). The owner's move is a work's turn, never a card's. */
export function cardRowView(view = {}, { settled = false } = {}) {
    const outcome = OUTCOME_TONES[view.phase] ? String(view.phase) : '';
    return { word: String(view.text || ''), secondary: String(view.secondary || ''),
        tone: OUTCOME_TONES[outcome] || (settled ? 'ended' : 'unknown'), mark: outcome ? 'none' : 'ring', owner: false,
        outcome, focus: '' };
}

/** The card a work's row goes to: its current episode's, else another of its episodes' root cards that is shown
 *  (Continue episodes share one card, task_episodes.js). */
export function workCard(work, records) {
    for (const id of [work?.episode?.task_id, work?.work_id, ...listOf(work?.members?.ids)]) {
        const record = records?.get?.(String(id || ''));
        if (record && !record.isSubagent && record.root?.isConnected && !record.root.hidden) return record;
    }
    return null;
}

/**
 * Mount the status control and its list for one chat instance. The caller passes its own owners by their
 * names; `update()` repaints on the next frame after any of them changed.
 */
/**
 * A list hung under a bar's capsules (room_work.css): from the status capsule's left edge to the right edge of the
 * bar's last visible control (a toolbar's own box reaches past its capsules), below the capsule. Measured where the bar
 * really laid them out and written as three custom properties on the list's anchor. A control inside a list (this
 * room's work, or the consciousness chip's) is never the bar's edge.
 */
export function hangUnderCapsules(bar, anchor) {
    const titleRow = bar?.querySelector('.app-page-title-row');
    if (!titleRow || !anchor?.isConnected) return;
    const edge = anchor.getBoundingClientRect();
    const capsule = titleRow.getBoundingClientRect();
    const ends = [...bar.querySelectorAll('button, summary')]
        .filter((node) => node.getClientRects().length && !titleRow.contains(node) && !node.closest('.rw-shade-anchor'))
        .map((node) => node.getBoundingClientRect().right);
    const end = ends.length ? Math.max(...ends) : edge.right;
    const px = (value) => `${Math.round(value)}px`;
    anchor.style.setProperty('--rw-left', px(Math.max(0, capsule.left - edge.left)));
    anchor.style.setProperty('--rw-right', px(Math.max(0, edge.right - end)));
    anchor.style.setProperty('--rw-top', px(capsule.bottom - edge.top));
}

export function createRoomWork({
    page, asPanel = false, ws = null, statusBadge, messagesDiv, reading, stateSnapshots = null, revealQuestion = null,
    works = () => null, liveCardRecords: records = new Map(),
    pendingSubmissions: pending = new Map(), pendingSuggestedNames: names = new Map(),
    cancelRunFromCard = null, refreshHeaderControlState = async () => {},
    captureVisibleTimelineAnchor = () => null, restoreVisibleTimelineAnchor = () => false, hostReady = () => true,
    historyPager = null, showLatest = null,
} = {}) {
    if (!page || !statusBadge || !messagesDiv) return null;
    const doc = page.ownerDocument;
    const uid = `room-work-${(sequence += 1)}`;
    const scope = asPanel ? 'project' : 'main';
    const frame = globalThis.requestAnimationFrame || ((fn) => setTimeout(fn, 16));
    // The one freshness predicate: the socket today; a shared connection epoch replaces it here.
    const connected = () => ws?.isConnected?.() !== false;
    const refresh = () => refreshHeaderControlState(true);
    const element = (tag, className = '', text = '') => {
        const node = doc.createElement(tag);
        if (className) node.className = className;
        if (text) node.textContent = text;
        return node;
    };
    const decorative = (node) => { node.setAttribute('aria-hidden', 'true'); return node; };

    // ---- the status control: the room's real status badge inside one disclosure button
    const trigger = element('button', 'rw-trigger ui-chrome-control');
    trigger.type = 'button';
    trigger.dataset.scope = scope;
    trigger.setAttribute('aria-haspopup', 'dialog');
    trigger.setAttribute('aria-expanded', 'false');
    trigger.setAttribute('aria-controls', `${uid}-surface`);
    // The status's one mark follows its word, as on the sidebar's rows.
    const mark = createStatusMark(doc);
    // Said as well as drawn: what waits for the owner is named in the control's name where the word does not say it.
    const waitNote = element('span', 'rw-sr');
    const caret = decorative(element('span', 'rw-caret'));
    const badgeParent = statusBadge.parentNode;
    const badgeNext = statusBadge.nextSibling || null;
    const panel = asPanel ? page.closest?.('.project-panel') || null : null;
    // One bar per room: the Project panel bar, or Main's header. The status follows the room's own title in the
    // bar's title row, so a changing word takes the title's room and never moves a neighbour.
    const bar = panel?.querySelector?.('.project-panel-bar') || (asPanel ? null : page.querySelector('.chat-page-header'));
    trigger.append(statusBadge, mark, waitNote, caret);
    const titleRow = bar?.querySelector('.app-page-title-row');
    if (titleRow) titleRow.appendChild(trigger);
    else badgeParent?.insertBefore(trigger, badgeNext);

    // ---- the surface: one floating list in every room, hung under the room's bar (room_work.css)
    const surface = element('div', 'rw-surface rw-shade');
    surface.id = `${uid}-surface`;
    surface.hidden = true;
    surface.tabIndex = -1;  // where a pointer open lands: a landing point, never a control
    surface.dataset.scope = scope;
    surface.setAttribute('role', 'dialog');
    // Names are authored text: the chrome in here is translated at the producer, so the overlay never walks it.
    surface.setAttribute('data-i18n-authored', '');
    const list = element('div', 'rw-list');
    const note = element('p', 'rw-note');
    note.hidden = true;
    // The commands a narrow bar keeps in its list (Main's Restart and budget: page_header.js); empty, it takes no room.
    const extras = element('div', 'rw-extras');
    surface.append(list, note, extras);
    // A zero-height anchor under the bar: the list follows the bar's real height at every width. Main's
    // header floats over the transcript, so its anchor is the header's own last child at its bottom edge.
    const shadeAnchor = element('div', 'rw-shade-anchor');
    shadeAnchor.appendChild(surface);
    if (bar) bar.appendChild(shadeAnchor);
    else if (asPanel) page.querySelector('.chat-panel-statusbar')?.insertAdjacentElement('afterend', shadeAnchor);
    else page.appendChild(shadeAnchor);
    // The element that says "open" for CSS: the Project panel, or Main's header.
    const openHost = panel || (asPanel ? null : bar);

    // The list hangs under the bar's capsules, measured again whenever the open list paints.
    function placeSurface() {
        if (bar) hangUnderCapsules(bar, shadeAnchor);
    }

    // ---- Back to reading: one explicit return to the passage a navigation left
    const back = element('button', 'rw-back');
    back.type = 'button';
    back.hidden = true;
    const backGlyph = decorative(element('span', 'rw-back-glyph'));
    backGlyph.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 14 4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/></svg>';
    const backLabel = element('span', 'rw-back-label', 'Back to reading');
    back.append(backGlyph, backLabel);
    const live = element('span', 'rw-sr');
    live.setAttribute('role', 'status');
    (page.querySelector('.chat-input-wrap') || page).appendChild(back);
    page.appendChild(live);

    // ---- state
    let open = false;
    let destroyed = false;
    let rows = [];
    let frozen = null;          // key -> { row, slot } while open: order and slots hold until close
    let order = [];
    let backAnchor = null;
    let backPlace = null;       // where Back returns when the passage was let go: its page, or the present
    let backTimer = 0;
    let focusTimer = 0;
    let outsideClickTimer = 0;
    let outsideClick = null;
    let scheduled = 0;
    let lastInside = false;
    let pressing = false;       // a pointer press until its click: focus moves meanwhile change no row
    let latestLabel = null;
    let detailSequence = 0;
    let menuIntent = 0;
    let navigationIntent = 0;
    const expanded = new Set();  // keys whose details are open (cleared on close)
    const lookups = new Map();   // id -> what a press found for an unloaded row, and for which census facts (cleared on close)
    const history = historyPager && createHistoryReveal({ pager: historyPager, reading, feed: messagesDiv, records,
        present: () => !destroyed && !page.hidden && page.isConnected });
    const views = new Map();     // key -> painted row
    const detailNames = new Map();
    const detailReads = new Set();
    const disposers = [];
    const listen = (target, type, fn, options) => {
        if (!target?.addEventListener) return;
        target.addEventListener(type, fn, options);
        disposers.push(() => target.removeEventListener?.(type, fn, options));
    };
    if (titleRow) {
        listen(titleRow, 'click', (event) => {
            if (!trigger.disabled && !trigger.hidden && !event.target?.closest?.('.rw-trigger')) trigger.click();
        });
    }

    // ---- model
    // The owner's own message for a direct turn or an unconfirmed send, by its exact client id.
    function bubbleFor(cmid) {
        if (!cmid) return null;
        for (const bubble of messagesDiv.querySelectorAll('.chat-bubble.user')) {
            if (bubble.dataset?.clientMessageId === cmid) return bubble;
        }
        return null;
    }

    // `text` is the bounded visible name; `full` is the whole one, for the accessible name and the details.
    // Its card's coined name first (the newest), then the work's own name (never a Continue's "Continue: …"
    // title), then the card's title.
    function nameOf(item, bubble) {
        const named = (full, source) => ({ text: clip(full, NAME_CHARS), full, source });
        const coinedOnCard = clean(item.record?.suggestedName);
        if (coinedOnCard) return named(coinedOnCard, 'card');
        const work = clean(item.work?.name);
        if (work) return named(work, 'work');
        const own = clean(item.record?.titleEl?.textContent);
        if (own) return named(own, 'card');
        const coined = clean(names.get(item.episode) || names.get(item.id));
        if (coined) return named(coined, 'task_named');
        const read = clean(detailNames.get(item.id));
        if (read) return named(read, 'task_detail');
        const said = clean(bubble?.querySelector?.('.message')?.textContent);
        if (said) return { text: fmt('Reply to “{text}”', { text: clip(said, REPLY_CHARS) }),
            full: fmt('Reply to “{text}”', { text: said }), source: 'owner_message' };
        const when = clock(item.startedAt);
        if (item.kind === 'task') return named(when ? fmt('Task started {time}', { time: when }) : tr('room_work.name.task', 'Task'), 'neutral');
        // A turn with no owner message of its own (a consciousness wake-up) is not a reply.
        if (!item.cmid) return named(when ? fmt('Turn started {time}', { time: when }) : tr('room_work.name.turn', 'Turn in this chat'), 'neutral');
        return named(tr('room_work.name.earlier', 'Reply to an earlier message'), 'neutral');
    }

    // The current work of this room, by its id: what a press re-checks before any effect.
    const workOf = (id) => listOf(works()?.works).find((work) => String(work.work_id) === id) || null;

    function collect() {
        // A closed socket or a host that is not ready proves nothing current: each work stays, unconfirmed (a ring)
        // and without controls, as an incomplete body leaves it.
        const current = connected() && hostReady();
        const state = works() || {};
        const items = [];
        for (const work of listOf(state.works)) {
            const id = String(work.work_id || '');
            if (!id) continue;
            const episode = String(work.episode?.task_id || id);
            const shown = current ? work : { ...work, activity: 'unknown', because: [], tails: [], turn: [], confirmed: false,
                mark: 'ring' };
            const view = workRowView(shown, { stopping: records.get(episode)?.cancelPendingPolicy || '' });
            items.push({ id, kind: work.kind === 'task' ? 'task' : 'direct', work, record: workCard(work, records), view,
                episode, attempt: state.attempts?.get?.(episode) ?? null, activity: current ? String(work.activity || '') : 'unknown',
                online: current, confirmed: current && work.confirmed !== false,
                cmid: String(work.client_message_id || ''),
                startedAt: Number(work.since) || 0 });
        }
        for (const [cmid, sent] of pending) {
            items.push({ id: `pending:${cmid}`, kind: 'pending', work: null, record: null, confirmed: true, cmid,
                startedAt: Math.floor((Number(sent?.timestamp) || 0) / 1000), episode: '', attempt: null, activity: 'pending',
                view: { word: tr('room_work.sending', 'Sending…'), secondary: '', tone: 'live', mark: 'moving', owner: false,
                    outcome: '', focus: '' } });
        }
        // A Project with nothing else keeps its way to the latest represented root card.
        if (asPanel && !items.length) {
            const latest = latestRepresentedRoot([...records.values()]);
            if (latest) items.push({ id: latest.groupId, kind: 'task', work: null, record: latest, latest: true,
                view: cardRowView(desiredLiveCardPhase(latest, replayTerminalPhase(latest)), { settled: latest.finished }),
                confirmed: false, cmid: '', startedAt: 0, episode: latest.groupId, attempt: null, activity: '' });
        }
        return items.map(decorate);
    }

    function decorate(item) {
        const { view, record, work } = item;
        const card = record?.root?.isConnected ? record.root : null;
        const bubble = item.cmid && !card ? bubbleFor(item.cmid) : null;
        const name = nameOf(item, bubble);
        const finished = Boolean(item.latest && record?.finished);
        // The owner's turn first, then moving work, then work that waits; unconfirmed work last.
        const group = item.latest ? 'latest' : view.tone === 'unknown' ? 'unconfirmed'
            : view.owner ? 'needs' : view.tone === 'live' ? 'progress' : 'waiting';
        const sub = { queued: 1, waiting: 2, paused: 3 }[item.activity] || 0;
        // The server's controls for a work it lists, even one it could not confirm (Stop stays offered); a work kept
        // from an incomplete body and every work over a closed socket has none.
        const controls = work && item.online ? workControls(work) : null;
        return {
            key: item.id, id: item.id, kind: item.kind, name: name.text, fullName: name.full, nameSource: name.source,
            view, group, slot: group, rank: GROUP_RANK[group] * 10 + sub, confirmed: item.confirmed,
            latest: Boolean(item.latest), settled: finished, outcomeKnown: finished, episode: item.episode,
            attempt: item.attempt, activity: item.activity, work,
            card: Boolean(card), target: card || bubble, loaded: Boolean(card || bubble),
            startedAt: item.startedAt, cmid: item.cmid, controls,
            canMenu: Boolean(controls && workMenuActions(controls).length), resume: controls?.resume || '',
            question: item.confirmed ? workQuestion(work) : null, tone: view.tone,
        };
    }

    // A press's answer (loading|absent|moved|unknown) holds while the work's facts stay as they were:
    // work that changed may have a message now, so its row claims nothing again until a press asks.
    const lookupFacts = (row) => `${row.activity}|${row.episode}|${row.attempt ?? ''}|${row.cmid || ''}`;
    function lookupOf(row) {
        const found = lookups.get(row.id);
        if (found && found.state !== 'loading' && found.facts !== lookupFacts(row)) lookups.delete(row.id);
        return lookups.get(row.id)?.state || '';
    }

    // Work that left the projection while the list is open: its row stays, settled, until close. Finish withdraws
    // the controls, never the way to its name, card or details; the word is its card's own final one, if known.
    function settled(row) {
        const record = records.get(row.episode) || records.get(row.id);
        const outcome = record?.finished ? desiredLiveCardPhase(record, replayTerminalPhase(record)) : null;
        const card = record?.root?.isConnected ? record.root : null;
        const target = card || (row.cmid ? bubbleFor(row.cmid) : null);
        const view = outcome ? cardRowView(outcome, { settled: true })
            : { word: tr('room_work.ended', 'No longer listed as active'), secondary: '', tone: 'ended', mark: 'ring',
                owner: false, outcome: '', focus: '' };
        const name = nameOf({ ...row, record }, target);
        const refreshedName = name.source === 'neutral' ? {} : {
            name: name.text, fullName: name.full, nameSource: name.source,
        };
        return { ...row, ...refreshedName, view, group: 'ended', settled: true, outcomeKnown: Boolean(outcome), canMenu: false,
            controls: null, resume: '', question: null, card: Boolean(card), target, loaded: Boolean(target),
            tone: view.tone };
    }

    const sorter = (a, b) => (a.rank - b.rank) || (b.startedAt - a.startedAt) || a.key.localeCompare(b.key);

    // While open, order and each row's slot hold; words, verbs and motion update in place, so no existing
    // node moves under the keyboard or the pointer. New rows append.
    function arrange(collected) {
        if (!open || !frozen) return [...collected].sort(sorter);
        const byKey = new Map(collected.map((row) => [row.key, row]));
        const out = [];
        for (const key of [...order]) {
            const kept = frozen.get(key);
            let fresh = byKey.get(key);
            if (!fresh && kept.row.kind === 'pending' && kept.row.cmid) {
                // The owner's send that the host turned into a turn takes the same place.
                // A progress card may already own the turn's slot before its census arrives.
                // In that case keep that view; the old send settles normally until close.
                fresh = [...byKey.values()].find((row) => row.kind !== 'pending'
                    && row.cmid === kept.row.cmid && !order.includes(row.key));
                if (fresh) {
                    order[order.indexOf(key)] = fresh.key;
                    frozen.delete(key);
                    // Identity changes, not the reader's node: preserve selection and focus in this slot.
                    const view = views.get(key);
                    if (view) { views.delete(key); views.set(fresh.key, view); }
                }
            }
            if (fresh) {
                byKey.delete(fresh.key);
                out.push({ ...fresh, slot: kept.slot });
            } else {
                out.push(settled(kept.row));
            }
        }
        for (const row of [...byKey.values()].sort(sorter)) {
            order.push(row.key);
            out.push(row);
        }
        for (const row of out) if (!row.settled || !frozen.has(row.key)) frozen.set(row.key, { row, slot: row.slot });
        return out;
    }

    // What a press on an unloaded row found out about where its message is (DESIGN "Room work status").
    function whereText(row) {
        const ended = row.settled || Boolean(row.view.outcome);
        return { loading: tr('room_work.lookup.loading', 'Loading its message…'),
            absent: ended ? tr('room_work.lookup.absent_ended', 'No message saved in this chat.') : tr('room_work.lookup.absent', 'No message yet.'),
            moved: tr('room_work.lookup.moved', 'This work is in another room.'),
            unknown: tr('room_work.lookup.unknown', 'Its message could not be checked.') }[row.lookup] || '';
    }

    function describe(row) {
        const lines = [];
        if (row.settled && !row.latest) lines.push(row.outcomeKnown
            ? [tr('room_work.detail.finished', 'Finished'),
                tr('room_work.detail.finished_text', 'Its recorded outcome is shown above. This row stays until the list closes.')]
            : [tr('room_work.detail.activity', 'Activity'),
                tr('room_work.detail.ended_text', 'No longer listed as active; its outcome is not known here. This row stays until the list closes.')]);
        const where = whereText(row);
        if (where) lines.push([tr('room_work.detail.where', 'Where'), where]);
        const uncontrolled = !row.confirmed && !row.canMenu && !row.resume && !row.settled && !row.latest;
        if (uncontrolled) lines.push([tr('room_work.detail.why', 'Why no controls'),
            tr('room_work.detail.why_text', 'The latest update did not confirm this work, so nothing here can be paused or stopped.')]);
        const what = row.kind === 'pending' ? tr('room_work.what.pending', 'Your message, being sent')
            : row.kind === 'direct' ? (row.cmid ? tr('room_work.what.reply', 'A reply to your message')
                : tr('room_work.what.turn', 'A turn in this chat'))
                : tr('room_work.what.task', 'A task');
        lines.push([tr('room_work.detail.what', 'What'), what]);
        return lines;
    }

    // ---- rows
    function viewFor(row) {
        let view = views.get(row.key);
        if (view) return view;
        const el = element('div', 'rw-row');
        el.dataset.id = row.id;
        // main: the row's one primary target, the way to its card or (no card loaded) its details
        const main = element('div', 'rw-main');
        const nameEl = element('span', 'rw-name');
        const state = element('span', 'rw-state');
        const mark = createStatusMark(doc);
        const word = element('span', 'rw-word');
        const secondary = element('span', 'rw-secondary');
        const where = element('span', 'rw-where');
        const chev = decorative(element('span', 'rw-chev'));
        where.append(chev);
        state.append(mark, word, secondary, where);
        // The episode's own focus line, quieter, under the state (work_status.workRowView's `focus`).
        const focus = element('span', 'rw-focus');
        focus.hidden = true;
        main.append(nameEl, state, focus);
        const verb = element('span', 'rw-verb');
        const moreSlot = element('span', 'rw-more-slot');
        const detail = element('div', 'rw-detail');
        detail.id = `${uid}-detail-${(detailSequence += 1)}`;
        detail.hidden = true;
        el.append(main, verb, moreSlot, detail);
        view = { el, main, nameEl, mark, word, secondary, focus, where, chev, verb, moreSlot,
            detail, question: null, resume: null, resumeTarget: '', more: null, row: null, full: null, detailKey: '',
            episode: undefined };
        // The name stays selectable text: a drag that selected it does not navigate.
        bindContentButton(main, () => activateRow(view.row), () => rowIsTarget(view.row));
        // Keyboard focus shows the whole name (CSS unclamps it), and only the row keyboard reads shows it. Not
        // :focus-within, and a press changes no row: nothing in the list moves under a pointer. A row folding on
        // the focus a press gives it, or on the blur a press elsewhere causes, would move the pressed row (or
        // the ⋯ a menu opens from) before mouse-up; a grown row folds when keyboard focus reads another row or
        // the list closes.
        main.addEventListener('focus', () => {
            if (pressing) return;
            let keyboard = false;
            try { keyboard = main.matches?.(':focus-visible') === true; } catch { /* an engine without the selector */ }
            if (keyboard) for (const other of views.values()) if (other !== view) other.el.dataset.reading = '0';
            el.dataset.reading = keyboard ? '1' : '0';
        });
        main.addEventListener('blur', () => { if (!pressing) el.dataset.reading = '0'; });
        views.set(row.key, view);
        return view;
    }

    const rowIsTarget = (row) => Boolean(row) && (Boolean(row.target?.isConnected) || row.kind !== 'pending');

    function activateRow(row) {
        if (!row) return;
        if (row.target?.isConnected) navigate(row);
        else if (row.kind === 'pending') return;
        else if (history && !['absent', 'moved', 'loading'].includes(row.lookup)) void reveal(row);
        else toggleDetail(row.key);
    }

    function toggleDetail(key) {
        if (expanded.has(key)) expanded.delete(key); else expanded.add(key);
        const row = rows.find((candidate) => candidate.key === key);
        if (row) paintRow(row);
    }

    function paintRow(row) {
        const view = viewFor(row);
        view.row = row;
        const { el, main } = view;
        const navigable = Boolean(row.target?.isConnected);
        // An unloaded row asks the room's history on its press: until then it claims nothing about where its message is.
        const lookable = !navigable && row.kind !== 'pending' && Boolean(history) && !['absent', 'moved'].includes(row.lookup);
        // Opened details survive a census change; a press then asks again where its message is.
        const detailable = !navigable && row.kind !== 'pending' && (!history || Boolean(row.lookup) || expanded.has(row.key));
        const isOpen = detailable && expanded.has(row.key);
        Object.assign(el.dataset, {
            id: row.id,
            slot: row.slot, kind: row.kind, tone: row.tone, settled: row.settled ? '1' : '0', open: isOpen ? '1' : '0',
        });
        const full = row.fullName || row.name;
        // Every row says its state under its name (5A); where an unloaded row's message is, its press finds out.
        const state = [row.view.word, row.view.secondary].filter(Boolean).join(' · ');
        if (navigable || lookable || detailable) {
            main.setAttribute('role', 'button');
            main.tabIndex = 0;
            main.setAttribute('aria-label', navigable
                ? fmt(row.card ? 'Go to task: {name}, {state}' : 'Go to message: {name}, {state}', { name: full, state })
                : lookable ? fmt('Open: {name}, {state}', { name: full, state })
                    : fmt('Details: {name}, {state}', { name: full, state }));
            if (row.lookup === 'loading') main.setAttribute('aria-busy', 'true');
            else main.removeAttribute('aria-busy');
            if (detailable) {
                main.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
                main.setAttribute('aria-controls', view.detail.id);
            } else {
                main.removeAttribute('aria-expanded');
                main.removeAttribute('aria-controls');
            }
        } else {
            for (const name of ['role', 'tabindex', 'aria-label', 'aria-expanded', 'aria-controls', 'aria-busy']) main.removeAttribute(name);
        }
        main.dataset.target = navigable ? 'card' : lookable ? 'history' : detailable ? 'details' : 'none';
        // The whole name is in the DOM; CSS clamps it to two lines, focus and open details show it whole.
        if (view.full !== full) { view.full = full; view.nameEl.textContent = full; }
        // The work's shape in its word's ink; something in it that waits for the owner is the amber dot in its place
        // (a pause is not the owner's move), and a known outcome is a dot in the outcome's tone beside its word.
        paintStatusMark(view.mark, markView({ mark: row.view.mark, owner: row.view.owner, outcome: row.view.outcome }));
        if (view.word.textContent !== row.view.word) view.word.textContent = row.view.word;
        const secondary = String(row.view.secondary || '');
        if (view.secondary.textContent !== secondary) view.secondary.textContent = secondary;
        view.secondary.hidden = !secondary;
        const focus = row.settled ? '' : String(row.view.focus || '');
        if (view.focus.textContent !== focus) view.focus.textContent = focus;
        view.focus.hidden = !focus;
        view.chev.hidden = !detailable;
        view.where.hidden = !detailable;
        view.detail.hidden = !isOpen;
        if (isOpen) {
            const lines = describe(row);
            const key = lines.map((line) => line.join('\u0000')).join('\u0001');
            if (view.detailKey !== key) {
                view.detailKey = key;
                view.detail.textContent = '';
                for (const [label, text] of lines) {
                    const line = element('p', 'rw-detail-line');
                    line.append(element('span', 'rw-detail-label', label), element('span', '', text));
                    view.detail.appendChild(line);
                }
            }
        }
        // A new episode (or a new attempt of it) never inherits a menu opened for the previous one.
        const episode = `${row.episode}|${row.attempt ?? ''}`;
        if (view.episode !== undefined && view.episode !== episode
            && view.more?.getAttribute('aria-expanded') === 'true') closeTaskControlMenu();
        view.episode = episode;
        reconcileActions(view, row);
    }

    function reconcileActions(view, row) {
        const keepFocus = (node, fallback) => {
            if (node && doc.activeElement === node) (fallback || view.more || view.main).focus?.({ preventScroll: true });
        };
        const full = row.fullName || row.name;
        const wantQuestion = Boolean(row.question) && !row.settled;
        if (wantQuestion && !view.question) {
            const button = element('button', 'rw-action rw-pill rw-pill-question', tr('room_work.view_question', 'View question'));
            button.type = 'button';
            button.addEventListener('click', (event) => {
                event?.stopPropagation?.();
                if (view.row?.question) void viewQuestion(view.row);
            });
            view.question = button;
            view.verb.prepend(button);
        } else if (!wantQuestion && view.question) {
            keepFocus(view.question);
            view.question.remove();
            view.question = null;
        }
        // It goes to the question; it never sends, selects or fills in an answer.
        view.question?.setAttribute('aria-label', fmt('View question: {name}', { name: full }));
        if (view.question) {
            const label = tr('room_work.view_question', 'View question');
            if (view.question.textContent !== label) view.question.textContent = label;
        }
        // Resume stands only where the work's own Resume control names its exact target.
        const wantResume = row.settled ? '' : row.resume;
        if (view.resume && view.resumeTarget !== wantResume) {
            keepFocus(view.resume);
            view.resume.remove();
            view.resume = null;
        }
        if (wantResume && !view.resume) {
            // The shared button re-checks the work's control at press time; its target never changes.
            view.resume = createTaskResumeButton(wantResume, {
                canResume: () => currentControls(view.row)?.resume === wantResume, onSettled: refresh,
            });
            view.resume.classList.add('rw-action', 'rw-pill', 'rw-pill-resume');
            view.resumeTarget = wantResume;
            view.verb.append(view.resume);
        }
        if (view.resume) {
            const label = tr('task.control.resume', 'Resume');
            if (view.resume.textContent !== label) view.resume.textContent = label;
        }
        // The ⋯ is the one door to every other action: the work's own controls, each aimed at its exact target.
        const wantMore = row.canMenu && !row.settled;
        const actionKey = wantMore ? workMenuActions(row.controls, {
            cancelPending: Boolean(records.get(row.episode)?.cancelPendingPolicy),
        }).join(',') : '';
        // A row may keep its menu door while the actions behind it change (Working → Paused).
        // Withdraw the old portal immediately, not just its ability to dispatch stale actions.
        if (view.actionKey !== actionKey && view.more?.getAttribute('aria-expanded') === 'true') {
            menuIntent += 1;
            closeTaskControlMenu();
        }
        view.actionKey = actionKey;
        if (wantMore && !view.more) {
            const button = element('button', 'rw-more');
            button.type = 'button';
            button.innerHTML = '<svg class="rw-more-glyph" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false"><circle cx="3" cy="8" r="1.4"/><circle cx="8" cy="8" r="1.4"/><circle cx="13" cy="8" r="1.4"/></svg>';
            button.setAttribute('aria-haspopup', 'menu');
            button.setAttribute('aria-expanded', 'false');
            button.addEventListener('click', (event) => {
                event?.stopPropagation?.();
                if (view.row?.canMenu) void openControls(view.row, button);
            });
            view.more = button;
            view.moreSlot.append(button);
        } else if (!wantMore && view.more) {
            if (view.more.getAttribute('aria-expanded') === 'true') closeTaskControlMenu();
            keepFocus(view.more, view.main.tabIndex >= 0 ? view.main : null);
            view.more.remove();
            view.more = null;
        }
        if (view.more) view.more.dataset.id = row.episode;  // Pause reads its exact target here, never the nearest card
        view.more?.setAttribute('aria-label', fmt('Task actions: {name}', { name: full }));
        view.el.dataset.verb = view.question ? 'question' : view.resume ? 'resume' : '';
    }

    // The work's current controls, only while the list is shown over a live connection and the projection still
    // lists the same episode (and attempt): a press re-checks them before any effect.
    function currentControls(row) {
        const work = row ? workOf(row.id) : null;
        const current = open && !destroyed && !page.hidden && page.isConnected && connected() && hostReady()
            && work && String(work.episode?.task_id || work.work_id) === row.episode
            && (works()?.attempts?.get?.(row.episode) ?? null) === row.attempt;
        return current ? workControls(work) : null;
    }

    // A loaded card stops through its own card path, so its chip follows at once; a row without a card first
    // reads the durable record, as Activity does, so a stop already pending is not offered again as a fresh Wrap up.
    async function openControls(row, anchor) {
        const intent = ++menuIntent;
        if (!currentControls(row)) return;
        const loaded = (id) => { const record = records.get(id); return record && !record.finished && cancelRunFromCard ? record : null; };
        let cancelPending = Boolean(records.get(row.episode)?.cancelPendingPolicy);
        if (!loaded(row.episode)) {
            let stored = null;
            try { stored = await fetchTaskDetailStrict(row.episode); } catch (exc) {
                if (exc?.status !== 404) {
                    showToast(`Could not refresh task state: ${exc?.message || exc}`, 'error');
                    return;
                }
            }
            if (intent !== menuIntent || !anchor.isConnected || !currentControls(row)) return;
            cancelPending = taskCancelPending(stored);
        }
        const canAct = (action) => {
            const current = currentControls(row);
            if (intent !== menuIntent || !anchor.isConnected || !current || taskControlBusy(row.episode)) return false;
            return workMenuActions(current, {
                cancelPending: cancelPending || Boolean(records.get(row.episode)?.cancelPendingPolicy),
            }).includes(action);
        };
        const stopOne = (target, action) => (loaded(target) ? cancelRunFromCard(loaded(target), action) : requestStop(target, action));
        openWorkControls(anchor, workOf(row.id), { cancelPending, canAct, stopOne, afterAction: () => refresh() });
    }

    // ---- paint
    function render() {
        if (destroyed) return;
        // Main mounts before the install's language arrives; the authored subtree has no overlay.
        const label = asPanel ? tr('room_work.scope.project', 'Work in this project')
            : tr('room_work.scope.main', 'Work in this chat');
        if (surface.getAttribute('aria-label') !== label) surface.setAttribute('aria-label', label);
        rows = arrange(collect()).map((row) => Object.assign(row, { lookup: lookupOf(row) }));
        paintTrigger();
        paintNote();
        if (!open) return;
        placeSurface();
        const keys = new Set(rows.map((row) => row.key));
        for (const [key, view] of views) {
            if (!keys.has(key)) { view.el.remove(); views.delete(key); expanded.delete(key); }
        }
        const children = [];
        rows.forEach((row, index) => {
            prefetchName(row);
            paintRow(row);
            const { el } = views.get(row.key);
            el.style.setProperty('--rw-i', String(Math.min(index, 6)));
            // Only the Project's latest root is labelled, so it never reads as current work.
            if (row.slot === 'latest' && !children.includes(latestLabel)) {
                latestLabel ||= element('div', 'rw-group');
                const label = tr('room_work.latest', 'Latest task');
                if (latestLabel.textContent !== label) latestLabel.textContent = label;
                children.push(latestLabel);
            }
            children.push(el);
        });
        if (latestLabel && !children.includes(latestLabel)) latestLabel.remove();
        // Nodes move only where the sequence differs, so a focused row is never detached.
        children.forEach((node, index) => {
            if (list.children[index] !== node) list.insertBefore(node, list.children[index] || null);
        });
    }

    // The word goes whole or not at all (DESIGN "Shell"): shown, measured, dropped when the row would cut it, in one
    // task so nothing paints between; a preparing turn's start step beside it (the badge's data-step) goes first. The
    // mark stays and the control's name keeps the whole status. Measured again when the word, step, mark or available
    // width changes. Remember the capsule's width AFTER fitting, so observing its content-sized box does not feed a
    // hide/show cycle back into the ResizeObserver.
    // A Project's name then takes a second line (room_work.css), and only a name still cut makes the consciousness chip
    // show its brain alone (its name keeps the whole state); Panic never yields.
    let fittedWord = null, fittedWidth = null, fitNeeded = false;
    const name = panel && bar?.querySelector('.app-page-title');
    function fitWord() {
        if (destroyed) return;
        delete trigger.dataset.word;
        delete trigger.dataset.step;
        const chip = name && bar.querySelector('.cx-chip');
        if (chip) delete chip.dataset.fit;
        const cut = () => statusBadge.scrollWidth > statusBadge.clientWidth + 1;
        if (statusBadge.dataset.step && cut()) trigger.dataset.step = 'off';
        if (cut()) trigger.dataset.word = 'off';
        if (chip && name.scrollHeight > name.clientHeight + 1) chip.dataset.fit = 'mark';
        fittedWidth = titleRow?.clientWidth;
    }

    function paintTrigger() {
        const text = [clean(statusBadge.textContent), clean(statusBadge.dataset.step)].filter(Boolean).join(' · ');
        const listed = rows.length > 0 || extras.children.length > 0;
        // Word, tone, mark and what waits for the owner come together from the chat's producer
        // (work_status.roomStatusView via setStatus); the control paints them and never ranks row tones or reads a
        // class name for a second opinion. Before the producer's first answer the room is not online yet: an unknown
        // tone paints as offline.
        const tone = statusBadge.dataset.tone || 'offline';
        trigger.disabled = !listed;
        trigger.dataset.tone = tone;
        trigger.dataset.open = open ? '1' : '0';
        trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
        trigger.title = text;  // the whole status, never clipped away (an extra, not the only path)
        caret.hidden = !listed;
        // The work's shape in the word's ink; while something in the room waits for the owner (a question, or an action
        // only the owner takes) the amber dot takes its place and the word keeps its own ink: Working with a question
        // elsewhere stays Working. A pause by the owner or by Ouroboros is not the owner's move. The room at rest says
        // Online beside its green dot.
        const owner = statusBadge.dataset.owner === '1';
        const reshaped = paintStatusMark(mark, tone === 'online' && !owner ? AT_REST : markView({ mark: statusBadge.dataset.mark, owner }));
        // The owner's move said in words too, where the word does not already say it.
        const said = amberNote(statusBadge.dataset.amber || '');
        if (waitNote.textContent !== said) waitNote.textContent = said;
        waitNote.hidden = !said;
        // A new word, or a mark of another width beside the same word (the owner's dot for the dots), is measured again.
        if (text !== fittedWord || reshaped || fitNeeded) { fittedWord = text; fitNeeded = false; fitWord(); }
        if (!listed && open) close({ restoreFocus: false });
    }

    function paintNote() {
        const latest = stateSnapshots?.latest?.();
        const message = !connected() ? tr('room_work.note.reconnecting', 'Reconnecting — showing the last known state.')
            : !hostReady() ? tr('room_work.note.starting', 'Ouroboros is starting…')
                : Array.isArray(latest?.works) && latest.works_complete === false
                    ? tr('room_work.note.partial', 'Not all current work could be confirmed; this list may be incomplete.') : '';
        if (note.textContent !== message) note.textContent = message;
        note.hidden = !message || !open;
    }

    // ---- behaviour
    // The list folds away the way it unfolded: closed at once (its state, its focus), it stays drawn, inert, until its
    // motion ends; opened again meanwhile, it stops folding where it is.
    let folding = null;
    function foldAway() {
        const ms = motionMs('--motion-fade');
        if (!ms || reducedMotion() || typeof surface.animate !== 'function') { surface.hidden = true; return; }
        surface.inert = true;
        const fold = surface.animate(SHADE_FOLDS, { duration: ms, easing: 'ease-in', fill: 'forwards' });
        folding = fold;
        fold.finished.then(() => { if (folding === fold && !open) surface.hidden = true; }, () => {}).finally(() => {
            if (folding !== fold) return;
            folding = null;
            surface.inert = false;
            fold.cancel();
        });
    }

    function openSurface(viaKeyboard = false) {
        if (open || destroyed || trigger.disabled || trigger.hidden) return;
        open = true;
        frozen = new Map();
        order = [];
        if (folding) { const fold = folding; folding = null; surface.inert = false; fold.cancel(); }
        surface.hidden = false;
        openHost?.setAttribute('data-rw-open', '');
        render();
        // A pointer open lands on the (ring-less) list; a keyboard open enters its first real control.
        const first = viaKeyboard ? list.querySelector('.rw-main[tabindex="0"], .rw-action, .rw-more') : null;
        (first || surface).focus?.({ preventScroll: true });
    }

    function close({ restoreFocus = true } = {}) {
        menuIntent += 1;
        if (!open) return;
        open = false;
        closeTaskControlMenu();
        foldAway();
        openHost?.removeAttribute('data-rw-open');
        frozen = null;
        order = [];
        expanded.clear();
        for (const view of views.values()) view.el.dataset.reading = '0';  // it reopens at rest
        // A list dismissed while a row still looks drops that look: the reader left before it answered.
        if ([...lookups.values()].some((found) => found.state === 'loading')) { navigationIntent += 1; live.textContent = ''; }
        lookups.clear();
        // Reconcile against the closed-list policy before restoring focus. The final retained
        // Main row may just have ended; disabling its trigger on the next frame would strand focus.
        render();
        if (restoreFocus && trigger.isConnected && !trigger.hidden) {
            if (!trigger.disabled) trigger.focus?.({ preventScroll: true });
            else (page.querySelector('#chat-input') || page.querySelector('.chat-input'))?.focus?.({ preventScroll: true });
        }
    }

    function prefetchName(row) {
        if (row.nameSource !== 'neutral' || row.kind !== 'task' || detailReads.has(row.id)) return;
        detailReads.add(row.id);
        fetchTaskDetail(row.id)
            .then((detail) => { if (!destroyed) detailNames.set(row.id, clean(detail?.suggested_name || detail?.title)); })
            .catch(() => {})
            .finally(() => schedule());
    }

    function scrollToNode(node) {
        reading.cancel();
        // Main's measured reserve includes its error note; Project's in-flow bars already clear the feed.
        const reserve = asPanel
            ? Math.max(0, (bar?.getBoundingClientRect().bottom || 0) - messagesDiv.getBoundingClientRect().top)
            : parseFloat(page.style?.getPropertyValue?.('--chat-header-reserve')) || 0;
        messagesDiv.scrollTop += node.getBoundingClientRect().top - messagesDiv.getBoundingClientRect().top - reserve;
        reading.stick = false;
        reading.scroll();
    }

    // Going to a card closes the list: the reader came to read it. Back to reading undoes the trip.
    function navigate(row) {
        const target = row?.target;
        if (!target?.isConnected) return;
        navigationIntent += 1;
        arrive(row, target, departure());
    }

    const departure = () => (history ? history.departure(captureVisibleTimelineAnchor())
        : { anchor: captureVisibleTimelineAnchor() });

    function arrive(row, target, place) {
        close({ restoreFocus: false });
        scrollToNode(target);
        showBack(place.anchor, place);
        // The card's one control of its own, "Details", takes the keyboard; a portal (which hides it) takes it itself.
        const disclosure = records.get(row.id)?.disclosureEl;
        const focus = disclosure?.isConnected && disclosure.getClientRects?.().length ? disclosure : target;
        if (focus === target && !target.hasAttribute?.('tabindex')) target.setAttribute('tabindex', '-1');
        focus.focus?.({ preventScroll: true });
    }

    // An unloaded row: the chat reads the page around its task and lands there (DESIGN "Room work status"); the
    // list stays open while it looks, and a history with no message of it here, or a read that cannot tell, says so.
    async function reveal(row) {
        const intent = ++navigationIntent;
        const place = departure();
        lookups.set(row.id, { state: 'loading', intent });
        expanded.add(row.key);
        live.textContent = tr('room_work.lookup.loading', 'Loading its message…');
        schedule();
        const found = await history.task(row.id, () => intent === navigationIntent);
        if (!found) {
            if (intent === navigationIntent) live.textContent = '';
            if (lookups.get(row.id)?.intent === intent) { lookups.delete(row.id); expanded.delete(row.key); schedule(); }
            return;
        }
        lookups.delete(row.id);
        if (found.target) { arrive(row, found.target, place); return; }
        if (found.left) showBack(place.anchor, place);  // its page was drawn yet shows nothing of it: Back undoes that
        // The owner's own message is a message: a row that carries one is never told it has none.
        const state = found.state === 'absent' && row.cmid ? 'unknown' : found.state;
        lookups.set(row.id, { state, facts: lookupFacts(row) });
        live.textContent = whereText({ ...row, lookup: state });
        schedule();
    }

    // Goes to the exact waited question and focuses the question itself, never its answer field.
    async function viewQuestion(row) {
        const intent = ++navigationIntent;
        const anchor = captureVisibleTimelineAnchor();
        close({ restoreFocus: false });
        // The task that asked (a Continue or a helper), never the work's first root.
        const shown = await revealQuestion?.(row.question.taskId || row.id, row.question.quizId, { focus: 'question' });
        if (shown && !destroyed && intent === navigationIntent && !page.hidden && page.isConnected) showBack(anchor);
    }

    function showBack(anchor, place = null) {
        backAnchor = anchor || null;
        backPlace = backAnchor ? place : null;
        clearTimeout(backTimer);
        backLabel.textContent = 'Back to reading';
        back.hidden = !backAnchor;
    }

    function hideBack() {
        // Hand focus to the passage, without moving it. A later dismissal must not steal focus
        // from a reader who already moved elsewhere while the missing-passage notice was shown.
        if (doc.activeElement === back && !destroyed && !page.hidden && page.isConnected) {
            if (!messagesDiv.hasAttribute('tabindex')) messagesDiv.setAttribute('tabindex', '-1');
            messagesDiv.focus?.({ preventScroll: true });
        }
        back.hidden = true;
    }

    async function goBack() {
        if (!backAnchor) return;
        const anchor = backAnchor, place = backPlace;
        const intent = ++navigationIntent;
        clearTimeout(backTimer);
        // The reader was following the present: Back is the return to it.
        if (place?.present && showLatest) { backAnchor = null; hideBack(); await showLatest(); return; }
        reading.cancel();
        const restore = () => (restoreVisibleTimelineAnchor(anchor, { exact: true }) ? 'exact'
            : restoreVisibleTimelineAnchor(anchor, { cardOnly: true }) ? 'near' : '');
        let returned = restore();
        if (!returned && place?.cursor && history) {  // the page the reader left was let go: read it again
            const drawn = await history.revisit(place.cursor, () => intent === navigationIntent);
            if (drawn === null) return;
            if (drawn === false) {  // the read failed for now: Back stays for another try
                live.textContent = 'The earlier reading place could not be read. Try again.';
                return;
            }
            if (drawn === true) returned = restore();
        }
        reading.stick = false;
        reading.scroll();
        backAnchor = null;
        if (!returned) {
            backLabel.textContent = 'Earlier place is no longer loaded';
            live.textContent = 'The earlier reading place is no longer loaded.';
            backTimer = setTimeout(hideBack, 4000);
            return;
        }
        live.textContent = returned === 'exact' ? 'Returned to where you were reading.' : 'Returned to the nearest loaded place.';
        hideBack();
    }

    // ---- wiring
    listen(globalThis.window, 'ouro:language-changed', schedule);
    // A click with detail 0 is Enter or Space on the focused control: a keyboard open lands in the list.
    listen(trigger, 'click', (event) => { if (open) close(); else openSurface(event?.detail === 0); });
    listen(trigger, 'keydown', (event) => {
        if ((event.key === 'ArrowDown' || event.key === 'ArrowUp') && !open && !trigger.disabled) {
            event.preventDefault();
            openSurface(true);
        }
    });
    listen(back, 'click', goBack);
    // A command chosen in the Shade is the end of this look (Restart asks its own question, the budget opens Costs).
    listen(extras, 'click', () => close({ restoreFocus: false }));
    listen(doc, 'pointerdown', (event) => {
        // A press lasts until the click its release makes (a touch's mouse events follow its pointerup), so a
        // focus move inside it, its own or a closing menu handing focus back, folds nothing under the pointer.
        pressing = true;
        const target = event.target;
        // The shared task-controls menu is a portal outside the list but part of its interaction.
        if (target?.closest?.('.task-control-menu')) return;
        lastInside = surface.contains(target) || trigger.contains(target) || Boolean(titleRow?.contains(target));
        if (!open || lastInside) return;
        close({ restoreFocus: false });
        // The first outside press only dismisses the list: a press on the dimmed Main must not also close the room.
        if (target?.closest?.('#project-panel-backdrop')) {
            if (outsideClick) doc.removeEventListener('click', outsideClick, true);
            clearTimeout(outsideClickTimer);
            outsideClick = (click) => { click.stopPropagation(); click.preventDefault(); };
            doc.addEventListener('click', outsideClick, { capture: true, once: true });
            outsideClickTimer = setTimeout(() => {
                doc.removeEventListener('click', outsideClick, true);
                outsideClick = null;
                outsideClickTimer = 0;
            }, 600);
        }
    }, true);
    // A press that makes no click (a cancelled touch, a release outside the window) ends at the next key or
    // press, so keyboard reading never waits on it.
    const endPress = () => { pressing = false; };
    listen(doc, 'click', endPress, true);
    listen(doc, 'pointercancel', endPress, true);
    listen(doc, 'keydown', endPress, true);
    listen(doc, 'focusin', (event) => {
        if (!event.target?.closest?.('.task-control-menu')) lastInside = surface.contains(event.target) || trigger.contains(event.target);
    }, true);
    // Some engines (WebKit) do not focus a pressed button, so after a press in the list the document holds
    // focus: the list still owns Escape when the reader's last press or focus was in it.
    listen(doc, 'keydown', (event) => {
        if (event.key !== 'Escape' || !open || event.defaultPrevented) return;
        const active = doc.activeElement;
        const nowhere = !active || active === doc.body;
        if (!surface.contains(active) && active !== trigger && !(nowhere && lastInside)) return;
        event.preventDefault();
        close();
    });
    listen(surface, 'focusout', (event) => {
        if (!open) return;
        const next = event.relatedTarget;
        // Tabbing out of the list (not into its own task-controls menu) dismisses it, like any dropdown.
        if (next && !surface.contains(next) && !trigger.contains(next) && !next.closest?.('.task-control-menu')) {
            close({ restoreFocus: false });
            return;
        }
        // A focused verb can vanish under the keyboard (Resume disables itself while its request runs and
        // Chromium drops focus to the document): keep the reader in the same row, else on the control.
        const lost = event.target;
        const rowEl = lost?.closest?.('.rw-row');
        if (next || !rowEl) return;
        clearTimeout(focusTimer);
        focusTimer = setTimeout(() => {
            focusTimer = 0;
            if (!open || destroyed) return;
            const active = doc.activeElement;
            if ((active && active !== doc.body) || (lost.isConnected && !lost.disabled)) return;
            const view = [...views.values()].find((candidate) => candidate.el === rowEl);
            const stop = [view?.more, view?.main].find((node) => node?.isConnected && !node.disabled && node.tabIndex >= 0);
            (stop || trigger).focus?.({ preventScroll: true });
        }, 0);
    });
    listen(surface, 'keydown', (event) => {
        if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
        const stops = [...surface.querySelectorAll('.rw-main[tabindex="0"]')];
        if (!stops.length) return;
        event.preventDefault();
        const index = stops.indexOf(event.target?.closest?.('.rw-main'));
        const step = event.key === 'ArrowDown' ? 1 : -1;
        const next = stops[index < 0 ? (step > 0 ? 0 : stops.length - 1) : Math.max(0, Math.min(stops.length - 1, index + step))];
        next?.focus?.({ preventScroll: true });
        next?.scrollIntoView?.({ block: 'nearest' });
    });
    // The Project panel's own close dismisses a list that lives outside the room's page.
    listen(doc, 'click', (event) => {
        if (open && event.target?.closest?.('#project-panel-close')) close({ restoreFocus: false });
    }, true);

    // A room hidden without being destroyed (a Project kept for its pending files) takes its control along.
    const Observer = globalThis.MutationObserver;
    const watcher = typeof Observer === 'function' ? new Observer(() => {
        const away = Boolean(page.hidden) || !page.isConnected;
        if (away) { navigationIntent += 1; close({ restoreFocus: false }); showBack(null); }
        trigger.hidden = away;
        if (!away) fitWord();
    }) : null;
    watcher?.observe(page, { attributes: true, attributeFilter: ['hidden'] });
    const Resize = globalThis.ResizeObserver;
    const sizer = bar && typeof Resize === 'function' ? new Resize((entries) => {
        if (entries.some(entry => entry.target === bar) || titleRow?.clientWidth !== fittedWidth) {
            // Use the existing paint frame: resizing an observed capsule inside delivery
            // would produce an undelivered-notifications loop warning in WebKit.
            fitNeeded = true; schedule();
        } else if (open) schedule();  // another capsule of the bar moved: the open list follows its edges
    }) : null;
    sizer?.observe(bar);
    if (titleRow) sizer?.observe(titleRow);
    for (const node of bar?.children || []) if (node !== shadeAnchor && node !== titleRow) sizer?.observe(node);

    function schedule() {
        if (destroyed || scheduled) return;
        scheduled = -1;
        const handle = frame(() => { scheduled = 0; render(); });
        if (scheduled) scheduled = handle || -1;  // a synchronous frame (the node fixture) has already rendered
    }

    render();
    return {
        update: schedule,
        extras,
        close: () => close({ restoreFocus: false }),
        // The room leaves the screen: the list closes and the saved passage no longer applies.
        leave() {
            navigationIntent += 1;
            // Main stays mounted behind a Project. Cancel the shared reading claim too, so an
            // outstanding canonical question read cannot scroll or focus that obscured room.
            reading.cancel();
            close({ restoreFocus: false });
            showBack(null);
        },
        isOpen: () => open,
        destroy() {
            if (destroyed) return;
            if (open) closeTaskControlMenu();
            destroyed = true;
            menuIntent += 1;
            navigationIntent += 1;
            if (scheduled > 0) globalThis.cancelAnimationFrame?.(scheduled);
            clearTimeout(backTimer);
            clearTimeout(focusTimer);
            clearTimeout(outsideClickTimer);
            if (outsideClick) doc.removeEventListener('click', outsideClick, true);
            watcher?.disconnect();
            sizer?.disconnect();
            for (const dispose of disposers.splice(0)) dispose();
            openHost?.removeAttribute('data-rw-open');
            // The badge returns where the product put it; this module's nodes go.
            if (badgeParent?.isConnected) badgeParent.insertBefore(statusBadge, badgeNext?.parentNode === badgeParent ? badgeNext : null);
            trigger.remove();
            (shadeAnchor || surface).remove();
            back.remove();
            live.remove();
            views.clear();
        },
    };
}
