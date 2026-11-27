// After Panic, saved work comes back only by the owner's choice (docs/DESIGN.md "Project question mirror"): one System
// message in Main names every work the host's `panic_recovery` census holds for the owner's Resume, from every room,
// each with a check box and the `Stop…` its owner-action card offered. One press sends exactly the chosen rows, each to
// the task's existing Resume door with the stop it was chosen under (`expected_panic_hold_id`): never a wildcard, an
// automatic or repeated send, or an acceptance of unknown work. Each row then says what its door answered for that
// stop: accepted (in the door's own words, never running), refused (in the host's), changed since it was chosen, or no
// answer; an answer about a stop that has changed since reads as the earlier stop's. `Choose all` and the claim that
// nothing waits stand only on today's census of saved work saying its list is whole.
// Opening the app, a reconnect or a repaint sends nothing. It owns no polling or store: main_waiting.js hands it each
// `/api/state` body and the socket's state; the rooms' own Resume doors stay as they are.
import { resumeTask, withinRequestLifetime } from './api_client.js';
import { announceMembershipGap, closeExpiredOffer, createTaskCloseButton, deadlineExpiredWord, holdResumes, resumeAcceptedWords, resumePending, resumeRefusalWords, taskControlBusy }
    from './task_control_menu.js';
import { createSystemMessageActions } from './ui_helpers.js';
import { fmt, tx } from './i18n.js';
import { showToast } from './toast.js';
import { indexDeadlineExpired } from './work_status.js';

// One Resume answer may take this long (its writes wait on the queue and the disk); past it the outcome is unknown.
export const RESUME_ANSWER_MS = 60_000;
// Rows shown before `Show all`: a long list would push the rest of Main's feed far away from its one Resume.
export const SHOWN_ROWS = 8;
const MAIN_CHAT_ID = 1;
const PANIC_TOLD = 'ouro.panic_list_unread_told';  // this tab's record of the host run whose unreadable list was told
// What one answer says, by kind: its tone and, unless the door's own words say it, its sentence.
const ANSWERS = {
    pending: { tone: '', words: () => tx('Sending…') },
    accepted: { tone: 'ok' },
    refused: { tone: 'error' },
    changed: { tone: 'warn', words: () => tx('Not resumed: its stop changed after you chose it.') },
    gone: { tone: 'neutral', words: () => tx('Not resumed: it no longer waits for Resume.') },
    unknown: { tone: 'neutral', words: () => tx('No answer came back; it may already be back in the queue.') },
};
const BACK = new Set(['accepted', 'pending']);  // a current stop these answers hold is no longer the owner's to choose

const text = (value) => (typeof value === 'string' ? value.trim() : '');
const list = (value) => (Array.isArray(value) ? value : []);
const errorCode = (error) => text(error?.body?.error) || text(error?.body?.code) || text(error?.body?.reason_code);

/** One census row, or null when it cannot be acted on exactly: a Resume needs the task and the stop it names. */
export function panicRow(raw) {
    const row = raw && typeof raw === 'object' ? raw : {};
    const taskId = text(row.task_id);
    const hold = text(row.panic_hold_id);
    if (!taskId || !hold) return null;
    return { task_id: taskId, task_name: text(row.task_name), panic_hold_id: hold, project_id: text(row.project_id),
        chat_id: Number.isInteger(row.chat_id) ? row.chat_id : null, root_task_id: text(row.root_task_id),
        ...(row.close_expired ? { close_expired: closeExpiredOffer(row.close_expired, taskId) } : {}),
        // The host's own word that this hold's hard deadline passed, apart from any Close offer or Stop.
        ...(row.deadline_expired === true ? { deadline_expired: true } : {}) };
}

/**
 * The `main_attention` cards this message stands for, by task: the card of a work it names that offers only that
 * work's Resume (`attention_reason: resume`, no model wait), whose `Stop…` its row carries. A card for anything else
 * (a spending limit, a restart, model access) stays its own beside the message, as does a work the list cannot name.
 */
export function panicCards(data) {
    const named = new Set(list(data?.panic_recovery).map(panicRow).filter(Boolean).map((row) => row.task_id));
    const cards = new Map();
    for (const card of list(data?.main_attention)) {
        const id = String(card?.task_id || card?.activity_id || '');
        if (named.has(id) && ['resume', 'deadline_expired'].includes(card.attention_reason) && !Object.keys(card.model_waits || {}).length) {
            cards.set(id, card);
        }
    }
    return cards;
}

/**
 * The message: `insertMessageNode` and `stampNodeTimestamp` are the feed's (it stands at the live edge when first
 * seen), `mutate` its stable-viewport write, `refresh` the state read that settles a press, `onChange` the pill, and
 * `stopControl(taskId, {row, allowed})` the `Stop…` of an owner-action card (main_waiting.js).
 */
export function createPanicRecovery({ insertMessageNode, stampNodeTimestamp, mutate = (fn) => fn(), refresh = () => {},
    onChange = () => {}, resume = resumeTask, now = () => Date.now(), lifetimeMs = RESUME_ANSWER_MS,
    stopControl = null, notify = (text, options) => showToast(text, 'warn', options),
    isVisible = () => true,  // Main shown (main_waiting.js); a blocking surface holds the page's notices (toast.js)
    storage = globalThis.sessionStorage } = {}) {
    // Task id -> {row, rank, current, unseen, gone, listedAt, checked, answer, card, vouched, view}, shown in
    // first-seen order; `rank` is its place in the latest census. `current`: the latest read named it while the socket
    // was open; `unseen`: an incomplete read did not name it (unconfirmed); `gone`: a complete read did not (it no
    // longer waits), kept only as the record of an answer. `answer` is the last Resume's, for the stop in its `hold`.
    // `card` is the owner-action card whose Stop the row carries (`panicCards`, kept while no read drops it), `vouched`
    // whether the latest read naming the work offered it.
    const records = new Map();
    // What today's census says of its list of saved work: `complete`, `partial` (a read that names rows but not the
    // whole set, or a core not ready), or `unknown` (no read, a failed one, a closed socket). Only `complete` counts.
    // `listSaid` is what the host itself said of that list in the latest read naming one: `complete`, `partial` (its
    // explicit `false`, or a core it says is not ready) or `` (a host that says nothing, whose silence is neither). A
    // missing read keeps it: it withdraws every claim of a whole list, never the host's word that it is not whole.
    let census = 'unknown', censusAt = -Infinity, listSaid = '';
    let connected = true, sending = false, showAll = false, view = null, destroyed = false, serial = 0;
    let projects = new Map();   // project id and room chat id -> the Project's name, from the same body
    // A ready host's own word that its list is not whole, with no row read: Main's notice, told once per episode while
    // Main is the room shown (`isVisible`; chat.js's room seam, room_notice.js), withdrawn when Main is left or covered
    // and told again on return while the list still is not whole and the person has not closed it. `unread`: the episode
    // is open; `owed`: it waits to be told; `notice`: its toast while it stands. The episode outlives a reload of the same
    // tab: the host run it was told for (`boot.since`, the lifespan's start) is kept in this tab's storage.
    let unread = false, owed = false, stale = false, run = '', notice = null;
    const withdraw = () => { notice?.dispose?.(); notice = null; };
    const told = { get: () => { try { return storage?.getItem(PANIC_TOLD) || ''; } catch { return ''; } },
        set: (run) => { try { if (run) storage?.setItem(PANIC_TOLD, run); else storage?.removeItem(PANIC_TOLD); } catch { /* no storage */ } } };
    function tell() {
        withdraw();
        owed = false;
        told.set(run);
        notice = notify(tx('Saved work could not be checked for anything waiting to resume.'), { ttl: 0, haptic: false,
            // A person's Retry reads again; a read that still is not whole (or fails) says so again.
            action: { label: tx('Retry'), run: async () => {
                try { await refresh(); } catch { /* answered below */ }
                if (!destroyed && unread && !records.size) { owed = true; if (isVisible()) tell(); }
            } } });
    }
    // The room seam's view of this notice (chat_render_batch.js `others`). Left or covered, a notice still standing is
    // withdrawn and owed again (one the person closed is not), but only a fresh read on return may tell it: Main reads
    // nothing while away, and the host may have recovered meanwhile.
    const roomNotice = {
        get owed() { return Boolean(owed && !stale && unread && !records.size && !destroyed); },
        announce() { if (roomNotice.owed) tell(); },
        hide() { if (notice) { owed = notice.isConnected && !notice.classList?.contains('is-hiding'); stale = true; withdraw(); } },
    };

    const label = (record) => record.row.task_name || fmt('Task {id}', { id: record.row.task_id.slice(0, 8) });
    const answerOf = (record) => (record.answer?.hold === record.row.panic_hold_id ? record.answer : null);
    // Still stopped as far as the reads know: not gone, and no accepted or outgoing Resume holds its current stop.
    const waiting = (record) => !record.gone && !BACK.has(answerOf(record)?.kind);
    // A hard deadline that has passed is the host's own word, read from this row, its Close offer, or the same body's
    // exact activity, card, member or work (`late`, set by each read naming the row): such a row is a status line with
    // no box, outside every choice and count, whatever else (its own Stop, only where the card grants it) it keeps.
    const expired = (record) => waiting(record) && record.late === true;
    // Whether the latest read offers it to choose: still stopped and named by that read, no Resume of it out (this
    // press or a room's own door), and no answer about its current stop newer than that read. `Choose all N` counts
    // these; a press out holds every box without changing which rows they are.
    function available(record) {
        if (!record.current || !waiting(record) || expired(record) || resumePending(record.row.task_id)) return false;
        const answer = answerOf(record);
        return !answer || record.listedAt > answer.at;
    }
    const choosable = (record) => connected && !sending && available(record);
    // A press sends in the latest census order (the server lists roots before their descendants); it infers nothing.
    const chosen = () => [...records.values()].filter((record) => record.checked && choosable(record))
        .sort((a, b) => a.rank - b.rank);
    // Its card's Stop is that work's own right, whatever its Resume answered: it stands while the card grants it, acts
    // while the latest read naming the work offered that card and no request of this work is out (another row's slow
    // answer never holds it), and leaves when a fresh read no longer grants it.
    const stopGranted = (record) => record.card?.cancelable === true && !record.gone;
    const stoppable = (record) => connected && record.current && record.vouched && stopGranted(record)
        && record.answer?.kind !== 'pending' && !taskControlBusy(record.row.task_id);

    function build() {
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble system owner-action panic-recovery';
        bubble.dataset.ownerAction = 'panic-recovery';
        bubble.tabIndex = -1;  // the pill and a press land focus here when no check box can take it
        const sender = document.createElement('div');
        sender.className = 'sender';
        sender.textContent = 'System';
        const root = document.createElement('section');
        root.className = 'owner-action-body';
        root.setAttribute('role', 'group');
        root.setAttribute('aria-label', tx('Stopped by Panic'));
        const heading = document.createElement('div');
        heading.className = 'model-wait-heading';
        const title = document.createElement('strong');
        title.textContent = tx('Stopped by Panic');
        heading.append(title);
        const lead = document.createElement('div');
        lead.className = 'owner-action-reason';
        const gap = document.createElement('div');
        gap.className = 'ui-status';
        gap.dataset.panicGap = '';
        const picks = document.createElement('div');
        picks.className = 'panic-recovery-picks';
        const count = document.createElement('span');
        count.className = 'panic-recovery-count';
        const all = document.createElement('button');
        all.type = 'button';
        all.className = 'btn btn-ghost btn-sm';
        all.dataset.panicAll = '';
        const none = document.createElement('button');
        none.type = 'button';
        none.className = 'btn btn-ghost btn-sm';
        none.dataset.panicNone = '';
        none.textContent = tx('Choose none');
        picks.append(count, all, none);
        const rows = document.createElement('ul');
        rows.className = 'panic-recovery-list';
        const more = document.createElement('button');
        more.type = 'button';
        more.className = 'btn btn-ghost btn-sm ui-list-more';
        more.dataset.panicMore = '';
        const summary = document.createElement('div');
        summary.className = 'owner-action-reason';
        summary.dataset.panicSummary = '';
        summary.setAttribute('aria-live', 'polite');
        const go = document.createElement('button');
        go.type = 'button';
        go.className = 'btn btn-sm btn-primary ui-hit-area';
        go.dataset.panicResume = '';
        const actions = createSystemMessageActions(go);
        root.append(heading, lead, gap, picks, rows, more, summary, actions);
        bubble.append(sender, root);
        all.addEventListener('click', () => choose(true));
        none.addEventListener('click', () => choose(false));
        more.addEventListener('click', showEveryRow);
        go.addEventListener('click', () => { void press(); });
        stampNodeTimestamp(bubble, new Date(now()).toISOString());
        insertMessageNode(bubble);
        return { bubble, root, heading, lead, gap, picks, count, all, none, list: rows, more, summary, actions, go };
    }

    // The whole row is its box's label, so a finger reaches it anywhere; the box is named by the work's name and
    // described by its place and its last answer. Its card's `Stop…` stands beside the label, never inside it.
    function rowView(record) {
        const id = `panic-recovery-${serial += 1}`;
        const item = document.createElement('li');
        item.className = 'panic-recovery-row';
        item.dataset.taskId = record.row.task_id;
        const pick = document.createElement('label');
        pick.className = 'panic-recovery-pick';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.className = 'ui-checkbox';
        const name = document.createElement('span');
        name.className = 'panic-recovery-name';
        name.id = `${id}-name`;
        const meta = document.createElement('span');
        meta.className = 'panic-recovery-meta';
        meta.id = `${id}-meta`;
        const outcome = document.createElement('span');
        outcome.className = 'ui-status panic-recovery-outcome';
        outcome.id = `${id}-outcome`;
        input.setAttribute('aria-labelledby', name.id);
        input.setAttribute('aria-describedby', `${meta.id} ${outcome.id}`);
        pick.append(input, name, meta, outcome);
        // Close and the card's Stop… stand together at the row's end, whichever of them the host grants.
        const acts = document.createElement('div');
        acts.className = 'panic-recovery-acts';
        item.append(pick, acts);
        input.addEventListener('change', () => {
            // A box that cannot be chosen now keeps the record's choice (a read changed it while the event was queued).
            if (choosable(record)) record.checked = input.checked === true;
            paint();
        });
        return { item, pick, input, name, meta, outcome, acts, stop: null };
    }

    // The card's `Stop…` (its menu offers no Resume: the box is that), made once and kept while its card grants it.
    function stopView(record) {
        const wanted = Boolean(stopControl) && stopGranted(record);
        if (wanted && !record.view.stop) {
            const stop = stopControl(record.row.task_id, { row: () => record.card, allowed: () => stoppable(record) });
            stop.setAttribute('aria-describedby', record.view.name.id);  // which work, in a list of them
            record.view.acts.append(stop);
            record.view.stop = stop;
        }
        if (!record.view.stop) return;
        record.view.stop.hidden = !wanted;
        record.view.stop.disabled = !stoppable(record);
    }

    function choose(on) {
        for (const record of records.values()) record.checked = on ? record.checked || choosable(record) : false;
        paint();
    }

    function showEveryRow() {
        const first = [...records.values()].find((record) => record.view?.item.hidden);
        showAll = true;
        paint();
        // The control leaves with its press: focus goes to the first row it revealed.
        (first && !first.view.input.disabled && !first.view.input.hidden ? first.view.input : view.bubble).focus?.({ preventScroll: true });
    }

    // Where a work lives, when the body says (its Project's name, or Main), a twin name's short id, and what the reads
    // say of a row they no longer name.
    function metaWords(record, twins) {
        const { project_id: projectId, chat_id: chatId, task_id: taskId } = record.row;
        const place = projects.get(`p:${projectId}`) || (chatId !== null ? projects.get(`c:${chatId}`) : '')
            || (!projectId && chatId === MAIN_CHAT_ID ? tx('Main') : '');
        const kind = record.answer?.kind;
        const read = record.gone ? (['refused', 'changed'].includes(kind) ? tx('no longer waits for Resume') : '')
            : record.unseen && connected ? tx('not confirmed by the latest read') : '';
        return [place, twins ? taskId.slice(0, 8) : '', read].filter(Boolean).join(' · ');
    }

    // A row says its last answer. One about a stop the work is no longer listed under is a record of that earlier
    // stop, said so and without a tone; a changed stop's answer already says it.
    function outcomeView(record, outcome) {
        const answer = record.answer;
        const said = answer ? ANSWERS[answer.kind] : null;
        let words = !said ? '' : answer.kind === 'unknown' && record.gone
            ? tx('No answer came back; the latest list no longer holds it.') : said.words ? said.words() : answer.words || '';
        const earlier = Boolean(words) && answer.hold !== record.row.panic_hold_id && answer.kind !== 'changed';
        if (earlier) words = fmt('Earlier stop: {answer}', { answer: words });
        outcome.textContent = words;
        if (said?.tone && words && !earlier) outcome.dataset.tone = said.tone; else delete outcome.dataset.tone;
    }

    // After the first answer shown, each row counted once, by the stop it is listed under: accepted (each row's line
    // says what acceptance means for it), still stopped (only what the latest read names; a refused row too, and one
    // whose answer was an earlier stop's: its line says why), no answer, and not resumed and no longer listed. A row
    // no longer listed counts by its last answer. A row the reads no longer confirm is not counted as stopped.
    function summaryWords(all) {
        if (!all.some((record) => record.answer && record.answer.kind !== 'pending')) return '';
        const counts = { accepted: 0, still: 0, lost: 0, missed: 0 };
        for (const record of all) {
            if (record.answer?.kind === 'pending') continue;
            const kind = (record.gone ? record.answer : answerOf(record))?.kind;
            if (kind === 'accepted') counts.accepted += 1;
            else if (kind === 'unknown') counts.lost += 1;
            else if (record.current && waiting(record)) counts.still += 1;
            else if (record.gone && record.answer) counts.missed += 1;
        }
        return [[counts.accepted, '{n} accepted'], [counts.still, '{n} still stopped'],
            [counts.lost, '{n} without an answer'], [counts.missed, '{n} not resumed']]
            .filter(([n]) => n > 0).map(([n, key]) => fmt(key, { n })).join(' · ');
    }

    function paint() {
        if (!view) return;
        mutate(() => {
            const all = [...records.values()];
            const open = all.filter(record => waiting(record) && !expired(record));
            const offered = all.filter(available);
            const picked = chosen();
            const names = new Map();
            for (const record of all) names.set(label(record), (names.get(label(record)) || 0) + 1);
            all.forEach((record, index) => {
                record.view ||= rowView(record);
                const { item, pick, input, name, meta, outcome } = record.view;
                if (item.parentNode !== view.list) view.list.append(item);
                name.textContent = label(record);
                meta.textContent = metaWords(record, names.get(label(record)) > 1);
                outcomeView(record, outcome);
                const can = choosable(record);
                input.checked = record.checked && can;
                input.disabled = !can;
                // A row that is no longer stopped keeps no box: it is the record of its answer.
                const late = expired(record);
                const token = late ? record.row.close_expired || '' : '';
                if (late) record.checked = false;
                input.hidden = !waiting(record) || late;
                pick.classList.toggle('is-locked', !can);
                // The deadline word leads; an answer already recorded for this row keeps its own words beside it.
                if (late) outcome.textContent = [deadlineExpiredWord(), outcome.textContent].filter(Boolean).join(' · ');
                if (record.view.close && record.view.close.offer !== token) {
                    record.view.close.remove(); record.view.close = null;
                }
                if (token && !record.view.close) {
                    const close = createTaskCloseButton(record.row.task_id, { onSettled: refresh, nameId: name.id,
                        tokenNow: () => connected && record.current && expired(record) && record.row.close_expired === token ? token : '',
                    });
                    close.offer = token;
                    record.view.close = close;
                    record.view.acts.prepend(close);
                }
                if (record.view.close) record.view.close.disabled = !connected || !record.current || taskControlBusy(record.row.task_id);
                stopView(record);
                item.hidden = !showAll && index >= SHOWN_ROWS && !record.checked;
            });
            view.more.hidden = !all.some((record) => record.view.item.hidden);
            view.more.textContent = fmt('Show all ({n})', { n: all.length });
            // While chosen Resumes are out nothing is claimed either way. Nothing waits only by a complete census read
            // after the last answer: it lists saved work Panic holds, not every interrupted work.
            const answered = Math.max(-Infinity, ...all.map((record) => record.answer?.at ?? -Infinity));
            view.lead.textContent = sending ? '' : open.length ? tx('Choose which to resume. The others stay stopped.')
                : census === 'complete' && censusAt > answered ? tx('No saved work stopped by Panic waits for Resume now.') : '';
            view.lead.hidden = !view.lead.textContent;
            // The host's word that its list is not whole stands until a read says it is, whatever is left to choose (the
            // last one answered) and through a missing read, which adds nothing of its own here (the room says the
            // outage). The message exists only while it names a row (`snapshot`).
            view.gap.textContent = listSaid === 'partial' ? tx('Some saved work could not be read, so this list may be incomplete.') : '';
            if (view.gap.textContent) view.gap.dataset.tone = 'warn'; else delete view.gap.dataset.tone;
            view.picks.hidden = open.length < 2;
            view.count.textContent = offered.length
                ? fmt('{chosen} of {total} chosen', { chosen: picked.length, total: offered.length }) : '';
            // `Choose all N` chooses exactly the N rows it names, never a row a read does not offer now.
            view.all.textContent = fmt('Choose all {total}', { total: offered.length });
            view.all.hidden = census !== 'complete' || offered.length < 2;
            view.all.disabled = sending || !offered.some((record) => !record.checked);
            view.none.disabled = !picked.length;
            view.summary.textContent = summaryWords(all);
            view.summary.hidden = !view.summary.textContent;
            view.go.hidden = !open.length && !sending;
            // With nothing left to resume the row of actions holds nothing: it takes no room.
            view.actions.hidden = view.go.hidden;
            view.go.textContent = sending ? tx('Sending…')
                : picked.length ? fmt('Resume {n} selected', { n: picked.length }) : tx('Resume selected');
            view.go.disabled = sending || !picked.length;
            if (sending) view.go.setAttribute('aria-busy', 'true'); else view.go.removeAttribute('aria-busy');
            return true;
        });
        onChange();
    }

    // One chosen row's Resume, answered in one of the kinds above; it never throws and never sends twice. Its lifetime's
    // signal reaches the transport: an ended lifetime cancels the request, never an effect it may already have had.
    async function resumeOne(taskId, hold) {
        try {
            const ack = await withinRequestLifetime((signal) => resume(taskId, { expected_panic_hold_id: hold }, { signal }),
                { lifetimeMs });
            if (ack?.ok !== true) return { kind: 'unknown' };
            announceMembershipGap(ack.membership_gaps, 'resume');
            return { kind: 'accepted', words: tx(resumeAcceptedWords(ack)) };
        } catch (error) {
            const status = Number(error?.status);
            // Only this code says the stop itself changed; every other refusal (the installation's own stop still in
            // force, a state that cannot be read, an independent pause...) is said in the host's words.
            if (status === 409 && errorCode(error) === 'panic_selection_changed') return { kind: 'changed' };
            if (status === 404) return { kind: 'gone' };
            if (status >= 400 && status < 500) return { kind: 'refused', words: resumeRefusalWords(error) };
            // A server failure, a lost answer or the lifetime proves nothing: no answer, which is no refusal.
            return { kind: 'unknown' };
        }
    }

    async function press() {
        const picked = connected && !sending ? chosen() : [];
        if (!picked.length) return;
        sending = true;
        const sent = picked.map((record) => ({ record, taskId: record.row.task_id, hold: record.row.panic_hold_id }));
        for (const { record, hold } of sent) {
            record.checked = false;
            record.answer = { kind: 'pending', hold, at: Infinity };
        }
        // The press disables its own button: focus stays in the message instead of falling to the page.
        if (view.bubble.contains?.(document.activeElement)) view.bubble.focus?.({ preventScroll: true });
        paint();
        try {
            await holdResumes(sent.map(({ taskId }) => taskId), async (answeredFor) => {
                for (const { record, taskId, hold } of sent) {
                    // A torn-down page sends nothing more: what it never sent stays stopped for the next look.
                    if (destroyed) break;
                    const answer = await resumeOne(taskId, hold);
                    answeredFor(taskId);  // its own controls are free again; every Resume door still waits for the read
                    record.answer = { ...answer, hold, at: now() };
                    paint();
                }
            }, async () => { try { if (!destroyed) await refresh(); } catch { /* the next census settles it */ } });
        } finally {
            sending = false;
            paint();
        }
    }

    function remove() {
        view?.bubble.remove();
        view = null;
        showAll = false;
        for (const record of records.values()) record.view = null;
    }

    return {
        // A body of `/api/state`, or null for a failed read; `requestedAt` is when that read was asked for.
        snapshot(data, requestedAt = Infinity) {
            // As main_waiting.js: an HTTP answer while the socket is down restores nothing until the reopen's census.
            if (!connected) return;
            const rows = Array.isArray(data?.panic_recovery) ? data.panic_recovery : null;
            if (!rows) {
                // A failed read, or a body that names no projection (an older host): nothing current is known, nothing
                // is said gone, whole or stopped.
                census = 'unknown';
                for (const record of records.values()) record.current = false;
                paint();
                return;
            }
            // Whole only by the host's own flag on a ready core; any row nothing can act on makes it partial.
            let complete = data.panic_recovery_complete === true && data.supervisor_ready !== false;
            projects = new Map();
            for (const project of Array.isArray(data.projects) ? data.projects : []) {
                if (!text(project?.id) || !text(project?.name)) continue;
                projects.set(`p:${project.id}`, project.name);
                if (Number.isInteger(project.chat_id)) projects.set(`c:${project.chat_id}`, project.name);
            }
            const cards = panicCards(data);
            const expiredIds = indexDeadlineExpired(data);
            // A work's card this read offers or leaves out: one the message stands for, one of its own beside it (its
            // own Stop there), none on a whole attention list; else (a list this read could not finish) the last one,
            // withdrawn.
            const attention = new Set(list(data.main_attention).map((card) => String(card?.task_id || card?.activity_id || '')));
            const cardOf = (taskId, kept) => cards.get(taskId)
                || (attention.has(taskId) || data.main_attention_complete === true ? null : kept);
            const named = new Set();
            for (const raw of rows) {
                const row = panicRow(raw);
                if (!row) { complete = false; continue; }  // a row nothing can act on: the set is not whole
                named.add(row.task_id);
                const record = records.get(row.task_id) || { checked: false, answer: null, card: null, view: null };
                // A choice belongs to the stop it was made under; an answer keeps its own stop (`outcomeView`).
                if (record.row && record.row.panic_hold_id !== row.panic_hold_id) record.checked = false;
                const late = Boolean(row.close_expired || row.deadline_expired === true || expiredIds.has(row.task_id));
                if (late) record.checked = false;
                Object.assign(record, { row, rank: named.size, current: true, unseen: false, gone: false,
                    listedAt: requestedAt, card: cardOf(row.task_id, record.card), vouched: cards.has(row.task_id), late });
                records.set(row.task_id, record);
            }
            census = complete ? 'complete' : 'partial';
            // Only the host's own word says the list is not whole (its flag, or a core it says is not ready): their
            // absence (an older host) says nothing either way.
            listSaid = complete ? 'complete'
                : data.panic_recovery_complete === false || data.supervisor_ready === false ? 'partial' : '';
            censusAt = requestedAt;
            for (const [taskId, record] of records) {
                if (named.has(taskId)) continue;
                record.current = false;
                record.vouched = false;
                record.card = cardOf(taskId, record.card);
                if (census !== 'complete') {
                    record.unseen = true;
                    // On a partial Panic list, an authoritative expiry for this exact task in the same response
                    // removes it from selection and counts immediately while preserving unseen non-expired neighbors.
                    if (expiredIds.has(taskId)) { record.late = true; record.checked = false; }
                    continue;
                }
                // A complete read proves the work no longer waits: an answered row stays as the record of its answer.
                if (record.answer) { record.gone = true; record.checked = false; continue; }
                // A row leaving under the keyboard hands focus to its message, never to the page.
                if (record.view?.item.contains?.(document.activeElement)) view?.bubble.focus?.({ preventScroll: true });
                record.view?.item.remove();
                records.delete(taskId);
            }
            // With no row read Main holds no message. A host not ready yet is said by the room bar's Starting…
            // (host_starting_status.js) and the setup wizard covers everything, so nothing more; a ready host's own word
            // that its list is not whole is told once per episode by the page's notice, claiming no Panic, stop or work.
            stale = false;
            if (census === 'complete' || records.size) { withdraw(); owed = false; }
            if (census === 'complete') { unread = false; told.set(''); }
            if (!records.size) {
                const ready = data.supervisor_ready === true && data.boot?.tasks_ready !== false;
                const seen = typeof data.boot?.since === 'string' ? data.boot.since : '';
                // A host that restarted inside this tab is a new run: its own episode, told once.
                if (unread && seen && run && seen !== run) { unread = false; owed = false; withdraw(); }
                if (listSaid === 'partial' && ready && !unread) {
                    unread = true;
                    run = seen;
                    owed = !run || told.get() !== run;  // a reload of this tab keeps the decision taken on this run
                }
                if (roomNotice.owed && isVisible()) tell();
                remove(); onChange(); return;
            }
            if (!view) mutate(() => { view = build(); return true; });
            paint();
        },
        setConnected(value) {
            connected = value === true;
            // A closed socket withdraws every box and the census; only one the reopen sequences names rows again.
            if (!connected) {
                census = 'unknown';
                for (const record of records.values()) record.current = false;
            }
            paint();
        },
        // The pill's target while something here waits on the owner: work still stopped (or not confirmed otherwise).
        target() {
            if (!view || ![...records.values()].some(record => waiting(record) && !expired(record))) return null;
            return { at: Number(view.bubble.dataset.ts) || 0 };
        },
        node: () => view?.bubble || null,
        // Where the pill's landing puts focus: the first shown box that can be chosen (choosing sends nothing).
        door: () => [...records.values()].find((record) => choosable(record) && !record.view?.item.hidden)?.view?.input || null,
        destroy() { destroyed = true; withdraw(); remove(); records.clear(); },
        notice: roomNotice,
    };
}
