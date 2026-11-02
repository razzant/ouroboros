/**
 * One work, one status in the browser (DESIGN "Room work status"): readers over the `/api/state` work projection
 * (ouroboros/gateway/work_status.py; shapes in work_types.js). The server groups every live member into works, rolls
 * works into rooms, decides each work's controls with their exact targets and names each work's and room's mark
 * (`mark`: moving | still | pause | ring | none); this module says those facts in the product's short words (a work's
 * word, and beside it its line: the episode's own reason, then what still runs beside it), keeps what an incomplete
 * body did not list again (unconfirmed: a ring), and lays a work's controls on the shared task-control doors. It
 * derives no word, mark or control of its own. Four facts only this client knows stay
 * overlays, each with its mark: the socket closed (Reconnecting…, a ring), the owner's own unconfirmed send (Sending,
 * moving), the supervisor not ready yet (Starting…, no mark) and the owner's own Stop press on a card (Cancelling… /
 * Finalizing…, moving). Whether something waits for the owner (`owner`: the server's `turn`) is a separate fact;
 * status_mark.js draws the two together.
 */
import { fmt, tr } from './i18n.js';
import { taskPresentation, taskTerminalPhase } from './log_events.js';
import { startingStatus } from './host_starting_status.js';
import { chipWord } from './task_phase_chip.js';
import {
    ACTION_FINALIZE, ACTION_HURRY, ACTION_PAUSE, ACTION_RESUME, ACTION_STOP_EARLIER, ACTION_STOP_NOW, REUSABLE_TASK_IDS,
    hurryTaskAction, openTaskControlMenu, openTaskControls, requestStop, requestStopAll, resumeTaskAction, taskControlActions,
    taskControlBusy,
} from './task_control_menu.js';

// A known outcome keeps its word's tone, its dot beside it (status_mark.js): Done and Done with warnings succeed (the
// word says the warnings), Stopped was someone's decision, the rest failed. An unfinished lifecycle (Finalizing…) is
// said beside it.
export const OUTCOME_TONES = { done: 'done', warn: 'done', cancelled: 'stopped', error: 'failed', timeout: 'failed',
    lifecycle_error: 'failed' };
const IDLE_ROOM = Object.freeze({ activity: 'idle', transition: 'none', call: '', turn: [], amber: false, mark: 'none',
    confirmed: true, works: [] });
const CALL_WORDS = { preparing: 'preparing', sending: 'sending', thinking: 'thinking' };
const list = (value) => (Array.isArray(value) ? value : []);
const asked = (state) => list(state?.turn).some((item) => item?.kind === 'question');

/** The state one chat instance shows before its first body: an idle room and no works. */
export function emptyWorkState() {
    return { room: IDLE_ROOM, works: [], episodes: new Map(), members: new Set(), attempts: new Map(), complete: false };
}

// What an incomplete body did not list again: still shown, never as confirmed work and never with controls or a line;
// its mark is the ring of unconfirmed state.
const unconfirmedWork = (work) => ({ ...work, activity: 'unknown', transition: 'none', call: '', because: [], tails: [],
    live: false, turn: [], controls: [], confirmed: false, mark: 'ring' });

/**
 * One body applied to an instance's work state. `room` and `works` are this chat's (`works[].room`, keyed as the
 * server keys `rooms`); `episodes` maps every work's current episode task id to its work in EVERY room (a Main
 * portal's work runs in its Project) and `members` holds every member id; `attempts` is each census row's attempt,
 * so a control opened for one attempt of an episode is not pressed on the next. Only a complete supervisor-ready
 * body read over an open socket proves absence; an incomplete one keeps the works it omitted, unconfirmed. A body
 * without the projection, or a failed read (null), changes nothing.
 */
export function reconcileWorkState(prior, data, chatId, connected = true) {
    const before = prior || emptyWorkState();
    if (!data || typeof data !== 'object' || !Array.isArray(data.works)) return before;
    const complete = Boolean(connected) && data.works_complete === true && data.supervisor_ready === true;
    const chat = String(chatId);
    const all = data.works.filter((work) => work && typeof work === 'object' && work.work_id);
    const listed = all.filter((work) => String(work.room || 0) === chat);
    const ids = new Set(listed.map((work) => work.work_id));
    const kept = complete ? [] : before.works.filter((work) => !ids.has(work.work_id)).map(unconfirmedWork);
    const served = data.rooms && typeof data.rooms === 'object' ? data.rooms[chat] : null;
    const room = served && typeof served === 'object' ? served
        : kept.length ? { ...IDLE_ROOM, activity: 'unknown', mark: 'ring', confirmed: false, works: kept.map((w) => w.work_id) }
            : { ...IDLE_ROOM, confirmed: complete };
    const episodes = new Map();
    const members = new Set();
    for (const work of all) {
        if (work.episode?.task_id) episodes.set(String(work.episode.task_id), work);
        for (const id of list(work.members?.ids)) members.add(String(id));
    }
    const attempts = new Map(list(data.active_chat_activities).filter((row) => row?.activity_id)
        .map((row) => [String(row.activity_id), row.task_attempt ?? null]));
    return { room, works: [...listed, ...kept], episodes, members, attempts, complete };
}

/** A work's or a room's one short word, from its activity, transition and exact call (the cards' chip words). */
export function activityWord(state = {}) {
    switch (state.activity) {
    case 'working':
        if (state.transition === 'finalizing') return chipWord('finalizing');
        if (state.transition === 'pausing') return chipWord('pausing');
        return chipWord(CALL_WORDS[state.call] || 'working');
    case 'queued': return chipWord('queued');
    case 'waiting': return asked(state) ? chipWord('ownerWait') : chipWord('waiting');
    case 'paused': return chipWord('paused');
    case 'idle': return 'Online';
    default: return chipWord('unconfirmed');
    }
}

// A counted reason says its number in the catalog's template, so a translation keeps the count. These are the
// server's sentences (gateway/work_status.py REASON_TEXT); the web fixture pins them to the server's own.
const COUNTED_REASONS = {
    earlier_attempt_finishing: (n) => (n === 1
        ? tr('work.because.earlier_attempt_finishing', 'Continue starts when the helper of the earlier attempt finishes')
        : fmt('Continue starts when {count} helpers of the earlier attempt finish', { count: n, n })),
    earlier_runs_finishing: (n) => (n === 1
        ? tr('work.because.earlier_runs_finishing', 'Continue starts when the external run of the earlier attempt finishes')
        : fmt('Continue starts when {count} external runs of the earlier attempt finish', { count: n, n })),
    helpers_running: (n) => (n === 1 ? tr('work.because.helpers_running', '1 helper still working')
        : fmt('{count} helpers still working', { count: n, n })),
    delegated_running: (n) => (n === 1 ? tr('work.because.delegated_running', '1 external run still working')
        : fmt('{count} external runs still working', { count: n, n })),
    external_runs_unconfirmed: (n) => (n === 1 ? tr('work.because.external_runs_unconfirmed', '1 external run may still be running')
        : fmt('{count} external runs may still be running', { count: n, n })),
    awaited_helpers_mind: (n) => (n === 1
        ? tr('work.because.awaited_helpers_mind', '1 helper it waits for is paused until Ouroboros resumes it')
        : fmt('{count} helpers it waits for are paused until Ouroboros resumes them', { count: n, n })),
    awaited_helpers_owner: (n) => (n === 1
        ? tr('work.because.awaited_helpers_owner', '1 helper it waits for is paused until you resume it')
        : fmt('{count} helpers it waits for are paused until you resume them', { count: n, n })),
    awaited_helpers_paused: (n) => (n === 1 ? tr('work.because.awaited_helpers_paused', '1 helper it waits for is paused')
        : fmt('{count} helpers it waits for are paused', { count: n, n })),
};

// Helpers and external runs still working side by side are said in one sentence, so the line never says it twice.
function tailWords(tails) {
    const counted = (code) => tails.find((tail) => tail?.code === code && Number.isInteger(tail.count));
    const helpers = counted('helpers_running');
    const runs = counted('delegated_running');
    if (!helpers || !runs) return tails.map(reasonText);
    const both = fmt('{helpers} and {runs} still working', {
        helpers: helpers.count === 1 ? tr('work.because.one_helper', '1 helper')
            : fmt('{count} helpers', { count: helpers.count, n: helpers.count }),
        runs: runs.count === 1 ? tr('work.because.one_external_run', '1 external run')
            : fmt('{count} external runs', { count: runs.count, n: runs.count }),
    });
    return tails.flatMap((tail) => (tail === helpers ? [both] : tail === runs ? [] : [reasonText(tail)]));
}

/** One reason in the owner's words: the server's code read through the catalog, its sentence as the English (a
 *  code this client has never seen still reads its sentence). A count stays a parameter of its template; a counted
 *  code this client does not know keeps the server's sentence as it is. An empty sentence (a Pause that declared
 *  nobody) says nothing, and Ouroboros's own sentence is its prose: only the frame around it is translated. */
export function reasonText(reason) {
    if (!reason || typeof reason !== 'object') return '';
    const code = String(reason.code || '');
    const text = typeof reason.text === 'string' ? reason.text : code;
    if (!code || !text) return text;
    if (Number.isInteger(reason.count)) return COUNTED_REASONS[code] ? COUNTED_REASONS[code](reason.count) : text;
    if (code === 'paused_by_mind' && reason.by?.text) return `${tr(`work.because.${code}`, text)}: ${String(reason.by.text)}`;
    return tr(`work.because.${code}`, text);
}

/** A moment in the viewer's clock (epoch seconds or an ISO instant): its time today, its date as well on another day,
 *  so an old one never reads as today's; '' when there is none. */
export function clock(at) {
    const ms = Number.isFinite(Number(at)) ? Number(at) * 1000 : Date.parse(String(at ?? ''));
    const date = new Date(ms);
    if (!ms || !Number.isFinite(date.getTime())) return '';
    return new Date().toDateString() === date.toDateString()
        ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        : date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

/**
 * A work's line, said beside its word (the card's quieter line, the work-list row's, the sidebar's name after the word):
 * the current episode's own sentence (the owner's Pause with the time it was pressed), then what still runs beside
 * it, one sentence per tail, joined " · ". The line never repeats `word`: an answer wait's own words are compared by
 * their code, in any language.
 */
export function workLine(work = {}, word = activityWord(work)) {
    const first = list(work.because)[0];
    const said = first?.code === 'owner_answer' && word === chipWord('ownerWait') ? '' : reasonText(first);
    const sentence = said === word ? '' : said;
    const pressed = sentence && first.code === 'paused_by_owner' ? clock(first.since) : '';
    return [sentence, pressed, ...tailWords(list(work.tails))].filter(Boolean).join(' · ');
}

/** The first of a room's works that says the room's own word, else null: a work Pausing beside a running one never
 *  lends its reason to the room's Working. */
export function roomSpeaker(room, works = [], word = activityWord(room)) {
    const byId = new Map(list(works).map((work) => [work?.work_id, work]));
    return list(room?.works).map((id) => byId.get(id)).find((work) => work && activityWord(work) === word) || null;
}

/** What waits for the owner in a room ('question' | 'action'), '' when nothing does or its own word says so. */
export const roomAmber = (room) => (!room?.amber || (room.activity === 'waiting' && asked(room)) ? ''
    : asked(room) ? 'question' : 'action');

/** The amber mark said in words, for the room's accessible name: the colour is never the only way it is told. */
export const amberNote = (kind) => (kind === 'question' ? tr('work.amber.question', 'A question waits for your answer')
    : kind === 'action' ? tr('work.amber.action', 'Something here waits for you') : '');

/**
 * The room's status in its bar: the server's room in short words, with the client's own facts on top. Tone is the
 * word's ink (live, queued, wait for the owner's own question, held, unknown, starting, offline, online); `mark` the
 * work's shape after it (the server's room mark under this module's four overlays; the transcript's typing bubble
 * shows while it is `moving`, whatever waits for the owner); `owner` whether something waits for the owner (the room's
 * turn: its amber dot takes the mark's place, the word keeps its ink); `amber` what waits (question | action), said in
 * words where the word does not already say it. `stopping` is the owner's own Stop press on every working work of the
 * room ('finalize' | 'immediate'); until the host is `ready`, an idle room says its start in the words of the host's
 * `boot` phases (host_starting_status.js). A room still preparing names as its `step` the step its speaking work
 * stands at (a direct turn's start step), said beside the word and dropped before it on a short bar, so nothing in
 * the bar reads as a model call before one is sent.
 */
export function roomStatusView(state, { connected = true, ready = true, boot = null, sending = false, stopping = '' } = {}) {
    const view = (kind, text, tone, mark, extra = {}) => ({ kind, text, tone, mark, owner: false, amber: '', ...extra });
    if (!connected) return view('offline', 'Reconnecting…', 'offline', 'ring');
    const room = state?.room || IDLE_ROOM;
    const mark = String(room.mark || 'none');
    const turn = { owner: list(room.turn).length > 0, amber: roomAmber(room) };
    if (room.activity !== 'working' && sending) return view('thinking', chipWord('sending'), 'live', 'moving', turn);
    switch (room.activity) {
    case 'working': {  // a Pause still settling keeps the neutral ink; its mark moves while work stops
        const word = stopping ? chipWord(stopping === 'finalize' ? 'finalizing' : 'cancelling') : activityWord(room);
        const step = !stopping && room.call === 'preparing'
            ? reasonText(list(roomSpeaker(room, state?.works, word)?.because)[0]) : '';
        return view('thinking', word, room.transition === 'pausing' && !stopping ? 'held' : 'live',
            stopping ? 'moving' : mark, step ? { ...turn, step } : turn);
    }
    case 'queued': return view('thinking', activityWord(room), 'queued', mark, turn);
    // The owner's own question says itself: the word takes the amber ink, beside the owner's dot.
    case 'waiting': return view('online', activityWord(room), asked(room) ? 'wait' : 'held', mark, turn);
    case 'paused': return view('online', activityWord(room), 'held', mark, turn);
    case 'unknown': return view('online', activityWord(room), 'unknown', mark, turn);  // a listed turn still waits
    default: return ready ? view('online', 'Online', 'online', mark)
        : { ...view('starting', 'Starting…', 'starting', 'none'), ...startingStatus(boot) };
    }
}

/** The owner's own Stop press on every working work of the room (a card's pending policy), else ''. */
export function ownStopPress(state, records) {
    const working = list(state?.works).filter((work) => work.activity === 'working' && work.confirmed);
    const policies = new Set(working.map((work) => records?.get?.(String(work.episode?.task_id || ''))?.cancelPendingPolicy || ''));
    return working.length && policies.size === 1 ? [...policies][0] : '';
}

/**
 * One work's row in the room's work list: its word (an outcome beside Finalizing…), its line beside it, the row's tone
 * (the word's ink), its `mark` (the work's shape as the server names it; the owner's own Stop press moves it), `owner`
 * whether something in the work waits for the owner (its turn: the amber dot takes the mark's place; a Pause by the
 * owner or by Ouroboros is not the owner's move) and `outcome`, the known outcome said beside Finalizing… (its dot in
 * the outcome's tone). `stopping` is the owner's own Stop press on the work's card ('finalize' | 'immediate'), said
 * at once as the card says it.
 */
export function workRowView(work = {}, { stopping = '' } = {}) {
    const owner = list(work.turn).length > 0;
    if (stopping && work.activity === 'working' && work.confirmed !== false) {
        return { word: chipWord(stopping === 'finalize' ? 'finalizing' : 'cancelling'), secondary: '', tone: 'live',
            mark: 'moving', owner, outcome: '', focus: '' };
    }
    const outcome = work.transition === 'finalizing' && work.outcome
        ? taskPresentation(taskTerminalPhase({ status: work.outcome })) : null;
    const word = outcome ? outcome.headline : activityWord(work);
    const secondary = [outcome ? chipWord('finalizing') : '', workLine(work, word)].filter(Boolean).join(' · ');
    const pausing = work.activity === 'working' && work.transition === 'pausing';
    const tone = outcome ? OUTCOME_TONES[outcome.phase] || 'live'
        : !work.confirmed || work.activity === 'unknown' ? 'unknown'
            : work.activity === 'waiting' && asked(work) ? 'wait'
                : work.activity === 'working' && !pausing ? 'live' : work.activity === 'queued' ? 'queued' : 'held';
    // The work's own focus line: what its author says it is doing, under the state (update_focus).
    const focus = String(work.focus?.text || '').trim();
    return { word, secondary, tone, mark: String(work.mark || 'none'), owner, outcome: outcome?.phase || '', focus };
}

/** The question a row's View question goes to: the first question in the work's turn, by its exact quiz id. */
export function workQuestion(work) {
    const question = list(work?.turn).find((item) => item?.kind === 'question' && item.quiz_id && item.task_id);
    return question ? { taskId: String(question.task_id), quizId: String(question.quiz_id) } : null;
}

/**
 * What the card of a work's current episode says (one work, one status): the work's word alone in the chip, and its
 * line as the chip's quieter second fact (the episode's reason, then what still runs beside it: reviews a Pause lets
 * finish, helpers, external runs). Plain running work with nothing to say beside it, finalizing work (its outcome
 * beside Finalizing…) and unconfirmed work keep the card's own words; so does a card whose episode is not live. The
 * word is the live one only while the work plainly runs; its mark is the work's own (applyCardWorkViews keeps it).
 */
export function cardWorkView(work) {
    if (!work || !work.confirmed || !work.episode?.live || work.activity === 'unknown' || work.transition === 'finalizing') return null;
    const word = activityWord(work);
    const line = workLine(work, word);
    if (work.activity === 'working' && work.transition === 'none' && !line) return null;
    const running = work.activity === 'working';
    const classes = list(work.turn).length ? 'warn' : running ? (work.transition === 'pausing' ? 'working waiting' : 'working')
        : work.activity;
    return {
        phase: running ? 'working' : work.activity, text: word, secondary: line, classes,
        tone: running && work.transition === 'none' ? 'live' : 'neutral', waiting: !running || work.transition !== 'none',
    };
}

/**
 * Every card's work view from one state; true when a card's view changed (its chip repaints with its next write).
 * Apart from the optional word, a work's live current episode keeps the work's look: its mark and whether something
 * in it waits for the owner (the server takes the turn from every member, a helper's included). A turn the body lists
 * is a fact even when another member could not be confirmed; a work an incomplete body omitted keeps no look.
 */
export function applyCardWorkViews(records, state) {
    let changed = false;
    for (const record of records?.values?.() || []) {
        if (record?.isSubagent) continue;
        const work = state?.episodes?.get(String(record.groupId || ''));
        const view = cardWorkView(work);
        const look = work?.episode?.live ? { mark: String(work.mark || 'none'), owner: list(work.turn).length > 0 } : null;
        if (JSON.stringify([view, look]) !== JSON.stringify([record.workView || null, record.workLook || null])) changed = true;
        record.workView = view;
        record.workLook = look;
    }
    return changed;
}

/**
 * A work's controls, one press each: Stop (every episode root that still owns live work, in the server's order),
 * Stop earlier attempt (each earlier root that does) and the current episode's Pause, Resume, Hurry and Wrap up.
 * Unknown actions are not offered.
 */
export function workControls(work) {
    const out = { stop: [], stopEarlier: [], pause: '', resume: '', hurry: '', wrapUp: '' };
    for (const control of list(work?.controls)) {
        // A reusable slot (`active`) hosts many cycles: no control reaches one of them by that id.
        const target = String(control?.target || '').trim();
        if (!target || REUSABLE_TASK_IDS.has(target)) continue;
        if (control.action === 'resume' && control.scope === 'single') continue;
        const many = { stop: out.stop, stop_earlier: out.stopEarlier }[control.action];
        if (many) { if (!many.includes(target)) many.push(target); continue; }
        const one = { pause: 'pause', resume: 'resume', hurry: 'hurry', wrap_up: 'wrapUp' }[control.action];
        if (one && !out[one]) out[one] = target;
    }
    return out;
}

/** The confirmed work whose live current episode is `taskId` in one body, else null. */
export function workOfEpisode(data, taskId) {
    const id = String(taskId || '');
    return list(data?.works).find((work) => work?.confirmed !== false && work.episode?.live
        && String(work.episode.task_id || '') === id) || null;
}

/** The server's state for an individual member, without interpreting queue carriers in the browser. */
export function memberState(data, taskId) {
    const id = String(taskId || '');
    return list(data?.works).find((work) => work?.confirmed !== false && work.members?.states?.[id])
        ?.members.states[id] || null;
}

/** Every task a confirmed work's Resume control names in one body: where Activity may offer Resume. */
export function resumeTargets(data) {
    return new Set(list(data?.works).filter((work) => work?.confirmed !== false)
        .flatMap((work) => list(work.controls).filter((control) => control.action === 'resume')
            .map((control) => String(control.target || '').trim()))
        .filter((target) => target && !REUSABLE_TASK_IDS.has(target)));
}

/** The shared menu's items for a work's controls. Resume is its own button wherever one stands beside the menu;
 *  a pending Stop leaves only its hard escalation. */
export function workMenuActions(grouped, { cancelPending = false, resume = false } = {}) {
    if (cancelPending) return grouped.stop.length ? [ACTION_STOP_NOW] : [];
    return [grouped.wrapUp && ACTION_FINALIZE, grouped.hurry && ACTION_HURRY, grouped.pause && ACTION_PAUSE,
        resume && grouped.resume && ACTION_RESUME, grouped.stopEarlier.length && ACTION_STOP_EARLIER,
        grouped.stop.length && ACTION_STOP_NOW].filter(Boolean);
}

/**
 * What a card offers from its work: `resume` when the work's Resume control targets the card's own task, `menu` when
 * the work offers anything else, and `work` when the card is a confirmed work's current episode that is live or
 * offers something. The server's controls decide, live episode or not: a parent that finished while its helper
 * works keeps its tree's Stop. A finished card's Stop… is that offer, never its own pending Stop, so every sync of
 * the card keeps it usable (chat.js syncCancelRunButtonMutation); its own request says a refusal itself.
 */
export function cardControls(state, taskId) {
    const id = String(taskId || '');
    const work = state?.episodes?.get(id);
    const grouped = work && work.confirmed !== false ? workControls(work) : null;
    const resume = Boolean(grouped && grouped.resume === id);
    const menu = Boolean(grouped && workMenuActions(grouped).length);
    return { work: grouped && (work.episode?.live || resume || menu) ? work : null, resume, menu };
}

/**
 * The cards whose work offer (`cardControls` menu or resume) differs between two bodies' states. Their Stop… and
 * Resume are re-synced: a finished card whose work has left the body drops the Stop… an earlier body gave it.
 */
export function changedOffers(records, before, after) {
    const offer = (state, id) => { const controls = cardControls(state, id); return `${controls.menu}|${controls.resume}`; };
    return [...(records?.values?.() || [])].filter((record) => record?.root
        && offer(before, record.groupId) !== offer(after, record.groupId));
}

/** The items a card's Stop… menu offers now: its work's controls, else the card's own task controls. */
export function cardMenuActions(state, record, activity = null) {
    const { work } = cardControls(state, record?.groupId);
    const cancelPending = Boolean(record?.cancelPendingPolicy);
    return work ? workMenuActions(workControls(work), { cancelPending })
        : taskControlActions({ cancelPending, budgetPaused: activity?.phase === 'budget_paused' });
}

/**
 * A card's Stop… menu: its work's controls when it is a work's current episode (the card's own target stops through
 * its card path, `stop`, while the card still runs; a finished card's tree stops through the door itself), else the
 * card's own task controls, which never offer Resume.
 */
export function openCardControls(anchor, record, work, { activity = null, stop, afterAction = null } = {}) {
    const cancelPending = Boolean(record?.cancelPendingPolicy);
    if (!work) return openTaskControls(anchor, record?.groupId, { activity, cancelPending, stop });
    const own = (target) => target === String(record?.groupId || '') && !record?.finished;
    return openWorkControls(anchor, work, { cancelPending, afterAction,
        stopOne: (target, action) => (own(target) ? stop(action) : requestStop(target, action)) });
}

/**
 * Open the shared menu on a work's controls: each action reaches the existing request owner at its exact target;
 * Stop and Stop earlier attempt stop each of their targets in order and say one combined result. `stopOne` is the
 * caller's own stop path for a target (a loaded card keeps its optimistic chip); `canAct` re-checks the current
 * controls before any effect; `afterAction` re-reads the state once an effect settled.
 */
export function openWorkControls(anchor, work, { cancelPending = false, canAct = () => true, stopOne = requestStop,
    afterAction = null, withResume = false } = {}) {
    const grouped = workControls(work);
    const actions = workMenuActions(grouped, { cancelPending, resume: withResume });
    if (!actions.length) return null;
    const stopAll = (targets, action) => requestStopAll(targets, action, { stopOne });
    return openTaskControlMenu(anchor, {
        actions, pauseTarget: grouped.pause, busy: taskControlBusy(grouped.pause || work?.episode?.task_id), canAct, afterAction,
        onAction: (action) => (action === ACTION_HURRY ? hurryTaskAction(grouped.hurry)
            : action === ACTION_RESUME ? resumeTaskAction(grouped.resume)
                : action === ACTION_FINALIZE ? stopAll([grouped.wrapUp], ACTION_FINALIZE)
                    : stopAll(action === ACTION_STOP_EARLIER ? grouped.stopEarlier : grouped.stop, ACTION_STOP_NOW)),
    });
}
