/**
 * One work, one status in the browser (DESIGN "Room work status"): readers over the `/api/state` work projection
 * (ouroboros/gateway/work_status.py; shapes in ouroboros/gateway/work_contracts.py). The server groups live members into works, rolls
 * works into rooms, decides each work's controls with their exact targets and names each work's and room's mark
 * (`mark`: moving | still | pause | ring | none); this module says those facts in the product's short words (a work's
 * word, and beside it its line: the episode's own reason, then what still runs beside it), keeps what an incomplete
 * body did not list again (unconfirmed: a ring), and lays a work's controls on the shared task-control doors. It
 * derives no word, mark or control of its own. Five facts only this client knows stay
 * overlays, each with its mark: the socket closed (Reconnecting…, a ring), a room back on screen or reconnected still
 * reading what it missed (Updating…, a ring), the owner's own unconfirmed send (Sending, moving), the supervisor not
 * ready yet (Starting…, no mark) and the owner's own Stop press on a card (Cancelling… / Finalizing…, moving). Whether something waits for the owner (`owner`: the server's `turn`) is a separate fact;
 * status_mark.js draws the two together.
 */
import { paintCardModel } from './card_model.js';
import { fmt, tr, tx } from './i18n.js';
import { isTerminalTaskDetail, taskPresentation, taskTerminalPhase } from './log_events.js';
import { startingStatus } from './host_starting_status.js';
import { activeModelWaits } from './model_wait.js';
import {
    activityWaitPhase, CENSUS_MARKS, chipWord, desiredLiveCardPhase, observeChildMember, observeChildPhase, setLiveCardPhase,
} from './task_phase_chip.js';
import {
    ACTION_FINALIZE, ACTION_HURRY, ACTION_PAUSE, ACTION_RESUME, ACTION_STOP_EARLIER, ACTION_STOP_NOW, REUSABLE_TASK_IDS,
    TASK_CONTROL_TRIGGER_LABEL, createTaskResumeButton, createTaskCloseButton, closeExpiredOffer, deadlineExpiredWord,
    hurryTaskAction, isRootTaskRow, openTaskControlMenu, openTaskControls, requestStop, requestStopAll, resumePending, resumeTaskAction,
    taskControlActions, taskControlBusy,
} from './task_control_menu.js';

// A known outcome keeps its word's tone, its dot beside it (status_mark.js): Done and Done with warnings succeed (the
// word says the warnings), Stopped was someone's decision, the rest failed. An unfinished lifecycle (Finalizing…) is
// said beside it.
export const OUTCOME_TONES = { done: 'done', warn: 'done', cancelled: 'stopped', error: 'failed', timeout: 'failed',
    lifecycle_error: 'failed' };
const IDLE_ROOM = Object.freeze({ activity: 'idle', transition: 'none', call: '', turn: [], amber: false, mark: 'none',
    confirmed: true, works: [] });
const CALL_WORDS = { preparing: 'preparing', sending: 'sending', thinking: 'thinking' };
const MEMBER_MARKS = { working: 'moving', waiting: 'still', queued: 'still', paused: 'pause' }; // before members had a mark
const UNKNOWN_MEMBER = { activity: 'unknown', transition: 'none', mark: 'ring' };
const list = (value) => (Array.isArray(value) ? value : []);

// A run can be observed even when its supervisor's phase is unknown.
function displayedMember(member) {
    if (!member) return null;
    if (member.confirmed === true && Object.hasOwn(MEMBER_MARKS, member.activity)) return member;
    return { ...UNKNOWN_MEMBER, confirmed: false, tails: list(member.tails),
        ...(Number.isSafeInteger(member.queue_attempt) && member.queue_attempt >= 0 ? { queue_attempt: member.queue_attempt } : {}) };
}

/** Whether one member state independently evidences the member's own current activity: its own phase is
 *  confirmed, or its own external work is observed (`tails`) while the phase itself stays unknown. A nested
 *  card this state speaks for is a current fact, never a quiet purely historical row (DESIGN "Room work
 *  status": a nested card shows its own member and external work). */
export function memberActivityEvidenced(member) {
    return Boolean(member) && (member.confirmed === true || list(member.tails).length > 0);
}
// `question`: a task waits for the owner's answer (the work's own word says so). `open_question`: a question that is
// still unanswered while its work goes on or after it ended: the owner's turn, never the work's word.
const asked = (state) => list(state?.turn).some((item) => item?.kind === 'question');
const QUESTIONS = new Set(['question', 'open_question']);

/** The state one chat instance shows before its first body: an idle room and no works. */
export function emptyWorkState() {
    return { room: IDLE_ROOM, works: [], episodes: new Map(), members: new Set(), attempts: new Map(), complete: false,
        memberStates: null, main: null, tiers: null, leads: new Map(), processings: new Map(), questions: new Map() };
}

// One question's key across the turn, the census rows and the work list: its task and its quiz.
export const questionKey = (item) => `${item.task_id}:${item.quiz_id || ''}`;

// What an incomplete body did not list again: still shown, never as confirmed work and never with controls or a line;
// its mark is the ring of unconfirmed state.
const unconfirmedWork = (work) => ({ ...work, activity: 'unknown', transition: 'none', call: '', because: [], tails: [],
    last_resume: undefined, live: false, turn: [], controls: [], confirmed: false, mark: 'ring' });

/**
 * One body applied to an instance's work state. `room` and `works` are this chat's (`works[].room`, keyed as the
 * server keys `rooms`); `episodes` maps every work's current episode task id to its work in EVERY room (a Main
 * portal's work runs in its Project) and `members` holds every member id; `attempts` is each census row's attempt,
 * so a control opened for one attempt of an episode is not pressed on the next. `main` is Main's route as the body
 * says it (the last one said, when this body does not) and `leads` the model fact of each activity row that
 * publishes one, for a direct turn's card (CLIENT_CONTRACT §5). `questions` holds the current-question row of each
 * `open_question` in this room's turn (by `task_id:quiz_id`), from the same body, else the row an earlier body carried
 * while the turn still lists it: a question's row says its own words for exactly as long as its item makes the row.
 * Only a complete supervisor-ready
 * body read over an open socket proves absence; an incomplete one keeps the works it omitted, unconfirmed. A body
 * without the projection, or a failed read (null), keeps the room but cannot confirm a member's last phase once a body
 * spoke the per-member contract, an empty map included (`memberReadFailed`: no live frame outranks that).
 */
export function reconcileWorkState(prior, data, chatId, connected = true, requestedAt = Infinity) {
    const before = prior || emptyWorkState();
    const current = connected && data?.supervisor_ready !== false;
    const rows = list(data?.active_chat_activities).filter((row) => row?.activity_id);
    const attempts = new Map();
    if (current) for (const row of rows) {
        if (Number.isInteger(row.task_attempt)) attempts.set(String(row.activity_id), row.task_attempt);
    }
    if (!Array.isArray(data?.works)) {
        const seen = attempts.size || before.attempts.size ? { ...before, attempts } : before;
        return before.memberStates ? { ...seen, memberStates: new Map(), memberReadAt: requestedAt, memberReadFailed: true } : seen;
    }
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
    // null until a body speaks the per-member contract. An empty map thereafter is a gap in that contract.
    let memberStates = before.memberStates ? new Map() : null;
    for (const work of all) {
        if (work.episode?.task_id) episodes.set(String(work.episode.task_id), work);
        for (const id of list(work.members?.ids)) members.add(String(id));
        if (!work.members || !Object.hasOwn(work.members, 'states')) continue;
        memberStates ??= new Map();
        const states = work.members.states;
        if (!states || typeof states !== 'object' || Array.isArray(states)) continue;
        for (const [id, member] of Object.entries(states)) {
            members.add(id);
            if (!current) continue;
            memberStates.set(id, member);
            if (!attempts.has(id) && Number.isInteger(member?.queue_attempt)) attempts.set(id, member.queue_attempt);
        }
    }
    const leads = new Map(rows.filter((row) => row.lead && typeof row.lead === 'object')
        .map((row) => [String(row.activity_id), row.lead]));
    const processings = new Map(rows.filter((row) => row.processing && typeof row.processing === 'object')
        .map((row) => [String(row.activity_id), row.processing]));
    const tiers = data.processing_tiers && typeof data.processing_tiers === 'object' ? data.processing_tiers : before.tiers || null;
    const main = data.main_route && typeof data.main_route === 'object' ? data.main_route : before.main || null;
    const carried = new Map(list(data.current_questions).filter((row) => row?.task_id).map((row) => [questionKey(row), row]));
    const questions = new Map();
    for (const item of list(room.turn)) {
        if (item?.kind !== 'open_question' || !item.task_id) continue;
        const said = carried.get(questionKey(item)) || before.questions?.get(questionKey(item));
        if (said) questions.set(questionKey(item), said);
    }
    return { room, works: [...listed, ...kept], episodes, members, memberStates, memberReadAt: requestedAt,
        memberReadFailed: false, attempts, complete, main, tiers, leads, processings, questions };
}

/** Whether the server says a parked state's hard deadline has passed: its own boolean or reason, never a clock here.
 *  The word is the status; Close is a separate offer (`workCloseOffer`), so a state may say this and offer no action. */
export function deadlineExpiredState(state) {
    return state?.deadline_expired === true || list(state?.because).some((reason) => reason?.code === 'deadline_expired');
}

/** Index every task id whose hard deadline one accepted `/api/state` body marks as passed, once per census
 *  (never per row) and separately from Close tokens: activity/attention flags, each confirmed displayed member
 *  state, and each confirmed live episode's work state. Never a clock here, and never inferred from Close or Stop. */
export function indexDeadlineExpired(census) {
    const expired = new Set();
    if (!census || typeof census !== 'object') return expired;
    const noteRow = (row) => {
        const id = String(row?.activity_id || row?.task_id || '');
        if (id && (row.deadline_expired === true || row.attention_reason === 'deadline_expired')) expired.add(id);
    };
    for (const row of list(census.active_chat_activities)) noteRow(row);
    for (const row of list(census.main_attention)) noteRow(row);
    const seenEpisodes = new Set();
    const seenMembers = new Set();
    for (const work of list(census.works)) {
        if (!work || typeof work !== 'object') continue;
        if (work.confirmed !== false && work.episode?.live) {
            const epId = String(work.episode.task_id || '');
            if (epId && !seenEpisodes.has(epId)) {
                seenEpisodes.add(epId);
                if (deadlineExpiredState(work)) expired.add(epId);
            }
        }
        const states = work.members?.states;
        if (!states || typeof states !== 'object' || Array.isArray(states)) continue;
        for (const [id, member] of Object.entries(states)) {
            if (!member || seenMembers.has(id)) continue;
            seenMembers.add(id);
            if (deadlineExpiredState(displayedMember(member))) expired.add(id);
        }
    }
    return expired;
}

/** Whether one `/api/state` body names a task's hard deadline as passed, by the task's own exact identity: its census
 *  activity, its owner-action card, its member state or its confirmed work, each saying the server's `deadline_expired`
 *  boolean or reason. Never a clock here, and never read from the presence or absence of a Close offer or a Stop. */
export function deadlineExpiredIn(data, taskId) {
    const id = String(taskId || '');
    if (!id || !data || typeof data !== 'object') return false;
    const marked = (row) => String(row?.activity_id || row?.task_id || '') === id
        && (row.deadline_expired === true || row.attention_reason === 'deadline_expired');
    return list(data.active_chat_activities).some(marked) || list(data.main_attention).some(marked)
        || deadlineExpiredState(memberState(data, id)) || deadlineExpiredState(workOfEpisode(data, id));
}

/** A work's or a room's one short word, from its activity, transition and exact call (the cards' chip words). */
export function activityWord(state = {}) {
    if (deadlineExpiredState(state)) return deadlineExpiredWord();
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
        ? tr('work.because.earlier_attempt_finishing', 'Continue starts when the subagent of the earlier attempt finishes')
        : fmt('Continue starts when {count} subagents of the earlier attempt finish', { count: n, n })),
    earlier_runs_finishing: (n) => (n === 1
        ? tr('work.because.earlier_runs_finishing', 'Continue starts when the external run of the earlier attempt finishes')
        : fmt('Continue starts when {count} external runs of the earlier attempt finish', { count: n, n })),
    helpers_running: (n) => (n === 1 ? tr('work.because.helpers_running', '1 subagent still working')
        : fmt('{count} subagents still working', { count: n, n })),
    delegated_running: (n) => (n === 1 ? tr('work.because.delegated_running', '1 external run still working')
        : fmt('{count} external runs still working', { count: n, n })),
    external_runs_open: (n) => (n === 1 ? tr('work.because.external_runs_open', '1 external run is open')
        : fmt('{count} external runs are open', { count: n, n })),
    external_runs_unconfirmed: (n) => (n === 1 ? tr('work.because.external_runs_unconfirmed', '1 external run may still be running')
        : fmt('{count} external runs may still be running', { count: n, n })),
    awaited_helpers_mind: (n) => (n === 1
        ? tr('work.because.awaited_helpers_mind', '1 subagent it waits for is paused until Ouroboros resumes it')
        : fmt('{count} subagents it waits for are paused until Ouroboros resumes them', { count: n, n })),
    awaited_helpers_owner: (n) => (n === 1
        ? tr('work.because.awaited_helpers_owner', '1 subagent it waits for is paused until you resume it')
        : fmt('{count} subagents it waits for are paused until you resume them', { count: n, n })),
    awaited_helpers_paused: (n) => (n === 1 ? tr('work.because.awaited_helpers_paused', '1 subagent it waits for is paused')
        : fmt('{count} subagents it waits for are paused', { count: n, n })),
};

// What a sleep waits for, one clause per kind the server named (gateway/work_status.py SLEEP_CLAUSES; the web fixture
// pins both), joined by "or" in its order, its time last: a name stays as written, a count stays a parameter and a
// chosen time reads in the viewer's clock.
const SLEEP_CLAUSES = {
    task: ({ name }) => fmt('{name} finishes', { name }),
    task_or_writes: ({ name }) => fmt('{name} finishes or writes', { name }),
    tasks: ({ count: n }) => (n === 1 ? tr('work.sleep.task', 'the task it chose finishes')
        : fmt('one of {count} tasks finishes', { count: n, n })),
    all_tasks: ({ count: n }) => fmt('all {count} tasks it chose finish', { count: n, n }),
    runs: ({ count: n }) => (n === 1 ? tr('work.sleep.run', 'its external run finishes')
        : fmt('one of {count} external runs finishes', { count: n, n })),
    service: () => tr('work.sleep.service', 'a service it started stops'),
    reviewer: () => tr('work.sleep.reviewer', 'a reviewer answers'),
    sender: ({ name }) => fmt('{name} writes', { name }),
    mail: () => tr('work.sleep.mail', 'a message arrives'),
    time: ({ at }) => clock(at),
};

/** A sleep's clauses in one sentence; '' when one is a kind this client does not know or a time it cannot read. */
function sleepText(wake) {
    const said = wake.map((clause) => (Object.hasOwn(SLEEP_CLAUSES, clause?.kind) ? SLEEP_CLAUSES[clause.kind](clause) : ''));
    if (!said.length || said.some((clause) => !clause)) return '';
    return fmt('Sleeping until {what}', { what: said.reduce((a, b) => fmt('{a} or {b}', { a, b })) });
}

// Helpers and external runs still working side by side are said in one sentence, so the line never says it twice.
function tailWords(tails) {
    const counted = (code) => tails.find((tail) => tail?.code === code && Number.isInteger(tail.count));
    const helpers = counted('helpers_running');
    const runs = counted('delegated_running');
    if (!helpers || !runs) return tails.map(reasonText);
    const both = fmt('{subagents} and {runs} still working', {
        subagents: helpers.count === 1 ? tr('work.because.one_helper', '1 subagent')
            : fmt('{count} subagents', { count: helpers.count, n: helpers.count }),
        runs: runs.count === 1 ? tr('work.because.one_external_run', '1 external run')
            : fmt('{count} external runs', { count: runs.count, n: runs.count }),
    });
    return tails.flatMap((tail) => (tail === helpers ? [both] : tail === runs ? [] : [reasonText(tail)]));
}

/** One reason in the owner's words: the server's code read through the catalog, its sentence as the English (a
 *  code this client has never seen still reads its sentence). A count stays a parameter of its template; a counted
 *  code this client does not know keeps the server's sentence as it is; a sleep says what it waits for. An empty
 *  sentence (a Pause that declared nobody) says nothing, and the words a Pause recorded are its author's, whoever
 *  declared it: only the frame before them is translated. */
export function reasonText(reason) {
    if (!reason || typeof reason !== 'object') return '';
    const code = String(reason.code || '');
    const text = typeof reason.text === 'string' ? reason.text : code;
    if (!code || !text) return text;
    if (Number.isInteger(reason.count)) return COUNTED_REASONS[code] ? COUNTED_REASONS[code](reason.count) : text;
    if (reason.by?.text) return `${tr(`work.because.${code}`, text)}: ${String(reason.by.text)}`;
    if (code === 'technical_pause' && text !== 'After a technical problem') return tx(text);
    if (code === 'sleeping' && Array.isArray(reason.wake)) return sleepText(reason.wake) || text;
    // The work a queued one stands behind, by its census name (gateway/work_status.py QUEUED_BEHIND; fixture pins both).
    if (code === 'queued' && reason.behind?.name) return fmt('behind “{name}”', { name: String(reason.behind.name) });
    // A silence dated by the host reads with its time: the date is the fact (owner 09.11).
    if (code === 'no_progress' && clock(reason.since)) return fmt('No recorded progress since {time}', { time: clock(reason.since) });
    return tr(`work.because.${code}`, text);
}

/** For a queued work's details: the id of the work it stands behind and the lane the scheduler recorded, its Project
 *  or a folder by its path; '' for any other work or one without a recorded holder. */
export function behindWords(work = {}) {
    const behind = work?.activity === 'queued' ? list(work.because)[0]?.behind : null;
    if (!behind?.task_id) return '';
    const lane = behind.lane === 'folder' && behind.folder ? fmt('the folder {path}', { path: String(behind.folder) })
        : tr('work.behind.project', 'its Project');
    return fmt('{id} keeps {lane}', { id: String(behind.task_id), lane });
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

/** A held work's latest refused Resume (`last_resume`, work_stop.last_resume on the server) as its line's last clause,
 *  dated and naming who asked as its door bound it; '' without one. History beside the reason, never a door or proof
 *  that what blocked it still does; the host's own refused start (`host`) is a waiting Continue's reason. */
export function resumeAttemptClause(attempt) {
    const time = attempt && typeof attempt === 'object' ? clock(attempt.at) : '';
    if (!time) return '';
    const actor = attempt.by?.actor;
    return actor === 'ouroboros' ? fmt('Resume by Ouroboros refused at {time}', { time })
        : actor === 'host' ? fmt('Start refused at {time}', { time }) : fmt('Resume refused at {time}', { time });
}

/** The same attempt with the door's own words after its clause (the card's details, Main's owner-action message):
 *  a host sentence, read in the install's language by its exact text (`tx`). */
export function resumeAttemptWords(attempt) {
    const clause = resumeAttemptClause(attempt);
    const text = clause ? tx(attempt.text) : '';
    return text ? `${clause}: ${text}` : clause;
}

/**
 * A work's line, said beside its word (the card's quieter line, the work-list row's, the sidebar's name after the word):
 * the current episode's own sentence (the owner's Pause with the time it was pressed), then what still runs beside
 * it, one sentence per tail, then its latest refused Resume with its time, joined " · ". The line never repeats
 * `word`: an answer wait's own words are compared by their code, in any language.
 */
// What a Pause says of itself once its deadline has passed: who or what stopped it, dated. The instruction to wait
// for a Resume (or a held word) is no longer true there, so it is not carried beside `Deadline expired`.
const PAUSE_ATTRIBUTION = new Set(['paused_by_owner', 'paused_by_mind', 'paused_unattributed', 'saved_on_stop',
    'after_restart', 'after_panic', 'spending_limit', 'technical_pause']);

export function workLine(work = {}, word = activityWord(work)) {
    const reasons = list(work.because).filter((reason) => reason?.code !== 'deadline_expired');
    const first = deadlineExpiredState(work) ? reasons.find((reason) => PAUSE_ATTRIBUTION.has(reason?.code)) : reasons[0];
    const said = first?.code === 'owner_answer' && word === chipWord('ownerWait') ? '' : reasonText(first);
    const sentence = said === word ? '' : said;
    const pressed = sentence && first.code === 'paused_by_owner' ? clock(first.since) : '';
    return [sentence, pressed, ...tailWords(list(work.tails)), resumeAttemptClause(work.last_resume)]
        .filter(Boolean).join(' · ');
}

/** The first of a room's works that says the room's own word, else null: a work Pausing beside a running one never
 *  lends its reason to the room's Working. */
export function roomSpeaker(room, works = [], word = activityWord(room)) {
    const byId = new Map(list(works).map((work) => [work?.work_id, work]));
    return list(room?.works).map((id) => byId.get(id)).find((work) => work && activityWord(work) === word) || null;
}

/** What a `turn` holds for the owner ('question' | 'action' | 'unanswered' | 'delivery'), '' when nothing: a
 *  question a task waits on, else an action only the owner takes, else a question still unanswered while its work
 *  goes on or after it ended, or only answers whose delivery is unknown. A room's note and a card chip's name both
 *  say it by this one rule. */
export const turnAmber = (turn) => {
    const items = list(turn);
    if (!items.length) return '';
    if (items.some((item) => item?.kind === 'question')) return 'question';
    const open = items.filter((item) => item?.kind === 'open_question');
    if (items.some((item) => item?.kind === 'action') || !open.length) return 'action';
    return open.every((item) => item.code === 'delivery_unknown') ? 'delivery' : 'unanswered';
};

/** The addressed question whose own words can say a `turn`, by turnAmber's rule: the first one a task waits on, else
 *  the first still unanswered; null where what waits is an action or only an answer whose delivery is unknown (their
 *  sentence says it) and for a bare wait without a quiz. The order is the server's; nothing here ranks questions. */
export const turnQuestion = (turn) => {
    const kind = turnAmber(turn);
    const named = (item) => Boolean(item?.task_id && item.quiz_id);
    if (kind === 'question') return list(turn).find((item) => item?.kind === 'question' && named(item)) || null;
    if (kind !== 'unanswered') return null;
    return list(turn).find((item) => item?.kind === 'open_question' && item.code !== 'delivery_unknown' && named(item)) || null;
};

/** What waits for the owner in a room (turnAmber), '' when nothing does or its own word says so. */
export const roomAmber = (room) => (!room?.amber || (room.activity === 'waiting' && asked(room)) ? ''
    : turnAmber(room.turn) || 'action');

/** The amber mark said in words, for the room's accessible name: the colour is never the only way it is told. An
 *  unanswered question nobody waits on is never said as a wait. */
export const amberNote = (kind) => (kind === 'question' ? tr('work.amber.question', 'A question waits for your answer')
    : kind === 'unanswered' ? tr('work.amber.unanswered', 'An unanswered question is here')
        : kind === 'delivery' ? tr('work.amber.delivery', 'An answer here was recorded; its delivery is unknown')
            : kind === 'action' ? tr('work.amber.action', 'Something here waits for you') : '');

/**
 * The room's status in its bar: the server's room in short words, with the client's own facts on top. Tone is the
 * word's ink (live, queued, wait for the owner's own question, held, unknown, starting, offline, online); `mark` the
 * work's shape after it (the server's room mark under this module's four overlays; the transcript's typing bubble
 * shows while it is `moving`, whatever waits for the owner); `owner` whether something waits for the owner (the room's
 * turn: status_mark.js draws it amber, the word keeps its ink); `amber` what waits (roomAmber's kinds), said in
 * words where the word does not already say it. `stopping` is the owner's own Stop press on every working work of the
 * room ('finalize' | 'immediate'); until the host is `ready`, an idle room says its start in the words of the host's
 * `boot` phases (host_starting_status.js). A room still preparing names as its `step` the step its speaking work
 * stands at (a direct turn's start step), said beside the word and dropped before it on a short bar, so nothing in
 * the bar reads as a model call before one is sent. `serverAnswers` (ws.serverAnswersWhileDown) is said the same way
 * beside `Reconnecting…`: an observation, never a cause.
 */
export function roomStatusView(state, {
    connected = true, serverAnswers = false, catchingUp = false, ready = true, boot = null, sending = false, stopping = '',
} = {}) {
    const view = (kind, text, tone, mark, extra = {}) => ({ kind, text, tone, mark, owner: false, amber: '', ...extra });
    if (!connected) {
        return view('offline', 'Reconnecting…', 'offline', 'ring', serverAnswers ? {
            step: tr('work.server_answers.step', 'server answers'),
            title: tr('work.server_answers', 'The server answers, but live updates cannot connect.') } : {});
    }
    const room = state?.room || IDLE_ROOM;
    const mark = String(room.mark || 'none');
    const turn = { owner: list(room.turn).length > 0, amber: roomAmber(room) };
    // Catch-up is visible without hiding an unanswered question the room already knows about.
    if (catchingUp) return view('starting', 'Updating…', 'updating', 'ring', turn);
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
    // A room at rest keeps the owner's turn: a question of an ended work still waits there, while the host starts too.
    default: return ready ? view('online', 'Online', 'online', mark, turn)
        : { ...view('starting', 'Starting…', 'starting', 'none', turn), ...startingStatus(boot) };
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
 * whether something in the work waits for the owner (its turn: status_mark.js draws it amber; a Pause by the
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
    // Only a settled lifecycle is worded beside Finalizing…; any other (`running`) is not known.
    const outcome = work.transition === 'finalizing' && isTerminalTaskDetail({ status: work.outcome })
        ? taskPresentation(taskTerminalPhase({ status: work.outcome })) : null;
    const word = outcome ? outcome.headline : activityWord(work);
    const secondary = [outcome ? chipWord('finalizing') : '', workLine(work, word)].filter(Boolean).join(' · ');
    const pausing = work.activity === 'working' && work.transition === 'pausing';
    const tone = outcome ? OUTCOME_TONES[outcome.phase] || 'live'
        : !work.confirmed || work.activity === 'unknown' ? 'unknown'
            : work.activity === 'waiting' && asked(work) ? 'wait'
                : work.activity === 'working' && !pausing ? 'live' : work.activity === 'queued' ? 'queued' : 'held';
    // The work's own focus line: what its author says it is doing, under the state (update_focus), led by the time it
    // wrote it in the viewer's clock (a line without its time says none).
    const text = String(work.focus?.text || '').trim();
    const focus = text ? [clock(work.focus.at), text].filter(Boolean).join(' · ') : '';
    return { word, secondary, tone, mark: String(work.mark || 'none'), owner, outcome: outcome?.phase || '', focus };
}

/** The question a row's View question goes to: the first question in the work's turn, by its exact quiz id. */
export function workQuestion(work) {
    const question = list(work?.turn).find((item) => QUESTIONS.has(item?.kind) && item.quiz_id && item.task_id);
    return question ? { taskId: String(question.task_id), quizId: String(question.quiz_id) } : null;
}

/**
 * What the card of a work's current episode says (one work, one status): the work's word alone in the chip, and its
 * line as the chip's quieter second fact (the episode's reason, then what still runs beside it: reviews a Pause lets
 * finish, helpers, external runs; then its latest refused Resume, dated, whose words are the details' `note`).
 * Plain running work with nothing to say beside it defers to its census row if present. Unconfirmed work and ordinary
 * cards in finalization keep their own words. A portal's current finalizing projection can also supply the quiet line
 * when its room has no census row. The card's known outcome is separate (task_phase_chip.js).
 */
export function cardWorkView(work, { census = true, finalizing = false } = {}) {
    if (!work || !work.confirmed || !work.episode?.live || work.activity === 'unknown'
            || (!finalizing && work.transition === 'finalizing')) return null;
    return projectedCardView(work, false, census);
}

function projectedCardView(work, member, census = true) {
    const word = activityWord(work);
    const line = workLine(work, word);
    if (!member && census && work.activity === 'working' && work.transition === 'none' && !line) return null;
    const running = work.activity === 'working';
    const finalizing = work.transition === 'finalizing';
    const waiting = !running || (work.transition !== 'none' && !finalizing);
    const classes = list(work.turn).length ? 'warn' : finalizing ? 'working finalizing'
        : running && waiting ? 'working waiting' : work.activity;
    // The latest refused Resume in the door's own words explains the word in the card's details (its line keeps the
    // dated clause alone); a queued work's details give the id and lane of the work it stands behind.
    const note = resumeAttemptWords(work.last_resume) || behindWords(work);
    return {
        phase: running ? 'working' : work.activity, text: word, secondary: line, classes,
        tone: waiting ? 'neutral' : 'live', waiting, ...(finalizing ? { finalizing } : {}),
        ...(note ? { note } : {}),
        ...(member && Number.isSafeInteger(work.queue_attempt) && work.queue_attempt >= 0 ? { queue_attempt: work.queue_attempt } : {}),
    };
}

// What waits in the work, said as the room says it: the chip's name names that, never only "something".
function applyCardWorkLook(record, subject, view) {
    const note = amberNote(turnAmber(subject?.turn));
    const mark = subject?.mark || (record.isSubagent && subject ? (subject.live ? 'moving' : MEMBER_MARKS[subject.activity]) : 'none');
    const look = subject ? { mark, owner: list(subject.turn).length > 0, ...(note ? { note } : {}) } : null;
    const changed = JSON.stringify([view, look]) !== JSON.stringify([record.workView || null, record.workLook || null]);
    record.workView = view;
    record.workLook = look;
    return changed;
}

// A portal's caller has already matched this episode to its bound room in the latest state body.
export function applyPortalWorkView(record, work, census) {
    if (record.cardModel) paintCardModel(record, null);
    return applyCardWorkLook(record, work, cardWorkView(work, { census, finalizing: true }));
}

/**
 * Each room card's view from one state: the current episode speaks its status and turn, and its `lead` is painted
 * from the episode or the activity row. A portal's cross-room phase is applied separately, with no model fact.
 * A nested card speaks for its own member. A live frame accepted since a read began (`livePhaseAt`) outranks the
 * read's silence, not an explicit member state, a recorded hold or a failed read.
 */
export function applyCardWorkViews(records, state) {
    let changed = false;
    for (const record of records?.values?.() || []) {
        const id = String(record.groupId || '');
        const child = record.isSubagent;
        if (record.root?.dataset?.portal === '1') {
            if (record.cardModel) paintCardModel(record, null);
            continue;  // its cross-room phase is read by work_portal.js
        }
        if (child && record.cardModel) paintCardModel(record, null);
        const silent = state?.memberStates && !state.memberReadFailed && !state.memberStates.has(id);
        if (child && (record.finished || (silent && record.livePhaseAt >= state.memberReadAt))) continue;
        const work = child ? null : state?.episodes?.get(id);
        const lead = child ? null : work?.lead || state?.leads?.get(id) || null;
        // An unconfirmed work offers nothing (cardControls): its fact stays, as plain text. So does the fact of a work
        // an incomplete body did not list again: only a complete body proves it gone.
        if (lead) {
            paintCardModel(record, { lead, target: work && work.confirmed !== false ? workControls(work).model : '',
                main: state.main, queued: work?.activity === 'queued', since: clock(lead.since),
                processing: work?.processing || state?.processings?.get(id) || null, tiers: state.tiers || null });
        } else if (record.cardModel && !state?.complete) paintCardModel(record, { ...record.cardModel.view, target: '' });
        else if (!child) paintCardModel(record, null);
        let subject = work?.episode?.live ? work : null;
        if (child && state?.memberStates) {
            subject = displayedMember(state.memberStates.get(id)) || UNKNOWN_MEMBER;
            if (subject.confirmed && record.censusPhase === 'unknown') record.parkedPhase = record.censusPhase = '';
            record.livePhaseAt = 0;  // the member's state now speaks for the card
        }
        const view = child && subject ? projectedCardView(subject, true) : cardWorkView(work);
        // Before the look stores this view: a requeue is read against the view it replaces (task_phase_chip.js).
        const startChanged = child && observeChildMember(record, view);
        changed = applyCardWorkLook(record, subject, view) || changed;
        if (child) {
            const desired = desiredLiveCardPhase(record);
            if (setLiveCardPhase(record, desired.phase, desired.text, desired.className, desired.secondary) || startChanged) {
                changed = true;
                record.paintMeta?.();  // its line's clock reads the chip and attempt it moved (chat_activity.js clockLabel)
            }
        }
    }
    return changed;
}

/** A card a child's own frames made before its lineage arrived becomes that child's card: the look it had as a root
 *  leaves it, and its live frame, when no read began after that frame, is the child's phase. Then the member's state
 *  applies as on any child card (applyCardWorkViews): an explicit state or a failed read decides, an older silence not. */
export function promoteChildCard(record, state) {
    record.workView = record.workLook = null;
    if (record.livePhaseAt && !(record.livePhaseAt < state?.memberReadAt)) observeChildPhase(record, 'working');
    return applyCardWorkViews([record], state);
}

/**
 * A work's controls, one press each: Stop (every episode root that still owns live work, in the server's order),
 * Stop earlier attempt (each earlier root that does), the current episode's Pause, Resume, Hurry and Wrap up, and
 * the task whose model may be changed (`model`: the card's model fact is its door, card_model.js).
 * Unknown actions are not offered.
 */
export function workControls(work) {
    const out = { stop: [], stopEarlier: [], pause: '', resume: '', hurry: '', wrapUp: '', model: '' };
    for (const control of list(work?.controls)) {
        // A reusable slot (`active`) hosts many cycles: no control reaches one of them by that id.
        const target = String(control?.target || '').trim();
        if (!target || REUSABLE_TASK_IDS.has(target)) continue;
        if (control.action === 'resume' && control.scope === 'single') continue;
        const many = { stop: out.stop, stop_earlier: out.stopEarlier }[control.action];
        if (many) { if (!many.includes(target)) many.push(target); continue; }
        const one = { pause: 'pause', resume: 'resume', hurry: 'hurry', wrap_up: 'wrapUp', model: 'model' }[control.action];
        if (one && !out[one]) out[one] = target;
    }
    return out;
}

export function workCloseOffer(work, taskId) {
    return list(work?.controls).map(offer => closeExpiredOffer(offer, taskId)).find(Boolean) || '';
}

/** The confirmed work whose live current episode is `taskId` in one body, else null. */
export function workOfEpisode(data, taskId) {
    const id = String(taskId || '');
    return list(data?.works).find((work) => work?.confirmed !== false && work.episode?.live
        && String(work.episode.task_id || '') === id) || null;
}

/** Whether the confirmed work holding `taskId` as a member, in any room, has it paused and not past its own hard
 *  deadline (its own server state): what such a task is sent waits for its Resume. */
export function memberPaused(state, taskId) {
    const id = String(taskId || '');
    if (!id || state?.memberReadFailed || (state?.memberStates instanceof Map && !state.memberStates.has(id))) return false;
    return [...(state?.episodes?.values?.() || [])].some((work) => {
        const member = work?.confirmed !== false ? work.members?.states?.[id] : null;
        return member?.confirmed !== false && member?.activity === 'paused' && !deadlineExpiredState(member);
    });
}

/** Whether the current reconciled per-member state positively confirms `taskId`'s hard deadline expired.
 *  Reads only the authoritative current `memberStates` entry and its own `confirmed === true` flag, so an
 *  unrelated unknown sibling making the aggregate work unconfirmed never erases a confirmed member's expiry,
 *  and a failed, disconnected or partial read never carries a stale expiry forward. */
export function memberDeadlineExpired(state, taskId) {
    const id = String(taskId || '');
    if (!id || state?.memberReadFailed || !(state?.memberStates instanceof Map)) return false;
    const member = state.memberStates.get(id);
    return member?.confirmed === true && deadlineExpiredState(member);
}

/** The server's state for an individual member, without interpreting queue carriers in the browser. */
export function memberState(data, taskId) {
    const id = String(taskId || '');
    if (data?.memberStates instanceof Map) return displayedMember(data.memberStates.get(id));
    const row = list(data?.works).find(work => work.members?.states?.[id]);
    return displayedMember(row?.members.states[id]);
}

function isExplicitChildQueueRow(q, ownId) {
    const id = String(ownId || '').trim();
    const t = q?.task;
    if (!id || REUSABLE_TASK_IDS.has(id) || !t || typeof t !== 'object' || t._presence_run) return false;
    const rowId = String(q.id ?? '').trim();
    const taskId = String(t.id ?? '').trim();
    if ((rowId || taskId) !== id || (rowId && rowId !== id) || (taskId && taskId !== id)) return false;
    if (isRootTaskRow(t, id)) return false;
    const field = (key) => String(t[key] ?? t.metadata?.[key] ?? '').trim();
    const parentId = field('parent_task_id');
    const rootId = field('root_task_id');
    const role = field('delegation_role').toLowerCase();
    if (rootId && rootId === id) return false;
    return (Boolean(parentId) && parentId !== id) || (role === 'subagent' && Boolean(rootId) && rootId !== id);
}

/** Matched safe-integer attempt when an explicit child queue row joins its own confirmed paused member in `/api/state`
 *  (outside root-only census; real producers `queue_publication._capture`, `gateway/state._raw_work_facts`), else null. */
export function confirmedPausedHelperAttempt(q, census, ownId) {
    const id = String(ownId || '').trim();
    if (!id || !census || census.supervisor_ready === false || !isExplicitChildQueueRow(q, id)) return null;
    if (list(census.active_chat_activities).some((entry) => String(entry?.activity_id || '') === id)) return null;
    if (workOfEpisode(census, id)) return null;
    const member = list(census.works).find((entry) => entry?.members?.states && Object.hasOwn(entry.members.states, id))
        ?.members.states[id];
    if (member?.confirmed !== true || member.activity !== 'paused') return null;
    const qAttempt = q.attempt;
    const mAttempt = member.queue_attempt;
    return Number.isSafeInteger(qAttempt) && qAttempt >= 0
        && Number.isSafeInteger(mAttempt) && mAttempt >= 0
        && qAttempt === mAttempt ? qAttempt : null;
}

/** Recheck the latest accepted queue + census for one exact confirmed paused helper row and return its attempt, or null. */
export function confirmedHelperAttemptInQueue(queue, census, taskId) {
    const id = String(taskId || '').trim();
    if (!id || !Array.isArray(queue?.running) || !Array.isArray(queue?.pending)) return null;
    const rows = [...queue.running.filter((q) => !q?.task?._presence_run), ...queue.pending]
        .filter((q) => String(q?.id || q?.task?.id || '').trim() === id);
    return rows.length === 1 ? confirmedPausedHelperAttempt(rows[0], census, id) : null;
}

/** Every task a confirmed work's Resume control names in one body: where Activity may offer Resume. */
export function resumeTargets(data) {
    return new Set(list(data?.works).filter((work) => work?.confirmed !== false)
        .flatMap((work) => list(work.controls).filter((control) => control.action === 'resume')
            .map((control) => String(control.target || '').trim()))
        .filter((target) => target && !REUSABLE_TASK_IDS.has(target)));
}

/** Index every exact Close offer in one accepted census by target task id, once per census (never per row).
 *  Precedence: confirmed live episode controls, confirmed member close_expired, census activity close_expired,
 *  then any remaining confirmed work Close control. */
export function indexCloseOffers(census) {
    const offers = new Map();
    if (!census || typeof census !== 'object') return offers;
    const put = (offer, target) => {
        const id = String(target || '').trim();
        if (!id || offers.has(id)) return;
        const token = closeExpiredOffer(offer, id);
        if (token) offers.set(id, token);
    };
    const works = list(census.works);
    const seenEpisodes = new Set();
    const fallbackControls = [];
    for (const work of works) {
        if (!work || work.confirmed === false) continue;
        const epId = work.episode?.live ? String(work.episode.task_id || '') : '';
        const firstEp = Boolean(epId) && !seenEpisodes.has(epId);
        if (firstEp) seenEpisodes.add(epId);
        const controls = work.controls;
        if (!Array.isArray(controls)) continue;
        for (const control of controls) {
            if (control?.action !== 'close') continue;
            if (firstEp && control.target === epId) put(control, epId);
            else fallbackControls.push(control);
        }
    }
    const seenMembers = new Set();
    for (const work of works) {
        const states = work?.members?.states;
        if (!states || typeof states !== 'object' || Array.isArray(states)) continue;
        for (const [id, member] of Object.entries(states)) {
            if (!member || seenMembers.has(id)) continue;
            seenMembers.add(id);
            if (member.confirmed === true && Object.hasOwn(MEMBER_MARKS, member.activity)) put(member.close_expired, id);
        }
    }
    const seenActivities = new Set();
    for (const row of list(census.active_chat_activities)) {
        const id = String(row?.activity_id || '');
        if (!id || seenActivities.has(id)) continue;
        seenActivities.add(id);
        put(row?.close_expired, id);
    }
    for (const control of fallbackControls) put(control, control.target);
    return offers;
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
 * the card keeps it usable; its own request says a refusal itself.
 */
export function cardControls(state, taskId, work = state?.episodes?.get(String(taskId || '')), roomMatched = false) {
    // A positively addressed portal can use the current projection's explicit offers despite a result gap, like
    // the room's work sheet. An omitted row has no offers, and its gap grants no phase or legacy authority.
    const grouped = work && (roomMatched || work.confirmed !== false) ? workControls(work) : null;
    const resume = Boolean(grouped && grouped.resume === String(taskId || ''));
    const menu = Boolean(grouped && workMenuActions(grouped).length);
    const close = grouped ? workCloseOffer(work, taskId) : '';
    return { work: grouped && (work.episode?.live || resume || menu || close) ? work : null, resume, menu,
        ...(close ? { close } : {}) };
}

/**
 * The cards whose work offer (`cardControls` menu or resume) differs between two bodies' states. Their Stop… and
 * Resume are re-synced: a finished card whose work has left the body drops the Stop… an earlier body gave it.
 */
export function changedOffers(records, before, after) {
    return [...(records?.values?.() || [])].filter((record) => record?.root
        && (cardControls(before, record.groupId).work !== cardControls(after, record.groupId).work
            || before.attempts?.get(record.groupId) !== after.attempts?.get(record.groupId)
            || memberState(before, record.groupId)?.close_expired !== memberState(after, record.groupId)?.close_expired));
}

// An ordinal from the state body owns projection and menu continuity, never the
// physical attempt that owns the card's outcome, words and counters.
export function noteControlQueue(record, ordinal) {
    const previous = record.controlQueueAttempt;
    if (!Number.isSafeInteger(ordinal) || (previous != null && ordinal <= previous)) return;
    if (previous != null) record.controlEpisode = {};
    record.controlQueueAttempt = ordinal;
}

/** The items a card's Stop… menu offers now: its work's controls, else the card's own task controls. */
export function cardMenuActions(state, record, activity = null, work = cardControls(state, record?.groupId).work, stopOnly = false) {
    const cancelPending = Boolean(record?.cancelPendingPolicy || stopOnly);
    if (work) return workMenuActions(workControls(work), { cancelPending });
    const phase = activity?.phase;
    const actions = taskControlActions({ cancelPending, budgetPaused: phase === 'budget_paused' });
    if (record?.root?.dataset?.portal !== '1') return actions;
    // A legacy portal has just the addressed census row. Use its own phase and explicit Presence capability;
    // no projected Resume, no inferred work or helper membership (gateway/work_status.py controls_for).
    const presence = activity?.kind === 'presence';
    const active = CENSUS_MARKS[phase] === 'moving' && phase !== 'budget_pausing';
    const running = active && !activityWaitPhase(activity) && !activity.project_admission_hold && !presence
        && !activeModelWaits(activity.model_waits, false, activity.task_attempt || 1).length;
    const pausable = (active || phase === 'queued' || phase === 'budget_pausing') && (!presence || activity.presence_pause === true);
    return actions.filter(action => action === ACTION_STOP_NOW || (action === ACTION_PAUSE ? pausable : running));
}

const actionTarget = (group, action) => ({ [ACTION_FINALIZE]: group.wrapUp, [ACTION_HURRY]: group.hurry,
    [ACTION_PAUSE]: group.pause, [ACTION_RESUME]: group.resume,
    [ACTION_STOP_EARLIER]: group.stopEarlier, [ACTION_STOP_NOW]: group.stop })[action];

const targetKey = (target) => JSON.stringify(Array.isArray(target) ? [...target].sort() : [target]);

/** The offered actions and their exact targets. Stop compares sets, but still sends in the opened order. */
export function workMenuTargets(grouped, options) {
    return new Map(grouped ? workMenuActions(grouped, options)
        .map(action => [action, targetKey(actionTarget(grouped, action))]) : []);
}

/** Recheck each opened item against the latest addressed controls; a change withdraws that item only. */
export function workMenuTargetGuard(grouped, options) {
    const opened = workMenuTargets(grouped, options);
    return (action, current, currentOptions = options) => {
        const next = workMenuTargets(current, currentOptions);
        for (const [name, target] of opened) if (next.get(name) !== target) opened.delete(name);
        return opened.has(action);
    };
}

// Opening a menu takes no authority from a later attempt or placement. The same target still offered by a new body
// works without reopening. The shared menu checks this even for Pause, which dispatches outside its onAction hook.
export function cardControlGuard(record, read, opened = read(record)) {
    const targetOf = (offer, action) => {
        if (action === ACTION_RESUME) return offer.resume ? record.groupId : '';
        if ((!offer.menu && (offer.work || !offer.legacy)) || !cardMenuActions(null, record, offer.activity, offer.work, offer.stopOnly).includes(action)) return '';
        return offer.work ? actionTarget(workControls(offer.work), action) : record.groupId;
    };
    const targets = new Map([ACTION_RESUME, ACTION_FINALIZE, ACTION_HURRY, ACTION_PAUSE, ACTION_STOP_EARLIER, ACTION_STOP_NOW]
        .map(action => [action, targetOf(opened, action)]).filter(([, target]) => target));
    // Only the still-open offer owns its targets. A disappeared item cannot
    // regain authority on this menu, but a continuous Stop stays available.
    const observe = next => {
        if (opened.episode !== next.episode || opened.scope !== next.scope) targets.clear();
        for (const [action, target] of targets) {
            if (targetKey(target) !== targetKey(targetOf(next, action))) targets.delete(action);
        }
    };
    record.observeControlMenu = observe;
    return (action) => {
        if (record.observeControlMenu !== observe || record.root?.isConnected === false || record.root?.dataset?.episodeOf) return '';
        observe(read(record));
        return targets.get(action) || '';
    };
}

/**
 * A card's Stop… menu: its work's controls when it is a work's current episode (the card's own target stops through
 * its card path, `stop`, while the card still runs; a finished card's tree stops through the door itself), else the
 * card's own task controls, which never offer Resume.
 */
function openCardControls(anchor, record, read, { stop, afterAction }) {
    const opened = read(record);
    const { work, activity, legacy, stopOnly } = opened;
    const cancelPending = Boolean(record?.cancelPendingPolicy);
    const targetNow = cardControlGuard(record, read, opened);
    const canAct = (action) => anchor.isConnected && !anchor.disabled && Boolean(targetNow(action));
    // The same cascade Stop can outlive its author's card. Choose its dispatcher at the press, not at opening.
    const stopOne = (target, action) => target === record.groupId && !record.finished ? stop(action) : requestStop(target, action);
    if (!work) return legacy ? openTaskControls(anchor, record.groupId, { canAct, afterAction,
        stop: action => stopOne(record.groupId, action), actions: cardMenuActions(null, record, activity, work, stopOnly) }) : null;
    return openWorkControls(anchor, work, { cancelPending, canAct, afterAction, stopOne });
}

let cardNameSeq = 0;

/** Reconcile a card's Resume and Stop… doors in place from the same current offer each press will read. */
export function syncCardControls(record, { read, actionsOf, stop, afterAction }) {
    if (!record?.root) return false;
    const canResume = () => read(record).resume;
    const offer = read(record);
    record.observeControlMenu?.(offer);
    const { resume, menu, legacy, close } = offer;
    let changed = false;
    if (record.closeRunBtn && record.closeRunBtn.offer !== close) {
        record.closeRunBtn.remove(); record.closeRunBtn = null; changed = true;
    }
    if (close && !record.closeRunBtn) {
        const actions = actionsOf(record);
        if (actions) {
            const titleEl = record.titleEl || record.root.querySelector?.('[data-live-title]') || null;
            const nameId = titleEl ? (titleEl.id ||= `chat-card-name-${cardNameSeq += 1}`) : '';
            record.closeRunBtn = createTaskCloseButton(record.groupId, {
                tokenNow: () => read(record).close === close ? close : '', onSettled: afterAction, nameId,
            });
            record.closeRunBtn.offer = close;
            actions.insertBefore(record.closeRunBtn, actions.querySelector('[data-cancel-run]'));
            changed = true;
        }
    }
    if (resume && !record.resumeRunBtn) {
        record.resumeRunBtn = createTaskResumeButton(record.groupId, { canResume, onSettled: afterAction });
        const actions = actionsOf(record);
        actions?.insertBefore(record.resumeRunBtn, actions.querySelector('[data-cancel-run]'));
        changed = true;
    } else if (!resume && record.resumeRunBtn) {
        record.resumeRunBtn.remove(); record.resumeRunBtn = null; changed = true;
    }
    const existing = record.root.querySelector('[data-cancel-run]');
    if (!menu && (offer.work || !legacy)) {
        if (!existing) return changed;
        existing.remove(); record.cancelRunBtn = null;
        return true;
    }
    if (existing) {
        record.cancelRunBtn = existing;
        if (record.finished) existing.disabled = false;
        return changed;
    }
    const actions = actionsOf(record);
    if (!actions) return changed;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-sm btn-default ui-hit-area';  // Resume or Continue stays the one primary.
    btn.dataset.cancelRun = '1';
    btn.textContent = TASK_CONTROL_TRIGGER_LABEL;
    btn.addEventListener('click', (event) => {
        event.stopPropagation();
        if (btn.isConnected && !btn.disabled) openCardControls(btn, record, read,
            { stop: (action) => stop(record, action), afterAction });
    });
    actions.appendChild(btn);
    record.cancelRunBtn = btn;
    return true;
}

/**
 * Open the shared menu on a work's controls: each action reaches the existing request owner at its exact target;
 * Stop and Stop earlier attempt stop each of their targets in order and say one combined result. `stopOne` is the
 * caller's own stop path for a target (a loaded card keeps its optimistic chip); `canAct` re-checks the current
 * controls before any effect; `afterAction` re-reads the state once an effect settled. A Resume's re-read is part of
 * its own request, which keeps the task pending until that read has settled (`resumeTaskAction`): it runs there, once.
 * Through that read the menu keeps its Stop enabled and offers no second Resume (`resumePending`).
 */
export function openWorkControls(anchor, work, { cancelPending = false, canAct = () => true, stopOne = requestStop,
    afterAction = null, withResume = false, held = undefined } = {}) {
    const grouped = workControls(work);
    const actions = workMenuActions(grouped, { cancelPending,
        resume: withResume && !resumePending(grouped.resume) });
    if (!actions.length) return null;
    const stopAll = (targets, action) => requestStopAll(targets, action, { stopOne });
    const reconcile = afterAction ? () => afterAction(ACTION_RESUME) : null;
    return openTaskControlMenu(anchor, {
        actions, pauseTarget: grouped.pause, busy: taskControlBusy(grouped.pause || work?.episode?.task_id), canAct,
        afterAction: afterAction && ((action) => (action === ACTION_RESUME ? undefined : afterAction(action))),
        onAction: (action) => {
            const target = actionTarget(grouped, action);
            if (action === ACTION_HURRY) return hurryTaskAction(target);
            if (action === ACTION_RESUME) return resumeTaskAction(target, { settle: reconcile, held });
            return action === ACTION_FINALIZE ? stopAll([target], action) : stopAll(target, ACTION_STOP_NOW);
        },
    });
}
