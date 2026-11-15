// Pure presentation helpers for the sidebar's status marks (docs/DESIGN.md "Status marks").
//
// Each place in the sidebar reads its room in the server's work projection (`/api/state` `rooms`, keyed by the
// room's chat id; ouroboros/gateway/work_status.py): the room's shape (`mark`: moving | still | pause | ring | none),
// whether something waits for the owner (the room's turn), and its word with the line of the work that says it. This
// module owns no timer, request, socket or durable state and derives no shape of its own; a caller supplies the
// previous projection when reconciling an incomplete body.

import { activeModelWaits, mergeModelWaits } from './model_wait.js';
import { activityWord, amberNote, questionKey, roomAmber, roomSpeaker, turnQuestion, workLine } from './work_status.js';
import { questionWords } from './question_presentation.js';
import { tr } from './i18n.js';

/** Whether a census row's own model waits are current, and whether one needs the owner (Activity reads the same rule). */
export function waitingModel(row) {
    const waits = row?.model_waits;
    if (!waits || typeof waits !== 'object') return { waiting: false, ownerAction: false };
    // The same admission rule the chat card applies: a malformed wait row is
    // dropped here exactly as there, so the sidebar cannot go static on a row
    // the card would never show.
    const active = activeModelWaits(mergeModelWaits({}, waits), false, Number(row?.task_attempt) || 0);
    return { waiting: active.length > 0, ownerAction: active.some((wait) => wait.requires_owner_action === true) };
}

/** A room with nothing to show. */
export const IDLE_SUMMARY = Object.freeze({ mark: 'none', owner: false, word: '', note: '', label: '' });
// The shapes from the strongest: work that moves, a queue or a wait that ends by itself, a pause, an unconfirmed ring.
export const MARK_RANK = Object.freeze(['moving', 'still', 'pause', 'ring', 'none']);

// A name and a tooltip take a question's opening words up to this many characters; the row's own line shows as many
// of them as its width holds (style.css `.nav-project-says`).
const ASK_CHARS = 160;

/**
 * The question that says a waiting room's turn, in its own opening words (`{ key, words }`), or null: the item
 * turnQuestion names, read in `asks` (the body's `current_questions` by `task_id:quiz_id`). A body that carries no row
 * for it (a read with a gap) keeps the words the room's `prior` summary held for the same question, for as long as its
 * item stands in the turn; a row that carries no words gives none.
 */
function askOf(room, asks, prior) {
    const item = room.amber ? turnQuestion(room.turn) : null;
    if (!item) return null;
    const key = questionKey(item);
    const row = asks instanceof Map ? asks.get(key) : null;
    const words = row ? questionWords(row.question, ASK_CHARS) : prior?.ask?.key === key ? prior.ask.words : '';
    return words ? { key, words } : null;
}

/**
 * One room as one mark: the server's shape, whether something waits for the owner (the room's turn, a server fact),
 * and its word with the line of the first work that says the same word (the episode's reason, then what still runs
 * beside it): a work Pausing beside a running one never lends its reason to the room's Working. `note` says what
 * waits for the owner when the word does not, so the label tells it in words and the amber is never colour alone;
 * `ask` holds the waiting question's own opening words when the body (`asks`) or the room's `prior` summary has
 * them: a field of its own, so the label (a row menu's first line too) stays the state's short words. A room that is
 * not at rest with a shape this client does not know is drawn unconfirmed, never as something it is not.
 */
export function summarizeRoom(room, works = [], asks = null, prior = null) {
    if (!room || typeof room !== 'object') return IDLE_SUMMARY;
    const ask = askOf(room, asks, prior);
    if (room.activity === 'idle') {
        // A room at rest shows nothing, unless a question of it still waits for the owner: then only the owner's mark.
        if (!room.amber) return IDLE_SUMMARY;
        const waits = amberNote(roomAmber(room));
        return { mark: 'none', owner: true, word: '', note: waits, label: waits, ...(ask ? { ask } : {}) };
    }
    const mark = MARK_RANK.includes(room.mark) ? room.mark : 'ring';
    const word = activityWord(room);
    const speaker = roomSpeaker(room, works, word);
    const line = speaker ? workLine(speaker, word) : '';
    const note = amberNote(roomAmber(room));
    // What a sidebar row says under its name (owner 12.11, the fifth round): the work's own name and its focus line.
    const lead = speaker || (Array.isArray(works) ? works.find((work) => work?.room === room.room
        || (Array.isArray(room.works) && room.works.includes(work?.work_id))) : null) || null;
    const name = String(lead?.name || '').trim();
    const focus = String(lead?.focus?.text || '').trim();
    return { mark, owner: Boolean(room.amber), word, note, line, name, focus,
        label: [word, line, note].filter(Boolean).join(' · '), ...(ask ? { ask } : {}) };
}

// A room an incomplete or unavailable body did not list again: still shown, never as confirmed work. The words of a
// question it held stay with it unseen, so a later body that lists the same question without its row can say them.
const unconfirmed = (prior = null) => {
    const word = activityWord({ activity: 'unknown' });
    return { mark: 'ring', owner: false, word, note: '', label: word, ...(prior?.ask ? { ask: prior.ask } : {}) };
};

/**
 * Reconcile one body against the previous rooms. A complete supervisor-ready body (`works_complete`) proves that
 * an unlisted room is idle; a partial, unavailable or disconnected one keeps every room it omitted, unconfirmed.
 * Rooms it lists describe their own observed state.
 */
export function reconcileProjectRooms(previous = new Map(), data = {}) {
    const prior = previous instanceof Map ? previous : new Map();
    const rooms = data?.rooms && typeof data.rooms === 'object' ? data.rooms : null;
    const complete = Boolean(rooms) && Array.isArray(data.works) && data.works_complete === true
        && data.supervisor_ready === true;
    const next = new Map();
    if (!complete) for (const [chat, before] of prior) next.set(chat, unconfirmed(before));
    // The questions this body carries, by their address: a waiting room's line says its question in these words.
    const asks = new Map((Array.isArray(data?.current_questions) ? data.current_questions : [])
        .filter((row) => row?.task_id).map((row) => [questionKey(row), row]));
    for (const [chat, room] of Object.entries(rooms || {})) {
        const summary = summarizeRoom(room, data.works, asks, prior.get(String(chat)));
        if (summary !== IDLE_SUMMARY) next.set(String(chat), summary);
    }
    return { rooms: next, complete };
}

/**
 * Index the Project rooms by chat id (`chatIds`, every reserved room, and the rows'). The aggregate stands for all of
 * them: the strongest shape, and whether any waits for the owner; its label joins their distinct words and what waits
 * for the owner. Ouroboros's picture reads its `owner` (pictureOwner). What waits counts only the rooms the feed's
 * `rows` can show, so an amber picture always has an amber row to reach; without rows, every room.
 */
export function buildProjectActivityIndex(rooms = new Map(), chatIds = null, rows = null) {
    const byChat = new Map();
    const shown = Array.isArray(rows) ? new Set(rows.map((row) => String(row?.chat_id))) : null;
    // The reserved rooms and the feed's own: a room newer than a held set still counts by its row (the isolation
    // set itself stays app.js's).
    for (const chat of new Set([...(Array.isArray(chatIds) ? chatIds : []), ...(rows || []).map((row) => row?.chat_id)]
        .map(String))) {
        const summary = rooms instanceof Map ? rooms.get(String(chat)) : null;
        if (summary) byChat.set(String(chat), summary);
    }
    const summaries = [...byChat.values()];
    if (!summaries.length) return { byChat, aggregate: IDLE_SUMMARY };
    const mark = summaries.map((summary) => summary.mark)
        .sort((a, b) => MARK_RANK.indexOf(a) - MARK_RANK.indexOf(b))[0];
    const reachable = shown ? [...byChat].filter(([chat]) => shown.has(chat)).map(([, summary]) => summary) : summaries;
    const distinct = (key, from = summaries) => [...new Set(from.map((summary) => summary[key]).filter(Boolean))];
    const [words, notes] = [distinct('word'), distinct('note', reachable)];
    return {
        byChat,
        aggregate: { mark, owner: reachable.some((summary) => summary.owner), word: words[0] || '',
            note: notes.join(' · '), label: [...words, ...notes].join(' · ') },
    };
}

/**
 * What Ouroboros's picture says for the rooms it stands for (docs/DESIGN.md "Status marks"): whether Main's own room
 * (`main`) or a Project the feed can show (`projects`, the index's aggregate) waits for the owner; `any` is the dot on
 * the picture, on his card and where it opens the drawer alike. `projects` also tells the card's words that the wait
 * is in a Project: Main's own is its room's word there already.
 */
export function pictureOwner(rooms = null, index = null) {
    const main = Boolean(rooms?.get?.('1')?.owner);
    const projects = Boolean(index?.aggregate?.owner);
    return { main, projects, any: main || projects };
}

/**
 * What a place's mark stands for, in words: its label and, when it has them, new messages (one translatable
 * phrase). A row's accessible name and tooltip and the first line of its menu say the mark with these words.
 */
export function markWords(summary = IDLE_SUMMARY, unread = false) {
    return [summary.label, unread ? tr('nav.unread', 'new messages') : ''].filter(Boolean).join(' · ');
}
