// Pure presentation helpers for the sidebar's status marks (docs/DESIGN.md "Status marks").
//
// Each place in the sidebar reads its room in the server's work projection (`/api/state` `rooms`, keyed by the
// room's chat id; ouroboros/gateway/work_status.py): the room's shape (`mark`: moving | still | pause | ring | none),
// whether something waits for the owner (the room's turn), and its word with the line of the work that says it. This
// module owns no timer, request, socket or durable state and derives no shape of its own; a caller supplies the
// previous projection when reconciling an incomplete body.

import { activeModelWaits, mergeModelWaits } from './model_wait.js';
import { activityWord, amberNote, roomAmber, roomSpeaker, workLine } from './work_status.js';
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

/**
 * One room as one mark: the server's shape, whether something waits for the owner (the room's turn, a server fact),
 * and its word with the line of the first work that says the same word (the episode's reason, then what still runs
 * beside it): a work Pausing beside a running one never lends its reason to the room's Working. `note` says what
 * waits for the owner when the word does not, so the label tells it in words and the amber is never colour alone. A
 * room that is not at rest with a shape this client does not know is drawn unconfirmed, never as something it is not.
 */
export function summarizeRoom(room, works = []) {
    if (!room || typeof room !== 'object' || room.activity === 'idle') return IDLE_SUMMARY;
    const mark = MARK_RANK.includes(room.mark) ? room.mark : 'ring';
    const word = activityWord(room);
    const speaker = roomSpeaker(room, works, word);
    const line = speaker ? workLine(speaker, word) : '';
    const note = amberNote(roomAmber(room));
    return { mark, owner: Boolean(room.amber), word, note, label: [word, line, note].filter(Boolean).join(' · ') };
}

// A room an incomplete or unavailable body did not list again: still shown, never as confirmed work.
const unconfirmed = () => {
    const word = activityWord({ activity: 'unknown' });
    return { mark: 'ring', owner: false, word, note: '', label: word };
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
    if (!complete) for (const chat of prior.keys()) next.set(chat, unconfirmed());
    for (const [chat, room] of Object.entries(rooms || {})) {
        const summary = summarizeRoom(room, data.works);
        if (summary !== IDLE_SUMMARY) next.set(String(chat), summary);
    }
    return { rooms: next, complete };
}

/**
 * Index the Project rooms by chat id. The Projects header's one mark stands for all of them: the strongest shape,
 * and whether any waits for the owner; its label joins their distinct words and what waits for the owner.
 */
export function buildProjectActivityIndex(rooms = new Map(), chatIds = []) {
    const byChat = new Map();
    for (const chat of chatIds) {
        const summary = rooms instanceof Map ? rooms.get(String(chat)) : null;
        if (summary) byChat.set(String(chat), summary);
    }
    const summaries = [...byChat.values()];
    if (!summaries.length) return { byChat, aggregate: IDLE_SUMMARY };
    const mark = summaries.map((summary) => summary.mark)
        .sort((a, b) => MARK_RANK.indexOf(a) - MARK_RANK.indexOf(b))[0];
    const distinct = (key) => [...new Set(summaries.map((summary) => summary[key]).filter(Boolean))];
    const [words, notes] = [distinct('word'), distinct('note')];
    return {
        byChat,
        aggregate: { mark, owner: summaries.some((summary) => summary.owner), word: words[0] || '',
            note: notes.join(' · '), label: [...words, ...notes].join(' · ') },
    };
}

/**
 * What a place's mark stands for, in words: its label and, when it has them, new messages (one translatable
 * phrase). A row's accessible name and tooltip and the first line of its menu say the mark with these words.
 */
export function markWords(summary = IDLE_SUMMARY, unread = false) {
    return [summary.label, unread ? tr('nav.unread', 'new messages') : ''].filter(Boolean).join(' · ');
}
