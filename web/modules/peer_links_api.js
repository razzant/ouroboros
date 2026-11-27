// The browser client of my human's doors to a link with another Ouroboros of theirs (`POST /api/peer-links/*`,
// gateway/peer_letters_api.py) and the reader of a linked place's own facts on `/api/places` (place_projection.py:
// `peer`, `peer_effective`, `peer_facts`, `link`). Every JSON shape of this surface is read here, so a field the
// server renames is one fix in this file. Readers are tolerant (a missing field stays absent, never a guessed zero); writers send exactly
// the door's keys.
import { apiFetch, jsonPost } from './api_client.js';

export const PEER_CONNECTOR = 'ouroboros';
/** A place that is another Ouroboros of the owner's (peer_links: connector and provider `ouroboros`). */
export const isPeerPlace = (place) => place?.connector === PEER_CONNECTOR;

const text = (value) => (value === null || value === undefined ? '' : String(value));
const object = (value) => (value && typeof value === 'object' && !Array.isArray(value) ? value : null);
const bool = (value) => (value === true ? true : value === false ? false : null);

/** The facts of one linked place in the browser's names; `{}` for any other place, so its shape stays as it was. Its
 *  conversation is the place's own (`rooms`, the place sheet), read like any place's.
 *  Python contract: `ouroboros/gateway/client_contracts.py::PlaceSummary`. */
export function peerPlaceFacts(raw) {
    const place = object(raw) || {};
    if (place.connector !== PEER_CONNECTOR) return {};
    const peer = object(place.peer), link = object(place.link), effective = object(place.peer_effective);
    const work = object(place.peer_work), room = object(place.peer_room);
    return {
        peer: peer ? { mine: peer.mine === true, mayStart: peer.may_start === true } : null,
        // What the letter door accepts from it now (peer_links.effective_permissions), never derived here:
        // `messages` and `assignments` true | false | null (null: the link registry could not be read), and
        // `reason`, the first fact that refuses what is refused ('' when nothing is).
        peerEffective: effective ? { messages: bool(effective.messages), assignments: bool(effective.assignments),
            reason: text(effective.reason) } : null,
        peerFacts: text(place.peer_facts),
        // `remoteUnknownAt`: since when the other side answers that it does not know this link (it needs a new
        // introduction; nothing is sent until then).
        link: link ? { linkId: text(link.link_id), code: text(link.code), codeCheck: text(link.code_check),
            peerInstallId: text(link.peer_install_id), createdAt: text(link.created_at), revokedAt: text(link.revoked_at),
            remoteMayStart: bool(link.remote_may_start), remoteUnknownAt: text(link.remote_unknown_at) } : null,
        // Its letters' work whose record changed last, as its acceptance record states it (place_projection
        // .peer_works, the owner's list read only): who sent it, its title, its stored status and the room it lives
        // in; no date (the record keeps no acceptance time). Absent: none was accepted; `unknown`: the store could not
        // be read now (never a guessed absence).
        peerWork: work?.unknown === true ? { unknown: true } : work && text(work.task_id) ? { taskId: text(work.task_id),
            title: text(work.title), status: text(work.status), phase: text(work.phase), from: text(work.from),
            roomChatId: Number.isSafeInteger(work.room_chat_id) ? work.room_chat_id : null } : null,
        // While nothing has arrived from it, the room this side wrote to it in and the time of its newest receipt there
        // (place_projection.peer_rooms, the owner's list read only); its words are read there like any room's.
        peerRoom: room?.unknown === true ? { unknown: true } : room && Number.isSafeInteger(room.room_chat_id)
            ? { roomChatId: room.room_chat_id, wroteAt: text(room.wrote_at) } : null,
    };
}

/** A link as the doors answer it (`peer_links.link_view`): never a key or a hash. */
export function normalizeLink(raw) {
    const link = object(raw) || {};
    return { linkId: text(link.link_id), placeId: text(link.place_id), peerLabel: text(link.peer_label),
        peerUrl: text(link.peer_url), peerInstallId: text(link.peer_install_id), code: text(link.code),
        codeCheck: text(link.code_check), createdAt: text(link.created_at), revokedAt: text(link.revoked_at),
        remoteMayStart: bool(link.remote_may_start), state: text(link.state) };
}

/** My human's invitation: `mayStart` is their answer about the other computer, `note` the rule for its place, `label`
 *  what the coming Ouroboros is called here. `{invite, linkId, expiresAt}`; the invite works once, for 15 minutes. */
export async function createInvite({ mayStart, note = '', label = '' }) {
    const data = await jsonPost('/api/peer-links/invite', { may_start: mayStart === true,
        ...(note.trim() ? { note: note.trim() } : {}), ...(label.trim() ? { label: label.trim() } : {}) });
    return { invite: text(data?.invite), linkId: text(data?.link_id), expiresAt: text(data?.expires_at) };
}

/** What a pasted invitation says, read without accepting it: who sent it, from where, until when. */
export async function previewInvite(invite) {
    const data = await jsonPost('/api/peer-links/accept', { invite: String(invite || '').trim(), preview: true });
    return { linkId: text(data?.link_id), peerLabel: text(data?.peer_label), peerUrl: text(data?.peer_url),
        peerInstallId: text(data?.peer_install_id), remoteMayStart: bool(data?.remote_may_start),
        expiresAt: text(data?.expires_at) };
}

/** Accept a pasted invitation: the link starts paused, its code unchecked (`confirmLink` starts it). */
export async function acceptInvite({ invite, mayStart, note = '', label = '' }) {
    return normalizeLink(await jsonPost('/api/peer-links/accept', { invite: String(invite || '').trim(),
        may_start: mayStart === true, ...(note.trim() ? { note: note.trim() } : {}),
        ...(label.trim() ? { label: label.trim() } : {}) }));
}

export async function readLink(linkId, signal) {
    const response = await apiFetch(`/api/peer-links/${encodeURIComponent(linkId)}`, { signal });
    if (!response.ok) throw new Error('The link could not be read.');
    const value = await response.json();
    return value?.link ? normalizeLink(value.link) : null;
}

/** My human compared the six digits on both sides: this side's letters start. */
export async function confirmLink(linkId) {
    return normalizeLink(await jsonPost(`/api/peer-links/${encodeURIComponent(linkId)}/confirm`, {}));
}

/** Remove the link on this side: its key leaves, letters stop both ways, the place pauses. */
export async function revokeLink(linkId) {
    return normalizeLink(await jsonPost(`/api/peer-links/${encodeURIComponent(linkId)}/revoke`, {}));
}

/** `host:port` of an Ouroboros's address (its computer has no other name here), '' when it is not a URL. */
export function addressOf(url) {
    try {
        return new URL(String(url || '')).host;
    } catch {
        return '';
    }
}
