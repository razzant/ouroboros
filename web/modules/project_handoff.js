/**
 * The hand-off receipt word of a Project binding (the Main history row that says a work moved into a room).
 * Main no longer draws a separate entry for it: the moved work keeps its own card (work_portal.js).
 */

// The gateway's typed receipt word (ouroboros/project_handoff.py RECEIPT_STATES).
// The binding is committed under every word; only these two prove that the Main
// history row is owed or already delivered.
export const DURABLE_RECEIPTS = new Set(['durable', 'already_delivered']);

/** Owner wording for a non-durable receipt; '' when nothing needs saying. */
export function receiptNotice(status) {
    if (DURABLE_RECEIPTS.has(status)) return '';
    return {
        unregistered: 'Project binding saved; the Main history receipt was sent but is not protected against a restart.',
        unavailable: 'Project binding saved; the Main history receipt could not be sent.',
        origin_unproven: 'Project binding saved; this conversation has no recorded Main origin, so no history receipt was written.',
    }[status] || 'Project binding saved; the Main history receipt is unconfirmed.';
}
