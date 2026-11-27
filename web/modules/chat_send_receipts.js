/** Owner-row receipts shared by live echoes and history. A projection never settles a send in this room. */
import { markIngressSaved } from './chat_activity.js';
import { ownerMessageBubble, ownerRowRoom } from './chat_history_replay.js';
import { markSwarmLabel } from './swarm_presentation.js';

export function createOwnerSendReceipts({ feed, chatId, unconfirmed, replies, composer }) {
    const local = row => !row.origin_projected && ownerRowRoom(row, chatId) === chatId;
    const ids = rows => rows.filter(row => row.role === 'user' && row.client_message_id && local(row))
        .map(row => String(row.client_message_id));

    // The host's saved row settles both the kept copy and the model choice its message named (composer.js).
    function settle(row) {
        if (!local(row)) return;
        unconfirmed.settle(row);
        composer()?.settleNewWorkRow(row);
    }

    function mark(row) {
        return markIngressSaved(feed, row, ownerMessageBubble(feed, row, chatId));
    }

    function confirm(row) {
        const bubble = ownerMessageBubble(feed, row, chatId);
        const changed = markIngressSaved(feed, row, bubble) | markSwarmLabel(feed, row, bubble) | replies.adopt(row);
        settle(row);
        return changed;
    }

    return { local, ids, settle, mark, confirm };
}
