// Per-frame thread routing for ONE chat instance: which socket frames belong to
// this column, and which of them may raise the global unread badge. One socket
// fans out client-side — a Project panel takes only its own thread, while Main
// keeps ordinary non-project traffic AND mirrors project progress, digests and
// logs as the "штаб" without ever showing raw project chat messages. The
// mirror is presentation only: a Project's own visible_revision stays the sole
// unread authority for that Project, so a mirrored frame never creates a second
// Main unread. The shared page state, the instance identity and the badge
// updater are handed over explicitly.
export function createFrameRouting({ state, isMain, chatId, updateUnreadBadge }) {
    const isKnownProjectFrame = (msg) => {
        const cid = Number(msg?.chat_id ?? 1);
        return state.projectChatIds instanceof Set && state.projectChatIds.has(cid);
    };

    function incrementUnreadIfNeeded(msg) {
        if (!isMain) return;  // the global unread badge tracks the main chat
        // Project visible_revision is the sole unread authority for a Project.
        // Main may mirror its summary/progress/log into the штаб live card, but
        // that presentation mirror must not create a second Main unread.
        if (isKnownProjectFrame(msg)) return;
        if (state.activePage === 'chat') return;
        state.unreadCount++;
        updateUnreadBadge();
    }

    // One socket, client-side fan-out: project instances take only their own
    // thread. The MAIN instance keeps ordinary non-project traffic AND mirrors
    // project progress/digests/logs as the "штаб", but never raw project chat
    // user/assistant messages.
    const isProjectMirrorFrame = (msg) => {
        if (!msg) return false;
        if (msg.type === 'log') return true;
        if (msg.is_progress) return true;
        if (msg.system_type === 'task_summary' || msg.system_type === 'project_digest') return true;
        return false;
    };

    const isMyThread = (msg, { mirrorProject = false } = {}) => {
        const cid = Number(msg?.chat_id ?? 1);
        if (isMain) {
            if (isKnownProjectFrame(msg)) {
                return mirrorProject && isProjectMirrorFrame(msg);
            }
            return true;
        }
        return cid === chatId;
    };

    return {
        isKnownProjectFrame,
        incrementUnreadIfNeeded,
        isProjectMirrorFrame,
        isMyThread,
    };
}
