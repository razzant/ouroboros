import { rawTimestampEpoch } from './utils.js';

// Message identity and presentation primitives for ONE chat instance.
// `buildMessageKey` and `rememberMessageKey` are the dedup contract shared by
// the live socket and history replay: the same durable row must produce the
// same key on both paths, and the seen-key window is bounded so a long session
// cannot grow without limit. `formatMsgTime`, `stampNodeTimestamp` and
// `getSenderLabel` are the presentation half — a sortable numeric stamp on the
// node and the sender text a reader sees. The dedup window and this browser
// tab's session id are handed over explicitly, so a Main chat and a Project
// panel keep separate windows and neither mislabels the other's messages.
export function createMessageIdentity({ chatSessionId, seenMessageKeys, messageKeyOrder }) {
    function buildMessageKey(role, text, timestamp, opts = {}) {
        if (opts.clientMessageId) return `client|${opts.clientMessageId}`;
        if (role !== 'user' && !opts.isProgress && opts.taskId) {
            return [
                'task',
                role,
                opts.systemType || '',
                opts.source || '',
                opts.taskId,
                text,
            ].join('|');
        }
        if (!timestamp) return '';
        return [
            role,
            opts.isProgress ? '1' : '0',
            opts.systemType || '',
            opts.source || '',
            opts.senderLabel || '',
            opts.senderSessionId || '',
            opts.taskId || '',
            timestamp,
            text,
        ].join('|');
    }

    function rememberMessageKey(key) {
        if (!key || seenMessageKeys.has(key)) return;
        seenMessageKeys.add(key);
        messageKeyOrder.push(key);
        if (messageKeyOrder.length > 2000) {
            const oldest = messageKeyOrder.shift();
            if (oldest) seenMessageKeys.delete(oldest);
        }
    }

    function formatMsgTime(isoStr) {
        if (!isoStr) return null;
        try {
            const d = new Date(isoStr);
            if (isNaN(d)) return null;
            const now = new Date();
            const pad = n => String(n).padStart(2, '0');
            const hhmm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            const todayStr = now.toDateString();
            const yesterday = new Date(now);
            yesterday.setDate(now.getDate() - 1);
            let short;
            if (d.toDateString() === todayStr) short = hhmm;
            else if (d.toDateString() === yesterday.toDateString()) short = `Yesterday, ${hhmm}`;
            else short = `${months[d.getMonth()]} ${d.getDate()}, ${hhmm}`;
            const full = `${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()} at ${hhmm}`;
            return { short, full };
        } catch {
            return null;
        }
    }

    function stampNodeTimestamp(node, raw, { anchor = false } = {}) {
        if (!node) return false;
        const epoch = rawTimestampEpoch(raw);
        if (!Number.isFinite(epoch)) return false;
        if (anchor && node.dataset.ts) {
            const current = Number(node.dataset.ts);
            const next = Number.isFinite(current) ? Math.min(current, epoch) : epoch;
            node.dataset.ts = String(next);
            return Number.isFinite(current) && next < current;
        } else {
            node.dataset.ts = String(epoch);
        }
        return false;
    }

    function getSenderLabel(role, isProgress = false, systemType = '', opts = {}) {
        if (role === 'user') {
            if (opts.source === 'telegram') return opts.senderLabel || 'Telegram';
            if (opts.senderSessionId && opts.senderSessionId !== chatSessionId) {
                return `WebUI (${opts.senderSessionId.slice(0, 8)})`;
            }
            return opts.senderLabel || 'You';
        }
        if (role === 'system') {
            if (systemType === 'task_summary') return '📋 Task Summary';
            if (systemType === 'skill_review') return '📋 Skill Review';
            return '📋 System';
        }
        if (isProgress) return '💬 Thought';
        return 'Ouroboros';
    }

    return {
        buildMessageKey,
        rememberMessageKey,
        formatMsgTime,
        stampNodeTimestamp,
        getSenderLabel,
    };
}
