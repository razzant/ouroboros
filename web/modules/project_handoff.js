/** Compact Project work entries; event identity and observed work stay separate. */
import { projectReference } from './project_reference.js';
import { activeModelWaits, mergeModelWaits } from './model_wait.js';
import { isTerminalTaskDetail } from './log_events.js';
import { censusTaskPhase, paintTaskPhase } from './task_phase_chip.js';
import { formatMsgTime } from './chat_activity.js';

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

export function handoffPhase(activity, detail, connected = true) {
    const waits = mergeModelWaits({}, activity?.model_waits);
    return censusTaskPhase(activity, detail, connected,
        activeModelWaits(waits, false, activity?.task_attempt || 0).length > 0);
}

// Transfer anchors share their recorded handoff identity: a converted live card
// (`card`, at the request's own position; several roots of ONE owner message may
// each convert, and every one of them stays a visible card — a card is never a
// shadow) and the durable receipt row (`receipt`, at most one visible per
// identity). A receipt is folded under the first visible card of its identity, or
// under the earlier receipt, and every folded node stays in `shadows` so an
// evicted anchor hands over instead of dropping the transfer from the feed.
// A creation entry retains its own identity and subject. Matching transfers
// fold it visually; they never turn creation into a transfer or lend it status.
export function createProjectHandoffs({ feed, fetchDetail, mutate, attachCopy }) {
    const rows = new Map();  // Converted task, transfer receipt, or creation entry.
    let connected = true, destroyed = false, complete = false;
    let activities = new Map();
    const inFeed = node => feed.contains(node);
    const current = row => !destroyed && inFeed(row.node) && rows.get(row.key) === row;
    const visibleAnchor = id => [...rows.values()].find(row => row.id === id && inFeed(row.node));
    const matching = (taskId, projectId) => [...rows.values()].find(row =>
        row.projectId === projectId && row.subjects.has(taskId) && inFeed(row.node));
    function stampTime(row, ts) {
        const time = formatMsgTime(ts);
        if (!time) return;
        row.ts = ts;
        row.time.textContent = time.short;
        row.time.title = time.full;
        row.time.hidden = false;
    }
    function paint(row) {
        const activity = activities.get(row.taskId);
        // Keep already observed result facts when connectivity or a partial
        // census withdraws activity. A retry clears this same-subject cache.
        if (activity?.status || activity?.outcome_axes) row.detail = { ...activity };
        const phase = handoffPhase(activity, row.detail, connected);
        paintTaskPhase(row.status, phase, row.secondary);
    }
    function reconcileStarted(node) {
        const represented = [...rows.values()].some(row => row.kind !== 'started'
            && row.projectId === node.dataset.projectId && row.subjects.has(node.dataset.taskId) && inFeed(row.node));
        node.hidden = represented;
    }
    function reconcileAnnotation(note) {
        const [projectId, , taskId] = (note.dataset.destinationKey || '').split('|');
        const represented = Boolean(matching(taskId, projectId)
            && ['scheduled', 'delivered'].includes(note.dataset.annotationStatus));
        note.hidden = represented;
        const actions = note.parentElement?.querySelector('.msg-routing-actions');
        if (actions) actions.hidden = represented;
    }
    // A durable receipt row folded under a card proves the transfer is in Main
    // history: the card's "not saved" mark, if any, was about exactly that gap.
    function fold(shadow, under) {
        shadow.node.hidden = true;
        under.shadows.push(shadow, ...shadow.shadows.splice(0));
        for (const taskId of shadow.subjects) under.subjects.add(taskId);
        // The receipt owns the transfer time. Conversion supplies no invented
        // date while its durable row is still in flight; ordering stays intact.
        if (shadow.kind === 'receipt') for (const row of rows.values()) {
            if (row.id === shadow.id && row.kind === 'card') stampTime(row, shadow.ts);
        }
        if (shadow.kind === 'receipt' && under.node.dataset.receipt) {
            delete under.node.dataset.receipt;
            under.node.classList.remove('project-handoff--unsaved');
        }
    }
    // An evicted anchor hands its shadows to a surviving anchor of the same
    // identity, else the first still-mounted shadow takes over — keeping the
    // CURRENT liveness subject (a followed retry), never the shadow's older one.
    function promote(row) {
        rows.delete(row.key);
        const survivor = visibleAnchor(row.id);
        if (survivor) for (const taskId of row.subjects) survivor.subjects.add(taskId);
        for (const next of row.shadows) {
            if (survivor) { fold(next, survivor); continue; }
            if (!inFeed(next.node)) continue;
            next.node.hidden = false;
            // Folding groups receipts, not execution evidence. A different root
            // sharing the origin must recover its OWN status when it reappears.
            const follows = row.followed.has(next.taskId);
            rows.set(next.key, { ...next, subjects: new Set([...row.subjects, ...next.subjects]),
                shadows: row.shadows.filter(other => other !== next),
                ...(follows ? { detail: row.detail, taskId: row.taskId, followed: new Set(row.followed) } : {}),
                epoch: row.epoch + 1, pending: false, checked: false });
            return true;
        }
        return true;
    }
    function sweep() {
        let changed = false;
        for (const row of [...rows.values()]) if (!inFeed(row.node)) changed = promote(row) || changed;
        return changed;
    }
    /** With a node: only that node's projection. Without: every dependent node. */
    function reconcile(node) {
        if (destroyed) return;
        if (node) {
            if (node.dataset?.systemType === 'project_started') reconcileStarted(node);
            const note = node.querySelector?.('.msg-routing-annotation');
            if (note) reconcileAnnotation(note);
            const mountedStart = node.dataset?.systemType === 'project_started'
                && rows.get(`started:${node.dataset.projectId}`)?.node === node;
            if (node.dataset?.systemType !== 'project_handoff' && !mountedStart) return;
        }
        sweep();
        for (const started of feed.querySelectorAll('[data-system-type="project_started"]')) reconcileStarted(started);
        for (const note of feed.querySelectorAll('.msg-routing-annotation')) reconcileAnnotation(note);
    }
    function follow(row, successor) {
        if (!successor || row.followed.has(successor)) return false;
        row.followed.add(successor);
        row.subjects.add(successor);
        row.taskId = successor;
        row.epoch++;
        row.checked = false;
        row.detail = null;
        return true;
    }
    function resolve(row) {
        if (!current(row) || !complete || !connected || activities.has(row.taskId)
            || row.node.hidden || row.pending || row.checked || isTerminalTaskDetail(row.detail)) return;
        const taskId = row.taskId, epoch = row.epoch;
        row.pending = true;
        row.checked = true;
        Promise.resolve().then(() => fetchDetail(taskId)).then(detail => {
            if (!current(row) || epoch !== row.epoch || taskId !== row.taskId) return;
            // The retained task result, never project activity, names a retry.
            const effectiveRetry = detail?.task_id !== taskId
                && (detail?.original_task_id === taskId || detail?.retry_lineage?.some(item => item.task_id === taskId));
            const successor = String((effectiveRetry ? detail.task_id : '') || detail?.superseded_by || detail?.retry_task_id || '');
            if (successor && successor !== taskId) {
                if (!follow(row, successor)) return;
            } else {
                row.detail = detail;
                // A successful read during withdrawal/retry publication is not
                // a settled absence. The existing census refresh revisits that
                // in-flight fact; failures and inactive history remain latched.
                if (['running', 'scheduled'].includes(detail?.status)) row.checked = false;
            }
            mutate(() => paint(row));
        }).catch(() => {
            // A failed read remains unknown until a real re-entry/reconnect,
            // not another costly detail request on each census tick.
        }).finally(() => {
            row.pending = false;
            if (current(row) && row.taskId !== taskId) resolve(row);
        });
    }
    function mount(node, { taskId, projectId, projectName, title, handoffId, kind = 'receipt', receipt = '', ts = '' }) {
        if (!taskId || !projectId || destroyed) return node;
        const id = kind === 'started' ? `started:${projectId}` : handoffId || `legacy:${JSON.stringify([taskId, projectId])}`;
        sweep();
        const anchor = visibleAnchor(id);
        if (anchor && anchor.node === node) { anchor.subjects.add(taskId); stampTime(anchor, ts); return node; }
        node.dataset.projectId = projectId;
        node.dataset.taskId = taskId;
        if (kind !== 'started') node.dataset.handoffId = id;
        node.dataset.systemType = kind === 'started' ? 'project_started' : 'project_handoff';
        node.classList.add('project-handoff');
        const line = document.createElement('div');
        line.className = 'project-handoff-heading';
        const status = document.createElement('span');
        status.setAttribute('role', 'status');
        const secondary = document.createElement('span');
        secondary.className = 'chat-live-phase-secondary';
        secondary.hidden = true;
        const name = document.createElement('span');
        name.className = 'project-handoff-title';
        name.textContent = title || projectName || 'Project';
        const phases = document.createElement('div');
        phases.className = 'project-handoff-phase';
        phases.append(status, secondary);
        line.append(name, phases);
        const reference = projectReference({ id: projectId, name: projectName }, { layout: 'inline', taskId });
        const footer = document.createElement('div');
        footer.className = 'project-handoff-footer';
        const time = document.createElement('div');
        time.className = 'msg-time';
        time.hidden = true;
        footer.append(time);
        // A converted card whose receipt is not durable is an honest live chip,
        // never a claim that Main history holds this transfer (it will not
        // survive a reload as an anchor; the binding and the pointer do).
        if (kind === 'card' && receipt && !DURABLE_RECEIPTS.has(receipt)) {
            node.dataset.receipt = receipt;
            node.classList.add('project-handoff--unsaved');
        }
        // One composition for conversion, creation and replay, independent of
        // the old task-card or message shell. Copy keeps the shared chat control.
        node.replaceChildren(line, reference, footer);
        const copy = attachCopy?.(node, `${name.textContent}\n${reference.querySelector('.chat-live-project-name').textContent}`);
        if (copy) footer.append(copy);
        const row = { key: kind === 'card' ? `card:${taskId}` : id, id, node, status, secondary, kind, taskId, projectId,
            time, ts: '', subjects: new Set([taskId]), followed: new Set([taskId]), shadows: [], detail: null, pending: false, checked: false, epoch: 0 };
        stampTime(row, ts || (kind === 'card' ? anchor?.ts : ''));
        if (kind === 'receipt' && anchor) {
            // Duplicate delivery is not evidence that a differently named
            // execution supersedes its subject: one receipt, the rest shadowed.
            fold(row, anchor);
            return anchor.node;
        }
        rows.set(row.key, row);
        if (kind === 'card' && anchor?.kind === 'receipt') {
            rows.delete(anchor.key);
            fold(anchor, row);
            if (row.taskId === anchor.taskId) row.detail = anchor.detail;
        }
        paint(row);
        // A card is already in the feed when it converts: the rows it now
        // represents (Started, routing) fold at once. addMessage appends a
        // receipt after mounting; its own reconcile follows the insert.
        if (kind === 'card' && inFeed(node)) reconcile();
        return node;
    }
    function snapshot(data) {
        if (destroyed) return;
        complete = data?.active_chat_activities_complete === true && data.supervisor_ready === true;
        const incoming = new Map((data?.active_chat_activities || []).map(a => [String(a.activity_id || ''), a]));
        // Only the host's explicit technical retry linkage advances a subject.
        // A shared Project, owner origin or an explicit Continue is not lineage.
        const successors = new Map();
        for (const [id, activity] of incoming) {
            const predecessor = String(activity.timeout_retry_from || activity.original_task_id || '');
            if (predecessor && predecessor !== id) successors.set(predecessor, id);
        }
        for (const row of rows.values()) {
            while (successors.has(row.taskId) && follow(row, successors.get(row.taskId))) { /* explicit chain */ }
            if (incoming.has(row.taskId) && !activities.has(row.taskId)) {
                row.epoch++;
                row.checked = false;
            }
        }
        activities = incoming;
        // An evicted anchor is the only reason the feed-wide projections move on a
        // census tick; a steady feed repaints its rows and nothing else.
        if (rows.size) mutate(() => { if (sweep()) reconcile(); for (const row of rows.values()) paint(row); });
        for (const row of rows.values()) resolve(row);
    }
    return { mount, reconcile, snapshot,
        setConnected(value) {
            if (connected !== value) {
                connected = value;
                // Reconnection alone is not a fresh activity observation.
                activities = new Map();
                for (const row of rows.values()) { row.epoch++; row.checked = false; }
            }
            if (rows.size) mutate(() => { for (const row of rows.values()) paint(row); });
        },
        destroy() { destroyed = true; rows.clear(); activities.clear(); },
    };
}
