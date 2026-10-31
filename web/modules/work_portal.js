/**
 * One work, one card (FINAL_DESIGN D11; owner decision 16.10 Q1 = A): a Main card whose work lives in a
 * Project room is that work's PORTAL. It stays the same `.chat-live-card` node at the request's
 * position and only changes form: status chip + quiet fact, title, one latest narration line, the
 * meta line, then one action row: the Project reference first, the controls at its end ([Resume] [Stop…]; Reply is
 * the row menu's) (owner 26.10, B).
 * The timeline, reviews and children live in the room's full card; nothing here appends rows, so Main
 * stays calm.
 *
 * Everything that binds a card ends here (`createWorkPortals`): the owner's Turn into project (the
 * same move the mind makes with `move_to_project`), the `/api/state` `task_bindings` fact, a census row
 * that names the work's Project, and a replayed `project_started` / `project_handoff` row (those rows
 * are data, not a second product; they only say which card moved where). The present comes from the
 * all-chat census (`last_line`, `last_line_at`, `tool_calls`, `detached`; CONTRACTS K8); a fact the
 * row does not carry stays absent, so an older host leaves today's card.
 *
 * A Continue is the next episode of the SAME card (owner QB = A) on every surface, whoever made it:
 * the predecessor's own record (its admitted link) is read again when the owner presses Continue on a
 * card here, or when the census lists a new root that may be its successor. A successor whose frames
 * reach this chat (a room, Main's own work) then joins its card (task_episodes.js); Main sees none of a
 * portal's successor's frames, so that successor's census row is its whole present there.
 */
import { apiClient, fetchTaskDetailStrict } from './api_client.js';
import { showToast } from './toast.js';
import { isTerminalTaskDetail, normalizeLogTs } from './log_events.js';
import { censusTaskFacts, liveCardLabel, syncParkedPhase } from './task_phase_chip.js';
import { boundActivityPreview, projectIdFromTask, refitCollapsedActivity, renderCollapsedActivity, taskCostProjection } from './chat_activity.js';
import { DURABLE_RECEIPTS, receiptNotice } from './project_handoff.js';
import { nameProjectReference, projectReference } from './project_reference.js';
import { admittedSuccessor } from './task_episodes.js';
import { continueControlSuccessor } from './task_continue.js';

const key = (value) => String(value || '').trim();
export const isPortal = (record) => record?.root?.dataset?.portal === '1';

/** Mark one root card as the portal of its Project. Idempotent; true when the DOM changed. */
export function bindWorkPortal(record, project = {}) {
    const root = record?.root;
    const projectId = key(project?.id);
    if (!root || record.isSubagent || !projectId) return false;
    let changed = false;
    if (root.dataset.portal !== '1') { root.dataset.portal = '1'; changed = true; }
    if (root.dataset.projectId !== projectId) { root.dataset.projectId = projectId; changed = true; }
    // In this form the card discloses nothing (the room holds what it would open): its "Details"
    // button is hidden (style.css) and the summary stays reading content. The reference below is the one door.
    // An unnamed Project reads `Project` (the reference's own rule), never its id; beside a title that already
    // says the Project's name it reads `Project` too, its accessible name whole.
    const target = { id: projectId, name: key(project?.name) };
    if (!record.portalRefEl) {
        record.portalRefEl = projectReference(target, { layout: 'footer', taskId: record.groupId, beside: record.titleEl });
        changed = true;
    } else {
        nameProjectReference(record.portalRefEl, target);
    }
    // The reference leads the card's own action row, under the meta line; the controls stay at the row's end.
    const row = record.footEndEl;
    if (row && row.firstElementChild !== record.portalRefEl) { row.prepend(record.portalRefEl); changed = true; }
    return changed;
}

/**
 * The bound work's present, as the host's live-works census states it. Absent fields stay absent:
 * the card never invents a line or a number.
 */
export function portalPresent(activity) {
    if (!activity || typeof activity !== 'object') return {};
    const line = typeof activity.last_line === 'string' ? activity.last_line.trim() : '';
    return {
        line,
        lineAt: typeof activity.last_line_at === 'string' ? activity.last_line_at : '',
        toolCalls: Number.isInteger(activity.tool_calls) ? activity.tool_calls : null,
    };
}

/**
 * The owner's routing note under his message names the room his words went to. While the moved
 * work's portal stands in the same feed, the portal is that note's whole content (same work, same
 * Project, same reference), so the note steps aside. A later message PASSED ON to that work (a steer,
 * a mailbox delivery) is its own fact: its receipt stays on the owner's message.
 */
const PASSED_ON = ['steer_task', 'mailbox_delivery'];

export function reconcilePortalNotes(feed, scope = feed) {
    if (!feed) return false;
    const notes = scope?.matches?.('.msg-routing-annotation') ? [scope]
        : [...(scope?.querySelectorAll?.('.msg-routing-annotation') || [])];
    let changed = false;
    let portals = null;  // the works the feed shows as portals, read once when a note first asks
    for (const note of notes) {
        const [projectId, , taskId] = String(note.dataset.destinationKey || '').split('|');
        const represented = Boolean(projectId && taskId && !PASSED_ON.includes(note.dataset.annotationAction || '')
            && ['scheduled', 'delivered'].includes(note.dataset.annotationStatus)
            && (portals ??= new Set([...feed.querySelectorAll('.chat-live-card[data-portal="1"]')]
                .map((card) => JSON.stringify([card.dataset.taskId, card.dataset.projectId]))))
                .has(JSON.stringify([taskId, projectId])));
        if (note.hidden !== represented) { note.hidden = represented; changed = true; }
        const actions = note.parentElement?.querySelector('.msg-routing-actions');
        if (actions && actions.hidden !== represented) { actions.hidden = represented; changed = true; }
    }
    return changed;
}

/**
 * The portals of one Main chat instance. `records` is the instance's card registry; `card` are its card
 * mechanics (each returns whether the DOM changed): `get(id)` mints or finds a root record, `show(record)`
 * mounts it, `reanchor(record, ts)`, `collapse(record)`, `restore(record, hold)`, `meta(record, cost?)`,
 * `controls(record)` (Stop/Resume), `finish(id, outcome)`, `missing(id)` (durable-detail read),
 * `note(record, detail)` (a settled card's detail), `summary(row)` (a final's typed outcome).
 * `onConverted` is the composer's "this chat is free" cue;
 * `append(row, text, opts)` draws an ordinary Project row, `remember(key)` keeps a history key, `actionsOf(record)`
 * is the card's own action row.
 */
export function createWorkPortals({ isMain, feed, records, card, onDomWrite, isConnected = () => true, onConverted = () => {},
    append, remember = () => {}, actionsOf }) {
    const census = new Map();
    let censusComplete = false;
    let bindings = {};
    let bindingsKnown = false;  // the first `task_bindings` is the baseline, not news
    const projects = new Map();
    // History reads newest pages first: a final can arrive before the older row that binds its card.
    const completions = new Map();
    // An admitted successor of a portal's work -> that portal's record, until the successor's card is here.
    const follows = new Map();
    // Followed successors whose own record is being read (one read at a time each).
    const hydrating = new Set();
    // `predecessor|root` pairs whose record already said whether that root is its Continue.
    const asked = new Set();

    /** The live activity a card's controls act on when its work runs in another chat (its room). */
    const activity = (record) => (isPortal(record) ? record.portalActivity || undefined : undefined);

    // The census line was the work's present, not its result. When the work leaves the census the portal
    // lets it go, so a settled portal reads the same whether this page watched the work or opened after
    // it: a cause line for warnings or failure (the outcome writes it), none for a clean Done (the result
    // is the author's final row in the feed).
    function dropLine(record) {
        if (!record.collapsedActivity) return false;
        record.collapsedActivity = '';
        return renderCollapsedActivity(record, '');
    }

    // A portal's title is renamed by its naming owner (task_episodes.js `renamed`), and any other writer is met at the
    // next census: the Project reference names itself again, `Project` beside the same words.
    function relabel(record) {
        if (isPortal(record) && record.portalRefEl) nameProjectReference(record.portalRefEl);
    }

    function present(record) {
        if (!isPortal(record) || record.finished) return false;
        relabel(record);
        const row = census.get(record.groupId);
        if (!row) {
            const had = Boolean(record.portalActivity);
            record.portalActivity = null;
            let dropped = false;
            if (censusComplete && record.root?.isConnected) {
                dropped = dropLine(record);
                card.missing(record.groupId);
            }
            return Boolean(card.controls(record) || had || dropped);
        }
        const facts = censusTaskFacts(row);
        if (facts.ended) {
            record.portalActivity = null;
            dropLine(record);
            return card.finish(record.groupId, facts.outcome);
        }
        record.portalActivity = row;
        const startedSec = Number(row.started_at);
        if (Number.isFinite(startedSec) && startedSec > 0) record.startedAt = startedSec * 1000;
        card.restore(record, row.project_admission_hold);
        let changed = Boolean(syncParkedPhase(record, row.phase, row));
        const now = portalPresent(row);
        if (now.line) {
            record.collapsedActivity = boundActivityPreview(now.line);
            changed = renderCollapsedActivity(record, record.collapsedActivity) || changed;
        }
        if (now.lineAt) {
            record.latestActivityRaw = now.lineAt;
            record.latestActivityTs = normalizeLogTs(now.lineAt);
        }
        if (Number.isInteger(now.toolCalls)) record.toolCalls = now.toolCalls;
        changed = card.meta(record, taskCostProjection(row, now.lineAt || '')) || changed;
        return Boolean(card.controls(record) || changed);
    }

    // The author's final reached Main (the folded message stays an ordinary message at its time). Its
    // portal settles from the row's own typed outcome when the host stamped one, else from the durable
    // task detail — never from the bare lifecycle word, which cannot tell Done from Failed.
    function settle(row) {
        const id = key(row?.task_id);
        if (!id) return false;
        const record = records.get(id);
        if (!isPortal(record)) {
            completions.set(id, row);
            while (completions.size > 200) completions.delete(completions.keys().next().value);
            // A final of a work Main holds no card for may be a portal's Continue that ended unseen.
            if (!record?.root?.isConnected) askAbout(id, key(row.project_id));
            return false;
        }
        completions.delete(id);
        if (record.finished) return false;
        if (row.outcome_axes) return card.summary({ ...row, outcome_final: true });
        card.missing(id);
        return false;
    }

    /** One door for every binding source; true when the DOM changed. */
    function bind(record, project = {}) {
        if (!isMain || !record || record.isSubagent || !project?.id) return false;
        const fresh = !isPortal(record);
        let changed = bindWorkPortal(record, { id: project.id, name: project.name || projects.get(project.id)?.name || '' });
        if (!fresh) return changed;
        refitCollapsedActivity(record);  // a portal shows two lines, not three
        // The room holds the timeline: a portal is never shown expanded, and it no longer converts.
        changed = card.collapse(record) || changed;
        if (record.turnProjectBtn) { record.turnProjectBtn.remove(); record.turnProjectBtn = null; changed = true; }
        const completion = completions.get(record.groupId);
        if (completion) changed = settle(completion) || changed;
        changed = present(record) || changed;
        changed = card.meta(record) || changed;
        reconcilePortalNotes(feed);
        return changed;
    }

    // The successor's census row names its Project: its card is minted as that Project's portal, at its own
    // start, and the episode chain then gives it the work's place and first name. No row, no card.
    function adopt(successorId) {
        const row = census.get(successorId);
        const projectId = key(bindings[successorId]?.project_id) || key(row?.project_id);
        if (!isPortal(follows.get(successorId)) || !row || !projectId) return false;
        follows.delete(successorId);
        const record = card.get(successorId);
        if (!record || record.isSubagent) return false;
        // A queued row may not know its start (0): stamping that would pin the card above the whole feed.
        const startedSec = Number(row.started_at);
        if (startedSec > 0) card.reanchor(record, startedSec * 1000);
        const changed = bind(record, { id: projectId, name: key(row.project_name) });
        card.show(record);
        return Boolean(changed || record.root.isConnected);
    }

    // A successor the census does not list (it may have ended before any census saw it) keeps its own durable record,
    // read through the existing task read; that record is then its whole present here. Read once: an absent record
    // keeps the old card, a failed read asks again at the next census.
    function hydrate(successorId) {
        if (census.has(successorId) || !follows.has(successorId) || hydrating.has(successorId)) return;
        hydrating.add(successorId);
        void fetchTaskDetailStrict(successorId).then((detail) => {
            if (detail) onDomWrite(() => adoptRecord(successorId, detail));
            else follows.delete(successorId);
        }, () => {}).finally(() => hydrating.delete(successorId));
    }

    // The record names the successor's Project and outcome: its card is minted as that Project's portal at its own
    // start, settled from its durable outcome, and the episode chain gives it the work's place and first name. A
    // record that names no Project is no portal's successor: the old card stays.
    function adoptRecord(successorId, detail) {
        if (census.has(successorId) || !isPortal(follows.get(successorId))) return false;
        const projectId = key(bindings[successorId]?.project_id) || key(detail?.project_id);
        follows.delete(successorId);
        const record = projectId ? card.get(successorId) : null;
        if (!record || record.isSubagent) return false;
        const startedAt = Date.parse(String(detail.terminal_time?.attempt?.started_at || ''));
        if (Number.isFinite(startedAt)) card.reanchor(record, startedAt);
        let changed = bind(record, { id: projectId, name: key(detail.project_name) });
        if (!record.finished && isTerminalTaskDetail(detail)) changed = card.summary({ ...detail, task_id: successorId }) || changed;
        card.show(record);
        return Boolean(changed || record.root.isConnected);
    }

    /** The episode chain's hand-off: a portal's admitted successor, which runs where Main sees no frame of it. */
    function follow(successorId, predecessor) {
        const id = key(successorId);
        if (!isMain || !id || !isPortal(predecessor)) return false;
        follows.set(id, predecessor);
        const adopted = Boolean(onDomWrite(() => adopt(id)));
        if (!adopted) hydrate(id);
        return adopted;
    }

    // Read a settled card's own record once for one root that may be its Continue; the card's normal detail path
    // (`card.note`) then folds an admitted link. An absent record is an answer; a failed read, or a claim on this
    // root still bound, asks again.
    function ask(record, rootId) {
        const pair = `${record?.groupId}|${rootId}`;
        if (!record || !rootId || asked.has(pair)) return;
        asked.add(pair);
        void fetchTaskDetailStrict(record.groupId).then((detail) => {
            if (detail?.continued_by?.successor_task_id === rootId && !admittedSuccessor(detail)) asked.delete(pair);
            if (detail && record.root?.isConnected) onDomWrite(() => card.note(record, detail));
        }, () => asked.delete(pair));
    }

    // The owner's own press on a card here (task_continue.js) names its successor at once: read the record now, so
    // the successor joins the same card when its first frame lands, not at the next census.
    feed?.addEventListener?.('ouro:task-continued', (event) => {
        const record = records.get(key(event?.detail?.taskId));
        if (record?.root?.isConnected) ask(record, key(event.detail.successorId));
    });

    // A Continue made anywhere else (another chat or tab, the mind) is written on the predecessor's own record. A work
    // that may be its successor (a new root in the census, or, for a portal, a new binding or a final of a work Main
    // holds no card for) is when a settled card here asks: one whose Continue control names no successor yet, or that
    // work. A root whose frames reach this chat (a room, Main's own work) pairs with a card of this chat; a cardless one
    // only with a Main portal of the Project it names. Project membership alone never joins the cards: only the
    // record's admitted link does.
    function askAbout(rootId, projectId = '') {
        if (!rootId || follows.has(rootId)) return;
        const candidate = records.get(rootId);
        const here = Boolean(candidate?.root?.isConnected);
        for (const record of records.values()) {
            if (record.isSubagent || !record.finished || !record.root?.isConnected || record.root.dataset.episodeOf
                || record === candidate) continue;
            const scoped = here ? !isPortal(record) && !candidate.episodeName
                : isMain && isPortal(record) && Boolean(projectId) && projectId === record.root.dataset.projectId;
            const named = scoped ? continueControlSuccessor(record) : null;
            if (named !== null && (!named || named === rootId)) ask(record, rootId);
        }
    }

    function askContinued() {
        for (const row of census.values()) {
            if (row?.kind === 'managed_task' && !row.parent_task_id) askAbout(key(row.activity_id), key(row.project_id));
        }
    }

    /** `/api/state`: the census of EVERY chat (a portal's work runs in its room's chat), bindings and names (Main). */
    function sync(data) {
        if (!data || typeof data !== 'object') return;
        if (Array.isArray(data.active_chat_activities)) {
            census.clear();
            for (const row of data.active_chat_activities) {
                const id = key(row?.activity_id);
                if (id) census.set(id, row);
            }
            censusComplete = data.active_chat_activities_complete === true && data.supervisor_ready === true && isConnected();
        }
        if (!isMain) return askContinued();  // a room has no portals; its cards still follow a Continue
        if (Array.isArray(data.projects)) for (const project of data.projects) if (project?.id) projects.set(project.id, project);
        let fresh = [];
        if (data.task_bindings && typeof data.task_bindings === 'object') {
            if (bindingsKnown) fresh = Object.keys(data.task_bindings).filter((id) => !(id in bindings));
            bindings = data.task_bindings;
            bindingsKnown = true;
        }
        onDomWrite(() => {
            let changed = false;
            for (const record of records.values()) {
                if (record.isSubagent || !record.root?.isConnected) continue;
                const row = census.get(record.groupId);
                const projectId = key(bindings[record.groupId]?.project_id) || key(row?.project_id);
                if (!isPortal(record) && projectId) {
                    changed = bind(record, { id: projectId, name: key(row?.project_name) }) || changed;
                } else changed = present(record) || changed;
            }
            for (const successorId of [...follows.keys()]) changed = adopt(successorId) || changed;
            return reconcilePortalNotes(feed) || changed;
        });
        for (const successorId of [...follows.keys()]) hydrate(successorId);
        for (const id of fresh) if (!records.get(id)?.root?.isConnected) askAbout(id, key(bindings[id]?.project_id));
        askContinued();
    }

    /**
     * A `project_started` / `project_handoff` row says which work moved where: it binds that work's card at
     * the row's place (the request's position) instead of drawing a second entry.
     */
    function bindRow(row) {
        const taskId = key(row?.task_id);
        const projectId = key(row?.project_id);
        if (!isMain || !taskId || !projectId) return false;
        return onDomWrite(() => {
            const record = card.get(taskId);
            if (!record || record.isSubagent) return false;
            const name = key(row.task_name);
            if (name && !record.suggestedName) {
                record.suggestedName = name;
                if (record.titleEl.textContent !== name) record.titleEl.textContent = name;
            }
            if (row.ts) card.reanchor(record, row.ts);
            const changed = bind(record, { id: projectId, name: key(row.project_name) });
            card.show(record);
            // Mounted now: the owner's routing note it represents steps aside.
            reconcilePortalNotes(feed);
            return Boolean(changed || record.root.isConnected);
        });
    }

    // Conversion keeps the card: the same node becomes the work's portal (status, latest line, meta,
    // its controls and the Project reference), so the work never disappears from Main. The timeline
    // lives in the room from now on; the task keeps running and keeps its Stop.
    function converted(record, { project, handoff_receipt }) {
        return onDomWrite(() => {
            delete record.root.dataset.projectCreating;
            const changed = bind(record, project || {});
            // A binding whose Main receipt is not durable keeps its visible gap mark (the toast says why).
            if (handoff_receipt && !DURABLE_RECEIPTS.has(handoff_receipt)) record.root.dataset.receipt = handoff_receipt;
            onConverted();
            return changed;
        });
    }

    /** The owner's Turn into project: the same move the mind makes, then this card becomes its portal. */
    async function turnIntoProject(record) {
        if (!record || isPortal(record) || record.root?.dataset?.projectCreating === '1') return;
        const taskId = key(record.groupId);
        record.root.dataset.projectCreating = '1';
        // The control itself reports the in-flight conversion; Stop stays where it was, because the
        // task keeps running whatever the bind answers.
        const btn = record.turnProjectBtn;
        if (btn) { btn.disabled = true; btn.textContent = liveCardLabel.creatingProject(); }
        try {
            // One-click convert (owner P1): no name prompt, no extra LLM call. The SERVER names the
            // project and adopts the project the task's owner message already has.
            const payload = await apiClient.projectFromTask(taskId, projectIdFromTask(taskId), '');
            const project = payload?.project;
            if (!project?.id || payload?.binding?.project_id !== project.id) {
                throw new Error('Project binding response is unconfirmed; check the Project list before retrying.');
            }
            showToast(`${payload.adopted ? 'Project opened' : 'Project created'}: ${project.name || project.id}`, 'ok');
            window.dispatchEvent(new CustomEvent('ouro:project-created', { detail: { project } }));
            converted(record, payload);
            const notice = receiptNotice(payload.handoff_receipt);
            if (notice) showToast(notice, 'warn');
        } catch (exc) {
            showToast(`Project conversion not confirmed: ${exc.message || exc}`, 'error');
            delete record.root.dataset.projectCreating;
            if (btn) { btn.disabled = false; btn.textContent = liveCardLabel.turnIntoProject(); }
        }
    }

    /** The card's Turn into project, wanted on a Main root card with work whose origin is not bound yet. */
    function syncConvert(record, wanted) {
        if (wanted === Boolean(record.turnProjectBtn)) return false;
        if (!wanted) {
            record.turnProjectBtn.remove();
            record.turnProjectBtn = null;
            return true;
        }
        const actions = actionsOf(record);
        if (!actions) return false;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-sm btn-default ui-hit-area';
        btn.dataset.turnIntoProject = '1';
        btn.textContent = liveCardLabel.turnIntoProject();
        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            turnIntoProject(record);
        });
        actions.prepend(btn);
        record.turnProjectBtn = btn;
        return true;
    }

    /**
     * A host-stamped Project lifecycle row (ARCHITECTURE §3 Main rows), live and replayed alike. A start or transfer
     * row in Main only says where the work went: Main shows that work's own card (its portal), never a second entry
     * beside it; a room has no such row. A completion row is the author's folded final and also settles the portal.
     */
    function row(msg, text, opts) {
        if (isMain && ['project_started', 'project_handoff'].includes(msg.system_type) && msg.task_id && msg.project_id) {
            if (opts?.historyId) remember(`history:${opts.historyId}`);
            return bindRow(msg);
        }
        const added = append(msg, text, { ...opts, reply: msg });
        if (isMain && msg.system_type === 'project_completion_summary') settle(msg);
        return added;
    }

    return { activity, bind, bindRow, follow, present, relabel, row, settle, sync, syncConvert, turnIntoProject };
}
