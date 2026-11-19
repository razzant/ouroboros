// A WKScriptMessage reply returns to this document. Only a stamped built-in main frame can
// open the shell's memory. Browser and Mini App keep the existing tab-scoped Storage.
import { apiClient, isStoreMigrating, withinRequestLifetime } from './api_client.js';

let desktop = null, opening = null;
const ACTIVE_PROJECT = 'ouro_chat_active_project';
const CHOOSE_ROOM = 'Choose Main or another Project to continue.';
export const roomOpening = () => opening;
export const whenRoomOpens = read => opening ? opening.then(read) : read();
export const roomStorage = () => desktop || globalThis.sessionStorage;
export const desktopRoomStorage = () => desktop;

const waitForProject = () => new Promise(resolve => setTimeout(resolve, 700));
// One request (including its body) may fail to respond; observed startup can take many such reads.
const bounded = (read, signal) => withinRequestLifetime(read, { signal, lifetimeMs: 12_000 });
const readBoot = signal => bounded(async signal => {
    const health = await apiClient.health({ signal });
    if (health?.error) throw new Error(health.error);
    return health?.boot;
}, signal);
const unavailableBoot = error => {
    if (error.status && error.status < 500) throw error;
    return null;
};

const startupFailed = () => Object.assign(new Error('Project startup failed'), { name: 'ProjectStartupFailed' });

async function readStartupProject(id, current, progress, signal) {
    const live = () => !signal.aborted && current();
    if (!progress) {
        // After a signed launcher exec the first window is already visible. Its saved Project
        // can precede registry reconciliation, even when the server would currently answer [].
        while (live()) {
            const boot = await readBoot(signal).catch(async error => {
                unavailableBoot(error);
                if (!live()) return null;
                await waitForProject();
                if (live()) return readBoot(signal).catch(unavailableBoot);
            });
            if (!live()) return;
            if (boot?.phase === 'failed') throw startupFailed();
            if (!['starting', 'serving'].includes(boot?.phase)
                || !Array.isArray(boot.steps)
                || boot.steps.some(step => step?.name === 'projects' && ['done', 'failed'].includes(step.state))) break;
            await waitForProject();
        }
    }
    let retried = false;
    if (live()) progress?.(id, 'mount');
    while (live()) {
        try {
            const data = await bounded(signal => apiClient.projectById(id, { signal }), signal);
            if (!live()) return;
            if (Array.isArray(data?.projects) || retried) return data;
            progress?.(id, 'retry');
            retried = true;
        } catch (error) {
            if (!live()) return;
            if (isStoreMigrating(error)) {
                await progress?.(id, 'migrating');  // present the responsive shell before reading health
                if (!live()) return;
                // The exact read says migration. A missing or unchanged cache cannot withdraw that fact.
                const health = readBoot(signal);
                const boot = await (progress ? health.catch(unavailableBoot) : health);
                if (!live()) return;
                if (boot?.phase === 'failed' || boot?.jobs?.dialogue_import?.state === 'failed') throw startupFailed();
                if (!boot && !progress) throw error; // older flat session was already on screen
                retried = false;
            } else {
                if (retried || error.name === 'StateReadEnded' || (error.status && error.status < 500)) throw error;
                progress?.(id, 'retry');
                retried = true;
            }
        }
        await waitForProject();  // repeat only this exact read; the panel is mounted once
    }
}

export function openDesktopRoomSession(host, doc, warn) {
    const meta = doc.querySelector?.('meta[name="ouroboros-desktop-document"]');
    if (!meta || host.top !== host || doc.documentElement.dataset.ouroborosHost === 'telegram') return null;
    const channel = host.webkit?.messageHandlers?.ouroborosRoomSession;
    if (!channel) return null;
    const panic = host.webkit.messageHandlers.ouroborosDesktopPanic;
    const ticket = meta.content;
    function post(body) {
        try {
            const request = channel.postMessage({ ...body, ticket });
            if (body.op === 'open') meta.remove();  // dispatched, even if WebKit still owes the reply
            return Promise.resolve(request);
        }
        catch (error) { return Promise.reject(error); }
    }
    const values = new Map();
    const changes = new Map();
    const submitting = new Set();
    let appMounted, choose, chosen, inputReady = false, warned = false;
    const app = new Promise(resolve => { appMounted = resolve; });
    const choice = new Promise(resolve => { choose = () => resolve({ explicit: true }); });
    const inputs = new Set();
    const failed = () => {
        if (!warned) warn('This window could not hand its room input to the desktop shell. Keep it open to recover your draft.');
        warned = true;
    };
    // Flush one input event as a unit. Send consumes the draft and records the held frame in the same turn.
    function flush() {
        if (opening || !changes.size) return;
        const batch = JSON.stringify([...changes]);
        changes.clear();
        post({ op: 'update', changes: batch }).catch(failed);
    }
    function change(key, value) {
        if (value === null ? !values.has(key) : values.get(key) === value) return;
        if (value === null) values.delete(key); else values.set(key, value);
        if (!changes.size) queueMicrotask(flush);
        changes.set(key, value);
    }
    desktop = {
        mounted: appMounted,
        warn,
        panicTicket: ticket,
        panic: typeof panic?.postMessage === 'function' ? () => panic.postMessage(ticket) : null,
        choice,
        get chosen() { return chosen !== undefined; },
        inputLocked: () => !inputReady,
        onInputReady(read) {
            inputs.add(read);
            return () => inputs.delete(read);
        },
        enableInput() {
            if (opening || desktop.refused || inputReady) return;
            inputReady = true;
            for (const read of inputs) read();
            inputs.clear();
        },
        selectRoom(projectId) {
            chosen = projectId || null;
            desktop.returnToProject = null;
            choose();
            if (!opening) change(ACTIVE_PROJECT, chosen);
            desktop.enableInput();
        },
        trackSend(work) {
            submitting.add(work);
            const finished = () => submitting.delete(work);
            work.then(finished, finished);
        },
        getItem: key => values.get(key) ?? null,
        setItem: (key, value) => change(key, String(value)),
        removeItem: key => change(key, null),
        mayFlushChat: () => post({ op: 'resume_queue' }).catch(() => false),
    };
    const editors = new Set();
    desktop.editor = (name, dirty) => {
        if (editors.has(name) === dirty) return;
        if (dirty) editors.add(name); else editors.delete(name);
        post({ op: 'editor', name, dirty }).catch(failed);
    };
    host.addEventListener('ouro:renderer-holds-edits', () =>
        warn('Save or copy your edits in ' + [...editors].join(', ') + ', then finish editing to replace this window.'));
    host.addEventListener('ouro:renderer-incompatible', () =>
        warn('The selected interface cannot restore room input. Copy and clear your drafts and replies, remove staged files, and resolve any unconfirmed sends. Ouroboros will try the new window again. Built-in setup is available in Help.'));
    host.addEventListener('ouro:renderer-refused', () =>
        warn('The new window could not open. You can keep using this window or use Help to open built-in setup.'));
    host.addEventListener('ouro:room-input-unconfirmed', () =>
        warn('The previous window did not confirm its last input. Check your draft, reply and room history before sending. Other unsaved forms may need attention.'));
    host.addEventListener('ouro:room-handoff', async event => {
        if (opening) return;
        // A pressed Send can still be saving its settings or uploading before it
        // has an ID. AppKit's one deadline also covers that original operation.
        if (submitting.size) await Promise.allSettled(submitting);
        flush();
        post({ op: 'drained', handoff: event.detail, items: JSON.stringify(Object.fromEntries(values)),
            editors: [...editors] }).catch(() => {}); // a late answer belongs to the completed exchange
    });
    host.addEventListener('ouro:room-restore-failed', () =>
        warn('This window could not restore the previous room input. Check the room history before sending again.'));
    opening = post({ op: 'open', staged: true }).then(async saved => {
        if (Object.hasOwn(saved, 'mountProject')) {
            const mount = await app;
            const mounted = await mount(saved.mountProject, (project, stage) => post({ op: 'project_importing', project, stage }));
            // A neutral shell acknowledges the saved address without claiming that its room is open.
            saved = await post({ op: 'ready', project: saved.mountProject, ...(!mounted && { selection: true }) });
        }
        for (const [key, value] of Object.entries(saved)) if (!changes.has(key)) values.set(key, value);
        if (chosen !== undefined) change(ACTIVE_PROJECT, chosen);
    }).catch(() => { desktop.refused = true; failed(); }).finally(() => { opening = null; flush(); });
    return opening;
}

export function finishDesktopStartup(shownRoom, openProject, connect) {
    if (!desktop) return connect();
    const session = desktop;
    const initialized = room => {
        const field = room?.composer?.input;
        return field?.isConnected && !field.disabled && (!field.readOnly || field.dataset?.desktopRoomOpening === '1');
    };
    const finishMount = ready => { if (ready) session.enableInput(); };
    let attempt;
    async function mount(id, progress, presentShell, returning = false) {
        if (session.chosen) return true;
        attempt?.abort();
        const readScope = new AbortController();
        attempt = readScope;
        const stopRead = () => readScope.abort();
        let shellOffered = false;
        let unavailableRoom = false;
        if (id) session.returnToProject = () => {
            if (!unavailableRoom && !session.chosen && !session.refused) {
                void mount(id, progress, presentShell, true).then(finishMount, () => session.warn(CHOOSE_ROOM));
            }
        };
        const obsolete = () => { if (!shellOffered) stopRead(); };
        const report = progress && (returning ? () => {} : (project, stage) => {
            if (shellOffered) return;
            if (stage === 'migrating') {
                if (!initialized(shownRoom())) throw Error('Room input did not initialize');
                shellOffered = true;
                presentShell(false);
                return opening.then(() => { if (session.refused) stopRead(); });
            }
            // Progress only guards the hidden document; it never holds up HTTP or owns an installed view.
            progress(project, stage).then(accepted => { if (!accepted) obsolete(); }, obsolete);
        });
        const { room, explicit, issue } = id ? await Promise.race([session.choice, new Promise((resolve, reject) => {
            const read = (key, current) => readStartupProject(key, current, report, readScope.signal);
            Promise.resolve(openProject(id, room => resolve({ room }), read,
                (issue, cause) => resolve({ issue: cause?.name === 'ProjectStartupFailed' ? 'startup-failed' : issue }), returning))
                .then(() => resolve({ room: null }), reject);
        })]).catch(error => {
            if (attempt === readScope) session.returnToProject = null;
            unavailableRoom = true;
            throw error;
        }).finally(stopRead) : { room: shownRoom() };
        if (attempt !== readScope || session.refused) return false;
        // Another page may retire the read, not the pending selection. A Back can still restore it.
        const shown = shownRoom();
        if (id && !session.chosen && !issue) {
            if ((shellOffered || returning) && (!shown || !room)) return false;
            if (!progress && !shown) return false;  // a flat snapshot already has a visible document
        }
        session.returnToProject = null;
        if (explicit || session.chosen) return true;
        if (room === shown && room?.projectId === id && initialized(room)) return true;
        unavailableRoom = true;
        if (!issue || (!shellOffered && !returning && !initialized(shownRoom()))) throw Error('Room input did not initialize');
        session.warn((issue === 'missing' ? 'Your last Project is no longer available.'
            : issue === 'deleting' ? 'Your last Project is being deleted.'
            : issue === 'startup-failed' ? 'Startup reported a failure while restoring your last Project.'
            : 'Your last Project could not be read.')
            + ' ' + CHOOSE_ROOM);
        return false;
    }
    let staged, restored;
    session.mounted(async (id, progress) => {
        let presentShell;
        const shell = new Promise(resolve => { presentShell = resolve; });
        restored = mount(id, progress, presentShell);
        return staged = await Promise.race([restored, shell]);
    });
    return whenRoomOpens(async () => {
        await Promise.resolve();  // composers mounted for a staged room registered their hydration after this callback
        if (session.refused) return;
        const mounting = restored ?? mount(session.getItem(ACTIVE_PROJECT) || '');
        if (staged || (restored && session.chosen)) session.enableInput();
        connect();  // Already presented, either by the native receipt or by source exec.
        try { finishMount(await mounting); }
        catch { session.warn(CHOOSE_ROOM); }
    });
}

export function markDesktopEditor(name, dirty) {
    desktop?.editor(name, Boolean(dirty));
}

export function desktopRestartNote() {
    return desktop ? '\n\nYour room text and reply return in the fresh window. Reattach unsent files. Save any other open forms first.' : '';
}
