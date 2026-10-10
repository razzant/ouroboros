/* Client-local notifications: pull the owner back to a question or a finished
   task while this client is running.

   Two halves, on purpose. The POLICY half below is pure over one frame plus the
   stored preferences, so the rules a reviewer must trust are testable without a
   DOM, a socket or a provider. The SHELL half owns the browser surfaces
   (permission, banner, tone, click) and holds no rules of its own.

   The subscription is CLIENT-level, not per chat instance. A chat instance dies
   with its room — closing a Project panel destroys it — so wiring notifications
   inside one would go silent in the very case they exist for: the owner left
   the window and the room is closed. `attach()` therefore subscribes once, on
   the shared socket, and decides from the frame plus the client's own room set.

   Replay safety is STRUCTURAL, not a persisted ledger: only live socket frames
   reach `handleFrame`, while history rendering, reconnect backfill and reload
   go through the chat instances' own readers, which never call it. So a reload
   cannot re-notify even though nothing about past notifications is stored. The
   bounded in-memory key set below collapses duplicates inside one live session
   — one finished task has several wire shapes, and a quiz frame can be
   delivered more than once.

   Preferences are client-local like the appearance choice (web/theme.js): the
   desktop window and each browser keep their own, nothing is sent to the
   server, and the controls carry no `s-` id, so the Settings collector never
   posts them.

   System notifications come first where the desktop app offers them
   (`show_native_notification`, launcher_background.DesktopApi). `submitted` means
   the system took the alert, in front or not, and owns its sound, so the page adds
   no tone and no toast; its click hands back a token that opens the same source a
   browser banner would. `unknown` means it was handed over and not answered in
   time: it may still appear, so the page adds nothing and keeps its click target.
   A refusal — not yet allowed, denied, unavailable, failed — or an app too old to
   have the method falls back to the browser banner or the in-app toast. Settings
   keeps the system's permission and what became of the last alert apart, so a
   failed hand-off never reads as a withdrawn permission. Only the owner's gesture
   (switching notifications on, Test) asks the system for permission.

   Not claimed: no OS permission prompt, Do Not Disturb setting or platform
   limit is bypassed. Where the Notification API is missing or denied, delivery
   degrades to the in-app toast and says so in Settings. */

import { TERMINAL_TASK_STATUSES } from './chat_activity.js';
import { getLogTaskGroupId } from './log_events.js';
import { shellBridgeApi } from './ui_helpers.js';
import { tr } from './i18n.js';

export const NOTIFY_PREFS_KEY = 'ouroboros.notifications';

/* Categories, in owner-facing order. The first two are the agreed REQUIRED
   signals (confirmed lifecycle state); `important` is the LLM-first one and
   `main_reply` is the separate, quieter toggle. */
export const NOTIFY_CATEGORIES = ['needs_answer', 'task_done', 'important', 'main_reply'];

/* Ordinary Main replies start OFF and message text starts HIDDEN. Both are my
   recommendation rather than an owner decision, and both are one click away. */
export const DEFAULT_NOTIFY_PREFS = Object.freeze({
    enabled: false,
    needs_answer: true,
    task_done: true,
    important: true,
    main_reply: false,
    sound: true,
    show_text: false,
});

export const NOTIFY_TITLES = Object.freeze({
    needs_answer: 'Ouroboros is waiting for your answer',
    task_done: 'Task finished',
    important: 'Message from Ouroboros',
    main_reply: 'Ouroboros replied',
});

// Titles are a four-word catalog outside the DOM: the install language reads them by code
// (web/modules/i18n.js); the body is the authored frame text and stays as written.
const notifyTitle = (category) => tr(`notify.title.${category}`, NOTIFY_TITLES[category]);

const BODY_CHARS = 140;
/* Bounded only for hygiene in a page that may live for days. A key evicted
   after this many distinct events could ring again if its frame reappeared much
   later; that is the disclosed tradeoff against unbounded growth. */
const SEEN_LIMIT = 2000;
const LINEAGE_LIMIT = 2000;
/* Click tokens of system notifications that were submitted or may have been; a
   click after a reload (or past this bound) only opens the window. */
const NATIVE_LIMIT = 200;
/* The web owner thread. A frame with no chat id at all is legacy Main traffic,
   exactly as the chat log gate reads it. */
const MAIN_CHAT_ID = 1;

const asBool = (value, fallback) => (typeof value === 'boolean' ? value : fallback);

/** Tolerant read of a stored preference object: unknown keys are dropped and a
 *  missing or malformed value keeps the shipped default for that key alone. */
export function normalizeNotifyPrefs(raw) {
    const source = raw && typeof raw === 'object' ? raw : {};
    const prefs = { ...DEFAULT_NOTIFY_PREFS };
    for (const key of Object.keys(DEFAULT_NOTIFY_PREFS)) {
        prefs[key] = asBool(source[key], DEFAULT_NOTIFY_PREFS[key]);
    }
    return prefs;
}

export function readNotifyPrefs(storage) {
    try {
        const raw = storage?.getItem?.(NOTIFY_PREFS_KEY);
        if (!raw) return { ...DEFAULT_NOTIFY_PREFS };
        return normalizeNotifyPrefs(JSON.parse(raw));
    } catch {
        // Unavailable or malformed storage is a gap, not a reason to notify
        // more than the owner asked for: fall back to the shipped defaults.
        return { ...DEFAULT_NOTIFY_PREFS };
    }
}

export function writeNotifyPrefs(storage, prefs) {
    const clean = normalizeNotifyPrefs(prefs);
    try {
        storage?.setItem?.(NOTIFY_PREFS_KEY, JSON.stringify(clean));
        return true;
    } catch {
        return false;
    }
}

const text = (value) => String(value == null ? '' : value);

/* A quiz frame carries its fields either flat or nested under `quiz`
   (chat_media.appendQuizMessage reads both shapes); use the same source. */
const quizSource = (frame) => (frame?.quiz && typeof frame.quiz === 'object' ? frame.quiz : frame);

function trimBody(value) {
    const body = text(value).replace(/\s+/g, ' ').trim();
    if (body.length <= BODY_CHARS) return body;
    return `${body.slice(0, BODY_CHARS - 1)}…`;
}

/* A reminder (a note Ouroboros wrote earlier, shown now by the host) and a
   granted skill's notice are System rows whose first line is the host's
   signature; the banner names the author from `source` and shows only the words
   after that line. Never the row's sentence in the title (§9: private by default). */
export function noticeTitle(systemType, source) {
    const author = text(source).trim();
    if (systemType === 'skill_notice') return author ? `Notice from ${author}` : 'Notice from a skill';
    return `Reminder from ${author || 'Ouroboros'}`;
}

function noticeBody(content) {
    const raw = text(content);
    const cut = raw.indexOf('\n');
    return cut >= 0 ? raw.slice(cut + 1) : raw;
}

/**
 * Classify ONE live frame into a notification candidate, or null.
 *
 * Pure. `kind` is the ws frame type as chat.js observed it ('chat', 'quiz' or
 * 'log'), `isMain` says whether the receiving instance is the Main thread, and
 * `isRoot`/`taskId` are the facts only the client holds: task lineage lives in
 * chat.js's own subagent map, and a log frame's task identity is resolved by
 * `getLogTaskGroupId`. `isRoot === false` is a positive statement that this is
 * a child; `undefined` means the caller could not tell.
 *
 * A child task never notifies the owner directly, so a child terminal is
 * dropped here on either of its two shapes: only root terminals and the frames
 * a parent itself emits reach a banner.
 *
 * A finished ROOT arrives in more than one live shape — the managed-queue
 * conclusion is a `log` frame with `type='task_done'`
 * (supervisor/events_task_done.py pushes it), while the authored summary
 * arrives as a chat row — and a turn's ordinary reply is the same ending. All
 * of them map to ONE key, `conclusion:<task_id>`, so whichever shape reaches
 * this client first rings and the others collapse.
 */
export function classifyLiveFrame(frame, { kind = 'chat', isMain = false, isRoot, taskId: taskIdHint = '' } = {}) {
    if (!frame || typeof frame !== 'object') return null;

    const taskId = text(taskIdHint) || text(frame.task_id);
    /* `isRoot === false` is a positive statement of child-ness; `undefined` only
       means the caller could not tell, and it is treated as eligible ON PURPOSE.
       Nothing on the wire ever declares a task to BE a root, so requiring proof
       of root-ness would silence every root terminal — the required category —
       to avoid a rare stray child banner. The disclosed residual (DESIGN §9) is
       the narrow case of a child whose very first observed frame is its own
       terminal AND which carries no delegation fact. */
    const notChild = isRoot !== false;
    const rootEligible = (id) => Boolean(id) && notChild;

    if (kind === 'quiz') {
        const source = quizSource(frame);
        const quizId = text(source.quiz_id || frame.quiz_id);
        if (!quizId || !notChild) return null;
        // An already answered/closed question must not ring on a late frame.
        const state = text(source.state || frame.quiz_state).toLowerCase();
        if (state && state !== 'open') return null;
        // `wait_for_answer === true` is the positive fact that the task is
        // actually blocked on the owner; anything else is an optional ask,
        // which belongs to the model-chosen `important` category.
        const waiting = source.wait_for_answer === true || frame.wait_for_answer === true;
        return {
            category: waiting ? 'needs_answer' : 'important',
            key: `quiz:${taskId}:${quizId}`,
            title: waiting ? notifyTitle('needs_answer') : tr('notify.title.asks_question', 'Ouroboros asks a question'),
            body: trimBody(source.question || frame.question || frame.content),
            target: { chatId: frame.chat_id, taskId, quizId },
        };
    }

    if (kind === 'log') {
        // Log frames carry runtime diagnostics, not owner-facing prose, so this
        // one stays title-only even when message text is switched on.
        if (text(frame.type || frame.event) !== 'task_done') return null;
        if (!rootEligible(taskId)) return null;
        // A task_done frame is not always an ending: an update or restart
        // teardown pushes a NON-settled `interrupted` terminal and requeues the
        // same task id afterwards. Ringing there would both lie and burn this
        // task's key, so its real completion would stay silent.
        if (!TERMINAL_TASK_STATUSES.has(text(frame.status).toLowerCase())) return null;
        // A DIRECT turn ends with this same frame: an ordinary Main reply, a
        // consciousness wake-up, any conversation turn. Calling that "a task
        // finished" would ring on every answer while the owner's ordinary-reply
        // toggle is off, which is the opposite of what the toggle promises. A
        // conversation's ending belongs to `main_reply` and nowhere else.
        if (frame._is_direct_chat === true) return null;
        return {
            category: 'task_done',
            key: `conclusion:${taskId}`,
            title: notifyTitle('task_done'),
            body: '',
            target: { chatId: frame.chat_id, taskId },
        };
    }

    if (kind !== 'chat') return null;
    if (frame.role !== 'assistant' && frame.role !== 'system') return null;
    if (frame.is_progress) return null;

    const systemType = text(frame.system_type);
    const rowKey = text(frame.card_row_id) || text(frame.client_message_id)
        || `${taskId}:${text(frame.ts)}`;

    // The model's own in-turn decision to interrupt the owner. The
    // discriminator already exists (tools/control_runtime._send_user_message
    // stamps it; `main_notice` is the same decision addressed to Main), so
    // importance needs no new host field and no second model call.
    if (systemType === 'proactive_message' || systemType === 'main_notice') {
        if (!notChild) return null;
        return {
            category: 'important',
            key: `important:${rowKey}`,
            title: notifyTitle('important'),
            body: trimBody(frame.content),
            target: { chatId: frame.chat_id, taskId },
        };
    }

    // A message for the owner that nobody speaks NOW: no new category, the same
    // `important` toggle (DESIGN §9).
    if (systemType === 'reminder' || systemType === 'skill_notice') {
        return {
            category: 'important',
            key: `important:${rowKey}`,
            title: noticeTitle(systemType, frame.source),
            body: trimBody(noticeBody(frame.content)),
            target: { chatId: frame.chat_id, taskId },
        };
    }

    // A child's row never reaches the owner, whatever else it carries.
    if (text(frame.delegation_role).toLowerCase() === 'subagent') return null;
    // Only the AUTHORED summary is a task's ending in chat form. A bare
    // `task_terminal_status` row is set exclusively for a DIRECT turn
    // (ouroboros/task_finalization.py), so reading it as a terminal would call
    // every stopped or failed conversation "a task finished" — it falls through
    // to the ordinary-reply category below instead.
    if (systemType === 'task_summary') {
        if (!rootEligible(taskId)) return null;
        return {
            category: 'task_done',
            key: `conclusion:${taskId}`,
            title: notifyTitle('task_done'),
            body: trimBody(frame.content),
            target: { chatId: frame.chat_id, taskId },
        };
    }

    // An ordinary finished reply in Main: no typed system row, no lifecycle
    // fact. Deliberately Main-only, as agreed.
    if (!systemType && frame.role === 'assistant' && isMain) {
        return {
            category: 'main_reply',
            // One turn ending is ONE event: an ordinary reply and the terminal
            // of the same task share this key, so the owner is pulled back once
            // rather than twice for the same conclusion.
            key: taskId ? `conclusion:${taskId}` : `main_reply:${rowKey}`,
            title: notifyTitle('main_reply'),
            body: trimBody(frame.content),
            target: { chatId: frame.chat_id, taskId },
        };
    }

    return null;
}

/**
 * Decide whether a classified candidate should actually be delivered.
 * Pure over (candidate, prefs, seenKeys) so the whole gate is testable.
 */
export function decideNotification(candidate, prefs, seenKeys) {
    if (!candidate) return { deliver: false, reason: 'not_notifiable' };
    const clean = normalizeNotifyPrefs(prefs);
    if (!clean.enabled) return { deliver: false, reason: 'disabled' };
    if (!NOTIFY_CATEGORIES.includes(candidate.category)) {
        return { deliver: false, reason: 'unknown_category' };
    }
    if (!clean[candidate.category]) return { deliver: false, reason: 'category_off' };
    if (seenKeys && typeof seenKeys.has === 'function' && seenKeys.has(candidate.key)) {
        return { deliver: false, reason: 'duplicate' };
    }
    return {
        deliver: true,
        reason: 'deliver',
        title: candidate.title,
        body: clean.show_text ? candidate.body : '',
        sound: clean.sound,
        category: candidate.category,
        key: candidate.key,
        target: candidate.target,
    };
}

/** Owner-facing one-line status for the Settings block. Pure. */
export function notifyStatusText({ enabled, supported, permission, storageAvailable = true } = {}) {
    if (!storageAvailable) {
        return 'This device blocks storage, so these choices apply until the window reloads.';
    }
    if (!enabled) return 'Notifications are off; nothing is requested from this system.';
    if (!supported) {
        return 'This client exposes no system notifications, so alerts appear in the app instead.';
    }
    if (permission === 'denied') {
        return 'This system denied notifications for this client, so alerts appear in the app instead. '
            + 'Allow them in the OS or browser settings to get banners.';
    }
    if (permission !== 'granted') {
        return 'Permission has not been granted yet; alerts appear in the app until it is.';
    }
    return 'System banners are enabled for this client.';
}

/** Explain the separate native-attention capability without calling it a banner. */
export function attentionStatusText({ enabled = true, nativeAttention = false, supported = false, status = '', bridge = false } = {}) {
    if (!enabled) return 'Notifications are off; no attention is requested from this system.';
    if (status === 'window_only') return 'Desktop attention can raise this window, but its system sound is unavailable; the app tone is used when needed.';
    if (status === 'unsupported' || status === 'unavailable') return 'Desktop attention is unavailable in this launcher; browser or in-app delivery remains available.';
    if (status === 'background') return 'Desktop attention is available; while Ouroboros runs in the background, alerts use a system banner or sound instead of opening this window.';
    if (status === 'native_sound' || nativeAttention) return 'Desktop attention is available; the launcher may raise this window and use the system sound.';
    if (bridge) return 'This desktop client exposes an attention bridge; its sound capability will be confirmed on the next alert.';
    if (supported) return 'Browser notifications are available; desktop attention depends on the client.';
    return 'This client has no native attention bridge; alerts stay inside the app.';
}

/** The system-notification line for a desktop app, or '' when the browser line applies. Pure.
 *  `native`: the app has `show_native_notification` (its absence, not a version, makes an app too old);
 *  `capability`: the system's permission as last answered, with any `limits` its server reported;
 *  `outcome`: what became of the last alert; `banner`: this client may show a browser banner, so a
 *  fallback is that banner or the in-app toast, never claimed as one of them. */
export function nativeStatusText({ enabled = true, shell = null, native = false, capability = null, outcome = null, banner = false } = {}) {
    if (!enabled || !shell?.desktop) return '';
    const app = `this desktop app${shell.version ? ` (${shell.version})` : ''}`;
    const fallback = banner ? 'a browser banner or the app' : 'the app';
    if (!native) {
        // Owner decision 3A: an app without the native method falls back, and says why.
        return `System notifications need the current Ouroboros desktop app; ${app} predates them, so alerts `
            + `fall back to ${fallback}. Updates inside Ouroboros do not replace the app itself.`;
    }
    const status = text(capability?.status);
    const why = (fact) => (text(fact?.reason) ? ` (${text(fact.reason)})` : '');
    if (status === 'not_determined' && text(capability?.reason)) {
        // The system's own error to its question: not the owner's denial (desktop_notifications.MacNotifier).
        return `The system could not ask whether Ouroboros may show notifications${why(capability)}, `
            + `so alerts fall back to ${fallback}.`;
    }
    if (status === 'not_determined') {
        return 'The system has not been asked yet whether Ouroboros may show notifications; Test asks it. '
            + `Until it is allowed, alerts fall back to ${fallback}.`;
    }
    if (status === 'denied') {
        return `The system denied notifications for Ouroboros, so alerts fall back to ${fallback}. `
            + 'Allow Ouroboros in the system notification settings to get them.';
    }
    if (status === 'unavailable') {
        return `System notifications are unavailable to ${app} here${why(capability)}, so alerts fall back to ${fallback}.`;
    }
    const last = text(outcome?.status);
    if (last === 'failed') {
        return `The last alert could not be handed to the system${why(outcome)}, so it fell back to ${fallback}.`;
    }
    if (last === 'unknown') {
        return 'The system did not confirm the last alert in time; it may still appear there, so the app did not repeat it.';
    }
    if (status === 'authorized' || last === 'submitted') {
        // A server that lacks an optional capability says so (Linux: `sound`, `actions`).
        const limits = Array.isArray(capability?.limits) ? capability.limits : [];
        const lacks = [limits.includes('no_sound') && 'plays no sound for them',
            limits.includes('no_click') && 'cannot open their source when clicked'].filter(Boolean);
        return 'System notifications are on: alerts go to the system, whose notification settings, Focus or '
            + 'Do Not Disturb decide how they appear and sound.'
            + (lacks.length ? ` This system's notification server ${lacks.join(' and ')}.` : '');
    }
    return '';
}

/* ---------------------------------------------------------------- shell ---- */

export function createNotifier({
    storage = globalThis.localStorage,
    notificationCtor = globalThis.Notification,
    audioContextCtor = globalThis.AudioContext || globalThis.webkitAudioContext,
    showToast = null,
    onActivate = null,
    focusWindow = () => globalThis.focus?.(),
    documentRef = globalThis.document,
    hostApi = null,
} = {}) {
    let prefs = readNotifyPrefs(storage);
    let storageAvailable = true;
    let destroyed = false;
    let audioCtx = null;
    let activate = onActivate;
    let toast = showToast;
    let nativeAttention = false;
    let attentionStatus = '';
    let shell = null; // the desktop app's own facts (desktop_shell.js), handed in by Settings
    let nativeCapability = null; // its system-notification permission, as last answered
    let nativeOutcome = null; // what became of the last alert handed to the system
    let tokenSeq = 0;
    const nativeTargets = new Map();
    const seen = new Set();
    /* Task lineage as the WIRE states it. A finished child and a finished root
       share one log shape with no lineage field
       (supervisor/events_task_done.py), so the only honest source is the
       delegation truth carried by the frames a subagent's own traffic brings.
       Learned client-wide, which is strictly more than any one room sees.
       Disclosed residual: a child whose FIRST observed frame is its own
       terminal cannot be recognised, and would notify once. */
    const childTasks = new Set();
    const attachDisposers = [];

    const boundedAdd = (set, value, limit) => {
        set.add(value);
        if (set.size > limit) set.delete(set.values().next().value);
    };

    function noteLineage(frame) {
        if (!frame || typeof frame !== 'object') return;
        const parent = text(frame.parent_task_id).trim();
        const child = text(frame.subagent_task_id).trim() || text(frame.task_id).trim();
        if (!child) return;
        const declared = text(frame.delegation_role).toLowerCase() === 'subagent';
        // A subagent's terminal log frame carries delegation-truth enrichment
        // (supervisor/subagent_task_truth.py stamps executor/substrate facts on
        // the pushed event); an ordinary root terminal does not. It is a second,
        // independent way to recognise a child whose earlier traffic this client
        // never saw.
        const delegated = Boolean(frame.executor_route || frame.actual_substrate
            || (frame.execution_evidence && typeof frame.execution_evidence === 'object'));
        if (!declared && !delegated && (!parent || parent === child)) return;
        boundedAdd(childTasks, child, LINEAGE_LIMIT);
    }

    const supported = () => typeof notificationCtor === 'function';
    const permission = () => (supported() ? text(notificationCtor.permission) || 'default' : 'unsupported');

    // Insertion-ordered: drop the oldest key, never the newest.
    const remember = (key) => boundedAdd(seen, key, SEEN_LIMIT);

    function tone() {
        if (!audioContextCtor) return;
        try {
            audioCtx = audioCtx || new audioContextCtor();
            audioCtx.resume?.();
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.value = 660;
            gain.gain.value = 0.05;
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            const now = audioCtx.currentTime || 0;
            osc.start(now);
            osc.stop(now + 0.12);
        } catch {
            // A device that refuses audio is not a delivery failure.
        }
    }

    function showBanner(decision) {
        try {
            // The OS owns the banner's sound; `silent` honours the toggle
            // so the owner never hears two sounds for one event.
            const note = new notificationCtor(decision.title, {
                body: decision.body || undefined,
                tag: decision.key,
                silent: !decision.sound,
            });
            note.onclick = () => {
                try { focusWindow(); } catch { /* a blocked focus is not fatal */ }
                try { activate?.(decision.target, decision); } catch { /* navigation is best-effort */ }
                try { note.close?.(); } catch { /* already closed */ }
            };
            return true;
        } catch {
            return false; // the in-app path takes over
        }
    }

    const bridge = () => hostApi || shellBridgeApi(globalThis);
    const nativeHost = () => {
        const api = bridge();
        return typeof api?.show_native_notification === 'function' ? api : null;
    };

    const PERMISSION_STATES = ['authorized', 'not_determined', 'denied', 'unavailable'];

    function noteCapability(answer) {
        const status = text(answer?.status);
        if (PERMISSION_STATES.includes(status)) {
            const limits = Array.isArray(answer?.limits) ? answer.limits.map(text) : [];
            nativeCapability = { ...nativeCapability, status, reason: text(answer?.reason), limits };
        }
        syncSettings();
    }

    /* One alert's answer: `submitted`, `unknown`, or `failed` for everything that never reached the
       system — a refusal, a broken bridge call, an answer this page cannot read. */
    function noteDelivery(answer) {
        const status = text(answer?.status);
        let outcome = 'failed';
        if (answer?.ok === true && status === 'submitted') {
            outcome = 'submitted';
            nativeCapability = { ...nativeCapability, status: 'authorized', reason: '' };
        } else if (status === 'unknown') {
            outcome = 'unknown';
        } else if (PERMISSION_STATES.includes(status)) {
            nativeCapability = { ...nativeCapability, status, reason: text(answer?.reason) };
        }
        nativeOutcome = { status: outcome, reason: outcome === 'submitted' ? '' : text(answer?.reason) };
        syncSettings();
        return outcome;
    }

    function deliver(decision) {
        const api = nativeHost();
        if (!api) return deliverInPage(decision);
        const token = `n${tokenSeq += 1}-${Date.now().toString(36)}`;
        nativeTargets.set(token, decision);
        if (nativeTargets.size > NATIVE_LIMIT) nativeTargets.delete(nativeTargets.keys().next().value);
        const settle = (answer) => {
            if (destroyed) return;
            // Submitted, or handed over without an answer: the system may show it, in front or not (2A),
            // and owns its sound (1A), so nothing else is shown and the click target stays.
            if (noteDelivery(answer) !== 'failed') return;
            nativeTargets.delete(token);
            deliverInPage(decision); // never reached the system: the typed fallback
        };
        try {
            void Promise.resolve(api.show_native_notification(
                decision.title, decision.body || '', Boolean(decision.sound), token,
            )).then(settle, () => settle(null));
        } catch { settle(null); }
        return 'native';
    }

    /** The desktop app hands back a clicked system notification's token (launcher_background.open_notification). */
    function activateNative(token) {
        const key = text(token);
        const decision = nativeTargets.get(key);
        if (!decision) return false;
        nativeTargets.delete(key);
        try { activate?.(decision.target, decision); } catch { /* navigation is best-effort */ }
        return true;
    }

    /* A legacy `notify_owner` bridge without `show_native_notification` cannot keep Sound off: hidden,
       (`request_attention` is the same method there) shows a Windows balloon with Windows' sound
       whatever the page asks, and a core update cannot change that frozen launcher. A silent alert
       therefore never asks it: the browser banner or the in-app toast, without a sound. */
    const attentionBridge = (decision) => {
        const api = bridge();
        return !decision.sound && !nativeHost() && typeof api?.notify_owner === 'function' ? null : api;
    };

    function deliverInPage(decision) {
        const banner = supported() && permission() === 'granted';
        const api = attentionBridge(decision);
        if (banner && typeof api?.notify_owner === 'function') {
            // A launcher that can hide its window on purpose answers first: hidden, its own native
            // signal is the one that opens the window (focus() cannot undo a hide); visible, it does
            // nothing (last argument) and the browser banner owns the sound, with no raise.
            const answer = (result) => {
                if (destroyed) return;
                if (result?.status === 'background') inApp(decision, api, result);
                else if (!showBanner(decision)) inApp(decision, api);
            };
            try {
                void Promise.resolve(api.notify_owner(Boolean(decision.sound), decision.title, decision.body || '', false))
                    .then(answer, () => answer(null));
            } catch { answer(null); }
            return 'host';
        }
        if (banner && showBanner(decision)) return 'banner';
        return inApp(decision, api);
    }

    function inApp(decision, api, answered = null) {
        if (answered || typeof api?.request_attention === 'function') {
            try {
                // A newer launcher takes the (privacy-filtered) text for its banner while its window is hidden on purpose.
                const cue = answered || (typeof api.notify_owner === 'function'
                    ? api.notify_owner(Boolean(decision.sound), decision.title, decision.body || '')
                    : api.request_attention(Boolean(decision.sound)));
                void Promise.resolve(cue).then((result) => {
                    if (destroyed) return;
                    nativeAttention = Boolean(result?.ok);
                    attentionStatus = String(result?.status || 'unavailable');
                    // A native banner owns its sound (queued: not reported as played); no page tone after it.
                    if (decision.sound && result?.sound_played !== true && result?.banner !== true) tone();
                    syncSettings();
                }).catch(() => { if (!destroyed && decision.sound) tone(); });
            } catch { if (decision.sound) tone(); }
        } else if (decision.sound) tone();
        const line = decision.body ? `${decision.title}: ${decision.body}` : decision.title;
        try {
            // The in-app surface must reach the source too, so the returned node
            // gets its own listener; the stack's own handler still dismisses it.
            const node = toast?.(line, 'info');
            node?.addEventListener?.('click', () => {
                try { activate?.(decision.target, decision); } catch { /* best-effort */ }
            });
        } catch { /* the toast stack owns its own errors */ }
        return 'in_app';
    }

    function handleFrame(frame, context) {
        if (destroyed) return null;
        // Lineage is learned even while notifications are off, so switching them
        // on mid-session does not start from an empty map.
        noteLineage(frame);
        if (!prefs.enabled) return null;
        const resolved = { ...context };
        if (resolved.isRoot === undefined) {
            const id = text(resolved.taskId) || text(frame?.task_id);
            if (id && childTasks.has(id)) resolved.isRoot = false;
        }
        const candidate = classifyLiveFrame(frame, resolved);
        const decision = decideNotification(candidate, prefs, seen);
        if (!decision.deliver) return null;
        remember(decision.key);
        const surface = deliver(decision);
        return { ...decision, surface };
    }

    function syncSettings(root = documentRef) {
        if (!root?.querySelectorAll) return;
        for (const input of root.querySelectorAll('[data-notify-pref]')) {
            const key = input.getAttribute('data-notify-pref');
            if (!(key in DEFAULT_NOTIFY_PREFS)) continue;
            input.checked = Boolean(prefs[key]);
            // Categories are meaningless while notifications are off, but the
            // owner's choices stay visible rather than being reset.
            if (key !== 'enabled') input.disabled = !prefs.enabled;
        }
        const native = Boolean(nativeHost());
        const facts = shell || (native ? { desktop: true } : null);
        const nativeLine = storageAvailable ? nativeStatusText({
            enabled: prefs.enabled, shell: facts, native, capability: nativeCapability, outcome: nativeOutcome,
            banner: supported() && permission() === 'granted',
        }) : '';
        for (const node of root.querySelectorAll('[data-notify-status]')) {
            const next = nativeLine || notifyStatusText({
                enabled: prefs.enabled,
                supported: supported(),
                permission: permission(),
                storageAvailable,
            });
            if (node.textContent !== next) node.textContent = next;
        }
        for (const node of root.querySelectorAll('[data-notify-attention-status]')) {
            const api = bridge();
            // While alerts go to the system, the attention cue is only their fallback: no second line
            // until an alert actually fell back to it.
            const systemOwns = native && text(nativeCapability?.status) === 'authorized'
                && text(nativeOutcome?.status) !== 'failed';
            const next = systemOwns ? '' : attentionStatusText({
                enabled: prefs.enabled,
                nativeAttention,
                supported: supported(),
                status: attentionStatus,
                bridge: typeof api?.request_attention === 'function',
            });
            if (node.textContent !== next) node.textContent = next;
        }
        for (const button of root.querySelectorAll('[data-notify-test]')) {
            button.disabled = !prefs.enabled;
        }
    }

    async function requestPermission() {
        const api = nativeHost();
        if (typeof api?.request_native_notifications === 'function') {
            // The desktop app's own question (macOS asks once); the browser API is not asked as well.
            try { noteCapability(await api.request_native_notifications()); } catch { /* the in-app path remains */ }
            return text(nativeCapability?.status) || 'unavailable';
        }
        if (!supported() || permission() === 'granted' || permission() === 'denied') return permission();
        try {
            const result = await notificationCtor.requestPermission?.();
            return text(result) || permission();
        } catch {
            return permission();
        }
    }

    async function setPref(key, value) {
        if (!(key in DEFAULT_NOTIFY_PREFS)) return prefs;
        prefs = normalizeNotifyPrefs({ ...prefs, [key]: Boolean(value) });
        storageAvailable = writeNotifyPrefs(storage, prefs);
        // Enabling is a user gesture: the only moment a permission prompt and
        // an audio context may legitimately be created.
        if (key === 'enabled' && prefs.enabled) {
            await requestPermission();
        }
        syncSettings();
        return prefs;
    }

    function test() {
        if (!prefs.enabled) return null;
        const decision = decideNotification({
            category: 'important',
            key: `test:${Date.now()}`,
            title: 'Ouroboros notifications are working',
            body: 'This is a test notification.',
            target: {},
        }, { ...prefs, important: true, show_text: true }, null);
        if (!decision.deliver) return null;
        if (typeof nativeHost()?.request_native_notifications !== 'function') return { ...decision, surface: deliver(decision) };
        // The Test click is the owner's gesture: the system may ask its one question first, then the
        // alert goes the ordinary way. A live alert never asks.
        void requestPermission().then(() => { if (!destroyed) deliver(decision); });
        return { ...decision, surface: 'native' };
    }

    const onChange = (event) => {
        const input = event?.target?.closest?.('[data-notify-pref]');
        if (!input) return;
        void setPref(input.getAttribute('data-notify-pref'), input.checked);
    };
    const onClick = (event) => {
        if (event?.target?.closest?.('[data-notify-test]')) test();
    };
    const onStorage = (event) => {
        if (event?.key && event.key !== NOTIFY_PREFS_KEY) return;
        prefs = readNotifyPrefs(storage);
        syncSettings();
    };

    documentRef?.addEventListener?.('change', onChange);
    documentRef?.addEventListener?.('click', onClick);
    globalThis.addEventListener?.('storage', onStorage);

    /* ONE subscription per client, on the shared socket. Rooms come and go; the
       socket and this notifier do not. `ownerVisibleChat` keeps machine traffic
       out: the hidden partition (chat 0) and A2A (negative) never notify, and
       an unknown positive chat id is not assumed to be the owner's. */
    function attach({ ws, ownerVisibleChat, mainChatId = MAIN_CHAT_ID } = {}) {
        if (!ws?.on) return () => {};
        const disposers = [];
        const known = (id) => Boolean(ownerVisibleChat ? ownerVisibleChat(id) : id === mainChatId);
        // A room the client knows, or — exactly as the Main thread itself reads
        // it (chat_activity.mainThreadAccepts) — any positive chat not stamped as
        // another Project's, which is how an external owner transport arrives.
        // The hidden partition and A2A ids are machine traffic and never notify.
        const projectStamped = (frame, envelope) =>
            Boolean(frame?.project_thread || envelope?.project_thread);
        const visible = (raw, frame, envelope) => {
            if (raw === undefined || raw === null || raw === '') return true;
            const id = Number(raw);
            if (!Number.isFinite(id) || id <= 0) return false;
            return known(id) || !projectStamped(frame, envelope);
        };
        const route = (frame, kind, envelope) => {
            if (destroyed || !frame || typeof frame !== 'object') return;
            try {
                const raw = frame.chat_id !== undefined && frame.chat_id !== null
                    ? frame.chat_id : envelope?.chat_id;
                if (!visible(raw, frame, envelope)) return;
                const id = Number(raw);
                const inProjectRoom = projectStamped(frame, envelope)
                    || (Number.isFinite(id) && id > 0 && id !== mainChatId && known(id));
                const context = { kind, isMain: !inProjectRoom };
                if (kind === 'log') context.taskId = getLogTaskGroupId(frame) || '';
                handleFrame(frame, context);
            } catch {
                /* This listener runs BEFORE the chat instances' own handlers and
                   ws.emit does not isolate them, so an error here would break
                   message rendering. A missed notification must never do that. */
            }
        };
        const on = (event, fn) => {
            const dispose = ws.on(event, fn);
            if (typeof dispose === 'function') disposers.push(dispose);
        };
        on('chat', (msg) => route(msg, 'chat', msg));
        on('quiz', (msg) => route(msg, 'quiz', msg));
        on('log', (msg) => route(msg?.data, 'log', msg));
        const release = () => { while (disposers.length) disposers.pop()?.(); };
        attachDisposers.push(release);
        return release;
    }

    return {
        get prefs() { return { ...prefs }; },
        attach,
        supported,
        permission,
        setPref,
        requestPermission,
        handleFrame,
        test,
        mountSettings: syncSettings,
        activateNative,
        configure({ onActivate: nextActivate, showToast: nextToast, shell: nextShell } = {}) {
            if (nextActivate) activate = nextActivate;
            if (nextToast) toast = nextToast;
            if (nextShell) {
                shell = nextShell;
                if (shell.native) nativeCapability = { ...shell.native };
            }
            syncSettings();
        },
        destroy() {
            destroyed = true;
            while (attachDisposers.length) attachDisposers.pop()?.();
            documentRef?.removeEventListener?.('change', onChange);
            documentRef?.removeEventListener?.('click', onClick);
            globalThis.removeEventListener?.('storage', onStorage);
            try { audioCtx?.close?.(); } catch { /* nothing to close */ }
            audioCtx = null;
            seen.clear();
            nativeTargets.clear();
        },
    };
}

/* One notifier per client: the dedupe set must be shared by the Main instance
   and every Project room, so a quiz delivered to both rings once. */
let singleton = null;

export function getNotifier(options) {
    if (!singleton) {
        singleton = createNotifier(options);
        // The desktop app calls this when one of its system notifications is clicked.
        globalThis.ouroNotifications = { activate: (token) => Boolean(singleton?.activateNative(token)) };
    }
    return singleton;
}

export function resetNotifier() {
    singleton?.destroy?.();
    singleton = null;
    delete globalThis.ouroNotifications;
}
