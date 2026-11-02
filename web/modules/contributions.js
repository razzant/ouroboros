// Skills' contributions in the page (docs/CLIENT_CONTRACT.md §11 "Contributions"): one store of the live projection
// `GET /api/ui/contributions`, read at boot, on every socket (re)open, on `extension_lifecycle`, on the end of any skill
// lifecycle job (its `skill_lifecycle` progress row: an install from Browse, an update, a turn-on once ready, whoever
// started it) and on the page's own `ouro:skill-lifecycle`; overlapping triggers share one read and a failed read
// keeps the last good snapshot. `onChange` hears a changed projection; `onLifecycle` hears every such lifecycle end,
// for what the projection leaves out (an off skill's files). Its
// selectors answer per placement and per provider kind; placements follow the owner's arrangement
// (`preferences.contributions.hidden` and `order`, contributed items only: the host's own controls are not in this
// store) and leave out a contribution that is off for this page. The fault book is that "off": a contribution whose
// page module failed (host_modules.js) stays off, with its reason, until its skill's revision changes or the page
// reloads. `runContributionAction` performs a placed control's verb; no verb sends anything as the owner.
import { apiClient, jsonPost } from './api_client.js';
import { fmt, tr } from './i18n.js';
import { showToast } from './toast.js';
import { boundedText } from './utils.js';

// The phases a skill lifecycle job's last progress row carries (ouroboros/skill_lifecycle_queue.py `_notify_chat`).
const LIFECYCLE_ENDS = new Set(['completed', 'failed', 'cancelled']);

const EMPTY = Object.freeze({ contributions: [], offers: {}, preferences: {}, voice: {}, icons: [], problems: {} });

const objectOr = (value, fallback) => (value && typeof value === 'object' && !Array.isArray(value) ? value : fallback);

function normalized(payload) {
    return {
        contributions: Array.isArray(payload?.contributions) ? payload.contributions : [],
        offers: objectOr(payload?.offers, {}),
        preferences: objectOr(payload?.preferences, {}),
        voice: objectOr(payload?.voice, {}),
        icons: Array.isArray(payload?.icons) ? payload.icons : [],
        problems: objectOr(payload?.problems, {}),
    };
}

/** One store; `client.uiContributions()` is the read. Pages use the shared one (`getContributions`). */
export function createContributionsStore({ client = apiClient } = {}) {
    let snapshot = EMPTY;
    let signature = '';
    let loaded = false;
    let lastError = null;
    let reading = null;  // the read in flight
    let again = false;  // a trigger landed during it: that read runs once more before it settles
    let destroyed = false;
    const faults = new Map();  // key -> { revision, reason, skill, skill_label, label }
    const listeners = new Set();
    const lifecycleListeners = new Set();
    const detachers = new Set();

    function notify(set = listeners) {
        for (const listener of [...set]) {
            try { listener(store); } catch (error) { console.error('contributions listener failed', error); }
        }
    }

    async function readOnce() {
        try {
            const next = normalized(await client.uiContributions());
            loaded = true;
            lastError = null;
            const text = JSON.stringify(next);
            if (text === signature) return false;
            signature = text;
            snapshot = next;
            return true;
        } catch (error) {
            lastError = error;  // the last good snapshot stays
            return false;
        }
    }

    const byKey = (key) => snapshot.contributions.find((item) => item.key === key) || null;
    /** The reason `key` is off for this page, while the revision that failed is still the live one. */
    function fault(key) {
        const entry = faults.get(key);
        return entry && byKey(key)?.revision === entry.revision ? entry.reason : '';
    }

    function arranged(items) {
        const arrangement = objectOr(snapshot.preferences.contributions, {});
        const order = Array.isArray(arrangement.order) ? arrangement.order : [];
        const rank = (item) => (order.includes(item.key) ? order.indexOf(item.key) : order.length);
        // The projection already sorts by (order, key); the owner's order moves named items ahead, stably.
        return items.map((item, index) => ({ item, index }))
            .sort((a, b) => rank(a.item) - rank(b.item) || a.index - b.index)
            .map(({ item }) => item);
    }

    const store = {
        /** Read the projection now; resolves true when the snapshot changed. */
        refresh() {
            if (destroyed) return Promise.resolve(false);
            if (reading) { again = true; return reading; }
            reading = (async () => {
                let changed = false;
                try {
                    do {
                        again = false;
                        if (await readOnce()) changed = true;
                    } while (again && !destroyed);
                } finally { reading = null; }
                if (changed && !destroyed) notify();
                return changed;
            })();
            return reading;
        },
        /** Placed controls of one placement (`composer.plus`, `composer.chips`, `message.action`), arranged. */
        byPlacement(name) {
            const hidden = new Set(objectOr(snapshot.preferences.contributions, {}).hidden || []);
            return arranged(snapshot.contributions.filter((item) => item.placement === name
                && !hidden.has(item.key) && !fault(item.key)));
        },
        /** Every live provider of one kind (`stt`, `tts`), off ones included: `fault(key)` says why one is off. */
        providers: (kind) => snapshot.contributions.filter((item) => item.provides === kind),
        /** Bundled providers that are installed but not live (`offers.stt`). */
        offers: (kind) => (Array.isArray(snapshot.offers[kind]) ? snapshot.offers[kind] : []),
        /** The saved engine choice of one kind: `{key, available, reason}`, or null before a read. */
        voice: (kind) => snapshot.voice[kind] || null,
        /** Contributions whose own page module runs in this page (`render.kind: "host"`). */
        hostEntries: () => snapshot.contributions.filter((item) => item.render?.kind === 'host' && item.render.url),
        byKey,
        fault,
        preferences: () => snapshot.preferences,
        icons: () => snapshot.icons,
        snapshot: () => snapshot,
        status: () => ({ loaded, error: lastError ? String(lastError.message || lastError) : '' }),
        /** Turn one contribution off for this page (its module failed); the record stays for Settings and Skills. */
        markFault(item, reason) {
            if (!item?.key) return;
            faults.set(item.key, { revision: item.revision, reason: String(reason || 'unknown error'),
                skill: item.skill, skill_label: item.skill_label, label: item.label });
            notify();
        },
        faults: () => [...faults].map(([key, entry]) => ({ key, ...entry, current: Boolean(fault(key)) })),
        onChange(listener) {
            listeners.add(listener);
            return () => listeners.delete(listener);
        },
        /** Called at each lifecycle trigger below, changed projection or not; never at boot or a socket open. */
        onLifecycle(listener) {
            lifecycleListeners.add(listener);
            return () => lifecycleListeners.delete(listener);
        },
        /** Read at boot and on every trigger that may change what is live. Returns the disposer. */
        attach({ ws = null, target = globalThis.window } = {}) {
            const read = () => { void store.refresh(); };
            const lifecycle = () => { read(); notify(lifecycleListeners); };
            const ended = (msg) => {
                const phase = (msg?.lifecycle || msg?.progress_meta?.lifecycle)?.phase;
                if (msg?.system_type === 'skill_lifecycle' && LIFECYCLE_ENDS.has(phase)) lifecycle();
            };
            const offs = [ws?.on?.('open', read), ws?.on?.('extension_lifecycle', lifecycle), ws?.on?.('chat', ended)];
            target?.addEventListener?.('ouro:skill-lifecycle', lifecycle);
            offs.push(() => target?.removeEventListener?.('ouro:skill-lifecycle', lifecycle));
            const detach = () => {
                for (const off of offs) if (typeof off === 'function') off();
                detachers.delete(detach);
            };
            detachers.add(detach);
            read();
            return detach;
        },
        destroy() {
            destroyed = true;
            for (const detach of [...detachers]) detach();
            listeners.clear();
            lifecycleListeners.clear();
        },
    };
    return store;
}

let shared = null;

/** The page's one contributions store. */
export function getContributions() {
    if (!shared) shared = createContributionsStore();
    return shared;
}

/**
 * The draft of the room `room()` names (its composer: `getDraft`, `insertAtCursor`, `setDraft`), for verbs and page
 * modules. Each call answers false when that room has no composer; nothing here sends.
 */
export function draftPort(room) {
    const composer = () => room()?.composer || null;
    return {
        text: () => composer()?.getDraft?.() ?? '',
        insert(text) {
            const target = composer();
            if (!target?.insertAtCursor) return false;
            target.insertAtCursor(String(text ?? ''));
            return true;
        },
        clear() {
            const target = composer();
            if (!target?.setDraft) return false;
            target.setDraft('');
            return true;
        },
    };
}

/** A contribution's `open.tab`: the Widgets page, at that skill's card (app.js listens). */
export function openSkillWidgets(skill) {
    globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:open-widgets', { detail: { skill: String(skill || '') } }));
}

const defaultNotice = (text, tone = 'info') => showToast(text, tone);

/**
 * Perform a placed control's verb (`draft.insert`, `skill.request`, `open.tab`, `notice`). `draft` is the visible
 * room's draft (`draftPort`), `row` the message a `message.action` was pressed on (message_actions.js `messageRow`),
 * `notice(text, tone)` the one honest line. Resolves true when the verb did its work; a failure says so in one line
 * and leaves the draft as it was. `skill.request` clears the draft only after a 2xx answer, and only while the draft
 * is still the text it sent.
 */
export async function runContributionAction(item, { draft = null, row = null, notice = defaultNotice,
    post = jsonPost, openTab = openSkillWidgets } = {}) {
    const action = item?.action || {};
    const label = item?.label || item?.skill_label || item?.skill || '';
    if (action.verb === 'draft.insert') {
        if (draft?.insert(action.text)) return true;
        notice(tr('contributions.no_draft', 'There is no message field here to put this in.'), 'warn');
        return false;
    }
    if (action.verb === 'notice') {
        notice(boundedText(action.text || '', 300), 'info');
        return true;
    }
    if (action.verb === 'open.tab') {
        openTab(item.skill);
        return true;
    }
    if (action.verb !== 'skill.request') {
        notice(fmt('{label} asks for an action this version does not know.', { label }), 'warn');
        return false;
    }
    const payload = {};
    let sent = null;
    if (action.with === 'message') Object.assign(payload, { text: row?.text ?? '', row_id: row?.address ?? '' });
    else if (action.with === 'draft') payload.text = sent = draft?.text() ?? '';
    try {
        const reply = await post(action.url || item.route_url, payload);
        if (action.then === 'clear_draft' && sent !== null && draft?.text() === sent) draft.clear();
        const said = typeof reply?.notice === 'string' && reply.notice.trim() ? reply.notice.trim()
            : fmt('{label}: done.', { label });
        notice(boundedText(said, 300), 'ok');
        return true;
    } catch (error) {
        notice(fmt('{label} did not work: {reason}', { label, reason: boundedText(error?.message || error || 'unknown error', 200) }),
            'danger');
        return false;
    }
}
