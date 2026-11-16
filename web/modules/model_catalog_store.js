// One client store over /api/model-catalog for every composer in the window (DESIGN "Composer owner controls"):
// the catalog as last read, the owner's recent and pinned models (`model_recents` and `model_pins` of
// /api/ui/preferences: the server writes the recents on each of his model acts; a pin is one addressed write of its
// own) and, when a saved choice names an API source the catalog did not list, the settings document: its connections
// say why a source cannot send. A read happens when a Model sheet opens and then only when stale or failed, never on
// a timer; a failed read keeps what an earlier one listed and says so. Another window's pin reaches this one by
// `ui_preferences_changed` (one read, another after it if a notice came while it was out). This window's pins are
// written one at a time in the order pressed, so each answer is the server's list after that press and the last
// press answers last; a read sent before a write answered never stands over that write's answer (one more read
// settles it). Settings › Models refreshing its own catalog feeds this one without another request. The circle
// editor's Model field takes its suggestions from the same read (circles.js).
import { apiClient, apiFetch } from './api_client.js';
import { apiConnections, MODEL_PROCESSING_PREFERENCES_KEY, PROCESSING_PREFERENCE_KEY, tierPriced } from './route_editor_primitives.js';
import { MODEL_CATALOG_TIMEOUT_MS, mergeModelCatalog } from './settings_catalog.js';
import { buildModelIndex, normalizeRoute, routeKey } from './chat_model_routes.js';

const CATALOG_FRESH_MS = 5 * 60 * 1000;

async function fetchCatalog() {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), MODEL_CATALOG_TIMEOUT_MS);
    try {
        const response = await apiFetch('/api/model-catalog', { cache: 'no-store', signal: controller.signal });
        const data = await response.json();
        if (!response.ok || !Array.isArray(data?.items)) throw new Error(data?.error || `HTTP ${response.status}`);
        return data;
    } catch (error) {
        throw error?.name === 'AbortError' ? new Error('timed out') : error;
    } finally {
        clearTimeout(timer);
    }
}

/**
 * @param {object} [deps]
 * @param {() => Promise<object>} [deps.readCatalog]  one `/api/model-catalog` answer, or a rejection
 * @param {() => Promise<object>} [deps.readPreferences]  `apiClient.uiPreferences`
 * @param {() => Promise<object>} [deps.readSettings]  `apiClient.settings`
 * @param {() => number} [deps.now]
 */
export function createModelCatalogStore({ readCatalog = fetchCatalog, readPreferences = () => apiClient.uiPreferences(),
    readSettings = () => apiClient.settings(), savePreferences = (body) => apiClient.saveUiPreferences(body),
    now = () => Date.now() } = {}) {
    // `served`: the pins as the server last said them; `pins`: what this window shows, those with each route's latest
    // unanswered press here on top (`pinning`: `{ route | null, seq }`). `pinOut` counts the writes out or queued,
    // `pinWriting` is the last of them; `pinsWritten` counts the answers adopted, so a read knows whether one came after
    // it was sent.
    const state = { catalog: null, readAt: 0, failed: false, reading: null, recents: [], recentsReading: null,
        recentsAgain: false, served: [], pins: [], pinning: new Map(), pinSeq: 0, pinOut: 0, pinWriting: null,
        pinsWritten: 0, connections: null, connectionsReading: null, index: null, tiers: null, cacheTier: null,
        readTier: undefined };
    const listeners = new Set();
    const notify = () => { state.index = null; listeners.forEach((listener) => listener()); };

    // A row's cache-write figure names the tier it is of (`price.cache_write_ttl`): it is said only while the
    // settings name that tier (`index`). A held figure of another tier (kept by a partial read, brought by a late or
    // another reader's answer, or read before the first settings document) makes the catalog stale until this store
    // has begun a read under the tier the settings name (`readTier`): one more read per tier, never one per opened
    // capsule, since where that read cannot bring it (a value the server resolves otherwise, a tier changed from
    // another window, a provider that keeps failing) the figure stays unsaid and the catalog keeps its usual life.
    const staleForTier = () => state.cacheTier !== null && state.readTier !== state.cacheTier
        && (state.catalog?.items || []).some((item) => item?.price?.cache_write_ttl && item.price.cache_write_ttl !== state.cacheTier);

    function adopt(data) {
        state.catalog = mergeModelCatalog(state.catalog || {}, data);
        state.readAt = staleForTier() ? 0 : now();
        state.failed = false;
    }

    function read({ force = false } = {}) {
        if (state.reading) return state.reading;
        if (!force && state.catalog && !state.failed && now() - state.readAt < CATALOG_FRESH_MS) return Promise.resolve();
        state.readTier = state.cacheTier;
        state.reading = (async () => {
            try {
                adopt(await readCatalog());
            } catch (error) {
                // A failed read keeps what an earlier one listed, and says so.
                state.catalog = mergeModelCatalog(state.catalog || {}, { read_state: 'transport',
                    errors: [{ provider_id: 'catalog', error: String(error?.message || error) }] });
                state.readAt = now();
                state.failed = true;
            } finally {
                state.reading = null;
                notify();
            }
        })();
        notify();
        return state.reading;
    }

    const routesOf = (value) => (Array.isArray(value) ? value : []).map(normalizeRoute).filter(Boolean);
    /** `pins` with one press applied: its route pinned (`route`) or unpinned (null). */
    function pressed(pins, key, route) {
        const at = pins.findIndex((pinned) => routeKey(pinned) === key);
        if (route && at < 0) return [...pins, route];
        return !route && at >= 0 ? pins.filter((_pinned, index) => index !== at) : pins;
    }
    /** What this window shows: the server's pins with every unanswered press here on top (each write answers itself). */
    function showPins() {
        let pins = state.served;
        for (const [key, { route }] of state.pinning) pins = pressed(pins, key, route);
        state.pins = pins;
    }
    /** The owner's pins as an answer or a read says them. */
    function adoptPins(value) {
        state.served = routesOf(value);
        showPins();
    }
    /** The owner's recent models, newest first, and pinned ones, in pin order; a body without a field, or a failed
     *  read, is none of that kind. */
    function readRecents() {
        if (state.recentsReading) {
            state.recentsAgain = true;  // a notice while a read is out: it may predate the change, so read once more
            return state.recentsReading;
        }
        const written = state.pinsWritten;
        state.recentsReading = Promise.resolve().then(readPreferences)
            .then((prefs) => {
                state.recents = routesOf(prefs?.model_recents);
                // A write answered while this read was out: which of the two the server served last is unknown, so
                // the write's answer stands and a read sent after it settles the list.
                if (state.pinsWritten === written) adoptPins(prefs?.model_pins);
                else state.recentsAgain = true;
            })
            .catch(() => {})
            .finally(() => {
                state.recentsReading = null;
                notify();
                if (state.recentsAgain) { state.recentsAgain = false; void readRecents(); }
            });
        return state.recentsReading;
    }
    /** Pin or unpin one route: one addressed write (`model_pins.pin`/`.unpin`), so a pin made in another window
     *  meanwhile stays. It chooses nothing. The press shows at once; its write waits for the earlier ones. Resolves
     *  true when saved; a refusal restores the list as the server has it. */
    function setPinned(raw, pinned) {
        const route = normalizeRoute(raw);
        if (!route) return Promise.resolve(false);
        const key = routeKey(route);
        const seq = ++state.pinSeq;
        state.pinning.set(key, { route: pinned ? route : null, seq });
        showPins();
        notify();
        // Only this press's own answer retires its mark: a later press of the same route is still unanswered.
        const settle = () => { if (state.pinning.get(key)?.seq === seq) state.pinning.delete(key); };
        const queued = state.pinOut > 0;
        state.pinOut += 1;
        const send = async () => {
            try {
                const reply = await savePreferences({ model_pins: { [pinned ? 'pin' : 'unpin']: [route] } });
                settle();
                state.pinsWritten += 1;
                if (Array.isArray(reply?.model_pins)) adoptPins(reply.model_pins);
                else {  // saved: the server has this press; a read brings the rest of its list
                    state.served = pressed(state.served, key, pinned ? route : null);
                    showPins();
                    void readRecents();
                }
                return true;
            } catch (error) {
                // Refused: the press no longer shows, even when the read of the server's list fails too.
                settle();
                showPins();
                void readRecents();
                throw error;
            } finally {
                state.pinOut -= 1;
                notify();
            }
        };
        // Nothing out: it goes now; else after the write before it, whatever that one's answer.
        const write = queued ? state.pinWriting.then(send) : send();
        state.pinWriting = write.catch(() => {});
        return write;
    }

    /** What the lists take from a settings document: its API connections, and the tiers Main and a lead pay. */
    function adoptSettings(settings) {
        state.connections = apiConnections(settings, settings?._meta?.setup_contract?.providerProfiles || {});
        let roles = settings?.[MODEL_PROCESSING_PREFERENCES_KEY];
        try { roles = typeof roles === 'string' ? JSON.parse(roles) : roles; } catch { roles = null; }
        adoptTiers({ main: String(roles?.main || ''), default: String(settings?.[PROCESSING_PREFERENCE_KEY] || '') });
        state.connectionsFailed = false;
        // The tier a call is sent with: another one than before makes the held catalog stale (the next opened capsule
        // reads that tier's figures), and a held figure of another tier is unsaid at once (`index`).
        const cacheTier = String(settings?.OUROBOROS_PROMPT_CACHE_TTL || '').trim().toLowerCase();
        const moved = state.cacheTier !== null && state.cacheTier !== cacheTier;
        state.cacheTier = cacheTier;
        if (state.catalog && (staleForTier() || moved)) state.readAt = 0;
    }
    /** Main's own tier and the default one (`/api/state` `processing_tiers`, or a settings document). */
    function adoptTiers(tiers) {
        if (!tiers || typeof tiers !== 'object') return false;
        const next = { main: String(tiers.main || ''), default: String(tiers.default || '') };
        const changed = next.main !== state.tiers?.main || next.default !== state.tiers?.default;
        state.tiers = next;
        return changed;
    }

    /** Whether a catalog price beside a model that `role` (`main`, `lead`) picks could be another tier's (a catalog
     *  price is Standard's: Fast and Flex are priced by their tier's endpoint): so it is until the tiers are read. */
    const tierPricedFor = (role) => (state.tiers ? tierPriced(role === 'main' ? state.tiers.main || state.tiers.default
        : state.tiers.default) : true);

    /** The settings document, read once: which key a saved source names and why it cannot send, and the tiers.
     *  A failed read is not repeated by every paint; Settings' own load or save brings the document again. */
    function readConnections() {
        if (state.connections || state.connectionsFailed) return Promise.resolve();
        state.connectionsReading ||= Promise.resolve().then(readSettings)
            .then(adoptSettings)
            .catch(() => { state.connectionsFailed = true; })
            .finally(() => { state.connectionsReading = null; notify(); });
        return state.connectionsReading;
    }

    return {
        read, readRecents, readConnections, setPinned,
        /** Another reader's `/api/model-catalog` answer (Settings › Models' Refresh). */
        adopt(data) { if (Array.isArray(data?.items)) { adopt(data); notify(); } },
        /** The settings document Settings loaded or saved: a changed tier or key reaches the list without a read. */
        adoptSettings(settings) { if (settings && typeof settings === 'object') { adoptSettings(settings); notify(); } },
        /** `/api/state`'s `processing_tiers`: Main's own tier and the default one. */
        adoptTiers(tiers) { if (adoptTiers(tiers)) notify(); },
        /** The tiers as last read (`{ main, default }`, '' for none), or null before any read. */
        tiers: () => state.tiers,
        /** The catalog as last read (the merged `/api/model-catalog` answer), or null before any read. */
        catalog: () => state.catalog,
        /** The index of the catalog as last read (chat_model_routes.js), built once per change. Each row keeps the
         *  catalog's own price; whoever says a row decides whether its context pays it (`describeRoute`'s tier). */
        index() { return state.index ||= buildModelIndex({ catalog: state.catalog || {}, connections: state.connections,
            cacheTier: state.cacheTier }); },
        tierPriced: tierPricedFor,
        recents: () => state.recents,
        pins: () => state.pins,
        isPinned: (route) => state.pins.some((pinned) => routeKey(pinned) === routeKey(route)),
        /** Re-read the pins and recents whenever another window (or this one) committed a pin. Returns the disposer. */
        attach({ ws = null } = {}) {
            const off = ws?.on?.('ui_preferences_changed', (msg) => {
                if (!Array.isArray(msg?.keys) || msg.keys.includes('model_pins')) void readRecents();
            });
            return () => { if (typeof off === 'function') off(); };
        },
        /** `reading`: a read is in flight; `read`: one has answered; `failed`: the last one did not. */
        status: () => ({ reading: Boolean(state.reading), read: Boolean(state.catalog), failed: state.failed }),
        subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    };
}

export const modelCatalog = createModelCatalogStore();

// A page that can be listened to: the module is also loaded where there is none (a reader of the work state).
if (typeof globalThis.document?.addEventListener === 'function') {
    document.addEventListener('settings-model-catalog:updated', (event) => modelCatalog.adopt(event?.detail));
    document.addEventListener('settings-document:applied', (event) => modelCatalog.adoptSettings(event?.detail));
}
