// One client store over /api/model-catalog for every composer in the window (DESIGN "Composer owner controls"):
// the catalog as last read, the owner's recent models (`model_recents` of /api/ui/preferences, written by the
// server on each of his model acts) and, only when a saved choice names an API source the catalog did not list,
// the settings document's connection list that says why. A read happens when a Model capsule opens and then only
// when stale or failed, never on a timer; a failed read keeps what an earlier one listed and says so. Settings ›
// Models refreshing its own catalog feeds this one without another request. The circle editor's Model field takes
// its suggestions from the same read (circles.js).
import { apiClient, apiFetch } from './api_client.js';
import { apiConnections, MODEL_PROCESSING_PREFERENCES_KEY, PROCESSING_PREFERENCE_KEY, tierPriced } from './route_editor_primitives.js';
import { MODEL_CATALOG_TIMEOUT_MS, mergeModelCatalog } from './settings_catalog.js';
import { buildModelIndex, normalizeRoute } from './chat_model_routes.js';

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
    readSettings = () => apiClient.settings(), now = () => Date.now() } = {}) {
    const state = { catalog: null, readAt: 0, failed: false, reading: null, recents: [], recentsReading: null,
        connections: null, connectionsReading: null, index: null, mainTierPriced: false };
    const listeners = new Set();
    const notify = () => { state.index = null; listeners.forEach((listener) => listener()); };

    function adopt(data) {
        state.catalog = mergeModelCatalog(state.catalog || {}, data);
        state.readAt = now();
        state.failed = false;
    }

    function read({ force = false } = {}) {
        if (state.reading) return state.reading;
        if (!force && state.catalog && !state.failed && now() - state.readAt < CATALOG_FRESH_MS) return Promise.resolve();
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

    /** The owner's recent models, newest first; a body without the field, or a failed read, is no recents. */
    function readRecents() {
        state.recentsReading ||= Promise.resolve().then(readPreferences)
            .then((prefs) => {
                state.recents = (Array.isArray(prefs?.model_recents) ? prefs.model_recents : [])
                    .map(normalizeRoute).filter(Boolean);
            })
            .catch(() => {})
            .finally(() => { state.recentsReading = null; notify(); });
        return state.recentsReading;
    }

    /** What the list takes from a settings document: its API connections and Main's processing tier. */
    function adoptSettings(settings) {
        state.connections = apiConnections(settings, settings?._meta?.setup_contract?.providerProfiles || {});
        // Main on Fast or Economy is priced by its tier's endpoint: the list then names no catalog price.
        let roles = settings?.[MODEL_PROCESSING_PREFERENCES_KEY];
        try { roles = typeof roles === 'string' ? JSON.parse(roles) : roles; } catch { roles = null; }
        state.mainTierPriced = tierPriced(String(roles?.main || settings?.[PROCESSING_PREFERENCE_KEY] || ''));
        state.connectionsFailed = false;
    }

    /** The settings document, read once: which key a saved source names and why it cannot send, and Main's tier.
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
        read, readRecents, readConnections,
        /** Another reader's `/api/model-catalog` answer (Settings › Models' Refresh). */
        adopt(data) { if (Array.isArray(data?.items)) { adopt(data); notify(); } },
        /** The settings document Settings loaded or saved: a changed tier or key reaches the list without a read. */
        adoptSettings(settings) { if (settings && typeof settings === 'object') { adoptSettings(settings); notify(); } },
        /** The catalog as last read (the merged `/api/model-catalog` answer), or null before any read. */
        catalog: () => state.catalog,
        /** The index of the catalog as last read (chat_model_routes.js), built once per change. */
        index() { return state.index ||= buildModelIndex({ catalog: state.catalog || {}, connections: state.connections, tierPriced: state.mainTierPriced }); },
        recents: () => state.recents,
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
