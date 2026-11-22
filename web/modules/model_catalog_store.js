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
//
// A processing tier is told for one route as it is bound. A subscription's is its accounts' own lists, which come
// with the catalog. A sheet using Main's pin refreshes it from Settings on opening (`readConnections`); a held pin
// may have changed without moving the model. Until confirmed for this route, the pin is unknown. A key's
// model has its own endpoints (`detail`): one addressed read per exact routed value (`readDetail`), asked for the
// route a sheet or a role's page is on and never for a list's rows, kept under that value so an answer for one model
// is never another's, and fresh as long as the catalog is. A read that failed, or answered for another value or with
// no tier, tells nothing: never that a model has only its ordinary service.
import { apiClient, apiFetch } from './api_client.js';
import {
    apiConnections, connectionParts, MODEL_ACCOUNTS_KEY, MODEL_PROCESSING_PREFERENCES_KEY, PROCESSING_PREFERENCE_KEY, tierPriced,
} from './route_editor_primitives.js';
import { MODEL_CATALOG_TIMEOUT_MS, catalogReadFailed, invalidateModelCatalogReads, mergeModelCatalog } from './settings_catalog.js';
import { buildModelIndex, normalizeRoute, routeKey, routeSource, sameRoute } from './chat_model_routes.js';

const CATALOG_FRESH_MS = 5 * 60 * 1000;
const MAIN_TIER_KEYS = ['OUROBOROS_MODEL', 'USE_LOCAL_MAIN', MODEL_ACCOUNTS_KEY, MODEL_PROCESSING_PREFERENCES_KEY];
/** The `if_unchanged` condition naming the API connection a write's route was shown on: its `_meta.connections`
 *  revision as displayed (owner_settings.connection_revision), compared with that connection now under the lock. */
export const CONNECTION_CONDITION = 'connection';
/** The picker id of the API connection a route sends to (`kind` or `kind@name`); '' for a subscription or the local
 *  model, which have none. */
export function apiConnectionOf(raw) {
    const route = normalizeRoute(raw);
    const source = route ? routeSource(route) : '';
    return !route || route.use_local || source === 'local' || source.startsWith('subscription:') ? '' : source;
}
/** Each API connection's revision as one settings read gives it (`_meta.connections[].revision`). */
const revisionsOf = (settings) => new Map((Array.isArray(settings?._meta?.connections) ? settings._meta.connections : [])
    .filter((row) => typeof row?.id === 'string' && typeof row.revision === 'string' && row.revision).map((row) => [row.id, row.revision]));
/** `route`'s connection revision in `settings`: '' where it has no API connection, null where the read names none. */
export function revisionIn(settings, route) {
    const id = apiConnectionOf(route);
    return id ? revisionsOf(settings).get(id) ?? null : '';
}
// The settings a key route's endpoint list and its transport depend on beyond its connection's revision: the
// provider routing, and the connection facts of a core that names no revision.
const CONNECTION_FACT = /^(OUROBOROS_CONNECTIONS|OUROBOROS_OR_PROVIDER|[A-Z0-9_]+_(BASE_URL|API_SURFACE|API_KEY))$/;
/** One string for everything a settings document says of its connections: another one means another source, so an
 *  endpoint list or a transport read under the earlier one tells nothing now. */
export function connectionGeneration(settings) {
    const tokens = settings?._meta?.tokens || {};
    return JSON.stringify([Object.keys(settings || {}).filter((key) => CONNECTION_FACT.test(key)).sort()
        .map((key) => [key, tokens[key] || JSON.stringify(settings[key] ?? null)]), [...revisionsOf(settings)].sort()]);
}
const transportConnection = (item) => item?.processing_transport && apiConnectionOf({ model: item.value, use_local: item.use_local });
const hasTransport = (catalog) => [catalog, catalog?.current].some((read) => read?.items?.some(transportConnection)
    || read?.api_sources?.some((source) => source?.processing_transport));
/** `catalog` without the transport tiers of `changed` connections (null: every API connection's), until a read
 *  says them again: a limit read under another source is no limit of this one. */
export function dropTransport(catalog, changed = null) {
    if (!catalog || typeof catalog !== 'object') return catalog;
    const strip = (row, id) => {
        if (!id || (changed && !changed.has(id))) return row;
        const { processing_transport: _gone, ...rest } = row;
        return rest;
    };
    const out = { ...catalog };
    if (Array.isArray(catalog.items)) out.items = catalog.items.map((item) => strip(item, transportConnection(item)));
    if (Array.isArray(catalog.api_sources)) out.api_sources = catalog.api_sources.map((source) => strip(source, source?.id));
    // The raw answer a merged catalog keeps (`current`, mergeModelCatalog) is what the tier index reads: the same.
    if (catalog.current && typeof catalog.current === 'object' && catalog.current !== catalog) out.current = dropTransport(catalog.current, changed);
    return out;
}

function mainOption(settings, key) {
    let roles = settings?.[key];
    try { roles = typeof roles === 'string' ? JSON.parse(roles) : roles; } catch { roles = null; }
    return String(roles?.main || '');
}

const mainRoute = (settings) => normalizeRoute({ model: settings.OUROBOROS_MODEL,
    use_local: /^(true|1|yes|on)$/i.test(String(settings.USE_LOCAL_MAIN).trim()) });

async function fetchCatalog(query = '', answered = (data) => Array.isArray(data?.items)) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), MODEL_CATALOG_TIMEOUT_MS);
    try {
        const response = await apiFetch(`/api/model-catalog${query}`, { cache: 'no-store', signal: controller.signal });
        const data = await response.json();
        if (!response.ok || !answered(data)) throw new Error(data?.error || `HTTP ${response.status}`);
        return data;
    } catch (error) {
        throw error?.name === 'AbortError' ? new Error('timed out') : error;
    } finally {
        clearTimeout(timer);
    }
}

/** Whether a route's tiers are told by its model's own endpoints: an OpenRouter key's model. */
const detailed = (route) => !route.use_local && connectionParts(routeSource(route)).kind === 'openrouter';
const tierWords = (list) => (Array.isArray(list) ? list.filter((tier) => typeof tier === 'string') : []);

/** One `?processing_model=` answer as the tiers of `value`: the modes its endpoints list, whether that list is whole,
 *  and each classified endpoint with its price (USD per 1M tokens). Throws for another model's answer or one that
 *  names no tier: neither is ever read as a model with only its ordinary service. */
function detailOf(value, data) {
    // Older cores ignore the selector and return their ordinary catalog. That says nothing about this model.
    if (Array.isArray(data?.items) && !Object.hasOwn(data, 'processing')) {
        return { state: 'unsupported', modes: new Set(), complete: false, endpoints: [] };
    }
    const said = data?.processing;
    const modes = tierWords(said?.modes);
    if (data?.value !== value || !modes.length) throw new Error('no tiers read for this model');
    return { state: 'ok', modes: new Set(modes), complete: said.complete === true,
        endpoints: (Array.isArray(said.endpoints) ? said.endpoints : []).filter((endpoint) => typeof endpoint?.mode === 'string')
            .map((endpoint) => ({ mode: endpoint.mode, tag: String(endpoint.tag || ''), price: endpoint.price ?? null })) };
}

/**
 * @param {object} [deps]
 * @param {() => Promise<object>} [deps.readCatalog]  one `/api/model-catalog` answer, or a rejection
 * @param {(value: string) => Promise<object>} [deps.readProcessing]  one `?processing_model=<value>` answer
 * @param {() => Promise<object>} [deps.readPreferences]  `apiClient.uiPreferences`
 * @param {() => Promise<object>} [deps.readSettings]  `apiClient.settings`
 * @param {() => number} [deps.now]
 */
export function createModelCatalogStore({ readCatalog = () => fetchCatalog(), readPreferences = () => apiClient.uiPreferences(),
    readSettings = () => apiClient.settings(), savePreferences = (body) => apiClient.saveUiPreferences(body),
    readProcessing = (value) => fetchCatalog(`?processing_model=${encodeURIComponent(value)}`, (data) =>
        Boolean(data?.processing) || Array.isArray(data?.items)),
    now = () => Date.now() } = {}) {
    // `served`: the pins as the server last said them; `pins`: what this window shows, those with each route's latest
    // unanswered press here on top (`pinning`: `{ route | null, seq }`). `pinOut` counts the writes out or queued,
    // `pinWriting` is the last of them; `pinsWritten` counts the answers adopted, so a read knows whether one came after
    // it was sent.
    const state = { catalog: null, readAt: null, failed: false, reading: null, recents: [], recentsReading: null,
        recentsAgain: false, served: [], pins: [], pinning: new Map(), pinSeq: 0, pinOut: 0, pinWriting: null,
        pinsWritten: 0, connections: null, connectionsReading: null, connectionsAgain: false,
        index: null, tiers: null, cacheTier: null, readTier: null,
        mainAccount: null, mainAccountRoute: null, details: new Map(), processingSelector: null, processingProbe: null,
        generation: null, detailEpoch: 0, revisions: null,
        // `sourceEpoch` counts connection changes; until a catalog read sent under the latest one answers, the
        // transport of `staleSources` (null: every API row's) is told by no earlier read; `readAgain`: one is owed.
        sourceEpoch: 0, staleSources: new Set(), readAgain: false, settingsSeen: 0 };
    /** A catalog answer as it may be adopted: one read under the current source whole; one from before a connection
     *  changed (or another reader's, while one is owed) without the changed sources' transport. */
    const fit = (data, epoch = -1) => {
        if (epoch === state.sourceEpoch) { state.staleSources = new Set(); return data; }
        if (epoch === 0 && state.reading && hasTransport(data)) state.readAgain = true;
        return state.staleSources === null || state.staleSources.size ? dropTransport(data, state.staleSources) : data;
    };
    const listeners = new Set();
    const notify = () => { state.index = null; listeners.forEach((listener) => listener()); };

    // The index only shows figures stamped for the settings' tier. `readTier` is the tier when the latest read began
    // (null before Settings, undefined after a tier change, including A→B→A). A read with no known tier can prove
    // the now-known tier only through its figures. Otherwise the next ordinary read makes one corrective attempt.
    const otherTierHeld = () => state.cacheTier !== null
        && (state.catalog?.items || []).some((item) => item?.price?.cache_write_ttl && item.price.cache_write_ttl !== state.cacheTier);
    const namedTierFigures = () => (state.cacheTier === null ? 0
        : (state.catalog?.items || []).filter((item) => item?.price?.cache_write_ttl === state.cacheTier).length);
    const staleForTier = () => state.readTier === undefined || (state.readTier !== state.cacheTier
        && (otherTierHeld() || namedTierFigures() === 0));

    /** One read outcome, however it arrived: a failed read keeps the last choices and owes a new read.
     *  `readAt` null means stale; a handed-over refresh has no known start tier or order. */
    function applyOutcome(data, handed = false) {
        const said = handed ? namedTierFigures() : 0;
        state.catalog = mergeModelCatalog(state.catalog || {}, data);
        // A stated failure, even with HTTP 200, is distinct from a successful empty or partial read.
        state.failed = catalogReadFailed(data);
        state.readAt = state.failed || staleForTier() || (handed && (otherTierHeld() || namedTierFigures() < said)) ? null : now();
    }

    function read({ force = false } = {}) {
        if (state.reading) {
            // An open while a read is out for a different or not-yet-known tier rechecks freshness when it answers.
            if (state.readTier !== state.cacheTier) return state.reading.then(() => read({ force }));
            if (!force) return state.reading;
            state.readAgain = true;  // the one out may predate the change: wait for the read it queues
            return state.reading.then(() => state.reading);
        }
        if (!force && state.catalog && !state.failed && state.readAt !== null && now() - state.readAt < CATALOG_FRESH_MS) {
            return Promise.resolve();
        }
        state.readTier = state.cacheTier;
        const epoch = state.sourceEpoch;
        state.reading = (async () => {
            try {
                const data = await readCatalog();
                // A failed read does not settle a changed source’s pending capability.
                applyOutcome(fit(data, catalogReadFailed(data) ? -1 : epoch));
            } catch (error) {
                // A failed read keeps what an earlier one listed, and says so.
                applyOutcome({ read_state: 'transport', errors: [{ provider_id: 'catalog', error: String(error?.message || error) }] });
            } finally {
                state.reading = null;
                notify();
                if (state.readAgain) { state.readAgain = false; void read({ force: true }); }
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
        state.settingsSeen += 1;  // a read of the settings sent before this one is of an older document
        state.connections = apiConnections(settings, settings?._meta?.setup_contract?.providerProfiles || {});
        // Another connection under the same routed value: what its endpoints and its transport said before is no longer
        // told (a late answer of the earlier one stands nowhere: readDetail), and the catalog is read again.
        const generation = connectionGeneration(settings);
        const revisions = revisionsOf(settings);
        const previous = state.generation;
        state.generation = generation;
        if (generation !== previous) {
            state.details.clear();
            state.processingSelector = state.processingProbe = null;
            state.detailEpoch += 1;
            const changed = state.revisions && revisions.size ? new Set([...new Set([...state.revisions.keys(), ...revisions.keys()])]
                .filter((id) => state.revisions.get(id) !== revisions.get(id))) : null;
            const sources = changed?.size ? changed : null;
            state.sourceEpoch += 1;
            invalidateModelCatalogReads();  // a Settings read already out is of the earlier source too
            const refresh = previous !== null ? state.catalog || state.reading : hasTransport(state.catalog);
            if (refresh || state.reading) {
                state.staleSources = sources && state.staleSources ? new Set([...state.staleSources, ...sources]) : null;
                if (refresh) {
                    state.catalog = dropTransport(state.catalog, sources);
                    state.readAt = null;
                    void read({ force: true });
                }
            }
        }
        state.revisions = revisions;
        adoptTiers({ main: mainOption(settings, MODEL_PROCESSING_PREFERENCES_KEY), default: String(settings?.[PROCESSING_PREFERENCE_KEY] || '') });
        state.mainAccount = mainOption(settings, MODEL_ACCOUNTS_KEY);
        state.mainAccountRoute = Object.hasOwn(settings, 'USE_LOCAL_MAIN') ? mainRoute(settings) : null;
        state.connectionsFailed = false;
        // The tier a call is sent with: another one than before makes the held catalog stale (the next opened capsule
        // reads that tier's figures), and a held figure of another tier is unsaid at once (`index`).
        const cacheTier = String(settings?.OUROBOROS_PROMPT_CACHE_TTL || '').trim().toLowerCase();
        if (state.cacheTier !== null && state.cacheTier !== cacheTier) state.readTier = undefined;
        state.cacheTier = cacheTier;
        if (state.catalog && staleForTier()) state.readAt = null;
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

    /** The settings document (connections, Main's pin and tiers). A fresh read sets aside the old pin at once.
     *  If a GET is already out, it may predate this opening: one fresh GET follows it, without adopting its answer.
     *  A document applied while this read was out is the newer one: the read's late answer, or its failure, never
     *  stands over it (`settingsSeen` counts the documents; a confirmed Main write is one of them). */
    function readConnections({ fresh = false } = {}) {
        if (fresh) {
            state.mainAccount = state.mainAccountRoute = null;
            if (state.connectionsReading) state.connectionsAgain = true;
        } else if (state.connections || state.connectionsFailed) return Promise.resolve();
        if (state.connectionsReading) return state.connectionsReading;
        let seen = state.settingsSeen;
        const superseded = () => state.connectionsAgain || state.settingsSeen !== seen;
        state.connectionsReading = Promise.resolve().then(async () => {
            for (;;) {
                try {
                    const settings = await readSettings();
                    if (!superseded()) adoptSettings(settings);
                } catch {
                    if (!superseded()) state.connectionsFailed = true;
                }
                if (!state.connectionsAgain) break;
                state.connectionsAgain = false;
                seen = state.settingsSeen;
            }
            state.connectionsReading = null;
            notify();
        });
        return state.connectionsReading;
    }

    /** One read of a key route's own endpoints, for the route a sheet or a role's page is on. Nothing is asked for a
     *  route whose tiers no such read tells, while one is out, or while a held answer is fresh; a failed one is asked
     *  again by the next call, never by a paint. The answer stands under its own routed value only. */
    function readDetail(raw) {
        const route = normalizeRoute(raw);
        if (!route || !detailed(route) || state.processingSelector === 'unsupported') return Promise.resolve();
        const value = route.model;
        const held = state.details.get(value);
        if (held?.reading) return held.reading;
        if (held?.state === 'ok' && now() - held.readAt < CATALOG_FRESH_MS) return Promise.resolve();
        const entry = { state: 'reading', modes: new Set(), complete: false, endpoints: [], readAt: now() };
        // An old server gives its entire catalog for each unknown selector. Until one response tells us otherwise,
        // let the first model's normal detail read also serve as the capability probe for the window.
        const probe = state.processingProbe;
        entry.reading = (probe || Promise.resolve()).then(async () => {
            if (state.details.get(value) !== entry) return null;
            if (state.processingSelector === 'unsupported') return { ...entry, state: 'unsupported' };
            return detailOf(value, await readProcessing(value));
        })
            // Not read: the tiers of an earlier read are not this moment's, so nothing is told (never Standard only).
            .catch(() => ({ state: 'failed', modes: new Set(), complete: false, endpoints: [] }))
            .then((read) => {
                // An answer that comes after its connection changed (its entry was dropped, or another one asked
                // since) is of the earlier source: it is not kept, and the next call asks again.
                if (state.processingProbe === entry.reading) state.processingProbe = null;
                if (read && state.details.get(value) === entry) {
                    if (read.state === 'unsupported' || read.state === 'ok') {
                        state.processingSelector = read.state === 'ok' ? 'supported' : 'unsupported';
                    }
                    state.details.set(value, { ...read, readAt: now(), reading: null });
                }
                notify();
            });
        if (!probe && state.processingSelector === null) state.processingProbe = entry.reading;
        state.details.set(value, entry);
        notify();
        return entry.reading;
    }

    return {
        read, readRecents, readConnections, setPinned, readDetail,
        /** Main as one GET has it, with the tokens of its binding (`binding`: the four keys) and the document itself
         *  (`settings`: its connections' revisions say which source a route was shown on). Keep the displayed sheet
         *  intact for comparison. */
        async readMainTierSettings() {
            const settings = await readSettings();
            const tokens = settings?._meta?.tokens;
            if (!MAIN_TIER_KEYS.every((key) => Object.hasOwn(settings || {}, key) && typeof tokens?.[key] === 'string' && tokens[key])) {
                throw new Error('Could not verify Main’s settings. Reload, then choose again.');
            }
            return { route: mainRoute(settings), account: mainOption(settings, MODEL_ACCOUNTS_KEY), tier: mainOption(settings, MODEL_PROCESSING_PREFERENCES_KEY),
                binding: Object.fromEntries(MAIN_TIER_KEYS.map((key) => [key, tokens[key]])), settings };
        },
        /** The revision of `route`'s API connection as this window last read the settings ('' for a route without one,
         *  null before a read or where the read names none). */
        revisionOf: (route) => (apiConnectionOf(route) ? state.revisions?.get(apiConnectionOf(route)) ?? null : ''),
        /** How many times the held endpoint answers were dropped for another connection: a reader that asked under an
         *  earlier count asks again. */
        detailEpoch: () => state.detailEpoch,
        /** A key route's own endpoints as last read (`{ state: 'reading' | 'ok' | 'failed' | 'unsupported', modes, complete, endpoints }`);
         *  null for a route never asked or whose tiers no such read tells. */
        detail(raw) {
            const route = normalizeRoute(raw);
            return (route && detailed(route) && state.details.get(route.model)) || null;
        },
        /** Main's account ('' is Auto). With a route, require that the settings read or write named that exact Main. */
        mainAccount: (route) => (route === undefined || sameRoute(normalizeRoute(route), state.mainAccountRoute) ? state.mainAccount : null),
        adoptMainAccount(account, route) {
            state.settingsSeen += 1;  // an older GET cannot undo a confirmed Main write
            state.mainAccount = String(account || '');
            state.mainAccountRoute = normalizeRoute(route);
            notify();
        },
        /** Another reader's `/api/model-catalog` answer (Settings › Models' Refresh). A failed or scoped-unsupported
         *  answer (no items) is still a read outcome: adopt it so its read_state invalidates the held price and keeps
         *  the rows for the chooser; only a body with no read at all is ignored. A failed read leaves the store
         *  failed, so the next opening reads again instead of showing an empty sheet forever. */
        adopt(data) { if (data && typeof data === 'object' && (Array.isArray(data.items) || data.read_state || data.error || Array.isArray(data.errors))) { applyOutcome(fit(data), true); notify(); } },
        /** The settings document Settings loaded or saved: a changed tier or key reaches the list without a read. */
        adoptSettings(settings) {
            if (settings && typeof settings === 'object') { adoptSettings(settings); notify(); }
        },
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

/**
 * The catalogue as it changes: the caller hears a new read outcome, not every pin press. A legacy
 * core's prices are a view of this one full read, so a failed refresh must reach the caller at once
 * (hiding the price without losing the rows) and a new success must bring it back.
 */
export function watchModelCatalog(onChange) {
    let last = modelCatalog.catalog();
    return modelCatalog.subscribe(() => {
        const catalog = modelCatalog.catalog();
        if (catalog === last) return;
        last = catalog;
        onChange(catalog);
    });
}

// A page that can be listened to: the module is also loaded where there is none (a reader of the work state).
if (typeof globalThis.document?.addEventListener === 'function') {
    document.addEventListener('settings-model-catalog:updated', (event) => modelCatalog.adopt(event?.detail));
    document.addEventListener('settings-document:applied', (event) => modelCatalog.adoptSettings(event?.detail));
}
