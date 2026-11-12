// What the composer's Model capsule knows (DESIGN "Composer owner controls"), as data: the room's choice, the
// sources that can lead a work and their models as the catalog lists them, one row per (source, model), and a search
// across them. composer_model.js draws it. The catalog suggests and never admits: a saved or typed route no catalog
// lists stays a route of its own source, never dropped or rewritten. A route's identity is its whole string, source
// included, so one model through two sources is two rows.
import {
    API_PROVIDER_ORDER, apiProviderLabel, composeModelSource, connectionParts, parseModelSource, priceWords,
    unavailableConnectionState,
} from './route_editor_primitives.js';
import { harnessPresentation } from './harness_presentation.js';

export const MAIN_CHOICE = Object.freeze({ kind: 'main' });
export const LOCAL_SOURCE = 'local';
const SUBSCRIPTION = 'subscription:';
// How a source is paid for, in the owner's words (a local model is neither).
export const PAY_WORDS = Object.freeze({ subscription: 'subscription', api: 'paid by key', local: 'on this computer' });

/** The canonical room choice; anything unreadable follows Main, which is what the room then runs as. */
export function normalizeChoice(raw) {
    if (raw?.kind === 'model' && typeof raw.model === 'string' && raw.model.trim()) {
        return { kind: 'model', model: raw.model.trim(), use_local: raw.use_local === true };
    }
    return { ...MAIN_CHOICE };
}

/** A route as the contract carries it (`{model, use_local}`), or null. */
export function normalizeRoute(raw) {
    const model = typeof raw?.model === 'string' ? raw.model.trim() : '';
    return model ? { model, use_local: raw.use_local === true } : null;
}

export const sameRoute = (a, b) => Boolean(a && b) && a.model === b.model && Boolean(a.use_local) === Boolean(b.use_local);
/** One string per route, for a DOM attribute and a set. */
export const routeKey = (route) => (route ? `${route.use_local ? 'local' : 'route'}|${route.model}` : '');

/** The source a route is reached through: `subscription:<id>`, an API connection (`kind`, `kind@name`) or local. */
export function routeSource(route) {
    if (route?.use_local) return LOCAL_SOURCE;
    return parseModelSource(route?.model).source || 'openrouter';
}

const laneOf = (source) => (source === LOCAL_SOURCE ? 'local' : source.startsWith(SUBSCRIPTION) ? 'subscription' : 'api');
const modelPart = (route) => (route.use_local ? route.model : parseModelSource(route.model).model || route.model);
/** The model's own name without its vendor path: `z-ai/glm-5.3` reads `glm-5.3`. */
export const shortName = (route) => { const id = route ? modelPart(route) : ''; return id.slice(id.lastIndexOf('/') + 1) || id; };

/** A window in the owner's words (872K, 1M); '' when unknown: an unknown window is omitted, never printed. */
export function windowWords(tokens) {
    const n = Number(tokens) || 0;
    if (n <= 0) return '';
    return n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : `${Math.max(1, Math.round(n / 1000))}K`;
}

// Newer models first (owner 30.10). No catalog states a release date, so the order reads the version a name states:
// a name compares part by part, its words A to Z and its numbers high to low, a dotted version as one number (6.1
// before 6 before 5.6, 6.10 before 6.9), so within one family (the words before its first number) the higher
// version comes first and families follow each other by name. A name that states no number claims no age and
// follows. A presentation convention only: it never claims a release date and never filters.
const NAME_PARTS = /(\d+(?:\.\d+)*)|(\D+)/gu;
const nameParts = (name) => [...String(name || '').toLocaleLowerCase().matchAll(NAME_PARTS)]
    .map(([, number, words]) => (number !== undefined ? number.split('.').map(Number)
        : words.replace(/[^\p{L}\p{N}]+/gu, ' ').trim()));

/** Two model names in newer-first order (negative: `a` first). */
export function compareNewerFirst(a, b) {
    const x = nameParts(a);
    const y = nameParts(b);
    const stated = (parts) => parts.some(Array.isArray);
    if (stated(x) !== stated(y)) return stated(x) ? -1 : 1;
    for (let at = 0; at < Math.max(x.length, y.length); at += 1) {
        const [p, q] = [x[at], y[at]];
        if (p === undefined || q === undefined) return p === undefined ? -1 : 1;
        if (Array.isArray(p) !== Array.isArray(q)) return Array.isArray(p) ? -1 : 1;
        if (Array.isArray(p)) {
            for (let k = 0; k < Math.max(p.length, q.length); k += 1) {
                if ((p[k] || 0) !== (q[k] || 0)) return (q[k] || 0) - (p[k] || 0);
            }
        } else {
            const order = p.localeCompare(q);
            if (order) return order;
        }
    }
    return String(a || '').localeCompare(String(b || ''));
}

/**
 * Read one `/api/model-catalog` answer into the capsule's index. The sources are the ones Settings offers the Main
 * role (model_roles.js `modelSourceGroups`, never agent sessions): the catalog's own subscription `model_sources`,
 * the API connections whose lists answered or failed (the server reads only an enabled connection holding its
 * credential), and the local model when one is configured. One row per (source, model): a subscription model many
 * accounts list is one row, available when any of them can serve it, and it never names an account.
 * `connections` (`apiConnections()` of a settings read, when one was made) names an API source that cannot send.
 * @param {{catalog?: object, connections?: ?Array}} input
 */
export function buildModelIndex({ catalog = {}, connections = null, tierPriced = false } = {}) {
    const sources = new Map();
    const rows = new Map();
    const descriptors = Array.isArray(catalog.model_sources) ? catalog.model_sources : [];
    const items = Array.isArray(catalog.items) ? catalog.items : [];
    const errors = Array.isArray(catalog.errors) ? catalog.errors : [];
    const problems = [];

    function source(id, extra = {}) {
        if (!sources.has(id)) {
            const lane = laneOf(id);
            const sub = lane === 'subscription' ? id.slice(SUBSCRIPTION.length) : '';
            const descriptor = descriptors.find((entry) => entry?.id === sub);
            const { kind, name } = connectionParts(id);
            sources.set(id, {
                id, lane, pay: PAY_WORDS[lane], leads: false, count: 0, problems: [],
                label: lane === 'local' ? 'Local' : lane === 'subscription'
                    ? String(descriptor?.label || harnessPresentation(sub).label) : apiProviderLabel(id, {}, connections || []),
                harness: lane === 'subscription' ? String(descriptor?.credentialHarness || sub) : '',
                kind: lane === 'api' ? kind : '', named: lane === 'api' && Boolean(name),
            });
        }
        return Object.assign(sources.get(id), extra);
    }
    for (const entry of descriptors) if (entry?.id) source(`${SUBSCRIPTION}${entry.id}`, { leads: true });

    for (const item of items) {
        // A local model's row names its artifact in `id`; its `value` is a label, never a route (gateway/models.py).
        const local = item?.use_local === true;
        const model = String((local ? item.id : item?.value) || '');
        if (!model) continue;
        const route = { model, use_local: local };
        const key = routeKey(route);
        const own = source(routeSource(route), { leads: true });
        if (own.named && item?.provider_label) own.label = String(item.provider_label);  // the connection's own name
        if (!rows.has(key)) {
            rows.set(key, { key, route, id: modelPart(route), name: String(item?.name || '') || modelPart(route),
                source: own.id, lane: own.lane, window: 0, available: null, states: new Set(), prices: new Map() });
            own.count += 1;
        }
        const row = rows.get(key);
        row.window = Math.max(row.window, Number(item?.max_context_window) || Number(item?.context_window) || 0);
        if (typeof item?.availability === 'string') row.states.add(item.availability);
        row.prices.set(JSON.stringify(item?.price ?? null), item?.price ?? null);
    }
    let priced = false;  // whether the catalog states any price: only then do the processing tiers change a word
    for (const row of rows.values()) {
        // Any account that can serve it makes the row available; unavailable only when every account says so.
        row.available = row.states.has('available') ? true
            : row.states.size === 1 && row.states.has('unavailable') ? false : null;
        // The catalog's price, while every item of the row states that same one (only OpenRouter's rows state any);
        // kept as the figures, so its words follow the language of each paint.
        const price = row.prices.size === 1 ? [...row.prices.values()][0] : null;
        priced ||= Boolean(price);
        row.price = tierPriced ? null : price;
        delete row.states;
        delete row.prices;
    }
    for (const error of errors) {
        const id = error?.source_id ? `${SUBSCRIPTION}${error.source_id}`
            : (error?.provider_id && !['claudexor', 'catalog', 'extra_ca_bundle'].includes(error.provider_id)
                ? String(error.provider_id) : '');
        const text = String(error?.error || error?.code || '').split(/\s*For more information check:/)[0].trim();
        if (id) source(id, { leads: true }).problems.push(text);
        else if (text) problems.push(text);
    }

    // Subscriptions in the catalog's own order, then API connections in the owner-facing provider order (a named
    // key after its built-in one), then the local model.
    const rank = (entry) => {
        if (entry.lane === 'subscription') {
            const at = descriptors.findIndex((item) => `${SUBSCRIPTION}${item?.id}` === entry.id);
            return at < 0 ? descriptors.length : at;
        }
        if (entry.lane === 'local') return 9999;
        const at = API_PROVIDER_ORDER.indexOf(entry.kind);
        return 1000 + (at < 0 ? 900 : at) * 2 + (entry.named ? 1 : 0);
    };
    const sorted = [...sources.values()].filter((entry) => entry.leads)
        .sort((a, b) => rank(a) - rank(b) || a.label.localeCompare(b.label));
    const place = new Map(sorted.map((entry, at) => [entry.id, at]));
    const ordered = [...rows.values()].sort((a, b) => place.get(a.source) - place.get(b.source)
        || compareNewerFirst(a.id, b.id) || a.key.localeCompare(b.key));
    // Across sources one model reads once per provider: a key added under Accounts repeats its provider's list, so
    // its rows stand under its own source only, unless no earlier key of that provider lists the model.
    const firstOfKind = new Set();
    for (const row of ordered) {
        const own = sources.get(row.source);
        const kindKey = `${own.lane === 'api' ? own.kind : own.id}|${row.id}`;
        row.repeat = firstOfKind.has(kindKey);
        firstOfKind.add(kindKey);
    }
    return {
        problems, connections, priced,
        sources: sorted,
        source: (id) => sources.get(id) || null,
        rows: ordered,
        row: (route) => rows.get(routeKey(route)) || null,
        sourceRows: (id) => ordered.filter((entry) => entry.source === id),
    };
}

/**
 * What a route is called wherever the capsule names it: its short name, its source by its own label and how that
 * source is paid, the catalog's price in words when it states one (`price`), its window when the catalog knows one.
 * A source that leads nothing keeps Settings' own words:
 * "(no key)", "(off)" or "(removed)" for an API key that cannot send.
 */
export function describeRoute(index, raw) {
    const route = normalizeRoute(raw);
    if (!route) return null;
    const id = routeSource(route);
    const lane = laneOf(id);
    const own = index?.source(id);
    const listed = index?.row(route) || null;
    // Settings' own decision (model_roles.js keyNote): a source no enabled key holding its credential serves.
    const cannotSend = lane === 'api' && Array.isArray(index?.connections)
        && !index.connections.some((entry) => entry.id === id && entry.enabled && entry.configured);
    const base = own?.label || (lane === 'local' ? 'Local' : lane === 'subscription'
        ? harnessPresentation(id.slice(SUBSCRIPTION.length)).label : apiProviderLabel(id, {}, index?.connections || []));
    const sourceLabel = cannotSend ? `${base} (${unavailableConnectionState(id, index.connections)})` : base;
    const window = windowWords(listed?.window);
    const price = priceWords(listed?.price);
    return {
        route, key: routeKey(route), id: modelPart(route), short: shortName(route), source: id, lane,
        sourceLabel, pay: PAY_WORDS[lane], price, window, listed: Boolean(listed),
        available: listed?.available ?? null,
        harness: own?.harness || (lane === 'subscription' ? id.slice(SUBSCRIPTION.length) : ''),
        kind: own?.kind || (lane === 'api' ? connectionParts(id).kind : ''),
        facts: [sourceLabel, price || PAY_WORDS[lane], window ? `${window} window` : ''].filter(Boolean).join(' · '),
    };
}

/** Whether the route's API source is none this catalog read listed: the settings read then says why. */
export function sourceUnlisted(index, raw) {
    const route = normalizeRoute(raw);
    if (!route || !index) return false;
    const id = routeSource(route);
    return laneOf(id) === 'api' && !index.source(id)?.leads;
}

// How well a row answers the last word typed: its name (or its id) starts with it, a word inside starts with it,
// or it only holds it.
function matchRank(row, word) {
    if (!word) return 3;
    const id = row.id.toLocaleLowerCase();
    const tail = id.slice(id.lastIndexOf('/') + 1);
    if (tail.startsWith(word) || id.startsWith(word)) return 0;
    for (let at = id.indexOf(word); at > 0; at = id.indexOf(word, at + 1)) {
        if (!/[\p{L}\p{N}]/u.test(id[at - 1])) return 1;
    }
    return 2;
}

/**
 * Rows whose model or source hold every word of the query: across the sources (each model once per provider), or
 * the rows of `only`. Best match first, then the sources' order, each source's models newer first.
 */
export function searchModels(index, query, only = '') {
    const words = String(query || '').trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    const last = words.at(-1) || '';
    const found = [];
    index.rows.forEach((row, order) => {
        if (only ? row.source !== only : row.repeat) return;
        const own = index.source(row.source);
        const text = `${own.label} ${row.id} ${row.name}`.toLocaleLowerCase();
        if (words.every((word) => text.includes(word))) found.push({ row, rank: matchRank(row, last), order });
    });
    return found.sort((a, b) => a.rank - b.rank || a.order - b.order).map((entry) => entry.row);
}

/**
 * The route a typed id names: through the chosen source, else the only source there is, else as the route grammar
 * reads it (a bare id is OpenRouter's) when that source can lead here. Null when the owner has to pick a source first.
 */
export function typedRoute(index, typed, only = '') {
    const text = String(typed || '').trim();
    if (!text) return null;
    const through = only || (index.sources.length === 1 ? index.sources[0].id : '');
    if (through === LOCAL_SOURCE) return { model: text, use_local: true };
    if (through) return { model: composeModelSource(through, text), use_local: false };
    const route = { model: text, use_local: false };
    return index.source(routeSource(route)) ? route : null;
}
