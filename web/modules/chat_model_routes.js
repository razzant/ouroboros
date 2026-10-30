// What the chat model picker shows (DESIGN "Composer owner controls"), as data: the room's choice and the two
// references it may name, the sources and routes the model catalog offers (newer first), the owner's stars, a search
// across all of them. composer_model_picker.js draws it. The catalog suggests and never admits: a saved, starred or
// typed route no catalog lists stays a route of its own source, said so, never dropped or rewritten. A route's
// identity is its whole string, source included, so one model through two sources is two routes (and two stars).
import {
    API_PROVIDER_ORDER, apiProviderLabel, composeModelSource, connectionParts, parseModelSource,
} from './route_editor_primitives.js';

export const LANE_LABELS = Object.freeze({ favorites: 'Favorites', subscription: 'Subscription', api: 'API' });
export const LANES = Object.freeze(['favorites', 'subscription', 'api']);
export const MAIN_REFERENCE = Object.freeze({ kind: 'main' });
export const LOCAL_SOURCE = 'local';
const SUBSCRIPTION = 'subscription:';
const CLAUDEXOR = 'claudexor::';

/** The canonical room choice; anything unreadable is Main's reference, which is what the room then runs as. */
export function normalizeChoice(raw) {
    if (raw?.kind === 'fallback') return { kind: 'fallback' };
    if (raw?.kind === 'model' && typeof raw.model === 'string' && raw.model.trim()) {
        return { kind: 'model', model: raw.model.trim(), use_local: raw.use_local === true };
    }
    return { ...MAIN_REFERENCE };
}

export const sameChoice = (a, b) => {
    const x = normalizeChoice(a);
    const y = normalizeChoice(b);
    return x.kind === y.kind && (x.kind !== 'model' || (x.model === y.model && x.use_local === y.use_local));
};

/** The source a route is reached through: `subscription:<id>`, an API connection (`kind`, `kind@name`) or local. */
export function routeSource(route, useLocal = false) {
    if (useLocal) return LOCAL_SOURCE;
    return parseModelSource(route).source || 'openrouter';
}

const laneOf = (source) => (source.startsWith(SUBSCRIPTION) ? 'subscription' : 'api');
const modelPart = (route) => parseModelSource(route).model || String(route || '');

// Newer models first (owner 30.10). No catalog states a release date, so the order reads the version a name states:
// a name compares part by part, its words A to Z and its numbers high to low, a dotted version as one number (6.1
// before 6 before 5.6, 6.10 before 6.9), so within one family (the words before its first number) the higher
// version comes first and families follow each other by name. A name that states no number claims no age and
// follows. A presentation convention only: it never claims a release date, never filters, and Favorites keep the
// order they were starred in.
const NAME_PARTS = /(\d+(?:\.\d+)*)|(\D+)/gu;
const nameParts = (name) => [...String(name || '').toLocaleLowerCase().matchAll(NAME_PARTS)]
    .map(([, number, words]) => (number !== undefined ? number.split('.').map(Number)
        : words.replace(/[^\p{L}\p{N}]+/gu, ' ').trim()));

/** Two display names in the picker's newer-first order (negative: `a` first). */
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

/** A route a source can send: what the owner types, in that source's own spelling. */
export function customRoute(source, typed) {
    const model = String(typed || '').trim();
    if (!model || source === LOCAL_SOURCE) return '';
    return composeModelSource(source, model);
}

/**
 * Read one `/api/model-catalog` answer (or the picker's merged copy) together with the references, the room's choice
 * and the stars into the picker's model. Subscription sources are the catalog's own `model_sources`, never a fixed
 * list; API sources are the connections whose catalogs answered or failed. A source the choice, a reference or a star
 * names stays listed even when no catalog mentions it.
 * `names` are display names an earlier read listed (`rememberedNames`): they name a route this read does not list yet,
 * never make it listed.
 * @param {{catalog?: object, routes?: {main?: {model:string,use_local:boolean}, fallback?: ?{model:string,use_local:boolean}},
 *   choice?: object, favorites?: string[], names?: Object<string,string>}} input
 */
export function buildRouteModel({ catalog = {}, routes = {}, choice = MAIN_REFERENCE, favorites = [], names = {} } = {}) {
    const sources = new Map();
    const rows = new Map();
    const descriptors = Array.isArray(catalog.model_sources) ? catalog.model_sources : [];
    const items = Array.isArray(catalog.items) ? catalog.items : [];
    const errors = Array.isArray(catalog.errors) ? catalog.errors : [];
    const problems = [];

    function source(id, extra = {}) {
        if (!sources.has(id)) {
            const lane = id === LOCAL_SOURCE ? 'api' : laneOf(id);
            const sub = lane === 'subscription' ? id.slice(SUBSCRIPTION.length) : '';
            const descriptor = descriptors.find((entry) => entry?.id === sub);
            const { kind, name } = connectionParts(id);
            sources.set(id, {
                id, lane, listed: false, problems: [],
                label: id === LOCAL_SOURCE ? 'Local model' : lane === 'subscription' ? String(descriptor?.label || sub)
                    : apiProviderLabel(id),
                harness: lane === 'subscription' ? String(descriptor?.credentialHarness || sub) : '',
                kind: lane === 'subscription' ? '' : (id === LOCAL_SOURCE ? LOCAL_SOURCE : kind),
                named: Boolean(name),
            });
        }
        return Object.assign(sources.get(id), extra);
    }
    for (const entry of descriptors) if (entry?.id) source(`${SUBSCRIPTION}${entry.id}`, { listed: true });

    function row(route) {
        const id = routeSource(route);
        source(id);
        if (!rows.has(route)) {
            const remembered = typeof names?.[route] === 'string' && names[route] ? names[route] : '';
            rows.set(route, { route, model: modelPart(route), name: remembered || modelPart(route), source: id, lane: laneOf(id),
                listed: false, shipped: false });
        }
        return rows.get(route);
    }
    for (const item of items) {
        const route = String(item?.value || '');
        if (!route) continue;
        const entry = row(route);
        // One row per route however many accounts list it: the first listing names it, and it reads as the
        // engine's shipped list only while every listing is (`origin: "hint"`).
        if (!entry.listed) Object.assign(entry, { name: String(item?.name || '') || entry.model, shipped: true });
        entry.listed = true;
        entry.shipped = entry.shipped && item?.origin === 'hint';
        const own = sources.get(entry.source);
        own.listed = true;
        if (own.named && item?.provider_label) own.label = String(item.provider_label);  // the connection's own name
    }
    for (const error of errors) {
        const id = error?.source_id ? `${SUBSCRIPTION}${error.source_id}`
            : (error?.provider_id && !['claudexor', 'catalog', 'extra_ca_bundle'].includes(error.provider_id)
                ? String(error.provider_id) : '');
        const text = String(error.error || error.code || '').split(/\s*For more information check:/)[0].trim();
        if (!id) {
            problems.push({ lane: error?.provider_id === 'claudexor' ? 'subscription' : '', text });
            continue;
        }
        const who = String(error.credential_profile_id || '');
        source(id).problems.push(who ? `${who}: ${text}` : text);
    }
    const chosen = normalizeChoice(choice);
    const pinned = [];
    if (chosen.kind === 'model') pinned.push([chosen.model, chosen.use_local]);
    for (const ref of [routes?.main, routes?.fallback]) if (ref?.model) pinned.push([ref.model, ref.use_local === true]);
    for (const [route, local] of pinned) {
        if (local) source(LOCAL_SOURCE);
        else if (!rows.has(route)) row(route);
    }
    const stars = [...new Set((Array.isArray(favorites) ? favorites : []).map((value) => String(value || '').trim()).filter(Boolean))];
    for (const route of stars) if (!rows.has(route)) row(route);

    // Subscriptions in the catalog's own order, then API connections in the owner-facing provider order (a named
    // key after its built-in one), then a local route.
    const rank = (entry) => {
        if (entry.lane === 'subscription') {
            const at = descriptors.findIndex((item) => `${SUBSCRIPTION}${item?.id}` === entry.id);
            return at < 0 ? descriptors.length : at;
        }
        const at = API_PROVIDER_ORDER.indexOf(entry.kind);
        return 1000 + (entry.id === LOCAL_SOURCE ? 999 : (at < 0 ? 900 : at) * 2 + (entry.named ? 1 : 0));
    };
    const sorted = [...sources.values()].sort((a, b) => rank(a) - rank(b) || a.label.localeCompare(b.label));
    // Every list of routes keeps the rail's order of sources, each source's routes newer first.
    const place = new Map(sorted.map((entry, at) => [entry.id, at]));
    const ordered = [...rows.values()].sort((a, b) => place.get(a.source) - place.get(b.source)
        || compareNewerFirst(a.name, b.name) || a.route.localeCompare(b.route));
    return {
        problems,
        sources: sorted,
        source: (id) => sources.get(id) || null,
        laneSources: (lane) => sorted.filter((entry) => entry.lane === lane),
        rows: ordered,
        sourceRows: (id) => ordered.filter((entry) => entry.source === id),
        laneRows: (lane) => ordered.filter((entry) => entry.lane === lane),
        row: (route) => rows.get(route) || null,
        favorites: stars.map((route) => rows.get(route)).filter(Boolean),
        starred: new Set(stars),
    };
}

/**
 * What a route (a reference's binding or the room's own) is called: the catalog's name when it lists the route, else
 * the model id as written; its source by its own label.
 */
export function describeRoute(model, { route = '', use_local: useLocal = false } = {}) {
    const text = String(route || '');
    if (!text) return null;
    const id = routeSource(text, useLocal);
    const listed = useLocal ? null : model.row(text);
    const own = model.source(id);
    return {
        route: text, local: useLocal, source: id, lane: id === LOCAL_SOURCE ? 'api' : laneOf(id),
        name: listed?.name || (useLocal ? text : modelPart(text)),
        sourceLabel: own?.label || (id === LOCAL_SOURCE ? 'Local model' : laneOf(id) === 'subscription' ? id.slice(SUBSCRIPTION.length) : apiProviderLabel(id)),
        harness: own?.harness || '', kind: own?.kind || '', listed: Boolean(listed?.listed),
    };
}

/**
 * The room's choice as the composer says it: what new work starts on (`current`), which reference it is, if any,
 * and both references for the footer. A Fallback reference with no Fallback configured admits work on Main, and
 * says so (`unconfigured`).
 */
export function resolveChoice(model, choice, routes = {}) {
    const chosen = normalizeChoice(choice);
    const main = routes?.main?.model ? describeRoute(model, { route: routes.main.model, use_local: routes.main.use_local === true }) : null;
    const fallback = routes?.fallback?.model
        ? describeRoute(model, { route: routes.fallback.model, use_local: routes.fallback.use_local === true }) : null;
    let current = main;
    if (chosen.kind === 'fallback') current = fallback || main;
    else if (chosen.kind === 'model') current = describeRoute(model, { route: chosen.model, use_local: chosen.use_local });
    return { choice: chosen, current, main, fallback, unconfigured: chosen.kind === 'fallback' && !fallback };
}

/** Rows whose name, model id or source match every word of a query, across all sources. */
export function searchRoutes(model, query) {
    const words = String(query || '').trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    return model.rows.filter((entry) => {
        const own = model.source(entry.source);
        const text = `${entry.name} ${entry.model} ${own?.label || ''} ${LANE_LABELS[entry.lane]}`.toLocaleLowerCase();
        return words.every((word) => text.includes(word));
    });
}

/** The route a list row stands for, as a choice: list rows are explicit routes, never a reference. */
export const rowChoice = (entry) => ({ kind: 'model', model: entry.route, use_local: false });

/** Which tab and source a reopened picker starts on: where the room's current route lives. */
export function homeOf(model, resolved) {
    const current = resolved?.current;
    if (!current) return { lane: 'subscription', source: model.laneSources('subscription')[0]?.id || '' };
    return { lane: current.lane, source: current.source };
}

export const isSubscriptionRoute = (route) => String(route || '').startsWith(CLAUDEXOR);

/** The display names a catalog read listed, merged over the remembered ones (newest last, at most `limit`). */
export function rememberedNames(previous = {}, catalog = {}, limit = 400) {
    const names = new Map(Object.entries(previous && typeof previous === 'object' ? previous : {}));
    for (const item of Array.isArray(catalog?.items) ? catalog.items : []) {
        const route = String(item?.value || '');
        const name = String(item?.name || '');
        if (!route || !name || name === modelPart(route)) continue;
        names.delete(route);
        names.set(route, name);
    }
    return Object.fromEntries([...names].slice(-limit));
}
