// A model on an agent can name an API connection, a subscription transport, or an agent session.
// The native program's session model catalog is independent of a raw transport with a similar name.
import { fetchJson } from './api_client.js';
import { modelCatalog } from './model_catalog_store.js';
import { ROUTE_KIND_AGENT_SESSION, routeModelFields } from './route_editor_primitives.js';
import { createScopedModelCatalog, mergeHarnessModelCatalog, modelSlotCatalogQuery } from './settings_catalog.js';

function catalogQuery(row) {
    if (!row?.route) return '';
    const { subscription, source, provider, harness } = routeModelFields(row.route);
    const pin = row.route.credential_profile_id || row.route.profile_id || '';
    if (row.route.kind !== ROUTE_KIND_AGENT_SESSION) return modelSlotCatalogQuery(subscription ? `subscription:${source}` : provider, pin);
    if (!harness) return '';
    const query = new URLSearchParams({ harness_id: harness });
    if (pin) query.set('credential_profile_id', pin);
    return query.toString();
}

export function createAgentModelCatalog({ state, saved, rows, readApiCatalog, apiCatalogAllowed, supportsScopes, readLegacySessions, changed, adoptSources }) {
    const requested = new Set();
    const sessions = new Map();
    const scoped = createScopedModelCatalog({ supportsScopes, legacyCatalog: () => saved ? modelCatalog.catalog() : state.modelCatalog,
        fetchCatalog: (url, init) => (readApiCatalog && new URLSearchParams(url.split('?')[1]).has('provider_id') ? readApiCatalog : fetchJson)(url, init),
        onUpdate: (query, data) => {
            if (query.startsWith('harness_id=')) {
                const prior = sessions.get(query);
                const next = data.session_catalog;
                if (next && !['failed', 'transport'].includes(data.read_state)) sessions.set(query, mergeHarnessModelCatalog(prior, next));
                else sessions.set(query, { models: prior?.models || [],
                    models_error: next?.models_error || data.errors?.[0]?.error || 'Session model catalog unavailable.' });
            }
            if (query === 'view=sources') adoptSources(data);
            else if (rows().some(row => catalogQuery(row) === query)) changed();
        } });
    function read(row, force = false) {
        const query = catalogQuery(row);
        if (!query || (!saved && !readApiCatalog)) return;
        const { provider } = routeModelFields(row.route);
        if (provider && (!apiCatalogAllowed() || !state.providers.some(entry => entry.id === provider))) return;
        const had = requested.has(query);
        requested.add(query);
        if (query.startsWith('harness_id=') && !supportsScopes()) {
            if (force || !had) return readLegacySessions();
            return;
        }
        return scoped.read(query, { force });
    }
    return {
        read,
        get(row) { return scoped.get(catalogQuery(row)); },
        session(row) { return sessions.get(catalogQuery(row)); },
        refresh({ descriptors = false, accounts = false, sessionsOnly = false, harness = '', force = true } = {}) {
            if (descriptors && saved) void scoped.read('view=sources', { force });
            const current = new Map(rows().filter(Boolean).map(row => [catalogQuery(row), row]));
            for (const [query, row] of current) {
                if (saved || requested.has(query)) {
                    const fields = routeModelFields(row.route, state.modelSources);
                    if ((sessionsOnly && row.route.kind !== ROUTE_KIND_AGENT_SESSION) || (accounts && fields.provider) || (harness && fields.harness !== harness)) continue;
                    void read(row, force);
                }
            }
        },
        reset() { requested.clear(); sessions.clear(); scoped.reset(); },
    };
}
