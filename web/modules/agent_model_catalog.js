// A model on an agent can name an API connection, a subscription transport, or an agent session.
// The native program's session model catalog is independent of a raw transport with a similar name.
import { fetchJson } from './api_client.js';
import { modelCatalog } from './model_catalog_store.js';
import { DEFAULT_API_PROVIDER, ROUTE_KIND_AGENT_SESSION, connectionParts, routeModelFields, tierPriced } from './route_editor_primitives.js';
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
                else sessions.set(query, { ...prior, models: prior?.models || [],
                    models_error: next?.models_error || data.errors?.[0]?.error || 'Session model catalog unavailable.' });
            }
            if (query === 'view=sources') adoptSources(data);
            else if (rows().some(row => catalogQuery(row) === query)) changed();
        } });
    function read(row, force = false) {
        const query = catalogQuery(row);
        if (!query || (!saved && !readApiCatalog)) return;
        const { provider } = routeModelFields(row.route);
        // Automatic list suppression before dedup lives in `refresh` (fields.local); an explicit `open()` for a
        // configured selected connection is allowed to load provider alternatives.
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
        open(row) {
            const prior = scoped.get(catalogQuery(row));
            return read(row, ['failed', 'transport'].includes(prior?.read_state));
        },
        get(row) { return scoped.get(catalogQuery(row)); },
        session(row) { return sessions.get(catalogQuery(row)); },
        // `list`: only what a list head draws — a session's verdict, an OpenRouter key's published rate, and the
        // program behind a subscription source (its descriptors, read with the roster, not on a sign-in; never that
        // source's model list).
        refresh({ descriptors = false, accounts = false, list = false, harness = '', force = true } = {}) {
            const roster = rows().filter(Boolean);
            const marks = list && !harness && !accounts && roster.some((row) => {
                const fields = routeModelFields(row.route);
                return fields.subscription && !fields.local;
            });
            if ((descriptors || marks) && saved) void scoped.read('view=sources', { force });
            // The screen comes BEFORE the roster's dedup: one query can serve a row that shows a rate and a row that
            // shows none (a Fast or Economy twin pays another rate), and only the row that shows one needs the read.
            const asked = new Set();
            for (const row of roster) {
                const fields = routeModelFields(row.route, state.modelSources);
                // A local row runs the local runtime: it has no provider or session catalog to price or mark, so
                // the automatic pass never reads one for it. An explicit chooser opening still reads its own
                // source and offers its models as the alternatives.
                if (fields.local) continue;
                if ((list && row.route.kind !== ROUTE_KIND_AGENT_SESSION
                        && (tierPriced(row.processing_preference || state.processingPreference)
                            || connectionParts(fields.provider).kind !== DEFAULT_API_PROVIDER))
                    || (accounts && fields.provider) || (harness && fields.harness !== harness)) continue;
                const query = catalogQuery(row);
                if (!query || asked.has(query)) continue;
                asked.add(query);
                if (saved || requested.has(query)) void read(row, force);
            }
        },
        reset() { requested.clear(); sessions.clear(); scoped.reset(); },
    };
}
