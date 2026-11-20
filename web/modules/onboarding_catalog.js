// The wizard's catalog reads use the access the owner has brought to Models (or explicitly retried).
// Passive account discovery only asks the engine which subscription sources exist.
import { compatibleEndpoint } from './route_editor_primitives.js';
import { fetchJson } from './api_client.js';
import { createScopedModelCatalog, readDraftCatalog } from './settings_catalog.js';

export function createOnboardingCatalog({ draftSettings, providerFields, supportsScopes, onReset, onSources, onFull, changed }) {
    let access = null, pendingFull = null, pendingSources = null, sourceRevision = 0;
    const descriptors = createScopedModelCatalog({ onUpdate: (_query, data) => onSources(data) });
    function identity(draft) {
        return JSON.stringify([draft.OUROBOROS_CONNECTIONS, providerFields.map(field => draft[field.settingKey]),
            draft.OPENAI_API_SURFACE, draft.LOCAL_MODEL_SOURCE, draft.LOCAL_MODEL_FILENAME, draft.LOCAL_MODEL_CONTEXT_LENGTH]);
    }
    function accept() {
        const next = identity(draftSettings());
        if (next === access) return;
        pendingFull?.abort(); pendingFull = null;
        access = next;
        onReset();
    }
    return {
        accept,
        allowed: () => access !== null && access === identity(draftSettings()),
        readApi(url, init) {
            const query = new URLSearchParams(url.split('?')[1]);
            if (!query.has('provider_id') || access !== identity(draftSettings())) throw new Error('The model source has changed.');
            return fetchJson(`/api/onboarding/model-catalog?${query}`, { ...init, method: 'POST',
                headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(draftSettings()) });
        },
        sources({ force = false } = {}) {
            if (force) sourceRevision++;
            if (!supportsScopes()) return;
            if (pendingSources && !force) return pendingSources;
            const read = descriptors.read('view=sources', { force });
            pendingSources = read;
            changed();
            return read.finally(() => { if (read === pendingSources) { pendingSources = null; changed(); } });
        },
        async full() {
            accept();
            pendingFull?.abort();
            const read = pendingFull = readDraftCatalog(draftSettings());
            const revision = sourceRevision;
            changed();
            const data = await read.promise;
            if (read !== pendingFull) return;
            pendingFull = null;
            if (access === identity(draftSettings())) {
                if (!supportsScopes() && revision === sourceRevision) onSources(data);
                onFull(data);
            }
            changed();
        },
        get pending() { return Boolean(pendingFull || pendingSources); },
        dispose() {
            descriptors.dispose(); pendingSources = null;
            pendingFull?.abort(); pendingFull = null;
        },
    };
}

export function bindCompatibleModelLoader(draftSettings, adopt) {
    const button = document.getElementById('load-compatible-models');
    const select = document.getElementById('compatible-model-source');
    const status = document.getElementById('compatible-load-status');
    if (!button) return;
    select.addEventListener('change', () => { status.textContent = 'Load this server’s models, or type a model ID below.'; });
    button.addEventListener('click', async () => {
        const source = select.value;
        const name = source.split('@')[1] || '';
        const row = name ? (draftSettings().OUROBOROS_CONNECTIONS || []).find((entry) => entry.kind === 'openai-compatible' && entry.name === name) : null;
        // The built-in source sends the one pair the draft reaches, naming whose key a kept one is.
        const { baseUrl, apiKey, apiKeySetting } = name ? { baseUrl: row?.base_url, apiKey: row?.api_key || '' }
            : compatibleEndpoint(draftSettings());
        const payload = { baseUrl, apiKey, ...(name ? { connection: name } : { apiKeySetting }) };
        const sent = JSON.stringify(payload);
        status.textContent = 'Loading…';
        button.disabled = true;
        try {
            const data = await fetchJson('/api/openai-compatible/models', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: sent, cache: 'no-store',
            });
            if (!button.isConnected) return;
            const models = (data.models || []).map(id => String(id || '').trim()).filter(Boolean).sort();
            adopt(source, models.map((id) => ({ value: `${source}::${id}`, label: id })));
            if (select.value === source) status.textContent = models.length
                ? `${models.length} model${models.length === 1 ? '' : 's'} listed for this server. Choose one in its role below.`
                : 'No models listed. You can type a model ID below.';
        } catch (error) {
            if (button.isConnected && select.value === source) status.textContent = `Could not load models: ${error.message}. You can type a model ID below.`;
        } finally {
            if (button.isConnected) button.disabled = false;
        }
    });
}
