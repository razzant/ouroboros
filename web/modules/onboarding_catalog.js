// The wizard's catalog reads use the access the owner has brought to Models (or explicitly retried).
// Passive account discovery only asks the engine which subscription sources exist.
import { fetchJson } from './api_client.js';
import { createScopedModelCatalog, readDraftCatalog } from './settings_catalog.js';

export function createOnboardingCatalog({ draftSettings, providerFields, onReset, onSources, onFull, changed }) {
    let access = null, pendingFull = null, pendingSources = null;
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
            changed();
            const data = await read.promise;
            if (read !== pendingFull) return;
            pendingFull = null;
            if (access === identity(draftSettings())) onFull(data);
            changed();
        },
        get pending() { return Boolean(pendingFull || pendingSources); },
        dispose() {
            descriptors.dispose(); pendingSources = null;
            pendingFull?.abort(); pendingFull = null;
        },
    };
}
