import { accountRows, claudexorStatus, READ_OK } from './claudexor_status_store.js';
import { apiFetch, fetchJson } from './api_client.js';
import { setResult } from './ui_primitives.js';
import { catalogEntryIdentity } from './route_editor_primitives.js';
export const MODEL_CATALOG_TIMEOUT_MS = 25000;
let catalogRefreshSeq = 0;
const buttonRefreshes = new WeakMap();

// Account login/status is the authority for subscription model discovery. Keep
// one small signature of the confirmed account facts so a newly settled login
// (or a changed account) can refresh the existing catalog without polling or
// replacing the owner's in-memory model draft.
export function accountCatalogRefreshKey(view) {
    if (view?.reads?.accounts !== READ_OK) return null;
    return JSON.stringify(accountRows(view.snapshot || {}).map((row) => ({
        harness: row.harness,
        profile_id: row.profile_id,
        display_name: row.display_name,
        enabled: row.enabled,
        identity: {
            email: row.identity?.email || '',
            plan: row.identity?.plan || '',
        },
        verification: row.status?.verification || '',
        availability: row.status?.availability || '',
    })));
}

/** Own the Accounts subscription beside catalog refresh; never reload Settings. */
export function watchAccountModelCatalog(refresh = refreshModelCatalog) {
    let ready = false;
    let lastKey = null;
    let confirmed = false;
    const dispose = claudexorStatus.subscribe((view) => {
        const next = accountCatalogRefreshKey(view);
        if (!ready || next === null) {
            // An unread/failed facet resets confirmation, so reconnect refreshes
            // even unchanged accounts; an identical settled poll stays quiet.
            lastKey = next;
            return;
        }
        if (next === lastKey) return;
        lastKey = next;
        const firstConfirmation = !confirmed;
        confirmed = true;
        void refresh({ firstConfirmation });
    });
    return {
        arm() {
            lastKey = accountCatalogRefreshKey(claudexorStatus);
            confirmed ||= lastKey !== null;
            ready = true;
        },
        dispose,
    };
}

/**
 * Read provenance belongs to discovery, never to the owner's saved assignment.
 * One unreachable source among several is `partial`: the catalogs that did
 * answer are real, and claiming the whole read failed hid working API models.
 */
export function catalogReadState(data = {}) {
    if (data.read_state) return data.read_state;
    if (data.stale || data.freshness === 'stale') return 'stale';
    if (data.error || data.errors?.length) {
        return Array.isArray(data.items) && data.items.length ? 'partial' : 'failed';
    }
    if (data.partial) return 'partial';
    return Array.isArray(data.items) ? 'ok' : 'not_read';
}

/**
 * A read that produced no usable evidence of its own: the whole read failed or never reached the server. A read
 * that STATES its failure is one, whatever rows it carried; a read that only lists a source's errors is not — the
 * rows that came are real, and the catalog keeps its usual life (no read per opened capsule). A successful empty
 * read (the catalog really lists nothing) and a partial one are not failures either; both keep their rows, and only
 * a failure asks the next opening to read again.
 */
export function catalogReadFailed(data = {}) {
    const state = catalogReadState(data);
    if (state === 'transport') return true;
    if (state !== 'failed') return false;
    return data?.read_state ? true : !Array.isArray(data.items);
}

/** Enrich the existing editor state; a read gap cannot erase last-known choices. `current` is the answer this read
 *  itself gave (null when it listed nothing): what a gap keeps from an earlier read is shown and proves nothing of
 *  what an account or a transport serves now (chat_model_routes.js `tiers`). */
export function mergeModelCatalog(previous = {}, incoming = {}) {
    const read_state = catalogReadState(incoming);
    const retain = read_state !== 'ok';
    const merge = (key, identity) => {
        const before = Array.isArray(previous[key]) ? previous[key] : [];
        if (!Array.isArray(incoming[key])) return before;
        if (!retain) return incoming[key];
        const retained = key === 'items' ? before.flatMap((item) => {
            const account = (incoming.account_catalogs || []).filter((envelope) => !item.source_id || item.source_id === envelope.source)
                .flatMap((envelope) => envelope.accounts || []).find((entry) => entry.credentialProfileId === item.credential_profile_id);
            return account?.catalog ? [] : [{ ...item, ...(account ? { availability: account.availability, problem: account.problem } : {}) }];
        }) : before;
        const entries = new Map(retained.map((item) => [identity(item), item]));
        for (const item of incoming[key]) entries.set(identity(item), item);
        return [...entries.values()];
    };
    return { ...(retain ? previous : {}), ...incoming, read_state,
        sources_read_state: Array.isArray(incoming.model_sources) ? read_state : (read_state === 'ok' ? 'not_read' : read_state),
        errors: incoming.errors || (incoming.error ? [{ error: incoming.error }] : []),
        items: merge('items', (item) => JSON.stringify([item.value || item.id, item.credential_profile_id || ''])),
        model_sources: merge('model_sources', (source) => source.id),
        current: 'current' in incoming ? incoming.current : Array.isArray(incoming.items) ? incoming : null,
    };
}

/** Address one model inventory. Subscription ids and API connections are distinct namespaces. */
export function modelSlotCatalogQuery(source, profile = '') {
    if (!source || source === 'inherit') return '';
    const subscription = source.startsWith('subscription:');
    const query = new URLSearchParams(subscription ? { source_id: source.slice(13) } : { provider_id: source });
    if (subscription && profile) query.set('credential_profile_id', profile);
    return query.toString();
}

/** The catalogs read by one editor. Only an answer for the same source/account can replace its evidence.
 * A superseding refresh or a discarded draft retires its in-flight reads. Nothing here publishes a whole catalog. */
export function createScopedModelCatalog({ fetchCatalog = fetchJson, onUpdate = () => {},
    supportsScopes = () => true, legacyCatalog = () => null } = {}) {
    const snapshots = new Map();
    const requests = new Map();
    // Older cores ignore unfamiliar query parameters. Only use their already read full catalog;
    // a form selecting one provider never turns its lookup into a request to every configured key.
    function fetchScope(key, signal) {
        const params = new URLSearchParams(key);
        if (supportsScopes() || params.has('source_id')) {
            return fetchCatalog(`/api/model-catalog?${key}`, { cache: 'no-store', signal });
        }
        const previous = legacyCatalog();
        if (!previous) return { items: [], model_sources: [], read_state: 'not_read', errors: [{
            code: 'scoped_catalog_unsupported', error: 'This core lists all models together. Read the full model catalog.' }] };
        if (params.has('view')) return { ...previous, items: [] };
        const provider = params.get('provider_id');
        // A legacy row is scoped by the identity the catalog states (its `provider_id` and opaque `id` when it has
        // them), never by reparsing a value that may hold another connection's spelling.
        return { ...previous, items: (previous.items || []).filter(item => provider === 'local' ? item.use_local
            : !item.use_local && catalogEntryIdentity(item).source === provider) };
    }
    function reset() {
        for (const { controller } of requests.values()) controller.abort();
        requests.clear();
        snapshots.clear();
    }
    return {
        get(query) { return snapshots.get(String(query)); },
        invalidate(query) {
            const key = String(query);
            const pending = requests.get(key);
            pending?.controller.abort();
            requests.delete(key);
            snapshots.delete(key);
            return Boolean(pending);
        },
        read(query, { force = false, joinPending = false } = {}) {
            const key = String(query);
            const pending = requests.get(key);
            if (pending && (!force || joinPending)) return pending.promise;
            if (!force && snapshots.has(key)) return Promise.resolve(snapshots.get(key));
            pending?.controller.abort();
            const controller = new AbortController();
            let timer;
            const timeout = new Promise(resolve => {
                timer = setTimeout(() => {
                    resolve({ read_state: 'transport', errors: [{ error: 'Model catalog read timed out.' }] });
                    controller.abort();
                }, MODEL_CATALOG_TIMEOUT_MS);
            });
            const read = Promise.resolve().then(() => fetchScope(key, controller.signal)).then(data => {
                const field = key === 'view=sources' ? 'model_sources' : 'items';
                if (!Array.isArray(data?.[field])) throw new Error(data?.error || 'Model catalog response is unreadable.');
                return data;
            }).catch(error => ({ read_state: 'transport', errors: [{ error: error.message || String(error) }] }));
            const promise = Promise.race([read, timeout])
                .then(data => {
                    if (requests.get(key)?.controller !== controller) return;
                    requests.delete(key);
                    const merged = mergeModelCatalog(snapshots.get(key), data);
                    snapshots.set(key, merged);
                    onUpdate(key, merged);
                    return merged;
                }).finally(() => clearTimeout(timer));
            requests.set(key, { controller, promise });
            return promise;
        },
        reset,
        dispose: reset,
    };
}

/** Overlay successful connection-local lists on a general catalog, preserving its read provenance: a general
 * catalog nobody read yet stays unread (the overlay's own item list is no read of it).
 * An empty list replaces that source too; identical model names in other sources are unrelated. */
export function withConnectionCatalogs(catalog = {}, connectionCatalogs = {}) {
    return { ...catalog, read_state: catalogReadState(catalog), items: [
        ...(catalog.items || []).filter((item) => !Object.hasOwn(connectionCatalogs, catalogEntryIdentity(item).source)),
        ...Object.values(connectionCatalogs).flat(),
    ] };
}

/** One read of the catalog an unsaved setup draft's access gives (POST /api/onboarding/model-catalog; the server
 * knows a typed key only from the draft), bounded like every catalog read: `{ promise, abort }`. The promise answers
 * a catalog to merge; a transport failure or a timeout is one too (`read_state: 'transport'`), never a rejection. */
export function readDraftCatalog(draft) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), MODEL_CATALOG_TIMEOUT_MS);
    const promise = fetchJson('/api/onboarding/model-catalog', { method: 'POST', cache: 'no-store', signal: controller.signal,
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(draft) }).then((catalog) => {
        if (!catalog || typeof catalog !== 'object') throw new Error('Model catalog response is unreadable.');
        return catalog;
    }).catch((error) => ({ read_state: 'transport',
        errors: [{ error: error.name === 'AbortError' ? 'Model catalog request timed out.' : error.message }] }))
        .finally(() => clearTimeout(timer));
    return { promise, abort: () => controller.abort() };
}

/** Keep last-known native choices only for account reads that did not succeed. */
export function mergeHarnessModelCatalog(previous, current) {
    return { ...current, models: mergeModelCatalog({ items: previous?.models || [] }, {
        items: current.models || [], partial: current.model_catalog?.partial,
        account_catalogs: current.model_catalog ? [current.model_catalog] : [],
        errors: current.models_error ? [{ error: current.models_error }] : [],
    }).items };
}

const READ_ERROR_CAUSE_MAX = 160;
const READ_ERROR_NAMES_MAX = 3;

/**
 * One clause per cause, with the profiles that hit it named in front: nine
 * accounts sharing one quota limit is one fact, not nine. Every distinct cause
 * survives; repeated subjects collapse, names past three become a count, and a
 * cause past 160 characters is clamped with a visible ellipsis. httpx appends a
 * documentation pointer to every status error; the owner needs the status, not
 * the link. The vendor owns the message text, so its length is bounded here
 * rather than trusted.
 */
export function summarizeReadErrors(errors = []) {
    const groups = new Map();
    const seen = new Set();
    for (const error of errors) {
        const text = String(error?.error || error?.code || error?.provider_id || '')
            .split(/\s*For more information check:/)[0].trim().replace(/[.;,\s]+$/, '');
        const cause = text.length > READ_ERROR_CAUSE_MAX
            ? `${text.slice(0, READ_ERROR_CAUSE_MAX - 1)}…` : text;
        const subject = String(error?.credential_profile_id || '').trim();
        const identity = `${subject}\u0000${cause}`;
        if (seen.has(identity)) continue;
        seen.add(identity);
        if (!groups.has(cause)) groups.set(cause, []);
        if (subject) groups.get(cause).push(subject);
    }
    return [...groups].map(([cause, subjects]) => {
        if (!subjects.length) return cause;
        const names = subjects.length > READ_ERROR_NAMES_MAX
            ? `${subjects.slice(0, READ_ERROR_NAMES_MAX).join(', ')} and ${subjects.length - READ_ERROR_NAMES_MAX} more`
            : subjects.join(', ');
        return cause ? `${names}: ${cause}` : names;
    }).filter(Boolean).join('; ');
}

/**
 * Name what actually happened: which read failed, and what is usable anyway.
 * Read failures are stated one clause per cause, with the profiles named per
 * cause, so a shared vendor limit reads as one sentence instead of a list.
 * `compact` is the per-row form under a section banner that already lists the
 * failed reads: one short sentence, never the error list repeated per row.
 */
export function catalogReadNote(data = {}, { compact = false } = {}) {
    const state = catalogReadState(data);
    if (state === 'ok') return '';
    if (compact) {
        const short = state === 'partial' ? 'Some model sources could not be read.'
            : state === 'stale' ? 'Model catalog is last known.'
                : ['failed', 'transport'].includes(state) ? 'Model catalog could not be read.'
                    : 'Model catalog has not been read yet.';
        return `${short} Existing suggestions and your selection are kept.`;
    }
    const errors = data.errors || [];
    const clauses = summarizeReadErrors(errors);
    const loaded = (Array.isArray(data.items) ? data.items : []).filter(
        (item) => !String(item?.value || item?.id || '').startsWith('claudexor::')).length;
    const reason = state === 'partial' && errors.length
        ? `Some model sources could not be read: ${clauses}.${loaded
            ? ` ${loaded} API model${loaded === 1 ? '' : 's'} loaded.` : ''}`
        : errors.length ? `Model catalog could not be read: ${clauses}.`
            : state === 'stale' ? 'Model catalog is last known.' : state === 'partial' ? 'Some account model lists could not be read.'
                : ['failed', 'transport'].includes(state) ? 'Model catalog could not be read.' : 'Model catalog has not been read yet.';
    return `${reason} Existing suggestions and your selection are kept. To retry, press Read again in Models.`;
}

// The catalog's result beside its Refresh button (ui.css .ui-action-row): a count ends by itself, a gap stays.
function setCatalogStatus(statusEl, text, tone = 'muted') {
    setResult(statusEl, { text, tone, ends: tone === 'ok' });
}

function broadcastCatalog(data) {
    document.dispatchEvent(new CustomEvent('settings-model-catalog:updated', {
        detail: data,
    }));
}

function fillCatalogDatalist(data) {
    const list = document.getElementById('settings-model-catalog');
    if (list) {
        const { items } = mergeModelCatalog({ items: [...list.options].map((option) => ({ value: option.value, label: option.label })) }, data);
        list.innerHTML = '';
        for (const item of items) {
            const option = document.createElement('option');
            option.value = item.value || item.id || '';
            option.label = item.label || item.provider || '';
            list.appendChild(option);
        }
    }
    broadcastCatalog(data);
}

/** A connection changed: a catalog read already out answers for the earlier source, so it is not broadcast (nor
 *  adopted by any reader); the next refresh is this source's. */
export function invalidateModelCatalogReads() {
    catalogRefreshSeq += 1;
}

export async function refreshModelCatalog({ button } = {}) {
    const refreshSeq = ++catalogRefreshSeq;
    const statusEl = document.getElementById('settings-model-catalog-status');
    setCatalogStatus(statusEl, 'Refreshing model catalog...', 'muted');
    if (button) {
        buttonRefreshes.set(button, refreshSeq);
        button.disabled = true;
        button.setAttribute('aria-busy', 'true');
    }
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), MODEL_CATALOG_TIMEOUT_MS);

    try {
        const resp = await apiFetch('/api/model-catalog', {
            cache: 'no-store',
            signal: controller.signal,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
        if (!Array.isArray(data?.items)) throw new Error(data?.error || 'Model catalog response has no model list');

        const items = Array.isArray(data.items) ? data.items : [];
        const errors = Array.isArray(data.errors) ? data.errors : [];
        if (refreshSeq !== catalogRefreshSeq) {
            return { items, errors, stale: true };
        }
        const read_state = catalogReadState(data);
        fillCatalogDatalist({ ...data, read_state });

        if (read_state !== 'ok') {
            setCatalogStatus(statusEl, catalogReadNote({ ...data, read_state }), 'warn');
        } else if (items.length) {
            setCatalogStatus(statusEl, `Loaded ${items.length} models.`, 'ok');
        } else {
            setCatalogStatus(statusEl, 'No provider catalogs available yet. This is optional.', 'muted');
        }
        return { ...data, items, errors, read_state };
    } catch (err) {
        if (refreshSeq !== catalogRefreshSeq) {
            return { items: [], errors: [{ provider_id: 'catalog', error: 'stale refresh' }], stale: true };
        }
        const message = err?.name === 'AbortError'
            ? `Timed out after ${Math.round(MODEL_CATALOG_TIMEOUT_MS / 1000)}s`
            : (err.message || err);
        fillCatalogDatalist({ read_state: 'transport', errors: [{ provider_id: 'catalog', error: String(message) }] });
        setCatalogStatus(
            statusEl,
            `Model catalog failed: ${message}. This is optional.`,
            'warn',
        );
        return { items: [], errors: [{ provider_id: 'catalog', error: String(message) }] };
    } finally {
        clearTimeout(timeoutId);
        // Global freshness governs data, not a particular button's busy lease.
        if (button && buttonRefreshes.get(button) === refreshSeq) {
            buttonRefreshes.delete(button);
            button.disabled = false;
            button.removeAttribute('aria-busy');
        }
    }
}

const settledPrograms = new Map(), programReads = new Map();
export function refreshMaintenanceCatalogs(harness, evidence) {
    const operation = evidence.id && `${evidence.id}:${evidence.finishedAt || ''}`;
    if (operation && settledPrograms.get(harness) === operation) return;
    let read = programReads.get(harness);
    if (!read) {
        document.dispatchEvent(new CustomEvent('harness-maintenance:settled', {detail: {harness}}));
        let timeout;
        const expired = new Promise(resolve => { timeout = setTimeout(() => resolve(false), MODEL_CATALOG_TIMEOUT_MS); });
        read = Promise.race([expired, Promise.allSettled([claudexorStatus.refresh({includeModels: true, afterCurrent: true}), refreshModelCatalog()])
            .then(([native, raw]) => {
                const row = native.value?.harnesses?.find((item) => item.id === harness);
                return native.status === 'fulfilled' && !claudexorStatus.error && row?.model_catalog
                    && !row.models_error && !row.model_catalog.partial
                    && raw.status === 'fulfilled' && catalogReadState(raw.value) === 'ok';
            })]).finally(() => { clearTimeout(timeout); programReads.delete(harness); });
        programReads.set(harness, read);
    }
    return read.then((succeeded) => {
        if (succeeded && operation) settledPrograms.set(harness, operation);
        return succeeded;
    });
}
