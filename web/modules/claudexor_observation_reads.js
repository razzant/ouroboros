// Independent account/quota observations beside the slower catalog read.
// Dispatch order belongs to this client; provider observation dates remain unchanged.
import { apiFetch } from './api_client.js';

const facets = {
    catalog: ['harnesses'],
    accounts: ['profiles', 'unified_accounts'],
    quota: ['quota', 'quota_absences', 'resources', 'availability_scope'],
};
const object = value => value && typeof value === 'object' && !Array.isArray(value);
export function passiveStatusValid(value) {
    return value?.view === 'quota' && object(value.profiles) && Array.isArray(value.quota)
        && Array.isArray(value.quota_absences) && object(value.reads)
        && ['accounts', 'quota'].every(key => ['ok', 'failed', 'not_read'].includes(value.reads[key]));
}

export function createObservationReads({ fetchImpl = apiFetch, getSnapshot, commit, changed }) {
    let sequence = 0, flight = null, follow = null, disposed = false;
    const versions = { catalog: 0, accounts: 0, quota: 0 };
    const actions = new Map();
    const keyFor = (harness, profile) => JSON.stringify([harness, profile]);
    let roster = null, projected = null, loginPolicy, policyStamp = 0, accountRead;
    function ticket() { return ++sequence; }
    function merge(payload, stamp, { passive = false } = {}) {
        const held = getSnapshot();
        // A confirmed row action can commit directly between reads. Keep its
        // corrected visible rows while retaining raw, not-yet-eligible rows.
        if (roster && projected && held?.profiles !== projected) {
            const current = new Map((held?.profiles?.profiles || []).map(row =>
                [keyFor(row.profile?.harness_id, row.profile?.profile_id), row]));
            const prior = new Set((projected.profiles || []).map(row =>
                keyFor(row.profile?.harness_id, row.profile?.profile_id)));
            roster = { ...roster, ...held?.profiles, profiles: (roster.profiles || []).flatMap(row => {
                const key = keyFor(row.profile?.harness_id, row.profile?.profile_id);
                return current.has(key) ? [current.get(key)] : prior.has(key) ? [] : [row];
            }) };
        }
        // Manifest policy has its own read order: an older full read may supply
        // the first policy after a newer passive roster has already answered.
        // A legacy full producer already applied its own filter; retain its
        // established fail-open behavior, never infer policy from catalog IDs.
        if (!passive && stamp >= policyStamp && (Object.hasOwn(payload.profiles || {}, 'login_capable_harnesses')
            || !payload.reads || payload.reads.accounts === 'ok')) {
            policyStamp = stamp;
            const policy = payload.profiles?.login_capable_harnesses;
            loginPolicy = policy == null ? null : Array.isArray(policy) ? policy : undefined;
        }
        const out = { profiles: {}, quota: [], harnesses: [], daemon: {},
            ...(held || {}), ...(passive ? {} : payload),
            reads: { ...(held?.reads || {}), ...(passive ? {} : payload.reads) },
            facets: { ...(held?.facets || {}), ...(passive ? {} : payload.facets) } };
        for (const [name, fields] of Object.entries(facets)) {
            const pending = name === 'accounts' && [...actions.values()].some(action => action.busy);
            if ((passive && name === 'catalog') || stamp < versions[name] || pending) {
                for (const field of fields) {
                    if (held && field in held) out[field] = held[field];
                    else delete out[field];
                }
                out.reads[name] = held?.reads?.[name] || 'not_read';
                if (held?.facets?.[name]) out.facets[name] = held.facets[name];
                else delete out.facets[name];
                continue;
            }
            versions[name] = stamp;
            const state = payload.reads?.[name];
            if (passive) {
                out.reads[name] = state;
                out.facets[name] = { observed_at: state === 'ok' ? new Date().toISOString()
                    : held?.facets?.[name]?.observed_at ?? null,
                stale: state !== 'ok' && Boolean(held),
                error: payload.read_errors?.[name]?.code || payload.read_errors?.discovery?.code || null };
            }
            if (name === 'accounts') accountRead = state;
            if (passive && state !== 'ok') continue;
            if (name === 'accounts') {
                // A passive roster does not read model marks; retain them as explicitly old.
                const marks = held?.profiles?.routing_evidence;
                if (passive && marks && payload.profiles) payload = { ...payload,
                    profiles: { ...payload.profiles, routing_evidence: marks } };
            }
            for (const field of fields) {
                if (field in payload) out[field] = payload[field];
                else if (!passive || field !== 'resources') delete out[field];
            }
            if (name === 'accounts') roster = out.profiles;
        }
        if (roster) out.profiles = projected = Array.isArray(roster.profiles) ? { ...roster,
            profiles: (roster.profiles || []).filter(row => loginPolicy === null
                || loginPolicy?.includes(row.profile?.harness_id)),
            ...(loginPolicy !== undefined ? { login_capable_harnesses: loginPolicy } : {}) } : roster;
        // Raw membership before the first full read cannot offer login actions.
        if (accountRead) out.reads.accounts = loginPolicy === undefined && accountRead === 'ok' ? 'not_read' : accountRead;
        if (!passive && !payload.reads && !held?.reads) delete out.reads;
        if (!Object.keys(out.facets).length && !payload.facets && !held?.facets) delete out.facets;
        return out;
    }
    function observe({ afterCurrent = false } = {}) {
        if (disposed) return Promise.resolve(getSnapshot());
        if (flight) {
            if (!afterCurrent) return flight;
            if (!follow) follow = flight.then(() => { follow = null; return observe(); });
            return follow;
        }
        const stamp = ticket();
        flight = (async () => {
            try {
                const response = await fetchImpl('/api/claudexor/status?view=quota', { cache: 'no-store' });
                const value = await response.json().catch(() => null);
                if (!response.ok || !passiveStatusValid(value)) throw new Error(value?.error || 'Account observations unavailable');
                if (!disposed) commit(merge(value, stamp, { passive: true }));
            } catch (error) {
                if (!disposed) commit(merge({ reads: { accounts: 'failed', quota: 'failed' },
                    read_errors: { discovery: { code: String(error?.message || error) } } }, stamp, { passive: true }));
            } finally {
                flight = null;
                if (!disposed) changed();
            }
            return getSnapshot();
        })();
        return flight;
    }
    async function setEnabled(harness, profile, enabled) {
        const key = keyFor(harness, profile);
        if (disposed || actions.get(key)?.busy) return null;
        const action = { busy: true, error: '', requestedEnabled: enabled };
        actions.set(key, action);
        // Fence every read begun before this write; do not wait for a catalog.
        versions.accounts = ticket();
        changed();
        try {
            const response = await fetchImpl(`/api/claudexor/credential-profiles/${encodeURIComponent(harness)}/${encodeURIComponent(profile)}`,
                { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }) });
            const receipt = await response.json().catch(() => null);
            if (!response.ok) throw new Error(receipt?.error || `HTTP ${response.status}`);
            if (receipt?.ok !== true || receipt.harness !== harness || receipt.profile_id !== profile || receipt.enabled !== enabled) {
                throw new Error('The account change was not confirmed; checking its saved state');
            }
            if (disposed) return receipt;
            const held = getSnapshot();
            if (held?.profiles?.profiles) commit({ ...held, profiles: { ...held.profiles,
                // Pool routing and marks were observed before this mutation.
                accountPools: [], routing_evidence: undefined,
                profiles: held.profiles.profiles.map(wrapper => wrapper.profile?.harness_id === harness
                    && wrapper.profile?.profile_id === profile
                    ? { ...wrapper, profile: { ...wrapper.profile, enabled } } : wrapper) } });
            return receipt;
        } catch (error) {
            action.error = String(error?.message || error);
            return null;
        } finally {
            versions.accounts = ticket();
            action.busy = false;
            if (!disposed) { changed(); void observe({ afterCurrent: true }); }
        }
    }
    return { ticket, merge, observe, setEnabled,
        newer: (name, stamp) => versions[name] > stamp,
        fenceQuota() { versions.quota = ticket(); },
        action: (harness, profile) => actions.get(keyFor(harness, profile)) || {},
        dispose() { disposed = true; },
    };
}
