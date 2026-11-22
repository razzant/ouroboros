// Independent account/quota observations beside the slower catalog read.
// Dispatch order belongs to this client; provider observation dates remain unchanged.
import { apiFetch, withinRequestLifetime } from './api_client.js';

// `resources` is optional quota evidence with its own order (see merge).
const facets = {
    catalog: ['harnesses'],
    accounts: ['profiles', 'unified_accounts'],
    quota: ['quota', 'quota_absences', 'availability_scope'],
};
const object = value => value && typeof value === 'object' && !Array.isArray(value);
export function passiveStatusValid(value) {
    return value?.view === 'quota' && object(value.profiles) && Array.isArray(value.quota)
        && Array.isArray(value.quota_absences) && object(value.reads)
        && ['accounts', 'quota'].every(key => ['ok', 'failed', 'not_read'].includes(value.reads[key]));
}

/** Whether a status answer read the operations catalog (resource capabilities), independent of reads.catalog. */
export function resourceCapabilitiesRead(payload) {
    return payload?.resource_capabilities_read ?? (payload?.resource_capabilities ? 'ok' : 'not_read');
}

export function createObservationReads({ fetchImpl = apiFetch, getSnapshot, commit, changed, quotaObserved = () => {},
    lifetimeMs } = {}) {
    let sequence = 0, flight = null, follow = null, disposed = false;
    let flightAbort = null;
    // Two clocks in dispatch order, never one for the other. A fence orders merges: an answer begun before it is
    // older than what this client holds (a row write, a Refresh's own reading). `observed` is the last read that
    // answered a facet; only it outranks a lost status request (store.facet), so a fence dates nothing.
    const fences = { catalog: 0, accounts: 0, quota: 0, resources: 0 };
    const observed = { catalog: 0, accounts: 0, quota: 0 };
    // Whether the engine serves, by dispatch order: a passive answer says so by an answered
    // account or quota read, a lifecycle read (full status, wake) by `running`. Each side keeps
    // when its verdict last changed, so one disagreement asks for one lifecycle read.
    let served = null, servedSince = 0, passiveStamp = 0, lifecycle = null, lifecycleSince = 0, lifecycleStamp = 0;
    const actions = new Map();
    const keyFor = (harness, profile) => JSON.stringify([harness, profile]);
    let roster = null, projected = null, loginPolicy, policyStamp = 0, accountRead;
    // Model marks exist only in full reads; they keep their own order like the policy.
    const markFields = ['routing_evidence', 'routing_evidence_read'];
    let marks, marksStamp = 0;
    function ticket() { return ++sequence; }
    function merge(payload, stamp, { passive = false } = {}) {
        const held = getSnapshot();
        if (!passive) {
            const serves = payload.daemon?.state === 'running';
            if (serves !== lifecycle) [lifecycle, lifecycleSince] = [serves, stamp];
            lifecycleStamp = stamp;
        } else if (payload.view === 'quota') { // a lost request is no evidence about the engine
            const serves = ['accounts', 'quota'].some(key => payload.reads?.[key] === 'ok');
            if (serves !== served) [served, servedSince] = [serves, stamp];
            passiveStamp = stamp;
        }
        // Beside a held lifecycle that does not serve, a passive answer that served nothing is
        // that same fact again: a facet that read left unread keeps speaking (its banner line
        // leads). One held as answered does not: there the failure is news.
        const quiet = passive && payload.view === 'quota' && lifecycle === false && !served;
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
        // A newer passive roster or a slower discovery must not erase them; a confirmed
        // write fences marks observed before it (a row write advances marksStamp).
        if (!passive && stamp >= marksStamp && (!payload.reads || payload.reads.accounts === 'ok')
            && payload.profiles && typeof payload.profiles === 'object') {
            marksStamp = stamp;
            marks = Object.fromEntries(markFields.filter(key => Object.hasOwn(payload.profiles, key))
                .map(key => [key, payload.profiles[key]]));
        }
        const out = { profiles: {}, quota: [], harnesses: [], daemon: {},
            ...(held || {}), ...(passive ? {} : payload),
            reads: { ...(held?.reads || {}), ...(passive ? {} : payload.reads) },
            facets: { ...(held?.facets || {}), ...(passive ? {} : payload.facets) } };
        for (const [name, fields] of Object.entries(facets)) {
            const pending = name === 'accounts' && [...actions.values()].some(action => action.busy);
            const state = payload.reads?.[name];
            // The host discards an answer that crossed a confirmed account write or a newer held read:
            // `not_read` although the engine was asked (passive: discovery succeeded; full: `running`).
            // Like an older ticket it observed nothing here, so the facet keeps what it held.
            const discarded = state === 'not_read' && Boolean(held?.reads?.[name])
                && (passive ? !payload.read_errors?.discovery : payload.daemon?.state === 'running');
            const older = (passive && name === 'catalog') || stamp < fences[name] || pending || discarded;
            // Any quota answer in order, a failure and a quiet one too, supersedes the Refresh readings taken before it.
            if (name === 'quota' && !older) quotaObserved(stamp);
            if (older || (quiet && held?.reads?.[name] !== 'ok')) {
                for (const field of fields) {
                    if (held && field in held) out[field] = held[field];
                    else delete out[field];
                }
                out.reads[name] = held?.reads?.[name] || 'not_read';
                if (held?.facets?.[name]) out.facets[name] = held.facets[name];
                else delete out.facets[name];
                continue;
            }
            // Only an answered facet (any legacy wire) outranks earlier tickets: a failed or
            // unread one reports its gap and never discards a concurrent answer. A full read's
            // unread roster is the host's memory of an earlier full read, which never saw the
            // passive rosters held here: as for quota (commitStatus), the held one keeps its date.
            if (!payload.reads || state === 'ok') fences[name] = observed[name] = stamp;
            const gap = state !== 'ok' && (passive || (name === 'accounts' && payload.reads
                && Array.isArray(roster?.profiles)));
            if (passive) {
                out.reads[name] = state;
                out.facets[name] = { observed_at: state === 'ok' ? new Date().toISOString()
                    : held?.facets?.[name]?.observed_at ?? null,
                stale: state !== 'ok' && Boolean(held),
                error: payload.read_errors?.[name]?.code || payload.read_errors?.discovery?.code || null };
            } else if (gap) {
                out.facets[name] = { ...payload.facets?.[name], observed_at: held?.facets?.[name]?.observed_at ?? null, stale: true };
                for (const field of fields) {
                    if (held && field in held) out[field] = held[field];
                    else delete out[field];
                }
            }
            if (name === 'accounts') accountRead = state;
            if (gap) continue;
            for (const field of fields) {
                if (field in payload) out[field] = payload[field];
                else delete out[field];
            }
            if (name === 'accounts') roster = out.profiles;
        }
        // Resources keep their own order, decided here alone. An explicit list (empty too) observes
        // them, and so does the absence a full answer negotiated from a read operations catalog. A
        // quota answer with neither observed none (a passive view of an engine that serves them only
        // through full status's fallback; a full read whose operations catalog was unread): it neither
        // replaces nor fences them. The list the host leaves beside full rows it discarded as older
        // (`not_read` from a running engine) is owned in ticket order: a read at least as new observed it.
        const listed = Array.isArray(payload.resources);
        const owned = !payload.reads || payload.reads.quota === 'ok'
            ? listed || (!passive && resourceCapabilitiesRead(payload) === 'ok')
            : listed && !passive && payload.reads.quota === 'not_read' && payload.daemon?.state === 'running';
        if (owned && stamp >= fences.resources) {
            fences.resources = stamp;
            if (listed) out.resources = payload.resources;
            else delete out.resources;
        } else if (held && 'resources' in held) out.resources = held.resources;
        else delete out.resources;
        if (roster && marks) roster = { ...Object.fromEntries(Object.entries(roster)
            .filter(([key]) => !markFields.includes(key))), ...marks };
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
        const abort = new AbortController();
        flightAbort = abort;
        flight = (async () => {
            try {
                const { response, value } = await withinRequestLifetime(async signal => {
                    const response = await fetchImpl('/api/claudexor/status?view=quota', { cache: 'no-store', signal });
                    return { response, value: await response.json().catch(() => null) };
                }, { lifetimeMs, signal: abort.signal });
                if (!response.ok || !passiveStatusValid(value)) throw new Error(value?.error || 'Account observations unavailable');
                if (!disposed) commit(merge(value, stamp, { passive: true }));
            } catch (error) {
                if (!disposed) commit(merge({ reads: { accounts: 'failed', quota: 'failed' },
                    read_errors: { discovery: { code: error?.name === 'StateReadEnded' && error.reason === 'expired'
                        ? 'timed out' : String(error?.message || error) } } }, stamp, { passive: true }));
            } finally {
                flight = null;
                if (flightAbort === abort) flightAbort = null;
                if (!disposed) changed();
            }
            return getSnapshot();
        })();
        return flight;
    }
    const profileUrl = (harness, profile) =>
        `/api/claudexor/credential-profiles/${encodeURIComponent(harness)}/${encodeURIComponent(profile)}`;
    // Only a 4xx proves a row write was refused unapplied. The host answers an engine refusal and a
    // transport loss after sending alike with 503 (CLIENT_CONTRACT), so a 5xx, like a lost, unexplained
    // or unmatched answer, leaves the outcome unknown; the read after the write shows the truth.
    const refused = response => response.status >= 400 && response.status < 500;
    // Every confirmed row write (Enabled, Remove) fences the reads begun before it and edits
    // the held row at its confirmation; do not wait for a catalog.
    async function rowWrite(harness, profile, send, edit, settled = () => {}) {
        marksStamp = fences.accounts = ticket();
        changed();
        try {
            const receipt = await send();
            if (disposed) return receipt;
            const held = getSnapshot();
            marks = {};
            if (held?.profiles?.profiles) commit({ ...held, profiles: { ...held.profiles,
                // Pool routing and marks were observed before this mutation.
                accountPools: [], routing_evidence: undefined, routing_evidence_read: undefined,
                profiles: held.profiles.profiles.flatMap(wrapper => wrapper.profile?.harness_id === harness
                    && wrapper.profile?.profile_id === profile ? edit(wrapper) : [wrapper]) } });
            return receipt;
        } finally {
            marksStamp = fences.accounts = ticket();
            settled();
            if (!disposed) { changed(); void observe({ afterCurrent: true }); }
        }
    }
    async function setEnabled(harness, profile, enabled) {
        const key = keyFor(harness, profile);
        if (disposed || actions.get(key)?.busy) return null;
        const action = { busy: true, error: '', requestedEnabled: enabled };
        actions.set(key, action);
        try {
            return await rowWrite(harness, profile, async () => {
                const response = await fetchImpl(profileUrl(harness, profile),
                    { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }) });
                const receipt = await response.json().catch(() => null);
                if (!response.ok) {
                    action.refused = refused(response);
                    throw new Error(receipt?.error || `HTTP ${response.status}`);
                }
                if (receipt?.ok !== true || receipt.harness !== harness || receipt.profile_id !== profile || receipt.enabled !== enabled) {
                    throw new Error('The answer did not match the request');
                }
                return receipt;
            }, wrapper => [{ ...wrapper, profile: { ...wrapper.profile, enabled } }], () => { action.busy = false; });
        } catch (error) {
            action.error = String(error?.message || error);
            return null;
        }
    }
    /** The engine's DELETE under the same ordering. A failure throws with the host's reason;
     *  `refused` marks the one that proves nothing was removed. */
    function remove(harness, profile) {
        if (disposed) return Promise.reject(Object.assign(new Error('The accounts view was closed'), { refused: true }));
        return rowWrite(harness, profile, async () => {
            const response = await fetchImpl(profileUrl(harness, profile), { method: 'DELETE' });
            const receipt = await response.json().catch(() => null);
            if (!response.ok) {
                throw Object.assign(new Error(String(receipt?.error || `HTTP ${response.status}`)), { refused: refused(response) });
            }
            // ClaudexorCredentialProfileDeleteResponse names the removed profile; any other 200 is unknown.
            if (receipt?.removed !== true || receipt.profile?.harness_id !== harness || receipt.profile?.profile_id !== profile) {
                throw Object.assign(new Error('The answer did not match the request'), { refused: false });
            }
            return receipt;
        }, () => []);
    }
    return { ticket, merge, observe, setEnabled, remove,
        /** A read that answered this facet began after `stamp`. */
        newer: (name, stamp) => observed[name] > stamp,
        /** A raw roster held back until a full read supplies the login policy. */
        awaitingPolicy: () => loginPolicy === undefined && accountRead === 'ok',
        /** A passive answer newer than the held lifecycle read disagrees with it: 'passive' when the passive
         *  verdict changed after that read, 'lifecycle' when that read's own verdict just changed. One read
         *  answers each; a disagreement that outlives it is a steady state. */
        contradiction() {
            if (served === null || lifecycle === null || served === lifecycle || passiveStamp < lifecycleStamp) return '';
            return servedSince > lifecycleStamp ? 'passive' : lifecycleSince === lifecycleStamp ? 'lifecycle' : '';
        },
        /** Orders a Refresh's or receipt's own reading before the reads begun earlier and returns the ticket that
         *  dates it. One account's reading observed no facet; a provider read of every account (`whole`) did. */
        fenceQuota(resources = false, whole = false) {
            fences.quota = ticket();
            if (resources) fences.resources = fences.quota;
            if (whole) observed.quota = fences.quota;
            return fences.quota;
        },
        action: (harness, profile) => actions.get(keyFor(harness, profile)) || {},
        dispose() { disposed = true; if (flightAbort) flightAbort.abort(); flightAbort = null; },
    };
}
