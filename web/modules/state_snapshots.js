import { readState } from './api_client.js';

/**
 * The page's state reads: one request/apply clock for every /api/state consumer
 * (app.js navigation, each chat's header, Evolution) and the reader they share.
 *
 * Responses may finish in either order; once generation N applies, an older
 * generation can no longer mutate any projection. requestedAt stays tied to
 * request start and is the barrier for the CARD scan (`lastLiveObservedAt`)
 * only — activity hydration is a plain projection of the census and has no
 * barrier.
 *
 * `gate(force)` is the page-wide single-flight admission for the readers: it
 * resolves to a request when the caller may read now. A periodic tick that
 * lands while a read is in flight is never queued: it resolves to null once
 * that read settles, so a caller that only needs some fresh read to have
 * landed (the boot prefetch before the socket opens) may await it. A forced
 * caller that lands mid-flight is coalesced with every other forced caller
 * into ONE follow-up read that starts when the in-flight read settles — the
 * first forced caller receives that request, the others resolve to null once
 * the follow-up has applied or failed. Gated requests settle via `apply`/`fail` even if a callback throws.
 * Only `fail` fans out `onUnavailable`; socket changes and confirmed mutations `retire()` older bodies.
 *
 * A gated request carries the `signal` of its read. `retire()` detaches the
 * read in flight: it aborts that signal, releases everyone waiting on the read
 * without a failure, and lets the coalesced forced follow-up start at once.
 * Nothing the detached read does later reaches the gate or a newer read. A read
 * that never answers ends as a failed read once its lifetime runs out
 * (`readState`), so no one stuck request holds the page's state.
 *
 * Each request names its `path`: once a full `task_bindings` map arrived with its revision, the
 * next read asks `bindings_since=<revision>`, and a body answering `task_bindings_unchanged` is
 * filled with that held map for that exact revision only; any other answer drops the held map,
 * so the read after it asks for the whole map again.
 */
export function createStateSnapshotSequencer(onApply, now = () => Date.now(), onUnavailable = () => {}) {
    let requestedGeneration = 0;
    let appliedGeneration = 0;
    // Newest applied body until a failed read or retire() clears it (late-mount seed).
    let latest = null;
    let held = null;  // {revision, bindings}: the last full task_bindings map this page applied
    let inflight = null;
    let cancelInflight = null;  // aborts the signal of the read in flight
    let settled = null;
    let followUp = null;
    const deferred = () => { let resolve; const promise = new Promise((r) => { resolve = r; }); return { promise, resolve }; };
    const begin = () => ({ generation: ++requestedGeneration, requestedAt: now(),
        path: held ? `/api/state?bindings_since=${encodeURIComponent(held.revision)}` : '/api/state' });
    const withBindings = (data) => {
        if (!data || typeof data !== 'object') return data;
        const revision = typeof data.task_bindings_revision === 'string' ? data.task_bindings_revision : '';
        if (data.task_bindings_unchanged === true && held && revision === held.revision) {
            return { ...data, task_bindings: held.bindings };
        }
        held = revision && data.task_bindings && typeof data.task_bindings === 'object'
            ? { revision, bindings: data.task_bindings } : null;
        return data;
    };
    const open = () => {
        const reading = new AbortController();
        settled = deferred();
        cancelInflight = () => reading.abort();
        inflight = { ...begin(), signal: reading.signal };
        return inflight;
    };
    const settle = (request) => {
        if (!inflight || request !== inflight) return;
        const done = settled;
        const next = followUp;
        inflight = settled = followUp = cancelInflight = null;
        if (next) next.resolve({ request: open(), done: settled.promise });
        done.resolve();
    };
    return {
        begin,
        retire() {
            appliedGeneration = begin().generation; latest = null; held = null;
            const cancel = cancelInflight;
            settle(inflight);
            cancel?.();
        },
        gate(force = false) {
            if (!inflight) return Promise.resolve(open());
            if (!force) return settled.promise.then(() => null);
            if (!followUp) { followUp = deferred(); return followUp.promise.then((f) => f.request); }
            return followUp.promise.then((f) => f.done).then(() => null);
        },
        apply(request, data) {
            try {
                const generation = Number(request?.generation) || 0;
                if (!generation || generation <= appliedGeneration) return false;
                appliedGeneration = generation;
                latest = data = withBindings(data);
                onApply(data, request.requestedAt, generation);
                return true;
            } finally { settle(request); }
        },
        isCurrent(request) {
            return (Number(request?.generation) || 0) > appliedGeneration;
        },
        fail(request) {
            try {
                const generation = Number(request?.generation) || 0;
                if (!generation || generation <= appliedGeneration) return false;
                appliedGeneration = generation;
                latest = null;
                onUnavailable(request.requestedAt, generation);
                return true;
            } finally { settle(request); }
        },
        latest: () => latest,
    };
}

/**
 * Answer one admitted request: read the state within its lifetime, then apply
 * the body or fail the read. A non-OK answer, a body that does not parse and an
 * expired or unreachable read fail it; a retired request's failure is inert, and
 * a consumer that throws while painting is swallowed as before. Readers keep
 * their own `gate()` call, so the request (and its stack) stays theirs.
 */
export async function readStateSnapshot(snapshots, request) {
    try {
        const { response, state } = await readState(request.path || '/api/state', { signal: request.signal });
        if (response.ok && state !== undefined) {
            snapshots.apply(request, state);
            return;
        }
    } catch { /* no answer: fail the read below */ }
    snapshots.fail?.(request);
}
