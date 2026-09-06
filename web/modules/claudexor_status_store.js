// ONE client-side store over `GET /api/claudexor/status` (phase 2 seam).
//
// Three surfaces used to read this endpoint independently — the Harness
// Accounts panel polled it every 5s, the review-lanes panel fetched it once per
// settings load, Subagents fetched it once more — so the app held three
// copies of the same truth, three failure handlings, and three staleness
// clocks. Worse, each of them mapped "we could not ask" onto the SAME empty
// answer the daemon gives when it is simply not running, so a stopped daemon
// decorated every saved reviewer row with "(not in discovery)" and explained
// nothing (owner report, 2026-08-08).
//
// PROVENANCE IS PER FACET, never one global verdict. `/api/claudexor/status`
// fans out to independent daemon reads — the harness CATALOG, the credential
// ACCOUNTS, the QUOTA windows — and one of them can fail while its siblings
// land. Collapsing them into a single "the service is down" would mislabel
// rows just as surely as the bug this seam fixes, so every consumer asks about
// the facet it actually renders:
//   * `ok`        — this facet was read; its collection is AUTHORITATIVE
//                   (empty means empty).
//   * `not_read`  — never asked. The owned daemon starts lazily, so an idle
//                   machine answers with empty lists nobody may read as "no
//                   account connected" (BIBLE P1: a gap is not a zero).
//   * `failed`    — asked, and the answer did not arrive.
//   * `transport` — the REQUEST itself did not complete (fetch threw, non-2xx).
//                   A client-side fact about this view, never part of the wire
//                   payload; it outranks whatever the last payload said.
//   * `unread`    — this client has not read yet (store-side only; the pure
//                   `facetReadState` never answers it).
//   * `indeterminate` — the answer as a WHOLE did not complete, and which
//                   facets landed is unknowable from it. The coarse verdict of
//                   the legacy wire; never a per-facet claim (see below).
// Only `ok` licenses a row-level "(not in discovery)" claim.
//
// Wire contract: the backend stamps `payload.reads = {catalog, accounts,
// quota}`. A payload WITHOUT that block is read legacy-style from the daemon
// state, so this store works unchanged on both wires and gains precision the
// day the field lands. Three rules keep the legacy path honest:
//   * A `reads` block that is PRESENT but not a valid stamp is a protocol
//     failure, never an excuse to fall back to the global verdict — a partial
//     `{catalog: ok}` used to make all three facets `ok`, which is the same
//     collapse in the other direction.
//   * `unreachable` is NOT `stopped`. The endpoint turns a refusal in ANY of
//     its fanned-out reads into one global `daemon.state = unreachable`, so
//     that state means «the status could not be read», never «the daemon is
//     not running».
//   * …and it is NOT a per-facet verdict either. Probed against the live
//     producer (`gateway/claudexor_accounts.py::_status_payload`): the catalog
//     and the accounts landed, only the quota probe refused, and the payload
//     carried BOTH successes beside `daemon.state = unreachable` and the
//     quota's error. Turning that one global refusal into three `failed`
//     facets made the accounts panel announce a failed ACCOUNT read WITH THE
//     QUOTA ERROR, and the reviewer rows a failed AGENT read — two reads that
//     had in fact succeeded. So a legacy global refusal answers ONE coarse
//     `indeterminate` for every facet, and its sentence names no facet and no
//     facet-specific error. PER-FACET verdicts are legitimate only when a
//     valid `reads` stamp actually says which read failed.
//
// Polling is a held resource, not a background fact: the store ticks only
// while it has subscribers, the page is visible, and either some subscriber
// says its surface is on screen or a caller holds it open (a live login job).
// Everything acquired — the timer and the visibilitychange listener — is
// released by `dispose()` (DEVELOPMENT.md «UI resources carry a disposer»).
//
// Pure helpers up top are node-tested without a DOM.

import { apiFetch } from './api_client.js';
import { harnessPresentation } from './harness_presentation.js';

export const STATUS_ENDPOINT = '/api/claudexor/status';
export const WAKE_ENDPOINT = '/api/claudexor/wake';
export const DEFAULT_POLL_MS = 5000;

export const READ_OK = 'ok';
export const READ_NOT_READ = 'not_read';
export const READ_FAILED = 'failed';
export const READ_TRANSPORT = 'transport';
export const READ_UNREAD = 'unread';
export const READ_INDETERMINATE = 'indeterminate';

// The independent facets of one status payload. The login-capability manifest
// read is deliberately NOT one: its failure is absorbed (fail-open) rather
// than reported, so it has no verdict to carry.
export const FACET_CATALOG = 'catalog';
export const FACET_ACCOUNTS = 'accounts';
export const FACET_QUOTA = 'quota';
export const STATUS_FACETS = [FACET_CATALOG, FACET_ACCOUNTS, FACET_QUOTA];

// ---------------------------------------------------------------------------
// Pure helpers.
// ---------------------------------------------------------------------------

// The states a PAYLOAD may stamp. `transport` and `unread` are client-side
// facts and are never accepted off the wire.
const WIRE_READ_STATES = new Set([READ_OK, READ_NOT_READ, READ_FAILED]);

// The daemon states that GENUINELY mean the daemon is not running — the only
// ones the «not running» sentence may be printed for, and the only non-running
// states that say anything per facet (nothing was asked, so no facet was read).
// Everything else that is not `running` (notably `unreachable`, which the
// endpoint also emits when a RUNNING daemon refused ONE of several fanned-out
// reads, plus any state spelling this client does not know) is the coarse
// «this answer did not complete» — see the header.
export const DAEMON_STATES_STOPPED = ['stale', 'not_provisioned', 'foreign_daemon'];
const STOPPED = new Set(DAEMON_STATES_STOPPED);

export function facetReadState(payload, facet, { transportError = '' } = {}) {
    // The ONE place that decides what an empty collection MEANS, for ONE facet.
    // A request that never arrived outranks whatever the last payload said: the
    // held snapshot describes a past read, never the current one.
    if (transportError) return READ_TRANSPORT;
    const body = payload && typeof payload === 'object' && !Array.isArray(payload) ? payload : null;
    if (body && 'reads' in body) {
        // The stamp EXISTS, so it — and only it — answers. A malformed or
        // partial block fails CLOSED: a 2xx is not a promise of meaning, and
        // guessing from the global verdict is exactly the collapse this store
        // was extracted to end.
        const reads = body.reads;
        const stamped = reads && typeof reads === 'object' && !Array.isArray(reads)
            ? String(reads[facet] || '') : '';
        return WIRE_READ_STATES.has(stamped) ? stamped : READ_FAILED;
    }
    // LEGACY payload (a backend without the `reads` block, or a fixture that
    // predates it): the daemon state carries the same fact at coarser
    // resolution — and COARSE is the whole point. A stopped daemon was asked
    // nothing, so every facet is honestly `not_read`; a global refusal says
    // only that the answer did not complete, so every facet is `indeterminate`
    // and NO facet may be accused of failing (the producer probe: two of the
    // three reads had actually succeeded and their data was in the payload).
    const state = String(body?.daemon?.state || '');
    if (state === 'running') return READ_OK;
    return STOPPED.has(state) ? READ_NOT_READ : READ_INDETERMINATE;
}

export function readsFor(payload, { transportError = '' } = {}) {
    const out = {};
    for (const facet of STATUS_FACETS) out[facet] = facetReadState(payload, facet, { transportError });
    return out;
}

export function facetKnown(readState) {
    // The single licence for a row-level "(not in discovery)" claim.
    return readState === READ_OK;
}

/** Current blocking evidence, independent of subject and model selection. */
export function quotaConstraintFact(constraint, nowMs = Date.now()) {
    const cooldown = String(constraint?.cooldown_until || '').trim();
    const cooldownAt = Date.parse(cooldown);
    // Preserve the explicit cooldown's conservative unknown-clock contract.
    const cooling = Boolean(cooldown)
        && (!Number.isFinite(cooldownAt) || cooldownAt > nowMs);
    const used = constraint?.used_ratio;
    const full = typeof used === 'number' && Number.isFinite(used) && used >= 1;
    const reset = String(constraint?.resets_at || '').trim();
    const resetAt = Date.parse(reset);
    const exhausted = full && Number.isFinite(resetAt) && resetAt > nowMs;
    return {
        exhausted: cooling || exhausted,
        resetsAt: cooling ? cooldown : (exhausted ? reset : ''),
        // A non-blocking incomplete reading never certifies available quota.
        unknown: full && !cooling && !exhausted,
    };
}

export function shouldPollStatus({
    hasSubscribers = false, hidden = false, surfaceVisible = false, held = false,
} = {}) {
    // The whole polling gate, pure. Each status tick fans out to four
    // CLI-probing daemon round-trips, so a background timer nobody is looking
    // at is pure waste: it needs a live subscriber, a visible page, and either
    // a surface actually on screen or a caller holding it open (a live login
    // job). An explicit `refresh()` is a request, not a poll, and is never
    // gated by this — that is what makes the first paint immediate.
    if (!hasSubscribers) return false;
    if (hidden) return false;
    return Boolean(held || surfaceVisible);
}

// What each facet is CALLED in a sentence to the owner. "agent", never "coding
// agent" (D-10): the same subscriptions run presentations, research and any
// other task, so the narrower word describes only one of their uses. Exported
// because the Agents tab's single banner speaks for SEVERAL facets at once and
// has to name them in the store's own words rather than invent a second set.
export const FACET_SUBJECT = {
    [FACET_CATALOG]: 'agents',
    [FACET_ACCOUNTS]: 'agent accounts',
    [FACET_QUOTA]: 'subscription limits',
};

// THE family display name, for every surface that shows one. It lives with the
// store because the store owns the payload this reads, and because two
// authorities is how a surface ends up printing a raw harness id: the settings
// tab preferred the engine's `display_name` while the onboarding wizard kept a
// private map of three and fell through to the id, so a renamed or fourth
// family would have reached the owner spelled `claude`.
// A daemon-provided display name is licensed only by an explicitly proven
// catalog read. The fail-closed default matters because the store deliberately
// retains its last snapshot across a failed request; an omitted provenance
// argument must never turn that retained label back into a fresh daemon fact.
export function familyLabel(harnessId, payload, { catalogKnown = false } = {}) {
    const id = String(harnessId || '');
    if (catalogKnown) {
        for (const harness of payload?.harnesses || []) {
            if (String(harness?.id || '') === id) {
                return harnessPresentation(id, { label: harness.display_name }).label;
            }
        }
    }
    return harnessPresentation(id).label;
}

export function facetGapClause(reads, facets = []) {
    // ONE clause naming the OTHER facets a surface renders that were not read,
    // so a surface can explain its primary gap without silently dropping the
    // second one. Every surface projects more than one facet — the accounts
    // panel renders rows AND quota, the reviewer rows render the catalog AND
    // account pins — and a banner that consults a single facet leaves a stale
    // value on screen dressed as a fresh one. '' when they were all read.
    //
    // Coalescing is by SUBJECT, never by enum. The callers used to drop a
    // secondary facet whose state merely EQUALLED the primary's, so with all
    // three facets failed the accounts panel said «your agent accounts could
    // not be read» and silently omitted agent discovery and subscription
    // limits, whose values stayed on screen looking fresh. Equal state is not
    // equal subject: one sentence may cover several subjects, but every
    // rendered facet that is not ok has to be named in it.
    //
    // `indeterminate` is the exception, and for the same reason: it is not a
    // verdict ABOUT a facet, so naming subjects under it would invent the
    // per-facet accusation the coarse state exists to avoid. Its own global
    // sentence covers the whole answer.
    const gaps = (facets || []).filter((name) => {
        const state = reads ? reads[name] : '';
        return Boolean(state) && state !== READ_OK && state !== READ_UNREAD
            && state !== READ_INDETERMINATE;
    });
    if (!gaps.length) return '';
    const subjects = [...new Set(gaps.map((name) => FACET_SUBJECT[name] || name))];
    const list = subjects.length > 1
        ? `${subjects.slice(0, -1).join(', ')} and ${subjects[subjects.length - 1]}`
        : subjects[0];
    // Every subject is a plural noun, and «were not read» is true of all three
    // gap states (never asked, asked and refused, request died) — «could not be
    // read» would overclaim an attempt for a daemon nobody asked.
    return `${list.charAt(0).toUpperCase()}${list.slice(1)} were not read, so anything `
        + 'shown for them is last known.';
}

export function statusUnavailableNote(readState, {
    error = '', facet = FACET_ACCOUNTS, subject = '',
} = {}) {
    // ONE sentence per read state, shared by every consumer, so the app cannot
    // explain the same gap three different ways. `null` = nothing to say.
    //
    // `action` is a SLOT, deliberately null here: this branch of the repo has
    // no owner action that can change the answer. The daemon is lazy by design,
    // and an owner-initiated "wake it and read again" is the natural occupant —
    // when such an endpoint exists, filling this field is the whole change, and
    // a surface that already renders `note.action` needs no rewrite. Shape:
    // `{ kind, label, run }` — `kind` names the action, `label` is the button
    // text, `run()` performs it and resolves once the store has re-read.
    // `subject` overrides the per-facet noun for a caller that speaks for MORE
    // than one facet — the Agents tab's single banner, when every facet failed
    // the same way and naming just one of them would under-report the gap.
    subject = subject || FACET_SUBJECT[facet] || FACET_SUBJECT[FACET_ACCOUNTS];
    if (readState === READ_TRANSPORT) {
        return {
            tone: 'error', action: null,
            text: `Could not read your ${subject}${error ? ` (${error})` : ''}, so nothing `
                + 'could be listed here. Your saved choices are unchanged — retry when the '
                + 'connection is back.',
        };
    }
    if (readState === READ_NOT_READ) {
        // "NOT ASKED" is what this state actually establishes, and it is the
        // only cause this sentence may name. Saying "the daemon is not running"
        // asserted a diagnosis the read state does not carry: a runtime that
        // needs repair, a foreign daemon on the stale port and an ownership
        // problem all land here too, and once the backend stamps `reads`
        // per facet a RUNNING daemon can leave one facet unasked. The banner
        // above (and only it) explains WHY nobody asked.
        return {
            tone: 'muted', action: null,
            text: `Ouroboros’s agent daemon was not asked, so your ${subject} were never `
                + 'checked. Nothing below is missing or wrong — your saved choices are '
                + 'unchanged, and the daemon is asked again on the next login or '
                + 'delegated run.',
        };
    }
    if (readState === READ_FAILED) {
        // The read did not land. That covers a RUNNING daemon refusing this one
        // read, an `unreachable` answer (which the endpoint also emits when one
        // fanned-out read of several refused), and a stamp this client cannot
        // trust. In every one of them telling the owner the daemon is not
        // running would be a second lie — so the sentence stops at what is
        // known: the status could not be read. Siblings stay authoritative.
        return {
            tone: 'warn', action: null,
            text: `Your ${subject} could not be read${error ? ` (${error})` : ''}. `
                + 'Nothing below is missing or wrong — your saved choices are unchanged.',
        };
    }
    if (readState === READ_INDETERMINATE) {
        // ONE global sentence, deliberately subject-free: the answer did not
        // complete AS A WHOLE, and it does not say which of its reads landed.
        // Naming `subject` here (or hanging the daemon's global error off it)
        // is exactly the misattribution the coarse state prevents — the probe
        // that found it had the accounts panel blaming a failed ACCOUNT read
        // for the QUOTA probe's error while the accounts were right there in
        // the payload. `error` is the daemon's own global last_error, which is
        // the only explanation a legacy payload carries at all.
        return {
            tone: 'warn', action: null,
            text: `The agent service did not finish answering${error ? ` (${error})` : ''}, `
                + 'so some of what is shown may be out of date — which parts is not known. '
                + 'Your saved choices are unchanged.',
        };
    }
    if (readState === READ_UNREAD) {
        return { tone: 'muted', action: null, text: `Reading your ${subject}…` };
    }
    return null;
}

const isObject = (v) => Boolean(v) && typeof v === 'object' && !Array.isArray(v);

export function unifiedAccounts(payload) {
    // Whether the engine behind this payload serves the UNIFIED account model
    // (every account a named registry row; routing facts in `accountPools`).
    // The fact is stamped server-side from the engine's own /v2/operations
    // catalog (`get:account-pools` present — sprint plan §L.2); anything else —
    // old engine, unreadable catalog, an older backend without the field —
    // reads false, which is the fail-closed legacy rendering.
    return payload?.unified_accounts === true;
}

export function statusPayloadValid(payload) {
    // The MINIMUM schema a status answer must satisfy before any facet is
    // derived from it. A 2xx is a transport fact, not a semantic one: a 200
    // carrying non-JSON (or an unrelated body) parsed to `{}` and then sailed
    // through every facet derivation as if the daemon had answered.
    //
    // The bar is the producer's UNCONDITIONAL fields, with their types —
    // `_status_payload` sets daemon/harnesses/profiles/quota before it reaches
    // the daemon at all, and these four are precisely what this store projects
    // (rows, windows, routes). Checking `daemon` alone was not depth: a bare
    // `{daemon:{state:"running"}}` passed and yielded three authoritative
    // facets over collections the body never contained. `config_dir` and
    // `subagent_last_delegation` are unconditional too but carry no projection
    // here, so requiring them would add brittleness, not truth.
    if (!isObject(payload)) return false;
    return isObject(payload.daemon)
        && Array.isArray(payload.harnesses)
        && isObject(payload.profiles)
        && Array.isArray(payload.quota);
}

export function accountRows(payload) {
    // Consume the REAL ControlCredentialProfilesResponse shape (Claudexor
    // packages/schema/src/credential-profile.ts): `profiles` is an ARRAY of
    // wrapper objects `{profile, status, identity}` with snake_case fields, and
    // `harnessAccounts` is an ARRAY of per-harness authority rows — not the flat
    // camelCase maps an earlier draft invented. Fields are read exactly as the
    // Zod schema names them so the golden fixture and the wire agree.
    //
    // DUAL-SHAPE (unified account model, sprint plan §L): a unified engine
    // migrates every default CLI login into a named registry row, so ALL of
    // its rows arrive as profile wrappers and `harnessAccounts` is emitted
    // empty — no native pseudo-row is synthesized there. The synthesis below
    // is additionally gated on the server-stamped `unified_accounts` fact so
    // a unified engine that still emitted a compatibility row could not
    // double-render the same account. A LEGACY engine keeps today's rule
    // behavior-identical — a native pseudo-row per harness, addressed by the
    // empty profile id — plus the additive fail-open `enabled` projection
    // every row now carries.
    //
    // Lives with the store because it is a projection OF the payload: the
    // accounts panel, the Subagents section and the login verify-race all read
    // it, and a second reader would be a second definition of "connected".
    const rows = [];
    const profiles = payload?.profiles?.profiles || [];
    const harnessAccounts = unifiedAccounts(payload)
        ? [] : (payload?.profiles?.harnessAccounts || []);
    for (const native of Array.isArray(harnessAccounts) ? harnessAccounts : []) {
        rows.push({
            harness: String(native?.harness_id || ''),
            profile_id: '',
            kind: 'native',
            identity: native?.identity || {},
            // Whether the default login participates in the engine's
            // credential ladder. Absent (older engines) reads as enabled —
            // the pre-toggle behavior, never a silent exclusion claim.
            enabled: native?.native_credentials_enabled !== false,
            // Both engine schema versions declare this row
            // additionalProperties:false with NO status field, so presence
            // projects as local-store evidence — detected, liveness unproven.
            status: {
                verification: native?.native_login_detected ? 'passed' : '',
                verification_source: 'local_store',
            },
        });
    }
    for (const wrapper of Array.isArray(profiles) ? profiles : []) {
        const profile = wrapper?.profile || {};
        rows.push({
            harness: String(profile.harness_id || ''),
            profile_id: String(profile.profile_id || ''),
            display_name: String(profile.display_name || ''),
            kind: 'profile',
            identity: wrapper?.identity || {},
            // The one user-settable routing control the registry row carries.
            // Absent reads as enabled (fail-open): an older payload without
            // the field must not paint every account excluded from rotation.
            enabled: profile.enabled !== false,
            status: wrapper?.status || {},
        });
    }
    return rows.filter((row) => row.harness);
}

export function nextUpAccount(payload, harness) {
    // WHO an unpinned run of this harness would route to next — the DUAL-WIRE
    // reader (one per app, same rule as accountRows): the unified engine's
    // additive `profiles.accountPools` answers first; the legacy per-harness
    // `harnessAccounts[].next_up` answers on engines that predate it. `null`
    // means the payload carries no verdict for this harness (absence, never a
    // synthesized "none"). The union is returned AS the wire spells it —
    // {kind: profile|api_key_route|none|native|…} — and consumers must render
    // an unknown kind fail-safe rather than crash: the legacy union and the
    // pool union are both closed TODAY, but this reader outlives both.
    const id = String(harness || '');
    if (!id) return null;
    const pools = payload?.profiles?.accountPools;
    for (const pool of Array.isArray(pools) ? pools : []) {
        if (String(pool?.harness_id || '') !== id) continue;
        const verdict = pool?.next_up;
        if (verdict && typeof verdict === 'object' && !Array.isArray(verdict)) return verdict;
    }
    const legacy = payload?.profiles?.harnessAccounts;
    for (const row of Array.isArray(legacy) ? legacy : []) {
        if (String(row?.harness_id || '') !== id) continue;
        const verdict = row?.next_up;
        if (verdict && typeof verdict === 'object' && !Array.isArray(verdict)) return verdict;
    }
    return null;
}

export function accountLoginConfirmed(payload, harness, profileId = '') {
    // The account-status truth the login verify-race re-check reads: the row
    // for THIS harness+profile shows a login — vendor-verified or the daemon's
    // own local-store detection (accountRows already projects both as 'passed').
    const row = accountRows(payload).find((r) =>
        r.harness === String(harness || '')
        && String(r.profile_id || '') === String(profileId || ''));
    return String(row?.status?.verification || '') === 'passed';
}

// ---------------------------------------------------------------------------
// The store.
// ---------------------------------------------------------------------------

/**
 * @param {object} [options]
 * @param {Function} [options.fetchImpl]  transport (default: the gateway client)
 * @param {Function|object} [options.doc] `document`, or a getter for it
 * @param {number} [options.pollMs]       tick spacing while polling is held open
 * @returns {object} the store
 */
export function createClaudexorStatusStore({
    fetchImpl = apiFetch,
    doc = () => (typeof document === 'undefined' ? null : document),
    pollMs = DEFAULT_POLL_MS,
} = {}) {
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    const inner = {
        snapshot: null,
        error: '',
        loading: false,
        everSettled: false,
        generation: 0,
        // Sticky-upgrading: the moment ANY consumer needs per-harness model
        // discovery every later read keeps carrying it. Downgrading the shared
        // snapshot would silently empty the model selects of whichever surface
        // did not happen to trigger the last read.
        includeModels: false,
        snapshotHasModels: false,
        inFlight: null,
        inFlightModels: false,
        // The owner-initiated wake (POST /api/claudexor/wake) in flight. A
        // WRITE the store owns for the same reason it owns the reads: its
        // answer is a fresh status payload, and a second committer of the
        // shared snapshot would be a second writer racing the first.
        wakeInFlight: null,
        queued: null,
        timer: 0,
        disposed: false,
        listeners: new Set(),
        holds: new Set(),
        visibilityBound: null,
    };

    function facet(name) {
        // The store adds ONE dimension the wire cannot carry: this client has
        // not read yet. Everything else is the pure, payload-level answer.
        if (!inner.everSettled) return READ_UNREAD;
        return facetReadState(inner.snapshot, name, { transportError: inner.error });
    }

    function reads() {
        const out = {};
        for (const name of STATUS_FACETS) out[name] = facet(name);
        return out;
    }

    function view() {
        return {
            snapshot: inner.snapshot,
            error: inner.error,
            loading: inner.loading,
            everSettled: inner.everSettled,
            generation: inner.generation,
            reads: reads(),
        };
    }

    function notify() {
        const payload = view();
        for (const entry of [...inner.listeners]) {
            try {
                entry.listener(payload);
            } catch (err) { /* one broken consumer must not stop the others */ }
        }
    }

    function shouldPoll() {
        if (inner.disposed) return false;
        const document_ = getDoc();
        let surfaceVisible = false;
        for (const entry of inner.listeners) {
            if (entry.visible && entry.visible()) { surfaceVisible = true; break; }
        }
        // A hidden page pauses EVERY reason to poll, a held login included:
        // nothing is on screen to update, and the daemon read is expensive
        // (it re-probes each agent CLI).
        return shouldPollStatus({
            hasSubscribers: inner.listeners.size > 0,
            hidden: Boolean(document_ && document_.hidden),
            surfaceVisible,
            held: inner.holds.size > 0,
        });
    }

    function clearTimer() {
        if (inner.timer) clearTimeout(inner.timer);
        inner.timer = 0;
    }

    function armPoll() {
        clearTimer();
        if (!shouldPoll()) return;
        // A setTimeout CHAIN, never an interval: the next tick is armed only
        // after the previous read settled, so a read slower than the interval
        // can never stack a second one behind it.
        inner.timer = setTimeout(() => {
            inner.timer = 0;
            if (!shouldPoll()) return;
            refresh();
        }, pollMs);
    }

    function ensureVisibilityListener() {
        if (inner.visibilityBound || inner.disposed) return;
        const document_ = getDoc();
        if (!document_ || typeof document_.addEventListener !== 'function') return;
        inner.visibilityBound = () => {
            if (inner.disposed) return;
            armPoll();
            // Resuming means catching up, not merely re-arming: the snapshot
            // held across a hidden stretch is as old as that stretch.
            if (!document_.hidden && shouldPoll()) refresh();
        };
        document_.addEventListener('visibilitychange', inner.visibilityBound);
    }

    function startRead(withModels) {
        // The READ PATH's own disposal guard, not just `refresh()`'s: the
        // model-upgrade continuation below reaches this function directly, so a
        // controller disposed while an upgrade was queued used to fan out two
        // more status reads — each of them four CLI-probing daemon round-trips
        // — for a surface with zero subscribers and polling off.
        if (inner.disposed) return Promise.resolve(inner.snapshot);
        inner.loading = true;
        inner.inFlightModels = withModels;
        // A pre-request repaint only until the first read SETTLES: with
        // anything already said, repainting each tick is churn — and against a
        // persistently failing endpoint it flickers state every tick.
        if (!inner.everSettled) notify();
        const url = withModels ? `${STATUS_ENDPOINT}?include=models` : STATUS_ENDPOINT;
        const read = (async () => {
            let payload = null;
            let error = '';
            try {
                const resp = await fetchImpl(url, { cache: 'no-store' });
                const data = await resp.json().catch(() => null);
                if (!resp || !resp.ok) {
                    error = String((data && data.error) || `HTTP ${resp ? resp.status : 'error'}`);
                } else if (statusPayloadValid(data)) {
                    payload = data;
                } else {
                    // A 200 whose body is not a status answer is a PROTOCOL
                    // failure, not an empty world: deriving facets from it would
                    // hand every consumer a confident "nothing is connected".
                    error = 'the status answer could not be understood';
                }
            } catch (err) {
                error = String(err?.message || err);
            }
            if (inner.disposed) return inner.snapshot;
            inner.loading = false;
            inner.everSettled = true;
            inner.inFlight = null;
            if (payload) {
                inner.snapshot = payload;
                inner.snapshotHasModels = withModels;
                inner.error = '';
            } else {
                // The snapshot is KEPT (a consumer may still want to show what
                // it had), but the service state now says nobody could be
                // asked — so no consumer may read absence as discovery.
                inner.error = error || 'unreachable';
            }
            inner.generation += 1;
            notify();
            armPoll();
            return inner.snapshot;
        })();
        inner.inFlight = read;
        return read;
    }

    /**
     * Read the status once. Concurrent callers SHARE one HTTP request.
     * `includeModels` upgrades the store permanently; a caller that needs
     * models while a model-less read is already in flight is served by ONE
     * queued follow-up read, shared by every other upgrading caller.
     */
    function refresh({ includeModels = false } = {}) {
        if (inner.disposed) return Promise.resolve(inner.snapshot);
        if (includeModels) inner.includeModels = true;
        // A wake OWNS the reading while it runs: it ensures the daemon and then
        // reads, so its answer is the one worth having — and a GET beside it is
        // a second writer whose order against it cannot be established from the
        // client at all. The refresh JOINS the wake instead (and the wake waits
        // out an already-running read before POSTing — see `wake`), so the two
        // writers never overlap in either order.
        if (inner.wakeInFlight) return inner.wakeInFlight.then(() => inner.snapshot);
        const wantModels = inner.includeModels;
        if (inner.inFlight) {
            if (!wantModels || inner.inFlightModels) return inner.inFlight;
            if (!inner.queued) {
                // BOTH arms re-check disposal: they run after an await, and the
                // world they were queued in may no longer exist.
                const follow = () => {
                    inner.queued = null;
                    if (inner.disposed) return inner.snapshot;
                    // The world may have moved while this upgrade waited: a wake
                    // that started meanwhile owns the view now, so join it (the
                    // re-entrant refresh keeps the models upgrade sticky).
                    if (inner.wakeInFlight) {
                        return inner.wakeInFlight.then(() => refresh({ includeModels: true }));
                    }
                    return startRead(true);
                };
                inner.queued = inner.inFlight.then(follow, follow);
            }
            return inner.queued;
        }
        return startRead(wantModels);
    }

    /**
     * OWNER action: ask the backend to start the daemon (`ensure_running`) and
     * commit the fresh status payload its answer carries. Never called by the
     * poll — the status GET stays side-effect-free, which is what keeps an
     * automatic 5-second wake impossible. Single-flighted: a cold runtime
     * install takes real time, and a second click must not start a second
     * provisioning.
     *
     * SERIALIZATION (the single-writer rule, both orders): a wake pressed
     * during a read waits that read out before POSTing, so the wake's
     * daemon-side read causally follows the poll's commit and its commit can
     * never resurrect an older snapshot; a refresh started during a wake joins
     * it (see `refresh`). The answer commits through the same snapshot /
     * error / generation fields as every read — still ONE writer of the view.
     *
     * @returns {Promise<{ok: boolean, error: string}>} the outcome. A failure
     *          deliberately does NOT touch the held snapshot or the store
     *          error: whether and how long to show a refused wake is the
     *          caller's lifecycle (the accounts panel keeps it until the
     *          daemon provably answers — `daemonAnswered`).
     */
    function wake() {
        if (inner.disposed) return Promise.resolve({ ok: false, error: 'store disposed' });
        if (inner.wakeInFlight) return inner.wakeInFlight;
        const pendingRead = inner.inFlight;
        inner.wakeInFlight = (async () => {
            let outcome;
            try {
                if (pendingRead) await pendingRead.then(() => {}, () => {});
                const resp = await fetchImpl(WAKE_ENDPOINT, { method: 'POST' });
                const data = await resp.json().catch(() => null);
                if (inner.disposed) return { ok: false, error: 'store disposed' };
                if (resp && resp.ok && statusPayloadValid(data)) {
                    inner.loading = false;
                    inner.everSettled = true;
                    inner.snapshot = data;
                    // The wake endpoint never carries model discovery
                    // (include_models=False server-side).
                    inner.snapshotHasModels = false;
                    inner.error = '';
                    inner.generation += 1;
                    outcome = { ok: true, error: '' };
                } else {
                    outcome = {
                        ok: false,
                        error: String((data && data.error)
                            || (resp && resp.ok
                                ? 'the status answer could not be understood'
                                : `HTTP ${resp ? resp.status : 'error'}`)),
                    };
                }
            } catch (err) {
                outcome = { ok: false, error: String(err?.message || err || 'request failed') };
            } finally {
                inner.wakeInFlight = null;
            }
            if (!inner.disposed) {
                // Re-arm the poll on EVERY settle, refusal included. A tick
                // that fired during the POST disarmed itself and joined the
                // wake; re-arming only on success left a visible panel with no
                // timer after a refusal — it could never notice the daemon
                // coming up on its own.
                if (outcome.ok) notify();
                armPoll();
                // A store already upgraded to model discovery must not keep
                // serving the wake's model-less snapshot: follow up once.
                if (outcome.ok && inner.includeModels) startRead(true);
            }
            return outcome;
        })();
        return inner.wakeInFlight;
    }

    /**
     * @param {Function} listener called with the store view on every settle
     * @param {{visible?: Function}} [options] `visible()` answers whether this
     *        consumer's surface is on screen; only such a subscriber can make
     *        the store poll.
     * @returns {Function} the disposer
     */
    function subscribe(listener, { visible = null } = {}) {
        if (typeof listener !== 'function' || inner.disposed) return () => {};
        const entry = { listener, visible: typeof visible === 'function' ? visible : null };
        inner.listeners.add(entry);
        ensureVisibilityListener();
        armPoll();
        let released = false;
        return () => {
            if (released) return;
            released = true;
            inner.listeners.delete(entry);
            armPoll();
        };
    }

    /**
     * Keep polling alive regardless of surface visibility — a live login job
     * needs the account rows to move under it.
     * @returns {Function} the release disposer
     */
    function holdPolling(reason = '') {
        if (inner.disposed) return () => {};
        const hold = { reason: String(reason || '') };
        inner.holds.add(hold);
        ensureVisibilityListener();
        armPoll();
        let released = false;
        return () => {
            if (released) return;
            released = true;
            inner.holds.delete(hold);
            armPoll();
        };
    }

    function dispose() {
        inner.disposed = true;
        clearTimer();
        const document_ = getDoc();
        if (inner.visibilityBound && document_?.removeEventListener) {
            document_.removeEventListener('visibilitychange', inner.visibilityBound);
        }
        inner.visibilityBound = null;
        inner.listeners.clear();
        inner.holds.clear();
        inner.inFlight = null;
        inner.wakeInFlight = null;
        inner.queued = null;
    }

    return {
        get snapshot() { return inner.snapshot; },
        get error() { return inner.error; },
        get loading() { return inner.loading; },
        get everSettled() { return inner.everSettled; },
        get generation() { return inner.generation; },
        // PER-FACET provenance — the primary question. `reads` is the whole map;
        // the three convenience getters are the questions each surface asks.
        get reads() { return reads(); },
        get catalogKnown() { return facetKnown(facet(FACET_CATALOG)); },
        get accountsKnown() { return facetKnown(facet(FACET_ACCOUNTS)); },
        get quotaKnown() { return facetKnown(facet(FACET_QUOTA)); },
        get includesModels() { return inner.snapshotHasModels; },
        get polling() { return Boolean(inner.timer); },
        get subscriberCount() { return inner.listeners.size; },
        facet,
        unavailableNote(name = FACET_ACCOUNTS, { subject = '' } = {}) {
            // The detail beside the sentence: a transport error when the request
            // itself died, otherwise — for an answer that did not land — the
            // daemon's OWN last_error. On a legacy payload that string is the
            // only explanation of an `unreachable` answer there is, and routing
            // that state to the shared sentence would otherwise have dropped it.
            // It is a GLOBAL error and rides only the global sentence; the
            // per-facet sentences never carry it, because attributing one
            // read's failure to another read's subject is the misattribution
            // this whole derivation exists to prevent.
            // Deliberately NOT attached to `not_read`: the stopped-daemon line
            // stays calm on purpose (a crashed daemon also lands in `stale`).
            //
            // `subject` widens the noun for a caller that speaks for SEVERAL
            // facets at once (the Agents tab's single banner). It rides through
            // this method rather than around it precisely so that caller keeps
            // the detail resolution above — a banner that assembled the sentence
            // itself printed "could not be read" over an `unreachable` daemon
            // and silently dropped the one explanation the owner had.
            const state = facet(name);
            const daemonError = String(inner.snapshot?.daemon?.last_error || '');
            const detail = inner.error
                || (state === READ_FAILED || state === READ_INDETERMINATE ? daemonError : '');
            return statusUnavailableNote(state, { error: detail, facet: name, subject });
        },
        refresh,
        wake,
        subscribe,
        holdPolling,
        dispose,
    };
}

// The app-wide instance every UI surface shares.
export const claudexorStatus = createClaudexorStatusStore();

// The settings shell announces a page and a sub-tab becoming active; reaching a
// panel is not a visibility CHANGE the store can observe on its own.
export const SURFACE_ACTIVATION_EVENTS = ['ouro:page-shown', 'ouro:settings-subtab-shown'];

/**
 * Bind ONE surface to the shared store: a subscription that can actually keep
 * the poll armed, plus a catch-up read when the surface becomes reachable.
 *
 * Deliberately NAME-FREE. The three consumers used to hardcode which tab they
 * lived on — and two of them supplied neither a visibility predicate nor an
 * activation hook, so their comments promised "the daemon recovering is picked
 * up without a reload" while the store never polled for them at all. The rule
 * here is structural: an element that is on screen (an inactive `.page` /
 * `.settings-panel` is display:none, which is exactly what `offsetParent`
 * reports) is a visible surface, wherever the sprint later moves the section.
 *
 * @param {object} store the shared status store
 * @param {object} options
 * @param {Function} options.listener called with the store view on every settle
 * @param {string} options.elementId id of the element that IS this surface
 * @param {boolean} [options.includeModels] the activation read needs discovery
 * @param {Function} [options.onActivate] optional owner-action override
 * @returns {Function} one disposer releasing the subscription and the listeners
 */
export function bindStatusSurface(store, {
    listener = () => {},
    elementId = '',
    includeModels = false,
    onActivate = null,
    doc = () => (typeof document === 'undefined' ? null : document),
    win = () => (typeof window === 'undefined' ? null : window),
    activationEvents = SURFACE_ACTIVATION_EVENTS,
} = {}) {
    const getDoc = typeof doc === 'function' ? doc : () => doc;
    const getWin = typeof win === 'function' ? win : () => win;
    const visible = () => {
        if (!elementId) return true;
        const el = getDoc()?.getElementById?.(elementId);
        return Boolean(el) && el.offsetParent != null;
    };
    const disposers = [store.subscribe(listener, { visible })];
    const target = getWin();
    if (target && typeof target.addEventListener === 'function') {
        // Activation is judged by the SAME predicate, so a tab that is not this
        // surface's tab costs nothing and this surface needs no name for its own.
        const activationAction = typeof onActivate === 'function'
            ? onActivate
            : () => store.refresh({ includeModels });
        const onActivated = () => { if (visible()) return activationAction(); };
        for (const name of activationEvents) {
            target.addEventListener(name, onActivated);
            disposers.push(() => target.removeEventListener(name, onActivated));
        }
    }
    return () => {
        for (const dispose of disposers.splice(0)) {
            try { dispose(); } catch (err) { /* a broken disposer must not block the rest */ }
        }
    };
}

/**
 * Refresh with a bounded wait: resolve when the read settles OR after
 * `beatMs`, whichever comes first — the refresh itself always runs to
 * completion and notifies subscribers/surfaces when it lands. For callers on
 * a user-facing critical path (the Settings Save flow) that want the settled
 * snapshot when it is cheap (warm daemon) but must not wait out a cold
 * daemon's wake-and-discover walk. The losing timer is cleared so a settled
 * refresh leaves nothing holding a node test-runner's loop open.
 */
export function boundedStatusRefresh(store, { includeModels = true, beatMs = 2000 } = {}) {
    const refresh = Promise.resolve(store.refresh({ includeModels })).catch(() => {});
    let timer = null;
    const beat = new Promise((resolve) => { timer = setTimeout(resolve, beatMs); });
    return Promise.race([refresh, beat]).finally(() => { if (timer) clearTimeout(timer); });
}

/**
 * The one sentence for "Claudexor is being made ready", phased by what is
 * actually happening: the runtime manager's status projection distinguishes
 * installing from ready, and the daemon aggregate says whether the engine is
 * serving — printing "Installing or checking" when the payload names the
 * phase made a minutes-long first install indistinguishable from a
 * sub-second probe. An absent or unread payload answers the honest generic:
 * this caller IS mid-check, it just has no phase evidence yet.
 */
export function claudexorPreparationLine(payload) {
    const daemon = payload?.daemon || {};
    const runtime = daemon.runtime || {};
    const state = String(runtime.state || '');
    if (daemon.ownership_problem) {
        // Checked before EVERY positive phase claim: ensure_running refuses a
        // foreign daemon home, so neither an install nor a start proceeds on
        // this payload however the runtime projection reads — the accounts
        // panel carries the ownership sentence itself.
        return 'Checking Claudexor…';
    }
    if (state === 'installing') {
        const version = runtime.target_version ? ` ${runtime.target_version}` : '';
        return `Installing Claudexor${version}…`;
    }
    if (state === 'ready' && String(daemon.state || '') === 'stale') {
        // POSITIVE knowledge only: 'stale' is the producer's own "installed,
        // engine idle, starts automatically" verdict. A partially failed
        // fan-out rewrites a LIVE daemon's aggregate to 'unreachable' while
        // preserving the runtime projection, so anything but 'stale' falls to
        // the generic — never a "Starting…" claim about an engine that may
        // already be serving.
        return 'Starting the Claudexor daemon…';
    }
    return 'Checking Claudexor…';
}
