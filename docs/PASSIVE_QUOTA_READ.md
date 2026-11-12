# Passive quota, resources and actor inspection

`GET /api/claudexor/status?view=quota` selects the passive projection before
`api_claudexor_status` calls the full Accounts builder. It takes precedence over
`include=models`; other status selectors keep their existing behavior.

`gateway/claudexor_passive.py` discovers the owned loopback descriptor and issues
exactly two concurrent reads, with bearer/protocol headers kept server-side:

- `GET /v2/credential-profiles` for the account roster.
- `GET /v2/quota?view=resources_freshness&requestKind=text` for one quota/resource
  envelope with per-constraint freshness and an explicit availability scope.

No handshake, catalog, doctor, status builder, provider refresh, runtime
preparation, daemon startup, settings write or account mutation is requested.
`claudexor_observations.read_quota` also serves full Accounts, delegated route
health and the primary-route observation: the readers share engine facts, not
separate resource/quota interpretations.
Full Accounts retains its existing operations read and diagnostics. If that catalog
advertises resources, an HTTP 400 or an ignored rich selector (neither resources nor scope)
uses one `GET /v2/quota?view=resources`; the complete fallback replaces the first
epoch. Missing advertised resources fails the facet, never authoritative empty.

Passive/inspection reads require no catalog discovery. An explicit HTTP 400 permits
view-only `constraint_freshness` (older engines reject model/kind keys); only its
explicit 400 permits plain `GET /v2/quota`. Transport failure, 5xx or malformed
success never repeats. Return one whole epoch, preserving each window's freshness.
Legacy facts do not prove honored model/kind filters: missing or mismatched
`availability_scope` never becomes exact availability, regardless of engine version.

## Response contract

`ClaudexorQuotaResponse` / `ClaudexorPassiveReadError` live in
`gateway/claudexor_contracts.py`, are re-exported by `contracts.py`, and have
JSDoc twins in `web/modules/api_types.js`.

| Field | Meaning |
| --- | --- |
| `view` | Always `quota`; an older core may ignore the selector and return full status. |
| `profiles` | Complete unchanged credential-profile envelope on success; `{}` otherwise. |
| `quota` | That quota response's `snapshots`, unchanged, or `[]` on failure. |
| `quota_absences` | Same response's `absences`; legacy omission means `[]`. |
| `resources` | Optional unchanged resource observations, never an admission balance. |
| `availability_scope` | Optional engine echo `{model: string\|null, requestKind: 'text'\|'image', basis: 'quota_block'}` of the exact requested binding, not a literal model-name matching policy. A sent selector alone proves nothing. Full status carries the same optional field. |
| `unified_accounts` | `reads.accounts` is `ok`: the roster is the unified model. False also covers unread/failed roster. |
| `reads.catalog` | Always `not_read`; no harness catalog or manifest filtering. |
| `reads.accounts`, `reads.quota` | `ok`: structurally usable (including empty); `failed`: attempted but unusable; `not_read`: discovery/client creation prevented it. |
| `read_errors` | Failed discovery/accounts/quota phases only; safe code and optional HTTP 400–599 status, never upstream text or credentials. |
| `timings_ms` | Nonnegative discovery, attempted facet and total milliseconds, saturated at 2,147,483,647. Concurrent times do not add. |

Read failures still return HTTP 200 with independent facet verdicts. HTTP success,
empty lists and missing fields are not proof that accounts or quota are absent.
`claudexor_passive.roster_error` decides the roster for this read and full Accounts:
object-list `profiles` (rows need `profile.harness_id` / `profile.profile_id`, any
`enabled` boolean) and object-list `accountPools` (3.6.0+); `harnessAccounts` is
unread. Its rows without `accountPools`: `account_model_unsupported`, not malformed.
An unread roster is never authoritative removal or an empty account set.

Quota requires object-list `snapshots`; present `absences` and `resources` must
also be object lists. Constraint freshness is all-or-none: if any constraint has
`freshness`, every snapshot has an object-list `constraints` and every constraint
has `fresh`, `stale` or `unknown`. Malformed membership or partial freshness fails
the facet. Resources preserve scope, usage kind, period, fee inclusion, units and
source fields without reinterpreting team resources as personal reserves.
Resources bind `target.{harness,profile_id}` (AccountTarget), unlike quota's
nullable `subject.{harness,subject_id}`.

Safe errors: `daemon_not_discovered`, `daemon_descriptor_unreadable`,
`daemon_descriptor_incomplete`, `daemon_token_unreadable`,
`daemon_endpoint_not_loopback`, `daemon_unreachable`, `malformed_response`,
`account_model_unsupported`, `protocol_incompatible`, `daemon_recovery_only`,
`observation_read_timeout`, `read_refused`, `read_failed`. One `claudexor_passive_quota_read` INFO record logs
only verdicts, safe errors and timings, not exception text, paths or bodies.

## Exact configured actor

`inspect_subagent(subagent_id, request_kind='text'|'image')` resolves the CURRENT
TASK settings snapshot, not today's edited Settings. It returns configured
route/model/account/access/enablement separately from independently dated cached
observations. Disabled configuration is not unknown live health; positive quota
availability is not permission, authentication, capability or execution success.
Direct API routes without a managed engine have unknown provider availability.
The tool works without any quota skill and never selects or starts an actor.
Resolution uses the offered catalog's reachable Main fallback, retaining disabled
saved rows; model-named effort is reported by the same owner as dispatch.

Managed routes send the literal model/kind to the same passive reader and filter
subjects locally. A strict account pin, `<harness>-default` included, matches only
its exact subject: never a healthy sibling or an unprofiled null window, which
Auto still sees. The engine owns applicability,
constraint freshness and `quota_block`; the host neither matches model names
fuzzily nor derives availability from ratios or money. Missing scope, stale or
incomplete facts stay unknown.
Rows and absences group by full subject (harness, credential route, subject id).
One subject's simultaneous rows are constraints, never alternatives: an engine
exhausted/cooldown verdict whose `resets_at` is a valid, still-future instant wins
beside stale, free or absent rows (a stale row too: the engine keeps such caps in
its admission ledger); a malformed, out-of-range (Feb 30, 24:00, +05:60) or passed
reset is no live block, read alike by both readers. A subject is
available only when every row is fresh and engine-available, it has no absence,
and one window is measured (finite `used_ratio`) for every model (no model ids or
prefixes) and the requested kind; model-scoped-only or other-kind evidence stays
unknown. Observed subjects (absence-only included) are alternatives: one available
suffices, one unknown prevents all-blocked, else the earliest reset is a recheck,
not readiness. The read roster is not the engine's enabled, ready pool.
`tests/fixtures/quota_subject_matrix` (engine-computed rows) binds the Python and
card readers. Migration: legacy engines without scope no longer
certify host-side quota exhaustion from raw ratios; the guard proceeds as unknown
to engine start admission. The engine POST still owns monetary admission. Neither
inspection nor this compatibility behavior guarantees a later start will succeed.
Available-subagent cards likewise use only exact echoed text scope and engine
verdicts. A generic full Accounts echo cannot certify a particular actor; account
and catalog facts (including a base-only `[1m]` qualifier) stay visible with
`Not checked`. Resource merges bind scope to quota: retained rows keep their scope;
unproven or mixed-scope replacement clears it. No new request is made.

## Primary-route observation

While a fallback acts, `primary_route_observation.observe_primary_route` asks
`claudexor_observations.quota_observation` about a Claudexor primary's exact
model, `requestKind=text` and pin or Auto, before its catalog check. One deadline,
the observation's shared 3 s budget less caller bounds, covers
`read_owned_gateway` and every legacy view (`read_quota(timeout_sec=)`;
`quota_state(timeout_sec=)`); no roster is read.
The catalog gets the remainder, and a spent budget reads nothing more. The row
carries the unchanged `quota_availability` verdict (`scope: quota_only`) beside
every scoped snapshot and absence row, none dropped: subject, source, `observed_at`,
freshness, fresh/total windows and the engine's own state and reset, never plan
labels, ratios, money, roster contents or absence prose. `read: failed` (safe code), `not_read`
(`observation_deadline_elapsed`) and an empty `ok` stay distinct. Auto rows cover
every engine subject of the harness, enabled or not. A stale same-subject source
keeps the aggregate unknown unless a live dated block remains. A newer `observed_at`
alone is not a changed fact; ageing into unknown is. Nothing refreshes, starts,
selects or generates; `tests/test_primary_quota_observation.py` drives the loop.

For an unchanged primary binding, checks are at least 120 s apart. A changed
fact re-emits the whole set into context, including unchanged rows. The
recorded 43-row sample was about 9.4 kB per emission. The interval bounds
cadence, not cumulative context cost. Actual emission frequency is unknown;
token price was not measured.

## Limits and checks

Accounts and quota reads are independent, not one atomic engine snapshot.
Successful transport does not refresh observations. Display each constraint's
freshness; legacy aggregate freshness does not prove independent freshness.
Preserve the last dated roster/value on read failure, not an invented zero.

The roster producer still calls `readHarnesses` through a config-versioned cache;
a cold/config-changed cache may run `statusAllForAccounts`. This core path removes
unrelated diagnostics, not that engine-internal latency. The absent `daemon`
field is not a daemon-down observation. Engines without `resources_freshness`
(including 4.2.0) refuse it with 400, so each quota read pays one extra
GET, older engines up to two; the negotiated view is deliberately not cached.

`test_claudexor_quota_passive.py` drives the real handler and synthetic gateway,
including exact selectors, no lifecycle work, concurrent facets, catalog barrier,
malformed/legacy/freshness cases, safe diagnostics and the shared roster decision.
`test_subagent_inspection.py` covers configured snapshots, strict subjects and the public registry;
route/review tests preserve dispatch and account-pin behavior.
