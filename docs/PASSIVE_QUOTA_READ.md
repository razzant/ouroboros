# Passive quota and account roster

`GET /api/claudexor/status?view=quota` is an opt-in projection for quota
consumers. The default status response and `?include=models` are unchanged.
The exact `view=quota` selector takes precedence over `include=models`.
Other `view` values retain the existing full-status behavior.

The implementation is `ouroboros/gateway/claudexor_passive.py`, selected by
`api_claudexor_status` before the full status builder is called. The
`ClaudexorQuotaResponse` and `ClaudexorPassiveReadError` TypedDicts live in
`claudexor_contracts.py`, re-exported by `contracts.py`; their JSDoc twins are in
`web/modules/api_types.js`. It discovers only the owned daemon descriptor,
then concurrently issues exactly two reads:

- `GET /v2/credential-profiles` for the necessary account roster.
- `GET /v2/quota?view=constraint_freshness` for snapshots and absences from
  one evidence envelope, opting in to the engine's per-constraint freshness.

There is no `status_dict()`, handshake, capability/operations/model catalog,
harness-manifest or doctor request, provider refresh, runtime preparation,
daemon startup, settings write, or account mutation. The existing gateway
enforces loopback discovery and sends its bearer and protocol-major headers;
the bearer remains server-side. This view makes no runtime-health or engine
version claim and does not negotiate execution features. The quota selector is
sent without a capability read. Some older engines ignore it and return legacy
metadata; released 3.25 engines reject it with HTTP 400. Only that explicit
HTTP status permits one plain `GET /v2/quota` fallback. Another refusal fails
the facet; transport failures, 5xx and malformed success never cause a second
request. Both responses use the same metadata checks. The fallback is passive,
not a provider refresh. Other quota readers never send this selector: full
Accounts status keeps its catalog-negotiated `view=resources` read or plain GET.

## Response contract

```json
{
  "view": "quota",
  "profiles": {"profiles": [], "harnessAccounts": [], "accountPools": []},
  "quota": [],
  "quota_absences": [],
  "unified_accounts": true,
  "reads": {"catalog": "not_read", "accounts": "ok", "quota": "ok"},
  "read_errors": {},
  "timings_ms": {"discovery": 1, "accounts": 2, "quota": 1, "total": 3}
}
```

Read failures, including missing discovery, return HTTP 200 with independent
facet verdicts. Do not interpret HTTP success or an empty collection alone as
evidence that an account or quota is absent.

| Field | Meaning |
| --- | --- |
| `view` | Always `"quota"`. Check this marker: an older core can ignore the query and return full status. |
| `profiles` | The complete, unfiltered credential-profile response on success; `{}` otherwise. Named/native identities, authentication facts, enablement and extensions are preserved. |
| `quota` | The quota response's `snapshots` list, or `[]` on failure. Snapshot fields, including reset credits, are unchanged. When the engine honored the selector, every `constraints[]` entry carries its own `freshness` (`fresh`, `stale` or `unknown`); otherwise none does. |
| `quota_absences` | The same response's `absences` list. Legacy omission means `[]`; a malformed present list fails the quota facet. |
| `unified_accounts` | True only if the successful roster carries the additive `accountPools` list, including an empty list. This envelope member shipped with the unified account model; no separate discovery is needed. False when absent, unread or failed, so inspect `reads.accounts` for roster authority. |
| `reads.catalog` | Always `"not_read"`; this view contains no `harnesses` catalog or manifest filter. |
| `reads.accounts`, `reads.quota` | `"ok"` = successfully read and structurally usable, including empty; `"failed"` = attempted without a usable response; `"not_read"` = discovery/client creation prevented the read. |
| `read_errors` | Only failed phases, keyed by `discovery`, `accounts` or `quota`; values contain a safe `code` and, when received, an HTTP `status_code` in 400–599. |
| `timings_ms` | Nonnegative integer elapsed milliseconds for `discovery` (including client creation), attempted `accounts`/`quota`, and `total`. Concurrent phase times are not additive. Values saturate at 2,147,483,647. |

Both roster lists (`profiles`, `harnessAccounts`) must be present and contain
objects. Named rows must have an identifiable `profile` with nonempty string
`harness_id`/`profile_id`, and boolean `enabled` if present; native rows need
`harness_id`. Optional `accountPools` must be an object list. Malformed
membership is a failed read, never an authoritative empty roster. The quota
envelope requires an object-list `snapshots` and accepts an optional
object-list `absences`. Per-constraint `freshness` is all or nothing: absent
from every constraint is the legacy envelope; present on any constraint
requires every snapshot's `constraints` to be a list of objects that each
carry `fresh`, `stale` or `unknown`. Anything else fails the quota facet
with `malformed_response`; partial freshness is never passed through.

Safe error codes are `daemon_not_discovered`, `daemon_descriptor_unreadable`,
`daemon_descriptor_incomplete`, `daemon_token_unreadable`,
`daemon_endpoint_not_loopback`, `daemon_unreachable`, `malformed_response`,
`protocol_incompatible`, `daemon_recovery_only`, `observation_read_timeout`,
`read_refused` (other typed refusals), and `read_failed` (unexpected exceptions).
No exception messages, paths, bearer tokens, arbitrary upstream codes, or
response bodies enter these diagnostics. One INFO record named
`claudexor_passive_quota_read` records only the verdicts, safe errors and timings.

## Consumer behavior and limits

Use `reads.accounts == "ok"` before applying roster additions/removals or
declaring it empty. On failed/unread roster, retain any prior roster as prior
evidence; do not interpret `{}` as removal. Pass `unified_accounts` alongside
the unchanged profile/quota fields when normalizing unified null aliases.
The flag does not make an unread roster authoritative. Account enablement is
the engine's roster fact; no harness-catalog enablement/status is synthesized.

Display each constraint by its own `freshness` when present. Without it, only
the conservative snapshot `freshness` applies; its absence is not evidence of
independent freshness. Freshness does not imply a refill, remaining quota,
authorization or provider acceptance.

Quota and roster reads succeed or fail independently. They are concurrent
responses, not an atomic combined engine snapshot. Snapshot freshness and
timestamps retain their existing meanings; gateway read success does not
make old quota data fresh. This route still waits for its two necessary reads
and uses the existing transport timeouts. It removes unrelated diagnostics
from the dependency path, not engine-side latency or quota-staleness defects.
The engine's roster producer (`buildPollResponse` in
`packages/cli/src/accounts-services.ts`) always supplies `accountPools` (even
empty) and the legacy `harnessAccounts: []`. This is the producer evidence for
the unified marker. That producer still calls
`readHarnesses`: its config-versioned, infinite-TTL cache avoids repeated
doctor work, but a cold/config-changed cache can run `statusAllForAccounts`.
The new gateway does not claim to remove that internal roster latency. It
must not synthesize a daemon-down warning from the deliberately absent
`daemon` field (and with it the full status's `last_exit` and `memory`
observations); the facet outcomes are the observation it supplies.

`tests/test_claudexor_quota_passive.py` exercises the real HTTP handler and
gateway transport against synthetic engine responses. Its catalog barrier
holds a full-status consumer pending while the passive consumer returns,
before the catalog is released. Other tests cover concurrent required reads,
empty/unreadable rosters, partial failures, unified identity evidence, safe
diagnostics, no lifecycle work, unchanged default dispatch, the engine's
recorded opt-in and legacy quota responses, and incomplete constraint
freshness.
