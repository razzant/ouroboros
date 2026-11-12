# Model-send observability: `model-visible ⟺ logged`

The physical-send observability contract is implemented in
`ouroboros/model_send_seal.py`, wired at
`llm_attempt._candidate_before_dispatch` and swept by the session-custodied
history child `ouroboros/startup_historical_audit.py` (explicit owner-started
audit, never launched by boot);
`tests/test_model_send_seal.py` verifies it. The claim is deliberately narrow:
a reconstruction mismatch is an observability fact, not a dispatch gate.

## 1. Scope of the claim

The invariant binds exactly one object: **`model_send` — the physical
candidate payload at the last host-controlled pre-transport seam**. That seam
already exists and is singular:
`ouroboros/llm_attempt.py::_candidate_before_dispatch` — the closure that runs
after the cache-marker finalizer produced the final send copy
(`_physical_candidate`) and before the transport's `send`. A funded realtime setup
attempt has no such candidate: its pre-dispatch callback (`realtime_turn.py`) seals
the exact provider wire plan through the same `persist_physical_candidate` owner and
returns the same manifest reference, so both directions below apply to it unchanged.

- **Forward (`sent ⟹ logged`)**: every physical attempt persists a sealed
  durable record of its exact send copy BEFORE dispatch.
- **Reverse (`logged ⟹ sent`)**: every sealed `model_send` record joins
  exactly one accounting attempt (dispatched, refused, or released). The
  reverse direction is asserted only for `model_send` records, not
  for any other log plane (events, chat, progress are narrations,
  not send truth).
- **Everything else is out of the byte domain by typed exclusion, never by
  silence** (§4).

"Model-visible" means *what the provider-hosted model receives*, i.e. the
send payload (`system`/`messages`/`tools`/params). It does not mean UI
rendering, and it does not mean the response side: what the model "sees" of
its own previous answers is whatever the host replays into the NEXT send, so
response-assembly truth is covered transitively by the next round's
`model_send` record (§5.2).

## 2. Shared mechanisms

| Existing mechanism | Where | Role in the invariant |
|---|---|---|
| Canonical digest of the exact send copy (`canonical_json_v1`: sort_keys, compact separators, `ensure_ascii=False`, `allow_nan=False`, `default=str`) | `llm_attempt._attempt_request` / `_canonical_candidate_bytes` | The canonical form and its versioned basis (`candidate_measurement_kind`) |
| Pre-dispatch identity re-check: digests re-derived from the closed-over candidate and compared with the reservation's expected identity; drift refuses dispatch (`PhysicalAttemptPreparationFailed: physical candidate changed before dispatch`) | `llm_attempt._candidate_before_dispatch` | Exact raw/context comparison without rebuilding token, clock or image observations; Claudexor uploads the validated canonical bytes with their measured hash and length |
| Durable candidate manifest + redacted CAS blob, written before dispatch, with two labelled digest domains (`canonical_json_v1_pre_redaction` facts vs `observability_json_v1_post_default_redaction_cas` blob); the blob is a request view stored once — its messages and tool schemas are parts, its skeleton placement is spliced back into the exact whole on read (`shared_values`, PERSISTENCE observability row), so the CAS digest and size are the whole's | `observability.persist_physical_candidate` / `persist_call` | The sealed record carrier |
| Attempt lifecycle `reserved → dispatched → settled|unresolved` / `reserved → released`, short-lock append + sequence replay | `usage_accounting` | The join target for the reverse direction |
| Anthropic native custody projection (opaque provider-native content replaced before persistence; disclosed as `anthropic_native_custody_projected`) | `anthropic_native_custody.physical_custody_projection` | Prototype of a typed exclusion |
| Secret redaction with per-hit `RedactionRecord`s, worked out once per part of a request view and reused by the task's other records (`redaction.part_fact`) | `redaction._redact_text` + rules | Prototype of a typed exclusion |

An identity gate that compares only two **in-memory** serializations cannot
establish whether the durable record matches either copy.
`model_send_seal.verify_sealed_candidate` checks that separate question by
**reconstructing from the durable record** and byte-comparing that
reconstruction against the wire-bound serialization, emitting a **typed durable
fact** on any inequality (§3.2 — a fact, not a refusal).

### Claudexor

The pre-dispatch gate compares exact raw/context identities without repeating token, clock or image measurement. Claudexor uploads those validated canonical bytes with their measured hash and length; the defensive send copy remains, and mutation after measurement refuses dispatch. Claudexor acceptance merges the operation identity and capture facts into its existing operational request manifest without rereading or serializing the payload. Raw/projection references and prior metadata survive; the separate pre-dispatch physical manifest and send seal remain immutable.

A released Claudexor prepare failure repeats on the primary within `max_retries` (three sends, 4 s then 8 s) only with `ClaudexorModelNotDispatched`, provider `claudexor`, an empty operation ID and gateway code `daemon_unreachable`, `daemon_not_answering`, `daemon_starting` or `daemon_descriptor_changed`. The configured-route walk follows a spent bound. An open transport wait or held unknown-repeat record still prevents a new burst. Direct-API connect failures, undated credential-pool refusals and local slot refusals retain one attempt; a lost create reply remains unknown and rejoins the same invocation.

## 3. The contract

### 3.1 Sealed record (`model_send` seal, v1)

The existing physical-candidate manifest carries a
`model_send_seal` block:

- `seal_version: 1`
- `canonical_basis: "canonical_json_v1"` — serializer identity. Any change to
  the serializer is a NEW basis string; a reader never re-interprets bytes
  under a different basis.
- `pre_redaction_sha256` / `size_bytes` — digest of the canonical bytes of the
  exact wire payload (`candidate_raw_sha256`).
- `exclusions: [...]` — every applied exclusion instance: `{class, path,
  opaque_sha256?}` (§4). An empty list is an explicit claim that the CAS blob
  reconstructs the wire bytes exactly (modulo nothing).
- `attempt_id` — the accounting join key.

### 3.2 Durable capture and explicit forward audit

The send seam holds one frozen request representation, verifies its in-memory
identity and persists one durable pre-send record. It never calls
`verify_sealed_candidate` or reads that record back. Persistence remains a
precondition of dispatch. Its redaction scans only parts the task has not met:
`redaction.part_fact` keeps each part's result by its bytes, in their key order,
and its parent's flags (`observability.task_parts`, shared with the logical request,
the Claudexor request and the working checkpoint), re-roots its records where the
part stands, and the stored bytes and exclusion rows equal the whole walk's.

The explicit audit compares the manifest's raw digest and size with the
attempt's recorded digest and size, checks serializer/exclusion metadata and
reads the CAS bytes to verify their digest. If an original candidate is supplied
to `verify_sealed_candidate`, it also compares the exclusion-aware projection
byte for byte. Without those original bytes, secret redaction and provider-native
custody are not reversible; the audit reports only what the retained evidence
can prove. It never fabricates a wire copy. Mismatches become typed durable
facts; they do not gate a later model call.

### 3.3 Reverse direction (audit, `model_send` only)

Run `python -m ouroboros.startup_historical_audit --data-root <root>` explicitly,
or `python -m ouroboros.startup_migrations --data-root <root>` to rebuild current
obligations and then audit. Neither job stamps an audit watermark. Both join directions stay:

- every `model_send` seal ⟶ exactly one attempt row in the usage-accounting
  replay (any terminal state, including refused-before-dispatch);
- every dispatched attempt row ⟶ exactly one seal.

An orphan on either side is a typed durable fact (§3.4), never a repair: the
sweep does not delete seals and does not fabricate attempts.

### 3.4 Mismatch = typed durable fact

One record shape for all three failure surfaces, appended to `events.jsonl`
AND written beside the seal (so it survives log rotation with its subject):

```
{type: "model_send_invariant_violation",
 kind: content_divergence | reconstruction_divergence | orphan_seal | unlogged_attempt,
 attempt_id, task_id,
 expected: {basis, sha256, size}, observed: {basis, sha256, size},
 divergence_class: §5-class, first_divergent_offset?: int}
```

No secret bytes ever enter the fact — digests and offsets only.

## 4. Typed exclusions (closed set)

Anything not in this enum is IN the byte domain. Each exclusion instance is
disclosed in the seal; an undisclosed transformation is by definition a
violation.

| class | what | why excluded | disclosure |
|---|---|---|---|
| `secret_redaction` | Secret VALUES masked in the CAS blob by the observability redaction rules; a tool schema's parameter names and shapes are not values (under a credential-named parameter all but its notes are) | The durable copy must not carry live credentials; equality is digest-anchored instead (pre-redaction sha256) | existing `RedactionRecord`s → `{class, path}` rows |
| `provider_native_custody` | Provider-owned opaque content (e.g. encrypted reasoning replay items) projected before persistence | Bytes are provider property; replay semantics are server-side | existing `anthropic_native_custody_projected` flag → per-item `{class, path, opaque_sha256}` |
| `transport_envelope` | HTTP headers, auth, SDK-added transport fields (user-agent, idempotency keys, `stream` flag where the SDK owns it) | Below the seam by construction; carries secrets and transport identity, not model-visible content | class-level row (no per-call enumeration) |
| `provider_side_transform` | Server-side effects the host cannot observe pre-flight: prompt-cache application, provider truncation/normalization | Not host-controlled; the seam is the LAST host-controlled point, not the last point | class-level row |

Delegated/harness model calls (`agent_session` executor lanes) are a
lane-level instance of `provider_side_transform`: the host never holds the
final wire bytes, so those lanes carry a disclosed
`model_send_seal: unobserved` limit (same honesty pattern as the scope
session's `read_provenance_expected` fact) rather than a fake seal.

## 5. Divergence classes and their canonicalization contract

### 5.1 Redaction (non-invertible)

Reconstruction can never reproduce secret bytes. Contract: equality is split
into the two existing labelled domains — byte equality on the redacted
projection, digest equality on the pre-redaction canonical bytes — and the
seal records which redaction rules fired where. A mismatch that disappears
when the redaction map is applied is `redaction_divergence` (a redaction-rule
change mid-flight, e.g. a hot-reloaded pattern), distinct from
`content_divergence`.

### 5.2 Streaming (response side)

Request bytes are fixed before transport, so streaming does not touch the
seal itself. The stream-assembly hazard is upstream: the assistant turn
assembled from chunks (including `finish_reason=null` partials) is replayed
into the NEXT round's messages. Contract: **single-assembly rule** — the
object persisted as the round's response record and the object appended to the
transcript must be the same object (one assembly, two consumers). Then the
next round's `model_send` byte compare transitively pins the assembly, and no
separate response-canonicalization machinery is needed. An implementation
that assembles twice (once for the log, once for the transcript) cannot
satisfy the invariant and is rejected at design level.

### 5.3 Retries and the recovery ladder

The ladder legitimately mutates payloads between attempts (dropped params,
stripped replayed reasoning, rerouted endpoint). Contract: the invariant is
**per physical attempt** — each rung's product is a NEW candidate with its own
seal and its own attempt id; identity across rungs is a logical-call concern
the invariant deliberately does not claim. Corollary: every ladder transform
must run ABOVE the seam; a transport that mutates the payload after the seal
is a `content_divergence` by construction, which is exactly the bug class the
byte compare exists to catch.

### 5.4 Serialization nondeterminism

Dict order, unicode escapes, float text, `default=str` coercions. Contract:
`canonical_json_v1` is the ONLY equality basis; both sides of every compare
are produced by the same versioned serializer, and the basis string travels in
the seal. A serializer change requires a new basis value and dual-reading is
per-record (`basis` names the rules), never heuristic.

### 5.5 In-place payload mutation below the host (SDK)

An SDK that mutates the dict it was handed (adding `stream: true`, coercing
types) after the seal was computed. This is the historical reason the
pre-dispatch identity re-check exists; explicit audit (§3.2) checks its retained record
from "our two copies agree" to "the durable record agrees with the wire".

## 6. Non-goals

- No byte claim for response objects, UI rendering, events/chat/progress
  narration planes (they remain projections; reverse-⟺ is `model_send` only).
- No logical-call identity across retry rungs (§5.3).
- No global "every log line reconstructs" framework — one seam, one record
  kind, one sweep.
- No new persistence plane: the seal extends the existing physical-candidate
  manifest; facts ride `events.jsonl` + the seal's own directory.

## 7. Implementation owners

- `llm_attempt.py` owns the pre-dispatch seam and its candidate identity gate;
  `realtime_turn.py` seals its funded wire plan through the same owner.
- `model_send_seal.py` stamps the physical-candidate manifest, reads back the
  durable projection, writes typed mismatch facts and reconciles both join
  directions. The seal is an additive key under the existing manifest schema.
- `startup_historical_audit.py` runs the bounded reconciliation as one
  owner-started job or the rebuild job's audit step, never an automatic boot child. Unknown accounting evidence does not become an orphan claim;
  the sweep records facts without deleting records or fabricating attempts.
- `tests/test_model_send_seal.py` covers reconstruction, typed divergence,
  non-blocking dispatch and reverse joins. History folded before the usage store
  existed is resolved through the retained evidence described in
  [Usage store](USAGE_STORE.md#8-explicit-history-audit).

## 8. Round receipt

Each round's `llm_round` carries what the host did to prepare its request
(`llm_observability.round_window` around the model call; `round_starts` marks
the batch's end). When the round's own call fails, its recovery (a configured
fallback, an owner-wait resend) runs in a window of its own: the answer's row
carries the recovery's facts from its start, without `round_top`, its
`model_call` step including the choice and fit of the route. Facts for the log
only; nothing decides by them.

- `work` (`context_timing.counted`): `secret_scan`, bytes of request-view parts
  the redaction worked out (once per part per task, §3.2); `new_parts`, bytes
  of shared parts the CAS stored for the first time (`observability.write_parts`).
  R1 (`tests/test_cost_round.py`): round 2 scans exactly the bytes it adds, at
  any start size. A scan is keyed by the part's own key order and the CAS by its
  sorted bytes, so a part met in two orders is scanned twice and stored once.
- `host_steps.steps`: `[wall ms, thread CPU ms]` between marks: `round_top`
  (the batch's end to the model call), `model_call` (to the first record),
  `request_record`, `physical_candidate`, `daemon_upload`, `create_operation`
  (CPU null: it ends on the model sender's thread).
- `host_steps.sections_ms`: wall ms of each record's `redact`, `blobs`,
  `lock_wait` and `manifest_write`; `context_facts`;
  `daemon.gateway|operations|upload|create`; `event_put`; `total` (the window).
- `host_steps.rusage`: the process's voluntary and involuntary context
  switches, major faults and block writes since the batch's end; absent on
  Windows.

About a kilobyte a round lands in the events archive kept for good (~14 MB a
day at the 14,000 rounds a busy install made on 09.11).

## 9. What a request did to its route's cached start

A provider keeps one cache per model and reads it in the wire order tools, system
blocks, messages. A cache mark closes a reuse scope: the shared blocks (tool schemas,
governance A′, the handbook D, resident memory B) are reused across captures and so
are bytes of their sources alone; the changing context C and the transcript are
reused within one captured view (DEVELOPMENT "Cache-friendliness invariant"). Before
a Main candidate is dispatched, the host compares it with
the last ANSWERED request of the same route in this task and writes the result on
the attempt's ledger row (`transcript_prefix.attempt_prefix_fact` →
`PhysicalAttemptContext.prefix` → `attempts.extra.physical_context.prefix`). It is an
observation of the authored prefix by `message_digest`'s identity (cache markers,
reasoning, outer whitespace and image detail are not in it), never the provider's
verdict: the row's measured `cached_tokens` and `cache_write_tokens` are that fact,
and an `append` beside a large write is not a contradiction this record resolves.

- `departs`: `first` (no answered request on the slot yet), `route` (answered
  requests, none on this route), `tools`, `system` (with `index`, `of` and `block`,
  the block's name as the round facts name it), `messages` (with `index`, `of`),
  `append` (the candidate only extends that request) or `unrecorded` (a slot
  restored from before the record, or a route whose full record was let go: the
  slot keeps four, in memory and in a checkpoint alike, and an older route keeps
  its name alone).
- `origins`: every seam that rewrote the view since the last answered request, in
  order (compaction, a restored continuation, a route or tool change, …).
- `from_route`: the route of the last answered request when it was another one.
- `base_at`: when this route's last answered request was answered (epoch seconds).
  Its start was earlier, so the time from `base_at` to an attempt's own dispatch is at
  most the time between the two starts (a cache lifetime runs from a request's start).

A candidate becomes its route's base when an answer of that route that read its
input arrives (`note_prefix_answer`, where the response arrives: input tokens
reported or output generated, no provider error; a body error read nothing): the accepted answer (it must
start with the candidate that waited for it; a send adds its own later rows, as its
clock line), or, at the next measurement, an answer the loop did not accept (an empty
or output-limited reply was still written and charged); the waiting candidate and the
answer ride a continuation checkpoint. A candidate never answered moves no base; an
answer of another route, or a route sent unmeasured, ends its wait. A retry inside one
call after its own answer extends that answered request (`refresh_attempt_prefix`);
a retry after a failed send keeps its call's fact. Seams named after the measurement
stay for the next request.
Sends outside Main's loop (helpers, reviews) carry no fact; the forced final takes its
fact from the input it measured. The routes' records ride the continuation
checkpoint with the observer's other state (`OBSERVER_STATE_ATTRS`).

One reader names the class (`transcript_prefix.prefix_class`), in the provider's
own order: `first_request`, `entered_model`, `tools_changed` (a reworded tool
description or parameter rewrites every running task's start once), `system_changed`
(which block), `history_rewritten` (origins); then, for pure extensions, `paused`
(from `base_at` to the row's own `ts_dispatched` is longer than the lifetime the
row's `prompt_cache_ttl` names, so the previous start was longer ago still:
arithmetic over recorded facts, said as time, never as "expired", and never derived
for `default`, an absent lifetime or an undispatched row), `returned_to_model`,
`append`; a row without a fact is `not_recorded`. Nothing decides by the fact or the
class.

The Costs page's period view (`gateway/cost_period.py`,
`GET /api/cost-breakdown?period=24h|7d|30d`) reads the calls reserved in a period
(membership by the reservation instant; the slices are found by position,
`usage_store.Txn.period_bounds`, in point reads; each slice is one short read, and the
tariffs are looked up outside it) and shares
each call's RECORDED cost among cache writes, cache reads, output and uncached
input by what the route's tariff charges for that call's own tokens. Only price
ratios enter; the parts and the cost that could not be shared (no tariff, a token
count the row does not have, a write whose lifetime neither the row nor the
provider's split names) add up to the recorded cost. Input is classed only for calls
that reported both cache counts; the others are counted apart (`uncounted_calls`,
`uncounted_input_tokens`), never as uncached. A money figure and the tokens beside it are of
the same calls: `tokens_shared` are the tokens of the calls whose cost was shared (`tokens`
counts every call), the cost no tariff or count could share is its own part with
`unshared_calls`, and a cause row's `write_cost` is of its `shared_write_tokens`. A model
none of whose calls has a recorded price has `cost: null`. `write_price_in_reads` is how many read tokens one
written token costs at `write_price_lifetime` (the lifetime most of the model's
tokens were written at, by the provider's split when it accounts for every write)
at the model's base price.
Delegated subscription sessions and unmetered dispatches carry no reservation time
and are not in a period.
The tariff is read without a fetch: what the process holds, else the install's
dated catalogue record (`state/openrouter_models.json`), whose date the view
names. Cache writes are listed by `prefix_class`. Nothing is stored and no
verdict, threshold or signal is derived.
