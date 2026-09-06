# Design note — runtime invariant `model-visible ⟺ logged` (CPL-5)

Status: LANDED (plan §7 item 5; batch-1 Q8=A confirmed; narrowed per roast
finding F15). The code is `ouroboros/model_send_seal.py`, wired at
`llm_attempt._candidate_before_dispatch` and swept from `server_maintenance`;
the pins are `tests/test_model_send_seal.py`. This note remains the contract,
kept narrow so neither the code nor a later reader drifts into a broader —
unprovable — claim. ONE clause changed between design and landing, and it is
marked in §3.2: a reconstruction mismatch is an OBSERVABILITY fact, not a
dispatch gate.

## 1. The claim, narrowed (F15)

The invariant binds exactly one object: **`model_send` — the physical
candidate payload at the last host-controlled pre-transport seam**. That seam
already exists and is singular:
`ouroboros/llm_attempt.py::_candidate_before_dispatch` — the closure that runs
after the cache-marker finalizer produced the final send copy
(`_physical_candidate`) and before the transport's `send`.

- **Forward (`sent ⟹ logged`)**: every physical attempt persists a sealed
  durable record of its exact send copy BEFORE dispatch.
- **Reverse (`logged ⟹ sent`)**: every sealed `model_send` record joins
  exactly one accounting attempt (dispatched, refused, or released). The
  reverse direction holds for `model_send` records ONLY — F15 explicitly does
  not claim it for any other log plane (events, chat, progress are narrations,
  not send truth).
- **Everything else is out of the byte domain by typed exclusion, never by
  silence** (§4).

"Model-visible" means *what the provider-hosted model receives*, i.e. the
send payload (`system`/`messages`/`tools`/params). It does not mean UI
rendering, and it does not mean the response side: what the model "sees" of
its own previous answers is whatever the host replays into the NEXT send, so
response-assembly truth is covered transitively by the next round's
`model_send` record (§5.2).

## 2. What already exists (reuse-first — the note extends, it does not mint)

| Existing mechanism | Where | Role in the invariant |
|---|---|---|
| Canonical digest of the exact send copy (`canonical_json_v1`: sort_keys, compact separators, `ensure_ascii=False`, `allow_nan=False`, `default=str`) | `llm_attempt._attempt_request` / `_canonical_candidate_bytes` | The canonical form and its versioned basis (`candidate_measurement_kind`) |
| Pre-dispatch identity re-check: digests re-derived from the closed-over candidate and compared with the reservation's expected identity; drift refuses dispatch (`PhysicalAttemptPreparationFailed: physical candidate changed before dispatch`) | `llm_attempt._candidate_before_dispatch` | The forward gate's skeleton — today a digest compare of two in-memory copies |
| Durable candidate manifest + redacted CAS blob, written before dispatch, with two labelled digest domains (`canonical_json_v1_pre_redaction` facts vs `observability_json_v1_post_default_redaction_cas` blob) | `observability.persist_physical_candidate` / `persist_call` | The sealed record carrier |
| Attempt lifecycle `reserved → dispatched → settled|unresolved` / `reserved → released`, short-lock append + sequence replay | `usage_accounting` | The join target for the reverse direction |
| Anthropic native custody projection (opaque provider-native content replaced before persistence; disclosed as `anthropic_native_custody_projected`) | `anthropic_native_custody.physical_custody_projection` | Prototype of a typed exclusion |
| Secret redaction with per-hit `RedactionRecord`s | `observability._redact_text` + rules | Prototype of a typed exclusion |

The gap this note was written to close: the pre-existing gate compared two
**in-memory** serializations, so a bug between "what we persisted" and "what we
believe we persisted" was assumed away rather than caught, and a mismatch was
only a raised exception. `model_send_seal.verify_sealed_candidate` closes it by
**reconstructing from the durable record** and byte-comparing that
reconstruction against the wire-bound serialization, emitting a **typed durable
fact** on any inequality (§3.2 — a fact, not a refusal).

## 3. The contract

### 3.1 Sealed record (`model_send` seal, v1)

Extend the existing physical-candidate manifest (no new plane) with a
`model_send_seal` block:

- `seal_version: 1`
- `canonical_basis: "canonical_json_v1"` — serializer identity. Any change to
  the serializer is a NEW basis string; a reader never re-interprets bytes
  under a different basis.
- `pre_redaction_sha256` / `size_bytes` — digest of the canonical bytes of the
  exact wire payload (exists today as `candidate_raw_sha256`).
- `exclusions: [...]` — every applied exclusion instance: `{class, path,
  opaque_sha256?}` (§4). An empty list is an explicit claim that the CAS blob
  reconstructs the wire bytes exactly (modulo nothing).
- `attempt_id` — the accounting join key (exists).

### 3.2 Verification on call (forward)

At the seam, in this order:

1. Serialize the wire-bound candidate to canonical bytes `W`.
2. Persist the sealed record (already the order today: persist, then gate).
3. **Reconstruct** `R` from the durable record just written: read back the
   blob, undo nothing — instead apply the SAME exclusion map to `W` (redaction
   and custody projection are not invertible; §5.1) — and compare byte-for-byte
   the comparable domain: `project(W, exclusions) == blob_bytes` AND
   `sha256(W) == pre_redaction_sha256`.
4. Any inequality → write the typed durable mismatch fact (§3.4). The call is
   NOT blocked, and the verification never raises: this invariant is
   observability, and `verify_sealed_candidate` is fail-soft by contract.

That last step is the one place the landed contract differs from the first
draft of this note, which asked for a fail-closed refusal through
`PhysicalAttemptPreparationFailed`. It was rejected on its own merits, not for
convenience:

- The refusal it would add is not the same question as the existing gate. The
  in-memory identity re-check above this call still refuses dispatch when the
  candidate itself changed between reservation and send — that is a candidate
  fact and it stays fail-closed, unchanged. A reconstruction mismatch is a fact
  about the RECORD (a corrupt blob, a tampered seal digest, a missing seal
  block, an undisclosed exclusion class, a foreign serializer basis) — a
  logging defect. Blocking a paid, otherwise-correct model call because the
  audit copy on disk is unreadable trades the product's function for the
  audit's tidiness, and it would let a full disk or a rotated file stop
  cognition.
- Fail-closed here would also be self-defeating: the durable fact IS the
  disclosure, and a refusal path that can itself fail (write error, unreadable
  root) would have to decide between a silent skip and a dead runtime.

So the landed rule is: the fact is mandatory, the block is not.
`tests/test_model_send_seal.py` pins exactly this — a corrupted blob, a
tampered seal digest, a dropped seal block, an undisclosed exclusion class and
a foreign basis each produce their typed fact while the attempt still settles.

The added cost is one read-back and one projection per physical attempt —
bounded, local, and on the same drive the record was just written to.

### 3.3 Reverse direction (audit, `model_send` only)

A bounded reconciliation sweep (rides the existing startup-sweep family in
`server_maintenance`, not a new scheduler):

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
| `secret_redaction` | Secret VALUES masked in the CAS blob by the observability redaction rules | The durable copy must not carry live credentials; equality is digest-anchored instead (pre-redaction sha256) | existing `RedactionRecord`s → `{class, path}` rows |
| `provider_native_custody` | Provider-owned opaque content (e.g. encrypted reasoning replay items) projected before persistence | Bytes are provider property; replay semantics are server-side | existing `anthropic_native_custody_projected` flag → per-item `{class, path, opaque_sha256}` |
| `transport_envelope` | HTTP headers, auth, SDK-added transport fields (user-agent, idempotency keys, `stream` flag where the SDK owns it) | Below the seam by construction; carries secrets and transport identity, not model-visible content | class-level row (no per-call enumeration) |
| `provider_side_transform` | Server-side effects the host cannot observe pre-flight: prompt-cache application, provider truncation/normalization | Not host-controlled; the seam is the LAST host-controlled point, not the last point | class-level row; conformance suite (CPL-6) owns per-provider characterization |

Delegated/harness model calls (`agent_session` executor lanes) are a
lane-level instance of `provider_side_transform`: the host never holds the
final wire bytes, so those lanes carry a disclosed
`model_send_seal: unobserved` limit (same honesty pattern as the scope
session's `host_file_read_attestation: unobserved`) rather than a fake seal.

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
pre-dispatch re-check exists; reconstruction-based compare (§3.2) extends it
from "our two copies agree" to "the durable record agrees with the wire".

## 6. Non-goals

- No byte claim for response objects, UI rendering, events/chat/progress
  narration planes (they remain projections; reverse-⟺ is `model_send` only).
- No logical-call identity across retry rungs (§5.3).
- No global "every log line reconstructs" framework — one seam, one record
  kind, one sweep (plan: local decisions, no generic framework).
- No new persistence plane: the seal extends the existing physical-candidate
  manifest; facts ride `events.jsonl` + the seal's own directory.

## 7. Implementation sketch for the next lane (not this one)

1. `llm_attempt.py`: extend `_candidate_before_dispatch` with read-back +
   projection compare; thread the typed fact writer (small; the seam is one
   closure).
2. `observability.py`: `model_send_seal` block in
   `persist_physical_candidate` manifests (schema_version bump of the
   manifest payload is NOT needed — additive key under the existing
   `SCHEMA_VERSION` object; readers ignore unknown keys).
3. `server_maintenance.py`: reconciliation sweep behind the existing startup
   sweep guardrails (fail-soft, bounded batch, UNKNOWN accounting state skips
   destructive conclusions — there are none to skip: the sweep only writes
   facts).
4. Tests: seal round-trip (write → reconstruct → equal); each §5 class forced
   (mutating fake SDK, redaction-rule flip, double-assembly guard, per-rung
   seals); reverse sweep on a synthetic orphan both ways; delegated-lane
   `unobserved` disclosure.
