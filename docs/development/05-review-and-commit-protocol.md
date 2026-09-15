# Review & Commit Protocol

This chapter owns the three stages of a reviewed commit — prepared preflight, the authoritative gate, and publication binding — together with the shared paid-cycle cap, the free-replay rules, the external-review evidence contract and the release-sync rule a pull request must obey. It exists because technical failure and commit permission are separate facts, and every rule here keeps a missing review from becoming a PASS.

Keep optional task evidence outside the stable governance prefix; shrink its excerpt before reducing existing review material. A source pointer gives a packet-only model no retrieval capability. Rejoin preserves the original hash and project-local view while any physical reviewer may still read it. Removing an ignored view never deletes the canonical source; no separate notes corpus, blanket ToolResult metadata or mandatory whole-history read belongs to this evidence.

Reviewed commits separate improvement evidence from candidate-bound authority.
Finish the edits and focused tests, then call `commit_reviewed`; standalone
`preflight_review` remains available when an earlier critique is useful.
`docs/CHECKLISTS.md` owns reviewer questions, severity and output contracts;
ARCHITECTURE "Review delivery" owns the dataflow.

1. **Prepared preflight.** Authorization and unresolved-work checks precede
   mechanical file preparation, staging/classification/protection and the
   fingerprint. Existing free-cycle/budget admission precedes any automatic
   preflight. When needed, the same `preflight_review` runs inline with the full
   `review_rebuttal` and the independently applicable test preflight. The
   candidate must remain unchanged before triad/scope dispatch. Explicit
   `skip_advisory_review=True`, disabled and unconfigured paths retain their
   audited behavior. A free advisory replay reads freshness but buys neither
   another preflight nor another triad/scope wave. Stale coverage still needs
   the explicit audited skip; applicable compensating tests run even when the
   reviewer backend is available. Explicit test skips are not green proof.
2. **Authoritative gate.** Independently configured deterministic test policy,
   staged fingerprinting, triad review, applicable scope review, aggregation,
   and pre/post revalidation. The exact binding: `docs/architecture/06-agent-core.md` § "Git and commit
   review".
3. **Publication binding.** The created commit/tag is checked against that same binding before push. Any
   mutation, rebase, conflict resolution, or changed landing parent
   invalidates exact-candidate authority and requires the applicable final
   gate again.

A technical review failure may permit continuing under owner-selected advisory
enforcement on a known, independently bound candidate. The failure's phase,
reason, received findings and full result stay recorded as failure, never PASS.
Outside Cyber Pro, Blocking enforcement still blocks. Cyber may continue
without prior review, preserving the original findings, missing sources and
pending invocation. It never fabricates candidate bytes, completed custody or
physical effects; Stop, deadline and cost facts remain independently recorded.
Diagnostic
`repo_commit_ready` projects this permission only from an exact repo/hash match;
it does not change the failed review's status or freshness.

Pending triad/scope reconciliation retains the prepared index and never
restages or reconstructs a lost index; pending delegated preflight rejoins its
exact durable invocation rather than posting a replacement, and an explicit
audited preflight skip releases only logical admission — it neither cancels
physical work nor erases its cost or custody (the custody mechanics, late
results and the definite start-failure discharge:
`docs/architecture/06-agent-core.md` § "Review delivery"). Both commit and
review-only entry points forward the explicit skip; a subsequent standalone
request can rejoin exact historical custody or check new evidence, and released
unrelated history is not a logical lock. The external review wrapper uses the
same cycle and retains its candidate checkout/index while custody remains
unresolved.

Triad slots review the staged diff against `docs/CHECKLISTS.md`; duplicate
model ids remain independent slots and `config.adaptive_quorum` owns quorum. A managed-update resolution commit reviews the declared M0→S resolution delta
(the managed exception: `docs/architecture/06-agent-core.md` § "Git and commit
review"). Scope slots inspect touched context plus the repository Atlas through the
guaranteed-fit ladder (`docs/architecture/06-agent-core.md` § "Review stack");
an artifact owed in full cannot buy fit by degrading into an invalid review. Owner-selected Low records the distinct
BIBLE P3 scope skip; other route or assembly failure is not a clean verdict.
An agent-session scope slot delivers by retrieval: its verdict is
authoritative once its window is sourced at ≥200K, and "the host did not
observe which files it read" is a provenance disclosure, never a
missing-authority finding. The gate is one logical reviewer interaction per
API slot, with at most one bounded second physical send on a same-route
transport rail for a PACKET api row; a hosted agent-session slot is one
multistep execution whose local extraction reuses its collected transcript.
A native tool-round slot (an api row bound to a configured subagent) is
likewise one multistep episode with no send count; its bounds and typed ends:
`docs/architecture/06-agent-core.md` § "Review delivery". A retrieving delivery canonicalizes
its answer by the surface's output SHAPE (`triad_review.review_output_shape`:
`array` | `object` | `report`), never by surface-name branches inside the
canonicalizer: the shape table is form only, and a new object- or
report-shaped surface registers there instead of teaching the extraction rail
another `if`.

Advisory row parsing and hosted-review identity evidence are mechanism
(`docs/architecture/06-agent-core.md` § "Review delivery"); what a change must
preserve: the full raw result, ordinary PASS rows and genuine empty-clean
responses in tests, and the exact contributor checker's refusal of unconfirmed
model identity, including a display label that cannot prove the pin.

Paid review cycles across the gates are bounded by one shared owner knob,
`OUROBOROS_REVIEW_MAX_CYCLES` — a STRING, positive integer or `unlimited`,
default `"2"` (Settings → Behavior → "Max Review Cycles"). Its SSOT is `ouroboros/review_cycles.py`; the four per-gate meanings are stated
once in `docs/architecture/06-agent-core.md` § "Review stack" (the retired
legacy key is migrated at settings load). `unlimited` removes only the local count —
deadline, budget, and lifecycle rails still bind — and a malformed value fails
closed to the default, logged once.

For task acceptance, the exact-binding tree-wallet claim is a strict
write-ahead stamp bound to every delivery the panel's rows run — one idempotent
claim per panel (owner R11, 2026-09-01: the paid identity is material, not
route). The per-delivery stamp points, the once-per-panel launch floor (owner
R55), the R23 clamps on a running panel and the disclosed deadline-cut residual
are stated once in `docs/architecture/06-agent-core.md` § "Task lifecycle" and
the `review_dispatch.py` row of
`docs/architecture/01-high-level-architecture.md`. Panel assembly, an
unavailable route, or another pre-transport refusal consumes no claim and
leaves the binding retryable; an unavailable claim releases the usage
reservation and blocks every parallel panel slot before reviewer transport
rather than degrading hard authority into fail-open cost telemetry. The
compatibility positive-capture residual (issue #588) is disclosed at its owner,
the `review_execution.py` row of the same map.

Never pay for byte-identical review material (`ouroboros/tools/commit_gate.py` owns
the mechanism): the commit gate refuses a byte-identical staged diff for free
from the FIRST verdict-block (`identical_diff_refused`, quoting the recorded
verdict), and skill review replays a recorded substantive verdict for an
identical snapshot at $0 while the persisted state still covers it. A rebuttal
is identified by CONTENT sha256 — a hash new to the streak buys exactly ONE
paid re-review; a repeated hash is refused free. The two axes stay distinct:
refusal-streak eligibility is about VERDICTS (a rebuttal is spent only by the
substantive verdict it bought), while money is about DISPATCH (every
physically dispatched wave counts whatever its terminal; infra facts refused
at assembly never dispatched and stay outside the count; the paid fact is
recorded write-ahead). A refusal that spent nothing is a typed `not_dispatched` fact, never a verdict
(the one shape of every $0 exit: `docs/architecture/06-agent-core.md` § "Review
stack").
Exhaustion is always the typed
`review_cycles_exhausted` event with honest exits — under advisory
enforcement a commit after exhaustion proceeds as a free replay with a loud
typed disclosure; blocking refuses it.

Scope of the review-contract fingerprint (deliberate): it covers the reviewer
roster, routes, enforcement, resolved efforts, and prompt constants —
including the session serialization only when Skill Review actually contains
an agent-session row — while governance-document CONTENTS — `BIBLE.md`,
`docs/CHECKLISTS.md`, `docs/ARCHITECTURE.md`, this handbook and
`docs/DESIGN.md` — are deliberately outside it, so editing those documents
neither lapses recorded verdicts nor frees replays. The accepted
trade-off is that an old verdict can replay under amended governance text;
this keeps routine documentation maintenance from repricing every recorded
review.

### External PR review is not commit authorization

The authoring agent freezes the final committed base-to-head range and gives
it to a separate agent context for read-only review; same-conversation
self-review does not count, and unavailable review is recorded `NOT_RUN`,
never silently presented as clean. `CONTRIBUTING.md` owns the public procedure
and evidence fields. `scripts/run_external_review.py --contributor` is
maintainer-grade large-window tooling: it freezes the configured triad/scope
rows, binds each row to its dispatched prompt receipt and observed response
receipt, and records exact base/head/tree/diff hashes, route/model/profile
facts, terminal settlement, capability deltas, and full redacted
agent-session transcripts; missing, tampered, drifted, unprovable, or
contradictory receipts make the packet `INCOMPLETE`. The lane always executes
the TARGET BASE's own review machinery — invoked from any other checkout it
re-runs itself from a detached worktree of the base commit — so a proposal is
never reviewed by its own copy of the review flow, whatever it touches. This
evidence establishes readiness; it does not authorize commit, push, merge, or
publication — maintainers choose the landing parent and release version,
preserve authorship, and run the normal final exact-candidate gate.

### Release sync

A pull request into `ouroboros` leaves every version carrier byte-identical to
its target (the carrier list and the one projection that writes them:
`docs/architecture/10-key-invariants.md`, invariant 2). At integration,
`ouroboros/tools/release_sync.py::sync_release_metadata()` projects the chosen
version and `version_carrier_desyncs()` verifies the file carriers (the history
row is pinned by the packaging-sync test); changelog prose remains a deliberate
maintainer edit. The installer filename templates, the immutable exact-tag
download links and the stable promotion of `main` are
`docs/architecture/08-git-branching-ci-and-build.md` § "Build scripts".

Hermetic preflight uses a disposable worktree, temporary
data/settings/pycache, and scrubbed runtime/secret-class environment. Tests
must rebind imported process-global roots and fail closed on the live data
root; setting only `OUROBOROS_DATA_DIR` is insufficient. A reviewed local
commit is the durability boundary; an `origin` push and CI are follow-up
signals, not prerequisites for local self-modification survival.

---

